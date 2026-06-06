//! GPU compute backend for `GPU { ... }` blocks.
//!
//! This is the first milestone: elementwise arithmetic on tensors and scalars,
//! e.g. `GPU { T1 + T2 }`. The design (see `docs/CONSIDERATIONS.md`) is to keep
//! all intermediate values GPU-resident and download only the final result.
//!
//! v1 uses an eager model: each AST node is evaluated to a `GpuVal` (a scalar, or
//! a GPU-resident buffer) immediately, recording and submitting one compute pass
//! per elementwise op. Captured CPU tensors are uploaded on first reference. The
//! record→fuse→dispatch and loop-residency machinery described in the design doc
//! are later milestones.

mod context;

use crate::ast::{BlockStmt, Def, Expr, Op};
use crate::eval::{BC, Env, FieldVal, TData, Val, Variance, fmt_val};
use std::sync::Arc;
use context::context;
use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

use context::GpuContext;

/// A value living in the GPU evaluation scope: either a CPU-side scalar (passed
/// into shaders as a literal), a GPU-resident f32 buffer, or a result that has
/// already been downloaded to host memory (`scan`'s assembled spacetime block —
/// kept on the host to avoid a pointless re-upload/re-download round-trip).
#[derive(Clone)]
enum GpuVal {
    Scalar(f64),
    Buffer {
        buf:   Rc<wgpu::Buffer>,
        shape: Vec<usize>,
        len:   usize,
    },
    Host {
        data:  Rc<Vec<f32>>,
        shape: Vec<usize>,
        len:   usize,
    },
    /// A GPU-resident complex tensor: interleaved (re, im) pairs as `vec2<f32>`,
    /// `buf` holding `2*len` f32. Produced by `fft`/`ifft`, complex captures, and
    /// complex arithmetic; downloaded as a `ComplexTensor`.
    Complex {
        buf:   Rc<wgpu::Buffer>,
        shape: Vec<usize>,
        len:   usize,
    },
    /// A complex scalar (e.g. the imaginary unit `i`, or `2 + 3*i`). Carried on the
    /// host like `Scalar` and baked into kernels as a `vec2<f32>` constant.
    CScalar(f64, f64),
    /// A gridded field (0-/k-form or vector field) — its component data lives in a
    /// GPU buffer (component-fastest: `data[p*ncomp + c]`) and its geometry/degree
    /// live host-side, exactly as in the CPU `FieldVal`. This is what makes
    /// `forms.*` and the field-polymorphic `ops.*` run inside a GPU block.
    Field {
        buf:      Rc<wgpu::Buffer>,
        grid:     Vec<usize>,
        spacing:  Vec<f64>,
        lo:       Vec<f64>,
        bc:       Vec<BC>,
        metric:   Vec<f64>,
        degree:   usize,
        variance: Variance,
        len:      usize,        // gt * ncomp
    },
    /// A tuple of values (e.g. coupled fields `(U, V)` carried as a single
    /// `iterate`/`scan` state). Each element is itself a `GpuVal`.
    Tuple(Vec<GpuVal>),
    /// A user lambda bound to a name inside a block (`f = x -> x*x`). Carries its
    /// parameter names and body; free variables resolve through the shared scope
    /// at apply time (no captured environment), exactly like an iterate step body.
    /// Applied by AST inlining (beta reduction) — WGSL has no first-class fns, so
    /// a `Fn` can be *called* but never returned or used as data.
    Fn(Rc<(Vec<String>, Expr)>),
}

/// Ensure a value is GPU-resident, uploading a host-side result if needed. A
/// `Host` value only stays on the host when it flows straight to the final
/// download; if it is consumed by another GPU op we materialize it here.
fn materialize(ctx: &GpuContext, v: GpuVal) -> GpuVal {
    match v {
        GpuVal::Host { data, shape, len } => {
            let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu-host-upload"),
                contents: bytemuck::cast_slice(if data.is_empty() { &[0.0f32][..] } else { &data[..] }),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });
            GpuVal::Buffer { buf: Rc::new(buf), shape, len }
        }
        GpuVal::Tuple(elems) => {
            GpuVal::Tuple(elems.into_iter().map(|e| materialize(ctx, e)).collect())
        }
        other => other,
    }
}

/// Lift a CPU `Val` into a GPU value: scalars stay scalar, real tensors are
/// uploaded, and tuples recurse. This is how a tuple of tensors (or the result
/// of a host helper like `get(cell)`) enters a GPU block.
fn lift_val(ctx: &GpuContext, v: &Val) -> Result<GpuVal, String> {
    match v {
        Val::Num(f) => Ok(GpuVal::Scalar(*f)),
        Val::Complex(r, i) => Ok(GpuVal::CScalar(*r, *i)),
        Val::Tensor { data, shape } => Ok(upload(ctx, data, shape)),
        Val::ComplexTensor { re, im, shape } => Ok(upload_complex(ctx, re, im, shape)),
        Val::Field(f) => Ok(upload_field(ctx, f)),
        Val::Tuple(items) => {
            let elems = items.iter().map(|it| lift_val(ctx, it)).collect::<Result<Vec<_>, _>>()?;
            Ok(GpuVal::Tuple(elems))
        }
        other => Err(format!(
            "GPU: only scalars, real tensors, and tuples of them can enter a GPU block; got {}",
            fmt_val(other)
        )),
    }
}

/// Pre-pass over a GPU block body: resolve `get(cell)` — the one permitted host
/// helper — on this initial walk, before any kernel runs. Each `get(...)` is
/// evaluated once on the CPU, lifted into the block scope under a synthetic name,
/// and its call site rewritten to reference that capture. This guarantees host
/// state crosses the boundary up front (never mid-kernel), and that *only* `get`
/// is dispatched to the host (any other unknown call is rejected during eval).
fn hoist_gets(
    e: &Expr,
    env: &Env,
    ctx: &GpuContext,
    scope: &mut HashMap<String, GpuVal>,
    counter: &mut usize,
) -> Result<Expr, String> {
    let rec = |x: &Expr, scope: &mut HashMap<String, GpuVal>, counter: &mut usize|
        hoist_gets(x, env, ctx, scope, counter);
    Ok(match e {
        Expr::Apply(f, args) => {
            if let Expr::Var(name) = &**f {
                if name == "get" {
                    // Host read of the cell, resolved now and captured.
                    let v = crate::eval::eval(e, env)
                        .map_err(|err| format!("GPU: get(cell): {err}"))?;
                    let gv = lift_val(ctx, &v)?;
                    let key = format!("__get{}", *counter);
                    *counter += 1;
                    scope.insert(key.clone(), gv);
                    return Ok(Expr::Var(key));
                }
            }
            let nf = rec(f, scope, counter)?;
            let nargs = args.iter().map(|a| rec(a, scope, counter)).collect::<Result<Vec<_>, _>>()?;
            Expr::Apply(Box::new(nf), nargs)
        }
        Expr::BinOp(l, op, r) =>
            Expr::BinOp(Box::new(rec(l, scope, counter)?), op.clone(), Box::new(rec(r, scope, counter)?)),
        Expr::Neg(x) => Expr::Neg(Box::new(rec(x, scope, counter)?)),
        Expr::Not(x) => Expr::Not(Box::new(rec(x, scope, counter)?)),
        Expr::Lambda(ps, ret, body) =>
            Expr::Lambda(ps.clone(), ret.clone(), Box::new(rec(body, scope, counter)?)),
        Expr::Tuple(xs) =>
            Expr::Tuple(xs.iter().map(|x| rec(x, scope, counter)).collect::<Result<Vec<_>, _>>()?),
        Expr::Array(xs) =>
            Expr::Array(xs.iter().map(|x| rec(x, scope, counter)).collect::<Result<Vec<_>, _>>()?),
        Expr::TensorLit(rows) => Expr::TensorLit(
            rows.iter().map(|row| row.iter().map(|x| rec(x, scope, counter)).collect::<Result<Vec<_>, _>>())
                .collect::<Result<Vec<_>, _>>()?),
        Expr::Index(b, i) =>
            Expr::Index(Box::new(rec(b, scope, counter)?), Box::new(rec(i, scope, counter)?)),
        Expr::Member(b, m) => Expr::Member(Box::new(rec(b, scope, counter)?), m.clone()),
        Expr::Range(l, r) =>
            Expr::Range(Box::new(rec(l, scope, counter)?), Box::new(rec(r, scope, counter)?)),
        Expr::Slice(a, b) => Expr::Slice(
            a.as_ref().map(|x| rec(x, scope, counter)).transpose()?.map(Box::new),
            b.as_ref().map(|x| rec(x, scope, counter)).transpose()?.map(Box::new)),
        Expr::GpuBlock(b) => Expr::GpuBlock(Box::new(rec(b, scope, counter)?)),
        Expr::Block(stmts) => {
            let nstmts = stmts.iter().map(|s| Ok(match s {
                BlockStmt::Expr(x) => BlockStmt::Expr(rec(x, scope, counter)?),
                BlockStmt::Def(Def::Var(n, x)) => BlockStmt::Def(Def::Var(n.clone(), rec(x, scope, counter)?)),
                BlockStmt::Def(Def::Func(n, ps, ret, x)) =>
                    BlockStmt::Def(Def::Func(n.clone(), ps.clone(), ret.clone(), rec(x, scope, counter)?)),
            })).collect::<Result<Vec<_>, String>>()?;
            Expr::Block(nstmts)
        }
        leaf @ (Expr::Num(_) | Expr::ImagLit(_) | Expr::Var(_)) => leaf.clone(),
    })
}

/// Entry point, called from the `Expr::GpuBlock` arm in `src/eval.rs`.
pub fn run_gpu_block(body: &Expr, env: &Env) -> Result<Val, String> {
    let ctx_mutex = context()?;
    let ctx = ctx_mutex.lock().map_err(|_| "GPU context poisoned".to_string())?;
    let mut scope: HashMap<String, GpuVal> = HashMap::new();
    // Resolve host helpers (`get(cell)`) up front on this initial walk, before
    // any kernel runs — never mid-kernel. They become block-scope captures.
    let mut gets = 0usize;
    let body = hoist_gets(body, env, &ctx, &mut scope, &mut gets)?;
    let body = &body;
    // Outer scope catches allocation errors from upload/download (e.g. a tensor
    // larger than the device's max buffer size) instead of panicking.
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = eval_gpu(body, env, &ctx, &mut scope).and_then(|v| to_val(&ctx, &v));
    if let Some(err) = pollster::block_on(ctx.device.pop_error_scope()) {
        return Err(format!("GPU: {err}"));
    }
    result
}

/// Recursively evaluate an expression on the GPU.
fn eval_gpu(
    e: &Expr,
    env: &Env,
    ctx: &GpuContext,
    scope: &mut HashMap<String, GpuVal>,
) -> Result<GpuVal, String> {
    match e {
        Expr::Num(n) => Ok(GpuVal::Scalar(*n)),
        // Imaginary literal `2i` → complex scalar 0 + 2i (matches `2*i`).
        Expr::ImagLit(v) => Ok(GpuVal::CScalar(0.0, *v)),

        Expr::Neg(x) => {
            let v = eval_gpu(x, env, ctx, scope)?;
            neg_gpu(ctx, v)
        }

        Expr::Var(name) => {
            if let Some(v) = scope.get(name) {
                return Ok(v.clone());
            }
            // Capture from the enclosing CPU scope, uploading tensors on demand.
            // Uploaded tensors are memoized into `scope` so repeated references
            // (and loop bodies) don't re-upload.
            match env.vars.get(name) {
                Some(Val::Num(f)) => Ok(GpuVal::Scalar(*f)),
                Some(Val::Complex(r, i)) => Ok(GpuVal::CScalar(*r, *i)),
                Some(Val::Tensor { data, shape }) => {
                    let gv = upload(ctx, data, shape);
                    scope.insert(name.clone(), gv.clone());
                    Ok(gv)
                }
                Some(Val::ComplexTensor { re, im, shape }) => {
                    let gv = upload_complex(ctx, re, im, shape);
                    scope.insert(name.clone(), gv.clone());
                    Ok(gv)
                }
                Some(Val::Field(f)) => {
                    let gv = upload_field(ctx, f);
                    scope.insert(name.clone(), gv.clone());
                    Ok(gv)
                }
                // A tuple of tensors/scalars (e.g. coupled fields) — upload each.
                Some(v @ Val::Tuple(_)) => {
                    let gv = lift_val(ctx, v)?;
                    scope.insert(name.clone(), gv.clone());
                    Ok(gv)
                }
                Some(other) => Err(format!(
                    "GPU: `{name}`: only scalars, real/complex tensors, and tuples of them are allowed in a GPU block; got {}",
                    fmt_val(other)
                )),
                None => Err(format!("undefined variable `{name}`")),
            }
        }

        Expr::Block(stmts) => {
            // Block scope is layered on top of the captured scope. Local bindings
            // are visible only inside the block.
            let mut last = GpuVal::Scalar(0.0);
            let mut saw_expr = false;
            for stmt in stmts {
                match stmt {
                    BlockStmt::Def(Def::Var(name, expr)) => {
                        let v = eval_gpu(expr, env, ctx, scope)?;
                        scope.insert(name.clone(), v);
                    }
                    // `f(x) = …` block-local function → a callable `Fn` value,
                    // applied by inlining (no recursion; WGSL has no call stack).
                    BlockStmt::Def(Def::Func(name, params, _, body)) => {
                        let f = GpuVal::Fn(Rc::new((
                            params.iter().map(|p| p.name.clone()).collect(),
                            body.clone(),
                        )));
                        scope.insert(name.clone(), f);
                    }
                    BlockStmt::Expr(expr) => {
                        last = eval_gpu(expr, env, ctx, scope)?;
                        saw_expr = true;
                    }
                }
            }
            if !saw_expr {
                return Err("GPU: block has no result expression".into());
            }
            Ok(last)
        }

        Expr::BinOp(l, op, r) => {
            let lv = eval_gpu(l, env, ctx, scope)?;
            let rv = eval_gpu(r, env, ctx, scope)?;
            binop(ctx, op, lv, rv)
        }

        Expr::Apply(f, args) => eval_apply(f, args, env, ctx, scope),

        // A tuple value, e.g. the `(U', V')` returned by a coupled step.
        Expr::Tuple(xs) => {
            let vals = xs.iter()
                .map(|x| eval_gpu(x, env, ctx, scope))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GpuVal::Tuple(vals))
        }

        // Indexing a tuple value, e.g. `s[1]`. (Indexing a *tensor* on the GPU
        // is only supported as a gather inside `tensor(...)`.)
        Expr::Index(base, idx) => {
            let bv = eval_gpu(base, env, ctx, scope)?;
            match bv {
                GpuVal::Tuple(elems) => {
                    let k = match int_lit(idx) {
                        Some(k) if k >= 0 && (k as usize) < elems.len() => k as usize,
                        Some(k) => return Err(format!(
                            "GPU: tuple index {k} out of range (len {})", elems.len())),
                        None => return Err("GPU: tuple index must be a constant integer".into()),
                    };
                    Ok(elems.into_iter().nth(k).unwrap())
                }
                _ => Err("GPU: indexing is only supported on tuples here (tensor gather lives inside tensor(...))".into()),
            }
        }

        // 1-D tensor literal `[a, b, c]` — elements must reduce to scalars.
        Expr::Array(elems) => {
            let data = eval_scalar_elems(elems, env, ctx, scope)?;
            let n = data.len();
            Ok(upload_f32(ctx, data, vec![n]))
        }
        // 2-D tensor literal `[a,b; c,d]` — elements must reduce to scalars.
        Expr::TensorLit(rows) => {
            let r = rows.len();
            let c = rows.first().map_or(0, |row| row.len());
            let mut data = Vec::with_capacity(r * c);
            for row in rows {
                if row.len() != c {
                    return Err("GPU: tensor literal rows must have equal length".into());
                }
                data.extend(eval_scalar_elems(row, env, ctx, scope)?);
            }
            Ok(upload_f32(ctx, data, vec![r, c]))
        }

        // A lambda bound to a name (`f = x -> x*x`) becomes a callable value in
        // the block scope. It is applied by inlining its body (see `eval_apply`).
        Expr::Lambda(params, _, body) => Ok(GpuVal::Fn(Rc::new((
            params.iter().map(|p| p.name.clone()).collect(),
            (**body).clone(),
        )))),

        other => Err(format!("GPU: unsupported expression in GPU block: {other:?}")),
    }
}

/// Evaluate a list of expressions that must each reduce to a scalar (tensor
/// literal elements), returning their f32 values.
fn eval_scalar_elems(
    elems: &[Expr],
    env: &Env,
    ctx: &GpuContext,
    scope: &mut HashMap<String, GpuVal>,
) -> Result<Vec<f32>, String> {
    elems.iter().map(|e| match eval_gpu(e, env, ctx, scope)? {
        GpuVal::Scalar(s) => Ok(s as f32),
        GpuVal::Buffer { .. } | GpuVal::Host { .. } | GpuVal::Tuple(_) | GpuVal::Fn(_)
        | GpuVal::Complex { .. } | GpuVal::CScalar(..) | GpuVal::Field { .. } =>
            Err("GPU: tensor literals must contain scalar elements".to_string()),
    }).collect()
}

/// Dispatch a function application inside a GPU block.
fn eval_apply(
    f: &Expr,
    args: &[Expr],
    env: &Env,
    ctx: &GpuContext,
    scope: &mut HashMap<String, GpuVal>,
) -> Result<GpuVal, String> {
    // Namespaced calls, e.g. `ops.lap(T, dx)` or `forms.d(f)`.
    if let Expr::Member(base, member) = f {
        if let Expr::Var(ns) = &**base {
            if ns == "ops" {
                return ops_op(member, args, env, ctx, scope);
            }
            if ns == "forms" {
                return forms_op(member, args, env, ctx, scope);
            }
        }
        return Err(format!("GPU: `{}.{member}` not supported in a GPU block", fmt_member_base(base)));
    }

    let name = match f {
        Expr::Var(n) => n.as_str(),
        _ => return Err("GPU: only named function calls are supported in a GPU block".into()),
    };

    // A lambda/function bound to this name in the block scope wins over builtins
    // (a block-local `f = …` shadows). Inline-apply it by beta reduction.
    if let Some(GpuVal::Fn(func)) = scope.get(name) {
        let func = func.clone();
        let argvals = args.iter()
            .map(|a| eval_gpu(a, env, ctx, scope))
            .collect::<Result<Vec<_>, _>>()?;
        return apply_gpu_fn(&func.0, &func.1, argvals, env, ctx, scope);
    }

    match name {
        "iterate" => gpu_iterate(args, env, ctx, scope),
        "scan"    => gpu_scan(args, env, ctx, scope),

        // `if(cond, a, b)` — lazy special form, exactly like the CPU: the
        // condition must reduce to a scalar and only the taken branch is
        // evaluated. (Per-element masking lives inside `tensor(...)` via `select`;
        // a tensor condition errors here, matching the CPU's `.num("if")`.)
        "if" => {
            if args.len() != 3 { return Err("if(cond, a, b) expects 3 args".into()); }
            let cond = match eval_gpu(&args[0], env, ctx, scope)? {
                GpuVal::Scalar(s) => s,
                _ => return Err("GPU: if: condition must be a scalar".into()),
            };
            if cond != 0.0 { eval_gpu(&args[1], env, ctx, scope) }
            else           { eval_gpu(&args[2], env, ctx, scope) }
        }

        // GPU-side tensor construction from an index lambda (built on the device,
        // never materialized cell-by-cell on the CPU). See `gpu_tensor`.
        "tensor"  => gpu_tensor(args, env, ctx, scope),

        // Stencils available as flat builtins.
        "shift" | "roll" => shift_roll(name, args, env, ctx, scope),

        // Whole-tensor reductions → scalar.
        "sum" | "mean" | "min" | "max" if args.len() == 1 => {
            let v = eval_gpu(&args[0], env, ctx, scope)?;
            reduce_val(ctx, name, v)
        }
        // `any`/`all` → scalar 0/1 (any/every leaf nonzero). Map to flags, reduce.
        "any" | "all" if args.len() == 1 => {
            let v = eval_gpu(&args[0], env, ctx, scope)?;
            any_all_val(ctx, name, v)
        }
        // Binary min/max (elementwise / broadcast).
        "min" | "max" if args.len() == 2 => {
            let a = eval_gpu(&args[0], env, ctx, scope)?;
            let b = eval_gpu(&args[1], env, ctx, scope)?;
            binary_minmax(ctx, name, a, b)
        }

        // matmul(A, B) — handled by a dedicated kernel.
        "matmul" if args.len() == 2 => gpu_matmul(args, env, ctx, scope),

        // fft(T) / ifft(T) — n-D DFT over all axes (real input → complex tensor).
        "fft"  => gpu_fft(args, env, ctx, scope, true),
        "ifft" => gpu_fft(args, env, ctx, scope, false),

        // lerp(a, b, t) = a + (b - a)*t, broadcasting scalars/tensors via binop.
        "lerp" if args.len() == 3 => {
            let a = eval_gpu(&args[0], env, ctx, scope)?;
            let b = eval_gpu(&args[1], env, ctx, scope)?;
            let t = eval_gpu(&args[2], env, ctx, scope)?;
            let bma = binop(ctx, &Op::Sub, b, a.clone())?;
            let tba = binop(ctx, &Op::Mul, t, bma)?;
            binop(ctx, &Op::Add, a, tba)
        }

        // clamp(x, lo, hi) — elementwise, with scalar lo/hi (matches CPU).
        "clamp" if args.len() == 3 => {
            let x  = eval_gpu(&args[0], env, ctx, scope)?;
            let lo = cpu_scalar(&args[1], env)?;
            let hi = cpu_scalar(&args[2], env)?;
            if lo > hi { return Err(format!("clamp: lo ({lo}) > hi ({hi})")); }
            let lower = binary_minmax(ctx, "max", x, GpuVal::Scalar(lo))?;
            binary_minmax(ctx, "min", lower, GpuVal::Scalar(hi))
        }

        // Unary math, plus the complex projections re/im/conj/arg (which work on
        // real values too). Complex inputs dispatch inside `unary_val`.
        _ if args.len() == 1 && (unary_wgsl(name, "x").is_some() || matches!(name, "re" | "im" | "conj" | "arg")) => {
            let v = eval_gpu(&args[0], env, ctx, scope)?;
            unary_val(ctx, name, v)
        }

        // `get(cell)` is the one permitted host helper. It is resolved up front
        // by `hoist_gets` (before any kernel runs) and rewritten to a capture, so
        // it should never reach here. If it does, it came from inside a *called*
        // function body rather than directly in the block — reject it clearly
        // rather than evaluating on the host mid-kernel.
        "get" => Err(
            "GPU: `get(cell)` must appear directly in the GPU block, not inside a called function".into()),

        // A user function from the enclosing CPU scope, applied by inlining.
        _ => {
            if let Some(Val::Fn(params, body, _, _, _)) = env.vars.get(name) {
                let params = params.clone();
                let body = body.clone();
                let argvals = args.iter()
                    .map(|a| eval_gpu(a, env, ctx, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                return apply_gpu_fn(&params, &body, argvals, env, ctx, scope);
            }
            Err(format!("GPU: function `{name}` is not available in a GPU block"))
        }
    }
}

fn fmt_member_base(e: &Expr) -> String {
    match e { Expr::Var(n) => n.clone(), _ => "<expr>".into() }
}

/// Evaluate a configuration scalar (dx / bc / axis / shift amount) on the CPU.
/// This resolves literals, captured scalars, and namespace sentinels such as
/// `ops.neumann`.
fn cpu_scalar(e: &Expr, env: &Env) -> Result<f64, String> {
    crate::eval::eval(e, env)?.num("GPU op argument")
}

fn expect_buffer(gctx: &GpuContext, v: GpuVal, what: &str) -> Result<(Rc<wgpu::Buffer>, Vec<usize>, usize), String> {
    match materialize(gctx, v) {
        GpuVal::Buffer { buf, shape, len } => Ok((buf, shape, len)),
        GpuVal::Scalar(_) => Err(format!("GPU: {what} expects a tensor, got a scalar")),
        GpuVal::Tuple(_) => Err(format!("GPU: {what} expects a tensor, got a tuple")),
        GpuVal::Fn(_) => Err(format!("GPU: {what} expects a tensor, got a function")),
        GpuVal::Complex { .. } => Err(format!("GPU: {what} expects a real tensor, got a complex tensor")),
        GpuVal::CScalar(..) => Err(format!("GPU: {what} expects a tensor, got a complex scalar")),
        GpuVal::Field { .. } => Err(format!("GPU: {what} expects a plain tensor, got a field")),
        GpuVal::Host { .. } => unreachable!("materialized above"),
    }
}

/// The WGSL infix/function form for an arithmetic op, given operand expressions.
fn op_expr(op: &Op, a: &str, b: &str) -> Result<String, String> {
    Ok(match op {
        Op::Add  => format!("({a} + {b})"),
        Op::Sub  => format!("({a} - {b})"),
        Op::Mul  => format!("({a} * {b})"),
        Op::Div  => format!("({a} / {b})"),
        Op::Pow  => format!("pow({a}, {b})"),
        Op::Lt   => format!("f32({a} < {b})"),
        Op::Gt   => format!("f32({a} > {b})"),
        Op::LtEq => format!("f32({a} <= {b})"),
        Op::GtEq => format!("f32({a} >= {b})"),
        Op::Eq   => format!("f32({a} == {b})"),
        Op::Ne   => format!("f32({a} != {b})"),
        // `//` floor-divide and `%` remainder. WGSL float `%` is `a - b*trunc(a/b)`
        // (sign of dividend), matching Rust's `f64 %` used on the CPU.
        Op::FloorDiv => format!("floor({a} / {b})"),
        Op::Rem      => format!("({a} % {b})"),
        // Logical `&`/`|` truncate to int (CPU uses `x as i64`) and test nonzero,
        // returning 0.0/1.0. WGSL `i32(f)` truncates toward zero, like `as i64`.
        Op::And  => format!("f32((i32({a}) != 0) && (i32({b}) != 0))"),
        Op::Or   => format!("f32((i32({a}) != 0) || (i32({b}) != 0))"),
    })
}


/// Unary negation on the GPU, recursing over tuple trees (like the CPU `neg_val`).
fn neg_gpu(ctx: &GpuContext, v: GpuVal) -> Result<GpuVal, String> {
    match materialize(ctx, v) {
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar(-s)),
        GpuVal::Buffer { buf, shape, len } => {
            let out = run_map(ctx, &[&buf], len, "-in0[i]")?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        GpuVal::Tuple(elems) => Ok(GpuVal::Tuple(
            elems.into_iter().map(|e| neg_gpu(ctx, e)).collect::<Result<Vec<_>, _>>()?)),
        GpuVal::Fn(_) => Err("GPU: cannot negate a function".into()),
        GpuVal::CScalar(r, i) => Ok(make_cscalar(-r, -i)),
        GpuVal::Complex { buf, shape, len } => {
            let out = run_cmap(ctx, &[(buf.as_ref(), true)], len, "-(in0[i])")?;
            Ok(GpuVal::Complex { buf: out, shape, len })
        }
        f @ GpuVal::Field { .. } => field_binop(ctx, &Op::Mul, GpuVal::Scalar(-1.0), f),
        GpuVal::Host { .. } => unreachable!("materialized above"),
    }
}

/// Evaluate `lhs op rhs` on the GPU (or CPU for scalar/scalar).
fn binop(ctx: &GpuContext, op: &Op, lhs: GpuVal, rhs: GpuVal) -> Result<GpuVal, String> {
    // Tree broadcast: a tuple is a node, scalars/buffers are leaves. Mirrors the
    // CPU `binop_tuple` — two tuples broadcast structurally, a tuple vs a leaf
    // broadcasts the leaf into every field. (Matches CPU tree semantics.)
    match (&lhs, &rhs) {
        (GpuVal::Tuple(_), _) | (_, GpuVal::Tuple(_)) => return match (lhs, rhs) {
            (GpuVal::Tuple(ls), GpuVal::Tuple(rs)) => {
                if ls.len() != rs.len() {
                    return Err(format!("GPU: tuple op tuple: length mismatch ({} vs {})", ls.len(), rs.len()));
                }
                Ok(GpuVal::Tuple(ls.into_iter().zip(rs)
                    .map(|(l, r)| binop(ctx, op, l, r)).collect::<Result<Vec<_>, _>>()?))
            }
            (GpuVal::Tuple(ls), leaf) => Ok(GpuVal::Tuple(ls.into_iter()
                .map(|l| binop(ctx, op, l, leaf.clone())).collect::<Result<Vec<_>, _>>()?)),
            (leaf, GpuVal::Tuple(rs)) => Ok(GpuVal::Tuple(rs.into_iter()
                .map(|r| binop(ctx, op, leaf.clone(), r)).collect::<Result<Vec<_>, _>>()?)),
            _ => unreachable!(),
        },
        _ => {}
    }
    // A field operand routes to the field elementwise path (geometry-preserving).
    if matches!(lhs, GpuVal::Field { .. }) || matches!(rhs, GpuVal::Field { .. }) {
        return field_binop(ctx, op, lhs, rhs);
    }
    // Any complex operand routes to the complex elementwise path.
    if is_complex_val(&lhs) || is_complex_val(&rhs) {
        return cbinop(ctx, op, lhs, rhs);
    }
    match (materialize(ctx, lhs), materialize(ctx, rhs)) {
        (GpuVal::Scalar(x), GpuVal::Scalar(y)) => Ok(GpuVal::Scalar(scalar_op(op, x, y)?)),

        (GpuVal::Buffer { buf, shape, len }, GpuVal::Scalar(s)) => {
            let expr = op_expr(op, "in0[i]", &wgsl_f32(s))?;
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }

        (GpuVal::Scalar(s), GpuVal::Buffer { buf, shape, len }) => {
            let expr = op_expr(op, &wgsl_f32(s), "in0[i]")?;
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }

        (
            GpuVal::Buffer { buf: a, shape: sa, len: la },
            GpuVal::Buffer { buf: b, shape: sb, len: lb },
        ) => {
            if sa != sb {
                return Err(format!(
                    "GPU: shape mismatch in elementwise op: {sa:?} vs {sb:?}"
                ));
            }
            let _ = lb;
            let expr = op_expr(op, "in0[i]", "in1[i]")?;
            let out = run_map(ctx, &[&a, &b], la, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape: sa, len: la })
        }
        (GpuVal::Tuple(_), _) | (_, GpuVal::Tuple(_)) =>
            Err("GPU: arithmetic on a tuple is not supported (operate on its fields)".into()),
        // Complex operands are routed to `cbinop` before this point.
        _ => unreachable!("complex routed earlier; host values materialized above"),
    }
}

fn scalar_op(op: &Op, x: f64, y: f64) -> Result<f64, String> {
    Ok(match op {
        Op::Add  => x + y,
        Op::Sub  => x - y,
        Op::Mul  => x * y,
        Op::Div  => x / y,
        Op::Pow  => x.powf(y),
        Op::Lt   => (x < y) as i64 as f64,
        Op::Gt   => (x > y) as i64 as f64,
        Op::LtEq => (x <= y) as i64 as f64,
        Op::GtEq => (x >= y) as i64 as f64,
        Op::Eq   => (x == y) as i64 as f64,
        Op::Ne   => (x != y) as i64 as f64,
        Op::FloorDiv => (x / y).floor(),
        Op::Rem      => x % y,
        Op::And  => ((x as i64 != 0) && (y as i64 != 0)) as i64 as f64,
        Op::Or   => ((x as i64 != 0) || (y as i64 != 0)) as i64 as f64,
    })
}

/// Format an f64 as a WGSL f32 literal (always with a decimal point).
fn wgsl_f32(x: f64) -> String {
    let v = x as f32;
    if v.is_finite() {
        let s = format!("{v:?}"); // Debug for f32 always emits a decimal point
        format!("f32({s})")
    } else if v.is_nan() {
        "f32(0.0) / f32(0.0)".into()
    } else if v > 0.0 {
        "f32(1.0) / f32(0.0)".into()
    } else {
        "f32(-1.0) / f32(0.0)".into()
    }
}

// ───────────────────────────── unary math ─────────────────────────────

/// The WGSL expression computing unary `name` applied to operand string `x`,
/// or `None` if `name` is not a supported GPU unary function.
fn unary_wgsl(name: &str, x: &str) -> Option<String> {
    Some(match name {
        "exp"   => format!("exp({x})"),
        "ln"    => format!("log({x})"),
        "log2"  => format!("log2({x})"),
        "log10" => format!("(log({x}) * f32(0.4342944819032518))"),
        "sqrt"  => format!("sqrt({x})"),
        "cbrt"  => format!("(sign({x}) * pow(abs({x}), f32(0.3333333333333333)))"),
        "sin"   => format!("sin({x})"),
        "cos"   => format!("cos({x})"),
        "tan"   => format!("tan({x})"),
        "asin"  => format!("asin({x})"),
        "acos"  => format!("acos({x})"),
        "atan"  => format!("atan({x})"),
        "sinh"  => format!("sinh({x})"),
        "cosh"  => format!("cosh({x})"),
        "tanh"  => format!("tanh({x})"),
        "abs"   => format!("abs({x})"),
        "sign"  => format!("sign({x})"),
        "floor" => format!("floor({x})"),
        "ceil"  => format!("ceil({x})"),
        "trunc" => format!("trunc({x})"),
        "frac"  => format!("fract({x})"),
        // Finiteness predicates (WGSL has no isnan/isinf): NaN ≠ itself; |Inf|
        // exceeds the f32 max. Return 0.0/1.0, matching the CPU builtins.
        "isnan"    => format!("f32({x} != {x})"),
        "isinf"    => format!("f32(abs({x}) > f32(3.4028235e38))"),
        "isfinite" => format!("f32(abs({x}) <= f32(3.4028235e38))"),
        _ => return None,
    })
}

/// CPU evaluation of a unary function (scalar fast path; must match `unary_wgsl`).
fn unary_cpu(name: &str, x: f64) -> f64 {
    match name {
        "exp" => x.exp(), "ln" => x.ln(), "log2" => x.log2(), "log10" => x.log10(),
        "sqrt" => x.sqrt(), "cbrt" => x.cbrt(),
        "sin" => x.sin(), "cos" => x.cos(), "tan" => x.tan(),
        "asin" => x.asin(), "acos" => x.acos(), "atan" => x.atan(),
        "sinh" => x.sinh(), "cosh" => x.cosh(), "tanh" => x.tanh(),
        "abs" => x.abs(),
        "sign" => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        "floor" => x.floor(), "ceil" => x.ceil(), "trunc" => x.trunc(),
        "frac" => x.fract(),
        "isnan"    => x.is_nan() as i64 as f64,
        "isinf"    => x.is_infinite() as i64 as f64,
        "isfinite" => x.is_finite() as i64 as f64,
        _ => f64::NAN,
    }
}

fn unary_val(ctx: &GpuContext, name: &str, v: GpuVal) -> Result<GpuVal, String> {
    use std::f64::consts::PI;
    match materialize(ctx, v) {
        // re/im/conj/arg on a real value behave as on a complex number with im=0.
        GpuVal::Scalar(s) if matches!(name, "re" | "im" | "conj" | "arg") => Ok(GpuVal::Scalar(match name {
            "re" | "conj" => s,
            "im"          => 0.0,
            _             => if s >= 0.0 { 0.0 } else { PI }, // arg
        })),
        GpuVal::Buffer { buf, shape, len } if matches!(name, "re" | "im" | "conj" | "arg") => {
            let expr = match name {
                "re" | "conj" => "in0[i]".to_string(),
                "im"          => "f32(0.0)".to_string(),
                _             => format!("select({}, 0.0, in0[i] >= 0.0)", wgsl_f32(PI)), // arg
            };
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar(unary_cpu(name, s))),
        GpuVal::Buffer { buf, shape, len } => {
            let expr = unary_wgsl(name, "in0[i]")
                .ok_or_else(|| format!("GPU: {name} is not available in a GPU block"))?;
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        // Unary math broadcasts over a tuple's leaves (tree map), like the CPU.
        GpuVal::Tuple(elems) => Ok(GpuVal::Tuple(
            elems.into_iter().map(|e| unary_val(ctx, name, e)).collect::<Result<Vec<_>, _>>()?)),
        GpuVal::Fn(_) => Err(format!("GPU: {name} does not apply to a function")),
        GpuVal::Complex { buf, shape, len } => complex_unary(ctx, name, buf, shape, len),
        GpuVal::CScalar(r, i) => complex_scalar_unary(name, r, i),
        // Unary math on a field is componentwise (real-valued), preserving geometry.
        GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, variance, len } => {
            let expr = match name {
                "re" | "conj" => "in0[i]".to_string(),
                "im"          => "f32(0.0)".to_string(),
                "arg"         => format!("select({}, 0.0, in0[i] >= 0.0)", wgsl_f32(PI)),
                _ => unary_wgsl(name, "in0[i]")
                    .ok_or_else(|| format!("GPU: {name} is not available in a GPU block"))?,
            };
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, out, degree, variance, len))
        }
        GpuVal::Host { .. } => unreachable!("materialized above"),
    }
}

/// Elementwise `min`/`max` of two operands.
fn binary_minmax(ctx: &GpuContext, name: &str, a: GpuVal, b: GpuVal) -> Result<GpuVal, String> {
    let f = name; // "min" | "max"
    match (materialize(ctx, a), materialize(ctx, b)) {
        (GpuVal::Scalar(x), GpuVal::Scalar(y)) =>
            Ok(GpuVal::Scalar(if f == "min" { x.min(y) } else { x.max(y) })),
        (GpuVal::Buffer { buf, shape, len }, GpuVal::Scalar(s)) => {
            let out = run_map(ctx, &[&buf], len, &format!("{f}(in0[i], {})", wgsl_f32(s)))?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        (GpuVal::Scalar(s), GpuVal::Buffer { buf, shape, len }) => {
            let out = run_map(ctx, &[&buf], len, &format!("{f}({}, in0[i])", wgsl_f32(s)))?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        (GpuVal::Buffer { buf: a, shape: sa, len: la },
         GpuVal::Buffer { buf: b, shape: sb, len: lb }) => {
            if sa != sb { return Err(format!("GPU: shape mismatch in {f}: {sa:?} vs {sb:?}")); }
            let _ = lb;
            let out = run_map(ctx, &[&a, &b], la, &format!("{f}(in0[i], in1[i])"))?;
            Ok(GpuVal::Buffer { buf: out, shape: sa, len: la })
        }
        (GpuVal::Tuple(_), _) | (_, GpuVal::Tuple(_)) =>
            Err(format!("GPU: {f} does not apply to a tuple")),
        (GpuVal::Complex { .. } | GpuVal::CScalar(..), _) | (_, GpuVal::Complex { .. } | GpuVal::CScalar(..)) =>
            Err(format!("GPU: {f} is not defined on complex values (no ordering)")),
        (GpuVal::Field { .. }, _) | (_, GpuVal::Field { .. }) =>
            Err(format!("GPU: {f} is not defined on fields")),
        _ => unreachable!("Host values are materialized above"),
    }
}

// ───────────────────────────── reductions ─────────────────────────────

/// `any`/`all` → scalar 0/1. Map each element to a nonzero flag, then reduce by
/// `max` (any) or `min` (all). Matches the CPU `any`/`all` builtins.
fn any_all_val(ctx: &GpuContext, name: &str, v: GpuVal) -> Result<GpuVal, String> {
    match materialize(ctx, v) {
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar((s != 0.0) as i64 as f64)),
        GpuVal::Buffer { buf, len, .. } => {
            if len == 0 { return Ok(GpuVal::Scalar(if name == "all" { 1.0 } else { 0.0 })); }
            let flags = run_map(ctx, &[&buf], len, "f32(in0[i] != f32(0.0))")?;
            let kind = if name == "any" { "max" } else { "min" };
            Ok(GpuVal::Scalar(reduce(ctx, flags, len, kind)?))
        }
        GpuVal::Tuple(_) => Err(format!("GPU: {name} over a tuple is not supported (reduce its fields)")),
        GpuVal::Fn(_) => Err(format!("GPU: {name} does not apply to a function")),
        GpuVal::Complex { .. } | GpuVal::CScalar(..) =>
            Err(format!("GPU: {name} is not defined on complex values")),
        // The CPU rejects any/all over a field (it is not a tuple/tensor) — match it.
        GpuVal::Field { .. } => Err(format!("GPU: {name} does not apply to a field (reduce its data, not the field)")),
        GpuVal::Host { .. } => unreachable!("materialized above"),
    }
}

/// Reduce a value with `sum`/`mean`/`min`/`max`, returning a scalar.
fn reduce_val(ctx: &GpuContext, name: &str, v: GpuVal) -> Result<GpuVal, String> {
    match materialize(ctx, v) {
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar(s)),
        GpuVal::Buffer { buf, len, .. } => {
            if len == 0 { return Err(format!("GPU: {name} of an empty tensor")); }
            let kind = match name { "sum" | "mean" => "sum", other => other };
            let total = reduce(ctx, buf, len, kind)?;
            Ok(GpuVal::Scalar(if name == "mean" { total / len as f64 } else { total }))
        }
        GpuVal::Tuple(_) => Err(format!("GPU: {name} does not apply to a tuple")),
        GpuVal::Fn(_) => Err(format!("GPU: {name} does not apply to a function")),
        // sum/mean over a complex tensor reduce the re and im parts independently.
        GpuVal::Complex { buf, len, .. } => {
            if len == 0 { return Err(format!("GPU: {name} of an empty tensor")); }
            match name {
                "sum" | "mean" => {
                    let re = run_c2r_map(ctx, &buf, len, "a0.x")?;
                    let im = run_c2r_map(ctx, &buf, len, "a0.y")?;
                    let sr = reduce(ctx, re, len, "sum")?;
                    let si = reduce(ctx, im, len, "sum")?;
                    if name == "mean" { Ok(make_cscalar(sr / len as f64, si / len as f64)) }
                    else { Ok(make_cscalar(sr, si)) }
                }
                _ => Err(format!("GPU: {name} is not defined on complex tensors (no ordering)")),
            }
        }
        GpuVal::CScalar(r, i) => match name {
            "sum" | "mean" => Ok(make_cscalar(r, i)),
            _ => Err(format!("GPU: {name} is not defined on complex values")),
        },
        // The CPU's `sum`/`mean`/… reject a bare field (it is not a tuple/tensor) —
        // match it so a GPU block and the CPU agree.
        GpuVal::Field { .. } => Err(format!("GPU: {name} does not apply to a field (reduce its data, not the field)")),
        GpuVal::Host { .. } => unreachable!("materialized above"),
    }
}

/// Tree reduction with a grid-stride load, looping passes on the CPU until a
/// single value remains. `kind` is "sum" | "min" | "max".
fn reduce(ctx: &GpuContext, src: Rc<wgpu::Buffer>, len: usize, kind: &str) -> Result<f64, String> {
    let (ident, comb) = match kind {
        "sum" => ("0.0", "a + b"),
        "min" => ("3.4028235e38", "min(a, b)"),
        "max" => ("-3.4028235e38", "max(a, b)"),
        _ => return Err(format!("GPU: unknown reduction {kind}")),
    };
    // For `sum`, the per-thread grid-stride loop is where the overwhelming
    // majority of additions happen, so it carries a Neumaier compensation term
    // (`comp`) to recover the low-order bits that plain f32 addition drops. This
    // gives ~f64-quality sums while keeping f32 storage. The (≤256-wide) workgroup
    // tree and the few cross-pass reductions stay plain — their error is negligible
    // by comparison. min/max have no rounding error, so they use the plain loop.
    let per_thread = if kind == "sum" {
        "    var acc: f32 = 0.0;\n\
    var comp: f32 = 0.0;\n\
    var idx: u32 = gid.x;\n\
    loop {\n\
        if (idx >= params.len) { break; }\n\
        let b = inp[idx];\n\
        let t = acc + b;\n\
        if (abs(acc) >= abs(b)) { comp = comp + ((acc - t) + b); }\n\
        else { comp = comp + ((b - t) + acc); }\n\
        acc = t;\n\
        idx = idx + params.total;\n\
    }\n\
    acc = acc + comp;\n".to_string()
    } else {
        format!(
            "    var acc: f32 = {ident};\n\
    var idx: u32 = gid.x;\n\
    loop {{\n\
        if (idx >= params.len) {{ break; }}\n\
        let a = acc; let b = inp[idx]; acc = {comb};\n\
        idx = idx + params.total;\n\
    }}\n"
        )
    };
    let src_wgsl = format!(
        "@group(0) @binding(0) var<storage, read> inp: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;\n\
struct Params {{ len: u32, total: u32 }};\n\
@group(0) @binding(2) var<uniform> params: Params;\n\
var<workgroup> sdata: array<f32, 256>;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
        @builtin(local_invocation_id) lid: vec3<u32>,\n\
        @builtin(workgroup_id) wid: vec3<u32>) {{\n\
{per_thread}\
    sdata[lid.x] = acc;\n\
    workgroupBarrier();\n\
    var s: u32 = 128u;\n\
    loop {{\n\
        if (s == 0u) {{ break; }}\n\
        if (lid.x < s) {{ let a = sdata[lid.x]; let b = sdata[lid.x + s]; sdata[lid.x] = {comb}; }}\n\
        workgroupBarrier();\n\
        s = s >> 1u;\n\
    }}\n\
    if (lid.x == 0u) {{ outp[wid.x] = sdata[0]; }}\n\
}}\n"
    );
    let pipeline = get_pipeline(ctx, &src_wgsl);

    const WG: u32 = 256;
    const MAX_DIM: u32 = 65535;
    let mut cur = src;
    let mut n = len;
    while n > 1 {
        let needed = (n as u32).div_ceil(WG).max(1);
        let groups = needed.min(MAX_DIM);
        let total = groups * WG;
        let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-reduce-out"),
            size: (groups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-reduce-params"),
            contents: bytemuck::cast_slice(&[n as u32, total]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu-reduce-bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cur.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-reduce-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-reduce-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
        cur = Rc::new(out);
        n = groups as usize;
    }
    Ok(download(ctx, &cur, 1)?[0] as f64)
}

// ───────────────────────── stencils (ops.*, shift, roll) ─────────────────────────

/// A stencil tap: an integer offset per axis and a coefficient.
type Tap = (Vec<i64>, f64);

/// Laplacian taps: Σ_axis (f[+1] + f[-1] - 2 f[0]) / dx².
fn lap_taps(ndim: usize, dx: f64) -> Vec<Tap> {
    let inv = 1.0 / (dx * dx);
    let mut taps: Vec<Tap> = vec![(vec![0; ndim], -2.0 * ndim as f64 * inv)];
    for a in 0..ndim {
        let mut up = vec![0; ndim]; up[a] = 1; taps.push((up, inv));
        let mut dn = vec![0; ndim]; dn[a] = -1; taps.push((dn, inv));
    }
    taps
}

/// Central first-derivative taps along `axis`: (f[+1] - f[-1]) / (2 dx).
fn grad_taps(ndim: usize, dx: f64, axis: usize) -> Vec<Tap> {
    let c = 1.0 / (2.0 * dx);
    let mut up = vec![0; ndim]; up[axis] = 1;
    let mut dn = vec![0; ndim]; dn[axis] = -1;
    vec![(up, c), (dn, -c)]
}

/// Generate the WGSL for a constant stencil with the given taps and per-axis BC
/// (Neumann = edge-clamp, else periodic = wrap). The grid dims/strides arrive in
/// a `meta` storage buffer; everything structural (ndim, taps, BC) is baked in.
fn stencil_wgsl(ndim: usize, taps: &[Tap], neumann: &[bool]) -> String {
    let mut s = String::new();
    s += "@group(0) @binding(0) var<storage, read> src: array<f32>;\n";
    s += "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n";
    s += "@group(0) @binding(2) var<storage, read> gdim: array<u32>;\n";
    s += "struct Params { len: u32, row: u32, ndim: u32 };\n";
    s += "@group(0) @binding(3) var<uniform> params: Params;\n";
    s += "@compute @workgroup_size(256)\n";
    s += "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
    s += "  let i = gid.y * params.row + gid.x;\n";
    s += "  if (i >= params.len) { return; }\n";
    for a in 0..ndim {
        s += &format!("  let d{a} = gdim[{a}u];\n");
    }
    for a in 0..ndim {
        s += &format!("  let s{a} = gdim[{}u];\n", ndim + a);
    }
    for a in 0..ndim {
        s += &format!("  let c{a} = i32((i / s{a}) % d{a});\n");
    }
    s += "  var acc: f32 = 0.0;\n";
    for (off, coef) in taps {
        s += "  {\n";
        s += "    var nidx: u32 = 0u;\n";
        for a in 0..ndim {
            let o = off[a];
            if neumann[a] {
                s += &format!("    let n{a} = clamp(c{a} + ({o}), 0, i32(d{a}) - 1);\n");
            } else {
                s += &format!("    let m{a} = i32(d{a});\n");
                s += &format!("    let n{a} = (((c{a} + ({o})) % m{a}) + m{a}) % m{a};\n");
            }
            s += &format!("    nidx += u32(n{a}) * s{a};\n");
        }
        s += &format!("    acc += {} * src[nidx];\n", wgsl_f32(*coef));
        s += "  }\n";
    }
    s += "  out[i] = acc;\n";
    s += "}\n";
    s
}

/// Run a constant-stencil kernel over `src` (shape known), returning a new buffer
/// of the same shape.
fn run_stencil(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    shape: &[usize],
    len: usize,
    taps: &[Tap],
    neumann: &[bool],
) -> Result<Rc<wgpu::Buffer>, String> {
    let ndim = shape.len();
    // Row-major strides.
    let mut strides = vec![1usize; ndim];
    for a in (0..ndim.saturating_sub(1)).rev() {
        strides[a] = strides[a + 1] * shape[a + 1];
    }
    let src_wgsl = stencil_wgsl(ndim, taps, neumann);
    let pipeline = get_pipeline(ctx, &src_wgsl);

    let mut meta: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    meta.extend(strides.iter().map(|&s| s as u32));
    let meta_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-stencil-meta"),
        contents: bytemuck::cast_slice(&meta),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-stencil-out"),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    const WG: u32 = 256;
    const MAX_DIM: u32 = 65535;
    let needed = (len as u32).div_ceil(WG).max(1);
    let groups_x = needed.min(MAX_DIM);
    let groups_y = needed.div_ceil(groups_x).max(1);
    let row = groups_x * WG;

    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-stencil-params"),
        contents: bytemuck::cast_slice(&[len as u32, row, ndim as u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-stencil-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-stencil-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-stencil-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(Rc::new(out))
}

/// Gather component `comp` (of `ncomp`) out of an interleaved vector field whose
/// trailing axis indexes the component (component-fastest layout: cell `i`,
/// component `c` lives at `i*ncomp + c`), returning a `base_total`-long buffer.
fn extract_component(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    base_total: usize,
    ncomp: usize,
    comp: usize,
) -> Result<Rc<wgpu::Buffer>, String> {
    let src_wgsl =
        "@group(0) @binding(0) var<storage, read> inp: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;\n\
struct Params { len: u32, row: u32, ncomp: u32, comp: u32 };\n\
@group(0) @binding(2) var<uniform> params: Params;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y * params.row + gid.x;\n\
    if (i >= params.len) { return; }\n\
    outp[i] = inp[i * params.ncomp + params.comp];\n\
}\n";
    let pipeline = get_pipeline(ctx, src_wgsl);

    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-extract-out"),
        size: (base_total.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    const WG: u32 = 256;
    const MAX_DIM: u32 = 65535;
    let needed = (base_total as u32).div_ceil(WG).max(1);
    let groups_x = needed.min(MAX_DIM);
    let groups_y = needed.div_ceil(groups_x).max(1);
    let row = groups_x * WG;

    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-extract-params"),
        contents: bytemuck::cast_slice(&[base_total as u32, row, ncomp as u32, comp as u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-extract-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-extract-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-extract-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(Rc::new(out))
}

/// `ops.<name>(...)` inside a GPU block.
fn ops_op(name: &str, args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    // Field-polymorphic form: `ops.lap(f)` etc. take exactly one field argument
    // (dx/bc come from the field). A single argument that evaluates to a field
    // routes here; tensor forms always carry an explicit dx (≥ 2 args).
    if args.len() == 1 {
        let v = materialize(ctx, eval_gpu(&args[0], env, ctx, scope)?);
        if let GpuVal::Field { .. } = v {
            return gpu_field_op(name, v, ctx);
        }
        return Err(format!(
            "GPU: ops.{name} needs an explicit dx (e.g. ops.{name}(T, dx)) unless called on a field"));
    }
    match name {
        // ops.lap(T, dx [, bc])  — Laplacian; bc 0=periodic (default), 1=neumann.
        "lap" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("ops.lap(T, dx [, bc]) expects 2 or 3 args".into());
            }
            let t = eval_gpu(&args[0], env, ctx, scope)?;
            let dx = cpu_scalar(&args[1], env)?;
            let neumann = args.len() == 3 && cpu_scalar(&args[2], env)? != 0.0;
            let (buf, shape, len) = expect_buffer(ctx, t, "ops.lap")?;
            if shape.is_empty() { return Err("ops.lap: need a tensor of rank >= 1".into()); }
            let taps = lap_taps(shape.len(), dx);
            let bcs = vec![neumann; shape.len()];
            let out = run_stencil(ctx, &buf, &shape, len, &taps, &bcs)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        // ops.grad(T, dx, axis) — central difference along one axis (same shape).
        "grad" => {
            if args.len() != 3 {
                return Err("GPU: ops.grad requires an explicit axis: ops.grad(T, dx, axis)".into());
            }
            let t = eval_gpu(&args[0], env, ctx, scope)?;
            let dx = cpu_scalar(&args[1], env)?;
            let axis = cpu_scalar(&args[2], env)? as usize;
            let (buf, shape, len) = expect_buffer(ctx, t, "ops.grad")?;
            if axis >= shape.len() {
                return Err(format!("ops.grad: axis {axis} out of range for rank-{} tensor", shape.len()));
            }
            let taps = grad_taps(shape.len(), dx, axis);
            let bcs = vec![false; shape.len()]; // grad is periodic (matches CPU `central`)
            let out = run_stencil(ctx, &buf, &shape, len, &taps, &bcs)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
        // ops.div(V, dx) — divergence of a vector field. V shape = base ++ [ncomp]
        // (trailing axis = component, component-fastest), ncomp must equal the base
        // rank: div = Σ_a ∂(V[..,a])/∂x_a, periodic central differences (matches CPU).
        "div" => {
            if args.len() != 2 { return Err("ops.div(V, dx) expects 2 args".into()); }
            let t = eval_gpu(&args[0], env, ctx, scope)?;
            let dx = cpu_scalar(&args[1], env)?;
            let (buf, shape, _len) = expect_buffer(ctx, t, "ops.div")?;
            if shape.len() < 2 {
                return Err("ops.div: need a vector field of rank >= 2 (grid..., ncomp)".into());
            }
            let comps = *shape.last().unwrap();
            let base: Vec<usize> = shape[..shape.len() - 1].to_vec();
            if comps != base.len() {
                return Err(format!(
                    "ops.div: vector field has {comps} components but base is {}-D", base.len()));
            }
            let base_total: usize = base.iter().product();
            let bcs = vec![false; base.len()];
            let mut acc: Option<Rc<wgpu::Buffer>> = None;
            for a in 0..comps {
                let comp = extract_component(ctx, &buf, base_total, comps, a)?;
                let taps = grad_taps(base.len(), dx, a);
                let d = run_stencil(ctx, &comp, &base, base_total, &taps, &bcs)?;
                acc = Some(match acc {
                    None => d,
                    Some(prev) => run_map(ctx, &[&prev, &d], base_total, "in0[i] + in1[i]")?,
                });
            }
            Ok(GpuVal::Buffer { buf: acc.unwrap(), shape: base, len: base_total })
        }
        // ops.curl(V, dx) — 2-D scalar curl ∂V_y/∂x − ∂V_x/∂y, V shape [r, c, 2].
        "curl" => {
            if args.len() != 2 { return Err("ops.curl(V, dx) expects 2 args".into()); }
            let t = eval_gpu(&args[0], env, ctx, scope)?;
            let dx = cpu_scalar(&args[1], env)?;
            let (buf, shape, _len) = expect_buffer(ctx, t, "ops.curl")?;
            if shape.len() != 3 || shape[2] != 2 {
                return Err("ops.curl: only the 2-D scalar curl is supported (V shape [r, c, 2])".into());
            }
            let base = vec![shape[0], shape[1]];
            let base_total = base[0] * base[1];
            let bcs = [false, false];
            let vx = extract_component(ctx, &buf, base_total, 2, 0)?;
            let vy = extract_component(ctx, &buf, base_total, 2, 1)?;
            let dvy = run_stencil(ctx, &vy, &base, base_total, &grad_taps(2, dx, 0), &bcs)?;
            let dvx = run_stencil(ctx, &vx, &base, base_total, &grad_taps(2, dx, 1), &bcs)?;
            let out = run_map(ctx, &[&dvy, &dvx], base_total, "in0[i] - in1[i]")?;
            Ok(GpuVal::Buffer { buf: out, shape: base, len: base_total })
        }
        // Spectral operators — built on the GPU FFT (src/gpu/fft.wgsl).
        "poisson" | "invlap" | "specgrad" => gpu_spectral_op(name, args, env, ctx, scope),

        _ => Err(format!("GPU: ops.{name} not supported in a GPU block")),
    }
}

/// `shift(T, n, axis)` (edge-clamp) and `roll(T, n, axis)` (wrap) as single-tap stencils.
fn shift_roll(name: &str, args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() != 3 {
        return Err(format!("{name}(T, n, axis) expects 3 args"));
    }
    let t = eval_gpu(&args[0], env, ctx, scope)?;
    let n = cpu_scalar(&args[1], env)? as i64;
    let axis = cpu_scalar(&args[2], env)? as usize;
    let (buf, shape, len) = expect_buffer(ctx, t, name)?;
    if axis >= shape.len() {
        return Err(format!("{name}: axis {axis} out of range for rank-{} tensor", shape.len()));
    }
    let mut off = vec![0i64; shape.len()];
    off[axis] = -n; // out[c] = in[c - n]
    let taps = vec![(off, 1.0)];
    let mut bcs = vec![false; shape.len()];
    if name == "shift" { bcs[axis] = true; } // clamp; roll wraps (periodic)
    let out = run_stencil(ctx, &buf, &shape, len, &taps, &bcs)?;
    Ok(GpuVal::Buffer { buf: out, shape, len })
}

// ───────────────────────── tensor construction (index lambdas) ─────────────────────────
//
// `tensor((i,j)->expr, m, n)` builds a tensor on the GPU: one fused kernel runs
// per output cell, recovering the multi-index from the thread id, binding the
// index parameters, and evaluating the lambda body — which is compiled directly
// from the AST to WGSL (no CPU per-cell evaluation). The body may also gather
// from captured tensors via `T[i,j]`. This is the GPU-side answer to slow
// CPU-built initial conditions.

/// A captured tensor referenced inside a tensor-lambda body, with the binding it
/// is wired to and its row-major strides (for gather index math).
struct TensorBind {
    binding: usize,
    shape:   Vec<usize>,
    strides: Vec<usize>,
}

/// Context threaded through the AST→WGSL emitter for a tensor lambda.
struct LamCtx<'a> {
    params:  &'a [String],
    tensors: &'a HashMap<String, TensorBind>,
    env:     &'a Env,
    scope:   &'a HashMap<String, GpuVal>,
}

/// Resolve the lambda argument of `tensor(...)` to (param names, body).
fn resolve_lambda(f: &Expr, env: &Env) -> Result<(Vec<String>, Expr), String> {
    match f {
        Expr::Lambda(ps, _, body) => {
            Ok((ps.iter().map(|p| p.name.clone()).collect(), (**body).clone()))
        }
        Expr::Var(name) => match env.vars.get(name) {
            Some(Val::Fn(ps, body, _, _, _)) => Ok((ps.clone(), body.clone())),
            Some(other) => Err(format!("GPU: tensor: `{name}` is not a function ({})", fmt_val(other))),
            None => Err(format!("GPU: undefined function `{name}`")),
        },
        _ => Err("GPU: tensor's first argument must be an index lambda".into()),
    }
}

/// Is `name` a tensor available for capture (already GPU-resident, or a CPU
/// tensor in the enclosing scope)?
fn is_tensor_capture(name: &str, env: &Env, scope: &HashMap<String, GpuVal>) -> bool {
    matches!(scope.get(name), Some(GpuVal::Buffer { .. }) | Some(GpuVal::Host { .. }))
        || matches!(env.vars.get(name), Some(Val::Tensor { .. }))
}

/// Walk a lambda body collecting the names of captured tensors it references
/// (so they can be uploaded and bound before the kernel runs).
fn collect_lambda_tensors(
    e: &Expr,
    params: &[String],
    env: &Env,
    scope: &HashMap<String, GpuVal>,
    out: &mut Vec<String>,
) {
    match e {
        Expr::Var(name) => {
            if !params.iter().any(|p| p == name)
                && is_tensor_capture(name, env, scope)
                && !out.iter().any(|o| o == name)
            {
                out.push(name.clone());
            }
        }
        Expr::Neg(x) | Expr::Not(x) => collect_lambda_tensors(x, params, env, scope, out),
        Expr::BinOp(l, _, r) | Expr::Range(l, r) => {
            collect_lambda_tensors(l, params, env, scope, out);
            collect_lambda_tensors(r, params, env, scope, out);
        }
        Expr::Index(b, i) => {
            collect_lambda_tensors(b, params, env, scope, out);
            collect_lambda_tensors(i, params, env, scope, out);
        }
        Expr::Apply(f, args) => {
            collect_lambda_tensors(f, params, env, scope, out);
            for a in args { collect_lambda_tensors(a, params, env, scope, out); }
        }
        Expr::Tuple(xs) | Expr::Array(xs) => {
            for x in xs { collect_lambda_tensors(x, params, env, scope, out); }
        }
        Expr::TensorLit(rows) => {
            for row in rows { for x in row { collect_lambda_tensors(x, params, env, scope, out); } }
        }
        Expr::Member(b, _) | Expr::GpuBlock(b) => collect_lambda_tensors(b, params, env, scope, out),
        Expr::Block(stmts) => {
            for s in stmts {
                match s {
                    BlockStmt::Expr(x) => collect_lambda_tensors(x, params, env, scope, out),
                    BlockStmt::Def(Def::Var(_, x)) => collect_lambda_tensors(x, params, env, scope, out),
                    BlockStmt::Def(Def::Func(..)) => {}
                }
            }
        }
        Expr::Slice(a, b) => {
            if let Some(a) = a { collect_lambda_tensors(a, params, env, scope, out); }
            if let Some(b) = b { collect_lambda_tensors(b, params, env, scope, out); }
        }
        _ => {}
    }
}

/// A non-negative integer literal, if `e` is one.
fn int_lit(e: &Expr) -> Option<i64> {
    match e {
        Expr::Num(n) if n.fract() == 0.0 => Some(*n as i64),
        _ => None,
    }
}

/// Resolve a bare variable to a CPU-known scalar (captured `Num`, or a scalar
/// block-local) so it can be baked into the kernel as a literal.
fn lookup_scalar(name: &str, ctx: &LamCtx) -> Option<f64> {
    if let Some(GpuVal::Scalar(s)) = ctx.scope.get(name) { return Some(*s); }
    if let Some(Val::Num(f)) = ctx.env.vars.get(name) { return Some(*f); }
    None
}

/// Emit a WGSL f32-valued expression for a tensor-lambda body node.
fn emit_expr(e: &Expr, ctx: &LamCtx) -> Result<String, String> {
    match e {
        Expr::Num(n) => Ok(wgsl_f32(*n)),
        Expr::Neg(x) => Ok(format!("(-({}))", emit_expr(x, ctx)?)),
        Expr::Not(x) => Ok(format!("f32(({}) == f32(0.0))", emit_expr(x, ctx)?)),

        Expr::Var(name) => {
            if ctx.params.iter().any(|p| p == name) {
                Ok(name.clone()) // declared as an f32 coordinate in the kernel preamble
            } else if let Some(s) = lookup_scalar(name, ctx) {
                Ok(wgsl_f32(s))
            } else if ctx.tensors.contains_key(name) {
                Err(format!("GPU: tensor `{name}` used without an index in a tensor lambda; write `{name}[i, …]`"))
            } else {
                Err(format!("GPU: undefined variable `{name}` in tensor lambda"))
            }
        }

        // Integer powers expand to repeated multiplication: exact and sign-correct
        // (WGSL `pow` returns NaN for a negative base, which breaks e.g. (i-c)^2).
        Expr::BinOp(l, Op::Pow, r) if int_lit(r).map_or(false, |k| (0..=16).contains(&k)) => {
            let k = int_lit(r).unwrap();
            if k == 0 { return Ok("f32(1.0)".into()); }
            let a = emit_expr(l, ctx)?;
            let factors = std::iter::repeat(format!("({a})")).take(k as usize).collect::<Vec<_>>().join(" * ");
            Ok(format!("({factors})"))
        }
        Expr::BinOp(l, op, r) => {
            let a = emit_expr(l, ctx)?;
            let b = emit_expr(r, ctx)?;
            op_expr(op, &a, &b)
        }

        Expr::Apply(f, args) => emit_apply(f, args, ctx),
        Expr::Index(base, idx) => emit_index(base, idx, ctx),

        Expr::Block(stmts) => {
            let mut last = None;
            for s in stmts {
                match s {
                    BlockStmt::Expr(x) => last = Some(emit_expr(x, ctx)?),
                    BlockStmt::Def(_) => return Err(
                        "GPU: local bindings inside a tensor lambda are not supported; inline the expression".into()),
                }
            }
            last.ok_or_else(|| "GPU: tensor lambda block has no result expression".to_string())
        }

        other => Err(format!("GPU: unsupported expression in tensor lambda: {other:?}")),
    }
}

/// Emit a function call inside a tensor lambda (unary math, or 2-arg min/max).
fn emit_apply(f: &Expr, args: &[Expr], ctx: &LamCtx) -> Result<String, String> {
    let name = match f {
        Expr::Var(n) => n.as_str(),
        _ => return Err("GPU: only named function calls are supported in a tensor lambda".into()),
    };
    if args.len() == 1 {
        if unary_wgsl(name, "x").is_some() {
            let a = emit_expr(&args[0], ctx)?;
            return Ok(unary_wgsl(name, &a).unwrap());
        }
    }
    if args.len() == 2 && (name == "min" || name == "max") {
        let a = emit_expr(&args[0], ctx)?;
        let b = emit_expr(&args[1], ctx)?;
        return Ok(format!("{name}({a}, {b})"));
    }
    // `if(cond, a, b)` per-thread → WGSL `select`. NOTE: unlike the CPU `if`
    // (lazy — only the taken branch runs), `select` evaluates BOTH branches and
    // discards the untaken one. For pure index math that is harmless (a discarded
    // NaN/inf is fine); avoid relying on a branch *not* being evaluated.
    if args.len() == 3 && name == "if" {
        let cond = emit_expr(&args[0], ctx)?;
        let a = emit_expr(&args[1], ctx)?;
        let b = emit_expr(&args[2], ctx)?;
        return Ok(format!("select({b}, {a}, ({cond}) != f32(0.0))"));
    }
    Err(format!("GPU: function `{name}` not supported in a tensor lambda"))
}

/// Emit a captured-tensor gather `T[i, j, …]` as a clamped linear-index read.
fn emit_index(base: &Expr, idx: &Expr, ctx: &LamCtx) -> Result<String, String> {
    let name = match base {
        Expr::Var(n) => n,
        _ => return Err("GPU: indexing in a tensor lambda requires a captured tensor".into()),
    };
    let tb = ctx.tensors.get(name)
        .ok_or_else(|| format!("GPU: `{name}` is not a captured tensor"))?;
    let comps: Vec<&Expr> = match idx {
        Expr::Tuple(v) => v.iter().collect(),
        single => vec![single],
    };
    if comps.len() != tb.shape.len() {
        return Err(format!(
            "GPU: `{name}` has rank {} but was indexed with {} {}",
            tb.shape.len(), comps.len(), if comps.len() == 1 { "index" } else { "indices" }
        ));
    }
    let mut terms = Vec::with_capacity(comps.len());
    for (k, c) in comps.iter().enumerate() {
        let ce = emit_expr(c, ctx)?;
        let max = tb.shape[k] as i64 - 1;
        terms.push(format!("u32(clamp(i32({ce}), 0i, {max}i)) * {}u", tb.strides[k]));
    }
    Ok(format!("t{}[{}]", tb.binding, terms.join(" + ")))
}

/// Generate the WGSL for a tensor-construction kernel.
fn tensor_lambda_wgsl(
    params:  &[String],
    shape:   &[usize],
    strides: &[usize],
    tensors: &HashMap<String, TensorBind>,
    body:    &str,
) -> String {
    let mut s = String::new();
    s += "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n";
    let mut tv: Vec<&TensorBind> = tensors.values().collect();
    tv.sort_by_key(|t| t.binding);
    for t in &tv {
        s += &format!("@group(0) @binding({0}) var<storage, read> t{0}: array<f32>;\n", t.binding);
    }
    let pbind = tensors.len() + 1;
    s += "struct Params { len: u32, row: u32 };\n";
    s += &format!("@group(0) @binding({pbind}) var<uniform> params: Params;\n");
    s += "@compute @workgroup_size(256)\n";
    s += "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
    s += "  let lin = gid.y * params.row + gid.x;\n";
    s += "  if (lin >= params.len) { return; }\n";
    for (k, p) in params.iter().enumerate() {
        // Recover the index along axis k from the linear id (row-major).
        s += &format!("  let {p} = f32((lin / {}u) % {}u);\n", strides[k], shape[k]);
    }
    s += &format!("  out[lin] = {body};\n");
    s += "}\n";
    s
}

/// Dispatch a tensor-construction kernel, returning the output buffer.
fn run_tensor_lambda(
    ctx:   &GpuContext,
    tbufs: &[Rc<wgpu::Buffer>],
    len:   usize,
    src:   &str,
) -> Result<Rc<wgpu::Buffer>, String> {
    let pipeline = get_pipeline(ctx, src);

    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-tensor-out"),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    const WG: u32 = 256;
    const MAX_DIM: u32 = 65535;
    let needed = (len as u32).div_ceil(WG).max(1);
    let groups_x = needed.min(MAX_DIM);
    let groups_y = needed.div_ceil(groups_x).max(1);
    let row = groups_x * WG;

    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-tensor-params"),
        contents: bytemuck::cast_slice(&[len as u32, row]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(tbufs.len() + 2);
    entries.push(wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() });
    for (j, b) in tbufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: (j + 1) as u32, resource: b.as_entire_binding() });
    }
    let pbind = tbufs.len() + 1;
    entries.push(wgpu::BindGroupEntry { binding: pbind as u32, resource: params.as_entire_binding() });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-tensor-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-tensor-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-tensor-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(Rc::new(out))
}

/// `tensor(f, n1, n2, …)` — build a tensor on the GPU from an index lambda.
fn gpu_tensor(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() < 2 {
        return Err("tensor(f, n1, n2, …) expects an index lambda and at least one dimension".into());
    }
    let (params, body) = resolve_lambda(&args[0], env)?;

    let shape: Vec<usize> = args[1..].iter().map(|a| {
        let d = cpu_scalar(a, env)?;
        if d < 0.0 || d.fract() != 0.0 {
            return Err(format!("GPU: tensor dimension must be a non-negative integer, got {d}"));
        }
        Ok(d as usize)
    }).collect::<Result<_, _>>()?;

    if params.len() != shape.len() {
        return Err(format!(
            "GPU: tensor lambda takes {} index argument(s) but {} dimension(s) were given",
            params.len(), shape.len()
        ));
    }
    let ndim = shape.len();
    let len: usize = shape.iter().product();

    // Row-major strides for the output grid.
    let mut strides = vec![1usize; ndim];
    for a in (0..ndim.saturating_sub(1)).rev() {
        strides[a] = strides[a + 1] * shape[a + 1];
    }

    // Upload + bind every captured tensor the body gathers from.
    let mut tnames: Vec<String> = Vec::new();
    collect_lambda_tensors(&body, &params, env, scope, &mut tnames);
    let mut tensors: HashMap<String, TensorBind> = HashMap::new();
    let mut tbufs: Vec<Rc<wgpu::Buffer>> = Vec::with_capacity(tnames.len());
    for (k, name) in tnames.iter().enumerate() {
        let gv = eval_gpu(&Expr::Var(name.clone()), env, ctx, scope)?;
        let (buf, tshape, _) = expect_buffer(ctx, gv, name)?;
        let mut tstrides = vec![1usize; tshape.len()];
        for a in (0..tshape.len().saturating_sub(1)).rev() {
            tstrides[a] = tstrides[a + 1] * tshape[a + 1];
        }
        tensors.insert(name.clone(), TensorBind { binding: k + 1, shape: tshape, strides: tstrides });
        tbufs.push(buf);
    }

    let body_wgsl = {
        let lctx = LamCtx { params: &params, tensors: &tensors, env, scope };
        emit_expr(&body, &lctx)?
    };
    let src = tensor_lambda_wgsl(&params, &shape, &strides, &tensors, &body_wgsl);
    let out = run_tensor_lambda(ctx, &tbufs, len, &src)?;
    Ok(GpuVal::Buffer { buf: out, shape, len })
}

// ───────────────────────── iterate / scan (residency) ─────────────────────────

/// A resolved iterate/scan step function.
enum Step {
    /// Inline lambda or named user function: bind `params`, evaluate `body`.
    /// One param binds the whole state; several params destructure a tuple
    /// state (coupled fields), e.g. `(U, V) -> (…, …)`.
    Body { params: Vec<String>, body: Expr },
    /// A builtin applied each step (e.g. `exp`, `sin`).
    Builtin(String),
}

/// Resolve a step function (named CPU fn, inline lambda, or builtin) for a loop.
fn resolve_step(f: &Expr, env: &Env) -> Result<Step, String> {
    match f {
        Expr::Lambda(params, _, body) => {
            if params.is_empty() {
                return Err("GPU: step function must take at least 1 argument".into());
            }
            Ok(Step::Body { params: params.iter().map(|p| p.name.clone()).collect(), body: (**body).clone() })
        }
        Expr::Var(name) => match env.vars.get(name) {
            Some(Val::Fn(params, body, _, _, _)) => {
                if params.is_empty() {
                    return Err(format!("GPU: step function `{name}` must take at least 1 argument"));
                }
                Ok(Step::Body { params: params.clone(), body: body.clone() })
            }
            Some(Val::Builtin(b)) => {
                if unary_wgsl(b, "x").is_none() {
                    return Err(format!("GPU: builtin `{b}` is not supported as an iterate/scan step"));
                }
                Ok(Step::Builtin(b.clone()))
            }
            Some(other) => Err(format!("GPU: `{name}` is not a function ({})", fmt_val(other))),
            None => Err(format!("GPU: undefined step function `{name}`")),
        },
        _ => Err("GPU: iterate/scan step must be a function or lambda".into()),
    }
}

/// Apply one step of the loop to `state`, producing the next state.
fn apply_step(
    step: &Step,
    state: GpuVal,
    env: &Env,
    ctx: &GpuContext,
    scope: &mut HashMap<String, GpuVal>,
) -> Result<GpuVal, String> {
    match step {
        Step::Body { params, body } => apply_gpu_fn(params, body, vec![state], env, ctx, scope),
        Step::Builtin(name) => unary_val(ctx, name, state),
    }
}

thread_local! {
    /// Depth of nested user-function inlining, to turn runaway recursion (which
    /// WGSL cannot express) into a clean error instead of a stack overflow.
    static APPLY_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Inline-apply a user lambda/function: bind `params` to `argvals` in `scope`,
/// evaluate `body`, then restore any shadowed bindings. Shared by `iterate`/`scan`
/// steps and direct calls (`f(x)`) so arity/tuple handling is identical.
///
/// Binding rules mirror the CPU's `apply_val`:
///   • `argvals.len() == params.len()` — bind positionally;
///   • one tuple argument with `params.len()` fields — destructure it (coupled
///     state, e.g. an `iterate((U,V) -> …, (U0,V0), n)`).
fn apply_gpu_fn(
    params: &[String],
    body: &Expr,
    argvals: Vec<GpuVal>,
    env: &Env,
    ctx: &GpuContext,
    scope: &mut HashMap<String, GpuVal>,
) -> Result<GpuVal, String> {
    const MAX_DEPTH: usize = 256;
    let depth = APPLY_DEPTH.with(|d| { let n = d.get() + 1; d.set(n); n });
    struct DepthGuard;
    impl Drop for DepthGuard {
        fn drop(&mut self) { APPLY_DEPTH.with(|d| d.set(d.get().saturating_sub(1))); }
    }
    let _guard = DepthGuard;
    if depth > MAX_DEPTH {
        return Err("GPU: function-call nesting too deep \
                    (recursion is not supported in a GPU block)".into());
    }

    let bindings: Vec<(String, GpuVal)> = if argvals.len() == params.len() {
        params.iter().cloned().zip(argvals).collect()
    } else if params.len() > 1 && argvals.len() == 1 {
        match argvals.into_iter().next().unwrap() {
            GpuVal::Tuple(elems) if elems.len() == params.len() =>
                params.iter().cloned().zip(elems).collect(),
            GpuVal::Tuple(elems) => return Err(format!(
                "GPU: function takes {} arguments but was given a {}-tuple",
                params.len(), elems.len())),
            _ => return Err(format!(
                "GPU: function takes {} arguments, so its single argument must be a {}-tuple",
                params.len(), params.len())),
        }
    } else {
        return Err(format!(
            "GPU: function takes {} argument(s) but {} were given",
            params.len(), argvals.len()));
    };

    // Save shadowed entries so the call is lexically scoped (params don't leak).
    let saved: Vec<(String, Option<GpuVal>)> =
        bindings.iter().map(|(k, _)| (k.clone(), scope.get(k).cloned())).collect();
    for (k, v) in bindings { scope.insert(k, v); }
    let result = eval_gpu(body, env, ctx, scope);
    for (k, old) in saved {
        match old { Some(v) => { scope.insert(k, v); } None => { scope.remove(&k); } }
    }
    result
}

/// Evaluate a non-negative integer count argument.
fn eval_count(arg: &Expr, env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<usize, String> {
    match eval_gpu(arg, env, ctx, scope)? {
        GpuVal::Scalar(s) if s >= 0.0 && s.fract() == 0.0 => Ok(s as usize),
        GpuVal::Scalar(s) => Err(format!("GPU: iterate/scan count must be a non-negative integer, got {s}")),
        GpuVal::Buffer { .. } | GpuVal::Host { .. } | GpuVal::Tuple(_) | GpuVal::Fn(_)
        | GpuVal::Complex { .. } | GpuVal::CScalar(..) | GpuVal::Field { .. } =>
            Err("GPU: iterate/scan count must be a scalar".into()),
    }
}

/// `iterate(step, x0, n)` — apply `step` n times, keeping state GPU-resident.
fn gpu_iterate(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() != 3 { return Err("iterate(f, x0, n) expects 3 args".into()); }
    let step = resolve_step(&args[0], env)?;
    let mut state = materialize(ctx, eval_gpu(&args[1], env, ctx, scope)?);
    let n = eval_count(&args[2], env, ctx, scope)?;

    let keys_before: Vec<String> = scope.keys().cloned().collect();
    for _ in 0..n {
        state = apply_step(&step, state, env, ctx, scope)?;
    }
    scope.retain(|k, _| keys_before.contains(k));
    Ok(state)
}

/// `scan(step, x0, n)` — the whole orbit [x0, …, step^n(x0)] stacked.
/// Scalar states → a 1-D tensor [n+1]; 1-D vector states (length d) → [n+1, d].
fn gpu_scan(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() != 3 { return Err("scan(f, x0, n) expects 3 args".into()); }
    let step = resolve_step(&args[0], env)?;
    let x0 = materialize(ctx, eval_gpu(&args[1], env, ctx, scope)?);
    let n = eval_count(&args[2], env, ctx, scope)?;

    let mut frames: Vec<GpuVal> = Vec::with_capacity(n + 1);
    frames.push(x0.clone());
    let mut state = x0;
    let keys_before: Vec<String> = scope.keys().cloned().collect();
    for _ in 0..n {
        state = apply_step(&step, state, env, ctx, scope)?;
        frames.push(state.clone());
    }
    scope.retain(|k, _| keys_before.contains(k));

    stack_frames(ctx, frames)
}

/// Stack a scan orbit into host-side blocks, mirroring the CPU `stack_rows`
/// semantics exactly:
///   • scalar states            → [k]            (k = n+1 frames)
///   • tensor states (shape s)  → [k, ...s]      (time as the leading axis)
///   • a flat tuple of scalars  → [k, arity]     (each state is a row)
///   • a *structured* tuple (any field is a tensor) → a tuple of per-field
///     stacks, each stacked independently (coupled fields stay apart).
/// Each frame is downloaded exactly once; the result is kept on the host so it
/// is not re-uploaded just to be downloaded again when the block returns.
fn stack_frames(ctx: &GpuContext, frames: Vec<GpuVal>) -> Result<GpuVal, String> {
    let k = frames.len();
    match &frames[0] {
        GpuVal::Scalar(_) => {
            let data: Vec<f32> = frames.iter().map(|f| match f {
                GpuVal::Scalar(s) => Ok(*s as f32),
                _ => Err("GPU: scan states must all be scalars".to_string()),
            }).collect::<Result<_, _>>()?;
            let len = data.len();
            Ok(GpuVal::Host { data: Rc::new(data), shape: vec![len], len })
        }
        GpuVal::Buffer { shape, len, .. } => {
            let d = *len;
            let frame_shape = shape.clone();
            let mut data: Vec<f32> = Vec::with_capacity(k * d);
            for f in &frames {
                match f {
                    GpuVal::Buffer { buf, len: l, .. } if *l == d => data.extend(download(ctx, buf, d)?),
                    _ => return Err("GPU: scan states must all have the same shape".into()),
                }
            }
            let mut out_shape = Vec::with_capacity(frame_shape.len() + 1);
            out_shape.push(k);
            out_shape.extend(frame_shape);
            let total = data.len();
            Ok(GpuVal::Host { data: Rc::new(data), shape: out_shape, len: total })
        }
        GpuVal::Tuple(first) => {
            let arity = first.len();
            // Flat numeric tuple (all scalars) → row mode: state i is a row of
            // width `arity`, giving [k, arity] (matches CPU's vector mode).
            if first.iter().all(|x| matches!(x, GpuVal::Scalar(_))) {
                let mut data: Vec<f32> = Vec::with_capacity(k * arity);
                for f in &frames {
                    match f {
                        GpuVal::Tuple(items) if items.len() == arity => {
                            for it in items {
                                match it {
                                    GpuVal::Scalar(s) => data.push(*s as f32),
                                    _ => return Err("GPU: scan states must all be same-arity scalar tuples".into()),
                                }
                            }
                        }
                        _ => return Err(format!("GPU: scan structured states must all be {arity}-tuples")),
                    }
                }
                let total = data.len();
                return Ok(GpuVal::Host { data: Rc::new(data), shape: vec![k, arity], len: total });
            }
            // Structured tuple (some field is a tensor) → stack each field apart.
            let mut columns: Vec<Vec<GpuVal>> = (0..arity).map(|_| Vec::with_capacity(k)).collect();
            for f in frames {
                match f {
                    GpuVal::Tuple(items) if items.len() == arity => {
                        for (j, it) in items.into_iter().enumerate() { columns[j].push(it); }
                    }
                    _ => return Err(format!("GPU: scan structured states must all be {arity}-tuples")),
                }
            }
            let fields = columns.into_iter()
                .map(|c| stack_frames(ctx, c))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GpuVal::Tuple(fields))
        }
        GpuVal::Host { .. } => Err("GPU: scan produced a host value (internal error)".into()),
        GpuVal::Fn(_) => Err("GPU: scan step must return a tensor/scalar state, not a function".into()),
        GpuVal::Complex { .. } | GpuVal::CScalar(..) | GpuVal::Field { .. } =>
            Err("GPU: scan/iterate over complex or field state is not supported".into()),
    }
}

// ───────────────────────────── plumbing ─────────────────────────────

/// Upload an f32 vector as a GPU buffer with the given shape.
/// Matrix multiply C[m,n] = A[m,k] @ B[k,n], one thread per output element. The
/// inner k-loop carries a Neumaier compensation term (§ compensated reductions)
/// so the dot products keep ~f64 accuracy despite f32 storage.
fn run_matmul(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Rc<wgpu::Buffer>, String> {
    let src_wgsl =
        "@group(0) @binding(0) var<storage, read> A: array<f32>;\n\
@group(0) @binding(1) var<storage, read> B: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\
struct Params { m: u32, k: u32, n: u32, pad: u32 };\n\
@group(0) @binding(3) var<uniform> params: Params;\n\
@compute @workgroup_size(16, 16)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y;\n\
    let j = gid.x;\n\
    if (i >= params.m || j >= params.n) { return; }\n\
    var acc: f32 = 0.0;\n\
    var comp: f32 = 0.0;\n\
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {\n\
        let prod = A[i * params.k + kk] * B[kk * params.n + j];\n\
        let t = acc + prod;\n\
        if (abs(acc) >= abs(prod)) { comp = comp + ((acc - t) + prod); }\n\
        else { comp = comp + ((prod - t) + acc); }\n\
        acc = t;\n\
    }\n\
    C[i * params.n + j] = acc + comp;\n\
}\n";
    let pipeline = get_pipeline(ctx, src_wgsl);

    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-matmul-out"),
        size: ((m * n).max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-matmul-params"),
        contents: bytemuck::cast_slice(&[m as u32, k as u32, n as u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-matmul-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-matmul-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-matmul-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let gx = (n as u32).div_ceil(16).max(1);
        let gy = (m as u32).div_ceil(16).max(1);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(Rc::new(out))
}

/// `matmul(A, B)` inside a GPU block: 2D×2D, 2D×1D, 1D×2D, and 1D×1D (dot),
/// mirroring the CPU builtin's shape rules.
fn gpu_matmul(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() != 2 { return Err("matmul(A, B) expects 2 args".into()); }
    let av = eval_gpu(&args[0], env, ctx, scope)?;
    let bv = eval_gpu(&args[1], env, ctx, scope)?;
    let (ab, ash, _) = expect_buffer(ctx, av, "matmul")?;
    let (bb, bsh, _) = expect_buffer(ctx, bv, "matmul")?;
    // Resolve (m, k, n) and the output shape from the operand ranks.
    let (m, k, n, out_shape): (usize, usize, usize, Vec<usize>) = match (ash.as_slice(), bsh.as_slice()) {
        ([ar, ac], [br, bc]) => {
            if ac != br { return Err(format!("matmul: shape mismatch ({ar}×{ac}) @ ({br}×{bc})")); }
            (*ar, *ac, *bc, vec![*ar, *bc])
        }
        ([ar, ac], [bl]) => {
            if ac != bl { return Err(format!("matmul: shape mismatch ({ar}×{ac}) @ ({bl},)")); }
            (*ar, *ac, 1, vec![*ar])
        }
        ([al], [br, bc]) => {
            if al != br { return Err(format!("matmul: shape mismatch ({al},) @ ({br}×{bc})")); }
            (1, *al, *bc, vec![*bc])
        }
        ([al], [bl]) => {
            if al != bl { return Err(format!("matmul: length mismatch ({al} vs {bl})")); }
            (1, *al, 1, vec![])
        }
        _ => return Err("matmul: arguments must be 1D or 2D tensors".into()),
    };
    let out = run_matmul(ctx, &ab, &bb, m, k, n)?;
    let len = m * n;
    // 1D×1D collapses to a scalar, matching the CPU dot product.
    if out_shape.is_empty() {
        let v = download(ctx, &out, 1)?;
        return Ok(GpuVal::Scalar(v[0] as f64));
    }
    Ok(GpuVal::Buffer { buf: out, shape: out_shape, len })
}

// ───────────────────────────── FFT / spectral ─────────────────────────────
//
// A complete GPU spectral path: real input → interleaved complex (vec2<f32>) →
// per-axis radix-2 Stockham FFT (src/gpu/fft.wgsl) → spectral multiply → inverse
// FFT → real part. This backs the `fft`/`ifft` builtins and the spectral
// operators `ops.poisson`, `ops.invlap`, `ops.specgrad`. Conventions match the
// CPU exactly: forward DFT exp(-2πi·kn/N) unnormalised, inverse exp(+2πi·kn/N)
// with a 1/N per-axis scale; wavenumber k = 2π·kfreq / (N·dx).

/// The radix-2 Stockham butterfly pass — one dispatch per stage. Kept in its own
/// file (the heart of the spectral backend) and compiled once via the cache.
const FFT_PASS_WGSL: &str = include_str!("fft.wgsl");

/// real `array<f32>` → interleaved complex `array<vec2<f32>>` (imag = 0).
const R2C_WGSL: &str = "\
@group(0) @binding(0) var<storage, read> inp: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> outp: array<vec2<f32>>;\n\
struct P { len: u32, row: u32 };\n\
@group(0) @binding(2) var<uniform> p: P;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y * p.row + gid.x;\n\
    if (i >= p.len) { return; }\n\
    outp[i] = vec2<f32>(inp[i], 0.0);\n\
}\n";

/// interleaved complex `array<vec2<f32>>` → real part as `array<f32>`.
const C2R_WGSL: &str = "\
@group(0) @binding(0) var<storage, read> inp: array<vec2<f32>>;\n\
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;\n\
struct P { len: u32, row: u32 };\n\
@group(0) @binding(2) var<uniform> p: P;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y * p.row + gid.x;\n\
    if (i >= p.len) { return; }\n\
    outp[i] = inp[i].x;\n\
}\n";

/// Spectral Poisson solve, in place on the complex spectrum: û ← -r̂/k²
/// (k=0 mode → 0, the zero-mean solution). `dims` carries [shape…, strides…];
/// `spc` is the per-axis grid spacing (so anisotropic fields work too).
const POISSON_WGSL: &str = "\
const TWO_PI: f32 = 6.28318530717958647692;\n\
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;\n\
@group(0) @binding(1) var<storage, read> dims: array<u32>;\n\
struct SP { len: u32, row: u32, ndim: u32, ax: u32 };\n\
@group(0) @binding(2) var<uniform> sp: SP;\n\
@group(0) @binding(3) var<storage, read> spc: array<f32>;\n\
fn kfreq(m: u32, n: u32) -> f32 { if (2u * m < n) { return f32(m); } return f32(m) - f32(n); }\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y * sp.row + gid.x;\n\
    if (i >= sp.len) { return; }\n\
    var k2: f32 = 0.0;\n\
    for (var a: u32 = 0u; a < sp.ndim; a = a + 1u) {\n\
        let na = dims[a];\n\
        let sa = dims[sp.ndim + a];\n\
        let idx = (i / sa) % na;\n\
        let kk = kfreq(idx, na) * TWO_PI / (f32(na) * spc[a]);\n\
        k2 = k2 + kk * kk;\n\
    }\n\
    if (k2 == 0.0) { data[i] = vec2<f32>(0.0, 0.0); }\n\
    else { data[i] = data[i] * (-1.0 / k2); }\n\
}\n";

/// Spectral derivative, in place: multiply the spectrum by i·k along axis `ax`.
const SPECGRAD_WGSL: &str = "\
const TWO_PI: f32 = 6.28318530717958647692;\n\
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;\n\
@group(0) @binding(1) var<storage, read> dims: array<u32>;\n\
struct SP { len: u32, row: u32, ndim: u32, ax: u32 };\n\
@group(0) @binding(2) var<uniform> sp: SP;\n\
@group(0) @binding(3) var<storage, read> spc: array<f32>;\n\
fn kfreq(m: u32, n: u32) -> f32 { if (2u * m < n) { return f32(m); } return f32(m) - f32(n); }\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y * sp.row + gid.x;\n\
    if (i >= sp.len) { return; }\n\
    let na = dims[sp.ax];\n\
    let sa = dims[sp.ndim + sp.ax];\n\
    let idx = (i / sa) % na;\n\
    let k = kfreq(idx, na) * TWO_PI / (f32(na) * spc[sp.ax]);\n\
    let c = data[i];\n\
    data[i] = vec2<f32>(-k * c.y, k * c.x);\n\
}\n";

fn is_pow2(n: usize) -> bool { n != 0 && (n & (n - 1)) == 0 }

/// 2-D dispatch grid for `threads` work-items at workgroup size 256, keeping each
/// dimension within the 65535 cap. Returns (groups_x, groups_y, row=gx*256).
fn grid(threads: usize) -> (u32, u32, u32) {
    const WG: u32 = 256;
    const MAX_DIM: u32 = 65535;
    let needed = (threads as u32).div_ceil(WG).max(1);
    let gx = needed.min(MAX_DIM);
    let gy = needed.div_ceil(gx).max(1);
    (gx, gy, gx * WG)
}

/// Allocate a zeroed interleaved-complex buffer holding `total` (re, im) pairs.
fn make_complex_buffer(ctx: &GpuContext, total: usize) -> Rc<wgpu::Buffer> {
    Rc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-fft-complex"),
        size: (total.max(1) * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Run a kernel that reads one buffer and writes another, both bound at 0/1 with a
/// {len,row} uniform at 2 — used by the real↔complex conversions.
fn run_convert(ctx: &GpuContext, src_wgsl: &str, inp: &wgpu::Buffer, out: &wgpu::Buffer, total: usize) {
    let pipeline = get_pipeline(ctx, src_wgsl);
    let (gx, gy, row) = grid(total);
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-convert-params"),
        contents: bytemuck::cast_slice(&[total as u32, row, 0u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-convert-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-convert-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-convert-pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

fn real_to_complex(ctx: &GpuContext, real: &wgpu::Buffer, total: usize) -> Rc<wgpu::Buffer> {
    let cbuf = make_complex_buffer(ctx, total);
    run_convert(ctx, R2C_WGSL, real, &cbuf, total);
    cbuf
}

/// Byte-copy an interleaved-complex buffer into a fresh one. Used when an FFT is
/// applied to an existing complex value, so the in-place ping-pong does not
/// clobber a buffer that may still be referenced elsewhere in the block scope.
fn clone_complex_buffer(ctx: &GpuContext, src: &wgpu::Buffer, total: usize) -> Rc<wgpu::Buffer> {
    let dst = make_complex_buffer(ctx, total);
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-complex-copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &dst, 0, (total.max(1) * 8) as u64);
    ctx.queue.submit(Some(encoder.finish()));
    dst
}

fn complex_real_part(ctx: &GpuContext, cbuf: &wgpu::Buffer, total: usize) -> Rc<wgpu::Buffer> {
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-fft-real"),
        size: (total.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    run_convert(ctx, C2R_WGSL, cbuf, &out, total);
    Rc::new(out)
}

/// One radix-2 Stockham butterfly pass over every line of one axis.
#[allow(clippy::too_many_arguments)]
fn fft_pass(
    ctx: &GpuContext, src: &wgpu::Buffer, dst: &wgpu::Buffer,
    n: usize, ns: usize, stride: usize, half: usize, nlines: usize, sign: f32, scale: f32,
) {
    let pipeline = get_pipeline(ctx, FFT_PASS_WGSL);
    let total_bf = nlines * half;
    let (gx, gy, row) = grid(total_bf);
    // Params layout matches struct in fft.wgsl: 8 u32 then 4 f32 (sign/scale + pad).
    let u: [u32; 12] = [
        n as u32, ns as u32, stride as u32, half as u32,
        nlines as u32, total_bf as u32, row, 0,
        sign.to_bits(), scale.to_bits(), 0, 0,
    ];
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-fft-params"),
        contents: bytemuck::cast_slice(&u),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-fft-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-fft-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-fft-pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

/// In-place n-D FFT over `axes` of an interleaved-complex buffer, ping-ponging
/// against a scratch buffer. Returns whichever buffer holds the final result.
/// Each transformed axis must have a power-of-two length (the common spectral
/// grid; other sizes fall back to the CPU).
fn fft_axes(
    ctx: &GpuContext, input: Rc<wgpu::Buffer>, shape: &[usize], total: usize,
    axes: &[usize], forward: bool,
) -> Result<Rc<wgpu::Buffer>, String> {
    let mut cur = input;
    let mut other = make_complex_buffer(ctx, total);
    let sign = if forward { -1.0f32 } else { 1.0f32 };
    for &a in axes {
        let n = shape[a];
        if n <= 1 { continue; }
        if !is_pow2(n) {
            return Err(format!(
                "GPU FFT: axis {a} has length {n}, which is not a power of two; GPU spectral \
                 operators require power-of-two axis lengths (use the CPU for other sizes)"));
        }
        let stride: usize = shape[a + 1..].iter().product();
        let half = n / 2;
        let nlines = total / n;
        let passes = n.trailing_zeros();
        let mut ns = 1usize;
        for s in 0..passes {
            let last = s == passes - 1;
            let scale = if !forward && last { 1.0 / n as f32 } else { 1.0 };
            fft_pass(ctx, &cur, &other, n, ns, stride, half, nlines, sign, scale);
            std::mem::swap(&mut cur, &mut other);
            ns *= 2;
        }
    }
    Ok(cur)
}

/// Apply a spectral multiply (`"poisson"` or `"specgrad"`) in place on the complex
/// spectrum, computing wavenumbers from the flat index via `meta` (shape+strides).
fn spectral_apply(
    ctx: &GpuContext, cbuf: &wgpu::Buffer, shape: &[usize], total: usize,
    spacing: &[f64], kind: &str, axis: usize,
) {
    let src_wgsl = if kind == "poisson" { POISSON_WGSL } else { SPECGRAD_WGSL };
    let pipeline = get_pipeline(ctx, src_wgsl);
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for a in (0..ndim.saturating_sub(1)).rev() {
        strides[a] = strides[a + 1] * shape[a + 1];
    }
    let mut meta: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
    meta.extend(strides.iter().map(|&s| s as u32));
    let meta_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-spectral-meta"),
        contents: bytemuck::cast_slice(&meta),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let spc: Vec<f32> = spacing.iter().map(|&s| s as f32).collect();
    let spc_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-spectral-spacing"),
        contents: bytemuck::cast_slice(&spc),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let (gx, gy, row) = grid(total);
    // SP: len, row, ndim, ax (u32).
    let u: [u32; 4] = [total as u32, row, ndim as u32, axis as u32];
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-spectral-params"),
        contents: bytemuck::cast_slice(&u),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-spectral-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: spc_buf.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-spectral-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-spectral-pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

/// `fft(T)` / `ifft(T)` inside a GPU block: forward/inverse n-D DFT over all axes
/// of a real tensor, returning a complex tensor. (The result can be the block's
/// value but cannot be fed into further GPU ops — see `GpuVal::Complex`.)
fn gpu_fft(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>, forward: bool) -> Result<GpuVal, String> {
    let name = if forward { "fft" } else { "ifft" };
    if args.len() != 1 {
        return Err(format!("GPU: {name}(T) takes one real tensor and transforms all axes"));
    }
    // Accept a real tensor (uploaded as complex with zero imaginary part) or an
    // already-complex value (e.g. `ifft(fft(T))`) — copied so the in-place
    // transform does not clobber the original.
    let v = materialize(ctx, eval_gpu(&args[0], env, ctx, scope)?);
    let (cbuf, shape) = match v {
        GpuVal::Complex { buf, shape, len } => (clone_complex_buffer(ctx, &buf, len), shape),
        other => {
            let (buf, shape, _len) = expect_buffer(ctx, other, name)?;
            if shape.is_empty() { return Err(format!("GPU: {name}: need a tensor of rank >= 1")); }
            let total: usize = shape.iter().product();
            (real_to_complex(ctx, &buf, total), shape)
        }
    };
    if shape.is_empty() { return Err(format!("GPU: {name}: need a tensor of rank >= 1")); }
    let total: usize = shape.iter().product();
    let axes: Vec<usize> = (0..shape.len()).collect();
    let out = fft_axes(ctx, cbuf, &shape, total, &axes, forward)?;
    Ok(GpuVal::Complex { buf: out, shape, len: total })
}

/// `ops.poisson(rhs, dx)` / `ops.invlap(T, dx)` / `ops.specgrad(T, dx, axis)` on
/// the GPU: real → FFT (all axes) → spectral multiply → inverse FFT → real part.
fn gpu_spectral_op(name: &str, args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    let (kind, axis, t_expr, dx_expr) = match name {
        "poisson" | "invlap" => {
            if args.len() != 2 { return Err(format!("ops.{name}(T, dx) expects 2 args")); }
            ("poisson", 0usize, &args[0], &args[1])
        }
        "specgrad" => {
            if args.len() != 3 {
                return Err("GPU: ops.specgrad requires an explicit axis: ops.specgrad(T, dx, axis)".into());
            }
            let axis = cpu_scalar(&args[2], env)? as usize;
            ("specgrad", axis, &args[0], &args[1])
        }
        _ => unreachable!(),
    };
    let t = eval_gpu(t_expr, env, ctx, scope)?;
    let dx = cpu_scalar(dx_expr, env)?;
    let (buf, shape, _len) = expect_buffer(ctx, t, name)?;
    if shape.is_empty() { return Err(format!("ops.{name}: need a tensor of rank >= 1")); }
    if kind == "specgrad" && axis >= shape.len() {
        return Err(format!("ops.specgrad: axis {axis} out of range for rank-{} tensor", shape.len()));
    }
    let total: usize = shape.iter().product();
    let axes: Vec<usize> = (0..shape.len()).collect();
    let spacing = vec![dx; shape.len()];
    let cbuf = real_to_complex(ctx, &buf, total);
    let spec = fft_axes(ctx, cbuf, &shape, total, &axes, true)?;
    spectral_apply(ctx, &spec, &shape, total, &spacing, kind, axis);
    let back = fft_axes(ctx, spec, &shape, total, &axes, false)?;
    let real = complex_real_part(ctx, &back, total);
    Ok(GpuVal::Buffer { buf: real, shape, len: total })
}

// ───────────────────────────── complex arithmetic ─────────────────────────────
//
// Complex tensors live as interleaved (re, im) `vec2<f32>` buffers (GpuVal::Complex)
// and complex scalars as host-side `CScalar` baked into kernels as vec2 constants.
// Elementwise ops promote a real operand to (x, 0). Everything mirrors the CPU's
// complex semantics (see eval.rs make_complex / binop_tuple / the unary table).

/// WGSL helpers shared by every complex kernel.
const CPLX_PRELUDE: &str = "\
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }\n\
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { let d = b.x*b.x + b.y*b.y; return vec2<f32>((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d); }\n\
fn cexp(a: vec2<f32>) -> vec2<f32> { let m = exp(a.x); return vec2<f32>(m*cos(a.y), m*sin(a.y)); }\n\
fn clog(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.5*log(a.x*a.x + a.y*a.y), atan2(a.y, a.x)); }\n\
fn csqrt(a: vec2<f32>) -> vec2<f32> { let r = sqrt(sqrt(a.x*a.x + a.y*a.y)); let t = 0.5*atan2(a.y, a.x); return vec2<f32>(r*cos(t), r*sin(t)); }\n\
fn cconj(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x, -a.y); }\n\
fn cpow(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return cexp(cmul(b, clog(a))); }\n\
fn csin(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(sin(a.x)*cosh(a.y), cos(a.x)*sinh(a.y)); }\n\
fn ccos(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(cos(a.x)*cosh(a.y), -sin(a.x)*sinh(a.y)); }\n";

fn is_complex_val(v: &GpuVal) -> bool { matches!(v, GpuVal::Complex { .. } | GpuVal::CScalar(..)) }
fn is_scalar_val(v: &GpuVal) -> bool { matches!(v, GpuVal::Scalar(_) | GpuVal::CScalar(..)) }

/// (re, im) of a host-side scalar (real or complex).
fn cscalar_parts(v: &GpuVal) -> (f64, f64) {
    match v {
        GpuVal::Scalar(s) => (*s, 0.0),
        GpuVal::CScalar(r, i) => (*r, *i),
        _ => (0.0, 0.0),
    }
}

/// Pack a complex scalar result, collapsing to a real `Scalar` when imag == 0.
fn make_cscalar(r: f64, i: f64) -> GpuVal {
    if i == 0.0 { GpuVal::Scalar(r) } else { GpuVal::CScalar(r, i) }
}

/// Complex scalar op evaluated on the host.
fn complex_scalar_op(op: &Op, ar: f64, ai: f64, br: f64, bi: f64) -> Result<GpuVal, String> {
    let (r, i) = match op {
        Op::Add => (ar + br, ai + bi),
        Op::Sub => (ar - br, ai - bi),
        Op::Mul => (ar*br - ai*bi, ar*bi + ai*br),
        Op::Div => {
            let d = br*br + bi*bi;
            ((ar*br + ai*bi)/d, (ai*br - ar*bi)/d)
        }
        Op::Pow => {
            // a^b = exp(b * log a)
            let (lr, li) = (0.5*(ar*ar + ai*ai).ln(), ai.atan2(ar));
            let (pr, pi) = (br*lr - bi*li, br*li + bi*lr);
            let m = pr.exp();
            (m*pi.cos(), m*pi.sin())
        }
        _ => return Err("GPU: that operator is not defined on complex values".into()),
    };
    Ok(make_cscalar(r, i))
}

/// The WGSL vec2 expression for a complex binary op (operands already vec2<f32>).
fn complex_binop_expr(op: &Op, a: &str, b: &str) -> Result<String, String> {
    Ok(match op {
        Op::Add => format!("({a} + {b})"),
        Op::Sub => format!("({a} - {b})"),
        Op::Mul => format!("cmul({a}, {b})"),
        Op::Div => format!("cdiv({a}, {b})"),
        Op::Pow => format!("cpow({a}, {b})"),
        _ => return Err("GPU: that operator is not defined on complex values".into()),
    })
}

/// Fetch expression (a vec2<f32>) for one operand, registering any buffer it needs.
fn operand_fetch(v: &GpuVal, bufs: &mut Vec<(Rc<wgpu::Buffer>, bool)>) -> String {
    match v {
        GpuVal::Scalar(s)     => format!("vec2<f32>({}, 0.0)", wgsl_f32(*s)),
        GpuVal::CScalar(r, i) => format!("vec2<f32>({}, {})", wgsl_f32(*r), wgsl_f32(*i)),
        GpuVal::Complex { buf, .. } => {
            let k = bufs.len(); bufs.push((buf.clone(), true)); format!("in{k}[i]")
        }
        GpuVal::Buffer { buf, .. } => {
            let k = bufs.len(); bufs.push((buf.clone(), false)); format!("vec2<f32>(in{k}[i], 0.0)")
        }
        _ => "vec2<f32>(0.0, 0.0)".to_string(),
    }
}

/// Run a complex elementwise kernel: `out[i] = <expr>` (a vec2<f32>), where the
/// `inK` are the registered buffers (complex `vec2<f32>` or real `f32`).
fn run_cmap(ctx: &GpuContext, inputs: &[(&wgpu::Buffer, bool)], len: usize, expr: &str) -> Result<Rc<wgpu::Buffer>, String> {
    let out = make_complex_buffer(ctx, len);
    let n_in = inputs.len();
    let mut decls = String::new();
    for (k, (_, is_c)) in inputs.iter().enumerate() {
        let ty = if *is_c { "vec2<f32>" } else { "f32" };
        decls += &format!("@group(0) @binding({k}) var<storage, read> in{k}: array<{ty}>;\n");
    }
    let (out_b, param_b) = (n_in, n_in + 1);
    let src = format!(
        "{CPLX_PRELUDE}{decls}\
@group(0) @binding({out_b}) var<storage, read_write> out: array<vec2<f32>>;\n\
struct P {{ len: u32, row: u32 }};\n\
@group(0) @binding({param_b}) var<uniform> params: P;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let i = gid.y * params.row + gid.x;\n\
    if (i >= params.len) {{ return; }}\n\
    out[i] = {expr};\n\
}}\n");
    let pipeline = get_pipeline(ctx, &src);
    let (gx, gy, row) = grid(len);
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-cmap-params"),
        contents: bytemuck::cast_slice(&[len as u32, row, 0u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let mut entries: Vec<wgpu::BindGroupEntry> = inputs.iter().enumerate()
        .map(|(k, (b, _))| wgpu::BindGroupEntry { binding: k as u32, resource: b.as_entire_binding() })
        .collect();
    entries.push(wgpu::BindGroupEntry { binding: out_b as u32, resource: out.as_entire_binding() });
    entries.push(wgpu::BindGroupEntry { binding: param_b as u32, resource: params.as_entire_binding() });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-cmap-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu-cmap-encoder") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gpu-cmap-pass"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    if let Some(err) = pollster::block_on(ctx.device.pop_error_scope()) {
        return Err(format!("GPU: {err}"));
    }
    Ok(out)
}

/// Map a complex buffer to a real `f32` buffer: `out[i] = <expr>` with the input
/// available as `a0` (vec2<f32>). Used by re/im/abs/arg and complex reductions.
fn run_c2r_map(ctx: &GpuContext, cbuf: &wgpu::Buffer, len: usize, expr: &str) -> Result<Rc<wgpu::Buffer>, String> {
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-c2r-out"),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let src = format!(
        "{CPLX_PRELUDE}\
@group(0) @binding(0) var<storage, read> in0: array<vec2<f32>>;\n\
@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n\
struct P {{ len: u32, row: u32 }};\n\
@group(0) @binding(2) var<uniform> params: P;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let i = gid.y * params.row + gid.x;\n\
    if (i >= params.len) {{ return; }}\n\
    let a0 = in0[i];\n\
    out[i] = {expr};\n\
}}\n");
    let pipeline = get_pipeline(ctx, &src);
    let (gx, gy, row) = grid(len);
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-c2r-params"),
        contents: bytemuck::cast_slice(&[len as u32, row, 0u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-c2r-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu-c2r-encoder") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gpu-c2r-pass"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    Ok(Rc::new(out))
}

/// Elementwise binary op where at least one operand is complex.
fn cbinop(ctx: &GpuContext, op: &Op, lhs: GpuVal, rhs: GpuVal) -> Result<GpuVal, String> {
    let lhs = materialize(ctx, lhs);
    let rhs = materialize(ctx, rhs);
    // Both host scalars → evaluate on the host.
    if is_scalar_val(&lhs) && is_scalar_val(&rhs) {
        let (ar, ai) = cscalar_parts(&lhs);
        let (br, bi) = cscalar_parts(&rhs);
        return complex_scalar_op(op, ar, ai, br, bi);
    }
    // Resolve the output shape/len from whichever operand(s) are buffers.
    let shape_len = |v: &GpuVal| -> Option<(Vec<usize>, usize)> {
        match v {
            GpuVal::Buffer { shape, len, .. } | GpuVal::Complex { shape, len, .. } => Some((shape.clone(), *len)),
            _ => None,
        }
    };
    let (shape, len) = match (shape_len(&lhs), shape_len(&rhs)) {
        (Some((sa, la)), Some((sb, _))) => {
            if sa != sb {
                return Err(format!("GPU: shape mismatch in complex elementwise op: {sa:?} vs {sb:?}"));
            }
            (sa, la)
        }
        (Some(sl), None) | (None, Some(sl)) => sl,
        // Neither operand is a buffer or a host scalar (e.g. a function value);
        // clean error rather than a panic.
        (None, None) => return Err("GPU: complex arithmetic operand must be a scalar or tensor".into()),
    };
    let mut bufs: Vec<(Rc<wgpu::Buffer>, bool)> = Vec::new();
    let a = operand_fetch(&lhs, &mut bufs);
    let b = operand_fetch(&rhs, &mut bufs);
    let expr = complex_binop_expr(op, &a, &b)?;
    let refs: Vec<(&wgpu::Buffer, bool)> = bufs.iter().map(|(buf, c)| (buf.as_ref(), *c)).collect();
    let out = run_cmap(ctx, &refs, len, &expr)?;
    Ok(GpuVal::Complex { buf: out, shape, len })
}

/// Unary functions on a complex buffer. re/im/abs/arg return a real buffer; the
/// rest return complex. Unsupported names error clearly.
fn complex_unary(ctx: &GpuContext, name: &str, buf: Rc<wgpu::Buffer>, shape: Vec<usize>, len: usize) -> Result<GpuVal, String> {
    // Complex → real projections.
    if let Some(expr) = match name {
        "re"  => Some("a0.x"),
        "im"  => Some("a0.y"),
        "abs" => Some("length(a0)"),
        "arg" => Some("atan2(a0.y, a0.x)"),
        _ => None,
    } {
        let out = run_c2r_map(ctx, &buf, len, expr)?;
        return Ok(GpuVal::Buffer { buf: out, shape, len });
    }
    // Complex → complex.
    let expr = match name {
        "conj" => "cconj(in0[i])",
        "exp"  => "cexp(in0[i])",
        "ln"   => "clog(in0[i])",
        "sqrt" => "csqrt(in0[i])",
        "sin"  => "csin(in0[i])",
        "cos"  => "ccos(in0[i])",
        _ => return Err(format!("GPU: {name} is not supported on complex values")),
    };
    let out = run_cmap(ctx, &[(buf.as_ref(), true)], len, expr)?;
    Ok(GpuVal::Complex { buf: out, shape, len })
}

/// Unary functions on a host complex scalar (mirrors the CPU complex unary table).
fn complex_scalar_unary(name: &str, r: f64, i: f64) -> Result<GpuVal, String> {
    use std::f64::consts::PI;
    Ok(match name {
        "re"   => GpuVal::Scalar(r),
        "im"   => GpuVal::Scalar(i),
        "abs"  => GpuVal::Scalar((r*r + i*i).sqrt()),
        "arg"  => GpuVal::Scalar(if i == 0.0 { if r >= 0.0 { 0.0 } else { PI } } else { i.atan2(r) }),
        "conj" => make_cscalar(r, -i),
        "exp"  => { let m = r.exp(); make_cscalar(m * i.cos(), m * i.sin()) }
        "ln"   => make_cscalar(0.5 * (r*r + i*i).ln(), i.atan2(r)),
        "sqrt" => { let m = (r*r + i*i).sqrt().sqrt(); let t = i.atan2(r) / 2.0; make_cscalar(m * t.cos(), m * t.sin()) }
        "sin"  => make_cscalar(r.sin() * i.cosh(), r.cos() * i.sinh()),
        "cos"  => make_cscalar(r.cos() * i.cosh(), -r.sin() * i.sinh()),
        _ => return Err(format!("GPU: {name} is not supported on complex values")),
    })
}

// ───────────────────────────── fields & forms ─────────────────────────────
//
// A field is a GPU buffer of component-fastest data (data[p*ncomp + c]) plus
// host-side geometry (grid, spacing, bc, metric, degree, variance) — the
// GpuVal::Field mirror of the CPU FieldVal. Every forms/field operator is a
// host-computed per-component recipe (which subset, sign, coefficient, derivative
// axis) dispatched through the existing stencil/map primitives, so the component
// ordering is guaranteed identical to the CPU (we reuse forms::subsets et al.).

use crate::ns::forms::{complement, perm_sign, subset_index, subsets};

fn bc_bools(bc: &[BC]) -> Vec<bool> { bc.iter().map(|&b| b == BC::Neumann).collect() }
fn grid_total(grid: &[usize]) -> usize { grid.iter().product::<usize>().max(1) }

/// Anisotropic Laplacian taps: Σ_a (f[+1] + f[-1] - 2 f[0]) / dx_a².
fn lap_taps_aniso(spacing: &[f64]) -> Vec<Tap> {
    let n = spacing.len();
    let center: f64 = spacing.iter().map(|&d| -2.0 / (d * d)).sum();
    let mut taps: Vec<Tap> = vec![(vec![0i64; n], center)];
    for a in 0..n {
        let inv = 1.0 / (spacing[a] * spacing[a]);
        let mut up = vec![0i64; n]; up[a] = 1; taps.push((up, inv));
        let mut dn = vec![0i64; n]; dn[a] = -1; taps.push((dn, inv));
    }
    taps
}

/// Allocate a real grid buffer (storage + copy-src).
fn make_real_buffer(ctx: &GpuContext, len: usize) -> Rc<wgpu::Buffer> {
    Rc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-field-buf"),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Write a grid buffer into component `c` of an interleaved field buffer:
/// out[p*ncomp + c] = comp[p]. (Inverse of `extract_component`.)
fn scatter_component(ctx: &GpuContext, out: &wgpu::Buffer, comp: &wgpu::Buffer, gt: usize, ncomp: usize, c: usize) {
    let src_wgsl =
        "@group(0) @binding(0) var<storage, read> inp: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;\n\
struct P { len: u32, row: u32, ncomp: u32, comp: u32 };\n\
@group(0) @binding(2) var<uniform> p: P;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.y * p.row + gid.x;\n\
    if (i >= p.len) { return; }\n\
    outp[i * p.ncomp + p.comp] = inp[i];\n\
}\n";
    let pipeline = get_pipeline(ctx, src_wgsl);
    let (gx, gy, row) = grid(gt);
    let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-scatter-params"),
        contents: bytemuck::cast_slice(&[gt as u32, row, ncomp as u32, c as u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-scatter-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: comp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gpu-scatter-encoder") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gpu-scatter-pass"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

/// Split a field buffer into its `ncomp` grid-component buffers.
fn field_components(ctx: &GpuContext, buf: &Rc<wgpu::Buffer>, gt: usize, ncomp: usize) -> Result<Vec<Rc<wgpu::Buffer>>, String> {
    if ncomp == 1 { return Ok(vec![buf.clone()]); }
    (0..ncomp).map(|c| extract_component(ctx, buf, gt, ncomp, c)).collect()
}

/// Pack grid-component buffers back into an interleaved field buffer.
fn assemble_field_buf(ctx: &GpuContext, comps: &[Rc<wgpu::Buffer>], gt: usize, nc_out: usize) -> Rc<wgpu::Buffer> {
    if nc_out == 1 { return comps[0].clone(); }
    let out = make_real_buffer(ctx, gt * nc_out);
    for (c, comp) in comps.iter().enumerate() {
        scatter_component(ctx, &out, comp, gt, nc_out, c);
    }
    out
}

/// acc ← (acc + coeff·term), or coeff·term if acc is empty.
fn accum(ctx: &GpuContext, acc: Option<Rc<wgpu::Buffer>>, term: &wgpu::Buffer, coeff: f64, gt: usize) -> Result<Rc<wgpu::Buffer>, String> {
    match acc {
        None => run_map(ctx, &[term], gt, &format!("{}*in0[i]", wgsl_f32(coeff))),
        Some(a) => run_map(ctx, &[&a, term], gt, &format!("in0[i] + {}*in1[i]", wgsl_f32(coeff))),
    }
}

/// Helper to rebuild a `GpuVal::Field` from geometry refs + new component data.
#[allow(clippy::too_many_arguments)]
fn rebuild_gfield(grid: &[usize], spacing: &[f64], lo: &[f64], bc: &[BC], metric: &[f64],
                  buf: Rc<wgpu::Buffer>, degree: usize, variance: Variance, len: usize) -> GpuVal {
    GpuVal::Field {
        buf, grid: grid.to_vec(), spacing: spacing.to_vec(), lo: lo.to_vec(),
        bc: bc.to_vec(), metric: metric.to_vec(), degree, variance, len,
    }
}

/// Scale a whole field by a constant (used by codiff's ★d★ sign).
fn field_scale(ctx: &GpuContext, f: GpuVal, s: f64) -> Result<GpuVal, String> {
    if let GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, variance, len } = f {
        let out = run_map(ctx, &[&buf], len, &format!("{}*in0[i]", wgsl_f32(s)))?;
        Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, out, degree, variance, len))
    } else { Err("field_scale: not a field".into()) }
}

/// Add two fields of identical geometry/degree.
fn field_add(ctx: &GpuContext, a: GpuVal, b: GpuVal) -> Result<GpuVal, String> {
    match (a, b) {
        (GpuVal::Field { buf: ba, grid, spacing, lo, bc, metric, degree, variance, len },
         GpuVal::Field { buf: bb, len: lb, .. }) => {
            if len != lb { return Err("forms: field add length mismatch".into()); }
            let out = run_map(ctx, &[&ba, &bb], len, "in0[i] + in1[i]")?;
            Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, out, degree, variance, len))
        }
        _ => Err("field_add: not fields".into()),
    }
}

/// Field ⊕ field / field ⊕ scalar elementwise arithmetic (preserves geometry).
fn field_binop(ctx: &GpuContext, op: &Op, lhs: GpuVal, rhs: GpuVal) -> Result<GpuVal, String> {
    match (lhs, rhs) {
        (GpuVal::Field { buf: ba, grid, spacing, lo, bc, metric, degree, variance, len },
         GpuVal::Field { buf: bb, grid: g2, degree: d2, len: lb, .. }) => {
            if grid != g2 || degree != d2 || len != lb {
                return Err("GPU: field op field requires matching grid and degree".into());
            }
            let expr = op_expr(op, "in0[i]", "in1[i]")?;
            let out = run_map(ctx, &[&ba, &bb], len, &expr)?;
            Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, out, degree, variance, len))
        }
        (GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, variance, len }, GpuVal::Scalar(s)) => {
            let expr = op_expr(op, "in0[i]", &wgsl_f32(s))?;
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, out, degree, variance, len))
        }
        (GpuVal::Scalar(s), GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, variance, len }) => {
            let expr = op_expr(op, &wgsl_f32(s), "in0[i]")?;
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, out, degree, variance, len))
        }
        _ => Err("GPU: field arithmetic supports field⊕field (same geometry) and field⊕scalar".into()),
    }
}

/// The field-polymorphic `ops.*` (one field arg; dx/bc come from the field).
fn gpu_field_op(name: &str, field: GpuVal, ctx: &GpuContext) -> Result<GpuVal, String> {
    let GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, variance, len } = field
        else { return Err(format!("ops.{name}: expected a field")); };
    let n = grid.len();
    let gt = grid_total(&grid);
    let ncomp = len / gt;
    let bcs = bc_bools(&bc);
    let geom = |b: Rc<wgpu::Buffer>, deg: usize, l: usize| rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, b, deg, variance, l);
    match name {
        // Componentwise anisotropic 3-point Laplacian (same degree).
        "lap" => {
            let comps = field_components(ctx, &buf, gt, ncomp)?;
            let taps = lap_taps_aniso(&spacing);
            let out: Vec<Rc<wgpu::Buffer>> = comps.iter()
                .map(|c| run_stencil(ctx, c, &grid, gt, &taps, &bcs))
                .collect::<Result<_, _>>()?;
            Ok(geom(assemble_field_buf(ctx, &out, gt, ncomp), degree, len))
        }
        // grad of a 0-form → 1-form (n components, central difference per axis).
        "grad" => {
            if ncomp != 1 { return Err("ops.grad: gradient is defined on a scalar (0-form) field".into()); }
            let out: Vec<Rc<wgpu::Buffer>> = (0..n)
                .map(|a| run_stencil(ctx, &buf, &grid, gt, &grad_taps(n, spacing[a], a), &bcs))
                .collect::<Result<_, _>>()?;
            Ok(geom(assemble_field_buf(ctx, &out, gt, n), 1, gt * n))
        }
        // div of an n-component field → 0-form (Σ_a ∂_a V_a).
        "div" => {
            if ncomp != n { return Err(format!("ops.div: field has {ncomp} components but the grid is {n}-D")); }
            let comps = field_components(ctx, &buf, gt, ncomp)?;
            let mut acc: Option<Rc<wgpu::Buffer>> = None;
            for a in 0..n {
                let d = run_stencil(ctx, &comps[a], &grid, gt, &grad_taps(n, spacing[a], a), &bcs)?;
                acc = Some(accum(ctx, acc, &d, 1.0, gt)?);
            }
            Ok(geom(acc.unwrap(), 0, gt))
        }
        // 2-D scalar curl ∂_0 V_1 − ∂_1 V_0 → 0-form.
        "curl" => {
            if n != 2 || ncomp != 2 { return Err("ops.curl: only the 2-D scalar curl is supported (2-component field on a 2-D grid)".into()); }
            let comps = field_components(ctx, &buf, gt, 2)?;
            let dv1 = run_stencil(ctx, &comps[1], &grid, gt, &grad_taps(2, spacing[0], 0), &bcs)?;
            let dv0 = run_stencil(ctx, &comps[0], &grid, gt, &grad_taps(2, spacing[1], 1), &bcs)?;
            let out = run_map(ctx, &[&dv1, &dv0], gt, "in0[i] - in1[i]")?;
            Ok(geom(out, 0, gt))
        }
        // Spectral solvers (periodic only, scalar field).
        "poisson" | "invlap" | "specgrad" => {
            if bc.iter().any(|&b| b != BC::Periodic) {
                return Err(format!("ops.{name}: spectral operators require a periodic field"));
            }
            if ncomp != 1 { return Err(format!("ops.{name}: expected a scalar (0-form) field")); }
            if grid.iter().any(|&d| d > 1 && !is_pow2(d)) {
                return Err(format!("ops.{name}: GPU spectral operators require power-of-two grid axes"));
            }
            let axes: Vec<usize> = (0..n).collect();
            if name == "specgrad" {
                // 1-form: spectral derivative along each axis.
                let mut out = Vec::with_capacity(n);
                for a in 0..n {
                    let cbuf = real_to_complex(ctx, &buf, gt);
                    let spec = fft_axes(ctx, cbuf, &grid, gt, &axes, true)?;
                    spectral_apply(ctx, &spec, &grid, gt, &spacing, "specgrad", a);
                    let back = fft_axes(ctx, spec, &grid, gt, &axes, false)?;
                    out.push(complex_real_part(ctx, &back, gt));
                }
                Ok(geom(assemble_field_buf(ctx, &out, gt, n), 1, gt * n))
            } else {
                let cbuf = real_to_complex(ctx, &buf, gt);
                let spec = fft_axes(ctx, cbuf, &grid, gt, &axes, true)?;
                spectral_apply(ctx, &spec, &grid, gt, &spacing, "poisson", 0);
                let back = fft_axes(ctx, spec, &grid, gt, &axes, false)?;
                Ok(geom(complex_real_part(ctx, &back, gt), 0, gt))
            }
        }
        _ => Err(format!("GPU: ops.{name} is not a field-polymorphic operator")),
    }
}

/// `forms.<name>(…)` inside a GPU block (exterior calculus on captured fields).
fn forms_op(name: &str, args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    let eval_field = |i: usize, scope: &mut HashMap<String, GpuVal>| -> Result<GpuVal, String> {
        match materialize(ctx, eval_gpu(&args[i], env, ctx, scope)?) {
            f @ GpuVal::Field { .. } => Ok(f),
            _ => Err(format!("forms.{name}: expected a field argument")),
        }
    };
    match name {
        "d"      => { let f = eval_field(0, scope)?; forms_d(ctx, f) }
        "hodge"  => { let f = eval_field(0, scope)?; forms_hodge(ctx, f) }
        "raise"  => { let f = eval_field(0, scope)?; forms_musical(ctx, f, true) }
        "lower"  => { let f = eval_field(0, scope)?; forms_musical(ctx, f, false) }
        "codiff" => { let f = eval_field(0, scope)?; forms_codiff(ctx, f) }
        "laplace"=> { let f = eval_field(0, scope)?; forms_laplace(ctx, f) }
        "wedge"  => {
            if args.len() != 2 { return Err("forms.wedge(a, b) expects 2 args".into()); }
            let a = eval_field(0, scope)?; let b = eval_field(1, scope)?; forms_wedge(ctx, a, b)
        }
        "contract" => {
            if args.len() != 2 { return Err("forms.contract(X, w) expects 2 args".into()); }
            let x = eval_field(0, scope)?; let w = eval_field(1, scope)?; forms_contract(ctx, x, w)
        }
        "form" | "vector" | "field" =>
            Err(format!("GPU: forms.{name} (field construction) must run on the CPU — build the field, then capture it in the block")),
        _ => Err(format!("GPU: forms.{name} is not supported in a GPU block")),
    }
}

// d(f): (dω)_J = Σ_p (-1)^p ∂_{J[p]} ω_{J∖{J[p]}}.
fn forms_d(ctx: &GpuContext, f: GpuVal) -> Result<GpuVal, String> {
    let GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree: k, variance, .. } = f
        else { return Err("forms.d: expected a field".into()); };
    let n = grid.len();
    if k + 1 > n { return Err(format!("forms.d: cannot differentiate a {k}-form on a {n}-D grid")); }
    let in_sub = subsets(n, k);
    let out_sub = subsets(n, k + 1);
    let (nc_in, nc_out) = (in_sub.len(), out_sub.len());
    let gt = grid_total(&grid);
    let in_comps = field_components(ctx, &buf, gt, nc_in)?;
    let bcs = bc_bools(&bc);
    let mut out: Vec<Option<Rc<wgpu::Buffer>>> = vec![None; nc_out];
    for (out_c, jset) in out_sub.iter().enumerate() {
        for (p, &j) in jset.iter().enumerate() {
            let mut iset = jset.clone(); iset.remove(p);
            let in_c = subset_index(&in_sub, &iset);
            let d = run_stencil(ctx, &in_comps[in_c], &grid, gt, &grad_taps(n, spacing[j], j), &bcs)?;
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            out[out_c] = Some(accum(ctx, out[out_c].take(), &d, sign, gt)?);
        }
    }
    let comps: Vec<Rc<wgpu::Buffer>> = out.into_iter().map(|o| o.unwrap()).collect();
    let buf = assemble_field_buf(ctx, &comps, gt, nc_out);
    Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, buf, k + 1, variance, gt * nc_out))
}

// hodge(f): ★(dx^I) = sqrt|det g|·(Π g^{ii})·ε(I,Iᶜ)·dx^{Iᶜ}.
fn forms_hodge(ctx: &GpuContext, f: GpuVal) -> Result<GpuVal, String> {
    let GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree: k, variance, .. } = f
        else { return Err("forms.hodge: expected a field".into()); };
    let n = grid.len();
    let in_sub = subsets(n, k);
    let out_sub = subsets(n, n - k);
    let (nc_in, nc_out) = (in_sub.len(), out_sub.len());
    let gt = grid_total(&grid);
    let sqrt_det: f64 = metric.iter().map(|g| g.abs().sqrt()).product();
    let in_comps = field_components(ctx, &buf, gt, nc_in)?;
    let mut out: Vec<Option<Rc<wgpu::Buffer>>> = vec![None; nc_out];
    for (in_c, iset) in in_sub.iter().enumerate() {
        let ic = complement(iset, n);
        let out_c = subset_index(&out_sub, &ic);
        let inv_g: f64 = iset.iter().map(|&i| 1.0 / metric[i]).product();
        let mut concat = iset.clone(); concat.extend_from_slice(&ic);
        let coeff = sqrt_det * inv_g * perm_sign(&concat) as f64;
        out[out_c] = Some(run_map(ctx, &[&in_comps[in_c]], gt, &format!("{}*in0[i]", wgsl_f32(coeff)))?);
    }
    let comps: Vec<Rc<wgpu::Buffer>> = out.into_iter().map(|o| o.unwrap()).collect();
    let buf = assemble_field_buf(ctx, &comps, gt, nc_out);
    Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, buf, n - k, variance, gt * nc_out))
}

// musical raise (♯, up) / lower (♭): scale component I by Π g^{ii} or Π g_ii.
fn forms_musical(ctx: &GpuContext, f: GpuVal, up: bool) -> Result<GpuVal, String> {
    let GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, .. } = f
        else { return Err("forms.raise/lower: expected a field".into()); };
    let n = grid.len();
    let sub = subsets(n, degree);
    let nc = sub.len();
    let gt = grid_total(&grid);
    let comps = field_components(ctx, &buf, gt, nc)?;
    let mut out = Vec::with_capacity(nc);
    for (c, iset) in sub.iter().enumerate() {
        let scale: f64 = iset.iter().map(|&i| if up { 1.0 / metric[i] } else { metric[i] }).product();
        out.push(run_map(ctx, &[&comps[c]], gt, &format!("{}*in0[i]", wgsl_f32(scale)))?);
    }
    let variance = if up { Variance::Vector } else { Variance::Form };
    let buf = assemble_field_buf(ctx, &out, gt, nc);
    Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, buf, degree, variance, gt * nc))
}

// wedge(a, b): (α∧β)_K = Σ_{I⊔J=K} ε(I,J) α_I β_J (pointwise).
fn forms_wedge(ctx: &GpuContext, a: GpuVal, b: GpuVal) -> Result<GpuVal, String> {
    let GpuVal::Field { buf: ba, grid, spacing, lo, bc, metric, degree: ka, variance, .. } = a
        else { return Err("forms.wedge: expected a field".into()); };
    let GpuVal::Field { buf: bb, grid: gb, spacing: sb, bc: bcb, metric: mb, degree: kb, .. } = b
        else { return Err("forms.wedge: expected a field".into()); };
    if grid != gb || spacing != sb || bc != bcb || metric != mb {
        return Err("forms.wedge: operands must share the same grid geometry".into());
    }
    let n = grid.len();
    if ka + kb > n { return Err(format!("forms.wedge: degree {ka}+{kb} exceeds grid dimension {n}")); }
    let a_sub = subsets(n, ka);
    let b_sub = subsets(n, kb);
    let out_sub = subsets(n, ka + kb);
    let gt = grid_total(&grid);
    let a_comps = field_components(ctx, &ba, gt, a_sub.len())?;
    let b_comps = field_components(ctx, &bb, gt, b_sub.len())?;
    let mut out: Vec<Option<Rc<wgpu::Buffer>>> = vec![None; out_sub.len()];
    for (ia, iset) in a_sub.iter().enumerate() {
        for (ib, jset) in b_sub.iter().enumerate() {
            if iset.iter().any(|x| jset.contains(x)) { continue; }
            let mut concat = iset.clone(); concat.extend_from_slice(jset);
            let eps = perm_sign(&concat) as f64;
            let mut kset = concat.clone(); kset.sort_unstable();
            let out_c = subset_index(&out_sub, &kset);
            let prod = run_map(ctx, &[&a_comps[ia], &b_comps[ib]], gt, "in0[i] * in1[i]")?;
            out[out_c] = Some(accum(ctx, out[out_c].take(), &prod, eps, gt)?);
        }
    }
    let comps: Vec<Rc<wgpu::Buffer>> = out.into_iter().map(|o| o.unwrap()).collect();
    let buf = assemble_field_buf(ctx, &comps, gt, out_sub.len());
    Ok(rebuild_gfield(&grid, &spacing, &lo, &bc, &metric, buf, ka + kb, variance, gt * out_sub.len()))
}

// contract(X, w): (ι_X ω)_J = Σ_{i∉J} sign(i,J) X^i ω_{sorted({i}∪J)}.
fn forms_contract(ctx: &GpuContext, x: GpuVal, w: GpuVal) -> Result<GpuVal, String> {
    let GpuVal::Field { buf: bx, grid, degree: dx, variance: vx, .. } = &x
        else { return Err("forms.contract: expected a field".into()); };
    let GpuVal::Field { buf: bw, grid: gw, spacing, lo, bc, metric, degree: k, variance: vw, .. } = &w
        else { return Err("forms.contract: expected a field".into()); };
    if *dx != 1 || *vx != Variance::Vector {
        return Err("forms.contract: the first argument must be a vector field (degree-1, contravariant)".into());
    }
    if *vw != Variance::Form { return Err("forms.contract: the second argument must be a form (covariant)".into()); }
    let (xsp, xbc, xmet) = match &x { GpuVal::Field { spacing, bc, metric, .. } => (spacing, bc, metric), _ => unreachable!() };
    if grid != gw || xsp != spacing || xbc != bc || xmet != metric {
        return Err("forms.contract: operands must share the same grid geometry".into());
    }
    let n = grid.len();
    let k = *k;
    if k == 0 { return Err("forms.contract: cannot contract a vector into a 0-form".into()); }
    let in_sub = subsets(n, k);
    let out_sub = subsets(n, k - 1);
    let gt = grid_total(grid);
    let xcomps = field_components(ctx, bx, gt, n)?;
    let wcomps = field_components(ctx, bw, gt, in_sub.len())?;
    let mut out: Vec<Option<Rc<wgpu::Buffer>>> = vec![None; out_sub.len()];
    for (out_c, jset) in out_sub.iter().enumerate() {
        for i in 0..n {
            if jset.contains(&i) { continue; }
            let mut iset = jset.clone(); iset.push(i); iset.sort_unstable();
            let p = iset.iter().position(|&v| v == i).unwrap();
            let in_c = subset_index(&in_sub, &iset);
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            let prod = run_map(ctx, &[&xcomps[i], &wcomps[in_c]], gt, "in0[i] * in1[i]")?;
            out[out_c] = Some(accum(ctx, out[out_c].take(), &prod, sign, gt)?);
        }
    }
    let comps: Vec<Rc<wgpu::Buffer>> = out.into_iter().map(|o| o.unwrap()).collect();
    let buf = assemble_field_buf(ctx, &comps, gt, out_sub.len());
    Ok(rebuild_gfield(grid, spacing, lo, bc, metric, buf, k - 1, Variance::Form, gt * out_sub.len()))
}

// codiff δ = (-1)^{n(k+1)+1} ★d★ (k-form → (k-1)-form).
fn forms_codiff(ctx: &GpuContext, f: GpuVal) -> Result<GpuVal, String> {
    let (n, k) = match &f {
        GpuVal::Field { grid, degree, .. } => (grid.len(), *degree),
        _ => return Err("forms.codiff: expected a field".into()),
    };
    if k == 0 { return Err("forms.codiff: the codifferential of a 0-form is 0 (degree -1)".into()); }
    let s1 = forms_hodge(ctx, f)?;
    let s2 = forms_d(ctx, s1)?;
    let s3 = forms_hodge(ctx, s2)?;
    let sign = if (n * (k + 1) + 1) % 2 == 0 { 1.0 } else { -1.0 };
    field_scale(ctx, s3, sign)
}

// laplace Δ = dδ + δd (Laplace–de Rham; 0-form → ordinary ∇²).
fn forms_laplace(ctx: &GpuContext, f: GpuVal) -> Result<GpuVal, String> {
    let k = match &f { GpuVal::Field { degree, .. } => *degree, _ => return Err("forms.laplace: expected a field".into()) };
    let dd = forms_d(ctx, f.clone())?;
    let term_dd = forms_codiff(ctx, dd)?;
    if k == 0 { return Ok(term_dd); }
    let cd = forms_codiff(ctx, f)?;
    let term_cd = forms_d(ctx, cd)?;
    field_add(ctx, term_dd, term_cd)
}

fn upload_f32(ctx: &GpuContext, data: Vec<f32>, shape: Vec<usize>) -> GpuVal {
    let len = data.len();
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-upload-f32"),
        contents: bytemuck::cast_slice(if data.is_empty() { &[0.0f32][..] } else { &data[..] }),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    GpuVal::Buffer { buf: Rc::new(buf), shape, len }
}

/// Upload a CPU tensor to a GPU storage buffer, converting f64 → f32.
fn upload(ctx: &GpuContext, data: &TData, shape: &[usize]) -> GpuVal {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let len = f32_data.len().max(1);
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-upload"),
        contents: bytemuck::cast_slice(&f32_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    GpuVal::Buffer {
        buf: Rc::new(buf),
        shape: shape.to_vec(),
        len: f32_data.len().min(len), // keep true length; len() may be 0
    }
}

/// Upload a CPU complex tensor as an interleaved (re, im) `vec2<f32>` buffer.
fn upload_complex(ctx: &GpuContext, re: &TData, im: &TData, shape: &[usize]) -> GpuVal {
    let len = re.len();
    let mut inter: Vec<f32> = Vec::with_capacity(len.max(1) * 2);
    for (r, i) in re.iter().zip(im.iter()) {
        inter.push(*r as f32);
        inter.push(*i as f32);
    }
    if inter.is_empty() { inter.extend_from_slice(&[0.0, 0.0]); }
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-upload-complex"),
        contents: bytemuck::cast_slice(&inter),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    GpuVal::Complex { buf: Rc::new(buf), shape: shape.to_vec(), len }
}

/// Upload a CPU field, copying its component data to the GPU and its geometry to
/// host-side metadata (the `GpuVal::Field` mirror of `FieldVal`).
fn upload_field(ctx: &GpuContext, f: &FieldVal) -> GpuVal {
    let f32_data: Vec<f32> = f.data.iter().map(|&x| x as f32).collect();
    let len = f32_data.len();
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-upload-field"),
        contents: bytemuck::cast_slice(if f32_data.is_empty() { &[0.0f32][..] } else { &f32_data[..] }),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    GpuVal::Field {
        buf: Rc::new(buf),
        grid: f.grid.clone(), spacing: f.spacing.clone(), lo: f.lo.clone(),
        bc: f.bc.clone(), metric: f.metric.clone(), degree: f.degree, variance: f.variance, len,
    }
}

/// Get a compute pipeline for a WGSL source, compiling and caching on first use.
fn get_pipeline(ctx: &GpuContext, src: &str) -> std::sync::Arc<wgpu::ComputePipeline> {
    if let Some(p) = ctx.pipelines.borrow().get(src) {
        return p.clone();
    }
    // Surface WGSL compilation errors with the offending source: without this the
    // failure only shows up later as an opaque "Pipeline is invalid" panic.
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-shader"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    if let Some(err) = pollster::block_on(ctx.device.pop_error_scope()) {
        eprintln!("GPU shader compile error: {err}\n--- shader source ---\n{src}\n---------------------");
    }
    let pipeline = std::sync::Arc::new(ctx.device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some("gpu-pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        },
    ));
    ctx.pipelines.borrow_mut().insert(src.to_string(), pipeline.clone());
    pipeline
}

/// Run an elementwise map kernel: `out[i] = <expr>` over `len` elements, where
/// `expr` references the input buffers as `in0[i]`, `in1[i]`, ….
fn run_map(
    ctx: &GpuContext,
    inputs: &[&wgpu::Buffer],
    len: usize,
    expr: &str,
) -> Result<Rc<wgpu::Buffer>, String> {
    let device = &ctx.device;
    let out_bytes = (len.max(1) * std::mem::size_of::<f32>()) as u64;

    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Build the shader source for this op.
    let n_in = inputs.len();
    let mut decls = String::new();
    for k in 0..n_in {
        decls += &format!(
            "@group(0) @binding({k}) var<storage, read> in{k}: array<f32>;\n"
        );
    }
    let out_binding = n_in;
    let param_binding = n_in + 1;
    // Spread work across a 2-D workgroup grid: a single dispatch dimension is
    // capped at 65535 groups, which a large tensor (e.g. 500³) blows past. The
    // shader recovers the linear index from a 2-D global id using `row` (the
    // number of threads per grid row = groups_x * workgroup_size).
    let src = format!(
        "{decls}\
@group(0) @binding({out_binding}) var<storage, read_write> out: array<f32>;\n\
struct Params {{ len: u32, row: u32 }};\n\
@group(0) @binding({param_binding}) var<uniform> params: Params;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let i = gid.y * params.row + gid.x;\n\
    if (i >= params.len) {{ return; }}\n\
    out[i] = {expr};\n\
}}\n"
    );

    let pipeline = get_pipeline(ctx, &src);

    // Dispatch grid: keep each dimension within the 65535 cap.
    const WG: u32 = 256;
    const MAX_DIM: u32 = 65535;
    let needed = (len as u32).div_ceil(WG).max(1);
    let groups_x = needed.min(MAX_DIM);
    let groups_y = needed.div_ceil(groups_x).max(1);
    let row = groups_x * WG;

    // Params uniform (padded to 16 bytes for uniform-buffer alignment).
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-params"),
        contents: bytemuck::cast_slice(&[len as u32, row, 0u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Bind group.
    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
    for (k, b) in inputs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: k as u32,
            resource: b.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: out_binding as u32,
        resource: out.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: param_binding as u32,
        resource: params.as_entire_binding(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    // Capture validation/OOM errors as a Result instead of letting wgpu's
    // default handler panic the whole process.
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));

    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        return Err(format!("GPU: {err}"));
    }

    Ok(Rc::new(out))
}

/// Download a GPU value back to a CPU `Val`.
fn to_val(ctx: &GpuContext, v: &GpuVal) -> Result<Val, String> {
    match v {
        GpuVal::Scalar(s) => Ok(Val::Num(*s)),
        GpuVal::Buffer { buf, shape, len } => {
            let data = download(ctx, buf, *len)?;
            Ok(Val::Tensor {
                data: TData::new(data.into_iter().map(|x| x as f64).collect()),
                shape: shape.clone(),
            })
        }
        // Already on the host (e.g. a `scan` spacetime block): convert directly,
        // skipping the upload→download round-trip a `Buffer` would incur.
        GpuVal::Host { data, shape, .. } => Ok(Val::Tensor {
            data: TData::new(data.iter().map(|&x| x as f64).collect()),
            shape: shape.clone(),
        }),
        // Interleaved (re, im) f32 pairs → split into parallel re/im f64 arrays.
        GpuVal::Complex { buf, shape, len } => {
            let inter = download(ctx, buf, *len * 2)?;
            let mut re = Vec::with_capacity(*len);
            let mut im = Vec::with_capacity(*len);
            for c in inter.chunks_exact(2) {
                re.push(c[0] as f64);
                im.push(c[1] as f64);
            }
            // Collapse to a real tensor when every imaginary part is exactly zero,
            // mirroring the CPU's `maybe_real` (fft output keeps tiny non-zero
            // imag and so stays complex, exactly as on the CPU).
            if im.iter().all(|&x| x == 0.0) {
                Ok(Val::Tensor { data: TData::new(re), shape: shape.clone() })
            } else {
                Ok(Val::ComplexTensor {
                    re: TData::new(re),
                    im: TData::new(im),
                    shape: shape.clone(),
                })
            }
        }
        // A complex scalar collapses to a real `Num` when the imaginary part is
        // exactly zero (matches the CPU `make_complex`).
        GpuVal::CScalar(r, i) => Ok(if *i == 0.0 { Val::Num(*r) } else { Val::Complex(*r, *i) }),
        GpuVal::Field { buf, grid, spacing, lo, bc, metric, degree, variance, len } => {
            let data = download(ctx, buf, *len)?;
            Ok(Val::Field(Arc::new(FieldVal {
                data: TData::new(data.into_iter().map(|x| x as f64).collect()),
                grid: grid.clone(), spacing: spacing.clone(), lo: lo.clone(),
                bc: bc.clone(), metric: metric.clone(), degree: *degree, variance: *variance,
            })))
        }
        GpuVal::Tuple(elems) => Ok(Val::Tuple(
            elems.iter().map(|e| to_val(ctx, e)).collect::<Result<Vec<_>, _>>()?,
        )),
        GpuVal::Fn(_) => Err("GPU: a function cannot be returned from a GPU block".into()),
    }
}

/// Copy a storage buffer back to the CPU as `Vec<f32>`.
fn download(ctx: &GpuContext, buf: &wgpu::Buffer, len: usize) -> Result<Vec<f32>, String> {
    let device = &ctx.device;
    let bytes = (len.max(1) * std::mem::size_of::<f32>()) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-download-encoder"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| "GPU: download callback dropped".to_string())?
        .map_err(|e| format!("GPU: buffer map failed: {e:?}"))?;

    let view = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&view)[..len].to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}
