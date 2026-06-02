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
use crate::eval::{Env, TData, Val, fmt_val};
use context::context;
use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

use context::GpuContext;

/// A value living in the GPU evaluation scope: either a CPU-side scalar (passed
/// into shaders as a literal) or a GPU-resident f32 buffer.
#[derive(Clone)]
enum GpuVal {
    Scalar(f64),
    Buffer {
        buf:   Rc<wgpu::Buffer>,
        shape: Vec<usize>,
        len:   usize,
    },
}

/// Entry point, called from the `Expr::GpuBlock` arm in `src/eval.rs`.
pub fn run_gpu_block(body: &Expr, env: &Env) -> Result<Val, String> {
    let ctx_mutex = context()?;
    let ctx = ctx_mutex.lock().map_err(|_| "GPU context poisoned".to_string())?;
    let mut scope: HashMap<String, GpuVal> = HashMap::new();
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

        Expr::Neg(x) => {
            let v = eval_gpu(x, env, ctx, scope)?;
            match v {
                GpuVal::Scalar(s) => Ok(GpuVal::Scalar(-s)),
                GpuVal::Buffer { buf, shape, len } => {
                    let out = run_map(ctx, &[&buf], len, "-in0[i]")?;
                    Ok(GpuVal::Buffer { buf: out, shape, len })
                }
            }
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
                Some(Val::Tensor { data, shape }) => {
                    let gv = upload(ctx, data, shape);
                    scope.insert(name.clone(), gv.clone());
                    Ok(gv)
                }
                Some(other) => Err(format!(
                    "GPU: `{name}`: only scalars and real tensors are allowed in a GPU block; got {}",
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
                    BlockStmt::Def(Def::Func(..)) => {
                        return Err("GPU: function definitions are not supported in a GPU block".into());
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
        GpuVal::Buffer { .. } => Err("GPU: tensor literals must contain scalar elements".to_string()),
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
    let name = match f {
        Expr::Var(n) => n.as_str(),
        _ => return Err("GPU: only named function calls are supported in a GPU block".into()),
    };
    match name {
        "iterate" => gpu_iterate(args, env, ctx, scope),
        "scan"    => gpu_scan(args, env, ctx, scope),

        // Whole-tensor reductions → scalar.
        "sum" | "mean" | "min" | "max" if args.len() == 1 => {
            let v = eval_gpu(&args[0], env, ctx, scope)?;
            reduce_val(ctx, name, v)
        }
        // Binary min/max (elementwise / broadcast).
        "min" | "max" if args.len() == 2 => {
            let a = eval_gpu(&args[0], env, ctx, scope)?;
            let b = eval_gpu(&args[1], env, ctx, scope)?;
            binary_minmax(ctx, name, a, b)
        }

        // Unary math.
        _ if args.len() == 1 && unary_wgsl(name, "x").is_some() => {
            let v = eval_gpu(&args[0], env, ctx, scope)?;
            unary_val(ctx, name, v)
        }

        _ => Err(format!("GPU: function `{name}` not supported in a GPU block")),
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
        other => return Err(format!("GPU: operator {other:?} not supported yet")),
    })
}

/// Evaluate `lhs op rhs` on the GPU (or CPU for scalar/scalar).
fn binop(ctx: &GpuContext, op: &Op, lhs: GpuVal, rhs: GpuVal) -> Result<GpuVal, String> {
    match (lhs, rhs) {
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
        other => return Err(format!("GPU: operator {other:?} not supported yet")),
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
        _ => f64::NAN,
    }
}

fn unary_val(ctx: &GpuContext, name: &str, v: GpuVal) -> Result<GpuVal, String> {
    match v {
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar(unary_cpu(name, s))),
        GpuVal::Buffer { buf, shape, len } => {
            let expr = unary_wgsl(name, "in0[i]").unwrap();
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
        }
    }
}

/// Elementwise `min`/`max` of two operands.
fn binary_minmax(ctx: &GpuContext, name: &str, a: GpuVal, b: GpuVal) -> Result<GpuVal, String> {
    let f = name; // "min" | "max"
    match (a, b) {
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
    }
}

// ───────────────────────────── reductions ─────────────────────────────

/// Reduce a value with `sum`/`mean`/`min`/`max`, returning a scalar.
fn reduce_val(ctx: &GpuContext, name: &str, v: GpuVal) -> Result<GpuVal, String> {
    match v {
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar(s)),
        GpuVal::Buffer { buf, len, .. } => {
            if len == 0 { return Err(format!("GPU: {name} of an empty tensor")); }
            let kind = match name { "sum" | "mean" => "sum", other => other };
            let total = reduce(ctx, buf, len, kind)?;
            Ok(GpuVal::Scalar(if name == "mean" { total / len as f64 } else { total }))
        }
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
    var acc: f32 = {ident};\n\
    var idx: u32 = gid.x;\n\
    loop {{\n\
        if (idx >= params.len) {{ break; }}\n\
        let a = acc; let b = inp[idx]; acc = {comb};\n\
        idx = idx + params.total;\n\
    }}\n\
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

// ───────────────────────── iterate / scan (residency) ─────────────────────────

/// Resolve a step function (named CPU fn or inline lambda) to its single
/// parameter name and body expression.
fn resolve_step<'a>(f: &'a Expr, env: &'a Env) -> Result<(String, Expr), String> {
    match f {
        Expr::Lambda(params, _, body) => {
            if params.len() != 1 {
                return Err(format!("GPU: step function must take 1 argument, got {}", params.len()));
            }
            Ok((params[0].name.clone(), (**body).clone()))
        }
        Expr::Var(name) => match env.vars.get(name) {
            Some(Val::Fn(params, body, _, _, _)) => {
                if params.len() != 1 {
                    return Err(format!("GPU: step function `{name}` must take 1 argument, got {}", params.len()));
                }
                Ok((params[0].clone(), body.clone()))
            }
            Some(other) => Err(format!("GPU: `{name}` is not a function ({})", fmt_val(other))),
            None => Err(format!("GPU: undefined step function `{name}`")),
        },
        _ => Err("GPU: iterate/scan step must be a function or lambda".into()),
    }
}

/// Evaluate a non-negative integer count argument.
fn eval_count(arg: &Expr, env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<usize, String> {
    match eval_gpu(arg, env, ctx, scope)? {
        GpuVal::Scalar(s) if s >= 0.0 && s.fract() == 0.0 => Ok(s as usize),
        GpuVal::Scalar(s) => Err(format!("GPU: iterate/scan count must be a non-negative integer, got {s}")),
        GpuVal::Buffer { .. } => Err("GPU: iterate/scan count must be a scalar".into()),
    }
}

/// `iterate(step, x0, n)` — apply `step` n times, keeping state GPU-resident.
fn gpu_iterate(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() != 3 { return Err("iterate(f, x0, n) expects 3 args".into()); }
    let (param, body) = resolve_step(&args[0], env)?;
    let mut state = eval_gpu(&args[1], env, ctx, scope)?;
    let n = eval_count(&args[2], env, ctx, scope)?;

    let keys_before: Vec<String> = scope.keys().cloned().collect();
    for _ in 0..n {
        scope.insert(param.clone(), state);
        state = eval_gpu(&body, env, ctx, scope)?;
    }
    scope.retain(|k, _| keys_before.contains(k));
    Ok(state)
}

/// `scan(step, x0, n)` — the whole orbit [x0, …, step^n(x0)] stacked.
/// Scalar states → a 1-D tensor [n+1]; 1-D vector states (length d) → [n+1, d].
fn gpu_scan(args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
    if args.len() != 3 { return Err("scan(f, x0, n) expects 3 args".into()); }
    let (param, body) = resolve_step(&args[0], env)?;
    let x0 = eval_gpu(&args[1], env, ctx, scope)?;
    let n = eval_count(&args[2], env, ctx, scope)?;

    let mut frames: Vec<GpuVal> = Vec::with_capacity(n + 1);
    frames.push(x0.clone());
    let mut state = x0;
    let keys_before: Vec<String> = scope.keys().cloned().collect();
    for _ in 0..n {
        scope.insert(param.clone(), state);
        state = eval_gpu(&body, env, ctx, scope)?;
        frames.push(state.clone());
    }
    scope.retain(|k, _| keys_before.contains(k));

    // Assemble frames into a single tensor (uploaded so it flows like any buffer).
    match &frames[0] {
        GpuVal::Scalar(_) => {
            let data: Vec<f32> = frames.iter().map(|f| match f {
                GpuVal::Scalar(s) => Ok(*s as f32),
                GpuVal::Buffer { .. } => Err("GPU: scan states must all be scalars or all tensors".to_string()),
            }).collect::<Result<_, _>>()?;
            Ok(upload_f32(ctx, data, vec![n + 1]))
        }
        GpuVal::Buffer { shape, len, .. } => {
            if shape.len() != 1 {
                return Err("GPU: scan tensor states must be 1-D".into());
            }
            let d = *len;
            let mut data: Vec<f32> = Vec::with_capacity((n + 1) * d);
            for f in &frames {
                match f {
                    GpuVal::Buffer { buf, len: l, .. } if *l == d => data.extend(download(ctx, buf, d)?),
                    _ => return Err("GPU: scan states must all have the same length".into()),
                }
            }
            Ok(upload_f32(ctx, data, vec![n + 1, d]))
        }
    }
}

// ───────────────────────────── plumbing ─────────────────────────────

/// Upload an f32 vector as a GPU buffer with the given shape.
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

/// Get a compute pipeline for a WGSL source, compiling and caching on first use.
fn get_pipeline(ctx: &GpuContext, src: &str) -> std::sync::Arc<wgpu::ComputePipeline> {
    if let Some(p) = ctx.pipelines.borrow().get(src) {
        return p.clone();
    }
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-shader"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
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
