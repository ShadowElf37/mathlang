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
        other => other,
    }
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
            let v = materialize(ctx, eval_gpu(x, env, ctx, scope)?);
            match v {
                GpuVal::Scalar(s) => Ok(GpuVal::Scalar(-s)),
                GpuVal::Buffer { buf, shape, len } => {
                    let out = run_map(ctx, &[&buf], len, "-in0[i]")?;
                    Ok(GpuVal::Buffer { buf: out, shape, len })
                }
                GpuVal::Host { .. } => unreachable!("materialized above"),
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
        GpuVal::Buffer { .. } | GpuVal::Host { .. } =>
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
    // Namespaced calls, e.g. `ops.lap(T, dx)`.
    if let Expr::Member(base, member) = f {
        if let Expr::Var(ns) = &**base {
            if ns == "ops" {
                return ops_op(member, args, env, ctx, scope);
            }
        }
        return Err(format!("GPU: `{}.{member}` not supported in a GPU block", fmt_member_base(base)));
    }

    let name = match f {
        Expr::Var(n) => n.as_str(),
        _ => return Err("GPU: only named function calls are supported in a GPU block".into()),
    };
    match name {
        "iterate" => gpu_iterate(args, env, ctx, scope),
        "scan"    => gpu_scan(args, env, ctx, scope),

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
        other => return Err(format!("GPU: operator {other:?} not supported yet")),
    })
}

/// Evaluate `lhs op rhs` on the GPU (or CPU for scalar/scalar).
fn binop(ctx: &GpuContext, op: &Op, lhs: GpuVal, rhs: GpuVal) -> Result<GpuVal, String> {
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
        _ => unreachable!("Host values are materialized above"),
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
    match materialize(ctx, v) {
        GpuVal::Scalar(s) => Ok(GpuVal::Scalar(unary_cpu(name, s))),
        GpuVal::Buffer { buf, shape, len } => {
            let expr = unary_wgsl(name, "in0[i]").unwrap();
            let out = run_map(ctx, &[&buf], len, &expr)?;
            Ok(GpuVal::Buffer { buf: out, shape, len })
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
        _ => unreachable!("Host values are materialized above"),
    }
}

// ───────────────────────────── reductions ─────────────────────────────

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

/// `ops.<name>(...)` inside a GPU block.
fn ops_op(name: &str, args: &[Expr], env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<GpuVal, String> {
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
        _ => Err(format!(
            "GPU: ops.{name} not supported in a GPU block (spectral operators need an FFT — not yet on GPU)"
        )),
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
    /// Inline lambda or named user function: bind `param`, evaluate `body`.
    Body { param: String, body: Expr },
    /// A builtin applied each step (e.g. `exp`, `sin`).
    Builtin(String),
}

/// Resolve a step function (named CPU fn, inline lambda, or builtin) for a loop.
fn resolve_step(f: &Expr, env: &Env) -> Result<Step, String> {
    match f {
        Expr::Lambda(params, _, body) => {
            if params.len() != 1 {
                return Err(format!("GPU: step function must take 1 argument, got {}", params.len()));
            }
            Ok(Step::Body { param: params[0].name.clone(), body: (**body).clone() })
        }
        Expr::Var(name) => match env.vars.get(name) {
            Some(Val::Fn(params, body, _, _, _)) => {
                if params.len() != 1 {
                    return Err(format!("GPU: step function `{name}` must take 1 argument, got {}", params.len()));
                }
                Ok(Step::Body { param: params[0].clone(), body: body.clone() })
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
        Step::Body { param, body } => {
            scope.insert(param.clone(), state);
            eval_gpu(body, env, ctx, scope)
        }
        Step::Builtin(name) => unary_val(ctx, name, state),
    }
}

/// Evaluate a non-negative integer count argument.
fn eval_count(arg: &Expr, env: &Env, ctx: &GpuContext, scope: &mut HashMap<String, GpuVal>) -> Result<usize, String> {
    match eval_gpu(arg, env, ctx, scope)? {
        GpuVal::Scalar(s) if s >= 0.0 && s.fract() == 0.0 => Ok(s as usize),
        GpuVal::Scalar(s) => Err(format!("GPU: iterate/scan count must be a non-negative integer, got {s}")),
        GpuVal::Buffer { .. } | GpuVal::Host { .. } => Err("GPU: iterate/scan count must be a scalar".into()),
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

    // Assemble the orbit into one host-side spacetime block. Each frame is
    // downloaded once and concatenated along a new leading axis, giving shape
    // [n+1, ...grid] — exactly what `!animate` wants from a single block, with no
    // per-frame round-trip on the caller's side. We keep the result as a `Host`
    // value so it is *not* re-uploaded just to be downloaded again at block exit.
    match &frames[0] {
        GpuVal::Scalar(_) => {
            let data: Vec<f32> = frames.iter().map(|f| match f {
                GpuVal::Scalar(s) => Ok(*s as f32),
                _ => Err("GPU: scan states must all be scalars or all tensors".to_string()),
            }).collect::<Result<_, _>>()?;
            let len = data.len();
            Ok(GpuVal::Host { data: Rc::new(data), shape: vec![n + 1], len })
        }
        GpuVal::Buffer { shape, len, .. } => {
            let d = *len;
            let frame_shape = shape.clone();
            let mut data: Vec<f32> = Vec::with_capacity((n + 1) * d);
            for f in &frames {
                match f {
                    GpuVal::Buffer { buf, len: l, .. } if *l == d => data.extend(download(ctx, buf, d)?),
                    _ => return Err("GPU: scan states must all have the same shape".into()),
                }
            }
            let mut out_shape = Vec::with_capacity(frame_shape.len() + 1);
            out_shape.push(n + 1);
            out_shape.extend(frame_shape);
            let total = data.len();
            Ok(GpuVal::Host { data: Rc::new(data), shape: out_shape, len: total })
        }
        GpuVal::Host { .. } => Err("GPU: scan produced a host value (internal error)".into()),
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
        // Already on the host (e.g. a `scan` spacetime block): convert directly,
        // skipping the upload→download round-trip a `Buffer` would incur.
        GpuVal::Host { data, shape, .. } => Ok(Val::Tensor {
            data: TData::new(data.iter().map(|&x| x as f64).collect()),
            shape: shape.clone(),
        }),
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
