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
    let result = eval_gpu(body, env, &ctx, &mut scope)?;
    to_val(&ctx, &result)
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
            match env.vars.get(name) {
                Some(Val::Num(f)) => Ok(GpuVal::Scalar(*f)),
                Some(Val::Tensor { data, shape }) => Ok(upload(ctx, data, shape)),
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

        other => Err(format!("GPU: unsupported expression in GPU block: {other:?}")),
    }
}

/// The WGSL infix/function form for an arithmetic op, given operand expressions.
fn op_expr(op: &Op, a: &str, b: &str) -> Result<String, String> {
    Ok(match op {
        Op::Add => format!("({a} + {b})"),
        Op::Sub => format!("({a} - {b})"),
        Op::Mul => format!("({a} * {b})"),
        Op::Div => format!("({a} / {b})"),
        Op::Pow => format!("pow({a}, {b})"),
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
        Op::Add => x + y,
        Op::Sub => x - y,
        Op::Mul => x * y,
        Op::Div => x / y,
        Op::Pow => x.powf(y),
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

// ───────────────────────────── plumbing ─────────────────────────────

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
    let src = format!(
        "{decls}\
@group(0) @binding({out_binding}) var<storage, read_write> out: array<f32>;\n\
struct Params {{ len: u32 }};\n\
@group(0) @binding({param_binding}) var<uniform> params: Params;\n\
@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
    let i = gid.x;\n\
    if (i >= params.len) {{ return; }}\n\
    out[i] = {expr};\n\
}}\n"
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu-elementwise"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-elementwise-pipeline"),
        layout: None,
        module: &module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Params uniform.
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu-params"),
        contents: bytemuck::cast_slice(&[len as u32]),
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
        let groups = (len as u32).div_ceil(256).max(1);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));

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
