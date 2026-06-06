//! The single compute path: backend-agnostic tensor handles that stay
//! device-resident, with one dispatch matrix over (backend × precision).
//!
//! Precision is designed in from the start (`Prec`): f32 and f64 are fully wired;
//! df64 (double-single) storage/round-trip works on every backend, while its
//! arithmetic kernels are staged (see `binop`/`unary`). Adding df64 math later
//! means filling in `kernels::df64` and two dispatch arms — no architectural
//! change.

mod kernels;
pub mod target;

pub use target::{Backend, Prec, Target};

use cubecl::prelude::*;
use cubecl::server::Handle;
use std::cell::RefCell;
use std::rc::Rc;

/// A device-resident tensor: a CubeCL handle plus the target it lives on and its
/// logical shape. Cloning is O(1) (the handle is `Rc`-shared); the data is only
/// pulled to the host on display or a host-only op.
#[derive(Clone)]
pub struct TensorVal {
    pub backend: Backend,
    pub prec: Prec,
    pub shape: Vec<usize>,
    pub len: usize,
    handle: Rc<Handle>,
}

impl TensorVal {
    /// Number of axes. Used by shape-aware ops (indexing/reductions) in later phases.
    #[allow(dead_code)]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

// ── per-backend client cache (lazily initialised, thread-local) ──────────────────

#[cfg(feature = "cpu")]
fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
    thread_local! { static C: RefCell<Option<ComputeClient<cubecl::cpu::CpuRuntime>>> = const { RefCell::new(None) }; }
    C.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| cubecl::cpu::CpuRuntime::client(&cubecl::cpu::CpuDevice))
            .clone()
    })
}

#[cfg(feature = "wgpu")]
fn wgpu_client() -> ComputeClient<cubecl::wgpu::WgpuRuntime> {
    thread_local! { static C: RefCell<Option<ComputeClient<cubecl::wgpu::WgpuRuntime>>> = const { RefCell::new(None) }; }
    C.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| cubecl::wgpu::WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::default()))
            .clone()
    })
}

#[cfg(feature = "cuda")]
fn cuda_client() -> ComputeClient<cubecl::cuda::CudaRuntime> {
    thread_local! { static C: RefCell<Option<ComputeClient<cubecl::cuda::CudaRuntime>>> = const { RefCell::new(None) }; }
    C.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| cubecl::cuda::CudaRuntime::client(&Default::default()))
            .clone()
    })
}

#[cfg(feature = "hip")]
fn hip_client() -> ComputeClient<cubecl::hip::HipRuntime> {
    thread_local! { static C: RefCell<Option<ComputeClient<cubecl::hip::HipRuntime>>> = const { RefCell::new(None) }; }
    C.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| cubecl::hip::HipRuntime::client(&Default::default()))
            .clone()
    })
}

/// Create a device buffer from raw bytes on the given backend.
fn create_on(backend: Backend, bytes: &[u8]) -> Result<Handle, String> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(cpu_client().create_from_slice(bytes)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(wgpu_client().create_from_slice(bytes)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(cuda_client().create_from_slice(bytes)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(hip_client().create_from_slice(bytes)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

/// Read a device buffer back to host bytes (copied out of the runtime `Bytes`).
fn read_from(backend: Backend, handle: Handle) -> Result<Vec<u8>, String> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(cpu_client().read_one_unchecked(handle).to_vec()),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(wgpu_client().read_one_unchecked(handle).to_vec()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(cuda_client().read_one_unchecked(handle).to_vec()),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(hip_client().read_one_unchecked(handle).to_vec()),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

// ── upload / download (host f64 ⇄ device, per precision) ─────────────────────────

/// Upload host f64 data as a tensor at `target`'s backend and precision.
pub fn upload(target: Target, host: &[f64], shape: Vec<usize>) -> Result<TensorVal, String> {
    if !target.backend.supports(target.prec) {
        return Err(format!(
            "{} has no native {} (try !prec f32 or df64)",
            target.backend.name(),
            target.prec.name()
        ));
    }
    let bytes: Vec<u8> = match target.prec {
        Prec::F32 => {
            let v: Vec<f32> = host.iter().map(|&x| x as f32).collect();
            f32::as_bytes(&v).to_vec()
        }
        Prec::F64 => f64::as_bytes(host).to_vec(),
        Prec::Df64 => {
            // double-single: hi = round-to-f32(x); lo = round-to-f32(x - hi).
            let mut v: Vec<f32> = Vec::with_capacity(host.len() * 2);
            for &x in host {
                let hi = x as f32;
                let lo = (x - hi as f64) as f32;
                v.push(hi);
                v.push(lo);
            }
            f32::as_bytes(&v).to_vec()
        }
    };
    let handle = create_on(target.backend, &bytes)?;
    let len = host.len();
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape, len, handle: Rc::new(handle) })
}

/// Download a tensor to host f64 (recombining df64 pairs).
pub fn download(tv: &TensorVal) -> Result<Vec<f64>, String> {
    let bytes = read_from(tv.backend, (*tv.handle).clone())?;
    Ok(match tv.prec {
        Prec::F32 => f32::from_bytes(&bytes).iter().map(|&x| x as f64).collect(),
        Prec::F64 => f64::from_bytes(&bytes).to_vec(),
        Prec::Df64 => {
            let pairs = f32::from_bytes(&bytes);
            (0..tv.len).map(|i| pairs[2 * i] as f64 + pairs[2 * i + 1] as f64).collect()
        }
    })
}

// ── elementwise dispatch ─────────────────────────────────────────────────────────

pub use kernels::{OP_ADD, OP_DIV, OP_EQ, OP_GE, OP_GT, OP_LE, OP_LT, OP_MUL, OP_NE, OP_POW, OP_SUB};
pub use kernels::{RED_MAX, RED_MIN, RED_PROD, RED_SUM};
// OP_MIN / OP_MAX kernels exist but the host min/max builtins reduce on the host
// for now; they'll be wired to the device elementwise form in Phase 3.
pub use kernels::{
    UN_ABS, UN_ACOS, UN_ASIN, UN_ATAN, UN_COS, UN_COSH, UN_DEG, UN_EXP, UN_LN, UN_NEG, UN_RAD, UN_SIN,
    UN_SINH, UN_SQRT, UN_TAN, UN_TANH, UN_TRUNC,
};

fn out_shape(a: &TensorVal, b: &TensorVal) -> Vec<usize> {
    if a.len >= b.len { a.shape.clone() } else { b.shape.clone() }
}

/// Elementwise binary op (with scalar/len-1 broadcast). Both operands must already
/// be on `target`'s backend (the interpreter guarantees this).
pub fn binop(target: Target, op: u32, a: &TensorVal, b: &TensorVal) -> Result<TensorVal, String> {
    let out_len = a.len.max(b.len);
    let shape = out_shape(a, b);
    let handle = match target.prec {
        Prec::F32 => dispatch_binop::<f32>(target.backend, op, &a.handle, &b.handle, a.len, b.len, out_len)?,
        Prec::F64 => dispatch_binop::<f64>(target.backend, op, &a.handle, &b.handle, a.len, b.len, out_len)?,
        Prec::Df64 => dispatch_df64_binop(target.backend, op, &a.handle, &b.handle, a.len, b.len, out_len)?,
    };
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape, len: out_len, handle: Rc::new(handle) })
}

/// Elementwise unary op.
pub fn unary(target: Target, op: u32, a: &TensorVal) -> Result<TensorVal, String> {
    let handle = match target.prec {
        Prec::F32 => dispatch_unary::<f32>(target.backend, op, &a.handle, a.len)?,
        Prec::F64 => dispatch_unary::<f64>(target.backend, op, &a.handle, a.len)?,
        Prec::Df64 => dispatch_df64_unary(target.backend, op, &a.handle, a.len)?,
    };
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape: a.shape.clone(), len: a.len, handle: Rc::new(handle) })
}

fn dispatch_binop<E: Float + CubeElement>(
    backend: Backend,
    op: u32,
    a: &Rc<Handle>,
    b: &Rc<Handle>,
    al: usize,
    bl: usize,
    out_len: usize,
) -> Result<Handle, String> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(run_binop::<cubecl::cpu::CpuRuntime, E>(cpu_client(), op, a, b, al, bl, out_len)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(run_binop::<cubecl::wgpu::WgpuRuntime, E>(wgpu_client(), op, a, b, al, bl, out_len)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(run_binop::<cubecl::cuda::CudaRuntime, E>(cuda_client(), op, a, b, al, bl, out_len)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(run_binop::<cubecl::hip::HipRuntime, E>(hip_client(), op, a, b, al, bl, out_len)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

fn run_binop<R: Runtime, E: Float + CubeElement>(
    client: ComputeClient<R>,
    op: u32,
    a: &Rc<Handle>,
    b: &Rc<Handle>,
    al: usize,
    bl: usize,
    out_len: usize,
) -> Handle {
    let out = client.empty(out_len * core::mem::size_of::<E>());
    let grid = out_len.div_ceil(256).max(1) as u32;
    kernels::ew_binop::launch::<E, R>(
        &client,
        CubeCount::Static(grid, 1, 1),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), al) },
        unsafe { ArrayArg::from_raw_parts((**b).clone(), bl) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) },
        op,
    );
    out
}

fn dispatch_unary<E: Float + CubeElement>(
    backend: Backend,
    op: u32,
    a: &Rc<Handle>,
    len: usize,
) -> Result<Handle, String> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(run_unary::<cubecl::cpu::CpuRuntime, E>(cpu_client(), op, a, len)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(run_unary::<cubecl::wgpu::WgpuRuntime, E>(wgpu_client(), op, a, len)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(run_unary::<cubecl::cuda::CudaRuntime, E>(cuda_client(), op, a, len)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(run_unary::<cubecl::hip::HipRuntime, E>(hip_client(), op, a, len)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

fn run_unary<R: Runtime, E: Float + CubeElement>(
    client: ComputeClient<R>,
    op: u32,
    a: &Rc<Handle>,
    len: usize,
) -> Handle {
    let out = client.empty(len * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::ew_unary::launch::<E, R>(
        &client,
        CubeCount::Static(grid, 1, 1),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len) },
        op,
    );
    out
}

// ── df64 (double-single) dispatch ────────────────────────────────────────────────
// df64 buffers hold 2 f32 per logical element, so the kernel array length is `2*len`.

/// Ops the df64 binop kernel implements. Pow needs df64 exp/ln (staged).
fn df64_binop_supported(op: u32) -> bool {
    matches!(
        op,
        OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_LT | OP_GT | OP_LE | OP_GE | OP_EQ | OP_NE
    )
}

/// df64's error-free transforms require IEEE-strict, non-reassociating float
/// evaluation. The LLVM-based backends (cpu/cuda/hip) provide that. The wgpu path
/// enables fp-fast-math and Metal reassociates regardless, which silently collapses
/// the low-order term to ~f32 — so we refuse rather than return a wrong answer.
pub fn df64_reliable(backend: Backend) -> bool {
    !matches!(backend, Backend::Wgpu)
}

const DF64_WGPU_MSG: &str = "df64 arithmetic is unreliable on wgpu/Metal: the driver's \
fast-math reassociates and collapses the error term to ~f32. Use a cpu/cuda/hip backend \
for df64, or !prec f32 here. (df64 storage/round-trip still works on wgpu.)";

fn dispatch_df64_binop(
    backend: Backend,
    op: u32,
    a: &Rc<Handle>,
    b: &Rc<Handle>,
    al: usize,
    bl: usize,
    out_len: usize,
) -> Result<Handle, String> {
    if !df64_reliable(backend) {
        return Err(DF64_WGPU_MSG.into());
    }
    if !df64_binop_supported(op) {
        return Err("df64 pow is staged (needs df64 exp/ln); use !prec f64/f32, or `x*x`".into());
    }
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(run_df64_binop::<cubecl::cpu::CpuRuntime>(cpu_client(), op, a, b, al, bl, out_len)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(run_df64_binop::<cubecl::wgpu::WgpuRuntime>(wgpu_client(), op, a, b, al, bl, out_len)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(run_df64_binop::<cubecl::cuda::CudaRuntime>(cuda_client(), op, a, b, al, bl, out_len)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(run_df64_binop::<cubecl::hip::HipRuntime>(hip_client(), op, a, b, al, bl, out_len)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

fn run_df64_binop<R: Runtime>(
    client: ComputeClient<R>,
    op: u32,
    a: &Rc<Handle>,
    b: &Rc<Handle>,
    al: usize,
    bl: usize,
    out_len: usize,
) -> Handle {
    let out = client.empty(out_len * 2 * core::mem::size_of::<f32>());
    let grid = out_len.div_ceil(256).max(1) as u32;
    kernels::df64_binop::launch::<R>(
        &client,
        CubeCount::Static(grid, 1, 1),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), al * 2) },
        unsafe { ArrayArg::from_raw_parts((**b).clone(), bl * 2) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), out_len * 2) },
        op,
    );
    out
}

fn dispatch_df64_unary(backend: Backend, op: u32, a: &Rc<Handle>, len: usize) -> Result<Handle, String> {
    if !df64_reliable(backend) {
        return Err(DF64_WGPU_MSG.into());
    }
    if !matches!(op, UN_NEG | UN_ABS) {
        return Err("df64 transcendentals (exp/ln/sin/…) are staged; use !prec f64/f32".into());
    }
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(run_df64_unary::<cubecl::cpu::CpuRuntime>(cpu_client(), op, a, len)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(run_df64_unary::<cubecl::wgpu::WgpuRuntime>(wgpu_client(), op, a, len)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(run_df64_unary::<cubecl::cuda::CudaRuntime>(cuda_client(), op, a, len)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(run_df64_unary::<cubecl::hip::HipRuntime>(hip_client(), op, a, len)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

fn run_df64_unary<R: Runtime>(client: ComputeClient<R>, op: u32, a: &Rc<Handle>, len: usize) -> Handle {
    let out = client.empty(len * 2 * core::mem::size_of::<f32>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::df64_unary::launch::<R>(
        &client,
        CubeCount::Static(grid, 1, 1),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len * 2) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len * 2) },
        op,
    );
    out
}

// ── matmul ───────────────────────────────────────────────────────────────────────
// `a` m×k, `b` k×n → `out` m×n (row-major). The interpreter computes m/k/n and
// reshapes vector cases (mat·vec, vec·mat, dot) before calling.
pub fn matmul(target: Target, a: &TensorVal, b: &TensorVal, m: usize, k: usize, n: usize) -> Result<TensorVal, String> {
    let out_len = m * n;
    let handle = match target.prec {
        Prec::F32 => dispatch_matmul::<f32>(target.backend, &a.handle, a.len, &b.handle, b.len, k, n, out_len)?,
        Prec::F64 => dispatch_matmul::<f64>(target.backend, &a.handle, a.len, &b.handle, b.len, k, n, out_len)?,
        Prec::Df64 => return Err("df64 matmul is staged; use !prec f64/f32 for matmul".into()),
    };
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape: vec![m, n], len: out_len, handle: Rc::new(handle) })
}

fn dispatch_matmul<E: Float + CubeElement>(
    backend: Backend,
    a: &Rc<Handle>,
    al: usize,
    b: &Rc<Handle>,
    bl: usize,
    k: usize,
    n: usize,
    out_len: usize,
) -> Result<Handle, String> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(run_matmul::<cubecl::cpu::CpuRuntime, E>(cpu_client(), a, al, b, bl, k, n, out_len)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(run_matmul::<cubecl::wgpu::WgpuRuntime, E>(wgpu_client(), a, al, b, bl, k, n, out_len)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(run_matmul::<cubecl::cuda::CudaRuntime, E>(cuda_client(), a, al, b, bl, k, n, out_len)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(run_matmul::<cubecl::hip::HipRuntime, E>(hip_client(), a, al, b, bl, k, n, out_len)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_matmul<R: Runtime, E: Float + CubeElement>(
    client: ComputeClient<R>,
    a: &Rc<Handle>,
    al: usize,
    b: &Rc<Handle>,
    bl: usize,
    k: usize,
    n: usize,
    out_len: usize,
) -> Handle {
    let out = client.empty(out_len * core::mem::size_of::<E>());
    let grid = out_len.div_ceil(256).max(1) as u32;
    kernels::matmul_kernel::launch::<E, R>(
        &client,
        CubeCount::Static(grid, 1, 1),
        CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), al) },
        unsafe { ArrayArg::from_raw_parts((**b).clone(), bl) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) },
        k,
        n,
    );
    out
}

// ── whole-tensor reduction → host scalar ─────────────────────────────────────────

const REDUCE_THREADS: usize = 256;

/// Reduce a tensor to a single f64. f32/f64 reduce on device (256 partials + host
/// combine, Neumaier sum); df64 falls back to host (download recombines the pairs).
pub fn reduce(target: Target, op: u32, a: &TensorVal) -> Result<f64, String> {
    if a.len == 0 {
        return Ok(host_identity(op));
    }
    if target.prec == Prec::Df64 {
        let host = download(a)?;
        return Ok(host_combine(op, &host));
    }
    let nt = REDUCE_THREADS.min(a.len.max(1));
    let partials_handle = match target.prec {
        Prec::F32 => dispatch_reduce::<f32>(target.backend, op, &a.handle, a.len, nt)?,
        Prec::F64 => dispatch_reduce::<f64>(target.backend, op, &a.handle, a.len, nt)?,
        Prec::Df64 => unreachable!(),
    };
    let bytes = read_from(target.backend, partials_handle)?;
    let partials: Vec<f64> = match target.prec {
        Prec::F32 => f32::from_bytes(&bytes)[..nt].iter().map(|&x| x as f64).collect(),
        Prec::F64 => f64::from_bytes(&bytes)[..nt].to_vec(),
        Prec::Df64 => unreachable!(),
    };
    Ok(host_combine(op, &partials))
}

fn dispatch_reduce<E: Float + CubeElement>(
    backend: Backend,
    op: u32,
    a: &Rc<Handle>,
    len: usize,
    nt: usize,
) -> Result<Handle, String> {
    match backend {
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(run_reduce::<cubecl::cpu::CpuRuntime, E>(cpu_client(), op, a, len, nt)),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(run_reduce::<cubecl::wgpu::WgpuRuntime, E>(wgpu_client(), op, a, len, nt)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(run_reduce::<cubecl::cuda::CudaRuntime, E>(cuda_client(), op, a, len, nt)),
        #[cfg(feature = "hip")]
        Backend::Hip => Ok(run_reduce::<cubecl::hip::HipRuntime, E>(hip_client(), op, a, len, nt)),
        #[allow(unreachable_patterns)]
        _ => Err(format!("backend {} is not compiled in", backend.name())),
    }
}

fn run_reduce<R: Runtime, E: Float + CubeElement>(
    client: ComputeClient<R>,
    op: u32,
    a: &Rc<Handle>,
    len: usize,
    nt: usize,
) -> Handle {
    let partials = client.empty(nt * core::mem::size_of::<E>());
    kernels::reduce_kernel::launch::<E, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(nt as u32),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len) },
        unsafe { ArrayArg::from_raw_parts(partials.clone(), nt) },
        op,
    );
    partials
}

// ── complex tensors (interleaved [re, im], f32/f64) ──────────────────────────────

pub use kernels::{CR_ABS, CR_ARG, CR_IM, CR_RE, CU_CONJ, CU_COS, CU_EXP, CU_LN, CU_NEG, CU_SIN, CU_SQRT};

/// A device-resident complex tensor: `2*len` interleaved (re, im) values of the
/// element type. Same backend/precision model as `TensorVal` (df64 not supported
/// for complex — use f32/f64).
#[derive(Clone)]
pub struct CTensor {
    pub backend: Backend,
    pub prec: Prec,
    pub shape: Vec<usize>,
    pub len: usize,
    handle: Rc<Handle>,
}

fn no_complex_df64() -> String {
    "complex tensors are f32/f64 only (df64 complex not supported); use !prec f32 or f64".into()
}

/// Upload host (re, im) as a complex tensor at `target`.
pub fn upload_complex(target: Target, re: &[f64], im: &[f64], shape: Vec<usize>) -> Result<CTensor, String> {
    if target.prec == Prec::Df64 {
        return Err(no_complex_df64());
    }
    let n = re.len();
    let bytes = match target.prec {
        Prec::F32 => {
            let mut v: Vec<f32> = Vec::with_capacity(2 * n);
            for i in 0..n { v.push(re[i] as f32); v.push(im[i] as f32); }
            f32::as_bytes(&v).to_vec()
        }
        Prec::F64 => {
            let mut v: Vec<f64> = Vec::with_capacity(2 * n);
            for i in 0..n { v.push(re[i]); v.push(im[i]); }
            f64::as_bytes(&v).to_vec()
        }
        Prec::Df64 => unreachable!(),
    };
    let handle = create_on(target.backend, &bytes)?;
    Ok(CTensor { backend: target.backend, prec: target.prec, shape, len: n, handle: Rc::new(handle) })
}

/// Download a complex tensor to host (re, im) f64 vectors.
pub fn download_complex(ct: &CTensor) -> Result<(Vec<f64>, Vec<f64>), String> {
    let bytes = read_from(ct.backend, (*ct.handle).clone())?;
    let flat: Vec<f64> = match ct.prec {
        Prec::F32 => f32::from_bytes(&bytes).iter().map(|&x| x as f64).collect(),
        Prec::F64 => f64::from_bytes(&bytes).to_vec(),
        Prec::Df64 => return Err(no_complex_df64()),
    };
    let mut re = Vec::with_capacity(ct.len);
    let mut im = Vec::with_capacity(ct.len);
    for i in 0..ct.len { re.push(flat[2 * i]); im.push(flat[2 * i + 1]); }
    Ok((re, im))
}

/// Promote a real tensor to a complex tensor (im = 0) on the device.
pub fn promote_real(target: Target, t: &TensorVal) -> Result<CTensor, String> {
    if target.prec == Prec::Df64 {
        return Err(no_complex_df64());
    }
    let handle = match target.prec {
        Prec::F32 => cdispatch_r2c::<f32>(target.backend, &t.handle, t.len)?,
        Prec::F64 => cdispatch_r2c::<f64>(target.backend, &t.handle, t.len)?,
        Prec::Df64 => unreachable!(),
    };
    Ok(CTensor { backend: target.backend, prec: target.prec, shape: t.shape.clone(), len: t.len, handle: Rc::new(handle) })
}

/// Ensure a complex tensor lives on `target` (re-materialise if backend/prec differ).
pub fn ensure_complex_on(ct: CTensor, target: Target) -> Result<CTensor, String> {
    if ct.backend == target.backend && ct.prec == target.prec {
        Ok(ct)
    } else {
        let (re, im) = download_complex(&ct)?;
        upload_complex(target, &re, &im, ct.shape.clone())
    }
}

pub fn cbinop(target: Target, op: u32, a: &CTensor, b: &CTensor) -> Result<CTensor, String> {
    let out_len = a.len.max(b.len);
    let shape = if a.len >= b.len { a.shape.clone() } else { b.shape.clone() };
    let handle = match target.prec {
        Prec::F32 => cdispatch_binop::<f32>(target.backend, op, &a.handle, a.len, &b.handle, b.len, out_len)?,
        Prec::F64 => cdispatch_binop::<f64>(target.backend, op, &a.handle, a.len, &b.handle, b.len, out_len)?,
        Prec::Df64 => return Err(no_complex_df64()),
    };
    Ok(CTensor { backend: target.backend, prec: target.prec, shape, len: out_len, handle: Rc::new(handle) })
}

/// complex → complex unary (neg/conj/exp/ln/sqrt/sin/cos).
pub fn cunary_c2c(target: Target, op: u32, a: &CTensor) -> Result<CTensor, String> {
    let handle = match target.prec {
        Prec::F32 => cdispatch_c2c::<f32>(target.backend, op, &a.handle, a.len)?,
        Prec::F64 => cdispatch_c2c::<f64>(target.backend, op, &a.handle, a.len)?,
        Prec::Df64 => return Err(no_complex_df64()),
    };
    Ok(CTensor { backend: target.backend, prec: target.prec, shape: a.shape.clone(), len: a.len, handle: Rc::new(handle) })
}

/// complex → real unary (re/im/abs/arg).
pub fn cunary_c2r(target: Target, op: u32, a: &CTensor) -> Result<TensorVal, String> {
    let handle = match target.prec {
        Prec::F32 => cdispatch_c2r::<f32>(target.backend, op, &a.handle, a.len)?,
        Prec::F64 => cdispatch_c2r::<f64>(target.backend, op, &a.handle, a.len)?,
        Prec::Df64 => return Err(no_complex_df64()),
    };
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape: a.shape.clone(), len: a.len, handle: Rc::new(handle) })
}

/// Complex whole-tensor sum (host: download + accumulate). mean = sum/n on the caller.
pub fn creduce_sum(ct: &CTensor) -> Result<(f64, f64), String> {
    let (re, im) = download_complex(ct)?;
    Ok((re.iter().sum(), im.iter().sum()))
}

// per-op complex dispatch (backend → run::<R, E>); E is f32/f64 (df64 excluded above).
// The per-backend match is inlined (a closure can't be generic over the runtime).
macro_rules! cdispatch {
    ($backend:expr, $run:ident :: <$e:ty> ( $($arg:expr),* )) => {
        match $backend {
            #[cfg(feature = "cpu")]
            Backend::Cpu => Ok($run::<cubecl::cpu::CpuRuntime, $e>(cpu_client(), $($arg),*)),
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => Ok($run::<cubecl::wgpu::WgpuRuntime, $e>(wgpu_client(), $($arg),*)),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Ok($run::<cubecl::cuda::CudaRuntime, $e>(cuda_client(), $($arg),*)),
            #[cfg(feature = "hip")]
            Backend::Hip => Ok($run::<cubecl::hip::HipRuntime, $e>(hip_client(), $($arg),*)),
            #[allow(unreachable_patterns)]
            _ => Err(format!("backend {} is not compiled in", $backend.name())),
        }
    };
}

fn cdispatch_r2c<E: Float + CubeElement>(backend: Backend, x: &Rc<Handle>, len: usize) -> Result<Handle, String> {
    cdispatch!(backend, run_r2c::<E>(x, len))
}
fn cdispatch_binop<E: Float + CubeElement>(backend: Backend, op: u32, a: &Rc<Handle>, al: usize, b: &Rc<Handle>, bl: usize, out_len: usize) -> Result<Handle, String> {
    cdispatch!(backend, run_cbinop::<E>(op, a, al, b, bl, out_len))
}
fn cdispatch_c2c<E: Float + CubeElement>(backend: Backend, op: u32, a: &Rc<Handle>, len: usize) -> Result<Handle, String> {
    cdispatch!(backend, run_c2c::<E>(op, a, len))
}
fn cdispatch_c2r<E: Float + CubeElement>(backend: Backend, op: u32, a: &Rc<Handle>, len: usize) -> Result<Handle, String> {
    cdispatch!(backend, run_c2r::<E>(op, a, len))
}

fn run_r2c<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, x: &Rc<Handle>, len: usize) -> Handle {
    let out = client.empty(len * 2 * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::real_to_complex::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**x).clone(), len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len * 2) });
    out
}
fn run_cbinop<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, op: u32, a: &Rc<Handle>, al: usize, b: &Rc<Handle>, bl: usize, out_len: usize) -> Handle {
    let out = client.empty(out_len * 2 * core::mem::size_of::<E>());
    let grid = out_len.div_ceil(256).max(1) as u32;
    kernels::cbinop::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), al * 2) },
        unsafe { ArrayArg::from_raw_parts((**b).clone(), bl * 2) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), out_len * 2) }, op);
    out
}
fn run_c2c<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, op: u32, a: &Rc<Handle>, len: usize) -> Handle {
    let out = client.empty(len * 2 * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::cunary_c2c::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len * 2) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len * 2) }, op);
    out
}
fn run_c2r<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, op: u32, a: &Rc<Handle>, len: usize) -> Handle {
    let out = client.empty(len * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::cunary_c2r::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len * 2) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len) }, op);
    out
}

// ── stencils (roll / shift / ops.lap / ops.grad), 1-D & 2-D, on device ───────────

/// Model a shape as r×c (1-D [N] → (N, 1)). Stencils support rank 1 and 2.
fn rc_of(shape: &[usize]) -> Result<(usize, usize), String> {
    match shape {
        [n] => Ok((*n, 1)),
        [r, c] => Ok((*r, *c)),
        _ => Err("stencils support 1-D and 2-D tensors only".into()),
    }
}

fn stencil_unsupported_df64() -> String {
    "stencils are f32/f64 only (df64 stencils staged); use !prec f32 or f64".into()
}

/// Periodic roll along `axis` by `n` (n may be negative).
pub fn roll(target: Target, a: &TensorVal, n: i64, axis: usize) -> Result<TensorVal, String> {
    if target.prec == Prec::Df64 {
        return Err(stencil_unsupported_df64());
    }
    let (r, c) = rc_of(&a.shape)?;
    let (add_r, add_c) = roll_offsets(n, axis, r, c)?;
    let handle = match target.prec {
        Prec::F32 => cdispatch!(target.backend, run_roll::<f32>(&a.handle, a.len, r, c, add_r, add_c)),
        Prec::F64 => cdispatch!(target.backend, run_roll::<f64>(&a.handle, a.len, r, c, add_r, add_c)),
        Prec::Df64 => unreachable!(),
    }?;
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape: a.shape.clone(), len: a.len, handle: Rc::new(handle) })
}

fn roll_offsets(n: i64, axis: usize, r: usize, c: usize) -> Result<(usize, usize), String> {
    let off = |len: usize| -> usize {
        let l = len as i64;
        let nm = ((n % l) + l) % l; // roll amount in 0..l
        (((l - nm) % l) as usize).min(len.saturating_sub(1).max(0))
    };
    match axis {
        0 => Ok((off(r), 0)),
        1 if c > 1 => Ok((0, off(c))),
        _ => Err(format!("roll: axis {axis} out of range for this tensor")),
    }
}

/// Edge-clamped (Neumann) shift along `axis` by `n`.
pub fn shift(target: Target, a: &TensorVal, n: i64, axis: usize) -> Result<TensorVal, String> {
    if target.prec == Prec::Df64 {
        return Err(stencil_unsupported_df64());
    }
    let (r, c) = rc_of(&a.shape)?;
    let (nr, nc) = match axis {
        0 => (n as i32, 0i32),
        1 if c > 1 => (0i32, n as i32),
        _ => return Err(format!("shift: axis {axis} out of range for this tensor")),
    };
    let handle = match target.prec {
        Prec::F32 => cdispatch!(target.backend, run_shift::<f32>(&a.handle, a.len, r, c, nr, nc)),
        Prec::F64 => cdispatch!(target.backend, run_shift::<f64>(&a.handle, a.len, r, c, nr, nc)),
        Prec::Df64 => unreachable!(),
    }?;
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape: a.shape.clone(), len: a.len, handle: Rc::new(handle) })
}

/// Laplacian (periodic if `periodic`, else Neumann). `dx` is the grid spacing.
pub fn lap(target: Target, a: &TensorVal, dx: f64, periodic: u32) -> Result<TensorVal, String> {
    stencil_scaled(target, a, 1.0 / (dx * dx), periodic, StencilKind::Lap, 0)
}

/// Central-difference gradient along `axis`. `dx` is the grid spacing.
pub fn grad(target: Target, a: &TensorVal, dx: f64, axis: usize, periodic: u32) -> Result<TensorVal, String> {
    let (_, c) = rc_of(&a.shape)?;
    if axis > 1 || (axis == 1 && c == 1) {
        return Err(format!("grad: axis {axis} out of range for this tensor"));
    }
    stencil_scaled(target, a, 1.0 / (2.0 * dx), periodic, StencilKind::Grad, axis as u32)
}

enum StencilKind {
    Lap,
    Grad,
}

fn stencil_scaled(target: Target, a: &TensorVal, scale_val: f64, periodic: u32, kind: StencilKind, axis: u32) -> Result<TensorVal, String> {
    if target.prec == Prec::Df64 {
        return Err(stencil_unsupported_df64());
    }
    let (r, c) = rc_of(&a.shape)?;
    let scale = upload(target, &[scale_val], vec![1])?;
    let handle = match (&kind, target.prec) {
        (StencilKind::Lap, Prec::F32) => cdispatch!(target.backend, run_lap::<f32>(&a.handle, &scale.handle, a.len, r, c, periodic)),
        (StencilKind::Lap, Prec::F64) => cdispatch!(target.backend, run_lap::<f64>(&a.handle, &scale.handle, a.len, r, c, periodic)),
        (StencilKind::Grad, Prec::F32) => cdispatch!(target.backend, run_grad::<f32>(&a.handle, &scale.handle, a.len, r, c, axis, periodic)),
        (StencilKind::Grad, Prec::F64) => cdispatch!(target.backend, run_grad::<f64>(&a.handle, &scale.handle, a.len, r, c, axis, periodic)),
        (_, Prec::Df64) => unreachable!(),
    }?;
    Ok(TensorVal { backend: target.backend, prec: target.prec, shape: a.shape.clone(), len: a.len, handle: Rc::new(handle) })
}

fn run_roll<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, a: &Rc<Handle>, len: usize, r: usize, c: usize, add_r: usize, add_c: usize) -> Handle {
    let out = client.empty(len * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::roll2d::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len) }, r, c, add_r, add_c);
    out
}
fn run_shift<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, a: &Rc<Handle>, len: usize, r: usize, c: usize, nr: i32, nc: i32) -> Handle {
    let out = client.empty(len * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::shift2d::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**a).clone(), len) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len) }, r, c, nr, nc);
    out
}
fn run_lap<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, u: &Rc<Handle>, scale: &Rc<Handle>, len: usize, r: usize, c: usize, periodic: u32) -> Handle {
    let out = client.empty(len * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::lap2d::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**u).clone(), len) },
        unsafe { ArrayArg::from_raw_parts((**scale).clone(), 1) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len) }, r, c, periodic);
    out
}
fn run_grad<R: Runtime, E: Float + CubeElement>(client: ComputeClient<R>, u: &Rc<Handle>, scale: &Rc<Handle>, len: usize, r: usize, c: usize, axis: u32, periodic: u32) -> Handle {
    let out = client.empty(len * core::mem::size_of::<E>());
    let grid = len.div_ceil(256).max(1) as u32;
    kernels::grad2d::launch::<E, R>(&client, CubeCount::Static(grid, 1, 1), CubeDim::new_1d(256),
        unsafe { ArrayArg::from_raw_parts((**u).clone(), len) },
        unsafe { ArrayArg::from_raw_parts((**scale).clone(), 1) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), len) }, r, c, axis, periodic);
    out
}

fn host_identity(op: u32) -> f64 {
    match op {
        RED_PROD => 1.0,
        RED_MIN => f64::INFINITY,
        RED_MAX => f64::NEG_INFINITY,
        _ => 0.0,
    }
}

fn host_combine(op: u32, parts: &[f64]) -> f64 {
    match op {
        RED_PROD => parts.iter().product(),
        RED_MIN => parts.iter().cloned().fold(f64::INFINITY, f64::min),
        RED_MAX => parts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        _ => {
            // Neumaier sum across partials
            let mut sum = 0.0;
            let mut c = 0.0;
            for &x in parts {
                let t = sum + x;
                if sum.abs() >= x.abs() {
                    c += (sum - t) + x;
                } else {
                    c += (x - t) + sum;
                }
                sum = t;
            }
            sum + c
        }
    }
}
