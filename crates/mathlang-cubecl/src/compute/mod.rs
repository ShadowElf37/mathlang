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
