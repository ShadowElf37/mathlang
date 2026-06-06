//! Phase 0 spike: prove one `#[cube]` kernel runs as native f64 on the cpu runtime
//! and f32 on wgpu/Metal — backend-chosen precision from a single source.
//!
//! `1.0 + 1e-10` is `1.0000000001` in f64 but collapses to `1.0` in f32 (the
//! `1e-10` is far below the ULP at magnitude 1). That one elementwise op is the
//! whole thesis of the port in miniature. This module is throwaway once the real
//! `compute` layer lands; it stays as an executable proof for now.

use cubecl::prelude::*;

#[cube(launch)]
fn add_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}

fn run_add<R: Runtime, F: Float + CubeElement>(device: &R::Device, lhs: &[F], rhs: &[F]) -> Vec<F> {
    let client = R::client(device);
    let n = lhs.len();

    let lhs_h = client.create_from_slice(F::as_bytes(lhs));
    let rhs_h = client.create_from_slice(F::as_bytes(rhs));
    let out_h = client.empty(n * core::mem::size_of::<F>());

    add_kernel::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(n as u32),
        unsafe { ArrayArg::from_raw_parts(lhs_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(rhs_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
    );

    let bytes = client.read_one_unchecked(out_h);
    F::from_bytes(&bytes).to_vec()
}

/// Run the precision demo on every compiled-in backend.
pub fn run() {
    println!("backend precision demo — computing  1.0 + 1e-10  elementwise:\n");

    #[cfg(feature = "cpu")]
    {
        let out = run_add::<cubecl::cpu::CpuRuntime, f64>(
            &cubecl::cpu::CpuDevice,
            &[1.0_f64],
            &[1e-10_f64],
        );
        println!("  cpu  (f64): {:.12}   <- native double, MLIR/LLVM backend", out[0]);
    }

    #[cfg(feature = "wgpu")]
    {
        let out = run_add::<cubecl::wgpu::WgpuRuntime, f32>(
            &cubecl::wgpu::WgpuDevice::default(),
            &[1.0_f32],
            &[1e-10_f32],
        );
        println!("  wgpu (f32): {:.12}   <- Metal, 1e-10 lost below the f32 ULP", out[0]);
    }
}
