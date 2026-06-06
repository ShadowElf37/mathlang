//! mathlang-cubecl prototype (`mc`). Phase 0 spike.
//!
//! Goal of this spike: prove the load-bearing claims of the CubeCL plan before
//! building the interpreter on top —
//!   1. a single `#[cube]` kernel, generic over the float element type, compiles
//!      and launches;
//!   2. the *same* kernel runs on the `cpu` runtime in **native f64** and on the
//!      `wgpu` (Metal) runtime in f32 — one source, backend-chosen precision;
//!   3. the precision difference is real, shown by a computation that only f64 can
//!      represent.
//!
//! The demo computes `1.0 + 1e-10` elementwise. In f64 the result is
//! `1.0000000001`; in f32 the `1e-10` is far below the ULP at magnitude 1, so the
//! result collapses to exactly `1.0`. That single elementwise op is the whole
//! thesis of the port in miniature.

use cubecl::prelude::*;

/// Elementwise add, generic over the CubeCL float type `F`. This is the one and
/// only kernel source; `F` is bound to `f64` or `f32` at launch by the chosen
/// runtime.
#[cube(launch)]
fn add_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}

/// Upload `lhs`/`rhs`, launch `add_kernel`, download the result. Generic over both
/// the runtime `R` (cpu/wgpu/…) and the element type `F`.
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

fn main() {
    println!("mc: mathlang-cubecl prototype — Phase 0 spike");
    println!("computing  1.0 + 1e-10  elementwise on each enabled backend:\n");

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

    println!("\nspike result: one kernel source, native f64 on cpu vs f32 on wgpu.");
}
