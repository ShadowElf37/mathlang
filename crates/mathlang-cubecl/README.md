# mathlang-cubecl (`mc`)

A clean-port prototype of [mathlang](../../README.md) on top of
[CubeCL](https://github.com/tracel-ai/cubecl), collapsing the current three numeric
execution paths (tree-walk eval, bytecode VM loops, WGSL `GPU {}` block) into **one
backend-generic compute path** with **native f64** where the hardware allows it.

Status: **Phase 1b complete** — host interpreter + REPL over the scalar/complex/
tuple/lambda core; Phase 0 spike proven. Tensors (the CubeCL compute path) are
Phase 2.

## What works now

```sh
mc                 # REPL
mc 'pi * 2^2'      # one-liner
mc --spike         # f64-vs-f32 backend precision demo
```

Scalars, complex (`i^2`, `exp(i*pi)`, `sqrt(-1)`, `ln(-1)`, `abs/conj/...`), tuple
trees with broadcasting, functions/lambdas/closures/recursion, `if`, comparisons,
`sum`/`prod`/`map`/`filter`/`reduce`/`iterate`, `compose`/`partial`, `cell`/`get`/
`set`, and the scalar math builtins — all evaluated **host-side in f64** (instant,
no kernel: the low-latency invariant). REPL commands: `!help !backend !type !defs
!clear !print !spike !version !q`.

Tensor-producing syntax (`[...]`, matrix literals, `range`) and tensor/linalg/fft/
field/pic/calculus builtins error with a clear "later phase" message until Phase 2.

## Why

The existing WGSL GPU backend is f32-only (WGSL has no `f64`). CubeCL lets one
`#[cube]` kernel target cpu / wgpu(Metal) / cuda / hip, with the float element type
chosen per backend — native f64 on cpu/cuda/hip, f32 on wgpu.

## Build & run

```sh
cargo run -p mathlang-cubecl                       # default: cpu + wgpu (macOS-friendly)
cargo check -p mathlang-cubecl --no-default-features --features cuda   # compile-check NVIDIA path
cargo check -p mathlang-cubecl --no-default-features --features hip    # compile-check AMD path
```

Feature flags map 1:1 to CubeCL runtimes: `cpu`, `wgpu`, `cuda`, `hip`.

## Phase 0 findings (the spike)

The spike ([src/main.rs](src/main.rs)) computes `1.0 + 1e-10` elementwise with one
generic kernel on each backend:

```
cpu  (f64): 1.000000000100   <- native double, MLIR/LLVM backend
wgpu (f32): 1.000000000000   <- Metal, 1e-10 lost below the f32 ULP
```

That single op is the whole thesis: one kernel source, backend-chosen precision.

Resolved go/no-go questions:

1. **CubeCL builds on macOS** — yes, ~30s cold (`cubecl` 0.10.0, pinned).
2. **`cpu` backend is MLIR/LLVM** — `cubecl-cpu` pulls a *bundled* LLVM 20.1.4
   (`tracel-llvm-bundler`), so we get the MLIR CPU JIT discussed in the design
   review **without managing an LLVM dependency ourselves**. Native f64, runnable
   locally.
3. **`cuda` and `hip` type-check on this Mac** *without* the CUDA/ROCm toolkits
   installed (`cargo check` passes) — the vendor runtime is loaded at run time, not
   link time. All four backends are compile-checkable here; only execution needs the
   hardware.
4. **Runtime kernel fusion is feasible (GO).** `#[cube]` expands to an impl with
   `id()` + `define() -> KernelDefinition`, where `define()` drives the public
   `KernelBuilder` (`input_array`/`output_array`/`scalar`) and emits IR into a
   `cubecl_ir::Scope` (`create_local`, `register(instruction)`, `child()`). We can
   build the same `KernelDefinition` from a runtime walk of a mathlang lambda AST,
   so an arbitrary `iterate(x -> <expr>, …)` body can fuse into one kernel — the
   backend-agnostic analogue of today's WGSL codegen.

## Architecture decision

- **Baseline (build first):** a library of compile-time `#[cube]` kernels generic
  over `<F: Float>` (elementwise, unary, reduce, matmul, stencil), composed eagerly
  by the host interpreter with tensors kept device-resident between ops. Correct,
  multi-backend, native-f64. Guaranteed feasible (this spike is its first kernel).
- **Fusion (layer on later):** AST → `cubecl_ir` lowering + a custom kernel
  `define()` for fused lambda bodies. Proven feasible in Phase 0; deferred so the
  baseline lands first.

Either way there is exactly **one compute path** — no bytecode VM, no `GPU {}`
syntax. CubeCL types are confined to the (forthcoming) `compute` module so the
crate's alpha churn touches one place.

## Pinned versions

`cubecl = "=0.10.0"` (alpha — breaks between minors; pin exactly).
