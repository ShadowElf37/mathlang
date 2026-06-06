# mathlang-cubecl (`mc`)

A clean-port prototype of [mathlang](../../README.md) on top of
[CubeCL](https://github.com/tracel-ai/cubecl), collapsing the current three numeric
execution paths (tree-walk eval, bytecode VM loops, WGSL `GPU {}` block) into **one
backend-generic compute path** with **native f64** where the hardware allows it.

Status: **Phase 2 complete** — tensors run on the CubeCL compute path with a
three-mode precision model (f32 / df64 / f64) threaded through from the start.

```sh
mc                 # REPL
mc 'pi * 2^2'      # one-liner
mc '[1,2,3] + [4,5,6]'
mc --spike         # f64-vs-f32 backend precision demo
```

## What works now

**Host core (instant, no kernel — the low-latency invariant):** scalars, complex
(`i^2`, `exp(i*pi)`, `sqrt(-1)`, `abs/conj/...`), tuple trees with broadcasting,
functions/lambdas/closures/recursion, `if`, comparisons, `sum`/`prod`/`map`/
`filter`/`reduce`/`iterate`, `compose`/`partial`, `cell`/`get`/`set`, scalar math.

**Tensors (the compute path):** `[a,b,c]`, matrices `(1,2; 3,4)`, `a..b`,
`zeros`/`ones`/`eye`/`linspace`/`range`; elementwise `+ - * / ^` and comparisons
with scalar↔tensor broadcasting; unary math (`sin`/`exp`/`sqrt`/...); `shape`/
`rows`/`cols`/`len`; `sum`/`prod` (host reduce for now). Every tensor op runs on
the selected backend/precision — **no `GPU {}` block needed** (an improvement over
the original, which was f32-only and block-scoped).

**Precision (`!prec f32|df64|f64`, `!backend cpu|wgpu|cuda|hip`):**
* `f64` — native on cpu/cuda/hip. `[1.0]+[1e-10]` → `[1.0000000001]`.
* `f32` — universal. On wgpu the same op → `[1]` (1e-10 below the ULP).
* `df64` — double-single (~16 digits) stored as `[hi,lo]` f32 pairs; **storage/
  round-trip works on every backend including wgpu/Metal** (which has no f64).
  Arithmetic kernels (TwoSum/TwoProd) are staged — see `compute/kernels.rs`.

Switching to wgpu auto-downgrades f64→f32; `!prec f64` on wgpu is rejected.

REPL commands: `!help !backend !prec !type !defs !clear !print !spike !version !q`.

Deferred to later phases: tensor indexing/slicing, matmul/linalg, on-device
reductions, fft, fields/forms, pic, calculus, file I/O, animation.

## Tests

`bash crates/mathlang-cubecl/tests.sh` — scalar/complex/tuple core, tensor
elementwise/unary/constructors, and the cross-backend precision behaviour.

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
