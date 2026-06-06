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
`zeros`/`ones`/`eye`/`linspace`/`range`, build-by-function `tensor((i,j)->…)`/
`matrix`/`lingrid`; elementwise `+ - * / ^` and comparisons with scalar↔tensor
broadcasting; unary math (`sin`/`exp`/`sqrt`/`sign`/`floor`/`ceil`/...); elementwise
`min`/`max`/`select(cond,a,b)`; `shape`/`rows`/`cols`/`len`. **Indexing & slicing:**
`T[i,j]`, `T[.., j]`, `T[i, ..]`, `T[a..b]` (host-side gather). **Assembly:**
`reshape`, `transpose`, `cat`, `vstack`, `hstack`. **Axis reductions:** `sum(T,axis)`,
`prod(T,axis)`. **Linear algebra & reductions on device:** `@`/`matmul`
(2D×2D, mat·vec, vec·mat, dot), and `sum`/`prod`/`mean`/`min`/`max`/`norm`/`std`
(parallel reduction, Neumaier-compensated sum; df64 reduces via host fallback).
**Stencils on device:** `shift`/`roll`, and the `ops` namespace `ops.lap(T,dx[,bc])`
and `ops.grad(T,dx[,axis])` (periodic or `ops.neumann`) — enough to run the heat
equation under `iterate` fully resident. **Dense linalg (host-side):** `det`/`inv`/
`solve` (Gaussian elimination), `trace`, `diag`, and `eig`/`eigvals` (unshifted
Householder-QR — converges for symmetric matrices). Every tensor op runs on
the selected backend/precision — **no `GPU {}` block needed** (an improvement over
the original, which was f32-only and block-scoped).

**Precision (`!prec f32|df64|f64`, `!backend cpu|wgpu|cuda|hip`):**
* `f64` — native on cpu/cuda/hip. `[1.0]+[1e-10]` → `[1.0000000001]`.
* `f32` — universal. On wgpu the same op → `[1]` (1e-10 below the ULP).
* `df64` — double-single (~16 digits), each value an unevaluated `(hi, lo)` f32
  pair. Arithmetic uses error-free transforms (TwoSum, Dekker TwoProd) in
  `compute/kernels.rs`. `[1.0]+[1e-10]` → `[1.0000000001]`, `[1.0]/[3.0]` →
  `[0.33333333333333304]` — full df64 on the **IEEE backends (cpu/cuda/hip)**.
  This is the win on **consumer CUDA/AMD**, where native f64 is throttled (1/32
  rate) but f32-based df64 runs near f32 speed.
  * **wgpu/Metal caveat:** df64 *storage/round-trip* works, but df64 *arithmetic*
    is **gated off** there — the Metal/Vulkan driver's fast-math reassociates
    `b-(s-a)` and collapses the error term to ~f32. Rather than return a wrong
    answer, df64 ops error on wgpu (`!prec f32` for honest f32, or use cpu/cuda/hip).
  * Still staged everywhere: df64 `pow` and transcendentals (exp/ln/sin/…) — they
    need range-reduced double-single series.

Switching to wgpu auto-downgrades f64→f32; `!prec f64` on wgpu is rejected.

REPL commands: `!help !backend !prec !type !defs !clear !print !spike !version !q`.

Deferred to later phases: tensor indexing/slicing, matmul/linalg, on-device
reductions, fft, fields/forms, pic, calculus, file I/O, animation.

## Complex tensors

First-class device-resident complex tensors (interleaved `[re, im]`, f32/f64; df64
complex is not supported). A complex literal anywhere in `[…]` or a matrix makes the
result complex, and a real tensor meeting a complex scalar/tensor promotes:

```
[1, 2, 3] + 2i        → [1 + 2i, 2 + 2i, 3 + 2i]
[1+1i] * [1+1i]       → [2i]
sqrt([3+4i])          → [2 + i]          exp([0, πi]) → [1, -1]
abs([3+4i, 5+12i])    → [5, 13]          conj/re/im/arg, sin/cos/ln, sum/mean
```

Arithmetic `+ − × ÷`, `re`/`im`/`abs`/`arg`/`conj`, `exp`/`ln`/`sqrt`/`sin`/`cos`,
and `sum`/`mean` all run on device. Display collapses negligible imaginary parts
(so `exp(πi)` shows `-1`) without forcing a per-op download.

## Loops & residency (`iterate` / `scan`)

`iterate(f, x0, n)` and `scan(f, x0, n)` are the **one** loop mechanism — the
interpreter drives the loop and each step runs compute ops. Because a tensor value
*is* a device handle and every op produces another device handle, **tensor/tuple
state stays resident on the device across all `n` steps** — x0 is uploaded once, the
result downloaded once, no per-step transfer. This single path replaces both the old
bytecode-VM loop and the WGSL GPU-resident loop.

```
iterate(u -> u*0.5, [1,2,3,4], 3)          → [0.125, 0.25, 0.375, 0.5]   (resident)
iterate((u,v) -> (v, u), ([1,2],[3,4]), 1) → ([3, 4], [1, 2])            (tuple of tensors)
scan(x -> 2*x, 1, 4)                        → [1, 2, 4, 8, 16]            (scalar → 1-D)
scan(v -> (v[1], -v[0]), (1,0), 100)        → [101, 2] trajectory
```

`scan` stacks with time as the leading axis (scalar→`[n+1]`, tensor→`[n+1,…shape]`,
flat tuple→`[n+1,k]`, structured tuple→a tuple of per-field stacks).

Caveat — the loop is *data-resident but host-driven*: the host issues one kernel
launch per step, so millions of tiny steps pay launch overhead (fine for the usual
hundreds/thousands of steps on real grids). Fusing the whole loop body into one
on-device kernel (the README's "loop inside the kernel") is a later optimization
built on the runtime-AST→IR codegen proven in Phase 0.

## Spectral

`fft`/`ifft` (n-D over all axes, or `fft(T, axes)`; real or complex input → complex
output) and the `ops` spectral operators built on them:

```
fft([1,1,0,0])                 → [2, 1 - i, 0, 1 + i]
ops.specgrad(sin(x), dx, 0)    → cos(x)   (machine-precision on smooth periodic data)
ops.poisson(rhs, dx)           → u solving ∇²u = rhs (zero-mean), spectral
ops.invlap(T, dx)              → inverse Laplacian (alias of poisson)
```

Host FFT via `rustfft` — **f64 and any size**, which *exceeds* the original GPU path
(f32, power-of-two only). A device-resident FFT (Stockham) is a later optimization.

## Calculus

`integral` and `deriv` (host-side, functions evaluated pointwise):

```
integral(x -> x^2, 0, 1)              → 0.3333…        deriv(x -> x^3, 2) → 12
integral((x,y) -> x*y, [0,0], [1,1])  → 0.25           # tensor-product Simpson
deriv((x,y) -> x^2*y, (3,5))          → [30, 9]        # full gradient (5-point)
deriv((x,y) -> x^2*y, (3,5), 0)       → 30             # partial ∂/∂x
```

Scalar forms match the original (composite Simpson, 5-point stencil). The
**gradients and multidimensional integrals are an improvement** — the README of the
original describes them but the binary errors on them; here they work, which is what
Newton/optimization and quadrature need (e.g. `iterate(x -> x - (x^2-2)/deriv(t ->
t^2-2, x), 1.0, 5)` → √2).

## Fields & forms (exterior calculus)

A **field** packages grid samples with geometry (box, per-axis boundary conditions,
spacing, diagonal metric). Build one with `field(data, lo, hi, bc[, metric])` or the
function form `field(f, lo, hi, counts, bc[, metric])`; `forms.form`/`forms.vector`
build higher-degree forms / vector fields. The **`forms`** namespace is exterior
calculus on them:

```
forms.d(f)              # exterior derivative: k-form → (k+1)-form (grad/curl/div)
forms.hodge(f)          # Hodge star ★ (metric-aware)
forms.wedge(a, b)       # exterior product ∧
forms.raise/lower(f)    # musical ♯ / ♭
forms.codiff(f)         # codifferential δ = ±★d★
forms.laplace(f)        # Laplace–de Rham Δ = dδ + δd  (−∇² on a 0-form)
forms.contract(X, w)    # interior product ι_X
```

`dx` enters only `d`; the metric enters only hodge/raise/lower/codiff/laplace — so
the same code does Euclidean and Minkowski (`metric (-1,1,1,1)` ⇒ `forms.laplace` is
the d'Alembertian □). Field arithmetic (`+ − * /` by scalars/matching fields) stays
in the field algebra; any other builtin decays a field to its tensor; `tensor(f)`
extracts the grid data. Host-side (matching the original's CPU semantics); fields are
a host geometric object that bridge to device tensors via `field(…)`/`tensor(…)`.

## PIC (particle/grid coupling)

`pic.scatter`, `pic.gather`, and `pic.gathergrad` implement particle-in-cell
deposition and interpolation on any field geometry. Three shape functions:
nearest-grid-point (`pic.ngp`), cloud-in-cell (`pic.cic`, linear, the default),
triangular-shaped-cloud (`pic.tsc`, quadratic). Boundary conditions follow the
field's per-axis BC (periodic/Neumann); n-D grids use tensor-product stencils.

```
# Deposit weight 1 at x=2.5 on a 5-node Neumann [0,4] grid (CIC)
tensor(pic.scatter([2.5],[1.0], field(zeros(5),0,4,forms.neumann)))  → [0,0,0.5,0.5,0]

# Interpolate f(x)=x — CIC is exact for piecewise-linear fields
pic.gather(field([0,1,2,3,4], 0, 4, forms.neumann), [2.5])           → [2.5]

# Gradient of the shape function — the variational force for energies of ρ
pic.gathergrad(field([0,1,2,3,4], 0, 4, forms.neumann), [2.5])       → [1]
```

Adjointness holds by construction: `⟨gather(f,X), w⟩ == ⟨f, scatter(X,w)⟩`
(verified < 1e-12). `gathergrad` is the exact transpose of scatter's positional
derivative — a Verlet stepper using it conserves the Hamiltonian exactly for
self-gravitating or barotropic particle-mesh gases (no grid-scale heating).

## Tests

```sh
cargo test -p mathlang-cubecl
```

123 `#[test]` functions covering scalar/complex/tuple core, tensor ops, linear
algebra, resident loops, fields & forms, spectral operators, calculus, PIC, and
cross-backend precision (including wgpu/Metal). `tests.sh` is retained as a
legacy reference; `cargo test` is canonical.

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
