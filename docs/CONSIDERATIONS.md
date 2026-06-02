# GPU Backend Considerations for mathlang

This document is written for a contributor unfamiliar with the project who will
implement the GPU compute backend. It covers project context, design rationale,
architecture, and a concrete implementation plan.

> **Revision note (v0.27-era).** The first version of this document was written
> when the only numerical primitives were elementwise math, reductions, `matmul`,
> and `shift`/`roll`. Since then the language grew (a) a full set of differential
> operators (`ops.*` stencil + spectral, `forms.*` exterior calculus on a new
> first-class `Field` type), (b) particle/grid PIC operators (`pic.*`), (c)
> symplectic integrators (`solver.*`), and (d) a VM `Loop` instruction that runs
> `iterate`/`scan`/`sum`/`prod` without native-stack growth. Those four changes
> move the *valuable* GPU target away from "one fused elementwise block" and
> toward "**a stencil/spectral kernel applied hundreds of times to GPU-resident
> state**." This revision reworks the design around that reality. Sections marked
> **(NEW)** did not exist before; the rest were rewritten to match.

---

## 1. Project Overview

**mathlang** (`m`) is a REPL-based mathematical scripting language written in
Rust. Its primary use case is numerical/scientific computing: tensor math, PDEs,
FFTs, animations. The evaluator is a tree-walking interpreter with a bytecode VM
fast path (`src/eval.rs` + `src/vm.rs`).

Key source files:
- `src/lexer.rs`   — tokenizer
- `src/ast.rs`     — AST node types (`Expr`, `Op`, `Def`, `BlockStmt`)
- `src/parser.rs`  — recursive-descent parser
- `src/vm.rs`      — **bytecode instruction set** (`Instruction`, `LoopForm`);
                     deliberately depends only on `crate::ast`, so `src/gpu/` can
                     import the IR without pulling in the evaluator
- `src/eval.rs`    — main evaluator (~5000 lines); defines `Val`, `Env`, all
                     builtins, the VM executor (`run_vm`) and compiler (`Compiler`)
- `src/repl.rs`    — REPL with namespace-aware highlighting, Tab completion, help
- `src/animate.rs` — `animate2D`/`animate2D_raw` for 2D tensor animation
- `src/ns/`        — namespaced builtins: `ops` (differential operators),
                     `forms` (exterior calculus), `pic` (particle/grid),
                     `solver` (ODE/symplectic integrators), `linalg`, `stats`,
                     `vec`, `bits`, `special`

### Value types (`Val` in eval.rs):
```rust
enum Val {
    Num(f64),
    Complex(f64, f64),
    Tensor { data: TData, shape: Vec<usize> },          // TData = Arc<Vec<f64>>, O(1) clone
    ComplexTensor { re: TData, im: TData, shape: Vec<usize> },
    Field(Arc<FieldVal>),  // NEW: gridded differential form / vector field
    Fn(Vec<String>, Expr, Arc<HashMap<String, Val>>, Arc<OnceLock<Option<Vec<Instruction>>>>),
    Builtin(String),
    Tuple(Vec<Val>),
    Cell(Arc<RefCell<Val>>),   // mutable reference cell; identity semantics on clone
}
```

`FieldVal` (see `src/eval.rs`) is **new and matters for the GPU backend**:
```rust
pub struct FieldVal {
    pub data:     TData,        // flat row-major; component index varies fastest
    pub shape:    Vec<usize>,   // grid shape ++ [ncomp]  (ncomp = C(n,k) for a k-form)
    pub dx:       Vec<f64>,     // grid spacing per axis
    pub metric:   Vec<f64>,     // diagonal metric g_ii per axis (Euclidean = all 1)
    pub bc:       Vec<BC>,      // BC::Periodic | BC::Neumann, per axis
    pub variance: Variance,     // Variance::Form | Variance::Vector
}
```
A field is just a tensor plus the metadata the differential operators need (spacing,
metric, boundary conditions). See §11.

### Bytecode VM (`src/vm.rs` + `src/eval.rs`):

mathlang has a stack-based bytecode VM for user-defined function bodies.
`compile_fn(params, body, captured)` compiles an `Expr` to `Vec<Instruction>` or
returns `None` for unsupported nodes. On first call, `apply_fn_direct()` uses
`OnceLock::get_or_init()` to compile once and cache; all `Val::Fn` clones share the
same `Arc<OnceLock>`. Subsequent calls run the VM directly.

The instruction set lives in `src/vm.rs` (it was moved out of `eval.rs` precisely so
`src/gpu/` can depend on it cleanly). The two GPU-relevant additions since v1 of
this doc:

- **`Instruction::Loop(LoopForm, usize)`** — `sum`/`prod`/`iterate`/`scan` now
  compile to a single flat VM loop instead of forcing a tree-walk fallback. The
  operands are evaluated once and the loop runs with no native-stack growth. This
  is **the in-VM form of the bounded-iteration special forms** and the single most
  important structure for the GPU backend — see §10.
- `Index`, `MakeClosure`, and non-recursive `Def::Func` in blocks all compile.

### Language features relevant to GPU:
- Tensors are row-major flat `Vec<f64>`, any rank
- `tensor((i,j)->expr, rows, cols)` — builds tensors from index lambdas
- `shift(T, n, axis)` (edge-replicating / Neumann), `roll(T, n, axis)` (circular)
- `ops.grad/div/curl/lap` — finite-difference stencils (roll/clamp compositions)
- `ops.specgrad/poisson/invlap` — **spectral** (FFT-based) operators
- `forms.d/codiff/laplace/hodge/wedge/raise/lower/contract` — exterior calculus on `Field`
- `pic.scatter/gather/gathergrad` — particle↔grid deposit/interpolate
- `solver.rk4/odeint/verlet/tao/cfl` — ODE & symplectic integrators
- `fft`/`ifft` — n-D FFT over all axes
- `lerp`, `clamp`, reductions (`sum`/`mean`/`min`/`max`), unary math

---

## 2. File Isolation Policy

**All GPU work must live in `src/gpu/`.** The only permitted touches to the
existing codebase are the minimum seams needed to connect the GPU backend:

| File | Allowed change | Status |
|------|---------------|--------|
| `src/vm.rs` | Import `Instruction`/`LoopForm` from here | ✅ **already done** — the IR lives here so gpu can import it without the evaluator |
| `src/ast.rs` | Add `Expr::GpuBlock { body }` variant | seam |
| `src/lexer.rs` | Add `Token::Gpu` keyword | seam |
| `src/parser.rs` | Add one `Token::Gpu` branch in `parse_primary()` | seam |
| `src/eval.rs` | Add one `Expr::GpuBlock` match arm (~15 lines) | seam |
| `src/repl.rs` | Add `"GPU"` to keyword highlighter | seam |
| `Cargo.toml` | Add `wgpu`, `bytemuck`, `pollster` under a `gpu` feature | seam |

**Everything else — all shaders, all GPU evaluation logic, memory management,
pipeline caching, the FFT, the precision-emulation layer — goes inside `src/gpu/`.**
The contributor should do ~95% of the work without opening `eval.rs`, `parser.rs`,
or `repl.rs`. The backend is also trivially removable: delete `src/gpu/`, revert
the six seams, and the project compiles without a trace of GPU code.

---

## 3. The GPU Block Syntax

```
GPU {
    intermediate = var1 * var2;
    intermediate + sum(var1)
}
```

- `GPU` is a **keyword form**, not a function call. The parser needs a dedicated branch.
- Free variables from the enclosing CPU scope are **automatically detected** by
  walking the block body AST (same approach as `collect_free_vars` in the VM). The
  allowed capture types are `Val::Num`, `Val::Tensor`, and `Val::Field` (see §11);
  anything else is a runtime error at block entry.
- The body is a standard mathlang block: `;`-separated statements, last expression
  is the return value. (Note: a newline is **not** a statement separator inside a
  single block — use `;`.)
- The result is downloaded back to the CPU as `Val::Num`, `Val::Tensor`, or
  `Val::Field` and is available in the enclosing scope like any other expression.
- Local bindings inside the block are GPU-local and never visible on the CPU side.

### Parser addition (ast.rs + parser.rs):
```rust
// ast.rs — new Expr variant:
GpuBlock { body: Box<Expr> }

// parser.rs — in parse_primary(), before the general Ident path:
Token::Gpu => {
    expect(LBrace);
    let body = parse_block();  // reuse the existing block parser
    expect(RBrace);
    Expr::GpuBlock { body: Box::new(body) }
}
```
Add `Token::Gpu` to the lexer as a reserved keyword.

---

## 4. Design Overview: record → (fuse) → dispatch, with a loop-aware IR

The backend is two-phase: **Phase 1** walks the block AST and *records* a program
of `GpuOp`s (no GPU calls, pure shape inference); **Phase 2** *dispatches* the whole
program with a single synchronization point. This is unchanged and correct.

What **is** new since v1 of this doc: the IR must be **loop-aware**. The recorded
program is not a flat `Vec<GpuOp>` — it is a small tree whose key node is

```rust
GpuOp::Loop { body: Vec<GpuOp>, n: GpuArg, state: Vec<BufId>, ... }
```

so the recorder emits the loop body **once** and the dispatcher owns the iteration
and the ping-pong of state buffers (§10). A flat list cannot express "replay this
sub-sequence N times reusing buffers," which is the single most important thing the
GPU backend does for real workloads. Everything else (a one-shot elementwise block)
is the degenerate `n = 1` case.

The optional **fusion pass** sits between record and dispatch, rewriting subtrees
(e.g. `a*b + c` → one `FusedMAD`, or a `roll` + elementwise chain → one `Stencil`,
§8) without the dispatcher knowing. Bytecode lowering (§15) plugs in here too, to
turn GPU-safe lambdas into single kernels.

---

## 5. Architecture

### 5.1 File structure

```
src/
  gpu/
    mod.rs          — re-exports; feature gate; lazy GpuContext init
    context.rs      — GpuContext: device, queue, pipeline cache, adapter info
    val.rs          — GpuVal enum; GpuBuf struct; upload/download (incl. Field)
    ops.rs          — GpuOp enum (incl. Loop, Stencil, Fft); shape_of(op, inputs)
    eval.rs         — gpu_eval(): AST → GpuOp program        (Phase 1)
    fuse.rs         — optional rewrite pass over the GpuOp tree
    dispatch.rs     — execute(program, ctx) → GpuVal          (Phase 2)
    precision.rs    — f32 / compensated-reduce / df64 mode selection (§7)
    vm_lower.rs     — bytecode Vec<Instruction> → WGSL (lambda lowering, §15)
    fft.rs          — Stockham radix-2 FFT driver (§9)
    shaders/
      elementwise.wgsl    — add/sub/mul/div/pow (scalar broadcast via uniform)
      unary.wgsl          — neg/exp/log/sqrt/sin/cos/abs/floor/ceil/round
      matmul.wgsl
      reduce.wgsl         — sum/mean/min/max; sum uses compensated accumulation
      stencil.wgsl        — generic n-D constant-stencil w/ per-axis BC (§8)
      scatter.wgsl        — particle→grid atomic deposit (§12)
      gather.wgsl         — grid→particle (value and gradient weights) (§12)
      fft_stockham.wgsl   — one radix-2 stage; driver loops stages/axes (§9)
      lerp_clamp.wgsl
      init_const.wgsl
      df64/*.wgsl         — double-single arithmetic include (§7), if df64 enabled
```

### 5.2 Core types

```rust
// val.rs
pub struct GpuBuf {
    pub buf:   wgpu::Buffer,   // STORAGE | COPY_SRC usage
    pub shape: Vec<usize>,
    pub len:   usize,          // shape.iter().product()
    pub prec:  Prec,           // F32 | Df64  (storage layout; see §7)
}

pub enum GpuVal {
    Scalar(f64),               // CPU-side; passed as uniform/push-constant
    Buffer(GpuBuf),            // lives on GPU until block exit
    Field(GpuBuf, FieldMeta),  // buffer + dx/metric/bc/variance carried as uniforms
}

pub struct FieldMeta {        // mirror of FieldVal minus the data
    pub grid:     Vec<u32>,
    pub ncomp:    u32,
    pub dx:       Vec<f32>,
    pub metric:   Vec<f32>,
    pub bc:       Vec<u32>,    // 0 = periodic, 1 = neumann
    pub variance: u32,
}
```

### 5.3 GpuContext

```rust
// context.rs
pub struct GpuContext {
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    pub limits:  wgpu::Limits,
    pub has_f64: bool,             // device feature; almost always false (see §7)
    pipelines:   HashMap<PipelineKey, wgpu::ComputePipeline>,  // (name, prec, specialization)
}
```
Lazy-initialize with `OnceLock<Mutex<GpuContext>>`. Pipelines are keyed by
`(shader, precision, specialization)` because the df64 variant of a shader is a
different pipeline. If no adapter is found, the GPU block returns a clear error:
`GPU block requires a compatible GPU; no adapter found on this system.`

### 5.4 GpuOp — the IR

```rust
// ops.rs
pub enum GpuOp {
    Upload   { src: Vec<f64>, shape: Vec<usize>, dst: BufId },
    Elemwise { op: ElemOp, lhs: GpuArg, rhs: GpuArg, dst: BufId, shape: Vec<usize> },
    Unary    { op: UnaryOp, src: GpuArg, dst: BufId, shape: Vec<usize> },
    MatMul   { a: BufId, b: BufId, dst: BufId, m: usize, k: usize, n: usize },
    Reduce   { op: ReduceOp, src: BufId, axis: Option<usize>, dst: BufId, out_shape: Vec<usize> },

    // NEW — the differential-operator workhorse (§8). One fused stencil kernel
    // covers grad/div/curl/lap and forms.d/codiff/laplace.
    Stencil  { src: BufId, taps: Vec<(Vec<i64>, f64)>, bc: Vec<Bc>, axis_scale: Vec<f64>,
               dst: BufId, shape: Vec<usize> },

    // NEW — spectral operators (§9). kind selects forward/inverse + k-multiplier.
    Fft      { src: BufId, inverse: bool, dst: BufId, shape: Vec<usize> },
    SpecOp   { src: BufId, kind: SpecKind, dx: Vec<f64>, dst: BufId, shape: Vec<usize> },

    // NEW — PIC (§12).
    Scatter    { pos: BufId, w: BufId, grid: BufId, kernel: u32, shape: Vec<usize> },
    Gather     { field: BufId, pos: BufId, kernel: u32, grad: bool, dst: BufId },

    // NEW — the centerpiece (§10). The body runs n times; state buffers ping-pong.
    Loop     { body: Vec<GpuOp>, n: GpuArg, state: Vec<BufId>, result: BufId },

    Shift    { src: BufId, n: i64, axis: usize, dst: BufId, shape: Vec<usize> },
    Roll     { src: BufId, n: i64, axis: usize, dst: BufId, shape: Vec<usize> },
    Lerp     { a: GpuArg, b: GpuArg, t: GpuArg, dst: BufId, shape: Vec<usize> },
    Clamp    { src: GpuArg, lo: f64, hi: f64, dst: BufId, shape: Vec<usize> },
}

pub enum GpuArg { Buf(BufId), Scalar(f64) }   // scalar broadcast is the common case
pub type BufId  = u32;                          // index into Vec<Option<GpuBuf>>
```

`ops.rs` also implements `shape_of(op, input_shapes) -> Vec<usize>` — pure shape
inference, no GPU calls. Phase 1 uses it to pre-allocate every `dst` buffer.

### 5.5 gpu_eval (Phase 1 — record)

```rust
// gpu/eval.rs
pub fn gpu_eval(expr: &Expr, env: &GpuEnv, prog: &mut GpuProgram, bufs: &mut BufPool)
    -> Result<GpuVal, String>
```
A straightforward AST match: `Num`→Scalar; `Var`→env lookup; `BinOp`→`Elemwise`;
`Neg`→`Unary`; `Block`→fold over stmts extending `env`, return last; `Apply(known
builtin, args)`→the matching `GpuOp`. The two cases that are *not* a flat emit:
- `Apply(Var("iterate"|"scan"|"sum"|"prod"), ...)` — recurse into the body **once**
  to record `Loop.body`, then emit `GpuOp::Loop` (§10).
- `Apply(Var("tensor"), [lambda, m, n])` — lower the lambda via §15 if GPU-safe.

Anything unsupported returns a clear `"... not supported in GPU block"` error.

### 5.6 dispatch (Phase 2 — execute)

```rust
// gpu/dispatch.rs
pub fn dispatch(prog: &GpuProgram, ctx: &mut GpuContext) -> Result<GpuVal, String>
```
1. Allocate all output buffers upfront (sizes from shape inference).
2. Create one `CommandEncoder`.
3. Walk the program recording a `ComputePass` per op. For `GpuOp::Loop`, record the
   body `n` times (or, if `n` is large/data-dependent, submit the body in a loop on
   the CPU — see §10) ping-ponging the `state` buffers between bind groups.
4. `queue.submit([encoder.finish()])`; `device.poll(Wait)`.
5. Download and return the final `GpuVal`.

### 5.7 CPU eval integration (one arm in src/eval.rs)

```rust
Expr::GpuBlock { body } => {
    let free = collect_gpu_free_vars(body, env);     // AST walk, like collect_free_vars
    let gpu_env = free.iter().map(|name| match env.vars.get(name) {
        Some(Val::Num(f))               => Ok((name.clone(), GpuVal::Scalar(*f))),
        Some(Val::Tensor{data, shape})  => Ok((name.clone(), gpu::upload_tensor(data, shape)?)),
        Some(Val::Field(f))             => Ok((name.clone(), gpu::upload_field(f)?)),   // NEW
        Some(other) => Err(format!("GPU: `{name}`: only scalars, tensors, and fields \
                                    are allowed; got {}", fmt_val(other))),
        None        => Err(format!("undefined variable `{name}`")),
    }).collect::<Result<_,_>>()?;

    let mut prog = GpuProgram::new();
    let mut bufs = BufPool::new();
    let result   = gpu::eval::gpu_eval(body, &gpu_env, &mut prog, &mut bufs)?;
    let ctx      = GPU_CTX.get_or_init(...);
    gpu::to_cpu_val(gpu::dispatch::dispatch(&prog, &mut ctx.lock().unwrap())?)
}
```
This is the **only** change to `src/eval.rs`.

---

## 6. Shader Design Philosophy

Once the plumbing works, adding a GPU builtin is: write `shaders/new_op.wgsl`; add a
`GpuOp::NewOp` variant; add its `shape_of` case; add a `dispatch_new_op()`; add a
`gpu_eval` arm. Keep it that mechanical — no macros needed.

All shaders follow one pattern (elementwise add shown):
```wgsl
@group(0) @binding(0) var<storage, read>       a:   array<f32>;
@group(0) @binding(1) var<storage, read>       b:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Params { len: u32, scalar_b: f32, use_scalar_b: u32 }
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.len { return; }
    let bval = select(b[i], p.scalar_b, p.use_scalar_b != 0u);
    out[i] = a[i] + bval;
}
```

Use specialization constants (via `PipelineCompilationOptions`) or separate shaders
per op — both fine at this scale. **Precision is a build-time variant of each
shader, not a runtime branch** — see §7.

---

## 7. Precision: f32, compensated reductions, and df64 — the real f64 story (NEW / expanded)

This is the most important correction to the original document, which waved off f32
as "acceptable for scientific computing." Two of mathlang's flagship workloads are
exactly the ones that aren't:

- **Symplectic integrators** (`solver.verlet`/`tao`). We have verified energy drift
  on the order of `1e-5` over hundreds of steps on the CPU. In f32 (~7 decimal
  digits) that conservation is destroyed: round-off in the per-step state update
  accumulates and the invariant visibly walks.
- **Spectral Poisson** (`ops.poisson`/`invlap`) divides by `k²`, which amplifies
  f32 round-off at the low modes that carry most of the energy.

### Why "wait for hardware f64" is a dead end

The blocker is **the WGSL language, not the hardware.** WGSL has no `f64` scalar
type at all, and there is no near-term path to one:

- **WebGPU/WGSL deliberately omit `f64`.** Even where a native backend's device
  exposes a double feature, you cannot *write* `f64` in WGSL, so wgpu cannot emit it
  through the normal shader path. (`GpuContext::has_f64` will read the device
  feature, but it is essentially never actionable from WGSL.)
- **On this project's own platform it is physically absent.** The dev machine is
  macOS → the wgpu backend is **Metal**, and the Metal Shading Language has **no
  `double` type whatsoever**, on any Apple GPU. So "f64 when modern GPUs support it"
  never arrives here regardless of language evolution.

Conclusion: a WGSL-based backend must treat f64 as **unavailable** and earn extra
precision in software. The good news is that the two techniques below are cheap
where it matters and get you most of the way to f64.

### Three-tier precision model (implement in `gpu/precision.rs`)

**Tier 0 — f32 storage, f32 math (default).** Buffers are `array<f32>`; tensors are
converted `f64→f32` on upload, `f32→f64` on download. Fine for visualization,
diffusion, advection, and anything you're going to animate anyway.

**Tier 1 — f32 storage, *compensated reductions* (default for reductions).** The
dominant error in scientific f32 is catastrophic cancellation in large sums, not
per-element round-off. So the `sum`/`mean` reduce shader (and `matmul`'s inner
accumulate, and energy/mass diagnostics) should use **Kahan/Neumaier compensated
summation** — carry a running compensation term per partial. This is a handful of
extra FLOPs and recovers most of the precision that matters for conserved-quantity
diagnostics, while keeping fields in compact f32. **Make this the default for every
reduction**; it is nearly free and removes the worst surprise.

**Tier 2 — `df64` (double-single) storage and math, opt-in.** Represent each
high-precision value as an *unevaluated sum of two f32s* `(hi, lo)` packed as
`vec2<f32>`, giving ~48 bits of mantissa (vs 52 for true f64 — about 1.5 decimal
digits short, far better than f32's ~7). Implement Dekker/Knuth `two_sum`,
`two_prod` (use `fma` where available) and build `df_add/df_sub/df_mul/df_div` in a
`shaders/df64/` include that the precision-critical shaders `#include`-style
concatenate. Cost is ~3–20× the ALU of plain f32 per op, but it needs **no hardware
f64** and runs everywhere, **including Metal**. Reserve it for the kernels that need
it — the symplectic state update, and optionally a split-precision FFT for spectral
Poisson — not for whole fields you only visualize.

This is the same trade-off real GPU scientific codes make (cuDoubleSingle, GLSL
"emulated double"): you don't get f64 from the hardware, you *synthesize* ~f64 from
pairs of f32 only on the hot path that needs it.

### Surfacing the choice

Default to Tier 0 + Tier-1 reductions automatically. Let the user opt a block into
df64 with a keyword modifier, e.g. `GPU f64 { ... }` (parser: optional `f64` token
after `GPU`), which sets every buffer's `Prec` to `Df64` and selects the df64
pipeline variants. Document the precision of each tier in `!help GPU`. Also emit a
one-line warning (once) when a `solver.verlet`/`tao` loop runs under Tier 0, since
that is the case most likely to silently mislead.

---

## 8. Differential operators on GPU — the generic `Stencil` op (NEW)

The `ops.grad/div/curl/lap` and `forms.d/codiff/laplace` operators are the main
reason to want a GPU here, and they are *already* in the GPU-friendly class: every
one of them is a composition of `roll` (periodic shift) and edge-clamp (Neumann
shift) plus a constant linear combination. Concretely, in the CPU implementation:

```
grad(T, dx)[axis] = (roll(T,-1,axis) - roll(T,+1,axis)) / (2*dx)
lap(T, dx)        = Σ_axis (roll(T,-1,axis) + roll(T,+1,axis) - 2*T) / dx²
div(V, dx)        = Σ_axis grad of component a along axis a
curl(V, dx)       = ∂V_y/∂x - ∂V_x/∂y         (2-D scalar curl)
```

The wrong way is to emit a separate `Roll` + `Elemwise` GpuOp for every term: a 3-D
Laplacian becomes ~7 kernel launches and 6 temporary buffers per call, and you do it
every timestep. The right way is **one generic `Stencil` GpuOp** parameterized by a
list of `(offset_vector, coefficient)` taps plus a per-axis BC mode and per-axis
`1/dxⁿ` scale. One `stencil.wgsl` then covers grad, lap, div, curl, and `forms.d`
in a single pass with one read per neighbor and one write:

```wgsl
// each thread computes one output cell: acc = Σ_taps coef * sample(src, idx + off, bc)
// sample() applies periodic wrap or edge clamp per axis from the bc uniform.
```

This is the single highest-leverage shader in the project. Build it early.

Two BC subtleties the original doc missed:
- The stencil must take **BC as a parameter** (periodic = wrap, Neumann = clamp),
  because the same operator runs on both kinds of field. Don't bake BC into the op
  identity the way `shift`/`roll` do.
- For `forms.*` the operator also needs the per-axis `dx` and `metric` and the
  component layout (`ncomp`, subset ordering). Those ride along in `FieldMeta`
  (§11); `d`/`codiff` select which component-to-component taps to apply.

---

## 9. Spectral operators and the FFT decision (NEW)

`ops.specgrad`, `ops.poisson`, and `ops.invlap` are **FFT-based**, and the original
doc excluded FFT ("wgpu has no FFT"). That exclusion is no longer tenable, because
**`ops.poisson` runs inside the inner loop of real examples** — e.g. the
self-gravitating gas computes the gravitational potential via spectral Poisson
*every Verlet step*. If the GPU backend can't do FFT, it must bounce that solve to
the CPU each iteration, which forces an upload+download per step and **destroys the
residency model of §10** — the whole point of running on the GPU.

So make a deliberate choice, and prefer (a):

**(a) Implement a GPU FFT (recommended).** A **Stockham radix-2** auto-sort FFT is
the standard wgpu approach: one `fft_stockham.wgsl` does a single butterfly stage,
and a CPU-side driver (`gpu/fft.rs`) loops over `log2(n)` stages per axis, applying
it to each axis in turn (mathlang's `fft` is n-D over all axes). It's a few hundred
lines, well-documented, and gives you `fft`/`ifft`, `specgrad`, and — most
importantly — `poisson`/`invlap` as a forward FFT → multiply by the `k`-space
kernel (`-1/|k|²` with the project's `lap(poisson(f)) = +f` convention) → inverse
FFT. Constraints: pad/restrict to power-of-two axis lengths for v1 (or add
Bluestein later); do the spectral multiply in df64 (§7) if Poisson accuracy
matters. The `SpecOp` GpuOp wraps this so the loop body can stay fully on-GPU.

**(b) Scope spectral out, document it.** If FFT is too much for v1, explicitly
restrict the GPU backend to the explicit-stencil PDE regime (§8) and state plainly
that spectral operators stay on CPU and break GPU residency. Acceptable as a stage,
but it means the gravity-gas / spectral-Poisson workloads see no benefit — so it
should be a known temporary limitation, not a silent gap.

Note `fft` returns a `ComplexTensor`; see §20.6 for the complex-on-GPU layout, which
the FFT forces you to confront anyway (the natural choice is two real buffers, or
`vec2<f32>` per element).

---

## 10. The `iterate`/`Loop` residency model — the centerpiece (NEW)

**This is the headline change.** The valuable GPU workload is not a single fused
block; it is **the same kernel applied hundreds of times to evolving state**: PDE
time-stepping, `solver.verlet`/`rk4`/`tao`, PIC pushes. The shape is always

```
GPU { iterate(step, u0, n) }      // or scan(step, u0, n) to keep every frame
```

where `step` is a pure function `state -> state`. If each step were its own
`GPU{...}` block, you'd upload `u` and download `u'` every iteration and PCIe
traffic would dwarf the compute — you'd lose to the CPU. The fix is **on-GPU
residency**: upload `u0` once, run all `n` steps keeping state on the GPU, download
once.

The VM already hands you the right structure. `Instruction::Loop(LoopForm::Iterate,
…)` evaluates the operands once and runs the loop with no stack growth, and
`iterate_vals`/`scan_vals` pre-evaluate the step function so its **bytecode cache is
shared across all N calls** (`src/eval.rs`). The GPU analogue:

1. In `gpu_eval`, when you hit `iterate(step, u0, n)`: record `step`'s body to a
   `Vec<GpuOp>` **once** (this is where §15 lambda lowering or a nested `gpu_eval`
   of the step's block goes), identify the `state` buffer(s), and emit
   `GpuOp::Loop { body, n, state, result }`.
2. Allocate **two** state buffers per state tensor (A and B) and **ping-pong**:
   step reads A writes B, next step reads B writes A. No per-step allocation; total
   CPU memory is `O(captures + state)` regardless of N.
3. Dispatch records the body once and replays it: either record `n` compute passes
   into one encoder (best when `n` is a known constant and not huge), or submit the
   body in a CPU loop reusing cached bind groups (when `n` is large or data-driven).
   Either way it's one `device.poll(Wait)` style sync at the end — not per step.
4. `scan` is the same but writes each frame into a slice of an `[n, …]` output
   buffer instead of ping-ponging in place — exactly what you want to feed
   `animate2D` after a single download.

This subsumes the original doc's "single block" as the `n = 1` case, and it is the
correct home for the spectral Poisson (§9) and PIC (§12) kernels: they must be
GPU-resident *inside* the loop body or the residency win evaporates.

`sum`/`prod` over a function (`sum(f, lo, hi)`) lower to the same `Loop` node with a
compensated accumulator (§7) instead of ping-ponged state.

---

## 11. `Field` as a first-class GPU value (NEW)

The original doc modeled only `Num` and `Tensor`. The differential operators in
`forms.*` consume `Val::Field`, which is a tensor plus `{ dx, metric, bc, variance,
ncomp }`. Two design decisions:

- **Upload:** a `Field` becomes a `GpuVal::Field(GpuBuf, FieldMeta)` — the `.data`
  goes into a buffer exactly like a tensor; `dx`/`metric`/`bc`/`variance`/`ncomp`
  ride along as a small uniform struct (`FieldMeta`, §5.2) that the stencil/spectral
  shaders read. BC **must** reach the kernel (periodic vs Neumann changes the
  neighbor sampling); this is why §8's stencil takes BC as a parameter.
- **Capture validation:** the §5.7 arm must accept `Val::Field` (the original code
  rejected anything non-Num/Tensor). On download, reconstruct a `Val::Field` with
  the original metadata when the result is a field, or a `Val::Tensor` when an
  operator collapsed it to a scalar field (e.g. `div`, `curl`).

If supporting `Field` directly is too much for a first cut, the fallback is: require
users to pass `field_data_as_tensor(f)` plus explicit `dx`/BC arguments into the
block and reconstruct the field on the CPU side. But first-class fields are not much
more work and keep `forms.*` code identical inside and outside the block.

---

## 12. PIC operators — `gather` is easy, `scatter` needs atomics (NEW)

`pic.gather` and `pic.gathergrad` are particle→grid **reads**: each particle (one
thread) samples the field at its neighbors with the shape-function weights
(`gathergrad` uses the *gradient* of the shape function). Embarrassingly parallel,
one kernel each (`grad: bool` flag selects which weights). Easy.

`pic.scatter` is the hard one: it is a grid **deposit** where many particles write
to the same cell, so it needs **atomic accumulation** into the grid. WGSL atomics
are integer-only, so you need one of:
- **atomic-CAS f32 accumulation** (`atomicCompareExchangeWeak` loop on a bitcast
  `atomic<u32>` view) — simplest, but contention-heavy for clustered particles, and
  **non-deterministic ordering** (f32 add isn't associative → bit-level run-to-run
  variation);
- **fixed-point integer accumulation** (`atomicAdd<i32>`/`i64` on a scaled grid),
  which is *deterministic* and often higher effective precision for sums — a good
  match for Tier-1 precision goals (§7);
- **sort-by-cell + segmented reduce** — most scalable, most work.

Recommend fixed-point atomic deposit for v1: deterministic, decent precision, and it
sidesteps the f32-associativity problem that would otherwise make scatter results
irreproducible. Flag scatter as the one PIC kernel that needs real design attention;
the gathers are trivial.

---

## 13. GPU-Allowed Builtins

### v1 target (covers the real workloads)
| Category | Builtins |
|----------|---------|
| Elementwise arithmetic | `+ - * / ^` (scalar-scalar, scalar-tensor, tensor-tensor) |
| Unary math | `- exp log sqrt sin cos abs floor ceil round` |
| Reductions | `sum mean min max` (whole-tensor and per-axis; compensated, §7) |
| Linear algebra | `matmul` |
| Tensor manip | `shift roll lerp clamp` |
| **Differential (stencil)** | `ops.grad ops.div ops.curl ops.lap`, `forms.d forms.codiff forms.laplace` (one `Stencil` op, §8) |
| **Bounded loops** | `iterate scan sum(f,lo,hi) prod(f,lo,hi)` (`Loop` op, §10) |
| Constructors | scalar literals, `zeros`, `ones` |
| Comparisons | `< > <= >= == !=` (→ 0.0/1.0 tensor) |

### v2 (high value, needs more shader work)
- **`fft ifft`, `ops.specgrad ops.poisson ops.invlap`** — Stockham FFT (§9)
- **`pic.gather pic.gathergrad`** (easy), **`pic.scatter`** (atomics, §12)
- **`tensor((i,j)->expr, m, n)`** via bytecode→WGSL lowering (§15)
- `forms.hodge/wedge/raise/lower/contract` (metric-weighted elementwise/permute)

### Explicitly NOT in scope
- `cell`/`get`/`set` (mutable state), dynamic dispatch (`CallVal`)
- recursion / `MakeClosure` (no nested fns in WGSL)
- `animate2D` (orthogonal — the block just returns the tensor/scan output)
- `solve` (would need a GPU LU/CG; later)

---

## 14. Memory model

**Upload (CPU→GPU):** at block entry each captured `Tensor`/`Field` is uploaded via
a staging buffer (`COPY_DST|COPY_SRC|STORAGE`), converting `f64→f32` (or `f64→df64`
pair, §7) on the way. All uploads are the first ops in the program.

**Download (GPU→CPU):** only the final result (and, for `scan`, the stacked frames)
is downloaded — `copy_buffer_to_buffer` to a `MAP_READ` staging buffer,
`map_async` + `poll(Wait)`, then `f32→f64` (or df64-pair→f64) back.

**O(1) intermediates + loop residency:** intermediate and per-step state buffers
live entirely on the GPU, referenced only by `BufId`. The §10 loop ping-pongs two
state buffers for the whole run. Total CPU-side memory for a block is
`O(captures + result)` regardless of how many ops or iterations execute. This is the
property that makes GPU PDE stepping actually faster than CPU.

---

## 15. Bytecode as GPU IR — lambda lowering (revised)

The bytecode VM (`src/vm.rs` defines the IR; `src/eval.rs` runs it) is the bridge
for lowering GPU-safe lambdas — needed for `tensor((i,j)->expr, m, n)` and for the
`step` function inside `iterate`/`scan` (§10).

> **Correction to the original doc.** It said `Instruction` lives in `eval.rs` and
> `src/gpu/` should import `crate::eval::Instruction`. That is stale: the IR now
> lives in **`src/vm.rs`** specifically so the GPU backend can depend on it without
> the evaluator. Import `crate::vm::{Instruction, LoopForm}`.

### GPU-safe subset of `Instruction`

| Instruction | WGSL | Safe? |
|---|---|---|
| `PushNum(f)` | `f32(f)` literal | ✓ |
| `PushComplex` | — | ✗ no complex on GPU (see §20.6) |
| `LoadParam(i)` | param variable | ✓ |
| `LoadCaptured(name)` | uniform (scalar) / buffer binding (tensor) | ✓ if scalar or uploaded tensor |
| `BinOp(Add\|Sub\|Mul\|Div)` | `+ - * /` | ✓ |
| `BinOp(Pow)` | `pow(a,b)` | ✓ |
| `BinOp(Lt\|Gt\|LtEq\|GtEq\|Eq\|Ne)` | comparison → `f32` 0/1 | ✓ |
| `Neg` | unary `-` | ✓ |
| `CallBuiltin(sin\|cos\|exp\|sqrt\|abs\|floor\|ceil\|round\|log, 1)` | WGSL builtin | ✓ |
| `JumpIfFalse`/`Jump` | `if/else` | ✓ |
| `StoreLocal`/`LoadLocal` | `var x: f32 = …;` | ✓ |
| `Pop`/`Return` | discard / return | ✓ |
| `Index` (base = `LoadCaptured` of an uploaded tensor) | `buf[linear_idx]` | ✓ needs name→buffer type map |
| `Index` (base = computed value) | — | ✗ no mid-kernel tensors |
| **`Loop(Sum\|Prod, _)`** | bounded `for` with (compensated) accumulator | ✓ if body scalar & GPU-safe |
| **`Loop(Iterate\|Scan, _)`** | bounded `for` over a scalar carry | ✓ *scalar* carry only; the **tensor** `iterate` is the §10 residency path, handled at the GpuOp layer, not inside a lambda |
| `MakeTuple`/`MakeArray` | — | ✗ heap alloc |
| `MakeClosure` | — | ✗ no nested fns |
| `CallVal` | — | ✗ dynamic dispatch |
| `CallBuiltin(special form)` | — | ✗ not scalar |

`is_gpu_safe(code, tensor_captures)` scans the `Vec<Instruction>` and accepts only
the ✓ rows (for `Index`, check that the base is a captured-tensor `LoadCaptured`).
The new rows are the two `Loop` forms — add them.

### `gpu/vm_lower.rs`

```rust
pub enum CaptureKind { Scalar(f32), Buffer { binding: u32, shape: Vec<u32> } }
pub fn lower_to_wgsl(code: &[Instruction], params: &[&str], n_locals: usize,
                     captures: &HashMap<String, CaptureKind>) -> Result<String, String>
```
A second linear pass turning the stack IR into WGSL: each push → `let t_N: f32 = …;`;
`StoreLocal(s)` → `local_s = t_N;`; `JumpIfFalse`+`Jump` → `if (t != 0.0) {…} else
{…}`; a `Loop(Sum,…)` → `var acc = 0.0; var c = 0.0; for (…) { /*Kahan*/ }`. The
result is an inline body embedded in a `TensorFromLambda` (for `tensor(…)`) or in
the §10 loop-body kernel (for `iterate`'s `step`).

If `is_gpu_safe` fails inside a `GPU{…}` block: **hard error** (don't silently fall
back to CPU and mask a performance bug). Tell the user which instruction was
unsupported and to move that computation outside the block.

### `tensor((i,j)->expr, m, n)` on GPU
Path: `gpu_eval` hits `Apply(Var("tensor"), [lambda, m, n])` → eval `m,n` →
`cache.get_or_init()` → `is_gpu_safe` → `lower_to_wgsl` → emit `TensorFromLambda`
→ dispatch `m*n` threads, each writes `out[i*n+j] = body(i,j)`. The no-index form
(`tensor((i,j)->i*j, 4,4)`) is GPU-safe immediately; the indexing form
(`tensor((i,j)->sin(T[i,j]), …)`) needs the capture→buffer map, available for free
from the uploaded captures.

---

## 16. REPL syntax highlighting

Render the `GPU` keyword (and an optional `f64` precision modifier, §7) in **bold
red** to signal a different execution context, and optionally color the matching
`{ } ( )` at the block's brace depth. `src/repl.rs` already has namespace-aware
highlighting; extend its keyword set. Add `GPU` (and `f64` in that position) to Tab
completion too.

---

## 17. Cargo.toml

```toml
[features]
default = []            # GPU off by default so plain `cargo build` needs no GPU
gpu     = ["dep:wgpu", "dep:bytemuck", "dep:pollster"]

[dependencies]
wgpu     = { version = "22", optional = true }
bytemuck = { version = "1",  optional = true, features = ["derive"] }
pollster = { version = "0.3", optional = true }
```
Wrap all GPU code in `#[cfg(feature = "gpu")]`. Without it, the `Expr::GpuBlock` arm
returns `error: GPU backend not compiled in (rebuild with --features gpu)`. (Whether
`gpu` belongs in `default` is a policy call — keeping it off keeps CI and
GPU-less machines building cleanly.)

---

## 18. Implementation order

| Phase | Task | Est. |
|-------|------|------|
| 0 | `Cargo.toml` feature gate; `src/gpu/` skeleton; lazy `GpuContext` | 0.5d |
| 1 | Seams: `Token::Gpu`, `Expr::GpuBlock`, parser branch, REPL highlight | 0.5d |
| 2 | `GpuVal`/`GpuBuf`/upload-download (tensor); `Prec` plumbing | 1d |
| 3 | `GpuOp` enum + `shape_of` + `BufPool` (incl. `Loop`, `Stencil` variants) | 0.5d |
| 4 | `gpu_eval`: Num/Var/Block/BinOp/Unary (elementwise) | 0.5d |
| 5 | `dispatch`: single-encoder loop; elementwise + unary shaders | 1d |
| 6 | Wire the one `src/eval.rs` arm; validate captures; parity tests | 0.5d |
| 7 | Reduction shaders **with compensated summation** (§7 Tier 1); min/max | 1d |
| 8 | **Generic `Stencil` shader** → `ops.grad/div/curl/lap`, `forms.d/...` (§8) | 1.5d |
| 9 | **`Loop`/`iterate` residency**: ping-pong state, scan output (§10) | 1.5d |
| 10 | `matmul`, `shift`, `roll`, `lerp`, `clamp` | 1d |
| 11 | `Field` capture/upload/download + BC into stencil (§11) | 1d |
| 12 | df64 layer (§7 Tier 2) + `GPU f64 { }` modifier; symplectic parity | 1.5d |
| 13 | Stockham FFT → `fft/ifft/specgrad/poisson/invlap` (§9) | 2–3d |
| 14 | PIC: `gather`/`gathergrad`, then `scatter` atomics (§12) | 2d |
| 15 | `vm_lower` + `tensor((i,j)->…)` lowering (§15) | 2d |
| 16 | `tests.sh` parity + error-case coverage throughout | (ongoing) |

Phases 0–10 are a genuinely useful v1 (elementwise + reductions + differential
stencils + GPU-resident time-stepping). 11–15 are the high-value extensions; do them
in the order your actual workloads demand (most PDE work wants 9 then 13).

---

## 19. Testing strategy

All tests go in `tests.sh` following the existing pattern. The core discipline is
**GPU-vs-CPU parity with a precision-appropriate tolerance**:

```bash
run "cpu.lap"  "T = ...; sum(abs(ops.lap(T, dx)))"            "<value>"
run "gpu.lap"  "T = ...; GPU{ sum(abs(ops.lap(T, dx))) }"     "<value, f32 tol>"
```
- Use a loose tolerance for Tier-0 f32, a tight one for `GPU f64 { }` (Tier 2).
- **Test the loop residency explicitly:** an `iterate` of a diffusion/Verlet step on
  GPU should match the CPU result over N steps, and a symplectic run should conserve
  energy to the tier's tolerance (this is the test that catches a broken precision
  path — see §7).
- Error cases: lambda/cell capture rejected; undefined var; non-GPU-safe lambda
  inside the block is a hard error; FFT on a non-power-of-two axis (if v1 restricts).
```bash
run_err "gpu.err.fn"     'f = x -> x; GPU{ f + 1 }'
run_err "gpu.err.undef"  'GPU{ no_such_var }'
run_err "gpu.err.cell"   'c = cell(0); GPU{ get(c) }'
```
Per `CLAUDE.md`: every new GPU builtin gets tests here, plus README and `!help`
updates, with version bumps as features land.

---

## 20. Open questions

1. **df64 coverage.** Tier-2 (§7) is opt-in per block. Is a per-block `GPU f64 { }`
   modifier the right granularity, or should specific operators (Poisson, the Verlet
   update) always run df64 regardless? Leaning per-block for simplicity, with the
   spectral multiply internally df64.

2. **`Loop` unrolling vs CPU-driven submit (§10).** For known small `n`, record `n`
   passes into one encoder; for large/data-dependent `n`, submit the body in a CPU
   loop with cached bind groups. Pick a crossover (e.g. unroll ≤ 64). Measure.

3. **FFT scope (§9).** Power-of-two only for v1, or Bluestein from the start? P2 is
   far less code; most grids are already P2. Recommend P2 first.

4. **Async REPL.** `device.poll(Wait)` blocks the REPL thread, fine for now. If GPU
   blocks get long, spawn a worker thread.

5. **Error spans.** wgpu OOM/validation errors are opaque. A `GpuError` that carries
   the AST span would help, but needs spans in `src/ast.rs` first. Defer.

6. **Complex tensors on GPU.** `Val::ComplexTensor { re, im, shape }` is first-class
   (FFT, complex arithmetic, `!savetensor`). The FFT (§9) forces a decision here
   regardless: represent complex as either (a) `vec2<f32>` per element with paired
   ops, or (b) two real buffers (re, im) and synthesize complex ops from real ones.
   (b) composes better with the existing real shaders and the df64 layer; (a) is
   more cache-friendly. Recommend (b) for the FFT/spectral path; revisit if a
   general complex-elementwise need appears.

7. **Determinism.** f32 reductions and atomic scatter (§12) are run-to-run
   non-deterministic unless you force it (compensated/fixed-point accumulation,
   ordered reduce). Decide whether bit-reproducibility is a requirement; if so,
   prefer the fixed-point scatter and a deterministic tree reduce.
