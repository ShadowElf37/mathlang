# GPU Backend Considerations for mathlang

This document is written for a contributor unfamiliar with the project who will
implement the GPU compute backend. It covers project context, design rationale,
architecture, and a concrete implementation plan.

---

## 1. Project Overview

**mathlang** (`m`) is a REPL-based mathematical scripting language written in
Rust. Its primary use case is numerical/scientific computing: tensor math, PDEs,
FFTs, animations. The evaluator is a tree-walking interpreter (`src/eval.rs`).

Key source files:
- `src/lexer.rs`   — tokenizer
- `src/ast.rs`     — AST node types (`Expr`, `Op`, `Def`, `BlockStmt`)
- `src/parser.rs`  — recursive-descent parser
- `src/eval.rs`    — main evaluator (~3300 lines); defines `Val`, `Env`, all builtins
- `src/repl.rs`    — REPL with syntax highlighting, help text, Ctrl+D handling
- `src/animate.rs` — `animate2D` and `animate2D_raw` for 2D tensor animation

### Value types (`Val` in eval.rs):
```rust
enum Val {
    Num(f64),
    Complex(Complex<f64>),
    Tensor { data: Vec<f64>, shape: Vec<usize> },
    Lambda { params, body, captured_env },
    Tuple(Vec<Val>),
    Cell(Arc<RefCell<Val>>),   // mutable reference cell
}
```

### Language features relevant to GPU:
- Tensors are row-major flat `Vec<f64>`, any rank
- `tensor((i,j)->expr, rows, cols)` — builds tensors from index lambdas
- `shift(T, n, axis)` — edge-replicating shift (Neumann BCs)
- `roll(T, n, axis)` — circular shift
- `lerp(a, b, t)`, `clamp(x, lo, hi)` — elementwise
- `sum`, `mean`, `min`, `max`, `matmul` — reductions / linear algebra
- `exp`, `log`, `sqrt`, `sin`, `cos`, `abs`, `floor`, `ceil`, `round` — unary math

---

## 2. File Isolation Policy

**All GPU work must live in `src/gpu/`.** The only permitted touches to the
existing codebase are the minimum seams needed to connect the GPU backend:

| File | Allowed change | Why |
|------|---------------|-----|
| `src/ast.rs` | Add `Expr::GpuBlock { captures, body }` variant | AST must know the node exists |
| `src/lexer.rs` | Add `Token::Gpu` keyword | Lexer must tokenize `GPU` |
| `src/parser.rs` | Add one `Token::Gpu` branch in `parse_primary()` | Parser must produce the node |
| `src/eval.rs` | Add one `Expr::GpuBlock` match arm (~15 lines) | Evaluator must dispatch to GPU |
| `src/repl.rs` | Add `"GPU"` to keyword highlighter | REPL coloring |
| `Cargo.toml` | Add `wgpu`, `bytemuck`, `pollster` under `[features]` | Deps |

**Everything else — all shaders, all GPU evaluation logic, all memory management,
all pipeline caching — goes inside `src/gpu/`.** Your friend should be able to do
95% of the work without ever opening `eval.rs`, `parser.rs`, or `repl.rs`.

This means the GPU backend is also trivially removable: delete `src/gpu/`, remove
the six seam changes, and the project compiles without a trace of GPU code.

---

## 3. The GPU Block Syntax

```
GPU(var1, var2, ...) {
    intermediate = var1 * var2;
    intermediate + sum(var1)
}
```

- `GPU(...)` is a **keyword form**, not a function call — like `if(...)`. The parser
  needs a dedicated branch.
- The capture list names variables from the enclosing CPU environment.
  Only `Val::Num` (scalar) and `Val::Tensor` are permitted; anything else is a
  runtime error at block entry.
- The body is a standard mathlang block: semicolon-separated statements, last
  expression is the return value.
- The return value is downloaded back to the CPU as a `Val::Tensor` or `Val::Num`
  and is available in the enclosing CPU scope just like any other expression.
- Local bindings inside the block (e.g. `intermediate = ...`) are GPU-local and
  never visible on the CPU side.

(N.B.: there may not be a need for the vars to be specified for movement, maybe you can analyze the code to find which vars need moving and only move them, automatically - then the block looks like `GPU{code}`)

### Parser addition (ast.rs + parser.rs):
```rust
// ast.rs — new Expr variant:
GpuBlock { captures: Vec<String>, body: Box<Expr> }

// parser.rs — in parse_primary(), before the general Ident path:
Token::Gpu => {
    expect(LParen);
    let captures = parse_comma_separated_idents();
    expect(RParen);
    expect(LBrace);
    let body = parse_block();  // reuse existing block parser
    expect(RBrace);
    Expr::GpuBlock { captures, body: Box::new(body) }
}
```

Add `Token::Gpu` to the lexer as a reserved keyword (alongside `if`, `tensor`,
etc.).

---

## 4. Why Not Wait for Bytecode First?

The concern is "bouncing between GPU and line-by-line AST evaluation." This is a
real issue, but the framing slightly misidentifies the problem.

**The actual bottleneck:** multiple `queue.submit()` calls. Each submission
triggers a CPU↔GPU synchronization barrier. N operations → N barriers → N×(~0.5ms
overhead) even for trivial workloads.

**The solution — two-phase eval — does NOT require bytecode:**

| Phase | What happens |
|-------|-------------|
| **Phase 1: Record** | Walk the AST; allocate output buffers (shapes known via inference); record all commands into a single `wgpu::CommandEncoder`. No GPU execution yet. |
| **Phase 2: Dispatch** | One `queue.submit([encoder.finish()])`. One `device.poll(Wait)`. Download the final result buffer. |

This gives you exactly one synchronization point per GPU block, regardless of how
many operations are inside it. This is the right wgpu idiom and is achievable with
a direct AST walker.

**What bytecode would add (later):** operation fusion — recognizing that `a*b + c`
can be a single MAD kernel instead of two. This is a nice-to-have optimization.
It is *not* required for correct or reasonably fast execution.

**Recommendation: implement GPU now.** Design the `gpu_eval` internals around
two-phase record/dispatch so the interface is bytecode-agnostic. If bytecode ever
arrives, you swap the AST-walking recorder for a bytecode-to-GpuOp compiler;
the dispatch and everything downstream stays the same.

---

## 5. Architecture

### 4.1 File Structure

```
src/
  gpu/
    mod.rs          — re-exports; feature gate; lazy GpuContext init
    context.rs      — GpuContext: device, queue, pipeline cache
    val.rs          — GpuVal enum; GpuBuf struct; upload/download helpers
    ops.rs          — GpuOp enum (one variant per supported operation)
                      shape_of(op, inputs) — shape inference (no execution)
    eval.rs         — gpu_eval(): AST → Vec<GpuOp>  (Phase 1)
    dispatch.rs     — execute(Vec<GpuOp>, ctx) → GpuVal  (Phase 2)
    shaders/
      elementwise.wgsl   — add/sub/mul/div/pow via specialization constants
      unary.wgsl          — neg/exp/log/sqrt/sin/cos/abs/floor/ceil/round
      matmul.wgsl
      reduce.wgsl         — sum/mean/min/max with optional axis
      shift.wgsl          — edge-replicating, n-D
      roll.wgsl           — circular, n-D
      lerp_clamp.wgsl
      init_const.wgsl     — fill a buffer with a scalar constant
```

### 4.2 Core Types

```rust
// val.rs
pub struct GpuBuf {
    pub buf: wgpu::Buffer,   // STORAGE | COPY_SRC usage
    pub shape: Vec<usize>,
    pub len: usize,          // shape.iter().product()
}

pub enum GpuVal {
    Scalar(f64),             // kept on CPU; passed as wgpu push constant / uniform
    Buffer(GpuBuf),          // lives on GPU until block exit
}
```

### 4.3 GpuContext

```rust
// context.rs
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    // Compiled pipelines are expensive (~50–200ms first time); cache aggressively.
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
}

impl GpuContext {
    pub fn pipeline(&mut self, name: &'static str, wgsl: &str) -> &wgpu::ComputePipeline {
        self.pipelines.entry(name).or_insert_with(|| compile(wgsl, &self.device))
    }
}
```

Lazy-initialize with `OnceLock<Mutex<GpuContext>>`. If no wgpu adapter is found,
the GPU block returns a clear runtime error:
```
error: GPU block requires a compatible GPU; no adapter found on this system.
```

### 4.4 GpuOp — the intermediate representation

```rust
// ops.rs
pub enum GpuOp {
    Upload   { src: Vec<f64>, shape: Vec<usize>, dst: BufId },
    Elemwise { op: ElemOp, lhs: GpuArg, rhs: GpuArg, dst: BufId, shape: Vec<usize> },
    Unary    { op: UnaryOp, src: GpuArg, dst: BufId, shape: Vec<usize> },
    MatMul   { a: BufId, b: BufId, dst: BufId, m: usize, k: usize, n: usize },
    Reduce   { op: ReduceOp, src: BufId, axis: Option<usize>, dst: BufId, out_shape: Vec<usize> },
    Shift    { src: BufId, n: i64, axis: usize, dst: BufId, shape: Vec<usize> },
    Roll     { src: BufId, n: i64, axis: usize, dst: BufId, shape: Vec<usize> },
    Lerp     { a: GpuArg, b: GpuArg, t: GpuArg, dst: BufId, shape: Vec<usize> },
    Clamp    { src: GpuArg, lo: f64, hi: f64,   dst: BufId, shape: Vec<usize> },
    // ... extend freely
}

pub enum GpuArg { Buf(BufId), Scalar(f64) }  // scalar broadcast is the common case
pub type BufId = u32;                          // index into a flat Vec<Option<GpuBuf>>
```

`ops.rs` also implements `shape_of(op, input_shapes) -> Vec<usize>` — pure shape
inference, no GPU calls. This is Phase 1: the eval walk calls `shape_of` to know
what buffer to pre-allocate for each `dst`.

### 4.5 gpu_eval (Phase 1 — record)

```rust
// eval.rs  (inside src/gpu/)
pub fn gpu_eval(
    expr: &Expr,
    env: &GpuEnv,          // HashMap<String, GpuVal>
    ops: &mut Vec<GpuOp>,  // append here
    bufs: &mut BufPool,    // allocator for BufIds
) -> Result<GpuVal, String>
```

This is a straightforward AST match:
- `Expr::Num(f)` → `GpuVal::Scalar(f)`
- `Expr::Var(name)` → look up in `env`
- `Expr::BinOp(l, op, r)` → recurse, emit `GpuOp::Elemwise`
- `Expr::Neg(e)` → emit `GpuOp::Unary { op: UnaryOp::Neg, ... }`
- `Expr::Apply(func, args)` where `func` is a known GPU builtin name → dispatch
- `Expr::Block(stmts)` → fold over stmts, extend `env`, return last value
- anything else → `Err("... not supported in GPU block")`

Note: user-defined functions, lambdas, cells, conditionals, and apply-to-tensor
(implicit map) are **not supported** in v1. Return a clear error.

### 4.6 dispatch (Phase 2 — execute)

```rust
// dispatch.rs
pub fn dispatch_all(
    ops: &[GpuOp],
    ctx: &mut GpuContext,
) -> Result<GpuVal, String>
```

1. Allocate all output `wgpu::Buffer`s upfront (sizes known from shape inference).
2. Create one `CommandEncoder`.
3. For each `GpuOp`, record a `ComputePass` into the encoder.
4. `queue.submit([encoder.finish()])`.
5. `device.poll(wgpu::Maintain::Wait)`.
6. Return the final `GpuVal` (download its buffer if needed).

This is the single synchronization point for the entire GPU block.

### 4.7 CPU eval integration (src/eval.rs)

Add one match arm:
```rust
Expr::GpuBlock { captures, body } => {
    // 1. Validate and collect captures from CPU env
    let gpu_env: HashMap<String, GpuVal> = captures.iter().map(|name| {
        match env.vars.get(name) {
            Some(Val::Num(f))    => Ok((name.clone(), GpuVal::Scalar(*f))),
            Some(Val::Tensor{data, shape}) => Ok((name.clone(), GpuVal::Buffer(
                gpu_upload(data, shape, ctx)? // creates staging buffer, copies
            ))),
            Some(other) => Err(format!("GPU capture `{}`: only tensors and scalars \
                                        are allowed; got {}", name, fmt_val(other))),
            None => Err(format!("undefined variable `{}`", name)),
        }
    }).collect::<Result<_, _>>()?;

    // 2. Record ops
    let mut ops  = Vec::new();
    let mut bufs = BufPool::new();
    let result   = gpu::eval::gpu_eval(body, &gpu_env, &mut ops, &mut bufs)?;

    // 3. Dispatch everything at once, download result
    let gpu_ctx = GPU_CTX.get_or_init(|| ...);
    let final_val = gpu::dispatch::dispatch_all(&ops, &mut gpu_ctx.lock().unwrap())?;
    gpu_to_cpu_val(final_val)  // download GpuBuf → Val::Tensor, or GpuVal::Scalar → Val::Num
}
```

This is the **only** change to `src/eval.rs`. Everything else lives in `src/gpu/`.

---

## 6. Shader Design Philosophy

**The main contribution after infrastructure is shipping.** Once the plumbing works,
adding a new GPU builtin is:
1. Write `src/gpu/shaders/new_op.wgsl`
2. Add a `GpuOp::NewOp { ... }` variant in `ops.rs`
3. Add `shape_of` case in `ops.rs`
4. Add a `dispatch_new_op()` call in `dispatch.rs`
5. Add a match arm in `gpu/eval.rs`

Keep it easy to extend. No magic, no macros needed.

### Shader template

All shaders follow the same pattern:
```wgsl
// elementwise add example
@group(0) @binding(0) var<storage, read>  a:   array<f32>;
@group(0) @binding(1) var<storage, read>  b:   array<f32>;
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

**Use specialization constants** (wgpu supports them via `PipelineCompilationOptions`)
to parametrize operations, or just have separate shaders per op — both are fine at
this scale.

**Use `f32` in WGSL** even though mathlang uses `f64` internally. Most consumer GPUs
don't support `f64` compute. Tensors should be converted `f64 → f32` on upload and
`f32 → f64` on download. For scientific computing this is acceptable; document it.
If `f64` support is ever needed, check for the `shader-f64` feature and enable it
conditionally.

---

## 7. GPU-Allowed Builtins (v1 scope)

### Supported from day one:
| Category | Builtins |
|----------|---------|
| Elementwise arithmetic | `+`, `-`, `*`, `/`, `^` (scalar-scalar, scalar-tensor, tensor-tensor) |
| Unary math | `-`, `exp`, `log`, `sqrt`, `sin`, `cos`, `abs`, `floor`, `ceil`, `round` |
| Reductions | `sum`, `mean`, `min`, `max` (whole-tensor and per-axis forms) |
| Linear algebra | `matmul` |
| Tensor manipulation | `shift`, `roll`, `lerp`, `clamp` |
| Constructors | scalar literals, `zeros(m,n)`, `ones(m,n)` |
| Comparisons | `<`, `>`, `<=`, `>=`, `==`, `!=` (return 0.0/1.0 tensor) |

### Explicitly NOT supported in v1:
- User-defined functions (`f(x) = ...`)
- Lambdas
- `cell`, `get`, `set` (mutable state)
- `if` (conditional — non-trivial on GPU; add later with mask-based select)
- `tensor((i,j)->expr, m, n)` — deferred (requires JIT or precompiled index lambda)
- `fft`, `ifft` — wgpu has no FFT; would need a GPU FFT library
- `animate2D` — orthogonal concern; GPU block just returns a tensor

### Future additions (easy to add once infra exists):
- `transpose(T)` — trivial shader
- `conv2D(T, kernel)` — useful for ML/signal processing
- `cumsum`, `diff` — scan patterns
- `norm(T)` — `sqrt(sum(T^2))`
- `solve(A, b)` — would need GPU LU or CG solver

---

## 8. REPL Syntax Highlighting

The keyword `GPU` and its delimiters `(`, `)`, `{`, `}` should render in **bold
red** in the REPL. This signals to the user that they are entering a different
execution context.

In `src/repl.rs`, the `Highlighter` impl (rustyline's `Highlighter` trait) already
colors keywords. Extend it:
- Add `"GPU"` to the keyword list with style `Bold + Red` (use `nu_ansi_term` or
  the escape codes already in use).
- Track brace/paren depth inside a GPU block to also color the matching `{`, `}`,
  `(`, `)` in red. This is optional but nice.

---

## 9. Cargo.toml

Add a feature gate so the GPU backend is optional (useful for CI or systems without
a GPU):

```toml
[features]
default = ["gpu"]
gpu = ["dep:wgpu", "dep:bytemuck", "dep:pollster"]

[dependencies]
wgpu     = { version = "22", optional = true }
bytemuck = { version = "1",  optional = true, features = ["derive"] }
pollster = { version = "0.3", optional = true }
```

Wrap all GPU code in `#[cfg(feature = "gpu")]`. Without the feature, the
`Expr::GpuBlock` match arm returns:
```
error: GPU backend not compiled in (rebuild with --features gpu)
```

---

## 10. Memory Model

### Upload (CPU → GPU)
At GPU block entry, each captured `Val::Tensor` is uploaded via a staging buffer:
1. Create `wgpu::Buffer` with `COPY_DST | COPY_SRC | STORAGE` usage.
2. Create a temporary staging buffer with `MAP_WRITE | COPY_SRC`.
3. Write `f64 → f32` converted data into staging buffer (via `map_async` + `device.poll`).
4. Record `copy_buffer_to_buffer` command.

The actual copy happens when the encoder is submitted in Phase 2. All uploads are
recorded in Phase 1 as the first `GpuOp::Upload` entries.

### Download (GPU → CPU)
Only the final result buffer is downloaded. Intermediate buffers stay on the GPU
throughout the block. Download:
1. Record `copy_buffer_to_buffer` to a `MAP_READ` staging buffer.
2. After `queue.submit`, `staging.map_async(MapMode::Read, ...)`.
3. `device.poll(Wait)`.
4. Read `f32 → f64` back into `Vec<f64>`.

### O(1) intermediate results
Since all intermediate buffers live on the GPU and are only referenced by `BufId`
inside the `Vec<GpuOp>`, there is no CPU allocation per operation. The total
CPU-side memory for a GPU block is `O(captures + result)` regardless of how many
intermediate tensors are computed.

---

## 11. Interaction with TODO #5 (Bytecode)

If bytecode compilation is eventually added to mathlang, the GPU backend benefits
from it in one specific way: **operation fusion**. A bytecode pass could recognize
patterns like `a * b + c` and emit a single `FusedMAD` GpuOp rather than two
separate ones.

However, this is an optimization pass, not a prerequisite. The current two-phase
design (AST → Vec<GpuOp> → single submit) already gives you:
- One CPU→GPU synchronization per GPU block
- Zero intermediate downloads
- Pipeline reuse across multiple GPU block executions (cached by `GpuContext`)

When/if bytecode lands, you add a fusion pass between Phase 1 and Phase 2:
```
AST → [gpu/eval.rs] → Vec<GpuOp> → [fusion pass, optional] → dispatch
```
The dispatch layer is unchanged.

---

## 12. Implementation Order

Work through these phases in order. Each phase is independently testable.

| Phase | Task | Est. |
|-------|------|------|
| 0 | Cargo.toml feature gate; skeleton `src/gpu/` with empty modules | 0.5d |
| 1 | Lexer (`Token::Gpu`); AST (`Expr::GpuBlock`); Parser; REPL highlighting | 0.5d |
| 2 | `GpuContext` lazy init; `GpuVal`/`GpuBuf`; upload/download helpers | 1d |
| 3 | `GpuOp` enum; `BufPool`; `shape_of()` for all ops | 0.5d |
| 4 | `gpu_eval()` skeleton: Num, Var, Block, BinOp (elemwise only) | 0.5d |
| 5 | `dispatch.rs`: single-encoder loop, elementwise shaders | 1d |
| 6 | Wire into `src/eval.rs` — one match arm, validate captures, call | 0.5d |
| 7 | Unary shaders, reduction shaders | 1d |
| 8 | `matmul.wgsl`, `shift.wgsl`, `roll.wgsl`, `lerp_clamp.wgsl` | 1d |
| 9 | Tests (`tests.sh`): verify GPU block output matches CPU for each op | 1d |

**Total estimated: ~7 developer-days** for a solid v1 covering the full op set
above.

---

## 13. Testing Strategy

All tests go in `tests.sh` following the existing pattern:
```bash
expect "GPU(a,b){ a + b }" "..." # with a and b defined as tensors
```

For GPU vs CPU parity, run both and compare (with tolerance for f32 rounding):
```bash
expect "GPU(T){ sum(T) }"         "some_value"
expect "sum(T)"                   "same_value"
```

Also test error cases:
```bash
expect_err 'GPU(f){ f + 1 }'  "only tensors and scalars"   # f is a lambda
expect_err 'GPU(){ no_such }' "undefined variable"
```

---

## 14. Open Questions

1. **f64 vs f32:** Document the precision loss explicitly. Consider a `--gpu-f64`
   flag that checks for the `shader-f64` wgpu feature and errors if unavailable.

2. **Multi-GPU:** `wgpu` supports adapter selection. For now, pick the first
   high-performance adapter. Defer multi-GPU.

3. **Async REPL:** The current REPL is synchronous. `device.poll(Wait)` blocks
   the REPL thread, which is fine for now. If GPU blocks become long-running,
   consider spawning a thread.

4. **`tensor((i,j)->expr, m, n)` on GPU:** Requires either (a) compiling the lambda
   to WGSL (hard), or (b) evaluating the lambda on the CPU and uploading the
   result (defeats the purpose). Defer to after bytecode.

5. **Error propagation inside GPU blocks:** wgpu errors (buffer overflow, OOM)
   surface as panics or opaque errors. Add a `GpuError` type that wraps wgpu
   errors with source location from the AST span (requires adding spans to AST
   first — see `src/ast.rs`).

6. **Complex tensor support on the GPU (NB):** The language has a first-class
   `Val::ComplexTensor { re, im, shape }` type that is used transparently by FFT,
   complex arithmetic, and now `!savetensor`/`!loadtensor`. The current GPU backend
   design (sections 1–13) assumes only real `f64` (promoted to `f32`) tensors.
   Supporting complex tensors on the GPU will require further design work — likely
   one of: (a) representing complex numbers as `vec2<f32>` in WGSL and emitting
   paired operations, or (b) splitting re/im into two real GPU buffers and
   synthesising complex ops from real ones. Neither is trivial, and both require
   the WGSL codegen layer to be aware of the `ComplexTensor` type distinction.
   **Do not attempt to add complex GPU support until the real-tensor backend is
   working end-to-end and bytecode compilation (TODO #1) is underway.** At that
   point the right representation will be much clearer.
