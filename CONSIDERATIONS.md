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
    Complex(f64, f64),
    Tensor { data: TData, shape: Vec<usize> },          // TData = Arc<Vec<f64>>, O(1) clone
    ComplexTensor { re: TData, im: TData, shape: Vec<usize> },
    /// Fn(params, body, captured_env, bytecode_cache)
    /// bytecode_cache is Arc<OnceLock<Option<Vec<Instruction>>>> — shared across clones,
    /// compiled once on first call via apply_fn_direct(), None = fall back to tree-walk.
    Fn(Vec<String>, Expr, Arc<HashMap<String, Val>>, Arc<OnceLock<Option<Vec<Instruction>>>>),
    Builtin(String),
    Tuple(Vec<Val>),
    Cell(Arc<RefCell<Val>>),   // mutable reference cell; identity semantics on clone
}
```

### Bytecode VM (`src/eval.rs`):

mathlang has a stack-based bytecode VM for user-defined function bodies.
`compile_fn(params, body, captured)` compiles an `Expr` to `Vec<Instruction>` or
returns `None` for unsupported nodes (slices, ranges, tensor literals, recursive
functions). On first call, `apply_fn_direct()` uses `OnceLock::get_or_init()` to
compile once and cache; all `Val::Fn` clones share the same `Arc<OnceLock>`.
Subsequent calls run the VM directly. `map`, `filter`, `integral`, `sum(f,lo,hi)`
pre-evaluate `f` before their loops so the cache is shared across all N iterations.
Tensor indexing (`T[i]`, `T[i,j]`), nested lambdas (`MakeClosure`), and
non-recursive `Def::Func` in blocks all compile to bytecode. Recursive local
functions fall back silently to the tree-walk evaluator.

The `Instruction` enum is currently defined in `eval.rs`. It is also the natural IR
for GPU lambda lowering — see section 15.

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
GPU {
    intermediate = var1 * var2;
    intermediate + sum(var1)
}
```

- `GPU` is a **keyword form**, not a function call. The parser needs a dedicated branch.
- Free variables from the enclosing CPU scope are **automatically detected** by
  walking the block body AST (same approach as `collect_free_vars` in the bytecode
  VM). Only `Val::Num` (scalar) and `Val::Tensor` are permitted; anything else is a
  runtime error at block entry.
- The body is a standard mathlang block: semicolon-separated statements, last
  expression is the return value.
- The return value is downloaded back to the CPU as a `Val::Tensor` or `Val::Num`
  and is available in the enclosing CPU scope just like any other expression.
- Local bindings inside the block (e.g. `intermediate = ...`) are GPU-local and
  never visible on the CPU side.

### Parser addition (ast.rs + parser.rs):
```rust
// ast.rs — new Expr variant:
GpuBlock { body: Box<Expr> }

// parser.rs — in parse_primary(), before the general Ident path:
Token::Gpu => {
    expect(LBrace);
    let body = parse_block();  // reuse existing block parser
    expect(RBrace);
    Expr::GpuBlock { body: Box::new(body) }
}
```

Add `Token::Gpu` to the lexer as a reserved keyword (alongside `if`, `tensor`,
etc.).

---

## 4. Bytecode and GPU: the Relationship

**The two-phase design does NOT require bytecode** — `gpu/eval.rs` walks the AST
directly to build `Vec<GpuOp>`, then `dispatch.rs` submits once. This is still the
right approach for the GPU block body.

**What bytecode enables on top of the GPU block:**

1. **Operation fusion.** A bytecode pass can recognize `a*b + c` and emit a single
   `FusedMAD` GpuOp instead of two. Optional optimization; not required for v1.

2. **Lambda-to-WGSL lowering.** The bytecode VM's `Vec<Instruction>` is a flat,
   typed IR that maps 1-to-1 to WGSL for the GPU-safe subset of operations. This
   enables `tensor((i,j)->expr, m, n)` on the GPU (previously blocked, now
   achievable — see section 15).

**Recommendation: implement GPU block now.** Design `gpu/eval.rs` as a direct AST
walker. The bytecode layer plugs in later (a) as a fusion pass between Phase 1 and
Phase 2, and (b) as the lambda-to-WGSL lowerer for index lambdas. Both additions
are surgically local to `src/gpu/`.

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
Expr::GpuBlock { body } => {
    // 1. Walk body to find free variables; validate each against CPU env.
    //    Only Val::Num and Val::Tensor are allowed; anything else is a runtime error.
    let free_names = collect_gpu_free_vars(body, env); // AST walk, same idea as collect_free_vars
    let gpu_env: HashMap<String, GpuVal> = free_names.iter().map(|name| {
        match env.vars.get(name) {
            Some(Val::Num(f))    => Ok((name.clone(), GpuVal::Scalar(*f))),
            Some(Val::Tensor{data, shape}) => Ok((name.clone(), GpuVal::Buffer(
                gpu_upload(data, shape, ctx)? // creates staging buffer, copies
            ))),
            Some(other) => Err(format!("GPU: `{}`: only tensors and scalars \
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

`collect_gpu_free_vars` is a simple AST walk that returns all `Expr::Var` names not
bound locally within the block. It can reuse or adapt `expr_contains_var` from
`eval.rs`. This is the **only** change to `src/eval.rs`; everything else lives in
`src/gpu/`.

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

## 11. Bytecode VM — Current State and GPU Interface

The bytecode VM is fully implemented in `src/eval.rs` (merged in v0.17.0).

### What compiles to bytecode

- **Arithmetic/logic**: `PushNum`, `PushComplex`, `BinOp`, `Neg`, `JumpIfFalse`/`Jump` (`if`)
- **Variables**: `LoadParam`, `LoadLocal`, `LoadCaptured`; captured scalar constants
  inlined as `PushNum`/`PushComplex`
- **Calls**: `CallBuiltin` (named builtins), `CallVal` (computed callees)
- **Collections**: `MakeTuple`, `MakeArray`
- **Block locals**: `StoreLocal`/`LoadLocal`; `Def::Var` and non-recursive `Def::Func`
- **Tensor indexing**: `Index` — handles `T[i]`, `T[i,j]`, negative indices,
  sub-tensor extraction; rejects slice expressions (fall back)
- **Nested lambdas**: `MakeClosure` — `collect_free_vars()` walks the inner body to
  find outer params/locals to snapshot; inner code eagerly pre-compiled via
  `Val::make_fn_compiled()`; `expr_contains_var()` detects recursive functions and
  falls back to tree-walk (which sets up the self-reference correctly)
- **Hot-loop caching**: `map`, `filter`, `integral`, `sum(f,lo,hi)` pre-evaluate `f`
  before their loops — OnceLock cache shared across all N calls

### What falls back to tree-walk (transparently, no behavior change)

- Slices (`T[lo..hi]`), ranges (`lo..hi`), tensor literals (`(1,2; 3,4)`)
- Recursive local functions (`f(x) = ... f(x-1) ...` inside a block)
- Special forms that require unevaluated Expr args: `sum`, `prod`, `integral`,
  `deriv`, `map`, `filter`, `reduce` (these are handled at the `eval` level with
  hot-loop f-caching, not compiled into the VM body)

### Pipeline with bytecode

```
AST → [gpu/eval.rs]  → Vec<GpuOp>  → [fusion pass, optional] → dispatch
                   ↑
             for lambda args:
    compile_fn() → Vec<Instruction>
         → gpu/vm_lower.rs → WGSL inline body
         → GpuOp::TensorFromLambda { wgsl_body, ... }
```

The dispatch layer is unchanged by both the fusion pass and the lambda lowering.

### Module placement note

`Instruction` currently lives in `eval.rs`. When `eval.rs` is eventually split
(planned but deferred), `Instruction` should move to `src/vm.rs` or `src/bytecode.rs`
so `src/gpu/` can import it without depending on the entire evaluator. For now,
`src/gpu/vm_lower.rs` can import `crate::eval::Instruction` directly — it compiles
fine within a single crate.

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
run "gpu.add"  "GPU{ a + b }"  "..."   # with a and b defined as tensors
```

For GPU vs CPU parity, run both and compare (with tolerance for f32 rounding):
```bash
run "gpu.sum"    "GPU{ sum(T) }"  "some_value"
run "cpu.sum"    "sum(T)"         "same_value"
```

Also test error cases:
```bash
run_err "gpu.err.fn"      'f = x -> x; GPU{ f + 1 }'   # f is a lambda — not allowed
run_err "gpu.err.undef"   'GPU{ no_such_var }'           # undefined variable
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

4. **`tensor((i,j)->expr, m, n)` on GPU:** Previously blocked; now achievable via
   bytecode lowering (see section 15). The lambda body is compiled to
   `Vec<Instruction>` by `compile_fn()`, then lowered to a WGSL inline function by
   `gpu/vm_lower.rs`. The GPU dispatcher emits a `TensorFromLambda` kernel.
   Implement in v2 (after the real-tensor GPU block works end-to-end).

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

---

## 15. Bytecode as GPU IR — Technical Spec

This section specifies how the bytecode VM's `Instruction` enum serves as the IR
bridge for GPU lambda lowering.

### GPU-safe subset of Instructions

Not all Instructions can be lowered to WGSL. The GPU-safe subset:

| Instruction | WGSL | Notes |
|---|---|---|
| `PushNum(f)` | `f32(f)` literal | ✓ |
| `PushComplex` | — | ✗ No complex on GPU |
| `LoadParam(i)` | param variable | ✓ |
| `LoadCaptured(name)` | uniform/constant | ✓ only if val is scalar or known-shape tensor |
| `BinOp(Add\|Sub\|Mul\|Div)` | `+`, `-`, `*`, `/` | ✓ |
| `BinOp(Pow)` | `pow(a, b)` | ✓ |
| `BinOp(Lt\|Gt\|LtEq\|GtEq\|Eq\|Ne)` | comparison → `f32` 0/1 | ✓ |
| `Neg` | unary `-` | ✓ |
| `CallBuiltin("sin"\|"cos"\|"exp"\|"sqrt"\|"abs"\|"floor"\|"ceil"\|"round"\|"log", 1)` | WGSL builtin | ✓ |
| `JumpIfFalse(t)` / `Jump(t)` | `if/else` block | ✓ |
| `StoreLocal(s)` / `LoadLocal(s)` | `var x: f32 = ...;` | ✓ |
| `Pop` | discard | ✓ |
| `Index` | — | ✗ Requires CPU tensor ref; use `LoadBuffer` (future) in GPU context |
| `MakeTuple(n)` / `MakeArray(n)` | — | ✗ Runtime heap allocation |
| `MakeClosure(...)` | — | ✗ No nested functions in WGSL; recursion impossible on GPU |
| `CallVal(...)` | — | ✗ Dynamic dispatch impossible |
| `CallBuiltin(special form, ...)` | — | ✗ sum, map, etc. are not scalar |

A lambda body is "GPU-safe" if its compiled `Vec<Instruction>` contains only
instructions from the ✓ column. Check function:
```rust
pub fn is_gpu_safe(code: &[Instruction]) -> bool {
    code.iter().all(|inst| matches!(inst,
        Instruction::PushNum(_) | Instruction::LoadParam(_) |
        Instruction::LoadCaptured(_) | Instruction::BinOp(_) | Instruction::Neg |
        Instruction::JumpIfFalse(_) | Instruction::Jump(_) |
        Instruction::StoreLocal(_) | Instruction::LoadLocal(_) |
        Instruction::Pop | Instruction::Return |
        Instruction::CallBuiltin(name, 1) if GPU_SAFE_UNARY.contains(name.as_str()) |
        Instruction::CallBuiltin(name, 2) if GPU_SAFE_BINARY.contains(name.as_str())
    ))
}
```

### `gpu/vm_lower.rs` — lowering to WGSL

```rust
// src/gpu/vm_lower.rs
pub fn lower_to_wgsl(
    code:        &[Instruction],
    param_names: &[&str],          // WGSL names for LoadParam(i)
    n_locals:    usize,            // pre-declare this many f32 vars
    captured:    &HashMap<String, GpuScalar>, // captured scalars → WGSL constants
) -> Result<String, String>
```

The lowering is a second pass over the linear instruction sequence. Since the code
is already in SSA-like form (stack), convert to a register-named form:
- Each push to the stack produces a fresh `let t_N: f32 = ...;` WGSL statement
- `StoreLocal(s)` emits `local_s = t_N;`
- `JumpIfFalse` + `Jump` pairs become `if (t_N != 0.0) { ... } else { ... }`

The result is an inline WGSL function body that can be embedded in a
`TensorFromLambda` kernel.

### `tensor((i,j)->expr, m, n)` on GPU — the full path

```
User writes:   GPU { tensor((i,j) -> sin(T[i,j]) + i*j, rows(T), cols(T)) }
                                                           ↑ T[i,j] — not GPU-safe in v1
                                                           (Index instruction rejected by is_gpu_safe)

Simpler form:  GPU { tensor((i,j) -> i*j, 4, 4) }
                                    ↑ GPU-safe: only LoadParam(0), LoadParam(1), BinOp(Mul)
```

Path for GPU-safe lambda:
1. `gpu/eval.rs` hits `Expr::Apply(Var("tensor"), [Lambda(...), m, n])`
2. Evaluate m, n → `GpuVal::Scalar`
3. Extract lambda's `Val::Fn(params, body, captured, cache)`
4. `cache.get_or_init(...)` → `Some(code)`; call `is_gpu_safe(code)`
5. If safe: `gpu/vm_lower::lower_to_wgsl(code, &params, ...)` → WGSL body string
6. Emit `GpuOp::TensorFromLambda { wgsl_body, m, n, dst }`
7. Dispatch: compile a `tensor_from_lambda.wgsl` template with the body inlined;
   dispatch `m*n` threads, each thread writes `out[i*n+j] = f(i, j)`

Add to `ops.rs`:
```rust
GpuOp::TensorFromLambda {
    wgsl_body: String,      // lowered from bytecode
    params:    Vec<String>, // param names used in the body
    m:         GpuArg,
    n:         GpuArg,
    dst:       BufId,
}
```

If `is_gpu_safe` returns false: **hard error**. The user wrote a GPU lambda
expecting GPU execution; silently falling back to CPU inside a `GPU{...}` block
masks performance bugs. Fix the lambda to use only GPU-safe operations (see table
above) or move the computation outside the `GPU{...}` block.

### Note on `T[i,j]` inside GPU lambdas

Tensor indexing (`Expr::Index`) is currently uncompilable by the bytecode VM. For
GPU lambdas that need `T[i,j]`, the approach is:
- Pass `T` as a captured `LoadCaptured("T")` which maps to a GPU buffer binding
- Emit a `LOAD_BUFFER` instruction (future addition to Instruction enum) that reads
  a specific element of a captured GPU buffer

This is a v3 addition. For v2, restrict to lambdas with numeric-only captured vars.
