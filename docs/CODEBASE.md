# Mathlang Codebase Structure

## Top-level layout

```
src/
  main.rs          — CLI entry point; dispatches to REPL or file mode
  lexer.rs         — Tokenizer (Token enum, Lexer struct)
  ast.rs           — AST nodes: Expr, Def, BlockStmt, Assign/LValue/PathSeg,
                     TypeHint, Param, Field, Op
  parser.rs        — Recursive descent parser (Parser struct)
  eval.rs          — Evaluator, builtins, VM, type inference (largest file ~5000 lines)
  ns/              — Standard namespaces (`.` access), each in its own module:
    mod.rs         — register_all (called by Env::new), routing to new-function dispatch
    ops.rs   — grad/div/curl/lap/poisson/invlap/specgrad (finite-diff + spectral);
                     also field-polymorphic when the first arg is a Val::Field
    solver.rs      — rk4/odeint/verlet(symplectic)/tao(nonsep symplectic)/cfl time integrators
    forms.rs       — field() constructor (data + function forms) + exterior calculus
                     (d/hodge/wedge/raise/lower/codiff/laplace) on Val::Field; metric-aware
    pic.rs         — particle-in-cell scatter(deposit)/gather(interpolate); adjoint
                     ngp/cic/tsc shape functions over a Val::Field grid
    special.rs bits.rs stats.rs linalg.rs vec.rs — relocated niche builtins (membership lists)
  repl.rs          — REPL loop, bang commands, syntax highlighting, tab completion
  graph.rs         — !graph command: sample → PNG via plotters → open in animator (bare RGB mode)
  animate.rs       — !animate2D / !animate2D_raw: stream MXFR frames to wgpu_animator

animator/          — Separate wgpu-based animation window (standalone binary: wgpu_animator)
  src/
    main.rs        — CLI (clap), event loop setup
    app.rs         — ApplicationHandler: stdin polling, frame buffering, zoom/pan/keyboard
    renderer.rs    — wgpu render pipeline (R32Float scalar or Rgba32Float RGB texture)
    stdin_reader.rs — Background thread reading MXFR frames from stdin
    colormap.rs    — Colormap LUTs (heat, inferno, viridis, rdbu, grayscale)
    norm.rs        — Normalization modes (global, per-frame, percentile, fixed)
    interp.rs      — Interpolation modes (nearest, linear, bicubic)
    ui.rs          — egui overlay (colorbar, axes, zoom/pan controls)
    shaders/data.wgsl — WGSL fragment shader: scalar→colormap LUT or RGB passthrough

examples/          — Example .math scripts
docs/              — This file and other documentation
```

## src/ast.rs

Defines all AST node types. Key enums:

- **`Expr`** — expression nodes: `Num`, `ImagLit`, `Var`, `BinOp`, `Neg`, `Not`, `Apply`, `Lambda`, `Tuple`, `Record`, `Array`, `TensorLit`, `Index`, `Member`, `Slice`, `Range`, `Block`, `GpuBlock`, `Named`, `Splat`
- **`Def`** — `Var(name, expr)` or `Func(name, params, ret_hint, body)`
- **`BlockStmt`** — `Def(Def)`, `Assign(Assign)`, or `Expr(Expr)` (statements inside `{...}` and at the top level)
- **`TypeHint`** — `Any | Num | Real | Complex | Int | Nat | Tensor | RealTensor | ComplexTensor | Fn | Cell | Tuple`
- **`Param`** — `{ name, hint: Option<TypeHint>, default: Option<Expr> }`
- **`Field`** — one item of a record literal: `{ name: Option<String>, value: Expr, private: bool, func: bool }`
- **`Assign` / `LValue` / `PathSeg`** — `root[i].f = v`: a root name plus `Index(Expr)` / `Field(String)` steps, with `op: Option<Op>` for the compound forms

### Two nodes that are only legal in a list

`Splat(e)` (`..x`) and `Named(name, e)` (`x = 1`) are argument/item wrappers, not
standalone expressions. Wrapping rather than changing `Apply`'s or `Tuple`'s
shape keeps every generic traversal working; evaluating one directly is an
error, and the parser only produces them where they mean something —
`Splat` in paren lists, array literals and call arguments (never inside `T[…]`,
where `..` is a slice); `Named` only in call arguments.

## src/lexer.rs

`Lexer::new(src).tokenize()` → `Vec<Token>`.  Token variants include: `Num(f64)`, `Imag(f64)`, `Ident(String)`, `Arrow`, `Colon`, `LParen`/`RParen`, `LBracket`/`RBracket`, `LBrace`/`RBrace`, `Comma`, `Semicolon`, `DotDot`, `Bang`, `Eq`/`EqEq`/`BangEq`, `PlusEq`/`MinusEq`/`StarEq`/`SlashEq`, arithmetic operators, `Eof`.

## src/parser.rs

`Parser::new(toks).parse_repl()` → `Vec<BlockStmt>`.

Key parsing methods:

- **`parse_repl()`** — parses one REPL line into `BlockStmt`s (defs, assignments and expressions interleaved, separated by `;`)
- **`parse_def()`** — `name = expr` (Var) or `name(params) [: ret] = expr` (Func)
- **`is_assign_start()` / `parse_assign()`** — an assignment statement: a root name, a balanced path of `[…]`/`.name` steps, then `=` or a compound `+= -= *= /=`. Distinct from `is_def_start`, which owns the bare `name = …` and `name(params) = …` forms
- **`parse_index_brackets()`** — `[ … ]` → the index expression; shared by `postfix` (reads) and `parse_assign` (writes) so both resolve indices identically
- **`parse_call_arg()`** — one call argument: `..x` splat, `name = expr` named argument, or positional
- **`parse_paren_item()`** — one paren-list item: `..x` splat, a `private`-marked definition, a named field, or positional
- **`expr()`** / **`cmp()`** / **`add()`** / **`mul()`** / **`pow()`** / **`postfix()`** / **`primary()`** — operator precedence chain
- **`primary()`** handles: numeric literals, imaginary literals, `(...)` tuples/lambdas/matrices, `[...]` array/matrix literals, `{...}` blocks, identifiers (variables, function calls, single-arg lambdas `x ->`, typed single-arg lambdas `x: type ->`)
- **`looks_like_paren_lambda()`** / **`is_multi_lambda()`** — lookahead helpers that determine whether to parse as lambda vs tuple/call
- **`parse_param()`** — `name [: type_hint] [= default]`
- **`skip_param_default()`** — lookahead helper so the paren-lambda scan steps over a default

Lambda forms supported:
- `x -> body` — bare single-arg
- `x: type -> body` — bare single-arg with type hint
- `(x, y) -> body` — paren multi-arg
- `(x: type, y) -> body` — paren multi-arg with hints
- `x, y -> body` — multi-arg without parens (via `is_multi_lambda`); no defaults in this form
- `() -> body` — zero-arg

## src/eval.rs

The core module (~5800 lines). Divided into:

### Value types

```rust
enum Val {
    Num(f64),
    Complex(f64, f64),
    Fn(params, body, captured_env, bytecode_cache, sig),
    Builtin(name: String),
    Tensor { data: TData, shape: Vec<usize> },
    ComplexTensor { re: TData, im: TData, shape: Vec<usize> },
    Tuple(Tup),                             // positional or named (records, namespaces)
    Cell(Arc<RefCell<Val>>),                // shared mutable container (identity semantics)
    Field(Arc<FieldVal>),                   // k-form / vector field on a grid
}
```

`Tup` is `{ items: Vec<Val>, names: Option<Arc<Vec<Option<String>>>> }` — it
collapses to positional when no slot is named, so "record" and "tuple" are one
type and a namespace is just a named tuple. There is no separate `Namespace`
variant.

`TData` is `Arc<Vec<f64>>` with copy-on-write semantics — O(1) clone.

### Namespaces

`Expr::Member(base, field)` (parsed from `base.field` — `Token::Dot` in the postfix
loop) evaluates `base` to a named `Val::Tuple` and looks up `field`. Standard
namespaces are registered in `Env::new` via `crate::ns::register_all`. Relocated
builtins are exposed as `Val::Builtin("<bare>")` so they dispatch through the
unchanged `eval_builtin` match; new PDE functions (ops/solver) route from
`eval_builtin` via `crate::ns::dispatch` into their module files. User namespaces
are built by `import_file` from an `!namespace`-headed file (see `NsBuild`).

### Fields and differential forms

`Val::Field(Arc<FieldVal>)` is a k-form sampled on a regular grid. `FieldVal`
carries the component `data` (flat, layout `grid ++ [C(n,degree)]`, component-axis
fastest), per-axis `grid`/`spacing`/`lo`/`bc`/`metric`, plus `degree` and
`variance` (Form vs Vector). Two per-axis quantities are kept separate by design:
`spacing` (dx) enters only the exterior derivative `d`; `metric` (diagonal g_ii,
Euclidean=all 1, Minkowski=e.g. -1,1,1,1) enters only hodge/raise/lower/codiff/
laplace. The `forms` module holds the constructor (`field`, special-cased in
`eval_builtin`) and the operators (in `forms::NAMES`, routed via `crate::ns::dispatch`);
components are indexed by sorted k-subsets in lexicographic order with Levi-Civita
signs from `perm_sign`. Arithmetic operators preserve field-ness (`field_binop` in
eval.rs); any other named builtin decays a field to its tensor (`field_data_as_tensor`)
on entry to `eval_builtin`. `ops::dispatch` checks for a leading `Val::Field` and
routes to its field-polymorphic branch (`field_dispatch`), reading dx/bc from the
field and returning a field.

`FnSig { params: Vec<Option<TypeHint>>, defaults: Vec<Option<Expr>>, ret: Option<TypeHint> }`
stored with each `Val::Fn`. `defaults` is empty when the function has none.
`sig.required(n)` is the index of the first defaulted parameter — how many
arguments a call must supply.

### Updates: two axes, one writer

Both update forms share `resolve_path` → `Vec<Step>` (indices already evaluated)
and `write_steps`, in the "Paths" section of eval.rs:

| | whole value | part of a value | semantics |
|---|---|---|---|
| binding | `T = v` | `T[i] = v`, `w.f = v` | `eval_assign` rebinds the name in the current scope |
| cell | `set(c, v)` | `set(c[i], v)`, `update(c.f, g)` | `eval_cell_write` writes through the `Arc<RefCell<Val>>` |

Neither dispatches on a runtime type: the syntax decides. Key invariants —

- `resolve_index_item` splits into `eval_index_item` + `resolve_idx_item`, so an
  assignment evaluates its whole path **before** taking the root out of the
  environment (`T[T[0]] = 1` still sees `T`).
- `set`/`update` are special forms in the `Expr::Apply` arm (they need the
  unevaluated path); the `eval_builtin` arms remain for their first-class-value
  use. `Compiler::compile` declines `set`/`update` whose first argument is an
  `Index`/`Member`, and any body containing a `BlockStmt::Assign`.
- Cell writes evaluate every argument before borrowing, and `update` applies its
  function with the borrow released, so a cell can be read inside its own update.
- Tensor stores go through `TData`'s `Arc::make_mut` once per store: in place
  when the buffer is unshared, one copy when it is not.

### Argument binding

`fill_defaults` runs at the top of `apply_fn_direct`, before hint coercion and
before the VM/tree-walk split, so `LoadParam` indices are unaffected and every
call path (VM, `map`, `iterate`) gets defaults. Named arguments are resolved
separately in the `Expr::Apply` arm by `eval_call_args` + `bind_named_args`,
where the callee's parameter names are known; they produce a complete positional
vector, so the positional path in `apply_val` is untouched.

### Bytecode VM

`Instruction` enum used by a stack-based VM. Compiled lazily on first call via `OnceLock`. Instructions: `PushNum`, `LoadParam`, `LoadCaptured`, `BinOp`, `Neg`, `CallBuiltin`, `CallVal`, `MakeTuple`, `MakeArray`, `JumpIfFalse`, `Jump`, `StoreLocal`, `LoadLocal`, `Index`, `MakeClosure`, `Return`.

`compile_fn(params, body, captured)` → `Option<Vec<Instruction>>` — returns `None` for bodies the compiler can't handle (slices, ranges, tensor literals, recursive fns, `sum`/`prod`/`map`/`filter`/`reduce`).

`run_vm(code, args, captured, env)` — executes bytecode; falls back automatically via `apply_fn_direct`.

### Key public functions

- **`eval(expr, env)`** — tree-walk evaluator; entry point for all expression evaluation
- **`apply_val(f, args, env)`** — apply any callable: Fn (coerce args, try VM then tree-walk), Builtin, Num (scalar multiply/fold), etc.
- **`eval_builtin(name, vals, env)`** — dispatch table for all ~130 builtin functions
- **`builtin_sig(name)`** — returns a human-readable signature string for `!type`
- **`is_protected(name)`** — returns true for builtins that cannot be shadowed
- **`infer_type(expr, params, env)`** — static type inference for `!type`
- **`hint_of_val(v)`** — runtime type → TypeHint
- **`fmt_val(v)`** — display formatting; `fmt_mat` for 2-D matrices (box characters ⎡⎢⎣⎤⎥⎦)
- **`fmt_f(n)`** — format a single f64 (integers without `.0`, NaN/inf special cases)

### Builtin categories

Arithmetic/algebra, trig, complex, stats, higher-order (`map`, `filter`, `reduce`, `compose`, `partial`), aggregates (`sum`, `prod`, `integral`, `deriv`), linspace/range, rand, bitwise, FFT/IFFT, tensor constructors (`zeros`, `ones`, `eye`, `diag`, `tensor`, `matrix`), tensor ops (reshape, permute, cat, squeeze, unsqueeze, outer, tensordot, matmul, etc.), linear algebra (det, inv, solve, eig, QR, diagonalize), shift/roll, lerp/clamp, lingrid, cell/get/set/update.

### apply_val destructuring

When a single argument is passed to an n-param function:
- **n-Tuple of n items** → destructured into n args
- **1-D Tensor of n items where n > 1** → destructured into n scalar args (n==1 skipped to avoid turning `[x]` into scalar before type-check)
- **n==1** → direct apply (tensor passed as-is)
- **k==n** → direct apply

### Env

`Env { vars: Arc<HashMap<String, Val>> }` — copy-on-write via `Arc::make_mut`. `Env::new()` pre-populates all constants and builtins as `Val::Builtin(name)`.

## src/repl.rs

### Key constants

- **`BUILTIN_FNS`** — list of builtin function names (for highlighting, tab completion, `!clear` filtering)
- **`BUILTIN_CONSTS`** — `["pi", "e", "phi", "inf", "i"]`
- **`TYPE_KEYWORDS`** — type-hint keywords for syntax highlighting

### eval_line(line, env, repl)

Entry point for each REPL line (also used for file execution). Parses, evaluates defs, evaluates expressions, prints results. Multi-line values (matrices) print with `result =` on its own line.

### bang_command(cmd, env)

Handles all `!`-prefixed REPL commands:

| Command | Description |
|---------|-------------|
| `!help` | Print full help text |
| `!type <expr>` | Show type signature |
| `!defs` / `!vars` | List user definitions |
| `!clear` | Clear all user definitions |
| `!include <file>` | Import a .math file |
| `!print <text>` | Print with `{expr}` interpolation |
| `!graph f [, a, b]` | Plot function → PNG + open animator |
| `!animate2D …` | Stream 2-D tensor frames to animator |
| `!animate2D_raw …` | Write MXFR to stdout |
| `!savetensor`/`!loadtensor` | Binary `.mlt` format |
| `!savenpy`/`!loadnpy` | NumPy `.npy` format |
| `!savehdf5`/`!loadhdf5` | HDF5 (feature-gated) |
| `!version` | Print version |
| `!helpdef <ns>[.<member>] <text>` | Set help text for a user namespace/member |

### MathHelper

Implements rustyline `Completer`, `Highlighter`, `Hinter` for syntax coloring, tab completion, and inline hints.

## src/graph.rs

`eval_graph(args, env)` — called by `!graph` command:

1. Evaluate args: `f` required; `a`, `b` optional (default -10, 10)
2. Sample `f` at `2*width` points; split into continuous segments at discontinuities
3. Compute y-range via 5th–95th percentile (handles singularities)
4. Render PNG via `plotters` into an in-memory RGB buffer
5. Save PNG to `graph_<timestamp>_<counter>.png` in CWD
6. Call `open_in_animator` → spawn `wgpu_animator --stdin --bare --title <filename>`, write one MXFR RGB frame (channels=3), drop stdin (EOF), reap child in background thread

## src/animate.rs

`eval_animate2d(args, env)` — called by `!animate2D` command:

1. Evaluate first arg (3-D Tensor or function)
2. Extract optional fps (default 30)
3. Spawn `wgpu_animator --stdin --colormap heat --fps <fps>`
4. Stream frames via `stream_frames()` → MXFR protocol (channels=1, scalar f32 per pixel)
5. Wait for animator to exit (blocks REPL until window is closed)

`eval_animate2d_raw(args, env)` — same but writes MXFR to stdout (no subprocess).

### Axis convention

Tensors use **T[x, y]** convention: first index is x (horizontal, columns), second is y (vertical, rows). Shape is `[nx, ny]`. `stream_frames` transposes x-major data to MXFR row-major format on the fly via `write_frame_xy`.

### Calling conventions for animate2D

- `!animate2D T [fps]` — T is 3-D Tensor `[n_frames, nx, ny]` (T[x,y] convention)
- `!animate2D f n [fps]` — f called at t=0..n-1; f must return `[nx, ny]` tensor
- `!animate2D f t_vals [fps]` — f called at each timestamp in 1-D tensor
- `!animate2D f t0 t1 n [fps]` — f called at linspace(t0, t1, n)

## MXFR protocol

Binary little-endian frame format used between mathlang and wgpu_animator:

```
Offset  Size   Type    Field
0       4      u8[4]   magic = b"MXFR"
4       4      u32     width W
8       4      u32     height H
12      4      u32     channels C  (1=scalar heat-map, 3=RGB passthrough)
16      8      f64     timestamp
24      W*H*C*4 f32[]  pixel data, row-major, C values per pixel, [0,1] range
```

## animator/ binary (wgpu_animator)

Standalone wgpu+winit+egui application. Flags: `--stdin`, `--fps`, `--colormap`, `--norm`, `--interp`, `--title`, `--bare`.

- Reads MXFR frames from stdin in a background thread (`spawn_reader`)
- Renders via a full-screen quad shader (`data.wgsl`): R32Float texture for scalar, Rgba32Float for RGB
- egui overlay: colorbar, timestamp, zoom/pan controls (hidden in `--bare` mode)
- Keyboard shortcuts: `n` norm, `i` interp, `c` colormap, `r` reset, space pause, ←/→ step frames, Esc quit

## Binary discovery (find_animator)

1. `$WGPU_ANIMATOR` env var
2. `./animator/target/release/wgpu_animator` (relative to CWD)
3. `wgpu_animator` (PATH)

Set `WGPU_ANIMATOR=/absolute/path` to ensure the animator is always found regardless of CWD.
