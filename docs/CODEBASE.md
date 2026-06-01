# Mathlang Codebase Structure

## Top-level layout

```
src/
  main.rs          — CLI entry point; dispatches to REPL or file mode
  lexer.rs         — Tokenizer (Token enum, Lexer struct)
  ast.rs           — AST nodes: Expr, Def, BlockStmt, TypeHint, Param, Op
  parser.rs        — Recursive descent parser (Parser struct)
  eval.rs          — Evaluator, builtins, VM, type inference (largest file ~5000 lines)
  ns/              — Standard namespaces (`.` access), each in its own module:
    mod.rs         — register_all (called by Env::new), routing to new-function dispatch
    ops.rs   — grad/div/curl/lap/poisson/invlap/specgrad (finite-diff + spectral);
                     also field-polymorphic when the first arg is a Val::Field
    solver.rs      — rk4/odeint/verlet(symplectic)/cfl time integrators
    forms.rs       — field() constructor + exterior calculus (d/hodge/wedge/raise/
                     lower/codiff/laplace) on Val::Field; metric-aware
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

- **`Expr`** — expression nodes: `Num`, `ImagLit`, `Var`, `BinOp`, `Neg`, `Apply`, `Lambda`, `Tuple`, `Array`, `TensorLit`, `Index`, `Slice`, `Range`, `Block`
- **`Def`** — `Var(name, expr)` or `Func(name, params, ret_hint, body)`
- **`BlockStmt`** — `Def(Def)` or `Expr(Expr)` (statements inside `{...}`)
- **`TypeHint`** — `Any | Num | Real | Complex | Int | Nat | Tensor | RealTensor | ComplexTensor | Fn | Cell | Tuple`
- **`Param`** — `{ name: String, hint: Option<TypeHint> }`

## src/lexer.rs

`Lexer::new(src).tokenize()` → `Vec<Token>`.  Token variants include: `Num(f64)`, `Imag(f64)`, `Ident(String)`, `Arrow`, `Colon`, `LParen`/`RParen`, `LBracket`/`RBracket`, `LBrace`/`RBrace`, `Comma`, `Semicolon`, `DotDot`, `Bang`, `Eq`/`EqEq`/`BangEq`, arithmetic operators, `Eof`.

## src/parser.rs

`Parser::new(toks).parse_repl()` → `(Vec<Def>, Vec<Expr>)`.

Key parsing methods:

- **`parse_repl()`** — parses one REPL line: zero or more `Def`s (separated by `;`), then zero or more `Expr`s
- **`parse_def()`** — `name = expr` (Var) or `name(params) [: ret] = expr` (Func)
- **`expr()`** / **`cmp()`** / **`add()`** / **`mul()`** / **`pow()`** / **`postfix()`** / **`primary()`** — operator precedence chain
- **`primary()`** handles: numeric literals, imaginary literals, `(...)` tuples/lambdas/matrices, `[...]` array/matrix literals, `{...}` blocks, identifiers (variables, function calls, single-arg lambdas `x ->`, typed single-arg lambdas `x: type ->`)
- **`looks_like_paren_lambda()`** / **`is_multi_lambda()`** — lookahead helpers that determine whether to parse as lambda vs tuple/call
- **`parse_param()`** — `name [: type_hint]`

Lambda forms supported:
- `x -> body` — bare single-arg
- `x: type -> body` — bare single-arg with type hint
- `(x, y) -> body` — paren multi-arg
- `(x: type, y) -> body` — paren multi-arg with hints
- `x, y -> body` — multi-arg without parens (via `is_multi_lambda`)
- `() -> body` — zero-arg

## src/eval.rs

The core module (~4100 lines). Divided into:

### Value types

```rust
enum Val {
    Num(f64),
    Complex(f64, f64),
    Fn(params, body, captured_env, bytecode_cache, sig),
    Builtin(name: String),
    Tensor { data: TData, shape: Vec<usize> },
    ComplexTensor { re: TData, im: TData, shape: Vec<usize> },
    Tuple(Vec<Val>),
    Cell(RefCell<Val>),
    Namespace(Arc<HashMap<String, Val>>),   // ns.member access
}
```

`TData` is `Arc<Vec<f64>>` with copy-on-write semantics — O(1) clone.

### Namespaces

`Expr::Member(base, field)` (parsed from `base.field` — `Token::Dot` in the postfix
loop) evaluates `base` to a `Val::Namespace` and looks up `field`. Standard
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

`FnSig { params: Vec<Option<TypeHint>>, ret: Option<TypeHint> }` stored with each `Val::Fn`.

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

Arithmetic/algebra, trig, complex, stats, higher-order (`map`, `filter`, `reduce`, `compose`, `partial`), aggregates (`sum`, `prod`, `integral`, `deriv`), linspace/range, rand, bitwise, FFT/IFFT, tensor constructors (`zeros`, `ones`, `eye`, `diag`, `tensor`, `matrix`), tensor ops (reshape, permute, cat, squeeze, unsqueeze, outer, tensordot, matmul, etc.), linear algebra (det, inv, solve, eig, QR, diagonalize), shift/roll, lerp/clamp, lingrid, cell/get/set.

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
