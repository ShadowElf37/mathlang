# mathlang evaluation report — TODO 0.x

*Date: 2026-05-30 · evaluator: Claude Opus · against v0.20.0*

> **Status:** all five bugs below (BUG-1…BUG-5) were fixed in **v0.20.1**; see
> `TODO_BUGS.txt`. All six feature recommendations (0.3/0.4 — FEAT-A…FEAT-E plus the
> singleton literal) shipped in **v0.21.0**: `iterate`/`scan`, `cumsum`/`cumprod`/
> `diff`, the `(x,)` singleton literal, and rank-promoting `vstack`/`hstack`/
> `append`/`concat`. See `TODO_FEATURES.txt` §5. Only FEAT-F (the VM `Loop`
> instruction, TODO 1e) is deferred — it's a perf/GPU concern tied to the separate
> gpu_eval path, not a correctness gap.

This report covers the "CONTINUAL FOR CLAUDE OPUS" items: build something real in
pure mathlang and note what's awkward (0.1), hunt for bugs / tech debt / outdated
docs (0.2), recommend features in keeping with the language's style (0.3), look
for unification opportunities (0.4), and write it up (0.5).

Method: built small numerical projects from an engineer/physicist's seat (RK4
integrator for a vector ODE, fixed-point/Newton iteration, orbit/trajectory
accumulation) and probed the evaluator and codebase directly. Concrete findings
are filed in `TODO_BUGS.txt` and `TODO_FEATURES.txt`; this is the narrative.

---

## Executive summary

mathlang is genuinely pleasant for one-shot tensor math. Broadcasting, vector
state (`f(t,y) = (y[1], -y[0])`), first-class functions, and the flat-loop
builtins (`sum(f,lo,hi)`, `map`) all "just work" and are fast — `sum(k->k,1,1e6)`
in 0.16s, a 100k-step cell-based oscillator in 0.17s. The RK4 integrator below
took five lines and worked first try:

```
rk4(f,t,y,h) = {k1=f(t,y); k2=f(t+h/2,y+h/2*k1); k3=f(t+h/2,y+h/2*k2);
                k4=f(t+h,y+h*k3); y + h/6*(k1+2*k2+2*k3+k4)}
```

The serious gaps are all in **iteration**, which is odd because iteration (PDEs,
time-stepping) is the stated primary use case:

1. **Recursion overflows the native stack at ~1500 depth and hard-aborts the
   process** (kills the whole REPL session). The *documented* time-stepping idiom
   relies on exactly this recursion.
2. **There is no iteration primitive that both scales and collects results.** The
   working pattern is a cell driven by `sum(k->{set(c,...);0}, 1, N)` — abusing
   `sum` for side effects and discarding its value. Powerful, but a hack, and
   undiscoverable.
3. A **silent correctness bug** in negative indexing returns element 0 instead of
   erroring on out-of-range.

None are hard to fix, and fixing them would close most of the distance between
"nice calculator" and "language I'd write a simulation in."

---

## 0.1 — Building something: where it got awkward

### Iteration is the whole story, and it fights you

The skill file teaches time-stepping as recursion:

```
evolve(T, n) = if(n <= 0, T, evolve(step(T), n-1))
```

This overflows the stack somewhere between n=1000 (works) and n=2000 (aborts).
For any realistic step count it crashes the process — and in the REPL that takes
all your definitions with it (see BUG-2). So the headline idiom is a trap.

The path that *does* scale is the cell-based stepper from `heat.math`: keep state
in a `cell`, and drive the loop with a flat-loop builtin:

```
{ c = cell((1,0));
  sum(k -> {set(c, step(get(c))); 0}, 1, 100000);   # 100k steps, 0.17s, O(1) stack
  get(c) }
```

This works and is fast, but every part of it is a workaround: `sum` is being used
as a `for` loop, the lambda returns a throwaway `0`, and the actual answer comes
out of a cell on the side. Nobody guesses this; it has to be taught.

### Accumulating a trajectory is disproportionately hard

A physicist wants the *whole orbit*, not just the final state. There's no clean
way to grow one:

- Recursion that appends → overflows for long runs (BUG-2).
- `vstack(state_vector, matrix)` → rejected: *"both arguments must be 2D tensors"*
  (a 1-D state isn't auto-promoted to a row).
- `append(orbit(...), x)` with a singleton base case → the base case `(x)` is just
  the scalar `x`, so `append` rejects it: *"first arg must be a 1D tensor or tuple."*
- The cell trick works (`set(c, append(get(c), ...))`) but, again, is a hack.

### No length-1 tensor/tuple literal

`(x)` is `x` and `(x,)` is also `x` — there is no singleton collection literal.
The only ways to get a `[x]` are `tensor(k->x,1)` or `append(zeros(0), x)`, both
non-obvious. This is exactly what breaks recursive accumulators whose base case is
one element.

### Minor: top-level can't sequence statements

`a; set(c,..); get(c)` at top level errors (*"unexpected token: Semicolon"*) — the
top level only takes `def; def; … ; final_expr`. You must wrap a sequence of
expression-statements in `{ }`. The README's "; separates statements" phrasing
implies otherwise.

---

## 0.2 — Bugs, tech debt, outdated docs

### BUG-1 — negative out-of-range index silently returns element 0 *(correctness)*

```
(10,20,30)[-1]   -> 30      # ok
(10,20,30)[-4]   -> 10      # WRONG: should be out-of-range error
(10,20,30)[-100] -> 10      # WRONG
(1,2,3;4,5,6)[0,-5] -> 1    # WRONG (2-D col)
```

Positive overflow correctly errors (`[10]` → "out of range"); negative overflow
does not. Root cause is the normalization `(dim + raw).max(0)` — when `dim+raw<0`
it clamps to 0 and the `i >= dim` check then passes. This silently feeds garbage
(the first element) into a computation instead of failing. **The same `.max(0)`
clamp is duplicated at 12 sites in `eval.rs`** (real/complex × single/multi-index,
tuples, VM and tree-walk paths). Fix once via a shared `norm_index` helper that
returns `Err` when `dim+raw<0`.

### BUG-2 — deep recursion overflows the stack and aborts the process *(robustness)*

Recursion deeper than ~1500 frames prints `fatal runtime error: stack overflow,
aborting` and kills the process. There is **no** mitigation anywhere in the
codebase: no large-stack worker thread, no recursion-depth guard, no
`catch_unwind`. `main.rs` runs everything on the default 8 MB main-thread stack.

Impact is worst in the REPL: one too-deep call discards the entire session. Two
cheap fixes, ideally both:
- Run the evaluator on a `std::thread::Builder::stack_size(512 MB)` thread →
  pushes the limit from ~1.5k to ~100k+ frames.
- Add a recursion-depth counter in `apply_val`/`eval` that returns a normal
  `Err("recursion limit exceeded")` instead of overflowing.

### BUG-3 — bundled example libraries don't load

`examples/advanced.math`, `examples/physics.math`, and
`examples/animation_test.math` use the **removed `:` block separator** and fail to
parse (*"expected ';' or '}', got Colon"*). `advanced.math` additionally tries to
define `ncr` and `quadratic`, which are now builtins → *"cannot redefine
built-in."* The README advertises `m -f advanced.math 'solveCubic(1,0,-3,-2)'` as a
working example; it currently errors out completely. Migrate these files to `;`
form and drop the now-builtin defs.

### BUG-4 — README ↔ heat.math drift

README's heat section shows `m -f heat.math 'solver(0)'` and a `heatSolver(...)`
entry point, but the file actually exposes `solver_demo` (built via
`heatSolverDisk`). `solver` is undefined, so the documented command fails.

### Outdated docs

`docs/MATHLANG_SKILL.md` still teaches `:` as the def/output separator throughout
(top level, blocks, CLI) — it's removed everywhere now. Since this is the file an
agent reads to write correct mathlang, it actively produces broken code (it's how
this evaluation started). It needs a pass to `;`/last-expr semantics.

### Tech debt

- `eval.rs` is 4662 lines (TODO 2.5 already flags the split). 151
  `unwrap/expect/panic/unreachable` sites; to its credit, almost all *user* inputs
  produce clean `error:` messages (empty reductions, singular matrices, shape
  mismatches, bad axes all handled) — the live exceptions are BUG-1 and BUG-2.
- `vm_tensor_index` / `vm_complex_tensor_index` and their tree-walk twins
  (~lines 3272–3332, 4282–4461) are near-duplicate index logic — the natural home
  for the `norm_index` helper from BUG-1.

---

## 0.3 / 0.4 — Feature recommendations (and unifications)

Filed in `TODO_FEATURES.txt`. The theme: **make iteration a first-class, scalable,
collectible operation**, in the same functional/builtin style as `sum`, `map`,
`reduce`.

### FEAT-A — `iterate(f, x0, n)` → f applied n times *(flat loop, O(1) stack)*

The safe replacement for the recursive `evolve` idiom. Same shape as
`sum(f,lo,hi)`: a builtin special form that loops internally, so it never touches
the call stack and works for n=10⁶. Directly backs TODO 1e (the VM `Loop`
instruction) and is GPU-safe.

### FEAT-B — `scan(f, x0, n)` / `iterates(f, x0, n)` → the whole orbit

Returns `[x0, f(x0), …, f^n(x0)]` stacked into a tensor. This is the single most
impactful addition for the target audience: it makes trajectories a one-liner and
retires the cell+`sum` hack. If `x0` is a vector, stack states as rows
(shape `[n+1, d]`) — which also motivates fixing the 1-D→2-D promotion below.

### FEAT-C — `cumsum` / `cumprod` / `diff`

Standard array scans; trivial to implement, constantly needed in signal/numerics
work. Natural companions to the existing `sum`/`prod` reductions.

### FEAT-D — singleton collection literal

Make `(x,)` a length-1 tensor (Python-style trailing comma), or add `vec(x)`.
Removes the `tensor(k->x,1)` boilerplate and fixes recursive-accumulator base
cases. Small, unifying.

### FEAT-E — generalize the stacking family *(unification)*

`vstack`/`hstack`/`append`/`concat` should auto-promote ranks: treat a 1-D vector
as a row when stacking onto a 2-D matrix, and let `append`/`concat` accept scalars
and empty operands. Today each has a narrow shape contract that the others don't
share; unifying them removes most manual `unsqueeze`/`reshape` glue and makes
FEAT-B's vector case fall out for free.

### FEAT-F — back A/B with a VM `Loop` instruction (TODO 1e)

Already on the roadmap; A and B are the user-facing surface that justifies it, and
it's the only GPU-safe form of bounded iteration.

---

## Closing opinion

The language's identity — tensor-first, eager, no-setup, fast — is real and worth
protecting. The one place reality diverges from the pitch is iteration: the docs
sell recursion, recursion crashes, and the thing that actually scales is an
undocumented side-effect hack. Shipping `iterate`/`scan` (FEAT-A/B), fixing the
stack abort (BUG-2), and correcting the silent index bug (BUG-1) would, together,
be a small amount of code that changes mathlang from "great calculator" to
"language I'd actually run a simulation in." Everything else here is cleanup.
