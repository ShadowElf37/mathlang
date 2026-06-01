# m — a math language

A fast, tensor-first math evaluator written in Rust. Use it in the REPL for interactive exploration, write `.math` files for reusable work, or fire off one-liners from the shell.

---

## Philosophy

- **Tensor-first.** Numbers, vectors, matrices, and n-D tensors are all first-class. Arithmetic broadcasts automatically — no explicit loops, no shape boilerplate.
- **Just works.** Type an expression, get an answer. Complex numbers, calculus, FFT, linear algebra, and plotting are all built in — no imports, no setup.
- **Fast.** Startup is near-instant (compiled binary, no runtime). Evaluation is eager and direct. You should never be waiting on `m`.
- **One-stop.** Quick arithmetic, tensor manipulation, eigenvalues, HDF5 I/O, animations — all in one tool. Load a physics library, solve a PDE, plot the result, save the tensor, done.
- **Write once, use everywhere.** A `.math` file is a first-class citizen: load it in the REPL with `!include`, run it from the shell with `m -f`, or source it from another file. `!commands` work the same way in files as in the interactive session.

---

## Quick start

**Interactive REPL** — `m` with no arguments:

```
> pi * 2^2
result = 12.566370614359172
> f = x -> x^2
> f(3)
result = 9
> A = (1,2; 3,4)
> eigvals(A)
result = [5.372…, -0.372…]
```

**`.math` files** — write definitions, run with `-f` or `!include`:

```zsh
# physics_problem.math
!include physics.math
E = 0.5 * m_e * (0.1 * c)^2    # kinetic energy at 10% c

m -f physics_problem.math 'E'   # evaluate from shell
```

Or from the REPL:

```
> !include physics_problem.math
included 2 definition(s) from physics_problem.math
> E
result = 4.09…e-16
```

**Shell one-liners** — quick calculations without entering the REPL:

```zsh
m '3 + 4'
m 'pi * 2^2'
m 'x=3; y=4; sqrt(x^2 + y^2)'
```

> **Always quote the argument** in the shell to avoid metacharacter expansion (`^`, `*`, `;`, `>`).

---

## REPL

Run `m` with no arguments to enter the interactive REPL. It supports syntax highlighting, tab completion, and history.

```
> x = 3
> f(n) = n^2
> f(x)
result = 9
> result + 1
result = 10
```

**REPL commands** — also work in `.math` files:

| Command | Effect |
|---------|--------|
| `!help` | show syntax reference |
| `!help <name>` | show detailed help for a builtin function or `!command` |
| `!defs` | list all user-defined names |
| `!include <file>` | load a `.math` file into the current session |
| `!clear` | clear all user definitions |
| `!version` | show version |
| `!type <expr>` | show the type of a value, expression, or function |
| `!print [text with {expr}]` | print text with interpolated expressions |
| `!graph f [, a, b]` | plot f over [a,b] (default -10..10); saves PNG and opens animator |
| `!animate2D T [fps]` | animate a 3-D tensor `[frames,H,W]`; spawns animator |
| `!animate2D f n [fps]` | animate f(t) for t=0..n-1 |
| `!animate2D f t0 t1 n [fps]` | animate f(t) over linspace(t0,t1,n) |
| `!animate2D_raw …` | write MXFR frames to stdout (for piping) |
| `!savetensor <var> <file>` | save tensor to binary `.mlt` file |
| `!loadtensor <var> <file>` | load tensor from `.mlt` file |
| `!savehdf5 <var> <file> …` | save tensor to HDF5 (requires `--features hdf5`) |
| `!loadhdf5 <var> <file> …` | load tensor from HDF5 |
| `!q` / `!quit` / `!exit` | quit |

---

## `.math` files

A `.math` file is plain text: one expression or definition per line. Lines starting with `#` are comments. Braces `{ }` can span multiple lines. **All `!commands` work in `.math` files** — `!include`, `!print`, `!savetensor`, `!loadtensor`, `!savehdf5`, `!loadhdf5`, `!defs`, `!version`, `!graph`, `!animate2D`, etc.

```
# my_lib.math
sq = x -> x^2
cube = x -> x^3
# load another file
!include conversions.math
```

`!print` lets you emit formatted output from a `.math` file. Use `{expr}` to interpolate any expression; `{{` / `}}` produce literal braces:

```
# solver.math
!print running solver...
n = 100
result = sum(x -> x^2, 1, n)
!print n = {n}, sum of squares = {result}
!print norm of solution: {norm(solve(A, b))}
!print use {{expr}} for interpolation
```

```
> !include solver.math
running solver...
n = 100, sum of squares = 338350
norm of solution: …
use {expr} for interpolation
```

`!print` with no argument prints a blank line.

### Loading files from the shell

```zsh
m -f defs.math                     # load file, no output
m -f defs.math 'f(10)'             # load file, then evaluate expression
m -f advanced.math 'solveCubic(1,0,-3,-2)'
```

### Loading files from the REPL or another file

```
> !include defs.math
included 5 definition(s) from defs.math
```

Files can `!include` other files — use this for modular libraries.

### Init file

Definitions in `~/.mathlangrc` are loaded automatically when the REPL starts.
Set `MATHLANG_INIT=/path/to/file.math` to use a different file.

### Included libraries

#### `advanced.math`

Combinatorics, polynomial solvers, and more:

```
ncr(n,r)                — binomial coefficient
gamma(z)                — Stirling approximation of Γ(z)
quadratic(a,b,c)        — roots of ax²+bx+c=0  (returns tuple)
solveCubic(a,b,c,d)     — 3 real roots of ax³+… (trig method, disc > 0)
cubicRoot(a,b,c,d)      — 1 real root  of ax³+… (Cardano, disc < 0)
solveQuartic(a,b,c,d,e) — 4 real roots of ax⁴+… (Ferrari's method)
```

Load: `!include advanced.math` (or `m -f advanced.math 'expr'`).

#### `physics.math`

Physical constants in SI units (`c`, `g`, `G`, `k_B`, `N_A`, `R`, `h`, `hbar`, etc.) plus `oscillator`, `oscillatorMBK`, `heatKernel`, `heatSpectral`.

#### `heat.math`

2-D heat equation solver with spatially-varying diffusivity.

```zsh
m -f heat.math 'solver_demo(0)'    # initial condition: 20×20 grid, cold disk in hot air
m -f heat.math 'solver_demo(10)'   # temperature after t=10 (50 steps)
m -f heat.math 'solver_demo(20)'   # mostly equilibrated (~100 ms)
```

`solver_demo` is a ready-made stateful solver (cold disk in hot air). To build your
own, `heatSolver(T0, alpha, dx, dt)` takes an initial temperature grid, diffusivity
field, and timestep and returns a stateful solver; `heatSolverDisk(N, dx, dt,
T_disk, T_air, a_disk, a_air)` is the convenience constructor used for `solver_demo`.
Uses energy-conserving divergence-form FD with Neumann BCs.

#### `integrators.math`

Symplectic and explicit ODE integrators, built in pure mathlang on top of
`iterate`/`scan` (no special builtins).

```
verletStep(dHdq, dHdp, dt)              — one velocity-Verlet step, (q,p) -> (q,p)
verletOrbit(dHdq, dHdp, q0, p0, dt, n)  — whole phase-space orbit via scan
verletFinal(dHdq, dHdp, q0, p0, dt, n)  — endpoint via iterate, O(1) stack
verlet(dHdq, dHdp, q0, p0, dt)          — stateful generator (cell), for animate
rk4Step / rk4Orbit / rk4Final           — classic RK4 for y' = f(t, y)
```

```zsh
m -f integrators.math 'shape(verletOrbit(q->q, p->p, 1.0, 0.0, 0.05, 200))'  # [201, 2]
```

For a separable Hamiltonian `H(q,p) = T(p) + V(q)`, pass `dHdq = ∂H/∂q` (force) and
`dHdp = ∂H/∂p` (velocity). The simple harmonic oscillator is `dHdq = q`, `dHdp = p`.
Verlet is symplectic, so energy stays bounded over long runs.

#### `conversions.math`

Unit conversion factors and functions for physics, chemistry, and engineering.

```
> !include conversions.math
> 1 * eV_to_J          # 1.602176634e-19
> C_to_K(100)          # 373.15
> 1 * atm_to_Pa        # 101325
```

---

## Syntax

Inside the REPL or a `.math` file, each line is an expression or a definition (`name = expr`). Multiple definitions on one line are separated by `;`.

For shell one-liners, the full argument is parsed as a sequence of statements separated by `;`; the value of the last expression is printed:

```zsh
m 'x=3; y=4; x^2 + y^2'      # 25   (; separates statements; last is the output)
m 'sqrt(2), sin(pi/2)'       # 1.414…  1  (comma = multiple outputs)
```

---

## Operators

| Operator | Meaning |
|----------|---------|
| `+` `-` `*` `/` | arithmetic |
| `^` or `**` | exponentiation (right-associative) |
| `//` | floor division |
| `%` | remainder |
| `<` `>` `<=` `>=` `==` `!=` | comparison (returns `1` or `0`) |
| `&` `\|` | bitwise AND / OR (also written `&&` / `\|\|`) |
| `n!` | postfix factorial (`5! = 120`) |
| `A @ B` | matrix / tensor multiply |

### Implicit multiplication

A number immediately followed by a name or `(` multiplies implicitly:

```
> 2pi
result = 6.283185307179586
> 3sin(pi/2)
result = 3
> 2(x+1)    # same as 2*(x+1)
```

Precedence is the same as `*`, so `2x^2` parses as `(2x)^2`. Use explicit `*` when that matters.

---

## Constants

`pi`, `e`, `phi`, `inf`, `i` (imaginary unit)

---

## Variables and functions

```
> x = 3
> x^2
result = 9
> f(n) = n^2
> f(3), f(4)
result = 9  16
> g(x,y) = x^2 + y^2
> g(3,4)
result = 25
```

Parameter names shadow globals — `f(pi) = pi+1; f(2)` → `3` (the parameter `pi` = 2, not 3.14…).

---

## Anonymous functions (lambdas)

Single-argument: `x -> expr`. Multi-argument: `x, y -> expr` or `(x, y) -> expr`.

```
> f = x -> x^2
> f(3)
result = 9
> ncr = n, r -> fact(n)/(fact(r)*fact(n-r))
> ncr(5,2)
result = 10
```

Lambdas are first-class — pass them to functions, apply them inline:

```
> (x -> x^2)(5)
result = 25
> sum(x -> x^2, 1, 10)
result = 385
```

---

## Type hints

Function parameters and return values can carry optional type hints. They are
checked when the function is called — a value that doesn't match (and can't be
coerced) raises an error.

```
> f(x: real) = x^2            # x must be real
> g(z: complex) = re(z)       # reals widen to complex automatically
> f(n: nat): real = sqrt(n)   # return type goes after the parameter list
> h(T: real tensor) = sum(T)  # tensor element-type hint
> k = (x: real): real -> x*2  # lambdas can be typed too
```

A complex value with a negligible imaginary part coerces to `real`; otherwise it
is rejected. The vocabulary is `real`, `complex`, `num`, `int`, `nat`, `tensor`,
`real tensor`, `complex tensor`, `fn`, `cell`, `tuple`, and `any`.

Use `!type` to see the type of any value, expression, or function. For a function
it prints the fused signature, inferring the return type from the body when it is
not annotated:

```
> !type 5
real
> !type 3 + 2i
complex
> f(x: real, y: complex) = y * linspace(0, 1, 10)
> !type f
(real, complex) -> complex tensor
```

---

## Vectors (1-D tensors)

Comma-separated numeric values in parentheses produce a **1-D tensor** displayed as `[...]`. Index with `[n]` (zero-based; negative indices count from the end).

```
> (1, 2, 3)
result = [1, 2, 3]
> (1, 2, 3)[1]
result = 2
> (1, 2, 3)[-1]
result = 3
```

Arithmetic broadcasts element-wise:

```
> (1, 2, 3) * 2
result = [2, 4, 6]
> (1, 2, 3) + (4, 5, 6)
result = [5, 7, 9]
> (1, 2, 3) @ (1, 2, 3)
result = 14    # dot product
```

### Slicing

```
> (1,2,3,4,5)[1..3]    # [2, 3, 4]   — bounded range (inclusive)
> (1,2,3,4,5)[2..]     # [3, 4, 5]   — from index 2 to end
> (1,2,3,4,5)[..2]     # [1, 2, 3]   — from start to index 2 (inclusive)
> (1,2,3,4,5)[..]      # [1, 2, 3, 4, 5]  — all elements
> len((1,2,3,4))
result = 4
```

---

## Blocks

Blocks `{...}` create a local scope. Statements are separated by `;`; the value of the last expression is the block's result.

```
> {x = 3; y = 4; x^2 + y^2}
result = 25
```

Blocks can appear anywhere an expression is expected — inside function bodies, multi-line files, etc.

---

## Comparisons

Comparison operators return `1` (true) or `0` (false):

```
> 3 < 5
result = 1
> 3 == 3
result = 1
> 3 != 4
result = 1
```

Combined with `if`:

```
> x=4; if(x > 0, sqrt(x), 0)
result = 2
```

---

## `if`

`if(cond, a, b)` returns `a` when `cond` is nonzero, `b` otherwise. Branches are evaluated lazily.

```
> if(1, 10, 20)
result = 10
> f = x -> if(x >= 0, sqrt(x), 0)
> f(4), f(-1)
result = 2  0
```

`if(cond, fn1, fn2)(x)` evaluates the chosen function on `x`:

```
> if(1, x -> x^2, x -> x^3)(5)
result = 25
```

---

## Calculus

```
> integral(x -> x^2, 0, 1)
result = 0.3333333333333333
> integral(x -> sin(x), 0, pi)
result = 2
> deriv(x -> x^3, 2)
result = 12    # 5-point stencil, dx=1e-5
```

`integral(f, a, b, n)` — Simpson's rule with `n` steps (default 1000).
`deriv(f, x, dx)` — custom step size.

---

## Aggregates

```
> sum(x -> x, 1, 100)
result = 5050
> prod(x -> x, 1, 10)
result = 3628800    # 10!
> sum((1, 2, 3, 4))
result = 10
```

`sum` and `prod` also work on tensors, including axis-wise reduction:

```
> sum(ones(3, 4))
result = 12
> sum(ones(2, 3), 0)
result = [2, 2, 2]    # reduce along axis 0
> sum(ones(2, 3), 1)
result = [3, 3]       # reduce along axis 1
```

`map(f, t)` applies `f` to each element:

```
> map(x -> x^2, (1,2,3,4))
result = [1, 4, 9, 16]
```

`filter(f, v)` keeps elements where `f` returns nonzero:

```
> filter(x -> x > 2, (1,2,3,4))
result = [3, 4]
```

`reduce(f, t)` left-folds with a 2-arg function:

```
> reduce((a,b) -> a+b, (1,2,3,4))
result = 10
```

---

## Iteration

For time-stepping and fixed-point iteration, `iterate` and `scan` apply a step
function repeatedly in a **flat internal loop** — O(1) stack, scaling to millions
of steps where deep recursion would overflow. They (along with `sum`/`prod`'s
function forms) compile to a single bytecode loop, so they stay fast even when
called from inside another user function.

`iterate(f, x0, n)` returns `fⁿ(x0)` (f applied n times):

```
> iterate(x -> 2*x, 1, 10)
result = 1024
> iterate(x -> x + 1, 0, 1000000)   # a million steps, O(1) stack
result = 1000000
```

`scan(f, x0, n)` returns the whole orbit `[x0, f(x0), …, fⁿ(x0)]` stacked into a
tensor. Scalar states give a 1-D tensor of length `n+1`; **vector** states of
length `d` give a 2-D tensor `[n+1, d]`, one state per row:

```
> scan(x -> 2*x, 1, 4)
result = [1, 2, 4, 8, 16]
> scan(v -> (v[1], -v[0]), (1,0), 4)   # harmonic-oscillator orbit, as rows
⎡  1   0 ⎤
⎢  0  -1 ⎥
⎢ -1   0 ⎥
⎢  0   1 ⎥
⎣  1   0 ⎦
```

This makes a stepper and its whole trajectory a one-liner. For example, one RK4
step of `y' = (y₁, -y₀)`, traced for the full orbit:

```
> rk4(f,y,h) = {k1=f(y); k2=f(y+h/2*k1); k3=f(y+h/2*k2); k4=f(y+h*k3); y+h/6*(k1+2*k2+2*k3+k4)}
> scan(y -> rk4(v -> (v[1], -v[0]), y, 0.1), (1,0), 100)   # 101×2 trajectory
```

A **structured** tuple state — one whose fields are themselves vectors, such as a
phase-space state `(q, p)` with vector `q`, `p` — is stacked field by field, so
`scan` returns a tuple `(Q, P)` of `[n+1, d]` matrices rather than trying to flatten
the state into one row. (A flat numeric tuple like `(a, b)` is still row-packed into
`[n+1, 2]`.) This is what lets `scan(verletStep, (q0, p0), n)` trace a multi-DOF
Hamiltonian trajectory directly.

`examples/integrators.math` builds **symplectic and explicit integrators in pure
mathlang** on top of these primitives — `verletStep`/`verletOrbit`/`verletFinal`
(velocity-Verlet for separable Hamiltonians), `rk4Step`/`rk4Orbit`, and a stateful
`verlet(…)` *generator* (a closure over a `cell`, the same pattern as `heat.math`)
for use with `animate`. No special integrator builtins are needed — the step is a
plain function and `iterate`/`scan` do the looping:

```
> !include integrators.math
> step = verletStep(q -> q, p -> p, 0.05)    # SHO: dH/dq = q, dH/dp = p
> shape(scan(step, (1.0, 0.0), 200))         # phase-space orbit
result = [201, 2]
```

`cumsum`, `cumprod`, and `diff` are running scans over a 1-D tensor or tuple
(`diff` is the inverse of `cumsum` up to the first element):

```
> cumsum([1, 2, 3, 4])
result = [1, 3, 6, 10]
> cumprod([1, 2, 3, 4])
result = [1, 2, 6, 24]
> diff([1, 4, 9, 16])
result = [3, 5, 7]
```

---

## Tensors and matrices

Tensors are n-dimensional arrays of real numbers. A 1-D tensor displays as `[...]`, a 2-D tensor as a boxed matrix.

### Constructors

```
> zeros(3, 4)                        # 3×4 zero matrix
> ones(2, 3, 4)                      # 2×3×4 tensor of ones (any rank)
> eye(3)                             # 3×3 identity matrix
> diag((1, 2, 3))                    # diagonal matrix from 1-D tensor
> rand(r, c)                         # r×c random matrix
> linspace(0, 1, 5)                  # [0, 0.25, 0.5, 0.75, 1]
> range(0, 5)                        # [0, 1, 2, 3, 4]  (exclusive end)
> matrix((i,j) -> i*3+j, 2, 3)      # 2×3 matrix filled by function (0-indexed)
> tensor((i,j,k) -> i+j+k, 2,3,4)   # arbitrary n-D tensor
```

Matrix literal syntax — semicolons separate rows inside parentheses:

```
> (1, 2; 3, 4)
⎡ 1  2 ⎤
⎣ 3  4 ⎦
```

### Indexing and slicing

Single-index `T[i]` returns the i-th row (or element for 1-D). Multi-index `T[i, j, …]` selects an element. All indices are zero-based; negative indices count from the end.

**Slice syntax** — works on tensors of any rank:

| Syntax | Meaning |
|--------|---------|
| `T[.., j]` | all rows, column j |
| `T[i, ..]` | row i, all columns |
| `T[n.., j]` | rows n to end, column j |
| `T[..n, j]` | rows 0 through n, column j |
| `T[a..b]` | elements a through b (inclusive) |
| `T[.., .., k]` | all of axes 0 and 1, index k on axis 2 |

```
> A = (1,2,3; 4,5,6; 7,8,9)
> A[1, 2]         # 6          — element
> A[0, ..]        # [1, 2, 3]  — first row
> A[.., 0]        # [1, 4, 7]  — first column
> A[1.., 0]       # [4, 7]     — rows 1 to end, col 0
> A[..1, 1..2]    # 2×2 submatrix (rows 0-1, cols 1-2)
```

### Arithmetic

All standard operators broadcast element-wise. Scalar-tensor and tensor-tensor (same shape) both work:

```
> eye(3) * 2                        # 3×3 with 2s on diagonal
> ones(3,3) + eye(3)                # all ones plus identity
> sin(zeros(2,3))                   # element-wise sin (any 1-arg fn works)
> (1,2; 3,4) @ (1,2; 3,4)          # matrix multiply via @
```

### Shape operations

```
> shape(ones(2, 3, 4))              # [2, 3, 4]
> rows(A)                           # 3
> cols(A)                           # 3
> reshape(ones(6), 2, 3)            # reshape to 2×3
> flatten(eye(2))                   # [1, 0, 0, 1]  — 1-D tensor
> transpose(A)                      # reverse all axes
> transpose(T, 0, 2)                # swap axes 0 and 2
> permute(T, 2, 0, 1)               # reorder axes by permutation
> squeeze(zeros(1, 3, 1))           # remove size-1 dims → shape [3]
> unsqueeze(zeros(3), 0)            # insert size-1 dim → shape [1, 3]
```

### Concatenation

```
> cat(0, eye(2), eye(2))            # stack vertically (= vstack)
> cat(1, eye(2), eye(2))            # stack horizontally (= hstack)
> hstack(A, B)
> vstack(A, B)
```

`hstack`/`vstack` **promote ranks**: a 1-D vector is treated as a row (for
`vstack`) or column (for `hstack`), and a scalar as a 1×1 block, so a vector
stacks directly onto a matrix:

```
> vstack((1,2;3,4), (5,6))          # append a row vector to a matrix → 3×2
> hstack((1,2), (3,4))              # two column vectors → 2×2
```

`append` and `concat` likewise accept scalars (and empty operands) as length-1
vectors, and `(x,)` is a singleton 1-D tensor — handy as an accumulator base
case (`(x)` alone is just the scalar `x`):

```
> (5,)                              # [5]   — singleton tensor (cf. (5) = 5)
> append(1, 2)                      # [1, 2]
> concat(zeros(0), [1, 2])          # [1, 2]
```

### Linear algebra

```
> det((1,2; 3,4))                   # -2
> inv((1,2; 3,4))                   # inverse
> trace(eye(4))                     # 4
> norm(ones(3, 4))                  # Frobenius norm = sqrt(12)
> solve((2,1; 1,3), (5,10))         # solve Ax=b  →  1-D tensor
> A @ B                             # matmul: 2D×2D, 2D×1D, 1D×2D, 1D×1D (dot)
> linalg.outer(ones(2), ones(3))    # outer product → shape (2, 3)
```

Less-common decompositions (`qr`, `diagonalize`, `eig_top`, `eig_bot`, `tensordot`,
`outer`) live in the `linalg` namespace; see below.

### Eigenvalues, QR, and diagonalization

**Eigenvalues and eigenvectors** — all functions accept square real matrices:

```
> eigvals((4,1; 1,3))             # [4.618…, 2.381…]   — eigenvalues only
> eig((4,1; 1,3))                 # (eigenvalues, V)    — V columns are eigenvectors
> linalg.eig_top((4,1; 1,3))      # (λ_max, v)          — largest eigenpair via power iteration
> linalg.eig_bot((4,1; 1,3))      # (λ_min, v)          — smallest eigenpair via inverse iteration
```

`linalg.eig_top` and `linalg.eig_bot` are much faster than full `eig` when only one eigenpair is needed.

Eigenvalue identities:

```
> sum(eigvals(A))    # equals trace(A)
> prod(eigvals(A))   # equals det(A)
```

**QR decomposition** — works for m×n matrices with m ≥ n; returns full Q (m×m) and R (m×n):

```
> linalg.qr((3,1; 1,2))                 # (Q, R)  with Q orthogonal, R upper-triangular
> linalg.qr((3,1; 1,2))[0]              # Q only
> shape(linalg.qr((1,2; 3,4; 5,6))[0])  # [3, 3]  — Q is always square
```

Verify: `Q @ R` recovers the original; `transpose(Q) @ Q` = `eye(m)`.

**Diagonalization** — returns `(V, D, V_inv)` where `V @ D @ V_inv = A`:

```
> linalg.diagonalize((4,1; 1,3))    # (V, D, inv(V))
```

`D` is a full diagonal matrix, usable directly in expressions. This enables the matrix exponential without a dedicated builtin:

```
# Matrix exponential: e^A = V @ diag(map(exp, eigvals(A))) @ inv(V)
> res = linalg.diagonalize(A)
> res[0] @ diag(map(exp, eigvals(A))) @ res[2]
```

### Queries, statistics, and argmax/argmin

```
> mean(ones(3, 4))     # 1
> std(eye(3))          # standard deviation over all elements
> stats.var(eye(3))    # variance  (stats.median, stats.mode too)
> argmax((3,1,4,1,5))  # 4         — index of max in 1-D tensor (scalar)
> argmax((1,8; 8,1))   # [0, 1]    — [row, col] index for 2-D tensor
> argmin(T)            # [i, j, …] — n-D index tensor for n-D input
```

### Grid construction (`lingrid`)

`lingrid(start, end, counts, f)` evaluates `f` at a uniformly-spaced grid. `start`, `end`, `counts` can be scalars (1-D) or tuples of the same length (n-D).

```
# 1-D: 5 points from 0 to 1
> lingrid(0, 1, 5, x -> x^2)

# 2-D: 20×20 scalar field
> lingrid((-2,-2),(2,2),(20,20),(x,y) -> x^2 + y^2)

# 2-D coordinate field — f returns a tuple, output is (20,20,2)
> G = lingrid((-1,-1),(1,1),(20,20),(x,y)->(x,y))
> G[.., .., 0]     # x-component grid (20×20)
> G[.., .., 1]     # y-component grid (20×20)

# f can return a tensor — output shape = grid_shape ++ value_shape
> shape(lingrid((0,0),(1,1),(3,3),(x,y)->eye(2)))    # [3, 3, 2, 2]
```

---

## Complex numbers

`i` is the imaginary unit (√−1). Write complex literals as `3 + 2i`, `2i`, `-i`, etc.
All arithmetic operators work on complex numbers.

```
> i^2
result = -1
> (1 + i) * (1 - i)
result = 2
> exp(i * pi)
result = -1    # Euler's formula
> sqrt(-1)
result = i
> ln(-1)
result = 3.141592653589793i    # = πi
> abs(3 + 4i)
result = 5    # modulus
> conj(3 + 4i)
result = 3 - 4i
> re(3 + 4i), im(3 + 4i)
result = 3  4
```

---

## Built-in functions

**Trig:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `sinh`, `cosh`, `tanh`, `sec`, `csc`, `cot`

**Algebra:** `sqrt`, `cbrt`, `abs`, `sign`, `heaviside`, `floor`, `ceil`, `round`, `round(x,n)`, `trunc`, `frac`, `exp`, `ln`, `log`/`log10`, `log(x,base)`, `log2`, `pow(x,y)`, `min`, `max`, `hypot(a,b)`

**Angle:** `deg(x)` (radians→degrees), `rad(x)` (degrees→radians)

**Complex:** `re(z)`, `im(z)`, `abs(z)`, `arg(z)`, `conj(z)`

**Number theory:** `gcd(a,b)`, `lcm(a,b)`, `fact(n)` / `n!`

**Array ops:** `len`, `sort`, `zip(a,b)`, `dot(a,b)`, `append(t,x)`, `concat(a,b)`, `flatten(t)`, `argmin(t)`, `argmax(t)`, `cumsum(t)`, `cumprod(t)`, `diff(t)`, `linspace(a,b,n)`, `range(a,b)`

**Tensor constructors:** `zeros(n1,n2,…)`, `ones(n1,n2,…)`, `eye(n)`, `diag(t)`, `matrix(f,r,c)`, `tensor(f,n1,n2,…)`, `rand()`, `rand(n1,n2,…)`

**Tensor shape:** `shape(T)`, `rows(T)`, `cols(T)`, `reshape(T,n1,n2,…)`, `flatten(T)`, `squeeze(T)`, `unsqueeze(T,dim)`

**Tensor reorder:** `transpose(T)`, `transpose(T,a,b)`, `permute(T,p0,p1,…)`

**Tensor combine:** `cat(axis,T1,T2,…)`, `hstack(A,B)`, `vstack(A,B)`, `tomat(t,r,c)`

**Tensor reduce:** `sum(T)`, `prod(T)`, `sum(T,axis)`, `prod(T,axis)`, `norm(T)`, `trace(T)`, `mean(T)`, `std(T)`

**Tensor shift:** `shift(T,n,axis)` (edge-replicating / Neumann), `roll(T,n,axis)` (periodic)

**Linear algebra:** `matmul(A,B)` / `A @ B`, `det(A)`, `inv(A)`, `solve(A,b)`, `row(T,i)`, `col(T,j)`, `eig(A)` → `(eigenvalues, V)`, `eigvals(A)`  (more in the `linalg` namespace)

**Statistics:** `mean`, `std`, `min`, `max`, `sum`, `prod`  (more in the `stats` namespace)

**Grid:** `lingrid(start,end,counts,f)` — n-D uniform grid; `f` may return a scalar, tuple, or tensor

**Comparisons (function form):** `lt`, `leq`, `gt`, `geq`, `eq`, `neq` — 2-arg comparison functions returning `0`/`1`; useful with `map`/`filter`/`partial`

**Higher-order:** `map(f,t)`, `filter(f,t)`, `reduce(f,t)`, `compose(f,g)`, `partial(f,a)`

**Iteration:** `iterate(f,x0,n)` → `fⁿ(x0)`, `scan(f,x0,n)` → orbit `[x0,…,fⁿ(x0)]` (flat loop, O(1) stack; scalar states → 1-D, vector states → 2-D rows). Both abort with a clear error if a step produces NaN/Inf, instead of silently returning a non-finite result.

**Spectral:** `fft(T)`, `ifft(T)` — n-D FFT/IFFT over all axes by default; also `fft(T, axes)`, `fft(Re, Im)`, `fft(Re, Im, axes)`.

**Random:** `rand()`, `rand(n1,n2,…)` — scalar or shaped tensor

### Namespaces

Niche functions live in **namespaces**, accessed with `.`:

```
> bits.xor(6, 3)            # 5      — bitwise ops
> special.erf(1)           # 0.8427 — special functions
> stats.median((3,1,2))    # 2
> linalg.qr((3,1; 1,2))    # (Q, R)
> vec.lerp(0, 10, 0.5)     # 5
```

The standard namespaces are:

| Namespace | Members |
|-----------|---------|
| `special` | `erf erfc j0 j1 jinc sinc sech csch gaussian gaussian_cdf delta` |
| `bits`    | `and or xor nand nor xnor shl shr not` |
| `stats`   | `median mode var` (mean, std are flat) |
| `linalg`  | `qr diagonalize tensordot outer eig_top eig_bot` |
| `vec`     | `lerp clamp` |
| `ops` / `solver` | differential operators and integrators — see below |
| `forms`   | exterior calculus on fields: `d hodge wedge raise lower codiff laplace` — see below |

A namespace is a first-class value (`f = bits.xor; f(6,3)`). Names placed in a
namespace are **not** reserved words, so `xor`, `lerp`, `var`, … are free to use
as your own variable names. Browse a namespace's members with `!help <namespace>`.

**Your own namespaces:** put `!namespace <name>` at the top of a `.math` file and
its definitions become members of `<name>` when you `!include` it. Prefix a
definition with `private` to keep it internal (usable by the file's other
definitions, but not exported):

```
# geo.math
!namespace geo
private k = 3.14159
area(r) = k * r^2          # geo.area(2) → 12.566
```

---

## Differential operators and solvers

The `ops` and `solver` namespaces provide gridded calculus for PDE work.
Every finite-difference operator takes the physical grid spacing `dx` as a
**required** argument — forgetting it (i.e. differentiating in index units) is a
classic and silent source of wrong results.

**`ops`** — spatial operators on a periodic grid:

```
> ops.grad(T, dx)          # central diff along every axis → trailing component axis
> ops.grad(T, dx, axis)    # derivative along one axis (same shape as T)
> ops.div(V, dx)           # divergence of a vector field (trailing component axis)
> ops.curl(V, dx)          # 2-D scalar curl
> ops.lap(T, dx)           # Laplacian (periodic); ops.lap(T, dx, ops.neumann) for no-flux
> ops.poisson(rhs, dx)     # spectral solve of ∇²u = rhs (zero-mean), returns a real field
> ops.specgrad(T, dx)      # spectral derivative via i·k (machine-precision on smooth fields)
```

**`solver`** — time integration and stability diagnostics:

```
> solver.rk4(f, y0, t0, t1, n)   # fixed-step RK4; f is dy/dt = f(t, y); state scalar or tensor
> solver.rk4((t,y)->y, 1, 0, 1, 100)        # ≈ e
> solver.odeint(f, y0, ts)       # RK4 sampled at the times in ts → stacked trajectory
> solver.cfl(V, dx, dt)          # Courant number dt·max|V|/dx (stability check)
```

`examples/fluid2D.math` is a worked 2-D turbulence simulation built on these.

---

## Fields and differential forms

A **field** packages a tensor of grid samples with the geometry needed to do
calculus on it — the box it lives on, the boundary conditions, the grid spacing,
and a metric. Build one with `field`:

```
> field(data, lo, hi, bc [, metric])
> f = field(tensor(k -> sin(2*pi*k/64), 64), 0, 2*pi, forms.periodic)
0-form [64] on [0, 6.283185307179586] periodic
…
```

* `lo`/`hi` are the lower/upper corners (a scalar broadcasts to every axis, or
  pass a tuple per axis). The spacing `dx` is derived from them: a `forms.periodic`
  axis treats `[lo, hi)` as a torus (`dx = (hi-lo)/N`); a `forms.neumann` (no-flux)
  axis includes both endpoints (`dx = (hi-lo)/(N-1)`).
* `bc` is `forms.periodic` or `forms.neumann`, per axis.
* `metric` is the optional **diagonal metric** `g_ii` (default all `1`, i.e.
  Euclidean). Minkowski is just `(-1, 1, 1, 1)`; an anisotropic grid is any other
  diagonal.

A field is a *k-form*: a scalar field is a 0-form, a gradient is a 1-form, and so
on. A k-form on an n-D grid has C(n,k) components, stored on a trailing axis.
Arithmetic (`+ - *` by scalars and matching fields) stays inside the field
algebra; any named builtin (`abs`, `max`, `sum`, `re`, …) decays a field to its
underlying tensor.

The **`forms`** namespace is exterior calculus:

```
> forms.d(f)            # exterior derivative: k-form → (k+1)-form
                        #   (grad on a 0-form, curl on a 1-form, div on an (n-1)-form)
> forms.hodge(f)        # Hodge star ★: k-form → (n-k)-form
> forms.wedge(a, b)     # exterior product ∧: (k-form, l-form) → (k+l)-form
> forms.raise(f)        # musical sharp ♯: form → vector field (raise indices)
> forms.lower(f)        # musical flat  ♭: vector field → form (lower indices)
> forms.codiff(f)       # codifferential δ = ±★d★: k-form → (k-1)-form
> forms.laplace(f)      # Laplace–de Rham Δ = dδ + δd
```

The two per-axis quantities are kept strictly separate. **The grid spacing `dx`
enters only `d`** (and grad/curl/div, which are `d` in disguise); `d` is otherwise
metric-free. **The metric enters only `hodge`/`raise`/`lower`/`codiff`/`laplace`.**
That separation is what makes the same code do Euclidean and Minkowski geometry:
on a Euclidean grid `raise`/`lower` are the identity and `forms.laplace` of a
0-form is `−∇²`; flip the timelike sign with metric `(-1,1,1,1)` and `forms.laplace`
becomes the d'Alembertian `□ = −∂ₜ² + ∇²` — no other change.

The `ops` operators are **field-polymorphic**: call `ops.grad`, `ops.div`,
`ops.curl`, `ops.lap`, `ops.specgrad`, or `ops.poisson` on a field (with no `dx`
argument) and they read the spacing and boundary conditions from the field and
return a field. `ops.lap(f)` uses the compact 3-point stencil (distinct from
`forms.laplace`'s wider `δd` stencil), and `ops.poisson(f)` does the spectral
Poisson solve — both staying inside the field algebra.

---

## Tensor I/O

Two serialization formats are available for saving and restoring tensors between sessions or exchanging data with other tools. Both handle real and complex tensors transparently. All tensor I/O commands work in `.math` files as well as the REPL.

### Native format (`.mlt`) — `!savetensor` / `!loadtensor`

`.mlt` is a compact binary format: 8-byte magic `MLTENSOR`, a type tag, ndim, shape, then raw `f64` data — all little-endian. One tensor per file. No dependencies; always available.

```
!savetensor <var> <file>
!loadtensor <var> <file>
```

```
> A = (1,2,3; 4,5,6)
> !savetensor A /tmp/matrix.mlt
saved A (6 elements, real) to /tmp/matrix.mlt
> !loadtensor B /tmp/matrix.mlt
loaded B (6 elements, real) from /tmp/matrix.mlt
> shape(B)
result = [2, 3]
```

Complex tensors (e.g. FFT output) are saved and restored automatically — the format stores real and imaginary blocks separately:

```
> C = fft([1, 1, 0, 0])
> !savetensor C /tmp/fft.mlt
saved C (4 elements, complex) to /tmp/fft.mlt
> !loadtensor D /tmp/fft.mlt
loaded D (4 elements, complex) from /tmp/fft.mlt
> im(D[1])
result = -1
```

### HDF5 format — `!savehdf5` / `!loadhdf5`

HDF5 is the standard scientific data format, natively supported by NumPy (`h5py`), MATLAB, Julia, R, and most scientific toolchains. It supports multiple datasets per file, hierarchical groups, and optional compression.

**Build requirement:** pass `--features hdf5` at build time (requires `libhdf5` to be installed):

```zsh
cargo build --release --features hdf5
# Arch/Manjaro:  sudo pacman -S hdf5
# Ubuntu/Debian: sudo apt install libhdf5-dev
# macOS:         brew install hdf5
```

Without the feature, both commands print a helpful message and do nothing.

#### Saving

```
!savehdf5 <var> <file> [/dataset] [--append] [--overwrite] [--gzip <0–9>]
```

| Option | Effect |
|--------|--------|
| `/dataset` | HDF5 path for the dataset (default: `/<varname>`). Supports nested groups: `/results/run1/temperature` — intermediate groups are created automatically. |
| `--append` | Add to an existing file instead of truncating it. Without this flag a new file is always created. |
| `--overwrite` | Replace an existing dataset at that path. Without this flag, writing to an occupied path is an error. |
| `--gzip <n>` | Compress with gzip at level `n` (0 = none, 9 = max). Chunked storage is enabled automatically when compression is requested. |

```
> M = rand(100, 100)
> !savehdf5 M results.h5
saved M (10000 elements) → results.h5:/M

> T = ones(10, 10)
> !savehdf5 T results.h5 /run1/temperature --append --gzip 6
saved T (100 elements) → results.h5:/run1/temperature
```

#### Loading

```
!loadhdf5 <var> <file> [/dataset] [--list]
```

| Option | Effect |
|--------|--------|
| `/dataset` | HDF5 path to load (default: `/<varname>`). |
| `--list` | Print the dataset tree of the file instead of loading anything. |

```
> !loadhdf5 N results.h5 /M
loaded N (10000 elements, real) ← results.h5:/M
> shape(N)
result = [100, 100]

> !loadhdf5 _ results.h5 --list
M  [100×100  f64]
run1/
  temperature  [10×10  f64]
```

#### Complex tensors in HDF5

Complex tensors are stored as a group containing `re` and `im` datasets with an `mlt_complex` attribute. This layout is directly readable from Python:

```python
import h5py, numpy as np

with h5py.File("data.h5") as f:
    # real tensor — standard dataset
    M = f["/M"][:]

    # complex tensor saved from mathlang
    re = f["/C/re"][:]
    im = f["/C/im"][:]
    C  = re + 1j * im
```

```
> C = fft([1, 1, 0, 0])
> !savehdf5 C data.h5
saved C (4 elements, complex) → data.h5:/C
> !loadhdf5 D data.h5 /C
loaded D (4 elements, complex) ← data.h5:/C
> re(D[0]), im(D[1])
result = 2  -1
```
