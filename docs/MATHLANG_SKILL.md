# mathlang skill file
# Read this to quickly write correct .math files and CLI expressions.

---

## Syntax rules

```
m 'def; def; ... ; expr, expr'   # CLI: ;-separated statements; last is the output
# In .math files: one definition per line; # is a comment
# In REPL: !include file.math to load; !defs to inspect
```

- `;` separates statements (definitions and expressions). The **last expression**
  is the output. (There is no `:` separator — it was removed.)
- Multiple comma-separated outputs are allowed in the final expression position:
  `x=3; y=4; x^2, y^2`.
- `{stmt; stmt; ... ; expr}` — block with local scope; returns the last expression.
- Blocks can be nested and appear anywhere an expression is expected.
- Note: at the **top level** only definitions may precede the final expression. To
  sequence side-effecting expression statements (e.g. `set` on a cell), wrap them
  in a `{ ... }` block.

---

## Literals and constants

```
42  3.14  1e-6          # numbers
2i   3+4i   -i          # complex: i is the imaginary unit
(1, 2, 3)               # → 1-D Tensor [1, 2, 3]  (auto-promoted)
(1, 2; 3, 4)            # → 2-D Tensor (matrix literal, rows sep by ;)
```

**Constants:** `pi`, `e`, `phi`, `inf`, `i`  
`(a, b, c)` with all-numeric elements becomes a `Tensor`, not a Tuple.  
Heterogeneous collections (mixed types) stay as `Tuple`.

---

## Operators

| Op | Meaning |
|----|---------|
| `+ - * / ^ **` | arithmetic (`^` right-assoc) |
| `//` | floor division |
| `%` | remainder |
| `< > <= >= == !=` | comparison → 0 or 1 |
| `& \| && \|\|` | bitwise AND / OR |
| `n!` | postfix factorial |
| `A @ B` | matmul (2D×2D, 2D×1D, 1D dot) |
| `2x`, `3sin(x)` | implicit multiply (same precedence as `*`) |

Arithmetic on tensors broadcasts element-wise.  
`2x^2` parses as `(2x)^2` — use explicit `*` when needed.

---

## Variables and user functions

```
x = 3                          # variable
f(x) = x^2                     # named function
g(x, y) = x^2 + y^2
f = x -> x^2                   # lambda (single arg)
f = (x, y) -> x + y            # lambda (multi-arg)
f = x, y -> x * y              # same, parens optional
```

- **Parameter shadowing**: parameter names override globals. `f(pi)=pi+1; f(2)` → `3` (pi inside body is the parameter, not 3.14…).  
- You **cannot** redefine the core built-in names (`step`, `sort`, `sin`, etc.) or the
  namespace names (`ops`, `bits`, …). But names that live in a namespace are no
  longer reserved, so `xor`, `lerp`, `var`, `qr`, … are free to use as your own variables.
- Functions are first-class: pass as arguments, return from functions, store in variables.

---

## Control flow

```
if(cond, a, b)                 # lazy: evaluates only chosen branch
if(cond, f, g)(x)              # select and call
```

Piecewise:
```
abs2 = x -> if(x >= 0, x, -x)
sign2 = x -> if(x > 0, 1, if(x < 0, -1, 0))
```

---

## Blocks

```
{x=3; y=4; x^2+y^2}          # 25, local scope
1 + {a=2; a*3}                # 7
f(n) = {half = n/2; half^2}   # block in function body
```

---

## Tensors (the primary array type)

`(a,b,c)` with numbers → 1-D Tensor displayed as `[a, b, c]`.  
`(a,b; c,d)` → 2-D Tensor (matrix).  
`(x,)` → length-1 Tensor `[x]` (trailing comma); `(x)` alone is just the scalar `x`.

### Constructors
```
zeros(n)               # 1-D zeros
zeros(r, c)            # 2-D zeros
ones(n1, n2, n3)       # any-rank, filled with 1
eye(n)                 # n×n identity
diag(v)                # diagonal matrix from 1-D tensor/tuple
rand()                 # scalar in [0,1)
rand(n)                # 1-D random tensor length n
rand(r, c)             # r×c random matrix
rand(n1, n2, n3)       # any-rank random tensor
linspace(a, b, n)      # n evenly spaced values in [a,b]
range(a, b)            # integers [a, b)  (like Python range)
matrix((i,j)->expr, r, c)         # 2-D, 0-indexed
tensor((i,j,k)->expr, n1, n2, n3) # any-rank, 0-indexed
```

### Indexing (0-based, negative counts from end)
```
T[i]            # element (1-D) or row (2-D+)
T[i, j]         # element of 2-D tensor
T[i, j, k]      # element of 3-D tensor
T[-1]           # last element / last row
T[2, -1]        # row 2, last column
```

### Slicing
```
T[a..b]         # inclusive range a to b along first axis
T[a..]          # from a to end
T[..b]          # from start to b (inclusive)
T[..]           # all elements
T[.., j]        # all rows, column j  →  1-D tensor
T[i, ..]        # row i  →  1-D tensor
T[.., .., k]    # all of axes 0,1; index k on axis 2
```

### Shape operations
```
shape(T)        # 1-D tensor of dims
rows(T), cols(T)
reshape(T, n1, n2, ...)
flatten(T)      # → 1-D
transpose(T)                  # reverses all axes
transpose(T, a, b)            # swap axes a and b
permute(T, p0, p1, ...)       # reorder axes
squeeze(T)                    # remove size-1 dims
unsqueeze(T, dim)             # insert size-1 dim
```

### Concatenation
```
cat(axis, T1, T2, ...)        # concat along axis
hstack(A, B)                  # cat(1,...); a 1-D vector is treated as a column
vstack(A, B)                  # cat(0,...); a 1-D vector is treated as a row
append(v, x)                  # grow a 1-D tensor/tuple; scalar v → [v, x]
concat(a, b)                  # join 1-D tensors/tuples; accepts scalars & empties
```
`hstack`/`vstack` rank-promote, so a vector stacks directly onto a matrix (e.g.
`vstack((1,2;3,4), (5,6))` → 3×2). Note `(x,)` is a length-1 tensor (unlike `(x)`,
which is just the scalar `x`) — useful as an accumulator base case.

### Linear algebra
```
A @ B           # matmul
det(A), inv(A), trace(A), norm(T)
solve(A, b)     # solve Ax=b  →  1-D tensor
row(T, i), col(T, j)          # extract row/col as 1-D tensor
outer(a, b)                   # outer product
tensordot(A, B, n)            # contract n innermost axes
```

### Reductions
```
sum(T)           # scalar total
sum(T, axis)     # reduce along axis  →  lower-rank tensor
prod(T), prod(T, axis)
mean(T), std(T), var(T), median(T), mode(T)
min(T), max(T)   # scalar extrema
min(a, b), max(a, b)          # 2-arg form
argmin(T)        # flat index (1-D) or index tensor (n-D)
argmax(T)
norm(T)          # Frobenius / Euclidean norm
```

### Grid construction
```
lingrid(start, end, counts, f)
# start, end, counts: scalar (1-D) or same-length tuples (n-D)
# f receives grid coordinates, may return scalar, tuple, or tensor
# output shape = grid_shape ++ value_shape

# 1-D example:
lingrid(0, 1, 5, x -> x^2)                        # shape [5]

# 2-D scalar field:
lingrid((-1,-1),(1,1),(10,10),(x,y)->x^2+y^2)      # shape [10,10]

# 2-D vector field:
lingrid((-1,-1),(1,1),(10,10),(x,y)->(x,y))        # shape [10,10,2]
```

---

## Complex numbers and ComplexTensor

```
2+3i, -i, 1i           # complex literals
re(z), im(z)           # real/imaginary parts
abs(z), arg(z)         # modulus, argument
conj(z)                # conjugate

# ComplexTensor: created when a tensor contains complex values
tensor(k -> k+k*i, 6)  # ComplexTensor of shape [6]
```

---

## Spectral (FFT)

`fft`/`ifft` are **n-dimensional** and transform over *all* axes by default.
(There is no `fftn`/`ifftn` — `fft` already does that.)

```
fft(T)             # n-D FFT over all axes  →  ComplexTensor
fft(T, axis)       # FFT along one axis
fft(T, (a,b))      # FFT along axes a and b
fft(Re, Im)        # FFT of Re+i*Im input (same-shape real tensors)
fft(Re, Im, axes)  # same, specified axes
ifft(T)            # n-D inverse FFT (same argument forms)
```

---

## Higher-order functions

```
map(f, T)                      # apply f element-wise, preserves shape
filter(f, v)                   # keep elements where f(x) != 0  →  1-D
reduce((a,b)->expr, v)         # left fold
compose(f, g)                  # x -> f(g(x))
partial(f, a)                  # x -> f(a, x)
iterate(f, x0, n)              # f^n(x0): apply f n times (flat loop, O(1) stack)
scan(f, x0, n)                 # orbit [x0, f(x0), …, f^n(x0)] stacked into a tensor
```

`iterate`/`scan` are the **preferred** way to time-step or iterate to a fixed
point — they loop internally (O(1) stack, scale to millions of steps), unlike
recursion which is capped at 100000 nested calls. `x0` may be a scalar, a vector,
a tuple, or complex. For `scan`, scalar states stack into a 1-D tensor of length
`n+1`; **vector** states of width `d` stack into a 2-D tensor `[n+1, d]`, one
state per row — so a whole trajectory is a one-liner:

```
iterate(x -> 2*x, 1, 10)                 # 1024
scan(x -> 2*x, 1, 4)                     # [1, 2, 4, 8, 16]
scan(v -> (v[1], -v[0]), (1,0), 100)     # [101, 2] orbit of a harmonic oscillator
```

---

## Calculus

```
integral(f, a, b)              # Simpson's rule, 1000 steps
integral(f, a, b, n)           # custom step count
deriv(f, x)                    # 5-point stencil, dx=1e-5
deriv(f, x, dx)                # custom step
sum(f, a, b)                   # integer sum Σ_{k=a}^{b} f(k)
prod(f, a, b)                  # integer product
```

---

## Built-in math functions

**Trig:** `sin cos tan asin acos atan atan2(y,x) sinh cosh tanh sec csc cot`  
**Algebra:** `sqrt cbrt abs sign step floor ceil round(x) round(x,n) trunc frac exp ln log log2 log10 log(x,base) pow(x,y) hypot(a,b)`  
**Angle:** `deg(x)` rad→deg, `rad(x)` deg→rad  
**Number theory:** `gcd(a,b) lcm(a,b) fact(n)` / `n!`  
**Complex:** `re im abs arg conj`  
**Comparison fns:** `lt leq gt geq eq neq` — 2-arg, return 0/1; good with `map`/`filter`  
**Misc:** `len(v)` / `length(v)`, `sort(v)`, `zip(a,b)`, `dot(a,b)`, `append(v,x)`, `concat(a,b)`, `flatten(v)`  
**Array scans:** `cumsum(v)`, `cumprod(v)`, `diff(v)` — running sum/product, first difference (over a 1-D tensor or tuple)  
**Plotting:** `graph(f)` or `graph(f, a, b)` — saves graph_N.png

---

## Namespaces (`.` access)

Niche functions live in namespaces, accessed as `ns.member`. They are **not**
reserved words, so the bare names (`xor`, `lerp`, `var`, …) are free for your own
variables. Browse with `!help <namespace>`.

```
special.{erf erfc j0 j1 jinc sinc sech csch gaussian gaussian_cdf delta}
bits.{and or xor nand nor xnor shl shr not}
stats.{median mode var}            # mean, std stay flat
linalg.{qr diagonalize tensordot outer eig_top eig_bot}   # det inv solve eig eigvals stay flat
vec.{lerp clamp}
forms.{d hodge wedge raise lower codiff laplace}   # exterior calculus on fields
```

**Differential operators / solvers** (for PDE work — `dx` is always required):

```
ops.grad(T, dx [, axis])     # central difference; all-axes form adds trailing component axis
ops.div(V, dx)  ops.curl(V, dx)            # vector-field ops
ops.lap(T, dx [, ops.neumann])             # Laplacian; default periodic, neumann = no-flux
ops.poisson(rhs, dx)         # spectral ∇²u = rhs (zero-mean), returns a real field
ops.specgrad(T, dx [, axis]) # spectral derivative via i·k
solver.rk4(f, y0, t0, t1, n)       # fixed-step RK4; f is dy/dt = f(t, y)
solver.odeint(f, y0, ts)           # RK4 sampled at the times in ts → stacked trajectory
solver.cfl(V, dx, dt)              # Courant number dt·max|V|/dx
```

**Fields and differential forms.** A `field` packages grid samples with their
geometry (box, boundary conditions, derived spacing, diagonal metric):

```
field(data, lo, hi, bc [, metric])   # bc = forms.periodic | forms.neumann (per axis)
f = field(tensor(k -> sin(2*pi*k/64), 64), 0, 2*pi, forms.periodic)
# metric defaults to Euclidean; Minkowski is just (-1, 1, 1, 1)
```

A field is a *k-form* (0-form = scalar, 1-form = gradient, …) with C(n,k) trailing
components. `field` makes a 0-form; build higher forms / vector fields directly
with `forms.form(data, degree, lo, hi, bc[, metric])` and
`forms.vector(data, lo, hi, bc[, metric])`. `forms.*` is exterior calculus:

```
forms.d(f)        # exterior derivative: k-form → (k+1)-form (grad/curl/div unified)
forms.hodge(f)    # Hodge star ★: k-form → (n-k)-form
forms.wedge(a,b)  # exterior product ∧
forms.raise(f) / forms.lower(f)   # musical isomorphisms ♯ / ♭ (form ↔ vector field)
forms.contract(X,w) # interior product ι_X: vector + k-form → (k-1)-form (metric-free)
forms.codiff(f)   # codifferential δ = ±★d★
forms.laplace(f)  # Laplace–de Rham Δ = dδ + δd
```

`forms.contract` is the natural vector–form pairing: at degree 1 it is the scalar
⟨ω,X⟩ = Σ_i ω_i X^i. Container builtins (`cell`/`get`/`set`/`id`) keep a field as a
field; other named builtins decay it to its tensor.

Design rule: **`dx` enters only `d`** (which is therefore metric-free), while the
**metric enters only `hodge`/`raise`/`lower`/`codiff`/`laplace`**. Hence the same
code does Euclidean (`forms.laplace` of a 0-form = −∇²) and Minkowski (metric
`(-1,1,1,1)` ⇒ `forms.laplace` = d'Alembertian □ = −∂ₜ² + ∇²). The `ops.*`
operators are field-polymorphic: `ops.lap(f)`, `ops.grad(f)`, `ops.poisson(f)`,
etc. read `dx`/bc from the field and return a field (`ops.lap` uses the compact
stencil, vs `forms.laplace`'s wider δd stencil).

**User namespaces:** `!namespace foo` at the top of an included `.math` file
collects its public definitions into `foo`; prefix a def with `private` to keep it
internal. See `examples/fluid2D.math` for operators in action.

---

## Common patterns and idioms

### Shifting a 2-D tensor (Neumann / no-flux BC)
```
# T shifted "up" (row i gets row i-1; row 0 gets row 0):
T_up = vstack(T[0..0,..], T[0..R-2,..])

# T shifted "down" (row i gets row i+1; last row stays):
T_down = vstack(T[1..R-1,..], T[R-1..R-1,..])

# Similarly for left/right with hstack
```

### 2-D discrete Laplacian
```
lap2d(T) = {
  R = rows(T); C = cols(T);
  T_up    = vstack(T[0..0,..],    T[0..R-2,..]);
  T_down  = vstack(T[1..R-1,..],  T[R-1..R-1,..]);
  T_left  = hstack(T[..,0..0],    T[..,0..C-2]);
  T_right = hstack(T[..,1..C-1],  T[..,C-1..C-1]);
  T_up + T_down + T_left + T_right - 4*T
}
```

### Time-stepping

**Preferred:** drive the loop with `iterate` (final state) or `scan` (whole
trajectory). Both run a flat internal loop — O(1) stack, scaling to millions of
steps — and `step` here can carry any state (tensor, vector, scalar, complex):
```
evolve(T0, n) = iterate(step, T0, n)             # final state after n steps
solver(T0) = t -> iterate(step, T0, round(t/dt)) # returns a solver lambda
traj = scan(step, T0, n)                          # every intermediate state
```

Recursion still works for **short** runs and reads naturally, but is capped at
100000 nested calls (beyond that you get a catchable `recursion limit exceeded`
error, not a crash), so prefer `iterate`/`scan` for anything long:
```
evolve(T, n) = if(n <= 0, T, evolve(step(T), n-1))
```

The older `cell`-driven flat loop (`{c = cell(T0); sum(k -> {set(c, step(get(c)));
0}, 1, n); get(c)}`) is still valid but is superseded by `iterate`/`scan` — reach
for a `cell` only when you genuinely need mutable state outside an iteration.

### Building initial conditions with spatial logic
```
tensor((i,j) -> if(condition_on_ij, val_a, val_b), N, N)
# center a disk: (i-cx)^2 + (j-cy)^2 <= R^2
```

### Element-wise function of a tensor
```
map(sin, T)           # equivalent to sin(T) for elt-wise builtins
map(x -> x^2, T)      # any lambda
```

### Named result extraction
```
{x=3; y=4; x, y}         # outputs two values: 3  4
```

---

## Gotchas and quirks

- **`step` is a built-in** (Heaviside). Don't name your function `step`.
- **`i` is the imaginary unit.** Using `i` as a loop index inside `tensor()` is fine (parameter shadows global), but outside `tensor()` it means √−1. Prefer `k`, `r`, `c` etc. for indices.
- **`pi` is shadowed by parameters** too: `f(pi)=pi+1; f(2)` → 3 (pi inside = parameter = 2).
- **Implicit multiply pitfall:** `2x^2` = `(2x)^2`, not `2*(x^2)`. Use `2*x^2` when in doubt.
- **No indexed assignment:** `T[i,j] = v` does NOT work. Build tensors from scratch with `tensor(...)`.
- **Mutable state via cells only.** Variables are immutable, but `cell(v)` / `get(c)`
  / `set(c, v)` provide an explicit mutable reference. This is the scalable way to
  carry state across a long flat loop (see *Time-stepping* above).
- **Negative indices are bounds-checked:** `(1,2,3)[-1]` → 3, but `(1,2,3)[-4]` is an
  out-of-range error (it does not wrap).
- **Range slices are inclusive on both ends:** `T[1..3]` gives elements 1, 2, 3 (4 elements if 0-indexed).
- **`range(a,b)` is exclusive on the right** (like Python): `range(0,5)` → `[0,1,2,3,4]`.
- **`linspace(a,b,n)` is inclusive on both ends:** n points from a to b.
- **`argmax`/`argmin`** on 1-D → scalar; on n-D → 1-D index tensor `[i, j, ...]`.
- **`solve(A,b)`** returns a 1-D tensor (not tuple).
- **`.math` files** use `;` to end each line only if putting multiple defs on one line. One def per line works fine without `;`.
- **Blocks in .math files:** definitions inside a block are local, so wrap multi-step computations in `{ defs; ...; result }`.
