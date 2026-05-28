# mathlang skill file
# Read this to quickly write correct .math files and CLI expressions.

---

## Syntax rules

```
m 'defs; ... : expr, expr'   # CLI: definitions before :, outputs after
# In .math files: one definition per line; # is a comment
# In REPL: !include file.math to load; !defs to inspect
```

- `;` separates definitions.  `:` separates definitions from output expressions.
- Without `:`, the whole input is the output expression.
- `{defs; ... : expr}` — block with local scope; returns last expr.
- Blocks can be nested and appear anywhere an expression is expected.

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
- You **cannot** redefine built-in names (`step`, `sort`, `sin`, etc.).
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
{x=3; y=4 : x^2+y^2}          # 25, local scope
1 + {a=2 : a*3}                # 7
f(n) = {half = n/2 : half^2}   # block in function body
```

---

## Tensors (the primary array type)

`(a,b,c)` with numbers → 1-D Tensor displayed as `[a, b, c]`.  
`(a,b; c,d)` → 2-D Tensor (matrix).

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
hstack(A, B)                  # cat(1,...)
vstack(A, B)                  # cat(0,...)
```

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

```
fft(v)             # 1-D FFT of 1-D tensor  →  ComplexTensor
ifft(v)            # 1-D IFFT
fftn(T)            # n-D FFT over all axes
fftn(T, axis)      # FFT along one axis
fftn(T, (a,b))     # FFT along axes a and b
fftn(Re, Im)       # FFT of Re+i*Im input (same shape tensors)
ifftn(T)           # n-D IFFT
```

---

## Higher-order functions

```
map(f, T)                      # apply f element-wise, preserves shape
filter(f, v)                   # keep elements where f(x) != 0  →  1-D
reduce((a,b)->expr, v)         # left fold
compose(f, g)                  # x -> f(g(x))
partial(f, a)                  # x -> f(a, x)
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
**Special:** `sinc sech csch erf erfc j0 j1 jinc gaussian(x,mu,sigma) gaussian_cdf delta(x)`  
**Number theory:** `gcd(a,b) lcm(a,b) fact(n)` / `n!`  
**Complex:** `re im abs arg conj`  
**Comparison fns:** `lt leq gt geq eq neq` — 2-arg, return 0/1; good with `map`/`filter`  
**Bitwise:** `and or xor nand nor xnor not shl(x,n) shr(x,n)`  
**Misc:** `len(v)` / `length(v)`, `sort(v)`, `zip(a,b)`, `dot(a,b)`, `append(v,x)`, `concat(a,b)`, `flatten(v)`  
**Plotting:** `graph(f)` or `graph(f, a, b)` — saves graph_N.png

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

### Recursive time-stepping
```
evolve(T, n) = if(n <= 0, T, evolve(step(T), n-1))
# Returns a solver lambda: physical time → state
solver(T0) = t -> evolve(T0, round(t/dt))
```

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
{x=3; y=4 : x, y}         # outputs two values: 3  4
```

---

## Gotchas and quirks

- **`step` is a built-in** (Heaviside). Don't name your function `step`.
- **`i` is the imaginary unit.** Using `i` as a loop index inside `tensor()` is fine (parameter shadows global), but outside `tensor()` it means √−1. Prefer `k`, `r`, `c` etc. for indices.
- **`pi` is shadowed by parameters** too: `f(pi)=pi+1; f(2)` → 3 (pi inside = parameter = 2).
- **Implicit multiply pitfall:** `2x^2` = `(2x)^2`, not `2*(x^2)`. Use `2*x^2` when in doubt.
- **No indexed assignment:** `T[i,j] = v` does NOT work. Build tensors from scratch with `tensor(...)`.
- **No mutable state.** Time evolution must use recursion.
- **Range slices are inclusive on both ends:** `T[1..3]` gives elements 1, 2, 3 (4 elements if 0-indexed).
- **`range(a,b)` is exclusive on the right** (like Python): `range(0,5)` → `[0,1,2,3,4]`.
- **`linspace(a,b,n)` is inclusive on both ends:** n points from a to b.
- **`argmax`/`argmin`** on 1-D → scalar; on n-D → 1-D index tensor `[i, j, ...]`.
- **`solve(A,b)`** returns a 1-D tensor (not tuple).
- **`.math` files** use `;` to end each line only if putting multiple defs on one line. One def per line works fine without `;`.
- **Blocks in .math files:** definitions before `:` are local, so wrap multi-step computations in `{ ... : result }`.
