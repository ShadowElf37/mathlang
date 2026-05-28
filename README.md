# m — command-line math

A fast expression evaluator written in Rust, hooked up to your shell as `m`.

## Syntax

```
m [definitions :] expression [, expression ...]
```

Definitions are separated by `;`. A `:` separates them from the output expressions.
Without a `:`, the whole input is evaluated as expression(s).

> **Always quote the argument** to avoid shell interference (`^` glob, `;` command separator, `*` glob, `>` redirect). The REPL needs no quoting.

---

## Quick examples

```zsh
m '3 + 4'                          # 7
m '2^10'                           # 1024
m 'pi * 2^2'                       # 12.566370614359172
m 'sqrt(2), sin(pi/2)'             # 1.4142135623730951  1
m 'x=3; y=4 : x^2 + y^2'          # 25
m 'x=3; y=4 : sqrt(x^2 + y^2)'    # 5
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

```zsh
m '2pi'          # 6.283185307179586
m '3sin(pi/2)'   # 3
m '2(x+1)'       # expands to 2*(x+1)
```

Precedence is the same as `*`, so `2x^2` parses as `(2x)^2`. Use explicit `*` when that matters.

---

## Constants

`pi`, `e`, `phi`, `inf`, `i` (imaginary unit)

---

## Variables and functions

```zsh
m 'x = 3 : x^2'               # 9
m 'a=2; b=3 : a^2 + b^2'      # 13
m 'f(x) = x^2 : f(3), f(4)'   # 9  16
m 'g(x,y) = x^2 + y^2 : g(3,4)' # 25
```

Parameter names shadow globals — `f(pi) = pi+1; f(2)` → `3` (the parameter `pi` = 2, not 3.14…).

---

## Anonymous functions (lambdas)

Single-argument: `x -> expr`. Multi-argument: `x, y -> expr` or `(x, y) -> expr`.

```zsh
m 'f = x -> x^2 : f(3)'                                   # 9
m 'ncr = n, r -> fact(n)/(fact(r)*fact(n-r)) : ncr(5,2)'  # 10
```

Lambdas are first-class — pass them to functions, apply them inline:

```zsh
m '(x -> x^2)(5)'         # 25
m 'sum(x -> x^2, 1, 10)'  # 385
```

---

## Vectors (1-D tensors)

Comma-separated numeric values in parentheses produce a **1-D tensor** displayed as `[...]`. Index with `[n]` (zero-based; negative indices count from the end).

```zsh
m '(1, 2, 3)'               # [1, 2, 3]
m '(1, 2, 3)[1]'            # 2
m '(1, 2, 3)[-1]'           # 3
```

Arithmetic broadcasts element-wise:

```zsh
m '(1, 2, 3) * 2'            # [2, 4, 6]
m '(1, 2, 3) + (4, 5, 6)'    # [5, 7, 9]
m '(10, 20, 30) / 10'        # [1, 2, 3]
m '(1, 2, 3) @ (1, 2, 3)'    # 14   (dot product)
```

### Slicing

```zsh
m '(1,2,3,4,5)[1..3]'       # [2, 3, 4]   — bounded range (inclusive)
m '(1,2,3,4,5)[2..]'        # [3, 4, 5]   — from index 2 to end
m '(1,2,3,4,5)[..2]'        # [1, 2, 3]   — from start to index 2 (inclusive)
m '(1,2,3,4,5)[..]'         # [1, 2, 3, 4, 5]  — all elements
m 'len((1,2,3,4))'          # 4
```

---

## Blocks

Blocks `{...}` create a local scope. Use `;` to separate definitions, `:` before the output expression(s).

```zsh
m '{x = 3; y = 4 : x^2 + y^2}'   # 25
```

Blocks can appear anywhere an expression is expected (inside function bodies, inline in CLI args, etc.).

---

## Comparisons

Comparison operators return `1` (true) or `0` (false):

```zsh
m '3 < 5'      # 1
m '3 > 5'      # 0
m '3 == 3'     # 1
m '3 != 4'     # 1
m '3 <= 3'     # 1
```

Combined with `if`:

```zsh
m 'x=4 : if(x > 0, sqrt(x), 0)'   # 2
```

---

## `if`

`if(cond, a, b)` returns `a` when `cond` is nonzero, `b` otherwise. Branches are evaluated lazily.

```zsh
m 'if(1, 10, 20)'               # 10
m 'if(0, 10, 20)'               # 20
```

`if` works with functions, enabling piecewise definitions:

```zsh
m 'f = x -> if(x >= 0, sqrt(x), 0) : f(4), f(-1)'   # 2  0
m 'sign2 = x -> if(x > 0, 1, if(x < 0, -1, 0))'
```

`if(cond, fn1, fn2)(x)` evaluates the chosen function on `x`:

```zsh
m 'if(1, x -> x^2, x -> x^3)(5)'   # 25
```

---

## Calculus

```zsh
m 'integral(x -> x^2, 0, 1)'           # 0.3333333333333333
m 'integral(x -> sin(x), 0, pi)'       # 2
m 'integral(f, a, b, n)'               # Simpson's rule with n steps (default 1000)
m 'deriv(x -> x^3, 2)'                 # 12  (5-point stencil, dx=1e-5)
m 'deriv(f, x, dx)'                    # custom step size
```

---

## Aggregates

```zsh
m 'sum(x -> x, 1, 100)'        # 5050
m 'sum(x -> x^2, 1, 10)'       # 385
m 'prod(x -> x, 1, 10)'        # 3628800  (10!)
m 'sum((1, 2, 3, 4))'          # 10       — sum over a 1-D tensor
m 'prod((1, 2, 3, 4))'         # 24
```

`sum` and `prod` also work on tensors, including axis-wise reduction:

```zsh
m 'sum(ones(3, 4))'            # 12
m 'sum(ones(2, 3), 0)'         # [2, 2, 2]  — reduce along axis 0
m 'sum(ones(2, 3), 1)'         # [3, 3]     — reduce along axis 1
```

`map(f, t)` applies `f` to each element and preserves the container type:

```zsh
m 'map(x -> x^2, (1,2,3,4))'  # [1, 4, 9, 16]
m 'map(x -> x*2, eye(3))'     # 3×3 matrix with 2s on diagonal
```

`filter(f, v)` keeps elements where `f` returns nonzero:

```zsh
m 'filter(x -> x > 2, (1,2,3,4))'   # [3, 4]
```

`reduce(f, t)` left-folds with a 2-arg function, on 1-D tensors or matrices:

```zsh
m 'reduce((a,b) -> a+b, (1,2,3,4))'          # 10
m 'reduce((a,b) -> if(a>b,a,b), ones(3,3))'  # 1  — max over all elements
```

---

## Tensors and matrices

Tensors are n-dimensional arrays of real numbers. A 1-D tensor displays as `[...]`, a 2-D tensor as a boxed matrix.

### Constructors

```zsh
m 'zeros(3, 4)'                       # 3×4 zero matrix
m 'ones(2, 3, 4)'                     # 2×3×4 tensor of ones (any rank)
m 'eye(3)'                            # 3×3 identity matrix
m 'diag((1, 2, 3))'                   # diagonal matrix from 1-D tensor
m 'rand()'                            # scalar in [0, 1)
m 'rand(n)'                           # 1-D random tensor of length n
m 'rand(r, c)'                        # r×c random matrix
m 'rand(n1, n2, n3)'                  # any-rank random tensor
m 'linspace(0, 1, 5)'                 # [0, 0.25, 0.5, 0.75, 1]
m 'range(0, 5)'                       # [0, 1, 2, 3, 4]  (exclusive end)
m 'matrix((i,j) -> i*3+j, 2, 3)'     # 2×3 matrix filled by function (0-indexed)
m 'tensor((i,j,k) -> i+j+k, 2,3,4)'  # arbitrary n-D tensor
```

Matrix literal syntax — semicolons separate rows inside parentheses:

```zsh
m '(1, 2; 3, 4)'      # ⎡ 1  2 ⎤
                       # ⎣ 3  4 ⎦
m '(1,2; 3,4) + eye(2)'
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

```zsh
m 'A = (1,2,3; 4,5,6; 7,8,9)'
m 'A[1, 2]'          # 6          — element
m 'A[0, ..]'         # [1, 2, 3]  — first row
m 'A[.., 0]'         # [1, 4, 7]  — first column
m 'A[1.., 0]'        # [4, 7]     — rows 1 to end, col 0
m 'A[..1, 1..2]'     # 2×2 submatrix (rows 0-1, cols 1-2)
m 'A[1, 0..1]'       # [4, 5]     — row 1, cols 0–1
```

### Arithmetic

All standard operators broadcast element-wise. Scalar-tensor and tensor-tensor (same shape) both work:

```zsh
m 'eye(3) * 2'                      # 3×3 with 2s on diagonal
m 'ones(3,3) + eye(3)'              # all ones plus identity
m 'sin(zeros(2,3))'                 # element-wise sin (any 1-arg fn works)
m '(1,2; 3,4) @ (1,2; 3,4)'        # matrix multiply via @
```

### Shape operations

```zsh
m 'shape(ones(2, 3, 4))'            # [2, 3, 4]
m 'rows(A)'                         # 3
m 'cols(A)'                         # 3
m 'reshape(ones(6), 2, 3)'          # reshape to 2×3
m 'reshape(eye(4), 2, 2, 2, 2)'     # reshape to 4-D
m 'flatten(eye(2))'                 # [1, 0, 0, 1]  — 1-D tensor
m 'transpose(A)'                    # reverse all axes (classic 2-D transpose)
m 'transpose(T, 0, 2)'              # swap axes 0 and 2
m 'permute(T, 2, 0, 1)'            # reorder axes by permutation
m 'squeeze(zeros(1, 3, 1))'        # remove size-1 dims → shape [3]
m 'unsqueeze(zeros(3), 0)'          # insert size-1 dim → shape [1, 3]
```

### Concatenation

```zsh
m 'cat(0, eye(2), eye(2))'          # stack vertically (= vstack)
m 'cat(1, eye(2), eye(2))'          # stack horizontally (= hstack)
m 'cat(2, ones(2,3,4), ones(2,3,4))' # concat along axis 2 → shape (2,3,8)
m 'hstack(A, B)'                    # 2-D horizontal stack
m 'vstack(A, B)'                    # 2-D vertical stack
```

### Linear algebra

```zsh
m 'det((1,2; 3,4))'                 # -2
m 'inv((1,2; 3,4))'                 # inverse
m 'trace(eye(4))'                   # 4
m 'norm(ones(3, 4))'                # Frobenius norm = sqrt(12)
m 'solve((2,1; 1,3), (5,10))'       # solve Ax=b  →  1-D tensor
m 'row(A, 1)'                       # row 1 as 1-D tensor
m 'col(A, 0)'                       # column 0 as 1-D tensor
m 'A @ B'                           # matmul: 2D×2D, 2D×1D, 1D×2D, 1D×1D (dot)
m 'outer(ones(2), ones(3))'         # outer product → shape (2, 3)
```

### Eigenvalues, QR, and diagonalization

**Eigenvalues and eigenvectors** — all functions accept square real matrices:

```zsh
m 'eigvals((4,1; 1,3))'          # [4.618…, 2.381…]   — eigenvalues only
m 'eig((4,1; 1,3))'              # (eigenvalues, V)    — V columns are eigenvectors
m 'eig_top((4,1; 1,3))'          # (λ_max, v)          — largest eigenpair via power iteration
m 'eig_bot((4,1; 1,3))'          # (λ_min, v)          — smallest eigenpair via inverse iteration
```

`eig_top` and `eig_bot` are much faster than the full `eig` when only one eigenpair is needed.

Eigenvalue identities:

```zsh
m 'sum(eigvals(A))'    # equals trace(A)
m 'prod(eigvals(A))'   # equals det(A)
```

**QR decomposition** — works for m×n matrices with m ≥ n; returns full Q (m×m) and R (m×n):

```zsh
m 'qr((3,1; 1,2))'               # (Q, R)  with Q orthogonal, R upper-triangular
m 'qr((3,1; 1,2))[0]'            # Q only
m 'shape(qr((1,2; 3,4; 5,6))[0])'  # [3, 3]  — Q is always square
```

Verify the decomposition: `Q @ R` should recover the original matrix, and `transpose(Q) @ Q` should equal `eye(m)`.

**Diagonalization** — returns `(V, D, V_inv)` where `V @ D @ V_inv = A`:

```zsh
m 'diagonalize((4,1; 1,3))'      # (V, D, inv(V))
```

`D` is a full diagonal matrix (not a vector), so you can use it directly in expressions. This enables the **matrix exponential** without a dedicated builtin:

```zsh
# Matrix exponential: e^A = V @ diag(map(exp, eigvals(A))) @ inv(V)
m 'res = diagonalize(A); res[0] @ diag(map(exp, eigvals(A))) @ res[2]'
```

### Queries, statistics, and argmax/argmin

```zsh
m 'mean(ones(3, 4))'    # 1
m 'std(eye(3))'         # standard deviation over all elements
m 'var(eye(3))'         # variance
m 'norm(T)'             # Frobenius / Euclidean norm
m 'argmax((3,1,4,1,5))' # 4         — index of max in 1-D tensor (scalar)
m 'argmax((1,8; 8,1))'  # [0, 1]    — [row, col] index for 2-D tensor
m 'argmin(T)'           # [i, j, …] — n-D index tensor for n-D input
```

### Grid construction (`lingrid`)

`lingrid(start, end, counts, f)` evaluates `f` at a uniformly-spaced grid. `start`, `end`, `counts` can be scalars (1-D) or tuples of the same length (n-D).

```zsh
# 1-D: 5 points from 0 to 1
m 'lingrid(0, 1, 5, x -> x^2)'

# 2-D: 20×20 scalar field
m 'lingrid((-2,-2),(2,2),(20,20),(x,y) -> x^2 + y^2)'

# 2-D coordinate field — f returns a tuple, output is (20,20,2)
m 'G = lingrid((-1,-1),(1,1),(20,20),(x,y)->(x,y))'
m 'G[.., .., 0]'     # x-component grid (20×20)
m 'G[.., .., 1]'     # y-component grid (20×20)

# f can return a tensor too — output shape = grid_shape ++ value_shape
m 'shape(lingrid((0,0),(1,1),(3,3),(x,y)->eye(2)))'  # [3, 3, 2, 2]
```

---

## Complex numbers

`i` is the imaginary unit (√−1). Write complex literals as `3 + 2i`, `2i`, `-i`, etc.
All arithmetic operators work on complex numbers.

```zsh
m 'i^2'                      # -1
m '(1 + i) * (1 - i)'        # 2
m 'exp(i * pi)'               # -1   (Euler's formula)
m 'sqrt(-1)'                  # i
m 'ln(-1)'                    # 3.141592653589793i  (= πi)
m 'abs(3 + 4i)'               # 5    (modulus)
m 'arg(i)'                    # 1.5707963267948966  (= π/2)
m 'conj(3 + 4i)'              # 3 - 4i
m 're(3 + 4i), im(3 + 4i)'   # 3  4
```

---

## Built-in functions

**Trig:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `sinh`, `cosh`, `tanh`, `sec`, `csc`, `cot`

**Algebra:** `sqrt`, `cbrt`, `abs`, `sign`, `step`, `floor`, `ceil`, `round`, `round(x,n)`, `trunc`, `frac`, `exp`, `ln`, `log`/`log10`, `log(x,base)`, `log2`, `pow(x,y)`, `min`, `max`, `hypot(a,b)`

**Angle:** `deg(x)` (radians→degrees), `rad(x)` (degrees→radians)

**Complex:** `re(z)`, `im(z)`, `abs(z)`, `arg(z)`, `conj(z)`

**Number theory:** `gcd(a,b)`, `lcm(a,b)`, `fact(n)` / `n!`, `delta(x)`

**Special:** `sinc`, `sech`, `csch`, `erf`, `erfc`, `j0`, `j1`, `jinc`, `gaussian(x,mu,sigma)`

**Statistics** (1-D tensors or matrices): `mean`, `median`, `mode`, `std`, `var`, `min`, `max`, `sum`, `prod`

**Array ops:** `len`, `sort`, `zip(a,b)`, `dot(a,b)`, `append(t,x)`, `concat(a,b)`, `flatten(t)`, `argmin(t)`, `argmax(t)`, `linspace(a,b,n)`, `range(a,b)`

**Tensor constructors:** `zeros(n1,n2,…)`, `ones(n1,n2,…)`, `eye(n)`, `diag(t)`, `matrix(f,r,c)`, `tensor(f,n1,n2,…)`, `rand()`, `rand(n1,n2,…)`

**Tensor shape:** `shape(T)`, `rows(T)`, `cols(T)`, `reshape(T,n1,n2,…)`, `flatten(T)`, `squeeze(T)`, `unsqueeze(T,dim)`

**Tensor reorder:** `transpose(T)`, `transpose(T,a,b)`, `permute(T,p0,p1,…)`

**Tensor combine:** `cat(axis,T1,T2,…)`, `hstack(A,B)`, `vstack(A,B)`, `tomat(t,r,c)`, `outer(A,B)`

**Tensor reduce:** `sum(T)`, `prod(T)`, `sum(T,axis)`, `prod(T,axis)`, `norm(T)`, `trace(T)`, `mean(T)`, `std(T)`, `var(T)`

**Linear algebra:** `matmul(A,B)` / `A @ B`, `det(A)`, `inv(A)`, `solve(A,b)`, `row(T,i)`, `col(T,j)`, `eig(A)` → `(eigenvalues, V)`, `eigvals(A)`, `eig_top(A)` → `(λ, v)`, `eig_bot(A)` → `(λ, v)`, `qr(A)` → `(Q, R)`, `diagonalize(A)` → `(V, D, V⁻¹)`

**Grid:** `lingrid(start,end,counts,f)` — n-D uniform grid; `f` may return a scalar, tuple, or tensor

**Comparisons (function form):** `lt`, `leq`, `gt`, `geq`, `eq`, `neq` — 2-arg comparison functions returning `0`/`1`; useful with `map`/`filter`/`partial`

**Higher-order:** `map(f,t)`, `filter(f,t)`, `reduce(f,t)`, `compose(f,g)`, `partial(f,a)`

**Spectral:** `fft(v)`, `ifft(v)` — 1-D FFT/IFFT on a 1-D tensor; `fftn(T)`, `ifftn(T)` — n-D FFT on tensors

**Random:** `rand()`, `rand(n1,n2,…)` — scalar or shaped tensor

**Bitwise** (operate on 64-bit integers):
`and`, `or`, `xor`, `nand`, `nor`, `xnor`, `not`, `shl(x,n)`, `shr(x,n)`

```zsh
m 'shl(1, 8)'       # 256
m 'and(12, 10)'     # 8
m 'not(0)'          # -1  (bitwise NOT, two's complement)
m 'delta(0)'        # 1   (1 if x == 0, else 0)
m 'step(-1)'        # 0   (Heaviside: 0 / 0.5 / 1)
```

---

## Multiple outputs

```zsh
m 'x=5 : x, x^2, x^3'       # 5  25  125
m 'sin(pi/6), cos(pi/3)'     # 0.5  0.5
```

---

## REPL

Run `m` with no arguments to enter the interactive REPL. It supports syntax highlighting, tab completion, and history hinting.

```
> x = 3
> f(n) = n^2
> f(x)
result = 9
> result + 1
result = 10
```

**REPL commands:**

| Command | Effect |
|---------|--------|
| `!help` | show syntax reference |
| `!defs` | list all user-defined names |
| `!include <file>` | load a `.math` file into the current session |
| `!clear` | clear all user definitions |
| `!version` | show version |
| `!savetensor <var> <file>` | save tensor to binary `.mlt` file (see [Tensor I/O](#tensor-io)) |
| `!loadtensor <var> <file>` | load tensor from `.mlt` file |
| `!savehdf5 <var> <file> …` | save tensor to HDF5 (requires `--features hdf5`) |
| `!loadhdf5 <var> <file> …` | load tensor from HDF5 |
| `q` / `exit` | quit |

---

## Tensor I/O

Two serialization formats are available for saving and restoring tensors between sessions or exchanging data with other tools. Both handle real and complex tensors transparently.

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

**Build requirement:** pass `--features hdf5` at build time (requires `libhdf5` to be installed; tested against HDF5 2.x):

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
| `--list` | Print the dataset tree of the file instead of loading anything. Shows shapes and types recursively. The `<var>` argument is still required but ignored. |

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

And from the REPL:

```
> C = fft([1, 1, 0, 0])
> !savehdf5 C data.h5
saved C (4 elements, complex) → data.h5:/C
> !loadhdf5 D data.h5 /C
loaded D (4 elements, complex) ← data.h5:/C
> re(D[0]), im(D[1])
result = 2  -1
```

---

## CLI file loading

Use `-f` to load a `.math` file before evaluating an expression:

```zsh
m -f heat.math 'solver(10)'           # load heat.math, then evaluate
m -f advanced.math 'solveCubic(1,0,-3,-2)'
```

---

## Init file and `.math` files

Definitions in `~/.mathlangrc` are loaded automatically when the REPL starts.
Set `MATHLANG_INIT=/path/to/file.math` to use a different file.

`.math` files are plain text, one definition per line. Lines starting with `#` are comments.

### `advanced.math`

An included library (`advanced.math`) provides combinatorics, polynomial solvers, and more:

```
ncr(n,r)              — binomial coefficient
gamma(z)              — Stirling approximation of Γ(z)
quadratic(a,b,c)      — roots of ax²+bx+c=0  (returns tuple)
solveCubic(a,b,c,d)   — 3 real roots of ax³+… (trig method, disc > 0)
cubicRoot(a,b,c,d)    — 1 real root  of ax³+… (Cardano, disc < 0)
solveQuartic(a,b,c,d,e) — 4 real roots of ax⁴+… (Ferrari's method)
```

Load it with `!include advanced.math` in the REPL (or `m -f advanced.math 'expr'` from CLI).

### `physics.math`

Physical constants in SI units (`c`, `g`, `G`, `k_B`, `N_A`, `R`, `h`, `hbar`, etc.) plus `oscillator`, `oscillatorMBK`, `heatKernel`, `heatSpectral` (1-D spectral heat solver).

### `heat.math`

2-D heat equation solver with spatially-varying diffusivity.

```zsh
m -f heat.math 'solver(0)'    # initial condition: 20×20 grid, cold disk in hot air
m -f heat.math 'solver(10)'   # temperature after t=10 (50 steps)
m -f heat.math 'solver(20)'   # mostly equilibrated (~100 ms)
```

`heatSolver(T0, alpha, dx, dt)` takes initial temperature grid, diffusivity field, and timestep; returns a function `t -> T(t)`. Uses energy-conserving divergence-form FD with Neumann BCs.

### `conversions.math`

Unit conversion factors and functions for physics, chemistry, and engineering. Load with `!include conversions.math`.

```zsh
m '!include conversions.math'
> 1 * eV_to_J          # 1.602176634e-19
> C_to_K(100)          # 373.15
> 1 * atm_to_Pa        # 101325
```
