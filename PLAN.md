# Tensor Support Plan

## Design decisions

- New `Val::Tensor { data: Vec<f64>, shape: Vec<usize> }` — row-major, real-only, arbitrary rank
  - 1D tensor = vector;  2D tensor = matrix;  3D+ supported uniformly
- `Val::Tuple` stays unchanged (backward compat; holds heterogeneous `Val` incl. functions/complex)
- `matrix(r, c, f)` constructor: `f(i,j)` called for each cell (0-indexed) → 2D tensor
- `(1,2; 3,4)` literal syntax (Tier 2): `;` as row separator inside `()`
- `M[i,j]` indexing: Val::Tensor dispatch → scalar; `M[i]` on 2D tensor → 1D row tensor
- Broadcasting: Tensor op Scalar → elementwise; Tensor op Tensor (same shape) → elementwise
- Unary math builtins (`sin`, `exp`, etc.) broadcast elementwise over Tensor via `broadcast1`
- `map(f, T)` extended to work elementwise on Tensor, returning Tensor same shape
- `lingrid((x0,y0),(x1,y1),(nx,ny),f)` → 2D Tensor where `T[i,j]=f(x_i,y_j)` (Tier 2)
- `matmul(A,B)` for 2D matrix multiply; `@` operator alias (Tier 3)
- No einstein notation, no animations
- REPL: 2D tensors pretty-printed as column-aligned rows (Tier 2 box-chars; Tier 1 simple brackets)

---

## Tier 1 — Core type, constructor, indexing, broadcasting, basic ops

### Val extension (`src/eval.rs`)

```rust
Tensor { data: Vec<f64>, shape: Vec<usize> }
```

Flat index (row-major): `flat = i0*s0 + i1*s1 + …` where strides `sk = shape[k+1]*…*shape[ndim-1]`.

### New builtins

| Name | Signature | Notes |
|------|-----------|-------|
| `matrix` | `matrix(r, c, f)` | `f(i,j)` for each cell; returns 2D tensor |
| `zeros` | `zeros(d0, d1, …)` | all-zero tensor of given shape |
| `ones` | `ones(d0, d1, …)` | all-one tensor |
| `eye` | `eye(n)` | identity n×n |
| `diag` | `diag(t)` | diagonal matrix from tuple |
| `shape` | `shape(T)` → Tuple | e.g. `(3, 4)` for 3×4 |
| `rows` | `rows(T)` → Num | first dimension (2D+) |
| `cols` | `cols(T)` → Num | second dimension (2D+) |
| `transpose` | `transpose(T)` | 2D only |
| `trace` | `trace(T)` | sum of diagonal |
| `norm` | `norm(T)` | Frobenius norm; also works on Tuple (L2) |
| `row` | `row(T, i)` → Tuple | row i of 2D tensor |
| `col` | `col(T, j)` → Tuple | col j of 2D tensor |
| `matmul` | `matmul(A, B)` | 2D matrix product |

### Extensions to existing builtins

- `len(T)` → first dimension (like numpy `len`)
- `flatten(T)` → Tuple of all elements row-major
- `sum(T)` / `prod(T)` (1-arg) → sum/product over all elements
- `map(f, T)` → apply f elementwise, return Tensor same shape
- `dot(T1, T2)` → dot product of two 1D tensors

### Indexing

```
T[i]       on 1D tensor → scalar
T[i]       on 2D tensor → 1D row tensor
T[i, j]    on 2D tensor → scalar
T[i, j, k] on 3D tensor → scalar
```

Parser already produces `Expr::Tuple(indices)` for `[i,j,k]` — dispatch on `Val::Tensor` in `eval()`.

### Broadcasting in `binop_tensor`

```
Tensor op Scalar    → elementwise (errors if scalar is Complex/Tuple)
Scalar op Tensor    → elementwise
Tensor op Tensor    → elementwise (same shape required; error otherwise)
```

Unary extension: `broadcast1` already handles general dispatch; add `Val::Tensor` arm.

---

## Tier 2 — Literal syntax, lingrid, map extension, det/inv, stack ops, REPL display

### Literal syntax

`(1, 2; 3, 4)` — parser detects `;` inside `()` and builds Tensor.
Each row must have equal element count.

```
(a, b; c, d)        →  2×2 tensor
(1, 2, 3; 4, 5, 6)  →  2×3 tensor
```

### lingrid

```
lingrid((x0,y0), (x1,y1), (nx,ny), f) → 2D Tensor  (nx × ny)
```

`x_i = x0 + i*(x1-x0)/(nx-1)`, `y_j = y0 + j*(y1-y0)/(ny-1)`.
`T[i,j] = f(x_i, y_j)`.

### New builtins

| Name | Signature | Notes |
|------|-----------|-------|
| `det` | `det(T)` | determinant (LU decomp, pure Rust) |
| `inv` | `inv(T)` | inverse (Gauss-Jordan, pure Rust) |
| `hstack` | `hstack(A, B)` | horizontal concat (same row count) |
| `vstack` | `vstack(A, B)` | vertical concat (same col count) |
| `tomat` | `tomat(t, r, c)` → Tensor | reshape Tuple into 2D tensor |

### REPL display (Tier 2)

Column-aligned matrix with box chars:
```
⎡  1  2  3 ⎤
⎢  4  5  6 ⎥
⎣  7  8  9 ⎦
```
(Tier 1 uses simple `[1  2  3]\n[4  5  6]` format.)

---

## Tier 3 — @ operator, solve, slicing

### `@` operator

Parser: `@` token → `Op::MatMul`; handled at `mul()` level.
Desugars to `matmul(A, B)`.

### solve

```
solve(A, b)  →  Tuple   — solves Ax=b, Gaussian elimination w/ partial pivoting
```

### Row/col slicing

```
T[i, 0..3]    — row i, cols 0–2
T[0..2, j]    — rows 0–1, col j
T[0..2, 0..3] — submatrix → 2D tensor
```

---

## Implementation order

1. `Val::Tensor` + `fmt_val` + `broadcast1` extension  ← **Tier 1 start**
2. `binop_tensor` + `Expr::BinOp` + `Expr::Neg` dispatch
3. `Expr::Index` for tensors
4. `matrix(r,c,f)` + `zeros/ones/eye/diag`
5. `shape/rows/cols/transpose/trace/norm/row/col/matmul`
6. Extend `len/flatten/sum/prod/map/dot` for tensors  ← **Tier 1 done**
7. `(1,2;3,4)` literal parsing
8. `lingrid`
9. `det`, `inv` (pure Rust LU/Gauss-Jordan)
10. `hstack`, `vstack`, `tomat`
11. REPL pretty-print + box chars  ← **Tier 2 done**
12. `@` operator
13. `solve`
14. 2D slice indexing  ← **Tier 3 done**

No new crate dependencies through Tier 2. Tier 3 stays dep-free.
