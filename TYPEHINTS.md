# Mathlang Gradual Type Hints

## Overview

Mathlang supports optional, gradual type hints on function parameters and return
types. Hints are enforced at call time (not statically). Unannoted parameters
accept any value (equivalent to `any`).

## Type Vocabulary

| Keyword         | Meaning                                                         |
|-----------------|----------------------------------------------------------------|
| `any`           | No constraint (default when no hint is written)                |
| `num`           | Any scalar — real (`Num`) or complex (`Complex`)               |
| `real`          | Real scalar; complex coerced if `|im| < f64::EPSILON`          |
| `complex`       | Complex scalar; reals accepted (widening)                      |
| `int`           | Integer real scalar (`frac == 0`); complex coerced if `|im|<eps` and `frac==0` |
| `nat`           | Non-negative integer real scalar                               |
| `tensor`        | Any tensor (real or complex), any rank                         |
| `real tensor`   | Real-only tensor; complex tensor coerced if all `|im| < eps`   |
| `complex tensor`| Complex tensor; real tensors accepted (widening)               |
| `fn`            | Any callable (user function or builtin)                        |
| `cell`          | Any cell (contents unchecked)                                  |
| `tuple`         | Any tuple                                                      |

## Syntax

### Named function with parameter hints

```
f(x: real, n: nat) -> tensor = x * linspace(0, 1, n)
```

### Lambda with parameter hints (no return type on lambdas)

```
g := (x: complex, T: real tensor) -> x * T
```

The `->` token is already the lambda body separator on anonymous lambdas; a
second `->` for return type would be ambiguous. Return type annotations are
therefore only supported on named `Def::Func` definitions.

### Unannoted (unchanged behaviour)

```
h := x -> x^2
```

### Return type only

```
positive(x) -> nat = abs(x)
```

### All hints optional

```
apply(f: fn, x) = f(x)
```

## Subtyping and Coercion Rules

### Scalar hierarchy

```
nat ≤ int ≤ real ≤ num
               complex ≤ num
               real ≤ complex     (widening, free)
```

- **`real ≤ num`** and **`complex ≤ num`**: both accepted by `num`.
- **`real ≤ complex`**: a real scalar is always accepted where `complex` is
  expected (widening, no cost).
- **`complex → real`**: coerces if `|im| < f64::EPSILON`; otherwise a type
  error is raised. This is the only "narrowing" coercion.
- **`nat ≤ int ≤ real`**: a nat is always a valid int or real (widening).
  A fractional real is rejected by `int` and `nat`.

### Tensor hierarchy

```
real tensor ≤ complex tensor ≤ tensor
```

- **`real tensor ≤ complex tensor`**: widening, free.
- **`complex tensor → real tensor`**: coerces if all `|im| < f64::EPSILON`.
- Scalars are **not** subtypes of tensors. Passing a scalar where a `tensor`
  is expected is always an error.

## Enforcement

Hints are enforced in `coerce_to_hint` (eval.rs), which is called:

1. **Per parameter** before the function body runs.
2. **On the return value** after the body completes (if a return type is given).

No coercion is performed inside the body — hints apply only at the call
boundary.

## `!type` Command

The `!type <name>` REPL command shows the type signature of any in-scope
function or builtin:

```
> f(x: real) -> nat = floor(x)
> !type f
f(x: real) -> nat
> !type sqrt
sqrt(x: num) -> num
```

## Future Work

| ID     | Feature                                                                      |
|--------|------------------------------------------------------------------------------|
| FN-01  | `fn(real) -> real` typed function signatures for higher-order parameters     |
| FN-02  | `cell(nat)` inner-type hints on cells                                        |
| FN-03  | `(real, tensor)` structural tuple-element hints                              |
| FN-04  | Tensor rank/shape hints: `tensor(_, _)` for rank-2, `tensor(3,3)` for exact |
| FN-05  | Refinements beyond int/nat: `pos`, `neg`, `unit`, `prob`                    |
| FN-06  | Auto-check builtins from signature table (replace ad-hoc `.num()` calls)    |
| FN-07  | `--strict` mode: `complex → real` is always an error (no ε scan)            |
| FN-08  | User-defined type aliases (`angle := real`)                                  |
