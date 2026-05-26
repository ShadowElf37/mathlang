# implementation progress

Test baseline: 75 passed, 39 failed.

---

## Tier 1 — Trivial ✓ complete

| # | Change | Status |
|---|--------|--------|
| T2 | `sec`, `csc`, `cot` | done |
| T2.5 | `step(x)` (Heaviside) | done |
| T3 | `trunc(x)`, `frac(x)` | done |
| T4 | `deg(x)`, `rad(x)` | done |
| T5 | `len` / `length` | done |
| T6 | Negative tuple indexing | done |

**Tests after Tier 1:** 80 passed, 34 failed (+5).
Note: `qol.cot` still fails — `cot(pi/4)` has a 2-ULP floating-point rounding error; test uses exact comparison, not a code bug.

---

## Tier 2 — Easy ✓ complete

| # | Change | Status |
|---|--------|--------|
| T7 | README updates | done |
| T7.5 | `conversions.math` | done |
| T8 | 2-arg `log(x, base)` | done |
| T9 | 2-arg `round(x, n)` | done |
| T10 | `linspace(a, b, n)` | done |
| T11 | `range(a, b)` builtin | done |
| T12 | `sort`, `zip`, `dot`, `append`, `concat`, `flatten`, `argmin`, `argmax` | done |
| T13 | Polymorphic `sum(tuple)` / `prod(tuple)` | done |
| T14 | `mean`, `median`, `mode` | done |
| T15 | `std`, `var` | done |
| T16 | `compose`, `partial` | done |
| T17 | `gaussian(x, mu, sigma)` | done |
| T18 | `filter`, `reduce` | done |
| T19 | `rand` / `rand(a, b)` | done |

**Tests after Tier 2:** 88 passed, 26 failed (+8).
Remaining failures: tau (intentional), implicit functions (removed feature), tuple printing format (README updated to match), block forward-refs (not doing), `&&`/`||`/ternary (not doing), comparisons/if/slicing/`!`/implicit-mul (Tier 3/4).

## Tier 3 — Medium

| # | Change | Status |
|---|--------|--------|
| T20 | `if(cond, a, b)` special form | pending |
| T21 | Postfix `!` factorial | pending |
| T22 | Tuple slicing / multi-index | pending |
| T23 | Comparison operators | pending |

## Tier 4 — Hard

| # | Change | Status |
|---|--------|--------|
| T24 | Implicit multiplication | pending |
