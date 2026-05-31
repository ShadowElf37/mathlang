# PDE / field-computation ergonomics — proposed builtins

> **Status (v0.22.0): implemented.** §1 (grad/div/curl/lap with required `dx`) and
> §3 (poisson/invlap/specgrad) shipped in the `ops` namespace; §4 (rk4,
> odeint, cfl) in the `solver` namespace; §5's non-finite guard is in iterate/scan.
> Deferred: the §2 grid-field type, and the IMEX/integrating-factor stepper (it
> can't be made generic without knowing the stiff linear operator). See README
> "Differential operators and solvers" and `examples/fluid2D.math`.

Design notes motivated by writing `examples/fluid2D.math`. That simulation had two
substantive bugs and a stability tuning slog, **all** of which trace to features
the language doesn't have. This doc lists candidate additions, ordered by how
directly each would have prevented the bugs we actually hit.

## Background: what went wrong, and why

| Bug | Root cause | Feature that prevents it |
|-----|-----------|--------------------------|
| No fluid motion (advection ~34× too weak) | Hand-rolled `d0`/`d1`/`lap` from `roll` with no grid spacing — `dx = 2π/N` lived only in a comment | Grid-aware differential operators (§1), field type (§2) |
| Velocity field reversed | Hand-derived spectral inverse-Laplacian sign (`ψ̂ = ω̂/k²`, not `−ω̂/k²`) | Spectral helpers (§3) |
| Blow-up / NaN | Explicit Euler + advection CFL violated + no large-scale dissipation | Integrators (§4), diagnostics (§5) |
| Slow to debug | `iterate` silently returned `NaN`; found the blow-up step by manual sweep | NaN guard + CFL helper (§5) |

What exists today (for reference): `deriv` (scalar `f(x)` only), `integral` (1-D
quadrature), `roll`, `solve`, n-D `fft`/`ifft`, `iterate`, `scan`, `lingrid`.
There are **no** gridded differential operators, ODE/PDE integrators, or spectral
helpers beyond the raw FFT.

---

## 1. Grid-aware differential operators — highest leverage

Every bug except the Poisson sign came from hand-rolling stencils out of `roll`
while the physical grid spacing `dx` lived only in your head.

```
grad(T, dx)            # central difference along every axis → adds a trailing vector axis
grad(T, dx, axis)      # one axis
div(V, dx)             # divergence of a vector field
curl(V, dx)            # 2-D scalar curl / 3-D vector
lap(T, dx)             # Laplacian, correctly ÷dx²
```

**Key design decision: spacing is a required argument, never defaulted to 1.**
NumPy's `np.gradient` defaults spacing to 1 — the same trap; `lap(T)` would have
reproduced the exact bug we just fixed. Forcing `dx` into the signature makes the
physical unit explicit at every call site.

Cost: ~30 lines each on top of `roll`. Between them they collapse the entire
operator block in `fluid2D.math` to three calls.

**Boundary conditions.** `roll`-based stencils are periodic; the `lap2d` idiom in
the docs is Neumann (clamped). A builtin needs a BC argument (default periodic to
match `roll`):

```
lap(T, dx, periodic)   # or: neumann / dirichlet
```

---

## 2. A "grid field" type that carries spacing + BCs — root-cause fix

Operators-with-`dx` still let you forget `dx`. The deeper fix is a value that
*remembers* its geometry:

```
F = field(omega0, (0,0), (2pi,2pi), periodic)   # data + extent + BC
grad(F)        # no dx needed — F knows it, and knows it's periodic
lap(F)
```

This also fixes a real ambiguity the current code papers over: **nothing records
whether a tensor wants periodic or Neumann BCs**, so it's on the author to pick a
matching stencil every time. A field type makes "periodic box [0,2π)²" a property
of the data instead of a comment.

Bigger lift than §1 (new type, display, broadcasting interactions), but it's the
thing that makes this *class* of bug structurally impossible rather than merely
easier to avoid.

---

## 3. Spectral helpers — would have killed the sign error

The `ψ̂ = ω̂/k²` sign flip came from hand-deriving a spectral inverse-Laplacian,
including the zero-mode `if(kk==0,1,kk)` dance.

```
poisson(rhs, dx)       # solve ∇²u = rhs spectrally (periodic), zero-mean
invlap(T, dx)          # ∇⁻²
specgrad(T, dx)        # spectral derivative via i·k — no finite-difference error
```

Bonus: a spectral `specgrad` for advection is *more accurate* than central
differences and sidesteps the grid-scale enstrophy pile-up that forced `nu` up to
0.005. These are thin wrappers over the existing n-D `fft`/`ifft` that bake in the
`k`-grid and zero-mode handling.

---

## 4. Integrators — partial help, be clear-eyed about it

```
rk4(f, y0, t0, t1, n)      # f is dy/dt; fixed-step RK4
odeint(f, y0, ts)          # adaptive, returns trajectory
imex(f_stiff, f_nonstiff)  # implicit diffusion + explicit advection
```

What these would and wouldn't have fixed here:

- **RK4** improves *accuracy* and modestly widens the stable region — but the
  blow-up was an **advection CFL** limit (`dt·u/dx < 1`). RK4 wouldn't have saved
  `dt=0.02`; you'd still drop `dt`. It mainly buys nicer trajectories per step.
- The real stability win for stiff PDEs is **IMEX / integrating-factor** stepping
  (linear diffusion+drag treated implicitly/exactly, advection explicitly). That
  would let you keep low `nu` without grid-scale instability — but it's hard to
  make generic, because it must know *which* part of the RHS is the stiff linear
  operator. Tends to end up spectral-specific rather than black-box.

`iterate`/`scan` already provide the loop; what's missing is a *good scheme*
inside it. RK4 is the cheap, high-value piece; IMEX is powerful but specialized.

---

## 5. Diagnostics — cheapest, highest ROI per line

- **NaN/Inf guard in `iterate`/`scan`**: abort with "non-finite at step N" instead
  of silently returning `NaN`. The stabilization sweep was really just hunting for
  where things blew up.
- **`cfl(V, dx, dt)` helper**: return the Courant number so "is `dt` safe?" is one
  call, not a parameter sweep.
- **Document that finite-difference operators are dimensionless-by-index** — even
  a doc note would have flagged the original mistake.

---

## Recommended priority

1. **§1** `grad`/`div`/`curl`/`lap` with mandatory `dx` — cheap, idiomatic for a
   tensor-first language, prevents the advection bug.
2. **§3** spectral `poisson`/`specgrad` — cheap (wraps existing FFT), prevents the
   sign bug, improves accuracy.
3. **§5** NaN guard — almost free, add regardless.
4. **§4** `rk4` — nice-to-have accuracy/stability.
5. **§2** field type — elegant long-term answer, real design commitment.

§1 + §3 together would have prevented *both* substantive bugs in `fluid2D.math`.
A natural first PR: `grad`/`lap`-with-`dx` + spectral `poisson` (with tests,
`!help`, README per CLAUDE.md), then rewrite `fluid2D.math` on top of them as the
worked example.
