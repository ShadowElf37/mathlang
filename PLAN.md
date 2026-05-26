# mathlang improvement plan

Changes ordered easiest → hardest. Each entry names the affected files and sketches the exact code change needed.

---

## Tier 1 — Trivial (1–5 lines, single file)

### T2. Add `sec`, `csc`, `cot` — `eval.rs`, `repl.rs`

Three `b1!` branches. All broadcast over tuples via `b1!`.

```rust
"sec" => b1!(|v| {
    let x = v.num("sec")?;
    let c = x.cos();
    if c == 0.0 { return Err("sec: undefined (cos is zero)".into()); }
    Ok(Val::Num(1.0 / c))
}),
"csc" => b1!(|v| {
    let x = v.num("csc")?;
    let s = x.sin();
    if s == 0.0 { return Err("csc: undefined (sin is zero)".into()); }
    Ok(Val::Num(1.0 / s))
}),
"cot" => b1!(|v| {
    let x = v.num("cot")?;
    let s = x.sin();
    if s == 0.0 { return Err("cot: undefined (sin is zero)".into()); }
    Ok(Val::Num(x.cos() / s))
}),
```

Add `"sec"`, `"csc"`, `"cot"` to `Env::new()` builtin list, `is_protected()`, and `repl::BUILTIN_FNS`.

---

### T2.5. Add `step(x)` (Heaviside) — `eval.rs`, `repl.rs`

`step(x)` is the Heaviside step function: `0` for x < 0, `0.5` at x = 0, `1` for x > 0. Like `sign` but maps negatives to `0` instead of `-1`.

```rust
"step" => b1!(|v| Ok(Val::Num(match v.num("step")? {
    x if x < 0.0 => 0.0,
    x if x > 0.0 => 1.0,
    _             => 0.5,
}))),
```

Add to `Env::new()`, `is_protected()`, `BUILTIN_FNS`.

---

### T3. Add `trunc(x)`, `frac(x)` — `eval.rs`, `repl.rs`

```rust
"trunc" => f1!(trunc),   // f64::trunc truncates toward zero
"frac"  => b1!(|v| {
    let x = v.num("frac")?;
    Ok(Val::Num(x - x.trunc()))
}),
```

Add to `Env::new()`, `is_protected()`, `BUILTIN_FNS`.

---

### T4. Add `deg(x)`, `rad(x)` — `eval.rs`, `repl.rs`

```rust
"deg" => b1!(|v| Ok(Val::Num(v.num("deg")? * (180.0 / std::f64::consts::PI)))),
"rad" => b1!(|v| Ok(Val::Num(v.num("rad")? * (std::f64::consts::PI / 180.0)))),
```

Add to `Env::new()`, `is_protected()`, `BUILTIN_FNS`.

---

### T5. Add `len` / `length` — `eval.rs`, `repl.rs`

```rust
"len" | "length" => {
    arity(name, 1, vals.len())?;
    match vals.into_iter().next().unwrap() {
        Val::Tuple(items) => Ok(Val::Num(items.len() as f64)),
        _ => Err(format!("{name}: argument must be a tuple")),
    }
}
```

Add both names to `Env::new()`, `is_protected()`, `BUILTIN_FNS`.

---

### T6. Negative tuple indexing — `eval.rs`

In the `Expr::Index` eval arm, change:
```rust
let i = eval(idx, env)?.num("index")? as usize;
```
to:
```rust
let raw = eval(idx, env)?.num("index")? as i64;
// (tuple length known after the match below)
```
And inside the `Val::Tuple(items)` match:
```rust
Val::Tuple(items) => {
    let len = items.len() as i64;
    let i = if raw < 0 { (len + raw).max(0) as usize } else { raw as usize };
    items.into_iter().nth(i).ok_or_else(|| format!("index {raw} out of range"))
}
```
The multi-index and slice paths (added in T15) also use this negative-aware lookup.

---

## Tier 2 — Easy (extend existing mechanisms, 10–30 lines)

### T7. README updates — `README.md`, `advanced.math`

Changes:
- **Remove** the entire "Implicit functions" section. Free-variable capture was removed from the evaluator; this section advertises behaviour that no longer exists.
- **Constants**: remove `tau` from the constants list (`pi`, `e`, `phi`, `inf`, `i`).
- **Tuple examples**: update the arithmetic examples to reflect the actual output format — tuples print with parentheses and commas, e.g. `(2, 4, 6)`, not space-separated.
- **Built-in functions table**: add new builtins: `sec/csc/cot`, `deg/rad`, `len`, `trunc/frac`, `log(x,base)`, `round(x,n)`, `linspace(a,b,n)`, `range(a,b)`, `sort`, `zip`, `dot`, `append/concat/flatten`, `argmin/argmax`, `mean/median/mode`, `std/var`, `filter/reduce`, `compose/partial`, `gaussian(x,mu,sigma)`, `rand/rand(a,b)`.
- **New `if` section**: document `if(cond, a, b)` with the lambda-composition note.
- **New Comparisons section**: document `<`, `>`, `<=`, `>=`, `==`, `!=` returning `0` or `1`.
- **Tuple indexing section**: document `tuple[a..b]` (inclusive slice) and `tuple[i,j,k]` (multi-index).
- **`advanced.math`**: remove `gaussian_pdf` and `gaussian_cdf` lines (replaced by `gaussian` builtin; `gaussian_cdf` needs rethinking — the CDF depends on `erf` and has no single clean closed form as a "gaussian" call, so remove both lines and let users build `gaussian_cdf` themselves with `erf` if needed).
- `physics.math`, `conversions.math` documentation
---

### T7.5. `conversions.math` library file — `conversions.math`

A new `.math` file loadable via `!import conversions.math`. Variables are named `unit1_to_unit2` and hold the multiplicative factor (so `x * eV_to_J` converts x eV to joules). Non-linear conversions (temperature scales) are defined as functions.

**Excluded on purpose:** trivial SI prefix ratios (`ms_to_s = 0.001`, `km_to_m = 1000`, etc.) — these are just powers of ten and add noise.

Planned contents (non-exhaustive — include everything useful for physics, chemistry, engineering):

```
# ── Energy ───────────────────────────────────────────────────────────────
eV_to_J        = 1.602176634e-19
J_to_eV        = 1 / eV_to_J
cal_to_J       = 4.184
kcal_to_J      = 4184
BTU_to_J       = 1055.06
kWh_to_J       = 3.6e6
erg_to_J       = 1e-7
hartree_to_eV  = 27.211386245988

# ── Pressure ─────────────────────────────────────────────────────────────
atm_to_Pa      = 101325
bar_to_Pa      = 1e5
mmHg_to_Pa     = 133.322387415     # torr = mmHg
psi_to_Pa      = 6894.757
torr_to_Pa     = 133.322387415

# ── Length ───────────────────────────────────────────────────────────────
in_to_m        = 0.0254
ft_to_m        = 0.3048
mi_to_m        = 1609.344
yd_to_m        = 0.9144
AU_to_m        = 1.495978707e11
ly_to_m        = 9.4607304725808e15
pc_to_m        = 3.085677581e16
angstrom_to_m  = 1e-10
bohr_to_m      = 5.29177210903e-11

# ── Mass ─────────────────────────────────────────────────────────────────
lb_to_kg       = 0.45359237
oz_to_kg       = 0.028349523125
amu_to_kg      = 1.66053906660e-27
ton_to_kg      = 907.18474          # short ton (US)
tonne_to_kg    = 1000.0             # metric ton

# ── Time ─────────────────────────────────────────────────────────────────
min_to_s       = 60
hr_to_s        = 3600
day_to_s       = 86400
yr_to_s        = 31557600           # Julian year (365.25 days)

# ── Speed ─────────────────────────────────────────────────────────────────
mph_to_mps     = 0.44704
kph_to_mps     = 1 / 3.6
knot_to_mps    = 0.514444

# ── Temperature (functions — non-linear) ─────────────────────────────────
C_to_K(t)   = t + 273.15
K_to_C(t)   = t - 273.15
F_to_C(t)   = (t - 32) * 5/9
C_to_F(t)   = t * 9/5 + 32
F_to_K(t)   = C_to_K(F_to_C(t))

# ── Angle ─────────────────────────────────────────────────────────────────
# (deg/rad builtins handle the common case; include arcminute/arcsecond)
arcmin_to_rad  = pi / 10800
arcsec_to_rad  = pi / 648000
grad_to_rad    = pi / 200

# ── Area ──────────────────────────────────────────────────────────────────
acre_to_m2     = 4046.8564224
ha_to_m2       = 10000
in2_to_m2      = 6.4516e-4
ft2_to_m2      = 0.09290304

# ── Volume ───────────────────────────────────────────────────────────────
L_to_m3        = 1e-3
mL_to_m3       = 1e-6
gal_to_m3      = 3.785411784e-3     # US gallon
floz_to_m3     = 2.95735295625e-5   # US fluid ounce
ft3_to_m3      = 0.028316846592

# ── Force ─────────────────────────────────────────────────────────────────
lbf_to_N       = 4.44822162
dyn_to_N       = 1e-5
kgf_to_N       = 9.80665

# ── Power ────────────────────────────────────────────────────────────────
hp_to_W        = 745.69987158227    # mechanical horsepower

# ── Magnetic field ───────────────────────────────────────────────────────
T_to_G         = 1e4               # tesla to gauss
G_to_T         = 1e-4

# ── Charge / electrical ──────────────────────────────────────────────────
Ah_to_C        = 3600              # ampere-hour to coulomb

# ── Data / information ────────────────────────────────────────────────────
bit_to_byte    = 1/8
byte_to_bit    = 8
```

File is plain `.math` syntax; users load it with `!import conversions.math` in the REPL or set it as part of their init file.

---

### T8. Extend `log` to 2-arg `log(x, base)` — `eval.rs`

Currently `"log" | "log10" => f1!(log10)`. Change `log` to its own arm:

```rust
"log" => {
    match vals.len() {
        1 => broadcast1(vals.into_iter().next().unwrap(),
                |v| Ok(Val::Num(v.num("log")?.log10()))),
        2 => {
            let mut it = vals.into_iter();
            let x    = it.next().unwrap().num("log")?;
            let base = it.next().unwrap().num("log")?;
            if base <= 0.0 || base == 1.0 {
                return Err("log: base must be positive and ≠ 1".into());
            }
            Ok(Val::Num(x.ln() / base.ln()))
        }
        n => Err(format!("log expects 1 or 2 args, got {n}")),
    }
}
```

`log10` keeps its own `f1!(log10)` arm unchanged.

---

### T9. Extend `round` to 2-arg `round(x, n)` — `eval.rs`

```rust
"round" => {
    match vals.len() {
        1 => broadcast1(vals.into_iter().next().unwrap(),
                |v| Ok(Val::Num(v.num("round")?.round()))),
        2 => {
            let mut it = vals.into_iter();
            let x = it.next().unwrap().num("round")?;
            let n = it.next().unwrap().num("round")? as i32;
            let factor = 10f64.powi(n);
            Ok(Val::Num((x * factor).round() / factor))
        }
        n => Err(format!("round expects 1 or 2 args, got {n}")),
    }
}
```

---

### T10. Add `linspace(a, b, n)` — `eval.rs`, `repl.rs`

```rust
"linspace" => {
    arity("linspace", 3, vals.len())?;
    let mut it = vals.into_iter();
    let a = it.next().unwrap().num("linspace")?;
    let b = it.next().unwrap().num("linspace")?;
    let n = it.next().unwrap().num("linspace")? as usize;
    if n == 0 { return Err("linspace: n must be ≥ 1".into()); }
    if n == 1 { return Ok(Val::Tuple(vec![Val::Num(a)])); }
    let items = (0..n)
        .map(|i| Val::Num(a + (b - a) * i as f64 / (n - 1) as f64))
        .collect();
    Ok(Val::Tuple(items))
}
```

---

### T11. Add `range(a, b)` builtin — `eval.rs`, `repl.rs`

Exclusive upper bound (Python-style). Coexists with paren-syntax `(a..b)` which remains inclusive.

```rust
"range" => {
    arity("range", 2, vals.len())?;
    let mut it = vals.into_iter();
    let a = it.next().unwrap().num("range")? as i64;
    let b = it.next().unwrap().num("range")? as i64;
    let items = (a..b).map(|n| Val::Num(n as f64)).collect();
    Ok(Val::Tuple(items))
}
```

---

### T12. Add `sort`, `zip`, `dot`, `append`, `concat`, `flatten`, `argmin`, `argmax` — `eval.rs`, `repl.rs`

All straightforward tuple operations. Each registered in `Env::new()`, `is_protected()`, `BUILTIN_FNS`.

```rust
"sort" => {
    arity("sort", 1, vals.len())?;
    let mut items = match vals.into_iter().next().unwrap() {
        Val::Tuple(v) => v,
        _ => return Err("sort: argument must be a tuple".into()),
    };
    let mut nums: Vec<f64> = items.iter()
        .map(|v| v.clone().num("sort")).collect::<Result<_, _>>()?;
    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Val::Tuple(nums.into_iter().map(Val::Num).collect()))
}

"zip" => {
    arity("zip", 2, vals.len())?;
    let mut it = vals.into_iter();
    let a = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("zip: args must be tuples".into()) };
    let b = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("zip: args must be tuples".into()) };
    let pairs = a.into_iter().zip(b).map(|(x, y)| Val::Tuple(vec![x, y])).collect();
    Ok(Val::Tuple(pairs))
}

"dot" => {
    arity("dot", 2, vals.len())?;
    let mut it = vals.into_iter();
    let a = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("dot: args must be tuples".into()) };
    let b = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("dot: args must be tuples".into()) };
    if a.len() != b.len() { return Err(format!("dot: length mismatch ({} vs {})", a.len(), b.len())); }
    let s: Result<f64, _> = a.into_iter().zip(b)
        .map(|(x, y)| Ok(x.num("dot")? * y.num("dot")?))
        .try_fold(0.0, |acc, r: Result<f64, String>| Ok(acc + r?));
    Ok(Val::Num(s?))
}

"append" => {
    arity("append", 2, vals.len())?;
    let mut it = vals.into_iter();
    let mut items = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("append: first arg must be a tuple".into()) };
    items.push(it.next().unwrap());
    Ok(Val::Tuple(items))
}

"concat" => {
    arity("concat", 2, vals.len())?;
    let mut it = vals.into_iter();
    let mut a = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("concat: args must be tuples".into()) };
    let b    = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("concat: args must be tuples".into()) };
    a.extend(b);
    Ok(Val::Tuple(a))
}

"flatten" => {
    arity("flatten", 1, vals.len())?;
    let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("flatten: argument must be a tuple".into()) };
    let flat = items.into_iter().flat_map(|v| match v {
        Val::Tuple(inner) => inner,
        other => vec![other],
    }).collect();
    Ok(Val::Tuple(flat))
}

"argmin" => {
    arity("argmin", 1, vals.len())?;
    let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("argmin: argument must be a tuple".into()) };
    if items.is_empty() { return Err("argmin: empty tuple".into()); }
    let mut best_i = 0;
    let mut best_v = items[0].clone().num("argmin")?;
    for (i, v) in items.iter().enumerate().skip(1) {
        let n = v.clone().num("argmin")?;
        if n < best_v { best_v = n; best_i = i; }
    }
    Ok(Val::Num(best_i as f64))
}

"argmax" => { /* mirror of argmin with > */ }
```

---

### T13. Polymorphic `sum(tuple)` and `prod(tuple)` — `eval.rs`

`eval_agg` currently requires exactly 3 args. Add a 1-arg branch at the top:

```rust
pub fn eval_agg(args: &[Expr], env: &Env, product: bool) -> Result<Val, String> {
    let label = if product { "prod" } else { "sum" };
    // 1-arg form: sum(tuple) or prod(tuple)
    if args.len() == 1 {
        let items = match eval(&args[0], env)? {
            Val::Tuple(v) => v,
            _ => return Err(format!("{label}: 1-arg form requires a tuple")),
        };
        let mut acc = if product { 1.0 } else { 0.0 };
        for v in items {
            let n = v.num(label)?;
            if product { acc *= n; } else { acc += n; }
        }
        return Ok(Val::Num(acc));
    }
    // existing 3-arg (fn, start, stop) path unchanged ...
}
```

---

### T14. Add `mean`, `median`, `mode` — `eval.rs`, `repl.rs`

```rust
"mean" => {
    arity("mean", 1, vals.len())?;
    let items = /* extract tuple */;
    if items.is_empty() { return Err("mean: empty tuple".into()); }
    let sum: Result<f64, _> = items.iter().map(|v| v.clone().num("mean")).try_fold(0.0, |a, r| Ok(a + r?));
    Ok(Val::Num(sum? / items.len() as f64))
}

"median" => {
    // sort the nums, pick middle or average of two middles
}

"mode" => {
    // count exact f64 matches (via bits); return most frequent.
    // Works reliably for integer-valued data.
}
```

---

### T15. Add `std`, `var` — `eval.rs`, `repl.rs`

```rust
"var" => {
    // compute mean, then mean of squared deviations (population variance)
}
"std" => {
    // sqrt(var(tuple))
    // reuse var logic then sqrt
}
```

---

### T16. Add `compose(f, g)` and `partial(f, a)` builtins — `eval.rs`, `repl.rs`

`compose` wraps the already-private `compose_fns`:
```rust
"compose" => {
    arity("compose", 2, vals.len())?;
    let mut it = vals.into_iter();
    let f = it.next().unwrap();
    let g = it.next().unwrap();
    // validate both are callable (Fn or Builtin)
    Ok(compose_fns(f, g))
}
```

`partial(f, a)` binds the first argument of an n-param function and returns an (n-1)-param closure:
```rust
"partial" => {
    arity("partial", 2, vals.len())?;
    let mut it = vals.into_iter();
    let f = it.next().unwrap();
    let a = it.next().unwrap();
    match f {
        Val::Fn(params, body, mut captured) => {
            if params.is_empty() { return Err("partial: function has no parameters".into()); }
            let first = params[0].clone();
            let rest  = params[1..].to_vec();
            captured.insert(first, a);
            Ok(Val::Fn(rest, body, captured))
        }
        Val::Builtin(name) => {
            // Wrap builtin in a closure with first arg captured
            let mut cap = HashMap::new();
            cap.insert("__b__".into(), Val::Builtin(name));
            cap.insert("__a__".into(), a);
            // body: __b__(__a__, __z__)
            let body = Expr::Apply(
                Box::new(Expr::Var("__b__".into())),
                vec![Expr::Var("__a__".into()), Expr::Var("__z__".into())],
            );
            Ok(Val::Fn(vec!["__z__".into()], body, cap))
        }
        _ => Err("partial: first argument must be a function".into()),
    }
}
```

---

### T17. Add `gaussian(x, mu, sigma)` builtin — `eval.rs`, `repl.rs`, `advanced.math`

```rust
"gaussian" => {
    arity("gaussian", 3, vals.len())?;
    let mut it = vals.into_iter();
    let x     = it.next().unwrap().num("gaussian")?;
    let mu    = it.next().unwrap().num("gaussian")?;
    let sigma = it.next().unwrap().num("gaussian")?;
    if sigma == 0.0 { return Err("gaussian: sigma must be nonzero".into()); }
    let z = (x - mu) / sigma;
    Ok(Val::Num((1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()))
}
```

Remove `gaussian_pdf` and `gaussian_cdf` from `advanced.math`. (`gaussian_cdf` can be written by the user using the `erf` builtin.)

---

### T18. Add `filter(f, tuple)` and `reduce(f, tuple)` as special forms — `eval.rs`, `repl.rs`

Handle in `Expr::Apply` special-form dispatch so the function arg can be a raw lambda expr (same as `map`, `sum`, `prod`):

```rust
"filter" => {
    if arg_exprs.len() != 2 { return Err("filter(f, tuple) expects 2 args".into()); }
    let items = match eval(&arg_exprs[1], env)? {
        Val::Tuple(v) => v,
        other => return Err(format!("filter: second arg must be a tuple, got {}", fmt_val(&other))),
    };
    let mut out = vec![];
    for item in items {
        let keep = call_fn1(&arg_exprs[0], item.clone(), env)?.num("filter")?;
        if keep != 0.0 { out.push(item); }
    }
    Ok(Val::Tuple(out))
}

"reduce" => {
    if arg_exprs.len() != 2 { return Err("reduce(f, tuple) expects 2 args".into()); }
    let items = match eval(&arg_exprs[1], env)? {
        Val::Tuple(v) => v,
        other => return Err(format!("reduce: second arg must be a tuple, got {}", fmt_val(&other))),
    };
    if items.is_empty() { return Err("reduce: empty tuple".into()); }
    let f_val = eval(&arg_exprs[0], env)?;
    let mut acc = items[0].clone();
    for item in items.into_iter().skip(1) {
        acc = apply_val(f_val.clone(), vec![acc, item], env)?;
    }
    Ok(acc)
}
```

Add `"filter"` and `"reduce"` to `Env::new()` (as `Builtin`), `is_protected()`, `BUILTIN_FNS`.

---

### T19. Add `rand()` / `rand(a, b)` — `Cargo.toml`, `eval.rs`, `repl.rs`

Add to `Cargo.toml`:
```toml
[dependencies]
rand = "0.8"
```

```rust
"rand" => {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    match vals.len() {
        0 => Ok(Val::Num(rng.gen::<f64>())),
        2 => {
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num("rand")?;
            let b = it.next().unwrap().num("rand")?;
            Ok(Val::Num(a + (b - a) * rng.gen::<f64>()))
        }
        n => Err(format!("rand expects 0 or 2 args, got {n}")),
    }
}
```

`rand` is a 0-or-2-arg builtin, so `arity()` isn't used; check `vals.len()` directly.

---

## Tier 3 — Medium (new parser/eval logic, multiple files)

### T20. Add `if(cond, a, b)` special form — `eval.rs`, `repl.rs`

**Why a special form:** lazy evaluation of branches prevents errors like `sqrt(-1)` when the condition is false. Also enables `if(cond, fn1, fn2)(x)`.

In `Expr::Apply` special-form dispatch (where `"sum"`, `"map"`, etc. are handled):

```rust
"if" => {
    if arg_exprs.len() != 3 {
        return Err("if(cond, a, b) expects 3 args".into());
    }
    let cond = eval(&arg_exprs[0], env)?.num("if")?;
    if cond != 0.0 { eval(&arg_exprs[1], env) }
    else           { eval(&arg_exprs[2], env) }
}
```

`if(cond, fn1, fn2)(x)` parses as `Apply(Apply(Var("if"), [cond, fn1, fn2]), [x])`. The inner Apply returns a `Val::Fn` via the special form; the outer Apply calls it with `x`. This works automatically with no extra machinery.

Add `"if"` to `Env::new()` as `Val::Builtin("if")`, to `is_protected()`, and to `repl::BUILTIN_FNS` (for tab-completion / highlighting). It is **not** dispatched via `eval_builtin`; the special-form arm in `eval` intercepts it first.

---

### T21. Postfix `!` factorial — `lexer.rs`, `parser.rs`

**Lexer** — add `Bang` token (reuse or add alongside the `BangEq` token that will be needed for `!=`). In the `b'!'` arm:
```rust
b'!' => {
    self.bump();
    if self.peek() == Some(b'=') { self.bump(); out.push(Token::BangEq); }
    else { out.push(Token::Bang); }
}
```

Add `Bang` and `BangEq` to the `Token` enum.

**Parser** `postfix()` — after every existing postfix step, check:
```rust
} else if *self.peek() == Token::Bang {
    self.bump();
    e = Expr::Apply(Box::new(Expr::Var("fact".into())), vec![e]);
}
```

No eval changes needed; `fact` already handles the computation.

---

### T22. Tuple slicing and multi-index `tuple[a..b]`, `tuple[i,j,k]` — `parser.rs`, `eval.rs`

#### Parser (`postfix()`)

Replace the `LBracket` arm with:
```rust
if *self.peek() == Token::LBracket {
    self.bump();
    let first = self.expr()?;
    if *self.peek() == Token::DotDot {
        // [a..b] — range slice
        self.bump();
        let last = self.expr()?;
        self.eat(&Token::RBracket)?;
        e = Expr::Index(Box::new(e), Box::new(Expr::Range(Box::new(first), Box::new(last))));
    } else if *self.peek() == Token::Comma {
        // [i, j, k] — multi-index
        let mut indices = vec![first];
        while *self.peek() == Token::Comma {
            self.bump();
            if *self.peek() == Token::RBracket { break; }
            indices.push(self.expr()?);
        }
        self.eat(&Token::RBracket)?;
        e = Expr::Index(Box::new(e), Box::new(Expr::Tuple(indices)));
    } else {
        // [n] — single index (existing)
        self.eat(&Token::RBracket)?;
        e = Expr::Index(Box::new(e), Box::new(first));
    }
}
```

#### Eval (`Expr::Index`)

`Expr::Range` already evaluates to a `Val::Tuple` of consecutive integers, so range-slice and multi-index share the same path:

```rust
Expr::Index(expr, idx) => {
    let v   = eval(expr, env)?;
    let idx = eval(idx, env)?;
    match (v, idx) {
        (Val::Tuple(items), Val::Num(n)) => {
            let len = items.len() as i64;
            let i   = if n as i64 < 0 { (len + n as i64).max(0) as usize } else { n as usize };
            items.into_iter().nth(i).ok_or_else(|| format!("index {n} out of range"))
        }
        (Val::Tuple(items), Val::Tuple(indices)) => {
            let len = items.len() as i64;
            let result: Result<Vec<Val>, _> = indices.into_iter().map(|iv| {
                let n = iv.num("index")? as i64;
                let i = if n < 0 { (len + n).max(0) as usize } else { n as usize };
                items.iter().nth(i).cloned().ok_or_else(|| format!("index {n} out of range"))
            }).collect();
            Ok(Val::Tuple(result?))
        }
        _ => Err("indexing requires a tuple".into()),
    }
}
```

`tuple[0..3]` → `Range(0,3)` → `Val::Tuple([0,1,2,3])` (inclusive, matching paren-syntax `(a..b)`) → multi-index path → elements 0,1,2,3. Documents cleanly: bracket-range is inclusive on both ends.

---

### T23. Comparison operators `<` `>` `<=` `>=` `==` `!=` — `lexer.rs`, `ast.rs`, `parser.rs`, `eval.rs`

#### Lexer

Add tokens: `Lt`, `Gt`, `LtEq`, `GtEq`, `EqEq`, `BangEq` (already needed for T21).

```rust
b'<' => {
    self.bump();
    if self.peek() == Some(b'=') { self.bump(); out.push(Token::LtEq); }
    else { out.push(Token::Lt); }
}
b'>' => {
    self.bump();
    if self.peek() == Some(b'=') { self.bump(); out.push(Token::GtEq); }
    else { out.push(Token::Gt); }
}
b'=' => {
    self.bump();
    if self.peek() == Some(b'=') { self.bump(); out.push(Token::EqEq); }
    else { out.push(Token::Eq); }  // assignment, unchanged
}
// b'!' handled in T21
```

#### AST

Add to `Op`:
```rust
pub enum Op { Add, Sub, Mul, Div, FloorDiv, Rem, Pow, Lt, Gt, LtEq, GtEq, Eq, Ne }
```

#### Parser

Insert a new `cmp()` method between `expr()` and `add()` in the precedence chain:

```rust
fn expr(&mut self) -> Result<Expr, String> { self.cmp() }

fn cmp(&mut self) -> Result<Expr, String> {
    let l = self.add()?;
    let op = match self.peek() {
        Token::Lt    => Op::Lt,
        Token::Gt    => Op::Gt,
        Token::LtEq  => Op::LtEq,
        Token::GtEq  => Op::GtEq,
        Token::EqEq  => Op::Eq,
        Token::BangEq => Op::Ne,
        _ => return Ok(l),
    };
    self.bump();
    Ok(Expr::BinOp(l.into(), op, self.add()?.into()))
}
```

Non-chaining: one comparison per expression. Chained `1 < 2 < 3` would require nesting comparisons explicitly or a loop; this can be added later if desired.

#### Eval — `scalar_binop`

Add arms to the `Op::*` match:
```rust
Op::Lt   => Ok(Val::Num(if la < ra  { 1.0 } else { 0.0 })),
Op::Gt   => Ok(Val::Num(if la > ra  { 1.0 } else { 0.0 })),
Op::LtEq => Ok(Val::Num(if la <= ra { 1.0 } else { 0.0 })),
Op::GtEq => Ok(Val::Num(if la >= ra { 1.0 } else { 0.0 })),
Op::Eq   => Ok(Val::Num(if la == ra { 1.0 } else { 0.0 })),
Op::Ne   => Ok(Val::Num(if la != ra { 1.0 } else { 0.0 })),
```

For complex numbers, define `==`/`!=` component-wise; `<`/`>` etc. on complex numbers return an error.

#### Tuple `==` / `!=` special case

In the `BinOp` eval arm, before the tuple-broadcast path, intercept `==`/`!=` on two tuples:
```rust
if matches!((&lv, &rv, op), (Val::Tuple(_), Val::Tuple(_), Op::Eq | Op::Ne)) {
    if let (Val::Tuple(ls), Val::Tuple(rs)) = (lv, rv) {
        let equal = ls.len() == rs.len() &&
            ls.iter().zip(rs.iter()).all(|(a, b)| {
                matches!((a, b), (Val::Num(x), Val::Num(y)) if x == y)
            });
        let result = if matches!(op, Op::Eq) { equal } else { !equal };
        return Ok(Val::Num(if result { 1.0 } else { 0.0 }));
    }
}
```

Other comparisons on tuples (`<`, `>`, etc.) fall through to the element-wise broadcast, producing a tuple of 0s and 1s.

#### Operator aliases for `and` / `or`

Add `&` and `|` as single-character infix operator aliases for the existing `and(a,b)` / `or(a,b)` bitwise builtins. Add `Amp` and `Pipe` tokens to the lexer (being careful not to consume `&&` or `||` — just single chars). Wire them as `Op::And` / `Op::Or` at the same precedence level as the other comparison/bitwise operators, evaluating identically to `and(a,b)` / `or(a,b)` (integer bitwise, not logical short-circuit).

#### Function aliases for comparison operators

Register `lt`, `leq`, `gt`, `geq`, `eq`, `neq` as regular 2-arg builtins in `eval_builtin` that perform the same scalar comparison as `<`, `<=`, `>`, `>=`, `==`, `!=` and return `0.0` or `1.0`. This allows comparisons to be passed to higher-order functions without wrapping in a lambda:

```
filter(lt(_, 3), (1,2,3,4,5))   # not possible yet — but partial(lt, 3) works
map(eq(_, 0), tuple)
```

In practice, `partial(lt, 3)` gives `x -> 3 < x`, so users can do `filter(partial(lt, 3), t)` immediately once comparisons land.

**Remove `delta` after this change.** `delta(x)` returns `1` if `x == 0`, else `0` — exactly `x == 0` with the new operator. It should be removed from `Env::new()`, `is_protected()`, `eval_builtin`, and `repl::BUILTIN_FNS`, and dropped from the README built-in functions table.

---

## Tier 4 — Hard

### T24. Implicit multiplication `2pi`, `2sin(x)`, `2(3+4)` — `parser.rs`

In `primary()`, the `Token::Num(n)` arm currently just returns `Expr::Num(n)`. Extend it:

```rust
Token::Num(n) => {
    self.bump();
    // Implicit multiplication when a number is immediately followed by
    // an identifier or an open paren (no whitespace required at token level).
    if matches!(self.peek(), Token::Ident(_) | Token::LParen) {
        let rhs = self.primary()?;
        Ok(Expr::BinOp(Box::new(Expr::Num(n)), Op::Mul, Box::new(rhs)))
    } else {
        Ok(Expr::Num(n))
    }
}
```

The recursive `self.primary()` call handles `2sin(x)` correctly because the Ident arm already parses function calls. Operator precedence is unchanged: since this fires inside `primary()`, the implicit mul has the same precedence as explicit `*`.

**Edge cases:**
- `2 + 3`: next token after `2` is `Plus`, not `Ident`/`LParen` → no implicit mul ✓
- `2^3`: next is `Caret` → no implicit mul ✓
- `2 -3`: next is `Minus` → no implicit mul; parses as `2 - 3 = -1` ✓
- `2 3` (space-separated): tokeniser produces `Num(2) Num(3)`, next token after `2` is `Num(3)` not Ident/LParen → **no** implicit mul; still a parse error ✓ (keep it this way to avoid silent bugs)
- `2(3, 4)`: next is `LParen` → `primary()` parses `(3, 4)` as a Tuple → `2 * (3, 4)` → `(6, 8)` via tuple broadcast ✓

---

### T25. FFT / IFFT for spectral PDE solving — `eval.rs`, `Cargo.toml`

Add `fft(tuple)` and `ifft(tuple)` builtins using the `rustfft` crate. Combined with `linspace` (already implemented), this enables O(N log N) spectral PDE methods as an alternative to the current finite-difference `heatSolution`.

**Use cases:**
- Replace `heatSolutionBounded` with a spectral method
- Wave equation and Schrödinger equation solvers
- Spectral differentiation (multiply by `i*ξ` in frequency domain)

**Sketch:**
- Add `rustfft` to `Cargo.toml`
- `fft(tuple)` → accepts a tuple of real or complex values, returns a tuple of complex values (the DFT)
- `ifft(tuple)` → inverse DFT, returns complex tuple
- Both broadcast over the full tuple; output length equals input length

---

## Files touched — summary

| File | Changes |
|------|---------|
| `src/eval.rs` | All new builtins, `if` special form, `filter`/`reduce` special forms, updated `sum`/`prod`, `Expr::Index` negative/multi/slice, comparison ops in `scalar_binop`/`BinOp` eval |
| `src/lexer.rs` | Add `Lt`, `Gt`, `LtEq`, `GtEq`, `EqEq`, `Bang`, `BangEq`; fix `=`/`==` and `!`/`!=` disambiguation |
| `src/ast.rs` | Add `Op::Lt, Gt, LtEq, GtEq, Eq, Ne` |
| `src/parser.rs` | `cmp()` precedence level; extended `[...]` (slice/multi-index); postfix `!`; implicit mul in `primary()` |
| `src/repl.rs` | Update `BUILTIN_FNS` and `BUILTIN_CONSTS` |
| `Cargo.toml` | Add `rand = "0.8"` |
| `README.md` | Remove implicit-functions section, remove tau from constants, update tuple-print examples, document all new features |
| `advanced.math` | Remove `gaussian_pdf` / `gaussian_cdf` lines |
