// forms — exterior calculus on gridded fields (differential forms / vector fields).
//
// A field carries grid spacing (dx, per axis) and a constant diagonal metric
// (g_ii, per axis). The two are kept strictly separate:
//
//   * the exterior derivative `d` is METRIC-FREE — it uses only the spacing and
//     boundary conditions. grad/curl/div are all `d` in disguise.
//   * `hodge`/`raise`/`lower`/`codiff`/`laplace` use the METRIC — Euclidean by
//     default (all g_ii = 1), Minkowski via a signature like (-1, 1, 1, 1).
//
// A k-form on an n-D grid has C(n,k) components, laid out with the component
// (subset) index varying fastest: component c at grid point p lives at
// data[p*ncomp + c]. Components are indexed by the sorted k-subsets of
// {0,..,n-1} in lexicographic order (see `subsets`).
use crate::eval::{Val, TData, FieldVal, BC, Variance, fmt_val};
use std::collections::HashMap;
use std::sync::Arc;

pub const NAMES: &[&str] = &["d", "hodge", "wedge", "raise", "lower", "codiff", "laplace"];

pub fn members() -> HashMap<String, Val> {
    let mut m: HashMap<String, Val> =
        NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect();
    // BC sentinels for the `field` constructor (mathlang has no string type).
    m.insert("periodic".into(), Val::Num(0.0));
    m.insert("neumann".into(),  Val::Num(1.0));
    m
}

pub fn dispatch(name: &str, vals: Vec<Val>, _env: &crate::eval::Env) -> Result<Val, String> {
    match name {
        "d"      => d(vals),
        "hodge"  => hodge(vals),
        "wedge"  => wedge(vals),
        "raise"  => raise(vals),
        "lower"  => lower(vals),
        "codiff" => codiff(vals),
        "laplace" => laplace(vals),
        _ => Err(format!("forms: unknown member '{name}'")),
    }
}

// ── argument coercion ─────────────────────────────────────────────────────────

fn as_field(v: Val, what: &str) -> Result<Arc<FieldVal>, String> {
    match v {
        Val::Field(f) => Ok(f),
        other => Err(format!("{what}: expected a field, got {}", fmt_val(&other))),
    }
}

/// Coerce a scalar-or-tuple argument into a length-n Vec<f64> (a scalar broadcasts).
fn as_axis_vec(v: &Val, n: usize, what: &str) -> Result<Vec<f64>, String> {
    match v {
        Val::Num(x) => Ok(vec![*x; n]),
        Val::Tuple(items) if items.len() == n =>
            items.iter().map(|t| t.clone().num(what)).collect(),
        Val::Tensor { data, shape } if shape.len() == 1 && data.len() == n =>
            Ok(data.to_vec()),
        other => Err(format!("{what}: expected a scalar or {n}-tuple, got {}", fmt_val(other))),
    }
}

// ── combinatorics ───────────────────────────────────────────────────────────

/// Sorted k-subsets of {0,..,n-1} in lexicographic order (the canonical basis
/// order for k-form components). subsets(n,0) == [[]].
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    if k > n { return res; }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        res.push(idx.clone());
        if k == 0 { break; }
        let mut i = k - 1;
        loop {
            if idx[i] < i + n - k {
                idx[i] += 1;
                for j in i + 1..k { idx[j] = idx[j - 1] + 1; }
                break;
            }
            if i == 0 { return res; }
            i -= 1;
        }
    }
    res
}

/// Position of a sorted subset within subsets(n, set.len()).
fn subset_index(table: &[Vec<usize>], set: &[usize]) -> usize {
    table.iter().position(|s| s == set).expect("subset present in table")
}

/// Sign of the permutation that sorts `seq` into ascending order: (-1)^inversions.
fn perm_sign(seq: &[usize]) -> i32 {
    let mut inv = 0usize;
    for i in 0..seq.len() {
        for j in i + 1..seq.len() {
            if seq[i] > seq[j] { inv += 1; }
        }
    }
    if inv % 2 == 0 { 1 } else { -1 }
}

/// Sorted complement of `set` within {0,..,n-1}.
fn complement(set: &[usize], n: usize) -> Vec<usize> {
    (0..n).filter(|i| !set.contains(i)).collect()
}

// ── grid plumbing ─────────────────────────────────────────────────────────────

fn grid_total(grid: &[usize]) -> usize { grid.iter().product::<usize>().max(1) }

/// Extract component `c` (a sorted-subset index) as a flat grid array.
fn component(f: &FieldVal, c: usize, ncomp: usize) -> Vec<f64> {
    let gt = grid_total(&f.grid);
    (0..gt).map(|p| f.data[p * ncomp + c]).collect()
}

/// Shift a grid array by `n` cells along `axis` (periodic wrap or edge clamp).
fn shift_axis(data: &[f64], grid: &[usize], n: i64, axis: usize, bc: BC) -> Vec<f64> {
    let total = data.len();
    let stride: usize = grid[axis + 1..].iter().product();
    let dim = grid[axis] as i64;
    let mut out = vec![0.0; total];
    for o in 0..total {
        let ax = ((o / stride) % grid[axis]) as i64;
        let in_ax = match bc {
            BC::Periodic => (ax - n).rem_euclid(dim),
            BC::Neumann  => (ax - n).clamp(0, dim - 1),
        };
        let in_flat = o as i64 + (in_ax - ax) * stride as i64;
        out[o] = data[in_flat as usize];
    }
    out
}

/// Central first derivative of a grid array along `axis`: (f[+1]-f[-1])/(2 dx).
fn partial(data: &[f64], grid: &[usize], dx: f64, axis: usize, bc: BC) -> Vec<f64> {
    let fwd = shift_axis(data, grid, -1, axis, bc);
    let bwd = shift_axis(data, grid,  1, axis, bc);
    fwd.iter().zip(bwd.iter()).map(|(&a, &b)| (a - b) / (2.0 * dx)).collect()
}

/// Rebuild a field with new component data + (possibly) new degree/variance,
/// preserving grid/spacing/lo/bc/metric.
fn rebuild(f: &FieldVal, data: Vec<f64>, degree: usize, variance: Variance) -> Val {
    Val::Field(Arc::new(FieldVal {
        data: TData::new(data), degree, variance,
        grid: f.grid.clone(), spacing: f.spacing.clone(), lo: f.lo.clone(),
        bc: f.bc.clone(), metric: f.metric.clone(),
    }))
}

// ── constructor (field) ───────────────────────────────────────────────────────

// field(data, lo, hi, bc [, metric]) — build a 0-form (scalar field) on the box
// [lo, hi] sampled by `data`. bc is forms.periodic (0) or forms.neumann (1) per
// axis (scalar broadcasts). Optional metric is the diagonal g_ii (default 1s).
pub fn field_ctor(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 4 || vals.len() > 5 {
        return Err("field(data, lo, hi, bc [, metric]) expects 4 or 5 args".into());
    }
    let mut it = vals.into_iter();
    let (data, grid) = match it.next().unwrap() {
        Val::Tensor { data, shape } => (data, shape),
        other => return Err(format!("field: data must be a real tensor, got {}", fmt_val(&other))),
    };
    let n = grid.len();
    if n == 0 { return Err("field: data must have rank >= 1".into()); }
    let lo  = as_axis_vec(&it.next().unwrap(), n, "field lo")?;
    let hi  = as_axis_vec(&it.next().unwrap(), n, "field hi")?;
    let bcv = as_axis_vec(&it.next().unwrap(), n, "field bc")?;
    let metric = match it.next() {
        Some(m) => as_axis_vec(&m, n, "field metric")?,
        None    => vec![1.0; n],
    };
    if metric.iter().any(|&g| g == 0.0) { return Err("field: metric entries must be nonzero".into()); }
    let bc: Vec<BC> = bcv.iter().map(|&b| if b == 0.0 { BC::Periodic } else { BC::Neumann }).collect();
    // Spacing: periodic boxes [lo,hi) exclude the duplicate endpoint (÷N); a
    // Neumann/clamped axis includes both endpoints (÷(N-1)).
    let spacing: Vec<f64> = (0..n).map(|a| {
        let span = hi[a] - lo[a];
        match bc[a] {
            BC::Periodic => span / grid[a] as f64,
            BC::Neumann  => span / (grid[a].max(2) - 1) as f64,
        }
    }).collect();
    Ok(Val::Field(Arc::new(FieldVal {
        data, grid, spacing, lo, bc, metric, degree: 0, variance: Variance::Form,
    })))
}

/// Rebuild a field with new component data but identical geometry/degree/variance.
pub fn with_data(f: &FieldVal, data: Vec<f64>) -> Val {
    Val::Field(Arc::new(FieldVal { data: TData::new(data), ..f.clone() }))
}

// ── primitives ─────────────────────────────────────────────────────────────────

// d(f) — exterior derivative, k-form -> (k+1)-form. Metric-free; uses spacing+BC.
//   (dω)_J = Σ_p (-1)^p ∂_{J[p]} ω_{J∖{J[p]}}.
fn d(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 { return Err("forms.d(f) expects 1 arg".into()); }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.d")?;
    let n = f.grid.len();
    let k = f.degree;
    if k + 1 > n {
        return Err(format!("forms.d: cannot differentiate a {k}-form on a {n}-D grid (degree would exceed {n})"));
    }
    let in_sub  = subsets(n, k);
    let out_sub = subsets(n, k + 1);
    let nc_in  = in_sub.len();
    let nc_out = out_sub.len();
    let gt = grid_total(&f.grid);
    let mut out = vec![0.0; gt * nc_out];
    for (out_c, jset) in out_sub.iter().enumerate() {
        for (p, &j) in jset.iter().enumerate() {
            let mut iset = jset.clone();
            iset.remove(p);                       // J ∖ {J[p]}, still sorted
            let in_c = subset_index(&in_sub, &iset);
            let comp = component(&f, in_c, nc_in);
            let dcomp = partial(&comp, &f.grid, f.spacing[j], j, f.bc[j]);
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            for gp in 0..gt { out[gp * nc_out + out_c] += sign * dcomp[gp]; }
        }
    }
    Ok(rebuild(&f, out, k + 1, f.variance))
}

// hodge(f) — Hodge star, k-form -> (n-k)-form (metric-aware).
//   ★(dx^I) = sqrt|det g| · (Π_{i∈I} g^{ii}) · ε(I, Iᶜ) · dx^{Iᶜ}.
fn hodge(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 { return Err("forms.hodge(f) expects 1 arg".into()); }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.hodge")?;
    let n = f.grid.len();
    let k = f.degree;
    let in_sub  = subsets(n, k);
    let out_sub = subsets(n, n - k);
    let nc_in  = in_sub.len();
    let nc_out = out_sub.len();
    let gt = grid_total(&f.grid);
    let sqrt_det = f.metric.iter().map(|g| g.abs().sqrt()).product::<f64>();
    let mut out = vec![0.0; gt * nc_out];
    for (in_c, iset) in in_sub.iter().enumerate() {
        let ic = complement(iset, n);
        let out_c = subset_index(&out_sub, &ic);
        let inv_g: f64 = iset.iter().map(|&i| 1.0 / f.metric[i]).product();
        let mut concat = iset.clone();
        concat.extend_from_slice(&ic);
        let eps = perm_sign(&concat) as f64;
        let coeff = sqrt_det * inv_g * eps;
        let comp = component(&f, in_c, nc_in);
        for gp in 0..gt { out[gp * nc_out + out_c] = coeff * comp[gp]; }
    }
    Ok(rebuild(&f, out, n - k, f.variance))
}

// wedge(a, b) — exterior product, (ka-form) ∧ (kb-form) -> (ka+kb)-form.
//   (α∧β)_K = Σ_{I⊔J=K} ε(I,J) α_I β_J  (pointwise on the shared grid).
fn wedge(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 2 { return Err("forms.wedge(a, b) expects 2 args".into()); }
    let mut it = vals.into_iter();
    let a = as_field(it.next().unwrap(), "forms.wedge")?;
    let b = as_field(it.next().unwrap(), "forms.wedge")?;
    if a.grid != b.grid || a.spacing != b.spacing || a.bc != b.bc || a.metric != b.metric {
        return Err("forms.wedge: operands must share the same grid geometry".into());
    }
    let n = a.grid.len();
    let (ka, kb) = (a.degree, b.degree);
    if ka + kb > n {
        return Err(format!("forms.wedge: degree {ka}+{kb} exceeds grid dimension {n}"));
    }
    let a_sub = subsets(n, ka);
    let b_sub = subsets(n, kb);
    let out_sub = subsets(n, ka + kb);
    let (nc_a, nc_b, nc_out) = (a_sub.len(), b_sub.len(), out_sub.len());
    let gt = grid_total(&a.grid);
    let mut out = vec![0.0; gt * nc_out];
    for (ia, iset) in a_sub.iter().enumerate() {
        let acomp = component(&a, ia, nc_a);
        for (ib, jset) in b_sub.iter().enumerate() {
            if iset.iter().any(|x| jset.contains(x)) { continue; }   // disjoint only
            let mut concat = iset.clone();
            concat.extend_from_slice(jset);
            let eps = perm_sign(&concat) as f64;
            let mut kset = concat.clone();
            kset.sort_unstable();
            let out_c = subset_index(&out_sub, &kset);
            let bcomp = component(&b, ib, nc_b);
            for gp in 0..gt { out[gp * nc_out + out_c] += eps * acomp[gp] * bcomp[gp]; }
        }
    }
    Ok(rebuild(&a, out, ka + kb, a.variance))
}

// lower(f) — musical flat (♭): vector field -> form, component I scaled by Π g_ii.
fn lower(vals: Vec<Val>) -> Result<Val, String> {
    musical(vals, "forms.lower", false)
}
// raise(f) — musical sharp (♯): form -> vector field, component I scaled by Π g^ii.
fn raise(vals: Vec<Val>) -> Result<Val, String> {
    musical(vals, "forms.raise", true)
}

fn musical(vals: Vec<Val>, what: &str, up: bool) -> Result<Val, String> {
    if vals.len() != 1 { return Err(format!("{what}(f) expects 1 arg")); }
    let f = as_field(vals.into_iter().next().unwrap(), what)?;
    let n = f.grid.len();
    let sub = subsets(n, f.degree);
    let nc = sub.len();
    let gt = grid_total(&f.grid);
    let mut out = f.data.to_vec();
    for (c, iset) in sub.iter().enumerate() {
        let scale: f64 = iset.iter()
            .map(|&i| if up { 1.0 / f.metric[i] } else { f.metric[i] })
            .product();
        for gp in 0..gt { out[gp * nc + c] *= scale; }
    }
    let variance = if up { Variance::Vector } else { Variance::Form };
    Ok(rebuild(&f, out, f.degree, variance))
}

// codiff(f) — codifferential δ = (-1)^{n(k+1)+1} ★d★, k-form -> (k-1)-form.
// The metric (incl. Minkowski signature) enters through the two Hodge stars.
fn codiff(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 { return Err("forms.codiff(f) expects 1 arg".into()); }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.codiff")?;
    let (n, k) = (f.grid.len(), f.degree);
    if k == 0 {
        return Err("forms.codiff: the codifferential of a 0-form is 0 (degree -1)".into());
    }
    let step1 = hodge(vec![Val::Field(f.clone())])?;                 // ★f      : (n-k)-form
    let step2 = d(vec![step1])?;                                     // d★f     : (n-k+1)-form
    let step3 = hodge(vec![step2])?;                                 // ★d★f    : (k-1)-form
    let sign = if (n * (k + 1) + 1) % 2 == 0 { 1.0 } else { -1.0 };
    match step3 {
        Val::Field(g) => {
            let scaled: Vec<f64> = g.data.iter().map(|x| sign * x).collect();
            Ok(rebuild(&g, scaled, g.degree, g.variance))
        }
        other => Ok(other),
    }
}

// laplace(f) — Laplace–de Rham operator Δ = dδ + δd. On a 0-form in Euclidean
// space this is the ordinary Laplacian; with a Minkowski metric it is the
// d'Alembertian □ = -∂_t² + ∇².
fn laplace(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 { return Err("forms.laplace(f) expects 1 arg".into()); }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.laplace")?;
    let k = f.degree;
    // δd f  (always defined)
    let dd = d(vec![Val::Field(f.clone())])?;
    let term_dd = codiff(vec![dd])?;
    if k == 0 { return Ok(term_dd); }              // δ(0-form)=0, so Δ = δd only
    // dδ f
    let cd = codiff(vec![Val::Field(f.clone())])?;
    let term_cd = d(vec![cd])?;
    let (a, b) = (as_field(term_dd, "forms.laplace")?, as_field(term_cd, "forms.laplace")?);
    let sum: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
    Ok(rebuild(&a, sum, a.degree, a.variance))
}
