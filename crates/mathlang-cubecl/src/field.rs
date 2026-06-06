//! Fields and differential forms — exterior calculus on gridded data.
//!
//! A field carries grid spacing (dx, per axis) and a constant diagonal metric
//! (g_ii, per axis), kept strictly separate: the exterior derivative `d` uses only
//! spacing + boundary conditions (metric-free), while hodge/raise/lower/codiff/
//! laplace use the metric (Euclidean by default; Minkowski via a signature like
//! (-1,1,1,1)). A k-form on an n-D grid has C(n,k) components laid out with the
//! component index fastest: `data[p*ncomp + c]`.
//!
//! This is a faithful host-side port of the original `src/ns/forms.rs`. Field data
//! lives on the host; `tensor(field)` uploads it to a device tensor, and `field`/
//! `forms.form`/`forms.vector` download the input tensor. A device-resident field is
//! a later optimization.

use crate::ast::Op;
use crate::compute;
use crate::interp::Env;
use crate::value::{fmt_f, fmt_val, Val};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BC {
    Periodic,
    Neumann,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variance {
    Form,
    Vector,
}

/// A k-form / vector field sampled on a regular grid (host data).
#[derive(Clone, Debug)]
pub struct FieldVal {
    pub data: Arc<Vec<f64>>,
    pub grid: Vec<usize>,
    pub spacing: Vec<f64>,
    pub lo: Vec<f64>,
    pub bc: Vec<BC>,
    pub metric: Vec<f64>,
    pub degree: usize,
    pub variance: Variance,
}

impl FieldVal {
    pub fn ncomp(&self) -> usize {
        binomial(self.grid.len(), self.degree)
    }
}

// ── combinatorics ───────────────────────────────────────────────────────────────

pub fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r = 1usize;
    for i in 0..k {
        r = r * (n - i) / (i + 1);
    }
    r
}

/// Sorted k-subsets of {0,..,n-1} in lexicographic order.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    if k > n {
        return res;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        res.push(idx.clone());
        if k == 0 {
            break;
        }
        let mut i = k - 1;
        loop {
            if idx[i] < i + n - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return res;
            }
            i -= 1;
        }
    }
    res
}

fn subset_index(table: &[Vec<usize>], set: &[usize]) -> usize {
    table.iter().position(|s| s == set).expect("subset present in table")
}

fn perm_sign(seq: &[usize]) -> i32 {
    let mut inv = 0usize;
    for i in 0..seq.len() {
        for j in i + 1..seq.len() {
            if seq[i] > seq[j] {
                inv += 1;
            }
        }
    }
    if inv % 2 == 0 { 1 } else { -1 }
}

fn complement(set: &[usize], n: usize) -> Vec<usize> {
    (0..n).filter(|i| !set.contains(i)).collect()
}

// ── grid plumbing ─────────────────────────────────────────────────────────────

fn grid_total(grid: &[usize]) -> usize {
    grid.iter().product::<usize>().max(1)
}

fn component(f: &FieldVal, c: usize, ncomp: usize) -> Vec<f64> {
    let gt = grid_total(&f.grid);
    (0..gt).map(|p| f.data[p * ncomp + c]).collect()
}

fn shift_axis(data: &[f64], grid: &[usize], n: i64, axis: usize, bc: BC) -> Vec<f64> {
    let total = data.len();
    let stride: usize = grid[axis + 1..].iter().product();
    let dim = grid[axis] as i64;
    let mut out = vec![0.0; total];
    for o in 0..total {
        let ax = ((o / stride) % grid[axis]) as i64;
        let in_ax = match bc {
            BC::Periodic => (ax - n).rem_euclid(dim),
            BC::Neumann => (ax - n).clamp(0, dim - 1),
        };
        let in_flat = o as i64 + (in_ax - ax) * stride as i64;
        out[o] = data[in_flat as usize];
    }
    out
}

fn partial(data: &[f64], grid: &[usize], dx: f64, axis: usize, bc: BC) -> Vec<f64> {
    let fwd = shift_axis(data, grid, -1, axis, bc);
    let bwd = shift_axis(data, grid, 1, axis, bc);
    fwd.iter().zip(bwd.iter()).map(|(&a, &b)| (a - b) / (2.0 * dx)).collect()
}

fn rebuild(f: &FieldVal, data: Vec<f64>, degree: usize, variance: Variance) -> Val {
    Val::Field(Arc::new(FieldVal {
        data: Arc::new(data),
        degree,
        variance,
        grid: f.grid.clone(),
        spacing: f.spacing.clone(),
        lo: f.lo.clone(),
        bc: f.bc.clone(),
        metric: f.metric.clone(),
    }))
}

/// Rebuild a field with new component data but identical geometry/degree/variance.
pub fn with_data(f: &FieldVal, data: Vec<f64>) -> Val {
    Val::Field(Arc::new(FieldVal { data: Arc::new(data), ..f.clone() }))
}

// ── argument coercion ─────────────────────────────────────────────────────────

fn as_field(v: Val, what: &str) -> Result<Arc<FieldVal>, String> {
    match v {
        Val::Field(f) => Ok(f),
        other => Err(format!("{what}: expected a field, got {}", fmt_val(&other))),
    }
}

fn as_axis_vec(v: &Val, n: usize, what: &str) -> Result<Vec<f64>, String> {
    match v {
        Val::Num(x) => Ok(vec![*x; n]),
        Val::Tuple(items) if items.len() == n => items.iter().map(|t| t.clone().num(what)).collect(),
        Val::Tensor(t) if t.shape.len() == 1 && t.len == n => compute::download(t),
        other => Err(format!("{what}: expected a scalar or {n}-tuple, got {}", fmt_val(other))),
    }
}

fn as_count_vec(v: &Val) -> Result<Vec<usize>, String> {
    match v {
        Val::Num(x) => Ok(vec![*x as usize]),
        Val::Tuple(items) => items.iter().map(|t| Ok(t.clone().num("field counts")? as usize)).collect(),
        Val::Tensor(t) if t.shape.len() == 1 => Ok(compute::download(t)?.iter().map(|&x| x as usize).collect()),
        other => Err(format!("field counts: expected a scalar or tuple, got {}", fmt_val(other))),
    }
}

fn axis_spacing(lo: &[f64], hi: &[f64], grid: &[usize], bc: &[BC]) -> Vec<f64> {
    (0..grid.len())
        .map(|a| {
            let span = hi[a] - lo[a];
            match bc[a] {
                BC::Periodic => span / grid[a] as f64,
                BC::Neumann => span / (grid[a].max(2) - 1) as f64,
            }
        })
        .collect()
}

fn split_shape(shape: &[usize], degree: usize, what: &str) -> Result<(Vec<usize>, usize), String> {
    if shape.is_empty() {
        return Err(format!("{what}: data must have rank >= 1"));
    }
    if degree > 0 && shape.len() >= 2 {
        let n = shape.len() - 1;
        let m = binomial(n, degree);
        if m == shape[shape.len() - 1] && m >= 1 {
            return Ok((shape[..n].to_vec(), m));
        }
    }
    let n = shape.len();
    if binomial(n, degree) == 1 {
        return Ok((shape.to_vec(), 1));
    }
    Err(format!("{what}: data shape {shape:?} is not consistent with a degree-{degree} field"))
}

fn assemble(data: Vec<f64>, grid: Vec<usize>, degree: usize, variance: Variance, geom: &[Val], what: &str) -> Result<Val, String> {
    let n = grid.len();
    if n == 0 {
        return Err(format!("{what}: grid must have rank >= 1"));
    }
    if geom.len() < 3 || geom.len() > 4 {
        return Err(format!("{what}: expected lo, hi, bc [, metric]"));
    }
    let lo = as_axis_vec(&geom[0], n, &format!("{what} lo"))?;
    let hi = as_axis_vec(&geom[1], n, &format!("{what} hi"))?;
    let bcv = as_axis_vec(&geom[2], n, &format!("{what} bc"))?;
    let metric = match geom.get(3) {
        Some(m) => as_axis_vec(m, n, &format!("{what} metric"))?,
        None => vec![1.0; n],
    };
    if metric.iter().any(|&g| g == 0.0) {
        return Err(format!("{what}: metric entries must be nonzero"));
    }
    let bc: Vec<BC> = bcv.iter().map(|&b| if b == 0.0 { BC::Periodic } else { BC::Neumann }).collect();
    let spacing = axis_spacing(&lo, &hi, &grid, &bc);
    Ok(Val::Field(Arc::new(FieldVal { data: Arc::new(data), grid, spacing, lo, bc, metric, degree, variance })))
}

// ── constructors ────────────────────────────────────────────────────────────────

/// field(data, lo, hi, bc [, metric]) — a 0-form; or field(f, lo, hi, counts, bc
/// [, metric]) — sample a function at each grid point's physical coordinates.
pub fn field_ctor(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if matches!(vals.first(), Some(Val::Fn { .. }) | Some(Val::Builtin(_))) {
        return field_from_fn(vals, env);
    }
    if vals.len() < 4 || vals.len() > 5 {
        return Err("field(data, lo, hi, bc [, metric]) expects 4 or 5 args (or field(f, lo, hi, counts, bc [, metric]))".into());
    }
    let mut it = vals.into_iter();
    let (data, grid) = match it.next().unwrap() {
        Val::Tensor(t) => (compute::download(&t)?, t.shape.clone()),
        other => return Err(format!("field: data must be a real tensor, got {}", fmt_val(&other))),
    };
    let geom: Vec<Val> = it.collect();
    assemble(data, grid, 0, Variance::Form, &geom, "field")
}

fn field_from_fn(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() < 5 || vals.len() > 6 {
        return Err("field(f, lo, hi, counts, bc [, metric]) expects 5 or 6 args".into());
    }
    let mut it = vals.into_iter();
    let f = it.next().unwrap();
    let lo_v = it.next().unwrap();
    let hi_v = it.next().unwrap();
    let counts_v = it.next().unwrap();
    let rest: Vec<Val> = it.collect();
    let grid = as_count_vec(&counts_v)?;
    let n = grid.len();
    if n == 0 || grid.iter().any(|&c| c == 0) {
        return Err("field: counts must be positive and rank >= 1".into());
    }
    let lo = as_axis_vec(&lo_v, n, "field lo")?;
    let hi = as_axis_vec(&hi_v, n, "field hi")?;
    let bcv = as_axis_vec(&rest[0], n, "field bc")?;
    let bc: Vec<BC> = bcv.iter().map(|&b| if b == 0.0 { BC::Periodic } else { BC::Neumann }).collect();
    let spacing = axis_spacing(&lo, &hi, &grid, &bc);
    let total: usize = grid.iter().product();
    let mut data = Vec::with_capacity(total);
    let mut idx = vec![0usize; n];
    for _ in 0..total {
        let coords: Vec<Val> = (0..n).map(|a| Val::Num(lo[a] + idx[a] as f64 * spacing[a])).collect();
        let v = crate::interp::apply_val(f.clone(), coords, env)?;
        data.push(v.num("field: f must return a real number")?);
        for k in (0..n).rev() {
            idx[k] += 1;
            if idx[k] < grid[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    let mut geom = vec![lo_v, hi_v];
    geom.extend(rest);
    assemble(data, grid, 0, Variance::Form, &geom, "field")
}

// ── dispatch (forms.*) ───────────────────────────────────────────────────────────

pub fn dispatch(name: &str, vals: Vec<Val>) -> Result<Val, String> {
    match name {
        "d" => d(vals),
        "hodge" => hodge(vals),
        "wedge" => wedge(vals),
        "raise" => raise(vals),
        "lower" => lower(vals),
        "codiff" => codiff(vals),
        "laplace" => laplace(vals),
        "form" => form_ctor(vals),
        "vector" => vector_ctor(vals),
        "contract" => contract(vals),
        _ => Err(format!("forms: unknown member '{name}'")),
    }
}

fn form_ctor(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 5 || vals.len() > 6 {
        return Err("forms.form(data, degree, lo, hi, bc [, metric]) expects 5 or 6 args".into());
    }
    let mut it = vals.into_iter();
    let (data, shape) = match it.next().unwrap() {
        Val::Tensor(t) => (compute::download(&t)?, t.shape.clone()),
        other => return Err(format!("forms.form: data must be a real tensor, got {}", fmt_val(&other))),
    };
    let degree = it.next().unwrap().num("forms.form degree")? as usize;
    let (grid, _) = split_shape(&shape, degree, "forms.form")?;
    let geom: Vec<Val> = it.collect();
    assemble(data, grid, degree, Variance::Form, &geom, "forms.form")
}

fn vector_ctor(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 4 || vals.len() > 5 {
        return Err("forms.vector(data, lo, hi, bc [, metric]) expects 4 or 5 args".into());
    }
    let mut it = vals.into_iter();
    let (data, shape) = match it.next().unwrap() {
        Val::Tensor(t) => (compute::download(&t)?, t.shape.clone()),
        other => return Err(format!("forms.vector: data must be a real tensor, got {}", fmt_val(&other))),
    };
    let (grid, _) = split_shape(&shape, 1, "forms.vector")?;
    let geom: Vec<Val> = it.collect();
    assemble(data, grid, 1, Variance::Vector, &geom, "forms.vector")
}

// ── operators ────────────────────────────────────────────────────────────────────

fn d(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 {
        return Err("forms.d(f) expects 1 arg".into());
    }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.d")?;
    let n = f.grid.len();
    let k = f.degree;
    if k + 1 > n {
        return Err(format!("forms.d: cannot differentiate a {k}-form on a {n}-D grid"));
    }
    let in_sub = subsets(n, k);
    let out_sub = subsets(n, k + 1);
    let (nc_in, nc_out) = (in_sub.len(), out_sub.len());
    let gt = grid_total(&f.grid);
    let mut out = vec![0.0; gt * nc_out];
    for (out_c, jset) in out_sub.iter().enumerate() {
        for (p, &j) in jset.iter().enumerate() {
            let mut iset = jset.clone();
            iset.remove(p);
            let in_c = subset_index(&in_sub, &iset);
            let comp = component(&f, in_c, nc_in);
            let dcomp = partial(&comp, &f.grid, f.spacing[j], j, f.bc[j]);
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            for gp in 0..gt {
                out[gp * nc_out + out_c] += sign * dcomp[gp];
            }
        }
    }
    Ok(rebuild(&f, out, k + 1, f.variance))
}

fn hodge(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 {
        return Err("forms.hodge(f) expects 1 arg".into());
    }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.hodge")?;
    let n = f.grid.len();
    let k = f.degree;
    let in_sub = subsets(n, k);
    let out_sub = subsets(n, n - k);
    let (nc_in, nc_out) = (in_sub.len(), out_sub.len());
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
        for gp in 0..gt {
            out[gp * nc_out + out_c] = coeff * comp[gp];
        }
    }
    Ok(rebuild(&f, out, n - k, f.variance))
}

fn wedge(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 2 {
        return Err("forms.wedge(a, b) expects 2 args".into());
    }
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
            if iset.iter().any(|x| jset.contains(x)) {
                continue;
            }
            let mut concat = iset.clone();
            concat.extend_from_slice(jset);
            let eps = perm_sign(&concat) as f64;
            let mut kset = concat.clone();
            kset.sort_unstable();
            let out_c = subset_index(&out_sub, &kset);
            let bcomp = component(&b, ib, nc_b);
            for gp in 0..gt {
                out[gp * nc_out + out_c] += eps * acomp[gp] * bcomp[gp];
            }
        }
    }
    Ok(rebuild(&a, out, ka + kb, a.variance))
}

fn lower(vals: Vec<Val>) -> Result<Val, String> {
    musical(vals, "forms.lower", false)
}
fn raise(vals: Vec<Val>) -> Result<Val, String> {
    musical(vals, "forms.raise", true)
}

fn musical(vals: Vec<Val>, what: &str, up: bool) -> Result<Val, String> {
    if vals.len() != 1 {
        return Err(format!("{what}(f) expects 1 arg"));
    }
    let f = as_field(vals.into_iter().next().unwrap(), what)?;
    let n = f.grid.len();
    let sub = subsets(n, f.degree);
    let nc = sub.len();
    let gt = grid_total(&f.grid);
    let mut out = f.data.as_ref().clone();
    for (c, iset) in sub.iter().enumerate() {
        let scale: f64 = iset.iter().map(|&i| if up { 1.0 / f.metric[i] } else { f.metric[i] }).product();
        for gp in 0..gt {
            out[gp * nc + c] *= scale;
        }
    }
    let variance = if up { Variance::Vector } else { Variance::Form };
    Ok(rebuild(&f, out, f.degree, variance))
}

fn codiff(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 {
        return Err("forms.codiff(f) expects 1 arg".into());
    }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.codiff")?;
    let (n, k) = (f.grid.len(), f.degree);
    if k == 0 {
        return Err("forms.codiff: the codifferential of a 0-form is 0 (degree -1)".into());
    }
    let step1 = hodge(vec![Val::Field(f.clone())])?;
    let step2 = d(vec![step1])?;
    let step3 = hodge(vec![step2])?;
    let sign = if (n * (k + 1) + 1) % 2 == 0 { 1.0 } else { -1.0 };
    match step3 {
        Val::Field(g) => {
            let scaled: Vec<f64> = g.data.iter().map(|x| sign * x).collect();
            Ok(rebuild(&g, scaled, g.degree, g.variance))
        }
        other => Ok(other),
    }
}

fn laplace(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 {
        return Err("forms.laplace(f) expects 1 arg".into());
    }
    let f = as_field(vals.into_iter().next().unwrap(), "forms.laplace")?;
    let k = f.degree;
    let dd = d(vec![Val::Field(f.clone())])?;
    let term_dd = codiff(vec![dd])?;
    if k == 0 {
        return Ok(term_dd);
    }
    let cd = codiff(vec![Val::Field(f.clone())])?;
    let term_cd = d(vec![cd])?;
    let a = as_field(term_dd, "forms.laplace")?;
    let b = as_field(term_cd, "forms.laplace")?;
    let sum: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
    Ok(rebuild(&a, sum, a.degree, a.variance))
}

fn contract(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 2 {
        return Err("forms.contract(X, w) expects 2 args".into());
    }
    let mut it = vals.into_iter();
    let x = as_field(it.next().unwrap(), "forms.contract")?;
    let w = as_field(it.next().unwrap(), "forms.contract")?;
    if x.degree != 1 || x.variance != Variance::Vector {
        return Err("forms.contract: the first argument must be a vector field (forms.vector or forms.raise)".into());
    }
    if w.variance != Variance::Form {
        return Err("forms.contract: the second argument must be a form (covariant)".into());
    }
    if x.grid != w.grid || x.spacing != w.spacing || x.bc != w.bc || x.metric != w.metric {
        return Err("forms.contract: operands must share the same grid geometry".into());
    }
    let n = w.grid.len();
    let k = w.degree;
    if k == 0 {
        return Err("forms.contract: cannot contract a vector into a 0-form".into());
    }
    let in_sub = subsets(n, k);
    let out_sub = subsets(n, k - 1);
    let (nc_in, nc_out) = (in_sub.len(), out_sub.len());
    let gt = grid_total(&w.grid);
    let xcomp: Vec<Vec<f64>> = (0..n).map(|i| component(&x, i, n)).collect();
    let mut out = vec![0.0; gt * nc_out];
    for (out_c, jset) in out_sub.iter().enumerate() {
        for i in 0..n {
            if jset.contains(&i) {
                continue;
            }
            let mut iset = jset.clone();
            iset.push(i);
            iset.sort_unstable();
            let p = iset.iter().position(|&v| v == i).unwrap();
            let in_c = subset_index(&in_sub, &iset);
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            let wi = component(&w, in_c, nc_in);
            for gp in 0..gt {
                out[gp * nc_out + out_c] += sign * xcomp[i][gp] * wi[gp];
            }
        }
    }
    Ok(rebuild(&w, out, k - 1, Variance::Form))
}

// ── field arithmetic (host, component-wise) ─────────────────────────────────────

fn scalar_op(op: &Op, a: f64, b: f64) -> Result<f64, String> {
    Ok(match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        Op::Pow => a.powf(b),
        _ => return Err("field op: only + - * / ^ are defined on fields".into()),
    })
}

/// `field ⊕ field` (same geometry) or `field ⊕ scalar` — component-wise, preserving
/// geometry/degree/variance.
pub fn field_binop(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    let meta: Arc<FieldVal> = match (&lv, &rv) {
        (Val::Field(a), _) => a.clone(),
        (_, Val::Field(b)) => b.clone(),
        _ => unreachable!(),
    };
    let data: Vec<f64> = match (lv, rv) {
        (Val::Field(a), Val::Field(b)) => {
            if a.grid != b.grid || a.degree != b.degree || a.variance != b.variance || a.spacing != b.spacing || a.bc != b.bc || a.metric != b.metric {
                return Err("field op: incompatible fields (grid, degree, variance, spacing, bc, metric must match)".into());
            }
            a.data.iter().zip(b.data.iter()).map(|(&x, &y)| scalar_op(op, x, y)).collect::<Result<_, _>>()?
        }
        (Val::Field(a), Val::Num(s)) => a.data.iter().map(|&x| scalar_op(op, x, s)).collect::<Result<_, _>>()?,
        (Val::Num(s), Val::Field(b)) => b.data.iter().map(|&x| scalar_op(op, s, x)).collect::<Result<_, _>>()?,
        (other_l, other_r) => {
            return Err(format!("field op: unsupported operands {} and {}", fmt_val(&other_l), fmt_val(&other_r)));
        }
    };
    Ok(with_data(&meta, data))
}

// ── bridge to tensors + display ─────────────────────────────────────────────────

/// A field's component data as (host data, shape) — the component axis is dropped
/// for a single-component field, so a 0-form extracts as an ordinary grid tensor.
pub fn field_tensor_shape(f: &FieldVal) -> (Vec<f64>, Vec<usize>) {
    let ncomp = f.ncomp();
    let shape = if ncomp == 1 {
        f.grid.clone()
    } else {
        let mut s = f.grid.clone();
        s.push(ncomp);
        s
    };
    (f.data.as_ref().clone(), shape)
}

pub fn format_field(f: &FieldVal) -> String {
    let kind = match f.variance {
        Variance::Form => "form",
        Variance::Vector => "vector field",
    };
    let dims = f.grid.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
    let extent = (0..f.grid.len())
        .map(|a| {
            let (lo, dx, n) = (f.lo[a], f.spacing[a], f.grid[a]);
            let cells = match f.bc[a] { BC::Periodic => n, BC::Neumann => n.saturating_sub(1) };
            format!("[{}, {}]", fmt_f(lo), fmt_f(lo + dx * cells as f64))
        })
        .collect::<Vec<_>>()
        .join("×");
    let bc = if f.bc.iter().all(|&b| b == BC::Periodic) {
        "periodic"
    } else if f.bc.iter().all(|&b| b == BC::Neumann) {
        "neumann"
    } else {
        "mixed-bc"
    };
    let metric = if f.metric.iter().all(|&g| g == 1.0) {
        String::new()
    } else {
        format!(" metric({})", f.metric.iter().map(|&g| fmt_f(g)).collect::<Vec<_>>().join(", "))
    };
    let (data, shape) = field_tensor_shape(f);
    format!("{}-{} [{}] on {} {}{}\n{}", f.degree, kind, dims, extent, bc, metric, crate::value::fmt_host_tensor(&data, &shape))
}
