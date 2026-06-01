// ops — differential operators on gridded fields (PDE_ERGONOMICS §1, §3).
//
// All finite-difference operators take the physical grid spacing `dx` as a
// REQUIRED argument (never defaulted to 1) — the missing-dx mistake is exactly
// what broke examples/fluid2D.math. The periodic forms reuse the same wrap-around
// stencil as the flat `roll` builtin; the spectral forms reuse fft_axis_inplace.
//
// Domain convention for the spectral solvers: a periodic box whose length along
// axis a is N_a * dx, so integer FFT frequencies map to wavenumbers
// k = 2*pi*freq / (N_a*dx).
use crate::eval::{Val, TData, FieldVal, BC, Variance, as_complex_tensor, unravel, fft_axis_inplace, fmt_val};
use std::f64::consts::PI;
use std::sync::Arc;

pub const NAMES: &[&str] = &[
    "grad", "div", "curl", "lap", "poisson", "invlap", "specgrad",
];

pub fn members() -> std::collections::HashMap<String, Val> {
    let mut m: std::collections::HashMap<String, Val> =
        NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect();
    // Named BC sentinels (mathlang has no string type):
    //   ops.lap(T, dx, ops.neumann)
    m.insert("periodic".into(), Val::Num(0.0));
    m.insert("neumann".into(),  Val::Num(1.0));
    m
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn real_tensor(v: Val, ctx: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Tensor { data, shape } => Ok((data.into_vec(), shape)),
        other => Err(format!("{ctx}: expected a real tensor, got {}", fmt_val(&other))),
    }
}

/// Periodic shift: out[ax] = in[ax - n] (mod dim) along `axis`. Matches `roll`.
fn roll_axis(data: &[f64], shape: &[usize], n: i64, axis: usize) -> Vec<f64> {
    let total: usize = shape.iter().product();
    let stride: usize = shape[axis + 1..].iter().product();
    let dim = shape[axis] as i64;
    let mut out = vec![0.0; total];
    for o in 0..total {
        let ax = ((o / stride) % shape[axis]) as i64;
        let in_ax = (ax - n).rem_euclid(dim);
        let in_flat = o as i64 + (in_ax - ax) * stride as i64;
        out[o] = data[in_flat as usize];
    }
    out
}

/// Edge-clamped shift (Neumann/no-flux): out[ax] = in[clamp(ax - n, 0, dim-1)].
fn clamp_axis(data: &[f64], shape: &[usize], n: i64, axis: usize) -> Vec<f64> {
    let total: usize = shape.iter().product();
    let stride: usize = shape[axis + 1..].iter().product();
    let dim = shape[axis] as i64;
    let mut out = vec![0.0; total];
    for o in 0..total {
        let ax = ((o / stride) % shape[axis]) as i64;
        let in_ax = (ax - n).clamp(0, dim - 1);
        let in_flat = o as i64 + (in_ax - ax) * stride as i64;
        out[o] = data[in_flat as usize];
    }
    out
}

/// Central first derivative along `axis`: (f[+1] - f[-1]) / (2 dx). Periodic.
fn central(data: &[f64], shape: &[usize], dx: f64, axis: usize) -> Vec<f64> {
    let fwd = roll_axis(data, shape, -1, axis);
    let bwd = roll_axis(data, shape, 1, axis);
    fwd.iter().zip(bwd.iter()).map(|(&a, &b)| (a - b) / (2.0 * dx)).collect()
}

fn kfreq(m: usize, n: usize) -> f64 {
    if 2 * m < n { m as f64 } else { m as f64 - n as f64 }
}

fn fftn(re: &mut [f64], im: &mut [f64], shape: &[usize], forward: bool) {
    for a in 0..shape.len() {
        fft_axis_inplace(re, im, shape, a, forward);
    }
}

// ── dispatch ────────────────────────────────────────────────────────────────────

pub fn dispatch(name: &str, vals: Vec<Val>, _env: &crate::eval::Env) -> Result<Val, String> {
    // Field-polymorphic forms: when called on a `field`, the per-axis spacing and
    // boundary conditions come from the field itself (no dx argument) and the
    // result is a field, so pipelines stay inside the form algebra. Plain
    // (tensor, dx [,…]) calls keep the original behaviour below.
    if matches!(vals.first(), Some(Val::Field(_))) {
        return field_dispatch(name, vals);
    }
    match name {
        "grad"     => grad(vals),
        "specgrad" => specgrad(vals),
        "div"      => div(vals),
        "curl"     => curl(vals),
        "lap"      => lap(vals),
        "poisson" | "invlap" => poisson(vals),
        _ => Err(format!("ops: unknown member '{name}'")),
    }
}

// grad(T, dx [, axis]) — central difference.
//   with axis: derivative along that axis, same shape as T.
//   without:   stacks the per-axis derivatives along a new trailing axis,
//              output shape = shape(T) ++ [ndim].
fn grad(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("ops.grad(T, dx [, axis]) expects 2 or 3 args".into());
    }
    let mut it = vals.into_iter();
    let (data, shape) = real_tensor(it.next().unwrap(), "ops.grad")?;
    let dx = it.next().unwrap().num("ops.grad dx")?;
    let ndim = shape.len();
    if ndim == 0 { return Err("ops.grad: need a tensor of rank >= 1".into()); }
    if let Some(axv) = it.next() {
        let axis = axv.num("ops.grad axis")? as usize;
        if axis >= ndim { return Err(format!("ops.grad: axis {axis} out of range for rank-{ndim} tensor")); }
        return Ok(Val::Tensor { data: TData::new(central(&data, &shape, dx, axis)), shape });
    }
    stack_per_axis(&data, &shape, |d, s, a| central(d, s, dx, a))
}

// specgrad(T, dx [, axis]) — spectral derivative via i*k.
fn specgrad(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("ops.specgrad(T, dx [, axis]) expects 2 or 3 args".into());
    }
    let mut it = vals.into_iter();
    let (data, shape) = real_tensor(it.next().unwrap(), "ops.specgrad")?;
    let dx = it.next().unwrap().num("ops.specgrad dx")?;
    let ndim = shape.len();
    if ndim == 0 { return Err("ops.specgrad: need a tensor of rank >= 1".into()); }
    if let Some(axv) = it.next() {
        let axis = axv.num("ops.specgrad axis")? as usize;
        if axis >= ndim { return Err(format!("ops.specgrad: axis {axis} out of range for rank-{ndim} tensor")); }
        return Ok(spec_deriv(&data, &shape, dx, axis));
    }
    // all axes → trailing component axis. spec_deriv returns a Val; reduce to data.
    stack_per_axis(&data, &shape, |d, s, a| spec_deriv_vec(d, s, dx, a))
}

fn spec_deriv_vec(data: &[f64], shape: &[usize], dx: f64, axis: usize) -> Vec<f64> {
    match spec_deriv(data, shape, dx, axis) {
        Val::Tensor { data, .. }           => data.into_vec(),
        Val::ComplexTensor { re, .. }       => re.into_vec(),
        _ => vec![0.0; data.len()],
    }
}

fn spec_deriv(data: &[f64], shape: &[usize], dx: f64, axis: usize) -> Val {
    let mut re = data.to_vec();
    let mut im = vec![0.0; data.len()];
    fftn(&mut re, &mut im, shape, true);
    let n_ax = shape[axis];
    for p in 0..re.len() {
        let multi = unravel(p, shape);
        let k = kfreq(multi[axis], n_ax) * 2.0 * PI / (n_ax as f64 * dx);
        // multiply by i*k: (a+bi)*(ik) = -k*b + i*k*a
        let nr = -k * im[p];
        let ni = k * re[p];
        re[p] = nr;
        im[p] = ni;
    }
    fftn(&mut re, &mut im, shape, false);
    // The derivative of a real field is real; discard FFT roundoff in the imag part.
    let _ = im;
    Val::Tensor { data: TData::new(re), shape: shape.to_vec() }
}

// div(V, dx) — divergence of a vector field. V shape = base ++ [ndim],
// the trailing axis indexing the components; div = sum_a dV[..,a]/dx_a.
fn div(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 2 { return Err("ops.div(V, dx) expects 2 args".into()); }
    let mut it = vals.into_iter();
    let (data, shape) = real_tensor(it.next().unwrap(), "ops.div")?;
    let dx = it.next().unwrap().num("ops.div dx")?;
    let (base, comps) = split_trailing(&shape)?;
    if comps != base.len() {
        return Err(format!("ops.div: vector field has {comps} components but base is {}-D", base.len()));
    }
    let base_total: usize = base.iter().product();
    let mut acc = vec![0.0; base_total];
    for a in 0..comps {
        let comp = extract_component(&data, base_total, comps, a);
        let d = central(&comp, &base, dx, a);
        for p in 0..base_total { acc[p] += d[p]; }
    }
    Ok(Val::Tensor { data: TData::new(acc), shape: base })
}

// curl(V, dx) — 2-D scalar curl: dV_y/dx - dV_x/dy. V shape = (r,c,2).
fn curl(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 2 { return Err("ops.curl(V, dx) expects 2 args".into()); }
    let mut it = vals.into_iter();
    let (data, shape) = real_tensor(it.next().unwrap(), "ops.curl")?;
    let dx = it.next().unwrap().num("ops.curl dx")?;
    let (base, comps) = split_trailing(&shape)?;
    if base.len() != 2 || comps != 2 {
        return Err("ops.curl: only the 2-D scalar curl is supported (V shape [r, c, 2])".into());
    }
    let base_total: usize = base.iter().product();
    let vx = extract_component(&data, base_total, 2, 0);
    let vy = extract_component(&data, base_total, 2, 1);
    let dvy = central(&vy, &base, dx, 0);
    let dvx = central(&vx, &base, dx, 1);
    let out: Vec<f64> = (0..base_total).map(|p| dvy[p] - dvx[p]).collect();
    Ok(Val::Tensor { data: TData::new(out), shape: base })
}

// lap(T, dx [, bc]) — Laplacian. bc: 0=periodic (default), 1=neumann (no-flux).
fn lap(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("ops.lap(T, dx [, bc]) expects 2 or 3 args".into());
    }
    let mut it = vals.into_iter();
    let (data, shape) = real_tensor(it.next().unwrap(), "ops.lap")?;
    let dx = it.next().unwrap().num("ops.lap dx")?;
    let neumann = match it.next() {
        Some(v) => v.num("ops.lap bc")? != 0.0,
        None => false,
    };
    let ndim = shape.len();
    if ndim == 0 { return Err("ops.lap: need a tensor of rank >= 1".into()); }
    let total = data.len();
    let inv = 1.0 / (dx * dx);
    let mut out = vec![0.0; total];
    for a in 0..ndim {
        let (plus, minus) = if neumann {
            (clamp_axis(&data, &shape, -1, a), clamp_axis(&data, &shape, 1, a))
        } else {
            (roll_axis(&data, &shape, -1, a), roll_axis(&data, &shape, 1, a))
        };
        for p in 0..total { out[p] += plus[p] + minus[p] - 2.0 * data[p]; }
    }
    for p in 0..total { out[p] *= inv; }
    Ok(Val::Tensor { data: TData::new(out), shape })
}

// poisson(rhs, dx) / invlap(T, dx) — spectral solve of  ∇²u = rhs  on the
// periodic box, returning the zero-mean solution (k=0 mode set to 0).
fn poisson(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 2 { return Err("ops.poisson(rhs, dx) expects 2 args".into()); }
    let mut it = vals.into_iter();
    let (mut re, mut im, shape) = as_complex_tensor(it.next().unwrap())
        .map_err(|_| "ops.poisson: first arg must be a tensor".to_string())?;
    let dx = it.next().unwrap().num("ops.poisson dx")?;
    let ndim = shape.len();
    if ndim == 0 { return Err("ops.poisson: need a tensor of rank >= 1".into()); }
    fftn(&mut re, &mut im, &shape, true);
    for p in 0..re.len() {
        let multi = unravel(p, &shape);
        let mut k2 = 0.0;
        for a in 0..ndim {
            let k = kfreq(multi[a], shape[a]) * 2.0 * PI / (shape[a] as f64 * dx);
            k2 += k * k;
        }
        if k2 == 0.0 {
            re[p] = 0.0; im[p] = 0.0;     // zero-mean solution
        } else {
            // ∇²u = rhs  ⇒  -k² û = r̂  ⇒  û = -r̂/k²
            re[p] = -re[p] / k2;
            im[p] = -im[p] / k2;
        }
    }
    fftn(&mut re, &mut im, &shape, false);
    // The potential for a real source is real; discard FFT roundoff in the imag part.
    let _ = im;
    Ok(Val::Tensor { data: TData::new(re), shape })
}

// ── field-polymorphic forms ─────────────────────────────────────────────────────

fn field_dispatch(name: &str, vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 1 {
        return Err(format!("ops.{name}(field): dx/bc come from the field, so it takes exactly 1 argument"));
    }
    let f = match vals.into_iter().next().unwrap() { Val::Field(f) => f, _ => unreachable!() };
    match name {
        "grad"               => field_grad(&f),
        "specgrad"           => field_specgrad(&f),
        "div"                => field_div(&f),
        "curl"               => field_curl(&f),
        "lap"                => field_lap(&f),
        "poisson" | "invlap" => field_poisson(&f),
        _ => Err(format!("ops: unknown member '{name}'")),
    }
}

fn gtot(grid: &[usize]) -> usize { grid.iter().product::<usize>().max(1) }

/// Component c (component-fastest layout) of a field as a flat grid array.
fn fcomp(data: &[f64], gt: usize, ncomp: usize, c: usize) -> Vec<f64> {
    (0..gt).map(|p| data[p * ncomp + c]).collect()
}

/// Central first derivative of a grid array along `axis`, honouring the BC.
fn d1(data: &[f64], grid: &[usize], dx: f64, axis: usize, bc: BC) -> Vec<f64> {
    let (fwd, bwd) = match bc {
        BC::Periodic => (roll_axis(data, grid, -1, axis), roll_axis(data, grid, 1, axis)),
        BC::Neumann  => (clamp_axis(data, grid, -1, axis), clamp_axis(data, grid, 1, axis)),
    };
    fwd.iter().zip(bwd.iter()).map(|(&a, &b)| (a - b) / (2.0 * dx)).collect()
}

fn mkfield(f: &FieldVal, data: Vec<f64>, degree: usize, variance: Variance) -> Val {
    Val::Field(Arc::new(FieldVal {
        data: TData::new(data), degree, variance,
        grid: f.grid.clone(), spacing: f.spacing.clone(), lo: f.lo.clone(),
        bc: f.bc.clone(), metric: f.metric.clone(),
    }))
}

fn require_scalar(f: &FieldVal, what: &str) -> Result<(), String> {
    if f.degree != 0 {
        return Err(format!("ops.{what}: expected a scalar (0-form) field, got degree {}", f.degree));
    }
    Ok(())
}

// grad(0-form field) -> 1-form field (compact central differences, per-axis dx/BC).
fn field_grad(f: &FieldVal) -> Result<Val, String> {
    require_scalar(f, "grad")?;
    let (n, gt) = (f.grid.len(), gtot(&f.grid));
    let mut out = vec![0.0; gt * n];
    for a in 0..n {
        let d = d1(&f.data, &f.grid, f.spacing[a], a, f.bc[a]);
        for p in 0..gt { out[p * n + a] = d[p]; }
    }
    Ok(mkfield(f, out, 1, f.variance))
}

// lap(field) -> field of the same degree: the compact 3-point Laplacian applied
// componentwise (∇², metric-free), distinct from forms.laplace's wide δd stencil.
fn field_lap(f: &FieldVal) -> Result<Val, String> {
    let (n, gt) = (f.grid.len(), gtot(&f.grid));
    let ncomp = f.data.len() / gt;
    let mut out = vec![0.0; f.data.len()];
    for c in 0..ncomp {
        let comp = fcomp(&f.data, gt, ncomp, c);
        let mut acc = vec![0.0; gt];
        for a in 0..n {
            let inv = 1.0 / (f.spacing[a] * f.spacing[a]);
            let (plus, minus) = match f.bc[a] {
                BC::Periodic => (roll_axis(&comp, &f.grid, -1, a), roll_axis(&comp, &f.grid, 1, a)),
                BC::Neumann  => (clamp_axis(&comp, &f.grid, -1, a), clamp_axis(&comp, &f.grid, 1, a)),
            };
            for p in 0..gt { acc[p] += (plus[p] + minus[p] - 2.0 * comp[p]) * inv; }
        }
        for p in 0..gt { out[p * ncomp + c] = acc[p]; }
    }
    Ok(mkfield(f, out, f.degree, f.variance))
}

// div(1-form/vector field with n components) -> 0-form field: Σ_a ∂_a V_a.
fn field_div(f: &FieldVal) -> Result<Val, String> {
    let (n, gt) = (f.grid.len(), gtot(&f.grid));
    let ncomp = f.data.len() / gt;
    if ncomp != n {
        return Err(format!("ops.div: field has {ncomp} components but the grid is {n}-D"));
    }
    let mut acc = vec![0.0; gt];
    for a in 0..n {
        let comp = fcomp(&f.data, gt, ncomp, a);
        let d = d1(&comp, &f.grid, f.spacing[a], a, f.bc[a]);
        for p in 0..gt { acc[p] += d[p]; }
    }
    Ok(mkfield(f, acc, 0, f.variance))
}

// curl(2-component field on a 2-D grid) -> 0-form field: ∂_0 V_1 - ∂_1 V_0.
fn field_curl(f: &FieldVal) -> Result<Val, String> {
    let (n, gt) = (f.grid.len(), gtot(&f.grid));
    let ncomp = f.data.len() / gt;
    if n != 2 || ncomp != 2 {
        return Err("ops.curl: only the 2-D scalar curl is supported (a 2-component field on a 2-D grid)".into());
    }
    let v0 = fcomp(&f.data, gt, 2, 0);
    let v1 = fcomp(&f.data, gt, 2, 1);
    let dv1 = d1(&v1, &f.grid, f.spacing[0], 0, f.bc[0]);
    let dv0 = d1(&v0, &f.grid, f.spacing[1], 1, f.bc[1]);
    let out: Vec<f64> = (0..gt).map(|p| dv1[p] - dv0[p]).collect();
    Ok(mkfield(f, out, 0, f.variance))
}

/// All-periodic check for the spectral field operators.
fn require_periodic(f: &FieldVal, what: &str) -> Result<(), String> {
    if f.bc.iter().any(|&b| b != BC::Periodic) {
        return Err(format!("ops.{what}: spectral operators require a periodic field"));
    }
    Ok(())
}

// specgrad(0-form field) -> 1-form field: spectral derivative via i*k (periodic).
fn field_specgrad(f: &FieldVal) -> Result<Val, String> {
    require_scalar(f, "specgrad")?;
    require_periodic(f, "specgrad")?;
    let (n, gt) = (f.grid.len(), gtot(&f.grid));
    let mut out = vec![0.0; gt * n];
    for a in 0..n {
        let mut re = f.data.to_vec();
        let mut im = vec![0.0; gt];
        fftn(&mut re, &mut im, &f.grid, true);
        for p in 0..gt {
            let multi = unravel(p, &f.grid);
            let k = kfreq(multi[a], f.grid[a]) * 2.0 * PI / (f.grid[a] as f64 * f.spacing[a]);
            let (nr, ni) = (-k * im[p], k * re[p]);
            re[p] = nr; im[p] = ni;
        }
        fftn(&mut re, &mut im, &f.grid, false);
        for p in 0..gt { out[p * n + a] = re[p]; }
    }
    Ok(mkfield(f, out, 1, f.variance))
}

// poisson/invlap(0-form field) -> 0-form field: spectral ∇²u = rhs (periodic, zero-mean).
fn field_poisson(f: &FieldVal) -> Result<Val, String> {
    require_scalar(f, "poisson")?;
    require_periodic(f, "poisson")?;
    let gt = gtot(&f.grid);
    let n = f.grid.len();
    let mut re = f.data.to_vec();
    let mut im = vec![0.0; gt];
    fftn(&mut re, &mut im, &f.grid, true);
    for p in 0..gt {
        let multi = unravel(p, &f.grid);
        let mut k2 = 0.0;
        for a in 0..n {
            let k = kfreq(multi[a], f.grid[a]) * 2.0 * PI / (f.grid[a] as f64 * f.spacing[a]);
            k2 += k * k;
        }
        if k2 == 0.0 { re[p] = 0.0; im[p] = 0.0; }
        else { re[p] = -re[p] / k2; im[p] = -im[p] / k2; }
    }
    fftn(&mut re, &mut im, &f.grid, false);
    Ok(mkfield(f, re, 0, f.variance))
}

// ── shape utilities ─────────────────────────────────────────────────────────────

/// (base_shape, trailing_size) for a vector-field tensor base ++ [comps].
fn split_trailing(shape: &[usize]) -> Result<(Vec<usize>, usize), String> {
    if shape.len() < 2 {
        return Err("expected a vector field with a trailing component axis (rank >= 2)".into());
    }
    let comps = *shape.last().unwrap();
    Ok((shape[..shape.len() - 1].to_vec(), comps))
}

/// Component `a` of a field laid out base ++ [comps] (trailing axis contiguous).
fn extract_component(data: &[f64], base_total: usize, comps: usize, a: usize) -> Vec<f64> {
    (0..base_total).map(|p| data[p * comps + a]).collect()
}

/// Apply `f(data, shape, axis)` for each axis and stack the results along a new
/// trailing component axis → output shape = shape ++ [ndim].
fn stack_per_axis<F>(data: &[f64], shape: &[usize], mut f: F) -> Result<Val, String>
where F: FnMut(&[f64], &[usize], usize) -> Vec<f64> {
    let ndim = shape.len();
    let total = data.len();
    let mut out = vec![0.0; total * ndim];
    for a in 0..ndim {
        let d = f(data, shape, a);
        for p in 0..total { out[p * ndim + a] = d[p]; }
    }
    let mut out_shape = shape.to_vec();
    out_shape.push(ndim);
    Ok(Val::Tensor { data: TData::new(out), shape: out_shape })
}
