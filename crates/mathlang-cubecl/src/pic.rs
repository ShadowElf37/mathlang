//! PIC — particle/grid coupling for particle-in-cell methods (host-side).
//!
//! Three operations, all exact transposes of each other (same shape function S):
//!
//!   * scatter (deposition):    particles → grid field   (charge ρ, current J)
//!       ρ_g = Σ_i w_i S(x_g − x_i)
//!   * gather  (interpolation): grid field → particles   (force sampling)
//!       E(x_i) = Σ_g E_g S(x_g − x_i)
//!   * gathergrad:              gradient of the interpolation kernel at particle positions
//!       ∂/∂x_i of (CIC/TSC interp of field) — the variational force for energies that
//!       are a function of the deposited field; exact transpose of scatter's positional
//!       derivative, so Verlet conserves H exactly for self-gravitating / barotropic gases.
//!
//! Shape functions: nearest-grid-point (pic.ngp, 0th order), cloud-in-cell
//! (pic.cic, linear — the default), triangular-shaped-cloud (pic.tsc, quadratic).
//! Boundary handling follows the field's per-axis BC: periodic wraps node indices;
//! neumann clamps them to the edge. Scatter and gather use the identical (node, weight)
//! stencil, so adjointness holds by construction: ⟨gather(f,X), w⟩ = ⟨f, scatter(X,w)⟩.

use crate::compute;
use crate::field::{with_data, BC, FieldVal};
use crate::interp::Env;
use crate::value::{fmt_val, Val};
use std::collections::HashMap;
use std::sync::Arc;

pub fn members() -> HashMap<String, Val> {
    let mut m = HashMap::new();
    for name in ["scatter", "gather", "gathergrad"] {
        m.insert(name.to_string(), Val::Builtin(format!("pic.{name}")));
    }
    m.insert("ngp".into(), Val::Num(0.0));
    m.insert("cic".into(), Val::Num(1.0));
    m.insert("tsc".into(), Val::Num(2.0));
    m
}

pub fn dispatch(name: &str, args: Vec<Val>, env: &Env) -> Result<Val, String> {
    match name {
        "scatter"    => scatter(args),
        "gather"     => gather(args, env),
        "gathergrad" => gathergrad(args, env),
        _ => Err(format!("pic: unknown member '{name}'")),
    }
}

// ── argument coercion ─────────────────────────────────────────────────────────

fn as_field(v: Val, what: &str) -> Result<Arc<FieldVal>, String> {
    match v {
        Val::Field(f) => Ok(f),
        other => Err(format!("{what}: expected a field, got {}", fmt_val(&other))),
    }
}

fn kernel_order(v: Option<&Val>, what: &str) -> Result<usize, String> {
    match v {
        None => Ok(1), // pic.cic default
        Some(Val::Num(x)) => {
            let o = x.round() as i64;
            if (x - o as f64).abs() > 1e-9 || !(0..=2).contains(&o) {
                return Err(format!("{what}: kernel must be pic.ngp (0), pic.cic (1), or pic.tsc (2)"));
            }
            Ok(o as usize)
        }
        Some(other) => Err(format!("{what}: kernel must be pic.ngp/pic.cic/pic.tsc, got {}", fmt_val(other))),
    }
}

/// Materialise a Val as host (data, shape). Downloads device tensors; wraps scalars.
fn host_data_shape(v: &Val, what: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Num(x) => Ok((vec![*x], vec![1])),
        Val::Tensor(t) => Ok((compute::download(t)?, t.shape.clone())),
        other => Err(format!("{what}: expected a tensor or scalar, got {}", fmt_val(other))),
    }
}

/// Parse particle positions → (P, flat row-major [P*n]).
///
/// Accepts:
///  - `[P, n]` tensor: P particles in n-D space
///  - `[P]` tensor when n==1: P scalar positions
///  - `[n]` tensor: a single position (P=1)
///  - scalar when n==1: one particle (P=1)
fn parse_positions(v: &Val, n: usize, what: &str) -> Result<(usize, Vec<f64>), String> {
    let (data, shape) = host_data_shape(v, what)?;
    if n == 1 {
        if shape == [1] || (shape.len() == 1) {
            return Ok((shape[0], data));
        }
    }
    if shape.len() == 2 && shape[1] == n {
        return Ok((shape[0], data));
    }
    if shape.len() == 1 && shape[0] == n {
        return Ok((1, data));
    }
    Err(format!("{what}: positions must be shape [P,{n}] (or [P] when n=1), got {shape:?}"))
}

/// Parse per-particle weights → flat row-major [P*ncomp].
///
/// Accepts `[P]` for scalar weight (ncomp==1) or `[P, ncomp]` for vector weight.
fn parse_weights(v: &Val, p: usize, ncomp: usize, what: &str) -> Result<Vec<f64>, String> {
    let (data, shape) = host_data_shape(v, what)?;
    if ncomp == 1 && shape.len() == 1 && shape[0] == p {
        return Ok(data);
    }
    if ncomp == 1 && shape == [1] && p == 1 {
        return Ok(data);
    }
    if shape.len() == 2 && shape[0] == p && shape[1] == ncomp {
        return Ok(data);
    }
    Err(format!("{what}: weights must be shape [{p}] (or [{p},{ncomp}]), got {shape:?}"))
}

// ── shape functions ────────────────────────────────────────────────────────────

fn strides(grid: &[usize]) -> Vec<usize> {
    let n = grid.len();
    let mut s = vec![1usize; n];
    for a in (0..n.saturating_sub(1)).rev() {
        s[a] = s[a + 1] * grid[a + 1];
    }
    s
}

/// 1-D shape-function stencil at dimensionless coordinate `u = (x - lo) / dx`.
/// Returns (grid node index, weight) after applying the boundary condition.
fn axis_stencil(u: f64, dim: usize, bc: BC, order: usize) -> Vec<(usize, f64)> {
    let raw: Vec<(i64, f64)> = match order {
        0 => vec![(u.round() as i64, 1.0)],
        1 => {
            let fl = u.floor();
            let f = u - fl;
            let i = fl as i64;
            vec![(i, 1.0 - f), (i + 1, f)]
        }
        _ => {
            let i0 = u.round() as i64;
            let d = u - i0 as f64; // ∈ [-0.5, 0.5]
            vec![
                (i0 - 1, 0.5 * (0.5 - d) * (0.5 - d)),
                (i0,     0.75 - d * d),
                (i0 + 1, 0.5 * (0.5 + d) * (0.5 + d)),
            ]
        }
    };
    let dim_i = dim as i64;
    raw.into_iter()
        .map(|(i, w)| {
            let node = match bc {
                BC::Periodic => i.rem_euclid(dim_i) as usize,
                BC::Neumann  => i.clamp(0, dim_i - 1) as usize,
            };
            (node, w)
        })
        .collect()
}

/// n-D stencil as tensor product of per-axis stencils: (flat node, weight) pairs.
fn node_stencil(f: &FieldVal, pos: &[f64], stride: &[usize], order: usize) -> Vec<(usize, f64)> {
    let n = f.grid.len();
    let mut acc: Vec<(usize, f64)> = vec![(0, 1.0)];
    for a in 0..n {
        let u = (pos[a] - f.lo[a]) / f.spacing[a];
        let ax = axis_stencil(u, f.grid[a], f.bc[a], order);
        let mut next = Vec::with_capacity(acc.len() * ax.len());
        for &(fi, fw) in &acc {
            for &(node, w) in &ax {
                next.push((fi + node * stride[a], fw * w));
            }
        }
        acc = next;
    }
    acc
}

/// 1-D shape-function + derivative stencil: (node, weight, dweight/du).
fn axis_stencil_d(u: f64, dim: usize, bc: BC, order: usize) -> Vec<(usize, f64, f64)> {
    let raw: Vec<(i64, f64, f64)> = match order {
        0 => vec![(u.round() as i64, 1.0, 0.0)],
        1 => {
            let fl = u.floor();
            let f = u - fl;
            let i = fl as i64;
            vec![(i, 1.0 - f, -1.0), (i + 1, f, 1.0)]
        }
        _ => {
            let i0 = u.round() as i64;
            let d = u - i0 as f64;
            vec![
                (i0 - 1, 0.5 * (0.5 - d) * (0.5 - d), d - 0.5),
                (i0,     0.75 - d * d,                -2.0 * d),
                (i0 + 1, 0.5 * (0.5 + d) * (0.5 + d), 0.5 + d),
            ]
        }
    };
    let dim_i = dim as i64;
    raw.into_iter()
        .map(|(i, w, dw)| {
            let node = match bc {
                BC::Periodic => i.rem_euclid(dim_i) as usize,
                BC::Neumann  => i.clamp(0, dim_i - 1) as usize,
            };
            (node, w, dw)
        })
        .collect()
}

/// n-D gradient stencil: (flat node, ∇_x S(x_g - x_i)) via product rule.
/// Axis `b` uses dS/du * (1/dx_b); other axes use the plain weight.
fn node_stencil_grad(f: &FieldVal, pos: &[f64], stride: &[usize], order: usize) -> Vec<(usize, Vec<f64>)> {
    let n = f.grid.len();
    let mut acc: Vec<(usize, f64, Vec<f64>)> = vec![(0, 1.0, vec![0.0; n])];
    for a in 0..n {
        let u = (pos[a] - f.lo[a]) / f.spacing[a];
        let ax = axis_stencil_d(u, f.grid[a], f.bc[a], order);
        let inv_dx = 1.0 / f.spacing[a];
        let mut next = Vec::with_capacity(acc.len() * ax.len());
        for (fi, fv, fg) in &acc {
            for &(node, w, dw) in &ax {
                let new_node = fi + node * stride[a];
                let new_val = fv * w;
                let mut new_grad = vec![0.0; n];
                for k in 0..n {
                    new_grad[k] = if k == a { fv * dw * inv_dx } else { fg[k] * w };
                }
                next.push((new_node, new_val, new_grad));
            }
        }
        acc = next;
    }
    acc.into_iter().map(|(node, _v, g)| (node, g)).collect()
}

// ── operations ──────────────────────────────────────────────────────────────────

/// `pic.scatter(positions, weights, template [, kernel])` — deposit weighted
/// particles onto the grid geometry described by `template`, returning a new
/// field with the same geometry. `weights` is [P] for a 0-form template (scalar
/// charge density) or [P, ncomp] for a higher-degree form (e.g. current J on a
/// vector-field template).
fn scatter(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 3 || vals.len() > 4 {
        return Err("pic.scatter(positions, weights, template [, kernel]) expects 3 or 4 args".into());
    }
    let order = kernel_order(vals.get(3), "pic.scatter")?;
    let mut it = vals.into_iter();
    let positions = it.next().unwrap();
    let weights   = it.next().unwrap();
    let template  = as_field(it.next().unwrap(), "pic.scatter")?;
    let n = template.grid.len();
    let ncomp = template.ncomp();
    let (p, pos) = parse_positions(&positions, n, "pic.scatter")?;
    let w = parse_weights(&weights, p, ncomp, "pic.scatter")?;
    let gt = template.grid.iter().product::<usize>().max(1);
    let stride = strides(&template.grid);
    let mut out = vec![0.0f64; gt * ncomp];
    for pi in 0..p {
        let st = node_stencil(&template, &pos[pi * n..pi * n + n], &stride, order);
        for (node, wt) in st {
            for c in 0..ncomp {
                out[node * ncomp + c] += wt * w[pi * ncomp + c];
            }
        }
    }
    Ok(with_data(&template, out))
}

/// `pic.gather(field, positions [, kernel])` — interpolate a field at particle
/// positions (the exact transpose of scatter). Returns a [P] tensor for a scalar
/// field or [P, ncomp] for a multi-component field (e.g. force from a vector field).
fn gather(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("pic.gather(field, positions [, kernel]) expects 2 or 3 args".into());
    }
    let order = kernel_order(vals.get(2), "pic.gather")?;
    let mut it = vals.into_iter();
    let field     = as_field(it.next().unwrap(), "pic.gather")?;
    let positions = it.next().unwrap();
    let n = field.grid.len();
    let ncomp = field.ncomp();
    let (p, pos) = parse_positions(&positions, n, "pic.gather")?;
    let stride = strides(&field.grid);
    let mut out = vec![0.0f64; p * ncomp];
    for pi in 0..p {
        let st = node_stencil(&field, &pos[pi * n..pi * n + n], &stride, order);
        for (node, wt) in st {
            for c in 0..ncomp {
                out[pi * ncomp + c] += wt * field.data[node * ncomp + c];
            }
        }
    }
    let shape = if ncomp == 1 { vec![p] } else { vec![p, ncomp] };
    compute::upload(env.target, &out, shape).map(Val::Tensor)
}

/// `pic.gathergrad(field, positions [, kernel])` — gather a *scalar* field using
/// the gradient of the shape function, returning ∂/∂x_i of the interpolation at
/// each particle: [P] in 1-D or [P, ndim] in n-D.
///
/// Unlike `gather(ops.grad(field))` (which finite-differences the field then
/// interpolates), this differentiates the kernel itself and is the EXACT transpose
/// of scatter's positional derivative — the variational force for energies that are
/// functions of the deposited field.
fn gathergrad(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("pic.gathergrad(field, positions [, kernel]) expects 2 or 3 args".into());
    }
    let order = kernel_order(vals.get(2), "pic.gathergrad")?;
    let mut it = vals.into_iter();
    let field     = as_field(it.next().unwrap(), "pic.gathergrad")?;
    let positions = it.next().unwrap();
    if field.ncomp() != 1 {
        return Err("pic.gathergrad: field must be scalar (a 0-form)".into());
    }
    let n = field.grid.len();
    let (p, pos) = parse_positions(&positions, n, "pic.gathergrad")?;
    let stride = strides(&field.grid);
    let mut out = vec![0.0f64; p * n];
    for pi in 0..p {
        let st = node_stencil_grad(&field, &pos[pi * n..pi * n + n], &stride, order);
        for (node, g) in st {
            for b in 0..n {
                out[pi * n + b] += g[b] * field.data[node];
            }
        }
    }
    let shape = if n == 1 { vec![p] } else { vec![p, n] };
    compute::upload(env.target, &out, shape).map(Val::Tensor)
}
