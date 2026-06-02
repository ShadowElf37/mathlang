// pic — particle/grid coupling for particle-in-cell (PIC) methods.
//
// Two operations, exact transposes of each other (same shape function S). Using
// one kernel for both is what keeps a Lagrangian PIC scheme free of self-force
// and — when S lives inside a single Hamiltonian, so that scatter = ∂H/∂A_grid
// and gather = ∂H/∂x_particle — canonical, so solver.tao integrates it:
//
//   * scatter (deposition):    particles -> grid field   (charge ρ, current J)
//       ρ_g = Σ_i w_i S(x_g − x_i)
//   * gather  (interpolation):  grid field -> particles   (force sampling)
//       E(x_i) = Σ_g E_g S(x_g − x_i)
//
// Shape functions: nearest-grid-point (pic.ngp, 0th order), cloud-in-cell
// (pic.cic, linear — the default and workhorse), triangular-shaped-cloud
// (pic.tsc, quadratic). Boundary handling follows the field's per-axis BC:
// periodic wraps node indices; neumann clamps them to the edge. Both stages use
// the identical (node, weight) stencil, so adjointness holds by construction.
use crate::eval::{Val, TData, FieldVal, BC, fmt_val};
use std::collections::HashMap;
use std::sync::Arc;

pub const NAMES: &[&str] = &["scatter", "gather", "gathergrad"];

pub fn members() -> HashMap<String, Val> {
    let mut m: HashMap<String, Val> =
        NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect();
    // kernel sentinels (mathlang has no string type), mirroring forms' BC consts.
    m.insert("ngp".into(), Val::Num(0.0));
    m.insert("cic".into(), Val::Num(1.0));
    m.insert("tsc".into(), Val::Num(2.0));
    m
}

pub fn dispatch(name: &str, vals: Vec<Val>, _env: &crate::eval::Env) -> Result<Val, String> {
    match name {
        "scatter"    => scatter(vals),
        "gather"     => gather(vals),
        "gathergrad" => gathergrad(vals),
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
        None => Ok(1), // pic.cic
        Some(Val::Num(x)) => {
            let o = x.round() as i64;
            if (x - o as f64).abs() > 1e-9 || !(0..=2).contains(&o) {
                return Err(format!("{what}: kernel must be pic.ngp, pic.cic, or pic.tsc"));
            }
            Ok(o as usize)
        }
        Some(other) => Err(format!("{what}: kernel must be pic.ngp/pic.cic/pic.tsc, got {}", fmt_val(other))),
    }
}

/// Parse particle positions into (P, flat row-major [P*n]). Accepts a [P,n]
/// tensor; for a 1-D grid also a [P] tensor or a bare scalar; a single position
/// vector [n] counts as one particle.
fn parse_positions(v: &Val, n: usize, what: &str) -> Result<(usize, Vec<f64>), String> {
    match v {
        Val::Num(x) if n == 1 => Ok((1, vec![*x])),
        Val::Tensor { data, shape } => {
            if n == 1 && shape.len() == 1 {
                Ok((shape[0], data.to_vec()))
            } else if shape.len() == 2 && shape[1] == n {
                Ok((shape[0], data.to_vec()))
            } else if shape.len() == 1 && shape[0] == n {
                Ok((1, data.to_vec()))
            } else {
                Err(format!("{what}: positions must be shape [P,{n}] (or [P] when n=1), got {shape:?}"))
            }
        }
        other => Err(format!("{what}: positions must be a tensor, got {}", fmt_val(other))),
    }
}

/// Parse per-particle weights into flat row-major [P*ncomp]. Accepts [P] (when
/// ncomp==1), [P,ncomp], or a bare scalar for a single scalar particle.
fn parse_weights(v: &Val, p: usize, ncomp: usize, what: &str) -> Result<Vec<f64>, String> {
    match v {
        Val::Num(x) if p == 1 && ncomp == 1 => Ok(vec![*x]),
        Val::Tensor { data, shape } => {
            if ncomp == 1 && shape.len() == 1 && shape[0] == p {
                Ok(data.to_vec())
            } else if shape.len() == 2 && shape[0] == p && shape[1] == ncomp {
                Ok(data.to_vec())
            } else {
                Err(format!("{what}: weights must be shape [{p}] (or [{p},{ncomp}]), got {shape:?}"))
            }
        }
        other => Err(format!("{what}: weights must be a tensor, got {}", fmt_val(other))),
    }
}

// ── shape functions ─────────────────────────────────────────────────────────

/// Row-major strides for a grid shape.
fn strides(grid: &[usize]) -> Vec<usize> {
    let n = grid.len();
    let mut s = vec![1usize; n];
    for a in (0..n.saturating_sub(1)).rev() {
        s[a] = s[a + 1] * grid[a + 1];
    }
    s
}

/// 1-D shape-function stencil at grid coordinate `u` (= (x−lo)/dx): the
/// contributing (node, weight) pairs along one axis, after applying the BC.
fn axis_stencil(u: f64, dim: usize, bc: BC, order: usize) -> Vec<(usize, f64)> {
    // Raw (integer node, weight) before the boundary map.
    let raw: Vec<(i64, f64)> = match order {
        0 => {
            // nearest-grid-point
            vec![(u.round() as i64, 1.0)]
        }
        1 => {
            // cloud-in-cell (linear)
            let fl = u.floor();
            let f = u - fl;
            let i = fl as i64;
            vec![(i, 1.0 - f), (i + 1, f)]
        }
        _ => {
            // triangular-shaped-cloud (quadratic), centred on the nearest node
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

/// n-D stencil at a particle position: tensor product of the per-axis stencils,
/// as (flat grid node, weight) pairs. Weights sum to 1 (interior) for any kernel.
fn node_stencil(f: &FieldVal, pos: &[f64], stride: &[usize], order: usize) -> Vec<(usize, f64)> {
    let n = f.grid.len();
    let mut acc: Vec<(usize, f64)> = vec![(0usize, 1.0)];
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

/// Like `axis_stencil`, but also returns d(weight)/du for each node — the
/// derivative of the 1-D shape function w.r.t. the (dimensionless) particle
/// coordinate u. Used to assemble the gradient of the interpolation kernel.
fn axis_stencil_d(u: f64, dim: usize, bc: BC, order: usize) -> Vec<(usize, f64, f64)> {
    // Raw (integer node, weight, dweight/du) before the boundary map.
    let raw: Vec<(i64, f64, f64)> = match order {
        0 => vec![(u.round() as i64, 1.0, 0.0)], // NGP: piecewise-constant, derivative 0
        1 => {
            let fl = u.floor();
            let f = u - fl;
            let i = fl as i64;
            vec![(i, 1.0 - f, -1.0), (i + 1, f, 1.0)]
        }
        _ => {
            let i0 = u.round() as i64;
            let d = u - i0 as f64; // ∈ [-0.5, 0.5]
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

/// n-D gradient stencil at a particle position: for each contributing grid node,
/// the gradient (w.r.t. the particle's position) of the tensor-product shape
/// function, ∂S/∂x along each axis. Built by the product rule — axis b uses the
/// per-axis derivative (scaled by 1/spacing), the other axes use the plain weight.
fn node_stencil_grad(f: &FieldVal, pos: &[f64], stride: &[usize], order: usize) -> Vec<(usize, Vec<f64>)> {
    let n = f.grid.len();
    let mut acc: Vec<(usize, f64, Vec<f64>)> = vec![(0usize, 1.0, vec![0.0; n])];
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

// ── operations ─────────────────────────────────────────────────────────────

// scatter(positions, weights, template [, kernel]) — deposit weighted particles
// onto the grid, producing a field with `template`'s geometry/degree/variance.
// weights is [P] for a scalar (0-form) template, else [P, ncomp]. The result is
// the standard PIC source: charge density (scalar weights = q_i) or current
// density (vector template, weights = q_i v_i).
fn scatter(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 3 || vals.len() > 4 {
        return Err("pic.scatter(positions, weights, template [, kernel]) expects 3 or 4 args".into());
    }
    let order = kernel_order(vals.get(3), "pic.scatter")?;
    let mut it = vals.into_iter();
    let positions = it.next().unwrap();
    let weights = it.next().unwrap();
    let template = as_field(it.next().unwrap(), "pic.scatter")?;
    let n = template.grid.len();
    let ncomp = template.ncomp();
    let (p, pos) = parse_positions(&positions, n, "pic.scatter")?;
    let w = parse_weights(&weights, p, ncomp, "pic.scatter")?;
    let gt: usize = template.grid.iter().product::<usize>().max(1);
    let stride = strides(&template.grid);
    let mut out = vec![0.0; gt * ncomp];
    for pi in 0..p {
        let st = node_stencil(&template, &pos[pi * n..pi * n + n], &stride, order);
        for (node, wt) in st {
            for c in 0..ncomp {
                out[node * ncomp + c] += wt * w[pi * ncomp + c];
            }
        }
    }
    Ok(crate::ns::forms::with_data(&template, out))
}

// gather(field, positions [, kernel]) — sample a field at particle positions
// (the exact transpose of scatter with the same kernel). Returns [P] for a
// scalar field, else [P, ncomp] (e.g. a vector field gives the force per particle).
fn gather(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("pic.gather(field, positions [, kernel]) expects 2 or 3 args".into());
    }
    let order = kernel_order(vals.get(2), "pic.gather")?;
    let mut it = vals.into_iter();
    let field = as_field(it.next().unwrap(), "pic.gather")?;
    let positions = it.next().unwrap();
    let n = field.grid.len();
    let ncomp = field.ncomp();
    let (p, pos) = parse_positions(&positions, n, "pic.gather")?;
    let stride = strides(&field.grid);
    let mut out = vec![0.0; p * ncomp];
    for pi in 0..p {
        let st = node_stencil(&field, &pos[pi * n..pi * n + n], &stride, order);
        for (node, wt) in st {
            for c in 0..ncomp {
                out[pi * ncomp + c] += wt * field.data[node * ncomp + c];
            }
        }
    }
    let shape = if ncomp == 1 { vec![p] } else { vec![p, ncomp] };
    Ok(Val::Tensor { data: TData::new(out), shape })
}

// gathergrad(field, positions [, kernel]) — gather a SCALAR field at the particle
// positions using the GRADIENT of the shape function, returning ∂/∂x_i of the
// interpolated value: [P, ndim] (or [P] in 1-D). Unlike gather(ops.grad(field)),
// which finite-differences the field then interpolates, this differentiates the
// kernel itself, so it is the EXACT transpose of scatter's position-derivative.
// That makes it the variational force for an energy that is a function of the
// deposited field: for V = Σ_g g(ρ_g) with ρ = scatter(x, m), the per-particle
// force is −m·dA·gathergrad(g'(ρ)) and Verlet conserves H exactly (no grid-scale
// heating). The workhorse for self-gravitating / barotropic particle-mesh gases.
fn gathergrad(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() < 2 || vals.len() > 3 {
        return Err("pic.gathergrad(field, positions [, kernel]) expects 2 or 3 args".into());
    }
    let order = kernel_order(vals.get(2), "pic.gathergrad")?;
    let mut it = vals.into_iter();
    let field = as_field(it.next().unwrap(), "pic.gathergrad")?;
    let positions = it.next().unwrap();
    if field.ncomp() != 1 {
        return Err("pic.gathergrad: field must be scalar (a 0-form)".into());
    }
    let n = field.grid.len();
    let (p, pos) = parse_positions(&positions, n, "pic.gathergrad")?;
    let stride = strides(&field.grid);
    let mut out = vec![0.0; p * n];
    for pi in 0..p {
        let st = node_stencil_grad(&field, &pos[pi * n..pi * n + n], &stride, order);
        for (node, g) in st {
            for b in 0..n {
                out[pi * n + b] += g[b] * field.data[node];
            }
        }
    }
    let shape = if n == 1 { vec![p] } else { vec![p, n] };
    Ok(Val::Tensor { data: TData::new(out), shape })
}
