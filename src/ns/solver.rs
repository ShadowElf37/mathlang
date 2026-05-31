// solver — time integrators and diagnostics (PDE_ERGONOMICS §4, §5).
//
// The RHS `f` is called as f(t, y) and must return dy/dt with the same shape as
// the state y. State may be a real scalar or a real tensor (vectors are rank-1
// tensors). RK4 is the cheap high-value scheme; the integrating-factor (IMEX)
// stepper from the doc is intentionally deferred (it can't be made generic
// without knowing which part of the RHS is the stiff linear operator).
use crate::eval::{Val, TData, Env, apply_val, fmt_val};

pub const NAMES: &[&str] = &["rk4", "odeint", "cfl"];

pub fn members() -> std::collections::HashMap<String, Val> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}

pub fn dispatch(name: &str, vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    match name {
        "rk4"    => rk4(vals, env),
        "odeint" => odeint(vals, env),
        "cfl"    => cfl(vals),
        _ => Err(format!("solver: unknown member '{name}'")),
    }
}

// State shape: scalar, or a tensor with a recorded shape.
enum Shape { Scalar, Tensor(Vec<usize>) }

fn to_state(v: &Val) -> Result<(Vec<f64>, Shape), String> {
    match v {
        Val::Num(x) => Ok((vec![*x], Shape::Scalar)),
        Val::Tensor { data, shape } => Ok((data.to_vec(), Shape::Tensor(shape.clone()))),
        other => Err(format!("solver: state must be a real scalar or tensor, got {}", fmt_val(other))),
    }
}

fn from_state(data: Vec<f64>, shape: &Shape) -> Val {
    match shape {
        Shape::Scalar     => Val::Num(data[0]),
        Shape::Tensor(s)  => Val::Tensor { data: TData::new(data), shape: s.clone() },
    }
}

/// Evaluate dy/dt = f(t, y); returns the derivative flat vector (length-checked).
fn call_f(f: &Val, t: f64, y: &[f64], sh: &Shape, env: &Env) -> Result<Vec<f64>, String> {
    let yv = from_state(y.to_vec(), sh);
    let out = apply_val(f.clone(), vec![Val::Num(t), yv], env)?;
    let (d, _) = to_state(&out)?;
    if d.len() != y.len() {
        return Err(format!("solver: f returned {} values but state has {}", d.len(), y.len()));
    }
    Ok(d)
}

/// One classical RK4 step of size h from time t.
fn rk4_step(f: &Val, t: f64, y: &[f64], sh: &Shape, h: f64, env: &Env) -> Result<Vec<f64>, String> {
    let n = y.len();
    let k1 = call_f(f, t, y, sh, env)?;
    let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let k2 = call_f(f, t + 0.5 * h, &y2, sh, env)?;
    let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k2[i]).collect();
    let k3 = call_f(f, t + 0.5 * h, &y3, sh, env)?;
    let y4: Vec<f64> = (0..n).map(|i| y[i] + h * k3[i]).collect();
    let k4 = call_f(f, t + h, &y4, sh, env)?;
    let out: Vec<f64> = (0..n)
        .map(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect();
    for (i, v) in out.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("solver.rk4: non-finite value at t={:.6} (component {i})", t + h));
        }
    }
    Ok(out)
}

// rk4(f, y0, t0, t1, n) — fixed-step RK4; returns the final state after n steps.
fn rk4(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() != 5 { return Err("solver.rk4(f, y0, t0, t1, n) expects 5 args".into()); }
    let mut it = vals.into_iter();
    let f = it.next().unwrap();
    let (mut y, sh) = to_state(&it.next().unwrap())?;
    let t0 = it.next().unwrap().num("solver.rk4 t0")?;
    let t1 = it.next().unwrap().num("solver.rk4 t1")?;
    let n = it.next().unwrap().num("solver.rk4 n")? as i64;
    if n <= 0 { return Err("solver.rk4: n must be a positive integer".into()); }
    let h = (t1 - t0) / n as f64;
    for step in 0..n {
        let t = t0 + step as f64 * h;
        y = rk4_step(&f, t, &y, &sh, h, env)?;
    }
    Ok(from_state(y, &sh))
}

// odeint(f, y0, ts) — RK4 sampled at the times in `ts` (one step per interval).
// Returns the trajectory stacked along a new leading axis: scalar states give a
// 1-D tensor [len(ts)]; a state of shape S gives a tensor [len(ts)] ++ S.
fn odeint(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() != 3 { return Err("solver.odeint(f, y0, ts) expects 3 args".into()); }
    let mut it = vals.into_iter();
    let f = it.next().unwrap();
    let (mut y, sh) = to_state(&it.next().unwrap())?;
    let ts: Vec<f64> = match it.next().unwrap() {
        Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
        Val::Tuple(items) => items.into_iter().map(|v| v.num("solver.odeint ts")).collect::<Result<_, _>>()?,
        other => return Err(format!("solver.odeint: ts must be a 1-D tensor of times, got {}", fmt_val(&other))),
    };
    if ts.len() < 2 { return Err("solver.odeint: ts needs at least 2 times".into()); }
    let l = y.len();
    let mut rows: Vec<f64> = Vec::with_capacity(ts.len() * l);
    rows.extend_from_slice(&y);                       // state at ts[0]
    for w in ts.windows(2) {
        let h = w[1] - w[0];
        y = rk4_step(&f, w[0], &y, &sh, h, env)?;
        rows.extend_from_slice(&y);
    }
    let mut out_shape = vec![ts.len()];
    if let Shape::Tensor(s) = &sh { out_shape.extend_from_slice(s); }
    Ok(Val::Tensor { data: TData::new(rows), shape: out_shape })
}

// cfl(V, dx, dt) — Courant number dt*max|V|/dx, a stability diagnostic.
fn cfl(vals: Vec<Val>) -> Result<Val, String> {
    if vals.len() != 3 { return Err("solver.cfl(V, dx, dt) expects 3 args".into()); }
    let mut it = vals.into_iter();
    let vmax = match it.next().unwrap() {
        Val::Num(x) => x.abs(),
        Val::Tensor { data, .. } => data.iter().fold(0.0_f64, |m, &x| m.max(x.abs())),
        other => return Err(format!("solver.cfl: V must be a real scalar or tensor, got {}", fmt_val(&other))),
    };
    let dx = it.next().unwrap().num("solver.cfl dx")?;
    let dt = it.next().unwrap().num("solver.cfl dt")?;
    Ok(Val::Num(dt * vmax / dx))
}
