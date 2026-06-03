// solver — time integrators and diagnostics (PDE_ERGONOMICS §4, §5).
//
// The RHS `f` is called as f(t, y) and must return dy/dt with the same shape as
// the state y. State may be a real scalar or a real tensor (vectors are rank-1
// tensors). RK4 is the cheap high-value scheme; the integrating-factor (IMEX)
// stepper from the doc is intentionally deferred (it can't be made generic
// without knowing which part of the RHS is the stiff linear operator).
use crate::eval::{Val, TData, Env, apply_val, fmt_val, FieldVal, binop_val, state_is_finite};
use crate::ast::Op;
use std::sync::Arc;

// Tree-broadcast arithmetic on whole structured states (scalar / tensor / field /
// nested tuple). These let the steppers combine states directly — `q + dt*v` — with
// no manual flatten/rebuild: `binop_val` broadcasts over the tree's leaves, so a
// heterogeneous (particles, field) phase space evolves as one. (`scale` etc.)
fn scale(s: f64, v: Val) -> Result<Val, String> { binop_val(Val::Num(s), &Op::Mul, v) }
fn add(a: Val, b: Val) -> Result<Val, String> { binop_val(a, &Op::Add, b) }
fn sub(a: Val, b: Val) -> Result<Val, String> { binop_val(a, &Op::Sub, b) }

pub const NAMES: &[&str] = &["rk4", "odeint", "verlet", "tao", "cfl"];

pub fn members() -> std::collections::HashMap<String, Val> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}

pub fn dispatch(name: &str, vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    match name {
        "rk4"    => rk4(vals, env),
        "odeint" => odeint(vals, env),
        "verlet" => verlet(vals, env),
        "tao"    => tao(vals, env),
        "cfl"    => cfl(vals),
        _ => Err(format!("solver: unknown member '{name}'")),
    }
}

// State template: a recipe for flattening a structured Val (scalar / tensor /
// field / tuple-of-these) into one contiguous f64 phase-space vector and rebuilding
// it afterwards. A Tuple lets you pour a whole heterogeneous collection — particle
// coordinates AND field samples — into a single canonical (q, p) pair so the
// integrators evolve everything together; the physics coupling lives in the
// gradient functions. Field templates carry the grid geometry so the data can be
// re-inflated into a proper field for those functions.
enum State {
    Scalar,
    Tensor(Vec<usize>),
    Field(Arc<FieldVal>),
    Tuple(Vec<State>),
}

// Recursively append a Val's numbers to `out`, returning its template.
fn flatten(v: &Val, out: &mut Vec<f64>) -> Result<State, String> {
    match v {
        Val::Num(x) => { out.push(*x); Ok(State::Scalar) }
        Val::Tensor { data, shape } => { out.extend_from_slice(data); Ok(State::Tensor(shape.clone())) }
        Val::Field(f) => { out.extend_from_slice(&f.data); Ok(State::Field(f.clone())) }
        Val::Tuple(items) => {
            let mut subs = Vec::with_capacity(items.len());
            for it in items { subs.push(flatten(it, out)?); }
            Ok(State::Tuple(subs))
        }
        other => Err(format!("solver: state must be a scalar, tensor, field, or a tuple of these, got {}", fmt_val(other))),
    }
}

fn to_state(v: &Val) -> Result<(Vec<f64>, State), String> {
    let mut data = Vec::new();
    let st = flatten(v, &mut data)?;
    Ok((data, st))
}

// Walk the template, consuming numbers from `data` at cursor `cur`.
fn rebuild(data: &[f64], st: &State, cur: &mut usize) -> Val {
    match st {
        State::Scalar => { let x = data[*cur]; *cur += 1; Val::Num(x) }
        State::Tensor(s) => {
            let n: usize = s.iter().product();
            let slice = data[*cur..*cur + n].to_vec();
            *cur += n;
            Val::Tensor { data: TData::new(slice), shape: s.clone() }
        }
        State::Field(f) => {
            let n = f.data.len();
            let slice = data[*cur..*cur + n].to_vec();
            *cur += n;
            crate::ns::forms::with_data(f, slice)
        }
        State::Tuple(subs) => Val::Tuple(subs.iter().map(|s| rebuild(data, s, cur)).collect()),
    }
}

fn from_state(data: Vec<f64>, st: &State) -> Val {
    let mut cur = 0;
    rebuild(&data, st, &mut cur)
}

/// Evaluate dy/dt = f(t, y); returns the derivative flat vector (length-checked).
fn call_f(f: &Val, t: f64, y: &[f64], sh: &State, env: &Env) -> Result<Vec<f64>, String> {
    let yv = from_state(y.to_vec(), sh);
    let out = apply_val(f.clone(), vec![Val::Num(t), yv], env)?;
    let (d, _) = to_state(&out)?;
    if d.len() != y.len() {
        return Err(format!("solver: f returned {} values but state has {}", d.len(), y.len()));
    }
    Ok(d)
}

/// One classical RK4 step of size h from time t.
fn rk4_step(f: &Val, t: f64, y: &[f64], sh: &State, h: f64, env: &Env) -> Result<Vec<f64>, String> {
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
    match &sh {
        State::Scalar    => {}
        State::Tensor(s) => out_shape.extend_from_slice(s),
        _                => if l > 1 { out_shape.push(l); }   // field/tuple: stack raw flat data
    }
    Ok(Val::Tensor { data: TData::new(rows), shape: out_shape })
}

// verlet(dVdq, dTdp, q0, p0, dt, n) — velocity-Verlet (leapfrog), the workhorse
// symplectic integrator for a SEPARABLE Hamiltonian H(q,p) = T(p) + V(q). It is
// symplectic and time-reversible, so energy stays bounded over long runs (unlike
// rk4, which drifts). You supply the two gradient pieces as 1-arg functions:
//   dVdq(q) = ∂H/∂q  (the force, -dp/dt),   dTdp(p) = ∂H/∂p  (the velocity, dq/dt)
// each returning the same shape as the state. Build them straight from a potential
// with `deriv`, e.g. dVdq = q -> deriv(V, q). q0/p0 may each be a scalar, tensor,
// field, or a tuple of these — a tuple lets you evolve particles AND fields as one
// phase space (the coupling lives in dVdq). One step is the kick-drift-kick
// sequence; returns the final (q, p) with the same structure you passed in.
fn verlet(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() != 6 { return Err("solver.verlet(dVdq, dTdp, q0, p0, dt, n) expects 6 args".into()); }
    let mut it = vals.into_iter();
    let dvdq = it.next().unwrap();
    let dtdp = it.next().unwrap();
    let mut q = it.next().unwrap();
    let mut p = it.next().unwrap();
    let dt = it.next().unwrap().num("solver.verlet dt")?;
    let n  = it.next().unwrap().num("solver.verlet n")? as i64;
    if n <= 0 { return Err("solver.verlet: n must be a positive integer".into()); }
    // Kick–drift–kick on whole structured states. Tree broadcast handles scalars,
    // tensors, fields and tuples of them uniformly (shape mismatches surface as
    // arithmetic errors); fields keep their geometry through `binop_val`.
    for step in 0..n {
        let fq  = apply_val(dvdq.clone(), vec![q.clone()], env)?;     // ∂H/∂q at q
        let ph  = sub(p.clone(), scale(0.5 * dt, fq)?)?;             // half kick
        let vp  = apply_val(dtdp.clone(), vec![ph.clone()], env)?;    // ∂H/∂p at p½
        let qn  = add(q.clone(), scale(dt, vp)?)?;                   // drift
        let fq2 = apply_val(dvdq.clone(), vec![qn.clone()], env)?;    // ∂H/∂q at q'
        let pn  = sub(ph, scale(0.5 * dt, fq2)?)?;                   // half kick
        if !state_is_finite(&qn) || !state_is_finite(&pn) {
            return Err(format!("solver.verlet: non-finite value at step {step}"));
        }
        q = qn; p = pn;
    }
    Ok(Val::Tuple(vec![q, p]))
}

// Apply a 2-arg gradient g(a, b) where `a` follows template `ast` and `b` follows
// `bst`; return its flat output, length-checked against `len`.
fn call_grad2(g: &Val, a: &[f64], ast: &State, b: &[f64], bst: &State, len: usize, env: &Env, what: &str)
    -> Result<Vec<f64>, String>
{
    let av = from_state(a.to_vec(), ast);
    let bv = from_state(b.to_vec(), bst);
    let out = apply_val(g.clone(), vec![av, bv], env)?;
    let (d, _) = to_state(&out)?;
    if d.len() != len {
        return Err(format!("{what} returned {} values but the state has {}", d.len(), len));
    }
    Ok(d)
}

// Tao sub-flow A, from H_A = H(q, y): exact, since q and y are held fixed.
//   p -= h·∂qH(q,y);   x += h·∂pH(q,y)
fn tao_phi_a(h: f64, dhdq: &Val, dhdp: &Val, q: &[f64], p: &mut [f64], x: &mut [f64], y: &[f64],
             qst: &State, pst: &State, len: usize, env: &Env) -> Result<(), String> {
    let hq = call_grad2(dhdq, q, qst, y, pst, len, env, "solver.tao: dHdq")?;
    let hp = call_grad2(dhdp, q, qst, y, pst, len, env, "solver.tao: dHdp")?;
    for i in 0..len { p[i] -= h * hq[i]; x[i] += h * hp[i]; }
    Ok(())
}

// Tao sub-flow B, from H_B = H(x, p): exact, since x and p are held fixed.
//   q += h·∂pH(x,p);   y -= h·∂qH(x,p)
fn tao_phi_b(h: f64, dhdq: &Val, dhdp: &Val, q: &mut [f64], p: &[f64], x: &[f64], y: &mut [f64],
             qst: &State, pst: &State, len: usize, env: &Env) -> Result<(), String> {
    let hp = call_grad2(dhdp, x, qst, p, pst, len, env, "solver.tao: dHdp")?;
    let hq = call_grad2(dhdq, x, qst, p, pst, len, env, "solver.tao: dHdq")?;
    for i in 0..len { q[i] += h * hp[i]; y[i] -= h * hq[i]; }
    Ok(())
}

// Tao sub-flow C, the harmonic binding ω·½(|q−x|²+|p−y|²): an exact rotation by
// angle 2ωh of the difference (q−x, p−y), leaving the mean (q+x, p+y) fixed.
fn tao_phi_c(h: f64, omega: f64, q: &mut [f64], p: &mut [f64], x: &mut [f64], y: &mut [f64], len: usize) {
    let theta = 2.0 * omega * h;
    let (c, s) = (theta.cos(), theta.sin());
    for i in 0..len {
        let (qp, qm, pp, pm) = (q[i] + x[i], q[i] - x[i], p[i] + y[i], p[i] - y[i]);
        q[i] = 0.5 * (qp + c * qm + s * pm);
        p[i] = 0.5 * (pp - s * qm + c * pm);
        x[i] = 0.5 * (qp - c * qm - s * pm);
        y[i] = 0.5 * (pp + s * qm - c * pm);
    }
}

// tao(dHdq, dHdp, q0, p0, dt, n[, omega]) — Tao's explicit symplectic integrator
// for a NON-separable but canonical Hamiltonian H(q,p). When H won't split into
// T(p)+V(q) (e.g. electromagnetic PIC, whose (p−qA)² term mixes a momentum with a
// field coordinate), `verlet` no longer applies. Tao duplicates the system into an
// extended phase space (q,p)⊕(x,y), binds the copies with a harmonic term of
// strength ω, and Strang-composes three exactly-solvable sub-flows
// (A·B·C·B·A), giving a 2nd-order explicit symplectic step on the original H.
// You supply dHdq(q,p)=∂H/∂q and dHdp(q,p)=∂H/∂p as 2-arg functions. q0/p0 follow
// the same flexible templates as `verlet`. ω (default 100) binds the two copies:
// larger ω tracks H more tightly but needs a smaller dt — tune it. Returns (q, p).
fn tao(vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    if vals.len() != 6 && vals.len() != 7 {
        return Err("solver.tao(dHdq, dHdp, q0, p0, dt, n[, omega]) expects 6 or 7 args".into());
    }
    let mut it = vals.into_iter();
    let dhdq = it.next().unwrap();
    let dhdp = it.next().unwrap();
    let (q0, qst) = to_state(&it.next().unwrap())?;
    let (p0, pst) = to_state(&it.next().unwrap())?;
    let dt = it.next().unwrap().num("solver.tao dt")?;
    let n  = it.next().unwrap().num("solver.tao n")? as i64;
    let omega = match it.next() { Some(v) => v.num("solver.tao omega")?, None => 100.0 };
    if n <= 0 { return Err("solver.tao: n must be a positive integer".into()); }
    if q0.len() != p0.len() {
        return Err(format!("solver.tao: q0 and p0 must have the same length ({} vs {})", q0.len(), p0.len()));
    }
    let len = q0.len();
    // Extended phase space: the working copy (q, p) and a shadow (x, y) = (q, p).
    let (mut q, mut p) = (q0.clone(), p0.clone());
    let (mut x, mut y) = (q0, p0);
    let h = 0.5 * dt;
    for step in 0..n {
        tao_phi_a(h, &dhdq, &dhdp, &q, &mut p, &mut x, &y, &qst, &pst, len, env)?;
        tao_phi_b(h, &dhdq, &dhdp, &mut q, &p, &x, &mut y, &qst, &pst, len, env)?;
        tao_phi_c(dt, omega, &mut q, &mut p, &mut x, &mut y, len);
        tao_phi_b(h, &dhdq, &dhdp, &mut q, &p, &x, &mut y, &qst, &pst, len, env)?;
        tao_phi_a(h, &dhdq, &dhdp, &q, &mut p, &mut x, &y, &qst, &pst, len, env)?;
        for (i, v) in q.iter().chain(p.iter()).enumerate() {
            if !v.is_finite() {
                return Err(format!("solver.tao: non-finite value at step {step} (component {i})"));
            }
        }
    }
    Ok(Val::Tuple(vec![from_state(q, &qst), from_state(p, &pst)]))
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
