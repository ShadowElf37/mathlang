//! Scalar/complex builtins for the Phase 1b host core. Unary math broadcasts over
//! tuple trees (matching the original's leaf semantics). Tensor/linalg/fft/field
//! builtins are deferred to later phases.

use crate::ast::Op;
use crate::compute::{self, TensorVal};
use crate::interp::{binop_val, ensure_on, Env};
use crate::value::{fmt_val, make_complex, to_complex, Val};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

pub fn eval_builtin(name: &str, args: Vec<Val>, env: &Env) -> Result<Val, String> {
    // Tensor fast-paths: a unary math builtin applied to a tensor runs on the
    // compute path. Constructors build tensors on the active target.
    if args.len() == 1 {
        if let Val::ComplexTensor(ct) = &args[0] {
            // complex → real
            let c2r = match name {
                "re" => Some(compute::CR_RE),
                "im" => Some(compute::CR_IM),
                "abs" => Some(compute::CR_ABS),
                "arg" => Some(compute::CR_ARG),
                _ => None,
            };
            if let Some(code) = c2r {
                return compute::cunary_c2r(env.target, code, ct).map(Val::Tensor);
            }
            // complex → complex
            let c2c = match name {
                "conj" => Some(compute::CU_CONJ),
                "exp" => Some(compute::CU_EXP),
                "ln" => Some(compute::CU_LN),
                "sqrt" => Some(compute::CU_SQRT),
                "sin" => Some(compute::CU_SIN),
                "cos" => Some(compute::CU_COS),
                _ => None,
            };
            if let Some(code) = c2c {
                return compute::cunary_c2c(env.target, code, ct).map(Val::ComplexTensor);
            }
            if name == "mean" {
                let (r, i) = compute::creduce_sum(ct)?;
                let n = ct.len.max(1) as f64;
                return Ok(make_complex(r / n, i / n));
            }
        }
        if let Val::Tensor(t) = &args[0] {
            if let Some(code) = unary_tensor_code(name) {
                return compute::unary(env.target, code, t).map(Val::Tensor);
            }
            // whole-tensor reductions (device; df64 falls back to host in compute)
            match name {
                "sum" => return compute::reduce(env.target, compute::RED_SUM, t).map(Val::Num),
                "prod" => return compute::reduce(env.target, compute::RED_PROD, t).map(Val::Num),
                "min" => return compute::reduce(env.target, compute::RED_MIN, t).map(Val::Num),
                "max" => return compute::reduce(env.target, compute::RED_MAX, t).map(Val::Num),
                "mean" => {
                    let s = compute::reduce(env.target, compute::RED_SUM, t)?;
                    return Ok(Val::Num(s / t.len.max(1) as f64));
                }
                "norm" => {
                    let sq = compute::binop(env.target, compute::OP_MUL, t, t)?;
                    let s = compute::reduce(env.target, compute::RED_SUM, &sq)?;
                    return Ok(Val::Num(s.sqrt()));
                }
                "std" => return tensor_std(env, t),
                _ => {}
            }
        }
    }
    if let Some(r) = tensor_constructor(name, &args, env) {
        return r;
    }

    match name {
        // ── complex-capable unary ──────────────────────────────────────────────
        "abs" => unary(args, name, |v| {
            let (a, b) = to_complex(v)?;
            Ok(Val::Num(a.hypot(b)))
        }),
        "re" => unary(args, name, |v| { let (a, _) = to_complex(v)?; Ok(Val::Num(a)) }),
        "im" => unary(args, name, |v| { let (_, b) = to_complex(v)?; Ok(Val::Num(b)) }),
        "arg" => unary(args, name, |v| { let (a, b) = to_complex(v)?; Ok(Val::Num(b.atan2(a))) }),
        "conj" => unary(args, name, |v| { let (a, b) = to_complex(v)?; Ok(make_complex(a, -b)) }),
        "exp" => unary(args, name, |v| match v {
            Val::Num(x) => Ok(Val::Num(x.exp())),
            other => { let (a, b) = to_complex(other)?; let m = a.exp(); Ok(make_complex(m * b.cos(), m * b.sin())) }
        }),
        "ln" => unary(args, name, |v| match v {
            Val::Num(x) if x > 0.0 => Ok(Val::Num(x.ln())),
            other => { let (a, b) = to_complex(other)?; let r = a.hypot(b); Ok(make_complex(r.ln(), b.atan2(a))) }
        }),
        "sqrt" => unary(args, name, |v| match v {
            Val::Num(x) if x >= 0.0 => Ok(Val::Num(x.sqrt())),
            other => { let (a, b) = to_complex(other)?; Ok(crate::interp::complex_pow(a, b, 0.5, 0.0)) }
        }),
        "sin" => unary(args, name, |v| match v {
            Val::Num(x) => Ok(Val::Num(x.sin())),
            other => { let (a, b) = to_complex(other)?; Ok(make_complex(a.sin() * b.cosh(), a.cos() * b.sinh())) }
        }),
        "cos" => unary(args, name, |v| match v {
            Val::Num(x) => Ok(Val::Num(x.cos())),
            other => { let (a, b) = to_complex(other)?; Ok(make_complex(a.cos() * b.cosh(), -a.sin() * b.sinh())) }
        }),

        // ── real-only unary ────────────────────────────────────────────────────
        "tan" => real_unary(args, name, f64::tan),
        "asin" => real_unary(args, name, f64::asin),
        "acos" => real_unary(args, name, f64::acos),
        "atan" => real_unary(args, name, f64::atan),
        "sinh" => real_unary(args, name, f64::sinh),
        "cosh" => real_unary(args, name, f64::cosh),
        "tanh" => real_unary(args, name, f64::tanh),
        "sec" => real_unary(args, name, |x| 1.0 / x.cos()),
        "csc" => real_unary(args, name, |x| 1.0 / x.sin()),
        "cot" => real_unary(args, name, |x| 1.0 / x.tan()),
        "cbrt" => real_unary(args, name, f64::cbrt),
        "expm1" => real_unary(args, name, f64::exp_m1),
        "floor" => real_unary(args, name, f64::floor),
        "ceil" => real_unary(args, name, f64::ceil),
        "round" => real_unary(args, name, f64::round),
        "trunc" => real_unary(args, name, f64::trunc),
        "frac" => real_unary(args, name, |x| x - x.trunc()),
        "sign" | "signum" => real_unary(args, name, |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
        "heaviside" => real_unary(args, name, |x| if x < 0.0 { 0.0 } else if x == 0.0 { 0.5 } else { 1.0 }),
        "deg" => real_unary(args, name, |x| x * 180.0 / std::f64::consts::PI),
        "rad" => real_unary(args, name, |x| x * std::f64::consts::PI / 180.0),
        "fact" | "factorial" => real_unary(args, name, |x| {
            let n = x as i64;
            (1..=n.max(0)).map(|k| k as f64).product()
        }),
        "id" => one(args, name),

        // ── logarithms (base-10 by default; log(x, base) too) ──────────────────
        "log10" => real_unary(args, name, f64::log10),
        "log2" => real_unary(args, name, f64::log2),
        "log" => {
            if args.len() == 1 {
                real_unary(args, name, f64::log10)
            } else if args.len() == 2 {
                let x = args[0].clone().num(name)?;
                let base = args[1].clone().num(name)?;
                Ok(Val::Num(x.ln() / base.ln()))
            } else {
                Err("log expects log(x) or log(x, base)".into())
            }
        }

        // ── binary numeric ─────────────────────────────────────────────────────
        "pow" => { let (a, b) = two(args, name)?; binop_val(a, &Op::Pow, b, env.target) }
        "atan2" => { let (y, x) = two_nums(args, name)?; Ok(Val::Num(y.atan2(x))) }
        "hypot" => { let (a, b) = two_nums(args, name)?; Ok(Val::Num(a.hypot(b))) }
        "gcd" => { let (a, b) = two_nums(args, name)?; Ok(Val::Num(gcd(a.abs() as u64, b.abs() as u64) as f64)) }
        "lcm" => { let (a, b) = two_nums(args, name)?; Ok(Val::Num(lcm(a.abs() as u64, b.abs() as u64) as f64)) }
        "ncr" => {
            let (n, r) = two_nums(args, name)?;
            let (n, r) = (n as i64, r as i64);
            if r < 0 || r > n || n < 0 { return Ok(Val::Num(0.0)); }
            let mut acc = 1.0_f64;
            for k in 0..r { acc = acc * (n - k) as f64 / (k + 1) as f64; }
            Ok(Val::Num(acc.round()))
        }
        "min" => Ok(Val::Num(num_leaves(args, name)?.into_iter().fold(f64::INFINITY, f64::min))),
        "max" => Ok(Val::Num(num_leaves(args, name)?.into_iter().fold(f64::NEG_INFINITY, f64::max))),

        // ── comparison functions (reuse the operator semantics) ────────────────
        "lt" => cmp(args, name, Op::Lt, env),
        "leq" => cmp(args, name, Op::LtEq, env),
        "gt" => cmp(args, name, Op::Gt, env),
        "geq" => cmp(args, name, Op::GtEq, env),
        "eq" => cmp(args, name, Op::Eq, env),
        "neq" => cmp(args, name, Op::Ne, env),

        // ── higher-order combinators ───────────────────────────────────────────
        "compose" => {
            let (f, g) = two(args, name)?;
            let mut cap = HashMap::new();
            cap.insert("__f__".into(), f);
            cap.insert("__g__".into(), g);
            let body = crate::ast::Expr::Apply(
                Box::new(crate::ast::Expr::Var("__f__".into())),
                vec![crate::ast::Expr::Apply(
                    Box::new(crate::ast::Expr::Var("__g__".into())),
                    vec![crate::ast::Expr::Var("__z__".into())],
                )],
            );
            Ok(Val::make_fn(vec!["__z__".into()], body, Arc::new(cap)))
        }
        "partial" => {
            let (f, a) = two(args, name)?;
            let mut cap = HashMap::new();
            cap.insert("__f__".into(), f);
            cap.insert("__a__".into(), a);
            let body = crate::ast::Expr::Apply(
                Box::new(crate::ast::Expr::Var("__f__".into())),
                vec![crate::ast::Expr::Var("__a__".into()), crate::ast::Expr::Var("__x__".into())],
            );
            Ok(Val::make_fn(vec!["__x__".into()], body, Arc::new(cap)))
        }

        // ── containers ─────────────────────────────────────────────────────────
        "len" | "length" => {
            let v = one(args, name)?;
            match v {
                Val::Tuple(items) => Ok(Val::Num(items.len() as f64)),
                Val::Tensor(t) => Ok(Val::Num(*t.shape.first().unwrap_or(&0) as f64)),
                other => Err(format!("{name}: expected a tuple or tensor, got {}", fmt_val(&other))),
            }
        }
        "cell" => Ok(Val::Cell(Arc::new(RefCell::new(one(args, name)?)))),
        "get" => match one(args, name)? {
            Val::Cell(c) => Ok(c.borrow().clone()),
            other => Err(format!("get: expected a cell, got {}", fmt_val(&other))),
        },
        "set" => {
            let (c, v) = two(args, name)?;
            match c {
                Val::Cell(cell) => { *cell.borrow_mut() = v.clone(); Ok(v) }
                other => Err(format!("set: first arg must be a cell, got {}", fmt_val(&other))),
            }
        }

        // ── linear algebra (the @ operator routes here) ─────────────────────────
        "matmul" => {
            let (a, b) = two(args, name)?;
            matmul_vals(a, b, env)
        }
        // reductions reach here only for non-tensor args (tensors handled above)
        "mean" | "norm" | "std" => {
            let v = one(args, name)?;
            Err(format!("{name}: expected a tensor, got {}", fmt_val(&v)))
        }

        // ── stencils ────────────────────────────────────────────────────────────
        "shift" | "roll" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(format!("{name}(T, n[, axis]) expects 2 or 3 args"));
            }
            let t = real_tensor_on(&args[0], env, name)?;
            let n = args[1].clone().num(name)? as i64;
            let axis = match args.get(2) { Some(v) => v.clone().num(name)? as usize, None => 0 };
            if name == "roll" { compute::roll(env.target, &t, n, axis) } else { compute::shift(env.target, &t, n, axis) }.map(Val::Tensor)
        }
        "ops.lap" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("ops.lap(T, dx[, bc]) expects 2 or 3 args".into());
            }
            let t = real_tensor_on(&args[0], env, "ops.lap")?;
            let dx = args[1].clone().num("ops.lap")?;
            // bc marker: ops.neumann (1) → Neumann, else periodic
            let neumann = matches!(args.get(2), Some(v) if v.clone().num("ops.lap").unwrap_or(0.0) != 0.0);
            compute::lap(env.target, &t, dx, if neumann { 0 } else { 1 }).map(Val::Tensor)
        }
        "ops.grad" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("ops.grad(T, dx[, axis]) expects 2 or 3 args".into());
            }
            let t = real_tensor_on(&args[0], env, "ops.grad")?;
            let dx = args[1].clone().num("ops.grad")?;
            let axis = match args.get(2) { Some(v) => v.clone().num("ops.grad")? as usize, None => 0 };
            compute::grad(env.target, &t, dx, axis, 1).map(Val::Tensor)
        }

        // ── dense linear algebra (host-side; eig is staged) ─────────────────────
        "det" => {
            let (mut m, n, c) = real_square(&one(args, name)?, "det")?;
            let _ = c;
            Ok(Val::Num(host_det(&mut m, n)))
        }
        "inv" => {
            let (m, n, _) = real_square(&one(args, name)?, "inv")?;
            let data = host_inv(m, n)?;
            compute::upload(env.target, &data, vec![n, n]).map(Val::Tensor)
        }
        "solve" => {
            let (a, b) = two(args, name)?;
            let (m, n, _) = real_square(&a, "solve")?;
            let bv = match &b {
                Val::Tensor(t) if t.shape.len() == 1 => compute::download(t)?,
                Val::Tuple(items) => items.iter().map(|v| v.clone().num("solve")).collect::<Result<Vec<_>, _>>()?,
                other => return Err(format!("solve: b must be a 1-D tensor or tuple, got {}", fmt_val(other))),
            };
            if bv.len() != n {
                return Err(format!("solve: A is {n}×{n} but b has length {}", bv.len()));
            }
            let x = host_solve(m, bv, n)?;
            compute::upload(env.target, &x, vec![n]).map(Val::Tensor)
        }
        "eig" | "eigvals" => Err(format!(
            "{name} is staged (a robust real eigensolver is pending); det/inv/solve are available"
        )),

        _ => {
            // Help the user: many names exist in the real `m` but await later phases.
            Err(format!("`{name}` is not available in the prototype yet (later phase)"))
        }
    }
}

// ── argument helpers ──────────────────────────────────────────────────────────

fn one(args: Vec<Val>, name: &str) -> Result<Val, String> {
    if args.len() != 1 {
        return Err(format!("{name} expects 1 arg, got {}", args.len()));
    }
    Ok(args.into_iter().next().unwrap())
}

fn two(args: Vec<Val>, name: &str) -> Result<(Val, Val), String> {
    if args.len() != 2 {
        return Err(format!("{name} expects 2 args, got {}", args.len()));
    }
    let mut it = args.into_iter();
    Ok((it.next().unwrap(), it.next().unwrap()))
}

fn two_nums(args: Vec<Val>, name: &str) -> Result<(f64, f64), String> {
    let (a, b) = two(args, name)?;
    Ok((a.num(name)?, b.num(name)?))
}

/// Apply a leaf op across a value, broadcasting over tuple trees.
fn map_leaves(v: Val, f: &dyn Fn(Val) -> Result<Val, String>) -> Result<Val, String> {
    match v {
        Val::Tuple(items) => Ok(Val::Tuple(
            items.into_iter().map(|x| map_leaves(x, f)).collect::<Result<Vec<_>, _>>()?,
        )),
        leaf => f(leaf),
    }
}

fn unary(args: Vec<Val>, name: &str, f: impl Fn(Val) -> Result<Val, String>) -> Result<Val, String> {
    let v = one(args, name)?;
    map_leaves(v, &f)
}

fn real_unary(args: Vec<Val>, name: &str, f: impl Fn(f64) -> f64) -> Result<Val, String> {
    let v = one(args, name)?;
    map_leaves(v, &|leaf| match leaf {
        Val::Num(x) => Ok(Val::Num(f(x))),
        other => Err(format!("{name}: expected a real number, got {}", fmt_val(&other))),
    })
}

fn cmp(args: Vec<Val>, name: &str, op: Op, env: &Env) -> Result<Val, String> {
    let (a, b) = two(args, name)?;
    binop_val(a, &op, b, env.target)
}

/// Map a unary-math builtin name to a compute unary op code (tensor fast-path).
fn unary_tensor_code(name: &str) -> Option<u32> {
    Some(match name {
        "abs" => compute::UN_ABS,
        "exp" => compute::UN_EXP,
        "ln" => compute::UN_LN,
        "sqrt" => compute::UN_SQRT,
        "sin" => compute::UN_SIN,
        "cos" => compute::UN_COS,
        "tan" => compute::UN_TAN,
        "asin" => compute::UN_ASIN,
        "acos" => compute::UN_ACOS,
        "atan" => compute::UN_ATAN,
        "sinh" => compute::UN_SINH,
        "cosh" => compute::UN_COSH,
        "tanh" => compute::UN_TANH,
        "trunc" => compute::UN_TRUNC,
        "deg" => compute::UN_DEG,
        "rad" => compute::UN_RAD,
        _ => return None,
    })
}

/// Tensor constructors and shape queries. Returns `None` if `name` isn't one,
/// so the main match handles it.
fn tensor_constructor(name: &str, args: &[Val], env: &Env) -> Option<Result<Val, String>> {
    let dims = |args: &[Val]| -> Result<Vec<usize>, String> {
        args.iter().map(|v| Ok(v.clone().num(name)? as usize)).collect()
    };
    let upload = |data: Vec<f64>, shape: Vec<usize>| compute::upload(env.target, &data, shape).map(Val::Tensor);
    Some(match name {
        "zeros" => match dims(args) {
            Ok(shape) => upload(vec![0.0; shape.iter().product()], shape),
            Err(e) => Err(e),
        },
        "ones" => match dims(args) {
            Ok(shape) => upload(vec![1.0; shape.iter().product()], shape),
            Err(e) => Err(e),
        },
        "eye" => {
            if args.len() != 1 {
                return Some(Err("eye(n) expects 1 arg".into()));
            }
            match args[0].clone().num(name) {
                Ok(nf) => {
                    let n = nf as usize;
                    let mut d = vec![0.0; n * n];
                    for i in 0..n {
                        d[i * n + i] = 1.0;
                    }
                    upload(d, vec![n, n])
                }
                Err(e) => Err(e),
            }
        }
        "linspace" => {
            if args.len() != 3 {
                return Some(Err("linspace(a, b, n) expects 3 args".into()));
            }
            let a = match args[0].clone().num(name) { Ok(v) => v, Err(e) => return Some(Err(e)) };
            let b = match args[1].clone().num(name) { Ok(v) => v, Err(e) => return Some(Err(e)) };
            let n = match args[2].clone().num(name) { Ok(v) => v as usize, Err(e) => return Some(Err(e)) };
            let data: Vec<f64> = if n <= 1 {
                vec![a; n]
            } else {
                (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
            };
            upload(data, vec![n])
        }
        "range" => {
            // exclusive end, matching the original `range(a, b)` builtin
            if args.len() != 2 {
                return Some(Err("range(a, b) expects 2 args".into()));
            }
            let a = match args[0].clone().num(name) { Ok(v) => v as i64, Err(e) => return Some(Err(e)) };
            let b = match args[1].clone().num(name) { Ok(v) => v as i64, Err(e) => return Some(Err(e)) };
            let data: Vec<f64> = (a..b).map(|n| n as f64).collect();
            let n = data.len();
            upload(data, vec![n])
        }
        "shape" => match one_ref(args, name) {
            Ok(Val::Tensor(t)) => {
                let s: Vec<f64> = t.shape.iter().map(|&d| d as f64).collect();
                let n = s.len();
                upload(s, vec![n])
            }
            Ok(other) => Err(format!("shape: expected a tensor, got {}", fmt_val(other))),
            Err(e) => Err(e),
        },
        "rows" => shape_axis(args, name, 0),
        "cols" => shape_axis(args, name, 1),
        _ => return None,
    })
}

fn one_ref<'a>(args: &'a [Val], name: &str) -> Result<&'a Val, String> {
    if args.len() != 1 {
        return Err(format!("{name} expects 1 arg, got {}", args.len()));
    }
    Ok(&args[0])
}

fn shape_axis(args: &[Val], name: &str, axis: usize) -> Result<Val, String> {
    match one_ref(args, name)? {
        Val::Tensor(t) => Ok(Val::Num(*t.shape.get(axis).unwrap_or(&1) as f64)),
        other => Err(format!("{name}: expected a tensor, got {}", fmt_val(other))),
    }
}

/// Flatten all numeric leaves of the args (so min/max accept scalars and tuples).
fn num_leaves(args: Vec<Val>, name: &str) -> Result<Vec<f64>, String> {
    fn rec(v: Val, name: &str, out: &mut Vec<f64>) -> Result<(), String> {
        match v {
            Val::Num(x) => { out.push(x); Ok(()) }
            Val::Tuple(items) => { for it in items { rec(it, name, out)?; } Ok(()) }
            other => Err(format!("{name}: expected numbers, got {}", fmt_val(&other))),
        }
    }
    let mut out = vec![];
    for a in args { rec(a, name, &mut out)?; }
    if out.is_empty() {
        return Err(format!("{name}: no numeric arguments"));
    }
    Ok(out)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a / gcd(a, b) * b }
}

/// `A @ B` / `matmul(A, B)`: 2D×2D, 2D×1D (mat·vec), 1D×2D (vec·mat), 1D×1D (dot).
fn matmul_vals(a: Val, b: Val, env: &Env) -> Result<Val, String> {
    let (ta, tb) = match (a, b) {
        (Val::Tensor(x), Val::Tensor(y)) => (x, y),
        (x, y) => return Err(format!("matmul (@): both operands must be tensors, got {} and {}", fmt_val(&x), fmt_val(&y))),
    };
    let ta = ensure_on(ta, env.target)?;
    let tb = ensure_on(tb, env.target)?;
    let bad = |k: usize, k2: usize| format!("matmul: inner dimensions differ ({k} vs {k2})");
    match (ta.shape.as_slice(), tb.shape.as_slice()) {
        ([m, k], [k2, n]) => {
            if k != k2 { return Err(bad(*k, *k2)); }
            compute::matmul(env.target, &ta, &tb, *m, *k, *n).map(Val::Tensor)
        }
        ([m, k], [k2]) => {
            if k != k2 { return Err(bad(*k, *k2)); }
            let mut out = compute::matmul(env.target, &ta, &tb, *m, *k, 1)?;
            out.shape = vec![*m];
            Ok(Val::Tensor(out))
        }
        ([k], [k2, n]) => {
            if k != k2 { return Err(bad(*k, *k2)); }
            let mut out = compute::matmul(env.target, &ta, &tb, 1, *k, *n)?;
            out.shape = vec![*n];
            Ok(Val::Tensor(out))
        }
        ([k], [k2]) => {
            if k != k2 { return Err(bad(*k, *k2)); }
            let out = compute::matmul(env.target, &ta, &tb, 1, *k, 1)?;
            Ok(Val::Num(compute::download(&out)?[0]))
        }
        (ash, bsh) => Err(format!("matmul: unsupported shapes {ash:?} @ {bsh:?}")),
    }
}

/// Coerce a value to a real tensor on the active target.
fn real_tensor_on(v: &Val, env: &Env, name: &str) -> Result<TensorVal, String> {
    match v {
        Val::Tensor(t) => ensure_on(t.clone(), env.target),
        other => Err(format!("{name}: expected a real tensor, got {}", fmt_val(other))),
    }
}

/// Download a real square matrix to host (data, n, n).
fn real_square(v: &Val, name: &str) -> Result<(Vec<f64>, usize, usize), String> {
    match v {
        Val::Tensor(t) if t.shape.len() == 2 && t.shape[0] == t.shape[1] => {
            Ok((compute::download(t)?, t.shape[0], t.shape[1]))
        }
        Val::Tensor(t) => Err(format!("{name}: expected a square matrix, got shape {:?}", t.shape)),
        other => Err(format!("{name}: expected a real matrix, got {}", fmt_val(other))),
    }
}

/// Determinant via Gaussian elimination with partial pivoting.
fn host_det(m: &mut [f64], n: usize) -> f64 {
    let mut det = 1.0;
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if m[piv * n + col] == 0.0 {
            return 0.0;
        }
        if piv != col {
            for cc in 0..n {
                m.swap(piv * n + cc, col * n + cc);
            }
            det = -det;
        }
        det *= m[col * n + col];
        for r in (col + 1)..n {
            let f = m[r * n + col] / m[col * n + col];
            for cc in col..n {
                m[r * n + cc] -= f * m[col * n + cc];
            }
        }
    }
    det
}

/// Matrix inverse via Gauss–Jordan elimination.
fn host_inv(mut m: Vec<f64>, n: usize) -> Result<Vec<f64>, String> {
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if m[piv * n + col].abs() == 0.0 {
            return Err("inv: matrix is singular".into());
        }
        if piv != col {
            for cc in 0..n {
                m.swap(piv * n + cc, col * n + cc);
                inv.swap(piv * n + cc, col * n + cc);
            }
        }
        let d = m[col * n + col];
        for cc in 0..n {
            m[col * n + cc] /= d;
            inv[col * n + cc] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            for cc in 0..n {
                m[r * n + cc] -= f * m[col * n + cc];
                inv[r * n + cc] -= f * inv[col * n + cc];
            }
        }
    }
    Ok(inv)
}

/// Solve Ax = b via Gaussian elimination with back-substitution.
fn host_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>, String> {
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() == 0.0 {
            return Err("solve: matrix is singular".into());
        }
        if piv != col {
            for cc in 0..n {
                a.swap(piv * n + cc, col * n + cc);
            }
            b.swap(piv, col);
        }
        for r in (col + 1)..n {
            let f = a[r * n + col] / a[col * n + col];
            for cc in col..n {
                a[r * n + cc] -= f * a[col * n + cc];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for cc in (col + 1)..n {
            s -= a[col * n + cc] * x[cc];
        }
        x[col] = s / a[col * n + col];
    }
    Ok(x)
}

/// Population standard deviation of a tensor: sqrt(mean((x − mean)²)) — all on device.
fn tensor_std(env: &Env, t: &TensorVal) -> Result<Val, String> {
    let n = t.len.max(1) as f64;
    let m = compute::reduce(env.target, compute::RED_SUM, t)? / n;
    let mean_t = compute::upload(env.target, &[m], vec![1])?;
    let dev = compute::binop(env.target, compute::OP_SUB, t, &mean_t)?;
    let sq = compute::binop(env.target, compute::OP_MUL, &dev, &dev)?;
    let var = compute::reduce(env.target, compute::RED_SUM, &sq)? / n;
    Ok(Val::Num(var.sqrt()))
}
