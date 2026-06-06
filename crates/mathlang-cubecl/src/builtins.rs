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
