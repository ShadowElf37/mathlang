//! Scalar/complex builtins for the Phase 1b host core. Unary math broadcasts over
//! tuple trees (matching the original's leaf semantics). Tensor/linalg/fft/field
//! builtins are deferred to later phases.

use crate::ast::Op;
use crate::compute::{self, TensorVal};
use crate::interp::{apply_val, binop_val, ensure_on, to_tensor_on, Env};
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
            // complex-projection ops on a *real* tensor: re/conj are identity,
            // im is zeros, arg is 0 where x≥0 and π where x<0.
            match name {
                "re" | "conj" => return Ok(Val::Tensor(t.clone())),
                "im" => {
                    let z = compute::upload(env.target, &[0.0], vec![1])?;
                    return compute::binop(env.target, compute::OP_MUL, t, &z).map(Val::Tensor);
                }
                "arg" => {
                    let z = compute::upload(env.target, &[0.0], vec![1])?;
                    let lt = compute::binop(env.target, compute::OP_LT, t, &z)?;
                    let pi = compute::upload(env.target, &[std::f64::consts::PI], vec![1])?;
                    return compute::binop(env.target, compute::OP_MUL, &lt, &pi).map(Val::Tensor);
                }
                _ => {}
            }
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
        // 2-arg elementwise on tensors; otherwise reduce numeric leaves on the host.
        // (A single tensor arg is handled by the whole-tensor reduction above.)
        "min" | "max" => {
            let elementwise = args.len() == 2 && args.iter().any(|a| matches!(a, Val::Tensor(_)));
            if elementwise {
                let code = if name == "min" { compute::OP_MIN } else { compute::OP_MAX };
                let mut it = args.into_iter();
                let a = to_tensor_on(it.next().unwrap(), env.target)?;
                let b = to_tensor_on(it.next().unwrap(), env.target)?;
                compute::binop(env.target, code, &a, &b).map(Val::Tensor)
            } else {
                let leaves = num_leaves(args, name)?;
                let init = if name == "min" { f64::INFINITY } else { f64::NEG_INFINITY };
                let folded = if name == "min" {
                    leaves.into_iter().fold(init, f64::min)
                } else {
                    leaves.into_iter().fold(init, f64::max)
                };
                Ok(Val::Num(folded))
            }
        }

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
        "trace" => {
            let (m, n, _) = real_square(&one(args, name)?, name)?;
            Ok(Val::Num((0..n).map(|i| m[i * n + i]).sum()))
        }
        "eigvals" => {
            let (m, n, _) = real_square(&one(args, name)?, name)?;
            let (lams, _) = eig_qr(&m, n);
            compute::upload(env.target, &lams, vec![n]).map(Val::Tensor)
        }
        "eig" => {
            let (m, n, _) = real_square(&one(args, name)?, name)?;
            let (lams, evecs) = eig_qr(&m, n);
            let lam = compute::upload(env.target, &lams, vec![n])?;
            let vecs = compute::upload(env.target, &evecs, vec![n, n])?;
            Ok(Val::Tuple(vec![Val::Tensor(lam), Val::Tensor(vecs)]))
        }

        // ── calculus ────────────────────────────────────────────────────────────
        "deriv" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("deriv(f, x[, dx]) or deriv(f, point[, axis])".into());
            }
            let f = args[0].clone();
            match &args[1] {
                // scalar: 5-point stencil (3rd arg = dx)
                Val::Num(x) => {
                    let x = *x;
                    let dx = match args.get(2) { Some(v) => v.clone().num("dx")?, None => 1e-5 };
                    Ok(Val::Num(deriv_scalar(&f, x, dx, env)?))
                }
                // multivariate: partial along an axis, or the full gradient
                _ => {
                    let p = flatten_point(&args[1], "deriv")?;
                    let dx = 1e-5;
                    match args.get(2) {
                        Some(axv) => {
                            let axis = axv.clone().num("axis")? as usize;
                            if axis >= p.len() {
                                return Err(format!("deriv: axis {axis} out of range for a {}-vector", p.len()));
                            }
                            Ok(Val::Num(grad_partial(&f, &p, axis, dx, env)?))
                        }
                        None => {
                            let g: Vec<f64> = (0..p.len()).map(|i| grad_partial(&f, &p, i, dx, env)).collect::<Result<_, _>>()?;
                            let n = g.len();
                            compute::upload(env.target, &g, vec![n]).map(Val::Tensor)
                        }
                    }
                }
            }
        }
        "integral" => {
            if args.len() < 3 || args.len() > 4 {
                return Err("integral(f, a, b[, n]) or integral(f, lo, hi[, n]) for a box".into());
            }
            let f = args[0].clone();
            match &args[1] {
                // scalar: composite Simpson (default n=1000)
                Val::Num(a) => {
                    let a = *a;
                    let b = args[2].clone().num("b")?;
                    let n = match args.get(3) { Some(v) => v.clone().num("n")? as usize, None => 1000 };
                    Ok(Val::Num(integ_scalar(&f, a, b, n, env)?))
                }
                // multidim: tensor-product Simpson over a box (default n=64 per axis)
                _ => {
                    let lo = flatten_point(&args[1], "integral")?;
                    let hi = flatten_point(&args[2], "integral")?;
                    if lo.len() != hi.len() {
                        return Err("integral: lo and hi must have the same length".into());
                    }
                    let n = match args.get(3) { Some(v) => v.clone().num("n")? as usize, None => 64 };
                    Ok(Val::Num(integ_nd(&f, &lo, &hi, n, env)?))
                }
            }
        }

        // ── spectral ────────────────────────────────────────────────────────────
        "fft" | "ifft" => {
            if args.is_empty() || args.len() > 2 {
                return Err(format!("{name}(T[, axes]) expects 1 or 2 args"));
            }
            let forward = name == "fft";
            let (mut re, mut im, shape) = as_complex_input(&args[0], name)?;
            let ndim = shape.len();
            let axes: Vec<usize> = match args.get(1) {
                None => (0..ndim).collect(),
                Some(Val::Num(n)) => vec![*n as usize],
                Some(Val::Tensor(t)) => compute::download(t)?.iter().map(|&x| x as usize).collect(),
                Some(Val::Tuple(items)) => items.iter().map(|v| Ok(v.clone().num(name)? as usize)).collect::<Result<_, String>>()?,
                Some(other) => return Err(format!("{name}: axes must be a number, tuple, or tensor, got {}", fmt_val(other))),
            };
            for &ax in &axes {
                if ax >= ndim {
                    return Err(format!("{name}: axis {ax} out of range for {ndim}-D tensor"));
                }
            }
            for &ax in &axes {
                fft_axis_inplace(&mut re, &mut im, &shape, ax, forward);
            }
            maybe_real_upload(env, re, im, shape)
        }
        "ops.specgrad" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("ops.specgrad(T, dx[, axis]) expects 2 or 3 args".into());
            }
            let (data, shape) = real_data(&args[0], "ops.specgrad")?;
            let dx = args[1].clone().num("ops.specgrad dx")?;
            let ndim = shape.len();
            let axis = match args.get(2) {
                Some(v) => v.clone().num("ops.specgrad axis")? as usize,
                None if ndim == 1 => 0,
                None => return Err("ops.specgrad: specify an axis for a multi-D tensor".into()),
            };
            if axis >= ndim {
                return Err(format!("ops.specgrad: axis {axis} out of range for {ndim}-D tensor"));
            }
            let out = spec_deriv(&data, &shape, dx, axis);
            compute::upload(env.target, &out, shape).map(Val::Tensor)
        }
        "ops.poisson" | "ops.invlap" => {
            if args.len() != 2 {
                return Err(format!("{name}(rhs, dx) expects 2 args"));
            }
            let (mut re, mut im, shape) = as_complex_input(&args[0], name)?;
            let dx = args[1].clone().num("dx")?;
            poisson_solve(&mut re, &mut im, &shape, dx);
            compute::upload(env.target, &re, shape).map(Val::Tensor)
        }

        // ── elementwise select (where): cond ? a : b, via masking ───────────────
        "select" => {
            if args.len() != 3 {
                return Err("select(cond, a, b) expects 3 args".into());
            }
            let mut it = args.into_iter();
            let (c, a, b) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
            let one_minus = binop_val(Val::Num(1.0), &Op::Sub, c.clone(), env.target)?;
            let ca = binop_val(c, &Op::Mul, a, env.target)?;
            let cb = binop_val(one_minus, &Op::Mul, b, env.target)?;
            binop_val(ca, &Op::Add, cb, env.target)
        }

        // ── build-by-function constructors ──────────────────────────────────────
        "tensor" | "matrix" => build_by_fn(args, name, env),
        "lingrid" => build_lingrid(args, env),

        // ── assembly ────────────────────────────────────────────────────────────
        "reshape" => {
            if args.len() < 2 {
                return Err("reshape(T, n1, ...) expects a tensor and new dims".into());
            }
            let dims: Vec<usize> = args[1..].iter().map(|v| Ok(v.clone().num(name)? as usize)).collect::<Result<_, String>>()?;
            let total: usize = dims.iter().product();
            match &args[0] {
                Val::Tensor(t) if t.len == total => { let mut t2 = t.clone(); t2.shape = dims; Ok(Val::Tensor(t2)) }
                Val::ComplexTensor(t) if t.len == total => { let mut t2 = t.clone(); t2.shape = dims; Ok(Val::ComplexTensor(t2)) }
                Val::Tensor(t) => Err(format!("reshape: {} elements can't form {dims:?}", t.len)),
                Val::ComplexTensor(t) => Err(format!("reshape: {} elements can't form {dims:?}", t.len)),
                other => Err(format!("reshape: expected a tensor, got {}", fmt_val(other))),
            }
        }
        "diag" => {
            // vector/numeric-tuple → diagonal matrix; square matrix → its diagonal
            match one(args, name)? {
                Val::Tuple(items) => {
                    let d: Vec<f64> = items.iter().map(|x| x.clone().num(name)).collect::<Result<_, _>>()?;
                    let n = d.len();
                    let mut m = vec![0.0; n * n];
                    for i in 0..n { m[i * n + i] = d[i]; }
                    compute::upload(env.target, &m, vec![n, n]).map(Val::Tensor)
                }
                Val::Tensor(t) => match t.shape.as_slice() {
                    [n] => {
                        let d = compute::download(&t)?;
                        let mut m = vec![0.0; n * n];
                        for i in 0..*n { m[i * n + i] = d[i]; }
                        compute::upload(env.target, &m, vec![*n, *n]).map(Val::Tensor)
                    }
                    [r, c] => {
                        let d = compute::download(&t)?;
                        let k = (*r).min(*c);
                        let diag: Vec<f64> = (0..k).map(|i| d[i * c + i]).collect();
                        compute::upload(env.target, &diag, vec![k]).map(Val::Tensor)
                    }
                    _ => Err("diag: expected a 1-D or 2-D tensor".into()),
                },
                other => Err(format!("diag: expected a vector or matrix, got {}", fmt_val(&other))),
            }
        }
        "transpose" => transpose_val(one(args, name)?, env),
        "cat" => {
            if args.len() < 2 {
                return Err("cat(axis, T1, T2, ...) expects an axis and tensors".into());
            }
            let axis = args[0].clone().num(name)? as usize;
            cat_vals(&args[1..], axis, env)
        }
        "vstack" => stack_vals(&args, 0, env),
        "hstack" => stack_vals(&args, 1, env),

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
        "sign" | "signum" => compute::UN_SIGN,
        "floor" => compute::UN_FLOOR,
        "ceil" => compute::UN_CEIL,
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

// ── eigenvalues / eigenvectors (unshifted Householder-QR iteration) ──────────────
// Converges for symmetric matrices to real eigenvalues + orthogonal eigenvectors;
// for non-symmetric input it returns the converged diagonal (matching the original).

fn eye_n(n: usize) -> Vec<f64> {
    let mut e = vec![0.0; n * n];
    for i in 0..n {
        e[i * n + i] = 1.0;
    }
    e
}

fn matmul_nn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Full QR via Householder reflections (n×n). Returns (Q, R).
fn qr_householder(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = a.to_vec();
    let mut q = eye_n(n);
    for k in 0..n.saturating_sub(1) {
        let len = n - k;
        let x: Vec<f64> = (k..n).map(|i| r[i * n + k]).collect();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x < 1e-14 {
            continue;
        }
        let mut hv = x;
        let sign = if hv[0] >= 0.0 { 1.0 } else { -1.0 };
        hv[0] += sign * norm_x;
        let norm_hv: f64 = hv.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_hv < 1e-14 {
            continue;
        }
        for v in &mut hv {
            *v /= norm_hv;
        }
        for j in k..n {
            let dot: f64 = (0..len).map(|i| hv[i] * r[(i + k) * n + j]).sum();
            for i in 0..len {
                r[(i + k) * n + j] -= 2.0 * hv[i] * dot;
            }
        }
        for i in 0..n {
            let dot: f64 = (0..len).map(|j| q[i * n + (j + k)] * hv[j]).sum();
            for j in 0..len {
                q[i * n + (j + k)] -= 2.0 * hv[j] * dot;
            }
        }
    }
    (q, r)
}

fn eig_qr(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ak = a.to_vec();
    let mut eigvecs = eye_n(n);
    for _ in 0..2000 {
        let (q, r) = qr_householder(&ak, n);
        ak = matmul_nn(&r, &q, n);
        eigvecs = matmul_nn(&eigvecs, &q, n);
        let off: f64 = (0..n)
            .flat_map(|i| (0..i).map(move |j| (i, j)))
            .map(|(i, j)| ak[i * n + j] * ak[i * n + j])
            .sum::<f64>()
            .sqrt();
        if off < 1e-12 {
            break;
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| ak[i * n + i]).collect();
    (eigenvalues, eigvecs)
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

/// `tensor(f, n1, ...)` / `matrix(f, r, c)` — build a tensor by calling `f` at each
/// integer multi-index (host-side; promotes to complex if `f` returns complex).
fn build_by_fn(args: Vec<Val>, name: &str, env: &Env) -> Result<Val, String> {
    if args.len() < 2 {
        return Err(format!("{name}(f, n1, ...) expects a function and dims"));
    }
    let f = args[0].clone();
    let dims: Vec<usize> = args[1..].iter().map(|v| Ok(v.clone().num(name)? as usize)).collect::<Result<_, String>>()?;
    let total: usize = dims.iter().product();
    let (mut re, mut im, mut has_c) = (Vec::with_capacity(total), Vec::with_capacity(total), false);
    let mut counter = vec![0usize; dims.len()];
    for _ in 0..total {
        let idx_args: Vec<Val> = counter.iter().map(|&c| Val::Num(c as f64)).collect();
        match apply_val(f.clone(), idx_args, env)? {
            Val::Num(x) => { re.push(x); im.push(0.0); }
            Val::Complex(a, b) => { re.push(a); im.push(b); has_c = true; }
            other => return Err(format!("{name}: f must return a number, got {}", fmt_val(&other))),
        }
        for k in (0..dims.len()).rev() {
            counter[k] += 1;
            if counter[k] < dims[k] { break; }
            counter[k] = 0;
        }
    }
    if has_c {
        compute::upload_complex(env.target, &re, &im, dims).map(Val::ComplexTensor)
    } else {
        compute::upload(env.target, &re, dims).map(Val::Tensor)
    }
}

/// `lingrid(start, end, counts, f)` — sample `f` at physical coords on a uniform grid.
fn build_lingrid(args: Vec<Val>, env: &Env) -> Result<Val, String> {
    if args.len() != 4 {
        return Err("lingrid(start, end, counts, f) expects 4 args".into());
    }
    let starts = vals_to_f64s(&args[0], "lingrid")?;
    let ends = vals_to_f64s(&args[1], "lingrid")?;
    let counts: Vec<usize> = vals_to_f64s(&args[2], "lingrid")?.iter().map(|&x| x as usize).collect();
    let ndim = counts.len();
    if starts.len() != ndim || ends.len() != ndim {
        return Err("lingrid: start/end/counts must have the same length".into());
    }
    let f = args[3].clone();
    let total: usize = counts.iter().product();
    let (mut re, mut im, mut has_c) = (Vec::with_capacity(total), Vec::with_capacity(total), false);
    let mut counter = vec![0usize; ndim];
    for _ in 0..total {
        let coords: Vec<Val> = (0..ndim).map(|a| {
            let n = counts[a];
            let x = if n <= 1 { starts[a] } else { starts[a] + (ends[a] - starts[a]) * counter[a] as f64 / (n - 1) as f64 };
            Val::Num(x)
        }).collect();
        match apply_val(f.clone(), coords, env)? {
            Val::Num(x) => { re.push(x); im.push(0.0); }
            Val::Complex(a, b) => { re.push(a); im.push(b); has_c = true; }
            other => return Err(format!("lingrid: f must return a number (tuple/tensor returns not supported yet), got {}", fmt_val(&other))),
        }
        for k in (0..ndim).rev() {
            counter[k] += 1;
            if counter[k] < counts[k] { break; }
            counter[k] = 0;
        }
    }
    if has_c {
        compute::upload_complex(env.target, &re, &im, counts).map(Val::ComplexTensor)
    } else {
        compute::upload(env.target, &re, counts).map(Val::Tensor)
    }
}

/// A scalar or numeric tuple → Vec<f64> (for lingrid's start/end/counts).
fn vals_to_f64s(v: &Val, name: &str) -> Result<Vec<f64>, String> {
    match v {
        Val::Num(x) => Ok(vec![*x]),
        Val::Tuple(items) => items.iter().map(|x| x.clone().num(name)).collect(),
        other => Err(format!("{name}: expected a scalar or tuple, got {}", fmt_val(other))),
    }
}

/// 2-D transpose (1-D is returned unchanged).
fn transpose_val(v: Val, env: &Env) -> Result<Val, String> {
    match v {
        Val::Tensor(t) => match t.shape.as_slice() {
            [_] => Ok(Val::Tensor(t)),
            [r, c] => {
                let (r, c) = (*r, *c);
                let data = compute::download(&t)?;
                let mut out = vec![0.0; data.len()];
                for i in 0..r {
                    for j in 0..c {
                        out[j * r + i] = data[i * c + j];
                    }
                }
                compute::upload(env.target, &out, vec![c, r]).map(Val::Tensor)
            }
            _ => Err("transpose: only 1-D and 2-D tensors are supported".into()),
        },
        Val::ComplexTensor(ct) => match ct.shape.as_slice() {
            [_] => Ok(Val::ComplexTensor(ct)),
            [r, c] => {
                let (r, c) = (*r, *c);
                let (re, im) = compute::download_complex(&ct)?;
                let (mut or, mut oi) = (vec![0.0; re.len()], vec![0.0; im.len()]);
                for i in 0..r {
                    for j in 0..c {
                        or[j * r + i] = re[i * c + j];
                        oi[j * r + i] = im[i * c + j];
                    }
                }
                compute::upload_complex(env.target, &or, &oi, vec![c, r]).map(Val::ComplexTensor)
            }
            _ => Err("transpose: only 1-D and 2-D tensors are supported".into()),
        },
        other => Err(format!("transpose: expected a tensor, got {}", fmt_val(&other))),
    }
}

/// Coerce a value to a real (data, shape) block: tensor, numeric tuple → 1-D, or
/// scalar → length-1. (Numeric tuples act as vectors, matching the original.)
fn as_real_block(v: &Val) -> Result<(Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Tensor(t) => Ok((compute::download(t)?, t.shape.clone())),
        Val::Tuple(items) => {
            let d: Vec<f64> = items.iter().map(|x| x.clone().num("stack")).collect::<Result<_, _>>()?;
            let n = d.len();
            Ok((d, vec![n]))
        }
        Val::Num(x) => Ok((vec![*x], vec![1])),
        other => Err(format!("expected a real tensor/vector, got {}", fmt_val(other))),
    }
}

/// Promote a block to 2-D for stacking: 1-D → row (axis 0) or column (axis 1).
fn promote_2d(block: (Vec<f64>, Vec<usize>), axis: usize) -> Result<(Vec<f64>, usize, usize), String> {
    let (d, s) = block;
    match s.as_slice() {
        [n] => Ok(if axis == 0 { (d, 1, *n) } else { (d, *n, 1) }),
        [r, c] => Ok((d, *r, *c)),
        _ => Err("stacking supports only scalars, 1-D, and 2-D tensors".into()),
    }
}

/// Concatenate along `axis`. All-1-D along axis 0 stays 1-D; otherwise blocks are
/// promoted to 2-D (numeric tuples act as vectors).
fn cat_vals(parts: &[Val], axis: usize, env: &Env) -> Result<Val, String> {
    let blocks: Vec<(Vec<f64>, Vec<usize>)> = parts.iter().map(as_real_block).collect::<Result<_, _>>()?;
    if blocks.is_empty() {
        return Err("cat: needs at least one tensor".into());
    }
    if blocks.iter().all(|(_, s)| s.len() == 1) && axis == 0 {
        let mut out = vec![];
        for (d, _) in &blocks {
            out.extend_from_slice(d);
        }
        let n = out.len();
        return compute::upload(env.target, &out, vec![n]).map(Val::Tensor);
    }
    let twod = blocks.into_iter().map(|b| promote_2d(b, axis)).collect::<Result<_, _>>()?;
    let (out, shape) = host_cat_2d(twod, axis)?;
    compute::upload(env.target, &out, shape).map(Val::Tensor)
}

/// vstack (axis 0) / hstack (axis 1) — always 2-D, with 1-D → row/column promotion.
fn stack_vals(parts: &[Val], axis: usize, env: &Env) -> Result<Val, String> {
    let twod = parts
        .iter()
        .map(|v| promote_2d(as_real_block(v)?, axis))
        .collect::<Result<_, _>>()?;
    let (out, shape) = host_cat_2d(twod, axis)?;
    compute::upload(env.target, &out, shape).map(Val::Tensor)
}

/// Concatenate row-major 2-D blocks along axis 0 (stack rows) or 1 (stack columns).
fn host_cat_2d(parts: Vec<(Vec<f64>, usize, usize)>, axis: usize) -> Result<(Vec<f64>, Vec<usize>), String> {
    if parts.is_empty() {
        return Err("cat: needs at least one tensor".into());
    }
    if axis == 0 {
        let c = parts[0].2;
        if parts.iter().any(|(_, _, pc)| *pc != c) {
            return Err("cat axis 0: all blocks must have the same number of columns".into());
        }
        let mut out = vec![];
        let mut rows = 0;
        for (d, r, _) in &parts {
            out.extend_from_slice(d);
            rows += r;
        }
        Ok((out, vec![rows, c]))
    } else if axis == 1 {
        let r = parts[0].1;
        if parts.iter().any(|(_, pr, _)| *pr != r) {
            return Err("cat axis 1: all blocks must have the same number of rows".into());
        }
        let total_c: usize = parts.iter().map(|(_, _, c)| c).sum();
        let mut out = vec![0.0; r * total_c];
        for i in 0..r {
            let mut col0 = 0;
            for (d, _, c) in &parts {
                for j in 0..*c {
                    out[i * total_c + col0 + j] = d[i * c + j];
                }
                col0 += c;
            }
        }
        Ok((out, vec![r, total_c]))
    } else {
        Err("cat: axis must be 0 or 1 for 2-D tensors".into())
    }
}

// ── spectral helpers (host FFT via rustfft; f64, any size) ──────────────────────

fn t_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut s = vec![1usize; n];
    for k in (0..n.saturating_sub(1)).rev() {
        s[k] = s[k + 1] * shape[k + 1];
    }
    s
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut idx = vec![0usize; n];
    for k in (0..n).rev() {
        idx[k] = flat % shape[k];
        flat /= shape[k];
    }
    idx
}

/// FFT frequency index: m for m ≤ n/2, else m − n (so derivatives use signed k).
fn kfreq(m: usize, n: usize) -> f64 {
    if 2 * m < n { m as f64 } else { m as f64 - n as f64 }
}

/// In-place FFT along one axis over a flat row-major (re, im) buffer.
fn fft_axis_inplace(re: &mut [f64], im: &mut [f64], shape: &[usize], axis: usize, forward: bool) {
    use rustfft::{num_complex::Complex64, FftPlanner};
    let n = shape[axis];
    let s = t_strides(shape);
    let axis_stride = s[axis];
    let other_shape: Vec<usize> = shape.iter().enumerate().filter(|&(k, _)| k != axis).map(|(_, &d)| d).collect();
    let other_strides: Vec<usize> = s.iter().enumerate().filter(|&(k, _)| k != axis).map(|(_, &st)| st).collect();
    let other_total: usize = if other_shape.is_empty() { 1 } else { other_shape.iter().product() };
    let mut planner = FftPlanner::new();
    let fft = if forward { planner.plan_fft_forward(n) } else { planner.plan_fft_inverse(n) };
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    for other_flat in 0..other_total {
        let other_multi = unravel(other_flat, &other_shape);
        let base: usize = other_multi.iter().zip(&other_strides).map(|(&i, &st)| i * st).sum();
        for i in 0..n {
            let f = base + i * axis_stride;
            buf[i] = Complex64::new(re[f], im[f]);
        }
        fft.process(&mut buf);
        if !forward {
            let sc = 1.0 / n as f64;
            for c in &mut buf {
                *c *= sc;
            }
        }
        for i in 0..n {
            let f = base + i * axis_stride;
            re[f] = buf[i].re;
            im[f] = buf[i].im;
        }
    }
}

fn fftn(re: &mut [f64], im: &mut [f64], shape: &[usize], forward: bool) {
    for a in 0..shape.len() {
        fft_axis_inplace(re, im, shape, a, forward);
    }
}

/// Real tensor → (data, shape).
fn real_data(v: &Val, name: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Tensor(t) => Ok((compute::download(t)?, t.shape.clone())),
        other => Err(format!("{name}: expected a real tensor, got {}", fmt_val(other))),
    }
}

/// Real or complex tensor → (re, im, shape) for FFT input.
fn as_complex_input(v: &Val, name: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Tensor(t) => {
            let re = compute::download(t)?;
            let im = vec![0.0; re.len()];
            Ok((re, im, t.shape.clone()))
        }
        Val::ComplexTensor(ct) => {
            let (re, im) = compute::download_complex(ct)?;
            Ok((re, im, ct.shape.clone()))
        }
        other => Err(format!("{name}: expected a tensor, got {}", fmt_val(other))),
    }
}

/// Upload (re, im) as a complex tensor, collapsing to real if every im is exactly 0.
fn maybe_real_upload(env: &Env, re: Vec<f64>, im: Vec<f64>, shape: Vec<usize>) -> Result<Val, String> {
    if im.iter().all(|&x| x == 0.0) {
        compute::upload(env.target, &re, shape).map(Val::Tensor)
    } else {
        compute::upload_complex(env.target, &re, &im, shape).map(Val::ComplexTensor)
    }
}

/// Spectral derivative along `axis`: FFT → ×ik → IFFT, real part.
fn spec_deriv(data: &[f64], shape: &[usize], dx: f64, axis: usize) -> Vec<f64> {
    let mut re = data.to_vec();
    let mut im = vec![0.0; data.len()];
    fftn(&mut re, &mut im, shape, true);
    let n_ax = shape[axis];
    for p in 0..re.len() {
        let multi = unravel(p, shape);
        let k = kfreq(multi[axis], n_ax) * 2.0 * std::f64::consts::PI / (n_ax as f64 * dx);
        let nr = -k * im[p];
        let ni = k * re[p];
        re[p] = nr;
        im[p] = ni;
    }
    fftn(&mut re, &mut im, shape, false);
    re
}

/// Spectral Poisson solve ∇²u = rhs (zero-mean): FFT → −r̂/k² → IFFT, real part.
fn poisson_solve(re: &mut [f64], im: &mut [f64], shape: &[usize], dx: f64) {
    fftn(re, im, shape, true);
    let ndim = shape.len();
    for p in 0..re.len() {
        let multi = unravel(p, shape);
        let mut k2 = 0.0;
        for a in 0..ndim {
            let k = kfreq(multi[a], shape[a]) * 2.0 * std::f64::consts::PI / (shape[a] as f64 * dx);
            k2 += k * k;
        }
        if k2 == 0.0 {
            re[p] = 0.0;
            im[p] = 0.0;
        } else {
            re[p] = -re[p] / k2;
            im[p] = -im[p] / k2;
        }
    }
    fftn(re, im, shape, false);
}

// ── calculus helpers (host; functions are evaluated pointwise) ──────────────────

/// Flatten an integration/derivative point (scalar, numeric tuple, or 1-D tensor).
fn flatten_point(v: &Val, name: &str) -> Result<Vec<f64>, String> {
    match v {
        Val::Num(x) => Ok(vec![*x]),
        Val::Tuple(items) => items.iter().map(|x| x.clone().num(name)).collect(),
        Val::Tensor(t) if t.shape.len() == 1 => compute::download(t),
        other => Err(format!("{name}: expected a scalar, vector, or tuple point, got {}", fmt_val(other))),
    }
}

/// 5-point central-difference derivative of a 1-arg function.
fn deriv_scalar(f: &Val, x: f64, dx: f64, env: &Env) -> Result<f64, String> {
    let at = |t: f64| apply_val(f.clone(), vec![Val::Num(t)], env)?.num("deriv f");
    Ok((-at(x + 2.0 * dx)? + 8.0 * at(x + dx)? - 8.0 * at(x - dx)? + at(x - 2.0 * dx)?) / (12.0 * dx))
}

/// Partial ∂f/∂xᵢ at `p` (5-point). The point is passed to `f` as a tuple, which
/// destructures into multiple params or is indexed as a vector — either calling style.
fn grad_partial(f: &Val, p: &[f64], i: usize, dx: f64, env: &Env) -> Result<f64, String> {
    let at = |d: f64| -> Result<f64, String> {
        let mut q: Vec<Val> = p.iter().map(|&x| Val::Num(x)).collect();
        q[i] = Val::Num(p[i] + d);
        apply_val(f.clone(), vec![Val::Tuple(q)], env)?.num("deriv f")
    };
    Ok((-at(2.0 * dx)? + 8.0 * at(dx)? - 8.0 * at(-dx)? + at(-2.0 * dx)?) / (12.0 * dx))
}

/// Composite Simpson's rule for a 1-arg function over [a, b].
fn integ_scalar(f: &Val, a: f64, b: f64, n: usize, env: &Env) -> Result<f64, String> {
    let n = n + n % 2;
    let h = (b - a) / n as f64;
    let at = |t: f64| apply_val(f.clone(), vec![Val::Num(t)], env)?.num("integral f");
    let mut s = at(a)? + at(b)?;
    for i in 1..n {
        s += at(a + i as f64 * h)? * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    Ok(s * h / 3.0)
}

fn simpson_w(i: usize, n: usize) -> f64 {
    if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }
}

/// Tensor-product Simpson over the box ∏[loₐ, hiₐ] with `n` steps per axis.
fn integ_nd(f: &Val, lo: &[f64], hi: &[f64], n: usize, env: &Env) -> Result<f64, String> {
    let n = n + n % 2;
    let ndim = lo.len();
    let h: Vec<f64> = (0..ndim).map(|a| (hi[a] - lo[a]) / n as f64).collect();
    let pts = n + 1;
    let total = pts.pow(ndim as u32);
    let mut sum = 0.0;
    for flat in 0..total {
        let mut idx = flat;
        let mut w = 1.0;
        let mut coords = Vec::with_capacity(ndim);
        for a in 0..ndim {
            let i = idx % pts;
            idx /= pts;
            coords.push(Val::Num(lo[a] + i as f64 * h[a]));
            w *= simpson_w(i, n);
        }
        sum += w * apply_val(f.clone(), vec![Val::Tuple(coords)], env)?.num("integral f")?;
    }
    let factor: f64 = h.iter().map(|hh| hh / 3.0).product();
    Ok(sum * factor)
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
