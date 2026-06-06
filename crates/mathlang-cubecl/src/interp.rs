//! The single host interpreter: a tree-walk over the AST. It owns scoping,
//! closures, namespaces, control flow, and dynamic dispatch; scalar and complex
//! values are computed directly in host f64 so the REPL stays instant (the
//! low-latency invariant). Array/tensor work will delegate to `compute::*` in
//! Phase 2 — there is no bytecode VM and no `GPU {}` path.
//!
//! Behaviour mirrors `src/eval.rs`. Tensor-producing constructs (`[...]`, matrix
//! literals, `range`, tensor builtins) error with a clear "Phase 2" message until
//! the compute path lands.

use crate::ast::{BlockStmt, Def, Expr, Op};
use crate::builtins::eval_builtin;
use crate::compute::{self, CTensor, Target, TensorVal};
use crate::value::{fmt_val, int, make_complex, to_complex, FnSig, Val};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Env {
    pub vars: Arc<HashMap<String, Val>>,
    /// The active compute target (backend × precision). Carried in the env so
    /// builtins/operators reach it without a global; updated by the REPL.
    pub target: Target,
}

impl Env {
    #[inline]
    pub fn define(&mut self, k: String, v: Val) {
        Arc::make_mut(&mut self.vars).insert(k, v);
    }

    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert("pi".into(), Val::Num(std::f64::consts::PI));
        vars.insert("e".into(), Val::Num(std::f64::consts::E));
        vars.insert("phi".into(), Val::Num(1.618033988749895));
        vars.insert("inf".into(), Val::Num(f64::INFINITY));
        vars.insert("i".into(), Val::Complex(0.0, 1.0));
        for name in BUILTINS {
            vars.insert((*name).into(), Val::Builtin((*name).into()));
        }
        // The `ops` namespace: finite-difference operators + boundary-condition markers.
        let mut ops = HashMap::new();
        ops.insert("lap".to_string(), Val::Builtin("ops.lap".into()));
        ops.insert("grad".to_string(), Val::Builtin("ops.grad".into()));
        ops.insert("specgrad".to_string(), Val::Builtin("ops.specgrad".into()));
        ops.insert("poisson".to_string(), Val::Builtin("ops.poisson".into()));
        ops.insert("invlap".to_string(), Val::Builtin("ops.invlap".into()));
        ops.insert("periodic".to_string(), Val::Num(0.0));
        ops.insert("neumann".to_string(), Val::Num(1.0));
        vars.insert("ops".into(), Val::Namespace(Arc::new(ops)));
        // The `forms` namespace: exterior calculus on fields + BC markers.
        let mut forms = HashMap::new();
        for m in ["d", "hodge", "wedge", "raise", "lower", "codiff", "laplace", "form", "vector", "contract"] {
            forms.insert(m.to_string(), Val::Builtin(format!("forms.{m}")));
        }
        forms.insert("periodic".to_string(), Val::Num(0.0));
        forms.insert("neumann".to_string(), Val::Num(1.0));
        vars.insert("forms".into(), Val::Namespace(Arc::new(forms)));
        // The `pic` namespace: particle-in-cell scatter/gather + kernel sentinels.
        let pic = crate::pic::members();
        vars.insert("pic".into(), Val::Namespace(Arc::new(pic)));
        Self { vars: Arc::new(vars), target: Target::default_target() }
    }
}

/// Builtins implemented in the Phase 1b host core. Tensor/linalg/fft/field
/// builtins are intentionally absent until the compute path lands.
pub const BUILTINS: &[&str] = &[
    // scalar/complex unary
    "abs", "re", "im", "arg", "conj", "sqrt", "cbrt", "exp", "expm1", "ln",
    "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sec", "csc", "cot",
    "floor", "ceil", "round", "trunc", "frac", "sign", "signum", "heaviside",
    "deg", "rad", "id", "fact", "factorial",
    // binary scalar
    "log", "pow", "atan2", "min", "max", "hypot", "gcd", "lcm", "ncr",
    // comparison fns
    "lt", "leq", "gt", "geq", "eq", "neq",
    // higher-order + containers
    "map", "filter", "reduce", "compose", "partial",
    "sum", "prod", "iterate", "scan", "if",
    // calculus
    "integral", "deriv",
    // spectral
    "fft", "ifft",
    "len", "length", "cell", "get", "set",
    // file I/O (.npy / .mlt / .h5)
    "save", "load",
    // animation (stream 2-D frames to wgpu_animator via MXFR)
    "animate2D", "animate2D_raw", "animate2Dforever",
    // tensor constructors / shape (the compute path)
    "zeros", "ones", "eye", "linspace", "range", "shape", "rows", "cols",
    "tensor", "matrix", "lingrid", "diag",
    // assembly
    "reshape", "transpose", "cat", "vstack", "hstack",
    // elementwise branching
    "select",
    // linear algebra + reductions
    "matmul", "norm", "mean", "std", "det", "inv", "solve", "eig", "eigvals", "trace",
    // stencils
    "shift", "roll",
    // fields & forms
    "field",
];

pub fn is_protected(name: &str) -> bool {
    matches!(name, "pi" | "e" | "phi" | "inf" | "i" | "ops" | "forms" | "pic") || BUILTINS.contains(&name)
}

// ── Evaluator ───────────────────────────────────────────────────────────────────

pub fn eval(expr: &Expr, env: &Env) -> Result<Val, String> {
    match expr {
        Expr::Num(n) => Ok(Val::Num(*n)),
        Expr::ImagLit(n) => Ok(if *n == 0.0 { Val::Num(0.0) } else { Val::Complex(0.0, *n) }),
        Expr::StrLit(s) => Ok(Val::Str(s.clone())),
        Expr::Var(n) => env.vars.get(n).cloned().ok_or_else(|| format!("undefined: {n}")),
        Expr::Lambda(params, ret_hint, body) => {
            let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
            let sig = FnSig {
                params: params.iter().map(|p| p.hint.clone()).collect(),
                ret: ret_hint.clone(),
            };
            Ok(Val::make_fn_with_sig(names, sig, (**body).clone(), Arc::clone(&env.vars)))
        }
        Expr::Tuple(exprs) => {
            let vals = exprs.iter().map(|e| eval(e, env)).collect::<Result<Vec<_>, _>>()?;
            Ok(Val::Tuple(vals))
        }
        Expr::Neg(e) => neg_val(eval(e, env)?, env.target),
        Expr::Not(e) => not_val(eval(e, env)?),
        Expr::BinOp(l, op, r) => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
            binop_val(lv, op, rv, env.target)
        }
        Expr::Block(stmts) => eval_block(stmts, env),
        Expr::Member(base, field) => {
            let base_val = eval(base, env)?;
            match base_val {
                Val::Namespace(map) => map.get(field).cloned().ok_or_else(|| {
                    let ns = match base.as_ref() { Expr::Var(n) => n.as_str(), _ => "namespace" };
                    format!("{ns} has no member '{field}'")
                }),
                other => Err(format!("'.{field}': expected a namespace, got {}", fmt_val(&other))),
            }
        }
        Expr::Index(base, idx) => {
            let v = eval(base, env)?;
            match v {
                Val::Tuple(items) => {
                    let i = eval(idx, env)?.num("index")? as i64;
                    let i = norm_index(i, items.len(), "tuple")?;
                    items.into_iter().nth(i).ok_or_else(|| "index out of range".into())
                }
                Val::Tensor(t) => {
                    let data = compute::download(&t)?;
                    let (lin, out_shape) = resolve_index(&t.shape, idx, env)?;
                    let out: Vec<f64> = lin.iter().map(|&i| data[i]).collect();
                    if out_shape.is_empty() {
                        Ok(Val::Num(out[0]))
                    } else {
                        compute::upload(env.target, &out, out_shape).map(Val::Tensor)
                    }
                }
                Val::ComplexTensor(ct) => {
                    let (re, im) = compute::download_complex(&ct)?;
                    let (lin, out_shape) = resolve_index(&ct.shape, idx, env)?;
                    let or: Vec<f64> = lin.iter().map(|&i| re[i]).collect();
                    let oi: Vec<f64> = lin.iter().map(|&i| im[i]).collect();
                    if out_shape.is_empty() {
                        Ok(make_complex(or[0], oi[0]))
                    } else {
                        compute::upload_complex(env.target, &or, &oi, out_shape).map(Val::ComplexTensor)
                    }
                }
                _ => Err("indexing requires a tuple or tensor".into()),
            }
        }
        Expr::Apply(f_expr, arg_exprs) => eval_apply(f_expr, arg_exprs, env),
        Expr::Array(exprs) => {
            // [a, b, c] — a 1-D tensor; promotes to complex if any element is complex.
            let mut re = Vec::with_capacity(exprs.len());
            let mut im = Vec::with_capacity(exprs.len());
            let mut has_complex = false;
            for e in exprs {
                match eval(e, env)? {
                    Val::Num(x) => { re.push(x); im.push(0.0); }
                    Val::Complex(a, b) => { re.push(a); im.push(b); has_complex = true; }
                    other => return Err(format!("[] requires numeric elements, got {}", fmt_val(&other))),
                }
            }
            let n = re.len();
            if has_complex {
                compute::upload_complex(env.target, &re, &im, vec![n]).map(Val::ComplexTensor)
            } else {
                compute::upload(env.target, &re, vec![n]).map(Val::Tensor)
            }
        }
        Expr::TensorLit(rows) => {
            // (1,2; 3,4) — a 2-D real tensor (row-major).
            if rows.is_empty() {
                return compute::upload(env.target, &[], vec![0, 0]).map(Val::Tensor);
            }
            let (r, c) = (rows.len(), rows[0].len());
            let mut re = Vec::with_capacity(r * c);
            let mut im = Vec::with_capacity(r * c);
            let mut has_complex = false;
            for (ri, row) in rows.iter().enumerate() {
                if row.len() != c {
                    return Err(format!("matrix row {ri} has {} elements, expected {c}", row.len()));
                }
                for e in row {
                    match eval(e, env)? {
                        Val::Num(x) => { re.push(x); im.push(0.0); }
                        Val::Complex(a, b) => { re.push(a); im.push(b); has_complex = true; }
                        other => return Err(format!("matrix literal: expected a number, got {}", fmt_val(&other))),
                    }
                }
            }
            if has_complex {
                compute::upload_complex(env.target, &re, &im, vec![r, c]).map(Val::ComplexTensor)
            } else {
                compute::upload(env.target, &re, vec![r, c]).map(Val::Tensor)
            }
        }
        Expr::Range(start, end) => {
            // `a..b` — inclusive integer range as a 1-D tensor (matches the original).
            let a = eval(start, env)?.num("range")? as i64;
            let b = eval(end, env)?.num("range")? as i64;
            let data: Vec<f64> = if a <= b {
                (a..=b).map(|n| n as f64).collect()
            } else {
                (b..=a).rev().map(|n| n as f64).collect()
            };
            let n = data.len();
            compute::upload(env.target, &data, vec![n]).map(Val::Tensor)
        }
        Expr::Slice(..) => Err("slice expression can only appear inside T[…]".into()),
    }
}

fn eval_block(stmts: &[BlockStmt], env: &Env) -> Result<Val, String> {
    let mut child = env.clone();
    let mut last = Val::Tuple(vec![]);
    for stmt in stmts {
        match stmt {
            BlockStmt::Def(def) => define_into(&mut child, def)?,
            BlockStmt::Expr(e) => last = eval(e, &child)?,
        }
    }
    Ok(last)
}

/// Install a definition into `env`, with self-capture so functions can recurse.
pub fn define_into(env: &mut Env, def: &Def) -> Result<(), String> {
    match def {
        Def::Var(name, expr) => {
            if is_protected(name) {
                return Err(format!("cannot redefine built-in '{name}'"));
            }
            let v = eval(expr, env)?;
            env.define(name.clone(), v);
        }
        Def::Func(name, params, ret_hint, body) => {
            if is_protected(name) {
                return Err(format!("cannot redefine built-in '{name}'"));
            }
            let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
            let sig = FnSig {
                params: params.iter().map(|p| p.hint.clone()).collect(),
                ret: ret_hint.clone(),
            };
            // Capture the current scope plus the function itself (for recursion).
            let mut captured = (*env.vars).clone();
            let provisional = Val::make_fn_with_sig(names.clone(), sig.clone(), body.clone(), Arc::new(captured.clone()));
            captured.insert(name.clone(), provisional);
            env.define(name.clone(), Val::make_fn_with_sig(names, sig, body.clone(), Arc::new(captured)));
        }
    }
    Ok(())
}

// ── Application (special forms first, then generic) ───────────────────────────────

fn eval_apply(f_expr: &Expr, arg_exprs: &[Expr], env: &Env) -> Result<Val, String> {
    if let Expr::Var(name) = f_expr {
        match name.as_str() {
            "if" => {
                if arg_exprs.len() != 3 {
                    return Err("if(cond, a, b) expects 3 args".into());
                }
                let cond = eval(&arg_exprs[0], env)?.num("if")?;
                return if cond != 0.0 { eval(&arg_exprs[1], env) } else { eval(&arg_exprs[2], env) };
            }
            "sum" => return eval_agg(arg_exprs, env, false),
            "prod" => return eval_agg(arg_exprs, env, true),
            "iterate" => return eval_iterate(arg_exprs, env),
            "scan" => return eval_scan(arg_exprs, env),
            "map" => return eval_map(arg_exprs, env),
            "filter" => return eval_filter(arg_exprs, env),
            "reduce" => return eval_reduce(arg_exprs, env),
            _ => {}
        }
    }
    let f_val = eval(f_expr, env)?;
    let args = arg_exprs.iter().map(|a| eval(a, env)).collect::<Result<Vec<_>, _>>()?;
    apply_val(f_val, args, env)
}

pub fn apply_val(f: Val, args: Vec<Val>, env: &Env) -> Result<Val, String> {
    match f {
        Val::Builtin(ref name) => eval_builtin(name, args, env),
        Val::Fn { ref params, ref body, ref captured, ref sig } => {
            let n = params.len();
            let k = args.len();
            if k == 0 && n > 0 {
                return Err(format!("function expects {n} arg(s), got 0"));
            }
            if k == 1 {
                if let Val::Fn { .. } = &args[0] {
                    let g = args.into_iter().next().unwrap();
                    return Ok(compose_fns(f, g));
                }
                if let Val::Tuple(items) = &args[0] {
                    if items.len() == n {
                        return apply_fn_direct(params, sig, body, captured, items.clone(), env);
                    }
                }
                if n == 1 {
                    return apply_fn_direct(params, sig, body, captured, args, env);
                }
                return Err(format!("function expects {n} args, got 1"));
            }
            if k == n {
                return apply_fn_direct(params, sig, body, captured, args, env);
            }
            // k args, all n-tuples → map with destructuring.
            let all_n_seqs = k > 0 && args.iter().all(|a| matches!(a, Val::Tuple(v) if v.len() == n));
            if all_n_seqs {
                let res: Result<Vec<Val>, _> = args
                    .into_iter()
                    .map(|a| match a {
                        Val::Tuple(v) => apply_fn_direct(params, sig, body, captured, v, env),
                        _ => unreachable!(),
                    })
                    .collect();
                return Ok(Val::Tuple(res?));
            }
            if n == 1 {
                let res: Result<Vec<Val>, _> = args
                    .into_iter()
                    .map(|a| apply_fn_direct(params, sig, body, captured, vec![a], env))
                    .collect();
                return Ok(Val::Tuple(res?));
            }
            Err(format!("function expects {n} args, got {k}"))
        }
        Val::Num(s) => apply_num(s, args),
        Val::Complex(a, b) => {
            if args.len() == 1 {
                let (ra, rb) = to_complex(args.into_iter().next().unwrap())?;
                return Ok(make_complex(a * ra - b * rb, a * rb + b * ra));
            }
            Err("complex: apply expects 1 arg".into())
        }
        Val::Tuple(items) => {
            if args.len() == 1 {
                let i = args.into_iter().next().unwrap().num("index")? as usize;
                return items.into_iter().nth(i).ok_or_else(|| format!("index {i} out of range"));
            }
            Err("tuple apply: expected a single index".into())
        }
        Val::Tensor(..) | Val::ComplexTensor(..) => Err("tensors are not callable".into()),
        Val::Field(..) => Err("fields are not callable".into()),
        Val::Str(..) => Err("strings are not callable".into()),
        Val::Cell(..) => Err("cells are not callable (use get/set)".into()),
        Val::Namespace(..) => Err("namespaces are not callable (use ns.member)".into()),
    }
}

/// Scalar-as-function: multiplication / scaling, matching the original.
fn apply_num(s: f64, args: Vec<Val>) -> Result<Val, String> {
    if args.len() == 1 {
        return match args.into_iter().next().unwrap() {
            Val::Fn { .. } => Err("scaling a function value is deferred in the prototype".into()),
            Val::Num(n) => Ok(Val::Num(s * n)),
            Val::Complex(a, b) => Ok(make_complex(s * a, s * b)),
            Val::Tuple(items) => Ok(Val::Tuple(
                items.into_iter().map(|v| match v {
                    Val::Num(n) => Val::Num(s * n),
                    other => other,
                }).collect(),
            )),
            Val::Tensor(..) | Val::ComplexTensor(..) => Err("scale a tensor with `*` (e.g. 2 * T), not juxtaposition".into()),
            Val::Field(..) => Err("scale a field with `*` (e.g. 2 * f), not juxtaposition".into()),
            Val::Builtin(_) => Err("cannot scale a builtin function".into()),
            Val::Str(..) => Err("cannot scale a string".into()),
            Val::Cell(..) => Err("cannot scale a cell (use get/set)".into()),
            Val::Namespace(..) => Err("cannot scale a namespace".into()),
        };
    }
    let nums: Result<Vec<f64>, _> = args.into_iter().map(|v| v.num("scalar-apply")).collect();
    Ok(Val::Num(nums?.iter().fold(s, |acc, n| acc * n)))
}

fn apply_fn_direct(
    params: &[String],
    _sig: &Arc<FnSig>,
    body: &Arc<Expr>,
    captured: &Arc<HashMap<String, Val>>,
    args: Vec<Val>,
    env: &Env,
) -> Result<Val, String> {
    if params.len() != args.len() {
        return Err(format!("function expects {} arg(s), got {}", params.len(), args.len()));
    }
    // NOTE: type-hint coercion (sig) is deferred to a later phase.
    let mut local = make_local(env, captured);
    for (p, a) in params.iter().zip(args) {
        local.define(p.clone(), a);
    }
    eval(body, &local)
}

/// Three-layer scope: globals (forward-declared names) → closure capture → params.
fn make_local(global: &Env, captured: &Arc<HashMap<String, Val>>) -> Env {
    let mut vars = (*global.vars).clone();
    vars.extend(captured.iter().map(|(k, v)| (k.clone(), v.clone())));
    Env { vars: Arc::new(vars), target: global.target }
}

fn compose_fns(f: Val, g: Val) -> Val {
    let mut captured = HashMap::new();
    captured.insert("__f__".into(), f);
    captured.insert("__g__".into(), g);
    let body = Expr::Apply(
        Box::new(Expr::Var("__f__".into())),
        vec![Expr::Apply(Box::new(Expr::Var("__g__".into())), vec![Expr::Var("__z__".into())])],
    );
    Val::make_fn(vec!["__z__".into()], body, Arc::new(captured))
}

// ── Operators ─────────────────────────────────────────────────────────────────

pub fn binop_val(lv: Val, op: &Op, rv: Val, target: Target) -> Result<Val, String> {
    if matches!((&lv, &rv), (Val::Tuple(_), _) | (_, Val::Tuple(_))) {
        if matches!(op, Op::Eq | Op::Ne) {
            if let (Val::Tuple(ls), Val::Tuple(rs)) = (&lv, &rv) {
                let eq = tuple_scalar_eq(ls, rs);
                return Ok(Val::Num(if matches!(op, Op::Eq) == eq { 1.0 } else { 0.0 }));
            }
        }
        return binop_tuple(lv, op, rv, target);
    }
    if matches!((&lv, &rv), (Val::Field(_), _) | (_, Val::Field(_))) {
        return crate::field::field_binop(lv, op, rv);
    }
    if is_complex_combo(&lv, &rv) {
        return complex_binop(lv, op, rv, target);
    }
    if matches!((&lv, &rv), (Val::Tensor(_), _) | (_, Val::Tensor(_))) {
        return tensor_binop(lv, op, rv, target);
    }
    scalar_binop(lv, op, rv)
}

/// A binop yields a complex *tensor* when either side is one, or when a real tensor
/// meets a complex scalar.
fn is_complex_combo(l: &Val, r: &Val) -> bool {
    matches!(
        (l, r),
        (Val::ComplexTensor(_), _)
            | (_, Val::ComplexTensor(_))
            | (Val::Tensor(_), Val::Complex(..))
            | (Val::Complex(..), Val::Tensor(_))
    )
}

/// Coerce any operand to a complex tensor on `target` (scalars → length-1 broadcast;
/// real tensors promote with im = 0).
fn to_ctensor_on(v: Val, target: Target) -> Result<CTensor, String> {
    match v {
        Val::ComplexTensor(ct) => compute::ensure_complex_on(ct, target),
        Val::Complex(a, b) => compute::upload_complex(target, &[a], &[b], vec![1]),
        Val::Num(x) => compute::upload_complex(target, &[x], &[0.0], vec![1]),
        Val::Tensor(t) => {
            let t = ensure_on(t, target)?;
            compute::promote_real(target, &t)
        }
        other => Err(format!("cannot combine {} with a complex tensor", fmt_val(&other))),
    }
}

fn complex_binop(lv: Val, op: &Op, rv: Val, target: Target) -> Result<Val, String> {
    let code = match op {
        Op::Add => compute::OP_ADD,
        Op::Sub => compute::OP_SUB,
        Op::Mul => compute::OP_MUL,
        Op::Div => compute::OP_DIV,
        _ => return Err(format!("operator `{op:?}` is not defined on complex tensors (only + - * /)")),
    };
    let a = to_ctensor_on(lv, target)?;
    let b = to_ctensor_on(rv, target)?;
    compute::cbinop(target, code, &a, &b).map(Val::ComplexTensor)
}

fn binop_tuple(lv: Val, op: &Op, rv: Val, target: Target) -> Result<Val, String> {
    match (lv, rv) {
        (Val::Tuple(ls), Val::Tuple(rs)) => {
            if ls.len() != rs.len() {
                return Err(format!("tuple op tuple: length mismatch ({} vs {})", ls.len(), rs.len()));
            }
            let out: Result<Vec<Val>, _> = ls.into_iter().zip(rs).map(|(l, r)| binop_val(l, op, r, target)).collect();
            Ok(Val::Tuple(out?))
        }
        (Val::Tuple(ls), leaf) => {
            let out: Result<Vec<Val>, _> = ls.into_iter().map(|l| binop_val(l, op, leaf.clone(), target)).collect();
            Ok(Val::Tuple(out?))
        }
        (leaf, Val::Tuple(rs)) => {
            let out: Result<Vec<Val>, _> = rs.into_iter().map(|r| binop_val(leaf.clone(), op, r, target)).collect();
            Ok(Val::Tuple(out?))
        }
        _ => unreachable!(),
    }
}

/// Map an AST operator to a compute op code, or `None` if not yet on the device.
pub fn op_code(op: &Op) -> Option<u32> {
    Some(match op {
        Op::Add => compute::OP_ADD,
        Op::Sub => compute::OP_SUB,
        Op::Mul => compute::OP_MUL,
        Op::Div => compute::OP_DIV,
        Op::Pow => compute::OP_POW,
        Op::Lt => compute::OP_LT,
        Op::Gt => compute::OP_GT,
        Op::LtEq => compute::OP_LE,
        Op::GtEq => compute::OP_GE,
        Op::Eq => compute::OP_EQ,
        Op::Ne => compute::OP_NE,
        Op::FloorDiv | Op::Rem | Op::And | Op::Or => return None,
    })
}

/// Move/clone a tensor onto `target` (re-materialising if backend/precision differ).
pub fn ensure_on(t: TensorVal, target: Target) -> Result<TensorVal, String> {
    if t.backend == target.backend && t.prec == target.prec {
        Ok(t)
    } else {
        let host = compute::download(&t)?;
        compute::upload(target, &host, t.shape.clone())
    }
}

/// Coerce any scalar/tensor operand to a tensor on `target` (scalars → length-1,
/// which the kernel broadcasts).
pub fn to_tensor_on(v: Val, target: Target) -> Result<TensorVal, String> {
    match v {
        Val::Tensor(t) => ensure_on(t, target),
        Val::Num(x) => compute::upload(target, &[x], vec![1]),
        Val::Complex(..) => Err("complex tensors are Phase 5".into()),
        other => Err(format!("cannot combine {} with a tensor", fmt_val(&other))),
    }
}

fn tensor_binop(lv: Val, op: &Op, rv: Val, target: Target) -> Result<Val, String> {
    let code = op_code(op)
        .ok_or_else(|| format!("operator `{op:?}` is not available on tensors yet"))?;
    let a = to_tensor_on(lv, target)?;
    let b = to_tensor_on(rv, target)?;
    compute::binop(target, code, &a, &b).map(Val::Tensor)
}

fn tuple_scalar_eq(ls: &[Val], rs: &[Val]) -> bool {
    ls.len() == rs.len()
        && ls.iter().zip(rs.iter()).all(|(a, b)| matches!((a, b), (Val::Num(x), Val::Num(y)) if x == y))
}

fn scalar_binop(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    if let (Val::Num(la), Val::Num(ra)) = (&lv, &rv) {
        return Ok(Val::Num(match op {
            Op::Add => la + ra,
            Op::Sub => la - ra,
            Op::Mul => la * ra,
            Op::Div => la / ra,
            Op::FloorDiv => (*la / *ra).floor(),
            Op::Rem => la % ra,
            Op::Pow => la.powf(*ra),
            Op::Lt => (la < ra) as i64 as f64,
            Op::Gt => (la > ra) as i64 as f64,
            Op::LtEq => (la <= ra) as i64 as f64,
            Op::GtEq => (la >= ra) as i64 as f64,
            Op::Eq => (la == ra) as i64 as f64,
            Op::Ne => (la != ra) as i64 as f64,
            Op::And => (int(*la) != 0 && int(*ra) != 0) as i64 as f64,
            Op::Or => (int(*la) != 0 || int(*ra) != 0) as i64 as f64,
        }));
    }
    let (la, lb) = to_complex(lv)?;
    let (ra, rb) = to_complex(rv)?;
    match op {
        Op::Add => Ok(make_complex(la + ra, lb + rb)),
        Op::Sub => Ok(make_complex(la - ra, lb - rb)),
        Op::Mul => Ok(make_complex(la * ra - lb * rb, la * rb + lb * ra)),
        Op::Div => {
            let d = ra * ra + rb * rb;
            if d == 0.0 {
                return Err("division by zero".into());
            }
            Ok(make_complex((la * ra + lb * rb) / d, (lb * ra - la * rb) / d))
        }
        Op::Pow => Ok(complex_pow(la, lb, ra, rb)),
        Op::FloorDiv | Op::Rem => Err("// and % not defined for complex numbers".into()),
        Op::Eq => Ok(Val::Num((la == ra && lb == rb) as i64 as f64)),
        Op::Ne => Ok(Val::Num((la != ra || lb != rb) as i64 as f64)),
        Op::Lt | Op::Gt | Op::LtEq | Op::GtEq => Err("comparison not defined for complex numbers".into()),
        Op::And | Op::Or => Err("& and | not defined for complex numbers".into()),
    }
}

pub fn complex_pow(la: f64, lb: f64, ra: f64, rb: f64) -> Val {
    if la == 0.0 && lb == 0.0 {
        return if ra == 0.0 && rb == 0.0 { Val::Num(1.0) } else { Val::Num(0.0) };
    }
    let r = (la * la + lb * lb).sqrt();
    let theta = lb.atan2(la);
    let new_re = ra * r.ln() - rb * theta;
    let new_im = ra * theta + rb * r.ln();
    let mag = new_re.exp();
    make_complex(mag * new_im.cos(), mag * new_im.sin())
}

pub fn neg_val(v: Val, target: Target) -> Result<Val, String> {
    match v {
        Val::Num(n) => Ok(Val::Num(-n)),
        Val::Complex(a, b) => Ok(make_complex(-a, -b)),
        Val::Tuple(items) => {
            Ok(Val::Tuple(items.into_iter().map(|x| neg_val(x, target)).collect::<Result<Vec<_>, _>>()?))
        }
        Val::Tensor(t) => compute::unary(target, compute::UN_NEG, &t).map(Val::Tensor),
        Val::ComplexTensor(ct) => compute::cunary_c2c(target, compute::CU_NEG, &ct).map(Val::ComplexTensor),
        other => Err(format!("unary minus: expected a number, got {}", fmt_val(&other))),
    }
}

fn not_val(v: Val) -> Result<Val, String> {
    fn lnot(x: f64) -> f64 {
        (int(x) == 0) as i64 as f64
    }
    match v {
        Val::Num(n) => Ok(Val::Num(lnot(n))),
        Val::Tuple(items) => {
            let r: Result<Vec<Val>, _> = items
                .into_iter()
                .map(|v| match v {
                    Val::Num(n) => Ok(Val::Num(lnot(n))),
                    other => Err(format!("~: expected a number, got {}", fmt_val(&other))),
                })
                .collect();
            Ok(Val::Tuple(r?))
        }
        other => Err(format!("~: expected a number, got {}", fmt_val(&other))),
    }
}

// ── index helper ────────────────────────────────────────────────────────────────

pub fn norm_index(i: i64, len: usize, what: &str) -> Result<usize, String> {
    let adj = if i < 0 { i + len as i64 } else { i };
    if adj < 0 || adj as usize >= len {
        return Err(format!("{what} index {i} out of range (len {len})"));
    }
    Ok(adj as usize)
}

// ── tensor indexing / slicing (host-side gather) ─────────────────────────────────

/// Resolve one index item for an axis of size `dim`: returns (is_range, indices).
/// `..`/`lo..`/`..hi`/`lo..hi` are ranges (keep the axis, inclusive); a scalar
/// collapses the axis. Matches the original's semantics.
fn resolve_index_item(item: &Expr, dim: usize, k: usize, env: &Env) -> Result<(bool, Vec<usize>), String> {
    let clamp = |raw: i64| -> Result<usize, String> {
        let i = if raw < 0 { raw + dim as i64 } else { raw };
        if i < 0 || i >= dim as i64 {
            Err(format!("index {raw} out of range for dim {k} (size={dim})"))
        } else {
            Ok(i as usize)
        }
    };
    match item {
        Expr::Slice(None, None) => Ok((true, (0..dim).collect())),
        Expr::Slice(Some(lo), None) => {
            let lo = eval(lo, env)?.num("slice lo")? as i64;
            let lo = if lo < 0 { (dim as i64 + lo).max(0) as usize } else { lo as usize };
            Ok((true, (lo.min(dim)..dim).collect()))
        }
        Expr::Slice(None, Some(hi)) => {
            let hi = clamp(eval(hi, env)?.num("slice hi")? as i64)?;
            Ok((true, (0..=hi).collect()))
        }
        Expr::Slice(Some(lo), Some(hi)) => {
            let lo = eval(lo, env)?.num("slice lo")? as i64;
            let lo = if lo < 0 { (dim as i64 + lo).max(0) as usize } else { lo as usize };
            let hi = clamp(eval(hi, env)?.num("slice hi")? as i64)?;
            if lo > hi { return Ok((true, vec![])); }
            Ok((true, (lo..=hi).collect()))
        }
        other => {
            let i = clamp(eval(other, env)?.num("tensor index")? as i64)?;
            Ok((false, vec![i]))
        }
    }
}

/// Resolve a full index expression against `shape`, returning the source linear
/// indices (in row-major result order) and the result shape (collapsed axes drop).
fn resolve_index(shape: &[usize], idx: &Expr, env: &Env) -> Result<(Vec<usize>, Vec<usize>), String> {
    let rank = shape.len();
    let items: Vec<&Expr> = match idx {
        Expr::Tuple(es) => es.iter().collect(),
        single => vec![single],
    };
    if items.len() > rank {
        return Err(format!("got {} indices for a {}-D tensor", items.len(), rank));
    }
    // Per-axis spec; axes beyond the given items are full slices.
    let mut sel: Vec<Vec<usize>> = Vec::with_capacity(rank);
    let mut out_shape: Vec<usize> = Vec::new();
    for k in 0..rank {
        let (is_range, idxs) = match items.get(k) {
            Some(item) => resolve_index_item(item, shape[k], k, env)?,
            None => (true, (0..shape[k]).collect()),
        };
        if is_range {
            out_shape.push(idxs.len());
        }
        sel.push(idxs);
    }
    // Row-major strides of the source.
    let mut strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    let total: usize = sel.iter().map(|s| s.len()).product();
    let mut lin = Vec::with_capacity(total);
    let mut counter = vec![0usize; rank];
    for _ in 0..total {
        let mut src = 0;
        for k in 0..rank {
            src += sel[k][counter[k]] * strides[k];
        }
        lin.push(src);
        for k in (0..rank).rev() {
            counter[k] += 1;
            if counter[k] < sel[k].len() {
                break;
            }
            counter[k] = 0;
        }
    }
    Ok((lin, out_shape))
}

// ── special forms ─────────────────────────────────────────────────────────────

fn eval_agg(args: &[Expr], env: &Env, product: bool) -> Result<Val, String> {
    let label = if product { "prod" } else { "sum" };
    // 1-arg: sum(tuple) or sum(tensor)
    if args.len() == 1 {
        return match eval(&args[0], env)? {
            // Device reduction (Neumaier sum); df64 falls back to host inside compute.
            Val::Tensor(t) => {
                let op = if product { compute::RED_PROD } else { compute::RED_SUM };
                Ok(Val::Num(compute::reduce(env.target, op, &t)?))
            }
            Val::ComplexTensor(ct) => {
                if product {
                    return Err("complex prod reduction is staged; use sum".into());
                }
                let (r, i) = compute::creduce_sum(&ct)?;
                Ok(make_complex(r, i))
            }
            Val::Tuple(items) => {
                let (mut acc_re, mut acc_im) = if product { (1.0, 0.0) } else { (0.0, 0.0) };
                for v in items {
                    let (r, i) = to_complex(v)?;
                    if product {
                        let nr = acc_re * r - acc_im * i;
                        let ni = acc_re * i + acc_im * r;
                        acc_re = nr;
                        acc_im = ni;
                    } else {
                        acc_re += r;
                        acc_im += i;
                    }
                }
                Ok(make_complex(acc_re, acc_im))
            }
            other => Err(format!("{label}: 1-arg form requires a tuple, got {}", fmt_val(&other))),
        };
    }
    // 2-arg: sum(tensor, axis) — axis reduction (host-side; result drops the axis)
    if args.len() == 2 {
        if let Val::Tensor(t) = eval(&args[0], env)? {
            let axis = eval(&args[1], env)?.num(label)? as usize;
            if axis >= t.shape.len() {
                return Err(format!("{label}: axis {axis} out of range for {}-D tensor", t.shape.len()));
            }
            let data = compute::download(&t)?;
            let (out, out_shape) = axis_reduce(&data, &t.shape, axis, product);
            return if out_shape.is_empty() {
                Ok(Val::Num(out[0]))
            } else {
                compute::upload(env.target, &out, out_shape).map(Val::Tensor)
            };
        }
        return Err(format!("{label}: 2-arg form is {label}(tensor, axis)"));
    }
    // 3-arg: sum(f, lo, hi) inclusive
    if args.len() == 3 {
        let f = eval(&args[0], env)?;
        let lo = eval(&args[1], env)?.num(label)? as i64;
        let hi = eval(&args[2], env)?.num(label)? as i64;
        let (mut acc_re, mut acc_im) = if product { (1.0, 0.0) } else { (0.0, 0.0) };
        for k in lo..=hi {
            let (r, i) = to_complex(apply_val(f.clone(), vec![Val::Num(k as f64)], env)?)?;
            if product {
                let nr = acc_re * r - acc_im * i;
                let ni = acc_re * i + acc_im * r;
                acc_re = nr;
                acc_im = ni;
            } else {
                acc_re += r;
                acc_im += i;
            }
        }
        return Ok(make_complex(acc_re, acc_im));
    }
    Err(format!("{label}: expected {label}(tuple) or {label}(f, lo, hi) in the prototype"))
}

/// Reduce a flat row-major tensor along one axis (sum or product). Returns the
/// flat result and its shape (the axis removed; empty shape ⇒ scalar).
fn axis_reduce(data: &[f64], shape: &[usize], axis: usize, product: bool) -> (Vec<f64>, Vec<usize>) {
    let rank = shape.len();
    let asize = shape[axis];
    let mut strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    let out_shape: Vec<usize> = (0..rank).filter(|&k| k != axis).map(|k| shape[k]).collect();
    let out_total: usize = out_shape.iter().product::<usize>().max(1);
    let mut out_strides = vec![1usize; out_shape.len()];
    for k in (0..out_shape.len().saturating_sub(1)).rev() {
        out_strides[k] = out_strides[k + 1] * out_shape[k + 1];
    }
    let mut out = vec![if product { 1.0 } else { 0.0 }; out_total];
    for (o, slot) in out.iter_mut().enumerate() {
        // decode the output multi-index
        let mut om = vec![0usize; out_shape.len()];
        let mut r = o;
        for k in 0..out_shape.len() {
            om[k] = r / out_strides[k];
            r %= out_strides[k];
        }
        let mut acc = if product { 1.0 } else { 0.0 };
        for j in 0..asize {
            let mut src = 0;
            let mut oi = 0;
            for (k, &stride) in strides.iter().enumerate() {
                let coord = if k == axis { j } else { let c = om[oi]; oi += 1; c };
                src += coord * stride;
            }
            if product { acc *= data[src]; } else { acc += data[src]; }
        }
        *slot = acc;
    }
    (out, out_shape)
}

fn eval_iterate(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 3 {
        return Err("iterate(f, x0, n) expects 3 args".into());
    }
    let f = eval(&args[0], env)?;
    let mut state = eval(&args[1], env)?;
    let n = eval(&args[2], env)?.num("iterate")? as i64;
    if n < 0 {
        return Err(format!("iterate: count must be non-negative, got {n}"));
    }
    for step in 0..n {
        state = apply_val(f.clone(), vec![state], env)?;
        if !state_is_finite(&state) {
            return Err(format!("iterate: non-finite value (NaN/Inf) at step {}", step + 1));
        }
    }
    Ok(state)
}

fn state_is_finite(v: &Val) -> bool {
    match v {
        Val::Num(x) => x.is_finite(),
        Val::Complex(a, b) => a.is_finite() && b.is_finite(),
        Val::Tuple(items) => items.iter().all(state_is_finite),
        // Tensors are device-resident; checking would force a per-step download and
        // break loop residency, so we don't (matches the original GPU path).
        _ => true,
    }
}

/// `scan(f, x0, n)` — the orbit `[x0, f(x0), …, fⁿ(x0)]`, stacked with time as the
/// leading axis. Same resident, host-driven loop as `iterate`; only the final
/// stacking assembles the result. Scalar states → 1-D `[n+1]`; tensor states →
/// `[n+1, …shape]`; a flat numeric tuple → `[n+1, k]`; a structured tuple → a tuple
/// of per-field stacks (matching the original's semantics).
fn eval_scan(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 3 {
        return Err("scan(f, x0, n) expects 3 args".into());
    }
    let f = eval(&args[0], env)?;
    let x0 = eval(&args[1], env)?;
    let n = eval(&args[2], env)?.num("scan")? as i64;
    if n < 0 {
        return Err(format!("scan: count must be non-negative, got {n}"));
    }
    let mut states = Vec::with_capacity((n + 1) as usize);
    states.push(x0.clone());
    let mut s = x0;
    for step in 0..n {
        s = apply_val(f.clone(), vec![s], env)?;
        if !state_is_finite(&s) {
            return Err(format!("scan: non-finite value (NaN/Inf) at step {}", step + 1));
        }
        states.push(s.clone());
    }
    stack_states(states, env)
}

/// Stack a list of states (one per time step) into a leading-time-axis result.
fn stack_states(states: Vec<Val>, env: &Env) -> Result<Val, String> {
    let rows = states.len();
    let first = states.first().ok_or("scan: empty orbit")?;
    match first {
        Val::Num(_) => {
            let data: Result<Vec<f64>, _> = states.iter().map(|v| v.clone().num("scan")).collect();
            compute::upload(env.target, &data?, vec![rows]).map(Val::Tensor)
        }
        Val::Tensor(t0) => {
            let shape0 = t0.shape.clone();
            let mut data = Vec::with_capacity(rows * t0.len);
            for v in &states {
                match v {
                    Val::Tensor(t) if t.shape == shape0 => data.extend(compute::download(t)?),
                    Val::Tensor(_) => return Err("scan: tensor shape changed across steps".into()),
                    other => return Err(format!("scan: mixed state types, got {}", fmt_val(other))),
                }
            }
            let mut out_shape = vec![rows];
            out_shape.extend(shape0);
            compute::upload(env.target, &data, out_shape).map(Val::Tensor)
        }
        Val::Tuple(first_items) => {
            let k = first_items.len();
            // A flat numeric tuple (a, b, …) row-packs into [rows, k].
            let all_flat = states.iter().all(|v| {
                matches!(v, Val::Tuple(items) if items.len() == k && items.iter().all(|x| matches!(x, Val::Num(_))))
            });
            if all_flat {
                let mut data = Vec::with_capacity(rows * k);
                for v in &states {
                    if let Val::Tuple(items) = v {
                        for it in items {
                            data.push(it.clone().num("scan")?);
                        }
                    }
                }
                return compute::upload(env.target, &data, vec![rows, k]).map(Val::Tensor);
            }
            // Structured tuple → a tuple of per-field stacks.
            let mut fields: Vec<Vec<Val>> = (0..k).map(|_| Vec::with_capacity(rows)).collect();
            for v in &states {
                match v {
                    Val::Tuple(items) if items.len() == k => {
                        for (j, it) in items.iter().enumerate() {
                            fields[j].push(it.clone());
                        }
                    }
                    _ => return Err("scan: tuple structure changed across steps".into()),
                }
            }
            let stacked: Result<Vec<Val>, _> = fields.into_iter().map(|f| stack_states(f, env)).collect();
            Ok(Val::Tuple(stacked?))
        }
        other => Err(format!("scan: cannot stack states of type {}", fmt_val(other))),
    }
}

fn eval_map(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 2 {
        return Err("map(f, tuple) expects 2 args".into());
    }
    let f = eval(&args[0], env)?;
    match eval(&args[1], env)? {
        Val::Tuple(items) => {
            let res: Result<Vec<Val>, _> = items.into_iter().map(|x| apply_val(f.clone(), vec![x], env)).collect();
            Ok(Val::Tuple(res?))
        }
        other => Err(format!("map: second arg must be a tuple (tensors in Phase 2), got {}", fmt_val(&other))),
    }
}

fn eval_filter(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 2 {
        return Err("filter(f, tuple) expects 2 args".into());
    }
    let f = eval(&args[0], env)?;
    match eval(&args[1], env)? {
        Val::Tuple(items) => {
            let mut out = vec![];
            for item in items {
                if apply_val(f.clone(), vec![item.clone()], env)?.num("filter")? != 0.0 {
                    out.push(item);
                }
            }
            Ok(Val::Tuple(out))
        }
        other => Err(format!("filter: second arg must be a tuple (tensors in Phase 2), got {}", fmt_val(&other))),
    }
}

fn eval_reduce(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 2 {
        return Err("reduce(f, tuple) expects 2 args".into());
    }
    let f = eval(&args[0], env)?;
    match eval(&args[1], env)? {
        Val::Tuple(items) => {
            let mut it = items.into_iter();
            let mut acc = it.next().ok_or("reduce: empty tuple")?;
            for item in it {
                acc = apply_val(f.clone(), vec![acc, item], env)?;
            }
            Ok(acc)
        }
        other => Err(format!("reduce: second arg must be a tuple (tensors in Phase 2), got {}", fmt_val(&other))),
    }
}
