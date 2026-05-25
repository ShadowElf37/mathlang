use std::collections::HashMap;
use std::collections::HashSet;
use crate::ast::{Expr, BlockStmt, Op, Def};

// ── Values ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Val {
    Num(f64),
    Complex(f64, f64),
    Fn(Vec<String>, Expr),
    Tuple(Vec<Val>),
}

impl Val {
    pub fn num(self, ctx: &str) -> Result<f64, String> {
        match self {
            Val::Num(n)      => Ok(n),
            Val::Complex(..) => Err(format!("{ctx}: expected a real number, got complex")),
            Val::Fn(..)      => Err(format!("{ctx}: expected a number, got a function")),
            Val::Tuple(..)   => Err(format!("{ctx}: expected a number, got a tuple")),
        }
    }
}

// ── Environment ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Env {
    pub vars: HashMap<String, Val>,
    pub fns:  HashMap<String, (Vec<String>, Expr)>,
}

impl Env {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        let num = |n: f64| Val::Num(n);
        vars.insert("pi".into(),  num(std::f64::consts::PI));
        vars.insert("e".into(),   num(std::f64::consts::E));
        vars.insert("tau".into(), num(std::f64::consts::TAU));
        vars.insert("phi".into(), num(1.618033988749895));
        vars.insert("inf".into(), num(f64::INFINITY));
        vars.insert("i".into(),   Val::Complex(0.0, 1.0));
        Self { vars, fns: HashMap::new() }
    }
}

// ── Output formatting ─────────────────────────────────────────────────────────

pub fn fmt_val(v: &Val) -> String {
    fn fmt_f(n: f64) -> String {
        if n.is_nan() { return "NaN".into(); }
        if n.is_infinite() { return if n > 0.0 { "inf".into() } else { "-inf".into() }; }
        if n.fract() == 0.0 && n.abs() < 1e15 { return format!("{}", n as i64); }
        format!("{n}")
    }
    match v {
        Val::Num(n) => fmt_f(*n),
        Val::Complex(a, b) => {
            let re = fmt_f(*a);
            let babs = b.abs();
            let im = if babs == 1.0 { String::new() } else { fmt_f(babs) };
            if *a == 0.0 {
                if *b < 0.0 { format!("-{im}i") } else { format!("{im}i") }
            } else if *b < 0.0 {
                format!("{re} - {im}i")
            } else {
                format!("{re} + {im}i")
            }
        }
        Val::Fn(params, _) => format!("<fn {} -> ...>", params.join(", ")),
        Val::Tuple(items) => format!("({})", items.iter().map(fmt_val).collect::<Vec<_>>().join(", ")),
    }
}

// ── Evaluator helpers ─────────────────────────────────────────────────────────

#[inline] fn int(x: f64) -> i64 { x as i64 }

// Collapse a+bi to Num(a) when b is negligibly small relative to the magnitude.
pub fn make_complex(a: f64, b: f64) -> Val {
    let scale = (a.abs() + b.abs()).max(1.0) * 1e-10;
    let a = if a.abs() < scale { 0.0 } else { a };
    let b = if b.abs() < scale { 0.0 } else { b };
    if b == 0.0 { Val::Num(a) } else { Val::Complex(a, b) }
}

fn to_complex(v: Val) -> Result<(f64, f64), String> {
    match v {
        Val::Num(n)        => Ok((n, 0.0)),
        Val::Complex(a, b) => Ok((a, b)),
        Val::Fn(..)        => Err("expected a number, got a function".into()),
        Val::Tuple(..)     => Err("expected a number, got a tuple".into()),
    }
}

// z^w via exp(w·ln z)
fn complex_pow(la: f64, lb: f64, ra: f64, rb: f64) -> Val {
    if la == 0.0 && lb == 0.0 {
        return if ra == 0.0 && rb == 0.0 { Val::Num(1.0) } else { Val::Num(0.0) };
    }
    let r     = (la*la + lb*lb).sqrt();
    let theta = lb.atan2(la);
    let new_re = ra * r.ln() - rb * theta;
    let new_im = ra * theta  + rb * r.ln();
    let mag = new_re.exp();
    make_complex(mag * new_im.cos(), mag * new_im.sin())
}

fn arity(name: &str, expected: usize, got: usize) -> Result<(), String> {
    if expected == got { Ok(()) }
    else { Err(format!("{name} expects {expected} arg(s), got {got}")) }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a / gcd(a, b) * b }
}

// ── Free variable collection ──────────────────────────────────────────────────

pub fn free_vars(expr: &Expr, env: &Env) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut order = vec![];
    collect_free(expr, env, &mut seen, &mut order);
    order
}

fn collect_free(expr: &Expr, env: &Env, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    match expr {
        Expr::Var(n) => {
            if !env.vars.contains_key(n) && !env.fns.contains_key(n) && !seen.contains(n) {
                seen.insert(n.clone());
                out.push(n.clone());
            }
        }
        Expr::Num(_) | Expr::ImagLit(_) => {}
        Expr::Lambda(params, body) => {
            // params are bound inside the lambda
            let mut inner_env = env.clone();
            for p in params { inner_env.vars.insert(p.clone(), Val::Num(0.0)); }
            collect_free(body, &inner_env, seen, out);
        }
        Expr::Neg(e) => collect_free(e, env, seen, out),
        Expr::BinOp(l, _, r) => { collect_free(l, env, seen, out); collect_free(r, env, seen, out); }
        Expr::Call(name, args) => {
            // name is a known fn or builtin — don't add it; recurse into args
            let _ = name;
            for a in args { collect_free(a, env, seen, out); }
        }
        Expr::Apply(f, args) => {
            collect_free(f, env, seen, out);
            for a in args { collect_free(a, env, seen, out); }
        }
        Expr::Tuple(es) => { for e in es { collect_free(e, env, seen, out); } }
        Expr::Index(e, i) => { collect_free(e, env, seen, out); collect_free(i, env, seen, out); }
        Expr::Block(stmts) => {
            let mut inner = env.clone();
            for s in stmts {
                match s {
                    BlockStmt::Def(Def::Var(n, e)) => {
                        collect_free(e, &inner, seen, out);
                        inner.vars.insert(n.clone(), Val::Num(0.0));
                    }
                    BlockStmt::Def(Def::Func(n, ps, b)) => {
                        let mut fe = inner.clone();
                        for p in ps { fe.vars.insert(p.clone(), Val::Num(0.0)); }
                        collect_free(b, &fe, seen, out);
                        inner.fns.insert(n.clone(), (ps.clone(), b.clone()));
                    }
                    BlockStmt::Expr(e) => collect_free(e, &inner, seen, out),
                }
            }
        }
    }
}

// ── Value application ─────────────────────────────────────────────────────────

// Apply a Val (fn, scalar, tuple) to a list of Val args with full type dispatch.
pub fn apply_val(f: Val, args: Vec<Val>, env: &Env) -> Result<Val, String> {
    match f {
        Val::Fn(ref params, ref body) => {
            let n = params.len();
            let k = args.len();
            // All args are Fn → compose (only single arg supported)
            if k == 1 {
                if let Val::Fn(_, _) = &args[0] {
                    let g = args.into_iter().next().unwrap();
                    // compose: new fn whose body calls f on the result of g
                    // We create a synthetic Val::Fn that captures both
                    return Ok(compose_fns(f, g));
                }
                // Single n-tuple arg → destructure into n params
                if let Val::Tuple(ref items) = args[0] {
                    if items.len() == n {
                        let mut local = env.clone();
                        for (p, v) in params.iter().zip(items.iter()) {
                            local.vars.insert(p.clone(), v.clone());
                        }
                        return eval(body, &local);
                    }
                }
                // Single scalar arg with 1-param fn → direct apply
                if n == 1 {
                    let mut local = env.clone();
                    local.vars.insert(params[0].clone(), args.into_iter().next().unwrap());
                    return eval(body, &local);
                }
                return Err(format!("function expects {n} args, got 1"));
            }
            // k args, all n-tuples → map with destructuring → k-tuple
            let all_n_tuples = args.iter().all(|a| matches!(a, Val::Tuple(v) if v.len() == n));
            if all_n_tuples {
                let results: Result<Vec<Val>, _> = args.into_iter().map(|a| {
                    if let Val::Tuple(items) = a {
                        let mut local = env.clone();
                        for (p, v) in params.iter().zip(items) { local.vars.insert(p.clone(), v); }
                        eval(body, &local)
                    } else { unreachable!() }
                }).collect();
                return Ok(Val::Tuple(results?));
            }
            // k scalar args, 1-param fn → map → k-tuple
            if n == 1 {
                let results: Result<Vec<Val>, _> = args.into_iter().map(|a| {
                    let mut local = env.clone();
                    local.vars.insert(params[0].clone(), a);
                    eval(body, &local)
                }).collect();
                return Ok(Val::Tuple(results?));
            }
            // k == n scalar args → direct apply
            if k == n {
                let mut local = env.clone();
                for (p, v) in params.iter().zip(args) { local.vars.insert(p.clone(), v); }
                return eval(body, &local);
            }
            Err(format!("function expects {n} args, got {k}"))
        }
        Val::Num(s) => {
            if args.len() == 1 {
                match &args[0] {
                    Val::Fn(_, _) => {
                        // scalar × fn → scale: new fn z -> s * g(z)
                        return Ok(scale_fn(s, args.into_iter().next().unwrap()));
                    }
                    Val::Num(n) => return Ok(Val::Num(s * n)),
                    Val::Tuple(items) => {
                        // scalar × tuple → element-wise scale
                        let scaled: Vec<Val> = items.iter().map(|v| match v {
                            Val::Num(n) => Val::Num(s * n),
                            _ => v.clone(),
                        }).collect();
                        return Ok(Val::Tuple(scaled));
                    }
                    Val::Complex(a, b) => return Ok(make_complex(s * a, s * b)),
                }
            }
            // scalar × multiple scalars → multiply all
            let nums: Result<Vec<f64>, _> = args.into_iter().map(|v| v.num("scalar-apply")).collect();
            Ok(Val::Num(nums?.iter().fold(s, |acc, n| acc * n)))
        }
        Val::Complex(a, b) => {
            if args.len() == 1 {
                let (ra, rb) = to_complex(args.into_iter().next().unwrap())?;
                return Ok(make_complex(a*ra - b*rb, a*rb + b*ra));
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
    }
}

fn compose_fns(f: Val, g: Val) -> Val {
    // Returns a Val::Fn that is the composition f∘g.
    // We represent it as a 1-param fn stored in env via a special approach:
    // Since we can't close over Val::Fn in Expr, we use a trick:
    // Store f and g as vars "__compose_f__" and "__compose_g__" in a synthetic env,
    // but that would require env access. Instead we use Lambda with Apply in the body.
    // Actually the cleanest representation: return a Rust closure... but Val::Fn only stores Expr.
    // Best approach: store the composed fn as Val::Fn with a synthetic param and Apply body.
    // But Apply is an Expr node that contains Expr::Var refs, not Val refs.
    // Solution: wrap g in a lambda if it's a Val::Fn, store both under gensym names,
    // but we have no way to inject Vals into the Expr tree.
    //
    // Practical solution: represent composition as a Val::Fn whose body is Expr::Apply,
    // where we inject the closed-over vals via the env when evaluating.
    // We use a special trick: store f as a lambda in the env under a gensym key.
    // But we don't have mutable env access here.
    //
    // Simplest correct solution: use a captured-closure approach by creating an Expr::Apply
    // of two Expr::Lambda nodes derived from f and g.
    let f_expr = val_to_lambda_expr(f, "__z__");
    let g_expr = val_to_lambda_expr(g, "__z__");
    // compose: z -> f(g(z))
    // f_expr is a lambda (or var), g_expr is a lambda (or var)
    // Body: Apply(f_expr, [Apply(g_expr, [Var("__z__")])])
    let inner = Expr::Apply(Box::new(g_expr), vec![Expr::Var("__z__".into())]);
    let outer = Expr::Apply(Box::new(f_expr), vec![inner]);
    Val::Fn(vec!["__z__".into()], outer)
}

fn scale_fn(s: f64, g: Val) -> Val {
    let g_expr = val_to_lambda_expr(g, "__z__");
    let call = Expr::Apply(Box::new(g_expr), vec![Expr::Var("__z__".into())]);
    let body = Expr::BinOp(Box::new(Expr::Num(s)), Op::Mul, Box::new(call));
    Val::Fn(vec!["__z__".into()], body)
}

fn val_to_lambda_expr(v: Val, param: &str) -> Expr {
    match v {
        Val::Fn(params, body) => Expr::Lambda(params, Box::new(body)),
        Val::Num(n) => Expr::Num(n),
        Val::Complex(a, b) => {
            // represent as a+b*i — but for composition purposes just use real part
            // In practice composing with a complex constant is unusual; do best effort
            let _ = b;
            let _ = param;
            Expr::Num(a)
        }
        Val::Tuple(_) => Expr::Num(0.0), // degenerate
    }
}

fn binop_tuple(lv: Val, op: &Op, rv: Val, _env: &Env) -> Result<Val, String> {
    match (lv, rv) {
        (Val::Tuple(ls), Val::Tuple(rs)) => {
            if ls.len() != rs.len() {
                return Err(format!("tuple op tuple: length mismatch ({} vs {})", ls.len(), rs.len()));
            }
            let out: Result<Vec<Val>, _> = ls.into_iter().zip(rs)
                .map(|(l, r)| scalar_binop(l, op, r))
                .collect();
            Ok(Val::Tuple(out?))
        }
        (Val::Tuple(ls), scalar) => {
            let out: Result<Vec<Val>, _> = ls.into_iter()
                .map(|l| scalar_binop(l, op, scalar.clone()))
                .collect();
            Ok(Val::Tuple(out?))
        }
        (scalar, Val::Tuple(rs)) => {
            let out: Result<Vec<Val>, _> = rs.into_iter()
                .map(|r| scalar_binop(scalar.clone(), op, r))
                .collect();
            Ok(Val::Tuple(out?))
        }
        _ => unreachable!(),
    }
}

fn scalar_binop(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    if let (Val::Num(la), Val::Num(ra)) = (&lv, &rv) {
        return Ok(Val::Num(match op {
            Op::Add      => la + ra,
            Op::Sub      => la - ra,
            Op::Mul      => la * ra,
            Op::Div      => la / ra,
            Op::FloorDiv => (int(*la) / int(*ra)) as f64,
            Op::Rem      => la % ra,
            Op::Pow      => la.powf(*ra),
        }));
    }
    let (la, lb) = to_complex(lv)?;
    let (ra, rb) = to_complex(rv)?;
    match op {
        Op::Add      => Ok(make_complex(la + ra, lb + rb)),
        Op::Sub      => Ok(make_complex(la - ra, lb - rb)),
        Op::Mul      => Ok(make_complex(la*ra - lb*rb, la*rb + lb*ra)),
        Op::Div      => {
            let d = ra*ra + rb*rb;
            if d == 0.0 { return Err("division by zero".into()); }
            Ok(make_complex((la*ra + lb*rb)/d, (lb*ra - la*rb)/d))
        }
        Op::Pow      => Ok(complex_pow(la, lb, ra, rb)),
        Op::FloorDiv | Op::Rem => Err("// and % not defined for complex numbers".into()),
    }
}

// ── Evaluator ─────────────────────────────────────────────────────────────────

pub fn eval(expr: &Expr, env: &Env) -> Result<Val, String> {
    match expr {
        Expr::Num(n)      => Ok(Val::Num(*n)),
        Expr::ImagLit(n)  => Ok(if *n == 0.0 { Val::Num(0.0) } else { Val::Complex(0.0, *n) }),
        Expr::Lambda(p, b) => Ok(Val::Fn(p.clone(), *b.clone())),
        Expr::Tuple(exprs) => {
            let vals: Result<Vec<Val>, _> = exprs.iter().map(|e| eval(e, env)).collect();
            Ok(Val::Tuple(vals?))
        }
        Expr::Index(expr, idx) => {
            let v = eval(expr, env)?;
            let i = eval(idx, env)?.num("index")? as usize;
            match v {
                Val::Tuple(items) => items.into_iter().nth(i)
                    .ok_or_else(|| format!("index {i} out of range")),
                _ => Err("indexing requires a tuple".into()),
            }
        }
        Expr::Block(stmts) => {
            let mut child = env.clone();
            let mut last_val = Val::Tuple(vec![]);
            for stmt in stmts {
                match stmt {
                    BlockStmt::Def(def) => match def {
                        Def::Var(name, expr) => {
                            let v = eval(expr, &child)?;
                            child.vars.insert(name.clone(), v);
                        }
                        Def::Func(name, params, body) => {
                            child.fns.insert(name.clone(), (params.clone(), body.clone()));
                        }
                    },
                    BlockStmt::Expr(e) => { last_val = eval(e, &child)?; }
                }
            }
            Ok(last_val)
        }
        Expr::Apply(f_expr, arg_exprs) => {
            let f_val = eval(f_expr, env)?;
            let args: Result<Vec<Val>, _> = arg_exprs.iter().map(|a| eval(a, env)).collect();
            apply_val(f_val, args?, env)
        }
        Expr::Var(n)      => env.vars.get(n).cloned()
            .ok_or_else(|| format!("undefined: {n}")),
        Expr::Neg(e) => match eval(e, env)? {
            Val::Num(n)        => Ok(Val::Num(-n)),
            Val::Complex(a, b) => Ok(make_complex(-a, -b)),
            Val::Tuple(items)  => {
                let neg: Result<Vec<Val>, _> = items.into_iter()
                    .map(|v| match v {
                        Val::Num(n) => Ok(Val::Num(-n)),
                        Val::Complex(a, b) => Ok(make_complex(-a, -b)),
                        other => Err(format!("unary minus: expected a number, got {}", fmt_val(&other))),
                    }).collect();
                Ok(Val::Tuple(neg?))
            }
            Val::Fn(..) => Err("unary minus: expected a number".into()),
        },
        Expr::BinOp(l, op, r) => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
            // Tuple broadcasting: map op element-wise
            if matches!((&lv, &rv), (Val::Tuple(_), _) | (_, Val::Tuple(_))) {
                return binop_tuple(lv, op, rv, env);
            }
            if let (Val::Num(la), Val::Num(ra)) = (&lv, &rv) {
                return Ok(Val::Num(match op {
                    Op::Add      => la + ra,
                    Op::Sub      => la - ra,
                    Op::Mul      => la * ra,
                    Op::Div      => la / ra,
                    Op::FloorDiv => (int(*la) / int(*ra)) as f64,
                    Op::Rem      => la % ra,
                    Op::Pow      => la.powf(*ra),
                }));
            }
            let (la, lb) = to_complex(lv)?;
            let (ra, rb) = to_complex(rv)?;
            match op {
                Op::Add      => Ok(make_complex(la + ra, lb + rb)),
                Op::Sub      => Ok(make_complex(la - ra, lb - rb)),
                Op::Mul      => Ok(make_complex(la*ra - lb*rb, la*rb + lb*ra)),
                Op::Div      => {
                    let d = ra*ra + rb*rb;
                    if d == 0.0 { return Err("division by zero".into()); }
                    Ok(make_complex((la*ra + lb*rb)/d, (lb*ra - la*rb)/d))
                }
                Op::Pow      => Ok(complex_pow(la, lb, ra, rb)),
                Op::FloorDiv | Op::Rem => Err("// and % not defined for complex numbers".into()),
            }
        }
        Expr::Call(name, args) => {
            // These receive raw Expr args so a fn arg isn't forced to f64
            match name.as_str() {
                "sum"      => return eval_agg(args, env, false),
                "prod"     => return eval_agg(args, env, true),
                "integral" => return eval_integral(args, env),
                "deriv"    => return eval_deriv(args, env),
                "map" => {
                    if args.len() != 2 {
                        return Err("map(f, tuple) expects 2 args".into());
                    }
                    let items = match eval(&args[1], env)? {
                        Val::Tuple(items) => items,
                        other => return Err(format!("map: second arg must be a tuple, got {}", fmt_val(&other))),
                    };
                    let results: Result<Vec<Val>, _> = items.into_iter()
                        .map(|item| call_fn1(&args[0], item, env))
                        .collect();
                    return Ok(Val::Tuple(results?));
                }
                _ => {}
            }

            let vals: Result<Vec<Val>, _> = args.iter().map(|a| eval(a, env)).collect();
            let vals = vals?;

            if let Some((params, body)) = env.fns.get(name).cloned() {
                return apply_val(Val::Fn(params, body), vals, env);
            }

            if let Some(Val::Fn(params, body)) = env.vars.get(name).cloned() {
                return apply_val(Val::Fn(params, body), vals, env);
            }

            // Complex-capable builtins
            macro_rules! cx1 {
                ($vname:ident, $real_arm:expr, $cx_arm:expr) => {{
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num($vname)        => $real_arm,
                        Val::Complex($vname, _) => $cx_arm,
                        Val::Fn(..) | Val::Tuple(..) => return Err(format!("{name}: expected a number")),
                    });
                }};
            }
            match name.as_str() {
                "abs" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(n.abs()),
                        Val::Complex(a, b) => Val::Num((a*a + b*b).sqrt()),
                        Val::Fn(..) | Val::Tuple(..) => return Err("abs: expected a number".into()),
                    });
                }
                "re"  => cx1!(n, Val::Num(n), Val::Num(n)),
                "im"  => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(_)        => Val::Num(0.0),
                        Val::Complex(_, b) => Val::Num(b),
                        Val::Fn(..) | Val::Tuple(..) => return Err("im: expected a number".into()),
                    });
                }
                "arg" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(if n >= 0.0 { 0.0 } else { std::f64::consts::PI }),
                        Val::Complex(a, b) => Val::Num(b.atan2(a)),
                        Val::Fn(..) | Val::Tuple(..) => return Err("arg: expected a number".into()),
                    });
                }
                "conj" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(n),
                        Val::Complex(a, b) => make_complex(a, -b),
                        Val::Fn(..) | Val::Tuple(..) => return Err("conj: expected a number".into()),
                    });
                }
                "sqrt" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n) if n >= 0.0 => Val::Num(n.sqrt()),
                        Val::Num(n)             => Val::Complex(0.0, (-n).sqrt()),
                        Val::Complex(a, b) => {
                            let r     = (a*a + b*b).sqrt().sqrt();
                            let theta = b.atan2(a) / 2.0;
                            make_complex(r * theta.cos(), r * theta.sin())
                        }
                        Val::Fn(..) | Val::Tuple(..) => return Err("sqrt: expected a number".into()),
                    });
                }
                "exp" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(n.exp()),
                        Val::Complex(a, b) => { let m = a.exp(); make_complex(m*b.cos(), m*b.sin()) }
                        Val::Fn(..) | Val::Tuple(..) => return Err("exp: expected a number".into()),
                    });
                }
                "ln" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n) if n >= 0.0 => Val::Num(n.ln()),
                        Val::Num(n)             => make_complex((-n).ln(), std::f64::consts::PI),
                        Val::Complex(a, b)      => make_complex((a*a + b*b).sqrt().ln(), b.atan2(a)),
                        Val::Fn(..) | Val::Tuple(..) => return Err("ln: expected a number".into()),
                    });
                }
                _ => {}
            }

            // Real-only builtins
            let v: Result<Vec<f64>, _> = vals.into_iter().map(|v| v.num(name)).collect();
            let v = v?;
            macro_rules! f1 { ($f:ident) => {{ arity(name,1,v.len())?; return Ok(Val::Num(v[0].$f())) }} }
            macro_rules! f2 { ($e:expr)  => {{ arity(name,2,v.len())?; return Ok(Val::Num($e)) }} }
            macro_rules! i2 { ($e:expr)  => {{ arity(name,2,v.len())?; return Ok(Val::Num(($e) as f64)) }} }
            macro_rules! i1 { ($e:expr)  => {{ arity(name,1,v.len())?; return Ok(Val::Num(($e) as f64)) }} }
            match name.as_str() {
                "id"     => { arity(name,1,v.len())?; return Ok(Val::Num(v[0])); }
                "delta"  => { arity(name,1,v.len())?; return Ok(Val::Num(if v[0] == 0.0 { 1.0 } else { 0.0 })); }
                "fact" | "factorial" => {
                    arity(name,1,v.len())?;
                    let n = v[0] as u64;
                    return Ok(Val::Num((1..=n).map(|k| k as f64).product()));
                }
                "sin"    => f1!(sin),   "cos"   => f1!(cos),   "tan"  => f1!(tan),
                "asin"   => f1!(asin),  "acos"  => f1!(acos),  "atan" => f1!(atan),
                "atan2"  => f2!(v[0].atan2(v[1])),
                "sinh"   => f1!(sinh),  "cosh"  => f1!(cosh),  "tanh" => f1!(tanh),
                "cbrt"   => f1!(cbrt),
                "sign" | "signum" => f1!(signum),
                "floor"  => f1!(floor), "ceil"  => f1!(ceil),  "round" => f1!(round),
                "log" | "log10" => f1!(log10),
                "log2"   => f1!(log2),
                "min"    => f2!(v[0].min(v[1])),
                "max"    => f2!(v[0].max(v[1])),
                "pow"    => f2!(v[0].powf(v[1])),
                "hypot"  => f2!(v[0].hypot(v[1])),
                "gcd"    => i2!(gcd(int(v[0]).unsigned_abs(), int(v[1]).unsigned_abs())),
                "lcm"    => i2!(lcm(int(v[0]).unsigned_abs(), int(v[1]).unsigned_abs())),
                "and"    => i2!(int(v[0]) & int(v[1])),
                "or"     => i2!(int(v[0]) | int(v[1])),
                "xor"    => i2!(int(v[0]) ^ int(v[1])),
                "nand"   => i2!(!(int(v[0]) & int(v[1]))),
                "nor"    => i2!(!(int(v[0]) | int(v[1]))),
                "xnor"   => i2!(!(int(v[0]) ^ int(v[1]))),
                "not"    => i1!(!int(v[0])),
                "shl"    => i2!(int(v[0]).wrapping_shl(int(v[1]) as u32)),
                "shr"    => i2!(int(v[0]).wrapping_shr(int(v[1]) as u32)),
                _ => return Err(format!("undefined function: {name}")),
            }
        }
    }
}

pub fn eval_agg(args: &[Expr], env: &Env, product: bool) -> Result<Val, String> {
    let label = if product { "prod" } else { "sum" };
    if args.len() != 3 {
        return Err(format!("{label} expects 3 args: (fn, start, stop)"));
    }
    let start = eval(&args[1], env)?.num("start")? as i64;
    let stop  = eval(&args[2], env)?.num("stop")?  as i64;
    let mut acc = if product { 1.0 } else { 0.0 };
    for k in start..=stop {
        let v = call_fn1(&args[0], Val::Num(k as f64), env)?.num(label)?;
        if product { acc *= v; } else { acc += v; }
    }
    Ok(Val::Num(acc))
}

// Simpson's rule: integral(f, a, b) or integral(f, a, b, n)
pub fn eval_integral(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() < 3 || args.len() > 4 {
        return Err("integral(f, a, b) or integral(f, a, b, n)".into());
    }
    let a = eval(&args[1], env)?.num("a")?;
    let b = eval(&args[2], env)?.num("b")?;
    let n = if args.len() == 4 { eval(&args[3], env)?.num("n")? as usize } else { 1000 };
    let n = n + n % 2;
    let h = (b - a) / n as f64;
    let fa = call_fn1(&args[0], Val::Num(a), env)?.num("f")?;
    let fb = call_fn1(&args[0], Val::Num(b), env)?.num("f")?;
    let mut s = fa + fb;
    for i in 1..n {
        let x  = a + i as f64 * h;
        let fx = call_fn1(&args[0], Val::Num(x), env)?.num("f")?;
        s += fx * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    Ok(Val::Num(s * h / 3.0))
}

// 5-point stencil derivative: deriv(f, x) or deriv(f, x, dx)
pub fn eval_deriv(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err("deriv(f, x) or deriv(f, x, dx)".into());
    }
    let x  = eval(&args[1], env)?.num("x")?;
    let dx = if args.len() == 3 { eval(&args[2], env)?.num("dx")? } else { 1e-5 };
    let f  = |t: f64| call_fn1(&args[0], Val::Num(t), env).and_then(|v| v.num("f"));
    Ok(Val::Num(
        (-f(x + 2.0*dx)? + 8.0*f(x + dx)? - 8.0*f(x - dx)? + f(x - 2.0*dx)?) / (12.0 * dx)
    ))
}

// Apply a 1-arg function expression to a value.
pub fn call_fn1(f_expr: &Expr, x: Val, env: &Env) -> Result<Val, String> {
    match f_expr {
        Expr::Lambda(params, body) => {
            if params.len() != 1 {
                return Err("lambda must take exactly 1 argument here".into());
            }
            let mut local = env.clone();
            local.vars.insert(params[0].clone(), x);
            eval(body, &local)
        }
        Expr::Var(name) => {
            if let Some(Val::Fn(params, body)) = env.vars.get(name).cloned() {
                if params.len() != 1 {
                    return Err(format!("{name} must be a 1-arg function"));
                }
                let mut local = env.clone();
                local.vars.insert(params[0].clone(), x);
                return eval(&body, &local);
            }
            if let Some((params, body)) = env.fns.get(name).cloned() {
                if params.len() != 1 {
                    return Err(format!("{name} must be a 1-arg function"));
                }
                let mut local = env.clone();
                local.vars.insert(params[0].clone(), x);
                return eval(&body, &local);
            }
            let xn = x.num(name)?;
            eval(&Expr::Call(name.clone(), vec![Expr::Num(xn)]), env)
        }
        _ => Err("expected a function (e.g. x -> x^2 or a named fn)".into()),
    }
}
