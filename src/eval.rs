use std::collections::HashMap;
use crate::ast::{Expr, BlockStmt, Op, Def};

// ── Values ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Val {
    Num(f64),
    Complex(f64, f64),
    Fn(Vec<String>, Expr, HashMap<String, Val>),
    Builtin(String),
    Tuple(Vec<Val>),
}

impl Val {
    pub fn num(self, ctx: &str) -> Result<f64, String> {
        match self {
            Val::Num(n)        => Ok(n),
            Val::Complex(..)   => Err(format!("{ctx}: expected a real number, got complex")),
            Val::Fn(..)        => Err(format!("{ctx}: expected a number, got a function")),
            Val::Builtin(n)    => Err(format!("{ctx}: expected a number, got builtin '{n}'")),
            Val::Tuple(..)     => Err(format!("{ctx}: expected a number, got a tuple")),
        }
    }
}

// ── Environment ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Env {
    pub vars: HashMap<String, Val>,
}

impl Env {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert("pi".into(),  Val::Num(std::f64::consts::PI));
        vars.insert("e".into(),   Val::Num(std::f64::consts::E));
        vars.insert("phi".into(), Val::Num(1.618033988749895));
        vars.insert("inf".into(), Val::Num(f64::INFINITY));
        vars.insert("i".into(),   Val::Complex(0.0, 1.0));
        for name in &[
            "abs", "re", "im", "arg", "conj", "sqrt", "exp", "ln",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "cbrt",
            "floor", "ceil", "round",
            "log", "log10", "log2",
            "sign", "signum", "id", "delta", "fact", "factorial", "not", "sinc",
            "atan2", "min", "max", "pow", "hypot",
            "gcd", "lcm",
            "and", "or", "xor", "nand", "nor", "xnor", "shl", "shr",
            "sum", "prod", "integral", "deriv", "map", "graph",
        ] {
            vars.insert(name.to_string(), Val::Builtin(name.to_string()));
        }
        Self { vars }
    }
}

pub fn is_protected(name: &str) -> bool {
    matches!(name,
        "pi" | "e" | "phi" | "inf" | "i"
        | "abs" | "re" | "im" | "arg" | "conj" | "sqrt" | "exp" | "ln"
        | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2"
        | "sinh" | "cosh" | "tanh" | "cbrt"
        | "floor" | "ceil" | "round"
        | "log" | "log10" | "log2"
        | "sign" | "signum" | "id" | "delta" | "fact" | "factorial" | "not" | "sinc"
        | "min" | "max" | "pow" | "hypot" | "gcd" | "lcm"
        | "and" | "or" | "xor" | "nand" | "nor" | "xnor" | "shl" | "shr"
        | "sum" | "prod" | "integral" | "deriv" | "map" | "graph"
    )
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
        Val::Fn(params, _, _) => format!("<fn({}) = …>", params.join(", ")),
        Val::Builtin(name) => format!("<builtin {name}>"),
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
        Val::Builtin(n)    => Err(format!("expected a number, got builtin '{n}'")),
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

// ── Broadcasting ──────────────────────────────────────────────────────────────

fn broadcast1(v: Val, f: impl Fn(Val) -> Result<Val, String>) -> Result<Val, String> {
    match v {
        Val::Tuple(items) => Ok(Val::Tuple(
            items.into_iter().map(f).collect::<Result<_, _>>()?
        )),
        other => f(other),
    }
}

// ── Builtin dispatch ──────────────────────────────────────────────────────────

pub fn eval_builtin(name: &str, vals: Vec<Val>, _env: &Env) -> Result<Val, String> {
    macro_rules! b1 {
        ($closure:expr) => {{
            arity(name, 1, vals.len())?;
            broadcast1(vals.into_iter().next().unwrap(), $closure)
        }};
    }
    macro_rules! f1 {
        ($method:ident) => {{
            arity(name, 1, vals.len())?;
            broadcast1(vals.into_iter().next().unwrap(), |v| Ok(Val::Num(v.num(name)?.$method())))
        }};
    }

    match name {
        // ── Complex-capable 1-arg ─────────────────────────────────────────────
        "abs"  => b1!(|v| match v {
            Val::Num(n)        => Ok(Val::Num(n.abs())),
            Val::Complex(a, b) => Ok(Val::Num((a*a + b*b).sqrt())),
            _ => Err(format!("{name}: expected a number")),
        }),
        "re"   => b1!(|v| {
            let (a, _) = to_complex(v)?; Ok(Val::Num(a))
        }),
        "im"   => b1!(|v| {
            let (_, b) = to_complex(v)?; Ok(Val::Num(b))
        }),
        "arg"  => b1!(|v| {
            let (a, b) = to_complex(v)?;
            Ok(Val::Num(if b == 0.0 { if a >= 0.0 { 0.0 } else { std::f64::consts::PI } } else { b.atan2(a) }))
        }),
        "conj" => b1!(|v| {
            let (a, b) = to_complex(v)?; Ok(make_complex(a, -b))
        }),
        "sqrt" => b1!(|v| match v {
            Val::Num(n) if n >= 0.0 => Ok(Val::Num(n.sqrt())),
            Val::Num(n)             => Ok(Val::Complex(0.0, (-n).sqrt())),
            Val::Complex(a, b) => {
                let r = (a*a + b*b).sqrt().sqrt();
                let theta = b.atan2(a) / 2.0;
                Ok(make_complex(r * theta.cos(), r * theta.sin()))
            }
            _ => Err(format!("{name}: expected a number")),
        }),
        "exp"  => b1!(|v| {
            let (a, b) = to_complex(v)?;
            let m = a.exp();
            Ok(make_complex(m * b.cos(), m * b.sin()))
        }),
        "ln"   => b1!(|v| match v {
            Val::Num(n) if n >= 0.0 => Ok(Val::Num(n.ln())),
            Val::Num(n)             => Ok(make_complex((-n).ln(), std::f64::consts::PI)),
            Val::Complex(a, b)      => Ok(make_complex((a*a+b*b).sqrt().ln(), b.atan2(a))),
            _ => Err(format!("{name}: expected a number")),
        }),

        // ── Real 1-arg ────────────────────────────────────────────────────────
        "sin" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            Ok(make_complex(x.sin() * y.cosh(), x.cos() * y.sinh()))
        }),
        "cos" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            Ok(make_complex(x.cos() * y.cosh(), -x.sin() * y.sinh()))
        }),
        "tan" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            let (sr, si) = (x.sin() * y.cosh(), x.cos() * y.sinh());
            let (cr, ci) = (x.cos() * y.cosh(), -x.sin() * y.sinh());
            let d = cr * cr + ci * ci;
            if d == 0.0 { return Err("tan: undefined (cosine is zero)".into()); }
            Ok(make_complex((sr*cr + si*ci)/d, (si*cr - sr*ci)/d))
        }),
        "sinc" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            if x == 0.0 && y == 0.0 { return Ok(Val::Num(1.0)); }
            let (sr, si) = (x.sin() * y.cosh(), x.cos() * y.sinh());
            let d = x * x + y * y;
            Ok(make_complex((sr*x + si*y)/d, (si*x - sr*y)/d))
        }),
        "asin"   => f1!(asin),  "acos"   => f1!(acos),  "atan" => f1!(atan),
        "sinh"   => f1!(sinh),  "cosh"   => f1!(cosh),  "tanh" => f1!(tanh),
        "cbrt"   => f1!(cbrt),
        "floor"  => f1!(floor), "ceil"   => f1!(ceil),  "round" => f1!(round),
        "log" | "log10" => f1!(log10),
        "log2"   => f1!(log2),
        "sign" | "signum" => f1!(signum),
        "id"     => b1!(|v| { v.num("id").map(Val::Num) }),
        "delta"  => b1!(|v| Ok(Val::Num(if v.num("delta")? == 0.0 { 1.0 } else { 0.0 }))),
        "not"    => b1!(|v| Ok(Val::Num(!int(v.num("not")?) as f64))),
        "fact" | "factorial" => b1!(|v| {
            let n = v.num("fact")? as u64;
            Ok(Val::Num((1..=n).map(|k| k as f64).product()))
        }),

        // ── Real 2-arg ────────────────────────────────────────────────────────
        "atan2" | "min" | "max" | "pow" | "hypot" |
        "gcd" | "lcm" | "and" | "or" | "xor" | "nand" | "nor" | "xnor" | "shl" | "shr" => {
            arity(name, 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num(name)?;
            let b = it.next().unwrap().num(name)?;
            match name {
                "atan2"  => Ok(Val::Num(a.atan2(b))),
                "min"    => Ok(Val::Num(a.min(b))),
                "max"    => Ok(Val::Num(a.max(b))),
                "pow"    => Ok(Val::Num(a.powf(b))),
                "hypot"  => Ok(Val::Num(a.hypot(b))),
                "gcd"    => Ok(Val::Num(gcd(int(a).unsigned_abs(), int(b).unsigned_abs()) as f64)),
                "lcm"    => Ok(Val::Num(lcm(int(a).unsigned_abs(), int(b).unsigned_abs()) as f64)),
                "and"    => Ok(Val::Num((int(a) & int(b)) as f64)),
                "or"     => Ok(Val::Num((int(a) | int(b)) as f64)),
                "xor"    => Ok(Val::Num((int(a) ^ int(b)) as f64)),
                "nand"   => Ok(Val::Num((!(int(a) & int(b))) as f64)),
                "nor"    => Ok(Val::Num((!(int(a) | int(b))) as f64)),
                "xnor"   => Ok(Val::Num((!(int(a) ^ int(b))) as f64)),
                "shl"    => Ok(Val::Num(int(a).wrapping_shl(int(b) as u32) as f64)),
                "shr"    => Ok(Val::Num(int(a).wrapping_shr(int(b) as u32) as f64)),
                _        => unreachable!(),
            }
        }

        _ => Err(format!("undefined function: {name}")),
    }
}

// ── Value application ─────────────────────────────────────────────────────────

pub fn apply_val(f: Val, args: Vec<Val>, env: &Env) -> Result<Val, String> {
    match f {
        Val::Builtin(ref name) => eval_builtin(name, args, env),
        Val::Fn(ref params, ref body, ref captured) => {
            let n = params.len();
            let k = args.len();
            // All args are Fn → compose (only single arg supported)
            if k == 1 {
                if let Val::Fn(_, _, _) = &args[0] {
                    let g = args.into_iter().next().unwrap();
                    return Ok(compose_fns(f, g));
                }
                // Single n-tuple arg → destructure into n params
                if let Val::Tuple(ref items) = args[0] {
                    if items.len() == n {
                        let mut local = make_local(env, captured);
                        for (p, v) in params.iter().zip(items.iter()) {
                            local.vars.insert(p.clone(), v.clone());
                        }
                        return eval(body, &local);
                    }
                }
                // Single scalar arg with 1-param fn → direct apply
                if n == 1 {
                    let mut local = make_local(env, captured);
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
                        let mut local = make_local(env, captured);
                        for (p, v) in params.iter().zip(items) { local.vars.insert(p.clone(), v); }
                        eval(body, &local)
                    } else { unreachable!() }
                }).collect();
                return Ok(Val::Tuple(results?));
            }
            // k scalar args, 1-param fn → map → k-tuple
            if n == 1 {
                let results: Result<Vec<Val>, _> = args.into_iter().map(|a| {
                    let mut local = make_local(env, captured);
                    local.vars.insert(params[0].clone(), a);
                    eval(body, &local)
                }).collect();
                return Ok(Val::Tuple(results?));
            }
            // k == n scalar args → direct apply
            if k == n {
                let mut local = make_local(env, captured);
                for (p, v) in params.iter().zip(args) { local.vars.insert(p.clone(), v); }
                return eval(body, &local);
            }
            Err(format!("function expects {n} args, got {k}"))
        }
        Val::Num(s) => {
            if args.len() == 1 {
                match &args[0] {
                    Val::Fn(_, _, _) => {
                        return Ok(scale_fn(s, args.into_iter().next().unwrap()));
                    }
                    Val::Num(n) => return Ok(Val::Num(s * n)),
                    Val::Tuple(items) => {
                        let scaled: Vec<Val> = items.iter().map(|v| match v {
                            Val::Num(n) => Val::Num(s * n),
                            _ => v.clone(),
                        }).collect();
                        return Ok(Val::Tuple(scaled));
                    }
                    Val::Complex(a, b) => return Ok(make_complex(s * a, s * b)),
                    Val::Builtin(_) => return Err("cannot scale a builtin function".into()),
                }
            }
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

// Three-layer env merge: global scope → closure's captured env → param bindings.
// Global scope provides forward-declared names; captured env provides lexical closure.
fn make_local(global: &Env, captured: &HashMap<String, Val>) -> Env {
    let mut vars = global.vars.clone();
    vars.extend(captured.iter().map(|(k, v)| (k.clone(), v.clone())));
    Env { vars }
}

fn compose_fns(f: Val, g: Val) -> Val {
    let mut captured = HashMap::new();
    captured.insert("__f__".into(), f);
    captured.insert("__g__".into(), g);
    let body = Expr::Apply(
        Box::new(Expr::Var("__f__".into())),
        vec![Expr::Apply(
            Box::new(Expr::Var("__g__".into())),
            vec![Expr::Var("__z__".into())],
        )],
    );
    Val::Fn(vec!["__z__".into()], body, captured)
}

fn scale_fn(s: f64, g: Val) -> Val {
    let mut captured = HashMap::new();
    captured.insert("__g__".into(), g);
    let body = Expr::BinOp(
        Box::new(Expr::Num(s)),
        Op::Mul,
        Box::new(Expr::Apply(
            Box::new(Expr::Var("__g__".into())),
            vec![Expr::Var("__z__".into())],
        )),
    );
    Val::Fn(vec!["__z__".into()], body, captured)
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
        Expr::Lambda(p, b) => Ok(Val::Fn(p.clone(), *b.clone(), env.vars.clone())),
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
                            if is_protected(name) {
                                return Err(format!("cannot redefine built-in '{name}'"));
                            }
                            let v = eval(expr, &child)?;
                            child.vars.insert(name.clone(), v);
                        }
                        Def::Func(name, params, body) => {
                            if is_protected(name) {
                                return Err(format!("cannot redefine built-in '{name}'"));
                            }
                            let mut captured = child.vars.clone();
                            let fn_val = Val::Fn(params.clone(), body.clone(), captured.clone());
                            captured.insert(name.clone(), fn_val);
                            child.vars.insert(name.clone(), Val::Fn(params.clone(), body.clone(), captured));
                        }
                    },
                    BlockStmt::Expr(e) => { last_val = eval(e, &child)?; }
                }
            }
            Ok(last_val)
        }
        Expr::Apply(f_expr, arg_exprs) => {
            // Special forms that need unevaluated Expr args (higher-order, still call_fn1-based)
            if let Expr::Var(name) = f_expr.as_ref() {
                match name.as_str() {
                    "sum"      => return eval_agg(arg_exprs, env, false),
                    "prod"     => return eval_agg(arg_exprs, env, true),
                    "integral" => return eval_integral(arg_exprs, env),
                    "deriv"    => return eval_deriv(arg_exprs, env),
                    "graph" => return crate::graph::eval_graph(arg_exprs, env),
                    "map" => {
                        if arg_exprs.len() != 2 {
                            return Err("map(f, tuple) expects 2 args".into());
                        }
                        let items = match eval(&arg_exprs[1], env)? {
                            Val::Tuple(items) => items,
                            other => return Err(format!("map: second arg must be a tuple, got {}", fmt_val(&other))),
                        };
                        let results: Result<Vec<Val>, _> = items.into_iter()
                            .map(|item| call_fn1(&arg_exprs[0], item, env))
                            .collect();
                        return Ok(Val::Tuple(results?));
                    }
                    _ => {}
                }
            }
            let f_val = eval(f_expr, env)?;
            let args: Result<Vec<Val>, _> = arg_exprs.iter().map(|a| eval(a, env)).collect();
            apply_val(f_val, args?, env)
        }
        Expr::Range(start, end) => {
            let a = eval(start, env)?.num("range")? as i64;
            let b = eval(end, env)?.num("range")? as i64;
            let items: Vec<Val> = if a <= b {
                (a..=b).map(|n| Val::Num(n as f64)).collect()
            } else {
                (b..=a).rev().map(|n| Val::Num(n as f64)).collect()
            };
            Ok(Val::Tuple(items))
        }
        Expr::Var(n) => env.vars.get(n).cloned()
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
            Val::Fn(..) | Val::Builtin(_) => Err("unary minus: expected a number".into()),
        },
        Expr::BinOp(l, op, r) => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
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
            if let Some(Val::Fn(params, body, captured)) = env.vars.get(name).cloned() {
                if params.len() != 1 {
                    return Err(format!("{name} must be a 1-arg function"));
                }
                let mut local = make_local(env, &captured);
                local.vars.insert(params[0].clone(), x);
                return eval(&body, &local);
            }
            if let Some(Val::Builtin(bname)) = env.vars.get(name).cloned() {
                return eval_builtin(&bname, vec![x], env);
            }
            Err(format!("undefined function: {name}"))
        }
        _ => Err("expected a function (e.g. x -> x^2 or a named fn)".into()),
    }
}
