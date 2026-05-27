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
    Tensor { data: Vec<f64>, shape: Vec<usize> },
}

impl Val {
    pub fn num(self, ctx: &str) -> Result<f64, String> {
        match self {
            Val::Num(n)        => Ok(n),
            Val::Complex(..)   => Err(format!("{ctx}: expected a real number, got complex")),
            Val::Fn(..)        => Err(format!("{ctx}: expected a number, got a function")),
            Val::Builtin(n)    => Err(format!("{ctx}: expected a number, got builtin '{n}'")),
            Val::Tuple(..)     => Err(format!("{ctx}: expected a number, got a tuple")),
            Val::Tensor { .. } => Err(format!("{ctx}: expected a number, got a tensor")),
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
            "sinh", "cosh", "tanh", "cbrt", "expm1",
            "sec", "csc", "cot",
            "floor", "ceil", "round",
            "trunc", "frac",
            "log", "log10", "log2",
            "sign", "signum", "id", "delta", "fact", "factorial", "not", "sinc",
            "sech", "csch",
            "erf", "erfc", "j0", "j1", "jinc",
            "step",
            "deg", "rad",
            "len", "length",
            "linspace", "range",
            "sort", "zip", "dot", "append", "concat", "flatten", "argmin", "argmax",
            "mean", "median", "mode", "std", "var",
            "compose", "partial",
            "gaussian", "gaussian_cdf",
            "filter", "reduce",
            "rand", "eps",
            "atan2", "min", "max", "pow", "hypot",
            "gcd", "lcm",
            "and", "or", "xor", "nand", "nor", "xnor", "shl", "shr",
            "lt", "leq", "gt", "geq", "eq", "neq",
            "if",
            "fft", "ifft",
            "sum", "prod", "integral", "deriv", "map", "graph",
            // Tensor ops
            "matrix", "zeros", "ones", "eye", "diag",
            "shape", "rows", "cols", "transpose", "trace", "norm",
            "row", "col", "matmul",
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
        | "sinh" | "cosh" | "tanh" | "cbrt" | "expm1"
        | "sec" | "csc" | "cot"
        | "floor" | "ceil" | "round" | "trunc" | "frac"
        | "log" | "log10" | "log2"
        | "sign" | "signum" | "id" | "delta" | "fact" | "factorial" | "not" | "sinc"
        | "sech" | "csch"
        | "erf" | "erfc" | "j0" | "j1" | "jinc"
        | "step"
        | "deg" | "rad"
        | "len" | "length"
        | "linspace" | "range"
        | "sort" | "zip" | "dot" | "append" | "concat" | "flatten" | "argmin" | "argmax"
        | "mean" | "median" | "mode" | "std" | "var"
        | "compose" | "partial"
        | "gaussian" | "gaussian_cdf"
        | "filter" | "reduce"
        | "rand" | "eps"
        | "min" | "max" | "pow" | "hypot" | "gcd" | "lcm"
        | "and" | "or" | "xor" | "nand" | "nor" | "xnor" | "shl" | "shr"
        | "lt" | "leq" | "gt" | "geq" | "eq" | "neq"
        | "if"
        | "fft" | "ifft"
        | "sum" | "prod" | "integral" | "deriv" | "map" | "graph"
        | "matrix" | "zeros" | "ones" | "eye" | "diag"
        | "shape" | "rows" | "cols" | "transpose" | "trace" | "norm"
        | "row" | "col" | "matmul"
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
        Val::Tensor { data, shape } => {
            if shape.is_empty() || data.is_empty() { return "[]".into(); }
            if shape.len() == 1 {
                let items: Vec<String> = data.iter().map(|x| fmt_f(*x)).collect();
                return format!("[{}]", items.join(", "));
            }
            if shape.len() == 2 {
                let (r, c) = (shape[0], shape[1]);
                let cells: Vec<Vec<String>> = (0..r).map(|i| {
                    (0..c).map(|j| fmt_f(data[i * c + j])).collect()
                }).collect();
                let col_widths: Vec<usize> = (0..c).map(|j| {
                    cells.iter().map(|row| row[j].len()).max().unwrap_or(0)
                }).collect();
                let rows: Vec<String> = cells.into_iter().map(|row| {
                    let padded: Vec<String> = row.into_iter().zip(&col_widths)
                        .map(|(s, &w)| format!("{:>w$}", s))
                        .collect();
                    format!("[{}]", padded.join("  "))
                }).collect();
                return rows.join("\n");
            }
            format!("<tensor shape={:?}>", shape)
        }
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
        Val::Tensor { .. } => Err("expected a number, got a tensor".into()),
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
        Val::Tensor { data, shape } => {
            let new_data: Result<Vec<f64>, _> = data.into_iter()
                .map(|x| f(Val::Num(x))?.num("broadcast"))
                .collect();
            Ok(Val::Tensor { data: new_data?, shape })
        }
        other => f(other),
    }
}

// ── Tensor helpers ────────────────────────────────────────────────────────────

fn binop_tensor(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    match (lv, rv) {
        (Val::Tensor { data: ld, shape: ls }, Val::Tensor { data: rd, shape: rs }) => {
            if ls != rs {
                return Err(format!("tensor op tensor: shape mismatch ({:?} vs {:?})", ls, rs));
            }
            let out: Result<Vec<f64>, _> = ld.into_iter().zip(rd)
                .map(|(l, r)| scalar_binop(Val::Num(l), op, Val::Num(r))?.num("tensor op"))
                .collect();
            Ok(Val::Tensor { data: out?, shape: ls })
        }
        (Val::Tensor { data, shape }, scalar) => {
            let s = scalar.num("tensor op scalar")?;
            let out: Result<Vec<f64>, _> = data.into_iter()
                .map(|x| scalar_binop(Val::Num(x), op, Val::Num(s))?.num("tensor op"))
                .collect();
            Ok(Val::Tensor { data: out?, shape })
        }
        (scalar, Val::Tensor { data, shape }) => {
            let s = scalar.num("scalar op tensor")?;
            let out: Result<Vec<f64>, _> = data.into_iter()
                .map(|x| scalar_binop(Val::Num(s), op, Val::Num(x))?.num("tensor op"))
                .collect();
            Ok(Val::Tensor { data: out?, shape })
        }
        _ => unreachable!(),
    }
}

// ── Builtin dispatch ──────────────────────────────────────────────────────────

pub fn eval_builtin(name: &str, vals: Vec<Val>, env: &Env) -> Result<Val, String> {
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
        "sech" => b1!(|v| {
            let (a, b) = to_complex(v)?;
            let (cr, ci) = (a.cosh() * b.cos(), a.sinh() * b.sin());
            let d = cr*cr + ci*ci;
            if d == 0.0 { return Err("sech: undefined".into()); }
            Ok(make_complex(cr/d, -ci/d))
        }),
        "csch" => b1!(|v| {
            let (a, b) = to_complex(v)?;
            let (cr, ci) = (a.sinh() * b.cos(), a.cosh() * b.sin());
            let d = cr*cr + ci*ci;
            if d == 0.0 { return Err("csch: undefined at zero".into()); }
            Ok(make_complex(cr/d, -ci/d))
        }),
        "erf"  => b1!(|v| Ok(Val::Num(libm::erf(v.num("erf")?)))),
        "erfc" => b1!(|v| Ok(Val::Num(libm::erfc(v.num("erfc")?)))),
        "j0"   => b1!(|v| Ok(Val::Num(libm::j0(v.num("j0")?)))),
        "j1"   => b1!(|v| Ok(Val::Num(libm::j1(v.num("j1")?)))),
        "jinc" => b1!(|v| {
            let x = v.num("jinc")?;
            if x == 0.0 { return Ok(Val::Num(0.5)); }
            Ok(Val::Num(libm::j1(x) / x))
        }),
        "sec" => b1!(|v| {
            let x = v.num("sec")?;
            let c = x.cos();
            if c == 0.0 { return Err("sec: undefined (cos is zero)".into()); }
            Ok(Val::Num(1.0 / c))
        }),
        "csc" => b1!(|v| {
            let x = v.num("csc")?;
            let s = x.sin();
            if s == 0.0 { return Err("csc: undefined (sin is zero)".into()); }
            Ok(Val::Num(1.0 / s))
        }),
        "cot" => b1!(|v| {
            let x = v.num("cot")?;
            let s = x.sin();
            if s == 0.0 { return Err("cot: undefined (sin is zero)".into()); }
            Ok(Val::Num(x.cos() / s))
        }),
        "step" => b1!(|v| Ok(Val::Num(match v.num("step")? {
            x if x < 0.0 => 0.0,
            x if x > 0.0 => 1.0,
            _             => 0.5,
        }))),
        "expm1"  => f1!(exp_m1),
        "cbrt"   => f1!(cbrt),
        "floor"  => f1!(floor), "ceil" => f1!(ceil),
        "round" => match vals.len() {
            1 => broadcast1(vals.into_iter().next().unwrap(),
                    |v| Ok(Val::Num(v.num("round")?.round()))),
            2 => {
                let mut it = vals.into_iter();
                let x = it.next().unwrap().num("round")?;
                let n = it.next().unwrap().num("round")? as i32;
                let f = 10f64.powi(n);
                Ok(Val::Num((x * f).round() / f))
            }
            n => Err(format!("round expects 1 or 2 args, got {n}")),
        },
        "trunc"  => f1!(trunc),
        "frac"   => b1!(|v| { let x = v.num("frac")?; Ok(Val::Num(x - x.trunc())) }),
        "log10"  => f1!(log10),
        "log" => match vals.len() {
            1 => broadcast1(vals.into_iter().next().unwrap(),
                    |v| Ok(Val::Num(v.num("log")?.log10()))),
            2 => {
                let mut it = vals.into_iter();
                let x    = it.next().unwrap().num("log")?;
                let base = it.next().unwrap().num("log")?;
                if base <= 0.0 || base == 1.0 {
                    return Err("log: base must be positive and ≠ 1".into());
                }
                Ok(Val::Num(x.ln() / base.ln()))
            }
            n => Err(format!("log expects 1 or 2 args, got {n}")),
        },
        "log2"   => f1!(log2),
        "sign" | "signum" => f1!(signum),
        "id"     => b1!(|v| { v.num("id").map(Val::Num) }),
        "delta"  => b1!(|v| Ok(Val::Num(if v.num("delta")? == 0.0 { 1.0 } else { 0.0 }))),
        "not"    => b1!(|v| Ok(Val::Num((int(v.num("not")?) == 0) as i64 as f64))),
        "deg"    => b1!(|v| Ok(Val::Num(v.num("deg")? * (180.0 / std::f64::consts::PI)))),
        "rad"    => b1!(|v| Ok(Val::Num(v.num("rad")? * (std::f64::consts::PI / 180.0)))),
        "len" | "length" => {
            arity(name, 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tuple(items) => Ok(Val::Num(items.len() as f64)),
                Val::Tensor { shape, .. } => Ok(Val::Num(shape[0] as f64)),
                _ => Err(format!("{name}: argument must be a tuple or tensor")),
            }
        }
        "fact" | "factorial" => b1!(|v| {
            let n = v.num("fact")? as u64;
            Ok(Val::Num((1..=n).map(|k| k as f64).product()))
        }),

        // ── Polymorphic min / max (scalar pair or tuple) ──────────────────────
        "min" | "max" => match (vals.len(), &vals[..]) {
            (1, _) => {
                let items = match vals.into_iter().next().unwrap() {
                    Val::Tuple(v) => v,
                    _ => return Err(format!("{name}: 1-arg form requires a tuple")),
                };
                if items.is_empty() { return Err(format!("{name}: empty tuple")); }
                let mut best = items[0].clone().num(name)?;
                for v in items.into_iter().skip(1) {
                    let n = v.num(name)?;
                    if name == "min" { if n < best { best = n; } }
                    else             { if n > best { best = n; } }
                }
                Ok(Val::Num(best))
            }
            (2, _) => {
                let mut it = vals.into_iter();
                let a = it.next().unwrap().num(name)?;
                let b = it.next().unwrap().num(name)?;
                if name == "min" { Ok(Val::Num(a.min(b))) } else { Ok(Val::Num(a.max(b))) }
            }
            (n, _) => Err(format!("{name} expects 1 or 2 args, got {n}")),
        },

        // ── Real 2-arg ────────────────────────────────────────────────────────
        "atan2" | "pow" | "hypot" |
        "gcd" | "lcm" | "and" | "or" | "xor" | "nand" | "nor" | "xnor" | "shl" | "shr" => {
            arity(name, 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num(name)?;
            let b = it.next().unwrap().num(name)?;
            match name {
                "atan2"  => Ok(Val::Num(a.atan2(b))),
                "pow"    => Ok(Val::Num(a.powf(b))),
                "hypot"  => Ok(Val::Num(a.hypot(b))),
                "gcd"    => Ok(Val::Num(gcd(int(a).unsigned_abs(), int(b).unsigned_abs()) as f64)),
                "lcm"    => Ok(Val::Num(lcm(int(a).unsigned_abs(), int(b).unsigned_abs()) as f64)),
                "and"    => Ok(Val::Num((int(a) & int(b)) as f64)),
                "or"     => Ok(Val::Num((int(a) | int(b)) as f64)),
                "xor"    => Ok(Val::Num((int(a) ^ int(b)) as f64)),
                "nand"   => Ok(Val::Num(((int(a) & int(b)) == 0) as i64 as f64)),
                "nor"    => Ok(Val::Num(((int(a) | int(b)) == 0) as i64 as f64)),
                "xnor"   => Ok(Val::Num(((int(a) ^ int(b)) == 0) as i64 as f64)),
                "shl"    => Ok(Val::Num(int(a).wrapping_shl(int(b) as u32) as f64)),
                "shr"    => Ok(Val::Num(int(a).wrapping_shr(int(b) as u32) as f64)),
                _        => unreachable!(),
            }
        }

        // ── Tuple combinators ─────────────────────────────────────────────────
        "sort" => {
            arity("sort", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() {
                Val::Tuple(v) => v,
                _ => return Err("sort: argument must be a tuple".into()),
            };
            let mut nums: Vec<f64> = items.into_iter()
                .map(|v| v.num("sort")).collect::<Result<_, _>>()?;
            nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Val::Tuple(nums.into_iter().map(Val::Num).collect()))
        }
        "zip" => {
            arity("zip", 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("zip: args must be tuples".into()) };
            let b = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("zip: args must be tuples".into()) };
            Ok(Val::Tuple(a.into_iter().zip(b).map(|(x, y)| Val::Tuple(vec![x, y])).collect()))
        }
        "dot" => {
            arity("dot", 2, vals.len())?;
            let mut it = vals.into_iter();
            match (it.next().unwrap(), it.next().unwrap()) {
                (Val::Tuple(a), Val::Tuple(b)) => {
                    if a.len() != b.len() { return Err(format!("dot: length mismatch ({} vs {})", a.len(), b.len())); }
                    let mut s = 0.0f64;
                    for (x, y) in a.into_iter().zip(b) { s += x.num("dot")? * y.num("dot")?; }
                    Ok(Val::Num(s))
                }
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh }) => {
                    if ash.len() != 1 || bsh.len() != 1 { return Err("dot: tensor args must be 1D".into()); }
                    if ash[0] != bsh[0] { return Err(format!("dot: length mismatch ({} vs {})", ash[0], bsh[0])); }
                    Ok(Val::Num(ad.iter().zip(bd.iter()).map(|(x, y)| x * y).sum()))
                }
                _ => Err("dot: args must be tuples or 1D tensors".into()),
            }
        }
        "append" => {
            arity("append", 2, vals.len())?;
            let mut it = vals.into_iter();
            let mut items = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("append: first arg must be a tuple".into()) };
            items.push(it.next().unwrap());
            Ok(Val::Tuple(items))
        }
        "concat" => {
            arity("concat", 2, vals.len())?;
            let mut it = vals.into_iter();
            let mut a = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("concat: args must be tuples".into()) };
            let     b = match it.next().unwrap() { Val::Tuple(v) => v, _ => return Err("concat: args must be tuples".into()) };
            a.extend(b);
            Ok(Val::Tuple(a))
        }
        "flatten" => {
            arity("flatten", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tuple(items) => Ok(Val::Tuple(items.into_iter().flat_map(|v| match v {
                    Val::Tuple(inner) => inner,
                    other             => vec![other],
                }).collect())),
                Val::Tensor { data, .. } => Ok(Val::Tuple(data.into_iter().map(Val::Num).collect())),
                _ => Err("flatten: argument must be a tuple or tensor".into()),
            }
        }
        "argmin" | "argmax" => {
            arity(name, 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err(format!("{name}: argument must be a tuple")) };
            if items.is_empty() { return Err(format!("{name}: empty tuple")); }
            let mut best_i = 0usize;
            let mut best_v = items[0].clone().num(name)?;
            for (i, v) in items.iter().enumerate().skip(1) {
                let n = v.clone().num(name)?;
                if name == "argmin" { if n < best_v { best_v = n; best_i = i; } }
                else                { if n > best_v { best_v = n; best_i = i; } }
            }
            Ok(Val::Num(best_i as f64))
        }

        // ── Statistics ────────────────────────────────────────────────────────
        "mean" => {
            arity("mean", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("mean: argument must be a tuple".into()) };
            if items.is_empty() { return Err("mean: empty tuple".into()); }
            let n = items.len() as f64;
            let s: f64 = items.into_iter().map(|v| v.num("mean")).collect::<Result<Vec<_>, _>>()?.into_iter().sum();
            Ok(Val::Num(s / n))
        }
        "median" => {
            arity("median", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("median: argument must be a tuple".into()) };
            if items.is_empty() { return Err("median: empty tuple".into()); }
            let mut nums: Vec<f64> = items.into_iter().map(|v| v.num("median")).collect::<Result<_, _>>()?;
            nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = nums.len() / 2;
            Ok(Val::Num(if nums.len() % 2 == 1 { nums[mid] } else { (nums[mid - 1] + nums[mid]) / 2.0 }))
        }
        "mode" => {
            arity("mode", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("mode: argument must be a tuple".into()) };
            if items.is_empty() { return Err("mode: empty tuple".into()); }
            let nums: Vec<f64> = items.into_iter().map(|v| v.num("mode")).collect::<Result<_, _>>()?;
            let mut best_val = nums[0];
            let mut best_cnt = 0usize;
            for &candidate in &nums {
                let cnt = nums.iter().filter(|&&x| x == candidate).count();
                if cnt > best_cnt { best_cnt = cnt; best_val = candidate; }
            }
            Ok(Val::Num(best_val))
        }
        "var" => {
            arity("var", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("var: argument must be a tuple".into()) };
            if items.is_empty() { return Err("var: empty tuple".into()); }
            let nums: Vec<f64> = items.into_iter().map(|v| v.num("var")).collect::<Result<_, _>>()?;
            let n = nums.len() as f64;
            let mean = nums.iter().sum::<f64>() / n;
            Ok(Val::Num(nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n))
        }
        "std" => {
            arity("std", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() { Val::Tuple(v) => v, _ => return Err("std: argument must be a tuple".into()) };
            if items.is_empty() { return Err("std: empty tuple".into()); }
            let nums: Vec<f64> = items.into_iter().map(|v| v.num("std")).collect::<Result<_, _>>()?;
            let n = nums.len() as f64;
            let mean = nums.iter().sum::<f64>() / n;
            Ok(Val::Num((nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()))
        }

        // ── Function combinators ──────────────────────────────────────────────
        "compose" => {
            arity("compose", 2, vals.len())?;
            let mut it = vals.into_iter();
            let f = it.next().unwrap();
            let g = it.next().unwrap();
            match (&f, &g) {
                (Val::Fn(..) | Val::Builtin(_), Val::Fn(..) | Val::Builtin(_)) => Ok(compose_fns(f, g)),
                _ => Err("compose: both arguments must be functions".into()),
            }
        }
        "partial" => {
            arity("partial", 2, vals.len())?;
            let mut it = vals.into_iter();
            let f = it.next().unwrap();
            let a = it.next().unwrap();
            match f {
                Val::Fn(params, body, mut captured) => {
                    if params.is_empty() { return Err("partial: function has no parameters".into()); }
                    let first = params[0].clone();
                    let rest  = params[1..].to_vec();
                    captured.insert(first, a);
                    Ok(Val::Fn(rest, body, captured))
                }
                Val::Builtin(bname) => {
                    let mut cap = HashMap::new();
                    cap.insert("__b__".into(), Val::Builtin(bname));
                    cap.insert("__a__".into(), a);
                    let body = Expr::Apply(
                        Box::new(Expr::Var("__b__".into())),
                        vec![Expr::Var("__a__".into()), Expr::Var("__z__".into())],
                    );
                    Ok(Val::Fn(vec!["__z__".into()], body, cap))
                }
                _ => Err("partial: first argument must be a function".into()),
            }
        }

        // ── Misc ──────────────────────────────────────────────────────────────
        "linspace" => {
            arity("linspace", 3, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num("linspace")?;
            let b = it.next().unwrap().num("linspace")?;
            let n = it.next().unwrap().num("linspace")? as usize;
            if n == 0 { return Err("linspace: n must be ≥ 1".into()); }
            if n == 1 { return Ok(Val::Tuple(vec![Val::Num(a)])); }
            Ok(Val::Tuple((0..n).map(|i| Val::Num(a + (b - a) * i as f64 / (n - 1) as f64)).collect()))
        }
        "range" => {
            arity("range", 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num("range")? as i64;
            let b = it.next().unwrap().num("range")? as i64;
            Ok(Val::Tuple((a..b).map(|n| Val::Num(n as f64)).collect()))
        }
        "gaussian" => {
            arity("gaussian", 3, vals.len())?;
            let mut it = vals.into_iter();
            let x     = it.next().unwrap().num("gaussian")?;
            let mu    = it.next().unwrap().num("gaussian")?;
            let sigma = it.next().unwrap().num("gaussian")?;
            if sigma == 0.0 { return Err("gaussian: sigma must be nonzero".into()); }
            let z = (x - mu) / sigma;
            Ok(Val::Num((1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()))
        }
        "gaussian_cdf" => {
            arity("gaussian_cdf", 3, vals.len())?;
            let mut it = vals.into_iter();
            let x     = it.next().unwrap().num("gaussian_cdf")?;
            let mu    = it.next().unwrap().num("gaussian_cdf")?;
            let sigma = it.next().unwrap().num("gaussian_cdf")?;
            if sigma == 0.0 { return Err("gaussian_cdf: sigma must be nonzero".into()); }
            Ok(Val::Num(0.5 * (1.0 + libm::erf((x - mu) / (sigma * std::f64::consts::SQRT_2)))))
        }
        "rand" => {
            match vals.len() {
                0 => Ok(Val::Num(rand::random::<f64>())),
                1 => {
                    let n = vals.into_iter().next().unwrap().num("rand")? as usize;
                    Ok(Val::Tuple((0..n).map(|_| Val::Num(rand::random::<f64>())).collect()))
                }
                2 => {
                    let mut it = vals.into_iter();
                    let a = it.next().unwrap().num("rand")?;
                    let b = it.next().unwrap().num("rand")?;
                    Ok(Val::Num(a + (b - a) * rand::random::<f64>()))
                }
                n => Err(format!("rand expects 0, 1, or 2 args, got {n}")),
            }
        }

        "eps" => {
            if vals.is_empty() { return Err("eps: requires at least 1 argument".into()); }
            let idxs: Vec<i64> = vals.into_iter()
                .map(|v| v.num("eps").map(|x| x as i64))
                .collect::<Result<_, _>>()?;
            let mut sorted = idxs.clone();
            sorted.sort();
            for w in sorted.windows(2) {
                if w[0] == w[1] { return Ok(Val::Num(0.0)); }
            }
            let n = idxs.len();
            let rank_of = |v: i64| sorted.iter().position(|&x| x == v).unwrap();
            let perm: Vec<usize> = idxs.iter().map(|&v| rank_of(v)).collect();
            let mut visited = vec![false; n];
            let mut cycles = 0usize;
            for i in 0..n {
                if !visited[i] {
                    cycles += 1;
                    let mut j = i;
                    while !visited[j] { visited[j] = true; j = perm[j]; }
                }
            }
            Ok(Val::Num(if (n - cycles) % 2 == 0 { 1.0 } else { -1.0 }))
        }

        "fft" | "ifft" => {
            arity(name, 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() {
                Val::Tuple(v) => v,
                _ => return Err(format!("{name}: argument must be a tuple")),
            };
            let n = items.len();
            if n == 0 { return Err(format!("{name}: empty tuple")); }
            use rustfft::num_complex::Complex64;
            let mut buf: Vec<Complex64> = items.into_iter().map(|v| match v {
                Val::Num(r)        => Ok(Complex64::new(r, 0.0)),
                Val::Complex(r, i) => Ok(Complex64::new(r, i)),
                _ => Err(format!("{name}: tuple elements must be numbers")),
            }).collect::<Result<_, _>>()?;
            let mut planner = rustfft::FftPlanner::new();
            if name == "fft" {
                planner.plan_fft_forward(n).process(&mut buf);
            } else {
                planner.plan_fft_inverse(n).process(&mut buf);
                let scale = 1.0 / n as f64;
                for c in &mut buf { *c *= scale; }
            }
            Ok(Val::Tuple(buf.into_iter().map(|c| make_complex(c.re, c.im)).collect()))
        }

        "lt" | "leq" | "gt" | "geq" | "eq" | "neq" => {
            arity(name, 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num(name)?;
            let b = it.next().unwrap().num(name)?;
            Ok(Val::Num(match name {
                "lt"  => if a < b  { 1.0 } else { 0.0 },
                "leq" => if a <= b { 1.0 } else { 0.0 },
                "gt"  => if a > b  { 1.0 } else { 0.0 },
                "geq" => if a >= b { 1.0 } else { 0.0 },
                "eq"  => if a == b { 1.0 } else { 0.0 },
                "neq" => if a != b { 1.0 } else { 0.0 },
                _     => unreachable!(),
            }))
        }

        // ── Tensor constructors ───────────────────────────────────────────────
        "matrix" => {
            if vals.len() != 3 { return Err("matrix(r, c, f) expects 3 args".into()); }
            let mut it = vals.into_iter();
            let r = it.next().unwrap().num("matrix")? as usize;
            let c = it.next().unwrap().num("matrix")? as usize;
            let f = it.next().unwrap();
            let mut data = Vec::with_capacity(r * c);
            for i in 0..r {
                for j in 0..c {
                    let v = apply_val(f.clone(), vec![Val::Num(i as f64), Val::Num(j as f64)], env)?;
                    data.push(v.num("matrix")?);
                }
            }
            Ok(Val::Tensor { data, shape: vec![r, c] })
        }
        "zeros" => {
            if vals.is_empty() { return Err("zeros(d0, d1, …) expects at least 1 arg".into()); }
            let shape: Vec<usize> = vals.into_iter()
                .map(|v| v.num("zeros").map(|x| x as usize))
                .collect::<Result<_, _>>()?;
            let n: usize = shape.iter().product();
            Ok(Val::Tensor { data: vec![0.0; n], shape })
        }
        "ones" => {
            if vals.is_empty() { return Err("ones(d0, d1, …) expects at least 1 arg".into()); }
            let shape: Vec<usize> = vals.into_iter()
                .map(|v| v.num("ones").map(|x| x as usize))
                .collect::<Result<_, _>>()?;
            let n: usize = shape.iter().product();
            Ok(Val::Tensor { data: vec![1.0; n], shape })
        }
        "eye" => {
            arity("eye", 1, vals.len())?;
            let n = vals.into_iter().next().unwrap().num("eye")? as usize;
            let mut data = vec![0.0f64; n * n];
            for i in 0..n { data[i * n + i] = 1.0; }
            Ok(Val::Tensor { data, shape: vec![n, n] })
        }
        "diag" => {
            arity("diag", 1, vals.len())?;
            let items = match vals.into_iter().next().unwrap() {
                Val::Tuple(v) => v,
                _ => return Err("diag: argument must be a tuple".into()),
            };
            let n = items.len();
            let nums: Vec<f64> = items.into_iter().map(|v| v.num("diag")).collect::<Result<_, _>>()?;
            let mut data = vec![0.0f64; n * n];
            for (i, &x) in nums.iter().enumerate() { data[i * n + i] = x; }
            Ok(Val::Tensor { data, shape: vec![n, n] })
        }

        // ── Tensor queries ────────────────────────────────────────────────────
        "shape" => {
            arity("shape", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { shape, .. } => Ok(Val::Tuple(shape.into_iter().map(|d| Val::Num(d as f64)).collect())),
                Val::Tuple(items) => Ok(Val::Tuple(vec![Val::Num(items.len() as f64)])),
                _ => Err("shape: argument must be a tensor or tuple".into()),
            }
        }
        "rows" => {
            arity("rows", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { shape, .. } if shape.len() >= 2 => Ok(Val::Num(shape[0] as f64)),
                Val::Tensor { .. } => Err("rows: tensor must be at least 2D".into()),
                _ => Err("rows: argument must be a 2D+ tensor".into()),
            }
        }
        "cols" => {
            arity("cols", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { shape, .. } if shape.len() >= 2 => Ok(Val::Num(shape[1] as f64)),
                Val::Tensor { .. } => Err("cols: tensor must be at least 2D".into()),
                _ => Err("cols: argument must be a 2D+ tensor".into()),
            }
        }

        // ── Tensor operations ─────────────────────────────────────────────────
        "transpose" => {
            arity("transpose", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let mut out = vec![0.0f64; r * c];
                    for i in 0..r {
                        for j in 0..c {
                            out[j * r + i] = data[i * c + j];
                        }
                    }
                    Ok(Val::Tensor { data: out, shape: vec![c, r] })
                }
                Val::Tensor { .. } => Err("transpose: tensor must be 2D".into()),
                _ => Err("transpose: argument must be a 2D tensor".into()),
            }
        }
        "trace" => {
            arity("trace", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let n = r.min(c);
                    Ok(Val::Num((0..n).map(|i| data[i * c + i]).sum()))
                }
                Val::Tensor { .. } => Err("trace: tensor must be 2D".into()),
                _ => Err("trace: argument must be a 2D tensor".into()),
            }
        }
        "norm" => {
            arity("norm", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, .. } => {
                    Ok(Val::Num(data.iter().map(|x| x * x).sum::<f64>().sqrt()))
                }
                Val::Tuple(items) => {
                    let sum: f64 = items.into_iter()
                        .map(|v| v.num("norm").map(|x| x * x))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter().sum();
                    Ok(Val::Num(sum.sqrt()))
                }
                _ => Err("norm: argument must be a tensor or tuple".into()),
            }
        }
        "row" => {
            arity("row", 2, vals.len())?;
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let i = it.next().unwrap().num("row")? as usize;
                    if i >= r { return Err(format!("row: index {i} out of range (rows={r})")); }
                    Ok(Val::Tuple((0..c).map(|j| Val::Num(data[i * c + j])).collect()))
                }
                Val::Tensor { .. } => Err("row: tensor must be 2D".into()),
                _ => Err("row: first argument must be a 2D tensor".into()),
            }
        }
        "col" => {
            arity("col", 2, vals.len())?;
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let j = it.next().unwrap().num("col")? as usize;
                    if j >= c { return Err(format!("col: index {j} out of range (cols={c})")); }
                    Ok(Val::Tuple((0..r).map(|i| Val::Num(data[i * c + j])).collect()))
                }
                Val::Tensor { .. } => Err("col: tensor must be 2D".into()),
                _ => Err("col: first argument must be a 2D tensor".into()),
            }
        }
        "matmul" => {
            arity("matmul", 2, vals.len())?;
            let mut it = vals.into_iter();
            match (it.next().unwrap(), it.next().unwrap()) {
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh })
                    if ash.len() == 2 && bsh.len() == 2 =>
                {
                    let (ar, ac) = (ash[0], ash[1]);
                    let (br, bc) = (bsh[0], bsh[1]);
                    if ac != br {
                        return Err(format!("matmul: shape mismatch ({ar}×{ac}) @ ({br}×{bc})"));
                    }
                    let mut out = vec![0.0f64; ar * bc];
                    for i in 0..ar {
                        for k in 0..ac {
                            for j in 0..bc {
                                out[i * bc + j] += ad[i * ac + k] * bd[k * bc + j];
                            }
                        }
                    }
                    Ok(Val::Tensor { data: out, shape: vec![ar, bc] })
                }
                _ => Err("matmul: both arguments must be 2D tensors".into()),
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
                    Val::Tensor { data, shape } => {
                        let scaled = data.iter().map(|&x| s * x).collect();
                        return Ok(Val::Tensor { data: scaled, shape: shape.clone() });
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
        Val::Tensor { .. } => Err("tensors are not callable".into()),
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
            Op::Lt       => if la < ra  { 1.0 } else { 0.0 },
            Op::Gt       => if la > ra  { 1.0 } else { 0.0 },
            Op::LtEq     => if la <= ra { 1.0 } else { 0.0 },
            Op::GtEq     => if la >= ra { 1.0 } else { 0.0 },
            Op::Eq       => if la == ra { 1.0 } else { 0.0 },
            Op::Ne       => if la != ra { 1.0 } else { 0.0 },
            Op::And      => (int(*la) & int(*ra)) as f64,
            Op::Or       => (int(*la) | int(*ra)) as f64,
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
        Op::Eq       => Ok(Val::Num(if la == ra && lb == rb { 1.0 } else { 0.0 })),
        Op::Ne       => Ok(Val::Num(if la != ra || lb != rb { 1.0 } else { 0.0 })),
        Op::Lt | Op::Gt | Op::LtEq | Op::GtEq => Err("comparison not defined for complex numbers".into()),
        Op::And | Op::Or => Err("& and | not defined for complex numbers".into()),
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
            let v       = eval(expr, env)?;
            let idx_val = eval(idx, env)?;
            match (v, idx_val) {
                // ── Tuple indexing ────────────────────────────────────────────
                (Val::Tuple(items), Val::Num(n)) => {
                    let raw = n as i64;
                    let len = items.len() as i64;
                    let i   = if raw < 0 { (len + raw).max(0) as usize } else { raw as usize };
                    items.into_iter().nth(i).ok_or_else(|| format!("index {raw} out of range"))
                }
                (Val::Tuple(items), Val::Tuple(indices)) => {
                    let len = items.len() as i64;
                    let result: Result<Vec<Val>, _> = indices.into_iter().map(|iv| {
                        let n = iv.num("index")? as i64;
                        let i = if n < 0 { (len + n).max(0) as usize } else { n as usize };
                        items.iter().nth(i).cloned().ok_or_else(|| format!("index {n} out of range"))
                    }).collect();
                    Ok(Val::Tuple(result?))
                }
                // ── Tensor indexing ───────────────────────────────────────────
                (Val::Tensor { data, shape }, Val::Num(n)) => {
                    let raw = n as i64;
                    let dim0 = shape[0] as i64;
                    let i = if raw < 0 { (dim0 + raw).max(0) as usize } else { raw as usize };
                    if i >= shape[0] {
                        return Err(format!("tensor index {raw} out of range (size={})", shape[0]));
                    }
                    if shape.len() == 1 {
                        Ok(Val::Num(data[i]))
                    } else {
                        let sub_size: usize = shape[1..].iter().product();
                        let start = i * sub_size;
                        Ok(Val::Tensor {
                            data: data[start..start + sub_size].to_vec(),
                            shape: shape[1..].to_vec(),
                        })
                    }
                }
                (Val::Tensor { data, shape }, Val::Tuple(indices)) => {
                    if indices.len() != shape.len() {
                        return Err(format!(
                            "tensor: expected {} indices, got {}",
                            shape.len(), indices.len()
                        ));
                    }
                    let mut flat = 0usize;
                    let mut stride = 1usize;
                    // Compute flat index in row-major order (right-to-left strides)
                    let idx_nums: Vec<i64> = indices.into_iter()
                        .map(|v| v.num("tensor index").map(|x| x as i64))
                        .collect::<Result<_, _>>()?;
                    for k in (0..shape.len()).rev() {
                        let dim = shape[k] as i64;
                        let raw = idx_nums[k];
                        let i = if raw < 0 { (dim + raw).max(0) as usize } else { raw as usize };
                        if i >= shape[k] {
                            return Err(format!(
                                "tensor index {raw} out of range for dim {k} (size={})", shape[k]
                            ));
                        }
                        flat += i * stride;
                        stride *= shape[k];
                    }
                    Ok(Val::Num(data[flat]))
                }
                _ => Err("indexing requires a tuple or tensor".into()),
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
                    "if" => {
                        if arg_exprs.len() != 3 {
                            return Err("if(cond, a, b) expects 3 args".into());
                        }
                        let cond = eval(&arg_exprs[0], env)?.num("if")?;
                        return if cond != 0.0 { eval(&arg_exprs[1], env) }
                               else           { eval(&arg_exprs[2], env) };
                    }
                    "sum"      => return eval_agg(arg_exprs, env, false),
                    "prod"     => return eval_agg(arg_exprs, env, true),
                    "integral" => return eval_integral(arg_exprs, env),
                    "deriv"    => return eval_deriv(arg_exprs, env),
                    "graph" => return crate::graph::eval_graph(arg_exprs, env),
                    "map" => {
                        if arg_exprs.len() != 2 {
                            return Err("map(f, tuple) expects 2 args".into());
                        }
                        let second = eval(&arg_exprs[1], env)?;
                        return match second {
                            Val::Tuple(items) => {
                                let results: Result<Vec<Val>, _> = items.into_iter()
                                    .map(|item| call_fn1(&arg_exprs[0], item, env))
                                    .collect();
                                Ok(Val::Tuple(results?))
                            }
                            Val::Tensor { data, shape } => {
                                let results: Result<Vec<f64>, _> = data.into_iter()
                                    .map(|x| call_fn1(&arg_exprs[0], Val::Num(x), env)?.num("map"))
                                    .collect();
                                Ok(Val::Tensor { data: results?, shape })
                            }
                            other => Err(format!("map: second arg must be a tuple or tensor, got {}", fmt_val(&other))),
                        };
                    }
                    "filter" => {
                        if arg_exprs.len() != 2 {
                            return Err("filter(f, tuple) expects 2 args".into());
                        }
                        let items = match eval(&arg_exprs[1], env)? {
                            Val::Tuple(v) => v,
                            other => return Err(format!("filter: second arg must be a tuple, got {}", fmt_val(&other))),
                        };
                        let mut out = vec![];
                        for item in items {
                            let keep = call_fn1(&arg_exprs[0], item.clone(), env)?.num("filter")?;
                            if keep != 0.0 { out.push(item); }
                        }
                        return Ok(Val::Tuple(out));
                    }
                    "reduce" => {
                        if arg_exprs.len() != 2 {
                            return Err("reduce(f, tuple) expects 2 args".into());
                        }
                        let items = match eval(&arg_exprs[1], env)? {
                            Val::Tuple(v) => v,
                            other => return Err(format!("reduce: second arg must be a tuple, got {}", fmt_val(&other))),
                        };
                        if items.is_empty() { return Err("reduce: empty tuple".into()); }
                        let f_val = eval(&arg_exprs[0], env)?;
                        let mut acc = items[0].clone();
                        for item in items.into_iter().skip(1) {
                            acc = apply_val(f_val.clone(), vec![acc, item], env)?;
                        }
                        return Ok(acc);
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
            Val::Tensor { data, shape } => {
                Ok(Val::Tensor { data: data.into_iter().map(|x| -x).collect(), shape })
            }
            Val::Fn(..) | Val::Builtin(_) => Err("unary minus: expected a number".into()),
        },
        Expr::BinOp(l, op, r) => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
            // Tensor dispatch (before tuple, since Tensor is not a Tuple)
            if matches!((&lv, &rv), (Val::Tensor { .. }, _) | (_, Val::Tensor { .. })) {
                return binop_tensor(lv, op, rv);
            }
            // Whole-tuple equality/inequality before element-wise broadcast
            if matches!(op, Op::Eq | Op::Ne) {
                if let (Val::Tuple(ls), Val::Tuple(rs)) = (&lv, &rv) {
                    let equal = ls.len() == rs.len() &&
                        ls.iter().zip(rs.iter()).all(|(a, b)| {
                            matches!((a, b), (Val::Num(x), Val::Num(y)) if x == y)
                        });
                    return Ok(Val::Num(if matches!(op, Op::Eq) == equal { 1.0 } else { 0.0 }));
                }
            }
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
                    Op::Lt       => if la < ra  { 1.0 } else { 0.0 },
                    Op::Gt       => if la > ra  { 1.0 } else { 0.0 },
                    Op::LtEq     => if la <= ra { 1.0 } else { 0.0 },
                    Op::GtEq     => if la >= ra { 1.0 } else { 0.0 },
                    Op::Eq       => if la == ra { 1.0 } else { 0.0 },
                    Op::Ne       => if la != ra { 1.0 } else { 0.0 },
                    Op::And      => (int(*la) & int(*ra)) as f64,
                    Op::Or       => (int(*la) | int(*ra)) as f64,
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
                Op::Eq       => Ok(Val::Num(if la == ra && lb == rb { 1.0 } else { 0.0 })),
                Op::Ne       => Ok(Val::Num(if la != ra || lb != rb { 1.0 } else { 0.0 })),
                Op::Lt | Op::Gt | Op::LtEq | Op::GtEq => Err("comparison not defined for complex numbers".into()),
                Op::And | Op::Or => Err("& and | not defined for complex numbers".into()),
            }
        }
    }
}

pub fn eval_agg(args: &[Expr], env: &Env, product: bool) -> Result<Val, String> {
    let label = if product { "prod" } else { "sum" };
    // 1-arg form: sum(tuple), sum(tensor), prod(tuple), prod(tensor)
    if args.len() == 1 {
        return match eval(&args[0], env)? {
            Val::Tuple(items) => {
                let mut acc = if product { 1.0 } else { 0.0 };
                for v in items {
                    let n = v.num(label)?;
                    if product { acc *= n; } else { acc += n; }
                }
                Ok(Val::Num(acc))
            }
            Val::Tensor { data, .. } => {
                let acc: f64 = if product { data.iter().product() } else { data.iter().sum() };
                Ok(Val::Num(acc))
            }
            _ => Err(format!("{label}: 1-arg form requires a tuple or tensor")),
        };
    }
    if args.len() != 3 {
        return Err(format!("{label} expects sum(tuple) or {label}(fn, start, stop)"));
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
                match eval_builtin(&bname, vec![x], env) {
                    Err(e) if e.contains("expects 0") => return eval_builtin(&bname, vec![], env),
                    other => return other,
                }
            }
            Err(format!("undefined function: {name}"))
        }
        _ => {
            let f_val = eval(f_expr, env)?;
            apply_val(f_val, vec![x], env)
        }
    }
}
