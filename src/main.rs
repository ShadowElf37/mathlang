use std::cell::RefCell;
use std::collections::HashMap;

// ── Tokens ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64), Imag(f64), Ident(String),
    Plus, Minus, Star, Slash, SlashSlash, Percent, Caret, StarStar,
    LParen, RParen,
    Comma, Colon, Semicolon, Eq, Arrow,
    Eof,
}

// ── Lexer ─────────────────────────────────────────────────────────────────────

struct Lexer<'a> { src: &'a [u8], pos: usize }

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self { Self { src: src.as_bytes(), pos: 0 } }
    fn peek(&self) -> Option<u8> { self.src.get(self.pos).copied() }
    fn bump(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() { self.pos += 1; }
        b
    }

    fn tokenize(mut self) -> Vec<Token> {
        let mut out = Vec::new();
        loop {
            while self.peek().map_or(false, |b| b.is_ascii_whitespace()) { self.bump(); }
            match self.peek() {
                None => { out.push(Token::Eof); break; }
                Some(b) => match b {
                    b'+' => { self.bump(); out.push(Token::Plus); }
                    b'-' => {
                        self.bump();
                        if self.peek() == Some(b'>') { self.bump(); out.push(Token::Arrow); }
                        else { out.push(Token::Minus); }
                    }
                    b'*' => {
                        self.bump();
                        if self.peek() == Some(b'*') { self.bump(); out.push(Token::StarStar); }
                        else { out.push(Token::Star); }
                    }
                    b'/' => {
                        self.bump();
                        if self.peek() == Some(b'/') { self.bump(); out.push(Token::SlashSlash); }
                        else { out.push(Token::Slash); }
                    }
                    b'%' => { self.bump(); out.push(Token::Percent); }
                    b'^' => { self.bump(); out.push(Token::Caret); }
                    b'(' => { self.bump(); out.push(Token::LParen); }
                    b')' => { self.bump(); out.push(Token::RParen); }
                    b',' => { self.bump(); out.push(Token::Comma); }
                    b':' => { self.bump(); out.push(Token::Colon); }
                    b';' => { self.bump(); out.push(Token::Semicolon); }
                    b'=' => { self.bump(); out.push(Token::Eq); }
                    b if b.is_ascii_digit() || b == b'.' => {
                        let start = self.pos;
                        while self.peek().map_or(false, |b| b.is_ascii_digit() || b == b'.') {
                            self.bump();
                        }
                        if self.peek().map_or(false, |b| b == b'e' || b == b'E') {
                            self.bump();
                            if self.peek().map_or(false, |b| b == b'+' || b == b'-') { self.bump(); }
                            while self.peek().map_or(false, |b| b.is_ascii_digit()) { self.bump(); }
                        }
                        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
                        let n: f64 = s.parse().unwrap();
                        if self.peek() == Some(b'i')
                            && !self.src.get(self.pos + 1).map_or(false, |&b| b.is_ascii_alphanumeric() || b == b'_')
                        {
                            self.bump();
                            out.push(Token::Imag(n));
                        } else {
                            out.push(Token::Num(n));
                        }
                    }
                    b if b.is_ascii_alphabetic() || b == b'_' => {
                        let start = self.pos;
                        while self.peek().map_or(false, |b| b.is_ascii_alphanumeric() || b == b'_') {
                            self.bump();
                        }
                        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
                        out.push(Token::Ident(s.to_string()));
                    }
                    b => { eprintln!("unknown char: {}", b as char); self.bump(); }
                }
            }
        }
        out
    }
}

// ── AST ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    ImagLit(f64),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    Neg(Box<Expr>),
    Call(String, Vec<Expr>),
    Lambda(Vec<String>, Box<Expr>),
}

#[derive(Debug, Clone)]
enum Op { Add, Sub, Mul, Div, FloorDiv, Rem, Pow }

#[derive(Debug)]
enum Def {
    Var(String, Expr),
    Func(String, Vec<String>, Expr),
}

// ── Parser ────────────────────────────────────────────────────────────────────

fn is_sep(t: &Token) -> bool  { matches!(t, Token::Colon | Token::Semicolon) }
fn has_eq(toks: &[Token]) -> bool { toks.iter().any(|t| *t == Token::Eq) }

struct Parser { toks: Vec<Token>, pos: usize }

impl Parser {
    fn new(toks: Vec<Token>) -> Self { Self { toks, pos: 0 } }
    fn peek(&self) -> &Token { &self.toks[self.pos] }
    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() { self.pos += 1; }
        t
    }
    fn eat(&mut self, expected: &Token) -> Result<(), String> {
        if self.peek() == expected { self.bump(); Ok(()) }
        else { Err(format!("expected {:?}, got {:?}", expected, self.peek())) }
    }

    // Ident (Comma Ident)+ Arrow  — bare multi-arg lambda like: n, r -> expr
    fn is_multi_lambda(&self) -> bool {
        let mut p = self.pos;
        if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
        p += 1;
        let mut count = 0;
        loop {
            if !matches!(self.toks.get(p), Some(Token::Comma)) { break; }
            p += 1;
            if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
            p += 1;
            count += 1;
        }
        count > 0 && matches!(self.toks.get(p), Some(Token::Arrow))
    }

    // Ident (Comma Ident)* RParen Arrow — paren-wrapped lambda: (n, r) -> expr
    fn looks_like_paren_lambda(&self) -> bool {
        let mut p = self.pos;
        if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
        p += 1;
        loop {
            match self.toks.get(p) {
                Some(Token::RParen) => return matches!(self.toks.get(p + 1), Some(Token::Arrow)),
                Some(Token::Comma) => {
                    p += 1;
                    if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
                    p += 1;
                }
                _ => return false,
            }
        }
    }

    fn parse_multi_lambda(&mut self) -> Result<Expr, String> {
        let mut params = vec![];
        loop {
            match self.bump() {
                Token::Ident(s) => params.push(s),
                t => return Err(format!("expected param name, got {:?}", t)),
            }
            if *self.peek() == Token::Comma { self.bump(); } else { break; }
        }
        self.eat(&Token::Arrow)?;
        Ok(Expr::Lambda(params, self.expr()?.into()))
    }

    // REPL mode: no separator + has '=' → definitions only (store, no output)
    //            no separator + no '='  → expressions (evaluate and print)
    //            has separator           → standard parse
    fn parse_repl(&mut self) -> Result<(Vec<Def>, Vec<Expr>), String> {
        if self.toks.iter().any(is_sep) { return self.parse(); }
        if has_eq(&self.toks) {
            Ok((self.parse_defs()?, vec![]))
        } else {
            Ok((vec![], self.parse_expr_list()?))
        }
    }

    fn parse(&mut self) -> Result<(Vec<Def>, Vec<Expr>), String> {
        let has_sep = self.toks.iter().any(is_sep);
        let defs = if has_sep {
            let d = self.parse_defs()?;
            if !is_sep(self.peek()) {
                return Err(format!("expected ':' or ';', got {:?}", self.peek()));
            }
            self.bump();
            d
        } else {
            vec![]
        };
        Ok((defs, self.parse_expr_list()?))
    }

    fn parse_defs(&mut self) -> Result<Vec<Def>, String> {
        if is_sep(self.peek()) || *self.peek() == Token::Eof { return Ok(vec![]); }
        let mut defs = vec![self.parse_def()?];
        while *self.peek() == Token::Comma {
            self.bump();
            if is_sep(self.peek()) || *self.peek() == Token::Eof { break; }
            defs.push(self.parse_def()?);
        }
        Ok(defs)
    }

    fn parse_def(&mut self) -> Result<Def, String> {
        let name = match self.bump() {
            Token::Ident(s) => s,
            t => return Err(format!("expected name, got {:?}", t)),
        };
        if *self.peek() == Token::LParen {
            self.bump();
            let mut params = vec![];
            if *self.peek() != Token::RParen {
                loop {
                    match self.bump() {
                        Token::Ident(s) => params.push(s),
                        t => return Err(format!("expected param name, got {:?}", t)),
                    }
                    if *self.peek() == Token::Comma { self.bump(); } else { break; }
                }
            }
            self.eat(&Token::RParen)?;
            self.eat(&Token::Eq)?;
            Ok(Def::Func(name, params, self.expr()?))
        } else {
            self.eat(&Token::Eq)?;
            if self.is_multi_lambda() {
                Ok(Def::Var(name, self.parse_multi_lambda()?))
            } else {
                Ok(Def::Var(name, self.expr()?))
            }
        }
    }

    fn parse_expr_list(&mut self) -> Result<Vec<Expr>, String> {
        if *self.peek() == Token::Eof { return Ok(vec![]); }
        let mut exprs = vec![self.expr()?];
        while *self.peek() == Token::Comma {
            self.bump();
            if *self.peek() == Token::Eof { break; }
            exprs.push(self.expr()?);
        }
        Ok(exprs)
    }

    fn expr(&mut self) -> Result<Expr, String> { self.add() }

    fn add(&mut self) -> Result<Expr, String> {
        let mut l = self.mul()?;
        loop {
            l = match self.peek() {
                Token::Plus  => { self.bump(); Expr::BinOp(l.into(), Op::Add, self.mul()?.into()) }
                Token::Minus => { self.bump(); Expr::BinOp(l.into(), Op::Sub, self.mul()?.into()) }
                _ => break,
            };
        }
        Ok(l)
    }

    fn mul(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        loop {
            l = match self.peek() {
                Token::Star       => { self.bump(); Expr::BinOp(l.into(), Op::Mul,      self.unary()?.into()) }
                Token::Slash      => { self.bump(); Expr::BinOp(l.into(), Op::Div,      self.unary()?.into()) }
                Token::SlashSlash => { self.bump(); Expr::BinOp(l.into(), Op::FloorDiv, self.unary()?.into()) }
                Token::Percent    => { self.bump(); Expr::BinOp(l.into(), Op::Rem,      self.unary()?.into()) }
                _ => break,
            };
        }
        Ok(l)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if *self.peek() == Token::Minus { self.bump(); return Ok(Expr::Neg(self.pow()?.into())); }
        self.pow()
    }

    fn pow(&mut self) -> Result<Expr, String> {
        let base = self.primary()?;
        if matches!(self.peek(), Token::Caret | Token::StarStar) {
            self.bump();
            Ok(Expr::BinOp(base.into(), Op::Pow, self.unary()?.into()))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Token::Num(n)  => { self.bump(); Ok(Expr::Num(n)) }
            Token::Imag(n) => { self.bump(); Ok(Expr::ImagLit(n)) }
            Token::LParen => {
                self.bump();
                if self.looks_like_paren_lambda() {
                    let mut params = vec![];
                    loop {
                        match self.bump() {
                            Token::Ident(s) => params.push(s),
                            t => return Err(format!("expected param, got {:?}", t)),
                        }
                        match self.peek().clone() {
                            Token::Comma  => { self.bump(); }
                            Token::RParen => { self.bump(); break; }
                            ref t => return Err(format!("expected ',' or ')', got {:?}", t)),
                        }
                    }
                    self.eat(&Token::Arrow)?;
                    Ok(Expr::Lambda(params, self.expr()?.into()))
                } else {
                    let e = self.expr()?;
                    self.eat(&Token::RParen)?;
                    Ok(e)
                }
            }
            Token::Minus => { self.bump(); Ok(Expr::Neg(self.primary()?.into())) }
            Token::Ident(name) => {
                self.bump();
                if *self.peek() == Token::Arrow {
                    self.bump();
                    Ok(Expr::Lambda(vec![name], self.expr()?.into()))
                } else if *self.peek() == Token::LParen {
                    self.bump();
                    let mut args = vec![];
                    if *self.peek() != Token::RParen {
                        loop {
                            args.push(self.expr()?);
                            if *self.peek() == Token::Comma { self.bump(); } else { break; }
                        }
                    }
                    self.eat(&Token::RParen)?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            t => Err(format!("unexpected token: {:?}", t)),
        }
    }
}

// ── Values ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Val {
    Num(f64),
    Complex(f64, f64),
    Fn(Vec<String>, Expr),
}

impl Val {
    fn num(self, ctx: &str) -> Result<f64, String> {
        match self {
            Val::Num(n) => Ok(n),
            Val::Complex(..) => Err(format!("{ctx}: expected a real number, got complex")),
            Val::Fn(..) => Err(format!("{ctx}: expected a number, got a function")),
        }
    }
}

// ── Environment ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Env {
    vars: HashMap<String, Val>,
    fns:  HashMap<String, (Vec<String>, Expr)>,  // multi-arg named functions
}

impl Env {
    fn new() -> Self {
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

// ── Evaluator ─────────────────────────────────────────────────────────────────

#[inline] fn i(x: f64) -> i64 { x as i64 }

// Collapse a+bi to Num(a) when b is negligibly small relative to the magnitude.
fn make_complex(a: f64, b: f64) -> Val {
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

fn eval(expr: &Expr, env: &Env) -> Result<Val, String> {
    match expr {
        Expr::Num(n)      => Ok(Val::Num(*n)),
        Expr::ImagLit(n)  => Ok(if *n == 0.0 { Val::Num(0.0) } else { Val::Complex(0.0, *n) }),
        Expr::Lambda(p, b) => Ok(Val::Fn(p.clone(), *b.clone())),
        Expr::Var(n)      => env.vars.get(n).cloned()
            .ok_or_else(|| format!("undefined: {n}")),
        Expr::Neg(e) => match eval(e, env)? {
            Val::Num(n)        => Ok(Val::Num(-n)),
            Val::Complex(a, b) => Ok(make_complex(-a, -b)),
            Val::Fn(..)        => Err("unary minus: expected a number".into()),
        },
        Expr::BinOp(l, op, r) => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
            if let (Val::Num(la), Val::Num(ra)) = (&lv, &rv) {
                return Ok(Val::Num(match op {
                    Op::Add      => la + ra,
                    Op::Sub      => la - ra,
                    Op::Mul      => la * ra,
                    Op::Div      => la / ra,
                    Op::FloorDiv => (i(*la) / i(*ra)) as f64,
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
            // These receive raw Expr args so the first arg (a fn) isn't forced to f64
            match name.as_str() {
                "sum"      => return eval_agg(args, env, false),
                "prod"     => return eval_agg(args, env, true),
                "integral" => return eval_integral(args, env),
                "deriv"    => return eval_deriv(args, env),
                _ => {}
            }

            // Evaluate args to Val (may include functions)
            let vals: Result<Vec<Val>, _> = args.iter().map(|a| eval(a, env)).collect();
            let vals = vals?;

            // User-defined named function (multi-arg)
            if let Some((params, body)) = env.fns.get(name).cloned() {
                arity(name, params.len(), vals.len())?;
                let mut local = env.clone();
                for (p, v) in params.iter().zip(vals) { local.vars.insert(p.clone(), v); }
                return eval(&body, &local);
            }

            // Lambda stored as a variable (first-class)
            if let Some(Val::Fn(params, body)) = env.vars.get(name).cloned() {
                arity(name, params.len(), vals.len())?;
                let mut local = env.clone();
                for (p, v) in params.iter().zip(vals) { local.vars.insert(p.clone(), v); }
                return eval(&body, &local);
            }

            // Complex-capable builtins (operate on Val directly)
            macro_rules! cx1 {
                ($vname:ident, $real_arm:expr, $cx_arm:expr) => {{
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num($vname)        => $real_arm,
                        Val::Complex($vname, _) => $cx_arm,
                        Val::Fn(..) => return Err(format!("{name}: expected a number")),
                    });
                }};
            }
            match name.as_str() {
                "abs" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(n.abs()),
                        Val::Complex(a, b) => Val::Num((a*a + b*b).sqrt()),
                        Val::Fn(..) => return Err("abs: expected a number".into()),
                    });
                }
                "re"  => cx1!(n, Val::Num(n),  Val::Num(n)),
                "im"  => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(_)        => Val::Num(0.0),
                        Val::Complex(_, b) => Val::Num(b),
                        Val::Fn(..) => return Err("im: expected a number".into()),
                    });
                }
                "arg" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(if n >= 0.0 { 0.0 } else { std::f64::consts::PI }),
                        Val::Complex(a, b) => Val::Num(b.atan2(a)),
                        Val::Fn(..) => return Err("arg: expected a number".into()),
                    });
                }
                "conj" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(n),
                        Val::Complex(a, b) => make_complex(a, -b),
                        Val::Fn(..) => return Err("conj: expected a number".into()),
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
                        Val::Fn(..) => return Err("sqrt: expected a number".into()),
                    });
                }
                "exp" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n)        => Val::Num(n.exp()),
                        Val::Complex(a, b) => { let m = a.exp(); make_complex(m*b.cos(), m*b.sin()) }
                        Val::Fn(..) => return Err("exp: expected a number".into()),
                    });
                }
                "ln" => {
                    arity(name, 1, vals.len())?;
                    return Ok(match vals.into_iter().next().unwrap() {
                        Val::Num(n) if n >= 0.0 => Val::Num(n.ln()),
                        Val::Num(n)             => make_complex((-n).ln(), std::f64::consts::PI),
                        Val::Complex(a, b)      => make_complex((a*a + b*b).sqrt().ln(), b.atan2(a)),
                        Val::Fn(..) => return Err("ln: expected a number".into()),
                    });
                }
                _ => {}
            }

            // Real-only builtins — convert to f64 first
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
                "gcd"    => i2!(gcd(i(v[0]).unsigned_abs(), i(v[1]).unsigned_abs())),
                "lcm"    => i2!(lcm(i(v[0]).unsigned_abs(), i(v[1]).unsigned_abs())),
                "and"    => i2!(i(v[0]) & i(v[1])),
                "or"     => i2!(i(v[0]) | i(v[1])),
                "xor"    => i2!(i(v[0]) ^ i(v[1])),
                "nand"   => i2!(!(i(v[0]) & i(v[1]))),
                "nor"    => i2!(!(i(v[0]) | i(v[1]))),
                "xnor"   => i2!(!(i(v[0]) ^ i(v[1]))),
                "not"    => i1!(!i(v[0])),
                "shl"    => i2!(i(v[0]).wrapping_shl(i(v[1]) as u32)),
                "shr"    => i2!(i(v[0]).wrapping_shr(i(v[1]) as u32)),
                _ => return Err(format!("undefined function: {name}")),
            }
        }
    }
}

fn eval_agg(args: &[Expr], env: &Env, product: bool) -> Result<Val, String> {
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
// O(h^4) accuracy vs the trapezoidal O(h^2) for the same number of evaluations.
fn eval_integral(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() < 3 || args.len() > 4 {
        return Err("integral(f, a, b) or integral(f, a, b, n)".into());
    }
    let a = eval(&args[1], env)?.num("a")?;
    let b = eval(&args[2], env)?.num("b")?;
    let n = if args.len() == 4 { eval(&args[3], env)?.num("n")? as usize } else { 1000 };
    let n = n + n % 2; // Simpson's requires an even number of intervals
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
// O(h^4) accuracy vs central-difference O(h^2).
fn eval_deriv(args: &[Expr], env: &Env) -> Result<Val, String> {
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
// f_expr can be: Lambda, a Var pointing to Val::Fn or env.fns, or a Call (partially).
fn call_fn1(f_expr: &Expr, x: Val, env: &Env) -> Result<Val, String> {
    match f_expr {
        Expr::Lambda(params, body) => {
            if params.len() != 1 {
                return Err("lambda passed to sum/prod/integral/deriv must take exactly 1 argument".into());
            }
            let mut local = env.clone();
            local.vars.insert(params[0].clone(), x);
            eval(body, &local)
        }
        Expr::Var(name) => {
            // stored lambda
            if let Some(Val::Fn(params, body)) = env.vars.get(name).cloned() {
                if params.len() != 1 {
                    return Err(format!("{name} must be a 1-arg function for use in sum/prod/integral/deriv"));
                }
                let mut local = env.clone();
                local.vars.insert(params[0].clone(), x);
                return eval(&body, &local);
            }
            // named user-defined function
            if let Some((params, body)) = env.fns.get(name).cloned() {
                if params.len() != 1 {
                    return Err(format!("{name} must be a 1-arg function"));
                }
                let mut local = env.clone();
                local.vars.insert(params[0].clone(), x);
                return eval(&body, &local);
            }
            // builtin: construct a call and let eval dispatch it normally
            let xn = x.num(name)?;
            eval(&Expr::Call(name.clone(), vec![Expr::Num(xn)]), env)
        }
        _ => Err("first argument to sum/prod must be a function (e.g. x -> x^2 or a named fn)".into()),
    }
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

// ── Output formatting ─────────────────────────────────────────────────────────

fn fmt_val(v: &Val) -> String {
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
    }
}

// ── Known names ───────────────────────────────────────────────────────────────

const BUILTIN_FNS: &[&str] = &[
    "id", "fact", "factorial", "delta",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "sqrt", "cbrt", "abs", "sign", "signum",
    "floor", "ceil", "round",
    "ln", "log", "log10", "log2", "exp",
    "re", "im", "arg", "conj",
    "min", "max", "pow", "hypot", "gcd", "lcm",
    "and", "or", "xor", "nand", "nor", "xnor", "not", "shl", "shr",
    "sum", "prod", "integral", "deriv",
];

const BUILTIN_CONSTS: &[&str] = &["pi", "e", "tau", "phi", "inf", "i"];

// ── REPL & entry point ────────────────────────────────────────────────────────

struct MathHelper {
    user_names: RefCell<Vec<String>>,
    hinter:     rustyline::hint::HistoryHinter,
}

impl MathHelper {
    fn new() -> Self {
        Self { user_names: RefCell::new(vec![]), hinter: rustyline::hint::HistoryHinter {} }
    }
    fn update(&self, env: &Env) {
        let mut n = self.user_names.borrow_mut();
        n.clear();
        n.extend(env.fns.keys().cloned());
        n.extend(env.vars.keys().filter(|k| !BUILTIN_CONSTS.contains(&k.as_str())).cloned());
    }
}

fn highlight_line(line: &str, user: &[String]) -> String {
    if line.starts_with('!') { return format!("\x1b[33m{line}\x1b[0m"); }
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 64);
    let mut i = 0;
    while i < line.len() {
        // whitespace
        if b[i].is_ascii_whitespace() { out.push(b[i] as char); i += 1; continue; }
        // number
        if b[i].is_ascii_digit() || (b[i] == b'.' && b.get(i+1).map_or(false, |c| c.is_ascii_digit())) {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') { i += 1; }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') { i += 1; }
                while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            }
            out.push_str(&format!("\x1b[36m{}\x1b[0m", &line[s..i]));
            continue;
        }
        // identifier
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            let name = &line[s..i];
            if BUILTIN_CONSTS.contains(&name) {
                out.push_str(&format!("\x1b[36m{name}\x1b[0m"));          // cyan — constants
            } else if BUILTIN_FNS.contains(&name) || user.iter().any(|u| u == name) {
                out.push_str(&format!("\x1b[95m{name}\x1b[0m"));          // purple — functions
            } else {
                out.push_str(name);                                         // default — unknown
            }
            continue;
        }
        // multi-char operators
        if i + 1 < b.len() {
            match (b[i], b[i+1]) {
                (b'-', b'>') | (b'*', b'*') | (b'/', b'/') => {
                    out.push_str(&format!("\x1b[33m{}\x1b[0m", &line[i..i+2]));
                    i += 2; continue;
                }
                _ => {}
            }
        }
        // single-char operators
        if matches!(b[i], b'+' | b'-' | b'*' | b'/' | b'%' | b'^') {
            out.push_str(&format!("\x1b[33m{}\x1b[0m", b[i] as char));
        } else {
            out.push(b[i] as char);
        }
        i += 1;
    }
    out
}

impl rustyline::highlight::Highlighter for MathHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> std::borrow::Cow<'l, str> {
        std::borrow::Cow::Owned(highlight_line(line, &self.user_names.borrow()))
    }
    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool { true }
}

impl rustyline::completion::Completer for MathHelper {
    type Candidate = String;
    fn complete(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>)
        -> rustyline::Result<(usize, Vec<String>)>
    {
        if line.starts_with('!') {
            let cmds = ["!clear", "!defs", "!help", "!import "];
            return Ok((0, cmds.iter().filter(|&&c| c.starts_with(line)).map(|s| s.to_string()).collect()));
        }
        let start = line[..pos].rfind(|c: char| !c.is_alphanumeric() && c != '_').map_or(0, |i| i+1);
        let word = &line[start..pos];
        if word.is_empty() { return Ok((pos, vec![])); }
        let user = self.user_names.borrow();
        let mut cs: Vec<String> = BUILTIN_FNS.iter().copied()
            .chain(BUILTIN_CONSTS.iter().copied())
            .chain(user.iter().map(String::as_str))
            .filter(|s| s.starts_with(word) && *s != word)
            .map(str::to_string)
            .collect();
        cs.sort(); cs.dedup();
        Ok((start, cs))
    }
}

impl rustyline::hint::Hinter for MathHelper {
    type Hint = String;
    fn hint(&self, line: &str, pos: usize, ctx: &rustyline::Context<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl rustyline::validate::Validator for MathHelper {
    fn validate(&self, _: &mut rustyline::validate::ValidationContext<'_>)
        -> rustyline::Result<rustyline::validate::ValidationResult>
    {
        Ok(rustyline::validate::ValidationResult::Valid(None))
    }
}

impl rustyline::Helper for MathHelper {}

fn eval_line(line: &str, env: &mut Env, repl: bool) -> bool {
    let line = line.trim();
    if line.is_empty() { return true; }
    let toks = Lexer::new(line).tokenize();
    let mut parser = Parser::new(toks);
    let (defs, exprs) = match parser.parse_repl() {
        Ok(v) => v,
        Err(e) => { eprintln!("error: {e}"); return false; }
    };
    for def in &defs {
        match def {
            Def::Var(name, expr) => match eval(expr, env) {
                Ok(v) => { env.vars.insert(name.clone(), v); }
                Err(e) => { eprintln!("error: {e}"); return false; }
            },
            Def::Func(name, params, body) => {
                env.fns.insert(name.clone(), (params.clone(), body.clone()));
            }
        }
    }
    for expr in &exprs {
        match eval(expr, env) {
            Ok(v) => {
                if repl {
                    println!("\x1b[2mresult = \x1b[0m{}", fmt_val(&v));
                    env.vars.insert("result".into(), v);
                } else {
                    println!("{}", fmt_val(&v));
                }
            }
            Err(e) => { eprintln!("error: {e}"); return false; }
        }
    }
    true
}

fn show_defs(env: &Env) {
    let mut items: Vec<(String, String)> = vec![];
    for (k, v) in &env.vars {
        if BUILTIN_CONSTS.contains(&k.as_str()) || k == "result" { continue; }
        let display = match v {
            Val::Fn(params, _) => format!("fn({}) = …", params.join(", ")),
            _ => fmt_val(v),
        };
        items.push((k.clone(), display));
    }
    for (k, (params, _)) in &env.fns {
        items.push((k.clone(), format!("fn({}) = …", params.join(", "))));
    }
    items.sort_by(|(a,_),(b,_)| a.cmp(b));
    if items.is_empty() { println!("(nothing defined)"); }
    else { for (k, v) in &items { println!("{k} = {v}"); } }
}

fn expand_path(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        format!("{}/{rest}", std::env::var("HOME").unwrap_or_default())
    } else {
        p.to_string()
    }
}

// Import a file, printing a summary only when verbose=true.
// Errors within lines are always printed.
fn import_file(path: &str, display: &str, env: &mut Env, verbose: bool) {
    match std::fs::read_to_string(path) {
        Ok(src) => {
            let mut n = 0;
            for line in src.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                eval_line(line, env, false);
                n += 1;
            }
            if verbose { println!("imported {n} line(s) from {display}"); }
        }
        Err(e) => eprintln!("import {display}: {e}"),
    }
}

fn bang_command(cmd: &str, env: &mut Env) {
    let (name, arg) = cmd.split_once(' ').map_or((cmd, ""), |(a, b)| (a, b.trim()));
    match name.trim() {
        "help" => print!(concat!(
            "Commands:  !help  !import <file>  !defs  !clear\n",
            "Init file: ~/.mathlangrc (auto-imported on start; override with $MATHLANG_INIT)\n",
            "Exit:      q / quit / exit / Ctrl-D\n\n",
            "Syntax:    x = 3              variable\n",
            "           f(x) = x^2         named function\n",
            "           f = x -> x^2       lambda (first-class)\n",
            "           g = n,r -> n+r     multi-arg lambda\n",
            "           defs : exprs       define, then evaluate\n\n",
            "Operators: + - * / // % ^ **   -> (lambda)\n",
            "Aggregates: sum(f,a,b)  prod(f,a,b)  integral(f,a,b[,n])  deriv(f,x[,dx])\n",
            "Builtins:  sin cos tan asin acos atan atan2  sinh cosh tanh\n",
            "           sqrt cbrt abs sign floor ceil round\n",
            "           ln log log2 exp  min max pow hypot  gcd lcm\n",
            "           and or xor nand nor xnor not shl shr  id fact delta\n",
            "Complex:   i (unit)  2+3i  re im abs arg conj  sqrt exp ln\n",
            "           Arithmetic +−×÷^ all work on complex numbers.\n",
            "Constants: pi e tau phi inf i\n",
        )),
        "import" => {
            if arg.is_empty() { eprintln!("usage: !import <file>"); return; }
            let path = expand_path(arg);
            import_file(&path, arg, env, true);
        }
        "defs" | "vars" | "fns" => show_defs(env),
        "clear" => {
            let n = env.vars.iter().filter(|(k,_)| !BUILTIN_CONSTS.contains(&k.as_str())).count()
                  + env.fns.len();
            *env = Env::new();
            println!("cleared {n} definition(s)");
        }
        _ => eprintln!("unknown command !{name}  (try !help)"),
    }
}

fn run_repl() {
    use rustyline::{Editor, error::ReadlineError, history::DefaultHistory};
    let mut env = Env::new();

    // Auto-import init file: $MATHLANG_INIT, or ~/.mathlangrc if it exists
    let init = std::env::var("MATHLANG_INIT").ok().or_else(|| {
        std::env::var("HOME").ok().map(|h| format!("{h}/.mathlangrc"))
    });
    if let Some(path) = init {
        if std::path::Path::new(&path).exists() {
            import_file(&path, &path, &mut env, false);
        }
    }

    let mut rl = Editor::<MathHelper, DefaultHistory>::new().expect("failed to init editor");
    rl.set_helper(Some(MathHelper::new()));
    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() { continue; }
                let _ = rl.add_history_entry(&line);
                if matches!(line.as_str(), "q" | "quit" | "exit") { break; }
                if let Some(rest) = line.strip_prefix('!') {
                    bang_command(rest.trim_start(), &mut env);
                } else {
                    eval_line(&line, &mut env, true);
                }
                if let Some(h) = rl.helper() { h.update(&env); }
            }
            Err(ReadlineError::Interrupted) => {}
            Err(_) => break,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        run_repl();
        return;
    }
    let mut env = Env::new();
    let ok = eval_line(&args.join(" "), &mut env, false);
    if !ok { std::process::exit(1); }
}
