use std::cell::RefCell;
use crate::lexer::Lexer;
use crate::ast::Def;
use crate::parser::Parser;
use crate::eval::{Val, Env, eval, fmt_val, is_protected};

pub const BUILTIN_FNS: &[&str] = &[
    "id", "fact", "factorial", "delta",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "expm1",
    "sec", "csc", "cot",
    "sqrt", "cbrt", "abs", "sign", "signum",
    "floor", "ceil", "round", "trunc", "frac",
    "step",
    "deg", "rad",
    "len", "length",
    "linspace", "range",
    "sort", "zip", "dot", "append", "concat", "flatten", "argmin", "argmax",
    "mean", "median", "mode", "std", "var",
    "compose", "partial",
    "gaussian", "gaussian_cdf", "eps",
    "filter", "reduce",
    "rand",
    "ln", "log", "log10", "log2", "exp",
    "re", "im", "arg", "conj",
    "min", "max", "pow", "hypot", "gcd", "lcm",
    "and", "or", "xor", "nand", "nor", "xnor", "not", "shl", "shr",
    "lt", "leq", "gt", "geq", "eq", "neq",
    "if",
    "fft", "ifft",
    "sum", "prod", "integral", "deriv", "map",
    "sinc", "sech", "csch", "erf", "erfc", "j0", "j1", "jinc",
    "graph",
    // Tensor ops
    "matrix", "zeros", "ones", "eye", "diag",
    "shape", "rows", "cols", "transpose", "trace", "norm",
    "row", "col", "matmul",
    "det", "inv", "solve",
    "hstack", "vstack", "tomat",
    "lingrid",
];

pub const BUILTIN_CONSTS: &[&str] = &["pi", "e", "phi", "inf", "i"];

// ── REPL helper ───────────────────────────────────────────────────────────────

struct MathHelper {
    user_fns:  RefCell<Vec<String>>,
    user_vars: RefCell<Vec<String>>,
    hinter:    rustyline::hint::HistoryHinter,
}

impl MathHelper {
    fn new() -> Self {
        Self { user_fns: RefCell::new(vec![]), user_vars: RefCell::new(vec![]), hinter: rustyline::hint::HistoryHinter {} }
    }
    fn update(&self, env: &Env) {
        let mut fns  = self.user_fns.borrow_mut();
        let mut vars = self.user_vars.borrow_mut();
        fns.clear(); vars.clear();
        for (k, v) in &env.vars {
            if BUILTIN_CONSTS.contains(&k.as_str()) || BUILTIN_FNS.contains(&k.as_str()) { continue; }
            if matches!(v, Val::Fn(..)) { fns.push(k.clone()); } else { vars.push(k.clone()); }
        }
    }
}

fn highlight_line(line: &str, user_fns: &[String], user_vars: &[String]) -> String {
    if line.starts_with('!') { return format!("\x1b[33m{line}\x1b[0m"); }
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 64);
    let mut i = 0;
    while i < line.len() {
        if b[i].is_ascii_whitespace() { out.push(b[i] as char); i += 1; continue; }
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
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            let name = &line[s..i];
            if BUILTIN_CONSTS.contains(&name) {
                out.push_str(&format!("\x1b[36m{name}\x1b[0m"));
            } else if BUILTIN_FNS.contains(&name) || user_fns.iter().any(|u| u == name) {
                out.push_str(&format!("\x1b[95m{name}\x1b[0m"));
            } else if user_vars.iter().any(|u| u == name) {
                out.push_str(name);
            } else {
                out.push_str(name);
            }
            continue;
        }
        if i + 1 < b.len() {
            match (b[i], b[i+1]) {
                (b'-', b'>') | (b'*', b'*') | (b'/', b'/') | (b'.', b'.') => {
                    out.push_str(&format!("\x1b[33m{}\x1b[0m", &line[i..i+2]));
                    i += 2; continue;
                }
                _ => {}
            }
        }
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
        std::borrow::Cow::Owned(highlight_line(line, &self.user_fns.borrow(), &self.user_vars.borrow()))
    }
    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool { true }
}

impl rustyline::completion::Completer for MathHelper {
    type Candidate = String;
    fn complete(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>)
        -> rustyline::Result<(usize, Vec<String>)>
    {
        if line.starts_with('!') {
            let cmds = ["!clear", "!defs", "!help", "!include ", "!version"];
            return Ok((0, cmds.iter().filter(|&&c| c.starts_with(line)).map(|s| s.to_string()).collect()));
        }
        let start = line[..pos].rfind(|c: char| !c.is_alphanumeric() && c != '_').map_or(0, |i| i+1);
        let word = &line[start..pos];
        if word.is_empty() { return Ok((pos, vec![])); }
        let user_fns  = self.user_fns.borrow();
        let user_vars = self.user_vars.borrow();
        let mut cs: Vec<String> = BUILTIN_FNS.iter().copied()
            .chain(BUILTIN_CONSTS.iter().copied())
            .chain(user_fns.iter().map(String::as_str))
            .chain(user_vars.iter().map(String::as_str))
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

// ── Evaluation and display ────────────────────────────────────────────────────

const REPL_TUPLE_LIMIT: usize = 12;
const REPL_TUPLE_PREVIEW: usize = 8;

fn fmt_repl(v: &Val) -> String {
    match v {
        Val::Tuple(items) if items.len() > REPL_TUPLE_LIMIT => {
            let preview: Vec<String> = items[..REPL_TUPLE_PREVIEW].iter().map(fmt_val).collect();
            format!("({}, ... [{} items])", preview.join(", "), items.len())
        }
        other => fmt_val(other),
    }
}

pub fn eval_line(line: &str, env: &mut Env, repl: bool) -> bool {
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
            Def::Var(name, expr) => {
                if is_protected(name) {
                    eprintln!("error: cannot redefine built-in '{name}'");
                    return false;
                }
                match eval(expr, env) {
                    Ok(v) => { env.vars.insert(name.clone(), v); }
                    Err(e) => { eprintln!("error: {e}"); return false; }
                }
            }
            Def::Func(name, params, body) => {
                if is_protected(name) {
                    eprintln!("error: cannot redefine built-in '{name}'");
                    return false;
                }
                let mut captured = env.vars.clone();
                let fn_val = Val::Fn(params.clone(), body.clone(), captured.clone());
                captured.insert(name.clone(), fn_val);
                env.vars.insert(name.clone(), Val::Fn(params.clone(), body.clone(), captured));
            }
        }
    }
    let vals: Vec<Val> = {
        let mut acc = vec![];
        for expr in &exprs {
            match eval(expr, env) {
                Ok(v) => acc.push(v),
                Err(e) => { eprintln!("error: {e}"); return false; }
            }
        }
        acc
    };
    if !vals.is_empty() {
        let v = if vals.len() == 1 { vals.into_iter().next().unwrap() } else { Val::Tuple(vals) };
        if repl {
            println!("\x1b[2mresult = \x1b[0m{}", fmt_repl(&v));
            env.vars.insert("result".into(), v);
        } else {
            println!("{}", fmt_val(&v));
        }
    }
    true
}

pub fn show_defs(env: &Env) {
    let mut items: Vec<(String, String)> = vec![];
    for (k, v) in &env.vars {
        if BUILTIN_CONSTS.contains(&k.as_str()) || BUILTIN_FNS.contains(&k.as_str()) || k == "result" { continue; }
        let display = match v {
            Val::Fn(params, _, _) => format!("fn({}) = …", params.join(", ")),
            _ => fmt_val(v),
        };
        items.push((k.clone(), display));
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

fn import_file(path: &str, display: &str, env: &mut Env, verbose: bool) {
    match std::fs::read_to_string(path) {
        Ok(src) => {
            let mut n = 0;
            let mut buf = String::new();
            let mut depth = 0i32;
            for line in src.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
                // Count braces only on the code portion (before any # comment)
                let code = if let Some(i) = trimmed.find('#') { &trimmed[..i] } else { trimmed };
                for ch in code.chars() {
                    if ch == '{' { depth += 1; }
                    else if ch == '}' { depth -= 1; }
                }
                if buf.is_empty() { buf.push_str(trimmed); } else { buf.push(' '); buf.push_str(trimmed); }
                if depth <= 0 {
                    eval_line(&buf, env, false);
                    n += 1;
                    buf.clear();
                    depth = 0;
                }
            }
            if !buf.is_empty() { eval_line(&buf, env, false); n += 1; }
            if verbose { println!("included {n} definition(s) from {display}"); }
        }
        Err(e) => eprintln!("include {display}: {e}"),
    }
}

fn bang_command(cmd: &str, env: &mut Env) {
    let (name, arg) = cmd.split_once(' ').map_or((cmd, ""), |(a, b)| (a, b.trim()));
    match name.trim() {
        "help" => print!(concat!(
            "Commands:  !help  !include <file>  !defs  !clear  !version\n",
            "Init file: ~/.mathlangrc (auto-imported on start; override with $MATHLANG_INIT)\n",
            "Exit:      quit / exit / Ctrl-D\n\n",
            "Syntax:    x = 3              variable\n",
            "           f(x) = x^2         named function\n",
            "           f = x -> x^2       lambda (first-class)\n",
            "           g = n,r -> n+r     multi-arg lambda\n",
            "           x=3; y=4 : x+y     define, then evaluate\n",
            "           {{x=3; y=4 : x^2+y^2}}  block with local scope\n\n",
            "Tuples:    (1,2,3)   t[0]   t[-1]   t[1..3]   t[0,2,4]\n",
            "Ranges:    (0..10)  — inclusive; range(a,b) — exclusive\n",
            "Operators: + - * / // % ^ **   -> (lambda)   n! (postfix factorial)\n",
            "           2pi  3sin(x)  2(x+1)  — implicit multiplication\n",
            "Compare:   < > <= >= == !=  (return 0 or 1)  & | (bitwise and/or)\n",
            "           lt leq gt geq eq neq  — comparison fns for use with map/filter\n",
            "Aggregates: sum(f,a,b)  prod(f,a,b)  sum(tuple)  prod(tuple)\n",
            "           integral(f,a,b[,n])  deriv(f,x[,dx])\n",
            "Grapher:   graph(f[,a,b])  saves graph_N.png to cwd\n\n",
            "Trig:      sin cos tan  asin acos atan atan2\n",
            "           sinh cosh tanh  sec csc cot\n",
            "Algebra:   sqrt cbrt abs sign step  floor ceil round(x[,n]) trunc frac\n",
            "           ln log(x[,base]) log2 log10 exp expm1  pow hypot\n",
            "           min max  gcd lcm  fact  n!\n",
            "Angle:     deg rad\n",
            "Special:   sinc  sech csch  erf erfc  j0 j1 jinc\n",
            "           gaussian(x,mu,sigma)  gaussian_cdf(x,mu,sigma)\n",
            "Tuple ops: len sort zip dot  append concat flatten  argmin argmax\n",
            "           linspace(a,b,n)  range(a,b)\n",
            "Stats:     mean median mode  std var\n",
            "HOF:       map(f,t)  filter(f,t)  reduce(f,t)  compose(f,g)  partial(f,a)\n",
            "Control:   if(cond,a,b)\n",
            "Spectral:  fft(t)  ifft(t)  — DFT / inverse DFT on a tuple of numbers\n",
            "Random:    rand()  rand(a,b)\n",
            "Bitwise:   and or xor nand nor xnor not shl shr\n",
            "Complex:   i  re im abs arg conj  (all operators work on complex numbers)\n",
            "Constants: pi e phi inf i\n",
            "Matrices:  (1,2; 3,4)  — literal;  A @ B  — matmul\n",
            "           zeros(r,c)  ones(r,c)  eye(n)  diag(t)  matrix(r,c,f)\n",
            "           shape rows cols transpose trace norm row col matmul\n",
            "           det inv solve(A,b)  hstack vstack tomat(t,r,c)\n",
            "           lingrid((x0,y0),(x1,y1),(nx,ny),f)\n",
            "           T[i,j]  T[i,a..b]  T[a..b,j]  T[a..b,c..d]\n",
        )),
        "include" | "import" => {
            if arg.is_empty() { eprintln!("usage: !include <file>"); return; }
            let path = expand_path(arg);
            if std::path::Path::new(&path).exists() {
                import_file(&path, arg, env, true);
            } else {
                let math_path = format!("{path}.math");
                if std::path::Path::new(&math_path).exists() {
                    import_file(&math_path, &math_path, env, true);
                } else {
                    import_file(&path, arg, env, true);
                }
            }
        }
        "version" => println!("mathlang v0.9"),
        "defs" | "vars" | "fns" => show_defs(env),
        "clear" => {
            let n = env.vars.iter().filter(|(k,_)| {
                !BUILTIN_CONSTS.contains(&k.as_str()) && !BUILTIN_FNS.contains(&k.as_str())
            }).count();
            *env = Env::new();
            println!("cleared {n} definition(s)");
        }
        _ => eprintln!("unknown command !{name}  (try !help)"),
    }
}

pub fn run_repl() {
    use rustyline::{Editor, error::ReadlineError, history::DefaultHistory};
    let mut env = Env::new();

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
    rl.bind_sequence(rustyline::KeyEvent::ctrl('D'), rustyline::EventHandler::Simple(rustyline::Cmd::EndOfFile));
    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() { continue; }
                let _ = rl.add_history_entry(&line);
                if matches!(line.as_str(), "quit" | "exit") { break; }
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
