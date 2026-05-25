use std::cell::RefCell;
use crate::lexer::Lexer;
use crate::ast::Def;
use crate::parser::Parser;
use crate::eval::{Val, Env, eval, fmt_val};

pub const BUILTIN_FNS: &[&str] = &[
    "id", "fact", "factorial", "delta",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "sqrt", "cbrt", "abs", "sign", "signum",
    "floor", "ceil", "round",
    "ln", "log", "log10", "log2", "exp",
    "re", "im", "arg", "conj",
    "min", "max", "pow", "hypot", "gcd", "lcm",
    "and", "or", "xor", "nand", "nor", "xnor", "not", "shl", "shr",
    "sum", "prod", "integral", "deriv", "map",
];

pub const BUILTIN_CONSTS: &[&str] = &["pi", "e", "tau", "phi", "inf", "i"];

// ── REPL helper ───────────────────────────────────────────────────────────────

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
            } else if BUILTIN_FNS.contains(&name) || user.iter().any(|u| u == name) {
                out.push_str(&format!("\x1b[95m{name}\x1b[0m"));
            } else {
                out.push_str(name);
            }
            continue;
        }
        if i + 1 < b.len() {
            match (b[i], b[i+1]) {
                (b'-', b'>') | (b'*', b'*') | (b'/', b'/') => {
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

// ── Evaluation and display ────────────────────────────────────────────────────

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
                let fvars = crate::eval::free_vars(expr, env);
                if !fvars.is_empty() {
                    // Implicit function: f = x^2 stored as Val::Fn(["x"], x^2)
                    env.vars.insert(name.clone(), crate::eval::Val::Fn(fvars, expr.clone()));
                } else {
                    match eval(expr, env) {
                        Ok(v) => { env.vars.insert(name.clone(), v); }
                        Err(e) => { eprintln!("error: {e}"); return false; }
                    }
                }
            }
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

pub fn show_defs(env: &Env) {
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
            "           x=3; y=4 : x+y     define, then evaluate\n",
            "           {{disc=b^2-4*a*c : (-b+sqrt(disc))/(2*a), (-b-sqrt(disc))/(2*a)}}\n",
            "                              block with local defs, returns tuple\n\n",
            "Tuples:    (1,2,3)   (a,b)[0]   map(f, tuple)\n",
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
