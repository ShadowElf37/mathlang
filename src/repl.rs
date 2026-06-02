use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::lexer::Lexer;
use crate::ast::Def;
use crate::parser::Parser;
use crate::eval::{Val, Env, TData, eval, fmt_val, is_protected, FnSig, builtin_sig, infer_type, hint_of_val};

// Only flat (unqualified) builtins — names that are in scope without a namespace prefix.
// Namespace-only functions (special.*, bits.*, stats.*, linalg.*, vec.*, ops.*, solver.*,
// forms.*) are NOT listed here; they appear in REPL completion/highlighting via the
// namespace member lists instead.
pub const BUILTIN_FNS: &[&str] = &[
    "id", "fact", "factorial", "ncr", "quadratic",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "expm1",
    "sec", "csc", "cot",
    "sqrt", "cbrt", "abs", "sign", "signum",
    "floor", "ceil", "round", "trunc", "frac",
    "heaviside",
    "deg", "rad",
    "len", "length",
    "linspace", "range",
    "sort", "zip", "dot", "append", "concat", "flatten", "argmin", "argmax",
    "cumsum", "cumprod", "diff",
    "mean", "std",
    "compose", "partial",
    "eps",
    "filter", "reduce",
    "rand",
    "ln", "log", "log10", "log2", "exp",
    "re", "im", "arg", "conj",
    "min", "max", "pow", "hypot", "gcd", "lcm",
    "lt", "leq", "gt", "geq", "eq", "neq",
    "if",
    "fft", "ifft",
    "sum", "prod", "integral", "deriv", "map",
    "iterate", "scan",
    "cell", "get", "set",
    "field",
    // Tensor ops
    "tensor", "matrix", "zeros", "ones", "eye", "diag",
    "shape", "rows", "cols", "transpose", "trace", "norm",
    "row", "col", "matmul",
    "det", "inv", "solve",
    "eig", "eigvals",
    "hstack", "vstack", "tomat",
    "shift", "roll",
    "lingrid",
    "reshape", "permute", "cat", "squeeze", "unsqueeze",
    "dim",
];

pub const BUILTIN_CONSTS: &[&str] = &["pi", "e", "phi", "inf", "i"];

// Type-annotation keywords. Some (tensor, fn, cell) are also function names, so
// the highlighter only colours them as types when they follow a ':'.
pub const TYPE_KEYWORDS: &[&str] = &["any", "num", "real", "complex", "int", "nat", "tensor", "fn", "cell", "tuple"];

// ── REPL helper ───────────────────────────────────────────────────────────────

struct MathHelper {
    user_fns:  RefCell<Vec<String>>,
    user_vars: RefCell<Vec<String>>,
    // namespace name -> sorted member names. Captures both the standard namespaces
    // (registered in env.vars) and user `!namespace` ones, refreshed by `update`.
    namespaces: RefCell<std::collections::HashMap<String, Vec<String>>>,
    hinter:    rustyline::hint::HistoryHinter,
}

impl MathHelper {
    fn new() -> Self {
        Self {
            user_fns: RefCell::new(vec![]), user_vars: RefCell::new(vec![]),
            namespaces: RefCell::new(std::collections::HashMap::new()),
            hinter: rustyline::hint::HistoryHinter {},
        }
    }
    fn update(&self, env: &Env) {
        let mut fns  = self.user_fns.borrow_mut();
        let mut vars = self.user_vars.borrow_mut();
        let mut nss  = self.namespaces.borrow_mut();
        fns.clear(); vars.clear(); nss.clear();
        for (k, v) in env.vars.iter() {
            match v {
                Val::Namespace(map) => {
                    let mut members: Vec<String> = map.keys().cloned().collect();
                    members.sort();
                    nss.insert(k.clone(), members);
                }
                _ if BUILTIN_CONSTS.contains(&k.as_str()) || BUILTIN_FNS.contains(&k.as_str()) => {}
                Val::Fn(..) => fns.push(k.clone()),
                _ => vars.push(k.clone()),
            }
        }
    }
}

// Highlight the argument portion of a !print template.
// Literal text (and escaped {{ / }}) is yellow; {expr} has yellow braces with math-highlighted expr inside.
fn highlight_print_args(arg: &str, user_fns: &[String], user_vars: &[String],
                        namespaces: &std::collections::HashMap<String, Vec<String>>) -> String {
    let mut out = String::new();
    let mut chars = arg.chars().peekable();
    let mut in_yellow = false;

    macro_rules! open_yellow  { () => { if !in_yellow { out.push_str("\x1b[33m"); in_yellow = true;  } } }
    macro_rules! close_yellow { () => { if  in_yellow { out.push_str("\x1b[0m");  in_yellow = false; } } }

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                open_yellow!();
                out.push('{');
            } else {
                let mut expr_src = String::new();
                let mut depth = 1usize;
                let mut closed = false;
                loop {
                    match chars.next() {
                        None => { open_yellow!(); out.push('{'); out.push_str(&expr_src); break; }
                        Some('}') => {
                            depth -= 1;
                            if depth == 0 { closed = true; break; }
                            expr_src.push('}');
                        }
                        Some('{') => { depth += 1; expr_src.push('{'); }
                        Some(c)   => expr_src.push(c),
                    }
                }
                if closed {
                    close_yellow!();
                    out.push_str("\x1b[33m{\x1b[0m");
                    out.push_str(&highlight_line(&expr_src, user_fns, user_vars, namespaces));
                    out.push_str("\x1b[33m}\x1b[0m");
                }
            }
        } else if ch == '}' && chars.peek() == Some(&'}') {
            chars.next();
            open_yellow!();
            out.push('}');
        } else {
            open_yellow!();
            out.push(ch);
        }
    }
    if in_yellow { out.push_str("\x1b[0m"); }
    out
}

fn highlight_line(line: &str, user_fns: &[String], user_vars: &[String],
                  namespaces: &std::collections::HashMap<String, Vec<String>>) -> String {
    if line.starts_with('!') {
        let cmd_end = line.find(|c: char| c.is_ascii_whitespace()).unwrap_or(line.len());
        let cmd_name = &line[1..cmd_end];
        let rest = &line[cmd_end..];
        let cmd_colored = format!("\x1b[33m{}\x1b[0m", &line[..cmd_end]);
        return match cmd_name {
            "type" | "graph" | "animate2D" | "animate2D_raw" =>
                format!("{}{}", cmd_colored, highlight_line(rest, user_fns, user_vars, namespaces)),
            "print" =>
                format!("{}{}", cmd_colored, highlight_print_args(rest, user_fns, user_vars, namespaces)),
            _ => format!("\x1b[33m{line}\x1b[0m"),
        };
    }
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 64);
    let mut i = 0;
    let mut expect_type = false; // true right after a ':' — next identifier is a type
    // After `<namespace>.`, holds that namespace's member list so the next
    // identifier is coloured as a member only if it actually belongs to it.
    let mut member_of: Option<&Vec<String>> = None;
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
            expect_type = false;
            continue;
        }
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            let name = &line[s..i];
            // Member access (`ns.member`): colour as a function (magenta) only if it
            // is a real member of that namespace; an unknown name stays plain.
            if let Some(members) = member_of.take() {
                if members.iter().any(|m| m == name) {
                    out.push_str(&format!("\x1b[95m{name}\x1b[0m"));
                } else {
                    out.push_str(name);
                }
                expect_type = false;
                continue;
            }
            // Type annotation (green) — only after a ':'. Keep the flag alive for
            // the second word of "real tensor" / "complex tensor".
            if expect_type && TYPE_KEYWORDS.contains(&name) {
                out.push_str(&format!("\x1b[32m{name}\x1b[0m"));
                expect_type = name == "real" || name == "complex";
                continue;
            }
            expect_type = false;
            // Namespace name (orange, 256-colour). If a single `.` follows, remember
            // its member list so the next identifier can be validated against it.
            if let Some(members) = namespaces.get(name) {
                out.push_str(&format!("\x1b[38;5;208m{name}\x1b[0m"));
                let mut j = i;
                while j < b.len() && b[j].is_ascii_whitespace() { j += 1; }
                if j < b.len() && b[j] == b'.' && b.get(j + 1).map_or(true, |c| *c != b'.') {
                    member_of = Some(members);
                }
                continue;
            }
            if BUILTIN_CONSTS.contains(&name) {
                out.push_str(&format!("\x1b[36m{name}\x1b[0m"));
            } else if BUILTIN_FNS.contains(&name) || user_fns.iter().any(|u| u == name) {
                out.push_str(&format!("\x1b[95m{name}\x1b[0m"));
            } else if user_vars.iter().any(|u| u == name) {
                // Bold for ALLCAPS or Leading-Capital names (likely tensors/matrices)
                let is_tensor_name = name.chars().next().map_or(false, |c| c.is_uppercase());
                if is_tensor_name {
                    out.push_str(&format!("\x1b[1;95m{name}\x1b[0m"));
                } else {
                    out.push_str(&format!("\x1b[95m{name}\x1b[0m"));
                }
            } else {
                out.push_str(name);
            }
            continue;
        }
        if i + 1 < b.len() {
            match (b[i], b[i+1]) {
                (b'-', b'>') | (b'*', b'*') | (b'/', b'/') | (b'.', b'.') => {
                    out.push_str(&format!("\x1b[33m{}\x1b[0m", &line[i..i+2]));
                    i += 2; expect_type = false; continue;
                }
                _ => {}
            }
        }
        if b[i] == b':' {
            // type-annotation operator — what follows is a type
            out.push(':');
            expect_type = true;
            i += 1;
            continue;
        }
        if matches!(b[i], b'+' | b'-' | b'*' | b'/' | b'%' | b'^' | b'~') {
            out.push_str(&format!("\x1b[33m{}\x1b[0m", b[i] as char));
        } else {
            out.push(b[i] as char);
        }
        expect_type = false;
        i += 1;
    }
    out
}

impl rustyline::highlight::Highlighter for MathHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> std::borrow::Cow<'l, str> {
        std::borrow::Cow::Owned(highlight_line(line, &self.user_fns.borrow(), &self.user_vars.borrow(), &self.namespaces.borrow()))
    }
    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool { true }
}

impl rustyline::completion::Completer for MathHelper {
    type Candidate = String;
    fn complete(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>)
        -> rustyline::Result<(usize, Vec<String>)>
    {
        if line.starts_with('!') {
            let cmds = ["!animate2D ", "!animate2D_raw ", "!clear", "!defs", "!exit", "!graph ", "!help", "!include ", "!loadhdf5 ", "!loadnpy ", "!loadtensor ", "!print ", "!q", "!quit", "!savehdf5 ", "!savenpy ", "!savetensor ", "!type ", "!version"];
            return Ok((0, cmds.iter().filter(|&&c| c.starts_with(line)).map(|s| s.to_string()).collect()));
        }
        let start = line[..pos].rfind(|c: char| !c.is_alphanumeric() && c != '_').map_or(0, |i| i+1);
        let word = &line[start..pos];
        let bytes = line.as_bytes();
        // Namespace member completion: `<ns>.<word>` (works even when <word> is
        // empty, e.g. `ops.<TAB>` lists every member).
        if start > 0 && bytes[start - 1] == b'.' && (start < 2 || bytes[start - 2] != b'.') {
            let ns_end = start - 1;
            let ns_start = line[..ns_end].rfind(|c: char| !c.is_alphanumeric() && c != '_').map_or(0, |i| i + 1);
            let ns = &line[ns_start..ns_end];
            if let Some(members) = self.namespaces.borrow().get(ns) {
                let cs: Vec<String> = members.iter()
                    .filter(|m| m.starts_with(word) && m.as_str() != word)
                    .cloned().collect();
                return Ok((start, cs));
            }
        }
        if word.is_empty() { return Ok((pos, vec![])); }
        let user_fns  = self.user_fns.borrow();
        let user_vars = self.user_vars.borrow();
        let namespaces = self.namespaces.borrow();
        let mut cs: Vec<String> = BUILTIN_FNS.iter().copied()
            .chain(BUILTIN_CONSTS.iter().copied())
            .chain(user_fns.iter().map(String::as_str))
            .chain(user_vars.iter().map(String::as_str))
            .chain(namespaces.keys().map(String::as_str))
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

const REPL_TUPLE_LIMIT:   usize = 12;
const REPL_TUPLE_PREVIEW: usize = 8;
const REPL_VEC_LIMIT:     usize = 20;
const REPL_VEC_PREVIEW:   usize = 10;

fn fmt_repl(v: &Val) -> String {
    match v {
        Val::Tuple(items) if items.len() > REPL_TUPLE_LIMIT => {
            let preview: Vec<String> = items[..REPL_TUPLE_PREVIEW].iter().map(fmt_val).collect();
            format!("({}, … [{} items])", preview.join(", "), items.len())
        }
        Val::Tensor { data, shape } if shape.len() == 1 && data.len() > REPL_VEC_LIMIT => {
            use crate::eval::fmt_f;
            let preview: Vec<String> = data[..REPL_VEC_PREVIEW].iter().map(|x| fmt_f(*x)).collect();
            format!("[{}, … ({} elements)]", preview.join(", "), data.len())
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
                    Ok(v) => { env.define(name.clone(), v); }
                    Err(e) => { eprintln!("error: {e}"); return false; }
                }
            }
            Def::Func(name, params, ret_hint, body) => {
                if is_protected(name) {
                    eprintln!("error: cannot redefine built-in '{name}'");
                    return false;
                }
                let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                let sig = FnSig {
                    params: params.iter().map(|p| p.hint.clone()).collect(),
                    ret: ret_hint.clone(),
                };
                let mut captured = (*env.vars).clone();
                let fn_val = Val::make_fn_with_sig(names.clone(), sig.clone(), body.clone(), std::sync::Arc::new(captured.clone()));
                captured.insert(name.clone(), fn_val);
                env.define(name.clone(), Val::make_fn_with_sig(names, sig, body.clone(), std::sync::Arc::new(captured)));
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
            let formatted = fmt_repl(&v);
            if formatted.contains('\n') {
                println!("\x1b[2mresult =\x1b[0m\n{formatted}");
            } else {
                println!("\x1b[2mresult = \x1b[0m{formatted}");
            }
            env.define("result".into(), v);
        } else {
            println!("{}", fmt_val(&v));
        }
    }
    true
}

pub fn show_defs(env: &Env) {
    let mut items: Vec<(String, String)> = vec![];
    for (k, v) in env.vars.iter() {
        if BUILTIN_CONSTS.contains(&k.as_str()) || BUILTIN_FNS.contains(&k.as_str()) || k == "result" { continue; }
        let display = match v {
            Val::Fn(params, _, _, _, sig) => {
                let param_strs: Vec<String> = params.iter().enumerate().map(|(i, p)| {
                    if let Some(Some(h)) = sig.params.get(i) {
                        format!("{}: {}", p, h.display())
                    } else { p.clone() }
                }).collect();
                let ret_str = if let Some(h) = &sig.ret { format!(" -> {}", h.display()) } else { String::new() };
                format!("fn({}){}= …", param_strs.join(", "), if ret_str.is_empty() { " ".into() } else { format!("{} ", ret_str) })
            }
            _ => fmt_val(&v),
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

// A namespace being built from an `!namespace`-headed included file. Definitions
// evaluate into a file-local env (seeded from the global env, so builtins and
// other namespaces resolve); `private`-marked names stay local and only the
// remaining top-level names are exported as `Val::Namespace` at finalize.
struct NsBuild {
    name:    String,
    env:     Env,
    private: std::collections::HashSet<String>,
    public:  Vec<String>,
}

impl NsBuild {
    fn new(name: String, global: &Env) -> Self {
        NsBuild { name, env: global.clone(), private: Default::default(), public: Vec::new() }
    }
    fn eval_stmt(&mut self, code: &str, is_private: bool) {
        let before: std::collections::HashSet<String> = self.env.vars.keys().cloned().collect();
        eval_line(code, &mut self.env, false);
        for k in self.env.vars.keys() {
            if before.contains(k) { continue; }
            if is_private { self.private.insert(k.clone()); }
            else if k != "result" { self.public.push(k.clone()); }
        }
    }
    fn finalize(self, env: &mut Env) {
        let mut members: std::collections::HashMap<String, Val> = Default::default();
        for k in &self.public {
            if self.private.contains(k) { continue; }
            if let Some(v) = self.env.vars.get(k) { members.insert(k.clone(), v.clone()); }
        }
        env.define(self.name.clone(), Val::Namespace(Arc::new(members)));
    }
}

/// Strip a leading `private` visibility keyword from a statement.
fn strip_private(stmt: &str) -> (bool, &str) {
    if let Some(rest) = stmt.strip_prefix("private") {
        if rest.starts_with(char::is_whitespace) {
            return (true, rest.trim_start());
        }
    }
    (false, stmt)
}

pub fn import_file(path: &str, display: &str, env: &mut Env, verbose: bool) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("include {display}: {e}"); return; }
    };

    // Phase A: split into top-level statements (brace-aware, comments stripped).
    let mut stmts: Vec<String> = vec![];
    let mut buf = String::new();
    let mut depth = 0i32;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
        // Strip any trailing # comment: brace-count and buffer only the code
        // portion, so inline comments inside a multi-line { } block don't
        // swallow the lines joined after them.
        let code = (if let Some(i) = trimmed.find('#') { &trimmed[..i] } else { trimmed }).trim_end();
        if code.is_empty() { continue; }
        // Depth-track all bracket kinds so a statement may span lines whenever a
        // {…}, (…) or […] is still open (mathlang has no string literals, so no
        // quoting to escape). A line that closes everything ends the statement.
        for ch in code.chars() {
            match ch {
                '{' | '(' | '[' => depth += 1,
                '}' | ')' | ']' => depth -= 1,
                _ => {}
            }
        }
        if buf.is_empty() { buf.push_str(code); } else { buf.push(' '); buf.push_str(code); }
        if depth <= 0 { stmts.push(std::mem::take(&mut buf)); depth = 0; }
    }
    if !buf.is_empty() { stmts.push(buf); }

    // Phase B: execute, optionally collecting into a namespace.
    let mut n = 0;
    let mut ns: Option<NsBuild> = None;
    for stmt in &stmts {
        if let Some(rest) = stmt.strip_prefix('!') {
            let cmd = rest.trim_start();
            // `!namespace <name>` — route the rest of the file into namespace <name>.
            if let Some(arg) = cmd.strip_prefix("namespace") {
                let name = arg.trim();
                if name.is_empty() { eprintln!("!namespace requires a name"); continue; }
                if let Some(prev) = ns.take() { prev.finalize(env); }
                ns = Some(NsBuild::new(name.to_string(), env));
                continue;
            }
            match ns.as_mut() {
                Some(nb) => bang_command(cmd, &mut nb.env),
                None     => bang_command(cmd, env),
            }
        } else {
            let (is_private, code) = strip_private(stmt);
            match ns.as_mut() {
                Some(nb) => nb.eval_stmt(code, is_private),
                None     => { eval_line(code, env, false); }
            }
        }
        n += 1;
    }
    if let Some(nb) = ns.take() { nb.finalize(env); }
    if verbose { println!("included {n} definition(s) from {display}"); }
}

// ── NumPy .npy I/O ────────────────────────────────────────────────────────────

fn npy_shape_str(shape: &[usize]) -> String {
    match shape {
        []  => "()".into(),
        [n] => format!("({n},)"),
        _   => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

fn npy_find_str(header: &str, key: &str) -> Option<String> {
    for q in ['"', '\''] {
        let pat = format!("{q}{key}{q}");
        if let Some(ki) = header.find(&pat) {
            let rest = header[ki + pat.len()..].trim_start().trim_start_matches(':').trim_start();
            let qv = rest.chars().next()?;
            if qv == '\'' || qv == '"' {
                let inner = &rest[1..];
                return Some(inner[..inner.find(qv)?].to_string());
            }
        }
    }
    None
}

fn npy_find_shape(header: &str) -> Option<String> {
    for q in ['"', '\''] {
        let pat = format!("{q}shape{q}");
        if let Some(ki) = header.find(&pat) {
            let rest = header[ki + pat.len()..].trim_start().trim_start_matches(':').trim_start();
            if rest.starts_with('(') {
                let end = rest.find(')')?;
                return Some(rest[..=end].to_string());
            }
        }
    }
    None
}

fn npy_parse_shape(s: &str) -> Result<Vec<usize>, String> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    if inner.trim().is_empty() { return Ok(vec![]); }
    inner.split(',').map(|p| p.trim()).filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>().map_err(|e| format!("bad shape dim '{p}': {e}")))
        .collect()
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 != 0 { -1.0f64 } else { 1.0 };
    let exp  = (bits >> 10 & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f64;
    match exp {
        0x1f => if mant != 0.0 { f64::NAN } else { sign * f64::INFINITY },
        0    => sign * mant / 1024.0 * 2.0f64.powi(-14),
        _    => sign * (1.0 + mant / 1024.0) * 2.0f64.powi(exp - 15),
    }
}

fn save_npy(path: &str, val: &Val) -> Result<(), String> {
    (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let (dtype, shape) = match val {
            Val::Tensor { shape, .. }        => ("<f8",  shape.as_slice()),
            Val::ComplexTensor { shape, .. } => ("<c16", shape.as_slice()),
            _ => unreachable!(),
        };
        let header = format!(
            "{{'descr': '{dtype}', 'fortran_order': False, 'shape': {}, }}",
            npy_shape_str(shape)
        );
        // Total must be multiple of 64: magic(6)+version(2)+hlen_field(2) = 10 prefix bytes.
        let min_total = 10 + header.len() + 1; // +1 for trailing '\n'
        let padded    = (min_total + 63) / 64 * 64;
        let mut hdr   = header;
        for _ in 0..(padded - min_total) { hdr.push(' '); }
        hdr.push('\n');
        f.write_all(b"\x93NUMPY")?;
        f.write_all(&[1u8, 0u8])?;
        f.write_all(&(hdr.len() as u16).to_le_bytes())?;
        f.write_all(hdr.as_bytes())?;
        match val {
            Val::Tensor { data, .. } => {
                for &v in data.iter() { f.write_all(&v.to_le_bytes())?; }
            }
            Val::ComplexTensor { re, im, .. } => {
                for (&r, &i) in re.iter().zip(im.iter()) {
                    f.write_all(&r.to_le_bytes())?;
                    f.write_all(&i.to_le_bytes())?;
                }
            }
            _ => unreachable!()
        }
        Ok(())
    })().map_err(|e| e.to_string())
}

fn load_npy(path: &str) -> Result<Val, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a valid .npy file".into());
    }
    let major = bytes[6];
    let (hlen, doff) = if major <= 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        if bytes.len() < 12 { return Err("truncated header".into()); }
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12usize)
    };
    if bytes.len() < doff + hlen { return Err("truncated header".into()); }
    let header = std::str::from_utf8(&bytes[doff..doff + hlen])
        .map_err(|_| "invalid header encoding")?;

    let descr = npy_find_str(header, "descr").ok_or("missing 'descr' in npy header")?;
    let shape_s = npy_find_shape(header).ok_or("missing 'shape' in npy header")?;
    let shape = npy_parse_shape(&shape_s)?;
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err("Fortran-order arrays are not supported".into());
    }

    let nelem: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    let shape = if shape.is_empty() { vec![1] } else { shape };
    let buf   = &bytes[doff + hlen..];

    // Parse dtype: endian-char + kind-char + byte-width, e.g. "<f8", ">c16", "|u1"
    let db = descr.as_bytes();
    if db.len() < 3 { return Err(format!("unrecognised dtype: {descr}")); }
    let be   = db[0] == b'>';
    let kind = db[1] as char;
    let nb: usize = descr[2..].parse()
        .map_err(|_| format!("unrecognised dtype: {descr}"))?;

    let need = |n: usize| -> Result<(), String> {
        if buf.len() < n { Err(format!("truncated data: need {n} bytes, have {}", buf.len())) }
        else { Ok(()) }
    };

    match (kind, nb) {
        ('f', 8) => {
            need(nelem * 8)?;
            Ok(Val::Tensor { shape, data: TData::new((0..nelem).map(|i| {
                let b: [u8; 8] = buf[i*8..(i+1)*8].try_into().unwrap();
                if be { f64::from_be_bytes(b) } else { f64::from_le_bytes(b) }
            }).collect()) })
        }
        ('f', 4) => {
            need(nelem * 4)?;
            Ok(Val::Tensor { shape, data: TData::new((0..nelem).map(|i| {
                let b: [u8; 4] = buf[i*4..(i+1)*4].try_into().unwrap();
                if be { f32::from_be_bytes(b) as f64 } else { f32::from_le_bytes(b) as f64 }
            }).collect()) })
        }
        ('f', 2) => {
            need(nelem * 2)?;
            Ok(Val::Tensor { shape, data: TData::new((0..nelem).map(|i| {
                let b: [u8; 2] = buf[i*2..(i+1)*2].try_into().unwrap();
                f16_to_f64(if be { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
            }).collect()) })
        }
        ('c', 16) => {
            need(nelem * 16)?;
            let mut re = Vec::with_capacity(nelem);
            let mut im = Vec::with_capacity(nelem);
            for i in 0..nelem {
                let br: [u8; 8] = buf[i*16..i*16+8].try_into().unwrap();
                let bi: [u8; 8] = buf[i*16+8..i*16+16].try_into().unwrap();
                re.push(if be { f64::from_be_bytes(br) } else { f64::from_le_bytes(br) });
                im.push(if be { f64::from_be_bytes(bi) } else { f64::from_le_bytes(bi) });
            }
            Ok(Val::ComplexTensor { re: TData::new(re), im: TData::new(im), shape })
        }
        ('c', 8) => {
            need(nelem * 8)?;
            let mut re = Vec::with_capacity(nelem);
            let mut im = Vec::with_capacity(nelem);
            for i in 0..nelem {
                let br: [u8; 4] = buf[i*8..i*8+4].try_into().unwrap();
                let bi: [u8; 4] = buf[i*8+4..i*8+8].try_into().unwrap();
                re.push(if be { f32::from_be_bytes(br) as f64 } else { f32::from_le_bytes(br) as f64 });
                im.push(if be { f32::from_be_bytes(bi) as f64 } else { f32::from_le_bytes(bi) as f64 });
            }
            Ok(Val::ComplexTensor { re: TData::new(re), im: TData::new(im), shape })
        }
        ('i' | 'u', nb) if nb <= 8 => {
            need(nelem * nb)?;
            let signed = kind == 'i';
            let data: Vec<f64> = (0..nelem).map(|i| {
                let sl = &buf[i*nb..(i+1)*nb];
                match (signed, nb, be) {
                    (_,     1, _    ) => if signed { sl[0] as i8 as f64 } else { sl[0] as f64 },
                    (true,  2, false) => i16::from_le_bytes(sl.try_into().unwrap()) as f64,
                    (true,  2, true ) => i16::from_be_bytes(sl.try_into().unwrap()) as f64,
                    (false, 2, false) => u16::from_le_bytes(sl.try_into().unwrap()) as f64,
                    (false, 2, true ) => u16::from_be_bytes(sl.try_into().unwrap()) as f64,
                    (true,  4, false) => i32::from_le_bytes(sl.try_into().unwrap()) as f64,
                    (true,  4, true ) => i32::from_be_bytes(sl.try_into().unwrap()) as f64,
                    (false, 4, false) => u32::from_le_bytes(sl.try_into().unwrap()) as f64,
                    (false, 4, true ) => u32::from_be_bytes(sl.try_into().unwrap()) as f64,
                    (true,  8, false) => i64::from_le_bytes(sl.try_into().unwrap()) as f64,
                    (true,  8, true ) => i64::from_be_bytes(sl.try_into().unwrap()) as f64,
                    (false, 8, false) => u64::from_le_bytes(sl.try_into().unwrap()) as f64,
                    (false, 8, true ) => u64::from_be_bytes(sl.try_into().unwrap()) as f64,
                    _ => 0.0,
                }
            }).collect();
            Ok(Val::Tensor { data: TData::new(data), shape })
        }
        ('b', 1) => {
            need(nelem)?;
            Ok(Val::Tensor { shape, data: TData::new(
                (0..nelem).map(|i| if buf[i] != 0 { 1.0 } else { 0.0 }).collect()
            ) })
        }
        _ => Err(format!("unsupported dtype '{descr}' — supported: f2/f4/f8, c8/c16, i/u 1/2/4/8, bool")),
    }
}

// ── Tensor serialization ──────────────────────────────────────────────────────
// Format: [8] "MLTENSOR"  [1] type (0=real, 1=complex)  [8] ndim  [ndim*8] shape
//         real:    [nelem*8] f64 data
//         complex: [nelem*8] f64 re, then [nelem*8] f64 im  (all little-endian)

const TENSOR_MAGIC: &[u8; 8] = b"MLTENSOR";
const MLT_REAL:    u8 = 0x00;
const MLT_COMPLEX: u8 = 0x01;

fn write_f64s(f: &mut impl std::io::Write, xs: &[f64]) -> std::io::Result<()> {
    for &x in xs { f.write_all(&x.to_le_bytes())?; }
    Ok(())
}

fn save_tensor_val(path: &str, val: &Val) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    match val {
        Val::Tensor { data, shape } => {
            f.write_all(TENSOR_MAGIC)?;
            f.write_all(&[MLT_REAL])?;
            f.write_all(&(shape.len() as u64).to_le_bytes())?;
            for &d in shape { f.write_all(&(d as u64).to_le_bytes())?; }
            write_f64s(&mut f, data)?;
        }
        Val::ComplexTensor { re, im, shape } => {
            f.write_all(TENSOR_MAGIC)?;
            f.write_all(&[MLT_COMPLEX])?;
            f.write_all(&(shape.len() as u64).to_le_bytes())?;
            for &d in shape { f.write_all(&(d as u64).to_le_bytes())?; }
            write_f64s(&mut f, re)?;
            write_f64s(&mut f, im)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn load_tensor(path: &str) -> Result<Val, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 17 { return Err("file too short".into()); }
    if &bytes[..8] != TENSOR_MAGIC { return Err("not a mathlang tensor file".into()); }
    let kind = bytes[8];
    let ndim = u64::from_le_bytes(bytes[9..17].try_into().unwrap()) as usize;
    let hdr = 17 + ndim * 8;
    if bytes.len() < hdr { return Err("truncated header".into()); }
    let shape: Vec<usize> = (0..ndim)
        .map(|i| u64::from_le_bytes(bytes[17+i*8..17+(i+1)*8].try_into().unwrap()) as usize)
        .collect();
    let nelem: usize = shape.iter().product();
    let read_f64s = |off: usize| -> Vec<f64> {
        (0..nelem).map(|i| f64::from_le_bytes(bytes[off+i*8..off+(i+1)*8].try_into().unwrap())).collect()
    };
    match kind {
        MLT_REAL => {
            if bytes.len() < hdr + nelem * 8 {
                return Err(format!("truncated data: need {nelem} f64s"));
            }
            Ok(Val::Tensor { data: TData::new(read_f64s(hdr)), shape })
        }
        MLT_COMPLEX => {
            if bytes.len() < hdr + nelem * 16 {
                return Err(format!("truncated complex data: need {nelem} complex f64 pairs"));
            }
            Ok(Val::ComplexTensor {
                re: TData::new(read_f64s(hdr)),
                im: TData::new(read_f64s(hdr + nelem * 8)),
                shape,
            })
        }
        _ => Err(format!("unknown tensor type byte: 0x{kind:02x}")),
    }
}

// ── HDF5 I/O ──────────────────────────────────────────────────────────────────

#[cfg(feature = "hdf5")]
fn h5_split(path: &str) -> (String, String) {
    let path = path.trim_start_matches('/');
    match path.rfind('/') {
        Some(i) => (path[..i].to_string(), path[i+1..].to_string()),
        None    => (String::new(), path.to_string()),
    }
}

#[cfg(feature = "hdf5")]
fn h5_write_ds(grp: &::hdf5::Group, name: &str, data: &[f64], shape: &[usize], gzip: Option<u32>) -> Result<(), String> {
    let mut b = grp.new_dataset::<f64>().shape(shape);
    if let Some(lvl) = gzip { b = b.chunk(shape).deflate(lvl as u8); }
    b.create(name).map_err(|e| e.to_string())?.write_raw(data).map_err(|e| e.to_string())
}

#[cfg(feature = "hdf5")]
fn h5_save(file_path: &str, ds_path: &str, val: &Val, append: bool, overwrite: bool, gzip: Option<u32>) -> Result<usize, String> {
    let file = if append && std::path::Path::new(file_path).exists() {
        ::hdf5::File::open_rw(file_path).map_err(|e| e.to_string())?
    } else {
        ::hdf5::File::create(file_path).map_err(|e| e.to_string())?
    };
    let (grp_path, ds_name) = h5_split(ds_path);
    let grp_owned: Option<::hdf5::Group> = if grp_path.is_empty() { None } else {
        Some(match file.group(&grp_path) {
            Ok(g) => g,
            Err(_) => file.create_group(&grp_path).map_err(|e| e.to_string())?,
        })
    };
    let grp: &::hdf5::Group = match &grp_owned { Some(g) => g, None => &*file };
    if overwrite { let _ = grp.unlink(&ds_name); }
    match val {
        Val::Tensor { data, shape } => {
            h5_write_ds(grp, &ds_name, data, shape, gzip)?;
            Ok(data.len())
        }
        Val::ComplexTensor { re, im, shape } => {
            let cg = grp.create_group(&ds_name).map_err(|e| e.to_string())?;
            h5_write_ds(&cg, "re", re, shape, gzip)?;
            h5_write_ds(&cg, "im", im, shape, gzip)?;
            cg.new_attr::<u8>().create("mlt_complex").map_err(|e| e.to_string())?
              .write_raw(&[1u8]).map_err(|e| e.to_string())?;
            Ok(re.len())
        }
        _ => unreachable!(),
    }
}

#[cfg(feature = "hdf5")]
fn h5_load(file_path: &str, ds_path: &str) -> Result<Val, String> {
    let file = ::hdf5::File::open(file_path).map_err(|e| e.to_string())?;
    let (grp_path, ds_name) = h5_split(ds_path);
    let grp_owned: Option<::hdf5::Group> = if grp_path.is_empty() { None } else {
        Some(file.group(&grp_path).map_err(|e| e.to_string())?)
    };
    let grp: &::hdf5::Group = match &grp_owned { Some(g) => g, None => &*file };
    if let Ok(ds) = grp.dataset(&ds_name) {
        let shape = ds.shape();
        let data  = ds.read_raw::<f64>().map_err(|e| e.to_string())?;
        return Ok(Val::Tensor { data: TData::new(data), shape });
    }
    if let Ok(cg) = grp.group(&ds_name) {
        if cg.attr("mlt_complex").is_ok() {
            let ds_re = cg.dataset("re").map_err(|e| e.to_string())?;
            let shape = ds_re.shape();
            let re    = ds_re.read_raw::<f64>().map_err(|e| e.to_string())?;
            let im    = cg.dataset("im").map_err(|e| e.to_string())?
                          .read_raw::<f64>().map_err(|e| e.to_string())?;
            return Ok(Val::ComplexTensor { re: TData::new(re), im: TData::new(im), shape });
        }
    }
    Err(format!("'{ds_name}' not found in '{file_path}'"))
}

#[cfg(feature = "hdf5")]
fn h5_list(file_path: &str) -> Result<(), String> {
    fn recurse(grp: &::hdf5::Group, depth: usize) -> Result<(), String> {
        let ind = "  ".repeat(depth);
        for name in grp.member_names().map_err(|e| e.to_string())? {
            if let Ok(ds) = grp.dataset(&name) {
                let dims: Vec<String> = ds.shape().iter().map(|d| d.to_string()).collect();
                println!("{}{}  [{}  f64]", ind, name, dims.join("×"));
            } else if let Ok(cg) = grp.group(&name) {
                if cg.attr("mlt_complex").is_ok() {
                    let dims: Vec<String> = cg.dataset("re").ok()
                        .map_or_else(Vec::new, |d| d.shape())
                        .iter().map(|d| d.to_string()).collect();
                    println!("{}{}  [complex {}  f64]", ind, name, dims.join("×"));
                } else {
                    println!("{}{}/", ind, name);
                    recurse(&cg, depth + 1)?;
                }
            }
        }
        Ok(())
    }
    let file = ::hdf5::File::open(file_path).map_err(|e| e.to_string())?;
    recurse(&*file, 0)
}

fn ns_builtin_desc(ns: &str) -> Option<&'static str> {
    match ns {
        "ops"     => Some("spatial PDE operators (grad div curl lap poisson specgrad); field-aware"),
        "solver"  => Some("ODE time integrators: rk4(f,y0,t0,t1,n)  odeint(f,y0,ts)  verlet(dVdq,dTdp,q0,p0,dt,n)  tao(dHdq,dHdp,q0,p0,dt,n[,omega])  cfl(V,dx,dt)"),
        "forms"   => Some("exterior calculus over fields: d hodge wedge raise lower codiff laplace contract\n  field(data,lo,hi,bc[,metric])  field(f,lo,hi,counts,bc[,metric])  form(data,deg,…)  vector(data,…)"),
        "pic"     => Some("particle-in-cell coupling: scatter(positions,weights,template[,kernel]) deposit→grid (ρ,J)\n  gather(field,positions[,kernel]) interpolate→particles\n  gathergrad(field,positions[,kernel]) gather a scalar field with the kernel gradient (exact variational force)\n  kernels: pic.ngp pic.cic pic.tsc"),
        "special" => Some("special functions: sinc sech csch  erf erfc  j0 j1 jinc  gaussian gaussian_cdf  delta"),
        "bits"    => Some("true bitwise ops (truncate to i64): and or xor nand nor xnor shl shr not\n  Note: operators & | ~ are LOGICAL (return 0/1); bits.and/or are bitwise"),
        "stats"   => Some("statistics beyond mean/std: median(t)  mode(t)  var(t)"),
        "linalg"  => Some("linear algebra extras: qr(A)  diagonalize(A)  tensordot(A,B,ax)  outer(a,b)\n  eig_top(A,k)  eig_bot(A,k)"),
        "vec"     => Some("interpolation/clamping: lerp(a,b,t)  clamp(x,lo,hi)"),
        _ => None,
    }
}

// User-supplied help text for namespaces, set via `!helpdef <ns> <text>`.
thread_local! {
    static NS_HELP: std::cell::RefCell<std::collections::HashMap<String,String>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn bang_command(cmd: &str, env: &mut Env) {
    let (name, arg) = cmd.split_once(' ').map_or((cmd, ""), |(a, b)| (a, b.trim()));
    match name.trim() {
        "help" => {
            if arg.is_empty() {
                print!(concat!(
                    "Commands:  !help [topic|ns]  !include <file>  !defs  !clear  !version  !q\n",
                    "           !type <name>  — show type signature\n",
                    "           !graph f [,a,b]  — plot f over [a,b] (default -10..10)\n",
                    "           !animate2D f n [fps]  — animate f(t) returning [nx,ny] tensor\n",
                    "           !animate2D f t0 t1 n [fps] | !animate2D T [fps]\n",
                    "           !print text with {{expr}} interpolation\n",
                    "           !savetensor/loadtensor  !savenpy/loadnpy  !savehdf5/loadhdf5\n",
                    "           !helpdef <ns> <text>  — set help text for a user namespace\n",
                    "Init:      ~/.mathlangrc  ($MATHLANG_INIT to override)\n\n",
                    "Syntax:    x = 3           f(x) = x^2        f = x -> x^2\n",
                    "           g = n,r -> n+r  f(x:real) = ...   f(x:nat)->real = ...\n",
                    "           {{x=3; y=4; x^2+y^2}}  block (local scope)\n\n",
                    "Operators: + - * / // % ^ **   -> (lambda)   n!   ~ (logical not)\n",
                    "           < > <= >= == !=   & | (logical and/or)   2pi  3sin(x)  (implicit mul)\n",
                    "           lt leq gt geq eq neq  — comparison fns for map/filter\n\n",
                    "Tuples:    (1,2,3)  t[0]  t[-1]  t[1..3]  t[0,2,4]  (x,)  (singleton)\n",
                    "           (0..10) inclusive  range(a,b) exclusive  linspace(a,b,n)\n\n",
                    "Math:      sin cos tan asin acos atan atan2  sinh cosh tanh  sec csc cot\n",
                    "           sqrt cbrt abs sign heaviside  floor ceil round trunc frac\n",
                    "           ln log(x[,b]) log2 log10 exp expm1  pow hypot  min max  gcd lcm\n",
                    "           deg rad  fact  n!  ncr(n,r)  quadratic(a,b,c)  if(c,a,b)\n\n",
                    "Tensors:   zeros ones eye diag tensor(f,…) matrix(f,r,c)  [a,b;c,d]  A@B\n",
                    "           shape dim rows cols  transpose trace norm  det inv solve\n",
                    "           eig eigvals  hstack vstack cat reshape permute squeeze unsqueeze\n",
                    "           shift(T,n,axis)  roll(T,n,axis)  lingrid  T[i,j]  T[a..b,..]\n\n",
                    "HOF:       map filter reduce  sum(f,a,b)  prod(f,a,b)\n",
                    "           integral(f,a,b) deriv(f,x)  — box bounds / tensor point ⇒ multivar\n",
                    "           iterate(f,x0,n)  scan(f,x0,n)  cumsum cumprod diff\n\n",
                    "Other:     mean std  fft ifft  rand  re im arg conj  cell get set\n",
                    "           field(data,lo,hi,bc) | field(f,lo,hi,counts,bc)  — scalar field (0-form)\n",
                    "           Constants: pi e phi inf i\n\n",
                    "Namespaces — use !help <ns> for members and usage:\n",
                    "  ops     PDE operators: grad div curl lap poisson specgrad\n",
                    "  solver  ODE steppers:  rk4 odeint verlet(symplectic) tao(nonsep symplectic) cfl\n",
                    "  forms   Exterior calc: d hodge wedge raise lower codiff laplace contract\n",
                    "  pic     Particle-in-cell: scatter(deposit) gather(interpolate)  [ngp cic tsc]\n",
                    "  special Special fns:   sinc sech csch erf erfc j0 j1 jinc gaussian delta\n",
                    "  bits    Bitwise:       and or xor nand nor xnor shl shr not  (also & | ~)\n",
                    "  stats   Statistics:    median mode var\n",
                    "  linalg  Linear alg:    qr diagonalize tensordot outer eig_top eig_bot\n",
                    "  vec     Interpolation: lerp clamp\n",
                ));
            } else {
                let topic = arg.trim_start_matches('!');

                // `!help ns.member` — look up the member inside the named namespace.
                if let Some(dot) = topic.find('.') {
                    let ns_name = &topic[..dot];
                    let member  = &topic[dot+1..];
                    if let Some(Val::Namespace(map)) = env.vars.get(ns_name) {
                        if map.contains_key(member) {
                            let sig = builtin_sig(member)
                                .map(|s| format!("\x1b[33m{s}\x1b[0m"))
                                .unwrap_or_else(|| format!("{ns_name}.{member}(…)"));
                            println!("\x1b[1m{ns_name}.{member}\x1b[0m  {sig}");
                        } else {
                            eprintln!("'{member}' is not a member of namespace '{ns_name}'");
                            let mut names: Vec<&String> = map.keys().collect();
                            names.sort();
                            eprintln!("  members: {}", names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("  "));
                        }
                        return;
                    }
                }

                // Namespace? Show description + member list.
                if let Some(Val::Namespace(map)) = env.vars.get(topic) {
                    let user_desc = NS_HELP.with(|h| h.borrow().get(topic).cloned());
                    let desc = user_desc.as_deref().or_else(|| ns_builtin_desc(topic));
                    if let Some(d) = desc {
                        println!("\x1b[1m{topic}\x1b[0m — {d}");
                    } else {
                        println!("\x1b[1m{topic}\x1b[0m namespace");
                    }
                    let mut names: Vec<&String> = map.keys().collect();
                    names.sort();
                    let list = names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("  ");
                    println!("Members (access as {topic}.<member>):\n  {list}");
                    return;
                }

                // Try bang command help file, then per-function help file.
                let bang_text = std::fs::read_to_string(format!("help/bang/{topic}.txt")).ok();
                if let Some(text) = bang_text {
                    print!("{text}");
                } else if let Ok(text) = std::fs::read_to_string(format!("help/{topic}.txt")) {
                    let sig_line = builtin_sig(topic)
                        .map(|s| format!("\x1b[1mUsage:\x1b[0m \x1b[33m{s}\x1b[0m\n\n"))
                        .unwrap_or_default();
                    if !sig_line.is_empty() {
                        if let Some(ex) = text.find("Examples:") {
                            let inject = text[..ex].rfind("\n\n").map(|p| p + 2).unwrap_or(ex);
                            print!("{}{}{}", &text[..inject], sig_line, &text[inject..]);
                        } else {
                            print!("{text}\n{sig_line}");
                        }
                    } else {
                        print!("{text}");
                    }
                } else if let Some(sig) = builtin_sig(topic) {
                    // No help file but we have a signature — find which namespace owns it.
                    let owner = env.vars.iter().find_map(|(ns, v)| {
                        if let Val::Namespace(map) = v {
                            if map.contains_key(topic) { Some(ns.clone()) } else { None }
                        } else { None }
                    });
                    let prefix = owner.as_deref()
                        .map(|ns| format!("({ns}.{topic})  "))
                        .unwrap_or_default();
                    println!("\x1b[33m{prefix}{sig}\x1b[0m");
                } else {
                    // Last resort: check if it's a member of any namespace.
                    let owner = env.vars.iter().find_map(|(ns, v)| {
                        if let Val::Namespace(map) = v {
                            if map.contains_key(topic) { Some(ns.clone()) } else { None }
                        } else { None }
                    });
                    if let Some(ns) = owner {
                        eprintln!("'{topic}' is a member of '{ns}' — try !help {ns}.{topic}");
                    } else {
                        eprintln!("no help for '{topic}'  (try !help with no argument)");
                    }
                }
            }
        }
        "helpdef" => {
            // !helpdef <ns> <description text>
            let (ns, desc) = arg.split_once(' ').map_or((arg, ""), |(a, b)| (a.trim(), b.trim()));
            if ns.is_empty() {
                eprintln!("usage: !helpdef <ns> <description>");
                return;
            }
            if desc.is_empty() {
                // Clear the description.
                NS_HELP.with(|h| h.borrow_mut().remove(ns));
            } else {
                NS_HELP.with(|h| h.borrow_mut().insert(ns.to_string(), desc.to_string()));
            }
        }
        "namespace" => {
            eprintln!("!namespace is only valid at the top of an included .math file");
        }
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
        "graph" => {
            if arg.is_empty() { eprintln!("usage: !graph f [, a, b]"); return; }
            let toks = Lexer::new(arg).tokenize();
            match Parser::new(toks).parse_repl() {
                Ok((_, exprs)) if !exprs.is_empty() => {
                    if let Err(e) = crate::graph::eval_graph(&exprs, env) {
                        eprintln!("graph: {e}");
                    }
                }
                Ok(_) => eprintln!("usage: !graph f [, a, b]"),
                Err(e) => eprintln!("!graph: {e}"),
            }
        }
        "animate2D" => {
            if arg.is_empty() {
                eprintln!("usage: !animate2D T [fps] | !animate2D f n [fps] | !animate2D f t0 t1 n [fps]");
                return;
            }
            let toks = Lexer::new(arg).tokenize();
            match Parser::new(toks).parse_repl() {
                Ok((_, exprs)) if !exprs.is_empty() => {
                    if let Err(e) = crate::animate::eval_animate2d(&exprs, env) {
                        eprintln!("animate2D: {e}");
                    }
                }
                Ok(_) => eprintln!("usage: !animate2D T [fps] | !animate2D f n [fps] | ..."),
                Err(e) => eprintln!("!animate2D: {e}"),
            }
        }
        "animate2D_raw" => {
            if arg.is_empty() {
                eprintln!("usage: !animate2D_raw T | !animate2D_raw f n | !animate2D_raw f t_vals | !animate2D_raw f t0 t1 n");
                return;
            }
            let toks = Lexer::new(arg).tokenize();
            match Parser::new(toks).parse_repl() {
                Ok((_, exprs)) if !exprs.is_empty() => {
                    if let Err(e) = crate::animate::eval_animate2d_raw(&exprs, env) {
                        eprintln!("animate2D_raw: {e}");
                    }
                }
                Ok(_) => eprintln!("usage: !animate2D_raw T | !animate2D_raw f n | ..."),
                Err(e) => eprintln!("!animate2D_raw: {e}"),
            }
        }
        "version" => println!("mathlang v{}", env!("CARGO_PKG_VERSION")),
        "print" => {
            let mut out = String::new();
            let mut chars = arg.chars().peekable();
            while let Some(ch) = chars.next() {
                if ch == '{' {
                    if chars.peek() == Some(&'{') {
                        chars.next();
                        out.push('{');
                    } else {
                        let mut expr_src = String::new();
                        let mut depth = 1usize;
                        loop {
                            match chars.next() {
                                None => { eprintln!("print: unclosed {{"); return; }
                                Some('}') => {
                                    depth -= 1;
                                    if depth == 0 { break; }
                                    expr_src.push('}');
                                }
                                Some('{') => { depth += 1; expr_src.push('{'); }
                                Some(c)   => expr_src.push(c),
                            }
                        }
                        let toks = Lexer::new(expr_src.trim()).tokenize();
                        match Parser::new(toks).parse_repl() {
                            Err(e) => out.push_str(&format!("<parse error: {e}>")),
                            Ok((_, exprs)) => match exprs.first() {
                                None => {}
                                Some(node) => match eval(node, env) {
                                    Ok(val) => out.push_str(&fmt_val(&val)),
                                    Err(e)  => out.push_str(&format!("<error: {e}>")),
                                },
                            },
                        }
                    }
                } else if ch == '}' && chars.peek() == Some(&'}') {
                    chars.next();
                    out.push('}');
                } else {
                    out.push(ch);
                }
            }
            println!("{out}");
        }
        "savetensor" => {
            let mut it = arg.splitn(2, ' ');
            let (var, fp) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
            if var.is_empty() || fp.is_empty() {
                eprintln!("usage: !savetensor <var> <file>"); return;
            }
            match env.vars.get(var) {
                Some(v @ Val::Tensor { data, .. }) => {
                    let (n, v) = (data.len(), v.clone());
                    match save_tensor_val(&expand_path(fp), &v) {
                        Ok(()) => println!("saved {var} ({n} elements, real) to {fp}"),
                        Err(e) => eprintln!("savetensor: {e}"),
                    }
                }
                Some(v @ Val::ComplexTensor { re, .. }) => {
                    let (n, v) = (re.len(), v.clone());
                    match save_tensor_val(&expand_path(fp), &v) {
                        Ok(()) => println!("saved {var} ({n} elements, complex) to {fp}"),
                        Err(e) => eprintln!("savetensor: {e}"),
                    }
                }
                Some(_) => eprintln!("savetensor: {var} is not a tensor"),
                None    => eprintln!("savetensor: {var} not defined"),
            }
        }
        "loadtensor" => {
            let mut it = arg.splitn(2, ' ');
            let (var, fp) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
            if var.is_empty() || fp.is_empty() {
                eprintln!("usage: !loadtensor <var> <file>"); return;
            }
            match load_tensor(&expand_path(fp)) {
                Ok(val) => {
                    let desc = match &val {
                        Val::Tensor { data, .. }        => format!("{} elements, real", data.len()),
                        Val::ComplexTensor { re, .. }   => format!("{} elements, complex", re.len()),
                        _ => unreachable!(),
                    };
                    env.define(var.to_string(), val);
                    println!("loaded {var} ({desc}) from {fp}");
                }
                Err(e) => eprintln!("loadtensor: {e}"),
            }
        }
        "savenpy" => {
            let mut it = arg.splitn(2, ' ');
            let (var, fp) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
            if var.is_empty() || fp.is_empty() {
                eprintln!("usage: !savenpy <var> <file.npy>"); return;
            }
            match env.vars.get(var) {
                Some(v @ Val::Tensor { data, .. }) => {
                    let (n, v) = (data.len(), v.clone());
                    match save_npy(&expand_path(fp), &v) {
                        Ok(()) => println!("saved {var} ({n} elements, real f64) → {fp}"),
                        Err(e) => eprintln!("savenpy: {e}"),
                    }
                }
                Some(v @ Val::ComplexTensor { re, .. }) => {
                    let (n, v) = (re.len(), v.clone());
                    match save_npy(&expand_path(fp), &v) {
                        Ok(()) => println!("saved {var} ({n} elements, complex f64) → {fp}"),
                        Err(e) => eprintln!("savenpy: {e}"),
                    }
                }
                Some(_) => eprintln!("savenpy: {var} is not a tensor"),
                None    => eprintln!("savenpy: {var} not defined"),
            }
        }
        "loadnpy" => {
            let mut it = arg.splitn(2, ' ');
            let (var, fp) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
            if var.is_empty() || fp.is_empty() {
                eprintln!("usage: !loadnpy <var> <file.npy>"); return;
            }
            match load_npy(&expand_path(fp)) {
                Ok(val) => {
                    let desc = match &val {
                        Val::Tensor { data, shape }        => format!("{shape:?} real f64, {} elem", data.len()),
                        Val::ComplexTensor { re, shape, .. } => format!("{shape:?} complex f64, {} elem", re.len()),
                        _ => unreachable!(),
                    };
                    env.define(var.to_string(), val);
                    println!("loaded {var} ({desc}) ← {fp}");
                }
                Err(e) => eprintln!("loadnpy: {e}"),
            }
        }
        "savehdf5" => {
            #[cfg(not(feature = "hdf5"))]
            eprintln!("savehdf5: build with --features hdf5 to enable HDF5 support");
            #[cfg(feature = "hdf5")]
            {
                let tokens: Vec<&str> = arg.split_whitespace().collect();
                if tokens.len() < 2 {
                    eprintln!("usage: !savehdf5 <var> <file> [/dataset] [--append] [--overwrite] [--gzip <0-9>]");
                    return;
                }
                let (var, fp) = (tokens[0], tokens[1]);
                let mut ds_path = format!("/{var}");
                let mut append   = false;
                let mut overwrite = false;
                let mut gzip: Option<u32> = None;
                let mut i = 2;
                loop {
                    match tokens.get(i).copied() {
                        None => break,
                        Some("--append")    => { append    = true; i += 1; }
                        Some("--overwrite") => { overwrite = true; i += 1; }
                        Some("--gzip") => {
                            i += 1;
                            match tokens.get(i).and_then(|s| s.parse::<u32>().ok()).filter(|&n| n <= 9) {
                                Some(n) => { gzip = Some(n); i += 1; }
                                None    => { eprintln!("savehdf5: --gzip requires a level 0–9"); return; }
                            }
                        }
                        Some(s) if !s.starts_with("--") => { ds_path = s.to_string(); i += 1; }
                        Some(s) => { eprintln!("savehdf5: unknown option {s}"); return; }
                    }
                }
                let val = match env.vars.get(var) {
                    Some(v @ Val::Tensor { .. }) | Some(v @ Val::ComplexTensor { .. }) => v.clone(),
                    Some(_) => { eprintln!("savehdf5: {var} is not a tensor"); return; }
                    None    => { eprintln!("savehdf5: {var} not defined"); return; }
                };
                match h5_save(&expand_path(fp), &ds_path, &val, append, overwrite, gzip) {
                    Ok(n)  => println!("saved {var} ({n} elements) → {fp}:{ds_path}"),
                    Err(e) => eprintln!("savehdf5: {e}"),
                }
            }
        }
        "loadhdf5" => {
            #[cfg(not(feature = "hdf5"))]
            eprintln!("loadhdf5: build with --features hdf5 to enable HDF5 support");
            #[cfg(feature = "hdf5")]
            {
                let tokens: Vec<&str> = arg.split_whitespace().collect();
                if tokens.len() < 2 {
                    eprintln!("usage: !loadhdf5 <var> <file> [/dataset] [--list]");
                    return;
                }
                let (var, fp) = (tokens[0], tokens[1]);
                let mut ds_path = format!("/{var}");
                let mut list_only = false;
                let mut i = 2;
                loop {
                    match tokens.get(i).copied() {
                        None => break,
                        Some("--list") => { list_only = true; i += 1; }
                        Some(s) if !s.starts_with("--") => { ds_path = s.to_string(); i += 1; }
                        Some(s) => { eprintln!("loadhdf5: unknown option {s}"); return; }
                    }
                }
                let fp_exp = expand_path(fp);
                if list_only {
                    if let Err(e) = h5_list(&fp_exp) { eprintln!("loadhdf5: {e}"); }
                    return;
                }
                match h5_load(&fp_exp, &ds_path) {
                    Ok(val) => {
                        let desc = match &val {
                            Val::Tensor { data, .. }      => format!("{} elements, real", data.len()),
                            Val::ComplexTensor { re, .. } => format!("{} elements, complex", re.len()),
                            _ => unreachable!(),
                        };
                        env.define(var.to_string(), val);
                        println!("loaded {var} ({desc}) ← {fp}:{ds_path}");
                    }
                    Err(e) => eprintln!("loadhdf5: {e}"),
                }
            }
        }
        "type" => {
            let name = arg.trim();
            if name.is_empty() {
                eprintln!("usage: !type <expr>");
                return;
            }
            // A name bound in the environment: functions show their fused
            // signature `(t1, t2) -> ret`; values show their type.
            if let Some(val) = env.vars.get(name) {
                match val {
                    Val::Fn(params, body, _, _, sig) => {
                        let param_strs: Vec<String> = params.iter().enumerate().map(|(i, _)| {
                            sig.params.get(i).and_then(|h| h.clone())
                                .map_or_else(|| "any".to_string(), |h| h.display().to_string())
                        }).collect();
                        let mut pmap = std::collections::HashMap::new();
                        for (i, pn) in params.iter().enumerate() {
                            if let Some(Some(h)) = sig.params.get(i) { pmap.insert(pn.clone(), h.clone()); }
                        }
                        let ret = match &sig.ret {
                            Some(h) => h.display().to_string(),
                            None    => infer_type(body, &pmap, env).display().to_string(),
                        };
                        println!("{name}({}) -> {ret}", param_strs.join(", "));
                        return;
                    }
                    Val::Builtin(bname) => {
                        println!("{}", builtin_sig(bname.as_str()).unwrap_or("fn"));
                        return;
                    }
                    other => {
                        println!("{}", hint_of_val(other).display());
                        return;
                    }
                }
            }
            // A builtin function name.
            if let Some(sig) = builtin_sig(name) { println!("{sig}"); return; }
            // Otherwise, parse and statically infer the type of the expression.
            let toks = Lexer::new(name).tokenize();
            match Parser::new(toks).parse_repl() {
                Ok((_, exprs)) => match exprs.first() {
                    Some(e) => {
                        let empty = std::collections::HashMap::new();
                        println!("{}", infer_type(e, &empty, env).display());
                    }
                    None => eprintln!("!type: nothing to type"),
                },
                Err(e) => eprintln!("!type: {e}"),
            }
        }
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

    // Ctrl+D with content in the buffer: rustyline's default EndOfFile only fires
    // on an empty line. We use a ConditionalEventHandler so Ctrl+D always exits.
    let ctrl_d_pressed = Arc::new(AtomicBool::new(false));
    struct CtrlDHandler(Arc<AtomicBool>);
    impl rustyline::ConditionalEventHandler for CtrlDHandler {
        fn handle(
            &self,
            _evt: &rustyline::Event,
            _n: rustyline::RepeatCount,
            _positive: bool,
            ctx: &rustyline::EventContext,
        ) -> Option<rustyline::Cmd> {
            if ctx.line().is_empty() {
                Some(rustyline::Cmd::EndOfFile)   // returns ReadlineError::Eof → break
            } else {
                self.0.store(true, Ordering::SeqCst);
                Some(rustyline::Cmd::Interrupt)   // returns ReadlineError::Interrupted → check flag
            }
        }
    }

    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::Circular)   // Tab cycles candidates inline
        .build();
    let mut rl = Editor::<MathHelper, DefaultHistory>::with_config(config).expect("failed to init editor");
    rl.set_helper(Some(MathHelper::new()));
    // Warm up the helper so the standard namespaces are known before the first line.
    if let Some(h) = rl.helper() { h.update(&env); }
    rl.bind_sequence(
        rustyline::KeyEvent::ctrl('D'),
        rustyline::EventHandler::Conditional(Box::new(CtrlDHandler(ctrl_d_pressed.clone()))),
    );

    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() { continue; }
                let _ = rl.add_history_entry(&line);
                if let Some(rest) = line.strip_prefix('!') {
                    let cmd = rest.trim_start();
                    if matches!(cmd, "q" | "quit" | "exit") { break; }
                    bang_command(cmd, &mut env);
                } else {
                    eval_line(&line, &mut env, true);
                }
                if let Some(h) = rl.helper() { h.update(&env); }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C (or Ctrl+D with content — check flag)
                if ctrl_d_pressed.swap(false, Ordering::SeqCst) { break; }
                // else Ctrl+C: do nothing (keeps current line-cancel behaviour)
            }
            Err(_) => break,  // Eof (Ctrl+D on empty line) or other error
        }
    }
}
