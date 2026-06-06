//! REPL + one-liner driver. Mirrors the original's `eval_line` printing (top-level
//! tuples display with parens via `fmt_val`; long tuples elide). Line editing uses
//! plain stdin for now — rustyline (history/highlighting) can be added later.

use crate::ast::BlockStmt;
use crate::compute::{Backend, Prec};
use crate::interp::{define_into, eval, is_protected, Env};
use crate::value::{fmt_val, Val};
use crate::{lexer::Lexer, parser::Parser};
use std::io::{self, BufRead, Write};

/// Long-tuple elision threshold for REPL display (matches the original's intent).
const REPL_TUPLE_LIMIT: usize = 24;
const REPL_TUPLE_PREVIEW: usize = 8;

pub struct Repl {
    pub env: Env,
}

impl Repl {
    pub fn new() -> Self {
        Repl { env: Env::new() }
    }

    fn target_line(&self) -> String {
        format!(
            "backend: {} · prec: {}   (backends: {})",
            self.env.target.backend.name(),
            self.env.target.prec.name(),
            Backend::available().iter().map(|b| b.name()).collect::<Vec<_>>().join(", "),
        )
    }

    /// Run the interactive loop.
    pub fn run(&mut self) {
        println!("mc — mathlang-cubecl prototype (Phase 2: tensors on the compute path)");
        println!("{}", self.target_line());
        println!("type !help for commands, !q to quit\n");

        let stdin = io::stdin();
        loop {
            print!("> ");
            let _ = io::stdout().flush();
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => { eprintln!("input error: {e}"); break; }
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('!') {
                if self.bang(line) {
                    break;
                }
                continue;
            }
            self.eval_line(line, true);
        }
    }

    /// Evaluate one line of statements; print the last expression's value.
    /// `interactive` selects REPL framing (`result = …`) vs bare one-liner output.
    pub fn eval_line(&mut self, line: &str, interactive: bool) -> bool {
        let toks = Lexer::new(line).tokenize();
        let stmts = match Parser::new(toks).parse_repl() {
            Ok(v) => v,
            Err(e) => { eprintln!("error: {e}"); return false; }
        };
        let mut last: Option<Val> = None;
        for stmt in &stmts {
            match stmt {
                BlockStmt::Def(def) => {
                    if let Err(e) = define_into(&mut self.env, def) {
                        eprintln!("error: {e}");
                        return false;
                    }
                }
                BlockStmt::Expr(e) => match eval(e, &self.env) {
                    Ok(v) => last = Some(v),
                    Err(err) => { eprintln!("error: {err}"); return false; }
                },
            }
        }
        if let Some(v) = last {
            if interactive {
                let formatted = fmt_repl(&v);
                if formatted.contains('\n') {
                    println!("\x1b[2mresult =\x1b[0m\n{formatted}");
                } else {
                    println!("\x1b[2mresult = \x1b[0m{formatted}");
                }
                self.env.define("result".into(), v);
            } else {
                println!("{}", fmt_val(&v));
            }
        }
        true
    }

    /// Handle a `!command`. Returns true if the REPL should quit.
    fn bang(&mut self, line: &str) -> bool {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim();
        match cmd {
            "!q" | "!quit" | "!exit" => return true,
            "!help" => print_help(),
            "!version" => println!("mc {} (prototype)", env!("CARGO_PKG_VERSION")),
            "!defs" | "!vars" => self.show_defs(),
            "!clear" => {
                self.env = Env::new();
                println!("cleared user definitions");
            }
            "!backend" => self.cmd_backend(rest),
            "!prec" => self.cmd_prec(rest),
            "!type" => self.cmd_type(rest),
            "!print" => self.cmd_print(rest),
            "!spike" => crate::spike::run(),
            other => eprintln!("unknown command: {other} (try !help)"),
        }
        false
    }

    fn cmd_backend(&mut self, rest: &str) {
        if rest.is_empty() {
            println!("{}", self.target_line());
            return;
        }
        match Backend::parse(rest) {
            Some(b) if b.compiled_in() => {
                self.env.target.backend = b;
                // If the current precision isn't supported here, drop to the
                // backend's default (e.g. switching to wgpu downgrades f64 → f32).
                if !b.supports(self.env.target.prec) {
                    let p = b.default_prec();
                    println!("note: {} has no {}; switching prec to {}", b.name(), self.env.target.prec.name(), p.name());
                    self.env.target.prec = p;
                }
                println!("{}", self.target_line());
            }
            Some(b) => eprintln!("backend '{}' is not compiled in (rebuild with --features {})", b.name(), b.name()),
            None => eprintln!("unknown backend '{rest}' (cpu|wgpu|cuda|hip)"),
        }
    }

    fn cmd_prec(&mut self, rest: &str) {
        if rest.is_empty() {
            println!("{}", self.target_line());
            return;
        }
        match Prec::parse(rest) {
            Some(p) if self.env.target.backend.supports(p) => {
                self.env.target.prec = p;
                println!("{}", self.target_line());
                if p == Prec::Df64 && !crate::compute::df64_reliable(self.env.target.backend) {
                    println!("note: df64 arithmetic is unsupported on this backend (driver fast-math); storage/round-trip only — use cpu/cuda/hip to compute in df64");
                }
            }
            Some(p) => eprintln!(
                "{} has no native {} (try f32 or df64)",
                self.env.target.backend.name(),
                p.name()
            ),
            None => eprintln!("unknown precision '{rest}' (f32|df64|f64)"),
        }
    }

    fn cmd_type(&mut self, rest: &str) {
        if rest.is_empty() {
            eprintln!("usage: !type <expr>");
            return;
        }
        let toks = Lexer::new(rest).tokenize();
        match Parser::new(toks).parse_repl() {
            Ok(stmts) => {
                // type of the last expression
                let mut t = None;
                for s in &stmts {
                    if let BlockStmt::Expr(e) = s {
                        match eval(e, &self.env) {
                            Ok(v) => t = Some(type_of(&v)),
                            Err(err) => { eprintln!("error: {err}"); return; }
                        }
                    }
                }
                match t {
                    Some(ty) => println!("{ty}"),
                    None => eprintln!("!type: expected an expression"),
                }
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }

    /// `!print text {expr} more` — interpolate `{expr}`; `{{`/`}}` are literal braces.
    fn cmd_print(&mut self, rest: &str) {
        let mut out = String::new();
        let bytes: Vec<char> = rest.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            if c == '{' && i + 1 < bytes.len() && bytes[i + 1] == '{' {
                out.push('{');
                i += 2;
            } else if c == '}' && i + 1 < bytes.len() && bytes[i + 1] == '}' {
                out.push('}');
                i += 2;
            } else if c == '{' {
                let mut j = i + 1;
                let mut expr = String::new();
                while j < bytes.len() && bytes[j] != '}' {
                    expr.push(bytes[j]);
                    j += 1;
                }
                if j >= bytes.len() {
                    eprintln!("!print: unmatched {{");
                    return;
                }
                let toks = Lexer::new(&expr).tokenize();
                match Parser::new(toks).parse_repl() {
                    Ok(stmts) => {
                        let mut val = None;
                        for s in &stmts {
                            if let BlockStmt::Expr(e) = s {
                                match eval(e, &self.env) {
                                    Ok(v) => val = Some(v),
                                    Err(err) => { eprintln!("error: {err}"); return; }
                                }
                            }
                        }
                        if let Some(v) = val {
                            out.push_str(&fmt_val(&v));
                        }
                    }
                    Err(e) => { eprintln!("error: {e}"); return; }
                }
                i = j + 1;
            } else {
                out.push(c);
                i += 1;
            }
        }
        println!("{out}");
    }

    fn show_defs(&self) {
        let mut items: Vec<(String, String)> = vec![];
        for (k, v) in self.env.vars.iter() {
            if is_protected(k) || k == "result" {
                continue;
            }
            items.push((k.clone(), type_of(v)));
        }
        if items.is_empty() {
            println!("(no user definitions)");
            return;
        }
        items.sort();
        for (k, ty) in items {
            println!("  {k} : {ty}");
        }
    }
}

fn fmt_repl(v: &Val) -> String {
    match v {
        Val::Tuple(items) if items.len() > REPL_TUPLE_LIMIT => {
            let preview: Vec<String> = items[..REPL_TUPLE_PREVIEW].iter().map(fmt_val).collect();
            format!("({}, … [{} items])", preview.join(", "), items.len())
        }
        other => fmt_val(other),
    }
}

fn type_of(v: &Val) -> String {
    match v {
        Val::Num(_) => "real".into(),
        Val::Complex(..) => "complex".into(),
        Val::Tensor(t) => {
            let dims = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            format!("real tensor [{dims}] ({})", t.prec.name())
        }
        Val::ComplexTensor(t) => {
            let dims = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            format!("complex tensor [{dims}] ({})", t.prec.name())
        }
        Val::Tuple(_) => "tuple".into(),
        Val::Cell(_) => "cell".into(),
        Val::Namespace(_) => "namespace".into(),
        Val::Builtin(_) => "fn".into(),
        Val::Fn { params, sig, .. } => {
            let ps: Vec<String> = params
                .iter()
                .enumerate()
                .map(|(i, p)| match sig.params.get(i) {
                    Some(Some(h)) => format!("{}: {}", p, h.display()),
                    _ => p.clone(),
                })
                .collect();
            let ret = sig.ret.as_ref().map(|h| h.display()).unwrap_or_else(|| "?".into());
            format!("({}) -> {}", ps.join(", "), ret)
        }
    }
}

fn print_help() {
    println!(
        "\
mc — mathlang-cubecl prototype (Phase 2)

Scalars, complex, tuples, lambdas, closures, blocks, if, comparisons.
Tensors run on the compute path: [a,b,c], matrices (1,2; 3,4), a..b,
zeros/ones/eye/linspace/range; elementwise + - * / ^ and comparisons,
broadcasting a scalar against a tensor; unary math (sin/exp/sqrt/...);
shape/rows/cols/len; indexing/slicing T[i,j]/T[..,j]/T[a..b]; constructors
tensor/matrix/lingrid; assembly reshape/transpose/cat/vstack/hstack;
select/elementwise min,max; sum(T,axis). @ / matmul; sum/prod/mean/min/max/
norm/std (device reductions). Complex tensors: [1+2i,…], promotion,
re/im/abs/arg/conj, exp/ln/sqrt/sin/cos, complex sum/mean. Precision: f32, f64
(cpu/cuda/hip), or df64 double-single (+ - * / & compares on cpu/cuda/hip;
gated on wgpu; pow/transcendentals staged).
Other builtins: min/max/pow/hypot/gcd/lcm/ncr, lt/leq/gt/geq/eq/neq,
map/filter/reduce/compose/partial, iterate/scan (resident loops over
scalar/tensor/tuple state), cell/get/set.

Commands:
  !help              this text
  !backend [name]    show or set the backend (cpu|wgpu|cuda|hip)
  !prec [name]       show or set precision (f32|df64|f64)
  !type <expr>       show the type of an expression
  !defs              list user definitions
  !clear             clear user definitions
  !print <text>      print with {{expr}} interpolation
  !spike             run the f64-vs-f32 backend precision demo
  !version           version
  !q / !quit         quit

Linalg: det/inv/solve/trace/diag/eig/eigvals. Stencils: shift/roll, ops.lap/grad
(ops.periodic/ops.neumann).

Not yet present (later phases): spectral (fft/ops.poisson), fields/forms,
pic, calculus, file I/O, animation."
    );
}
