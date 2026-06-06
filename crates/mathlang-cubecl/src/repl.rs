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
            "!savenpy" => self.cmd_save(rest, "savenpy", "npy"),
            "!loadnpy" => self.cmd_load(rest, "loadnpy", "npy"),
            "!savetensor" => self.cmd_save(rest, "savetensor", "mlt"),
            "!loadtensor" => self.cmd_load(rest, "loadtensor", "mlt"),
            "!savehdf5" => self.cmd_save_hdf5(rest),
            "!loadhdf5" => self.cmd_load_hdf5(rest),
            "!include" | "!run" => self.cmd_include(rest),
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

    // ── file I/O bang-commands (operate on a named variable) ───────────────────

    /// `!savenpy/!savetensor <var> <file>` — write a variable's tensor to disk.
    fn cmd_save(&mut self, rest: &str, cmd: &str, fmt: &str) {
        let mut it = rest.splitn(2, char::is_whitespace);
        let (var, file) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
        if var.is_empty() || file.is_empty() {
            eprintln!("usage: !{cmd} <var> <file>");
            return;
        }
        let val = match self.env.vars.get(var) {
            Some(v) => v.clone(),
            None => { eprintln!("{cmd}: {var} not defined"); return; }
        };
        let res = match fmt {
            "npy" => crate::io::save_npy_val(file, &val),
            _ => crate::io::save_mlt_val(file, &val),
        };
        match res {
            Ok(n) => println!("saved {var} ({n} elements, {}) → {file}", io_kind(&val)),
            Err(e) => eprintln!("{cmd}: {e}"),
        }
    }

    /// `!loadnpy/!loadtensor <var> <file>` — read a tensor into a variable.
    fn cmd_load(&mut self, rest: &str, cmd: &str, fmt: &str) {
        let mut it = rest.splitn(2, char::is_whitespace);
        let (var, file) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
        if var.is_empty() || file.is_empty() {
            eprintln!("usage: !{cmd} <var> <file>");
            return;
        }
        let res = match fmt {
            "npy" => crate::io::load_npy_val(file, self.env.target),
            _ => crate::io::load_mlt_val(file, self.env.target),
        };
        match res {
            Ok(val) => {
                let desc = io_kind(&val);
                self.env.define(var.to_string(), val);
                println!("loaded {var} ({desc}) ← {file}");
            }
            Err(e) => eprintln!("{cmd}: {e}"),
        }
    }

    /// `!savehdf5 <var> <file> [/dataset] [--append] [--overwrite] [--gzip N]`.
    fn cmd_save_hdf5(&mut self, rest: &str) {
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.len() < 2 {
            eprintln!("usage: !savehdf5 <var> <file> [/dataset] [--append] [--overwrite] [--gzip <0-9>]");
            return;
        }
        let (var, file) = (tokens[0], tokens[1]);
        let mut ds_path = format!("/{var}");
        let mut append = false;
        let mut gzip: Option<u32> = None;
        let mut i = 2;
        loop {
            match tokens.get(i).copied() {
                None => break,
                Some("--append") => { append = true; i += 1; }
                Some("--overwrite") => { append = false; i += 1; }
                Some("--gzip") => {
                    i += 1;
                    match tokens.get(i).and_then(|s| s.parse::<u32>().ok()).filter(|&n| n <= 9) {
                        Some(n) => { gzip = Some(n); i += 1; }
                        None => { eprintln!("savehdf5: --gzip requires a level 0–9"); return; }
                    }
                }
                Some(s) if !s.starts_with("--") => { ds_path = s.to_string(); i += 1; }
                Some(s) => { eprintln!("savehdf5: unknown option {s}"); return; }
            }
        }
        let val = match self.env.vars.get(var) {
            Some(v) => v.clone(),
            None => { eprintln!("savehdf5: {var} not defined"); return; }
        };
        match crate::io::save_hdf5_val(file, &ds_path, &val, append, gzip) {
            Ok(n) => println!("saved {var} ({n} elements) → {file}:{ds_path}"),
            Err(e) => eprintln!("savehdf5: {e}"),
        }
    }

    /// `!loadhdf5 <var> <file> [/dataset] [--list]`.
    fn cmd_load_hdf5(&mut self, rest: &str) {
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.len() < 2 {
            eprintln!("usage: !loadhdf5 <var> <file> [/dataset] [--list]");
            return;
        }
        let (var, file) = (tokens[0], tokens[1]);
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
        if list_only {
            if let Err(e) = crate::io::h5_list(&crate::io::expand_path(file)) {
                eprintln!("loadhdf5: {e}");
            }
            return;
        }
        match crate::io::load_hdf5_val(file, &ds_path, self.env.target) {
            Ok(val) => {
                let desc = io_kind(&val);
                self.env.define(var.to_string(), val);
                println!("loaded {var} ({desc}) ← {file}:{ds_path}");
            }
            Err(e) => eprintln!("loadhdf5: {e}"),
        }
    }

    /// `!include <file>` (alias `!run`) — evaluate a multi-line program file.
    /// Statements are newline-separated; a `{`/`(`/`[` keeps a statement open
    /// across lines (so a lambda body may span lines). Bare expressions print
    /// their value (one-liner framing); assignments are silent.
    fn cmd_include(&mut self, rest: &str) {
        let path = io_expand(rest.trim());
        if path.is_empty() {
            eprintln!("usage: !include <file>");
            return;
        }
        let n = self.run_file(&path);
        if n >= 0 {
            eprintln!("included {n} statement(s) from {path}");
        }
    }

    /// Evaluate every statement in a program file. Returns the statement count, or
    /// -1 if the file could not be read.
    pub fn run_file(&mut self, path: &str) -> i64 {
        let src = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => { eprintln!("include {path}: {e}"); return -1; }
        };
        let stmts = split_program(&src);
        for stmt in &stmts {
            self.eval_line(stmt, false);
        }
        stmts.len() as i64
    }
}

/// `~/` expansion shared with the I/O layer.
fn io_expand(p: &str) -> String {
    crate::io::expand_path(p)
}

/// Split a program file into complete statements. Newlines separate statements at
/// bracket depth 0; inside `{}`/`()`/`[]` a newline continues the statement (joined
/// with a space). `#` starts a comment to end-of-line, except inside a `"…"` string.
fn split_program(src: &str) -> Vec<String> {
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut in_comment = false;
    for ch in src.chars() {
        if in_comment {
            if ch == '\n' { in_comment = false; } else { continue; }
        }
        if in_str {
            cur.push(ch);
            if esc { esc = false; }
            else if ch == '\\' { esc = true; }
            else if ch == '"' { in_str = false; }
            continue;
        }
        match ch {
            '#' => in_comment = true,
            '"' => { in_str = true; cur.push(ch); }
            '{' | '(' | '[' => { depth += 1; cur.push(ch); }
            '}' | ')' | ']' => { depth -= 1; cur.push(ch); }
            '\n' => {
                if depth <= 0 {
                    let s = cur.trim().to_string();
                    if !s.is_empty() { stmts.push(s); }
                    cur.clear();
                } else {
                    cur.push(' ');
                }
            }
            _ => cur.push(ch),
        }
    }
    let s = cur.trim().to_string();
    if !s.is_empty() { stmts.push(s); }
    stmts
}

/// Short human description of a value for I/O status lines.
fn io_kind(v: &Val) -> String {
    match v {
        Val::Tensor(t) => format!("{:?} real f64", t.shape),
        Val::ComplexTensor(t) => format!("{:?} complex f64", t.shape),
        Val::Num(_) => "real scalar".into(),
        Val::Complex(..) => "complex scalar".into(),
        _ => "value".into(),
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
        Val::Str(_) => "string".into(),
        Val::Tensor(t) => {
            let dims = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            format!("real tensor [{dims}] ({})", t.prec.name())
        }
        Val::ComplexTensor(t) => {
            let dims = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            format!("complex tensor [{dims}] ({})", t.prec.name())
        }
        Val::Field(f) => {
            let dims = f.grid.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            format!("{}-form/field [{dims}]", f.degree)
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
  !savenpy/!loadnpy <var> <file>     NumPy .npy I/O
  !savetensor/!loadtensor <var> <file>   native .mlt I/O
  !savehdf5/!loadhdf5 <var> <file> [/ds] [opts]   HDF5 (build --features hdf5)
  !include <file>    run a program file (also: mc <file>)
  !version           version
  !q / !quit         quit

Linalg: det/inv/solve/trace/diag/eig/eigvals. Stencils: shift/roll, ops.lap/grad
(ops.periodic/ops.neumann). Calculus: integral/deriv (scalar; multidim integrals
and gradients too). Spectral: fft/ifft, ops.specgrad/poisson/invlap.
Fields & forms: field(...), forms.d/hodge/wedge/raise/lower/codiff/laplace/
contract; tensor(field) extracts.
PIC (particle/grid): pic.scatter(pos, w, template [, kernel]) deposits particles
onto a field; pic.gather(field, pos [, kernel]) interpolates; pic.gathergrad(field,
pos [, kernel]) returns the gradient of the shape function (variational force).
Kernels: pic.ngp (nearest), pic.cic (linear, default), pic.tsc (quadratic).
File I/O: save(value, \"path\" [, \"/dataset\"]) writes and returns value;
load(\"path\" [, \"/dataset\"]) reads a tensor. Format from extension: .npy
(NumPy), .mlt (native), .h5/.hdf5 (HDF5, --features hdf5). Real + complex f64.
Animation: animate2D(T) streams a 3-D movie [n,nx,ny]; animate2D(f, n) calls
f(t)->[nx,ny] for t=0..n-1; animate2D(f, t0, t1, n) over a range; optional fps.
animate2Dforever(f) streams until the window closes; animate2D_raw(...) writes
MXFR to stdout. Needs the wgpu_animator GUI (build animator/ or set
WGPU_ANIMATOR). Pairs with scan: animate2D(scan(step, u0, n)).

Not yet present (later phases): field-polymorphic ops.*(field); !graph plotting."
    );
}
