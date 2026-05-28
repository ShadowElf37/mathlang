use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::lexer::Lexer;
use crate::ast::Def;
use crate::parser::Parser;
use crate::eval::{Val, Env, TData, eval, fmt_val, is_protected};

pub const BUILTIN_FNS: &[&str] = &[
    "id", "fact", "factorial", "delta",
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
    "graph", "animate2D", "animate2D_raw",
    "cell", "get", "set",
    // Tensor ops
    "tensor", "matrix", "zeros", "ones", "eye", "diag",
    "shape", "rows", "cols", "transpose", "trace", "norm",
    "row", "col", "matmul", "outer",
    "det", "inv", "solve",
    "eig", "eigvals", "eig_top", "eig_bot",
    "qr", "diagonalize",
    "hstack", "vstack", "tomat",
    "lerp", "clamp", "shift", "roll",
    "lingrid",
    "reshape", "permute", "cat", "squeeze", "unsqueeze",
    "dim", "tensordot",
    "fftn", "ifftn",
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
        for (k, v) in env.vars.iter() {
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
            let cmds = ["!clear", "!defs", "!help", "!include ", "!loadhdf5 ", "!loadtensor ", "!print ", "!savehdf5 ", "!savetensor ", "!version"];
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
            Def::Func(name, params, body) => {
                if is_protected(name) {
                    eprintln!("error: cannot redefine built-in '{name}'");
                    return false;
                }
                let mut captured = (*env.vars).clone();
                let fn_val = Val::make_fn(params.clone(), body.clone(), std::sync::Arc::new(captured.clone()));
                captured.insert(name.clone(), fn_val);
                env.define(name.clone(), Val::make_fn(params.clone(), body.clone(), std::sync::Arc::new(captured)));
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
            Val::Fn(params, ..) => format!("fn({}) = …", params.join(", ")),
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

pub fn import_file(path: &str, display: &str, env: &mut Env, verbose: bool) {
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
                    if buf.starts_with('!') {
                        bang_command(buf[1..].trim_start(), env);
                    } else {
                        eval_line(&buf, env, false);
                    }
                    n += 1;
                    buf.clear();
                    depth = 0;
                }
            }
            if !buf.is_empty() {
                if buf.starts_with('!') {
                    bang_command(buf[1..].trim_start(), env);
                } else {
                    eval_line(&buf, env, false);
                }
                n += 1;
            }
            if verbose { println!("included {n} definition(s) from {display}"); }
        }
        Err(e) => eprintln!("include {display}: {e}"),
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

fn bang_command(cmd: &str, env: &mut Env) {
    let (name, arg) = cmd.split_once(' ').map_or((cmd, ""), |(a, b)| (a, b.trim()));
    match name.trim() {
        "help" => print!(concat!(
            "Commands:  !help  !include <file>  !defs  !clear  !version\n",
            "           !print [text with {{expr}} interpolation]  — print formatted output\n",
            "           !savetensor <var> <file>  — save tensor to binary .mlt file\n",
            "           !loadtensor <var> <file>  — load tensor from .mlt file\n",
            "           !savehdf5 <var> <file> [/ds] [--append] [--overwrite] [--gzip 0-9]\n",
            "           !loadhdf5 <var> <file> [/ds] [--list]\n",
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
            "Aggregates: sum(f,a,b)  prod(f,a,b)  sum(f,n)  sum(T)  sum(T,axis)\n",
            "           integral(f,a,b[,n])  deriv(f,x[,dx])\n",
            "Grapher:   graph(f[,a,b])  saves graph_N.png to cwd\n\n",
            "Trig:      sin cos tan  asin acos atan atan2\n",
            "           sinh cosh tanh  sec csc cot\n",
            "Algebra:   sqrt cbrt abs sign heaviside  floor ceil round(x[,n]) trunc frac\n",
            "           ln log(x[,base]) log2 log10 exp expm1  pow hypot\n",
            "           min max  gcd lcm  fact  n!\n",
            "Angle:     deg rad\n",
            "Special:   sinc  sech csch  erf erfc  j0 j1 jinc\n",
            "           gaussian(x,mu,sigma)  gaussian_cdf(x,mu,sigma)\n",
            "Tuple ops: len sort zip dot  append concat flatten  argmin argmax\n",
            "           linspace(a,b,n)  range(a,b)\n",
            "Stats:     mean median mode  std var  (accept tuples or tensors)\n",
            "HOF:       map(f,t)  filter(f,t)  reduce(f,t)  compose(f,g)  partial(f,a)\n",
            "Control:   if(cond,a,b)\n",
            "Spectral:  fft(t)  ifft(t)  — DFT / inverse DFT on a tuple of numbers\n",
            "           fftn(T[,axes])  ifftn(T[,axes])  — n-D DFT on tensors\n",
            "           fftn/ifftn also accept (Re,Im[,axes]) for explicit complex input\n",
            "Random:    rand()  rand(a,b)\n",
            "Bitwise:   and or xor nand nor xnor not shl shr\n",
            "Complex:   i  re im abs arg conj  (all operators work on complex numbers)\n",
            "Constants: pi e phi inf i\n",
            "Tensors:   (1,2; 3,4)  or  [1,2; 3,4]  — 2D literal;  A @ B  — matmul\n",
            "           [a, b, c]  — tensor literal (same as (a,b,c) with auto-promotion)\n",
            "           zeros(n1,n2,…)  ones(n1,n2,…)  eye(n)  diag(t|T)\n",
            "           tensor(f, n1, n2, …)  matrix(f, r, c)\n",
            "           shape(T)  dim(T,axis)  rows cols  transpose([a,b])  trace norm\n",
            "           reshape(T, n1, n2, …)  permute(T, p0, p1, …)\n",
            "           cat(axis, T1, T2, …)  squeeze(T)  unsqueeze(T, dim)\n",
            "           outer(T1, T2)  tensordot(T1,T2,n)  tensordot(T1,T2,(a,b))\n",
            "           matmul(A,B)  row col\n",
            "           det inv solve(A,b)  hstack vstack tomat(t,r,c)\n",
            "           eig(A) eigvals(A) eig_top(A) eig_bot(A)  — eigenvalues/eigenvectors\n",
            "           qr(A) → (Q,R)   diagonalize(A) → (V,D,V⁻¹)\n",
            "           shift(T,n,axis)  — edge-replicating shift (Neumann BCs)\n",
            "           roll(T,n,axis)   — circular/periodic shift\n",
            "           lerp(a,b,t)      — linear interpolation: a*(1-t)+b*t (elementwise)\n",
            "           clamp(x,lo,hi)   — clamp value/tensor to [lo,hi]\n",
            "           lingrid(start,end,counts,f)  — supports any n-D via tuples\n",
            "           T[i,j,…]  T[i,a..b]  T[..,j]  T[i..,j]  T[..k,j]  — n-D slicing\n",
            "           T[..]  = all  T[n..]  = from n  T[..n]  = to n (tuples too)\n",
            "           sum(T,axis)  prod(T,axis)  mean std var on tensors\n",
            "           flatten(T)→1D  reduce(f,T)  map(f,T)  norm(T)\n",
            "Animate:   animate2D(f,t0,t1,n[,fps])  animate2D(T[,fps])  — spawn animator\n",
            "           animate2D_raw(…)  — write MXFR to stdout for manual piping\n",
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

    let mut rl = Editor::<MathHelper, DefaultHistory>::new().expect("failed to init editor");
    rl.set_helper(Some(MathHelper::new()));
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
                if matches!(line.as_str(), "quit" | "exit") { break; }
                if let Some(rest) = line.strip_prefix('!') {
                    bang_command(rest.trim_start(), &mut env);
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
