//! Runtime values for the host interpreter.
//!
//! Phase 1b covers the dynamic/scalar core: numbers, complex, tuples (trees),
//! closures, builtins, cells, and namespaces. Tensor variants land in Phase 2 as a
//! backend-agnostic `compute::TensorVal` handle — deliberately absent here so the
//! interpreter stays purely host-side and instant for scalar work (the low-latency
//! invariant).
//!
//! Semantics mirror the original `src/eval.rs` exactly (the parity harness depends
//! on it): `make_complex` collapses negligible imaginary parts, formatting matches
//! `fmt_f`/`fmt_val`, etc.

use crate::ast::{Expr, TypeHint};
use crate::compute::{self, CTensor, TensorVal};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// A function's optional type hints (stored, not yet enforced in Phase 1b).
#[derive(Clone, Debug, Default)]
pub struct FnSig {
    pub params: Vec<Option<TypeHint>>,
    pub ret: Option<TypeHint>,
}

/// A captured lexical environment, shared O(1) on closure clone.
pub type Captured = Arc<HashMap<String, Val>>;

#[derive(Clone)]
pub enum Val {
    Num(f64),
    Complex(f64, f64),
    /// A user closure: parameter names, body, captured env, and (unenforced) sig.
    Fn {
        params: Vec<String>,
        body: Arc<Expr>,
        captured: Captured,
        sig: Arc<FnSig>,
    },
    Builtin(String),
    /// A device-resident real tensor (the CubeCL compute path).
    Tensor(TensorVal),
    /// A device-resident complex tensor (interleaved re/im).
    ComplexTensor(CTensor),
    /// A tuple tree — heterogeneous leaves; ops broadcast over it.
    Tuple(Vec<Val>),
    /// Shared mutable container (identity semantics on clone).
    Cell(Arc<RefCell<Val>>),
    /// `ns.member` map. Forward-compat: namespaces are registered in a later phase.
    #[allow(dead_code)]
    Namespace(Arc<HashMap<String, Val>>),
}

impl Val {
    pub fn make_fn(params: Vec<String>, body: Expr, captured: Captured) -> Self {
        Val::Fn { params, body: Arc::new(body), captured, sig: Arc::new(FnSig::default()) }
    }

    pub fn make_fn_with_sig(params: Vec<String>, sig: FnSig, body: Expr, captured: Captured) -> Self {
        Val::Fn { params, body: Arc::new(body), captured, sig: Arc::new(sig) }
    }

    /// Extract a real number or explain why the value isn't one.
    pub fn num(self, ctx: &str) -> Result<f64, String> {
        match self {
            Val::Num(n) => Ok(n),
            Val::Complex(..) => Err(format!("{ctx}: expected a real number, got complex")),
            Val::Fn { .. } => Err(format!("{ctx}: expected a number, got a function")),
            Val::Builtin(n) => Err(format!("{ctx}: expected a number, got builtin '{n}'")),
            Val::Tensor(..) => Err(format!("{ctx}: expected a number, got a tensor")),
            Val::ComplexTensor(..) => Err(format!("{ctx}: expected a number, got a complex tensor")),
            Val::Tuple(..) => Err(format!("{ctx}: expected a number, got a tuple")),
            Val::Cell(..) => Err(format!("{ctx}: expected a number, got a cell (use get())")),
            Val::Namespace(..) => Err(format!("{ctx}: expected a number, got a namespace")),
        }
    }
}

/// Collapse `a+bi` to `Num(a)` when `b` is negligible relative to the magnitude —
/// identical rule to the original evaluator so results round-trip.
pub fn make_complex(a: f64, b: f64) -> Val {
    let scale = (a.abs() + b.abs()).max(1.0) * 1e-10;
    let a = if a.abs() < scale { 0.0 } else { a };
    let b = if b.abs() < scale { 0.0 } else { b };
    if b == 0.0 { Val::Num(a) } else { Val::Complex(a, b) }
}

/// View any scalar value as a complex pair.
pub fn to_complex(v: Val) -> Result<(f64, f64), String> {
    match v {
        Val::Num(n) => Ok((n, 0.0)),
        Val::Complex(a, b) => Ok((a, b)),
        other => Err(format!("expected a number, got {}", fmt_val(&other))),
    }
}

#[inline]
pub fn int(x: f64) -> i64 {
    x as i64
}

// ── Formatting (matches src/eval.rs) ────────────────────────────────────────────

pub fn fmt_f(n: f64) -> String {
    if n.is_nan() {
        return "NaN".into();
    }
    if n.is_infinite() {
        return if n > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if n.fract() == 0.0 && n.abs() < 1e15 {
        return format!("{}", n as i64);
    }
    format!("{n}")
}

pub fn fmt_val(v: &Val) -> String {
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
        Val::Fn { params, sig, .. } => {
            let param_strs: Vec<String> = params
                .iter()
                .enumerate()
                .map(|(i, name)| match sig.params.get(i) {
                    Some(Some(h)) => format!("{}: {}", name, h.display()),
                    _ => name.clone(),
                })
                .collect();
            let ret_str = sig.ret.as_ref().map(|h| format!(" -> {}", h.display())).unwrap_or_default();
            format!("<fn({}){}= …>", param_strs.join(", "), if ret_str.is_empty() { " ".into() } else { format!("{ret_str} ") })
        }
        Val::Builtin(name) => format!("<builtin {name}>"),
        Val::Cell(c) => format!("cell({})", fmt_val(&c.borrow())),
        Val::Namespace(map) => {
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            format!("namespace{{{}}}", names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
        }
        Val::Tuple(items) => {
            format!("({})", items.iter().map(fmt_val).collect::<Vec<_>>().join(", "))
        }
        Val::Tensor(t) => fmt_tensor(t),
        Val::ComplexTensor(t) => fmt_complex_tensor(t),
    }
}

/// Format one complex element (real if im == 0), matching the original.
fn fmt_complex_elem(r: f64, i: f64) -> String {
    if i == 0.0 {
        return fmt_f(r);
    }
    let babs = i.abs();
    let im = if babs == 1.0 { String::new() } else { fmt_f(babs) };
    if r == 0.0 {
        if i < 0.0 { format!("-{im}i") } else { format!("{im}i") }
    } else if i < 0.0 {
        format!("{} - {im}i", fmt_f(r))
    } else {
        format!("{} + {im}i", fmt_f(r))
    }
}

fn fmt_complex_tensor(t: &CTensor) -> String {
    let (re, im) = match compute::download_complex(t) {
        Ok(d) => d,
        Err(e) => return format!("<complex tensor read error: {e}>"),
    };
    // Collapse negligible re/im at display time (matches the original's make_complex
    // rule) without forcing a per-op download that would break loop residency.
    let cell = |k: usize| {
        let (mut r, mut i) = (re[k], im[k]);
        let scale = (r.abs() + i.abs()).max(1.0) * 1e-10;
        if i.abs() < scale { i = 0.0; }
        if r.abs() < scale { r = 0.0; }
        fmt_complex_elem(r, i)
    };
    match t.shape.as_slice() {
        [] | [_] => format!("[{}]", (0..t.len).map(cell).collect::<Vec<_>>().join(", ")),
        [r, c] => {
            let cells: Vec<Vec<String>> = (0..*r).map(|i| (0..*c).map(|j| cell(i * c + j)).collect()).collect();
            fmt_mat_cells(cells)
        }
        shape => {
            let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            format!("complex tensor[{dims}] [{}]", (0..t.len).map(cell).collect::<Vec<_>>().join(", "))
        }
    }
}

/// Format a device tensor by pulling it to the host. 1-D → `[…]`, 2-D → a boxed
/// matrix, higher rank → a shape header plus the flat data.
fn fmt_tensor(t: &TensorVal) -> String {
    let data = match compute::download(t) {
        Ok(d) => d,
        Err(e) => return format!("<tensor read error: {e}>"),
    };
    match t.shape.as_slice() {
        [] | [_] => format!("[{}]", data.iter().map(|x| fmt_f(*x)).collect::<Vec<_>>().join(", ")),
        [r, c] => fmt_mat(&data, *r, *c),
        shape => {
            let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            let body = data.iter().map(|x| fmt_f(*x)).collect::<Vec<_>>().join(", ");
            format!("tensor[{dims}] [{body}]")
        }
    }
}

/// Boxed 2-D matrix display (⎡⎢⎣ … ⎤⎥⎦), right-aligned columns.
fn fmt_mat(data: &[f64], r: usize, c: usize) -> String {
    let cells: Vec<Vec<String>> =
        (0..r).map(|i| (0..c).map(|j| fmt_f(data[i * c + j])).collect()).collect();
    fmt_mat_cells(cells)
}

/// Box a grid of pre-formatted cell strings (shared by real and complex matrices).
fn fmt_mat_cells(cells: Vec<Vec<String>>) -> String {
    let r = cells.len();
    let c = cells.first().map(|row| row.len()).unwrap_or(0);
    let widths: Vec<usize> =
        (0..c).map(|j| cells.iter().map(|row| row[j].chars().count()).max().unwrap_or(0)).collect();
    cells
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let content = row
                .into_iter()
                .zip(&widths)
                .map(|(s, &w)| format!("{}{}", " ".repeat(w - s.chars().count()), s))
                .collect::<Vec<_>>()
                .join("  ");
            let (l, rr) = if r == 1 || i == 0 {
                ('\u{23A1}', '\u{23A4}') // ⎡ ⎤
            } else if i == r - 1 {
                ('\u{23A3}', '\u{23A6}') // ⎣ ⎦
            } else {
                ('\u{23A2}', '\u{23A5}') // ⎢ ⎥
            };
            format!("{l} {content} {rr}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}
