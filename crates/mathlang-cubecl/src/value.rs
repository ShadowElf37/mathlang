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
    }
}
