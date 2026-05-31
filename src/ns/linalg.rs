// linalg — niche linear-algebra ops, relocated out of the flat builtin table.
// (det, inv, solve, trace, eig, eigvals stay flat — common quick-REPL needs.)
use crate::eval::Val;
use std::collections::HashMap;

pub const NAMES: &[&str] = &[
    "qr", "diagonalize", "tensordot", "outer", "eig_top", "eig_bot",
];

pub fn members() -> HashMap<String, Val> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}
