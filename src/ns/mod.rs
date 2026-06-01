// Standard (Rust-backed) namespaces, accessed with `.` syntax (e.g. ops.lap,
// bits.xor). Each namespace lives in its own module file and is registered
// unconditionally at startup by `register_all` (called from Env::new). New PDE
// functions (ops/solver) carry their own implementations and dispatch via a
// routing block in eval_builtin; relocated niche builtins (special/bits/stats/
// linalg/vec) keep their existing eval_builtin arms and are exposed here by name.

pub mod ops;
pub mod solver;
pub mod forms;
pub mod pic;
pub mod special;
pub mod bits;
pub mod stats;
pub mod linalg;
pub mod vec;

use crate::eval::Val;
use std::collections::HashMap;
use std::sync::Arc;

/// Bare builtin names that live ONLY inside a namespace (the new PDE functions).
/// eval_builtin routes these to the relevant module's dispatch().
pub fn is_ns_builtin(name: &str) -> bool {
    ops::NAMES.contains(&name) || solver::NAMES.contains(&name) || forms::NAMES.contains(&name)
        || pic::NAMES.contains(&name)
}

/// Route a namespaced new-function call to its module. Returns None if `name`
/// is not one of the new PDE builtins (caller falls through to eval_builtin).
pub fn dispatch(name: &str, vals: Vec<Val>, env: &crate::eval::Env) -> Option<Result<Val, String>> {
    if ops::NAMES.contains(&name)    { return Some(ops::dispatch(name, vals, env)); }
    if solver::NAMES.contains(&name) { return Some(solver::dispatch(name, vals, env)); }
    if forms::NAMES.contains(&name)  { return Some(forms::dispatch(name, vals, env)); }
    if pic::NAMES.contains(&name)    { return Some(pic::dispatch(name, vals, env)); }
    None
}

fn insert_ns(vars: &mut HashMap<String, Val>, name: &str, members: HashMap<String, Val>) {
    vars.insert(name.to_string(), Val::Namespace(Arc::new(members)));
}

pub fn register_all(vars: &mut HashMap<String, Val>) {
    insert_ns(vars, "ops", ops::members());
    insert_ns(vars, "solver",    solver::members());
    insert_ns(vars, "forms",     forms::members());
    insert_ns(vars, "pic",       pic::members());
    insert_ns(vars, "special",   special::members());
    insert_ns(vars, "bits",      bits::members());
    insert_ns(vars, "stats",     stats::members());
    insert_ns(vars, "linalg",    linalg::members());
    insert_ns(vars, "vec",       vec::members());
}
