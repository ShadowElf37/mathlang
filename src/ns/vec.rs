// vec — niche elementwise vector helpers, relocated out of the flat builtin table.
// (shift/roll stay flat — they are tensor primitives used by PDE code.)
use crate::eval::Val;

pub const NAMES: &[&str] = &["lerp", "clamp", "cross", "normalize", "vdot"];

pub fn members() -> Vec<(String, Val)> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}
