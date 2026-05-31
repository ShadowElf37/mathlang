// stats — niche statistics, relocated out of the flat builtin table.
// (mean and std stay flat — common quick-REPL needs.)
use crate::eval::Val;
use std::collections::HashMap;

pub const NAMES: &[&str] = &["median", "mode", "var"];

pub fn members() -> HashMap<String, Val> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}
