// bits — bitwise integer operations, relocated out of the flat builtin table.
use crate::eval::Val;
use std::collections::HashMap;

pub const NAMES: &[&str] = &[
    "and", "or", "xor", "nand", "nor", "xnor", "shl", "shr", "not",
];

pub fn members() -> HashMap<String, Val> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}
