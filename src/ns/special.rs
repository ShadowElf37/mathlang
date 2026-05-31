// special — special functions and niche kernels, relocated out of the flat
// builtin table. Members dispatch to their existing eval_builtin arms by bare name.
use crate::eval::Val;
use std::collections::HashMap;

pub const NAMES: &[&str] = &[
    "erf", "erfc", "j0", "j1", "jinc", "sinc",
    "sech", "csch", "gaussian", "gaussian_cdf", "delta",
];

pub fn members() -> HashMap<String, Val> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}
