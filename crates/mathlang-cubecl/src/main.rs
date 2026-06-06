//! mathlang-cubecl prototype (`mc`).
//!
//! A clean port of mathlang onto CubeCL: one host interpreter for dynamic/scalar
//! work, one backend-generic CubeCL compute path for array work (cpu/wgpu/cuda/hip),
//! native f64 where the hardware allows. No bytecode VM, no `GPU {}` syntax.
//!
//! Build state: Phase 0 spike + Phase 1b host interpreter (scalars/complex/tuples/
//! lambdas) and REPL. Tensors on the compute path arrive in Phase 2.
//!
//! Usage:
//!   mc                 interactive REPL
//!   mc 'expr'          evaluate a one-liner and print the result
//!   mc --spike         run the f64-vs-f32 backend precision demo

// Ported frontend (unchanged semantics; `GpuBlock` removed). Kept as a library
// surface, so some items are unused here.
#[allow(dead_code)]
mod ast;
#[allow(dead_code)]
mod lexer;
#[allow(dead_code)]
mod parser;

mod builtins;
mod compute;
mod interp;
mod repl;
mod spike;
mod value;

use repl::Repl;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.len() == 1 && args[0] == "--spike" {
        spike::run();
        return;
    }

    if args.is_empty() {
        Repl::new().run();
    } else {
        // One-liner mode: the joined args are a statement sequence; print the last.
        let line = args.join(" ");
        let mut repl = Repl::new();
        repl.eval_line(&line, false);
    }
}
