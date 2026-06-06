//! mathlang-cubecl prototype (`mc`).
//!
//! A clean port of mathlang onto CubeCL: one host interpreter for dynamic/scalar
//! work, one backend-generic CubeCL compute path for array work (cpu/wgpu/cuda/hip),
//! native f64 where the hardware allows. No bytecode VM, no `GPU {}` syntax.
//!
//! Build state: Phase 0 spike done (see [spike]); Phase 1 frontend ported. The
//! interpreter, value types, and REPL land next.

// Ported frontend (shared, unchanged semantics; `GpuBlock` removed — backend is
// runtime config, not syntax). `dead_code` is allowed until the interpreter and
// REPL consume the full surface in the next step.
#[allow(dead_code)]
mod ast;
#[allow(dead_code)]
mod lexer;
#[allow(dead_code)]
mod parser;

mod spike;

use lexer::Lexer;
use parser::Parser;

fn main() {
    println!("mc: mathlang-cubecl prototype — Phase 1 (frontend ported)\n");
    spike::run();
    println!();
    frontend_smoke();
}

/// Prove the ported lexer+parser compile and parse inside the new crate. Replaced
/// by the real REPL once the interpreter lands.
fn frontend_smoke() {
    println!("frontend smoke — lex + parse (no evaluation yet):");
    for src in ["2pi", "f(x) = x^2 + 1", "A = (1,2; 3,4)", "iterate(x -> 2*x, 1, 10)"] {
        let toks = Lexer::new(src).tokenize();
        match Parser::new(toks).parse_repl() {
            Ok(stmts) => println!("  ok  {src:<28} -> {} statement(s)", stmts.len()),
            Err(e) => println!("  ERR {src:<28} -> {e}"),
        }
    }
}
