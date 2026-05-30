// ── Bytecode instruction set ──────────────────────────────────────────────────
//
// The flat instruction set for the stack-based bytecode VM. A `Val::Fn` lazily
// compiles its body to `Vec<Instruction>` on first call (see `eval.rs`).
//
// This type deliberately depends only on `crate::ast` (Op, Expr) and never on
// `Val` or anything else in `eval.rs`, so the GPU backend (`src/gpu/`) can import
// the instruction set without pulling in the whole evaluator (TODO 1f). The VM
// *executor* (`run_vm`) and the *compiler* (`Compiler`) still live in `eval.rs`,
// since they operate on `Val`; only the data definition moved here.

use std::sync::Arc;
use crate::ast::{Expr, Op};

#[derive(Debug, Clone)]
pub enum Instruction {
    PushNum(f64),
    PushComplex(f64, f64),
    LoadParam(usize),           // bind from args[i]
    LoadCaptured(String),       // live env lookup (Cells, Fns, Tensors)
    BinOp(Op),                  // pop 2, push 1
    Neg,                        // pop 1, push 1
    CallBuiltin(String, usize), // pop argc args, call builtin, push result
    CallVal(usize),             // pop callee then argc args, call, push result
    MakeTuple(usize),           // pop n, promote to Tensor if all-numeric
    MakeArray(usize),           // pop n, always produce Tensor ([] syntax)
    JumpIfFalse(usize),         // pop cond, jump to absolute pc if 0.0
    Jump(usize),                // unconditional absolute jump
    StoreLocal(usize),          // pop → locals[slot]
    LoadLocal(usize),           // push locals[slot]
    Pop,                        // discard top of stack
    Return,                     // result is top of stack
    Index,                      // pop idx then base → element (scalar indices only, no slices)
    Loop(LoopForm, usize),      // pop `usize` already-evaluated args, run a flat
                                // bounded-iteration loop (sum/prod/iterate/scan),
                                // push result. The only GPU-safe recursion analogue
                                // and the in-VM form of the special-form loops, so
                                // they no longer force a tree-walk fallback (TODO 1e).
    MakeClosure {               // build Val::Fn capturing free vars from the stack
        params:    Vec<String>,
        body:      Arc<Expr>,
        code:      Arc<Vec<Instruction>>, // eagerly pre-compiled; empty = lazy fallback
        free_vars: Vec<String>,           // names to pop from stack into captured env
    },
}

/// Which bounded-iteration special form a `Loop` instruction runs. All four share
/// the "evaluate the operands once, then loop with no native-stack growth" shape;
/// the executor dispatches on this tag to the matching `*_vals` core in `eval.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopForm {
    Sum,      // sum(T) | sum(T,axis) | sum(f,n) | sum(f,lo,hi)
    Prod,     // prod(T) | prod(f,n) | prod(f,lo,hi)
    Iterate,  // iterate(f, x0, n)  → fⁿ(x0)
    Scan,     // scan(f, x0, n)     → [x0, …, fⁿ(x0)] stacked
}
