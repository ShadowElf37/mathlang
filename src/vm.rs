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
use crate::ast::{Assign, Expr, Op};

#[derive(Debug, Clone)]
pub enum Instruction {
    PushNum(f64),
    PushComplex(f64, f64),
    LoadParam(usize),           // bind from args[i]
    LoadCaptured(String),       // live env lookup (Cells, Fns, Tensors)
    /// Index into the function's constant pool — a captured name resolved to a
    /// slot at compile time instead of hashing the string on every access.
    ///
    /// Sound because `captured` never changes for the life of a `Val::Fn`, so
    /// `captured[name]` is the same value on every call. Scalars are already
    /// inlined as literals, so this covers captured tensors, functions and cells,
    /// measured at ~140 ns per access before pooling.
    ///
    /// Two names must NOT be pooled, and both fail silently with a wrong value
    /// rather than an error (see TODO 1k):
    ///   - the function's own `FnSig::self_name`, since the executor resolves
    ///     that against the function being applied and a *previous* binding of
    ///     the same name may sit in `captured`;
    ///   - any name absent from `captured`, which stays `LoadCaptured(String)` so
    ///     forward references keep resolving live.
    LoadCapturedSlot(usize),
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
    /// Pop a record/namespace → its field. Without this, every body calling
    /// `vec.normalize(…)` or `bits.and(…)` — i.e. the whole namespaced library
    /// style — fell into the compiler's catch-all and tree-walked (TODO 1i).
    ///
    /// `who` is the base's source spelling, captured at compile time purely so a
    /// missing field reports the same message the tree-walk evaluator does.
    Member { field: String, who: String },
    Loop(LoopForm, usize),      // pop `usize` already-evaluated args, run a flat
                                // bounded-iteration loop (sum/prod/iterate/scan),
                                // push result. The only GPU-safe recursion analogue
                                // and the in-VM form of the special-form loops, so
                                // they no longer force a tree-walk fallback (TODO 1e).
    /// Tree-walk one *sub-expression* and push its value.
    ///
    /// The escape hatch that retires whole-body fallback. A node the compiler
    /// cannot emit code for used to abort the entire body — and because the
    /// `OnceLock` cached that failure, permanently — so one slice in one branch
    /// made all of the body's arithmetic pay interpreter cost too. Now only the
    /// unsupported node itself is interpreted.
    ///
    /// `binds` names the parameters and locals the sub-expression reads, resolved
    /// to their slots at compile time; the executor copies just those into a
    /// frame layered over the captured scope. Dedicated instructions (Slice,
    /// Range, …) are optimizations over this, not prerequisites for it.
    ///
    /// Not usable for anything that *writes* a binding: copying a value into a
    /// sub-frame detaches it from the slot being written.
    EvalSub { expr: Arc<Expr>, binds: Vec<(String, Slot)> },

    /// `T[i] = v` / `w.f += 1` inside a compiled block, writing a VM slot in place.
    ///
    /// The one thing `EvalSub` cannot do: an assignment mutates a *binding*, and
    /// copying that binding into a sub-frame would detach it from the slot the
    /// write has to land in. So the value is moved out of `slot`, mutated by the
    /// shared tree-walk writer (identical index rules and error text), and moved
    /// back — refcount 1 throughout, which is what keeps repeated writes O(1)
    /// instead of copying the buffer each time (TODO 1h).
    StoreAssign {
        slot:   usize,          // local holding the root binding
        assign: Arc<Assign>,    // the whole statement, re-used verbatim
        binds:  Vec<(String, Slot)>, // params/locals the RHS and indices read
    },

    MakeClosure {               // build Val::Fn capturing free vars from the stack
        params:    Vec<String>,
        body:      Arc<Expr>,
        code:      Arc<Vec<Instruction>>, // eagerly pre-compiled; empty = lazy fallback
        free_vars: Vec<String>,           // names to pop from stack into captured env
        /// Names backing `code`'s `LoadCapturedSlot` indices. Resolved against
        /// the closure's *actual* captured map when the closure is built, not at
        /// compile time — the inner body is compiled against a hint map holding
        /// placeholders for `free_vars`, so a compile-time pool would freeze the
        /// placeholder. Keeping names here (rather than values) is also what lets
        /// this file stay free of any dependency on `Val`.
        pool_names: Vec<String>,
        /// Set for a named local `f(x) = …`, so the closure can call itself
        /// (see FnSig::self_name). None for an anonymous lambda, which has no
        /// name to recurse through.
        self_name: Option<String>,
    },
}

/// Where an `EvalSub` binding's value comes from in the running frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Param(usize),
    Local(usize),
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
