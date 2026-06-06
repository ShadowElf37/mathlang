//! Generic `#[cube]` kernels — the eager primitive library. Each is generic over
//! the float element type `F`, so the *same source* monomorphizes to f32 (any
//! backend) or f64 (cpu/cuda/hip). The comptime `op` selector keeps each operation
//! a tight specialized kernel while avoiding one kernel per operator.
//!
//! Broadcasting: operands are indexed `i % len`, so a length-1 operand broadcasts a
//! scalar against a tensor and equal lengths do elementwise — more than the
//! original WGSL backend offered (which required matching shapes).
//!
//! df64 (double-single) kernels are a separate path (`mod df64`) because they
//! operate on packed f32 pairs, not a single `F`. They are staged — see the notes
//! there.

use cubecl::prelude::*;

// ── binary op codes (shared with mod.rs dispatch) ───────────────────────────────
pub const OP_ADD: u32 = 0;
pub const OP_SUB: u32 = 1;
pub const OP_MUL: u32 = 2;
pub const OP_DIV: u32 = 3;
pub const OP_POW: u32 = 4;
pub const OP_MIN: u32 = 5;
pub const OP_MAX: u32 = 6;
pub const OP_LT: u32 = 7;
pub const OP_GT: u32 = 8;
pub const OP_LE: u32 = 9;
pub const OP_GE: u32 = 10;
pub const OP_EQ: u32 = 11;
pub const OP_NE: u32 = 12;

#[cube]
fn apply_bin<F: Float>(a: F, b: F, #[comptime] op: u32) -> F {
    let mut r = a + b;
    if comptime![op == OP_SUB] {
        r = a - b;
    } else if comptime![op == OP_MUL] {
        r = a * b;
    } else if comptime![op == OP_DIV] {
        r = a / b;
    } else if comptime![op == OP_POW] {
        r = F::powf(a, b);
    } else if comptime![op == OP_MIN] {
        r = F::min(a, b);
    } else if comptime![op == OP_MAX] {
        r = F::max(a, b);
    } else if comptime![op == OP_LT] {
        r = F::cast_from(a < b);
    } else if comptime![op == OP_GT] {
        r = F::cast_from(a > b);
    } else if comptime![op == OP_LE] {
        r = F::cast_from(a <= b);
    } else if comptime![op == OP_GE] {
        r = F::cast_from(a >= b);
    } else if comptime![op == OP_EQ] {
        r = F::cast_from(a == b);
    } else if comptime![op == OP_NE] {
        r = F::cast_from(a != b);
    }
    r
}

#[cube(launch)]
pub fn ew_binop<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let a = lhs[i % lhs.len()];
        let b = rhs[i % rhs.len()];
        out[i] = apply_bin::<F>(a, b, op);
    }
}

// ── unary op codes ──────────────────────────────────────────────────────────────
pub const UN_NEG: u32 = 0;
pub const UN_ABS: u32 = 1;
pub const UN_EXP: u32 = 2;
pub const UN_LN: u32 = 3;
pub const UN_SQRT: u32 = 4;
pub const UN_SIN: u32 = 5;
pub const UN_COS: u32 = 6;
pub const UN_TAN: u32 = 7;
pub const UN_ASIN: u32 = 8;
pub const UN_ACOS: u32 = 9;
pub const UN_ATAN: u32 = 10;
pub const UN_SINH: u32 = 11;
pub const UN_COSH: u32 = 12;
pub const UN_TANH: u32 = 13;
pub const UN_TRUNC: u32 = 14;
pub const UN_DEG: u32 = 15;
pub const UN_RAD: u32 = 16;

#[cube]
fn apply_un<F: Float>(x: F, #[comptime] op: u32) -> F {
    let mut r = F::abs(x);
    if comptime![op == UN_NEG] {
        r = F::new(0.0) - x;
    } else if comptime![op == UN_EXP] {
        r = F::exp(x);
    } else if comptime![op == UN_LN] {
        r = F::ln(x);
    } else if comptime![op == UN_SQRT] {
        r = F::sqrt(x);
    } else if comptime![op == UN_SIN] {
        r = F::sin(x);
    } else if comptime![op == UN_COS] {
        r = F::cos(x);
    } else if comptime![op == UN_TAN] {
        r = F::tan(x);
    } else if comptime![op == UN_ASIN] {
        r = F::asin(x);
    } else if comptime![op == UN_ACOS] {
        r = F::acos(x);
    } else if comptime![op == UN_ATAN] {
        r = F::atan(x);
    } else if comptime![op == UN_SINH] {
        r = F::sinh(x);
    } else if comptime![op == UN_COSH] {
        r = F::cosh(x);
    } else if comptime![op == UN_TANH] {
        r = F::tanh(x);
    } else if comptime![op == UN_TRUNC] {
        r = F::trunc(x);
    } else if comptime![op == UN_DEG] {
        r = F::to_degrees(x);
    } else if comptime![op == UN_RAD] {
        r = F::to_radians(x);
    }
    r
}

#[cube(launch)]
pub fn ew_unary<F: Float>(x: &Array<F>, out: &mut Array<F>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = apply_un::<F>(x[i], op);
    }
}
