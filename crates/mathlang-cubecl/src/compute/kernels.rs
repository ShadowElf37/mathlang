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

// ── df64 (double-single) kernels ────────────────────────────────────────────────
//
// Each logical element is two f32 `(hi, lo)` stored interleaved: `data[2i]=hi`,
// `data[2i+1]=lo`, with `value ≈ hi + lo` and `|lo| ≤ ½ ulp(hi)` after
// normalization. Arithmetic uses error-free transforms (Knuth TwoSum, Dekker
// TwoProd via a 12-bit split) — no FMA dependency, so it runs on every backend
// including wgpu/Metal. References: Dekker 1971; Hida/Li/Bailey double-double.
//
// Implemented: + − × ÷, comparisons, neg, abs. Pow and transcendentals (exp, ln,
// sin, …) are not yet df64 — they need range-reduced double-single series and are
// staged in the dispatcher.

#[cube]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[cube]
fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    // assumes |a| ≥ |b|
    let s = a + b;
    let err = b - (s - a);
    (s, err)
}

#[cube]
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    // Dekker split at 12 bits (f32 significand is 24 bits): factor 2^12 + 1.
    let split = f32::new(4097.0);
    let ca = split * a;
    let ahi = ca - (ca - a);
    let alo = a - ahi;
    let cb = split * b;
    let bhi = cb - (cb - b);
    let blo = b - bhi;
    let p = a * b;
    let err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo;
    (p, err)
}

#[cube]
fn df_add(ah: f32, al: f32, bh: f32, bl: f32) -> (f32, f32) {
    let (sh, sl) = two_sum(ah, bh);
    let (th, tl) = two_sum(al, bl);
    let (vh, vl) = quick_two_sum(sh, sl + th);
    quick_two_sum(vh, vl + tl)
}

#[cube]
fn df_mul(ah: f32, al: f32, bh: f32, bl: f32) -> (f32, f32) {
    let (ph, pl) = two_prod(ah, bh);
    quick_two_sum(ph, pl + (ah * bl + al * bh))
}

#[cube]
fn df_mul_scalar(ah: f32, al: f32, s: f32) -> (f32, f32) {
    let (ph, pl) = two_prod(ah, s);
    quick_two_sum(ph, pl + al * s)
}

#[cube]
fn df_div(ah: f32, al: f32, bh: f32, bl: f32) -> (f32, f32) {
    let q1 = ah / bh;
    let (m1h, m1l) = df_mul_scalar(bh, bl, q1);
    let (r1h, r1l) = df_add(ah, al, -m1h, -m1l); // a - q1*b
    let q2 = r1h / bh;
    let (m2h, m2l) = df_mul_scalar(bh, bl, q2);
    let (r2h, _r2l) = df_add(r1h, r1l, -m2h, -m2l); // r - q2*b (only hi needed for q3)
    let q3 = r2h / bh;
    let (sh, sl) = quick_two_sum(q1, q2);
    df_add(sh, sl, q3, f32::new(0.0))
}

#[cube(launch)]
pub fn df64_binop(lhs: &Array<f32>, rhs: &Array<f32>, out: &mut Array<f32>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    let n = out.len() / 2;
    if i < n {
        let llen = lhs.len() / 2;
        let rlen = rhs.len() / 2;
        let li = (i % llen) * 2;
        let ri = (i % rlen) * 2;
        let ah = lhs[li];
        let al = lhs[li + 1];
        let bh = rhs[ri];
        let bl = rhs[ri + 1];

        let mut rh = f32::new(0.0);
        let mut rl = f32::new(0.0);
        if comptime![op == OP_ADD] {
            let (h, l) = df_add(ah, al, bh, bl);
            rh = h;
            rl = l;
        } else if comptime![op == OP_SUB] {
            let (h, l) = df_add(ah, al, -bh, -bl);
            rh = h;
            rl = l;
        } else if comptime![op == OP_MUL] {
            let (h, l) = df_mul(ah, al, bh, bl);
            rh = h;
            rl = l;
        } else if comptime![op == OP_DIV] {
            let (h, l) = df_div(ah, al, bh, bl);
            rh = h;
            rl = l;
        } else {
            // comparisons: sign of the normalized difference a − b (hi decides)
            let (dh, _dl) = df_add(ah, al, -bh, -bl);
            let zero = f32::new(0.0);
            if comptime![op == OP_LT] {
                rh = f32::cast_from(dh < zero);
            } else if comptime![op == OP_GT] {
                rh = f32::cast_from(dh > zero);
            } else if comptime![op == OP_LE] {
                rh = f32::cast_from(dh <= zero);
            } else if comptime![op == OP_GE] {
                rh = f32::cast_from(dh >= zero);
            } else if comptime![op == OP_EQ] {
                rh = f32::cast_from(dh == zero);
            } else if comptime![op == OP_NE] {
                rh = f32::cast_from(dh != zero);
            }
        }
        out[i * 2] = rh;
        out[i * 2 + 1] = rl;
    }
}

#[cube(launch)]
pub fn df64_unary(x: &Array<f32>, out: &mut Array<f32>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    let n = out.len() / 2;
    if i < n {
        let h = x[i * 2];
        let l = x[i * 2 + 1];
        let mut rh = h;
        let mut rl = l;
        if comptime![op == UN_NEG] {
            rh = -h;
            rl = -l;
        } else if comptime![op == UN_ABS] {
            if h < f32::new(0.0) {
                rh = -h;
                rl = -l;
            }
        }
        out[i * 2] = rh;
        out[i * 2 + 1] = rl;
    }
}
