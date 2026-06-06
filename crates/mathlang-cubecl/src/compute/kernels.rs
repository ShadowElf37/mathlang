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

// ── stencils (1-D and 2-D, modelled as r×c; 1-D [N] = (N,1)) ────────────────────
// Periodic (roll) and edge-clamp (shift/Neumann) neighbour access. `scale` is a
// length-1 array carrying 1/dx² (lap) or 1/(2dx) (grad), avoiding a scalar arg.

/// roll along one axis: a periodic gather. `add_r`/`add_c` = (L − n mod L) mod L,
/// precomputed host-side (only one is nonzero for a single-axis roll).
#[cube(launch)]
pub fn roll2d<F: Float>(input: &Array<F>, out: &mut Array<F>, #[comptime] r: usize, #[comptime] c: usize, #[comptime] add_r: usize, #[comptime] add_c: usize) {
    let idx = ABSOLUTE_POS;
    if idx < r * c {
        let row = idx / c;
        let col = idx % c;
        out[idx] = input[((row + add_r) % r) * c + (col + add_c) % c];
    }
}

/// shift along one axis: edge-clamped (Neumann) gather. `nr`/`nc` = shift amount
/// (one nonzero), as comptime i32 so the clamp branches resolve per shape.
#[cube(launch)]
pub fn shift2d<F: Float>(input: &Array<F>, out: &mut Array<F>, #[comptime] r: usize, #[comptime] c: usize, #[comptime] nr: i32, #[comptime] nc: i32) {
    let idx = ABSOLUTE_POS;
    if idx < r * c {
        let row = idx / c;
        let col = idx % c;
        let sr = clamp_idx(row, nr, r);
        let sc = clamp_idx(col, nc, c);
        out[idx] = input[sr * c + sc];
    }
}

#[cube]
fn clamp_idx(i: usize, n: i32, len: usize) -> usize {
    let p = i32::cast_from(i) - n;
    let mut s = p;
    if p < 0i32 {
        s = 0i32;
    } else if p >= i32::cast_from(len) {
        s = i32::cast_from(len) - 1i32;
    }
    usize::cast_from(s)
}

/// Index one step off `i` along an axis of length `len`, periodic or clamped.
#[cube]
fn nb(i: usize, up: bool, len: usize, #[comptime] periodic: u32) -> usize {
    let mut out = i;
    if up {
        if comptime![periodic == 1] {
            out = (i + 1) % len;
        } else if i + 1 < len {
            out = i + 1;
        }
    } else if comptime![periodic == 1] {
        out = (i + len - 1) % len;
    } else if i > 0 {
        out = i - 1;
    }
    out
}

/// 2-D (and 1-D, via c=1) Laplacian. `scale[0]` = 1/dx². For 1-D the column term
/// vanishes (c=1 ⇒ both column neighbours are the centre).
#[cube(launch)]
pub fn lap2d<F: Float>(u: &Array<F>, scale: &Array<F>, out: &mut Array<F>, #[comptime] r: usize, #[comptime] c: usize, #[comptime] periodic: u32) {
    let idx = ABSOLUTE_POS;
    if idx < r * c {
        let row = idx / c;
        let col = idx % c;
        let up = nb(row, true, r, periodic);
        let dn = nb(row, false, r, periodic);
        let lf = nb(col, false, c, periodic);
        let rt = nb(col, true, c, periodic);
        let center = u[row * c + col];
        let sum = u[up * c + col] + u[dn * c + col] + u[row * c + lf] + u[row * c + rt];
        out[idx] = (sum - F::new(4.0) * center) * scale[0];
    }
}

/// Central-difference gradient along one axis. `scale[0]` = 1/(2·dx).
#[cube(launch)]
pub fn grad2d<F: Float>(u: &Array<F>, scale: &Array<F>, out: &mut Array<F>, #[comptime] r: usize, #[comptime] c: usize, #[comptime] axis: u32, #[comptime] periodic: u32) {
    let idx = ABSOLUTE_POS;
    if idx < r * c {
        let row = idx / c;
        let col = idx % c;
        let mut hi = u[idx];
        let mut lo = u[idx];
        if comptime![axis == 0] {
            hi = u[nb(row, true, r, periodic) * c + col];
            lo = u[nb(row, false, r, periodic) * c + col];
        } else {
            hi = u[row * c + nb(col, true, c, periodic)];
            lo = u[row * c + nb(col, false, c, periodic)];
        }
        out[idx] = (hi - lo) * scale[0];
    }
}

// ── complex kernels (interleaved [re, im], generic f32/f64) ─────────────────────
// A complex tensor of n logical elements is 2n values: data[2i]=re, data[2i+1]=im.
// cbinop reuses the real OP_* codes (+ - * /); unary ops have their own codes.

/// Promote a real array to interleaved complex (im = 0).
#[cube(launch)]
pub fn real_to_complex<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < x.len() {
        out[i * 2] = x[i];
        out[i * 2 + 1] = F::new(0.0);
    }
}

#[cube(launch)]
pub fn cbinop<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    let n = out.len() / 2;
    if i < n {
        let na = a.len() / 2;
        let nb = b.len() / 2;
        let ia = (i % na) * 2;
        let ib = (i % nb) * 2;
        let ar = a[ia];
        let ai = a[ia + 1];
        let br = b[ib];
        let bi = b[ib + 1];
        let mut rr = ar + br;
        let mut ri = ai + bi;
        if comptime![op == OP_SUB] {
            rr = ar - br;
            ri = ai - bi;
        } else if comptime![op == OP_MUL] {
            rr = ar * br - ai * bi;
            ri = ar * bi + ai * br;
        } else if comptime![op == OP_DIV] {
            let d = br * br + bi * bi;
            rr = (ar * br + ai * bi) / d;
            ri = (ai * br - ar * bi) / d;
        }
        out[i * 2] = rr;
        out[i * 2 + 1] = ri;
    }
}

// complex → complex unary
pub const CU_NEG: u32 = 0;
pub const CU_CONJ: u32 = 1;
pub const CU_EXP: u32 = 2;
pub const CU_LN: u32 = 3;
pub const CU_SQRT: u32 = 4;
pub const CU_SIN: u32 = 5;
pub const CU_COS: u32 = 6;

#[cube(launch)]
pub fn cunary_c2c<F: Float>(x: &Array<F>, out: &mut Array<F>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    let n = out.len() / 2;
    if i < n {
        let xr = x[i * 2];
        let xi = x[i * 2 + 1];
        let mut rr = -xr;
        let mut ri = -xi;
        if comptime![op == CU_CONJ] {
            rr = xr;
            ri = -xi;
        } else if comptime![op == CU_EXP] {
            let m = F::exp(xr);
            rr = m * F::cos(xi);
            ri = m * F::sin(xi);
        } else if comptime![op == CU_LN] {
            rr = F::ln(F::sqrt(xr * xr + xi * xi));
            ri = F::atan2(xi, xr);
        } else if comptime![op == CU_SQRT] {
            let r = F::sqrt(F::sqrt(xr * xr + xi * xi));
            let th = F::atan2(xi, xr) * F::new(0.5);
            rr = r * F::cos(th);
            ri = r * F::sin(th);
        } else if comptime![op == CU_SIN] {
            rr = F::sin(xr) * F::cosh(xi);
            ri = F::cos(xr) * F::sinh(xi);
        } else if comptime![op == CU_COS] {
            rr = F::cos(xr) * F::cosh(xi);
            ri = -(F::sin(xr) * F::sinh(xi));
        }
        out[i * 2] = rr;
        out[i * 2 + 1] = ri;
    }
}

// complex → real unary
pub const CR_RE: u32 = 0;
pub const CR_IM: u32 = 1;
pub const CR_ABS: u32 = 2;
pub const CR_ARG: u32 = 3;

#[cube(launch)]
pub fn cunary_c2r<F: Float>(x: &Array<F>, out: &mut Array<F>, #[comptime] op: u32) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let xr = x[i * 2];
        let xi = x[i * 2 + 1];
        let mut r = xr;
        if comptime![op == CR_IM] {
            r = xi;
        } else if comptime![op == CR_ABS] {
            r = F::sqrt(xr * xr + xi * xi);
        } else if comptime![op == CR_ARG] {
            r = F::atan2(xi, xr);
        }
        out[i] = r;
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

// ── matmul (naive one-thread-per-output, generic f32/f64) ───────────────────────
// `a` is m×k row-major, `b` is k×n row-major, `out` is m×n. k and n are comptime so
// the kernel specializes per shape (cached per (k, n, element type)).
#[cube(launch)]
pub fn matmul_kernel<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < out.len() {
        let row = idx / n;
        let col = idx % n;
        let mut acc = F::new(0.0);
        for p in 0..k {
            acc += a[row * k + p] * b[p * n + col];
        }
        out[idx] = acc;
    }
}

// ── whole-tensor reduction ──────────────────────────────────────────────────────
// `nt` threads (= partials.len()) each grid-stride reduce their slice; the host
// combines the `nt` partials. Sum uses Neumaier compensation. Assumes n ≥ 1.
pub const RED_SUM: u32 = 0;
pub const RED_PROD: u32 = 1;
pub const RED_MIN: u32 = 2;
pub const RED_MAX: u32 = 3;

#[cube(launch)]
pub fn reduce_kernel<F: Float>(input: &Array<F>, partials: &mut Array<F>, #[comptime] op: u32) {
    let t = ABSOLUTE_POS;
    let nt = partials.len();
    let n = input.len();
    if t < nt {
        // identity: sum→0, prod→1, min/max→input[0] (a real element, harmless to re-include)
        let mut acc = F::new(0.0);
        if comptime![op == RED_PROD] {
            acc = F::new(1.0);
        } else if comptime![op == RED_MIN] {
            acc = input[0];
        } else if comptime![op == RED_MAX] {
            acc = input[0];
        }
        let mut comp = F::new(0.0); // Neumaier compensation (sum only)
        let num_blocks = (n + nt - 1) / nt;
        for blk in 0..num_blocks {
            let idx = blk * nt + t;
            if idx < n {
                let x = input[idx];
                if comptime![op == RED_SUM] {
                    let s = acc + x;
                    if F::abs(acc) >= F::abs(x) {
                        comp += (acc - s) + x;
                    } else {
                        comp += (x - s) + acc;
                    }
                    acc = s;
                } else if comptime![op == RED_PROD] {
                    acc *= x;
                } else if comptime![op == RED_MIN] {
                    acc = F::min(acc, x);
                } else if comptime![op == RED_MAX] {
                    acc = F::max(acc, x);
                }
            }
        }
        if comptime![op == RED_SUM] {
            acc += comp;
        }
        partials[t] = acc;
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
