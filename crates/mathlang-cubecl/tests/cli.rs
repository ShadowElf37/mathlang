//! CLI integration tests for the `mc` binary.
//!
//! Mirrors tests.sh: `check` → exact match on stdout+stderr (trailing newlines
//! stripped, same as shell `$()`); `check_repl` → stdin-piped REPL session,
//! assert output contains a substring.

use std::io::Write;
use std::process::{Command, Stdio};

fn mc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mc")
}

/// Run `mc <expr>` and return stdout+stderr with trailing newlines stripped.
fn run(expr: &str) -> String {
    let out = Command::new(mc_bin())
        .arg(expr)
        .output()
        .expect("failed to run mc");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    combined.trim_end_matches('\n').to_string()
}

/// Run `mc` in REPL mode with `input` piped to stdin, return combined output.
fn run_repl(input: &str) -> String {
    let mut child = Command::new(mc_bin())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn mc");
    let feed = if input.ends_with('\n') {
        input.to_string()
    } else {
        format!("{}\n", input)
    };
    child.stdin.as_mut().unwrap().write_all(feed.as_bytes()).unwrap();
    drop(child.stdin.take());
    let out = child.wait_with_output().unwrap();
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    )
}

/// Assert exact output match (mirrors shell `check`).
macro_rules! check {
    ($expr:expr, $expected:expr) => {{
        let got = run($expr);
        assert_eq!(got, $expected, "expr: {:?}", $expr);
    }};
}

/// Assert REPL output contains substring (mirrors shell `check_repl`).
macro_rules! check_repl {
    ($input:expr, $substr:expr) => {{
        let got = run_repl($input);
        assert!(
            got.contains($substr),
            "expected to find {:?}\ngot:\n{}",
            $substr,
            got
        );
    }};
}

/// Assert two expressions produce identical output (for tests using command-
/// substitution expected values in tests.sh, e.g. `"$($MC '(1,2;3,4)')"` ).
fn check_parity(expr: &str, oracle: &str) {
    assert_eq!(
        run(expr),
        run(oracle),
        "parity: {:?} vs {:?}",
        expr,
        oracle
    );
}

// ── scalar / complex / tuple core (Phase 1b) ─────────────────────────────────

#[test]
fn scalar_pi_squared() {
    check!("pi * 2^2", "12.566370614359172");
}

#[test]
fn scalar_implicit_mul() {
    check!("2pi", "6.283185307179586");
}

#[test]
fn scalar_pythagorean() {
    check!("x=3; y=4; sqrt(x^2+y^2)", "5");
}

#[test]
fn complex_i_squared() {
    check!("i^2", "-1");
}

#[test]
fn complex_euler() {
    check!("exp(i*pi)", "-1");
}

#[test]
fn complex_sqrt_neg1() {
    check!("sqrt(-1)", "i");
}

#[test]
fn complex_abs() {
    check!("abs(3+4i)", "5");
}

#[test]
fn tuple_add() {
    check!("(1,2,3) + (4,5,6)", "(5, 7, 9)");
}

#[test]
fn sum_fn() {
    check!("sum(x -> x, 1, 100)", "5050");
}

#[test]
fn iterate_double() {
    check!("iterate(x -> 2*x, 1, 10)", "1024");
}

#[test]
fn map_square() {
    check!("map(x -> x^2, (1,2,3,4))", "(1, 4, 9, 16)");
}

// ── tensors on the compute path (Phase 2) ────────────────────────────────────

#[test]
fn tensor_add() {
    check!("[1,2,3] + [4,5,6]", "[5, 7, 9]");
}

#[test]
fn tensor_scalar_mul_right() {
    check!("[1,2,3] * 2", "[2, 4, 6]");
}

#[test]
fn tensor_scalar_mul_left() {
    check!("2 * [1,2,3]", "[2, 4, 6]");
}

#[test]
fn tensor_scalar_div() {
    check!("[1,2,3,4] / 2", "[0.5, 1, 1.5, 2]");
}

#[test]
fn tensor_pow() {
    check!("[1,2,3] ^ 2", "[1, 4, 9]");
}

#[test]
fn tensor_gt() {
    check!("[1,2,3] > 2", "[0, 0, 1]");
}

#[test]
fn tensor_sin_zeros() {
    check!("sin(zeros(3))", "[0, 0, 0]");
}

#[test]
fn tensor_sqrt() {
    check!("sqrt([1,4,9,16])", "[1, 2, 3, 4]");
}

#[test]
fn tensor_exp() {
    check!("exp([0,1])", "[1, 2.718281828459045]");
}

#[test]
fn tensor_linspace() {
    check!("linspace(0,1,5)", "[0, 0.25, 0.5, 0.75, 1]");
}

#[test]
fn tensor_range() {
    check!("range(0,5)", "[0, 1, 2, 3, 4]");
}

#[test]
fn tensor_shape_3d() {
    check!("shape(ones(2,3,4))", "[2, 3, 4]");
}

#[test]
fn tensor_rows_matrix() {
    check!("rows((1,2,3;4,5,6))", "2");
}

#[test]
fn tensor_sum() {
    check!("sum([1,2,3,4])", "10");
}

#[test]
fn tensor_len() {
    check!("len(linspace(0,1,10))", "10");
}

#[test]
fn tensor_neg() {
    check!("-[1,2,3]", "[-1, -2, -3]");
}

// ── linear algebra + reductions (Phase 3) ────────────────────────────────────

#[test]
fn linalg_matvec() {
    check!("(1,2;3,4) @ [1,1]", "[3, 7]");
}

#[test]
fn linalg_vecmat() {
    check!("[1,1] @ (1,2;3,4)", "[4, 6]");
}

#[test]
fn linalg_dot() {
    check!("[1,2,3] @ [4,5,6]", "32");
}

#[test]
fn linalg_sum() {
    check!("sum([1,2,3,4,5])", "15");
}

#[test]
fn linalg_prod() {
    check!("prod([1,2,3,4])", "24");
}

#[test]
fn linalg_mean() {
    check!("mean([1,2,3,4])", "2.5");
}

#[test]
fn linalg_min() {
    check!("min([3,1,4,1,5])", "1");
}

#[test]
fn linalg_max() {
    check!("max([3,1,4,1,5])", "5");
}

#[test]
fn linalg_norm() {
    check!("norm([3,4])", "5");
}

#[test]
fn linalg_std() {
    check!("std([2,4,4,4,5,5,7,9])", "2");
}

#[test]
fn linalg_large_sum() {
    check!("sum(ones(100,100))", "10000");
}

#[test]
fn linalg_df64_matmul_staged() {
    check_repl!("!prec df64\n(1,2;3,4) @ (5,6;7,8)", "df64 matmul is staged");
}

// ── complex tensors (Phase 5) ─────────────────────────────────────────────────

#[test]
fn complex_tensor_display() {
    check!("[1+2i, 3+4i]", "[1 + 2i, 3 + 4i]");
}

#[test]
fn complex_tensor_real_plus_cx_scalar() {
    check!("[1, 2, 3] + 2i", "[1 + 2i, 2 + 2i, 3 + 2i]");
}

#[test]
fn complex_tensor_mul() {
    check!("[1+1i] * [1+1i]", "[2i]");
}

#[test]
fn complex_tensor_abs() {
    check!("abs([3+4i, 5+12i])", "[5, 13]");
}

#[test]
fn complex_tensor_re() {
    check!("re([1+2i, 3+4i])", "[1, 3]");
}

#[test]
fn complex_tensor_conj() {
    check!("conj([1+2i, 3-4i])", "[1 - 2i, 3 + 4i]");
}

#[test]
fn complex_tensor_sqrt() {
    check!("sqrt([3+4i])", "[2 + i]");
}

#[test]
fn complex_tensor_exp_euler() {
    check!(
        "exp([0+0i, 0+3.141592653589793i])",
        "[1, -1]"
    );
}

#[test]
fn complex_tensor_sum() {
    check!("sum([1+2i, 3+4i, 5+6i])", "9 + 12i");
}

#[test]
fn complex_tensor_type_repl() {
    check_repl!("!type [1+2i, 3]", "complex tensor");
}

// ── fields & differential forms ──────────────────────────────────────────────

#[test]
fn field_tensor_roundtrip() {
    check!("tensor(field([1,2,3,4], 0, 4, forms.periodic))", "[1, 2, 3, 4]");
}

#[test]
fn field_fn_ctor() {
    // expected: $($MC '[0, 0.0625, 0.25, 0.5625, 1]') — the array literal itself
    check!(
        "tensor(field((x) -> x*x, 0, 1, 5, forms.neumann))",
        "[0, 0.0625, 0.25, 0.5625, 1]"
    );
}

#[test]
fn field_arith() {
    check!(
        "tensor(2*field([1,2,3,4],0,4,forms.periodic) + field([10,20,30,40],0,4,forms.periodic))",
        "[12, 24, 36, 48]"
    );
}

#[test]
fn field_hodge_roundtrip() {
    check!("tensor(forms.hodge(field([1,2,3],0,3,forms.periodic)))", "[1, 2, 3]");
}

#[test]
fn field_type_repl() {
    check_repl!("!type field([1,2,3], 0, 3, forms.periodic)", "form");
}

#[test]
fn field_laplace_sign() {
    check_repl!(
        "f = field((x)->sin(x), 0, 2*pi, 32, forms.periodic)\nnorm(tensor(forms.laplace(f)) - tensor(f)) < 0.1",
        "1"
    );
}

#[test]
fn forms_dd_zero() {
    check_repl!(
        "f = field((x,y)->x*y, 0, 1, (8,8), forms.periodic)\nnorm(tensor(forms.d(forms.d(f)))) < 1e-9",
        "1"
    );
}

// ── spectral: fft / ifft + ops ────────────────────────────────────────────────

#[test]
fn fft_basic() {
    check!("fft([1,1,0,0])", "[2, 1 - i, 0, 1 + i]");
}

#[test]
fn fft_four() {
    check!("fft([1,2,3,4])", "[10, -2 + 2i, -2, -2 - 2i]");
}

#[test]
fn ifft_roundtrip_re() {
    check!("re(ifft(fft([1,2,3,4])))", "[1, 2, 3, 4]");
}

#[test]
fn ifft_roundtrip_im() {
    check!("im(ifft(fft([1,2,3,4])))", "[0, 0, 0, 0]");
}

#[test]
fn specgrad_exact() {
    check_repl!(
        "x = linspace(0, 2*pi - 2*pi/32, 32)\nnorm(ops.specgrad(sin(x), 2*pi/32, 0) - cos(x)) < 1e-10",
        "1"
    );
}

#[test]
fn poisson_zero_mean() {
    check!(
        "round(sum(ops.poisson([1.0,-2.0,1.0,0.0,1.0,-2.0,1.0,0.0], 1.0)))",
        "0"
    );
}

// ── calculus (Phase: integral/deriv, scalar + multidim) ───────────────────────

#[test]
fn calc_integral_x2() {
    check!("integral(x -> x^2, 0, 1)", "0.33333333333333315");
}

#[test]
fn calc_deriv_x3() {
    check!("deriv(x -> x^3, 2)", "12.000000000182233");
}

#[test]
fn calc_partial_dx() {
    check!("round(deriv((x,y) -> x^2*y, (3,5), 0))", "30");
}

#[test]
fn calc_gradient_sum() {
    check!("round(sum(deriv((x,y) -> x^2*y, (3,5))))", "39");
}

#[test]
fn calc_double_integral() {
    check!("integral((x,y) -> x*y, [0,0], [1,1])", "0.25");
}

#[test]
fn calc_box_volume() {
    check!("integral((x,y,z) -> 1, [0,0,0], [2,3,4])", "24");
}

#[test]
fn calc_newton_sqrt2() {
    check!(
        "iterate(x -> x - (x^2-2)/deriv(t -> t^2-2, x), 1.0, 5)",
        "1.4142135623730951"
    );
}

// ── indexing / slicing / constructors / assembly / branching / axis reductions

#[test]
fn idx_2d_scalar() {
    check!("A=(1,2,3;4,5,6;7,8,9); A[1,2]", "6");
}

#[test]
fn idx_2d_row_all() {
    check!("A=(1,2,3;4,5,6;7,8,9); A[0, ..]", "[1, 2, 3]");
}

#[test]
fn idx_2d_col_all() {
    check!("A=(1,2,3;4,5,6;7,8,9); A[.., 0]", "[1, 4, 7]");
}

#[test]
fn idx_1d_slice() {
    check!("[10,20,30,40,50][1..3]", "[20, 30, 40]");
}

#[test]
fn lingrid_1d_square() {
    check!("lingrid(0, 1, 5, x -> x^2)", "[0, 0.0625, 0.25, 0.5625, 1]");
}

#[test]
fn tensor_shape_2d() {
    check!("shape(tensor((i,j) -> i+j, 3, 4))", "[3, 4]");
}

#[test]
fn reshape_parity() {
    check_parity("reshape([1,2,3,4,5,6], 2, 3)", "(1,2,3;4,5,6)");
}

#[test]
fn transpose_parity() {
    check_parity("transpose((1,2,3;4,5,6))", "(1,4;2,5;3,6)");
}

#[test]
fn vstack_parity() {
    check_parity("vstack((1,2;3,4),(5,6))", "(1,2;3,4;5,6)");
}

#[test]
fn hstack_parity() {
    check_parity("hstack((1,2),(3,4))", "(1,3;2,4)");
}

#[test]
fn sign_vector() {
    check!("sign([-2, 0, 3])", "[-1, 0, 1]");
}

#[test]
fn max_elementwise() {
    check!("max([1,5,2], [3,1,4])", "[3, 5, 4]");
}

#[test]
fn select_op() {
    check!("select([1,0,1], [10,20,30], [-1,-2,-3])", "[10, -2, 30]");
}

#[test]
fn sum_axis0() {
    check!("sum((1,2,3;4,5,6), 0)", "[5, 7, 9]");
}

#[test]
fn sum_axis1() {
    check!("sum((1,2,3;4,5,6), 1)", "[6, 15]");
}

// ── stencils + dense linalg (Phase 3.x) ──────────────────────────────────────

#[test]
fn roll_right() {
    check!("roll([1,2,3,4], 1, 0)", "[4, 1, 2, 3]");
}

#[test]
fn roll_left() {
    check!("roll([1,2,3,4], -1, 0)", "[2, 3, 4, 1]");
}

#[test]
fn shift_right() {
    check!("shift([1,2,3,4], 1, 0)", "[1, 1, 2, 3]");
}

#[test]
fn ops_lap_1d() {
    check!("ops.lap([0,0,1,0,0], 1)", "[0, 1, -2, 1, 0]");
}

#[test]
fn ops_grad_1d() {
    check!("ops.grad([1,4,9,16,25], 1, 0)", "[-10.5, 4, 6, 8, -7.5]");
}

#[test]
fn det_2x2() {
    check!("det((1,2;3,4))", "-2");
}

#[test]
fn det_3x3_diag() {
    check!("det((2,0,0;0,3,0;0,0,4))", "24");
}

#[test]
fn solve_2x2_tuple() {
    check!("solve((2,1;1,3),(5,10))", "[1, 3]");
}

#[test]
fn solve_2x2_array() {
    check!("solve((2,1;1,3),[5,10])", "[1, 3]");
}

#[test]
fn heat_neumann_conservation() {
    check!(
        "round(sum(iterate(u -> u + 0.2*ops.lap(u,1,ops.neumann), (0,0,0;0,9,0;0,0,0), 10)))",
        "9"
    );
}

#[test]
fn eigvals_diag() {
    check!("eigvals((2,0;0,3))", "[2, 3]");
}

#[test]
fn trace_2x2() {
    check!("trace((1,2;3,4))", "5");
}

#[test]
fn diag_from_tuple_parity() {
    check_parity("diag((1,2,3))", "(1,0,0;0,2,0;0,0,3)");
}

#[test]
fn diag_from_matrix() {
    check!("diag((2,0;0,5))", "[2, 5]");
}

// ── resident loops: iterate / scan (Phase 4) ──────────────────────────────────

#[test]
fn iterate_scalar() {
    check!("iterate(x -> 2*x, 1, 10)", "1024");
}

#[test]
fn iterate_tensor_resident() {
    check!("iterate(u -> u*0.5, [1,2,3,4], 3)", "[0.125, 0.25, 0.375, 0.5]");
}

#[test]
fn iterate_tuple_swap() {
    check!("iterate((u,v) -> (v, u), ([1,2],[3,4]), 1)", "([3, 4], [1, 2])");
}

#[test]
fn scan_scalar() {
    check!("scan(x -> 2*x, 1, 4)", "[1, 2, 4, 8, 16]");
}

#[test]
fn scan_increment() {
    check!("scan(x -> x+1, 0, 3)", "[0, 1, 2, 3]");
}

#[test]
fn scan_tensor_shape() {
    check!("shape(scan(u -> u*0.5, [1,2,3], 5))", "[6, 3]");
}

#[test]
fn scan_tuple_shape() {
    check!("shape(scan(v -> (v[1], -v[0]), (1,0), 4))", "[5, 2]");
}

// ── precision: f64 on cpu; wgpu downgrades to f32; df64 arithmetic ────────────

#[test]
fn cpu_f64_precision() {
    check!("[1.0] + [1e-10]", "[1.0000000001]");
}

#[test]
fn df64_add() {
    check_repl!("!prec df64\n[1.0] + [1e-10]", "1.0000000001");
}

#[test]
fn df64_div() {
    check_repl!("!prec df64\n[1.0] / [3.0]", "0.33333333333333");
}

#[test]
fn df64_mul() {
    check_repl!("!prec df64\n[0.1] * [0.1]", "0.0100000000000000");
}

// wgpu tests require GPU hardware — skipped by default, run with `cargo test -- --ignored`

#[test]
#[ignore = "requires wgpu hardware"]
fn wgpu_f32_loses_precision() {
    check_repl!("!backend wgpu\n[1.0] + [1e-10]", "[1]");
}

#[test]
#[ignore = "requires wgpu hardware"]
fn wgpu_rejects_f64() {
    check_repl!("!backend wgpu\n!prec f64", "no native f64");
}

#[test]
#[ignore = "requires wgpu hardware"]
fn wgpu_df64_storage_roundtrip() {
    check_repl!("!backend wgpu\n!prec df64\n[0.5, 0.25]", "[0.5, 0.25]");
}

#[test]
#[ignore = "requires wgpu hardware"]
fn wgpu_df64_warns_unreliable() {
    check_repl!(
        "!backend wgpu\n!prec df64\n[1.0] + [1.0]",
        "unreliable on wgpu"
    );
}
