#!/usr/bin/env bash
# Comprehensive test suite for mathlang.

M="${MATHLANG_BIN:-$HOME/mathlang/target/release/m}"
PASS=0
FAIL=0
FAILS=()

norm() { echo "$1" | tr -s ' \t\n' ' ' | sed 's/^ //; s/ $//'; }

run() {
    local label="$1" expr="$2" expected="$3" got
    got=$("$M" "$expr" 2>&1)
    if [ "$(norm "$got")" = "$(norm "$expected")" ]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | expected='$(norm "$expected")' | got='$(norm "$got")'")
    fi
}

run_match() {
    local label="$1" expr="$2" pat="$3" got
    got=$("$M" "$expr" 2>&1)
    if echo "$got" | grep -qE "$pat"; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | pattern='$pat' | got='$(norm "$got")'")
    fi
}

run_ok() {
    local label="$1" expr="$2" got
    got=$("$M" "$expr" 2>&1)
    if [ $? -eq 0 ] && ! echo "$got" | grep -qiE 'error|unknown|undefined|fail|panic'; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | got='$(norm "$got")'")
    fi
}

run_err() {
    local label="$1" expr="$2" got
    got=$("$M" "$expr" 2>&1)
    if echo "$got" | grep -qiE 'error|unknown|undefined'; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | expected error | got='$(norm "$got")'")
    fi
}

# run_lib: like `run`, but loads a mathlang library file first (m -f lib 'expr').
LIB_INT="${MATHLANG_LIB_INT:-examples/integrators.math}"
run_lib() {
    local label="$1" lib="$2" expr="$3" expected="$4" got
    got=$("$M" -f "$lib" "$expr" 2>&1)
    if [ "$(norm "$got")" = "$(norm "$expected")" ]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | lib='$lib' expr='$expr' | expected='$(norm "$expected")' | got='$(norm "$got")'")
    fi
}

section() { echo; echo "=== $1 ==="; }

# ── Arithmetic ────────────────────────────────────────────────────────────────
section "ARITHMETIC"
run "arith.add"       "3 + 4"         "7"
run "arith.sub"       "10 - 3"        "7"
run "arith.mul"       "6 * 7"         "42"
run "arith.div"       "10 / 4"        "2.5"
run "arith.floordiv"  "7 // 2"        "3"
run "arith.floordiv_neg" "-7 // 2"   "-4"
run "arith.mod"       "7 % 3"         "1"
run "arith.mod_neg"   "-7 % 3"        "-1"
run "arith.pow"       "2^10"          "1024"
run "arith.pow_star"  "2**10"         "1024"
run "arith.pow_frac"  "8^(1/3)"       "2"
run "arith.neg"       "-5"            "-5"
run "arith.neg_expr"  "-(3 + 4)"      "-7"
run "arith.prec_mul_add" "2 + 3 * 4" "14"
run "arith.prec_pow"  "2 + 3^2"       "11"
run "arith.right_assoc_pow" "2^3^2"  "512"
run "arith.parens"    "(2 + 3) * 4"  "20"
run "arith.unary_neg_pow" "-2^2"     "-4"
run "arith.zero_div"  "1 / 0"        "inf"
run "arith.neg_zero_div" "-1 / 0"    "-inf"

# ── Constants ─────────────────────────────────────────────────────────────────
section "CONSTANTS"
run_match "const.pi"   "pi"    "3.14159"
run_match "const.e"    "e"     "2.71828"
run_match "const.phi"  "phi"   "1.61803"
run       "const.inf"  "inf"   "inf"
run       "const.i2"   "i^2"   "-1"
run       "const.neginf" "-inf" "-inf"

# ── Variables ─────────────────────────────────────────────────────────────────
section "VARIABLES"
run "var.simple"    "x=3; y=4; x^2 + y^2"        "25"
run "var.reuse"     "x=5; x * x"                 "25"
run "var.chain"     "a=2; b=a*3; b"               "6"
run "var.expr_rhs"  "x=2^8; x"                    "256"
run "var.no_colon"  "x=3; x^2"                     "9"

# ── User functions ────────────────────────────────────────────────────────────
section "USER FUNCTIONS"
run "fn.one_arg"    "f(x) = x^2; f(5)"             "25"
run "fn.two_arg"    "g(x,y) = x^2 + y^2; g(3,4)"  "25"
run "fn.three_arg"  "h(a,b,c) = a+b+c; h(1,2,3)"  "6"
run "fn.compose"    "f(x) = x+1; g(x) = f(f(x)); g(3)" "5"
run "fn.recursive"  "f(n) = if(n <= 1, 1, n * f(n-1)); f(6)" "720"
run "fn.mutual_ref" "a(x) = x^2; b(x) = a(x) + 1; b(4)" "17"
# parameter shadowing: param names shadow global names like i, pi
run "fn.shadow_i"       "f(i) = i+1; f(2)"              "3"
run "fn.shadow_pi"      "g(pi) = pi+1; g(2)"            "3"
run "fn.shadow_cx_body" "f(i) = i+1i; f(1)"             "1 + i"

# ── Lambdas ───────────────────────────────────────────────────────────────────
section "LAMBDAS"
run "lambda.single"         "f = x -> x^2; f(4)"                    "16"
run "lambda.multi"          "f = (x, y) -> x + y; f(3, 4)"          "7"
run "lambda.multi_bare"     "f = x, y -> x * y; f(3, 4)"            "12"
run "lambda.inline"         "(x -> x^2)(5)"                           "25"
run "lambda.inline_call"    "(x -> x + 1)(9)"                         "10"
run "lambda.as_arg"         "sum(x -> x, 1, 10)"                      "55"
run "lambda.closure"        "a=10; f = x -> x + a; f(5)"            "15"
run "lambda.nested"         "f = x -> (y -> x + y); f(3)(4)"        "7"
run "lambda.in_sum"         "sum(x -> x^2, 1, 10)"                   "385"
run "lambda.in_prod"        "prod(x -> x, 1, 5)"                     "120"

# ── Blocks ────────────────────────────────────────────────────────────────────
section "BLOCKS"
run "block.simple"          "{x = 3; y = 4; x + y}"                 "7"
run "block.semicolon_out"   "{a = 2; b = 3; a * b}"                 "6"
run "block.isolation"       "x=99; {x = 1; x}"                      "1"
run "block.fn_in_block"     "{f(x) = x^2; f(5)}"                   "25"
run "block.as_expr"         "1 + {x=3; x*2}"                        "7"
run "block.multi_out"       "{a=1; b=2; (a, b)}"                    "[1, 2]"

# ── Comparisons ───────────────────────────────────────────────────────────────
section "COMPARISONS"
run "cmp.lt_true"   "3 < 5"     "1"
run "cmp.lt_false"  "5 < 3"     "0"
run "cmp.gt_true"   "5 > 3"     "1"
run "cmp.gt_false"  "3 > 5"     "0"
run "cmp.leq_eq"    "3 <= 3"    "1"
run "cmp.leq_lt"    "2 <= 3"    "1"
run "cmp.leq_false" "4 <= 3"    "0"
run "cmp.geq_eq"    "3 >= 3"    "1"
run "cmp.geq_gt"    "4 >= 3"    "1"
run "cmp.geq_false" "2 >= 3"    "0"
run "cmp.eq_true"   "3 == 3"    "1"
run "cmp.eq_false"  "3 == 4"    "0"
run "cmp.ne_true"   "3 != 4"    "1"
run "cmp.ne_false"  "3 != 3"    "0"
run "cmp.chained"   "1 < 2 < 3" "1"
run "cmp.chained2"  "3 < 2 < 5" "1"
run "cmp.and_op"    "1 & 1"     "1"
run "cmp.or_op"     "0 | 1"     "1"
run "cmp.and_false" "1 & 0"     "0"
run "cmp.or_false"  "0 | 0"     "0"
run "cmp.double_and" "1 && 1"   "1"
run "cmp.double_or"  "0 || 1"   "1"

# ── if ────────────────────────────────────────────────────────────────────────
section "IF"
run "if.true"       "if(1, 10, 20)"              "10"
run "if.false"      "if(0, 10, 20)"              "20"
run "if.cond_expr"  "if(3 > 2, 1, 0)"            "1"
run "if.nested"     "if(1, if(0, 1, 2), 3)"      "2"
run "if.lazy_false" "if(0, 1/0, 42)"             "42"
run "if.lazy_true"  "if(1, 42, 1/0)"             "42"
run "if.fn_branch"  "if(1, x -> x^2, x -> x)(5)" "25"

# ── Implicit multiplication ───────────────────────────────────────────────────
section "IMPLICIT MULTIPLICATION"
run_match "impl.num_const"   "2pi"       "6.28318"
run       "impl.num_paren"   "2(3+4)"    "14"
run       "impl.num_var"     "x=3; 2x"  "6"
run_match "impl.num_fn"      "3sin(pi/2)" "3"

# ── Factorial ─────────────────────────────────────────────────────────────────
section "FACTORIAL"
run "fact.zero"     "0!"    "1"
run "fact.one"      "1!"    "1"
run "fact.five"     "5!"    "120"
run "fact.ten"      "10!"   "3628800"
run "fact.fn"       "fact(7)" "5040"
run "fact.expr"     "(3+2)!" "120"

# ── Vectors (formerly tuples; (a,b,c) → 1D Tensor) ──────────────────────────
section "VECTORS"
run "tup.create"        "(1, 2, 3)"                         "[1, 2, 3]"
run "tup.index0"        "(10, 20, 30)[0]"                   "10"
run "tup.index1"        "(10, 20, 30)[1]"                   "20"
run "tup.index_last"    "(10, 20, 30)[2]"                   "30"
run "tup.neg_index"     "(1, 2, 3)[-1]"                     "3"
run "tup.neg_index2"    "(1, 2, 3)[-2]"                     "2"
run "tup.range_index"   "(1,2,3,4,5)[1..3]"                 "[2, 3, 4]"
run "tup.nested"        "shape((1,2,3))"                    "[3]"
run "tup.len"           "len((1,2,3,4,5))"                  "5"
run "tup.add"           "(1,2,3) + (4,5,6)"                 "[5, 7, 9]"
run "tup.sub"           "(5,6,7) - (1,2,3)"                 "[4, 4, 4]"
run "tup.scalar_mul"    "(1,2,3) * 3"                       "[3, 6, 9]"
run "tup.scalar_mul_l"  "3 * (1,2,3)"                       "[3, 6, 9]"
run "tup.scalar_div"    "(4,6,8) / 2"                       "[2, 3, 4]"
run "tup.scalar_pow"    "(1,2,3)^2"                         "[1, 4, 9]"
run "tup.scalar_add"    "(1,2,3) + 10"                      "[11, 12, 13]"
run "tup.eq_ew"         "(1,2,3) == (1,2,3)"                "[1, 1, 1]"
run "tup.neq_ew"        "(1,2,3) == (1,2,4)"                "[1, 1, 0]"
run "tup.all_eq"        "sum((1,2,3) == (1,2,3))"           "3"
run "tup.neg"           "-(1,2,3)"                          "[-1, -2, -3]"
run "tup.fn_apply"      "f(x)=x^2; f((1,2,3))"            "[1, 4, 9]"
run "tup.append"        "append((1,2,3), 4)"                "[1, 2, 3, 4]"
run "tup.concat"        "concat((1,2),(3,4))"               "[1, 2, 3, 4]"
run "tup.flatten"       "flatten(zeros(2,3))"               "[0, 0, 0, 0, 0, 0]"
run "tup.zip"           "shape(zip((1,2,3),(4,5,6)))"       "[3, 2]"
run "tup.zip_val"       "zip((1,2),(3,4))[0,0]"             "1"
run "tup.dot"           "dot((1,2,3),(4,5,6))"              "32"
run "tup.sort"          "sort((3,1,4,1,5,9))"               "[1, 1, 3, 4, 5, 9]"
run "tup.argmin"        "argmin((3,1,4,1,5))"               "1"
run "tup.argmax"        "argmax((3,1,4,1,5))"               "4"
# 2-D: argmax/argmin return [row, col]
run "tup.argmax_2d"     "argmax((1,8; 8,1))"                "[0, 1]"
run "tup.argmin_2d"     "argmin((1,8; 8,1))"                "[0, 0]"
# 3-D: returns [i, j, k]
run "tup.argmax_3d"     "argmax(tensor((i,j,k)->if(i+j+k==6,99,0), 2,3,4))"  "[1, 2, 3]"
run "tup.matmul"        "(1,2,3) @ (1,2,3)"                 "14"

# ── Aggregates on tuples ──────────────────────────────────────────────────────
section "AGGREGATES"
run "agg.sum_tuple"   "sum((1,2,3,4,5))"         "15"
run "agg.prod_tuple"  "prod((1,2,3,4))"           "24"
run "agg.sum_fn"      "sum(x -> x, 1, 100)"       "5050"
run "agg.prod_fn"     "prod(x -> x, 1, 10)"       "3628800"
run "agg.sum_x2"      "sum(x -> x^2, 1, 10)"      "385"
run "agg.map_sq"      "map(x -> x^2, (1,2,3,4))"  "[1, 4, 9, 16]"
run "agg.map_neg"     "map(x -> -x, (1,2,3))"     "[-1, -2, -3]"
run "agg.filter"      "filter(x -> x > 2, (1,2,3,4))" "[3, 4]"
run "agg.filter_none" "filter(x -> x > 9, (1,2,3))"   "[]"
run "agg.reduce_add"  "reduce((a,b) -> a+b, (1,2,3,4))" "10"
run "agg.reduce_mul"  "reduce((a,b) -> a*b, (1,2,3,4))" "24"
run "agg.reduce_max"  "reduce((a,b) -> if(a>b,a,b), (3,1,4,1,5))" "5"

# ── Statistics ────────────────────────────────────────────────────────────────
section "STATISTICS"
run "stat.mean"       "mean((1,2,3,4,5))"    "3"
run "stat.mean_even"  "mean((1,2,3,4))"      "2.5"
run "stat.median_odd" "median((3,1,4,1,5))"  "3"
run "stat.median_even" "median((1,2,3,4))"   "2.5"
run "stat.mode"       "mode((1,2,2,3))"      "2"
run "stat.min_tup"    "min((3,1,4,1,5))"     "1"
run "stat.max_tup"    "max((3,1,4,1,5))"     "5"
run "stat.sum_tup"    "sum((1,2,3,4,5))"     "15"
run_match "stat.std"  "std((2,4,4,4,5,5,7,9))" "^2"
run_match "stat.var"  "var((2,4,4,4,5,5,7,9))" "^4"

# ── Higher-order functions ────────────────────────────────────────────────────
section "HIGHER-ORDER"
run "ho.compose"          "compose(x -> x+1, x -> x^2)(3)"         "10"
run "ho.compose_builtins" "compose(sqrt, abs)(-9)"                  "3"
run "ho.partial_add"      "add5 = partial((x,y) -> x+y, 5); add5(3)" "8"
run "ho.partial_builtin"  "sq = partial(pow, 2); sq(10)"           "1024"
run "ho.map_partial"      "map(partial((x,y) -> x+y, 10), (1,2,3))"  "[11, 12, 13]"
run "ho.filter_partial"   "map(partial(pow,2), (1,2,3,4))"          "[2, 4, 8, 16]"

# ── Calculus ──────────────────────────────────────────────────────────────────
section "CALCULUS"
run_match "calc.integral_x2"   "integral(x -> x^2, 0, 1)"          "0.333"
run_match "calc.integral_sin"  "integral(x -> sin(x), 0, pi)"      "^2($|\\.0|\\.[0-9]*[0-9]$)|^1\\.999"
run_match "calc.integral_const" "integral(x -> 1, 0, 5)"           "^5"
run_match "calc.deriv_x3"      "deriv(x -> x^3, 2)"                "^12"
run_match "calc.deriv_sin"     "deriv(sin, 0)"                      "^1|^0\\.9999"
run_match "calc.deriv_cos"     "deriv(cos, pi/2)"                   "^-1|^-0\\.9999"
run_match "calc.deriv_custom_dx" "deriv(x -> x^2, 3, 1e-7)"        "^6|^5\\.9999"

# ── Complex numbers ───────────────────────────────────────────────────────────
section "COMPLEX"
run       "cx.i2"         "i^2"                   "-1"
run       "cx.i4"         "i^4"                   "1"
run       "cx.add"        "(1+2i) + (3+4i)"       "4 + 6i"
run       "cx.sub"        "(3+4i) - (1+2i)"       "2 + 2i"
run       "cx.mul"        "(1+i)*(1-i)"            "2"
run       "cx.mul2"       "(2+3i)*(1+i)"           "-1 + 5i"
run       "cx.div"        "(2+2i)/(1+i)"           "2"
run_match "cx.euler"      "exp(i*pi)"              "^-1|^-0\\.9999"
run       "cx.sqrt_neg"   "sqrt(-4)"               "2i"
run       "cx.sqrt_neg1"  "sqrt(-1)"               "i"
run       "cx.abs"        "abs(3+4i)"              "5"
run_match "cx.arg_i"      "arg(i)"                 "1.5707963"
run_match "cx.arg_neg"    "arg(-1)"                "3.14159"
run       "cx.conj"       "conj(3+4i)"             "3 - 4i"
run       "cx.re"         "re(3+4i)"               "3"
run       "cx.im"         "im(3+4i)"               "4"
run       "cx.ln_neg1"    "im(ln(-1))"             "3.141592653589793"
run_match "cx.pow_cx"     "i^i"                    "0.20787"

# ── Trig ─────────────────────────────────────────────────────────────────────
section "TRIG"
run       "trig.sin0"     "sin(0)"       "0"
run_match "trig.sin_pi6"  "sin(pi/6)"    "^0\\.5|^0\\.4999"
run       "trig.sin_pi2"  "sin(pi/2)"    "1"
run_match "trig.cos0"     "cos(0)"       "^1"
run_match "trig.cos_pi3"  "cos(pi/3)"    "^0\\.5|^0\\.4999"
run_match "trig.cos_pi"   "cos(pi)"      "^-1|^-0\\.9999"
run_match "trig.tan_pi4"  "tan(pi/4)"    "^1|^0\\.9999"
run_match "trig.asin"     "asin(1)"      "1.5707963"
run_match "trig.acos"     "acos(0)"      "1.5707963"
run_match "trig.atan"     "atan(1)"      "0.7853981"
run       "trig.atan2_11" "atan2(1,1)"   "$(python3 -c 'import math; print(math.atan2(1,1))')"
run_match "trig.sinh0"    "sinh(0)"      "^0"
run_match "trig.cosh0"    "cosh(0)"      "^1"
run_match "trig.tanh0"    "tanh(0)"      "^0"
run_match "trig.sec0"     "sec(0)"       "^1"
run_match "trig.csc_pi2"  "csc(pi/2)"    "^1"
run_match "trig.cot_pi4"  "cot(pi/4)"    "^0\\.9999|^1"
run       "trig.deg"      "deg(pi)"      "180"
run_match "trig.rad"      "rad(180)"     "3.14159"
run_match "trig.sinc0"    "sinc(0)"      "^1"

# ── Algebra functions ─────────────────────────────────────────────────────────
section "ALGEBRA"
run "alg.sqrt4"     "sqrt(4)"           "2"
run "alg.sqrt2"     "sqrt(2)"           "1.4142135623730951"
run "alg.cbrt8"     "cbrt(8)"           "2"
run "alg.cbrt27"    "cbrt(27)"          "3"
run "alg.abs_pos"   "abs(5)"            "5"
run "alg.abs_neg"   "abs(-5)"           "5"
run "alg.sign_pos"  "sign(7)"           "1"
run "alg.sign_neg"  "sign(-7)"          "-1"
run "alg.sign_zero" "sign(0)"           "1"
run "alg.floor"     "floor(3.7)"        "3"
run "alg.floor_neg" "floor(-3.2)"       "-4"
run "alg.ceil"      "ceil(3.2)"         "4"
run "alg.ceil_neg"  "ceil(-3.7)"        "-3"
run "alg.round_up"  "round(3.5)"        "4"
run "alg.round_dn"  "round(3.4)"        "3"
run "alg.round_n"   "round(3.14159, 2)" "3.14"
run "alg.trunc_pos" "trunc(3.9)"        "3"
run "alg.trunc_neg" "trunc(-3.9)"       "-3"
run_match "alg.frac" "frac(3.75)"       "0.75"
run "alg.exp1"      "ln(exp(1))"        "1"
run "alg.log10"     "log10(1000)"       "3"
run "alg.log2"      "log2(8)"           "3"
run "alg.log_base"  "log(8,2)"          "3"
run "alg.pow_int"   "pow(2,10)"         "1024"
run "alg.min2"      "min(3,7)"          "3"
run "alg.max2"      "max(3,7)"          "7"
run "alg.hypot"     "hypot(3,4)"        "5"
run "alg.heaviside_neg"  "heaviside(-1)"  "0"
run "alg.heaviside_pos"  "heaviside(1)"   "1"
run "alg.heaviside_zero" "heaviside(0)"   "0.5"
run "alg.expm1"     "expm1(0)"          "0"
run_match "alg.expm1_1" "expm1(1)"      "1.71828"

# ── Number theory ─────────────────────────────────────────────────────────────
section "NUMBER THEORY"
run "nt.gcd"          "gcd(12,18)"    "6"
run "nt.gcd_prime"    "gcd(7,13)"     "1"
run "nt.lcm"          "lcm(4,6)"      "12"
run "nt.lcm_prime"    "lcm(5,7)"      "35"
run "nt.fact0"        "fact(0)"       "1"
run "nt.fact1"        "fact(1)"       "1"
run "nt.fact5"        "fact(5)"       "120"
run "nt.fact10"       "10!"           "3628800"
run "nt.delta0"       "delta(0)"      "1"
run "nt.delta_nz"     "delta(5)"      "0"

# ── Bitwise ───────────────────────────────────────────────────────────────────
section "BITWISE"
run "bit.and"   "and(12,10)"   "8"
run "bit.or"    "or(12,10)"    "14"
run "bit.xor"   "xor(12,10)"   "6"
run "bit.shl"   "shl(1,8)"     "256"
run "bit.shr"   "shr(256,4)"   "16"
run "bit.nor"   "nor(0,0)"     "1"
run "bit.nor2"  "nor(1,0)"     "0"
run "bit.xnor"  "xnor(5,5)"    "1"
run "bit.xnor2" "xnor(5,3)"    "0"

# ── Special functions ─────────────────────────────────────────────────────────
section "SPECIAL FUNCTIONS"
run_match "spec.erf0"    "erf(0)"      "^0"
run_match "spec.erf1"    "erf(1)"      "0.84270"
run_match "spec.erfc0"   "erfc(0)"     "^1"
run_match "spec.erfc1"   "erfc(1)"     "0.15729"
run       "spec.sinc0"   "sinc(0)"     "1"
run_match "spec.sinc1"   "sinc(1)"     "0.84147"
run_match "spec.j0_0"    "j0(0)"       "^1"
run_match "spec.j0_z"    "j0(2.4048)"  "^0\\.00|^-0\\.00"
run_match "spec.j1_0"    "j1(0)"       "^0"
run_match "spec.sech0"   "sech(0)"     "^1"
run_match "spec.csch1"   "csch(1)"     "0.8509"

# ── linspace / range ─────────────────────────────────────────────────────────
section "LINSPACE / RANGE"
run "ls.basic"      "linspace(0,1,3)"       "[0, 0.5, 1]"
run "ls.one"        "linspace(0,10,1)"      "[0]"
run "ls.five"       "linspace(0,4,5)"       "[0, 1, 2, 3, 4]"
run "range.basic"   "range(0,5)"            "[0, 1, 2, 3, 4]"
run "range.zero"    "range(0,0)"            "[]"
run "range.offset"  "range(3,7)"            "[3, 4, 5, 6]"

# ── rand ─────────────────────────────────────────────────────────────────────
section "RAND"
run_ok "rand.scalar"  "rand()"
run_ok "rand.vec"     "rand(10)"
run_ok "rand.mat"     "rand(3, 4)"
run    "rand.shape"   "shape(rand(3, 4))"     "[3, 4]"
run    "rand.1d_len"  "len(rand(7))"          "7"

# ── FFT ──────────────────────────────────────────────────────────────────────
section "FFT"
run_match "fft.dc"        "re(fft((1,1,1,1))[0])"          "^4"
run_match "fft.nyquist"   "abs(fft((1,-1,1,-1))[2])"       "^4"
run_match "fft.roundtrip" "re(ifft(fft((1,2,3,4)))[0])"    "^1"
run_match "fft.roundtrip1" "re(ifft(fft((1,2,3,4)))[1])"   "^2"
run_match "fft.zero_ac"   "re(fft((1,-1,1,-1))[0])"        "^0"

# ── Tensor constructors ───────────────────────────────────────────────────────
section "TENSORS - CONSTRUCTORS"
run "ten.zeros_2d"   "zeros(2,3)"                       "⎡ 0  0  0 ⎤ ⎣ 0  0  0 ⎦"
run "ten.ones_2d"    "ones(2,2)"                        "⎡ 1  1 ⎤ ⎣ 1  1 ⎦"
run "ten.eye2"       "eye(2)"                           "⎡ 1  0 ⎤ ⎣ 0  1 ⎦"
run "ten.eye3_trace" "trace(eye(3))"                    "3"
run "ten.diag"       "diag((2,3,4))"                    "⎡ 2  0  0 ⎤ ⎢ 0  3  0 ⎥ ⎣ 0  0  4 ⎦"
run "ten.diag_trace" "trace(diag((1,2,3)))"             "6"
run "ten.matrix_fn"  "shape(matrix((i,j)->0, 3, 4))"      "[3, 4]"
run "ten.matrix_val" "matrix((i,j)->i*2+j, 2, 2)[0,1]"   "1"
run "ten.zeros_1d"   "zeros(4)"                         "[0, 0, 0, 0]"
run "ten.ones_1d"    "ones(3)"                          "[1, 1, 1]"
run "ten.zeros_3d"   "shape(zeros(2,3,4))"              "[2, 3, 4]"

# ── Tensor shape queries ──────────────────────────────────────────────────────
section "TENSORS - SHAPE"
run "ten.shape_2d"   "shape(eye(3))"         "[3, 3]"
run "ten.rows"       "rows(eye(4))"          "4"
run "ten.cols"       "cols(zeros(3,5))"      "5"
run "ten.len_1d"     "len(zeros(7))"         "7"
run "ten.len_2d"     "len(eye(4))"           "4"
run "ten.shape_1d"   "shape(zeros(5))"       "[5]"

# ── Tensor indexing ───────────────────────────────────────────────────────────
section "TENSORS - INDEXING"
run "ten.idx_1d"     "zeros(3)[1]"                         "0"
run "ten.idx_2d"     "eye(3)[1,1]"                         "1"
run "ten.idx_2d_off" "eye(3)[0,1]"                         "0"
run "ten.idx_row"    "matrix((i,j)->i*3+j, 2, 3)[0]"         "[0, 1, 2]"
run "ten.idx_row2"   "matrix((i,j)->i*3+j, 2, 3)[1]"         "[3, 4, 5]"
run "ten.neg_idx_1d" "ones(4)[-1]"                         "1"
run "ten.row_fn"     "row(eye(3), 1)"                      "[0, 1, 0]"
run "ten.col_fn"     "col(eye(3), 2)"                      "[0, 0, 1]"

# ── Tensor arithmetic ─────────────────────────────────────────────────────────
section "TENSORS - ARITHMETIC"
run "ten.add_scalar"    "ones(2,2) + 1"                    "⎡ 2  2 ⎤ ⎣ 2  2 ⎦"
run "ten.sub_scalar"    "ones(2,2) * 3 - 1"               "⎡ 2  2 ⎤ ⎣ 2  2 ⎦"
run "ten.mul_scalar"    "eye(3) * 5"                       "⎡ 5  0  0 ⎤ ⎢ 0  5  0 ⎥ ⎣ 0  0  5 ⎦"
run "ten.div_scalar"    "ones(2,2) * 4 / 2"               "⎡ 2  2 ⎤ ⎣ 2  2 ⎦"
run "ten.add_tensor"    "eye(2) + eye(2)"                  "⎡ 2  0 ⎤ ⎣ 0  2 ⎦"
run "ten.sub_tensor"    "eye(2) - eye(2)"                  "⎡ 0  0 ⎤ ⎣ 0  0 ⎦"
run "ten.neg_tensor"    "-eye(2)"                          "⎡ -1   0 ⎤ ⎣  0  -1 ⎦"
run "ten.pow_scalar"    "ones(2,2) * 3 ^ 2"               "⎡ 9  9 ⎤ ⎣ 9  9 ⎦"

# ── Tensor operations ─────────────────────────────────────────────────────────
section "TENSORS - OPERATIONS"
run "ten.transpose"     "transpose(matrix((i,j)->i*3+j, 2, 3))" "⎡ 0  3 ⎤ ⎢ 1  4 ⎥ ⎣ 2  5 ⎦"
run "ten.transpose_sq"  "transpose(eye(3))"                   "⎡ 1  0  0 ⎤ ⎢ 0  1  0 ⎥ ⎣ 0  0  1 ⎦"
run "ten.trace_eye"     "trace(eye(5))"                       "5"
run "ten.norm_eye3"     "norm(eye(3))"                        "1.7320508075688772"
run "ten.norm_ones"     "norm(ones(4))"                       "2"
run "ten.matmul_id"     "matmul(eye(2), eye(2))"              "⎡ 1  0 ⎤ ⎣ 0  1 ⎦"
run "ten.matmul_basic"  "matmul(matrix((i,j)->j+1, 1, 2), matrix((i,j)->i+1, 2, 1))" "⎡ 5 ⎤"
run "ten.matmul_2x2"    "trace(matmul(eye(3), ones(3,3)))"    "3"
run "ten.flatten"       "flatten(eye(2))"                     "[1, 0, 0, 1]"
run "ten.sum_tensor"    "sum(ones(3,3))"                      "9"
run "ten.prod_tensor"   "prod(ones(2,2) * 2)"                 "16"
run "ten.map_tensor"    "map(x -> x*2, eye(2))"               "⎡ 2  0 ⎤ ⎣ 0  2 ⎦"
run "ten.unary_sin"     "sin(zeros(3))"                       "[0, 0, 0]"
run "ten.unary_exp"     "sum(exp(zeros(2,2)))"                "4"
run "ten.dot_1d"        "dot(ones(3), ones(3))"               "3"

# ── Tensor error cases ────────────────────────────────────────────────────────
section "TENSORS - ERRORS"
run_err "ten.err.shape_mismatch" "eye(2) + eye(3)"
run_err "ten.err.matmul_bad"     "matmul(eye(2), eye(3))"
run_err "ten.err.rows_1d"        "rows(zeros(5))"
run_err "ten.err.cols_1d"        "cols(zeros(5))"
run_err "ten.err.idx_oob"        "eye(3)[5,5]"

# ── Dot product ───────────────────────────────────────────────────────────────
section "DOT"
run "dot.tup"    "dot((1,2,3),(4,5,6))"  "32"
run "dot.unit"   "dot((1,0),(0,1))"      "0"
run "dot.self"   "dot((3,4),(3,4))"      "25"
run "dot.ten_1d" "dot(ones(3),ones(3))"  "3"

# ── Norm ──────────────────────────────────────────────────────────────────────
section "NORM"
run "norm.tup"    "norm((3,4))"       "5"
run "norm.tup3"   "norm((1,0,0))"     "1"
run "norm.tensor" "norm(eye(3))"      "1.7320508075688772"

# ── Comparison functions ──────────────────────────────────────────────────────
section "COMPARISON FUNCTIONS"
run "cmpfn.lt"    "lt(2,3)"    "1"
run "cmpfn.lt2"   "lt(3,2)"    "0"
run "cmpfn.leq"   "leq(3,3)"   "1"
run "cmpfn.gt"    "gt(5,3)"    "1"
run "cmpfn.geq"   "geq(3,3)"   "1"
run "cmpfn.eq"    "eq(4,4)"    "1"
run "cmpfn.eq2"   "eq(4,5)"    "0"
run "cmpfn.neq"   "neq(4,5)"   "1"
run "cmpfn.neq2"  "neq(4,4)"   "0"
run "cmpfn.map"   "map(partial(lt,3), (1,2,3,4,5))" "[0, 0, 0, 1, 1]"

# ── gaussian ──────────────────────────────────────────────────────────────────
section "GAUSSIAN"
run_match "gauss.peak"   "gaussian(0,0,1)"   "0.39894"
run_match "gauss.cdf_0"  "gaussian_cdf(0,0,1)" "^0\\.5"
run_match "gauss.cdf_inf" "gaussian_cdf(100,0,1)" "^1"

# ── eps (Levi-Civita) ─────────────────────────────────────────────────────────
section "EPSILON"
run "eps.123"   "eps(1,2,3)"  "1"
run "eps.132"   "eps(1,3,2)"  "-1"
run "eps.112"   "eps(1,1,2)"  "0"
run "eps.213"   "eps(2,1,3)"  "-1"

# ── Scientific notation ───────────────────────────────────────────────────────
section "SCIENTIFIC NOTATION"
run       "sci.e3"    "1e3"    "1000"
run       "sci.e0"    "1e0"    "1"
run_match "sci.neg"   "1.5e-2" "0.015"
run       "sci.big"   "1e10"   "10000000000"

# ── Edge cases and error handling ─────────────────────────────────────────────
section "EDGE CASES"
run       "edge.inf_add"    "inf + 1"         "inf"
run       "edge.inf_mul"    "inf * 2"         "inf"
run_match "edge.nan"        "0 * inf"         "NaN|nan"
run       "edge.neg_pow"    "2^-2"            "0.25"
run       "edge.zero_pow"   "0^0"             "1"
run       "edge.large_fact" "20!"             "2432902008176640000"
run       "edge.paren_expr" "(((3 + 4)))"     "7"
run_err   "edge.undef_var"  "xyz"
run_err   "edge.bad_index"  "(1,2,3)[5]"

# ── Negative-index bounds checking (regression: must error, not wrap to 0) ────
section "NEGATIVE INDEX BOUNDS"
run       "negidx.last"        "(10,20,30)[-1]"        "30"
run       "negidx.first"       "(10,20,30)[-3]"        "10"
run_err   "negidx.under1"      "(10,20,30)[-4]"
run_err   "negidx.under_big"   "(10,20,30)[-100]"
run_err   "negidx.mat_row"     "(1,2;3,4)[-3]"
run_err   "negidx.mat_col"     "(1,2,3;4,5,6)[0,-5]"
run_err   "negidx.tuple"       "{t=(1,(2,3),4); t[-9]}"
run       "negidx.slice_ok"    "(1,2,3,4,5)[1..3]"     "[2, 3, 4]"

# ── Recursion depth guard (regression: catchable error, not stack overflow) ───
section "RECURSION GUARD"
run       "rec.shallow_ok"   "f(n)=if(n<=0,0,f(n-1)+1); f(5000)"           "5000"
run       "rec.deep_ok"      "f(n)=if(n<=0,0,f(n-1)+1); f(90000)"          "90000"
run_match "rec.over_limit"   "f(n)=if(n<=0,0,f(n-1)+1); f(500000)"         "recursion limit exceeded"

section "ITERATE / SCAN"
# iterate(f, x0, n) — apply f n times, flat loop
run       "iter.scalar"      "iterate(x->2*x, 1, 10)"                      "1024"
run       "iter.zero"        "iterate(x->x+1, 5, 0)"                       "5"
run       "iter.big"         "iterate(x->x+1, 0, 1000000)"                 "1000000"
run       "iter.vector"      "iterate(v->(v[1], -v[0]), (1,0), 4)"         "[1, 0]"
run       "iter.complex"     "iterate(z->z*i, 1, 2)"                       "-1"
run_err   "iter.neg"         "iterate(x->x, 0, -1)"
run_err   "iter.arity"       "iterate(x->x, 0)"
# scan(f, x0, n) — the whole orbit [x0, f(x0), …], stacked
run       "scan.scalar"      "scan(x->2*x, 1, 4)"                          "[1, 2, 4, 8, 16]"
run       "scan.zero"        "scan(x->x+1, 7, 0)"                          "[7]"
run       "scan.shape_vec"   "shape(scan(v->(v[1], -v[0]), (1,0), 3))"     "[4, 2]"
run       "scan.vec_first"   "scan(v->(v[1], -v[0]), (1,0), 3)[0]"         "[1, 0]"
run       "scan.complex"     "scan(z->z*i, 1, 4)"                          "[1, i, -1, -i, 1]"
run       "scan.len"         "len(scan(x->x+1, 0, 100))"                   "101"
run_err   "scan.neg"         "scan(x->x, 0, -1)"
# structured tuple state (q,p with vector q,p) → componentwise tuple of stacks
run       "scan.struct.shapeQ" "shape(scan(s->(s[0]+s[1], s[1]), ((1,2),(3,4)), 2)[0])" "[3, 2]"
run       "scan.struct.shapeP" "shape(scan(s->(s[0]+s[1], s[1]), ((1,2),(3,4)), 2)[1])" "[3, 2]"
run       "scan.struct.rowQ"   "scan(s->(s[0]+s[1], s[1]), ((1,2),(3,4)), 2)[0][2]"     "[7, 10]"
run       "scan.struct.rowP"   "scan(s->(s[0]+s[1], s[1]), ((1,2),(3,4)), 2)[1][0]"     "[3, 4]"
run_err   "scan.struct.bad"    "scan(s->5, ((1,2),(3,4)), 2)"

section "CUMSUM / CUMPROD / DIFF"
run       "cumsum.basic"     "cumsum([1,2,3,4])"                           "[1, 3, 6, 10]"
run       "cumsum.tuple"     "cumsum((1,2,3,4))"                           "[1, 3, 6, 10]"
run       "cumprod.basic"    "cumprod([1,2,3,4])"                          "[1, 2, 6, 24]"
run       "diff.basic"       "diff([1,4,9,16])"                            "[3, 5, 7]"
run       "diff.single"      "diff([5])"                                   "[]"
run       "diff.cumsum_inv"  "diff(cumsum([3,1,4,1,5]))"                   "[1, 4, 1, 5]"
run_err   "cumsum.2d"        "cumsum((1,2;3,4))"

section "SINGLETON LITERAL"
run       "single.value"     "(5,)"                                        "[5]"
run       "single.shape"     "shape((5,))"                                 "[1]"
run       "single.append"    "append((5,), 6)"                             "[5, 6]"
run       "single.expr"      "(2+3,)"                                      "[5]"
run       "single.no_comma"  "(5)"                                         "5"
run       "single.pair"      "(1,2,)"                                      "[1, 2]"

section "GENERALIZED STACKING"
run       "vstack.vec_vec"   "shape(vstack((1,2,3),(4,5,6)))"              "[2, 3]"
run       "vstack.vec_mat"   "shape(vstack((1,2;3,4),(5,6)))"              "[3, 2]"
run       "vstack.mat_vec"   "shape(vstack((5,6),(1,2;3,4)))"              "[3, 2]"
run       "vstack.scalars"   "shape(vstack(1,2))"                          "[2, 1]"
run_err   "vstack.colmismatch" "vstack((1,2,3),(4,5))"
run       "hstack.vec_vec"   "shape(hstack((1,2),(3,4)))"                  "[2, 2]"
run       "hstack.val"       "hstack((1,2),(3,4))[0,0]"                    "1"
run       "append.scalar"    "append(1, 2)"                               "[1, 2]"
run       "concat.scalars"   "concat(1, 2)"                               "[1, 2]"
run       "concat.empty"     "concat(zeros(0), [1,2])"                     "[1, 2]"
run       "concat.mixed"     "concat(1, [2,3])"                            "[1, 2, 3]"

# ── Chained / compound expressions ───────────────────────────────────────────
section "COMPOUND EXPRESSIONS"
run "comp.fn_chain"       "f(x)=x+1; g(x)=x^2; g(f(3))"         "16"
run "comp.let_in_arg"     "a=3; b=4; sqrt(a^2 + b^2)"            "5"
run "comp.block_in_fn"    "f(x) = {y=x^2; y+1}; f(4)"           "17"
run "comp.tuple_in_fn"    "f(t) = t[0] + t[1]; f((3,4))"         "7"
run "comp.map_then_sum"   "sum(map(x -> x^2, (1,2,3,4)))"         "30"
run "comp.filter_sum"     "sum(filter(x -> x > 2, (1,2,3,4,5)))"  "12"
run "comp.compose_chain"  "f=x->x+1; g=x->x*2; compose(f,g)(5)" "11"
run "comp.lambda_in_if"   "(if(1>0, x->x^2, x->x))(5)"           "25"

# ── tensor() constructor ──────────────────────────────────────────────────────
section "TENSOR CONSTRUCTOR"
run "tc.1d_shape"    "shape(tensor(i -> i, 5))"                    "[5]"
run "tc.1d_val"      "tensor(i -> i^2, 4)[2]"                      "4"
run "tc.1d_sum"      "sum(tensor(i -> 1, 6))"                      "6"
run "tc.2d_shape"    "shape(tensor((i,j) -> 0, 3, 4))"             "[3, 4]"
run "tc.2d_val"      "tensor((i,j) -> i*3+j, 2, 3)[1,2]"          "5"
run "tc.2d_eq_mat"   "tensor((i,j) -> i*3+j, 2, 3)[0]"            "[0, 1, 2]"
run "tc.3d_shape"    "shape(tensor((i,j,k) -> 0, 2, 3, 4))"        "[2, 3, 4]"
run "tc.3d_sum"      "sum(tensor((i,j,k) -> 1, 2, 3, 4))"          "24"
run "tc.eye_via_ten" "trace(tensor((i,j) -> if(i==j,1,0), 4, 4))" "4"
run "tc.matrix_f1"   "shape(matrix((i,j) -> 0, 3, 4))"            "[3, 4]"
run "tc.matrix_f2"   "matrix((i,j) -> i+j, 2, 2)[1,1]"            "2"

# ── Tensor literal syntax (1,2;3,4) ──────────────────────────────────────────
section "TENSOR LITERALS"
run "lit.2x2"        "(1,2; 3,4)"                         "⎡ 1  2 ⎤ ⎣ 3  4 ⎦"
run "lit.2x3"        "(1,2,3; 4,5,6)"                     "⎡ 1  2  3 ⎤ ⎣ 4  5  6 ⎦"
run "lit.shape"      "shape((1,2; 3,4))"                  "[2, 2]"
run "lit.index"      "(1,2; 3,4)[0,1]"                    "2"
run "lit.index2"     "(1,2; 3,4)[1,0]"                    "3"
run "lit.det"        "det((1,2; 3,4))"                    "-2"
run "lit.assign"     "M = (1,2; 3,4); M[1,1]"           "4"
run "lit.1col"       "shape((1; 2; 3))"                   "[3, 1]"
run "lit.arith"      "(1,0; 0,1) + (0,1; 1,0)"            "⎡ 1  1 ⎤ ⎣ 1  1 ⎦"

# ── @ matmul operator ─────────────────────────────────────────────────────────
section "@ OPERATOR"
run "at.eye"         "eye(2) @ eye(2)"                    "⎡ 1  0 ⎤ ⎣ 0  1 ⎦"
run "at.basic"       "(1,2; 3,4) @ (1,0; 0,1)"            "⎡ 1  2 ⎤ ⎣ 3  4 ⎦"
run "at.chain"       "trace(eye(3) @ ones(3,3))"          "3"
run "at.rect"        "rows((ones(2,3) @ ones(3,4)))"      "2"
run "at.cols"        "cols((ones(2,3) @ ones(3,4)))"      "4"

# ── det / inv ─────────────────────────────────────────────────────────────────
section "DET / INV"
run "det.eye2"       "det(eye(2))"                        "1"
run "det.eye3"       "det(eye(3))"                        "1"
run "det.2x2"        "det((2,0; 0,3))"                    "6"
run "det.singular"   "det((1,2; 2,4))"                    "0"
run_match "det.3x3"  "det((1,2,3; 4,5,6; 7,8,9))"        "^0|-?0\\.000"
run "inv.eye2"       "inv(eye(2))"                        "⎡ 1  0 ⎤ ⎣ 0  1 ⎦"
run "inv.diag"       "trace(inv((2,0; 0,4)))"             "0.75"
run_match "inv.roundtrip"  "trace((1,2; 3,4) @ inv((1,2; 3,4)))" "^2|^1\.999"
run_err "inv.singular" "inv((1,2; 2,4))"

# ── solve ─────────────────────────────────────────────────────────────────────
section "SOLVE"
run "solve.eye"      "solve(eye(2), (1,2))"               "[1, 2]"
run "solve.2x2"      "solve((2,0; 0,3), (4,9))"           "[2, 3]"
run_match "solve.3x3" "solve((1,2,3; 0,1,4; 5,6,0), (1,2,0))[0]" "^12|^11\\.9"
run_err "solve.singular" "solve((1,2; 2,4), (1,2))"

# ── eigenvalues / eigenvectors ────────────────────────────────────────────────
section "EIGENVALUES / EIGENVECTORS"
run     "eigvals.eye2"       "eigvals(eye(2))"                            "[1, 1]"
run     "eigvals.eye3"       "eigvals(eye(3))"                            "[1, 1, 1]"
run     "eigvals.diag"       "eigvals((4,0; 0,3))"                        "[4, 3]"
run     "eigvals.diag3"      "eigvals((5,0,0; 0,3,0; 0,0,1))"            "[5, 3, 1]"
run_match "eigvals.trace"    "sum(eigvals((4,1; 1,3)))"                   "^7|^6\.9"
run_match "eigvals.det"      "prod(eigvals((4,1; 1,3)))"                  "^11|^10\.9"
run     "eig_top.diag.val"   "eig_top((9,0; 0,1))[0]"                    "9"
run     "eig_bot.diag.val"   "eig_bot((9,0; 0,1))[0]"                    "1"
run_match "eig_top.val"      "eig_top((4,1; 1,3))[0]"                    "^4\\.6"
run_match "eig_bot.val"      "eig_bot((4,1; 1,3))[0]"                    "^2\\.3"
run_match "eig_top.residual" "norm((4,1; 1,3) @ eig_top((4,1; 1,3))[1] - eig_top((4,1; 1,3))[0] * eig_top((4,1; 1,3))[1])" "^0|^[0-9]e-"
run_match "eig_bot.residual" "norm((4,1; 1,3) @ eig_bot((4,1; 1,3))[1] - eig_bot((4,1; 1,3))[0] * eig_bot((4,1; 1,3))[1])" "^0|^[0-9]e-"
run_match "eig.ortho"        "dot(col(eig((4,1; 1,3))[1], 0), col(eig((4,1; 1,3))[1], 1))" "^0|-?0\\.0000"
run     "eig.consistency"    "norm(eig((4,1; 1,3))[0] - eigvals((4,1; 1,3)))"  "0"
run_err "eigvals.nonsquare"  "eigvals((1,2; 3,4; 5,6))"
run_err "eig.nonsquare"      "eig((1,2,3; 4,5,6))"
run_err "eig_top.nonsquare"  "eig_top((1,2,3; 4,5,6))"
run_err "eig_bot.nonsquare"  "eig_bot((1,2,3; 4,5,6))"

# ── QR decomposition ──────────────────────────────────────────────────────────
section "QR DECOMPOSITION"
run_match "qr.roundtrip"   "norm(qr((3,1; 1,2))[0] @ qr((3,1; 1,2))[1] - (3,1; 1,2))"           "^0|^[0-9]e-"
run_match "qr.orthogonal"  "norm(transpose(qr((3,1; 1,2))[0]) @ qr((3,1; 1,2))[0] - eye(2))"    "^0|^[0-9]e-"
run       "qr.q_shape"     "shape(qr((3,1; 1,2))[0])"                                            "[2, 2]"
run       "qr.r_shape"     "shape(qr((3,1; 1,2))[1])"                                            "[2, 2]"
run       "qr.rect_q"      "shape(qr((1,2; 3,4; 5,6))[0])"                                       "[3, 3]"
run       "qr.rect_r"      "shape(qr((1,2; 3,4; 5,6))[1])"                                       "[3, 2]"
run_match "qr.rect_roundtrip" "norm(qr((1,2; 3,4; 5,6))[0] @ qr((1,2; 3,4; 5,6))[1] - (1,2; 3,4; 5,6))" "^0|^[0-9]e-"
run_err   "qr.fat"         "qr((1,2,3; 4,5,6))"

# ── diagonalize ───────────────────────────────────────────────────────────────
section "DIAGONALIZE"
run_match "diag.roundtrip" "norm(diagonalize((4,1; 1,3))[0] @ diagonalize((4,1; 1,3))[1] @ diagonalize((4,1; 1,3))[2] - (4,1; 1,3))" "^0|^[0-9]e-"
run_match "diag.d_is_diag" "diagonalize((4,1; 1,3))[1][0,1]"                                     "^0|-?0\\.0000"
run_match "diag.d_eigs"    "diagonalize((4,1; 1,3))[1][0,0]"                                     "^4\\.6"
run_err   "diag.nonsquare" "diagonalize((1,2,3; 4,5,6))"

# ── hstack / vstack / tomat ───────────────────────────────────────────────────
section "HSTACK / VSTACK / TOMAT"
run "hst.shape"      "shape(hstack(eye(2), eye(2)))"      "[2, 4]"
run "hst.val"        "hstack(eye(2), eye(2))[0,2]"        "1"
run "vst.shape"      "shape(vstack(eye(2), eye(2)))"      "[4, 2]"
run "vst.val"        "vstack(eye(2), eye(2))[2,0]"        "1"
run "tomat.basic"    "shape(tomat((1,2,3,4), 2, 2))"      "[2, 2]"
run "tomat.val"      "tomat((1,2,3,4), 2, 2)[0,1]"        "2"
run_err "tomat.bad"  "tomat((1,2,3), 2, 2)"

# ── lingrid ───────────────────────────────────────────────────────────────────
section "LINGRID"
run "lg.1d_scalar"   "lingrid(0, 1, 3, x -> x^2)"                    "[0, 0.25, 1]"
run "lg.1d_tuple"    "lingrid((0,),(1,),(3,), x -> x^2)"              "[0, 0.25, 1]"
run "lg.1d_len"      "len(lingrid(0, 4, 5, x -> x))"                  "5"
run "lg.2d_shape"    "shape(lingrid((0,0),(1,1),(3,3),(x,y)->x+y))"   "[3, 3]"
run "lg.2d_corner"   "lingrid((0,0),(1,1),(3,3),(x,y)->x+y)[0,0]"    "0"
run "lg.2d_mid"      "lingrid((0,0),(1,1),(3,3),(x,y)->x+y)[1,1]"    "1"
run "lg.2d_far"      "lingrid((0,0),(1,1),(3,3),(x,y)->x+y)[2,2]"    "2"
run "lg.2d_sum"      "sum(lingrid((0,0),(1,1),(2,2),(x,y)->1))"       "4"
run "lg.3d_shape"    "shape(lingrid((0,0,0),(1,1,1),(3,3,3),(x,y,z)->x+y+z))" "[3, 3, 3]"
run "lg.3d_origin"   "lingrid((0,0,0),(1,1,1),(3,3,3),(x,y,z)->x+y+z)[0,0,0]" "0"
run "lg.3d_far"      "lingrid((0,0,0),(1,1,1),(3,3,3),(x,y,z)->x+y+z)[2,2,2]" "3"
# vector-valued f: output shape = grid_shape ++ value_shape
run "lg.vec_shape"   "shape(lingrid((-2,-2),(2,2),(5,5),(x,y)->(x,y)))"   "[5, 5, 2]"
run "lg.vec_x"       "lingrid((-1,-1),(1,1),(3,3),(x,y)->(x,y))[.., .., 0][0,0]" "-1"
run "lg.vec_y"       "lingrid((-1,-1),(1,1),(3,3),(x,y)->(x,y))[.., .., 1][2,2]" "1"
run "lg.vec_xgrid"   "shape(lingrid((-1,-1),(1,1),(3,3),(x,y)->(x,y))[.., .., 0])" "[3, 3]"
run "lg.1d_vec"      "shape(lingrid(0, 1, 4, x -> (sin(x), cos(x))))"     "[4, 2]"
run "lg.1d_vec_sum"  "sum(lingrid(0,1,4, x -> (0.0, 1.0)), 0)"           "[0, 4]"
run "lg.ten_val"     "shape(lingrid((0,0),(1,1),(3,3),(x,y)->eye(2)))"    "[3, 3, 2, 2]"

# ── 2D slice indexing ─────────────────────────────────────────────────────────
section "TENSOR SLICING"
run "sl.scalar_scalar" "eye(3)[1,1]"                      "1"
run "sl.row_slice"     "eye(3)[0, 0..1]"                  "[1, 0]"
run "sl.col_slice"     "eye(3)[0..1, 0]"                  "[1, 0]"
run "sl.submat_shape"  "shape(eye(3)[0..1, 0..1])"        "[2, 2]"
run "sl.submat_trace"  "trace(eye(4)[0..1, 0..1])"        "2"
run "sl.row1_slice"    "(1,2,3; 4,5,6)[1, 0..2]"          "[4, 5, 6]"
run "sl.col1_slice"    "(1,2,3; 4,5,6; 7,8,9)[0..2, 1]"  "[2, 5, 8]"
run "sl.1d_range"      "tensor(i -> i, 6)[2..4]"           "[2, 3, 4]"
run "sl.3d_scalar"     "tensor((i,j,k)->i*4+j*2+k, 2,2,2)[1,0,1]" "5"
run "sl.3d_mixed"      "shape(zeros(3,4,5)[0, 0..2, 1..3])" "[3, 3]"
run "sl.3d_slice"      "sum(ones(2,3,4)[0, 0..2, 0..1])"   "6"
# "all" slices with ..
run "sl.all_col"       "(1,2,3; 4,5,6; 7,8,9)[.., 0]"     "[1, 4, 7]"
run "sl.all_row"       "(1,2,3; 4,5,6; 7,8,9)[0, ..]"     "[1, 2, 3]"
run "sl.all_shape"     "shape((1,2,3; 4,5,6)[.., ..])"     "[2, 3]"
run "sl.open_lo"       "(1,2,3; 4,5,6; 7,8,9)[1.., 0]"    "[4, 7]"
run "sl.open_hi"       "(1,2,3; 4,5,6; 7,8,9)[..1, 0]"    "[1, 4]"
run "sl.1d_all"        "tensor(i->i+1, 5)[..]"             "[1, 2, 3, 4, 5]"
run "sl.1d_open_lo"    "tensor(i->i+1, 5)[2..]"            "[3, 4, 5]"
run "sl.1d_open_hi"    "tensor(i->i+1, 5)[..2]"            "[1, 2, 3]"
run "sl.3d_all"        "shape(zeros(2,3,4)[..,1,..])"      "[2, 4]"
run "sl.neg_all"       "(1,2,3; 4,5,6; 7,8,9)[-1, ..]"    "[7, 8, 9]"
# tuple slices
run "sl.tup_all"       "(10,20,30,40)[..]"                 "[10, 20, 30, 40]"
run "sl.tup_open_lo"   "(10,20,30,40)[2..]"                "[30, 40]"
run "sl.tup_open_hi"   "(10,20,30,40)[..1]"                "[10, 20]"
run "sl.tup_range"     "(10,20,30,40)[1..2]"               "[20, 30]"

# ── outer product ─────────────────────────────────────────────────────────────
section "OUTER PRODUCT"
run "out.shape_2d"    "shape(outer(ones(2), ones(3)))"             "[2, 3]"
run "out.sum_ones"    "sum(outer(ones(3), ones(4)))"               "12"
run "out.vals"        "outer(tensor(i->i+1,2), tensor(i->i+1,3))[1,2]" "6"
run "out.3d_shape"    "shape(outer(ones(2), outer(ones(3), ones(4))))" "[2, 3, 4]"

# ── reshape ───────────────────────────────────────────────────────────────────
section "RESHAPE"
run "rs.2d_shape"     "shape(reshape(ones(6), 2, 3))"              "[2, 3]"
run "rs.3d_shape"     "shape(reshape(zeros(24), 2, 3, 4))"         "[2, 3, 4]"
run "rs.1d_from_2d"   "shape(reshape(eye(3), 9))"                  "[9]"
run "rs.data_preserved" "sum(reshape(eye(3), 9))"                  "3"
run "rs.roundtrip"    "trace(reshape(reshape(eye(3), 9), 3, 3))"   "3"
run_err "rs.size_mismatch" "reshape(ones(6), 2, 4)"

# ── permute ───────────────────────────────────────────────────────────────────
section "PERMUTE"
run "pm.2d_swap"      "shape(permute(zeros(2,3), 1, 0))"           "[3, 2]"
run "pm.3d_shape"     "shape(permute(zeros(2,3,4), 2, 0, 1))"      "[4, 2, 3]"
run "pm.identity"     "trace(permute(eye(3), 0, 1))"               "3"
run "pm.val_check"    "permute(matrix((i,j)->i*3+j, 2, 3), 1, 0)[2,1]" "5"

# ── transpose (generalized) ───────────────────────────────────────────────────
section "TRANSPOSE GEN"
run "tr.2d_classic"   "transpose(matrix((i,j)->i*3+j, 2, 3))[0,1]"  "3"
run "tr.3d_rev_shape" "shape(transpose(zeros(2,3,4)))"               "[4, 3, 2]"
run "tr.swap_axes"    "shape(transpose(zeros(2,3,4), 0, 2))"         "[4, 3, 2]"
run "tr.swap_mid"     "shape(transpose(zeros(2,3,4), 1, 2))"         "[2, 4, 3]"

# ── cat ───────────────────────────────────────────────────────────────────────
section "CAT"
run "cat.axis0"       "shape(cat(0, eye(2), eye(2)))"               "[4, 2]"
run "cat.axis1"       "shape(cat(1, eye(2), eye(2)))"               "[2, 4]"
run "cat.1d"          "cat(0, tensor(i->i,3), tensor(i->i,3))"      "[0, 1, 2, 0, 1, 2]"
run "cat.vstack_eq"   "trace(cat(0, eye(3), eye(3)))"               "3"
run "cat.hstack_eq"   "trace(cat(0, eye(3), eye(3)))"               "3"
run "cat.3_tensors"   "shape(cat(0, ones(2,3), ones(2,3), ones(2,3)))" "[6, 3]"

# ── squeeze / unsqueeze ───────────────────────────────────────────────────────
section "SQUEEZE / UNSQUEEZE"
run "sq.remove_ones"  "shape(squeeze(zeros(1,3,1)))"               "[3]"
run "sq.all_ones"     "squeeze(zeros(1,1,1))"                      "0"
run "sq.no_ones"      "shape(squeeze(zeros(2,3)))"                 "[2, 3]"
run "us.front"        "shape(unsqueeze(zeros(3), 0))"              "[1, 3]"
run "us.back"         "shape(unsqueeze(zeros(3), 1))"              "[3, 1]"
run "us.mid"          "shape(unsqueeze(zeros(2,4), 1))"            "[2, 1, 4]"

# ── sum / prod by axis ────────────────────────────────────────────────────────
section "AXIS REDUCTION"
run "ax.sum_axis0"    "sum(ones(2,3), 0)"                          "[2, 2, 2]"
run "ax.sum_axis1"    "sum(ones(2,3), 1)"                          "[3, 3]"
run "ax.prod_axis0"   "prod(ones(2,3)*2, 0)"                       "[4, 4, 4]"
run "ax.sum_3d"       "shape(sum(zeros(2,3,4), 1))"                "[2, 4]"
run "ax.sum_1d"       "sum(tensor(i->i,5), 0)"                     "10"

# ── stats on tensors ──────────────────────────────────────────────────────────
section "TENSOR STATS"
run "ts.mean_2d"      "mean(ones(3,3))"                            "1"
run "ts.mean_vals"    "mean(matrix((i,j)->i*2+j, 2, 2))"          "1.5"
run "ts.std_zeros"    "std(ones(3,3))"                             "0"
run "ts.var_uniform"  "var(ones(3,3))"                             "0"
run "ts.median_2d"    "median(matrix((i,j)->i*3+j+1, 2, 3))"      "3.5"

# ── reduce on tensors ────────────────────────────────────────────────────────
section "REDUCE TENSOR"
run "rdt.sum"         "reduce((a,b)->a+b, ones(2,3))"              "6"
run "rdt.max"         "reduce((a,b)->if(a>b,a,b), matrix((i,j)->i*3+j, 2, 3))" "5"
run "rdt.product"     "reduce((a,b)->a*b, tensor(i->i+1,4))"       "24"

# ── diag from 1D tensor ───────────────────────────────────────────────────────
section "DIAG FROM 1D TENSOR"
run "dg.from_1d"      "trace(diag(tensor(i->i+1, 3)))"             "6"
run "dg.from_flat"    "diag(flatten(eye(2))[0..1])"                "⎡ 1  0 ⎤ ⎣ 0  0 ⎦"

# ── matmul extended (1D/2D mixed) ─────────────────────────────────────────────
section "MATMUL EXTENDED"
run "mm.1d_1d"        "tensor(i->i+1,3) @ tensor(i->i+1,3)"       "14"
run "mm.2d_1d"        "matrix((i,j)->if(i==j,1,0),3,3) @ tensor(i->i,3)" "[0, 1, 2]"
run "mm.1d_2d"        "tensor(i->1,3) @ eye(3)"                   "[1, 1, 1]"
run "mm.2d_1d_sum"    "sum(eye(3) @ tensor(i->i,3))"              "3"

# ── dim(T, axis) ─────────────────────────────────────────────────────────────
section "DIM"
run "dm.0"            "dim(eye(3), 0)"                              "3"
run "dm.1"            "dim(eye(3), 1)"                              "3"
run "dm.3d_last"      "dim(zeros(2,3,4), 2)"                        "4"
run "dm.3d_first"     "dim(zeros(2,3,4), 0)"                        "2"
run "dm.shape_check"  "dim(zeros(2,3,4), 1)"                        "3"
run "dm.tuple"        "dim((1,2,3,4), 0)"                           "4"

# ── sum(f, n) two-arg form ────────────────────────────────────────────────────
section "SUM TWO-ARG"
run "s2.sum_n"        "sum(k->k, 5)"                                "10"
run "s2.sum_sq"       "sum(k->k^2, 4)"                              "14"
run "s2.sum_zero"     "sum(k->k, 0)"                                "0"
run "s2.prod_n"       "prod(k->k+1, 4)"                             "24"
run "s2.with_dim"     "sum(k->k, dim(eye(3),0))"                   "3"
run "s2.contraction"  "sum(k->tensor(i->i+1,3)[k]*tensor(i->i+1,3)[k], dim(eye(3),0))" "14"

# ── tensordot ─────────────────────────────────────────────────────────────────
section "TENSORDOT"
run "td.matmul_scalar"  "tensordot(eye(3), eye(3), 1)"                     "⎡ 1  0  0 ⎤ ⎢ 0  1  0 ⎥ ⎣ 0  0  1 ⎦"
run "td.dot_1d"         "tensordot(tensor(i->i+1,3), tensor(i->i+1,3), 1)" "14"
run "td.pair_matmul"    "tensordot(eye(3), eye(3), (1,0))"                 "⎡ 1  0  0 ⎤ ⎢ 0  1  0 ⎥ ⎣ 0  0  1 ⎦"
run "td.pair_dot"       "tensordot(tensor(i->i+1,3), tensor(i->i+1,3), (0,0))" "14"
run "td.2x3_3x2"        "shape(tensordot(zeros(2,3), zeros(3,2), 1))"      "[2, 2]"
run "td.outer_via_0"    "shape(tensordot(zeros(2,3), zeros(4,5), 0))"      "[2, 3, 4, 5]"
run "td.3d_contract"    "shape(tensordot(zeros(2,3,4), zeros(4,5), 1))"    "[2, 3, 5]"
run "td.scalar_result"  "tensordot(tensor(i->1,4), tensor(i->1,4), 1)"     "4"

# ── ComplexTensor construction ────────────────────────────────────────────────
section "COMPLEX TENSOR"
run "ct.tensor_complex"   "tensor(k->k+i, 4)"                          "[i, 1 + i, 2 + i, 3 + i]"
run "ct.tensor_2d"        "shape(tensor((i,j)->i+j*i, 3, 4))"          "[3, 4]"
run "ct.tensor_imag_only" "tensor(k->k*i, 3)"                          "[0, i, 2i]"
run "ct.tensor_real_fn"   "tensor(k->k, 3)"                            "[0, 1, 2]"

# ── ComplexTensor arithmetic ──────────────────────────────────────────────────
section "COMPLEX TENSOR ARITHMETIC"
run "cta.add"    "tensor(k->k+k*i, 3) + tensor(k->1, 3)"              "[1, 2 + i, 3 + 2i]"
run "cta.sub"    "tensor(k->k*(1+i), 3) - tensor(k->k, 3)"            "[0, i, 2i]"
run "cta.mul"    "tensor(k->1+i, 2) * tensor(k->1+i, 2)"              "[2i, 2i]"
run "cta.scale"  "2 * tensor(k->k+k*i, 3)"                            "[0, 2 + 2i, 4 + 4i]"
run "cta.neg"    "-(tensor(k->k+k*i, 3))"                             "[0, -1 - i, -2 - 2i]"
run "cta.ct_num" "tensor(k->k+k*i, 3) + 1"                            "[1, 2 + i, 3 + 2i]"
run "cta.ct_cx"  "tensor(k->k, 3) + i"                                "[i, 1 + i, 2 + i]"

# ── ComplexTensor queries ─────────────────────────────────────────────────────
section "COMPLEX TENSOR QUERIES"
run "ctq.shape"  "shape(tensor(k->k+i, 6))"                           "[6]"
run "ctq.shape2" "shape(tensor((r,c)->r+c*i, 3, 4))"                  "[3, 4]"
run "ctq.len"    "len(tensor(k->k+i, 5))"                             "5"
run "ctq.rows"   "rows(tensor((r,c)->r+c*i, 3, 4))"                   "3"
run "ctq.cols"   "cols(tensor((r,c)->r+c*i, 3, 4))"                   "4"
run "ctq.dim0"   "dim(tensor((r,c)->r+c*i, 3, 4), 0)"                 "3"
run "ctq.dim1"   "dim(tensor((r,c)->r+c*i, 3, 4), 1)"                 "4"

# ── ComplexTensor indexing ────────────────────────────────────────────────────
section "COMPLEX TENSOR INDEXING"
run "cti.scalar" "tensor(k->k+k*i, 5)[2]"                             "2 + 2i"
run "cti.slice"  "tensor(k->k+k*i, 5)[1..3]"                          "[1 + i, 2 + 2i, 3 + 3i]"
run "cti.neg1"   "tensor(k->k+k*i, 4)[-1]"                            "3 + 3i"
run "cti.2d_row"  "shape(tensor((r,c)->r+c*i, 3, 4)[1])"               "[4]"
run "cti.2d_elem" "tensor((r,c)->r+c*i, 3, 3)[(1,2)]"                "1 + 2i"

# ── ComplexTensor map / sum ───────────────────────────────────────────────────
section "COMPLEX TENSOR MAP SUM"
run "ctms.map_conj"     "map(conj, tensor(k->k+k*i, 3))"              "[0, 1 - i, 2 - 2i]"
run "ctms.map_abs"      "map(abs, tensor(k->k+k*i, 3))"               "[0, 1.4142135623730951, 2.8284271247461903]"
run "ctms.sum_total"    "sum(tensor(k->k+k*i, 4))"                    "6 + 6i"
run "ctms.sum_axis"     "sum(tensor((r,c)->r+c*i, 3, 3), 0)"          "[3, 3 + 3i, 3 + 6i]"
run "ctms.sum_fn_cx"    "sum(k->k*i, 4)"                              "6i"
run "ctms.sum_lo_hi"    "sum(k->k*i, 0, 3)"                           "6i"

# ── fft / ifft with ComplexTensor (n-D) ──────────────────────────────────────
section "FFT COMPLEX"
run "ffc.shape_1d"      "shape(fft(tensor(k->k+k*i, 8)))"            "[8]"
run "ffc.shape_2d"      "shape(fft(tensor((r,c)->r+c*i, 4, 4)))"     "[4, 4]"
run "ffc.roundtrip"     "T = tensor(k->k+k*i, 4); sum(abs(ifft(fft(T)) - T))" "0"
run "ffc.axis"          "shape(fft(tensor((r,c)->r+c*i, 3, 4), 1))"  "[3, 4]"
run "ffc.axes_tuple"    "shape(fft(tensor((r,c)->r+c*i, 3, 4), (0,1)))" "[3, 4]"
run "ffc.real_input_ct" "T = tensor(k->k, 4); shape(fft(T))"         "[4]"
run "ffc.re_im_pair"    "Re = zeros(4); Im = tensor(k->k, 4); shape(fft(Re, Im))" "[4]"

# ── zero-arg lambdas ─────────────────────────────────────────────────────────
section "ZERO-ARG LAMBDA"
run "zl.call"           "f = () -> 42; f()"                    "42"
run "zl.inline"         "(() -> 99)()"                          "99"
run "zl.tensor_ret"     "f = () -> [1,2,3]; f()"              "[1, 2, 3]"
run "zl.side_effect"    "{n = cell(0); tick = () -> set(n, get(n)+1); tick(); get(n)}"  "1"

# ── cell / get / set ──────────────────────────────────────────────────────────
section "CELL"
# Basic read/write
run "cell.init"         "{c = cell(10); get(c)}"                "10"
run "cell.set"          "{c = cell(10); set(c, 42); get(c)}"    "42"
run "cell.set_ret"      "{c = cell(0); set(c, 99)}"             "99"
# Shared identity: two names, one cell
run "cell.shared"       "{c = cell(1); d = c; set(c, 2); get(d)}"  "2"
# Tensor in a cell
run "cell.tensor"       "{c = cell([1,2,3]); set(c, get(c)*2); get(c)}"  "[2, 4, 6]"
# Stateful counter via zero-arg lambda
run "cell.counter"      "{n = cell(0); f = () -> set(n, get(n)+1); f(); f(); f(); get(n)}"  "3"
# Step-based accumulation
run "cell.step_vec"     "{s = cell([0,0,0]); bump = () -> set(s, get(s)+1); bump(); bump(); get(s)}"  "[2, 2, 2]"
# Display format
run "cell.display_num"  "cell(42)"                              "cell(42)"
run "cell.display_vec"  "cell([1,2,3])"                         "cell([1, 2, 3])"
# Errors
run_err "cell.err_get"  "get(5)"
run_err "cell.err_set"  "set(5, 10)"

# ── [] array literals (Expr::Array) ──────────────────────────────────────────
section "ARRAY LITERALS"
# Basic construction
run "arr.empty"         "[]"                                    "[]"
run "arr.one"           "[1]"                                   "[1]"
run "arr.three"         "[1, 2, 3]"                             "[1, 2, 3]"
run "arr.float"         "[0.5, 1.5]"                            "[0.5, 1.5]"
run "arr.expr"          "[1+1, 2*3, 4^2]"                      "[2, 6, 16]"
run "arr.var"           "x=7; [x, x+1, x+2]"                 "[7, 8, 9]"
# [x] is a length-1 tensor, unlike (x) which is a scalar
run "arr.one_vs_paren"  "len([42])"                             "1"
run "arr.paren_scalar"  "(42)"                                  "42"
# Complex elements are fine
run "arr.complex"       "len([1+2i, 3+4i])"                    "2"
# Matrix literal with []
run "arr.matrix"        "rows([1,2;3,4])"                       "2"
run "arr.matrix_val"    "[1,2;3,4][0,1]"                       "2"
# Indexing works the same as tensor from ()
run "arr.index"         "[10,20,30][1]"                         "20"
run "arr.slice"         "[10,20,30][0..1]"                      "[10, 20]"
# Arithmetic on [] tensors
run "arr.add"           "[1,2,3] + [4,5,6]"                    "[5, 7, 9]"
run "arr.scale"         "2 * [1,2,3]"                          "[2, 4, 6]"
# Round-trip: output of a tensor can be typed back as input
run "arr.roundtrip"     "x = [1,2,3]; x"                      "[1, 2, 3]"
# Errors: non-numeric elements
run_err "arr.err_tuple"   "[(1,2),(3,4)]"
run_err "arr.err_fn"      "[x->x]"
run_err "arr.err_nested"  "[[1,2],[3,4]]"

# ── shift ─────────────────────────────────────────────────────────────────────
section "SHIFT"
# 1-D: positive n pushes content toward higher indices, replicates leading edge
run "shift.right1"      "shift([1,2,3,4,5], 1, 0)"             "[1, 1, 2, 3, 4]"
run "shift.right2"      "shift([1,2,3,4,5], 2, 0)"             "[1, 1, 1, 2, 3]"
run "shift.left1"       "shift([1,2,3,4,5], -1, 0)"            "[2, 3, 4, 5, 5]"
run "shift.left2"       "shift([1,2,3,4,5], -2, 0)"            "[3, 4, 5, 5, 5]"
run "shift.zero"        "shift([1,2,3], 0, 0)"                 "[1, 2, 3]"
# 2-D: axis 0 = rows, axis 1 = cols
run "shift.2d_row_dn"   "shift([1,2;3,4], 1, 0)"               "⎡ 1  2 ⎤
⎣ 1  2 ⎦"
run "shift.2d_row_up"   "shift([1,2;3,4], -1, 0)"              "⎡ 3  4 ⎤
⎣ 3  4 ⎦"
run "shift.2d_col_rt"   "shift([1,2;3,4], 1, 1)"               "⎡ 1  1 ⎤
⎣ 3  3 ⎦"
run "shift.2d_col_lf"   "shift([1,2;3,4], -1, 1)"              "⎡ 2  2 ⎤
⎣ 4  4 ⎦"
# Edge values are replicated (Neumann BC), not zeros
run "shift.edge_rep"    "shift([10,20,30], 5, 0)[0]"           "10"
run "shift.edge_rep2"   "shift([10,20,30], -5, 0)[2]"          "30"

# ── roll ──────────────────────────────────────────────────────────────────────
section "ROLL"
# Positive n: last element wraps to front
run "roll.right1"       "roll([1,2,3,4,5], 1, 0)"              "[5, 1, 2, 3, 4]"
run "roll.right2"       "roll([1,2,3,4,5], 2, 0)"              "[4, 5, 1, 2, 3]"
run "roll.left1"        "roll([1,2,3,4,5], -1, 0)"             "[2, 3, 4, 5, 1]"
run "roll.zero"         "roll([1,2,3], 0, 0)"                  "[1, 2, 3]"
# Full wrap is identity
run "roll.full_wrap"    "roll([1,2,3], 3, 0)"                  "[1, 2, 3]"
# 2-D: axis 0 = rows
run "roll.2d_row"       "roll([1,2;3,4], 1, 0)"                "⎡ 3  4 ⎤
⎣ 1  2 ⎦"
run "roll.2d_col"       "roll([1,2;3,4], 1, 1)"                "⎡ 2  1 ⎤
⎣ 4  3 ⎦"
# roll vs shift differ at boundaries
run "roll.vs_shift"     "roll([1,2,3], 1, 0)[0]"               "3"

# ── lerp ──────────────────────────────────────────────────────────────────────
section "LERP"
run "lerp.scalar_0"     "lerp(0, 10, 0)"                       "0"
run "lerp.scalar_1"     "lerp(0, 10, 1)"                       "10"
run "lerp.scalar_half"  "lerp(0, 10, 0.5)"                     "5"
run "lerp.scalar_frac"  "lerp(2, 8, 0.25)"                     "3.5"
run "lerp.vec_t"        "lerp(0, 10, [0, 0.5, 1])"             "[0, 5, 10]"
run "lerp.vec_ab"       "lerp([0,0], [10,20], 0.5)"            "[5, 10]"
run "lerp.all_vecs"     "lerp([1,2], [3,4], [0,1])"            "[1, 4]"
run "lerp.mask_blend"   "lerp(500, 250, [0,0,1,1])"            "[500, 500, 250, 250]"
# lerp(a,b,0)=a and lerp(a,b,1)=b for tensors too
run "lerp.tensor_t0"    "lerp([1,2,3], [4,5,6], 0)"            "[1, 2, 3]"
run "lerp.tensor_t1"    "lerp([1,2,3], [4,5,6], 1)"            "[4, 5, 6]"

# ── clamp ─────────────────────────────────────────────────────────────────────
section "CLAMP"
run "clamp.above"       "clamp(5, 0, 3)"                       "3"
run "clamp.below"       "clamp(-1, 0, 3)"                      "0"
run "clamp.inside"      "clamp(2, 0, 3)"                       "2"
run "clamp.at_lo"       "clamp(0, 0, 3)"                       "0"
run "clamp.at_hi"       "clamp(3, 0, 3)"                       "3"
run "clamp.vec"         "clamp([-1, 0.5, 2], 0, 1)"            "[0, 0.5, 1]"
run "clamp.vec_lo"      "clamp([-5,-3,-1], -2, 0)"             "[-2, -2, -1]"
run "clamp.negrange"    "clamp(-1.5, -2, -1)"                  "-1.5"
run_err "clamp.bad_range"   "clamp(1, 5, 0)"

# ── !savetensor / !loadtensor ─────────────────────────────────────────────────
section "SAVETENSOR / LOADTENSOR"
_TMLT=$(mktemp /tmp/mlt_test_XXXXXX.mlt)

_repl_check() {
    local label="$1" script="$2" pat="$3"
    local out
    out=$(printf '%s\n' "$script" | "$M" 2>&1)
    if echo "$out" | grep -qE -- "$pat"; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | pat='$pat' | got='$(norm "$out")'")
    fi
}

_repl_check "mlt.roundtrip.val"   "T=(1.5,2.5,3.5)
!savetensor T $_TMLT
!loadtensor U $_TMLT
U[1]"                                      "2\.5"

_repl_check "mlt.roundtrip.shape" "M=(1,2;3,4)
!savetensor M $_TMLT
!loadtensor N $_TMLT
shape(N)"                                  "\[2, 2\]"

_repl_check "mlt.roundtrip.3d"    "C=ones(2,3,4)
!savetensor C $_TMLT
!loadtensor D $_TMLT
shape(D)"                                  "\[2, 3, 4\]"

_repl_check "mlt.save_confirms"   "V=eye(3)
!savetensor V $_TMLT"                      "saved V"

_repl_check "mlt.load_confirms"   "!loadtensor W $_TMLT" "loaded W"

_repl_check "mlt.err_undef"       "!savetensor nosuchvar /tmp/x.mlt" \
                                   "not defined"
_repl_check "mlt.err_nonten"      "x=42
!savetensor x /tmp/x.mlt"                 "not a tensor"
_repl_check "mlt.err_nofile"      "!loadtensor Z /tmp/mlt_no_such_file_xyz.mlt" \
                                   "loadtensor:"

# complex tensor roundtrip
_repl_check "mlt.complex.roundtrip" "C = fft([1,1,0,0])
!savetensor C $_TMLT
!loadtensor D $_TMLT
re(D[1])"                              "1"
_repl_check "mlt.complex.im"        "C = fft([1,1,0,0])
!savetensor C $_TMLT
!loadtensor D $_TMLT
im(D[1])"                              "-1"
_repl_check "mlt.complex.confirms"  "C = fft([1,1,0,0])
!savetensor C $_TMLT"                  "complex"
_repl_check "mlt.complex.shape"     "C = fft([1,1,0,0])
!savetensor C $_TMLT
!loadtensor D $_TMLT
shape(D)"                              "\[4\]"

rm -f "$_TMLT"

# ── Bang commands in .math files ──────────────────────────────────────────────
section "BANG COMMANDS IN FILES"

# Helper: write a temp .math file, run it with -f, check output matches pattern
_file_check() {
    local label="$1" script="$2" pat="$3"
    local tf; tf=$(mktemp /tmp/mlt_test_XXXXXX.math)
    printf '%s\n' "$script" > "$tf"
    local out; out=$("$M" -f "$tf" 2>&1)
    rm -f "$tf"
    if [ -z "$pat" ] || echo "$out" | grep -qE -- "$pat"; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | pat='$pat' | got='$(norm "$out")'")
    fi
}

_file_check "file.bang.version"   "!version"                           "mathlang v"
_file_check "file.bang.defs"      "x = 99
!defs"                                                                  "x"

# include chaining: write a lib file, include it from a main file
_TLIB=$(mktemp /tmp/mlt_lib_XXXXXX.math)
printf 'sq = x -> x*x\n' > "$_TLIB"
_file_check "file.bang.include.chain"  "!include $_TLIB
sq(4)"                                                                  "16"
rm -f "$_TLIB"

_file_check "file.bang.savetensor" "T = (1.0,2.0,3.0)
!savetensor T /tmp/mlt_file_bang_test.mlt"                              "saved T"

_file_check "file.bang.loadtensor" "T = (1.0,2.0,3.0)
!savetensor T /tmp/mlt_file_bang_test.mlt
!loadtensor U /tmp/mlt_file_bang_test.mlt"                              "loaded U"

_file_check "file.bang.tensor.val" "T = (4.0,5.0,6.0)
!savetensor T /tmp/mlt_file_bang_test.mlt
!loadtensor U /tmp/mlt_file_bang_test.mlt
U[1]"                                                                   "5"

rm -f /tmp/mlt_file_bang_test.mlt

# ── !print ────────────────────────────────────────────────────────────────────
section "!print"

_repl_check "print.plain"       "!print hello world"              "hello world"
_repl_check "print.interp"      "x = 7
!print x is {x}"                                                   "x is 7"
_repl_check "print.expr"        "!print 2 + 2 = {2 + 2}"          "2 \+ 2 = 4"
_repl_check "print.multi"       "a = 3
b = 4
!print {a} {b} {sqrt(a^2+b^2)}"                                    "3 4 5"
_repl_check "print.blank"       "!print"                           "^$"
_repl_check "print.escape"      "!print {{x}} is a placeholder"   "\{x\} is a placeholder"
_repl_check "print.tensor"      "!print {(1,2,3)}"                 "\[1, 2, 3\]"
_repl_check "print.err"         "!print {nosuchvar}"               "<error:"
_file_check  "print.in_file"    "n = 5
!print sum = {sum(x -> x, 1, n)}"                                  "sum = 15"

# ── VM: tensor indexing in lambda bodies ─────────────────────────────────────
section "vm.index"
run "vm.index.1d.0"     "f(v) = v[0]; f([3,7,9])"                              "3"
run "vm.index.1d.1"     "f(v) = v[1]; f([3,7,9])"                              "7"
run "vm.index.1d.neg"   "f(v) = v[-1]; f([3,7,9])"                             "9"
run "vm.index.param"    "f(v,i) = v[i]; f([10,20,30], 2)"                      "30"
run "vm.index.2d"       "f(M,i,j) = M[i,j]; M = tensor((i,j)->i*10+j, 3, 4); f(M,1,2)" "12"
run "vm.index.2d.sum"   "f(M) = M[0,0] + M[1,1]; M = tensor((i,j)->i*10+j, 3, 4); f(M)" "11"
run "vm.index.expr"     "f(v,n) = v[n-1]; f([5,6,7], 3)"                      "7"
run "vm.index.tensor.1" "v = [10,20,30,40]; tensor((i)->v[i], 4)"             "[10, 20, 30, 40]"
run "vm.index.tensor.2" "M = tensor((i,j)->i*3+j, 2,3); sum(tensor((i,j)->M[i,j]*2, 2, 3))" "30"

# ── VM: nested lambdas (MakeClosure) ─────────────────────────────────────────
section "vm.closure"
run "vm.closure.curry"     "add(x) = y -> x + y; add(3)(4)"                   "7"
run "vm.closure.store"     "add(x) = y -> x + y; f = add(10); f(5)"           "15"
run "vm.closure.mul"       "mul(x) = y -> x * y; mul(6)(7)"                   "42"
run "vm.closure.two"       "f(a,b) = x -> a*x + b; g = f(2,3); g(5)"         "13"
run "vm.closure.local"     "f(n) = { k = n*2; x -> x + k }; f(5)(3)"         "13"
run "vm.closure.map"       "adder(n) = x -> x + n; map(adder(10), [1,2,3])"  "[11, 12, 13]"
run "vm.closure.tensor"    "add(x) = y -> x + y; tensor((i)->add(i)(i*2), 4)" "[0, 3, 6, 9]"
run "vm.closure.filter"    "above(n) = x -> x > n; filter(above(3), [1,2,3,4,5])"  "[4, 5]"
run "vm.closure.localstep" "f(n) = { k = n + 1; x -> x * k }; map(f(2), [1,2,3])" "[3, 6, 9]"

# ── VM: Def::Func in block (non-recursive compiles; recursive falls back) ────
section "vm.deffunc"
run "vm.deffunc.simple"   "f(n) = { h(x) = x*2; h(n) + h(n+1) }; f(3)"        "14"
run "vm.deffunc.capture"  "f(n) = { h(x) = x + n; h(3) + h(4) }; f(10)"       "27"
run "vm.deffunc.maplocal" "f(n) = { h(x) = x*n; map(h, [1,2,3]) }; f(5)"      "[5, 10, 15]"
run "vm.deffunc.chain"    "f(n) = { a(x) = x+1; b(x) = a(x)*n; b(2) }; f(4)"  "12"
run "vm.deffunc.rec"      "f(n) = { g(x) = if(x<=0,0,x+g(x-1)); g(n) }; f(4)" "10"

# ── VM: Loop instruction (sum/prod/iterate/scan compiled in lambda bodies) ───
# These exercise the in-VM flat loop (TODO 1e): the special forms no longer force
# a tree-walk fallback. Results must match the tree-walk path exactly.
section "vm.loop"
run "vm.loop.sum3"       "g(n) = sum(k->k*k, 1, n); g(10)"                     "385"
run "vm.loop.sum2"       "g(n) = sum(k->k, n); g(5)"                           "10"
run "vm.loop.sum1"       "h(t) = sum(t); h((1,2,3,4))"                         "10"
run "vm.loop.prod3"      "f(n) = prod(k->k, 1, n); f(5)"                       "120"
run "vm.loop.prod2"      "f(n) = prod(k->k+1, n); f(4)"                        "24"
run "vm.loop.sumcap"     "f(n,m) = sum(k->k*m, 1, n); f(4, 10)"                "100"
run "vm.loop.sumcplx"    "f(n) = sum(k->k*1i, 1, n); f(3)"                     "6i"
run "vm.loop.iter"       "step(x)=x*2+1; go(n)=iterate(step, 0, n); go(5)"     "31"
run "vm.loop.iter.cap"   "go(a,n)=iterate(x->x+a, 0, n); go(3, 4)"             "12"
run "vm.loop.iter.big"   "f(n)=iterate(x->x+1, 0, n); f(1000000)"             "1000000"
run "vm.loop.scan"       "go(n)=scan(k->k+1, 0, n); go(4)"                     "[0, 1, 2, 3, 4]"
run "vm.loop.scan.vshape" "orbit(n)=scan(v->(v[0]+v[1], v[1]), (1,1), n); shape(orbit(2))" "[3, 2]"
run "vm.loop.scan.vrow"   "orbit(n)=scan(v->(v[0]+v[1], v[1]), (1,1), n); orbit(2)[2]"     "[3, 1]"
run "vm.loop.nested"     "f(n)=sum(j->iterate(x->x+1, 0, j), 1, n); f(4)"     "10"
run "vm.loop.sum.range"  "f(n)=sum(k->2*k, 0, n); f(100)"                     "10100"
run_err "vm.loop.iter.neg" "f(n)=iterate(g->g, 0, n); f(-1)"

# ── stdlib: examples/integrators.math (verlet / rk4, built in pure mathlang) ──
section "stdlib.integrators"
run_lib "int.load"        "$LIB_INT" "1+1"                                          "2"
# Velocity-Verlet SHO (dH/dq=q, dH/dp=p): energy (q²+p²)/2 ≈ 0.5, bounded drift
run_lib "int.verlet.efinal" "$LIB_INT" "f=verletFinal(q->q,p->p,1.0,0.0,0.05,200); round((f[0]^2+f[1]^2)/2, 2)" "0.5"
run_lib "int.verlet.shape" "$LIB_INT" "shape(verletOrbit(q->q,p->p,1.0,0.0,0.05,200))" "[201, 2]"
run_lib "int.verlet.seed"  "$LIB_INT" "verletOrbit(q->q,p->p,1.0,0.0,0.05,3)[0]"      "[1, 0]"
# Generator: first call returns the seed, then advances; cells are independent
run_lib "int.gen.seed"     "$LIB_INT" "{ g=verlet(q->q,p->p,1.0,0.0,0.05); g(0)[0] }" "1"
run_lib "int.gen.advances" "$LIB_INT" "{ g=verlet(q->q,p->p,1.0,0.0,0.05); g(0); round(g(0)[0], 5) }" "0.99875"
run_lib "int.gen.indep"    "$LIB_INT" "{ g=verlet(q->q,p->p,1.0,0.0,0.05); h=verlet(q->q,p->p,1.0,0.0,0.05); g(0); h(0)[0] }" "1"
# RK4: y'=y, y(0)=1 → e at t=1 ; vector RK4 returns (T, Y) with separate shapes
run_lib "int.rk4.exp"      "$LIB_INT" "round(rk4Final((t,y)->y, 0.0, 1.0, 0.001, 1000), 5)" "2.71828"
run_lib "int.rk4.vshapeT"  "$LIB_INT" "shape(rk4Orbit((t,y)->(y[1],-y[0]), 0.0, (1.0,0.0), 0.01, 50)[0])" "[51]"
run_lib "int.rk4.vshapeY"  "$LIB_INT" "shape(rk4Orbit((t,y)->(y[1],-y[0]), 0.0, (1.0,0.0), 0.01, 50)[1])" "[51, 2]"

# ── HDF5 (skipped unless built with --features hdf5) ─────────────────────────
section "HDF5"
_H5F=$(mktemp /tmp/mlt_test_XXXXXX.h5)
_H5F2=$(mktemp /tmp/mlt_test_XXXXXX.h5)
# Probe: feature absent → "build with --features hdf5"; feature present → "not defined"
_h5_probe=$(printf '!savehdf5 __probe__ /dev/null\n' | "$M" 2>&1)
if echo "$_h5_probe" | grep -q "not defined"; then
    _H5_OK=true
else
    _H5_OK=false
fi

if $_H5_OK; then
    _repl_check "h5.real.val"     "A=(1,2,3;4,5,6)
!savehdf5 A $_H5F
!loadhdf5 B $_H5F /A
B[0,1]"                                         "2"

    _repl_check "h5.real.shape"   "A=(1,2,3;4,5,6)
!savehdf5 A $_H5F --overwrite
!loadhdf5 B $_H5F /A
shape(B)"                                       "\[2, 3\]"

    _repl_check "h5.append"       "A=[10,20,30]
!savehdf5 A $_H5F --overwrite
B=[7,8,9]
!savehdf5 B $_H5F /B --append
!loadhdf5 C $_H5F /B
C[2]"                                           "9"

    _repl_check "h5.nested"       "V=[1,2,3]
!savehdf5 V $_H5F2 /grp/sub/data
!loadhdf5 W $_H5F2 /grp/sub/data
W[1]"                                           "2"

    _repl_check "h5.list"         "A=(1,2;3,4)
!savehdf5 A $_H5F2 --overwrite
!loadhdf5 _ $_H5F2 --list"                      "f64"

    _repl_check "h5.complex.val"  "C=fft([1,1,0,0])
!savehdf5 C $_H5F2 --overwrite
!loadhdf5 D $_H5F2 /C
im(D[1])"                                       "-1"

    _repl_check "h5.complex.shape" "C=fft([1,1,0,0])
!savehdf5 C $_H5F2 --overwrite
!loadhdf5 D $_H5F2 /C
shape(D)"                                       "\[4\]"

    _repl_check "h5.gzip"         "T=ones(10,10)
!savehdf5 T $_H5F2 --overwrite --gzip 6
!loadhdf5 U $_H5F2 /T
U[0,0]"                                         "1"

    _repl_check "h5.err.undef"    "!savehdf5 nosuchvar $_H5F2"   "not defined"
    _repl_check "h5.err.nonten"   "x=42
!savehdf5 x $_H5F2"                             "not a tensor"
    _repl_check "h5.err.nofile"   "!loadhdf5 Z /tmp/no_such_file_h5_xyz.h5" \
                                                "loadhdf5:"
else
    echo "(skipping HDF5 tests — binary not compiled with --features hdf5)"
fi
rm -f "$_H5F" "$_H5F2"

# ── TYPE HINTS ────────────────────────────────────────────────────────────────
section "TYPE HINTS"
run "colon.named_ret"      'f(x: real): real = x^2; f(3)'        "9"
run_err "colon.ret_reject" 'f(x): real = x+1i; f(3)'
run "colon.lambda_ret"     'g = (x: real): real -> x*2; g(4)'    "8"
run_err "colon.lam_reject" 'g = (x): real -> x+1i; g(2)'
run "colon.zero_arg"       'c = (): real -> 5; c()'              "5"
run_err "colon.old_arrow"  'f(x) -> real = x; f(2)'
run "const.e_val"          'round(e, 6)'                         "2.718282"
run "typehint.real_param"        'f(x: real) = x^2; f(3)'          "9"
run "typehint.complex_to_real"   'f(x: real) = x^2; f(3+0i)'       "9"
run_err "typehint.complex_reject" 'f(x: real) = x; f(1+2i)'
run "typehint.nat_param"         'f(n: nat) = n+1; f(5)'           "6"
run_err "typehint.nat_neg"       'f(n: nat) = n+1; f(-1)'
run_err "typehint.nat_frac"      'f(n: nat) = n+1; f(1.5)'
run "typehint.int_param"         'f(n: int) = n; f(-3)'            "-3"
run_err "typehint.int_frac"      'f(n: int) = n; f(1.5)'
run "typehint.tensor_param"      'f(T: tensor) = sum(T); f(linspace(1,3,3))'  "6"
run_err "typehint.tensor_scalar" 'f(T: tensor) = T; f(5)'
run "typehint.real_tensor"       'f(T: real tensor) = sum(T); f(linspace(0,1,5))'  "2.5"
run "typehint.complex_widen"     'f(x: complex) = re(x); f(3.0)'   "3"
run "typehint.fn_param"          'apply(f: fn, x) = f(x); apply(sqrt, 4)'  "2"
run_err "typehint.fn_reject"     'apply(f: fn, x) = f(x); apply(5, 4)'
run "typehint.ret_hint"          'f(x: real): real = x^2; f(3)'  "9"
run_err "typehint.ret_complex"   'f(x): real = x+1i; f(3)'
run "typehint.num_any"           'f(x: num) = re(x); f(3+2i)'      "3"

# ── Bug fixes (TODO_BUGS) ────────────────────────────────────────────────────
section "BUG FIXES"

# Bug 8: block reassignment — variables can be updated within a block
run "bug8.block_reassign_simple"   'f(x) = { y = x; y = y + 1; y }'           ""   # just define
run "bug8.block_reassign_value"    'f(x) = { y = x; y = y + 1; y }; f(3)'     "4"
run "bug8.block_reassign_chain"    'f(x) = { x = x + 1; x = x * 2; x }; f(3)' "8"
run "bug8.block_reassign_treelike" '{ a = 1; a = 2; a }'                       "2"
run "bug8.block_reassign_func"     '(x -> { y = x; y = y*y; y })(5)'           "25"

# Bug 1a: bare single-arg typed lambda (f = x: tensor -> x)
run "bug1a.bare_typed_lambda"        'f = x: tensor -> len(x); f([1,2,3])'     "3"
run "bug1a.bare_typed_lambda_real"   'f = x: real -> x^2; f(3)'                "9"
run "bug1a.bare_typed_lambda_nat"    'f = n: nat -> n+1; f(5)'                 "6"
run_err "bug1a.bare_typed_lambda_reject" 'f = x: nat -> x; f(-1)'

# Bug 1b: 1-element tensor should NOT be destructured when param has tensor hint
run    "bug1b.tensor_hint_no_destruct"   'f = (x: tensor) -> len(x); f([5])'          "1"
run    "bug1b.tensor_hint_multi_elem"    'f = (x: tensor) -> sum(x); f([1,2,3])'      "6"
run_ok "bug1b.no_hint_still_works"      'f = (x, y) -> x + y; f([3, 4])'

# Bug 2: !type <fn> shows the function name
_repl_check "bug2.type_shows_name" "f = (x: tensor) -> x
!type f" "^f\("

_repl_check "bug2.type_builtin_name" "!type sin" "^sin\("

# Bug 3: rand help text is accurate (rand(a,b) does NOT mean range [a,b])
run    "bug3.rand_2args_shape"  'shape(rand(3,4))'   "[3, 4]"
run    "bug3.rand_1arg_len"     'len(rand(5))'        "5"
run_ok "bug3.rand_scalar"       'rand()'

# Bug 4: tensordot has a builtin_sig entry
_repl_check "bug4.tensordot_sig" "!type tensordot" "tensordot"

# Bug 5: graph/animate2D moved to ! commands; calling as functions is now an error
run_err "bug5.graph_fn_gone"        'graph(sin)'
run_err "bug5.animate2D_fn_gone"    'animate2D([1,2;3,4])'

# Bug 7: matrix display – value is correct regardless of display formatting
run    "bug7.matrix_val_correct"   'zeros(2,2)[0,0]'   "0"
run    "bug7.matrix_shape_correct" 'shape(zeros(2,2))' "[2, 2]"
# Verify the REPL multiline display starts a new line before the matrix
_repl_check "bug7.matrix_display_newline" "zeros(2,2)" "result =
"

# ── ncr / quadratic ───────────────────────────────────────────────────────────
run    "ncr.basic"         'ncr(10, 3)'        "120"
run    "ncr.zero_r"        'ncr(5, 0)'         "1"
run    "ncr.r_eq_n"        'ncr(5, 5)'         "1"
run    "ncr.r_gt_n"        'ncr(3, 5)'         "0"
run    "ncr.symmetry"      'ncr(8,3) == ncr(8,5)' "1"
run    "quad.two_real"     'quadratic(1,-5,6)' "(3, 2)"
run    "quad.double_root"  'quadratic(1,-2,1)' "(1, 1)"
run_match "quad.complex"   'quadratic(1,0,1)'  "i"
run_err   "quad.a_zero"    'quadratic(0,1,1)'

# ── !help <name> ──────────────────────────────────────────────────────────────
# Builtin help: shows description and injected type signature
_repl_check "help.builtin.sin"       "!help sin"          "sin\(x: num\) -> num"
_repl_check "help.builtin.map"       "!help map"          "map\(f: fn"
_repl_check "help.builtin.zeros"     "!help zeros"        "zeros"
# Bang command help works with and without leading !
_repl_check "help.bang.graph"        "!help !graph"       "!graph"
_repl_check "help.bang.graph_nob"    "!help graph"        "!graph"
_repl_check "help.bang.animate2D"    "!help !animate2D"   "animate2D"
_repl_check "help.bang.type"         "!help !type"        "!type"
_repl_check "help.bang.include"      "!help include"      "!include"
_repl_check "help.bang.quit"         "!help !quit"        "quit"
# Unknown name gives error message
_repl_check "help.unknown"           "!help nosuchfn"     "no help for"

# ── print summary ─────────────────────────────────────────────────────────────
echo
echo "================================"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "================================"
if [ ${#FAILS[@]} -gt 0 ]; then
    echo
    echo "FAILURES:"
    for f in "${FAILS[@]}"; do
        echo "  - $f"
    done
fi
