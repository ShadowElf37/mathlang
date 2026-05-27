#!/usr/bin/env bash
# Comprehensive test suite for mathlang.

M="$HOME/mathlang/target/release/m"
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
run "var.simple"    "x=3; y=4 : x^2 + y^2"        "25"
run "var.reuse"     "x=5 : x * x"                 "25"
run "var.chain"     "a=2; b=a*3 : b"               "6"
run "var.expr_rhs"  "x=2^8 : x"                    "256"
run "var.no_colon"  "x=3; x^2"                     "9"

# ── User functions ────────────────────────────────────────────────────────────
section "USER FUNCTIONS"
run "fn.one_arg"    "f(x) = x^2 : f(5)"             "25"
run "fn.two_arg"    "g(x,y) = x^2 + y^2 : g(3,4)"  "25"
run "fn.three_arg"  "h(a,b,c) = a+b+c : h(1,2,3)"  "6"
run "fn.compose"    "f(x) = x+1; g(x) = f(f(x)) : g(3)" "5"
run "fn.recursive"  "f(n) = if(n <= 1, 1, n * f(n-1)) : f(6)" "720"
run "fn.mutual_ref" "a(x) = x^2; b(x) = a(x) + 1 : b(4)" "17"

# ── Lambdas ───────────────────────────────────────────────────────────────────
section "LAMBDAS"
run "lambda.single"         "f = x -> x^2 : f(4)"                    "16"
run "lambda.multi"          "f = (x, y) -> x + y : f(3, 4)"          "7"
run "lambda.multi_bare"     "f = x, y -> x * y : f(3, 4)"            "12"
run "lambda.inline"         "(x -> x^2)(5)"                           "25"
run "lambda.inline_call"    "(x -> x + 1)(9)"                         "10"
run "lambda.as_arg"         "sum(x -> x, 1, 10)"                      "55"
run "lambda.closure"        "a=10; f = x -> x + a : f(5)"            "15"
run "lambda.nested"         "f = x -> (y -> x + y) : f(3)(4)"        "7"
run "lambda.in_sum"         "sum(x -> x^2, 1, 10)"                   "385"
run "lambda.in_prod"        "prod(x -> x, 1, 5)"                     "120"

# ── Blocks ────────────────────────────────────────────────────────────────────
section "BLOCKS"
run "block.simple"          "{x = 3; y = 4 : x + y}"                 "7"
run "block.colon_out"       "{a = 2; b = 3 : a * b}"                 "6"
run "block.isolation"       "x=99; {x = 1 : x}"                      "1"
run "block.fn_in_block"     "{f(x) = x^2 : f(5)}"                   "25"
run "block.as_expr"         "1 + {x=3 : x*2}"                        "7"
run "block.multi_out"       "{a=1; b=2 : (a, b)}"                    "(1, 2)"

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

# ── Tuples ────────────────────────────────────────────────────────────────────
section "TUPLES"
run "tup.create"        "(1, 2, 3)"                         "(1, 2, 3)"
run "tup.index0"        "(10, 20, 30)[0]"                   "10"
run "tup.index1"        "(10, 20, 30)[1]"                   "20"
run "tup.index_last"    "(10, 20, 30)[2]"                   "30"
run "tup.neg_index"     "(1, 2, 3)[-1]"                     "3"
run "tup.neg_index2"    "(1, 2, 3)[-2]"                     "2"
run "tup.multi_index"   "(10,20,30,40)[0,2]"                "(10, 30)"
run "tup.range_index"   "(1,2,3,4,5)[1..3]"                "(2, 3, 4)"
run "tup.nested"        "((1,2),(3,4))[1][0]"               "3"
run "tup.len"           "len((1,2,3,4,5))"                  "5"
run "tup.add"           "(1,2,3) + (4,5,6)"                 "(5, 7, 9)"
run "tup.sub"           "(5,6,7) - (1,2,3)"                 "(4, 4, 4)"
run "tup.scalar_mul"    "(1,2,3) * 3"                       "(3, 6, 9)"
run "tup.scalar_mul_l"  "3 * (1,2,3)"                       "(3, 6, 9)"
run "tup.scalar_div"    "(4,6,8) / 2"                       "(2, 3, 4)"
run "tup.scalar_pow"    "(1,2,3)^2"                         "(1, 4, 9)"
run "tup.scalar_add"    "(1,2,3) + 10"                      "(11, 12, 13)"
run "tup.eq"            "(1,2,3) == (1,2,3)"                "1"
run "tup.neq"           "(1,2,3) == (1,2,4)"                "0"
run "tup.neg"           "-(1,2,3)"                          "(-1, -2, -3)"
run "tup.fn_apply"      "f(x)=x^2 : f((1,2,3))"            "(1, 4, 9)"
run "tup.append"        "append((1,2,3), 4)"                "(1, 2, 3, 4)"
run "tup.concat"        "concat((1,2),(3,4))"               "(1, 2, 3, 4)"
run "tup.flatten"       "flatten(((1,2),(3,4)))"            "(1, 2, 3, 4)"
run "tup.zip"           "zip((1,2,3),(4,5,6))"              "((1, 4), (2, 5), (3, 6))"
run "tup.dot"           "dot((1,2,3),(4,5,6))"              "32"
run "tup.sort"          "sort((3,1,4,1,5,9))"               "(1, 1, 3, 4, 5, 9)"
run "tup.argmin"        "argmin((3,1,4,1,5))"               "1"
run "tup.argmax"        "argmax((3,1,4,1,5))"               "4"

# ── Aggregates on tuples ──────────────────────────────────────────────────────
section "AGGREGATES"
run "agg.sum_tuple"   "sum((1,2,3,4,5))"         "15"
run "agg.prod_tuple"  "prod((1,2,3,4))"           "24"
run "agg.sum_fn"      "sum(x -> x, 1, 100)"       "5050"
run "agg.prod_fn"     "prod(x -> x, 1, 10)"       "3628800"
run "agg.sum_x2"      "sum(x -> x^2, 1, 10)"      "385"
run "agg.map_sq"      "map(x -> x^2, (1,2,3,4))"  "(1, 4, 9, 16)"
run "agg.map_neg"     "map(x -> -x, (1,2,3))"     "(-1, -2, -3)"
run "agg.filter"      "filter(x -> x > 2, (1,2,3,4))" "(3, 4)"
run "agg.filter_none" "filter(x -> x > 9, (1,2,3))"   "()"
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
run "ho.partial_add"      "add5 = partial((x,y) -> x+y, 5) : add5(3)" "8"
run "ho.partial_builtin"  "sq = partial(pow, 2) : sq(10)"           "1024"
run "ho.map_partial"      "map(partial((x,y) -> x+y, 10), (1,2,3))"  "(11, 12, 13)"
run "ho.filter_partial"   "map(partial(pow,2), (1,2,3,4))"          "(2, 4, 8, 16)"

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
run "alg.step_neg"  "step(-1)"          "0"
run "alg.step_pos"  "step(1)"           "1"
run "alg.step_zero" "step(0)"           "0.5"
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
run "ls.basic"      "linspace(0,1,3)"       "(0, 0.5, 1)"
run "ls.one"        "linspace(0,10,1)"      "(0)"
run "ls.five"       "linspace(0,4,5)"       "(0, 1, 2, 3, 4)"
run "range.basic"   "range(0,5)"            "(0, 1, 2, 3, 4)"
run "range.zero"    "range(0,0)"            "()"
run "range.offset"  "range(3,7)"            "(3, 4, 5, 6)"

# ── rand ─────────────────────────────────────────────────────────────────────
section "RAND"
run_ok "rand.scalar"  "rand()"
run_ok "rand.range"   "rand(0, 1)"
run_ok "rand.tuple"   "rand(10)"

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
run "ten.matrix_fn"  "shape(matrix(3,4,(i,j)->0))"      "(3, 4)"
run "ten.matrix_val" "matrix(2,2,(i,j)->i*2+j)[0,1]"   "1"
run "ten.zeros_1d"   "zeros(4)"                         "[0, 0, 0, 0]"
run "ten.ones_1d"    "ones(3)"                          "[1, 1, 1]"
run "ten.zeros_3d"   "shape(zeros(2,3,4))"              "(2, 3, 4)"

# ── Tensor shape queries ──────────────────────────────────────────────────────
section "TENSORS - SHAPE"
run "ten.shape_2d"   "shape(eye(3))"         "(3, 3)"
run "ten.rows"       "rows(eye(4))"          "4"
run "ten.cols"       "cols(zeros(3,5))"      "5"
run "ten.len_1d"     "len(zeros(7))"         "7"
run "ten.len_2d"     "len(eye(4))"           "4"
run "ten.shape_1d"   "shape(zeros(5))"       "(5)"

# ── Tensor indexing ───────────────────────────────────────────────────────────
section "TENSORS - INDEXING"
run "ten.idx_1d"     "zeros(3)[1]"                         "0"
run "ten.idx_2d"     "eye(3)[1,1]"                         "1"
run "ten.idx_2d_off" "eye(3)[0,1]"                         "0"
run "ten.idx_row"    "matrix(2,3,(i,j)->i*3+j)[0]"         "[0, 1, 2]"
run "ten.idx_row2"   "matrix(2,3,(i,j)->i*3+j)[1]"         "[3, 4, 5]"
run "ten.neg_idx_1d" "ones(4)[-1]"                         "1"
run "ten.row_fn"     "row(eye(3), 1)"                      "(0, 1, 0)"
run "ten.col_fn"     "col(eye(3), 2)"                      "(0, 0, 1)"

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
run "ten.transpose"     "transpose(matrix(2,3,(i,j)->i*3+j))" "⎡ 0  3 ⎤ ⎢ 1  4 ⎥ ⎣ 2  5 ⎦"
run "ten.transpose_sq"  "transpose(eye(3))"                   "⎡ 1  0  0 ⎤ ⎢ 0  1  0 ⎥ ⎣ 0  0  1 ⎦"
run "ten.trace_eye"     "trace(eye(5))"                       "5"
run "ten.norm_eye3"     "norm(eye(3))"                        "1.7320508075688772"
run "ten.norm_ones"     "norm(ones(4))"                       "2"
run "ten.matmul_id"     "matmul(eye(2), eye(2))"              "⎡ 1  0 ⎤ ⎣ 0  1 ⎦"
run "ten.matmul_basic"  "matmul(matrix(1,2,(i,j)->j+1), matrix(2,1,(i,j)->i+1))" "⎡ 5 ⎤"
run "ten.matmul_2x2"    "trace(matmul(eye(3), ones(3,3)))"    "3"
run "ten.flatten"       "flatten(eye(2))"                     "(1, 0, 0, 1)"
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
run "cmpfn.map"   "map(partial(lt,3), (1,2,3,4,5))" "(0, 0, 0, 1, 1)"

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

# ── Chained / compound expressions ───────────────────────────────────────────
section "COMPOUND EXPRESSIONS"
run "comp.fn_chain"       "f(x)=x+1; g(x)=x^2 : g(f(3))"         "16"
run "comp.let_in_arg"     "a=3; b=4 : sqrt(a^2 + b^2)"            "5"
run "comp.block_in_fn"    "f(x) = {y=x^2 : y+1} : f(4)"           "17"
run "comp.tuple_in_fn"    "f(t) = t[0] + t[1] : f((3,4))"         "7"
run "comp.map_then_sum"   "sum(map(x -> x^2, (1,2,3,4)))"         "30"
run "comp.filter_sum"     "sum(filter(x -> x > 2, (1,2,3,4,5)))"  "12"
run "comp.compose_chain"  "f=x->x+1; g=x->x*2 : compose(f,g)(5)" "11"
run "comp.lambda_in_if"   "(if(1>0, x->x^2, x->x))(5)"           "25"

# ── Tensor literal syntax (1,2;3,4) ──────────────────────────────────────────
section "TENSOR LITERALS"
run "lit.2x2"        "(1,2; 3,4)"                         "⎡ 1  2 ⎤ ⎣ 3  4 ⎦"
run "lit.2x3"        "(1,2,3; 4,5,6)"                     "⎡ 1  2  3 ⎤ ⎣ 4  5  6 ⎦"
run "lit.shape"      "shape((1,2; 3,4))"                  "(2, 2)"
run "lit.index"      "(1,2; 3,4)[0,1]"                    "2"
run "lit.index2"     "(1,2; 3,4)[1,0]"                    "3"
run "lit.det"        "det((1,2; 3,4))"                    "-2"
run "lit.assign"     "M = (1,2; 3,4) : M[1,1]"           "4"
run "lit.1col"       "shape((1; 2; 3))"                   "(3, 1)"
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
run "solve.eye"      "solve(eye(2), (1,2))"               "(1, 2)"
run "solve.2x2"      "solve((2,0; 0,3), (4,9))"           "(2, 3)"
run_match "solve.3x3" "solve((1,2,3; 0,1,4; 5,6,0), (1,2,0))[0]" "^12|^11\\.9"
run_err "solve.singular" "solve((1,2; 2,4), (1,2))"

# ── hstack / vstack / tomat ───────────────────────────────────────────────────
section "HSTACK / VSTACK / TOMAT"
run "hst.shape"      "shape(hstack(eye(2), eye(2)))"      "(2, 4)"
run "hst.val"        "hstack(eye(2), eye(2))[0,2]"        "1"
run "vst.shape"      "shape(vstack(eye(2), eye(2)))"      "(4, 2)"
run "vst.val"        "vstack(eye(2), eye(2))[2,0]"        "1"
run "tomat.basic"    "shape(tomat((1,2,3,4), 2, 2))"      "(2, 2)"
run "tomat.val"      "tomat((1,2,3,4), 2, 2)[0,1]"        "2"
run_err "tomat.bad"  "tomat((1,2,3), 2, 2)"

# ── lingrid ───────────────────────────────────────────────────────────────────
section "LINGRID"
run "lg.shape"       "shape(lingrid((0,0),(1,1),(3,3),(x,y)->x+y))"  "(3, 3)"
run "lg.corner"      "lingrid((0,0),(1,1),(3,3),(x,y)->x+y)[0,0]"    "0"
run "lg.mid"         "lingrid((0,0),(1,1),(3,3),(x,y)->x+y)[1,1]"    "1"
run "lg.far"         "lingrid((0,0),(1,1),(3,3),(x,y)->x+y)[2,2]"    "2"
run "lg.sum"         "sum(lingrid((0,0),(1,1),(2,2),(x,y)->1))"       "4"

# ── 2D slice indexing ─────────────────────────────────────────────────────────
section "TENSOR SLICING"
run "sl.scalar_scalar" "eye(3)[1,1]"                      "1"
run "sl.row_slice"     "eye(3)[0, 0..1]"                  "[1, 0]"
run "sl.col_slice"     "eye(3)[0..1, 0]"                  "[1, 0]"
run "sl.submat_shape"  "shape(eye(3)[0..1, 0..1])"        "(2, 2)"
run "sl.submat_trace"  "trace(eye(4)[0..1, 0..1])"        "2"
run "sl.row1_slice"    "(1,2,3; 4,5,6)[1, 0..2]"          "[4, 5, 6]"
run "sl.col1_slice"    "(1,2,3; 4,5,6; 7,8,9)[0..2, 1]"  "[2, 5, 8]"

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
