#!/usr/bin/env bash
# Test suite for mathlang. Each test runs `m <expr>` and checks output.
# Tests are grouped: DOCUMENTED (must pass per README) and EXPECTED (QoL we'd hope exists).

M="$HOME/mathlang/target/release/m"
PASS=0
FAIL=0
FAILS=()

# normalize whitespace for comparison
norm() { echo "$1" | tr -s ' \t\n' ' ' | sed 's/^ //; s/ $//'; }

run() {
    local label="$1"
    local expr="$2"
    local expected="$3"
    local got
    got=$("$M" "$expr" 2>&1)
    got_n=$(norm "$got")
    exp_n=$(norm "$expected")
    if [ "$got_n" = "$exp_n" ]; then
        PASS=$((PASS+1))
        # echo "PASS: $label"
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | expected='$exp_n' | got='$got_n'")
    fi
}

# Match if `got` contains `pat` (regex). Useful for floats with imprecise tails.
run_match() {
    local label="$1"
    local expr="$2"
    local pat="$3"
    local got
    got=$("$M" "$expr" 2>&1)
    if echo "$got" | grep -qE "$pat"; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | pattern='$pat' | got='$(norm "$got")'")
    fi
}

# Test passes if exit code is 0 (no error). Output content not checked.
run_ok() {
    local label="$1"
    local expr="$2"
    local got
    got=$("$M" "$expr" 2>&1)
    if [ $? -eq 0 ] && ! echo "$got" | grep -qiE 'error|unknown|undefined|fail|panic'; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILS+=("$label | expr='$expr' | got='$(norm "$got")'")
    fi
}

echo "=== DOCUMENTED FEATURES (from README) ==="

# basic arithmetic
run "arith.add" "3 + 4" "7"
run "arith.pow" "2^10" "1024"
run "arith.pow_star" "2**10" "1024"
run "arith.floordiv" "7 // 2" "3"
run "arith.mod" "7 % 3" "1"
run_match "arith.pi" "pi * 2^2" "12.56637"

# variables
run "var.simple" "x=3; y=4 : x^2 + y^2" "25"
run "var.sqrt" "x=3; y=4 : sqrt(x^2 + y^2)" "5"

# constants
run_match "const.e" "e" "2.718281"
run_match "const.phi" "phi" "1.618033"
run "const.inf" "inf" "inf"
run "const.i" "i^2" "-1"

# user functions
run "fn.two_arg" "g(x,y) = x^2 + y^2 : g(3,4)" "25"

# lambdas
run "lambda.single" "f = x -> x^2 : f(3)" "9"
run "lambda.multi" "ncr = n, r -> fact(n)/(fact(r)*fact(n-r)) : ncr(5,2)" "10"
run "lambda.inline" "(x -> x^2)(5)" "25"
run "lambda.in_sum" "sum(x -> x^2, 1, 10)" "385"

# tuples
run "tuple.index" "(1, 2, 3)[1]" "2"

# calculus
run_match "calc.integral_x2" "integral(x -> x^2, 0, 1)" "0.333"
run_match "calc.integral_sin" "integral(x -> sin(x), 0, pi)" "^2($|\\.0|\\.[0-9]*[0-9]$)|^1\\.999"
run_match "calc.deriv" "deriv(x -> x^3, 2)" "^12"

# aggregates
run "agg.sum_x" "sum(x -> x, 1, 100)" "5050"
run "agg.sum_x2" "sum(x -> x^2, 1, 10)" "385"
run "agg.prod_fact" "prod(x -> x, 1, 10)" "3628800"

# complex
run "complex.i2" "i^2" "-1"
run "complex.conj" "(1 + i) * (1 - i)" "2"
run_match "complex.euler" "exp(i * pi)" "^-1|^-0.999"
run "complex.sqrt_neg1" "sqrt(-1)" "i"
run "complex.abs" "abs(3 + 4i)" "5"
run_match "complex.arg" "arg(i)" "1.5707963"
run "complex.conj_fn" "conj(3 + 4i)" "3 - 4i"

# trig
run_match "trig.sin" "sin(pi/6)" "^0\\.5|^0\\.4999"
run_match "trig.cos" "cos(pi/3)" "^0\\.5|^0\\.4999|^0\\.5000"
run "trig.atan2" "atan2(1, 1)" "$(python3 -c 'import math;print(math.atan2(1,1))')"

# algebra fns
run "alg.sqrt" "sqrt(16)" "4"
run "alg.cbrt" "cbrt(27)" "3"
run "alg.abs" "abs(-7)" "7"
run "alg.sign_neg" "sign(-3)" "-1"
run "alg.floor" "floor(3.7)" "3"
run "alg.ceil" "ceil(3.2)" "4"
run "alg.round" "round(3.5)" "4"
run_match "alg.exp" "exp(1)" "2.71828"
run "alg.ln" "ln(e)" "1"
run "alg.log10" "log10(1000)" "3"
run "alg.log2" "log2(8)" "3"
run "alg.pow" "pow(2,10)" "1024"
run "alg.min" "min(3,7)" "3"
run "alg.max" "max(3,7)" "7"
run "alg.hypot" "hypot(3,4)" "5"

# number theory
run "nt.gcd" "gcd(12, 18)" "6"
run "nt.lcm" "lcm(4, 6)" "12"
run "nt.fact" "fact(5)" "120"
run "nt.delta0" "delta(0)" "1"
run "nt.delta_nz" "delta(3)" "0"

# bitwise
run "bit.shl" "shl(1, 8)" "256"
run "bit.shr" "shr(256, 4)" "16"
run "bit.and" "and(12, 10)" "8"
run "bit.or" "or(12, 10)" "14"
run "bit.xor" "xor(12, 10)" "6"

echo
echo "=== EXPECTED / QOL FEATURES (may not exist yet) ==="

# negative indexing
run "qol.tuple_neg_index" "(1, 2, 3)[-1]" "3"

# tuple length
run "qol.tuple_len" "len((1, 2, 3, 4))" "4"
run "qol.tuple_length" "length((1, 2, 3))" "3"

# range / linspace
run "qol.sum_tuple" "sum((1, 2, 3, 4))" "10"
run "qol.prod_tuple" "prod((1, 2, 3, 4))" "24"
run "qol.mean" "mean((1, 2, 3, 4, 5))" "3"
run "qol.max_tuple" "max((1, 5, 3, 2))" "5"
run "qol.min_tuple" "min((1, 5, 3, 2))" "1"

# reduce
run "qol.reduce" "reduce((a, b) -> a + b, (1, 2, 3, 4))" "10"

# comparisons / booleans
run "qol.cmp_lt" "3 < 5" "1"
run "qol.cmp_gt" "3 > 5" "0"
run "qol.cmp_eq" "3 == 3" "1"
run "qol.bool_and" "1 && 0" "0"
run "qol.bool_or" "1 || 0" "1"

# if
run "qol.if" "if(1 < 2, 10, 20)" "10"

# postfix factorial
run "qol.fact_postfix" "5!" "120"

# implicit multiplication
run_match "qol.implicit_mult_pi" "2pi" "6.283185"
run "qol.implicit_mult_paren" "2(3+4)" "14"

# scientific notation
run "qol.scientific" "1e3" "1000"
run_match "qol.scientific_neg" "1.5e-2" "0.015"

# log with base
run "qol.log_base" "log(8, 2)" "3"

# secondary trig
run "qol.sec" "sec(0)" "1"
run "qol.csc" "csc(pi/2)" "1"

# erf
run_match "qol.erf" "erf(0)" "^0"
run_match "qol.erf1" "erf(1)" "0.84270"

# sinc
run "qol.sinc0" "sinc(0)" "1"

# nested tuple indexing
run "qol.nested_tuple" "((1,2),(3,4))[0][1]" "2"

# tuple equality
run "qol.tuple_eq" "(1,2,3) == (1,2,3)" "1"

# dot product
run "qol.dot" "dot((1,2,3), (4,5,6))" "32"

# chained comparison
run "qol.chained_cmp" "1 < 2 < 3" "1"

# semicolon without colon
run "qol.semi_no_colon" "x=3; x^2" "9"

# negative power
run "qol.neg_pow" "2^-2" "0.25"

# paren lambda
run "qol.lambda_paren_multi" "f = (x, y) -> x + y : f(3, 4)" "7"

# print summary
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
