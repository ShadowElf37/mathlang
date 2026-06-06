#!/usr/bin/env bash
# Smoke + parity tests for the mathlang-cubecl prototype (`mc`).
# Runs one-liners on the cpu backend and checks expected output; also checks the
# cross-backend precision behaviour. A fuller parity harness vs the original `m`
# lands in Phase 6.
set -u
cd "$(dirname "$0")/../.." || exit 1
cargo build -q -p mathlang-cubecl || { echo "build failed"; exit 1; }
MC=./target/debug/mc
pass=0; fail=0

# check <expr> <expected>   (runs on the default cpu/f64 target)
check() {
  local got; got=$("$MC" "$1" 2>&1)
  if [ "$got" = "$2" ]; then pass=$((pass+1));
  else fail=$((fail+1)); printf 'FAIL  %-34s got=[%s] want=[%s]\n' "$1" "$got" "$2"; fi
}

# check_repl <stdin> <substring>   (asserts the session output contains substring)
check_repl() {
  local got; got=$(printf '%s\n' "$1" | "$MC" 2>&1)
  if printf '%s' "$got" | grep -qF "$2"; then pass=$((pass+1));
  else fail=$((fail+1)); printf 'FAIL(repl) want substring [%s] in:\n%s\n' "$2" "$got"; fi
}

# ── scalar / complex / tuple core (Phase 1b) ───────────────────────────────────
check 'pi * 2^2'                 '12.566370614359172'
check '2pi'                      '6.283185307179586'
check 'x=3; y=4; sqrt(x^2+y^2)'  '5'
check 'i^2'                      '-1'
check 'exp(i*pi)'                '-1'
check 'sqrt(-1)'                 'i'
check 'abs(3+4i)'                '5'
check '(1,2,3) + (4,5,6)'        '(5, 7, 9)'
check 'sum(x -> x, 1, 100)'      '5050'
check 'iterate(x -> 2*x, 1, 10)' '1024'
check 'map(x -> x^2, (1,2,3,4))' '(1, 4, 9, 16)'

# ── tensors on the compute path (Phase 2) ──────────────────────────────────────
check '[1,2,3] + [4,5,6]'        '[5, 7, 9]'
check '[1,2,3] * 2'              '[2, 4, 6]'
check '2 * [1,2,3]'              '[2, 4, 6]'
check '[1,2,3,4] / 2'           '[0.5, 1, 1.5, 2]'
check '[1,2,3] ^ 2'             '[1, 4, 9]'
check '[1,2,3] > 2'             '[0, 0, 1]'
check 'sin(zeros(3))'           '[0, 0, 0]'
check 'sqrt([1,4,9,16])'        '[1, 2, 3, 4]'
check 'exp([0,1])'              '[1, 2.718281828459045]'
check 'linspace(0,1,5)'         '[0, 0.25, 0.5, 0.75, 1]'
check 'range(0,5)'              '[0, 1, 2, 3, 4]'
check 'shape(ones(2,3,4))'      '[2, 3, 4]'
check 'rows((1,2,3;4,5,6))'     '2'
check 'sum([1,2,3,4])'          '10'
check 'len(linspace(0,1,10))'   '10'
check '-[1,2,3]'                '[-1, -2, -3]'

# ── precision: native f64 on cpu; wgpu downgrades to f32; df64 storage round-trip
check '[1.0] + [1e-10]'         '[1.0000000001]'                  # cpu f64
if printf 'a\n' | "$MC" --spike >/dev/null 2>&1 || true; then :; fi
check_repl $'!backend wgpu\n[1.0] + [1e-10]' '[1]'                # wgpu f32 loses 1e-10
check_repl $'!backend wgpu\n!prec f64'       'no native f64'      # f64 rejected on wgpu
check_repl $'!backend wgpu\n!prec df64\n[0.5, 0.25]' '[0.5, 0.25]' # df64 storage round-trip

echo "-----"
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
