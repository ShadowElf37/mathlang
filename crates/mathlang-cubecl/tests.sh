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

# ── linear algebra + reductions (Phase 3, on device) ───────────────────────────
check '(1,2;3,4) @ [1,1]'       '[3, 7]'           # mat·vec
check '[1,1] @ (1,2;3,4)'       '[4, 6]'           # vec·mat
check '[1,2,3] @ [4,5,6]'       '32'               # dot → scalar
check 'sum([1,2,3,4,5])'        '15'
check 'prod([1,2,3,4])'         '24'
check 'mean([1,2,3,4])'         '2.5'
check 'min([3,1,4,1,5])'        '1'
check 'max([3,1,4,1,5])'        '5'
check 'norm([3,4])'             '5'
check 'std([2,4,4,4,5,5,7,9])'  '2'
check 'sum(ones(100,100))'      '10000'
check_repl $'!prec df64\n(1,2;3,4) @ (5,6;7,8)' 'df64 matmul is staged'

# ── complex tensors (Phase 5) ───────────────────────────────────────────────────
check '[1+2i, 3+4i]'            '[1 + 2i, 3 + 4i]'
check '[1, 2, 3] + 2i'          '[1 + 2i, 2 + 2i, 3 + 2i]'   # real tensor + complex scalar
check '[1+1i] * [1+1i]'         '[2i]'
check 'abs([3+4i, 5+12i])'      '[5, 13]'
check 're([1+2i, 3+4i])'        '[1, 3]'
check 'conj([1+2i, 3-4i])'      '[1 - 2i, 3 + 4i]'
check 'sqrt([3+4i])'            '[2 + i]'
check 'exp([0+0i, 0+3.141592653589793i])' '[1, -1]'           # display collapses tiny im
check 'sum([1+2i, 3+4i, 5+6i])' '9 + 12i'
check_repl $'!type [1+2i, 3]'   'complex tensor'

# ── indexing/slicing, constructors, assembly, branching, axis reductions ────────
check 'A=(1,2,3;4,5,6;7,8,9); A[1,2]'    '6'
check 'A=(1,2,3;4,5,6;7,8,9); A[0, ..]'  '[1, 2, 3]'
check 'A=(1,2,3;4,5,6;7,8,9); A[.., 0]'  '[1, 4, 7]'
check '[10,20,30,40,50][1..3]'           '[20, 30, 40]'
check 'lingrid(0, 1, 5, x -> x^2)'       '[0, 0.0625, 0.25, 0.5625, 1]'
check 'shape(tensor((i,j) -> i+j, 3, 4))' '[3, 4]'
check 'reshape([1,2,3,4,5,6], 2, 3)'     "$($MC '(1,2,3;4,5,6)')"
check 'transpose((1,2,3;4,5,6))'         "$($MC '(1,4;2,5;3,6)')"
check 'vstack((1,2;3,4),(5,6))'          "$($MC '(1,2;3,4;5,6)')"
check 'hstack((1,2),(3,4))'              "$($MC '(1,3;2,4)')"
check 'sign([-2, 0, 3])'                 '[-1, 0, 1]'
check 'max([1,5,2], [3,1,4])'            '[3, 5, 4]'
check 'select([1,0,1], [10,20,30], [-1,-2,-3])' '[10, -2, 30]'
check 'sum((1,2,3;4,5,6), 0)'            '[5, 7, 9]'
check 'sum((1,2,3;4,5,6), 1)'            '[6, 15]'

# ── stencils + dense linalg (Phase 3.x) ─────────────────────────────────────────
check 'roll([1,2,3,4], 1, 0)'   '[4, 1, 2, 3]'
check 'roll([1,2,3,4], -1, 0)'  '[2, 3, 4, 1]'
check 'shift([1,2,3,4], 1, 0)'  '[1, 1, 2, 3]'
check 'ops.lap([0,0,1,0,0], 1)' '[0, 1, -2, 1, 0]'
check 'ops.grad([1,4,9,16,25], 1, 0)' '[-10.5, 4, 6, 8, -7.5]'
check 'det((1,2;3,4))'          '-2'
check 'det((2,0,0;0,3,0;0,0,4))' '24'
check 'solve((2,1;1,3),(5,10))' '[1, 3]'
check 'solve((2,1;1,3),[5,10])' '[1, 3]'
# heat equation: total heat conserved under Neumann (≈ 9)
check 'round(sum(iterate(u -> u + 0.2*ops.lap(u,1,ops.neumann), (0,0,0;0,9,0;0,0,0), 10)))' '9'
check_repl $'eigvals((2,0;0,3))' 'staged'

# ── resident loops: iterate / scan (Phase 4) ───────────────────────────────────
check 'iterate(x -> 2*x, 1, 10)'        '1024'                    # scalar
check 'iterate(u -> u*0.5, [1,2,3,4], 3)' '[0.125, 0.25, 0.375, 0.5]'  # tensor, resident
check 'iterate((u,v) -> (v, u), ([1,2],[3,4]), 1)' '([3, 4], [1, 2])'  # tuple of tensors
check 'scan(x -> 2*x, 1, 4)'            '[1, 2, 4, 8, 16]'        # scalar → 1-D
check 'scan(x -> x+1, 0, 3)'            '[0, 1, 2, 3]'
check 'shape(scan(u -> u*0.5, [1,2,3], 5))' '[6, 3]'             # tensor → [n+1, d]
check 'shape(scan(v -> (v[1], -v[0]), (1,0), 4))' '[5, 2]'      # flat tuple → [n+1, k]

# ── precision: f64 on cpu; wgpu downgrades to f32; df64 arithmetic on cpu ───────
check '[1.0] + [1e-10]'         '[1.0000000001]'                  # cpu f64
check_repl $'!backend wgpu\n[1.0] + [1e-10]' '[1]'                # wgpu f32 loses 1e-10
check_repl $'!backend wgpu\n!prec f64'       'no native f64'      # f64 rejected on wgpu
check_repl $'!backend wgpu\n!prec df64\n[0.5, 0.25]' '[0.5, 0.25]' # df64 storage round-trip

# df64 arithmetic (double-single) is correct on the IEEE cpu backend
check_repl $'!prec df64\n[1.0] + [1e-10]' '1.0000000001'         # add keeps the lo term
check_repl $'!prec df64\n[1.0] / [3.0]'   '0.33333333333333'     # ~16 digits, not f32's 0.3333333
check_repl $'!prec df64\n[0.1] * [0.1]'   '0.0100000000000000'   # df64 product
# df64 arithmetic is gated off wgpu/Metal (driver fast-math), not silently wrong
check_repl $'!backend wgpu\n!prec df64\n[1.0] + [1.0]' 'unreliable on wgpu'

echo "-----"
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
