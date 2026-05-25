# m — command-line math

A fast expression evaluator written in Rust, hooked up to your shell as `m`.

## Syntax

```
m [definitions :] expression [, expression ...]
```

Definitions are comma-separated. A `:` or `;` separates them from the output expressions.
Without a separator, the whole input is evaluated as expression(s).

> **Always quote the argument** to avoid shell interference (`^` glob, `;` command separator, `*` glob, `>` redirect). The REPL needs no quoting.

---

## Quick examples

```zsh
m '3 + 4'                          # 7
m '2^10'                           # 1024
m 'pi * 2^2'                       # 12.566370614359172
m 'sqrt(2), sin(pi/2)'             # 1.4142135623730951  1
m 'x=3, y=4 : x^2 + y^2'          # 25
m 'x=3, y=4 : sqrt(x^2 + y^2)'    # 5
```

---

## Operators

| Operator | Meaning         |
|----------|-----------------|
| `+` `-` `*` `/` | arithmetic |
| `^` or `**` | exponentiation (right-associative) |
| `//`     | floor division  |
| `%`      | remainder       |

---

## Constants

`pi`, `e`, `tau`, `phi`, `inf`, `i` (imaginary unit)

---

## Defining variables

```zsh
m 'x = 3 : x^2'           # 9
m 'a=2, b=3 : a^2 + b^2'  # 13
```

---

## Defining functions

```zsh
m 'f(x) = x^2 : f(3), f(4)'      # 9  16
m 'g(x,y) = x^2 + y^2 : g(3,4)'  # 25
```

---

## Anonymous functions

Single-argument lambdas with `x -> expr`, multi-argument with `x, y -> expr` or `(x, y) -> expr`:

```zsh
m 'f = x -> x^2 : f(3)'                                  # 9
m 'ncr = n, r -> fact(n)/(fact(r)*fact(n-r)) : ncr(5,2)'  # 10
```

---

## sum and prod

`sum(f, start, stop)` and `prod(f, start, stop)` iterate over integers `[start, stop]`.
`f` can be an inline lambda or a named function:

```zsh
m 'sum(x -> x, 1, 100)'         # 5050
m 'sum(x -> x^2, 1, 10)'        # 385
m 'prod(x -> x, 1, 10)'         # 3628800  (10!)
m 'f(x) = x^3 : sum(f, 1, 5)'  # 225
```

---

## Complex numbers

`i` is the imaginary unit (√−1). Write complex literals as `3 + 2i`, `2i`, `-i`, etc.
All arithmetic operators (`+`, `-`, `*`, `/`, `^`) work on complex numbers.

```zsh
m 'i^2'                      # -1
m '(1 + i) * (1 - i)'        # 2
m 'exp(i * pi)'               # -1   (Euler's formula)
m 'exp(i * pi/2)'             # i
m 'sqrt(-1)'                  # i
m 'ln(-1)'                    # 3.141592653589793i  (= πi)
m 'abs(3 + 4i)'               # 5    (modulus)
m 'arg(i)'                    # 1.5707963267948966  (= π/2)
m 'conj(3 + 4i)'              # 3 - 4i
m 're(3 + 4i), im(3 + 4i)'   # 3  4
```

**Complex-capable builtins:** `abs`, `re`, `im`, `arg`, `conj`, `sqrt`, `exp`, `ln`

Note: trigonometric functions (`sin`, `cos`, …) and `log`/`log2` are real-only for now.

---

## Built-in functions

**Trig:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `sinh`, `cosh`, `tanh`

**Algebra:** `sqrt`, `cbrt`, `abs`, `sign`, `floor`, `ceil`, `round`, `exp`, `ln`, `log`/`log10`, `log2`, `pow(x,y)`, `min(a,b)`, `max(a,b)`, `hypot(a,b)`

**Complex:** `re(z)`, `im(z)`, `abs(z)`, `arg(z)`, `conj(z)`

**Number theory:** `gcd(a,b)`, `lcm(a,b)`, `fact(n)`, `delta(x)`

**Bitwise** (operate on 64-bit integers):
`and`, `or`, `xor`, `nand`, `nor`, `xnor`, `not`, `shl(x,n)`, `shr(x,n)`

```zsh
m 'shl(1, 8)'       # 256
m 'and(12, 10)'     # 8
m 'not(0)'          # -1  (bitwise NOT, two's complement)
m 'delta(0)'        # 1   (1 if x == 0, else 0)
```

---

## Multiple outputs

```zsh
m 'x=5 : x, x^2, x^3'       # 5  25  125
m 'sin(pi/6), cos(pi/3)'     # 0.5  0.5
```

---

## Init file

Definitions in `~/.mathlangrc` are loaded automatically when the REPL starts.
Set `MATHLANG_INIT=/path/to/file.m` to use a different file.
