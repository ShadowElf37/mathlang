# m — command-line math

A fast expression evaluator written in Rust, hooked up to your shell as `m`.

## Syntax

```
m [definitions :] expression [, expression ...]
```

Definitions are separated by `;`. A `:` separates them from the output expressions.
Without a `:`, the whole input is evaluated as expression(s).

> **Always quote the argument** to avoid shell interference (`^` glob, `;` command separator, `*` glob, `>` redirect). The REPL needs no quoting.

---

## Quick examples

```zsh
m '3 + 4'                          # 7
m '2^10'                           # 1024
m 'pi * 2^2'                       # 12.566370614359172
m 'sqrt(2), sin(pi/2)'             # 1.4142135623730951  1
m 'x=3; y=4 : x^2 + y^2'          # 25
m 'x=3; y=4 : sqrt(x^2 + y^2)'    # 5
```

---

## Operators

| Operator | Meaning |
|----------|---------|
| `+` `-` `*` `/` | arithmetic |
| `^` or `**` | exponentiation (right-associative) |
| `//` | floor division |
| `%` | remainder |

---

## Constants

`pi`, `e`, `tau`, `phi`, `inf`, `i` (imaginary unit)

---

## Variables and functions

```zsh
m 'x = 3 : x^2'               # 9
m 'a=2; b=3 : a^2 + b^2'      # 13
m 'f(x) = x^2 : f(3), f(4)'   # 9  16
m 'g(x,y) = x^2 + y^2 : g(3,4)' # 25
```

---

## Anonymous functions (lambdas)

Single-argument: `x -> expr`. Multi-argument: `x, y -> expr` or `(x, y) -> expr`.

```zsh
m 'f = x -> x^2 : f(3)'                                   # 9
m 'ncr = n, r -> fact(n)/(fact(r)*fact(n-r)) : ncr(5,2)'  # 10
```

Lambdas are first-class — pass them to functions, apply them inline:

```zsh
m '(x -> x^2)(5)'         # 25
m 'sum(x -> x^2, 1, 10)'  # 385
```

---

## Implicit functions

Assigning an expression that contains free variables automatically creates a function:

```zsh
m 'f = x^2 + 1 : f(3)'        # 10  (f is stored as f(x) = x^2+1)
m 'h = x*y : h(3, 4)'         # 12  (h is stored as h(x,y) = x*y)
```

Free variables are captured in order of first appearance.

---

## Tuples

Comma-separated values in parentheses form a tuple. Index with `[n]` (zero-based).

```zsh
m '(1, 2, 3)[1]'            # 2
m 'x=3; y=4 : (x, y, x+y)' # 3  4  7
```

Arithmetic on tuples broadcasts element-wise:

```zsh
m '(1, 2, 3) * 2'            # 2  4  6
m '(1, 2, 3) + (4, 5, 6)'    # 5  7  9
m '(10, 20, 30) / 10'        # 1  2  3
```

Functions that return tuples compose naturally with tuple arithmetic:

```zsh
m 'pm(a,b) = (a+b, a-b) : pm(5, 3)'               # 8  2
m 'quadratic(a,b,c) = pm(-b, sqrt(b^2-4*a*c))/(2*a) : quadratic(1,-3,2)'  # 2  1
```

---

## Blocks

Blocks `{...}` create a local scope. Use `;` to separate definitions, `:` before the output expression(s).

```zsh
m '{disc = b^2 - 4*a*c; a=1; b=-5; c=6 : (-b+sqrt(disc))/(2*a), (-b-sqrt(disc))/(2*a)}'  # 3  2
```

Blocks can appear anywhere an expression is expected (inside function bodies, inline in CLI args, etc.).

---

## Calculus

```zsh
m 'integral(x -> x^2, 0, 1)'           # 0.3333333333333333
m 'integral(x -> sin(x), 0, pi)'       # 2
m 'integral(f, a, b, n)'               # Simpson's rule with n steps (default 1000)
m 'deriv(x -> x^3, 2)'                 # 12  (5-point stencil, dx=1e-5)
m 'deriv(f, x, dx)'                    # custom step size
```

---

## Aggregates

```zsh
m 'sum(x -> x, 1, 100)'        # 5050
m 'sum(x -> x^2, 1, 10)'       # 385
m 'prod(x -> x, 1, 10)'        # 3628800  (10!)
m 'f(x) = x^3 : sum(f, 1, 5)' # 225
```

`map(f, tuple)` applies `f` to each element:

```zsh
m 'map(x -> x^2, (1,2,3,4))'   # 1  4  9  16
```

---

## Complex numbers

`i` is the imaginary unit (√−1). Write complex literals as `3 + 2i`, `2i`, `-i`, etc.
All arithmetic operators work on complex numbers.

```zsh
m 'i^2'                      # -1
m '(1 + i) * (1 - i)'        # 2
m 'exp(i * pi)'               # -1   (Euler's formula)
m 'sqrt(-1)'                  # i
m 'ln(-1)'                    # 3.141592653589793i  (= πi)
m 'abs(3 + 4i)'               # 5    (modulus)
m 'arg(i)'                    # 1.5707963267948966  (= π/2)
m 'conj(3 + 4i)'              # 3 - 4i
m 're(3 + 4i), im(3 + 4i)'   # 3  4
```

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

## REPL

Run `m` with no arguments to enter the interactive REPL. It supports syntax highlighting, tab completion, and history hinting.

```
> x = 3
> f(n) = n^2
> f(x)
result = 9
> result + 1
result = 10
```

**REPL commands:**

| Command | Effect |
|---------|--------|
| `!help` | show syntax reference |
| `!defs` | list all user-defined names |
| `!import <file>` | load a `.m` file into the current session |
| `!clear` | clear all user definitions |
| `q` / `exit` | quit |

---

## Init file and `.m` files

Definitions in `~/.mathlangrc` are loaded automatically when the REPL starts.
Set `MATHLANG_INIT=/path/to/file.m` to use a different file.

`.m` files are plain text, one definition per line. Lines starting with `#` are comments.

### `advanced.m`

An included library (`advanced.m`) provides combinatorics, polynomial solvers, and more:

```
ncr(n,r)              — binomial coefficient
gamma(z)              — Stirling approximation of Γ(z)
sin/cos/tan(x)        — complex-capable trig via Euler's formula
quadratic(a,b,c)      — roots of ax²+bx+c=0  (returns tuple)
solveCubic(a,b,c,d)   — 3 real roots of ax³+… (trig method, disc > 0)
cubicRoot(a,b,c,d)    — 1 real root  of ax³+… (Cardano, disc < 0)
solveQuartic(a,b,c,d,e) — 4 real roots of ax⁴+… (Ferrari's method)
```

Load it with `!import advanced.m` in the REPL (or set it as your init file).
