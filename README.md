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

`pi`, `e`, `phi`, `inf`, `i` (imaginary unit)

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

## Tuples

Comma-separated values in parentheses form a tuple. Index with `[n]` (zero-based; negative indices count from the end).

```zsh
m '(1, 2, 3)[1]'            # 2
m '(1, 2, 3)[-1]'           # 3
m 'x=3; y=4 : (x, y, x+y)' # (3, 4, 7)
```

Arithmetic on tuples broadcasts element-wise:

```zsh
m '(1, 2, 3) * 2'            # (2, 4, 6)
m '(1, 2, 3) + (4, 5, 6)'    # (5, 7, 9)
m '(10, 20, 30) / 10'        # (1, 2, 3)
```

Functions that return tuples compose naturally with tuple arithmetic:

```zsh
m 'pm(a,b) = (a+b, a-b) : pm(5, 3)'               # (8, 2)
m 'quadratic(a,b,c) = pm(-b, sqrt(b^2-4*a*c))/(2*a) : quadratic(1,-3,2)'  # (2, 1)
```

### Tuple indexing

```zsh
m '(1,2,3,4,5)[1..3]'       # (2, 3, 4)   — inclusive range
m '(10,20,30,40)[0,2]'      # (10, 30)    — pick by index list
m 'len((1,2,3,4))'          # 4
```

---

## Blocks

Blocks `{...}` create a local scope. Use `;` to separate definitions, `:` before the output expression(s).

```zsh
m '{x = 3; y = 4 : x^2 + y^2}'   # 25
```

Blocks can appear anywhere an expression is expected (inside function bodies, inline in CLI args, etc.).

---

## Comparisons

Comparison operators return `1` (true) or `0` (false):

```zsh
m '3 < 5'      # 1
m '3 > 5'      # 0
m '3 == 3'     # 1
m '3 != 4'     # 1
m '3 <= 3'     # 1
```

Combined with `if`:

```zsh
m 'x=4 : if(x > 0, sqrt(x), 0)'   # 2
```

---

## `if`

`if(cond, a, b)` returns `a` when `cond` is nonzero, `b` otherwise. Branches are evaluated lazily.

```zsh
m 'if(1, 10, 20)'               # 10
m 'if(0, 10, 20)'               # 20
```

`if` works with functions, enabling piecewise definitions:

```zsh
m 'f = x -> if(x >= 0, sqrt(x), 0) : f(4), f(-1)'   # 2  0
m 'sign2 = x -> if(x > 0, 1, if(x < 0, -1, 0))'
```

`if(cond, fn1, fn2)(x)` evaluates the chosen function on `x`:

```zsh
m 'if(1, x -> x^2, x -> x^3)(5)'   # 25
```

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
m 'sum((1, 2, 3, 4))'          # 10       — sum over a tuple
m 'prod((1, 2, 3, 4))'         # 24
```

`map(f, tuple)` applies `f` to each element:

```zsh
m 'map(x -> x^2, (1,2,3,4))'   # (1, 4, 9, 16)
```

`filter(f, tuple)` keeps elements where `f` returns nonzero:

```zsh
m 'filter(x -> x > 2, (1,2,3,4))'   # (3, 4)
```

`reduce(f, tuple)` left-folds with a 2-arg function:

```zsh
m 'reduce((a,b) -> a+b, (1,2,3,4))'   # 10
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

**Trig:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `sinh`, `cosh`, `tanh`, `sec`, `csc`, `cot`

**Algebra:** `sqrt`, `cbrt`, `abs`, `sign`, `step`, `floor`, `ceil`, `round`, `round(x,n)`, `trunc`, `frac`, `exp`, `ln`, `log`/`log10`, `log(x,base)`, `log2`, `pow(x,y)`, `min`, `max`, `hypot(a,b)`

**Angle:** `deg(x)` (radians→degrees), `rad(x)` (degrees→radians)

**Complex:** `re(z)`, `im(z)`, `abs(z)`, `arg(z)`, `conj(z)`

**Number theory:** `gcd(a,b)`, `lcm(a,b)`, `fact(n)`, `delta(x)`

**Special:** `sinc`, `sech`, `csch`, `erf`, `erfc`, `j0`, `j1`, `jinc`, `gaussian(x,mu,sigma)`

**Statistics (on tuples):** `mean`, `median`, `mode`, `std`, `var`, `min`, `max`, `sum`, `prod`

**Tuple ops:** `len`, `sort`, `zip(a,b)`, `dot(a,b)`, `append(t,x)`, `concat(a,b)`, `flatten(t)`, `argmin(t)`, `argmax(t)`, `linspace(a,b,n)`, `range(a,b)`

**Higher-order:** `map(f,t)`, `filter(f,t)`, `reduce(f,t)`, `compose(f,g)`, `partial(f,a)`

**Random:** `rand()`, `rand(a,b)`

**Bitwise** (operate on 64-bit integers):
`and`, `or`, `xor`, `nand`, `nor`, `xnor`, `not`, `shl(x,n)`, `shr(x,n)`

```zsh
m 'shl(1, 8)'       # 256
m 'and(12, 10)'     # 8
m 'not(0)'          # -1  (bitwise NOT, two's complement)
m 'delta(0)'        # 1   (1 if x == 0, else 0)
m 'step(-1)'        # 0   (Heaviside: 0 / 0.5 / 1)
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
| `!import <file>` | load a `.math` file into the current session |
| `!clear` | clear all user definitions |
| `q` / `exit` | quit |

---

## Init file and `.math` files

Definitions in `~/.mathlangrc` are loaded automatically when the REPL starts.
Set `MATHLANG_INIT=/path/to/file.math` to use a different file.

`.math` files are plain text, one definition per line. Lines starting with `#` are comments.

### `advanced.math`

An included library (`advanced.math`) provides combinatorics, polynomial solvers, and more:

```
ncr(n,r)              — binomial coefficient
gamma(z)              — Stirling approximation of Γ(z)
quadratic(a,b,c)      — roots of ax²+bx+c=0  (returns tuple)
solveCubic(a,b,c,d)   — 3 real roots of ax³+… (trig method, disc > 0)
cubicRoot(a,b,c,d)    — 1 real root  of ax³+… (Cardano, disc < 0)
solveQuartic(a,b,c,d,e) — 4 real roots of ax⁴+… (Ferrari's method)
```

Load it with `!import advanced.math` in the REPL (or set it as your init file).

### `physics.math`

Physical constants in SI units (`c`, `g`, `G`, `k_B`, `N_A`, `R`, `h`, `hbar`, etc.) plus `oscillator`, `oscillatorMBK`, `heatKernel`, `heatSolutionBounded`.

### `conversions.math`

Unit conversion factors and functions for physics, chemistry, and engineering. Load with `!import conversions.math`.

```zsh
m '!import conversions.math'
> 1 * eV_to_J          # 1.602176634e-19
> C_to_K(100)          # 373.15
> 1 * atm_to_Pa        # 101325
```
