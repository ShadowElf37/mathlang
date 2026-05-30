## !type

Show the type signature of a builtin, user-defined function, or any expression.

**Usage:** `!type <name>` | `!type <expr>`

**Examples:**
```
> !type sin
sin(x: num) -> num
> f(x: real) = x^2
> !type f
f(x: real) -> real
> !type 3+4i
num
```
