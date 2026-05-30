## !print

Print a line of text with embedded expression interpolation. Use `{expr}` to embed any mathlang expression. Use `{{` / `}}` to escape literal braces.

**Usage:** `!print <text with {expr} interpolation>`

**Examples:**
```
> a = 3; b = 4
> !print hypotenuse = {sqrt(a^2 + b^2)}
hypotenuse = 5
> !print pi ≈ {round(pi, 5)} ({{approximately}})
pi ≈ 3.14159 (approximately)
```
