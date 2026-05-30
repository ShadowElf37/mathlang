## partial

Partially applies a function by fixing its first argument.

**Examples:**
```
> sub = (x, y) -> x - y
> sub5 = partial(sub, 5)
> sub5(3)
result = 2
```
