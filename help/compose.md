## compose

Composes two functions: creates a function that applies g then f.

**Examples:**
```
> f = x -> x^2
> g = x -> x + 1
> h = compose(f, g)
> h(2)
result = 9
```
