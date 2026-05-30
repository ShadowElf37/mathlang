## !graph

Plot a function over a range and open the result in the animator. The range defaults to [-10, 10] if omitted.

**Usage:** `!graph f` | `!graph f, a, b`

**Examples:**
```
> !graph sin
(plots sin over [-10, 10])
> !graph x -> x^2, -5, 5
(plots x² over [-5, 5])
> g(x) = exp(-x^2)
> !graph g, -3, 3
```
