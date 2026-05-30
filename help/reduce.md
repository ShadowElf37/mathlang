## reduce

Left-folds a 2-argument function over a tensor or tuple.

**Examples:**
```
> reduce((a,b) -> a+b, (1,2,3,4))
result = 10
> reduce((a,b) -> a*b, (1,2,3,4))
result = 24
```
