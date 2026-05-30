## sum

Sums elements of a tensor or computes a sum over a function. With axis, reduces along that dimension.

**Examples:**
```
> sum((1,2,3,4))
result = 10
> sum(ones(2,3))
result = 6
> sum(x -> x^2, 1, 5)
result = 55
```
