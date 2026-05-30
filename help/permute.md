## permute

Reorders axes of a tensor according to a permutation.

**Examples:**
```
> T = ones(2, 3, 4)
> shape(permute(T, 2, 0, 1))
result = [4, 2, 3]
```
