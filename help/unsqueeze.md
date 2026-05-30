## unsqueeze

Inserts a new dimension of size 1 at the specified axis position.

**Examples:**
```
> unsqueeze([1,2,3], 0)
result = [[1, 2, 3]]
> shape(unsqueeze([1,2,3], 0))
result = [1, 3]
```
