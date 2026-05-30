## squeeze

Removes dimensions of size 1 from a tensor.

**Examples:**
```
> squeeze(zeros(1, 3, 1))
result = [0, 0, 0]
> shape(squeeze(zeros(1, 3, 1)))
result = [3]
```
