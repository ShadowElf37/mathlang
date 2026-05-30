## lerp

Linear interpolation: computes a*(1-t) + b*t. Works element-wise on tensors.

**Examples:**
```
> lerp(0, 10, 0.5)
result = 5
> lerp((0, 0), (10, 10), 0.5)
result = [5, 5]
```
