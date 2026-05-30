## !savenpy

Save a tensor variable to a NumPy `.npy` file. Real tensors are saved as `f64`; complex tensors as `c128`.

**Usage:** `!savenpy <var> <file.npy>`

**Examples:**
```
> T = linspace(0, 1, 100)
> !savenpy T data.npy
saved T (100 elements) to data.npy
```
