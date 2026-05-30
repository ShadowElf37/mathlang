## !savetensor

Save a tensor variable to a binary `.mlt` file. Works for both real and complex tensors.

**Usage:** `!savetensor <var> <file>`

**Examples:**
```
> T = zeros(10, 10)
> !savetensor T data.mlt
saved T (100 elements, real) to data.mlt
```
