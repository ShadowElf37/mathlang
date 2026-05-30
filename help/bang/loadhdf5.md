## !loadhdf5

Load a dataset from an HDF5 file into a variable. Use `--list` to list available datasets without loading.

**Usage:** `!loadhdf5 <var> <file> [/dataset] [--list]`

**Examples:**
```
> !loadhdf5 T data.h5
> !loadhdf5 T data.h5 /results
> !loadhdf5 _ data.h5 --list
```
