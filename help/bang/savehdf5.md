## !savehdf5

Save a tensor variable to an HDF5 file. Optionally specify a dataset path, and use `--append`, `--overwrite`, or `--gzip` flags.

**Usage:** `!savehdf5 <var> <file> [/dataset] [--append] [--overwrite] [--gzip 0-9]`

**Examples:**
```
> !savehdf5 T data.h5
> !savehdf5 T data.h5 /results --gzip 6
> !savehdf5 T data.h5 /run2 --append
```
