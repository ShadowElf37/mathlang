[1;4m!savehdf5[0m

Save a tensor variable to an HDF5 file. Optionally specify a dataset path, and use [33m--append[0m, [33m--overwrite[0m, or [33m--gzip[0m flags.

[1mUsage:[0m [33m!savehdf5 <var> <file> [/dataset] [--append] [--overwrite] [--gzip 0-9][0m

[1mExamples:[0m
[0m> !savehdf5 T data.h5
> !savehdf5 T data.h5 /results --gzip 6
> !savehdf5 T data.h5 /run2 --append
[0m
