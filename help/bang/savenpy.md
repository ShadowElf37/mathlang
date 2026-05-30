[1;4m!savenpy[0m

Save a tensor variable to a NumPy [33m.npy[0m file. Real tensors are saved as [33mf64[0m; complex tensors as [33mc128[0m.

[1mUsage:[0m [33m!savenpy <var> <file.npy>[0m

[1mExamples:[0m
[2m> T = linspace(0, 1, 100)
> !savenpy T data.npy
saved T (100 elements) to data.npy
[0m
