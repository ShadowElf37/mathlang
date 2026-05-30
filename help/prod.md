[1;4mprod[0m

Multiplies all elements of a tensor or computes a product over a function. With axis, reduces along that dimension.

[1mExamples:[0m
[0m> prod((1,2,3,4))
result = 24
> prod(ones(2,3))
result = 1
> prod(x -> x, 1, 5)
result = 120
[0m
