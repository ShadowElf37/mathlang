[1;4msum[0m

Sums elements of a tensor or computes a sum over a function. With axis, reduces along that dimension.

[1mExamples:[0m
[2m> sum((1,2,3,4))
result = 10
> sum(ones(2,3))
result = 6
> sum(x -> x^2, 1, 5)
result = 55
[0m
