[1;4mpartial[0m

Partially applies a function by fixing its first argument.

[1mExamples:[0m
[2m> sub = (x, y) -> x - y
> sub5 = partial(sub, 5)
> sub5(3)
result = 2
[0m
