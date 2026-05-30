[1;4mcompose[0m

Composes two functions: creates a function that applies g then f.

[1mExamples:[0m
[0m> f = x -> x^2
> g = x -> x + 1
> h = compose(f, g)
> h(2)
result = 9
[0m
