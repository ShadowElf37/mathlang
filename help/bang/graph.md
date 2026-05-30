[1;4m!graph[0m

Plot a function over a range and open the result in the animator. The range defaults to [-10, 10] if omitted.

[1mUsage:[0m [33m!graph f[0m | [33m!graph f, a, b[0m

[1mExamples:[0m
[2m> !graph sin
(plots sin over [-10, 10])
> !graph x -> x^2, -5, 5
(plots x² over [-5, 5])
> g(x) = exp(-x^2)
> !graph g, -3, 3
[0m
