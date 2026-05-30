[1;4mlingrid[0m

Evaluates a function over a uniform grid. start/end/counts can be scalars (1-D) or tuples (n-D).

[1mExamples:[0m
[0m> lingrid(0, 1, 5, x -> x^2)
result = [0, 0.0625, 0.25, 0.5625, 1]
> lingrid((-1,-1), (1,1), (3,3), (x,y) -> x^2 + y^2)
result = [2, 1.5, 2; 1, 0.5, 1; 2, 1.5, 2]
[0m
