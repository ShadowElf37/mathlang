[1;4m!type[0m

Show the type signature of a builtin, user-defined function, or any expression.

[1mUsage:[0m [33m!type <name>[0m | [33m!type <expr>[0m

[1mExamples:[0m
[0m> !type sin
sin(x: num) -> num
> f(x: real) = x^2
> !type f
f(x: real) -> real
> !type 3+4i
num
[0m
