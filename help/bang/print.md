[1;4m!print[0m

Print a line of text with embedded expression interpolation. Use [33m{expr}[0m to embed any mathlang expression. Use [33m{{[0m / [33m}}[0m to escape literal braces.

[1mUsage:[0m [33m!print <text with {expr} interpolation>[0m

[1mExamples:[0m
[0m> a = 3; b = 4
> !print hypotenuse = {sqrt(a^2 + b^2)}
hypotenuse = 5
> !print pi ≈ {round(pi, 5)} ({{approximately}})
pi ≈ 3.14159 (approximately)
[0m
