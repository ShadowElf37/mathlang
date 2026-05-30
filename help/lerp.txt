[1;4mlerp[0m

Linear interpolation: computes a*(1-t) + b*t. Works element-wise on tensors.

[1mExamples:[0m
[0m> lerp(0, 10, 0.5)
result = 5
> lerp((0, 0), (10, 10), 0.5)
result = [5, 5]
[0m
