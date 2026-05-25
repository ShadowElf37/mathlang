ncr(n,r) = fact(n)/(fact(r) * fact(n-r))

# stirling approx
gamma(z) = exp(z * ln(z) - z + 0.5 * ln(2 * pi / z))

sin(x) = (exp(i*x) - exp(-i*x)) * 0.5
cos(x) = (exp(i*x) + exp(-i*x)) * 0.5
tan(x) = sin(x) / cos(x)
