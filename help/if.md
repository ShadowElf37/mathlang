## if

Conditional evaluation: returns a if cond is nonzero, otherwise returns b. Branches are evaluated lazily.

**Examples:**
```
> if(1, 10, 20)
result = 10
> if(0, 10, 20)
result = 20
> x = 5; if(x > 0, sqrt(x), 0)
result = 2.236...
```
