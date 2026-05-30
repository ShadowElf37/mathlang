## !animate2D

Animate a 2D field and display it in the wgpu animator window. Accepts a pre-computed 3D tensor or a function of time. Uses T[x, y] axis convention.

**Usage:**
- `!animate2D T [fps]` — animate a [frames, nx, ny] tensor T
- `!animate2D f n [fps]` — call f(t) for t = 0..n-1; each call returns an [nx, ny] tensor
- `!animate2D f t0 t1 n [fps]` — call f(t) for t in linspace(t0, t1, n)

**Examples:**
```
> T = tensor((t,x,y) -> sin(x+t)*cos(y), 60, 50, 50)
> !animate2D T 30
> !animate2D t -> tensor((x,y) -> sin(x+t), 50, 50), 0, 2pi, 60, 24
```
