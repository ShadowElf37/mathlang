## !animate2D_raw

Like `!animate2D` but writes MXFR frames to stdout instead of opening the animator. Useful for piping to an external viewer or recording.

**Usage:**
- `!animate2D_raw T` — stream a [frames, nx, ny] tensor as MXFR frames
- `!animate2D_raw f n` — stream f(t) for t = 0..n-1
- `!animate2D_raw f t0 t1 n` — stream f(t) over linspace(t0, t1, n)

**Examples:**
```
> T = tensor((t,x,y) -> sin(x+t), 60, 50, 50)
> !animate2D_raw T | wgpu_animator
```
