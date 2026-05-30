[1;4m!animate2D_raw[0m

Like [33m!animate2D[0m but writes MXFR frames to stdout instead of opening the animator. Useful for piping to an external viewer or recording.

[1mUsage:[0m
- [33m!animate2D_raw T[0m — stream a [frames, nx, ny] tensor as MXFR frames
- [33m!animate2D_raw f n[0m — stream f(t) for t = 0..n-1
- [33m!animate2D_raw f t0 t1 n[0m — stream f(t) over linspace(t0, t1, n)

[1mExamples:[0m
[2m> T = tensor((t,x,y) -> sin(x+t), 60, 50, 50)
> !animate2D_raw T | wgpu_animator
[0m
