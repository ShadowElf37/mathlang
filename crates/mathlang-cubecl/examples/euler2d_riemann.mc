# 2-D compressible Euler equations — Riemann problem (Schulz-Rinne configuration 3),
# solved with a Lax-Friedrichs finite-volume scheme on a periodic grid, then the
# density is animated through wgpu_animator.
#
#   Run:  mc crates/mathlang-cubecl/examples/euler2d_riemann.mc
#
#   The animator GUI must be available — build it once with
#       (cd animator && cargo build --release)
#   or point mc at a binary:  WGPU_ANIMATOR=/path/to/wgpu_animator
#
# State U = (rho, mx, my, E) with mx = rho*u, my = rho*v, E the total energy.
# Each component is an [N,N] tensor; the 4-tuple stays device-resident across all
# steps (uploaded once, the density movie downloaded once for streaming).

g  = 1.4                 # ratio of specific heats
N  = 128                 # grid points per axis
dx = 1.0 / N
dy = 1.0 / N
dt = 0.0015              # CFL ~ 0.5 for this configuration
nframes = 200
cx = dt / (2 * dx)
cy = dt / (2 * dy)

# Initial data: four constant states, one per quadrant of the unit square
# (top-right, top-left, bottom-left, bottom-right) — the classic config-3 values.
rho0 = tensor((i,j) -> if(i >= N/2, if(j >= N/2, 1.5, 0.5323), if(j >= N/2, 0.5323, 0.138)), N, N)
u0   = tensor((i,j) -> if(i >= N/2, 0.0, 1.206), N, N)
v0   = tensor((i,j) -> if(j >= N/2, 0.0, 1.206), N, N)
p0   = tensor((i,j) -> if(i >= N/2, if(j >= N/2, 1.5, 0.3), if(j >= N/2, 0.3, 0.029)), N, N)

mx0 = rho0 * u0
my0 = rho0 * v0
E0  = p0 / (g - 1) + 0.5 * rho0 * (u0*u0 + v0*v0)

# One Lax-Friedrichs step:
#   U <- avg(4 neighbours) - (dt/2dx)(F[i+1]-F[i-1]) - (dt/2dy)(G[j+1]-G[j-1])
# where F is the x-flux and G the y-flux of the Euler system. roll(.,±1,axis) is
# the periodic neighbour shift along x (axis 0) or y (axis 1).
step = (r, mx, my, E) -> {
  u = mx / r;
  v = my / r;
  p = (g - 1) * (E - 0.5 * (mx*mx + my*my) / r);
  F0 = mx;
  F1 = mx*u + p;
  F2 = mx*v;
  F3 = (E + p) * u;
  G0 = my;
  G1 = my*u;
  G2 = my*v + p;
  G3 = (E + p) * v;
  avg = q -> 0.25 * (roll(q,-1,0) + roll(q,1,0) + roll(q,-1,1) + roll(q,1,1));
  ddx = f -> roll(f,-1,0) - roll(f,1,0);
  ddy = h -> roll(h,-1,1) - roll(h,1,1);
  (avg(r)  - cx*ddx(F0) - cy*ddy(G0),
   avg(mx) - cx*ddx(F1) - cy*ddy(G1),
   avg(my) - cx*ddx(F2) - cy*ddy(G2),
   avg(E)  - cx*ddx(F3) - cy*ddy(G3))
}

# Evolve, stacking every frame. scan over a tuple state returns a tuple of stacks:
#   (rho_movie, mx_movie, my_movie, E_movie), each of shape [nframes+1, N, N].
# Animate the density component at 30 fps.
movie = scan(step, (rho0, mx0, my0, E0), nframes)
animate2D(movie[0], 30)
