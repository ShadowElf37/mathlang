// Magnetostatic demag kernel — a faithful port of mumax3's mag/demagkernel.go
// (brute-force integration of magnetic surface charges over cell faces, averaged
// over destination cell volumes). Geometry-only: depends on grid + cell size, not
// on the magnetization, so it is computed once and reused for the FFT convolution.
//
// Returns the six unique components of the symmetric demag tensor N (dimensionless)
// as [xx, xy, xz, yy, yz, zz], each of length Kz*Ky*Kx, in mumax's "wrapped" FFT
// layout (index 0 = zero offset; negative offsets wrap to the top). For a 2-D
// problem (nz=1) the xz and yz components are identically zero.

const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;
const ACCURACY: f64 = 6.0; // mumax default DemagAccuracy
const SMALL_N: usize = 10;

fn delta(d: i64) -> f64 {
    let mut d = d.abs();
    if d > 0 { d -= 1; }
    d as f64
}

fn wrap(mut n: i64, m: i64) -> i64 {
    while n < 0 { n += m; }
    while n >= m { n -= m; }
    n
}

// Padded kernel size (non-periodic): in-plane axes double; a thin (small) Z axis
// uses minimal padding.
fn pad_size(size_in: [usize; 3]) -> [usize; 3] {
    let mut p = [0usize; 3];
    for i in 0..3 {
        p[i] = if i != Z || size_in[i] > SMALL_N { size_in[i] * 2 }
               else { (size_in[i] * 2).saturating_sub(1).max(1) };
    }
    p
}

/// Compute the demag tensor. Returns (components, [Kz, Ky, Kx]).
/// components[0..6] = xx, xy, xz, yy, yz, zz.
pub fn demag_kernel(size_in: [usize; 3], cellsize: [f64; 3]) -> (Vec<Vec<f64>>, [usize; 3]) {
    let size = pad_size(size_in);
    let (kx, ky, kz) = (size[X], size[Y], size[Z]);
    let n = kx * ky * kz;
    // array[i][j], i<=j -> 6 unique; index into (z,y,x) as (z*ky + y)*kx + x
    let mut arr: Vec<Vec<f64>> = vec![vec![0.0; n]; 6];
    // map (i,j) with i<=j to a slot 0..6
    let slot = |i: usize, j: usize| -> usize {
        match (i, j) {
            (0, 0) => 0, (0, 1) => 1, (0, 2) => 2,
            (1, 1) => 3, (1, 2) => 4, (2, 2) => 5,
            _ => unreachable!(),
        }
    };
    let idx = |z: usize, y: usize, x: usize| (z * ky + y) * kx + x;

    // smallest cell dimension = typical length scale
    let l = cellsize[X].min(cellsize[Y]).min(cellsize[Z]);

    // integration ranges (non-periodic)
    let r1 = [-(((size[X] - 1) / 2) as i64), -(((size[Y] - 1) / 2) as i64), -(((size[Z] - 1) / 2) as i64)];
    let r2 = [((size[X] - 1) / 2) as i64, ((size[Y] - 1) / 2) as i64,
              if size[Z] == 1 { 0 } else { ((size[Z] - 1) / 2) as i64 }];

    for s in 0..3 {
        let (u, v, w) = (s, (s + 1) % 3, (s + 2) % 3);
        let mut r_pos = [0.0f64; 3];
        for zc in r1[Z]..=r2[Z] {
            let zw = wrap(zc, size[Z] as i64);
            if zw > (size[Z] / 2) as i64 { continue; }
            r_pos[Z] = zc as f64 * cellsize[Z];
            for yc in r1[Y]..=r2[Y] {
                let yw = wrap(yc, size[Y] as i64);
                if yw > (size[Y] / 2) as i64 { continue; }
                r_pos[Y] = yc as f64 * cellsize[Y];
                for xc in r1[X]..=r2[X] {
                    let xw = wrap(xc, size[X] as i64);
                    if xw > (size[X] / 2) as i64 { continue; }
                    r_pos[X] = xc as f64 * cellsize[X];

                    // number of integration points from distance/accuracy
                    let dx = delta(xc) * cellsize[X];
                    let dy = delta(yc) * cellsize[Y];
                    let dz = delta(zc) * cellsize[Z];
                    let mut d = (dx * dx + dy * dy + dz * dz).sqrt();
                    if d == 0.0 { d = l; }
                    let max_size = d / ACCURACY;
                    let npt = |c: usize| ((cellsize[c] / max_size).max(1.0) + 0.5) as usize;
                    let nv = npt(v) * 2;
                    let nw = npt(w) * 2;
                    let nx = npt(X);
                    let ny = npt(Y);
                    let nz = npt(Z);

                    let scale = 1.0 / (nv * nw * nx * ny * nz) as f64;
                    let surface = cellsize[v] * cellsize[w];
                    let charge = surface * scale;
                    let pu1 = cellsize[u] / 2.0;
                    let pu2 = -pu1;

                    let mut b = [0.0f64; 3];
                    let mut pole = [0.0f64; 3];
                    for i in 0..nv {
                        pole[v] = -(cellsize[v] / 2.0) + cellsize[v] / (2.0 * nv as f64) + i as f64 * (cellsize[v] / nv as f64);
                        for j in 0..nw {
                            pole[w] = -(cellsize[w] / 2.0) + cellsize[w] / (2.0 * nw as f64) + j as f64 * (cellsize[w] / nw as f64);
                            for a in 0..nx {
                                let rx = r_pos[X] - cellsize[X] / 2.0 + cellsize[X] / (2.0 * nx as f64) + (cellsize[X] / nx as f64) * a as f64;
                                for be in 0..ny {
                                    let ry = r_pos[Y] - cellsize[Y] / 2.0 + cellsize[Y] / (2.0 * ny as f64) + (cellsize[Y] / ny as f64) * be as f64;
                                    for ga in 0..nz {
                                        let rz = r_pos[Z] - cellsize[Z] / 2.0 + cellsize[Z] / (2.0 * nz as f64) + (cellsize[Z] / nz as f64) * ga as f64;

                                        pole[u] = pu1;
                                        let r2x = rx - pole[X];
                                        let r2y = ry - pole[Y];
                                        let r2z = rz - pole[Z];
                                        let mut r = (r2x * r2x + r2y * r2y + r2z * r2z).sqrt();
                                        let mut qr = charge / (4.0 * std::f64::consts::PI * r * r * r);
                                        let (bx, by, bz) = (r2x * qr, r2y * qr, r2z * qr);

                                        pole[u] = pu2;
                                        let r2x = rx - pole[X];
                                        let r2y = ry - pole[Y];
                                        let r2z = rz - pole[Z];
                                        r = (r2x * r2x + r2y * r2y + r2z * r2z).sqrt();
                                        qr = -charge / (4.0 * std::f64::consts::PI * r * r * r);
                                        b[X] += bx + r2x * qr;
                                        b[Y] += by + r2y * qr;
                                        b[Z] += bz + r2z * qr;
                                    }
                                }
                            }
                        }
                    }
                    for dd in s..3 {
                        arr[slot(s, dd)][idx(zw as usize, yw as usize, xw as usize)] += b[dd];
                    }
                }
            }
        }
    }

    // Reconstruct skipped halves from symmetry (X, then Y, then Z).
    // sign per component under reflection of each axis.
    // components order: 0=xx,1=xy,2=xz,3=yy,4=yz,5=zz
    let reflect_x = [1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let reflect_y = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let reflect_z = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    for z in 0..kz {
        for y in 0..ky {
            for x in (kx / 2 + 1)..kx {
                let x2 = kx - x;
                for c in 0..6 { arr[c][idx(z, y, x)] = reflect_x[c] * arr[c][idx(z, y, x2)]; }
            }
        }
    }
    for z in 0..kz {
        for y in (ky / 2 + 1)..ky {
            let y2 = ky - y;
            for x in 0..kx {
                for c in 0..6 { arr[c][idx(z, y, x)] = reflect_y[c] * arr[c][idx(z, y2, x)]; }
            }
        }
    }
    for z in (kz / 2 + 1)..kz {
        let z2 = kz - z;
        for y in 0..ky {
            for x in 0..kx {
                for c in 0..6 { arr[c][idx(z, y, x)] = reflect_z[c] * arr[c][idx(z2, y, x)]; }
            }
        }
    }

    // 2-D: the out-of-plane cross terms vanish.
    if size[Z] == 1 {
        for v in arr[2].iter_mut() { *v = 0.0; }
        for v in arr[4].iter_mut() { *v = 0.0; }
    }

    (arr, [kz, ky, kx])
}

// ── Compiled demag field (FFT convolution, kernel FFTs cached) ──────────────────
// Replaces the interpreted .math pad/crop/FFT orchestration (which evaluated the
// padding lambda per cell, ~10^8 interpreted calls for a std4 run). Everything here
// is compiled; the geometry-only kernel FFTs are cached across time steps.

use std::cell::RefCell;
use crate::eval::fft_axis_inplace;

struct DemagCache {
    key: u64,
    kx: usize,
    ky: usize,
    xx: (Vec<f64>, Vec<f64>),
    xy: (Vec<f64>, Vec<f64>),
    yy: (Vec<f64>, Vec<f64>),
    zz: (Vec<f64>, Vec<f64>),
}

thread_local! {
    static CACHE: RefCell<Option<DemagCache>> = const { RefCell::new(None) };
}

fn fft2(re: &mut [f64], im: &mut [f64], ky: usize, kx: usize, forward: bool) {
    let shape = [ky, kx];
    fft_axis_inplace(re, im, &shape, 0, forward);
    fft_axis_inplace(re, im, &shape, 1, forward);
}

fn geom_key(nx: usize, ny: usize, dx: f64, dy: f64, dz: f64) -> u64 {
    let mut h = 1469598103934665603u64;
    for v in [nx as u64, ny as u64, dx.to_bits(), dy.to_bits(), dz.to_bits()] {
        h ^= v;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// Full demag field B_demag (Tesla) for a 2-D magnetization m of shape [nx,ny,3]
/// (flat, row-major, component-fastest). Kernel FFTs cached by geometry.
pub fn demag_field(m: &[f64], nx: usize, ny: usize, msat: f64, dx: f64, dy: f64, dz: f64) -> Vec<f64> {
    let key = geom_key(nx, ny, dx, dy, dz);
    CACHE.with(|c| {
        {
            let mut cache = c.borrow_mut();
            if cache.as_ref().map_or(true, |k| k.key != key) {
                let (comps, k) = demag_kernel([nx, ny, 1], [dx, dy, dz]);
                let (ky, kx) = (k[1], k[2]);
                let mkfft = |src: &Vec<f64>| -> (Vec<f64>, Vec<f64>) {
                    let mut re = src.clone();
                    let mut im = vec![0.0; re.len()];
                    fft2(&mut re, &mut im, ky, kx, true);
                    (re, im)
                };
                *cache = Some(DemagCache {
                    key, kx, ky,
                    xx: mkfft(&comps[0]), xy: mkfft(&comps[1]),
                    yy: mkfft(&comps[3]), zz: mkfft(&comps[5]),
                });
            }
        }
        let cache = c.borrow();
        let cache = cache.as_ref().unwrap();
        let (kx, ky) = (cache.kx, cache.ky);
        let np = kx * ky;

        // pad + transpose m components into the kernel's (y,x) layout
        let mut mxr = vec![0.0; np]; let mut mxi = vec![0.0; np];
        let mut myr = vec![0.0; np]; let mut myi = vec![0.0; np];
        let mut mzr = vec![0.0; np]; let mut mzi = vec![0.0; np];
        for x in 0..nx {
            for y in 0..ny {
                let base = (x * ny + y) * 3;
                let p = y * kx + x;
                mxr[p] = m[base];
                myr[p] = m[base + 1];
                mzr[p] = m[base + 2];
            }
        }
        fft2(&mut mxr, &mut mxi, ky, kx, true);
        fft2(&mut myr, &mut myi, ky, kx, true);
        fft2(&mut mzr, &mut mzi, ky, kx, true);

        // frequency-space multiply: B_i = Σ_j K_ij m_j (complex)
        let mut bxr = vec![0.0; np]; let mut bxi = vec![0.0; np];
        let mut byr = vec![0.0; np]; let mut byi = vec![0.0; np];
        let mut bzr = vec![0.0; np]; let mut bzi = vec![0.0; np];
        for i in 0..np {
            let (xxr, xxi) = (cache.xx.0[i], cache.xx.1[i]);
            let (xyr, xyi) = (cache.xy.0[i], cache.xy.1[i]);
            let (yyr, yyi) = (cache.yy.0[i], cache.yy.1[i]);
            let (zzr, zzi) = (cache.zz.0[i], cache.zz.1[i]);
            // Bx = Kxx*mx + Kxy*my
            bxr[i] = xxr*mxr[i]-xxi*mxi[i] + xyr*myr[i]-xyi*myi[i];
            bxi[i] = xxr*mxi[i]+xxi*mxr[i] + xyr*myi[i]+xyi*myr[i];
            // By = Kxy*mx + Kyy*my
            byr[i] = xyr*mxr[i]-xyi*mxi[i] + yyr*myr[i]-yyi*myi[i];
            byi[i] = xyr*mxi[i]+xyi*mxr[i] + yyr*myi[i]+yyi*myr[i];
            // Bz = Kzz*mz
            bzr[i] = zzr*mzr[i]-zzi*mzi[i];
            bzi[i] = zzr*mzi[i]+zzi*mzr[i];
        }
        fft2(&mut bxr, &mut bxi, ky, kx, false);
        fft2(&mut byr, &mut byi, ky, kx, false);
        fft2(&mut bzr, &mut bzi, ky, kx, false);

        // crop + scale back to [nx,ny,3]
        let scl = 4e-7 * std::f64::consts::PI * msat; // mu0 * Msat
        let mut out = vec![0.0; nx * ny * 3];
        for x in 0..nx {
            for y in 0..ny {
                let p = y * kx + x;
                let base = (x * ny + y) * 3;
                out[base]     = scl * bxr[p];
                out[base + 1] = scl * byr[p];
                out[base + 2] = scl * bzr[p];
            }
        }
        out
    })
}
