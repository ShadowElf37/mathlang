// Radix-2 Stockham auto-sort FFT — one butterfly pass per dispatch.
//
// This is the core kernel behind every GPU spectral operator (ops.poisson,
// ops.invlap, ops.specgrad) and the fft/ifft builtins. The Stockham auto-sort
// formulation is the standard GPU choice (GLFFT, cuFFT-style libraries): it is
// out-of-place and produces naturally-ordered output, so it needs NO bit-reversal
// permutation pass — only log2(N) butterfly passes, ping-ponging between two
// buffers.
//
// One dispatch performs a single pass over EVERY independent line at once. A line
// is a 1-D run of N complex samples along the transform axis; an n-D transform is
// the per-axis 1-D transform applied to all `total/N` lines (the host loops over
// axes). Lines are addressed by stride so no transpose is ever needed:
//
//   element p of a line  ->  flat index  base + p*stride
//   base = (line / stride) * (N*stride)  +  (line % stride)
//
// Complex samples are stored interleaved as vec2<f32> = (re, im).
//
// Radix-2 DIF Stockham butterfly, verified against the textbook DFT for N=2,4:
//   one thread owns butterfly i in [0, N/2) of one line. With sub-transform size
//   Ns (= 1,2,4,…,N/2, doubling each pass):
//       j   = i mod Ns
//       lo  = 2*(i - j) + j          (= (i/Ns)*2*Ns + j)
//       hi  = lo + Ns
//       w   = exp(sign * 2*pi*i * j / (2*Ns))     sign = -1 forward, +1 inverse
//       a   = in[i]                  b = in[i + N/2]            (the two halves)
//       out[lo] = a + w*b            out[hi] = a - w*b
//   After log2(N) passes the data is in natural order. `scale` (1/N on the final
//   inverse pass, 1 otherwise) folds the inverse normalisation into the butterfly.

const TWO_PI: f32 = 6.28318530717958647692;

struct Params {
    n:        u32,   // transform length along this axis (power of two)
    ns:       u32,   // current sub-transform size (1,2,…,N/2), doubles per pass
    stride:   u32,   // element stride of the axis in the flat array
    half:     u32,   // N/2 — butterflies per line
    nlines:   u32,   // total / N — number of independent lines
    total_bf: u32,   // nlines * half — total butterflies this pass
    row:      u32,   // threads per 2-D dispatch row (groups_x * workgroup_size)
    _pad0:    u32,
    sign:     f32,   // -1.0 forward, +1.0 inverse
    scale:    f32,   // output multiplier (1/N on the final inverse pass)
    _pad1:    f32,
    _pad2:    f32,
};

@group(0) @binding(0) var<storage, read>       src:    array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst:    array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.y * params.row + gid.x;
    if (g >= params.total_bf) { return; }

    let line = g / params.half;   // which line
    let i    = g % params.half;   // butterfly within the line, in [0, N/2)

    // Base flat offset of this line (strided, no transpose).
    let nstride = params.n * params.stride;
    let inner   = line % params.stride;
    let outer   = line / params.stride;
    let base    = outer * nstride + inner;

    let j  = i & (params.ns - 1u);     // i mod Ns  (Ns is a power of two)
    let lo = 2u * (i - j) + j;         // (i/Ns)*2*Ns + j
    let hi = lo + params.ns;

    let a = src[base + i * params.stride];
    let b = src[base + (i + params.half) * params.stride];

    let ang = params.sign * TWO_PI * f32(j) / f32(2u * params.ns);
    let w   = vec2<f32>(cos(ang), sin(ang));
    let t   = vec2<f32>(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);

    dst[base + lo * params.stride] = (a + t) * params.scale;
    dst[base + hi * params.stride] = (a - t) * params.scale;
}
