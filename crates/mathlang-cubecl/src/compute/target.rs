//! Backend (which CubeCL runtime) and precision (which float representation) — the
//! two axes of a compute *target*. Threaded through every tensor value and op so
//! the three precision modes are designed in, not retrofitted.

/// Which CubeCL runtime executes kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Cpu,
    Wgpu,
    Cuda,
    Hip,
}

/// Float representation for on-device storage and math.
///
/// * `F32` — single precision. Universal; ~7 digits.
/// * `Df64` — *double-single*: each value is an unevaluated sum of two f32
///   `(hi, lo)`, ~14 digits. Runs **everywhere, including wgpu/Metal** (which has
///   no native f64). Stored as interleaved `[hi, lo]` f32 pairs. Arithmetic uses
///   error-free transforms (TwoSum/TwoProd) — see `kernels::df64`.
/// * `F64` — native double. Available on cpu/cuda/hip (not wgpu — WGSL has no f64).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prec {
    F32,
    Df64,
    F64,
}

impl Prec {
    pub fn name(self) -> &'static str {
        match self {
            Prec::F32 => "f32",
            Prec::Df64 => "df64",
            Prec::F64 => "f64",
        }
    }

    pub fn parse(s: &str) -> Option<Prec> {
        match s {
            "f32" => Some(Prec::F32),
            "df64" => Some(Prec::Df64),
            "f64" => Some(Prec::F64),
            _ => None,
        }
    }

    /// Bytes per logical element (df64 stores two f32).
    #[allow(dead_code)]
    pub fn elem_bytes(self) -> usize {
        match self {
            Prec::F32 => 4,
            Prec::Df64 => 8,
            Prec::F64 => 8,
        }
    }
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Wgpu => "wgpu",
            Backend::Cuda => "cuda",
            Backend::Hip => "hip",
        }
    }

    pub fn parse(s: &str) -> Option<Backend> {
        match s {
            "cpu" => Some(Backend::Cpu),
            "wgpu" => Some(Backend::Wgpu),
            "cuda" => Some(Backend::Cuda),
            "hip" => Some(Backend::Hip),
            _ => None,
        }
    }

    pub fn compiled_in(self) -> bool {
        match self {
            Backend::Cpu => cfg!(feature = "cpu"),
            Backend::Wgpu => cfg!(feature = "wgpu"),
            Backend::Cuda => cfg!(feature = "cuda"),
            Backend::Hip => cfg!(feature = "hip"),
        }
    }

    /// Whether the backend has hardware f64 (wgpu/Metal does not).
    pub fn native_f64(self) -> bool {
        !matches!(self, Backend::Wgpu)
    }

    /// Can this backend run the given precision?
    pub fn supports(self, p: Prec) -> bool {
        match (self, p) {
            (Backend::Wgpu, Prec::F64) => false, // WGSL has no f64
            _ => true,
        }
    }

    /// Best default precision: native f64 where available, else f32.
    pub fn default_prec(self) -> Prec {
        if self.native_f64() { Prec::F64 } else { Prec::F32 }
    }

    pub fn available() -> Vec<Backend> {
        [Backend::Cpu, Backend::Wgpu, Backend::Cuda, Backend::Hip]
            .into_iter()
            .filter(|b| b.compiled_in())
            .collect()
    }

    /// $MATHLANG_BACKEND if valid+compiled, else first available (cpu preferred).
    pub fn default_choice() -> Backend {
        if let Ok(s) = std::env::var("MATHLANG_BACKEND") {
            if let Some(b) = Backend::parse(&s) {
                if b.compiled_in() {
                    return b;
                }
            }
        }
        Backend::available().into_iter().next().unwrap_or(Backend::Cpu)
    }
}

/// A full compute target: where to run and at what precision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Target {
    pub backend: Backend,
    pub prec: Prec,
}

impl Target {
    pub fn default_target() -> Target {
        let backend = Backend::default_choice();
        let prec = std::env::var("MATHLANG_PREC")
            .ok()
            .and_then(|s| Prec::parse(&s))
            .filter(|&p| backend.supports(p))
            .unwrap_or_else(|| backend.default_prec());
        Target { backend, prec }
    }
}
