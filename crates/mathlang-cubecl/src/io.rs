//! Tensor file I/O: NumPy `.npy`, native `.mlt`, and (feature-gated) HDF5.
//!
//! mc tensors are device-resident, so save downloads to host before writing and
//! load uploads to the active target after reading. Format is auto-detected from
//! the file extension (`.npy` / `.mlt` / `.h5`|`.hdf5`). `.npy` and `.mlt` are
//! pure-Rust (no dependencies); HDF5 is behind the `hdf5` cargo feature.
//!
//! Real and complex f64 tensors are supported. The `.npy` *loader* additionally
//! decodes f2/f4/f8, c8/c16, signed/unsigned ints (1/2/4/8 bytes) and bool, so
//! arrays written by NumPy elsewhere read back as f64.

use crate::compute::{self, Target};
use crate::value::{fmt_val, Val};

/// Expand a leading `~/` to `$HOME/`.
pub fn expand_path(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        format!("{}/{rest}", std::env::var("HOME").unwrap_or_default())
    } else {
        p.to_string()
    }
}

/// A host-resident tensor — the bridge between `Val` and the on-disk formats.
pub enum HostTensor {
    Real { data: Vec<f64>, shape: Vec<usize> },
    Complex { re: Vec<f64>, im: Vec<f64>, shape: Vec<usize> },
}

impl HostTensor {
    pub fn nelem(&self) -> usize {
        match self {
            HostTensor::Real { data, .. } => data.len(),
            HostTensor::Complex { re, .. } => re.len(),
        }
    }
}

/// Pull a `Val` to the host for serialization. Device tensors are downloaded;
/// scalars become length-1 tensors (NumPy has no distinct scalar type).
pub fn val_to_host(val: &Val) -> Result<HostTensor, String> {
    match val {
        Val::Tensor(t) => Ok(HostTensor::Real { data: compute::download(t)?, shape: t.shape.clone() }),
        Val::ComplexTensor(t) => {
            let (re, im) = compute::download_complex(t)?;
            Ok(HostTensor::Complex { re, im, shape: t.shape.clone() })
        }
        Val::Num(x) => Ok(HostTensor::Real { data: vec![*x], shape: vec![1] }),
        Val::Complex(a, b) => Ok(HostTensor::Complex { re: vec![*a], im: vec![*b], shape: vec![1] }),
        other => Err(format!("save: can only serialize tensors/scalars, got {}", fmt_val(other))),
    }
}

/// Upload a host tensor to the active target as a `Val`.
pub fn host_to_val(h: HostTensor, target: Target) -> Result<Val, String> {
    match h {
        HostTensor::Real { data, shape } => compute::upload(target, &data, shape).map(Val::Tensor),
        HostTensor::Complex { re, im, shape } => {
            compute::upload_complex(target, &re, &im, shape).map(Val::ComplexTensor)
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Npy,
    Mlt,
    Hdf5,
}

fn detect_format(path: &str) -> Result<Format, String> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".npy") {
        Ok(Format::Npy)
    } else if lower.ends_with(".mlt") {
        Ok(Format::Mlt)
    } else if lower.ends_with(".h5") || lower.ends_with(".hdf5") || lower.ends_with(".he5") {
        Ok(Format::Hdf5)
    } else {
        Err(format!("unrecognised extension on '{path}' (use .npy, .mlt, or .h5)"))
    }
}

/// Save a value to `path`, picking the format from the extension. Returns the
/// element count written. `dataset` applies only to HDF5 (default `/data`).
pub fn save_value(path: &str, val: &Val, dataset: Option<&str>) -> Result<usize, String> {
    let fp = expand_path(path);
    let h = val_to_host(val)?;
    match detect_format(&fp)? {
        Format::Npy => npy_save(&fp, &h).map(|_| h.nelem()),
        Format::Mlt => mlt_save(&fp, &h).map(|_| h.nelem()),
        Format::Hdf5 => h5_save(&fp, dataset.unwrap_or("/data"), &h, false, None).map(|_| h.nelem()),
    }
}

/// Load a value from `path`, picking the format from the extension, and upload it
/// to `target`. `dataset` applies only to HDF5 (default `/data`).
pub fn load_value(path: &str, target: Target, dataset: Option<&str>) -> Result<Val, String> {
    let fp = expand_path(path);
    let h = match detect_format(&fp)? {
        Format::Npy => npy_load(&fp)?,
        Format::Mlt => mlt_load(&fp)?,
        Format::Hdf5 => h5_load(&fp, dataset.unwrap_or("/data"))?,
    };
    host_to_val(h, target)
}

// Format-explicit wrappers for the REPL bang-commands (which name the format
// directly, for parity with the original `m`). Path `~/` expansion is applied.

pub fn save_npy_val(path: &str, val: &Val) -> Result<usize, String> {
    let h = val_to_host(val)?;
    npy_save(&expand_path(path), &h).map(|_| h.nelem())
}
pub fn load_npy_val(path: &str, target: Target) -> Result<Val, String> {
    host_to_val(npy_load(&expand_path(path))?, target)
}
pub fn save_mlt_val(path: &str, val: &Val) -> Result<usize, String> {
    let h = val_to_host(val)?;
    mlt_save(&expand_path(path), &h).map(|_| h.nelem())
}
pub fn load_mlt_val(path: &str, target: Target) -> Result<Val, String> {
    host_to_val(mlt_load(&expand_path(path))?, target)
}
pub fn save_hdf5_val(path: &str, ds: &str, val: &Val, append: bool, gzip: Option<u32>) -> Result<usize, String> {
    let h = val_to_host(val)?;
    h5_save(&expand_path(path), ds, &h, append, gzip)
}
pub fn load_hdf5_val(path: &str, ds: &str, target: Target) -> Result<Val, String> {
    host_to_val(h5_load(&expand_path(path), ds)?, target)
}

// ── NumPy .npy ────────────────────────────────────────────────────────────────

fn npy_shape_str(shape: &[usize]) -> String {
    match shape {
        [] => "()".into(),
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

fn npy_save(path: &str, h: &HostTensor) -> Result<(), String> {
    (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let (dtype, shape) = match h {
            HostTensor::Real { shape, .. } => ("<f8", shape.as_slice()),
            HostTensor::Complex { shape, .. } => ("<c16", shape.as_slice()),
        };
        let header = format!(
            "{{'descr': '{dtype}', 'fortran_order': False, 'shape': {}, }}",
            npy_shape_str(shape)
        );
        // Total header must be a multiple of 64 bytes: 6 magic + 2 version + 2 hlen.
        let min_total = 10 + header.len() + 1; // +1 for trailing '\n'
        let padded = min_total.div_ceil(64) * 64;
        let mut hdr = header;
        for _ in 0..(padded - min_total) {
            hdr.push(' ');
        }
        hdr.push('\n');
        f.write_all(b"\x93NUMPY")?;
        f.write_all(&[1u8, 0u8])?;
        f.write_all(&(hdr.len() as u16).to_le_bytes())?;
        f.write_all(hdr.as_bytes())?;
        match h {
            HostTensor::Real { data, .. } => {
                for &v in data.iter() {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            HostTensor::Complex { re, im, .. } => {
                for (&r, &i) in re.iter().zip(im.iter()) {
                    f.write_all(&r.to_le_bytes())?;
                    f.write_all(&i.to_le_bytes())?;
                }
            }
        }
        Ok(())
    })()
    .map_err(|e| e.to_string())
}

fn npy_find_str(header: &str, key: &str) -> Option<String> {
    for q in ['"', '\''] {
        let pat = format!("{q}{key}{q}");
        if let Some(ki) = header.find(&pat) {
            let rest = header[ki + pat.len()..].trim_start().trim_start_matches(':').trim_start();
            let qv = rest.chars().next()?;
            if qv == '\'' || qv == '"' {
                let inner = &rest[1..];
                return Some(inner[..inner.find(qv)?].to_string());
            }
        }
    }
    None
}

fn npy_find_shape(header: &str) -> Option<String> {
    for q in ['"', '\''] {
        let pat = format!("{q}shape{q}");
        if let Some(ki) = header.find(&pat) {
            let rest = header[ki + pat.len()..].trim_start().trim_start_matches(':').trim_start();
            if rest.starts_with('(') {
                let end = rest.find(')')?;
                return Some(rest[..=end].to_string());
            }
        }
    }
    None
}

fn npy_parse_shape(s: &str) -> Result<Vec<usize>, String> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>().map_err(|e| format!("bad shape dim '{p}': {e}")))
        .collect()
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 != 0 { -1.0f64 } else { 1.0 };
    let exp = (bits >> 10 & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f64;
    match exp {
        0x1f => if mant != 0.0 { f64::NAN } else { sign * f64::INFINITY },
        0 => sign * mant / 1024.0 * 2.0f64.powi(-14),
        _ => sign * (1.0 + mant / 1024.0) * 2.0f64.powi(exp - 15),
    }
}

fn npy_load(path: &str) -> Result<HostTensor, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a valid .npy file".into());
    }
    let major = bytes[6];
    let (hlen, doff) = if major <= 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        if bytes.len() < 12 {
            return Err("truncated header".into());
        }
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12usize)
    };
    if bytes.len() < doff + hlen {
        return Err("truncated header".into());
    }
    let header = std::str::from_utf8(&bytes[doff..doff + hlen]).map_err(|_| "invalid header encoding")?;

    let descr = npy_find_str(header, "descr").ok_or("missing 'descr' in npy header")?;
    let shape_s = npy_find_shape(header).ok_or("missing 'shape' in npy header")?;
    let shape = npy_parse_shape(&shape_s)?;
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err("Fortran-order arrays are not supported".into());
    }

    let nelem: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    let shape = if shape.is_empty() { vec![1] } else { shape };
    let buf = &bytes[doff + hlen..];

    // dtype = endian-char + kind-char + byte-width, e.g. "<f8", ">c16", "|u1".
    let db = descr.as_bytes();
    if db.len() < 3 {
        return Err(format!("unrecognised dtype: {descr}"));
    }
    let be = db[0] == b'>';
    let kind = db[1] as char;
    let nb: usize = descr[2..].parse().map_err(|_| format!("unrecognised dtype: {descr}"))?;

    let need = |n: usize| -> Result<(), String> {
        if buf.len() < n {
            Err(format!("truncated data: need {n} bytes, have {}", buf.len()))
        } else {
            Ok(())
        }
    };

    match (kind, nb) {
        ('f', 8) => {
            need(nelem * 8)?;
            let data = (0..nelem)
                .map(|i| {
                    let b: [u8; 8] = buf[i * 8..(i + 1) * 8].try_into().unwrap();
                    if be { f64::from_be_bytes(b) } else { f64::from_le_bytes(b) }
                })
                .collect();
            Ok(HostTensor::Real { data, shape })
        }
        ('f', 4) => {
            need(nelem * 4)?;
            let data = (0..nelem)
                .map(|i| {
                    let b: [u8; 4] = buf[i * 4..(i + 1) * 4].try_into().unwrap();
                    if be { f32::from_be_bytes(b) as f64 } else { f32::from_le_bytes(b) as f64 }
                })
                .collect();
            Ok(HostTensor::Real { data, shape })
        }
        ('f', 2) => {
            need(nelem * 2)?;
            let data = (0..nelem)
                .map(|i| {
                    let b: [u8; 2] = buf[i * 2..(i + 1) * 2].try_into().unwrap();
                    f16_to_f64(if be { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
                })
                .collect();
            Ok(HostTensor::Real { data, shape })
        }
        ('c', 16) => {
            need(nelem * 16)?;
            let mut re = Vec::with_capacity(nelem);
            let mut im = Vec::with_capacity(nelem);
            for i in 0..nelem {
                let br: [u8; 8] = buf[i * 16..i * 16 + 8].try_into().unwrap();
                let bi: [u8; 8] = buf[i * 16 + 8..i * 16 + 16].try_into().unwrap();
                re.push(if be { f64::from_be_bytes(br) } else { f64::from_le_bytes(br) });
                im.push(if be { f64::from_be_bytes(bi) } else { f64::from_le_bytes(bi) });
            }
            Ok(HostTensor::Complex { re, im, shape })
        }
        ('c', 8) => {
            need(nelem * 8)?;
            let mut re = Vec::with_capacity(nelem);
            let mut im = Vec::with_capacity(nelem);
            for i in 0..nelem {
                let br: [u8; 4] = buf[i * 8..i * 8 + 4].try_into().unwrap();
                let bi: [u8; 4] = buf[i * 8 + 4..i * 8 + 8].try_into().unwrap();
                re.push(if be { f32::from_be_bytes(br) as f64 } else { f32::from_le_bytes(br) as f64 });
                im.push(if be { f32::from_be_bytes(bi) as f64 } else { f32::from_le_bytes(bi) as f64 });
            }
            Ok(HostTensor::Complex { re, im, shape })
        }
        ('i' | 'u', nb) if nb <= 8 => {
            need(nelem * nb)?;
            let signed = kind == 'i';
            let data: Vec<f64> = (0..nelem)
                .map(|i| {
                    let sl = &buf[i * nb..(i + 1) * nb];
                    match (signed, nb, be) {
                        (_, 1, _) => if signed { sl[0] as i8 as f64 } else { sl[0] as f64 },
                        (true, 2, false) => i16::from_le_bytes(sl.try_into().unwrap()) as f64,
                        (true, 2, true) => i16::from_be_bytes(sl.try_into().unwrap()) as f64,
                        (false, 2, false) => u16::from_le_bytes(sl.try_into().unwrap()) as f64,
                        (false, 2, true) => u16::from_be_bytes(sl.try_into().unwrap()) as f64,
                        (true, 4, false) => i32::from_le_bytes(sl.try_into().unwrap()) as f64,
                        (true, 4, true) => i32::from_be_bytes(sl.try_into().unwrap()) as f64,
                        (false, 4, false) => u32::from_le_bytes(sl.try_into().unwrap()) as f64,
                        (false, 4, true) => u32::from_be_bytes(sl.try_into().unwrap()) as f64,
                        (true, 8, false) => i64::from_le_bytes(sl.try_into().unwrap()) as f64,
                        (true, 8, true) => i64::from_be_bytes(sl.try_into().unwrap()) as f64,
                        (false, 8, false) => u64::from_le_bytes(sl.try_into().unwrap()) as f64,
                        (false, 8, true) => u64::from_be_bytes(sl.try_into().unwrap()) as f64,
                        _ => 0.0,
                    }
                })
                .collect();
            Ok(HostTensor::Real { data, shape })
        }
        ('b', 1) => {
            need(nelem)?;
            let data = (0..nelem).map(|i| if buf[i] != 0 { 1.0 } else { 0.0 }).collect();
            Ok(HostTensor::Real { data, shape })
        }
        _ => Err(format!("unsupported dtype '{descr}' — supported: f2/f4/f8, c8/c16, i/u 1/2/4/8, bool")),
    }
}

// ── native .mlt ─────────────────────────────────────────────────────────────────
// [8] "MLTENSOR"  [1] type (0=real,1=complex)  [8] ndim  [ndim*8] shape (u64 LE)
//   real:    [nelem*8] f64                complex: [nelem*8] re, then [nelem*8] im

const TENSOR_MAGIC: &[u8; 8] = b"MLTENSOR";
const MLT_REAL: u8 = 0x00;
const MLT_COMPLEX: u8 = 0x01;

fn write_f64s(f: &mut impl std::io::Write, xs: &[f64]) -> std::io::Result<()> {
    for &x in xs {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn mlt_save(path: &str, h: &HostTensor) -> Result<(), String> {
    (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(TENSOR_MAGIC)?;
        match h {
            HostTensor::Real { data, shape } => {
                f.write_all(&[MLT_REAL])?;
                f.write_all(&(shape.len() as u64).to_le_bytes())?;
                for &d in shape {
                    f.write_all(&(d as u64).to_le_bytes())?;
                }
                write_f64s(&mut f, data)?;
            }
            HostTensor::Complex { re, im, shape } => {
                f.write_all(&[MLT_COMPLEX])?;
                f.write_all(&(shape.len() as u64).to_le_bytes())?;
                for &d in shape {
                    f.write_all(&(d as u64).to_le_bytes())?;
                }
                write_f64s(&mut f, re)?;
                write_f64s(&mut f, im)?;
            }
        }
        Ok(())
    })()
    .map_err(|e| e.to_string())
}

fn mlt_load(path: &str) -> Result<HostTensor, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 17 {
        return Err("file too short".into());
    }
    if &bytes[..8] != TENSOR_MAGIC {
        return Err("not a mathlang tensor (.mlt) file".into());
    }
    let kind = bytes[8];
    let ndim = u64::from_le_bytes(bytes[9..17].try_into().unwrap()) as usize;
    let hdr = 17 + ndim * 8;
    if bytes.len() < hdr {
        return Err("truncated header".into());
    }
    let shape: Vec<usize> = (0..ndim)
        .map(|i| u64::from_le_bytes(bytes[17 + i * 8..17 + (i + 1) * 8].try_into().unwrap()) as usize)
        .collect();
    let nelem: usize = shape.iter().product();
    let read_f64s = |off: usize| -> Vec<f64> {
        (0..nelem)
            .map(|i| f64::from_le_bytes(bytes[off + i * 8..off + (i + 1) * 8].try_into().unwrap()))
            .collect()
    };
    match kind {
        MLT_REAL => {
            if bytes.len() < hdr + nelem * 8 {
                return Err(format!("truncated data: need {nelem} f64s"));
            }
            Ok(HostTensor::Real { data: read_f64s(hdr), shape })
        }
        MLT_COMPLEX => {
            if bytes.len() < hdr + nelem * 16 {
                return Err(format!("truncated complex data: need {nelem} complex f64 pairs"));
            }
            Ok(HostTensor::Complex { re: read_f64s(hdr), im: read_f64s(hdr + nelem * 8), shape })
        }
        _ => Err(format!("unknown tensor type byte: 0x{kind:02x}")),
    }
}

// ── HDF5 (feature-gated) ─────────────────────────────────────────────────────────
// Real tensors are a single f64 dataset at `ds_path`. Complex tensors are a group
// with an `mlt_complex` attribute holding `re`/`im` datasets (matches the original).

#[cfg(feature = "hdf5")]
fn h5_split(path: &str) -> (String, String) {
    let path = path.trim_start_matches('/');
    match path.rfind('/') {
        Some(i) => (path[..i].to_string(), path[i + 1..].to_string()),
        None => (String::new(), path.to_string()),
    }
}

#[cfg(feature = "hdf5")]
fn h5_write_ds(grp: &::hdf5::Group, name: &str, data: &[f64], shape: &[usize], gzip: Option<u32>) -> Result<(), String> {
    let mut b = grp.new_dataset::<f64>().shape(shape);
    if let Some(lvl) = gzip {
        b = b.chunk(shape).deflate(lvl as u8);
    }
    b.create(name).map_err(|e| e.to_string())?.write_raw(data).map_err(|e| e.to_string())
}

#[cfg(feature = "hdf5")]
pub fn h5_save(file_path: &str, ds_path: &str, h: &HostTensor, append: bool, gzip: Option<u32>) -> Result<usize, String> {
    let file = if append && std::path::Path::new(file_path).exists() {
        ::hdf5::File::open_rw(file_path).map_err(|e| e.to_string())?
    } else {
        ::hdf5::File::create(file_path).map_err(|e| e.to_string())?
    };
    let (grp_path, name) = h5_split(ds_path);
    let grp_owned: Option<::hdf5::Group> = if grp_path.is_empty() {
        None
    } else {
        Some(file.create_group(&grp_path).or_else(|_| file.group(&grp_path)).map_err(|e| e.to_string())?)
    };
    let grp: &::hdf5::Group = match &grp_owned {
        Some(g) => g,
        None => &file,
    };
    match h {
        HostTensor::Real { data, shape } => {
            h5_write_ds(grp, &name, data, shape, gzip)?;
            Ok(data.len())
        }
        HostTensor::Complex { re, im, shape } => {
            let cg = grp.create_group(&name).map_err(|e| e.to_string())?;
            cg.new_attr::<u8>().create("mlt_complex").map_err(|e| e.to_string())?
                .write_scalar(&1u8).map_err(|e| e.to_string())?;
            h5_write_ds(&cg, "re", re, shape, gzip)?;
            h5_write_ds(&cg, "im", im, shape, gzip)?;
            Ok(re.len())
        }
    }
}

#[cfg(feature = "hdf5")]
pub fn h5_load(file_path: &str, ds_path: &str) -> Result<HostTensor, String> {
    let file = ::hdf5::File::open(file_path).map_err(|e| e.to_string())?;
    let (grp_path, ds_name) = h5_split(ds_path);
    let grp_owned: Option<::hdf5::Group> = if grp_path.is_empty() {
        None
    } else {
        Some(file.group(&grp_path).map_err(|e| e.to_string())?)
    };
    let grp: &::hdf5::Group = match &grp_owned {
        Some(g) => g,
        None => &file,
    };
    if let Ok(ds) = grp.dataset(&ds_name) {
        let shape = ds.shape();
        let data = ds.read_raw::<f64>().map_err(|e| e.to_string())?;
        return Ok(HostTensor::Real { data, shape });
    }
    if let Ok(cg) = grp.group(&ds_name) {
        if cg.attr("mlt_complex").is_ok() {
            let ds_re = cg.dataset("re").map_err(|e| e.to_string())?;
            let shape = ds_re.shape();
            let re = ds_re.read_raw::<f64>().map_err(|e| e.to_string())?;
            let im = cg.dataset("im").map_err(|e| e.to_string())?.read_raw::<f64>().map_err(|e| e.to_string())?;
            return Ok(HostTensor::Complex { re, im, shape });
        }
    }
    Err(format!("'{ds_name}' not found in '{file_path}'"))
}

#[cfg(feature = "hdf5")]
pub fn h5_list(file_path: &str) -> Result<(), String> {
    fn recurse(grp: &::hdf5::Group, depth: usize) -> Result<(), String> {
        let ind = "  ".repeat(depth);
        for name in grp.member_names().map_err(|e| e.to_string())? {
            if let Ok(ds) = grp.dataset(&name) {
                let dims: Vec<String> = ds.shape().iter().map(|d| d.to_string()).collect();
                println!("{ind}{name}  [{}  f64]", dims.join("×"));
            } else if let Ok(cg) = grp.group(&name) {
                if cg.attr("mlt_complex").is_ok() {
                    let dims: Vec<String> = cg.dataset("re").ok()
                        .map_or_else(Vec::new, |d| d.shape())
                        .iter().map(|d| d.to_string()).collect();
                    println!("{ind}{name}  [complex {}  f64]", dims.join("×"));
                } else {
                    println!("{ind}{name}/");
                    recurse(&cg, depth + 1)?;
                }
            }
        }
        Ok(())
    }
    let file = ::hdf5::File::open(file_path).map_err(|e| e.to_string())?;
    recurse(&file, 0)
}

// Fallbacks when HDF5 is not compiled in — a clear, actionable error.
#[cfg(not(feature = "hdf5"))]
pub fn h5_save(_file_path: &str, _ds_path: &str, _h: &HostTensor, _append: bool, _gzip: Option<u32>) -> Result<usize, String> {
    Err("HDF5 support not compiled in — rebuild with `--features hdf5`".into())
}

#[cfg(not(feature = "hdf5"))]
pub fn h5_load(_file_path: &str, _ds_path: &str) -> Result<HostTensor, String> {
    Err("HDF5 support not compiled in — rebuild with `--features hdf5`".into())
}

#[cfg(not(feature = "hdf5"))]
pub fn h5_list(_file_path: &str) -> Result<(), String> {
    Err("HDF5 support not compiled in — rebuild with `--features hdf5`".into())
}
