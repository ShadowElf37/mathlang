//! `animate2D` — stream 2-D tensor frames to the standalone `wgpu_animator` GUI
//! via the MXFR protocol (the same viewer and wire format as the original `m`).
//!
//! mc tensors are device-resident, so each frame is downloaded to the host before
//! it is written. Three builtins:
//!   * `animate2D(value [, fps])`        — spawn the animator and stream to it
//!   * `animate2D_raw(value)`            — write MXFR frames to stdout (pipe/test)
//!   * `animate2Dforever(f [, fps])`     — stream f(0), f(1), … until the window closes
//!
//! Axis convention: a frame tensor is indexed `T[x, y]` (shape `[nx, ny]`), stored
//! x-major (`data[x*ny + y]`). MXFR is row-major (y outer), so `write_frame_xy`
//! transposes on the fly. A 3-D tensor `[n_frames, nx, ny]` is a prebuilt movie —
//! exactly what `scan(step, u0, n)` produces for a 2-D state.
//!
//! Calling conventions (fps optional, default 30; `animate2D` only):
//!   animate2D(T)                 T: 3-D tensor [n_frames, nx, ny]
//!   animate2D(f, n)              f: t -> 2-D tensor [nx, ny]; frames at t = 0..n-1
//!   animate2D(f, t_vals)         f + 1-D tensor of timestamps
//!   animate2D(f, t0, t1, n)      f + linspace(t0, t1, n)

use crate::compute;
use crate::interp::{apply_val, Env};
use crate::value::{fmt_val, Val};
use std::io::Write;

const DEFAULT_FPS: f64 = 30.0;

// ── MXFR binary frame ────────────────────────────────────────────────────────────

/// Write one scalar MXFR frame. `data` is x-major (`data[x*ny + y]`); the output is
/// row-major (`pixel[y*nx + x]`), width = nx, height = ny. The frame is assembled in
/// one buffer and written once (per-pixel writes are far slower for large grids).
fn write_frame_xy(out: &mut impl Write, data: &[f64], nx: usize, ny: usize, t: f64) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(24 + nx * ny * 4);
    buf.extend_from_slice(b"MXFR");
    buf.extend_from_slice(&(nx as u32).to_le_bytes()); // width
    buf.extend_from_slice(&(ny as u32).to_le_bytes()); // height
    buf.extend_from_slice(&1u32.to_le_bytes()); // channels (scalar)
    buf.extend_from_slice(&t.to_le_bytes()); // timestamp
    for y in 0..ny {
        for x in 0..nx {
            buf.extend_from_slice(&(data[x * ny + y] as f32).to_le_bytes());
        }
    }
    out.write_all(&buf)?;
    out.flush()
}

/// Locate the animator binary: `$WGPU_ANIMATOR`, then a local build, then `PATH`.
pub fn find_animator() -> String {
    if let Ok(p) = std::env::var("WGPU_ANIMATOR") {
        return p;
    }
    let local = "./animator/target/release/wgpu_animator";
    if std::path::Path::new(local).exists() {
        return local.to_string();
    }
    "wgpu_animator".to_string()
}

/// Apply `f(t)`, download the result, and return (x-major data, nx, ny). A 2-D
/// tensor is `[nx, ny]`; a 1-D tensor becomes a horizontal strip (`ny = 1`).
fn call_for_frame(f: &Val, t: f64, env: &Env) -> Result<(Vec<f64>, usize, usize), String> {
    match apply_val(f.clone(), vec![Val::Num(t)], env)? {
        Val::Tensor(tv) => {
            let data = compute::download(&tv)?;
            match tv.shape.as_slice() {
                [nx, ny] => Ok((data, *nx, *ny)),
                [n] => Ok((data, *n, 1)),
                shape => Err(format!(
                    "animate2D: f(t) must return a 1-D or 2-D tensor, got shape {shape:?}"
                )),
            }
        }
        other => Err(format!(
            "animate2D: f(t) must return a 2-D tensor [nx, ny], got {}",
            fmt_val(&other)
        )),
    }
}

// ── frame-streaming core ─────────────────────────────────────────────────────────

/// Stream frames from `first` (a 3-D tensor movie or a frame function) to `out`.
/// `rest` is the already-evaluated tail with any fps argument removed.
fn stream_frames(first: Val, rest: &[Val], env: &Env, out: &mut impl Write) -> Result<usize, String> {
    match first {
        // animate2D(T) — a prebuilt movie [n_frames, nx, ny].
        Val::Tensor(tv) if tv.shape.len() == 3 => {
            if !rest.is_empty() {
                return Err(format!("animate2D: tensor form takes no extra args after T (got {})", rest.len()));
            }
            let (nf, nx, ny) = (tv.shape[0], tv.shape[1], tv.shape[2]);
            let frame_size = nx * ny;
            let data = compute::download(&tv)?;
            eprintln!("animate2D: streaming {nf} frames ({nx}×{ny})");
            for f in 0..nf {
                let slice = &data[f * frame_size..(f + 1) * frame_size];
                write_frame_xy(out, slice, nx, ny, f as f64).map_err(|e| format!("animate2D: write error: {e}"))?;
            }
            eprintln!("animate2D: done ({nf} frames)");
            Ok(nf)
        }

        // animate2D(f, …) — call f(t) per frame.
        f_val @ (Val::Fn { .. } | Val::Builtin(_)) => {
            let t_vals: Vec<f64> = match rest.len() {
                1 => match &rest[0] {
                    Val::Tensor(tv) if tv.shape.len() == 1 => compute::download(tv)?,
                    Val::Num(n) => (0..*n as usize).map(|k| k as f64).collect(),
                    other => return Err(format!(
                        "animate2D: 2nd arg must be a 1-D tensor of timestamps or a count, got {}",
                        fmt_val(other)
                    )),
                },
                3 => {
                    let t0 = rest[0].clone().num("animate2D t0")?;
                    let t1 = rest[1].clone().num("animate2D t1")?;
                    let n = rest[2].clone().num("animate2D n")? as usize;
                    if n < 2 {
                        return Err("animate2D: n must be >= 2".into());
                    }
                    (0..n).map(|k| t0 + (t1 - t0) * k as f64 / (n - 1) as f64).collect()
                }
                n => return Err(format!(
                    "animate2D: function form expects 1 or 3 timestamp args (after fps strip), got {n}"
                )),
            };

            let n_frames = t_vals.len();
            if n_frames == 0 {
                return Err("animate2D: no frames to animate (count is 0 / empty timestamps)".into());
            }
            eprintln!("animate2D: computing and streaming {n_frames} frames …");

            let (first_data, nx, ny) = call_for_frame(&f_val, t_vals[0], env)?;
            eprintln!("animate2D: grid {nx}×{ny}");
            write_frame_xy(out, &first_data, nx, ny, t_vals[0]).map_err(|e| format!("animate2D: write error: {e}"))?;

            for &t in &t_vals[1..] {
                let (frame_data, fnx, fny) = call_for_frame(&f_val, t, env)?;
                if fnx != nx || fny != ny {
                    return Err(format!(
                        "animate2D: frame at t={t} has shape [{fnx},{fny}], expected [{nx},{ny}]"
                    ));
                }
                write_frame_xy(out, &frame_data, fnx, fny, t).map_err(|e| format!("animate2D: write error: {e}"))?;
            }
            eprintln!("animate2D: done ({n_frames} frames)");
            Ok(n_frames)
        }

        other => Err(format!("animate2D: first arg must be a 3-D tensor or a function, got {}", fmt_val(&other))),
    }
}

/// Split an optional trailing fps from `rest`, by first-arg type and arg count.
fn extract_fps<'a>(first: &Val, rest: &'a [Val]) -> Result<(f64, &'a [Val]), String> {
    match first {
        Val::Tensor(tv) if tv.shape.len() == 3 => match rest.len() {
            0 => Ok((DEFAULT_FPS, &rest[..0])),
            1 => Ok((rest[0].clone().num("animate2D fps")?, &rest[..0])),
            n => Err(format!("animate2D: tensor form takes 0 or 1 extra args (fps), got {n}")),
        },
        Val::Fn { .. } | Val::Builtin(_) => match rest.len() {
            1 => Ok((DEFAULT_FPS, rest)),       // (f, n_or_tvals)
            2 => Ok((rest[1].clone().num("animate2D fps")?, &rest[..1])), // (f, n_or_tvals, fps)
            3 => Ok((DEFAULT_FPS, rest)),       // (f, t0, t1, n)
            4 => Ok((rest[3].clone().num("animate2D fps")?, &rest[..3])), // (f, t0, t1, n, fps)
            n => Err(format!("animate2D: function form expects 1–4 extra args, got {n}")),
        },
        other => Err(format!("animate2D: first arg must be a 3-D tensor or a function, got {}", fmt_val(other))),
    }
}

/// Build the animator command-line args (shared by `animate2D`/`animate2Dforever`).
fn animator_args(fps: f64, stream: bool) -> Vec<String> {
    let mut a: Vec<String> = vec!["--stdin".into()];
    if stream {
        a.push("--stream".into());
    }
    a.push("--colormap".into());
    a.push("heat".into());
    a.push("--fps".into());
    a.push(format!("{}", fps as u32));
    if let Ok(title) = std::env::var("WGPU_TITLE") {
        a.push("--title".into());
        a.push(title);
    }
    a
}

// ── builtins ──────────────────────────────────────────────────────────────────────

/// `animate2D(value [, fps])` — spawn the animator and stream frames to it.
pub fn animate2d(args: Vec<Val>, env: &Env) -> Result<Val, String> {
    if args.is_empty() {
        return Err("animate2D(T [,fps]) | animate2D(f, n [,fps]) | animate2D(f, t_vals [,fps]) | animate2D(f, t0, t1, n [,fps])".into());
    }
    let first = args[0].clone();
    let (fps, core_rest) = extract_fps(&first, &args[1..])?;

    let animator = find_animator();
    eprintln!("animate2D: spawning '{animator}'");
    let mut child = std::process::Command::new(&animator)
        .args(&animator_args(fps, false))
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!(
            "animate2D: failed to spawn '{animator}': {e}\n  hint: build animator/ (cargo build --release) or set WGPU_ANIMATOR=/path/to/wgpu_animator"
        ))?;

    let n = {
        let child_stdin = child.stdin.take().ok_or("animate2D: could not get animator stdin")?;
        let mut out = std::io::BufWriter::new(child_stdin);
        stream_frames(first, core_rest, env, &mut out)?
        // out dropped here → EOF to animator
    };

    match child.wait() {
        Ok(status) if !status.success() => eprintln!("animate2D: animator exited with {status}"),
        Err(e) => eprintln!("animate2D: error waiting for animator: {e}"),
        _ => {}
    }
    Ok(Val::Num(n as f64))
}

/// `animate2D_raw(value)` — write MXFR frames to stdout (no fps; for piping/tests).
pub fn animate2d_raw(args: Vec<Val>, env: &Env) -> Result<Val, String> {
    if args.is_empty() {
        return Err("animate2D_raw(T) | animate2D_raw(f, n) | animate2D_raw(f, t_vals) | animate2D_raw(f, t0, t1, n)".into());
    }
    let first = args[0].clone();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let n = stream_frames(first, &args[1..], env, &mut out)?;
    Ok(Val::Num(n as f64))
}

/// `animate2Dforever(f [, fps])` — stream f(0), f(1), … until the window closes.
pub fn animate2d_forever(args: Vec<Val>, env: &Env) -> Result<Val, String> {
    if args.is_empty() {
        return Err("animate2Dforever(f [,fps]) — f: t -> 2-D tensor [nx,ny]".into());
    }
    let f_val = match &args[0] {
        v @ (Val::Fn { .. } | Val::Builtin(_)) => v.clone(),
        other => return Err(format!(
            "animate2Dforever: first arg must be a function t -> [nx,ny] tensor, got {}",
            fmt_val(other)
        )),
    };
    let fps = match args.len() {
        1 => DEFAULT_FPS,
        2 => args[1].clone().num("animate2Dforever fps")?,
        n => return Err(format!("animate2Dforever: expects f [,fps], got {n} args")),
    };

    let animator = find_animator();
    eprintln!("animate2Dforever: spawning '{animator}' (close the window to stop)");
    let mut child = std::process::Command::new(&animator)
        .args(&animator_args(fps, true))
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!(
            "animate2Dforever: failed to spawn '{animator}': {e}\n  hint: build animator/ or set WGPU_ANIMATOR=/path/to/wgpu_animator"
        ))?;

    let mut n: u64 = 0;
    {
        let child_stdin = child.stdin.take().ok_or("animate2Dforever: could not get animator stdin")?;
        let mut out = std::io::BufWriter::new(child_stdin);
        let (first_data, nx, ny) = call_for_frame(&f_val, 0.0, env)?;
        eprintln!("animate2Dforever: grid {nx}×{ny}, streaming t = 0,1,2,…");
        if write_frame_xy(&mut out, &first_data, nx, ny, 0.0).is_ok() {
            n = 1;
            let mut t = 1.0_f64;
            loop {
                let (data, fnx, fny) = call_for_frame(&f_val, t, env)?;
                if fnx != nx || fny != ny {
                    return Err(format!(
                        "animate2Dforever: frame at t={t} has shape [{fnx},{fny}], expected [{nx},{ny}]"
                    ));
                }
                if write_frame_xy(&mut out, &data, fnx, fny, t).is_err() {
                    break; // window closed → broken pipe
                }
                n += 1;
                t += 1.0;
            }
        }
    }
    let _ = child.wait();
    eprintln!("animate2Dforever: stopped ({n} frames)");
    Ok(Val::Num(n as f64))
}
