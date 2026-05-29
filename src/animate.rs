/// animate2D — spawn wgpu_animator and stream 2D tensor frames to it via the MXFR protocol.
/// animate2D_raw — same, but writes raw MXFR to stdout for manual piping.
///
/// Protocol (from animator/PROTOCOL.md):
///   Per frame, write (little-endian):
///     b"MXFR"  — 4 magic bytes
///     W: u32   — width  (cols)
///     H: u32   — height (rows)
///     t: f64   — timestamp
///     data: W*H f32 values, row-major, row-0 = top of display
///
/// Calling conventions (both animate2D and animate2D_raw unless noted):
///   animate2D(T)                — T: 3-D Tensor [n_frames, H, W]
///   animate2D(T, fps)           — same, with fps (animate2D only)
///   animate2D(f, n)             — f: t→2-D Tensor, n frames at t=0..n-1
///   animate2D(f, n, fps)        — same, with fps (animate2D only)
///   animate2D(f, t_vals)        — f + 1-D tensor of timestamps
///   animate2D(f, t_vals, fps)   — same, with fps (animate2D only)
///   animate2D(f, t0, t1, n)     — f + linspace(t0, t1, n)
///   animate2D(f, t0, t1, n, fps)— same, with fps (animate2D only)
///
/// Animator binary discovery (animate2D only):
///   1. $WGPU_ANIMATOR env var
///   2. ./animator/target/release/wgpu_animator (relative to CWD)
///   3. wgpu_animator (PATH)

use std::io::Write;
use crate::ast::Expr;
use crate::eval::{Val, Env, eval, apply_val};

// ── Binary helpers ─────────────────────────────────────────────────────────────

fn write_frame(out: &mut impl Write, data: &[f64], rows: usize, cols: usize, t: f64)
    -> std::io::Result<()>
{
    out.write_all(b"MXFR")?;
    out.write_all(&(cols as u32).to_le_bytes())?;
    out.write_all(&(rows as u32).to_le_bytes())?;
    out.write_all(&1u32.to_le_bytes())?;  // channels = 1 (scalar)
    out.write_all(&t.to_le_bytes())?;
    for &v in data {
        out.write_all(&(v as f32).to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

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

// ── Call f(t), expect a 2-D (or 1-D) Tensor ───────────────────────────────────

fn call_for_frame(f: &Val, t: f64, env: &Env) -> Result<(Vec<f64>, usize, usize), String> {
    let result = apply_val(f.clone(), vec![Val::Num(t)], env)?;
    match result {
        Val::Tensor { data, shape } if shape.len() == 2 => Ok((data.into_vec(), shape[0], shape[1])),
        Val::Tensor { data, shape } if shape.len() == 1 => {
            let cols = data.len();
            Ok((data.into_vec(), 1, cols))
        }
        other => {
            let type_desc = match &other {
                Val::Tensor { shape, .. } => format!("{}D tensor (shape {:?})", shape.len(), shape),
                Val::ComplexTensor { shape, .. } =>
                    format!("complex {}D tensor (shape {:?}) — check for NaN/division-by-zero", shape.len(), shape),
                Val::Num(_)   => "scalar".into(),
                Val::Tuple(v) => format!("tuple of {} elements", v.len()),
                _             => "non-tensor value".into(),
            };
            Err(format!("animate2D: f(t) must return a 2-D tensor, got {type_desc}"))
        }
    }
}

// ── Locate the wgpu_animator binary (defined above write_frame) ───────────────

// ── Shared frame-streaming core ────────────────────────────────────────────────
//
// `first` — already-evaluated first arg (either a 3-D Tensor or a function).
// `rest`  — remaining *un-evaluated* Expr args (timestamps / count, already
//            stripped of fps by the caller).

fn stream_frames(
    first: Val,
    rest: &[Expr],
    env: &Env,
    out: &mut impl Write,
) -> Result<usize, String> {
    match first {
        // ── animate2D(T) — 3-D Tensor [n_frames, H, W] ───────────────────────
        Val::Tensor { ref data, ref shape } if shape.len() == 3 => {
            if !rest.is_empty() {
                return Err(format!(
                    "animate2D: tensor form takes no extra args after T (got {})",
                    rest.len()
                ));
            }
            let (nf, h, w) = (shape[0], shape[1], shape[2]);
            let frame_size = h * w;
            eprintln!("animate2D: streaming {nf} frames ({h}×{w})");
            for f in 0..nf {
                let slice = &data[f * frame_size .. (f + 1) * frame_size];
                write_frame(out, slice, h, w, f as f64)
                    .map_err(|e| format!("animate2D: write error: {e}"))?;
            }
            eprintln!("animate2D: done ({nf} frames)");
            Ok(nf)
        }

        // ── animate2D(f, …) — function form ───────────────────────────────────
        f_val @ (Val::Fn(..) | Val::Builtin(..)) => {
            let t_vals: Vec<f64> = match rest.len() {
                // animate2D(f, n) or animate2D(f, t_vals)
                1 => {
                    let tv = eval(&rest[0], env)?;
                    match tv {
                        Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
                        Val::Num(n) => (0..n as usize).map(|k| k as f64).collect(),
                        other => return Err(format!(
                            "animate2D: 2nd arg must be a 1-D tensor of timestamps or a count, got {}",
                            crate::eval::fmt_val(&other)
                        )),
                    }
                }
                // animate2D(f, t0, t1, n)
                3 => {
                    let t0 = eval(&rest[0], env)?.num("animate2D t0")?;
                    let t1 = eval(&rest[1], env)?.num("animate2D t1")?;
                    let n  = eval(&rest[2], env)?.num("animate2D n")? as usize;
                    if n < 2 { return Err("animate2D: n must be >= 2".into()); }
                    (0..n).map(|k| t0 + (t1 - t0) * k as f64 / (n - 1) as f64).collect()
                }
                n => return Err(format!(
                    "animate2D: function form expects 1 or 3 timestamp args (after fps strip), got {n}"
                )),
            };

            let n_frames = t_vals.len();
            eprintln!("animate2D: computing and streaming {n_frames} frames …");

            let (first_data, rows, cols) = call_for_frame(&f_val, t_vals[0], env)?;
            eprintln!("animate2D: grid {rows}×{cols}");
            write_frame(out, &first_data, rows, cols, t_vals[0])
                .map_err(|e| format!("animate2D: write error: {e}"))?;

            for &t in &t_vals[1..] {
                let (frame_data, fr, fc) = call_for_frame(&f_val, t, env)?;
                if fr != rows || fc != cols {
                    return Err(format!(
                        "animate2D: frame at t={t} has shape [{fr},{fc}], expected [{rows},{cols}]"
                    ));
                }
                write_frame(out, &frame_data, rows, cols, t)
                    .map_err(|e| format!("animate2D: write error: {e}"))?;
            }

            eprintln!("animate2D: done ({n_frames} frames)");
            Ok(n_frames)
        }

        other => Err(format!(
            "animate2D: first arg must be a 3-D tensor or a function, got {}",
            crate::eval::fmt_val(&other)
        )),
    }
}

// ── Extract optional fps from the tail of `rest`, based on first-arg type ─────
//
// Returns (fps, core_rest) where core_rest is the slice without the fps expr.
// Default fps = 30.
//
// Rules:
//   first = Tensor(3D)  + rest.len()==0  → fps=default, core=[]
//   first = Tensor(3D)  + rest.len()==1  → fps=rest[0],  core=[]
//   first = Fn/Builtin  + rest.len()==1  → fps=default, core=rest      (n or t_vals)
//   first = Fn/Builtin  + rest.len()==2  → fps=rest[1],  core=rest[..1]
//   first = Fn/Builtin  + rest.len()==4  → fps=default, core=rest      (t0,t1,n)  — old 4-arg
//   first = Fn/Builtin  + rest.len()==5  → fps=rest[4],  core=rest[..4]   (wait: t0,t1,n,fps = 3 non-f args + fps)
//
// NOTE: args[0] is already consumed as `first`; `rest = &args[1..]`.

fn extract_fps<'a>(
    first: &Val,
    rest: &'a [Expr],
    env: &Env,
) -> Result<(f64, &'a [Expr]), String> {
    const DEFAULT: f64 = 30.0;

    match first {
        Val::Tensor { shape, .. } if shape.len() == 3 => match rest.len() {
            0 => Ok((DEFAULT, &rest[..0])),
            1 => {
                let fps = eval(&rest[0], env)?.num("animate2D fps")?;
                Ok((fps, &rest[..0]))
            }
            n => Err(format!(
                "animate2D: tensor form takes 0 or 1 extra args (fps), got {n}"
            )),
        },

        Val::Fn(..) | Val::Builtin(..) => match rest.len() {
            1 => Ok((DEFAULT, rest)),           // (f, n_or_tvals)
            2 => {                              // (f, n_or_tvals, fps)
                let fps = eval(&rest[1], env)?.num("animate2D fps")?;
                Ok((fps, &rest[..1]))
            }
            3 => Ok((DEFAULT, rest)),           // (f, t0, t1, n)
            4 => {                              // (f, t0, t1, n, fps)
                let fps = eval(&rest[3], env)?.num("animate2D fps")?;
                Ok((fps, &rest[..3]))
            }
            n => Err(format!(
                "animate2D: function form expects 1–4 extra args, got {n}"
            )),
        },

        other => Err(format!(
            "animate2D: first arg must be a 3-D tensor or a function, got {}",
            crate::eval::fmt_val(other)
        )),
    }
}

// ── Public entry points ────────────────────────────────────────────────────────

/// animate2D_raw — write MXFR frames to stdout (for manual piping).
/// Does not accept an fps argument (there is no animator to tell).
pub fn eval_animate2d_raw(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.is_empty() {
        return Err(
            "animate2D_raw(T) | animate2D_raw(f, n) | animate2D_raw(f, t_vals) | animate2D_raw(f, t0, t1, n)".into()
        );
    }
    let first = eval(&args[0], env)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let n = stream_frames(first, &args[1..], env, &mut out)?;
    Ok(Val::Num(n as f64))
}

/// animate2D — spawn wgpu_animator and stream frames to it.
/// Optional fps arg at the end of any calling convention (default 30).
pub fn eval_animate2d(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.is_empty() {
        return Err(concat!(
            "animate2D(T [,fps]) | animate2D(f, n [,fps]) | ",
            "animate2D(f, t_vals [,fps]) | animate2D(f, t0, t1, n [,fps])"
        ).into());
    }

    let first = eval(&args[0], env)?;
    let (fps, core_rest) = extract_fps(&first, &args[1..], env)?;

    let animator = find_animator();
    let fps_str = format!("{}", fps as u32);

    let mut cmd_args: Vec<String> = vec![
        "--stdin".into(), "--colormap".into(), "heat".into(),
        "--fps".into(), fps_str,
    ];
    if let Ok(title) = std::env::var("WGPU_TITLE") {
        cmd_args.push("--title".into());
        cmd_args.push(title);
    }
    eprintln!("animate2D: spawning '{animator}'");

    let mut child = std::process::Command::new(&animator)
        .args(&cmd_args)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!(
            "animate2D: failed to spawn '{animator}': {e}\n  hint: set WGPU_ANIMATOR=/path/to/wgpu_animator"
        ))?;

    let n = {
        // Write frames to child stdin, then drop it so the animator sees EOF.
        let child_stdin = child.stdin.take()
            .ok_or_else(|| "animate2D: could not get animator stdin".to_string())?;
        let mut out = std::io::BufWriter::new(child_stdin);
        let n = stream_frames(first, core_rest, env, &mut out)?;
        // `out` dropped here → stdin closed → EOF to animator
        n
    };

    let status = child.wait()
        .map_err(|e| format!("animate2D: error waiting for animator: {e}"))?;
    if !status.success() {
        eprintln!("animate2D: animator exited with {status}");
    }

    Ok(Val::Num(n as f64))
}
