/// animate2D — stream 2D tensor frames to stdout in MXFR format for wgpu_animator.
///
/// Protocol (from animator/PROTOCOL.md):
///   Per frame, write to stdout (little-endian):
///     b"MXFR"  — 4 magic bytes
///     W: u32   — width  (cols)
///     H: u32   — height (rows)
///     t: f64   — timestamp
///     data: W*H f32 values, row-major, row-0 = top of display
///
/// Calling conventions:
///   animate2D(T)              — T: 3-D Tensor [n_frames, H, W]
///   animate2D(f, t_vals)      — f: t → 2-D Tensor, t_vals: 1-D Tensor of timestamps
///   animate2D(f, t0, t1, n)   — f: t → 2-D Tensor, n frames linspace(t0, t1, n)
///
/// Usage:
///   m 'animate2D(solver, 0, 20, 50)' | ./animator/target/release/wgpu_animator --stdin --colormap heat
///
/// All human-readable output goes to stderr.  The binary MXFR stream goes to stdout.

use std::io::Write;
use crate::ast::Expr;
use crate::eval::{Val, Env, eval, apply_val};

// ── Binary helpers ─────────────────────────────────────────────────────────────

fn write_frame(out: &mut impl Write, data: &[f64], rows: usize, cols: usize, t: f64)
    -> std::io::Result<()>
{
    // Header: magic + W + H + timestamp
    out.write_all(b"MXFR")?;
    out.write_all(&(cols as u32).to_le_bytes())?;
    out.write_all(&(rows as u32).to_le_bytes())?;
    out.write_all(&t.to_le_bytes())?;
    // Pixel data as f32 LE
    for &v in data {
        out.write_all(&(v as f32).to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

// ── Call a function val with one f64 argument, expect 2-D Tensor ──────────────

fn call_for_frame(f: &Val, t: f64, env: &Env) -> Result<(Vec<f64>, usize, usize), String> {
    let result = apply_val(f.clone(), vec![Val::Num(t)], env)?;
    match result {
        Val::Tensor { data, shape } if shape.len() == 2 => {
            Ok((data, shape[0], shape[1]))
        }
        Val::Tensor { data, shape } if shape.len() == 1 => {
            // 1-D result: treat as single-row
            let cols = data.len();
            Ok((data, 1, cols))
        }
        other => Err(format!(
            "animate2D: f(t) must return a 2-D tensor, got {}",
            crate::eval::fmt_val(&other)
        )),
    }
}

// ── Public entry point ─────────────────────────────────────────────────────────

pub fn eval_animate2d(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.is_empty() {
        return Err(
            "animate2D(T) | animate2D(f, t_vals) | animate2D(f, t0, t1, n)".into()
        );
    }

    let first = eval(&args[0], env)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    match first {
        // ── animate2D(T) — 3-D Tensor [n_frames, H, W] ───────────────────────
        Val::Tensor { ref data, ref shape } if shape.len() == 3 => {
            let (nf, h, w) = (shape[0], shape[1], shape[2]);
            let frame_size = h * w;
            eprintln!("animate2D: streaming {nf} frames ({h}×{w}) to stdout");
            for f in 0..nf {
                let slice = &data[f * frame_size .. (f + 1) * frame_size];
                write_frame(&mut out, slice, h, w, f as f64)
                    .map_err(|e| format!("animate2D: write error: {e}"))?;
            }
            eprintln!("animate2D: done ({nf} frames)");
            Ok(Val::Num(nf as f64))
        }

        // ── animate2D(f, …) — function form ───────────────────────────────────
        f_val @ (Val::Fn(..) | Val::Builtin(..)) => {
            let t_vals: Vec<f64> = match args.len() {
                // animate2D(f, t_vals)  — t_vals is a 1-D tensor
                2 => {
                    let tv = eval(&args[1], env)?;
                    match tv {
                        Val::Tensor { data, shape } if shape.len() == 1 => data,
                        Val::Num(n) => {
                            // animate2D(f, n) — n integer frames 0..n-1
                            (0..n as usize).map(|k| k as f64).collect()
                        }
                        other => return Err(format!(
                            "animate2D: 2nd arg must be a 1-D tensor of timestamps or a count, got {}",
                            crate::eval::fmt_val(&other)
                        )),
                    }
                }
                // animate2D(f, t0, t1, n)  — linspace
                4 => {
                    let t0 = eval(&args[1], env)?.num("animate2D t0")?;
                    let t1 = eval(&args[2], env)?.num("animate2D t1")?;
                    let n  = eval(&args[3], env)?.num("animate2D n")? as usize;
                    if n < 2 { return Err("animate2D: n must be >= 2".into()); }
                    (0..n).map(|k| t0 + (t1 - t0) * k as f64 / (n - 1) as f64).collect()
                }
                other => return Err(format!(
                    "animate2D: expected 1, 2, or 4 args, got {other}"
                )),
            };

            let n_frames = t_vals.len();
            eprintln!("animate2D: computing and streaming {n_frames} frames …");

            // Compute first frame to determine grid size
            let (first_data, rows, cols) = call_for_frame(&f_val, t_vals[0], env)?;
            eprintln!("animate2D: grid {rows}×{cols}, streaming to stdout");
            write_frame(&mut out, &first_data, rows, cols, t_vals[0])
                .map_err(|e| format!("animate2D: write error: {e}"))?;

            for &t in &t_vals[1..] {
                let (frame_data, fr, fc) = call_for_frame(&f_val, t, env)?;
                if fr != rows || fc != cols {
                    return Err(format!(
                        "animate2D: frame at t={t} has shape [{fr},{fc}], expected [{rows},{cols}]"
                    ));
                }
                write_frame(&mut out, &frame_data, rows, cols, t)
                    .map_err(|e| format!("animate2D: write error: {e}"))?;
            }

            eprintln!("animate2D: done ({n_frames} frames)");
            Ok(Val::Num(n_frames as f64))
        }

        other => Err(format!(
            "animate2D: first arg must be a 3-D tensor or a function, got {}",
            crate::eval::fmt_val(&other)
        )),
    }
}
