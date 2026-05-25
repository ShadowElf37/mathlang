use std::sync::atomic::{AtomicUsize, Ordering};
use plotters::prelude::*;
use crate::ast::Expr;
use crate::eval::{Val, Env, eval, call_fn1};

static GRAPH_N: AtomicUsize = AtomicUsize::new(1);

pub fn eval_graph(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.is_empty() || args.len() > 3 {
        return Err("graph(f) or graph(f, a, b)".into());
    }
    let a = if args.len() >= 2 { eval(&args[1], env)?.num("a")? } else { -10.0 };
    let b = if args.len() >= 3 { eval(&args[2], env)?.num("b")? } else {  10.0 };
    if a >= b { return Err("graph: a must be less than b".into()); }

    let w = 900u32;
    let h = 600u32;
    let n = (w * 2) as usize;

    // Sample the function, splitting into continuous segments at discontinuities.
    let mut segments: Vec<Vec<(f32, f32)>> = vec![vec![]];
    let mut valid_ys: Vec<f32> = vec![];

    for i in 0..=n {
        let x = a + (b - a) * i as f64 / n as f64;
        let y = match call_fn1(&args[0], Val::Num(x), env) {
            Ok(v) => match v.num("graph") { Ok(r) => r as f32, Err(_) => f32::NAN },
            Err(_) => f32::NAN,
        };
        if y.is_finite() {
            valid_ys.push(y);
            segments.last_mut().unwrap().push((x as f32, y));
        } else if !segments.last().unwrap().is_empty() {
            segments.push(vec![]);
        }
    }

    if valid_ys.is_empty() {
        return Err("graph: function produced no finite values in range".into());
    }

    // Use 5th–95th percentile for y range to handle singularities gracefully.
    valid_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = valid_ys[valid_ys.len() / 20];
    let hi = valid_ys[valid_ys.len() * 19 / 20];
    let pad = (hi - lo).abs().max(1e-6) * 0.08;
    let y_lo = lo - pad;
    let y_hi = hi + pad;

    let n = GRAPH_N.fetch_add(1, Ordering::Relaxed);
    let filename = format!("graph_{n}.png");

    let root = BitMapBackend::new(&filename, (w, h)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| format!("graph: {e}"))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(24)
        .x_label_area_size(36)
        .y_label_area_size(54)
        .build_cartesian_2d(a as f32..b as f32, y_lo..y_hi)
        .map_err(|e| format!("graph: {e}"))?;

    chart.configure_mesh()
        .light_line_style(RGBColor(235, 235, 235))
        .bold_line_style(RGBColor(200, 200, 200))
        .axis_style(RGBColor(100, 100, 100))
        .draw()
        .map_err(|e| format!("graph: {e}"))?;

    let blue = RGBColor(30, 100, 220);
    for seg in segments {
        if seg.len() < 2 { continue; }
        chart.draw_series(LineSeries::new(seg, blue.stroke_width(2)))
            .map_err(|e| format!("graph: {e}"))?;
    }

    root.present().map_err(|e| format!("graph: {e}"))?;
    eprintln!("saved: {filename}");
    Ok(Val::Num(0.0))
}
