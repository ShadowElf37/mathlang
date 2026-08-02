// mumax — a transpiler backend that drives mumax3 from mathlang. Each op reads a
// `world` record (the same shape as mag.world, plus optional physics fields) and a
// magnetization tensor, generates a .mx3 script, runs mumax3, and returns the
// resulting magnetization as a tensor. The output .ovf comes straight back as the
// function's return value, so mumax feels like the native `mag` library — swap
// `mag.relax` -> `mumax.relax` on identical code.
//
// mathlang has no string type, so all .mx3 text (and the file/process work) lives
// here in Rust.
use crate::eval::{Val, Tup, Env, fmt_val};
use crate::repl::{load_ovf, write_ovf};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const NAMES: &[&str] = &["relax", "run", "reverse", "evolve"];

pub fn members() -> Vec<(String, Val)> {
    NAMES.iter().map(|n| (n.to_string(), Val::Builtin(n.to_string()))).collect()
}

// ── world-record readers ────────────────────────────────────────────────────────
fn req(w: &Tup, key: &str) -> Result<f64, String> {
    w.lookup(key).ok_or_else(|| format!("mumax: world.{key} is required"))?
        .clone().num(key)
}
fn opt(w: &Tup, key: &str) -> Result<Option<f64>, String> {
    match w.lookup(key) { Some(v) => Ok(Some(v.clone().num(key)?)), None => Ok(None) }
}
fn as_nums(v: Val, ctx: &str) -> Result<Vec<f64>, String> {
    match v {
        Val::Tensor { data, .. } => Ok(data.into_vec()),
        Val::Tuple(t) => t.into_iter().map(|x| x.num(ctx)).collect(),
        Val::Num(n) => Ok(vec![n]),
        other => Err(format!("mumax: {ctx} must be a vector, got {}", fmt_val(&other))),
    }
}

// (nx, ny, nz, dx, dy, dz) from the world — accepts flat dx/dy/dz or a `cell` tuple.
fn geometry(w: &Tup) -> Result<(usize, usize, usize, f64, f64, f64), String> {
    let nx = req(w, "nx")? as usize;
    let ny = req(w, "ny")? as usize;
    let nz = opt(w, "nz")?.map(|x| x as usize).unwrap_or(1);
    let (dx, dy, dz) = if w.lookup("dx").is_some() {
        (req(w, "dx")?, req(w, "dy")?, req(w, "dz")?)
    } else if let Some(Val::Tuple(c)) = w.lookup("cell") {
        if c.len() < 3 { return Err("mumax: world.cell must be (dx, dy, dz)".into()); }
        (c[0].clone().num("cell.dx")?, c[1].clone().num("cell.dy")?, c[2].clone().num("cell.dz")?)
    } else {
        return Err("mumax: world needs dx,dy,dz (or a cell=(dx,dy,dz) tuple)".into());
    };
    Ok((nx, ny, nz, dx, dy, dz))
}

fn tensor_parts(v: &Val, ctx: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Tensor { data, shape } => Ok((data.clone().into_vec(), shape.clone())),
        other => Err(format!("mumax: {ctx} must be a tensor, got {}", fmt_val(other))),
    }
}

// ── mumax binary + scratch dir ──────────────────────────────────────────────────
static SCRATCH_N: AtomicUsize = AtomicUsize::new(0);

fn find_mumax() -> String {
    std::env::var("MUMAX3").unwrap_or_else(|_| "mumax3".to_string())
}

fn scratch_dir() -> Result<PathBuf, String> {
    let base = std::env::var("MUMAX3_SCRATCH").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let n = SCRATCH_N.fetch_add(1, Ordering::Relaxed);
    let dir = base.join(format!("mumax3-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("mumax: scratch dir {}: {e}", dir.display()))?;
    Ok(dir)
}

// ── the shared .mx3 emitter ─────────────────────────────────────────────────────
// Static world params (geometry, material, anisotropy, DMI, magnetoelastic const,
// temperature). `drive` holds per-op excitations (B_ext / strain); `op` is the
// dynamics line ("relax()" or "run(t)").
fn emit_and_run(w: &Tup, m0: &Val, drive: &str, op: &str) -> Result<Val, String> {
    let (nx, ny, nz, dx, dy, dz) = geometry(w)?;
    let (m0data, m0shape) = tensor_parts(m0, "m0")?;
    if m0shape != vec![nx, ny, 3] && !(nz > 1 && m0shape == vec![nx, ny, nz, 3]) {
        return Err(format!("mumax: m0 shape {m0shape:?} must be [{nx},{ny},3]"));
    }
    let dir = scratch_dir()?;
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();

    // initial magnetization
    write_ovf(&p("m0.ovf"), &m0data, nx, ny, nz, 3, dx, dy, dz)?;

    let mut s = String::new();
    s += &format!("SetGridSize({nx}, {ny}, {nz})\n");
    s += &format!("SetCellSize({dx:e}, {dy:e}, {dz:e})\n");
    // geometry mask (optional)
    if let Some(g) = w.lookup("geom") {
        let (gdata, _gshape) = tensor_parts(g, "world.geom")?;
        write_ovf(&p("geom.ovf"), &gdata, nx, ny, nz, 1, dx, dy, dz)?;
        s += &format!("SetGeom(VoxelShape(LoadFile(\"{}\"), {dx:e}, {dy:e}, {dz:e}))\n", p("geom.ovf"));
    }
    // material
    s += &format!("Msat = {:e}\n", req(w, "Msat")?);
    s += &format!("Aex = {:e}\n", req(w, "Aex")?);
    if let Some(a) = opt(w, "alpha")? { s += &format!("alpha = {a:e}\n"); }
    // anisotropy
    if let Some(k) = opt(w, "Ku1")? { s += &format!("Ku1 = {k:e}\n"); }
    if let Some(k) = opt(w, "Ku2")? { s += &format!("Ku2 = {k:e}\n"); }
    if let Some(u) = w.lookup("anisU") {
        let u = as_nums(u.clone(), "anisU")?;
        if u.len() != 3 { return Err("mumax: world.anisU must be a 3-vector".into()); }
        s += &format!("anisU = vector({:e}, {:e}, {:e})\n", u[0], u[1], u[2]);
    }
    // DMI
    if let Some(d) = opt(w, "Dind")? { s += &format!("Dind = {d:e}\n"); }
    if let Some(d) = opt(w, "Dbulk")? { s += &format!("Dbulk = {d:e}\n"); }
    // magnetoelastic coupling constants (static); strain is a drive
    if let Some(b) = opt(w, "B1")? { s += &format!("B1 = {b:e}\n"); }
    if let Some(b) = opt(w, "B2")? { s += &format!("B2 = {b:e}\n"); }
    // temperature + seeds
    if let Some(t) = opt(w, "Temp")? {
        let seed = opt(w, "seed")?.unwrap_or(0.0) as i64;
        s += &format!("Temp = {t:e}\nrandSeed({seed})\nThermSeed({seed})\n");
    }
    if let Some(e) = opt(w, "EnableDemag")? {
        s += &format!("EnableDemag = {}\n", if e != 0.0 { "true" } else { "false" });
    }

    // initial state, per-op drive, dynamics, output
    s += &format!("m.LoadFile(\"{}\")\n", p("m0.ovf"));
    s += drive;
    s += "OutputFormat = OVF2_TEXT\n";
    s += op;
    s += "\nsaveAs(m, \"out\")\n";

    let script = p("script.mx3");
    std::fs::write(&script, &s).map_err(|e| format!("mumax: write script: {e}"))?;

    // Output goes to a SEPARATE subdir so we never clobber script.mx3 / m0.ovf.
    let outdir = dir.join("out");
    let outdir_s = outdir.to_string_lossy().into_owned();

    // run mumax3 headless
    let bin = find_mumax();
    let out = std::process::Command::new(&bin)
        .args(["-http", "", "-f", "-o", &outdir_s, &script])
        .output()
        .map_err(|e| format!("mumax: failed to run '{bin}': {e}\n  hint: set MUMAX3=/path/to/mumax3 (or add it to PATH)"))?;
    if !out.status.success() {
        return Err(format!("mumax3 failed ({}):\n{}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    load_ovf(&outdir.join("out.ovf").to_string_lossy())
}

// ── drive builders ──────────────────────────────────────────────────────────────
fn drive_bext(bext: &[f64]) -> Result<String, String> {
    if bext.len() != 3 { return Err("mumax: B_ext must be a 3-vector".into()); }
    Ok(format!("B_ext = vector({:e}, {:e}, {:e})\n", bext[0], bext[1], bext[2]))
}

// evolve's `drive` record: optional Bext (3-vector) and/or strain (6-tuple).
fn drive_record(rec: &Val) -> Result<String, String> {
    let t = match rec { Val::Tuple(t) => t, other => return Err(format!("mumax.evolve: drive must be a record, got {}", fmt_val(other))) };
    let mut s = String::new();
    if let Some(b) = t.lookup("Bext") { s += &drive_bext(&as_nums(b.clone(), "drive.Bext")?)?; }
    if let Some(e) = t.lookup("strain") {
        let e = as_nums(e.clone(), "drive.strain")?;
        if e.len() != 6 { return Err("mumax.evolve: strain must be (exx,eyy,ezz,exy,exz,eyz)".into()); }
        let names = ["exx", "eyy", "ezz", "exy", "exz", "eyz"];
        for (n, v) in names.iter().zip(&e) { s += &format!("{n} = {v:e}\n"); }
    }
    Ok(s)
}

// ── dispatch ────────────────────────────────────────────────────────────────────
pub fn dispatch(name: &str, vals: Vec<Val>, _env: &Env) -> Result<Val, String> {
    let world = |v: &Val| -> Result<Tup, String> {
        match v { Val::Tuple(t) => Ok(t.clone()), other => Err(format!("mumax.{name}: world must be a record, got {}", fmt_val(other))) }
    };
    match name {
        "relax" => {
            if vals.len() != 2 { return Err("mumax.relax(world, m0)".into()); }
            emit_and_run(&world(&vals[0])?, &vals[1], "", "relax()")
        }
        "run" => {
            if vals.len() != 3 { return Err("mumax.run(world, m0, t)".into()); }
            let t = vals[2].clone().num("t")?;
            emit_and_run(&world(&vals[0])?, &vals[1], "", &format!("run({t:e})"))
        }
        "reverse" => {
            if vals.len() != 4 { return Err("mumax.reverse(world, m0, Bext, t)".into()); }
            let bext = as_nums(vals[2].clone(), "Bext")?;
            let t = vals[3].clone().num("t")?;
            emit_and_run(&world(&vals[0])?, &vals[1], &drive_bext(&bext)?, &format!("run({t:e})"))
        }
        "evolve" => {
            if vals.len() != 4 { return Err("mumax.evolve(world, m0, drive, t)".into()); }
            let drive = drive_record(&vals[2])?;
            let t = vals[3].clone().num("t")?;
            emit_and_run(&world(&vals[0])?, &vals[1], &drive, &format!("run({t:e})"))
        }
        _ => Err(format!("mumax: unknown operation '{name}'")),
    }
}
