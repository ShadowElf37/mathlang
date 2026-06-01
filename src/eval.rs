use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::cell::{Cell as StdCell, RefCell};
use crate::ast::{Expr, BlockStmt, Op, Def, TypeHint};
use crate::vm::{Instruction, LoopForm};

// ── Recursion guard ───────────────────────────────────────────────────────────
// User-function calls all funnel through apply_fn_direct. A thread-local depth
// counter turns runaway recursion into a catchable error instead of a native
// stack overflow (which would abort the whole process / REPL session). The limit
// is generous because the evaluator runs on a large-stack worker thread (see
// main.rs / repl.rs); it only needs to trip before the real stack does.
const MAX_CALL_DEPTH: u32 = 100_000;

thread_local! {
    static CALL_DEPTH: StdCell<u32> = const { StdCell::new(0) };
}

/// RAII guard: increments the call-depth counter on entry, decrements on drop.
/// `DepthGuard::enter()` returns an error if the limit is exceeded.
struct DepthGuard;

impl DepthGuard {
    fn enter() -> Result<DepthGuard, String> {
        CALL_DEPTH.with(|d| {
            let depth = d.get() + 1;
            if depth > MAX_CALL_DEPTH {
                Err(format!("recursion limit exceeded ({MAX_CALL_DEPTH} nested calls); \
                             for long iterations use a flat loop (sum/prod over a range, \
                             optionally driving a cell) instead of deep recursion"))
            } else {
                d.set(depth);
                Ok(DepthGuard)
            }
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

// The bytecode instruction set (`Instruction`, `LoopForm`) lives in `src/vm.rs`
// so the GPU backend can import it without pulling in this whole module (TODO 1f).
// The VM executor (`run_vm`) and compiler (`Compiler`) stay here — they need `Val`.

// ── TData: Arc-wrapped tensor data ────────────────────────────────────────────
//
// Cloning TData is O(1) — it just increments the Arc refcount.  The underlying
// Vec<f64> is only deep-copied when Arc::make_mut detects multiple owners
// (copy-on-write).  This eliminates the O(n²) memory explosion that previously
// occurred from env.vars.clone() deep-copying all tensor data on every function
// definition or call (see TODO #10).

#[derive(Debug)]
pub struct TData(Arc<Vec<f64>>);

impl TData {
    /// Wrap a Vec<f64> in a new TData.  Construction is always O(n) — we don't
    /// avoid building the Vec, only avoid copying it on subsequent clones.
    #[inline] pub fn new(v: Vec<f64>) -> Self { TData(Arc::new(v)) }

    /// Unwrap to Vec<f64>.  O(1) if this is the sole owner; CoW-clone otherwise.
    #[inline] pub fn into_vec(self) -> Vec<f64> {
        Arc::try_unwrap(self.0).unwrap_or_else(|a| (*a).clone())
    }
}

/// O(1) clone — just increments the Arc refcount.
impl Clone for TData {
    #[inline] fn clone(&self) -> Self { TData(Arc::clone(&self.0)) }
}

/// Read-only deref to Vec<f64>:  data.len(), data[i], data.iter(), &data[a..b] all work.
impl std::ops::Deref for TData {
    type Target = Vec<f64>;
    #[inline] fn deref(&self) -> &Vec<f64> { &self.0 }
}

/// CoW write deref — only clones the Vec if the Arc has multiple owners.
impl std::ops::DerefMut for TData {
    #[inline] fn deref_mut(&mut self) -> &mut Vec<f64> { Arc::make_mut(&mut self.0) }
}

/// Consuming iteration: data.into_iter() → Iterator<Item=f64>.
/// O(1) if sole owner (Arc::try_unwrap succeeds), CoW-clone otherwise.
impl IntoIterator for TData {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;
    #[inline] fn into_iter(self) -> Self::IntoIter { self.into_vec().into_iter() }
}

// ── Values ────────────────────────────────────────────────────────────────────

/// Type hints stored with a user function at creation time.
/// Empty (all None) for functions created without hints.
#[derive(Debug, Clone, Default)]
pub struct FnSig {
    pub params: Vec<Option<TypeHint>>,  // parallel to Val::Fn params Vec<String>
    pub ret:    Option<TypeHint>,
}

/// Boundary condition for a grid axis (governs the finite-difference stencil in `d`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BC { Periodic, Neumann }

/// A differential-form field is covariant (Form, the default) or contravariant
/// (Vector). On a Euclidean Cartesian grid the two are numerically identical
/// component-by-component; the tag tracks intent (raise/lower flip it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Variance { Form, Vector }

/// A k-form (or k-vector) field sampled on a regular Cartesian grid.
///
/// `data` is flat row-major with logical shape `grid ++ [ncomp]`, where
/// `ncomp = C(n, degree)` and the components are the sorted `degree`-subsets of
/// `0..n` in lexicographic order (n = grid.len()). `spacing`/`lo`/`bc`/`metric`
/// are per spatial axis.
///
/// Two distinct per-axis quantities, deliberately separate:
///   - `spacing` (dx): the discretization step. Enters ONLY the exterior
///     derivative `d` (finite differences ÷ dx). `d` is otherwise metric-free.
///   - `metric` (diagonal g_ii): the geometry/signature. Enters `hodge`,
///     `raise`/`lower`, `codiff`, `lap`. Euclidean = all +1; Minkowski =
///     e.g. (-1, 1, 1, 1). Constant over the grid (position-dependent metrics
///     are a future extension — and `d` already needs no change for them).
#[derive(Debug, Clone)]
pub struct FieldVal {
    pub data:     TData,
    pub grid:     Vec<usize>,
    pub spacing:  Vec<f64>,
    pub lo:       Vec<f64>,
    pub bc:       Vec<BC>,
    pub metric:   Vec<f64>,   // diagonal metric g_ii (constant); Euclidean = all 1
    pub degree:   usize,
    pub variance: Variance,
}

impl FieldVal {
    /// Number of components, C(n, degree).
    pub fn ncomp(&self) -> usize { binomial(self.grid.len(), self.degree) }
}

/// Binomial coefficient C(n, k).
pub fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    let k = k.min(n - k);
    let mut r = 1usize;
    for i in 0..k { r = r * (n - i) / (i + 1); }
    r
}

/// Arithmetic on fields: operate component-wise on the data, carrying the
/// geometry. field∘field requires matching grid/degree/variance/spacing/bc/metric
/// (a mismatch is a real error — adding fields on different grids is a bug); a
/// field combines with a scalar or a same-shape tensor by broadcasting the data.
fn field_binop(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    let meta: Arc<FieldVal> = match (&lv, &rv) {
        (Val::Field(a), _) => a.clone(),
        (_, Val::Field(b)) => b.clone(),
        _ => unreachable!(),
    };
    if let (Val::Field(a), Val::Field(b)) = (&lv, &rv) {
        if a.grid != b.grid || a.degree != b.degree || a.variance != b.variance
            || a.spacing != b.spacing || a.bc != b.bc || a.metric != b.metric {
            return Err("field op: incompatible fields (grid, degree, variance, spacing, bc, metric must match)".into());
        }
    }
    let lt = match lv { Val::Field(ref a) => field_data_as_tensor(a), other => other };
    let rt = match rv { Val::Field(ref b) => field_data_as_tensor(b), other => other };
    match binop_tensor(lt, op, rt)? {
        Val::Tensor { data, .. } => Ok(crate::ns::forms::with_data(&meta, data.into_vec())),
        other => Err(format!("field op produced a non-real result: {}", fmt_val(&other))),
    }
}

/// View a field's component data as a plain Tensor (the component axis is dropped
/// for a 0-form, so a scalar field displays/extracts as an ordinary grid tensor).
pub(crate) fn field_data_as_tensor(f: &FieldVal) -> Val {
    let ncomp = f.ncomp();
    let shape = if ncomp == 1 {
        f.grid.clone()
    } else {
        let mut s = f.grid.clone();
        s.push(ncomp);
        s
    };
    Val::Tensor { data: f.data.clone(), shape }
}

#[derive(Clone, Debug)]
pub enum Val {
    Num(f64),
    Complex(f64, f64),
    /// Fn(params, body, captured_env)
    /// `captured_env` is Arc-wrapped so cloning a closure is O(1) regardless
    /// of how many functions are in scope.
    /// Fn(params, body, captured_env, bytecode_cache, sig)
    /// `bytecode_cache` is an Arc<OnceLock> so all clones share the compiled code.
    /// Initialised on first call via apply_fn_direct; None means fall back to tree-walk.
    Fn(Vec<String>, Expr, Arc<HashMap<String, Val>>, Arc<OnceLock<Option<Vec<Instruction>>>>, Arc<FnSig>),
    Builtin(String),
    Tuple(Vec<Val>),
    /// Real-valued tensor (row-major flat storage).
    /// `data` is a TData (Arc<Vec<f64>>), so cloning a Tensor Val is O(1).
    Tensor { data: TData, shape: Vec<usize> },
    /// Complex tensor: two parallel real arrays (re, im) with identical shape.
    /// Both re and im are TData for O(1) cloning.
    ComplexTensor { re: TData, im: TData, shape: Vec<usize> },
    /// Mutable cell — a shared, reference-counted mutable container.
    /// Cloning a Cell shares the same RefCell (identity semantics).
    /// Created with cell(v); read with get(c); written with set(c, v).
    Cell(Arc<RefCell<Val>>),
    /// A differential-form / vector field on a regular grid (see FieldVal).
    /// Arc-wrapped for O(1) clone.
    Field(Arc<FieldVal>),
    /// A namespace: a map from member name to value, accessed with `ns.member`.
    /// Builtin namespaces (ops, special, …) are registered in Env::new;
    /// user namespaces are built by an `!namespace`-headed included file.
    Namespace(Arc<HashMap<String, Val>>),
}

impl Val {
    pub fn num(self, ctx: &str) -> Result<f64, String> {
        match self {
            Val::Num(n)               => Ok(n),
            Val::Complex(..)          => Err(format!("{ctx}: expected a real number, got complex")),
            Val::Fn(..)               => Err(format!("{ctx}: expected a number, got a function")),
            Val::Builtin(n)           => Err(format!("{ctx}: expected a number, got builtin '{n}'")),
            Val::Tuple(..)            => Err(format!("{ctx}: expected a number, got a tuple")),
            Val::Tensor { .. }        => Err(format!("{ctx}: expected a number, got a tensor")),
            Val::ComplexTensor { .. } => Err(format!("{ctx}: expected a number, got a complex tensor")),
            Val::Cell(..)             => Err(format!("{ctx}: expected a number, got a cell (use get())")),
            Val::Namespace(..)        => Err(format!("{ctx}: expected a number, got a namespace")),
            Val::Field(..)            => Err(format!("{ctx}: expected a number, got a field")),
        }
    }

    /// Construct a new user function with a fresh (empty) bytecode cache and no type hints.
    pub fn make_fn(params: Vec<String>, body: Expr, captured: Arc<HashMap<String, Val>>) -> Self {
        Val::Fn(params, body, captured, Arc::new(OnceLock::new()), Arc::new(FnSig::default()))
    }

    /// Construct a user function with bytecode pre-filled — zero recompile cost on first call.
    /// Used by MakeClosure to pass eagerly-compiled inner code to the resulting Val::Fn.
    pub fn make_fn_compiled(params: Vec<String>, body: Expr, captured: Arc<HashMap<String, Val>>, code: Vec<Instruction>) -> Self {
        let lock = OnceLock::new();
        let _ = lock.set(Some(code));
        Val::Fn(params, body, captured, Arc::new(lock), Arc::new(FnSig::default()))
    }

    /// Construct a user function with type signature hints.
    pub fn make_fn_with_sig(params: Vec<String>, sig: FnSig, body: Expr, captured: Arc<HashMap<String, Val>>) -> Self {
        Val::Fn(params, body, captured, Arc::new(OnceLock::new()), Arc::new(sig))
    }
}

// ── Environment ───────────────────────────────────────────────────────────────
//
// vars is Arc-wrapped so env.clone() is O(1).  Mutations (variable definitions,
// parameter binding) use Arc::make_mut() for copy-on-write semantics: the
// HashMap is only cloned when there are multiple outstanding references to it,
// which is rare in practice.

#[derive(Clone)]
pub struct Env {
    pub vars: Arc<HashMap<String, Val>>,
}

impl Env {
    /// Insert a variable into this env, CoW-cloning the HashMap only if needed.
    #[inline]
    pub fn define(&mut self, k: String, v: Val) {
        Arc::make_mut(&mut self.vars).insert(k, v);
    }
}

impl Env {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert("pi".into(),  Val::Num(std::f64::consts::PI));
        vars.insert("e".into(),   Val::Num(std::f64::consts::E));
        vars.insert("phi".into(), Val::Num(1.618033988749895));
        vars.insert("inf".into(), Val::Num(f64::INFINITY));
        vars.insert("i".into(),   Val::Complex(0.0, 1.0));
        // Flat (unqualified) builtins — the curated core. Niche functions live in
        // namespaces instead (see crate::ns): special, bits, stats, linalg, vec.
        for name in &[
            "abs", "re", "im", "arg", "conj", "sqrt", "exp", "ln",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "cbrt", "expm1",
            "sec", "csc", "cot",
            "floor", "ceil", "round",
            "trunc", "frac",
            "log", "log10", "log2",
            "sign", "signum", "id", "fact", "factorial", "ncr", "quadratic",
            "heaviside",
            "deg", "rad",
            "len", "length",
            "linspace", "range",
            "sort", "zip", "dot", "append", "concat", "flatten", "argmin", "argmax",
            "cumsum", "cumprod", "diff",
            "mean", "std",
            "compose", "partial",
            "filter", "reduce",
            "rand", "eps",
            "atan2", "min", "max", "pow", "hypot",
            "gcd", "lcm",
            "lt", "leq", "gt", "geq", "eq", "neq",
            "if",
            "fft", "ifft",
            "sum", "prod", "integral", "deriv", "map",
            "iterate", "scan",
            "cell", "get", "set",
            "field",
            // Tensor ops
            "tensor", "matrix", "zeros", "ones", "eye", "diag",
            "shape", "rows", "cols", "transpose", "trace", "norm",
            "row", "col", "matmul",
            "det", "inv", "solve",
            "eig", "eigvals",
            "hstack", "vstack", "tomat",
            "shift", "roll",
            "lingrid",
            "reshape", "permute", "cat", "squeeze", "unsqueeze",
            "dim",
        ] {
            vars.insert(name.to_string(), Val::Builtin(name.to_string()));
        }
        // Standard namespaces (ops, solver, special, bits, …) — loaded by force.
        crate::ns::register_all(&mut vars);
        Self { vars: Arc::new(vars) }
    }
}

// ───────────────────────── type inference ─────────────────────────

/// The static TypeHint of a runtime value (best-effort; used by `!type`).
pub fn hint_of_val(v: &Val) -> TypeHint {
    match v {
        Val::Num(_)               => TypeHint::Real,
        Val::Complex(..)          => TypeHint::Complex,
        Val::Tensor { .. }        => TypeHint::RealTensor,
        Val::ComplexTensor { .. } => TypeHint::ComplexTensor,
        Val::Tuple(_)             => TypeHint::Tuple,
        Val::Cell(_)              => TypeHint::Cell,
        Val::Fn(..) | Val::Builtin(_) => TypeHint::Fn,
        Val::Namespace(_)         => TypeHint::Any,
        Val::Field(_)             => TypeHint::Any,
    }
}

/// Map a type keyword (as it appears in a signature string) to a TypeHint.
fn hint_from_kw(s: &str) -> TypeHint {
    match s.trim() {
        "num"            => TypeHint::Num,
        "real"           => TypeHint::Real,
        "complex"        => TypeHint::Complex,
        "int"            => TypeHint::Int,
        "nat"            => TypeHint::Nat,
        "tensor"         => TypeHint::Tensor,
        "real tensor"    => TypeHint::RealTensor,
        "complex tensor" => TypeHint::ComplexTensor,
        "fn"             => TypeHint::Fn,
        "cell"           => TypeHint::Cell,
        "tuple"          => TypeHint::Tuple,
        _                => TypeHint::Any,
    }
}

/// Return-type of a builtin, parsed from the tail of its signature string.
fn builtin_ret_hint(name: &str) -> Option<TypeHint> {
    let sig = builtin_sig(name)?;
    let ret = sig.rsplit("->").next()?.trim();
    Some(hint_from_kw(ret))
}

fn is_tensor_hint(t: &TypeHint) -> bool {
    matches!(t, TypeHint::Tensor | TypeHint::RealTensor | TypeHint::ComplexTensor)
}
fn is_complex_hint(t: &TypeHint) -> bool {
    matches!(t, TypeHint::Complex | TypeHint::ComplexTensor)
}
fn is_real_hint(t: &TypeHint) -> bool {
    matches!(t, TypeHint::Real | TypeHint::Int | TypeHint::Nat | TypeHint::RealTensor)
}

/// Fuse the operand types of an arithmetic operator into a result type.
fn fuse_arith(a: TypeHint, b: TypeHint) -> TypeHint {
    use TypeHint::*;
    if a == Any || b == Any { return Any; }
    if is_tensor_hint(&a) || is_tensor_hint(&b) {
        if is_complex_hint(&a) || is_complex_hint(&b) { ComplexTensor }
        else if is_real_hint(&a) && is_real_hint(&b)  { RealTensor }
        else { Tensor }
    } else if a == Complex || b == Complex {
        Complex
    } else if a == Num || b == Num {
        Num
    } else {
        Real
    }
}

/// Fuse the element types of a tensor/array literal into the tensor's type.
fn fuse_elems(elems: &[&Expr], params: &HashMap<String, TypeHint>, env: &Env) -> TypeHint {
    use TypeHint::*;
    if elems.is_empty() { return RealTensor; }
    let mut any_complex = false;
    let mut any_unknown = false;
    for e in elems {
        match infer_type(e, params, env) {
            Complex | ComplexTensor => any_complex = true,
            Real | Int | Nat | Num | RealTensor | Tensor => {}
            _ => any_unknown = true,
        }
    }
    if any_complex { ComplexTensor } else if any_unknown { Tensor } else { RealTensor }
}

/// Best-effort static type inference for an expression, given the parameter
/// hints in scope. Returns `Any` when the type cannot be determined. Used by
/// `!type` to infer function return types (signature fusion).
pub fn infer_type(expr: &Expr, params: &HashMap<String, TypeHint>, env: &Env) -> TypeHint {
    use TypeHint::*;
    match expr {
        Expr::Num(_)     => Real,
        Expr::ImagLit(_) => Complex,
        Expr::Var(name) => {
            if let Some(h) = params.get(name) { return h.clone(); }
            if let Some(v) = env.vars.get(name) { return hint_of_val(v); }
            if builtin_sig(name).is_some() { return Fn; }
            match name.as_str() {
                "i"                                => Complex,
                "pi" | "e" | "phi" | "inf" | "tau" => Real,
                _                                  => Any,
            }
        }
        Expr::Neg(e) | Expr::Not(e) => infer_type(e, params, env),
        Expr::BinOp(l, op, r) => match op {
            Op::Lt | Op::Gt | Op::LtEq | Op::GtEq | Op::Eq | Op::Ne | Op::And | Op::Or => Real,
            _ => fuse_arith(infer_type(l, params, env), infer_type(r, params, env)),
        },
        Expr::Tuple(_)     => Tuple,
        Expr::Array(elems) => fuse_elems(&elems.iter().collect::<Vec<_>>(), params, env),
        Expr::TensorLit(rows) => {
            let flat: Vec<&Expr> = rows.iter().flatten().collect();
            fuse_elems(&flat, params, env)
        }
        Expr::Range(..) => RealTensor,
        Expr::Index(base, _) => match infer_type(base, params, env) {
            RealTensor    => Real,
            ComplexTensor => Complex,
            Tensor        => Num,
            _             => Any,
        },
        Expr::Slice(..)  => Any,
        Expr::Member(..) => Any,
        Expr::Lambda(..) => Fn,
        Expr::Block(stmts) => {
            let mut p = params.clone();
            let mut last = Any;
            for s in stmts {
                match s {
                    BlockStmt::Def(Def::Var(n, e))   => { let t = infer_type(e, &p, env); p.insert(n.clone(), t); }
                    BlockStmt::Def(Def::Func(n, ..)) => { p.insert(n.clone(), Fn); }
                    BlockStmt::Expr(e)               => last = infer_type(e, &p, env),
                }
            }
            last
        }
        Expr::Apply(f, _args) => {
            if let Expr::Var(name) = &**f {
                if params.get(name).is_none() {
                    if let Some(Val::Fn(_, _, _, _, sig)) = env.vars.get(name) {
                        return sig.ret.clone().unwrap_or(Any);
                    }
                    if let Some(r) = builtin_ret_hint(name) { return r; }
                }
            }
            Any
        }
    }
}

// ───────────────────────── builtin signatures ─────────────────────────

pub fn builtin_sig(name: &str) -> Option<&'static str> {
    match name {
        "abs"    => Some("abs(x: num) -> num"),
        "re"     => Some("re(x: num) -> real"),
        "im"     => Some("im(x: num) -> real"),
        "arg"    => Some("arg(x: num) -> real"),
        "conj"   => Some("conj(x: num) -> num"),
        "sqrt"   => Some("sqrt(x: num) -> num"),
        "exp"    => Some("exp(x: num) -> num"),
        "ln"     => Some("ln(x: num) -> num"),
        "sin"    => Some("sin(x: num) -> num"),
        "cos"    => Some("cos(x: num) -> num"),
        "tan"    => Some("tan(x: num) -> num"),
        "asin"   => Some("asin(x: num) -> num"),
        "acos"   => Some("acos(x: num) -> num"),
        "atan"   => Some("atan(x: num) -> num"),
        "sinh"   => Some("sinh(x: num) -> num"),
        "cosh"   => Some("cosh(x: num) -> num"),
        "tanh"   => Some("tanh(x: num) -> num"),
        "cbrt"   => Some("cbrt(x: real) -> real"),
        "floor"  => Some("floor(x: real) -> int"),
        "ceil"   => Some("ceil(x: real) -> int"),
        "round"  => Some("round(x: real, n: nat) -> real"),
        "trunc"  => Some("trunc(x: real) -> int"),
        "frac"   => Some("frac(x: real) -> real"),
        "log"    => Some("log(x: num, base: real) -> num"),
        "log2"   => Some("log2(x: num) -> num"),
        "log10"  => Some("log10(x: num) -> num"),
        "sign"   => Some("sign(x: real) -> real"),
        "fact"   => Some("fact(n: nat) -> real"),
        "erf"    => Some("erf(x: real) -> real"),
        "erfc"   => Some("erfc(x: real) -> real"),
        "j0"     => Some("j0(x: real) -> real"),
        "j1"     => Some("j1(x: real) -> real"),
        "len"    => Some("len(x: any) -> nat"),
        "linspace" => Some("linspace(a: real, b: real, n: nat) -> tensor"),
        "range"  => Some("range(a: real, b: real) -> tensor"),
        "sort"   => Some("sort(x: tensor) -> tensor"),
        "dot"    => Some("dot(a: tensor, b: tensor) -> real"),
        "mean"   => Some("mean(x: tensor) -> real"),
        "std"    => Some("std(x: tensor) -> real"),
        "var"    => Some("var(x: tensor) -> real"),
        "min"    => Some("min(a: num, b: num) -> num"),
        "max"    => Some("max(a: num, b: num) -> num"),
        "pow"    => Some("pow(base: num, exp: num) -> num"),
        "atan2"  => Some("atan2(y: real, x: real) -> real"),
        "hypot"  => Some("hypot(a: real, b: real) -> real"),
        "gcd"    => Some("gcd(a: int, b: int) -> nat"),
        "lcm"    => Some("lcm(a: int, b: int) -> nat"),
        "zeros"  => Some("zeros(dims: nat...) -> real tensor"),
        "ones"   => Some("ones(dims: nat...) -> real tensor"),
        "eye"    => Some("eye(n: nat) -> real tensor"),
        "diag"   => Some("diag(v: tensor) -> real tensor"),
        "shape"  => Some("shape(T: tensor) -> tuple"),
        "rows"   => Some("rows(M: tensor) -> nat"),
        "cols"   => Some("cols(M: tensor) -> nat"),
        "transpose" => Some("transpose(M: tensor) -> tensor"),
        "trace"  => Some("trace(M: tensor) -> num"),
        "norm"   => Some("norm(x: tensor) -> real"),
        "matmul" => Some("matmul(A: tensor, B: tensor) -> tensor"),
        "det"    => Some("det(M: tensor) -> num"),
        "inv"    => Some("inv(M: tensor) -> tensor"),
        "reshape" => Some("reshape(T: tensor, dims: nat...) -> tensor"),
        "map"    => Some("map(f: fn, x: any) -> any"),
        "filter" => Some("filter(f: fn, x: any) -> any"),
        "reduce" => Some("reduce(f: fn, x: any) -> any"),
        "compose" => Some("compose(f: fn, g: fn) -> fn"),
        "partial" => Some("partial(f: fn, a: any) -> fn"),
        "sum"    => Some("sum(f: fn, a: real, b: real) | sum(T: tensor) | sum(T: tensor, axis: nat)"),
        "prod"   => Some("prod(f: fn, a: real, b: real) | prod(T: tensor)"),
        "integral" => Some("integral(f: fn, a: real, b: real, n: nat) -> real"),
        "deriv"  => Some("deriv(f: fn, x: real, dx: real) -> real"),
        "iterate" => Some("iterate(f: fn, x0: any, n: nat) -> any"),
        "scan"   => Some("scan(f: fn, x0: any, n: nat) -> tensor"),
        "fft"    => Some("fft(T: tensor) -> complex tensor"),
        "ifft"   => Some("ifft(T: complex tensor) -> complex tensor"),
        "cell"   => Some("cell(init: any) -> cell"),
        "get"    => Some("get(c: cell) -> any"),
        "set"    => Some("set(c: cell, val: any) -> any"),
        "rand"   => Some("rand() -> real | rand(n: nat) -> tensor | rand(n1: nat, n2: nat, …) -> tensor"),
        "tensordot" => Some("tensordot(T1: tensor, T2: tensor, n: nat) | tensordot(T1, T2, (a, b)) | tensordot(T1, T2, ((a1,…),(b1,…)))"),
        // trig aliases / specials
        "sec"    => Some("sec(x: num) -> num"),
        "csc"    => Some("csc(x: num) -> num"),
        "cot"    => Some("cot(x: num) -> num"),
        "sech"   => Some("sech(x: num) -> num"),
        "csch"   => Some("csch(x: num) -> num"),
        "sinc"   => Some("sinc(x: real) -> real"),
        "jinc"   => Some("jinc(x: real) -> real"),
        "expm1"  => Some("expm1(x: num) -> num"),
        // scalar utilities
        "deg"     => Some("deg(x: real) -> real"),
        "rad"     => Some("rad(x: real) -> real"),
        "delta"   => Some("delta(x: real) -> int"),
        "heaviside" => Some("heaviside(x: real) -> real"),
        "signum"  => Some("signum(x: real) -> real"),
        "id"      => Some("id(x: any) -> any"),
        "not"     => Some("not(x: int) -> int"),
        "factorial" => Some("factorial(n: nat) -> real"),
        "ncr"       => Some("ncr(n: nat, r: nat) -> real"),
        "quadratic" => Some("quadratic(a: real, b: real, c: real) -> tuple"),
        "length"  => Some("length(x: any) -> nat"),
        // stats
        "median"  => Some("median(x: tensor) -> real"),
        "mode"    => Some("mode(x: tensor) -> real"),
        // probability
        "gaussian"     => Some("gaussian(x: real, mu: real, sigma: real) -> real"),
        "gaussian_cdf" => Some("gaussian_cdf(x: real, mu: real, sigma: real) -> real"),
        // levi-civita
        "eps"     => Some("eps(i: int, j: int, …) -> int"),
        // tuple / 1-D tensor ops
        "append"  => Some("append(t: any, x: any) -> any"),
        "concat"  => Some("concat(a: any, b: any) -> any"),
        "flatten" => Some("flatten(T: tensor) -> tensor"),
        "cumsum"  => Some("cumsum(t: tensor) -> tensor"),
        "cumprod" => Some("cumprod(t: tensor) -> tensor"),
        "diff"    => Some("diff(t: tensor) -> tensor"),
        "argmin"  => Some("argmin(t: tensor) -> nat"),
        "argmax"  => Some("argmax(t: tensor) -> nat"),
        "zip"     => Some("zip(a: tensor, b: tensor) -> tensor"),
        // comparison functions
        "lt"  => Some("lt(a: num, b: num) -> int"),
        "leq" => Some("leq(a: num, b: num) -> int"),
        "gt"  => Some("gt(a: num, b: num) -> int"),
        "geq" => Some("geq(a: num, b: num) -> int"),
        "eq"  => Some("eq(a: num, b: num) -> int"),
        "neq" => Some("neq(a: num, b: num) -> int"),
        // bitwise
        "and"  => Some("and(a: int, b: int) -> int"),
        "or"   => Some("or(a: int, b: int) -> int"),
        "xor"  => Some("xor(a: int, b: int) -> int"),
        "nand" => Some("nand(a: int, b: int) -> int"),
        "nor"  => Some("nor(a: int, b: int) -> int"),
        "xnor" => Some("xnor(a: int, b: int) -> int"),
        "shl"  => Some("shl(a: int, b: int) -> int"),
        "shr"  => Some("shr(a: int, b: int) -> int"),
        // control flow
        "if" => Some("if(cond: num, a: any, b: any) -> any"),
        // tensor construction
        "tensor" => Some("tensor(f: fn, n1: nat, n2: nat, …) -> tensor   |   tensor(field) -> tensor  (extract grid data)"),
        "matrix" => Some("matrix(f: fn, r: nat, c: nat) -> tensor"),
        // tensor indexing
        "row" => Some("row(M: tensor, i: nat) -> tensor"),
        "col" => Some("col(M: tensor, j: nat) -> tensor"),
        "dim" => Some("dim(T: tensor, axis: nat) -> nat"),
        // tensor reshaping
        "cat"      => Some("cat(axis: nat, T1: tensor, T2: tensor, …) -> tensor"),
        "squeeze"  => Some("squeeze(T: tensor) -> tensor"),
        "unsqueeze" => Some("unsqueeze(T: tensor, dim: nat) -> tensor"),
        "permute"  => Some("permute(T: tensor, p0: nat, p1: nat, …) -> tensor"),
        // linear algebra
        "solve"      => Some("solve(A: tensor, b: tensor) -> tensor"),
        "eig"        => Some("eig(M: tensor) -> tuple"),
        "eigvals"    => Some("eigvals(M: tensor) -> tensor"),
        "eig_top"    => Some("eig_top(M: tensor) -> tuple"),
        "eig_bot"    => Some("eig_bot(M: tensor) -> tuple"),
        "qr"         => Some("qr(M: tensor) -> tuple"),
        "diagonalize" => Some("diagonalize(M: tensor) -> tuple"),
        "hstack"     => Some("hstack(A: tensor, B: tensor) -> tensor"),
        "vstack"     => Some("vstack(A: tensor, B: tensor) -> tensor"),
        "tomat"      => Some("tomat(t: tensor, r: nat, c: nat) -> tensor"),
        "outer"      => Some("outer(A: tensor, B: tensor) -> tensor"),
        // elementwise
        "lerp"  => Some("lerp(a: any, b: any, t: any) -> any"),
        "clamp" => Some("clamp(x: any, lo: real, hi: real) -> any"),
        "shift" => Some("shift(T: tensor, n: int, axis: nat) -> tensor"),
        "roll"  => Some("roll(T: tensor, n: int, axis: nat) -> tensor"),
        // n-D grid
        "lingrid" => Some("lingrid(start: any, end: any, counts: any, f: fn) -> tensor"),
        // ops namespace
        "grad"    => Some("ops.grad(T, dx [, axis: nat]) -> tensor | field"),
        "div"     => Some("ops.div(V, dx) -> tensor | field"),
        "curl"    => Some("ops.curl(V, dx) -> tensor | field"),
        "lap"     => Some("ops.lap(T, dx) -> tensor | field"),
        "poisson" => Some("ops.poisson(rhs, dx) -> tensor | field"),
        "invlap"  => Some("ops.invlap(T, dx) -> tensor | field"),
        "specgrad" => Some("ops.specgrad(T, dx [, axis: nat]) -> tensor"),
        // solver namespace
        "rk4"    => Some("solver.rk4(f: fn, y0: any, t0: real, t1: real, n: nat) -> any"),
        "odeint" => Some("solver.odeint(f: fn, y0: any, ts: tensor) -> tensor"),
        "verlet" => Some("solver.verlet(dVdq: fn, dTdp: fn, q0: any, p0: any, dt: real, n: nat) -> (q, p)  [symplectic, H=T(p)+V(q); q0/p0 may be scalar/tensor/field/tuple]"),
        "tao" => Some("solver.tao(dHdq: fn, dHdp: fn, q0: any, p0: any, dt: real, n: nat [, omega: real]) -> (q, p)  [explicit symplectic for NON-separable canonical H(q,p)]"),
        "cfl"    => Some("solver.cfl(V: tensor, dx: real, dt: real) -> real"),
        // forms namespace
        "d"        => Some("forms.d(f: field) -> field  (exterior derivative)"),
        "hodge"    => Some("forms.hodge(w: field) -> field  (Hodge star ★)"),
        "wedge"    => Some("forms.wedge(a: field, b: field) -> field  (wedge product ∧)"),
        "raise"    => Some("forms.raise(w: field) -> field  (♯: lower index with metric)"),
        "lower"    => Some("forms.lower(X: field) -> field  (♭: raise index with metric)"),
        "codiff"   => Some("forms.codiff(w: field) -> field  (codifferential δ = ±★d★)"),
        "laplace"  => Some("forms.laplace(w: field) -> field  (Laplace–de Rham Δ = dδ+δd)"),
        "contract" => Some("forms.contract(X: field, w: field) -> field  (interior product ι_X)"),
        "form"     => Some("forms.form(data: tensor, degree: nat, lo, hi, bc [, metric]) -> field"),
        "vector"   => Some("forms.vector(data: tensor, lo, hi, bc [, metric]) -> field"),
        "field"    => Some("field(data, lo, hi, bc [, metric]) -> field   |   field(f: fn, lo, hi, counts, bc [, metric]) -> field  (sample f at grid coords)"),
        // pic namespace
        "scatter"  => Some("pic.scatter(positions, weights, template: field [, kernel]) -> field  (deposit particles → grid: ρ, J)"),
        "gather"   => Some("pic.gather(field, positions [, kernel]) -> tensor  (interpolate grid → particles; transpose of scatter)"),
        _ => None,
    }
}

pub fn is_protected(name: &str) -> bool {
    matches!(name,
        "pi" | "e" | "phi" | "inf" | "i"
        | "abs" | "re" | "im" | "arg" | "conj" | "sqrt" | "exp" | "ln"
        | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2"
        | "sinh" | "cosh" | "tanh" | "cbrt" | "expm1"
        | "sec" | "csc" | "cot"
        | "floor" | "ceil" | "round" | "trunc" | "frac"
        | "log" | "log10" | "log2"
        | "sign" | "signum" | "id" | "fact" | "factorial" | "ncr" | "quadratic"
        | "heaviside"
        | "deg" | "rad"
        | "len" | "length"
        | "linspace" | "range"
        | "sort" | "zip" | "dot" | "append" | "concat" | "flatten" | "argmin" | "argmax"
        | "cumsum" | "cumprod" | "diff"
        | "mean" | "std"
        | "compose" | "partial"
        | "filter" | "reduce"
        | "rand" | "eps"
        | "min" | "max" | "pow" | "hypot" | "gcd" | "lcm"
        | "lt" | "leq" | "gt" | "geq" | "eq" | "neq"
        | "if"
        | "fft" | "ifft"
        | "sum" | "prod" | "integral" | "deriv" | "map"
        | "iterate" | "scan"
        | "cell" | "get" | "set"
        | "field"
        | "tensor" | "matrix" | "zeros" | "ones" | "eye" | "diag"
        | "shape" | "rows" | "cols" | "transpose" | "trace" | "norm"
        | "row" | "col" | "matmul"
        | "det" | "inv" | "solve"
        | "eig" | "eigvals"
        | "hstack" | "vstack" | "tomat"
        | "shift" | "roll"
        | "lingrid"
        | "reshape" | "permute" | "cat" | "squeeze" | "unsqueeze"
        | "dim"
        // Standard namespace names (ops, special, bits, …) are reserved too.
        | "ops" | "solver" | "forms" | "pic" | "special" | "bits" | "stats" | "linalg" | "vec"
    )
}

// ── Output formatting ─────────────────────────────────────────────────────────

pub fn fmt_f(n: f64) -> String {
    if n.is_nan() { return "NaN".into(); }
    if n.is_infinite() { return if n > 0.0 { "inf".into() } else { "-inf".into() }; }
    if n.fract() == 0.0 && n.abs() < 1e15 { return format!("{}", n as i64); }
    format!("{n}")
}

/// Format one complex element (real if im==0).
fn fmt_complex_elem(r: f64, i: f64) -> String {
    if i == 0.0 { return fmt_f(r); }
    let babs = i.abs();
    let im = if babs == 1.0 { String::new() } else { fmt_f(babs) };
    if r == 0.0 {
        if i < 0.0 { format!("-{im}i") } else { format!("{im}i") }
    } else if i < 0.0 {
        format!("{} - {im}i", fmt_f(r))
    } else {
        format!("{} + {im}i", fmt_f(r))
    }
}

/// Box-character display for a 2D slice of data with given rows × cols.
fn fmt_mat(data: &[f64], r: usize, c: usize) -> String {
    let cells: Vec<Vec<String>> = (0..r).map(|i| {
        (0..c).map(|j| fmt_f(data[i * c + j])).collect()
    }).collect();
    let col_widths: Vec<usize> = (0..c).map(|j| {
        cells.iter().map(|row| row[j].len()).max().unwrap_or(0)
    }).collect();
    cells.into_iter().enumerate().map(|(ri, row)| {
        let padded: Vec<String> = row.into_iter().zip(&col_widths)
            .map(|(s, &w)| format!("{:>w$}", s))
            .collect();
        let content = padded.join("  ");
        if r == 1 || ri == 0   { format!("\u{23A1} {} \u{23A4}", content) }  // ⎡ ⎤
        else if ri == r - 1    { format!("\u{23A3} {} \u{23A6}", content) }  // ⎣ ⎦
        else                   { format!("\u{23A2} {} \u{23A5}", content) }  // ⎢ ⎥
    }).collect::<Vec<_>>().join("\n")
}

pub fn fmt_val(v: &Val) -> String {
    match v {
        Val::Num(n) => fmt_f(*n),
        Val::Complex(a, b) => {
            let re = fmt_f(*a);
            let babs = b.abs();
            let im = if babs == 1.0 { String::new() } else { fmt_f(babs) };
            if *a == 0.0 {
                if *b < 0.0 { format!("-{im}i") } else { format!("{im}i") }
            } else if *b < 0.0 {
                format!("{re} - {im}i")
            } else {
                format!("{re} + {im}i")
            }
        }
        Val::Fn(params, _, _, _, sig) => {
            let param_strs: Vec<String> = params.iter().enumerate().map(|(i, name)| {
                if let Some(Some(h)) = sig.params.get(i) {
                    format!("{}: {}", name, h.display())
                } else {
                    name.clone()
                }
            }).collect();
            let ret_str = if let Some(h) = &sig.ret { format!(" -> {}", h.display()) } else { String::new() };
            format!("<fn({}){}= …>", param_strs.join(", "), if ret_str.is_empty() { " ".into() } else { format!("{} ", ret_str) })
        }
        Val::Builtin(name) => format!("<builtin {name}>"),
        Val::Cell(c) => format!("cell({})", fmt_val(&c.borrow())),
        Val::Namespace(map) => {
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            format!("namespace{{{}}}", names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
        }
        Val::Field(f) => {
            let kind = match f.variance { Variance::Form => "form", Variance::Vector => "vector field" };
            let dims = f.grid.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
            let extent = (0..f.grid.len())
                .map(|a| {
                    let (lo, dx, n) = (f.lo[a], f.spacing[a], f.grid[a]);
                    // periodic excludes the duplicate endpoint (hi = lo + dx·N);
                    // neumann includes both endpoints (hi = lo + dx·(N−1)).
                    let cells = match f.bc[a] { BC::Periodic => n, BC::Neumann => n.saturating_sub(1) };
                    format!("[{}, {}]", fmt_f(lo), fmt_f(lo + dx * cells as f64))
                })
                .collect::<Vec<_>>().join("×");
            let bc = if f.bc.iter().all(|&b| b == BC::Periodic) { "periodic" }
                     else if f.bc.iter().all(|&b| b == BC::Neumann) { "neumann" }
                     else { "mixed-bc" };
            let metric = if f.metric.iter().all(|&g| g == 1.0) { String::new() }
                else { format!(" metric({})", f.metric.iter().map(|&g| fmt_f(g)).collect::<Vec<_>>().join(", ")) };
            format!("{}-{} [{}] on {} {}{}\n{}",
                f.degree, kind, dims, extent, bc, metric,
                fmt_val(&field_data_as_tensor(f)))
        }
        Val::Tuple(items) => {
            let parts: Vec<String> = items.iter().map(|item| {
                let s = fmt_val(item);
                match item {
                    Val::Tensor { shape, .. } if shape.len() >= 2 => format!("\n{s}"),
                    Val::ComplexTensor { shape, .. } if shape.len() >= 2 => format!("\n{s}"),
                    _ => s,
                }
            }).collect();
            format!("({})", parts.join(", "))
        }
        Val::Tensor { data, shape } => {
            if shape.is_empty() || data.is_empty() { return "[]".into(); }
            if shape.len() == 1 {
                let items: Vec<String> = data.iter().map(|x| fmt_f(*x)).collect();
                return format!("[{}]", items.join(", "));
            }
            if shape.len() == 2 {
                return fmt_mat(data, shape[0], shape[1]);
            }
            if shape.len() == 3 {
                let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
                let slice_size = d1 * d2;
                return (0..d0).map(|k| {
                    let slice = &data[k*slice_size..(k+1)*slice_size];
                    format!("[{k}]\n{}", fmt_mat(slice, d1, d2))
                }).collect::<Vec<_>>().join("\n");
            }
            format!("<tensor shape={:?}>", shape)
        }
        Val::ComplexTensor { re, im, shape } => {
            if shape.is_empty() || re.is_empty() { return "[]".into(); }
            // Helper: matrix-style display using complex element strings.
            let fmt_cmat = |re: &[f64], im: &[f64], r: usize, c: usize| -> String {
                let cells: Vec<Vec<String>> = (0..r).map(|i| {
                    (0..c).map(|j| fmt_complex_elem(re[i*c+j], im[i*c+j])).collect()
                }).collect();
                let col_widths: Vec<usize> = (0..c).map(|j| {
                    cells.iter().map(|row| row[j].len()).max().unwrap_or(0)
                }).collect();
                cells.into_iter().enumerate().map(|(ri, row)| {
                    let padded: Vec<String> = row.into_iter().zip(&col_widths)
                        .map(|(s, &w)| format!("{:>w$}", s)).collect();
                    let content = padded.join("  ");
                    if r == 1 || ri == 0 { format!("\u{23A1} {} \u{23A4}", content) }
                    else if ri == r - 1  { format!("\u{23A3} {} \u{23A6}", content) }
                    else                 { format!("\u{23A2} {} \u{23A5}", content) }
                }).collect::<Vec<_>>().join("\n")
            };
            if shape.len() == 1 {
                let items: Vec<String> = re.iter().zip(im.iter()).map(|(&r, &i)| fmt_complex_elem(r, i)).collect();
                return format!("[{}]", items.join(", "));
            }
            if shape.len() == 2 {
                return fmt_cmat(re, im, shape[0], shape[1]);
            }
            if shape.len() == 3 {
                let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
                let ss = d1 * d2;
                return (0..d0).map(|k| {
                    format!("[{k}]\n{}", fmt_cmat(&re[k*ss..(k+1)*ss], &im[k*ss..(k+1)*ss], d1, d2))
                }).collect::<Vec<_>>().join("\n");
            }
            format!("<complex tensor shape={:?}>", shape)
        }
    }
}

// ── Evaluator helpers ─────────────────────────────────────────────────────────

#[inline] fn int(x: f64) -> i64 { x as i64 }

// Collapse a+bi to Num(a) when b is negligibly small relative to the magnitude.
pub fn make_complex(a: f64, b: f64) -> Val {
    let scale = (a.abs() + b.abs()).max(1.0) * 1e-10;
    let a = if a.abs() < scale { 0.0 } else { a };
    let b = if b.abs() < scale { 0.0 } else { b };
    if b == 0.0 { Val::Num(a) } else { Val::Complex(a, b) }
}

fn to_complex(v: Val) -> Result<(f64, f64), String> {
    match v {
        Val::Num(n)               => Ok((n, 0.0)),
        Val::Complex(a, b)        => Ok((a, b)),
        Val::Fn(..)               => Err("expected a number, got a function".into()),
        Val::Builtin(n)           => Err(format!("expected a number, got builtin '{n}'")),
        Val::Tuple(..)            => Err("expected a number, got a tuple".into()),
        Val::Tensor { .. }        => Err("expected a number, got a tensor".into()),
        Val::ComplexTensor { .. } => Err("expected a number, got a complex tensor".into()),
        Val::Cell(..)             => Err("expected a number, got a cell (use get())".into()),
        Val::Namespace(..)        => Err("expected a number, got a namespace".into()),
        Val::Field(..)            => Err("expected a number, got a field".into()),
    }
}

/// Return Tensor if all imaginary parts are negligibly zero, else ComplexTensor.
#[inline]
pub(crate) fn maybe_real(re: Vec<f64>, im: Vec<f64>, shape: Vec<usize>) -> Val {
    if im.iter().all(|&x| x == 0.0) {
        Val::Tensor { data: TData::new(re), shape }
    } else {
        Val::ComplexTensor { re: TData::new(re), im: TData::new(im), shape }
    }
}

/// Element-wise complex binop on two parallel (re,im) arrays.
fn complex_tensors_binop(
    re1: Vec<f64>, im1: Vec<f64>, shape: Vec<usize>,
    op: &Op, re2: &[f64], im2: &[f64],
) -> Result<Val, String> {
    let n = re1.len();
    let mut re_out = Vec::with_capacity(n);
    let mut im_out = Vec::with_capacity(n);
    for i in 0..n {
        let v1 = if im1[i] == 0.0 { Val::Num(re1[i]) } else { Val::Complex(re1[i], im1[i]) };
        let v2 = if im2[i] == 0.0 { Val::Num(re2[i]) } else { Val::Complex(re2[i], im2[i]) };
        match scalar_binop(v1, op, v2)? {
            Val::Num(r)        => { re_out.push(r); im_out.push(0.0); }
            Val::Complex(r, i) => { re_out.push(r); im_out.push(i); }
            other => return Err(format!("binop: unexpected {}", fmt_val(&other))),
        }
    }
    Ok(maybe_real(re_out, im_out, shape))
}

// z^w via exp(w·ln z)
fn complex_pow(la: f64, lb: f64, ra: f64, rb: f64) -> Val {
    if la == 0.0 && lb == 0.0 {
        return if ra == 0.0 && rb == 0.0 { Val::Num(1.0) } else { Val::Num(0.0) };
    }
    let r     = (la*la + lb*lb).sqrt();
    let theta = lb.atan2(la);
    let new_re = ra * r.ln() - rb * theta;
    let new_im = ra * theta  + rb * r.ln();
    let mag = new_re.exp();
    make_complex(mag * new_im.cos(), mag * new_im.sin())
}

fn arity(name: &str, expected: usize, got: usize) -> Result<(), String> {
    if expected == got { Ok(()) }
    else { Err(format!("{name} expects {expected} arg(s), got {got}")) }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a / gcd(a, b) * b }
}

// ── Broadcasting ──────────────────────────────────────────────────────────────

fn broadcast1(v: Val, f: impl Fn(Val) -> Result<Val, String>) -> Result<Val, String> {
    match v {
        Val::Tuple(items) => Ok(Val::Tuple(
            items.into_iter().map(f).collect::<Result<_, _>>()?
        )),
        Val::Tensor { data, shape } => {
            let new_data: Result<Vec<f64>, _> = data.into_iter()
                .map(|x| f(Val::Num(x))?.num("broadcast"))
                .collect();
            Ok(Val::Tensor { data: TData::new(new_data?), shape })
        }
        Val::ComplexTensor { re, im, shape } => {
            let mut re_out = Vec::with_capacity(re.len());
            let mut im_out = Vec::with_capacity(re.len());
            for (r, i) in re.into_iter().zip(im.into_iter()) {
                let v = if i == 0.0 { Val::Num(r) } else { Val::Complex(r, i) };
                match f(v)? {
                    Val::Num(n)        => { re_out.push(n); im_out.push(0.0); }
                    Val::Complex(a, b) => { re_out.push(a); im_out.push(b); }
                    other => return Err(format!("broadcast: expected a number, got {}", fmt_val(&other))),
                }
            }
            Ok(maybe_real(re_out, im_out, shape))
        }
        other => f(other),
    }
}

// ── Tensor helpers ────────────────────────────────────────────────────────────

/// Promote a Tensor or ComplexTensor into (re, im, shape) triple.
pub(crate) fn as_complex_tensor(v: Val) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>), String> {
    match v {
        Val::Tensor { data, shape } => { let n = data.len(); Ok((data.into_vec(), vec![0.0f64; n], shape)) }
        Val::ComplexTensor { re, im, shape } => Ok((re.into_vec(), im.into_vec(), shape)),
        _ => Err("expected a tensor".into()),
    }
}

fn binop_tensor(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    macro_rules! shape_check {
        ($ls:expr, $rs:expr) => {
            if $ls != $rs {
                return Err(format!("tensor op tensor: shape mismatch ({:?} vs {:?})", $ls, $rs));
            }
        };
    }
    match (lv, rv) {
        // ── Real × Real ────────────────────────────────────────────────────
        (Val::Tensor { data: ld, shape: ls }, Val::Tensor { data: rd, shape: rs }) => {
            shape_check!(ls, rs);
            let out: Result<Vec<f64>, _> = ld.into_iter().zip(rd.into_iter())
                .map(|(l, r)| scalar_binop(Val::Num(l), op, Val::Num(r))?.num("tensor op"))
                .collect();
            Ok(Val::Tensor { data: TData::new(out?), shape: ls })
        }
        // ── Real tensor × real scalar ──────────────────────────────────────
        (Val::Tensor { data, shape }, Val::Num(s)) => {
            let out: Result<Vec<f64>, _> = data.into_iter()
                .map(|x| scalar_binop(Val::Num(x), op, Val::Num(s))?.num("tensor op"))
                .collect();
            Ok(Val::Tensor { data: TData::new(out?), shape })
        }
        (Val::Num(s), Val::Tensor { data, shape }) => {
            let out: Result<Vec<f64>, _> = data.into_iter()
                .map(|x| scalar_binop(Val::Num(s), op, Val::Num(x))?.num("tensor op"))
                .collect();
            Ok(Val::Tensor { data: TData::new(out?), shape })
        }
        // ── ComplexTensor × ComplexTensor ──────────────────────────────────
        (Val::ComplexTensor { re: lr, im: li, shape: ls }, Val::ComplexTensor { re: rr, im: ri, shape: rs }) => {
            shape_check!(ls, rs);
            complex_tensors_binop(lr.into_vec(), li.into_vec(), ls, op, &rr, &ri)
        }
        // ── ComplexTensor × Tensor ─────────────────────────────────────────
        (Val::ComplexTensor { re: lr, im: li, shape: ls }, Val::Tensor { data: rd, shape: rs }) => {
            shape_check!(ls, rs);
            let ri = vec![0.0f64; rd.len()];
            complex_tensors_binop(lr.into_vec(), li.into_vec(), ls, op, &rd, &ri)
        }
        (Val::Tensor { data: ld, shape: ls }, Val::ComplexTensor { re: rr, im: ri, shape: rs }) => {
            shape_check!(ls, rs);
            let li = vec![0.0f64; ld.len()];
            complex_tensors_binop(ld.into_vec(), li, ls, op, &rr, &ri)
        }
        // ── ComplexTensor × scalar ─────────────────────────────────────────
        (Val::ComplexTensor { re, im, shape }, Val::Num(s)) => {
            let n = re.len();
            complex_tensors_binop(re.into_vec(), im.into_vec(), shape, op, &vec![s; n], &vec![0.0; n])
        }
        (Val::Num(s), Val::ComplexTensor { re, im, shape }) => {
            let n = re.len();
            complex_tensors_binop(vec![s; n], vec![0.0; n], shape, op, &re, &im)
        }
        (Val::ComplexTensor { re, im, shape }, Val::Complex(sr, si)) => {
            let n = re.len();
            complex_tensors_binop(re.into_vec(), im.into_vec(), shape, op, &vec![sr; n], &vec![si; n])
        }
        (Val::Complex(sr, si), Val::ComplexTensor { re, im, shape }) => {
            let n = re.len();
            complex_tensors_binop(vec![sr; n], vec![si; n], shape, op, &re, &im)
        }
        // ── Tensor × complex scalar ────────────────────────────────────────
        (Val::Tensor { data, shape }, Val::Complex(sr, si)) => {
            let n = data.len();
            complex_tensors_binop(data.into_vec(), vec![0.0; n], shape, op, &vec![sr; n], &vec![si; n])
        }
        (Val::Complex(sr, si), Val::Tensor { data, shape }) => {
            let n = data.len();
            complex_tensors_binop(vec![sr; n], vec![si; n], shape, op, &data, &vec![0.0; n])
        }
        _ => unreachable!(),
    }
}

// ── Linear algebra helpers ────────────────────────────────────────────────────

/// Gaussian elimination with partial pivoting.
/// Returns (upper-triangular U in-place, sign of permutation).
fn lu_upper(data: &[f64], n: usize) -> (Vec<f64>, i32) {
    let mut a = data.to_vec();
    let mut sign = 1i32;
    for k in 0..n {
        // Find pivot row
        let max_row = (k..n)
            .max_by(|&i, &j| a[i*n+k].abs().partial_cmp(&a[j*n+k].abs())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if max_row != k {
            for j in 0..n { a.swap(k*n+j, max_row*n+j); }
            sign = -sign;
        }
        let pivot = a[k*n+k];
        if pivot.abs() < 1e-14 { continue; }
        for i in (k+1)..n {
            let factor = a[i*n+k] / pivot;
            for j in k..n {
                let p = a[k*n+j];
                a[i*n+j] -= factor * p;
            }
        }
    }
    (a, sign)
}

fn det_nxn(data: &[f64], n: usize) -> f64 {
    let (u, sign) = lu_upper(data, n);
    let d: f64 = (0..n).map(|i| u[i*n+i]).product();
    d * sign as f64
}

fn inv_nxn(data: &[f64], n: usize) -> Result<Vec<f64>, String> {
    // Gauss-Jordan on augmented [A | I]
    let w = 2 * n;
    let mut aug = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n { aug[i*w+j] = data[i*n+j]; }
        aug[i*w+n+i] = 1.0;
    }
    for k in 0..n {
        let max_row = (k..n)
            .max_by(|&i, &j| aug[i*w+k].abs().partial_cmp(&aug[j*w+k].abs())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if max_row != k { for j in 0..w { aug.swap(k*w+j, max_row*w+j); } }
        let pivot = aug[k*w+k];
        if pivot.abs() < 1e-14 { return Err("inv: matrix is singular".into()); }
        for j in 0..w { aug[k*w+j] /= pivot; }
        for i in 0..n {
            if i == k { continue; }
            let factor = aug[i*w+k];
            for j in 0..w { let p = aug[k*w+j]; aug[i*w+j] -= factor * p; }
        }
    }
    let mut out = vec![0.0f64; n*n];
    for i in 0..n { for j in 0..n { out[i*n+j] = aug[i*w+n+j]; } }
    Ok(out)
}

fn solve_nxn(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let w = n + 1;
    let mut aug = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n { aug[i*w+j] = a[i*n+j]; }
        aug[i*w+n] = b[i];
    }
    for k in 0..n {
        let max_row = (k..n)
            .max_by(|&i, &j| aug[i*w+k].abs().partial_cmp(&aug[j*w+k].abs())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if max_row != k { for j in 0..w { aug.swap(k*w+j, max_row*w+j); } }
        let pivot = aug[k*w+k];
        if pivot.abs() < 1e-14 { return Err("solve: matrix is singular".into()); }
        for j in k..w { aug[k*w+j] /= pivot; }
        for i in 0..n {
            if i == k { continue; }
            let factor = aug[i*w+k];
            for j in k..w { let p = aug[k*w+j]; aug[i*w+j] -= factor * p; }
        }
    }
    Ok((0..n).map(|i| aug[i*w+n]).collect())
}

fn eye_n(n: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; n * n];
    for i in 0..n { e[i*n+i] = 1.0; }
    e
}

fn matmul_nn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i*n+k];
            for j in 0..n { c[i*n+j] += aik * b[k*n+j]; }
        }
    }
    c
}

/// Full QR via Householder reflections. Returns (Q: m×m, R: m×n).
fn qr_householder(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = a.to_vec();   // m×n row-major
    let mut q = eye_n(m);     // m×m
    for k in 0..n.min(m.saturating_sub(1)) {
        let len = m - k;
        let x: Vec<f64> = (k..m).map(|i| r[i*n+k]).collect();
        let norm_x: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm_x < 1e-14 { continue; }
        let mut hv = x;
        let sign = if hv[0] >= 0.0 { 1.0 } else { -1.0 };
        hv[0] += sign * norm_x;
        let norm_hv: f64 = hv.iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm_hv < 1e-14 { continue; }
        for v in &mut hv { *v /= norm_hv; }
        for j in k..n {
            let dot: f64 = (0..len).map(|i| hv[i] * r[(i+k)*n+j]).sum();
            for i in 0..len { r[(i+k)*n+j] -= 2.0 * hv[i] * dot; }
        }
        for i in 0..m {
            let dot: f64 = (0..len).map(|j| q[i*m+(j+k)] * hv[j]).sum();
            for j in 0..len { q[i*m+(j+k)] -= 2.0 * hv[j] * dot; }
        }
    }
    (q, r)
}

fn eig_qr_impl(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ak = a.to_vec();
    let mut eigvecs = eye_n(n);
    for _ in 0..2000 {
        let (q, r) = qr_householder(&ak, n, n);
        ak = matmul_nn(&r, &q, n);
        eigvecs = matmul_nn(&eigvecs, &q, n);
        let off: f64 = (0..n).flat_map(|i| (0..i).map(move |j| (i, j)))
            .map(|(i, j)| ak[i*n+j] * ak[i*n+j])
            .sum::<f64>()
            .sqrt();
        if off < 1e-12 { break; }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| ak[i*n+i]).collect();
    (eigenvalues, eigvecs)
}

fn power_iter(a: &[f64], n: usize) -> (f64, Vec<f64>) {
    let init = (n as f64).sqrt().recip();
    let mut b: Vec<f64> = vec![init; n];
    for _ in 0..1000 {
        let mut b_new = vec![0.0f64; n];
        for i in 0..n { for j in 0..n { b_new[i] += a[i*n+j] * b[j]; } }
        let norm: f64 = b_new.iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm < 1e-14 { break; }
        let b_norm: Vec<f64> = b_new.iter().map(|v| v / norm).collect();
        let diff: f64 = b.iter().zip(b_norm.iter()).map(|(x, y)| (x-y).abs()).sum();
        let diff2: f64 = b.iter().zip(b_norm.iter()).map(|(x, y)| (x+y).abs()).sum();
        b = b_norm;
        if diff < 1e-10 || diff2 < 1e-10 { break; }
    }
    let mut ab = vec![0.0f64; n];
    for i in 0..n { for j in 0..n { ab[i] += a[i*n+j] * b[j]; } }
    let lam: f64 = b.iter().zip(ab.iter()).map(|(x, y)| x * y).sum();
    (lam, b)
}

fn inv_power_iter(a: &[f64], n: usize) -> Result<(f64, Vec<f64>), String> {
    let init = (n as f64).sqrt().recip();
    let mut b: Vec<f64> = vec![init; n];
    for _ in 0..1000 {
        let b_new = solve_nxn(a, &b, n)?;
        let norm: f64 = b_new.iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm < 1e-14 { break; }
        let b_norm: Vec<f64> = b_new.iter().map(|v| v / norm).collect();
        let diff: f64 = b.iter().zip(b_norm.iter()).map(|(x, y)| (x-y).abs()).sum();
        let diff2: f64 = b.iter().zip(b_norm.iter()).map(|(x, y)| (x+y).abs()).sum();
        b = b_norm;
        if diff < 1e-10 || diff2 < 1e-10 { break; }
    }
    let mut ab = vec![0.0f64; n];
    for i in 0..n { for j in 0..n { ab[i] += a[i*n+j] * b[j]; } }
    let lam: f64 = b.iter().zip(ab.iter()).map(|(x, y)| x * y).sum();
    Ok((lam, b))
}

// ── Tensor axis utilities ─────────────────────────────────────────────────────

/// Row-major strides for a given shape.
pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut s = vec![1usize; n];
    for k in (0..n.saturating_sub(1)).rev() { s[k] = s[k + 1] * shape[k + 1]; }
    s
}

/// Decompose a flat index into a multi-index for the given shape (row-major).
pub(crate) fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut idx = vec![0usize; n];
    for k in (0..n).rev() {
        idx[k] = flat % shape[k];
        flat /= shape[k];
    }
    idx
}

/// Apply an axis permutation to a tensor.
/// `perm[k]` = which input axis feeds output axis k.
fn apply_permutation(data: Vec<f64>, shape: &[usize], perm: &[usize]) -> Result<Val, String> {
    let ndim = shape.len();
    let new_shape: Vec<usize> = perm.iter().map(|&k| shape[k]).collect();
    let old_strides = strides(shape);
    let n = data.len();
    let mut out = vec![0.0f64; n];
    for out_flat in 0..n {
        // Decompose out_flat in new_shape space
        let out_multi = unravel(out_flat, &new_shape);
        // Map to input multi-index: in_multi[perm[k]] = out_multi[k]
        let mut in_multi = vec![0usize; ndim];
        for k in 0..ndim { in_multi[perm[k]] = out_multi[k]; }
        let in_flat: usize = in_multi.iter().zip(&old_strides).map(|(&i, &s)| i * s).sum();
        out[out_flat] = data[in_flat];
    }
    Ok(Val::Tensor { data: TData::new(out), shape: new_shape })
}

/// Apply a 1-D FFT/IFFT in-place along one axis of a complex tensor stored as
/// two real arrays (re, im) with the given row-major shape.
pub(crate) fn fft_axis_inplace(re: &mut [f64], im: &mut [f64], shape: &[usize], axis: usize, forward: bool) {
    use rustfft::num_complex::Complex64;
    let n = shape[axis];
    let s = strides(shape);
    let axis_stride = s[axis];

    // Strides for all dims except `axis` — used to enumerate orthogonal slices.
    let other_shape: Vec<usize> = shape.iter().enumerate()
        .filter(|&(k, _)| k != axis).map(|(_, &d)| d).collect();
    let other_strides: Vec<usize> = s.iter().enumerate()
        .filter(|&(k, _)| k != axis).map(|(_, &st)| st).collect();
    let other_total: usize = if other_shape.is_empty() { 1 } else { other_shape.iter().product() };

    let mut planner = rustfft::FftPlanner::new();
    let fft = if forward { planner.plan_fft_forward(n) } else { planner.plan_fft_inverse(n) };
    let mut buf = vec![Complex64::new(0.0, 0.0); n];

    for other_flat in 0..other_total {
        let other_multi = unravel(other_flat, &other_shape);
        let base: usize = other_multi.iter().zip(&other_strides).map(|(&i, &st)| i * st).sum();

        for i in 0..n {
            let flat = base + i * axis_stride;
            buf[i] = Complex64::new(re[flat], im[flat]);
        }
        fft.process(&mut buf);
        if !forward {
            let scale = 1.0 / n as f64;
            for c in &mut buf { *c *= scale; }
        }
        for i in 0..n {
            let flat = base + i * axis_stride;
            re[flat] = buf[i].re;
            im[flat] = buf[i].im;
        }
    }
}

// ── Builtin dispatch ──────────────────────────────────────────────────────────

pub fn eval_builtin(name: &str, mut vals: Vec<Val>, env: &Env) -> Result<Val, String> {
    macro_rules! b1 {
        ($closure:expr) => {{
            arity(name, 1, vals.len())?;
            broadcast1(vals.into_iter().next().unwrap(), $closure)
        }};
    }
    macro_rules! f1 {
        ($method:ident) => {{
            arity(name, 1, vals.len())?;
            broadcast1(vals.into_iter().next().unwrap(), |v| Ok(Val::Num(v.num(name)?.$method())))
        }};
    }

    // New namespaced PDE functions (ops.*, solver.*) carry their own
    // implementations in src/ns/. Route them out before the flat match (guarded
    // so the flat match still owns `vals` for every other name).
    if crate::ns::is_ns_builtin(name) {
        return crate::ns::dispatch(name, vals, env).unwrap();
    }
    if name == "field" {
        return crate::ns::forms::field_ctor(vals, env);
    }

    // A field decays to its component tensor under most flat builtins reached here
    // (abs/max/sum/sin/…). Arithmetic operators preserve field-ness via
    // `field_binop`, and the form/operator namespaces (forms.*, ops.*) are routed
    // out above with their fields intact. Container/identity builtins are
    // field-transparent — they must store and return the field untouched, not its
    // raw data (otherwise `cell(field)`/`get`/`set` would silently lose the type).
    if !matches!(name, "cell" | "get" | "set" | "id") {
        for i in 0..vals.len() {
            if let Val::Field(f) = &vals[i] {
                let t = field_data_as_tensor(f);
                vals[i] = t;
            }
        }
    }

    match name {
        // ── Complex-capable 1-arg ─────────────────────────────────────────────
        "abs"  => b1!(|v| match v {
            Val::Num(n)        => Ok(Val::Num(n.abs())),
            Val::Complex(a, b) => Ok(Val::Num((a*a + b*b).sqrt())),
            _ => Err(format!("{name}: expected a number")),
        }),
        "re"   => b1!(|v| {
            let (a, _) = to_complex(v)?; Ok(Val::Num(a))
        }),
        "im"   => b1!(|v| {
            let (_, b) = to_complex(v)?; Ok(Val::Num(b))
        }),
        "arg"  => b1!(|v| {
            let (a, b) = to_complex(v)?;
            Ok(Val::Num(if b == 0.0 { if a >= 0.0 { 0.0 } else { std::f64::consts::PI } } else { b.atan2(a) }))
        }),
        "conj" => b1!(|v| {
            let (a, b) = to_complex(v)?; Ok(make_complex(a, -b))
        }),
        "sqrt" => b1!(|v| match v {
            Val::Num(n) if n >= 0.0 => Ok(Val::Num(n.sqrt())),
            Val::Num(n)             => Ok(Val::Complex(0.0, (-n).sqrt())),
            Val::Complex(a, b) => {
                let r = (a*a + b*b).sqrt().sqrt();
                let theta = b.atan2(a) / 2.0;
                Ok(make_complex(r * theta.cos(), r * theta.sin()))
            }
            _ => Err(format!("{name}: expected a number")),
        }),
        "exp"  => b1!(|v| {
            let (a, b) = to_complex(v)?;
            let m = a.exp();
            Ok(make_complex(m * b.cos(), m * b.sin()))
        }),
        "ln"   => b1!(|v| match v {
            Val::Num(n) if n >= 0.0 => Ok(Val::Num(n.ln())),
            Val::Num(n)             => Ok(make_complex((-n).ln(), std::f64::consts::PI)),
            Val::Complex(a, b)      => Ok(make_complex((a*a+b*b).sqrt().ln(), b.atan2(a))),
            _ => Err(format!("{name}: expected a number")),
        }),

        // ── Real 1-arg ────────────────────────────────────────────────────────
        "sin" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            Ok(make_complex(x.sin() * y.cosh(), x.cos() * y.sinh()))
        }),
        "cos" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            Ok(make_complex(x.cos() * y.cosh(), -x.sin() * y.sinh()))
        }),
        "tan" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            let (sr, si) = (x.sin() * y.cosh(), x.cos() * y.sinh());
            let (cr, ci) = (x.cos() * y.cosh(), -x.sin() * y.sinh());
            let d = cr * cr + ci * ci;
            if d == 0.0 { return Err("tan: undefined (cosine is zero)".into()); }
            Ok(make_complex((sr*cr + si*ci)/d, (si*cr - sr*ci)/d))
        }),
        "sinc" => b1!(|v| {
            let (x, y) = to_complex(v)?;
            if x == 0.0 && y == 0.0 { return Ok(Val::Num(1.0)); }
            let (sr, si) = (x.sin() * y.cosh(), x.cos() * y.sinh());
            let d = x * x + y * y;
            Ok(make_complex((sr*x + si*y)/d, (si*x - sr*y)/d))
        }),
        "asin"   => f1!(asin),  "acos"   => f1!(acos),  "atan" => f1!(atan),
        "sinh"   => f1!(sinh),  "cosh"   => f1!(cosh),  "tanh" => f1!(tanh),
        "sech" => b1!(|v| {
            let (a, b) = to_complex(v)?;
            let (cr, ci) = (a.cosh() * b.cos(), a.sinh() * b.sin());
            let d = cr*cr + ci*ci;
            if d == 0.0 { return Err("sech: undefined".into()); }
            Ok(make_complex(cr/d, -ci/d))
        }),
        "csch" => b1!(|v| {
            let (a, b) = to_complex(v)?;
            let (cr, ci) = (a.sinh() * b.cos(), a.cosh() * b.sin());
            let d = cr*cr + ci*ci;
            if d == 0.0 { return Err("csch: undefined at zero".into()); }
            Ok(make_complex(cr/d, -ci/d))
        }),
        "erf"  => b1!(|v| Ok(Val::Num(libm::erf(v.num("erf")?)))),
        "erfc" => b1!(|v| Ok(Val::Num(libm::erfc(v.num("erfc")?)))),
        "j0"   => b1!(|v| Ok(Val::Num(libm::j0(v.num("j0")?)))),
        "j1"   => b1!(|v| Ok(Val::Num(libm::j1(v.num("j1")?)))),
        "jinc" => b1!(|v| {
            let x = v.num("jinc")?;
            if x == 0.0 { return Ok(Val::Num(0.5)); }
            Ok(Val::Num(libm::j1(x) / x))
        }),
        "sec" => b1!(|v| {
            let x = v.num("sec")?;
            let c = x.cos();
            if c == 0.0 { return Err("sec: undefined (cos is zero)".into()); }
            Ok(Val::Num(1.0 / c))
        }),
        "csc" => b1!(|v| {
            let x = v.num("csc")?;
            let s = x.sin();
            if s == 0.0 { return Err("csc: undefined (sin is zero)".into()); }
            Ok(Val::Num(1.0 / s))
        }),
        "cot" => b1!(|v| {
            let x = v.num("cot")?;
            let s = x.sin();
            if s == 0.0 { return Err("cot: undefined (sin is zero)".into()); }
            Ok(Val::Num(x.cos() / s))
        }),
        "heaviside" => b1!(|v| Ok(Val::Num(match v.num("heaviside")? {
            x if x < 0.0 => 0.0,
            x if x > 0.0 => 1.0,
            _             => 0.5,
        }))),
        "expm1"  => f1!(exp_m1),
        "cbrt"   => f1!(cbrt),
        "floor"  => f1!(floor), "ceil" => f1!(ceil),
        "round" => match vals.len() {
            1 => broadcast1(vals.into_iter().next().unwrap(),
                    |v| Ok(Val::Num(v.num("round")?.round()))),
            2 => {
                let mut it = vals.into_iter();
                let x = it.next().unwrap().num("round")?;
                let n = it.next().unwrap().num("round")? as i32;
                let f = 10f64.powi(n);
                Ok(Val::Num((x * f).round() / f))
            }
            n => Err(format!("round expects 1 or 2 args, got {n}")),
        },
        "trunc"  => f1!(trunc),
        "frac"   => b1!(|v| { let x = v.num("frac")?; Ok(Val::Num(x - x.trunc())) }),
        "log10"  => f1!(log10),
        "log" => match vals.len() {
            1 => broadcast1(vals.into_iter().next().unwrap(),
                    |v| Ok(Val::Num(v.num("log")?.log10()))),
            2 => {
                let mut it = vals.into_iter();
                let x    = it.next().unwrap().num("log")?;
                let base = it.next().unwrap().num("log")?;
                if base <= 0.0 || base == 1.0 {
                    return Err("log: base must be positive and ≠ 1".into());
                }
                Ok(Val::Num(x.ln() / base.ln()))
            }
            n => Err(format!("log expects 1 or 2 args, got {n}")),
        },
        "log2"   => f1!(log2),
        "sign" | "signum" => f1!(signum),
        "id"     => b1!(|v| { v.num("id").map(Val::Num) }),
        "delta"  => b1!(|v| Ok(Val::Num(if v.num("delta")? == 0.0 { 1.0 } else { 0.0 }))),
        "not"    => b1!(|v| Ok(Val::Num((int(v.num("not")?) == 0) as i64 as f64))),
        "deg"    => b1!(|v| Ok(Val::Num(v.num("deg")? * (180.0 / std::f64::consts::PI)))),
        "rad"    => b1!(|v| Ok(Val::Num(v.num("rad")? * (std::f64::consts::PI / 180.0)))),
        "len" | "length" => {
            arity(name, 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tuple(items) => Ok(Val::Num(items.len() as f64)),
                Val::Tensor { shape, .. } => Ok(Val::Num(shape[0] as f64)),
                Val::ComplexTensor { shape, .. } => Ok(Val::Num(shape[0] as f64)),
                _ => Err(format!("{name}: argument must be a tuple or tensor")),
            }
        }
        "fact" | "factorial" => b1!(|v| {
            let n = v.num("fact")? as u64;
            Ok(Val::Num((1..=n).map(|k| k as f64).product()))
        }),
        "ncr" => {
            arity("ncr", 2, vals.len())?;
            let mut it = vals.into_iter();
            let n = it.next().unwrap().num("ncr")? as u64;
            let r = it.next().unwrap().num("ncr")? as u64;
            if r > n { return Ok(Val::Num(0.0)); }
            let r = r.min(n - r); // use smaller of r, n-r for efficiency
            let result: f64 = (0..r).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
            Ok(Val::Num(result))
        }
        "quadratic" => {
            arity("quadratic", 3, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num("quadratic")?;
            let b = it.next().unwrap().num("quadratic")?;
            let c = it.next().unwrap().num("quadratic")?;
            if a == 0.0 { return Err("quadratic: leading coefficient a must be nonzero".into()); }
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let s = disc.sqrt();
                Ok(Val::Tuple(vec![
                    Val::Num((-b + s) / (2.0 * a)),
                    Val::Num((-b - s) / (2.0 * a)),
                ]))
            } else {
                let re = -b / (2.0 * a);
                let im = (-disc).sqrt() / (2.0 * a);
                Ok(Val::Tuple(vec![
                    Val::Complex(re,  im),
                    Val::Complex(re, -im),
                ]))
            }
        }

        // ── Polymorphic min / max (scalar pair, tuple, or tensor) ────────────
        "min" | "max" => match (vals.len(), &vals[..]) {
            (1, _) => {
                let nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                    Val::Tensor { data, .. } => data.into_vec(),
                    Val::Tuple(v) => v.into_iter().map(|x| x.num(name)).collect::<Result<_,_>>()?,
                    _ => return Err(format!("{name}: 1-arg form requires a tensor or tuple")),
                };
                if nums.is_empty() { return Err(format!("{name}: empty")); }
                let best = nums.iter().copied().reduce(|a, b| if name == "min" { a.min(b) } else { a.max(b) }).unwrap();
                Ok(Val::Num(best))
            }
            (2, _) => {
                let mut it = vals.into_iter();
                let a = it.next().unwrap().num(name)?;
                let b = it.next().unwrap().num(name)?;
                if name == "min" { Ok(Val::Num(a.min(b))) } else { Ok(Val::Num(a.max(b))) }
            }
            (n, _) => Err(format!("{name} expects 1 or 2 args, got {n}")),
        },

        // ── Real 2-arg ────────────────────────────────────────────────────────
        "atan2" | "pow" | "hypot" |
        "gcd" | "lcm" | "and" | "or" | "xor" | "nand" | "nor" | "xnor" | "shl" | "shr" => {
            arity(name, 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num(name)?;
            let b = it.next().unwrap().num(name)?;
            match name {
                "atan2"  => Ok(Val::Num(a.atan2(b))),
                "pow"    => Ok(Val::Num(a.powf(b))),
                "hypot"  => Ok(Val::Num(a.hypot(b))),
                "gcd"    => Ok(Val::Num(gcd(int(a).unsigned_abs(), int(b).unsigned_abs()) as f64)),
                "lcm"    => Ok(Val::Num(lcm(int(a).unsigned_abs(), int(b).unsigned_abs()) as f64)),
                "and"    => Ok(Val::Num((int(a) & int(b)) as f64)),
                "or"     => Ok(Val::Num((int(a) | int(b)) as f64)),
                "xor"    => Ok(Val::Num((int(a) ^ int(b)) as f64)),
                "nand"   => Ok(Val::Num(((int(a) & int(b)) == 0) as i64 as f64)),
                "nor"    => Ok(Val::Num(((int(a) | int(b)) == 0) as i64 as f64)),
                "xnor"   => Ok(Val::Num(((int(a) ^ int(b)) == 0) as i64 as f64)),
                "shl"    => Ok(Val::Num(int(a).wrapping_shl(int(b) as u32) as f64)),
                "shr"    => Ok(Val::Num(int(a).wrapping_shr(int(b) as u32) as f64)),
                _        => unreachable!(),
            }
        }

        // ── Sequence combinators ──────────────────────────────────────────────
        "sort" => {
            arity("sort", 1, vals.len())?;
            let mut nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tensor { data, .. } => data.into_vec(),
                Val::Tuple(v) => v.into_iter().map(|x| x.num("sort")).collect::<Result<_, _>>()?,
                _ => return Err("sort: argument must be a tensor or tuple".into()),
            };
            nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = nums.len();
            Ok(Val::Tensor { data: TData::new(nums), shape: vec![n] })
        }
        "zip" => {
            arity("zip", 2, vals.len())?;
            // zip(a, b) → 2-D Tensor of shape [n, 2] where each row is a pair
            let mut it = vals.into_iter();
            let av = it.next().unwrap();
            let bv = it.next().unwrap();
            let a: Vec<f64> = match av {
                Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
                Val::Tuple(v) => v.into_iter().map(|x| x.num("zip")).collect::<Result<_, _>>()?,
                _ => return Err("zip: args must be 1D tensors or tuples".into()),
            };
            let b: Vec<f64> = match bv {
                Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
                Val::Tuple(v) => v.into_iter().map(|x| x.num("zip")).collect::<Result<_, _>>()?,
                _ => return Err("zip: args must be 1D tensors or tuples".into()),
            };
            if a.len() != b.len() { return Err(format!("zip: length mismatch ({} vs {})", a.len(), b.len())); }
            let n = a.len();
            let data: Vec<f64> = a.into_iter().zip(b).flat_map(|(x, y)| [x, y]).collect();
            Ok(Val::Tensor { data: TData::new(data), shape: vec![n, 2] })
        }
        "dot" => {
            arity("dot", 2, vals.len())?;
            let mut it = vals.into_iter();
            let av = it.next().unwrap();
            let bv = it.next().unwrap();
            let a: Vec<f64> = match av {
                Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
                Val::Tuple(v) => v.into_iter().map(|x| x.num("dot")).collect::<Result<_, _>>()?,
                _ => return Err("dot: args must be 1D tensors".into()),
            };
            let b: Vec<f64> = match bv {
                Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
                Val::Tuple(v) => v.into_iter().map(|x| x.num("dot")).collect::<Result<_, _>>()?,
                _ => return Err("dot: args must be 1D tensors".into()),
            };
            if a.len() != b.len() { return Err(format!("dot: length mismatch ({} vs {})", a.len(), b.len())); }
            Ok(Val::Num(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()))
        }
        "append" => {
            // append(T, x) — append element x to 1D Tensor T
            arity("append", 2, vals.len())?;
            let mut it = vals.into_iter();
            let first = it.next().unwrap();
            let elem  = it.next().unwrap();
            match first {
                Val::Tensor { mut data, shape } if shape.len() <= 1 => {
                    data.push(elem.num("append")?);  // DerefMut CoW-clones only if Arc has multiple owners
                    let n = data.len();
                    Ok(Val::Tensor { data, shape: vec![n] })
                }
                Val::Tuple(mut v) => { v.push(elem); Ok(Val::Tuple(v)) }
                // Scalar seed: append(x, y) → the 1-D tensor [x, y]. Makes singleton
                // accumulator base cases work without boilerplate (FEAT-E).
                Val::Num(x) => Ok(Val::Tensor { data: TData::new(vec![x, elem.num("append")?]), shape: vec![2] }),
                _ => Err("append: first arg must be a number, 1D tensor, or tuple".into()),
            }
        }
        "concat" => {
            // concat(A, B) — concatenate two 1D tensors along their axis
            arity("concat", 2, vals.len())?;
            let mut it = vals.into_iter();
            let av = it.next().unwrap();
            let bv = it.next().unwrap();
            match (av, bv) {
                (Val::Tuple(mut a), Val::Tuple(b)) => { a.extend(b); Ok(Val::Tuple(a)) }
                // Numeric path: accept scalars, 1-D tensors, and empty operands (FEAT-E).
                (a, b) => {
                    fn as_vec1(v: Val) -> Result<Vec<f64>, String> {
                        match v {
                            Val::Num(x) => Ok(vec![x]),
                            Val::Tensor { data, shape } if shape.len() <= 1 => Ok(data.into_vec()),
                            Val::Tensor { shape, .. } => Err(format!("concat: expected a 1-D tensor, got {}-D", shape.len())),
                            other => Err(format!("concat: cannot concat {}", fmt_val(&other))),
                        }
                    }
                    let mut a = as_vec1(a)?;
                    a.extend(as_vec1(b)?);
                    let n = a.len();
                    Ok(Val::Tensor { data: TData::new(a), shape: vec![n] })
                }
            }
        }
        "flatten" => {
            arity("flatten", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, .. } => {
                    let n = data.len();
                    Ok(Val::Tensor { data, shape: vec![n] })
                }
                Val::Tuple(items) => Ok(Val::Tuple(items.into_iter().flat_map(|v| match v {
                    Val::Tuple(inner) => inner,
                    other             => vec![other],
                }).collect())),
                _ => Err("flatten: argument must be a tensor or tuple".into()),
            }
        }
        "cumsum" | "cumprod" => {
            // Running sum/product of a 1-D tensor or tuple; output length == input.
            arity(name, 1, vals.len())?;
            let product = name == "cumprod";
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() <= 1 => {
                    let mut acc = if product { 1.0 } else { 0.0 };
                    let out: Vec<f64> = data.iter().map(|&x| {
                        if product { acc *= x; } else { acc += x; }
                        acc
                    }).collect();
                    let n = out.len();
                    Ok(Val::Tensor { data: TData::new(out), shape: vec![n] })
                }
                Val::Tuple(items) => {
                    let mut acc_re = if product { 1.0 } else { 0.0 };
                    let mut acc_im = 0.0;
                    let mut out = Vec::with_capacity(items.len());
                    for it in items {
                        let (r, i) = to_complex(it)?;
                        if product {
                            let nr = acc_re * r - acc_im * i;
                            let ni = acc_re * i + acc_im * r;
                            acc_re = nr; acc_im = ni;
                        } else {
                            acc_re += r; acc_im += i;
                        }
                        out.push(make_complex(acc_re, acc_im));
                    }
                    Ok(Val::Tuple(out))
                }
                _ => Err(format!("{name}: argument must be a 1-D tensor or tuple")),
            }
        }
        "diff" => {
            // First difference: out[k] = t[k+1] - t[k]; length == input - 1.
            arity("diff", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() <= 1 => {
                    if data.len() < 2 {
                        return Ok(Val::Tensor { data: TData::new(vec![]), shape: vec![0] });
                    }
                    let out: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
                    let n = out.len();
                    Ok(Val::Tensor { data: TData::new(out), shape: vec![n] })
                }
                Val::Tuple(items) => {
                    if items.len() < 2 { return Ok(Val::Tuple(vec![])); }
                    let mut out = Vec::with_capacity(items.len() - 1);
                    for w in items.windows(2) {
                        let (ar, ai) = to_complex(w[0].clone())?;
                        let (br, bi) = to_complex(w[1].clone())?;
                        out.push(make_complex(br - ar, bi - ai));
                    }
                    Ok(Val::Tuple(out))
                }
                _ => Err("diff: argument must be a 1-D tensor or tuple".into()),
            }
        }
        "argmin" | "argmax" => {
            arity(name, 1, vals.len())?;
            let (nums, shape): (Vec<f64>, Vec<usize>) = match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } => (data.into_vec(), shape),
                Val::Tuple(v) => {
                    let n = v.len();
                    let data = v.into_iter().map(|x| x.num(name)).collect::<Result<Vec<_>, _>>()?;
                    (data, vec![n])
                }
                _ => return Err(format!("{name}: argument must be a tensor or tuple")),
            };
            if nums.is_empty() { return Err(format!("{name}: empty")); }
            let mut best_i = 0usize;
            let mut best_v = nums[0];
            for (i, &n) in nums.iter().enumerate().skip(1) {
                if name == "argmin" { if n < best_v { best_v = n; best_i = i; } }
                else                { if n > best_v { best_v = n; best_i = i; } }
            }
            // For 1-D tensors return a scalar; for n-D return a 1-D index tensor
            if shape.len() == 1 {
                Ok(Val::Num(best_i as f64))
            } else {
                let ndim = shape.len();
                let mut idx = vec![0usize; ndim];
                let mut rem = best_i;
                for k in (0..ndim).rev() {
                    idx[k] = rem % shape[k];
                    rem /= shape[k];
                }
                Ok(Val::Tensor { data: TData::new(idx.into_iter().map(|x| x as f64).collect()), shape: vec![ndim] })
            }
        }

        // ── Statistics ────────────────────────────────────────────────────────
        "mean" => {
            arity("mean", 1, vals.len())?;
            let nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tuple(v)       => v.into_iter().map(|x| x.num("mean")).collect::<Result<_, _>>()?,
                Val::Tensor { data, .. } => data.into_vec(),
                _ => return Err("mean: argument must be a tuple or tensor".into()),
            };
            if nums.is_empty() { return Err("mean: empty".into()); }
            let n = nums.len() as f64;
            Ok(Val::Num(nums.iter().sum::<f64>() / n))
        }
        "median" => {
            arity("median", 1, vals.len())?;
            let mut nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tuple(v)       => v.into_iter().map(|x| x.num("median")).collect::<Result<_, _>>()?,
                Val::Tensor { data, .. } => data.into_vec(),
                _ => return Err("median: argument must be a tuple or tensor".into()),
            };
            if nums.is_empty() { return Err("median: empty".into()); }
            nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = nums.len() / 2;
            Ok(Val::Num(if nums.len() % 2 == 1 { nums[mid] } else { (nums[mid - 1] + nums[mid]) / 2.0 }))
        }
        "mode" => {
            arity("mode", 1, vals.len())?;
            let nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tuple(v)       => v.into_iter().map(|x| x.num("mode")).collect::<Result<_, _>>()?,
                Val::Tensor { data, .. } => data.into_vec(),
                _ => return Err("mode: argument must be a tuple or tensor".into()),
            };
            if nums.is_empty() { return Err("mode: empty".into()); }
            let mut best_val = nums[0];
            let mut best_cnt = 0usize;
            for &candidate in &nums {
                let cnt = nums.iter().filter(|&&x| x == candidate).count();
                if cnt > best_cnt { best_cnt = cnt; best_val = candidate; }
            }
            Ok(Val::Num(best_val))
        }
        "var" => {
            arity("var", 1, vals.len())?;
            let nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tuple(v)       => v.into_iter().map(|x| x.num("var")).collect::<Result<_, _>>()?,
                Val::Tensor { data, .. } => data.into_vec(),
                _ => return Err("var: argument must be a tuple or tensor".into()),
            };
            if nums.is_empty() { return Err("var: empty".into()); }
            let n = nums.len() as f64;
            let mean = nums.iter().sum::<f64>() / n;
            Ok(Val::Num(nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n))
        }
        "std" => {
            arity("std", 1, vals.len())?;
            let nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tuple(v)       => v.into_iter().map(|x| x.num("std")).collect::<Result<_, _>>()?,
                Val::Tensor { data, .. } => data.into_vec(),
                _ => return Err("std: argument must be a tuple or tensor".into()),
            };
            if nums.is_empty() { return Err("std: empty".into()); }
            let n = nums.len() as f64;
            let mean = nums.iter().sum::<f64>() / n;
            Ok(Val::Num((nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()))
        }

        // ── Function combinators ──────────────────────────────────────────────
        // ── Mutable cells ─────────────────────────────────────────────────────
        // cell(v)    — create a new mutable cell holding v
        // get(c)     — read the current value of cell c
        // set(c, v)  — write v into c; returns v (so set can be used inline)
        "cell" => {
            arity("cell", 1, vals.len())?;
            let v = vals.into_iter().next().unwrap();
            Ok(Val::Cell(Arc::new(RefCell::new(v))))
        }
        "get" => {
            arity("get", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Cell(c) => Ok(c.borrow().clone()),
                other => Err(format!("get: expected a cell, got {}", fmt_val(&other))),
            }
        }
        "set" => {
            arity("set", 2, vals.len())?;
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Cell(c) => {
                    let v = it.next().unwrap();
                    *c.borrow_mut() = v.clone();
                    Ok(v)
                }
                other => Err(format!("set: first arg must be a cell, got {}", fmt_val(&other))),
            }
        }

        "compose" => {
            arity("compose", 2, vals.len())?;
            let mut it = vals.into_iter();
            let f = it.next().unwrap();
            let g = it.next().unwrap();
            match (&f, &g) {
                (Val::Fn(..) | Val::Builtin(_), Val::Fn(..) | Val::Builtin(_)) => Ok(compose_fns(f, g)),
                _ => Err("compose: both arguments must be functions".into()),
            }
        }
        "partial" => {
            arity("partial", 2, vals.len())?;
            let mut it = vals.into_iter();
            let f = it.next().unwrap();
            let a = it.next().unwrap();
            match f {
                Val::Fn(params, body, captured, _, sig) => {
                    if params.is_empty() { return Err("partial: function has no parameters".into()); }
                    let first = params[0].clone();
                    let rest  = params[1..].to_vec();
                    let mut new_cap = (*captured).clone();
                    new_cap.insert(first, a);
                    let new_sig = Arc::new(FnSig {
                        params: sig.params.get(1..).unwrap_or(&[]).to_vec(),
                        ret:    sig.ret.clone(),
                    });
                    Ok(Val::Fn(rest, body, Arc::new(new_cap), Arc::new(OnceLock::new()), new_sig))
                }
                Val::Builtin(bname) => {
                    let mut cap = HashMap::new();
                    cap.insert("__b__".into(), Val::Builtin(bname));
                    cap.insert("__a__".into(), a);
                    let body = Expr::Apply(
                        Box::new(Expr::Var("__b__".into())),
                        vec![Expr::Var("__a__".into()), Expr::Var("__z__".into())],
                    );
                    Ok(Val::make_fn(vec!["__z__".into()], body, Arc::new(cap)))
                }
                _ => Err("partial: first argument must be a function".into()),
            }
        }

        // ── Misc ──────────────────────────────────────────────────────────────
        "linspace" => {
            arity("linspace", 3, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num("linspace")?;
            let b = it.next().unwrap().num("linspace")?;
            let n = it.next().unwrap().num("linspace")? as usize;
            if n == 0 { return Err("linspace: n must be ≥ 1".into()); }
            if n == 1 { return Ok(Val::Tensor { data: TData::new(vec![a]), shape: vec![1] }); }
            let data: Vec<f64> = (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect();
            Ok(Val::Tensor { data: TData::new(data), shape: vec![n] })
        }
        "range" => {
            arity("range", 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num("range")? as i64;
            let b = it.next().unwrap().num("range")? as i64;
            let data: Vec<f64> = (a..b).map(|n| n as f64).collect();
            let n = data.len();
            Ok(Val::Tensor { data: TData::new(data), shape: vec![n] })
        }
        "gaussian" => {
            arity("gaussian", 3, vals.len())?;
            let mut it = vals.into_iter();
            let x     = it.next().unwrap().num("gaussian")?;
            let mu    = it.next().unwrap().num("gaussian")?;
            let sigma = it.next().unwrap().num("gaussian")?;
            if sigma == 0.0 { return Err("gaussian: sigma must be nonzero".into()); }
            let z = (x - mu) / sigma;
            Ok(Val::Num((1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()))
        }
        "gaussian_cdf" => {
            arity("gaussian_cdf", 3, vals.len())?;
            let mut it = vals.into_iter();
            let x     = it.next().unwrap().num("gaussian_cdf")?;
            let mu    = it.next().unwrap().num("gaussian_cdf")?;
            let sigma = it.next().unwrap().num("gaussian_cdf")?;
            if sigma == 0.0 { return Err("gaussian_cdf: sigma must be nonzero".into()); }
            Ok(Val::Num(0.5 * (1.0 + libm::erf((x - mu) / (sigma * std::f64::consts::SQRT_2)))))
        }
        "rand" => {
            // rand()              → scalar in [0, 1)
            // rand(n)             → 1-D Tensor of n values
            // rand(n1, n2, …)     → n-D Tensor of that shape
            match vals.len() {
                0 => Ok(Val::Num(rand::random::<f64>())),
                _ => {
                    let shape: Vec<usize> = vals.into_iter()
                        .map(|v| v.num("rand").map(|x| x as usize))
                        .collect::<Result<_, _>>()?;
                    let n: usize = shape.iter().product();
                    let data: Vec<f64> = (0..n).map(|_| rand::random::<f64>()).collect();
                    Ok(Val::Tensor { data: TData::new(data), shape })
                }
            }
        }

        "eps" => {
            if vals.is_empty() { return Err("eps: requires at least 1 argument".into()); }
            let idxs: Vec<i64> = vals.into_iter()
                .map(|v| v.num("eps").map(|x| x as i64))
                .collect::<Result<_, _>>()?;
            let mut sorted = idxs.clone();
            sorted.sort();
            for w in sorted.windows(2) {
                if w[0] == w[1] { return Ok(Val::Num(0.0)); }
            }
            let n = idxs.len();
            let rank_of = |v: i64| sorted.iter().position(|&x| x == v).unwrap();
            let perm: Vec<usize> = idxs.iter().map(|&v| rank_of(v)).collect();
            let mut visited = vec![false; n];
            let mut cycles = 0usize;
            for i in 0..n {
                if !visited[i] {
                    cycles += 1;
                    let mut j = i;
                    while !visited[j] { visited[j] = true; j = perm[j]; }
                }
            }
            Ok(Val::Num(if (n - cycles) % 2 == 0 { 1.0 } else { -1.0 }))
        }

        // fft / ifft — n-D DFT along any subset of axes.
        // Returns a ComplexTensor (or real Tensor if all imaginary parts collapse to zero).
        //
        // Signatures (T = real tensor, axes = scalar or tuple of axis indices):
        //   fft(T)             – forward DFT along all axes
        //   fft(T, axes)       – forward DFT along specified axes
        //   fft(Re, Im)        – forward DFT of complex tensor along all axes
        //   fft(Re, Im, axes)  – forward DFT of complex tensor along specified axes
        // ifft: same, inverse DFT (each axis divided by its size)
        "fft" | "ifft" => {
            let forward = name == "fft";

            // Signatures:
            //   fftn(T)              – forward DFT on real tensor, all axes
            //   fftn(CT)             – forward DFT on complex tensor, all axes
            //   fftn(T, axes)        – forward DFT on real tensor, specified axes
            //   fftn(CT, axes)       – forward DFT on complex tensor, specified axes
            //   fftn(Re, Im)         – forward DFT, Re+i*Im real-tensor pair, all axes
            //   fftn(Re, Im, axes)   – forward DFT, Re+i*Im real-tensor pair, specified axes
            let (mut re_data, mut im_data, shape, axes_opt): (Vec<f64>, Vec<f64>, Vec<usize>, Option<Val>) =
                match vals.len() {
                    1 => {
                        let v = vals.into_iter().next().unwrap();
                        let (re, im, shape) = as_complex_tensor(v)
                            .map_err(|_| format!("{name}: argument must be a tensor or complex tensor"))?;
                        (re, im, shape, None)
                    },
                    2 => {
                        let mut it = vals.into_iter();
                        let a = it.next().unwrap();
                        let b = it.next().unwrap();
                        // Disambiguate: if a and b have the same shape → (Re, Im) pair.
                        // Otherwise → (T, axes).
                        let a_shape: Option<Vec<usize>> = match &a {
                            Val::Tensor { shape, .. } | Val::ComplexTensor { shape, .. }
                                => Some(shape.clone()),
                            _ => None,
                        };
                        let b_shape: Option<Vec<usize>> = match &b {
                            Val::Tensor { shape, .. } | Val::ComplexTensor { shape, .. }
                                => Some(shape.clone()),
                            _ => None,
                        };
                        let same_shape = a_shape.is_some() && a_shape == b_shape;
                        if same_shape {
                            // fftn(Re, Im) — same-shape tensor pair as complex input
                            let (re1, _, sh1) = as_complex_tensor(a)
                                .map_err(|_| format!("{name}: first argument must be a tensor"))?;
                            let (im2, _, _) = as_complex_tensor(b)
                                .map_err(|_| format!("{name}: second argument must be a tensor"))?;
                            (re1, im2, sh1, None)
                        } else {
                            // fftn(T, axes) or fftn(CT, axes)
                            let (re, im, shape) = as_complex_tensor(a)
                                .map_err(|_| format!("{name}: first argument must be a tensor or complex tensor"))?;
                            (re, im, shape, Some(b))
                        }
                    }
                    3 => {
                        let mut it = vals.into_iter();
                        let a = it.next().unwrap();
                        let b = it.next().unwrap();
                        let axes_v = it.next().unwrap();
                        // fftn(Re, Im, axes) — real-tensor pair as complex input, specified axes
                        let (re1, _, sh1) = as_complex_tensor(a)
                            .map_err(|_| format!("{name}: first argument must be a tensor"))?;
                        let (im2, _, sh2) = as_complex_tensor(b)
                            .map_err(|_| format!("{name}: second argument must be a tensor"))?;
                        if sh1 != sh2 {
                            return Err(format!("{name}: Re and Im must have the same shape"));
                        }
                        (re1, im2, sh1, Some(axes_v))
                    }
                    n => return Err(format!("{name} expects 1–3 args, got {n}")),
                };

            // Resolve axes → Vec<usize>
            let axes: Vec<usize> = match axes_opt {
                None                  => (0..shape.len()).collect(),
                Some(Val::Num(n))     => vec![n as usize],
                Some(Val::Tensor { data, .. }) => data.into_iter().map(|x| x as usize).collect(),
                Some(Val::Tuple(items)) => items.into_iter()
                    .map(|v| v.num(name).map(|x| x as usize))
                    .collect::<Result<_, _>>()?,
                _ => return Err(format!("{name}: axes must be a number or tensor of axis indices")),
            };
            for &ax in &axes {
                if ax >= shape.len() {
                    return Err(format!("{name}: axis {ax} out of range for {}-D tensor", shape.len()));
                }
            }

            for &ax in &axes {
                fft_axis_inplace(&mut re_data, &mut im_data, &shape, ax, forward);
            }

            // Return a ComplexTensor (collapses to real Tensor if all im parts are zero)
            Ok(maybe_real(re_data, im_data, shape))
        }

        "lt" | "leq" | "gt" | "geq" | "eq" | "neq" => {
            arity(name, 2, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap().num(name)?;
            let b = it.next().unwrap().num(name)?;
            Ok(Val::Num(match name {
                "lt"  => if a < b  { 1.0 } else { 0.0 },
                "leq" => if a <= b { 1.0 } else { 0.0 },
                "gt"  => if a > b  { 1.0 } else { 0.0 },
                "geq" => if a >= b { 1.0 } else { 0.0 },
                "eq"  => if a == b { 1.0 } else { 0.0 },
                "neq" => if a != b { 1.0 } else { 0.0 },
                _     => unreachable!(),
            }))
        }

        // ── Tensor constructors ───────────────────────────────────────────────
        "tensor" => {
            // tensor(field) — extract a field's grid data as a plain tensor (the
            // field already decayed to its component tensor above: a 0-form gives
            // a grid-shaped tensor, a vector field a grid++[ncomp] tensor). The
            // geometry (lo/hi/spacing) is dropped; rebuild a field with field(...).
            if vals.len() == 1 {
                return match vals.into_iter().next().unwrap() {
                    t @ Val::Tensor { .. } => Ok(t),
                    other => Err(format!("tensor(x): single-arg form converts a field/tensor, got {}", fmt_val(&other))),
                };
            }
            // tensor(f, n1, n2, ...) — variadic; f called with (i0, i1, ...) for each cell
            // f may return real or complex values; if any element is complex, returns ComplexTensor.
            if vals.len() < 2 { return Err("tensor(f, n1, n2, …) expects at least 2 args".into()); }
            let mut it = vals.into_iter();
            let f = it.next().unwrap();
            let shape: Vec<usize> = it.map(|v| v.num("tensor").map(|x| x as usize))
                .collect::<Result<_, _>>()?;
            let ndim = shape.len();
            let total: usize = shape.iter().product();
            let mut results = Vec::with_capacity(total);
            let mut indices = vec![0usize; ndim];
            let mut has_complex = false;
            for _ in 0..total {
                let args: Vec<Val> = indices.iter().map(|&i| Val::Num(i as f64)).collect();
                let v = apply_val(f.clone(), args, env)?;
                match &v {
                    Val::Complex(..) => { has_complex = true; }
                    Val::Num(_) => {}
                    other => return Err(format!("tensor: f must return a number or complex, got {}", fmt_val(other))),
                }
                results.push(v);
                // Advance row-major (rightmost index fastest)
                for k in (0..ndim).rev() {
                    indices[k] += 1;
                    if indices[k] < shape[k] { break; }
                    indices[k] = 0;
                }
            }
            if has_complex {
                let mut re_data = Vec::with_capacity(total);
                let mut im_data = Vec::with_capacity(total);
                for v in results {
                    match v {
                        Val::Num(x)        => { re_data.push(x); im_data.push(0.0); }
                        Val::Complex(a, b) => { re_data.push(a); im_data.push(b); }
                        _                  => unreachable!(),
                    }
                }
                Ok(maybe_real(re_data, im_data, shape))
            } else {
                let data: Vec<f64> = results.into_iter().map(|v| match v { Val::Num(x) => x, _ => unreachable!() }).collect();
                Ok(Val::Tensor { data: TData::new(data), shape })
            }
        }
        "matrix" => {
            // matrix(f, r, c) — 2D convenience wrapper around tensor
            if vals.len() != 3 { return Err("matrix(f, r, c) expects 3 args".into()); }
            let mut it = vals.into_iter();
            let f = it.next().unwrap();
            let r = it.next().unwrap().num("matrix")? as usize;
            let c = it.next().unwrap().num("matrix")? as usize;
            let mut data = Vec::with_capacity(r * c);
            for i in 0..r {
                for j in 0..c {
                    let v = apply_val(f.clone(), vec![Val::Num(i as f64), Val::Num(j as f64)], env)?;
                    data.push(v.num("matrix")?);
                }
            }
            Ok(Val::Tensor { data: TData::new(data), shape: vec![r, c] })
        }
        "zeros" => {
            if vals.is_empty() { return Err("zeros(d0, d1, …) expects at least 1 arg".into()); }
            let shape: Vec<usize> = vals.into_iter()
                .map(|v| v.num("zeros").map(|x| x as usize))
                .collect::<Result<_, _>>()?;
            let n: usize = shape.iter().product();
            Ok(Val::Tensor { data: TData::new(vec![0.0; n]), shape })
        }
        "ones" => {
            if vals.is_empty() { return Err("ones(d0, d1, …) expects at least 1 arg".into()); }
            let shape: Vec<usize> = vals.into_iter()
                .map(|v| v.num("ones").map(|x| x as usize))
                .collect::<Result<_, _>>()?;
            let n: usize = shape.iter().product();
            Ok(Val::Tensor { data: TData::new(vec![1.0; n]), shape })
        }
        "eye" => {
            arity("eye", 1, vals.len())?;
            let n = vals.into_iter().next().unwrap().num("eye")? as usize;
            let mut data = vec![0.0f64; n * n];
            for i in 0..n { data[i * n + i] = 1.0; }
            Ok(Val::Tensor { data: TData::new(data), shape: vec![n, n] })
        }
        "diag" => {
            arity("diag", 1, vals.len())?;
            let nums: Vec<f64> = match vals.into_iter().next().unwrap() {
                Val::Tuple(v) => v.into_iter().map(|x| x.num("diag")).collect::<Result<_, _>>()?,
                Val::Tensor { data, shape } if shape.len() == 1 => data.into_vec(),
                Val::Tensor { .. } => return Err("diag: tensor argument must be 1D".into()),
                _ => return Err("diag: argument must be a tuple or 1D tensor".into()),
            };
            let n = nums.len();
            let mut data = vec![0.0f64; n * n];
            for (i, &x) in nums.iter().enumerate() { data[i * n + i] = x; }
            Ok(Val::Tensor { data: TData::new(data), shape: vec![n, n] })
        }

        // ── Tensor queries ────────────────────────────────────────────────────
        "shape" => {
            arity("shape", 1, vals.len())?;
            let dims: Vec<usize> = match vals.into_iter().next().unwrap() {
                Val::Tensor { shape, .. } | Val::ComplexTensor { shape, .. } => shape,
                Val::Tuple(items) => vec![items.len()],
                _ => return Err("shape: argument must be a tensor or tuple".into()),
            };
            let n = dims.len();
            Ok(Val::Tensor { data: TData::new(dims.into_iter().map(|d| d as f64).collect()), shape: vec![n] })
        }
        "rows" => {
            arity("rows", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { shape, .. } | Val::ComplexTensor { shape, .. } if shape.len() >= 2
                    => Ok(Val::Num(shape[0] as f64)),
                Val::Tensor { .. } | Val::ComplexTensor { .. } => Err("rows: tensor must be at least 2D".into()),
                _ => Err("rows: argument must be a 2D+ tensor".into()),
            }
        }
        "cols" => {
            arity("cols", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { shape, .. } | Val::ComplexTensor { shape, .. } if shape.len() >= 2
                    => Ok(Val::Num(shape[1] as f64)),
                Val::Tensor { .. } | Val::ComplexTensor { .. } => Err("cols: tensor must be at least 2D".into()),
                _ => Err("cols: argument must be a 2D+ tensor".into()),
            }
        }
        "dim" => {
            arity("dim", 2, vals.len())?;
            let mut it = vals.into_iter();
            let axis = {
                let first = it.next().unwrap();
                let axis_val = it.next().unwrap().num("dim")? as usize;
                match first {
                    Val::Tensor { shape, .. } | Val::ComplexTensor { shape, .. } => {
                        if axis_val >= shape.len() {
                            return Err(format!("dim: axis {axis_val} out of range for {}-D tensor", shape.len()));
                        }
                        Ok(Val::Num(shape[axis_val] as f64))
                    }
                    Val::Tuple(items) => {
                        if axis_val != 0 {
                            return Err(format!("dim: axis {axis_val} out of range for 1-D tuple (only axis 0 exists)"));
                        }
                        Ok(Val::Num(items.len() as f64))
                    }
                    _ => Err("dim: first argument must be a tensor or tuple".into()),
                }
            };
            axis
        }

        // ── Tensor operations ─────────────────────────────────────────────────

        // Helper available in this scope for permute-based transforms
        // transpose(T)       – reverse all axes (= classic 2-D transpose for matrices)
        // transpose(T, a, b) – swap axes a and b
        "transpose" => {
            if vals.is_empty() || vals.len() == 2 || vals.len() > 3 {
                return Err("transpose(T) or transpose(T, a, b): expects 1 or 3 args".into());
            }
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } => {
                    let ndim = shape.len();
                    let perm: Vec<usize> = if ndim == 0 {
                        vec![]
                    } else if it.len() == 0 {
                        // Reverse axes
                        (0..ndim).rev().collect()
                    } else {
                        let a = it.next().unwrap().num("transpose")? as usize;
                        let b = it.next().unwrap().num("transpose")? as usize;
                        if a >= ndim || b >= ndim {
                            return Err(format!("transpose: axis out of range for {ndim}-D tensor"));
                        }
                        let mut p: Vec<usize> = (0..ndim).collect();
                        p.swap(a, b);
                        p
                    };
                    apply_permutation(data.into_vec(), &shape, &perm)
                }
                _ => Err("transpose: argument must be a tensor".into()),
            }
        }
        "trace" => {
            arity("trace", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let n = r.min(c);
                    Ok(Val::Num((0..n).map(|i| data[i * c + i]).sum()))
                }
                Val::Tensor { .. } => Err("trace: tensor must be 2D".into()),
                _ => Err("trace: argument must be a 2D tensor".into()),
            }
        }
        "norm" => {
            arity("norm", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, .. } => {
                    Ok(Val::Num(data.iter().map(|x| x * x).sum::<f64>().sqrt()))
                }
                Val::Tuple(items) => {
                    let sum: f64 = items.into_iter()
                        .map(|v| v.num("norm").map(|x| x * x))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter().sum();
                    Ok(Val::Num(sum.sqrt()))
                }
                _ => Err("norm: argument must be a tensor or tuple".into()),
            }
        }
        "row" => {
            arity("row", 2, vals.len())?;
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let i = it.next().unwrap().num("row")? as usize;
                    if i >= r { return Err(format!("row: index {i} out of range (rows={r})")); }
                    Ok(Val::Tensor { data: TData::new(data[i*c..(i+1)*c].to_vec()), shape: vec![c] })
                }
                Val::Tensor { .. } => Err("row: tensor must be 2D".into()),
                _ => Err("row: first argument must be a 2D tensor".into()),
            }
        }
        "col" => {
            arity("col", 2, vals.len())?;
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    let j = it.next().unwrap().num("col")? as usize;
                    if j >= c { return Err(format!("col: index {j} out of range (cols={c})")); }
                    Ok(Val::Tensor { data: TData::new((0..r).map(|i| data[i * c + j]).collect()), shape: vec![r] })
                }
                Val::Tensor { .. } => Err("col: tensor must be 2D".into()),
                _ => Err("col: first argument must be a 2D tensor".into()),
            }
        }
        "matmul" => {
            // Supports 2D×2D, 2D×1D, 1D×2D, and 1D×1D (dot product)
            arity("matmul", 2, vals.len())?;
            let mut it = vals.into_iter();
            match (it.next().unwrap(), it.next().unwrap()) {
                // 2D × 2D → 2D
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh })
                    if ash.len() == 2 && bsh.len() == 2 =>
                {
                    let (ar, ac) = (ash[0], ash[1]);
                    let (br, bc) = (bsh[0], bsh[1]);
                    if ac != br {
                        return Err(format!("matmul: shape mismatch ({ar}×{ac}) @ ({br}×{bc})"));
                    }
                    let mut out = vec![0.0f64; ar * bc];
                    for i in 0..ar {
                        for k in 0..ac {
                            for j in 0..bc {
                                out[i * bc + j] += ad[i * ac + k] * bd[k * bc + j];
                            }
                        }
                    }
                    Ok(Val::Tensor { data: TData::new(out), shape: vec![ar, bc] })
                }
                // 2D × 1D → 1D  (matrix-vector)
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh })
                    if ash.len() == 2 && bsh.len() == 1 =>
                {
                    let (ar, ac) = (ash[0], ash[1]);
                    if ac != bsh[0] {
                        return Err(format!("matmul: shape mismatch ({ar}×{ac}) @ ({},)", bsh[0]));
                    }
                    let mut out = vec![0.0f64; ar];
                    for i in 0..ar {
                        for k in 0..ac { out[i] += ad[i * ac + k] * bd[k]; }
                    }
                    Ok(Val::Tensor { data: TData::new(out), shape: vec![ar] })
                }
                // 1D × 2D → 1D  (vector-matrix)
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh })
                    if ash.len() == 1 && bsh.len() == 2 =>
                {
                    let (br, bc) = (bsh[0], bsh[1]);
                    if ash[0] != br {
                        return Err(format!("matmul: shape mismatch ({},) @ ({br}×{bc})", ash[0]));
                    }
                    let mut out = vec![0.0f64; bc];
                    for j in 0..bc {
                        for k in 0..br { out[j] += ad[k] * bd[k * bc + j]; }
                    }
                    Ok(Val::Tensor { data: TData::new(out), shape: vec![bc] })
                }
                // 1D × 1D → scalar  (dot product)
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh })
                    if ash.len() == 1 && bsh.len() == 1 =>
                {
                    if ash[0] != bsh[0] {
                        return Err(format!("matmul: length mismatch ({} vs {})", ash[0], bsh[0]));
                    }
                    Ok(Val::Num(ad.iter().zip(bd.iter()).map(|(x, y)| x * y).sum()))
                }
                _ => Err("matmul: arguments must be 1D or 2D tensors".into()),
            }
        }
        "outer" => {
            // outer(a, b): outer product of two tensors
            // (d1,...,dm) × (e1,...,en) → (d1,...,dm,e1,...,en)
            arity("outer", 2, vals.len())?;
            let mut it = vals.into_iter();
            match (it.next().unwrap(), it.next().unwrap()) {
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh }) => {
                    let mut shape = ash.clone();
                    shape.extend_from_slice(&bsh);
                    let mut data = Vec::with_capacity(ad.len() * bd.len());
                    for &x in &*ad { for &y in &*bd { data.push(x * y); } }
                    Ok(Val::Tensor { data: TData::new(data), shape })
                }
                _ => Err("outer: both arguments must be tensors".into()),
            }
        }
        "einsum" => {
            // Not yet implemented — requires string literal support.
            let _ = vals;
            return Err("einsum is not yet implemented".into());
        }

        // tensordot(T1, T2, n)       – contract last n axes of T1 with first n of T2
        // tensordot(T1, T2, (a, b))  – contract axis a of T1 with axis b of T2
        // tensordot(T1, T2, ((a1,…),(b1,…))) – contract multiple axis pairs
        "tensordot" => {
            arity("tensordot", 3, vals.len())?;
            let mut it = vals.into_iter();
            let (ad, ash) = match it.next().unwrap() {
                Val::Tensor { data, shape } => (data, shape),
                _ => return Err("tensordot: first argument must be a tensor".into()),
            };
            let (bd, bsh) = match it.next().unwrap() {
                Val::Tensor { data, shape } => (data, shape),
                _ => return Err("tensordot: second argument must be a tensor".into()),
            };
            let axes_val = it.next().unwrap();

            // Resolve the axes spec to (a_axes, b_axes) — which dims to contract.
            // axes can be:
            //   scalar n     → last n of T1 vs first n of T2
            //   [a, b]       → 1D tensor/tuple of 2 → single-axis pair
            //   ([a1,…],[b1,…]) → pair of 1D tensors/tuples → multi-axis pairs
            let (a_axes, b_axes): (Vec<usize>, Vec<usize>) = match axes_val {
                Val::Num(n) => {
                    let n = n as usize;
                    if n > ash.len() || n > bsh.len() {
                        return Err(format!(
                            "tensordot: cannot contract {n} axes (T1 is {}-D, T2 is {}-D)",
                            ash.len(), bsh.len()
                        ));
                    }
                    ((ash.len()-n..ash.len()).collect(), (0..n).collect())
                }
                // 1-D Tensor of 2 elements → (a, b) pair
                Val::Tensor { ref data, ref shape } if shape.len() == 1 && data.len() == 2 => {
                    (vec![data[0] as usize], vec![data[1] as usize])
                }
                // Tuple of 2 → either (a, b) numbers or ((a_axes), (b_axes)) lists
                Val::Tuple(ref pair) if pair.len() == 2 => {
                    match (pair[0].clone(), pair[1].clone()) {
                        (Val::Num(a), Val::Num(b)) => (vec![a as usize], vec![b as usize]),
                        (Val::Tuple(al), Val::Tuple(bl)) => {
                            let a_axes: Result<Vec<usize>, _> = al.into_iter()
                                .map(|v| v.num("tensordot").map(|x| x as usize)).collect();
                            let b_axes: Result<Vec<usize>, _> = bl.into_iter()
                                .map(|v| v.num("tensordot").map(|x| x as usize)).collect();
                            (a_axes?, b_axes?)
                        }
                        // [a1,…] tensor pair
                        (Val::Tensor { data: al, shape: as_ }, Val::Tensor { data: bl, shape: bs_ })
                            if as_.len() == 1 && bs_.len() == 1 => {
                            (al.into_iter().map(|x| x as usize).collect(),
                             bl.into_iter().map(|x| x as usize).collect())
                        }
                        _ => return Err("tensordot: axes pair must be (a, b) or two index lists".into()),
                    }
                }
                _ => return Err("tensordot: axes must be a scalar, [a,b] tensor, or pair".into()),
            };

            if a_axes.len() != b_axes.len() {
                return Err("tensordot: axes lists must have the same length".into());
            }
            for (&a, &b) in a_axes.iter().zip(&b_axes) {
                if a >= ash.len() {
                    return Err(format!("tensordot: axis {a} out of range for {}-D T1", ash.len()));
                }
                if b >= bsh.len() {
                    return Err(format!("tensordot: axis {b} out of range for {}-D T2", bsh.len()));
                }
                if ash[a] != bsh[b] {
                    return Err(format!(
                        "tensordot: contracted axis size mismatch (T1 axis {a} has size {}, T2 axis {b} has size {})",
                        ash[a], bsh[b]
                    ));
                }
            }

            // Free axes: all axes not being contracted.
            let a_free: Vec<usize> = (0..ash.len()).filter(|k| !a_axes.contains(k)).collect();
            let b_free: Vec<usize> = (0..bsh.len()).filter(|k| !b_axes.contains(k)).collect();

            // Output shape = [ash[a_free[0]], …, bsh[b_free[0]], …]
            let out_shape: Vec<usize> = a_free.iter().map(|&k| ash[k])
                .chain(b_free.iter().map(|&k| bsh[k]))
                .collect();

            let a_free_shape: Vec<usize> = a_free.iter().map(|&k| ash[k]).collect();
            let b_free_shape: Vec<usize> = b_free.iter().map(|&k| bsh[k]).collect();
            let contracted_shape: Vec<usize> = a_axes.iter().map(|&k| ash[k]).collect();

            let a_free_total: usize = if a_free_shape.is_empty() { 1 } else { a_free_shape.iter().product() };
            let b_free_total: usize = if b_free_shape.is_empty() { 1 } else { b_free_shape.iter().product() };
            let contracted_total: usize = if contracted_shape.is_empty() { 1 } else { contracted_shape.iter().product() };

            let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
            let out_strides = strides(&out_shape);
            let a_strides = strides(&ash);
            let b_strides = strides(&bsh);

            let mut out_data = vec![0.0f64; out_size];

            for af in 0..a_free_total {
                let af_multi = unravel(af, &a_free_shape);
                for bf in 0..b_free_total {
                    let bf_multi = unravel(bf, &b_free_shape);

                    // Flat index into output
                    let out_flat: usize = af_multi.iter().chain(bf_multi.iter())
                        .zip(out_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();

                    // Sum over contracted indices
                    let mut dot = 0.0f64;
                    for ck in 0..contracted_total {
                        let c_multi = unravel(ck, &contracted_shape);

                        let mut a_multi = vec![0usize; ash.len()];
                        for (&fa, &fi) in a_free.iter().zip(&af_multi) { a_multi[fa] = fi; }
                        for (&ca, &ci) in a_axes.iter().zip(&c_multi)  { a_multi[ca] = ci; }

                        let mut b_multi = vec![0usize; bsh.len()];
                        for (&fb, &fi) in b_free.iter().zip(&bf_multi) { b_multi[fb] = fi; }
                        for (&cb, &ci) in b_axes.iter().zip(&c_multi)  { b_multi[cb] = ci; }

                        let a_flat: usize = a_multi.iter().zip(&a_strides).map(|(&i, &s)| i * s).sum();
                        let b_flat: usize = b_multi.iter().zip(&b_strides).map(|(&i, &s)| i * s).sum();
                        dot += ad[a_flat] * bd[b_flat];
                    }
                    out_data[out_flat] = dot;
                }
            }

            if out_shape.is_empty() {
                Ok(Val::Num(out_data[0]))
            } else {
                Ok(Val::Tensor { data: TData::new(out_data), shape: out_shape })
            }
        }

        // ── Linear algebra ────────────────────────────────────────────────────
        "det" => {
            arity("det", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("det: matrix must be square ({r}×{c})")); }
                    Ok(Val::Num(det_nxn(&data, r)))
                }
                Val::Tensor { .. } => Err("det: argument must be a 2D tensor".into()),
                _ => Err("det: argument must be a square matrix".into()),
            }
        }
        "inv" => {
            arity("inv", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("inv: matrix must be square ({r}×{c})")); }
                    let out = inv_nxn(&data, r)?;
                    Ok(Val::Tensor { data: TData::new(out), shape: vec![r, r] })
                }
                Val::Tensor { .. } => Err("inv: argument must be a 2D tensor".into()),
                _ => Err("inv: argument must be a square matrix".into()),
            }
        }
        "solve" => {
            arity("solve", 2, vals.len())?;
            let mut it = vals.into_iter();
            match (it.next().unwrap(), it.next().unwrap()) {
                (Val::Tensor { data: ad, shape: ash }, Val::Tuple(bv))
                    if ash.len() == 2 =>
                {
                    let (r, c) = (ash[0], ash[1]);
                    if r != c { return Err(format!("solve: matrix must be square ({r}×{c})")); }
                    if bv.len() != r { return Err(format!("solve: b length {} ≠ matrix rows {r}", bv.len())); }
                    let b: Vec<f64> = bv.into_iter().map(|v| v.num("solve")).collect::<Result<_, _>>()?;
                    let x = solve_nxn(&ad, &b, r)?;
                    let n = x.len();
                    Ok(Val::Tensor { data: TData::new(x), shape: vec![n] })
                }
                (Val::Tensor { data: ad, shape: ash }, Val::Tensor { data: bd, shape: bsh })
                    if ash.len() == 2 && bsh.len() == 1 =>
                {
                    let (r, c) = (ash[0], ash[1]);
                    if r != c { return Err(format!("solve: matrix must be square ({r}×{c})")); }
                    if bd.len() != r { return Err(format!("solve: b length {} ≠ matrix rows {r}", bd.len())); }
                    let x = solve_nxn(&ad, &bd, r)?;
                    let n = x.len();
                    Ok(Val::Tensor { data: TData::new(x), shape: vec![n] })
                }
                _ => Err("solve(A, b): A must be a 2D tensor, b must be a tuple or 1D tensor".into()),
            }
        }

        "eig" => {
            arity("eig", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("eig: matrix must be square ({r}×{c})")); }
                    let (lams, evecs) = eig_qr_impl(&data, r);
                    Ok(Val::Tuple(vec![
                        Val::Tensor { data: TData::new(lams), shape: vec![r] },
                        Val::Tensor { data: TData::new(evecs), shape: vec![r, r] },
                    ]))
                }
                Val::Tensor { .. } => Err("eig: argument must be a 2D tensor".into()),
                _ => Err("eig: argument must be a square matrix".into()),
            }
        }
        "eigvals" => {
            arity("eigvals", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("eigvals: matrix must be square ({r}×{c})")); }
                    let (lams, _) = eig_qr_impl(&data, r);
                    Ok(Val::Tensor { data: TData::new(lams), shape: vec![r] })
                }
                Val::Tensor { .. } => Err("eigvals: argument must be a 2D tensor".into()),
                _ => Err("eigvals: argument must be a square matrix".into()),
            }
        }
        "eig_top" => {
            arity("eig_top", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("eig_top: matrix must be square ({r}×{c})")); }
                    let (lam, evec) = power_iter(&data, r);
                    Ok(Val::Tuple(vec![
                        Val::Num(lam),
                        Val::Tensor { data: TData::new(evec), shape: vec![r] },
                    ]))
                }
                Val::Tensor { .. } => Err("eig_top: argument must be a 2D tensor".into()),
                _ => Err("eig_top: argument must be a square matrix".into()),
            }
        }
        "eig_bot" => {
            arity("eig_bot", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("eig_bot: matrix must be square ({r}×{c})")); }
                    let (lam, evec) = inv_power_iter(&data, r)?;
                    Ok(Val::Tuple(vec![
                        Val::Num(lam),
                        Val::Tensor { data: TData::new(evec), shape: vec![r] },
                    ]))
                }
                Val::Tensor { .. } => Err("eig_bot: argument must be a 2D tensor".into()),
                _ => Err("eig_bot: argument must be a square matrix".into()),
            }
        }
        "qr" => {
            arity("qr", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (m, n) = (shape[0], shape[1]);
                    if m < n { return Err(format!("qr: need m ≥ n (got {m}×{n})")); }
                    let (q, r) = qr_householder(&data, m, n);
                    Ok(Val::Tuple(vec![
                        Val::Tensor { data: TData::new(q), shape: vec![m, m] },
                        Val::Tensor { data: TData::new(r), shape: vec![m, n] },
                    ]))
                }
                Val::Tensor { .. } => Err("qr: argument must be a 2D tensor".into()),
                _ => Err("qr: argument must be a matrix".into()),
            }
        }
        "diagonalize" => {
            arity("diagonalize", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 2 => {
                    let (r, c) = (shape[0], shape[1]);
                    if r != c { return Err(format!("diagonalize: matrix must be square ({r}×{c})")); }
                    let (lams, evecs) = eig_qr_impl(&data, r);
                    let mut d = vec![0.0f64; r * r];
                    for i in 0..r { d[i*r+i] = lams[i]; }
                    let v_inv = inv_nxn(&evecs, r)?;
                    Ok(Val::Tuple(vec![
                        Val::Tensor { data: TData::new(evecs), shape: vec![r, r] },
                        Val::Tensor { data: TData::new(d), shape: vec![r, r] },
                        Val::Tensor { data: TData::new(v_inv), shape: vec![r, r] },
                    ]))
                }
                Val::Tensor { .. } => Err("diagonalize: argument must be a 2D tensor".into()),
                _ => Err("diagonalize: argument must be a square matrix".into()),
            }
        }

        // ── Tensor construction / reshaping ───────────────────────────────────
        "hstack" => {
            // Concatenate side-by-side. 1-D vectors are treated as columns [n,1];
            // scalars as [1,1]; matrices kept as-is (rank-promoting, FEAT-E).
            arity("hstack", 2, vals.len())?;
            let mut it = vals.into_iter();
            let (are, aim, ar, ac) = to_block(it.next().unwrap(), false, "hstack")?;
            let (bre, bim, br, bc) = to_block(it.next().unwrap(), false, "hstack")?;
            if ar != br { return Err(format!("hstack: row count mismatch ({ar} vs {br})")); }
            let mut re = Vec::with_capacity(ar * (ac + bc));
            let mut im = Vec::with_capacity(ar * (ac + bc));
            for i in 0..ar {
                re.extend_from_slice(&are[i*ac..(i+1)*ac]);
                re.extend_from_slice(&bre[i*bc..(i+1)*bc]);
                im.extend_from_slice(&aim[i*ac..(i+1)*ac]);
                im.extend_from_slice(&bim[i*bc..(i+1)*bc]);
            }
            Ok(maybe_real(re, im, vec![ar, ac + bc]))
        }
        "vstack" => {
            // Stack vertically. 1-D vectors are treated as rows [1,n]; scalars as
            // [1,1]; matrices kept as-is (rank-promoting, FEAT-E).
            arity("vstack", 2, vals.len())?;
            let mut it = vals.into_iter();
            let (mut are, mut aim, ar, ac) = to_block(it.next().unwrap(), true, "vstack")?;
            let (bre, bim, br, bc) = to_block(it.next().unwrap(), true, "vstack")?;
            if ac != bc { return Err(format!("vstack: column count mismatch ({ac} vs {bc})")); }
            are.extend(bre);
            aim.extend(bim);
            Ok(maybe_real(are, aim, vec![ar + br, ac]))
        }
        "tomat" => {
            arity("tomat", 3, vals.len())?;
            let mut it = vals.into_iter();
            let data: TData = match it.next().unwrap() {
                Val::Tensor { data, shape } if shape.len() == 1 => data,
                Val::Tuple(v) => TData::new(v.into_iter().map(|x| x.num("tomat")).collect::<Result<_, _>>()?),
                _ => return Err("tomat(t, r, c): first arg must be a 1D tensor or tuple".into()),
            };
            let r = it.next().unwrap().num("tomat")? as usize;
            let c = it.next().unwrap().num("tomat")? as usize;
            if data.len() != r * c {
                return Err(format!("tomat: length {} ≠ {r}×{c}={}", data.len(), r*c));
            }
            Ok(Val::Tensor { data, shape: vec![r, c] })
        }
        // ── lerp(a, b, t) — linear interpolation: a*(1-t) + b*t, elementwise ──
        "lerp" => {
            arity("lerp", 3, vals.len())?;
            let mut it = vals.into_iter();
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            let t = it.next().unwrap();
            // Route each step through the right binop (scalar vs tensor).
            fn vop(l: Val, op: &crate::ast::Op, r: Val) -> Result<Val, String> {
                match (&l, &r) {
                    (Val::Num(_), Val::Num(_)) => scalar_binop(l, op, r),
                    _ => binop_tensor(l, op, r),
                }
            }
            let ba  = vop(b, &crate::ast::Op::Sub, a.clone())?;
            let tba = vop(t, &crate::ast::Op::Mul, ba)?;
            vop(a, &crate::ast::Op::Add, tba)
        }

        // ── clamp(x, lo, hi) — elementwise clamp to [lo, hi] ────────────────
        "clamp" => {
            arity("clamp", 3, vals.len())?;
            let mut it = vals.into_iter();
            let x  = it.next().unwrap();
            let lo = it.next().unwrap().num("clamp lo")?;
            let hi = it.next().unwrap().num("clamp hi")?;
            if lo > hi { return Err(format!("clamp: lo ({lo}) > hi ({hi})")); }
            match x {
                Val::Num(v) => Ok(Val::Num(v.clamp(lo, hi))),
                Val::Tensor { data, shape } => Ok(Val::Tensor {
                    data: TData::new(data.into_iter().map(|v| v.clamp(lo, hi)).collect()),
                    shape,
                }),
                other => Err(format!("clamp: expected number or tensor, got {}", fmt_val(&other))),
            }
        }

        // ── shift(T, n, axis) — shift along axis with edge replication ───────
        // Positive n: content moves toward higher indices (pad start with edge).
        // Negative n: content moves toward lower  indices (pad end  with edge).
        // Works on tensors of any rank.
        "shift" => {
            arity("shift", 3, vals.len())?;
            let mut it = vals.into_iter();
            let (data, shape) = match it.next().unwrap() {
                Val::Tensor { data, shape } => (data, shape),
                other => return Err(format!("shift: first arg must be a tensor, got {}", fmt_val(&other))),
            };
            let n    = it.next().unwrap().num("shift n")? as i64;
            let axis = it.next().unwrap().num("shift axis")? as usize;
            if axis >= shape.len() {
                return Err(format!("shift: axis {axis} out of range for rank-{} tensor", shape.len()));
            }
            let total: usize = shape.iter().product();
            // Compute stride for the shifted axis.
            let stride: usize = shape[axis+1..].iter().product();
            let dim_size = shape[axis] as i64;
            let mut out = vec![0.0f64; total];
            for out_flat in 0..total {
                // Decode multi-index for shifted axis only.
                let ax_idx = ((out_flat / stride) % shape[axis]) as i64;
                // Where does this slot come from in the input?
                let in_ax  = (ax_idx - n).clamp(0, dim_size - 1);
                let in_flat = out_flat as i64
                    + (in_ax - ax_idx) * stride as i64;
                out[out_flat] = data[in_flat as usize];
            }
            Ok(Val::Tensor { data: TData::new(out), shape })
        }

        // ── roll(T, n, axis) — circular shift along axis (periodic) ──────────
        // Positive n: content moves toward higher indices (last wraps to front).
        // Equivalent to numpy.roll(T, n, axis).
        "roll" => {
            arity("roll", 3, vals.len())?;
            let mut it = vals.into_iter();
            let (data, shape) = match it.next().unwrap() {
                Val::Tensor { data, shape } => (data, shape),
                other => return Err(format!("roll: first arg must be a tensor, got {}", fmt_val(&other))),
            };
            let n    = it.next().unwrap().num("roll n")? as i64;
            let axis = it.next().unwrap().num("roll axis")? as usize;
            if axis >= shape.len() {
                return Err(format!("roll: axis {axis} out of range for rank-{} tensor", shape.len()));
            }
            let total: usize = shape.iter().product();
            let stride: usize = shape[axis+1..].iter().product();
            let dim_size = shape[axis] as i64;
            let mut out = vec![0.0f64; total];
            for out_flat in 0..total {
                let ax_idx = ((out_flat / stride) % shape[axis]) as i64;
                let in_ax  = (ax_idx - n).rem_euclid(dim_size);
                let in_flat = out_flat as i64
                    + (in_ax - ax_idx) * stride as i64;
                out[out_flat] = data[in_flat as usize];
            }
            Ok(Val::Tensor { data: TData::new(out), shape })
        }

        "lingrid" => {
            // lingrid(start, end, counts, f)
            // start / end / counts can each be a scalar (1-D) or a k-tuple (k-D)
            if vals.len() != 4 { return Err("lingrid(start, end, counts, f) expects 4 args".into()); }
            let mut it = vals.into_iter();

            // Helper: extract a Vec<f64> from a scalar, 1D tensor, or tuple
            fn as_vec(v: Val, label: &str) -> Result<Vec<f64>, String> {
                match v {
                    Val::Num(n)                               => Ok(vec![n]),
                    Val::Tensor { data, shape } if shape.len() == 1 => Ok(data.into_vec()),
                    Val::Tuple(items)                         => items.into_iter()
                        .map(|x| x.num(label))
                        .collect::<Result<Vec<_>, _>>(),
                    _ => Err(format!("lingrid: {label} must be a number or 1D tensor")),
                }
            }

            let starts = as_vec(it.next().unwrap(), "start")?;
            let ends   = as_vec(it.next().unwrap(), "end")?;
            let ns_f   = as_vec(it.next().unwrap(), "counts")?;
            let f      = it.next().unwrap();

            let ndim = starts.len();
            if ends.len() != ndim || ns_f.len() != ndim {
                return Err(format!(
                    "lingrid: start/end/counts must all have the same length \
                     (got {}, {}, {})", ndim, ends.len(), ns_f.len()
                ));
            }
            let ns: Vec<usize> = ns_f.iter().map(|&x| x as usize).collect();
            for (k, &n) in ns.iter().enumerate() {
                if n < 2 { return Err(format!("lingrid: counts[{k}] must be >= 2")); }
            }

            let total: usize = ns.iter().product();

            // Helper: flatten a function return value to (flat_data, value_shape).
            // Scalar        → (vec![n], [])
            // 1-D Tensor    → (data, [k])
            // n-D Tensor    → (data, shape)
            // Tuple         → (vec![n0,n1,…], [k])  (legacy; numeric tuples now auto-promoted)
            fn flatten_val(v: Val) -> Result<(Vec<f64>, Vec<usize>), String> {
                match v {
                    Val::Num(n)                 => Ok((vec![n], vec![])),
                    Val::Tensor { data, shape } => Ok((data.into_vec(), shape)),
                    Val::Tuple(items)           => {
                        let d: Vec<f64> = items.into_iter()
                            .map(|x| x.num("lingrid value"))
                            .collect::<Result<_, _>>()?;
                        let k = d.len();
                        Ok((d, vec![k]))
                    }
                    other => Err(format!(
                        "lingrid: function must return a number or tensor (got {})",
                        fmt_val(&other)
                    )),
                }
            }

            // Evaluate all grid points; determine value_shape from the first result.
            let mut data: Vec<f64> = Vec::new();
            let mut val_shape: Option<Vec<usize>> = None;
            let mut idx = vec![0usize; ndim];

            for _ in 0..total {
                let coords: Vec<Val> = (0..ndim).map(|k| {
                    let t = idx[k] as f64 / (ns[k] - 1) as f64;
                    Val::Num(starts[k] + t * (ends[k] - starts[k]))
                }).collect();
                let v = apply_val(f.clone(), coords, env)?;
                let (flat, vshape) = flatten_val(v)?;
                // Validate consistency with first result
                match &val_shape {
                    None     => { val_shape = Some(vshape.clone()); }
                    Some(vs) => {
                        if *vs != vshape {
                            return Err(format!(
                                "lingrid: function returned inconsistent shapes \
                                 ({:?} vs {:?})", vs, vshape
                            ));
                        }
                    }
                }
                data.extend(flat);
                // Advance row-major
                for k in (0..ndim).rev() {
                    idx[k] += 1;
                    if idx[k] < ns[k] { break; }
                    idx[k] = 0;
                }
            }

            // Output shape = grid_shape ++ value_shape
            let mut out_shape = ns.clone();
            out_shape.extend(val_shape.unwrap_or_default());
            Ok(Val::Tensor { data: TData::new(data), shape: out_shape })
        }

        // ── Shape manipulation ────────────────────────────────────────────────

        // reshape(T, n0, n1, …)  – reinterpret data with a new shape
        "reshape" => {
            if vals.len() < 2 { return Err("reshape(T, n0, n1, …) expects at least 2 args".into()); }
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } => {
                    let new_shape: Vec<usize> = it.map(|v| v.num("reshape").map(|x| x as usize))
                        .collect::<Result<_, _>>()?;
                    let old_n: usize = shape.iter().product();
                    let new_n: usize = new_shape.iter().product();
                    if old_n != new_n {
                        return Err(format!("reshape: size mismatch ({old_n} vs {new_n})"));
                    }
                    Ok(Val::Tensor { data, shape: new_shape })  // data (TData) reused as-is: O(1)
                }
                _ => Err("reshape: first argument must be a tensor".into()),
            }
        }

        // permute(T, p0, p1, …)  – reorder axes; perm[k] = which input axis → output axis k
        "permute" => {
            if vals.len() < 2 { return Err("permute(T, p0, p1, …) expects at least 2 args".into()); }
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, shape } => {
                    let ndim = shape.len();
                    let perm: Vec<usize> = it.map(|v| v.num("permute").map(|x| x as usize))
                        .collect::<Result<_, _>>()?;
                    if perm.len() != ndim {
                        return Err(format!("permute: need {ndim} axis indices, got {}", perm.len()));
                    }
                    let mut seen = vec![false; ndim];
                    for &p in &perm {
                        if p >= ndim { return Err(format!("permute: axis {p} out of range for {ndim}-D tensor")); }
                        if seen[p]  { return Err(format!("permute: axis {p} appears more than once")); }
                        seen[p] = true;
                    }
                    apply_permutation(data.into_vec(), &shape, &perm)
                }
                _ => Err("permute: first argument must be a tensor".into()),
            }
        }

        // cat(axis, T1, T2, …)  – concatenate tensors along an existing axis
        "cat" => {
            if vals.len() < 3 { return Err("cat(axis, T1, T2, …) expects at least 3 args".into()); }
            let mut it = vals.into_iter();
            let axis = it.next().unwrap().num("cat")? as usize;
            let tensors: Vec<(Vec<f64>, Vec<usize>)> = it.map(|v| match v {
                Val::Tensor { data, shape } => Ok((data.into_vec(), shape)),
                _ => Err(String::from("cat: all arguments after axis must be tensors")),
            }).collect::<Result<Vec<(Vec<f64>, Vec<usize>)>, String>>()?;
            let ndim = tensors[0].1.len();
            for (_, sh) in &tensors {
                if sh.len() != ndim {
                    return Err("cat: all tensors must have the same number of dimensions".into());
                }
            }
            if axis >= ndim { return Err(format!("cat: axis {axis} out of range for {ndim}-D tensors")); }
            for dim in 0..ndim {
                if dim == axis { continue; }
                let d0 = tensors[0].1[dim];
                for (_, sh) in &tensors {
                    if sh[dim] != d0 {
                        return Err(format!("cat: dimension {dim} mismatch ({d0} vs {})", sh[dim]));
                    }
                }
            }
            // Build output shape
            let mut out_shape = tensors[0].1.clone();
            out_shape[axis] = tensors.iter().map(|(_, s)| s[axis]).sum();
            let out_strides = strides(&out_shape);
            let out_size: usize = out_shape.iter().product();
            let mut out_data = vec![0.0f64; out_size];
            // Copy each tensor's data at the right offset
            let mut axis_offset = 0usize;
            for (data, shape) in tensors {
                let t_size: usize = shape.iter().product();
                for in_flat in 0..t_size {
                    let mut multi = unravel(in_flat, &shape);
                    multi[axis] += axis_offset;
                    let out_flat: usize = multi.iter().zip(&out_strides).map(|(&i, &s)| i * s).sum();
                    out_data[out_flat] = data[in_flat];
                }
                axis_offset += shape[axis];
            }
            Ok(Val::Tensor { data: TData::new(out_data), shape: out_shape })
        }

        // squeeze(T)        – remove all size-1 dimensions
        "squeeze" => {
            arity("squeeze", 1, vals.len())?;
            match vals.into_iter().next().unwrap() {
                Val::Tensor { data, shape } => {
                    let new_shape: Vec<usize> = shape.into_iter().filter(|&d| d != 1).collect();
                    if new_shape.is_empty() {
                        Ok(Val::Num(*data.first().unwrap_or(&0.0)))
                    } else {
                        Ok(Val::Tensor { data, shape: new_shape })  // data (TData) reused: O(1)
                    }
                }
                _ => Err("squeeze: argument must be a tensor".into()),
            }
        }

        // unsqueeze(T, dim) – insert a size-1 dimension at position dim
        "unsqueeze" => {
            arity("unsqueeze", 2, vals.len())?;
            let mut it = vals.into_iter();
            match it.next().unwrap() {
                Val::Tensor { data, mut shape } => {
                    let dim = it.next().unwrap().num("unsqueeze")? as usize;
                    if dim > shape.len() {
                        return Err(format!("unsqueeze: dim {dim} out of range (ndim={})", shape.len()));
                    }
                    shape.insert(dim, 1);
                    Ok(Val::Tensor { data, shape })  // data (TData) reused: O(1)
                }
                _ => Err("unsqueeze: first argument must be a tensor".into()),
            }
        }

        _ => Err(format!("undefined function: {name}")),
    }
}

// ── Bytecode compiler ─────────────────────────────────────────────────────────
// Compiles a lambda body to Vec<Instruction>.  Returns None for any unsupported
// node (range, tensor literal, func-def in block, special forms that need
// unevaluated Expr args).  None triggers a tree-walk fallback.

fn has_slice(expr: &Expr) -> bool {
    matches!(expr, Expr::Slice(..))
        || matches!(expr, Expr::Tuple(es) if es.iter().any(|e| matches!(e, Expr::Slice(..))))
}

/// Collect names from `expr` that are free in the inner lambda — i.e. they appear
/// in `outer_params` or `outer_locals` but not in `inner_params` or `outer_captured`.
/// These must be explicitly pushed onto the stack before a MakeClosure instruction.
fn collect_free_vars(
    expr:           &Expr,
    inner_params:   &[String],
    outer_params:   &[String],
    outer_locals:   &[String],
    outer_captured: &HashMap<String, Val>,
) -> Vec<String> {
    let mut vars: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    cfv(expr, inner_params, outer_params, outer_locals, outer_captured, &mut vars, &mut seen);
    vars
}

fn cfv(
    expr:           &Expr,
    inner_params:   &[String],
    outer_params:   &[String],
    outer_locals:   &[String],
    outer_captured: &HashMap<String, Val>,
    vars:           &mut Vec<String>,
    seen:           &mut std::collections::HashSet<String>,
) {
    match expr {
        Expr::Num(_) | Expr::ImagLit(_) => {}
        Expr::Var(name) => {
            // Capture if the name is bound in the outer scope (params or locals) but not
            // already an inner param.  We intentionally ignore outer_captured here: a
            // local variable can shadow a captured builtin, and the local must win.
            if !inner_params.contains(name)
                && (outer_params.contains(name) || outer_locals.contains(name))
                && seen.insert(name.clone())
            {
                vars.push(name.clone());
            }
        }
        Expr::BinOp(l, _, r) => {
            cfv(l, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
            cfv(r, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
        }
        Expr::Neg(e) | Expr::Not(e) => cfv(e, inner_params, outer_params, outer_locals, outer_captured, vars, seen),
        Expr::Apply(f, args) => {
            cfv(f, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
            for a in args { cfv(a, inner_params, outer_params, outer_locals, outer_captured, vars, seen); }
        }
        Expr::Tuple(es) | Expr::Array(es) => {
            for e in es { cfv(e, inner_params, outer_params, outer_locals, outer_captured, vars, seen); }
        }
        Expr::Index(base, idx) => {
            cfv(base, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
            cfv(idx, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
        }
        Expr::Member(base, _) => cfv(base, inner_params, outer_params, outer_locals, outer_captured, vars, seen),
        Expr::Lambda(ps, _, body) => {
            let mut new_inner = inner_params.to_vec();
            new_inner.extend(ps.iter().map(|p| p.name.clone()));
            cfv(body, &new_inner, outer_params, outer_locals, outer_captured, vars, seen);
        }
        Expr::Block(stmts) => {
            let mut ext = inner_params.to_vec();
            for stmt in stmts {
                match stmt {
                    BlockStmt::Def(Def::Var(name, body)) => {
                        cfv(body, &ext, outer_params, outer_locals, outer_captured, vars, seen);
                        ext.push(name.clone());
                    }
                    BlockStmt::Def(Def::Func(name, ps, _, body)) => {
                        ext.push(name.clone());
                        let mut fn_inner = ext.clone();
                        fn_inner.extend(ps.iter().map(|p| p.name.clone()));
                        cfv(body, &fn_inner, outer_params, outer_locals, outer_captured, vars, seen);
                    }
                    BlockStmt::Expr(e) => cfv(e, &ext, outer_params, outer_locals, outer_captured, vars, seen),
                }
            }
        }
        Expr::TensorLit(rows) => {
            for row in rows { for e in row { cfv(e, inner_params, outer_params, outer_locals, outer_captured, vars, seen); } }
        }
        Expr::Range(lo, hi) => {
            cfv(lo, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
            cfv(hi, inner_params, outer_params, outer_locals, outer_captured, vars, seen);
        }
        Expr::Slice(lo, hi) => {
            if let Some(e) = lo { cfv(e, inner_params, outer_params, outer_locals, outer_captured, vars, seen); }
            if let Some(e) = hi { cfv(e, inner_params, outer_params, outer_locals, outer_captured, vars, seen); }
        }
    }
}

/// Returns true if `target` appears as a free variable anywhere in `expr`
/// (accounting for shadowing by inner params/locals).
fn expr_contains_var(expr: &Expr, target: &str) -> bool {
    match expr {
        Expr::Var(name)      => name == target,
        Expr::Num(_) | Expr::ImagLit(_) => false,
        Expr::BinOp(l, _, r) => expr_contains_var(l, target) || expr_contains_var(r, target),
        Expr::Neg(e) | Expr::Not(e) => expr_contains_var(e, target),
        Expr::Apply(f, args) => expr_contains_var(f, target) || args.iter().any(|a| expr_contains_var(a, target)),
        Expr::Tuple(es) | Expr::Array(es) => es.iter().any(|e| expr_contains_var(e, target)),
        Expr::Index(b, i)    => expr_contains_var(b, target) || expr_contains_var(i, target),
        Expr::Member(b, _)   => expr_contains_var(b, target),
        Expr::Lambda(ps, _, body) => !ps.iter().any(|p| p.name == target) && expr_contains_var(body, target),
        Expr::Block(stmts) => {
            for stmt in stmts {
                match stmt {
                    BlockStmt::Def(Def::Var(name, body)) => {
                        if expr_contains_var(body, target) { return true; }
                        if name == target { return false; } // shadowed from here on
                    }
                    BlockStmt::Def(Def::Func(name, ps, _, body)) => {
                        if name == target { return false; } // shadowed from here on
                        if !ps.iter().any(|p| p.name == target) && expr_contains_var(body, target) { return true; }
                    }
                    BlockStmt::Expr(e) => if expr_contains_var(e, target) { return true; },
                }
            }
            false
        }
        Expr::TensorLit(rows) => rows.iter().any(|row| row.iter().any(|e| expr_contains_var(e, target))),
        Expr::Range(lo, hi)  => expr_contains_var(lo, target) || expr_contains_var(hi, target),
        Expr::Slice(lo, hi)  => lo.as_ref().map_or(false, |e| expr_contains_var(e, target))
                             || hi.as_ref().map_or(false, |e| expr_contains_var(e, target)),
    }
}

struct Compiler<'a> {
    params:   &'a [String],
    captured: &'a HashMap<String, Val>,
    code:     Vec<Instruction>,
    locals:   Vec<String>,
}

impl<'a> Compiler<'a> {
    fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p == name)
    }
    fn local_index(&self, name: &str) -> Option<usize> {
        self.locals.iter().position(|l| l == name)
    }

    fn compile(&mut self, expr: &Expr) -> Result<(), ()> {
        match expr {
            Expr::Num(n)     => self.code.push(Instruction::PushNum(*n)),
            Expr::ImagLit(n) => self.code.push(Instruction::PushComplex(0.0, *n)),

            Expr::Var(name) => {
                // Locals shadow params (enables reassignment of params inside blocks).
                if let Some(i) = self.local_index(name) {
                    self.code.push(Instruction::LoadLocal(i));
                } else if let Some(i) = self.param_index(name) {
                    self.code.push(Instruction::LoadParam(i));
                } else if let Some(v) = self.captured.get(name.as_str()) {
                    match v {
                        // Inline scalar constants — avoids a HashMap lookup on every call.
                        Val::Num(x)        => self.code.push(Instruction::PushNum(*x)),
                        Val::Complex(a, b) => self.code.push(Instruction::PushComplex(*a, *b)),
                        // Cells must be loaded live; their inner value can change.
                        // Everything else (Builtin, Fn, Tensor, Tuple) is also loaded live
                        // so the VM uses the current binding, not a stale snapshot.
                        _                  => self.code.push(Instruction::LoadCaptured(name.clone())),
                    }
                } else {
                    return Err(());
                }
            }

            Expr::BinOp(l, op, r) => {
                self.compile(l)?;
                self.compile(r)?;
                self.code.push(Instruction::BinOp(op.clone()));
            }

            Expr::Neg(e) => { self.compile(e)?; self.code.push(Instruction::Neg); }
            Expr::Not(_) => return Err(()),  // fall back to tree-walk

            Expr::Apply(f_expr, arg_exprs) => {
                if let Expr::Var(name) = f_expr.as_ref() {
                    match name.as_str() {
                        // if → conditional jump pair
                        "if" => {
                            if arg_exprs.len() != 3 { return Err(()); }
                            self.compile(&arg_exprs[0])?;
                            let jf_pos = self.code.len();
                            self.code.push(Instruction::JumpIfFalse(0)); // patched below
                            self.compile(&arg_exprs[1])?;
                            let jmp_pos = self.code.len();
                            self.code.push(Instruction::Jump(0));         // patched below
                            let else_pc = self.code.len();
                            self.compile(&arg_exprs[2])?;
                            let end_pc  = self.code.len();
                            self.code[jf_pos]  = Instruction::JumpIfFalse(else_pc);
                            self.code[jmp_pos]  = Instruction::Jump(end_pc);
                            return Ok(());
                        }
                        // Bounded-iteration special forms compile to a flat VM Loop:
                        // push the operands (function + bounds/seed/count), then a
                        // single Loop instruction runs the iteration with no native-
                        // stack growth — the only GPU-safe recursion analogue (TODO 1e).
                        // f is evaluated once, exactly as the tree-walk front-end does.
                        "sum" | "prod" | "iterate" | "scan" => {
                            let form = match name.as_str() {
                                "sum"     => LoopForm::Sum,
                                "prod"    => LoopForm::Prod,
                                "iterate" => LoopForm::Iterate,
                                _         => LoopForm::Scan,
                            };
                            for a in arg_exprs { self.compile(a)?; }
                            self.code.push(Instruction::Loop(form, arg_exprs.len()));
                            return Ok(());
                        }
                        // Still need unevaluated Expr args / bespoke handling — fall back.
                        "integral" | "deriv" | "map" | "filter" | "reduce" => return Err(()),
                        _ => {}
                    }
                    // Treat as builtin when not shadowed by a param/local.
                    // Use the *actual* builtin name from captured, not the variable name —
                    // compose/partial alias builtins as __f__, __g__, __b__ etc.
                    let builtin_name = if self.param_index(name).is_none()
                        && self.local_index(name).is_none()
                    {
                        if let Some(Val::Builtin(bname)) = self.captured.get(name.as_str()) {
                            Some(bname.clone())
                        } else { None }
                    } else { None };
                    if let Some(bname) = builtin_name {
                        for a in arg_exprs { self.compile(a)?; }
                        self.code.push(Instruction::CallBuiltin(bname, arg_exprs.len()));
                    } else {
                        // Computed callable (user fn stored in a captured var or local).
                        self.compile(f_expr)?;
                        for a in arg_exprs { self.compile(a)?; }
                        self.code.push(Instruction::CallVal(arg_exprs.len()));
                    }
                } else {
                    // Non-name callee (e.g. result of an expression).
                    self.compile(f_expr)?;
                    for a in arg_exprs { self.compile(a)?; }
                    self.code.push(Instruction::CallVal(arg_exprs.len()));
                }
            }

            Expr::Tuple(exprs) => {
                for e in exprs { self.compile(e)?; }
                self.code.push(Instruction::MakeTuple(exprs.len()));
            }

            Expr::Array(exprs) => {
                for e in exprs { self.compile(e)?; }
                self.code.push(Instruction::MakeArray(exprs.len()));
            }

            Expr::Block(stmts) => {
                let n = stmts.len();
                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i + 1 == n;
                    match stmt {
                        BlockStmt::Def(Def::Var(name, body)) => {
                            self.compile(body)?;
                            // Reuse the existing slot if this name was already defined in
                            // this block — enables in-order reassignment (x = 1; x = 2).
                            let slot = if let Some(existing) = self.local_index(name) {
                                existing
                            } else {
                                let s = self.locals.len();
                                self.locals.push(name.clone());
                                s
                            };
                            self.code.push(Instruction::StoreLocal(slot));
                        }
                        // Non-recursive Def::Func compiles as a lambda stored in a local.
                        // Recursive functions (body references own name) fall back to the
                        // tree-walk evaluator, which sets up the self-reference correctly.
                        BlockStmt::Def(Def::Func(name, params, _ret, body)) => {
                            if expr_contains_var(body, name) { return Err(()); }
                            self.compile(&Expr::Lambda(params.clone(), None, Box::new(body.clone())))?;
                            let slot = if let Some(existing) = self.local_index(name) {
                                existing
                            } else {
                                let s = self.locals.len();
                                self.locals.push(name.clone());
                                s
                            };
                            self.code.push(Instruction::StoreLocal(slot));
                        }
                        BlockStmt::Expr(e) => {
                            self.compile(e)?;
                            if !is_last { self.code.push(Instruction::Pop); }
                        }
                    }
                }
            }

            Expr::Index(base, idx) => {
                if has_slice(idx) { return Err(()); }
                self.compile(base)?;
                self.compile(idx)?;
                self.code.push(Instruction::Index);
            }

            Expr::Lambda(params_with_hints, _ret, inner_body) => {
                let inner_params: Vec<String> = params_with_hints.iter().map(|p| p.name.clone()).collect();
                let inner_params_slice = inner_params.as_slice();
                let free_vars = collect_free_vars(
                    inner_body,
                    inner_params_slice,
                    self.params,
                    &self.locals,
                    self.captured,
                );
                for name in &free_vars {
                    if let Some(i) = self.local_index(name) {
                        self.code.push(Instruction::LoadLocal(i));
                    } else if let Some(i) = self.param_index(name) {
                        self.code.push(Instruction::LoadParam(i));
                    } else {
                        return Err(());
                    }
                }
                // Build a captured hint for the inner compiler: real outer-captured values
                // (so scalars get inlined), plus a non-inlinable placeholder for each free var
                // (so the inner compiler emits LoadCaptured rather than a stale literal).
                let mut hint = self.captured.clone();
                for name in &free_vars {
                    hint.insert(name.clone(), Val::Tuple(vec![]));
                }
                let hint_arc = Arc::new(hint);
                let inner_code = compile_fn(inner_params_slice, inner_body, &hint_arc)
                    .map(Arc::new)
                    .unwrap_or_else(|| Arc::new(vec![]));
                self.code.push(Instruction::MakeClosure {
                    params: inner_params,
                    body:   Arc::new(*inner_body.clone()),
                    code:   inner_code,
                    free_vars,
                });
            }

            // Unsupported: ranges, slices, tensor literals.
            _ => return Err(()),
        }
        Ok(())
    }
}

/// Compile a function body to bytecode.  Returns None if any node is unsupported;
/// the caller falls back to the tree-walk evaluator.
fn compile_fn(
    params:   &[String],
    body:     &Expr,
    captured: &Arc<HashMap<String, Val>>,
) -> Option<Vec<Instruction>> {
    let mut c = Compiler { params, captured, code: vec![], locals: vec![] };
    c.compile(body).ok()?;
    c.code.push(Instruction::Return);
    Some(c.code)
}

// ── Bytecode VM helpers ────────────────────────────────────────────────────────

/// Normalize an index along an axis of length `dim`: negatives count from the end.
/// Errors on out-of-range in *either* direction (positive overflow AND negative
/// underflow) — the latter previously clamped silently to 0, masking bugs.
fn norm_index(raw: i64, dim: usize, what: &str) -> Result<usize, String> {
    let i = if raw < 0 { raw + dim as i64 } else { raw };
    if i < 0 || i >= dim as i64 {
        return Err(format!("{what} index {raw} out of range (size={dim})"));
    }
    Ok(i as usize)
}

fn vm_tensor_index(data: &[f64], shape: &[usize], idx: &Val) -> Result<Val, String> {
    let indices: Vec<i64> = match idx {
        Val::Num(f) => vec![*f as i64],
        Val::Tensor { data: id, shape: is_ } if is_.len() == 1 => id.iter().map(|&f| f as i64).collect(),
        Val::Tuple(items) => items.iter().map(|v| match v {
            Val::Num(f) => Ok(*f as i64),
            _ => Err("vm []: multi-index must contain numbers".to_string()),
        }).collect::<Result<_, _>>()?,
        _ => return Err("vm []: invalid index type".into()),
    };
    if indices.len() == 1 {
        let i = norm_index(indices[0], shape[0], "tensor")?;
        if shape.len() == 1 {
            Ok(Val::Num(data[i]))
        } else {
            let sub: usize = shape[1..].iter().product();
            Ok(Val::Tensor { data: TData::new(data[i*sub..(i+1)*sub].to_vec()), shape: shape[1..].to_vec() })
        }
    } else {
        if indices.len() != shape.len() {
            return Err(format!("tensor: expected {} indices, got {}", shape.len(), indices.len()));
        }
        let mut linear = 0usize;
        let mut stride = 1usize;
        for (&raw, &dim) in indices.iter().rev().zip(shape.iter().rev()) {
            let i = norm_index(raw, dim, "tensor")?;
            linear += i * stride;
            stride *= dim;
        }
        Ok(Val::Num(data[linear]))
    }
}

fn vm_complex_tensor_index(re: &[f64], im: &[f64], shape: &[usize], idx: &Val) -> Result<Val, String> {
    let indices: Vec<i64> = match idx {
        Val::Num(f) => vec![*f as i64],
        Val::Tensor { data: id, shape: is_ } if is_.len() == 1 => id.iter().map(|&f| f as i64).collect(),
        Val::Tuple(items) => items.iter().map(|v| match v {
            Val::Num(f) => Ok(*f as i64),
            _ => Err("vm []: multi-index must contain numbers".to_string()),
        }).collect::<Result<_, _>>()?,
        _ => return Err("vm []: invalid index type".into()),
    };
    if indices.len() == 1 {
        let i = norm_index(indices[0], shape[0], "tensor")?;
        if shape.len() == 1 {
            Ok(make_complex(re[i], im[i]))
        } else {
            let sub: usize = shape[1..].iter().product();
            let s = i * sub;
            Ok(maybe_real(re[s..s+sub].to_vec(), im[s..s+sub].to_vec(), shape[1..].to_vec()))
        }
    } else {
        if indices.len() != shape.len() {
            return Err(format!("tensor: expected {} indices, got {}", shape.len(), indices.len()));
        }
        let mut linear = 0usize;
        let mut stride = 1usize;
        for (&raw, &dim) in indices.iter().rev().zip(shape.iter().rev()) {
            let i = norm_index(raw, dim, "tensor")?;
            linear += i * stride;
            stride *= dim;
        }
        Ok(make_complex(re[linear], im[linear]))
    }
}

// ── Bytecode VM ───────────────────────────────────────────────────────────────

fn run_vm(
    code:     &[Instruction],
    args:     &[Val],
    captured: &Arc<HashMap<String, Val>>,
    env:      &Env,
) -> Result<Val, String> {
    let mut stack:  Vec<Val> = Vec::with_capacity(16);
    let mut locals: Vec<Val> = Vec::new();
    let mut pc = 0usize;

    loop {
        match &code[pc] {
            Instruction::PushNum(n)         => stack.push(Val::Num(*n)),
            Instruction::PushComplex(a, b)  => stack.push(make_complex(*a, *b)),
            Instruction::LoadParam(i)       => stack.push(args[*i].clone()),
            Instruction::LoadCaptured(name) => {
                let v = captured.get(name.as_str())
                    .or_else(|| env.vars.get(name.as_str()))
                    .cloned()
                    .ok_or_else(|| format!("vm: undefined: {name}"))?;
                stack.push(v);
            }
            Instruction::BinOp(op) => {
                let rv = stack.pop().unwrap();
                let lv = stack.pop().unwrap();
                let result = if matches!((&lv, &rv),
                    (Val::Tensor { .. }, _) | (_, Val::Tensor { .. }) |
                    (Val::ComplexTensor { .. }, _) | (_, Val::ComplexTensor { .. }))
                {
                    binop_tensor(lv, op, rv)
                } else if matches!(op, Op::Eq | Op::Ne) {
                    if let (Val::Tuple(ls), Val::Tuple(rs)) = (&lv, &rv) {
                        let eq = ls.len() == rs.len()
                            && ls.iter().zip(rs.iter()).all(|(a, b)|
                                matches!((a, b), (Val::Num(x), Val::Num(y)) if x == y));
                        Ok(Val::Num(if matches!(op, Op::Eq) == eq { 1.0 } else { 0.0 }))
                    } else if matches!((&lv, &rv), (Val::Tuple(_), _) | (_, Val::Tuple(_))) {
                        binop_tuple(lv, op, rv, env)
                    } else {
                        scalar_binop(lv, op, rv)
                    }
                } else if matches!((&lv, &rv), (Val::Tuple(_), _) | (_, Val::Tuple(_))) {
                    binop_tuple(lv, op, rv, env)
                } else {
                    scalar_binop(lv, op, rv)
                }?;
                stack.push(result);
            }
            Instruction::Neg => {
                let v = stack.pop().unwrap();
                let result = match v {
                    Val::Num(n)       => Val::Num(-n),
                    Val::Complex(a,b) => make_complex(-a, -b),
                    Val::Tensor { data, shape } => Val::Tensor {
                        data: TData::new(data.into_iter().map(|x| -x).collect()),
                        shape,
                    },
                    Val::ComplexTensor { re, im, shape } => maybe_real(
                        re.into_iter().map(|x| -x).collect(),
                        im.into_iter().map(|x| -x).collect(),
                        shape,
                    ),
                    other => return Err(format!("vm neg: expected number, got {}", fmt_val(&other))),
                };
                stack.push(result);
            }
            Instruction::CallBuiltin(name, argc) => {
                let start = stack.len() - argc;
                let call_args: Vec<Val> = stack.drain(start..).collect();
                stack.push(eval_builtin(name, call_args, env)?);
            }
            Instruction::CallVal(argc) => {
                let start  = stack.len() - argc;
                let call_args: Vec<Val> = stack.drain(start..).collect();
                let callee = stack.pop().unwrap();
                stack.push(apply_val(callee, call_args, env)?);
            }
            Instruction::MakeTuple(n) => {
                let start = stack.len() - n;
                let items: Vec<Val> = stack.drain(start..).collect();
                let all_num = !items.is_empty() && items.iter().all(|v| matches!(v, Val::Num(_)));
                let all_cx  = !items.is_empty() && items.iter().all(|v| matches!(v, Val::Num(_) | Val::Complex(..)));
                let result = if all_num {
                    let data: Vec<f64> = items.into_iter()
                        .map(|v| match v { Val::Num(x) => x, _ => 0.0 }).collect();
                    let nn = data.len();
                    Val::Tensor { data: TData::new(data), shape: vec![nn] }
                } else if all_cx {
                    let mut re = Vec::with_capacity(*n);
                    let mut im = Vec::with_capacity(*n);
                    for v in items {
                        match v {
                            Val::Num(x)       => { re.push(x); im.push(0.0); }
                            Val::Complex(a,b) => { re.push(a); im.push(b); }
                            _ => {}
                        }
                    }
                    let nn = re.len();
                    maybe_real(re, im, vec![nn])
                } else {
                    Val::Tuple(items)
                };
                stack.push(result);
            }
            Instruction::MakeArray(n) => {
                let start = stack.len() - n;
                let items: Vec<Val> = stack.drain(start..).collect();
                if items.is_empty() {
                    stack.push(Val::Tensor { data: TData::new(vec![]), shape: vec![0] });
                } else {
                    let mut data: Vec<f64> = Vec::with_capacity(*n);
                    let mut re_d: Vec<f64> = Vec::with_capacity(*n);
                    let mut im_d: Vec<f64> = Vec::with_capacity(*n);
                    let mut has_cx = false;
                    for v in items {
                        match v {
                            Val::Num(x)       => { data.push(x); re_d.push(x); im_d.push(0.0); }
                            Val::Complex(a,b) => { has_cx = true; re_d.push(a); im_d.push(b); data.push(a); }
                            other => return Err(format!(
                                "vm []: requires numeric elements, got {}", fmt_val(&other)
                            )),
                        }
                    }
                    let nn = re_d.len();
                    stack.push(if has_cx {
                        maybe_real(re_d, im_d, vec![nn])
                    } else {
                        Val::Tensor { data: TData::new(data), shape: vec![nn] }
                    });
                }
            }
            Instruction::JumpIfFalse(target) => {
                let cond = stack.pop().unwrap().num("vm if")?;
                if cond == 0.0 { pc = *target; continue; }
            }
            Instruction::Jump(target)      => { pc = *target; continue; }
            Instruction::StoreLocal(slot)  => {
                let v = stack.pop().unwrap();
                if *slot == locals.len() { locals.push(v); } else { locals[*slot] = v; }
            }
            Instruction::LoadLocal(slot)   => stack.push(locals[*slot].clone()),
            Instruction::Pop               => { stack.pop(); }
            Instruction::Return            => break,
            Instruction::Index => {
                let idx_val = stack.pop().unwrap();
                let base    = stack.pop().unwrap();
                let result = match &base {
                    Val::Tensor { data, shape } => vm_tensor_index(data, shape, &idx_val),
                    Val::ComplexTensor { re, im, shape } => vm_complex_tensor_index(re, im, shape, &idx_val),
                    Val::Tuple(items) => {
                        let raw = idx_val.num("vm []")? as i64;
                        let i = norm_index(raw, items.len(), "tuple")?;
                        Ok(items[i].clone())
                    }
                    other => Err(format!("vm []: cannot index {}", fmt_val(other))),
                }?;
                stack.push(result);
            }
            Instruction::Loop(form, argc) => {
                // Operands were evaluated left-to-right onto the stack; pop them and
                // run the loop in the same Val-based core the tree-walk path uses, so
                // results and errors are identical — only the call frame is flat.
                let start = stack.len() - *argc;
                let call_args: Vec<Val> = stack.drain(start..).collect();
                let result = match form {
                    LoopForm::Sum     => agg_vals(call_args, env, false),
                    LoopForm::Prod    => agg_vals(call_args, env, true),
                    LoopForm::Iterate => iterate_vals(call_args, env),
                    LoopForm::Scan    => scan_vals(call_args, env),
                }?;
                stack.push(result);
            }
            Instruction::MakeClosure { params, body, code, free_vars } => {
                let start = stack.len() - free_vars.len();
                let vals: Vec<Val> = stack.drain(start..).collect();
                let mut new_captured = (**captured).clone();
                for (name, val) in free_vars.iter().zip(vals) {
                    new_captured.insert(name.clone(), val);
                }
                let new_cap = Arc::new(new_captured);
                let fn_val = if code.is_empty() {
                    Val::make_fn(params.clone(), (**body).clone(), new_cap)
                } else {
                    Val::make_fn_compiled(params.clone(), (**body).clone(), new_cap, (**code).clone())
                };
                stack.push(fn_val);
            }
        }
        pc += 1;
    }

    stack.pop().ok_or_else(|| "vm: empty stack".into())
}

// ── Value application ─────────────────────────────────────────────────────────

fn coerce_to_hint(val: Val, hint: &TypeHint, ctx: &str) -> Result<Val, String> {
    match hint {
        TypeHint::Any => Ok(val),
        TypeHint::Num => match &val {
            Val::Num(_) | Val::Complex(_, _) => Ok(val),
            other => Err(format!("{ctx}: expected num (scalar), got {}", fmt_val(other))),
        },
        TypeHint::Real => match val {
            Val::Num(_) => Ok(val),
            Val::Complex(r, i) => {
                if i.abs() < f64::EPSILON { Ok(Val::Num(r)) }
                else { Err(format!("{ctx}: expected real, got complex {r}+{i}i")) }
            }
            other => Err(format!("{ctx}: expected real number, got {}", fmt_val(&other))),
        },
        TypeHint::Complex => match val {
            Val::Num(_) | Val::Complex(_, _) => Ok(val),
            other => Err(format!("{ctx}: expected complex number, got {}", fmt_val(&other))),
        },
        TypeHint::Int => match val {
            Val::Num(x) if x.fract() == 0.0 => Ok(Val::Num(x)),
            Val::Num(x) => Err(format!("{ctx}: expected integer, got {x}")),
            Val::Complex(r, i) if i.abs() < f64::EPSILON && r.fract() == 0.0 => Ok(Val::Num(r)),
            Val::Complex(_, _) => Err(format!("{ctx}: expected integer, got complex")),
            other => Err(format!("{ctx}: expected integer, got {}", fmt_val(&other))),
        },
        TypeHint::Nat => match val {
            Val::Num(x) if x.fract() == 0.0 && x >= 0.0 => Ok(Val::Num(x)),
            Val::Num(x) => Err(format!("{ctx}: expected nat (non-negative integer), got {x}")),
            Val::Complex(r, i) if i.abs() < f64::EPSILON && r.fract() == 0.0 && r >= 0.0 => Ok(Val::Num(r)),
            Val::Complex(_, _) => Err(format!("{ctx}: expected nat, got complex")),
            other => Err(format!("{ctx}: expected nat, got {}", fmt_val(&other))),
        },
        TypeHint::Tensor => match &val {
            Val::Tensor { .. } | Val::ComplexTensor { .. } => Ok(val),
            other => Err(format!("{ctx}: expected tensor, got {}", fmt_val(other))),
        },
        TypeHint::RealTensor => match val {
            Val::Tensor { .. } => Ok(val),
            Val::ComplexTensor { re, im, shape } => {
                if im.iter().all(|&x| x.abs() < f64::EPSILON) {
                    Ok(Val::Tensor { data: re, shape })
                } else {
                    Err(format!("{ctx}: expected real tensor, got complex tensor"))
                }
            }
            other => Err(format!("{ctx}: expected real tensor, got {}", fmt_val(&other))),
        },
        TypeHint::ComplexTensor => match &val {
            Val::Tensor { .. } | Val::ComplexTensor { .. } => Ok(val),
            other => Err(format!("{ctx}: expected complex tensor, got {}", fmt_val(other))),
        },
        TypeHint::Fn => match &val {
            Val::Fn(..) | Val::Builtin(_) => Ok(val),
            other => Err(format!("{ctx}: expected function, got {}", fmt_val(other))),
        },
        TypeHint::Cell => match &val {
            Val::Cell(_) => Ok(val),
            other => Err(format!("{ctx}: expected cell, got {}", fmt_val(other))),
        },
        TypeHint::Tuple => match &val {
            Val::Tuple(_) => Ok(val),
            other => Err(format!("{ctx}: expected tuple, got {}", fmt_val(other))),
        },
    }
}

/// Run a user function: try bytecode VM first (compile on first call), fall back
/// to the tree-walk evaluator for any body the compiler cannot handle.
fn apply_fn_direct(
    params:   &[String],
    sig:      &FnSig,
    body:     &Expr,
    captured: &Arc<HashMap<String, Val>>,
    cache:    &Arc<OnceLock<Option<Vec<Instruction>>>>,
    args:     Vec<Val>,
    env:      &Env,
) -> Result<Val, String> {
    // Guard against runaway recursion (catchable error, not a stack overflow).
    let _depth = DepthGuard::enter()?;
    // Coerce args per param hints
    let args = if sig.params.is_empty() {
        args
    } else {
        args.into_iter().enumerate().map(|(i, v)| {
            if let Some(Some(hint)) = sig.params.get(i) {
                let ctx = params.get(i).map_or("param", |s| s.as_str());
                coerce_to_hint(v, hint, ctx)
            } else {
                Ok(v)
            }
        }).collect::<Result<Vec<_>, _>>()?
    };

    let code = cache.get_or_init(|| compile_fn(params, body, captured));
    let result = match code {
        Some(code) => run_vm(code, &args, captured, env),
        None => {
            let mut local = make_local(env, captured);
            for (p, v) in params.iter().zip(args) { local.define(p.clone(), v); }
            eval(body, &local)
        }
    }?;

    // Coerce return value
    if let Some(hint) = &sig.ret {
        coerce_to_hint(result, hint, "return value")
    } else {
        Ok(result)
    }
}

pub fn apply_val(f: Val, args: Vec<Val>, env: &Env) -> Result<Val, String> {
    match f {
        Val::Builtin(ref name) => eval_builtin(name, args, env),
        Val::Fn(ref params, ref body, ref captured, ref cache, ref sig) => {
            let n = params.len();
            let k = args.len();
            // BUG-6: a zero-arg call to an arity>0 function previously fell through
            // to the mapping branch and vacuously returned an empty tensor. Error
            // cleanly instead (the n>1 case already did; this closes n==1).
            if k == 0 && n > 0 {
                return Err(format!("function expects {n} arg(s), got 0"));
            }
            // All args are Fn → compose (only single arg supported)
            if k == 1 {
                if let Val::Fn(..) = &args[0] {
                    let g = args.into_iter().next().unwrap();
                    return Ok(compose_fns(f, g));
                }
                // Single n-element arg → destructure into n params.
                // Accepts: n-Tuple, or 1-D Tensor/ComplexTensor of n elements.
                let arg0 = &args[0];
                let destructured: Option<Vec<Val>> = match arg0 {
                    Val::Tuple(items) if items.len() == n => Some(items.clone()),
                    Val::Tensor { data, shape } if shape.len() == 1 && data.len() == n && n > 1
                        => Some(data.iter().map(|&x| Val::Num(x)).collect()),
                    Val::ComplexTensor { re, im, shape } if shape.len() == 1 && re.len() == n && n > 1
                        => Some(re.iter().zip(im.iter()).map(|(&r, &i)| make_complex(r, i)).collect()),
                    _ => None,
                };
                if let Some(items) = destructured {
                    return apply_fn_direct(params, sig, body, captured, cache, items, env);
                }
                // Single scalar/complex arg with 1-param fn → direct apply
                if n == 1 {
                    return apply_fn_direct(params, sig, body, captured, cache, args, env);
                }
                return Err(format!("function expects {n} args, got 1"));
            }
            // k == n: direct apply (catches the zero-arg case k==n==0 before the
            // vacuous all_n_seqs branch below would produce an empty tensor).
            if k == n {
                return apply_fn_direct(params, sig, body, captured, cache, args, env);
            }
            // k args, all n-element sequences → map with destructuring
            // Sequences can be n-Tuples or 1-D Tensors of size n
            let all_n_seqs = k > 0 && args.iter().all(|a| match a {
                Val::Tuple(v) => v.len() == n,
                Val::Tensor { data, shape } => shape.len() == 1 && data.len() == n,
                _ => false,
            });
            if all_n_seqs {
                let results: Result<Vec<Val>, _> = args.into_iter().map(|a| {
                    let items: Vec<Val> = match a {
                        Val::Tuple(v) => v,
                        Val::Tensor { data, .. } => data.into_iter().map(Val::Num).collect(),
                        _ => unreachable!(),
                    };
                    apply_fn_direct(params, sig, body, captured, cache, items, env)
                }).collect();
                // Promote result to Tensor if all-numeric
                let res = results?;
                let all_num = res.iter().all(|v| matches!(v, Val::Num(_)));
                let all_cx  = res.iter().all(|v| matches!(v, Val::Num(_) | Val::Complex(_, _)));
                return if all_num {
                    let data = res.into_iter().map(|v| match v { Val::Num(x) => x, _ => 0.0 }).collect::<Vec<_>>();
                    let nn = data.len();
                    Ok(Val::Tensor { data: TData::new(data), shape: vec![nn] })
                } else if all_cx {
                    let (re, im): (Vec<f64>, Vec<f64>) = res.into_iter().map(|v| match v {
                        Val::Num(x) => (x, 0.0), Val::Complex(a, b) => (a, b), _ => (0.0, 0.0)
                    }).unzip();
                    let nn = re.len();
                    Ok(maybe_real(re, im, vec![nn]))
                } else {
                    Ok(Val::Tuple(res))
                };
            }
            // k scalar args, 1-param fn → map → Tensor if all-numeric
            if n == 1 {
                let results: Result<Vec<Val>, _> = args.into_iter()
                    .map(|a| apply_fn_direct(params, sig, body, captured, cache, vec![a], env))
                    .collect();
                let res = results?;
                let all_num = res.iter().all(|v| matches!(v, Val::Num(_)));
                let all_cx  = res.iter().all(|v| matches!(v, Val::Num(_) | Val::Complex(_, _)));
                return if all_num {
                    let data = res.into_iter().map(|v| match v { Val::Num(x) => x, _ => 0.0 }).collect::<Vec<_>>();
                    let nn = data.len();
                    Ok(Val::Tensor { data: TData::new(data), shape: vec![nn] })
                } else if all_cx {
                    let (re, im): (Vec<f64>, Vec<f64>) = res.into_iter().map(|v| match v {
                        Val::Num(x) => (x, 0.0), Val::Complex(a, b) => (a, b), _ => (0.0, 0.0)
                    }).unzip();
                    let nn = re.len();
                    Ok(maybe_real(re, im, vec![nn]))
                } else {
                    Ok(Val::Tuple(res))
                };
            }
            Err(format!("function expects {n} args, got {k}"))
        }
        Val::Num(s) => {
            if args.len() == 1 {
                match &args[0] {
                    Val::Fn(..) => {
                        return Ok(scale_fn(s, args.into_iter().next().unwrap()));
                    }
                    Val::Num(n) => return Ok(Val::Num(s * n)),
                    Val::Tuple(items) => {
                        let scaled: Vec<Val> = items.iter().map(|v| match v {
                            Val::Num(n) => Val::Num(s * n),
                            _ => v.clone(),
                        }).collect();
                        return Ok(Val::Tuple(scaled));
                    }
                    Val::Tensor { data, shape } => {
                        let scaled = data.iter().map(|&x| s * x).collect();
                        return Ok(Val::Tensor { data: TData::new(scaled), shape: shape.clone() });
                    }
                    Val::Complex(a, b) => return Ok(make_complex(s * a, s * b)),
                    Val::ComplexTensor { re, im, shape } => {
                        let re_scaled: Vec<f64> = re.iter().map(|&x| s * x).collect();
                        let im_scaled: Vec<f64> = im.iter().map(|&x| s * x).collect();
                        return Ok(maybe_real(re_scaled, im_scaled, shape.clone()));
                    }
                    Val::Builtin(_) => return Err("cannot scale a builtin function".into()),
                    Val::Cell(..) => return Err("cannot scale a cell (use get/set)".into()),
                    Val::Namespace(..) => return Err("cannot scale a namespace".into()),
                    Val::Field(f) => {
                        let scaled = f.data.iter().map(|&x| s * x).collect();
                        return Ok(Val::Field(Arc::new(FieldVal { data: TData::new(scaled), ..(**f).clone() })));
                    }
                }
            }
            let nums: Result<Vec<f64>, _> = args.into_iter().map(|v| v.num("scalar-apply")).collect();
            Ok(Val::Num(nums?.iter().fold(s, |acc, n| acc * n)))
        }
        Val::Complex(a, b) => {
            if args.len() == 1 {
                let (ra, rb) = to_complex(args.into_iter().next().unwrap())?;
                return Ok(make_complex(a*ra - b*rb, a*rb + b*ra));
            }
            Err("complex: apply expects 1 arg".into())
        }
        Val::Tuple(items) => {
            if args.len() == 1 {
                let i = args.into_iter().next().unwrap().num("index")? as usize;
                return items.into_iter().nth(i).ok_or_else(|| format!("index {i} out of range"));
            }
            Err("tuple apply: expected a single index".into())
        }
        Val::Tensor { .. } => Err("tensors are not callable".into()),
        Val::ComplexTensor { .. } => Err("complex tensors are not callable".into()),
        Val::Cell(..) => Err("cells are not callable (use get/set)".into()),
        Val::Namespace(..) => Err("namespaces are not callable (use ns.member)".into()),
        Val::Field(..) => Err("fields are not callable".into()),
    }
}

// Three-layer env merge: global scope → closure's captured env → param bindings.
// Global scope provides forward-declared names; captured env provides lexical closure.
fn make_local(global: &Env, captured: &Arc<HashMap<String, Val>>) -> Env {
    let mut vars = (*global.vars).clone();
    vars.extend(captured.iter().map(|(k, v)| (k.clone(), v.clone())));
    Env { vars: Arc::new(vars) }
}

fn compose_fns(f: Val, g: Val) -> Val {
    let mut captured = HashMap::new();
    captured.insert("__f__".into(), f);
    captured.insert("__g__".into(), g);
    let body = Expr::Apply(
        Box::new(Expr::Var("__f__".into())),
        vec![Expr::Apply(
            Box::new(Expr::Var("__g__".into())),
            vec![Expr::Var("__z__".into())],
        )],
    );
    Val::make_fn(vec!["__z__".into()], body, Arc::new(captured))
}

fn scale_fn(s: f64, g: Val) -> Val {
    let mut captured = HashMap::new();
    captured.insert("__g__".into(), g);
    let body = Expr::BinOp(
        Box::new(Expr::Num(s)),
        Op::Mul,
        Box::new(Expr::Apply(
            Box::new(Expr::Var("__g__".into())),
            vec![Expr::Var("__z__".into())],
        )),
    );
    Val::make_fn(vec!["__z__".into()], body, Arc::new(captured))
}

fn binop_tuple(lv: Val, op: &Op, rv: Val, _env: &Env) -> Result<Val, String> {
    match (lv, rv) {
        (Val::Tuple(ls), Val::Tuple(rs)) => {
            if ls.len() != rs.len() {
                return Err(format!("tuple op tuple: length mismatch ({} vs {})", ls.len(), rs.len()));
            }
            let out: Result<Vec<Val>, _> = ls.into_iter().zip(rs)
                .map(|(l, r)| scalar_binop(l, op, r))
                .collect();
            Ok(Val::Tuple(out?))
        }
        (Val::Tuple(ls), scalar) => {
            let out: Result<Vec<Val>, _> = ls.into_iter()
                .map(|l| scalar_binop(l, op, scalar.clone()))
                .collect();
            Ok(Val::Tuple(out?))
        }
        (scalar, Val::Tuple(rs)) => {
            let out: Result<Vec<Val>, _> = rs.into_iter()
                .map(|r| scalar_binop(scalar.clone(), op, r))
                .collect();
            Ok(Val::Tuple(out?))
        }
        _ => unreachable!(),
    }
}

fn scalar_binop(lv: Val, op: &Op, rv: Val) -> Result<Val, String> {
    if let (Val::Num(la), Val::Num(ra)) = (&lv, &rv) {
        return Ok(Val::Num(match op {
            Op::Add      => la + ra,
            Op::Sub      => la - ra,
            Op::Mul      => la * ra,
            Op::Div      => la / ra,
            Op::FloorDiv => (*la / *ra).floor(),
            Op::Rem      => la % ra,
            Op::Pow      => la.powf(*ra),
            Op::Lt       => if la < ra  { 1.0 } else { 0.0 },
            Op::Gt       => if la > ra  { 1.0 } else { 0.0 },
            Op::LtEq     => if la <= ra { 1.0 } else { 0.0 },
            Op::GtEq     => if la >= ra { 1.0 } else { 0.0 },
            Op::Eq       => if la == ra { 1.0 } else { 0.0 },
            Op::Ne       => if la != ra { 1.0 } else { 0.0 },
            Op::And      => if int(*la) != 0 && int(*ra) != 0 { 1.0 } else { 0.0 },
            Op::Or       => if int(*la) != 0 || int(*ra) != 0 { 1.0 } else { 0.0 },
        }));
    }
    let (la, lb) = to_complex(lv)?;
    let (ra, rb) = to_complex(rv)?;
    match op {
        Op::Add      => Ok(make_complex(la + ra, lb + rb)),
        Op::Sub      => Ok(make_complex(la - ra, lb - rb)),
        Op::Mul      => Ok(make_complex(la*ra - lb*rb, la*rb + lb*ra)),
        Op::Div      => {
            let d = ra*ra + rb*rb;
            if d == 0.0 { return Err("division by zero".into()); }
            Ok(make_complex((la*ra + lb*rb)/d, (lb*ra - la*rb)/d))
        }
        Op::Pow      => Ok(complex_pow(la, lb, ra, rb)),
        Op::FloorDiv | Op::Rem => Err("// and % not defined for complex numbers".into()),
        Op::Eq       => Ok(Val::Num(if la == ra && lb == rb { 1.0 } else { 0.0 })),
        Op::Ne       => Ok(Val::Num(if la != ra || lb != rb { 1.0 } else { 0.0 })),
        Op::Lt | Op::Gt | Op::LtEq | Op::GtEq => Err("comparison not defined for complex numbers".into()),
        Op::And | Op::Or => Err("& and | not defined for complex numbers".into()),
    }
}

// ── Evaluator ─────────────────────────────────────────────────────────────────

pub fn eval(expr: &Expr, env: &Env) -> Result<Val, String> {
    match expr {
        Expr::Num(n)      => Ok(Val::Num(*n)),
        Expr::ImagLit(n)  => Ok(if *n == 0.0 { Val::Num(0.0) } else { Val::Complex(0.0, *n) }),
        Expr::Lambda(params, ret_hint, body) => {
            let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
            let sig = FnSig {
                params: params.iter().map(|p| p.hint.clone()).collect(),
                ret:    ret_hint.clone(),
            };
            Ok(Val::make_fn_with_sig(names, sig, *body.clone(), Arc::clone(&env.vars)))
        }
        Expr::Tuple(exprs) => {
            let vals: Result<Vec<Val>, _> = exprs.iter().map(|e| eval(e, env)).collect();
            let vals = vals?;
            // Auto-promote: all-numeric tuples → 1D Tensor / ComplexTensor.
            // Empty tuple stays as Tuple (unit value for empty blocks, etc.)
            if !vals.is_empty() {
                let all_numeric = vals.iter().all(|v| matches!(v, Val::Num(_) | Val::Complex(_, _)));
                if all_numeric {
                    let has_cx = vals.iter().any(|v| matches!(v, Val::Complex(_, _)));
                    let n = vals.len();
                    if has_cx {
                        let mut re = Vec::with_capacity(n);
                        let mut im = Vec::with_capacity(n);
                        for v in &vals {
                            match v {
                                Val::Num(x)        => { re.push(*x); im.push(0.0); }
                                Val::Complex(a, b) => { re.push(*a); im.push(*b); }
                                _ => unreachable!(),
                            }
                        }
                        return Ok(maybe_real(re, im, vec![n]));
                    } else {
                        let data: Vec<f64> = vals.into_iter().map(|v| match v { Val::Num(x) => x, _ => unreachable!() }).collect();
                        return Ok(Val::Tensor { data: TData::new(data), shape: vec![n] });
                    }
                }
            }
            Ok(Val::Tuple(vals))
        }
        Expr::Array(exprs) => {
            // [a, b, c] — explicit 1-D tensor literal.
            // All elements must evaluate to numbers; non-numeric values are an error.
            // Unlike (a,b,c) which auto-promotes, [] always means tensor.
            // [x]  → length-1 tensor;  [] → empty tensor.
            if exprs.is_empty() {
                return Ok(Val::Tensor { data: TData::new(vec![]), shape: vec![0] });
            }
            let mut data = Vec::with_capacity(exprs.len());
            let mut re_data: Vec<f64> = Vec::new();
            let mut im_data: Vec<f64> = Vec::new();
            let mut has_complex = false;
            for expr in exprs {
                match eval(expr, env)? {
                    Val::Num(x) => {
                        data.push(x);
                        re_data.push(x);
                        im_data.push(0.0);
                    }
                    Val::Complex(a, b) => {
                        has_complex = true;
                        re_data.push(a);
                        im_data.push(b);
                        data.push(a); // placeholder, unused if complex
                    }
                    other => return Err(format!(
                        "[] requires numeric elements; got {}. Use () for tuples.",
                        fmt_val(&other)
                    )),
                }
            }
            let n = re_data.len();
            if has_complex {
                Ok(maybe_real(re_data, im_data, vec![n]))
            } else {
                Ok(Val::Tensor { data: TData::new(data), shape: vec![n] })
            }
        }
        Expr::TensorLit(rows) => {
            if rows.is_empty() { return Ok(Val::Tensor { data: TData::new(vec![]), shape: vec![0, 0] }); }
            let r = rows.len();
            let c = rows[0].len();
            let mut data = Vec::with_capacity(r * c);
            for (ri, row) in rows.iter().enumerate() {
                if row.len() != c {
                    return Err(format!(
                        "tensor literal: row {} has {} elements, expected {}", ri, row.len(), c
                    ));
                }
                for expr in row {
                    data.push(eval(expr, env)?.num("tensor literal")?);
                }
            }
            Ok(Val::Tensor { data: TData::new(data), shape: vec![r, c] })
        }
        Expr::Slice(..) => Err("slice expression can only appear inside T[…]".into()),

        Expr::Index(expr, idx) => {
            let v = eval(expr, env)?;
            match v {
                // ── Tensor: shape-aware indexing, handles Expr::Slice ────────
                Val::Tensor { data, shape } => eval_tensor_index_ast(&data, &shape, idx, env),
                // ── ComplexTensor: same indexing, returns Complex or ComplexTensor ──
                Val::ComplexTensor { re, im, shape } => eval_complex_tensor_index_ast(&re, &im, &shape, idx, env),
                // ── Tuple: also supports slice expressions ────────────────────
                Val::Tuple(items) => eval_tuple_index_ast(items, idx, env),
                _ => Err("indexing requires a tuple or tensor".into()),
            }
        }
        Expr::Block(stmts) => {
            let mut child = env.clone();
            let mut last_val = Val::Tuple(vec![]);
            for stmt in stmts {
                match stmt {
                    BlockStmt::Def(def) => match def {
                        Def::Var(name, expr) => {
                            if is_protected(name) {
                                return Err(format!("cannot redefine built-in '{name}'"));
                            }
                            let v = eval(expr, &child)?;
                            child.define(name.clone(), v);
                        }
                        Def::Func(name, params, ret_hint, body) => {
                            if is_protected(name) {
                                return Err(format!("cannot redefine built-in '{name}'"));
                            }
                            let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                            let sig = FnSig {
                                params: params.iter().map(|p| p.hint.clone()).collect(),
                                ret:    ret_hint.clone(),
                            };
                            let mut captured = (*child.vars).clone();
                            let fn_val = Val::make_fn_with_sig(names.clone(), sig.clone(), body.clone(), Arc::new(captured.clone()));
                            captured.insert(name.clone(), fn_val);
                            child.define(name.clone(), Val::make_fn_with_sig(names, sig, body.clone(), Arc::new(captured)));
                        }
                    },
                    BlockStmt::Expr(e) => { last_val = eval(e, &child)?; }
                }
            }
            Ok(last_val)
        }
        Expr::Apply(f_expr, arg_exprs) => {
            // Special forms that need unevaluated Expr args (higher-order, still call_fn1-based)
            if let Expr::Var(name) = f_expr.as_ref() {
                match name.as_str() {
                    "if" => {
                        if arg_exprs.len() != 3 {
                            return Err("if(cond, a, b) expects 3 args".into());
                        }
                        let cond = eval(&arg_exprs[0], env)?.num("if")?;
                        return if cond != 0.0 { eval(&arg_exprs[1], env) }
                               else           { eval(&arg_exprs[2], env) };
                    }
                    "sum"      => return eval_agg(arg_exprs, env, false),
                    "prod"     => return eval_agg(arg_exprs, env, true),
                    "integral" => return eval_integral(arg_exprs, env),
                    "deriv"    => return eval_deriv(arg_exprs, env),
                    "iterate"  => return eval_iterate(arg_exprs, env),
                    "scan"     => return eval_scan(arg_exprs, env),
                    "map" => {
                        if arg_exprs.len() != 2 {
                            return Err("map(f, tuple) expects 2 args".into());
                        }
                        // Evaluate f once so the bytecode cache is shared across all calls.
                        let f_val  = eval(&arg_exprs[0], env)?;
                        let second = eval(&arg_exprs[1], env)?;
                        return match second {
                            Val::Tuple(items) => {
                                let results: Result<Vec<Val>, _> = items.into_iter()
                                    .map(|item| apply_val(f_val.clone(), vec![item], env))
                                    .collect();
                                Ok(Val::Tuple(results?))
                            }
                            Val::Tensor { data, shape } => {
                                let mut re_out = Vec::with_capacity(data.len());
                                let mut im_out = Vec::with_capacity(data.len());
                                let mut has_complex = false;
                                for x in data {
                                    let v = apply_val(f_val.clone(), vec![Val::Num(x)], env)?;
                                    match v {
                                        Val::Num(n)        => { re_out.push(n); im_out.push(0.0); }
                                        Val::Complex(a, b) => { re_out.push(a); im_out.push(b); has_complex = true; }
                                        other => return Err(format!("map: f must return a number or complex, got {}", fmt_val(&other))),
                                    }
                                }
                                if has_complex {
                                    Ok(maybe_real(re_out, im_out, shape))
                                } else {
                                    Ok(Val::Tensor { data: TData::new(re_out), shape })
                                }
                            }
                            Val::ComplexTensor { re, im, shape } => {
                                broadcast1(Val::ComplexTensor { re, im, shape }, |v| apply_val(f_val.clone(), vec![v], env))
                            }
                            other => Err(format!("map: second arg must be a tuple or tensor, got {}", fmt_val(&other))),
                        };
                    }
                    "filter" => {
                        if arg_exprs.len() != 2 {
                            return Err("filter(f, seq) expects 2 args".into());
                        }
                        let f_val = eval(&arg_exprs[0], env)?;
                        return match eval(&arg_exprs[1], env)? {
                            Val::Tensor { data, .. } => {
                                let mut out = vec![];
                                for x in data {
                                    let keep = apply_val(f_val.clone(), vec![Val::Num(x)], env)?.num("filter")?;
                                    if keep != 0.0 { out.push(x); }
                                }
                                let n = out.len();
                                Ok(Val::Tensor { data: TData::new(out), shape: vec![n] })
                            }
                            Val::Tuple(items) => {
                                let mut out = vec![];
                                for item in items {
                                    let keep = apply_val(f_val.clone(), vec![item.clone()], env)?.num("filter")?;
                                    if keep != 0.0 { out.push(item); }
                                }
                                Ok(Val::Tuple(out))
                            }
                            other => Err(format!("filter: second arg must be a tensor or tuple, got {}", fmt_val(&other))),
                        };
                    }
                    "reduce" => {
                        if arg_exprs.len() != 2 {
                            return Err("reduce(f, tuple) expects 2 args".into());
                        }
                        let second = eval(&arg_exprs[1], env)?;
                        let (first_item, rest): (Val, Box<dyn Iterator<Item=Val>>) = match second {
                            Val::Tuple(v) => {
                                if v.is_empty() { return Err("reduce: empty tuple".into()); }
                                let mut it = v.into_iter();
                                let first = it.next().unwrap();
                                (first, Box::new(it))
                            }
                            Val::Tensor { data, .. } => {
                                if data.is_empty() { return Err("reduce: empty tensor".into()); }
                                let mut it = data.into_iter().map(Val::Num);
                                let first = it.next().unwrap();
                                (first, Box::new(it))
                            }
                            other => return Err(format!("reduce: second arg must be a tuple or tensor, got {}", fmt_val(&other))),
                        };
                        let f_val = eval(&arg_exprs[0], env)?;
                        let mut acc = first_item;
                        for item in rest {
                            acc = apply_val(f_val.clone(), vec![acc, item], env)?;
                        }
                        return Ok(acc);
                    }
                    _ => {}
                }
            }
            let f_val = eval(f_expr, env)?;
            let args: Result<Vec<Val>, _> = arg_exprs.iter().map(|a| eval(a, env)).collect();
            apply_val(f_val, args?, env)
        }
        Expr::Range(start, end) => {
            let a = eval(start, env)?.num("range")? as i64;
            let b = eval(end, env)?.num("range")? as i64;
            let items: Vec<Val> = if a <= b {
                (a..=b).map(|n| Val::Num(n as f64)).collect()
            } else {
                (b..=a).rev().map(|n| Val::Num(n as f64)).collect()
            };
            Ok(Val::Tuple(items))
        }
        Expr::Var(n) => env.vars.get(n).cloned()
            .ok_or_else(|| format!("undefined: {n}")),
        Expr::Member(base, field) => {
            let base_val = eval(base, env)?;
            match base_val {
                Val::Namespace(map) => map.get(field).cloned()
                    .ok_or_else(|| {
                        let ns = match base.as_ref() { Expr::Var(n) => n.as_str(), _ => "namespace" };
                        format!("{ns} has no member '{field}'")
                    }),
                other => Err(format!("'.{field}': expected a namespace, got {}", fmt_val(&other))),
            }
        }
        Expr::Neg(e) => match eval(e, env)? {
            Val::Num(n)        => Ok(Val::Num(-n)),
            Val::Complex(a, b) => Ok(make_complex(-a, -b)),
            Val::Tuple(items)  => {
                let neg: Result<Vec<Val>, _> = items.into_iter()
                    .map(|v| match v {
                        Val::Num(n) => Ok(Val::Num(-n)),
                        Val::Complex(a, b) => Ok(make_complex(-a, -b)),
                        other => Err(format!("unary minus: expected a number, got {}", fmt_val(&other))),
                    }).collect();
                Ok(Val::Tuple(neg?))
            }
            Val::Tensor { data, shape } => {
                Ok(Val::Tensor { data: TData::new(data.into_iter().map(|x| -x).collect()), shape })
            }
            Val::ComplexTensor { re, im, shape } => {
                Ok(maybe_real(re.into_iter().map(|x| -x).collect(),
                              im.into_iter().map(|x| -x).collect(),
                              shape))
            }
            Val::Field(f) => {
                let neg = f.data.iter().map(|&x| -x).collect();
                Ok(Val::Field(Arc::new(FieldVal { data: TData::new(neg), ..(*f).clone() })))
            }
            Val::Fn(..) | Val::Builtin(_) | Val::Cell(..) | Val::Namespace(..) => Err("unary minus: expected a number".into()),
        },
        Expr::Not(e) => {
            #[inline] fn lnot(x: f64) -> f64 { (int(x) == 0) as i64 as f64 }
            match eval(e, env)? {
                Val::Num(n) => Ok(Val::Num(lnot(n))),
                Val::Tuple(items) => {
                    let r: Result<Vec<Val>, _> = items.into_iter().map(|v| match v {
                        Val::Num(n) => Ok(Val::Num(lnot(n))),
                        other => Err(format!("~: expected a number, got {}", fmt_val(&other))),
                    }).collect();
                    Ok(Val::Tuple(r?))
                }
                Val::Tensor { data, shape } => {
                    Ok(Val::Tensor { data: TData::new(data.into_iter().map(lnot).collect()), shape })
                }
                Val::Field(f) => {
                    let d = f.data.iter().map(|&x| lnot(x)).collect();
                    Ok(Val::Field(Arc::new(FieldVal { data: TData::new(d), ..(*f).clone() })))
                }
                other => Err(format!("~: expected a number, got {}", fmt_val(&other))),
            }
        }
        Expr::BinOp(l, op, r) => {
            let lv = eval(l, env)?;
            let rv = eval(r, env)?;
            // Field arithmetic: operate on component data, carrying geometry.
            if matches!((&lv, &rv), (Val::Field(_), _) | (_, Val::Field(_))) {
                return field_binop(lv, op, rv);
            }
            // Tensor/ComplexTensor dispatch (before tuple, since Tensor is not a Tuple)
            if matches!((&lv, &rv),
                (Val::Tensor { .. }, _) | (_, Val::Tensor { .. }) |
                (Val::ComplexTensor { .. }, _) | (_, Val::ComplexTensor { .. }))
            {
                return binop_tensor(lv, op, rv);
            }
            // Whole-tuple equality/inequality before element-wise broadcast
            if matches!(op, Op::Eq | Op::Ne) {
                if let (Val::Tuple(ls), Val::Tuple(rs)) = (&lv, &rv) {
                    let equal = ls.len() == rs.len() &&
                        ls.iter().zip(rs.iter()).all(|(a, b)| {
                            matches!((a, b), (Val::Num(x), Val::Num(y)) if x == y)
                        });
                    return Ok(Val::Num(if matches!(op, Op::Eq) == equal { 1.0 } else { 0.0 }));
                }
            }
            if matches!((&lv, &rv), (Val::Tuple(_), _) | (_, Val::Tuple(_))) {
                return binop_tuple(lv, op, rv, env);
            }
            if let (Val::Num(la), Val::Num(ra)) = (&lv, &rv) {
                return Ok(Val::Num(match op {
                    Op::Add      => la + ra,
                    Op::Sub      => la - ra,
                    Op::Mul      => la * ra,
                    Op::Div      => la / ra,
                    Op::FloorDiv => (*la / *ra).floor(),
                    Op::Rem      => la % ra,
                    Op::Pow      => la.powf(*ra),
                    Op::Lt       => if la < ra  { 1.0 } else { 0.0 },
                    Op::Gt       => if la > ra  { 1.0 } else { 0.0 },
                    Op::LtEq     => if la <= ra { 1.0 } else { 0.0 },
                    Op::GtEq     => if la >= ra { 1.0 } else { 0.0 },
                    Op::Eq       => if la == ra { 1.0 } else { 0.0 },
                    Op::Ne       => if la != ra { 1.0 } else { 0.0 },
                    Op::And      => if int(*la) != 0 && int(*ra) != 0 { 1.0 } else { 0.0 },
                    Op::Or       => if int(*la) != 0 || int(*ra) != 0 { 1.0 } else { 0.0 },
                }));
            }
            let (la, lb) = to_complex(lv)?;
            let (ra, rb) = to_complex(rv)?;
            match op {
                Op::Add      => Ok(make_complex(la + ra, lb + rb)),
                Op::Sub      => Ok(make_complex(la - ra, lb - rb)),
                Op::Mul      => Ok(make_complex(la*ra - lb*rb, la*rb + lb*ra)),
                Op::Div      => {
                    let d = ra*ra + rb*rb;
                    if d == 0.0 { return Err("division by zero".into()); }
                    Ok(make_complex((la*ra + lb*rb)/d, (lb*ra - la*rb)/d))
                }
                Op::Pow      => Ok(complex_pow(la, lb, ra, rb)),
                Op::FloorDiv | Op::Rem => Err("// and % not defined for complex numbers".into()),
                Op::Eq       => Ok(Val::Num(if la == ra && lb == rb { 1.0 } else { 0.0 })),
                Op::Ne       => Ok(Val::Num(if la != ra || lb != rb { 1.0 } else { 0.0 })),
                Op::Lt | Op::Gt | Op::LtEq | Op::GtEq => Err("comparison not defined for complex numbers".into()),
                Op::And | Op::Or => Err("& and | not defined for complex numbers".into()),
            }
        }
    }
}

// ── Index resolution ──────────────────────────────────────────────────────────

/// Tuple indexing: supports slices, multi-index selects, and plain scalar.
fn eval_tuple_index_ast(items: Vec<Val>, idx: &Expr, env: &Env) -> Result<Val, String> {
    let dim = items.len();
    // Determine whether any slice form is involved.
    let is_slice = matches!(idx, Expr::Slice(..))
        || matches!(idx, Expr::Tuple(es) if es.iter().any(|e| matches!(e, Expr::Slice(..))));

    if is_slice {
        // Route through slice resolver (treat tuple as 1-D array)
        let idx_items: Vec<&Expr> = match idx {
            Expr::Tuple(es) => {
                if es.len() != 1 {
                    return Err(format!("tuple is 1-D: expected 1 index, got {}", es.len()));
                }
                vec![&es[0]]
            }
            single => vec![single],
        };
        let (is_range, selected) = resolve_index_item(idx_items[0], dim, 0, env)?;
        if !is_range {
            // Single element (scalar result from a degenerate slice)
            items.into_iter().nth(selected[0])
                .ok_or_else(|| "index out of range".into())
        } else if selected.len() == 1 {
            items.into_iter().nth(selected[0])
                .ok_or_else(|| "index out of range".into())
        } else {
            Ok(Val::Tuple(
                selected.iter()
                    .filter_map(|&i| items.get(i).cloned())
                    .collect()
            ))
        }
    } else {
        // Original behaviour: evaluate index, match Num or Tuple(of scalars)
        let idx_val = eval(idx, env)?;
        match idx_val {
            Val::Num(n) => {
                let i = norm_index(n as i64, dim, "tuple")?;
                items.into_iter().nth(i).ok_or_else(|| format!("index {n} out of range"))
            }
            Val::Tuple(indices) => {
                let result: Result<Vec<Val>, _> = indices.into_iter().map(|iv| {
                    let raw = iv.num("index")? as i64;
                    let i = norm_index(raw, dim, "tuple")?;
                    items.get(i).cloned().ok_or_else(|| format!("index {raw} out of range"))
                }).collect();
                Ok(Val::Tuple(result?))
            }
            other => Err(format!("tuple index must be a number, got {}", fmt_val(&other))),
        }
    }
}

/// Shape-aware tensor indexing that handles Expr::Slice.
/// `idx` is the raw index expression: either a single item or Expr::Tuple of items.
fn eval_tensor_index_ast(data: &[f64], shape: &[usize], idx: &Expr, env: &Env) -> Result<Val, String> {
    // Flatten index expressions into a list of items per dimension.
    let items: Vec<&Expr> = match idx {
        Expr::Tuple(es) => es.iter().collect(),
        single          => vec![single],
    };

    // Single scalar index `T[n]` — return row/sub-tensor (no slice check needed).
    if items.len() == 1 && !matches!(items[0], Expr::Slice(..)) {
        let idx_val = eval(items[0], env)?;
        let raw = idx_val.num("tensor index")? as i64;
        let i = norm_index(raw, shape[0], "tensor")?;
        return if shape.len() == 1 {
            Ok(Val::Num(data[i]))
        } else {
            let sub_size: usize = shape[1..].iter().product();
            let start = i * sub_size;
            Ok(Val::Tensor {
                data: TData::new(data[start..start + sub_size].to_vec()),
                shape: shape[1..].to_vec(),
            })
        };
    }

    // Multi-dim or slice indexing: must match ndim exactly.
    if items.len() != shape.len() {
        return Err(format!("tensor: expected {} index/slice items, got {}", shape.len(), items.len()));
    }

    // Resolve each item into (is_range, selected_indices).
    let resolved: Vec<(bool, Vec<usize>)> = items.iter().zip(shape.iter()).enumerate()
        .map(|(k, (item, &dim))| resolve_index_item(item, dim, k, env))
        .collect::<Result<_, _>>()?;

    // Output shape: keep dims whose index is a range/slice.
    let out_shape: Vec<usize> = resolved.iter()
        .filter(|(keep, _)| *keep)
        .map(|(_, idxs)| idxs.len())
        .collect();

    let range_sizes: Vec<usize> = resolved.iter().map(|(_, idxs)| idxs.len()).collect();
    let total: usize = range_sizes.iter().product();
    let in_strides = strides(shape);
    let mut out_data = Vec::with_capacity(total);

    for out_flat in 0..total {
        let combo = unravel(out_flat, &range_sizes);
        let in_flat: usize = combo.iter().zip(&resolved).zip(&in_strides)
            .map(|((&ci, (_, idxs)), &stride)| idxs[ci] * stride)
            .sum();
        out_data.push(data[in_flat]);
    }

    if out_shape.is_empty() {
        Ok(Val::Num(out_data[0]))
    } else {
        Ok(Val::Tensor { data: TData::new(out_data), shape: out_shape })
    }
}

/// Same as eval_tensor_index_ast but for ComplexTensor (re, im parallel arrays).
/// Returns Complex for scalar result, or ComplexTensor (possibly collapsed to Tensor) for slices.
fn eval_complex_tensor_index_ast(re: &[f64], im: &[f64], shape: &[usize], idx: &Expr, env: &Env) -> Result<Val, String> {
    let items: Vec<&Expr> = match idx {
        Expr::Tuple(es) => es.iter().collect(),
        single          => vec![single],
    };

    // Single scalar index `CT[n]` — return row/sub-tensor.
    if items.len() == 1 && !matches!(items[0], Expr::Slice(..)) {
        let idx_val = eval(items[0], env)?;
        let raw = idx_val.num("tensor index")? as i64;
        let i = norm_index(raw, shape[0], "tensor")?;
        return if shape.len() == 1 {
            Ok(make_complex(re[i], im[i]))
        } else {
            let sub_size: usize = shape[1..].iter().product();
            let start = i * sub_size;
            Ok(maybe_real(
                re[start..start + sub_size].to_vec(),
                im[start..start + sub_size].to_vec(),
                shape[1..].to_vec(),
            ))
        };
    }

    // Multi-dim or slice indexing: must match ndim exactly.
    if items.len() != shape.len() {
        return Err(format!("tensor: expected {} index/slice items, got {}", shape.len(), items.len()));
    }

    let resolved: Vec<(bool, Vec<usize>)> = items.iter().zip(shape.iter()).enumerate()
        .map(|(k, (item, &dim))| resolve_index_item(item, dim, k, env))
        .collect::<Result<_, _>>()?;

    let out_shape: Vec<usize> = resolved.iter()
        .filter(|(keep, _)| *keep)
        .map(|(_, idxs)| idxs.len())
        .collect();

    let range_sizes: Vec<usize> = resolved.iter().map(|(_, idxs)| idxs.len()).collect();
    let total: usize = range_sizes.iter().product();
    let in_strides = strides(shape);
    let mut out_re = Vec::with_capacity(total);
    let mut out_im = Vec::with_capacity(total);

    for out_flat in 0..total {
        let combo = unravel(out_flat, &range_sizes);
        let in_flat: usize = combo.iter().zip(&resolved).zip(&in_strides)
            .map(|((&ci, (_, idxs)), &stride)| idxs[ci] * stride)
            .sum();
        out_re.push(re[in_flat]);
        out_im.push(im[in_flat]);
    }

    if out_shape.is_empty() {
        Ok(make_complex(out_re[0], out_im[0]))
    } else {
        Ok(maybe_real(out_re, out_im, out_shape))
    }
}

/// Resolve one index item (Expr::Slice or a scalar/range expression) for a dimension of given size.
/// Returns (is_range, selected_indices).
///   is_range = true  → this dimension is kept in the output
///   is_range = false → this dimension is collapsed (scalar index)
fn resolve_index_item(item: &Expr, dim: usize, k: usize, env: &Env) -> Result<(bool, Vec<usize>), String> {
    // Resolve a signed scalar index to [0, dim), erroring on out-of-range either way.
    let clamp = |raw: i64| -> Result<usize, String> {
        let i = if raw < 0 { raw + dim as i64 } else { raw };
        if i < 0 || i >= dim as i64 { Err(format!("index {raw} out of range for dim {k} (size={dim})")) }
        else { Ok(i as usize) }
    };
    match item {
        // T[..] — all indices
        Expr::Slice(None, None) => Ok((true, (0..dim).collect())),
        // T[lo..] — from lo to end
        Expr::Slice(Some(lo_expr), None) => {
            let lo = eval(lo_expr, env)?.num("slice lo")? as i64;
            let lo = if lo < 0 { (dim as i64 + lo).max(0) as usize } else { lo as usize };
            Ok((true, (lo..dim).collect()))
        }
        // T[..hi] — from start to hi (inclusive)
        Expr::Slice(None, Some(hi_expr)) => {
            let hi = eval(hi_expr, env)?.num("slice hi")? as i64;
            let hi = clamp(hi)?;
            Ok((true, (0..=hi).collect()))
        }
        // T[lo..hi] — bounded slice (inclusive on both ends)
        Expr::Slice(Some(lo_expr), Some(hi_expr)) => {
            let lo = eval(lo_expr, env)?.num("slice lo")? as i64;
            let hi = eval(hi_expr, env)?.num("slice hi")? as i64;
            let lo = if lo < 0 { (dim as i64 + lo).max(0) as usize } else { lo as usize };
            let hi = clamp(hi)?;
            if lo > hi { return Ok((true, vec![])); }
            Ok((true, (lo..=hi).collect()))
        }
        // Anything else: evaluate, must be a scalar index.
        other => {
            let val = eval(other, env)?;
            let raw = val.num("tensor index")? as i64;
            let i = clamp(raw)?;
            Ok((false, vec![i]))
        }
    }
}

// Front-end: evaluate the argument expressions (left-to-right, so cell side
// effects sequence as written) and hand off to the Val-based core. The VM's
// `Loop` instruction calls `agg_vals` directly with operands already on the stack.
pub fn eval_agg(args: &[Expr], env: &Env, product: bool) -> Result<Val, String> {
    let vals = args.iter().map(|a| eval(a, env)).collect::<Result<Vec<_>, _>>()?;
    agg_vals(vals, env, product)
}

pub fn agg_vals(args: Vec<Val>, env: &Env, product: bool) -> Result<Val, String> {
    let label = if product { "prod" } else { "sum" };

    /// Accumulate (acc_re, acc_im) with (r, i), either summing or multiplying.
    fn accum(acc_re: &mut f64, acc_im: &mut f64, r: f64, i: f64, product: bool) {
        if product {
            // (acc_re + i*acc_im) * (r + i*i) = (acc_re*r - acc_im*i) + i*(acc_re*i + acc_im*r)
            let new_re = *acc_re * r - *acc_im * i;
            let new_im = *acc_re * i + *acc_im * r;
            *acc_re = new_re;
            *acc_im = new_im;
        } else {
            *acc_re += r;
            *acc_im += i;
        }
    }

    // 1-arg form: sum(tuple), sum(tensor), sum(ComplexTensor)
    if args.len() == 1 {
        return match args.into_iter().next().unwrap() {
            Val::Tuple(items) => {
                let mut acc_re = if product { 1.0 } else { 0.0 };
                let mut acc_im = 0.0;
                for v in items {
                    let (r, i) = to_complex(v)?;
                    accum(&mut acc_re, &mut acc_im, r, i, product);
                }
                Ok(make_complex(acc_re, acc_im))
            }
            Val::Tensor { data, .. } => {
                let acc: f64 = if product { data.iter().product() } else { data.iter().sum() };
                Ok(Val::Num(acc))
            }
            Val::ComplexTensor { re, im, .. } => {
                let (init_re, init_im) = if product { (1.0, 0.0) } else { (0.0, 0.0) };
                let mut acc_re = init_re;
                let mut acc_im = init_im;
                for (&r, &i) in re.iter().zip(im.iter()) {
                    accum(&mut acc_re, &mut acc_im, r, i, product);
                }
                Ok(make_complex(acc_re, acc_im))
            }
            _ => Err(format!("{label}: 1-arg form requires a tuple or tensor")),
        };
    }
    // 2-arg form: sum(T, axis), sum(CT, axis), or sum(f, n)
    if args.len() == 2 {
        let mut it = args.into_iter();
        let v0 = it.next().unwrap();
        let v1 = it.next().unwrap();
        return match v0 {
            // sum(T, axis) — reduce real tensor along one axis
            Val::Tensor { data, shape } => {
                let axis = v1.num(label)? as usize;
                if axis >= shape.len() {
                    return Err(format!("{label}: axis {axis} out of range for {}-D tensor", shape.len()));
                }
                let mut out_shape = shape.clone();
                out_shape.remove(axis);
                let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
                let init = if product { 1.0 } else { 0.0 };
                let mut out_data = vec![init; out_size];
                let out_strides: Vec<usize> = strides(&out_shape);
                for in_flat in 0..data.len() {
                    let multi = unravel(in_flat, &shape);
                    let out_multi: Vec<usize> = multi.iter().enumerate()
                        .filter(|&(k, _)| k != axis)
                        .map(|(_, &i)| i)
                        .collect();
                    let out_flat: usize = if out_multi.is_empty() {
                        0
                    } else {
                        out_multi.iter().zip(&out_strides).map(|(&i, &s)| i * s).sum()
                    };
                    if product { out_data[out_flat] *= data[in_flat]; }
                    else       { out_data[out_flat] += data[in_flat]; }
                }
                if out_shape.is_empty() {
                    Ok(Val::Num(out_data[0]))
                } else {
                    Ok(Val::Tensor { data: TData::new(out_data), shape: out_shape })
                }
            }
            // sum(CT, axis) — reduce complex tensor along one axis
            Val::ComplexTensor { re, im, shape } => {
                let axis = v1.num(label)? as usize;
                if axis >= shape.len() {
                    return Err(format!("{label}: axis {axis} out of range for {}-D tensor", shape.len()));
                }
                let mut out_shape = shape.clone();
                out_shape.remove(axis);
                let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
                let (init_re, init_im) = if product { (1.0, 0.0) } else { (0.0, 0.0) };
                let mut out_re = vec![init_re; out_size];
                let mut out_im = vec![init_im; out_size];
                let out_strides: Vec<usize> = strides(&out_shape);
                for in_flat in 0..re.len() {
                    let multi = unravel(in_flat, &shape);
                    let out_multi: Vec<usize> = multi.iter().enumerate()
                        .filter(|&(k, _)| k != axis)
                        .map(|(_, &i)| i)
                        .collect();
                    let out_flat: usize = if out_multi.is_empty() {
                        0
                    } else {
                        out_multi.iter().zip(&out_strides).map(|(&i, &s)| i * s).sum()
                    };
                    accum(&mut out_re[out_flat], &mut out_im[out_flat], re[in_flat], im[in_flat], product);
                }
                if out_shape.is_empty() {
                    Ok(make_complex(out_re[0], out_im[0]))
                } else {
                    Ok(maybe_real(out_re, out_im, out_shape))
                }
            }
            // sum(f, n) — sum f(k) for k in 0..n; f may return complex
            f @ (Val::Fn(..) | Val::Builtin(_)) => {
                let n = v1.num(label)? as i64;
                if n < 0 {
                    return Err(format!("{label}: count must be non-negative, got {n}"));
                }
                let (init_re, init_im) = if product { (1.0, 0.0) } else { (0.0, 0.0) };
                let mut acc_re = init_re;
                let mut acc_im = init_im;
                for k in 0..n {
                    let v = apply_val(f.clone(), vec![Val::Num(k as f64)], env)?;
                    let (r, i) = to_complex(v).map_err(|_| format!("{label}: f must return a number or complex"))?;
                    accum(&mut acc_re, &mut acc_im, r, i, product);
                }
                Ok(make_complex(acc_re, acc_im))
            }
            _ => Err(format!("{label}: 2-arg form is {label}(T, axis) or {label}(f, n)")),
        };
    }
    if args.len() != 3 {
        return Err(format!("{label} expects {label}(T), {label}(T,axis), {label}(f,n), or {label}(f,lo,hi)"));
    }
    // f is already evaluated once (shared bytecode cache across the summation loop).
    let mut it = args.into_iter();
    let f_val = it.next().unwrap();
    let start = it.next().unwrap().num("start")? as i64;
    let stop  = it.next().unwrap().num("stop")?  as i64;
    let (init_re, init_im) = if product { (1.0, 0.0) } else { (0.0, 0.0) };
    let mut acc_re = init_re;
    let mut acc_im = init_im;
    for k in start..=stop {
        let v = apply_val(f_val.clone(), vec![Val::Num(k as f64)], env)?;
        let (r, i) = to_complex(v).map_err(|_| format!("{label}: f must return a number or complex"))?;
        accum(&mut acc_re, &mut acc_im, r, i, product);
    }
    Ok(make_complex(acc_re, acc_im))
}

// iterate(f, x0, n) — apply f to x0 n times (f^n(x0)). Flat loop, O(1) stack:
// the safe, scalable replacement for the recursive evolve()/step() idiom.
pub fn eval_iterate(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 3 {
        return Err("iterate(f, x0, n) expects 3 args".into());
    }
    // Evaluate f once so the bytecode cache is shared across the whole loop.
    let vals = args.iter().map(|a| eval(a, env)).collect::<Result<Vec<_>, _>>()?;
    iterate_vals(vals, env)
}

// Val-based core, shared by the tree-walk front-end and the VM `Loop` instruction.
pub fn iterate_vals(args: Vec<Val>, env: &Env) -> Result<Val, String> {
    if args.len() != 3 {
        return Err("iterate(f, x0, n) expects 3 args".into());
    }
    let mut it = args.into_iter();
    let f_val = it.next().unwrap();
    let mut state = it.next().unwrap();
    let n = it.next().unwrap().num("iterate")? as i64;
    if n < 0 {
        return Err(format!("iterate: count must be non-negative, got {n}"));
    }
    for step in 0..n {
        state = apply_val(f_val.clone(), vec![state], env)?;
        if !state_is_finite(&state) {
            return Err(format!("iterate: non-finite value (NaN/Inf) at step {}", step + 1));
        }
    }
    Ok(state)
}

/// True unless the value contains a NaN/Inf. Used by iterate/scan to abort a
/// diverging time-step with the step index instead of silently returning NaN.
pub(crate) fn state_is_finite(v: &Val) -> bool {
    match v {
        Val::Num(x)                   => x.is_finite(),
        Val::Complex(a, b)            => a.is_finite() && b.is_finite(),
        Val::Tensor { data, .. }      => data.iter().all(|x| x.is_finite()),
        Val::ComplexTensor { re, im, .. } => re.iter().all(|x| x.is_finite()) && im.iter().all(|x| x.is_finite()),
        Val::Tuple(items)             => items.iter().all(state_is_finite),
        _                             => true,
    }
}

// scan(f, x0, n) — the whole orbit [x0, f(x0), …, f^n(x0)] stacked into a tensor.
// Scalar states → a 1-D tensor of length n+1; vector states (length d) → a 2-D
// tensor [n+1, d] with each state as a row. Flat loop, O(1) stack besides output.
pub fn eval_scan(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() != 3 {
        return Err("scan(f, x0, n) expects 3 args".into());
    }
    let vals = args.iter().map(|a| eval(a, env)).collect::<Result<Vec<_>, _>>()?;
    scan_vals(vals, env)
}

// Val-based core, shared by the tree-walk front-end and the VM `Loop` instruction.
pub fn scan_vals(args: Vec<Val>, env: &Env) -> Result<Val, String> {
    if args.len() != 3 {
        return Err("scan(f, x0, n) expects 3 args".into());
    }
    let mut it = args.into_iter();
    let f_val = it.next().unwrap();
    let mut state = it.next().unwrap();
    let n = it.next().unwrap().num("scan")? as i64;
    if n < 0 {
        return Err(format!("scan: count must be non-negative, got {n}"));
    }
    let mut states: Vec<Val> = Vec::with_capacity(n as usize + 1);
    states.push(state.clone());
    for step in 0..n {
        state = apply_val(f_val.clone(), vec![state], env)?;
        if !state_is_finite(&state) {
            return Err(format!("scan: non-finite value (NaN/Inf) at step {}", step + 1));
        }
        states.push(state.clone());
    }
    stack_rows(states, "scan")
}

// Stack a list of states as the rows of a tensor (used by scan, and the engine
// behind a vector trajectory). All-scalar states → a 1-D tensor [k]; equal-length
// 1-D vectors / flat numeric tuples → a 2-D tensor [k, d]. A *structured* tuple
// (one whose elements are themselves vectors/tensors, e.g. phase-space (q, p) with
// vector q, p) is stacked component-wise: each field is scanned independently and
// the result is a tuple of stacks, (Q, P). Real fast path; complex via maybe_real.
fn stack_rows(states: Vec<Val>, who: &str) -> Result<Val, String> {
    if states.is_empty() {
        return Ok(Val::Tensor { data: TData::new(vec![]), shape: vec![0] });
    }
    // Structured tuple state: keep the fields apart and stack each one. A flat
    // tuple of numbers stays in row mode below (so (a,b) → [k,2], as before).
    if let Val::Tuple(first) = &states[0] {
        if first.iter().any(|x| !matches!(x, Val::Num(_) | Val::Complex(..))) {
            let arity = first.len();
            let mut columns: Vec<Vec<Val>> = (0..arity).map(|_| Vec::with_capacity(states.len())).collect();
            for s in states {
                match s {
                    Val::Tuple(items) if items.len() == arity => {
                        for (j, it) in items.into_iter().enumerate() { columns[j].push(it); }
                    }
                    other => return Err(format!(
                        "{who}: structured states must all be {arity}-tuples (got {})", fmt_val(&other))),
                }
            }
            let fields = columns.into_iter()
                .map(|c| stack_rows(c, who))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Val::Tuple(fields));
        }
    }
    if matches!(states[0], Val::Num(_) | Val::Complex(..)) {
        let mut re = Vec::with_capacity(states.len());
        let mut im = Vec::with_capacity(states.len());
        let mut complex = false;
        for s in states {
            match s {
                Val::Num(x)        => { re.push(x); im.push(0.0); }
                Val::Complex(a, b) => { re.push(a); im.push(b); complex = true; }
                other => return Err(format!(
                    "{who}: states must all be scalars or all be equal-length vectors (got {})",
                    fmt_val(&other))),
            }
        }
        let n = re.len();
        return Ok(if complex { maybe_real(re, im, vec![n]) }
                  else        { Val::Tensor { data: TData::new(re), shape: vec![n] } });
    }
    // Vector/tuple mode: every state is a row of width d.
    let mut re: Vec<f64> = Vec::new();
    let mut im: Vec<f64> = Vec::new();
    let mut complex = false;
    let mut width: Option<usize> = None;
    let k = states.len();
    for s in states {
        let (r, i): (Vec<f64>, Vec<f64>) = match s {
            Val::Tensor { data, shape } if shape.len() == 1 => {
                let d = data.len();
                (data.into_vec(), vec![0.0; d])
            }
            Val::ComplexTensor { re, im, shape } if shape.len() == 1 => {
                complex = true;
                (re.into_vec(), im.into_vec())
            }
            Val::Tuple(items) => {
                let mut r = Vec::with_capacity(items.len());
                let mut i = Vec::with_capacity(items.len());
                for it in items {
                    let (a, b) = to_complex(it)?;
                    if b != 0.0 { complex = true; }
                    r.push(a); i.push(b);
                }
                (r, i)
            }
            other => return Err(format!(
                "{who}: states must all be scalars or all be equal-length 1-D vectors (got {})",
                fmt_val(&other))),
        };
        match width {
            None => width = Some(r.len()),
            Some(w) if w != r.len() =>
                return Err(format!("{who}: inconsistent state length ({w} vs {})", r.len())),
            _ => {}
        }
        re.extend(r);
        im.extend(i);
    }
    let d = width.unwrap_or(0);
    Ok(if complex { maybe_real(re, im, vec![k, d]) }
       else        { Val::Tensor { data: TData::new(re), shape: vec![k, d] } })
}

// Promote a value to a 2-D block (re, im, rows, cols) for the stacking family.
// A 1-D vector becomes a row [1,n] when `as_row`, else a column [n,1]; a scalar
// becomes [1,1]; a 2-D tensor is kept as-is. Lets vstack/hstack accept scalars,
// vectors, and matrices uniformly (FEAT-E rank promotion).
fn to_block(v: Val, as_row: bool, who: &str) -> Result<(Vec<f64>, Vec<f64>, usize, usize), String> {
    let vec_dims = |n: usize| if as_row { (1, n) } else { (n, 1) };
    match v {
        Val::Num(x)        => Ok((vec![x], vec![0.0], 1, 1)),
        Val::Complex(a, b) => Ok((vec![a], vec![b], 1, 1)),
        Val::Tensor { data, shape } => match shape.len() {
            0 | 1 => { let n = data.len(); let (r, c) = vec_dims(n); Ok((data.into_vec(), vec![0.0; n], r, c)) }
            2     => { let (r, c) = (shape[0], shape[1]); let n = data.len(); Ok((data.into_vec(), vec![0.0; n], r, c)) }
            _     => Err(format!("{who}: tensors must be 1-D or 2-D, got {}-D", shape.len())),
        },
        Val::ComplexTensor { re, im, shape } => match shape.len() {
            0 | 1 => { let n = re.len(); let (r, c) = vec_dims(n); Ok((re.into_vec(), im.into_vec(), r, c)) }
            2     => { let (r, c) = (shape[0], shape[1]); Ok((re.into_vec(), im.into_vec(), r, c)) }
            _     => Err(format!("{who}: tensors must be 1-D or 2-D, got {}-D", shape.len())),
        },
        Val::Tuple(items) => {
            let n = items.len();
            let mut re = Vec::with_capacity(n);
            let mut im = Vec::with_capacity(n);
            for it in items { let (a, b) = to_complex(it)?; re.push(a); im.push(b); }
            let (r, c) = vec_dims(n);
            Ok((re, im, r, c))
        }
        other => Err(format!("{who}: cannot stack {}", fmt_val(&other))),
    }
}

// Simpson's rule: integral(f, a, b) or integral(f, a, b, n)
pub fn eval_integral(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() < 3 || args.len() > 4 {
        return Err("integral(f, a, b) or integral(f, a, b, n)".into());
    }
    // Evaluate f once so the bytecode cache is shared across ~1000 calls.
    let f_val = eval(&args[0], env)?;
    let call  = |x: f64| apply_val(f_val.clone(), vec![Val::Num(x)], env).and_then(|v| v.num("f"));
    let a = eval(&args[1], env)?.num("a")?;
    let b = eval(&args[2], env)?.num("b")?;
    let n = if args.len() == 4 { eval(&args[3], env)?.num("n")? as usize } else { 1000 };
    let n = n + n % 2;
    let h = (b - a) / n as f64;
    let mut s = call(a)? + call(b)?;
    for i in 1..n {
        s += call(a + i as f64 * h)? * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    Ok(Val::Num(s * h / 3.0))
}

// 5-point stencil derivative: deriv(f, x) or deriv(f, x, dx)
pub fn eval_deriv(args: &[Expr], env: &Env) -> Result<Val, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err("deriv(f, x) or deriv(f, x, dx)".into());
    }
    let x  = eval(&args[1], env)?.num("x")?;
    let dx = if args.len() == 3 { eval(&args[2], env)?.num("dx")? } else { 1e-5 };
    let f  = |t: f64| call_fn1(&args[0], Val::Num(t), env).and_then(|v| v.num("f"));
    Ok(Val::Num(
        (-f(x + 2.0*dx)? + 8.0*f(x + dx)? - 8.0*f(x - dx)? + f(x - 2.0*dx)?) / (12.0 * dx)
    ))
}

// Apply a 1-arg function expression to a value.
pub fn call_fn1(f_expr: &Expr, x: Val, env: &Env) -> Result<Val, String> {
    let f_val = eval(f_expr, env)?;
    apply_val(f_val, vec![x], env)
}
