#[derive(Debug, Clone, PartialEq)]
pub enum TypeHint {
    Any,
    Num,
    Real,
    Complex,
    Int,
    Nat,
    Tensor,
    RealTensor,
    ComplexTensor,
    Fn,
    Cell,
    Tuple,
}

impl TypeHint {
    pub fn display(&self) -> &'static str {
        match self {
            TypeHint::Any           => "any",
            TypeHint::Num           => "num",
            TypeHint::Real          => "real",
            TypeHint::Complex       => "complex",
            TypeHint::Int           => "int",
            TypeHint::Nat           => "nat",
            TypeHint::Tensor        => "tensor",
            TypeHint::RealTensor    => "real tensor",
            TypeHint::ComplexTensor => "complex tensor",
            TypeHint::Fn            => "fn",
            TypeHint::Cell          => "cell",
            TypeHint::Tuple         => "tuple",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub hint: Option<TypeHint>,
}

/// One item of a record literal. `name: None` is a positional item.
/// A function field (`mag(v) = norm(v)`) is just a field whose value is a
/// `Lambda`, so it needs no separate representation.
#[derive(Debug, Clone)]
pub struct Field {
    pub name:  Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    ImagLit(f64),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    // params, optional return hint (only Def::Func supports return hint), body
    Lambda(Vec<Param>, Option<TypeHint>, Box<Expr>),
    Tuple(Vec<Expr>),
    /// A named tuple / record literal: `(x = 1, y = 2)`.
    ///
    /// Only produced when at least one field is named — an all-positional
    /// paren list still parses to `Tuple`, so nothing that existed before
    /// routes through this node. Fields may be mixed: `(1, y = 2)`.
    Record(Vec<Field>),
    TensorLit(Vec<Vec<Expr>>),   // (1,2; 3,4) — rows separated by ;
    Array(Vec<Expr>),            // [a,b,c]    — 1-D tensor literal; all elements must be numeric
    Index(Box<Expr>, Box<Expr>),
    /// Namespace member access: `ns.member` (e.g. ops.grad).
    Member(Box<Expr>, String),
    Block(Vec<BlockStmt>),
    Apply(Box<Expr>, Vec<Expr>),
    Range(Box<Expr>, Box<Expr>),
    /// Index-position slice: T[lo..hi]  T[lo..]  T[..hi]  T[..]
    /// Only produced by parse_index_item; never appears outside Index children.
    Slice(Option<Box<Expr>>, Option<Box<Expr>>),
    /// `..x` — splice a value's slots into the enclosing list.
    ///
    /// Legal only as an item of a paren list, an array literal, or a call
    /// argument list; every other position (including inside `T[…]`, where
    /// `..` already means a slice) rejects it at parse time. A tuple/record
    /// contributes its items with their field names, a 1-D array its elements.
    Splat(Box<Expr>),
    /// `GPU { ... }` — a block evaluated on the GPU compute backend.
    /// The body is a standard `Expr::Block`.
    GpuBlock(Box<Expr>),
}

/// One step of an assignment path: `T[i]` or `w.alpha`.
#[derive(Debug, Clone)]
pub enum PathSeg {
    /// The index expression exactly as the reader builds it — `T[i,j]` is a
    /// `Tuple`, `T[a..b]` a `Slice` — so writes resolve indices through the
    /// same code as reads and inherit its negative-index and slice rules.
    Index(Expr),
    Field(String),
}

/// The left-hand side of an assignment: a root name and a path into it.
/// An empty path is a plain `x = …` rebinding of the whole value.
#[derive(Debug, Clone)]
pub struct LValue {
    pub root: String,
    pub path: Vec<PathSeg>,
}

/// `T[i] = v`, `w.alpha += 1`. `op` is `None` for a plain `=` and `Some(op)`
/// for a compound assignment, which reads the slot and combines before storing.
///
/// Deliberately *not* a `Def`: an assignment introduces no name, and `Def` also
/// appears in record fields and namespace files where an assignment is invalid.
#[derive(Debug, Clone)]
pub struct Assign {
    pub lvalue: LValue,
    pub op:     Option<Op>,
    pub value:  Expr,
}

#[derive(Debug, Clone)]
pub enum BlockStmt {
    Def(Def),
    Assign(Assign),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum Op { Add, Sub, Mul, Div, FloorDiv, Rem, Pow, Lt, Gt, LtEq, GtEq, Eq, Ne, And, Or }

#[derive(Debug, Clone)]
pub enum Def {
    Var(String, Expr),
    // name, params (with hints), return hint, body
    Func(String, Vec<Param>, Option<TypeHint>, Expr),
}
