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

#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    ImagLit(f64),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    Neg(Box<Expr>),
    // params, optional return hint (only Def::Func supports return hint), body
    Lambda(Vec<Param>, Option<TypeHint>, Box<Expr>),
    Tuple(Vec<Expr>),
    TensorLit(Vec<Vec<Expr>>),   // (1,2; 3,4) — rows separated by ;
    Array(Vec<Expr>),            // [a,b,c]    — 1-D tensor literal; all elements must be numeric
    Index(Box<Expr>, Box<Expr>),
    /// Namespace member access: `ns.member` (e.g. operators.grad).
    Member(Box<Expr>, String),
    Block(Vec<BlockStmt>),
    Apply(Box<Expr>, Vec<Expr>),
    Range(Box<Expr>, Box<Expr>),
    /// Index-position slice: T[lo..hi]  T[lo..]  T[..hi]  T[..]
    /// Only produced by parse_index_item; never appears outside Index children.
    Slice(Option<Box<Expr>>, Option<Box<Expr>>),
}

#[derive(Debug, Clone)]
pub enum BlockStmt {
    Def(Def),
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
