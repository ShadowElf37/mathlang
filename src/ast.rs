#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    ImagLit(f64),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    Neg(Box<Expr>),
    Lambda(Vec<String>, Box<Expr>),
    Tuple(Vec<Expr>),
    TensorLit(Vec<Vec<Expr>>),   // (1,2; 3,4) — rows separated by ;
    Array(Vec<Expr>),            // [a,b,c]    — 1-D tensor literal; all elements must be numeric
    Index(Box<Expr>, Box<Expr>),
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
    Func(String, Vec<String>, Expr),
}
