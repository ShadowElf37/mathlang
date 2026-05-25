#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    ImagLit(f64),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    Neg(Box<Expr>),
    Lambda(Vec<String>, Box<Expr>),
    Tuple(Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Block(Vec<BlockStmt>),
    Apply(Box<Expr>, Vec<Expr>),
    Range(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum BlockStmt {
    Def(Def),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum Op { Add, Sub, Mul, Div, FloorDiv, Rem, Pow }

#[derive(Debug, Clone)]
pub enum Def {
    Var(String, Expr),
    Func(String, Vec<String>, Expr),
}
