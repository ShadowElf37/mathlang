use crate::lexer::Token;
use crate::ast::{Expr, BlockStmt, Op, Def};

pub struct Parser { toks: Vec<Token>, pos: usize }

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self { Self { toks, pos: 0 } }
    fn peek(&self) -> &Token { &self.toks[self.pos] }
    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() { self.pos += 1; }
        t
    }
    fn eat(&mut self, expected: &Token) -> Result<(), String> {
        if self.peek() == expected { self.bump(); Ok(()) }
        else { Err(format!("expected {:?}, got {:?}", expected, self.peek())) }
    }

    // Ident (Comma Ident)+ Arrow — bare multi-arg lambda: n, r -> expr
    fn is_multi_lambda(&self) -> bool {
        let mut p = self.pos;
        if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
        p += 1;
        let mut count = 0;
        loop {
            if !matches!(self.toks.get(p), Some(Token::Comma)) { break; }
            p += 1;
            if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
            p += 1;
            count += 1;
        }
        count > 0 && matches!(self.toks.get(p), Some(Token::Arrow))
    }

    // Ident (Comma Ident)* RParen Arrow — paren-wrapped lambda: (n, r) -> expr
    // Also accepts (n, r -> expr) — arrow inside the parens after the last param.
    fn looks_like_paren_lambda(&self) -> bool {
        let mut p = self.pos;
        if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
        p += 1;
        let mut count = 0;
        loop {
            match self.toks.get(p) {
                Some(Token::RParen) => return matches!(self.toks.get(p + 1), Some(Token::Arrow)),
                Some(Token::Arrow) if count > 0 => return true,
                Some(Token::Comma) => {
                    p += 1;
                    if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
                    p += 1;
                    count += 1;
                }
                _ => return false,
            }
        }
    }

    fn parse_multi_lambda(&mut self) -> Result<Expr, String> {
        let mut params = vec![];
        loop {
            match self.bump() {
                Token::Ident(s) => params.push(s),
                t => return Err(format!("expected param name, got {:?}", t)),
            }
            if *self.peek() == Token::Comma { self.bump(); } else { break; }
        }
        self.eat(&Token::Arrow)?;
        Ok(Expr::Lambda(params, self.expr()?.into()))
    }

    // Ident '=' ...  or  Ident '(' params ')' '=' ...
    fn is_def_start(&self) -> bool {
        let p = self.pos;
        if !matches!(self.toks.get(p), Some(Token::Ident(_))) { return false; }
        if matches!(self.toks.get(p + 1), Some(Token::Eq)) { return true; }
        if matches!(self.toks.get(p + 1), Some(Token::LParen)) {
            let mut q = p + 2;
            let mut depth = 1usize;
            while q < self.toks.len() {
                match &self.toks[q] {
                    Token::LParen => depth += 1,
                    Token::RParen => {
                        depth -= 1;
                        if depth == 0 {
                            return matches!(self.toks.get(q + 1), Some(Token::Eq));
                        }
                    }
                    Token::Eof => return false,
                    _ => {}
                }
                q += 1;
            }
        }
        false
    }

    fn has_top_level_colon(&self) -> bool {
        let mut depth = 0usize;
        for t in &self.toks {
            match t {
                Token::LParen | Token::LBrace | Token::LBracket => depth += 1,
                Token::RParen | Token::RBrace | Token::RBracket => { if depth > 0 { depth -= 1; } }
                Token::Colon if depth == 0 => return true,
                Token::Eof => break,
                _ => {}
            }
        }
        false
    }

    pub fn parse_repl(&mut self) -> Result<(Vec<Def>, Vec<Expr>), String> {
        if self.has_top_level_colon() { return self.parse(); }
        let mut defs = vec![];
        while self.is_def_start() {
            defs.push(self.parse_def()?);
            if *self.peek() == Token::Semicolon { self.bump(); } else { break; }
        }
        let exprs = if *self.peek() == Token::Eof { vec![] } else { self.parse_expr_list()? };
        if *self.peek() != Token::Eof {
            return Err(format!("unexpected token: {:?}", self.peek()));
        }
        Ok((defs, exprs))
    }

    fn parse(&mut self) -> Result<(Vec<Def>, Vec<Expr>), String> {
        let defs = self.parse_defs()?;
        if *self.peek() != Token::Colon {
            return Err(format!("expected ':', got {:?}", self.peek()));
        }
        self.bump();
        Ok((defs, self.parse_expr_list()?))
    }

    // Defs separated by ';'; stops before ':' or when next item is not a def.
    fn parse_defs(&mut self) -> Result<Vec<Def>, String> {
        if *self.peek() == Token::Colon || *self.peek() == Token::Eof { return Ok(vec![]); }
        let mut defs = vec![self.parse_def()?];
        while *self.peek() == Token::Semicolon {
            self.bump();
            if *self.peek() == Token::Colon || *self.peek() == Token::Eof { break; }
            if !self.is_def_start() { break; }
            defs.push(self.parse_def()?);
        }
        Ok(defs)
    }

    // Parse { stmts } contents (cursor is after '{').
    // ';' separates stmts. ':' introduces explicit output list (→ Tuple if multiple).
    // Without ':', the last Expr stmt is the output.
    fn parse_block_inner(&mut self) -> Result<Expr, String> {
        let mut stmts: Vec<BlockStmt> = vec![];
        loop {
            while *self.peek() == Token::Semicolon { self.bump(); }
            match self.peek() {
                Token::RBrace | Token::Eof => break,
                Token::Colon => {
                    self.bump();
                    let mut outs = vec![];
                    if *self.peek() != Token::RBrace {
                        outs.push(self.expr()?);
                        while *self.peek() == Token::Comma {
                            self.bump();
                            if *self.peek() == Token::RBrace { break; }
                            outs.push(self.expr()?);
                        }
                    }
                    let out_expr = if outs.len() == 1 {
                        outs.into_iter().next().unwrap()
                    } else {
                        Expr::Tuple(outs)
                    };
                    stmts.push(BlockStmt::Expr(out_expr));
                    break;
                }
                _ => {}
            }
            if self.is_def_start() {
                stmts.push(BlockStmt::Def(self.parse_def()?));
            } else {
                stmts.push(BlockStmt::Expr(self.expr()?));
            }
            match self.peek() {
                Token::Semicolon | Token::RBrace | Token::Colon | Token::Eof => {}
                t => return Err(format!("expected ';' or '}}', got {:?}", t.clone())),
            }
        }
        self.eat(&Token::RBrace)?;
        Ok(Expr::Block(stmts))
    }

    fn parse_def(&mut self) -> Result<Def, String> {
        let name = match self.bump() {
            Token::Ident(s) => s,
            t => return Err(format!("expected name, got {:?}", t)),
        };
        if *self.peek() == Token::LParen {
            self.bump();
            let mut params = vec![];
            if *self.peek() != Token::RParen {
                loop {
                    match self.bump() {
                        Token::Ident(s) => params.push(s),
                        t => return Err(format!("expected param name, got {:?}", t)),
                    }
                    if *self.peek() == Token::Comma { self.bump(); } else { break; }
                }
            }
            self.eat(&Token::RParen)?;
            self.eat(&Token::Eq)?;
            Ok(Def::Func(name, params, self.expr()?))
        } else {
            self.eat(&Token::Eq)?;
            if self.is_multi_lambda() {
                Ok(Def::Var(name, self.parse_multi_lambda()?))
            } else {
                Ok(Def::Var(name, self.expr()?))
            }
        }
    }

    fn parse_expr_list(&mut self) -> Result<Vec<Expr>, String> {
        if *self.peek() == Token::Eof { return Ok(vec![]); }
        let mut exprs = vec![self.expr()?];
        while *self.peek() == Token::Comma {
            self.bump();
            if *self.peek() == Token::Eof { break; }
            exprs.push(self.expr()?);
        }
        Ok(exprs)
    }

    fn expr(&mut self) -> Result<Expr, String> { self.cmp() }

    fn cmp(&mut self) -> Result<Expr, String> {
        let mut l = self.add()?;
        loop {
            let op = match self.peek() {
                Token::Lt     => Op::Lt,
                Token::Gt     => Op::Gt,
                Token::LtEq   => Op::LtEq,
                Token::GtEq   => Op::GtEq,
                Token::EqEq   => Op::Eq,
                Token::BangEq => Op::Ne,
                Token::Amp    => Op::And,
                Token::Pipe   => Op::Or,
                _ => break,
            };
            self.bump();
            l = Expr::BinOp(l.into(), op, self.add()?.into());
        }
        Ok(l)
    }

    fn add(&mut self) -> Result<Expr, String> {
        let mut l = self.mul()?;
        loop {
            l = match self.peek() {
                Token::Plus  => { self.bump(); Expr::BinOp(l.into(), Op::Add, self.mul()?.into()) }
                Token::Minus => { self.bump(); Expr::BinOp(l.into(), Op::Sub, self.mul()?.into()) }
                _ => break,
            };
        }
        Ok(l)
    }

    fn mul(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        loop {
            l = match self.peek() {
                Token::Star       => { self.bump(); Expr::BinOp(l.into(), Op::Mul,      self.unary()?.into()) }
                Token::Slash      => { self.bump(); Expr::BinOp(l.into(), Op::Div,      self.unary()?.into()) }
                Token::SlashSlash => { self.bump(); Expr::BinOp(l.into(), Op::FloorDiv, self.unary()?.into()) }
                Token::Percent    => { self.bump(); Expr::BinOp(l.into(), Op::Rem,      self.unary()?.into()) }
                Token::At         => {
                    self.bump();
                    let r = self.unary()?;
                    Expr::Apply(Box::new(Expr::Var("matmul".into())), vec![l, r])
                }
                _ => break,
            };
        }
        Ok(l)
    }

    // Parse one bracket index item: expr or expr..expr (range)
    /// Parse one element inside `T[…]`:
    ///   expr          → scalar index (no slice)
    ///   expr ..       → slice from expr to end: T[1..]
    ///   expr .. expr  → bounded slice: T[1..3]
    ///        ..       → all indices of this dimension: T[..]
    ///        .. expr  → slice from start to expr: T[..3]
    fn parse_index_item(&mut self) -> Result<Expr, String> {
        // Standalone `..` (or `.. end`)
        if *self.peek() == Token::DotDot {
            self.bump();
            let hi = match self.peek() {
                Token::Comma | Token::RBracket | Token::Eof => None,
                _ => Some(Box::new(self.expr()?)),
            };
            return Ok(Expr::Slice(None, hi));
        }
        let e = self.expr()?;
        if *self.peek() == Token::DotDot {
            self.bump();
            // `expr ..` with nothing after → open-ended slice
            let hi = match self.peek() {
                Token::Comma | Token::RBracket | Token::Eof => None,
                _ => Some(Box::new(self.expr()?)),
            };
            Ok(Expr::Slice(Some(Box::new(e)), hi))
        } else {
            Ok(e)
        }
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if *self.peek() == Token::Minus { self.bump(); return Ok(Expr::Neg(self.pow()?.into())); }
        self.pow()
    }

    fn pow(&mut self) -> Result<Expr, String> {
        let base = self.postfix()?;
        if matches!(self.peek(), Token::Caret | Token::StarStar) {
            self.bump();
            Ok(Expr::BinOp(base.into(), Op::Pow, self.unary()?.into()))
        } else {
            Ok(base)
        }
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        loop {
            if *self.peek() == Token::Bang {
                self.bump();
                e = Expr::Apply(Box::new(Expr::Var("fact".into())), vec![e]);
            } else if *self.peek() == Token::LBracket {
                self.bump();
                let first = self.parse_index_item()?;
                if *self.peek() == Token::Comma {
                    let mut indices = vec![first];
                    while *self.peek() == Token::Comma {
                        self.bump();
                        if *self.peek() == Token::RBracket { break; }
                        indices.push(self.parse_index_item()?);
                    }
                    self.eat(&Token::RBracket)?;
                    e = Expr::Index(Box::new(e), Box::new(Expr::Tuple(indices)));
                } else {
                    self.eat(&Token::RBracket)?;
                    e = Expr::Index(Box::new(e), Box::new(first));
                }
            } else if *self.peek() == Token::LParen {
                // Only treat as Apply if this is NOT an Ident (those are parsed as Call in primary)
                // For all other expressions (lambdas, tuples, blocks, etc.) postfix () = Apply
                if matches!(e, Expr::Var(_)) { break; }
                self.bump();
                let mut args = vec![];
                if *self.peek() != Token::RParen {
                    loop {
                        args.push(self.expr()?);
                        if *self.peek() == Token::Comma { self.bump(); } else { break; }
                    }
                }
                self.eat(&Token::RParen)?;
                e = Expr::Apply(Box::new(e), args);
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Token::Num(n)  => {
                self.bump();
                if matches!(self.peek(), Token::Ident(_) | Token::LParen) {
                    let rhs = self.primary()?;
                    Ok(Expr::BinOp(Box::new(Expr::Num(n)), Op::Mul, Box::new(rhs)))
                } else {
                    Ok(Expr::Num(n))
                }
            }
            Token::Imag(n) => { self.bump(); Ok(Expr::ImagLit(n)) }
            Token::LBrace => {
                self.bump();
                self.parse_block_inner()
            }
            Token::LParen => {
                self.bump();
                if self.looks_like_paren_lambda() {
                    let mut params = vec![];
                    let mut had_rparen = false;
                    loop {
                        match self.bump() {
                            Token::Ident(s) => params.push(s),
                            t => return Err(format!("expected param, got {:?}", t)),
                        }
                        match self.peek().clone() {
                            Token::Comma  => { self.bump(); }
                            Token::RParen => { self.bump(); had_rparen = true; break; }
                            Token::Arrow  => { break; }
                            ref t => return Err(format!("expected ',' or ')', got {:?}", t)),
                        }
                    }
                    self.eat(&Token::Arrow)?;
                    let body = self.expr()?;
                    if !had_rparen { self.eat(&Token::RParen)?; }
                    Ok(Expr::Lambda(params, body.into()))
                } else {
                    // Empty parens → empty tuple
                    if *self.peek() == Token::RParen {
                        self.bump();
                        return Ok(Expr::Tuple(vec![]));
                    }
                    let first = self.expr()?;
                    // Range literal (a..b)
                    if *self.peek() == Token::DotDot {
                        self.bump();
                        let last = self.expr()?;
                        self.eat(&Token::RParen)?;
                        return Ok(Expr::Range(Box::new(first), Box::new(last)));
                    }
                    // Collect first row (comma-separated items)
                    let mut row0 = vec![first];
                    while *self.peek() == Token::Comma {
                        self.bump();
                        if matches!(self.peek(), Token::RParen | Token::Semicolon) { break; }
                        row0.push(self.expr()?);
                    }
                    // Matrix literal: rows separated by ;
                    if *self.peek() == Token::Semicolon {
                        let mut rows = vec![row0];
                        while *self.peek() == Token::Semicolon {
                            self.bump();
                            if *self.peek() == Token::RParen { break; }
                            let mut row = vec![self.expr()?];
                            while *self.peek() == Token::Comma {
                                self.bump();
                                if matches!(self.peek(), Token::RParen | Token::Semicolon) { break; }
                                row.push(self.expr()?);
                            }
                            rows.push(row);
                        }
                        self.eat(&Token::RParen)?;
                        return Ok(Expr::TensorLit(rows));
                    }
                    self.eat(&Token::RParen)?;
                    if row0.len() == 1 {
                        Ok(row0.into_iter().next().unwrap())
                    } else {
                        Ok(Expr::Tuple(row0))
                    }
                }
            }
            // [] tensor literals — same semantics as () but with square brackets.
            // [a, b, c] → Tuple (auto-promotes to Tensor when all-numeric).
            // [1,2;3,4] → TensorLit (matrix).
            // []        → empty Tuple.
            Token::LBracket => {
                self.bump();
                if *self.peek() == Token::RBracket {
                    self.bump();
                    return Ok(Expr::Tuple(vec![]));
                }
                let first = self.expr()?;
                let mut row0 = vec![first];
                while *self.peek() == Token::Comma {
                    self.bump();
                    if matches!(self.peek(), Token::RBracket | Token::Semicolon) { break; }
                    row0.push(self.expr()?);
                }
                if *self.peek() == Token::Semicolon {
                    let mut rows = vec![row0];
                    while *self.peek() == Token::Semicolon {
                        self.bump();
                        if *self.peek() == Token::RBracket { break; }
                        let mut row = vec![self.expr()?];
                        while *self.peek() == Token::Comma {
                            self.bump();
                            if matches!(self.peek(), Token::RBracket | Token::Semicolon) { break; }
                            row.push(self.expr()?);
                        }
                        rows.push(row);
                    }
                    self.eat(&Token::RBracket)?;
                    return Ok(Expr::TensorLit(rows));
                }
                self.eat(&Token::RBracket)?;
                if row0.len() == 1 {
                    Ok(row0.into_iter().next().unwrap())
                } else {
                    Ok(Expr::Tuple(row0))
                }
            }
            Token::Minus => { self.bump(); Ok(Expr::Neg(self.primary()?.into())) }
            Token::Ident(name) => {
                self.bump();
                if *self.peek() == Token::Arrow {
                    self.bump();
                    Ok(Expr::Lambda(vec![name], self.expr()?.into()))
                } else if *self.peek() == Token::LParen {
                    self.bump();
                    let mut args = vec![];
                    if *self.peek() != Token::RParen {
                        loop {
                            args.push(self.expr()?);
                            if *self.peek() == Token::Comma { self.bump(); } else { break; }
                        }
                    }
                    self.eat(&Token::RParen)?;
                    Ok(Expr::Apply(Box::new(Expr::Var(name)), args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            t => Err(format!("unexpected token: {:?}", t)),
        }
    }
}
