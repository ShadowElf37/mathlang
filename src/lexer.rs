#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(f64), Imag(f64), Ident(String),
    Plus, Minus, Star, Slash, SlashSlash, Percent, Caret, StarStar,
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Colon, Semicolon, Eq, Arrow, DotDot, Dot,
    Lt, Gt, LtEq, GtEq, EqEq, Bang, BangEq,
    Amp, Pipe, At, Tilde,
    /// A source line break. Only ever present in the *raw* token stream:
    /// `tokenize()` elides the meaningless ones and rewrites the rest into
    /// `Comma`/`Semicolon`, so the parser never sees this variant.
    Newline,
    Eof,
}

/// Can this token legally be the *last* token of a complete expression?
/// A line ending in anything else is unfinished, so the following newline
/// is not a separator. (`Bang` is postfix factorial — `5!` ends an expression.)
fn can_end_expr(t: &Token) -> bool {
    matches!(t,
        Token::Num(_) | Token::Imag(_) | Token::Ident(_) |
        Token::RParen | Token::RBracket | Token::RBrace | Token::Bang)
}

/// Can this token legally be the *first* token of an expression or statement?
/// A line starting with anything else is a continuation of the previous line
/// (e.g. a leading `.member`, or a binary operator).
///
/// `Minus` and `Tilde` are included: they are prefix operators and may begin a
/// statement. `Amp`/`Pipe` (and/or), `At` (matmul) and `Bang` (factorial) are
/// strictly infix/postfix, so a line starting with one is always a continuation.
fn can_start_expr(t: &Token) -> bool {
    matches!(t,
        Token::Num(_) | Token::Imag(_) | Token::Ident(_) |
        Token::LParen | Token::LBracket | Token::LBrace |
        Token::Minus | Token::Tilde)
}

pub struct Lexer<'a> { src: &'a [u8], pos: usize }

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self { Self { src: src.as_bytes(), pos: 0 } }
    fn peek(&self) -> Option<u8> { self.src.get(self.pos).copied() }
    fn bump(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() { self.pos += 1; }
        b
    }

    /// Lex `src` into the token stream the parser consumes: newlines are
    /// resolved into the ordinary `Comma`/`Semicolon` separators (see
    /// `elide_newlines` / `newlines_to_separators`), so `Token::Newline`
    /// never escapes this function.
    pub fn tokenize(self) -> Vec<Token> {
        newlines_to_separators(elide_newlines(self.tokenize_raw()))
    }

    /// Lex without any newline processing. `Token::Newline` marks each source
    /// line break; blank lines and comment-only lines still produce one.
    pub fn tokenize_raw(mut self) -> Vec<Token> {
        let mut out = Vec::new();
        loop {
            // Skip whitespace *except* line breaks, which are significant.
            while self.peek().map_or(false, |b| b.is_ascii_whitespace() && b != b'\n' && b != b'\r') {
                self.bump();
            }
            match self.peek() {
                None => { out.push(Token::Eof); break; }
                // `#` comments run to end of line — not to end of input, which
                // would truncate every file at its first comment.
                Some(b'#') => {
                    while self.peek().map_or(false, |b| b != b'\n' && b != b'\r') { self.bump(); }
                }
                Some(b'\r') | Some(b'\n') => {
                    if self.peek() == Some(b'\r') { self.bump(); }
                    if self.peek() == Some(b'\n') { self.bump(); }
                    out.push(Token::Newline);
                }
                Some(b) => match b {
                    b'+' => { self.bump(); out.push(Token::Plus); }
                    b'-' => {
                        self.bump();
                        if self.peek() == Some(b'>') { self.bump(); out.push(Token::Arrow); }
                        else { out.push(Token::Minus); }
                    }
                    b'*' => {
                        self.bump();
                        if self.peek() == Some(b'*') { self.bump(); out.push(Token::StarStar); }
                        else { out.push(Token::Star); }
                    }
                    b'/' => {
                        self.bump();
                        if self.peek() == Some(b'/') { self.bump(); out.push(Token::SlashSlash); }
                        else { out.push(Token::Slash); }
                    }
                    b'%' => { self.bump(); out.push(Token::Percent); }
                    b'^' => { self.bump(); out.push(Token::Caret); }
                    b'(' => { self.bump(); out.push(Token::LParen); }
                    b')' => { self.bump(); out.push(Token::RParen); }
                    b'{' => { self.bump(); out.push(Token::LBrace); }
                    b'}' => { self.bump(); out.push(Token::RBrace); }
                    b'[' => { self.bump(); out.push(Token::LBracket); }
                    b']' => { self.bump(); out.push(Token::RBracket); }
                    b',' => { self.bump(); out.push(Token::Comma); }
                    b':' => { self.bump(); out.push(Token::Colon); }
                    b';' => { self.bump(); out.push(Token::Semicolon); }
                    b'=' => {
                        self.bump();
                        if self.peek() == Some(b'=') { self.bump(); out.push(Token::EqEq); }
                        else { out.push(Token::Eq); }
                    }
                    b'!' => {
                        self.bump();
                        if self.peek() == Some(b'=') { self.bump(); out.push(Token::BangEq); }
                        else { out.push(Token::Bang); }
                    }
                    b'<' => {
                        self.bump();
                        if self.peek() == Some(b'=') { self.bump(); out.push(Token::LtEq); }
                        else { out.push(Token::Lt); }
                    }
                    b'>' => {
                        self.bump();
                        if self.peek() == Some(b'=') { self.bump(); out.push(Token::GtEq); }
                        else { out.push(Token::Gt); }
                    }
                    b'&' => {
                        self.bump();
                        if self.peek() == Some(b'&') { self.bump(); }
                        out.push(Token::Amp);
                    }
                    b'|' => {
                        self.bump();
                        if self.peek() == Some(b'|') { self.bump(); }
                        out.push(Token::Pipe);
                    }
                    b'@' => { self.bump(); out.push(Token::At); }
                    b'~' => { self.bump(); out.push(Token::Tilde); }
                    b'.' => {
                        self.bump();
                        if self.peek() == Some(b'.') { self.bump(); out.push(Token::DotDot); }
                        else { out.push(Token::Dot); }
                    }
                    b if b.is_ascii_digit() || (b == b'.' && self.src.get(self.pos + 1).map_or(false, |&n| n.is_ascii_digit())) => {
                        let start = self.pos;
                        while self.peek().map_or(false, |b| b.is_ascii_digit()) { self.bump(); }
                        if self.peek() == Some(b'.') && self.src.get(self.pos + 1).map_or(false, |&n| n.is_ascii_digit()) {
                            self.bump();
                            while self.peek().map_or(false, |b| b.is_ascii_digit()) { self.bump(); }
                        }
                        if self.peek().map_or(false, |b| b == b'e' || b == b'E') {
                            let next1 = self.src.get(self.pos + 1).copied();
                            let next2 = self.src.get(self.pos + 2).copied();
                            let valid_exp = match next1 {
                                Some(b'+') | Some(b'-') => next2.map_or(false, |b| b.is_ascii_digit()),
                                Some(b) => b.is_ascii_digit(),
                                None => false,
                            };
                            if valid_exp {
                                self.bump();
                                if self.peek().map_or(false, |b| b == b'+' || b == b'-') { self.bump(); }
                                while self.peek().map_or(false, |b| b.is_ascii_digit()) { self.bump(); }
                            }
                        }
                        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
                        let n: f64 = s.parse().unwrap();
                        if self.peek() == Some(b'i')
                            && !self.src.get(self.pos + 1).map_or(false, |&b| b.is_ascii_alphanumeric() || b == b'_')
                        {
                            self.bump();
                            out.push(Token::Imag(n));
                        } else {
                            out.push(Token::Num(n));
                        }
                    }
                    b if b.is_ascii_alphabetic() || b == b'_' => {
                        let start = self.pos;
                        while self.peek().map_or(false, |b| b.is_ascii_alphanumeric() || b == b'_') {
                            self.bump();
                        }
                        let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
                        out.push(Token::Ident(s.to_string()));
                    }
                    b => { eprintln!("unknown char: {}", b as char); self.bump(); }
                }
            }
        }
        out
    }
}

/// Pass A — drop every `Newline` that cannot be a separator: after a token that
/// can't end an expression (operator, `=`, `->`, `,`, an opening bracket, …) or
/// before a token that can't start one (closing bracket, `.member`, infix
/// operator, …). Consecutive newlines collapse, so blank lines are free.
fn elide_newlines(toks: Vec<Token>) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::with_capacity(toks.len());
    for (i, t) in toks.iter().enumerate() {
        if *t != Token::Newline { out.push(t.clone()); continue; }
        // Leading newlines, and runs of them, have no preceding ender.
        match out.last() {
            Some(prev) if can_end_expr(prev) => {}
            _ => continue,
        }
        // Look past any further newlines to the next real token.
        let next = toks[i + 1..].iter().find(|t| **t != Token::Newline);
        match next {
            Some(n) if can_start_expr(n) => out.push(Token::Newline),
            _ => {} // trailing newlines, or a continuation line
        }
    }
    out
}

/// Pass B — rewrite each surviving `Newline` into the separator that bracket
/// context already gives it: `,` inside `(`/`[` (an item list), `;` inside `{`
/// or at top level (a statement list). After this the parser sees only the
/// separators it already understands, so none of its token lookahead
/// (`is_def_start`, `looks_like_paren_lambda`, …) needs to change.
fn newlines_to_separators(toks: Vec<Token>) -> Vec<Token> {
    let mut stack: Vec<Token> = Vec::new();
    toks.into_iter()
        .map(|t| {
            match t {
                Token::LParen | Token::LBracket | Token::LBrace => stack.push(t.clone()),
                Token::RParen | Token::RBracket | Token::RBrace => { stack.pop(); }
                Token::Newline => {
                    return match stack.last() {
                        Some(Token::LParen) | Some(Token::LBracket) => Token::Comma,
                        _ => Token::Semicolon,
                    };
                }
                _ => {}
            }
            t
        })
        .collect()
}

/// True if `line` continues the *previous* line rather than starting a new
/// statement — i.e. its first token cannot begin an expression (a leading
/// `.member`, or an infix operator like `+` / `@`).
///
/// The mirror image of `needs_continuation`, for callers that split input into
/// lines before lexing (`import_file`). Together the two reproduce the lexer's
/// own elision rule across a line-oriented split.
pub fn starts_continuation(line: &str) -> bool {
    match Lexer::new(line).tokenize_raw().iter().find(|t| **t != Token::Newline) {
        Some(Token::Eof) | None => false, // blank or comment-only
        Some(first) => !can_start_expr(first),
    }
}

/// True if `src` is not yet a complete input and the caller should read another
/// line: a bracket is still open, or the last token cannot end an expression
/// (a trailing operator, `=`, `->`, `,`, …).
///
/// Shared by the interactive REPL prompt and by `import_file`'s statement
/// splitting, so all multi-line input obeys one rule.
pub fn needs_continuation(src: &str) -> bool {
    let toks = Lexer::new(src).tokenize_raw();
    let mut depth = 0i32;
    for t in &toks {
        match t {
            Token::LParen | Token::LBracket | Token::LBrace => depth += 1,
            Token::RParen | Token::RBracket | Token::RBrace => depth -= 1,
            _ => {}
        }
    }
    if depth > 0 { return true; }
    match toks.iter().rev().find(|t| **t != Token::Eof && **t != Token::Newline) {
        Some(last) => !can_end_expr(last),
        None => false, // empty / comment-only input is complete, not dangling
    }
}
