#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(f64), Imag(f64), Ident(String),
    Plus, Minus, Star, Slash, SlashSlash, Percent, Caret, StarStar,
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Colon, Semicolon, Eq, Arrow, DotDot, Dot,
    Lt, Gt, LtEq, GtEq, EqEq, Bang, BangEq,
    Amp, Pipe, At, Tilde,
    Eof,
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

    pub fn tokenize(mut self) -> Vec<Token> {
        let mut out = Vec::new();
        loop {
            while self.peek().map_or(false, |b| b.is_ascii_whitespace()) { self.bump(); }
            match self.peek() {
                None | Some(b'#') => { out.push(Token::Eof); break; }
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
