//! Line-oriented lexer. Strips `#` comments, joins continuation lines (indented
//! lines starting with `+` or `-`) onto the previous statement, and tokenizes.

use crate::error::{Result, WirError};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    /// Unsigned literal; signs are separate tokens. `p/q` literals are folded here.
    Num(f64),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Slash,
    Dot,
    DotDot,
    Comma,
    Colon,
    Arrow,
    Eq,
    EqEq,
    Plus,
    Minus,
    Star,
}

impl Tok {
    pub fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Num(x) => format!("number {x}"),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::LBrace => "`{`".into(),
            Tok::RBrace => "`}`".into(),
            Tok::LBracket => "`[`".into(),
            Tok::RBracket => "`]`".into(),
            Tok::Slash => "`/`".into(),
            Tok::Dot => "`.`".into(),
            Tok::DotDot => "`..`".into(),
            Tok::Comma => "`,`".into(),
            Tok::Colon => "`:`".into(),
            Tok::Arrow => "`->`".into(),
            Tok::Eq => "`=`".into(),
            Tok::EqEq => "`==`".into(),
            Tok::Plus => "`+`".into(),
            Tok::Minus => "`-`".into(),
            Tok::Star => "`*`".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
    pub col: usize,
}

/// One logical statement: its first line, joined text and tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub line: usize,
    pub text: String,
    pub tokens: Vec<Token>,
}

pub fn split_statements(src: &str) -> Result<Vec<Statement>> {
    let mut out: Vec<Statement> = Vec::new();
    for (idx, raw) in src.lines().enumerate() {
        let line_no = idx + 1;
        let code = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        }
        .trim_end();
        let trimmed = code.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let indented = code.len() != trimmed.len();
        let continues = indented
            && (trimmed.starts_with('+') || trimmed.starts_with('-') || trimmed.starts_with('\u{2212}'));
        let col0 = code.len() - trimmed.len();
        if continues && !out.is_empty() {
            let stmt = out.last_mut().unwrap();
            let toks = tokenize(trimmed, line_no, col0, &format!("{} {}", stmt.text, trimmed), stmt.line)?;
            stmt.text.push(' ');
            stmt.text.push_str(trimmed);
            stmt.tokens.extend(toks);
        } else {
            let tokens = tokenize(trimmed, line_no, col0, trimmed, line_no)?;
            out.push(Statement { line: line_no, text: trimmed.to_string(), tokens });
        }
    }
    Ok(out)
}

fn tokenize(s: &str, line: usize, col0: usize, stmt_text: &str, stmt_line: usize) -> Result<Vec<Token>> {
    let chars: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let col = col0 + i + 1;
        let push = |toks: &mut Vec<Token>, tok| toks.push(Token { tok, line, col });
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            push(&mut toks, Tok::Ident(chars[start..i].iter().collect()));
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let (value, next) = lex_number(&chars, i)
                .ok_or_else(|| WirError::at(stmt_line, stmt_text, format!("malformed number at column {col}")))?;
            i = next;
            // `p/q` rational literal.
            if chars.get(i) == Some(&'/') && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit() || *d == '.') {
                let (den, next) = lex_number(&chars, i + 1).ok_or_else(|| {
                    WirError::at(stmt_line, stmt_text, format!("malformed fraction at column {col}"))
                })?;
                if den == 0.0 {
                    return Err(WirError::at(stmt_line, stmt_text, format!("division by zero at column {col}")));
                }
                i = next;
                push(&mut toks, Tok::Num(value / den));
            } else {
                push(&mut toks, Tok::Num(value));
            }
            continue;
        }
        let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
        let (tok, width) = match (two.as_str(), c) {
            ("->", _) => (Tok::Arrow, 2),
            ("==", _) => (Tok::EqEq, 2),
            ("..", _) => (Tok::DotDot, 2),
            (_, '.') => (Tok::Dot, 1),
            (_, '[') => (Tok::LBracket, 1),
            (_, ']') => (Tok::RBracket, 1),
            (_, '/') => (Tok::Slash, 1),
            (_, '(') => (Tok::LParen, 1),
            (_, ')') => (Tok::RParen, 1),
            (_, '{') => (Tok::LBrace, 1),
            (_, '}') => (Tok::RBrace, 1),
            (_, ',') => (Tok::Comma, 1),
            (_, ':') => (Tok::Colon, 1),
            (_, '=') => (Tok::Eq, 1),
            (_, '+') => (Tok::Plus, 1),
            (_, '-') | (_, '\u{2212}') => (Tok::Minus, 1),
            (_, '*') => (Tok::Star, 1),
            _ => {
                return Err(WirError::at(
                    stmt_line,
                    stmt_text,
                    format!("unexpected character '{c}' at line {line}, column {col}"),
                ));
            }
        };
        push(&mut toks, tok);
        i += width;
    }
    Ok(toks)
}

/// `digits [. digits] [e [+-] digits]`, returning the value and next index.
fn lex_number(chars: &[char], start: usize) -> Option<(f64, usize)> {
    let mut i = start;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
        let mut j = i + 1;
        if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
            j += 1;
        }
        if j < chars.len() && chars[j].is_ascii_digit() {
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    let text: String = chars[start..i].iter().collect();
    text.parse::<f64>().ok().map(|v| (v, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_continuations_and_strips_comments() {
        let src = "y = hinge at 0 turn +1   # first\n  + hinge at 0.5 turn -2\n\nassert y(1/3) == 2e-1\n";
        let stmts = split_statements(src).unwrap();
        assert_eq!(stmts.len(), 2);
        assert_eq!(stmts[0].line, 1);
        assert_eq!(stmts[0].text, "y = hinge at 0 turn +1 + hinge at 0.5 turn -2");
        assert_eq!(stmts[1].line, 4);
        assert!(stmts[1].tokens.contains(&Token { tok: Tok::Num(1.0 / 3.0), line: 4, col: 10 }));
        assert!(stmts[1].tokens.iter().any(|t| t.tok == Tok::Num(0.2)));
    }

    #[test]
    fn arrow_then_negative() {
        let s = split_statements("(0,0)->-1").unwrap();
        let kinds: Vec<_> = s[0].tokens.iter().map(|t| t.tok.clone()).collect();
        assert_eq!(kinds[5], Tok::Arrow);
        assert_eq!(kinds[6], Tok::Minus);
    }
}
