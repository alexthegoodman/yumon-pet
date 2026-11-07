//! Recursive-descent parser: WIR text -> `ast::Source`.

use crate::ast::*;
use crate::error::{Result, WirError};
use crate::lexer::{split_statements, Statement, Tok, Token};

pub fn parse(src: &str) -> Result<Source> {
    let mut stmts = Vec::new();
    for st in split_statements(src)? {
        stmts.push(Cursor::new(&st).statement()?);
    }
    Ok(Source { stmts })
}

struct Cursor<'a> {
    st: &'a Statement,
    toks: &'a [Token],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(st: &'a Statement) -> Self {
        Self { st, toks: &st.tokens, pos: 0 }
    }

    fn err(&self, msg: impl Into<String>) -> WirError {
        WirError::at(self.st.line, &self.st.text, msg)
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }

    fn peek_line(&self) -> usize {
        self.toks.get(self.pos).map_or(self.st.line, |t| t.line)
    }

    fn found(&self) -> String {
        match self.toks.get(self.pos) {
            Some(t) => format!("{} (column {})", t.tok.describe(), t.col),
            None => "end of statement".into(),
        }
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).map(|t| t.tok.clone());
        self.pos += 1;
        t
    }

    fn at_end(&self) -> bool {
        self.pos >= self.toks.len()
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }

    fn expect(&mut self, want: Tok) -> Result<()> {
        if self.peek() == Some(&want) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(format!("expected {}, found {}", want.describe(), self.found())))
        }
    }

    fn keyword(&mut self, kw: &str) -> Result<()> {
        if self.is_kw(kw) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(format!("expected `{kw}`, found {}", self.found())))
        }
    }

    fn name(&mut self) -> Result<String> {
        match self.peek() {
            Some(Tok::Ident(s)) if !KEYWORDS.contains(&s.as_str()) => {
                let s = s.clone();
                self.pos += 1;
                Ok(s)
            }
            Some(Tok::Ident(s)) => Err(self.err(format!("`{s}` is a keyword and cannot be used as a name"))),
            _ => Err(self.err(format!("expected a name, found {}", self.found()))),
        }
    }

    /// `[+|-] NUM`
    fn number(&mut self) -> Result<f64> {
        let sign = match self.peek() {
            Some(Tok::Plus) => {
                self.pos += 1;
                1.0
            }
            Some(Tok::Minus) => {
                self.pos += 1;
                -1.0
            }
            _ => 1.0,
        };
        match self.peek() {
            Some(Tok::Num(x)) => {
                let x = *x;
                self.pos += 1;
                Ok(sign * x)
            }
            _ => Err(self.err(format!("expected a number, found {}", self.found()))),
        }
    }

    fn number_follows(&self) -> bool {
        match self.peek() {
            Some(Tok::Num(_)) => true,
            Some(Tok::Plus | Tok::Minus) => matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Num(_))),
            _ => false,
        }
    }

    /// `( NUM {, NUM} )`, or a bare number as a 1D point.
    fn point(&mut self) -> Result<Vec<f64>> {
        if self.peek() != Some(&Tok::LParen) {
            if self.number_follows() {
                return Ok(vec![self.number()?]);
            }
            return Err(self.err(format!("expected a point like (1, 0), found {}", self.found())));
        }
        self.pos += 1;
        let mut v = vec![self.number()?];
        while self.peek() == Some(&Tok::Comma) {
            self.pos += 1;
            v.push(self.number()?);
        }
        self.expect(Tok::RParen)?;
        Ok(v)
    }

    fn dim(&mut self) -> Result<usize> {
        match self.next() {
            Some(Tok::Ident(s)) if s.len() > 1 && s.starts_with('R') && s[1..].chars().all(|c| c.is_ascii_digit()) => {
                let d: usize = s[1..].parse().map_err(|_| self.err("dimension out of range"))?;
                if d == 0 {
                    return Err(self.err("dimension must be at least 1 (R1)"));
                }
                Ok(d)
            }
            _ => {
                self.pos -= 1;
                Err(self.err(format!("expected a space like R2, found {}", self.found())))
            }
        }
    }

    fn finish(&self) -> Result<()> {
        if self.at_end() {
            Ok(())
        } else {
            Err(self.err(format!("unexpected {} after the end of the statement", self.found())))
        }
    }

    fn statement(&mut self) -> Result<Stmt> {
        let kind = match self.peek() {
            Some(Tok::Ident(s)) => match s.as_str() {
                "wir" => {
                    self.pos += 1;
                    StmtKind::Header { version: self.number()? }
                }
                "input" | "output" => {
                    let is_input = s == "input";
                    self.pos += 1;
                    let name = self.name()?;
                    self.expect(Tok::Colon)?;
                    let dim = self.dim()?;
                    if is_input { StmtKind::Input { name, dim } } else { StmtKind::Output { name, dim } }
                }
                "crease" => {
                    self.pos += 1;
                    self.crease()?
                }
                "assert" => {
                    self.pos += 1;
                    self.assert()?
                }
                _ => self.readout()?,
            },
            _ => return Err(self.err(format!("expected a statement, found {}", self.found()))),
        };
        self.finish()?;
        Ok(Stmt { kind, line: self.st.line, text: self.st.text.clone() })
    }

    fn crease(&mut self) -> Result<StmtKind> {
        let name = self.name()?;
        self.expect(Tok::Colon)?;
        let plane = if self.is_kw("zero") {
            self.pos += 1;
            self.keyword("on")?;
            if self.is_kw("line") {
                self.pos += 1;
                let p = self.point()?;
                self.expect(Tok::Arrow)?;
                let q = self.point()?;
                PlaneSpec::Line { p, q }
            } else if self.is_kw("plane") {
                self.pos += 1;
                self.keyword("normal")?;
                let v = self.point()?;
                if self.is_kw("through") {
                    self.pos += 1;
                    PlaneSpec::NormalThrough { v, p: self.point()? }
                } else {
                    self.keyword("offset")?;
                    PlaneSpec::Normal { v, offset: self.number()? }
                }
            } else {
                return Err(self.err(format!("expected `line` or `plane` after `zero on`, found {}", self.found())));
            }
        } else if self.is_kw("between") {
            self.pos += 1;
            let p = self.point()?;
            self.keyword("and")?;
            let q = self.point()?;
            PlaneSpec::Between { p, q }
        } else {
            return Err(self.err(format!(
                "expected a plane (`zero on line`, `zero on plane normal`, `between`), found {}",
                self.found()
            )));
        };
        self.expect(Tok::Comma)?;
        self.keyword("equals")?;
        let value = self.number()?;
        self.keyword("at")?;
        let at = self.point()?;
        Ok(StmtKind::Crease { name, plane, anchor: Anchor { value, at } })
    }

    fn assert(&mut self) -> Result<StmtKind> {
        let mut checks = Vec::new();
        loop {
            let output = self.name()?;
            let point = if self.peek() == Some(&Tok::LParen) {
                self.point()?
            } else {
                return Err(self.err(format!("expected `(` after `{output}`, found {}", self.found())));
            };
            self.expect(Tok::EqEq)?;
            let value = self.number()?;
            checks.push(Check { output, point, value });
            if self.peek() == Some(&Tok::Comma) {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(StmtKind::Assert { checks })
    }

    fn readout(&mut self) -> Result<StmtKind> {
        let target = self.name()?;
        self.expect(Tok::Eq)?;
        if self.is_kw("solve") {
            self.pos += 1;
            self.keyword("over")?;
            self.expect(Tok::LBrace)?;
            let mut over = vec![self.name()?];
            while self.peek() == Some(&Tok::Comma) {
                self.pos += 1;
                over.push(self.name()?);
            }
            self.expect(Tok::RBrace)?;
            self.expect(Tok::Colon)?;
            let mut pairs = Vec::new();
            loop {
                let p = self.point()?;
                self.expect(Tok::Arrow)?;
                pairs.push((p, self.number()?));
                if self.peek() == Some(&Tok::Comma) {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            return Ok(StmtKind::Solve { target, over, pairs });
        }
        let mut terms = Vec::new();
        let mut sign = 1.0;
        loop {
            terms.push(self.term(sign)?);
            match self.peek() {
                Some(Tok::Plus) => sign = 1.0,
                Some(Tok::Minus) => sign = -1.0,
                _ => break,
            }
            self.pos += 1;
        }
        Ok(StmtKind::Readout { target, terms })
    }

    /// `[NUM "*"] (NAME | hinge) | NUM`; the first term may carry a sign.
    fn term(&mut self, sign: f64) -> Result<Term> {
        let line = self.peek_line();
        let mut coef = sign;
        if self.number_follows() {
            coef *= self.number()?;
            if self.peek() != Some(&Tok::Star) {
                return Ok(Term { coef, kind: TermKind::Const, line });
            }
            self.pos += 1;
        } else if matches!(self.peek(), Some(Tok::Minus)) {
            // Leading `-name` / `-hinge ...`.
            self.pos += 1;
            coef = -coef;
        }
        if self.is_kw("hinge") {
            self.pos += 1;
            let left = self.is_kw("left");
            if left {
                self.pos += 1;
            }
            self.keyword("at")?;
            let at = self.number()?;
            self.keyword("turn")?;
            let turn = self.number()?;
            return Ok(Term { coef, kind: TermKind::Hinge { left, at, turn }, line });
        }
        Ok(Term { coef, kind: TermKind::Name(self.name()?), line })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xor() {
        let src = "wir 0.1\ninput  x: R2\noutput y: R1\n\
                   crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)\n\
                   y = a - 2*b\nassert y(0,0)==0, y(1,1)==0\n";
        let s = parse(src).unwrap();
        assert_eq!(s.stmts.len(), 6);
        match &s.stmts[4].kind {
            StmtKind::Readout { terms, .. } => {
                assert_eq!(terms[0].coef, 1.0);
                assert_eq!(terms[1].coef, -2.0);
                assert_eq!(terms[1].kind, TermKind::Name("b".into()));
            }
            k => panic!("{k:?}"),
        }
    }

    #[test]
    fn signed_hinge_turns() {
        let s = parse("y = -hinge left at -1 turn -2 + 3 - 0.5*hinge at 1/2 turn +4").unwrap();
        let StmtKind::Readout { terms, .. } = &s.stmts[0].kind else { panic!() };
        assert_eq!(terms[0], Term { coef: -1.0, kind: TermKind::Hinge { left: true, at: -1.0, turn: -2.0 }, line: 1 });
        assert_eq!(terms[1].kind, TermKind::Const);
        assert_eq!(terms[1].coef, 3.0);
        assert_eq!(terms[2], Term { coef: -0.5, kind: TermKind::Hinge { left: false, at: 0.5, turn: 4.0 }, line: 1 });
    }

    #[test]
    fn reports_line_and_statement() {
        let e = parse("wir 0.1\ncrease a: zero on circle (0,0)").unwrap_err();
        assert_eq!(e.line, Some(2));
        assert!(e.to_string().contains("crease a: zero on circle"), "{e}");
    }
}
