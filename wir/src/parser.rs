//! Recursive-descent parser: WIR text -> `ast::Source`.

use crate::ast::*;
use crate::codebook::{codebook, Codebook, Codes, NAMES};
use crate::error::{Result, WirError};
use crate::lexer::{split_statements, Statement, Tok, Token};

pub fn parse(src: &str) -> Result<Source> {
    let mut stmts = Vec::new();
    let mut codes: Option<Codes> = None;
    for st in split_statements(src)? {
        let mut c = Cursor::new(&st, codes.take());
        let stmt = c.statement()?;
        codes = c.codes;
        if let StmtKind::Codebook { name } = &stmt.kind {
            if codes.is_some() {
                return Err(WirError::at(st.line, &st.text, "a program names at most one codebook"));
            }
            let book = codebook(name).ok_or_else(|| {
                WirError::at(st.line, &st.text, format!("unknown codebook `{name}` (known: {})", NAMES.join(", ")))
            })?;
            codes = Some(Codes::new(book));
        }
        stmts.push(stmt);
    }
    Ok(Source { stmts, codes })
}

struct Cursor<'a> {
    st: &'a Statement,
    toks: &'a [Token],
    pos: usize,
    codes: Option<Codes>,
}

impl<'a> Cursor<'a> {
    fn new(st: &'a Statement, codes: Option<Codes>) -> Self {
        Self { st, toks: &st.tokens, pos: 0, codes }
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

    /// `[ NUM NUM ... ] [/ NUM]`: entries separated by spaces (commas allowed),
    /// all divided by the optional scale.
    fn vector(&mut self) -> Result<Vec<f64>> {
        if self.peek() == Some(&Tok::LBrace) {
            return self.coded();
        }
        self.expect(Tok::LBracket)?;
        let mut v = Vec::new();
        while self.peek() != Some(&Tok::RBracket) {
            if self.at_end() {
                return Err(self.err("unclosed `[` vector"));
            }
            v.push(self.number()?);
            if self.peek() == Some(&Tok::Comma) {
                self.pos += 1;
            }
        }
        self.pos += 1;
        if v.is_empty() {
            return Err(self.err("empty vector `[]`"));
        }
        if self.peek() == Some(&Tok::Slash) {
            self.pos += 1;
            let q = self.number()?;
            if !(q != 0.0 && q.is_finite()) {
                return Err(self.err("vector scale must be a finite non-zero number"));
            }
            v.iter_mut().for_each(|x| *x /= q);
        }
        Ok(v)
    }

    /// `{ WORD WORD ... } [/ NUM]`: each word's codebook entry, end to end, divided
    /// by the optional scale.
    fn coded(&mut self) -> Result<Vec<f64>> {
        self.expect(Tok::LBrace)?;
        let Some(book): Option<&'static Codebook> = self.codes.as_ref().map(|c| c.book) else {
            return Err(self.err("`{...}` vectors need a `codebook` line after the header"));
        };
        let mut idx = Vec::new();
        loop {
            match self.next() {
                Some(Tok::RBrace) => break,
                Some(Tok::Ident(w)) => match book.index(&w) {
                    Some(i) => idx.push(i),
                    None => return Err(self.err(format!("`{w}` is not a word of codebook {}", book.name))),
                },
                None => return Err(self.err("unclosed `{` vector")),
                Some(t) => return Err(self.err(format!("expected a codebook word, found {}", t.describe()))),
            }
        }
        if idx.is_empty() {
            return Err(self.err("empty vector `{}`"));
        }
        let q = if self.peek() == Some(&Tok::Slash) {
            self.pos += 1;
            let q = self.number()?;
            if !(q != 0.0 && q.is_finite()) {
                return Err(self.err("vector scale must be a finite non-zero number"));
            }
            q
        } else {
            1.0
        };
        let v = book.expand(&idx, q);
        self.codes.as_mut().unwrap().insert(&v, q, idx);
        Ok(v)
    }

    /// A vector, or `patch NAME VEC {NAME VEC}` over the 3x3 basis.
    fn direction(&mut self) -> Result<Direction> {
        if !self.is_kw("patch") {
            return Ok(Direction::Flat(self.point()?));
        }
        self.pos += 1;
        let mut parts: Vec<(String, Vec<f64>)> = Vec::new();
        while let Some(Tok::Ident(n)) = self.peek() {
            let n = n.clone();
            if !PATCH_BASIS.iter().any(|b| b.0 == n) {
                break;
            }
            if parts.iter().any(|p| p.0 == n) {
                return Err(self.err(format!("patch component `{n}` appears twice")));
            }
            self.pos += 1;
            parts.push((n, self.vector()?));
        }
        if parts.is_empty() {
            let names: Vec<&str> = PATCH_BASIS.iter().map(|b| b.0).collect();
            return Err(self.err(format!("`patch` needs components ({}), found {}", names.join(", "), self.found())));
        }
        Ok(Direction::Patch(parts))
    }

    /// `along DIR [above NUM]`
    fn ramp(&mut self) -> Result<Ramp> {
        self.keyword("along")?;
        let along = self.direction()?;
        let above = if self.is_kw("above") {
            self.pos += 1;
            self.number()?
        } else {
            0.0
        };
        Ok(Ramp { along, above })
    }

    /// `( NUM {, NUM} )`, a `[...]` or `{...}` vector, or a bare number as a 1D point.
    fn point(&mut self) -> Result<Vec<f64>> {
        if matches!(self.peek(), Some(Tok::LBracket | Tok::LBrace)) {
            return self.vector();
        }
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

    /// `R<d>` or a shaped space `R<a>x<b>x...`: (dimension, shape if shaped).
    fn dim(&mut self) -> Result<(usize, Option<Vec<usize>>)> {
        match self.next() {
            Some(Tok::Ident(s))
                if s.len() > 1
                    && s.starts_with('R')
                    && s[1..].split('x').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())) =>
            {
                let shape: Vec<usize> = s[1..]
                    .split('x')
                    .map(|p| p.parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|_| self.err("dimension out of range"))?;
                let d: usize = shape.iter().product();
                if d == 0 {
                    return Err(self.err("dimension must be at least 1 (R1)"));
                }
                Ok((d, (shape.len() > 1).then_some(shape)))
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
                "codebook" => {
                    self.pos += 1;
                    match self.next() {
                        Some(Tok::Ident(name)) => StmtKind::Codebook { name },
                        _ => {
                            self.pos -= 1;
                            return Err(self.err(format!("expected a codebook name, found {}", self.found())));
                        }
                    }
                }
                "input" | "output" => {
                    let is_input = s == "input";
                    self.pos += 1;
                    let name = self.name()?;
                    let names = if !is_input && self.peek() == Some(&Tok::DotDot) {
                        self.pos += 1;
                        let last = self.name()?;
                        self.name_range(&name, &last)?
                    } else {
                        vec![name.clone()]
                    };
                    self.expect(Tok::Colon)?;
                    let (dim, shape) = self.dim()?;
                    if is_input {
                        let domain = if self.is_kw("in") {
                            self.pos += 1;
                            self.expect(Tok::LBracket)?;
                            let lo = self.number()?;
                            self.expect(Tok::Comma)?;
                            let hi = self.number()?;
                            self.expect(Tok::RBracket)?;
                            if !(lo < hi) {
                                return Err(self.err("domain `in [lo, hi]` needs lo < hi"));
                            }
                            Some((lo, hi))
                        } else {
                            None
                        };
                        StmtKind::Input { name, dim, shape, domain }
                    } else {
                        if shape.is_some() {
                            return Err(self.err("outputs are scalar in v0; declare `output <name>: R1` components"));
                        }
                        StmtKind::Output { names, dim }
                    }
                }
                "crease" => {
                    self.pos += 1;
                    self.crease()?
                }
                "gate" => {
                    self.pos += 1;
                    let name = self.name()?;
                    self.expect(Tok::Colon)?;
                    self.keyword("silu")?;
                    let gate = self.ramp()?;
                    self.expect(Tok::Comma)?;
                    self.keyword("times")?;
                    let up = self.ramp()?;
                    StmtKind::Gate { name, gate, up }
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
        if self.is_kw("ramp") {
            self.pos += 1;
            return Ok(StmtKind::RampCrease { name, ramp: self.ramp()? });
        }
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
                "expected a plane (`zero on line`, `zero on plane normal`, `between`) or `ramp along`, found {}",
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
        if matches!(self.peek(), Some(Tok::LBracket | Tok::LBrace)) {
            let coefs = self.vector()?;
            self.expect(Tok::Dot)?;
            self.expect(Tok::LParen)?;
            let mut names = Vec::new();
            loop {
                let first = self.name()?;
                if self.peek() == Some(&Tok::DotDot) {
                    self.pos += 1;
                    let last = self.name()?;
                    names.extend(self.name_range(&first, &last)?);
                } else {
                    names.push(first);
                }
                if self.peek() == Some(&Tok::Comma) {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
            let coefs = match self.codes.as_mut().and_then(|c| c.truncate(&coefs, names.len())) {
                Some(cut) => cut,
                None => coefs,
            };
            if coefs.len() != names.len() {
                return Err(self.err(format!("{} coefficients for {} names", coefs.len(), names.len())));
            }
            return Ok(Term { coef, kind: TermKind::Dot { coefs, names }, line });
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

impl Cursor<'_> {
    /// `c1..c64` -> c1, c2, ..., c64 (same prefix, ascending numbers).
    fn name_range(&self, first: &str, last: &str) -> Result<Vec<String>> {
        let split = |n: &str| {
            let i = n.trim_end_matches(|c: char| c.is_ascii_digit()).len();
            (n[..i].to_string(), n[i..].parse::<usize>().ok())
        };
        match (split(first), split(last)) {
            ((p, Some(a)), (q, Some(b))) if p == q && a <= b => Ok((a..=b).map(|i| format!("{p}{i}")).collect()),
            _ => Err(self.err(format!("`{first}..{last}` is not a range like c1..c64"))),
        }
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
