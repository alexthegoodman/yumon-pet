//! Source-level syntax tree: one node per statement, numbers as written.

#[derive(Debug, Clone, PartialEq)]
pub enum PlaneSpec {
    /// `zero on line P -> Q` (2D only).
    Line { p: Vec<f64>, q: Vec<f64> },
    /// `zero on plane normal V offset D`: the plane {x : (V/|V|)·x = D}.
    Normal { v: Vec<f64>, offset: f64 },
    /// `zero on plane normal V through P`: the plane through P perpendicular to V.
    NormalThrough { v: Vec<f64>, p: Vec<f64> },
    /// `between P and Q`: perpendicular bisector, rising toward Q.
    Between { p: Vec<f64>, q: Vec<f64> },
}

/// `equals VALUE at POINT`.
#[derive(Debug, Clone, PartialEq)]
pub struct Anchor {
    pub value: f64,
    pub at: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TermKind {
    /// A named crease, scaled by `coef`.
    Name(String),
    /// `hinge [left] at K turn T`, scaled by `coef`.
    Hinge { left: bool, at: f64, turn: f64 },
    /// A bare constant; the value is `coef`.
    Const,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub coef: f64,
    pub kind: TermKind,
    /// Line the term starts on (continuations span lines).
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    pub output: String,
    pub point: Vec<f64>,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Header { version: f64 },
    Input { name: String, dim: usize },
    Output { name: String, dim: usize },
    Crease { name: String, plane: PlaneSpec, anchor: Anchor },
    Readout { target: String, terms: Vec<Term> },
    Solve { target: String, over: Vec<String>, pairs: Vec<(Vec<f64>, f64)> },
    Assert { checks: Vec<Check> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub line: usize,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Source {
    pub stmts: Vec<Stmt>,
}

pub const KEYWORDS: &[&str] = &[
    "wir", "input", "output", "crease", "assert", "solve", "over", "hinge", "left", "at", "turn", "zero",
    "on", "line", "plane", "normal", "offset", "through", "between", "and", "equals",
];
