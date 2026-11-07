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

/// A direction in input space: a vector literal, or for `R<c>x3x3` inputs a
/// `patch` of named 3x3 basis components, each a per-channel vector.
#[derive(Debug, Clone, PartialEq)]
pub enum Direction {
    Flat(Vec<f64>),
    Patch(Vec<(String, Vec<f64>)>),
}

/// `along DIR [above E]`: the linear form DIR . x - E.
#[derive(Debug, Clone, PartialEq)]
pub struct Ramp {
    pub along: Direction,
    pub above: f64,
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
    /// `[c1 c2 ...] . (a, b..d)`: one coefficient per named unit, scaled by `coef`.
    Dot { coefs: Vec<f64>, names: Vec<String> },
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
    Input { name: String, dim: usize, shape: Option<Vec<usize>>, domain: Option<(f64, f64)> },
    Output { name: String, dim: usize },
    Crease { name: String, plane: PlaneSpec, anchor: Anchor },
    /// `crease NAME: ramp along DIR [above E]`: relu(DIR . x - E).
    RampCrease { name: String, ramp: Ramp },
    /// `gate NAME: silu along DIR [above E], times along DIR [above E]`.
    Gate { name: String, gate: Ramp, up: Ramp },
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
    "on", "line", "plane", "normal", "offset", "through", "between", "and", "equals", "ramp", "along",
    "above", "gate", "silu", "times", "patch", "in",
];

/// Names of the 3x3 patch basis, in canonical order. Component (i, j) is
/// p_i(column) * p_j(row) with p_0 = (1, 1, 1), p_1 = (-1, 0, 1), p_2 = (1, -2, 1):
/// level, left-to-right slope, top-to-bottom slope, then curvatures. The nine are
/// orthogonal, so any 3x3 kernel has exactly one patch form.
pub const PATCH_BASIS: [(&str, usize, usize); 9] = [
    ("level", 0, 0),
    ("dx", 1, 0),
    ("dy", 0, 1),
    ("dxx", 2, 0),
    ("dyy", 0, 2),
    ("dxy", 1, 1),
    ("dxxy", 2, 1),
    ("dxyy", 1, 2),
    ("dxxyy", 2, 2),
];

pub fn patch_profile(i: usize, t: usize) -> f64 {
    [[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, -2.0, 1.0]][i][t]
}
