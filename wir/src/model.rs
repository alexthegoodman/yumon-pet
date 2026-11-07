//! The core data model shared by the compiler, printer and decompiler.
//!
//! A `Program` denotes y_o(x) = sum_i c_{o,i} * h_i(x) + bias_o, where a crease is
//! h(x) = relu(w . x + b) and a gate (SwiGLU unit) is h(x) = silu(w . x + b) * (u . x + c).
//! Each `Unit` stores its raw first-layer row (w, b) so that geometry derived
//! from simple points compiles to exact weights; the geometric view the spec
//! talks about (unit normal, offset, slope) is derived by accessors:
//!
//!   slope s = |w|,  normal n = w / s,  offset d = -b / s
//!
//! In canonical form every unit has s = 1, so w *is* the unit normal and b = -d.

use crate::codebook::Codes;
use crate::error::{Result, WirError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    /// Printed as a `crease` declaration and referenced by name.
    Crease,
    /// 1D, single-output only: printed inline as `hinge [left] at k turn t`.
    Hinge,
    /// A SwiGLU unit, printed as a `gate` declaration; its `up` factor is set.
    Gate,
}

/// The linear factor of a gate: u . x + c.
#[derive(Debug, Clone, PartialEq)]
pub struct Up {
    pub u: Vec<f64>,
    pub c: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    pub name: String,
    pub kind: UnitKind,
    pub w: Vec<f64>,
    pub b: f64,
    /// Readout coefficient per output.
    pub coeffs: Vec<f64>,
    /// Gates only: the linear factor multiplying silu(w . x + b).
    pub up: Option<Up>,
    /// Source line that introduced the unit, if any.
    pub line: Option<usize>,
}

impl Unit {
    pub fn slope(&self) -> f64 {
        norm(&self.w)
    }

    pub fn normal(&self) -> Vec<f64> {
        let s = self.slope();
        self.w.iter().map(|x| x / s).collect()
    }

    pub fn offset(&self) -> f64 {
        -self.b / self.slope()
    }

    pub fn eval(&self, x: &[f64]) -> f64 {
        let z = dot(&self.w, x) + self.b;
        match &self.up {
            None => z.max(0.0),
            Some(up) => silu(z) * (dot(&up.u, x) + up.c),
        }
    }
}

pub fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub name: String,
    pub bias: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssertCheck {
    pub output: usize,
    pub point: Vec<f64>,
    pub value: f64,
    pub line: Option<usize>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub input: String,
    pub dim: usize,
    /// Optional tensor shape of the input (`R64x3x3`); its product is `dim`.
    pub shape: Option<Vec<usize>>,
    /// Optional box domain `in [lo, hi]` on every coordinate.
    pub domain: Option<(f64, f64)>,
    pub outputs: Vec<Output>,
    pub units: Vec<Unit>,
    pub asserts: Vec<AssertCheck>,
    /// Vectors written as codebook words (`{...}`), so they print back as words.
    pub codes: Option<Codes>,
}

impl Program {
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let mut y: Vec<f64> = self.outputs.iter().map(|o| o.bias).collect();
        for u in &self.units {
            let h = u.eval(x);
            if h != 0.0 {
                for (yo, c) in y.iter_mut().zip(&u.coeffs) {
                    *yo += c * h;
                }
            }
        }
        y
    }

    /// Weights of a ReLU program, or of a SwiGLU program when every unit is a gate
    /// (the compiler rejects mixtures).
    pub fn to_weights(&self) -> Weights {
        let h = self.units.len();
        let o = self.outputs.len();
        let mut w1 = Mat::zeros(h, self.dim);
        let mut w2 = Mat::zeros(o, h);
        let gated = !self.units.is_empty() && self.units.iter().all(|u| u.up.is_some());
        let mut w3 = Mat::zeros(h, self.dim);
        for (i, u) in self.units.iter().enumerate() {
            w1.row_mut(i).copy_from_slice(&u.w);
            for (k, c) in u.coeffs.iter().enumerate() {
                w2.set(k, i, *c);
            }
            if let Some(up) = &u.up {
                w3.row_mut(i).copy_from_slice(&up.u);
            }
        }
        let up = gated.then(|| (w3, self.units.iter().map(|u| u.up.as_ref().unwrap().c).collect()));
        Weights {
            w1,
            b1: self.units.iter().map(|u| u.b).collect(),
            w2,
            b2: self.outputs.iter().map(|o| o.bias).collect(),
            up,
            input: Some(self.input.clone()),
            outputs: Some(self.outputs.iter().map(|o| o.name.clone()).collect()),
            shape: self.shape.clone(),
        }
    }
}

/// Row-major dense matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let cols = rows.first().map_or(0, |r| r.len());
        if rows.iter().any(|r| r.len() != cols) {
            return Err(WirError::new("ragged matrix rows"));
        }
        Ok(Self { rows: rows.len(), cols, data: rows.concat() })
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        (0..self.rows).map(|r| self.row(r).to_vec()).collect()
    }
}

/// One-hidden-layer network in PyTorch `nn.Linear` layout: W1 [H, D], b1 [H],
/// W2 [O, H], b2 [O]. Hidden units are relu(W1 x + b1), or with `up` = (W3 [H, D],
/// b3 [H]) SwiGLU units silu(W1 x + b1) * (W3 x + b3).
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    pub w1: Mat,
    pub b1: Vec<f64>,
    pub w2: Mat,
    pub b2: Vec<f64>,
    /// SwiGLU up projection (W3, b3); `None` for ReLU.
    pub up: Option<(Mat, Vec<f64>)>,
    /// Optional names carried through exports so decompiling keeps them.
    pub input: Option<String>,
    pub outputs: Option<Vec<String>>,
    /// Optional input tensor shape (e.g. [64, 3, 3] for a conv patch).
    pub shape: Option<Vec<usize>>,
}

impl Weights {
    pub fn hidden(&self) -> usize {
        self.w1.rows
    }

    pub fn input_dim(&self) -> usize {
        self.w1.cols
    }

    pub fn output_dim(&self) -> usize {
        self.w2.rows
    }

    pub fn param_count(&self) -> usize {
        let up = self.up.as_ref().map_or(0, |(w3, b3)| w3.data.len() + b3.len());
        self.w1.data.len() + self.b1.len() + self.w2.data.len() + self.b2.len() + up
    }

    pub fn check_shapes(&self) -> Result<()> {
        let (h, d, o) = (self.hidden(), self.input_dim(), self.output_dim());
        if d == 0 {
            return Err(WirError::new("W1 has zero input columns"));
        }
        if self.b1.len() != h {
            return Err(WirError::new(format!("b1 has length {} but W1 has {h} rows", self.b1.len())));
        }
        if self.w2.cols != h {
            return Err(WirError::new(format!("W2 is [{o}, {}] but W1 has {h} hidden units", self.w2.cols)));
        }
        if self.b2.len() != o {
            return Err(WirError::new(format!("b2 has length {} but W2 has {o} rows", self.b2.len())));
        }
        if let Some(names) = &self.outputs
            && names.len() != o {
                return Err(WirError::new(format!("{} output names for {o} outputs", names.len())));
            }
        if let Some((w3, b3)) = &self.up
            && (w3.rows != h || w3.cols != d || b3.len() != h) {
                return Err(WirError::new(format!("W3/b3 must be [{h}, {d}] and [{h}] like W1/b1")));
            }
        if let Some(shape) = &self.shape
            && shape.iter().product::<usize>() != d {
                return Err(WirError::new(format!("input shape {shape:?} does not match {d} input columns")));
            }
        let up = self.up.iter().flat_map(|(w3, b3)| w3.data.iter().chain(b3));
        let finite = self.w1.data.iter().chain(&self.b1).chain(&self.w2.data).chain(&self.b2).chain(up).all(|x| x.is_finite());
        if !finite {
            return Err(WirError::new("weights contain NaN or infinity"));
        }
        Ok(())
    }

    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.b2.clone();
        for i in 0..self.hidden() {
            let z = dot(self.w1.row(i), x) + self.b1[i];
            let h = match &self.up {
                None => z.max(0.0),
                Some((w3, b3)) => silu(z) * (dot(w3.row(i), x) + b3[i]),
            };
            if h != 0.0 {
                for (k, yk) in y.iter_mut().enumerate() {
                    *yk += self.w2.get(k, i) * h;
                }
            }
        }
        y
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
