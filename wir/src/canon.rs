//! Canonical form and the decompiler.
//!
//! One function has many weight settings. The canonical form removes the two
//! v0 symmetries (positive per-unit scaling and permutation of hidden units)
//! and the redundancies (dead units, duplicated planes), so weights -> WIR is
//! deterministic. Both `canonicalize` (source path) and `decompile` (weights
//! path) end in `normal_form`, so the two paths meet in one routine.
//!
//! Scale is fixed by the max norm: every canonical unit has max_i |w_i| = 1
//! (the spec proposes |w| = 1). A unit-length normal is almost never an exact
//! decimal, so 6-digit text could not carry "slope 1" exactly and the error
//! leaked into readout coefficients. A max-normalized normal has one component
//! that is exactly +-1, the printer anchors one step along that axis, and every
//! canonical crease reads `equals 1` with exact coefficients (XOR decompiles to
//! `y = c1 - 2*c2`). In 1D both norms agree.
//!
//! Tolerances are relative with a floor of 1: a quantity counts as zero when
//! |x| <= eps * max(1, largest |x| of its kind in the network).
//!
//! Gates (SwiGLU units) keep their silu argument as is (silu has no scaling
//! symmetry); their up vector is scaled so its largest-magnitude entry is +1, with
//! the factor (and sign) moved into the readout. Gates are never merged.
//!
//! Domains: with a box domain (`in [lo, hi]`) or decompile-time sample inputs,
//! creases that are never active there are dropped (the function on the domain is
//! unchanged).

use crate::ast::{patch_profile, PATCH_BASIS};
use crate::compiler::box_max;
use crate::error::{Result, WirError};
use crate::model::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerances {
    /// A unit with |w| below this (relative) outputs a constant and is folded into the bias.
    pub dead_weight: f64,
    /// A unit whose readout coefficients are all below this (relative) is dropped.
    pub dead_coeff: f64,
    /// Units whose unit normals and offsets agree within this are merged.
    pub merge: f64,
    /// Grid used when ordering normals and offsets, so float noise cannot reorder units.
    pub sort_grid: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self { dead_weight: 1e-9, dead_coeff: 1e-9, merge: 1e-7, sort_grid: 1e-6 }
    }
}

/// Canonical form of a source-level program. Asserts are dropped: the
/// canonical form denotes only the function, and the decompiler never
/// invents asserts.
pub fn canonicalize(p: &Program, tol: Tolerances) -> Program {
    let mut units = p.units.clone();
    if let Some((lo, hi)) = p.domain {
        units.retain(|u| u.up.is_some() || box_max(&u.w, u.b, lo, hi) > 0.0);
    }
    let mut out = normal_form(p.input.clone(), p.dim, p.outputs.clone(), units, tol);
    out.shape = p.shape.clone();
    out.domain = p.domain;
    out
}

#[derive(Debug, Clone, Default)]
pub struct DecompileOptions {
    pub tolerances: Tolerances,
    /// Overrides the input name carried by the weights (default `x`).
    pub input: Option<String>,
    /// Overrides output names (default `y`, or `y1..yO` for vector outputs).
    pub outputs: Option<Vec<String>>,
    /// Points at which to emit `assert` lines with the network's values.
    pub assert_at: Vec<Vec<f64>>,
    /// Sample inputs the network is used on: creases never active on all of them are dropped.
    pub domain_samples: Option<Vec<Vec<f64>>>,
    /// Input shape override (else the weights' own); `R<c>x3x3` prints `patch` directions.
    pub shape: Option<Vec<usize>>,
    /// Keep only the first N of the 9 patch components (lossy; needs a `<c>x3x3` shape).
    pub kernel_terms: Option<usize>,
}

/// Project every 3x3 kernel of `v` (laid out channel-major, `c*9 + row*3 + col`) onto
/// the first `n` patch basis components.
pub fn project_patch(v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (src, dst) in v.chunks(9).zip(out.chunks_mut(9)) {
        for &(_, i, j) in &PATCH_BASIS[..n.min(9)] {
            let basis = |r: usize, c: usize| patch_profile(i, c) * patch_profile(j, r);
            let (mut num, mut den) = (0.0, 0.0);
            for r in 0..3 {
                for c in 0..3 {
                    num += src[r * 3 + c] * basis(r, c);
                    den += basis(r, c) * basis(r, c);
                }
            }
            for r in 0..3 {
                for c in 0..3 {
                    dst[r * 3 + c] += num / den * basis(r, c);
                }
            }
        }
    }
    out
}

/// Weight tensors -> canonical `Program`.
pub fn decompile(w: &Weights, opts: &DecompileOptions) -> Result<Program> {
    w.check_shapes()?;
    let (h, d, o) = (w.hidden(), w.input_dim(), w.output_dim());
    let input = opts.input.clone().or_else(|| w.input.clone()).unwrap_or_else(|| "x".into());
    let names = opts.outputs.clone().or_else(|| w.outputs.clone()).unwrap_or_else(|| {
        if o == 1 { vec!["y".into()] } else { (1..=o).map(|k| format!("y{k}")).collect() }
    });
    if names.len() != o {
        return Err(WirError::new(format!("{} output names given for {o} outputs", names.len())));
    }
    let outputs = names.into_iter().zip(&w.b2).map(|(name, &bias)| Output { name, bias }).collect();
    let shape = opts.shape.clone().or_else(|| w.shape.clone());
    if let Some(s) = &shape
        && s.iter().product::<usize>() != d {
            return Err(WirError::new(format!("shape {s:?} does not match {d} input columns")));
        }
    let project = match (opts.kernel_terms, shape.as_deref()) {
        (None, _) | (Some(9..), _) => None,
        (Some(n), Some(&[_, 3, 3])) => Some(n),
        (Some(_), _) => return Err(WirError::new("kernel terms need an input shape like <c>x3x3")),
    };
    let proj = |v: &[f64]| match project {
        Some(n) => project_patch(v, n),
        None => v.to_vec(),
    };
    let mut units: Vec<Unit> = (0..h)
        .map(|i| Unit {
            name: format!("u{}", i + 1),
            kind: if w.up.is_some() { UnitKind::Gate } else { UnitKind::Crease },
            w: proj(w.w1.row(i)),
            b: w.b1[i],
            coeffs: (0..o).map(|k| w.w2.get(k, i)).collect(),
            up: w.up.as_ref().map(|(w3, b3)| Up { u: proj(w3.row(i)), c: b3[i] }),
            line: None,
        })
        .collect();
    if let Some(xs) = &opts.domain_samples {
        if let Some(x) = xs.iter().find(|x| x.len() != d) {
            return Err(WirError::new(format!("domain sample has {} coordinates but the input is R{d}", x.len())));
        }
        units.retain(|u| u.up.is_some() || xs.iter().any(|x| dot(&u.w, x) + u.b > 0.0));
    }
    let mut p = normal_form(input, d, outputs, units, opts.tolerances);
    p.shape = shape;
    for pt in &opts.assert_at {
        if pt.len() != d {
            return Err(WirError::new(format!("assert point has {} coordinates but the input is R{d}", pt.len())));
        }
        let y = w.eval(pt);
        for (k, v) in y.into_iter().enumerate() {
            p.asserts.push(AssertCheck { output: k, point: pt.clone(), value: v, line: None, text: None });
        }
    }
    Ok(p)
}

fn normal_form(input: String, dim: usize, mut outputs: Vec<Output>, units: Vec<Unit>, tol: Tolerances) -> Program {
    let n_out = outputs.len();
    // 1. Fold constant units: relu(0.x + b) = relu(b).
    let w_scale = units.iter().map(|u| norm(&u.w)).fold(1.0, f64::max);
    let mut live = Vec::new();
    for u in units {
        if let Some(up) = &u.up {
            // Gates: scale the up vector so its largest-|.| entry is +1.
            let k = (0..up.u.len()).fold(0, |k, i| if up.u[i].abs() > up.u[k].abs() { i } else { k });
            let s = if up.u.is_empty() || up.u[k] == 0.0 { up.c } else { up.u[k] };
            if s == 0.0 {
                continue; // silu(.) * 0
            }
            live.push(Unit {
                coeffs: u.coeffs.iter().map(|c| c * s).collect(),
                up: Some(Up { u: up.u.iter().map(|x| x / s).collect(), c: up.c / s }),
                ..u
            });
            continue;
        }
        let s = norm(&u.w);
        if s <= tol.dead_weight * w_scale {
            let h = u.b.max(0.0);
            for (o, c) in outputs.iter_mut().zip(&u.coeffs) {
                o.bias += c * h;
            }
        } else {
            // 2. Normalize so the largest |w_i| is 1; the scale moves into the readout.
            let s = max_abs(&u.w);
            live.push(Unit {
                w: u.w.iter().map(|x| x / s).collect(),
                b: u.b / s,
                coeffs: u.coeffs.iter().map(|c| c * s).collect(),
                ..u
            });
        }
    }
    let c_scale = |units: &[Unit]| units.iter().flat_map(|u| u.coeffs.iter()).fold(1.0f64, |m, c| m.max(c.abs()));
    let dead = |u: &Unit, scale: f64| u.coeffs.iter().all(|c| c.abs() <= tol.dead_coeff * scale);
    // 3. Drop units that never reach an output.
    let scale = c_scale(&live);
    live.retain(|u| !dead(u, scale));
    // 4. Sort, then merge identical oriented planes (first in order keeps the geometry).
    sort_units(&mut live, tol);
    let mut merged: Vec<Unit> = Vec::new();
    for u in live {
        let same = merged.iter_mut().find(|m| {
            m.up.is_none() && u.up.is_none()
                && m.w.iter().zip(&u.w).all(|(a, b)| (a - b).abs() <= tol.merge)
                && (m.b - u.b).abs() <= tol.merge * m.b.abs().max(1.0)
        });
        match same {
            Some(m) => {
                for (a, b) in m.coeffs.iter_mut().zip(&u.coeffs) {
                    *a += b;
                }
            }
            None => merged.push(u),
        }
    }
    // 5. Merging can cancel units.
    let scale = c_scale(&merged);
    merged.retain(|u| !dead(u, scale));
    let kind = if dim == 1 && n_out == 1 { UnitKind::Hinge } else { UnitKind::Crease };
    for (i, u) in merged.iter_mut().enumerate() {
        u.name = format!("c{}", i + 1);
        u.kind = if u.up.is_some() { UnitKind::Gate } else { kind };
        u.line = None;
    }
    Program { input, dim, shape: None, domain: None, outputs, units: merged, asserts: Vec::new(), codes: None }
}

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Lexicographic by (max-normalized) normal, then by offset -b, both on a grid so that
/// float noise cannot swap units; exact values break remaining ties.
fn sort_units(units: &mut [Unit], tol: Tolerances) {
    let q = |x: f64| (x / tol.sort_grid).round() as i64;
    let up = |u: &Unit| -> Vec<f64> { u.up.as_ref().map_or(vec![], |g| g.u.iter().copied().chain([g.c]).collect()) };
    let exact = |u: &Unit| -> Vec<f64> {
        u.w.iter().copied().chain([-u.b]).chain(up(u)).chain(u.coeffs.iter().copied()).collect()
    };
    units.sort_by(|a, b| {
        let ka = a.w.iter().map(|x| q(*x)).chain([q(-a.b)]).chain(up(a).into_iter().map(q));
        let kb = b.w.iter().map(|x| q(*x)).chain([q(-b.b)]).chain(up(b).into_iter().map(q));
        ka.cmp(kb).then_with(|| {
            let (ea, eb) = (exact(a), exact(b));
            ea.iter().zip(&eb).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal)
        })
    });
}
