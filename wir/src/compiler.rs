//! WIR source -> `Program` -> weights.
//!
//! Plane forms reduce to a raw normal n and offset e (plane n.x = e, n not
//! normalized). The anchor `equals v at A` then gives delta = n.A - e and the
//! unit w = (v / delta) n, b = -(v / delta) e. That is the spec's
//! (n_hat, d, s) construction with the orientation flip folded into the sign of
//! v / delta, and it keeps weights exact when points are simple (XOR compiles
//! to W1 = [[1,1],[1,1]], not 1.0000000000000002).

use crate::ast::*;
use crate::error::{Result, Warning, WirError};
use crate::linalg::lstsq;
use crate::model::*;
use crate::num::{format_num, NumFormat};
use crate::parser::parse;

pub const SUPPORTED_VERSION: f64 = 0.1;

#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    /// Tolerance for `assert` checks and `solve` residuals.
    pub tolerance: f64,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self { tolerance: 1e-6 }
    }
}

/// Hidden units and parameters attributed to one statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CostLine {
    pub line: usize,
    pub text: String,
    pub units: usize,
    pub params: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveReport {
    pub line: usize,
    pub output: String,
    pub max_residual: f64,
    pub rank: usize,
    pub columns: usize,
}

#[derive(Debug, Clone)]
pub struct Lowered {
    pub program: Program,
    pub costs: Vec<CostLine>,
    pub solves: Vec<SolveReport>,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone)]
pub struct Compiled {
    pub program: Program,
    pub weights: Weights,
    pub costs: Vec<CostLine>,
    pub solves: Vec<SolveReport>,
    pub warnings: Vec<Warning>,
}

/// Parse, lower, build weights and check every assert.
pub fn compile(src: &str, opts: CompileOptions) -> Result<Compiled> {
    let lowered = lower(&parse(src)?, opts)?;
    let weights = lowered.program.to_weights();
    check_asserts(&lowered.program, &weights, opts.tolerance)?;
    Ok(Compiled {
        program: lowered.program,
        weights,
        costs: lowered.costs,
        solves: lowered.solves,
        warnings: lowered.warnings,
    })
}

/// Parse and lower without checking asserts.
pub fn lower_str(src: &str) -> Result<Program> {
    Ok(lower(&parse(src)?, CompileOptions::default())?.program)
}

pub fn check_asserts(program: &Program, weights: &Weights, tol: f64) -> Result<()> {
    for a in &program.asserts {
        let got = weights.eval(&a.point)[a.output];
        let diff = (got - a.value).abs();
        if !(diff <= tol) {
            let name = &program.outputs[a.output].name;
            let pt = fmt_point(&a.point);
            let msg = format!(
                "assert failed: {name}({pt}) = {} but expected {} (|diff| {:.3e} > tolerance {:.0e})",
                format_num(got, NumFormat::CANONICAL),
                format_num(a.value, NumFormat::CANONICAL),
                diff,
                tol
            );
            return Err(match (a.line, &a.text) {
                (Some(l), Some(t)) => WirError::at(l, t, msg),
                _ => WirError::new(msg),
            });
        }
    }
    Ok(())
}

fn fmt_point(p: &[f64]) -> String {
    p.iter().map(|x| format_num(*x, NumFormat::CANONICAL)).collect::<Vec<_>>().join(", ")
}

struct CreaseInfo {
    unit: usize,
    used: bool,
    cost: usize,
}

pub fn lower(src: &Source, opts: CompileOptions) -> Result<Lowered> {
    let mut stmts = src.stmts.iter();
    match stmts.next() {
        Some(Stmt { kind: StmtKind::Header { version }, line, text }) => {
            if (*version - SUPPORTED_VERSION).abs() > 1e-12 {
                return Err(WirError::at(*line, text, format!("unsupported WIR version {version}; this compiler reads wir 0.1")));
            }
        }
        Some(s) => return Err(WirError::at(s.line, &s.text, "a program must start with the header `wir 0.1`")),
        None => return Err(WirError::new("empty program: expected the header `wir 0.1`")),
    }

    // Declarations first: they fix D and the output list for everything after.
    let mut input: Option<(String, usize, Option<Vec<usize>>, Option<(f64, f64)>)> = None;
    let mut outputs: Vec<Output> = Vec::new();
    let mut body = Vec::new();
    for s in stmts {
        let err = |m: String| WirError::at(s.line, &s.text, m);
        match &s.kind {
            StmtKind::Header { .. } => return Err(err("duplicate `wir` header".into())),
            StmtKind::Input { name, dim, shape, domain } => {
                if !body.is_empty() {
                    return Err(err("declarations must come before creases, readouts and asserts".into()));
                }
                if input.is_some() {
                    return Err(err("v0 takes exactly one input; use one vector input like `input x: R3`".into()));
                }
                input = Some((name.clone(), *dim, shape.clone(), *domain));
            }
            StmtKind::Output { name, dim } => {
                if !body.is_empty() {
                    return Err(err("declarations must come before creases, readouts and asserts".into()));
                }
                if *dim != 1 {
                    return Err(err(format!(
                        "outputs are scalar in v0; declare {dim} separate `output <name>: R1` components"
                    )));
                }
                if outputs.iter().any(|o| &o.name == name) || input.as_ref().is_some_and(|i| &i.0 == name) {
                    return Err(err(format!("`{name}` is already declared")));
                }
                outputs.push(Output { name: name.clone(), bias: 0.0 });
            }
            _ => body.push(s),
        }
    }
    let Some((input_name, dim, shape, domain)) = input else {
        return Err(WirError::new("missing input declaration, e.g. `input x: R2`"));
    };
    if outputs.is_empty() {
        return Err(WirError::new("missing output declaration, e.g. `output y: R1`"));
    }
    let n_out = outputs.len();

    let mut units: Vec<Unit> = Vec::new();
    let mut creases: Vec<(String, CreaseInfo)> = Vec::new();
    let mut defined = vec![false; n_out];
    let mut costs: Vec<CostLine> = Vec::new();
    let mut solves = Vec::new();
    let mut warnings = Vec::new();
    let mut asserts = Vec::new();
    let unit_params = dim + 1 + n_out;

    let check_dim = |p: &[f64], what: &str, s: &Stmt| -> Result<()> {
        if p.len() != dim {
            return Err(WirError::at(
                s.line,
                &s.text,
                format!("{what} ({}) has {} coordinates but the input is R{dim}", fmt_point(p), p.len()),
            ));
        }
        if p.iter().any(|x| !x.is_finite()) {
            return Err(WirError::at(s.line, &s.text, format!("{what} has a non-finite coordinate")));
        }
        Ok(())
    };

    for s in body {
        let err = |m: String| WirError::at(s.line, &s.text, m);
        let taken = |name: &String, creases: &[(String, CreaseInfo)]| {
            name == &input_name || outputs.iter().any(|o| &o.name == name) || creases.iter().any(|c| &c.0 == name)
        };
        match &s.kind {
            StmtKind::RampCrease { name, ramp } | StmtKind::Gate { name, gate: ramp, .. } => {
                if taken(name, &creases) {
                    return Err(err(format!("`{name}` is already declared")));
                }
                let (w, b) = ramp_form(ramp, dim, shape.as_deref(), s)?;
                let (up, params) = match &s.kind {
                    StmtKind::Gate { up, .. } => {
                        let (u, c) = ramp_form(up, dim, shape.as_deref(), s)?;
                        (Some(Up { u, c }), 2 * dim + 2 + n_out)
                    }
                    _ => {
                        if w.iter().all(|x| *x == 0.0) {
                            return Err(err("ramp direction is the zero vector".into()));
                        }
                        (None, unit_params)
                    }
                };
                let kind = if up.is_some() { UnitKind::Gate } else { UnitKind::Crease };
                creases.push((name.clone(), CreaseInfo { unit: units.len(), used: false, cost: costs.len() }));
                units.push(Unit { name: name.clone(), kind, w, b, coeffs: vec![0.0; n_out], up, line: Some(s.line) });
                costs.push(CostLine { line: s.line, text: s.text.clone(), units: 1, params });
            }
            StmtKind::Crease { name, plane, anchor } => {
                if taken(name, &creases) {
                    return Err(err(format!("`{name}` is already declared")));
                }
                let (w, b) = crease_weights(plane, anchor, dim, &check_dim, s)?;
                creases.push((name.clone(), CreaseInfo { unit: units.len(), used: false, cost: costs.len() }));
                units.push(Unit {
                    name: name.clone(),
                    kind: UnitKind::Crease,
                    w,
                    b,
                    coeffs: vec![0.0; n_out],
                    up: None,
                    line: Some(s.line),
                });
                costs.push(CostLine { line: s.line, text: s.text.clone(), units: 1, params: unit_params });
            }
            StmtKind::Readout { target, terms } => {
                let o = output_index(&outputs, target, s)?;
                if std::mem::replace(&mut defined[o], true) {
                    return Err(err(format!("output `{target}` already has a readout")));
                }
                let mut hinges = 0;
                for t in terms {
                    match &t.kind {
                        TermKind::Const => outputs[o].bias += t.coef,
                        TermKind::Name(n) => {
                            let Some((_, info)) = creases.iter_mut().find(|c| &c.0 == n) else {
                                return Err(err(format!("`{n}` is not a declared crease (declare it before the readout)")));
                            };
                            info.used = true;
                            units[info.unit].coeffs[o] += t.coef;
                        }
                        TermKind::Dot { coefs, names } => {
                            for (n, c) in names.iter().zip(coefs) {
                                let Some((_, info)) = creases.iter_mut().find(|cr| &cr.0 == n) else {
                                    return Err(err(format!("`{n}` is not a declared crease or gate (declare it before the readout)")));
                                };
                                info.used = true;
                                units[info.unit].coeffs[o] += t.coef * c;
                            }
                        }
                        TermKind::Hinge { left, at, turn } => {
                            if dim != 1 {
                                return Err(err(format!("`hinge` is 1D sugar but the input is R{dim}; use a crease")));
                            }
                            if !at.is_finite() || !turn.is_finite() {
                                return Err(err("hinge position and turn must be finite".into()));
                            }
                            let (w, b) = if *left { (-1.0, *at) } else { (1.0, -*at) };
                            let mut coeffs = vec![0.0; n_out];
                            coeffs[o] = t.coef * turn;
                            let name = fresh_name(&units, &creases, &input_name, &outputs, "h");
                            units.push(Unit { name, kind: UnitKind::Hinge, w: vec![w], b, coeffs, up: None, line: Some(t.line) });
                            hinges += 1;
                        }
                    }
                }
                costs.push(CostLine {
                    line: s.line,
                    text: s.text.clone(),
                    units: hinges,
                    params: hinges * unit_params + 1,
                });
            }
            StmtKind::Solve { target, over, pairs } => {
                let o = output_index(&outputs, target, s)?;
                if std::mem::replace(&mut defined[o], true) {
                    return Err(err(format!("output `{target}` already has a readout")));
                }
                let mut cols = Vec::new();
                for n in over {
                    let Some((_, info)) = creases.iter_mut().find(|c| &c.0 == n) else {
                        return Err(err(format!("`{n}` is not a declared crease (declare it before the solve)")));
                    };
                    if cols.contains(&info.unit) {
                        return Err(err(format!("`{n}` appears twice in the solve set")));
                    }
                    info.used = true;
                    cols.push(info.unit);
                }
                for (p, v) in pairs {
                    check_dim(p, "solve point", s)?;
                    if !v.is_finite() {
                        return Err(err("solve target is not finite".into()));
                    }
                }
                let k = cols.len();
                let mut h = Mat::zeros(pairs.len(), k + 1);
                for (j, (p, _)) in pairs.iter().enumerate() {
                    for (i, &u) in cols.iter().enumerate() {
                        h.set(j, i, units[u].eval(p));
                    }
                    h.set(j, k, 1.0);
                }
                let targets: Vec<f64> = pairs.iter().map(|(_, v)| *v).collect();
                let sol = lstsq(&h, &targets);
                let mut worst = (0.0f64, 0usize);
                for (j, target) in targets.iter().enumerate() {
                    let fit: f64 = (0..=k).map(|i| h.get(j, i) * sol.x[i]).sum();
                    let r = (fit - target).abs();
                    if r > worst.0 || r.is_nan() {
                        worst = (r, j);
                    }
                }
                if !(worst.0 <= opts.tolerance) {
                    let (p, v) = &pairs[worst.1];
                    return Err(err(format!(
                        "solve cannot fit `{target}` with creases {{{}}}: max residual {:.3e} at ({}) -> {} exceeds tolerance {:.0e}; the creases cannot express the target",
                        over.join(", "),
                        worst.0,
                        fmt_point(p),
                        format_num(*v, NumFormat::CANONICAL),
                        opts.tolerance
                    )));
                }
                if sol.rank < k + 1 {
                    warnings.push(Warning {
                        line: Some(s.line),
                        message: format!(
                            "solve for `{target}` is rank-deficient (rank {} of {} columns incl. the constant); coefficients are the minimum-norm choice, not unique",
                            sol.rank,
                            k + 1
                        ),
                    });
                }
                for (i, &u) in cols.iter().enumerate() {
                    units[u].coeffs[o] += sol.x[i];
                }
                outputs[o].bias += sol.x[k];
                solves.push(SolveReport { line: s.line, output: target.clone(), max_residual: worst.0, rank: sol.rank, columns: k + 1 });
                costs.push(CostLine { line: s.line, text: s.text.clone(), units: 0, params: 1 });
            }
            StmtKind::Assert { checks } => {
                for c in checks {
                    let o = output_index(&outputs, &c.output, s)?;
                    check_dim(&c.point, "assert point", s)?;
                    asserts.push(AssertCheck {
                        output: o,
                        point: c.point.clone(),
                        value: c.value,
                        line: Some(s.line),
                        text: Some(s.text.clone()),
                    });
                }
            }
            StmtKind::Header { .. } | StmtKind::Input { .. } | StmtKind::Output { .. } => unreachable!(),
        }
    }

    if let Some(o) = defined.iter().position(|d| !d) {
        return Err(WirError::new(format!("output `{}` has no readout", outputs[o].name)));
    }

    // Drop unused creases (warning), flag used ones that contribute nothing.
    let mut keep = vec![true; units.len()];
    for (name, info) in &creases {
        if !info.used {
            keep[info.unit] = false;
            costs[info.cost].units = 0;
            costs[info.cost].params = 0;
            warnings.push(Warning {
                line: units[info.unit].line,
                message: format!("crease `{name}` is declared but never used; dropped"),
            });
        } else if units[info.unit].coeffs.iter().all(|c| *c == 0.0) {
            warnings.push(Warning {
                line: units[info.unit].line,
                message: format!("crease `{name}` has a zero readout coefficient and contributes nothing"),
            });
        }
    }
    let mut it = keep.iter();
    units.retain(|_| *it.next().unwrap());

    if units.iter().any(|u| u.up.is_some()) && units.iter().any(|u| u.up.is_none()) {
        return Err(WirError::new(
            "a program is either ReLU (creases, hinges) or SwiGLU (gates); this one mixes both",
        ));
    }
    if let Some((lo, hi)) = domain {
        for u in units.iter().filter(|u| u.up.is_none()) {
            if box_max(&u.w, u.b, lo, hi) <= 0.0 {
                warnings.push(Warning {
                    line: u.line,
                    message: format!("crease `{}` is never active on the domain [{lo}, {hi}]^{dim}", u.name),
                });
            }
        }
    }

    Ok(Lowered {
        program: Program { input: input_name, dim, shape, domain, outputs, units, asserts },
        costs,
        solves,
        warnings,
    })
}

fn output_index(outputs: &[Output], name: &str, s: &Stmt) -> Result<usize> {
    outputs
        .iter()
        .position(|o| o.name == name)
        .ok_or_else(|| WirError::at(s.line, &s.text, format!("`{name}` is not a declared output")))
}

fn fresh_name(units: &[Unit], creases: &[(String, CreaseInfo)], input: &str, outputs: &[Output], prefix: &str) -> String {
    (1..)
        .map(|i| format!("{prefix}{i}"))
        .find(|n| {
            n != input
                && !units.iter().any(|u| &u.name == n)
                && !creases.iter().any(|c| &c.0 == n)
                && !outputs.iter().any(|o| &o.name == n)
        })
        .unwrap()
}

/// Raw plane equation n.x = e for a (validated) plane form; n is not normalized.
pub fn plane_equation(plane: &PlaneSpec) -> (Vec<f64>, f64) {
    match plane {
        PlaneSpec::Line { p, q } => {
            let n = vec![-(q[1] - p[1]), q[0] - p[0]];
            let e = dot(&n, p);
            (n, e)
        }
        PlaneSpec::Normal { v, offset } => (v.clone(), offset * norm(v)),
        PlaneSpec::NormalThrough { v, p } => (v.clone(), dot(v, p)),
        PlaneSpec::Between { p, q } => {
            let n: Vec<f64> = q.iter().zip(p).map(|(a, b)| a - b).collect();
            let mid: Vec<f64> = q.iter().zip(p).map(|(a, b)| (a + b) / 2.0).collect();
            let e = dot(&n, &mid);
            (n, e)
        }
    }
}

fn crease_weights(
    plane: &PlaneSpec,
    anchor: &Anchor,
    dim: usize,
    check_dim: &dyn Fn(&[f64], &str, &Stmt) -> Result<()>,
    s: &Stmt,
) -> Result<(Vec<f64>, f64)> {
    let err = |m: String| WirError::at(s.line, &s.text, m);
    match plane {
        PlaneSpec::Line { p, q } => {
            if dim != 2 {
                return Err(err(format!("`zero on line` is 2D only but the input is R{dim}; use `zero on plane normal` or `between`")));
            }
            check_dim(p, "line point", s)?;
            check_dim(q, "line point", s)?;
            if p == q {
                return Err(err("line endpoints coincide, so they do not define a line".into()));
            }
        }
        PlaneSpec::Normal { v, offset } => {
            check_dim(v, "normal", s)?;
            if norm(v) == 0.0 {
                return Err(err("plane normal is the zero vector".into()));
            }
            if !offset.is_finite() {
                return Err(err("plane offset is not finite".into()));
            }
        }
        PlaneSpec::NormalThrough { v, p } => {
            check_dim(v, "normal", s)?;
            check_dim(p, "plane point", s)?;
            if norm(v) == 0.0 {
                return Err(err("plane normal is the zero vector".into()));
            }
        }
        PlaneSpec::Between { p, q } => {
            check_dim(p, "between point", s)?;
            check_dim(q, "between point", s)?;
            if p == q {
                return Err(err("`between` points coincide, so there is no bisector".into()));
            }
        }
    }
    let (n, e) = plane_equation(plane);
    check_dim(&anchor.at, "anchor", s)?;
    if !(anchor.value > 0.0) || !anchor.value.is_finite() {
        return Err(err(format!(
            "anchor value must be positive, got {}: a unit's output is never negative and a zero anchor cannot set scale; put negative contributions in the readout coefficient",
            format_num(anchor.value, NumFormat::CANONICAL)
        )));
    }
    let delta = dot(&n, &anchor.at) - e;
    let len = norm(&n);
    let scale = norm(&anchor.at).max((e / len).abs()).max(1.0);
    if delta.abs() / len <= 1e-12 * scale {
        return Err(err(format!(
            "anchor ({}) lies on the crease's plane, so it cannot set orientation or slope",
            fmt_point(&anchor.at)
        )));
    }
    let k = anchor.value / delta;
    Ok((n.iter().map(|x| k * x).collect(), -k * e))
}

/// Largest value of w . x + b over the box [lo, hi]^D.
pub fn box_max(w: &[f64], b: f64, lo: f64, hi: f64) -> f64 {
    b + w.iter().map(|x| (x * lo).max(x * hi)).sum::<f64>()
}

/// `along DIR [above E]` -> (DIR as a flat vector, -E).
fn ramp_form(r: &Ramp, dim: usize, shape: Option<&[usize]>, s: &Stmt) -> Result<(Vec<f64>, f64)> {
    let err = |m: String| WirError::at(s.line, &s.text, m);
    if !r.above.is_finite() {
        return Err(err("`above` level is not finite".into()));
    }
    let v = expand_direction(&r.along, dim, shape).map_err(err)?;
    Ok((v, -r.above))
}

/// Flat vector of a direction; `patch` needs a `R<c>x3x3` input.
pub fn expand_direction(d: &Direction, dim: usize, shape: Option<&[usize]>) -> std::result::Result<Vec<f64>, String> {
    let v = match d {
        Direction::Flat(v) => {
            if v.len() != dim {
                return Err(format!("direction has {} entries but the input is R{dim}", v.len()));
            }
            v.clone()
        }
        Direction::Patch(parts) => {
            let Some(&[c, 3, 3]) = shape else {
                return Err(format!("`patch` needs an input shaped like R<c>x3x3, but the input is R{dim}"));
            };
            let mut v = vec![0.0; dim];
            for (name, coefs) in parts {
                if coefs.len() != c {
                    return Err(format!("patch component `{name}` has {} entries for {c} channels", coefs.len()));
                }
                let &(_, i, j) = PATCH_BASIS.iter().find(|b| b.0 == name).unwrap();
                for (ch, k) in coefs.iter().enumerate() {
                    for row in 0..3 {
                        for col in 0..3 {
                            v[ch * 9 + row * 3 + col] += k * patch_profile(i, col) * patch_profile(j, row);
                        }
                    }
                }
            }
            v
        }
    };
    if v.iter().any(|x| !x.is_finite()) {
        return Err("direction has a non-finite entry".into());
    }
    Ok(v)
}
