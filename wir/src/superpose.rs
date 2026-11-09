//! `wir superpose-choice-layer`: squeeze the features that a program's choices read through
//! one layer of shared neurons (superposition), with no training (docs/POLYSEMANTIC.md).
//!
//! A choice program computes features (lets), then key groups that are affine in them, and
//! picks the largest key of each group (a gt tournament, a max tree, or a host that samples
//! the keys). The tool:
//! 1. reads each key's exact coefficient on each feature, by lowering the program again with
//!    the features as symbolic inputs (`deep::lower_with`); a key that reads a feature through
//!    a nonlinearity is an error,
//! 2. builds the encoder in closed form: weighted PCA of the features over the given states,
//!    each feature scaled by its largest key coefficient, written as `unit(...)` neurons,
//! 3. solves the decoder (features read back linearly from the neurons) with one linear
//!    program: every state's winner leads by `margin` (or every key's value, least absolute
//!    error), states weighted by how often they occur,
//! 4. rewrites the program text: the bottleneck after the last feature, keys reading the
//!    decoded features, renamed copies of lets between features and keys (every other reader
//!    keeps the exact value), and checks the result on every state (the certificate).

use std::collections::HashMap;

use crate::deep::{self, Affine, DeepProgram, Node, SYMBOL};
use crate::error::{Result, WirError};
use crate::lexer::{split_statements, Statement, Tok};

/// What the decoder must keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Preserve {
    /// Each group's winner, ahead of every other key by at least `margin` (1 keeps compiled
    /// gt/ge bits exact on integer keys).
    Winner { margin: f64 },
    /// Every key's value, least weighted absolute error (keys a host samples from).
    Values,
}

#[derive(Debug, Clone)]
pub struct ChoiceLayerSpec {
    /// Lets (or outputs) squeezed through the layer.
    pub features: Vec<String>,
    /// Key groups; each group's largest key is one choice.
    pub groups: Vec<Vec<String>>,
    pub neurons: usize,
    pub preserve: Preserve,
    /// Decimal places of the written coefficients.
    pub digits: usize,
    /// `(name, factor)`: states where `name` is above 0.5 count `factor` times.
    pub weights: Vec<(String, f64)>,
    /// Winner mode: cost of key value error relative to a missed margin (keeps the decoded
    /// features near their values; 0 lets the solver scale keys freely).
    pub fidelity: f64,
}

#[derive(Debug, Clone)]
pub struct GroupReport {
    pub keys: Vec<String>,
    /// Weighted share of states whose winner is unchanged.
    pub winner_kept: f64,
    pub mean_abs: f64,
    pub max_abs: f64,
}

#[derive(Debug, Clone)]
pub struct ChoiceLayer {
    pub source: String,
    pub states: usize,
    /// Distinct (features, rest of the keys) states the solve saw.
    pub distinct: usize,
    /// Share of the weighted feature variance the encoder keeps.
    pub variance_kept: f64,
    pub groups: Vec<GroupReport>,
    /// Weighted share of states where every output of the program is unchanged.
    pub outputs_kept: f64,
    /// Weighted share of states where every group's winner is unchanged and leads by the
    /// margin (Winner mode; compiled comparison bits are then exact).
    pub certified: f64,
    /// Hidden widths and parameters, before and after.
    pub before: (Vec<usize>, usize),
    pub after: (Vec<usize>, usize),
}

/// One state: input vector and weight.
pub type State = (Vec<f64>, f64);

pub fn choice_layer(src: &str, spec: &ChoiceLayerSpec, states: &[State]) -> Result<ChoiceLayer> {
    let prog = deep::lower(src)?;
    if prog.grid.is_some() {
        return Err(WirError::new("superpose-choice-layer is for scalar choice programs, not grid programs"));
    }
    let nf = spec.features.len();
    let m = spec.neurons;
    if nf < 2 || m == 0 || m > nf {
        return Err(WirError::new(format!("need at least 2 features and 1..={nf} neurons, got {m}")));
    }
    if spec.groups.iter().any(|g| g.len() < 2) {
        return Err(WirError::new("each key group needs at least 2 keys"));
    }
    let keys: Vec<String> = spec.groups.iter().flatten().cloned().collect();
    for n in spec.features.iter().chain(&keys).chain(spec.weights.iter().map(|(n, _)| n)) {
        if prog.named(n).is_none() {
            return Err(WirError::new(format!("`{n}` is not a let or output of the program")));
        }
    }
    for (i, n) in spec.features.iter().enumerate() {
        if spec.features[..i].contains(n) || keys.contains(n) {
            return Err(WirError::new(format!("`{n}` is listed twice (features and keys must be distinct)")));
        }
    }
    if states.is_empty() {
        return Err(WirError::new("no states"));
    }

    // 1. key coefficients on the features (features as symbolic inputs)
    let sym = deep::lower_with(src, &spec.features)?;
    let n_in = prog.inputs.len();
    let sym_node: Vec<usize> = spec
        .features
        .iter()
        .map(|f| {
            let k = sym.inputs.iter().position(|(n, _)| *n == format!("{SYMBOL}{f}")).expect("symbol input");
            sym.graph.nodes.iter().position(|nd| *nd == Node::Input(k)).expect("symbol node")
        })
        .collect();
    let mut dep = vec![false; sym.graph.nodes.len()];
    for i in 0..dep.len() {
        dep[i] = match &sym.graph.nodes[i] {
            Node::Input(k) => *k >= n_in,
            Node::Relu(a) => a.terms.keys().any(|&j| dep[j]),
        };
    }
    let depends = |a: &Affine| a.terms.keys().any(|&j| dep[j]);
    let mut a_rows: Vec<Vec<f64>> = Vec::with_capacity(keys.len());
    for k in &keys {
        let a = sym.named(k).unwrap();
        let mut row = vec![0.0; nf];
        for (&j, &c) in &a.terms {
            if let Some(i) = sym_node.iter().position(|&s| s == j) {
                row[i] = c;
            } else if dep[j] {
                return Err(WirError::new(format!(
                    "key `{k}` reads a feature through a nonlinearity (built on line {}); keys must be affine in the features",
                    sym.graph.line[j]
                )));
            }
        }
        a_rows.push(row);
    }
    let scale: Vec<f64> = (0..nf).map(|i| a_rows.iter().map(|r| r[i].abs()).fold(0.0, f64::max)).collect();
    if let Some(i) = scale.iter().position(|&s| s == 0.0) {
        return Err(WirError::new(format!("feature `{}` is not read by any key", spec.features[i])));
    }

    // 2. evaluate every state: features, exact keys, the rest of each key
    struct Row {
        f: Vec<f64>,
        rest: Vec<f64>,
        key: Vec<f64>,
        w: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut seen: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut xs = vec![0.0; sym.inputs.len()];
    for (x, w0) in states {
        if x.len() != n_in {
            return Err(WirError::new(format!("a state has {} values, the program has {n_in} inputs", x.len())));
        }
        let v = prog.graph.eval(x);
        let f: Vec<f64> = spec.features.iter().map(|n| prog.named(n).unwrap().eval(&v)).collect();
        let key: Vec<f64> = keys.iter().map(|n| prog.named(n).unwrap().eval(&v)).collect();
        xs[..n_in].copy_from_slice(x);
        let vs = sym.graph.eval(&xs);
        let rest: Vec<f64> = keys.iter().map(|n| sym.named(n).unwrap().eval(&vs)).collect();
        for k in 0..keys.len() {
            let rec = dot(&a_rows[k], &f) + rest[k];
            if (rec - key[k]).abs() > 1e-6 * (1.0 + key[k].abs()) {
                return Err(WirError::new(format!("key `{}` is not affine in the features ({rec} vs {})", keys[k], key[k])));
            }
        }
        let mut w = *w0;
        for (n, fac) in &spec.weights {
            if prog.named(n).unwrap().eval(&v) > 0.5 {
                w *= fac;
            }
        }
        // a key's error is A_k . (decoded - f), so in Values mode the rest does not matter
        let by_rest: &[f64] = if matches!(spec.preserve, Preserve::Values) { &[] } else { &rest };
        let id: Vec<u64> = f.iter().chain(by_rest).map(|x| x.to_bits()).collect();
        match seen.get(&id) {
            Some(&r) => rows[r].w += w,
            None => {
                seen.insert(id, rows.len());
                rows.push(Row { f, rest, key, w });
            }
        }
    }
    let total: f64 = rows.iter().map(|r| r.w).sum();
    if !(total > 0.0) {
        return Err(WirError::new("states have no weight"));
    }

    // 3. encoder: weighted PCA of the scaled features
    let mut mean = vec![0.0; nf];
    for r in &rows {
        for i in 0..nf {
            mean[i] += r.w * scale[i] * r.f[i] / total;
        }
    }
    let mut cov = vec![vec![0.0; nf]; nf];
    for r in &rows {
        let y: Vec<f64> = (0..nf).map(|i| scale[i] * r.f[i] - mean[i]).collect();
        for i in 0..nf {
            for j in 0..nf {
                cov[i][j] += r.w * y[i] * y[j] / total;
            }
        }
    }
    let (vals, vecs) = sym_eigen(cov);
    let all: f64 = vals.iter().map(|v| v.max(0.0)).sum();
    let variance_kept = if all > 0.0 { vals[..m].iter().map(|v| v.max(0.0)).sum::<f64>() / all } else { 1.0 };
    let enc: Vec<Vec<f64>> = vecs[..m].iter().map(|u| (0..nf).map(|i| round(u[i] * scale[i], spec.digits)).collect()).collect();
    let z: Vec<Vec<f64>> = rows.iter().map(|r| enc.iter().map(|e| dot(e, &r.f)).collect()).collect();

    // 4. decoder: one linear program
    let (dec, dec0) = solve_decoder(spec, &a_rows, &rows.iter().map(|r| (&r.rest[..], &r.key[..], r.w / total)).collect::<Vec<_>>(), &z)?;
    let dec: Vec<Vec<f64>> = dec.iter().map(|r| r.iter().map(|&x| round(x, spec.digits)).collect()).collect();
    let dec0: Vec<f64> = dec0.iter().map(|&x| round(x, spec.digits)).collect();

    // 5. the new program, then check it on every state
    let source = rewrite(src, &prog, &sym, &depends, spec, &keys, &enc, &dec, &dec0)?;
    let out = deep::lower(&source)?;
    let group_of: Vec<std::ops::Range<usize>> = {
        let mut at = 0;
        spec.groups.iter().map(|g| { at += g.len(); at - g.len()..at }).collect()
    };
    let margin = match spec.preserve { Preserve::Winner { margin } => margin, Preserve::Values => 0.0 };
    let mut kept = vec![0.0; spec.groups.len()];
    let mut abs_sum = vec![0.0; spec.groups.len()];
    let mut abs_max = vec![0.0f64; spec.groups.len()];
    let (mut outputs_kept, mut certified, mut weight) = (0.0, 0.0, 0.0);
    for (x, w0) in states {
        let mut w = *w0;
        let v0 = prog.graph.eval(x);
        for (n, fac) in &spec.weights {
            if prog.named(n).unwrap().eval(&v0) > 0.5 {
                w *= fac;
            }
        }
        let v1 = out.graph.eval(x);
        weight += w;
        let same = prog.outputs.iter().zip(&out.outputs).all(|((_, a), (_, b))| (a.eval(&v0) - b.eval(&v1)).abs() < 1e-9);
        outputs_kept += w * same as u8 as f64;
        let mut sure = true;
        for (g, range) in group_of.iter().enumerate() {
            let k0: Vec<f64> = keys[range.clone()].iter().map(|n| prog.named(n).unwrap().eval(&v0)).collect();
            let k1: Vec<f64> = keys[range.clone()].iter().map(|n| out.named(n).unwrap().eval(&v1)).collect();
            let (y0, y1) = (argmax(&k0), argmax(&k1));
            kept[g] += w * (y0 == y1) as u8 as f64;
            let lead = k1.iter().enumerate().filter(|&(j, _)| j != y1).map(|(_, &v)| k1[y1] - v).fold(f64::INFINITY, f64::min);
            sure &= y0 == y1 && lead >= margin - 1e-9;
            for (a, b) in k0.iter().zip(&k1) {
                abs_sum[g] += w * (a - b).abs() / k0.len() as f64;
                abs_max[g] = abs_max[g].max((a - b).abs());
            }
        }
        certified += w * sure as u8 as f64;
    }
    let groups: Vec<GroupReport> = spec
        .groups
        .iter()
        .enumerate()
        .map(|(g, ks)| GroupReport { keys: ks.clone(), winner_kept: kept[g] / weight, mean_abs: abs_sum[g] / weight, max_abs: abs_max[g] })
        .collect();
    let cost = |p: &DeepProgram| {
        let (w, _) = deep::layer(p);
        (w.hidden(), w.param_count())
    };
    // a header saying what was done and what the certificate found
    let mut head = vec![format!(
        "# superpose-choice-layer: {} share {} neurons (PCA encoder, LP decoder, no training).",
        spec.features.join(", "),
        m
    )];
    head.push(format!(
        "# On {} states ({} distinct{}): {}; all outputs unchanged {:.2}%.",
        states.len(),
        rows.len(),
        if spec.weights.is_empty() { String::new() } else { format!(", weighted {}", spec.weights.iter().map(|(n, f)| format!("{n}={f}")).collect::<Vec<_>>().join(" ")) },
        groups.iter().map(|g| format!("{}..{} winner kept {:.2}%", g.keys[0], g.keys[g.keys.len() - 1], 100.0 * g.winner_kept)).collect::<Vec<_>>().join(", "),
        100.0 * outputs_kept / weight
    ));
    let source = match source.split_once('\n') {
        Some((first, rest)) => format!("{first}\n{}\n{rest}", head.join("\n")),
        None => source,
    };
    Ok(ChoiceLayer {
        before: cost(&prog),
        after: cost(&out),
        source,
        states: states.len(),
        distinct: rows.len(),
        variance_kept,
        groups,
        outputs_kept: outputs_kept / weight,
        certified: if matches!(spec.preserve, Preserve::Winner { .. }) { certified / weight } else { f64::NAN },
    })
}

/// Decoder `r = D z + d` from one LP. `rows`: (rest of each key, exact keys, weight).
fn solve_decoder(spec: &ChoiceLayerSpec, a_rows: &[Vec<f64>], rows: &[(&[f64], &[f64], f64)], z: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    use microlp::{ComparisonOp, OptimizationDirection, Problem};
    let nf = spec.features.len();
    let m = spec.neurons;
    let mut lp = Problem::new(OptimizationDirection::Minimize);
    const BOUND: f64 = 1e4;
    let dv: Vec<Vec<microlp::Variable>> = (0..nf).map(|_| (0..m).map(|_| lp.add_var(0.0, (-BOUND, BOUND))).collect()).collect();
    let d0: Vec<microlp::Variable> = (0..nf).map(|_| lp.add_var(0.0, (-BOUND, BOUND))).collect();
    // the key's decoded value as (variable, coefficient) terms plus nothing constant
    let key_terms = |k: usize, zs: &[f64], sign: f64, out: &mut HashMap<usize, (microlp::Variable, f64)>| {
        for i in 0..nf {
            let a = a_rows[k][i] * sign;
            if a == 0.0 {
                continue;
            }
            for c in 0..m {
                let e = out.entry(i * (m + 1) + c).or_insert((dv[i][c], 0.0));
                e.1 += a * zs[c];
            }
            let e = out.entry(i * (m + 1) + m).or_insert((d0[i], 0.0));
            e.1 += a;
        }
    };
    let mut at = 0;
    let ranges: Vec<std::ops::Range<usize>> = spec.groups.iter().map(|g| { at += g.len(); at - g.len()..at }).collect();
    // |A_k . r + rest_k - key_k| <= t, costing `cost` per unit of error
    let fidelity = |lp: &mut Problem, k: usize, zs: &[f64], rest: f64, key: f64, cost: f64| {
        let t_var = lp.add_var(cost, (0.0, f64::INFINITY));
        let mut t = HashMap::new();
        key_terms(k, zs, 1.0, &mut t);
        let lin: Vec<(microlp::Variable, f64)> = t.into_values().filter(|&(_, c)| c != 0.0).collect();
        let mut up = lin.clone();
        up.push((t_var, -1.0));
        lp.add_constraint(up.as_slice(), ComparisonOp::Le, key - rest);
        let mut lo = lin;
        lo.push((t_var, 1.0));
        lp.add_constraint(lo.as_slice(), ComparisonOp::Ge, key - rest);
    };
    for (s, &(rest, key, w)) in rows.iter().enumerate() {
        match spec.preserve {
            Preserve::Winner { margin } => {
                // keep keys near their values too (lightly), or scaling every key up would
                // satisfy the margins and the decoded features would lose their meaning
                for k in 0..key.len() {
                    fidelity(&mut lp, k, &z[s], rest[k], key[k], w * spec.fidelity);
                }
                for range in &ranges {
                    let y = range.start + argmax(&key[range.clone()]);
                    let slack = lp.add_var(w, (0.0, f64::INFINITY));
                    for j in range.clone() {
                        if j == y || key[j] == key[y] {
                            continue;
                        }
                        // (A_y - A_j) . r + rest_y - rest_j + slack >= margin
                        let mut t = HashMap::new();
                        key_terms(y, &z[s], 1.0, &mut t);
                        key_terms(j, &z[s], -1.0, &mut t);
                        let mut expr: Vec<(microlp::Variable, f64)> = t.into_values().filter(|&(_, c)| c != 0.0).collect();
                        expr.push((slack, 1.0));
                        lp.add_constraint(expr.as_slice(), ComparisonOp::Ge, margin - (rest[y] - rest[j]));
                    }
                }
            }
            Preserve::Values => {
                for k in 0..key.len() {
                    fidelity(&mut lp, k, &z[s], rest[k], key[k], w);
                }
            }
        }
    }
    let sol = match lp.solve().map_err(|e| WirError::new(format!("decoder LP: {e}")))? {
        microlp::SolveOutcome::Solution(s) => s,
        microlp::SolveOutcome::Interrupted(_) => return Err(WirError::new("decoder LP was interrupted before a solution")),
    };
    Ok((dv.iter().map(|r| r.iter().map(|&v| sol[v]).collect()).collect(), d0.iter().map(|&v| sol[v]).collect()))
}

/// The program with the bottleneck written in.
#[allow(clippy::too_many_arguments)]
fn rewrite(
    src: &str,
    prog: &DeepProgram,
    sym: &DeepProgram,
    depends: &dyn Fn(&Affine) -> bool,
    spec: &ChoiceLayerSpec,
    keys: &[String],
    enc: &[Vec<f64>],
    dec: &[Vec<f64>],
    dec0: &[f64],
) -> Result<String> {
    let stmts = split_statements(src)?;
    // name -> statement index (let/output)
    let mut at: HashMap<String, usize> = HashMap::new();
    for (i, s) in stmts.iter().enumerate() {
        if let [h, n, ..] = &s.tokens[..]
            && matches!(&h.tok, Tok::Ident(w) if w == "let" || w == "output")
            && let Tok::Ident(name) = &n.tok
        {
            at.insert(name.clone(), i);
        }
    }
    let refs = |i: usize| -> Vec<String> {
        let t = &stmts[i].tokens;
        (3..t.len())
            .filter_map(|j| match &t[j].tok {
                Tok::Ident(n) if t.get(j + 1).map(|x| &x.tok) != Some(&Tok::LParen) => Some(n.clone()),
                _ => None,
            })
            .collect()
    };
    // lets between keys and features: reachable from a key, depending on a feature
    let is_feature = |n: &str| spec.features.iter().any(|f| f == n);
    let mut path: Vec<String> = Vec::new();
    let mut stack: Vec<String> = keys.iter().filter(|k| depends(sym.named(k).unwrap())).cloned().collect();
    while let Some(n) = stack.pop() {
        if path.contains(&n) {
            continue;
        }
        path.push(n.clone());
        for r in refs(at[&n]) {
            if !is_feature(&r) && at.contains_key(&r) && depends(sym.named(&r).unwrap()) {
                stack.push(r);
            }
        }
    }
    let mut rename: HashMap<String, String> = HashMap::new();
    for n in spec.features.iter().chain(path.iter().filter(|n| !keys.contains(n))) {
        rename.insert(n.clone(), format!("sp_{n}"));
    }
    let zname = |j: usize| format!("sp_z{j}");
    for n in rename.values().cloned().chain((0..spec.neurons).map(zname)) {
        if prog.named(&n).is_some() || prog.inputs.iter().any(|(i, _)| *i == n) {
            return Err(WirError::new(format!("the program already uses `{n}`")));
        }
    }
    let last_feature = spec.features.iter().map(|f| at[f]).max().unwrap();
    let path_keys: Vec<&String> = path.iter().filter(|n| keys.contains(n)).collect();
    if let Some(k) = path_keys.iter().find(|k| at[**k] < last_feature) {
        return Err(WirError::new(format!(
            "key `{k}` comes before feature `{}`; define every feature before the first key that reads one",
            spec.features.iter().find(|f| at[*f] == last_feature).unwrap()
        )));
    }
    let print = |i: usize, head: &str, name: &str| -> String {
        let toks: Vec<Tok> = stmts[i].tokens[3..]
            .iter()
            .enumerate()
            .map(|(j, t)| match &t.tok {
                Tok::Ident(n) if stmts[i].tokens.get(j + 4).map(|x| &x.tok) != Some(&Tok::LParen) => {
                    Tok::Ident(rename.get(n).cloned().unwrap_or_else(|| n.clone()))
                }
                t => t.clone(),
            })
            .collect();
        format!("{head} {name} = {}", print_tokens(&toks))
    };
    let mut block = vec![format!(
        "# --- superpose-choice-layer: {} features share {} neurons (PCA encoder, LP decoder) ---",
        spec.features.len(),
        spec.neurons
    )];
    for (j, e) in enc.iter().enumerate() {
        block.push(format!("let {} = unit({})", zname(j), linear(e, &spec.features, 0.0, spec.digits)));
    }
    let zs: Vec<String> = (0..spec.neurons).map(zname).collect();
    for (i, f) in spec.features.iter().enumerate() {
        block.push(format!("let sp_{f} = {}", linear(&dec[i], &zs, dec0[i], spec.digits)));
    }
    let mut after: HashMap<usize, Vec<String>> = HashMap::new(); // copies after their statement
    let mut copies: Vec<&String> = path.iter().filter(|n| !keys.contains(n)).collect();
    copies.sort_by_key(|n| at[*n]);
    for n in copies {
        let line = print(at[n], "let", &rename[n]);
        if at[n] < last_feature { block.push(line) } else { after.entry(at[n]).or_default().push(line) }
    }
    block.push("# --- end of superpose-choice-layer ---".into());
    after.entry(last_feature).or_default().splice(0..0, block);
    // in-place rewrites of the keys that read features
    let mut replace: HashMap<usize, String> = HashMap::new();
    for k in &path_keys {
        let i = at[*k];
        let head = match &stmts[i].tokens[0].tok { Tok::Ident(h) => h.clone(), _ => unreachable!() };
        replace.insert(i, print(i, &head, k));
    }
    // emit, line by line
    let lines: Vec<&str> = src.lines().collect();
    let stmt_lines = |s: &Statement| -> (usize, usize) {
        let last = s.tokens.iter().map(|t| t.line).max().unwrap_or(s.line);
        (s.line, last)
    };
    let mut first_of: HashMap<usize, usize> = HashMap::new(); // first line -> stmt
    let mut last_of: HashMap<usize, usize> = HashMap::new(); // last line -> stmt
    let mut skip = vec![false; lines.len() + 1];
    for (i, s) in stmts.iter().enumerate() {
        let (a, b) = stmt_lines(s);
        first_of.insert(a, i);
        last_of.insert(b, i);
        if replace.contains_key(&i) {
            for l in a + 1..=b {
                skip[l] = true;
            }
        }
    }
    let mut out = String::new();
    for (idx, raw) in lines.iter().enumerate() {
        let ln = idx + 1;
        if skip[ln] {
            continue;
        }
        match first_of.get(&ln).and_then(|i| replace.get(i)) {
            Some(text) => out.push_str(text),
            None => out.push_str(raw),
        }
        out.push('\n');
        if let Some(i) = last_of.get(&ln)
            && let Some(extra) = after.get(i)
        {
            for l in extra {
                out.push_str(l);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

fn print_tokens(toks: &[Tok]) -> String {
    let mut s = String::new();
    let mut prev: Option<&Tok> = None;
    for t in toks {
        let space = match (prev, t) {
            (None, _) => false,
            (_, Tok::RParen | Tok::RBracket | Tok::Comma) => false,
            (Some(Tok::LParen | Tok::LBracket), _) => false,
            (Some(Tok::Ident(_)), Tok::LParen) => false,
            (_, Tok::Star | Tok::Slash) | (Some(Tok::Star | Tok::Slash), _) => false,
            (Some(Tok::Minus), _) if s.ends_with("(-") || s == "-" || s.ends_with(", -") || s.ends_with("[-") => false,
            _ => true,
        };
        if space {
            s.push(' ');
        }
        s.push_str(&match t {
            Tok::Ident(n) => n.clone(),
            Tok::Num(v) => num_text(*v),
            Tok::LParen => "(".into(),
            Tok::RParen => ")".into(),
            Tok::LBrace => "{".into(),
            Tok::RBrace => "}".into(),
            Tok::LBracket => "[".into(),
            Tok::RBracket => "]".into(),
            Tok::Slash => "/".into(),
            Tok::Dot => ".".into(),
            Tok::DotDot => "..".into(),
            Tok::Comma => ",".into(),
            Tok::Colon => ":".into(),
            Tok::Arrow => "->".into(),
            Tok::Eq => "=".into(),
            Tok::EqEq => "==".into(),
            Tok::Plus => "+".into(),
            Tok::Minus => "-".into(),
            Tok::Star => "*".into(),
        });
        prev = Some(t);
    }
    s
}

/// Shortest text that reads back to the same f64.
fn num_text(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 { format!("{}", v as i64) } else { format!("{v}") }
}

fn round(x: f64, digits: usize) -> f64 {
    let p = 10f64.powi(digits as i32);
    let r = (x * p).round() / p;
    if r == 0.0 { 0.0 } else { r }
}

fn coef_text(x: f64, digits: usize) -> String {
    let s = format!("{:.*}", digits, x.abs());
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    s
}

/// `c + k1*n1 - k2*n2 ...`, zero terms dropped.
fn linear(k: &[f64], names: &[impl AsRef<str>], c: f64, digits: usize) -> String {
    let mut s = String::new();
    for (k, n) in k.iter().zip(names) {
        if *k == 0.0 {
            continue;
        }
        let body = if (k.abs() - 1.0).abs() < 1e-12 { n.as_ref().to_string() } else { format!("{}*{}", coef_text(*k, digits), n.as_ref()) };
        if s.is_empty() {
            s = if *k < 0.0 { format!("-{body}") } else { body };
        } else {
            s.push_str(if *k < 0.0 { " - " } else { " + " });
            s.push_str(&body);
        }
    }
    if c != 0.0 || s.is_empty() {
        let ct = coef_text(c, digits);
        s = if s.is_empty() { if c < 0.0 { format!("-{ct}") } else { ct } } else { format!("{s} {} {ct}", if c < 0.0 { "-" } else { "+" }) };
    }
    s
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// First index of the largest value.
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Eigenvalues (descending) and unit eigenvectors (rows, largest entry positive) of a
/// symmetric matrix, by cyclic Jacobi rotations.
pub fn sym_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| (i == j) as u8 as f64).collect()).collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        let scale: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum::<f64>().max(1e-300);
        if off <= 1e-28 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let vals = order.iter().map(|&i| a[i][i]).collect();
    let vecs = order
        .iter()
        .map(|&j| {
            let mut u: Vec<f64> = (0..n).map(|i| v[i][j]).collect();
            let big = u.iter().cloned().fold(0.0f64, |m, x| if x.abs() > m.abs() { x } else { m });
            if big < 0.0 {
                u.iter_mut().for_each(|x| *x = -*x);
            }
            u
        })
        .collect();
    (vals, vecs)
}

/// Compiled brain variants for a demo page (`"brains"` in its JSON): each `(label, source)`
/// must read and write the same names, in the same order, as the page's exact brain.
pub fn variants_json(variants: &[(String, String)], inputs: &[String], outputs: &[String]) -> Result<serde_json::Value> {
    let mut out = Vec::new();
    for (label, src) in variants {
        let c = deep::compile(src, 1e-6)?;
        let w = &c.weights;
        if w.inputs != inputs || w.outputs != outputs {
            return Err(WirError::new(format!("variant `{label}` does not read and write the same names as the exact brain")));
        }
        out.push(serde_json::json!({
            "label": label, "inputs": w.inputs, "outputs": w.outputs,
            "hidden": w.hidden(), "params": w.param_count(),
            "layers": crate::tactics::sparse_layers(w), "source": src,
        }));
    }
    Ok(serde_json::Value::Array(out))
}

/// Random states from the declared input intervals (integers when both bounds are
/// integers), weight 1 each. Every combination counts, reachable or not.
pub fn sample_states(src: &str, n: usize, seed: u64) -> Result<Vec<State>> {
    let prog = deep::lower(src)?;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for (name, (lo, hi)) in &prog.inputs {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(WirError::new(format!("input `{name}` has no finite interval to sample from")));
        }
    }
    Ok((0..n)
        .map(|_| {
            let x = prog
                .inputs
                .iter()
                .map(|(_, (lo, hi))| {
                    let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
                    if lo.fract() == 0.0 && hi.fract() == 0.0 { (lo + (u * (hi - lo + 1.0)).floor()).min(*hi) } else { lo + u * (hi - lo) }
                })
                .collect();
            (x, 1.0)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalizes() {
        let a = vec![vec![4.0, 1.0, 2.0], vec![1.0, 3.0, 0.5], vec![2.0, 0.5, 5.0]];
        let (vals, vecs) = sym_eigen(a.clone());
        assert!(vals.windows(2).all(|w| w[0] >= w[1]));
        for (l, u) in vals.iter().zip(&vecs) {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i][j] * u[j]).sum();
                assert!((av - l * u[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn linear_text() {
        assert_eq!(linear(&[1.0, -0.5, 0.0], &["a", "b", "c"], 0.25, 4), "a - 0.5*b + 0.25");
        assert_eq!(linear(&[0.0], &["a"], -2.0, 4), "-2");
    }
}
