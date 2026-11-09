//! WIR 0.3: sequence programs with attention heads, compiled to a small transformer.
//!
//! A 0.3 program reads a token sequence. Every `let`/`output` is a per-position value
//! written with the 0.2 combinators, plus two attention reads (after RASP's aggregate
//! and selector width, Weiss et al. 2021; the compilation follows Tracr, Lindner et
//! al. 2023):
//!
//!   mean(v, pred)   v averaged over the key positions where pred holds (0 if none)
//!   count(pred)     how many key positions pred holds at
//!
//! `pred` is `true`, `k(a) == q(b)` or `eq/ne/lt/le/gt/ge(side, side)` with sides
//! `k(expr)` (the expression at the key position) and `q(expr)` (at the query
//! position). Both sides need integer intervals. `pos` is the position (0-based) and
//! the token input its integer value.
//!
//! ```text
//! wir 0.3
//! sequence len 8                  # max length; `causal` keeps keys at or before the query
//! input tok: token in [0, 7]
//! let n = count(true)
//! output rev = mean(tok, k(pos) == q(n - pos - 1))
//! ```
//!
//! The network is a norm-free transformer: blocks of attention then a one-hidden-layer
//! ReLU MLP, each adding into a residual stream. Position 0 is a BOS token. Residual
//! dimensions: `isbos`, `one` (1 at every real position, 0 at BOS, and the only way
//! constants enter), the token and position one-hot plus value, then one per relu node,
//! head and count decoder. Every dimension but `isbos` is 0 at BOS (up to the leak: BOS's
//! own query reads BOS in a mean head, the real keys' 0 in a count head), so a head with
//! no matching key reads 0 there. A head scores matching keys `sharpness` (beta), BOS
//! beta / 2 (mean) or beta (count) and other keys 0, so it is hard attention up to a
//! leak of about e^(-beta / 2). `count` reads 1/(w + 1) from a head that always
//! includes BOS and turns it back into w with a piecewise-linear MLP, exact at the
//! integers.
//!
//! The program's own semantics (`SeqProgram::eval`) uses exact hard attention; the
//! compiled weights (`SeqWeights::eval`) use softmax, and the two are compared.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

use crate::deep::{Affine, Cursor, Graph, Node, Scope};
use crate::error::{Result, WirError};
use crate::lexer::{split_statements, Tok};
use crate::model::Mat;

pub const VERSION: f64 = 0.3;
pub const FUNCTIONS: &[&str] = &["mean", "count"];
pub const DEFAULT_SHARPNESS: f64 = 40.0;
/// Most values a one-hot side may take.
pub const MAX_ONEHOT: usize = 512;

// ---------------------------------------------------------------- program

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    True,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn of(name: &str) -> Option<Cmp> {
        Some(match name {
            "eq" => Cmp::Eq,
            "ne" => Cmp::Ne,
            "lt" => Cmp::Lt,
            "le" => Cmp::Le,
            "gt" => Cmp::Gt,
            "ge" => Cmp::Ge,
            _ => return None,
        })
    }

    fn holds(self, a: i64, b: i64) -> bool {
        match self {
            Cmp::True => true,
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }
}

/// One side of a predicate: its value, smallest value and one-hot (entry c is 1 when
/// the value is lo + c).
#[derive(Debug, Clone)]
pub struct Side {
    pub value: Affine,
    pub lo: i64,
    pub onehot: Vec<Affine>,
}

#[derive(Debug, Clone)]
pub enum HeadKind {
    Mean(Affine),
    Count,
}

/// An attention head: holds at (query i, key j) when `cmp` relates the sides,
/// written in source order (`key_first`: the key side was written first).
#[derive(Debug, Clone)]
pub struct Head {
    /// Graph node holding the head's output (mean, or 1/(w + 1) for count).
    pub node: usize,
    pub kind: HeadKind,
    pub q: Side,
    pub k: Side,
    pub cmp: Cmp,
    pub key_first: bool,
    pub line: usize,
}

impl Head {
    pub fn holds(&self, qv: i64, kv: i64) -> bool {
        if self.key_first { self.cmp.holds(kv, qv) } else { self.cmp.holds(qv, kv) }
    }
}

/// Parse state shared with the expression parser.
#[derive(Debug, Clone, Default)]
pub struct SeqCtx {
    pub len: usize,
    pub causal: bool,
    pub sharpness: f64,
    /// Token value range and the graph nodes of its value and one-hot.
    pub vocab: (i64, i64),
    pub tok_value: usize,
    pub tok_onehot: Vec<usize>,
    pub pos_value: usize,
    pub pos_onehot: Vec<usize>,
    pub heads: Vec<Head>,
    /// Graph input nodes that are head outputs, to their head.
    pub head_of: HashMap<usize, usize>,
    /// Nodes an MLP writes directly: value = the affine (over relu nodes of one layer).
    pub defs: HashMap<usize, Affine>,
    onehots: HashMap<Vec<u64>, Side>,
    next_input: usize,
}

impl SeqCtx {
    fn fresh_input(&mut self, g: &mut Graph, range: (f64, f64), line: usize) -> Affine {
        self.next_input += 1;
        g.input(self.next_input - 1, range, line)
    }

    /// One-hot encoding of an integer value: free for the token and position (embedding
    /// dimensions), otherwise hats r(c - 1) - 2 r(c) + r(c + 1) over r(t) = relu(v - t).
    fn side(&mut self, g: &mut Graph, v: &Affine, line: usize) -> std::result::Result<Side, String> {
        let key: Vec<u64> = v.terms.iter().flat_map(|(&n, k)| [n as u64, k.to_bits()]).chain([v.c.to_bits()]).collect();
        if let Some(s) = self.onehots.get(&key) {
            return Ok(s.clone());
        }
        let (lo, hi) = g.range_of(v);
        if !lo.is_finite() || !hi.is_finite() || lo.fract() != 0.0 || hi.fract() != 0.0 {
            return Err(format!("attention keys and queries need integer intervals, got [{lo}, {hi}]"));
        }
        let count = (hi - lo) as usize + 1;
        if count > MAX_ONEHOT {
            return Err(format!("a key or query side takes {count} values (at most {MAX_ONEHOT})"));
        }
        let direct = v.terms.len() == 1 && v.c.fract() == 0.0 && {
            let (&n, &k) = v.terms.iter().next().unwrap();
            k == 1.0 && (n == self.tok_value || n == self.pos_value)
        };
        let onehot = if direct {
            // value = node + c: entry for value x is the node's one-hot at x - c
            let (&n, _) = v.terms.iter().next().unwrap();
            let (base, nodes) =
                if n == self.tok_value { (self.vocab.0, &self.tok_onehot) } else { (0, &self.pos_onehot) };
            (0..count)
                .map(|i| {
                    let x = lo as i64 + i as i64 - v.c as i64 - base;
                    if x >= 0 && (x as usize) < nodes.len() { Affine::node(nodes[x as usize]) } else { Affine::default() }
                })
                .collect()
        } else {
            let r: Vec<Affine> = (0..count + 2).map(|i| g.relu(&v.plus(-(lo + i as f64 - 1.0)), line)).collect();
            (0..count).map(|c| Affine::lin(&r[c].add(&r[c + 2]), 1.0, &r[c + 1], -2.0)).collect()
        };
        let s = Side { value: v.clone(), lo: lo as i64, onehot };
        self.onehots.insert(key, s.clone());
        Ok(s)
    }
}

/// A parsed and lowered 0.3 program.
#[derive(Debug, Clone)]
pub struct SeqProgram {
    pub ctx: SeqCtx,
    pub graph: Graph,
    pub token_name: String,
    pub outputs: Vec<(String, Affine)>,
    pub names: Vec<(String, Affine)>,
}

/// True when the first statement is a `wir 0.3` header.
pub fn is_seq_source(src: &str) -> bool {
    let Ok(stmts) = split_statements(src) else { return false };
    matches!(stmts.first().map(|s| s.tokens.iter().map(|t| &t.tok).collect::<Vec<_>>()).as_deref(),
        Some([Tok::Ident(w), Tok::Num(v)]) if w == "wir" && *v == VERSION)
}

pub fn lower(src: &str) -> Result<SeqProgram> {
    let stmts = split_statements(src)?;
    let mut g = Graph::default();
    let mut ctx = SeqCtx { sharpness: DEFAULT_SHARPNESS, ..Default::default() };
    let mut env: HashMap<String, Affine> = HashMap::new();
    let mut outputs = vec![];
    let mut names = vec![];
    let mut token_name = None;
    let mut sequence = false;
    // Unused by sequence programs, required by the expression parser.
    let (mut inputs, mut taps, lets, mut shifted) = (vec![], HashMap::new(), HashMap::new(), HashMap::new());
    for (i, s) in stmts.iter().enumerate() {
        let mut p = Cursor { t: &s.tokens, i: 0, s };
        let head = p.ident()?;
        if i == 0 {
            if head != "wir" || !matches!(p.next()?, Tok::Num(v) if v == VERSION) {
                return Err(p.err("a sequence program starts with `wir 0.3`"));
            }
            p.end()?;
            continue;
        }
        match head.as_str() {
            "sequence" => {
                if sequence {
                    return Err(p.err("one `sequence` statement per program"));
                }
                sequence = true;
                while p.peek().is_some() {
                    match p.ident()?.as_str() {
                        "len" => {
                            let n = p.signed()?;
                            if n.fract() != 0.0 || !(1.0..=4096.0).contains(&n) {
                                return Err(p.err("`len` is an integer in 1..4096"));
                            }
                            ctx.len = n as usize;
                        }
                        "causal" => ctx.causal = true,
                        "sharpness" => {
                            ctx.sharpness = p.signed()?;
                            if !(ctx.sharpness > 0.0) {
                                return Err(p.err("sharpness must be positive"));
                            }
                        }
                        w => return Err(p.err(&format!("unknown sequence option `{w}` (len, causal, sharpness)"))),
                    }
                }
                if ctx.len == 0 {
                    return Err(p.err("`sequence` needs `len N`"));
                }
                ctx.pos_value = ctx.fresh_input(&mut g, (0.0, ctx.len as f64 - 1.0), s.line).terms.keys().next().copied().unwrap();
                for _ in 0..ctx.len {
                    let a = ctx.fresh_input(&mut g, (0.0, 1.0), s.line);
                    ctx.pos_onehot.push(*a.terms.keys().next().unwrap());
                }
                env.insert("pos".into(), Affine::node(ctx.pos_value));
            }
            "input" => {
                if !sequence {
                    return Err(p.err("declare `sequence len N` before the input"));
                }
                if token_name.is_some() {
                    return Err(p.err("a sequence program has one token input"));
                }
                let name = p.ident()?;
                p.fresh(&name, &env)?;
                p.expect(&Tok::Colon)?;
                if p.ident()? != "token" || !p.eat_ident("in") {
                    return Err(p.err("sequence inputs are `input tok: token in [lo, hi]`"));
                }
                p.expect(&Tok::LBracket)?;
                let lo = p.signed()?;
                p.expect(&Tok::Comma)?;
                let hi = p.signed()?;
                p.expect(&Tok::RBracket)?;
                p.end()?;
                if lo.fract() != 0.0 || hi.fract() != 0.0 || !(lo <= hi) || hi - lo >= MAX_ONEHOT as f64 {
                    return Err(p.err(&format!("token values are integers lo <= hi, at most {MAX_ONEHOT} of them")));
                }
                ctx.vocab = (lo as i64, hi as i64);
                ctx.tok_value = *ctx.fresh_input(&mut g, (lo, hi), s.line).terms.keys().next().unwrap();
                for _ in ctx.vocab.0..=ctx.vocab.1 {
                    let a = ctx.fresh_input(&mut g, (0.0, 1.0), s.line);
                    ctx.tok_onehot.push(*a.terms.keys().next().unwrap());
                }
                env.insert(name.clone(), Affine::node(ctx.tok_value));
                token_name = Some(name);
            }
            "let" | "output" => {
                if token_name.is_none() {
                    return Err(p.err("declare the token input before `let` and `output`"));
                }
                let name = p.ident()?;
                if FUNCTIONS.contains(&name.as_str()) || ["k", "q", "true", "sequence", "token"].contains(&name.as_str()) {
                    return Err(p.err(&format!("`{name}` is reserved")));
                }
                p.fresh(&name, &env)?;
                p.expect(&Tok::Eq)?;
                let mut sc = Scope {
                    g: &mut g,
                    env: &env,
                    grid: None,
                    inputs: &mut inputs,
                    taps: &mut taps,
                    stmts: &stmts,
                    lets: &lets,
                    shifted: &mut shifted,
                    shift: (0, 0),
                    seq: Some(&mut ctx),
                };
                let v = p.expr(&mut sc)?;
                p.end()?;
                names.push((name.clone(), v.clone()));
                if head == "output" {
                    outputs.push((name.clone(), v.clone()));
                }
                env.insert(name, v);
            }
            other => return Err(p.err(&format!("unknown statement `{other}` (0.3 has sequence, input, let, output)"))),
        }
    }
    let token_name = token_name.ok_or_else(|| WirError::new("a sequence program needs `input tok: token in [lo, hi]`"))?;
    if outputs.is_empty() {
        return Err(WirError::new("a sequence program needs at least one output"));
    }
    Ok(SeqProgram { ctx, graph: g, token_name, outputs, names })
}

/// `mean(v, pred)` and `count(pred)`, called from the expression parser.
pub(crate) fn call(p: &mut Cursor, f: &str, sc: &mut Scope) -> Result<Affine> {
    let line = p.s.line;
    p.expect(&Tok::LParen)?;
    let value = if f == "mean" {
        let v = p.expr(sc)?;
        p.expect(&Tok::Comma)?;
        Some(v)
    } else {
        None
    };
    let (cmp, first, second) = predicate(p, sc)?;
    p.expect(&Tok::RParen)?;
    let (key_first, qv, kv) = match (first, second) {
        (Some((true, a)), Some((false, b))) => (true, b, a),
        (Some((false, a)), Some((true, b))) => (false, a, b),
        (None, None) => (false, Affine::default(), Affine::default()),
        _ => return Err(p.err("a predicate compares one `k(..)` side with one `q(..)` side")),
    };
    let g = &mut *sc.g;
    let ctx = sc.seq.as_deref_mut().unwrap();
    let (q, k) = if cmp == Cmp::True {
        let all = Side { value: Affine::default(), lo: 0, onehot: vec![Affine::constant(1.0)] };
        (all.clone(), all)
    } else {
        (ctx.side(g, &qv, line).map_err(|m| p.err(&m))?, ctx.side(g, &kv, line).map_err(|m| p.err(&m))?)
    };
    let n = ctx.len as f64;
    match value {
        Some(v) => {
            let (lo, hi) = g.range_of(&v);
            let out = ctx.fresh_input(g, (lo.min(0.0), hi.max(0.0)), line);
            let node = *out.terms.keys().next().unwrap();
            ctx.head_of.insert(node, ctx.heads.len());
            ctx.heads.push(Head { node, kind: HeadKind::Mean(v), q, k, cmp, key_first, line });
            Ok(out)
        }
        None => {
            // h = 1 / (w + 1) for w = 0..N. Decode with the line through the points
            // (1/(w+1), w): f(h) = N + s0 relu(h - x0) + sum of slope changes at each point.
            let x0 = 1.0 / (n + 1.0);
            let h = ctx.fresh_input(g, (x0, 1.0), line);
            let node = *h.terms.keys().next().unwrap();
            ctx.head_of.insert(node, ctx.heads.len());
            ctx.heads.push(Head { node, kind: HeadKind::Count, q, k, cmp, key_first, line });
            let pts: Vec<(f64, f64)> = (0..=ctx.len).rev().map(|w| (1.0 / (w as f64 + 1.0), w as f64)).collect();
            let slope = |a: usize| (pts[a + 1].1 - pts[a].1) / (pts[a + 1].0 - pts[a].0);
            let first = g.unit(&h, line).unwrap().plus(-x0); // relu(h - x0) on its own unit
            let mut body = first.scale(slope(0));
            for a in 1..pts.len() - 1 {
                let r = g.relu(&h.plus(-pts[a].0), line);
                body = body.add(&r.scale(slope(a) - slope(a - 1)));
            }
            let def = ctx.fresh_input(g, (-n, 0.0), line);
            let d = *def.terms.keys().next().unwrap();
            ctx.defs.insert(d, body);
            Ok(def.plus(n))
        }
    }
}

type SideRef = Option<(bool, Affine)>;

fn predicate(p: &mut Cursor, sc: &mut Scope) -> Result<(Cmp, SideRef, SideRef)> {
    if p.eat_ident("true") {
        return Ok((Cmp::True, None, None));
    }
    let call = matches!(p.peek(), Some(Tok::Ident(w)) if Cmp::of(w).is_some())
        && matches!(p.t.get(p.i + 1).map(|t| &t.tok), Some(Tok::LParen));
    if call {
        let cmp = Cmp::of(&p.ident()?).unwrap();
        p.expect(&Tok::LParen)?;
        let a = side(p, sc)?;
        p.expect(&Tok::Comma)?;
        let b = side(p, sc)?;
        p.expect(&Tok::RParen)?;
        return Ok((cmp, Some(a), Some(b)));
    }
    let a = side(p, sc)?;
    p.expect(&Tok::EqEq)?;
    let b = side(p, sc)?;
    Ok((Cmp::Eq, Some(a), Some(b)))
}

/// `k(expr)` -> (true, expr), `q(expr)` -> (false, expr)
fn side(p: &mut Cursor, sc: &mut Scope) -> Result<(bool, Affine)> {
    let is_key = match p.ident()?.as_str() {
        "k" => true,
        "q" => false,
        w => return Err(p.err(&format!("expected `k(..)` or `q(..)` in a predicate, found `{w}`"))),
    };
    p.expect(&Tok::LParen)?;
    let v = p.expr(sc)?;
    p.expect(&Tok::RParen)?;
    Ok((is_key, v))
}

// ---------------------------------------------------------------- reference semantics

impl SeqProgram {
    fn check_tokens(&self, tokens: &[i64]) {
        assert!(!tokens.is_empty() && tokens.len() <= self.ctx.len, "sequence length must be 1..={}", self.ctx.len);
        assert!(tokens.iter().all(|t| (self.ctx.vocab.0..=self.ctx.vocab.1).contains(t)), "token out of range");
    }

    /// Node values at every position under exact hard attention: [node][position].
    pub fn node_values(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        self.check_tokens(tokens);
        let c = &self.ctx;
        let n = tokens.len();
        let mut vals: Vec<Vec<f64>> = Vec::with_capacity(self.graph.nodes.len());
        let at = |a: &Affine, vals: &[Vec<f64>], i: usize| a.c + a.terms.iter().map(|(&m, k)| k * vals[m][i]).sum::<f64>();
        let int = |x: f64| -> i64 {
            let r = x.round();
            assert!((x - r).abs() < 1e-6, "attention side value {x} is not an integer");
            r as i64
        };
        for (id, node) in self.graph.nodes.iter().enumerate() {
            let col: Vec<f64> = match node {
                Node::Relu(a) => (0..n).map(|i| at(a, &vals, i).max(0.0)).collect(),
                Node::Input(_) if id == c.tok_value => tokens.iter().map(|&t| t as f64).collect(),
                Node::Input(_) if id == c.pos_value => (0..n).map(|i| i as f64).collect(),
                Node::Input(_) if let Some(k) = c.tok_onehot.iter().position(|&m| m == id) => {
                    tokens.iter().map(|&t| (t - c.vocab.0 == k as i64) as i64 as f64).collect()
                }
                Node::Input(_) if let Some(k) = c.pos_onehot.iter().position(|&m| m == id) => (0..n).map(|i| (i == k) as i64 as f64).collect(),
                Node::Input(_) if let Some(body) = c.defs.get(&id) => (0..n).map(|i| at(body, &vals, i)).collect(),
                Node::Input(_) => {
                    let h = &c.heads[c.head_of[&id]];
                    let kv: Vec<i64> = (0..n).map(|j| int(at(&h.k.value, &vals, j))).collect();
                    (0..n)
                        .map(|i| {
                            let qv = int(at(&h.q.value, &vals, i));
                            let keys: Vec<usize> = (0..n).filter(|&j| (!c.causal || j <= i) && h.holds(qv, kv[j])).collect();
                            match &h.kind {
                                HeadKind::Count => 1.0 / (keys.len() as f64 + 1.0),
                                HeadKind::Mean(_) if keys.is_empty() => 0.0,
                                HeadKind::Mean(v) => keys.iter().map(|&j| at(v, &vals, j)).sum::<f64>() / keys.len() as f64,
                            }
                        })
                        .collect()
                }
            };
            vals.push(col);
        }
        vals
    }

    /// Outputs per position: [position][output].
    pub fn eval(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        let vals = self.node_values(tokens);
        (0..tokens.len())
            .map(|i| self.outputs.iter().map(|(_, a)| a.c + a.terms.iter().map(|(&m, k)| k * vals[m][i]).sum::<f64>()).collect())
            .collect()
    }
}

// ---------------------------------------------------------------- weights

#[derive(Debug, Clone, PartialEq)]
pub struct AttnHead {
    /// [d_head, d_model]; scores are (Wq x_i) . (Wk x_j), no 1/sqrt(d_head).
    pub wq: Mat,
    pub wk: Mat,
    /// [1, d_model] and [d_model, 1].
    pub wv: Mat,
    pub wo: Mat,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub heads: Vec<AttnHead>,
    /// [hidden, d_model] and [d_model, hidden]; no biases.
    pub w_in: Mat,
    pub w_out: Mat,
}

/// A norm-free transformer: x = tok_embed[t] + pos_embed[i]; per block
/// x += sum of heads, then x += w_out relu(w_in x); outputs = unembed x.
#[derive(Debug, Clone, PartialEq)]
pub struct SeqWeights {
    pub dims: Vec<String>,
    pub len: usize,
    pub causal: bool,
    pub vocab: (i64, i64),
    /// Row 0 is BOS, row 1 + t - vocab.0 is token t. [V + 1, d_model]
    pub tok_embed: Mat,
    /// Row 0 is BOS's position, row 1 + i is position i. [len + 1, d_model]
    pub pos_embed: Mat,
    pub blocks: Vec<Block>,
    /// [outputs, d_model]
    pub unembed: Mat,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeqCost {
    pub d_model: usize,
    pub blocks: usize,
    pub heads: usize,
    pub d_head_max: usize,
    pub mlp_hidden: Vec<usize>,
    pub params: usize,
    pub nonzero: usize,
}

impl SeqWeights {
    pub fn d_model(&self) -> usize {
        self.dims.len()
    }

    fn mats(&self) -> Vec<&Mat> {
        let mut m = vec![&self.tok_embed, &self.pos_embed, &self.unembed];
        for b in &self.blocks {
            for h in &b.heads {
                m.extend([&h.wq, &h.wk, &h.wv, &h.wo]);
            }
            m.extend([&b.w_in, &b.w_out]);
        }
        m
    }

    pub fn cost(&self) -> SeqCost {
        let mats = self.mats();
        SeqCost {
            d_model: self.d_model(),
            blocks: self.blocks.len(),
            heads: self.blocks.iter().map(|b| b.heads.len()).sum(),
            d_head_max: self.blocks.iter().flat_map(|b| b.heads.iter().map(|h| h.wq.rows)).max().unwrap_or(0),
            mlp_hidden: self.blocks.iter().map(|b| b.w_in.rows).collect(),
            params: mats.iter().map(|m| m.data.len()).sum(),
            nonzero: mats.iter().map(|m| m.data.iter().filter(|x| **x != 0.0).count()).sum(),
        }
    }

    /// Residual stream at every position (BOS first) after the last block.
    pub fn residual(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        let d = self.d_model();
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(tokens.len() + 1);
        x.push((0..d).map(|c| self.tok_embed.get(0, c) + self.pos_embed.get(0, c)).collect());
        for (i, &t) in tokens.iter().enumerate() {
            let r = (1 + t - self.vocab.0) as usize;
            x.push((0..d).map(|c| self.tok_embed.get(r, c) + self.pos_embed.get(1 + i, c)).collect());
        }
        let n = x.len();
        let mv = |m: &Mat, v: &[f64]| -> Vec<f64> { (0..m.rows).map(|r| m.row(r).iter().zip(v).map(|(a, b)| a * b).sum()).collect() };
        for b in &self.blocks {
            let mut add = vec![vec![0.0; d]; n];
            for h in &b.heads {
                let q: Vec<Vec<f64>> = x.iter().map(|v| mv(&h.wq, v)).collect();
                let k: Vec<Vec<f64>> = x.iter().map(|v| mv(&h.wk, v)).collect();
                let val: Vec<f64> = x.iter().map(|v| mv(&h.wv, v)[0]).collect();
                for i in 0..n {
                    let keys = if self.causal { i + 1 } else { n };
                    let s: Vec<f64> = (0..keys).map(|j| q[i].iter().zip(&k[j]).map(|(a, b)| a * b).sum()).collect();
                    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    let out: f64 = e.iter().zip(&val).map(|(w, v)| w * v).sum::<f64>() / z;
                    for (c, a) in add[i].iter_mut().enumerate() {
                        *a += h.wo.get(c, 0) * out;
                    }
                }
            }
            for i in 0..n {
                for c in 0..d {
                    x[i][c] += add[i][c];
                }
                let hid: Vec<f64> = mv(&b.w_in, &x[i]).into_iter().map(|z| z.max(0.0)).collect();
                let o = mv(&b.w_out, &hid);
                for c in 0..d {
                    x[i][c] += o[c];
                }
            }
        }
        x
    }

    /// Outputs per real position: [position][output].
    pub fn eval(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        let x = self.residual(tokens);
        x[1..].iter().map(|v| (0..self.unembed.rows).map(|r| self.unembed.row(r).iter().zip(v).map(|(a, b)| a * b).sum()).collect()).collect()
    }

    pub fn to_json(&self) -> Value {
        let m = |m: &Mat| json!(m.to_rows());
        json!({
            "format": "wir-seq-weights",
            "version": VERSION,
            "notes": "norm-free transformer; scores (Wq x_i).(Wk x_j) without 1/sqrt(d_head); no biases; position 0 is BOS (token row 0, position row 0)",
            "dims": self.dims,
            "len": self.len,
            "causal": self.causal,
            "vocab": [self.vocab.0, self.vocab.1],
            "tok_embed": m(&self.tok_embed),
            "pos_embed": m(&self.pos_embed),
            "blocks": self.blocks.iter().map(|b| json!({
                "heads": b.heads.iter().map(|h| json!({"wq": m(&h.wq), "wk": m(&h.wk), "wv": m(&h.wv), "wo": m(&h.wo)})).collect::<Vec<_>>(),
                "w_in": m(&b.w_in),
                "w_out": m(&b.w_out),
            })).collect::<Vec<_>>(),
            "unembed": m(&self.unembed),
            "outputs": self.outputs,
        })
    }
}

// ---------------------------------------------------------------- compiler

/// Sublayer index: 0 is the embedding, block b (1-based) is attention 2b - 1, MLP 2b.
fn stages(p: &SeqProgram) -> Vec<usize> {
    let g = &p.graph;
    let c = &p.ctx;
    let mut st = vec![0usize; g.nodes.len()];
    let deps = |a: &Affine, st: &[usize]| a.terms.keys().map(|&m| st[m]).max().unwrap_or(0);
    for (id, node) in g.nodes.iter().enumerate() {
        st[id] = match node {
            Node::Relu(a) => {
                let d = deps(a, &st);
                d + 1 + (d % 2 == 0) as usize // next even
            }
            Node::Input(_) if let Some(body) = c.defs.get(&id) => deps(body, &st),
            Node::Input(_) if let Some(&h) = c.head_of.get(&id) => {
                let h = &c.heads[h];
                let mut d = h.q.onehot.iter().chain(&h.k.onehot).map(|a| deps(a, &st)).max().unwrap_or(0);
                if let HeadKind::Mean(v) = &h.kind {
                    d = d.max(deps(v, &st));
                }
                d + 1 + (d % 2 == 1) as usize // next odd
            }
            Node::Input(_) => 0,
        };
    }
    st
}

pub fn compile(p: &SeqProgram) -> Result<(SeqWeights, SeqCost)> {
    let g = &p.graph;
    let c = &p.ctx;
    let st = stages(p);
    // Live nodes, from the outputs back.
    let mut live = vec![false; g.nodes.len()];
    let mut stack: Vec<usize> = p.outputs.iter().flat_map(|(_, a)| a.terms.keys().copied()).collect();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut live[id], true) {
            continue;
        }
        match &g.nodes[id] {
            Node::Relu(a) => stack.extend(a.terms.keys()),
            Node::Input(_) if let Some(body) = c.defs.get(&id) => stack.extend(body.terms.keys()),
            Node::Input(_) if let Some(&h) = c.head_of.get(&id) => {
                let h = &c.heads[h];
                for a in h.q.onehot.iter().chain(&h.k.onehot) {
                    stack.extend(a.terms.keys());
                }
                if let HeadKind::Mean(v) = &h.kind {
                    stack.extend(v.terms.keys());
                }
            }
            Node::Input(_) => {}
        }
    }
    // Residual dimensions: isbos, one, the embedding, then live nodes.
    let mut dims = vec!["isbos".to_string(), "one".to_string()];
    let mut dim_of: HashMap<usize, usize> = HashMap::new();
    fn add_dim(dim_of: &mut HashMap<usize, usize>, id: usize, name: String, dims: &mut Vec<String>) {
        dim_of.insert(id, dims.len());
        dims.push(name);
    }
    add_dim(&mut dim_of, c.tok_value, p.token_name.clone(), &mut dims);
    for (k, &id) in c.tok_onehot.iter().enumerate() {
        add_dim(&mut dim_of, id, format!("{}={}", p.token_name, c.vocab.0 + k as i64), &mut dims);
    }
    add_dim(&mut dim_of, c.pos_value, "pos".into(), &mut dims);
    for (k, &id) in c.pos_onehot.iter().enumerate() {
        add_dim(&mut dim_of, id, format!("pos={k}"), &mut dims);
    }
    let name_of = |id: usize| -> String {
        for (n, a) in &p.names {
            if a.terms.len() == 1 && a.terms.get(&id) == Some(&1.0) {
                return n.clone();
            }
        }
        format!("n{id}")
    };
    // A relu read only by count decoders is a hidden unit only (the decoder's dimension
    // holds the sum); everything else read later gets a residual dimension.
    let mut read: std::collections::HashSet<usize> = p.outputs.iter().flat_map(|(_, a)| a.terms.keys().copied()).collect();
    for (id, node) in g.nodes.iter().enumerate().filter(|&(id, _)| live[id]) {
        match node {
            Node::Relu(a) => read.extend(a.terms.keys()),
            Node::Input(_) if let Some(&h) = c.head_of.get(&id) => {
                let h = &c.heads[h];
                for a in h.q.onehot.iter().chain(&h.k.onehot) {
                    read.extend(a.terms.keys());
                }
                if let HeadKind::Mean(v) = &h.kind {
                    read.extend(v.terms.keys());
                }
            }
            Node::Input(_) => {}
        }
    }
    for (id, node) in g.nodes.iter().enumerate() {
        if !live[id] || dim_of.contains_key(&id) {
            continue;
        }
        match node {
            Node::Relu(_) if !read.contains(&id) => {}
            Node::Relu(_) => add_dim(&mut dim_of, id, format!("relu{id}"), &mut dims),
            Node::Input(_) if c.defs.contains_key(&id) => add_dim(&mut dim_of, id, format!("{}(count)", name_of(id)), &mut dims),
            Node::Input(_) => add_dim(&mut dim_of, id, format!("{}(head)", name_of(id)), &mut dims),
        }
    }
    let d = dims.len();
    let (isbos, one) = (0usize, 1usize);
    // An affine form as a residual row (constants on `one`).
    let row = |a: &Affine| -> Vec<f64> {
        let mut r = vec![0.0; d];
        r[one] += a.c;
        for (&m, &k) in &a.terms {
            let dm = *dim_of.get(&m).unwrap_or_else(|| panic!("node {m} has no residual dimension"));
            r[dm] += k;
        }
        r
    };
    // Embeddings.
    let v = (c.vocab.1 - c.vocab.0 + 1) as usize;
    let mut tok_embed = Mat::zeros(v + 1, d);
    tok_embed.set(0, isbos, 1.0);
    for t in 0..v {
        tok_embed.set(1 + t, one, 1.0);
        tok_embed.set(1 + t, dim_of[&c.tok_value], (c.vocab.0 + t as i64) as f64);
        tok_embed.set(1 + t, dim_of[&c.tok_onehot[t]], 1.0);
    }
    let mut pos_embed = Mat::zeros(c.len + 1, d);
    for i in 0..c.len {
        pos_embed.set(1 + i, dim_of[&c.pos_value], i as f64);
        pos_embed.set(1 + i, dim_of[&c.pos_onehot[i]], 1.0);
    }
    // Blocks.
    let last = (0..g.nodes.len()).filter(|&i| live[i]).map(|i| st[i]).max().unwrap_or(0);
    let nblocks = last.div_ceil(2);
    let beta = c.sharpness;
    let mut blocks = Vec::new();
    for b in 1..=nblocks {
        let mut heads = Vec::new();
        for h in c.heads.iter().filter(|h| live[h.node] && st[h.node] == 2 * b - 1) {
            let kk = h.k.onehot.len();
            let mut wq = Mat::zeros(kk + 1, d);
            let mut wk = Mat::zeros(kk + 1, d);
            for j in 0..kk {
                let kv = h.k.lo + j as i64;
                let mut qa = Affine::default();
                for (a, oh) in h.q.onehot.iter().enumerate() {
                    if h.holds(h.q.lo + a as i64, kv) {
                        qa = qa.add(oh);
                    }
                }
                wq.row_mut(j).copy_from_slice(&row(&qa.scale(beta)));
                wk.row_mut(j).copy_from_slice(&row(&h.k.onehot[j]));
            }
            // BOS slot: real queries give BOS beta/2 (mean) or beta (count). BOS's own
            // query must read 0 to keep BOS's residual zero: a mean head attends to BOS
            // (value 0 there), a count head away from it (its value is 1 only at BOS).
            let count = matches!(h.kind, HeadKind::Count);
            wq.set(kk, one, if count { beta } else { beta / 2.0 });
            wq.set(kk, isbos, if count { -beta } else { beta });
            wk.set(kk, isbos, 1.0);
            let mut wv = Mat::zeros(1, d);
            match &h.kind {
                HeadKind::Mean(val) => wv.row_mut(0).copy_from_slice(&row(val)),
                HeadKind::Count => wv.set(0, isbos, 1.0),
            }
            let mut wo = Mat::zeros(d, 1);
            wo.set(dim_of[&h.node], 0, 1.0);
            heads.push(AttnHead { wq, wk, wv, wo });
        }
        let units: Vec<usize> =
            (0..g.nodes.len()).filter(|&i| live[i] && st[i] == 2 * b && matches!(g.nodes[i], Node::Relu(_))).collect();
        let mut w_in = Mat::zeros(units.len(), d);
        let mut w_out = Mat::zeros(d, units.len());
        for (u, &id) in units.iter().enumerate() {
            let Node::Relu(a) = &g.nodes[id] else { unreachable!() };
            w_in.row_mut(u).copy_from_slice(&row(a));
            if let Some(&dm) = dim_of.get(&id) {
                w_out.set(dm, u, 1.0);
            }
        }
        // Count decoders: the def dim is a sum of this layer's units.
        for (&def, body) in &c.defs {
            if !live[def] || st[def] != 2 * b {
                continue;
            }
            if !body.c.eq(&0.0) {
                return Err(WirError::new("internal: count decoder with a constant"));
            }
            for (&r, &k) in &body.terms {
                let u = units.iter().position(|&x| x == r).ok_or_else(|| WirError::new("internal: count decoder unit missing"))?;
                w_out.set(dim_of[&def], u, k);
            }
        }
        blocks.push(Block { heads, w_in, w_out });
    }
    let mut unembed = Mat::zeros(p.outputs.len(), d);
    for (o, (_, a)) in p.outputs.iter().enumerate() {
        unembed.row_mut(o).copy_from_slice(&row(a));
    }
    let w = SeqWeights {
        dims,
        len: c.len,
        causal: c.causal,
        vocab: c.vocab,
        tok_embed,
        pos_embed,
        blocks,
        unembed,
        outputs: p.outputs.iter().map(|(n, _)| n.clone()).collect(),
    };
    let cost = w.cost();
    Ok((w, cost))
}

pub fn compile_str(src: &str) -> Result<(SeqProgram, SeqWeights, SeqCost)> {
    let p = lower(src)?;
    let (w, c) = compile(&p)?;
    Ok((p, w, c))
}

// ---------------------------------------------------------------- checks

/// Every sequence of length 1..=max_len over the vocabulary, in order.
pub fn all_sequences(vocab: (i64, i64), max_len: usize) -> impl Iterator<Item = Vec<i64>> {
    let v = (vocab.1 - vocab.0 + 1) as u64;
    (1..=max_len).flat_map(move |n| {
        (0..v.pow(n as u32)).map(move |mut code| {
            (0..n)
                .map(|_| {
                    let t = vocab.0 + (code % v) as i64;
                    code /= v;
                    t
                })
                .collect()
        })
    })
}

/// `count` random sequences with lengths min_len..=max_len (xorshift, seeded).
pub fn random_sequences(vocab: (i64, i64), min_len: usize, max_len: usize, count: u64, seed: u64) -> impl Iterator<Item = Vec<i64>> {
    let mut r = seed.max(1);
    let mut next = move || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    let v = (vocab.1 - vocab.0 + 1) as u64;
    (0..count).map(move |_| {
        let n = min_len + (next() % (max_len - min_len + 1) as u64) as usize;
        (0..n).map(|_| vocab.0 + (next() % v) as i64).collect()
    })
}

/// Reference tasks for the primitive programs: per-position expected outputs.
pub fn task(name: &str, tokens: &[i64]) -> Option<Vec<f64>> {
    let n = tokens.len();
    Some(match name {
        "reverse" => (0..n).map(|i| tokens[n - 1 - i] as f64).collect(),
        "hist" => tokens.iter().map(|t| tokens.iter().filter(|u| *u == t).count() as f64).collect(),
        "sort" => {
            let mut s = tokens.to_vec();
            s.sort();
            s.into_iter().map(|t| t as f64).collect()
        }
        "length" => vec![n as f64; n],
        "prefix_hist" => (0..n).map(|i| tokens[..=i].iter().filter(|u| **u == tokens[i]).count() as f64).collect(),
        _ => return None,
    })
}

pub const TASKS: &[&str] = &["reverse", "hist", "sort", "length", "prefix_hist"];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeqCheck {
    pub sequences: u64,
    /// Sequences where the program (exact attention) differs from the task.
    pub program_wrong: u64,
    /// Sequences where the rounded network output differs from the task.
    pub network_wrong: u64,
    /// Largest |network - program| over every output.
    pub worst: f64,
}

/// Check output `out` against a task on every sequence the iterator gives.
pub fn check(p: &SeqProgram, w: &SeqWeights, task_name: &str, out: usize, seqs: impl Iterator<Item = Vec<i64>>) -> Option<SeqCheck> {
    task(task_name, &[0])?;
    let mut c = SeqCheck::default();
    for s in seqs {
        let want = task(task_name, &s).unwrap();
        let prog = p.eval(&s);
        let net = w.eval(&s);
        c.sequences += 1;
        c.program_wrong += (0..s.len()).any(|i| (prog[i][out] - want[i]).abs() > 1e-9) as u64;
        c.network_wrong += (0..s.len()).any(|i| net[i][out].round() != want[i]) as u64;
        for i in 0..s.len() {
            for o in 0..prog[i].len() {
                c.worst = c.worst.max((net[i][o] - prog[i][o]).abs());
            }
        }
    }
    Some(c)
}

/// Stage-by-stage layout, for reports.
pub fn layout(p: &SeqProgram) -> BTreeMap<usize, (usize, usize)> {
    let st = stages(p);
    let mut m: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for h in &p.ctx.heads {
        m.entry(st[h.node].div_ceil(2)).or_default().0 += 1;
    }
    for (id, n) in p.graph.nodes.iter().enumerate() {
        if matches!(n, Node::Relu(_)) {
            m.entry(st[id] / 2).or_default().1 += 1;
        }
    }
    m
}
