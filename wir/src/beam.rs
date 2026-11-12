//! WIR Beam: one step of beam search, written as a 0.3 attention network.
//!
//! A step has `k` live beams and `v` tokens, so `k * v` candidate tokens: position
//! `p = beam * v + token`. Costs are small integers (lower is better; the host quantizes
//! negative log-probabilities and subtracts the best beam's cost so `s` stays in `0..=cap`).
//! Per candidate the inputs are
//!
//! - `s`: the beam's cost so far (0..=cap),
//! - `l`: the token's cost under the language model (0..=qmax),
//! - `fin`: 1 when the beam already ended. A finished beam offers one candidate, token 0
//!   (the pad / EOS id) at no extra cost; every other token of it costs `dead` = cap + qmax + 1.
//!
//! `c = s + l` (or the finished-beam rule) is the candidate's cost. Two count heads rank
//! it: `below` counts candidates with a lower cost, `ties` counts equal-cost candidates at an
//! earlier position, so ranks are a permutation of 0..k*v. Output slot `r` reads the
//! candidate with rank `r` (one `mean` head per output): its cost, parent beam and token.
//! The first `k` positions are the new beams, best first; the rest is the full ranking. The
//! fourth output is every candidate's own rank (what a page can draw on the candidate grid).
//! With `v >= k` a step never picks a dead candidate unless every beam's costs are capped.

use crate::seq::{self, SeqProgram, SeqRunner, SeqWeights};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub k: usize,
    pub v: usize,
    /// Largest beam cost relative to the best beam.
    pub cap: i64,
    /// Largest token cost.
    pub qmax: i64,
}

impl Default for Config {
    fn default() -> Self {
        Config { k: 4, v: 8, cap: 24, qmax: 15 }
    }
}

impl Config {
    pub fn len(&self) -> usize {
        self.k * self.v
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Cost of a candidate that cannot be chosen (a finished beam's non-pad token).
    pub fn dead(&self) -> i64 {
        self.cap + self.qmax + 1
    }
}

pub fn program(c: &Config) -> String {
    format!(
        "wir 0.3
# One beam search step: rank the {k} x {v} candidates by cost, keep the best {k}.
sequence len {n}
input s: token in [0, {cap}]
input l: token in [0, {qmax}]
input fin: token in [0, 1]
table col(x) = mod(x, {v})
table parent(x) = div(x, {v})
let c = select(fin, select(eq(col(pos), 0), s, {dead}), s + l)
let below = count(lt(k(c), q(c)))
let ties = count(and(k(c) == q(c), lt(k(pos), q(pos))))
let place = below + ties
output cost = mean(c, k(place) == q(pos))
output from = mean(parent(pos), k(place) == q(pos))
output tok = mean(col(pos), k(place) == q(pos))
output rank = place
",
        k = c.k,
        v = c.v,
        n = c.len(),
        cap = c.cap,
        qmax = c.qmax,
        dead = c.dead()
    )
}

/// One input step: `s[beam]`, `l[beam][token]`, `fin[beam]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub s: Vec<i64>,
    pub l: Vec<Vec<i64>>,
    pub fin: Vec<i64>,
}

/// A chosen candidate: new cost, parent beam, token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pick {
    pub cost: i64,
    pub from: usize,
    pub tok: usize,
}

/// Candidate costs in position order (`beam * v + token`).
pub fn costs(cfg: &Config, st: &Step) -> Vec<i64> {
    let mut out = Vec::with_capacity(cfg.len());
    for i in 0..cfg.k {
        for j in 0..cfg.v {
            out.push(if st.fin[i] != 0 {
                if j == 0 {
                    st.s[i]
                } else {
                    cfg.dead()
                }
            } else {
                st.s[i] + st.l[i][j]
            });
        }
    }
    out
}

/// Rank of every candidate in position order (0 = best; cost, then position).
pub fn ranks(cfg: &Config, st: &Step) -> Vec<usize> {
    let c = costs(cfg, st);
    let mut order: Vec<usize> = (0..c.len()).collect();
    order.sort_by_key(|&p| (c[p], p));
    let mut r = vec![0; c.len()];
    for (rank, p) in order.into_iter().enumerate() {
        r[p] = rank;
    }
    r
}

/// The whole ranking, best first (cost, then position).
pub fn reference(cfg: &Config, st: &Step) -> Vec<Pick> {
    let c = costs(cfg, st);
    let mut order: Vec<usize> = (0..c.len()).collect();
    order.sort_by_key(|&p| (c[p], p));
    order.into_iter().map(|p| Pick { cost: c[p], from: p / cfg.v, tok: p % cfg.v }).collect()
}

pub fn tokens(cfg: &Config, st: &Step) -> Vec<Vec<i64>> {
    let mut t = Vec::with_capacity(cfg.len());
    for i in 0..cfg.k {
        for j in 0..cfg.v {
            t.push(vec![st.s[i], st.l[i][j], st.fin[i]]);
        }
    }
    t
}

pub fn build(cfg: &Config) -> crate::error::Result<(SeqProgram, SeqWeights, Vec<f64>)> {
    let (p, w, _) = seq::compile_str(&program(cfg))?;
    let bound = seq::error_bound(&p).outputs;
    Ok((p, w, bound))
}

/// Random step; `tight` draws costs from a tiny range so ties are common.
pub fn random_step(cfg: &Config, rng: &mut u64, tight: bool) -> Step {
    let mut next = |lo: i64, hi: i64| -> i64 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        lo + (*rng % (hi - lo + 1) as u64) as i64
    };
    let (smax, qmax) = if tight { (cfg.cap.min(2), cfg.qmax.min(2)) } else { (cfg.cap, cfg.qmax) };
    Step {
        s: (0..cfg.k).map(|_| next(0, smax)).collect(),
        l: (0..cfg.k).map(|_| (0..cfg.v).map(|_| next(0, qmax)).collect()).collect(),
        fin: (0..cfg.k).map(|_| (next(0, 3) == 0) as i64).collect(),
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeamCheck {
    pub steps: u64,
    /// Steps where any slot of the full ranking differs from the reference.
    pub wrong: u64,
    pub worst: f64,
}

/// Network against the reference on steps: the full ranking (all `k * v` slots, three outputs).
pub fn check(cfg: &Config, w: &SeqWeights, steps: impl Iterator<Item = Step>) -> BeamCheck {
    let runner = SeqRunner::new(w);
    let mut r = BeamCheck::default();
    for st in steps {
        let net = runner.eval_multi(&tokens(cfg, &st));
        let want = reference(cfg, &st);
        r.steps += 1;
        let mut wrong = false;
        for (p, rk) in ranks(cfg, &st).into_iter().enumerate() {
            r.worst = r.worst.max((net[p][3] - rk as f64).abs());
            wrong |= net[p][3].round() != rk as f64;
        }
        for (slot, p) in want.iter().enumerate() {
            for (o, v) in [p.cost as f64, p.from as f64, p.tok as f64].into_iter().enumerate() {
                r.worst = r.worst.max((net[slot][o] - v).abs());
                wrong |= net[slot][o].round() != v;
            }
        }
        r.wrong += wrong as u64;
    }
    r
}

/// Every step of a small configuration (all `s`, `l` and `fin`).
pub fn all_steps(cfg: &Config) -> impl Iterator<Item = Step> {
    let cfg = *cfg;
    let per_beam = (cfg.cap + 1) as u128 * ((cfg.qmax + 1) as u128).pow(cfg.v as u32) * 2;
    let total = per_beam.pow(cfg.k as u32);
    (0..total).map(move |mut n| {
        let mut st = Step { s: vec![], l: vec![], fin: vec![] };
        for _ in 0..cfg.k {
            st.fin.push((n % 2) as i64);
            n /= 2;
            st.s.push((n % (cfg.cap + 1) as u128) as i64);
            n /= (cfg.cap + 1) as u128;
            let mut row = vec![];
            for _ in 0..cfg.v {
                row.push((n % (cfg.qmax + 1) as u128) as i64);
                n /= (cfg.qmax + 1) as u128;
            }
            st.l.push(row);
        }
        st
    })
}
