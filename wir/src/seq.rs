//! WIR 0.3: sequence programs with attention heads, compiled to a small transformer.
//!
//! A 0.3 program reads a sequence of tokens, each a tuple of integer inputs. Every
//! `let`/`output` is a per-position value written with the 0.2 combinators, plus
//! attention reads (after RASP's aggregate and selector width, Weiss et al. 2021; the
//! compilation follows Tracr, Lindner et al. 2023):
//!
//!   mean(v, cond)          v averaged over the key positions where cond holds (0 if none)
//!   count(cond)            how many key positions cond holds at
//!   best(v, cond, grade)   v averaged over the matching keys with the highest grade
//!
//! A condition is `true`, one clause, or `and(clause, ..)`. A clause relates a key side
//! `k(expr)` (the expression at the key position) and a query side `q(expr)`:
//! `k(a) == q(b)`, `eq/ne/lt/le/gt/ge(side, side)`, or a call of a program table with
//! sides and constants as arguments (it holds where the table is nonzero; one side may
//! be missing). A grade is such a table call with integer values >= 0. Sides need
//! integer intervals.
//!
//! Tables are compile-time functions of integers: `table name(a, b) = expr` (numbers,
//! parameters, + - * /, abs min max div mod, eq ne lt le gt ge, and or not select, other
//! tables) or `table name(a) = [v0, v1, ..]` (nested lists for more parameters, indexed
//! from 0, 0 outside). Called outside a condition with per-position arguments, a table
//! is a lookup: free when its argument is an input or `pos` (written by the embedding),
//! otherwise through a one-hot of the argument; with two non-constant arguments it
//! expands over the one with fewer values (`select` per value).
//!
//! ```text
//! wir 0.3
//! sequence len 8                  # max length; `causal` keeps keys at or before the query
//! input tok: token in [0, 7]      # one or more token inputs
//! let n = count(true)
//! output rev = mean(tok, k(pos) == q(n - pos - 1))
//! ```
//!
//! The network is a norm-free transformer: blocks of attention then a one-hidden-layer
//! ReLU MLP, each adding into a residual stream. Position 0 is a BOS token. Residual
//! dimensions: `isbos`, `one` (1 at every real position, 0 at BOS, and the only way
//! constants enter), each input's value and one-hot, the position's, embedding lookups,
//! then one per relu node, head value and count decoder. Every dimension but `isbos`
//! should be 0 at BOS, so a head with no matching key reads 0 there: BOS's own query
//! reads BOS in a mean head and the real keys' 0 in a count head (a causal count head
//! can only read BOS's 1), and every relu that could be nonzero at BOS gets a -M isbos
//! gate sized by `error_bound`.
//!
//! Scores (in units of the sharpness beta): each condition clause adds M when it holds
//! (M = 1 + the largest grade), the grade adds its value, BOS gets M m - 1/2 for a mean
//! (m clauses) and m for a count. Winners lead every other key by at least beta and BOS
//! by beta/2, and with no match BOS leads every key by beta/2, so this is hard attention
//! up to a leak of about e^(-beta/2). `count` reads 1/(w + 1) from a head that always
//! includes BOS and turns it back into w with a piecewise-linear MLP, exact at the
//! integers. Heads with the same scores share one attention head (several values).
//!
//! The program's own semantics (`SeqProgram::eval_multi`) uses exact hard attention; the
//! compiled weights (`SeqWeights::eval_multi`, or the sparse `SeqRunner`) use softmax.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{json, Value};

use crate::deep::{Affine, Cursor, Graph, Node, Scope};
use crate::error::{Result, WirError};
use crate::lexer::{split_statements, Tok};
use crate::model::Mat;

pub const VERSION: f64 = 0.3;
pub const HEADS: &[&str] = &["mean", "count", "best"];
pub const DEFAULT_SHARPNESS: f64 = 40.0;
/// Most values a one-hot side may take.
pub const MAX_ONEHOT: usize = 512;
const CMPS: &[&str] = &["eq", "ne", "lt", "le", "gt", "ge"];

// ---------------------------------------------------------------- tables

#[derive(Debug, Clone, PartialEq)]
pub enum TExpr {
    Num(f64),
    Var(usize),
    Neg(Box<TExpr>),
    Bin(char, Box<TExpr>, Box<TExpr>),
    Call(String, Vec<TExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TList {
    Leaf(f64),
    List(Vec<TList>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableBody {
    Expr(TExpr),
    List(TList),
}

/// A compile-time integer function declared with `table`.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub name: String,
    pub params: Vec<String>,
    pub body: TableBody,
}

const TFUNCS: &[&str] =
    &["abs", "min", "max", "div", "mod", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not", "select"];

impl Table {
    pub fn eval(&self, args: &[f64], tables: &HashMap<String, Table>) -> std::result::Result<f64, String> {
        if args.len() != self.params.len() {
            return Err(format!("table `{}` takes {} arguments, got {}", self.name, self.params.len(), args.len()));
        }
        match &self.body {
            TableBody::List(l) => {
                let mut cur = l;
                for &a in args {
                    cur = match cur {
                        TList::List(items) if a >= 0.0 && a.fract() == 0.0 && (a as usize) < items.len() => &items[a as usize],
                        TList::List(_) => return Ok(0.0),
                        TList::Leaf(_) => return Err(format!("table `{}` has fewer list levels than parameters", self.name)),
                    };
                }
                match cur {
                    TList::Leaf(v) => Ok(*v),
                    TList::List(_) => Err(format!("table `{}` has more list levels than parameters", self.name)),
                }
            }
            TableBody::Expr(e) => eval_texpr(e, args, tables, 0),
        }
    }
}

fn eval_texpr(e: &TExpr, vars: &[f64], tables: &HashMap<String, Table>, depth: usize) -> std::result::Result<f64, String> {
    if depth > 64 {
        return Err("table calls nest too deeply".into());
    }
    let ev = |x: &TExpr| eval_texpr(x, vars, tables, depth + 1);
    let b = |x: bool| x as i64 as f64;
    Ok(match e {
        TExpr::Num(v) => *v,
        TExpr::Var(i) => vars[*i],
        TExpr::Neg(x) => -ev(x)?,
        TExpr::Bin(op, x, y) => {
            let (x, y) = (ev(x)?, ev(y)?);
            match op {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                _ => {
                    if y == 0.0 {
                        return Err("table division by zero".into());
                    }
                    x / y
                }
            }
        }
        TExpr::Call(f, a) => {
            let v: Vec<f64> = a.iter().map(ev).collect::<std::result::Result<_, _>>()?;
            let n = |k: usize| -> std::result::Result<(), String> {
                if v.len() == k { Ok(()) } else { Err(format!("`{f}` takes {k} argument(s) in a table")) }
            };
            match f.as_str() {
                "abs" => { n(1)?; v[0].abs() }
                "min" | "max" if v.is_empty() => return Err(format!("`{f}` needs arguments")),
                "min" => v.iter().cloned().fold(f64::INFINITY, f64::min),
                "max" => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "div" | "mod" => {
                    n(2)?;
                    if v[1] == 0.0 {
                        return Err("table div/mod by zero".into());
                    }
                    let q = (v[0] / v[1]).floor();
                    if f == "div" { q } else { v[0] - q * v[1] }
                }
                "eq" => { n(2)?; b(v[0] == v[1]) }
                "ne" => { n(2)?; b(v[0] != v[1]) }
                "lt" => { n(2)?; b(v[0] < v[1]) }
                "le" => { n(2)?; b(v[0] <= v[1]) }
                "gt" => { n(2)?; b(v[0] > v[1]) }
                "ge" => { n(2)?; b(v[0] >= v[1]) }
                "and" => b(v.iter().all(|x| *x != 0.0)),
                "or" => b(v.iter().any(|x| *x != 0.0)),
                "not" => { n(1)?; b(v[0] == 0.0) }
                "select" => { n(3)?; if v[0] != 0.0 { v[1] } else { v[2] } }
                t => tables.get(t).ok_or_else(|| format!("unknown function `{t}` in a table"))?.eval(&v, tables)?,
            }
        }
    })
}

fn parse_texpr(p: &mut Cursor, params: &[String], tables: &HashMap<String, Table>) -> Result<TExpr> {
    let mut acc = texpr_term(p, params, tables)?;
    loop {
        let op = if p.eat(&Tok::Plus) { '+' } else if p.eat(&Tok::Minus) { '-' } else { return Ok(acc) };
        acc = TExpr::Bin(op, Box::new(acc), Box::new(texpr_term(p, params, tables)?));
    }
}

fn texpr_term(p: &mut Cursor, params: &[String], tables: &HashMap<String, Table>) -> Result<TExpr> {
    let mut acc = texpr_unary(p, params, tables)?;
    loop {
        let op = if p.eat(&Tok::Star) { '*' } else if p.eat(&Tok::Slash) { '/' } else { return Ok(acc) };
        acc = TExpr::Bin(op, Box::new(acc), Box::new(texpr_unary(p, params, tables)?));
    }
}

fn texpr_unary(p: &mut Cursor, params: &[String], tables: &HashMap<String, Table>) -> Result<TExpr> {
    if p.eat(&Tok::Minus) {
        return Ok(TExpr::Neg(Box::new(texpr_unary(p, params, tables)?)));
    }
    p.eat(&Tok::Plus);
    match p.next()? {
        Tok::Num(v) => Ok(TExpr::Num(v)),
        Tok::LParen => {
            let e = parse_texpr(p, params, tables)?;
            p.expect(&Tok::RParen)?;
            Ok(e)
        }
        Tok::Ident(name) if p.peek() == Some(&Tok::LParen) => {
            if !TFUNCS.contains(&name.as_str()) && !tables.contains_key(&name) {
                return Err(p.err(&format!("unknown function `{name}` in a table (have {} and earlier tables)", TFUNCS.join(", "))));
            }
            p.expect(&Tok::LParen)?;
            let mut args = vec![];
            if !p.eat(&Tok::RParen) {
                loop {
                    args.push(parse_texpr(p, params, tables)?);
                    if p.eat(&Tok::RParen) {
                        break;
                    }
                    p.expect(&Tok::Comma)?;
                }
            }
            Ok(TExpr::Call(name, args))
        }
        Tok::Ident(name) => match params.iter().position(|x| *x == name) {
            Some(i) => Ok(TExpr::Var(i)),
            None => Err(p.err(&format!("`{name}` is not a parameter of this table"))),
        },
        t => Err(p.err(&format!("expected a table expression, found {}", t.describe()))),
    }
}

fn parse_tlist(p: &mut Cursor) -> Result<TList> {
    if p.eat(&Tok::LBracket) {
        let mut items = vec![];
        if !p.eat(&Tok::RBracket) {
            loop {
                items.push(parse_tlist(p)?);
                if p.eat(&Tok::RBracket) {
                    break;
                }
                p.expect(&Tok::Comma)?;
            }
        }
        return Ok(TList::List(items));
    }
    Ok(TList::Leaf(p.signed()?))
}

// ---------------------------------------------------------------- program

/// Where a one-hot comes from for free: a token input or the position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Src {
    Input(usize),
    Pos,
}

/// A token input (or the position): its range, value node and one-hot nodes.
#[derive(Debug, Clone, Default)]
pub struct Source {
    pub name: String,
    pub lo: i64,
    pub hi: i64,
    pub value: usize,
    pub onehot: Vec<usize>,
}

/// A value the embedding writes: `values[t - lo]` for the source's value t.
#[derive(Debug, Clone)]
pub struct Lookup {
    pub node: usize,
    pub src: Src,
    pub values: Vec<f64>,
}

/// One side of a clause: its value, smallest value and one-hot (entry c is 1 when the
/// value is lo + c). A missing side is the constant 0.
#[derive(Debug, Clone)]
pub struct Side {
    pub value: Affine,
    pub lo: i64,
    pub onehot: Vec<Affine>,
}

/// table[a][b]: the clause's value at query side value q.lo + a and key side value k.lo + b.
#[derive(Debug, Clone)]
pub struct Clause {
    pub q: Side,
    pub k: Side,
    pub table: Vec<Vec<f64>>,
}

impl Clause {
    fn at(&self, qv: i64, kv: i64) -> f64 {
        let (a, b) = (qv - self.q.lo, kv - self.k.lo);
        if a < 0 || b < 0 || a as usize >= self.table.len() || b as usize >= self.table[0].len() {
            return 0.0;
        }
        self.table[a as usize][b as usize]
    }

    fn largest(&self) -> f64 {
        self.table.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()))
    }
}

#[derive(Debug, Clone)]
pub enum HeadKind {
    Mean(Affine),
    Count,
}

/// An attention head: keys where every condition clause holds (nonzero) match; a mean
/// reads the matches with the highest grade.
#[derive(Debug, Clone)]
pub struct Head {
    /// Graph node holding the head's output (mean, or 1/(w + 1) for count).
    pub node: usize,
    pub kind: HeadKind,
    pub conds: Vec<Clause>,
    pub grade: Option<Clause>,
    /// Most keys that can match one query (count decoders cover 0..=wmax).
    pub wmax: usize,
    /// Equal for heads with the same scores (they share one attention head).
    pub sig: Vec<u64>,
    pub line: usize,
}

impl Head {
    /// M: one condition clause outweighs every grade.
    pub fn m_unit(&self) -> f64 {
        1.0 + self.grade.as_ref().map_or(0.0, |g| g.largest())
    }

    /// BOS's score in units of beta, at a real query.
    pub fn bos_units(&self) -> f64 {
        let m = self.conds.len() as f64;
        match self.kind {
            HeadKind::Count => m,
            HeadKind::Mean(_) => self.m_unit() * m - 0.5,
        }
    }
}

/// Parse state shared with the expression parser.
#[derive(Debug, Clone, Default)]
pub struct SeqCtx {
    pub len: usize,
    pub causal: bool,
    pub sharpness: f64,
    pub inputs: Vec<Source>,
    pub pos: Source,
    pub lookups: Vec<Lookup>,
    pub tables: HashMap<String, Table>,
    pub heads: Vec<Head>,
    /// Graph input nodes that are head outputs, to their head.
    pub head_of: HashMap<usize, usize>,
    /// Nodes an MLP writes directly: value = the affine (over relu nodes of one layer).
    pub defs: HashMap<usize, Affine>,
    pub lookup_of: HashMap<usize, usize>,
    onehots: HashMap<Vec<u64>, Side>,
    lookup_cache: HashMap<(Src, Vec<u64>), usize>,
    head_cache: HashMap<Vec<u64>, Affine>,
    next_input: usize,
}

fn affine_key(v: &Affine) -> Vec<u64> {
    v.terms.iter().flat_map(|(&n, k)| [n as u64, k.to_bits()]).chain([v.c.to_bits()]).collect()
}

pub fn handles(ctx: &SeqCtx, f: &str) -> bool {
    HEADS.contains(&f) || ctx.tables.contains_key(f)
}

impl SeqCtx {
    fn fresh_input(&mut self, g: &mut Graph, range: (f64, f64), line: usize) -> usize {
        self.next_input += 1;
        *g.input(self.next_input - 1, range, line).terms.keys().next().unwrap()
    }

    fn source(&self, s: Src) -> &Source {
        match s {
            Src::Input(i) => &self.inputs[i],
            Src::Pos => &self.pos,
        }
    }

    /// `v` is a source's value plus an integer: (source, offset).
    fn direct(&self, v: &Affine) -> Option<(Src, i64)> {
        if v.terms.len() != 1 || v.c.fract() != 0.0 {
            return None;
        }
        let (&n, &k) = v.terms.iter().next().unwrap();
        if k != 1.0 {
            return None;
        }
        let s = if n == self.pos.value { Src::Pos } else { Src::Input(self.inputs.iter().position(|s| s.value == n)?) };
        Some((s, v.c as i64))
    }

    /// One-hot of an integer value: free for a source (embedding dimensions), otherwise
    /// hats r(c - 1) - 2 r(c) + r(c + 1) over r(t) = relu(v - t).
    fn side(&mut self, g: &mut Graph, v: &Affine, line: usize) -> std::result::Result<Side, String> {
        let key = affine_key(v);
        if let Some(s) = self.onehots.get(&key) {
            return Ok(s.clone());
        }
        let (lo, hi) = g.range_of(v);
        if !lo.is_finite() || !hi.is_finite() || lo.fract() != 0.0 || hi.fract() != 0.0 {
            return Err(format!("attention sides and lookups need integer intervals, got [{lo}, {hi}]"));
        }
        let count = (hi - lo) as usize + 1;
        if count > MAX_ONEHOT {
            return Err(format!("a side or lookup argument takes {count} values (at most {MAX_ONEHOT})"));
        }
        let onehot = if let Some((src, off)) = self.direct(v) {
            let s = self.source(src);
            (0..count)
                .map(|i| {
                    let x = lo as i64 + i as i64 - off - s.lo;
                    if x >= 0 && (x as usize) < s.onehot.len() { Affine::node(s.onehot[x as usize]) } else { Affine::default() }
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

    fn constant_side(c: i64) -> Side {
        Side { value: Affine::constant(c as f64), lo: c, onehot: vec![Affine::constant(1.0)] }
    }

    /// An embedding-written lookup of a source.
    fn lookup(&mut self, g: &mut Graph, src: Src, values: Vec<f64>, line: usize) -> Affine {
        let key = (src, values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        if let Some(&n) = self.lookup_cache.get(&key) {
            return Affine::node(n);
        }
        let lo = values.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0);
        let node = self.fresh_input(g, (lo, hi), line);
        self.lookup_of.insert(node, self.lookups.len());
        self.lookups.push(Lookup { node, src, values });
        self.lookup_cache.insert(key, node);
        Affine::node(node)
    }
}

/// A parsed and lowered 0.3 program.
#[derive(Debug, Clone)]
pub struct SeqProgram {
    pub ctx: SeqCtx,
    pub graph: Graph,
    pub outputs: Vec<(String, Affine)>,
    pub names: Vec<(String, Affine)>,
}

/// True when the first statement is a `wir 0.3` header.
pub fn is_seq_source(src: &str) -> bool {
    let Ok(stmts) = split_statements(src) else { return false };
    matches!(stmts.first().map(|s| s.tokens.iter().map(|t| &t.tok).collect::<Vec<_>>()).as_deref(),
        Some([Tok::Ident(w), Tok::Num(v)]) if w == "wir" && *v == VERSION)
}

const RESERVED: &[&str] = &["k", "q", "true", "sequence", "token", "table", "pos"];

pub fn lower(src: &str) -> Result<SeqProgram> {
    let stmts = split_statements(src)?;
    let mut g = Graph::default();
    let mut ctx = SeqCtx { sharpness: DEFAULT_SHARPNESS, ..Default::default() };
    let mut env: HashMap<String, Affine> = HashMap::new();
    let mut outputs = vec![];
    let mut names = vec![];
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
        let fresh = |p: &Cursor, name: &str, env: &HashMap<String, Affine>, ctx: &SeqCtx| -> Result<()> {
            if HEADS.contains(&name) || RESERVED.contains(&name) || ctx.tables.contains_key(name) {
                return Err(p.err(&format!("`{name}` is reserved or a table")));
            }
            p.fresh(name, env)
        };
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
                let value = ctx.fresh_input(&mut g, (0.0, ctx.len as f64 - 1.0), s.line);
                let onehot = (0..ctx.len).map(|_| ctx.fresh_input(&mut g, (0.0, 1.0), s.line)).collect();
                ctx.pos = Source { name: "pos".into(), lo: 0, hi: ctx.len as i64 - 1, value, onehot };
                env.insert("pos".into(), Affine::node(value));
            }
            "input" => {
                if !sequence {
                    return Err(p.err("declare `sequence len N` before the inputs"));
                }
                if !outputs.is_empty() || env.len() > 1 + ctx.inputs.len() {
                    return Err(p.err("declare inputs before `let` and `output`"));
                }
                let name = p.ident()?;
                fresh(&p, &name, &env, &ctx)?;
                p.expect(&Tok::Colon)?;
                if p.ident()? != "token" || !p.eat_ident("in") {
                    return Err(p.err("sequence inputs are `input name: token in [lo, hi]`"));
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
                let value = ctx.fresh_input(&mut g, (lo, hi), s.line);
                let onehot = (lo as i64..=hi as i64).map(|_| ctx.fresh_input(&mut g, (0.0, 1.0), s.line)).collect();
                ctx.inputs.push(Source { name: name.clone(), lo: lo as i64, hi: hi as i64, value, onehot });
                env.insert(name, Affine::node(value));
            }
            "table" => {
                let name = p.ident()?;
                fresh(&p, &name, &env, &ctx)?;
                if TFUNCS.contains(&name.as_str()) || CMPS.contains(&name.as_str()) || crate::deep::FUNCTIONS.contains(&name.as_str()) {
                    return Err(p.err(&format!("`{name}` is a built-in function")));
                }
                p.expect(&Tok::LParen)?;
                let mut params = vec![p.ident()?];
                while p.eat(&Tok::Comma) {
                    params.push(p.ident()?);
                }
                p.expect(&Tok::RParen)?;
                p.expect(&Tok::Eq)?;
                let body = if p.peek() == Some(&Tok::LBracket) {
                    TableBody::List(parse_tlist(&mut p)?)
                } else {
                    TableBody::Expr(parse_texpr(&mut p, &params, &ctx.tables)?)
                };
                p.end()?;
                ctx.tables.insert(name.clone(), Table { name, params, body });
            }
            "let" | "output" => {
                if ctx.inputs.is_empty() {
                    return Err(p.err("declare a token input before `let` and `output`"));
                }
                let name = p.ident()?;
                fresh(&p, &name, &env, &ctx)?;
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
            other => return Err(p.err(&format!("unknown statement `{other}` (0.3 has sequence, input, table, let, output)"))),
        }
    }
    if ctx.inputs.is_empty() {
        return Err(WirError::new("a sequence program needs `input name: token in [lo, hi]`"));
    }
    if outputs.is_empty() {
        return Err(WirError::new("a sequence program needs at least one output"));
    }
    Ok(SeqProgram { ctx, graph: g, outputs, names })
}

/// A parsed argument of a clause or table call.
enum Arg {
    Key(Affine),
    Query(Affine),
    Const(f64),
}

/// Heads (`mean`, `count`, `best`) and table lookups, called from the expression parser.
pub(crate) fn call(p: &mut Cursor, f: &str, sc: &mut Scope) -> Result<Affine> {
    let line = p.s.line;
    if !HEADS.contains(&f) {
        return lookup_call(p, f, sc);
    }
    p.expect(&Tok::LParen)?;
    let value = if f != "count" {
        let v = p.expr(sc)?;
        p.expect(&Tok::Comma)?;
        Some(v)
    } else {
        None
    };
    let mut raw: Vec<(String, Vec<Arg>)> = vec![];
    if !p.eat_ident("true") {
        let and = matches!(p.peek(), Some(Tok::Ident(w)) if w == "and") && matches!(p.t.get(p.i + 1).map(|t| &t.tok), Some(Tok::LParen));
        if and {
            p.ident()?;
            p.expect(&Tok::LParen)?;
            loop {
                raw.push(clause(p, sc)?);
                if p.eat(&Tok::RParen) {
                    break;
                }
                p.expect(&Tok::Comma)?;
            }
        } else {
            raw.push(clause(p, sc)?);
        }
    }
    let grade_raw = if f == "best" {
        p.expect(&Tok::Comma)?;
        Some(clause(p, sc)?)
    } else {
        None
    };
    p.expect(&Tok::RParen)?;
    let g = &mut *sc.g;
    let ctx = sc.seq.as_deref_mut().unwrap();
    let build = |ctx: &mut SeqCtx, g: &mut Graph, (name, args): (String, Vec<Arg>), grade: bool| -> std::result::Result<Clause, String> {
        let (mut qv, mut kv) = (None, None);
        for a in &args {
            match a {
                Arg::Key(v) if kv.is_none() => kv = Some(v.clone()),
                Arg::Query(v) if qv.is_none() => qv = Some(v.clone()),
                Arg::Key(_) | Arg::Query(_) => return Err("a clause has at most one `k(..)` and one `q(..)` side".into()),
                Arg::Const(_) => {}
            }
        }
        if qv.is_none() && kv.is_none() {
            return Err("a clause needs a `k(..)` or `q(..)` side".into());
        }
        let q = match qv { Some(v) => ctx.side(g, &v, line)?, None => SeqCtx::constant_side(0) };
        let k = match kv { Some(v) => ctx.side(g, &v, line)?, None => SeqCtx::constant_side(0) };
        let (kq, kk) = (q.onehot.len(), k.onehot.len());
        let mut table = vec![vec![0.0; kk]; kq];
        for (a, row) in table.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                let vals: Vec<f64> = args
                    .iter()
                    .map(|x| match x {
                        Arg::Key(_) => (k.lo + b as i64) as f64,
                        Arg::Query(_) => (q.lo + a as i64) as f64,
                        Arg::Const(c) => *c,
                    })
                    .collect();
                let v = match name.as_str() {
                    "eq" => (vals[0] == vals[1]) as i64 as f64,
                    "ne" => (vals[0] != vals[1]) as i64 as f64,
                    "lt" => (vals[0] < vals[1]) as i64 as f64,
                    "le" => (vals[0] <= vals[1]) as i64 as f64,
                    "gt" => (vals[0] > vals[1]) as i64 as f64,
                    "ge" => (vals[0] >= vals[1]) as i64 as f64,
                    t => ctx.tables[t].eval(&vals, &ctx.tables)?,
                };
                *cell = if grade {
                    if v < 0.0 || v.fract() != 0.0 || !v.is_finite() {
                        return Err(format!("grades are integers >= 0; `{name}` gives {v}"));
                    }
                    v
                } else {
                    (v != 0.0) as i64 as f64
                };
            }
        }
        Ok(Clause { q, k, table })
    };
    let mut conds = vec![];
    for r in raw {
        conds.push(build(ctx, g, r, false).map_err(|m| p.err(&m))?);
    }
    let grade = match grade_raw {
        Some(r) => Some(build(ctx, g, r, true).map_err(|m| p.err(&m))?),
        None => None,
    };
    // Most keys one query can match: a condition on the key's position allows at most
    // as many keys as its table row has nonzero entries.
    let mut wmax = ctx.len;
    for c in &conds {
        if let Some((Src::Pos, _)) = ctx.direct(&c.k.value) {
            let row = c.table.iter().map(|r| r.iter().filter(|v| **v != 0.0).count()).max().unwrap_or(0);
            wmax = wmax.min(row);
        }
    }
    let mut sig = vec![value.is_none() as u64];
    for c in conds.iter().chain(&grade) {
        sig.push(u64::MAX);
        sig.extend(affine_key(&c.q.value));
        sig.push(u64::MAX - 1);
        sig.extend(affine_key(&c.k.value));
        sig.extend(c.table.iter().flatten().map(|v| v.to_bits()));
    }
    sig.push(grade.is_some() as u64);
    let mut cache_key = sig.clone();
    if let Some(v) = &value {
        cache_key.push(u64::MAX - 2);
        cache_key.extend(affine_key(v));
    }
    if let Some(a) = ctx.head_cache.get(&cache_key) {
        return Ok(a.clone());
    }
    let n = wmax as f64;
    let result = match value {
        Some(v) => {
            let (lo, hi) = g.range_of(&v);
            let node = ctx.fresh_input(g, (lo.min(0.0), hi.max(0.0)), line);
            ctx.head_of.insert(node, ctx.heads.len());
            ctx.heads.push(Head { node, kind: HeadKind::Mean(v), conds, grade, wmax, sig, line });
            Affine::node(node)
        }
        None => {
            // h = 1 / (w + 1) for w = 0..wmax. Decode with the line through the points
            // (1/(w+1), w): f(h) = wmax + s0 relu(h - x0) + sum of slope changes at each point.
            let x0 = 1.0 / (n + 1.0);
            let node = ctx.fresh_input(g, (x0, 1.0), line);
            let h = Affine::node(node);
            ctx.head_of.insert(node, ctx.heads.len());
            ctx.heads.push(Head { node, kind: HeadKind::Count, conds, grade, wmax, sig, line });
            if wmax == 0 {
                Affine::default()
            } else {
                let pts: Vec<(f64, f64)> = (0..=wmax).rev().map(|w| (1.0 / (w as f64 + 1.0), w as f64)).collect();
                let slope = |a: usize| (pts[a + 1].1 - pts[a].1) / (pts[a + 1].0 - pts[a].0);
                let first = g.unit(&h, line).unwrap().plus(-x0); // relu(h - x0) on its own unit
                let mut body = first.scale(slope(0));
                for a in 1..pts.len() - 1 {
                    let r = g.relu(&h.plus(-pts[a].0), line);
                    body = body.add(&r.scale(slope(a) - slope(a - 1)));
                }
                let d = ctx.fresh_input(g, (-n, 0.0), line);
                ctx.defs.insert(d, body);
                Affine::node(d).plus(n)
            }
        }
    };
    ctx.head_cache.insert(cache_key, result.clone());
    Ok(result)
}

/// One clause: `k(a) == q(b)`, `cmp(side, side)` or `table(args)`.
fn clause(p: &mut Cursor, sc: &mut Scope) -> Result<(String, Vec<Arg>)> {
    let named = matches!(p.peek(), Some(Tok::Ident(w)) if w != "k" && w != "q")
        && matches!(p.t.get(p.i + 1).map(|t| &t.tok), Some(Tok::LParen));
    if named {
        let name = p.ident()?;
        let known = CMPS.contains(&name.as_str()) || sc.seq.as_deref().unwrap().tables.contains_key(&name);
        if !known {
            return Err(p.err(&format!("`{name}` is not a comparison or a table")));
        }
        p.expect(&Tok::LParen)?;
        let mut args = vec![];
        loop {
            args.push(arg(p, sc)?);
            if p.eat(&Tok::RParen) {
                break;
            }
            p.expect(&Tok::Comma)?;
        }
        if CMPS.contains(&name.as_str()) && (args.len() != 2 || args.iter().any(|a| matches!(a, Arg::Const(_)))) {
            return Err(p.err(&format!("`{name}` in a condition compares two sides: {name}(k(..), q(..))")));
        }
        return Ok((name, args));
    }
    let a = arg(p, sc)?;
    p.expect(&Tok::EqEq)?;
    let b = arg(p, sc)?;
    if matches!(a, Arg::Const(_)) || matches!(b, Arg::Const(_)) {
        return Err(p.err("`==` in a condition compares two sides"));
    }
    Ok(("eq".into(), vec![a, b]))
}

fn arg(p: &mut Cursor, sc: &mut Scope) -> Result<Arg> {
    let side = matches!(p.peek(), Some(Tok::Ident(w)) if w == "k" || w == "q");
    if !side {
        return match p.signed() {
            Ok(v) => Ok(Arg::Const(v)),
            Err(_) => Err(p.err("condition arguments are `k(..)`, `q(..)` or numbers")),
        };
    }
    let key = p.ident()? == "k";
    p.expect(&Tok::LParen)?;
    let v = p.expr(sc)?;
    p.expect(&Tok::RParen)?;
    Ok(if key { Arg::Key(v) } else { Arg::Query(v) })
}

/// `table(args)` outside a condition: a per-position lookup.
fn lookup_call(p: &mut Cursor, f: &str, sc: &mut Scope) -> Result<Affine> {
    let line = p.s.line;
    p.expect(&Tok::LParen)?;
    let mut args = vec![p.expr(sc)?];
    while p.eat(&Tok::Comma) {
        args.push(p.expr(sc)?);
    }
    p.expect(&Tok::RParen)?;
    let g = &mut *sc.g;
    let ctx = sc.seq.as_deref_mut().unwrap();
    let table = ctx.tables[f].clone();
    let vars: Vec<usize> = (0..args.len()).filter(|&i| args[i].as_const().is_none()).collect();
    let eval = |vals: &[f64], ctx: &SeqCtx| table.eval(vals, &ctx.tables);
    let base: Vec<f64> = args.iter().map(|a| a.as_const().unwrap_or(0.0)).collect();
    // Values of one non-constant argument (with the others fixed) as a lookup.
    let one = |ctx: &mut SeqCtx, g: &mut Graph, i: usize, fixed: &[f64]| -> std::result::Result<Affine, String> {
        let s = ctx.side(g, &args[i], line)?;
        let mut vals = fixed.to_vec();
        let mut col = vec![];
        for c in 0..s.onehot.len() {
            vals[i] = (s.lo + c as i64) as f64;
            col.push(eval(&vals, ctx)?);
        }
        if let Some((src, off)) = ctx.direct(&args[i]) {
            // Embedding-written: values per source value t (side value t + off).
            let so = ctx.source(src).clone();
            let values = (so.lo..=so.hi).map(|t| {
                let c = t + off - s.lo;
                if c >= 0 && (c as usize) < col.len() { col[c as usize] } else { 0.0 }
            }).collect();
            return Ok(ctx.lookup(g, src, values, line));
        }
        let mut acc = Affine::default();
        for (c, v) in col.iter().enumerate() {
            if *v != 0.0 {
                acc = acc.add(&s.onehot[c].scale(*v));
            }
        }
        Ok(acc)
    };
    let r = match vars.len() {
        0 => Ok(Affine::constant(eval(&base, ctx).map_err(|m| p.err(&m))?)),
        1 => one(ctx, g, vars[0], &base),
        2 => {
            // Expand over the argument with fewer values: sum of select(onehot_c, lookup_c, 0).
            let (r0, r1) = (g.range_of(&args[vars[0]]), g.range_of(&args[vars[1]]));
            let (a, b) = if r0.1 - r0.0 <= r1.1 - r1.0 { (vars[0], vars[1]) } else { (vars[1], vars[0]) };
            (|| {
                let s = ctx.side(g, &args[a], line)?;
                let mut acc = Affine::default();
                for c in 0..s.onehot.len() {
                    let mut fixed = base.clone();
                    fixed[a] = (s.lo + c as i64) as f64;
                    let l = one(ctx, g, b, &fixed)?;
                    if l.as_const() == Some(0.0) || l.terms.is_empty() && l.c == 0.0 {
                        continue;
                    }
                    acc = acc.add(&g.mul_bit(&s.onehot[c], &l, line)?);
                }
                Ok(acc)
            })()
        }
        _ => Err(format!("a lookup takes at most two non-constant arguments (`{f}` got {})", vars.len())),
    };
    r.map_err(|m: String| p.err(&m))
}

// ---------------------------------------------------------------- reference semantics

impl SeqProgram {
    fn check_tokens(&self, tokens: &[Vec<i64>]) {
        assert!(!tokens.is_empty() && tokens.len() <= self.ctx.len, "sequence length must be 1..={}", self.ctx.len);
        for t in tokens {
            assert_eq!(t.len(), self.ctx.inputs.len(), "each position needs one value per input");
            for (v, s) in t.iter().zip(&self.ctx.inputs) {
                assert!((s.lo..=s.hi).contains(v), "`{}` = {v} out of [{}, {}]", s.name, s.lo, s.hi);
            }
        }
    }

    /// Node values at every position under exact hard attention: [node][position].
    pub fn node_values(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
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
        let src_val = |s: Src, i: usize| -> i64 {
            match s {
                Src::Pos => i as i64,
                Src::Input(k) => tokens[i][k],
            }
        };
        let mut role: HashMap<usize, (Src, Option<usize>)> = HashMap::new();
        for (k, s) in c.inputs.iter().enumerate() {
            role.insert(s.value, (Src::Input(k), None));
            for (j, &o) in s.onehot.iter().enumerate() {
                role.insert(o, (Src::Input(k), Some(j)));
            }
        }
        role.insert(c.pos.value, (Src::Pos, None));
        for (j, &o) in c.pos.onehot.iter().enumerate() {
            role.insert(o, (Src::Pos, Some(j)));
        }
        for (id, node) in self.graph.nodes.iter().enumerate() {
            let col: Vec<f64> = match node {
                Node::Relu(a) => (0..n).map(|i| at(a, &vals, i).max(0.0)).collect(),
                Node::Input(_) if let Some(&(s, oh)) = role.get(&id) => {
                    let lo = c.source(s).lo;
                    (0..n)
                        .map(|i| match oh {
                            None => src_val(s, i) as f64,
                            Some(j) => (src_val(s, i) - lo == j as i64) as i64 as f64,
                        })
                        .collect()
                }
                Node::Input(_) if let Some(&l) = c.lookup_of.get(&id) => {
                    let l = &c.lookups[l];
                    let lo = c.source(l.src).lo;
                    (0..n).map(|i| l.values[(src_val(l.src, i) - lo) as usize]).collect()
                }
                Node::Input(_) if let Some(body) = c.defs.get(&id) => (0..n).map(|i| at(body, &vals, i)).collect(),
                Node::Input(_) => {
                    let h = &c.heads[c.head_of[&id]];
                    let side = |s: &Side, j: usize, vals: &[Vec<f64>]| int(at(&s.value, vals, j));
                    let clauses: Vec<&Clause> = h.conds.iter().chain(&h.grade).collect();
                    let kv: Vec<Vec<i64>> = (0..n).map(|j| clauses.iter().map(|cl| side(&cl.k, j, &vals)).collect()).collect();
                    (0..n)
                        .map(|i| {
                            let qv: Vec<i64> = clauses.iter().map(|cl| side(&cl.q, i, &vals)).collect();
                            let mut keys: Vec<(usize, f64)> = vec![];
                            for j in (0..n).filter(|&j| !c.causal || j <= i) {
                                if h.conds.iter().enumerate().all(|(t, cl)| cl.at(qv[t], kv[j][t]) != 0.0) {
                                    let gr = h.grade.as_ref().map_or(0.0, |cl| cl.at(qv[h.conds.len()], kv[j][h.conds.len()]));
                                    keys.push((j, gr));
                                }
                            }
                            let top = keys.iter().map(|k| k.1).fold(f64::NEG_INFINITY, f64::max);
                            match &h.kind {
                                HeadKind::Count => 1.0 / (keys.len() as f64 + 1.0),
                                HeadKind::Mean(_) if keys.is_empty() => 0.0,
                                HeadKind::Mean(v) => {
                                    let win: Vec<usize> = keys.iter().filter(|k| k.1 == top).map(|k| k.0).collect();
                                    win.iter().map(|&j| at(v, &vals, j)).sum::<f64>() / win.len() as f64
                                }
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
    pub fn eval_multi(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
        let vals = self.node_values(tokens);
        (0..tokens.len())
            .map(|i| self.outputs.iter().map(|(_, a)| a.c + a.terms.iter().map(|(&m, k)| k * vals[m][i]).sum::<f64>()).collect())
            .collect()
    }

    /// `eval_multi` for a program with one input.
    pub fn eval(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        self.eval_multi(&tokens.iter().map(|&t| vec![t]).collect::<Vec<_>>())
    }

    pub fn vocab(&self) -> (i64, i64) {
        (self.ctx.inputs[0].lo, self.ctx.inputs[0].hi)
    }
}

// ---------------------------------------------------------------- weights

#[derive(Debug, Clone, PartialEq)]
pub struct AttnHead {
    /// [d_head, d_model]; scores are (Wq x_i) . (Wk x_j), no 1/sqrt(d_head).
    pub wq: Mat,
    pub wk: Mat,
    /// [values, d_model] and [d_model, values].
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

/// One token input's embedding: row 0 is BOS, row 1 + t - lo is value t. [V + 1, d_model]
#[derive(Debug, Clone, PartialEq)]
pub struct Embed {
    pub name: String,
    pub lo: i64,
    pub hi: i64,
    pub table: Mat,
}

/// A norm-free transformer: x = sum of input embeddings + pos_embed[i]; per block
/// x += sum of heads, then x += w_out relu(w_in x); outputs = unembed x.
#[derive(Debug, Clone, PartialEq)]
pub struct SeqWeights {
    pub dims: Vec<String>,
    pub len: usize,
    pub causal: bool,
    pub embeds: Vec<Embed>,
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
        let mut m = vec![&self.pos_embed, &self.unembed];
        m.extend(self.embeds.iter().map(|e| &e.table));
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

    fn embed(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
        let d = self.d_model();
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(tokens.len() + 1);
        let row = |pos: usize, t: Option<&Vec<i64>>| -> Vec<f64> {
            let mut v = self.pos_embed.row(pos).to_vec();
            for (k, e) in self.embeds.iter().enumerate() {
                let r = t.map_or(0, |t| (1 + t[k] - e.lo) as usize);
                for (a, b) in v.iter_mut().zip(e.table.row(r)) {
                    *a += b;
                }
            }
            v
        };
        x.push(row(0, None));
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.len(), self.embeds.len(), "one value per input");
            x.push(row(1 + i, Some(t)));
        }
        debug_assert!(x.iter().all(|r| r.len() == d));
        x
    }

    /// Residual stream at every position (BOS first) after the last block (dense, exact softmax).
    pub fn residual(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
        let d = self.d_model();
        let mut x = self.embed(tokens);
        let n = x.len();
        let mv = |m: &Mat, v: &[f64]| -> Vec<f64> { (0..m.rows).map(|r| m.row(r).iter().zip(v).map(|(a, b)| a * b).sum()).collect() };
        for b in &self.blocks {
            let mut add = vec![vec![0.0; d]; n];
            for h in &b.heads {
                let q: Vec<Vec<f64>> = x.iter().map(|v| mv(&h.wq, v)).collect();
                let k: Vec<Vec<f64>> = x.iter().map(|v| mv(&h.wk, v)).collect();
                let val: Vec<Vec<f64>> = x.iter().map(|v| mv(&h.wv, v)).collect();
                for i in 0..n {
                    let keys = if self.causal { i + 1 } else { n };
                    let s: Vec<f64> = (0..keys).map(|j| q[i].iter().zip(&k[j]).map(|(a, b)| a * b).sum()).collect();
                    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for o in 0..h.wv.rows {
                        let out: f64 = e.iter().zip(&val).map(|(w, v)| w * v[o]).sum::<f64>() / z;
                        for (c, a) in add[i].iter_mut().enumerate() {
                            *a += h.wo.get(c, o) * out;
                        }
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
    pub fn eval_multi(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
        let x = self.residual(tokens);
        x[1..].iter().map(|v| (0..self.unembed.rows).map(|r| self.unembed.row(r).iter().zip(v).map(|(a, b)| a * b).sum()).collect()).collect()
    }

    pub fn eval(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        self.eval_multi(&tokens.iter().map(|&t| vec![t]).collect::<Vec<_>>())
    }

    pub fn to_json(&self) -> Value {
        let m = |m: &Mat| json!(m.to_rows());
        json!({
            "format": "wir-seq-weights",
            "version": VERSION,
            "notes": "norm-free transformer; scores (Wq x_i).(Wk x_j) without 1/sqrt(d_head); no biases; position 0 is BOS (row 0 of every embedding and of pos_embed)",
            "dims": self.dims,
            "len": self.len,
            "causal": self.causal,
            "embeds": self.embeds.iter().map(|e| json!({"name": e.name, "lo": e.lo, "hi": e.hi, "table": m(&e.table)})).collect::<Vec<_>>(),
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

    /// Sparse JSON: every matrix as {rows, cols, nz: [[r, c, v], ..]}.
    pub fn to_sparse_json(&self) -> Value {
        let m = |m: &Mat| {
            let nz: Vec<Value> = (0..m.rows)
                .flat_map(|r| (0..m.cols).filter(move |&c| m.get(r, c) != 0.0).map(move |c| json!([r, c, m.get(r, c)])))
                .collect();
            json!({"rows": m.rows, "cols": m.cols, "nz": nz})
        };
        json!({
            "format": "wir-seq-weights-sparse",
            "version": VERSION,
            "dims": self.dims,
            "len": self.len,
            "causal": self.causal,
            "embeds": self.embeds.iter().map(|e| json!({"name": e.name, "lo": e.lo, "hi": e.hi, "table": m(&e.table)})).collect::<Vec<_>>(),
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

/// Keys scoring this far below the best are skipped by `SeqRunner` (weight < e^-36,
/// below f64 resolution next to the best key's weight 1).
pub const SKIP: f64 = 36.0;

type Sparse = Vec<Vec<(usize, f64)>>;

fn rows_of(m: &Mat) -> Sparse {
    (0..m.rows).map(|r| m.row(r).iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(c, v)| (c, *v)).collect()).collect()
}

fn cols_of(m: &Mat) -> Sparse {
    (0..m.cols).map(|c| (0..m.rows).filter(|&r| m.get(r, c) != 0.0).map(|r| (r, m.get(r, c))).collect()).collect()
}

/// Sparse evaluation of `SeqWeights`: the same network, with keys more than `SKIP`
/// below a query's best score left out of its softmax.
pub struct SeqRunner {
    pub w: SeqWeights,
    heads: Vec<Vec<(Sparse, Sparse, Sparse, Sparse)>>,
    mlp: Vec<(Sparse, Sparse)>,
    unembed: Sparse,
}

impl SeqRunner {
    pub fn new(w: &SeqWeights) -> Self {
        Self {
            w: w.clone(),
            heads: w.blocks.iter().map(|b| b.heads.iter().map(|h| (rows_of(&h.wq), rows_of(&h.wk), rows_of(&h.wv), cols_of(&h.wo))).collect()).collect(),
            mlp: w.blocks.iter().map(|b| (rows_of(&b.w_in), cols_of(&b.w_out))).collect(),
            unembed: rows_of(&w.unembed),
        }
    }

    pub fn residual(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
        let mut x = self.w.embed(tokens);
        let n = x.len();
        let dot = |row: &[(usize, f64)], v: &[f64]| row.iter().map(|&(c, w)| w * v[c]).sum::<f64>();
        for (bi, heads) in self.heads.iter().enumerate() {
            let mut add: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
            for (wq, wk, wv, wo) in heads {
                let q: Vec<Vec<f64>> = x.iter().map(|v| wq.iter().map(|r| dot(r, v)).collect()).collect();
                let k: Vec<Vec<(usize, f64)>> = x
                    .iter()
                    .map(|v| wk.iter().enumerate().map(|(s, r)| (s, dot(r, v))).filter(|(_, a)| *a != 0.0).collect())
                    .collect();
                let val: Vec<Vec<f64>> = x.iter().map(|v| wv.iter().map(|r| dot(r, v)).collect()).collect();
                let mut s = vec![0.0; n];
                for i in 0..n {
                    let keys = if self.w.causal { i + 1 } else { n };
                    let mut m = f64::NEG_INFINITY;
                    for j in 0..keys {
                        s[j] = k[j].iter().map(|&(c, a)| a * q[i][c]).sum();
                        m = m.max(s[j]);
                    }
                    let (mut z, mut out) = (0.0, vec![0.0; wv.len()]);
                    for j in 0..keys {
                        if s[j] < m - SKIP {
                            continue;
                        }
                        let e = (s[j] - m).exp();
                        z += e;
                        for (o, v) in val[j].iter().enumerate() {
                            out[o] += e * v;
                        }
                    }
                    for (o, col) in wo.iter().enumerate() {
                        let y = out[o] / z;
                        if y != 0.0 {
                            for &(c, w) in col {
                                add[i].push((c, w * y));
                            }
                        }
                    }
                }
            }
            let (w_in, w_out) = &self.mlp[bi];
            for i in 0..n {
                for &(c, a) in &add[i] {
                    x[i][c] += a;
                }
                let hid: Vec<f64> = w_in.iter().map(|r| dot(r, &x[i]).max(0.0)).collect();
                for (u, h) in hid.iter().enumerate() {
                    if *h != 0.0 {
                        for &(c, w) in &w_out[u] {
                            x[i][c] += w * h;
                        }
                    }
                }
            }
        }
        x
    }

    pub fn eval_multi(&self, tokens: &[Vec<i64>]) -> Vec<Vec<f64>> {
        let x = self.residual(tokens);
        x[1..].iter().map(|v| self.unembed.iter().map(|r| r.iter().map(|&(c, w)| w * v[c]).sum()).collect()).collect()
    }

    pub fn eval(&self, tokens: &[i64]) -> Vec<Vec<f64>> {
        self.eval_multi(&tokens.iter().map(|&t| vec![t]).collect::<Vec<_>>())
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
                let d = head_deps(&c.heads[h]).iter().map(|a| deps(a, &st)).max().unwrap_or(0);
                d + 1 + (d % 2 == 1) as usize // next odd
            }
            Node::Input(_) => 0,
        };
    }
    st
}

/// Every affine form a head reads.
fn head_deps(h: &Head) -> Vec<&Affine> {
    let mut v: Vec<&Affine> = h.conds.iter().chain(&h.grade).flat_map(|c| c.q.onehot.iter().chain(&c.k.onehot)).collect();
    if let HeadKind::Mean(val) = &h.kind {
        v.push(val);
    }
    v
}

pub fn compile(p: &SeqProgram) -> Result<(SeqWeights, SeqCost)> {
    let g = &p.graph;
    let c = &p.ctx;
    let st = stages(p);
    let bound = error_bound(p);
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
                for a in head_deps(&c.heads[h]) {
                    stack.extend(a.terms.keys());
                }
            }
            Node::Input(_) => {}
        }
    }
    // A relu read only by count decoders is a hidden unit only (the decoder's dimension
    // holds the sum); everything else read later gets a residual dimension.
    let mut read: HashSet<usize> = p.outputs.iter().flat_map(|(_, a)| a.terms.keys().copied()).collect();
    for (id, node) in g.nodes.iter().enumerate().filter(|&(id, _)| live[id]) {
        match node {
            Node::Relu(a) => read.extend(a.terms.keys()),
            Node::Input(_) if let Some(&h) = c.head_of.get(&id) => {
                for a in head_deps(&c.heads[h]) {
                    read.extend(a.terms.keys());
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
    for s in c.inputs.iter().chain([&c.pos]) {
        add_dim(&mut dim_of, s.value, s.name.clone(), &mut dims);
        for (k, &id) in s.onehot.iter().enumerate() {
            add_dim(&mut dim_of, id, format!("{}={}", s.name, s.lo + k as i64), &mut dims);
        }
    }
    let name_of = |id: usize| -> String {
        for (n, a) in &p.names {
            if a.terms.len() == 1 && a.terms.get(&id) == Some(&1.0) {
                return n.clone();
            }
        }
        format!("n{id}")
    };
    for (id, node) in g.nodes.iter().enumerate() {
        if !live[id] || dim_of.contains_key(&id) {
            continue;
        }
        match node {
            Node::Relu(_) if !read.contains(&id) => {}
            Node::Relu(_) => add_dim(&mut dim_of, id, format!("relu{id}"), &mut dims),
            Node::Input(_) if c.defs.contains_key(&id) => add_dim(&mut dim_of, id, format!("{}(count)", name_of(id)), &mut dims),
            Node::Input(_) if c.lookup_of.contains_key(&id) => add_dim(&mut dim_of, id, format!("{}(lookup)", name_of(id)), &mut dims),
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
    // Embeddings: each input's value and one-hot, lookups of it; `one` and `isbos` on the first.
    let mut embeds = vec![];
    for (k, s) in c.inputs.iter().enumerate() {
        let v = (s.hi - s.lo + 1) as usize;
        let mut t = Mat::zeros(v + 1, d);
        if k == 0 {
            t.set(0, isbos, 1.0);
        }
        for x in 0..v {
            if k == 0 {
                t.set(1 + x, one, 1.0);
            }
            t.set(1 + x, dim_of[&s.value], (s.lo + x as i64) as f64);
            t.set(1 + x, dim_of[&s.onehot[x]], 1.0);
        }
        embeds.push(Embed { name: s.name.clone(), lo: s.lo, hi: s.hi, table: t });
    }
    let mut pos_embed = Mat::zeros(c.len + 1, d);
    for i in 0..c.len {
        pos_embed.set(1 + i, dim_of[&c.pos.value], i as f64);
        pos_embed.set(1 + i, dim_of[&c.pos.onehot[i]], 1.0);
    }
    for l in c.lookups.iter().filter(|l| live[l.node] && dim_of.contains_key(&l.node)) {
        let dm = dim_of[&l.node];
        for (x, v) in l.values.iter().enumerate() {
            match l.src {
                Src::Pos => pos_embed.set(1 + x, dm, *v),
                Src::Input(k) => embeds[k].table.set(1 + x, dm, *v),
            }
        }
    }
    // Blocks.
    let last = (0..g.nodes.len()).filter(|&i| live[i]).map(|i| st[i]).max().unwrap_or(0);
    let nblocks = last.div_ceil(2);
    let beta = c.sharpness;
    let mut blocks = Vec::new();
    for b in 1..=nblocks {
        // Heads with equal scores share one attention head.
        let mut groups: Vec<(Vec<u64>, Vec<&Head>)> = vec![];
        for h in c.heads.iter().filter(|h| live[h.node] && st[h.node] == 2 * b - 1) {
            match groups.iter_mut().find(|(s, hs)| *s == h.sig && matches!(hs[0].kind, HeadKind::Mean(_))) {
                Some((_, hs)) if matches!(h.kind, HeadKind::Mean(_)) => hs.push(h),
                _ => groups.push((h.sig.clone(), vec![h])),
            }
        }
        let mut heads = Vec::new();
        for (_, hs) in &groups {
            let h = hs[0];
            let mu = h.m_unit();
            let clauses: Vec<(&Clause, f64)> = h.conds.iter().map(|cl| (cl, mu)).chain(h.grade.iter().map(|cl| (cl, 1.0))).collect();
            let dh = clauses.iter().map(|(cl, _)| cl.k.onehot.len()).sum::<usize>() + 1;
            let mut wq = Mat::zeros(dh, d);
            let mut wk = Mat::zeros(dh, d);
            let mut slot = 0;
            for (cl, scale) in &clauses {
                for j in 0..cl.k.onehot.len() {
                    let mut qa = Affine::default();
                    for (a, oh) in cl.q.onehot.iter().enumerate() {
                        let t = cl.table[a][j];
                        if t != 0.0 {
                            qa = qa.add(&oh.scale(t));
                        }
                    }
                    wq.row_mut(slot).copy_from_slice(&row(&qa.scale(beta * scale)));
                    wk.row_mut(slot).copy_from_slice(&row(&cl.k.onehot[j]));
                    slot += 1;
                }
            }
            // BOS slot: real queries give BOS its score; BOS's own query must read 0 to keep
            // BOS's residual zero: a mean head attends to BOS (value 0 there), a count head
            // away from it (its value is 1 only at BOS).
            let count = matches!(h.kind, HeadKind::Count);
            wq.set(slot, one, beta * h.bos_units());
            wq.set(slot, isbos, if count { -beta } else { beta });
            wk.set(slot, isbos, 1.0);
            let mut wv = Mat::zeros(hs.len(), d);
            let mut wo = Mat::zeros(d, hs.len());
            for (o, hh) in hs.iter().enumerate() {
                match &hh.kind {
                    HeadKind::Mean(val) => wv.row_mut(o).copy_from_slice(&row(val)),
                    HeadKind::Count => wv.set(o, isbos, 1.0),
                }
                wo.set(dim_of[&hh.node], o, 1.0);
            }
            heads.push(AttnHead { wq, wk, wv, wo });
        }
        let units: Vec<usize> =
            (0..g.nodes.len()).filter(|&i| live[i] && st[i] == 2 * b && matches!(g.nodes[i], Node::Relu(_))).collect();
        let unit_of: HashMap<usize, usize> = units.iter().enumerate().map(|(u, &id)| (id, u)).collect();
        let mut w_in = Mat::zeros(units.len(), d);
        let mut w_out = Mat::zeros(d, units.len());
        for (u, &id) in units.iter().enumerate() {
            let Node::Relu(a) = &g.nodes[id] else { unreachable!() };
            w_in.row_mut(u).copy_from_slice(&row(a));
            if let Some(m) = bound.gates.get(&id) {
                w_in.set(u, isbos, -m);
            }
            if let Some(&dm) = dim_of.get(&id) {
                w_out.set(dm, u, 1.0);
            }
        }
        // Count decoders: the def dim is a sum of this layer's units.
        for (&def, body) in &c.defs {
            if !live[def] || st[def] != 2 * b {
                continue;
            }
            if body.c != 0.0 {
                return Err(WirError::new("internal: count decoder with a constant"));
            }
            for (&r, &k) in &body.terms {
                let u = *unit_of.get(&r).ok_or_else(|| WirError::new("internal: count decoder unit missing"))?;
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
        embeds,
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
        "max" => vec![*tokens.iter().max().unwrap() as f64; n],
        _ => return None,
    })
}

pub const TASKS: &[&str] = &["reverse", "hist", "sort", "length", "prefix_hist", "max"];

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

/// Check output `out` against a task on every sequence the iterator gives (network run
/// with `SeqRunner`).
pub fn check(p: &SeqProgram, w: &SeqWeights, task_name: &str, out: usize, seqs: impl Iterator<Item = Vec<i64>>) -> Option<SeqCheck> {
    task(task_name, &[0])?;
    let runner = SeqRunner::new(w);
    let mut c = SeqCheck::default();
    for s in seqs {
        let want = task(task_name, &s).unwrap();
        let prog = p.eval(&s);
        let net = runner.eval(&s);
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

// ---------------------------------------------------------------- error bound

/// A sound bound on |network - program| for every sequence up to the program's length,
/// in exact arithmetic (f64/f32 rounding is not included).
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorBound {
    /// Per output, the largest |network - program| at any real position.
    pub outputs: Vec<f64>,
    /// Per head (in `ctx.heads` order): the bound on its output at real positions.
    pub heads: Vec<f64>,
    /// Per relu node that can be nonzero at BOS: the `isbos` weight -M that zeroes it there.
    pub gates: HashMap<usize, f64>,
}

/// Every residual dimension is written once, by the sublayer that computes it, and
/// the embedding, `one` and `isbos` are exact, so errors start only at heads. Relus are
/// gated at BOS (`gates`: -M isbos, M from the BOS bound), so they are exactly 0 there.
/// Track per node a bound e at real positions and eb at BOS (where the exact value is 0),
/// push them through affine forms (sum of |coef| e) and relus (1-Lipschitz), and bound
/// each head from the score perturbation its one-hot inputs allow:
///
/// - each clause with one-hot L1 errors Eq, Ek (real), Ebq, Ekb (BOS) and largest table
///   entry T times its scale s (M for a condition, 1 for the grade) moves a real key's score
///   by at most beta s T (Eq (1 + Ek) + Ek) and BOS's by beta s T (1 + Eq) Ekb; d and d_b
///   are the sums over clauses.
/// - mean: winners lead every other key by beta and BOS by beta/2 (exactly). With winners,
///   the mass off them is at most (N e^d + e^(beta/2 + d_b)) / e^(beta - d) (N e^(2d - beta)
///   + e^(d + d_b - beta/2)) and their weights differ by at most e^(2d); with none, BOS leads
///   by beta/2 and the mass off it is at most N e^(d + d_b - beta/2). The error is that mass
///   times the value span plus the value error.
/// - count: p_BOS against 1/(w + 1), bounded for every w <= wmax and every number of
///   non-matching keys up to N - w.
/// - at BOS: other keys score at most beta sum(s T Ebq (1 + Ek)); a mean head reads BOS
///   (score beta), a count head avoids it (score -beta), except a causal one, which reads 1.
pub fn error_bound(p: &SeqProgram) -> ErrorBound {
    let g = &p.graph;
    let c = &p.ctx;
    let (beta, n) = (c.sharpness, c.len as f64);
    let mut e = vec![0.0f64; g.nodes.len()];
    let mut eb = vec![0.0f64; g.nodes.len()];
    let lin = |a: &Affine, e: &[f64]| a.terms.iter().map(|(&m, k)| k.abs() * e[m]).sum::<f64>();
    let l1 = |s: &Side, e: &[f64]| s.onehot.iter().map(|a| lin(a, e)).sum::<f64>();
    let mut heads = vec![0.0; c.heads.len()];
    let mut gates = HashMap::new();
    for (id, node) in g.nodes.iter().enumerate() {
        match node {
            Node::Relu(a) => {
                // At BOS the argument is at most lin(a, eb) (its exact value is 0 there);
                // a gate of that size keeps the unit at exactly 0.
                e[id] = lin(a, &e);
                let at_bos = lin(a, &eb);
                if at_bos > 0.0 {
                    gates.insert(id, (at_bos + 1.0).ceil());
                }
            }
            Node::Input(_) if let Some(body) = c.defs.get(&id) => {
                e[id] = lin(body, &e);
                eb[id] = lin(body, &eb);
            }
            Node::Input(_) if let Some(&hi) = c.head_of.get(&id) => {
                let h = &c.heads[hi];
                let mu = h.m_unit();
                let (mut d, mut db, mut d_at_bos, mut d_self) = (0.0, 0.0, 0.0, 0.0);
                for (cl, s) in h.conds.iter().map(|cl| (cl, mu)).chain(h.grade.iter().map(|cl| (cl, 1.0))) {
                    let t = beta * s * cl.largest();
                    let (eq, ek, ebq, ekb) = (l1(&cl.q, &e), l1(&cl.k, &e), l1(&cl.q, &eb), l1(&cl.k, &eb));
                    d += t * (eq * (1.0 + ek) + ek);
                    db += t * (1.0 + eq) * ekb;
                    d_at_bos += t * ebq * (1.0 + ek);
                    d_self += t * ebq * ekb;
                }
                let (real, bos) = match &h.kind {
                    HeadKind::Mean(v) => {
                        let (lo, hi_) = g.range_of(v);
                        let ev = lin(v, &e).max(lin(v, &eb));
                        let span = hi_.max(0.0) - lo.min(0.0);
                        let maxabs = lo.abs().max(hi_.abs()) + ev;
                        let m1 = n * (2.0 * d - beta).exp() + (d + db - beta / 2.0).exp();
                        let m0 = n * (d + db - beta / 2.0).exp();
                        let real = ev + ((2.0 * d).exp_m1() * span + m1 * span).max(m0 * maxabs);
                        let mb = n * (d_at_bos + d_self - beta).exp();
                        (real, ev + mb * maxabs)
                    }
                    HeadKind::Count => {
                        let mut dev = 0.0f64;
                        for w in 0..=h.wmax {
                            let (w, r) = (w as f64, n - w as f64);
                            let exact = 1.0 / (w + 1.0);
                            let lo = (beta - db).exp() / ((beta - db).exp() + w * (beta + d).exp() + r * d.exp());
                            let hi_ = (beta + db).exp() / ((beta + db).exp() + w * (beta - d).exp());
                            dev = dev.max((lo - exact).abs()).max((hi_ - exact).abs());
                        }
                        // Causal: BOS's query sees only BOS and reads its value, 1.
                        (dev, if c.causal { 1.0 } else { (-beta + d_self + d_at_bos).exp().min(1.0) })
                    }
                };
                e[id] = real;
                eb[id] = bos;
                heads[hi] = real;
            }
            Node::Input(_) => {}
        }
    }
    ErrorBound { outputs: p.outputs.iter().map(|(_, a)| lin(a, &e)).collect(), heads, gates }
}
