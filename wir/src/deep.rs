//! WIR 0.2: deep ReLU programs written with combinators.
//!
//! A 0.2 program names scalar inputs (each with an optional interval), binds
//! expressions with `let`, and defines outputs. Expressions are affine arithmetic
//! (`+ -`, `*` and `/` by constants) plus combinators that compile to exact ReLU
//! constructions:
//!
//!   relu(a)  abs(a)  max(a, b, ..)  min(a, b, ..)  clip(a, lo, hi)  step(a)
//!   gt/ge/lt/le/eq(a, b)  and(..)  or(..)  not(a)  select(c, a, b)
//!
//! `step(a)` is clip(a, 0, 1): 0 at a <= 0, 1 at a >= 1, a ramp between. The
//! comparisons are steps of a - b, so they are exact whenever a - b is an integer
//! (`ge(a, b)` = step(a - b + 1)). `and`/`or`/`not` are exact on bits (0 or 1), and
//! `select(c, a, b)` (a when the bit c is 1, else b) uses the big-M product
//! c * d = relu(d - M+ (1 - c)) - relu(-d - M- (1 - c)), with M from the interval of
//! d = a - b, so its inputs need declared intervals.
//!
//! Everything lowers to a graph of affine forms over inputs and ReLU nodes. Interval
//! arithmetic over the declared input domains tracks every node's range: a ReLU whose
//! argument is never negative is the argument itself, one that is never positive is
//! zero. Those shortcuts, like the big-M constants, hold only while inputs stay in
//! their domains. Each ReLU node sits one layer above the deepest node it reads; the
//! network has one hidden layer per depth, and values that skip layers ride along on
//! carry units (relu(v - lo) + lo, one unit when v has a lower bound).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{json, Value};

use crate::error::{Result, WirError};
use crate::io::{self, DType, Format};
use crate::lexer::{split_statements, Statement, Tok, Token};
use crate::model::{dot, Mat};
use crate::num::{format_num, NumFormat};

pub const VERSION: f64 = 0.2;

pub const FUNCTIONS: &[&str] =
    &["relu", "abs", "max", "min", "clip", "step", "gt", "ge", "lt", "le", "eq", "and", "or", "not", "select"];
const RESERVED: &[&str] = &["wir", "input", "let", "output", "assert", "in"];

// ---------------------------------------------------------------- weights

/// One linear map `y = W x + b`, W in `nn.Linear` layout [out, in].
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub w: Mat,
    pub b: Vec<f64>,
}

/// A ReLU MLP: ReLU after every layer but the last. With one layer it is affine.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepWeights {
    pub layers: Vec<Layer>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl DeepWeights {
    pub fn input_dim(&self) -> usize {
        self.layers[0].w.cols
    }

    pub fn output_dim(&self) -> usize {
        self.layers.last().unwrap().w.rows
    }

    /// Units per hidden layer.
    pub fn hidden(&self) -> Vec<usize> {
        self.layers[..self.layers.len() - 1].iter().map(|l| l.w.rows).collect()
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.w.data.len() + l.b.len()).sum()
    }

    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let mut v = x.to_vec();
        for (k, l) in self.layers.iter().enumerate() {
            let last = k + 1 == self.layers.len();
            v = (0..l.w.rows)
                .map(|r| {
                    let z = dot(l.w.row(r), &v) + l.b[r];
                    if last { z } else { z.max(0.0) }
                })
                .collect();
        }
        v
    }

    pub fn check(&self) -> Result<()> {
        let first = self.layers.first().ok_or_else(|| WirError::new("network has no layers"))?;
        let mut width = first.w.cols;
        for (k, l) in self.layers.iter().enumerate() {
            if l.w.cols != width {
                return Err(WirError::new(format!("layer {k} takes {} inputs but receives {width}", l.w.cols)));
            }
            if l.b.len() != l.w.rows {
                return Err(WirError::new(format!("layer {k}: bias has {} entries for {} rows", l.b.len(), l.w.rows)));
            }
            if !l.w.data.iter().chain(&l.b).all(|x| x.is_finite()) {
                return Err(WirError::new(format!("layer {k} has NaN or infinite weights")));
            }
            width = l.w.rows;
        }
        if self.inputs.len() != first.w.cols || self.outputs.len() != width {
            return Err(WirError::new("input/output names do not match the layer shapes"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- graph

/// sum(coef * node) + c over graph nodes, zero coefficients removed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Affine {
    pub terms: BTreeMap<usize, f64>,
    pub c: f64,
}

impl Affine {
    pub fn constant(c: f64) -> Self {
        Self { terms: BTreeMap::new(), c }
    }

    pub fn node(id: usize) -> Self {
        Self { terms: BTreeMap::from([(id, 1.0)]), c: 0.0 }
    }

    pub fn as_const(&self) -> Option<f64> {
        self.terms.is_empty().then_some(self.c)
    }

    /// ka * a + kb * b
    pub fn lin(a: &Affine, ka: f64, b: &Affine, kb: f64) -> Affine {
        let mut terms = BTreeMap::new();
        for (side, k) in [(a, ka), (b, kb)] {
            for (&n, &v) in &side.terms {
                *terms.entry(n).or_insert(0.0) += k * v;
            }
        }
        terms.retain(|_, v| *v != 0.0);
        Affine { terms, c: ka * a.c + kb * b.c }
    }

    pub fn add(&self, b: &Affine) -> Affine {
        Affine::lin(self, 1.0, b, 1.0)
    }

    pub fn sub(&self, b: &Affine) -> Affine {
        Affine::lin(self, 1.0, b, -1.0)
    }

    pub fn scale(&self, k: f64) -> Affine {
        Affine::lin(self, k, &Affine::default(), 0.0)
    }

    pub fn plus(&self, c: f64) -> Affine {
        Affine { terms: self.terms.clone(), c: self.c + c }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Input(usize),
    Relu(Affine),
}

/// ReLU nodes over inputs, in creation (topological) order, with ranges and depths.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub range: Vec<(f64, f64)>,
    pub depth: Vec<usize>,
    /// Source line of the statement that first built each node.
    pub line: Vec<usize>,
    dedupe: HashMap<Vec<u64>, usize>,
}

impl Graph {
    pub fn input(&mut self, index: usize, range: (f64, f64), line: usize) -> Affine {
        self.push(Node::Input(index), range, 0, line);
        Affine::node(self.nodes.len() - 1)
    }

    fn push(&mut self, n: Node, range: (f64, f64), depth: usize, line: usize) {
        self.nodes.push(n);
        self.range.push(range);
        self.depth.push(depth);
        self.line.push(line);
    }

    /// Interval of an affine form (sound, not tight).
    pub fn range_of(&self, a: &Affine) -> (f64, f64) {
        let (mut lo, mut hi) = (a.c, a.c);
        for (&n, &k) in &a.terms {
            let (l, h) = self.range[n];
            if k > 0.0 {
                lo += k * l;
                hi += k * h;
            } else {
                lo += k * h;
                hi += k * l;
            }
        }
        (lo, hi)
    }

    /// relu(a), scaled so its largest coefficient is 1 and shared with any identical node.
    pub fn relu(&mut self, a: &Affine, line: usize) -> Affine {
        let (lo, hi) = self.range_of(a);
        if lo >= 0.0 {
            return a.clone();
        }
        if hi <= 0.0 {
            return Affine::default();
        }
        let s = a.terms.values().fold(0.0f64, |m, v| m.max(v.abs()));
        let unit = a.scale(1.0 / s);
        let key: Vec<u64> = unit.terms.iter().flat_map(|(&n, v)| [n as u64, v.to_bits()]).chain([unit.c.to_bits()]).collect();
        let id = match self.dedupe.get(&key) {
            Some(&id) => id,
            None => {
                let depth = 1 + unit.terms.keys().map(|&n| self.depth[n]).max().unwrap_or(0);
                let (l, h) = self.range_of(&unit);
                self.push(Node::Relu(unit), (l.max(0.0), h.max(0.0)), depth, line);
                let id = self.nodes.len() - 1;
                self.dedupe.insert(key, id);
                id
            }
        };
        Affine::node(id).scale(s)
    }

    /// Values of every node at an input point.
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let value = match n {
                Node::Input(i) => x[*i],
                Node::Relu(a) => eval_affine(a, &v).max(0.0),
            };
            v.push(value);
        }
        v
    }

    // ---- combinators, all exact compositions of relu and affine maps

    pub fn abs(&mut self, a: &Affine, line: usize) -> Affine {
        let p = self.relu(a, line);
        let m = self.relu(&a.scale(-1.0), line);
        p.add(&m)
    }

    /// max(a, b) = b + relu(a - b)
    pub fn max2(&mut self, a: &Affine, b: &Affine, line: usize) -> Affine {
        b.add(&self.relu(&a.sub(b), line))
    }

    /// min(a, b) = a - relu(a - b)
    pub fn min2(&mut self, a: &Affine, b: &Affine, line: usize) -> Affine {
        a.sub(&self.relu(&a.sub(b), line))
    }

    /// clip(a, lo, hi) = lo + relu(a - lo) - relu(a - hi)
    pub fn clip(&mut self, a: &Affine, lo: f64, hi: f64, line: usize) -> Affine {
        let up = self.relu(&a.plus(-lo), line);
        let down = self.relu(&a.plus(-hi), line);
        up.sub(&down).plus(lo)
    }

    pub fn step(&mut self, a: &Affine, line: usize) -> Affine {
        self.clip(a, 0.0, 1.0, line)
    }

    /// and of bits: relu(sum - (n - 1))
    pub fn and(&mut self, xs: &[Affine], line: usize) -> Affine {
        let sum = xs.iter().fold(Affine::default(), |s, x| s.add(x));
        self.relu(&sum.plus(-(xs.len() as f64 - 1.0)), line)
    }

    /// or of bits: min(sum, 1) = sum - relu(sum - 1)
    pub fn or(&mut self, xs: &[Affine], line: usize) -> Affine {
        let sum = xs.iter().fold(Affine::default(), |s, x| s.add(x));
        sum.sub(&self.relu(&sum.plus(-1.0), line))
    }

    /// c * d for a bit c and bounded d (big-M).
    pub fn mul_bit(&mut self, c: &Affine, d: &Affine, line: usize) -> std::result::Result<Affine, String> {
        let (lo, hi) = self.range_of(d);
        if !lo.is_finite() || !hi.is_finite() {
            return Err("select needs bounded values: declare `in [lo, hi]` on the inputs they depend on".into());
        }
        let (mp, mn) = (hi.max(0.0), (-lo).max(0.0));
        // relu(d + M+ c - M+) - relu(-d + M- c - M-)
        let p = self.relu(&Affine::lin(d, 1.0, c, mp).plus(-mp), line);
        let m = self.relu(&Affine::lin(d, -1.0, c, mn).plus(-mn), line);
        Ok(p.sub(&m))
    }
}

fn eval_affine(a: &Affine, values: &[f64]) -> f64 {
    a.c + a.terms.iter().map(|(&n, k)| k * values[n]).sum::<f64>()
}

// ---------------------------------------------------------------- program

#[derive(Debug, Clone, PartialEq)]
pub struct DeepAssert {
    pub output: usize,
    pub point: Vec<f64>,
    pub value: f64,
    pub line: usize,
    pub text: String,
}

/// A parsed and lowered 0.2 program.
#[derive(Debug, Clone)]
pub struct DeepProgram {
    pub inputs: Vec<(String, (f64, f64))>,
    pub outputs: Vec<(String, Affine)>,
    pub graph: Graph,
    pub asserts: Vec<DeepAssert>,
}

impl DeepProgram {
    /// The program's own semantics (graph evaluation), independent of layering.
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let v = self.graph.eval(x);
        self.outputs.iter().map(|(_, a)| eval_affine(a, &v)).collect()
    }
}

/// Units of one hidden layer attributed to one statement (or to carries).
#[derive(Debug, Clone, PartialEq)]
pub struct DeepCost {
    pub layer: usize,
    pub line: Option<usize>,
    pub units: usize,
}

#[derive(Debug, Clone)]
pub struct DeepCompiled {
    pub program: DeepProgram,
    pub weights: DeepWeights,
    pub costs: Vec<DeepCost>,
}

/// True when the first statement is a `wir 0.2` header.
pub fn is_deep_source(src: &str) -> bool {
    let Ok(stmts) = split_statements(src) else { return false };
    matches!(stmts.first().map(|s| s.tokens.as_slice()), Some([Token { tok: Tok::Ident(w), .. }, Token { tok: Tok::Num(v), .. }]) if w == "wir" && *v == VERSION)
}

pub fn compile(src: &str, tolerance: f64) -> Result<DeepCompiled> {
    let program = lower(src)?;
    let (weights, costs) = layer(&program);
    for a in &program.asserts {
        let got = weights.eval(&a.point)[a.output];
        if !((got - a.value).abs() <= tolerance) {
            return Err(WirError::at(a.line, &a.text, format!(
                "assert failed: {}{} = {} (expected {})",
                program.outputs[a.output].0,
                point_text(&a.point),
                format_num(got, NumFormat::CANONICAL),
                format_num(a.value, NumFormat::CANONICAL)
            )));
        }
    }
    Ok(DeepCompiled { program, weights, costs })
}

fn point_text(p: &[f64]) -> String {
    format!("({})", p.iter().map(|x| format_num(*x, NumFormat::CANONICAL)).collect::<Vec<_>>().join(", "))
}

/// Parse and lower to the graph.
pub fn lower(src: &str) -> Result<DeepProgram> {
    let stmts = split_statements(src)?;
    let mut prog = DeepProgram { inputs: vec![], outputs: vec![], graph: Graph::default(), asserts: vec![] };
    let mut env: HashMap<String, Affine> = HashMap::new();
    for (i, s) in stmts.iter().enumerate() {
        let mut p = Cursor { t: &s.tokens, i: 0, s };
        let head = p.ident()?;
        if i == 0 {
            if head != "wir" {
                return Err(p.err("a 0.2 program starts with `wir 0.2`"));
            }
            match p.next()? {
                Tok::Num(v) if v == VERSION => {}
                _ => return Err(p.err("only version 0.2 is supported here")),
            }
            p.end()?;
            continue;
        }
        match head.as_str() {
            "input" => {
                let mut names = vec![p.ident()?];
                while p.eat(&Tok::Comma) {
                    names.push(p.ident()?);
                }
                p.expect(&Tok::Colon)?;
                if p.ident()? != "R1" {
                    return Err(p.err("0.2 inputs are scalars: `input a, b: R1 [in [lo, hi]]`"));
                }
                let mut range = (f64::NEG_INFINITY, f64::INFINITY);
                if p.eat_ident("in") {
                    p.expect(&Tok::LBracket)?;
                    let lo = p.signed()?;
                    p.expect(&Tok::Comma)?;
                    let hi = p.signed()?;
                    p.expect(&Tok::RBracket)?;
                    if !(lo <= hi) {
                        return Err(p.err("domain needs lo <= hi"));
                    }
                    range = (lo, hi);
                }
                p.end()?;
                if !prog.outputs.is_empty() || env.len() > prog.inputs.len() {
                    return Err(p.err("declare inputs before `let` and `output`"));
                }
                for n in names {
                    p.fresh(&n, &env)?;
                    let a = prog.graph.input(prog.inputs.len(), range, s.line);
                    env.insert(n.clone(), a);
                    prog.inputs.push((n, range));
                }
            }
            "let" | "output" => {
                let name = p.ident()?;
                p.fresh(&name, &env)?;
                p.expect(&Tok::Eq)?;
                let v = p.expr(&mut prog.graph, &env)?;
                p.end()?;
                if head == "output" {
                    prog.outputs.push((name.clone(), v.clone()));
                }
                env.insert(name, v);
            }
            "assert" => loop {
                let name = p.ident()?;
                let output = prog.outputs.iter().position(|(n, _)| *n == name).ok_or_else(|| p.err(&format!("`{name}` is not an output")))?;
                p.expect(&Tok::LParen)?;
                let mut point = vec![];
                if !p.eat(&Tok::RParen) {
                    loop {
                        point.push(p.signed()?);
                        if p.eat(&Tok::RParen) {
                            break;
                        }
                        p.expect(&Tok::Comma)?;
                    }
                }
                if point.len() != prog.inputs.len() {
                    return Err(p.err(&format!("`{name}(..)` needs {} values, one per input in declaration order", prog.inputs.len())));
                }
                p.expect(&Tok::EqEq)?;
                let value = p.signed()?;
                prog.asserts.push(DeepAssert { output, point, value, line: s.line, text: s.text.clone() });
                if !p.eat(&Tok::Comma) {
                    p.end()?;
                    break;
                }
            },
            other => return Err(p.err(&format!("unknown statement `{other}` (0.2 has input, let, output, assert)"))),
        }
    }
    if stmts.is_empty() {
        return Err(WirError::new("empty program"));
    }
    if prog.inputs.is_empty() || prog.outputs.is_empty() {
        return Err(WirError::new("a 0.2 program needs at least one input and one output"));
    }
    Ok(prog)
}

struct Cursor<'a> {
    t: &'a [Token],
    i: usize,
    s: &'a Statement,
}

impl Cursor<'_> {
    fn err(&self, msg: &str) -> WirError {
        WirError::at(self.s.line, &self.s.text, msg)
    }

    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i).map(|t| &t.tok)
    }

    fn next(&mut self) -> Result<Tok> {
        let t = self.peek().cloned().ok_or_else(|| self.err("statement ends too early"))?;
        self.i += 1;
        Ok(t)
    }

    fn eat(&mut self, t: &Tok) -> bool {
        let hit = self.peek() == Some(t);
        self.i += hit as usize;
        hit
    }

    fn eat_ident(&mut self, word: &str) -> bool {
        self.eat(&Tok::Ident(word.into()))
    }

    fn expect(&mut self, t: &Tok) -> Result<()> {
        match self.peek() {
            Some(x) if x == t => {
                self.i += 1;
                Ok(())
            }
            Some(x) => Err(self.err(&format!("expected {}, found {}", t.describe(), x.describe()))),
            None => Err(self.err(&format!("expected {} at the end", t.describe()))),
        }
    }

    fn ident(&mut self) -> Result<String> {
        match self.next()? {
            Tok::Ident(s) => Ok(s),
            t => Err(self.err(&format!("expected a name, found {}", t.describe()))),
        }
    }

    fn signed(&mut self) -> Result<f64> {
        let sign = if self.eat(&Tok::Minus) { -1.0 } else { self.eat(&Tok::Plus); 1.0 };
        match self.next()? {
            Tok::Num(v) => Ok(sign * v),
            t => Err(self.err(&format!("expected a number, found {}", t.describe()))),
        }
    }

    fn end(&self) -> Result<()> {
        match self.peek() {
            None => Ok(()),
            Some(t) => Err(self.err(&format!("unexpected {}", t.describe()))),
        }
    }

    fn fresh(&self, name: &str, env: &HashMap<String, Affine>) -> Result<()> {
        if FUNCTIONS.contains(&name) || RESERVED.contains(&name) {
            return Err(self.err(&format!("`{name}` is reserved")));
        }
        if env.contains_key(name) {
            return Err(self.err(&format!("`{name}` is already defined")));
        }
        Ok(())
    }

    fn expr(&mut self, g: &mut Graph, env: &HashMap<String, Affine>) -> Result<Affine> {
        let mut acc = if self.eat(&Tok::Minus) {
            self.term(g, env)?.scale(-1.0)
        } else {
            self.eat(&Tok::Plus);
            self.term(g, env)?
        };
        loop {
            if self.eat(&Tok::Plus) {
                acc = acc.add(&self.term(g, env)?);
            } else if self.eat(&Tok::Minus) {
                acc = acc.sub(&self.term(g, env)?);
            } else {
                return Ok(acc);
            }
        }
    }

    fn term(&mut self, g: &mut Graph, env: &HashMap<String, Affine>) -> Result<Affine> {
        let mut acc = self.unary(g, env)?;
        loop {
            if self.eat(&Tok::Star) {
                let rhs = self.unary(g, env)?;
                acc = match (acc.as_const(), rhs.as_const()) {
                    (Some(k), _) => rhs.scale(k),
                    (_, Some(k)) => acc.scale(k),
                    _ => return Err(self.err("`*` needs a constant on one side (for bit * value use select(c, v, 0))")),
                };
            } else if self.eat(&Tok::Slash) {
                let rhs = self.unary(g, env)?;
                match rhs.as_const() {
                    Some(k) if k != 0.0 => acc = acc.scale(1.0 / k),
                    _ => return Err(self.err("`/` needs a nonzero constant divisor")),
                }
            } else {
                return Ok(acc);
            }
        }
    }

    fn unary(&mut self, g: &mut Graph, env: &HashMap<String, Affine>) -> Result<Affine> {
        if self.eat(&Tok::Minus) {
            return Ok(self.unary(g, env)?.scale(-1.0));
        }
        match self.next()? {
            Tok::Num(v) => Ok(Affine::constant(v)),
            Tok::LParen => {
                let v = self.expr(g, env)?;
                self.expect(&Tok::RParen)?;
                Ok(v)
            }
            Tok::Ident(name) if self.peek() == Some(&Tok::LParen) => self.call(&name, g, env),
            Tok::Ident(name) => env.get(&name).cloned().ok_or_else(|| self.err(&format!("unknown name `{name}`"))),
            t => Err(self.err(&format!("expected an expression, found {}", t.describe()))),
        }
    }

    fn call(&mut self, f: &str, g: &mut Graph, env: &HashMap<String, Affine>) -> Result<Affine> {
        self.expect(&Tok::LParen)?;
        let mut args = vec![self.expr(g, env)?];
        while self.eat(&Tok::Comma) {
            args.push(self.expr(g, env)?);
        }
        self.expect(&Tok::RParen)?;
        let line = self.s.line;
        let arity = |n: usize| -> Result<()> {
            if args.len() == n { Ok(()) } else { Err(self.err(&format!("{f} takes {n} argument(s), got {}", args.len()))) }
        };
        let at_least = |n: usize| -> Result<()> {
            if args.len() >= n { Ok(()) } else { Err(self.err(&format!("{f} takes at least {n} arguments"))) }
        };
        Ok(match f {
            "relu" => { arity(1)?; g.relu(&args[0], line) }
            "abs" => { arity(1)?; g.abs(&args[0], line) }
            "step" => { arity(1)?; g.step(&args[0], line) }
            "not" => { arity(1)?; args[0].scale(-1.0).plus(1.0) }
            "max" | "min" => {
                at_least(2)?;
                let mut acc = args[0].clone();
                for a in &args[1..] {
                    acc = if f == "max" { g.max2(&acc, a, line) } else { g.min2(&acc, a, line) };
                }
                acc
            }
            "clip" => {
                arity(3)?;
                let (Some(lo), Some(hi)) = (args[1].as_const(), args[2].as_const()) else {
                    return Err(self.err("clip bounds must be constants"));
                };
                if !(lo <= hi) {
                    return Err(self.err("clip needs lo <= hi"));
                }
                g.clip(&args[0], lo, hi, line)
            }
            "gt" | "ge" | "lt" | "le" | "eq" => {
                arity(2)?;
                let d = args[0].sub(&args[1]);
                match f {
                    "gt" => g.step(&d, line),
                    "ge" => g.step(&d.plus(1.0), line),
                    "lt" => g.step(&d.scale(-1.0), line),
                    "le" => g.step(&d.scale(-1.0).plus(1.0), line),
                    _ => {
                        let ge = g.step(&d.plus(1.0), line);
                        let le = g.step(&d.scale(-1.0).plus(1.0), line);
                        g.and(&[ge, le], line)
                    }
                }
            }
            "and" => { at_least(2)?; g.and(&args, line) }
            "or" => { at_least(2)?; g.or(&args, line) }
            "select" => {
                arity(3)?;
                let d = args[1].sub(&args[2]);
                let prod = g.mul_bit(&args[0], &d, line).map_err(|m| self.err(&m))?;
                args[2].add(&prod)
            }
            _ => return Err(self.err(&format!("unknown function `{f}` (have {})", FUNCTIONS.join(", ")))),
        })
    }
}

// ---------------------------------------------------------------- layering

/// Lay the graph out as an MLP: one hidden layer per ReLU depth, carries for values
/// read more than one layer above where they are made.
pub fn layer(p: &DeepProgram) -> (DeepWeights, Vec<DeepCost>) {
    let g = &p.graph;
    let n = g.nodes.len();
    // Reachable nodes from the outputs.
    let mut live = vec![false; n];
    let mut stack: Vec<usize> = p.outputs.iter().flat_map(|(_, a)| a.terms.keys().copied()).collect();
    while let Some(id) = stack.pop() {
        if !std::mem::replace(&mut live[id], true)
            && let Node::Relu(a) = &g.nodes[id] {
                stack.extend(a.terms.keys());
            }
    }
    let depth_max = (0..n).filter(|&i| live[i]).map(|i| g.depth[i]).max().unwrap_or(0);
    // need[i]: the last hidden layer at which node i must be present (its readers sit one above).
    let mut need = vec![0usize; n];
    for i in (0..n).filter(|&i| live[i]) {
        if let Node::Relu(a) = &g.nodes[i] {
            for &j in a.terms.keys() {
                need[j] = need[j].max(g.depth[i] - 1);
            }
        }
    }
    for (_, a) in &p.outputs {
        for &j in a.terms.keys() {
            need[j] = need[j].max(depth_max);
        }
    }
    // rep[i]: node i as affine form over the current layer's units: (unit, coef) list + const.
    let mut rep: Vec<Option<(Vec<(usize, f64)>, f64)>> = vec![None; n];
    for (i, node) in g.nodes.iter().enumerate() {
        if let Node::Input(k) = node {
            rep[i] = Some((vec![(*k, 1.0)], 0.0));
        }
    }
    let mut layers = Vec::new();
    let mut costs = Vec::new();
    let mut width = p.inputs.len();
    let row_of = |a: &Affine, rep: &[Option<(Vec<(usize, f64)>, f64)>], width: usize| -> (Vec<f64>, f64) {
        let mut w = vec![0.0; width];
        let mut b = a.c;
        for (&j, &k) in &a.terms {
            let (units, c) = rep[j].as_ref().expect("node read before it exists");
            for &(u, v) in units {
                w[u] += k * v;
            }
            b += k * c;
        }
        (w, b)
    };
    for d in 1..=depth_max {
        let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut next: Vec<Option<(Vec<(usize, f64)>, f64)>> = vec![None; n];
        let mut per_line: BTreeMap<Option<usize>, usize> = BTreeMap::new();
        for i in (0..n).filter(|&i| live[i]) {
            if g.depth[i] == d {
                let Node::Relu(a) = &g.nodes[i] else { unreachable!() };
                rows.push(row_of(a, &rep, width));
                next[i] = Some((vec![(rows.len() - 1, 1.0)], 0.0));
                *per_line.entry(Some(g.line[i])).or_default() += 1;
            } else if g.depth[i] < d && need[i] >= d {
                let v = Affine::node(i);
                let (lo, hi) = g.range[i];
                let first = rows.len();
                if lo.is_finite() {
                    rows.push(row_of(&v.plus(-lo), &rep, width));
                    next[i] = Some((vec![(first, 1.0)], lo));
                } else if hi.is_finite() {
                    rows.push(row_of(&v.scale(-1.0).plus(hi), &rep, width));
                    next[i] = Some((vec![(first, -1.0)], hi));
                } else {
                    rows.push(row_of(&v, &rep, width));
                    rows.push(row_of(&v.scale(-1.0), &rep, width));
                    next[i] = Some((vec![(first, 1.0), (first + 1, -1.0)], 0.0));
                }
                *per_line.entry(None).or_default() += rows.len() - first;
            }
        }
        costs.extend(per_line.into_iter().map(|(line, units)| DeepCost { layer: d, line, units }));
        layers.push(to_layer(rows, width));
        width = layers.last().unwrap().w.rows;
        rep = next;
    }
    let rows: Vec<(Vec<f64>, f64)> = p.outputs.iter().map(|(_, a)| row_of(a, &rep, width)).collect();
    layers.push(to_layer(rows, width));
    let w = DeepWeights {
        layers,
        inputs: p.inputs.iter().map(|(n, _)| n.clone()).collect(),
        outputs: p.outputs.iter().map(|(n, _)| n.clone()).collect(),
    };
    (w, costs)
}

fn to_layer(rows: Vec<(Vec<f64>, f64)>, cols: usize) -> Layer {
    let mut w = Mat::zeros(rows.len(), cols);
    let mut b = Vec::with_capacity(rows.len());
    for (r, (row, bias)) in rows.into_iter().enumerate() {
        w.row_mut(r).copy_from_slice(&row);
        b.push(bias);
    }
    Layer { w, b }
}

// ---------------------------------------------------------------- decompiler

/// Canonical form: every hidden unit scaled so its largest incoming weight is 1 in
/// magnitude, constant and unread units folded away, duplicates merged, and units
/// sorted within each layer. Scaling and permuting hidden units leaves it unchanged.
pub fn canonicalize(w: &DeepWeights, tol: f64) -> DeepWeights {
    let mut layers = w.layers.clone();
    for k in 0..layers.len() - 1 {
        let (head, tail) = layers.split_at_mut(k + 1);
        let (cur, nxt) = (&mut head[k], &mut tail[0]);
        let big = cur.w.data.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(cur.b.iter().fold(0.0f64, |m, x| m.max(x.abs()))).max(1.0);
        let mut keep: Vec<(Vec<f64>, f64, Vec<f64>)> = Vec::new(); // (row, bias, outgoing column)
        for r in 0..cur.w.rows {
            let row = cur.w.row(r).to_vec();
            let col: Vec<f64> = (0..nxt.w.rows).map(|o| nxt.w.get(o, r)).collect();
            let s = row.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if s <= tol * big {
                // Constant unit relu(b): fold into the next bias.
                let h = cur.b[r].max(0.0);
                for (o, c) in col.iter().enumerate() {
                    nxt.b[o] += c * h;
                }
                continue;
            }
            if col.iter().all(|c| *c == 0.0) {
                continue;
            }
            let row: Vec<f64> = row.iter().map(|x| x / s).collect();
            let col: Vec<f64> = col.iter().map(|c| c * s).collect();
            let bias = cur.b[r] / s;
            match keep.iter_mut().find(|(r2, b2, _)| (b2 - bias).abs() <= 1e-9 * bias.abs().max(1.0) && r2.iter().zip(&row).all(|(a, b)| (a - b).abs() <= 1e-9)) {
                Some((_, _, c2)) => c2.iter_mut().zip(&col).for_each(|(a, b)| *a += b),
                None => keep.push((row, bias, col)),
            }
        }
        let key = |r: &(Vec<f64>, f64, Vec<f64>)| -> Vec<i64> {
            r.0.iter().chain([&r.1]).map(|x| (x * 1e6).round() as i64).collect()
        };
        keep.sort_by(|a, b| key(b).cmp(&key(a)));
        let cols = cur.w.cols;
        let mut w2 = Mat::zeros(keep.len(), cols);
        let mut nw = Mat::zeros(nxt.w.rows, keep.len());
        for (i, (row, _, col)) in keep.iter().enumerate() {
            w2.row_mut(i).copy_from_slice(row);
            for (o, c) in col.iter().enumerate() {
                nw.set(o, i, *c);
            }
        }
        cur.b = keep.iter().map(|k| k.1).collect();
        cur.w = w2;
        nxt.w = nw;
    }
    DeepWeights { layers, inputs: w.inputs.clone(), outputs: w.outputs.clone() }
}

/// Canonical 0.2 text: one `let` per hidden unit (`h<layer>_<unit> = relu(...)`) and
/// one `output` per output.
pub fn decompile(w: &DeepWeights, fmt: NumFormat) -> Result<String> {
    w.check()?;
    let c = canonicalize(w, 1e-12);
    let mut out = String::from("wir 0.2\n");
    for name in &c.inputs {
        out.push_str(&format!("input {name}: R1\n"));
    }
    let mut prev: Vec<String> = c.inputs.clone();
    let hidden = c.layers.len() - 1;
    for (k, l) in c.layers.iter().enumerate() {
        out.push('\n');
        let names: Vec<String> = if k == hidden { c.outputs.clone() } else { (1..=l.w.rows).map(|u| format!("h{}_{u}", k + 1)).collect() };
        for (r, name) in names.iter().enumerate() {
            let body = affine_text(l.w.row(r), l.b[r], &prev, fmt);
            if k == hidden {
                out.push_str(&format!("output {name} = {body}\n"));
            } else {
                out.push_str(&format!("let {name} = relu({body})\n"));
            }
        }
        prev = names;
    }
    Ok(out)
}

fn affine_text(w: &[f64], b: f64, names: &[String], fmt: NumFormat) -> String {
    let mut s = String::new();
    for (x, name) in w.iter().zip(names) {
        if *x == 0.0 {
            continue;
        }
        let mag = number(x.abs(), fmt);
        let term = if mag == "1" { name.clone() } else { format!("{mag}*{name}") };
        if s.is_empty() {
            s = if *x < 0.0 { format!("-{term}") } else { term };
        } else {
            s.push_str(if *x < 0.0 { " - " } else { " + " });
            s.push_str(&term);
        }
    }
    if b != 0.0 || s.is_empty() {
        let mag = number(b.abs(), fmt);
        if s.is_empty() {
            s = if b < 0.0 { format!("-{mag}") } else { mag };
        } else {
            s.push_str(if b < 0.0 { " - " } else { " + " });
            s.push_str(&mag);
        }
    }
    s
}

/// Largest denominator snapped in 0.2 text. Programs on integer grids compose into
/// rationals like 35/72, which six digits cannot hold; `p/q` reads back exactly.
pub const DEEP_MAX_DENOMINATOR: i64 = 1000;

fn number(x: f64, fmt: NumFormat) -> String {
    if fmt.snap_rationals && x.is_finite() && x.abs() <= 1e6 {
        let tol = 1e-9 * x.abs().max(1.0);
        for q in 1..=DEEP_MAX_DENOMINATOR {
            let p = (x * q as f64).round();
            if (x - p / q as f64).abs() <= tol {
                return if q <= crate::num::MAX_DENOMINATOR { format_num(p / q as f64, fmt) } else { format!("{}/{q}", p as i64) };
            }
        }
    }
    format_num(x, fmt)
}

// ---------------------------------------------------------------- files

/// JSON: {"format": "wir-deep-weights", inputs, outputs, layers: [{W, b}, ...]}.
/// safetensors: nn.Sequential(Linear, ReLU, ..., Linear) keys `0.weight`, `0.bias`,
/// `2.weight`, ... with `wir.inputs` / `wir.outputs` metadata.
pub fn save(w: &DeepWeights, path: &Path, dtype: DType) -> Result<()> {
    let bytes = match Format::from_path(path)? {
        Format::Json => {
            let layers: Vec<Value> = w.layers.iter().map(|l| json!({ "W": l.w.to_rows(), "b": l.b })).collect();
            let v = json!({ "format": "wir-deep-weights", "inputs": w.inputs, "outputs": w.outputs, "layers": layers });
            serde_json::to_string_pretty(&v).unwrap().into_bytes()
        }
        Format::Safetensors => {
            let mut tensors: Vec<(String, Vec<usize>, &[f64])> = Vec::new();
            for (k, l) in w.layers.iter().enumerate() {
                tensors.push((format!("{}.weight", 2 * k), vec![l.w.rows, l.w.cols], &l.w.data));
                tensors.push((format!("{}.bias", 2 * k), vec![l.b.len()], &l.b));
            }
            let meta = json!({
                "wir.inputs": serde_json::to_string(&w.inputs).unwrap(),
                "wir.outputs": serde_json::to_string(&w.outputs).unwrap(),
            });
            io::safetensors_bytes(&tensors, meta, dtype)
        }
        Format::Npz => return Err(WirError::new("deep networks save as .json or .safetensors")),
    };
    std::fs::write(path, bytes).map_err(|e| WirError::new(format!("{}: {e}", path.display())))
}

/// Read a deep network, or `None` when the file holds a plain v0 network (one hidden layer).
pub fn load(path: &Path) -> Result<Option<DeepWeights>> {
    let bytes = std::fs::read(path).map_err(|e| WirError::new(format!("{}: {e}", path.display())))?;
    let bad = |m: String| WirError::new(format!("{}: {m}", path.display()));
    let w = match Format::from_path(path)? {
        Format::Json => {
            let v: Value = serde_json::from_slice(&bytes).map_err(|e| bad(format!("invalid JSON: {e}")))?;
            let Some(layers) = v.get("layers").and_then(|l| l.as_array()) else { return Ok(None) };
            let mut out = Vec::new();
            for (k, l) in layers.iter().enumerate() {
                let rows: Vec<Vec<f64>> = l["W"]
                    .as_array()
                    .and_then(|r| r.iter().map(|row| row.as_array()?.iter().map(|x| x.as_f64()).collect()).collect())
                    .ok_or_else(|| bad(format!("layer {k}: W must be a matrix")))?;
                let b: Vec<f64> = l["b"].as_array().and_then(|b| b.iter().map(|x| x.as_f64()).collect()).ok_or_else(|| bad(format!("layer {k}: b must be a vector")))?;
                let w = Mat::from_rows(&rows).map_err(|e| bad(e.message))?;
                out.push(Layer { w, b });
            }
            let names = |key: &str| -> Option<Vec<String>> { v.get(key)?.as_array()?.iter().map(|s| s.as_str().map(String::from)).collect() };
            named(out, names("inputs"), names("outputs"))
        }
        Format::Safetensors => {
            let file = io::read_tensors(&bytes, Format::Safetensors).map_err(|e| bad(e.message))?;
            let mut out = Vec::new();
            for k in (0..).step_by(2) {
                let (Some(w), Some(b)) = (file.tensors.get(&format!("{k}.weight")), file.tensors.get(&format!("{k}.bias"))) else { break };
                if w.shape.len() != 2 {
                    return Err(bad(format!("{k}.weight is not a matrix")));
                }
                out.push(Layer { w: Mat { rows: w.shape[0], cols: w.shape[1], data: w.data.clone() }, b: b.data.clone() });
            }
            if out.len() < 3 {
                return Ok(None);
            }
            let meta = io::safetensors_metadata(&bytes);
            let names = |key: &str| -> Option<Vec<String>> { serde_json::from_str(meta.get(key)?.as_str()?).ok() };
            named(out, names("wir.inputs"), names("wir.outputs"))
        }
        Format::Npz => return Ok(None),
    };
    w.check().map_err(|e| bad(e.message))?;
    Ok(Some(w))
}

fn named(layers: Vec<Layer>, inputs: Option<Vec<String>>, outputs: Option<Vec<String>>) -> DeepWeights {
    let d = layers.first().map_or(0, |l| l.w.cols);
    let o = layers.last().map_or(0, |l| l.w.rows);
    DeepWeights {
        inputs: inputs.filter(|v| v.len() == d).unwrap_or_else(|| (1..=d).map(|i| format!("x{i}")).collect()),
        outputs: outputs.filter(|v| v.len() == o).unwrap_or_else(|| (1..=o).map(|i| format!("y{i}")).collect()),
        layers,
    }
}
