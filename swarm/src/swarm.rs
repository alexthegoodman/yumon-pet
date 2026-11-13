//! Phase 1: the pipeline around the workers. A dispatcher walks a tree of open slots,
//! asks a `Filler` for each one, checks the fill with `syn`, splices it in and
//! remembers which fill wrote which bytes. Fillers are oracle (ground truth), or
//! wrappers that swap in a wrong fill for one hole, which is how a trained role is
//! tested with everything else perfect.
use crate::skeleton::{self, KINDS, Node, marker, syn_placeholder};
use anyhow::{Result, bail};
use std::collections::VecDeque;

/// What a worker is asked: one hole, in its context.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub kind: &'a str,
    pub path: &'a [usize],
    /// Whole chunk, ancestors expanded, siblings open, target as `<target_KIND>`.
    pub context: &'a str,
    /// 0 for the first try of a hole; retries are 1, 2, ... (workers sample on those).
    pub attempt: usize,
}

/// Anything that can write a hole: the oracle, a sampler, later a trained role.
pub trait Filler {
    /// The fill with its own sub-holes written as `<hole_KIND>`, or None to give up.
    fn fill(&mut self, req: &Request) -> Option<String>;
    /// Name recorded against the bytes this filler wrote.
    fn name(&self, _kind: &str) -> &str {
        "oracle"
    }
}

/// Ground truth: rebuilds the chunk the tree came from.
pub struct Oracle<'a>(pub &'a Node);

impl Oracle<'_> {
    fn node(&self, path: &[usize]) -> Option<&Node> {
        let mut n = self.0;
        for &i in path {
            n = n.children.get(i)?;
        }
        Some(n)
    }
}

impl Filler for Oracle<'_> {
    fn fill(&mut self, req: &Request) -> Option<String> {
        self.node(req.path).map(Node::skeleton)
    }
}

/// Oracle everywhere except one path, where `text` (a closed fill) is written instead.
pub struct SwapOne<'a> {
    pub truth: Oracle<'a>,
    pub path: Vec<usize>,
    pub text: String,
    pub role: String,
}

impl Filler for SwapOne<'_> {
    fn fill(&mut self, req: &Request) -> Option<String> {
        if req.path == self.path.as_slice() { Some(self.text.clone()) } else { self.truth.fill(req) }
    }
    fn name(&self, kind: &str) -> &str {
        if kind == self.role { "swap" } else { "oracle" }
    }
}

/// One slot of the working tree. `fill` is None until a worker writes it.
#[derive(Debug, Clone)]
pub struct Slot {
    pub kind: String,
    pub fill: Option<Filled>,
}

#[derive(Debug, Clone)]
pub struct Filled {
    pub segments: Vec<String>,
    pub children: Vec<Slot>,
    pub by: String,
    pub tries: usize,
}

impl Slot {
    fn open(kind: &str) -> Self {
        Self { kind: kind.into(), fill: None }
    }

    fn at(&mut self, path: &[usize]) -> &mut Slot {
        let mut s = self;
        for &i in path {
            s = &mut s.fill.as_mut().expect("path through an unfilled slot").children[i];
        }
        s
    }

    /// Worker context for `path`: ancestors expanded, siblings open, target marked.
    /// Filled siblings are shown open too, so a worker sees what it would at training time.
    fn context(&self, target: &[usize]) -> String {
        self.render_ctx(target)
    }

    fn render_ctx(&self, target: &[usize]) -> String {
        let Some(f) = &self.fill else { return marker(&self.kind) };
        let mut out = String::new();
        for (i, seg) in f.segments.iter().enumerate() {
            out.push_str(seg);
            if let Some(c) = f.children.get(i) {
                match target.split_first() {
                    Some((&p, rest)) if p == i && rest.is_empty() => out.push_str(&skeleton::target_marker(&c.kind)),
                    Some((&p, rest)) if p == i => out.push_str(&c.render_ctx(rest)),
                    _ => out.push_str(&marker(&c.kind)),
                }
            }
        }
        out
    }

    /// Final text and the byte ranges each fill wrote (outer ranges include inner ones).
    pub fn assemble(&self) -> Option<(String, Vec<Span>)> {
        let mut text = String::new();
        let mut spans = Vec::new();
        self.write(&mut text, &mut spans, &mut Vec::new())?;
        Some((text, spans))
    }

    fn write(&self, out: &mut String, spans: &mut Vec<Span>, path: &mut Vec<usize>) -> Option<()> {
        let f = self.fill.as_ref()?;
        let start = out.len();
        let slot = spans.len();
        spans.push(Span { start, end: start, kind: self.kind.clone(), by: f.by.clone(), path: path.clone() });
        for (i, seg) in f.segments.iter().enumerate() {
            out.push_str(seg);
            if let Some(c) = f.children.get(i) {
                path.push(i);
                c.write(out, spans, path)?;
                path.pop();
            }
        }
        spans[slot].end = out.len();
        Some(())
    }

    pub fn open_slots(&self) -> usize {
        match &self.fill {
            None => 1,
            Some(f) => f.children.iter().map(Slot::open_slots).sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: String,
    pub by: String,
    pub path: Vec<usize>,
}

/// Deepest fill whose bytes contain `[start, end)`: who gets charged for an error there.
pub fn charge(spans: &[Span], start: usize, end: usize) -> Option<&Span> {
    spans.iter().filter(|s| s.start <= start && end <= s.end).min_by_key(|s| s.end - s.start)
}

/// Split filled text into segments and open child slots at `<hole_KIND>` markers.
pub fn parse_fill(kind: &str, text: &str) -> Result<Filled> {
    let mut segments = Vec::new();
    let mut children = Vec::new();
    let mut at = 0;
    let mut rest = text;
    while let Some(i) = rest.find("<hole_") {
        let tail = &rest[i + 6..];
        let Some(j) = tail.find('>') else { bail!("unterminated marker in fill") };
        let k = &tail[..j];
        if !KINDS.contains(&k) {
            bail!("unknown hole kind {k}");
        }
        segments.push(text[at..at + i].to_owned());
        children.push(Slot::open(k));
        at += i + 6 + j + 1;
        rest = &text[at..];
    }
    segments.push(text[at..].to_owned());
    if text.contains("<target_") || text.contains("<fill>") {
        bail!("fill contains a target or fill marker");
    }
    let _ = kind;
    Ok(Filled { segments, children, by: String::new(), tries: 0 })
}

/// Does this text parse as a fill of `kind` (sub-holes written as `<hole_KIND>`)?
pub fn check_fill(kind: &str, text: &str) -> bool {
    parse_fill(kind, text).is_ok_and(|f| fill_ok(kind, &f))
}

/// Does this fill (sub-holes still open) pass its `syn` check?
fn fill_ok(kind: &str, f: &Filled) -> bool {
    let mut text = String::new();
    for (i, seg) in f.segments.iter().enumerate() {
        text.push_str(seg);
        if let Some(c) = f.children.get(i) {
            text.push_str(&syn_placeholder(&c.kind));
        }
    }
    skeleton::fill_parses(kind, &text)
}

#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub fills: usize,
    pub retries: usize,
    pub rejected: usize,
    pub gave_up: usize,
}

pub struct Config {
    pub max_retries: usize,
    pub max_fills: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self { max_retries: 2, max_fills: 4000 }
    }
}

/// Breadth-first dispatch of every open slot, shallow holes first.
pub fn run(filler: &mut dyn Filler, root_kind: &str, cfg: &Config) -> (Slot, RunStats) {
    let mut root = Slot::open(root_kind);
    let mut stats = RunStats::default();
    let mut queue: VecDeque<Vec<usize>> = VecDeque::from([Vec::new()]);
    while let Some(path) = queue.pop_front() {
        if stats.fills >= cfg.max_fills {
            break;
        }
        let kind = root.at(&path).kind.clone();
        let context = if path.is_empty() { marker(&kind) } else { root.context(&path) };
        let mut accepted = None;
        for attempt in 0..=cfg.max_retries {
            let req = Request { kind: &kind, path: &path, context: &context, attempt };
            let Some(text) = filler.fill(&req) else {
                stats.gave_up += 1;
                break;
            };
            match parse_fill(&kind, &text) {
                Ok(mut f) if fill_ok(&kind, &f) => {
                    f.by = filler.name(&kind).to_owned();
                    f.tries = attempt + 1;
                    accepted = Some(f);
                    break;
                }
                _ => stats.rejected += 1,
            }
            stats.retries += 1;
        }
        let Some(f) = accepted else { continue };
        stats.fills += 1;
        for i in 0..f.children.len() {
            let mut p = path.clone();
            p.push(i);
            queue.push_back(p);
        }
        root.at(&path).fill = Some(f);
    }
    (root, stats)
}

/// Run the oracle on a ground-truth tree.
pub fn run_oracle(truth: &Node) -> (Slot, RunStats) {
    run(&mut Oracle(truth), &truth.kind, &Config::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::extract;

    const SRC: &str = "/// Sums.\nfn total(rows: &[u32], gap: u32) -> u32 {\n    let mut t = 0;\n    for r in rows.iter() {\n        t += r + gap;\n    }\n    t\n}\n";

    #[test]
    fn oracle_rebuilds() {
        let truth = extract(SRC).unwrap();
        let (slot, st) = run_oracle(&truth);
        assert_eq!(slot.open_slots(), 0);
        assert_eq!(st.rejected, 0);
        assert_eq!(st.fills, truth.count());
        let (text, spans) = slot.assemble().unwrap();
        assert_eq!(text, SRC);
        assert_eq!(spans.len(), truth.count());
    }

    #[test]
    fn swap_is_charged_to_its_hole() {
        let truth = extract(SRC).unwrap();
        let mut path = None;
        truth.walk(&mut vec![], &mut |n, p| {
            if n.kind == "iter" {
                path = Some(p.to_vec());
            }
        });
        let path = path.unwrap();
        let mut f = SwapOne { truth: Oracle(&truth), path: path.clone(), text: "missing_rows".into(), role: "iter".into() };
        let (slot, _) = run(&mut f, "item", &Config::default());
        let (text, spans) = slot.assemble().unwrap();
        let at = text.find("missing_rows").unwrap();
        let s = charge(&spans, at, at + 4).unwrap();
        assert_eq!((s.path.as_slice(), s.by.as_str()), (path.as_slice(), "swap"));
    }

    #[test]
    fn bad_fills_are_rejected_then_given_up() {
        struct Bad;
        impl Filler for Bad {
            fn fill(&mut self, req: &Request) -> Option<String> {
                if req.path.is_empty() { Some("fn f() { <hole_expr> }".into()) } else { Some("let let".into()) }
            }
        }
        let (slot, st) = run(&mut Bad, "item", &Config::default());
        assert_eq!(st.fills, 1);
        assert_eq!(st.rejected, 3);
        assert_eq!(slot.open_slots(), 1);
        assert!(slot.assemble().is_none());
    }

    #[test]
    fn markers_split_into_slots() {
        let f = parse_fill("block", "{ let <hole_pat> = <hole_expr>; }").unwrap();
        assert_eq!(f.children.len(), 2);
        assert_eq!(f.segments, ["{ let ", " = ", "; }"]);
        assert!(parse_fill("block", "<hole_nope>").is_err());
    }
}
