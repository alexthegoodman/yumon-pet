//! Hole markers, extraction (source -> tree of skeletons and fills) and splicing.
//!
//! A `Node` is one fill. Its text is `segments[0] child0 segments[1] child1 ...`;
//! the skeleton is the same text with every child replaced by its `<hole_kind>` token text.
//! Extraction is exact: `assemble` gives back the original bytes.
use anyhow::{Context, Result, bail};
use proc_macro2::LineColumn;
use syn::spanned::Spanned;

/// Hole kinds the extractor produces. `item` is the root (a whole chunk) and never a hole.
pub const KINDS: [&str; 12] = [
    "block", "expr", "literal", "call_args", "pat", "iter", "cond", "arms", "fields", "variants",
    "sig", "name",
];
pub const MAX_DEPTH: usize = 8;

/// Open hole as models see it: one special token per kind (ids 2048.. in
/// `swarm/tokenizer.json`, built by `make_tokenizer`).
pub fn marker(kind: &str) -> String {
    format!("<hole_{kind}>")
}
/// The hole a worker must fill, also one token per kind.
pub fn target_marker(kind: &str) -> String {
    format!("<target_{kind}>")
}
/// Between a worker's context and its fill.
pub const FILL: &str = "<fill>";

/// Every swarm special token, in id order starting at the base vocab size.
pub fn special_tokens() -> Vec<String> {
    KINDS
        .iter()
        .map(|k| marker(k))
        .chain(KINDS.iter().map(|k| target_marker(k)))
        .chain(std::iter::once(FILL.to_owned()))
        .collect()
}

/// What `syn` is given where a skeleton is checked for parseability: the special
/// tokens are not Rust. Models never see these.
pub fn syn_placeholder(kind: &str) -> String {
    match kind {
        "name" => "__hole".into(),
        "sig" => "(__hole_sig: ())".into(),
        "fields" => "__hole: ()".into(),
        "variants" => "__Hole".into(),
        "arms" => "__ => (),".into(),
        other => format!("hole!({other})"),
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: String,
    pub segments: Vec<String>,
    pub children: Vec<Node>,
}

impl Node {
    /// The fill with its own holes still open (what a worker writes).
    pub fn skeleton(&self) -> String {
        self.render(&[], |c| marker(&c.kind))
    }

    /// Full text with every hole filled (what ground truth rebuilds to).
    pub fn assemble(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.segments.iter().enumerate() {
            out.push_str(seg);
            if let Some(c) = self.children.get(i) {
                out.push_str(&c.assemble());
            }
        }
        out
    }

    fn render(&self, path: &[usize], other: impl Fn(&Node) -> String + Copy) -> String {
        let mut out = String::new();
        for (i, seg) in self.segments.iter().enumerate() {
            out.push_str(seg);
            if let Some(c) = self.children.get(i) {
                match path.split_first() {
                    Some((&p, rest)) if p == i && rest.is_empty() => out.push_str(&target_marker(&c.kind)),
                    Some((&p, rest)) if p == i => out.push_str(&c.render(rest, other)),
                    _ => out.push_str(&other(c)),
                }
            }
        }
        out
    }

    /// Worker context: ancestors of the target expanded, siblings open, target marked.
    pub fn context(&self, path: &[usize]) -> String {
        self.render(path, |c| marker(&c.kind))
    }

    pub fn walk<'a>(&'a self, path: &mut Vec<usize>, f: &mut impl FnMut(&Node, &[usize])) {
        f(self, path);
        for (i, c) in self.children.iter().enumerate() {
            path.push(i);
            c.walk(path, f);
            path.pop();
        }
    }

    pub fn count(&self) -> usize {
        1 + self.children.iter().map(Node::count).sum::<usize>()
    }
}

// --- parsing wrappers ------------------------------------------------------

fn wrapper(kind: &str) -> (&'static str, &'static str) {
    // Suffixes start on a new line so a trailing `//` comment can't swallow them.
    match kind {
        "block" => ("fn __f() {", "\n}"),
        "arms" => ("fn __f() { match __x {", "\n}}"),
        "expr" | "iter" | "cond" | "literal" => ("fn __f() { let _ = ", "\n;\n}"),
        "call_args" => ("fn __f() { __g(", "\n); }"),
        "pat" => ("fn __f() { match 0 { ", "\n => (), } }"),
        "fields" => ("struct __S {", "\n}"),
        "variants" => ("enum __E {", "\n}"),
        "sig" => ("fn __f", "\n {}"),
        _ => ("", ""),
    }
}

/// Does `text` parse as a fill of this kind?
pub fn fill_parses(kind: &str, text: &str) -> bool {
    match kind {
        "item" => syn::parse_file(text).is_ok(),
        "name" => syn::parse_str::<syn::Ident>(text).is_ok(),
        _ => {
            let (pre, post) = wrapper(kind);
            let src = format!("{pre}{text}{post}");
            match kind {
                "fields" => syn::parse_str::<syn::ItemStruct>(&src).is_ok(),
                "variants" => syn::parse_str::<syn::ItemEnum>(&src).is_ok(),
                _ => syn::parse_str::<syn::ItemFn>(&src).is_ok(),
            }
        }
    }
}

/// Does the skeleton (with text markers) parse once list-position markers are
/// swapped for the syn placeholders?
pub fn skeleton_parses(node: &Node) -> bool {
    let mut text = String::new();
    for (i, seg) in node.segments.iter().enumerate() {
        text.push_str(seg);
        if let Some(c) = node.children.get(i) {
            text.push_str(&syn_placeholder(&c.kind));
        }
    }
    fill_parses(&node.kind, &text)
}

struct LineIndex<'a> {
    src: &'a str,
    starts: Vec<usize>,
}

impl<'a> LineIndex<'a> {
    fn new(src: &'a str) -> Self {
        let mut starts = vec![0];
        starts.extend(src.match_indices('\n').map(|(i, _)| i + 1));
        Self { src, starts }
    }
    fn offset(&self, lc: LineColumn) -> usize {
        let Some(&start) = self.starts.get(lc.line.saturating_sub(1)) else { return self.src.len() };
        let line = &self.src[start..];
        start + line.char_indices().nth(lc.column).map_or(line.len().min(line.find('\n').unwrap_or(line.len())), |(i, _)| i)
    }
}

struct Collector<'a> {
    idx: LineIndex<'a>,
    pre: usize,
    out: Vec<(usize, usize, &'static str)>,
}

type R = (usize, usize);

impl Collector<'_> {
    fn span<T: Spanned>(&self, t: &T) -> R {
        let s = t.span();
        (self.idx.offset(s.start()), self.idx.offset(s.end()))
    }
    fn add(&mut self, r: R, kind: &'static str) {
        if r.1 > r.0 && r.0 >= self.pre {
            self.out.push((r.0 - self.pre, r.1 - self.pre, kind));
        }
    }
    fn add_span<T: Spanned>(&mut self, t: &T, kind: &'static str) {
        let r = self.span(t);
        self.add(r, kind);
    }
    fn inner(&self, d: &proc_macro2::extra::DelimSpan) -> R {
        (self.idx.offset(d.open().end()), self.idx.offset(d.close().start()))
    }
    fn block_inner(&mut self, b: &syn::Block) {
        let r = self.inner(&b.brace_token.span);
        self.add(r, "block");
    }

    fn value(&mut self, e: &syn::Expr) {
        let lit = match e {
            syn::Expr::Lit(_) => true,
            syn::Expr::Unary(u) => matches!(*u.expr, syn::Expr::Lit(_)),
            _ => false,
        };
        self.add_span(e, if lit { "literal" } else { "expr" });
    }

    fn is_structure(e: &syn::Expr) -> bool {
        use syn::Expr::*;
        matches!(e, ForLoop(_) | While(_) | Loop(_) | If(_) | Match(_) | Block(_) | Unsafe(_))
    }

    fn structure(&mut self, e: &syn::Expr) {
        use syn::Expr::*;
        match e {
            ForLoop(f) => {
                self.add_span(&*f.pat, "pat");
                self.add_span(&*f.expr, "iter");
                self.block_inner(&f.body);
            }
            While(w) => {
                self.add_span(&*w.cond, "cond");
                self.block_inner(&w.body);
            }
            Loop(l) => self.block_inner(&l.body),
            If(i) => {
                self.add_span(&*i.cond, "cond");
                self.block_inner(&i.then_branch);
                match i.else_branch.as_ref().map(|(_, e)| &**e) {
                    Some(Block(b)) => self.block_inner(&b.block),
                    Some(other @ If(_)) => self.structure(other),
                    _ => {}
                }
            }
            Match(m) => {
                self.add_span(&*m.expr, "expr");
                let r = self.inner(&m.brace_token.span);
                self.add(r, "arms");
            }
            Block(b) => self.block_inner(&b.block),
            Unsafe(u) => self.block_inner(&u.block),
            _ => {}
        }
    }

    fn stmts(&mut self, stmts: &[syn::Stmt]) {
        for (i, s) in stmts.iter().enumerate() {
            let last = i + 1 == stmts.len();
            match s {
                syn::Stmt::Local(l) => {
                    match &l.pat {
                        syn::Pat::Type(t) => self.add_span(&*t.pat, "pat"),
                        other => self.add_span(other, "pat"),
                    }
                    if let Some(init) = &l.init {
                        self.value(&init.expr);
                    }
                }
                syn::Stmt::Expr(e, semi) => {
                    if Self::is_structure(e) {
                        self.structure(e)
                    } else if semi.is_some() || last {
                        self.value(e)
                    }
                }
                syn::Stmt::Macro(m) => {
                    if m.semi_token.is_some() || last {
                        self.add_span(&m.mac, "expr")
                    }
                }
                syn::Stmt::Item(_) => {}
            }
        }
    }

    fn arms(&mut self, arms: &[syn::Arm]) {
        for a in arms {
            self.add_span(&a.pat, "pat");
            match &*a.body {
                syn::Expr::Block(b) if b.attrs.is_empty() && b.label.is_none() => self.block_inner(&b.block),
                other => self.value(other),
            }
        }
    }

    fn expr_inner(&mut self, e: &syn::Expr) {
        match e {
            e if Self::is_structure(e) => self.structure(e),
            syn::Expr::Call(c) if !c.args.is_empty() => {
                let r = self.inner(&c.paren_token.span);
                self.add(r, "call_args");
            }
            syn::Expr::MethodCall(c) if !c.args.is_empty() => {
                let r = self.inner(&c.paren_token.span);
                self.add(r, "call_args");
            }
            _ => {}
        }
    }

    fn func(&mut self, sig: &syn::Signature, body: Option<&syn::Block>) {
        self.add_span(&sig.ident, "name");
        if !sig.inputs.is_empty() || !matches!(sig.output, syn::ReturnType::Default) {
            let open = self.idx.offset(sig.paren_token.span.open().start());
            let end = match &sig.output {
                syn::ReturnType::Type(_, ty) => self.span(&**ty).1,
                syn::ReturnType::Default => self.idx.offset(sig.paren_token.span.close().end()),
            };
            self.add((open, end), "sig");
        }
        if let Some(b) = body {
            self.block_inner(b);
        }
    }

    fn items(&mut self, items: &[syn::Item]) {
        for item in items {
            match item {
                syn::Item::Fn(f) => self.func(&f.sig, Some(&f.block)),
                syn::Item::Struct(s) => {
                    self.add_span(&s.ident, "name");
                    if let syn::Fields::Named(n) = &s.fields {
                        let r = self.inner(&n.brace_token.span);
                        self.add(r, "fields");
                    }
                }
                syn::Item::Enum(e) => {
                    self.add_span(&e.ident, "name");
                    let r = self.inner(&e.brace_token.span);
                    self.add(r, "variants");
                }
                syn::Item::Impl(i) => {
                    for m in &i.items {
                        if let syn::ImplItem::Fn(f) = m {
                            self.func(&f.sig, Some(&f.block));
                        }
                    }
                }
                syn::Item::Trait(t) => {
                    for m in &t.items {
                        if let syn::TraitItem::Fn(f) = m {
                            self.func(&f.sig, f.default.as_ref());
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Outermost holes of a fill of `kind`, as byte ranges within `text`.
fn find_holes(text: &str, kind: &str) -> Vec<(usize, usize, &'static str)> {
    let (pre, post) = wrapper(kind);
    let src = format!("{pre}{text}{post}");
    let mut c = Collector { idx: LineIndex::new(&src), pre: pre.len(), out: Vec::new() };
    match kind {
        "item" => {
            let Ok(file) = syn::parse_file(&src) else { return Vec::new() };
            c.items(&file.items);
        }
        "block" => {
            let Ok(f) = syn::parse_str::<syn::ItemFn>(&src) else { return Vec::new() };
            c.stmts(&f.block.stmts);
        }
        "arms" => {
            let Ok(f) = syn::parse_str::<syn::ItemFn>(&src) else { return Vec::new() };
            for s in &f.block.stmts {
                if let syn::Stmt::Expr(syn::Expr::Match(m), _) = s {
                    c.arms(&m.arms);
                }
            }
        }
        "expr" | "iter" | "cond" => {
            let Ok(f) = syn::parse_str::<syn::ItemFn>(&src) else { return Vec::new() };
            if let Some(syn::Stmt::Local(l)) = f.block.stmts.first() {
                if let Some(init) = &l.init {
                    c.expr_inner(&init.expr);
                }
            }
        }
        _ => {}
    }
    c.out
}

/// Trim a hole to its non-whitespace core so skeleton layout survives.
fn trimmed(text: &str, s: usize, e: usize) -> Option<(usize, usize)> {
    if e > text.len() || s >= e || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
        return None;
    }
    let body = &text[s..e];
    let lead = body.len() - body.trim_start().len();
    let trail = body.len() - body.trim_end().len();
    (lead + trail < body.len()).then_some((s + lead, e - trail))
}

pub fn decompose(text: &str, kind: &str, depth: usize) -> Node {
    let mut holes: Vec<(usize, usize, &str)> = if depth >= MAX_DEPTH {
        Vec::new()
    } else {
        find_holes(text, kind)
            .into_iter()
            .filter_map(|(s, e, k)| trimmed(text, s, e).map(|(s, e)| (s, e, k)))
            .collect()
    };
    holes.sort_by_key(|h| (h.0, std::cmp::Reverse(h.1)));
    let mut kept: Vec<(usize, usize, &str)> = Vec::new();
    for h in holes {
        if kept.last().is_none_or(|l| h.0 >= l.1) {
            kept.push(h);
        }
    }
    let mut segments = Vec::new();
    let mut children = Vec::new();
    let mut at = 0;
    for (s, e, k) in kept {
        segments.push(text[at..s].to_owned());
        children.push(decompose(&text[s..e], k, depth + 1));
        at = e;
    }
    segments.push(text[at..].to_owned());
    Node { kind: kind.to_owned(), segments, children }
}

/// Root node for one chunk, or an error if the chunk doesn't parse.
pub fn extract(source: &str) -> Result<Node> {
    if source.starts_with('\u{feff}') || source.starts_with("#!") {
        bail!("BOM or shebang");
    }
    syn::parse_file(source).context("chunk does not parse")?;
    Ok(decompose(source, "item", 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"/// Sums widths.
fn total_width(rows: &[Row], gap: u32) -> u32 {
    let mut total = 0;
    for r in rows.iter() {
        total += r.width + gap;
    }
    match total {
        0 => 1,
        n => { n.max(2) }
    }
}

struct P { x: f32, y: f32 }
enum E { A, B(u8) }
impl P { fn len(&self) -> f32 { self.x.hypot(self.y) } }
"#;

    #[test]
    fn rebuilds_byte_for_byte() {
        let n = extract(SRC).unwrap();
        assert_eq!(n.assemble(), SRC);
        assert!(n.count() > 15, "{}", n.count());
        assert!(skeleton_parses(&n));
        n.walk(&mut vec![], &mut |node, _| {
            assert!(skeleton_parses(node), "{} {:?}", node.kind, node.skeleton());
            assert!(fill_parses(&node.kind, &node.assemble()), "{} {:?}", node.kind, node.assemble());
        });
    }

    #[test]
    fn skeleton_shape() {
        let n = extract(SRC).unwrap();
        let s = n.skeleton();
        assert!(s.contains("fn <hole_name><hole_sig> {"), "{s}");
        assert!(s.contains("struct <hole_name> {<hole_fields>}") || s.contains("struct <hole_name> { <hole_fields> }"), "{s}");
        assert!(s.contains("enum <hole_name> {<hole_variants>}") || s.contains("enum <hole_name> { <hole_variants> }"), "{s}");
        let body = &n.children[2];
        assert_eq!(body.kind, "block");
        let sk = body.skeleton();
        assert!(sk.contains("let <hole_pat> = <hole_literal>;"), "{sk}");
        assert!(sk.contains("for <hole_pat> in <hole_iter> {"), "{sk}");
        assert!(sk.contains("match <hole_expr> {"), "{sk}");
    }

    #[test]
    fn special_tokens_are_unique() {
        let t = special_tokens();
        assert_eq!(t.len(), 2 * KINDS.len() + 1);
        let set: std::collections::HashSet<_> = t.iter().collect();
        assert_eq!(set.len(), t.len());
    }

    #[test]
    fn contexts_mark_the_target() {
        let n = extract(SRC).unwrap();
        let ctx = n.context(&[2]);
        assert!(ctx.contains("<target_block>"), "{ctx}");
        assert!(ctx.contains("<hole_name>"), "{ctx}");
        let deep = n.context(&[2, 2]);
        assert!(deep.contains("let <hole_pat>"), "{deep}");
    }
}
