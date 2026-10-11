//! Official Rust books from the toolchain's `rust-docs` component, prepared as a
//! raw-text stage that precedes the code curriculum. Each mdBook `print.html` is
//! converted to Markdown-like text: headings, paragraphs, lists, tables and
//! fenced code blocks that keep their attributes and rustdoc's hidden lines.
#[cfg(test)]
#[path = "../../tests/code_docs/mod.rs"]
pub(crate) mod tests;
use super::{
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    bpe::TokenizerKind,
    code_corpus::CodeConfig,
    sample_cache::{self, CacheObjective},
    samples::{Action, CardinalDir, Sample, TrainingStage, WorldContext},
};
use anyhow::{Context, Result, ensure};
use scraper::{ElementRef, Html, Node, Selector};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, path::{Path, PathBuf}};

pub const PREPARATION: &str = "rust-docs-v1";
/// Chapters without prose, such as the Reference's rule-coverage tables.
pub const SKIPPED_CHAPTERS: [&str; 1] = ["Test summary"];
pub const CACHE_FILE: &str = "rust_docs.bin";

/// Books listed on doc.rust-lang.org/stable, in the order they are prepared.
/// The unstable book is opt-in because it documents nightly-only features.
pub const DEFAULT_BOOKS: [&str; 12] = [
    "book", "rust-by-example", "reference", "nomicon", "edition-guide", "error_codes",
    "cargo", "rustdoc", "clippy", "style-guide", "embedded-book", "rustc",
];

#[derive(Debug, PartialEq)]
pub enum Block {
    Heading(usize, String),
    Text(String),
    Code(String),
}

/// `$(rustc --print sysroot)/share/doc/rust/html`, installed by `rustup component add rust-docs`.
pub fn toolchain_html() -> Result<(PathBuf, String)> {
    let run = |arg: &str| -> Result<String> {
        let out = std::process::Command::new("rustc").arg(arg).output()
            .context("running rustc to locate the rust-docs component")?;
        ensure!(out.status.success(), "rustc {arg} failed");
        Ok(String::from_utf8(out.stdout)?.trim().to_owned())
    };
    let html = PathBuf::from(run("--print=sysroot")?).join("share/doc/rust/html");
    ensure!(html.join("book/print.html").is_file(),
        "{} has no books; run `rustup component add rust-docs`", html.display());
    Ok((html, run("-V")?))
}

fn has_class(element: &ElementRef, name: &str) -> bool {
    element.value().classes().any(|class| class == name)
}

/// Navigation, rule labels, railroad diagrams and footnote markers carry no prose.
fn skipped(element: &ElementRef) -> bool {
    matches!(element.value().name(), "script" | "style" | "svg" | "nav" | "template" | "button"
        | "img" | "hr" | "wbr" | "input")
        || ["rule", "tests-popup", "grammar-railroad", "footnote-reference", "nav-wrapper", "nav-wide-wrapper"]
            .iter().any(|class| has_class(element, class))
}

fn inline(element: ElementRef, out: &mut String) {
    for child in element.children() {
        match child.value() {
            Node::Text(text) => out.push_str(text),
            Node::Element(_) => {
                let child = ElementRef::wrap(child).unwrap();
                if skipped(&child) { continue; }
                match child.value().name() {
                    "br" => out.push('\n'),
                    "code" => {
                        out.push('`');
                        out.push_str(&child.text().collect::<String>());
                        out.push('`');
                    }
                    _ => inline(child, out),
                }
            }
            _ => {}
        }
    }
}

/// Collapse HTML whitespace within each line; `<br>` line breaks are kept.
fn inline_text(element: ElementRef) -> String {
    let mut raw = String::new();
    inline(element, &mut raw);
    raw.lines().map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty()).collect::<Vec<_>>().join("\n")
}

/// `language-rust ignore edition2024` becomes the mdBook fence `rust,ignore,edition2024`.
fn fence(pre: ElementRef) -> Block {
    let code = pre.children().filter_map(ElementRef::wrap).find(|e| e.value().name() == "code");
    let info = code.and_then(|c| c.value().attr("class")).unwrap_or("")
        .split_whitespace().map(|class| class.strip_prefix("language-").unwrap_or(class))
        .filter(|class| !class.starts_with("hljs")).collect::<Vec<_>>().join(",");
    // Hidden `boring` lines are ordinary text nodes here, so examples stay complete.
    let body: String = code.map_or_else(|| pre.text().collect(), |c| c.text().collect());
    Block::Code(format!("```{info}\n{}\n```", body.trim_end_matches('\n')))
}

fn list(element: ElementRef, depth: usize, out: &mut Vec<String>) {
    let ordered = element.value().name() == "ol";
    let mut number = 0;
    for item in element.children().filter_map(ElementRef::wrap).filter(|e| e.value().name() == "li") {
        let mut text = String::new();
        let mut nested = Vec::new();
        for child in item.children() {
            match ElementRef::wrap(child) {
                Some(e) if matches!(e.value().name(), "ul" | "ol") => list(e, depth + 1, &mut nested),
                Some(e) if e.value().name() == "pre" => {
                    if let Block::Code(code) = fence(e) { nested.push(code); }
                }
                Some(e) if !skipped(&e) => {
                    let part = if e.value().name() == "code" {
                        format!("`{}`", e.text().collect::<String>())
                    } else { inline_text(e) };
                    if !part.is_empty() {
                        if !text.is_empty() { text.push(' '); }
                        text.push_str(&part);
                    }
                }
                Some(_) => {}
                None => if let Node::Text(t) = child.value() {
                    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !t.is_empty() {
                        if !text.is_empty() { text.push(' '); }
                        text.push_str(&t);
                    }
                }
            }
        }
        // A bare link item is a table of contents entry, not documentation.
        let links_only = item.children().filter_map(ElementRef::wrap)
            .all(|e| e.value().name() == "a") && item.children().any(|c| c.value().is_element());
        if text.is_empty() || (links_only && nested.is_empty()) { continue; }
        number += 1;
        let marker = if ordered { format!("{number}.") } else { "-".into() };
        out.push(format!("{}{marker} {text}", "  ".repeat(depth)));
        out.extend(nested);
    }
}

fn table(element: ElementRef) -> String {
    let row = Selector::parse("tr").unwrap();
    let mut lines = Vec::new();
    for (index, tr) in element.select(&row).enumerate() {
        let cells: Vec<_> = tr.children().filter_map(ElementRef::wrap)
            .filter(|e| matches!(e.value().name(), "td" | "th"))
            .map(|cell| inline_text(cell).replace('\n', " ")).collect();
        lines.push(format!("| {} |", cells.join(" | ")));
        if index == 0 { lines.push(format!("|{}", " --- |".repeat(cells.len()))); }
    }
    lines.join("\n")
}

fn blocks(element: ElementRef, out: &mut Vec<Block>) {
    for child in element.children() {
        let Some(child) = ElementRef::wrap(child) else {
            if let Node::Text(text) = child.value() {
                let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
                if !text.is_empty() { out.push(Block::Text(text)); }
            }
            continue;
        };
        if skipped(&child) { continue; }
        let name = child.value().name();
        let text = match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let heading = inline_text(child);
                if !heading.is_empty() { out.push(Block::Heading(name[1..].parse().unwrap(), heading)); }
                continue;
            }
            "pre" => { out.push(fence(child)); continue; }
            "ul" | "ol" => {
                let mut lines = Vec::new();
                list(child, 0, &mut lines);
                lines.join("\n")
            }
            "table" => table(child),
            "blockquote" => {
                let mut inner = Vec::new();
                blocks(child, &mut inner);
                inner.iter().map(|block| match block {
                    Block::Heading(_, t) | Block::Text(t) | Block::Code(t) => t.lines()
                        .map(|line| format!("> {line}")).collect::<Vec<_>>().join("\n"),
                }).collect::<Vec<_>>().join("\n>\n")
            }
            "dl" => child.children().filter_map(ElementRef::wrap).map(|e| match e.value().name() {
                "dt" => format!("- {}", inline_text(e)),
                _ => format!("  {}", inline_text(e)),
            }).collect::<Vec<_>>().join("\n"),
            "p" | "figcaption" | "dt" | "dd" => inline_text(child),
            "div" if has_class(&child, "file-name") => inline_text(child),
            "div" | "section" | "main" | "figure" | "span" | "article" | "aside" | "details" => {
                blocks(child, out);
                continue;
            }
            _ => inline_text(child),
        };
        if !text.trim().is_empty() { out.push(Block::Text(text)); }
    }
}

/// Convert one mdBook `print.html` into ordered blocks from its `<main>` element.
pub fn html_blocks(html: &str) -> Result<Vec<Block>> {
    let document = Html::parse_document(html);
    let main = document.select(&Selector::parse("main").unwrap()).next()
        .context("mdBook print.html has no <main> element")?;
    let mut out = Vec::new();
    blocks(main, &mut out);
    Ok(out)
}

#[derive(Default, Debug)]
pub struct BookStats {
    pub blocks: usize,
    pub chunks: usize,
    pub tokens: usize,
    pub bytes: usize,
    pub split_text_blocks: usize,
    pub skipped_code_blocks: usize,
    pub skipped_text_pieces: usize,
    pub largest_skipped: usize,
}

struct Chunker<'a> {
    tokenizer: &'a TokenizerKind,
    budget: usize,
    title: String,
    path: Vec<(usize, String)>,
    current: String,
    /// Breadcrumb that opened `current`, used as the sample label.
    label: String,
    has_body: bool,
    chunks: Vec<(String, String)>,
}

impl Chunker<'_> {
    fn tokens(&self, text: &str) -> usize { self.tokenizer.encode(text).len() }
    fn breadcrumb(&self) -> String {
        std::iter::once(self.title.as_str())
            .chain(self.path.iter().map(|(_, h)| h.as_str()).filter(|h| *h != self.title))
            .collect::<Vec<_>>().join(" > ")
    }
    fn flush(&mut self) {
        if self.has_body {
            self.chunks.push((std::mem::take(&mut self.label), std::mem::take(&mut self.current)));
        }
        self.current.clear();
        self.has_body = false;
    }
    /// Append a block, starting a new chunk (with a breadcrumb line) when it does not fit.
    fn push(&mut self, block: &str, body: bool) -> bool {
        let joined = if self.current.is_empty() { format!("{}\n\n{block}", self.breadcrumb()) }
            else { format!("{}\n\n{block}", self.current) };
        if self.tokens(&joined) <= self.budget {
            if self.current.is_empty() { self.label = self.breadcrumb(); }
            self.current = joined;
            self.has_body |= body;
            return true;
        }
        if !self.current.is_empty() {
            // A heading-only chunk is dropped; the new breadcrumb names that heading.
            self.flush();
            if !body { return true; }
            let start = format!("{}\n\n{block}", self.breadcrumb());
            if self.tokens(&start) <= self.budget {
                self.label = self.breadcrumb();
                self.current = start;
                self.has_body = body;
                return true;
            }
        }
        false
    }
}

/// Pack a book's blocks into contexts of at most `max_seq_len - 1` tokens.
/// Chapters (h1) always start a new sample; smaller sections are packed. Text
/// larger than one context is regrouped at line, then sentence boundaries; code
/// blocks are never split, so an oversized code block is reported and skipped.
pub fn chunk_book(title: &str, blocks: &[Block], tokenizer: &TokenizerKind, max_seq_len: usize)
    -> Result<(Vec<(String, String)>, BookStats)> {
    ensure!(max_seq_len >= 16, "max_seq_len is too small for documentation chunks");
    let mut stats = BookStats { blocks: blocks.len(), ..Default::default() };
    let mut chunker = Chunker { tokenizer, budget: max_seq_len - 1, title: title.into(),
        path: Vec::new(), current: String::new(), label: String::new(), has_body: false, chunks: Vec::new() };
    let mut skipping = false;
    for block in blocks {
        if let Block::Heading(1, text) = block { skipping = SKIPPED_CHAPTERS.contains(&text.as_str()); }
        if skipping { continue; }
        match block {
            Block::Heading(level, text) => {
                if *level == 1 { chunker.flush(); }
                chunker.path.retain(|(l, _)| l < level);
                chunker.path.push((*level, text.clone()));
                if !chunker.current.is_empty() || chunker.has_body {
                    chunker.push(&format!("{} {text}", "#".repeat(*level)), false);
                }
            }
            Block::Code(code) => if !chunker.push(code, true) {
                stats.skipped_code_blocks += 1;
                stats.largest_skipped = stats.largest_skipped.max(chunker.tokens(code) + 1);
            },
            Block::Text(text) => if !chunker.push(text, true) {
                stats.split_text_blocks += 1;
                // Lines (sentences of very long lines) regrouped into half-context pieces.
                let mut pieces = Vec::new();
                for line in text.lines() {
                    if chunker.tokens(line) * 2 <= chunker.budget { pieces.push(("
", line)); continue; }
                    pieces.extend(line.split_inclusive(". ").enumerate()
                        .map(|(i, sentence)| (if i == 0 { "
" } else { "" }, sentence)));
                }
                let mut group = String::new();
                for (joiner, piece) in pieces {
                    let next = if group.is_empty() { piece.to_owned() } else { format!("{group}{joiner}{piece}") };
                    if !group.is_empty() && chunker.tokens(&next) * 2 > chunker.budget {
                        if !chunker.push(group.trim_end(), true) { stats.skipped_text_pieces += 1; }
                        group = piece.to_owned();
                    } else { group = next; }
                }
                if !group.is_empty() && !chunker.push(group.trim_end(), true) { stats.skipped_text_pieces += 1; }
            },
        }
    }
    chunker.flush();
    stats.chunks = chunker.chunks.len();
    Ok((chunker.chunks, stats))
}

fn sample(text: &str, tokenizer: &TokenizerKind, max_seq_len: usize, source: String, line: String) -> Result<Sample> {
    let TokenizerKind::Bpe(tok) = tokenizer else { anyhow::bail!("Rust docs require byte-level BPE") };
    let encoding = tok.inner.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = encoding.get_ids();
    ensure!(ids.len() < max_seq_len, "chunk exceeds context");
    ensure!(ids.iter().all(|&id| id > 3), "docs contain a control or unknown token");
    ensure!(tok.inner.decode(ids, false).map_err(|e| anyhow::anyhow!("{e}"))? == text,
        "tokenizer does not preserve documentation bytes for {source}");
    let mut input_ids = vec![BOS_TOKEN];
    input_ids.extend(ids.iter().map(|&x| x as usize));
    let mut target_labels: Vec<usize> = ids.iter().map(|&x| x as usize).collect();
    target_labels.push(EOS_TOKEN);
    input_ids.resize(max_seq_len, PAD_TOKEN);
    target_labels.resize(max_seq_len, PAD_TOKEN);
    Ok(Sample { input_ids, target_labels, action: Action::Sit, motion_dir: CardinalDir::None,
        world: WorldContext::default(), target_json: String::new(), pair: (source, line) })
}

/// print.html starts with each book's first chapter, not its title.
fn book_title(id: &str) -> &str {
    match id {
        "book" => "The Rust Programming Language",
        "rust-by-example" => "Rust by Example",
        "reference" => "The Rust Reference",
        "nomicon" => "The Rustonomicon",
        "edition-guide" => "The Rust Edition Guide",
        "error_codes" => "Rust Compiler Error Index",
        "cargo" => "The Cargo Book",
        "rustdoc" => "The rustdoc Book",
        "clippy" => "The Clippy Book",
        "style-guide" => "The Rust Style Guide",
        "embedded-book" => "The Embedded Rust Book",
        "rustc" => "The rustc Book",
        "unstable-book" => "The Unstable Book",
        other => other,
    }
}

/// Build `<output>/rust_docs.bin`, `report.json` and an inspection `samples.jsonl`. The output directory
/// is staged and renamed, so a failed run never leaves a partial stage behind.
pub fn prepare(config: &CodeConfig, tokenizer: &TokenizerKind, html: &Path, toolchain: &str,
    books: &[String], output: &Path, preview: usize) -> Result<Value> {
    ensure!(!output.exists(), "{} already exists; choose a new docs directory", output.display());
    let mut samples = Vec::new();
    let mut seen = HashSet::new();
    let mut report_books = Vec::new();
    let mut texts = Vec::new();
    for id in books {
        let path = html.join(id).join("print.html");
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let blocks = html_blocks(std::str::from_utf8(&bytes)?)?;
        let title = book_title(id).to_owned();
        let (chunks, mut stats) = chunk_book(&title, &blocks, tokenizer, config.max_seq_len)?;
        let mut duplicates = 0;
        for (index, (heading, text)) in chunks.iter().enumerate() {
            if !seen.insert(Sha256::digest(text.as_bytes())) { duplicates += 1; continue; }
            let s = sample(text, tokenizer, config.max_seq_len, format!("rust-docs/{id}"),
                format!("chunk {index}: {heading}"))?;
            stats.tokens += s.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count();
            stats.bytes += text.len();
            texts.push(json!({"source": s.pair, "text": text}));
            samples.push(s);
        }
        println!("[RustDocs] {id}: {} blocks -> {} chunks ({duplicates} duplicate), {} tokens; \
            {} text blocks split ({} sentences skipped), {} oversized code blocks skipped (largest needs context {})",
            stats.blocks, stats.chunks, stats.tokens, stats.split_text_blocks, stats.skipped_text_pieces,
            stats.skipped_code_blocks, stats.largest_skipped);
        report_books.push(json!({"id": id, "title": title, "print_html_sha256": format!("{:x}", Sha256::digest(&bytes)),
            "blocks": stats.blocks, "samples": stats.chunks - duplicates, "tokens": stats.tokens, "bytes": stats.bytes,
            "split_text_blocks": stats.split_text_blocks, "skipped_text_pieces": stats.skipped_text_pieces, "skipped_code_blocks": stats.skipped_code_blocks}));
    }
    for (i, s) in samples.iter().take(preview).enumerate() {
        println!("\n=== Docs sample {} / {} | {} | {} ===\n{}", i + 1, samples.len(), s.pair.0, s.pair.1,
            tokenizer.decode(&s.target_labels));
    }
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".rust-docs-preparing-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&staging)?;
    let metadata = sample_cache::write_cache_with_objective(&staging.join(CACHE_FILE), tokenizer,
        TrainingStage::Language, config.max_seq_len, config.seed, &samples, CacheObjective::RawCode)?;
    let total = |key: &str| report_books.iter().map(|b| b[key].as_u64().unwrap()).sum::<u64>();
    let (tokens, bytes) = (total("tokens"), total("bytes"));
    println!("[RustDocs] {} samples, {tokens} tokens, {:.2} bytes/token", samples.len(), bytes as f64 / tokens as f64);
    let report = json!({"preparation": PREPARATION, "toolchain": toolchain, "books": report_books,
        "samples": samples.len(), "tokens": tokens, "bytes": bytes,
        "cache_sha256": super::code_curriculum::file_sha256(&staging.join(CACHE_FILE))?,
        "metadata": metadata});
    std::fs::write(staging.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    // For inspection only; training reads the digest-checked cache.
    let lines: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    std::fs::write(staging.join("samples.jsonl"), lines.join("
") + "
")?;
    std::fs::rename(&staging, output).context("publishing Rust docs stage")?;
    Ok(report)
}

/// Load the frozen docs stage, verifying its report digest, context and tokenizer.
pub fn load(directory: &Path, config: &CodeConfig, tokenizer: &TokenizerKind) -> Result<(Vec<Sample>, Value)> {
    let report: Value = serde_json::from_slice(&std::fs::read(directory.join("report.json"))
        .context("Rust docs stage requires report.json; prepare it with cache_samples --rust-docs")?)?;
    ensure!(report["preparation"] == PREPARATION, "Rust docs stage uses another preparation; rebuild it");
    let path = directory.join(CACHE_FILE);
    let sha = super::code_curriculum::file_sha256(&path)?;
    ensure!(report["cache_sha256"].as_str() == Some(sha.as_str()), "Rust docs cache digest mismatch");
    let samples = sample_cache::load_cache_with_objective(&path, tokenizer, TrainingStage::Language,
        config.max_seq_len, CacheObjective::RawCode)?;
    let books: Vec<_> = report["books"].as_array().into_iter().flatten()
        .map(|b| json!({"id": b["id"], "print_html_sha256": b["print_html_sha256"]})).collect();
    Ok((samples, json!({"preparation": PREPARATION, "toolchain": report["toolchain"],
        "cache_sha256": sha, "samples": report["samples"], "books": books})))
}
