//! Per-role samples cut from code chunks, plus the loader for `training-cache/code.bin`.
use crate::skeleton::{Node, extract};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Between the worker's context and its fill. Plain text the code BPE already encodes.
pub const FILL_SEP: &str = "\n// fill\n";

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub kind: String,
    pub depth: usize,
    pub path: Vec<usize>,
    /// Ancestors expanded, siblings open, target marked `hole!(TARGET kind)`.
    pub context: String,
    /// What the worker writes: the fill with its own holes still open.
    pub target: String,
}

/// One instance per non-root node.
pub fn instances(root: &Node) -> Vec<Instance> {
    let mut out = Vec::with_capacity(root.count());
    root.walk(&mut Vec::new(), &mut |node, path| {
        if path.is_empty() {
            return;
        }
        out.push(Instance {
            kind: node.kind.clone(),
            depth: path.len(),
            path: path.to_vec(),
            context: root.context(path),
            target: node.skeleton(),
        });
    });
    out
}

/// Whole files are held out, never single chunks: about a tenth of files.
pub fn is_validation(file: &str) -> bool {
    Sha256::digest(file.as_bytes())[0] % 10 == 0
}

pub fn load_chunks(cache: &Path, tokenizer: &Path, limit: Option<usize>) -> Result<Vec<Chunk>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (_, samples) = yumon_pet::brain::sample_cache::load_cache_lenient_limit(cache, 512, limit)
        .with_context(|| format!("loading {}", cache.display()))?;
    let mut chunks = Vec::with_capacity(samples.len());
    for s in samples {
        let end = s.target_labels.iter().position(|&t| t <= 3).unwrap_or(s.target_labels.len());
        let ids: Vec<u32> = s.target_labels[..end].iter().map(|&t| t as u32).collect();
        let text = tok.decode(&ids, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        chunks.push(Chunk { file: s.pair.0, text });
    }
    Ok(chunks)
}

pub fn extract_chunk(chunk: &Chunk) -> Result<Node> {
    extract(&chunk.text)
}
