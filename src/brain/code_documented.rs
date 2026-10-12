//! Reconstruct saved documentation without endpoint requests or journal mutation.
use anyhow::{Context, Result, ensure};
use syn::spanned::Spanned;

pub fn validate_comment(text: &str) -> Result<String> {
    let text = text.trim();
    ensure!(!text.is_empty(), "model returned an empty doc comment");
    let mut has_text = false;
    for line in text.lines() {
        let body = line
            .strip_prefix("///")
            .context("expected only /// doc-comment lines")?;
        ensure!(!body.starts_with('/'), "//// is not a Rust doc comment");
        has_text |= !body.trim().is_empty();
    }
    ensure!(has_text, "model returned an empty doc comment");
    Ok(text.to_owned())
}

/// Select the actual item, including its attributes, inside an impl/trait wrapper.
pub fn document(source: &str, comment: &str) -> Result<String> {
    let comment = validate_comment(comment)?;
    let file = syn::parse_file(source).context("parsing input Rust sample")?;
    let mut positions = Vec::new();
    for item in &file.items {
        match item {
            syn::Item::Fn(_) | syn::Item::Struct(_) => {
                positions.push(item.span().byte_range().start)
            }
            syn::Item::Impl(block) => {
                for member in &block.items {
                    if let syn::ImplItem::Fn(function) = member {
                        positions.push(function.span().byte_range().start);
                    }
                }
            }
            syn::Item::Trait(block) => {
                for member in &block.items {
                    if let syn::TraitItem::Fn(function) = member {
                        positions.push(function.span().byte_range().start);
                    }
                }
            }
            _ => {}
        }
    }
    ensure!(
        positions.len() == 1,
        "expected one function, method, or struct per sample"
    );
    // syn spans exclude an initial BOM/shebang; restore their byte offsets.
    let bom = usize::from(source.starts_with('\u{feff}')) * '\u{feff}'.len_utf8();
    let start = positions[0] + bom + file.shebang.as_ref().map_or(0, String::len);
    ensure!(source.is_char_boundary(start), "invalid Rust item span");
    let documented = format!("{}{}\n{}", &source[..start], comment, &source[start..]);
    syn::parse_file(&documented).context("doc comment produced invalid Rust")?;
    Ok(documented)
}

use super::{
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, bpe::TokenizerKind, code_corpus::CodeConfig,
    code_curriculum::ALLOWED_BUCKETS, samples::Sample,
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, path::Path};

#[derive(Deserialize)]
struct Entry {
    index: usize,
    source_sha256: String,
    doc_comment: String,
    #[serde(default)]
    skipped: bool,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn code_key(source: &str) -> Result<String> {
    // Token spelling ignores whitespace and ordinary comments, preserving literals.
    let tokens: proc_macro2::TokenStream = source
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid Rust token stream: {e}"))?;
    Ok(digest(tokens.to_string().as_bytes()))
}

fn has_docs(source: &str) -> Result<bool> {
    struct Docs(bool);
    impl<'ast> syn::visit::Visit<'ast> for Docs {
        fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
            self.0 |= attr.path().is_ident("doc");
        }
    }
    let file = syn::parse_file(source)?;
    let mut visitor = Docs(false);
    syn::visit::Visit::visit_file(&mut visitor, &file);
    Ok(visitor.0)
}

#[derive(Debug)]
pub(crate) struct OversizedSample;

impl std::fmt::Display for OversizedSample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sample exceeds configured max_seq_len")
    }
}
impl std::error::Error for OversizedSample {}

pub(crate) fn encode(
    sample: &mut Sample,
    tokenizer: &TokenizerKind,
    source: &str,
    context: usize,
) -> Result<()> {
    let TokenizerKind::Bpe(tok) = tokenizer else {
        anyhow::bail!("Code requires BPE")
    };
    let encoding = tok
        .inner
        .encode(source, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = encoding.get_ids();
    ensure!(
        ids.iter().all(|&id| id > 3) && tok.decode(ids)? == source,
        "documented source must round-trip without control/unknown tokens"
    );
    if ids.len() >= context {
        return Err(OversizedSample.into());
    }
    sample.input_ids = std::iter::once(BOS_TOKEN)
        .chain(ids.iter().map(|&id| id as usize))
        .collect();
    sample.target_labels = ids
        .iter()
        .map(|&id| id as usize)
        .chain(std::iter::once(EOS_TOKEN))
        .collect();
    sample.input_ids.resize(context, PAD_TOKEN);
    sample.target_labels.resize(context, PAD_TOKEN);
    Ok(())
}

/// Use fitting committed journal entries and choose equally many distinct raw items
/// in each tier AND partition. The held-out files are fixed before selection.
pub fn mix(
    directory: &Path,
    raw: &[Sample],
    lengths: [usize; 3],
    held_out: &HashSet<String>,
    config: &CodeConfig,
    tokenizer: &TokenizerKind,
    reference_sha: &str,
    buckets: &[Value],
) -> Result<(Vec<Sample>, [usize; 3], Value)> {
    let manifest_bytes = std::fs::read(directory.join("generation.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest["reference_sha256"].as_str() == Some(reference_sha),
        "documentation reference mismatch"
    );
    let TokenizerKind::Bpe(tok) = tokenizer else {
        anyhow::bail!("Code requires BPE")
    };
    let tokenizer_path = Path::new(&config.tokenizer).join("tokenizer.json");
    ensure!(
        manifest["tokenizer_sha256"].as_str()
            == Some(super::code_curriculum::file_sha256(&tokenizer_path)?.as_str()),
        "documentation tokenizer mismatch"
    );
    let sources = raw
        .iter()
        .map(|sample| {
            let ids: Vec<u32> = sample.target_labels.iter().map(|&id| id as u32).collect();
            tok.decode(&ids)
        })
        .collect::<Result<Vec<_>>>()?;
    let keys = sources
        .iter()
        .map(|source| code_key(source))
        .collect::<Result<Vec<_>>>()?;
    let mut excluded_keys = HashSet::new();
    let mut excluded_items = HashSet::new();
    let mut documented: [Vec<Sample>; 3] = std::array::from_fn(|_| Vec::new());
    let mut identities = Vec::new();
    let mut start = 0;
    let mut total_records = 0;
    let mut skipped_records = 0;
    for (tier, name) in ALLOWED_BUCKETS.iter().enumerate() {
        ensure!(
            manifest["inputs"][name] == buckets[tier]["sha256"],
            "{name} documentation input digest mismatch"
        );
        // Read once; never truncate a journal being written by the generator.
        let bytes = std::fs::read(directory.join(format!("{name}.jsonl")))?;
        let committed_end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        let committed = &bytes[..committed_end];
        for (index, line) in committed
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .enumerate()
        {
            let entry: Entry = serde_json::from_slice(line)
                .with_context(|| format!("{name} journal record {index}"))?;
            ensure!(
                entry.index == index && index < lengths[tier],
                "{name} journal index mismatch"
            );
            let offset = start + index;
            let source = &sources[offset];
            ensure!(
                entry.source_sha256 == digest(source.as_bytes()),
                "{name} journal source mismatch at {index}"
            );
            if entry.skipped {
                // Never documented, so it remains eligible as an undocumented sample.
                continue;
            }
            excluded_keys.insert(keys[offset].clone());
            excluded_items.insert(raw[offset].pair.clone());
            let mut sample = raw[offset].clone();
            total_records += 1;
            let result = encode(
                &mut sample,
                tokenizer,
                &document(source, &entry.doc_comment)?,
                config.max_seq_len,
            );
            if matches!(&result, Err(error) if error.is::<OversizedSample>()) {
                skipped_records += 1;
                continue;
            }
            result.with_context(|| format!("{name} documented record {index}"))?;
            documented[tier].push(sample);
        }
        identities.push(json!({"name": name, "journal_sha256": digest(committed), "documented": documented[tier].len()}));
        start += lengths[tier];
    }
    println!("[CodeDocumented] Skipped {skipped_records} of {total_records} documented records exceeding max_seq_len {}; no records truncated", config.max_seq_len);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut mixed = Vec::new();
    let mut mixed_lengths = [0; 3];
    start = 0;
    let mut raw_total = 0;
    let mut raw_skipped = 0;
    for tier in 0..3 {
        let mut candidates: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());
        let mut seen = HashSet::new();
        for offset in start..start + lengths[tier] {
            if excluded_keys.contains(&keys[offset])
                || excluded_items.contains(&raw[offset].pair)
                || has_docs(&sources[offset])?
                || !seen.insert(keys[offset].clone())
            {
                continue;
            }
            raw_total += 1;
            let mut sample = raw[offset].clone();
            match encode(&mut sample, tokenizer, &sources[offset], config.max_seq_len) {
                Ok(()) => {}
                Err(error) if error.is::<OversizedSample>() => {
                    raw_skipped += 1;
                    continue;
                }
                Err(error) => return Err(error),
            }
            candidates[usize::from(held_out.contains(&raw[offset].pair.0))].push(offset);
        }
        let mut counts = [0; 2];
        for sample in &documented[tier] {
            counts[usize::from(held_out.contains(&sample.pair.0))] += 1;
        }
        let before = mixed.len();
        mixed.append(&mut documented[tier]);
        for partition in 0..2 {
            let pool = &mut candidates[partition];
            ensure!(
                pool.len() >= counts[partition],
                "{} needs {} distinct undocumented {} samples, only {} available; supply a larger raw corpus",
                ALLOWED_BUCKETS[tier],
                counts[partition],
                if partition == 0 {
                    "training"
                } else {
                    "validation"
                },
                pool.len()
            );
            pool.shuffle(&mut rng);
            for &offset in pool.iter().take(counts[partition]) {
                let mut sample = raw[offset].clone();
                encode(&mut sample, tokenizer, &sources[offset], config.max_seq_len)?;
                mixed.push(sample);
            }
        }
        mixed_lengths[tier] = mixed.len() - before;
        identities[tier]["training_each"] = json!(counts[0]);
        identities[tier]["validation_each"] = json!(counts[1]);
        println!(
            "{}: {} documented + {} undocumented train; {} + {} validation",
            ALLOWED_BUCKETS[tier], counts[0], counts[0], counts[1], counts[1]
        );
        start += lengths[tier];
    }
    println!("[CodeDocumented] Skipped {raw_skipped} of {raw_total} eligible undocumented records exceeding max_seq_len {}", config.max_seq_len);
    Ok((
        mixed,
        mixed_lengths,
        json!({"manifest_sha256": digest(&manifest_bytes), "tiers": identities,
        "selection": "seeded-distinct-undocumented-per-tier-and-partition-v1", "context": config.max_seq_len}),
    ))
}
