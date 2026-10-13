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

use super::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, bpe::TokenizerKind, samples::Sample};

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
