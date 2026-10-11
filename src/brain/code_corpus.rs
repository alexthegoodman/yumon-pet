//! Lossless Rust corpus preparation. No prompts, captions, or JSON targets.
#[cfg(test)]
#[path = "../../tests/code_corpus/mod.rs"]
mod tests;
use std::{collections::HashSet, path::{Path, PathBuf}};
use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::{AddedToken, Tokenizer, models::bpe::{BPE, BpeTrainerBuilder},
    pre_tokenizers::byte_level::ByteLevel, models::TrainerWrapper};
use super::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, bpe::{BpeTokenizer, TokenizerKind},
    samples::{Action, CardinalDir, Sample, WorldContext}};

/// Missing architecture fields in older configs keep the dense decoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodeArchitecture {
    #[default]
    DecoderOnly,
    Moe,
    #[serde(rename = "xlstm")]
    XLstm,
    EncoderDecoder,
}

/// Paths are relative to the working directory, so the same config works in /app.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CodeConfig {
    pub architecture: CodeArchitecture,
    pub num_experts: usize,
    pub top_k: usize,
    pub source: PathBuf,
    pub cache: PathBuf,
    /// Optional frozen bucket directory. Only tiers 1-3 are eligible for training.
    pub curriculum: Option<PathBuf>,
    /// Saved synthetic comment journals to mix 1:1 with distinct undocumented code.
    pub documented: Option<PathBuf>,
    /// Optional frozen Rust books stage (`cache_samples --rust-docs`), trained
    /// alone in epoch 1 and kept in every later epoch.
    pub docs: Option<PathBuf>,
    /// Optional docs replay after epoch 1: target share of each later epoch's
    /// supervised tokens, as a rotating window. 0 (default) = docs pretraining only.
    pub docs_share: f64,
    pub tokenizer: String,
    pub max_seq_len: usize,
    pub exclude_dirs: Vec<String>,
    pub seed: u64,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub first_lr: f64,
    pub last_lr: f64,
    pub validation_fraction: f64,
    pub out_dir: PathBuf,
}

impl Default for CodeConfig {
    fn default() -> Self {
        Self {
            architecture: CodeArchitecture::DecoderOnly, num_experts: 4, top_k: 1,
            source: "../rust-code".into(), cache: "training-cache/code.bin".into(), curriculum: None, documented: None, docs: None, docs_share: 0.0,
            tokenizer: "yumon_code_bpe".into(), max_seq_len: 512,
            exclude_dirs: [".git", "target", "node_modules"].map(String::from).to_vec(),
            seed: 4815162342, vocab_size: 16384, embed_dim: 512, n_layers: 16,
            attn_heads: 16, ff_dim: 2048, batch_size: 8, epochs: 15,
            first_lr: 2e-4, last_lr: 2e-5, validation_fraction: 0.01,
            out_dir: "/workspace/checkpoints/yumon-code".into(),
        }
    }
}

impl CodeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.documented.is_none() || self.curriculum.is_some(),
            "documented samples require curriculum raw buckets");
        ensure!(self.num_experts > 0 && self.top_k > 0 && self.top_k <= self.num_experts,
            "require num_experts > 0 and 1 <= top_k <= num_experts");
        ensure!(self.max_seq_len >= 2, "max_seq_len must be >= 2");
        ensure!(self.docs_share >= 0.0 && self.docs_share < 1.0, "docs_share must be in [0, 1)");
        ensure!(self.vocab_size >= 260, "vocab_size must cover 256 bytes and four special tokens");
        ensure!(self.embed_dim > 0 && self.attn_heads > 0 && self.embed_dim % self.attn_heads == 0,
            "embed_dim must be positive and divisible by attn_heads");
        ensure!((self.embed_dim / self.attn_heads) % 2 == 0, "RoPE requires an even head dimension");
        ensure!(self.n_layers > 0 && self.ff_dim > 0 && self.batch_size > 0 && self.epochs > 0,
            "layers, ff_dim, batch_size and epochs must be positive");
        ensure!(self.first_lr.is_finite() && self.last_lr.is_finite() && self.first_lr > 0.0
            && self.last_lr > 0.0 && self.last_lr <= self.first_lr, "invalid learning-rate range");
        ensure!(self.validation_fraction > 0.0 && self.validation_fraction < 1.0,
            "validation_fraction must be between 0 and 1");
        Ok(())
    }
}

pub fn rust_files(config: &CodeConfig) -> Result<Vec<PathBuf>> {
    ensure!(config.source.is_dir(), "Rust source folder does not exist: {}", config.source.display());
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(&config.source).follow_links(false).into_iter()
        .filter_entry(|e| e.depth() == 0 || !e.file_type().is_dir()
            || !config.exclude_dirs.iter().any(|name| e.file_name() == name.as_str())) {
        let entry = entry.context("walking Rust source folder")?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|x| x == "rs") {
            files.push(entry.into_path());
        }
    }
    files.sort();
    ensure!(!files.is_empty(), "No .rs files found in {}", config.source.display());
    Ok(files)
}

/// Byte-level BPE, with no normalization and complete byte coverage.
pub fn train_tokenizer(files: &[PathBuf], vocab_size: usize, output: &str) -> Result<()> {
    ensure!(!Path::new(output).join("tokenizer.json").exists(), "tokenizer already exists: {output}");
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(ByteLevel::new(false, true, true));
    tokenizer.with_decoder(ByteLevel::default());
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new().vocab_size(vocab_size)
        .initial_alphabet(ByteLevel::alphabet()).min_frequency(2)
        .special_tokens(["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
            .map(|s| AddedToken::from(s, true)).to_vec()).build().into();
    let paths = files.iter().map(|p| p.to_str().map(String::from)
        .context("tokenizer source path is not UTF-8")).collect::<Result<Vec<_>>>()?;
    tokenizer.train_from_files(&mut trainer, paths).map_err(|e| anyhow::anyhow!("{e}"))?;
    BpeTokenizer { vocab_size: tokenizer.get_vocab_size(true), inner: tokenizer }.save(output)
}

pub fn load_tokenizer(path: &str) -> Result<TokenizerKind> {
    let mut tok = BpeTokenizer::load(path)?;
    ensure!(tok.inner.get_normalizer().is_none(), "Code tokenizer must have no normalizer; create one with --train-code-tokenizer");
    ensure!(tok.inner.get_truncation().is_none() && tok.inner.get_padding().is_none(),
        "Code tokenizer must not truncate or pad during encoding");
    for (id, name) in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"].iter().enumerate() {
        ensure!(tok.inner.token_to_id(name) == Some(id as u32), "invalid special token ID for {name}");
    }
    // Source string literals like "<PAD>" must be ordinary code, not control IDs.
    tok.inner.set_encode_special_tokens(true);
    Ok(TokenizerKind::Bpe(tok))
}

/// Extract complete structs and functions using Rust syntax, never brace counting.
/// Methods retain the original impl/trait header and a closing brace. Attributes
/// and doc comments belong to their item; macro bodies are not expanded.
fn rust_units(text: &str) -> Result<Vec<(String, usize)>> {
    use syn::spanned::Spanned;
    fn slice(text: &str, span: proc_macro2::Span) -> Result<&str> {
        text.get(span.byte_range()).context("Rust parser returned an invalid source span")
    }
    fn collect(text: &str, items: &[syn::Item], units: &mut Vec<(String, usize)>) -> Result<()> {
        for item in items {
            match item {
                syn::Item::Struct(_) | syn::Item::Fn(_) => {
                    units.push((slice(text, item.span())?.to_owned(), item.span().start().line));
                }
                syn::Item::Mod(module) => {
                    if let Some((_, items)) = &module.content { collect(text, items, units)?; }
                }
                syn::Item::Impl(block) => {
                    let header = text.get(item.span().byte_range().start..block.brace_token.span.open().byte_range().end)
                        .context("invalid impl header span")?;
                    for member in &block.items {
                        if let syn::ImplItem::Fn(function) = member {
                            units.push((format!("{header}\n{}\n}}", slice(text, function.span())?), function.span().start().line));
                        }
                    }
                }
                syn::Item::Trait(block) => {
                    let header = text.get(item.span().byte_range().start..block.brace_token.span.open().byte_range().end)
                        .context("invalid trait header span")?;
                    for member in &block.items {
                        if let syn::TraitItem::Fn(function) = member {
                            // Only functions with bodies, not abstract signatures.
                            if function.default.is_some() {
                                units.push((format!("{header}\n{}\n}}", slice(text, function.span())?), function.span().start().line));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
    // syn removes BOM/shebang text before assigning spans. Slice the same
    // substring so item offsets remain accurate (the shebang newline stays).
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let file = syn::parse_file(text).context("parsing Rust source (no partial-block fallback)")?;
    let text = if let Some(shebang) = &file.shebang { &text[shebang.len()..] } else { text };
    let mut units = Vec::new();
    collect(text, &file.items, &mut units)?;
    Ok(units)
}

/// One complete struct/function per sample. Oversized items are reported and
/// skipped, never truncated. Increase max_seq_len to include larger items.
pub fn chunk_source(text: &str, tokenizer: &TokenizerKind, max_seq_len: usize, source: &str) -> Result<Vec<Sample>> {
    ensure!(max_seq_len >= 2, "max_seq_len must be >= 2");
    let TokenizerKind::Bpe(tok) = tokenizer else { anyhow::bail!("Code requires byte-level BPE") };
    let units = rust_units(text).with_context(|| format!("parsing {source}"))?;
    let budget = max_seq_len - 1; // input BOS + code; targets code + EOS
    let mut samples = Vec::new();
    let mut skipped = 0;
    let mut largest = 0;
    for (unit, line) in &units {
        let encoding = tok.inner.encode(unit.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
        let code = encoding.get_ids();
        ensure!(code.iter().all(|&id| id > 3), "Code contains a control or unknown token");
        ensure!(tok.inner.decode(code, false).map_err(|e| anyhow::anyhow!("{e}"))? == *unit,
            "Tokenizer does not preserve source bytes for {source}:{line}");
        if code.len() > budget {
            skipped += 1;
            largest = largest.max(code.len() + 1);
            if skipped <= 5 {
                eprintln!("[CodeChunks] skipped {source}:{line}: complete item needs max_seq_len >= {} (configured {max_seq_len})", code.len() + 1);
            }
            continue;
        }
        let mut input_ids = vec![BOS_TOKEN];
        input_ids.extend(code.iter().map(|&x| x as usize));
        let mut target_labels: Vec<usize> = code.iter().map(|&x| x as usize).collect();
        target_labels.push(EOS_TOKEN);
        input_ids.resize(max_seq_len, PAD_TOKEN);
        target_labels.resize(max_seq_len, PAD_TOKEN);
        samples.push(Sample { input_ids, target_labels, action: Action::Sit,
            motion_dir: CardinalDir::None, world: WorldContext::default(),
            target_json: String::new(), pair: (source.into(), format!("line {line}")) });
    }
    println!("[CodeChunks] {source}: {} whole items cached, {skipped} oversized items skipped, {} eligible items; largest skipped needs context {largest}", samples.len(), units.len());
    Ok(samples)
}

pub fn prepare(config: &CodeConfig, files: &[PathBuf], tokenizer: &TokenizerKind, limit: Option<usize>) -> Result<Vec<Sample>> {
    let mut samples = Vec::new();
    let mut seen = HashSet::new();
    for path in files {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        if text.trim().is_empty() || !seen.insert(Sha256::digest(text.as_bytes())) { continue; }
        let relative = path.strip_prefix(&config.source)?.to_string_lossy().replace('\\', "/");
        samples.extend(chunk_source(&text, tokenizer, config.max_seq_len, &relative)
            .with_context(|| format!("chunking {}", path.display()))?);
    }
    samples.shuffle(&mut StdRng::seed_from_u64(config.seed));
    if let Some(limit) = limit { samples.truncate(limit); }
    ensure!(!samples.is_empty(), "No complete Rust structs/functions fit the context; increase max_seq_len or choose another corpus");
    println!("Prepared {} chunks from {} discovered Rust files (duplicate files removed)", samples.len(), files.len());
    Ok(samples)
}

/// Split by source file so neighboring chunks never leak into validation.
pub fn split_validation(samples: Vec<Sample>, config: &CodeConfig) -> Result<(Vec<Sample>, Vec<Sample>)> {
    let held_out = validation_files(&samples, config)?;
    let (validation, training) = samples.into_iter().partition(|s| held_out.contains(&s.pair.0));
    Ok((training, validation))
}

pub fn validation_files(samples: &[Sample], config: &CodeConfig) -> Result<HashSet<String>> {
    let mut files: Vec<String> = samples.iter().map(|s| s.pair.0.clone()).collect();
    files.sort(); files.dedup();
    ensure!(files.len() >= 2, "Code training needs at least two distinct nonempty files for held-out validation");
    files.shuffle(&mut StdRng::seed_from_u64(config.seed));
    let count = ((files.len() as f64 * config.validation_fraction).ceil() as usize).clamp(1, files.len() - 1);
    Ok(files[..count].iter().cloned().collect())
}

/// Prepare the shared curriculum's training text, fit it with `train_tokenizer`,
/// and rebuild matching caches, skipping oversized training records without truncation.
pub fn prepare_training_tokenizer(config: &CodeConfig, output: &Path) -> Result<CodeConfig> {
    use std::{fs, io::Write};
    use serde_json::{Value, json};
    use super::{code_curriculum::{self, ALLOWED_BUCKETS}, code_documented,
        sample_cache::{self, CacheObjective}, samples::TrainingStage};

    fn utf8(path: &Path) -> Result<&str> {
        path.to_str().context("path must be UTF-8")
    }
    fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
        fs::write(path, serde_json::to_vec_pretty(value)?).map_err(Into::into)
    }
    fn sources(samples: &[Sample], tokenizer: &TokenizerKind) -> Result<Vec<String>> {
        let TokenizerKind::Bpe(tok) = tokenizer else {
            anyhow::bail!("Code requires byte-level BPE")
        };
        samples
            .iter()
            .map(|sample| {
                let ids: Vec<_> = sample.target_labels.iter().map(|&id| id as u32).collect();
                tok.decode(&ids)
            })
            .collect()
    }

    config.validate()?;
    ensure!(
        !output.exists(),
        "output already exists: {}; choose a new directory",
        output.display()
    );
    let buckets = config
        .curriculum
        .as_ref()
        .context("Code BPE retraining requires curriculum buckets")?;
    let documented = config
        .documented
        .as_ref()
        .context("Code BPE retraining requires documented journals")?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".code-tokenizer-preparing-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir(&staging)?;
    let new_docs = staging.join("documented");
    fs::create_dir(&new_docs)?;
    fs::copy(
        documented.join("generation.json"),
        new_docs.join("generation.json"),
    )?;
    for name in ALLOWED_BUCKETS {
        let bytes = fs::read(documented.join(format!("{name}.jsonl")))?;
        let end = bytes
            .iter()
            .rposition(|&byte| byte == b'\n')
            .map_or(0, |index| index + 1);
        fs::write(new_docs.join(format!("{name}.jsonl")), &bytes[..end])?;
    }
    let old = load_tokenizer(&config.tokenizer)?;
    let mut snapshot = config.clone();
    snapshot.documented = Some(new_docs.clone());
    // The Rust docs stage is tied to a tokenizer; rebuild it for the new bundle.
    snapshot.docs = None;
    let before = code_curriculum::prepare(&snapshot, &old)?;
    // Fit once on the final cumulative TRAINING pool. Held-out files never
    // influence vocabulary merges; there is no generic prose or raw-source scan.
    let texts = sources(&before.training, &old)?;
    println!(
        "Fitting Code BPE on {} balanced training samples; {} held-out samples excluded",
        texts.len(),
        before.validation.len()
    );
    let corpus_path = staging.join("tokenizer-corpus.jsonl");
    let mut corpus = fs::File::create(&corpus_path)?;
    for (sample, text) in before.training.iter().zip(&texts) {
        serde_json::to_writer(&mut corpus, &json!({"source": sample.pair, "text": text}))?;
        corpus.write_all(b"\n")?;
    }
    corpus.sync_all()?;
    drop(corpus);
    // Feed plain source text to the existing Code BPE trainer. JSON provenance
    // above is for inspection only and never enters vocabulary fitting.
    let text_path = staging.join("tokenizer-corpus.txt");
    {
        let mut text_file = fs::File::create(&text_path)?;
        for source in &texts {
            writeln!(text_file, "{source}")?;
        }
    }
    train_tokenizer(&[text_path], config.vocab_size, utf8(&staging.join("tokenizer"))?)?;
    let new = load_tokenizer(utf8(&staging.join("tokenizer"))?)?;
    let new_buckets = staging.join("buckets");
    fs::create_dir(&new_buckets)?;
    fs::copy(
        buckets.join("reference.json"),
        new_buckets.join("reference.json"),
    )?;
    let mut report = json!({"reference_sha256": before.identity["reference_sha256"],
        "bucket_counts": {}, "bucket_caches": {},
        "classification": "Frozen original tier assignments preserved during retokenization"});
    let mut manifest: Value = serde_json::from_slice(&fs::read(new_docs.join("generation.json"))?)?;
    manifest["inputs"] = json!({});
    manifest["tokenizer_sha256"] = json!(code_curriculum::file_sha256(
        &staging.join("tokenizer/tokenizer.json")
    )?);
    let mut total = 0;
    for (tier, name) in ALLOWED_BUCKETS.iter().enumerate() {
        let input_path = buckets.join(format!("{name}.bin"));
        ensure!(
            code_curriculum::file_sha256(&input_path)?
                == before.identity["buckets"][tier]["sha256"],
            "{name} source bucket changed during tokenizer training"
        );
        let metadata = sample_cache::read_metadata(&input_path)?;
        let mut samples = sample_cache::load_cache_with_objective(
            &input_path,
            &old,
            TrainingStage::Language,
            metadata.max_seq_len,
            CacheObjective::RawCode,
        )?;
        let raw_text = sources(&samples, &old)?;
        // Keep raw bucket indices stable: journals address these exact indices.
        // Training selection below skips oversized records; storage must preserve
        // complete source text even when the new tokenizer makes it longer.
        let cache_context = raw_text.iter()
            .map(|source| new.encode(source).len() + 1)
            .max().unwrap_or(config.max_seq_len).max(config.max_seq_len);
        for (sample, source) in samples.iter_mut().zip(&raw_text) {
            code_documented::encode(sample, &new, source, cache_context)
                .with_context(|| format!("retokenizing {name}: {:?}", sample.pair))?;
        }
        let path = new_buckets.join(format!("{name}.bin"));
        let metadata = sample_cache::write_cache_with_objective(
            &path,
            &new,
            TrainingStage::Language,
            cache_context,
            config.seed,
            &samples,
            CacheObjective::RawCode,
        )?;
        let sha = code_curriculum::file_sha256(&path)?;
        total += samples.len();
        report["bucket_counts"][name] = json!(samples.len());
        report["bucket_caches"][name] = json!({"sha256": sha, "metadata": metadata});
        manifest["inputs"][name] = json!(sha);
    }
    report["total"] = json!(total);
    write_json(&new_docs.join("generation.json"), &manifest)?;
    report["generation_sha256"] = json!(code_curriculum::file_sha256(
        &new_docs.join("generation.json")
    )?);
    write_json(&new_buckets.join("report.json"), &report)?;
    let mut check = config.clone();
    check.tokenizer = utf8(&staging.join("tokenizer"))?.into();
    check.curriculum = Some(new_buckets);
    check.documented = Some(new_docs);
    check.vocab_size = new.vocab_size();
    check.docs = None;
    let after = code_curriculum::prepare(&check, &new)?;
    // Each retained source was round-trip checked during encoding. Selection
    // may change when records no longer fit under the new tokenizer.
    println!("Retokenized curriculum: {} training and {} held-out samples (previously {} and {})",
        after.training.len(), after.validation.len(), before.training.len(), before.validation.len());
    write_json(
        &staging.join("provenance.json"),
        &json!({
        "source_identity": before.identity, "output_identity": after.identity,
        "tokenizer_training_samples": texts.len(), "held_out_samples": before.validation.len(),
        "corpus_sha256": code_curriculum::file_sha256(&corpus_path)?,
        "fit": "final cumulative training split, each selected sample once; validation excluded"}),
    )?;
    let mut deployed = check.clone();
    deployed.tokenizer = "yumon_code_bpe".into();
    deployed.curriculum = Some("training-cache/code_buckets".into());
    deployed.documented = Some("training-cache/code_buckets_documented".into());
    deployed.cache = "training-cache/code.bin".into();
    deployed.out_dir = config.out_dir.join("tokenizer-retrained");
    deployed.docs = config.docs.as_ref().map(|_| "training-cache/rust_docs".into());
    write_json(&staging.join("docker-config.json"), &deployed)?;
    let mut local = deployed;
    local.tokenizer = utf8(&output.join("tokenizer"))?.into();
    local.curriculum = Some(output.join("buckets"));
    local.documented = Some(output.join("documented"));
    local.docs = config.docs.as_ref().map(|_| output.join("rust_docs"));
    write_json(&staging.join("config.json"), &local)?;
    // A partial/failed bundle never appears under the requested output name.
    fs::rename(&staging, output).context("publishing tokenizer and compatible cache bundle")?;
    Ok(local)
}
