//! Model-independent cumulative curriculum, restricted to low/mild/moderate.
use super::{
    PAD_TOKEN,
    bpe::TokenizerKind,
    code_corpus::{self, CodeConfig},
    sample_cache::{self, CacheObjective},
    samples::{Sample, TrainingStage},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::File, path::Path};

pub const ALLOWED_BUCKETS: [&str; 3] = ["tier_1_low", "tier_2_mild", "tier_3_moderate"];

pub struct PreparedCode {
    pub training: Vec<Sample>,
    pub validation: Vec<Sample>,
    /// Cumulative lengths after the single global file-level validation split.
    /// With a docs stage these are offsets into `training`, after the docs.
    pub tier_ends: Option<[usize; 3]>,
    /// Rust books prepended to `training`: epoch 1 is docs only, then code epochs
    /// follow with the docs kept in every slice. Docs never enter validation.
    pub docs_len: usize,
    pub docs_share: f64,
    pub identity: Value,
    pub seed: u64,
}
impl PreparedCode {
    fn code_end(&self, code_epoch: usize) -> usize {
        self.tier_ends.map_or(self.training.len(), |ends| ends[code_epoch.min(2)])
    }
    fn tokens(&self, index: usize) -> usize {
        self.training[index].target_labels.iter().filter(|&&t| t != PAD_TOKEN).count()
    }
    /// Rotating docs window for a later epoch, sized so docs take about
    /// `docs_share` of that epoch's supervised tokens. Windows continue where
    /// the previous epoch stopped, so replay cycles through every docs sample.
    fn docs_window(&self, code_epoch: usize) -> Vec<usize> {
        let mut start = 0;
        let mut window = Vec::new();
        for e in 0..=code_epoch {
            let code: usize = (self.docs_len..self.code_end(e)).map(|i| self.tokens(i)).sum();
            let target = code as f64 * self.docs_share / (1.0 - self.docs_share);
            let mut taken = 0;
            window.clear();
            while (taken as f64) < target && window.len() < self.docs_len {
                let index = (start + window.len()) % self.docs_len;
                taken += self.tokens(index);
                window.push(index);
            }
            start = (start + window.len()) % self.docs_len;
        }
        window
    }
    /// Training indices for an epoch. With docs, epoch 1 is the whole docs stage;
    /// later epochs add a rotating docs window to the cumulative code tiers.
    pub fn epoch_indices(&self, epoch: usize) -> Vec<usize> {
        match (self.docs_len, epoch) {
            (0, epoch) => (0..self.code_end(epoch)).collect(),
            (docs, 0) => (0..docs).collect(),
            (_, epoch) => {
                let mut indices = self.docs_window(epoch - 1);
                indices.extend(self.docs_len..self.code_end(epoch - 1));
                indices
            }
        }
    }
    pub fn epoch_len(&self, epoch: usize) -> usize {
        self.epoch_indices(epoch).len()
    }
    pub fn batch_plan(&self, epochs: usize, batch_size: usize) -> Vec<usize> {
        (0..epochs)
            .map(|e| self.epoch_len(e).div_ceil(batch_size))
            .collect()
    }
    pub fn print_plan(&self, epochs: usize, batch_size: usize) {
        println!(
            "Code data: {} train ({} Rust docs), {} held-out; high/excluded/unscored are ineligible in curriculum mode",
            self.training.len(),
            self.docs_len,
            self.validation.len()
        );
        for epoch in 0..epochs {
            let window = if self.docs_len > 0 && epoch > 0 { self.docs_window(epoch - 1) } else { Vec::new() };
            let docs = if !window.is_empty() {
                let tokens: usize = window.iter().map(|&i| self.tokens(i)).sum();
                let total: usize = self.epoch_indices(epoch).iter().map(|&i| self.tokens(i)).sum();
                format!("{} docs = {:.0}% of tokens + ", window.len(), 100.0 * tokens as f64 / total.max(1) as f64)
            } else { String::new() };
            let code = epoch - (self.docs_len > 0 && epoch > 0) as usize;
            let label = match (self.docs_len > 0 && epoch == 0, self.tier_ends) {
                (true, _) => " (Rust docs)".to_owned(),
                (false, Some(_)) => format!(" ({docs}{})", ["low", "low + mild", "low + mild + moderate"][code.min(2)]),
                (false, None) if self.docs_len > 0 => format!(" ({docs}all code)"),
                _ => String::new(),
            };
            println!(
                "Epoch {}: {} samples, {} batches{label}",
                epoch + 1,
                self.epoch_len(epoch),
                self.epoch_len(epoch).div_ceil(batch_size),
            );
        }
    }
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

pub fn prepare(config: &CodeConfig, tokenizer: &TokenizerKind) -> Result<PreparedCode> {
    let mut prepared = prepare_code(config, tokenizer)?;
    if let Some(directory) = &config.docs {
        let (mut docs, identity) = super::code_docs::load(directory, config, tokenizer)?;
        let docs_len = docs.len();
        docs.append(&mut prepared.training);
        prepared.training = docs;
        prepared.tier_ends = prepared.tier_ends.map(|ends| ends.map(|end| end + docs_len));
        prepared.docs_len = docs_len;
        prepared.docs_share = config.docs_share;
        prepared.identity["docs"] = identity;
        prepared.identity["docs"]["share"] = json!(config.docs_share);
    }
    Ok(prepared)
}

fn prepare_code(config: &CodeConfig, tokenizer: &TokenizerKind) -> Result<PreparedCode> {
    config.validate()?;
    let Some(directory) = &config.curriculum else {
        let metadata = sample_cache::read_metadata(&config.cache)?;
        ensure!(
            metadata.seed == config.seed,
            "cache seed differs from config"
        );
        let samples = sample_cache::load_cache_with_objective(
            &config.cache,
            tokenizer,
            TrainingStage::Language,
            config.max_seq_len,
            CacheObjective::RawCode,
        )?;
        let (training, validation) = code_corpus::split_validation(samples, config)?;
        return Ok(PreparedCode {
            training,
            validation,
            tier_ends: None,
            docs_len: 0,
            docs_share: 0.0,
            seed: config.seed,
            identity: json!({"mode": "full-cache", "cache": metadata, "seed": config.seed,
                "validation_fraction": config.validation_fraction}),
        });
    };
    let report: Value = serde_json::from_slice(
        &std::fs::read(directory.join("report.json"))
            .context("curriculum requires the sorter's report.json")?,
    )?;
    let reference_path = directory.join("reference.json");
    let reference_sha = file_sha256(&reference_path)?;
    ensure!(
        report["reference_sha256"].as_str() == Some(reference_sha.as_str()),
        "curriculum reference digest mismatch"
    );
    let reference: super::code_complexity::Reference =
        serde_json::from_slice(&std::fs::read(reference_path)?)?;
    reference.validate()?;
    let mut buckets = Vec::new();
    let mut samples = Vec::new();
    let mut lengths = [0; 3];
    // This whitelist is deliberate: no directory glob or original code.bin fallback.
    for (tier, name) in ALLOWED_BUCKETS.iter().enumerate() {
        let count = report["bucket_counts"][name]
            .as_u64()
            .context("missing curriculum bucket count")?;
        if count == 0 {
            buckets.push(json!({"name": name, "samples": 0}));
            continue;
        }
        let path = directory.join(format!("{name}.bin"));
        let sha =
            file_sha256(&path).with_context(|| format!("curriculum bucket {}", path.display()))?;
        ensure!(
            report["bucket_caches"][name]["sha256"].as_str() == Some(sha.as_str()),
            "{name} digest mismatch"
        );
        let metadata = sample_cache::read_metadata(&path)?;
        ensure!(
            metadata.seed == config.seed && metadata.samples as u64 == count,
            "{name} seed or count mismatch"
        );
        let group = sample_cache::load_cache_with_objective(
            &path,
            tokenizer,
            TrainingStage::Language,
            if config.documented.is_some() {
                metadata.max_seq_len
            } else {
                config.max_seq_len
            },
            CacheObjective::RawCode,
        )?;
        lengths[tier] = group.len();
        samples.extend(group);
        buckets.push(json!({"name": name, "sha256": sha, "metadata": metadata}));
    }
    let held_out = code_corpus::validation_files(&samples, config)?;
    let documented_identity = if let Some(directory) = &config.documented {
        let (mixed, ends, identity) = super::code_documented::mix(
            directory,
            &samples,
            lengths,
            &held_out,
            config,
            tokenizer,
            &reference_sha,
            &buckets,
        )?;
        samples = mixed;
        lengths = ends;
        Some(identity)
    } else {
        None
    };
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut tier_ends = [0; 3];
    let mut samples = samples.into_iter();
    for tier in 0..3 {
        for sample in samples.by_ref().take(lengths[tier]) {
            if held_out.contains(&sample.pair.0) {
                validation.push(sample);
            } else {
                training.push(sample);
            }
        }
        tier_ends[tier] = training.len();
    }
    ensure!(
        tier_ends[0] > 0,
        "curriculum needs low-tier training samples after validation split"
    );
    ensure!(
        !validation.is_empty(),
        "curriculum needs held-out validation samples"
    );
    let mut identity = json!({"mode": "low-mild-moderate-v1", "reference_sha256": reference_sha,
        "buckets": buckets, "seed": config.seed, "validation_fraction": config.validation_fraction,
        "tier_ends": tier_ends});
    if let Some(documented) = documented_identity {
        identity["mode"] = json!("documented-balanced-lmm-v1");
        identity["documented"] = documented;
    }
    Ok(PreparedCode {
        training,
        validation,
        tier_ends: Some(tier_ends),
        docs_len: 0,
        docs_share: 0.0,
        seed: config.seed,
        identity,
    })
}

#[cfg(test)]
#[path = "../../tests/code_curriculum/mod.rs"]
mod tests;
