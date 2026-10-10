//! Model-independent cumulative curriculum, restricted to low/mild/moderate.
use super::{
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
    pub tier_ends: Option<[usize; 3]>,
    pub identity: Value,
    pub seed: u64,
}
impl PreparedCode {
    pub fn epoch_len(&self, epoch: usize) -> usize {
        self.tier_ends
            .map_or(self.training.len(), |ends| ends[epoch.min(2)])
    }
    pub fn batch_plan(&self, epochs: usize, batch_size: usize) -> Vec<usize> {
        (0..epochs)
            .map(|e| self.epoch_len(e).div_ceil(batch_size))
            .collect()
    }
    pub fn print_plan(&self, epochs: usize, batch_size: usize) {
        println!(
            "Code data: {} train, {} held-out; high/excluded/unscored are ineligible in curriculum mode",
            self.training.len(),
            self.validation.len()
        );
        for epoch in 0..epochs {
            println!(
                "Epoch {}: {} samples, {} batches{}",
                epoch + 1,
                self.epoch_len(epoch),
                self.epoch_len(epoch).div_ceil(batch_size),
                if self.tier_ends.is_some() {
                    [" (low)", " (low + mild)", " (low + mild + moderate)"][epoch.min(2)]
                } else {
                    ""
                }
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
            config.max_seq_len,
            CacheObjective::RawCode,
        )?;
        lengths[tier] = group.len();
        samples.extend(group);
        buckets.push(json!({"name": name, "sha256": sha, "metadata": metadata}));
    }
    let held_out = code_corpus::validation_files(&samples, config)?;
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
    Ok(PreparedCode {
        training,
        validation,
        tier_ends: Some(tier_ends),
        seed: config.seed,
        identity: json!({"mode": "low-mild-moderate-v1", "reference_sha256": reference_sha,
            "buckets": buckets, "seed": config.seed, "validation_fraction": config.validation_fraction,
            "tier_ends": tier_ends}),
    })
}

#[cfg(test)]
#[path = "../../tests/code_curriculum/mod.rs"]
mod tests;
