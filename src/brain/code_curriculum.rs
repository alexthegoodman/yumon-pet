//! Model-independent mixed curriculum, restricted to low/mild/moderate.
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
    /// Training index range of a tier (offsets include any docs prefix).
    fn tier_range(&self, tier: usize) -> std::ops::Range<usize> {
        let ends = self.tier_ends.expect("tier ranges need a curriculum");
        (if tier == 0 { self.docs_len } else { ends[tier - 1] })..ends[tier]
    }
    /// Code samples for a code epoch. Without tiers that is everything. With tiers
    /// the epoch is a mix anchored on its hardest tier, which is taken whole:
    /// epoch 1 100% low, epoch 2 20% low + 80% mild, epoch 3+ 10% low + 20% mild
    /// + 70% moderate. Easier tiers are capped by availability and rotate through
    /// their samples from one epoch to the next.
    fn code_indices(&self, code_epoch: usize) -> Vec<usize> {
        if self.tier_ends.is_none() {
            return (self.docs_len..self.training.len()).collect();
        }
        let (anchor, shares) = match code_epoch {
            0 => (0, [1.0, 0.0, 0.0]),
            1 => (1, [0.2, 0.8, 0.0]),
            _ => (2, [0.1, 0.2, 0.7]),
        };
        let anchor_len = self.tier_range(anchor).len();
        let mut indices = Vec::new();
        for tier in 0..=anchor {
            let range = self.tier_range(tier);
            let want = (anchor_len as f64 * shares[tier] / shares[anchor]).round() as usize;
            let take = if tier == anchor { anchor_len } else { want.min(range.len()) };
            let start = if range.is_empty() { 0 } else { code_epoch * take % range.len() };
            indices.extend((0..take).map(|i| range.start + (start + i) % range.len()));
        }
        indices
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
            let code: usize = self.code_indices(e).into_iter().map(|i| self.tokens(i)).sum();
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
            (0, epoch) => self.code_indices(epoch),
            (docs, 0) => (0..docs).collect(),
            (_, epoch) => {
                let mut indices = self.docs_window(epoch - 1);
                indices.extend(self.code_indices(epoch - 1));
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
                (false, Some(_)) => format!(" ({docs}{})", ["low", "20% low + 80% mild", "10% low + 20% mild + 70% moderate"][code.min(2)]),
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
    use super::code_data::{self, Kind, TIERS};
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
    // Each tier is its documented file followed by its undocumented file: the same
    // code with and without a doc comment. Nothing else is read (no high tier).
    let mut tiers = Vec::new();
    let mut files = Vec::new();
    for tier in 0..TIERS.len() {
        let documented = code_data::scan(&code_data::file(directory, Kind::Documented, tier))?.records;
        let undocumented = code_data::scan(&code_data::file(directory, Kind::Undocumented, tier))?.records;
        files.extend(documented.iter().chain(&undocumented).map(|r| r.path.clone()));
        tiers.push((documented, undocumented));
    }
    ensure!(!files.is_empty(), "no samples under {}; run gen_synthetic_captions", directory.display());
    let held_out = code_data::held_out_files(files.iter().map(String::as_str), config)?;
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut tier_ends = [0; 3];
    let mut summary = Vec::new();
    for (tier, (documented, undocumented)) in tiers.iter().enumerate() {
        let mut samples = Vec::new();
        let mut skipped = 0;
        for (kind, records) in [(Kind::Documented, documented), (Kind::Undocumented, undocumented)] {
            for record in records {
                match code_data::to_sample(record, tokenizer, config.max_seq_len)? {
                    Some(sample) => samples.push((kind, sample)),
                    None => skipped += 1,
                }
            }
        }
        // Mix the two kinds so any slice of a tier (see `code_indices`) is balanced.
        samples.shuffle(&mut StdRng::seed_from_u64(config.seed.wrapping_add(tier as u64)));
        let mut counts = [[0usize; 2]; 2];
        for (kind, sample) in samples {
            let split = usize::from(held_out.contains(&sample.pair.0));
            counts[split][usize::from(kind == Kind::Undocumented)] += 1;
            if split == 1 { validation.push(sample) } else { training.push(sample) }
        }
        tier_ends[tier] = training.len();
        println!("{}: {} documented + {} undocumented train; {} + {} validation; {skipped} over {} tokens skipped",
            TIERS[tier], counts[0][0], counts[0][1], counts[1][0], counts[1][1], config.max_seq_len);
        summary.push(json!({"name": TIERS[tier], "documented": documented.len(),
            "undocumented": undocumented.len(), "skipped_oversized": skipped}));
    }
    ensure!(
        tier_ends[0] > 0,
        "curriculum needs low-tier training samples after validation split"
    );
    ensure!(
        !validation.is_empty(),
        "curriculum needs held-out validation samples"
    );
    let identity = json!({"mode": "documented-tiers-v3", "tiers": summary, "context": config.max_seq_len,
        "seed": config.seed, "validation_fraction": config.validation_fraction, "tier_ends": tier_ends});
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
