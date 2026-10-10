//! CPU-only curriculum cache sorting. No model or inference initialization.
#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    std::{
        collections::BTreeMap,
        fs::File,
        io::{BufWriter, Write},
        path::PathBuf,
    },
    yumon_pet::brain::{
        bpe::TokenizerKind,
        code_complexity::{self as complexity, Preprocessing, Record, Reference},
        code_corpus::{self, CodeConfig},
        sample_cache::{self, CacheObjective},
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(about = "Sort a prepared Rust cache into frozen empirical complexity tiers")]
struct Args {
    #[arg(long, default_value = "configs/yumon-code.json")]
    code_config: PathBuf,
    /// Frozen calibration JSON. Never fitted implicitly or overwritten.
    #[arg(long)]
    reference: PathBuf,
    /// Explicitly fit a NEW reference from the input cache with this version.
    #[arg(long)]
    calibrate: Option<String>,
    /// Five primary weights in the documented order; only valid during calibration.
    #[arg(long, value_delimiter = ',', num_args = 1, requires = "calibrate")]
    weights: Option<Vec<f64>>,
    /// Optional preprocessing JSON; frozen into the reference during calibration.
    #[arg(long, requires = "calibrate")]
    preprocessing: Option<PathBuf>,
    /// Must be a new directory, so an older run cannot be mixed with this run.
    #[arg(long, default_value = "training-cache/code_buckets")]
    output: PathBuf,
    #[arg(long, default_value_t = 12)]
    preview_samples: usize,
}

#[cfg(not(target_arch = "wasm32"))]
fn write_json(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    let file = File::options().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(not(target_arch = "wasm32"))]
fn run(args: Args) -> Result<()> {
    ensure!(
        !args.output.exists(),
        "output directory already exists: {}",
        args.output.display()
    );
    if args.calibrate.is_some() {
        ensure!(
            !args.reference.exists(),
            "reference already exists; choose a new version and path"
        );
    }
    let config = CodeConfig::load(&args.code_config)?;
    let metadata = sample_cache::read_metadata(&config.cache)?;
    ensure!(
        metadata.objective == CacheObjective::RawCode,
        "requires a raw-code cache"
    );
    let tokenizer = code_corpus::load_tokenizer(&config.tokenizer)?;
    let samples = sample_cache::load_cache_with_objective(
        &config.cache,
        &tokenizer,
        metadata.stage,
        metadata.max_seq_len,
        CacheObjective::RawCode,
    )?;
    let input_sha = file_digest(&config.cache)?;
    let existing = if args.calibrate.is_none() {
        let reference: Reference = serde_json::from_slice(
            &std::fs::read(&args.reference)
                .context("reading frozen reference (use --calibrate VERSION to create one)")?,
        )?;
        reference.validate()?;
        Some(reference)
    } else {
        None
    };
    let preprocessing = if let Some(reference) = &existing {
        reference.preprocessing.clone()
    } else if let Some(path) = args.preprocessing {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        Preprocessing::default()
    };
    let mut records = Vec::with_capacity(samples.len());
    for (i, sample) in samples.iter().enumerate() {
        let decoded = decode(&tokenizer, &sample.target_labels);
        let source = decoded.as_deref().unwrap_or("");
        let mut record = complexity::analyze(
            &format!("{input_sha}:{i}"),
            "rust",
            source,
            &sample.pair.0,
            &sample.pair.1,
        );
        if let Err(error) = decoded.as_ref() {
            record.analysis_status = "decode_error".into();
            record.metrics = Default::default();
            record.analysis_errors = vec![error.to_string()];
        }
        if preprocessing.excluded(&source, &sample.pair.0) {
            record.analysis_status = "excluded".into();
            record.metrics = Default::default();
            record.analysis_errors = vec!["Excluded by frozen preprocessing rules".into()];
        }
        records.push(record);
        if (i + 1) % 5000 == 0 {
            println!("Analyzed {} / {} samples", i + 1, samples.len());
        }
    }
    let reference = if let Some(reference) = existing {
        reference
    } else {
        let weights = args
            .weights
            .map(|w| {
                <[f64; 5]>::try_from(w).map_err(|_| {
                    anyhow::anyhow!("--weights requires exactly five comma-separated values")
                })
            })
            .transpose()?
            .unwrap_or(complexity::DEFAULT_WEIGHTS);
        let reference = Reference::fit(
            &records,
            args.calibrate.as_deref().unwrap(),
            input_sha.clone(),
            weights,
            preprocessing,
        )?;
        if let Some(parent) = args
            .reference
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        write_json(&args.reference, &reference)?;
        reference
    };
    let reference_sha = file_digest(&args.reference)?;
    for record in &mut records {
        reference.classify(record, &reference_sha);
    }

    // Work in an unpublished sibling directory; failed runs never look complete.
    let parent = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".code-buckets-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&temporary)?;
    let mut record_writer = BufWriter::new(File::create(temporary.join("records.jsonl"))?);
    for record in &records {
        serde_json::to_writer(&mut record_writer, record)?;
        record_writer.write_all(b"\n")?;
    }
    record_writer.flush()?;
    drop(record_writer);
    let mut groups: [Vec<_>; 7] = std::array::from_fn(|_| Vec::new());
    let mut status_counts = BTreeMap::<String, usize>::new();
    let mut missing = [0usize; 9];
    let mut whitespace_stable = 0usize;
    for (i, (sample, record)) in samples.into_iter().zip(&records).enumerate() {
        *status_counts
            .entry(record.analysis_status.clone())
            .or_default() += 1;
        for (i, value) in record.metrics.values().iter().enumerate() {
            if value.is_none() {
                missing[i] += 1;
            }
        }
        let group = record.complexity_tier.map(|t| t as usize - 1).unwrap_or(
            if record.analysis_status == "excluded" {
                4
            } else if record.analysis_status == "missing_metrics" {
                5
            } else {
                6
            },
        );
        if i < args.preview_samples {
            println!(
                "\n{} {} -> {} (score {:?})\n{}",
                sample.pair.0,
                sample.pair.1,
                BUCKETS[group],
                record.structural_complexity_score,
                tokenizer.decode(&sample.target_labels)
            );
        }
        if record.complexity_tier.is_some() {
            let source = tokenizer.decode(&sample.target_labels);
            let mut changed =
                complexity::analyze("validation", "rust", &format!("\n{source}\n"), "", "");
            reference.classify(&mut changed, &reference_sha);
            if changed.complexity_tier == record.complexity_tier {
                whitespace_stable += 1;
            }
        }
        groups[group].push(sample);
    }
    let mut bucket_counts = BTreeMap::new();
    let mut bucket_metadata = BTreeMap::new();
    for (name, group) in BUCKETS.iter().zip(&groups) {
        bucket_counts.insert(*name, group.len());
        // Empty buckets have no .bin: the existing training loader rejects empty datasets.
        if !group.is_empty() {
            let meta = sample_cache::write_cache_with_objective(
                &temporary.join(format!("{name}.bin")),
                &tokenizer,
                metadata.stage,
                metadata.max_seq_len,
                metadata.seed,
                group,
                CacheObjective::RawCode,
            )?;
            bucket_metadata.insert(
                *name,
                serde_json::json!({"metadata": meta,
                "sha256": file_digest(&temporary.join(format!("{name}.bin")))?}),
            );
        }
    }
    std::fs::copy(&args.reference, temporary.join("reference.json"))?;
    let mut report = validation_report(&records, &reference);
    report["input_cache_sha256"] = input_sha.into();
    report["reference_sha256"] = reference_sha.into();
    report["bucket_counts"] = serde_json::to_value(bucket_counts)?;
    report["bucket_caches"] = serde_json::to_value(bucket_metadata)?;
    report["status_counts"] = serde_json::to_value(&status_counts)?;
    report["analysis_failure_rate"] = serde_json::json!(
        status_counts
            .iter()
            .filter(|(s, _)| s.as_str() != "success" && s.as_str() != "excluded")
            .map(|(_, n)| *n)
            .sum::<usize>() as f64
            / records.len() as f64
    );
    report["missing_metric_rates"] = serde_json::to_value(
        complexity::FEATURES
            .iter()
            .zip(missing)
            .map(|(name, n)| (*name, n as f64 / records.len() as f64))
            .collect::<BTreeMap<_, _>>(),
    )?;
    report["whitespace_tier_stability"] = serde_json::json!({"unchanged": whitespace_stable,
        "evaluated": records.iter().filter(|r| r.complexity_tier.is_some()).count()});
    write_json(&temporary.join("report.json"), &report)?;
    std::fs::rename(&temporary, &args.output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report["bucket_counts"])?
    );
    println!(
        "Wrote caches, records and validation report to {}",
        args.output.display()
    );
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
const BUCKETS: [&str; 7] = [
    "tier_1_low",
    "tier_2_mild",
    "tier_3_moderate",
    "tier_4_high",
    "excluded",
    "unscored",
    "failed",
];

#[cfg(not(target_arch = "wasm32"))]
fn file_digest(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(not(target_arch = "wasm32"))]
fn decode(tokenizer: &TokenizerKind, ids: &[usize]) -> Result<String> {
    let TokenizerKind::Bpe(tokenizer) = tokenizer else {
        anyhow::bail!("requires code BPE")
    };
    let ids = ids
        .iter()
        .map(|&id| u32::try_from(id))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    tokenizer.decode(&ids)
}

#[cfg(not(target_arch = "wasm32"))]
fn validation_report(records: &[Record], reference: &Reference) -> serde_json::Value {
    let scored: Vec<_> = records
        .iter()
        .filter(|r| r.complexity_tier.is_some())
        .collect();
    let features: [Vec<f64>; 9] = std::array::from_fn(|i| {
        scored
            .iter()
            .map(|r| r.metrics.values()[i].unwrap())
            .collect()
    });
    let scores: Vec<_> = scored
        .iter()
        .map(|r| r.structural_complexity_score.unwrap())
        .collect();
    let correlations: Vec<Vec<_>> = features
        .iter()
        .map(|x| {
            features
                .iter()
                .map(|y| complexity::correlation(x, y))
                .collect()
        })
        .collect();
    let mut baselines = BTreeMap::new();
    for index in [5, 0, 1] {
        let mut sorted = features[index].clone();
        sorted.sort_by(f64::total_cmp);
        let mut agreement = 0;
        for (i, record) in scored.iter().enumerate() {
            let rank = complexity::percentile(&sorted, features[index][i]);
            if complexity::tier(rank, &[0.25, 0.5, 0.75]) == record.complexity_tier.unwrap() {
                agreement += 1;
            }
        }
        baselines.insert(complexity::FEATURES[index], serde_json::json!({"pearson_with_composite": complexity::correlation(&features[index], &scores),
            "batch_quartile_tier_agreement_count": agreement}));
    }
    let mut sensitivity = vec![];
    for index in 0..5 {
        for factor in [0.9, 1.1] {
            let mut changed = reference.clone();
            changed.weights[index] *= factor;
            let sum = changed.weights.iter().sum::<f64>();
            for w in &mut changed.weights {
                *w /= sum;
            }
            let unchanged = scored
                .iter()
                .filter(|r| {
                    complexity::tier(
                        changed.score(&r.metrics.primary().unwrap()),
                        &reference.boundaries,
                    ) == r.complexity_tier.unwrap()
                })
                .count();
            sensitivity.push(
                serde_json::json!({"feature": complexity::FEATURES[index], "factor": factor,
                "unchanged_tiers_at_frozen_boundaries": unchanged}),
            );
        }
    }
    serde_json::json!({"total": records.len(), "scored": scored.len(), "language_counts": {"rust": records.len()},
        "scoring_version_usage": {reference.scoring_version.clone(): records.len()},
        "features": complexity::FEATURES, "pearson_correlations": correlations, "baseline_comparisons": baselines,
        "weight_sensitivity": sensitivity, "downstream_outcome_analysis": null,
        "language_comparison": "Rust only; no cross-language claims", "analyzer_version": complexity::ANALYZER_VERSION})
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Code bucket sorting requires a native build.");
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn cli_freezes_reference_and_publishes_reproducible_partition() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("yumon-sort-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory)?;
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(directory.clone());
        let source = "fn a() {}\nfn b(x: bool) { if x { return; } }\nfn c() { opaque!(); }";
        let source_path = directory.join("input.rs");
        std::fs::write(&source_path, source)?;
        let tokenizer_dir = directory.join("tokenizer");
        code_corpus::train_tokenizer(&[source_path], 300, tokenizer_dir.to_str().unwrap())?;
        let tokenizer = code_corpus::load_tokenizer(tokenizer_dir.to_str().unwrap())?;
        let samples = code_corpus::chunk_source(source, &tokenizer, 256, "input.rs")?;
        let config = CodeConfig {
            cache: directory.join("code.bin"),
            max_seq_len: 256,
            tokenizer: tokenizer_dir.to_str().unwrap().into(),
            ..Default::default()
        };
        sample_cache::write_cache_with_objective(
            &config.cache,
            &tokenizer,
            yumon_pet::brain::samples::TrainingStage::Language,
            256,
            config.seed,
            &samples,
            CacheObjective::RawCode,
        )?;
        let config_path = directory.join("config.json");
        write_json(&config_path, &config)?;
        let reference = directory.join("reference.json");
        let first = directory.join("first");
        let second = directory.join("second");
        let args = |output, calibrate| Args {
            code_config: config_path.clone(),
            reference: reference.clone(),
            calibrate,
            weights: None,
            preprocessing: None,
            output,
            preview_samples: 0,
        };
        run(args(first.clone(), Some("test-1".into())))?;
        let original_reference = std::fs::read(&reference)?;
        run(args(second.clone(), None))?;
        assert_eq!(original_reference, std::fs::read(&reference)?);
        assert_eq!(
            original_reference,
            std::fs::read(first.join("reference.json"))?
        );
        assert_eq!(
            std::fs::read(first.join("records.jsonl"))?,
            std::fs::read(second.join("records.jsonl"))?
        );
        assert_eq!(
            std::fs::read(first.join("report.json"))?,
            std::fs::read(second.join("report.json"))?
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(first.join("report.json"))?)?;
        assert_eq!(report["total"], 3);
        assert_eq!(report["scored"], 2);
        assert_eq!(report["bucket_counts"]["unscored"], 1);
        let total: u64 = report["bucket_counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|n| n.as_u64().unwrap())
            .sum();
        assert_eq!(total, 3);
        for name in BUCKETS {
            let path = first.join(format!("{name}.bin"));
            let count = report["bucket_counts"][name].as_u64().unwrap();
            assert_eq!(path.exists(), count > 0);
            if count > 0 {
                let group = sample_cache::load_cache_with_objective(
                    &path,
                    &tokenizer,
                    yumon_pet::brain::samples::TrainingStage::Language,
                    256,
                    CacheObjective::RawCode,
                )?;
                assert_eq!(group.len() as u64, count);
            }
        }
        assert!(run(args(first, None)).is_err());
        assert!(run(args(directory.join("third"), Some("test-1".into()))).is_err());
        Ok(())
    }

    #[test]
    fn custom_weights_require_explicit_calibration() {
        let weights = "0.25,0.30,0.20,0.15,0.10";
        assert!(
            Args::try_parse_from(["sort", "--reference", "ref.json", "--weights", weights])
                .is_err()
        );
        let args = Args::try_parse_from([
            "sort",
            "--reference",
            "ref.json",
            "--calibrate",
            "v2",
            "--weights",
            weights,
        ])
        .unwrap();
        assert_eq!(args.weights.unwrap(), complexity::DEFAULT_WEIGHTS);
    }
}
