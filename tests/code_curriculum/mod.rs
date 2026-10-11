use super::*;
use crate::brain::code_complexity::{self, DEFAULT_WEIGHTS, Preprocessing, Reference};
use std::{collections::HashSet, path::PathBuf};

struct Fixture {
    directory: PathBuf,
    config: CodeConfig,
    tokenizer: TokenizerKind,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
impl Fixture {
    fn new() -> Result<Self> {
        let directory =
            std::env::temp_dir().join(format!("yumon-curriculum-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("buckets"))?;
        let sources = [
            "fn low(x: u32) -> u32 { x }",
            "fn mild(x: u32) -> u32 { if x > 2 { x } else { 2 } }",
            "fn moderate(mut x: u32) -> u32 { while x > 2 { if x > 5 { x -= 1; } else { break; } } x }",
        ];
        let source = directory.join("source.rs");
        std::fs::write(&source, sources.join("\n"))?;
        let tokenizer_path = directory.join("tokenizer");
        code_corpus::train_tokenizer(&[source], 320, tokenizer_path.to_str().unwrap())?;
        let tokenizer = code_corpus::load_tokenizer(tokenizer_path.to_str().unwrap())?;
        let config = CodeConfig {
            cache: directory.join("missing-original.bin"),
            curriculum: Some(directory.join("buckets")),
            tokenizer: tokenizer_path.to_str().unwrap().into(),
            max_seq_len: 128,
            validation_fraction: 0.25,
            epochs: 5,
            batch_size: 3,
            ..Default::default()
        };
        let records: Vec<_> = sources
            .iter()
            .map(|s| code_complexity::analyze("fixture", "rust", s, "", ""))
            .collect();
        let reference = Reference::fit(
            &records,
            "test-1",
            "0".repeat(64),
            DEFAULT_WEIGHTS,
            Preprocessing::default(),
        )?;
        let buckets = config.curriculum.as_ref().unwrap();
        std::fs::write(
            buckets.join("reference.json"),
            serde_json::to_vec(&reference)?,
        )?;
        let mut report = json!({"reference_sha256": file_sha256(&buckets.join("reference.json"))?,
            "bucket_counts": {}, "bucket_caches": {}});
        for (tier, name) in ALLOWED_BUCKETS.iter().enumerate() {
            let mut group = Vec::new();
            for file in 0..6 {
                group.extend(code_corpus::chunk_source(
                    sources[tier],
                    &tokenizer,
                    128,
                    &format!("file-{file}.rs"),
                )?);
            }
            let path = buckets.join(format!("{name}.bin"));
            let metadata = sample_cache::write_cache_with_objective(
                &path,
                &tokenizer,
                TrainingStage::Language,
                128,
                config.seed,
                &group,
                CacheObjective::RawCode,
            )?;
            report["bucket_counts"][name] = json!(group.len());
            report["bucket_caches"][name] =
                json!({"sha256": file_sha256(&path)?, "metadata": metadata});
        }
        // These are deliberately invalid files. Any attempt to open them would fail.
        for name in ["tier_4_high", "excluded", "unscored", "failed"] {
            std::fs::write(buckets.join(format!("{name}.bin")), b"must never be loaded")?;
            report["bucket_counts"][name] = json!(999);
        }
        std::fs::write(buckets.join("report.json"), serde_json::to_vec(&report)?)?;
        Ok(Self {
            directory,
            config,
            tokenizer,
        })
    }
}

#[test]
fn cumulative_schedule_never_loads_high_or_original_and_has_no_file_leakage() -> Result<()> {
    let fixture = Fixture::new()?;
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_eq!(prepared.tier_ends, Some([4, 8, 12]));
    assert_eq!(
        (0..6).map(|e| prepared.epoch_len(e)).collect::<Vec<_>>(),
        [4, 8, 12, 12, 12, 12]
    );
    assert_eq!(prepared.batch_plan(5, 3), [2, 3, 4, 4, 4]);
    assert_eq!(prepared.validation.len(), 6);
    let training_files: HashSet<_> = prepared.training.iter().map(|s| &s.pair.0).collect();
    assert!(
        prepared
            .validation
            .iter()
            .all(|s| !training_files.contains(&s.pair.0))
    );
    for epoch in 0..5 {
        let decoded = prepared.training[..prepared.epoch_len(epoch)]
            .iter()
            .map(|s| fixture.tokenizer.decode(&s.target_labels))
            .collect::<Vec<_>>();
        assert_eq!(decoded.iter().filter(|s| s.contains("fn low(")).count(), 4);
        assert_eq!(
            decoded.iter().filter(|s| s.contains("fn mild(")).count(),
            if epoch >= 1 { 4 } else { 0 }
        );
        assert_eq!(
            decoded
                .iter()
                .filter(|s| s.contains("fn moderate("))
                .count(),
            if epoch >= 2 { 4 } else { 0 }
        );
    }
    let repeated = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_eq!(prepared.identity, repeated.identity);
    assert_eq!(
        format!("{:?}", prepared.training),
        format!("{:?}", repeated.training)
    );
    Ok(())
}

#[test]
fn missing_corrupt_and_recalibrated_inputs_fail_without_falling_back() -> Result<()> {
    let fixture = Fixture::new()?;
    let buckets = fixture.config.curriculum.as_ref().unwrap();
    let path = buckets.join("tier_2_mild.bin");
    let original = std::fs::read(&path)?;
    std::fs::write(&path, b"corrupt")?;
    assert!(prepare(&fixture.config, &fixture.tokenizer).is_err());
    std::fs::remove_file(&path)?;
    assert!(prepare(&fixture.config, &fixture.tokenizer).is_err());
    std::fs::write(&path, original)?;
    let before = prepare(&fixture.config, &fixture.tokenizer)?.identity;
    let mut changed = fixture.config.clone();
    changed.validation_fraction = 0.5;
    assert_ne!(before, prepare(&changed, &fixture.tokenizer)?.identity);
    std::fs::write(buckets.join("reference.json"), b"{}")?;
    assert!(prepare(&fixture.config, &fixture.tokenizer).is_err());
    Ok(())
}

#[test]
fn every_code_architecture_uses_the_same_curriculum_check_path() -> Result<()> {
    let fixture = Fixture::new()?;
    for name in ["decoder-only", "moe", "xlstm", "encoder-decoder"] {
        let mut config = fixture.config.clone();
        config.architecture = serde_json::from_value(json!(name))?;
        crate::brain::train::run_code(&config, true)?;
    }
    Ok(())
}

impl Fixture {
    fn with_documentation() -> Result<Self> {
        let mut fixture = Self::new()?;
        let docs = fixture.directory.join("documented");
        std::fs::create_dir(&docs)?;
        fixture.config.documented = Some(docs.clone());
        let buckets = fixture.config.curriculum.as_ref().unwrap();
        let mut report: Value =
            serde_json::from_slice(&std::fs::read(buckets.join("report.json"))?)?;
        let mut manifest = json!({"reference_sha256": report["reference_sha256"],
            "tokenizer_sha256": file_sha256(&Path::new(&fixture.config.tokenizer).join("tokenizer.json"))?, "inputs": {}});
        for name in ALLOWED_BUCKETS {
            let path = buckets.join(format!("{name}.bin"));
            let mut group = sample_cache::load_cache_with_objective(
                &path,
                &fixture.tokenizer,
                TrainingStage::Language,
                128,
                CacheObjective::RawCode,
            )?;
            let mut journal = Vec::new();
            for (index, sample) in group.iter().enumerate() {
                let source = fixture.tokenizer.decode(&sample.target_labels);
                serde_json::to_writer(
                    &mut journal,
                    &json!({"index": index,
                    "source_sha256": format!("{:x}", Sha256::digest(source.as_bytes())),
                    "doc_comment": "/// Returns a value."}),
                )?;
                journal.push(b'\n');
            }
            // Distinct candidates in every source file, plus a naturally documented
            // candidate that must never be used as the undocumented half.
            for file in 0..6 {
                for index in 0..10 {
                    group.extend(code_corpus::chunk_source(
                        &format!(
                            "

fn extra_{file}_{index}() -> u32 {{ {index} }}"
                        ),
                        &fixture.tokenizer,
                        128,
                        &format!("file-{file}.rs"),
                    )?);
                }
                group.extend(code_corpus::chunk_source(
                    "/// Existing documentation.\nfn natural() {}",
                    &fixture.tokenizer,
                    128,
                    &format!("file-{file}.rs"),
                )?);
            }
            std::fs::remove_file(&path)?;
            let metadata = sample_cache::write_cache_with_objective(
                &path,
                &fixture.tokenizer,
                TrainingStage::Language,
                128,
                fixture.config.seed,
                &group,
                CacheObjective::RawCode,
            )?;
            report["bucket_counts"][name] = json!(group.len());
            let sha = file_sha256(&path)?;
            report["bucket_caches"][name] = json!({"sha256": sha, "metadata": metadata});
            manifest["inputs"][name] = json!(sha);
            // An interrupted trailing record is ignored, never rewritten.
            journal.extend_from_slice(b"{\"index\":");
            std::fs::write(docs.join(format!("{name}.jsonl")), journal)?;
        }
        std::fs::write(buckets.join("report.json"), serde_json::to_vec(&report)?)?;
        std::fs::write(docs.join("generation.json"), serde_json::to_vec(&manifest)?)?;
        // Also exercise loading 128-token originals into a larger training context.
        fixture.config.max_seq_len = 256;
        Ok(fixture)
    }
}

#[test]
fn saved_documents_are_balanced_each_epoch_and_partition_without_source_overlap() -> Result<()> {
    let fixture = Fixture::with_documentation()?;
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_eq!(prepared.tier_ends, Some([8, 16, 24]));
    assert_eq!(prepared.validation.len(), 12);
    for samples in [
        &prepared.validation[..],
        &prepared.training[..8],
        &prepared.training[..16],
        &prepared.training[..],
    ] {
        let decoded = samples
            .iter()
            .map(|sample| fixture.tokenizer.decode(&sample.target_labels))
            .collect::<Vec<_>>();
        assert_eq!(
            decoded
                .iter()
                .filter(|source| source.contains("/// Returns a value."))
                .count(),
            samples.len() / 2
        );
        assert_eq!(
            decoded
                .iter()
                .filter(|source| source.contains("fn extra_"))
                .count(),
            samples.len() / 2
        );
        assert!(!decoded.iter().any(|source| source.contains("fn natural")));
        assert!(samples.iter().all(|sample| sample.input_ids.len() == 256));
    }
    let files: HashSet<_> = prepared
        .training
        .iter()
        .map(|sample| &sample.pair.0)
        .collect();
    assert!(
        prepared
            .validation
            .iter()
            .all(|sample| !files.contains(&sample.pair.0))
    );
    assert_eq!(
        prepared.identity,
        prepare(&fixture.config, &fixture.tokenizer)?.identity
    );
    Ok(())
}

#[test]
fn oversized_documented_records_are_skipped_without_losing_fitting_records() -> Result<()> {
    let fixture = Fixture::with_documentation()?;
    let before = prepare(&fixture.config, &fixture.tokenizer)?;
    let journal = fixture.config.documented.as_ref().unwrap().join("tier_1_low.jsonl");
    let original = std::fs::read_to_string(&journal)?;
    let mut entries = original.lines().map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    entries[0]["doc_comment"] = json!(format!("/// {}", "long comment ".repeat(1000)));
    let changed = entries.iter().map(|entry| format!("{entry}\n")).collect::<String>();
    std::fs::write(&journal, changed)?;
    let after = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_eq!(after.training.len() + after.validation.len(),
        before.training.len() + before.validation.len() - 2);
    assert!(after.training.iter().chain(&after.validation)
        .all(|sample| sample.input_ids.len() == fixture.config.max_seq_len));
    for tier in after.identity["documented"]["tiers"].as_array().unwrap() {
        assert_eq!(tier["documented"].as_u64().unwrap(),
            tier["training_each"].as_u64().unwrap() + tier["validation_each"].as_u64().unwrap());
    }
    Ok(())
}

#[test]
fn documented_inputs_fail_closed_on_source_mismatch_or_insufficient_context() -> Result<()> {
    let fixture = Fixture::with_documentation()?;
    let journal = fixture
        .config
        .documented
        .as_ref()
        .unwrap()
        .join("tier_1_low.jsonl");
    let original = std::fs::read(&journal)?;
    let mut entry: Value = serde_json::from_slice(original.split(|&b| b == b'\n').next().unwrap())?;
    entry["source_sha256"] = json!("wrong");
    std::fs::write(&journal, format!("{}\n", entry))?;
    assert!(
        prepare(&fixture.config, &fixture.tokenizer)
            .err()
            .unwrap()
            .to_string()
            .contains("source mismatch")
    );
    std::fs::write(&journal, &original)?;
    let mut config = fixture.config.clone();
    config.max_seq_len = 2;
    assert!(prepare(&config, &fixture.tokenizer).is_err());
    Ok(())
}

#[test]
fn insufficient_distinct_undocumented_samples_fail_without_reusing_originals() -> Result<()> {
    let fixture = Fixture::with_documentation()?;
    let buckets = fixture.config.curriculum.as_ref().unwrap();
    let docs = fixture.config.documented.as_ref().unwrap();
    let path = buckets.join("tier_1_low.bin");
    let mut group = sample_cache::load_cache_with_objective(
        &path,
        &fixture.tokenizer,
        TrainingStage::Language,
        128,
        CacheObjective::RawCode,
    )?;
    group.truncate(6);
    std::fs::remove_file(&path)?;
    let metadata = sample_cache::write_cache_with_objective(
        &path,
        &fixture.tokenizer,
        TrainingStage::Language,
        128,
        fixture.config.seed,
        &group,
        CacheObjective::RawCode,
    )?;
    let sha = file_sha256(&path)?;
    let mut report: Value = serde_json::from_slice(&std::fs::read(buckets.join("report.json"))?)?;
    report["bucket_counts"]["tier_1_low"] = json!(6);
    report["bucket_caches"]["tier_1_low"] = json!({"sha256": sha, "metadata": metadata});
    std::fs::write(buckets.join("report.json"), serde_json::to_vec(&report)?)?;
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(docs.join("generation.json"))?)?;
    manifest["inputs"]["tier_1_low"] = json!(sha);
    std::fs::write(docs.join("generation.json"), serde_json::to_vec(&manifest)?)?;
    let error = prepare(&fixture.config, &fixture.tokenizer)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("distinct undocumented"), "{error}");
    Ok(())
}

#[test]
fn existing_code_bpe_trainer_uses_balanced_training_split_and_rebuilds_matching_caches()
-> Result<()> {
    let fixture = Fixture::with_documentation()?;
    let before = prepare(&fixture.config, &fixture.tokenizer)?;
    let original_tokenizer =
        file_sha256(&Path::new(&fixture.config.tokenizer).join("tokenizer.json"))?;
    let original_journal = file_sha256(
        &fixture
            .config
            .documented
            .as_ref()
            .unwrap()
            .join("tier_1_low.jsonl"),
    )?;
    let output = fixture.directory.join("retrained");
    let config = code_corpus::prepare_training_tokenizer(&fixture.config, &output)?;
    let tokenizer = code_corpus::load_tokenizer(&config.tokenizer)?;
    let after = prepare(&config, &tokenizer)?;
    assert_eq!(before.tier_ends, after.tier_ends);
    assert_eq!(before.training.len(), after.training.len());
    assert_eq!(before.validation.len(), after.validation.len());
    let corpus = std::fs::read_to_string(output.join("tokenizer-corpus.jsonl"))?;
    let records = corpus
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(records.len(), before.training.len());
    for (record, sample) in records.iter().zip(&before.training) {
        assert_eq!(record["source"], json!(sample.pair));
        assert_eq!(
            record["text"],
            fixture.tokenizer.decode(&sample.target_labels)
        );
        assert!(
            !before
                .validation
                .iter()
                .any(|held_out| held_out.pair.0 == sample.pair.0)
        );
    }
    assert_eq!(
        records
            .iter()
            .filter(|record| record["text"]
                .as_str()
                .unwrap()
                .contains("/// Returns a value."))
            .count(),
        records.len() / 2
    );
    assert_eq!(
        file_sha256(&Path::new(&fixture.config.tokenizer).join("tokenizer.json"))?,
        original_tokenizer
    );
    assert_eq!(
        file_sha256(
            &fixture
                .config
                .documented
                .as_ref()
                .unwrap()
                .join("tier_1_low.jsonl")
        )?,
        original_journal
    );
    assert!(code_corpus::prepare_training_tokenizer(&fixture.config, &output).is_err());
    let deployed = CodeConfig::load(&output.join("docker-config.json"))?;
    assert_eq!(deployed.tokenizer, "yumon_code_bpe");
    assert!(!output.join("buckets/tier_4_high.bin").exists());
    Ok(())
}
