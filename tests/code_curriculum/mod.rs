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
