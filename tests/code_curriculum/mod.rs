use super::*;
use crate::brain::code_data::{self, Kind, Record, Writer};
use std::{collections::HashSet, path::PathBuf};

const SOURCES: [&str; 3] = [
    "fn low(x: u32) -> u32 { x }",
    "fn mild(x: u32) -> u32 { if x > 2 { x } else { 2 } }",
    "fn moderate(mut x: u32) -> u32 { while x > 2 { if x > 5 { x -= 1; } else { break; } } x }",
];
const COMMENT: &str = "/// Returns a value.\n";

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
    /// Six files per tier, each stored with and without a doc comment.
    fn new() -> Result<Self> {
        let directory =
            std::env::temp_dir().join(format!("yumon-curriculum-{}", uuid::Uuid::new_v4()));
        let data = directory.join("data");
        std::fs::create_dir_all(&data)?;
        let source = directory.join("source.rs");
        std::fs::write(&source, SOURCES.join("\n"))?;
        let tokenizer_path = directory.join("tokenizer");
        code_corpus::train_tokenizer(&[source], 320, tokenizer_path.to_str().unwrap())?;
        let tokenizer = code_corpus::load_tokenizer(tokenizer_path.to_str().unwrap())?;
        let config = CodeConfig {
            cache: directory.join("missing-original.bin"),
            curriculum: Some(data.clone()),
            tokenizer: tokenizer_path.to_str().unwrap().into(),
            max_seq_len: 128,
            vocab_size: 320,
            validation_fraction: 0.25,
            epochs: 5,
            batch_size: 3,
            ..Default::default()
        };
        for (tier, source) in SOURCES.iter().enumerate() {
            let mut documented = Writer::open(&code_data::file(&data, Kind::Documented, tier))?;
            let mut undocumented = Writer::open(&code_data::file(&data, Kind::Undocumented, tier))?;
            for file in 0..6 {
                let record = |text: String| Record {
                    path: format!("file-{file}.rs"),
                    location: "line 1".into(),
                    text,
                };
                documented.append(&record(format!("{COMMENT}{source}")))?;
                undocumented.append(&record(source.to_string()))?;
            }
        }
        // High-tier files exist but must never be opened.
        for kind in [Kind::Documented, Kind::Undocumented] {
            let dir = code_data::file(&data, kind, 0).parent().unwrap().to_path_buf();
            std::fs::write(dir.join("tier_4_high.bin"), b"must never be loaded")?;
        }
        Ok(Self { directory, config, tokenizer })
    }

    fn tier_file(&self, kind: Kind, tier: usize) -> PathBuf {
        code_data::file(self.config.curriculum.as_ref().unwrap(), kind, tier)
    }
}

fn decoded(fixture: &Fixture, samples: &[crate::brain::samples::Sample]) -> Vec<String> {
    samples.iter().map(|s| fixture.tokenizer.decode(&s.target_labels)).collect()
}

#[test]
fn cumulative_schedule_never_loads_high_and_has_no_file_leakage() -> Result<()> {
    let fixture = Fixture::new()?;
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_eq!(prepared.tier_ends, Some([8, 16, 24]));
    assert_eq!(
        (0..6).map(|e| prepared.epoch_len(e)).collect::<Vec<_>>(),
        [8, 10, 11, 11, 11, 11]
    );
    assert_eq!(prepared.batch_plan(5, 3), [3, 4, 4, 4, 4]);
    assert_eq!(prepared.validation.len(), 12);
    let training_files: HashSet<_> = prepared.training.iter().map(|s| &s.pair.0).collect();
    assert!(prepared.validation.iter().all(|s| !training_files.contains(&s.pair.0)));
    for epoch in 0..5 {
        let texts = prepared
            .epoch_indices(epoch)
            .into_iter()
            .map(|i| fixture.tokenizer.decode(&prepared.training[i].target_labels))
            .collect::<Vec<_>>();
        let count = |name: &str| texts.iter().filter(|s| s.contains(name)).count();
        assert_eq!(count("fn low("), [8, 2, 1, 1, 1][epoch]);
        assert_eq!(count("fn mild("), [0, 8, 2, 2, 2][epoch]);
        assert_eq!(count("fn moderate("), if epoch >= 2 { 8 } else { 0 });
    }
    let repeated = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_eq!(prepared.identity, repeated.identity);
    assert_eq!(format!("{:?}", prepared.training), format!("{:?}", repeated.training));
    Ok(())
}

#[test]
fn documented_and_undocumented_are_balanced_in_every_tier_and_partition() -> Result<()> {
    let fixture = Fixture::new()?;
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;
    let ends = prepared.tier_ends.unwrap();
    for samples in [
        &prepared.validation[..],
        &prepared.training[..ends[0]],
        &prepared.training[ends[0]..ends[1]],
        &prepared.training[ends[1]..],
    ] {
        let texts = decoded(&fixture, samples);
        let documented = texts.iter().filter(|t| t.contains("/// Returns a value.")).count();
        assert!(documented > 0 && documented * 2 == samples.len(), "{documented} of {}", samples.len());
        assert!(samples.iter().all(|s| s.input_ids.len() == 128));
    }
    Ok(())
}

#[test]
fn damaged_inputs_fail_and_an_interrupted_tail_is_ignored() -> Result<()> {
    let fixture = Fixture::new()?;
    let path = fixture.tier_file(Kind::Documented, 1);
    let original = std::fs::read(&path)?;
    let before = prepare(&fixture.config, &fixture.tokenizer)?;
    // Half an appended record is dropped, as after an interrupted write.
    let mut interrupted = original.clone();
    interrupted.extend_from_slice(&[200, 0, 0, 0, b'x']);
    std::fs::write(&path, interrupted)?;
    assert_eq!(prepare(&fixture.config, &fixture.tokenizer)?.identity, before.identity);
    std::fs::write(&path, b"corrupt")?;
    assert!(prepare(&fixture.config, &fixture.tokenizer).is_err());
    std::fs::write(&path, &original)?;
    let mut changed = fixture.config.clone();
    changed.validation_fraction = 0.5;
    assert_ne!(before.identity, prepare(&changed, &fixture.tokenizer)?.identity);
    let mut tiny = fixture.config.clone();
    tiny.max_seq_len = 2;
    assert!(prepare(&tiny, &fixture.tokenizer).is_err());
    Ok(())
}

#[test]
fn samples_too_long_for_the_context_are_skipped_individually() -> Result<()> {
    let fixture = Fixture::new()?;
    let base = prepare(&fixture.config, &fixture.tokenizer)?;
    // Just fits the undocumented moderate function, but not its documented twin.
    let TokenizerKind::Bpe(bpe) = &fixture.tokenizer else { unreachable!() };
    let tokens = bpe.inner.encode(SOURCES[2], false).unwrap().get_ids().len();
    let mut config = fixture.config.clone();
    config.max_seq_len = tokens + 2;
    let smaller = prepare(&config, &fixture.tokenizer)?;
    assert!(smaller.training.len() + smaller.validation.len() < base.training.len() + base.validation.len());
    assert!(smaller.training.iter().any(|s| fixture.tokenizer.decode(&s.target_labels).contains("fn moderate(")));
    assert!(!smaller.training.iter().any(|s| {
        let text = fixture.tokenizer.decode(&s.target_labels);
        text.contains("fn moderate(") && text.contains("///")
    }));
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

#[test]
fn tokenizer_is_fit_on_the_training_pool_and_only_tokenizer_json_is_written() -> Result<()> {
    let fixture = Fixture::new()?;
    let output = fixture.directory.join("fitted");
    code_corpus::train_tokenizer_from_data(&fixture.config, output.to_str().unwrap(), false)?;
    let files: Vec<_> = std::fs::read_dir(&output)?
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files, ["tokenizer.json"]);
    let tokenizer = code_corpus::load_tokenizer(output.to_str().unwrap())?;
    // Data is stored as text, so the new tokenizer loads the same samples unchanged.
    let mut config = fixture.config.clone();
    config.tokenizer = output.to_str().unwrap().into();
    let before = prepare(&fixture.config, &fixture.tokenizer)?;
    let after = prepare(&config, &tokenizer)?;
    assert_eq!(before.tier_ends, after.tier_ends);
    assert_eq!(before.validation.len(), after.validation.len());
    // An existing tokenizer is only replaced on request.
    assert!(code_corpus::train_tokenizer_from_data(&fixture.config, output.to_str().unwrap(), false).is_err());
    code_corpus::train_tokenizer_from_data(&fixture.config, output.to_str().unwrap(), true)?;
    Ok(())
}

#[test]
fn rust_docs_stage_trains_alone_first_then_replays_by_token_share() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let without = prepare(&fixture.config, &fixture.tokenizer)?;
    let html = fixture.directory.join("html");
    std::fs::create_dir_all(html.join("fixture"))?;
    std::fs::write(
        html.join("fixture/print.html"),
        crate::brain::code_docs::tests::PRINT_HTML,
    )?;
    let docs = fixture.directory.join("docs");
    crate::brain::code_docs::prepare(
        &fixture.config,
        &fixture.tokenizer,
        &html,
        "rustc test",
        &["fixture".into()],
        &docs,
        0,
    )?;
    fixture.config.docs = Some(docs);
    fixture.config.pretrain_share = 0.0;
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;
    // Context 128 splits the fixture chapters into several samples.
    let d = prepared.pretrain_len;
    assert!(d >= 2);
    assert_eq!(prepared.tier_ends, Some([d + 8, d + 16, d + 24]));
    // pretrain_share 0: the docs train once, then the code curriculum runs unchanged.
    assert_eq!(
        (0..6).map(|e| prepared.epoch_len(e)).collect::<Vec<_>>(),
        [d, 8, 10, 11, 11, 11]
    );
    let mut expected = vec![d + 2, d + 3];
    expected.extend(d + 8..d + 16);
    assert_eq!(prepared.epoch_indices(2), expected);
    assert_eq!(prepared.batch_plan(3, 3), [d.div_ceil(3), 3, 4]);
    assert!(prepared.training[..d].iter().all(|s| s.pair.0 == "rust-docs/fixture"));
    assert_eq!(
        format!("{:?}", &prepared.training[d..]),
        format!("{:?}", without.training)
    );
    assert_eq!(
        format!("{:?}", prepared.validation),
        format!("{:?}", without.validation)
    );
    assert_eq!(prepared.identity["pretrain"]["toolchain"], "rustc test");
    assert_eq!(prepared.identity["mode"], without.identity["mode"]);

    // A docs share rotates through the docs, keeps every code sample, and
    // takes roughly the requested share of supervised tokens.
    fixture.config.pretrain_share = 0.5;
    let rotating = prepare(&fixture.config, &fixture.tokenizer)?;
    assert_ne!(rotating.identity, prepared.identity);
    let tokens = |i: usize| rotating.training[i].target_labels.iter().filter(|&&t| t != PAD_TOKEN).count();
    let mut seen = HashSet::<usize>::new();
    let mut next = 0;
    for epoch in 1..8 {
        let (docs, code): (Vec<usize>, Vec<usize>) =
            rotating.epoch_indices(epoch).into_iter().partition(|&i| i < d);
        let mut expected = prepared.code_indices(epoch - 1);
        expected.sort();
        let mut code_sorted = code.clone();
        code_sorted.sort();
        assert_eq!(code_sorted, expected);
        let code_tokens: usize = code.iter().map(|&i| tokens(i)).sum();
        let docs_tokens: usize = docs.iter().map(|&i| tokens(i)).sum();
        assert!(docs_tokens >= code_tokens || docs.len() == d);
        assert!(docs_tokens - tokens(*docs.last().unwrap()) < code_tokens);
        assert_eq!(docs[0], next);
        next = (docs.last().unwrap() + 1) % d;
        seen.extend(docs);
    }
    assert_eq!(seen.len(), d);
    Ok(())
}

#[test]
fn code_pretraining_trains_all_of_code_bin_first_without_held_out_files() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let without = prepare(&fixture.config, &fixture.tokenizer)?;
    // code.bin has every curriculum file plus code the tiers never stored.
    let record = |path: &str, text: &str| Record { path: path.into(), location: "line 1".into(), text: text.into() };
    let mut records: Vec<_> = (0..6).flat_map(|f| SOURCES.map(|s| record(&format!("file-{f}.rs"), s))).collect();
    records.push(record("high.rs", "fn high() { loop { match 1 { _ => break } } }"));
    let samples: Vec<_> = records.iter()
        .map(|r| code_data::to_sample(r, &fixture.tokenizer, fixture.config.max_seq_len).map(Option::unwrap))
        .collect::<Result<_>>()?;
    let cache = fixture.directory.join("code.bin");
    sample_cache::write_cache_with_objective(&cache, &fixture.tokenizer, TrainingStage::Language,
        fixture.config.max_seq_len, fixture.config.seed, &samples, CacheObjective::RawCode)?;
    fixture.config.cache = cache;
    fixture.config.pretrain_code = true;
    fixture.config.pretrain_epochs = 2;
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;

    let held_out: HashSet<_> = without.validation.iter().map(|s| s.pair.0.clone()).collect();
    let p = prepared.pretrain_len;
    assert_eq!(p, samples.iter().filter(|s| !held_out.contains(&s.pair.0)).count());
    assert!(prepared.training[..p].iter().all(|s| !held_out.contains(&s.pair.0)));
    assert!(prepared.training[..p].iter().any(|s| s.pair.0 == "high.rs"));
    assert_eq!(prepared.identity["pretrain"]["held_out_dropped"], samples.len() - p);
    // Two pretraining epochs, then the unchanged curriculum.
    assert_eq!(prepared.epoch_indices(0), (0..p).collect::<Vec<_>>());
    assert_eq!(prepared.epoch_indices(1), (0..p).collect::<Vec<_>>());
    assert_eq!(prepared.epoch_indices(2), without.epoch_indices(0).iter().map(|i| i + p).collect::<Vec<_>>());
    assert_eq!(prepared.epoch_indices(4), without.epoch_indices(2).iter().map(|i| i + p).collect::<Vec<_>>());
    assert_eq!(format!("{:?}", &prepared.training[p..]), format!("{:?}", without.training));
    assert_eq!(format!("{:?}", prepared.validation), format!("{:?}", without.validation));

    fixture.config.docs = Some(fixture.directory.join("docs"));
    assert!(fixture.config.validate().is_err());
    Ok(())
}

#[test]
fn code_pretraining_reads_a_longer_pretrain_cache_at_the_config_context() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let record = |path: &str, text: String| Record { path: path.into(), location: "line 1".into(), text };
    // An item longer than the short cache's context, as the old 512 code.bin skipped.
    let long = format!("fn long() {{ {} }}", "let x = 1; ".repeat(5));
    let write = |name: &str, context: usize, records: &[Record]| -> Result<PathBuf> {
        let samples: Vec<_> = records.iter()
            .filter_map(|r| code_data::to_sample(r, &fixture.tokenizer, context).transpose())
            .collect::<Result<_>>()?;
        let path = fixture.directory.join(name);
        sample_cache::write_cache_with_objective(&path, &fixture.tokenizer, TrainingStage::Language,
            context, fixture.config.seed, &samples, CacheObjective::RawCode)?;
        Ok(path)
    };
    let mut records: Vec<_> = (0..6).map(|f| record(&format!("file-{f}.rs"), SOURCES[0].into())).collect();
    records.push(record("long.rs", long.clone()));
    assert!(code_data::to_sample(&records[6], &fixture.tokenizer, 32)?.is_none());
    fixture.config.cache = write("code.bin", 32, &records)?;
    fixture.config.pretrain_code = true;

    // The short cache cannot serve a longer context.
    let error = prepare(&fixture.config, &fixture.tokenizer).err().expect("context mismatch");
    assert!(format!("{error:#}").contains("pretrain_cache"), "{error:#}");

    fixture.config.pretrain_cache = Some(write("code_long.bin", fixture.config.max_seq_len, &records)?);
    let prepared = prepare(&fixture.config, &fixture.tokenizer)?;
    let p = prepared.pretrain_len;
    assert!(prepared.training[..p].iter().any(|s| s.pair.0 == "long.rs"));
    assert!(prepared.training.iter().all(|s| s.input_ids.len() == fixture.config.max_seq_len));
    assert_eq!(prepared.identity["pretrain"]["source"], "code_long.bin");

    fixture.config.pretrain_code = false;
    assert!(fixture.config.validate().is_err());
    Ok(())
}
