//! CPU-only acceptance scenarios; never initializes a model or GPU.
use std::{collections::HashSet, path::PathBuf};
use crate::brain::{
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    code_corpus::{self, CodeConfig},
    sample_cache::{self, CacheObjective}, samples::TrainingStage,
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("yumon-code-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(path.join("nested")).unwrap();
        std::fs::create_dir_all(path.join("target")).unwrap();
        std::fs::write(path.join("one.rs"), "pub struct MyType {\n    pub Name: String,\n}\n\nfn main() { println!(\"Hello 世界 <PAD>\"); }\n").unwrap();
        std::fs::write(path.join("nested/two.rs"), "// OtherFile\r\nfn Second() {\r\n\tlet s = \"🦀 Rust\";\r\n}\r\n").unwrap();
        std::fs::write(path.join("target/ignored.rs"), "ignored").unwrap();
        std::fs::write(path.join("ignored.txt"), "ignored").unwrap();
        Self(path)
    }
    fn config(&self) -> CodeConfig {
        CodeConfig { source: self.0.clone(), tokenizer: self.0.join("tokenizer").to_str().unwrap().into(),
            cache: self.0.join("code.bin"), vocab_size: 300, max_seq_len: 512, ..Default::default() }
    }
}
impl Drop for Fixture { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

#[test]
fn recursive_rust_cache_round_trip_and_file_held_out_split() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let files = code_corpus::rust_files(&config).unwrap();
    assert_eq!(files.len(), 2);
    code_corpus::train_tokenizer(&files, config.vocab_size, &config.tokenizer).unwrap();
    let tokenizer = code_corpus::load_tokenizer(&config.tokenizer).unwrap();
    let samples = code_corpus::prepare(&config, &files, &tokenizer, None).unwrap();
    let repeated = code_corpus::prepare(&config, &files, &tokenizer, None).unwrap();
    assert_eq!(samples.iter().map(|s| &s.input_ids).collect::<Vec<_>>(), repeated.iter().map(|s| &s.input_ids).collect::<Vec<_>>());
    let meta = sample_cache::write_cache_with_objective(&config.cache, &tokenizer,
        TrainingStage::Language, config.max_seq_len, config.seed, &samples, CacheObjective::RawCode).unwrap();
    assert_eq!(meta.objective, CacheObjective::RawCode);
    assert!(meta.corpus_sha256.is_some());
    assert_eq!(meta.version, 2);
    assert_eq!(meta.preparation.as_deref(), Some("rust-items-v1"));
    let loaded = sample_cache::load_cache_with_objective(&config.cache, &tokenizer,
        TrainingStage::Language, config.max_seq_len, CacheObjective::RawCode).unwrap();
    assert_eq!(samples.iter().map(|s| &s.target_labels).collect::<Vec<_>>(), loaded.iter().map(|s| &s.target_labels).collect::<Vec<_>>());
    let (train, val) = code_corpus::split_validation(loaded, &config).unwrap();
    assert!(!train.is_empty() && !val.is_empty());
    let train_files: HashSet<_> = train.iter().map(|s| &s.pair.0).collect();
    assert!(val.iter().all(|s| !train_files.contains(&s.pair.0)));
    assert!(sample_cache::load_cache(&config.cache, &tokenizer, TrainingStage::Language, 32).is_err());
    assert!(sample_cache::load_cache_with_objective(&config.cache, &tokenizer,
        TrainingStage::Language, 64, CacheObjective::RawCode).is_err());
    assert!(sample_cache::write_cache_with_objective(&config.cache, &tokenizer,
        TrainingStage::Language, 32, config.seed, &samples, CacheObjective::RawCode).is_err());
    crate::brain::train::run_code(&config, true).unwrap();

    // A previous token-window cache must not silently keep training fragments.
    let old = std::fs::read(&config.cache).unwrap();
    let header_len = u32::from_le_bytes(old[8..12].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&old[12..12 + header_len]).unwrap();
    header.as_object_mut().unwrap().remove("preparation");
    let header = serde_json::to_vec(&header).unwrap();
    let mut outdated = old[..8].to_vec();
    outdated.extend((header.len() as u32).to_le_bytes());
    outdated.extend(header);
    outdated.extend_from_slice(&old[12 + header_len..]);
    std::fs::write(&config.cache, outdated).unwrap();
    let error = sample_cache::load_cache_with_objective(&config.cache, &tokenizer,
        TrainingStage::Language, config.max_seq_len, CacheObjective::RawCode).unwrap_err();
    assert!(error.to_string().contains("old chunking"));
}

#[test]
fn chunks_preserve_whole_items_and_skip_oversized_items() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let files = code_corpus::rust_files(&config).unwrap();
    code_corpus::train_tokenizer(&files, config.vocab_size, &config.tokenizer).unwrap();
    let tokenizer = code_corpus::load_tokenizer(&config.tokenizer).unwrap();
    for path in files {
        let text = std::fs::read_to_string(path).unwrap();
        let units = code_corpus::rust_units(&text).unwrap();
        for context in [2, 8, 32, 512] {
            let chunks = code_corpus::chunk_source(&text, &tokenizer, context, "test.rs").unwrap();
            let expected: Vec<_> = units.iter().filter(|(unit, _)| tokenizer.encode(unit).len() < context).collect();
            assert_eq!(chunks.len(), expected.len());
            for (chunk, (unit, _)) in chunks.iter().zip(expected) {
                let active = chunk.target_labels.iter().position(|&id| id == PAD_TOKEN).unwrap_or(context);
                assert_eq!(chunk.input_ids[0], BOS_TOKEN);
                assert_eq!(chunk.target_labels[active - 1], EOS_TOKEN);
                assert_eq!(chunk.input_ids[1..active], chunk.target_labels[..active - 1]);
                assert_eq!(chunk.input_ids.len(), context);
                assert_eq!(chunk.target_labels.len(), context);
                assert_eq!(tokenizer.decode(&chunk.target_labels), *unit);
                syn::parse_file(unit).unwrap();
            }
        }
        // Exactly fitting items are retained, but one token less skips the item.
        let unit = &units[0].0;
        let length = tokenizer.encode(unit).len();
        assert_eq!(code_corpus::chunk_source(unit, &tokenizer, length + 1, "test.rs").unwrap().len(), 1);
        assert!(code_corpus::chunk_source(unit, &tokenizer, length, "test.rs").unwrap().is_empty());
    }
    assert!(code_corpus::chunk_source("fn X() {}", &tokenizer, 1, "bad.rs").is_err());
    assert!(code_corpus::chunk_source("fn broken() {", &tokenizer, 512, "bad.rs").is_err());
}

#[test]
fn parser_keeps_attributes_nested_bodies_and_method_context() {
    let script = code_corpus::rust_units("\u{feff}#!/usr/bin/env rust-script\nfn main() {}\n").unwrap();
    assert_eq!(script, vec![("fn main() {}".into(), 2)]);
    let text = r###"
// Unicode before items must not shift byte spans: ?? ??
use std::fmt;
/// A documented type.
#[derive(Clone)]
pub struct Thing<T> { value: T }
pub struct Tuple(pub usize);
pub struct Unit;
mod nested {
    #[cfg(test)]
    pub async fn complete() {
        let raw = r#"} fn fake() {"#;
        // } misleading brace
        /* { nested /* } */ comment */
        if true { println!("{raw}"); }
    }
}
impl<T> Thing<T> where T: Clone {
    /// Keep this with its method.
    pub fn copy(&self) -> T { self.value.clone() }
    fn other() { let c = '}'; }
}
trait Example {
    fn abstract_method();
    fn concrete(&self) { if true {} }
}
macro_rules! ignored { () => { fn fake() {} }; }
"###;
    let units = code_corpus::rust_units(text).unwrap();
    assert_eq!(units.len(), 7);
    for (unit, _) in &units { syn::parse_file(unit).unwrap(); }
    assert!(units[0].0.starts_with("/// A documented type."));
    assert!(units[0].0.contains("#[derive(Clone)]"));
    assert!(units[3].0.starts_with("#[cfg(test)]"));
    assert!(units[3].0.ends_with("    }"));
    assert!(units[4].0.starts_with("impl<T> Thing<T> where T: Clone {"));
    assert!(units[4].0.contains("/// Keep this with its method."));
    assert!(!units[4].0.contains("fn other"));
    assert!(units[6].0.starts_with("trait Example {"));
    assert!(!units[6].0.contains("abstract_method"));
}

#[test]
fn duplicate_files_are_removed_and_bad_input_fails_explicitly() {
    let fixture = Fixture::new();
    let config = fixture.config();
    std::fs::copy(fixture.0.join("one.rs"), fixture.0.join("duplicate.rs")).unwrap();
    let files = code_corpus::rust_files(&config).unwrap();
    code_corpus::train_tokenizer(&files, config.vocab_size, &config.tokenizer).unwrap();
    let tokenizer = code_corpus::load_tokenizer(&config.tokenizer).unwrap();
    let samples = code_corpus::prepare(&config, &files, &tokenizer, None).unwrap();
    assert_eq!(samples.iter().map(|s| &s.pair.0).collect::<HashSet<_>>().len(), 2);
    assert!(code_corpus::prepare(&config, &files, &tokenizer, Some(0)).is_err());
    std::fs::write(fixture.0.join("one.rs"), [0xff]).unwrap();
    assert!(code_corpus::prepare(&config, &files, &tokenizer, None).is_err());
    assert!(code_corpus::train_tokenizer(&files, config.vocab_size, &config.tokenizer).is_err());
}

#[test]
fn configuration_rejects_invalid_dimensions_and_unknown_fields() {
    let mut config = CodeConfig::default();
    config.max_seq_len = 1;
    assert!(config.validate().is_err());
    config.max_seq_len = 512;
    config.attn_heads = 7;
    assert!(config.validate().is_err());
    assert!(serde_json::from_str::<CodeConfig>(r#"{"sequence_length": 1024}"#).is_err());
}
