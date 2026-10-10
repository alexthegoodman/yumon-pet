use super::*;
fn record(source: &str) -> Record {
    analyze("sample", "rust", source, "src/test.rs", "line 1")
}
fn reference() -> Reference {
    let sources = [
        "fn a() {}",
        "fn b() { let x = 1; }",
        "fn c(x: bool) { if x { return; } }",
        "fn d(x: bool) { while x { if x { break; } } }",
    ];
    Reference::fit(
        &sources.map(record),
        "test-1",
        "0".repeat(64),
        DEFAULT_WEIGHTS,
        Preprocessing::default(),
    )
    .unwrap()
}

#[test]
fn extraction_counts_control_flow_mutations_and_calls() {
    let result =
        record("fn f(x: bool) { let mut y = 0; while x { if x && y < 2 { y += 1; foo(y); } } }");
    assert_eq!(result.analysis_status, "success");
    assert_eq!(result.metrics.cyclomatic_complexity, Some(4.0));
    assert_eq!(result.metrics.cognitive_complexity, Some(4.0));
    assert_eq!(result.metrics.control_flow_depth, Some(2.0));
    assert_eq!(result.metrics.state_mutation_complexity, Some(2.0));
    assert_eq!(result.metrics.dependency_count, Some(1.0));
    assert!(result.metrics.halstead_volume.unwrap() > 0.0);
    assert_eq!(
        result,
        record("fn f(x: bool) { let mut y = 0; while x { if x && y < 2 { y += 1; foo(y); } } }")
    );
}

#[test]
fn methods_structs_errors_and_opaque_macros_are_explicit() {
    assert_eq!(
        record("impl X { fn a(&self) { self.foo(); } }")
            .metrics
            .dependency_count,
        Some(1.0)
    );
    let structure = record("struct X { field: usize }");
    assert_eq!(structure.analysis_unit, "item");
    assert_eq!(structure.metrics.cyclomatic_complexity, Some(0.0));
    assert_eq!(record("fn broken() {").analysis_status, "parse_error");
    assert!(record("fn broken() {").metrics.primary().is_none());
    let opaque = record("fn f() { some_macro!(if true { return; }); }");
    assert_eq!(opaque.analysis_status, "missing_metrics");
    assert!(opaque.metrics.primary().is_none());
    assert!(opaque.metrics.token_count.is_some());
    assert_eq!(
        analyze("x", "python", "def f(): pass", "", "").analysis_status,
        "unsupported"
    );
}

#[test]
fn midrank_ties_extremes_and_fixed_boundary_classification() {
    assert_eq!(percentile(&[1.0, 2.0, 2.0, 4.0], 2.0), 0.5);
    assert_eq!(percentile(&[1.0, 2.0], 0.0), 0.0);
    assert_eq!(percentile(&[1.0, 2.0], 3.0), 1.0);
    assert_eq!(tier(0.5, &[0.5, 0.5, 0.5]), 1);
    assert_eq!(tier(0.6, &[0.5, 0.5, 0.5]), 4);
    let reference = reference();
    let reloaded: Reference =
        serde_json::from_str(&serde_json::to_string(&reference).unwrap()).unwrap();
    assert_eq!(reference.populations, reloaded.populations);
    assert_eq!(reference.boundaries, reloaded.boundaries);
    let mut first = record("fn a() {}");
    let mut second = first.clone();
    reference.classify(&mut first, "sha");
    reference.classify(&mut second, "sha");
    assert_eq!(first, second);
    let before = serde_json::to_string(&reference).unwrap();
    let mut extra = record("fn more() { loop { if true { break; } } }");
    reference.classify(&mut extra, "sha");
    assert_eq!(before, serde_json::to_string(&reference).unwrap());
    assert!((1..=4).contains(&extra.complexity_tier.unwrap()));
}

#[test]
fn contextual_features_do_not_enter_score_and_missing_metrics_do_not_score() {
    let reference = reference();
    let mut first = record("fn f() {}");
    let mut second = first.clone();
    second.metrics.token_count = Some(100000.0);
    second.metrics.dependency_count = Some(1000.0);
    reference.classify(&mut first, "sha");
    reference.classify(&mut second, "sha");
    assert_eq!(
        first.structural_complexity_score,
        second.structural_complexity_score
    );
    let mut broken = record("fn x() {");
    reference.classify(&mut broken, "sha");
    assert!(broken.complexity_tier.is_none());
    assert!(broken.scoring_version.is_some());
}

#[test]
fn invalid_references_and_weights_are_rejected() {
    let mut reference = reference();
    reference.populations[0].clear();
    assert!(reference.validate().is_err());
    assert!(
        Reference::fit(
            &[record("fn a() {}")],
            "x",
            "0".repeat(64),
            [0.0; 5],
            Preprocessing::default()
        )
        .is_err()
    );
    assert!(
        Reference::fit(
            &[record("broken")],
            "x",
            "0".repeat(64),
            DEFAULT_WEIGHTS,
            Preprocessing::default()
        )
        .is_err()
    );
    let mut reference = super::tests::reference();
    reference.boundaries = [0.8, 0.2, 0.5];
    assert!(reference.validate().is_err());
}

#[test]
fn preprocessing_and_whitespace_stability() {
    let policy = Preprocessing::default();
    assert!(policy.excluded("fn f() {}", "project/vendor/a.rs"));
    assert!(policy.excluded("// @generated\nfn f() {}", "a.rs"));
    assert!(!policy.excluded("fn f() {}", "vendor_tools/a.rs"));
    assert_eq!(
        record("fn f() { if true { return; } }").metrics,
        record("\n// comment\nfn f ( ) {\n if true { return ; }\n}\n").metrics
    );
    assert_eq!(correlation(&[1.0, 2.0], &[2.0, 4.0]), Some(1.0));
    assert_eq!(correlation(&[1.0, 1.0], &[2.0, 4.0]), None);
}

#[test]
fn bucket_caches_preserve_original_training_tensors_and_provenance() {
    use crate::brain::{code_corpus, sample_cache, samples::TrainingStage};
    let directory = std::env::temp_dir().join(format!("yumon-buckets-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(directory.clone());
    let source = "fn a() {}\nfn b(x: bool) { if x { return; } }\nfn c() { opaque!(); }";
    let source_path = directory.join("input.rs");
    std::fs::write(&source_path, source).unwrap();
    let tokenizer_path = directory.join("tokenizer");
    code_corpus::train_tokenizer(&[source_path], 300, tokenizer_path.to_str().unwrap()).unwrap();
    let tokenizer = code_corpus::load_tokenizer(tokenizer_path.to_str().unwrap()).unwrap();
    let samples = code_corpus::chunk_source(source, &tokenizer, 256, "input.rs").unwrap();
    let mut records: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            analyze(
                &i.to_string(),
                "rust",
                &tokenizer.decode(&s.target_labels),
                &s.pair.0,
                &s.pair.1,
            )
        })
        .collect();
    let reference = Reference::fit(
        &records,
        "test",
        "0".repeat(64),
        DEFAULT_WEIGHTS,
        Preprocessing::default(),
    )
    .unwrap();
    for r in &mut records {
        reference.classify(r, "sha");
    }
    assert_eq!(
        records
            .iter()
            .filter(|r| r.complexity_tier.is_some())
            .count(),
        2
    );
    let mut seen = 0;
    for destination in 0..=4 {
        let group: Vec<_> = samples
            .iter()
            .zip(&records)
            .filter(|(_, r)| r.complexity_tier.unwrap_or(0) == destination)
            .map(|(s, _)| s.clone())
            .collect();
        if group.is_empty() {
            continue;
        }
        let path = directory.join(format!("tier-{destination}.bin"));
        sample_cache::write_cache_with_objective(
            &path,
            &tokenizer,
            TrainingStage::Language,
            256,
            42,
            &group,
            sample_cache::CacheObjective::RawCode,
        )
        .unwrap();
        let loaded = sample_cache::load_cache_with_objective(
            &path,
            &tokenizer,
            TrainingStage::Language,
            256,
            sample_cache::CacheObjective::RawCode,
        )
        .unwrap();
        assert_eq!(format!("{group:?}"), format!("{loaded:?}"));
        seen += loaded.len();
    }
    assert_eq!(seen, samples.len());
}
