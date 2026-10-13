//! CPU-only checks for the adaptive learning-value registry; no model or GPU.
use super::*;
use crate::brain::code_corpus::CodeConfig;

const PAD: i32 = PAD_TOKEN as i32;

fn config(mode: AdaptiveMode) -> AdaptiveLossConfig {
    AdaptiveLossConfig { mode, statistics_interval_steps: 2, progress_history_windows: 2,
        confidence_kappa: 4.0, difficulty_beta: 0.5, ..Default::default() }
}

fn observe(stats: &mut AdaptiveLoss, targets: &[i32], losses: &[f32]) -> bool {
    let keys = stats.keys(targets, None);
    stats.observe(&keys, losses)
}

/// One two-step window: the observations, then a PAD-only step that closes it.
fn window(stats: &mut AdaptiveLoss, observations: &[(i32, f32)]) {
    let (targets, losses): (Vec<i32>, Vec<f32>) = observations.iter().copied().unzip();
    assert!(!observe(stats, &targets, &losses));
    assert!(observe(stats, &[PAD], &[123.0]));
}

#[test]
fn config_section_is_optional_validated_and_defaults_match_the_design() {
    let old: CodeConfig = serde_json::from_str("{}").unwrap();
    assert!(old.adaptive_loss.is_none());
    let parsed: CodeConfig = serde_json::from_str(r#"{"adaptive_loss": {"mode": "difficulty", "lambda": 0.2}}"#).unwrap();
    let adaptive = parsed.adaptive_loss.unwrap();
    assert_eq!((adaptive.mode, adaptive.lambda, adaptive.statistics_interval_steps), (AdaptiveMode::Difficulty, 0.2, 100));
    assert_eq!((adaptive.min_weight, adaptive.max_weight, adaptive.difficulty_beta), (0.5, 2.0, 0.99));
    assert!(serde_json::from_str::<CodeConfig>(r#"{"adaptive_loss": {"bogus": 1}}"#).is_err());
    for bad in [r#"{"lambda": 1.5}"#, r#"{"min_weight": 0.0}"#, r#"{"max_weight": 0.9}"#,
        r#"{"difficulty_beta": 1.0}"#, r#"{"statistics_interval_steps": 0}"#, r#"{"warmup_fraction": 0.7, "ramp_fraction": 0.5}"#] {
        let cfg: AdaptiveLossConfig = serde_json::from_str(bad).unwrap();
        assert!(cfg.validate().is_err(), "{bad} should be rejected");
    }
    let disabled: AdaptiveLossConfig = serde_json::from_str(r#"{"enabled": false}"#).unwrap();
    assert_eq!(disabled.effective_mode(), AdaptiveMode::Disabled);
    let testing = CodeConfig::load(std::path::Path::new("configs/yumon-code-moe-mini-adaptive.json")).unwrap()
        .adaptive_loss.unwrap();
    assert_eq!((testing.mode, testing.warmup_fraction), (AdaptiveMode::Context, 0.0));
}

#[test]
fn warm_up_and_ramp_schedule_lambda() {
    let cfg = AdaptiveLossConfig::default();
    assert_eq!(cfg.lambda_at(0, 1000), 0.0);
    assert_eq!(cfg.lambda_at(99, 1000), 0.0);
    assert!((cfg.lambda_at(150, 1000) - 0.15).abs() < 1e-12);
    assert!((cfg.lambda_at(200, 1000) - 0.3).abs() < 1e-12);
    assert!((cfg.lambda_at(999, 1000) - 0.3).abs() < 1e-12);
    let disabled = AdaptiveLossConfig { mode: AdaptiveMode::Disabled, ..cfg.clone() };
    assert_eq!(disabled.lambda_at(999, 1000), 0.0);
    let immediate = AdaptiveLossConfig { warmup_fraction: 0.0, ramp_fraction: 0.0, ..cfg };
    assert_eq!(immediate.lambda_at(0, 1000), 0.3);
}

#[test]
fn difficulty_ema_updates_per_window_ignores_pad_and_keeps_unseen_tokens() {
    let mut stats = AdaptiveLoss::new(config(AdaptiveMode::Difficulty), 16);
    // Weights stay neutral until a window completes.
    assert!(stats.weights().iter().all(|&w| w == 1.0));
    window(&mut stats, &[(5, 2.0), (5, 4.0), (6, 1.0), (PAD, 99.0), (7, f32::NAN), (40, 9.0)]);
    // First window initializes D to the window mean (no zero-start bias).
    let r = 2.0 / (2.0 + 4.0);
    assert!((stats.smoothed_difficulty(5) - (r * 3.0 + (1.0 - r) * (7.0 / 3.0))).abs() < 1e-6);
    window(&mut stats, &[(6, 3.0)]);
    // Token 5 unseen this window: D stays 3.0; its confidence decays with its count.
    let global = 0.5 * (7.0 / 3.0) + 0.5 * 3.0;
    let r5 = 1.0 / (1.0 + 4.0);
    assert!((stats.smoothed_difficulty(5) - (r5 * 3.0 + (1.0 - r5) * global)).abs() < 1e-6);
    // Token 6: 0.5 * 1.0 + 0.5 * 3.0.
    let r6 = 1.5 / (1.5 + 4.0);
    assert!((stats.smoothed_difficulty(6) - (r6 * 2.0 + (1.0 - r6) * global)).abs() < 1e-6);
    // Never-seen and NaN-only tokens fall back to D_global.
    assert!((stats.smoothed_difficulty(9) - global).abs() < 1e-6);
    assert!((stats.smoothed_difficulty(7) - global).abs() < 1e-6);
    assert_eq!(stats.windows, 2);
}

#[test]
fn progress_compares_against_the_oldest_retained_snapshot() {
    let mut stats = AdaptiveLoss::new(config(AdaptiveMode::DifficultyProgress), 16);
    window(&mut stats, &[(5, 4.0), (6, 1.0)]);
    assert_eq!(stats.progress(5), 0.0);
    window(&mut stats, &[(5, 2.0), (6, 3.0)]);
    assert!((stats.progress(5) - 1.0).abs() < 1e-6); // 4 -> 3
    assert!((stats.progress(6) + 1.0).abs() < 1e-6); // 1 -> 2, negative
    window(&mut stats, &[(5, 2.0)]);
    assert!((stats.progress(5) - 1.5).abs() < 1e-6); // 4 -> 2.5 over h = 2 windows
    window(&mut stats, &[(5, 2.5)]);
    // The window-1 snapshot (4.0) has rolled out; compare with window 2 (3.0).
    assert!((stats.progress(5) - 0.5).abs() < 1e-6, "{}", stats.progress(5));
    // A token first seen after the oldest snapshot has no progress estimate.
    window(&mut stats, &[(8, 5.0)]);
    assert_eq!(stats.progress(8), 0.0);
}

#[test]
fn weights_are_bounded_favor_learnable_difficulty_and_rows_mix_lambda() {
    let mut stats = AdaptiveLoss::new(AdaptiveLossConfig { confidence_kappa: 1.0, ..config(AdaptiveMode::DifficultyProgress) }, 16);
    let many = |t: i32, l: f32| std::iter::repeat((t, l)).take(50).collect::<Vec<_>>();
    // 5: hard and improving. 6: easy and flat. 7: hard and flat.
    window(&mut stats, &[many(5, 6.0), many(6, 0.2), many(7, 6.0)].concat());
    window(&mut stats, &[many(5, 3.0), many(6, 0.2), many(7, 6.0)].concat());
    let w = stats.weights();
    assert!(w.iter().all(|&x| (0.5..=2.0).contains(&x) && x.is_finite()));
    assert!(w[5] > w[7] && w[7] > w[6], "{:?}", &w[5..8]);
    assert!(w[6] == 0.5, "easy flat token should hit the floor: {}", w[6]);
    let keys = stats.keys(&[5, PAD, 6, 999, -3], None);
    assert_eq!(&keys[1..], &[NO_KEY, 6, NO_KEY, NO_KEY]);
    let rows = stats.row_weights(&keys, 0.3);
    assert_eq!((rows[1], rows[3], rows[4]), (0.0, 0.0, 0.0));
    assert!((rows[0] - (0.7 + 0.3 * w[5])).abs() < 1e-6);
    assert!((rows[2] - (0.7 + 0.3 * 0.5)).abs() < 1e-6);
    // Unseen tokens score near neutral rather than at the floor.
    assert!((stats.weights()[12] - 1.0).abs() < 0.5, "{}", stats.weights()[12]);
    let report = stats.report(&FrequencyBuckets::from_counts(&[0; 16]));
    assert_eq!(report.observed_keys, 3);
    assert!(report.weight_at_min_fraction > 0.0);
}

#[test]
fn difficulty_and_frequency_modes_rank_as_expected() {
    let many = |t: i32, l: f32, n: usize| std::iter::repeat((t, l)).take(n).collect::<Vec<_>>();
    let mut difficulty = AdaptiveLoss::new(config(AdaptiveMode::Difficulty), 16);
    let mut frequency = AdaptiveLoss::new(config(AdaptiveMode::Frequency), 16);
    for stats in [&mut difficulty, &mut frequency] {
        window(stats, &[many(5, 4.0, 200), many(6, 0.5, 5)].concat());
    }
    assert!(difficulty.weights()[5] > difficulty.weights()[6]);
    assert!(frequency.weights()[6] > frequency.weights()[5]);
}

/// Labeler over a tiny hand-made vocabulary: one token per piece of text.
fn labeler() -> (ContextLabeler, impl Fn(&str) -> i32) {
    let vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "///", " Returns", " the", " glyphs", "\n",
        "fn", " glyphs", "(", ")", " {", " \"", "//", " not", "\"", "}", "/*", "*/", "'\"'", "\\\""];
    let text: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
    let lookup = text.clone();
    let id = move |piece: &str| lookup.iter().position(|t| t == piece).unwrap() as i32;
    (ContextLabeler::new(text), id)
}

#[test]
fn context_labels_split_comments_code_symbols_and_copies() {
    let (labeler, id) = labeler();
    let (sym, novel, copy, c_novel, c_copy) = (0u8, 1, 2, 3, 4);
    // "/// Returns the glyphs\nfn glyphs() {" then PAD padding to seq_len 12.
    let row = [id("///"), id(" Returns"), id(" the"), id(" glyphs"), id("\n"), id("fn"), 10, id("("), id(")"), id(" {"), PAD, PAD];
    let labels = labeler.label(&row, 12);
    // The marker itself is a code symbol; the newline ends the comment. Token 10
    // is " glyphs" spelled by a different id, so it is novel in code; token 7 is
    // " glyphs" first seen in the comment.
    assert_eq!(labels, vec![sym, c_novel, c_novel, c_novel, c_novel, novel, novel, sym, sym, sym, NO_LABEL, NO_LABEL]);

    // `//` inside a string literal is not a comment; the escaped quote stays inside it.
    let row = [id("fn"), id(" \""), id("\\\""), id("//"), id(" not"), id("\""), id(" the"), id("fn"), PAD];
    assert_eq!(labeler.label(&row, 9), vec![novel, sym, sym, sym, novel, sym, novel, copy, NO_LABEL]);
    // A `'"'` char literal does not open a string, so the `//` after it is a comment.
    let row = [id("'\"'"), id("//"), id(" the"), id(" the"), id("\n"), id(" the")];
    assert_eq!(labeler.label(&row, 6), vec![sym, sym, c_novel, c_copy, c_novel, copy]);
    // Block comments close on `*/`; each row starts in code with fresh copies.
    let row = [id("/*"), id(" the"), id("*/"), id(" the"), id(" the"), id(" Returns")];
    assert_eq!(labeler.label(&row, 3), vec![sym, c_novel, c_novel, novel, copy, novel]);
}

#[test]
fn context_mode_keys_by_category_with_category_priors() {
    let (labeler, id) = labeler();
    let cfg = AdaptiveLossConfig { confidence_kappa: 1.0, ..config(AdaptiveMode::Context) };
    let mut stats = AdaptiveLoss::new(cfg, 32);
    assert!(stats.uses_context());
    // The same token " the": copied in code (easy), novel in a comment (hard).
    let row = [id("fn"), id(" the"), id(" the"), id("//"), id(" Returns"), id(" the")];
    let labels = labeler.label(&row, row.len());
    let keys = stats.keys(&row, Some(&labels));
    let the = id(" the") as u32;
    assert_eq!((keys[1], keys[2], keys[5]), (32 + the, 2 * 32 + the, 4 * 32 + the));
    let losses = [1.0, 6.0, 0.5, 0.2, 7.0, 5.0];
    for _ in 0..3 {
        assert!(!stats.observe(&keys, &losses));
        assert!(stats.observe(&[NO_KEY], &[0.0]));
    }
    let w = stats.weights();
    assert!(w[(32 + the) as usize] > w[(2 * 32 + the) as usize], "novel vs copied ' the'");
    // An unseen token in a category falls back to that category's prior, not a global one.
    let unseen = id(" glyphs") as usize;
    assert!((stats.smoothed_difficulty(4 * 32 + unseen) - 5.0).abs() < 1e-5);
    assert!((stats.smoothed_difficulty(2 * 32 + unseen) - 0.5).abs() < 1e-5);
}

#[test]
fn checkpoint_state_round_trips_and_rejects_config_changes() {
    let dir = std::env::temp_dir().join(format!("yumon-adaptive-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = config(AdaptiveMode::DifficultyProgress);
    // Checkpoints from before adaptive loss start fresh.
    let fresh = AdaptiveLoss::load_or_new(&dir, &cfg, 16).unwrap();
    assert_eq!(fresh.windows, 0);
    let mut stats = AdaptiveLoss::new(cfg.clone(), 16);
    window(&mut stats, &[(5, 4.0), (6, 1.0)]);
    window(&mut stats, &[(5, 2.0)]);
    // A pending half-window must survive the round trip too.
    assert!(!observe(&mut stats, &[5], &[1.0]));
    stats.save(&dir).unwrap();
    let mut restored = AdaptiveLoss::load_or_new(&dir, &cfg, 16).unwrap();
    assert_eq!(restored.windows, 2);
    assert_eq!(restored.weights(), stats.weights());
    assert!(observe(&mut restored, &[PAD], &[0.0]));
    assert!(observe(&mut stats, &[PAD], &[0.0]));
    assert_eq!(restored.weights(), stats.weights());
    assert_eq!(restored.progress(5), stats.progress(5));
    let changed = AdaptiveLossConfig { lambda: 0.1, ..cfg.clone() };
    assert!(AdaptiveLoss::load_or_new(&dir, &changed, 16).is_err());
    assert!(AdaptiveLoss::load_or_new(&dir, &cfg, 32).is_err());
    // An unreadable older state format restarts the statistics instead of failing.
    std::fs::write(dir.join(STATE_FILE), b"old").unwrap();
    assert_eq!(AdaptiveLoss::load_or_new(&dir, &cfg, 16).unwrap().windows, 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn group_losses_split_validation_by_frequency_and_context() {
    let buckets = FrequencyBuckets::from_counts(&[0, 5, 50, 500, 5000, 50000]);
    let mut loss = GroupLoss::default();
    loss.add_by_frequency(&buckets, &[1, 5, 5, PAD, 2], &[1.0, 2.0, 4.0, 9.0, f32::INFINITY]);
    assert_eq!(loss.0[0], (1.0, 1.0, 1));
    assert_eq!(loss.0[4], (6.0, 2.0, 2));
    assert_eq!(loss.0[1], (0.0, 0.0, 0));
    assert_eq!(loss.to_json(&FrequencyBuckets::LABELS, false)[">=10k"]["ce"], 3.0);
    let mut context = GroupLoss::default();
    context.add_by_label(&[2, NO_LABEL, 2, 0], &[1.0, 9.0, 3.0, 0.5], Some(&[1.5, 0.0, 0.5, 1.0]));
    let json = context.to_json(&CONTEXT_LABELS, true);
    assert_eq!(json["code_copy"]["ce"], 2.0);
    assert_eq!(json["code_copy"]["mean_weight"], 1.0);
    assert_eq!(json["code_symbol"]["tokens"], 1);
}

/// Prints real-tokenizer labels for held-out snippets. Run with:
/// cargo test --release --lib real_tokenizer_context_labels -- --ignored --nocapture
#[test]
#[ignore = "needs the yumon_code_bpe tokenizer; prints labels for inspection"]
fn real_tokenizer_context_labels() {
    let tokenizer = crate::brain::code_corpus::load_tokenizer("yumon_code_bpe").unwrap();
    let labeler = ContextLabeler::from_tokenizer(&tokenizer, tokenizer.vocab_size());
    let snippets = [
        "/// Represents a video file generated by the Veo 3.1 Lite model.\n/// The `url` field contains the link.\n#[derive(Debug)]\npub struct VideoFile {\n    pub url: String, // not \"here\"\n}\n",
        "impl FormattedText {\n    /* block // still */ pub fn get_glyphs(&self) -> &[TextGlyph] {\n        let s = \"a // b\"; let c = '\"'; &self.glyphs\n    }\n}\n",
    ];
    for snippet in snippets {
        let ids: Vec<i32> = tokenizer.encode(snippet).into_iter().map(|t| t as i32).collect();
        let labels = labeler.label(&ids, ids.len());
        let text: Vec<String> = ids.iter().zip(&labels)
            .map(|(&t, &l)| format!("{:?}:{}", tokenizer.decode(&[t as usize]), ["S", "N", "C", "cn", "cc"][l as usize])).collect();
        println!("{}\n", text.join(" "));
    }
}
