//! Integration tests for the Contextual N-Gram Statistical Disambiguation Decompressor.

use std::collections::HashMap;
use yumon_pet::brain::{
    decompressor::NGramDecompressor,
    yumon_talk::{TranslationContext, YumonTalkVocab},
};

fn test_vocab() -> YumonTalkVocab {
    let special = vec!["<PAD>".into(), "<BOS>".into(), "<EOS>".into(), "<UNK>".into()];
    let punct = vec![".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "/", "\n"]
        .into_iter()
        .map(String::from)
        .collect();
    let placeholders = (0..48).map(|i| format!("<NAME_{i}>"))
        .chain((0..20).map(|i| format!("<NUM_{i}>"))).collect();
    let canonical = vec![
        "i".into(), "be".into(), "happy".into(), "have".into(), "big".into(),
        "impact".into(), "they".into(), "she".into(), "he".into(), "do".into(),
        "a".into(), "good".into(), "friend".into(), "water".into(), "give".into(),
        "some".into(), "because".into(), "thirsty".into(), "device".into(),
        "new".into(), "announced".into(),
    ];
    let mut clusters = HashMap::new();
    clusters.insert("happy".into(), vec!["happy".into(), "delighted".into(), "glad".into()]);
    clusters.insert("big".into(), vec!["big".into(), "huge".into(), "massive".into(), "significant".into()]);
    clusters.insert("good".into(), vec!["good".into(), "great".into(), "fine".into(), "excellent".into()]);
    clusters.insert("give".into(), vec!["give".into(), "provide".into(), "grant".into()]);

    YumonTalkVocab::new(special, punct, placeholders, canonical, clusters)
}

#[test]
fn test_pronoun_verb_agreement_disambiguation() {
    let mut dec = NGramDecompressor::new(3);
    dec.train_on_text("I am very happy today. They were happy to see us.");
    dec.train_on_text("Give me some water because I am thirsty.");
    dec.train_on_text("He has a great friend. She was delighted with the news.");

    let vocab = test_vocab();
    let ctx = TranslationContext::new();

    // 1. "i be happy" -> Should select "I am" or "I was" instead of "I are" or "I were"
    let decomp = dec.decompress_with_context("i be happy .", &vocab, &ctx).unwrap();
    assert!(decomp.starts_with("I am") || decomp.starts_with("I was"),
        "Expected 'I am' or 'I was', got: {decomp}");

    // 2. "they be happy" -> Should select "They were" or "They are"
    let decomp_they = dec.decompress_with_context("they be happy .", &vocab, &ctx).unwrap();
    assert!(decomp_they.starts_with("They were") || decomp_they.starts_with("They are"),
        "Expected 'They were' or 'They are', got: {decomp_they}");

    // 3. "give i some water" -> "give" context should select objective "me" over nominative "i"
    let decomp_give = dec.decompress_with_context("give i some water .", &vocab, &ctx).unwrap();
    assert!(decomp_give.to_lowercase().contains("give me") || decomp_give.to_lowercase().contains("provide me"),
        "Expected 'give me', got: {decomp_give}");
}

#[test]
fn test_collocation_disambiguation_significant_impact() {
    let mut dec = NGramDecompressor::new(3);
    dec.train_on_text("This project made a significant impact on performance.");
    dec.train_on_text("The update had a significant impact across all benchmarks.");
    dec.train_on_text("It caused a massive explosion.");

    let vocab = test_vocab();
    let ctx = TranslationContext::new();

    // "a big impact ." -> "a significant impact ." because "significant impact" has strong bi-gram frequency
    let decomp = dec.decompress_with_context("a big impact .", &vocab, &ctx).unwrap();
    assert!(decomp.contains("significant impact"),
        "Expected 'significant impact', got: {decomp}");
}

#[test]
fn test_entity_and_number_restoration() {
    let mut dec = NGramDecompressor::new(3);
    dec.train_on_text("Nintendo announced a new device.");
    dec.train_on_text("I ordered 42 items yesterday.");

    let vocab = test_vocab();
    let mut ctx = TranslationContext::new();
    ctx.entities.push("Nintendo".to_string());
    ctx.numbers.push("42".to_string());

    let decomp = dec.decompress_with_context("<NAME_0> announced <NUM_0> new device .", &vocab, &ctx).unwrap();
    assert!(decomp.contains("Nintendo"), "Entity 'Nintendo' not found in: {decomp}");
    assert!(decomp.contains("42"), "Number '42' not found in: {decomp}");
}

#[test]
fn test_greedy_vs_beam_search() {
    let mut dec = NGramDecompressor::new(3);
    dec.train_on_text("I am very happy to see my good friend.");

    let vocab = test_vocab();
    let ctx = TranslationContext::new();

    let beam_out = dec.decompress_with_context("i be happy .", &vocab, &ctx).unwrap();
    let greedy_out = dec.decompress_greedy("i be happy .", &vocab, &ctx).unwrap();

    assert!(!beam_out.is_empty());
    assert!(!greedy_out.is_empty());
}
