//! Contextual N-Gram / Statistical Disambiguation Decompressor Model.
//!
//! Decompresses compressed Yumon Talk / Kilo Talk representations back into natural, realistic English.
//!
//! For each canonical token $c_t$ with allowable word set $\{w_1, \dots, w_k\}$ (synonyms + inflections):
//! picks:
//!     $$\arg\max_{w_i} P(w_i \mid w_{t-2}, w_{t-1})$$
//! using Stupid Backoff over an $n$-gram language model built on raw English corpora.
//! Zero neural overhead, runs in microseconds on CPU, and ensures grammatical collocations sound natural.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use super::yumon_talk::{TranslationContext, YumonTalkVocab, get_global_vocab};

pub type WordId = u32;

pub const START_TOKEN: &str = "<s>";
pub const END_TOKEN: &str = "</s>";
pub const UNK_TOKEN: &str = "<unk>";

pub const START_ID: WordId = 0;
pub const END_ID: WordId = 1;
pub const UNK_ID: WordId = 2;

/// A statistical N-gram language model for contextual disambiguation and decompression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NGramDecompressor {
    pub order: usize,
    pub word_to_id: HashMap<String, WordId>,
    pub id_to_word: Vec<String>,
    pub unigram_counts: Vec<u32>,
    pub total_unigrams: u64,
    pub bigrams: HashMap<(WordId, WordId), u32>,
    pub trigrams: HashMap<(WordId, WordId, WordId), u32>,
    pub fourgrams: HashMap<(WordId, WordId, WordId, WordId), u32>,
    pub bigram_contexts: HashMap<WordId, u32>,
    pub trigram_contexts: HashMap<(WordId, WordId), u32>,
    pub fourgram_contexts: HashMap<(WordId, WordId, WordId), u32>,
}

#[derive(Clone, Debug)]
struct BeamHypothesis {
    log_score: f64,
    words: Vec<String>,
    ctx: [WordId; 3],
    cap_next: bool,
}

impl NGramDecompressor {
    /// Create a new, empty decompressor with the specified maximum n-gram order (e.g. 3 or 4).
    pub fn new(order: usize) -> Self {
        let order = order.clamp(2, 4);
        let mut model = Self {
            order,
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
            unigram_counts: Vec::new(),
            total_unigrams: 0,
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            fourgrams: HashMap::new(),
            bigram_contexts: HashMap::new(),
            trigram_contexts: HashMap::new(),
            fourgram_contexts: HashMap::new(),
        };

        // Seed standard special tokens
        model.get_or_add_word(START_TOKEN);
        model.get_or_add_word(END_TOKEN);
        model.get_or_add_word(UNK_TOKEN);

        model
    }

    /// Retrieve or allocate a WordId for a word.
    pub fn get_or_add_word(&mut self, word: &str) -> WordId {
        let clean = word.to_lowercase();
        if let Some(&id) = self.word_to_id.get(&clean) {
            return id;
        }
        let id = self.id_to_word.len() as WordId;
        self.word_to_id.insert(clean.clone(), id);
        self.id_to_word.push(clean);
        self.unigram_counts.push(0);
        id
    }

    /// Look up the WordId for a word if it exists in the vocabulary.
    pub fn get_word_id(&self, word: &str) -> Option<WordId> {
        let clean = word.to_lowercase();
        self.word_to_id.get(&clean).copied()
    }

    /// Look up word string by WordId.
    pub fn get_word(&self, id: WordId) -> Option<&str> {
        self.id_to_word.get(id as usize).map(|s| s.as_str())
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    /// Train on a sequence of clean token strings representing one sentence.
    pub fn train_on_sentence(&mut self, tokens: &[&str]) {
        if tokens.is_empty() {
            return;
        }

        let mut ids = Vec::with_capacity(tokens.len() + 4);
        // Prefix with start tokens
        ids.push(START_ID);
        ids.push(START_ID);
        ids.push(START_ID);

        for &tok in tokens {
            let wid = self.get_or_add_word(tok);
            self.unigram_counts[wid as usize] += 1;
            self.total_unigrams += 1;
            ids.push(wid);
        }

        ids.push(END_ID);

        // Record n-grams
        for i in 3..ids.len() {
            let w0 = ids[i - 3];
            let w1 = ids[i - 2];
            let w2 = ids[i - 1];
            let w3 = ids[i];

            // Bigram (w2, w3)
            *self.bigrams.entry((w2, w3)).or_insert(0) += 1;
            *self.bigram_contexts.entry(w2).or_insert(0) += 1;

            // Trigram (w1, w2, w3)
            if self.order >= 3 {
                *self.trigrams.entry((w1, w2, w3)).or_insert(0) += 1;
                *self.trigram_contexts.entry((w1, w2)).or_insert(0) += 1;
            }

            // 4-gram (w0, w1, w2, w3)
            if self.order >= 4 {
                *self.fourgrams.entry((w0, w1, w2, w3)).or_insert(0) += 1;
                *self.fourgram_contexts.entry((w0, w1, w2)).or_insert(0) += 1;
            }
        }
    }

    /// Normalize raw English text and extract sentences for training.
    pub fn train_on_text(&mut self, text: &str) {
        let normalized = text
            .replace("n't", " not")
            .replace("'re", " are")
            .replace("'ve", " have")
            .replace("'ll", " will")
            .replace("'d", " would")
            .replace("'m", " am")
            .replace("can't", "can not")
            .replace("won't", "will not")
            .replace("let's", "let us");

        // Split into sentences on sentence boundaries
        for raw_sentence in normalized.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n') {
            let mut words = Vec::new();
            for word_part in raw_sentence.split(|c: char| !c.is_alphabetic() && c != '\'') {
                let clean = word_part.trim_matches('\'').to_lowercase();
                if clean.len() >= 2 || clean == "i" || clean == "a" {
                    let base = clean.strip_suffix("'s").unwrap_or(&clean);
                    if base.chars().all(|c| c.is_ascii_lowercase()) && !base.is_empty() {
                        words.push(base.to_string());
                    }
                }
            }
            if !words.is_empty() {
                let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
                self.train_on_sentence(&word_refs);
            }
        }
    }

    /// Stream sample pairs from a training cache file (e.g. `training-cache/samples.bin`).
    pub fn train_on_cache(&mut self, cache_path: &Path, sample_limit: Option<usize>) -> Result<usize> {
        use super::sample_cache;
        ensure!(cache_path.exists(), "Cache file not found: {}", cache_path.display());
        let mut count = 0usize;
        sample_cache::stream_pairs_lenient(cache_path, sample_limit, |(prompt, reply)| {
            self.train_on_text(&prompt);
            self.train_on_text(&reply);
            count += 1;
            Ok(())
        })?;
        Ok(count)
    }

    /// Save the compressed model to a binary file using zstd compression.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let encoder = zstd::stream::write::Encoder::new(file, 3)?;
        let mut writer = BufWriter::new(encoder.auto_finish());
        bincode::serialize_into(&mut writer, self)
            .with_context(|| format!("serializing NGramDecompressor to {}", path.display()))?;
        writer.flush()?;
        Ok(())
    }

    /// Load the model from a binary zstd-compressed file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let decoder = zstd::stream::read::Decoder::new(file)?;
        let mut reader = BufReader::new(decoder);
        let model: Self = bincode::deserialize_from(&mut reader)
            .with_context(|| format!("deserializing NGramDecompressor from {}", path.display()))?;
        Ok(model)
    }

    /// Load from `model_path` if present; otherwise train on `cache_path` and save.
    pub fn load_or_train(model_path: &Path, cache_path: &Path, order: usize) -> Result<Self> {
        if model_path.exists() {
            if let Ok(model) = Self::load_from_file(model_path) {
                return Ok(model);
            }
        }
        let mut model = Self::new(order);
        if cache_path.exists() {
            model.train_on_cache(cache_path, None)?;
            let _ = model.save_to_file(model_path);
        }
        Ok(model)
    }

    /// Compute Stupid Backoff score for candidate word ID given 3 preceding context word IDs.
    pub fn score_candidate(&self, candidate_id: WordId, c3: WordId, c2: WordId, c1: WordId, is_canonical: bool) -> f64 {
        const ALPHA: f64 = 0.4;
        let mut score = 0.0;
        let mut found = false;

        // 1. Try 4-gram: P(candidate | c3, c2, c1)
        if self.order >= 4 && c3 != UNK_ID && c2 != UNK_ID && c1 != UNK_ID {
            if let Some(&count) = self.fourgrams.get(&(c3, c2, c1, candidate_id)) {
                if let Some(&ctx_total) = self.fourgram_contexts.get(&(c3, c2, c1)) {
                    if ctx_total > 0 {
                        score = (count as f64) / (ctx_total as f64);
                        found = true;
                    }
                }
            }
        }

        // 2. Try 3-gram: P(candidate | c2, c1)
        if !found && c2 != UNK_ID && c1 != UNK_ID {
            let weight = if self.order >= 4 { ALPHA } else { 1.0 };
            if let Some(&count) = self.trigrams.get(&(c2, c1, candidate_id)) {
                if let Some(&ctx_total) = self.trigram_contexts.get(&(c2, c1)) {
                    if ctx_total > 0 {
                        score = weight * (count as f64) / (ctx_total as f64);
                        found = true;
                    }
                }
            }
        }

        // 3. Try 2-gram: P(candidate | c1)
        if !found && c1 != UNK_ID {
            let weight = if self.order >= 4 { ALPHA * ALPHA } else { ALPHA };
            if let Some(&count) = self.bigrams.get(&(c1, candidate_id)) {
                if let Some(&ctx_total) = self.bigram_contexts.get(&c1) {
                    if ctx_total > 0 {
                        score = weight * (count as f64) / (ctx_total as f64);
                        found = true;
                    }
                }
            }
        }

        // 4. Try unigram: P(candidate)
        if !found {
            let weight = if self.order >= 4 { ALPHA * ALPHA * ALPHA } else { ALPHA * ALPHA };
            let unigram_c = self.unigram_counts.get(candidate_id as usize).copied().unwrap_or(0);
            let vocab_size = self.id_to_word.len() as f64;
            let total = (self.total_unigrams.max(1)) as f64;
            if unigram_c > 0 {
                score = weight * (unigram_c as f64) / total;
            } else {
                score = weight * 1.0 / (total + vocab_size);
            }
        }

        // Slight tie-breaker bonus for canonical token to avoid arbitrary synonym jitter
        if is_canonical {
            score += 1e-7;
        }

        score
    }

    /// Decompress a Yumon Talk / Kilo Talk token sequence into natural English using beam search.
    pub fn decompress_tokens(
        &self,
        tokens: &[usize],
        vocab: &YumonTalkVocab,
        ctx: &TranslationContext,
        beam_width: usize,
    ) -> Result<String> {
        let beam_width = beam_width.max(1);
        let initial = BeamHypothesis {
            log_score: 0.0,
            words: Vec::new(),
            ctx: [START_ID, START_ID, START_ID],
            cap_next: true,
        };

        let mut beam = vec![initial];

        for &id in tokens {
            if id == super::yumon_talk::PAD_ID
                || id == super::yumon_talk::BOS_ID
                || id == super::yumon_talk::EOS_ID
            {
                continue;
            }

            let tok = vocab.id_to_token(id).unwrap_or("<UNK>");

            // 1. Punctuation
            if tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":" || tok == ")" || tok == "]" {
                for h in beam.iter_mut() {
                    h.words.push(tok.to_string());
                    if tok == "." || tok == "!" || tok == "?" {
                        h.cap_next = true;
                        h.ctx = [START_ID, START_ID, START_ID];
                    }
                }
                continue;
            }

            if tok == "(" || tok == "[" {
                for h in beam.iter_mut() {
                    h.words.push(tok.to_string());
                }
                continue;
            }

            if tok == "\n" {
                for h in beam.iter_mut() {
                    h.words.push("\n".to_string());
                    h.cap_next = true;
                    h.ctx = [START_ID, START_ID, START_ID];
                }
                continue;
            }

            // 2. Entity or Number Placeholders
            if tok.starts_with("<NAME_") && tok.ends_with('>') {
                let num_part = &tok[6..tok.len() - 1];
                let entity_str = if let Ok(idx) = num_part.parse::<usize>() {
                    ctx.entities.get(idx).cloned().unwrap_or_else(|| format!("Entity_{idx}"))
                } else {
                    tok.to_string()
                };

                for h in beam.iter_mut() {
                    h.words.push(entity_str.clone());
                    // Update context with entity words
                    for part in entity_str.split(|c: char| !c.is_alphabetic()) {
                        let clean = part.to_lowercase();
                        if !clean.is_empty() {
                            let wid = self.get_word_id(&clean).unwrap_or(UNK_ID);
                            h.ctx = [h.ctx[1], h.ctx[2], wid];
                        }
                    }
                    h.cap_next = false;
                }
                continue;
            }

            if tok.starts_with("<NUM_") && tok.ends_with('>') {
                let num_part = &tok[5..tok.len() - 1];
                let num_str = if let Ok(idx) = num_part.parse::<usize>() {
                    ctx.numbers.get(idx).cloned().unwrap_or_else(|| format!("{idx}"))
                } else {
                    tok.to_string()
                };

                for h in beam.iter_mut() {
                    h.words.push(num_str.clone());
                    h.cap_next = false;
                }
                continue;
            }

            if tok.starts_with("<SLOT_") && tok.ends_with('>') {
                let num_part = &tok[6..tok.len() - 1];
                let slot_str = if let Ok(idx) = num_part.parse::<usize>() {
                    ctx.slots.get(idx).cloned().unwrap_or_else(|| format!("Slot_{idx}"))
                } else {
                    tok.to_string()
                };

                for h in beam.iter_mut() {
                    let formatted = if h.cap_next {
                        capitalize_word(&slot_str)
                    } else {
                        slot_str.clone()
                    };
                    h.words.push(formatted);
                    for part in slot_str.split(|c: char| !c.is_alphabetic()) {
                        let clean = part.to_lowercase();
                        if !clean.is_empty() {
                            let wid = self.get_word_id(&clean).unwrap_or(UNK_ID);
                            h.ctx = [h.ctx[1], h.ctx[2], wid];
                        }
                    }
                    h.cap_next = false;
                }
                continue;
            }

            // 3. Canonical Word: Disambiguation across allowable synonyms & inflections
            let allowable = vocab.allowable_words(tok);
            let candidates: &[String] = if allowable.is_empty() {
                &[tok.to_string()]
            } else {
                &allowable
            };

            let mut next_beam = Vec::new();

            for h in &beam {
                for cand in candidates {
                    let is_canonical = cand.eq_ignore_ascii_case(tok);
                    let wid = self.get_word_id(cand).unwrap_or(UNK_ID);
                    let score = self.score_candidate(wid, h.ctx[0], h.ctx[1], h.ctx[2], is_canonical);
                    let cand_log = score.max(1e-12).ln();

                    let formatted = if h.cap_next {
                        capitalize_word(cand)
                    } else if cand == "i" {
                        "I".to_string()
                    } else {
                        cand.clone()
                    };

                    let mut new_words = h.words.clone();
                    new_words.push(formatted);

                    next_beam.push(BeamHypothesis {
                        log_score: h.log_score + cand_log,
                        words: new_words,
                        ctx: [h.ctx[1], h.ctx[2], wid],
                        cap_next: false,
                    });
                }
            }

            // Keep top beam_width hypotheses
            next_beam.sort_by(|a, b| b.log_score.partial_cmp(&a.log_score).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(beam_width);
            beam = next_beam;
        }

        let best = beam.into_iter().next().context("beam search produced no hypothesis")?;
        Ok(format_sentence(&best.words))
    }

    /// Decompress a Yumon Talk string into natural English using beam search (default beam_width = 4).
    pub fn decompress_with_context(
        &self,
        yumon_text: &str,
        vocab: &YumonTalkVocab,
        ctx: &TranslationContext,
    ) -> Result<String> {
        let tokens = vocab.tokenize_yumon(yumon_text);
        self.decompress_tokens(&tokens, vocab, ctx, 4)
    }

    /// Fast greedy decompression (beam_width = 1).
    pub fn decompress_greedy(
        &self,
        yumon_text: &str,
        vocab: &YumonTalkVocab,
        ctx: &TranslationContext,
    ) -> Result<String> {
        let tokens = vocab.tokenize_yumon(yumon_text);
        self.decompress_tokens(&tokens, vocab, ctx, 1)
    }
}

/// Helper to capitalize a word's first character.
fn capitalize_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
    }
}

/// Helper to format words into natural English text with correct spacing around punctuation.
fn format_sentence(tokens: &[String]) -> String {
    let mut out = String::new();
    for tok in tokens {
        if tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":" || tok == ")" || tok == "]" {
            out.push_str(tok);
            continue;
        }
        if tok == "(" || tok == "[" {
            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
            out.push_str(tok);
            continue;
        }
        if tok == "\n" {
            out.push('\n');
            continue;
        }
        if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') && !out.ends_with('(') && !out.ends_with('[') {
            out.push(' ');
        }
        out.push_str(tok);
    }
    out
}

static GLOBAL_DECOMPRESSOR: OnceLock<RwLock<Option<Arc<NGramDecompressor>>>> = OnceLock::new();

/// Retrieve or lazy-initialize the global N-gram decompressor.
pub fn get_global_decompressor() -> Option<Arc<NGramDecompressor>> {
    let rw = GLOBAL_DECOMPRESSOR.get_or_init(|| {
        let env_path = std::env::var("YUMON_DECOMPRESSOR").ok().map(std::path::PathBuf::from);
        let default_path = Path::new("training-cache/ngram_decompressor.bin");
        let path = env_path.as_deref().unwrap_or(default_path);
        if path.exists() {
            if let Ok(model) = NGramDecompressor::load_from_file(path) {
                return RwLock::new(Some(Arc::new(model)));
            }
        }
        // Try building from samples.bin if available
        let samples_path = Path::new("training-cache/samples.bin");
        if samples_path.exists() {
            let mut model = NGramDecompressor::new(3);
            if model.train_on_cache(samples_path, None).is_ok() {
                let _ = model.save_to_file(path);
                return RwLock::new(Some(Arc::new(model)));
            }
        }
        RwLock::new(None)
    });
    rw.read().ok().and_then(|g| g.clone())
}

/// Set or override the global N-gram decompressor.
pub fn set_global_decompressor(decompressor: NGramDecompressor) {
    let rw = GLOBAL_DECOMPRESSOR.get_or_init(|| RwLock::new(None));
    if let Ok(mut guard) = rw.write() {
        *guard = Some(Arc::new(decompressor));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vocab() -> YumonTalkVocab {
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
            "a".into(), "good".into(), "friend".into(), "weather".into(), "cold".into(),
        ];
        let mut clusters = HashMap::new();
        clusters.insert("happy".into(), vec!["happy".into(), "glad".into(), "delighted".into()]);
        clusters.insert("big".into(), vec!["big".into(), "huge".into(), "massive".into(), "significant".into()]);
        clusters.insert("cold".into(), vec!["cold".into(), "chilly".into(), "freezing".into()]);

        YumonTalkVocab::new(special, punct, placeholders, canonical, clusters)
    }

    #[test]
    fn test_ngram_training_and_disambiguation() {
        let mut dec = NGramDecompressor::new(3);
        dec.train_on_text("I am very happy today. They were happy yesterday.");
        dec.train_on_text("This project made a significant impact on performance.");
        dec.train_on_text("It was a massive earthquake.");

        let vocab = sample_vocab();
        let ctx = TranslationContext::new();

        // Check pronoun and verb agreement disambiguation:
        // "i be happy" -> "I am happy." (or "I was happy.")
        let decompressed = dec.decompress_with_context("i be happy .", &vocab, &ctx).unwrap();
        assert!(decompressed.starts_with("I am") || decompressed.starts_with("I was"),
            "Expected 'I am' or 'I was', got: {decompressed}");

        // "they be happy" -> "They were happy."
        let decompressed_they = dec.decompress_with_context("they be happy .", &vocab, &ctx).unwrap();
        assert!(decompressed_they.starts_with("They were"),
            "Expected 'They were', got: {decompressed_they}");

        // "significant impact" vs "massive impact"
        let decomp_impact = dec.decompress_with_context("a big impact .", &vocab, &ctx).unwrap();
        assert!(decomp_impact.contains("significant impact") || decomp_impact.contains("massive impact"),
            "Expected collocation with impact, got: {decomp_impact}");
    }

    #[test]
    fn test_placeholder_restoration_and_capitalization() {
        let mut dec = NGramDecompressor::new(3);
        dec.train_on_text("Google released a new product today.");

        let vocab = sample_vocab();
        let mut ctx = TranslationContext::new();
        ctx.entities.push("Google".to_string());

        let decompressed = dec.decompress_with_context("<NAME_0> have a big impact .", &vocab, &ctx).unwrap();
        assert!(decompressed.starts_with("Google"), "Entity was not preserved: {decompressed}");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut dec = NGramDecompressor::new(3);
        dec.train_on_text("Natural language processing with statistical decompression.");

        let dir = std::env::temp_dir().join(format!("decompressor-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");

        dec.save_to_file(&path).unwrap();
        let loaded = NGramDecompressor::load_from_file(&path).unwrap();

        assert_eq!(dec.order, loaded.order);
        assert_eq!(dec.vocab_size(), loaded.vocab_size());
        assert_eq!(dec.total_unigrams, loaded.total_unigrams);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
