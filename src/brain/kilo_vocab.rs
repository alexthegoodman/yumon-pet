//! Kilo Tokenizer / Vocab Mapping Model (4096 bounded tokens).
//!
//! Maps natural English into a strictly bounded 4,096-token vocabulary designed for
//! downstream Mixture-of-Experts training and contextual N-gram decompression.
//!
//! Unlike static thesaurus-based clustering, this model:
//! 1. Retains distinct words as separate vocabulary entries (e.g. `world` != `universe`).
//! 2. Mines and matches high-frequency recurring multi-word phrases (e.g. `as well as`, `for example`)
//!    using greedy longest-match prefix search.
//! 3. Preserves out-of-vocabulary entities, numbers, and template slots in `TranslationContext`.
//! 4. Maintains full compatibility with `NGramDecompressor`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use super::yumon_talk::{
    SEALED_WORDS,
    TranslationContext,
    morphological_variants,
    normalize_contractions,
};

pub const VOCAB_SIZE: usize = 4096;
pub const PAD_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const UNK_ID: usize = 3;

pub const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<s>", "</s>", "<unk>"];

pub const STANDARD_PUNCTUATION: [&str; 24] = [
    ".", ",", "!", "?", ";", ":",
    "(", ")", "[", "]", "{", "}",
    "\"", "'", "`", "-", "—", "/",
    "\\", "@", "#", "$", "%", "\n",
];

pub const MAX_NAME_SLOTS: usize = 48;
pub const MAX_NUM_SLOTS: usize = 20;
pub const MAX_DYNAMIC_SLOTS: usize = 16;
pub const TOTAL_PLACEHOLDERS: usize = MAX_NAME_SLOTS + MAX_NUM_SLOTS + MAX_DYNAMIC_SLOTS; // 84

pub const FIXED_NON_VOCAB_COUNT: usize = SPECIAL_TOKENS.len() + STANDARD_PUNCTUATION.len() + TOTAL_PLACEHOLDERS; // 112
pub const TARGET_LEARNED_VOCAB: usize = VOCAB_SIZE - FIXED_NON_VOCAB_COUNT; // 3984

/// Prefix trie node for greedy longest-match multi-word phrase matching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhraseTrieNode {
    pub token_id: Option<usize>,
    pub children: HashMap<String, PhraseTrieNode>,
}

impl PhraseTrieNode {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, phrase_words: &[String], token_id: usize) {
        if phrase_words.is_empty() {
            return;
        }
        let mut curr = self;
        for word in phrase_words {
            let key = word.to_lowercase();
            curr = curr.children.entry(key).or_default();
        }
        curr.token_id = Some(token_id);
    }

    pub fn longest_match(&self, words: &[String]) -> Option<(usize, usize)> {
        let mut curr = self;
        let mut best = None;
        for (i, word) in words.iter().enumerate() {
            let key = word.to_lowercase();
            if let Some(next) = curr.children.get(&key) {
                if let Some(id) = next.token_id {
                    best = Some((i + 1, id));
                }
                curr = next;
            } else {
                break;
            }
        }
        best
    }
}

/// Compression and encoding statistics for an input text sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenStats {
    pub raw_word_count: usize,
    pub token_count: usize,
    pub phrase_tokens_used: usize,
    pub entities_preserved: usize,
    pub numbers_preserved: usize,
    pub slots_preserved: usize,
    pub direct_hits: usize,
}

impl TokenStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.token_count == 0 {
            0.0
        } else {
            self.raw_word_count as f64 / self.token_count as f64
        }
    }
}

/// The Kilo Tokenizer / Vocab Mapping Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KiloVocab {
    pub version: usize,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub punctuation: Vec<String>,
    pub placeholders: Vec<String>,
    pub learned_vocab: Vec<String>,
    pub phrase_trie: PhraseTrieNode,

    #[serde(skip)]
    pub token_to_id: HashMap<String, usize>,
    #[serde(skip)]
    pub id_to_token: Vec<String>,
    #[serde(skip)]
    pub word_to_id: HashMap<String, usize>,
    #[serde(skip)]
    pub allowable_cache: HashMap<String, Vec<String>>,
}

impl Default for KiloVocab {
    fn default() -> Self {
        Self::empty()
    }
}

impl KiloVocab {
    /// Create an unpopulated baseline vocabulary with standard special, punctuation, and placeholder tokens.
    pub fn empty() -> Self {
        let special_tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|&s| s.to_string()).collect();
        let punctuation: Vec<String> = STANDARD_PUNCTUATION.iter().map(|&s| s.to_string()).collect();
        let mut placeholders = Vec::with_capacity(TOTAL_PLACEHOLDERS);

        for i in 0..MAX_NAME_SLOTS {
            placeholders.push(format!("<NAME_{i}>"));
        }
        for i in 0..MAX_NUM_SLOTS {
            placeholders.push(format!("<NUM_{i}>"));
        }
        for i in 0..MAX_DYNAMIC_SLOTS {
            placeholders.push(format!("<SLOT_{i}>"));
        }

        let mut vocab = Self {
            version: 1,
            vocab_size: special_tokens.len() + punctuation.len() + placeholders.len(),
            special_tokens,
            punctuation,
            placeholders,
            learned_vocab: Vec::new(),
            phrase_trie: PhraseTrieNode::new(),
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            word_to_id: HashMap::new(),
            allowable_cache: HashMap::new(),
        };
        vocab.rebuild_indices();
        vocab
    }

    /// Rebuild runtime lookup tables and prefix trie from serializable vectors.
    pub fn rebuild_indices(&mut self) {
        let total = self.special_tokens.len()
            + self.punctuation.len()
            + self.placeholders.len()
            + self.learned_vocab.len();

        let mut id_to_token = Vec::with_capacity(total);
        let mut token_to_id = HashMap::with_capacity(total);
        let mut word_to_id = HashMap::with_capacity(total * 2);
        let mut trie = PhraseTrieNode::new();

        // 1. Special tokens (0..3)
        for tok in &self.special_tokens {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }

        // 2. Punctuation
        for tok in &self.punctuation {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }

        // 3. Placeholders (<NAME_x>, <NUM_x>, <SLOT_x>)
        for tok in &self.placeholders {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }

        // 4. Learned vocabulary (words and multi-word phrases)
        for tok in &self.learned_vocab {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());

            let lower = tok.to_lowercase();
            if lower.contains(' ') {
                let parts: Vec<String> = lower.split_whitespace().map(|s| s.to_string()).collect();
                trie.insert(&parts, id);
            } else {
                word_to_id.insert(lower, id);
            }
        }

        // 5. Sealed closed-class word aliases (pronoun forms, auxiliaries)
        for &(canon, syns) in SEALED_WORDS {
            if let Some(&canon_id) = word_to_id.get(canon) {
                for &syn in syns {
                    word_to_id.entry(syn.to_lowercase()).or_insert(canon_id);
                }
            }
        }

        self.vocab_size = total;
        self.token_to_id = token_to_id;
        self.id_to_token = id_to_token;
        self.word_to_id = word_to_id;
        self.phrase_trie = trie;
        self.allowable_cache.clear();
    }

    /// Learn a 4,096-token vocabulary from empirical corpus frequencies without thesaurus collapsing.
    ///
    /// Distinct words (e.g. `world` and `universe`) retain separate canonical tokens.
    /// Frequent multi-word phrases (e.g. `as well as`) are assigned single token IDs.
    pub fn learn_from_corpus(
        word_counts: &HashMap<String, usize>,
        phrase_counts: &HashMap<String, usize>,
        target_vocab_size: usize,
        max_phrases: usize,
    ) -> Self {
        let mut model = Self::empty();
        let target_learned = target_vocab_size.saturating_sub(FIXED_NON_VOCAB_COUNT);

        let mut learned = Vec::with_capacity(target_learned);
        let mut seen = HashSet::new();

        // 1. Seed closed-class words (pronouns, prepositions, conjunctions)
        for &(canon, _) in SEALED_WORDS {
            let clean = canon.to_lowercase();
            if !seen.contains(&clean) && learned.len() < target_learned {
                seen.insert(clean.clone());
                learned.push(clean);
            }
        }

        // 2. Select top multi-word phrases (collocations with >= 2 words)
        let mut sorted_phrases: Vec<(String, usize)> = phrase_counts
            .iter()
            .filter(|(phrase, _)| phrase.contains(' '))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        sorted_phrases.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let phrase_limit = max_phrases.min(target_learned.saturating_sub(learned.len()) / 4);
        for (phrase, _) in sorted_phrases.into_iter().take(phrase_limit) {
            let clean = phrase.trim().to_lowercase();
            if !clean.is_empty() && !seen.contains(&clean) && learned.len() < target_learned {
                seen.insert(clean.clone());
                learned.push(clean);
            }
        }

        // 3. Select top empirical unigrams by frequency (preserving distinct words)
        let mut sorted_words: Vec<(String, usize)> = word_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for (word, _) in sorted_words {
            if learned.len() >= target_learned {
                break;
            }
            let clean = word.trim().to_lowercase();
            if clean.len() >= 2 || clean == "a" || clean == "i" {
                if !seen.contains(&clean) {
                    seen.insert(clean.clone());
                    learned.push(clean);
                }
            }
        }

        model.learned_vocab = learned;
        model.rebuild_indices();
        model
    }

    /// Look up token ID by exact string.
    pub fn token_to_id(&self, tok: &str) -> Option<usize> {
        self.token_to_id.get(tok).copied()
    }

    /// Look up token string by ID.
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Check if an ID belongs to a named entity placeholder.
    pub fn is_name_token(&self, id: usize) -> bool {
        self.id_to_token(id).map_or(false, |tok| tok.starts_with("<NAME_") && tok.ends_with('>'))
    }

    /// Check if an ID belongs to a number placeholder.
    pub fn is_num_token(&self, id: usize) -> bool {
        self.id_to_token(id).map_or(false, |tok| tok.starts_with("<NUM_") && tok.ends_with('>'))
    }

    /// Check if an ID belongs to a template slot placeholder.
    pub fn is_slot_token(&self, id: usize) -> bool {
        self.id_to_token(id).map_or(false, |tok| tok.starts_with("<SLOT_") && tok.ends_with('>'))
    }

    /// Resolve an individual English word to a canonical token ID, checking direct vocabulary
    /// matches and morphological lemmatization variants.
    pub fn resolve_word(&self, word: &str) -> Option<usize> {
        let lower = word.to_lowercase();
        if let Some(&id) = self.word_to_id.get(&lower) {
            return Some(id);
        }

        // Morphological stemming fallback
        for variant in morphological_variants(&lower) {
            if variant != lower && !variant.is_empty() {
                if let Some(&id) = self.word_to_id.get(&variant) {
                    return Some(id);
                }
            }
        }

        None
    }

    /// Provide allowable word alternatives for a canonical token for `NGramDecompressor` beam search.
    ///
    /// Rather than arbitrary thesaurus synonyms, returns the canonical word plus its legitimate
    /// morphological inflections (e.g. `run` -> `run`, `runs`, `running`, `ran`).
    pub fn allowable_words(&self, tok: &str) -> Vec<String> {
        let lower = tok.to_lowercase();
        let mut candidates = Vec::with_capacity(6);
        candidates.push(lower.clone());

        // Add morphological variants
        for variant in morphological_variants(&lower) {
            if variant != lower && !variant.is_empty() && !candidates.contains(&variant) {
                candidates.push(variant);
            }
        }

        // Add sealed word aliases if applicable
        for &(canon, syns) in SEALED_WORDS {
            if canon == lower {
                for &s in syns {
                    let s_low = s.to_lowercase();
                    if !candidates.contains(&s_low) {
                        candidates.push(s_low);
                    }
                }
            }
        }

        candidates
    }

    /// Tokenize English text into Kilo token IDs using greedy longest-match phrase matching
    /// and entity/number/slot placeholder tracking.
    pub fn encode(&self, text: &str, ctx: &mut TranslationContext) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let normalized = normalize_contractions(text);
        let chars: Vec<char> = normalized.chars().collect();
        let len = chars.len();
        let mut i = 0;

        let mut entity_map: HashMap<String, usize> = HashMap::new();
        let mut num_map: HashMap<String, usize> = HashMap::new();
        let mut slot_map: HashMap<String, usize> = HashMap::new();

        while i < len {
            // Whitespace handling
            if chars[i].is_whitespace() {
                if chars[i] == '\n' {
                    if let Some(id) = self.token_to_id("\n") {
                        tokens.push(id);
                    }
                }
                i += 1;
                continue;
            }

            // Word or phrase candidate
            if chars[i].is_alphabetic() || chars[i] == '_' {
                // Peek ahead to collect consecutive words for multi-word phrase matching
                let mut word_spans = Vec::new();
                let mut peek_idx = i;

                while peek_idx < len {
                    while peek_idx < len && chars[peek_idx].is_whitespace() && chars[peek_idx] != '\n' {
                        peek_idx += 1;
                    }
                    if peek_idx < len && (chars[peek_idx].is_alphabetic() || chars[peek_idx] == '_') {
                        let w_start = peek_idx;
                        while peek_idx < len && (chars[peek_idx].is_alphanumeric() || chars[peek_idx] == '_' || (chars[peek_idx] == '\'' && peek_idx + 1 < len && chars[peek_idx + 1].is_alphabetic())) {
                            peek_idx += 1;
                        }
                        let w: String = chars[w_start..peek_idx].iter().collect();
                        word_spans.push((w, peek_idx));
                        if word_spans.len() >= 4 {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // 1. Try greedy multi-word phrase match
                if word_spans.len() >= 2 {
                    let words: Vec<String> = word_spans.iter().map(|(w, _)| w.clone()).collect();
                    if let Some((match_len, phrase_id)) = self.phrase_trie.longest_match(&words) {
                        tokens.push(phrase_id);
                        i = word_spans[match_len - 1].1;
                        continue;
                    }
                }

                // 2. Single-word resolution
                let (single_word, next_i) = &word_spans[0];
                let word = single_word.clone();
                let start_ch = chars[i];
                i = *next_i;

                if let Some(id) = self.resolve_word(&word) {
                    tokens.push(id);
                    continue;
                }

                // 3. Fallback: Named entity vs. Numeric vs. Template slot
                let is_capitalized_entity = start_ch.is_uppercase()
                    && (word.len() > 1 || (start_ch != 'A' && start_ch != 'I'));

                if is_capitalized_entity {
                    let ent_idx = *entity_map.entry(word.clone()).or_insert_with(|| {
                        let idx = ctx.entities.len();
                        ctx.entities.push(word);
                        idx % MAX_NAME_SLOTS
                    });
                    let slot_str = format!("<NAME_{ent_idx}>");
                    tokens.push(self.token_to_id(&slot_str).unwrap_or(UNK_ID));
                } else {
                    let slot_idx = *slot_map.entry(word.clone()).or_insert_with(|| {
                        let idx = ctx.slots.len();
                        ctx.slots.push(word);
                        idx % MAX_DYNAMIC_SLOTS
                    });
                    let slot_str = format!("<SLOT_{slot_idx}>");
                    tokens.push(self.token_to_id(&slot_str).unwrap_or(UNK_ID));
                }
                continue;
            }

            // Numeric literals
            if chars[i].is_ascii_digit() {
                let start = i;
                while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let idx = *num_map.entry(num_str.clone()).or_insert_with(|| {
                    let curr = ctx.numbers.len();
                    ctx.numbers.push(num_str);
                    curr % MAX_NUM_SLOTS
                });
                let slot = format!("<NUM_{idx}>");
                tokens.push(self.token_to_id(&slot).unwrap_or(UNK_ID));
                continue;
            }

            // Punctuation
            let punct = chars[i].to_string();
            if let Some(id) = self.token_to_id(&punct) {
                tokens.push(id);
                i += 1;
                continue;
            }

            // Fallback unmapped char
            i += 1;
        }

        Ok(tokens)
    }

    /// Detokenize Kilo token IDs back into canonical token strings.
    pub fn decode_tokens(&self, tokens: &[usize]) -> String {
        let mut parts = Vec::with_capacity(tokens.len());
        for &id in tokens {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            if let Some(tok) = self.id_to_token(id) {
                parts.push(tok);
            }
        }
        parts.join(" ")
    }

    /// Detokenize token IDs expanding `<NAME_x>`, `<NUM_x>`, and `<SLOT_x>` using `TranslationContext`.
    pub fn decode_with_context(&self, tokens: &[usize], ctx: &TranslationContext) -> String {
        let mut words = Vec::with_capacity(tokens.len());
        for &id in tokens {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            let tok = match self.id_to_token(id) {
                Some(t) => t,
                None => continue,
            };

            if tok.starts_with("<NAME_") && tok.ends_with('>') {
                let num_str = &tok[6..tok.len() - 1];
                if let Ok(idx) = num_str.parse::<usize>() {
                    let val = ctx.entities.get(idx).cloned().unwrap_or_else(|| tok.to_string());
                    words.push(val);
                    continue;
                }
            } else if tok.starts_with("<NUM_") && tok.ends_with('>') {
                let num_str = &tok[5..tok.len() - 1];
                if let Ok(idx) = num_str.parse::<usize>() {
                    let val = ctx.numbers.get(idx).cloned().unwrap_or_else(|| tok.to_string());
                    words.push(val);
                    continue;
                }
            } else if tok.starts_with("<SLOT_") && tok.ends_with('>') {
                let num_str = &tok[6..tok.len() - 1];
                if let Ok(idx) = num_str.parse::<usize>() {
                    let val = ctx.slots.get(idx).cloned().unwrap_or_else(|| tok.to_string());
                    words.push(val);
                    continue;
                }
            }

            words.push(tok.to_string());
        }

        // Format clean natural spacing
        let mut result = String::new();
        for (i, word) in words.iter().enumerate() {
            let is_punct = word == "." || word == "," || word == "!" || word == "?" || word == ";" || word == ":";
            if i > 0 && !is_punct && word != "\n" && !result.ends_with('\n') {
                result.push(' ');
            }
            result.push_str(word);
        }
        result
    }

    /// Calculate encoding and compression statistics for a text string.
    pub fn analyze_encoding(&self, text: &str) -> Result<TokenStats> {
        let mut ctx = TranslationContext::new();
        let tokens = self.encode(text, &mut ctx)?;
        let raw_words = text.split_whitespace().count();

        let mut phrase_count = 0;
        let mut direct_count = 0;

        for &id in &tokens {
            if let Some(tok) = self.id_to_token(id) {
                if tok.contains(' ') {
                    phrase_count += 1;
                } else if !self.is_name_token(id) && !self.is_num_token(id) && !self.is_slot_token(id) {
                    direct_count += 1;
                }
            }
        }

        Ok(TokenStats {
            raw_word_count: raw_words,
            token_count: tokens.len(),
            phrase_tokens_used: phrase_count,
            entities_preserved: ctx.entities.len(),
            numbers_preserved: ctx.numbers.len(),
            slots_preserved: ctx.slots.len(),
            direct_hits: direct_count,
        })
    }

    /// Save the vocabulary model to a JSON file.
    pub fn save_to_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .with_context(|| format!("serializing KiloVocab to {}", path.display()))?;
        Ok(())
    }

    /// Load the vocabulary model from a JSON file.
    pub fn load_from_json(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut model: Self = serde_json::from_reader(reader)
            .with_context(|| format!("deserializing KiloVocab from {}", path.display()))?;
        model.rebuild_indices();
        Ok(model)
    }

    /// Convert to a YumonTalkVocab for full backwards-compatibility with legacy tools.
    pub fn to_yumon_talk_vocab(&self) -> super::yumon_talk::YumonTalkVocab {
        let mut synonym_clusters = HashMap::new();
        for word in &self.learned_vocab {
            let allow = self.allowable_words(word);
            if !allow.is_empty() {
                synonym_clusters.insert(word.clone(), allow);
            }
        }
        super::yumon_talk::YumonTalkVocab::new(
            self.special_tokens.clone(),
            self.punctuation.clone(),
            self.placeholders.clone(),
            self.learned_vocab.clone(),
            synonym_clusters,
        )
    }

    /// Export as a standard HuggingFace WordLevel `tokenizer.json` for Burn / tokenizers runtime.
    pub fn save_tokenizer_json(&self, dir: &Path) -> Result<()> {
        let file_path = if dir.extension().map_or(false, |ext| ext == "json") {
            dir.to_path_buf()
        } else {
            dir.join("tokenizer.json")
        };
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let added_tokens = serde_json::json!([
            { "id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
        ]);
        let mut vocab_map = serde_json::Map::new();
        for (i, tok) in self.id_to_token.iter().enumerate() {
            vocab_map.insert(tok.clone(), serde_json::Value::Number(serde_json::Number::from(i)));
        }
        let root = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "WhitespaceSplit"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab_map,
                "unk_token": "<unk>"
            }
        });
        let json = serde_json::to_string_pretty(&root)?;
        std::fs::write(&file_path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_vocab_size_is_4096() {
        let mut model = KiloVocab::empty();
        let needed = TARGET_LEARNED_VOCAB;
        for i in 0..needed {
            model.learned_vocab.push(format!("word_{i}"));
        }
        model.rebuild_indices();
        assert_eq!(model.vocab_size, 4096);
        assert_eq!(model.id_to_token.len(), 4096);
        assert_eq!(model.token_to_id.len(), 4096);
    }

    #[test]
    fn test_special_tokens_and_placeholders() {
        let model = KiloVocab::empty();
        assert_eq!(model.token_to_id("<pad>"), Some(PAD_ID));
        assert_eq!(model.token_to_id("<s>"), Some(BOS_ID));
        assert_eq!(model.token_to_id("</s>"), Some(EOS_ID));
        assert_eq!(model.token_to_id("<unk>"), Some(UNK_ID));

        assert_eq!(model.token_to_id("<NAME_0>"), Some(28)); // 4 special + 24 punct = 28
        assert_eq!(model.token_to_id("<NUM_0>"), Some(28 + 48));
        assert_eq!(model.token_to_id("<SLOT_0>"), Some(28 + 48 + 20));
    }

    #[test]
    fn test_distinct_words_preserved_no_thesaurus_collapse() {
        let mut word_counts = HashMap::new();
        word_counts.insert("world".to_string(), 1000);
        word_counts.insert("universe".to_string(), 900);
        word_counts.insert("galaxy".to_string(), 800);
        word_counts.insert("planet".to_string(), 700);

        let phrase_counts = HashMap::new();
        let model = KiloVocab::learn_from_corpus(&word_counts, &phrase_counts, 4096, 50);

        let id_world = model.token_to_id("world").expect("world must exist");
        let id_universe = model.token_to_id("universe").expect("universe must exist");
        let id_galaxy = model.token_to_id("galaxy").expect("galaxy must exist");
        let id_planet = model.token_to_id("planet").expect("planet must exist");

        // Crucial requirement: world and universe are NOT collapsed!
        assert_ne!(id_world, id_universe);
        assert_ne!(id_world, id_galaxy);
        assert_ne!(id_universe, id_planet);
    }

    #[test]
    fn test_multiword_phrase_greedy_longest_match() {
        let mut word_counts = HashMap::new();
        word_counts.insert("well".to_string(), 500);
        word_counts.insert("as".to_string(), 600);

        let mut phrase_counts = HashMap::new();
        phrase_counts.insert("as well as".to_string(), 300);

        let model = KiloVocab::learn_from_corpus(&word_counts, &phrase_counts, 4096, 50);
        assert!(model.token_to_id("as well as").is_some());

        let mut ctx = TranslationContext::new();
        let tokens = model.encode("He likes cats as well as dogs.", &mut ctx).unwrap();

        let phrase_id = model.token_to_id("as well as").unwrap();
        assert!(tokens.contains(&phrase_id), "multi-word phrase 'as well as' should be matched atomically");
    }

    #[test]
    fn test_entity_number_and_slot_tracking() {
        let mut word_counts = HashMap::new();
        word_counts.insert("likes".to_string(), 100);
        word_counts.insert("apples".to_string(), 100);

        let phrase_counts = HashMap::new();
        let model = KiloVocab::learn_from_corpus(&word_counts, &phrase_counts, 4096, 50);

        let mut ctx = TranslationContext::new();
        let tokens = model.encode("Alice bought 42 exotic mangosteens.", &mut ctx).unwrap();

        // Alice -> <NAME_0>
        assert_eq!(ctx.entities, vec!["Alice"]);
        assert!(tokens.contains(&model.token_to_id("<NAME_0>").unwrap()));

        // 42 -> <NUM_0>
        assert_eq!(ctx.numbers, vec!["42"]);
        assert!(tokens.contains(&model.token_to_id("<NUM_0>").unwrap()));

        // exotic / mangosteens -> <SLOT_0>, <SLOT_1>
        assert!(ctx.slots.contains(&"exotic".to_string()));
        assert!(ctx.slots.contains(&"mangosteens".to_string()));

        // Decode with context reconstructs raw text
        let decoded = model.decode_with_context(&tokens, &ctx);
        assert!(decoded.contains("Alice"));
        assert!(decoded.contains("42"));
        assert!(decoded.contains("mangosteens"));
    }

    #[test]
    fn test_json_serialization_roundtrip() {
        let mut word_counts = HashMap::new();
        word_counts.insert("hello".to_string(), 10);
        word_counts.insert("world".to_string(), 10);
        let phrase_counts = HashMap::new();

        let model = KiloVocab::learn_from_corpus(&word_counts, &phrase_counts, 4096, 20);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_kilo_vocab.json");

        model.save_to_json(&path).unwrap();
        let loaded = KiloVocab::load_from_json(&path).unwrap();

        assert_eq!(loaded.vocab_size, model.vocab_size);
        assert_eq!(loaded.token_to_id("hello"), model.token_to_id("hello"));
        assert_eq!(loaded.token_to_id("world"), model.token_to_id("world"));

        let _ = std::fs::remove_file(path);
    }
}
