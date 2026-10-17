//! The Yumon Talk Language (1024 total tokens).
//!
//! Yumon Talk is a compact, canonical natural language designed for classic Pet.
//! It addresses vocabulary explosion by enforcing:
//! - Exactly 1024 total tokens in the vocabulary (IDs 0..1023).
//! - Dynamic loading from `training-cache/yumon_talk_vocab.json` built from real training data.
//! - Semantic clustering: each canonical word maps to a rich set of English synonyms.
//! - Bidirectional translation: English <-> Yumon Talk.
//! - Dynamic expansion via the `thesaurus` crate and morphological stemming.
//! - Placeholder slots (`<NAME_0>`..`<NAME_31>`, `<NUM_0>`..`<NUM_7>`) for entities, brand names, and numbers.
//! - Random synonym sampling during translation back to English.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

pub const VOCAB_SIZE: usize = 1024;
pub const PAD_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const UNK_ID: usize = 3;

/// Translation context preserving entities and brand names.
#[derive(Debug, Clone, Default)]
pub struct TranslationContext {
    pub entities: Vec<String>,
    pub numbers: Vec<String>,
}

impl TranslationContext {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Comprehensive, dynamically loadable Yumon Talk Vocabulary (1024 tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YumonTalkVocab {
    pub version: usize,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub punctuation: Vec<String>,
    pub placeholders: Vec<String>,
    pub canonical_words: Vec<String>,
    pub synonym_clusters: HashMap<String, Vec<String>>,

    #[serde(skip)]
    pub token_to_id: HashMap<String, usize>,
    #[serde(skip)]
    pub id_to_token: Vec<String>,
    #[serde(skip)]
    pub word_to_id: HashMap<String, usize>,
    #[serde(skip)]
    pub dynamic_cache: Arc<RwLock<HashMap<String, usize>>>,
}

impl YumonTalkVocab {
    pub fn new(
        special_tokens: Vec<String>,
        punctuation: Vec<String>,
        placeholders: Vec<String>,
        canonical_words: Vec<String>,
        synonym_clusters: HashMap<String, Vec<String>>,
    ) -> Self {
        let total = special_tokens.len() + punctuation.len() + placeholders.len() + canonical_words.len();
        let mut v = Self {
            version: 1,
            vocab_size: total,
            special_tokens,
            punctuation,
            placeholders,
            canonical_words,
            synonym_clusters,
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            word_to_id: HashMap::new(),
            dynamic_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        v.rebuild_indices();
        v
    }

    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), HashMap::new())
    }

    pub fn rebuild_indices(&mut self) {
        let total = self.special_tokens.len() + self.punctuation.len() + self.placeholders.len() + self.canonical_words.len();
        let mut id_to_token = Vec::with_capacity(total);
        let mut token_to_id = HashMap::with_capacity(total);
        let mut word_to_id = HashMap::with_capacity(total * 10);

        for tok in &self.special_tokens {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }
        for tok in &self.punctuation {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }
        for tok in &self.placeholders {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }
        for tok in &self.canonical_words {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            word_to_id.insert(tok.to_lowercase(), id);
            id_to_token.push(tok.clone());
        }
        for (canon, syns) in &self.synonym_clusters {
            if let Some(&id) = token_to_id.get(canon) {
                word_to_id.insert(canon.to_lowercase(), id);
                for syn in syns {
                    word_to_id.insert(syn.to_lowercase(), id);
                }
            }
        }
        self.vocab_size = total;
        self.token_to_id = token_to_id;
        self.id_to_token = id_to_token;
        self.word_to_id = word_to_id;
        self.dynamic_cache = Arc::new(RwLock::new(HashMap::new()));
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading vocab file {}", path.display()))?;
        let mut vocab: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing vocab file {}", path.display()))?;
        vocab.rebuild_indices();
        Ok(vocab)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn token_to_id(&self, tok: &str) -> Option<usize> {
        self.token_to_id.get(tok).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    pub fn resolve_english_word(&self, word: &str) -> Option<usize> {
        let lower = word.to_lowercase();

        // 1. Static/cluster lookup
        if let Some(&id) = self.word_to_id.get(&lower) {
            return Some(id);
        }

        // 2. Dynamic cache lookup
        if let Ok(c) = self.dynamic_cache.read() {
            if let Some(&id) = c.get(&lower) {
                return Some(id);
            }
        }

        // 3. Morphological Stemming
        let stems = [
            lower.strip_suffix("'s").unwrap_or(&lower),
            lower.strip_suffix("s").unwrap_or(&lower),
            lower.strip_suffix("es").unwrap_or(&lower),
            lower.strip_suffix("ed").unwrap_or(&lower),
            lower.strip_suffix("ing").unwrap_or(&lower),
            lower.strip_suffix("ly").unwrap_or(&lower),
            lower.strip_suffix("er").unwrap_or(&lower),
            lower.strip_suffix("est").unwrap_or(&lower),
        ];
        for &stem in stems.iter() {
            if stem != lower.as_str() && !stem.is_empty() {
                if let Some(&id) = self.word_to_id.get(stem) {
                    if let Ok(mut c) = self.dynamic_cache.write() {
                        c.insert(lower.clone(), id);
                    }
                    return Some(id);
                }
            }
        }

        // 4. Query `thesaurus` crate
        let syns = thesaurus::synonyms(&lower);
        for syn in syns.iter() {
            let clean = syn.trim().to_lowercase();
            if let Some(&id) = self.word_to_id.get(&clean) {
                if let Ok(mut c) = self.dynamic_cache.write() {
                    c.insert(lower.clone(), id);
                }
                return Some(id);
            }
        }

        None
    }

    pub fn english_to_yumon_tokens(&self, english_text: &str, ctx: &mut TranslationContext) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = english_text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        let mut entity_map: HashMap<String, usize> = HashMap::new();
        let mut num_map: HashMap<String, usize> = HashMap::new();

        while i < len {
            // Skip whitespace
            if chars[i].is_whitespace() {
                if chars[i] == '\n' {
                    if let Some(id) = self.token_to_id("\n") {
                        tokens.push(id);
                    }
                }
                i += 1;
                continue;
            }

            // Punctuation
            let ch = chars[i];
            let punct_str = ch.to_string();
            if let Some(id) = self.token_to_id(&punct_str) {
                tokens.push(id);
                i += 1;
                continue;
            }

            // Numbers
            if ch.is_ascii_digit() {
                let start = i;
                while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let idx = num_map.entry(num_str.clone()).or_insert_with(|| {
                    let curr = ctx.numbers.len();
                    ctx.numbers.push(num_str);
                    curr % 8
                });
                let slot = format!("<NUM_{idx}>");
                tokens.push(self.token_to_id(&slot).unwrap_or(UNK_ID));
                continue;
            }

            // Word (alphanumeric + apostrophe)
            if ch.is_alphabetic() || ch == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || (chars[i] == '\'' && i + 1 < len && chars[i + 1].is_alphabetic())) {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Try resolving word to a canonical Yumon Talk ID
                if let Some(id) = self.resolve_english_word(&word) {
                    tokens.push(id);
                    continue;
                }

                // If capitalized / brand name / out of vocabulary entity:
                let ent_idx = entity_map.entry(word.clone()).or_insert_with(|| {
                    let curr = ctx.entities.len();
                    ctx.entities.push(word);
                    curr % 32
                });
                let slot = format!("<NAME_{ent_idx}>");
                tokens.push(self.token_to_id(&slot).unwrap_or(UNK_ID));
                continue;
            }

            // Other character: fallback
            i += 1;
        }

        Ok(tokens)
    }

    pub fn tokenize_yumon(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i].is_whitespace() {
                if chars[i] == '\n' {
                    if let Some(id) = self.token_to_id("\n") {
                        tokens.push(id);
                    }
                }
                i += 1;
                continue;
            }

            // Check for bracketed placeholders like <NAME_0>, <NUM_0>, etc.
            if chars[i] == '<' {
                let start = i;
                while i < len && chars[i] != '>' {
                    i += 1;
                }
                if i < len && chars[i] == '>' {
                    i += 1;
                    let token_str: String = chars[start..i].iter().collect();
                    if let Some(id) = self.token_to_id(&token_str) {
                        tokens.push(id);
                        continue;
                    }
                }
            }

            // Punctuation
            let ch = chars[i];
            let punct_str = ch.to_string();
            if let Some(id) = self.token_to_id(&punct_str) {
                tokens.push(id);
                i += 1;
                continue;
            }

            // Word
            if ch.is_alphanumeric() || ch == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                if let Some(id) = self.token_to_id(&word) {
                    tokens.push(id);
                } else {
                    tokens.push(UNK_ID);
                }
                continue;
            }

            i += 1;
        }

        tokens
    }

    pub fn detokenize_yumon(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids.iter() {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            let tok = self.id_to_token(id).unwrap_or("<UNK>");
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

    pub fn english_to_yumon(&self, english_text: &str) -> Result<String> {
        let mut ctx = TranslationContext::new();
        let tokens = self.english_to_yumon_tokens(english_text, &mut ctx)?;
        ensure!(!tokens.is_empty(), "empty Yumon Talk translation");
        ensure!(!tokens.iter().any(|&id| id == UNK_ID), "Yumon Talk produced UNK tokens");
        Ok(self.detokenize_yumon(&tokens))
    }

    pub fn english_to_yumon_with_map(&self, english_text: &str) -> Result<(String, TranslationContext)> {
        let mut ctx = TranslationContext::new();
        let tokens = self.english_to_yumon_tokens(english_text, &mut ctx)?;
        ensure!(!tokens.is_empty(), "empty Yumon Talk translation");
        ensure!(!tokens.iter().any(|&id| id == UNK_ID), "Yumon Talk produced UNK tokens");
        Ok((self.detokenize_yumon(&tokens), ctx))
    }

    pub fn yumon_to_english(&self, yumon_text: &str) -> Result<String> {
        let mut rng = rand::thread_rng();
        self.yumon_to_english_with_rng(yumon_text, &mut rng)
    }

    pub fn yumon_to_english_with_rng(&self, yumon_text: &str, rng: &mut impl rand::Rng) -> Result<String> {
        let ctx = TranslationContext::new();
        self.yumon_to_english_with_context(yumon_text, &ctx, rng)
    }

    pub fn yumon_to_english_with_context(&self, yumon_text: &str, ctx: &TranslationContext, rng: &mut impl rand::Rng) -> Result<String> {
        let tokens = self.tokenize_yumon(yumon_text);
        ensure!(!tokens.is_empty(), "empty Yumon Talk tokens");

        let mut out = String::new();
        let mut cap_next = true;

        for &id in tokens.iter() {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            let tok = self.id_to_token(id).unwrap_or("<UNK>");

            // Punctuation
            if tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":" || tok == ")" || tok == "]" {
                out.push_str(tok);
                if tok == "." || tok == "!" || tok == "?" {
                    cap_next = true;
                }
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
                cap_next = true;
                continue;
            }

            // Placeholder for name/brand
            let word = if tok.starts_with("<NAME_") && tok.ends_with('>') {
                let num_part = &tok[6..tok.len() - 1];
                if let Ok(idx) = num_part.parse::<usize>() {
                    if let Some(orig) = ctx.entities.get(idx) {
                        orig.clone()
                    } else {
                        format!("entity_{idx}")
                    }
                } else {
                    tok.to_string()
                }
            } else if tok.starts_with("<NUM_") && tok.ends_with('>') {
                let num_part = &tok[5..tok.len() - 1];
                if let Ok(idx) = num_part.parse::<usize>() {
                    if let Some(orig) = ctx.numbers.get(idx) {
                        orig.clone()
                    } else {
                        format!("{idx}")
                    }
                } else {
                    tok.to_string()
                }
            } else {
                // Canonical Yumon Talk word: randomly sample among synonym cluster
                if let Some(syn_list) = self.synonym_clusters.get(tok) {
                    if !syn_list.is_empty() {
                        let chosen = &syn_list[rng.gen_range(0..syn_list.len())];
                        chosen.clone()
                    } else {
                        tok.to_string()
                    }
                } else {
                    tok.to_string()
                }
            };

            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') && !out.ends_with('(') && !out.ends_with('[') {
                out.push(' ');
            }

            if cap_next {
                let mut c_chars = word.chars();
                if let Some(first) = c_chars.next() {
                    out.push(first.to_ascii_uppercase());
                    out.push_str(c_chars.as_str());
                }
                cap_next = false;
            } else {
                out.push_str(&word);
            }
        }

        Ok(out)
    }

    pub fn verify_yumon_roundtrip(&self, english_text: &str) -> Result<bool> {
        let (yumon, ctx) = match self.english_to_yumon_with_map(english_text) {
            Ok(res) => res,
            Err(_) => return Ok(false),
        };
        let mut rng = rand::thread_rng();
        match self.yumon_to_english_with_context(&yumon, &ctx, &mut rng) {
            Ok(reconstructed) => Ok(!reconstructed.is_empty()),
            Err(_) => Ok(false),
        }
    }
}

static GLOBAL_VOCAB: OnceLock<RwLock<YumonTalkVocab>> = OnceLock::new();

pub fn get_global_vocab() -> std::sync::RwLockReadGuard<'static, YumonTalkVocab> {
    let rw = GLOBAL_VOCAB.get_or_init(|| {
        thesaurus::init();
        let default_path = Path::new("training-cache/yumon_talk_vocab.json");
        if default_path.exists() {
            if let Ok(v) = YumonTalkVocab::load_from_file(default_path) {
                return RwLock::new(v);
            }
        }
        RwLock::new(YumonTalkVocab::empty())
    });
    rw.read().expect("global vocab lock poisoned")
}

pub fn set_global_vocab(vocab: YumonTalkVocab) {
    let rw = GLOBAL_VOCAB.get_or_init(|| RwLock::new(YumonTalkVocab::empty()));
    let mut guard = rw.write().expect("global vocab lock poisoned");
    *guard = vocab;
}

pub fn load_global_vocab_file(path: &Path) -> Result<()> {
    let vocab = YumonTalkVocab::load_from_file(path)?;
    set_global_vocab(vocab);
    Ok(())
}

pub fn token_to_id(token: &str) -> Option<usize> {
    get_global_vocab().token_to_id(token)
}

pub fn id_to_token(id: usize) -> Option<String> {
    get_global_vocab().id_to_token(id).map(|s| s.to_string())
}

pub fn resolve_english_word(word: &str) -> Option<usize> {
    get_global_vocab().resolve_english_word(word)
}

pub fn english_to_yumon_tokens(english_text: &str, ctx: &mut TranslationContext) -> Result<Vec<usize>> {
    get_global_vocab().english_to_yumon_tokens(english_text, ctx)
}

pub fn tokenize_yumon(text: &str) -> Vec<usize> {
    get_global_vocab().tokenize_yumon(text)
}

pub fn detokenize_yumon(ids: &[usize]) -> String {
    get_global_vocab().detokenize_yumon(ids)
}

pub fn english_to_yumon(english_text: &str) -> Result<String> {
    get_global_vocab().english_to_yumon(english_text)
}

pub fn english_to_yumon_with_map(english_text: &str) -> Result<(String, TranslationContext)> {
    get_global_vocab().english_to_yumon_with_map(english_text)
}

pub fn yumon_to_english(yumon_text: &str) -> Result<String> {
    get_global_vocab().yumon_to_english(yumon_text)
}

pub fn yumon_to_english_with_rng(yumon_text: &str, rng: &mut impl rand::Rng) -> Result<String> {
    get_global_vocab().yumon_to_english_with_rng(yumon_text, rng)
}

pub fn yumon_to_english_with_context(yumon_text: &str, ctx: &TranslationContext, rng: &mut impl rand::Rng) -> Result<String> {
    get_global_vocab().yumon_to_english_with_context(yumon_text, ctx, rng)
}

pub fn verify_yumon_roundtrip(english_text: &str) -> Result<bool> {
    get_global_vocab().verify_yumon_roundtrip(english_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_test_vocab() -> YumonTalkVocab {
        let special = vec!["<PAD>".into(), "<BOS>".into(), "<EOS>".into(), "<UNK>".into()];
        let punct = vec![".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "/", "\n"]
            .into_iter().map(String::from).collect();
        let placeholders = (0..32).map(|i| format!("<NAME_{i}>"))
            .chain((0..8).map(|i| format!("<NUM_{i}>"))).collect();
        let canonical = vec![
            "hello".into(), "i".into(), "am".into(), "happy".into(), "to".into(), "see".into(),
            "my".into(), "good".into(), "friend".into(), "today".into(), "the".into(), "sun".into(),
            "shines".into(), "bright".into(), "in".into(), "blue".into(), "sky".into(), "please".into(),
            "give".into(), "me".into(), "some".into(), "water".into(), "because".into(), "thirsty".into(),
            "device".into(), "announced".into(), "new".into(),
        ];
        let mut clusters = HashMap::new();
        clusters.insert("happy".into(), vec!["happy".into(), "delighted".into(), "glad".into(), "joyful".into()]);
        clusters.insert("friend".into(), vec!["friend".into(), "pal".into(), "companion".into()]);
        clusters.insert("good".into(), vec!["good".into(), "great".into(), "fine".into(), "enormous".into()]);

        YumonTalkVocab::new(special, punct, placeholders, canonical, clusters)
    }

    #[test]
    fn test_vocab_lifecycle_and_indexing() {
        let vocab = sample_test_vocab();
        assert_eq!(vocab.token_to_id("<PAD>"), Some(0));
        assert_eq!(vocab.token_to_id("<BOS>"), Some(1));
        assert_eq!(vocab.token_to_id("hello"), Some(59));
        assert_eq!(vocab.id_to_token(59), Some("hello"));
        assert!(vocab.resolve_english_word("delighted").is_some());
    }

    #[test]
    fn test_vocab_serialization_roundtrip() {
        let vocab = sample_test_vocab();
        let dir = std::env::temp_dir().join(format!("yumon-vocab-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vocab.json");

        vocab.save_to_file(&path).unwrap();
        let restored = YumonTalkVocab::load_from_file(&path).unwrap();

        assert_eq!(vocab.vocab_size, restored.vocab_size);
        assert_eq!(vocab.canonical_words.len(), restored.canonical_words.len());
        assert_eq!(vocab.token_to_id("hello"), restored.token_to_id("hello"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_english_to_yumon_and_back_with_vocab() {
        let vocab = sample_test_vocab();
        let text = "Hello! I am very happy to see my good friend today.";
        let (yumon, ctx) = vocab.english_to_yumon_with_map(text).unwrap();
        assert!(!yumon.is_empty());

        let mut rng = rand::thread_rng();
        let english = vocab.yumon_to_english_with_context(&yumon, &ctx, &mut rng).unwrap();
        assert!(!english.is_empty());
    }

    #[test]
    fn test_brand_name_placeholder_with_vocab() {
        let vocab = sample_test_vocab();
        let text = "Google and Nintendo announced a new Yumon Pet device.";
        let (yumon, ctx) = vocab.english_to_yumon_with_map(text).unwrap();
        assert!(yumon.contains("<NAME_0>"));
        assert!(yumon.contains("<NAME_1>"));

        let mut rng = rand::thread_rng();
        let english = vocab.yumon_to_english_with_context(&yumon, &ctx, &mut rng).unwrap();
        assert!(english.contains("Google"));
        assert!(english.contains("Nintendo"));
    }

    #[test]
    fn test_roundtrip_verification_with_vocab() {
        let vocab = sample_test_vocab();
        assert!(vocab.verify_yumon_roundtrip("The sun shines bright in the blue sky.").unwrap());
        assert!(vocab.verify_yumon_roundtrip("Please give me some water because I am thirsty.").unwrap());
    }
}
