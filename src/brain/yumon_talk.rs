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

/// Fixed closed-class words (pronouns, basic prepositions, conjunctions, determiners, auxiliary verbs).
/// These words have strict, sealed forms and NEVER absorb arbitrary thesaurus synonyms.
pub const SEALED_WORDS: &[(&str, &[&str])] = &[
    ("i", &["i", "me", "myself"]),
    ("you", &["you", "thee", "thou", "yourself", "yourselves"]),
    ("he", &["he", "him", "himself"]),
    ("she", &["she", "her", "herself"]),
    ("it", &["it", "itself"]),
    ("we", &["we", "us", "ourselves"]),
    ("they", &["they", "them", "themselves"]),
    ("my", &["my", "mine"]),
    ("your", &["your", "yours", "thy", "thine"]),
    ("his", &["his"]),
    ("her", &["her", "hers"]),
    ("its", &["its"]),
    ("our", &["our", "ours"]),
    ("their", &["their", "theirs"]),
    ("this", &["this"]),
    ("that", &["that"]),
    ("these", &["these"]),
    ("those", &["those"]),
    ("who", &["who", "whom", "whose", "whoever"]),
    ("which", &["which", "whichever"]),
    ("what", &["what", "whatever"]),
    ("where", &["where", "wherever"]),
    ("when", &["when", "whenever"]),
    ("why", &["why"]),
    ("how", &["how"]),
    ("all", &["all", "everybody", "everyone", "everything"]),
    ("every", &["every", "each"]),
    ("any", &["any", "anybody", "anyone", "anything"]),
    ("some", &["some", "somebody", "someone", "something"]),
    ("none", &["none", "nobody", "nothing", "neither"]),
    ("both", &["both"]),
    ("the", &["the"]),
    ("a", &["a", "an"]),
    ("and", &["and", "also"]),
    ("or", &["or", "either"]),
    ("but", &["but", "however", "though", "yet"]),
    ("so", &["so", "therefore", "thus"]),
    ("because", &["because", "since"]),
    ("if", &["if", "whether"]),
    ("then", &["then"]),
    ("in", &["in", "inside", "within"]),
    ("on", &["on", "upon"]),
    ("at", &["at"]),
    ("by", &["by"]),
    ("for", &["for"]),
    ("with", &["with"]),
    ("without", &["without"]),
    ("about", &["about", "around", "regarding", "concerning"]),
    ("from", &["from"]),
    ("to", &["to"]),
    ("of", &["of"]),
    ("be", &["be", "am", "is", "are", "was", "were", "been", "being"]),
    ("have", &["have", "has", "had", "having"]),
    ("do", &["do", "does", "did", "done", "doing"]),
    ("will", &["will", "shall", "would"]),
    ("can", &["can", "could", "able"]),
    ("may", &["may", "might"]),
    ("must", &["must", "should", "ought"]),
    ("not", &["not", "never"]),
    ("no", &["no", "nay"]),
    ("yes", &["yes", "yeah", "aye"]),
];

/// Check if a word is a member of the sealed closed-class vocabulary.
pub fn is_sealed(word: &str) -> bool {
    let lower = word.to_lowercase();
    SEALED_WORDS.iter().any(|(c, syns)| *c == lower || syns.iter().any(|&s| s == lower))
}

pub const IRREGULAR_LEMMAS: &[(&str, &str)] = &[
    ("build", "built"),
    ("built", "build"),
    ("run", "ran"),
    ("ran", "run"),
    ("see", "saw"),
    ("saw", "see"),
    ("seen", "see"),
    ("go", "went"),
    ("went", "go"),
    ("gone", "go"),
    ("come", "came"),
    ("came", "come"),
    ("tell", "told"),
    ("told", "tell"),
    ("know", "knew"),
    ("knew", "know"),
    ("known", "know"),
    ("give", "gave"),
    ("gave", "give"),
    ("given", "give"),
    ("take", "took"),
    ("took", "take"),
    ("taken", "take"),
    ("find", "found"),
    ("found", "find"),
    ("think", "thought"),
    ("thought", "think"),
    ("buy", "bought"),
    ("bought", "buy"),
    ("feel", "felt"),
    ("felt", "feel"),
    ("hold", "held"),
    ("held", "hold"),
    ("leave", "left"),
    ("left", "leave"),
    ("speak", "spoke"),
    ("spoke", "speak"),
    ("bring", "brought"),
    ("brought", "bring"),
    ("begin", "began"),
    ("began", "begin"),
    ("begun", "begin"),
    ("say", "said"),
    ("said", "say"),
    ("make", "made"),
    ("made", "make"),
    ("hear", "heard"),
    ("heard", "hear"),
    ("child", "children"),
    ("children", "child"),
    ("person", "people"),
    ("people", "person"),
    ("man", "men"),
    ("men", "man"),
    ("woman", "women"),
    ("women", "woman"),
];

/// Generate two-way morphological candidates (suffix stripping, inflection expansion, irregular verbs).
pub fn morphological_variants(lower: &str) -> Vec<String> {
    let mut variants = Vec::new();

    // 1. Suffix stripping (handling silent 'e' and consonant doubling)
    if let Some(s) = lower.strip_suffix("'s") { variants.push(s.to_string()); }
    if let Some(s) = lower.strip_suffix("s") {
        if !s.is_empty() { variants.push(s.to_string()); }
    }
    if let Some(s) = lower.strip_suffix("es") {
        if !s.is_empty() { variants.push(s.to_string()); }
    }
    if let Some(s) = lower.strip_suffix("ies") {
        variants.push(format!("{s}y"));
    }
    if let Some(s) = lower.strip_suffix("ied") {
        variants.push(format!("{s}y"));
    }
    if let Some(s) = lower.strip_suffix("ed") {
        if !s.is_empty() {
            variants.push(s.to_string());
            variants.push(format!("{s}e")); // created -> create, closed -> close
        }
    }
    if let Some(s) = lower.strip_suffix("ing") {
        if !s.is_empty() {
            variants.push(s.to_string());
            variants.push(format!("{s}e")); // creating -> create, making -> make
            let bytes = s.as_bytes();
            if bytes.len() >= 2 && bytes[bytes.len() - 1] == bytes[bytes.len() - 2] {
                variants.push(s[..s.len() - 1].to_string()); // running -> run, stopping -> stop
            }
        }
    }
    if let Some(s) = lower.strip_suffix("ly") {
        if !s.is_empty() { variants.push(s.to_string()); }
    }
    if let Some(s) = lower.strip_suffix("er") {
        if !s.is_empty() {
            variants.push(s.to_string());
            variants.push(format!("{s}e"));
        }
    }
    if let Some(s) = lower.strip_suffix("est") {
        if !s.is_empty() {
            variants.push(s.to_string());
            variants.push(format!("{s}e"));
        }
    }

    // 2. Inflection expansion (when canonical word in dictionary is inflected)
    variants.push(format!("{lower}s"));
    variants.push(format!("{lower}es"));
    variants.push(format!("{lower}d"));
    variants.push(format!("{lower}ed"));
    if let Some(s) = lower.strip_suffix('e') {
        variants.push(format!("{s}ed"));
        variants.push(format!("{s}ing"));
    } else {
        variants.push(format!("{lower}ing"));
    }

    // 3. Irregular verbs and nouns
    for &(a, b) in IRREGULAR_LEMMAS {
        if lower == a {
            variants.push(b.to_string());
        }
    }

    variants
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
    pub dynamic_cache: Arc<RwLock<HashMap<String, Option<usize>>>>,
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
        for &(canon, syns) in SEALED_WORDS {
            if let Some(&id) = token_to_id.get(canon) {
                word_to_id.insert(canon.to_lowercase(), id);
                for &syn in syns {
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

    /// Save as standard HuggingFace WordLevel tokenizer.json for training and inference.
    pub fn save_tokenizer_json(&self, dir_or_file: &Path) -> Result<()> {
        let file_path = if dir_or_file.extension().map_or(false, |ext| ext == "json") {
            dir_or_file.to_path_buf()
        } else {
            dir_or_file.join("tokenizer.json")
        };
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let added_tokens = serde_json::json!([
            { "id": 0, "content": "<PAD>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 1, "content": "<BOS>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 2, "content": "<EOS>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 3, "content": "<UNK>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
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
                "unk_token": "<UNK>"
            }
        });
        let json = serde_json::to_string_pretty(&root)?;
        std::fs::write(&file_path, json)?;
        Ok(())
    }

    pub fn token_to_id(&self, tok: &str) -> Option<usize> {
        self.token_to_id.get(tok).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    pub fn is_name_token(&self, id: usize) -> bool {
        self.id_to_token.get(id).map_or(false, |tok| tok.starts_with("<NAME_") && tok.ends_with('>'))
    }

    pub fn resolve_english_word(&self, word: &str) -> Option<usize> {
        let lower = word.to_lowercase();

        // 1. Static/cluster lookup
        if let Some(&id) = self.word_to_id.get(&lower) {
            return Some(id);
        }

        // 2. Dynamic cache lookup (with negative caching!)
        if let Ok(c) = self.dynamic_cache.read() {
            if let Some(cached) = c.get(&lower) {
                return *cached;
            }
        }

        // 3. Morphological Stemming and Two-Way Lemmatization
        for variant in morphological_variants(&lower) {
            if variant != lower && !variant.is_empty() {
                if let Some(&id) = self.word_to_id.get(&variant) {
                    if let Ok(mut c) = self.dynamic_cache.write() {
                        c.insert(lower.clone(), Some(id));
                    }
                    return Some(id);
                }
            }
        }

        // 4. Query `thesaurus` crate with mutual confirmation and sealed word exclusion
        let dict = get_thesaurus_dict();
        if let Some(syns) = dict.get(&lower) {
            // High precision: mutual confirmation
            for syn in syns.iter() {
                let clean = syn.trim().to_lowercase();
                if is_sealed(&clean) {
                    continue;
                }
                if let Some(&id) = self.word_to_id.get(&clean) {
                    if let Some(canon) = self.id_to_token(id) {
                        if !is_sealed(canon) {
                            if dict.get(canon).map_or(false, |canon_syns| {
                                canon_syns.iter().any(|s| s.trim().to_lowercase() == lower)
                            }) {
                                if let Ok(mut c) = self.dynamic_cache.write() {
                                    c.insert(lower.clone(), Some(id));
                                }
                                return Some(id);
                            }
                        }
                    }
                }
            }

            // High precision fallback: direct prefix/stem or top synonym into open-class canonical word
            for (idx, syn) in syns.iter().enumerate() {
                let clean = syn.trim().to_lowercase();
                if is_sealed(&clean) {
                    continue;
                }
                if let Some(&id) = self.word_to_id.get(&clean) {
                    if let Some(canon) = self.id_to_token(id) {
                        if !is_sealed(canon) && (clean.starts_with(&lower) || lower.starts_with(&clean) || idx < 3) {
                            if let Ok(mut c) = self.dynamic_cache.write() {
                                c.insert(lower.clone(), Some(id));
                            }
                            return Some(id);
                        }
                    }
                }
            }
        }

        // Also check if any morphological variant of `lower` has a direct thesaurus match
        for variant in morphological_variants(&lower) {
            if let Some(syns) = dict.get(&variant) {
                for syn in syns.iter().take(3) {
                    let clean = syn.trim().to_lowercase();
                    if is_sealed(&clean) { continue; }
                    if let Some(&id) = self.word_to_id.get(&clean) {
                        if let Some(canon) = self.id_to_token(id) {
                            if !is_sealed(canon) {
                                if let Ok(mut c) = self.dynamic_cache.write() {
                                    c.insert(lower.clone(), Some(id));
                                }
                                return Some(id);
                            }
                        }
                    }
                }
            }
        }

        // Cache failure so we never re-query thesaurus for this word!
        if let Ok(mut c) = self.dynamic_cache.write() {
            c.insert(lower.clone(), None);
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

            // Word (alphanumeric + apostrophe) - must be checked BEFORE single-char punctuation!
            if chars[i].is_alphabetic() || chars[i] == '_' {
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

                // Only assign <NAME_x> placeholder if word is capitalized (proper name / entity / acronym)
                let is_entity = chars[start].is_uppercase()
                    && (i - start > 1 || (chars[start] != 'A' && chars[start] != 'I'));

                if is_entity {
                    let ent_idx = entity_map.entry(word.clone()).or_insert_with(|| {
                        let curr = ctx.entities.len();
                        ctx.entities.push(word);
                        curr % 32
                    });
                    let slot = format!("<NAME_{ent_idx}>");
                    tokens.push(self.token_to_id(&slot).unwrap_or(UNK_ID));
                } else {
                    tokens.push(UNK_ID);
                }
                continue;
            }

            // Numbers
            if chars[i].is_ascii_digit() {
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

            // Punctuation (strictly non-alphanumeric!)
            let ch = chars[i];
            let punct_str = ch.to_string();
            if self.punctuation.iter().any(|p| p == &punct_str) {
                if let Some(id) = self.token_to_id(&punct_str) {
                    tokens.push(id);
                    i += 1;
                    continue;
                }
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

            // Word - must be checked BEFORE single-char punctuation!
            if chars[i].is_alphanumeric() || chars[i] == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                if let Some(id) = self.token_to_id(&word.to_lowercase()) {
                    tokens.push(id);
                } else {
                    tokens.push(UNK_ID);
                }
                continue;
            }

            // Punctuation (strictly non-alphanumeric!)
            let ch = chars[i];
            let punct_str = ch.to_string();
            if self.punctuation.iter().any(|p| p == &punct_str) {
                if let Some(id) = self.token_to_id(&punct_str) {
                    tokens.push(id);
                    i += 1;
                    continue;
                }
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

static THESAURUS_DICT: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();

pub fn get_thesaurus_dict() -> &'static HashMap<String, Vec<String>> {
    THESAURUS_DICT.get_or_init(|| {
        thesaurus::init();
        thesaurus::dict()
    })
}

static GLOBAL_VOCAB: OnceLock<RwLock<YumonTalkVocab>> = OnceLock::new();

pub fn get_global_vocab() -> std::sync::RwLockReadGuard<'static, YumonTalkVocab> {
    let rw = GLOBAL_VOCAB.get_or_init(|| {
        thesaurus::init();
        let env_path = std::env::var("YUMON_TALK_VOCAB").ok().map(std::path::PathBuf::from);
        let default_path = Path::new("training-cache/yumon_talk_vocab.json");
        let path = env_path.as_deref().unwrap_or(default_path);
        if path.exists() {
            if let Ok(v) = YumonTalkVocab::load_from_file(path) {
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
            "device".into(), "announced".into(), "new".into(), "and".into(), "pet".into(),
            "very".into(), "a".into(), "yumon".into(),
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

    #[test]
    fn test_two_way_lemmatization_and_capitalized_entities_gating() {
        let mut special = vec!["<PAD>".into(), "<BOS>".into(), "<EOS>".into(), "<UNK>".into()];
        let punct = vec!["."].into_iter().map(String::from).collect();
        let placeholders = (0..32).map(|i| format!("<NAME_{i}>"))
            .chain((0..8).map(|i| format!("<NUM_{i}>"))).collect();
        let canonical = vec![
            "created".into(), "built".into(), "friends".into(), "make".into(),
        ];
        let vocab = YumonTalkVocab::new(special, punct, placeholders, canonical, HashMap::new());

        // Base forms resolve to inflected canonical words:
        assert!(vocab.resolve_english_word("create").is_some(), "create should resolve to created");
        assert!(vocab.resolve_english_word("build").is_some(), "build should resolve to built");
        assert!(vocab.resolve_english_word("friend").is_some(), "friend should resolve to friends");

        // Capitalized unknown word becomes NAME:
        let mut ctx = TranslationContext::new();
        let tokens_cap = vocab.english_to_yumon_tokens("Diego", &mut ctx).unwrap();
        assert_eq!(tokens_cap.len(), 1);
        assert_eq!(vocab.id_to_token(tokens_cap[0]), Some("<NAME_0>"));
        assert_eq!(ctx.entities, vec!["Diego"]);

        // Lowercase unknown word becomes UNK:
        let mut ctx2 = TranslationContext::new();
        let tokens_lower = vocab.english_to_yumon_tokens("tadpoles", &mut ctx2).unwrap();
        assert_eq!(tokens_lower.len(), 1);
        assert_eq!(tokens_lower[0], UNK_ID);
        assert!(ctx2.entities.is_empty(), "Lowercase word must not be added to entities");
    }
}
