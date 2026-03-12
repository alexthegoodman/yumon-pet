/// Character-level tokenizer for Yumon's language brain.
///
/// We use character-level tokenisation because:
///   1. Vocabulary is small and fixed — no OOV issues.
///   2. Works well at tweet scale (≤140 tokens output).
///   3. SimpleWiki provides enough character diversity.
///
/// Special tokens:
///   <PAD> = 0   — padding
///   <BOS> = 1   — begin of sequence
///   <EOS> = 2   — end of sequence
///   <UNK> = 3   — unknown character

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

pub const PAD_TOKEN: usize = 0;
pub const BOS_TOKEN: usize = 1;
pub const EOS_TOKEN: usize = 2;
pub const UNK_TOKEN: usize = 3;
const RESERVED:      usize = 4;

#[derive(Debug, Serialize, Deserialize)]
pub struct Tokenizer {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
    pub vocab_size:  usize,
}

impl Tokenizer {
    /// Build vocabulary from a corpus string.
    pub fn build_from_text(text: &str, max_vocab: usize) -> Self {
        // Count character frequencies
        let mut freq: HashMap<char, usize> = HashMap::new();
        for c in text.chars() {
            if c.is_ascii() { // ASCII only for simplicity
                *freq.entry(c).or_insert(0) += 1;
            }
        }

        // Sort by frequency descending, take up to max_vocab - RESERVED
        let mut chars: Vec<(char, usize)> = freq.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));
        chars.truncate(max_vocab.saturating_sub(RESERVED));

        let mut idx_to_char = vec!['\0', '\x01', '\x02', '\x03']; // PAD BOS EOS UNK
        let mut char_to_idx: HashMap<char, usize> = HashMap::new();
        char_to_idx.insert('\0', PAD_TOKEN);
        char_to_idx.insert('\x01', BOS_TOKEN);
        char_to_idx.insert('\x02', EOS_TOKEN);
        char_to_idx.insert('\x03', UNK_TOKEN);

        for (c, _) in &chars {
            let idx = idx_to_char.len();
            idx_to_char.push(*c);
            char_to_idx.insert(*c, idx);
        }

        let vocab_size = idx_to_char.len();
        Self { char_to_idx, idx_to_char, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter(|c| c.is_ascii())
            .map(|c| *self.char_to_idx.get(&c).unwrap_or(&UNK_TOKEN))
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter(|&&t| t != PAD_TOKEN && t != BOS_TOKEN && t != EOS_TOKEN)
            .map(|&t| self.idx_to_char.get(t).copied().unwrap_or('?'))
            .collect()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}
