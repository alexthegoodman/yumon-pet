use anyhow::{Context, Result};

use crate::brain::mdx::{ChatBlock, HandcraftedChats, Memory};
 
pub fn load_distilled_chats(dict_path: &str, soft_limit: i32) -> Result<HandcraftedChats> {
    println!("📖 Loading distilled: {dict_path}");
 
    let content = std::fs::read_to_string(dict_path)
        .with_context(|| format!("Failed to read file: {dict_path}"))?;
 
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());
 
    // Find the index of the "conversations" column
    let headers = rdr.headers()?.clone();
    let conv_idx = headers
        .iter()
        .position(|h| h == "conversations")
        .context("No 'conversations' column found in CSV")?;
 
    let mut blocks = Vec::new();
    let mut x = 0;
    for result in rdr.records() {
        let record = result?;
        let raw = match record.get(conv_idx) {
            Some(v) => v,
            None => continue,
        };
 
        let memories = parse_conversations(raw);
        if !memories.is_empty() {
            blocks.push(ChatBlock { memories });
        }

        if (x > soft_limit) {
            break;
        }

        x = x + 1;
    }
 
    println!(
        "✅ Loaded {} blocks ({} memories total)",
        blocks.len(),
        blocks.iter().map(|b| b.memories.len()).sum::<usize>()
    );
 
    Ok(HandcraftedChats { blocks })
}
 
/// Parse the Python-repr conversation string into Memory pairs.
///
/// Each conversation is a list of dicts like:
///   [{'from': 'human', 'value': '...'} {'from': 'gpt', 'value': '...'}]
///
/// We extract consecutive human/gpt pairs as Memories.
fn parse_conversations(raw: &str) -> Vec<Memory> {
    // Pull out all ('from', 'value') pairs using a simple state machine
    // rather than a full Python-literal parser.
    let turns = extract_turns(raw);
 
    let mut memories = Vec::new();
    let mut i = 0;
    while i + 1 < turns.len() {
        let (from_a, val_a) = &turns[i];
        let (from_b, val_b) = &turns[i + 1];
 
        if from_a == "human" && from_b == "gpt" {
            memories.push(Memory {
                human: val_a.clone(),
                bot: val_b.clone(),
            });
            i += 2;
        } else {
            // Skip unexpected ordering and try to re-sync
            i += 1;
        }
    }
 
    memories
}
 
/// Extract (from, value) pairs from the raw Python-repr string.
///
/// Strategy: scan for `'from': '<role>'` and the following `'value': '<text>'`.
/// Uses a tiny hand-rolled extractor that handles escaped quotes inside values.
fn extract_turns(raw: &str) -> Vec<(String, String)> {
    let mut turns = Vec::new();
    let bytes = raw.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
 
    while pos < len {
        // Find next occurrence of "'from': '"
        let Some(from_start) = find_substr(raw, pos, "'from': '") else {
            break;
        };
        pos = from_start + "'from': '".len();
 
        // Read until closing unescaped single quote
        let (role, next) = read_quoted_value(raw, pos);
        pos = next;
 
        // Find the 'value': ' that follows this 'from'
        let Some(val_start) = find_substr(raw, pos, "'value': '") else {
            break;
        };
        pos = val_start + "'value': '".len();
 
        let (value, next) = read_quoted_value(raw, pos);
        pos = next;
 
        turns.push((role, value));
    }
 
    turns
}
 
/// Find the byte offset of `needle` in `haystack` starting from `from`.
fn find_substr(haystack: &str, from: usize, needle: &str) -> Option<usize> {
    haystack[from..].find(needle).map(|i| from + i)
}
 
/// Read characters from `start` until an unescaped `'` is found.
/// Returns (extracted_string, position_after_closing_quote).
fn read_quoted_value(s: &str, start: usize) -> (String, usize) {
    let chars: Vec<char> = s[start..].chars().collect();
    let mut result = String::new();
    let mut i = 0;
 
    while i < chars.len() {
        match chars[i] {
            '\\' if i + 1 < chars.len() => {
                // Unescape common sequences
                match chars[i + 1] {
                    '\'' => result.push('\''),
                    'n' => result.push('\n'),
                    't' => result.push('\t'),
                    '\\' => result.push('\\'),
                    other => {
                        result.push('\\');
                        result.push(other);
                    }
                }
                i += 2;
            }
            '\'' => {
                i += 1;
                break; // closing quote
            }
            c => {
                result.push(c);
                i += 1;
            }
        }
    }
 
    // Convert char-index back to byte offset
    let byte_offset: usize = s[start..]
        .char_indices()
        .nth(i)
        .map(|(b, _)| b)
        .unwrap_or(s.len() - start);
 
    (result, start + byte_offset)
}