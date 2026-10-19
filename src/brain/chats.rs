use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use crate::brain::loading::map_batches;

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

    // Preserve the legacy soft cap (including the record at soft_limit + 1).
    let records = rdr
        .records()
        .filter(|record| match record {
            Ok(record) => record.get(conv_idx).is_some(),
            Err(_) => true,
        })
        .take((i64::from(soft_limit).max(-1) + 2) as usize);
    let parsed = map_batches(records, |result| -> Result<Option<ChatBlock>> {
        let record = result?;
        let Some(raw) = record.get(conv_idx) else {
            return Ok(None);
        };
        let memories = parse_conversations(raw);
        Ok((!memories.is_empty()).then_some(ChatBlock { memories }))
    });
    let blocks: Vec<_> = parsed
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

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

/// Load UltraChat conversations from a CSV file or directory of CSV files.
///
/// UltraChat CSV files (`test_gen.csv`, `test_sft.csv`, `train_gen.csv`, `train_sft.csv`)
/// contain columns `prompt`, `prompt_id`, and `messages`. The `messages` column contains
/// a Python/JSON list representation of multi-turn dialogues with keys `content` and `role`.
pub fn load_ultrachat_chats(path: &str, limit: Option<usize>) -> Result<HandcraftedChats> {
    println!("📖 Loading UltraChat: {path}");
    let path_obj = Path::new(path);
    let mut csv_files: Vec<PathBuf> = Vec::new();

    if path_obj.is_dir() {
        for entry in std::fs::read_dir(path_obj)
            .with_context(|| format!("Failed to read directory: {path}"))?
        {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() && p.extension().map_or(false, |ext| ext == "csv") {
                csv_files.push(p);
            }
        }
        csv_files.sort();
    } else if path_obj.is_file() {
        csv_files.push(path_obj.to_path_buf());
    } else {
        anyhow::bail!("UltraChat path does not exist: {path}");
    }

    let mut blocks = Vec::new();

    for file_path in &csv_files {
        if let Some(lim) = limit {
            if blocks.len() >= lim {
                break;
            }
        }

        let file = File::open(file_path)
            .with_context(|| format!("Failed to open CSV file: {}", file_path.display()))?;
        let reader = BufReader::new(file);

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(reader);

        let headers = rdr.headers()?.clone();
        let msg_idx = headers
            .iter()
            .position(|h| h.trim() == "messages")
            .with_context(|| format!("Missing 'messages' column in {}", file_path.display()))?;

        for record in rdr.records() {
            let record = match record {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Warning: skipping malformed CSV record in {}: {e}", file_path.display());
                    continue;
                }
            };

            let Some(raw_msg) = record.get(msg_idx) else {
                continue;
            };

            let turns = parse_ultrachat_turns(raw_msg);
            let memories = ultrachat_turns_to_memories(&turns);
            if !memories.is_empty() {
                blocks.push(ChatBlock { memories });
                if let Some(lim) = limit {
                    if blocks.len() >= lim {
                        break;
                    }
                }
            }
        }
    }

    println!(
        "✅ Loaded {} blocks ({} memories total) from UltraChat",
        blocks.len(),
        blocks.iter().map(|b| b.memories.len()).sum::<usize>()
    );

    Ok(HandcraftedChats { blocks })
}

/// Parse Python-style list of dicts in UltraChat's `messages` column.
///
/// Handles both single quotes and double quotes, as well as escape sequences
/// (`\n`, `\t`, `\r`, `\'`, `\"`, `\\`, `\xHH`, and `\uXXXX`).
pub fn parse_ultrachat_turns(raw: &str) -> Vec<(String, String)> {
    let mut turns = Vec::new();
    let bytes = raw.as_bytes();
    let n = bytes.len();
    let mut i = 0;

    while i < n {
        // Find next dict start '{'
        while i < n && bytes[i] != b'{' {
            i += 1;
        }
        if i >= n {
            break;
        }
        i += 1; // skip '{'

        let mut content: Option<String> = None;
        let mut role: Option<String> = None;

        while i < n {
            // Skip whitespace, newlines, commas
            while i < n && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\r' || bytes[i] == b'\n' || bytes[i] == b',') {
                i += 1;
            }
            if i >= n || bytes[i] == b'}' {
                if i < n {
                    i += 1;
                }
                break;
            }

            // Key quote
            if bytes[i] != b'\'' && bytes[i] != b'"' {
                i += 1;
                continue;
            }
            let key_quote = bytes[i];
            i += 1;
            let k_start = i;
            while i < n && bytes[i] != key_quote {
                if bytes[i] == b'\\' && i + 1 < n {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            let key = &raw[k_start..i];
            if i < n {
                i += 1; // skip closing quote
            }

            // Skip to ':'
            while i < n && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\r' || bytes[i] == b'\n') {
                i += 1;
            }
            if i < n && bytes[i] == b':' {
                i += 1;
            }
            while i < n && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\r' || bytes[i] == b'\n') {
                i += 1;
            }

            // Value quote
            if i < n && (bytes[i] == b'\'' || bytes[i] == b'"') {
                let v_quote = bytes[i];
                i += 1;
                let (val, next_i) = parse_ultrachat_quoted_value(raw, i, v_quote);
                i = next_i;

                if key == "content" {
                    content = Some(val);
                } else if key == "role" {
                    role = Some(val);
                }
            }
        }

        if let (Some(c), Some(r)) = (content, role) {
            turns.push((r, c));
        }
    }

    turns
}

fn parse_ultrachat_quoted_value(raw: &str, mut i: usize, quote_char: u8) -> (String, usize) {
    let bytes = raw.as_bytes();
    let n = bytes.len();
    let mut result = String::new();

    while i < n {
        let b = bytes[i];
        if b == b'\\' && i + 1 < n {
            match bytes[i + 1] {
                b'n' => {
                    result.push('\n');
                    i += 2;
                }
                b't' => {
                    result.push('\t');
                    i += 2;
                }
                b'r' => {
                    result.push('\r');
                    i += 2;
                }
                b'\\' => {
                    result.push('\\');
                    i += 2;
                }
                b'\'' => {
                    result.push('\'');
                    i += 2;
                }
                b'"' => {
                    result.push('"');
                    i += 2;
                }
                b'x' if i + 3 < n => {
                    if let Ok(byte_val) = u8::from_str_radix(&raw[i + 2..i + 4], 16) {
                        result.push(byte_val as char);
                        i += 4;
                    } else {
                        result.push('\\');
                        result.push('x');
                        i += 2;
                    }
                }
                b'u' if i + 5 < n => {
                    if let Ok(code_val) = u32::from_str_radix(&raw[i + 2..i + 6], 16) {
                        if let Some(ch) = char::from_u32(code_val) {
                            result.push(ch);
                            i += 6;
                        } else {
                            result.push('\\');
                            result.push('u');
                            i += 2;
                        }
                    } else {
                        result.push('\\');
                        result.push('u');
                        i += 2;
                    }
                }
                other => {
                    result.push('\\');
                    result.push(other as char);
                    i += 2;
                }
            }
        } else if b == quote_char {
            i += 1;
            break;
        } else {
            let s = &raw[i..];
            if let Some(ch) = s.chars().next() {
                result.push(ch);
                i += ch.len_utf8();
            } else {
                i += 1;
            }
        }
    }

    (result, i)
}

pub fn ultrachat_turns_to_memories(turns: &[(String, String)]) -> Vec<Memory> {
    let mut memories = Vec::new();
    let mut i = 0;
    while i < turns.len() {
        let role = turns[i].0.to_lowercase();
        if role == "user" || role == "human" {
            let human = turns[i].1.trim();
            if i + 1 < turns.len() {
                let next_role = turns[i + 1].0.to_lowercase();
                if next_role == "assistant" || next_role == "gpt" || next_role == "bot" {
                    let bot = turns[i + 1].1.trim();
                    if !human.is_empty() && !bot.is_empty() {
                        memories.push(Memory {
                            human: human.to_string(),
                            bot: bot.to_string(),
                        });
                    }
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    memories
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ultrachat_turns() {
        let raw = "[{'content': 'Hello\\nWorld', 'role': 'user'}\n {'content': \"I'm Yumon!\", 'role': 'assistant'}]";
        let turns = parse_ultrachat_turns(raw);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].0, "user");
        assert_eq!(turns[0].1, "Hello\nWorld");
        assert_eq!(turns[1].0, "assistant");
        assert_eq!(turns[1].1, "I'm Yumon!");

        let memories = ultrachat_turns_to_memories(&turns);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].human, "Hello\nWorld");
        assert_eq!(memories[0].bot, "I'm Yumon!");
    }

    #[test]
    fn test_parse_ultrachat_escapes() {
        let raw = "[{'content': 'test\\xadword and \\u0041', 'role': 'user'}, {'content': 'reply', 'role': 'assistant'}]";
        let turns = parse_ultrachat_turns(raw);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].1, "test\u{00ad}word and A");
    }
}