//! AM-DeepSeek-R1-Distilled-1.4M loader (`am_0.9M.jsonl.zst` and friends).
//!
//! One JSON object per line: `{"messages": [{role:"user", content, info}, {role:"assistant",
//! content:"<think>...</think>answer", info:{think_content, answer_content, ..}}]}`.
//!
//! The file is streamed (zstd-decoded on the fly), never read whole, and reading stops as soon
//! as `limit` usable pairs are collected. A truncated download (e.g. a `curl -r 0-N` prefix) is
//! fine: the decoder error at the cut is treated as end of data.
//!
//! Yumon gives short replies and has a ~200 char window, so each conversation becomes one
//! Memory: the user prompt and the first sentence of the final answer (reasoning discarded).
//! Pairs where either side is too long, or the answer opens with markup, are skipped.

use std::io::{BufRead, BufReader, Read};

use anyhow::{Context, Result};
use serde::Deserialize;
use crate::brain::loading::map_ordered;

use crate::brain::mdx::{ChatBlock, HandcraftedChats, Memory};
use crate::brain::train::MAX_SEQ_LEN_CHARS;

#[derive(Deserialize)]
struct Line {
    messages: Vec<Msg>,
}

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: String,
    #[serde(default)]
    info: Option<Info>,
}

#[derive(Deserialize, Default)]
struct Info {
    #[serde(default)]
    answer_content: Option<String>,
}

/// Final answer text of the assistant turn, without the `<think>` block.
fn answer_text(msg: &Msg) -> Option<String> {
    if let Some(a) = msg.info.as_ref().and_then(|i| i.answer_content.as_deref()) {
        if !a.trim().is_empty() {
            return Some(a.to_string());
        }
    }
    // Fall back to whatever follows the closing think tag.
    let (_, after) = msg.content.rsplit_once("</think>")?;
    (!after.trim().is_empty()).then(|| after.to_string())
}

/// First prose sentence of an answer, or None if it is markup / too long to be a Yumon reply.
fn short_reply(answer: &str) -> Option<String> {
    let text = answer.lines().map(str::trim).find(|l| {
        !l.is_empty() && !l.starts_with("**") && !l.starts_with('#') && !l.starts_with("$$") && !l.starts_with("\\[")
    })?;
    // End at the first terminator followed by whitespace, so "3.5" survives. Single-letter
    // initials ("Jesus H. Christ") and common abbreviations are not sentence ends.
    let bytes = text.as_bytes();
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate() {
        let terminator = matches!(b, b'.' | b'!' | b'?');
        let boundary = bytes.get(i + 1).map_or(true, |n| n.is_ascii_whitespace());
        if terminator && boundary && !(b == b'.' && is_abbreviation(&text[..i])) {
            end = Some(i + 1);
            break;
        }
    }
    let s = text[..end?].trim(); // no terminator => cut off or a heading/list lead-in
    let len = s.chars().count();
    if len < 12 || len > MAX_SEQ_LEN_CHARS || s.ends_with("..") || !is_plain_english(s) {
        return None;
    }
    Some(s.to_string())
}

/// True if the text before a '.' ends in an initial or common abbreviation.
fn is_abbreviation(before: &str) -> bool {
    let word = before.rsplit(char::is_whitespace).next().unwrap_or("");
    let letters = word.trim_start_matches(['(', '"', '\'']);
    letters.chars().count() == 1 && letters.chars().all(|c| c.is_alphabetic())
        || matches!(letters.to_ascii_lowercase().as_str(), "mr" | "mrs" | "ms" | "dr" | "st" | "vs" | "etc" | "e.g" | "i.e" | "no" | "inc" | "jr" | "sr")
}

/// The tokenizer and the rest of Yumon's data are English: drop CJK/emoji-heavy text and
/// markdown / LaTeX / code / bracket markup that makes a poor chat line.
fn is_plain_english(s: &str) -> bool {
    if s.contains(['\\', '$', '`', '*', '#', '[', ']', '{', '}', '|', '<', '>', '_']) {
        return false;
    }
    let total = s.chars().count();
    let ascii = s.chars().filter(|c| c.is_ascii()).count();
    ascii * 100 >= total * 97
}

fn memory_from_line(line: &str) -> Option<Memory> {
    let parsed: Line = serde_json::from_str(line).ok()?;
    let [user, assistant] = parsed.messages.as_slice() else { return None };
    if user.role != "user" || assistant.role != "assistant" {
        return None;
    }
    let human = user.content.trim();
    let len = human.chars().count();
    if len < 3 || len > MAX_SEQ_LEN_CHARS || !is_plain_english(human) {
        return None;
    }
    let bot = short_reply(&answer_text(assistant)?)?;
    Some(Memory { human: human.to_string(), bot })
}

/// Stream memories out of any reader of JSONL bytes. Stops after `limit` memories.
/// Returns the memories and how many lines were read.
pub fn memories_from_reader(reader: impl Read, limit: Option<usize>) -> (Vec<Memory>, usize) {
    let mut reader = BufReader::with_capacity(1 << 20, reader);
    let mut memories = Vec::new();
    let mut lines = 0usize;
    let mut ended = false;

    while !ended && !limit.is_some_and(|n| memories.len() >= n) {
        // Never read more lines than the remaining usable-pair cap. This
        // preserves early stopping even when every line is usable.
        let batch_size = limit.map_or(1024, |n| (n - memories.len()).min(1024));
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut buf = Vec::new();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) => {
                    ended = true;
                    break;
                }
                Ok(_) => {
                    lines += 1;
                    batch.push(buf);
                }
                Err(e) => {
                    // Keep completed lines from truncated downloads.
                    println!("stream ended early after {lines} lines: {e}");
                    ended = true;
                    break;
                }
            }
        }
        memories.extend(
            map_ordered(batch, |buf| {
                memory_from_line(&String::from_utf8_lossy(&buf))
            })
            .into_iter()
            .flatten(),
        );
    }
    (memories, lines)
}

/// Load a `.jsonl` or `.jsonl.zst` file. `limit` caps usable pairs (reading stops early).
pub fn load_am_distill_chats(path: &str, limit: Option<usize>) -> Result<HandcraftedChats> {
    println!("📖 Loading AM-DeepSeek distill: {path} (limit {limit:?})");

    let file = std::fs::File::open(path).with_context(|| format!("Failed to open {path}"))?;
    let (memories, lines) = if path.ends_with(".zst") {
        memories_from_reader(zstd::stream::read::Decoder::new(file)?, limit)
    } else {
        memories_from_reader(file, limit)
    };

    println!("✅ AM distill: {} usable pairs from {} lines", memories.len(), lines);

    // One memory per block: the conversations are independent single turns.
    let blocks = map_ordered(memories, |m| ChatBlock { memories: vec![m] });
    Ok(HandcraftedChats { blocks })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ZST: &str = "data/am_deepseek/am_0.9M_sample_1k.jsonl.zst";
    const SAMPLE_RAW: &str = "data/am_deepseek/am_0.9M_sample_1k.jsonl";

    #[test]
    fn batched_stream_preserves_order_and_exact_early_stop() {
        let mut data = String::new();
        for i in 0..2051 {
            if i % 7 == 0 {
                data.push_str("not json\n");
            }
            data.push_str(&format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"Question number {i}?\"}},{{\"role\":\"assistant\",\"content\":\"<think>x</think>This is a useful answer.\"}}]}}\n"));
        }
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let (memories, lines) = memories_from_reader(data.as_bytes(), Some(1030));
                    assert_eq!(memories.len(), 1030);
                    assert_eq!(lines, 1030 + (1029 / 7 + 1));
                    for (i, memory) in memories.iter().enumerate() {
                        assert_eq!(memory.human, format!("Question number {i}?"));
                    }
                    let (memories, lines) = memories_from_reader(data.as_bytes(), Some(0));
                    assert!(memories.is_empty());
                    assert_eq!(lines, 0);
                });
        }
    }

    #[test]
    fn completed_batch_survives_reader_failure() {
        struct FailingReader(Option<Vec<u8>>);
        impl Read for FailingReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                match self.0.take() {
                    Some(data) => {
                        buf[..data.len()].copy_from_slice(&data);
                        Ok(data.len())
                    }
                    None => Err(std::io::Error::other("truncated stream")),
                }
            }
        }
        let line = b"{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello please\"},{\"role\":\"assistant\",\"content\":\"<think>x</think>Hello there, friend!\"}]}\n";
        let (memories, lines) = memories_from_reader(FailingReader(Some(line.to_vec())), None);
        assert_eq!(lines, 1);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].bot, "Hello there, friend!");
    }

    #[test]
    fn parses_synthetic_line() {
        let line = r#"{"messages":[{"role":"user","content":"What is the capital of France?","info":{}},{"role":"assistant","content":"<think>hmm</think>Paris is the capital. More text.","info":{"think_content":"hmm","answer_content":"Paris is the capital. More text."}}]}"#;
        let m = memory_from_line(line).unwrap();
        assert_eq!(m.human, "What is the capital of France?");
        assert_eq!(m.bot, "Paris is the capital.");
    }

    #[test]
    fn falls_back_to_think_tag_split() {
        let line = r#"{"messages":[{"role":"user","content":"Say hello please","info":{}},{"role":"assistant","content":"<think>x</think>\nHello there, friend! Bye.","info":{}}]}"#;
        assert_eq!(memory_from_line(line).unwrap().bot, "Hello there, friend!");
    }

    #[test]
    fn rejects_latex_and_wrong_shape() {
        let latex = r#"{"messages":[{"role":"user","content":"Solve it for me","info":{}},{"role":"assistant","content":"x","info":{"answer_content":"The answer is \\( x = 4 \\)."}}]}"#;
        assert!(memory_from_line(latex).is_none());
        assert!(memory_from_line("not json").is_none());
        assert!(memory_from_line(r#"{"messages":[]}"#).is_none());
    }

    #[test]
    fn keeps_decimals_inside_a_sentence() {
        assert_eq!(short_reply("It costs 3.5 dollars. Next.").unwrap(), "It costs 3.5 dollars.");
    }

    #[test]
    fn skips_initials_markup_and_cjk() {
        assert_eq!(short_reply("Jesus H. Christ is an expression of surprise. Next.").unwrap(), "Jesus H. Christ is an expression of surprise.");
        assert!(short_reply("Here is a formatted table of **words**.").is_none());
        assert!(short_reply("Here is the table you asked for:").is_none()); // no terminator
        assert!(short_reply("林黛玉是《红楼梦》中最具复杂心理层次的艺术形象。").is_none());
        let zh = r#"{"messages":[{"role":"user","content":"为学生提供在家学习的技巧和建议","info":{}},{"role":"assistant","content":"x","info":{"answer_content":"Study at home with a plan."}}]}"#;
        assert!(memory_from_line(zh).is_none());
    }

    #[test]
    fn zst_and_raw_sample_agree() {
        if !std::path::Path::new(SAMPLE_ZST).exists() {
            eprintln!("sample not downloaded, skipping");
            return;
        }
        let z = load_am_distill_chats(SAMPLE_ZST, None).unwrap();
        let r = load_am_distill_chats(SAMPLE_RAW, None).unwrap();
        assert_eq!(z.blocks.len(), r.blocks.len());
        assert!(z.blocks.len() > 20, "expected a usable share of the 1k sample, got {}", z.blocks.len());
        for b in &z.blocks {
            let m = &b.memories[0];
            assert!(m.human.chars().count() <= MAX_SEQ_LEN_CHARS && m.bot.chars().count() <= MAX_SEQ_LEN_CHARS);
        }
        let limited = load_am_distill_chats(SAMPLE_ZST, Some(5)).unwrap();
        assert_eq!(limited.blocks.len(), 5);
    }

    /// `cargo test --lib am_distill::tests::print_pairs -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn print_pairs() {
        let c = load_am_distill_chats(SAMPLE_ZST, Some(15)).unwrap();
        for b in c.blocks {
            println!("H: {}\nB: {}\n", b.memories[0].human, b.memories[0].bot);
        }
    }

    #[test]
    fn truncated_zst_is_tolerated() {
        if !std::path::Path::new(SAMPLE_ZST).exists() {
            return;
        }
        let bytes = std::fs::read(SAMPLE_ZST).unwrap();
        let cut = &bytes[..bytes.len() / 2];
        let (mems, _) = memories_from_reader(zstd::stream::read::Decoder::new(cut).unwrap(), None);
        assert!(!mems.is_empty());
    }
}
