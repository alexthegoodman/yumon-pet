use crate::brain::wiki::is_good_sentence;
use anyhow::Result;
use crate::brain::loading::{map_batches, map_ordered};

#[cfg(not(target_arch = "wasm32"))]
pub fn load_mdx_sentences(mdx_dir: &str) -> Result<Vec<String>> {
    println!("📖 Scanning MDX directory: {mdx_dir}");

    let paths: Vec<_> = walkdir::WalkDir::new(mdx_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("mdx"))
        .map(|entry| entry.into_path())
        .collect();
    let files_loaded = paths.len();
    let sentences: Vec<_> = map_ordered(paths, |path| -> Result<Vec<String>> {
        let mut sentences = Vec::new();
        let content = std::fs::read_to_string(&path)?;

        for line in content.lines() {
            let trimmed = line.trim().replace("<br />", "");
            if is_good_sentence(&trimmed) && !trimmed.starts_with("# ") {
                sentences.push(trimmed.to_string());
            }
        }

        Ok(sentences)
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten()
    .collect();

    let mut final_sentences = Vec::new();

    for sents in sentences.chunks(2) {
        // appropriate length, decent connections
        final_sentences.push(sents.join(" "));
    }

    println!(
        "✅ Loaded {} sentences from {} MDX files",
        final_sentences.len(),
        files_loaded
    );
    Ok(final_sentences)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_notion_sentences(notion_dir: &str) -> Result<Vec<String>> {
    println!("📖 Scanning Notion directory: {notion_dir}");

    let paths: Vec<_> = walkdir::WalkDir::new(notion_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
        .map(|entry| entry.into_path())
        .collect();
    let files_loaded = paths.len();
    let sentences: Vec<_> = map_ordered(paths, |path| -> Result<Vec<String>> {
        let mut sentences = Vec::new();
        let content = std::fs::read_to_string(&path)?;

        for line in content.lines() {
            let trimmed = line.trim();

            for sent in trimmed.split(".") {
                if is_good_sentence(sent) {
                    sentences.push(sent.to_string());
                }
            }
        }

        Ok(sentences)
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten()
    .collect();

    println!(
        "✅ Loaded {} sentences from {} Notion files",
        sentences.len(),
        files_loaded
    );
    Ok(sentences)
}

// #[derive(Debug)]
// pub struct Quote {
//     pub text: String,
//     pub author: String,
//     pub tags: Vec<String>,
// }

#[cfg(not(target_arch = "wasm32"))]
pub fn load_csv_quotes(csv_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading quotes CSV: {csv_path}");

    let mut rdr = csv::Reader::from_path(csv_path)?;
    let quotes: Vec<_> = map_batches(
        rdr.records()
            .filter(|record| match record {
                Ok(record) => !record.get(0).unwrap_or("").trim().is_empty(),
                Err(_) => true,
            })
            .take(100001),
        |record| -> Result<String> {
            let record = record?;
            Ok(record.get(0).unwrap_or("").trim().to_string())
        },
    )
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    println!("✅ Loaded {} quotes from {csv_path}", quotes.len());
    Ok(quotes)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_csv_qna(csv_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading qna CSV: {csv_path}");

    let mut rdr = csv::Reader::from_path(csv_path)?;
    let quotes: Vec<_> = map_batches(rdr.records().take(10001), |record| -> Result<String> {
        let record = record?;
        Ok(format!(
            "{} {}",
            record.get(0).unwrap_or("").trim(),
            record.get(1).unwrap_or("").trim()
        ))
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    println!(
        "✅ Loaded {} questions and answers from {csv_path}",
        quotes.len()
    );
    Ok(quotes)
}

// pub fn load_csv_bible(bible_path: &str) -> Result<Vec<String>> {
//     println!("📖 Loading bible CSV: {bible_path}");

//     let mut rdr = csv::Reader::from_path(bible_path)?;
//     let mut quotes = Vec::new();

//     let mut count = 0;

//     for result in rdr.records() {
//         let record = result?;

//         let id   = record.get(0).unwrap_or("").trim().to_string();
//         let b = record.get(1).unwrap_or("").trim().to_string();
//         let c = record.get(2).unwrap_or("").trim().to_string();
//         let v = record.get(3).unwrap_or("").trim().to_string();
//         let verse = record.get(4).unwrap_or("").trim().to_string();

//         // if (b == "40" || b == "41" || b == "42" || b == "43") { // gospel only
//         // if (b == "20") { // proverbs
//             quotes.push(verse);

//             count = count + 1;

//             if count > 20_000 { break; }
//         // }
//     }

//     println!("✅ Loaded {} verses from {bible_path}", quotes.len());
//     Ok(quotes)
// }

/// Pairs up consecutive verses (verse N -> verse N+1) instead of splitting a
/// concatenated chunk mid-sentence, so each side of the pair is a real,
/// complete verse.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_csv_bible_pairs(bible_path: &str) -> Result<Vec<(String, String)>> {
    println!("📖 Loading bible CSV: {bible_path}");

    let mut rdr = csv::Reader::from_path(bible_path)?;

    let verses: Vec<_> = map_batches(rdr.records(), |result| -> Result<Option<String>> {
        let record = result?;
        let verse = record.get(4).unwrap_or("").trim().to_string();

        if verse.is_empty() {
            return Ok(None);
        }

        Ok(Some(verse))
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten()
    .collect();

    let pairs = map_batches(verses.chunks_exact(2), |pair| {
        (pair[0].clone(), pair[1].clone())
    });

    println!("✅ Loaded {} verse pairs from {bible_path}", pairs.len());
    Ok(pairs)
}

pub fn load_dictionary_sentences(dict_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading dictionary: {dict_path}");

    let content = std::fs::read_to_string(dict_path)?;

    let sentences: Vec<_> = map_batches(content.lines(), |line| {
        let trimmed = line.trim();

        // Skip empty lines or very short lines
        if trimmed.len() < 10 {
            return None;
        }

        // Skip lines that are just a single letter (section headers like "A")
        if trimmed
            .chars()
            .all(|c| c.is_alphabetic() || c.is_whitespace())
            && trimmed.split_whitespace().count() <= 1
        {
            return None;
        }

        // Try to extract a clean definition sentence
        // if let Some(sentence) = extract_definition(trimmed) {
        if is_good_sentence(&trimmed) {
            return Some(trimmed.to_string());
        }
        // }

        None
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} dictionary sentences", sentences.len());
    Ok(sentences)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_csv_words(csv_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading words CSV: {csv_path}");

    let mut rdr = csv::Reader::from_path(csv_path)?;
    let words: Vec<_> = map_batches(rdr.records().take(10001), |record| -> Result<String> {
        let record = record?;
        Ok(record.get(1).unwrap_or("").trim().to_string())
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    println!("✅ Loaded {} words from {csv_path}", words.len());
    Ok(words)
}

pub fn contains_word(trimmed: &str, selected_words: &Vec<String>) -> bool {
    let mut contains_word = false;
    for word in selected_words {
        if trimmed.contains(word) {
            contains_word = true;
        }
    }
    contains_word
}

pub fn load_specific_dict_sentences(
    dict_path: &str,
    selected_words: &Vec<String>,
) -> Result<Vec<String>> {
    println!("📖 Loading dictionary: {dict_path}");

    let content = std::fs::read_to_string(dict_path)?;

    let sentences: Vec<_> = map_batches(content.lines(), |line| {
        let trimmed = line.trim();

        // Skip empty lines or very short lines
        if trimmed.len() < 10 {
            return None;
        }

        // Skip lines that are just a single letter (section headers like "A")
        if trimmed
            .chars()
            .all(|c| c.is_alphabetic() || c.is_whitespace())
            && trimmed.split_whitespace().count() <= 1
        {
            return None;
        }

        // Try to extract a clean definition sentence
        // if let Some(sentence) = extract_definition(trimmed) {
        if is_good_sentence(&trimmed) && contains_word(trimmed, &selected_words) {
            return Some(trimmed.to_string());
        }
        // }

        None
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} dictionary sentences", sentences.len());
    Ok(sentences)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_handcrafted_sentences(dict_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading handcrafted: {dict_path}");

    let content = std::fs::read_to_string(dict_path)?;

    let sentences: Vec<_> = map_batches(content.lines(), |line| {
        let trimmed = line.trim().replace("<br />", "");

        if is_good_sentence(&trimmed) {
            return Some(trimmed.to_string());
        }

        None
    })
    .into_iter()
    .flatten()
    .collect();

    // let mut final_sentences = Vec::new();

    // for sents in sentences.chunks(2) { // appropriate length, decent connections
    //     final_sentences.push(sents.join(" "));
    // }

    println!("✅ Loaded {} handcrafted sentences", sentences.len());
    Ok(sentences)
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub human: String,
    pub bot: String,
}

#[derive(Debug, Clone)]
pub struct ChatBlock {
    pub memories: Vec<Memory>,
}

#[derive(Debug)]
pub struct HandcraftedChats {
    pub blocks: Vec<ChatBlock>,
}

pub fn load_handcrafted_chats(dict_path: &str) -> Result<HandcraftedChats> {
    println!("📖 Loading handcrafted: {dict_path}");

    let content = std::fs::read_to_string(dict_path)?;

    let mut raw_blocks = Vec::new();
    let mut current_block: Vec<String> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if !current_block.is_empty() {
                raw_blocks.push(std::mem::take(&mut current_block));
            }
        } else {
            current_block.push(trimmed.to_string());
        }
    }

    // Handle trailing block with no final blank line
    if !current_block.is_empty() {
        raw_blocks.push(current_block);
    }

    let blocks = map_ordered(raw_blocks, |lines| parse_block(&lines));

    println!(
        "✅ Loaded {} blocks ({} memories total)",
        blocks.len(),
        blocks.iter().map(|b| b.memories.len()).sum::<usize>()
    );

    Ok(HandcraftedChats { blocks })
}

use anyhow::{Context};
use csv::ReaderBuilder;
use std::collections::HashMap;

pub fn load_chats_from_csv(csv_path: &str) -> Result<HandcraftedChats> {
    println!("📖 Loading CSV chats: {csv_path}");

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .with_context(|| format!("Failed to open {csv_path}"))?;

    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Result<usize> {
        headers
            .iter()
            .position(|h| h.trim() == name)
            .with_context(|| format!("Missing column: {name}"))
    };

    let (ci_season, ci_episode, ci_scene, ci_line) =
        (col("season")?, col("episode")?, col("scene")?, col("line")?);

    let mut scene_order: Vec<(u32, u32, u32)> = Vec::new();
    let mut scene_lines: HashMap<(u32, u32, u32), Vec<String>> = HashMap::new();

    for result in rdr.records() {
        let record = result.context("Failed to parse CSV record")?;

        let season: u32 = record[ci_season].trim().parse().unwrap_or(0);
        let episode: u32 = record[ci_episode].trim().parse().unwrap_or(0);
        let scene: u32 = record[ci_scene].trim().parse().unwrap_or(0);
        let line = record[ci_line].trim().to_string();

        if line.is_empty() {
            continue;
        }

        let key = (season, episode, scene);
        if !scene_lines.contains_key(&key) {
            scene_order.push(key);
        }
        scene_lines.entry(key).or_default().push(line);
    }

    let blocks: Vec<ChatBlock> = map_ordered(scene_order, |key| {
        let lines = &scene_lines[&key];
        let memories = lines
            .chunks(2)
            .filter_map(|pair| {
                if pair.len() == 2 {
                    Some(Memory {
                        human: pair[0].clone(),
                        bot: pair[1].clone(),
                    })
                } else {
                    None
                }
            })
            .collect();
        ChatBlock { memories }
    });

    println!(
        "✅ Loaded {} blocks ({} memories total)",
        blocks.len(),
        blocks.iter().map(|b| b.memories.len()).sum::<usize>()
    );

    Ok(HandcraftedChats { blocks })
}

pub fn load_chats_from_friends_csv(csv_path: &str) -> Result<HandcraftedChats> {
    println!("📖 Loading CSV chats: {csv_path}");

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .with_context(|| format!("Failed to open {csv_path}"))?;

    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Result<usize> {
        headers
            .iter()
            .position(|h| h.trim() == name)
            .with_context(|| format!("Missing column: {name}"))
    };

    let ci_type = col("type")?;
    let ci_speaker = col("speaker")?;
    let ci_dialogue = col("dialogue_clean")?;
    let ci_season = col("season")?;
    let ci_episode = col("episode")?;

    // Each entry is (season, episode, scene_index, lines)
    let mut scene_order: Vec<(u32, u32, u32)> = Vec::new();
    let mut scene_lines: HashMap<(u32, u32, u32), Vec<String>> = HashMap::new();

    let mut scene_index: u32 = 0;
    let mut current_key: Option<(u32, u32, u32)> = None;

    for result in rdr.records() {
        let record = result.context("Failed to parse CSV record")?;

        let row_type = record[ci_type].trim();
        let season: u32 = record[ci_season].trim().parse().unwrap_or(0);
        let episode: u32 = record[ci_episode].trim().parse().unwrap_or(0);

        match row_type {
            "scene_note" | "scene_direction" => {
                // Start a new scene block
                scene_index += 1;
                let key = (season, episode, scene_index);
                scene_order.push(key);
                current_key = Some(key);
            }
            "dialogue" => {
                let speaker = record[ci_speaker].trim();
                let line = record[ci_dialogue].trim();

                if line.is_empty() {
                    continue;
                }

                // If no scene has started yet, open an implicit scene 0
                let key = current_key.get_or_insert_with(|| {
                    let k = (season, episode, 0);
                    scene_order.push(k);
                    k
                });

                // let formatted = if speaker.is_empty() {
                //     line.to_string()
                // } else {
                //     format!("{speaker}: {line}")
                // };

                let formatted = line.to_string();

                scene_lines.entry(*key).or_default().push(formatted);
            }
            // stage_direction and anything else: skip
            _ => {}
        }
    }

    let blocks: Vec<ChatBlock> = map_ordered(scene_order, |key| {
        let lines = scene_lines.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        let memories = lines
            .chunks(2)
            .filter_map(|pair| {
                if pair.len() == 2 {
                    Some(Memory {
                        human: pair[0].clone(),
                        bot: pair[1].clone(),
                    })
                } else {
                    None
                }
            })
            .collect();
        ChatBlock { memories }
    });

    println!(
        "✅ Loaded {} blocks ({} memories total)",
        blocks.len(),
        blocks.iter().map(|b| b.memories.len()).sum::<usize>()
    );

    Ok(HandcraftedChats { blocks })
}

use serde::Deserialize;

#[derive(Deserialize)]
struct ConversationTurn {
    content: String,
    role: String,
}

#[derive(Deserialize)]
struct ArenaEntry {
    conversation_a: Vec<ConversationTurn>,
    conversation_b: Vec<ConversationTurn>,
    winner: String,
}

fn first_sentence(text: &str) -> String {
    text.split(['.', '!', '?'])
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|s| s + ".")
        .unwrap_or_else(|| text.trim().to_string())
}

fn extract_memories(conversation: &[ConversationTurn]) -> Vec<Memory> {
    let mut memories = Vec::new();
    let mut i = 0;

    while i + 1 < conversation.len() {
        let turn = &conversation[i];
        let reply = &conversation[i + 1];

        if turn.role == "user" && reply.role == "assistant" {
            memories.push(Memory {
                human: turn.content.trim().to_string(),
                bot: first_sentence(&reply.content),
            });
        }
        i += 2;
    }

    memories
}

pub fn load_arena_chats(path: &str) -> Result<HandcraftedChats> {
    println!("📖 Loading arena JSONL: {path}");

    let content = std::fs::read_to_string(path)?;
    let parsed = map_batches(content.lines(), |line| -> Result<Option<ChatBlock>> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let entry: ArenaEntry = serde_json::from_str(trimmed)?;

        // Pick the winning conversation, or fall back to conversation_a
        let convo = match entry.winner.as_str() {
            "model_b" => &entry.conversation_b,
            _ => &entry.conversation_a,
        };

        let memories = extract_memories(convo);
        if !memories.is_empty() {
            return Ok(Some(ChatBlock { memories }));
        }
        Ok(None)
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

fn parse_block(lines: &[String]) -> ChatBlock {
    let mut memories = Vec::new();
    let mut i = 0;

    while i + 1 < lines.len() {
        memories.push(Memory {
            human: lines[i].clone(),
            bot: lines[i + 1].clone(),
        });
        i += 2;
    }

    ChatBlock { memories }
}

fn flush_block(block: &[String], memories: &mut Vec<Memory>) {
    // Odd-indexed lines (0, 2, 4...) are human; even-indexed (1, 3, 5...) are bot
    let mut i = 0;
    while i + 1 < block.len() {
        memories.push(Memory {
            human: block[i].clone(),
            bot: block[i + 1].clone(),
        });
        i += 2;
    }
    // If block has an odd number of lines, the last human turn has no bot reply — skip it
}

pub fn load_txt_sentences(path: &str) -> Result<Vec<String>> {
    println!("📖 Loading txt: {path}");

    let content = std::fs::read_to_string(path)?;
    let sentences: Vec<_> = map_batches(content.lines(), |line| {
        let mut sentences = Vec::new();
        let trimmed = line.trim();

        for sent in trimmed.split(".") {
            if is_good_sentence(&sent) && sent.len() > 15 {
                // have a nice sensible minimum length
                sentences.push(sent.to_string());
            }
        }
        sentences
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} txt sentences", sentences.len());
    Ok(sentences)
}

pub fn load_txt_lines(path: &str) -> Result<Vec<String>> {
    println!("📖 Loading txt: {path}");

    let content = std::fs::read_to_string(path)?;

    let sentences: Vec<_> = map_batches(content.lines(), |line| {
        let trimmed = line.trim().replace("\"", "");

        Some(trimmed.to_string())
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} txt lines", sentences.len());
    Ok(sentences)
}

/// One paragraph per line (e.g. wiki_extract.txt). Skips leftover wiki markup
/// (image captions, links, tables) and very short fragments.
pub fn load_text_paragraphs(path: &str) -> Result<Vec<String>> {
    println!("📖 Loading paragraphs: {path}");

    let content = std::fs::read_to_string(path)?;

    // Talk pages, deletion discussions and template docs are not article text.
    const SKIP: [&str; 10] = [
        "(utc)", "talk)", "wp:", "help:", "user:", "<span", "&#", "}}", "template", "redirect",
    ];
    const ENTITIES: [(&str, &str); 5] = [
        ("&ndash;", "-"),
        ("&mdash;", "-"),
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&middot;", "-"),
    ];

    let paragraphs: Vec<_> = map_batches(content.lines(), |line| {
        let mut trimmed = line.trim().replace("''", "");
        if trimmed.contains('|')
            || trimmed.contains("http")
            || trimmed.contains("[[")
            || trimmed.contains("{{")
        {
            return None;
        }
        // ASCII lowercase keeps byte offsets valid for truncate below.
        let lower = trimmed.to_ascii_lowercase();
        if SKIP.iter().any(|s| lower.contains(s)) {
            return None;
        }
        // Category lists (often after "References") trail the last paragraph.
        if let Some(i) = lower.find("category:") {
            trimmed.truncate(i);
            let end = trimmed.trim_end().len();
            if trimmed[..end].to_ascii_lowercase().ends_with("references") {
                trimmed.truncate(end - "references".len());
            }
        }
        for (entity, text) in ENTITIES {
            trimmed = trimmed.replace(entity, text);
        }
        let trimmed = trimmed.trim().to_string();
        if trimmed.contains('&') && trimmed.contains(';') {
            return None;
        } // other entities
        if trimmed.split_whitespace().count() < 8 {
            return None;
        }
        Some(trimmed)
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} paragraphs", paragraphs.len());
    Ok(paragraphs)
}

/// Splits prose into sentences after `.`, `!` or `?` when the next word starts
/// with an uppercase letter, digit or quote. Abbreviations ("Mr.", "U.S.",
/// "e.g.") and single initials do not end a sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    const ABBREVIATIONS: [&str; 14] = ["mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "no.", "mt.", "ft.", "approx.", "inc."];
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut sentences = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for (i, word) in words.iter().enumerate() {
        current.push(word);
        let trimmed = word.trim_end_matches(|c| c == '"' || c == '\'' || c == ')');
        let ends = trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?');
        let next_starts = words.get(i + 1)
            .and_then(|w| w.trim_start_matches(|c| c == '"' || c == '\'' || c == '(').chars().next())
            .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit());
        let body = trimmed.trim_end_matches(|c| c == '.' || c == '!' || c == '?');
        let abbreviation = ABBREVIATIONS.contains(&trimmed.to_lowercase().as_str())
            || body.contains('.') // U.S., e.g.
            || (body.chars().count() == 1 && body.chars().all(char::is_uppercase)); // J. Smith
        if (ends && next_starts && !abbreviation) || i + 1 == words.len() {
            sentences.push(current.join(" "));
            current.clear();
        }
    }
    sentences
}

/// Pairs sentences into Human/Yumon turns. An odd last sentence joins the
/// last reply. With an opening question, the first sentence answers it.
fn sentence_turns(question: Option<String>, sentences: Vec<String>) -> Vec<Memory> {
    let mut queue: std::collections::VecDeque<String> = sentences.into();
    let mut turns = Vec::new();
    if let Some(question) = question {
        let Some(answer) = queue.pop_front() else { return turns; };
        turns.push(Memory { human: question, bot: answer });
    }
    while queue.len() >= 2 {
        let human = queue.pop_front().unwrap();
        let bot = queue.pop_front().unwrap();
        turns.push(Memory { human, bot });
    }
    if let (Some(rest), Some(last)) = (queue.pop_front(), turns.last_mut()) {
        last.bot = format!("{} {rest}", last.bot);
    }
    turns
}

/// Stable template choice so reruns build identical samples.
fn pick<'a>(options: &[&'a str], key: &str) -> &'a str {
    let sum = key.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
    options[sum % options.len()]
}

/// wiki_extract.txt as conversations: each paragraph's sentences become
/// alternating Human/Yumon turns (one block per paragraph). Article openings
/// ("'April' (Apr.) is ...") start with a question about the title.
pub fn load_wiki_chats(path: &str) -> Result<HandcraftedChats> {
    println!("📖 Loading wiki chats: {path}");
    const QUESTIONS: [&str; 6] = [
        "tell me about {}",
        "what is {}?",
        "what do you know about {}?",
        "can you explain {}?",
        "what can you tell me about {}?",
        "explain {} to me",
    ];

    let parsed = map_ordered(load_text_paragraphs(path)?, |paragraph| {
        // Opening paragraphs quote the title once: 'April' (Apr.) is ...
        let title = paragraph
            .strip_prefix('\'')
            .and_then(|rest| rest.split_once('\''))
            .filter(|(title, after)| {
                !title.is_empty() && title.len() <= 60 && after.starts_with(' ')
            })
            .map(|(title, _)| title.to_string());
        let text = match &title {
            Some(title) => format!("{title}{}", &paragraph[title.len() + 2..]),
            None => paragraph,
        };
        let question = title.map(|t| pick(&QUESTIONS, &t).replace("{}", &t.to_lowercase()));
        let titled = usize::from(question.is_some());

        let memories = sentence_turns(question, split_sentences(&text));
        (
            (!memories.is_empty()).then_some(ChatBlock { memories }),
            titled,
        )
    });
    let titled: usize = parsed.iter().map(|(_, titled)| titled).sum();
    let blocks: Vec<_> = parsed.into_iter().filter_map(|(block, _)| block).collect();

    println!(
        "✅ Loaded {} wiki conversations ({} open with a title question)",
        blocks.len(),
        titled
    );
    Ok(HandcraftedChats { blocks })
}

/// quotes.csv as one-turn conversations: a request built from the quote's
/// first category tag, answered with the quote.
pub fn load_quote_chats(path: &str) -> Result<HandcraftedChats> {
    println!("📖 Loading quote chats: {path}");
    const REQUESTS: [&str; 5] = [
        "share a quote about {}",
        "tell me something wise about {}",
        "what is a good quote about {}?",
        "say something about {}",
        "do you know a quote about {}?",
    ];

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;
    let headers = rdr.headers()?.clone();
    let column = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| anyhow::anyhow!("No '{name}' column found in {path}"))
    };
    let (quote_idx, category_idx) = (column("quote")?, column("category")?);

    let blocks: Vec<_> = map_batches(rdr.records(), |record| {
        let Ok(record) = record else {
            return None;
        };
        let Some(quote) = record.get(quote_idx).map(str::trim) else {
            return None;
        };
        if quote.split_whitespace().count() < 3 {
            return None;
        }
        let tag = record
            .get(category_idx)
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .find(|t| !t.is_empty() && !t.starts_with("attributed"));
        let human = match tag {
            Some(tag) => pick(&REQUESTS, quote).replace("{}", &tag.replace('-', " ")),
            None => "share a quote".to_string(),
        };
        Some(ChatBlock {
            memories: vec![Memory {
                human,
                bot: quote.to_string(),
            }],
        })
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} quote chats", blocks.len());
    Ok(HandcraftedChats { blocks })
}

/// Quote text from a `quote,author,category` CSV.
pub fn load_quotes_csv(path: &str) -> Result<Vec<String>> {
    println!("📖 Loading quotes: {path}");

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;
    let quote_idx = rdr
        .headers()?
        .iter()
        .position(|h| h == "quote")
        .ok_or_else(|| anyhow::anyhow!("No 'quote' column found in {path}"))?;

    let quotes: Vec<_> = map_batches(rdr.records(), |record| {
        let Ok(record) = record else {
            return None;
        };
        let Some(quote) = record.get(quote_idx) else {
            return None;
        };
        let quote = quote.trim();
        if quote.split_whitespace().count() >= 3 {
            return Some(quote.to_string());
        }

        None
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} quotes", quotes.len());
    Ok(quotes)
}

pub fn load_qa_pairs(path: &str) -> Result<Vec<(String, String)>> {
    println!("📖 Loading QA pairs: {path}");

    let content = std::fs::read_to_string(path)?;

    // 1. Collect non-empty, trimmed lines
    let lines: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    // 2. Iterate through lines in steps of 2
    let pairs: Vec<_> = map_batches(lines.chunks(2), |chunk| {
        if chunk.len() == 2 {
            let question = chunk[0].clone();
            let answer = chunk[1].clone();

            // You can still apply your "is_good" logic here
            if is_good_sentence(&(question.clone() + &answer)) {
                return Some((question, answer));
            }
        }

        None
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} QA pairs", pairs.len());
    Ok(pairs)
}

pub fn load_qa_singles(path: &str) -> Result<Vec<String>> {
    println!("📖 Loading QA singles: {path}");

    let content = std::fs::read_to_string(path)?;

    // 1. Collect non-empty, trimmed lines
    let lines: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    // 2. Iterate through lines in steps of 2
    let pairs: Vec<_> = map_batches(lines.chunks(2), |chunk| {
        if chunk.len() == 2 {
            let question = chunk[0].clone();
            let answer = chunk[1].clone();

            // You can still apply your "is_good" logic here
            if is_good_sentence(&question) && is_good_sentence(&answer) {
                return Some(question + " " + &answer);
            }
        }

        None
    })
    .into_iter()
    .flatten()
    .collect();

    println!("✅ Loaded {} QA singles", pairs.len());
    Ok(pairs)
}

// pub fn load_txt_sentences(path: &str) -> Result<Vec<String>> {
//     println!("📖 Loading txt: {path}");

//     let content = std::fs::read_to_string(path)?;
//     let mut sentences = Vec::new();
//     let mut buffer = String::new();

//     for line in content.lines() {
//         let trimmed = line.trim();

//         for sent in trimmed.split('.') {
//             if !is_good_sentence(&sent) {
//                 continue;
//             }

//             if buffer.is_empty() {
//                 buffer.push_str(sent.trim());
//             } else {
//                 buffer.push_str(". ");
//                 buffer.push_str(sent.trim());
//             }

//             if buffer.len() >= 150 {
//                 sentences.push(buffer.clone());
//                 buffer.clear();
//             }
//         }
//     }

//     // Flush any remaining content
//     if !buffer.is_empty() {
//         sentences.push(buffer);
//     }

//     println!("✅ Loaded {} txt sentences", sentences.len());
//     Ok(sentences)
// }

#[cfg(not(target_arch = "wasm32"))]
fn extract_definition(line: &str) -> Option<String> {
    // Strip etymology brackets at the end e.g. [latin from greek]
    let line = regex::Regex::new(r"\[.*?\]")
        .ok()?
        .replace_all(line, "")
        .trim()
        .to_string();

    // Find where the definition starts — after the headword and pos tag
    // Pattern: "Word  —n. Definition" or "Word  n. Definition"
    // We look for the first lowercase run after the headword
    let mut chars = line.char_indices().peekable();

    // Skip the headword (leading capitals/mixed case word)
    while let Some((_, c)) = chars.peek() {
        if c.is_whitespace() { break; }
        chars.next();
    }

    // Skip whitespace
    while let Some((_, c)) = chars.peek() {
        if !c.is_whitespace() { break; }
        chars.next();
    }

    // Get the rest as the definition
    let rest = chars
        .map(|(i, _)| line.chars().nth(i).unwrap_or(' '))
        .collect::<String>();

    // Strip leading part-of-speech markers like "—n. " "—v. " "abbr."
    let cleaned = regex::Regex::new(r"^[—–]?[a-z]+\.\s*")
        .ok()?
        .replace(&rest, "")
        .trim()
        .to_string();

    if cleaned.is_empty() { None } else { Some(cleaned) }
}
#[cfg(test)]
mod language_chat_source_tests {
    use super::*;

    fn temp_file(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("yumon-chat-src-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    fn remove(path: &std::path::Path) {
        let dir = path.parent().unwrap();
        assert_eq!(dir.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn language_sentences_split_without_breaking_abbreviations() {
        let text = "April (Apr.) is the fourth month of the year. Mr. Smith lived in the U.S. Army base for 3 years! Did he? 2024 was a leap year. J. R. R. Tolkien wrote books.";
        assert_eq!(split_sentences(text), vec![
            "April (Apr.) is the fourth month of the year.",
            "Mr. Smith lived in the U.S. Army base for 3 years!",
            "Did he?",
            "2024 was a leap year.",
            "J. R. R. Tolkien wrote books.",
        ]);
    }

    #[test]
    fn language_wiki_paragraphs_become_titled_conversations() {
        let path = temp_file("wiki.txt", concat!(
            "'April' (Apr.) is the fourth month of the year in the calendar. It has thirty days in it every year. Many festivals happen in April each year.\n",
            "April ends on the same day of the week as December every single year. This is because they are exactly 35 weeks apart. Easter is often in April too. Rain is common in many places.\n",
            "thumb|200px|right|An April Fools' Day hoax for April 1 in Copenhagen with many words here.\n",
            "Please discuss this request below, but keep in mind that you should not vote. --Someone (talk) 10:47, 22 November 2024 (UTC)\n",
            "Rome &ndash; the capital of Italy &ndash; is very old. It was founded long ago by Romulus. References Category:Cities in Italy Category:Capitals\n",
        ));
        let chats = load_wiki_chats(path.to_str().unwrap()).unwrap();
        remove(&path);

        assert_eq!(chats.blocks.len(), 3, "markup and talk-page lines are skipped");
        let opening = &chats.blocks[0].memories;
        assert!(opening[0].human.contains("april"), "{:?}", opening[0].human);
        assert_eq!(opening[0].bot, "April (Apr.) is the fourth month of the year in the calendar.");
        assert_eq!(opening[1].human, "It has thirty days in it every year.");
        assert_eq!(opening[1].bot, "Many festivals happen in April each year.");

        let body = &chats.blocks[1].memories;
        assert_eq!(body.len(), 2);
        assert_eq!(body[0].human, "April ends on the same day of the week as December every single year.");
        assert_eq!(body[1].bot, "Rain is common in many places.");

        // Entities decoded, trailing categories removed.
        let rome = &chats.blocks[2].memories;
        assert_eq!(rome.len(), 1);
        assert_eq!(rome[0].human, "Rome - the capital of Italy - is very old.");
        assert_eq!(rome[0].bot, "It was founded long ago by Romulus.");
    }

    #[test]
    fn language_quotes_answer_a_category_request() {
        let path = temp_file("quotes.csv", concat!(
            "quote,author,category\n",
            "\"Be yourself; everyone else is already taken.\",Oscar Wilde,\"attributed-no-source, be-yourself, honesty\"\n",
            "Too short,Someone,life\n",
            "Simple words can still mean a lot.,Someone,\n",
        ));
        let chats = load_quote_chats(path.to_str().unwrap()).unwrap();
        remove(&path);

        assert_eq!(chats.blocks.len(), 2);
        let first = &chats.blocks[0].memories[0];
        assert!(first.human.contains("be yourself"), "{:?}", first.human);
        assert_eq!(first.bot, "Be yourself; everyone else is already taken.");
        assert_eq!(chats.blocks[1].memories[0].human, "share a quote");
    }
}
