use crate::brain::wiki::is_good_sentence;
use anyhow::Result;

pub fn load_mdx_sentences(mdx_dir: &str) -> Result<Vec<String>> {
    println!("📖 Scanning MDX directory: {mdx_dir}");

    let mut sentences = Vec::new();
    let mut files_loaded = 0usize;

    for entry in walkdir::WalkDir::new(mdx_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("mdx"))
    {
        let path = entry.path();
        let content = std::fs::read_to_string(path)?;

        for line in content.lines() {
            let trimmed = line.trim().replace("<br />", "");
            if is_good_sentence(&trimmed) && !trimmed.starts_with("# ") {
                sentences.push(trimmed.to_string());
            }
        }

        files_loaded += 1;
        if files_loaded % 100 == 0 {
            println!("  … {files_loaded} files, {} sentences so far", sentences.len());
        }
    }

    let mut final_sentences = Vec::new();

    for sents in sentences.chunks(2) { // appropriate length, decent connections
        final_sentences.push(sents.join(" "));
    }

    println!("✅ Loaded {} sentences from {} MDX files", final_sentences.len(), files_loaded);
    Ok(final_sentences)
}

pub fn load_notion_sentences(notion_dir: &str) -> Result<Vec<String>> {
    println!("📖 Scanning Notion directory: {notion_dir}");

    let mut sentences = Vec::new();
    let mut files_loaded = 0usize;

    for entry in walkdir::WalkDir::new(notion_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
    {
        let path = entry.path();
        let content = std::fs::read_to_string(path)?;

        for line in content.lines() {
            let trimmed = line.trim();

            for sent in trimmed.split(".") {
                if is_good_sentence(sent) {
                    sentences.push(sent.to_string());
                }
            }
        }

        files_loaded += 1;
        if files_loaded % 100 == 0 {
            println!("  … {files_loaded} files, {} sentences so far", sentences.len());
        }
    }

    println!("✅ Loaded {} sentences from {} Notion files", sentences.len(), files_loaded);
    Ok(sentences)
}

// #[derive(Debug)]
// pub struct Quote {
//     pub text: String,
//     pub author: String,
//     pub tags: Vec<String>,
// }

pub fn load_csv_quotes(csv_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading quotes CSV: {csv_path}");

    let mut rdr = csv::Reader::from_path(csv_path)?;
    let mut quotes = Vec::new();

    let mut count = 0;

    for result in rdr.records() {
        let record = result?;

        let text   = record.get(0).unwrap_or("").trim().to_string();
        let author = record.get(1).unwrap_or("").trim().to_string();
        let tags   = record
            .get(2)
            .unwrap_or("")
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>();

        if text.is_empty() {
            continue;
        }

        quotes.push(text);

        count = count + 1;

        if count > 100_000 { break; }
    }

    println!("✅ Loaded {} quotes from {csv_path}", quotes.len());
    Ok(quotes)
}

pub fn load_csv_qna(csv_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading qna CSV: {csv_path}");

    let mut rdr = csv::Reader::from_path(csv_path)?;
    let mut quotes = Vec::new();

    let mut count = 0;

    for result in rdr.records() {
        let record = result?;

        let question   = record.get(0).unwrap_or("").trim().to_string();
        let answer = record.get(1).unwrap_or("").trim().to_string();

        quotes.push(question + " " + &answer);

        count = count + 1;

        if count > 10_000 { break; }
    }

    println!("✅ Loaded {} questions and answers from {csv_path}", quotes.len());
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

pub fn load_csv_bible(bible_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading bible CSV: {bible_path}");

    let mut rdr = csv::Reader::from_path(bible_path)?;
    let mut quotes = Vec::new();
    let mut buffer = String::new();

    let mut count = 0;

    for result in rdr.records() {
        let record = result?;

        let bk = record.get(1).unwrap_or("").trim().to_string();
        let verse = record.get(4).unwrap_or("").trim().to_string();

        if (bk == "20") { // proverbs only right now
            if verse.is_empty() {
                continue;
            }

            if buffer.is_empty() {
                buffer.push_str(&verse);
            } else {
                buffer.push(' ');
                buffer.push_str(&verse);
            }

            if buffer.len() >= 90 {
                quotes.push(buffer.clone());
                buffer.clear();

                count += 1;
                if count >= 20_000 {
                    break;
                }
            }
        }
    }

    if !buffer.is_empty() {
        quotes.push(buffer);
    }

    println!("✅ Loaded {} verses from {bible_path}", quotes.len());
    Ok(quotes)
}

pub fn load_dictionary_sentences(dict_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading dictionary: {dict_path}");

    let content = std::fs::read_to_string(dict_path)?;
    let mut sentences = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip empty lines or very short lines
        if trimmed.len() < 10 {
            continue;
        }

        // Skip lines that are just a single letter (section headers like "A")
        if trimmed.chars().all(|c| c.is_alphabetic() || c.is_whitespace())
            && trimmed.split_whitespace().count() <= 1
        {
            continue;
        }

        // Try to extract a clean definition sentence
        // if let Some(sentence) = extract_definition(trimmed) {
            if is_good_sentence(&trimmed) {
                sentences.push(trimmed.to_string());
            }
        // }
    }

    println!("✅ Loaded {} dictionary sentences", sentences.len());
    Ok(sentences)
}

pub fn load_handcrafted_sentences(dict_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading handcrafted: {dict_path}");

    let content = std::fs::read_to_string(dict_path)?;
    let mut sentences = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim().replace("<br />", "");

         if is_good_sentence(&trimmed) {
            sentences.push(trimmed.to_string());
        }
    }

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

    let mut blocks = Vec::new();
    let mut current_block: Vec<String> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if !current_block.is_empty() {
                blocks.push(parse_block(&current_block));
                current_block.clear();
            }
        } else {
            current_block.push(trimmed.to_string());
        }
    }

    // Handle trailing block with no final blank line
    if !current_block.is_empty() {
        blocks.push(parse_block(&current_block));
    }

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
    let mut sentences = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        for sent in trimmed.split(".") {
            if is_good_sentence(&sent) && sent.len() > 50 { // have a nice sensible minimum length
                sentences.push(sent.to_string());
            }
        }
    }

    println!("✅ Loaded {} txt sentences", sentences.len());
    Ok(sentences)
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

    let mut pairs = Vec::new();

    // 2. Iterate through lines in steps of 2
    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let question = chunk[0].clone();
            let answer = chunk[1].clone();
            
            // You can still apply your "is_good" logic here
            if is_good_sentence(&(question.clone() + &answer)) {
                pairs.push((question, answer));
            }
        }
    }

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

    let mut pairs = Vec::new();

    // 2. Iterate through lines in steps of 2
    for chunk in lines.chunks(2) {
        if chunk.len() == 2 {
            let question = chunk[0].clone();
            let answer = chunk[1].clone();
            
            // You can still apply your "is_good" logic here
            if is_good_sentence(&question) && is_good_sentence(&answer) {
                pairs.push(question + " " + &answer);
            }
        }
    }

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