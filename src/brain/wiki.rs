/// SimpleWiki XML article parser.
///
/// Reads `simplewiki-latest-pages-articles.xml` and extracts clean
/// plain-text sentences suitable for language model pre-training.
///
/// Pipeline:
///   XML → extract <text> content → strip wiki markup → sentence split
///
/// We keep sentences that are:
///   - Between 20 and 200 characters long
///   - Mostly ASCII
///   - Don't start with special wiki chars (=, {, |, !)

use anyhow::Result;
use burn::prelude::ToElement;
use quick_xml::Reader;
use quick_xml::events::Event;
use std::io::BufRead;

use crate::brain::train::{MAX_SEQ_LEN, MAX_SEQ_LEN_CHARS};

use std::fs::File;
use std::io::{BufWriter, Write};

pub fn save_sentences_to_file(sentences: &[String], output_path: &str) -> Result<()> {
    println!("💾 Saving {} sentences to: {}", sentences.len(), output_path);
    
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    for sentence in sentences {
        // We write each sentence followed by a newline
        writeln!(writer, "{}", sentence)?;
    }

    writer.flush()?; // Ensure everything is written to disk
    println!("✅ Save complete.");
    Ok(())
}

/// Returns a vector of clean sentences, up to `max_articles` articles processed.
pub fn load_wiki_sentences(xml_path: &str, max_articles: usize, split_count: usize) -> Result<Vec<String>> {
    println!("📖 Parsing SimpleWiki XML: {xml_path}");

    let file   = std::fs::File::open(xml_path)?;
    let reader = std::io::BufReader::new(file);
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut sentences    = Vec::new();
    // let mut in_text      = false;
    let mut buf          = Vec::new();
    let mut articles     = 0usize;

    // loop {
    //     match xml.read_event_into(&mut buf) {
    //         Ok(Event::Start(ref e)) if e.name().as_ref() == b"text" => {
    //             in_text = true;
    //         }
    //         Ok(Event::Text(ref e)) if in_text => {
    //             let raw = e.unescape().unwrap_or_default();
    //             let cleaned = strip_wiki_markup(&raw);
    //             let mut n = 0;
    //             let split = split_sentences_chunked(&cleaned);
    //             for s in split {
    //                 if is_good_sentence(&s) {
    //                     if n > 250 { break; }
    //                     sentences.push(s);
    //                     n = n + 1;
    //                 }
    //             }
    //             in_text = false;
    //             articles += 1;
    //             if articles % 5000 == 0 {
    //                 println!("  … {articles} articles, {} sentences so far", sentences.len());
    //             }
    //             if max_articles > 0 && articles >= max_articles { break; }
    //         }
    //         Ok(Event::End(ref e)) if e.name().as_ref() == b"text" => {
    //             in_text = false;
    //         }
    //         Ok(Event::Eof) => break,
    //         Err(e) => {
    //             eprintln!("XML error at {}: {e}", xml.buffer_position());
    //             break;
    //         }
    //         _ => {}
    //     }
    //     buf.clear();
    // }

    // 1. Update the loop to track if we are inside an actual article page
    let mut in_page = false; 
    let mut in_text = false;

    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"page" => in_page = true,
                b"text" if in_page => in_text = true,
                _ => {}
            },
            Ok(Event::Text(ref e)) if in_text => {
                let raw = e.unescape().unwrap_or_default();
                
                // SKIP: If the text starts with #REDIRECT, it's not an article
                if raw.to_uppercase().starts_with("#REDIRECT") { 
                    in_text = false; 
                    continue; 
                }

                let cleaned = strip_wiki_markup(&raw);
                let split = split_sentences_chunked(&cleaned);
                
                for s in split {
                    if is_good_sentence(&s) {
                        sentences.push(clean_sentence(&s));
                    }
                }
                // Don't set in_text = false here; let the End event do it
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"page" => {
                    in_page = false;
                    articles += 1;
                    if articles % 1000 == 0 { println!("Loading articles..."); }
                    if max_articles > 0 && articles >= max_articles { break; }
                },
                b"text" => in_text = false,
                _ => {}
            },
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }

    println!("✅ Loaded {} sentences from {} articles", sentences.len(), articles);
    Ok(sentences)
}

/// Very lightweight wiki markup stripper (handles most common patterns).
fn strip_wiki_markup(text: &str) -> String {
    let mut out     = String::with_capacity(text.len());
    let mut chars   = text.chars().peekable();
    let mut depth   = 0i32;  // bracket nesting

    while let Some(c) = chars.next() {
        match c {
            // Remove [[links|display]] — keep display text
            '[' if chars.peek() == Some(&'[') => {
                chars.next(); // consume second [
                depth += 1;
                // Skip to | or ]]
                let mut inner = String::new();
                let mut pipe_idx = None;
                loop {
                    match chars.next() {
                        Some(']') if chars.peek() == Some(&']') => {
                            chars.next();
                            depth -= 1;
                            break;
                        }
                        Some('|') if pipe_idx.is_none() => {
                            pipe_idx = Some(inner.len());
                        }
                        Some(ch) => inner.push(ch),
                        None     => break,
                    }
                }
                // Keep the display part (after |), or full link text
                let display = if let Some(pi) = pipe_idx {
                    inner[pi..].to_string()
                } else {
                    inner
                };
                out.push_str(&display);
            }
            // Remove {{templates}}
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                depth += 1;
                while let Some(ch) = chars.next() {
                    if ch == '}' && chars.peek() == Some(&'}') {
                        chars.next();
                        depth -= 1;
                        break;
                    }
                }
            }
            // Remove <ref>...</ref> and other HTML tags
            '<' => {
                while let Some(ch) = chars.next() {
                    if ch == '>' { break; }
                }
            }
            // Remove == headings ==, skip the = signs
            '=' => {}
            // Remove leading */#/: list markers (handled in sentence filter)
            _ => out.push(c),
        }
    }

    // Collapse whitespace
    let mut result = String::new();
    let mut prev_space = false;
    for c in out.chars() {
        if c.is_whitespace() {
            if !prev_space { result.push(' '); }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

/// Split text into sentences on `. `, `! `, `? ` boundaries.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current   = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        current.push(chars[i]);
        if matches!(chars[i], '.' | '!' | '?') {
            let next_is_space = i + 1 < len && chars[i + 1] == ' ';
            let next_is_upper = i + 2 < len && chars[i + 2].is_uppercase();
            if next_is_space && (next_is_upper || i + 2 >= len) {
                sentences.push(current.trim().to_string());
                current = String::new();
            }
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}

/// Split text into sentences on `. `, `! `, `? ` boundaries,
/// returning chunks of `n` consecutive sentences joined together.
// fn split_sentences_chunked(text: &str, n: usize) -> Vec<String> {
//     let mut sentences = Vec::new();
//     let mut current   = String::new();

//     let chars: Vec<char> = text.chars().collect();
//     let len = chars.len();

//     for i in 0..len {
//         current.push(chars[i]);
//         if matches!(chars[i], '.' | '!' | '?') {
//             let next_is_space = i + 1 < len && chars[i + 1] == ' ';
//             let next_is_upper = i + 2 < len && chars[i + 2].is_uppercase();
//             if next_is_space && (next_is_upper || i + 2 >= len) {
//                 sentences.push(current.trim().to_string());
//                 current = String::new();
//             }
//         }
//     }
//     if !current.trim().is_empty() {
//         sentences.push(current.trim().to_string());
//     }

//     // Group into chunks of n
//     // Group into chunks of n, then filter by combined length
//     let min_len = if n > 1 { 50 } else { 30 };

//     sentences
//         .chunks(n)
//         .map(|chunk| chunk.join(" "))
//         .filter(|chunk| chunk.len() > min_len)
//         .collect()
// }

fn split_sentences_chunked(text: &str) -> Vec<String> {
    let min_len = (MAX_SEQ_LEN_CHARS as f32 * 0.65).floor().to_usize();
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    // 1. Split into individual sentences (Your original logic)
    for i in 0..len {
        current.push(chars[i]);
        if matches!(chars[i], '.' | '!' | '?') {
            let next_is_space = i + 1 < len && chars[i + 1] == ' ';
            if next_is_space {
                sentences.push(current.trim().to_string());
                current = String::new();
            }
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    // 2. Greedy grouping by length
    let mut result = Vec::new();
    let mut buffer = String::new();

    for sentence in sentences {
        if !buffer.is_empty() {
            buffer.push(' ');
        }
        buffer.push_str(&sentence);

        // If we've hit the minimum length, save this chunk and start a new one
        if buffer.len() >= min_len {
            result.push(buffer);
            buffer = String::new();
        }
    }

    // Handle the leftovers: 
    // Usually, we append the last small bit to the previous chunk 
    // or keep it as its own if the result is empty.
    if !buffer.is_empty() {
        if let Some(last_chunk) = result.last_mut() {
            last_chunk.push(' ');
            last_chunk.push_str(&buffer);
        } else {
            result.push(buffer);
        }
    }

    result
}

// pub fn is_good_sentence(s: &str) -> bool {
//     let len = s.len();
//     if len < 20 || len > 4000 { return false; }

//     if s.contains("'''") { return false; }
//     if s.contains("[[") { return false; }
//     if s.contains("]]") { return false; }
//     if s.contains("|") { return false; }
//     if s.contains("*") { return false; }
//     if s.contains("#") { return false; }
//     if s.contains("''") { return false; }
//     if s.contains("ISBN") { return false; }
//     if s.contains(".com") { return false; }
//     if s.contains(".net") { return false; }
//     if s.contains(".org") { return false; }
//     // if s.contains("{") { return false; } // keep for json
//     // if s.contains("}") { return false; }
//     if s.contains(".md") { return false; }
//     if s.contains("~") { return false; }
//     if s.contains("http") { return false; }

//     // Reject lines starting with wiki junk
//     let first = s.chars().next().unwrap_or(' ');
//     if matches!(first, '=' | '{' | '|' | '!' | '*' | '#' | ':') { return false; }

//     // Require mostly ASCII
//     let ascii_ratio = s.chars().filter(|c| c.is_ascii()).count() as f32 / s.chars().count() as f32;
//     if ascii_ratio < 0.90 { return false; }

//     // Must contain at least one space (not a single token)
//     s.contains(' ')
// }

use regex::Regex;
use std::borrow::Cow;

pub fn clean_sentence(s: &str) -> String {
    // 1. Unescape common HTML entities
    let decoded = s.replace("&quot;", "\"")
                   .replace("&amp;", "&")
                   .replace("&lt;", "<")
                   .replace("&gt;", ">")
                   .replace("&nbsp;", " ");

    // 2. Remove specific Wikitext patterns like [[Internal Links|Display Text]] 
    // This regex targets the [[Target|Text]] format and keeps only 'Text'
    let re_links = Regex::new(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]").unwrap();
    let cleaned_links = re_links.replace_all(&decoded, "$1");

    // 3. Remove citations like [1], [23], or [citation needed]
    let re_cites = Regex::new(r"\[\d+\]|\[edit\]|\[citation needed\]").unwrap();
    let final_clean = re_cites.replace_all(&cleaned_links, "");

    final_clean.trim().to_string()
}

pub fn is_good_sentence(s: &str) -> bool {
    let len = s.len();
    // Adjusted bounds: 4000 is quite long for a sentence, but fine for a "chunk"
    if len < 5 || len > 4000 { return false; }

    // List of "red flag" patterns and their allowed limits
    // If any of these appear MORE than 4 times, we reject the string.
    let red_flags = [
        // "'''", 
        // "[[", 
        // "]]", 
        // "|",
        //  "*", 
        //  "#", 
        //  "''", 
        //  "ISBN", 
        // ".com", 
        // ".net", 
        // ".org", 
        // ".md", 
        // "~", 
        // "http", 
        "background:",
        "width:",
        "%;",
        "em;",
        "px;",
        "||",
        // "\\",
        // "\\\\",
        // "\\\\\\"
    ];

    for flag in red_flags {
        if s.matches(flag).count() > 4 {
            return false;
        }
    }

    // Strict rejection for starts-of-lines (Wiki structural junk)
    // let first = s.chars().next().unwrap_or(' ');
    // if matches!(first, '=' | '{' | '|' | '!' | '*' | '#' | ':') { 
    //     return false; 
    // }

    // Require mostly ASCII (90%)
    // let total_chars = s.chars().count();
    // let ascii_count = s.chars().filter(|c| c.is_ascii()).count();
    // if (ascii_count as f32 / total_chars as f32) < 0.90 { 
    //     return false; 
    // }

    // Must contain at least one space
    s.contains(' ')
}
