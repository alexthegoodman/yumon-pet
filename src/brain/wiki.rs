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
use quick_xml::Reader;
use quick_xml::events::Event;
use std::io::BufRead;

/// Returns a vector of clean sentences, up to `max_articles` articles processed.
pub fn load_wiki_sentences(xml_path: &str, max_articles: usize) -> Result<Vec<String>> {
    println!("📖 Parsing SimpleWiki XML: {xml_path}");

    let file   = std::fs::File::open(xml_path)?;
    let reader = std::io::BufReader::new(file);
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut sentences    = Vec::new();
    let mut in_text      = false;
    let mut buf          = Vec::new();
    let mut articles     = 0usize;

    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"text" => {
                in_text = true;
            }
            Ok(Event::Text(ref e)) if in_text => {
                let raw = e.unescape().unwrap_or_default();
                let cleaned = strip_wiki_markup(&raw);
                for s in split_sentences(&cleaned) {
                    if is_good_sentence(&s) {
                        sentences.push(s);
                    }
                }
                in_text = false;
                articles += 1;
                if articles % 5000 == 0 {
                    println!("  … {articles} articles, {} sentences so far", sentences.len());
                }
                if max_articles > 0 && articles >= max_articles { break; }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"text" => {
                in_text = false;
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                eprintln!("XML error at {}: {e}", xml.buffer_position());
                break;
            }
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

fn is_good_sentence(s: &str) -> bool {
    let len = s.len();
    if len < 20 || len > 200 { return false; }

    // Reject lines starting with wiki junk
    let first = s.chars().next().unwrap_or(' ');
    if matches!(first, '=' | '{' | '|' | '!' | '*' | '#' | ':') { return false; }

    // Require mostly ASCII
    let ascii_ratio = s.chars().filter(|c| c.is_ascii()).count() as f32 / s.chars().count() as f32;
    if ascii_ratio < 0.90 { return false; }

    // Must contain at least one space (not a single token)
    s.contains(' ')
}
