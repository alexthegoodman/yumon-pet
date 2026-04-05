// use crate::brain::wiki::is_good_sentence;
// use anyhow::Result;

// pub fn load_pdf_ebook_sentences(pdf_path: &str) -> Result<Vec<String>> {
//     println!("📖 Loading ebook PDF: {pdf_path}");

//     let bytes = std::fs::read(pdf_path)?;
//     let raw = pdf_extract::extract_text_from_mem(&bytes)?;
//     let mut sentences = Vec::new();

//     let mut count = 0;
//     for line in raw.lines() {
//         let trimmed = line.trim();

//         // Split on ". " so we catch multiple sentences packed onto one line
//         // (common in justified ebook text that pdftotext flattens).
//         for fragment in trimmed.split(". ") {
//             let frag = fragment.trim();
//             if is_good_sentence(frag) {
//                 sentences.push(frag.to_string());
//             }
//         }

//         count = count + 1;

//         if count > 10_000 { break; }
//     }

//     println!("✅ Loaded {} sentences from {pdf_path}", sentences.len());
//     Ok(sentences)
// }

use crate::brain::{train::MAX_SEQ_LEN, wiki::is_good_sentence};
use anyhow::Result;

// pub fn load_pdf_ebook_sentences(pdf_path: &str) -> Result<Vec<String>> {
//     println!("📖 Loading ebook PDF: {pdf_path}");

//     let bytes = std::fs::read(pdf_path)?;
//     let raw = pdf_extract::extract_text_from_mem(&bytes)?;
    
//     let mut sentences = Vec::new();
//     let mut current_buffer = String::new();

//     // 1. Clean up the raw text by joining lines with a space
//     // to handle sentences split across line breaks.
//     for line in raw.lines() {
//         let trimmed = line.trim();
//         if trimmed.is_empty() { continue; }

//         if !current_buffer.is_empty() {
//             current_buffer.push(' ');
//         }
//         current_buffer.push_str(trimmed);

//         // 2. Check if the buffer contains a sentence-ending period
//         // We use a loop in case there are multiple sentences now in the buffer
//         while let Some(dot_index) = current_buffer.find(". ") {
//             // Extract everything up to and including the period
//             let sentence: String = current_buffer.drain(..dot_index + 1).collect();
//             let cleaned = sentence.trim();

//             if is_good_sentence(cleaned) {
//                 sentences.push(cleaned.to_string());
//             }

//             // Cap it to prevent infinite loops or memory bloat on massive files
//             if sentences.len() > 20_000 {
//                 return Ok(sentences);
//             }
//         }
//     }

//     println!("✅ Loaded {} sentences from {pdf_path}", sentences.len());
//     Ok(sentences)
// }

#[cfg(target_os = "windows")]
use rand::Rng; // Add this for random range generation

#[cfg(target_os = "windows")]
pub fn load_pdfs(paths: Vec<String>) -> Vec<String> {
    let mut all_samples = Vec::new();

    for path in paths {
        let mut ebooks = load_pdf_ebook_sentences(&path);
        let mut ebooks = ebooks.as_ref().expect("Couldn't get ebook");

        all_samples.extend(ebooks.clone());
    }

    all_samples
}

#[cfg(target_os = "windows")]
pub fn load_pdf_ebook_sentences(pdf_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading ebook PDF: {pdf_path}");

    let bytes = std::fs::read(pdf_path)?;
    let raw = pdf_extract::extract_text_from_mem(&bytes)?;
    
    let mut raw_sentences = Vec::new();
    let mut current_buffer = String::new();

    let mut line_num = 0;

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        if !current_buffer.is_empty() {
            current_buffer.push(' ');
        }
        current_buffer.push_str(trimmed);

        line_num = line_num + 1;

        while let Some(dot_index) = current_buffer.find(". ") {
            let sentence: String = current_buffer.drain(..dot_index + 1).collect();
            let cleaned = sentence.trim();

            if is_good_sentence(cleaned) && line_num > 200 { // skip intro junk
                raw_sentences.push(cleaned.to_string());
            }

            if raw_sentences.len() > 50_000 {
                break; 
            }
        }
    }

    // // --- New Concatenation Logic ---
    // let mut rng = rand::thread_rng();
    // let mut final_sentences = Vec::new();
    // let mut i = 0;

    // while i < raw_sentences.len() {
    //     // Randomly pick a chunk size between 1 and 4
    //     let chunk_size = rng.gen_range(2..=4);
        
    //     // Take up to 'chunk_size' elements, ensuring we don't go out of bounds
    //     let end = (i + chunk_size).min(raw_sentences.len());
    //     let combined = raw_sentences[i..end].join(" ");
        
    //     final_sentences.push(combined);
    //     i = end;
    // }

    // --- New Concatenation Logic ---
    let mut final_sentences = Vec::new();
    let mut i = 0;

    let bottom_range = MAX_SEQ_LEN * 1;
    let top_range = MAX_SEQ_LEN * 3;

    while i < raw_sentences.len() {
        let mut combined = String::new();

        while i < raw_sentences.len() {
            let next = &raw_sentences[i];
            let candidate = if combined.is_empty() {
                next.clone()
            } else {
                format!("{combined} {next}")
            };

            if candidate.len() > top_range {
                // If combined is already within range, stop here
                if combined.len() >= bottom_range {
                    break;
                }
                // Otherwise we have no choice but to accept the overshoot
                combined = candidate;
                i += 1;
                break;
            }

            combined = candidate;
            i += 1;

            if combined.len() >= bottom_range {
                break;
            }
        }

        if !combined.is_empty() {
            final_sentences.push(combined);
        }
    }

    println!("✅ Loaded {} grouped entries from {pdf_path}", final_sentences.len());
    Ok(final_sentences)
}