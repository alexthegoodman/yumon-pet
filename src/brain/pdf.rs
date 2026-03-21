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

use crate::brain::wiki::is_good_sentence;
use anyhow::Result;

pub fn load_pdf_ebook_sentences(pdf_path: &str) -> Result<Vec<String>> {
    println!("📖 Loading ebook PDF: {pdf_path}");

    let bytes = std::fs::read(pdf_path)?;
    let raw = pdf_extract::extract_text_from_mem(&bytes)?;
    
    let mut sentences = Vec::new();
    let mut current_buffer = String::new();

    // 1. Clean up the raw text by joining lines with a space
    // to handle sentences split across line breaks.
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        if !current_buffer.is_empty() {
            current_buffer.push(' ');
        }
        current_buffer.push_str(trimmed);

        // 2. Check if the buffer contains a sentence-ending period
        // We use a loop in case there are multiple sentences now in the buffer
        while let Some(dot_index) = current_buffer.find(". ") {
            // Extract everything up to and including the period
            let sentence: String = current_buffer.drain(..dot_index + 1).collect();
            let cleaned = sentence.trim();

            if is_good_sentence(cleaned) {
                sentences.push(cleaned.to_string());
            }

            // Cap it to prevent infinite loops or memory bloat on massive files
            if sentences.len() > 20_000 {
                return Ok(sentences);
            }
        }
    }

    println!("✅ Loaded {} sentences from {pdf_path}", sentences.len());
    Ok(sentences)
}