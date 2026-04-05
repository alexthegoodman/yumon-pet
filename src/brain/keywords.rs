// use keyword_extraction::prelude::*;   // or specific imports
#[cfg(target_os = "windows")]
use keyword_extraction::yake::Yake;   // or text_rank::TextRank, etc.
#[cfg(target_os = "windows")]
use stop_words::{get, LANGUAGE};

// Usually you process per-document / per-sentence
#[cfg(target_os = "windows")]
pub fn extract_keywords(text: &str) -> Vec<(String, f32)> {
    let stop_words = get(LANGUAGE::English);

    // Option A: YAKE (very good default choice in many cases)
    let extractor = Yake::new(keyword_extraction::yake::YakeParams::WithDefaults(text, &stop_words));            // up to trigrams

    // let keywords = extractor.extract(text);

    let n = 2;

    let keywords = extractor.get_ranked_keyword_scores(n);

    // println!("prepare paired samples {:?}", keywords.get(0));

    // Returns sorted Vec<(keyword: String, score: f64)> — lower score = better
    keywords.into_iter()
        .take(n)                       // top-k
        .collect()
}

// Option B: TextRank (graph-based → often catches thematic words better)
// fn extract_with_textrank(text: &str) -> Vec<String> {
//     let mut extractor = keyword_extraction::textrank::TextRank::new();
//     // configure if needed: window, damping factor, etc.

//     let ranked = extractor.rank(text);
//     ranked.into_iter()
//         .map(|(kw, _score)| kw)
//         .take(5)
//         .collect()
// }