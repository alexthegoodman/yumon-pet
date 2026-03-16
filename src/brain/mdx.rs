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
            let trimmed = line.trim();
            if is_good_sentence(trimmed) {
                sentences.push(trimmed.to_string());
            }
        }

        files_loaded += 1;
        if files_loaded % 100 == 0 {
            println!("  … {files_loaded} files, {} sentences so far", sentences.len());
        }
    }

    println!("✅ Loaded {} sentences from {} MDX files", sentences.len(), files_loaded);
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

        if count > 1000 { break; }
    }

    println!("✅ Loaded {} quotes from {csv_path}", quotes.len());
    Ok(quotes)
}