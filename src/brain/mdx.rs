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