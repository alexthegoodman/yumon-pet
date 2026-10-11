use super::*;
use crate::brain::code_corpus;

pub(crate) const PRINT_HTML: &str = r##"<!DOCTYPE html><html><head><script>var x = 1;</script></head><body>
<nav class="sidebar"><ol><li><a href="#intro">Intro</a></li></ol></nav>
<main>
<h1 id="intro"><a class="header" href="#intro">Getting Started</a></h1>
<div class="rule" id="r-intro"><a class="rule-link" href="#r-intro"><span>[intro<wbr>.rule]</span></a></div>
<div class="tests-popup popup-hidden">Tests with this rule:<ul><li><a href="x">tests/ui/a.rs</a></li></ul></div>
<p>Use <code>cargo new</code> to start
a project &amp; build it.</p>
<ul><li><a href="#a">Only a link</a></li></ul>
<ol><li><p>First <em>step</em></p></li><li>Second step<ul><li>nested</li></ul></li></ol>
<div class="listing"><div class="file-name">Filename: src/main.rs</div>
<pre><code class="language-rust ignore edition2024"><span class="boring">#![allow(unused)]
</span><span class="boring">fn main() {
</span>let x = 1 &lt; 2;
<span class="boring">}</span></code></pre></div>
<blockquote><p>Note: hidden lines stay.</p></blockquote>
<table><thead><tr><th>Name</th><th>Kind</th></tr></thead><tbody><tr><td><code>u8</code></td><td>int</td></tr></tbody></table>
<h2 id="more"><a class="header" href="#more">More</a></h2>
<p>Second section text.</p>
<h1 id="test-summary"><a class="header" href="#test-summary">Test summary</a></h1>
<p>Coverage table that must be skipped.</p>
<h1 id="next"><a class="header" href="#next">Next Chapter</a></h1>
<p>Chapter two.</p>
</main></body></html>"##;

pub(crate) fn tokenizer(directory: &Path) -> TokenizerKind {
    std::fs::create_dir_all(directory).unwrap();
    let source = directory.join("tokenizer-source.txt");
    std::fs::write(&source, PRINT_HTML).unwrap();
    let path = directory.join("tokenizer");
    code_corpus::train_tokenizer(&[source], 320, path.to_str().unwrap()).unwrap();
    code_corpus::load_tokenizer(path.to_str().unwrap()).unwrap()
}

fn temp() -> PathBuf {
    std::env::temp_dir().join(format!("yumon-docs-{}", uuid::Uuid::new_v4()))
}

#[test]
fn print_html_becomes_markdown_blocks() -> Result<()> {
    let blocks = html_blocks(PRINT_HTML)?;
    assert_eq!(blocks[..9], [
        Block::Heading(1, "Getting Started".into()),
        Block::Text("Use `cargo new` to start\na project & build it.".into()),
        Block::Text("1. First step\n2. Second step\n  - nested".into()),
        Block::Text("Filename: src/main.rs".into()),
        Block::Code("```rust,ignore,edition2024\n#![allow(unused)]\nfn main() {\nlet x = 1 < 2;\n}\n```".into()),
        Block::Text("> Note: hidden lines stay.".into()),
        Block::Text("| Name | Kind |\n| --- | --- |\n| `u8` | int |".into()),
        Block::Heading(2, "More".into()),
        Block::Text("Second section text.".into()),
    ]);
    let all = format!("{blocks:?}");
    for noise in ["Tests with this rule", "intro.rule", "Only a link", "var x", "Intro\""] {
        assert!(!all.contains(noise), "{noise} leaked into {all}");
    }
    Ok(())
}

#[test]
fn chunks_fit_context_start_at_chapters_and_never_split_code() -> Result<()> {
    let directory = temp();
    let tokenizer = tokenizer(&directory);
    let blocks = html_blocks(PRINT_HTML)?;
    let (chunks, stats) = chunk_book("Fixture Book", &blocks, &tokenizer, 512)?;
    assert_eq!(chunks.len(), 2);
    assert!(chunks[0].1.starts_with("Fixture Book > Getting Started\n\nUse `cargo new`"));
    assert!(chunks[0].1.contains("## More\n\nSecond section text."));
    assert_eq!(chunks[1].1, "Fixture Book > Next Chapter\n\nChapter two.");
    assert!(chunks.iter().all(|(_, text)| !text.contains("Coverage table")));
    assert_eq!(stats.skipped_code_blocks, 0);

    // A tight context skips the whole code block rather than truncating it, and
    // regroups long prose by lines without exceeding the budget.
    let long = Block::Text((0..40).map(|i| format!("Line {i} of prose.")).collect::<Vec<_>>().join("\n"));
    let small = [Block::Heading(1, "Tight".into()), long, Block::Heading(2, "Code".into()),
        Block::Code(format!("```rust\n{}\n```", "let value = 1;\n".repeat(30)))];
    let (chunks, stats) = chunk_book("T", &small, &tokenizer, 64)?;
    assert_eq!(stats.skipped_code_blocks, 1);
    assert_eq!(stats.split_text_blocks, 1);
    assert!(chunks.len() > 2);
    for (_, text) in &chunks {
        assert!(tokenizer.encode(text).len() <= 63, "{text}");
        assert!(!text.contains("let value"));
    }
    let joined: String = chunks.iter().map(|(_, t)| t.as_str()).collect();
    for i in 0..40 { assert!(joined.contains(&format!("Line {i} of prose."))); }
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn prepared_stage_round_trips_and_rejects_tampering() -> Result<()> {
    let directory = temp();
    let tokenizer = tokenizer(&directory);
    let html = directory.join("html");
    std::fs::create_dir_all(html.join("fixture"))?;
    std::fs::write(html.join("fixture/print.html"), PRINT_HTML)?;
    let config = CodeConfig { max_seq_len: 512, ..Default::default() };
    let output = directory.join("docs");
    let report = prepare(&config, &tokenizer, &html, "rustc test", &["fixture".into()], &output, 0)?;
    assert_eq!(report["samples"], 2);
    assert!(prepare(&config, &tokenizer, &html, "rustc test", &["fixture".into()], &output, 0).is_err());
    let (samples, identity) = load(&output, &config, &tokenizer)?;
    assert_eq!(samples.len(), 2);
    assert_eq!(identity["toolchain"], "rustc test");
    assert!(samples.iter().all(|s| s.pair.0 == "rust-docs/fixture"
        && s.input_ids[0] == BOS_TOKEN && s.target_labels.contains(&EOS_TOKEN)));
    let other_context = CodeConfig { max_seq_len: 256, ..config.clone() };
    assert!(load(&output, &other_context, &tokenizer).is_err());
    let cache = output.join(CACHE_FILE);
    let mut bytes = std::fs::read(&cache)?;
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&cache, bytes)?;
    assert!(load(&output, &config, &tokenizer).unwrap_err().to_string().contains("digest mismatch"));
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
