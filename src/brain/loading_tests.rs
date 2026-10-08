use super::{
    loader::{DataLoader, FileKind},
    mdx,
    samples::TrainingStage,
};
use rayon::ThreadPoolBuilder;
use std::path::PathBuf;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("yumon-loading-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, content: impl AsRef<[u8]>) -> String {
        let path = self.0.join(name);
        std::fs::write(&path, content).unwrap();
        path.to_str().unwrap().to_string()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn text_and_pairs_keep_order_across_batches() {
    let fixture = Fixture::new();
    let lines: Vec<_> = (0..2051)
        .map(|i| format!("This is a sensible training sentence number {i}."))
        .collect();
    let path = fixture.write("lines.txt", lines.join("\r\n"));
    for threads in [1, 4] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                assert_eq!(mdx::load_txt_lines(&path).unwrap(), lines);
                let pairs = mdx::load_qa_pairs(&path).unwrap();
                assert_eq!(pairs.len(), lines.len() / 2);
                for (pair, expected) in pairs.iter().zip(lines.chunks_exact(2)) {
                    assert_eq!(pair, &(expected[0].clone(), expected[1].clone()));
                }
            });
    }
}

#[test]
fn chats_preserve_boundaries_and_csv_quoted_newlines() {
    let fixture = Fixture::new();
    let path = fixture.write(
        "chats.txt",
        "human one\r\nbot one\r\nunanswered\r\n \r\nhuman two\r\nbot two",
    );
    let chats = mdx::load_handcrafted_chats(&path).unwrap();
    assert_eq!(chats.blocks.len(), 2);
    assert_eq!(chats.blocks[0].memories.len(), 1);
    assert_eq!(chats.blocks[1].memories[0].human, "human two");

    let path = fixture.write("quotes.csv", "quote,author,category\n\"A wise quote,\nwith another line\",Someone,wisdom\nshort,Someone,wisdom\nAnother useful quote here,Someone,life\n");
    let quotes = mdx::load_quotes_csv(&path).unwrap();
    assert_eq!(
        quotes,
        [
            "A wise quote,\nwith another line",
            "Another useful quote here"
        ]
    );
    let chats = mdx::load_quote_chats(&path).unwrap();
    assert_eq!(chats.blocks.len(), 2);
    assert_eq!(chats.blocks[0].memories[0].bot, quotes[0]);

    let malformed = fixture.write("bad.jsonl", "not json\n");
    assert!(mdx::load_arena_chats(&malformed).is_err());
    let malformed = fixture.write("bad.csv", "a,b,c,d,verse\n1,2\n");
    assert!(mdx::load_csv_bible_pairs(&malformed).is_err());
}

#[test]
fn source_caps_and_deduplication_are_independent_of_thread_count() {
    let fixture = Fixture::new();
    let first = fixture.write(
        "first.txt",
        (0..150)
            .map(|i| format!("First useful sentence number {i}\n"))
            .collect::<String>(),
    );
    let second = fixture.write(
        "second.txt",
        (0..150)
            .map(|i| format!("Second useful sentence number {i}\n"))
            .collect::<String>(),
    );
    let load = |threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                DataLoader::new(TrainingStage::Language)
                    .add(&first, FileKind::TxtLines, Some(50))
                    .add(&second, FileKind::TxtLines, Some(50))
                    .seed(123)
                    .total_limit(70)
                    .load_sentences()
                    .unwrap()
            })
    };
    let serial = load(1);
    assert_eq!(serial.len(), 70);
    assert_eq!(serial, load(4));

    let first = fixture.write("duplicate-one.txt", "Original sentence\nUnique sentence\n");
    let second = fixture.write(
        "duplicate-two.txt",
        " original SENTENCE \nAnother sentence\n",
    );
    let sentences = DataLoader::new(TrainingStage::Language)
        .add(first, FileKind::TxtLines, None)
        .add(second, FileKind::TxtLines, None)
        .load_sentences()
        .unwrap();
    assert_eq!(sentences.len(), 3);
    assert!(sentences.contains(&"Original sentence".to_string()));
}

#[test]
fn cifar_parallel_conversion_preserves_labels_and_normalization() {
    let fixture = Fixture::new();
    let mut raw = Vec::new();
    for label in 0..4u8 {
        raw.extend([label + 10, label]);
        raw.extend(std::iter::repeat_n(label * 50, 3072));
    }
    raw.extend([1, 2, 3]); // Legacy behavior ignores an incomplete final record.
    let path = fixture.write("cifar.bin", raw);
    let data = crate::vision::cifar::CifarDataset::load(&path).unwrap();
    assert_eq!(data.records.len(), 4);
    for (i, record) in data.records.iter().enumerate() {
        assert_eq!(record.fine_label, i as u8);
        assert_eq!(record.coarse_label, i as u8 + 10);
        for (channel, (mean, std)) in [(0.5071, 0.2675), (0.4867, 0.2565), (0.4408, 0.2761)]
            .into_iter()
            .enumerate()
        {
            let expected = (i as f32 * 50.0 / 255.0 - mean) / std;
            assert!(
                record.pixels[channel * 1024..(channel + 1) * 1024]
                    .iter()
                    .all(|v| (*v - expected).abs() < 1e-6)
            );
        }
    }
}

#[test]
fn sample_loading_and_reports_agree_on_source_priority() {
    let fixture = Fixture::new();
    let content = "Tell me about the moon\nThe moon orbits the Earth.\n\nTell me about the sun\nThe sun is a bright star.\n";
    let first = fixture.write("first-chat.txt", content);
    let second = fixture.write("second-chat.txt", content);
    let tokenizer =
        super::bpe::TokenizerKind::Char(super::Tokenizer::build_from_text(content, 256));
    let keywords = std::collections::HashMap::new();
    let loader = || {
        DataLoader::new(TrainingStage::Language)
            .add(&first, FileKind::Chats, Some(2))
            .add(&second, FileKind::Chats, Some(2))
            .seed(17)
    };
    let reports = loader().report(&tokenizer, &keywords, 256).unwrap();
    assert_eq!(reports[0].samples, 2);
    assert_eq!(reports[1].samples, 0);
    assert_eq!(reports[1].duplicates, 2);
    let mut expected = None;
    for threads in [1, 4] {
        let samples = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| loader().load(&tokenizer, &keywords, 256).unwrap());
        assert_eq!(
            samples.len(),
            reports.iter().map(|r| r.samples).sum::<usize>()
        );
        let tokens: Vec<_> = samples
            .into_iter()
            .map(|s| (s.input_ids, s.target_labels))
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&tokens, expected);
        }
        expected = Some(tokens);
    }
}

#[test]
fn wiki_article_caps_flush_the_final_parallel_batch() {
    let fixture = Fixture::new();
    let xml = "<mediawiki><page><revision><text>The moon is a natural satellite of the planet Earth.</text></revision></page><page><revision><text>The sun is the star at the center of our solar system.</text></revision></page></mediawiki>";
    let path = fixture.write("wiki.xml", xml);
    let capped = super::wiki::load_wiki_sentences(&path, 1, 1).unwrap();
    let all = super::wiki::load_wiki_sentences(&path, 0, 1).unwrap();
    assert!(!capped.is_empty());
    assert!(all.len() > capped.len());
    assert_eq!(&all[..capped.len()], capped.as_slice());
}

#[test]
fn fer_loading_keeps_class_labels_and_skips_corrupt_images() {
    let fixture = Fixture::new();
    for (class, color) in [("angry", [255, 0, 0]), ("happy", [0, 255, 0])] {
        let dir = fixture.0.join(class);
        std::fs::create_dir(&dir).unwrap();
        image::RgbImage::from_pixel(32, 32, image::Rgb(color))
            .save(dir.join("valid.png"))
            .unwrap();
        std::fs::write(dir.join("corrupt.jpg"), b"invalid image").unwrap();
    }
    let dataset = crate::vision::fer::FerDataset::load(fixture.0.to_str().unwrap()).unwrap();
    assert_eq!(dataset.records.len(), 2);
    assert_eq!(dataset.records[0].emote_idx, 0);
    assert_eq!(dataset.records[1].emote_idx, 3);
    assert!(dataset.records[0].pixels[..1024].iter().all(|&v| v == 1.0));
    assert!(dataset.records[0].pixels[1024..].iter().all(|&v| v == -1.0));
}
