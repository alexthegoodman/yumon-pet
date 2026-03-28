use std::collections::HashMap;

use rand::{seq::SliceRandom, SeedableRng};
use rand::rngs::StdRng;

use crate::brain::bpe::TokenizerKind;
use crate::brain::mdx::{load_csv_bible, load_handcrafted_sentences, load_mdx_sentences, load_qa_pairs};
use crate::brain::samples::{Sample, TrainingStage, prepare_paired_samples_split, prepare_paired_samples_split_sep};

// ── File-source descriptor ────────────────────────────────────────────────────

/// How a file should be loaded and how many samples to keep from it.
pub struct FileEntry {
    pub path: String,
    pub kind: FileKind,
    /// Cap on samples drawn from this file. `None` = no limit.
    pub limit: Option<usize>,
}

/// Which loader function to dispatch to.
pub enum FileKind {
    Mdx,
    BibleCsv,
    Handcrafted,
    QaPairs,
    // extend with WikiXml, Txt, Pdf, … as needed
}

impl FileEntry {
    pub fn new(path: impl Into<String>, kind: FileKind, limit: impl Into<Option<usize>>) -> Self {
        Self { path: path.into(), kind, limit: limit.into() }
    }
}

// ── DataLoader ────────────────────────────────────────────────────────────────

pub struct DataLoader {
    entries:      Vec<FileEntry>,
    total_limit:  Option<usize>,
    seed:         u64,
    stage:        TrainingStage,
}

impl DataLoader {
    pub fn new(stage: TrainingStage) -> Self {
        Self {
            entries:     Vec::new(),
            total_limit: None,
            seed:        4815162342,
            stage,
        }
    }

    /// Add a file source with an optional per-file sample cap.
    pub fn add(mut self, path: impl Into<String>, kind: FileKind, limit: impl Into<Option<usize>>) -> Self {
        self.entries.push(FileEntry::new(path, kind, limit));
        self
    }

    /// Global cap applied after all files are merged and shuffled.
    pub fn total_limit(mut self, n: usize) -> Self {
        self.total_limit = Some(n);
        self
    }

    /// Seed for the RNG (default: 4815162342 for reproducibility).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Load, prepare, merge, shuffle, and cap all samples.
    pub fn load(
        self,
        tokenizer:     &TokenizerKind,
        keyword_index: &HashMap<std::string::String, Vec<usize>>,
    ) -> anyhow::Result<Vec<Sample>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut all: Vec<Sample> = Vec::new();

        for entry in &self.entries {
            // 1. Raw sentences from disk
            let mut sentences = load_sentences(&entry.path, &entry.kind)?;
            println!(
                "[DataLoader] {:?}: {} sentences loaded",
                entry.path,
                sentences.len()
            );

            // 2. Prepare training samples
            let mut samples = match entry.kind {
                FileKind::QaPairs => {
                    let pairs = load_qa_pairs_raw(&entry.path)?;
                    prepare_paired_samples_split_sep(
                        pairs, tokenizer, keyword_index, &mut rng, self.stage,
                    )
                }
                _ => {
                    prepare_paired_samples_split(
                        sentences, tokenizer, keyword_index, &mut rng, self.stage,
                    )
                }
            };

            // 3. Per-file limit (shuffle first so the truncation is random)
            if let Some(n) = entry.limit {
                samples.shuffle(&mut rng);
                samples.truncate(n);
            }

            println!(
                "[DataLoader] {:?}: {} samples after per-file limit",
                entry.path,
                samples.len()
            );

            all.extend(samples);
        }

        // 4. Global shuffle then total cap
        all.shuffle(&mut rng);
        if let Some(n) = self.total_limit {
            all.truncate(n);
        }

        println!("[DataLoader] total samples returned: {}", all.len());
        Ok(all)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Dispatch to the appropriate sentence-loader based on FileKind.
fn load_sentences(path: &str, kind: &FileKind) -> anyhow::Result<Vec<String>> {
    match kind {
        FileKind::Mdx         => load_mdx_sentences(path),
        FileKind::BibleCsv    => load_csv_bible(path),
        FileKind::Handcrafted => load_handcrafted_sentences(path),
        // QA pairs are handled separately — return empty here
        FileKind::QaPairs     => Ok(Vec::new()),
    }
}

/// Thin wrapper so the QA path stays unified in `load()`.
fn load_qa_pairs_raw(path: &str) -> anyhow::Result<Vec<(String, String)>> {
    load_qa_pairs(path)
}