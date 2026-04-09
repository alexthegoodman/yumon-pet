use std::collections::HashMap;

use rand::{seq::SliceRandom, SeedableRng};
use rand::rngs::StdRng;

use crate::brain::bpe::TokenizerKind;

#[cfg(target_os = "windows")]
use crate::brain::mdx::{load_arena_chats, load_csv_bible, load_csv_words, load_dictionary_sentences, load_handcrafted_chats, load_handcrafted_sentences, load_mdx_sentences, load_qa_pairs, load_specific_dict_sentences, load_txt_sentences};

#[cfg(target_os = "windows")]
use crate::brain::pdf::load_pdfs;
use crate::brain::samples::{Sample, TrainingStage, prepare_paired_samples_chats, prepare_paired_samples_split, prepare_paired_samples_split_sep};

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
    Chats,
    JsonChats,
    Txt,
    SpecificDict,
    PDF
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

    pub fn load_sentences(self) -> anyhow::Result<Vec<String>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut all: Vec<String> = Vec::new();

        for entry in &self.entries {
            // 1. Raw sentences from disk

            // // Per-file limit before sample prep to reduce load
            // if let Some(n) = entry.limit {
            //     sentences.shuffle(&mut rng);
            //     sentences.truncate(n);
            // }

            let mut sentences = match entry.kind {
                FileKind::QaPairs => {
                    let mut pairs = load_qa_pairs_raw(&entry.path)?;
                    
                    // Per-file limit before sample prep to reduce load
                    if let Some(n) = entry.limit {
                        pairs.shuffle(&mut rng);
                        pairs.truncate(n);
                    }

                    let mut sents = Vec::new();
                    for pair in pairs {
                        sents.push(pair.0);
                        sents.push(pair.1);
                    }

                    sents
                }
                FileKind::Chats => {
                    let mut chats = load_handcrafted_chats(&entry.path)?;

                    // Per-file limit before sample prep to reduce load
                    if let Some(n) = entry.limit {
                        chats.blocks.shuffle(&mut rng);
                        chats.blocks.truncate(n);
                    }

                    let mut sents = Vec::new();
                    for chat in chats.blocks {
                        for mem in chat.memories {
                            sents.push(mem.bot);
                            sents.push(mem.human);
                        }
                    }

                    sents
                }
                FileKind::JsonChats => {
                    let mut chats = load_arena_chats(&entry.path)?;

                    // Per-file limit before sample prep to reduce load
                    if let Some(n) = entry.limit {
                        chats.blocks.shuffle(&mut rng);
                        chats.blocks.truncate(n);
                    }

                    let mut sents = Vec::new();
                    for chat in chats.blocks {
                        for mem in chat.memories {
                            sents.push(mem.bot);
                            sents.push(mem.human);
                        }
                    }

                    sents
                },
                _ => {
                    load_sentences(&entry.path, &entry.kind)?
                }
            };

            all.extend(sentences);
        }
        
        // 4. Global shuffle then total cap
        all.shuffle(&mut rng);
        if let Some(n) = self.total_limit {
            all.truncate(n);
        }

        println!("[DataLoader] total sentences returned: {}", all.len());
        Ok(all)
    }
 
    /// Load, prepare, merge, shuffle, and cap all samples.
    pub fn load(
        self,
        tokenizer:     &TokenizerKind,
        keyword_index: &HashMap<std::string::String, Vec<usize>>,
        max_seq_len:   usize,
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

            // Per-file limit before sample prep to reduce load
            if let Some(n) = entry.limit {
                sentences.shuffle(&mut rng);
                sentences.truncate(n);
            }

            // 2. Prepare training samples
            #[cfg(target_os = "windows")]
            let mut samples = match entry.kind {
                FileKind::QaPairs => {
                    let mut pairs = load_qa_pairs_raw(&entry.path)?;

                    // Per-file limit before sample prep to reduce load
                    if let Some(n) = entry.limit {
                        pairs.shuffle(&mut rng);
                        pairs.truncate(n);
                    }

                    prepare_paired_samples_split_sep(
                        pairs, tokenizer, keyword_index, &mut rng, self.stage, max_seq_len,
                    )
                }
                FileKind::Chats => {
                    let mut chats = load_handcrafted_chats(&entry.path)?;

                    // Per-file limit before sample prep to reduce load
                    if let Some(n) = entry.limit {
                        chats.blocks.shuffle(&mut rng);
                        chats.blocks.truncate(n);
                    }

                    prepare_paired_samples_chats(
                        chats, tokenizer, keyword_index, &mut rng, self.stage, max_seq_len,
                    )
                }
                FileKind::JsonChats => {
                    let mut chats = load_arena_chats(&entry.path)?;

                    // Per-file limit before sample prep to reduce load
                    if let Some(n) = entry.limit {
                        chats.blocks.shuffle(&mut rng);
                        chats.blocks.truncate(n);
                    }

                    prepare_paired_samples_chats(
                        chats, tokenizer, keyword_index, &mut rng, self.stage, max_seq_len,
                    )
                },
                _ => {
                    prepare_paired_samples_split(
                        sentences, tokenizer, keyword_index, &mut rng, self.stage, max_seq_len,
                    )
                }
            };

            // no need to train in wasm
            #[cfg(target_arch = "wasm32")]
            let mut samples = match entry.kind {
                _ => {
                    prepare_paired_samples_split(
                        sentences, tokenizer, keyword_index, &mut rng, self.stage, max_seq_len,
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
#[cfg(target_os = "windows")]
fn load_sentences(path: &str, kind: &FileKind) -> anyhow::Result<Vec<String>> {
    match kind {
        FileKind::Mdx         => load_mdx_sentences(path),
        FileKind::BibleCsv    => load_csv_bible(path),
        FileKind::Handcrafted => load_handcrafted_sentences(path),
        FileKind::Txt         => load_txt_sentences(path),
        FileKind::PDF       => {
            let paths: Vec<&str> = path.split(", ").collect();
            Ok(load_pdfs(paths))
        },
        FileKind::SpecificDict => {
            // let all_words = load_csv_words("archive/word_counts.csv");
            // let all_words =  all_words.as_ref().expect("Couldn't get words");
            
            // let dict_sentences = load_specific_dict_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt", all_words);
            let dict_sentences = load_dictionary_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt");

            dict_sentences
        },
        // QA pairs are handled separately — return empty here
        FileKind::QaPairs     => Ok(Vec::new()),
        FileKind::Chats       => Ok(Vec::new()),
        FileKind::JsonChats       => Ok(Vec::new())
    }
}

#[cfg(target_arch = "wasm32")]
fn load_sentences(path: &str, kind: &FileKind) -> anyhow::Result<Vec<String>> {
    match kind {
        FileKind::Mdx         => Ok(Vec::new()),
        FileKind::BibleCsv    => Ok(Vec::new()),
        FileKind::Handcrafted => Ok(Vec::new()),
        FileKind::Txt         => Ok(Vec::new()),
        FileKind::SpecificDict => Ok(Vec::new()),
        // QA pairs are handled separately — return empty here
        FileKind::QaPairs     => Ok(Vec::new()),
        FileKind::Chats       => Ok(Vec::new()),
        FileKind::JsonChats       => Ok(Vec::new())
    }
}

/// Thin wrapper so the QA path stays unified in `load()`.
#[cfg(target_os = "windows")]
fn load_qa_pairs_raw(path: &str) -> anyhow::Result<Vec<(String, String)>> {
    load_qa_pairs(path)
}