use std::collections::HashMap;

use rand::{seq::SliceRandom, SeedableRng};
use rand::rngs::StdRng;
use rand::Rng;
use crate::brain::loading::map_ordered;

use crate::brain::bpe::TokenizerKind;

use crate::brain::chats::load_distilled_chats;
#[cfg(not(target_arch = "wasm32"))]
use crate::brain::am_distill::load_am_distill_chats;
use crate::brain::mdx::{load_chats_from_csv, load_chats_from_friends_csv};
#[cfg(not(target_arch = "wasm32"))]
use crate::brain::mdx::{load_arena_chats, load_csv_bible_pairs, load_csv_words, load_dictionary_sentences, load_handcrafted_chats, load_handcrafted_sentences, load_mdx_sentences, load_qa_pairs, load_quote_chats, load_quotes_csv, load_specific_dict_sentences, load_text_paragraphs, load_txt_lines, load_wiki_chats, load_txt_sentences};

#[cfg(not(target_arch = "wasm32"))]
use crate::brain::pdf::load_pdfs;
use crate::brain::samples::{Sample, TrainingStage, prepare_paired_samples_chats, prepare_paired_samples_split, prepare_paired_samples_split_sep};
use crate::brain::PAD_TOKEN;

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
    TxtLines,
    SpecificDict,
    PDF,
    DistillChat,
    /// AM-DeepSeek-R1-Distilled `.jsonl` / `.jsonl.zst`, streamed. The per-file
    /// limit stops reading early (the full file is ~20 GB decoded).
    AmDistill,
    DialogueCsv,
    FriendsCsv,
    /// One paragraph per line (wiki_extract.txt). Training: sentences become
    /// Human/Yumon turns; article openings start with a title question.
    /// Tokenizer corpus: whole paragraphs.
    Paragraphs,
    /// `quote,author,category` CSV. Training: a category request answered by
    /// the quote. Tokenizer corpus: quote text.
    QuotesCsv,
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

        let batches = map_ordered(
            self.entries
                .iter()
                .zip(source_seeds(self.seed, self.entries.len()))
                .collect(),
            |(entry, seed)| -> anyhow::Result<Vec<String>> {
                let mut rng = StdRng::seed_from_u64(seed);
                // 1. Raw sentences from disk

                // // Per-file limit before sample prep to reduce load
                // if let Some(n) = entry.limit {
                //     sentences.shuffle(&mut rng);
                //     sentences.truncate(n);
                // }

                let sentences = match entry.kind {
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
                    FileKind::BibleCsv => {
                        let mut pairs = load_csv_bible_pairs(&entry.path)?;

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
                    FileKind::DialogueCsv => {
                        let mut chats = load_chats_from_csv(&entry.path)?;

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
                    FileKind::FriendsCsv => {
                        let mut chats = load_chats_from_friends_csv(&entry.path)?;

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
                    FileKind::DistillChat => {
                        let mut chats = load_distilled_chats(&entry.path, i32::MAX)?;

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
                    FileKind::AmDistill => {
                        // Reading already stops at the limit.
                        let chats = load_am_distill_chats(&entry.path, entry.limit)?;

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
                    }
                    _ => {
                        let mut pairs = load_sentences(&entry.path, &entry.kind)?;

                        // Per-file limit before sample prep to reduce load
                        if let Some(n) = entry.limit {
                            pairs.shuffle(&mut rng);
                            pairs.truncate(n);
                        }

                        pairs
                    }
                };

                Ok(sentences)
            },
        );
        for sentences in batches {
            all.extend(sentences?);
        }

        // Exact duplicate sentences would skew merge counts (ideas.txt repeats
        // each line ~18 times).
        let before = all.len();
        let mut seen = std::collections::HashSet::new();
        all.retain(|sentence| seen.insert(sentence.trim().to_lowercase()));
        println!(
            "[DataLoader] {} duplicate sentences removed",
            before - all.len()
        );

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
        tokenizer: &TokenizerKind,
        keyword_index: &HashMap<std::string::String, Vec<usize>>,
        max_seq_len: usize,
    ) -> anyhow::Result<Vec<Sample>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut all: Vec<Sample> = Vec::new();
        // Exact duplicates (same prompt and target tokens) across every source.
        let mut seen = std::collections::HashSet::new();

        // Indexed collection preserves source priority for deduplication. Each source
        // gets its own RNG, so scheduling does not affect per-file selection.
        let batches = map_ordered(
            self.entries
                .iter()
                .zip(source_seeds(self.seed, self.entries.len()))
                .collect(),
            |(entry, seed)| {
                self.load_entry(
                    entry,
                    tokenizer,
                    keyword_index,
                    max_seq_len,
                    &mut StdRng::seed_from_u64(seed),
                )
            },
        );
        for (entry, samples) in self.entries.iter().zip(batches) {
            let mut samples = samples?;
            let before = samples.len();
            samples.retain(|sample| seen.insert(sample_key(sample)));
            println!(
                "[DataLoader] {:?}: {} samples after per-file limit, {} duplicates removed",
                entry.path,
                samples.len(),
                before - samples.len()
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

    /// Same pipeline and dedupe as `load`, one source at a time, keeping only
    /// counts so the whole data set never has to fit in memory at once.
    /// Ignores the total cap.
    pub fn report(
        self,
        tokenizer: &TokenizerKind,
        keyword_index: &HashMap<std::string::String, Vec<usize>>,
        max_seq_len: usize,
    ) -> anyhow::Result<Vec<SourceReport>> {
        let mut seen = std::collections::HashSet::new();
        let mut reports = Vec::new();

        for (entry, seed) in self
            .entries
            .iter()
            .zip(source_seeds(self.seed, self.entries.len()))
        {
            let samples = self.load_entry(
                entry,
                tokenizer,
                keyword_index,
                max_seq_len,
                &mut StdRng::seed_from_u64(seed),
            )?;
            let mut report = SourceReport {
                path: entry.path.clone(),
                ..Default::default()
            };
            for sample in &samples {
                if !seen.insert(sample_key(sample)) {
                    report.duplicates += 1;
                    continue;
                }
                let active = |ids: &[usize]| ids.iter().take_while(|&&t| t != PAD_TOKEN).count();
                report.samples += 1;
                report.prompt_tokens += active(&sample.input_ids);
                report.target_tokens += active(&sample.target_labels);
            }
            reports.push(report);
        }

        Ok(reports)
    }

    /// One source: read, apply its per-file cap, prepare samples.
    fn load_entry(
        &self,
        entry:         &FileEntry,
        tokenizer:     &TokenizerKind,
        keyword_index: &HashMap<std::string::String, Vec<usize>>,
        max_seq_len:   usize,
        rng:           &mut StdRng,
    ) -> anyhow::Result<Vec<Sample>> {
        // 1. Raw sentences from disk
        // Chat-built sources read their own files below.
        let mut sentences = match entry.kind {
            FileKind::Paragraphs | FileKind::QuotesCsv | FileKind::AmDistill => Vec::new(),
            _ => load_sentences(&entry.path, &entry.kind)?,
        };
        println!(
            "[DataLoader] {:?}: {} sentences loaded",
            entry.path,
            sentences.len()
        );

        // Per-file limit before sample prep to reduce load
        if let Some(n) = entry.limit {
            sentences.shuffle(rng);
            sentences.truncate(n);
        }

        // 2. Prepare training samples
        #[cfg(not(target_arch = "wasm32"))]
        let mut samples = match entry.kind {
            FileKind::QaPairs => {
                let mut pairs = load_qa_pairs_raw(&entry.path)?;

                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    pairs.shuffle(rng);
                    pairs.truncate(n);
                }

                prepare_paired_samples_split_sep(
                    pairs, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::BibleCsv => {
                let mut pairs = load_csv_bible_pairs(&entry.path)?;

                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    pairs.shuffle(rng);
                    pairs.truncate(n);
                }

                prepare_paired_samples_split_sep(
                    pairs, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::Chats => {
                let mut chats = load_handcrafted_chats(&entry.path)?;

                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    chats.blocks.shuffle(rng);
                    chats.blocks.truncate(n);
                }

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::DistillChat => {
                let mut chats = load_distilled_chats(&entry.path, i32::MAX)?;

                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    chats.blocks.shuffle(rng);
                    chats.blocks.truncate(n);
                }

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::DialogueCsv => {
                let mut chats = load_chats_from_csv(&entry.path)?;
                
                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    chats.blocks.shuffle(rng);
                    chats.blocks.truncate(n);
                }

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::FriendsCsv => {
                use crate::brain::mdx::load_chats_from_friends_csv;

                let mut chats = load_chats_from_friends_csv(&entry.path)?;
                
                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    chats.blocks.shuffle(rng);
                    chats.blocks.truncate(n);
                }

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::AmDistill => {
                // Reading already stops at the limit.
                let chats = load_am_distill_chats(&entry.path, entry.limit)?;

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            FileKind::JsonChats => {
                let mut chats = load_arena_chats(&entry.path)?;

                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    chats.blocks.shuffle(rng);
                    chats.blocks.truncate(n);
                }

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            },
            FileKind::Paragraphs | FileKind::QuotesCsv => {
                let mut chats = match entry.kind {
                    FileKind::Paragraphs => load_wiki_chats(&entry.path)?,
                    _ => load_quote_chats(&entry.path)?,
                };

                // Per-file limit before sample prep to reduce load
                if let Some(n) = entry.limit {
                    chats.blocks.shuffle(rng);
                    chats.blocks.truncate(n);
                }

                prepare_paired_samples_chats(
                    chats, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
            _ => {
                prepare_paired_samples_split(
                    sentences, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
        };

        // no need to train in wasm
        #[cfg(target_arch = "wasm32")]
        let mut samples = match entry.kind {
            _ => {
                prepare_paired_samples_split(
                    sentences, tokenizer, keyword_index, rng, self.stage, max_seq_len,
                )
            }
        };

        // 3. Per-file limit (shuffle first so the truncation is random)
        if let Some(n) = entry.limit {
            samples.shuffle(rng);
            samples.truncate(n);
        }

        Ok(samples)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Dispatch to the appropriate sentence-loader based on FileKind.
#[cfg(not(target_arch = "wasm32"))]
fn load_sentences(path: &str, kind: &FileKind) -> anyhow::Result<Vec<String>> {
    match kind {
        FileKind::Mdx         => load_mdx_sentences(path),
        // Bible verse pairs are handled separately — return empty here
        FileKind::BibleCsv    => Ok(Vec::new()),
        FileKind::Handcrafted => load_handcrafted_sentences(path),
        FileKind::Txt         => load_txt_sentences(path),
        FileKind::TxtLines    => load_txt_lines(path),
        FileKind::Paragraphs  => load_text_paragraphs(path),
        FileKind::QuotesCsv   => load_quotes_csv(path),
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
        FileKind::JsonChats       => Ok(Vec::new()),
        FileKind::DistillChat   => Ok(Vec::new()),
        FileKind::AmDistill     => Ok(Vec::new()),
        FileKind::DialogueCsv   => Ok(Vec::new()),
        FileKind::FriendsCsv    => Ok(Vec::new()),
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
        FileKind::JsonChats       => Ok(Vec::new()),
        FileKind::DistillChat   => Ok(Vec::new())
    }
}

/// Per-source totals from `DataLoader::report`, after dedupe.
#[derive(Debug, Default)]
pub struct SourceReport {
    pub path:          String,
    pub samples:       usize,
    pub duplicates:    usize,
    pub prompt_tokens: usize,
    pub target_tokens: usize,
}

/// Hash of a sample's non-padding prompt and target tokens.
fn sample_key(sample: &Sample) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let tokens = |ids: &[usize]| ids.iter().copied().take_while(|&t| t != PAD_TOKEN).collect::<Vec<_>>();
    tokens(&sample.input_ids).hash(&mut hasher);
    tokens(&sample.target_labels).hash(&mut hasher);
    hasher.finish()
}

/// Thin wrapper so the QA path stays unified in `load()`.
#[cfg(not(target_arch = "wasm32"))]
fn load_qa_pairs_raw(path: &str) -> anyhow::Result<Vec<(String, String)>> {
    load_qa_pairs(path)
}
/// Stable source RNGs shared by loading and memory-bounded reporting.
fn source_seeds(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.r#gen()).collect()
}
