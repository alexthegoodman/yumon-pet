/// BPE (Byte-Pair Encoding) tokenizer for Yumon's language brain.
///
/// Trained on the same SimpleWiki corpus as the char tokenizer,
/// but encodes at the subword level — balancing vocabulary size
/// against sequence length.
///
/// Special tokens mirror the char tokenizer for compatibility:
///   <PAD> = 0
///   <BOS> = 1
///   <EOS> = 2
///   <UNK> = 3
///
/// Usage:
///   let tok = BpeTokenizer::train(&sentences, 4096)?;
///   tok.save("yumon_bpe")?;
///   // later:
///   let tok = BpeTokenizer::load("yumon_bpe")?;
///   let ids  = tok.encode("hello world")?;
///   let text = tok.decode(&ids)?;

use anyhow::{Result, Context};
use tokenizers::{
    AddedToken, Tokenizer, decoders::{bpe::BPEDecoder, byte_level::ByteLevel}, models::{TrainerWrapper, bpe::{BPE, BpeTrainerBuilder}}, normalizers::{Lowercase, NFC, Sequence as NormSequence}, pre_tokenizers::whitespace::Whitespace, processors::template::TemplateProcessing
};
use std::io::Write;
use std::path::Path;
// use quick_error::ResultExt;

// Special token strings — match your existing conventions
pub const PAD_STR: &str = "<PAD>";
pub const BOS_STR: &str = "<BOS>";
pub const EOS_STR: &str = "<EOS>";
pub const UNK_STR: &str = "<UNK>";

pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;

pub struct BpeTokenizer {
    inner:      Tokenizer,
    pub vocab_size: usize,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer from an in-memory slice of sentences.
    ///
    /// `vocab_size` — total vocabulary including special tokens.
    ///   Recommended: 2048–8192 for a small LSTM.
    ///   Start with 4096 and see how sequence lengths feel.
    pub fn train(
        // sentences: &[String], 
        sentences: Vec<&String>,
        vocab_size: usize
    ) -> Result<Self> {
        println!("🔧 Training BPE tokenizer on {} sentences …", sentences.len());
        println!("   Target vocab size: {vocab_size}");

        // ── Write corpus to a temp file (tokenizers trainer needs file paths) ──
        let tmp_path = "tmp/yumon_bpe_corpus.txt";
        {
            let mut f = std::fs::File::create(tmp_path)
                .context("creating temp corpus file")?;
            for s in sentences {
                writeln!(f, "{}", s)?;
            }
        }
        println!("   Corpus written to {tmp_path}");

        // ── Build trainer ──────────────────────────────────────────────────────
        let special_tokens = vec![
            AddedToken::from(PAD_STR, true),
            AddedToken::from(BOS_STR, true),
            AddedToken::from(EOS_STR, true),
            AddedToken::from(UNK_STR, true),
        ];

        let trainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(100)           // ignore hapax legomena
            .special_tokens(special_tokens.clone())
            // .continuing_subword_prefix("##".to_string()) // WordPiece-style visible joins (not needed with ByteLevel)
            .show_progress(true)
            .build();

        // ── Assemble tokenizer with pre/post processing ────────────────────────
        let mut tokenizer = Tokenizer::new(BPE::default());

        // Normalise: NFC unicode then lowercase
        // (SimpleWiki is mixed-case; lowercasing shrinks vocab nicely)
        tokenizer.with_normalizer(NormSequence::new(vec![
            NFC.into(),
            Lowercase.into(),
        ]));

        // Split on whitespace before BPE merges
        // tokenizer.with_pre_tokenizer(Whitespace::default());
        tokenizer.with_pre_tokenizer(ByteLevel::default());

        // BPE decoder — reconstructs spaces correctly
        // tokenizer.with_decoder(BPEDecoder::default());
        tokenizer.with_decoder(ByteLevel::default());

        tokenizer.with_post_processor(
            anyhow::Context::context(
                TemplateProcessing::builder()
                    .try_single(format!("{BOS_STR} $A {EOS_STR}"))
                    .map_err(|e| anyhow::anyhow!(e))?
                    .try_pair(format!("{BOS_STR} $A {EOS_STR} $B:1 {EOS_STR}:1"))
                    .map_err(|e| anyhow::anyhow!(e))?
                    .special_tokens(vec![
                        (BOS_STR, BOS_ID),
                        (EOS_STR, EOS_ID),
                    ])
                    .build(),
                "building TemplateProcessing"
            )?,
        );

        // ── Train ──────────────────────────────────────────────────────────────
        let mut trainer_wrapper: TrainerWrapper = trainer.clone().into();
        tokenizer
            .train_from_files(&mut trainer_wrapper, vec![tmp_path.to_string()])
            .map_err(|e| anyhow::anyhow!("BPE training failed: {}", e))?;

        // Verify special token IDs landed where we expect
        Self::assert_special_token_ids(&tokenizer)?;

        let vocab_size = tokenizer.get_vocab_size(true);
        println!("✅ BPE tokenizer trained — actual vocab size: {vocab_size}");

        Ok(Self { inner: tokenizer, vocab_size })
    }

    /// Encode a string → token ID sequence (includes BOS/EOS).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false)
             .map_err(|e| anyhow::anyhow!("encode tokenizer failed: {}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Encode without BOS/EOS — useful for prompt continuation.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        // Temporarily bypass post-processor by encoding as a pair with empty B
        // Simpler: just strip BOS/EOS from the output
        let mut ids = self.encode(text)?;
        ids.retain(|&id| id != BOS_ID && id != EOS_ID);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids.iter()
            .copied()
            .filter(|&id| id != PAD_ID && id != BOS_ID && id != EOS_ID)
            .collect();
        self.inner.decode(&filtered, true)
            .map_err(|e| anyhow::anyhow!(e))  // ← convert tokenizers error to anyhow
    }

    /// Save tokenizer to `dir/` — creates two files:
    pub fn save(&self, dir: &str) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = format!("{dir}/tokenizer.json");
        self.inner.save(&path, false)
            .map_err(|e| anyhow::anyhow!("saving tokenizer: {}", e))?;
        // println!("💾 BPE tokenizer saved to {path}");
        Ok(())
    }

    /// Load from a directory saved by `save()`.
    pub fn load(dir: &str) -> Result<Self> {
        let path = format!("{dir}/tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer from {path}: {}", e))?;
        let vocab_size = inner.get_vocab_size(true);
        println!("📂 BPE tokenizer loaded — vocab size: {vocab_size}");
        Ok(Self { inner, vocab_size })
    }

    /// Quick sanity check: print a round-trip example.
    pub fn demo(&self, text: &str) -> Result<()> {
        let ids   = self.encode(text)?;
        let back  = self.decode(&ids)?;
        println!("  input : {text}");
        println!("  tokens: {:?}", ids.len());
        println!("  output: {back}");
        Ok(())
    }

    // ── Internal helpers ───────────────────────────────────────────────────────

    fn assert_special_token_ids(tok: &Tokenizer) -> Result<()> {
        let vocab = tok.get_vocab(true);
        for (name, expected_id) in [
            (PAD_STR, PAD_ID),
            (BOS_STR, BOS_ID),
            (EOS_STR, EOS_ID),
            (UNK_STR, UNK_ID),
        ] {
            match vocab.get(name) {
                Some(&id) if id == expected_id => {}
                Some(&id) => eprintln!(
                    "⚠️  Special token {name} has id {id}, expected {expected_id}"
                ),
                None => eprintln!("⚠️  Special token {name} not found in vocab"),
            }
        }
        Ok(())
    }
}

// ── TokenizerKind enum — drop-in selector ─────────────────────────────────────
//
// Use this in your training config to switch between char and BPE:
//
//   [tokenizer]
//   kind = "bpe"          # or "char"
//   bpe_vocab_size = 4096
//   bpe_dir = "yumon_bpe"

/// Unified encode/decode interface over both tokenizer types.
/// Add this to your existing tokenizer.rs or a new unified_tokenizer.rs.
pub enum TokenizerKind {
    Char(crate::brain::Tokenizer),   // your existing char tokenizer
    Bpe(BpeTokenizer),
}

impl TokenizerKind {
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Char(t) => t.vocab_size,
            Self::Bpe(t)  => t.vocab_size,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Self::Char(t) => t.encode(text),
            Self::Bpe(t)  => t.encode_raw(text)
                .unwrap_or_default()
                .into_iter()
                .map(|x| x as usize)
                .collect(),
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        match self {
            Self::Char(t) => t.decode(ids),
            Self::Bpe(t)  => {
                let u32_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
                t.decode(&u32_ids).unwrap_or_default()
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        match self {
            Self::Char(t) => t.save(path),
            Self::Bpe(t)  => {
                t.save(path)
            }
        }
    }
}