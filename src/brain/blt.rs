//! Byte Latent Transformer (Pagnoni et al. 2024, "Byte Latent Transformer: Patches
//! Scale Better Than Tokens", after facebookresearch/blt). See docs/BLT.md.
//!
//! Bytes are grouped into patches. A small local encoder turns each patch into one
//! vector, a global transformer runs causally over patches, and a small local decoder
//! predicts the next byte from the encoder's byte states plus the global output of the
//! last patch that ends at or before it.
//!
//! - Embeddings: a byte table plus hashed n-gram tables (`hash_ngrams`, `hash_buckets`
//!   rows each), summed and scaled by 1/sqrt(tables).
//! - Encoder: `encoder_layers` causal blocks over bytes, then one cross-attention where
//!   each patch (query = mean of its bytes) attends to its own bytes.
//! - Global: `n_layers` causal blocks of width `embed_dim` over patches.
//! - Decoder: per layer, each byte attends over `cross_k` vectors read from its patch's
//!   global output, then a causal block over bytes; a 260-way byte head.
//!
//! Patch 0 is BOS alone. Position p predicts byte p + 1 and reads the global output of
//! patch `id(p + 1) - 1`, the last patch inside bytes 0..=p, so nothing leaks. Patch
//! starts depend only on earlier bytes (`Patching::starts`), so the patch a next byte
//! would open is known before it is sampled.
//!
//! Data: the trainer reads the usual BPE cache and turns each batch row into bytes
//! (`TokenBytes`, exact for byte-level BPE), so a byte model trains on the same text
//! and supervision as a token run, and its loss converts to nats per BPE token.
use super::{
    bpe::TokenizerKind,
    code_corpus::CodeConfig,
    code_curriculum::PreparedCode,
    decoder_model::{DecBlock, KvCache, KvSlot, RopeTables, YumonDecBrain, YumonDecBrainConfig},
    flash_attn::backend::FlashAttention,
    model::{RMSNorm, RMSNormConfig, TEMPERATURE, TOP_K},
    sample_cache::CacheObjective,
    samples::{Sample, TrainingStage},
    tokenizer::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN},
    train::TrainBackend,
};
use anyhow::{Context, Result, ensure};
use burn::{
    module::{AutodiffModule, Ignored},
    record::Recorder,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, Optimizer, grad_clipping::GradientClippingConfig},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{activation::{log_softmax, softmax}, backend::AutodiffBackend},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

type Inner = <TrainBackend as AutodiffBackend>::InnerBackend;

/// Byte ids: PAD, BOS, EOS and UNK keep the token ids 0..3, byte b is b + 4.
pub const BYTE_OFFSET: u32 = 4;
pub const BYTE_VOCAB: usize = 256 + BYTE_OFFSET as usize;
const PAD: u32 = PAD_TOKEN as u32;
const BOS: u32 = BOS_TOKEN as u32;
const EOS: u32 = EOS_TOKEN as u32;

pub fn byte_id(b: u8) -> u32 { b as u32 + BYTE_OFFSET }

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BltKind {
    /// The patch model.
    #[default]
    Blt,
    /// A plain byte-level decoder (`YumonDecBrain` over 260 ids): the entropy model for
    /// `entropy` patching, and an unpatched byte baseline.
    ByteLm,
}

/// Where patches start. Every mode starts patch 0 (BOS) and patch 1 at byte 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Patching {
    /// A new patch every `size` bytes.
    Static { size: usize },
    /// A new patch after the first space-like byte of a run (anything but an ASCII
    /// letter, digit, `_` or a UTF-8 byte), or when a patch reaches `max_len`.
    Space {
        #[serde(default = "default_max_patch")]
        max_len: usize,
    },
    /// A new patch where the entropy model's next-byte entropy (nats) exceeds
    /// `threshold`, or when a patch reaches `max_len`. `model` is a byte-lm run directory.
    Entropy {
        model: PathBuf,
        threshold: f32,
        #[serde(default = "default_max_patch")]
        max_len: usize,
    },
}
fn default_max_patch() -> usize { 16 }
impl Default for Patching {
    fn default() -> Self { Self::Space { max_len: default_max_patch() } }
}

/// The `blt` section of a code config. The global transformer (or the byte LM) takes
/// its shape from the config's `embed_dim`, `n_layers`, `attn_heads` and `ff_dim`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BltConfig {
    pub model: BltKind,
    /// Byte context. Rows are cut at a token boundary to fit.
    pub max_bytes: usize,
    pub patching: Patching,
    pub local_dim: usize,
    pub local_heads: usize,
    pub local_ff: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub hash_ngrams: Vec<usize>,
    pub hash_buckets: usize,
    /// Vectors per patch the decoder's cross-attention reads.
    pub cross_k: usize,
    /// Bytes generated per evaluation prompt.
    pub eval_bytes: usize,
}
impl Default for BltConfig {
    fn default() -> Self {
        Self {
            model: BltKind::Blt, max_bytes: 1024, patching: Patching::default(),
            local_dim: 128, local_heads: 4, local_ff: 512, encoder_layers: 1, decoder_layers: 2,
            hash_ngrams: vec![3, 4, 5, 6, 7, 8], hash_buckets: 4096, cross_k: 2, eval_bytes: 160,
        }
    }
}

impl Patching {
    /// Run-name part, e.g. `space16`, `static4`, `entropy1.5m16`.
    pub fn tag(&self) -> String {
        match self {
            Self::Static { size } => format!("static{size}"),
            Self::Space { max_len } => format!("space{max_len}"),
            Self::Entropy { threshold, max_len, .. } => format!("entropy{threshold}m{max_len}"),
        }
    }

    /// Patch starts over positions 0..=len of `bytes` (len = bytes.len(); index len is
    /// the position the next byte would take). Position p's start depends only on bytes
    /// before p. `entropy[q]` is the entropy model's next-byte entropy at position q.
    pub fn starts(&self, bytes: &[u32], entropy: Option<&[f32]>) -> Vec<bool> {
        let len = bytes.len();
        let mut starts = vec![false; len + 1];
        starts[0] = true;
        if len == 0 { return starts; }
        starts[1] = true;
        let cap = match self {
            Self::Static { .. } => usize::MAX,
            Self::Space { max_len } | Self::Entropy { max_len, .. } => *max_len,
        };
        let mut start = 1;
        for p in 2..=len {
            let rule = match self {
                Self::Static { size } => (p - 1) % size == 0,
                Self::Space { .. } => space_like(bytes[p - 1]) && !space_like(bytes[p - 2]),
                Self::Entropy { threshold, .. } => entropy.expect("entropy patching needs entropies")[p - 1] > *threshold,
            };
            if rule || p - start >= cap {
                starts[p] = true;
                start = p;
            }
        }
        starts
    }
}

/// Specials and every byte but ASCII letters, digits, `_` and UTF-8 bytes.
fn space_like(id: u32) -> bool {
    if id < BYTE_OFFSET { return true; }
    let b = (id - BYTE_OFFSET) as u8;
    !(b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80)
}

/// (patch of each byte, patch whose global output each position reads, patch count)
/// from `Patching::starts` (length len + 1).
pub fn patch_ids(starts: &[bool]) -> (Vec<usize>, Vec<usize>, usize) {
    let len = starts.len() - 1;
    let mut count = 0;
    let enc: Vec<usize> = (0..len).map(|p| { count += starts[p] as usize; count - 1 }).collect();
    let through_next = count + starts[len] as usize;
    let dec = (0..len).map(|p| if p + 1 < len { enc[p + 1] } else { through_next - 1 } - 1).collect();
    (enc, dec, count)
}

impl BltConfig {
    pub fn validate(&self, config: &CodeConfig) -> Result<()> {
        ensure!(self.max_bytes >= 8, "blt.max_bytes must be at least 8");
        ensure!(config.embed_dim % config.attn_heads == 0 && (config.embed_dim / config.attn_heads) % 2 == 0,
            "embed_dim must split into attn_heads of even width");
        match &self.patching {
            Patching::Static { size } => ensure!(*size >= 1, "static patch size must be positive"),
            Patching::Space { max_len } => ensure!(*max_len >= 1, "patching.max_len must be positive"),
            Patching::Entropy { threshold, max_len, .. } => {
                ensure!(*max_len >= 1, "patching.max_len must be positive");
                ensure!(threshold.is_finite(), "patching.threshold must be finite");
            }
        }
        if self.model == BltKind::Blt {
            ensure!(self.local_heads > 0 && self.local_dim % self.local_heads == 0
                && (self.local_dim / self.local_heads) % 2 == 0, "local_dim must split into local_heads of even width");
            ensure!(self.decoder_layers >= 1, "blt needs at least one decoder layer");
            ensure!(!self.hash_ngrams.is_empty() && self.hash_ngrams.iter().all(|&n| n >= 1), "hash_ngrams must be positive sizes");
            ensure!(self.hash_buckets >= 1 && self.cross_k >= 1, "hash_buckets and cross_k must be positive");
        }
        Ok(())
    }

    fn model_config(&self, config: &CodeConfig, dropout: f64) -> BltModelConfig {
        BltModelConfig::new(self.local_dim, self.local_heads, self.local_ff, self.encoder_layers,
            self.decoder_layers, config.embed_dim, config.attn_heads, config.ff_dim, config.n_layers,
            self.hash_ngrams.clone(), self.hash_buckets, self.cross_k).with_dropout(dropout)
    }

    fn byte_lm_config(&self, config: &CodeConfig, dropout: f64) -> YumonDecBrainConfig {
        YumonDecBrainConfig::new(BYTE_VOCAB, TrainingStage::Language)
            .with_embed_dim(config.embed_dim).with_hidden_units(config.embed_dim)
            .with_n_layers(config.n_layers).with_attn_heads(config.attn_heads)
            .with_ff_dim(config.ff_dim).with_max_seq_len(self.max_bytes)
            .with_dropout_rate(dropout).with_tie_embeddings(config.tie_embeddings)
    }

    fn run_name(&self, config: &CodeConfig) -> String {
        let global = format!("{}h_{}l_{}a_{}ff", config.embed_dim, config.n_layers, config.attn_heads, config.ff_dim);
        let tail = format!("{}b_b{}{}", self.max_bytes, config.batch_size,
            config.overfit_samples.map_or(String::new(), |n| format!("_overfit{n}")));
        match self.model {
            BltKind::ByteLm => format!("bytelm_{global}_{tail}{}", if config.tie_embeddings { "_tied" } else { "" }),
            BltKind::Blt => {
                let grams: Vec<String> = self.hash_ngrams.iter().map(|n| n.to_string()).collect();
                format!("blt_{}_g{global}_l{}x{}e{}d{}_hash{}x{}_k{}_{tail}", self.patching.tag(), self.local_dim,
                    self.local_ff, self.encoder_layers, self.decoder_layers, grams.join(""), self.hash_buckets, self.cross_k)
            }
        }
    }
}

// ─── Tokens to bytes ─────────────────────────────────────────────────────────

/// Byte ids of every BPE token: byte-level tokens through the GPT-2 byte <-> char
/// table, added tokens as their text, BOS/EOS/UNK as themselves, PAD as nothing.
pub struct TokenBytes {
    table: Vec<Vec<u32>>,
}

/// GPT-2 / tokenizers ByteLevel: printable bytes map to their own char, the other 68
/// to 256 + n in byte order.
fn byte_level_chars() -> [char; 256] {
    let mut chars = ['\0'; 256];
    let mut n = 0;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        chars[b as usize] = char::from_u32(if printable { b } else { n += 1; 255 + n }).unwrap();
    }
    chars
}

/// One batch row in bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct ByteRow {
    /// Byte ids, BOS first.
    pub input: Vec<u32>,
    /// Next byte id, or PAD where the token run had no target.
    pub targets: Vec<u32>,
    /// Supervised BPE targets inside the row.
    pub tokens: usize,
    /// Supervised BPE targets cut at `max_bytes`.
    pub dropped: usize,
}

impl TokenBytes {
    pub fn new(tokenizer: &TokenizerKind) -> Result<Self> {
        let TokenizerKind::Bpe(t) = tokenizer else { anyhow::bail!("BLT reads a byte-level BPE cache") };
        let to_byte: std::collections::HashMap<char, u8> =
            byte_level_chars().iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let added: std::collections::HashSet<u32> = t.inner.get_added_tokens_decoder().keys().copied().collect();
        let vocab = t.inner.get_vocab(true);
        let size = vocab.values().map(|&id| id as usize + 1).max().unwrap_or(0).max(t.vocab_size);
        let mut table = vec![Vec::new(); size];
        for (token, id) in vocab {
            table[id as usize] = match id as usize {
                PAD_TOKEN => Vec::new(),
                BOS_TOKEN | EOS_TOKEN | UNK_TOKEN => vec![id],
                _ if added.contains(&id) => token.bytes().map(byte_id).collect(),
                _ => token.chars().map(|c| to_byte.get(&c).map(|&b| byte_id(b))
                    .with_context(|| format!("token {id} {token:?} is not byte-level"))).collect::<Result<_>>()?,
            };
        }
        Ok(Self { table })
    }

    pub fn bytes(&self, token: usize) -> &[u32] { self.table.get(token).map_or(&[], Vec::as_slice) }

    /// A token row (ids and next-token targets, PAD where unsupervised, as
    /// `build_moe_batch` writes them) in bytes: each byte of a token is supervised when
    /// the token was a target. Stops at EOS; cut at a token boundary to `max_bytes`.
    pub fn row(&self, ids: &[i32], targets: &[i32], max_bytes: usize) -> ByteRow {
        let pad = PAD_TOKEN as i32;
        let mut stream: Vec<(u32, bool)> = Vec::new();
        let (mut tokens, mut dropped) = (0, 0);
        if let Some(&first) = ids.first().filter(|&&t| t != pad) {
            stream.extend(self.bytes(first as usize).iter().map(|&b| (b, false)));
            for i in 0..ids.len() {
                let (next, supervised) = if targets[i] != pad {
                    (targets[i], true)
                } else if i + 1 < ids.len() && ids[i + 1] != pad {
                    (ids[i + 1], false)
                } else {
                    break;
                };
                let bytes = self.bytes(next as usize);
                if stream.len() + bytes.len() > max_bytes + 1 {
                    dropped = targets[i..].iter().filter(|&&t| t != pad).count();
                    break;
                }
                stream.extend(bytes.iter().map(|&b| (b, supervised)));
                tokens += supervised as usize;
                if next == EOS_TOKEN as i32 { break; }
            }
        }
        let input = stream[..stream.len().saturating_sub(1)].iter().map(|s| s.0).collect();
        let targets = stream.iter().skip(1).map(|&(b, s)| if s { b } else { PAD }).collect();
        ByteRow { input, targets, tokens, dropped }
    }
}

/// FNV-1a over the n bytes ending at `p` (0 before the row start).
fn ngram_hash(bytes: &[u32], p: usize, n: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ n as u64;
    for q in (p + 1).saturating_sub(n)..=p {
        h = (h ^ (bytes[q] as u64 + 1)).wrapping_mul(0x0100_0000_01b3);
    }
    for _ in (p + 1)..n { h = (h ^ 0x1ff).wrapping_mul(0x0100_0000_01b3); }
    h ^ (h >> 31)
}

// ─── Batches ─────────────────────────────────────────────────────────────────

pub struct PatchBatch<B: Backend> {
    /// Patches per row (the batch's most).
    pub count: usize,
    /// [rows, seq * ngrams]: hashed n-gram ids with table offsets.
    pub hashes: Tensor<B, 2, Int>,
    /// [rows, count, seq]: 1 where the byte belongs to the patch.
    pub member: Tensor<B, 3>,
    /// [rows, 1, count, seq]: 0 at a patch's own bytes, -1e9 elsewhere.
    pub bias: Tensor<B, 4>,
    /// [rows * seq]: row * count + the patch each position reads.
    pub read: Tensor<B, 1, Int>,
    /// Patches that hold bytes, summed over rows.
    pub real: usize,
}

pub struct ByteBatch<B: Backend> {
    pub bytes: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 1, Int>,
    pub targets_cpu: Vec<i32>,
    pub supervised: usize,
    pub patches: Option<PatchBatch<B>>,
}

/// Right-padded batch of byte rows; with `patches`, also the patch tensors (`starts`
/// per row from `Patching::starts`).
pub fn make_batch<B: Backend>(rows: &[ByteRow], patches: Option<(&[Vec<bool>], &BltModelConfig)>,
    device: &B::Device) -> ByteBatch<B> {
    let n = rows.len();
    let seq = rows.iter().map(|r| r.input.len()).max().unwrap_or(0).max(1);
    let (mut bytes, mut targets) = (vec![PAD as i32; n * seq], vec![PAD as i32; n * seq]);
    for (r, row) in rows.iter().enumerate() {
        for (p, (&b, &t)) in row.input.iter().zip(&row.targets).enumerate() {
            bytes[r * seq + p] = b as i32;
            targets[r * seq + p] = t as i32;
        }
    }
    let supervised = targets.iter().filter(|&&t| t != PAD as i32).count();
    let patches = patches.map(|(starts, config)| {
        let ids: Vec<_> = starts.iter().map(|s| patch_ids(s)).collect();
        let count = ids.iter().map(|x| x.2).max().unwrap_or(0).max(1);
        let mut member = vec![0f32; n * count * seq];
        let mut bias = vec![-1e9f32; n * count * seq];
        let mut read = vec![0i32; n * seq];
        for (r, (enc, dec, _)) in ids.iter().enumerate() {
            for (p, (&e, &d)) in enc.iter().zip(dec).enumerate() {
                member[(r * count + e) * seq + p] = 1.0;
                bias[(r * count + e) * seq + p] = 0.0;
                read[r * seq + p] = (r * count + d) as i32;
            }
            for p in enc.len()..seq { read[r * seq + p] = (r * count) as i32; }
        }
        let (grams, buckets) = (&config.hash_ngrams, config.hash_buckets);
        let mut hashes = Vec::with_capacity(n * seq * grams.len());
        for row in rows {
            for p in 0..seq {
                for (k, &g) in grams.iter().enumerate() {
                    let h = if p < row.input.len() { (ngram_hash(&row.input, p, g) % buckets as u64) as usize } else { 0 };
                    hashes.push((k * buckets + h) as i32);
                }
            }
        }
        PatchBatch {
            count,
            hashes: Tensor::from_ints(TensorData::new(hashes, [n, seq * grams.len()]), device),
            member: Tensor::from_data(TensorData::new(member, [n, count, seq]), device),
            bias: Tensor::from_data(TensorData::new(bias, [n, 1, count, seq]), device),
            read: Tensor::from_ints(TensorData::new(read, [n * seq]), device),
            real: ids.iter().map(|x| x.2).sum(),
        }
    });
    ByteBatch {
        bytes: Tensor::from_ints(TensorData::new(bytes, [n, seq]), device),
        targets: Tensor::from_ints(TensorData::new(targets.clone(), [n * seq]), device),
        targets_cpu: targets,
        supervised,
        patches,
    }
}

// ─── Patcher (entropy model) ─────────────────────────────────────────────────

/// Patch starts for byte rows; entropy patching runs its byte LM here.
pub struct Patcher<E: Backend> {
    pub patching: Patching,
    entropy: Option<YumonDecBrain<E>>,
}

/// A byte-lm run directory: its decoder and config.
pub fn load_byte_lm<E: FlashAttention>(dir: &Path, device: &E::Device) -> Result<(YumonDecBrain<E>, YumonDecBrainConfig)> {
    let saved: Value = serde_json::from_slice(&std::fs::read(dir.join(BYTE_LM_FILE))
        .with_context(|| format!("reading {} (train it with a blt.model \"byte-lm\" config)", dir.join(BYTE_LM_FILE).display()))?)?;
    let config: YumonDecBrainConfig = serde_json::from_value(saved["model"].clone())?;
    let model = config.init::<E>(device)
        .load_file(dir.join("model"), &BinFileRecorder::<FullPrecisionSettings>::new(), device)
        .map_err(|e| anyhow::anyhow!("loading {}: {e:?}", dir.display()))?;
    Ok((model, config))
}

impl<E: FlashAttention> Patcher<E> {
    pub fn new(patching: &Patching, max_bytes: usize, device: &E::Device) -> Result<Self> {
        let entropy = match patching {
            Patching::Entropy { model, .. } => {
                let (m, config) = load_byte_lm::<E>(model, device)?;
                ensure!(config.max_seq_len >= max_bytes, "entropy model context {} is shorter than max_bytes {max_bytes}", config.max_seq_len);
                Some(m)
            }
            _ => None,
        };
        Ok(Self { patching: patching.clone(), entropy })
    }

    /// Next-byte entropy (nats) at every position of each row.
    fn entropies(model: &YumonDecBrain<E>, rows: &[ByteRow], device: &E::Device) -> Vec<Vec<f32>> {
        let seq = rows.iter().map(|r| r.input.len()).max().unwrap_or(0).max(1);
        let mut ids = vec![PAD as i32; rows.len() * seq];
        for (r, row) in rows.iter().enumerate() {
            for (p, &b) in row.input.iter().enumerate() { ids[r * seq + p] = b as i32; }
        }
        let logits = model.forward(Tensor::from_ints(TensorData::new(ids, [rows.len(), seq]), device));
        let lp = log_softmax(logits, 2);
        let h = (lp.clone().exp() * lp).sum_dim(2).neg();
        let h: Vec<f32> = h.into_data().convert::<f32>().to_vec().unwrap();
        rows.iter().enumerate().map(|(r, row)| h[r * seq..r * seq + row.input.len()].to_vec()).collect()
    }

    pub fn starts(&self, rows: &[ByteRow], device: &E::Device) -> Vec<Vec<bool>> {
        let entropies = self.entropy.as_ref().map(|m| Self::entropies(m, rows, device));
        rows.iter().enumerate()
            .map(|(i, r)| self.patching.starts(&r.input, entropies.as_ref().map(|e| e[i].as_slice())))
            .collect()
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct BltModelConfig {
    pub local_dim: usize,
    pub local_heads: usize,
    pub local_ff: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub global_dim: usize,
    pub global_heads: usize,
    pub global_ff: usize,
    pub global_layers: usize,
    pub hash_ngrams: Vec<usize>,
    pub hash_buckets: usize,
    pub cross_k: usize,
    #[config(default = 0.05)]
    pub dropout: f64,
}

/// Patches read their own bytes: query = mean of the patch's byte states.
#[derive(Module, Debug)]
struct EncoderCross<B: Backend> {
    norm: RMSNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    /// Attention output, local -> global width.
    o: Linear<B>,
    /// Mean of the patch's bytes, local -> global width.
    pool: Linear<B>,
    heads: usize,
}

/// Bytes read `k` vectors from their patch's global output.
#[derive(Module, Debug)]
struct DecoderCross<B: Backend> {
    norm: RMSNorm<B>,
    q: Linear<B>,
    /// global -> k * local
    key: Linear<B>,
    value: Linear<B>,
    o: Linear<B>,
    heads: usize,
    k: usize,
}

#[derive(Module, Debug)]
pub struct BltModel<B: Backend> {
    pub config: Ignored<BltModelConfig>,
    bytes: Embedding<B>,
    /// One table of `hash_buckets` rows per n-gram size, stacked.
    hashes: Embedding<B>,
    encoder: Vec<DecBlock<B>>,
    encoder_cross: EncoderCross<B>,
    global: Vec<DecBlock<B>>,
    global_norm: RMSNorm<B>,
    decoder_cross: Vec<DecoderCross<B>>,
    decoder: Vec<DecBlock<B>>,
    norm: RMSNorm<B>,
    head: Linear<B>,
    dropout: Dropout,
}

impl BltModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BltModel<B> {
        let (l, g) = (self.local_dim, self.global_dim);
        let linear = |from: usize, to: usize| LinearConfig::new(from, to).with_bias(false).init(device);
        BltModel {
            config: Ignored(self.clone()),
            bytes: EmbeddingConfig::new(BYTE_VOCAB, l).init(device),
            hashes: EmbeddingConfig::new(self.hash_ngrams.len() * self.hash_buckets, l).init(device),
            encoder: (0..self.encoder_layers).map(|_| DecBlock::new(l, self.local_heads, self.local_ff, device)).collect(),
            encoder_cross: EncoderCross {
                norm: RMSNormConfig::new(l).init(device),
                q: linear(l, l), k: linear(l, l), v: linear(l, l), o: linear(l, g), pool: linear(l, g),
                heads: self.local_heads,
            },
            global: (0..self.global_layers).map(|_| DecBlock::new(g, self.global_heads, self.global_ff, device)).collect(),
            global_norm: RMSNormConfig::new(g).init(device),
            decoder_cross: (0..self.decoder_layers).map(|_| DecoderCross {
                norm: RMSNormConfig::new(l).init(device),
                q: linear(l, l), key: linear(g, self.cross_k * l), value: linear(g, self.cross_k * l), o: linear(l, l),
                heads: self.local_heads, k: self.cross_k,
            }).collect(),
            decoder: (0..self.decoder_layers).map(|_| DecBlock::new(l, self.local_heads, self.local_ff, device)).collect(),
            norm: RMSNormConfig::new(l).init(device),
            head: LinearConfig::new(l, BYTE_VOCAB).with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 }).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> EncoderCross<B> {
    fn forward(&self, h: Tensor<B, 3>, p: &PatchBatch<B>) -> Tensor<B, 3> {
        let [rows, seq, d] = h.dims();
        let (heads, count) = (self.heads, p.count);
        let dh = d / heads;
        let sizes = p.member.clone().sum_dim(2).clamp_min(1.0);
        let pooled = p.member.clone().matmul(h.clone()) / sizes;
        let kv = self.norm.forward(h);
        let split = |t: Tensor<B, 3>, len: usize| t.reshape([rows, len, heads, dh]).swap_dims(1, 2);
        let q = split(self.q.forward(self.norm.forward(pooled.clone())), count);
        let k = split(self.k.forward(kv.clone()), seq);
        let v = split(self.v.forward(kv), seq);
        let scores = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt() + p.bias.clone();
        let attn = softmax(scores, 3).matmul(v).swap_dims(1, 2).reshape([rows, count, d]);
        self.pool.forward(pooled) + self.o.forward(attn)
    }

    /// `forward` for one patch's own bytes h [1, len, d]: its vector [1, 1, global].
    fn patch(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, len, d] = h.dims();
        let heads = self.heads;
        let dh = d / heads;
        let pooled = h.clone().sum_dim(1) / len as f64;
        let kv = self.norm.forward(h);
        let split = |t: Tensor<B, 3>, len: usize| t.reshape([1, len, heads, dh]).swap_dims(1, 2);
        let q = split(self.q.forward(self.norm.forward(pooled.clone())), 1);
        let k = split(self.k.forward(kv.clone()), len);
        let v = split(self.v.forward(kv), len);
        let scores = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt();
        let attn = softmax(scores, 3).matmul(v).swap_dims(1, 2).reshape([1, 1, d]);
        self.pool.forward(pooled) + self.o.forward(attn)
    }
}

impl<B: Backend> DecoderCross<B> {
    fn forward(&self, h: Tensor<B, 3>, global: Tensor<B, 3>, read: &Tensor<B, 1, Int>) -> Tensor<B, 3> {
        let [rows, seq, d] = h.dims();
        let [_, count, width] = global.dims();
        let (heads, k) = (self.heads, self.k);
        let dh = d / heads;
        let patch = global.reshape([rows * count, width]).select(0, read.clone());
        let keys = self.key.forward(patch.clone()).reshape([rows, seq, k, heads, dh]);
        let values = self.value.forward(patch).reshape([rows, seq, k, heads, dh]);
        let q = self.q.forward(self.norm.forward(h.clone())).reshape([rows, seq, 1, heads, dh]);
        let weights = softmax((q * keys).sum_dim(4) / (dh as f64).sqrt(), 2);
        h + self.o.forward((weights * values).sum_dim(2).reshape([rows, seq, d]))
    }
}

impl<B: FlashAttention> BltModel<B> {
    /// Byte logits [rows, seq, 260].
    pub fn forward(&self, batch: &ByteBatch<B>) -> Tensor<B, 3> {
        let c = &self.config;
        let p = batch.patches.as_ref().expect("BLT batches carry patches");
        let [rows, seq] = batch.bytes.dims();
        let device = batch.bytes.device();
        let n = c.hash_ngrams.len();
        let hashed = self.hashes.forward(p.hashes.clone()).reshape([rows, seq, n, c.local_dim]).sum_dim(2)
            .reshape([rows, seq, c.local_dim]);
        let x = (self.bytes.forward(batch.bytes.clone()) + hashed) / ((n + 1) as f64).sqrt();
        let mut h = self.dropout.forward(x);
        let byte_rope = RopeTables::new(seq, c.local_dim / c.local_heads, &device);
        for block in &self.encoder {
            h = block.forward_plain(h, &byte_rope);
        }
        let mut g = self.encoder_cross.forward(h.clone(), p);
        let patch_rope = RopeTables::new(p.count, c.global_dim / c.global_heads, &device);
        for block in &self.global {
            g = block.forward_plain(g, &patch_rope);
        }
        let g = self.global_norm.forward(g);
        for (cross, block) in self.decoder_cross.iter().zip(&self.decoder) {
            h = cross.forward(h, g.clone(), &p.read);
            h = block.forward_plain(h, &byte_rope);
        }
        self.head.forward(self.norm.forward(h))
    }
}

/// What the byte trainer needs from a model.
pub trait ByteModel<B: Backend> {
    fn byte_logits(&self, batch: &ByteBatch<B>) -> Tensor<B, 3>;
    /// A KV-cached decoder over one row of at most `max_bytes` bytes.
    fn decoder<'a>(&'a self, patcher: Option<&'a Patcher<B>>, max_bytes: usize, device: &B::Device)
        -> Box<dyn ByteDecoder<B> + 'a>;
}
impl<B: FlashAttention> ByteModel<B> for BltModel<B> {
    fn byte_logits(&self, batch: &ByteBatch<B>) -> Tensor<B, 3> { self.forward(batch) }
    fn decoder<'a>(&'a self, patcher: Option<&'a Patcher<B>>, max_bytes: usize, device: &B::Device)
        -> Box<dyn ByteDecoder<B> + 'a> {
        Box::new(BltDecoder::new(self, patcher.expect("BLT decoding needs a patcher"), max_bytes, device))
    }
}
impl<B: FlashAttention> ByteModel<B> for YumonDecBrain<B> {
    fn byte_logits(&self, batch: &ByteBatch<B>) -> Tensor<B, 3> { self.forward(batch.bytes.clone()) }
    fn decoder<'a>(&'a self, _: Option<&'a Patcher<B>>, _: usize, device: &B::Device) -> Box<dyn ByteDecoder<B> + 'a> {
        Box::new(ByteLmDecoder { model: self, cache: self.new_kv_cache(device), device: device.clone() })
    }
}

/// Feeds bytes one chunk at a time (the prompt, then one byte per step).
pub trait ByteDecoder<B: Backend> {
    /// Logits [t, 260] at the `t` new positions.
    fn feed(&mut self, bytes: &[u32]) -> Tensor<B, 2>;
}

fn ids_row<B: Backend>(bytes: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_ints(TensorData::new(bytes.iter().map(|&b| b as i32).collect::<Vec<_>>(), [1, bytes.len()]), device)
}

struct ByteLmDecoder<'a, B: FlashAttention> {
    model: &'a YumonDecBrain<B>,
    cache: KvCache<B>,
    device: B::Device,
}
impl<B: FlashAttention> ByteDecoder<B> for ByteLmDecoder<'_, B> {
    fn feed(&mut self, bytes: &[u32]) -> Tensor<B, 2> {
        let t = bytes.len();
        self.model.forward_cached(ids_row(bytes, &self.device), &mut self.cache).reshape([t, BYTE_VOCAB])
    }
}

/// KV-cached BLT decoding, exact against `BltModel::forward`. A position only ever
/// reads closed patches, and patch starts depend only on earlier bytes, so each byte
/// runs the encoder and decoder blocks once, and each patch runs the encoder
/// cross-attention and the global blocks once, when its last byte arrives.
pub struct BltDecoder<'a, B: FlashAttention> {
    model: &'a BltModel<B>,
    patcher: &'a Patcher<B>,
    /// The entropy model's cache, for entropy patching.
    entropy: Option<KvCache<B>>,
    bytes: Vec<u32>,
    entropies: Vec<f32>,
    encoder: Vec<KvSlot<B>>,
    global: Vec<KvSlot<B>>,
    decoder: Vec<KvSlot<B>>,
    byte_rope: RopeTables<B>,
    patch_rope: RopeTables<B>,
    /// Encoder states [1, len, local] of the open patch's bytes.
    open: Option<Tensor<B, 3>>,
    /// Normed global outputs [closed, global] of closed patches.
    closed: Option<Tensor<B, 2>>,
    count: usize,
    max_bytes: usize,
    device: B::Device,
}

impl<'a, B: FlashAttention> BltDecoder<'a, B> {
    pub fn new(model: &'a BltModel<B>, patcher: &'a Patcher<B>, max_bytes: usize, device: &B::Device) -> Self {
        let c = &model.config;
        Self {
            model, patcher,
            entropy: patcher.entropy.as_ref().map(|m| m.new_kv_cache(device)),
            bytes: Vec::new(), entropies: Vec::new(),
            encoder: (0..model.encoder.len()).map(|_| None).collect(),
            global: (0..model.global.len()).map(|_| None).collect(),
            decoder: (0..model.decoder.len()).map(|_| None).collect(),
            byte_rope: RopeTables::new(max_bytes, c.local_dim / c.local_heads, device),
            patch_rope: RopeTables::new(max_bytes, c.global_dim / c.global_heads, device),
            open: None, closed: None, count: 0, max_bytes, device: device.clone(),
        }
    }
}

impl<B: FlashAttention> ByteDecoder<B> for BltDecoder<'_, B> {
    fn feed(&mut self, new: &[u32]) -> Tensor<B, 2> {
        let (m, device) = (self.model, &self.device);
        let c = &m.config;
        let (start, t) = (self.bytes.len(), new.len());
        assert!(t > 0 && start + t <= self.max_bytes, "BLT decoding past max_bytes");
        self.bytes.extend_from_slice(new);
        if let (Some(lm), Some(cache)) = (&self.patcher.entropy, self.entropy.as_mut()) {
            let lp = log_softmax(lm.forward_cached(ids_row(new, device), cache), 2);
            let h: Vec<f32> = (lp.clone().exp() * lp).sum_dim(2).neg().into_data().convert::<f32>().to_vec().unwrap();
            self.entropies.extend(h);
        }
        let entropies = self.patcher.entropy.as_ref().map(|_| self.entropies.as_slice());
        // Through position start + t: whether each new byte closes its patch.
        let starts = self.patcher.patching.starts(&self.bytes, entropies);

        let (grams, buckets) = (&c.hash_ngrams, c.hash_buckets);
        let mut hashes = Vec::with_capacity(t * grams.len());
        for p in start..start + t {
            for (k, &g) in grams.iter().enumerate() {
                hashes.push((k * buckets + (ngram_hash(&self.bytes, p, g) % buckets as u64) as usize) as i32);
            }
        }
        let n = grams.len();
        let hashed = m.hashes.forward(Tensor::from_ints(TensorData::new(hashes, [1, t * n]), device))
            .reshape([1, t, n, c.local_dim]).sum_dim(2).reshape([1, t, c.local_dim]);
        let mut h = (m.bytes.forward(ids_row(new, device)) + hashed) / ((n + 1) as f64).sqrt();
        let rope = self.byte_rope.slice(start, t);
        for (block, slot) in m.encoder.iter().zip(&mut self.encoder) {
            h = block.forward_cached_plain(h, &rope, slot);
        }

        // Close the patches that end inside this chunk.
        let states = match self.open.take() {
            Some(open) => Tensor::cat(vec![open, h.clone()], 1),
            None => h.clone(),
        };
        let first = start + t - states.dims()[1];
        let (mut from, mut patches, mut reads) = (first, Vec::new(), Vec::with_capacity(t));
        for p in start..start + t {
            if starts[p + 1] {
                patches.push(m.encoder_cross.patch(states.clone().narrow(1, from - first, p + 1 - from)));
                from = p + 1;
            }
            reads.push((self.count + patches.len()) as i32 - 1);
        }
        if from < start + t { self.open = Some(states.narrow(1, from - first, start + t - from)); }
        if !patches.is_empty() {
            let new_patches = patches.len();
            let mut g = Tensor::cat(patches, 1);
            let rope = self.patch_rope.slice(self.count, new_patches);
            for (block, slot) in m.global.iter().zip(&mut self.global) {
                g = block.forward_cached_plain(g, &rope, slot);
            }
            let g = m.global_norm.forward(g).reshape([new_patches, c.global_dim]);
            self.closed = Some(match self.closed.take() {
                Some(closed) => Tensor::cat(vec![closed, g], 0),
                None => g,
            });
            self.count += new_patches;
        }

        let global = self.closed.clone().expect("position 0 closes the BOS patch").reshape([1, self.count, c.global_dim]);
        let read = Tensor::<B, 1, Int>::from_ints(TensorData::new(reads, [t]), device);
        for ((cross, block), slot) in m.decoder_cross.iter().zip(&m.decoder).zip(&mut self.decoder) {
            h = cross.forward(h, global.clone(), &read);
            h = block.forward_cached_plain(h, &rope, slot);
        }
        m.head.forward(m.norm.forward(h)).reshape([t, BYTE_VOCAB])
    }
}

// ─── Training ────────────────────────────────────────────────────────────────

const BLT_FILE: &str = "blt_model.json";
const BYTE_LM_FILE: &str = "byte_lm.json";

/// Rows of training samples in bytes.
struct Source {
    table: TokenBytes,
    sep: Vec<usize>,
    raw_code: bool,
    max_seq_len: usize,
    max_bytes: usize,
}
impl Source {
    fn rows(&self, samples: &[Sample], idx: &[usize]) -> Vec<ByteRow> {
        let (ids, targets, _) = super::train::build_moe_batch(samples, idx, &self.sep, self.max_seq_len, self.raw_code);
        ids.chunks(self.max_seq_len).zip(targets.chunks(self.max_seq_len))
            .map(|(i, t)| self.table.row(i, t, self.max_bytes)).collect()
    }
}

/// Where a run stands (progress.json).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Progress {
    /// Epoch in progress and batches of it done.
    epoch: usize,
    batch: usize,
    steps: usize,
    #[serde(default)]
    val: Option<Value>,
}

/// Held-out sums.
#[derive(Default)]
struct Held {
    nll: f64,
    bytes: usize,
    tokens: usize,
    dropped: usize,
    correct: usize,
    patches: usize,
    patched_bytes: usize,
}
impl Held {
    fn report(&self) -> Value {
        let ln2 = std::f64::consts::LN_2;
        let mut v = json!({
            "bpb": self.nll / self.bytes.max(1) as f64 / ln2,
            "nats_per_byte": self.nll / self.bytes.max(1) as f64,
            "nats_per_token": self.nll / self.tokens.max(1) as f64,
            "top1": self.correct as f64 / self.bytes.max(1) as f64,
            "bytes": self.bytes, "tokens": self.tokens, "dropped_tokens": self.dropped,
        });
        if self.patches > 0 { v["bytes_per_patch"] = json!(self.patched_bytes as f64 / self.patches as f64); }
        v
    }
}

/// Per-position NLL sum and correct argmax count over supervised bytes.
fn score<B: Backend>(logits: Tensor<B, 3>, batch: &ByteBatch<B>) -> (f64, usize) {
    let [n, seq, _] = logits.dims();
    let lp = log_softmax(logits, 2);
    let mask = batch.targets.clone().not_equal_elem(PAD as i32).float();
    let nll = lp.clone().gather(2, batch.targets.clone().reshape([n, seq, 1])).reshape([n * seq]).neg() * mask.clone();
    let nll: f32 = nll.sum().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    let best = lp.argmax(2).reshape([n * seq]);
    let correct = best.equal(batch.targets.clone()).float() * mask;
    let correct: f32 = correct.sum().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    (nll as f64, correct.round() as usize)
}

fn validate<B: FlashAttention, M: ByteModel<B>>(model: &M, source: &Source, samples: &[Sample],
    patcher: Option<&Patcher<B>>, config: Option<&BltModelConfig>, batch_size: usize, device: &B::Device) -> Held {
    let mut held = Held::default();
    let idx: Vec<usize> = (0..samples.len()).collect();
    for chunk in idx.chunks(batch_size) {
        let rows = source.rows(samples, chunk);
        let starts = patcher.map(|p| p.starts(&rows, device));
        let batch = make_batch::<B>(&rows, starts.as_deref().zip(config), device);
        if batch.supervised == 0 { continue; }
        let (nll, correct) = score(model.byte_logits(&batch), &batch);
        held.nll += nll;
        held.correct += correct;
        held.bytes += batch.supervised;
        held.tokens += rows.iter().map(|r| r.tokens).sum::<usize>();
        held.dropped += rows.iter().map(|r| r.dropped).sum::<usize>();
        if let Some(p) = &batch.patches {
            held.patches += p.real;
            held.patched_bytes += rows.iter().map(|r| r.input.len()).sum::<usize>();
        }
    }
    held
}

/// Samples `max_new` bytes after `prompt` (lowercased like the yumon_bpe normalizer)
/// with the model's KV-cached decoder.
fn generate<B: FlashAttention, M: ByteModel<B>>(model: &M, patcher: Option<&Patcher<B>>,
    prompt: &str, max_new: usize, max_bytes: usize, seed: u64, device: &B::Device) -> String {
    let mut input = vec![BOS];
    input.extend(prompt.to_lowercase().bytes().map(byte_id));
    input.truncate(max_bytes);
    let start = input.len();
    let mut state = seed | 1;
    let mut unit = move || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut decoder = model.decoder(patcher, max_bytes, device);
    let mut fed = 0;
    while input.len() < max_bytes && input.len() - start < max_new {
        let logits = decoder.feed(&input[fed..]);
        fed = input.len();
        let rows = logits.dims()[0];
        let logits: Vec<f32> = logits.narrow(0, rows - 1, 1).into_data().convert::<f32>().to_vec().unwrap();
        let mut order: Vec<usize> = (0..logits.len()).filter(|&i| i != PAD as usize && i != BOS as usize).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(TOP_K.max(1));
        let top = logits[order[0]];
        let weights: Vec<f64> = order.iter().map(|&i| (((logits[i] - top) / TEMPERATURE) as f64).exp()).collect();
        let mut pick = unit() * weights.iter().sum::<f64>();
        let mut next = order[0];
        for (&i, w) in order.iter().zip(&weights) {
            if pick < *w { next = i; break; }
            pick -= w;
        }
        if next as u32 == EOS { break; }
        input.push(next as u32);
    }
    let bytes: Vec<u8> = input[start..].iter().filter(|&&b| b >= BYTE_OFFSET).map(|&b| (b - BYTE_OFFSET) as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn append(path: &Path, line: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Byte statistics of `limit` training rows spread evenly over the set (and patch sizes
/// when `patcher` is given).
fn data_report(source: &Source, samples: &[Sample], patcher: Option<&Patcher<Inner>>, limit: usize,
    device: &<Inner as Backend>::Device) -> Value {
    let take = samples.len().min(limit);
    let idx: Vec<usize> = (0..take).map(|i| i * samples.len() / take).collect();
    let (mut rows_n, mut bytes, mut supervised, mut tokens, mut dropped, mut full, mut patches) = (0, 0, 0, 0, 0, 0, 0);
    for chunk in idx.chunks(64) {
        let rows = source.rows(samples, chunk);
        if let Some(p) = patcher {
            patches += p.starts(&rows, device).iter().map(|s| patch_ids(s).2).sum::<usize>();
        }
        for r in &rows {
            rows_n += 1;
            bytes += r.input.len();
            supervised += r.targets.iter().filter(|&&t| t != PAD).count();
            tokens += r.tokens;
            dropped += r.dropped;
            full += (r.dropped > 0) as usize;
        }
    }
    let mut v = json!({"rows": rows_n, "mean_bytes": bytes as f64 / rows_n.max(1) as f64,
        "bytes_per_token": supervised as f64 / tokens.max(1) as f64,
        "rows_cut": full, "dropped_token_share": dropped as f64 / (tokens + dropped).max(1) as f64});
    if patches > 0 { v["bytes_per_patch"] = json!(bytes as f64 / patches as f64); }
    v
}

/// `train_code` with a `blt` section: trains a BLT or a byte LM on the prepared cache.
pub fn run(config: &CodeConfig, blt: &BltConfig, tokenizer: &TokenizerKind, prepared: PreparedCode,
    prompts: &[String], check_only: bool) -> Result<()> {
    ensure!(prepared.objective != CacheObjective::DpoPreference, "blt does not train DPO pairs");
    let raw_code = prepared.objective == CacheObjective::RawCode;
    let sep = if raw_code { Vec::new() } else { tokenizer.encode(super::samples::language_separator_for(tokenizer)) };
    let source = Source { table: TokenBytes::new(tokenizer)?, sep, raw_code, max_seq_len: config.max_seq_len, max_bytes: blt.max_bytes };
    let device = <Inner as Backend>::Device::default();
    // Static and space patching need no model, so --check can report patch sizes.
    let check_patcher = (blt.model == BltKind::Blt && !matches!(blt.patching, Patching::Entropy { .. }))
        .then(|| Patcher::<Inner>::new(&blt.patching, blt.max_bytes, &device)).transpose()?;
    let report = data_report(&source, &prepared.training, check_patcher.as_ref(), 2000, &device);
    println!("Bytes ({} training rows spread over the set, max_bytes {}): {report}", report["rows"], blt.max_bytes);
    if check_only { return Ok(()); }

    let overfit = config.overfit_samples.is_some();
    let dropout = if overfit { 0.0 } else { 0.05 };
    let run_dir = config.out_dir.join(blt.run_name(config));
    std::fs::create_dir_all(&run_dir)?;
    let identity = json!({"config": config, "cache": prepared.identity});
    let identity_path = run_dir.join("code-config.json");
    if identity_path.exists() {
        let previous: Value = serde_json::from_slice(&std::fs::read(&identity_path)?)?;
        ensure!(previous["cache"] == identity["cache"], "checkpoint cache identity differs; use a new out_dir");
    }
    std::fs::write(&identity_path, serde_json::to_vec_pretty(&identity)?)?;
    <TrainBackend as Backend>::seed(&device, config.seed);
    println!("Run directory: {}", run_dir.display());
    match blt.model {
        BltKind::Blt => {
            let model_config = blt.model_config(config, dropout);
            let saved = json!({"model": model_config, "patching": blt.patching, "max_bytes": blt.max_bytes});
            let (model, progress) = init_or_resume(&run_dir, BLT_FILE, saved, model_config.init::<TrainBackend>(&device), &device)?;
            let patcher = Patcher::<Inner>::new(&blt.patching, blt.max_bytes, &device)?;
            println!("BLT: {:.2}M parameters ({:.2}M in hash tables), patching {:?}.",
                model.num_params() as f64 / 1e6,
                (model_config.hash_ngrams.len() * model_config.hash_buckets * model_config.local_dim) as f64 / 1e6, blt.patching);
            train(model, progress, config, blt, &source, prepared, Some((&patcher, &model_config)), prompts, &run_dir, &device)
        }
        BltKind::ByteLm => {
            let model_config = blt.byte_lm_config(config, dropout);
            let saved = json!({"model": model_config});
            let (model, progress) = init_or_resume(&run_dir, BYTE_LM_FILE, saved, model_config.init::<TrainBackend>(&device), &device)?;
            println!("Byte LM: {:.2}M parameters.", model.num_params() as f64 / 1e6);
            train(model, progress, config, blt, &source, prepared, None, prompts, &run_dir, &device)
        }
    }
}

fn init_or_resume<M: Module<TrainBackend>>(dir: &Path, file: &str, config: Value, fresh: M,
    device: &<TrainBackend as Backend>::Device) -> Result<(M, Progress)> {
    let path = dir.join(file);
    if dir.join("model.bin").exists() {
        let saved: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        ensure!(saved == config, "{} differs from the requested model; use a new out_dir", path.display());
        let model = fresh.load_file(dir.join("model"), &BinFileRecorder::<FullPrecisionSettings>::new(), device)
            .map_err(|e| anyhow::anyhow!("loading {}: {e:?}", dir.display()))?;
        let progress: Progress = serde_json::from_slice(&std::fs::read(dir.join("progress.json"))?)?;
        println!("Resuming at epoch {}, batch {} ({} steps done).", progress.epoch + 1, progress.batch, progress.steps);
        return Ok((model, progress));
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&config)?)?;
    Ok((fresh, Progress::default()))
}

fn save<M: AutodiffModule<TrainBackend>>(model: &M, optimizer: Option<&impl Optimizer<M, TrainBackend>>,
    progress: &Progress, dir: &Path) -> Result<()> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(dir.join("model"), &recorder).map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;
    if let Some(optimizer) = optimizer {
        recorder.record(optimizer.to_record(), dir.join("optimizer")).map_err(|e| anyhow::anyhow!("saving optimizer: {e:?}"))?;
    }
    std::fs::write(dir.join("progress.json"), serde_json::to_vec_pretty(progress)?)?;
    Ok(())
}

/// Validation point on the dashboard: nats per byte, so it shares the loss axis.
fn record_val(state: &mut super::chart::TrainingState, report: &Value, reply: String) {
    let x = state.global_step as f64;
    if let Some(v) = report["nats_per_byte"].as_f64() { state.val_loss_history.push((x, v)); }
    if let Some(v) = report["top1"].as_f64() { state.val_top3_history.push((x, v)); }
    state.last_reply = reply;
}

#[allow(clippy::too_many_arguments)]
fn train<M>(mut model: M, mut progress: Progress, config: &CodeConfig, blt: &BltConfig, source: &Source,
    prepared: PreparedCode, patches: Option<(&Patcher<Inner>, &BltModelConfig)>, prompts: &[String],
    run_dir: &Path, device: &<TrainBackend as Backend>::Device) -> Result<()>
where
    M: AutodiffModule<TrainBackend> + ByteModel<TrainBackend>,
    M::InnerModule: ByteModel<Inner>,
{
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
    let overfit = config.overfit_samples.is_some();
    let last_lr = if overfit { config.first_lr } else { config.last_lr };
    let plan = prepared.batch_plan(config.epochs, config.batch_size);
    let total: usize = plan.iter().sum();
    let mut optimizer = AdamWConfig::new().with_epsilon(1e-7).with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))).init::<TrainBackend, M>();
    // AdamW moments, so a resumed run continues instead of restarting them.
    if progress.steps > 0 && run_dir.join("optimizer.bin").exists() {
        let record = BinFileRecorder::<FullPrecisionSettings>::new().load(run_dir.join("optimizer"), device)
            .map_err(|e| anyhow::anyhow!("loading optimizer state: {e:?}"))?;
        optimizer = optimizer.load_record(record);
    }
    let (patcher, model_config) = (patches.map(|p| p.0), patches.map(|p| p.1));
    let log = run_dir.join("blt.jsonl");
    let inference_log = run_dir.join("inference_log.txt");
    println!("Training {} rows, {} held out; {total} steps over {} epochs.", prepared.training.len(), prepared.validation.len(), config.epochs);
    // The decoder trainer's dashboard: losses in nats per byte, top-3 over bytes.
    use std::io::IsTerminal;
    let mut terminal = if std::io::stdout().is_terminal() {
        Some(ratatui::Terminal::with_options(ratatui::prelude::CrosstermBackend::new(std::io::stdout()),
            ratatui::TerminalOptions { viewport: ratatui::Viewport::Inline(46) })?)
    } else {
        None
    };
    let mut state = super::chart::TrainingState {
        loss_history: vec![], avg_loss_history: vec![], val_loss_history: vec![], val_top3_history: vec![],
        current_loss: 0.0, avg_loss: 0.0, epoch: 0, total_epochs: config.epochs, batch: 0, total_batches: 0,
        current_lr: config.first_lr, lr_history: vec![], global_step: progress.steps, entropy: 0.0,
        entropy_history: vec![], top3_accuracy: None, avg_top3_accuracy: None, top3_history: vec![],
        last_reply: String::new(),
    };
    let chart_path = run_dir.join("blt_chart.png").to_string_lossy().into_owned();
    let quiet = terminal.is_some();

    let held_out = |model: &M::InnerModule, epoch: usize, steps: usize| -> (Value, String) {
        let started = std::time::Instant::now();
        let held = validate(model, source, &prepared.validation, patcher, model_config, config.batch_size, device);
        let mut report = held.report();
        report["epoch"] = json!(epoch);
        report["step"] = json!(steps);
        if !quiet { println!("val {report} ({:.1}s)", started.elapsed().as_secs_f64()); }
        append(&log, &json!({"validation": report}).to_string());
        let mut lines = vec![format!("--- step {steps} (epoch {epoch}) val bpb {:.4}, nats/token {:.4}",
            report["bpb"].as_f64().unwrap_or(f64::NAN), report["nats_per_token"].as_f64().unwrap_or(f64::NAN))];
        for (i, prompt) in prompts.iter().enumerate() {
            let text = generate(model, patcher, prompt, blt.eval_bytes, blt.max_bytes, steps as u64 + i as u64, device);
            lines.push(format!("{prompt:?} -> {text:?}"));
        }
        if !quiet { for line in &lines { println!("{line}"); } }
        append(&inference_log, &lines.join("\n"));
        let reply = lines.get(1).cloned().unwrap_or_default();
        (report, reply)
    };

    // Training sums since the last log line.
    #[derive(Default)]
    struct Window { nll: f64, supervised: usize, tokens: usize, bytes: usize, patches: usize, steps: usize }
    let mut window = Window::default();
    let mut window_start = std::time::Instant::now();
    for epoch in progress.epoch..config.epochs {
        let (mut epoch_nll, mut epoch_bytes, mut epoch_top3) = (0.0f64, 0usize, 0usize);
        let mut idx = prepared.epoch_indices(epoch);
        if prepared.shuffle {
            idx.shuffle(&mut StdRng::seed_from_u64(prepared.seed.wrapping_add(epoch as u64)));
        }
        let batches: Vec<&[usize]> = idx.chunks(config.batch_size).collect();
        state.epoch = epoch + 1;
        state.total_batches = batches.len();
        for (b, chunk) in batches.iter().enumerate().skip(progress.batch) {
            let step = plan[..epoch].iter().sum::<usize>() + b;
            let t = step as f64 / total.max(1) as f64;
            let lr = config.first_lr * (1.0 - t) + last_lr * t;
            let rows = source.rows(&prepared.training, chunk);
            let starts = patcher.map(|p| p.starts(&rows, device));
            let batch = make_batch::<TrainBackend>(&rows, starts.as_deref().zip(model_config), device);
            progress.batch = b + 1;
            progress.steps += 1;
            if batch.supervised == 0 { continue; }
            let [n, seq] = batch.bytes.dims();
            let logits = model.byte_logits(&batch).reshape([n * seq, BYTE_VOCAB]);
            let top3 = super::train::top3_correct(logits.clone().inner(), batch.targets.clone().inner());
            // Mean predictive entropy over real positions, on detached logits.
            let lp = log_softmax(logits.clone().detach(), 1);
            let real = batch.bytes.clone().reshape([n * seq]).not_equal_elem(PAD as i32).float();
            let entropy: f32 = ((lp.clone().exp() * lp).sum_dim(1).reshape([n * seq]).neg() * real.clone()).sum()
                .div(real.sum().clamp_min(1.0)).into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
            let loss = super::train::masked_token_ce(logits, batch.targets.clone(), batch.supervised, 0.0);
            let value: f32 = loss.clone().inner().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(lr, model, grads);
            window.nll += value as f64 * batch.supervised as f64;
            window.supervised += batch.supervised;
            window.tokens += rows.iter().map(|r| r.tokens).sum::<usize>();
            window.bytes += rows.iter().map(|r| r.input.len()).sum::<usize>();
            window.patches += batch.patches.as_ref().map_or(0, |p| p.real);
            window.steps += 1;
            epoch_nll += value as f64 * batch.supervised as f64;
            epoch_bytes += batch.supervised;
            epoch_top3 += top3;
            let x = progress.steps as f64;
            state.global_step = progress.steps;
            state.batch = b + 1;
            state.current_lr = lr;
            state.current_loss = value;
            state.avg_loss = (epoch_nll / epoch_bytes.max(1) as f64) as f32;
            state.entropy = entropy;
            state.top3_accuracy = Some(top3 as f64 / batch.supervised as f64);
            state.avg_top3_accuracy = Some(epoch_top3 as f64 / epoch_bytes.max(1) as f64);
            state.loss_history.push((x, value as f64));
            state.avg_loss_history.push((x, state.avg_loss as f64));
            state.entropy_history.push((x, entropy as f64));
            state.lr_history.push((x, lr));
            state.top3_history.push((x, top3 as f64 / batch.supervised as f64));
            if let Some(term) = terminal.as_mut() { term.draw(|frame| super::chart::render(frame, &state))?; }
            if progress.steps % 50 == 0 {
                let secs = window_start.elapsed().as_secs_f64();
                let line = json!({"step": progress.steps, "epoch": epoch + 1, "lr": lr,
                    "bpb": window.nll / window.supervised.max(1) as f64 / std::f64::consts::LN_2,
                    "nats_per_token": window.nll / window.tokens.max(1) as f64,
                    "bytes_per_patch": (window.patches > 0).then(|| window.bytes as f64 / window.patches as f64),
                    "ms_per_step": 1e3 * secs / window.steps.max(1) as f64, "bytes_per_sec": window.bytes as f64 / secs});
                if !quiet { println!("epoch {}/{} batch {}/{} {line}", epoch + 1, config.epochs, b + 1, batches.len()); }
                append(&log, &json!({"train": line}).to_string());
                window = Window::default();
                window_start = std::time::Instant::now();
            }
            if progress.steps % 2000 == 0 {
                let (report, reply) = held_out(&model.valid(), epoch + 1, progress.steps);
                record_val(&mut state, &report, reply);
                progress.val = Some(report);
            }
            if progress.steps % 500 == 0 {
                save(&model, Some(&optimizer), &progress, run_dir)?;
                if let Err(e) = state.save_chart_image(&chart_path) { eprintln!("chart: {e}"); }
            }
        }
        progress.epoch = epoch + 1;
        progress.batch = 0;
        let (report, reply) = held_out(&model.valid(), epoch + 1, progress.steps);
        record_val(&mut state, &report, reply);
        progress.val = Some(report);
        save(&model, Some(&optimizer), &progress, run_dir)?;
        if let Err(e) = state.save_chart_image(&chart_path) { eprintln!("chart: {e}"); }
    }
    if let Some(term) = terminal.as_mut() { term.clear()?; }
    if let Some(val) = &progress.val { println!("Final validation: {val}"); }
    println!("Chart: {chart_path}");
    println!("Done: {}", run_dir.display());
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/blt/mod.rs"]
mod tests;
