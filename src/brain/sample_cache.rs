//! Portable, versioned snapshots of prepared training samples.
//! Bump FORMAT_VERSION whenever sample preparation or the stored schema changes.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use bincode::Options;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    PAD_TOKEN,
    bpe::TokenizerKind,
    samples::{Action, CardinalDir, Sample, TrainingStage, WorldContext},
};

const MAGIC: &[u8; 8] = b"YUMONSMP";
const FORMAT_VERSION: u32 = 1;
const BUFFER_SIZE: usize = 1 << 20;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheObjective {
    #[default]
    PromptReply,
    RawCode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub version: u32,
    #[serde(default)]
    pub objective: CacheObjective,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation: Option<String>,
    pub stage: TrainingStage,
    pub max_seq_len: usize,
    pub tokenizer_sha256: String,
    pub samples: usize,
    pub seed: u64,
}

// Persist active token IDs as u32 and reconstruct padding on load. This avoids
// serializing hundreds of padding IDs for every short training pair.
#[derive(Serialize, Deserialize)]
struct StoredSample {
    input_ids: Vec<u32>,
    target_labels: Vec<u32>,
    action: Action,
    motion_dir: CardinalDir,
    world: WorldContext,
    target_json: String,
    pair: (String, String),
}

impl StoredSample {
    fn from_sample(sample: &Sample) -> Result<Self> {
        let compact = |ids: &[usize]| -> Result<Vec<u32>> {
            let end = ids
                .iter()
                .rposition(|&id| id != PAD_TOKEN)
                .map_or(0, |i| i + 1);
            ids[..end]
                .iter()
                .map(|&id| u32::try_from(id).map_err(Into::into))
                .collect()
        };
        Ok(Self {
            input_ids: compact(&sample.input_ids)?,
            target_labels: compact(&sample.target_labels)?,
            action: sample.action,
            motion_dir: sample.motion_dir,
            world: sample.world,
            target_json: sample.target_json.clone(),
            pair: sample.pair.clone(),
        })
    }

    fn into_sample(self, max_seq_len: usize) -> Result<Sample> {
        ensure!(
            self.input_ids.len() <= max_seq_len && self.target_labels.len() <= max_seq_len,
            "cached sample exceeds the context length"
        );
        let pad = |ids: Vec<u32>| {
            let mut ids: Vec<usize> = ids.into_iter().map(|id| id as usize).collect();
            ids.resize(max_seq_len, PAD_TOKEN);
            ids
        };
        Ok(Sample {
            input_ids: pad(self.input_ids),
            target_labels: pad(self.target_labels),
            action: self.action,
            motion_dir: self.motion_dir,
            world: self.world,
            target_json: self.target_json,
            pair: self.pair,
        })
    }
}

/// Canonical JSON includes vocabulary, normalization and special-token settings.
fn tokenizer_fingerprint(tokenizer: &TokenizerKind) -> Result<String> {
    let (kind, value) = match tokenizer {
        TokenizerKind::Char(tokenizer) => ("char", serde_json::to_value(tokenizer)?),
        TokenizerKind::Bpe(tokenizer) => {
            let json = tokenizer
                .inner
                .to_string(false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            ("bpe", serde_json::from_str(&json)?)
        }
    };
    let bytes = serde_json::to_vec(&(kind, value))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_sample(sample: &Sample, max_seq_len: usize, vocab_size: usize) -> Result<()> {
    ensure!(
        sample.input_ids.len() == max_seq_len && sample.target_labels.len() == max_seq_len,
        "sample tensor dimensions do not match cache context length"
    );
    ensure!(
        sample
            .input_ids
            .iter()
            .chain(&sample.target_labels)
            .all(|&id| id < vocab_size),
        "sample contains a token outside the tokenizer vocabulary"
    );
    Ok(())
}

fn code_digest(samples: &[Sample]) -> String {
    let mut digest = Sha256::new();
    for sample in samples {
        digest.update((sample.pair.0.len() as u64).to_le_bytes());
        digest.update(sample.pair.0.as_bytes());
        for ids in [&sample.input_ids, &sample.target_labels] {
            digest.update((ids.len() as u64).to_le_bytes());
            for &id in ids { digest.update((id as u64).to_le_bytes()); }
        }
    }
    format!("{digest:x}", digest = digest.finalize())
}

fn validate_code_sample(sample: &Sample) -> Result<()> {
    use super::{BOS_TOKEN, EOS_TOKEN};
    let end = sample.target_labels.iter().position(|&t| t == PAD_TOKEN)
        .unwrap_or(sample.target_labels.len());
    ensure!(end >= 2 && sample.input_ids[0] == BOS_TOKEN
        && sample.target_labels[end - 1] == EOS_TOKEN
        && sample.input_ids[1..end] == sample.target_labels[..end - 1]
        && sample.target_labels[..end - 1].iter().all(|&id| id > 3)
        && sample.input_ids[end..].iter().all(|&id| id == PAD_TOKEN)
        && sample.target_labels[end..].iter().all(|&id| id == PAD_TOKEN)
        && !sample.pair.0.is_empty(), "invalid shifted raw-code sample");
    Ok(())
}

struct TemporaryFile(PathBuf);
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Stream a compressed binary snapshot to a sibling temporary file, then publish it.
/// Existing caches are never overwritten. No second copy of the dataset is allocated.
pub fn write_cache(
    path: &Path,
    tokenizer: &TokenizerKind,
    stage: TrainingStage,
    max_seq_len: usize,
    seed: u64,
    samples: &[Sample],
) -> Result<CacheMetadata> {
    write_cache_with_objective(path, tokenizer, stage, max_seq_len, seed, samples, CacheObjective::PromptReply)
}

pub fn write_cache_with_objective(
    path: &Path, tokenizer: &TokenizerKind, stage: TrainingStage,
    max_seq_len: usize, seed: u64, samples: &[Sample], objective: CacheObjective,
) -> Result<CacheMetadata> {
    ensure!(max_seq_len >= 2, "context length must be at least 2");
    ensure!(
        !samples.is_empty(),
        "refusing to cache an empty training dataset"
    );
    ensure!(
        !path.exists(),
        "{} already exists; choose a new output path",
        path.display()
    );
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temporary = TemporaryFile(parent.join(format!(".samples-{}.tmp", uuid::Uuid::new_v4())));
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&temporary.0)?;
    let metadata = CacheMetadata {
        // Old binaries must refuse raw-code caches: their batcher uses prompt masking.
        version: if objective == CacheObjective::RawCode { 2 } else { FORMAT_VERSION },
        objective,
        corpus_sha256: (objective == CacheObjective::RawCode).then(|| code_digest(samples)),
        preparation: (objective == CacheObjective::RawCode).then(|| "rust-items-v1".to_owned()),
        stage,
        max_seq_len,
        tokenizer_sha256: tokenizer_fingerprint(tokenizer)?,
        samples: samples.len(),
        seed,
    };
    let header = serde_json::to_vec(&metadata)?;
    file.write_all(MAGIC)?;
    file.write_all(&u32::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    // Compression shrinks padding and repeated strings without re-tokenizing on load.
    let mut encoder = zstd::stream::write::Encoder::new(file, 1)?;
    encoder.include_checksum(true)?;
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, encoder);
    for (index, sample) in samples.iter().enumerate() {
        validate_sample(sample, max_seq_len, tokenizer.vocab_size())
            .with_context(|| format!("sample {index}"))?;
        if objective == CacheObjective::RawCode { validate_code_sample(sample)?; }
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_little_endian()
            .serialize_into(&mut writer, &StoredSample::from_sample(sample)?)?;
    }
    let encoder = writer
        .into_inner()
        .map_err(|e| anyhow::anyhow!("flushing sample cache: {e}"))?;
    let file = encoder.finish()?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary.0, path)
        .with_context(|| format!("publishing {}", path.display()))?;
    Ok(metadata)
}

/// Read only the small uncompressed header, without loading any samples.
pub fn read_metadata(path: &Path) -> Result<CacheMetadata> {
    read_header(&mut File::open(path).with_context(|| format!("opening {}", path.display()))?)
}

fn read_header(reader: &mut impl Read) -> Result<CacheMetadata> {
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic == MAGIC, "not a Yumon sample cache");
    let mut length = [0; 4];
    reader.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    ensure!(length <= BUFFER_SIZE, "invalid cache header size");
    let mut header = vec![0; length];
    reader.read_exact(&mut header)?;
    let metadata: CacheMetadata = serde_json::from_slice(&header)?;
    ensure!(metadata.objective != CacheObjective::RawCode ||
        (metadata.version == 2 && metadata.corpus_sha256.is_some()), "raw-code cache requires version 2 and a corpus digest");
    ensure!(
        metadata.version == FORMAT_VERSION || metadata.version == 2,
        "unsupported sample cache version {}; rebuild locally",
        metadata.version
    );
    ensure!(
        metadata.max_seq_len >= 2 && metadata.samples > 0,
        "invalid sample cache metadata"
    );
    Ok(metadata)
}

/// Load exactly the cached samples. A mismatch or damaged file is an error;
/// training never silently falls back to expensive preprocessing.
pub fn load_cache(
    path: &Path,
    tokenizer: &TokenizerKind,
    stage: TrainingStage,
    max_seq_len: usize,
) -> Result<Vec<Sample>> {
    load_cache_with_objective(path, tokenizer, stage, max_seq_len, CacheObjective::PromptReply)
}

pub fn load_cache_with_objective(
    path: &Path, tokenizer: &TokenizerKind, stage: TrainingStage,
    max_seq_len: usize, objective: CacheObjective,
) -> Result<Vec<Sample>> {
    let mut file =
        File::open(path).with_context(|| format!("opening sample cache {}", path.display()))?;
    let metadata = read_header(&mut file)?;
    ensure!(metadata.objective == objective, "cache objective {:?} does not match {:?}", metadata.objective, objective);
    if objective == CacheObjective::RawCode {
        ensure!(metadata.preparation.as_deref() == Some("rust-items-v1"),
            "code cache uses old chunking; rebuild with cache_samples to use complete Rust items");
    }
    ensure!(
        metadata.stage == stage,
        "cache stage {:?} does not match {:?}; rebuild locally",
        metadata.stage,
        stage
    );
    ensure!(
        metadata.max_seq_len == max_seq_len,
        "cache context length {} does not match {}; rebuild locally",
        metadata.max_seq_len,
        max_seq_len
    );
    ensure!(
        metadata.tokenizer_sha256 == tokenizer_fingerprint(tokenizer)?,
        "cache tokenizer does not match the training tokenizer; rebuild locally"
    );
    let decoder = zstd::stream::read::Decoder::new(file)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, decoder);
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(metadata.samples)
        .context("allocating cached samples")?;
    for index in 0..metadata.samples {
        let stored: StoredSample = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_little_endian()
            .with_limit(64 << 20)
            .deserialize_from(&mut reader)
            .with_context(|| format!("reading cached sample {index}"))?;
        let sample = stored.into_sample(max_seq_len)?;
        validate_sample(&sample, max_seq_len, tokenizer.vocab_size())
            .with_context(|| format!("cached sample {index}"))?;
        if objective == CacheObjective::RawCode { validate_code_sample(&sample)?; }
        samples.push(sample);
    }
    let mut trailing = [0; 1];
    if objective == CacheObjective::RawCode {
        ensure!(metadata.corpus_sha256.as_deref() == Some(code_digest(&samples).as_str()), "cache corpus digest mismatch");
    }
    ensure!(
        reader.read(&mut trailing)? == 0,
        "unexpected extra data in sample cache"
    );
    println!(
        "[SampleCache] loaded {} {:?} samples (context {})",
        samples.len(),
        stage,
        max_seq_len
    );
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{Tokenizer, samples::prepare_paired_samples_split_sep};
    use rand::{SeedableRng, rngs::StdRng};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("yumon-cache-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> PathBuf {
            self.0.join("samples.bin")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tokenizer() -> TokenizerKind {
        let ascii: String = (32u8..=126).map(char::from).chain(['\n']).collect();
        TokenizerKind::Char(Tokenizer::build_from_text(&ascii, 256))
    }
    fn samples(tokenizer: &TokenizerKind) -> Vec<Sample> {
        prepare_paired_samples_split_sep(
            vec![(
                "Tell me about the moon".into(),
                "The moon orbits the Earth.".into(),
            )],
            tokenizer,
            &Default::default(),
            &mut StdRng::seed_from_u64(42),
            TrainingStage::Language,
            128,
        )
    }

    #[test]
    fn round_trip_preserves_every_sample_field_and_metadata() {
        let fixture = Fixture::new();
        let tokenizer = tokenizer();
        let samples = samples(&tokenizer);
        assert_eq!(samples.len(), 1);
        let metadata = write_cache(
            &fixture.path(),
            &tokenizer,
            TrainingStage::Language,
            128,
            42,
            &samples,
        )
        .unwrap();
        assert_eq!(metadata.samples, 1);
        assert_eq!(read_metadata(&fixture.path()).unwrap().seed, 42);
        let loaded = load_cache(&fixture.path(), &tokenizer, TrainingStage::Language, 128).unwrap();
        assert_eq!(format!("{loaded:?}"), format!("{samples:?}"));
        assert!(
            write_cache(
                &fixture.path(),
                &tokenizer,
                TrainingStage::Language,
                128,
                42,
                &samples
            )
            .is_err()
        );
        assert_eq!(
            load_cache(&fixture.path(), &tokenizer, TrainingStage::Language, 128)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn incompatible_stage_context_and_tokenizer_are_rejected() {
        let fixture = Fixture::new();
        let tokenizer = tokenizer();
        write_cache(
            &fixture.path(),
            &tokenizer,
            TrainingStage::Language,
            128,
            42,
            &samples(&tokenizer),
        )
        .unwrap();
        assert!(
            load_cache(&fixture.path(), &tokenizer, TrainingStage::Structured, 128)
                .unwrap_err()
                .to_string()
                .contains("stage")
        );
        assert!(
            load_cache(&fixture.path(), &tokenizer, TrainingStage::Language, 256)
                .unwrap_err()
                .to_string()
                .contains("context length")
        );
        let different = TokenizerKind::Char(Tokenizer::build_from_text("another vocabulary", 256));
        assert!(
            load_cache(&fixture.path(), &different, TrainingStage::Language, 128)
                .unwrap_err()
                .to_string()
                .contains("tokenizer")
        );
    }

    #[test]
    fn truncated_payload_bad_magic_and_future_versions_are_rejected() {
        let fixture = Fixture::new();
        let tokenizer = tokenizer();
        write_cache(
            &fixture.path(),
            &tokenizer,
            TrainingStage::Language,
            128,
            42,
            &samples(&tokenizer),
        )
        .unwrap();
        let mut bytes = std::fs::read(fixture.path()).unwrap();
        bytes.truncate(bytes.len() - 4);
        std::fs::write(fixture.path(), &bytes).unwrap();
        assert!(load_cache(&fixture.path(), &tokenizer, TrainingStage::Language, 128).is_err());
        std::fs::write(fixture.path(), b"NOTCACHE").unwrap();
        assert!(read_metadata(&fixture.path()).is_err());
        let header = serde_json::to_vec(&CacheMetadata {
            version: 99,
            objective: CacheObjective::PromptReply,
            corpus_sha256: None,
            preparation: None,
            stage: TrainingStage::Language,
            max_seq_len: 128,
            tokenizer_sha256: String::new(),
            samples: 1,
            seed: 42,
        })
        .unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend((header.len() as u32).to_le_bytes());
        bytes.extend(header);
        std::fs::write(fixture.path(), bytes).unwrap();
        assert!(
            read_metadata(&fixture.path())
                .unwrap_err()
                .to_string()
                .contains("version")
        );
    }

    #[test]
    fn failed_write_leaves_no_partial_cache() {
        let fixture = Fixture::new();
        let tokenizer = tokenizer();
        let mut samples = samples(&tokenizer);
        samples[0].input_ids[0] = tokenizer.vocab_size();
        assert!(
            write_cache(
                &fixture.path(),
                &tokenizer,
                TrainingStage::Language,
                128,
                42,
                &samples
            )
            .is_err()
        );
        assert!(!fixture.path().exists());
        assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 0);
    }

    #[test]
    fn bpe_fingerprint_survives_saving_and_reloading_the_tokenizer() {
        use crate::brain::bpe::BpeTokenizer;
        let fixture = Fixture::new();
        let corpus: Vec<String> = std::iter::repeat_n(
            "Tell me about the moon. The moon orbits the Earth.".to_string(),
            20,
        )
        .collect();
        let bpe = BpeTokenizer::train(corpus.iter().collect(), 300).unwrap();
        bpe.save(fixture.0.to_str().unwrap()).unwrap();
        let original = TokenizerKind::Bpe(bpe);
        let restored = TokenizerKind::Bpe(BpeTokenizer::load(fixture.0.to_str().unwrap()).unwrap());
        assert_eq!(
            tokenizer_fingerprint(&original).unwrap(),
            tokenizer_fingerprint(&restored).unwrap()
        );
        let samples = samples(&original);
        write_cache(
            &fixture.path(),
            &original,
            TrainingStage::Language,
            128,
            42,
            &samples,
        )
        .unwrap();
        let loaded = load_cache(&fixture.path(), &restored, TrainingStage::Language, 128).unwrap();
        assert_eq!(format!("{loaded:?}"), format!("{samples:?}"));
    }
}
