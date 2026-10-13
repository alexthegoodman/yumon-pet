//! Plain append-only tier files that training reads directly:
//! `<root>/documented/<tier>.bin` holds code with its doc comment and
//! `<root>/undocumented/<tier>.bin` the same code without it. A record is the
//! source file, location and text; there are no hashes or positions to keep in sync,
//! and everything is tokenized at load time, so a new tokenizer needs no rebuild.
use super::{
    bpe::TokenizerKind,
    code_corpus::CodeConfig,
    code_documented::{self, OversizedSample},
    samples::{Action, CardinalDir, Sample, WorldContext},
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

/// Tiers used for training. High, excluded and unscored code is never stored.
pub const TIERS: [&str; 3] = ["tier_1_low", "tier_2_mild", "tier_3_moderate"];
const MAGIC: &[u8; 4] = b"YCD1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Documented,
    Undocumented,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub path: String,
    pub location: String,
    pub text: String,
}

pub fn file(root: &Path, kind: Kind, tier: usize) -> PathBuf {
    let dir = match kind {
        Kind::Documented => "documented",
        Kind::Undocumented => "undocumented",
    };
    root.join(dir).join(format!("{}.bin", TIERS[tier]))
}

/// Records plus the byte offset where each one ends.
pub struct Scan {
    pub records: Vec<Record>,
    pub ends: Vec<u64>,
}

/// A missing file is empty. An incomplete final record (an interrupted append)
/// is ignored; any other damage is an error.
pub fn scan(path: &Path) -> Result<Scan> {
    let mut scan = Scan { records: Vec::new(), ends: Vec::new() };
    if !path.exists() {
        return Ok(scan);
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() >= MAGIC.len() && &bytes[..MAGIC.len()] == MAGIC,
        "{} is not a Yumon Code data file",
        path.display()
    );
    fn field(bytes: &[u8], pos: &mut usize) -> Result<Option<String>> {
        let Some(len) = bytes.get(*pos..*pos + 4) else { return Ok(None) };
        let len = u32::from_le_bytes(len.try_into().unwrap()) as usize;
        let Some(body) = bytes.get(*pos + 4..*pos + 4 + len) else { return Ok(None) };
        *pos += 4 + len;
        Ok(Some(String::from_utf8(body.to_vec()).context("record is not UTF-8")?))
    }
    let mut pos = MAGIC.len();
    loop {
        let mut next = pos;
        let (Some(path_), Some(location), Some(text)) = (
            field(&bytes, &mut next)?,
            field(&bytes, &mut next)?,
            field(&bytes, &mut next)?,
        ) else {
            break;
        };
        pos = next;
        scan.records.push(Record { path: path_, location, text });
        scan.ends.push(pos as u64);
    }
    Ok(scan)
}

pub struct Writer {
    file: File,
}

impl Writer {
    /// Creates the file if needed and drops an interrupted trailing record.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let valid = scan(path)?.ends.last().copied().unwrap_or(MAGIC.len() as u64);
        let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(false).open(path)?;
        if file.metadata()?.len() == 0 {
            file.write_all(MAGIC)?;
        } else if file.metadata()?.len() > valid {
            file.set_len(valid)?;
        }
        let mut file = OpenOptions::new().append(true).open(path)?;
        file.flush()?;
        Ok(Self { file })
    }

    /// Written and synced before returning, so a stop never loses an acknowledged record.
    pub fn append(&mut self, record: &Record) -> Result<()> {
        let mut buffer = Vec::with_capacity(record.text.len() + 64);
        for field in [&record.path, &record.location, &record.text] {
            buffer.extend_from_slice(&(field.len() as u32).to_le_bytes());
            buffer.extend_from_slice(field.as_bytes());
        }
        self.file.write_all(&buffer)?;
        self.file.sync_data()?;
        Ok(())
    }
}

/// Drop everything after the first `keep` records.
pub fn keep_first(path: &Path, keep: usize) -> Result<()> {
    let scan = scan(path)?;
    if scan.records.len() > keep {
        let end = if keep == 0 { MAGIC.len() as u64 } else { scan.ends[keep - 1] };
        OpenOptions::new().write(true).open(path)?.set_len(end)?;
    }
    Ok(())
}

/// The two folders grow in step: one comment-bearing sample and one plain sample
/// per append. A stop between the two appends can leave the plain file one record
/// longer; trim it back. Returns (documented, plain) counts. A plain file that is
/// shorter is the caller's to top up.
pub fn reconcile(root: &Path, tier: usize) -> Result<(usize, usize)> {
    let documented = scan(&file(root, Kind::Documented, tier))?.records.len();
    let plain_file = file(root, Kind::Undocumented, tier);
    let plain = scan(&plain_file)?.records.len();
    keep_first(&plain_file, documented)?;
    Ok((documented, plain.min(documented)))
}

/// Files held out of training, chosen from every stored file path so the split
/// does not depend on the tokenizer or the context length.
pub fn held_out_files<'a>(paths: impl Iterator<Item = &'a str>, config: &CodeConfig) -> Result<HashSet<String>> {
    super::code_corpus::validation_file_set(paths.map(String::from).collect(), config)
}

/// None when the text needs more than `context` tokens.
pub fn to_sample(record: &Record, tokenizer: &TokenizerKind, context: usize) -> Result<Option<Sample>> {
    let mut sample = Sample {
        input_ids: Vec::new(),
        target_labels: Vec::new(),
        action: Action::Sit,
        motion_dir: CardinalDir::None,
        world: WorldContext::default(),
        target_json: String::new(),
        pair: (record.path.clone(), record.location.clone()),
    };
    match code_documented::encode(&mut sample, tokenizer, &record.text, context) {
        Ok(()) => Ok(Some(sample)),
        Err(error) if error.is::<OversizedSample>() => Ok(None),
        Err(error) => bail!("{}:{}: {error:#}", record.path, record.location),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(n: usize) -> Record {
        Record { path: format!("a/{n}.rs"), location: format!("line {n}"), text: format!("fn f{n}() {{}}") }
    }

    #[test]
    fn appends_round_trip_and_survive_an_interrupted_write() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("yumon-data-{}", uuid::Uuid::new_v4()));
        let path = dir.join("documented/tier_1_low.bin");
        let mut writer = Writer::open(&path)?;
        for n in 0..3 {
            writer.append(&record(n))?;
        }
        drop(writer);
        // Half a record, as if the process died mid-write.
        let mut raw = OpenOptions::new().append(true).open(&path)?;
        raw.write_all(&[9, 0, 0, 0, b'x'])?;
        drop(raw);
        assert_eq!(scan(&path)?.records, (0..3).map(record).collect::<Vec<_>>());
        let mut writer = Writer::open(&path)?;
        writer.append(&record(3))?;
        assert_eq!(scan(&path)?.records.len(), 4);
        assert_eq!(scan(&path)?.records[3], record(3));
        keep_first(&path, 2)?;
        assert_eq!(scan(&path)?.records, (0..2).map(record).collect::<Vec<_>>());
        assert!(scan(&dir.join("missing.bin"))?.records.is_empty());
        std::fs::write(&path, b"nope")?;
        assert!(scan(&path).is_err());
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn reconcile_trims_extra_plain_records_only() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("yumon-data-{}", uuid::Uuid::new_v4()));
        let mut docs = Writer::open(&file(&dir, Kind::Documented, 0))?;
        let mut plain = Writer::open(&file(&dir, Kind::Undocumented, 0))?;
        for n in 0..3 {
            plain.append(&record(n))?;
        }
        for n in 0..2 {
            docs.append(&record(n))?;
        }
        assert_eq!(reconcile(&dir, 0)?, (2, 2));
        assert_eq!(scan(&file(&dir, Kind::Undocumented, 0))?.records.len(), 2);
        // Fewer plain than documented is left alone for the caller to top up.
        docs.append(&record(2))?;
        docs.append(&record(3))?;
        assert_eq!(reconcile(&dir, 0)?, (4, 2));
        assert_eq!(scan(&file(&dir, Kind::Documented, 0))?.records.len(), 4);
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }
}
