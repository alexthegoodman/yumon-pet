//! Phase 3: the item planner. It gets the leading `///` doc comment of a chunk and writes
//! the root skeleton (everything below the doc, with typed holes for names, signatures,
//! fields, variants and bodies). Workers fill those holes; block workers write the
//! statement-level skeletons, so they are the "L2" planners already.
//!
//! Sample = `[BOS] doc lines [<fill>] skeleton [EOS]`, loss on the skeleton only. The
//! doc lines are not part of the target (the glue puts them back in front), but docs on
//! inner items stay in the skeleton text, so the planner invents those.
use crate::{
    dataset::{self, Chunk},
    skeleton::KINDS,
    swarm::{Filler, Request, check_fill},
    worker::{BOS, CONTEXT, EOS, Rows, Sample, VOCAB, Worker, encode},
};
use anyhow::{Result, ensure};
use rayon::prelude::*;
use std::path::Path;
use tokenizers::Tokenizer;

/// Root kind the dispatcher starts from.
pub const ROOT: &str = "item";

/// Leading run of `///` lines of a chunk, newlines included. Empty if there is none.
pub fn doc_prefix(text: &str) -> &str {
    let mut end = 0;
    for line in text.split_inclusive('\n') {
        if line.ends_with('\n') && line.trim_start().starts_with("///") {
            end += line.len();
        } else {
            break;
        }
    }
    &text[..end]
}

/// Turn free text into doc lines: lines already starting with `///` stay, others get it.
pub fn as_doc(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let t = line.trim_start();
        if t.starts_with("///") {
            out.push_str(t);
        } else {
            out.push_str("/// ");
            out.push_str(t);
        }
        out.push('\n');
    }
    out
}

pub struct PlannerData {
    pub rows: Rows,
    pub train: Vec<Sample>,
    pub val: Vec<Sample>,
    /// Chunks with no leading doc comment (not used).
    pub undocumented: usize,
    /// Samples over the 512 context.
    pub too_long: usize,
    /// (doc, skeleton) of the first few training and held-out samples, for `--check`.
    pub examples: Vec<(String, String)>,
}

impl PlannerData {
    pub fn build(chunks: &[Chunk], tok: &Tokenizer, out_vocab: usize) -> Result<PlannerData> {
        let fill_id = encode(tok, crate::skeleton::FILL);
        ensure!(fill_id.len() == 1, "<fill> is not one token; run make_tokenizer");
        let fill_id = fill_id[0];
        struct Raw {
            ids: Vec<u16>,
            prompt_len: usize,
            val: bool,
            doc: String,
            body: String,
        }
        let raws: Vec<Option<Raw>> = chunks
            .par_iter()
            .map(|chunk| {
                let prefix = doc_prefix(&chunk.text);
                if prefix.is_empty() {
                    return None;
                }
                let root = dataset::extract_chunk(chunk).ok()?;
                let skel = root.skeleton();
                let body = skel.strip_prefix(prefix)?;
                let mut ids = vec![BOS];
                ids.extend(encode(tok, prefix));
                ids.push(fill_id);
                let prompt_len = ids.len();
                ids.extend(encode(tok, body));
                ids.push(EOS);
                Some(Raw {
                    ids: ids.iter().map(|&t| t as u16).collect(),
                    prompt_len,
                    val: dataset::is_validation(&chunk.file),
                    doc: prefix.to_owned(),
                    body: body.to_owned(),
                })
            })
            .collect();
        let undocumented = chunks.iter().filter(|c| doc_prefix(&c.text).is_empty()).count();
        let mut hist = vec![0u64; VOCAB];
        for r in raws.iter().flatten().filter(|r| !r.val) {
            for &t in &r.ids[r.prompt_len..] {
                hist[t as usize] += 1;
            }
        }
        let rows = Rows::select(&hist, out_vocab);
        let (mut train, mut val, mut too_long) = (Vec::new(), Vec::new(), 0);
        let mut examples = Vec::new();
        for r in raws.into_iter().flatten() {
            if r.ids.len() - 1 > CONTEXT {
                too_long += 1;
                continue;
            }
            if examples.len() < 3 && !r.val {
                examples.push((r.doc.clone(), r.body.clone()));
            }
            let s = Sample { ids: r.ids, prompt_len: r.prompt_len, kind: 0, in_role: true };
            if r.val { val.push(s) } else { train.push(s) }
        }
        Ok(PlannerData { rows, train, val, undocumented, too_long, examples })
    }
}

/// Does `doc + plan` parse as a whole chunk (holes as `syn` placeholders)?
pub fn plan_parses(doc: &str, plan: &str) -> bool {
    check_fill(ROOT, &format!("{doc}{plan}"))
}

/// Hole kinds of a skeleton, in order (its "shape").
pub fn shape(text: &str) -> Vec<String> {
    crate::swarm::parse_fill(ROOT, text).map(|f| f.children.into_iter().map(|c| c.kind).collect()).unwrap_or_default()
}

pub struct Planner {
    pub worker: Worker,
}

impl Planner {
    pub fn load(dir: &Path) -> Result<Planner> {
        Ok(Planner { worker: Worker::load(dir)? })
    }

    /// The whole root text for these doc lines: the doc itself, then the planned skeleton.
    pub fn plan(&self, doc: &str, attempt: usize) -> Option<String> {
        let body = self.worker.fill_attempt(doc, attempt)?;
        Some(format!("{doc}{body}"))
    }
}

/// Dispatcher filler for a full run: the planner writes the root, workers the rest.
pub struct SwarmFiller<'a> {
    pub planner: &'a Planner,
    pub doc: String,
    pub workers: &'a [Worker],
}

impl<'a> SwarmFiller<'a> {
    pub fn new(planner: &'a Planner, doc: String, workers: &'a [Worker]) -> Self {
        Self { planner, doc, workers }
    }
    fn worker(&self, kind: &str) -> Option<&Worker> {
        self.workers.iter().find(|w| w.handles(kind))
    }
}

impl Filler for SwarmFiller<'_> {
    fn fill(&mut self, req: &Request) -> Option<String> {
        if req.path.is_empty() {
            return self.planner.plan(&self.doc, req.attempt);
        }
        self.worker(req.kind)?.fill_attempt(req.context, req.attempt)
    }
    fn name(&self, kind: &str) -> &str {
        if kind == ROOT { "planner" } else { self.worker(kind).map_or("none", |w| w.meta.role.as_str()) }
    }
}

#[allow(dead_code)]
fn _kinds() -> usize {
    KINDS.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_prefix_takes_leading_doc_lines_only() {
        let t = "/// A.\n  /// B.\nfn f() {}\n/// later\n";
        assert_eq!(doc_prefix(t), "/// A.\n  /// B.\n");
        assert_eq!(doc_prefix("fn f() {}\n"), "");
        assert_eq!(doc_prefix("/// only"), "");
    }

    #[test]
    fn free_text_becomes_doc_lines() {
        assert_eq!(as_doc("Sums rows.\n\n/// Already.\n"), "/// Sums rows.\n/// Already.\n");
    }

    #[test]
    fn skeleton_after_doc_parses_with_placeholders() {
        let src = "/// Sums.\nfn total(rows: &[u32]) -> u32 {\n    rows.iter().sum()\n}\n";
        let root = crate::skeleton::extract(src).unwrap();
        let doc = doc_prefix(src);
        let body = root.skeleton().strip_prefix(doc).unwrap().to_owned();
        assert!(plan_parses(doc, &body), "{body}");
        assert_eq!(shape(&body), ["name", "sig", "block"]);
    }
}
