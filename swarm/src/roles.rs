//! Role configs: which hole kinds a model owns, how its prompt is cropped and how
//! many output rows it keeps. The role set is data (`roles/*.json`), not code.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub kinds: Vec<String>,
    /// Prompt tokens kept around the target hole (the base context is 512).
    #[serde(default = "default_crop")]
    pub crop_tokens: usize,
    /// Output rows of the tied table this role may write (includes EOS and markers).
    #[serde(default = "default_vocab")]
    pub out_vocab: usize,
    /// Filled by written code instead of a model.
    #[serde(default)]
    pub written: bool,
    /// Own model shape, or another checkpoint. Absent = fine-tune the set's `base`.
    #[serde(default)]
    pub model: Option<ModelSpec>,
    /// Training overrides (the set's `train` supplies the rest).
    #[serde(default)]
    pub train: TrainSpec,
}

/// A role's own network. With `checkpoint` it is fine-tuned from there; without, it is
/// initialised from scratch at this shape (the swarm vocab, tied table, `ffn:16` JL map
/// like the base). Roles may differ in size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    #[serde(default)]
    pub checkpoint: Option<String>,
    #[serde(default = "d_embed")]
    pub embed_dim: usize,
    #[serde(default = "d_layers")]
    pub n_layers: usize,
    #[serde(default = "d_heads")]
    pub attn_heads: usize,
    #[serde(default = "d_ff")]
    pub ff_dim: usize,
    #[serde(default = "d_tok")]
    pub token_embed_dim: usize,
    /// Frozen JL map on every MLP input (0 = none).
    #[serde(default = "d_jl")]
    pub jl_ffn: usize,
}
fn d_embed() -> usize { 32 }
fn d_layers() -> usize { 4 }
fn d_heads() -> usize { 2 }
fn d_ff() -> usize { 128 }
fn d_tok() -> usize { 64 }
fn d_jl() -> usize { 16 }

/// Optimiser settings; every field is optional so a role can override one.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainSpec {
    pub epochs: Option<usize>,
    pub batch: Option<usize>,
    pub first_lr: Option<f64>,
    pub last_lr: Option<f64>,
    pub weight_decay: Option<f64>,
    /// Cap on training samples per epoch (random subset each epoch), for quick runs.
    pub max_samples: Option<usize>,
}

fn default_crop() -> usize {
    448
}
fn default_vocab() -> usize {
    256
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleSet {
    pub name: String,
    /// Checkpoint every role without its own `model` starts from.
    #[serde(default = "default_base")]
    pub base: String,
    /// Defaults for roles' `train` blocks.
    #[serde(default)]
    pub train: TrainSpec,
    pub roles: Vec<Role>,
}

fn default_base() -> String {
    "../checkpoints/yumon-code-dec-sugar-maniac/code_32h_4l_2a_128ff_512len_b16_DecoderOnly_f32_curriculum_documented_tiers_v3_tok64_jl_ffn16_at0_tied".into()
}

impl RoleSet {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let set: RoleSet = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        set.validate()?;
        Ok(set)
    }

    /// Every extracted kind belongs to exactly one role.
    pub fn validate(&self) -> Result<()> {
        let mut seen: HashMap<&str, &str> = HashMap::new();
        for r in &self.roles {
            ensure!(r.out_vocab >= 8, "role {} out_vocab too small", r.name);
            for k in &r.kinds {
                ensure!(crate::skeleton::KINDS.contains(&k.as_str()), "unknown kind {k} in role {}", r.name);
                if let Some(prev) = seen.insert(k, &r.name) {
                    anyhow::bail!("kind {k} is in both {prev} and {}", r.name);
                }
            }
        }
        for k in crate::skeleton::KINDS {
            ensure!(seen.contains_key(k), "kind {k} has no role in set {}", self.name);
        }
        Ok(())
    }

    pub fn role(&self, name: &str) -> Option<&Role> {
        self.roles.iter().find(|r| r.name == name)
    }

    pub fn role_of(&self, kind: &str) -> Option<&Role> {
        self.roles.iter().find(|r| r.kinds.iter().any(|k| k == kind))
    }
}

/// Training settings after merging CLI > role > set > built-in defaults.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub epochs: usize,
    pub batch: usize,
    pub first_lr: f64,
    pub last_lr: f64,
    pub weight_decay: f64,
    pub max_samples: Option<usize>,
}

impl RoleSet {
    pub fn resolve(&self, role: &Role) -> Resolved {
        let pick = |f: fn(&TrainSpec) -> Option<f64>, d: f64| f(&role.train).or(f(&self.train)).unwrap_or(d);
        Resolved {
            epochs: role.train.epochs.or(self.train.epochs).unwrap_or(3),
            batch: role.train.batch.or(self.train.batch).unwrap_or(16),
            first_lr: pick(|t| t.first_lr, 5e-4),
            last_lr: pick(|t| t.last_lr, 5e-5),
            weight_decay: pick(|t| t.weight_decay, 0.01),
            max_samples: role.train.max_samples.or(self.train.max_samples),
        }
    }
}
