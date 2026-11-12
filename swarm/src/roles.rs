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
    pub roles: Vec<Role>,
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

    pub fn role_of(&self, kind: &str) -> Option<&Role> {
        self.roles.iter().find(|r| r.kinds.iter().any(|k| k == kind))
    }
}
