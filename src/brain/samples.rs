// ── Types ──────────────────────────────────────────────────────────────────────

use std::collections::{HashMap, HashSet};
use rand::{Rng, rngs::StdRng};
use rayon::prelude::*;
use crate::brain::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, bpe::TokenizerKind, keywords::extract_keywords, train::{MAX_SEQ_LEN, keyword_emote_label, matched_classes}};
use rand::SeedableRng;
use rand::prelude::SliceRandom;

#[derive(Clone, Copy, Debug)]
pub enum TrainingStage {
    Language,   // phase 1: plain sentence → sentence
    Structured, // phase 2: sentence → JSON
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CardinalDir {
    North, South, East, West, None,
}

impl CardinalDir {
    pub fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::East  => Self::West,
            Self::West  => Self::East,
            Self::None  => Self::None,
        }
    }

    /// One-hot [N, S, E, W] — zero vec when None
    pub fn to_onehot(self) -> [f32; 4] {
        match self {
            Self::North => [1., 0., 0., 0.],
            Self::South => [0., 1., 0., 0.],
            Self::East  => [0., 0., 1., 0.],
            Self::West  => [0., 0., 0., 1.],
            Self::None  => [0., 0., 0., 0.],
        }
    }

    pub fn random(rng: &mut impl rand::Rng) -> Self {
        match rng.gen_range(0..4u8) {
            0 => Self::North, 1 => Self::South,
            2 => Self::East,  _ => Self::West,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Speak, Build, Travel, Idle, Use,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Speak  => "speak",
            Self::Build  => "build",
            Self::Travel => "travel",
            Self::Idle   => "idle",
            Self::Use    => "use",
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct WorldEntity {
    pub dir:  CardinalDir,
    pub dist: f32,          // normalised 0..1
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WorldContext {
    pub obstacle: Option<WorldEntity>,
    pub resource: Option<WorldEntity>,
    pub building: Option<WorldEntity>,
}

impl WorldContext {
    /// Flat 18-float slice: [obstacle_onehot:4, dist:1, resource…, building…]
    pub fn to_context_slice(&self) -> [f32; 18] {
        let mut out = [0f32; 18];
        let encode = |entity: &Option<WorldEntity>, buf: &mut [f32]| {
            if let Some(e) = entity {
                buf[..4].copy_from_slice(&e.dir.to_onehot());
                buf[4] = e.dist;
            }
            // zeros already if None
        };
        encode(&self.obstacle, &mut out[0..5]);
        encode(&self.resource, &mut out[6..11]);  // intentional gap → 6 floats per entity
        encode(&self.building, &mut out[12..17]);
        out
    }

    pub fn random(rng: &mut impl rand::Rng) -> Self {
        let mut rng = &mut rand::rngs::StdRng::from_entropy();
        let maybe = |rng: &mut StdRng| -> Option<WorldEntity> {
            if rng.gen_bool(0.4) {
                Some(WorldEntity {
                    dir:  CardinalDir::random(rng),
                    dist: rng.gen_range(0.05f32..1.0),
                })
            } else {
                None
            }
        };
        Self {
            obstacle: maybe(rng),
            resource: maybe(rng),
            building: maybe(rng),
        }
    }
}

// ── Action derivation ──────────────────────────────────────────────────────────

pub fn derive_action(world: &WorldContext) -> (Action, CardinalDir) {
    // Priority order: build > use > navigate > speak/idle
    if let Some(b) = &world.building {
        if b.dist < 0.2 {
            return (Action::Build, CardinalDir::None);
        }
        // Building exists but not close — travel toward it
        return (Action::Travel, b.dir);
    }
    if let Some(r) = &world.resource {
        if r.dist < 0.25 {
            return (Action::Use, CardinalDir::None);
        }
        // Resource exists but not close — travel toward it unless blocked
        if let Some(obs) = &world.obstacle {
            if obs.dist < 0.3 && obs.dir == r.dir {
                // Obstacle in same direction as resource — go around
                return (Action::Travel, obs.dir.opposite());
            }
        }
        return (Action::Travel, r.dir);
    }
    if let Some(obs) = &world.obstacle {
        if obs.dist < 0.3 {
            return (Action::Travel, obs.dir.opposite());
        }
    }
    (Action::Speak, CardinalDir::None)
}

// ── Sample ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Sample {
    pub input_ids:       Vec<usize>,
    pub target_labels:   Vec<usize>,
    pub action:          Action,
    pub motion_dir:      CardinalDir,
    pub world:           WorldContext,
    pub target_json: String,
}

// ── prepare_samples ────────────────────────────────────────────────────────────

pub fn prepare_paired_samples_singles(
    // sentences:    &[String],
    sentences:  Vec<String>,
    tokenizer:    &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
    rng:          &mut impl Rng,
    stage:        TrainingStage,
) -> Vec<Sample> {
    println!("prepare paired samples split");

    let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
    let mut rng_local = rand::thread_rng();
    let mut samples = Vec::new();

    for sent in sentences {
        if bad_words.iter().any(|&w| sent.to_lowercase().contains(w)) { continue; }

        let words: Vec<&str> = sent.split_whitespace().collect();
        if words.len() < 4 { continue; }

        let world = WorldContext::random(&mut rng_local);
        let (action, motion_dir) = derive_action(&world);

        // let input_encoded  = tokenizer.encode(&sent_a);

        let mut obstacle_dir = if let Some(obst) = world.obstacle {
            obst.dir
        } else {
            CardinalDir::None
        };
        let mut building_dir = if let Some(obst) = world.building {
            obst.dir
        } else {
            CardinalDir::None
        };
        let mut resource_dir = if let Some(obst) = world.resource {
            obst.dir
        } else {
            CardinalDir::None
        };
        
        let (input_encoded, input_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent);
                let json    = sent.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "obstacle_dir":     format!("{:?}", obstacle_dir).to_lowercase(),
                    "building_dir":     format!("{:?}", building_dir).to_lowercase(),
                    "resource_dir":     format!("{:?}", resource_dir).to_lowercase(),
                    "message":           sent,
                })).unwrap();
                let encoded = match tokenizer {
                    TokenizerKind::Bpe(b) => b.encode_raw(&json)
                        .unwrap_or_default()
                        .into_iter().map(|x| x as usize).collect(),
                    TokenizerKind::Char(c) => c.encode(&json),
                };
                (encoded, json)
            }
        };

        let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(input_encoded.iter().cloned())
            .collect();

        if enc_input.len() > MAX_SEQ_LEN { continue; }
        if enc_input.len() < MAX_SEQ_LEN / 2 { continue; }

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(MAX_SEQ_LEN, PAD_TOKEN);
            v
        };

        let input_ids     = pad(enc_input);

        let mut target: Vec<usize> = input_encoded.clone();
        target.push(EOS_TOKEN);
        let target_labels = pad(target);

        samples.push(Sample {
            input_ids,
            target_labels,
            action,
            motion_dir,
            world,
            target_json: String::new(),
        });
    }

    samples
}

pub fn prepare_paired_samples_split(
    // sentences:    &[String],
    sentences:  Vec<String>,
    tokenizer:    &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
    rng:          &mut impl Rng,
    stage:        TrainingStage,
) -> Vec<Sample> {
    println!("prepare paired samples split");

    let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
    let mut rng_local = rand::thread_rng();
    let mut samples = Vec::new();

    for sent in sentences {
        if bad_words.iter().any(|&w| sent.to_lowercase().contains(w)) { continue; }

        let words: Vec<&str> = sent.split_whitespace().collect();
        if words.len() < 3 { continue; }

        let world = WorldContext::random(&mut rng_local);
        let (action, motion_dir) = derive_action(&world);

        // let point = words.len() / 4;
        let point = words.len() / 2;
        let sent_a = words[..point].join(" ");
        let sent_b = words[point..].join(" ");

        // let input_encoded  = tokenizer.encode(&sent_a);

        let mut obstacle_dir = if let Some(obst) = world.obstacle {
            obst.dir
        } else {
            CardinalDir::None
        };
        let mut building_dir = if let Some(obst) = world.building {
            obst.dir
        } else {
            CardinalDir::None
        };
        let mut resource_dir = if let Some(obst) = world.resource {
            obst.dir
        } else {
            CardinalDir::None
        };
        
        let (input_encoded, input_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_a);
                let json    = sent_a.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "obstacle_dir":     format!("{:?}", obstacle_dir).to_lowercase(),
                    "building_dir":     format!("{:?}", building_dir).to_lowercase(),
                    "resource_dir":     format!("{:?}", resource_dir).to_lowercase(),
                    "message":           sent_a,
                })).unwrap();
                let encoded = match tokenizer {
                    TokenizerKind::Bpe(b) => b.encode_raw(&json)
                        .unwrap_or_default()
                        .into_iter().map(|x| x as usize).collect(),
                    TokenizerKind::Char(c) => c.encode(&json),
                };
                (encoded, json)
            }
        };

        let sent_b_for_target = sent_b.clone();

        let (target_encoded, target_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_b_for_target);
                let json    = sent_b_for_target.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "action":     action.as_str(),
                    "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
                    "reply":      sent_b_for_target,
                })).unwrap();
                let encoded = match tokenizer {
                    TokenizerKind::Bpe(b) => b.encode_raw(&json)
                        .unwrap_or_default()
                        .into_iter().map(|x| x as usize).collect(),
                    TokenizerKind::Char(c) => c.encode(&json),
                };
                (encoded, json)
            }
        };

        if target_encoded.is_empty() || target_encoded.len() > MAX_SEQ_LEN - 2 { continue; }

        let target_labels: Vec<usize> = target_encoded.iter().cloned()
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(input_encoded.iter().cloned())
            .collect();

        if enc_input.len() > MAX_SEQ_LEN { continue; }

        if enc_input.len() + target_encoded.len() > MAX_SEQ_LEN { continue; }
        if enc_input.len() + target_encoded.len() < MAX_SEQ_LEN / 6 { continue; }

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(MAX_SEQ_LEN, PAD_TOKEN);
            v
        };

        let input_ids     = pad(enc_input);
        let target_labels = pad(target_labels);

        // // Combine into one sequence
        // let full_input: Vec<usize> = std::iter::once(BOS_TOKEN)
        //     .chain(input_encoded.iter().cloned())
        //     .chain(target_encoded.iter().cloned())
        //     .chain(std::iter::once(EOS_TOKEN))
        //     .collect();

        // if full_input.len() > MAX_SEQ_LEN { continue; }
        // if full_input.len() < MAX_SEQ_LEN / 2 { continue; }

        // // target_labels is full_input shifted left by 1 (drop BOS, add trailing PAD)
        // let mut target_labels: Vec<usize> = full_input[1..].to_vec();
        // target_labels.resize(MAX_SEQ_LEN, PAD_TOKEN);

        // let input_ids = pad(full_input);

        samples.push(Sample {
            input_ids,
            target_labels,
            action,
            motion_dir,
            world,
            target_json,
        });
    }

    samples
}

pub fn prepare_paired_samples_split_sep(
    // sentences:    &[String],
    sentences:  Vec<(String, String)>,
    tokenizer:    &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
    rng:          &mut impl Rng,
    stage:        TrainingStage,
) -> Vec<Sample> {
    println!("prepare paired samples split");

    let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
    let mut rng_local = rand::thread_rng();
    let mut samples = Vec::new();

    for sents in sentences {
        let sent_a = sents.0;
        let sent_b = sents.1;

        if bad_words.iter().any(|&w| sent_a.to_lowercase().contains(w)) { continue; }
        if bad_words.iter().any(|&w| sent_b.to_lowercase().contains(w)) { continue; }

        let world = WorldContext::random(&mut rng_local);
        let (action, motion_dir) = derive_action(&world);

        let mut obstacle_dir = if let Some(obst) = world.obstacle {
            obst.dir
        } else {
            CardinalDir::None
        };
        let mut building_dir = if let Some(obst) = world.building {
            obst.dir
        } else {
            CardinalDir::None
        };
        let mut resource_dir = if let Some(obst) = world.resource {
            obst.dir
        } else {
            CardinalDir::None
        };
        
        let (input_encoded, input_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_a);
                let json    = sent_a.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "obstacle_dir":     format!("{:?}", obstacle_dir).to_lowercase(),
                    "building_dir":     format!("{:?}", building_dir).to_lowercase(),
                    "resource_dir":     format!("{:?}", resource_dir).to_lowercase(),
                    "message":           sent_a,
                })).unwrap();
                let encoded = match tokenizer {
                    TokenizerKind::Bpe(b) => b.encode_raw(&json)
                        .unwrap_or_default()
                        .into_iter().map(|x| x as usize).collect(),
                    TokenizerKind::Char(c) => c.encode(&json),
                };
                (encoded, json)
            }
        };

        let sent_b_for_target = sent_b.clone();

        let (target_encoded, target_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_b_for_target);
                let json    = sent_b_for_target.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "action":     action.as_str(),
                    "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
                    "reply":      sent_b_for_target,
                })).unwrap();
                let encoded = match tokenizer {
                    TokenizerKind::Bpe(b) => b.encode_raw(&json)
                        .unwrap_or_default()
                        .into_iter().map(|x| x as usize).collect(),
                    TokenizerKind::Char(c) => c.encode(&json),
                };
                (encoded, json)
            }
        };

        if target_encoded.is_empty() || target_encoded.len() > MAX_SEQ_LEN - 2 { continue; }

        let target_labels: Vec<usize> = target_encoded.iter().cloned()
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(input_encoded.iter().cloned())
            .collect();

        if enc_input.len() > MAX_SEQ_LEN { continue; }

        if enc_input.len() + target_encoded.len() > MAX_SEQ_LEN { continue; }
        if enc_input.len() + target_encoded.len() < MAX_SEQ_LEN / 6 { continue; }

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(MAX_SEQ_LEN, PAD_TOKEN);
            v
        };

        let input_ids     = pad(enc_input);
        let target_labels = pad(target_labels);

        // // Combine into one sequence
        // let full_input: Vec<usize> = std::iter::once(BOS_TOKEN)
        //     .chain(input_encoded.iter().cloned())
        //     .chain(target_encoded.iter().cloned())
        //     .chain(std::iter::once(EOS_TOKEN))
        //     .collect();

        // if full_input.len() > MAX_SEQ_LEN { continue; }
        // if full_input.len() < MAX_SEQ_LEN / 2 { continue; }

        // // target_labels is full_input shifted left by 1 (drop BOS, add trailing PAD)
        // let mut target_labels: Vec<usize> = full_input[1..].to_vec();
        // target_labels.resize(MAX_SEQ_LEN, PAD_TOKEN);

        // let input_ids = pad(full_input);

        samples.push(Sample {
            input_ids,
            target_labels,
            action,
            motion_dir,
            world,
            target_json,
        });
    }

    samples
}