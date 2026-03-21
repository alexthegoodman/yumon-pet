// ── Types ──────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use rand::{Rng, rngs::StdRng};
use crate::brain::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, bpe::TokenizerKind, train::{MAX_SEQ_LEN, keyword_emote_label, matched_classes}};
use rand::SeedableRng;

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

#[derive(Debug, Clone)]
pub struct WorldEntity {
    pub dir:  CardinalDir,
    pub dist: f32,          // normalised 0..1
}

#[derive(Debug, Clone, Default)]
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

pub struct Sample {
    pub input_ids:       Vec<usize>,
    pub target_labels:   Vec<usize>,
    pub context_vec:     Vec<f32>,      // replaces the old flat ctx built in forward()
    pub emote_label:     usize,
    pub matched_classes: Vec<usize>,
    pub action:          Action,
    pub motion_dir:      CardinalDir,
    pub world:           WorldContext,
    pub target_json: String,
}

// ── prepare_samples ────────────────────────────────────────────────────────────

pub fn prepare_samples(
    sentences:     &[String],
    tokenizer:     &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
) -> Vec<Sample> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut samples = Vec::new();

    for sentence in sentences {
        // The reply portion is just the sentence itself —
        // the model learns to echo/continue it as natural speech
        let (action, motion_dir) = derive_action(&WorldContext::random(&mut rng));
        let world = WorldContext::random(&mut rng);

        // Build the JSON target string
        // let target_json = serde_json::json!({
        //     "action":     action.as_str(),
        //     "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
        //     "reply":      sentence,
        // })
        // .to_string();
        let target_json = serde_json::to_string_pretty(&serde_json::json!({
            "action":     action.as_str(),
            "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
            "reply":      sentence,
        })).unwrap();

        // Tokenize both input (the sentence) and the JSON target
        let input_encoded  = tokenizer.encode(sentence);
        let target_encoded = tokenizer.encode(&target_json.clone());

        // Length guard on the target (that's what the model must generate)
        if target_encoded.len() < (MAX_SEQ_LEN - 30)
            || target_encoded.len() > MAX_SEQ_LEN
        {
            continue;
        }

        let input_ids: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(input_encoded.iter().cloned().take(MAX_SEQ_LEN - 1))
            .collect();

        let target_labels: Vec<usize> = target_encoded
            .iter()
            .cloned()
            .take(MAX_SEQ_LEN - 1)
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(MAX_SEQ_LEN, PAD_TOKEN);
            v
        };

        // Build the 132-float context vector:
        //   [class_probs:100][user_emote_probs:7][user_emote_onehot:7][world:18]
        let matched   = matched_classes(sentence, keyword_index);
        let emote_lbl = keyword_emote_label(sentence);

        let mut ctx = Vec::with_capacity(132);
        // class_probs: uniform over matched classes, else near-uniform
        let mut class_probs = vec![1.0f32 / 100.0; 100];
        for &c in &matched {
            class_probs[c] = 0.5;
        }
        ctx.extend_from_slice(&class_probs);
        // user_emote_probs: one-hot at emote_lbl
        let mut emote_probs = vec![0.0f32; 7];
        emote_probs[emote_lbl.min(6)] = 1.0;
        ctx.extend_from_slice(&emote_probs);
        // user_emote_onehot (same for training)
        ctx.extend_from_slice(&emote_probs);
        // world spatial context
        ctx.extend_from_slice(&world.to_context_slice());

        debug_assert_eq!(ctx.len(), 132);

        samples.push(Sample {
            input_ids:       pad(input_ids),
            target_labels:   pad(target_labels),
            context_vec:     ctx,
            emote_label:     emote_lbl,
            matched_classes: matched,
            action,
            motion_dir,
            world,
            target_json
        });
    }

    samples
}