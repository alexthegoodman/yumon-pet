/// world_augment.rs
///
/// Procedural world-state augmentation for Yumon training samples.
///
/// For each sentence sample this module:
///   1. Rolls a random world state (obstacle, resource, building directions + distances)
///   2. Derives a rule-based action label from that world state
///   3. Derives a motion label (direction index 0–7) consistent with the action
///   4. Encodes the three context tokens and prepends them to input_ids
///
/// The text content of the sentence is intentionally ignored during world-state
/// r#generation — fully random context ensures the model learns to attend to context
/// tokens rather than memorising sentence↔action correlations.
///
/// Output heads this feeds:
///   - action_label  → action choice head (8 classes: see Action enum)
///   - motion_label  → motion direction head (8 classes: N=0, NE=1, … NW=7)
///   - input_ids     → prepended with [obs_tok, res_tok, bld_tok] before BOS

use rand::Rng;
use crate::brain::bpe::{BpeTokenizer, DIRS, ENTITY_OBS, ENTITY_RES, ENTITY_BLD};

// ── Direction ─────────────────────────────────────────────────────────────────

/// Cardinal + intercardinal directions, stored as indices into DIRS.
/// N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dir(pub usize); // 0–7

impl Dir {
    /// Return the direction directly opposite to self.
    pub fn opposite(self) -> Dir {
        Dir((self.0 + 4) % 8)
    }

    pub fn as_str(self) -> &'static str {
        DIRS[self.0]
    }
}

// ── Distance ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dist { Near, Far }

impl Dist {
    pub fn as_str(self) -> &'static str {
        match self { Dist::Near => "near", Dist::Far => "far" }
    }
}

// ── EntityState ───────────────────────────────────────────────────────────────

/// The perceived state of one entity type in Yumon's vicinity.
#[derive(Debug, Clone, Copy)]
pub enum EntityState {
    None,
    Present { dir: Dir, dist: Dist },
}

impl EntityState {
    pub fn is_none(self) -> bool { matches!(self, EntityState::None) }

    pub fn is_near(self) -> bool {
        matches!(self, EntityState::Present { dist: Dist::Near, .. })
    }

    pub fn dir(self) -> Option<Dir> {
        match self { EntityState::Present { dir, .. } => Some(dir), _ => None }
    }
}

// ── WorldState ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct WorldState {
    pub obstacle: EntityState,
    pub resource: EntityState,
    pub building: EntityState,
}

impl WorldState {
    /// r#generate a fully random world state.
    /// Each entity independently has a 30% chance of being absent,
    /// 35% near, 35% far — keeping "none" common enough to teach idle.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        WorldState {
            obstacle: random_entity(rng),
            resource: random_entity(rng),
            building: random_entity(rng),
        }
    }
}

fn random_entity<R: Rng>(rng: &mut R) -> EntityState {
    let roll: f32 = rng.r#gen();
    if roll < 0.30 {
        EntityState::None
    } else {
        let dir  = Dir(rng.r#gen_range(0..8));
        let dist = if rng.r#gen::<f32>() < 0.5 { Dist::Near } else { Dist::Far };
        EntityState::Present { dir, dist }
    }
}

// ── Action ────────────────────────────────────────────────────────────────────

/// All possible Yumon actions. Ordinal = class index for the action head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Action {
    Travel  = 0,
    Build   = 1,
    Collect = 2,
    Speak   = 3,
    Emote   = 4,
    Idle    = 5,
    Wait    = 6,
    // slot 7 reserved for future use (e.g. Inspect)
    _Reserved = 7,
}

impl Action {
    pub fn as_label(self) -> usize { self as usize }
}

// ── Rule-based action derivation ──────────────────────────────────────────────

/// Derive the most appropriate action from the current world state.
///
/// Priority order:
///   1. Obstacle near AND it's blocking the path to a goal → Wait
///   2. Building near → Build
///   3. Resource near → Collect
///   4. Resource present (far) → Travel toward it
///   5. Building present (far) → Travel toward it
///   6. 10% chance → Speak or Emote (social actions, context-independent)
///   7. Default → Idle
pub fn derive_action<R: Rng>(ws: &WorldState, rng: &mut R) -> Action {
    let obs = ws.obstacle;
    let res = ws.resource;
    let bld = ws.building;

    // 1. Wait: obstacle is near AND is blocking the direction toward our goal
    if obs.is_near() {
        let goal_dir = res.dir().or_else(|| bld.dir());
        if let (Some(obs_dir), Some(goal_dir)) = (obs.dir(), goal_dir) {
            if is_blocking(obs_dir, goal_dir) {
                return Action::Wait;
            }
        }
    }

    // 2. Building is near → build it
    if bld.is_near() { return Action::Build; }

    // 3. Resource is near → collect it
    if res.is_near() { return Action::Collect; }

    // 4. Resource exists (far) → travel toward it
    if !res.is_none() { return Action::Travel; }

    // 5. Building exists (far) → travel toward it
    if !bld.is_none() { return Action::Travel; }

    // 6. Small chance of social action regardless of world state
    let social_roll: f32 = rng.r#gen();
    if social_roll < 0.05 { return Action::Speak; }
    if social_roll < 0.10 { return Action::Emote; }

    // 7. Nothing to do
    Action::Idle
}

/// True if an obstacle in `obs_dir` is blocking movement toward `goal_dir`.
/// We consider a ±45° cone in front of the goal direction as "blocking".
fn is_blocking(obs_dir: Dir, goal_dir: Dir) -> bool {
    let diff = (obs_dir.0 + 8 - goal_dir.0) % 8;
    // diff == 0 → head-on, diff == 1 or 7 → one step off (±45°)
    diff == 0 || diff == 1 || diff == 7
}

// ── Motion label derivation ───────────────────────────────────────────────────

/// Derive the motion direction (0–7) consistent with the chosen action.
///
/// - Travel to resource  → motion = resource direction
/// - Travel to building  → motion = building direction
/// - Wait (blocked)      → motion = intended direction (opposite of obstacle),
///                         so the model knows where Yumon *wants* to go
/// - All other actions   → motion = direction of most relevant nearby entity,
///                         or random if none (model learns to ignore it)
pub fn derive_motion<R: Rng>(action: Action, ws: &WorldState, rng: &mut R) -> Dir {
    match action {
        Action::Travel => {
            // prefer resource, fall back to building
            ws.resource.dir()
                .or_else(|| ws.building.dir())
                .unwrap_or_else(|| Dir(rng.r#gen_range(0..8)))
        }
        Action::Wait => {
            // Yumon wants to go toward the goal but is blocked —
            // express intended direction rather than obstacle direction
            ws.resource.dir()
                .or_else(|| ws.building.dir())
                .unwrap_or_else(|| {
                    // no explicit goal: express movement away from obstacle
                    ws.obstacle.dir()
                        .map(|d| d.opposite())
                        .unwrap_or_else(|| Dir(rng.r#gen_range(0..8)))
                })
        }
        Action::Build | Action::Collect => {
            // face the thing being interacted with
            ws.building.dir()
                .or_else(|| ws.resource.dir())
                .unwrap_or_else(|| Dir(rng.r#gen_range(0..8)))
        }
        // Social / idle: motion is noise — random is fine
        _ => Dir(rng.r#gen_range(0..8)),
    }
}

// ── Context token encoding ────────────────────────────────────────────────────

/// Encode the world state into three context token IDs.
/// Returns [obs_tok, res_tok, bld_tok] in that stable order.
pub fn encode_world_state(
    tokenizer: &BpeTokenizer,
    ws: &WorldState,
) -> anyhow::Result<[usize; 3]> {
    let obs_tok = encode_entity(tokenizer, ENTITY_OBS, ws.obstacle)?;
    let res_tok = encode_entity(tokenizer, ENTITY_RES, ws.resource)?;
    let bld_tok = encode_entity(tokenizer, ENTITY_BLD, ws.building)?;
    Ok([obs_tok, res_tok, bld_tok])
}

fn encode_entity(
    tokenizer: &BpeTokenizer,
    entity: &str,
    state: EntityState,
) -> anyhow::Result<usize> {
    let id = match state {
        EntityState::None => tokenizer.encode_context(entity, None, None)?,
        EntityState::Present { dir, dist } =>
            tokenizer.encode_context(entity, Some(dir.as_str()), Some(dist.as_str()))?,
    };
    Ok(id as usize)
}

// ── Top-level augmentation ────────────────────────────────────────────────────

/// Everything needed to augment one sample. Call this inside `prepare_samples`.
#[derive(Debug, Clone)]
pub struct AugmentedContext {
    /// [obs_tok, res_tok, bld_tok] — prepend before BOS in input_ids
    pub context_token_ids: [usize; 3],
    /// Action head target (0–7)
    pub action_label: usize,
    /// Motion head target (0–7)
    pub motion_label: usize,
}

/// r#generate a random world state and derive all labels for one sample.
pub fn augment<R: Rng>(
    tokenizer: &BpeTokenizer,
    rng: &mut R,
) -> anyhow::Result<AugmentedContext> {
    let ws     = WorldState::random(rng);
    let action = derive_action(&ws, rng);
    let motion = derive_motion(action, &ws, rng);

    Ok(AugmentedContext {
        context_token_ids: encode_world_state(tokenizer, &ws)?,
        action_label:      action.as_label(),
        motion_label:      motion.0,
    })
}