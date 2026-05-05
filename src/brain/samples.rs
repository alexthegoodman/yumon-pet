// ── Types ──────────────────────────────────────────────────────────────────────

use std::collections::{HashMap, HashSet};
use rand::{Rng, rngs::StdRng};
#[cfg(target_os = "windows")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::brain::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, bpe::TokenizerKind, mdx::HandcraftedChats, sentiment::EmotionAnalyzer, train::{MAX_SEQ_LEN, keyword_emote_label, matched_classes}};
use rand::SeedableRng;
use rand::prelude::SliceRandom;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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

// #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
// #[serde(rename_all = "lowercase")]
// pub enum Action {
//     // Speak, // always speaks silently, in writing 
//     // Build, // be more specific
//     // Travel, // be more specific
//     // Idle, 
//     // Use,
//     GoToDestination, GoHome, Follow, GetHelp, Survey, Collect, Stack, Sit
//     // Play?
// }

// impl Action {
//     pub fn as_str(self) -> &'static str {
//         match self {
//             // Self::Speak  => "speak",
//             // Self::Build  => "build",
//             // Self::Travel => "travel",
//             // Self::Idle   => "idle",
//             // Self::Use    => "use",
//             Self::GoToDestination => "go to destination", Self::GoHome => "go home", 
//             Self::Follow => "follow", Self::GetHelp => "get help",
//             Self::Survey => "survey area", Self::Collect => "collect items", 
//             Self::Stack => "stack items", Self::Sit => "sit"
//         }

//     }
// }

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

// // ── Action derivation ──────────────────────────────────────────────────────────

// pub fn derive_action(world: &WorldContext) -> (Action, CardinalDir) {
//     // Priority order: build > use > navigate > speak/idle
//     if let Some(b) = &world.building {
//         if b.dist < 0.2 {
//             return (Action::Build, CardinalDir::None);
//         }
//         // Building exists but not close — travel toward it
//         return (Action::Travel, b.dir);
//     }
//     if let Some(r) = &world.resource {
//         if r.dist < 0.25 {
//             return (Action::Use, CardinalDir::None);
//         }
//         // Resource exists but not close — travel toward it unless blocked
//         if let Some(obs) = &world.obstacle {
//             if obs.dist < 0.3 && obs.dir == r.dir {
//                 // Obstacle in same direction as resource — go around
//                 return (Action::Travel, obs.dir.opposite());
//             }
//         }
//         return (Action::Travel, r.dir);
//     }
//     if let Some(obs) = &world.obstacle {
//         if obs.dist < 0.3 {
//             return (Action::Travel, obs.dir.opposite());
//         }
//     }
//     (Action::Speak, CardinalDir::None)
// }

// // ── Vocabulary tables ──────────────────────────────────────────────────────────

// // Resource nouns — things an agent picks up / uses
// const RESOURCE_NOUNS: &[&str] = &[
//     "red fruit", "glowing crystal", "supply crate", "herbs", "ore chunk",
//     "fuel cell", "food ration", "strange artifact", "water flask", "mossy stone",
// ];

// // Building nouns — things an agent constructs at / interacts with
// const BUILDING_NOUNS: &[&str] = &[
//     "shelter", "outpost", "campfire", "beacon tower", "storage depot",
//     "watchtower", "workshop", "relay station", "fortification", "trading post",
// ];

// // Obstacle nouns — things an agent avoids / routes around
// const OBSTACLE_NOUNS: &[&str] = &[
//     "boulder", "collapsed wall", "thorny thicket", "broken machine",
//     "deep ravine", "toxic pool", "barricade", "debris field", "downed tree", "rubble pile",
// ];

// // Travel / approach verbs
// const TRAVEL_VERBS: &[(&str, &str)] = &[
//     ("move toward",     "move toward"),
//     ("head to",         "head to"),
//     ("go to",           "go to"),
//     ("navigate to",     "navigate to"),
//     ("approach",        "approach"),
//     ("walk toward",     "walk toward"),
//     ("make your way to","make your way to"),
//     ("advance on",      "advance on"),
// ];

// // Use / collect verbs (for resources)
// const USE_VERBS: &[(&str, &str)] = &[
//     ("pick up",         "pick up"),
//     ("collect",         "collect"),
//     ("grab",            "grab"),
//     ("retrieve",        "retrieve"),
//     ("take",            "take"),
//     ("gather",          "gather"),
//     ("acquire",         "acquire"),
//     ("secure",          "secure"),
// ];

// // Build / interact verbs (for buildings)
// const BUILD_VERBS: &[(&str, &str)] = &[
//     ("build",           "build"),
//     ("construct",       "construct"),
//     ("set up",          "set up"),
//     ("establish",       "establish"),
//     ("work on",         "work on"),
//     ("assemble",        "assemble"),
//     ("reinforce",       "reinforce"),
//     ("complete",        "complete"),
// ];

// // Avoid / evade verbs (for obstacles)
// const AVOID_VERBS: &[(&str, &str)] = &[
//     ("avoid",           "avoid"),
//     ("go around",       "go around"),
//     ("steer clear of",  "steer clear of"),
//     ("route around",    "route around"),
//     ("bypass",          "bypass"),
//     ("detour around",   "detour around"),
//     ("evade",           "evade"),
// ];

// // Idle / speak filler targets when no interesting entity
// const IDLE_TARGETS: &[&str] = &[
//     "the area", "your surroundings", "nearby", "the environment",
//     "the zone", "this location",
// ];

// const IDLE_VERBS: &[(&str, &str)] = &[
//     ("survey",          "survey"),
//     ("scan",            "scan"),
//     ("observe",         "observe"),
//     ("report on",       "report on"),
//     ("assess",          "assess"),
// ];

// // Sentence templates: {verb} and {target} are placeholders
// // Some add directional flavour, some are terse commands, some are phrased as requests
// const TEMPLATES: &[&str] = &[
//     "{verb} the {target}",
//     "please {verb} the {target}",
//     "you should {verb} the {target}",
//     "I need you to {verb} the {target}",
//     "can you {verb} the {target}?",
//     "{verb} the {target} immediately",
//     "your task is to {verb} the {target}",
//     "go and {verb} the {target}",
//     "try to {verb} the {target}",
// ];

// // ── Output types ───────────────────────────────────────────────────────────────

// /// A single procedurally generated natural-language command.
// #[derive(Debug, Clone)]
// pub struct CommandPhrase {
//     /// Full natural-language instruction, e.g. "go and pick up the red fruit"
//     pub command: String,
//     /// Canonical action verb phrase, e.g. "pick up"
//     pub action_str: String,
//     /// Target object description, e.g. "red fruit"
//     pub target: String,
// }

// /// Combined training sample: an optional command paired with the ground-truth
// /// action derived from `derive_action`.
// #[derive(Debug, Clone)]
// pub struct TrainingSample {
//     /// Present ~67 % of the time; absent ~33 % of the time.
//     pub command: Option<CommandPhrase>,
//     /// Ground-truth action from `derive_action`.
//     pub action: Action,
//     /// Ground-truth direction from `derive_action`.
//     pub dir: CardinalDir,
// }

// // ── Core generation ────────────────────────────────────────────────────────────

// fn pick<'a, T>(rng: &mut impl Rng, slice: &'a [T]) -> &'a T {
//     &slice[rng.gen_range(0..slice.len())]
// }

// /// Generate a `CommandPhrase` that is *consistent* with the given action and
// /// direction — i.e. the language instructs the same thing `derive_action`
// /// would decide to do.
// fn generate_phrase(
//     rng: &mut impl Rng,
//     world: &WorldContext,
//     action: Action,
//     _dir: CardinalDir,
// ) -> CommandPhrase {
//     let (verb_display, action_str, target): (&str, &str, String) = match action {
//         Action::Build => {
//             let (v, a) = pick(rng, BUILD_VERBS);
//             let noun = world
//                 .building
//                 .as_ref()
//                 .map(|_| *pick(rng, BUILDING_NOUNS))
//                 .unwrap_or("structure");
//             (v, a, noun.to_string())
//         }
//         Action::Use => {
//             let (v, a) = pick(rng, USE_VERBS);
//             let noun = world
//                 .resource
//                 .as_ref()
//                 .map(|_| *pick(rng, RESOURCE_NOUNS))
//                 .unwrap_or("item");
//             (v, a, noun.to_string())
//         }
//         Action::Travel => {
//             // Figure out *what* we are travelling toward (or away from)
//             if let Some(_b) = &world.building {
//                 // Travelling toward building
//                 let (v, a) = pick(rng, TRAVEL_VERBS);
//                 let noun = pick(rng, BUILDING_NOUNS);
//                 (v, a, noun.to_string())
//             } else if let Some(_r) = &world.resource {
//                 // Could be travelling toward resource OR away from an obstacle
//                 if let Some(_obs) = &world.obstacle {
//                     // Obstacle blocking resource — avoid phrasing
//                     let (v, a) = pick(rng, AVOID_VERBS);
//                     let noun = pick(rng, OBSTACLE_NOUNS);
//                     (v, a, noun.to_string())
//                 } else {
//                     let (v, a) = pick(rng, TRAVEL_VERBS);
//                     let noun = pick(rng, RESOURCE_NOUNS);
//                     (v, a, noun.to_string())
//                 }
//             } else if let Some(_obs) = &world.obstacle {
//                 // Pure avoidance
//                 let (v, a) = pick(rng, AVOID_VERBS);
//                 let noun = pick(rng, OBSTACLE_NOUNS);
//                 (v, a, noun.to_string())
//             } else {
//                 // Fallback — generic travel
//                 let (v, a) = pick(rng, TRAVEL_VERBS);
//                 let noun = pick(rng, IDLE_TARGETS);
//                 (v, a, noun.to_string())
//             }
//         }
//         Action::Speak | Action::Idle => {
//             let (v, a) = pick(rng, IDLE_VERBS);
//             let noun = pick(rng, IDLE_TARGETS);
//             (v, a, noun.to_string())
//         }
//     };

//     let template = pick(rng, TEMPLATES);
//     let command = template
//         .replace("{verb}", verb_display)
//         .replace("{target}", &target);

//     CommandPhrase {
//         command,
//         action_str: action_str.to_string(),
//         target,
//     }
// }

// // ── Public API ─────────────────────────────────────────────────────────────────

// /// Generate a complete `TrainingSample` from a `WorldContext`.
// ///
// /// Internally calls `derive_action` for the ground-truth label, then with
// /// ~67 % probability generates a matching natural-language command phrase.
// pub fn generate_training_sample(rng: &mut impl Rng, world: &WorldContext) -> TrainingSample {
//     let (action, dir) = derive_action(world);

//     // ~33 % chance of no command (agent acts from world-state alone)
//     let command = if rng.gen_bool(0.67) {
//         Some(generate_phrase(rng, world, action, dir))
//     } else {
//         None
//     };

//     TrainingSample { command, action, dir }
// }

// // ── Tests ──────────────────────────────────────────────────────────────────────

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::SeedableRng;

//     fn seeded() -> StdRng {
//         StdRng::seed_from_u64(42)
//     }

//     #[test]
//     fn sample_count_sanity() {
//         let mut rng = seeded();
//         let samples: Vec<_> = (0..300)
//             .map(|_| {
//                 let world = WorldContext::random(&mut rng);
//                 generate_training_sample(&mut rng, &world)
//             })
//             .collect();

//         let with_cmd = samples.iter().filter(|s| s.command.is_some()).count();
//         // Expect roughly 67 % — allow wide band for seed variance
//         assert!(with_cmd > 150 && with_cmd < 250, "got {with_cmd} / 300");
//     }

//     #[test]
//     fn build_action_uses_build_vocab() {
//         let world = WorldContext {
//             obstacle: None,
//             resource: None,
//             building: Some(WorldEntity { dir: CardinalDir::North, dist: 0.05 }),
//         };
//         let mut rng = seeded();
//         // derive_action should return Build
//         let (action, _) = derive_action(&world);
//         assert_eq!(action, Action::Build);

//         let phrase = generate_phrase(&mut rng, &world, action, CardinalDir::None);
//         // action_str must come from BUILD_VERBS
//         assert!(
//             BUILD_VERBS.iter().any(|(_, a)| *a == phrase.action_str),
//             "unexpected action_str: {}", phrase.action_str
//         );
//     }

//     #[test]
//     fn use_action_uses_use_vocab() {
//         let world = WorldContext {
//             obstacle: None,
//             resource: Some(WorldEntity { dir: CardinalDir::East, dist: 0.1 }),
//             building: None,
//         };
//         let mut rng = seeded();
//         let (action, _) = derive_action(&world);
//         assert_eq!(action, Action::Use);

//         let phrase = generate_phrase(&mut rng, &world, action, CardinalDir::None);
//         assert!(
//             USE_VERBS.iter().any(|(_, a)| *a == phrase.action_str),
//             "unexpected action_str: {}", phrase.action_str
//         );
//     }

//     #[test]
//     fn obstacle_avoidance_uses_avoid_vocab() {
//         // Obstacle close, no resource, no building → Travel (away)
//         // But we test the phrase generator directly with Travel + obstacle context
//         let world = WorldContext {
//             obstacle: Some(WorldEntity { dir: CardinalDir::South, dist: 0.1 }),
//             resource: None,
//             building: None,
//         };
//         let mut rng = seeded();
//         let phrase = generate_phrase(&mut rng, &world, Action::Travel, CardinalDir::North);
//         assert!(
//             AVOID_VERBS.iter().any(|(_, a)| *a == phrase.action_str),
//             "unexpected action_str for obstacle avoidance: {}", phrase.action_str
//         );
//     }

//     #[test]
//     fn prints_varied_samples() {
//         let mut rng = seeded();
//         for _ in 0..10 {
//             let world = WorldContext::random(&mut rng);
//             let sample = generate_training_sample(&mut rng, &world);
//             if let Some(c) = &sample.command {
//                 println!(
//                     "[{:?}/{:?}] cmd=\"{}\" | action=\"{}\" | target=\"{}\"",
//                     sample.action, sample.dir, c.command, c.action_str, c.target
//                 );
//             } else {
//                 println!("[{:?}/{:?}] (no command)", sample.action, sample.dir);
//             }
//         }
//     }
// }

// ── Action ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    GoToDestination, GoHome, Follow, GetHelp, Survey, Collect, Stack, Sit,
    // Play?
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GoToDestination => "go to destination",
            Self::GoHome         => "go home",
            Self::Follow         => "follow",
            Self::GetHelp        => "get help",
            Self::Survey         => "survey area",
            Self::Collect        => "collect items",
            Self::Stack          => "stack items",
            Self::Sit            => "sit",
        }
    }
}

// ── Vocabulary tables ──────────────────────────────────────────────────────────

const GO_TO_VERBS:   &[&str] = &["go to", "head to", "travel to", "make your way to", "navigate to", "move toward", "advance on", "proceed to"];
const GO_TO_TARGETS: &[&str] = &["the cave", "the marker", "the ruins", "the camp", "the ridge", "the old bridge", "the waypoint", "the cliffs", "the river crossing", "the summit"];

// GoHome is self-contained — templates drop the {target} slot
const GO_HOME_VERBS: &[&str] = &["go home", "head home", "return home", "head back", "come back", "return to base", "make your way back", "retreat to camp"];

const FOLLOW_VERBS:   &[&str] = &["follow", "stay close to", "keep up with", "trail", "shadow", "stick with", "keep pace with"];
const FOLLOW_TARGETS: &[&str] = &["me", "us", "the group"];

// GetHelp is self-contained, but can optionally include a source
const GET_HELP_VERBS:   &[&str] = &["get help", "find help", "go get assistance", "fetch help", "seek help", "alert someone", "find backup"];
const GET_HELP_TARGETS: &[&str] = &["from the village", "from the outpost", "from the nearest camp", "immediately", "as fast as you can", "now"];

const SURVEY_VERBS:   &[&str] = &["scan", "survey", "look around", "observe", "check out", "scout out", "inspect", "assess"];
const SURVEY_TARGETS: &[&str] = &["the area", "your surroundings", "the perimeter", "nearby", "this location", "the terrain", "the zone", "the environment"];

const COLLECT_VERBS:   &[&str] = &["collect", "gather", "pick up", "retrieve", "grab", "take", "harvest", "secure"];
const COLLECT_TARGETS: &[&str] = &["the herbs", "the berries", "the ore", "those supplies", "the mushrooms", "the wood", "the fruit", "the crystals", "the roots", "that item"];

const STACK_VERBS:   &[&str] = &["stack", "pile up", "organize", "arrange", "sort", "store", "bundle up", "consolidate"];
const STACK_TARGETS: &[&str] = &["the crates", "the wood", "the supplies", "the stones", "the logs", "those items", "the gear", "the provisions"];

// Sit is self-contained but can optionally include a location
const SIT_VERBS:   &[&str] = &["sit", "rest", "wait", "stay", "hold position", "settle down", "sit down", "stay put"];
const SIT_TARGETS: &[&str] = &["here", "nearby", "for now", "and wait", "a moment", "right here"];

// Sentence templates
// {target} is optional — self-contained actions use the targetless set
const TEMPLATES_WITH_TARGET: &[&str] = &[
    "{verb} {target}",
    "please {verb} {target}",
    "you should {verb} {target}",
    "I need you to {verb} {target}",
    "can you {verb} {target}?",
    "{verb} {target} immediately",
    "your task is to {verb} {target}",
    "go and {verb} {target}",
    "try to {verb} {target}",
];

const TEMPLATES_SELF_CONTAINED: &[&str] = &[
    "{verb}",
    "please {verb}",
    "you should {verb}",
    "I need you to {verb}",
    "can you {verb}?",
    "{verb} right now",
    "your task is to {verb}",
    "just {verb}",
];

// ── Output types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CommandPhrase {
    /// Full natural-language instruction, e.g. "gather the herbs"
    pub command: String,
    /// Canonical action verb phrase, e.g. "gather"
    pub action_str: String,
    /// Target object/location, e.g. "the herbs" (empty string if self-contained)
    pub target: String,
}

/// A command is always present — the robot only acts when told.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub command: CommandPhrase,
    pub action:  Action,
}

// ── Core generation ────────────────────────────────────────────────────────────

fn pick<'a, T>(rng: &mut impl Rng, slice: &'a [T]) -> &'a T {
    &slice[rng.gen_range(0..slice.len())]
}

fn generate_phrase(rng: &mut impl Rng, action: Action) -> CommandPhrase {
    // Returns (verb, target) — target is "" for self-contained actions
    let (verb, target): (&str, String) = match action {
        Action::GoToDestination => (*pick(rng, GO_TO_VERBS),   pick(rng, GO_TO_TARGETS).to_string()),
        Action::Follow          => (*pick(rng, FOLLOW_VERBS),  pick(rng, FOLLOW_TARGETS).to_string()),
        Action::GetHelp         => (*pick(rng, GET_HELP_VERBS), pick(rng, GET_HELP_TARGETS).to_string()),
        Action::Survey          => (*pick(rng, SURVEY_VERBS),  pick(rng, SURVEY_TARGETS).to_string()),
        Action::Collect         => (*pick(rng, COLLECT_VERBS), pick(rng, COLLECT_TARGETS).to_string()),
        Action::Stack           => (*pick(rng, STACK_VERBS),   pick(rng, STACK_TARGETS).to_string()),
        // Self-contained — verb already carries full meaning, target is optional flavour
        Action::GoHome          => (*pick(rng, GO_HOME_VERBS),  String::new()),
        Action::Sit             => (*pick(rng, SIT_VERBS),      pick(rng, SIT_TARGETS).to_string()),
    };

    let command = if target.is_empty() {
        let template = pick(rng, TEMPLATES_SELF_CONTAINED);
        template.replace("{verb}", verb)
    } else {
        let template = pick(rng, TEMPLATES_WITH_TARGET);
        template.replace("{verb}", verb).replace("{target}", &target)
    };

    CommandPhrase { command, action_str: verb.to_string(), target }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Pick a random action and generate a matching command phrase.
pub fn generate_training_sample(rng: &mut impl Rng) -> TrainingSample {
    const ALL_ACTIONS: &[Action] = &[
        Action::GoToDestination, Action::GoHome, Action::Follow, Action::GetHelp,
        Action::Survey, Action::Collect, Action::Stack, Action::Sit,
    ];
    let action  = *pick(rng, ALL_ACTIONS);
    let command = generate_phrase(rng, action);
    TrainingSample { command, action }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn seeded() -> StdRng { StdRng::seed_from_u64(42) }

    #[test]
    fn all_actions_produce_commands() {
        let mut rng = seeded();
        let actions = [
            Action::GoToDestination, Action::GoHome, Action::Follow, Action::GetHelp,
            Action::Survey, Action::Collect, Action::Stack, Action::Sit,
        ];
        for action in actions {
            let phrase = generate_phrase(&mut rng, action);
            assert!(!phrase.command.is_empty(), "empty command for {action:?}");
            assert!(!phrase.action_str.is_empty(), "empty action_str for {action:?}");
        }
    }

    #[test]
    fn collect_uses_collect_vocab() {
        let mut rng = seeded();
        let phrase = generate_phrase(&mut rng, Action::Collect);
        assert!(
            COLLECT_VERBS.contains(&phrase.action_str.as_str()),
            "unexpected action_str: {}", phrase.action_str
        );
    }

    #[test]
    fn go_home_is_self_contained() {
        let mut rng = seeded();
        // GoHome should never produce a dangling {target} in the output
        for _ in 0..20 {
            let phrase = generate_phrase(&mut rng, Action::GoHome);
            assert!(!phrase.command.contains("{target}"), "unresolved template: {}", phrase.command);
        }
    }

    #[test]
    fn prints_varied_samples() {
        let mut rng = seeded();
        for _ in 0..16 {
            let sample = generate_training_sample(&mut rng);
            println!(
                "[{:?}] cmd=\"{}\" | verb=\"{}\" | target=\"{}\"",
                sample.action, sample.command.command,
                sample.command.action_str, sample.command.target
            );
        }
    }
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
    pub pair: (String, String),
}

// ── prepare_samples ────────────────────────────────────────────────────────────

// pub fn prepare_paired_samples_singles(
//     // sentences:    &[String],
//     sentences:  Vec<String>,
//     tokenizer:    &TokenizerKind,
//     keyword_index: &HashMap<String, Vec<usize>>,
//     rng:          &mut impl Rng,
//     stage:        TrainingStage,
// ) -> Vec<Sample> {
//     println!("prepare paired samples split");

//     let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
//     let mut rng_local = rand::thread_rng();
//     let mut samples = Vec::new();

//     for sent in sentences {
//         if bad_words.iter().any(|&w| sent.to_lowercase().contains(w)) { continue; }

//         let words: Vec<&str> = sent.split_whitespace().collect();
//         if words.len() < 4 { continue; }

//         let world = WorldContext::random(&mut rng_local);
//         let (action, motion_dir) = derive_action(&world);

//         // let input_encoded  = tokenizer.encode(&sent_a);

//         let mut obstacle_dir = if let Some(obst) = world.obstacle {
//             obst.dir
//         } else {
//             CardinalDir::None
//         };
//         let mut building_dir = if let Some(obst) = world.building {
//             obst.dir
//         } else {
//             CardinalDir::None
//         };
//         let mut resource_dir = if let Some(obst) = world.resource {
//             obst.dir
//         } else {
//             CardinalDir::None
//         };
        
//         let (input_encoded, input_json) = match stage {
//             TrainingStage::Language => {
//                 let encoded = tokenizer.encode(&sent);
//                 let json    = sent.clone();
//                 (encoded, json)
//             }
//             TrainingStage::Structured => {
//                 let json = serde_json::to_string_pretty(&serde_json::json!({
//                     "obstacle_dir":     format!("{:?}", obstacle_dir).to_lowercase(),
//                     "building_dir":     format!("{:?}", building_dir).to_lowercase(),
//                     "resource_dir":     format!("{:?}", resource_dir).to_lowercase(),
//                     "message":           sent,
//                 })).unwrap();
//                 let encoded = match tokenizer {
//                     TokenizerKind::Bpe(b) => b.encode_raw(&json)
//                         .unwrap_or_default()
//                         .into_iter().map(|x| x as usize).collect(),
//                     TokenizerKind::Char(c) => c.encode(&json),
//                 };
//                 (encoded, json)
//             }
//         };

//         let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
//             .chain(input_encoded.iter().cloned())
//             .collect();

//         if enc_input.len() > MAX_SEQ_LEN { continue; }
//         if enc_input.len() < MAX_SEQ_LEN / 2 { continue; }

//         let pad = |mut v: Vec<usize>| -> Vec<usize> {
//             v.resize(MAX_SEQ_LEN, PAD_TOKEN);
//             v
//         };

//         let input_ids     = pad(enc_input);

//         let mut target: Vec<usize> = input_encoded.clone();
//         target.push(EOS_TOKEN);
//         let target_labels = pad(target);

//         samples.push(Sample {
//             input_ids,
//             target_labels,
//             action,
//             motion_dir,
//             world,
//             target_json: String::new(),
//         });
//     }

//     samples
// }

pub fn prepare_paired_samples_split(
    // sentences:    &[String],
    sentences:  Vec<String>,
    tokenizer:    &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
    rng:          &mut impl Rng,
    stage:        TrainingStage,
    max_seq_len:  usize,
) -> Vec<Sample> {
    println!("prepare paired samples split");

    let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
    let mut rng_local = rand::thread_rng();
    let mut samples = Vec::new();

    let analyzer = EmotionAnalyzer::new();

    for sent in sentences {
        if bad_words.iter().any(|&w| sent.to_lowercase().contains(w)) { continue; }

        let words: Vec<&str> = sent.split_whitespace().collect();
        if words.len() < 3 { continue; }

        let world = WorldContext::random(&mut rng_local);
        // let (action, motion_dir) = derive_action(&world);

        let tsample = generate_training_sample(&mut rng_local);

        let action = tsample.action;
        // let motion_dir = tsample.dir;
        let mut command = tsample.command.command;

        let mut target = tsample.command.target;

        // let point = words.len() / 4;
        let point = words.len() / 2;
        let sent_a = words[..point].join(" ");
        let sent_b = words[point..].join(" ");

        // let input_encoded  = tokenizer.encode(&sent_a);

        // let mut obstacle_dir = if let Some(obst) = world.obstacle {
        //     nearby_objects.push("obstacle".to_string());
        //     obst.dir
        // } else {
        //     CardinalDir::None
        // };
        // let mut building_dir = if let Some(obst) = world.building {
        //     nearby_objects.push("building".to_string());
        //     obst.dir
        // } else {
        //     CardinalDir::None
        // };
        // let mut resource_dir = if let Some(obst) = world.resource {
        //     nearby_objects.push("resource".to_string());
        //     obst.dir
        // } else {
        //     CardinalDir::None
        // };

        // Build the memories array: all pairs in this block before the current one
        let prior_memories: Vec<String> = Vec::new();
        
        let (input_encoded, input_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_a);
                let json    = sent_a.clone();
                (encoded, json)
            }
            // TrainingStage::Structured => {
            //     let json = serde_json::to_string_pretty(&serde_json::json!({
            //         "obstacle_dir": format!("{:?}", obstacle_dir).to_lowercase(),
            //         "building_dir": format!("{:?}", building_dir).to_lowercase(),
            //         "resource_dir": format!("{:?}", resource_dir).to_lowercase(),
            //         "nearby_objects": nearby_objects,
            //         "memories":     prior_memories,
            //         "command":      command,
            //         "message":      sent_a,
            //     })).unwrap();
            //     let encoded = match tokenizer {
            //         TokenizerKind::Bpe(b) => b.encode_raw(&json)
            //             .unwrap_or_default()
            //             .into_iter().map(|x| x as usize).collect(),
            //         TokenizerKind::Char(c) => c.encode(&json),
            //     };
            //     (encoded, json)
            // }
            TrainingStage::Structured => {
                // let mut dirs = serde_json::Map::new();

                // if let Some(cmd) = &tsample.command {
                //     // Primary: named target → the direction we're actually moving
                //     dirs.insert(
                //         format!("{} direction", cmd.target),
                //         serde_json::json!(format!("{:?}", motion_dir).to_lowercase()),
                //     );
                // }
                // // Secondary: any other entities in the world that weren't the command target
                // if obstacle_dir != CardinalDir::None {
                //     dirs.insert("obstacle direction".into(),
                //         serde_json::json!(format!("{:?}", obstacle_dir).to_lowercase()));
                // }
                // if building_dir != CardinalDir::None {
                //     dirs.insert("building direction".into(),
                //         serde_json::json!(format!("{:?}", building_dir).to_lowercase()));
                // }
                // if resource_dir != CardinalDir::None {
                //     dirs.insert("resource direction".into(),
                //         serde_json::json!(format!("{:?}", resource_dir).to_lowercase()));
                // }

                let json = serde_json::to_string_pretty(&serde_json::json!({
                    // "nearby_objects": nearby_objects,
                    "memories":       prior_memories,
                    "command":        command,
                    "message":        sent_a,
                    // "directions":     dirs,
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

        let sentiment = sentiment::analyze(sent_b_for_target.to_string());
        let emotion = analyzer.analyze(&sent_b_for_target, &sentiment);

        let (target_encoded, target_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_b_for_target);
                let json    = sent_b_for_target.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                // let json = serde_json::to_string_pretty(&serde_json::json!({
                //     "action":     action.as_str(),
                //     "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
                //     "reply":      sent_b_for_target,
                // })).unwrap();

                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "action":     action.as_str(),
                    // "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
                    "emotion":    emotion,
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

        if target_encoded.is_empty() || target_encoded.len() > max_seq_len - 2 { continue; }

        let target_labels: Vec<usize> = target_encoded.iter().cloned()
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(input_encoded.iter().cloned())
            .collect();

        // if enc_input.len() > max_seq_len { continue; }
        // if enc_input.len() + target_encoded.len() > max_seq_len { continue; }
        // if enc_input.len() + target_encoded.len() < max_seq_len / 4 { continue; }

        if target_encoded.len() < max_seq_len / 2 { continue; }
        if enc_input.len() < max_seq_len / 4 { continue; }
        if target_encoded.len() > max_seq_len { continue; }
        if enc_input.len() > max_seq_len { continue; }
        // if enc_input.len() + target_encoded.len() > max_seq_len { continue; }

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(max_seq_len, PAD_TOKEN);
            v
        };

        let input_ids     = pad(enc_input);
        let target_labels = pad(target_labels);

        samples.push(Sample {
            input_ids,
            target_labels,
            action,
            motion_dir: CardinalDir::None,
            world,
            target_json,
            pair: (sent_a, sent_b)
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
    max_seq_len:  usize,
) -> Vec<Sample> {
    println!("prepare paired samples split");

    let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
    let mut rng_local = rand::thread_rng();
    let mut samples = Vec::new();

    let analyzer = EmotionAnalyzer::new();

    for sents in sentences {
        let sent_a = sents.0;
        let sent_b = sents.1;

        if bad_words.iter().any(|&w| sent_a.to_lowercase().contains(w)) { continue; }
        if bad_words.iter().any(|&w| sent_b.to_lowercase().contains(w)) { continue; }

        let world = WorldContext::random(&mut rng_local);
        // let (action, motion_dir) = derive_action(&world);

        let tsample = generate_training_sample(&mut rng_local);

        let action = tsample.action;
        // let motion_dir = tsample.dir;
        let mut command = tsample.command.command;
        let mut target = tsample.command.target;

        // let mut obstacle_dir = if let Some(obst) = world.obstacle {
        //     nearby_objects.push("obstacle".to_string());
        //     obst.dir
        // } else {
        //     CardinalDir::None
        // };
        // let mut building_dir = if let Some(obst) = world.building {
        //     nearby_objects.push("building".to_string());
        //     obst.dir
        // } else {
        //     CardinalDir::None
        // };
        // let mut resource_dir = if let Some(obst) = world.resource {
        //     nearby_objects.push("resource".to_string());
        //     obst.dir
        // } else {
        //     CardinalDir::None
        // };

        // Build the memories array: all pairs in this block before the current one
        let prior_memories: Vec<String> = Vec::new();
        
        let (input_encoded, input_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_a);
                let json    = sent_a.clone();
                (encoded, json)
            }
            // TrainingStage::Structured => {
            //     let json = serde_json::to_string_pretty(&serde_json::json!({
            //         "obstacle_dir": format!("{:?}", obstacle_dir).to_lowercase(),
            //         "building_dir": format!("{:?}", building_dir).to_lowercase(),
            //         "resource_dir": format!("{:?}", resource_dir).to_lowercase(),
            //         "nearby_objects": nearby_objects,
            //         "memories":     prior_memories,
            //         "command":      command,
            //         "message":      sent_a,
            //     })).unwrap();
            //     let encoded = match tokenizer {
            //         TokenizerKind::Bpe(b) => b.encode_raw(&json)
            //             .unwrap_or_default()
            //             .into_iter().map(|x| x as usize).collect(),
            //         TokenizerKind::Char(c) => c.encode(&json),
            //     };
            //     (encoded, json)
            // }
            TrainingStage::Structured => {
                // let mut dirs = serde_json::Map::new();

                // if let Some(cmd) = &tsample.command {
                //     // Primary: named target → the direction we're actually moving
                //     dirs.insert(
                //         format!("{} direction", cmd.target),
                //         serde_json::json!(format!("{:?}", motion_dir).to_lowercase()),
                //     );
                // }
                // // Secondary: any other entities in the world that weren't the command target
                // if obstacle_dir != CardinalDir::None {
                //     dirs.insert("obstacle direction".into(),
                //         serde_json::json!(format!("{:?}", obstacle_dir).to_lowercase()));
                // }
                // if building_dir != CardinalDir::None {
                //     dirs.insert("building direction".into(),
                //         serde_json::json!(format!("{:?}", building_dir).to_lowercase()));
                // }
                // if resource_dir != CardinalDir::None {
                //     dirs.insert("resource direction".into(),
                //         serde_json::json!(format!("{:?}", resource_dir).to_lowercase()));
                // }

                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "memories":       prior_memories,
                    "command":        command,
                    "message":        sent_a,
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

        let sentiment = sentiment::analyze(sent_b_for_target.to_string());
        let emotion = analyzer.analyze(&sent_b_for_target, &sentiment);

        let (target_encoded, target_json) = match stage {
            TrainingStage::Language => {
                let encoded = tokenizer.encode(&sent_b_for_target);
                let json    = sent_b_for_target.clone();
                (encoded, json)
            }
            TrainingStage::Structured => {
                // let json = serde_json::to_string_pretty(&serde_json::json!({
                //     "action":     action.as_str(),
                //     "motion_dir": format!("{:?}", motion_dir).to_lowercase(),
                //     "reply":      sent_b_for_target,
                // })).unwrap();
                let json = serde_json::to_string_pretty(&serde_json::json!({
                    "action":     action.as_str(),
                    "emotion":    emotion,
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

        if target_encoded.is_empty() || target_encoded.len() > max_seq_len - 2 { continue; }

        let target_labels: Vec<usize> = target_encoded.iter().cloned()
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(input_encoded.iter().cloned())
            .collect();

        // if enc_input.len() > max_seq_len { continue; }

        // if enc_input.len() + target_encoded.len() > max_seq_len { continue; }
        // if enc_input.len() + target_encoded.len() < max_seq_len / 4 { continue; }

        // if target_encoded.len() < max_seq_len / 4 { continue; }
        // if enc_input.len() + target_encoded.len() > max_seq_len { continue; }

        if target_encoded.len() < max_seq_len / 2 { continue; }
        if enc_input.len() < max_seq_len / 4 { continue; }
        if target_encoded.len() > max_seq_len { continue; }
        if enc_input.len() > max_seq_len { continue; }

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(max_seq_len, PAD_TOKEN);
            v
        };

        let input_ids     = pad(enc_input);
        let target_labels = pad(target_labels);

        samples.push(Sample {
            input_ids,
            target_labels,
            action,
            motion_dir: CardinalDir::None,
            world,
            target_json,
            pair: (sent_a, sent_b)
        });
    }

    samples
}

pub fn prepare_paired_samples_chats(
    chats:         HandcraftedChats,
    tokenizer:     &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
    rng:           &mut impl Rng,
    stage:        TrainingStage,
    max_seq_len:  usize,
) -> Vec<Sample> {
    println!("prepare paired samples chat");

    let bad_words = vec!["sex", "drug", "kill", "rape", "nazi"];
    let mut rng_local = rand::thread_rng();
    let mut samples = Vec::new();

    let analyzer = EmotionAnalyzer::new();

    for block in &chats.blocks {
        for (i, memory) in block.memories.iter().enumerate() {
            let sent = &memory.human;

            if bad_words.iter().any(|&w| sent.to_lowercase().contains(w)) { continue; }

            let words: Vec<&str> = sent.split_whitespace().collect();
            if words.len() < 3 { continue; }

            let world = WorldContext::random(&mut rng_local);
            
            // let (action, motion_dir) = derive_action(&world);

            let tsample = generate_training_sample(&mut rng_local);

            let action = tsample.action;
            let mut command = tsample.command.command;
            let mut target = tsample.command.target;

            let sentiment = sentiment::analyze(sent.to_string());
            let emotion = analyzer.analyze(sent, &sentiment);

            // let mut obstacle_dir = if let Some(obst) = world.obstacle {
            //     nearby_objects.push("obstacle".to_string());
            //     obst.dir
            // } else {
            //     CardinalDir::None
            // };
            // let mut building_dir = if let Some(obst) = world.building {
            //     nearby_objects.push("building".to_string());
            //     obst.dir
            // } else {
            //     CardinalDir::None
            // };
            // let mut resource_dir = if let Some(obst) = world.resource {
            //     nearby_objects.push("resource".to_string());
            //     obst.dir
            // } else {
            //     CardinalDir::None
            // };

            // Build the memories array: all pairs in this block before the current one
            let prior_memories: Vec<serde_json::Value> = block.memories[..i]
                .iter()
                .map(|m| serde_json::json!({
                    "human": m.human,
                    "yumon": m.bot,
                }))
                .collect();

            let (input_encoded, input_json) = match stage {
                TrainingStage::Language => {
                    let encoded = tokenizer.encode(sent);
                    let json    = sent.clone();
                    (encoded, json)
                }
                // TrainingStage::Structured => {
                //     let json = serde_json::to_string_pretty(&serde_json::json!({
                //         // TODO: we actually want this to be dynamic, like "red fruit direction", will also simplify
                //         "obstacle_dir": format!("{:?}", obstacle_dir).to_lowercase(),
                //         "building_dir": format!("{:?}", building_dir).to_lowercase(),
                //         "resource_dir": format!("{:?}", resource_dir).to_lowercase(),
                //         "nearby_objects": nearby_objects,
                //         "memories":     prior_memories,
                //         "command":      command,
                //         "message":      memory.human,
                //     })).unwrap();
                //     let encoded = match tokenizer {
                //         TokenizerKind::Bpe(b) => b.encode_raw(&json)
                //             .unwrap_or_default()
                //             .into_iter().map(|x| x as usize).collect(),
                //         TokenizerKind::Char(c) => c.encode(&json),
                //     };
                //     (encoded, json)
                // }
                TrainingStage::Structured => {
                    let json = serde_json::to_string_pretty(&serde_json::json!({
                        // "scene description": "a beautiful outdoors oasis".to_string(),
                        "memories":       prior_memories,
                        "command":        command,
                        "message":        memory.human,
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

            let (target_encoded, target_json) = match stage {
                TrainingStage::Language => {
                    let encoded = tokenizer.encode(&memory.bot);
                    let json    = memory.bot.clone();
                    (encoded, json)
                }
                TrainingStage::Structured => {
                    let json = serde_json::to_string_pretty(&serde_json::json!({
                        "action":     action.as_str(),
                        "emotion":    emotion,
                        "reply":      memory.bot,
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

            if target_encoded.is_empty() || target_encoded.len() > max_seq_len - 2 { continue; }

            let target_labels: Vec<usize> = target_encoded.iter().cloned()
                .chain(std::iter::once(EOS_TOKEN))
                .collect();

            let enc_input: Vec<usize> = std::iter::once(BOS_TOKEN)
                .chain(input_encoded.iter().cloned())
                .collect();

            // if enc_input.len() > max_seq_len { continue; }
            // if target_encoded.len() < max_seq_len / 4 { continue; }
            // if enc_input.len() + target_encoded.len() > max_seq_len { continue; }
            // if enc_input.len() + target_encoded.len() < max_seq_len / 4 { continue; } // on Language stage, you want the short ones

            if target_encoded.len() < max_seq_len / 2 { continue; }
            if enc_input.len() < max_seq_len / 4 { continue; }
            if target_encoded.len() > max_seq_len { continue; }
            if enc_input.len() > max_seq_len { continue; }

            let pad = |mut v: Vec<usize>| -> Vec<usize> {
                v.resize(max_seq_len, PAD_TOKEN);
                v
            };

            let input_ids     = pad(enc_input);
            let target_labels = pad(target_labels);

            samples.push(Sample {
                input_ids,
                target_labels,
                action,
                motion_dir: CardinalDir::None,
                world,
                target_json,
                pair: (memory.human.clone(), memory.bot.clone()),
            });
        }
    }

    samples
}
