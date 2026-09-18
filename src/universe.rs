//! Procedural neighborhood and the plain-language bridge used by Universe.
//! Kept independent of rendering and model inference so world behavior is testable.

use clap::ValueEnum;

pub const EXTENT: f32 = 30.0;
pub const SIGHT: f32 = 12.0;

/// Selects which real-world setting the generated world, its scenery, and its
/// procedural prompts are skinned as. The underlying `Kind` slots, layout, and
/// behavior categories are identical across themes - only the nouns, the odd
/// verb, and rendering differ, so both settings share one tested world model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum Theme {
    #[default]
    Suburban,
    Urban,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub z: f32,
}
impl Point {
    pub fn distance(self, other: Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    House,
    Shop,
    Tree,
    Flowers,
    Bench,
    Ball,
    Garden,
    Mailbox,
    Pond,
}
impl Kind {
    /// The same slot carries a different noun per theme. Urban nouns are pulled
    /// from `archive/synthetic/business.txt` - the actual corpus this checkpoint's
    /// Language stage trains on (see `load_stage_data` in `src/brain/train.rs`) -
    /// and picked by frequency there, not invented office jargon the model has
    /// never seen a token of. Layout, radius, and behavior compatibility never
    /// change with theme.
    pub fn noun(self, theme: Theme) -> &'static str {
        match (self, theme) {
            (Self::House, Theme::Suburban) => "house",
            (Self::House, Theme::Urban) => "business", // 613 occurrences (incl. plural)
            (Self::Shop, Theme::Suburban) => "shop",
            (Self::Shop, Theme::Urban) => "market", // 126
            (Self::Tree, Theme::Suburban) => "tree",
            (Self::Tree, Theme::Urban) => "debt", // 132
            (Self::Flowers, Theme::Suburban) => "flowers",
            (Self::Flowers, Theme::Urban) => "money", // 141
            (Self::Bench, Theme::Suburban) => "bench",
            (Self::Bench, Theme::Urban) => "coffee", // 3, kept for "rest by" fit
            (Self::Ball, Theme::Suburban) => "ball",
            (Self::Ball, Theme::Urban) => "product", // 142
            (Self::Garden, Theme::Suburban) => "garden",
            (Self::Garden, Theme::Urban) => "budget", // 48
            (Self::Mailbox, Theme::Suburban) => "mailbox",
            (Self::Mailbox, Theme::Urban) => "deal", // 31
            (Self::Pond, Theme::Suburban) => "pond",
            (Self::Pond, Theme::Urban) => "brand", // 115
        }
    }
    pub fn radius(self) -> f32 {
        match self {
            Self::House | Self::Shop => 2.4,
            Self::Pond => 1.6,
            Self::Tree => 0.5,
            _ => 0.4,
        }
    }
    /// The verb offered in a question, paired with the behavior it maps to.
    /// Only the garden/budget slot needs a themed verb - "tend" reads fine for a
    /// garden but not for a budget, which gets "maintain" instead. `maintain`
    /// itself is common in `business.txt` (paired there with brand/cash flow/
    /// partnership, not literally with budget), so it isn't a novel token either.
    pub fn verb_and_behavior(self, theme: Theme) -> (&'static str, Behavior) {
        match self {
            Self::Bench => ("rest by", Behavior::Rest),
            Self::Ball => ("play with", Behavior::Play),
            Self::Garden => match theme {
                Theme::Suburban => ("tend", Behavior::Tend),
                Theme::Urban => ("maintain", Behavior::Tend),
            },
            Self::Flowers => ("collect", Behavior::Collect),
            Self::House | Self::Shop => ("visit", Behavior::Visit),
            _ => ("look at", Behavior::Inspect),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub kind: Kind,
    pub pos: Point,
    pub tint: [u8; 3],
    pub activity_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Behavior {
    Rest,
    Visit,
    Play,
    Collect,
    Tend,
    Inspect,
    Explore,
}
impl Behavior {
    /// "gardening" only reads right for a literal garden - a server gets
    /// "maintaining" instead. Every other label already fits both themes.
    pub fn label(self, theme: Theme) -> &'static str {
        match self {
            Self::Rest => "resting",
            Self::Visit => "visiting",
            Self::Play => "playing",
            Self::Collect => "collecting",
            Self::Tend => match theme {
                Theme::Suburban => "gardening",
                Theme::Urban => "maintaining",
            },
            Self::Inspect => "looking around",
            Self::Explore => "exploring",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Question {
    pub text: String,
    pub place: Option<usize>,
    pub offered: Behavior,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub behavior: Behavior,
    pub place: Option<usize>,
}

pub struct Neighborhood {
    pub seed: u64,
    pub theme: Theme,
    pub places: Vec<Place>,
}
impl Neighborhood {
    pub fn generate(seed: u64, theme: Theme) -> Self {
        let mut state = seed;
        let mut random = || {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut n = state;
            n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((n ^ (n >> 31)) >> 40) as f32 / (1u32 << 24) as f32
        };
        let mut places = Vec::new();
        for row in 0..4 {
            for col in 0..4 {
                let x = -22.5 + col as f32 * 15.0;
                let z = -22.5 + row as f32 * 15.0;
                let main = if row == 1 && col == 1 {
                    Kind::Pond
                } else if (row + col) % 5 == 0 {
                    Kind::Shop
                } else {
                    Kind::House
                };
                // Suburban tints stay warm/earthy; urban tints shift cool toward
                // glass and steel to read as an office park at a glance.
                let tint = match theme {
                    Theme::Suburban => [
                        140 + (random() * 90.0) as u8,
                        140 + (random() * 80.0) as u8,
                        120 + (random() * 90.0) as u8,
                    ],
                    Theme::Urban => [
                        90 + (random() * 60.0) as u8,
                        100 + (random() * 65.0) as u8,
                        120 + (random() * 75.0) as u8,
                    ],
                };
                places.push(Place {
                    kind: main,
                    pos: Point {
                        x: x + random() - 0.5,
                        z: z + random() - 0.5,
                    },
                    tint,
                    activity_count: 0,
                });
                for (slot, kind) in [
                    Kind::Tree,
                    Kind::Flowers,
                    Kind::Bench,
                    Kind::Ball,
                    Kind::Garden,
                    Kind::Mailbox,
                ]
                .into_iter()
                .enumerate()
                {
                    let angle = slot as f32 * std::f32::consts::TAU / 6.0;
                    let radius = 4.3 + random() * 0.5;
                    places.push(Place {
                        kind,
                        pos: Point {
                            x: x + angle.cos() * radius,
                            z: z + angle.sin() * radius,
                        },
                        tint,
                        activity_count: 0,
                    });
                }
            }
        }
        Self { seed, theme, places }
    }

    pub fn nearby(&self, pos: Point) -> Vec<usize> {
        let mut ids: Vec<_> = self
            .places
            .iter()
            .enumerate()
            .filter(|(_, p)| p.pos.distance(pos) <= SIGHT)
            .map(|(i, _)| i)
            .collect();
        ids.sort_by(|&a, &b| {
            self.places[a]
                .pos
                .distance(pos)
                .total_cmp(&self.places[b].pos.distance(pos))
        });
        ids
    }

    pub fn question(&self, pos: Point, turn: usize) -> Question {
        let nearby = self.nearby(pos);
        if nearby.is_empty() {
            return Question {
                text: "Would you like to explore the neighborhood?".into(),
                place: None,
                offered: Behavior::Explore,
            };
        }
        // Rotate among the closest few objects, keeping prompts small for Language checkpoints.
        let id = nearby[turn % nearby.len().min(4)];
        let place = &self.places[id];
        let (verb, offered) = place.kind.verb_and_behavior(self.theme);
        Question {
            text: format!(
                "The {} is {}. Would you like to {} it?",
                place.kind.noun(self.theme),
                direction(pos, place.pos),
                verb
            ),
            place: Some(id),
            offered,
        }
    }

    pub fn walkable(&self, pos: Point) -> bool {
        pos.x.abs() < EXTENT - 0.6
            && pos.z.abs() < EXTENT - 0.6
            && self
                .places
                .iter()
                .all(|p| pos.distance(p.pos) > p.kind.radius() + 0.35)
    }

    pub fn approach(&self, from: Point, id: usize) -> Point {
        let p = &self.places[id];
        let dx = from.x - p.pos.x;
        let dz = from.z - p.pos.z;
        let distance = dx.hypot(dz).max(0.01);
        Point {
            x: p.pos.x + dx / distance * (p.kind.radius() + 0.8),
            z: p.pos.z + dz / distance * (p.kind.radius() + 0.8),
        }
    }
}

pub fn direction(from: Point, to: Point) -> &'static str {
    let dx = to.x - from.x;
    let dz = to.z - from.z;
    if dx.abs() > dz.abs() {
        if dx > 0.0 { "east" } else { "west" }
    } else if dz > 0.0 {
        "south"
    } else {
        "north"
    }
}

/// Conservative phrase matching: a refusal or unknown reply never invents an action.
/// Targets are restricted to the current local observation and the question's referent.
pub fn infer_reply(reply: &str, question: &Question, world: &Neighborhood, pos: Point) -> Decision {
    let normalized: String = reply
        .to_lowercase()
        .replace('’', "'")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let text = format!(
        " {} ",
        normalized.split_whitespace().collect::<Vec<_>>().join(" ")
    );
    let has = |phrase: &str| text.contains(&format!(" {phrase} "));
    let rest = Decision {
        behavior: Behavior::Rest,
        place: None,
    };
    if [
        "no",
        "not",
        "don't",
        "won't",
        "can't",
        "cannot",
        "never",
        "unsure",
        "maybe",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "rather",
    ]
    .iter()
    .any(|p| has(p))
    {
        return rest;
    }
    let behavior = if ["sit", "rest", "relax", "wait", "sleep"]
        .iter()
        .any(|p| has(p))
    {
        Some(Behavior::Rest)
    } else if ["play", "kick", "throw"].iter().any(|p| has(p)) {
        Some(Behavior::Play)
    } else if ["collect", "gather", "pick", "pick up", "restock", "grab"]
        .iter()
        .any(|p| has(p))
    {
        Some(Behavior::Collect)
    } else if ["water", "tend", "plant", "maintain", "fix"]
        .iter()
        .any(|p| has(p))
    {
        Some(Behavior::Tend)
    } else if ["look", "inspect", "watch", "check"].iter().any(|p| has(p)) {
        Some(Behavior::Inspect)
    } else if ["visit", "go", "walk", "head"].iter().any(|p| has(p)) {
        Some(Behavior::Visit)
    } else if ["explore", "wander"].iter().any(|p| has(p)) {
        Some(Behavior::Explore)
    } else {
        None
    };
    let accepted = [
        "yes",
        "sure",
        "okay",
        "ok",
        "let's",
        "sounds good",
        "i'd love to",
        "i would love to",
    ]
    .iter()
    .any(|p| has(p));
    let Some(behavior) = behavior.or_else(|| accepted.then_some(question.offered)) else {
        return rest;
    };
    let nearby = world.nearby(pos);
    let named = nearby
        .iter()
        .copied()
        .find(|&id| has(world.places[id].kind.noun(world.theme)));
    let target = named.or(question.place.filter(|id| nearby.contains(id)));
    let compatible = |kind| match behavior {
        Behavior::Play => kind == Kind::Ball,
        Behavior::Collect => kind == Kind::Flowers,
        Behavior::Tend => kind == Kind::Garden,
        Behavior::Rest => kind == Kind::Bench,
        Behavior::Explore => false,
        _ => true,
    };
    let place = target.filter(|&id| compatible(world.places[id].kind));
    if matches!(
        behavior,
        Behavior::Play | Behavior::Collect | Behavior::Tend | Behavior::Visit | Behavior::Inspect
    ) && place.is_none()
    {
        return rest;
    }
    Decision { behavior, place }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seeded_world_is_varied_and_safe() {
        for theme in [Theme::Suburban, Theme::Urban] {
            let world = Neighborhood::generate(42, theme);
            assert_eq!(world.places, Neighborhood::generate(42, theme).places);
            assert_ne!(world.places, Neighborhood::generate(43, theme).places);
            for (i, p) in world.places.iter().enumerate() {
                assert!(p.pos.x.abs() + p.kind.radius() < EXTENT);
                assert!(p.pos.z.abs() + p.kind.radius() < EXTENT);
                for other in world.places.iter().skip(i + 1) {
                    assert!(p.pos.distance(other.pos) > p.kind.radius() + other.kind.radius());
                }
            }
            assert!(world.walkable(Point { x: 0.0, z: 0.0 }));
            assert!(!world.walkable(world.places[0].pos));
        }
        // The suburban and urban layouts share the same seeded positions/kinds -
        // only the theming (noun, verb, tint) differs.
        let suburban = Neighborhood::generate(42, Theme::Suburban);
        let urban = Neighborhood::generate(42, Theme::Urban);
        assert_eq!(
            suburban.places.iter().map(|p| p.kind).collect::<Vec<_>>(),
            urban.places.iter().map(|p| p.kind).collect::<Vec<_>>()
        );
    }
    #[test]
    fn urban_theme_renames_every_kind_without_changing_behavior() {
        let world = Neighborhood::generate(9, Theme::Urban);
        for kind in [
            Kind::House,
            Kind::Shop,
            Kind::Tree,
            Kind::Flowers,
            Kind::Bench,
            Kind::Ball,
            Kind::Garden,
            Kind::Mailbox,
            Kind::Pond,
        ] {
            assert_ne!(kind.noun(Theme::Urban), kind.noun(Theme::Suburban));
        }
        let id = world
            .places
            .iter()
            .position(|p| p.kind == Kind::Garden)
            .unwrap();
        let q = world.question(world.places[id].pos, 0);
        assert!(q.text.contains("budget"));
        assert!(q.text.contains("maintain"));
        assert_eq!(q.offered, Behavior::Tend);
        let pos = world.approach(Point { x: 0.0, z: 0.0 }, id);
        let decision = infer_reply("Sure, I'll maintain it", &q, &world, pos);
        assert_eq!(
            decision,
            Decision {
                behavior: Behavior::Tend,
                place: Some(id)
            }
        );
    }
    #[test]
    fn urban_nouns_are_present_in_the_training_corpus() {
        // Every Urban noun must actually occur in the corpus the Language stage
        // trains on (src/brain/train.rs's load_stage_data), not just read as
        // plausible business vocabulary - otherwise the checkpoint has no real
        // grounding for the word and the whole point of this theme is lost.
        let corpus = std::fs::read_to_string("archive/synthetic/business.txt")
            .expect("training corpus for the Urban theme")
            .to_lowercase();
        for kind in [
            Kind::House,
            Kind::Shop,
            Kind::Tree,
            Kind::Flowers,
            Kind::Bench,
            Kind::Ball,
            Kind::Garden,
            Kind::Mailbox,
            Kind::Pond,
        ] {
            let noun = kind.noun(Theme::Urban);
            assert!(
                corpus.contains(noun),
                "Urban noun {noun:?} for {kind:?} does not appear in business.txt"
            );
        }
    }
    #[test]
    fn questions_and_replies_are_grounded() {
        let world = Neighborhood::generate(9, Theme::Suburban);
        let id = world
            .places
            .iter()
            .position(|p| p.kind == Kind::Ball)
            .unwrap();
        let pos = world.approach(Point { x: 0.0, z: 0.0 }, id);
        let q = Question {
            text: "Would you like to play with the ball?".into(),
            place: Some(id),
            offered: Behavior::Play,
        };
        assert_eq!(
            infer_reply("Yes, please!", &q, &world, pos),
            Decision {
                behavior: Behavior::Play,
                place: Some(id)
            }
        );
        for reply in [
            "No thanks",
            "I don't want to play",
            "I won’t play",
            "I wouldn't play",
            "Maybe",
            "Yesterday was nice",
            "noteworthy",
        ] {
            assert_eq!(infer_reply(reply, &q, &world, pos).place, None);
        }
        assert_eq!(
            infer_reply("I'll water it", &q, &world, pos).behavior,
            Behavior::Rest
        );
        for turn in 0..16 {
            let q = world.question(pos, turn);
            assert!(world.nearby(pos).contains(&q.place.unwrap()));
            assert!(!q.text.contains('{'));
        }
        assert_eq!(
            infer_reply("Yes", &q, &world, Point { x: 29.0, z: 29.0 }).place,
            None
        );
    }

    #[test]
    fn generated_questions_fit_small_language_context() {
        use crate::brain::bpe::BpeTokenizer;
        let tokenizer = BpeTokenizer::load("yumon_bpe").expect("repository tokenizer");
        for theme in [Theme::Suburban, Theme::Urban] {
            let world = Neighborhood::generate(42, theme);
            for place in &world.places {
                for turn in 0..4 {
                    let q = world.question(place.pos, turn);
                    assert!(
                        tokenizer.encode(&q.text).unwrap().len()
                            + tokenizer.encode(" ").unwrap().len()
                            + 9
                            <= 32,
                        "Question exceeds the default Language context: {}",
                        q.text
                    );
                }
            }
        }
    }
}
