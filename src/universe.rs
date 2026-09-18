//! Procedural neighborhood and the plain-language bridge used by Universe.
//! Kept independent of rendering and model inference so world behavior is testable.

pub const EXTENT: f32 = 30.0;
pub const SIGHT: f32 = 12.0;

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
    pub fn noun(self) -> &'static str {
        match self {
            Self::House => "house",
            Self::Shop => "shop",
            Self::Tree => "tree",
            Self::Flowers => "flowers",
            Self::Bench => "bench",
            Self::Ball => "ball",
            Self::Garden => "garden",
            Self::Mailbox => "mailbox",
            Self::Pond => "pond",
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
    pub fn label(self) -> &'static str {
        match self {
            Self::Rest => "resting",
            Self::Visit => "visiting",
            Self::Play => "playing",
            Self::Collect => "collecting",
            Self::Tend => "gardening",
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
    pub places: Vec<Place>,
}
impl Neighborhood {
    pub fn generate(seed: u64) -> Self {
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
                let tint = [
                    140 + (random() * 90.0) as u8,
                    140 + (random() * 80.0) as u8,
                    120 + (random() * 90.0) as u8,
                ];
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
        Self { seed, places }
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
        let (verb, offered) = match place.kind {
            Kind::Bench => ("rest by", Behavior::Rest),
            Kind::Ball => ("play with", Behavior::Play),
            Kind::Garden => ("tend", Behavior::Tend),
            Kind::Flowers => ("collect", Behavior::Collect),
            Kind::House | Kind::Shop => ("visit", Behavior::Visit),
            _ => ("look at", Behavior::Inspect),
        };
        Question {
            text: format!(
                "The {} is {}. Would you like to {} it?",
                place.kind.noun(),
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
    } else if ["collect", "gather", "pick", "pick up"]
        .iter()
        .any(|p| has(p))
    {
        Some(Behavior::Collect)
    } else if ["water", "tend", "plant"].iter().any(|p| has(p)) {
        Some(Behavior::Tend)
    } else if ["look", "inspect", "watch"].iter().any(|p| has(p)) {
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
        .find(|&id| has(world.places[id].kind.noun()));
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
        let world = Neighborhood::generate(42);
        assert_eq!(world.places, Neighborhood::generate(42).places);
        assert_ne!(world.places, Neighborhood::generate(43).places);
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
    #[test]
    fn questions_and_replies_are_grounded() {
        let world = Neighborhood::generate(9);
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
        let world = Neighborhood::generate(42);
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
