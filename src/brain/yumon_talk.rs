//! The Yumon Talk Language (1024 total tokens).
//!
//! Yumon Talk is a compact, canonical natural language designed for classic Pet.
//! It addresses vocabulary explosion by enforcing:
//! - Exactly 1024 total tokens in the vocabulary (IDs 0..1023).
//! - Semantic clustering: each canonical word maps to a rich set of English synonyms.
//! - Bidirectional translation: English <-> Yumon Talk.
//! - Dynamic expansion via the `thesaurus` crate and morphological stemming.
//! - Placeholder slots (`<NAME_0>`..`<NAME_31>`, `<NUM_0>`..`<NUM_7>`) for entities, brand names, and numbers.
//! - Random synonym sampling during translation back to English.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

pub const VOCAB_SIZE: usize = 1024;
pub const PAD_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const UNK_ID: usize = 3;

/// Canonical 1024-token vocabulary.
pub static VOCAB: [&str; VOCAB_SIZE] = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "-",
    "'",
    "\"",
    "(",
    ")",
    "[",
    "]",
    "/",
    "\n",
    "<NAME_0>",
    "<NAME_1>",
    "<NAME_2>",
    "<NAME_3>",
    "<NAME_4>",
    "<NAME_5>",
    "<NAME_6>",
    "<NAME_7>",
    "<NAME_8>",
    "<NAME_9>",
    "<NAME_10>",
    "<NAME_11>",
    "<NAME_12>",
    "<NAME_13>",
    "<NAME_14>",
    "<NAME_15>",
    "<NAME_16>",
    "<NAME_17>",
    "<NAME_18>",
    "<NAME_19>",
    "<NAME_20>",
    "<NAME_21>",
    "<NAME_22>",
    "<NAME_23>",
    "<NAME_24>",
    "<NAME_25>",
    "<NAME_26>",
    "<NAME_27>",
    "<NAME_28>",
    "<NAME_29>",
    "<NAME_30>",
    "<NAME_31>",
    "<NUM_0>",
    "<NUM_1>",
    "<NUM_2>",
    "<NUM_3>",
    "<NUM_4>",
    "<NUM_5>",
    "<NUM_6>",
    "<NUM_7>",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whose",
    "which",
    "what",
    "all",
    "every",
    "any",
    "some",
    "none",
    "both",
    "few",
    "many",
    "much",
    "more",
    "most",
    "other",
    "same",
    "such",
    "the",
    "a",
    "and",
    "or",
    "but",
    "so",
    "because",
    "if",
    "then",
    "than",
    "as",
    "while",
    "unless",
    "until",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "without",
    "about",
    "against",
    "between",
    "among",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "under",
    "over",
    "into",
    "onto",
    "out",
    "up",
    "down",
    "from",
    "to",
    "off",
    "be",
    "is",
    "was",
    "have",
    "do",
    "will",
    "shall",
    "can",
    "may",
    "must",
    "not",
    "never",
    "no",
    "yes",
    "hello",
    "goodbye",
    "please",
    "thank",
    "sorry",
    "ok",
    "well",
    "oh",
    "amen",
    "where",
    "when",
    "why",
    "how",
    "very",
    "really",
    "quite",
    "almost",
    "too",
    "even",
    "just",
    "only",
    "enough",
    "again",
    "already",
    "still",
    "now",
    "soon",
    "here",
    "there",
    "together",
    "alone",
    "always",
    "often",
    "sometimes",
    "perhaps",
    "suddenly",
    "finally",
    "instead",
    "almost_degree",
    "beside",
    "beyond",
    "except",
    "seldom",
    "away",
    "time",
    "day",
    "night",
    "morning",
    "evening",
    "noon",
    "hour",
    "minute",
    "second",
    "week",
    "month",
    "year",
    "season",
    "spring_time",
    "summer",
    "autumn",
    "winter",
    "today",
    "yesterday",
    "tomorrow",
    "age",
    "past",
    "present",
    "future",
    "early",
    "late",
    "date",
    "forever",
    "say",
    "ask",
    "answer",
    "call",
    "cry",
    "whisper",
    "explain",
    "promise",
    "praise",
    "blame",
    "warn",
    "greet",
    "argue",
    "discuss",
    "advise",
    "refuse",
    "agree",
    "command",
    "obey",
    "see",
    "hear",
    "feel",
    "smell",
    "taste",
    "think",
    "know",
    "understand",
    "remember",
    "forget",
    "learn",
    "teach",
    "believe",
    "doubt",
    "hope",
    "wonder",
    "dream",
    "imagine",
    "decide",
    "choose",
    "judge",
    "read",
    "write",
    "go",
    "come",
    "walk",
    "run",
    "jump",
    "climb",
    "fly",
    "swim",
    "drive",
    "fall",
    "rise",
    "stand",
    "sit",
    "lie",
    "sleep",
    "wake",
    "stay",
    "leave",
    "return",
    "stop",
    "wait",
    "follow",
    "lead",
    "shake",
    "hang",
    "roll",
    "turn",
    "make",
    "build",
    "break",
    "destroy",
    "fix",
    "cut",
    "open",
    "close",
    "cover",
    "hide",
    "show",
    "give",
    "take",
    "hold",
    "carry",
    "bring",
    "send",
    "put",
    "drop",
    "throw",
    "pull",
    "push",
    "find",
    "lose",
    "seek",
    "help",
    "hurt",
    "heal",
    "kill",
    "die",
    "live",
    "eat",
    "drink",
    "wear",
    "wash",
    "burn",
    "fight",
    "win",
    "fail",
    "work",
    "play",
    "try",
    "use",
    "buy",
    "sell",
    "pay",
    "save",
    "guard",
    "attack",
    "allow",
    "love",
    "hate",
    "fear",
    "care",
    "like",
    "want",
    "start",
    "end",
    "continue",
    "change",
    "grow",
    "shrink",
    "shine",
    "appear",
    "disappear",
    "happen",
    "become",
    "draw",
    "sing",
    "laugh",
    "smile",
    "pour",
    "fill",
    "tie",
    "release",
    "shoot",
    "blow",
    "suffer",
    "rejoice",
    "mourn",
    "strike",
    "touch",
    "meet",
    "bend",
    "fold",
    "spread",
    "gather",
    "count",
    "measure_verb",
    "burn_light",
    "cook",
    "hunt",
    "catch_trap",
    "lead_guide",
    "person",
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "baby",
    "father",
    "mother",
    "parent",
    "son",
    "daughter",
    "brother",
    "sister",
    "family",
    "husband",
    "wife",
    "friend",
    "enemy",
    "neighbor",
    "guest",
    "host",
    "stranger",
    "crowd",
    "people",
    "king",
    "queen",
    "prince",
    "princess",
    "lord",
    "lady",
    "master",
    "servant",
    "slave",
    "ruler",
    "leader",
    "soldier",
    "knight",
    "officer",
    "priest",
    "prophet",
    "teacher",
    "student",
    "doctor",
    "nurse",
    "worker",
    "farmer",
    "merchant",
    "driver",
    "sailor",
    "thief",
    "criminal",
    "hero",
    "witness",
    "messenger",
    "author",
    "artist",
    "musician",
    "actor",
    "fool",
    "giant",
    "ancestor",
    "generation",
    "crowd_mob",
    "ally",
    "citizen",
    "elder",
    "body",
    "head",
    "brain",
    "hair",
    "face",
    "forehead",
    "eye",
    "ear",
    "nose",
    "mouth",
    "lip",
    "tooth",
    "tongue",
    "chin",
    "neck",
    "shoulder",
    "arm",
    "elbow",
    "wrist",
    "hand",
    "finger",
    "thumb",
    "chest",
    "heart",
    "stomach",
    "back",
    "waist",
    "leg",
    "knee",
    "foot",
    "toe",
    "skin",
    "bone",
    "blood",
    "flesh",
    "muscle",
    "nerve",
    "soul",
    "spirit",
    "mind",
    "breath",
    "voice",
    "tear",
    "sweat",
    "wound",
    "scar",
    "disease",
    "pain",
    "health",
    "sleep_state",
    "fever",
    "cough",
    "sigh",
    "touch_sense",
    "world",
    "earth",
    "universe",
    "sky",
    "heaven",
    "sun",
    "moon",
    "star",
    "planet",
    "space",
    "light",
    "darkness",
    "shadow",
    "cloud",
    "rain",
    "snow",
    "ice",
    "wind",
    "storm",
    "thunder",
    "lightning",
    "fire",
    "flame",
    "smoke",
    "ash",
    "water",
    "sea",
    "river",
    "lake",
    "wave",
    "tide",
    "flood",
    "spring",
    "mountain",
    "hill",
    "valley",
    "cliff",
    "cave",
    "desert",
    "island",
    "coast",
    "plain",
    "field",
    "forest",
    "tree",
    "branch",
    "leaf",
    "root",
    "bark",
    "wood",
    "plant",
    "grass",
    "flower",
    "rose",
    "lily",
    "weed",
    "bush",
    "seed",
    "fruit",
    "stone",
    "sand",
    "dust",
    "mud",
    "gold",
    "silver",
    "iron",
    "copper",
    "metal",
    "mineral",
    "jewel",
    "air",
    "fog",
    "dew",
    "heat",
    "cold",
    "nature",
    "animal",
    "pet",
    "monster",
    "dragon",
    "dog",
    "cat",
    "mouse",
    "rat",
    "horse",
    "donkey",
    "cow",
    "bull",
    "calf",
    "sheep",
    "lamb",
    "goat",
    "pig",
    "bear",
    "wolf",
    "fox",
    "lion",
    "tiger",
    "deer",
    "rabbit",
    "monkey",
    "elephant",
    "camel",
    "bird",
    "eagle",
    "hawk",
    "dove",
    "crow",
    "duck",
    "goose",
    "swan",
    "chicken",
    "owl",
    "sparrow",
    "fish",
    "whale",
    "shark",
    "salmon",
    "snake",
    "frog",
    "worm",
    "insect",
    "bee",
    "ant",
    "spider",
    "butterfly",
    "place",
    "building",
    "house",
    "room",
    "wall",
    "door",
    "gate",
    "window",
    "roof",
    "floor",
    "ceiling",
    "hall",
    "kitchen",
    "bedroom",
    "cellar",
    "stairs",
    "chimney",
    "fireplace",
    "tower",
    "castle",
    "palace",
    "temple",
    "church",
    "prison",
    "fortress",
    "fence",
    "table",
    "chair",
    "bed",
    "shelf",
    "box",
    "chest_furniture",
    "bag",
    "basket",
    "bottle",
    "cup",
    "glass",
    "dish",
    "bowl",
    "pot",
    "kettle",
    "bucket",
    "barrel",
    "lamp",
    "mirror",
    "road",
    "path",
    "bridge",
    "tunnel",
    "city",
    "town",
    "village",
    "country",
    "kingdom",
    "market",
    "shop",
    "court",
    "camp",
    "harbor",
    "ship",
    "cart",
    "car",
    "train",
    "tool",
    "weapon",
    "sword",
    "knife",
    "spear",
    "shield",
    "armor",
    "helmet",
    "bow",
    "arrow",
    "gun",
    "bullet",
    "rope",
    "chain",
    "wire",
    "nail",
    "pin",
    "hook",
    "ring",
    "stick",
    "wheel",
    "bell",
    "lock",
    "key",
    "clock",
    "book",
    "paper",
    "letter",
    "page",
    "pen",
    "picture",
    "map",
    "clothes",
    "coat",
    "dress",
    "shirt",
    "pants",
    "shoe",
    "boot",
    "hat",
    "crown",
    "glove",
    "belt",
    "cloth",
    "thread",
    "wool",
    "cotton",
    "silk",
    "leather",
    "blanket",
    "curtain",
    "flag",
    "cross",
    "staff",
    "plow",
    "axe",
    "hammer",
    "net",
    "trap",
    "horn",
    "drum",
    "pipe",
    "string",
    "crown_jewel",
    "food",
    "meat",
    "bread",
    "grain",
    "corn",
    "wheat",
    "rice",
    "flour",
    "dish_meal",
    "egg",
    "cheese",
    "butter",
    "milk",
    "oil",
    "fat",
    "salt",
    "sugar",
    "honey",
    "spice",
    "soup",
    "pie",
    "cake",
    "apple",
    "grape",
    "wine",
    "beer",
    "tea",
    "coffee",
    "feast",
    "poison",
    "potatoes",
    "beans",
    "onion",
    "nut",
    "berry",
    "olive",
    "fig",
    "vinegar",
    "crust",
    "dough",
    "pottage",
    "sweet",
    "hunger",
    "thirst",
    "taste_flavor",
    "thing",
    "name",
    "word",
    "sound",
    "music",
    "song",
    "story",
    "news",
    "message",
    "truth",
    "secret",
    "error",
    "fact",
    "reason",
    "cause",
    "effect",
    "purpose",
    "plan",
    "idea",
    "thought",
    "law",
    "rule",
    "order",
    "duty",
    "right",
    "justice",
    "peace",
    "war",
    "battle",
    "power",
    "strength",
    "force",
    "life",
    "death",
    "danger",
    "safety",
    "courage",
    "shame",
    "honor",
    "glory",
    "blessing",
    "curse",
    "prayer",
    "god",
    "angel",
    "devil",
    "faith",
    "grace",
    "mercy",
    "sin",
    "crime",
    "price",
    "value",
    "money",
    "wealth",
    "treasure",
    "debt",
    "gift",
    "reward",
    "punishment",
    "work_labor",
    "job",
    "business",
    "company",
    "society",
    "custom",
    "art",
    "science",
    "magic",
    "sign",
    "symbol",
    "form",
    "manner",
    "system",
    "state",
    "trouble",
    "pity",
    "envy",
    "anger",
    "pride",
    "humility",
    "wisdom",
    "folly",
    "joy",
    "sorrow",
    "number",
    "amount",
    "part",
    "piece",
    "half",
    "quarter",
    "whole",
    "total",
    "pair",
    "group",
    "shape",
    "circle",
    "square",
    "line",
    "point",
    "side",
    "edge",
    "corner",
    "middle",
    "center",
    "front",
    "top",
    "bottom",
    "inside",
    "outside",
    "beginning",
    "size",
    "length",
    "width",
    "height",
    "depth",
    "weight",
    "speed",
    "distance",
    "north",
    "south",
    "east",
    "west",
    "left",
    "right_dir",
    "level",
    "weight_pound",
    "measure",
    "zero",
    "big",
    "small",
    "great",
    "tall",
    "short",
    "high",
    "low",
    "deep",
    "shallow",
    "long",
    "broad",
    "narrow",
    "thick",
    "thin",
    "heavy",
    "light_weight",
    "hard",
    "soft",
    "rough",
    "smooth",
    "sharp",
    "dull",
    "straight",
    "crooked",
    "round",
    "flat",
    "hot",
    "warm",
    "cool",
    "dry",
    "wet",
    "bright",
    "dark",
    "clear",
    "cloudy",
    "loud",
    "quiet",
    "bitter",
    "fresh",
    "stale",
    "clean",
    "dirty",
    "pure",
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "brown",
    "purple",
    "pink",
    "fast",
    "slow",
    "tight",
    "loose",
    "dense",
    "bare",
    "rich",
    "poor",
    "good",
    "excellent",
    "bad",
    "terrible",
    "evil",
    "holy",
    "happy",
    "sad",
    "angry",
    "calm",
    "afraid",
    "brave",
    "proud",
    "humble",
    "kind",
    "cruel",
    "wise",
    "smart",
    "foolish",
    "honest",
    "false",
    "true",
    "right_correct",
    "wrong",
    "safe",
    "dangerous",
    "easy",
    "difficult",
    "simple",
    "complex",
];

/// Synonym clusters for each canonical Yumon Talk word.
pub static SYNONYM_TABLE: [(&str, &[&str]); 965] = [
    ("i", &["i", "me", "myself"]),
    ("you", &["you", "thee", "thou", "yourself", "yourselves"]),
    ("he", &["he", "him", "himself"]),
    ("she", &["she", "her", "herself"]),
    ("it", &["it", "itself"]),
    ("we", &["we", "ourselves", "us"]),
    ("they", &["they", "them", "themselves"]),
    ("my", &["my", "mine"]),
    ("your", &["your", "thine", "thy", "yours"]),
    ("his", &["his"]),
    ("her", &["her", "hers"]),
    ("its", &["its"]),
    ("our", &["our", "ours"]),
    ("their", &["their", "theirs"]),
    ("this", &["this"]),
    ("that", &["that"]),
    ("these", &["these"]),
    ("those", &["those"]),
    ("who", &["who", "whoever", "whom"]),
    ("whose", &["whose"]),
    ("which", &["which", "whichever"]),
    ("what", &["what", "whatever"]),
    ("all", &["all", "complete", "entire", "everybody", "everyone", "everything", "whole"]),
    ("every", &["every", "each"]),
    ("any", &["any", "anybody", "anyhow", "anyone", "anything", "anyway"]),
    ("some", &["some", "somebody", "someone", "something", "somewhat"]),
    ("none", &["none", "neither", "nobody", "nothing"]),
    ("both", &["both"]),
    ("few", &["few", "infrequent", "scant", "sparse"]),
    ("many", &["many", "countless", "innumerable", "multiple", "numerous"]),
    ("much", &["much", "plenty"]),
    ("more", &["more", "additional", "extra", "further"]),
    ("most", &["most", "utmost"]),
    ("other", &["other", "alternate", "another", "different", "else"]),
    ("same", &["same", "equal", "equivalent", "identical", "matching"]),
    ("such", &["such"]),
    ("the", &["the"]),
    ("a", &["a", "an"]),
    ("and", &["and", "also", "furthermore", "moreover", "plus"]),
    ("or", &["or", "either"]),
    ("but", &["but", "although", "however", "nevertheless", "though", "yet"]),
    ("so", &["so", "consequently", "hence", "therefore", "thus"]),
    ("because", &["because", "forasmuch", "seeing", "since"]),
    ("if", &["if", "assuming", "provided", "supposing"]),
    ("then", &["then", "afterwards", "subsequently"]),
    ("than", &["than"]),
    ("as", &["as", "similarly"]),
    ("while", &["while", "whilst"]),
    ("unless", &["unless"]),
    ("until", &["until", "till"]),
    ("in", &["in", "inward", "inwards", "within"]),
    ("on", &["on", "atop", "upon"]),
    ("at", &["at"]),
    ("by", &["by", "alongside", "beside"]),
    ("for", &["for"]),
    ("with", &["with", "accompanied", "along"]),
    ("without", &["without", "devoid", "lacking"]),
    ("about", &["about", "concerning", "regarding", "respecting"]),
    ("against", &["against", "contrary", "opposed", "versus"]),
    ("between", &["between", "betwixt", "midst"]),
    ("among", &["among", "amid", "amidst", "amongst"]),
    ("through", &["through", "across", "throughout"]),
    ("during", &["during"]),
    ("before", &["before", "ahead", "earlier", "formerly", "prior"]),
    ("after", &["after", "behind", "following", "subsequent"]),
    ("above", &["above", "higher", "overhead"]),
    ("below", &["below", "beneath", "lower", "underneath"]),
    ("under", &["under"]),
    ("over", &["over"]),
    ("into", &["into"]),
    ("onto", &["onto"]),
    ("out", &["out", "outdoors", "outside", "outward"]),
    ("up", &["up", "aloft", "upward", "upwards"]),
    ("down", &["down", "downward", "downwards"]),
    ("from", &["from"]),
    ("to", &["to", "toward", "towards"]),
    ("off", &["off"]),
    ("be", &["be", "been", "being"]),
    ("is", &["is", "am", "are", "art"]),
    ("was", &["was", "wast", "were"]),
    ("have", &["have", "had", "has", "hath", "having"]),
    ("do", &["do", "did", "didst", "does", "doing", "done", "doth"]),
    ("will", &["will", "wilt", "would"]),
    ("shall", &["shall", "shalt", "should"]),
    ("can", &["can", "able", "capable", "could"]),
    ("may", &["may", "might"]),
    ("must", &["must", "obliged", "ought", "required"]),
    ("not", &["not", "n't", "non"]),
    ("never", &["never"]),
    ("no", &["no", "nay", "negative"]),
    ("yes", &["yes", "affirmative", "aye", "certainly", "indeed", "yeah", "yep"]),
    ("hello", &["hello", "greetings", "hey", "hi", "howdy", "welcome"]),
    ("goodbye", &["goodbye", "adieu", "bye", "farewell"]),
    ("please", &["please", "kindly"]),
    ("thank", &["thank", "appreciate", "appreciated", "grateful", "gratitude", "thankful", "thanks"]),
    ("sorry", &["sorry", "apologize", "apology", "excuse", "forgive", "pardon", "regret"]),
    ("ok", &["ok", "agreed", "alright", "cool", "fine", "okay"]),
    ("well", &["well"]),
    ("oh", &["oh", "ah", "alas", "ooh", "wow"]),
    ("amen", &["amen", "truly", "verily"]),
    ("where", &["where", "wherever"]),
    ("when", &["when", "whenever"]),
    ("why", &["why", "wherefore"]),
    ("how", &["how", "however"]),
    ("very", &["very", "deeply", "exceedingly", "extremely", "greatly", "highly"]),
    ("really", &["really", "actually", "genuinely", "truly"]),
    ("quite", &["quite", "fairly", "pretty", "rather"]),
    ("almost", &["almost", "nearly", "practically", "virtually"]),
    ("too", &["too", "excessively", "overly"]),
    ("even", &["even"]),
    ("just", &["just", "barely", "merely", "simply"]),
    ("only", &["only", "exclusively", "solely"]),
    ("enough", &["enough", "adequate", "sufficient"]),
    ("again", &["again", "anew", "once more", "repeatedly"]),
    ("already", &["already"]),
    ("still", &["still", "yet"]),
    ("now", &["now", "currently", "nowadays", "presently"]),
    ("soon", &["soon", "presently", "promptly", "shortly"]),
    ("here", &["here", "hither"]),
    ("there", &["there", "thither", "yonder"]),
    ("together", &["together", "collectively", "jointly"]),
    ("alone", &["alone", "isolated", "sole", "solitary"]),
    ("always", &["always", "continually", "ever", "forever", "perpetually"]),
    ("often", &["often", "frequently", "regularly", "repeatedly"]),
    ("sometimes", &["sometimes", "at times", "occasionally"]),
    ("perhaps", &["perhaps", "maybe", "peradventure", "possibly"]),
    ("suddenly", &["suddenly", "abruptly", "unexpectedly"]),
    ("finally", &["finally", "at last", "eventually", "ultimately"]),
    ("instead", &["instead", "alternatively", "rather"]),
    ("almost_degree", &["almost_degree", "nigh"]),
    ("beside", &["beside", "aside", "next to"]),
    ("beyond", &["beyond", "further than", "past"]),
    ("except", &["except", "apart from", "excluding", "save"]),
    ("seldom", &["seldom", "hardly", "rarely", "scarcely"]),
    ("away", &["away", "distant", "elsewhere", "off"]),
    ("time", &["time", "duration", "interval", "period"]),
    ("day", &["day", "daily", "daytime"]),
    ("night", &["night", "nightly", "nighttime"]),
    ("morning", &["morning", "dawn", "daybreak", "morn", "sunrise"]),
    ("evening", &["evening", "dusk", "eve", "sunset", "twilight"]),
    ("noon", &["noon", "midday"]),
    ("hour", &["hour"]),
    ("minute", &["minute"]),
    ("second", &["second", "flash", "instant", "moment"]),
    ("week", &["week", "weekly"]),
    ("month", &["month", "monthly"]),
    ("year", &["year", "annual", "yearly"]),
    ("season", &["season"]),
    ("spring_time", &["spring_time", "spring", "springtime"]),
    ("summer", &["summer", "summertime"]),
    ("autumn", &["autumn", "fall", "harvest"]),
    ("winter", &["winter", "wintertime"]),
    ("today", &["today"]),
    ("yesterday", &["yesterday"]),
    ("tomorrow", &["tomorrow"]),
    ("age", &["age", "eon", "epoch", "era", "generation"]),
    ("past", &["past", "history", "yesteryear"]),
    ("present", &["present", "current", "nowadays"]),
    ("future", &["future", "destiny", "hereafter"]),
    ("early", &["early", "ahead of time", "premature"]),
    ("late", &["late", "belated", "overdue", "tardy"]),
    ("date", &["date", "calendar", "day of month"]),
    ("forever", &["forever", "eternal", "everlasting", "perpetual"]),
    ("say", &["say", "declare", "declared", "declares", "said", "saying", "says", "speak", "speaking", "speaks", "spoke", "spoken", "state", "stated", "states", "stating", "talk", "talked", "talking", "talks", "tell", "telling", "tells", "told", "utter", "uttered", "utters", "vocalize"]),
    ("ask", &["ask", "asked", "asking", "asks", "demand", "demanded", "demands", "inquire", "inquired", "inquires", "inquiry", "queries", "query", "question", "questioned", "questioning", "questions"]),
    ("answer", &["answer", "answered", "answering", "answers", "replied", "replies", "reply", "replying", "respond", "responded", "responding", "responds", "response", "responses"]),
    ("call", &["call", "called", "calling", "calls", "invoke", "invoked", "invokes", "name", "named", "names", "naming", "summon", "summoned", "summoning", "summons"]),
    ("cry", &["cry", "bawl", "bawled", "cried", "cries", "crying", "scream", "screamed", "screaming", "screams", "shout", "shouted", "shouting", "shouts", "wail", "wailed", "wails", "weep", "weeping", "weeps", "wept", "yell", "yelled", "yelling", "yells"]),
    ("whisper", &["whisper", "murmur", "murmured", "murmurs", "mutter", "muttered", "mutters", "whispered", "whispering", "whispers"]),
    ("explain", &["explain", "clarified", "clarifies", "clarify", "demonstrate", "describe", "described", "describes", "elucidate", "explained", "explaining", "explains", "expound", "illustrate"]),
    ("promise", &["promise", "pledge", "pledged", "pledges", "promised", "promises", "promising", "swear", "swears", "swore", "sworn", "vow", "vowed", "vows"]),
    ("praise", &["praise", "applaud", "commend", "compliment", "extol", "glorified", "glorifies", "glorify", "laud", "praised", "praises", "praising"]),
    ("blame", &["blame", "accuse", "accused", "accuses", "blamed", "blames", "blaming", "censure", "criticize", "fault", "rebuke", "reprimand", "reproach", "scold"]),
    ("warn", &["warn", "admonish", "alert", "caution", "cautioned", "cautions", "warned", "warning", "warns"]),
    ("greet", &["greet", "greeted", "greeting", "greets", "hail", "salute", "welcome"]),
    ("argue", &["argue", "argued", "argues", "arguing", "bicker", "debated", "dispute", "disputes"]),
    ("discuss", &["discuss", "confer", "consult", "debate", "discussed", "discusses", "discussing"]),
    ("advise", &["advise", "advised", "advises", "advising", "counsel", "recommend", "suggest"]),
    ("refuse", &["refuse", "decline", "declined", "declines", "denied", "denies", "deny", "refused", "refuses", "refusing", "reject", "rejected", "rejects"]),
    ("agree", &["agree", "accept", "accepted", "accepts", "agreed", "agreeing", "agrees", "approve", "approved", "concur"]),
    ("command", &["command", "commanded", "commanding", "commands", "decree", "order", "ordered", "orders", "rule"]),
    ("obey", &["obey", "complied", "complies", "comply", "conform", "obeyed", "obeying", "obeys", "submit"]),
    ("see", &["see", "beheld", "behold", "gaze", "gazed", "gazes", "glance", "glimpse", "look", "looked", "looking", "looks", "notice", "noticed", "notices", "observe", "observed", "observes", "observing", "saw", "seeing", "seen", "sees", "spot", "spots", "spotted", "stare", "stared", "stares", "view", "viewed", "viewing", "views", "watch", "watched", "watches", "watching", "witness", "witnessed", "witnesses"]),
    ("hear", &["hear", "heard", "hearing", "hearken", "hears", "listen", "listened", "listening", "listens", "overhear"]),
    ("feel", &["feel", "feeling", "feels", "felt", "perceive", "perceived", "perceives", "sense", "sensed", "senses", "touch", "touched", "touches", "touching"]),
    ("smell", &["smell", "aroma", "fragrance", "odor", "scent", "smelled", "smells", "smelt", "sniff"]),
    ("taste", &["taste", "flavor", "relish", "savor", "tasted", "tastes", "tasting"]),
    ("think", &["think", "contemplate", "deliberate", "meditate", "ponder", "pondered", "ponders", "reason", "reflect", "reflected", "reflects", "thinking", "thinks", "thought"]),
    ("know", &["know", "cognizant", "familiar", "knew", "knowing", "knowledge", "known", "knows"]),
    ("understand", &["understand", "comprehend", "comprehended", "comprehends", "fathom", "grasp", "grasped", "grasps", "understanding", "understands", "understood"]),
    ("remember", &["remember", "memorize", "recall", "recalled", "recalls", "recollect", "remembered", "remembering", "remembers", "remembrance"]),
    ("forget", &["forget", "forgets", "forgetting", "forgot", "forgotten", "neglect", "overlook"]),
    ("learn", &["learn", "absorb", "acquire", "learned", "learning", "learns", "learnt", "master", "studied", "studies", "study", "studying"]),
    ("teach", &["teach", "educate", "guide", "instruct", "instructed", "instructs", "taught", "teaches", "teaching", "train", "tutor"]),
    ("believe", &["believe", "believed", "believes", "believing", "confide", "credit", "faith", "trust", "trusted", "trusting", "trusts"]),
    ("doubt", &["doubt", "disbelieve", "distrust", "doubted", "doubting", "doubts", "hesitate", "skeptical", "suspect"]),
    ("hope", &["hope", "aspire", "desire", "hoped", "hopes", "hoping", "long", "wish", "wished", "wishes", "wishing", "yearn"]),
    ("wonder", &["wonder", "amaze", "astonish", "marvel", "marveled", "marvels", "wondered", "wondering", "wonders"]),
    ("dream", &["dream", "daydream", "dreamed", "dreaming", "dreams", "dreamt", "fantasy", "nightmare", "vision"]),
    ("imagine", &["imagine", "conceive", "envision", "fancied", "imagined", "imagines", "imagining", "picture", "visualize"]),
    ("decide", &["decide", "decided", "decides", "deciding", "determine", "determined", "determines", "resolve", "settle"]),
    ("choose", &["choose", "chooses", "choosing", "chose", "chosen", "opt", "pick", "picked", "picks", "prefer", "select", "selected", "selecting", "selects"]),
    ("judge", &["judge", "appraise", "assess", "evaluate", "judged", "judges", "judging", "verdict"]),
    ("read", &["read", "decipher", "peruse", "reading", "reads", "scan"]),
    ("write", &["write", "compose", "draft", "inscribe", "jot", "pen", "record", "writes", "writing", "written", "wrote"]),
    ("go", &["go", "advance", "advanced", "advances", "goes", "going", "gone", "head", "headed", "journey", "move", "moved", "moves", "moving", "pass", "passed", "passes", "proceed", "proceeded", "proceeds", "travel", "traveled", "travelling", "travels", "went"]),
    ("come", &["come", "approach", "approached", "approaches", "arrive", "arrived", "arrives", "arriving", "came", "comes", "coming", "near", "nearing", "nears", "reach", "reached", "reaches", "reaching"]),
    ("walk", &["walk", "march", "marched", "marches", "pace", "paced", "paces", "step", "stepped", "steps", "stroll", "tread", "trod", "walked", "walking", "walks", "wander", "wandered"]),
    ("run", &["run", "dash", "dashed", "dashes", "hasten", "hurried", "hurries", "hurry", "jog", "race", "raced", "races", "ran", "running", "runs", "rush", "rushed", "rushes", "sprint", "sprinted", "sprints"]),
    ("jump", &["jump", "bound", "bounds", "hop", "hopped", "hops", "jumped", "jumping", "jumps", "leap", "leaped", "leaps", "leapt", "skip", "skipped", "skips", "sprang", "spring", "springs"]),
    ("climb", &["climb", "ascend", "ascended", "ascends", "climbed", "climbing", "climbs", "mount", "mounted", "mounts", "scale"]),
    ("fly", &["fly", "flew", "flies", "flight", "flown", "flying", "glide", "glided", "glides", "hover", "hovered", "soar", "soared", "soars"]),
    ("swim", &["swim", "dive", "dived", "dives", "dove", "float", "floated", "floats", "swam", "swimming", "swims", "swum"]),
    ("drive", &["drive", "driven", "drives", "driving", "drove", "ridden", "ride", "rides", "riding", "rode", "steer"]),
    ("fall", &["fall", "collapse", "collapsed", "drop", "dropped", "dropping", "drops", "fallen", "falling", "falls", "fell", "sank", "sink", "sinks", "sunk", "tumble", "tumbled"]),
    ("rise", &["rise", "arise", "arisen", "arises", "arose", "elevate", "lift", "lifted", "lifts", "risen", "rises", "rising", "rose", "soar", "surge"]),
    ("stand", &["stand", "erect", "standing", "stands", "stood", "upright"]),
    ("sit", &["sit", "perch", "perched", "sat", "seat", "seated", "sits", "sitting"]),
    ("lie", &["lie", "lain", "lay", "lies", "lying", "recline", "reclined", "reclines", "rest", "rested", "resting", "rests"]),
    ("sleep", &["sleep", "asleep", "dormant", "nap", "napped", "naps", "sleeping", "sleeps", "slept", "slumber", "snooze"]),
    ("wake", &["wake", "arouse", "awake", "awaken", "awakened", "awakens", "wakes", "waking", "woke", "woken"]),
    ("stay", &["stay", "abide", "abides", "abode", "dwell", "dwells", "dwelt", "lodge", "remain", "remained", "remains", "stayed", "staying", "stays", "tarry"]),
    ("leave", &["leave", "abandon", "abandoned", "depart", "departed", "departing", "departs", "escape", "escaped", "exit", "exited", "exits", "fled", "flee", "flees", "leaves", "leaving", "left", "quit"]),
    ("return", &["return", "rebound", "retreat", "retreated", "retreats", "returned", "returning", "returns", "revert"]),
    ("stop", &["stop", "cease", "ceased", "ceases", "conclude", "end", "ended", "ends", "halt", "halted", "halts", "pause", "paused", "pauses", "stopped", "stopping", "stops"]),
    ("wait", &["wait", "await", "awaited", "awaits", "delay", "delayed", "delays", "waited", "waiting", "waits"]),
    ("follow", &["follow", "chase", "chased", "chases", "followed", "following", "follows", "pursue", "pursued", "pursues", "pursuit", "shadow", "track", "trail"]),
    ("lead", &["lead", "conduct", "direct", "directed", "directs", "guide", "guided", "guides", "leading", "leads", "led", "pilot"]),
    ("shake", &["shake", "quiver", "shaken", "shakes", "shaking", "shiver", "shook", "tremble", "trembled", "trembles", "vibrate"]),
    ("hang", &["hang", "dangle", "hanged", "hanging", "hangs", "hung", "suspend", "suspended"]),
    ("roll", &["roll", "rolled", "rolling", "rolls", "rotate", "spin", "spins", "spun", "wheel"]),
    ("turn", &["turn", "pivot", "turned", "turning", "turns", "twist", "twisted", "twists"]),
    ("make", &["make", "construct", "constructed", "constructs", "craft", "create", "created", "creates", "creating", "forge", "forged", "form", "formed", "forms", "made", "makes", "making", "manufacture", "produce", "produced", "produces"]),
    ("build", &["build", "assemble", "assembled", "building", "builds", "built", "erect", "erected", "structure"]),
    ("break", &["break", "breaking", "breaks", "broke", "broken", "burst", "bursts", "crack", "cracked", "cracks", "fracture", "shatter", "shattered", "smash", "smashed", "smashes", "snap", "snapped"]),
    ("destroy", &["destroy", "annihilate", "crush", "crushed", "demolish", "demolished", "destroyed", "destroying", "destroys", "devastate", "ruin", "ruined", "ruins", "wreck", "wrecked"]),
    ("fix", &["fix", "cure", "cured", "fixed", "fixes", "fixing", "heal", "healed", "mend", "mended", "mends", "repair", "repaired", "repairing", "repairs", "restore", "restored", "restores"]),
    ("cut", &["cut", "carve", "carved", "chop", "chopped", "chops", "cuts", "cutting", "sever", "severed", "severs", "slash", "slice", "sliced", "slices", "split", "splits"]),
    ("open", &["open", "opened", "opening", "opens", "uncover", "unlock", "unlocked", "unlocks", "unseal"]),
    ("close", &["close", "closed", "closes", "closing", "lock", "locked", "locks", "seal", "sealed", "seals", "shut", "shuts", "shutting"]),
    ("cover", &["cover", "blanket", "covered", "covering", "covers", "envelop", "shield", "wrap", "wrapped", "wraps"]),
    ("hide", &["hide", "conceal", "concealed", "conceals", "disguise", "hid", "hidden", "hides", "hiding", "mask", "masked", "masks"]),
    ("show", &["show", "display", "displayed", "displays", "exhibit", "expose", "manifest", "reveal", "revealed", "reveals", "showed", "showing", "shown", "shows"]),
    ("give", &["give", "bestow", "bestowed", "donate", "donated", "donates", "gave", "given", "gives", "giving", "grant", "granted", "granting", "grants", "offer", "offered", "offering", "offers", "present", "presented", "presents", "provide", "provided", "provides", "providing", "supplied", "supplies", "supply"]),
    ("take", &["take", "capture", "captured", "catch", "catches", "caught", "clutch", "grab", "grabbed", "grabbing", "grabs", "seize", "seized", "seizes", "seizing", "snatch", "snatched", "snatches", "taken", "takes", "taking", "took"]),
    ("hold", &["hold", "clasp", "grip", "held", "holding", "holds", "keep", "keeping", "keeps", "kept", "maintain", "possess", "retain", "retained", "retains"]),
    ("carry", &["carry", "bear", "bears", "bore", "borne", "carried", "carries", "carrying", "cart", "convey", "haul", "tote", "transport"]),
    ("bring", &["bring", "bringing", "brings", "brought", "deliver", "delivered", "delivers", "fetch", "fetched", "fetches", "fetching"]),
    ("send", &["send", "dispatch", "dispatched", "dispatches", "forward", "mail", "sending", "sends", "sent", "ship", "transmit"]),
    ("put", &["put", "deposit", "install", "laid", "lay", "laying", "lays", "place", "placed", "places", "placing", "position", "puts", "putting", "set", "sets", "setting"]),
    ("drop", &["drop", "cast", "casting", "casts", "dropped", "dropping", "drops", "shed"]),
    ("throw", &["throw", "fling", "flung", "hurl", "hurled", "pitch", "threw", "throwing", "thrown", "throws", "toss", "tossed", "tosses"]),
    ("pull", &["pull", "drag", "dragged", "dragging", "drags", "draw", "drawing", "drawn", "draws", "drew", "pulled", "pulling", "pulls", "tow", "tug", "tugged", "tugs"]),
    ("push", &["push", "press", "pressed", "presses", "propel", "pushed", "pushes", "pushing", "shove", "shoved", "shoves", "thrust", "thrusts"]),
    ("find", &["find", "detect", "discover", "discovered", "discovering", "discovers", "encounter", "finding", "finds", "found", "locate", "located", "locates", "uncover"]),
    ("lose", &["lose", "forfeit", "loses", "losing", "lost", "misplace", "misplaced", "misplaces"]),
    ("seek", &["seek", "explore", "explored", "explores", "hunt", "hunted", "hunting", "hunts", "search", "searched", "searches", "searching", "seeking", "seeks", "sought"]),
    ("help", &["help", "aid", "aided", "aids", "assist", "assisted", "assisting", "assists", "helped", "helping", "helps", "serve", "served", "serves", "service", "succor", "support", "supported", "supports"]),
    ("hurt", &["hurt", "damage", "damaged", "damages", "harm", "harmed", "harming", "harms", "hurting", "hurts", "injure", "injured", "injures", "pain", "pained", "pains", "wound", "wounded", "wounds"]),
    ("heal", &["heal", "cure", "cured", "cures", "curing", "healed", "healing", "heals", "recover", "recovered", "recovers", "remedy", "soothe"]),
    ("kill", &["kill", "assassinate", "execute", "killed", "killing", "kills", "murder", "murdered", "murders", "slain", "slaughter", "slay", "slays", "slew"]),
    ("die", &["die", "decease", "died", "dies", "dying", "expire", "expired", "expires", "perish", "perished", "perishes"]),
    ("live", &["live", "dwell", "exist", "existed", "exists", "lived", "lives", "living", "reside", "survive", "survived", "survives", "thrive"]),
    ("eat", &["eat", "ate", "bit", "bite", "bites", "bitten", "chew", "chewed", "consume", "consumed", "consumes", "devour", "devoured", "devours", "dine", "dined", "eaten", "eating", "eats", "fed", "feed", "feeding", "feeds", "snack", "swallow", "swallowed"]),
    ("drink", &["drink", "beverage", "drank", "draught", "drinking", "drinks", "drunk", "gulp", "quench", "sip", "sipped", "sips"]),
    ("wear", &["wear", "clothe", "clothed", "clothes", "don", "donned", "dons", "dress", "dressed", "dresses", "wearing", "wears", "wore", "worn"]),
    ("wash", &["wash", "bathe", "clean", "cleaned", "cleaning", "cleans", "cleanse", "purify", "rinse", "scrub", "washed", "washes", "washing"]),
    ("burn", &["burn", "blaze", "blazed", "blazes", "burned", "burning", "burns", "burnt", "char", "combust", "ignite", "ignited", "scorch"]),
    ("fight", &["fight", "battle", "battled", "battles", "brawl", "clash", "clashed", "combat", "fighting", "fights", "fought", "quarrel", "struggle", "struggled", "struggles", "wrestle", "wrestled"]),
    ("win", &["win", "conquer", "conquered", "prevail", "prevailed", "prevails", "succeed", "succeeded", "succeeds", "triumph", "triumphed", "triumphs", "victory", "winning", "wins", "won"]),
    ("fail", &["fail", "defeat", "defeated", "defeats", "failed", "failing", "fails", "surrender", "yield", "yielded", "yields"]),
    ("work", &["work", "function", "functioning", "functions", "labor", "labored", "labors", "operate", "operated", "operates", "task", "toil", "toiled", "toils", "worked", "working", "works"]),
    ("play", &["play", "amuse", "entertain", "frolic", "game", "games", "played", "playing", "plays", "recreate", "sport"]),
    ("try", &["try", "attempt", "attempted", "attempts", "endeavor", "strive", "strives", "strove", "tried", "tries", "trying"]),
    ("use", &["use", "applied", "applies", "apply", "employ", "employed", "employs", "used", "uses", "using", "utilize", "utilized", "utilizes"]),
    ("buy", &["buy", "acquire", "bought", "buying", "buys", "purchase", "purchased", "purchases"]),
    ("sell", &["sell", "market", "selling", "sells", "sold", "trade", "traded", "trades", "vend"]),
    ("pay", &["pay", "cost", "costs", "disburse", "fee", "paid", "paying", "pays", "spend", "spending", "spends", "spent"]),
    ("save", &["save", "deliver", "preserve", "preserved", "preserves", "protect", "protected", "protecting", "protects", "rescue", "rescued", "rescues", "rescuing", "salvage", "saved", "saves", "saving", "spare"]),
    ("guard", &["guard", "defend", "defended", "defending", "defends", "guarded", "guarding", "guards", "patrol"]),
    ("attack", &["attack", "assault", "assaulted", "assaults", "attacked", "attacking", "attacks", "beat", "beating", "beats", "hit", "hits", "hitting", "invade", "invaded", "strike", "strikes", "struck"]),
    ("allow", &["allow", "admit", "allowed", "allowing", "allows", "consent", "let", "lets", "letting", "permit", "permits", "permitted"]),
    ("love", &["love", "adore", "adored", "adores", "affection", "beloved", "cherish", "cherished", "cherishes", "fondness", "loved", "loves", "loving"]),
    ("hate", &["hate", "despise", "despised", "despises", "detest", "detested", "detests", "dislike", "hated", "hates", "hating", "loathe", "loathed", "loathes"]),
    ("fear", &["fear", "afraid", "alarm", "dread", "dreaded", "dreads", "feared", "fearing", "fears", "fright", "frightened", "horror", "panic", "scared", "terrified", "terror"]),
    ("care", &["care", "cared", "cares", "caring", "concern", "concerned", "concerns", "heed", "mind", "minded", "minds", "regard"]),
    ("like", &["like", "enjoy", "enjoyed", "enjoying", "enjoys", "fancy", "liked", "likes", "liking"]),
    ("want", &["want", "crave", "craved", "craves", "desire", "desired", "desires", "need", "needed", "needing", "needs", "require", "required", "requires", "wanted", "wanting", "wants"]),
    ("start", &["start", "began", "begin", "beginning", "begins", "begun", "commence", "commenced", "commences", "initiate", "initiated", "launch", "launched", "started", "starting", "starts"]),
    ("end", &["end", "complete", "completed", "completes", "completing", "conclude", "concluded", "ended", "ending", "ends", "finish", "finished", "finishes", "finishing", "terminate"]),
    ("continue", &["continue", "continued", "continues", "continuing", "endure", "endured", "endures", "last", "lasted", "lasts", "persist", "persisted", "persists", "proceed", "proceeds", "resume"]),
    ("change", &["change", "alter", "altered", "alters", "changed", "changes", "changing", "convert", "modified", "modifies", "modify", "shift", "transform", "transformed", "vary"]),
    ("grow", &["grow", "develop", "developed", "develops", "expand", "expanded", "expands", "flourish", "grew", "growing", "grown", "grows", "increase", "increased", "increases"]),
    ("shrink", &["shrink", "contract", "decrease", "decreased", "decreases", "diminish", "diminished", "dwindle", "reduce", "reduced", "reduces", "shrank", "shrinks", "shrunk"]),
    ("shine", &["shine", "beam", "flare", "flash", "gleam", "glitter", "glow", "glowed", "glowing", "glows", "radiate", "shined", "shines", "shining", "shone", "sparkle"]),
    ("appear", &["appear", "appeared", "appearing", "appears", "emerge", "emerged", "emerges", "seem", "seemed", "seeming", "seems"]),
    ("disappear", &["disappear", "disappeared", "disappears", "fade", "faded", "fades", "vanish", "vanished", "vanishes"]),
    ("happen", &["happen", "befall", "chance", "happened", "happening", "happens", "occur", "occurred", "occurring", "occurs", "transpire"]),
    ("become", &["become", "became", "becomes", "becoming"]),
    ("draw", &["draw", "depict", "drawn", "draws", "drew", "illustrate", "paint", "painted", "painting", "paints", "sketch"]),
    ("sing", &["sing", "carol", "chant", "chanted", "chants", "hum", "hummed", "hums", "sang", "singing", "sings", "sung"]),
    ("laugh", &["laugh", "chuckle", "chuckled", "chuckles", "giggle", "giggled", "giggles", "grin", "grinned", "grins", "laughed", "laughing", "laughs", "snicker"]),
    ("smile", &["smile", "beam", "smiled", "smiles", "smiling"]),
    ("pour", &["pour", "flow", "flowed", "flowing", "flows", "poured", "pouring", "pours", "stream", "streamed", "streams"]),
    ("fill", &["fill", "filled", "filling", "fills", "load", "loaded", "pack", "packed"]),
    ("tie", &["tie", "bind", "binding", "binds", "bound", "fasten", "knot", "secure", "tied", "ties", "tying"]),
    ("release", &["release", "free", "freed", "frees", "loosen", "released", "releases", "releasing", "unbind"]),
    ("shoot", &["shoot", "blast", "discharge", "fire", "fired", "fires", "firing", "shooting", "shoots", "shot"]),
    ("blow", &["blow", "blew", "blowing", "blown", "blows", "breeze", "exhale", "gust", "puff"]),
    ("suffer", &["suffer", "bear", "endure", "suffered", "suffering", "suffers", "tolerate"]),
    ("rejoice", &["rejoice", "celebrate", "celebrated", "celebrates", "exult", "rejoiced", "rejoices", "rejoicing"]),
    ("mourn", &["mourn", "grieve", "grieved", "grieves", "lament", "mourned", "mourning", "mourns"]),
    ("strike", &["strike", "pound", "slap", "smite", "strikes", "striking", "struck"]),
    ("touch", &["touch", "contact", "handle", "touched", "touches", "touching"]),
    ("meet", &["meet", "encounter", "encountered", "encounters", "meeting", "meets", "met"]),
    ("bend", &["bend", "bending", "bends", "bent", "bow", "bowed", "bows", "curve", "curved", "curves", "flex"]),
    ("fold", &["fold", "crease", "folded", "folding", "folds", "pleat"]),
    ("spread", &["spread", "disperse", "distribute", "extend", "scatter", "spreading", "spreads"]),
    ("gather", &["gather", "assemble", "collect", "collected", "collects", "gathered", "gathering", "gathers", "harvest"]),
    ("count", &["count", "calculate", "counted", "counting", "counts", "enumerate", "reckon", "tally"]),
    ("measure_verb", &["measure_verb", "gauge", "quantify", "weigh", "weighed", "weighs"]),
    ("burn_light", &["burn_light", "illuminate", "kindle", "light", "lighting", "lights", "lit"]),
    ("cook", &["cook", "bake", "baked", "bakes", "boil", "cooked", "cooking", "cooks", "fry", "roast"]),
    ("hunt", &["hunt", "hunted", "hunting", "hunts", "pursue", "stalk", "track"]),
    ("catch_trap", &["catch_trap", "ensnare", "entangle", "net"]),
    ("lead_guide", &["lead_guide", "pilot", "shepherd", "steer", "usher"]),
    ("person", &["person", "being", "fellow", "human", "individual", "somebody", "someone", "soul"]),
    ("man", &["man", "chap", "gentleman", "male"]),
    ("woman", &["woman", "dame", "female", "lady"]),
    ("child", &["child", "kid", "offspring", "youth"]),
    ("boy", &["boy", "lad", "schoolboy", "youngster"]),
    ("girl", &["girl", "lass", "maiden", "schoolgirl"]),
    ("baby", &["baby", "babe", "infant", "newborn", "toddler"]),
    ("father", &["father", "dad", "daddy", "papa", "paternal", "sire"]),
    ("mother", &["mother", "mama", "maternal", "mom", "mommy"]),
    ("parent", &["parent", "folks", "guardians"]),
    ("son", &["son"]),
    ("daughter", &["daughter"]),
    ("brother", &["brother", "sibling"]),
    ("sister", &["sister"]),
    ("family", &["family", "clan", "household", "kin", "kindred", "relatives"]),
    ("husband", &["husband", "spouse"]),
    ("wife", &["wife", "bride", "consort", "spouse"]),
    ("friend", &["friend", "ally", "buddy", "chum", "companion", "mate", "pal", "partner"]),
    ("enemy", &["enemy", "adversary", "antagonist", "foe", "nemesis", "opponent", "rival"]),
    ("neighbor", &["neighbor"]),
    ("guest", &["guest", "caller", "visitor"]),
    ("host", &["host"]),
    ("stranger", &["stranger", "alien", "foreigner", "outsider"]),
    ("crowd", &["crowd", "assembly", "gathering", "mob", "multitude", "throng"]),
    ("people", &["people", "citizens", "folk", "populace", "public"]),
    ("king", &["king", "emperor", "monarch", "sovereign"]),
    ("queen", &["queen", "empress"]),
    ("prince", &["prince"]),
    ("princess", &["princess"]),
    ("lord", &["lord", "nobleman", "peer"]),
    ("lady", &["lady", "noblewoman"]),
    ("master", &["master", "boss", "commander", "overseer", "owner"]),
    ("servant", &["servant", "attendant", "helper", "maid", "retainer"]),
    ("slave", &["slave", "bondman", "captive"]),
    ("ruler", &["ruler", "governor", "potentate"]),
    ("leader", &["leader", "captain", "chief", "director", "head"]),
    ("soldier", &["soldier", "fighter", "infantryman", "trooper", "warrior"]),
    ("knight", &["knight", "cavalier", "champion"]),
    ("officer", &["officer"]),
    ("priest", &["priest", "clergyman", "minister", "monk", "pastor", "preacher", "reverend"]),
    ("prophet", &["prophet", "oracle", "seer"]),
    ("teacher", &["teacher", "educator", "instructor", "master", "professor", "tutor"]),
    ("student", &["student", "learner", "pupil", "scholar"]),
    ("doctor", &["doctor", "healer", "medic", "physician", "surgeon"]),
    ("nurse", &["nurse", "caregiver"]),
    ("worker", &["worker", "employee", "hand", "laborer", "workman"]),
    ("farmer", &["farmer", "cultivator", "grower", "peasant", "planter"]),
    ("merchant", &["merchant", "dealer", "seller", "shopkeeper", "trader", "vendor"]),
    ("driver", &["driver", "chauffeur", "motorist"]),
    ("sailor", &["sailor", "mariner", "navigator", "seaman"]),
    ("thief", &["thief", "bandit", "burglar", "pirate", "robber", "stealer"]),
    ("criminal", &["criminal", "culprit", "felon", "outlaw", "villain"]),
    ("hero", &["hero", "champion", "conqueror", "idol"]),
    ("witness", &["witness", "bystander", "onlooker"]),
    ("messenger", &["messenger", "courier", "emissary", "envoy", "herald"]),
    ("author", &["author", "novelist", "poet", "writer"]),
    ("artist", &["artist", "painter", "sculptor"]),
    ("musician", &["musician", "player", "singer"]),
    ("actor", &["actor", "performer", "player"]),
    ("fool", &["fool", "buffoon", "clown", "idiot", "simpleton"]),
    ("giant", &["giant", "colossus", "titan"]),
    ("ancestor", &["ancestor", "forebear", "forefather"]),
    ("generation", &["generation"]),
    ("crowd_mob", &["crowd_mob", "horde", "pack"]),
    ("ally", &["ally", "compatriot", "confederate"]),
    ("citizen", &["citizen", "denizen", "inhabitant", "resident"]),
    ("elder", &["elder", "patriarch", "senior", "veteran"]),
    ("body", &["body", "flesh", "form", "frame", "physique"]),
    ("head", &["head", "cranium", "skull"]),
    ("brain", &["brain", "intellect", "wit"]),
    ("hair", &["hair", "locks", "mane", "tresses"]),
    ("face", &["face", "countenance", "features", "visage"]),
    ("forehead", &["forehead", "brow"]),
    ("eye", &["eye", "eyes", "sight", "vision"]),
    ("ear", &["ear", "ears", "hearing"]),
    ("nose", &["nose", "snout"]),
    ("mouth", &["mouth", "lips", "maw"]),
    ("lip", &["lip"]),
    ("tooth", &["tooth", "fang", "fangs", "teeth", "tusk"]),
    ("tongue", &["tongue"]),
    ("chin", &["chin", "jaw"]),
    ("neck", &["neck", "nape", "throat"]),
    ("shoulder", &["shoulder"]),
    ("arm", &["arm", "bicep", "limb"]),
    ("elbow", &["elbow"]),
    ("wrist", &["wrist"]),
    ("hand", &["hand", "fist", "hands", "palm"]),
    ("finger", &["finger", "digit", "fingers"]),
    ("thumb", &["thumb"]),
    ("chest", &["chest", "breast", "ribs", "torso"]),
    ("heart", &["heart", "cardiac"]),
    ("stomach", &["stomach", "abdomen", "belly", "gut", "tummy"]),
    ("back", &["back", "backbone", "spine"]),
    ("waist", &["waist", "hips"]),
    ("leg", &["leg", "calf", "legs", "thigh"]),
    ("knee", &["knee"]),
    ("foot", &["foot", "feet", "hoof", "paw"]),
    ("toe", &["toe", "toes"]),
    ("skin", &["skin", "coat", "dermis", "hide"]),
    ("bone", &["bone", "bones", "skeleton"]),
    ("blood", &["blood", "vital fluid"]),
    ("flesh", &["flesh", "meat", "muscle"]),
    ("muscle", &["muscle", "brawn", "sinew"]),
    ("nerve", &["nerve", "sinew"]),
    ("soul", &["soul", "psyche", "spirit"]),
    ("spirit", &["spirit", "ghost", "phantom", "shade", "specter"]),
    ("mind", &["mind", "intellect", "psyche", "reason"]),
    ("breath", &["breath", "breathing", "respiration", "wind"]),
    ("voice", &["voice", "tone", "vocal"]),
    ("tear", &["tear", "tears", "weeping"]),
    ("sweat", &["sweat", "perspiration"]),
    ("wound", &["wound", "bruise", "gash", "injury", "lesion"]),
    ("scar", &["scar"]),
    ("disease", &["disease", "ailment", "illness", "infection", "malady", "sickness"]),
    ("pain", &["pain", "ache", "agony", "anguish", "suffering", "torment"]),
    ("health", &["health", "fitness", "vigor", "vitality", "wellness"]),
    ("sleep_state", &["sleep_state", "rest", "slumber"]),
    ("fever", &["fever"]),
    ("cough", &["cough"]),
    ("sigh", &["sigh", "groan", "moan"]),
    ("touch_sense", &["touch_sense", "feeling", "sensation"]),
    ("world", &["world", "earth", "globe", "realm"]),
    ("earth", &["earth", "clay", "dirt", "ground", "land", "soil"]),
    ("universe", &["universe", "cosmos", "creation"]),
    ("sky", &["sky", "firmament", "heavens", "welkin"]),
    ("heaven", &["heaven", "paradise", "skies"]),
    ("sun", &["sun", "solar", "sunlight", "sunshine"]),
    ("moon", &["moon", "lunar", "moonlight"]),
    ("star", &["star", "constellation", "stars", "stellar"]),
    ("planet", &["planet"]),
    ("space", &["space", "outer space", "void"]),
    ("light", &["light", "brightness", "daylight", "glow", "illumination", "radiance"]),
    ("darkness", &["darkness", "dark", "gloom", "obscurity", "shadow"]),
    ("shadow", &["shadow", "shade"]),
    ("cloud", &["cloud", "clouds", "overcast"]),
    ("rain", &["rain", "downpour", "precipitation", "rainfall", "shower"]),
    ("snow", &["snow", "blizzard", "snowfall", "snowflake"]),
    ("ice", &["ice", "frost", "glacier", "icicle"]),
    ("wind", &["wind", "breeze", "draft", "gale", "gust"]),
    ("storm", &["storm", "hurricane", "tempest", "thunderstorm", "tornado"]),
    ("thunder", &["thunder"]),
    ("lightning", &["lightning", "flash", "thunderbolt"]),
    ("fire", &["fire", "blaze", "flames", "inferno"]),
    ("flame", &["flame"]),
    ("smoke", &["smoke", "fumes", "haze"]),
    ("ash", &["ash", "ashes", "cinders", "embers"]),
    ("water", &["water", "aqua", "fluid", "liquid"]),
    ("sea", &["sea", "brine", "deep", "ocean"]),
    ("river", &["river", "brook", "creek", "stream", "waterway"]),
    ("lake", &["lake", "loch", "pond"]),
    ("wave", &["wave", "billow", "breaker", "surf", "waves"]),
    ("tide", &["tide"]),
    ("flood", &["flood", "deluge", "inundation"]),
    ("spring", &["spring", "fountain", "geyser", "well"]),
    ("mountain", &["mountain", "alp", "mount", "peak", "summit"]),
    ("hill", &["hill", "hillside", "knoll", "mound"]),
    ("valley", &["valley", "canyon", "dell", "glen", "ravine", "vale"]),
    ("cliff", &["cliff", "bluff", "crag", "precipice"]),
    ("cave", &["cave", "cavern", "grotto"]),
    ("desert", &["desert", "solitude", "wasteland", "wilderness"]),
    ("island", &["island", "atoll", "isle"]),
    ("coast", &["coast", "beach", "seaside", "shore", "strand"]),
    ("plain", &["plain", "grassland", "prairie", "savanna", "steppe"]),
    ("field", &["field", "meadow", "paddock", "pasture"]),
    ("forest", &["forest", "grove", "jungle", "timberland", "woods"]),
    ("tree", &["tree", "sapling", "trees"]),
    ("branch", &["branch", "bough", "limb", "twig"]),
    ("leaf", &["leaf", "foliage", "leaves"]),
    ("root", &["root", "roots"]),
    ("bark", &["bark"]),
    ("wood", &["wood", "lumber", "timber"]),
    ("plant", &["plant", "flora", "greenery", "vegetation"]),
    ("grass", &["grass", "lawn", "sod", "turf"]),
    ("flower", &["flower", "bloom", "blossom", "petal"]),
    ("rose", &["rose"]),
    ("lily", &["lily"]),
    ("weed", &["weed"]),
    ("bush", &["bush", "hedge", "shrub"]),
    ("seed", &["seed", "grain", "kernel", "pip"]),
    ("fruit", &["fruit", "produce"]),
    ("stone", &["stone", "boulder", "pebble", "rock"]),
    ("sand", &["sand", "dune"]),
    ("dust", &["dust", "powder"]),
    ("mud", &["mud", "mire", "muck", "slime"]),
    ("gold", &["gold", "gilded"]),
    ("silver", &["silver"]),
    ("iron", &["iron", "steel"]),
    ("copper", &["copper", "brass", "bronze"]),
    ("metal", &["metal"]),
    ("mineral", &["mineral", "ore"]),
    ("jewel", &["jewel", "diamond", "emerald", "gem", "pearl", "ruby"]),
    ("air", &["air", "atmosphere"]),
    ("fog", &["fog", "mist", "smog"]),
    ("dew", &["dew"]),
    ("heat", &["heat", "hotness", "warmth"]),
    ("cold", &["cold", "chill", "coldness", "frostiness"]),
    ("nature", &["nature", "natural world", "wild"]),
    ("animal", &["animal", "beast", "creature", "fauna"]),
    ("pet", &["pet", "domestic animal"]),
    ("monster", &["monster", "behemoth", "fiend"]),
    ("dragon", &["dragon"]),
    ("dog", &["dog", "canine", "hound", "pooch", "pup", "puppy"]),
    ("cat", &["cat", "feline", "kitten", "kitty"]),
    ("mouse", &["mouse", "mice", "rodent"]),
    ("rat", &["rat"]),
    ("horse", &["horse", "colt", "mare", "pony", "stallion", "steed"]),
    ("donkey", &["donkey", "ass", "mule"]),
    ("cow", &["cow", "bovine", "cattle", "kine"]),
    ("bull", &["bull", "ox", "steer"]),
    ("calf", &["calf"]),
    ("sheep", &["sheep", "ewe", "flock", "ram"]),
    ("lamb", &["lamb"]),
    ("goat", &["goat", "billy", "kid"]),
    ("pig", &["pig", "boar", "hog", "piglet", "swine"]),
    ("bear", &["bear"]),
    ("wolf", &["wolf", "wolves"]),
    ("fox", &["fox"]),
    ("lion", &["lion"]),
    ("tiger", &["tiger"]),
    ("deer", &["deer", "buck", "fawn", "hart", "hind", "stag"]),
    ("rabbit", &["rabbit", "bunny", "hare"]),
    ("monkey", &["monkey", "ape", "chimpanzee"]),
    ("elephant", &["elephant"]),
    ("camel", &["camel"]),
    ("bird", &["bird", "avian", "fowl"]),
    ("eagle", &["eagle"]),
    ("hawk", &["hawk", "falcon"]),
    ("dove", &["dove", "pigeon"]),
    ("crow", &["crow", "raven"]),
    ("duck", &["duck", "drake"]),
    ("goose", &["goose", "gander", "geese"]),
    ("swan", &["swan"]),
    ("chicken", &["chicken", "cock", "hen", "poultry", "rooster"]),
    ("owl", &["owl"]),
    ("sparrow", &["sparrow"]),
    ("fish", &["fish", "fin", "fishes"]),
    ("whale", &["whale"]),
    ("shark", &["shark"]),
    ("salmon", &["salmon"]),
    ("snake", &["snake", "adder", "cobra", "serpent", "viper"]),
    ("frog", &["frog", "toad"]),
    ("worm", &["worm", "maggot"]),
    ("insect", &["insect", "bug"]),
    ("bee", &["bee", "honeybee", "hornet", "wasp"]),
    ("ant", &["ant"]),
    ("spider", &["spider"]),
    ("butterfly", &["butterfly", "moth"]),
    ("place", &["place", "locale", "location", "position", "site", "spot"]),
    ("building", &["building", "edifice", "structure"]),
    ("house", &["house", "abode", "domicile", "dwelling", "home", "residence"]),
    ("room", &["room", "chamber"]),
    ("wall", &["wall", "partition"]),
    ("door", &["door", "doorway", "entrance", "entry", "portal"]),
    ("gate", &["gate", "barrier", "gateway"]),
    ("window", &["window", "casement"]),
    ("roof", &["roof", "rooftop"]),
    ("floor", &["floor", "flooring"]),
    ("ceiling", &["ceiling"]),
    ("hall", &["hall", "corridor", "foyer", "hallway"]),
    ("kitchen", &["kitchen"]),
    ("bedroom", &["bedroom"]),
    ("cellar", &["cellar", "basement", "crypt"]),
    ("stairs", &["stairs", "staircase", "steps"]),
    ("chimney", &["chimney", "flue"]),
    ("fireplace", &["fireplace", "hearth"]),
    ("tower", &["tower", "spire", "steeple", "turret"]),
    ("castle", &["castle", "chateau", "citadel", "palace"]),
    ("palace", &["palace", "court"]),
    ("temple", &["temple", "sanctuary", "shrine"]),
    ("church", &["church", "cathedral", "chapel"]),
    ("prison", &["prison", "cell", "dungeon", "jail"]),
    ("fortress", &["fortress", "fort", "stronghold"]),
    ("fence", &["fence", "enclosure", "hedge"]),
    ("table", &["table", "counter", "desk"]),
    ("chair", &["chair", "bench", "seat", "stool"]),
    ("bed", &["bed", "bunk", "couch", "mattress"]),
    ("shelf", &["shelf", "bookshelf", "ledge"]),
    ("box", &["box", "carton", "case", "casket", "crate"]),
    ("chest_furniture", &["chest_furniture", "coffer", "trunk"]),
    ("bag", &["bag", "pack", "pouch", "purse", "sack"]),
    ("basket", &["basket", "hamper"]),
    ("bottle", &["bottle", "flask", "vial"]),
    ("cup", &["cup", "chalice", "goblet", "mug"]),
    ("glass", &["glass", "tumbler"]),
    ("dish", &["dish", "plate", "platter", "saucer"]),
    ("bowl", &["bowl", "basin"]),
    ("pot", &["pot", "cauldron", "pan", "skillet"]),
    ("kettle", &["kettle"]),
    ("bucket", &["bucket", "pail"]),
    ("barrel", &["barrel", "cask", "keg", "tub"]),
    ("lamp", &["lamp", "candle", "lantern", "torch"]),
    ("mirror", &["mirror", "looking-glass"]),
    ("road", &["road", "avenue", "boulevard", "highway", "lane", "street"]),
    ("path", &["path", "footpath", "track", "trail", "walkway"]),
    ("bridge", &["bridge", "viaduct"]),
    ("tunnel", &["tunnel"]),
    ("city", &["city", "metropolis"]),
    ("town", &["town", "borough", "municipality"]),
    ("village", &["village", "hamlet", "settlement"]),
    ("country", &["country", "land", "nation", "realm", "state"]),
    ("kingdom", &["kingdom", "empire", "principality"]),
    ("market", &["market", "bazaar", "fair", "mart"]),
    ("shop", &["shop", "boutique", "store"]),
    ("court", &["court", "tribunal"]),
    ("camp", &["camp", "bivouac", "encampment"]),
    ("harbor", &["harbor", "dock", "haven", "pier", "port", "wharf"]),
    ("ship", &["ship", "barge", "boat", "craft", "vessel"]),
    ("cart", &["cart", "carriage", "chariot", "dray", "wagon"]),
    ("car", &["car", "automobile", "motor", "truck", "vehicle"]),
    ("train", &["train", "locomotive", "railway"]),
    ("tool", &["tool", "apparatus", "gadget", "implement", "instrument", "utensil"]),
    ("weapon", &["weapon", "arm", "arms", "weaponry"]),
    ("sword", &["sword", "blade", "broadsword", "rapier", "saber"]),
    ("knife", &["knife", "dagger", "dirk", "scalpel"]),
    ("spear", &["spear", "javelin", "lance", "pike"]),
    ("shield", &["shield", "buckler"]),
    ("armor", &["armor", "cuirass", "mail"]),
    ("helmet", &["helmet", "casque", "helm"]),
    ("bow", &["bow", "crossbow", "longbow"]),
    ("arrow", &["arrow", "bolt", "dart", "shaft"]),
    ("gun", &["gun", "cannon", "firearm", "musket", "pistol", "rifle"]),
    ("bullet", &["bullet", "projectile", "shot"]),
    ("rope", &["rope", "cable", "cord", "line", "twine"]),
    ("chain", &["chain", "bonds", "chains", "fetter", "shackles"]),
    ("wire", &["wire"]),
    ("nail", &["nail", "spike", "tack"]),
    ("pin", &["pin", "needle"]),
    ("hook", &["hook"]),
    ("ring", &["ring", "band", "circle", "hoop"]),
    ("stick", &["stick", "cane", "pole", "rod", "staff", "wand"]),
    ("wheel", &["wheel"]),
    ("bell", &["bell", "chime"]),
    ("lock", &["lock", "bolt", "latch", "padlock"]),
    ("key", &["key"]),
    ("clock", &["clock", "timepiece", "watch"]),
    ("book", &["book", "manuscript", "tome", "volume"]),
    ("paper", &["paper", "parchment", "scroll", "sheet"]),
    ("letter", &["letter", "dispatch", "epistle", "note"]),
    ("page", &["page", "folio", "leaf"]),
    ("pen", &["pen", "pencil", "quill"]),
    ("picture", &["picture", "drawing", "image", "painting", "photo", "photograph", "portrait"]),
    ("map", &["map", "chart"]),
    ("clothes", &["clothes", "apparel", "attire", "clothing", "garment", "garments", "raiment"]),
    ("coat", &["coat", "cape", "cloak", "jacket", "mantle", "robe"]),
    ("dress", &["dress", "frock", "gown"]),
    ("shirt", &["shirt", "blouse", "tunic"]),
    ("pants", &["pants", "breeches", "trousers"]),
    ("shoe", &["shoe", "footwear", "sandal", "shoes", "slippers"]),
    ("boot", &["boot", "boots"]),
    ("hat", &["hat", "bonnet", "cap", "headdress"]),
    ("crown", &["crown", "coronet", "diadem"]),
    ("glove", &["glove", "gauntlet", "gloves"]),
    ("belt", &["belt", "girdle", "sash", "strap"]),
    ("cloth", &["cloth", "fabric", "material", "textile"]),
    ("thread", &["thread", "fiber", "yarn"]),
    ("wool", &["wool", "fleece"]),
    ("cotton", &["cotton"]),
    ("silk", &["silk"]),
    ("leather", &["leather", "hide"]),
    ("blanket", &["blanket", "coverlet", "quilt"]),
    ("curtain", &["curtain", "drape", "veil"]),
    ("flag", &["flag", "banner", "colors", "ensign", "standard"]),
    ("cross", &["cross", "crucifix"]),
    ("staff", &["staff", "scepter"]),
    ("plow", &["plow", "plough"]),
    ("axe", &["axe", "hatchet"]),
    ("hammer", &["hammer", "mallet"]),
    ("net", &["net", "mesh"]),
    ("trap", &["trap", "snare"]),
    ("horn", &["horn", "bugle", "trumpet"]),
    ("drum", &["drum"]),
    ("pipe", &["pipe", "tube"]),
    ("string", &["string", "chord"]),
    ("crown_jewel", &["crown_jewel", "tiara"]),
    ("food", &["food", "edibles", "nourishment", "provisions", "sustenance", "victuals"]),
    ("meat", &["meat", "beef", "flesh", "mutton", "pork", "poultry", "venison"]),
    ("bread", &["bread", "crust", "loaf"]),
    ("grain", &["grain", "barley", "cereal", "oats"]),
    ("corn", &["corn", "maize"]),
    ("wheat", &["wheat"]),
    ("rice", &["rice"]),
    ("flour", &["flour", "meal"]),
    ("dish_meal", &["dish_meal", "banquet", "course", "feast"]),
    ("egg", &["egg", "eggs"]),
    ("cheese", &["cheese"]),
    ("butter", &["butter"]),
    ("milk", &["milk"]),
    ("oil", &["oil", "ointment"]),
    ("fat", &["fat", "grease", "lard", "tallow"]),
    ("salt", &["salt"]),
    ("sugar", &["sugar", "sweetener"]),
    ("honey", &["honey"]),
    ("spice", &["spice", "herb", "seasoning"]),
    ("soup", &["soup", "broth", "stew"]),
    ("pie", &["pie", "pastry", "tart"]),
    ("cake", &["cake", "biscuit", "cookie"]),
    ("apple", &["apple"]),
    ("grape", &["grape", "grapes", "raisin", "vine"]),
    ("wine", &["wine", "vintage"]),
    ("beer", &["beer", "ale", "brew"]),
    ("tea", &["tea"]),
    ("coffee", &["coffee"]),
    ("feast", &["feast", "banquet", "celebration"]),
    ("poison", &["poison", "bane", "toxin", "venom"]),
    ("potatoes", &["potatoes", "potato"]),
    ("beans", &["beans", "bean", "pulse"]),
    ("onion", &["onion", "garlic"]),
    ("nut", &["nut", "acorn", "nuts"]),
    ("berry", &["berry", "berries"]),
    ("olive", &["olive"]),
    ("fig", &["fig"]),
    ("vinegar", &["vinegar"]),
    ("crust", &["crust"]),
    ("dough", &["dough"]),
    ("pottage", &["pottage", "gruel", "porridge"]),
    ("sweet", &["sweet", "candy", "confection"]),
    ("hunger", &["hunger", "appetite", "famine", "starvation"]),
    ("thirst", &["thirst"]),
    ("taste_flavor", &["taste_flavor", "relish", "savor"]),
    ("thing", &["thing", "article", "entity", "item", "matter", "object"]),
    ("name", &["name", "appellation", "designation", "label", "moniker", "title"]),
    ("word", &["word", "expression", "remark", "term", "utterance"]),
    ("sound", &["sound", "clamor", "din", "murmur", "noise", "rumble"]),
    ("music", &["music", "harmony", "melody", "symphony", "tune"]),
    ("song", &["song", "anthem", "ballad", "carol", "chant", "hymn"]),
    ("story", &["story", "account", "fable", "legend", "myth", "narrative", "tale", "yarn"]),
    ("news", &["news", "information", "intelligence", "report", "tidings"]),
    ("message", &["message", "communication", "dispatch", "note", "word"]),
    ("truth", &["truth", "accuracy", "fact", "reality", "veracity"]),
    ("secret", &["secret", "confidential", "covert", "mystery"]),
    ("error", &["error", "blunder", "fault", "mistake", "oversight", "slip"]),
    ("fact", &["fact", "actuality", "certainty"]),
    ("reason", &["reason", "cause", "grounds", "logic", "motive", "rationale"]),
    ("cause", &["cause", "origin", "root", "source"]),
    ("effect", &["effect", "consequence", "outcome", "result"]),
    ("purpose", &["purpose", "aim", "end", "goal", "intention", "objective"]),
    ("plan", &["plan", "design", "project", "scheme", "strategy"]),
    ("idea", &["idea", "concept", "conception", "notion", "thought"]),
    ("thought", &["thought", "cogitation", "reflection", "thinking"]),
    ("law", &["law", "code", "commandment", "decree", "ordinance", "statute"]),
    ("rule", &["rule", "canon", "guideline", "precept", "regulation"]),
    ("order", &["order", "directive", "instruction", "mandate"]),
    ("duty", &["duty", "obligation", "responsibility", "task"]),
    ("right", &["right", "entitlement", "equity", "prerogative"]),
    ("justice", &["justice", "equity", "fairness", "righteousness"]),
    ("peace", &["peace", "calm", "harmony", "serenity", "tranquility", "truce"]),
    ("war", &["war", "conflict", "hostilities", "warfare"]),
    ("battle", &["battle", "clash", "combat", "engagement", "fight"]),
    ("power", &["power", "authority", "dominion", "might", "potency", "sway"]),
    ("strength", &["strength", "brawn", "force", "stamina", "sturdiness", "vigor"]),
    ("force", &["force", "compulsion", "energy", "pressure"]),
    ("life", &["life", "being", "existence", "lifespan", "vitality"]),
    ("death", &["death", "decease", "demise", "end", "mortality", "passing"]),
    ("danger", &["danger", "hazard", "jeopardy", "peril", "risk"]),
    ("safety", &["safety", "protection", "security", "shelter"]),
    ("courage", &["courage", "bravery", "fearlessness", "grit", "valor"]),
    ("shame", &["shame", "disgrace", "dishonor", "embarrassment", "humiliation", "ignominy"]),
    ("honor", &["honor", "esteem", "glory", "honour", "renown", "reputation", "respect"]),
    ("glory", &["glory", "magnificence", "majesty", "radiance", "splendor"]),
    ("blessing", &["blessing", "benediction", "boon"]),
    ("curse", &["curse", "anathema", "bane", "hex", "malediction"]),
    ("prayer", &["prayer", "invocation", "orison", "petition", "plea", "supplication"]),
    ("god", &["god", "almighty", "creator", "deity", "divinity"]),
    ("angel", &["angel", "archangel", "cherub", "seraph"]),
    ("devil", &["devil", "demon", "evil spirit", "fiend", "satan"]),
    ("faith", &["faith", "belief", "conviction", "creed", "devotion", "trust"]),
    ("grace", &["grace", "clemency", "favor", "mercy"]),
    ("mercy", &["mercy", "clemency", "compassion", "forgiveness", "leniency", "pity"]),
    ("sin", &["sin", "iniquity", "offense", "transgression", "wickedness", "wrongdoing"]),
    ("crime", &["crime", "felony", "misdeed", "outrage"]),
    ("price", &["price", "charge", "cost", "fare", "fee", "rate"]),
    ("value", &["value", "importance", "merit", "significance", "worth"]),
    ("money", &["money", "cash", "coin", "coins", "currency", "funds", "riches", "wealth"]),
    ("wealth", &["wealth", "affluence", "fortune", "opulence", "prosperity", "riches"]),
    ("treasure", &["treasure", "hoard", "riches", "valuables"]),
    ("debt", &["debt", "arrears", "dues", "liability"]),
    ("gift", &["gift", "bestowal", "boon", "donation", "grant", "offering", "present"]),
    ("reward", &["reward", "award", "premium", "prize", "recompense"]),
    ("punishment", &["punishment", "chastisement", "penalty", "retribution", "sentence"]),
    ("work_labor", &["work_labor", "employment", "occupation", "profession", "trade", "vocation"]),
    ("job", &["job", "position", "post", "role"]),
    ("business", &["business", "affair", "commerce", "enterprise", "industry"]),
    ("company", &["company", "association", "corporation", "firm"]),
    ("society", &["society", "commonwealth", "community", "order"]),
    ("custom", &["custom", "convention", "habit", "practice", "tradition", "usage"]),
    ("art", &["art", "craft", "skill", "technique"]),
    ("science", &["science", "knowledge", "learning", "scholarship"]),
    ("magic", &["magic", "enchantment", "sorcery", "spell", "witchcraft", "wizardry"]),
    ("sign", &["sign", "indication", "mark", "omen", "signal", "token"]),
    ("symbol", &["symbol", "badge", "emblem", "insignia"]),
    ("form", &["form", "contour", "figure", "outline"]),
    ("manner", &["manner", "fashion", "method", "mode", "style", "way"]),
    ("system", &["system", "framework", "organization", "structure"]),
    ("state", &["state", "circumstance", "plight", "situation", "status"]),
    ("trouble", &["trouble", "adversity", "difficulty", "distress", "hardship"]),
    ("pity", &["pity", "commiseration", "condolence", "sympathy"]),
    ("envy", &["envy", "jealousy"]),
    ("anger", &["anger", "fury", "indignation", "ire", "rage", "wrath"]),
    ("pride", &["pride", "arrogance", "conceit", "haughtiness", "vanity"]),
    ("humility", &["humility", "meekness", "modesty"]),
    ("wisdom", &["wisdom", "discernment", "insight", "prudence", "sagacity"]),
    ("folly", &["folly", "absurdity", "foolishness", "silliness", "stupidity"]),
    ("joy", &["joy", "bliss", "delight", "glee", "happiness", "merriment"]),
    ("sorrow", &["sorrow", "grief", "heartache", "misery", "sadness", "woe"]),
    ("number", &["number", "count", "digit", "figure", "numeral"]),
    ("amount", &["amount", "quantity", "sum", "volume"]),
    ("part", &["part", "fragment", "portion", "section", "segment", "share"]),
    ("piece", &["piece", "bit", "chunk", "morsel", "scrap", "slice"]),
    ("half", &["half", "halves"]),
    ("quarter", &["quarter"]),
    ("whole", &["whole", "aggregate", "totality"]),
    ("total", &["total", "entirety", "sum"]),
    ("pair", &["pair", "brace", "duo"]),
    ("group", &["group", "band", "cluster", "collection", "set", "troop"]),
    ("shape", &["shape", "geometry"]),
    ("circle", &["circle", "disk", "ring"]),
    ("square", &["square"]),
    ("line", &["line", "row", "streak", "stripe"]),
    ("point", &["point", "dot", "spot", "tip", "vertex"]),
    ("side", &["side", "flank"]),
    ("edge", &["edge", "border", "brim", "fringe", "margin", "rim", "verge"]),
    ("corner", &["corner", "angle", "nook"]),
    ("middle", &["middle", "center", "core", "heart", "midst"]),
    ("center", &["center", "focal point", "hub"]),
    ("front", &["front", "fore", "vanguard"]),
    ("top", &["top", "apex", "crest", "peak", "summit"]),
    ("bottom", &["bottom", "base", "foot", "foundation"]),
    ("inside", &["inside", "interior"]),
    ("outside", &["outside", "exterior"]),
    ("beginning", &["beginning", "commencement", "onset", "origin", "outset", "start"]),
    ("size", &["size", "bulk", "dimensions", "extent", "magnitude"]),
    ("length", &["length", "extent", "span"]),
    ("width", &["width", "breadth"]),
    ("height", &["height", "altitude", "elevation", "tallness"]),
    ("depth", &["depth", "abyss", "profundity"]),
    ("weight", &["weight", "heaviness", "mass"]),
    ("speed", &["speed", "pace", "rapidity", "swiftness", "tempo", "velocity"]),
    ("distance", &["distance", "interval", "range", "remoteness"]),
    ("north", &["north", "northern"]),
    ("south", &["south", "southern"]),
    ("east", &["east", "eastern", "orient"]),
    ("west", &["west", "occident", "western"]),
    ("left", &["left"]),
    ("right_dir", &["right_dir"]),
    ("level", &["level", "plane"]),
    ("weight_pound", &["weight_pound", "gram", "ounce", "ton"]),
    ("measure", &["measure", "foot_len", "meter", "mile", "yard"]),
    ("zero", &["zero", "naught", "nil", "null"]),
    ("big", &["big", "colossal", "enormous", "gigantic", "grand", "huge", "immense", "large", "massive", "tremendous"]),
    ("small", &["small", "compact", "little", "mini", "miniature", "minute", "petite", "slight", "tiny"]),
    ("great", &["great", "mighty", "towering", "vast"]),
    ("tall", &["tall", "elevated", "lofty"]),
    ("short", &["short", "brief", "lowly"]),
    ("high", &["high", "lofty", "steep"]),
    ("low", &["low"]),
    ("deep", &["deep", "profound"]),
    ("shallow", &["shallow"]),
    ("long", &["long", "extended", "lengthy", "prolonged"]),
    ("broad", &["broad", "expansive", "roomy", "spacious", "wide"]),
    ("narrow", &["narrow", "slender", "slim", "thin", "tight"]),
    ("thick", &["thick", "dense", "heavy", "stout"]),
    ("thin", &["thin", "lean", "scrawny", "skinny", "slender", "slim"]),
    ("heavy", &["heavy", "burdensome", "ponderous", "weighty"]),
    ("light_weight", &["light_weight", "buoyant", "feathery", "lightweight"]),
    ("hard", &["hard", "firm", "rigid", "solid", "stiff", "tough"]),
    ("soft", &["soft", "cushioned", "pliable", "tender", "yielding"]),
    ("rough", &["rough", "bumpy", "coarse", "jagged", "rugged", "uneven"]),
    ("smooth", &["smooth", "even", "level", "polished", "sleek", "slick"]),
    ("sharp", &["sharp", "acute", "keen", "pointed", "razor-sharp"]),
    ("dull", &["dull", "blunt", "unsharpened"]),
    ("straight", &["straight", "direct", "unswerving"]),
    ("crooked", &["crooked", "bent", "curved", "twisted", "winding"]),
    ("round", &["round", "circular", "globular", "spherical"]),
    ("flat", &["flat", "horizontal", "planar"]),
    ("hot", &["hot", "blazing", "burning", "fiery", "scalding", "scorching"]),
    ("warm", &["warm", "balmy", "mild", "tepid"]),
    ("cool", &["cool", "brisk", "chilly"]),
    ("dry", &["dry", "arid", "dehydrated", "parched", "waterless"]),
    ("wet", &["wet", "damp", "drenched", "moist", "soaked"]),
    ("bright", &["bright", "brilliant", "glowing", "luminous", "radiant", "shining", "vivid"]),
    ("dark", &["dark", "dim", "gloomy", "murky", "obscure", "shadowy", "somber"]),
    ("clear", &["clear", "crystal", "distinct", "limpid", "lucid", "transparent"]),
    ("cloudy", &["cloudy", "foggy", "hazy", "misty", "overcast"]),
    ("loud", &["loud", "booming", "clamorous", "deafening", "noisy", "roaring"]),
    ("quiet", &["quiet", "hushed", "noiseless", "silent", "soundless", "still"]),
    ("bitter", &["bitter", "acrid", "sour", "tart"]),
    ("fresh", &["fresh", "crisp", "new-made", "refreshing"]),
    ("stale", &["stale", "decayed", "moldy", "musty"]),
    ("clean", &["clean", "immaculate", "pure", "spotless", "stainless"]),
    ("dirty", &["dirty", "filthy", "foul", "grimy", "soiled", "unclean"]),
    ("pure", &["pure", "chaste", "unblemished", "undefiled", "untainted"]),
    ("red", &["red", "crimson", "ruby", "scarlet"]),
    ("blue", &["blue", "azure", "cerulean", "navy"]),
    ("green", &["green", "emerald", "verdant"]),
    ("yellow", &["yellow", "amber", "blonde", "golden"]),
    ("black", &["black", "darkest", "ebony", "pitch-black", "sable"]),
    ("white", &["white", "chalky", "ivory", "pale", "snowy"]),
    ("gray", &["gray", "ash-colored", "grey", "silver"]),
    ("brown", &["brown", "bronze", "tan", "tawny"]),
    ("purple", &["purple", "plum", "violet"]),
    ("pink", &["pink", "rose", "rosy"]),
    ("fast", &["fast", "brisk", "hasty", "quick", "rapid", "speedy", "swift"]),
    ("slow", &["slow", "gradual", "leisurely", "sluggish", "unhurried"]),
    ("tight", &["tight", "constricted", "snug", "taut"]),
    ("loose", &["loose", "relaxed", "slack", "unfastened"]),
    ("dense", &["dense", "compact", "crowded"]),
    ("bare", &["bare", "naked", "nude", "stripped", "unadorned"]),
    ("rich", &["rich", "affluent", "opulent", "prosperous", "wealthy"]),
    ("poor", &["poor", "destitute", "impoverished", "needy", "penniless"]),
    ("good", &["good", "admirable", "fabulous", "fine", "nice", "pleasant", "positive", "splendid"]),
    ("excellent", &["excellent", "awesome", "exceptional", "fantastic", "marvelous", "outstanding", "superb", "terrific", "wonderful"]),
    ("bad", &["bad", "dreadful", "foul", "inferior", "negative", "poor", "substandard"]),
    ("terrible", &["terrible", "abominable", "awful", "dire", "ghastly", "horrible", "horrific"]),
    ("evil", &["evil", "depraved", "malicious", "nefarious", "sinful", "vile", "wicked"]),
    ("holy", &["holy", "blessed", "consecrated", "divine", "hallowed", "sacred", "saintly"]),
    ("happy", &["happy", "cheerful", "content", "delighted", "ecstatic", "glad", "jolly", "joyful", "merry", "thrilled"]),
    ("sad", &["sad", "depressed", "downcast", "gloomy", "heartbroken", "miserable", "mournful", "sorrowful", "unhappy"]),
    ("angry", &["angry", "enraged", "furious", "indignant", "irate", "mad", "wrathful"]),
    ("calm", &["calm", "composed", "peaceful", "placid", "serene", "tranquil", "unruffled"]),
    ("afraid", &["afraid", "alarmed", "fearful", "frightened", "nervous", "scared", "terrified", "timid"]),
    ("brave", &["brave", "bold", "courageous", "daring", "fearless", "heroic", "intrepid", "valiant"]),
    ("proud", &["proud", "arrogant", "boastful", "dignified", "haughty", "vainglorious"]),
    ("humble", &["humble", "lowly", "meek", "modest", "unassuming"]),
    ("kind", &["kind", "benevolent", "compassionate", "gentle", "gracious", "sympathetic", "tender"]),
    ("cruel", &["cruel", "brutal", "callous", "fierce", "heartless", "merciless", "ruthless", "savage"]),
    ("wise", &["wise", "knowledgeable", "learned", "prudent", "sagacious", "sensible"]),
    ("smart", &["smart", "astute", "bright", "clever", "ingenious", "intelligent", "sharp"]),
    ("foolish", &["foolish", "absurd", "dumb", "idiotic", "ridiculous", "senseless", "silly", "stupid"]),
    ("honest", &["honest", "candid", "frank", "sincere", "trustworthy", "truthful", "upright"]),
    ("false", &["false", "counterfeit", "deceitful", "dishonest", "fake", "fraudulent", "spurious"]),
    ("true", &["true", "actual", "authentic", "genuine", "real", "veritable"]),
    ("right_correct", &["right_correct", "accurate", "correct", "fitting", "just", "proper", "suitable"]),
    ("wrong", &["wrong", "erroneous", "faulty", "improper", "inappropriate", "incorrect", "mistaken"]),
    ("safe", &["safe", "harmless", "protected", "secure", "sheltered"]),
    ("dangerous", &["dangerous", "hazardous", "perilous", "risky", "treacherous", "unsafe"]),
    ("easy", &["easy", "effortless", "elementary", "facile", "simple"]),
    ("difficult", &["difficult", "arduous", "burdensome", "hard", "laborious", "onerous", "tough"]),
    ("simple", &["simple", "plain", "straightforward", "uncomplicated"]),
    ("complex", &["complex", "complicated", "elaborate", "intricate", "involved"]),
];

/// Translation context preserving entities and brand names.
#[derive(Debug, Clone, Default)]
pub struct TranslationContext {
    pub entities: Vec<String>,
    pub numbers: Vec<String>,
}

impl TranslationContext {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Comprehensive, dynamically loadable Yumon Talk Vocabulary (1024 tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YumonTalkVocab {
    pub version: usize,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub punctuation: Vec<String>,
    pub placeholders: Vec<String>,
    pub canonical_words: Vec<String>,
    pub synonym_clusters: HashMap<String, Vec<String>>,

    #[serde(skip)]
    pub token_to_id: HashMap<String, usize>,
    #[serde(skip)]
    pub id_to_token: Vec<String>,
    #[serde(skip)]
    pub word_to_id: HashMap<String, usize>,
    #[serde(skip)]
    pub dynamic_cache: Arc<RwLock<HashMap<String, usize>>>,
}

impl YumonTalkVocab {
    pub fn new(
        special_tokens: Vec<String>,
        punctuation: Vec<String>,
        placeholders: Vec<String>,
        canonical_words: Vec<String>,
        synonym_clusters: HashMap<String, Vec<String>>,
    ) -> Self {
        let total = special_tokens.len() + punctuation.len() + placeholders.len() + canonical_words.len();
        let mut v = Self {
            version: 1,
            vocab_size: total,
            special_tokens,
            punctuation,
            placeholders,
            canonical_words,
            synonym_clusters,
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            word_to_id: HashMap::new(),
            dynamic_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        v.rebuild_indices();
        v
    }

    pub fn rebuild_indices(&mut self) {
        let total = self.special_tokens.len() + self.punctuation.len() + self.placeholders.len() + self.canonical_words.len();
        let mut id_to_token = Vec::with_capacity(total);
        let mut token_to_id = HashMap::with_capacity(total);
        let mut word_to_id = HashMap::with_capacity(total * 10);

        for tok in &self.special_tokens {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }
        for tok in &self.punctuation {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }
        for tok in &self.placeholders {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            id_to_token.push(tok.clone());
        }
        for tok in &self.canonical_words {
            let id = id_to_token.len();
            token_to_id.insert(tok.clone(), id);
            word_to_id.insert(tok.to_lowercase(), id);
            id_to_token.push(tok.clone());
        }
        for (canon, syns) in &self.synonym_clusters {
            if let Some(&id) = token_to_id.get(canon) {
                word_to_id.insert(canon.to_lowercase(), id);
                for syn in syns {
                    word_to_id.insert(syn.to_lowercase(), id);
                }
            }
        }
        self.vocab_size = total;
        self.token_to_id = token_to_id;
        self.id_to_token = id_to_token;
        self.word_to_id = word_to_id;
        self.dynamic_cache = Arc::new(RwLock::new(HashMap::new()));
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading vocab file {}", path.display()))?;
        let mut vocab: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing vocab file {}", path.display()))?;
        vocab.rebuild_indices();
        Ok(vocab)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn default_vocab() -> Self {
        let special_tokens = vec![
            "<PAD>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "<UNK>".to_string(),
        ];
        let punctuation = vec![
            ".".to_string(),
            ",".to_string(),
            "!".to_string(),
            "?".to_string(),
            ";".to_string(),
            ":".to_string(),
            "-".to_string(),
            "'".to_string(),
            "\"".to_string(),
            "(".to_string(),
            ")".to_string(),
            "[".to_string(),
            "]".to_string(),
            "/".to_string(),
            "\n".to_string(),
        ];
        let mut placeholders = Vec::with_capacity(40);
        for i in 0..32 {
            placeholders.push(format!("<NAME_{i}>"));
        }
        for i in 0..8 {
            placeholders.push(format!("<NUM_{i}>"));
        }
        let canonical_words = VOCAB[59..].iter().map(|&s| s.to_string()).collect();
        let mut synonym_clusters = HashMap::new();
        for &(canon, syns) in SYNONYM_TABLE.iter() {
            synonym_clusters.insert(canon.to_string(), syns.iter().map(|&s| s.to_string()).collect());
        }
        Self::new(special_tokens, punctuation, placeholders, canonical_words, synonym_clusters)
    }

    pub fn token_to_id(&self, tok: &str) -> Option<usize> {
        self.token_to_id.get(tok).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    pub fn resolve_english_word(&self, word: &str) -> Option<usize> {
        let lower = word.to_lowercase();

        // 1. Static/cluster lookup
        if let Some(&id) = self.word_to_id.get(&lower) {
            return Some(id);
        }

        // 2. Dynamic cache lookup
        if let Ok(c) = self.dynamic_cache.read() {
            if let Some(&id) = c.get(&lower) {
                return Some(id);
            }
        }

        // 3. Morphological Stemming
        let stems = [
            lower.strip_suffix("'s").unwrap_or(&lower),
            lower.strip_suffix("s").unwrap_or(&lower),
            lower.strip_suffix("es").unwrap_or(&lower),
            lower.strip_suffix("ed").unwrap_or(&lower),
            lower.strip_suffix("ing").unwrap_or(&lower),
            lower.strip_suffix("ly").unwrap_or(&lower),
            lower.strip_suffix("er").unwrap_or(&lower),
            lower.strip_suffix("est").unwrap_or(&lower),
        ];
        for &stem in stems.iter() {
            if stem != lower.as_str() && !stem.is_empty() {
                if let Some(&id) = self.word_to_id.get(stem) {
                    if let Ok(mut c) = self.dynamic_cache.write() {
                        c.insert(lower.clone(), id);
                    }
                    return Some(id);
                }
            }
        }

        // 4. Query `thesaurus` crate
        let syns = thesaurus::synonyms(&lower);
        for syn in syns.iter() {
            let clean = syn.trim().to_lowercase();
            if let Some(&id) = self.word_to_id.get(&clean) {
                if let Ok(mut c) = self.dynamic_cache.write() {
                    c.insert(lower.clone(), id);
                }
                return Some(id);
            }
        }

        None
    }

    pub fn english_to_yumon_tokens(&self, english_text: &str, ctx: &mut TranslationContext) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = english_text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        let mut entity_map: HashMap<String, usize> = HashMap::new();
        let mut num_map: HashMap<String, usize> = HashMap::new();

        while i < len {
            // Skip whitespace
            if chars[i].is_whitespace() {
                if chars[i] == '\n' {
                    if let Some(id) = self.token_to_id("\n") {
                        tokens.push(id);
                    }
                }
                i += 1;
                continue;
            }

            // Punctuation
            let ch = chars[i];
            let punct_str = ch.to_string();
            if let Some(id) = self.token_to_id(&punct_str) {
                tokens.push(id);
                i += 1;
                continue;
            }

            // Numbers
            if ch.is_ascii_digit() {
                let start = i;
                while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let idx = num_map.entry(num_str.clone()).or_insert_with(|| {
                    let curr = ctx.numbers.len();
                    ctx.numbers.push(num_str);
                    curr % 8
                });
                let slot = format!("<NUM_{idx}>");
                tokens.push(self.token_to_id(&slot).unwrap_or(UNK_ID));
                continue;
            }

            // Word (alphanumeric + apostrophe)
            if ch.is_alphabetic() || ch == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || (chars[i] == '\'' && i + 1 < len && chars[i + 1].is_alphabetic())) {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Try resolving word to a canonical Yumon Talk ID
                if let Some(id) = self.resolve_english_word(&word) {
                    tokens.push(id);
                    continue;
                }

                // If capitalized / brand name / out of vocabulary entity:
                let ent_idx = entity_map.entry(word.clone()).or_insert_with(|| {
                    let curr = ctx.entities.len();
                    ctx.entities.push(word);
                    curr % 32
                });
                let slot = format!("<NAME_{ent_idx}>");
                tokens.push(self.token_to_id(&slot).unwrap_or(UNK_ID));
                continue;
            }

            // Other character: fallback
            i += 1;
        }

        Ok(tokens)
    }

    pub fn tokenize_yumon(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i].is_whitespace() {
                if chars[i] == '\n' {
                    if let Some(id) = self.token_to_id("\n") {
                        tokens.push(id);
                    }
                }
                i += 1;
                continue;
            }

            // Check for bracketed placeholders like <NAME_0>, <NUM_0>, etc.
            if chars[i] == '<' {
                let start = i;
                while i < len && chars[i] != '>' {
                    i += 1;
                }
                if i < len && chars[i] == '>' {
                    i += 1;
                    let token_str: String = chars[start..i].iter().collect();
                    if let Some(id) = self.token_to_id(&token_str) {
                        tokens.push(id);
                        continue;
                    }
                }
            }

            // Punctuation
            let ch = chars[i];
            let punct_str = ch.to_string();
            if let Some(id) = self.token_to_id(&punct_str) {
                tokens.push(id);
                i += 1;
                continue;
            }

            // Word
            if ch.is_alphanumeric() || ch == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                if let Some(id) = self.token_to_id(&word) {
                    tokens.push(id);
                } else {
                    tokens.push(UNK_ID);
                }
                continue;
            }

            i += 1;
        }

        tokens
    }

    pub fn detokenize_yumon(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids.iter() {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            let tok = self.id_to_token(id).unwrap_or("<UNK>");
            if tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":" || tok == ")" || tok == "]" {
                out.push_str(tok);
                continue;
            }
            if tok == "(" || tok == "[" {
                if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') {
                    out.push(' ');
                }
                out.push_str(tok);
                continue;
            }
            if tok == "\n" {
                out.push('\n');
                continue;
            }
            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') && !out.ends_with('(') && !out.ends_with('[') {
                out.push(' ');
            }
            out.push_str(tok);
        }
        out
    }

    pub fn english_to_yumon(&self, english_text: &str) -> Result<String> {
        let mut ctx = TranslationContext::new();
        let tokens = self.english_to_yumon_tokens(english_text, &mut ctx)?;
        ensure!(!tokens.is_empty(), "empty Yumon Talk translation");
        ensure!(!tokens.iter().any(|&id| id == UNK_ID), "Yumon Talk produced UNK tokens");
        Ok(self.detokenize_yumon(&tokens))
    }

    pub fn english_to_yumon_with_map(&self, english_text: &str) -> Result<(String, TranslationContext)> {
        let mut ctx = TranslationContext::new();
        let tokens = self.english_to_yumon_tokens(english_text, &mut ctx)?;
        ensure!(!tokens.is_empty(), "empty Yumon Talk translation");
        ensure!(!tokens.iter().any(|&id| id == UNK_ID), "Yumon Talk produced UNK tokens");
        Ok((self.detokenize_yumon(&tokens), ctx))
    }

    pub fn yumon_to_english(&self, yumon_text: &str) -> Result<String> {
        let mut rng = rand::thread_rng();
        self.yumon_to_english_with_rng(yumon_text, &mut rng)
    }

    pub fn yumon_to_english_with_rng(&self, yumon_text: &str, rng: &mut impl rand::Rng) -> Result<String> {
        let ctx = TranslationContext::new();
        self.yumon_to_english_with_context(yumon_text, &ctx, rng)
    }

    pub fn yumon_to_english_with_context(&self, yumon_text: &str, ctx: &TranslationContext, rng: &mut impl rand::Rng) -> Result<String> {
        let tokens = self.tokenize_yumon(yumon_text);
        ensure!(!tokens.is_empty(), "empty Yumon Talk tokens");

        let mut out = String::new();
        let mut cap_next = true;

        for &id in tokens.iter() {
            if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                continue;
            }
            let tok = self.id_to_token(id).unwrap_or("<UNK>");

            // Punctuation
            if tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":" || tok == ")" || tok == "]" {
                out.push_str(tok);
                if tok == "." || tok == "!" || tok == "?" {
                    cap_next = true;
                }
                continue;
            }
            if tok == "(" || tok == "[" {
                if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') {
                    out.push(' ');
                }
                out.push_str(tok);
                continue;
            }
            if tok == "\n" {
                out.push('\n');
                cap_next = true;
                continue;
            }

            // Placeholder for name/brand
            let word = if tok.starts_with("<NAME_") && tok.ends_with('>') {
                let num_part = &tok[6..tok.len() - 1];
                if let Ok(idx) = num_part.parse::<usize>() {
                    if let Some(orig) = ctx.entities.get(idx) {
                        orig.clone()
                    } else {
                        format!("entity_{idx}")
                    }
                } else {
                    tok.to_string()
                }
            } else if tok.starts_with("<NUM_") && tok.ends_with('>') {
                let num_part = &tok[5..tok.len() - 1];
                if let Ok(idx) = num_part.parse::<usize>() {
                    if let Some(orig) = ctx.numbers.get(idx) {
                        orig.clone()
                    } else {
                        format!("{idx}")
                    }
                } else {
                    tok.to_string()
                }
            } else {
                // Canonical Yumon Talk word: randomly sample among synonym cluster
                if let Some(syn_list) = self.synonym_clusters.get(tok) {
                    if !syn_list.is_empty() {
                        let chosen = &syn_list[rng.gen_range(0..syn_list.len())];
                        chosen.clone()
                    } else {
                        tok.to_string()
                    }
                } else {
                    tok.to_string()
                }
            };

            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') && !out.ends_with('(') && !out.ends_with('[') {
                out.push(' ');
            }

            if cap_next {
                let mut c_chars = word.chars();
                if let Some(first) = c_chars.next() {
                    out.push(first.to_ascii_uppercase());
                    out.push_str(c_chars.as_str());
                }
                cap_next = false;
            } else {
                out.push_str(&word);
            }
        }

        Ok(out)
    }

    pub fn verify_yumon_roundtrip(&self, english_text: &str) -> Result<bool> {
        let (yumon, ctx) = match self.english_to_yumon_with_map(english_text) {
            Ok(res) => res,
            Err(_) => return Ok(false),
        };
        let mut rng = rand::thread_rng();
        match self.yumon_to_english_with_context(&yumon, &ctx, &mut rng) {
            Ok(reconstructed) => Ok(!reconstructed.is_empty()),
            Err(_) => Ok(false),
        }
    }
}

static GLOBAL_VOCAB: OnceLock<RwLock<YumonTalkVocab>> = OnceLock::new();

pub fn get_global_vocab() -> std::sync::RwLockReadGuard<'static, YumonTalkVocab> {
    let rw = GLOBAL_VOCAB.get_or_init(|| {
        let default_path = Path::new("training-cache/yumon_talk_vocab.json");
        if default_path.exists() {
            if let Ok(v) = YumonTalkVocab::load_from_file(default_path) {
                return RwLock::new(v);
            }
        }
        RwLock::new(YumonTalkVocab::default_vocab())
    });
    rw.read().expect("global vocab lock poisoned")
}

pub fn set_global_vocab(vocab: YumonTalkVocab) {
    let rw = GLOBAL_VOCAB.get_or_init(|| RwLock::new(YumonTalkVocab::default_vocab()));
    let mut guard = rw.write().expect("global vocab lock poisoned");
    *guard = vocab;
}

pub fn load_global_vocab_file(path: &Path) -> Result<()> {
    let vocab = YumonTalkVocab::load_from_file(path)?;
    set_global_vocab(vocab);
    Ok(())
}

pub fn token_to_id(token: &str) -> Option<usize> {
    get_global_vocab().token_to_id(token)
}

pub fn id_to_token(id: usize) -> Option<String> {
    get_global_vocab().id_to_token(id).map(|s| s.to_string())
}

pub fn resolve_english_word(word: &str) -> Option<usize> {
    get_global_vocab().resolve_english_word(word)
}

pub fn english_to_yumon_tokens(english_text: &str, ctx: &mut TranslationContext) -> Result<Vec<usize>> {
    get_global_vocab().english_to_yumon_tokens(english_text, ctx)
}

pub fn tokenize_yumon(text: &str) -> Vec<usize> {
    get_global_vocab().tokenize_yumon(text)
}

pub fn detokenize_yumon(ids: &[usize]) -> String {
    get_global_vocab().detokenize_yumon(ids)
}

pub fn english_to_yumon(english_text: &str) -> Result<String> {
    get_global_vocab().english_to_yumon(english_text)
}

pub fn english_to_yumon_with_map(english_text: &str) -> Result<(String, TranslationContext)> {
    get_global_vocab().english_to_yumon_with_map(english_text)
}

pub fn yumon_to_english(yumon_text: &str) -> Result<String> {
    get_global_vocab().yumon_to_english(yumon_text)
}

pub fn yumon_to_english_with_rng(yumon_text: &str, rng: &mut impl rand::Rng) -> Result<String> {
    get_global_vocab().yumon_to_english_with_rng(yumon_text, rng)
}

pub fn yumon_to_english_with_context(yumon_text: &str, ctx: &TranslationContext, rng: &mut impl rand::Rng) -> Result<String> {
    get_global_vocab().yumon_to_english_with_context(yumon_text, ctx, rng)
}

pub fn verify_yumon_roundtrip(english_text: &str) -> Result<bool> {
    get_global_vocab().verify_yumon_roundtrip(english_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size_is_1024() {
        assert_eq!(VOCAB.len(), 1024);
        assert_eq!(VOCAB_SIZE, 1024);
        let mut unique = std::collections::HashSet::new();
        for &t in VOCAB.iter() {
            assert!(unique.insert(t), "Duplicate token in VOCAB: {t}");
        }
    }

    #[test]
    fn test_synonym_table_matches_canonical_words() {
        for &(canon, syns) in SYNONYM_TABLE.iter() {
            assert!(token_to_id(canon).is_some(), "Canonical word {canon} not in VOCAB");
            assert!(!syns.is_empty(), "Synonym list for {canon} cannot be empty");
        }
    }

    #[test]
    fn test_english_to_yumon_and_back() {
        let text = "Hello! I am very happy to see my good friend today.";
        let (yumon, ctx) = english_to_yumon_with_map(text).unwrap();
        assert!(!yumon.is_empty());
        assert!(!yumon.contains("<UNK>"));

        let mut rng = rand::thread_rng();
        let english = yumon_to_english_with_context(&yumon, &ctx, &mut rng).unwrap();
        assert!(!english.is_empty());
        println!("Original: {text}");
        println!("Yumon:    {yumon}");
        println!("Back:     {english}");
    }

    #[test]
    fn test_brand_name_placeholder() {
        let text = "Google and Nintendo announced a new Yumon Pet device.";
        let (yumon, ctx) = english_to_yumon_with_map(text).unwrap();
        assert!(yumon.contains("<NAME_0>"));
        assert!(yumon.contains("<NAME_1>"));

        let mut rng = rand::thread_rng();
        let english = yumon_to_english_with_context(&yumon, &ctx, &mut rng).unwrap();
        assert!(english.contains("Google"));
        assert!(english.contains("Nintendo"));
    }

    #[test]
    fn test_thesaurus_synonym_resolution() {
        // Words like 'delighted', 'enormous', 'chatting' should resolve into canonical words
        assert!(resolve_english_word("delighted").is_some());
        assert!(resolve_english_word("enormous").is_some());
    }

    #[test]
    fn test_roundtrip_verification() {
        assert!(verify_yumon_roundtrip("The sun shines bright in the blue sky.").unwrap());
        assert!(verify_yumon_roundtrip("Please give me some water because I am thirsty.").unwrap());
    }
}
