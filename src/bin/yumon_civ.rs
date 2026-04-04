// Playground and training ground for Yumon

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
    Frame, Terminal,
};
use std::io;

const MAP_W: usize = 180;
const MAP_H: usize = 36;
const TECH_COLS: usize = 20;
const TECH_ROWS: usize = 5;

/// Change this to spawn more or fewer AI opponents.
const NUM_AI_PLAYERS: usize = 24;

#[derive(Clone, Copy, PartialEq)]
enum Tile { DeepWater, ShallowWater, Sand, Plains, Forest, Hills, Mountain, Snow }

impl Tile {
    fn from_height(h: f64) -> Self {
        match h {
            h if h < -0.55 => Tile::DeepWater,
            h if h < -0.35 => Tile::ShallowWater,
            h if h < 0.02  => Tile::Sand,
            h if h < 0.25  => Tile::Plains,
            h if h < 0.45  => Tile::Forest,
            h if h < 0.65  => Tile::Hills,
            h if h < 0.82  => Tile::Mountain,
            _               => Tile::Snow,
        }
    }
    fn glyph(self) -> &'static str {
        match self {
            Tile::DeepWater    => "≈",
            Tile::ShallowWater => "~",
            Tile::Sand         => ".",
            Tile::Plains       => ",",
            Tile::Forest       => "f",
            Tile::Hills        => "n",
            Tile::Mountain     => "^",
            Tile::Snow         => "*",
        }
    }
    fn fg(self) -> Color {
        match self {
            Tile::DeepWater    => Color::Rgb(30,  80,  160),
            Tile::ShallowWater => Color::Rgb(60,  140, 200),
            Tile::Sand         => Color::Rgb(210, 190, 120),
            Tile::Plains       => Color::Rgb(120, 180, 80),
            Tile::Forest       => Color::Rgb(40,  120, 40),
            Tile::Hills        => Color::Rgb(160, 130, 70),
            Tile::Mountain     => Color::Rgb(160, 160, 160),
            Tile::Snow         => Color::Rgb(230, 240, 255),
        }
    }
    fn bg(self) -> Color {
        match self {
            Tile::DeepWater    => Color::Rgb(10,  40,  100),
            Tile::ShallowWater => Color::Rgb(20,  80,  140),
            Tile::Sand         => Color::Rgb(180, 160, 80),
            Tile::Plains       => Color::Rgb(70,  130, 40),
            Tile::Forest       => Color::Rgb(15,  70,  15),
            Tile::Hills        => Color::Rgb(100, 75,  35),
            Tile::Mountain     => Color::Rgb(80,  80,  80),
            Tile::Snow         => Color::Rgb(190, 210, 230),
        }
    }
    fn passable(self) -> bool {
        !matches!(self, Tile::DeepWater | Tile::ShallowWater | Tile::Mountain | Tile::Snow)
    }
}

#[derive(Clone)]
struct Stat {
    food: i32,
    atk: i32,
    prod: i32,
    mov: i32,
    def: i32,
    gold: i32,
    sci: i32,
    cult: i32
}

#[derive(Clone)]
struct Tech {
    name: &'static str,
    unit: Option<&'static str>,
    stats: Stat,
    desc: &'static str,
    cost: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum Track { Military, Economy, Infrastructure, Culture, Science }

impl Track {
    fn label(self) -> &'static str {
        match self {
            Track::Military       => "Military",
            Track::Economy        => "Economy",
            Track::Infrastructure => "Infrastr.",
            Track::Culture        => "Culture",
            Track::Science        => "Science",
        }
    }
    fn color(self) -> Color {
        match self {
            Track::Military       => Color::Rgb(210, 90,  48),
            Track::Economy        => Color::Rgb(90,  170, 40),
            Track::Infrastructure => Color::Rgb(55,  138, 221),
            Track::Culture        => Color::Rgb(150, 130, 240),
            Track::Science        => Color::Rgb(210, 140, 30),
        }
    }
    fn dim_color(self) -> Color {
        match self {
            Track::Military       => Color::Rgb(100, 45,  20),
            Track::Economy        => Color::Rgb(30,  80,  15),
            Track::Infrastructure => Color::Rgb(15,  50,  110),
            Track::Culture        => Color::Rgb(55,  45,  120),
            Track::Science        => Color::Rgb(90,  55,  10),
        }
    }
}

fn all_techs() -> Vec<(Track, Vec<Tech>)> {
    let costs: [u32; 20] = [3,3,3,7,8,9,9,9,12,15,18,22,26,30,32,36,38,42,45,52];
    macro_rules! track {
        ($track:expr, [ $( ($n:expr, $u:expr, $s:expr, $d:expr) ),* $(,)? ]) => {{
            let raw: Vec<(&str, Option<&str>, Stat, &str)> = vec![ $( ($n, $u, $s, $d) ),* ];
            let techs: Vec<Tech> = raw.into_iter().enumerate().map(|(i, (n,u,s,d))| Tech {
                name: n, unit: u, stats: s, desc: d, cost: costs[i],
            }).collect();
            ($track, techs)
        }};
    }
    vec![
        track!(Track::Military, [
            ("Foraging",       Some("Scout"),          Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },          "Ranging parties map the land."),
            ("Stone Tools",    Some("Warrior"),        Stat { food: 0, atk: 1, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Chipped flint arms the first fighters."),
            ("Spear",          Some("Spearman"),       Stat { food: 0, atk: 1, prod: 1, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },   "Hafted point for hunting and combat."),
            ("Hunting",        Some("Archer"),         Stat { food: 1, atk: 0, prod: 1, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },  "Bow and trap techniques."),
            ("Bronze Weapons", Some("Swordsman"),      Stat { food: 0, atk: 2, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Smelted bronze blades."),
            ("Shield Craft",   Some("Heavy Infantry"), Stat { food: 0, atk: 2, prod: 0, mov: 0, def: 2, gold: 0, sci: 0, cult: 0 },           "Wicker and hide shields."),
            ("Phalanx",        Some("Hoplite"),        Stat { food: 0, atk: 2, prod: 0, mov: 1, def: 0, gold: 0, sci: 0, cult: 0 },    "Locked-shield spear wall."),
            ("War Chariot",    Some("Chariot"),        Stat { food: 0, atk: 3, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },    "Horse-drawn shock flanks."),
            ("Iron Forging",   Some("Iron Swordsman"), Stat { food: 0, atk: 2, prod: 1, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Harder iron replaces bronze."),
            ("Siege Craft",    Some("Catapult"),       Stat { food: 0, atk: 3, prod: 0, mov: 2, def: 0, gold: 0, sci: 0, cult: 0 },   "Tension engines hurl stones."),
            ("Cavalry",        Some("Knight"),         Stat { food: 0, atk: 3, prod: 0, mov: 2, def: 0, gold: 0, sci: 0, cult: 0 },    "Armored horsemen break lines."),
            ("Crossbow",       Some("Crossbowman"),    Stat { food: 0, atk: 2, prod: 0, mov: 0, def: 1, gold: 0, sci: 0, cult: 0 },    "Mechanical bow penetrates plate."),
            ("Plate Armor",    Some("Man-at-Arms"),    Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 4, gold: 0, sci: 0, cult: 0 },           "Full steel plate cuts casualties."),
            ("Cannon",         Some("Cannon"),         Stat { food: 0, atk: 4, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Gunpowder shatters stone walls."),
            ("Musketry",       Some("Musketeer"),      Stat { food: 0, atk: 3, prod: 0, mov: 0, def: 1, gold: 0, sci: 0, cult: 0 },    "Matchlock firearms replace bows."),
            ("Mil. Drill",     Some("Grenadier"),      Stat { food: 0, atk: 3, prod: 0, mov: 0, def: 2, gold: 0, sci: 0, cult: 0 },    "Drill and explosive grenades."),
            ("Field Artillery",Some("Field Gun"),      Stat { food: 0, atk: 5, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Mobile cannon supports infantry."),
            ("Rifling",        Some("Rifleman"),       Stat { food: 0, atk: 4, prod: 0, mov: 0, def: 1, gold: 0, sci: 0, cult: 0 },    "Grooved barrels improve accuracy."),
            ("Conscription",   Some("Line Infantry"),  Stat { food: 0, atk: 5, prod: 0, mov: 0, def: 3, gold: 0, sci: 0, cult: 0 },    "Mass levy troops."),
            ("Industrial War", Some("Artillery"),      Stat { food: 0, atk: 6, prod: 0, mov: 0, def: 2, gold: 0, sci: 0, cult: 0 },    "Rail-supplied breech-loading guns."),
        ]),
        track!(Track::Economy, [
            ("Gathering",     Some("Settler"),        Stat { food: 2, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Systematic collection of plants."),
            ("Barter",        Some("Trader"),         Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 1, sci: 0, cult: 0 },           "Exchange of surplus goods."),
            ("Farming",       Some("Farmer"),         Stat { food: 3, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },           "Settled cultivation of wheat."),
            ("Herding",       Some("Herder"),        Stat { food: 2, atk: 0, prod: 1, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },   "Domesticated cattle."),
            ("Irrigation",    None,                  Stat { food: 4, atk: 0, prod: 0, mov: 0, def: 0, gold: 1, sci: 0, cult: 0 },   "Canals extend arable land."),
            ("Markets",       Some("Merchant"),       Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 3, sci: 0, cult: 0 },           "Permanent stalls concentrate trade."),
            ("Coinage",       None,                   Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 4, sci: 0, cult: 0 },           "Standardized tokens."),
            ("Galley",        Some("Galley"),         Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 0 },   "Oared warship opens coastal trade."),
            ("Guilds",        Some("Artisan"),        Stat { food: 0, atk: 0, prod: 3, mov: 0, def: 0, gold: 2, sci: 0, cult: 0 },   "Craftsmen associations."),
            ("Banking",       None,                   Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 5, sci: 0, cult: 0 },           "Letters of credit and lending."),
            ("Taxation",      None,                  Stat { food: 0, atk: 0, prod: 1, mov: 0, def: 0, gold: 4, sci: 0, cult: 0 },   "State collection funds armies."),
            ("Carrack",       Some("Carrack"),       Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 4, sci: 0, cult: 0 },   "Deep-hulled ocean trade vessel."),
            ("Joint Stock",   Some("Colonist"),      Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 5, sci: 0, cult: 0 },   "Shared ownership spreads risk."),
            ("Mercantilism",  None,                  Stat { food: 0, atk: 0, prod: 1, mov: 0, def: 0, gold: 6, sci: 0, cult: 0 },   "State-directed surplus trade."),
            ("Plantation",    Some("Overseer"),      Stat { food: 6, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 0 },   "Large-scale cash crops."),
            ("Manufacture",   Some("Engineer"),      Stat { food: 0, atk: 0, prod: 5, mov: 0, def: 0, gold: 2, sci: 0, cult: 0 },   "Centralized craft production."),
            ("Steam Power",   None,                  Stat { food: 0, atk: 0, prod: 7, mov: 0, def: 0, gold: 3, sci: 0, cult: 0 },   "Coal-fired engines."),
            ("Cotton Gin",    None,                  Stat { food: 5, atk: 0, prod: 5, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 },   "Mechanical fiber separation."),
            ("Rail Trade",    Some("Rail Merchant"), Stat { food: 0, atk: 0, prod: 2, mov: 0, def: 0, gold: 8, sci: 0, cult: 0 },   "Locomotives connect markets."),
            ("Stock Exchange",None,                  Stat { food: 0, atk: 0, prod: 3, mov: 0, def: 0, gold: 10, sci: 0, cult: 0 },  "Public equity markets."),
        ]),
        track!(Track::Infrastructure, [
            ("Fire",           Some("Worker"),    Stat { food: 1, atk: 0, prod: 1, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Controlled flame for warmth."),
            ("Shelter",        None,              Stat { food: 2, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Hides form weatherproof dwellings."),
            ("Pottery",        None,              Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 1, sci: 0, cult: 0 }, "Fired clay stores grain."),
            ("Well",           None,              Stat { food: 2, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Deep shafts access groundwater."),
            ("Roads",          None,              Stat { food: 0, atk: 0, prod: 0, mov: 1, def: 0, gold: 1, sci: 0, cult: 0 }, "Packed earth connects settlements."),
            ("Masonry",        Some("Mason"),     Stat { food: 0, atk: 0, prod: 2, mov: 0, def: 1, gold: 0, sci: 0, cult: 0 }, "Cut stone for walls and towers."),
            ("Aqueduct",       None,              Stat { food: 3, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Channels bring water to cities."),
            ("Granary",        None,              Stat { food: 4, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Sealed silos protect harvests."),
            ("Harbor",         Some("Galley"),    Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 0 }, "Docks enable maritime trade."),
            ("Paved Roads",    None,              Stat { food: 0, atk: 0, prod: 0, mov: 2, def: 0, gold: 2, sci: 0, cult: 0 }, "Cobbled surfaces survive loads."),
            ("Castle",         Some("Garrison"),  Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 3, gold: 1, sci: 0, cult: 0 }, "Stone keeps anchor defense."),
            ("Windmill",       None,              Stat { food: 1, atk: 0, prod: 3, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Wind-driven millstones."),
            ("Sewers",         None,              Stat { food: 2, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 0 }, "Channels remove city waste."),
            ("Lighthouse",     Some("Carrack"),   Stat { food: 0, atk: 0, prod: 0, mov: 1, def: 0, gold: 3, sci: 0, cult: 0 }, "Beacons guide ships to port."),
            ("Cathedral",      Some("Priest"),    Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 1, sci: 0, cult: 3 }, "Stone churches serve faith."),
            ("Canal",          None,              Stat { food: 0, atk: 0, prod: 0, mov: 2, def: 0, gold: 4, sci: 0, cult: 0 }, "Waterways bypass terrain."),
            ("Frigate",        Some("Frigate"),   Stat { food: 0, atk: 2, prod: 0, mov: 0, def: 0, gold: 4, sci: 0, cult: 0 }, "Broadside warship."),
            ("Iron Bridge",    None,              Stat { food: 0, atk: 0, prod: 2, mov: 3, def: 0, gold: 0, sci: 0, cult: 0 }, "Cast iron spans rivers."),
            ("Telegraph",      None,              Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 4, sci: 2, cult: 0 }, "Electrical signals transmit orders."),
            ("Railroad",       Some("Ironclad"),  Stat { food: 0, atk: 0, prod: 4, mov: 5, def: 0, gold: 0, sci: 0, cult: 0 }, "Steam rail reshapes the map."),
        ]),
        track!(Track::Culture, [
            ("Language",       None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 1 }, "Spoken symbols allow communication."),
            ("Mythology",      None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 2 }, "Shared stories bind communities."),
            ("Ritual",         Some("Shaman"),  Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 2 }, "Ceremonies mark seasons."),
            ("Painting",       None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 1, sci: 0, cult: 2 }, "Pigment records life and belief."),
            ("Writing",        Some("Scribe"),  Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 1, cult: 3 }, "Symbols store knowledge."),
            ("Polytheism",     Some("Priest"),  Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 1, sci: 0, cult: 3 }, "Multiple deities govern life."),
            ("Epic Poetry",    None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 1, cult: 3 }, "Verse immortalizes heroes."),
            ("Drama",          None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 2, cult: 3 }, "Theater explores moral themes."),
            ("Philosophy",     Some("Scholar"), Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 3, cult: 3 }, "Systematic reasoning."),
            ("Monotheism",     None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 1, gold: 0, sci: 0, cult: 4 }, "One god unifies communities."),
            ("Chivalry",       Some("Paladin"), Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 3, gold: 0, sci: 0, cult: 3 }, "Knightly code and valor."),
            ("Scholasticism",  None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 3, cult: 4 }, "Cathedral schools preserve texts."),
            ("Renais. Art",    Some("Artist"),  Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 5 }, "Perspective transforms painting."),
            ("Humanism",       None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 3, cult: 4 }, "Human dignity and reason."),
            ("Nation-State",   None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 5 }, "Shared language defines nations."),
            ("Journalism",     None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 5 }, "Press shapes public opinion."),
            ("Romanticism",    None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 0, cult: 6 }, "Emotion over reason."),
            ("Pub. Schools",   Some("Teacher"), Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 4, cult: 5 }, "Literacy programs educate all."),
            ("Museums",        None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 2, sci: 0, cult: 7 }, "Collections preserve heritage."),
            ("Mass Media",     None,            Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 3, sci: 0, cult: 8 }, "Newspapers reach millions."),
        ]),
        track!(Track::Science, [
            ("Observation",   None,              Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 1, cult: 0 },            "Careful watching of nature."),
            ("Mathematics",   None,              Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 2, cult: 0 },            "Counting and early algebra."),
            ("Astronomy",     Some("Explorer"),  Stat { food: 0, atk: 0, prod: 0, mov: 1, def: 0, gold: 0, sci: 2, cult: 0 },     "Celestial tracking for nav."),
            ("Medicine",      Some("Physician"), Stat { food: 1, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 2, cult: 0 },    "Herbal treatments."),
            ("Geometry",      None,             Stat { food: 0, atk: 0, prod: 1, mov: 0, def: 0, gold: 0, sci: 3, cult: 0 },    "Proofs underpin architecture."),
            ("Alchemy",       None,             Stat { food: 0, atk: 0, prod: 2, mov: 0, def: 0, gold: 0, sci: 2, cult: 0 },    "Proto-chemistry of materials."),
            ("Optics",        None,             Stat { food: 0, atk: 1, prod: 0, mov: 0, def: 0, gold: 0, sci: 3, cult: 0 },     "Lenses improve reconnaissance."),
            ("Cartography",   Some("Explorer"), Stat { food: 0, atk: 0, prod: 0, mov: 2, def: 0, gold: 0, sci: 3, cult: 0 },     "Accurate maps guide exploration."),
            ("Anatomy",       Some("Surgeon"),  Stat { food: 2, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 3, cult: 0 },    "Study of the human body."),
            ("Heliocentric",  None,             Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 4, cult: 0 },            "Sun-centered model confirmed."),
            ("Sci. Method",   None,             Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 5, cult: 0 },            "Hypothesis and falsification."),
            ("Printing Press",None,             Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 4, cult: 2 },    "Mass reproduction of texts."),
            ("Calculus",      None,             Stat { food: 0, atk: 0, prod: 1, mov: 0, def: 0, gold: 0, sci: 5, cult: 0 },    "Rates of change and physics."),
            ("Electricity",   None,             Stat { food: 0, atk: 0, prod: 2, mov: 0, def: 0, gold: 0, sci: 5, cult: 0 },    "Current electricity harnessed."),
            ("Chemistry",     Some("Chemist"),  Stat { food: 0, atk: 0, prod: 3, mov: 0, def: 0, gold: 0, sci: 5, cult: 0 },    "Periodic table drives industry."),
            ("Thermodynamics",None,             Stat { food: 0, atk: 0, prod: 4, mov: 0, def: 0, gold: 0, sci: 5, cult: 0 },    "Heat-work engine design."),
            ("Evolution",     None,             Stat { food: 0, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 5, cult: 3 },    "Natural selection explained."),
            ("Germ Theory",   Some("Field Medic"),Stat { food: 3, atk: 0, prod: 0, mov: 0, def: 0, gold: 0, sci: 5, cult: 0 },    "Microbes cause disease."),
            ("Electronics",   None,              Stat { food: 0, atk: 0, prod: 2, mov: 0, def: 0, gold: 0, sci: 7, cult: 0 },    "Vacuum tubes handle signals."),
            ("Industrialism", None,             Stat { food: 0, atk: 0, prod: 5, mov: 0, def: 0, gold: 0, sci: 8, cult: 0 },    "Machine production transforms all."),
        ]),
    ]
}

#[derive(PartialEq)]
enum Screen { Map, Tech, Help, City(usize) }

struct Unit {
    x: usize,
    y: usize,
    name: &'static str,
    owner: usize,
    health: i32,
    moved: bool,
}

struct City {
    x: usize,
    y: usize,
    name: String,
    owner: usize,
    /// None means no production order — blocks human end-turn
    build_queue: Option<(&'static str, i32)>, // (unit name, gold cost)
    build_progress: i32,
}

struct Player {
    name: String,
    is_ai: bool,
    gold: i32,
    food: i32,
    science: i32,
    culture: i32,
    production: i32,
    gold_inc: i32,
    food_inc: i32,
    sci_inc: i32,
    cult_inc: i32,
    prod_inc: i32,
    researched: Vec<Vec<bool>>,
    research_progress: Option<(usize, usize, u32)>,
}

struct App {
    screen: Screen,
    map: Vec<Vec<Tile>>,
    players: Vec<Player>,
    current_player: usize,
    cities: Vec<City>,
    units: Vec<Unit>,
    selected_unit: Option<usize>,
    cam_x: usize,
    cam_y: usize,
    tech_cursor: (usize, usize),
    techs: Vec<(Track, Vec<Tech>)>,
    turn: u32,
    status: String,
}

impl App {
    fn new() -> Self {
        let seed = rand::thread_rng().r#gen::<u32>();
        let map = generate_map(seed);

        // All passable land tiles for spawning
        let land_tiles: Vec<(usize, usize)> = (0..MAP_H)
            .flat_map(|y| (0..MAP_W).map(move |x| (x, y)))
            .filter(|&(x, y)| map[y][x].passable())
            .collect();

        let mut rng = rand::thread_rng();

        // Pick spawn far from all existing occupied positions
        let pick_spawn = |occupied: &Vec<(usize, usize)>, rng: &mut rand::rngs::ThreadRng| -> (usize, usize) {
            let mut best = land_tiles[rng.gen_range(0..land_tiles.len())];
            let mut best_dist = 0usize;
            for _ in 0..200 {
                let cand = land_tiles[rng.gen_range(0..land_tiles.len())];
                let min_d = occupied.iter().map(|&(ox, oy)| {
                    let dx = (cand.0 as i32 - ox as i32).unsigned_abs() as usize;
                    let dy = (cand.1 as i32 - oy as i32).unsigned_abs() as usize;
                    dx + dy
                }).min().unwrap_or(usize::MAX);
                if min_d > best_dist { best_dist = min_d; best = cand; }
            }
            best
        };

        let mut occupied: Vec<(usize, usize)> = Vec::new();
        let (sx, sy) = find_start(&map);
        occupied.push((sx, sy));

        let p1 = Player {
            name: "Player 1".into(), is_ai: false,
            gold: 50, food: 10, science: 0, culture: 0, production: 0,
            gold_inc: 5, food_inc: 2, sci_inc: 1, cult_inc: 1, prod_inc: 2,
            researched: vec![vec![false; TECH_COLS]; TECH_ROWS],
            research_progress: None,
        };

        let mut players = vec![p1];
        let mut cities: Vec<City> = vec![City {
            x: sx, y: sy, name: "Capital".into(), owner: 0,
            build_queue: None, build_progress: 0,
        }];
        let mut units: Vec<Unit> = vec![
            Unit { x: sx, y: sy, name: "Warrior", owner: 0, health: 100, moved: false }
        ];

        for i in 0..NUM_AI_PLAYERS {
            let pid = i + 1;
            let (ax, ay) = pick_spawn(&occupied, &mut rng);
            occupied.push((ax, ay));
            players.push(Player {
                name: format!("AI {}", pid), is_ai: true,
                gold: 50, food: 10, science: 0, culture: 0, production: 0,
                gold_inc: 5, food_inc: 2, sci_inc: 1, cult_inc: 1, prod_inc: 2,
                researched: vec![vec![false; TECH_COLS]; TECH_ROWS],
                research_progress: None,
            });
            cities.push(City {
                x: ax, y: ay, name: format!("AI Nest {}", pid), owner: pid,
                build_queue: None, build_progress: 0,
            });
            units.push(Unit { x: ax, y: ay, name: "Warrior", owner: pid, health: 100, moved: false });
        }

        App {
            screen: Screen::Map, map, players,
            current_player: 0, cities, units,
            selected_unit: Some(0),
            cam_x: sx.saturating_sub(30), cam_y: sy.saturating_sub(14),
            tech_cursor: (0, 0), techs: all_techs(), turn: 1,
            status: "Capital founded! Move all units, pick research (T), set city build (C) — then Enter".into(),
        }
    }

    fn can_research(&self, col: usize, row: usize) -> bool {
        self.can_research_for(self.current_player, col, row)
    }

    fn can_research_for(&self, pid: usize, col: usize, row: usize) -> bool {
        let p = &self.players[pid];
        if p.researched[row][col] { return false; }
        if p.research_progress.is_some() { return false; }
        if col == 0 { return true; }
        if p.researched[row][col - 1] { return true; }
        if row > 0 && p.researched[row - 1][col] { return true; }
        if row < TECH_ROWS - 1 && p.researched[row + 1][col] { return true; }
        false
    }

    fn start_research(&mut self, col: usize, row: usize) {
        if self.can_research(col, row) {
            let cost = self.techs[row].1[col].cost;
            self.players[self.current_player].research_progress = Some((col, row, cost));
            let name = self.techs[row].1[col].name;
            self.status = format!("Researching {} ({} turns)…", name, cost);
        } else {
            let p = &self.players[self.current_player];
            if p.research_progress.is_some() { self.status = "Already researching!".into(); }
            else if p.researched[row][col]    { self.status = "Already researched.".into(); }
            else                              { self.status = "Locked — research adjacent first.".into(); }
        }
    }

    /// Human-facing blockers that must be cleared before end-turn.
    fn end_turn_blockers(&self) -> Vec<String> {
        let mut v = Vec::new();

        let unmoved: Vec<_> = self.units.iter()
            .filter(|u| u.owner == self.current_player && !u.moved)
            .map(|u| u.name).collect();
        if !unmoved.is_empty() {
            v.push(format!("Move all units ({} remaining: {})", unmoved.len(), unmoved.join(", ")));
        }

        if self.players[self.current_player].research_progress.is_none() {
            v.push("Choose a technology to research (T)".into());
        }

        let idle: Vec<_> = self.cities.iter()
            .filter(|c| c.owner == self.current_player && c.build_queue.is_none())
            .map(|c| c.name.clone()).collect();
        if !idle.is_empty() {
            v.push(format!("Set production in: {} (C)", idle.join(", ")));
        }

        v
    }

    fn end_turn(&mut self) {
        if !self.players[self.current_player].is_ai {
            let blockers = self.end_turn_blockers();
            if !blockers.is_empty() {
                self.status = blockers.join("  |  ");
                return;
            }
        }

        // Resources & research
        {
            let p = &mut self.players[self.current_player];
            p.gold += p.gold_inc; p.food += p.food_inc;
            p.science += p.sci_inc; p.culture += p.cult_inc; p.production += p.prod_inc;

            if let Some((col, row, left)) = p.research_progress {
                if left <= 1 {
                    p.researched[row][col] = true;
                    p.research_progress = None;
                    let tech = self.techs[row].1[col].clone();
                    apply_yields(&tech.stats, p);
                    if !p.is_ai { self.status = format!("Discovered: {}!", tech.name); }
                } else {
                    p.research_progress = Some((col, row, left - 1));
                }
            }
        }

        // City production: 1 gold/turn toward build cost
        {
            let pid = self.current_player;
            let mut to_spawn: Vec<(usize, usize, &'static str)> = Vec::new();
            for city in self.cities.iter_mut().filter(|c| c.owner == pid) {
                if let Some((unit_name, cost)) = city.build_queue {
                    city.build_progress += 1;
                    if city.build_progress >= cost {
                        to_spawn.push((city.x, city.y, unit_name));
                        city.build_queue = None;
                        city.build_progress = 0;
                    }
                }
            }
            let is_ai = self.players[pid].is_ai;
            for (cx, cy, name) in to_spawn {
                self.players[pid].gold -= 1;
                self.units.push(Unit { x: cx, y: cy, name, owner: pid, health: 100, moved: true });
                if !is_ai { self.status = format!("{} finished training!", name); }
            }
        }

        // Advance player
        self.current_player = (self.current_player + 1) % self.players.len();
        if self.current_player == 0 { self.turn += 1; }

        // Reset units for incoming player
        for u in &mut self.units {
            if u.owner == self.current_player { u.moved = false; }
        }

        if self.players[self.current_player].is_ai {
            self.do_ai_turn();
            self.end_turn();
        } else {
            self.next_unit();
            self.status = format!("Turn {}: Move all units, set research & city builds, then Enter.", self.turn);
        }
    }

    fn do_ai_turn(&mut self) {
        let pid = self.current_player;

        // Research
        if self.players[pid].research_progress.is_none() {
            'r: for r in 0..TECH_ROWS {
                for c in 0..TECH_COLS {
                    if self.can_research_for(pid, c, r) {
                        let cost = self.techs[r].1[c].cost;
                        self.players[pid].research_progress = Some((c, r, cost));
                        break 'r;
                    }
                }
            }
        }

        // City builds
        {
            let available = self.get_available_units(pid);
            if let Some(&(unit_name, cost)) = available.first() {
                for city in self.cities.iter_mut().filter(|c| c.owner == pid && c.build_queue.is_none()) {
                    city.build_queue = Some((unit_name, cost));
                }
            }
        }

        // Move units
        let mut rng = rand::thread_rng();
        for i in 0..self.units.len() {
            if self.units[i].owner == pid && !self.units[i].moved {
                let dx = rng.gen_range(-1i32..=1);
                let dy = rng.gen_range(-1i32..=1);
                let nx = (self.units[i].x as i32 + dx).clamp(0, MAP_W as i32 - 1) as usize;
                let ny = (self.units[i].y as i32 + dy).clamp(0, MAP_H as i32 - 1) as usize;
                if self.map[ny][nx].passable() {
                    self.units[i].x = nx;
                    self.units[i].y = ny;
                }
                self.units[i].moved = true;
            }
        }
    }

    fn move_unit(&mut self, dx: i32, dy: i32) {
        let uidx = match self.selected_unit { Some(i) => i, None => return };
        if self.units[uidx].moved { self.status = "Unit already moved this turn.".into(); return; }

        let nx = (self.units[uidx].x as i32 + dx).clamp(0, MAP_W as i32 - 1) as usize;
        let ny = (self.units[uidx].y as i32 + dy).clamp(0, MAP_H as i32 - 1) as usize;

        if self.map[ny][nx].passable() {
            self.units[uidx].x = nx;
            self.units[uidx].y = ny;
            self.units[uidx].moved = true;

            let vw = 70usize; let vh = 30usize;
            if nx < self.cam_x + 10 { self.cam_x = nx.saturating_sub(10); }
            if nx > self.cam_x + vw - 10 { self.cam_x = (nx + 10).saturating_sub(vw); }
            if ny < self.cam_y + 6 { self.cam_y = ny.saturating_sub(6); }
            if ny > self.cam_y + vh - 6 { self.cam_y = (ny + 6).saturating_sub(vh); }

            self.next_unit();
            let b = self.end_turn_blockers();
            self.status = if b.is_empty() { "All actions complete — press Enter to end turn.".into() }
                          else { b.join("  |  ") };
        } else {
            self.status = "Can't move there (impassable).".into();
        }
    }

    fn next_unit(&mut self) {
        let start = self.selected_unit.map(|i| i + 1).unwrap_or(0);
        for i in 0..self.units.len() {
            let idx = (start + i) % self.units.len();
            if self.units[idx].owner == self.current_player && !self.units[idx].moved {
                self.selected_unit = Some(idx);
                let u = &self.units[idx];
                self.cam_x = u.x.saturating_sub(35);
                self.cam_y = u.y.saturating_sub(15);
                return;
            }
        }
        self.selected_unit = self.units.iter().position(|u| u.owner == self.current_player);
    }

    fn found_city(&mut self) {
        let uidx = match self.selected_unit { Some(i) => i, None => return };
        let (ux, uy, owner) = (self.units[uidx].x, self.units[uidx].y, self.units[uidx].owner);
        if self.units[uidx].name == "Settler" {
            self.cities.push(City {
                x: ux, y: uy, name: format!("City {}", self.cities.len() + 1),
                owner, build_queue: None, build_progress: 0,
            });
            self.units.remove(uidx);
            self.selected_unit = None;
            self.next_unit();
            self.status = "New city founded! Set its production order (C).".into();
        } else {
            self.status = "Only Settlers can found cities!".into();
        }
    }

    fn get_available_units(&self, p_idx: usize) -> Vec<(&'static str, i32)> {
        let p = &self.players[p_idx];
        let mut available: Vec<(&'static str, i32)> = vec![("Warrior", 20), ("Settler", 80)];
        for r in 0..TECH_ROWS {
            for c in 0..TECH_COLS {
                if p.researched[r][c] {
                    if let Some(unit_name) = self.techs[r].1[c].unit {
                        if unit_name != "Warrior" && unit_name != "Settler" {
                            available.push((unit_name, 30 + (c as i32 * 10)));
                        }
                    }
                }
            }
        }
        available
    }

    fn set_city_build(&mut self, city_idx: usize, unit_idx: usize) {
        let available = self.get_available_units(self.current_player);
        if unit_idx >= available.len() { return; }
        let (name, cost) = available[unit_idx];
        let city = &mut self.cities[city_idx];
        city.build_queue = Some((name, cost));
        city.build_progress = 0;
        self.screen = Screen::Map;
        let b = self.end_turn_blockers();
        self.status = if b.is_empty() {
            format!("{} queued — all done! Press Enter to end turn.", name)
        } else {
            format!("{} queued.  Still needed: {}", name, b.join("  |  "))
        };
    }
}

fn apply_yields(stats: &Stat, p: &mut Player) {
    p.gold_inc += stats.gold;
    p.food_inc += stats.food;
    p.sci_inc  += stats.sci;
    p.cult_inc += stats.cult;
    p.prod_inc += stats.prod;
}

fn generate_map(seed: u32) -> Vec<Vec<Tile>> {
    let perlin = Perlin::new(seed);
    let scale = 0.055;
    (0..MAP_H).map(|y| {
        (0..MAP_W).map(|x| {
            let nx = x as f64 * scale; let ny = y as f64 * scale;
            let h = perlin.get([nx, ny])
                  + 0.5  * perlin.get([nx * 2.1, ny * 2.1])
                  + 0.25 * perlin.get([nx * 4.3, ny * 4.3]);
            Tile::from_height(h / 1.75)
        }).collect()
    }).collect()
}

fn find_start(map: &[Vec<Tile>]) -> (usize, usize) {
    let (cx, cy) = (MAP_W / 2, MAP_H / 2);
    for r in 0..30 {
        for dy in -(r as i32)..=(r as i32) {
            for dx in -(r as i32)..=(r as i32) {
                let x = (cx as i32 + dx).clamp(0, MAP_W as i32 - 1) as usize;
                let y = (cy as i32 + dy).clamp(0, MAP_H as i32 - 1) as usize;
                if map[y][x].passable() { return (x, y); }
            }
        }
    }
    (cx, cy)
}

fn draw(f: &mut Frame, app: &App) {
    let area = f.size();
    match app.screen {
        Screen::Map  => draw_map(f, app, area),
        Screen::Tech => draw_tech(f, app, area),
        Screen::Help => { draw_map(f, app, area); draw_help(f, area); }
        Screen::City(idx) => draw_city(f, app, idx, area),
    }
}

fn draw_map(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(area);

    f.render_widget(MapWidget { app }, chunks[0]);

    let p = &app.players[app.current_player];
    let prog = if let Some((col, row, left)) = p.research_progress {
        format!("  |  {} — {}t left", app.techs[row].1[col].name, left)
    } else {
        "  |  [no research — press T!]".into()
    };

    let city_builds: Vec<String> = app.cities.iter()
        .filter(|c| c.owner == app.current_player)
        .map(|c| match c.build_queue {
            Some((name, _)) => format!("{}: {}", c.name, name),
            None => format!("{}: [IDLE!]", c.name),
        }).collect();

    let status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(format!(" Tr:{:<3}", app.turn), Style::default().fg(Color::Yellow)),
            Span::raw(format!("  G:{}(+{}) F:{}(+{}) Sc:{}(+{}) Cu:{}(+{}) Pr:{}(+{})",
                p.gold, p.gold_inc, p.food, p.food_inc, p.science, p.sci_inc, p.culture, p.cult_inc, p.production, p.prod_inc)),
            Span::styled(prog, Style::default().fg(Color::Cyan)),
            Span::styled(format!("  |  {}", city_builds.join("  ")), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(Span::styled(format!(" {}", app.status), Style::default().fg(Color::Gray))),
    ]).block(Block::default().borders(Borders::TOP));
    f.render_widget(status, chunks[1]);
}

struct MapWidget<'a> { app: &'a App }

impl Widget for MapWidget<'_> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let app = self.app;
        for sy in 0..(area.height as usize) {
            for sx in 0..(area.width as usize) {
                let mx = app.cam_x + sx; let my = app.cam_y + sy;
                if mx >= MAP_W || my >= MAP_H { continue; }
                let tile = app.map[my][mx];
                let bx = area.x + sx as u16; let by = area.y + sy as u16;

                let is_selected = app.selected_unit.map(|i| app.units[i].x == mx && app.units[i].y == my).unwrap_or(false);
                let unit = app.units.iter().find(|u| u.x == mx && u.y == my);
                let city = app.cities.iter().find(|c| c.x == mx && c.y == my);

                if is_selected {
                    buf.get_mut(bx, by).set_symbol("@").set_fg(Color::Green).set_bg(tile.bg()).set_style(Modifier::BOLD);
                } else if let Some(u) = unit {
                    buf.get_mut(bx, by).set_symbol("@").set_fg(owner_color(u.owner)).set_bg(tile.bg());
                } else if let Some(c) = city {
                    buf.get_mut(bx, by).set_symbol("H").set_fg(owner_color(c.owner)).set_bg(tile.bg()).set_style(Modifier::BOLD);
                } else {
                    buf.get_mut(bx, by).set_symbol(tile.glyph()).set_fg(tile.fg()).set_bg(tile.bg());
                }
            }
        }
    }
}

fn owner_color(owner: usize) -> Color {
    match owner {
        0 => Color::White,
        1 => Color::Red,
        2 => Color::Magenta,
        3 => Color::Cyan,
        4 => Color::Green,
        _ => Color::Black,
    }
}

fn draw_city(f: &mut Frame, app: &App, city_idx: usize, area: Rect) {
    let city = &app.cities[city_idx];
    let p = &app.players[app.current_player];

    let block = Block::default().borders(Borders::ALL).title(format!(" City: {} ", city.name))
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1), Constraint::Length(2)])
        .split(inner);

    let build_info = match city.build_queue {
        Some((name, cost)) => format!("  Building: {} (cost {}g, {}g paid)", name, cost, city.build_progress),
        None => "  [No production order — pick one below!]".into(),
    };
    f.render_widget(Paragraph::new(vec![
        Line::from(format!("  Owner: {} | Gold: {} (+{})", p.name, p.gold, p.gold_inc)),
        Line::from(Span::styled(build_info, Style::default().fg(Color::Cyan))),
    ]), chunks[0]);

    let available = app.get_available_units(app.current_player);
    let items: Vec<Line> = available.iter().enumerate().map(|(i, (name, cost))| {
        let is_current = city.build_queue.map(|(n, _)| n == *name).unwrap_or(false);
        let style = if is_current { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) }
                    else if p.gold >= *cost { Style::default().fg(Color::White) }
                    else { Style::default().fg(Color::DarkGray) };
        let marker = if is_current { "►" } else { " " };
        Line::from(Span::styled(format!("  {}[{}] {:<18} cost: {}g", marker, i+1, name, cost), style))
    }).collect();

    f.render_widget(Paragraph::new(items).block(Block::default().title(" Set Production (number key) ")), chunks[1]);
    f.render_widget(Paragraph::new("  M / Esc — back to map").style(Style::default().fg(Color::DarkGray)), chunks[2]);
}

fn draw_tech(f: &mut Frame, app: &App, area: Rect) {
    let cp = app.players.get(app.current_player).expect("player");

    let chunks = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(5)]).split(area);
    let grid_area = chunks[0]; let info_area = chunks[1];

    let cols = Layout::default().direction(Direction::Horizontal)
        .constraints([Constraint::Length(10), Constraint::Min(1)]).split(grid_area);
    let label_col = cols[0]; let cells_col = cols[1];
    let cell_w = (cells_col.width / TECH_COLS as u16).max(3);
    let row_h   = ((grid_area.height.saturating_sub(2)) / TECH_ROWS as u16).max(3);

    let era_names = ["Ancient", "Classical", "Medieval", "Renaissance", "Industrial"];
    for (i, era) in era_names.iter().enumerate() {
        let ex = cells_col.x + (i as u16 * 4) * cell_w;
        let ew = cell_w * 4;
        if ex < cells_col.x + cells_col.width {
            let r = Rect::new(ex, grid_area.y, ew.min(cells_col.width - (ex - cells_col.x)), 1);
            f.render_widget(Paragraph::new(center_pad(era, ew as usize)).style(Style::default().fg(Color::DarkGray)), r);
        }
    }

    for (row_idx, (track, techs)) in app.techs.iter().enumerate() {
        let ry = grid_area.y + 1 + row_idx as u16 * row_h;
        if ry + row_h > grid_area.y + grid_area.height { break; }

        f.render_widget(
            Paragraph::new(track.label()).style(Style::default().fg(track.color()).add_modifier(Modifier::BOLD)),
            Rect::new(label_col.x, ry, label_col.width, row_h),
        );

        for col_idx in 0..TECH_COLS {
            let cx = cells_col.x + col_idx as u16 * cell_w;
            if cx + cell_w > cells_col.x + cells_col.width { break; }
            let crect = Rect::new(cx, ry, cell_w, row_h);
            let tech = &techs[col_idx];
            let is_cursor     = app.tech_cursor == (col_idx, row_idx);
            let is_researched = cp.researched[row_idx][col_idx];
            let in_progress   = cp.research_progress.map(|(c,r,_)| c==col_idx&&r==row_idx).unwrap_or(false);
            let available     = app.can_research(col_idx, row_idx);

            let (fg, bg, bfg) = if is_cursor { (Color::Black, track.color(), Color::White) }
                else if in_progress   { (Color::Cyan,        Color::Reset, Color::Cyan) }
                else if is_researched { (track.dim_color(),  Color::Reset, track.dim_color()) }
                else if available     { (track.color(),      Color::Reset, track.color()) }
                else                  { (Color::DarkGray,    Color::Reset, Color::DarkGray) };

            let name = truncate(tech.name, cell_w.saturating_sub(2) as usize);
            let cost = if is_researched { "ok".into() } else if in_progress { "..".into() } else { format!("{}t", tech.cost) };

            let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(bfg));
            let inner = block.inner(crect);
            f.render_widget(block, crect);
            if inner.height >= 1 {
                f.render_widget(Paragraph::new(name).style(Style::default().fg(fg).bg(bg)), Rect::new(inner.x, inner.y, inner.width, 1));
            }
            if inner.height >= 2 {
                let cost_fg = if is_researched { Color::Green } else { Color::DarkGray };
                f.render_widget(Paragraph::new(cost).style(Style::default().fg(cost_fg)), Rect::new(inner.x, inner.y+1, inner.width, 1));
            }
        }
    }

    let (cc, cr) = app.tech_cursor;
    let tech  = &app.techs[cr].1[cc];
    let track = &app.techs[cr].0;
    let state = if cp.researched[cr][cc] { "Researched".into() }
                else if cp.research_progress.map(|(c,r,_)| c==cc&&r==cr).unwrap_or(false) { "In progress".into() }
                else if app.can_research(cc, cr) { format!("Available — {} turns", tech.cost) }
                else { "Locked".into() };
    let unit_str = tech.unit.map(|u| format!("  Unit: {}  ", u)).unwrap_or_default();

    f.render_widget(Paragraph::new(vec![
        Line::from(vec![
            Span::styled(format!(" {} ", tech.name), Style::default().fg(track.color()).add_modifier(Modifier::BOLD)),
            Span::styled(format!("[{}]  ", track.label()), Style::default().fg(Color::DarkGray)),
            Span::styled(state, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(Span::styled(format!(
            "{} atk: {} cult: {} def: {} food: {} gold: {} mov: {} prod: {} sci: {} —  {}", 
            unit_str, tech.stats.atk, tech.stats.cult, tech.stats.def, tech.stats.food, tech.stats.gold, tech.stats.mov, tech.stats.prod, tech.stats.sci, 
            tech.desc
        ), Style::default().fg(Color::Gray))),
        Line::from(Span::styled(" ↑↓←→ navigate    Enter = research    M = back to map", Style::default().fg(Color::DarkGray))),
    ]).block(Block::default().borders(Borders::TOP)), info_area);
}

fn draw_help(f: &mut Frame, area: Rect) {
    let w = 52u16; let h = 28u16;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let popup = Rect::new(x, y, w, h);
    f.render_widget(Clear, popup);
    f.render_widget(Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled("  Controls", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(Span::styled("  Map:", Style::default().fg(Color::Cyan))),
        Line::from("    W A S D / arrows — move selected unit"),
        Line::from("    Tab              — cycle to next unmoved unit"),
        Line::from("    B                — found city (Settler only)"),
        Line::from("    C                — open city production menu"),
        Line::from("    T                — open tech grid"),
        Line::from("    H                — toggle help"),
        Line::from("    Enter            — end turn (when all done)"),
        Line::from("    Q                — quit"),
        Line::from(""),
        Line::from(Span::styled("  To end your turn you MUST:", Style::default().fg(Color::Yellow))),
        Line::from("    • Move every single unit"),
        Line::from("    • Have an active research (T → Enter on a tech)"),
        Line::from("    • Have a production order in every city (C → number)"),
        Line::from(""),
        Line::from(Span::styled("  City menu:", Style::default().fg(Color::Cyan))),
        Line::from("    1-9     — set production order for this city"),
        Line::from("    M / Esc — return to map"),
        Line::from(""),
        Line::from(Span::styled("  Map legend:", Style::default().fg(Color::Cyan))),
        Line::from("    ≈ deep  ~ shallow  . sand  , plains"),
        Line::from("    f forest  n hills  ^ mountain  * snow"),
        Line::from("    @ unit  H city  (white=you, red/magenta/cyan=AI)"),
        Line::from(""),
    ]).block(Block::default().borders(Borders::ALL).title(" Help — H to close ")
        .border_style(Style::default().fg(Color::Yellow))), popup);
}

fn center_pad(s: &str, width: usize) -> String {
    if s.len() >= width { return s.to_string(); }
    let pad = (width - s.len()) / 2;
    format!("{}{}", " ".repeat(pad), s)
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 { return String::new(); }
    if s.len() <= max { s.to_string() } else { s[..max].to_string() }
}

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();

    loop {
        terminal.draw(|f| draw(f, &app))?;
        if let Event::Key(key) = event::read()? {
            if key.kind == event::KeyEventKind::Press {
                match app.screen {
                    Screen::Map | Screen::Help => match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => break,
                        KeyCode::Char('h') | KeyCode::Char('H') => {
                            app.screen = if app.screen == Screen::Help { Screen::Map } else { Screen::Help };
                        }
                        KeyCode::Char('t') | KeyCode::Char('T') => { app.screen = Screen::Tech; }
                        KeyCode::Char('w') | KeyCode::Up    => app.move_unit(0, -1),
                        KeyCode::Char('s') | KeyCode::Down  => app.move_unit(0,  1),
                        KeyCode::Char('a') | KeyCode::Left  => app.move_unit(-1, 0),
                        KeyCode::Char('d') | KeyCode::Right => app.move_unit(1,  0),
                        KeyCode::Char('b') | KeyCode::Char('B') => app.found_city(),
                        KeyCode::Char('c') | KeyCode::Char('C') => {
                            let city_idx = app.selected_unit.and_then(|uidx| {
                                let (ux, uy) = (app.units[uidx].x, app.units[uidx].y);
                                app.cities.iter().position(|c| c.x == ux && c.y == uy && c.owner == app.current_player)
                            }).or_else(|| app.cities.iter().position(|c| c.owner == app.current_player));
                            if let Some(cidx) = city_idx {
                                app.screen = Screen::City(cidx);
                            } else {
                                app.status = "No friendly city found!".into();
                            }
                        }
                        KeyCode::Tab   => app.next_unit(),
                        KeyCode::Enter => app.end_turn(),
                        _ => {}
                    },
                    Screen::Tech => match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => break,
                        KeyCode::Char('m') | KeyCode::Char('M') | KeyCode::Esc => { app.screen = Screen::Map; }
                        KeyCode::Char('h') | KeyCode::Char('H') => { app.screen = Screen::Help; }
                        KeyCode::Up    => { if app.tech_cursor.1 > 0 { app.tech_cursor.1 -= 1; } }
                        KeyCode::Down  => { if app.tech_cursor.1 < TECH_ROWS-1 { app.tech_cursor.1 += 1; } }
                        KeyCode::Left  => { if app.tech_cursor.0 > 0 { app.tech_cursor.0 -= 1; } }
                        KeyCode::Right => { if app.tech_cursor.0 < TECH_COLS-1 { app.tech_cursor.0 += 1; } }
                        KeyCode::Enter => { let (c,r) = app.tech_cursor; app.start_research(c, r); }
                        _ => {}
                    },
                    Screen::City(cidx) => match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => break,
                        KeyCode::Char('m') | KeyCode::Char('M') | KeyCode::Esc => { app.screen = Screen::Map; }
                        KeyCode::Char(c) if c.is_ascii_digit() => {
                            let digit = c.to_digit(10).unwrap() as usize;
                            if digit > 0 { app.set_city_build(cidx, digit - 1); }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}