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

#[derive(Clone, Copy, PartialEq)]
enum Tile { DeepWater, ShallowWater, Sand, Plains, Forest, Hills, Mountain, Snow }

impl Tile {
    fn from_height(h: f64) -> Self {
        match h {
            h if h < -0.35 => Tile::DeepWater,
            h if h < -0.05 => Tile::ShallowWater,
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
struct Tech {
    name: &'static str,
    unit: Option<&'static str>,
    stats: &'static str,
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
    let costs: [u32; 20] = [1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,6,6,7,8];
    macro_rules! track {
        ($track:expr, [ $( ($n:expr, $u:expr, $s:expr, $d:expr) ),* $(,)? ]) => {{
            let raw: Vec<(&str, Option<&str>, &str, &str)> = vec![ $( ($n, $u, $s, $d) ),* ];
            let techs: Vec<Tech> = raw.into_iter().enumerate().map(|(i, (n,u,s,d))| Tech {
                name: n, unit: u, stats: s, desc: d, cost: costs[i],
            }).collect();
            ($track, techs)
        }};
    }
    vec![
        track!(Track::Military, [
            ("Foraging",       Some("Scout"),          "+1 food",          "Ranging parties map the land."),
            ("Stone Tools",    Some("Warrior"),        "+1 atk",           "Chipped flint arms the first fighters."),
            ("Spear",          Some("Spearman"),       "+1 atk +1 food",   "Hafted point for hunting and combat."),
            ("Hunting",        Some("Archer"),         "+1 food +1 prod",  "Bow and trap techniques."),
            ("Bronze Weapons", Some("Swordsman"),      "+2 atk",           "Smelted bronze blades."),
            ("Shield Craft",   Some("Heavy Infantry"), "+2 def",           "Wicker and hide shields."),
            ("Phalanx",        Some("Hoplite"),        "+2 atk +2 def",    "Locked-shield spear wall."),
            ("War Chariot",    Some("Chariot"),        "+2 atk +1 mov",    "Horse-drawn shock flanks."),
            ("Iron Forging",   Some("Iron Swordsman"), "+3 atk",           "Harder iron replaces bronze."),
            ("Siege Craft",    Some("Catapult"),       "+2 atk +1 prod",   "Tension engines hurl stones."),
            ("Cavalry",        Some("Knight"),         "+3 atk +2 mov",    "Armored horsemen break lines."),
            ("Crossbow",       Some("Crossbowman"),    "+2 atk +1 def",    "Mechanical bow penetrates plate."),
            ("Plate Armor",    Some("Man-at-Arms"),    "+4 def",           "Full steel plate cuts casualties."),
            ("Cannon",         Some("Cannon"),         "+4 atk",           "Gunpowder shatters stone walls."),
            ("Musketry",       Some("Musketeer"),      "+3 atk +1 def",    "Matchlock firearms replace bows."),
            ("Mil. Drill",     Some("Grenadier"),      "+3 atk +2 def",    "Drill and explosive grenades."),
            ("Field Artillery",Some("Field Gun"),      "+5 atk",           "Mobile cannon supports infantry."),
            ("Rifling",        Some("Rifleman"),       "+4 atk +1 def",    "Grooved barrels improve accuracy."),
            ("Conscription",   Some("Line Infantry"),  "+5 atk +3 def",    "Mass levy troops."),
            ("Industrial War", Some("Artillery"),      "+6 atk +2 def",    "Rail-supplied breech-loading guns."),
        ]),
        track!(Track::Economy, [
            ("Gathering",     Some("Settler"),        "+2 food",           "Systematic collection of plants."),
            ("Barter",        Some("Trader"),         "+1 gold",           "Exchange of surplus goods."),
            ("Farming",       Some("Farmer"),         "+3 food",           "Settled cultivation of wheat."),
            ("Herding",       Some("Herder"),         "+2 food +1 prod",   "Domesticated cattle."),
            ("Irrigation",    None,                   "+4 food +1 gold",   "Canals extend arable land."),
            ("Markets",       Some("Merchant"),       "+3 gold",           "Permanent stalls concentrate trade."),
            ("Coinage",       None,                   "+4 gold",           "Standardized tokens."),
            ("Galley",        Some("Galley"),         "+2 gold +1 food",   "Oared warship opens coastal trade."),
            ("Guilds",        Some("Artisan"),        "+3 prod +2 gold",   "Craftsmen associations."),
            ("Banking",       None,                   "+5 gold",           "Letters of credit and lending."),
            ("Taxation",      None,                   "+4 gold +1 prod",   "State collection funds armies."),
            ("Carrack",       Some("Carrack"),        "+4 gold +1 food",   "Deep-hulled ocean trade vessel."),
            ("Joint Stock",   Some("Colonist"),       "+5 gold +1 food",   "Shared ownership spreads risk."),
            ("Mercantilism",  None,                   "+6 gold +1 prod",   "State-directed surplus trade."),
            ("Plantation",    Some("Overseer"),       "+6 food +2 gold",   "Large-scale cash crops."),
            ("Manufacture",   Some("Engineer"),       "+5 prod +2 gold",   "Centralized craft production."),
            ("Steam Power",   None,                   "+7 prod +3 gold",   "Coal-fired engines."),
            ("Cotton Gin",    None,                   "+5 food +5 prod",   "Mechanical fiber separation."),
            ("Rail Trade",    Some("Rail Merchant"),  "+8 gold +2 prod",   "Locomotives connect markets."),
            ("Stock Exchange",None,                   "+10 gold +3 prod",  "Public equity markets."),
        ]),
        track!(Track::Infrastructure, [
            ("Fire",         Some("Worker"),    "+1 food +1 prod",   "Controlled flame for warmth."),
            ("Shelter",      None,              "+2 food",           "Hides form weatherproof dwellings."),
            ("Pottery",      None,              "+1 food +1 gold",   "Fired clay stores grain."),
            ("Well",         None,              "+2 food +1 pop",    "Deep shafts access groundwater."),
            ("Roads",        None,              "+1 mov +1 gold",    "Packed earth connects settlements."),
            ("Masonry",      Some("Mason"),     "+2 prod +1 def",    "Cut stone for walls and towers."),
            ("Aqueduct",     None,              "+3 food +1 pop",    "Channels bring water to cities."),
            ("Granary",      None,              "+4 food",           "Sealed silos protect harvests."),
            ("Harbor",       Some("Galley"),    "+2 gold +1 food",   "Docks enable maritime trade."),
            ("Paved Roads",  None,              "+2 mov +2 gold",    "Cobbled surfaces survive loads."),
            ("Castle",       Some("Garrison"),  "+3 def +1 gold",    "Stone keeps anchor defense."),
            ("Windmill",     None,              "+3 prod +1 food",   "Wind-driven millstones."),
            ("Sewers",       None,              "+2 food +1 pop",    "Channels remove city waste."),
            ("Lighthouse",   Some("Carrack"),   "+3 gold +1 mov",    "Beacons guide ships to port."),
            ("Cathedral",    Some("Priest"),    "+3 cult +1 gold",   "Stone churches serve faith."),
            ("Canal",        None,              "+4 gold +2 mov",    "Waterways bypass terrain."),
            ("Frigate",      Some("Frigate"),   "+4 gold +2 atk",    "Broadside warship."),
            ("Iron Bridge",  None,              "+3 mov +2 prod",    "Cast iron spans rivers."),
            ("Telegraph",    None,              "+4 gold +2 sci",    "Electrical signals transmit orders."),
            ("Railroad",     Some("Ironclad"),  "+5 mov +4 prod",    "Steam rail reshapes the map."),
        ]),
        track!(Track::Culture, [
            ("Language",      None,             "+1 cult",           "Spoken symbols allow communication."),
            ("Mythology",     None,             "+2 cult",           "Shared stories bind communities."),
            ("Ritual",        Some("Shaman"),   "+2 cult +1 food",   "Ceremonies mark seasons."),
            ("Painting",      None,             "+2 cult +1 gold",   "Pigment records life and belief."),
            ("Writing",       Some("Scribe"),   "+3 cult +1 sci",    "Symbols store knowledge."),
            ("Polytheism",    Some("Priest"),   "+3 cult +1 gold",   "Multiple deities govern life."),
            ("Epic Poetry",   None,             "+3 cult +1 sci",    "Verse immortalizes heroes."),
            ("Drama",         None,             "+3 cult +2 sci",    "Theater explores moral themes."),
            ("Philosophy",    Some("Scholar"),  "+3 cult +3 sci",    "Systematic reasoning."),
            ("Monotheism",    None,             "+4 cult +1 def",    "One god unifies communities."),
            ("Chivalry",      Some("Paladin"),  "+3 cult +3 def",    "Knightly code and valor."),
            ("Scholasticism", None,             "+4 cult +3 sci",    "Cathedral schools preserve texts."),
            ("Renais. Art",   Some("Artist"),   "+5 cult +2 gold",   "Perspective transforms painting."),
            ("Humanism",      None,             "+4 cult +3 sci",    "Human dignity and reason."),
            ("Nation-State",  None,             "+5 cult +2 gold",   "Shared language defines nations."),
            ("Journalism",    None,             "+5 cult +2 gold",   "Press shapes public opinion."),
            ("Romanticism",   None,             "+6 cult",           "Emotion over reason."),
            ("Pub. Schools",  Some("Teacher"),  "+5 cult +4 sci",    "Literacy programs educate all."),
            ("Museums",       None,             "+7 cult +2 gold",   "Collections preserve heritage."),
            ("Mass Media",    None,             "+8 cult +3 gold",   "Newspapers reach millions."),
        ]),
        track!(Track::Science, [
            ("Observation",   None,               "+1 sci",            "Careful watching of nature."),
            ("Mathematics",   None,               "+2 sci",            "Counting and early algebra."),
            ("Astronomy",     Some("Explorer"),   "+2 sci +1 mov",     "Celestial tracking for nav."),
            ("Medicine",      Some("Physician"),  "+2 sci +1 food",    "Herbal treatments."),
            ("Geometry",      None,               "+3 sci +1 prod",    "Proofs underpin architecture."),
            ("Alchemy",       None,               "+2 sci +2 prod",    "Proto-chemistry of materials."),
            ("Optics",        None,               "+3 sci +1 atk",     "Lenses improve reconnaissance."),
            ("Cartography",   Some("Explorer"),   "+3 sci +2 mov",     "Accurate maps guide exploration."),
            ("Anatomy",       Some("Surgeon"),    "+3 sci +2 food",    "Study of the human body."),
            ("Heliocentric",  None,               "+4 sci",            "Sun-centered model confirmed."),
            ("Sci. Method",   None,               "+5 sci",            "Hypothesis and falsification."),
            ("Printing Press",None,               "+4 sci +2 cult",    "Mass reproduction of texts."),
            ("Calculus",      None,               "+5 sci +1 prod",    "Rates of change and physics."),
            ("Electricity",   None,               "+5 sci +2 prod",    "Current electricity harnessed."),
            ("Chemistry",     Some("Chemist"),    "+5 sci +3 prod",    "Periodic table drives industry."),
            ("Thermodynamics",None,               "+5 sci +4 prod",    "Heat-work engine design."),
            ("Evolution",     None,               "+5 sci +3 cult",    "Natural selection explained."),
            ("Germ Theory",   Some("Field Medic"),"+5 sci +3 food",    "Microbes cause disease."),
            ("Electronics",   None,               "+7 sci +2 prod",    "Vacuum tubes handle signals."),
            ("Industrialism", None,               "+8 sci +5 prod",    "Machine production transforms all."),
        ]),
    ]
}

#[derive(PartialEq)]
enum Screen { Map, Tech, Help }

struct Unit { x: usize, y: usize, name: &'static str }

struct App {
    screen: Screen,
    map: Vec<Vec<Tile>>,
    units: Vec<Unit>,
    cam_x: usize,
    cam_y: usize,
    researched: Vec<Vec<bool>>,
    tech_cursor: (usize, usize),
    research_progress: Option<(usize, usize, u32)>,
    techs: Vec<(Track, Vec<Tech>)>,
    turn: u32,
    gold: i32,
    food: i32,
    science: i32,
    culture: i32,
    production: i32,
    status: String,
}

impl App {
    fn new() -> Self {
        let seed = rand::thread_rng().r#gen::<u32>();
        let map = generate_map(seed);
        let (sx, sy) = find_start(&map);
        App {
            screen: Screen::Map,
            map,
            units: vec![Unit { x: sx, y: sy, name: "Settler" }],
            cam_x: sx.saturating_sub(30),
            cam_y: sy.saturating_sub(14),
            researched: vec![vec![false; TECH_COLS]; TECH_ROWS],
            tech_cursor: (0, 0),
            research_progress: None,
            techs: all_techs(),
            turn: 1,
            gold: 0, food: 5, science: 1, culture: 1, production: 2,
            status: String::from("Welcome! T=tech grid  Enter=end turn  H=help"),
        }
    }

    fn can_research(&self, col: usize, row: usize) -> bool {
        if self.researched[row][col] { return false; }
        if self.research_progress.is_some() { return false; }
        if col == 0 { return true; }
        if self.researched[row][col - 1] { return true; }
        if row > 0 && self.researched[row - 1][col] { return true; }
        if row < TECH_ROWS - 1 && self.researched[row + 1][col] { return true; }
        false
    }

    fn start_research(&mut self, col: usize, row: usize) {
        if self.can_research(col, row) {
            let cost = self.techs[row].1[col].cost;
            self.research_progress = Some((col, row, cost));
            let name = self.techs[row].1[col].name;
            self.status = format!("Researching {} ({} turns)...", name, cost);
        } else if self.research_progress.is_some() {
            self.status = "Already researching! Finish current tech first.".into();
        } else if self.researched[row][col] {
            self.status = "Already researched.".into();
        } else {
            self.status = "Locked — research an adjacent tile first.".into();
        }
    }

    fn end_turn(&mut self) {
        self.turn += 1;
        if let Some((col, row, left)) = self.research_progress {
            if left <= 1 {
                self.researched[row][col] = true;
                self.research_progress = None;
                let tech = &self.techs[row].1[col];
                let stats = tech.stats;
                let name = tech.name;
                self.status = format!("Discovered: {}! ({})", name, stats);
                apply_yields(stats, &mut self.gold, &mut self.food,
                             &mut self.science, &mut self.culture, &mut self.production);
            } else {
                self.research_progress = Some((col, row, left - 1));
                let name = self.techs[row].1[col].name;
                self.status = format!("Researching {} ({} turns left)...", name, left - 1);
            }
        } else {
            self.status = format!("Turn {}. Press T to pick a tech to research.", self.turn);
        }
    }

    fn move_unit(&mut self, dx: i32, dy: i32) {
        if self.units.is_empty() { return; }
        let u = &self.units[0];
        let nx = (u.x as i32 + dx).clamp(0, MAP_W as i32 - 1) as usize;
        let ny = (u.y as i32 + dy).clamp(0, MAP_H as i32 - 1) as usize;
        if self.map[ny][nx].passable() {
            self.units[0].x = nx;
            self.units[0].y = ny;
            let vw = 70usize;
            let vh = 30usize;
            if nx < self.cam_x + 6 { self.cam_x = nx.saturating_sub(6); }
            if nx > self.cam_x + vw - 6 { self.cam_x = (nx + 6).saturating_sub(vw); }
            if ny < self.cam_y + 4 { self.cam_y = ny.saturating_sub(4); }
            if ny > self.cam_y + vh - 4 { self.cam_y = (ny + 4).saturating_sub(vh); }
        }
    }
}

fn apply_yields(stats: &str, gold: &mut i32, food: &mut i32,
                sci: &mut i32, cult: &mut i32, prod: &mut i32) {
    if stats.contains("gold") { *gold  += 2; }
    if stats.contains("food") { *food  += 2; }
    if stats.contains("sci")  { *sci   += 2; }
    if stats.contains("cult") { *cult  += 2; }
    if stats.contains("prod") { *prod  += 2; }
}

fn generate_map(seed: u32) -> Vec<Vec<Tile>> {
    let perlin = Perlin::new(seed);
    let scale = 0.055;
    (0..MAP_H).map(|y| {
        (0..MAP_W).map(|x| {
            let nx = x as f64 * scale;
            let ny = y as f64 * scale;
            let h = perlin.get([nx, ny])
                  + 0.5  * perlin.get([nx * 2.1, ny * 2.1])
                  + 0.25 * perlin.get([nx * 4.3, ny * 4.3]);
            Tile::from_height(h / 1.75)
        }).collect()
    }).collect()
}

fn find_start(map: &[Vec<Tile>]) -> (usize, usize) {
    let cx = MAP_W / 2;
    let cy = MAP_H / 2;
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
    }
}

fn draw_map(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(area);

    f.render_widget(MapWidget { app }, chunks[0]);

    let prog = if let Some((col, row, left)) = app.research_progress {
        let name = app.techs[row].1[col].name;
        format!("  |  {} — {}t left", name, left)
    } else { String::new() };

    let status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(format!(" Tr:{:<3}", app.turn), Style::default().fg(Color::Yellow)),
            Span::raw(format!("  G:{:<4} F:{:<4} Sc:{:<4} Cu:{:<4} Pr:{:<4}", app.gold, app.food, app.science, app.culture, app.production)),
            Span::styled(prog, Style::default().fg(Color::Cyan)),
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
                let mx = app.cam_x + sx;
                let my = app.cam_y + sy;
                if mx >= MAP_W || my >= MAP_H { continue; }
                let tile = app.map[my][mx];
                let bx = area.x + sx as u16;
                let by = area.y + sy as u16;
                if app.units.iter().any(|u| u.x == mx && u.y == my) {
                    buf.get_mut(bx, by).set_symbol("@").set_fg(Color::White).set_bg(tile.bg());
                } else {
                    buf.get_mut(bx, by).set_symbol(tile.glyph()).set_fg(tile.fg()).set_bg(tile.bg());
                }
            }
        }
    }
}

fn draw_tech(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(5)])
        .split(area);

    let grid_area = chunks[0];
    let info_area = chunks[1];

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(10), Constraint::Min(1)])
        .split(grid_area);

    let label_col = cols[0];
    let cells_col = cols[1];
    let cell_w = (cells_col.width / TECH_COLS as u16).max(3);
    let row_h   = ((grid_area.height.saturating_sub(2)) / TECH_ROWS as u16).max(3);

    // Era header
    let era_names = ["Ancient", "Classical", "Medieval", "Renaissance", "Industrial"];
    for (i, era) in era_names.iter().enumerate() {
        let ex = cells_col.x + (i as u16 * 4) * cell_w;
        let ew = cell_w * 4;
        if ex < cells_col.x + cells_col.width {
            let r = Rect::new(ex, grid_area.y, ew.min(cells_col.width - (ex - cells_col.x)), 1);
            f.render_widget(
                Paragraph::new(center_pad(era, ew as usize))
                    .style(Style::default().fg(Color::DarkGray)),
                r,
            );
        }
    }

    for (row_idx, (track, techs)) in app.techs.iter().enumerate() {
        let ry = grid_area.y + 1 + row_idx as u16 * row_h;
        if ry + row_h > grid_area.y + grid_area.height { break; }

        // Row label
        let lrect = Rect::new(label_col.x, ry, label_col.width, row_h);
        f.render_widget(
            Paragraph::new(track.label())
                .style(Style::default().fg(track.color()).add_modifier(Modifier::BOLD)),
            lrect,
        );

        for col_idx in 0..TECH_COLS {
            let cx = cells_col.x + col_idx as u16 * cell_w;
            if cx + cell_w > cells_col.x + cells_col.width { break; }
            let crect = Rect::new(cx, ry, cell_w, row_h);

            let tech = &techs[col_idx];
            let is_cursor     = app.tech_cursor == (col_idx, row_idx);
            let is_researched = app.researched[row_idx][col_idx];
            let in_progress   = app.research_progress.map(|(c,r,_)| c==col_idx && r==row_idx).unwrap_or(false);
            let available     = app.can_research(col_idx, row_idx);

            let (fg, bg, bfg) = if is_cursor {
                (Color::Black, track.color(), Color::White)
            } else if in_progress {
                (Color::Cyan, Color::Reset, Color::Cyan)
            } else if is_researched {
                (track.dim_color(), Color::Reset, track.dim_color())
            } else if available {
                (track.color(), Color::Reset, track.color())
            } else {
                (Color::DarkGray, Color::Reset, Color::DarkGray)
            };

            let name = truncate(tech.name, cell_w.saturating_sub(2) as usize);
            let cost = if is_researched { "ok".into() }
                       else if in_progress { "..".into() }
                       else { format!("{}t", tech.cost) };

            let block = Block::default().borders(Borders::ALL)
                .border_style(Style::default().fg(bfg));
            let inner = block.inner(crect);
            f.render_widget(block, crect);

            if inner.height >= 1 {
                f.render_widget(
                    Paragraph::new(name).style(Style::default().fg(fg).bg(bg)),
                    Rect::new(inner.x, inner.y, inner.width, 1),
                );
            }
            if inner.height >= 2 {
                let cost_fg = if is_researched { Color::Green } else { Color::DarkGray };
                f.render_widget(
                    Paragraph::new(cost).style(Style::default().fg(cost_fg)),
                    Rect::new(inner.x, inner.y + 1, inner.width, 1),
                );
            }
        }
    }

    // Info panel
    let (cc, cr) = app.tech_cursor;
    let tech  = &app.techs[cr].1[cc];
    let track = &app.techs[cr].0;
    let state = if app.researched[cr][cc] { "Researched".to_string() }
                else if app.research_progress.map(|(c,r,_)| c==cc&&r==cr).unwrap_or(false) { "In progress".to_string() }
                else if app.can_research(cc, cr) { format!("Available — {} turns", tech.cost) }
                else { "Locked".to_string() };
    let unit_str = tech.unit.map(|u| format!("  Unit: {}  ", u)).unwrap_or_default();

    let info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(format!(" {} ", tech.name), Style::default().fg(track.color()).add_modifier(Modifier::BOLD)),
            Span::styled(format!("[{}]  ", track.label()), Style::default().fg(Color::DarkGray)),
            Span::styled(state, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(Span::styled(
            format!("{}{}  —  {}", unit_str, tech.stats, tech.desc),
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            " ↑↓←→ navigate    Enter research    M back to map    Enter(map) end turn",
            Style::default().fg(Color::DarkGray),
        )),
    ]).block(Block::default().borders(Borders::TOP));
    f.render_widget(info, info_area);
}

fn draw_help(f: &mut Frame, area: Rect) {
    let w = 46u16; let h = 20u16;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let popup = Rect::new(x, y, w, h);
    f.render_widget(Clear, popup);
    let help = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled("  Controls", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(Span::styled("  Map:", Style::default().fg(Color::Cyan))),
        Line::from("    W A S D / arrows — move unit"),
        Line::from("    Enter            — end turn"),
        Line::from("    T                — tech grid"),
        Line::from("    H                — this help"),
        Line::from("    Q                — quit"),
        Line::from(""),
        Line::from(Span::styled("  Tech grid:", Style::default().fg(Color::Cyan))),
        Line::from("    Arrows  — navigate tiles"),
        Line::from("    Enter   — start research"),
        Line::from("    M / Esc — back to map"),
        Line::from(""),
        Line::from(Span::styled("  Map legend:", Style::default().fg(Color::Cyan))),
        Line::from("    ≈ deep sea   ~ shallows  . sand"),
        Line::from("    , plains     f forest    n hills"),
        Line::from("    ^ mountain   * snow      @ you"),
        Line::from(""),
    ]).block(Block::default().borders(Borders::ALL).title(" Help — H to close ")
        .border_style(Style::default().fg(Color::Yellow)));
    f.render_widget(help, popup);
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
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}