#![recursion_limit = "256"]

//! src/bin/yumon_world.rs
//!
//! Standalone Yumon World viewer — separate binary from the TUI chat.
//! Run with: `cargo run --bin yumon_world`
//!
//! ── Controls ────────────────────────────────────────────────────────────────
//!   Left-drag   — orbit camera
//!   Scroll      — zoom
//!   Right-drag  — pan

use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

#[cfg(target_os = "windows")]
use three_d::{egui::{CollapsingHeader, Color32, RichText, SidePanel}, *};

use yumon_pet::{
    brain::{
        bpe::TokenizerKind,
        model::{GenerationResult, YUMON_SCHEMA, YumonBrain}, samples::{Action, CardinalDir, WorldContext}, train::MAX_SEQ_LEN,
    },
    vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES},
};

#[cfg(target_os = "windows")]
use three_d::renderer::geometry::Mesh;

// ─── Tunables ────────────────────────────────────────────────────────────────

// const YUMON_COUNT: usize = 4;
// const SPAWN_RADIUS: f32 = 3.5;
// const ARENA: f32 = 6.0;
const YUMON_COUNT: usize = 40;
const SPAWN_RADIUS: f32 = 30.5;
const ARENA: f32 = 60.0;
const WALL_H: f32 = 1.2;
const WALL_T: f32 = 0.35;
const MOVE_SPEED: f32 = 1.8;
const TRAVEL_DIST: f32 = 2.5;
const BUBBLE_TTL: f32 = 7.0;
const ACTION_INTERVAL_SECS: u64 = 10;

/// Fill these in with your Kenney GLB paths.
const MODEL_PATHS: [&str; YUMON_COUNT] = [
    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",

    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",
];

#[cfg(target_os = "windows")]
const TINTS: [Srgba; YUMON_COUNT] = [
    Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },
    
     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },

     Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },
];

const YUMON_NAMES: [&str; YUMON_COUNT] = ["Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol",
                                            "Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol",
                                            "Ember", "Ripple", "Fern", "Sol","Ember", "Ripple", "Fern", "Sol"];

// ─── World objects ────────────────────────────────────────────────────────────

/// A static building at the world centre.
#[cfg(target_os = "windows")]
const BUILDING_POS: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

/// Fixed resource nodes scattered around the arena.
#[cfg(target_os = "windows")]
const RESOURCE_POSITIONS: [Vec3; 3] = [
    Vec3 { x: -4.0, y: 0.0, z: -2.5 },
    Vec3 { x:  4.5, y: 0.0, z:  1.0 },
    Vec3 { x:  0.5, y: 0.0, z:  4.5 },
];

/// Fixed obstacles (boulders / logs) in the arena.
#[cfg(target_os = "windows")]
const OBSTACLE_POSITIONS: [Vec3; 3] = [
    Vec3 { x: -2.0, y: 0.0, z:  2.5 },
    Vec3 { x:  2.5, y: 0.0, z: -3.0 },
    Vec3 { x: -4.5, y: 0.0, z: -4.0 },
];

// ─── Direction helpers ────────────────────────────────────────────────────────

/// Return the cardinal direction from `from` toward `to`, or None if they are
/// essentially at the same spot.
#[cfg(target_os = "windows")]
fn relative_cardinal(from: Vec3, to: Vec3) -> CardinalDir {
    let dx = to.x - from.x;
    let dz = to.z - from.z;
    // "None" zone: object is within ~0.5 units (practically on top of us)
    if dx.abs() < 0.5 && dz.abs() < 0.5 {
        return CardinalDir::None;
    }
    // Pick whichever axis dominates
    if dx.abs() >= dz.abs() {
        if dx > 0.0 { CardinalDir::East } else { CardinalDir::West }
    } else {
        if dz < 0.0 { CardinalDir::North } else { CardinalDir::South }
    }
}

/// Among a slice of positions, return the direction toward the *closest* one.
#[cfg(target_os = "windows")]
fn nearest_cardinal(from: Vec3, positions: &[Vec3]) -> CardinalDir {
    positions
        .iter()
        .min_by(|a, b| {
            let da = ((**a) - from).magnitude2();
            let db = ((**b) - from).magnitude2();
            da.partial_cmp(&db).unwrap()
        })
        .map(|&p| relative_cardinal(from, p))
        .unwrap_or(CardinalDir::None)
}

#[cfg(target_os = "windows")]
fn cardinal_dir_name(d: CardinalDir) -> &'static str {
    match d {
        CardinalDir::North => "north",
        CardinalDir::South => "south",
        CardinalDir::East  => "east",
        CardinalDir::West  => "west",
        CardinalDir::None  => "none",
    }
}

// ─── Brain ↔ world message ────────────────────────────────────────────────────

pub struct WorldPrompt {
    pub yumon_id:  usize,
    pub prompt:    String,
    pub emote_idx: usize,
    pub world_ctx: WorldContext,
}

// ─── Per-Yumon state ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum AnimState { Idle, Walking, Building }

#[cfg(target_os = "windows")]
struct Yumon {
    id:                usize,
    pos:               Vec3,
    target:            Vec3,
    facing:            f32,
    action:            Action,
    anim:              AnimState,
    speech:            String,
    speech_timer:      f32,
    next_action_at:    Instant,
    waiting_for_brain: bool,
    emote_idx:         usize,
    log:               Vec<String>,
    pending_announcement: Option<String>,
}

#[cfg(target_os = "windows")]
impl Yumon {
    fn new(id: usize, pos: Vec3) -> Self {
        let stagger = Duration::from_secs(5 + id as u64);
        Self {
            id,
            pos,
            target: pos,
            facing: (id as f32) * std::f32::consts::FRAC_PI_2,
            action: Action::Idle,
            anim: AnimState::Idle,
            speech: String::new(),
            speech_timer: 0.0,
            next_action_at: Instant::now() + stagger,
            waiting_for_brain: false,
            emote_idx: 0,
            log: Vec::new(),
            pending_announcement: None
        }
    }

    fn clamp_arena(mut p: Vec3) -> Vec3 {
        let lim = ARENA - 0.6;
        p.x = p.x.clamp(-lim, lim);
        p.z = p.z.clamp(-lim, lim);
        p
    }

    fn apply_result(&mut self, r: GenerationResult) {
        self.action    = r.action;
        self.emote_idx = r.yumon_emote_idx;
        self.waiting_for_brain = false;

        let emote = EMOTE_NAMES.get(r.yumon_emote_idx).copied().unwrap_or("?");

        self.push_log(format!("[RAW]: {:?}", r.raw_output));
        self.push_log(format!("[ACTION]: {:?}", r.action));

        if !r.reply.is_empty() {
            self.set_speech(r.reply.clone());
            self.push_log(format!("💬 {}", r.reply));
        } else {
            self.push_log(format!("({emote})"));
        }
        
        match r.action {
            Action::Speak | Action::Idle => {
                self.anim   = AnimState::Idle;
                self.target = self.pos;
            }
            Action::Travel => {
                let dv = cardinal_vec(r.motion_dir);
                self.target = Self::clamp_arena(self.pos + dv * TRAVEL_DIST);
                self.facing = cardinal_angle(r.motion_dir);
                self.anim   = AnimState::Walking;
                self.push_log(format!("🚶 {:?}", r.motion_dir));
            }
            Action::Build => {
                self.anim   = AnimState::Building;
                self.target = self.pos;
                let msg = if r.reply.is_empty() { "🔨 building…".into() }
                          else { format!("🔨 {}", r.reply) };
                self.set_speech(msg.clone());
                self.push_log(msg);
            }
            Action::Use => {
                self.anim   = AnimState::Building;
                self.target = self.pos;
                if !r.reply.is_empty() {
                    self.set_speech(r.reply.clone());
                    self.push_log(format!("⚙️  {}", r.reply));
                }
            }
        }

        self.next_action_at =
            Instant::now() + Duration::from_secs(ACTION_INTERVAL_SECS);
    }

    fn set_speech(&mut self, s: String) {
        self.speech       = s;
        self.speech_timer = BUBBLE_TTL;
    }

    fn push_log(&mut self, s: String) {
        self.log.push(s);
        if self.log.len() > 12 { self.log.remove(0); }
    }

    fn tick(&mut self, dt: f32) {
        let diff = self.target - self.pos;
        let dist = diff.magnitude();
        if dist > 0.02 {
            self.pos += diff.normalize() * (MOVE_SPEED * dt).min(dist);
        } else if self.anim == AnimState::Walking {
            self.anim = AnimState::Idle;
        }

        if self.speech_timer > 0.0 {
            self.speech_timer -= dt;
            if self.speech_timer <= 0.0 { self.speech.clear(); }
        }
    }

    fn bob(&self, t: f64) -> f32 {
        let ph = t as f32 + self.id as f32 * 1.3;
        match self.anim {
            AnimState::Idle     => (ph * 1.1).sin() * 0.04,
            AnimState::Walking  => (ph * 4.2).sin() * 0.07,
            AnimState::Building => (ph * 2.8).sin().abs() * 0.09,
        }
    }

    fn world_transform(&self, t: f64) -> Mat4 {
        let y = 0.45 + self.bob(t);
        let squash = if self.anim == AnimState::Building {
            Mat4::from_nonuniform_scale(1.15, 0.78, 1.15)
        } else {
            Mat4::identity()
        };
        Mat4::from_translation(Vec3::new(self.pos.x, y, self.pos.z))
            * Mat4::from_angle_y(radians(self.facing))
            * squash
            * Mat4::from_scale(0.55)
    }

    /// Build the JSON prompt string that describes this Yumon's surroundings.
    fn build_prompt(&mut self, others: &[Building]) -> String {
        let nearest_built = others.iter()
            .min_by(|a, b| (a.pos - self.pos).magnitude2().partial_cmp(&(b.pos - self.pos).magnitude2()).unwrap())
            .map(|b| relative_cardinal(self.pos, b.pos))
            .unwrap_or(CardinalDir::None);

        let obstacle_dir = nearest_cardinal(self.pos, &OBSTACLE_POSITIONS);
        let building_dir = relative_cardinal(self.pos, BUILDING_POS);
        let resource_dir = nearest_cardinal(self.pos, &RESOURCE_POSITIONS);
        
        // Take the announcement if it exists, otherwise empty string
        let text = self.pending_announcement.take().unwrap_or_default(); // can be empty

        serde_json::to_string_pretty(&serde_json::json!({
            "obstacle_dir": cardinal_dir_name(obstacle_dir),
            "building_dir": cardinal_dir_name(building_dir),
            "resource_dir": cardinal_dir_name(resource_dir),
            "message":      text,
        }))
        .unwrap()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn cardinal_vec(d: CardinalDir) -> Vec3 {
    match d {
        CardinalDir::North => Vec3::new( 0.0, 0.0, -1.0),
        CardinalDir::South => Vec3::new( 0.0, 0.0,  1.0),
        CardinalDir::East  => Vec3::new( 1.0, 0.0,  0.0),
        CardinalDir::West  => Vec3::new(-1.0, 0.0,  0.0),
        CardinalDir::None  => Vec3::zero(),
    }
}

fn cardinal_angle(d: CardinalDir) -> f32 {
    match d {
        CardinalDir::North => 0.0,
        CardinalDir::East  =>  std::f32::consts::FRAC_PI_2,
        CardinalDir::South =>  std::f32::consts::PI,
        CardinalDir::West  => -std::f32::consts::FRAC_PI_2,
        CardinalDir::None  => 0.0,
    }
}

fn action_icon(a: Action) -> &'static str {
    match a {
        Action::Speak  => "💬",
        Action::Build  => "🔨",
        Action::Travel => "🚶",
        Action::Use    => "⚙️ ",
        Action::Idle   => "💤",
    }
}

// ─── GLB loader helper ────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn try_load_glb(context: &Context, path: &str) -> Option<Model<PhysicalMaterial>> {
    let p = std::path::Path::new(path);
    let filename = p.file_name().unwrap().to_str().unwrap();

    let mut loaded = three_d_asset::io::load(&[p])
        .unwrap_or_else(|e| panic!("[world] Failed to load '{path}': {e}"));

    let cpu: CpuModel = loaded.deserialize(filename)
        .unwrap_or_else(|e| panic!("[world] Failed to deserialize '{path}': {e}"));

    let model = Model::<PhysicalMaterial>::new(context, &cpu)
        .unwrap_or_else(|e| panic!("[world] GPU upload failed for '{path}': {e}"));

    println!("[world] Loaded {path}");
    Some(model)
}

#[cfg(target_os = "windows")]
struct Building {
    gm: Gm<Mesh, PhysicalMaterial>,
    pos: Vec3,
    level: u32, // Track how many are stacked
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    #[cfg(target_os = "windows")]
{    
    // ── Channels ─────────────────────────────────────────────────────────────
    let (tx_prompt, rx_prompt) = mpsc::channel::<WorldPrompt>();
    let (tx_result, rx_result) = mpsc::channel::<(usize, GenerationResult)>();

    // ── Brain thread ──────────────────────────────────────────────────────────
    {
        use burn::backend::Wgpu;
        let tx = tx_result;
        std::thread::spawn(move || {
            let device: burn::prelude::Device<Wgpu> = Default::default();
            let (brain, tokenizer, config) =
                match YumonBrain::<Wgpu>::load("checkpoints/brain", &device) {
                    Ok(m) => m,
                    Err(e) => { eprintln!("[brain] load failed: {e}"); return; }
                };

            let bpe = match &tokenizer {
                TokenizerKind::Bpe(b) => b,
                _ => { eprintln!("[brain] BPE tokenizer required"); return; }
            };
            let index = match YumonBrain::<Wgpu>::build_outlines_index(bpe, YUMON_SCHEMA) {
                Ok(i) => i,
                Err(e) => { eprintln!("[brain] index failed: {e}"); return; }
            };

            // let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
            // let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

            while let Ok(p) = rx_prompt.recv() {
                let result = brain.generate_unmasked_parsed(
                    &tokenizer,
                    &p.prompt, config.max_seq_len, &device,
                );
                let _ = tx.send((p.yumon_id, result));
            }
        });
    }

    // ── Window ────────────────────────────────────────────────────────────────
    let window = Window::new(WindowSettings {
        title:    "Yumon World".into(),
        max_size: Some((1440, 900)),
        ..Default::default()
    })
    .expect("Failed to open window");

    let context = window.gl();

    // ── Camera + OrbitControl ─────────────────────────────────────────────────
    let mut camera = Camera::new_perspective(
        window.viewport(),
        Vec3::new(0.0, 13.0, 16.0),
        Vec3::new(0.0,  0.0,  0.0),
        Vec3::unit_y(),
        degrees(42.0),
        0.1,
        200.0,
    );
    let mut orbit = OrbitControl::new(*camera.target(), 2.0, 50.0);

    // ── Lighting ──────────────────────────────────────────────────────────────
    let ambient     = AmbientLight::new(&context, 0.45, Srgba::WHITE);
    let directional = DirectionalLight::new(
        &context, 1.3, Srgba::WHITE, &Vec3::new(-1.0, -2.5, -1.5),
    );

    let mut dynamic_buildings: Vec<Building> = Vec::new();

    // ── Floor ─────────────────────────────────────────────────────────────────
    let fs = ARENA * 2.0;
    let fs64 = fs as f64;
    let mut floor_mesh = CpuMesh::square();
    match &mut floor_mesh.positions {
        Positions::F32(x) => {
            for p in x {
                p.x *= fs * 0.5;
                p.z *= fs * 0.5;
            }
        },
        Positions::F64(x) => {
            for p in x {
                p.x *= fs64 * 0.5;
                p.z *= fs64 * 0.5;
            }
        }
    }
    floor_mesh.compute_normals();
    let mut floor = Gm::new(
        Mesh::new(&context, &floor_mesh),
        PhysicalMaterial::new_opaque(&context, &CpuMaterial {
            albedo: Srgba::new(55, 85, 55, 255),
            roughness: 0.92,
            metallic: 0.0,
            ..Default::default()
        }),
    );
    floor.set_transformation(
        Mat4::from_translation(Vec3::new(0.0, -0.01, 0.0))
            * Mat4::from_angle_x(degrees(-90.0)),
    );

    // ── Walls ─────────────────────────────────────────────────────────────────
    let wall_cpu = CpuMesh::cube();
    let wall_mat = CpuMaterial {
        albedo: Srgba::new(175, 155, 125, 255),
        roughness: 0.85,
        metallic: 0.0,
        ..Default::default()
    };
    let half = ARENA + WALL_T * 0.5;
    let wall_specs: [(Vec3, Vec3); 4] = [
        (Vec3::new( 0.0, WALL_H*0.5, -half), Vec3::new(fs + WALL_T*2.0, WALL_H, WALL_T)),
        (Vec3::new( 0.0, WALL_H*0.5,  half), Vec3::new(fs + WALL_T*2.0, WALL_H, WALL_T)),
        (Vec3::new( half, WALL_H*0.5, 0.0),  Vec3::new(WALL_T, WALL_H, fs)),
        (Vec3::new(-half, WALL_H*0.5, 0.0),  Vec3::new(WALL_T, WALL_H, fs)),
    ];
    let walls: Vec<Gm<Mesh, PhysicalMaterial>> = wall_specs.iter().map(|(pos, s)| {
        let mut gm = Gm::new(
            Mesh::new(&context, &wall_cpu),
            PhysicalMaterial::new_opaque(&context, &wall_mat),
        );
        gm.set_transformation(
            Mat4::from_translation(*pos)
                * Mat4::from_nonuniform_scale(s.x*0.5, s.y*0.5, s.z*0.5),
        );
        gm
    }).collect();

    // ── Central building (flat-topped tower at origin) ─────────────────────────
    let building_cpu = CpuMesh::cube();
    let mut central_building = Gm::new(
        Mesh::new(&context, &building_cpu),
        PhysicalMaterial::new_opaque(&context, &CpuMaterial {
            albedo: Srgba::new(190, 170, 130, 255),
            roughness: 0.70,
            metallic: 0.05,
            ..Default::default()
        }),
    );
    // Scale: 1.2 wide × 1.6 tall × 1.2 deep, centred at origin
    central_building.set_transformation(
        Mat4::from_translation(Vec3::new(0.0, 0.8, 0.0))
            * Mat4::from_nonuniform_scale(0.6, 0.8, 0.6),
    );

    // ── Resource nodes (glowing-ish gold spheres) ─────────────────────────────
    let resource_sphere_cpu = CpuMesh::sphere(14);
    let resource_mat = CpuMaterial {
        albedo:    Srgba::new(220, 190, 60, 255),
        roughness: 0.30,
        metallic:  0.75,
        ..Default::default()
    };
    let resource_nodes: Vec<Gm<Mesh, PhysicalMaterial>> = RESOURCE_POSITIONS.iter().map(|&pos| {
        let mut gm = Gm::new(
            Mesh::new(&context, &resource_sphere_cpu),
            PhysicalMaterial::new_opaque(&context, &resource_mat),
        );
        gm.set_transformation(
            Mat4::from_translation(Vec3::new(pos.x, 0.30, pos.z))
                * Mat4::from_scale(0.28),
        );
        gm
    }).collect();

    // ── Obstacles (rough dark boulders) ──────────────────────────────────────
    let obstacle_cpu = CpuMesh::sphere(8); // low-poly → boulder feel
    let obstacle_mat = CpuMaterial {
        albedo:    Srgba::new(90, 80, 75, 255),
        roughness: 0.95,
        metallic:  0.0,
        ..Default::default()
    };
    let obstacles: Vec<Gm<Mesh, PhysicalMaterial>> = OBSTACLE_POSITIONS.iter().enumerate().map(|(i, &pos)| {
        // Vary sizes slightly so they don't look identical
        let scale = 0.30 + (i as f32) * 0.06;
        let mut gm = Gm::new(
            Mesh::new(&context, &obstacle_cpu),
            PhysicalMaterial::new_opaque(&context, &obstacle_mat),
        );
        gm.set_transformation(
            Mat4::from_translation(Vec3::new(pos.x, scale * 0.5, pos.z))
                * Mat4::from_nonuniform_scale(scale, scale * 0.75, scale),
        );
        gm
    }).collect();

    // ── Per-Yumon GLB models + sphere fallbacks ───────────────────────────────
    let mut gpu_models: Vec<Option<Model<PhysicalMaterial>>> = MODEL_PATHS.iter()
        .map(|p| try_load_glb(&context, p))
        .collect();

    let sphere_cpu = CpuMesh::sphere(20);
    let mut fallback_spheres: Vec<Gm<Mesh, PhysicalMaterial>> = TINTS.iter().map(|&tint| {
        Gm::new(
            Mesh::new(&context, &sphere_cpu),
            PhysicalMaterial::new_opaque(&context, &CpuMaterial {
                albedo: tint,
                roughness: 0.55,
                metallic: 0.1,
                ..Default::default()
            }),
        )
    }).collect();

    // ── Yumon state ───────────────────────────────────────────────────────────
    let mut yumons: Vec<Yumon> = (0..YUMON_COUNT).map(|i| {
        let angle = (i as f32 / YUMON_COUNT as f32) * std::f32::consts::TAU;
        Yumon::new(i, Vec3::new(angle.cos() * SPAWN_RADIUS, 0.0, angle.sin() * SPAWN_RADIUS))
    }).collect();

    // ── egui ──────────────────────────────────────────────────────────────────
    let mut gui = GUI::new(&context);

    // ── Frame timing ──────────────────────────────────────────────────────────
    let mut last_frame = Instant::now();

    let mut ui_inputs = vec![String::new(); YUMON_COUNT];

    // ── Render loop ───────────────────────────────────────────────────────────
    window.render_loop(move |mut frame_input| {
        let now = Instant::now();
        let dt  = now.duration_since(last_frame).as_secs_f32().min(0.1);
        last_frame = now;
        let t = frame_input.accumulated_time;

        // Drain brain results
        while let Ok((id, result)) = rx_result.try_recv() {
            if id < YUMON_COUNT { yumons[id].apply_result(result.clone()); }

            // Inside the while let Ok((id, result)) = rx_result.try_recv() loop:
            if result.action == Action::Build {
                let y = &yumons[id];
                let build_pos = y.pos; // Build at Yumon's current feet

                // Check if we are building "on top" of an existing one nearby
                let existing = dynamic_buildings.iter_mut()
                    .find(|b| (b.pos - build_pos).magnitude() < 0.8);

                if let Some(b) = existing {
                    // Stack: Increase level and move the mesh up
                    b.level += 1;
                    let new_y = 0.8 + (b.level as f32 * 0.4); // 0.8 is base height
                    b.gm.set_transformation(Mat4::from_translation(Vec3::new(b.pos.x, new_y, b.pos.z)) 
                        * Mat4::from_nonuniform_scale(0.6, 0.2, 0.6)); // Make it a "slab"
                } else {
                    let mat = CpuMaterial {
                        albedo: Srgba::new(190, 70, 30, 255),
                        roughness: 0.70,
                        metallic: 0.05,
                        ..Default::default()
                    };
                    // New building: Create a fresh Gm
                    let mut gm = Gm::new(
                        Mesh::new(&context, &building_cpu),
                        PhysicalMaterial::new_opaque(&context, &mat),
                    );
                    gm.set_transformation(Mat4::from_translation(Vec3::new(build_pos.x, 0.4, build_pos.z)) 
                        * Mat4::from_nonuniform_scale(0.5, 0.4, 0.5));
                    
                    dynamic_buildings.push(Building { gm, pos: build_pos, level: 0 });
                }
            }
        }

        // Schedule autonomous actions
        for y in yumons.iter_mut() {
            if !y.waiting_for_brain && Instant::now() >= y.next_action_at {
                y.waiting_for_brain = true;
                let prompt = y.build_prompt(&dynamic_buildings);
                let _ = tx_prompt.send(WorldPrompt {
                    yumon_id:  y.id,
                    prompt,
                    emote_idx: y.emote_idx,
                    world_ctx: WorldContext::default(),
                });
            }
        }

        // Tick world state
        for y in yumons.iter_mut() { y.tick(dt); }

        // Camera
        camera.set_viewport(frame_input.viewport);
        orbit.handle_events(&mut camera, &mut frame_input.events);

        // Update fallback sphere transforms
        for (i, y) in yumons.iter().enumerate() {
            if gpu_models[i].is_none() {
                fallback_spheres[i].set_transformation(y.world_transform(t));
            }
        }

        

        // egui panel
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |ctx| {
                SidePanel::right("yumon_panel")
                    .min_width(240.0)
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.add_space(6.0);
                        ui.heading("🌿 Yumon World");
                        ui.separator();

                        ui.add_space(4.0);

                        for y in &mut yumons {
                            let waiting_str = if y.waiting_for_brain { "  ⏳" } else { "" };
                            let emote = EMOTE_NAMES.get(y.emote_idx).copied().unwrap_or("");
                            let header = format!(
                                "{} {}  {}{}",
                                action_icon(y.action),
                                YUMON_NAMES[y.id],
                                emote,
                                waiting_str,
                            );

                            CollapsingHeader::new(header)
                                .default_open(true)
                                .show(ui, |ui| {
                                    if !y.speech.is_empty() {
                                        let alpha = ((y.speech_timer / BUBBLE_TTL) * 255.0)
                                            .clamp(0.0, 255.0) as u8;
                                        ui.colored_label(
                                            Color32::from_rgba_unmultiplied(230, 225, 170, alpha),
                                            format!("\"{}\"", y.speech),
                                        );
                                        ui.add_space(2.0);
                                    }

                                    for entry in y.log.iter().rev().take(5) {
                                        ui.label(
                                            RichText::new(entry)
                                                .size(11.0)
                                                .color(Color32::from_gray(160)),
                                        );
                                    }

                                    ui.horizontal(|ui| {
                                        ui.add(egui::TextEdit::singleline(&mut ui_inputs[y.id])
                                            .hint_text("Announcement..."));
                                        
                                        if ui.button("Send").clicked() {
                                            // Find the actual yumon in the vec and set the message
                                            y.pending_announcement = Some(ui_inputs[y.id].clone());
                                            ui_inputs[y.id].clear();
                                            
                                            // Optional: Force the brain to trigger immediately
                                            y.next_action_at = std::time::Instant::now();
                                        }
                                    });
                                });

                            ui.add_space(6.0);
                        }

                        ui.with_layout(
                            egui::Layout::bottom_up(egui::Align::LEFT),
                            |ui| {
                                ui.separator();
                                ui.label(
                                    RichText::new("drag: orbit  •  scroll: zoom  •  right-drag: pan")
                                        .size(10.0)
                                        .color(Color32::from_gray(100)),
                                );
                            },
                        );
                    });
            },
        );

        // ── Draw ─────────────────────────────────────────────────────────────
        let lights: [&dyn Light; 2] = [&ambient, &directional];
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.10, 0.12, 0.15, 1.0, 1.0));

        // Floor + walls
        screen.render(&camera, [&floor as &dyn Object], &lights);
        for w in &walls {
            screen.render(&camera, [w as &dyn Object], &lights);
        }

        // Central building
        screen.render(&camera, [&central_building as &dyn Object], &lights);

        for b in &dynamic_buildings {
            screen.render(&camera, [&b.gm as &dyn Object], &lights);
        }

        // Resource nodes
        for r in &resource_nodes {
            screen.render(&camera, [r as &dyn Object], &lights);
        }

        // Obstacles
        for o in &obstacles {
            screen.render(&camera, [o as &dyn Object], &lights);
        }

        // Yumon bodies
        for (i, y) in yumons.iter().enumerate() {
            let xform = y.world_transform(t);
            if let Some(ref mut model) = gpu_models[i] {
                for primitive in model.iter_mut() {
                    let original = primitive.transformation();
                    primitive.set_transformation(xform * original);
                    screen.render(&camera, [primitive as &dyn Object], &lights);
                    primitive.set_transformation(original);
                }
            } else {
                screen.render(&camera, [&fallback_spheres[i] as &dyn Object], &lights);
            }
        }

        // egui on top
        screen.write(|| gui.render()).unwrap();

        FrameOutput::default()
    });
}
}