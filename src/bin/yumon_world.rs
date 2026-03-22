#![recursion_limit = "256"]

//! src/bin/yumon_world.rs
//!
//! Standalone Yumon World viewer — separate binary from the TUI chat.
//! Run with: `cargo run --bin yumon_world`
//!
//! ── Cargo.toml additions ────────────────────────────────────────────────────
//!
//! [[bin]]
//! name = "yumon_world"
//! path = "src/bin/yumon_world.rs"
//!
//! [dependencies]
//! three-d       = { version = "0.17", features = ["egui"] }
//! three-d-asset = "0.7"
//!
//! ── Controls ────────────────────────────────────────────────────────────────
//!   Left-drag   — orbit camera
//!   Scroll      — zoom
//!   Right-drag  — pan

use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use three_d::{egui::{CollapsingHeader, Color32, RichText, SidePanel}, *};

use yumon_pet::{
    brain::{
        bpe::TokenizerKind,
        model::{GenerationResult, YUMON_SCHEMA, YumonBrain}, samples::{Action, CardinalDir, WorldContext},
    },
    vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES},
};
use three_d::renderer::geometry::Mesh;

// ─── Tunables ────────────────────────────────────────────────────────────────

const YUMON_COUNT: usize = 4;
const SPAWN_RADIUS: f32 = 3.5;
const ARENA: f32 = 6.0;
const WALL_H: f32 = 1.2;
const WALL_T: f32 = 0.35;
const MOVE_SPEED: f32 = 1.8;
const TRAVEL_DIST: f32 = 2.5;
const BUBBLE_TTL: f32 = 7.0;
const ACTION_INTERVAL_SECS: u64 = 30;

/// Fill these in with your Kenney GLB paths.
const MODEL_PATHS: [&str; YUMON_COUNT] = [
    "data/models/animal-fish.glb",
    "data/models/animal-giraffe.glb",
    "data/models/animal-lion.glb",
    "data/models/animal-parrot.glb",
];

const TINTS: [Srgba; YUMON_COUNT] = [
    Srgba { r: 220, g: 110, b: 110, a: 255 },
    Srgba { r: 110, g: 170, b: 220, a: 255 },
    Srgba { r: 130, g: 210, b: 140, a: 255 },
    Srgba { r: 215, g: 185, b:  90, a: 255 },
];

const YUMON_NAMES: [&str; YUMON_COUNT] = ["Ember", "Ripple", "Fern", "Sol"];

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
    log:               Vec<String>,  // recent activity, newest last
}

impl Yumon {
    fn new(id: usize, pos: Vec3) -> Self {
        // Stagger first fires: 0→+5s, 1→+15s, 2→+25s, 3→+35s
        let stagger = Duration::from_secs(5 + id as u64 * 10);
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

        match r.action {
            Action::Speak | Action::Idle => {
                if !r.reply.is_empty() {
                    self.set_speech(r.reply.clone());
                    self.push_log(format!("💬 {}", r.reply));
                } else {
                    self.push_log(format!("({emote})"));
                }
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

    /// Y bob for ambient life
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
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    // ── Channels ─────────────────────────────────────────────────────────────
    let (tx_prompt, rx_prompt) = mpsc::channel::<WorldPrompt>();
    let (tx_result, rx_result) = mpsc::channel::<(usize, GenerationResult)>();

    // ── Brain thread ──────────────────────────────────────────────────────────
    {
        use burn::backend::Wgpu;
        let tx = tx_result;
        std::thread::spawn(move || {
            let device: burn::prelude::Device<Wgpu> = Default::default();
            let (brain, tokenizer) =
                match YumonBrain::<Wgpu>::load("checkpoints/brain-stephen-time", &device) {
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

            let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
            let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

            while let Ok(p) = rx_prompt.recv() {
                let result = brain.generate_structured(
                    &tokenizer, &index, &p.world_ctx,
                    &class_probs, &emote_probs,
                    p.emote_idx, &p.prompt, 110, &device,
                );
                let _ = tx.send((p.yumon_id, result));
            }
        });
    }

    // ── Window (main thread — required on macOS) ──────────────────────────────
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

    // ── Floor ─────────────────────────────────────────────────────────────────
    let fs = ARENA * 2.0;
    let fs64 = fs as f64;
    let mut floor_mesh = CpuMesh::square();
    match &mut floor_mesh.positions {
        Positions::F32(x) => {
            for p in x {
                p.x *= fs * 0.5;
                p.z *= fs * 0.5; // square() is in XY; we rotate it below
            }
        },
        Positions::F64(x) => {
            for p in x {
                p.x *= fs64 * 0.5;
                p.z *= fs64 * 0.5; // square() is in XY; we rotate it below
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

    // ── Render loop ───────────────────────────────────────────────────────────
    window.render_loop(move |mut frame_input| {
        let now = Instant::now();
        let dt  = now.duration_since(last_frame).as_secs_f32().min(0.1);
        last_frame = now;
        let t = frame_input.accumulated_time;

        // Drain brain results
        while let Ok((id, result)) = rx_result.try_recv() {
            if id < YUMON_COUNT { yumons[id].apply_result(result); }
        }

        // Schedule autonomous actions
        for y in yumons.iter_mut() {
            if !y.waiting_for_brain && Instant::now() >= y.next_action_at {
                y.waiting_for_brain = true;
                let _ = tx_prompt.send(WorldPrompt {
                    yumon_id:  y.id,
                    prompt:    String::new(),
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

                        for y in &yumons {
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
                                    // Fading speech bubble
                                    if !y.speech.is_empty() {
                                        let alpha = ((y.speech_timer / BUBBLE_TTL) * 255.0)
                                            .clamp(0.0, 255.0) as u8;
                                        ui.colored_label(
                                            Color32::from_rgba_unmultiplied(230, 225, 170, alpha),
                                            format!("\"{}\"", y.speech),
                                        );
                                        ui.add_space(2.0);
                                    }

                                    // Activity log — newest at top
                                    for entry in y.log.iter().rev().take(5) {
                                        ui.label(
                                            RichText::new(entry)
                                                .size(11.0)
                                                .color(Color32::from_gray(160)),
                                        );
                                    }
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