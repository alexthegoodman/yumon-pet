#![recursion_limit = "256"]

//! src/bin/yumon_world.rs
//!
//! Yumon World — single Yumon + player, command-driven actions.
//! Run with: `cargo run --bin yumon_world`
//!
//! ── Controls ────────────────────────────────────────────────────────────────
//!   Left-drag          — orbit camera
//!   Scroll             — zoom
//!   Right-drag         — pan
//!   Left-click (ground)— move player OR set destination (see panel toggle)

use std::{
    f32::consts::{PI, TAU},
    sync::mpsc,
    time::{Duration, Instant},
};

#[cfg(target_os = "windows")]
use three_d::{
    egui::{CollapsingHeader, Color32, RichText, SidePanel},
    *,
};

use three_d_asset::PixelPoint;

use yumon_pet::brain::{
    bpe::TokenizerKind,
    model::{GenerationResult, YumonBrain},
    samples::{Action, CardinalDir},
    train::MAX_SEQ_LEN,
};

#[cfg(target_os = "windows")]
use three_d::renderer::geometry::Mesh;

// ─── Tunables ────────────────────────────────────────────────────────────────

const ARENA: f32          = 20.0;
const WALL_H: f32         = 1.2;
const WALL_T: f32         = 0.35;
const MOVE_SPEED: f32     = 3.0;
const BUBBLE_TTL: f32     = 7.0;
const ACTION_INTERVAL_SECS: f32 = 10.0;

/// How close to a target counts as "arrived".
const ARRIVE_THRESH: f32  = 0.6;

/// Home position — back-left corner.
const HOME_POS: Vec3      = Vec3 { x: -ARENA + 2.0, y: 0.0, z: -ARENA + 2.0 };

/// Initial destination marker position.
const DEST_INIT: Vec3     = Vec3 { x: 5.0, y: 0.0, z: 5.0 };

/// Player start position.
const PLAYER_INIT: Vec3   = Vec3 { x: 3.0, y: 0.0, z: -3.0 };

/// Player movement speed (click-to-move).
const PLAYER_SPEED: f32   = 4.5;

const MODEL_PATH: &str    = "data/models/animal-parrot.glb";

// ─── Click mode ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum ClickMode { Camera, MovePlayer, SetDestination }

// ─── Anim state ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum AnimState { Idle, Walking, Building }

// ─── GetHelp orbit state ──────────────────────────────────────────────────────

/// Tracks progress around the arena perimeter for the GetHelp action.
struct OrbitState {
    /// Current angle around the arena (radians).
    angle: f32,
}

impl OrbitState {
    fn new(start_pos: Vec3) -> Self {
        // Initialise from the Yumon's current position so it doesn't snap.
        let angle = start_pos.z.atan2(start_pos.x);
        Self { angle }
    }

    /// Advance one step and return the next waypoint.
    fn next_waypoint(&mut self) -> Vec3 {
        // Advance ~45° each interval so a full orbit takes ~8 intervals.
        self.angle = (self.angle + TAU / 8.0) % TAU;
        let r = ARENA - 1.5;
        Vec3::new(r * self.angle.cos(), 0.0, r * self.angle.sin())
    }
}

// ─── Brain ↔ world message ────────────────────────────────────────────────────

pub struct WorldPrompt {
    pub prompt: String,
}

// ─── Player ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
struct Player {
    pos:    Vec3,
    target: Vec3,
}

#[cfg(target_os = "windows")]
impl Player {
    fn new() -> Self {
        Self { pos: PLAYER_INIT, target: PLAYER_INIT }
    }

    fn tick(&mut self, dt: f32) {
        let diff = self.target - self.pos;
        let dist = diff.magnitude();
        if dist > 0.02 {
            self.pos += diff.normalize() * (PLAYER_SPEED * dt).min(dist);
        }
    }

    fn clamp_arena(mut p: Vec3) -> Vec3 {
        let lim = ARENA - 0.6;
        p.x = p.x.clamp(-lim, lim);
        p.z = p.z.clamp(-lim, lim);
        p
    }
}

// ─── Yumon ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
struct Yumon {
    pos:               Vec3,
    target:            Vec3,
    facing:            f32,
    action:            Action,
    anim:              AnimState,
    speech:            String,
    speech_timer:      f32,
    /// Counts down to next brain call (reset after each result).
    interval_timer:    f32,
    waiting_for_brain: bool,
    parsed_emotion:    String,
    log:               Vec<String>,
    /// Persists across intervals until changed.
    current_command:   String,
    /// One-shot; cleared after each brain call.
    pending_message:   String,
    /// Orbit state, created when action = GetHelp.
    orbit:             Option<OrbitState>,
    /// Survey / Sit timer (seconds remaining for timed-idle actions).
    idle_timer:        f32,
}

#[cfg(target_os = "windows")]
impl Yumon {
    fn new() -> Self {
        Self {
            pos:               Vec3::new(0.0, 0.0, 0.0),
            target:            Vec3::new(0.0, 0.0, 0.0),
            facing:            0.0,
            action:            Action::Sit,
            anim:              AnimState::Idle,
            speech:            String::new(),
            speech_timer:      0.0,
            interval_timer:    ACTION_INTERVAL_SECS,
            waiting_for_brain: false,
            parsed_emotion:    "neutral".to_string(),
            log:               Vec::new(),
            current_command:   String::new(),
            pending_message:   String::new(),
            orbit:             None,
            idle_timer:        0.0,
        }
    }

    fn clamp_arena(mut p: Vec3) -> Vec3 {
        let lim = ARENA - 0.6;
        p.x = p.x.clamp(-lim, lim);
        p.z = p.z.clamp(-lim, lim);
        p
    }

    fn apply_result(&mut self, r: GenerationResult, player_pos: Vec3, dest_pos: Vec3) {
        self.action         = r.action;
        self.parsed_emotion = r.parsed_emotion.clone();
        self.waiting_for_brain = false;

        self.push_log(format!("[ACTION]: {:?}", r.action));

        if !r.reply.is_empty() {
            self.set_speech(r.reply.clone());
            self.push_log(format!("💬 {}", r.reply));
        }

        // Clear the one-shot message now that we've consumed it.
        self.pending_message.clear();

        // Apply movement / anim based on new action.
        self.orbit     = None;
        self.idle_timer = 0.0;

        match r.action {
            Action::GoToDestination => {
                self.set_walk_toward(dest_pos);
            }
            Action::GoHome => {
                self.set_walk_toward(HOME_POS);
            }
            Action::Follow | Action::GetHelp => {
                // Follow: re-targeted each tick in tick().
                // GetHelp: orbit — set up orbit state and pick first waypoint.
                if r.action == Action::GetHelp {
                    let mut orbit = OrbitState::new(self.pos);
                    let wp = orbit.next_waypoint();
                    self.orbit = Some(orbit);
                    self.set_walk_toward(wp);
                } else {
                    self.set_walk_toward(player_pos);
                }
            }
            Action::Survey => {
                self.anim       = AnimState::Idle;
                self.target     = self.pos;
                self.idle_timer = 6.0; // slow rotation for 6 s
            }
            Action::Collect => {
                // Walk toward nearest resource node.
                let nearest = RESOURCE_POSITIONS
                    .iter()
                    .min_by(|a, b| {
                        ((**a) - self.pos).magnitude2()
                            .partial_cmp(&((**b) - self.pos).magnitude2())
                            .unwrap()
                    })
                    .copied()
                    .unwrap_or(Vec3::zero());
                self.set_walk_toward(nearest);
            }
            Action::Stack => {
                // Walk to central building, then building anim.
                self.set_walk_toward(BUILDING_POS);
                self.anim = AnimState::Building;
            }
            Action::Sit => {
                self.anim       = AnimState::Idle;
                self.target     = self.pos;
                self.idle_timer = 5.0;
            }
        }

        // Reset interval timer.
        self.interval_timer = ACTION_INTERVAL_SECS;
    }

    fn set_walk_toward(&mut self, dest: Vec3) {
        self.target = Self::clamp_arena(dest);
        let diff    = self.target - self.pos;
        if diff.magnitude() > 0.1 {
            self.facing = diff.z.atan2(diff.x) - PI * 0.5;
        }
        self.anim = AnimState::Walking;
    }

    fn set_speech(&mut self, s: String) {
        self.speech       = s;
        self.speech_timer = BUBBLE_TTL;
    }

    fn push_log(&mut self, s: String) {
        self.log.push(s);
        if self.log.len() > 12 { self.log.remove(0); }
    }

    fn tick(&mut self, dt: f32, player_pos: Vec3) {
        // Walk toward target.
        let diff = self.target - self.pos;
        let dist = diff.magnitude();
        if dist > 0.02 {
            self.pos += diff.normalize() * (MOVE_SPEED * dt).min(dist);
        } else if self.anim == AnimState::Walking {
            self.anim = AnimState::Idle;
        }

        // Follow: continuously re-target player.
        if self.action == Action::Follow {
            let offset = (self.pos - player_pos).normalize() * 1.2;
            self.set_walk_toward(player_pos + offset);
        }

        // Survey: slow rotation in place.
        if self.action == Action::Survey && self.idle_timer > 0.0 {
            self.facing += dt * 0.6;
            self.idle_timer -= dt;
        }

        // Sit idle timer.
        if self.action == Action::Sit && self.idle_timer > 0.0 {
            self.idle_timer -= dt;
        }

        // Speech bubble fade.
        if self.speech_timer > 0.0 {
            self.speech_timer -= dt;
            if self.speech_timer <= 0.0 { self.speech.clear(); }
        }
    }

    fn bob(&self, t: f64) -> f32 {
        let ph = t as f32;
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

    /// Build the JSON prompt string for the brain.
    fn build_prompt(&self) -> String {
        let memories: Vec<serde_json::Value> = Vec::new(); // extend later
        serde_json::to_string_pretty(&serde_json::json!({
            "command": self.current_command,
            "message": self.pending_message,
            "memories": memories,
        }))
        .unwrap()
    }
}

// ─── World constants ──────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
const BUILDING_POS: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

#[cfg(target_os = "windows")]
const RESOURCE_POSITIONS: [Vec3; 3] = [
    Vec3 { x: -4.0, y: 0.0, z: -2.5 },
    Vec3 { x:  4.5, y: 0.0, z:  1.0 },
    Vec3 { x:  0.5, y: 0.0, z:  4.5 },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn action_icon(a: Action) -> &'static str {
    match a {
        Action::GoToDestination => "📍",
        Action::GoHome          => "🏠",
        Action::Follow          => "🐾",
        Action::GetHelp         => "🆘",
        Action::Survey          => "🔭",
        Action::Collect         => "🌿",
        Action::Stack           => "🔨",
        Action::Sit             => "💤",
    }
}

/// Raycast a mouse position against the ground plane (y = 0).
/// Returns None if the ray points away from the ground.
// #[cfg(target_os = "windows")]
// fn ray_ground_intersect(camera: &Camera, screen_pos: (f64, f64), viewport: Viewport) -> Option<Vec3> {
//     let ray = camera.pixel_ray(Vec2::new(screen_pos.0 as f32, screen_pos.1 as f32));
//     let denom = ray.direction.y;
//     if denom.abs() < 1e-5 { return None; }
//     let t = -ray.origin.y / denom;
//     if t < 0.0 { return None; }
//     let p = ray.origin + ray.direction * t;
//     let lim = ARENA - 0.6;
//     Some(Vec3::new(p.x.clamp(-lim, lim), 0.0, p.z.clamp(-lim, lim)))
// }

#[cfg(target_os = "windows")]
fn ray_ground_intersect(camera: &Camera, pixel: PhysicalPoint) -> Option<Vec3> {
    // Camera position (ray origin)
    let origin = camera.position();
    // View direction through this pixel
    let direction = camera.view_direction_at_pixel(pixel);

    // Intersect with y = 0 plane
    let denom = direction.y;
    if denom.abs() < 1e-5 { return None; }
    let t = -origin.y / denom;
    if t < 0.0 { return None; }
    let p = origin + direction * t;
    let lim = ARENA - 0.6;
    Some(Vec3::new(p.x.clamp(-lim, lim), 0.0, p.z.clamp(-lim, lim)))
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    #[cfg(target_os = "windows")]
    {
        // ── Channels ──────────────────────────────────────────────────────────
        let (tx_prompt, rx_prompt) = mpsc::channel::<WorldPrompt>();
        let (tx_result, rx_result) = mpsc::channel::<GenerationResult>();

        // ── Brain thread ──────────────────────────────────────────────────────
        {
            let tx = tx_result;
            std::thread::spawn(move || {
                use burn::backend::Wgpu;
                let device: burn::prelude::Device<Wgpu> = Default::default();
                let (brain, tokenizer, config) =
                    match YumonBrain::<Wgpu>::load(
                        "checkpoints/brain/128h_2l_2a_220len_6e", &device,
                    ) {
                        Ok(m) => m,
                        Err(e) => { eprintln!("[brain] load failed: {e}"); return; }
                    };

                while let Ok(p) = rx_prompt.recv() {
                    let result = brain.generate_unmasked_parsed(
                        &tokenizer, &p.prompt, config.max_seq_len, &device,
                    );

                    println!("parsed result {:?}", result);

                    let _ = tx.send(result);
                }
            });
        }

        // ── Window ────────────────────────────────────────────────────────────
        let window = Window::new(WindowSettings {
            title:    "Yumon World".into(),
            max_size: Some((1440, 900)),
            ..Default::default()
        })
        .expect("Failed to open window");

        let context = window.gl();

        // ── Camera ────────────────────────────────────────────────────────────
        let mut camera = Camera::new_perspective(
            window.viewport(),
            Vec3::new(0.0, 18.0, 20.0),
            Vec3::new(0.0,  0.0,  0.0),
            Vec3::unit_y(),
            degrees(42.0),
            0.1,
            200.0,
        );
        let mut orbit_ctrl = OrbitControl::new(*camera.target(), 2.0, 60.0);

        // ── Lighting ──────────────────────────────────────────────────────────
        let ambient     = AmbientLight::new(&context, 0.45, Srgba::WHITE);
        let directional = DirectionalLight::new(
            &context, 1.3, Srgba::WHITE, &Vec3::new(-1.0, -2.5, -1.5),
        );

        // ── Floor ─────────────────────────────────────────────────────────────
        let fs = ARENA * 2.0;
        let mut floor_mesh = CpuMesh::square();
        if let Positions::F32(ref mut v) = floor_mesh.positions {
            for p in v.iter_mut() { p.x *= fs * 0.5; p.z *= fs * 0.5; }
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

        // ── Walls ─────────────────────────────────────────────────────────────
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

        // ── Central building ──────────────────────────────────────────────────
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
        central_building.set_transformation(
            Mat4::from_translation(Vec3::new(0.0, 0.8, 0.0))
                * Mat4::from_nonuniform_scale(0.6, 0.8, 0.6),
        );

        // ── Resource nodes ────────────────────────────────────────────────────
        let resource_sphere_cpu = CpuMesh::sphere(14);
        let resource_mat = CpuMaterial {
            albedo:    Srgba::new(220, 190, 60, 255),
            roughness: 0.30,
            metallic:  0.75,
            ..Default::default()
        };
        let resource_nodes: Vec<Gm<Mesh, PhysicalMaterial>> =
            RESOURCE_POSITIONS.iter().map(|&pos| {
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

        // ── Home marker (blue sphere) ─────────────────────────────────────────
        let mut home_marker = Gm::new(
            Mesh::new(&context, &CpuMesh::sphere(14)),
            PhysicalMaterial::new_opaque(&context, &CpuMaterial {
                albedo:    Srgba::new(80, 140, 220, 255),
                roughness: 0.40,
                metallic:  0.5,
                ..Default::default()
            }),
        );
        home_marker.set_transformation(
            Mat4::from_translation(Vec3::new(HOME_POS.x, 0.35, HOME_POS.z))
                * Mat4::from_scale(0.30),
        );

        // ── Destination marker (orange flat disc) ─────────────────────────────
        let mut dest_pos    = DEST_INIT;
        let disc_cpu = CpuMesh::cylinder(16);
        let mut dest_marker = Gm::new(
            Mesh::new(&context, &disc_cpu),
            PhysicalMaterial::new_opaque(&context, &CpuMaterial {
                albedo:    Srgba::new(230, 120, 40, 255),
                roughness: 0.60,
                metallic:  0.1,
                ..Default::default()
            }),
        );
        // Flat disc: scale y very small, radius ~0.4
        dest_marker.set_transformation(
            Mat4::from_translation(Vec3::new(dest_pos.x, 0.03, dest_pos.z))
                * Mat4::from_nonuniform_scale(0.4, 0.04, 0.4),
        );

        // ── Player cylinder (white/grey) ──────────────────────────────────────
        let mut player = Player::new();
        let cyl_cpu = CpuMesh::cylinder(20);
        let mut player_gm = Gm::new(
            Mesh::new(&context, &cyl_cpu),
            PhysicalMaterial::new_opaque(&context, &CpuMaterial {
                albedo:    Srgba::new(220, 220, 210, 255),
                roughness: 0.50,
                metallic:  0.15,
                ..Default::default()
            }),
        );

        // ── Yumon GLB + fallback ──────────────────────────────────────────────
        let mut yumon         = Yumon::new();
        let gpu_model: Option<Model<PhysicalMaterial>> = {
            let p = std::path::Path::new(MODEL_PATH);
            let filename = p.file_name().unwrap().to_str().unwrap();
            three_d_asset::io::load(&[p]).ok().and_then(|mut loaded| {
                loaded.deserialize(filename).ok().and_then(|cpu: CpuModel| {
                    Model::<PhysicalMaterial>::new(&context, &cpu).ok()
                })
            })
        };
        let mut gpu_model = gpu_model;

        let mut fallback_sphere = Gm::new(
            Mesh::new(&context, &CpuMesh::sphere(20)),
            PhysicalMaterial::new_opaque(&context, &CpuMaterial {
                albedo:    Srgba::new(180, 120, 200, 255),
                roughness: 0.55,
                metallic:  0.1,
                ..Default::default()
            }),
        );

        // ── UI state ──────────────────────────────────────────────────────────
        let mut gui           = GUI::new(&context);
        let mut click_mode    = ClickMode::Camera;
        let mut ui_command    = String::new(); // editing buffer for command field
        let mut ui_message    = String::new(); // editing buffer for message field
        let mut last_frame    = Instant::now();

        // ── Render loop ───────────────────────────────────────────────────────
        window.render_loop(move |mut frame_input| {
            let now = Instant::now();
            let dt  = now.duration_since(last_frame).as_secs_f32().min(0.1);
            last_frame = now;
            let t = frame_input.accumulated_time;

            // ── Brain results ─────────────────────────────────────────────────
            if let Ok(result) = rx_result.try_recv() {
                yumon.apply_result(result, player.pos, dest_pos);
            }

            // ── Interval timer → fire brain ───────────────────────────────────
            if !yumon.waiting_for_brain {
                yumon.interval_timer -= dt;
                if yumon.interval_timer <= 0.0 {
                    yumon.waiting_for_brain = true;
                    let prompt = yumon.build_prompt();
                    let _ = tx_prompt.send(WorldPrompt { prompt });
                }
            }

            // ── Tick ──────────────────────────────────────────────────────────
            player.tick(dt);
            yumon.tick(dt, player.pos);

            // ── Handle orbit waypoint advancement ─────────────────────────────
            // When GetHelp and the Yumon arrives at its orbit waypoint, pick next.
            if yumon.action == Action::GetHelp {
                let diff = yumon.target - yumon.pos;
                if diff.magnitude() < ARRIVE_THRESH {
                    if let Some(ref mut orb) = yumon.orbit {
                        let wp = orb.next_waypoint();
                        yumon.set_walk_toward(wp);
                    }
                }
            }

            // ── egui panel ────────────────────────────────────────────────────
            let mut gui_consumed = false;
            gui.update(
                &mut frame_input.events,
                frame_input.accumulated_time,
                frame_input.viewport,
                frame_input.device_pixel_ratio,
                |ctx| {
                    gui_consumed = ctx.wants_pointer_input();

                    SidePanel::right("yumon_panel")
                        .min_width(260.0)
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.add_space(6.0);
                            ui.heading("🌿 Yumon World");
                            ui.separator();

                            // ── Click mode toggle ──────────────────────────────
                            ui.add_space(4.0);
                            ui.label(RichText::new("Click mode").size(11.0).color(Color32::from_gray(140)));
                            ui.horizontal(|ui| {
                                ui.radio_value(
                                    &mut click_mode,
                                    ClickMode::Camera,
                                    "🎥 Camera",
                                );
                                ui.radio_value(
                                    &mut click_mode,
                                    ClickMode::MovePlayer,
                                    "🚶 Move player",
                                );
                                ui.radio_value(
                                    &mut click_mode,
                                    ClickMode::SetDestination,
                                    "📍 Set destination",
                                );
                            });

                            ui.separator();

                            // ── Yumon panel ────────────────────────────────────
                            let waiting_str = if yumon.waiting_for_brain { "  ⏳" } else { "" };
                            let timer_str   = format!("  {:.1}s", yumon.interval_timer.max(0.0));
                            let header = format!(
                                "{} Yumon  {}{}",
                                action_icon(yumon.action),
                                waiting_str,
                                timer_str,
                            );

                            CollapsingHeader::new(header)
                                .default_open(true)
                                .show(ui, |ui| {
                                    // Speech bubble
                                    if !yumon.speech.is_empty() {
                                        let alpha = ((yumon.speech_timer / BUBBLE_TTL) * 255.0)
                                            .clamp(0.0, 255.0) as u8;
                                        ui.colored_label(
                                            Color32::from_rgba_unmultiplied(230, 225, 170, alpha),
                                            format!("\"{}\"", yumon.speech),
                                        );
                                        ui.add_space(2.0);
                                    }

                                    // Log
                                    for entry in yumon.log.iter().rev().take(5) {
                                        ui.label(
                                            RichText::new(entry)
                                                .size(11.0)
                                                .color(Color32::from_gray(160)),
                                        );
                                    }

                                    ui.add_space(4.0);
                                    ui.separator();

                                    // Command field (persists)
                                    ui.label(
                                        RichText::new("Command (persists)")
                                            .size(11.0)
                                            .color(Color32::from_gray(140)),
                                    );
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut ui_command)
                                                .hint_text("e.g. follow me"),
                                        );
                                        if ui.button("Set").clicked() && !ui_command.is_empty() {
                                            yumon.current_command = ui_command.clone();
                                            yumon.push_log(
                                                format!("[CMD] {}", yumon.current_command),
                                            );
                                            // Trigger brain immediately on new command.
                                            yumon.interval_timer    = 0.0;
                                        }
                                    });
                                    if !yumon.current_command.is_empty() {
                                        ui.label(
                                            RichText::new(
                                                format!("▶ {}", yumon.current_command),
                                            )
                                            .size(11.0)
                                            .color(Color32::from_rgb(130, 200, 130)),
                                        );
                                    }

                                    ui.add_space(4.0);

                                    // Message field (one-shot)
                                    ui.label(
                                        RichText::new("Message (one-shot)")
                                            .size(11.0)
                                            .color(Color32::from_gray(140)),
                                    );
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut ui_message)
                                                .hint_text("e.g. hurry up!"),
                                        );
                                        if ui.button("Send").clicked() && !ui_message.is_empty() {
                                            yumon.pending_message = ui_message.clone();
                                            ui_message.clear();
                                            // Trigger brain immediately for one-shot message.
                                            yumon.interval_timer = 0.0;
                                        }
                                    });
                                });

                            ui.with_layout(
                                egui::Layout::bottom_up(egui::Align::LEFT),
                                |ui| {
                                    ui.separator();
                                    ui.label(
                                        RichText::new(
                                            "drag: orbit  •  scroll: zoom  •  right-drag: pan",
                                        )
                                        .size(10.0)
                                        .color(Color32::from_gray(100)),
                                    );
                                },
                            );
                        });
                },
            );

            // ── Camera ────────────────────────────────────────────────────────
            camera.set_viewport(frame_input.viewport);

            if !gui_consumed {
                // Separate orbit events from click events so we can do ground pick.
                let mut left_click: Option<PixelPoint> = None;
                for event in &frame_input.events {
                    match event {
                        Event::MousePress {
                            button: MouseButton::Left,
                            position,
                            ..
                        } => {
                            left_click = Some(*position); // PhysicalPoint directly
                        }
                        _ => {}
                    }
                }
                orbit_ctrl.handle_events(&mut camera, &mut frame_input.events);

                // Ground click — after orbit handled (orbit consumes drag, not click).
                if let Some(pixel) = left_click {
                    if let Some(world_pt) = ray_ground_intersect(&camera, pixel) {
                        match click_mode {
                            ClickMode::Camera => {} // No-op: purposeful decision
                            ClickMode::MovePlayer => {
                                player.target = Player::clamp_arena(world_pt);
                            }
                            ClickMode::SetDestination => {
                                dest_pos = world_pt;
                                dest_marker.set_transformation(
                                    Mat4::from_translation(Vec3::new(
                                        dest_pos.x, 0.03, dest_pos.z,
                                    )) * Mat4::from_nonuniform_scale(0.4, 0.04, 0.4),
                                );
                            }
                        }
                    }
                }
            }

            // ── Update transforms ─────────────────────────────────────────────
            // Player cylinder: y scale = 0.5 height, centred at y=0.5
            player_gm.set_transformation(
                Mat4::from_translation(Vec3::new(player.pos.x, 0.5, player.pos.z))
                    * Mat4::from_nonuniform_scale(0.25, 0.5, 0.25),
            );

            let yumon_xform = yumon.world_transform(t);
            if gpu_model.is_none() {
                fallback_sphere.set_transformation(yumon_xform);
            }

            // ── Draw ──────────────────────────────────────────────────────────
            let lights: [&dyn Light; 2] = [&ambient, &directional];
            let screen = frame_input.screen();
            screen.clear(ClearState::color_and_depth(0.10, 0.12, 0.15, 1.0, 1.0));

            screen.render(&camera, [&floor as &dyn Object], &lights);
            for w in &walls {
                screen.render(&camera, [w as &dyn Object], &lights);
            }
            screen.render(&camera, [&central_building as &dyn Object], &lights);
            for r in &resource_nodes {
                screen.render(&camera, [r as &dyn Object], &lights);
            }
            screen.render(&camera, [&home_marker    as &dyn Object], &lights);
            screen.render(&camera, [&dest_marker    as &dyn Object], &lights);
            screen.render(&camera, [&player_gm      as &dyn Object], &lights);

            // Yumon body
            if let Some(ref mut model) = gpu_model {
                for primitive in model.iter_mut() {
                    let original = primitive.transformation();
                    primitive.set_transformation(yumon_xform * original);
                    screen.render(&camera, [primitive as &dyn Object], &lights);
                    primitive.set_transformation(original);
                }
            } else {
                screen.render(&camera, [&fallback_sphere as &dyn Object], &lights);
            }

            screen.write(|| gui.render()).unwrap();
            FrameOutput::default()
        });
    }
}