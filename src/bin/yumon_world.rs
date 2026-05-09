#![recursion_limit = "256"]

//! src/bin/yumon_world.rs
//!
//! Yumon World — single Yumon + player, command-driven actions.
//! Run with: `cargo run --bin yumon_world`
//!
//! ── Controls ────────────────────────────────────────────────────────────────
//!   [Orbit mode]
//!     Left-drag          — orbit camera
//!     Scroll             — zoom
//!     Right-drag         — pan
//!     Left-click (ground)— move player OR set destination (see panel toggle)
//!
//!   [FPS mode]  press F or the "FPS" button to toggle
//!     W/A/S/D            — move player (relative to camera yaw)
//!     Mouse move         — look around (yaw / pitch)
//!     Left-click (ground)— set destination marker
//!     Esc                — return to Orbit mode

use std::{
    f32::consts::{PI, TAU},
    sync::mpsc,
    time::Instant,
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

// ─── noise crate ─────────────────────────────────────────────────────────────
use noise::{Fbm, NoiseFn, Perlin};

// ─── Tunables ────────────────────────────────────────────────────────────────

const ARENA: f32              = 20.0;
const WALL_H: f32             = 1.2;
const WALL_T: f32             = 0.35;
const MOVE_SPEED: f32         = 3.0;
const BUBBLE_TTL: f32         = 7.0;
const ACTION_INTERVAL_SECS: f32 = 10.0;
const ARRIVE_THRESH: f32      = 0.6;
const HOME_POS: Vec3          = Vec3 { x: -ARENA + 2.0, y: 0.0, z: -ARENA + 2.0 };
const DEST_INIT: Vec3         = Vec3 { x: 5.0, y: 0.0, z: 5.0 };
const PLAYER_INIT: Vec3       = Vec3 { x: 3.0, y: 0.0, z: -3.0 };
const PLAYER_SPEED: f32       = 4.5;
const MODEL_PATH: &str        = "data/models/animal-parrot.glb";

/// Terrain grid resolution (vertices per side).
const TERRAIN_RES: usize      = 128;
/// Maximum terrain height displacement.
const TERRAIN_AMPLITUDE: f32  = 2.8;
/// Noise frequency scale.
const TERRAIN_FREQ: f64       = 0.07;
/// Eye height above terrain for FPS camera.
const EYE_HEIGHT: f32         = 1.7;
/// FPS mouse sensitivity (radians per pixel).
const MOUSE_SENS: f32         = 0.0025;
/// FPS pitch limits.
const PITCH_MIN: f32          = -PI * 0.45;
const PITCH_MAX: f32          = PI * 0.45;

// ─── Click mode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum ClickMode { Camera, MovePlayer, SetDestination }

// ─── Camera mode ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum CameraMode { Orbit, Fps }

// ─── Anim / action states ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum AnimState { Idle, Walking, Building }

struct OrbitState {
    angle: f32,
}
impl OrbitState {
    fn new(start_pos: Vec3) -> Self {
        Self { angle: start_pos.z.atan2(start_pos.x) }
    }
    fn next_waypoint(&mut self) -> Vec3 {
        self.angle = (self.angle + TAU / 8.0) % TAU;
        let r = ARENA - 1.5;
        Vec3::new(r * self.angle.cos(), 0.0, r * self.angle.sin())
    }
}

// ─── Brain ↔ world message ────────────────────────────────────────────────────

pub struct WorldPrompt {
    pub prompt: String,
}

// ─── Terrain heightmap ────────────────────────────────────────────────────────

/// Generates a grid of (position, normal) pairs using fractional Brownian motion.
/// Returns (positions, normals, indices) ready for CpuMesh.
#[cfg(target_os = "windows")]
fn build_terrain_mesh() -> CpuMesh {
    let fbm: Fbm<Perlin> = Fbm::new(0);
    let n = TERRAIN_RES;
    let step = (ARENA * 2.0) / (n - 1) as f32;

    // ── Sample heights ────────────────────────────────────────────────────────
    let mut heights = vec![0.0f32; n * n];
    for zi in 0..n {
        for xi in 0..n {
            let wx = -ARENA + xi as f32 * step;
            let wz = -ARENA + zi as f32 * step;

            // Fade to 0 at the arena edges so walls sit flush.
            let edge_fade = {
                let fx = (1.0 - (wx / ARENA).abs()).clamp(0.0, 1.0);
                let fz = (1.0 - (wz / ARENA).abs()).clamp(0.0, 1.0);
                // Smooth-step both axes, take minimum.
                let sx = fx * fx * (3.0 - 2.0 * fx);
                let sz = fz * fz * (3.0 - 2.0 * fz);
                sx.min(sz)
            };

            let h = fbm.get([wx as f64 * TERRAIN_FREQ, wz as f64 * TERRAIN_FREQ]) as f32;
            heights[zi * n + xi] = h * TERRAIN_AMPLITUDE * edge_fade;
        }
    }

    // ── Build vertex positions ────────────────────────────────────────────────
    let mut positions: Vec<Vec3> = Vec::with_capacity(n * n);
    for zi in 0..n {
        for xi in 0..n {
            let wx = -ARENA + xi as f32 * step;
            let wz = -ARENA + zi as f32 * step;
            positions.push(Vec3::new(wx, heights[zi * n + xi], wz));
        }
    }

    // ── Build indices (two triangles per quad) ────────────────────────────────
    let mut indices: Vec<u32> = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for zi in 0..(n - 1) {
        for xi in 0..(n - 1) {
            let tl = (zi * n + xi) as u32;
            let tr = tl + 1;
            let bl = tl + n as u32;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    // ── Compute per-vertex normals (average of adjacent face normals) ──────────
    let mut normals: Vec<Vec3> = vec![Vec3::zero(); n * n];
    for i in (0..indices.len()).step_by(3) {
        let (a, b, c) = (
            indices[i]     as usize,
            indices[i + 1] as usize,
            indices[i + 2] as usize,
        );
        let n_face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += n_face;
        normals[b] += n_face;
        normals[c] += n_face;
    }
    let normals: Vec<Vec3> = normals.iter().map(|n| n.normalize()).collect();

    let mut mesh = CpuMesh {
        positions: Positions::F32(positions),
        normals:   Some(normals),
        indices:   Indices::U32(indices),
        ..Default::default()
    };
    mesh
}

/// Sample terrain height at world (x, z) using bilinear interpolation.
#[cfg(target_os = "windows")]
fn terrain_height_at(heights: &[f32], wx: f32, wz: f32) -> f32 {
    let n = TERRAIN_RES as f32;
    let step = (ARENA * 2.0) / (n - 1.0);
    let fx = ((wx + ARENA) / step).clamp(0.0, n - 1.0 - 1e-3);
    let fz = ((wz + ARENA) / step).clamp(0.0, n - 1.0 - 1e-3);
    let xi = fx as usize;
    let zi = fz as usize;
    let tx = fx - xi as f32;
    let tz = fz - zi as f32;
    let ni = TERRAIN_RES;
    let h00 = heights[zi * ni + xi];
    let h10 = heights[zi * ni + xi + 1];
    let h01 = heights[(zi + 1) * ni + xi];
    let h11 = heights[(zi + 1) * ni + xi + 1];
    let h0 = h00 + (h10 - h00) * tx;
    let h1 = h01 + (h11 - h01) * tx;
    h0 + (h1 - h0) * tz
}

// ─── Player ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
struct Player {
    pos:    Vec3,
    target: Vec3,
    /// Accumulated WASD velocity (set each frame in FPS mode).
    wasd:   Vec3,
}

#[cfg(target_os = "windows")]
impl Player {
    fn new() -> Self {
        Self {
            pos:    PLAYER_INIT,
            target: PLAYER_INIT,
            wasd:   Vec3::zero(),
        }
    }

    fn tick(&mut self, dt: f32, heights: &[f32]) {
        // In FPS mode we apply wasd velocity directly.
        if self.wasd.magnitude() > 0.001 {
            self.pos = Self::clamp_arena(self.pos + self.wasd * dt);
            self.target = self.pos;
        } else {
            // Click-to-move.
            let diff = self.target - self.pos;
            let dist = diff.magnitude();
            if dist > 0.02 {
                let move_vec = diff.normalize() * (PLAYER_SPEED * dt).min(dist);
                self.pos = Self::clamp_arena(self.pos + move_vec);
            }
        }
        // Snap player Y to terrain surface.
        self.pos.y = terrain_height_at(heights, self.pos.x, self.pos.z);
    }

    fn clamp_arena(mut p: Vec3) -> Vec3 {
        let lim = ARENA - 0.8;
        p.x = p.x.clamp(-lim, lim);
        p.z = p.z.clamp(-lim, lim);
        p
    }
}

// ─── FPS camera state ─────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
struct FpsState {
    yaw:   f32,  // horizontal rotation (radians)
    pitch: f32,  // vertical rotation (radians)
}

#[cfg(target_os = "windows")]
impl FpsState {
    fn new() -> Self {
        Self { yaw: 0.0, pitch: -0.15 }
    }

    /// Apply mouse delta (pixels).
    fn apply_mouse(&mut self, dx: f32, dy: f32) {
        self.yaw   -= dx * MOUSE_SENS;
        self.pitch  = (self.pitch - dy * MOUSE_SENS).clamp(PITCH_MIN, PITCH_MAX);
    }

    /// Forward direction (horizontal plane only, for WASD).
    fn forward_xz(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }

    fn right_xz(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    /// Full look direction (includes pitch).
    fn look_dir(&self) -> Vec3 {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        Vec3::new(sy * cp, sp, cy * cp).normalize()
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
    interval_timer:    f32,
    waiting_for_brain: bool,
    parsed_emotion:    String,
    log:               Vec<String>,
    current_command:   String,
    pending_message:   String,
    orbit:             Option<OrbitState>,
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
        self.pending_message.clear();
        self.orbit     = None;
        self.idle_timer = 0.0;
        match r.action {
            Action::GoToDestination => { self.set_walk_toward(dest_pos); }
            Action::GoHome          => { self.set_walk_toward(HOME_POS); }
            Action::Follow | Action::GetHelp => {
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
                self.idle_timer = 6.0;
            }
            Action::Collect => {
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
                self.set_walk_toward(BUILDING_POS);
                self.anim = AnimState::Building;
            }
            Action::Sit => {
                self.anim       = AnimState::Idle;
                self.target     = self.pos;
                self.idle_timer = 5.0;
            }
        }
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

    fn tick(&mut self, dt: f32, player_pos: Vec3, heights: &[f32]) {
        let diff = self.target - self.pos;
        let dist = diff.magnitude();
        if dist > 0.02 {
            let new_xz = self.pos + diff.normalize() * (MOVE_SPEED * dt).min(dist);
            self.pos.x = new_xz.x;
            self.pos.z = new_xz.z;
        } else if self.anim == AnimState::Walking {
            self.anim = AnimState::Idle;
        }
        // Snap Yumon Y to terrain.
        self.pos.y = terrain_height_at(heights, self.pos.x, self.pos.z);

        if self.action == Action::Follow {
            let offset = (self.pos - player_pos).normalize() * 1.2;
            self.set_walk_toward(player_pos + offset);
        }
        if self.action == Action::Survey && self.idle_timer > 0.0 {
            self.facing += dt * 0.6;
            self.idle_timer -= dt;
        }
        if self.action == Action::Sit && self.idle_timer > 0.0 {
            self.idle_timer -= dt;
        }
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
        let y = self.pos.y + 0.45 + self.bob(t);
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

    fn build_prompt(&self) -> String {
        let memories: Vec<serde_json::Value> = Vec::new();
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

#[cfg(target_os = "windows")]
fn ray_ground_intersect(camera: &Camera, pixel: PhysicalPoint) -> Option<Vec3> {
    let origin    = camera.position();
    let direction = camera.view_direction_at_pixel(pixel);
    let denom = direction.y;
    if denom.abs() < 1e-5 { return None; }
    let t = -origin.y / denom;
    if t < 0.0 { return None; }
    let p = origin + direction * t;
    let lim = ARENA - 0.6;
    Some(Vec3::new(p.x.clamp(-lim, lim), 0.0, p.z.clamp(-lim, lim)))
}

// ─── Water shader (GLSL fragment) ─────────────────────────────────────────────
//
// Ported from afl_ext's Shadertoy shader (MIT).
// three-d exposes a `GlowMaterial` / raw `Program` path; we use a custom
// `ColorMaterial` override with a hand-rolled `FragmentShader`.

/// GLSL fragment source for the water surface.
/// Uniforms: uTime (float), uResolution (vec2), uMouse (vec2)
const WATER_FRAG_GLSL: &str = r#"
precision highp float;

uniform float uTime;
uniform vec2  uResolution;
uniform vec2  uMouse;

in vec2 v_texcoord;   // [0..1] from vertex shader

#define DRAG_MULT         0.38
#define WATER_DEPTH       1.0
#define CAMERA_HEIGHT     1.5
#define ITERATIONS_RAYMARCH 12
#define ITERATIONS_NORMAL   36

// ── Wave helpers ──────────────────────────────────────────────────────────────
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
    float x    = dot(direction, position) * frequency + timeshift;
    float wave = exp(sin(x) - 1.0);
    float dx   = wave * cos(x);
    return vec2(wave, -dx);
}

float getwaves(vec2 position, int iterations) {
    float wavePhaseShift = length(position) * 0.1;
    float iter           = 0.0;
    float frequency      = 1.0;
    float timeMultiplier = 2.0;
    float weight         = 1.0;
    float sumOfValues    = 0.0;
    float sumOfWeights   = 0.0;
    for (int i = 0; i < iterations; i++) {
        vec2  p   = vec2(sin(iter), cos(iter));
        vec2  res = wavedx(position, p, frequency, uTime * timeMultiplier + wavePhaseShift);
        position += p * res.y * weight * DRAG_MULT;
        sumOfValues  += res.x * weight;
        sumOfWeights += weight;
        weight        = mix(weight, 0.0, 0.2);
        frequency    *= 1.18;
        timeMultiplier *= 1.07;
        iter          += 1232.399963;
    }
    return sumOfValues / sumOfWeights;
}

float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth) {
    vec3 pos = start;
    vec3 dir = normalize(end - start);
    for (int i = 0; i < 64; i++) {
        float height = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
        if (height + 0.01 > pos.y) { return distance(pos, camera); }
        pos += dir * (pos.y - height);
    }
    return distance(start, camera);
}

vec3 normal(vec2 pos, float e, float depth) {
    vec2  ex = vec2(e, 0.0);
    float H  = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
    vec3  a  = vec3(pos.x, H, pos.y);
    return normalize(cross(
        a - vec3(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y),
        a - vec3(pos.x,     getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
    ));
}

mat3 rotAxisAngle(vec3 axis, float angle) {
    float s = sin(angle), c = cos(angle), oc = 1.0 - c;
    return mat3(
        oc*axis.x*axis.x+c,       oc*axis.x*axis.y-axis.z*s, oc*axis.z*axis.x+axis.y*s,
        oc*axis.x*axis.y+axis.z*s, oc*axis.y*axis.y+c,       oc*axis.y*axis.z-axis.x*s,
        oc*axis.z*axis.x-axis.y*s, oc*axis.y*axis.z+axis.x*s, oc*axis.z*axis.z+c
    );
}

vec3 getRay(vec2 fragCoord) {
    vec2 nm = uMouse / uResolution;
    vec2 uv = ((fragCoord / uResolution) * 2.0 - 1.0) * vec2(uResolution.x / uResolution.y, 1.0);
    vec3 proj = normalize(vec3(uv.x, uv.y, 1.5));
    if (uResolution.x < 600.0) return proj;
    return rotAxisAngle(vec3(0.0, -1.0, 0.0), 3.0 * ((nm.x + 0.5) * 2.0 - 1.0))
         * rotAxisAngle(vec3(1.0,  0.0, 0.0), 0.5 + 1.5 * (((nm.y == 0.0 ? 0.27 : nm.y)) * 2.0 - 1.0))
         * proj;
}

float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 nrm) {
    return clamp(dot(point - origin, nrm) / dot(direction, nrm), -1.0, 9991999.0);
}

vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir) {
    float special_trick  = 1.0 / (raydir.y * 1.0 + 0.1);
    float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
    float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
    float sundt    = pow(max(0.0, dot(sundir, raydir)), 8.0);
    float mymie    = sundt * special_trick * 0.2;
    vec3 suncolor  = mix(vec3(1.0),
                        max(vec3(0.0), vec3(1.0) - vec3(5.5, 13.0, 22.4) / 22.4),
                        special_trick2);
    vec3 bluesky   = vec3(5.5, 13.0, 22.4) / 22.4 * suncolor;
    vec3 bluesky2  = max(vec3(0.0),
                        bluesky - vec3(5.5, 13.0, 22.4) * 0.002 *
                        (special_trick + -6.0 * sundir.y * sundir.y));
    bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
    return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0));
}

vec3 getSunDirection() {
    return normalize(vec3(-0.077, 0.5 + sin(uTime * 0.2 + 2.6) * 0.45, 0.577));
}
vec3 getAtmosphere(vec3 dir) { return extra_cheap_atmosphere(dir, getSunDirection()) * 0.5; }
float getSun(vec3 dir)        { return pow(max(0.0, dot(dir, getSunDirection())), 720.0) * 210.0; }

vec3 aces_tonemap(vec3 color) {
    mat3 m1 = mat3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
    mat3 m2 = mat3(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
    vec3 v  = m1 * color;
    vec3 a  = v * (v + 0.0245786) - 0.000090537;
    vec3 b  = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

out vec4 fragColor;

void main() {
    vec2 fragCoord = v_texcoord * uResolution;
    vec3 ray       = getRay(fragCoord);
    if (ray.y >= 0.0) {
        vec3 C = getAtmosphere(ray) + getSun(ray);
        fragColor = vec4(aces_tonemap(C * 2.0), 0.85);
        return;
    }
    vec3 origin = vec3(uTime * 0.2, CAMERA_HEIGHT, 1.0);
    float highPlaneHit = intersectPlane(origin, ray, vec3(0.0), vec3(0.0, 1.0, 0.0));
    float lowPlaneHit  = intersectPlane(origin, ray, vec3(0.0, -WATER_DEPTH, 0.0), vec3(0.0, 1.0, 0.0));
    vec3 highHitPos = origin + ray * highPlaneHit;
    vec3 lowHitPos  = origin + ray * lowPlaneHit;
    float dist      = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
    vec3  waterHitPos = origin + ray * dist;
    vec3  N  = normal(waterHitPos.xz, 0.01, WATER_DEPTH);
    N = mix(N, vec3(0.0, 1.0, 0.0), 0.8 * min(1.0, sqrt(dist * 0.01) * 1.1));
    float fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - max(0.0, dot(-N, ray)), 5.0);
    vec3 R = normalize(reflect(ray, N));
    R.y = abs(R.y);
    vec3 reflection  = getAtmosphere(R) + getSun(R);
    vec3 scattering  = vec3(0.0293, 0.0698, 0.1717) * 0.1 *
                       (0.2 + (waterHitPos.y + WATER_DEPTH) / WATER_DEPTH);
    vec3 C = fresnel * reflection + scattering;
    fragColor = vec4(aces_tonemap(C * 2.0), 0.92);
}
"#;

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
                        "checkpoints/brain/128h_2l_2a_64len_6e", &device,
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

        // ── Terrain data (CPU side, for height sampling) ───────────────────────
        let fbm_cpu: Fbm<Perlin> = Fbm::new(0);
        let terrain_n = TERRAIN_RES;
        let terrain_step = (ARENA * 2.0) / (terrain_n - 1) as f32;
        let mut terrain_heights = vec![0.0f32; terrain_n * terrain_n];
        for zi in 0..terrain_n {
            for xi in 0..terrain_n {
                let wx = -ARENA + xi as f32 * terrain_step;
                let wz = -ARENA + zi as f32 * terrain_step;
                let edge_fade = {
                    let fx = (1.0 - (wx / ARENA).abs()).clamp(0.0, 1.0);
                    let fz = (1.0 - (wz / ARENA).abs()).clamp(0.0, 1.0);
                    let sx = fx * fx * (3.0 - 2.0 * fx);
                    let sz = fz * fz * (3.0 - 2.0 * fz);
                    sx.min(sz)
                };
                let h = fbm_cpu.get([wx as f64 * TERRAIN_FREQ, wz as f64 * TERRAIN_FREQ]) as f32;
                terrain_heights[zi * terrain_n + xi] = h * TERRAIN_AMPLITUDE * edge_fade;
            }
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

        // FPS state
        let mut fps_state  = FpsState::new();
        let mut camera_mode = CameraMode::Orbit;
        let mut keys_held   = std::collections::HashSet::<Key>::new();

        // ── Lighting ──────────────────────────────────────────────────────────
        let ambient     = AmbientLight::new(&context, 0.45, Srgba::WHITE);
        let directional = DirectionalLight::new(
            &context, 1.3, Srgba::WHITE, &Vec3::new(-1.0, -2.5, -1.5),
        );

        // ── Terrain mesh ──────────────────────────────────────────────────────
        let terrain_cpu = build_terrain_mesh();
        let mut terrain_gm = Gm::new(
            Mesh::new(&context, &terrain_cpu),
            PhysicalMaterial::new_opaque(&context, &CpuMaterial {
                albedo:    Srgba::new(60, 90, 50, 255),
                roughness: 0.90,
                metallic:  0.0,
                ..Default::default()
            }),
        );

        // ── Water quad (fullscreen on y=0 plane) ──────────────────────────────
        //
        // We render a large flat quad at y = 0 covering the arena, then in the
        // fragment shader the "camera" and ray are purely synthetic — independent
        // of the 3-D camera — matching the original Shadertoy behaviour.
        //
        // three-d doesn't have a first-class "custom GLSL material" API exposed
        // in the high-level path, so we construct a raw GL Program and draw it
        // ourselves after the main pass.
        //
        // Build a simple fullscreen quad VAO in world space (y=0, covers arena).
        let water_size = ARENA * 1.2;
        // We'll use ColorMaterial + override the shader via `Program`.
        // Instead, build vertex positions + upload via raw GL.
        use three_d::context::{
            Buffer, ARRAY_BUFFER, FLOAT, STATIC_DRAW, TRIANGLES,
        };
        let gl = context.clone();
        let water_verts: [f32; 20] = [
            // x,       y,  z,       u,    v
            -water_size, 0.0, -water_size, 0.0, 1.0,
             water_size, 0.0, -water_size, 1.0, 1.0,
             water_size, 0.0,  water_size, 1.0, 0.0,
            -water_size, 0.0,  water_size, 0.0, 0.0,
        ];
        let water_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        // Upload to GPU using three-d's context helpers.
        // `three_d::context` re-exports `glow` (GL on Web) or platform GL.
        // We use the `Program` struct that three_d provides for raw shaders.
        let water_vert_glsl = r#"
            in vec3 a_position;
            in vec2 a_uv;
            out vec2 v_texcoord;
            uniform mat4 viewProjection;
            void main() {
                v_texcoord  = a_uv;
                gl_Position = viewProjection * vec4(a_position, 1.0);
            }
        "#;

        // Build the raw shader Program (three-d wraps glow).
        // three-d 0.16+ exposes `Program::from_source`.
        let water_program = Program::from_source(
            &context,
            water_vert_glsl,
            WATER_FRAG_GLSL,
        )
        .expect("Failed to compile water shader");

        // Build VBO / IBO with three-d's VertexBuffer / ElementBuffer wrappers.
        let water_vbo = VertexBuffer::new_with_data(
            &context,
            &water_verts.to_vec(),
        );
        let water_ibo = ElementBuffer::new_with_data(
            &context,
            &water_indices.to_vec(),
        );

        // ── Walls ─────────────────────────────────────────────────────────────
        let wall_cpu = CpuMesh::cube();
        let wall_mat = CpuMaterial {
            albedo: Srgba::new(175, 155, 125, 255),
            roughness: 0.85,
            metallic: 0.0,
            ..Default::default()
        };
        let fs   = ARENA * 2.0;
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
                let h = terrain_height_at(&terrain_heights, pos.x, pos.z);
                let mut gm = Gm::new(
                    Mesh::new(&context, &resource_sphere_cpu),
                    PhysicalMaterial::new_opaque(&context, &resource_mat),
                );
                gm.set_transformation(
                    Mat4::from_translation(Vec3::new(pos.x, h + 0.30, pos.z))
                        * Mat4::from_scale(0.28),
                );
                gm
            }).collect();

        // ── Home marker ───────────────────────────────────────────────────────
        let home_h = terrain_height_at(&terrain_heights, HOME_POS.x, HOME_POS.z);
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
            Mat4::from_translation(Vec3::new(HOME_POS.x, home_h + 0.35, HOME_POS.z))
                * Mat4::from_scale(0.30),
        );

        // ── Destination marker ────────────────────────────────────────────────
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
        {
            let h = terrain_height_at(&terrain_heights, dest_pos.x, dest_pos.z);
            dest_marker.set_transformation(
                Mat4::from_translation(Vec3::new(dest_pos.x, h + 0.03, dest_pos.z))
                    * Mat4::from_nonuniform_scale(0.4, 0.04, 0.4),
            );
        }

        // ── Player cylinder ───────────────────────────────────────────────────
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

        // ── Yumon GLB ─────────────────────────────────────────────────────────
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
        let mut ui_command    = String::new();
        let mut ui_message    = String::new();
        let mut last_frame    = Instant::now();
        let mut mouse_pos: (f32, f32) = (0.0, 0.0);

        // ── Render loop ───────────────────────────────────────────────────────
        window.render_loop(move |mut frame_input| {
            let now = Instant::now();
            let dt  = now.duration_since(last_frame).as_secs_f32().min(0.1);
            last_frame = now;
            let t   = frame_input.accumulated_time;
            let vp  = frame_input.viewport;

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

            // ── Input processing ──────────────────────────────────────────────
            let mut gui_consumed    = false;
            let mut left_click: Option<PixelPoint> = None;
            let mut fps_mouse_delta = (0.0f32, 0.0f32);

            for event in &frame_input.events {
                match event {
                    Event::MouseMotion { delta, position, .. } => {
                        mouse_pos = (position.x as f32, position.y as f32);
                        if camera_mode == CameraMode::Fps {
                            fps_mouse_delta.0 += delta.0 as f32;
                            fps_mouse_delta.1 += delta.1 as f32;
                        }
                    }
                    Event::MousePress {
                        button: MouseButton::Left,
                        position, ..
                    } => {
                        left_click = Some(*position);
                    }
                    Event::KeyPress { kind, .. } => {
                        keys_held.insert(*kind);
                        // Toggle FPS/Orbit with F key.
                        if *kind == Key::F {
                            camera_mode = match camera_mode {
                                CameraMode::Orbit => CameraMode::Fps,
                                CameraMode::Fps   => CameraMode::Orbit,
                            };
                        }
                        if *kind == Key::Escape && camera_mode == CameraMode::Fps {
                            camera_mode = CameraMode::Orbit;
                        }
                    }
                    Event::KeyRelease { kind, .. } => {
                        keys_held.remove(kind);
                    }
                    _ => {}
                }
            }

            // ── FPS camera update ─────────────────────────────────────────────
            if camera_mode == CameraMode::Fps {
                // Mouse look.
                fps_state.apply_mouse(fps_mouse_delta.0, fps_mouse_delta.1);

                // WASD movement — move the player, camera follows.
                let fwd   = fps_state.forward_xz();
                let right = fps_state.right_xz();
                let mut move_vel = Vec3::zero();
                if keys_held.contains(&Key::W) { move_vel += fwd;   }
                if keys_held.contains(&Key::S) { move_vel -= fwd;   }
                if keys_held.contains(&Key::A) { move_vel -= right;  }
                if keys_held.contains(&Key::D) { move_vel += right;  }
                if move_vel.magnitude() > 0.001 {
                    player.wasd = move_vel.normalize() * PLAYER_SPEED;
                } else {
                    player.wasd = Vec3::zero();
                }

                // Position camera at player eye height.
                let eye = Vec3::new(
                    player.pos.x,
                    player.pos.y + EYE_HEIGHT,
                    player.pos.z,
                );
                let look_dir  = fps_state.look_dir();
                let target    = eye + look_dir;
                camera.set_view(eye, target, Vec3::unit_y());
            } else {
                // Orbit mode: clear WASD velocity.
                player.wasd = Vec3::zero();
                if !gui_consumed {
                    orbit_ctrl.handle_events(&mut camera, &mut frame_input.events);
                }

                // Ground click.
                if !gui_consumed {
                    if let Some(pixel) = left_click {
                        if let Some(world_pt) = ray_ground_intersect(&camera, pixel) {
                            match click_mode {
                                ClickMode::Camera => {}
                                ClickMode::MovePlayer => {
                                    player.target = Player::clamp_arena(world_pt);
                                }
                                ClickMode::SetDestination => {
                                    dest_pos = world_pt;
                                    let h = terrain_height_at(&terrain_heights, dest_pos.x, dest_pos.z);
                                    dest_marker.set_transformation(
                                        Mat4::from_translation(Vec3::new(dest_pos.x, h + 0.03, dest_pos.z))
                                            * Mat4::from_nonuniform_scale(0.4, 0.04, 0.4),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // ── Tick ──────────────────────────────────────────────────────────
            player.tick(dt, &terrain_heights);
            yumon.tick(dt, player.pos, &terrain_heights);

            // GetHelp orbit waypoint advancement.
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

                            // ── Camera mode toggle ─────────────────────────────
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new("Camera mode  (F to toggle)")
                                    .size(11.0)
                                    .color(Color32::from_gray(140)),
                            );
                            ui.horizontal(|ui| {
                                if ui.radio(
                                    camera_mode == CameraMode::Orbit, "🎥 Orbit",
                                ).clicked() {
                                    camera_mode = CameraMode::Orbit;
                                }
                                if ui.radio(
                                    camera_mode == CameraMode::Fps, "👁 FPS",
                                ).clicked() {
                                    camera_mode = CameraMode::Fps;
                                }
                            });
                            if camera_mode == CameraMode::Fps {
                                ui.label(
                                    RichText::new("WASD: move  •  Mouse: look  •  Esc: exit FPS")
                                        .size(10.0)
                                        .color(Color32::from_rgb(130, 200, 130)),
                                );
                            }

                            ui.separator();

                            // ── Click mode toggle (Orbit only) ─────────────────
                            if camera_mode == CameraMode::Orbit {
                                ui.label(
                                    RichText::new("Click mode")
                                        .size(11.0)
                                        .color(Color32::from_gray(140)),
                                );
                                ui.horizontal(|ui| {
                                    ui.radio_value(&mut click_mode, ClickMode::Camera, "🎥 Camera");
                                    ui.radio_value(&mut click_mode, ClickMode::MovePlayer, "🚶 Player");
                                    ui.radio_value(&mut click_mode, ClickMode::SetDestination, "📍 Dest");
                                });
                                ui.separator();
                            }

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
                                    if !yumon.speech.is_empty() {
                                        let alpha = ((yumon.speech_timer / BUBBLE_TTL) * 255.0)
                                            .clamp(0.0, 255.0) as u8;
                                        ui.colored_label(
                                            Color32::from_rgba_unmultiplied(230, 225, 170, alpha),
                                            format!("\"{}\"", yumon.speech),
                                        );
                                        ui.add_space(2.0);
                                    }
                                    for entry in yumon.log.iter().rev().take(5) {
                                        ui.label(
                                            RichText::new(entry)
                                                .size(11.0)
                                                .color(Color32::from_gray(160)),
                                        );
                                    }
                                    ui.add_space(4.0);
                                    ui.separator();
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
                                            yumon.push_log(format!("[CMD] {}", yumon.current_command));
                                            yumon.interval_timer = 0.0;
                                        }
                                    });
                                    if !yumon.current_command.is_empty() {
                                        ui.label(
                                            RichText::new(format!("▶ {}", yumon.current_command))
                                                .size(11.0)
                                                .color(Color32::from_rgb(130, 200, 130)),
                                        );
                                    }
                                    ui.add_space(4.0);
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
                                            "F: toggle FPS  •  drag: orbit  •  scroll: zoom",
                                        )
                                        .size(10.0)
                                        .color(Color32::from_gray(100)),
                                    );
                                },
                            );
                        });
                },
            );

            // ── Camera viewport ───────────────────────────────────────────────
            camera.set_viewport(vp);

            // ── Update player/Yumon transforms ────────────────────────────────
            let player_h = terrain_height_at(&terrain_heights, player.pos.x, player.pos.z);
            player_gm.set_transformation(
                Mat4::from_translation(Vec3::new(player.pos.x, player_h + 0.5, player.pos.z))
                    * Mat4::from_nonuniform_scale(0.25, 0.5, 0.25),
            );
            // In FPS mode the player gm is the camera, hide it by pushing underground.
            if camera_mode == CameraMode::Fps {
                player_gm.set_transformation(
                    Mat4::from_translation(Vec3::new(player.pos.x, player_h - 2.0, player.pos.z))
                        * Mat4::from_nonuniform_scale(0.25, 0.5, 0.25),
                );
            }

            let yumon_xform = yumon.world_transform(t);
            if gpu_model.is_none() {
                fallback_sphere.set_transformation(yumon_xform);
            }

            // ── Draw ──────────────────────────────────────────────────────────
            let lights: [&dyn Light; 2] = [&ambient, &directional];
            let screen = frame_input.screen();
            screen.clear(ClearState::color_and_depth(0.10, 0.12, 0.15, 1.0, 1.0));

            // Terrain replaces the old flat floor.
            screen.render(&camera, [&terrain_gm as &dyn Object], &lights);

            for w in &walls {
                screen.render(&camera, [w as &dyn Object], &lights);
            }
            screen.render(&camera, [&central_building as &dyn Object], &lights);
            for r in &resource_nodes {
                screen.render(&camera, [r as &dyn Object], &lights);
            }
            screen.render(&camera, [&home_marker as &dyn Object], &lights);
            screen.render(&camera, [&dest_marker  as &dyn Object], &lights);
            screen.render(&camera, [&player_gm   as &dyn Object], &lights);

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

            // ── Water pass ────────────────────────────────────────────────────
            // Draw last so alpha blending composites over terrain.
            screen.write::<RendererError>(|| {
                use three_d::context::*;
                let gl_ctx = context.as_ref(); // raw glow Context

                // Enable blending for translucent water.
                unsafe {
                    gl_ctx.enable(BLEND);
                    gl_ctx.blend_func(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
                }

                let vp_mat = camera.projection() * camera.view();

                water_program.use_uniform("viewProjection", vp_mat);
                water_program.use_uniform("uTime",       t as f32 * 0.001);
                water_program.use_uniform("uResolution", Vec2::new(vp.width as f32, vp.height as f32));
                water_program.use_uniform("uMouse",      Vec2::new(mouse_pos.0, mouse_pos.1));

                // Bind position (stride = 5 floats, offset 0, 3 components).
                water_program.use_vertex_attribute("a_position", &water_vbo);
                // Bind UV (stride = 5 floats, offset 3, 2 components).
                water_program.use_vertex_attribute("a_uv", &water_vbo);

                water_program.draw_elements(
                    RenderStates {
                        blend: Blend::TRANSPARENCY,
                        depth_test: DepthTest::Less,
                        ..Default::default()
                    },
                    vp,
                    &water_ibo,
                );

                unsafe { gl_ctx.disable(BLEND); }

                Ok(())
            }).unwrap();

            screen.write(|| gui.render()).unwrap();

            FrameOutput::default()
        });
    }
}