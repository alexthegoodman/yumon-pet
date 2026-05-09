#![recursion_limit = "256"]

//! src/bin/yumon_world.rs
//!
//! Yumon World — FPS-only, gamepad-controlled, command-driven.
//!
//! ── Controls ────────────────────────────────────────────────────────────────
//!   [Gamepad]
//!     Left Stick         — move player
//!     Right Stick        — look around
//!     Button North (Y)   — toggle command view
//!     Button South (A)   — send command (in command view)
//!     Button East (B)    — cancel command
//!
//!   [Keyboard]
//!     W/A/S/D            — move player
//!     Mouse move         — look around
//!     Enter              — send command (if active)
//!     Esc                — exit command view / quit

use std::{
    f32::consts::{PI, TAU},
    sync::{mpsc, Arc},
    time::Instant,
};

use three_d::*;

use yumon_pet::brain::{
    model::{GenerationResult, YumonBrain},
    samples::Action,
};

// ─── noise crate ─────────────────────────────────────────────────────────────
use noise::{Fbm, NoiseFn, Perlin};

// ─── Gamepad ─────────────────────────────────────────────────────────────────
use gilrs::{Axis, Button, Gilrs};

// ─── Text Rendering ──────────────────────────────────────────────────────────
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};

// ─── Tunables ────────────────────────────────────────────────────────────────

const ARENA: f32              = 20.0;
const MOVE_SPEED: f32         = 4.5;
const BUBBLE_TTL: f32         = 7.0;
const ACTION_INTERVAL_SECS: f32 = 10.0;
const PLAYER_INIT: Vec3       = Vec3 { x: 3.0, y: 0.0, z: -3.0 };
const HOME_POS: Vec3          = Vec3 { x: -ARENA + 2.0, y: 0.0, z: -ARENA + 2.0 };
const DEST_INIT: Vec3         = Vec3 { x: 5.0, y: 0.0, z: 5.0 };
const MODEL_PATH: &str        = "data/models/animal-parrot.glb";

const TERRAIN_RES: usize      = 128;
const TERRAIN_AMPLITUDE: f32  = 2.8;
const TERRAIN_FREQ: f64       = 0.07;
const EYE_HEIGHT: f32         = 1.7;
const MOUSE_SENS: f32         = 0.0025;
const GAMEPAD_SENS: f32       = 2.0;
const PITCH_MIN: f32          = -PI * 0.45;
const PITCH_MAX: f32          = PI * 0.45;

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

pub struct WorldPrompt {
    pub prompt: String,
}

// ─── Terrain ─────────────────────────────────────────────────────────────────

fn build_terrain_mesh() -> CpuMesh {
    let fbm: Fbm<Perlin> = Fbm::new(0);
    let n = TERRAIN_RES;
    let step = (ARENA * 2.0) / (n - 1) as f32;

    let mut heights = vec![0.0f32; n * n];
    for zi in 0..n {
        for xi in 0..n {
            let wx = -ARENA + xi as f32 * step;
            let wz = -ARENA + zi as f32 * step;
            let edge_fade = {
                let fx = (1.0 - (wx / ARENA).abs()).clamp(0.0, 1.0);
                let fz = (1.0 - (wz / ARENA).abs()).clamp(0.0, 1.0);
                let sx = fx * fx * (3.0 - 2.0 * fx);
                let sz = fz * fz * (3.0 - 2.0 * fz);
                sx.min(sz)
            };
            let h = fbm.get([wx as f64 * TERRAIN_FREQ, wz as f64 * TERRAIN_FREQ]) as f32;
            heights[zi * n + xi] = h * TERRAIN_AMPLITUDE * edge_fade;
        }
    }

    let mut positions: Vec<Vec3> = Vec::with_capacity(n * n);
    for zi in 0..n {
        for xi in 0..n {
            let wx = -ARENA + xi as f32 * step;
            let wz = -ARENA + zi as f32 * step;
            positions.push(Vec3::new(wx, heights[zi * n + xi], wz));
        }
    }

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

    let mut normals: Vec<Vec3> = vec![Vec3::zero(); n * n];
    for i in (0..indices.len()).step_by(3) {
        let (a, b, c) = (indices[i] as usize, indices[i + 1] as usize, indices[i + 2] as usize);
        let n_face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += n_face;
        normals[b] += n_face;
        normals[c] += n_face;
    }
    let normals: Vec<Vec3> = normals.iter().map(|n| n.normalize()).collect();

    CpuMesh {
        positions: Positions::F32(positions),
        normals:   Some(normals),
        indices:   Indices::U32(indices),
        ..Default::default()
    }
}

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

struct Player {
    pos:    Vec3,
    vel:    Vec3,
}

impl Player {
    fn new() -> Self {
        Self {
            pos:    PLAYER_INIT,
            vel:    Vec3::zero(),
        }
    }

    fn tick(&mut self, dt: f32, heights: &[f32]) {
        if self.vel.magnitude() > 0.001 {
            self.pos = Self::clamp_arena(self.pos + self.vel * dt);
        }
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

struct FpsState {
    yaw:   f32,
    pitch: f32,
}

impl FpsState {
    fn new() -> Self {
        Self { yaw: 0.0, pitch: -0.15 }
    }

    fn apply_mouse(&mut self, dx: f32, dy: f32) {
        self.yaw   -= dx * MOUSE_SENS;
        self.pitch  = (self.pitch - dy * MOUSE_SENS).clamp(PITCH_MIN, PITCH_MAX);
    }

    fn apply_gamepad(&mut self, dx: f32, dy: f32, dt: f32) {
        self.yaw   -= dx * GAMEPAD_SENS * dt;
        self.pitch  = (self.pitch - dy * GAMEPAD_SENS * dt).clamp(PITCH_MIN, PITCH_MAX);
    }

    fn forward_xz(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }

    fn right_xz(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    fn look_dir(&self) -> Vec3 {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        Vec3::new(sy * cp, sp, cy * cp).normalize()
    }
}

// ─── Yumon ───────────────────────────────────────────────────────────────────

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
    current_command:   String,
    pending_message:   String,
    orbit:             Option<OrbitState>,
    idle_timer:        f32,
}

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
            current_command:   String::new(),
            pending_message:   String::new(),
            orbit:             None,
            idle_timer:        0.0,
        }
    }

    fn apply_result(&mut self, r: GenerationResult, player_pos: Vec3, dest_pos: Vec3) {
        self.action         = r.action;
        self.parsed_emotion = r.parsed_emotion.clone();
        self.waiting_for_brain = false;
        if !r.reply.is_empty() {
            self.speech       = r.reply;
            self.speech_timer = BUBBLE_TTL;
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
                let nearest = RESOURCE_POSITIONS.iter().min_by(|a, b| {
                    ((**a) - self.pos).magnitude2().partial_cmp(&((**b) - self.pos).magnitude2()).unwrap()
                }).copied().unwrap_or(Vec3::zero());
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
        self.target = Vec3::new(dest.x.clamp(-ARENA+0.6, ARENA-0.6), 0.0, dest.z.clamp(-ARENA+0.6, ARENA-0.6));
        let diff    = self.target - self.pos;
        if diff.magnitude() > 0.1 {
            self.facing = diff.z.atan2(diff.x) - PI * 0.5;
        }
        self.anim = AnimState::Walking;
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
        self.pos.y = terrain_height_at(heights, self.pos.x, self.pos.z);

        if self.action == Action::Follow {
            let offset = (self.pos - player_pos).normalize() * 1.2;
            self.set_walk_toward(player_pos + offset);
        }
        if self.action == Action::Survey && self.idle_timer > 0.0 {
            self.facing += dt * 0.6;
            self.idle_timer -= dt;
        }
        if self.speech_timer > 0.0 {
            self.speech_timer -= dt;
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
        serde_json::to_string_pretty(&serde_json::json!({
            "command": self.current_command,
            "message": self.pending_message,
            "memories": Vec::<serde_json::Value>::new(),
        }))
        .unwrap()
    }
}

const BUILDING_POS: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
const RESOURCE_POSITIONS: [Vec3; 3] = [
    Vec3 { x: -4.0, y: 0.0, z: -2.5 },
    Vec3 { x:  4.5, y: 0.0, z:  1.0 },
    Vec3 { x:  0.5, y: 0.0, z:  4.5 },
];

// ─── Water shader (GLSL fragment) ─────────────────────────────────────────────

const WATER_FRAG_GLSL: &str = r#"
precision highp float;
uniform vec4  uTimeRes;
uniform vec3  uEye;
in vec3 v_world_pos;
#define uTime uTimeRes.x
#define WATER_DEPTH 1.0
#define ITERATIONS_RAYMARCH 12
#define ITERATIONS_NORMAL 36
#define DRAG_MULT 0.38

vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

float getwaves(vec2 position, int iterations) {
    float wavePhaseShift = length(position) * 0.1;
    float iter = 0.0;
    float frequency = 1.0;
    float timeMultiplier = 2.0;
    float weight = 1.0;
    float sumOfValues = 0.0;
    float sumOfWeights = 0.0;
    for (int i = 0; i < iterations; i++) {
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, frequency, uTime * timeMultiplier + wavePhaseShift);
        position += p * res.y * weight * DRAG_MULT;
        sumOfValues += res.x * weight;
        sumOfWeights += weight;
        weight = mix(weight, 0.0, 0.2);
        frequency *= 1.18;
        timeMultiplier *= 1.07;
        iter += 1232.399963;
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
    vec2 ex = vec2(e, 0.0);
    float H = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
    vec3 a = vec3(pos.x, H, pos.y);
    return normalize(cross(
        a - vec3(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y),
        a - vec3(pos.x,     getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
    ));
}

float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 nrm) {
    return clamp(dot(point - origin, nrm) / dot(direction, nrm), -1.0, 9991999.0);
}

vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir) {
    float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
    float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
    float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
    float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
    float mymie = sundt * special_trick * 0.2;
    vec3 suncolor = mix(vec3(1.0), max(vec3(0.0), vec3(1.0) - vec3(5.5, 13.0, 22.4) / 22.4), special_trick2);
    vec3 bluesky = vec3(5.5, 13.0, 22.4) / 22.4 * suncolor;
    vec3 bluesky2 = max(vec3(0.0), bluesky - vec3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
    bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
    return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0));
}

vec3 getSunDirection() { return normalize(vec3(-0.077, 0.5 + sin(uTime * 0.2 + 2.6) * 0.45, 0.577)); }
vec3 getAtmosphere(vec3 dir) { return extra_cheap_atmosphere(dir, getSunDirection()) * 0.5; }
float getSun(vec3 dir) { return pow(max(0.0, dot(dir, getSunDirection())), 720.0) * 210.0; }

vec3 aces_tonemap(vec3 color) {
    mat3 m1 = mat3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
    mat3 m2 = mat3(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

out vec4 fragColor;
void main() {
    vec3 ray = normalize(v_world_pos - uEye);
    if (ray.y >= 0.0) {
        vec3 C = getAtmosphere(ray) + getSun(ray);
        fragColor = vec4(aces_tonemap(C * 2.0), 0.85);
        return;
    }
    vec3 origin = uEye;
    float highPlaneHit = intersectPlane(origin, ray, vec3(0.0), vec3(0.0, 1.0, 0.0));
    float lowPlaneHit = intersectPlane(origin, ray, vec3(0.0, -WATER_DEPTH, 0.0), vec3(0.0, 1.0, 0.0));
    vec3 highHitPos = origin + ray * highPlaneHit;
    vec3 lowHitPos = origin + ray * lowPlaneHit;
    float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
    vec3 waterHitPos = origin + ray * dist;
    vec3 N = normal(waterHitPos.xz, 0.01, WATER_DEPTH);
    N = mix(N, vec3(0.0, 1.0, 0.0), 0.8 * min(1.0, sqrt(dist * 0.01) * 1.1));
    float fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - max(0.0, dot(-N, ray)), 5.0);
    vec3 R = normalize(reflect(ray, N));
    R.y = abs(R.y);
    vec3 reflection = getAtmosphere(R) + getSun(R);
    vec3 scattering = vec3(0.0293, 0.0698, 0.1717) * 0.1 * (0.2 + (waterHitPos.y + WATER_DEPTH) / WATER_DEPTH);
    vec3 C = fresnel * reflection + scattering;
    fragColor = vec4(aces_tonemap(C * 2.0), 0.92);
}
"#;

// ─── Text Renderer ───────────────────────────────────────────────────────────

struct TextRenderer {
    context: Context,
    font:    Option<FontRef<'static>>,
    quad:    Gm<Mesh, ColorMaterial>,
    texture: Option<Texture2DRef>,
}

impl TextRenderer {
    fn new(context: &Context) -> Self {
        let paths = [
            "C:/Users/alext/projects/yumon/yumon-pet/src/bin/Lemon-Regular.ttf"
        ];
        let font = paths.iter().find_map(|p| {
            std::fs::read(p).ok().and_then(|data| {
                FontRef::try_from_slice(Box::leak(data.into_boxed_slice())).ok()
            })
        });

        let quad = Gm::new(
            Mesh::new(context, &CpuMesh::square()),
            ColorMaterial {
                color: Srgba::WHITE,
                ..Default::default()
            },
        );

        Self { context: context.clone(), font, quad, texture: None }
    }

    fn update(&mut self, text: &str, viewport: Viewport) {
        let font = match &self.font {
            Some(f) => f,
            None => return,
        };
        if text.is_empty() {
            self.quad.set_transformation(Mat4::from_scale(0.0));
            return;
        }

        let scale = PxScale::from(24.0);
        let scaled_font = font.as_scaled(scale);
        
        let mut glyphs = Vec::new();
        let mut x = 0.0;
        for c in text.chars() {
            let id = font.glyph_id(c);
            let glyph = id.with_scale_and_position(scale, ab_glyph::point(x, 0.0));
            x += scaled_font.h_advance(id);
            glyphs.push(glyph);
        }

        if glyphs.is_empty() { return; }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for g in &glyphs {
            if let Some(q) = font.outline_glyph(g.clone()) {
                let b = q.px_bounds();
                min_x = min_x.min(b.min.x);
                min_y = min_y.min(b.min.y);
                max_x = max_x.max(b.max.x);
                max_y = max_y.max(b.max.y);
            }
        }
        
        // Padding
        let w = (max_x - min_x).ceil() as u32 + 8;
        let h = (max_y - min_y).ceil() as u32 + 8;
        if w == 0 || h == 0 { return; }

        let mut pixels = vec![[0u8; 4]; (w * h) as usize];
        for g in glyphs {
            if let Some(q) = font.outline_glyph(g) {
                let b = q.px_bounds();
                q.draw(|x, y, v| {
                    let px = (x as f32 + b.min.x - min_x + 4.0) as u32;
                    let py = (y as f32 + b.min.y - min_y + 4.0) as u32;
                    if px < w && py < h {
                        let i = (py * w + px) as usize;
                        pixels[i] = [255, 255, 255, (v * 255.0) as u8];
                    }
                });
            }
        }

        let cpu_texture = CpuTexture {
            data: TextureData::RgbaU8(pixels),
            width: w,
            height: h,
            ..Default::default()
        };
        self.texture = Some(Texture2DRef::from_texture(Texture2D::new(&self.context, &cpu_texture)));
        self.quad.material.texture = self.texture.clone();
        
        // Position in screen space (bottom-center)
        let screen_w = viewport.width as f32;
        let screen_h = viewport.height as f32;
        let aspect = w as f32 / h as f32;
        let quad_h = 40.0;
        let quad_w = quad_h * aspect;
        
        self.quad.set_transformation(
            Mat4::from_translation(Vec3::new(0.0, -0.85, 0.0))
            * Mat4::from_nonuniform_scale(quad_w / screen_w, quad_h / screen_h, 1.0)
        );
    }

    fn render(&self, camera: &Camera) {
        self.quad.render(camera, &[]);
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let (tx_prompt, rx_prompt) = mpsc::channel::<WorldPrompt>();
    let (tx_result, rx_result) = mpsc::channel::<GenerationResult>();

    // Brain thread
    std::thread::spawn(move || {
        use burn::backend::Wgpu;
        let device: burn::prelude::Device<Wgpu> = Default::default();
        let (brain, tokenizer, config) = match YumonBrain::<Wgpu>::load("checkpoints/brain/128h_2l_2a_64len_6e", &device) {
            Ok(m) => m,
            Err(e) => { eprintln!("[brain] load failed: {e}"); return; }
        };
        while let Ok(p) = rx_prompt.recv() {
            let result = brain.generate_unmasked_parsed(&tokenizer, &p.prompt, config.max_seq_len, &device);
            let _ = tx_result.send(result);
        }
    });

    let window = Window::new(WindowSettings {
        title: "Yumon World".into(),
        max_size: Some((1440, 900)),
        ..Default::default()
    }).expect("Failed to open window");

    let context = window.gl();
    let mut camera = Camera::new_perspective(
        window.viewport(),
        Vec3::new(0.0, 1.7, 0.0),
        Vec3::new(0.0, 1.7, 1.0),
        Vec3::unit_y(),
        degrees(75.0),
        0.1,
        200.0,
    );

    let mut fps_state = FpsState::new();
    let mut player    = Player::new();
    let mut yumon     = Yumon::new();
    let mut keys_held = std::collections::HashSet::<Key>::new();
    let mut command_mode = false;
    let mut command_input = String::new();

    let mut gilrs = Gilrs::new().expect("Failed to init gilrs");
    let mut active_gamepad = None;

    let ambient     = AmbientLight::new(&context, 0.45, Srgba::WHITE);
    let directional = DirectionalLight::new(&context, 1.3, Srgba::WHITE, &Vec3::new(-1.0, -2.5, -1.5));
    
    let terrain_cpu = build_terrain_mesh();
    let terrain_gm = Gm::new(Mesh::new(&context, &terrain_cpu), PhysicalMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(60, 90, 50, 255), roughness: 0.90, metallic: 0.0, ..Default::default()
    }));

    // Water Quad
    let water_size = ARENA * 1.5;
    let water_positions: Vec<Vec3> = vec![
        Vec3::new(-water_size, 0.0, -water_size), Vec3::new( water_size, 0.0, -water_size),
        Vec3::new( water_size, 0.0,  water_size), Vec3::new(-water_size, 0.0,  water_size),
    ];
    let water_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
    let water_program = Program::from_source(&context, 
        "in vec3 a_position; out vec3 v_world_pos; uniform mat4 viewProjection; void main() { v_world_pos = a_position; gl_Position = viewProjection * vec4(a_position, 1.0); }",
        WATER_FRAG_GLSL
    ).expect("Water shader failed");
    let water_vbo = VertexBuffer::new_with_data(&context, &water_positions);
    let water_ibo = ElementBuffer::new_with_data(&context, &water_indices.to_vec());

    // Yumon Model
    let mut gpu_model: Option<Model<PhysicalMaterial>> = {
        let p = std::path::Path::new(MODEL_PATH);
        three_d_asset::io::load(&[p]).ok().and_then(|mut loaded| {
            loaded.deserialize(p.file_name().unwrap().to_str().unwrap()).ok().and_then(|cpu: CpuModel| {
                Model::<PhysicalMaterial>::new(&context, &cpu).ok()
            })
        })
    };
    let mut fallback_sphere = Gm::new(Mesh::new(&context, &CpuMesh::sphere(20)), PhysicalMaterial::new_opaque(&context, &CpuMaterial {
        albedo: Srgba::new(180, 120, 200, 255), roughness: 0.55, metallic: 0.1, ..Default::default()
    }));

    // UI Text
    let mut text_renderer = TextRenderer::new(&context);
    let ui_camera = Camera::new_orthographic(window.viewport(), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 2.0, 0.0, 10.0);

    let mut last_frame = Instant::now();
    let mut dest_pos = DEST_INIT;

    window.render_loop(move |mut frame_input| {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32().min(0.1);
        last_frame = now;
        let t = frame_input.accumulated_time;

        // ── Brain ─────────────────────────────────────────────────────────────
        if let Ok(result) = rx_result.try_recv() {
            yumon.apply_result(result, player.pos, dest_pos);
        }
        if !yumon.waiting_for_brain {
            yumon.interval_timer -= dt;
            if yumon.interval_timer <= 0.0 {
                yumon.waiting_for_brain = true;
                let _ = tx_prompt.send(WorldPrompt { prompt: yumon.build_prompt() });
            }
        }

        // ── Input ─────────────────────────────────────────────────────────────
        while let Some(gilrs_event) = gilrs.next_event() {
            active_gamepad = Some(gilrs_event.id);
            match gilrs_event.event {
                gilrs::EventType::ButtonPressed(Button::North, _) => {
                    command_mode = !command_mode;
                    if !command_mode { command_input.clear(); }
                }
                gilrs::EventType::ButtonPressed(Button::South, _) if command_mode => {
                    if !command_input.is_empty() {
                        yumon.current_command = command_input.clone();
                        command_input.clear();
                        command_mode = false;
                        yumon.interval_timer = 0.0;
                    }
                }
                gilrs::EventType::ButtonPressed(Button::East, _) if command_mode => {
                    command_mode = false;
                    command_input.clear();
                }
                _ => {}
            }
        }

        for event in &frame_input.events {
            match event {
                three_d::Event::MouseMotion { delta, .. } if !command_mode => {
                    fps_state.apply_mouse(delta.0 as f32, delta.1 as f32);
                }
                three_d::Event::KeyPress { kind, .. } => {
                    keys_held.insert(*kind);
                    if *kind == Key::Enter && command_mode {
                        if !command_input.is_empty() {
                            yumon.current_command = command_input.clone();
                            command_input.clear();
                            command_mode = false;
                            yumon.interval_timer = 0.0;
                        }
                    }
                    if *kind == Key::Escape {
                        if command_mode { command_mode = false; command_input.clear(); }
                    }
                }
                three_d::Event::KeyRelease { kind, .. } => { keys_held.remove(kind); }
                three_d::Event::Text(text) if command_mode => {
                    command_input.push_str(text);
                }
                _ => {}
            }
        }

        // Gamepad sticks
        if let Some(id) = active_gamepad {
            let gamepad = gilrs.gamepad(id);
            if !command_mode {
                let lx = gamepad.value(Axis::LeftStickX);
                let ly = gamepad.value(Axis::LeftStickY);
                let rx = gamepad.value(Axis::RightStickX);
                let ry = gamepad.value(Axis::RightStickY);
                
                let fwd = fps_state.forward_xz();
                let right = -fps_state.right_xz();
                player.vel = (fwd * ly + right * lx) * MOVE_SPEED;
                fps_state.apply_gamepad(rx, -ry, dt);
            } else {
                player.vel = Vec3::zero();
            }
        } else {
            // Keyboard fallback
            let mut move_vel = Vec3::zero();
            if keys_held.contains(&Key::W) { move_vel += fps_state.forward_xz(); }
            if keys_held.contains(&Key::S) { move_vel -= fps_state.forward_xz(); }
            if keys_held.contains(&Key::A) { move_vel -= fps_state.right_xz(); }
            if keys_held.contains(&Key::D) { move_vel += fps_state.right_xz(); }
            player.vel = if move_vel.magnitude() > 0.001 { move_vel.normalize() * MOVE_SPEED } else { Vec3::zero() };
        }

        // ── Tick ──────────────────────────────────────────────────────────────
        player.tick(dt, &terrain_cpu.heights().unwrap());
        yumon.tick(dt, player.pos, &terrain_cpu.heights().unwrap());

        // Update camera
        let eye = Vec3::new(player.pos.x, player.pos.y + EYE_HEIGHT, player.pos.z);
        camera.set_view(eye, eye + fps_state.look_dir(), Vec3::unit_y());
        camera.set_viewport(frame_input.viewport);

        // ── UI Update ─────────────────────────────────────────────────────────
        let display_text = if command_mode {
            format!("Command: {}", command_input)
        } else if yumon.speech_timer > 0.0 {
            format!("Yumon: {}", yumon.speech)
        } else {
            String::new()
        };
        text_renderer.update(&display_text, frame_input.viewport);

        // ── Draw ──────────────────────────────────────────────────────────────
        let lights: [&dyn Light; 2] = [&ambient, &directional];
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.10, 0.12, 0.15, 1.0, 1.0));

        screen.render(&camera, [&terrain_gm as &dyn Object], &lights);
        
        let yumon_xform = yumon.world_transform(t);
        if let Some(ref mut model) = gpu_model {
            for p in model.iter_mut() {
                let orig = p.transformation();
                p.set_transformation(yumon_xform * orig);
                screen.render(&camera, [p as &dyn Object], &lights);
                p.set_transformation(orig);
            }
        } else {
            fallback_sphere.set_transformation(yumon_xform);
            screen.render(&camera, [&fallback_sphere as &dyn Object], &lights);
        }

        // Water
        screen.write::<RendererError>(|| {
            let gl = context.as_ref();
            unsafe { gl.enable(three_d::context::BLEND); gl.blend_func(three_d::context::SRC_ALPHA, three_d::context::ONE_MINUS_SRC_ALPHA); }
            water_program.use_uniform("viewProjection", camera.projection() * camera.view());
            water_program.use_uniform("uTimeRes", Vec4::new(t as f32 * 0.001, 0.0, 0.0, 0.0));
            water_program.use_uniform("uEye", *camera.position());
            water_program.use_vertex_attribute("a_position", &water_vbo);
            water_program.draw_elements(RenderStates { blend: Blend::TRANSPARENCY, depth_test: DepthTest::LessOrEqual, ..Default::default() }, frame_input.viewport, &water_ibo);
            unsafe { gl.disable(three_d::context::BLEND); }
            Ok(())
        }).unwrap();

        // UI pass
        screen.render(&ui_camera, [&text_renderer.quad as &dyn Object], &[]);

        FrameOutput::default()
    });
}

trait CpuMeshExt {
    fn heights(&self) -> Option<Vec<f32>>;
}
impl CpuMeshExt for CpuMesh {
    fn heights(&self) -> Option<Vec<f32>> {
        if let Positions::F32(ref pos) = self.positions {
            Some(pos.iter().map(|p| p.y).collect())
        } else { None }
    }
}
