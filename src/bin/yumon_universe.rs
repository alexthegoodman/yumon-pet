//! A procedural suburb driven by Language-stage replies.
#![recursion_limit = "256"]

#[cfg(all(target_os = "windows", feature = "desktop"))]
#[path = "universe_court/mod.rs"]
mod court;

#[cfg(all(target_os = "windows", feature = "desktop"))]
mod desktop {
    use clap::{Parser, ValueEnum};
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };
    use three_d::{
        egui::{self, Color32, RichText},
        *,
    };
    use yumon_pet::{
        brain::{
            bpe::TokenizerKind, model::YumonBrain, moe_model::YumonMoeBrain,
            samples::TrainingStage, xlstm_model::YumonXLstmBrain,
        },
        kingdom::{COLORS, NAMES, ProjectKind, Realm},
        universe::{
            Behavior, Decision, EXTENT, Kind, Neighborhood, Point, Question, Theme, infer_reply,
        },
    };

    #[derive(Clone, Copy, ValueEnum)]
    enum Architecture {
        Moe,
        Xlstm,
        EncoderDecoder,
    }
    #[derive(Parser)]
    #[command(about = "Explore a procedural suburb with a Language-stage Yumon checkpoint")]
    pub struct Args {
        #[arg(
            long,
            default_value = "checkpoints/brain/256h_16l_4a_32len_b8_Moe_e4_k1_Language"
        )]
        checkpoint: String,
        #[arg(long, value_enum, default_value = "moe")]
        architecture: Architecture,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, value_enum, default_value = "suburban")]
        theme: Theme,
    }
    enum BrainEvent {
        Ready,
        Error(String),
        Reply(usize, Result<String, String>),
    }
    struct Request {
        id: usize,
        text: String,
    }

    fn brain_worker(args: Args, rx: mpsc::Receiver<Request>, tx: mpsc::Sender<BrainEvent>) {
        // Validate metadata before GPU initialization, with a useful visible error.
        let run = || -> anyhow::Result<()> {
            let metadata: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
                std::path::Path::new(&args.checkpoint).join("metadata.json"),
            )?)?;
            anyhow::ensure!(
                metadata["training_stage"] == "Language",
                "Universe requires a Language checkpoint; this checkpoint is {:?}",
                metadata["training_stage"]
            );
            use burn::backend::Wgpu;
            let device = Default::default();
            // All architectures receive plain messages. Only raw generated text drives actions.
            macro_rules! serve {
                ($model:ty, $generate:expr) => {{
                    let (brain, tokenizer, config) = <$model>::load(&args.checkpoint, &device)?;
                    anyhow::ensure!(config.training_stage == TrainingStage::Language, "Expected Language stage");
                    let _ = tx.send(BrainEvent::Ready);
                    while let Ok(request) = rx.recv() {
                        let count = tokenizer.encode(&request.text).len();
                        // Reserve BOS, separator and room for a meaningful short reply.
                        let result = if count + tokenizer.encode(" ").len() + 9 > config.max_seq_len {
                            Err(format!("Message needs {} tokens plus reply space; checkpoint context is {}. Use a shorter message or a longer-context Language checkpoint.", count, config.max_seq_len))
                        } else {
                            let generate: fn(& $model, &TokenizerKind, &str, usize, &_) -> String = $generate;
                            Ok(generate(&brain, &tokenizer, &request.text, config.max_seq_len.min(48), &device))
                        };
                        if tx.send(BrainEvent::Reply(request.id, result)).is_err() { break; }
                    }
                }};
            }
            match args.architecture {
                Architecture::Moe => serve!(YumonMoeBrain<Wgpu>, |b, t, p, n, d| b
                    .generate_unmasked_parsed(t, p, n, d)
                    .raw_output),
                Architecture::Xlstm => serve!(YumonXLstmBrain<Wgpu>, |b, t, p, n, d| b
                    .generate_unmasked_parsed(t, p, n, d)
                    .raw_output),
                Architecture::EncoderDecoder => serve!(YumonBrain<Wgpu>, |b, t, p, n, d| b
                    .generate_unmasked_parsed::<cubecl::wgpu::WgpuRuntime>(t, p, n, d)
                    .raw_output),
            }
            Ok(())
        };
        if let Err(e) = run() {
            let _ = tx.send(BrainEvent::Error(format!(
                "Could not load {}: {e}",
                args.checkpoint
            )));
        }
    }

    struct Yumon {
        pos: Point,
        target: Point,
        decision: Decision,
        question: Option<Question>,
        next: Instant,
        waiting: bool,
        turn: usize,
        log: Vec<String>,
        input: String,
        arrived: bool,
        facing: f32,
        travel_seconds: f32,
    }
    impl Yumon {
        fn log(&mut self, text: String) {
            self.log.push(text);
            if self.log.len() > 8 {
                self.log.remove(0);
            }
        }
        fn apply(&mut self, reply: String, world: &Neighborhood) {
            let Some(question) = self.question.take() else {
                return;
            };
            self.decision = infer_reply(&reply, &question, world, self.pos);
            self.log(format!("Yumon: {reply}"));
            self.log(format!(
                "Action: {}",
                self.decision.behavior.label(world.theme)
            ));
            self.target = if let Some(id) = self.decision.place {
                world.approach(self.pos, id)
            } else if self.decision.behavior == Behavior::Explore {
                let angle = self.turn as f32 * 2.4;
                Point {
                    x: (self.pos.x + angle.cos() * 5.0).clamp(-28.0, 28.0),
                    z: (self.pos.z + angle.sin() * 5.0).clamp(-28.0, 28.0),
                }
            } else {
                self.pos
            };
            self.arrived = false;
            self.travel_seconds = 0.0;
        }
        fn tick(&mut self, dt: f32, world: &mut Neighborhood) {
            let distance = self.pos.distance(self.target);
            if distance > 0.08 {
                self.travel_seconds += dt;
                if self.travel_seconds > 15.0 {
                    self.target = self.pos;
                    self.arrived = true;
                    self.log("Could not reach that place; looking for another activity.".into());
                    return;
                }
                let dx = (self.target.x - self.pos.x) / distance;
                let dz = (self.target.z - self.pos.z) / distance;
                self.facing = dx.atan2(dz);
                let step = (dt * 2.4).min(distance);
                let proposed = Point {
                    x: self.pos.x + dx * step,
                    z: self.pos.z + dz * step,
                };
                if world.walkable(proposed) {
                    self.pos = proposed;
                } else {
                    // Slide along obstacles; stop if neither axis is safe.
                    let x_only = Point {
                        x: proposed.x,
                        z: self.pos.z,
                    };
                    let z_only = Point {
                        x: self.pos.x,
                        z: proposed.z,
                    };
                    if dx.abs() > 0.01 && world.walkable(x_only) {
                        self.pos = x_only;
                    } else if dz.abs() > 0.01 && world.walkable(z_only) {
                        self.pos = z_only;
                    } else {
                        self.target = self.pos;
                        self.arrived = true;
                        self.log("Path blocked; waiting for another decision.".into());
                    }
                }
            } else if !self.arrived {
                self.arrived = true;
                if let Some(id) = self.decision.place {
                    let theme = world.theme;
                    let place = &mut world.places[id];
                    if self.pos.distance(place.pos) <= place.kind.radius() + 1.0 {
                        if !matches!(place.kind, Kind::Business | Kind::Church | Kind::Telescope)
                            || self.decision.behavior == Behavior::Build
                        {
                            place.activity_count += 1;
                        }
                        // For a buildable landmark, activity_count doubles as
                        // its build tier - "level" reads right there, "visit"
                        // doesn't.
                        let counter = if matches!(
                            place.kind,
                            Kind::Business | Kind::Church | Kind::Telescope
                        ) {
                            "level"
                        } else {
                            "visit"
                        };
                        self.log(format!(
                            "{} by the {} ({} {}).",
                            self.decision.behavior.label(theme),
                            place.kind.noun(theme),
                            counter,
                            place.activity_count
                        ));
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn resident(pos: Point) -> Yumon {
            Yumon {
                pos,
                target: pos,
                decision: Decision {
                    behavior: Behavior::Rest,
                    place: None,
                },
                question: None,
                next: Instant::now(),
                waiting: false,
                turn: 0,
                log: vec![],
                input: String::new(),
                arrived: true,
                facing: 0.0,
                travel_seconds: 0.0,
            }
        }

        #[test]
        fn affirmative_reply_moves_to_ball_and_interacts_once() {
            let mut world = Neighborhood::generate(42, Theme::Suburban);
            let id = world
                .places
                .iter()
                .position(|p| p.kind == Kind::Ball)
                .unwrap();
            let pos = world.approach(Point { x: 0.0, z: 0.0 }, id);
            let center = world.places[id].pos;
            let start = Point {
                x: pos.x + (pos.x - center.x) * 0.5,
                z: pos.z + (pos.z - center.z) * 0.5,
            };
            let mut y = resident(start);
            y.question = Some(Question {
                text: "Play with the ball?".into(),
                place: Some(id),
                offered: Behavior::Play,
            });
            y.apply("Yes, let's play!".into(), &world);
            for _ in 0..100 {
                y.tick(0.05, &mut world);
            }
            assert_eq!(y.decision.behavior, Behavior::Play);
            assert_eq!(world.places[id].activity_count, 1);
            assert!(y.pos.distance(center) < start.distance(center));
            assert!(y.arrived);
        }

        #[test]
        fn repeated_build_replies_raise_a_tower_tier() {
            // The whole point of the build mechanic: a Yumon offered the same
            // landmark again and again keeps raising its tier (activity_count),
            // which `grow()` in the render layer reads to make it visibly
            // taller. This only exercises the decision/tick logic - `grow`
            // itself needs a GL context, so it's covered separately by hand
            // (three-d has no headless render target for a unit test here).
            let mut world = Neighborhood::generate(9, Theme::Suburban);
            let id = world
                .places
                .iter()
                .position(|p| p.kind == Kind::Business)
                .unwrap();
            let pos = world.approach(Point { x: 0.0, z: 0.0 }, id);
            let mut y = resident(pos);
            for turn in 0..3 {
                y.question = Some(Question {
                    text: "Build the tower taller?".into(),
                    place: Some(id),
                    offered: Behavior::Build,
                });
                y.turn = turn;
                y.apply("Yes, let's build it!".into(), &world);
                for _ in 0..50 {
                    y.tick(0.05, &mut world);
                }
            }
            assert_eq!(y.decision.behavior, Behavior::Build);
            assert_eq!(world.places[id].activity_count, 3);
        }

        #[test]
        fn stopped_simulation_freezes_citizens_economy_and_construction() {
            let mut world = Neighborhood::generate(42, Theme::Suburban);
            let mut realm = Realm::new(42);
            realm.fund(0, 0).unwrap();
            let mut citizens = vec![resident(Point { x: 0.0, z: 0.0 })];
            let treasury = realm.kingdoms[0].treasury;
            tick_simulation(&mut world, &mut realm, &mut citizens, 50.0, false);
            assert_eq!(realm.season, 1);
            assert_eq!(realm.kingdoms[0].treasury, treasury);
            assert_eq!(realm.projects[0].remaining, 2);
            assert_eq!(citizens[0].pos, Point { x: 0.0, z: 0.0 });
            assert_eq!(citizens[0].decision.behavior, Behavior::Rest);

            tick_simulation(&mut world, &mut realm, &mut citizens, 0.1, true);
            assert_eq!(citizens[0].decision.behavior, Behavior::Build);
            let position = citizens[0].pos;
            let elapsed = realm.elapsed;
            // Also covers a worker disconnect or a user pause after play began.
            tick_simulation(&mut world, &mut realm, &mut citizens, 50.0, false);
            assert_eq!(citizens[0].pos, position);
            assert_eq!(realm.elapsed, elapsed);
            assert_eq!(realm.projects[0].remaining, 2);
            tick_simulation(&mut world, &mut realm, &mut citizens, 50.0, true);
            assert!(realm.projects[0].completed);
        }

        #[test]
        fn movement_stays_outside_solid_objects_and_eventually_stops() {
            let mut world = Neighborhood::generate(42, Theme::Suburban);
            let mut y = resident(Point { x: 0.0, z: 0.0 });
            y.target = world.places[0].pos;
            y.arrived = false;
            for _ in 0..400 {
                y.tick(0.05, &mut world);
                assert!(world.walkable(y.pos));
            }
            assert!(y.arrived);
            assert_eq!(world.places[0].activity_count, 0);
        }
    }

    fn shape(
        context: &Context,
        sphere: bool,
        pos: [f32; 3],
        size: [f32; 3],
        color: [u8; 3],
    ) -> Gm<Mesh, PhysicalMaterial> {
        let cpu = if sphere {
            CpuMesh::sphere(10)
        } else {
            CpuMesh::cube()
        };
        let mut gm = Gm::new(
            Mesh::new(context, &cpu),
            PhysicalMaterial::new_opaque(
                context,
                &CpuMaterial {
                    albedo: Srgba::new(color[0], color[1], color[2], 255),
                    roughness: 0.9,
                    ..Default::default()
                },
            ),
        );
        gm.set_transformation(
            Mat4::from_translation(vec3(pos[0], pos[1], pos[2]))
                * Mat4::from_nonuniform_scale(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0),
        );
        gm
    }

    /// A cylinder/cone mesh with no transformation set yet - `mount_transform`
    /// positions it every time a landmark grows. Kept separate from `shape` so
    /// creating a landmark's meshes (once, at startup) never has to be redone
    /// just to move them.
    fn cyl_shape(context: &Context, cone: bool, color: [u8; 3]) -> Gm<Mesh, PhysicalMaterial> {
        let cpu = if cone {
            CpuMesh::cone(10)
        } else {
            CpuMesh::cylinder(10)
        };
        Gm::new(
            Mesh::new(context, &cpu),
            PhysicalMaterial::new_opaque(
                context,
                &CpuMaterial {
                    albedo: Srgba::new(color[0], color[1], color[2], 255),
                    roughness: 0.85,
                    ..Default::default()
                },
            ),
        )
    }

    /// A cone/cylinder's own local axis is +x, running length `[0, 1]` with a
    /// radius-1 cross-section. `angle_deg` of 90 stands it straight up from
    /// `base` (its pivot/ground end, not its center); other angles mount it at
    /// an elevation, e.g. a telescope barrel.
    fn mount_transform(base: [f32; 3], angle_deg: f32, length: f32, radius: f32) -> Mat4 {
        Mat4::from_translation(vec3(base[0], base[1], base[2]))
            * Mat4::from_angle_z(degrees(angle_deg))
            * Mat4::from_nonuniform_scale(length.max(0.01), radius, radius)
    }

    /// Caps how tall a landmark keeps visibly growing. Its build tier
    /// (`Place::activity_count`) itself is never capped - the counter just
    /// stops changing the render past this point, the same way a real skyline
    /// eventually plateaus.
    const MAX_BUILD_TIER: u32 = 60;

    /// A buildable landmark's fixed mesh set, repositioned in place as it
    /// grows rather than rebuilt - so a simulation left running for a long
    /// time never allocates new GPU buffers just because a tower got taller.
    struct Landmark {
        place: usize,
        kind: Kind,
        pos: Point,
        meshes: Vec<Gm<Mesh, PhysicalMaterial>>,
    }

    fn landmarks_for(context: &Context, world: &Neighborhood) -> Vec<Landmark> {
        world
            .places
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.kind, Kind::Business | Kind::Church | Kind::Telescope))
            .map(|(place, p)| {
                let meshes = match p.kind {
                    Kind::Business => vec![
                        shape(context, false, [0.0; 3], [1.0; 3], p.tint), // shaft
                        shape(context, false, [0.0; 3], [1.0; 3], [90, 92, 98]), // roof cap
                        shape(context, false, [0.0; 3], [1.0; 3], [60, 62, 68]), // antenna
                        shape(context, false, [0.0; 3], [1.0; 3], [157, 205, 225]), // window band
                    ],
                    Kind::Church => vec![
                        shape(context, false, [0.0; 3], [1.0; 3], p.tint), // nave
                        shape(context, false, [0.0; 3], [1.0; 3], [110, 74, 65]), // roof
                        cyl_shape(context, true, [214, 205, 190]),         // spire
                        shape(context, false, [0.0; 3], [1.0; 3], [222, 188, 92]), // finial
                    ],
                    Kind::Telescope => vec![
                        cyl_shape(context, false, [168, 168, 163]), // drum
                        shape(context, true, [0.0; 3], [1.0; 3], [200, 200, 195]), // dome
                        cyl_shape(context, false, [66, 133, 199]),  // barrel
                    ],
                    _ => unreachable!("filtered to buildable kinds above"),
                };
                let mut landmark = Landmark {
                    place,
                    kind: p.kind,
                    pos: p.pos,
                    meshes,
                };
                grow(&mut landmark, 0);
                landmark
            })
            .collect()
    }

    /// Recomputes a landmark's mesh transforms from its current build tier.
    /// Mesh count per kind never changes - only their size/position does.
    fn grow(landmark: &mut Landmark, tier: u32) {
        let t = tier.min(MAX_BUILD_TIER) as f32;
        let Point { x, z } = landmark.pos;
        match landmark.kind {
            Kind::Business => {
                let height = 3.0 + t * 0.85;
                landmark.meshes[0].set_transformation(
                    Mat4::from_translation(vec3(x, height / 2.0, z))
                        * Mat4::from_nonuniform_scale(1.6, height / 2.0, 1.6),
                );
                landmark.meshes[1].set_transformation(
                    Mat4::from_translation(vec3(x, height + 0.15, z))
                        * Mat4::from_nonuniform_scale(1.8, 0.15, 1.8),
                );
                let antenna = 0.8 + t * 0.04;
                landmark.meshes[2].set_transformation(
                    Mat4::from_translation(vec3(x, height + 0.3 + antenna / 2.0, z))
                        * Mat4::from_nonuniform_scale(0.08, antenna / 2.0, 0.08),
                );
                landmark.meshes[3].set_transformation(
                    Mat4::from_translation(vec3(x, height * 0.6, z + 1.62))
                        * Mat4::from_nonuniform_scale(1.5, 0.25, 0.04),
                );
            }
            Kind::Church => {
                let nave = 3.0;
                landmark.meshes[0].set_transformation(
                    Mat4::from_translation(vec3(x, nave / 2.0, z))
                        * Mat4::from_nonuniform_scale(2.0, nave / 2.0, 1.8),
                );
                landmark.meshes[1].set_transformation(
                    Mat4::from_translation(vec3(x, nave + 0.2, z))
                        * Mat4::from_nonuniform_scale(2.2, 0.2, 2.0),
                );
                let spire = 1.5 + t * 0.5;
                landmark.meshes[2].set_transformation(mount_transform(
                    [x, nave + 0.4, z],
                    90.0,
                    spire,
                    0.85,
                ));
                landmark.meshes[3].set_transformation(
                    Mat4::from_translation(vec3(x, nave + 0.4 + spire + 0.1, z))
                        * Mat4::from_nonuniform_scale(0.1, 0.3, 0.1),
                );
            }
            Kind::Telescope => {
                landmark.meshes[0].set_transformation(mount_transform([x, 0.0, z], 90.0, 1.6, 1.1));
                landmark.meshes[1].set_transformation(
                    Mat4::from_translation(vec3(x, 1.6 + 1.1, z))
                        * Mat4::from_nonuniform_scale(1.1, 1.1, 1.1),
                );
                let barrel = 1.2 + t * 0.22;
                landmark.meshes[2].set_transformation(mount_transform(
                    [x, 1.9, z + 0.5],
                    55.0,
                    barrel,
                    0.22,
                ));
            }
            _ => {}
        }
    }

    fn scenery(context: &Context, world: &Neighborhood) -> Vec<Gm<Mesh, PhysicalMaterial>> {
        let mut meshes = vec![];
        let mut add =
            |sphere, pos, size, color| meshes.push(shape(context, sphere, pos, size, color));
        let ground = match world.theme {
            Theme::Suburban => [115, 163, 91],
            Theme::Urban => [168, 168, 163],
        };
        add(
            false,
            [0.0, -0.15, 0.0],
            [EXTENT * 2.0, 0.25, EXTENT * 2.0],
            ground,
        );
        for axis in [-15.0, 0.0, 15.0] {
            add(false, [axis, 0.01, 0.0], [5.0, 0.05, 60.0], [192, 190, 178]);
            add(false, [0.0, 0.01, axis], [60.0, 0.05, 5.0], [192, 190, 178]);
            add(false, [axis, 0.05, 0.0], [3.2, 0.05, 60.0], [72, 79, 88]);
            add(false, [0.0, 0.05, axis], [60.0, 0.05, 3.2], [72, 79, 88]);
            for n in -9..10 {
                let v = n as f32 * 3.0;
                add(false, [axis, 0.085, v], [0.08, 0.015, 1.0], [240, 227, 173]);
                add(false, [v, 0.085, axis], [1.0, 0.015, 0.08], [240, 227, 173]);
            }
        }
        for p in &world.places {
            let x = p.pos.x;
            let z = p.pos.z;
            match (p.kind, world.theme) {
                (Kind::House, Theme::Suburban) | (Kind::Shop, Theme::Suburban) => {
                    let height = if p.kind == Kind::Shop { 2.2 } else { 2.8 };
                    add(false, [x, height / 2.0, z], [3.6, height, 3.3], p.tint);
                    add(false, [x, height + 0.25, z], [4.1, 0.5, 3.8], [110, 74, 65]);
                    add(false, [x, 0.65, z + 1.67], [0.65, 1.3, 0.1], [91, 66, 49]);
                    for side in [-1.1, 1.1] {
                        add(
                            false,
                            [x + side, 1.55, z + 1.68],
                            [0.7, 0.8, 0.1],
                            [157, 215, 230],
                        );
                    }
                    if p.kind == Kind::Shop {
                        add(false, [x, 2.0, z + 1.9], [3.8, 0.25, 0.8], [227, 157, 78]);
                    } else {
                        add(
                            false,
                            [x + 1.0, height + 0.6, z - 0.7],
                            [0.45, 1.0, 0.5],
                            [144, 85, 66],
                        );
                    }
                }
                // Market: the same stall silhouette as the suburban shop (an
                // awning already reads as a market stall) with a cooler tone.
                (Kind::Shop, Theme::Urban) => {
                    add(false, [x, 1.1, z], [3.6, 2.2, 3.3], p.tint);
                    add(false, [x, 2.35, z], [4.1, 0.5, 3.8], [90, 92, 98]);
                    add(false, [x, 0.65, z + 1.67], [0.65, 1.3, 0.1], [70, 72, 78]);
                    for side in [-1.1, 1.1] {
                        add(
                            false,
                            [x + side, 1.55, z + 1.68],
                            [0.7, 0.8, 0.1],
                            [157, 215, 230],
                        );
                    }
                    add(false, [x, 2.0, z + 1.9], [3.8, 0.25, 0.8], [201, 62, 58]);
                }
                // Business: a flat-roofed glass tower - a rooftop utility unit
                // and window bands stand in for the house's pitched roof/chimney.
                (Kind::House, Theme::Urban) => {
                    let height = 4.4;
                    add(false, [x, height / 2.0, z], [3.6, height, 3.3], p.tint);
                    add(false, [x, height + 0.15, z], [3.8, 0.3, 3.5], [70, 72, 78]);
                    for band in [1.2, 2.2, 3.2] {
                        add(
                            false,
                            [x, band, z + 1.66],
                            [3.2, 0.4, 0.08],
                            [157, 205, 225],
                        );
                    }
                    add(
                        false,
                        [x + 0.9, height + 0.55, z - 0.6],
                        [0.7, 0.5, 0.7],
                        [60, 62, 68],
                    );
                }
                (Kind::Tree, Theme::Suburban) => {
                    add(false, [x, 0.85, z], [0.4, 1.7, 0.4], [112, 81, 51]);
                    add(true, [x, 2.1, z], [2.1, 2.4, 2.1], [59, 119, 62]);
                }
                // Debt: a notice board on a post instead of a tree.
                (Kind::Tree, Theme::Urban) => {
                    add(false, [x, 0.85, z], [0.15, 1.7, 0.15], [90, 92, 98]);
                    add(false, [x, 1.85, z], [1.2, 0.8, 0.08], [196, 74, 68]);
                }
                (Kind::Bench, Theme::Suburban) => {
                    add(false, [x, 0.45, z], [1.4, 0.15, 0.6], [153, 105, 62]);
                    add(false, [x, 0.8, z - 0.25], [1.4, 0.65, 0.12], [153, 105, 62]);
                    for side in [-0.5, 0.5] {
                        add(false, [x + side, 0.2, z], [0.1, 0.4, 0.5], [65, 70, 74]);
                    }
                }
                // Coffee: the bench's own counter shape, recolored, with a cup on top.
                (Kind::Bench, Theme::Urban) => {
                    add(false, [x, 0.45, z], [1.4, 0.15, 0.6], [80, 58, 45]);
                    add(
                        false,
                        [x, 0.8, z - 0.25],
                        [1.4, 0.65, 0.12],
                        [214, 205, 190],
                    );
                    for side in [-0.5, 0.5] {
                        add(false, [x + side, 0.2, z], [0.1, 0.4, 0.5], [65, 70, 74]);
                    }
                    add(
                        true,
                        [x, 0.58, z + 0.15],
                        [0.22, 0.22, 0.22],
                        [235, 235, 232],
                    );
                }
                (Kind::Ball, Theme::Suburban) => {
                    add(true, [x, 0.3, z], [0.6, 0.6, 0.6], [241, 121, 81])
                }
                // Product: a crate on display with a label stripe instead of a ball.
                (Kind::Ball, Theme::Urban) => {
                    add(false, [x, 0.3, z], [0.6, 0.6, 0.6], [214, 178, 128]);
                    add(false, [x, 0.3, z], [0.62, 0.14, 0.62], [66, 133, 199]);
                }
                (Kind::Garden, Theme::Suburban) => {
                    add(false, [x, 0.09, z], [0.9, 0.18, 0.9], [106, 77, 49]);
                    for offset in [-0.25, 0.0, 0.25] {
                        add(
                            true,
                            [x + offset, 0.22, z],
                            [0.2, 0.35, 0.65],
                            [85, 155, 62],
                        );
                    }
                }
                // Budget: a small ascending bar chart instead of a garden bed.
                (Kind::Garden, Theme::Urban) => {
                    add(false, [x, 0.05, z], [0.9, 0.1, 0.9], [70, 72, 78]);
                    for (i, height) in [0.25, 0.45, 0.7].into_iter().enumerate() {
                        let offset = (i as f32 - 1.0) * 0.28;
                        add(
                            false,
                            [x + offset, height / 2.0, z],
                            [0.2, height, 0.2],
                            [66, 133, 199],
                        );
                    }
                }
                (Kind::Flowers, Theme::Suburban) => {
                    for (i, color) in [[239, 121, 157], [248, 203, 79], [168, 134, 210]]
                        .into_iter()
                        .enumerate()
                    {
                        add(
                            true,
                            [x + (i as f32 - 1.0) * 0.23, 0.25, z],
                            [0.25, 0.3, 0.3],
                            color,
                        );
                    }
                }
                // Money: stacked bills topped with a coin instead of flowers.
                (Kind::Flowers, Theme::Urban) => {
                    add(false, [x - 0.15, 0.08, z], [0.4, 0.05, 0.25], [76, 140, 90]);
                    add(
                        false,
                        [x + 0.05, 0.13, z + 0.05],
                        [0.4, 0.05, 0.25],
                        [90, 158, 104],
                    );
                    add(true, [x, 0.24, z], [0.18, 0.06, 0.18], [222, 188, 92]);
                }
                (Kind::Mailbox, Theme::Suburban) => {
                    add(false, [x, 0.5, z], [0.12, 1.0, 0.12], [96, 85, 73]);
                    add(false, [x, 1.05, z], [0.6, 0.4, 0.45], [73, 115, 155]);
                }
                // Deal: a signing podium with a folder instead of a mailbox.
                (Kind::Mailbox, Theme::Urban) => {
                    add(false, [x, 0.5, z], [0.16, 1.0, 0.16], [90, 92, 98]);
                    add(false, [x, 1.05, z], [0.6, 0.08, 0.45], [201, 178, 122]);
                }
                (Kind::Pond, Theme::Suburban) => {
                    add(true, [x, 0.02, z], [3.2, 0.15, 3.2], [88, 165, 184]);
                }
                // Brand: a plaza medallion instead of a pond.
                (Kind::Pond, Theme::Urban) => {
                    add(true, [x, 0.02, z], [3.2, 0.12, 3.2], [200, 200, 195]);
                    add(true, [x, 0.05, z], [1.8, 0.1, 1.8], [66, 133, 199]);
                }
                // These grow with a Yumon's build decisions, so they're mutable
                // Landmarks (see `landmarks_for`/`grow`) instead of static scenery.
                (Kind::Business, _) | (Kind::Church, _) | (Kind::Telescope, _) => {}
            }
        }
        meshes
    }

    fn territory_meshes(
        context: &Context,
        realm: &Realm,
        selected: usize,
    ) -> Vec<Gm<Mesh, PhysicalMaterial>> {
        let mut meshes = Vec::new();
        for (id, parcel) in realm.parcels.iter().enumerate() {
            let (x, z) = Realm::center(id);
            let color = if id == selected {
                [245, 245, 220]
            } else {
                parcel.owner.map(|c| COLORS[c]).unwrap_or([110, 120, 125])
            };
            for dz in [-7.4, 7.4] {
                meshes.push(shape(
                    context,
                    false,
                    [x, 0.07, z + dz],
                    [14.8, 0.08, 0.12],
                    color,
                ));
            }
            for dx in [-7.4, 7.4] {
                meshes.push(shape(
                    context,
                    false,
                    [x + dx, 0.07, z],
                    [0.12, 0.08, 14.8],
                    color,
                ));
            }
            if let Some(project) = parcel.project {
                let p = &realm.projects[project];
                let improvements = realm
                    .projects
                    .iter()
                    .filter(|p| p.parcel == Some(id) && p.completed)
                    .count()
                    .saturating_sub(p.completed as usize);
                let height = (if p.completed {
                    2.5
                } else {
                    0.8 + (2 - p.remaining) as f32 * 0.6
                }) + (improvements.min(8) as f32 * 0.4);
                // Public-project models stand beside the existing parcel scenery.
                let (px, pz) = (x + 4.5, z + 4.5);
                match p.kind {
                    ProjectKind::Workshop => {
                        meshes.push(shape(
                            context,
                            false,
                            [px, height / 2.0, pz],
                            [2.5, height, 2.0],
                            color,
                        ));
                        meshes.push(shape(
                            context,
                            false,
                            [px, height + 0.15, pz],
                            [2.9, 0.3, 2.4],
                            [65, 75, 90],
                        ));
                    }
                    ProjectKind::Garden => {
                        meshes.push(shape(
                            context,
                            false,
                            [px, 0.2, pz],
                            [3.0, 0.4, 2.5],
                            [110, 75, 45],
                        ));
                        for offset in [-0.8, 0.0, 0.8] {
                            meshes.push(shape(
                                context,
                                true,
                                [px + offset, height * 0.35, pz],
                                [0.7, height * 0.5, 1.6],
                                [80, 175, 90],
                            ));
                        }
                    }
                    ProjectKind::Observatory => {
                        meshes.push(shape(
                            context,
                            false,
                            [px, height / 2.0, pz],
                            [2.0, height, 2.0],
                            color,
                        ));
                        meshes.push(shape(
                            context,
                            true,
                            [px, height, pz],
                            [2.5, 1.6, 2.5],
                            [210, 225, 240],
                        ));
                    }
                }
            }
        }
        meshes
    }

    fn civic_question(
        world: &Neighborhood,
        realm: &Realm,
        id: usize,
        pos: Point,
        turn: usize,
    ) -> Question {
        let civ = realm.citizens[id].kingdom;
        let mut ids: Vec<_> = world
            .places
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                Realm::parcel_at(p.pos.x, p.pos.z)
                    .is_some_and(|s| realm.parcels[s].owner == Some(civ))
                    && !matches!(p.kind, Kind::Business | Kind::Church | Kind::Telescope)
            })
            .map(|(i, _)| i)
            .collect();
        ids.sort_by(|&a, &b| {
            world.places[a]
                .pos
                .distance(pos)
                .total_cmp(&world.places[b].pos.distance(pos))
        });
        let Some(&place) = ids.get(turn % ids.len().max(1).min(5)) else {
            return world.question(pos, turn);
        };
        let p = &world.places[place];
        let (verb, offered) = p.kind.verb_and_behavior(world.theme);
        Question {
            text: format!(
                "Would you like to {} the {}?",
                verb,
                p.kind.noun(world.theme)
            ),
            place: Some(place),
            offered,
        }
    }

    fn tick_simulation(
        world: &mut Neighborhood,
        realm: &mut Realm,
        yumons: &mut [Yumon],
        dt: f32,
        running: bool,
    ) {
        if !running {
            return;
        }
        realm.advance(dt);
        for (id, y) in yumons.iter_mut().enumerate() {
            // Funded construction is a game rule, never a replacement for a missing brain.
            if y.decision.behavior == Behavior::Build {
                y.decision.behavior = Behavior::Inspect;
            }
            if let Some(project) = realm
                .projects
                .iter()
                .find(|p| p.citizen == id && p.parcel.is_some() && !p.completed)
            {
                let (x, z) = Realm::center(project.parcel.unwrap());
                let target = [
                    Point {
                        x: x + 4.5,
                        z: z + 2.5,
                    },
                    Point {
                        x: x + 2.5,
                        z: z + 4.5,
                    },
                    Point {
                        x: x + 6.0,
                        z: z + 4.5,
                    },
                ]
                .into_iter()
                .find(|&p| world.walkable(p))
                .unwrap_or(y.pos);
                if y.target.distance(target) > 0.1 {
                    y.target = target;
                    y.arrived = false;
                    y.travel_seconds = 0.0;
                }
                y.decision = Decision {
                    behavior: Behavior::Build,
                    place: None,
                };
            }
            y.tick(dt, world);
        }
    }

    pub fn run() -> anyhow::Result<()> {
        let args = Args::parse();
        let seed = args.seed.unwrap_or_else(rand::random);
        let theme = args.theme;
        let mut world = Neighborhood::generate(seed, theme);
        let mut realm = Realm::new(seed);
        let (tx_prompt, rx_prompt) = mpsc::channel::<Request>();
        let (tx_result, rx_result) = mpsc::channel();
        std::thread::spawn(move || brain_worker(args, rx_prompt, tx_result));
        let place_word = match theme {
            Theme::Suburban => "Neighborhood",
            Theme::Urban => "Downtown",
        };
        let window = Window::new(WindowSettings {
            title: format!("Yumon Universe — {place_word}"),
            max_size: Some((1440, 900)),
            ..Default::default()
        })
        .expect("Could not open Universe");
        let context = window.gl();
        let mut camera = Camera::new_perspective(
            window.viewport(),
            vec3(38.0, 42.0, 48.0),
            vec3(0.0, 0.0, 0.0),
            Vec3::unit_y(),
            degrees(48.0),
            0.1,
            250.0,
        );
        let mut orbit = OrbitControl::new(*camera.target(), 4.0, 110.0);
        let ambient = AmbientLight::new(&context, 0.6, Srgba::WHITE);
        let sun = DirectionalLight::new(&context, 1.1, Srgba::WHITE, &vec3(-1.0, -2.0, -1.0));
        let scene = scenery(&context, &world);
        let mut landmarks = landmarks_for(&context, &world);
        let mut territory = territory_meshes(&context, &realm, 0);
        let mut territory_key = (0u64, 0usize, [(None, None); 16]);
        let mut selected = 0usize;
        let mut selected_parcel = 0usize;
        let mut notice = String::new();
        let mut show_names = true;
        let mut models: Vec<Model<PhysicalMaterial>> = (0..10)
            .map(|id| {
                let animals = ["fish", "giraffe", "lion", "parrot"];
                let path = format!("data/models/animal-{}.glb", animals[id % 4]);
                let result = (|| -> anyhow::Result<Model<PhysicalMaterial>> {
                    let mut assets = three_d_asset::io::load(&[std::path::Path::new(&path)])?;
                    let cpu: CpuModel = assets.deserialize(
                        std::path::Path::new(&path)
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap(),
                    )?;
                    Ok(Model::new(&context, &cpu)?)
                })();
                result.map_err(|e| {
                    anyhow::anyhow!("Required Yumon model {path} could not be loaded: {e}")
                })
            })
            .collect::<anyhow::Result<_>>()?;
        let mut yumons: Vec<_> = (0..10)
            .map(|i| {
                let home = [0, 3, 15][Realm::citizen_kingdom(i)];
                let (x, z) = Realm::center(home);
                let pos = Point {
                    x: x - 6.0,
                    z: z - 5.5 + (i % 4) as f32 * 2.0,
                };
                Yumon {
                    pos,
                    target: pos,
                    decision: Decision {
                        behavior: Behavior::Rest,
                        place: None,
                    },
                    question: None,
                    next: Instant::now() + Duration::from_secs(i as u64),
                    waiting: false,
                    turn: i,
                    log: vec![],
                    input: String::new(),
                    arrived: true,
                    facing: 0.0,
                    travel_seconds: 0.0,
                }
            })
            .collect();
        let mut gui = GUI::new(&context);
        let mut status = "Loading Language checkpoint…".to_string();
        let mut ready = false;
        let mut disconnected = false;
        let mut paused = false;
        let mut last = Instant::now();
        window.render_loop(move |mut frame| {
            let now = Instant::now();
            let dt = now.duration_since(last).as_secs_f32().min(0.1); last=now;
            loop {
                match rx_result.try_recv() {
                    Ok(BrainEvent::Ready) => { ready=true;status="Language brain ready".into(); }
                    Ok(BrainEvent::Error(e)) => { ready=false;status=e; }
                    Ok(BrainEvent::Reply(id,result)) => {
                        let y=&mut yumons[id]; y.waiting=false;y.next=now+Duration::from_secs(10);
                        match result { Ok(reply) => y.apply(reply,&world),Err(e) => { y.question=None;y.log(e); } }
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if !disconnected {
                            if ready { status="Brain worker stopped; restart Universe to reconnect.".into(); }
                            else if status.starts_with("Loading") { status="Brain worker failed during initialization. See terminal output.".into(); }
                            ready=false; disconnected=true; for y in &mut yumons { y.waiting=false; }
                        }
                        break;
                    }
                }
            }
            // One outstanding request per Yumon; the worker serializes GPU inference.
            let running=ready&&!paused;
            tick_simulation(&mut world,&mut realm,&mut yumons,dt,running);
            let key = (realm.season, selected_parcel, std::array::from_fn(|i| (realm.parcels[i].owner, realm.parcels[i].project)));
            if key != territory_key { territory = territory_meshes(&context, &realm, selected_parcel); territory_key = key; }
            // Cheap (a few matrix writes, no GPU buffer allocation) so it's fine
            // to just recompute every frame rather than track a dirty flag.
            for lm in &mut landmarks {
                let place=&mut world.places[lm.place];
                let parcel=Realm::parcel_at(place.pos.x,place.pos.z);
                place.activity_count=realm.projects.iter().filter(|p| p.completed && p.parcel==parcel).count() as u32;
                grow(lm, place.activity_count);
            }
            if running {
                for (id,y) in yumons.iter_mut().enumerate() {
                    if realm.projects.iter().any(|p|p.citizen==id && p.parcel.is_some() && !p.completed) { continue; }
                    if !y.waiting && y.arrived && now>=y.next {
                        let q=civic_question(&world,&realm,id,y.pos,y.turn);y.turn+=1;
                        y.log(format!("World: {}",q.text));
                        if tx_prompt.send(Request { id,text:q.text.clone() }).is_ok() { y.question=Some(q);y.waiting=true; }
                    }
                }
            }
            camera.set_viewport(frame.viewport);
            gui.update(&mut frame.events,frame.accumulated_time,frame.viewport,frame.device_pixel_ratio,|ctx| {
                let sidebar=egui::SidePanel::right("neighborhood").default_width(370.0).show(ctx,|ui| {
                    ui.heading("Yumon Universe");ui.label(format!("{place_word} seed: {}",world.seed));
                    if !ready { ui.colored_label(Color32::LIGHT_YELLOW,&status);ui.label("Simulation stopped. A working Language checkpoint is required."); }
                    ui.checkbox(&mut paused,"Pause simulation");ui.checkbox(&mut show_names,"Show citizen names");ui.separator();
                    egui::ScrollArea::vertical().show(ui,|ui| {
                    let citizen=&realm.citizens[selected];
                    ui.heading(format!("{} · {}",NAMES[selected],realm.kingdoms[citizen.kingdom].name));
                    ui.label(format!("Personal savings: {} coins",citizen.savings));
                    ui.label(&citizen.memory);
                    ui.label(format!("Currently {}",yumons[selected].decision.behavior.label(theme)));
                    ui.separator();
                    ui.add_enabled_ui(ready,|ui| { super::court::panel(ui,&mut realm,&mut selected,&mut selected_parcel,&mut notice); });
                    ui.separator();
                    egui::CollapsingHeader::new("Language brain").show(ui,|ui| {
                        ui.label(&status);
                        if !ready { ui.small("Load a valid Language checkpoint and restart Universe to play."); }
                    });
                    egui::CollapsingHeader::new("Skyline").default_open(false).show(ui,|ui| {
                        for lm in &landmarks {
                            let place=&world.places[lm.place];
                            ui.label(format!("{} - level {}",place.kind.noun(theme),place.activity_count));
                        }
                    });
                    ui.separator();
                        for (id,y) in yumons.iter_mut().enumerate() {
                            if ui.selectable_label(selected==id,format!("{} · {}",NAMES[id],realm.kingdoms[realm.citizens[id].kingdom].name)).clicked() { selected=id; }
                            egui::CollapsingHeader::new(format!("{} — {}{}",NAMES[id],y.decision.behavior.label(theme),if y.waiting { " (thinking)" } else { "" })).default_open(id==0).show(ui,|ui| {
                                for entry in &y.log { ui.label(RichText::new(entry).size(12.0).color(Color32::from_gray(205))); }
                                ui.text_edit_singleline(&mut y.input);
                                if ui.add_enabled(ready&&!y.waiting&&!y.input.trim().is_empty(),egui::Button::new("Send message")).clicked() {
                                    let text=std::mem::take(&mut y.input);
                                    y.log(format!("You: {text}"));
                                    // A direct message has no implied target or affirmative offer.
                                    let q=Question { text:text.clone(),place:None,offered:Behavior::Rest };
                                    if tx_prompt.send(Request { id,text }).is_ok() { y.question=Some(q);y.waiting=true; }
                                }
                            });
                        }
                    });
                    ui.separator();ui.small("Drag to orbit · scroll to zoom");
                });
                let sidebar_pixels=(sidebar.response.rect.width()*frame.device_pixel_ratio as f32).ceil() as u32;
                camera.set_viewport(Viewport { width:frame.viewport.width.saturating_sub(sidebar_pixels).max(1), ..frame.viewport });
                // Project into logical egui coordinates (physical viewport / DPI).
                if show_names {
                    let view_projection=*camera.projection() * *camera.view();
                    let scale=frame.device_pixel_ratio as f32;
                    for (id,y) in yumons.iter().enumerate() {
                        let clip=view_projection*vec4(y.pos.x,2.4,y.pos.z,1.0);
                        if clip.w<=0.0 { continue; }
                        let ndc=clip.truncate()/clip.w;
                        if ndc.x.abs()>1.0 || ndc.y.abs()>1.0 || ndc.z.abs()>1.0 { continue; }
                        let pos=egui::pos2((ndc.x+1.0)*0.5*camera.viewport().width as f32/scale,(1.0-ndc.y)*0.5*camera.viewport().height as f32/scale);
                        if !ctx.available_rect().contains(pos) { continue; }
                        let c=COLORS[realm.citizens[id].kingdom];
                        egui::Area::new(egui::Id::new(("citizen_label",id))).fixed_pos(pos).pivot(egui::Align2::CENTER_BOTTOM).order(egui::Order::Middle).show(ctx,|ui| {
                            if ui.add(egui::Button::new(RichText::new(NAMES[id]).color(Color32::WHITE)).fill(Color32::from_rgb(c[0]/2,c[1]/2,c[2]/2)).stroke(egui::Stroke::new(if selected==id {2.0} else {0.0},Color32::WHITE))).clicked() { selected=id; }
                        });
                    }
                }
            });
            orbit.handle_events(&mut camera,&mut frame.events);
            let lights: [&dyn Light;2]=[&ambient,&sun];
            let screen=frame.screen();screen.clear(ClearState::color_and_depth(0.64,0.79,0.88,1.0,1.0));
            screen.render(&camera,scene.iter().chain(landmarks.iter().flat_map(|l| l.meshes.iter())).map(|m| m as &dyn Object),&lights);
            screen.render(&camera,territory.iter().map(|m| m as &dyn Object),&lights);
            for (id,y) in yumons.iter().enumerate() {
                let moving=y.pos.distance(y.target)>0.08;
                let active=matches!(y.decision.behavior,Behavior::Play|Behavior::Collect|Behavior::Tend|Behavior::Build);
                let bob=if running&&(moving||active) { (frame.accumulated_time as f32*0.006+id as f32).sin().abs()*0.14 } else { 0.0 };
                let transform=Mat4::from_translation(vec3(y.pos.x,0.5+bob,y.pos.z))*Mat4::from_angle_y(radians(y.facing))*Mat4::from_scale(0.55);
                for mesh in models[id].iter_mut() { let original=mesh.transformation();mesh.set_transformation(transform*original);screen.render(&camera,[&*mesh as &dyn Object],&lights);mesh.set_transformation(original); }
            }
            screen.write(|| gui.render()).unwrap();FrameOutput::default()
        });
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(all(target_os = "windows", feature = "desktop"))]
    desktop::run()?;
    #[cfg(not(all(target_os = "windows", feature = "desktop")))]
    eprintln!("Yumon Universe requires Windows and the desktop Cargo feature.");
    Ok(())
}
