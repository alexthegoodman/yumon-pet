//! A procedural suburb driven by Language-stage replies.
#![recursion_limit = "256"]

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
        universe::{Behavior, Decision, EXTENT, Kind, Neighborhood, Point, Question, infer_reply},
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

    const NAMES: [&str; 10] = [
        "Ember", "Ripple", "Fern", "Sol", "Maple", "Clover", "Pip", "Willow", "Peach", "Moss",
    ];
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
            self.log(format!("Action: {}", self.decision.behavior.label()));
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
                    let place = &mut world.places[id];
                    if self.pos.distance(place.pos) <= place.kind.radius() + 1.0 {
                        place.activity_count += 1;
                        self.log(format!(
                            "{} by the {} (visit {}).",
                            self.decision.behavior.label(),
                            place.kind.noun(),
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
            let mut world = Neighborhood::generate(42);
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
        fn movement_stays_outside_solid_objects_and_eventually_stops() {
            let mut world = Neighborhood::generate(42);
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

    fn scenery(context: &Context, world: &Neighborhood) -> Vec<Gm<Mesh, PhysicalMaterial>> {
        let mut meshes = vec![];
        let mut add =
            |sphere, pos, size, color| meshes.push(shape(context, sphere, pos, size, color));
        add(
            false,
            [0.0, -0.15, 0.0],
            [EXTENT * 2.0, 0.25, EXTENT * 2.0],
            [115, 163, 91],
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
            match p.kind {
                Kind::House | Kind::Shop => {
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
                Kind::Tree => {
                    add(false, [x, 0.85, z], [0.4, 1.7, 0.4], [112, 81, 51]);
                    add(true, [x, 2.1, z], [2.1, 2.4, 2.1], [59, 119, 62]);
                }
                Kind::Bench => {
                    add(false, [x, 0.45, z], [1.4, 0.15, 0.6], [153, 105, 62]);
                    add(false, [x, 0.8, z - 0.25], [1.4, 0.65, 0.12], [153, 105, 62]);
                    for side in [-0.5, 0.5] {
                        add(false, [x + side, 0.2, z], [0.1, 0.4, 0.5], [65, 70, 74]);
                    }
                }
                Kind::Ball => add(true, [x, 0.3, z], [0.6, 0.6, 0.6], [241, 121, 81]),
                Kind::Garden => {
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
                Kind::Flowers => {
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
                Kind::Mailbox => {
                    add(false, [x, 0.5, z], [0.12, 1.0, 0.12], [96, 85, 73]);
                    add(false, [x, 1.05, z], [0.6, 0.4, 0.45], [73, 115, 155]);
                }
                Kind::Pond => {
                    add(true, [x, 0.02, z], [3.2, 0.15, 3.2], [88, 165, 184]);
                }
            }
        }
        meshes
    }

    pub fn run() {
        let args = Args::parse();
        let seed = args.seed.unwrap_or_else(rand::random);
        let mut world = Neighborhood::generate(seed);
        let (tx_prompt, rx_prompt) = mpsc::channel::<Request>();
        let (tx_result, rx_result) = mpsc::channel();
        std::thread::spawn(move || brain_worker(args, rx_prompt, tx_result));
        let window = Window::new(WindowSettings {
            title: "Yumon Universe — Neighborhood".into(),
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
        let mut models: Vec<Option<Model<PhysicalMaterial>>> = (0..10)
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
                result
                    .map_err(|e| eprintln!("Using sphere for {path}: {e}"))
                    .ok()
            })
            .collect();
        let mut bodies: Vec<_> = (0..10)
            .map(|i| {
                shape(
                    &context,
                    true,
                    [0.0, 0.5, 0.0],
                    [1.0, 1.0, 1.0],
                    [220, 120 + i * 10, 100 + i * 12],
                )
            })
            .collect();
        let mut yumons: Vec<_> = (0..10)
            .map(|i| {
                let pos = Point {
                    x: 0.0,
                    z: -13.5 + i as f32 * 3.0,
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
            for y in &mut yumons { if !paused { y.tick(dt,&mut world); } }
            if ready && !paused {
                for (id,y) in yumons.iter_mut().enumerate() {
                    if !y.waiting && y.arrived && now>=y.next {
                        let q=world.question(y.pos,y.turn);y.turn+=1;
                        y.log(format!("World: {}",q.text));
                        if tx_prompt.send(Request { id,text:q.text.clone() }).is_ok() { y.question=Some(q);y.waiting=true; }
                    }
                }
            }
            gui.update(&mut frame.events,frame.accumulated_time,frame.viewport,frame.device_pixel_ratio,|ctx| {
                egui::SidePanel::right("neighborhood").default_width(320.0).show(ctx,|ui| {
                    ui.heading("Yumon Universe");ui.label(format!("Neighborhood seed: {}",world.seed));
                    ui.label(&status);ui.checkbox(&mut paused,"Pause simulation");ui.separator();
                    egui::ScrollArea::vertical().show(ui,|ui| {
                        for (id,y) in yumons.iter_mut().enumerate() {
                            egui::CollapsingHeader::new(format!("{} — {}{}",NAMES[id],y.decision.behavior.label(),if y.waiting { " (thinking)" } else { "" })).default_open(id==0).show(ui,|ui| {
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
            });
            camera.set_viewport(frame.viewport);orbit.handle_events(&mut camera,&mut frame.events);
            let lights: [&dyn Light;2]=[&ambient,&sun];
            let screen=frame.screen();screen.clear(ClearState::color_and_depth(0.64,0.79,0.88,1.0,1.0));
            screen.render(&camera,scene.iter().map(|m| m as &dyn Object),&lights);
            for (id,y) in yumons.iter().enumerate() {
                let moving=y.pos.distance(y.target)>0.08;
                let active=matches!(y.decision.behavior,Behavior::Play|Behavior::Collect|Behavior::Tend);
                let bob=if !paused&&(moving||active) { (frame.accumulated_time as f32*0.006+id as f32).sin().abs()*0.14 } else { 0.0 };
                let transform=Mat4::from_translation(vec3(y.pos.x,0.5+bob,y.pos.z))*Mat4::from_angle_y(radians(y.facing))*Mat4::from_scale(0.55);
                if let Some(model)=&mut models[id] {
                    for mesh in model.iter_mut() { let original=mesh.transformation();mesh.set_transformation(transform*original);screen.render(&camera,[&*mesh as &dyn Object],&lights);mesh.set_transformation(original); }
                } else { bodies[id].set_transformation(transform);screen.render(&camera,[&bodies[id] as &dyn Object],&lights); }
            }
            screen.write(|| gui.render()).unwrap();FrameOutput::default()
        });
    }
}

fn main() {
    #[cfg(all(target_os = "windows", feature = "desktop"))]
    desktop::run();
    #[cfg(not(all(target_os = "windows", feature = "desktop")))]
    eprintln!("Yumon Universe requires Windows and the desktop Cargo feature.");
}
