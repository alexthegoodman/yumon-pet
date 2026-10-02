#![allow(warnings)]

//! Procedural DAW workflow data generator.
//!
//! Generates synthetic action trajectories that simulate realistic DAW usage
//! patterns. Each trajectory is a sequence of semantic actions drawn from
//! task templates (create a beat, layer a melody, mix, arrange, etc.).
//!
//! Usage:
//!   cargo run --release --bin gen_daw_data -- \
//!       --out data/daw_sequences.bin \
//!       --count 50000 \
//!       --seed 42
//!
//! The output file is a plain bincode-encoded Vec<Trajectory> that the
//! training binary loads directly.

use anyhow::Result;
use clap::Parser;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::io::Write;
use yumon_pet::brain::daw_actions::*;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "gen_daw_data", about = "Generate synthetic DAW action sequences")]
struct Cli {
    /// Output file path (bincode format).
    #[arg(long, default_value = "data/daw_sequences.bin")]
    out: String,

    /// Number of trajectories to generate.
    #[arg(long, default_value_t = 50_000)]
    count: usize,

    /// RNG seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum actions per trajectory.
    #[arg(long, default_value_t = 64)]
    max_len: usize,
}

// ── Trajectory ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub task_id: u32,
    pub steps: Vec<ActionStep>,
}

// ── Task generators ───────────────────────────────────────────────────────────

fn gen_create_beat(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();
    let bpm = rng.gen_range(80.0..160.0f32);

    // Set tempo
    steps.push(ActionStep::new(DawAction::SetBpm, &[bpm]));

    // Add a drum track
    steps.push(ActionStep::new(DawAction::AddTrack, &[1.0])); // 1 = drum
    steps.push(ActionStep::new(DawAction::SelectTrack, &[0.0]));

    // Load a preset
    let preset = rng.gen_range(0..8) as f32;
    steps.push(ActionStep::new(DawAction::LoadPreset, &[preset]));

    // Create a pattern
    let pattern_steps = [16, 32, 64][rng.gen_range(0..3)] as f32;
    steps.push(ActionStep::new(DawAction::NewPattern, &[pattern_steps]));
    steps.push(ActionStep::new(DawAction::SelectPattern, &[0.0]));
    steps.push(ActionStep::new(DawAction::OpenPianoRoll, &[]));

    // Write kick pattern (every 4 steps)
    let total_steps = pattern_steps as u32;
    for step in (0..total_steps).step_by(4) {
        let vel = rng.gen_range(0.7..1.0f32);
        steps.push(ActionStep::new(DawAction::AddNote, &[0.0, step as f32, 1.0, vel]));
    }

    // Write snare pattern (beats 2 and 4)
    for step in (0..total_steps).filter(|s| s % 8 == 4) {
        let vel = rng.gen_range(0.6..0.95f32);
        steps.push(ActionStep::new(DawAction::AddNote, &[1.0, step as f32, 1.0, vel]));
    }

    // Write hi-hat pattern
    let hat_interval = [2, 4][rng.gen_range(0..2)];
    for step in (0..total_steps).step_by(hat_interval) {
        let vel = if step % 4 == 0 { rng.gen_range(0.5..0.8) } else { rng.gen_range(0.3..0.6) };
        steps.push(ActionStep::new(DawAction::AddNote, &[2.0, step as f32, 1.0, vel]));
    }

    // Place clip in arrangement
    steps.push(ActionStep::new(DawAction::PlaceClip, &[0.0, 0.0]));

    // Play and listen
    steps.push(ActionStep::new(DawAction::Rewind, &[]));
    steps.push(ActionStep::new(DawAction::Play, &[]));

    // Maybe adjust volume
    if rng.gen_bool(0.5) {
        let vol = rng.gen_range(0.6..1.0f32);
        steps.push(ActionStep::new(DawAction::SetVolume, &[vol]));
    }

    steps.push(ActionStep::new(DawAction::Stop, &[]));
    steps
}

fn gen_layer_melody(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();

    // Add a synth track
    steps.push(ActionStep::new(DawAction::AddTrack, &[0.0])); // 0 = synth
    let track_idx = rng.gen_range(0..4) as f32;
    steps.push(ActionStep::new(DawAction::SelectTrack, &[track_idx]));

    // Load instrument and preset
    let instrument = rng.gen_range(0..6) as f32;
    steps.push(ActionStep::new(DawAction::LoadInstrument, &[instrument]));
    let preset = rng.gen_range(0..12) as f32;
    steps.push(ActionStep::new(DawAction::LoadPreset, &[preset]));

    // Set scale and root
    let scale = rng.gen_range(0..8) as f32;
    let root = [48, 50, 52, 53, 55, 57, 59, 60][rng.gen_range(0..8)] as f32;
    steps.push(ActionStep::new(DawAction::SetScale, &[scale]));
    steps.push(ActionStep::new(DawAction::SetRootNote, &[root]));

    // Preview some notes
    for _ in 0..rng.gen_range(2..5) {
        let midi = rng.gen_range(48..84) as f32;
        let vel = rng.gen_range(0.5..1.0f32);
        steps.push(ActionStep::new(DawAction::PlayNotePreview, &[midi, vel]));
    }

    // Create pattern and write melody
    steps.push(ActionStep::new(DawAction::NewPattern, &[32.0]));
    steps.push(ActionStep::new(DawAction::SelectPattern, &[0.0]));
    steps.push(ActionStep::new(DawAction::OpenPianoRoll, &[]));

    let note_count = rng.gen_range(8..20);
    let mut current_step = 0u32;
    for _ in 0..note_count {
        let row = rng.gen_range(0..12);
        let length = [1, 2, 4, 8][rng.gen_range(0..4)] as f32;
        let vel = rng.gen_range(0.5..1.0f32);
        steps.push(ActionStep::new(DawAction::AddNote, &[
            row as f32, current_step as f32, length, vel,
        ]));
        current_step += length as u32;
        if current_step >= 32 { break; }
    }

    // Place and play
    steps.push(ActionStep::new(DawAction::PlaceClip, &[0.0, 0.0]));
    steps.push(ActionStep::new(DawAction::Rewind, &[]));
    steps.push(ActionStep::new(DawAction::Play, &[]));
    steps.push(ActionStep::new(DawAction::Stop, &[]));

    steps
}

fn gen_mix_and_master(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();
    let num_tracks = rng.gen_range(2..6);

    steps.push(ActionStep::new(DawAction::OpenMixer, &[]));

    for track in 0..num_tracks {
        steps.push(ActionStep::new(DawAction::SelectTrack, &[track as f32]));

        // Set volume
        let vol = rng.gen_range(0.4..1.0f32);
        steps.push(ActionStep::new(DawAction::SetVolume, &[vol]));

        // Set pan
        let pan = rng.gen_range(-0.8..0.8f32);
        steps.push(ActionStep::new(DawAction::SetPan, &[pan]));

        // Maybe add reverb
        if rng.gen_bool(0.6) {
            let wet = rng.gen_range(0.1..0.5f32);
            steps.push(ActionStep::new(DawAction::SetReverb, &[0.0, wet]));
        }

        // Maybe EQ
        if rng.gen_bool(0.5) {
            let band = rng.gen_range(0..4) as f32;
            let gain = rng.gen_range(-6.0..6.0f32);
            steps.push(ActionStep::new(DawAction::SetEq, &[band, gain]));
        }

        // Maybe mute or solo
        if rng.gen_bool(0.2) {
            steps.push(ActionStep::new(DawAction::MuteTrack, &[]));
        }
        if rng.gen_bool(0.15) {
            steps.push(ActionStep::new(DawAction::SoloTrack, &[]));
        }
    }

    // Toggle analyzer to check levels
    steps.push(ActionStep::new(DawAction::ToggleAnalyzer, &[]));
    steps.push(ActionStep::new(DawAction::Rewind, &[]));
    steps.push(ActionStep::new(DawAction::Play, &[]));
    steps.push(ActionStep::new(DawAction::Stop, &[]));

    steps
}

fn gen_arrange_song(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();
    let num_patterns = rng.gen_range(2..5);

    // Create several patterns
    for p in 0..num_patterns {
        let pattern_len = [16, 32][rng.gen_range(0..2)] as f32;
        steps.push(ActionStep::new(DawAction::NewPattern, &[pattern_len]));
        steps.push(ActionStep::new(DawAction::SelectPattern, &[p as f32]));

        // Write some notes in each
        let note_count = rng.gen_range(4..12);
        for _ in 0..note_count {
            let row = rng.gen_range(0..12) as f32;
            let step = rng.gen_range(0..pattern_len as u32) as f32;
            let len = [1.0, 2.0, 4.0][rng.gen_range(0..3)];
            let vel = rng.gen_range(0.5..1.0f32);
            steps.push(ActionStep::new(DawAction::AddNote, &[row, step, len, vel]));
        }
    }

    // Place clips in arrangement (verse-chorus-verse structure)
    let structure = [0, 0, 1, 0, 0, 1, 2, 1]; // pattern indices for each bar
    for (bar, &pattern) in structure.iter().enumerate() {
        if (pattern as usize) < num_patterns {
            steps.push(ActionStep::new(DawAction::PlaceClip, &[bar as f32, pattern as f32]));
        }
    }

    // Maybe remove a clip and replace it
    if rng.gen_bool(0.3) {
        let bar = rng.gen_range(0..structure.len()) as f32;
        steps.push(ActionStep::new(DawAction::RemoveClip, &[bar]));
        let alt_pattern = rng.gen_range(0..num_patterns) as f32;
        steps.push(ActionStep::new(DawAction::PlaceClip, &[bar, alt_pattern]));
    }

    steps.push(ActionStep::new(DawAction::Rewind, &[]));
    steps.push(ActionStep::new(DawAction::Play, &[]));
    steps.push(ActionStep::new(DawAction::Stop, &[]));

    steps
}

fn gen_sound_design(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();

    steps.push(ActionStep::new(DawAction::AddTrack, &[0.0])); // synth
    steps.push(ActionStep::new(DawAction::SelectTrack, &[0.0]));

    // Load instrument
    let instrument = rng.gen_range(0..6) as f32;
    steps.push(ActionStep::new(DawAction::LoadInstrument, &[instrument]));

    // Iterate through presets and preview
    let num_presets_to_try = rng.gen_range(3..8);
    for _ in 0..num_presets_to_try {
        let preset = rng.gen_range(0..16) as f32;
        steps.push(ActionStep::new(DawAction::LoadPreset, &[preset]));

        // Preview a note
        let midi = rng.gen_range(48..72) as f32;
        steps.push(ActionStep::new(DawAction::PlayNotePreview, &[midi, 0.8]));

        // Tweak character knobs
        let num_tweaks = rng.gen_range(1..4);
        for _ in 0..num_tweaks {
            let knob = rng.gen_range(0..8) as f32;
            let value = rng.gen_range(0.0..1.0f32);
            steps.push(ActionStep::new(DawAction::SetCharacterKnob, &[knob, value]));
        }

        // Preview again after tweaking
        steps.push(ActionStep::new(DawAction::PlayNotePreview, &[midi, 0.8]));
    }

    steps
}

fn gen_record_and_export(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();

    // Assume existing project: rewind, play, listen, maybe adjust, export
    steps.push(ActionStep::new(DawAction::Rewind, &[]));
    steps.push(ActionStep::new(DawAction::Play, &[]));

    // Seek to a specific point
    if rng.gen_bool(0.5) {
        let seek_step = rng.gen_range(0..128) as f32;
        steps.push(ActionStep::new(DawAction::Seek, &[seek_step]));
    }

    steps.push(ActionStep::new(DawAction::Stop, &[]));

    // Maybe adjust a few things
    if rng.gen_bool(0.6) {
        let track = rng.gen_range(0..4) as f32;
        steps.push(ActionStep::new(DawAction::SelectTrack, &[track]));
        let vol = rng.gen_range(0.5..1.0f32);
        steps.push(ActionStep::new(DawAction::SetVolume, &[vol]));
    }

    // Save and export
    steps.push(ActionStep::new(DawAction::SaveProject, &[]));
    steps.push(ActionStep::new(DawAction::ExportWav, &[]));

    steps
}

fn gen_quick_sketch(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();

    let bpm = rng.gen_range(90.0..140.0f32);
    steps.push(ActionStep::new(DawAction::SetBpm, &[bpm]));
    steps.push(ActionStep::new(DawAction::AddTrack, &[0.0])); // synth
    steps.push(ActionStep::new(DawAction::SelectTrack, &[0.0]));

    let preset = rng.gen_range(0..8) as f32;
    steps.push(ActionStep::new(DawAction::LoadPreset, &[preset]));
    steps.push(ActionStep::new(DawAction::NewPattern, &[16.0]));
    steps.push(ActionStep::new(DawAction::OpenPianoRoll, &[]));

    // A few notes
    for i in 0..rng.gen_range(3..8) {
        let row = rng.gen_range(0..12) as f32;
        let vel = rng.gen_range(0.5..1.0f32);
        steps.push(ActionStep::new(DawAction::AddNote, &[row, (i * 2) as f32, 2.0, vel]));
    }

    steps.push(ActionStep::new(DawAction::PlaceClip, &[0.0, 0.0]));
    steps.push(ActionStep::new(DawAction::Play, &[]));
    steps.push(ActionStep::new(DawAction::Stop, &[]));

    steps
}

fn gen_edit_pattern(rng: &mut StdRng) -> Vec<ActionStep> {
    let mut steps = Vec::new();

    let pattern = rng.gen_range(0..4) as f32;
    steps.push(ActionStep::new(DawAction::SelectPattern, &[pattern]));
    steps.push(ActionStep::new(DawAction::OpenPianoRoll, &[]));

    // Remove some notes and add new ones
    let removals = rng.gen_range(2..6);
    for _ in 0..removals {
        let row = rng.gen_range(0..12) as f32;
        let step = rng.gen_range(0..32) as f32;
        steps.push(ActionStep::new(DawAction::RemoveNote, &[row, step]));
    }

    let additions = rng.gen_range(3..10);
    for _ in 0..additions {
        let row = rng.gen_range(0..12) as f32;
        let step = rng.gen_range(0..32) as f32;
        let len = [1.0, 2.0, 4.0][rng.gen_range(0..3)];
        let vel = rng.gen_range(0.4..1.0f32);
        steps.push(ActionStep::new(DawAction::AddNote, &[row, step, len, vel]));
    }

    // Undo a few times
    if rng.gen_bool(0.3) {
        for _ in 0..rng.gen_range(1..3) {
            steps.push(ActionStep::new(DawAction::Undo, &[]));
        }
    }

    // Play to hear result
    steps.push(ActionStep::new(DawAction::Rewind, &[]));
    steps.push(ActionStep::new(DawAction::Play, &[]));
    steps.push(ActionStep::new(DawAction::Stop, &[]));

    steps
}

// ── Generator dispatch ────────────────────────────────────────────────────────

fn generate_trajectory(rng: &mut StdRng, max_len: usize) -> Trajectory {
    let task = DawTask::ALL[rng.gen_range(0..DawTask::ALL.len())];

    let mut steps = match task {
        DawTask::CreateBeat      => gen_create_beat(rng),
        DawTask::LayerMelody     => gen_layer_melody(rng),
        DawTask::MixAndMaster    => gen_mix_and_master(rng),
        DawTask::ArrangeSong     => gen_arrange_song(rng),
        DawTask::SoundDesign     => gen_sound_design(rng),
        DawTask::RecordAndExport => gen_record_and_export(rng),
        DawTask::QuickSketch     => gen_quick_sketch(rng),
        DawTask::EditPattern     => gen_edit_pattern(rng),
    };

    // Prepend BOS and append EOS
    steps.insert(0, ActionStep::bos());
    steps.push(ActionStep::eos());

    // Truncate to max_len
    steps.truncate(max_len);

    Trajectory {
        task_id: task.id(),
        steps,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Generating {} DAW action trajectories (seed={}, max_len={})",
        cli.count, cli.seed, cli.max_len);

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut trajectories = Vec::with_capacity(cli.count);
    let mut task_counts = [0usize; NUM_DAW_TASKS];

    let progress = indicatif::ProgressBar::new(cli.count as u64);
    progress.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap(),
    );

    for _ in 0..cli.count {
        let traj = generate_trajectory(&mut rng, cli.max_len);
        task_counts[traj.task_id as usize] += 1;
        trajectories.push(traj);
        progress.inc(1);
    }
    progress.finish_with_message("done");

    // Print statistics
    println!("\nTask distribution:");
    for (i, task) in DawTask::ALL.iter().enumerate() {
        println!("  {:?}: {} ({:.1}%)",
            task, task_counts[i],
            100.0 * task_counts[i] as f64 / cli.count as f64);
    }

    let total_steps: usize = trajectories.iter().map(|t| t.steps.len()).sum();
    let avg_len = total_steps as f64 / cli.count as f64;
    println!("\nTotal action steps: {}", total_steps);
    println!("Average trajectory length: {:.1}", avg_len);

    // Serialize to disk
    let encoded = serde_json::to_vec(&trajectories)?;
    let out_path = std::path::Path::new(&cli.out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(&cli.out)?;
    file.write_all(&encoded)?;

    let file_size = std::fs::metadata(&cli.out)?.len();
    println!("\nWrote {} ({:.1} MB)", cli.out, file_size as f64 / 1_048_576.0);

    Ok(())
}
