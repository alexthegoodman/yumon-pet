//! DAW semantic action vocabulary for the UI prediction model.
//!
//! Defines the fixed set of semantic actions a DAW user can take, their
//! parameter schemas, and encoding/decoding utilities. This is the shared
//! vocabulary consumed by both the procedural data generator and the
//! prediction model.

use serde::{Deserialize, Serialize};

// ── Action vocabulary ─────────────────────────────────────────────────────────

/// Maximum number of f32 parameters any single action carries.
pub const MAX_ACTION_PARAMS: usize = 4;

/// Total number of distinct semantic actions in the DAW vocabulary.
pub const NUM_DAW_ACTIONS: usize = 32;

/// Padding action ID (analogous to PAD_TOKEN in the language model).
pub const PAD_ACTION: u32 = NUM_DAW_ACTIONS as u32;

/// Beginning-of-sequence sentinel.
pub const BOS_ACTION: u32 = NUM_DAW_ACTIONS as u32 + 1;

/// End-of-sequence sentinel.
pub const EOS_ACTION: u32 = NUM_DAW_ACTIONS as u32 + 2;

/// Full vocabulary size including sentinels.
pub const ACTION_VOCAB_SIZE: usize = NUM_DAW_ACTIONS + 3; // actions + PAD + BOS + EOS

/// Every semantic action the DAW exposes to the prediction model.
///
/// Each variant maps to a fixed integer ID (its discriminant). The model
/// predicts over this vocabulary; the application maps predicted IDs back
/// to concrete controls in the Suggested Next Steps panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum DawAction {
    AddTrack         = 0,
    RemoveTrack      = 1,
    SelectTrack      = 2,
    SetBpm           = 3,
    Play             = 4,
    Stop             = 5,
    Rewind           = 6,
    Seek             = 7,
    AddNote          = 8,
    RemoveNote       = 9,
    SelectPattern    = 10,
    NewPattern       = 11,
    PlaceClip        = 12,
    RemoveClip       = 13,
    LoadInstrument   = 14,
    LoadPreset       = 15,
    SetVolume        = 16,
    SetPan           = 17,
    MuteTrack        = 18,
    SoloTrack        = 19,
    SetReverb        = 20,
    SetEq            = 21,
    OpenMixer        = 22,
    OpenPianoRoll    = 23,
    PlayNotePreview  = 24,
    SetScale         = 25,
    SetRootNote      = 26,
    ExportWav        = 27,
    SaveProject      = 28,
    Undo             = 29,
    ToggleAnalyzer   = 30,
    SetCharacterKnob = 31,
}

impl DawAction {
    pub const ALL: [DawAction; NUM_DAW_ACTIONS] = [
        Self::AddTrack, Self::RemoveTrack, Self::SelectTrack, Self::SetBpm,
        Self::Play, Self::Stop, Self::Rewind, Self::Seek,
        Self::AddNote, Self::RemoveNote, Self::SelectPattern, Self::NewPattern,
        Self::PlaceClip, Self::RemoveClip, Self::LoadInstrument, Self::LoadPreset,
        Self::SetVolume, Self::SetPan, Self::MuteTrack, Self::SoloTrack,
        Self::SetReverb, Self::SetEq, Self::OpenMixer, Self::OpenPianoRoll,
        Self::PlayNotePreview, Self::SetScale, Self::SetRootNote, Self::ExportWav,
        Self::SaveProject, Self::Undo, Self::ToggleAnalyzer, Self::SetCharacterKnob,
    ];

    pub fn id(self) -> u32 { self as u32 }

    pub fn from_id(id: u32) -> Option<Self> {
        if (id as usize) < NUM_DAW_ACTIONS {
            Some(Self::ALL[id as usize])
        } else {
            None
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::AddTrack         => "add_track",
            Self::RemoveTrack      => "remove_track",
            Self::SelectTrack      => "select_track",
            Self::SetBpm           => "set_bpm",
            Self::Play             => "play",
            Self::Stop             => "stop",
            Self::Rewind           => "rewind",
            Self::Seek             => "seek",
            Self::AddNote          => "add_note",
            Self::RemoveNote       => "remove_note",
            Self::SelectPattern    => "select_pattern",
            Self::NewPattern       => "new_pattern",
            Self::PlaceClip        => "place_clip",
            Self::RemoveClip       => "remove_clip",
            Self::LoadInstrument   => "load_instrument",
            Self::LoadPreset       => "load_preset",
            Self::SetVolume        => "set_volume",
            Self::SetPan           => "set_pan",
            Self::MuteTrack        => "mute_track",
            Self::SoloTrack        => "solo_track",
            Self::SetReverb        => "set_reverb",
            Self::SetEq            => "set_eq",
            Self::OpenMixer        => "open_mixer",
            Self::OpenPianoRoll    => "open_piano_roll",
            Self::PlayNotePreview  => "play_note_preview",
            Self::SetScale         => "set_scale",
            Self::SetRootNote      => "set_root_note",
            Self::ExportWav        => "export_wav",
            Self::SaveProject      => "save_project",
            Self::Undo             => "undo",
            Self::ToggleAnalyzer   => "toggle_analyzer",
            Self::SetCharacterKnob => "set_character_knob",
        }
    }

    /// Number of meaningful float parameters this action carries (0..=4).
    pub fn param_count(self) -> usize {
        match self {
            Self::AddTrack         => 1, // kind: 0=synth, 1=drum
            Self::RemoveTrack      => 1, // track index
            Self::SelectTrack      => 1, // track index
            Self::SetBpm           => 1, // bpm
            Self::Play | Self::Stop | Self::Rewind => 0,
            Self::Seek             => 1, // step
            Self::AddNote          => 4, // row, step, length, velocity
            Self::RemoveNote       => 2, // row, step
            Self::SelectPattern    => 1, // pattern index
            Self::NewPattern       => 1, // steps
            Self::PlaceClip        => 2, // bar, pattern index
            Self::RemoveClip       => 1, // bar
            Self::LoadInstrument   => 1, // instrument index
            Self::LoadPreset       => 1, // preset index
            Self::SetVolume        => 1, // level 0..1
            Self::SetPan           => 1, // pan -1..1
            Self::MuteTrack | Self::SoloTrack => 0,
            Self::SetReverb        => 2, // param index, value
            Self::SetEq            => 2, // band, gain
            Self::OpenMixer | Self::OpenPianoRoll => 0,
            Self::PlayNotePreview  => 2, // midi, velocity
            Self::SetScale         => 1, // scale index
            Self::SetRootNote      => 1, // midi
            Self::ExportWav | Self::SaveProject | Self::Undo => 0,
            Self::ToggleAnalyzer   => 0,
            Self::SetCharacterKnob => 2, // knob index, value
        }
    }
}

// ── Encoded action step ───────────────────────────────────────────────────────

/// A single timestep in an action sequence: an action ID plus up to
/// MAX_ACTION_PARAMS float parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionStep {
    pub action_id: u32,
    pub params: [f32; MAX_ACTION_PARAMS],
}

impl ActionStep {
    pub fn new(action: DawAction, params: &[f32]) -> Self {
        let mut p = [0.0f32; MAX_ACTION_PARAMS];
        let n = params.len().min(MAX_ACTION_PARAMS);
        p[..n].copy_from_slice(&params[..n]);
        Self { action_id: action.id(), params: p }
    }

    pub fn pad() -> Self {
        Self { action_id: PAD_ACTION, params: [0.0; MAX_ACTION_PARAMS] }
    }

    pub fn bos() -> Self {
        Self { action_id: BOS_ACTION, params: [0.0; MAX_ACTION_PARAMS] }
    }

    pub fn eos() -> Self {
        Self { action_id: EOS_ACTION, params: [0.0; MAX_ACTION_PARAMS] }
    }
}

// ── Task templates ────────────────────────────────────────────────────────────

/// High-level task categories that the procedural generator samples from.
/// Each defines a distribution of plausible action trajectories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum DawTask {
    CreateBeat       = 0,
    LayerMelody      = 1,
    MixAndMaster     = 2,
    ArrangeSong      = 3,
    SoundDesign      = 4,
    RecordAndExport  = 5,
    QuickSketch      = 6,  // short: set bpm, add track, write a few notes, play
    EditPattern      = 7,  // focused pattern editing session
}

impl DawTask {
    pub const ALL: [DawTask; 8] = [
        Self::CreateBeat, Self::LayerMelody, Self::MixAndMaster,
        Self::ArrangeSong, Self::SoundDesign, Self::RecordAndExport,
        Self::QuickSketch, Self::EditPattern,
    ];

    pub fn id(self) -> u32 { self as u32 }
}

pub const NUM_DAW_TASKS: usize = DawTask::ALL.len();

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_ids_are_contiguous() {
        for (i, action) in DawAction::ALL.iter().enumerate() {
            assert_eq!(action.id() as usize, i);
        }
    }

    #[test]
    fn round_trip_action_id() {
        for action in DawAction::ALL {
            assert_eq!(DawAction::from_id(action.id()), Some(action));
        }
        assert_eq!(DawAction::from_id(NUM_DAW_ACTIONS as u32), None);
    }

    #[test]
    fn sentinel_ids_are_distinct() {
        assert_ne!(PAD_ACTION, BOS_ACTION);
        assert_ne!(BOS_ACTION, EOS_ACTION);
        assert!(PAD_ACTION >= NUM_DAW_ACTIONS as u32);
    }

    #[test]
    fn action_step_pad_has_pad_id() {
        let step = ActionStep::pad();
        assert_eq!(step.action_id, PAD_ACTION);
    }

    #[test]
    fn action_step_preserves_params() {
        let step = ActionStep::new(DawAction::AddNote, &[5.0, 8.0, 4.0, 0.8]);
        assert_eq!(step.action_id, DawAction::AddNote.id());
        assert_eq!(step.params, [5.0, 8.0, 4.0, 0.8]);
    }

    #[test]
    fn action_step_truncates_excess_params() {
        let step = ActionStep::new(DawAction::Play, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Only MAX_ACTION_PARAMS kept
        assert_eq!(step.params.len(), MAX_ACTION_PARAMS);
    }
}
