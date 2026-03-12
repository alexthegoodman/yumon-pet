/// Vision subsystem — CNN with dual heads:
///   - classification_head : 100-class CIFAR-100 (fine labels)
///   - emote_head          : 7-class FER2013 emotion detection

pub mod model;
pub mod loader;
pub mod cifar;
pub mod fer;
pub mod train;

// ─── Constants ────────────────────────────────────────────────────────────────

pub const CIFAR_CLASSES:  usize = 100;
pub const EMOTE_CLASSES:  usize = 7;
pub const IMG_SIZE:       usize = 32;    // CIFAR-100 native size
pub const FER_SIZE:       usize = 48;    // FER2013 native size; we resize to IMG_SIZE

/// FER2013 canonical emotion names, index-aligned.
pub const EMOTE_NAMES: [&str; EMOTE_CLASSES] = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
];

pub fn emote_name_to_idx(name: &str) -> Option<usize> {
    EMOTE_NAMES.iter().position(|&n| n.eq_ignore_ascii_case(name))
}
