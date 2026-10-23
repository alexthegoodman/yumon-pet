/// Vision subsystem
///   - classification_head : 100-class CIFAR-100 (fine labels)
///   - emote_head          : 7-class FER2013 emotion detection
///   - patch_encoder       : 2D coordinate-aware 64x64 canvas patch encoder for MoE
///   - paint_action        : Discrete paint commands & canvas progression

pub mod model;
pub mod loader;
pub mod cifar;
pub mod fer;
pub mod train;
pub mod paint_action;
pub mod patch_encoder;
pub mod layered_box;

pub use paint_action::{
    PaintAction, CANVAS_SIZE, PATCH_SIZE, GRID_SIZE, NUM_PATCHES, ART_VOCAB_SIZE,
    CANVAS_START_ID, palette_to_rgb, rgb_to_palette,
};
pub use patch_encoder::{VisionPatchEncoder, VisionPatchEncoderConfig};
pub use layered_box::{
    LayerBox, BOX_TOKEN_ID, DONE_TOKEN_ID, X_START_ID, Y_START_ID, COLOR_START_ID,
    BOX_VOCAB_SIZE, boxes_to_tokens, decode_box_tokens, render_boxes_to_hwc,
    extract_layered_boxes, extract_truecolor_boxes,
};

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
