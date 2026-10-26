/// Vision subsystem
///   - classification_head : 100-class CIFAR-100 (fine labels)
///   - emote_head          : 7-class FER2013 emotion detection
///   - patch_encoder       : 2D coordinate-aware 64x64 canvas patch encoder for MoE
///   - paint_action        : Discrete paint commands & canvas progression
///   - upscaler            : Pass 4 deep residual super-resolution (32x32 -> 256x256)

pub mod model;
pub mod loader;
pub mod cifar;
pub mod fer;
pub mod train;
pub mod paint_action;
pub mod patch_encoder;
pub mod upscaler;
pub mod colorizer;
pub mod paint_studio;
pub mod chart;

pub use paint_action::{PaintAction, CANVAS_SIZE, PATCH_SIZE, GRID_SIZE, NUM_PATCHES, ART_VOCAB_SIZE};
pub use patch_encoder::{VisionPatchEncoder, VisionPatchEncoderConfig};
pub use upscaler::{UpscalerModel, UpscalerModelConfig, UpscalerDiscriminator, UpscalerDiscriminatorConfig};
pub use colorizer::{
    ColorizerModel, ColorizerModelConfig, ColorizerLut, COLOR_BINS_AB, NUM_COLOR_BINS,
    DEFAULT_TEMPERATURE, srgb_to_lab, lab_to_srgb, colorize_fused_rgb, rebalanced_cross_entropy,
};
pub use chart::{
    UpscalerTrainingState, render as render_upscaler_chart,
    ColorizerTrainingState, render_colorizer_chart,
};
pub use paint_studio::{GalleryItem, PaintConfig, PaintPaths, PaintResult, PaintStudio, PassProgress};

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
