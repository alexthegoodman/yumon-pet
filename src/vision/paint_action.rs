//! Discrete paint actions, 64-color palette, coordinate quantization,
//! and canvas progression generation for Yumon Art.

use serde::{Deserialize, Serialize};

pub const CANVAS_SIZE: usize = 64;
pub const CANVAS_CHANNELS: usize = 3;
pub const PATCH_SIZE: usize = 4;
pub const GRID_SIZE: usize = CANVAS_SIZE / PATCH_SIZE; // 16
pub const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE;  // 256
pub const PALETTE_SIZE: usize = 64;

// Dedicated paint token space (mapped above the 4096 Kilo Talk vocabulary)
pub const CANVAS_START_ID: usize = 4096;
pub const CANVAS_END_ID: usize   = 4097;
pub const PAINT_ID: usize        = 4098;
pub const DONE_ID: usize         = 4099;
pub const COORD_START_ID: usize  = 4100; // 4100..4163 (64 coordinates for x and y: 0..63)
pub const COLOR_START_ID: usize  = 4164; // 4164..4227 (64 palette colors: 0..63)
pub const RADIUS_START_ID: usize = 4228; // 4228..4235 (8 brush radii: 1..8)
pub const ART_VOCAB_SIZE: usize  = 4240;

/// Map standard 24-bit RGB [0..255] into a uniform 64-color palette (4 levels per channel).
#[inline]
pub fn rgb_to_palette(r: u8, g: u8, b: u8) -> u8 {
    let r_idx = ((r as usize * 3 + 127) / 255).min(3) as u8;
    let g_idx = ((g as usize * 3 + 127) / 255).min(3) as u8;
    let b_idx = ((b as usize * 3 + 127) / 255).min(3) as u8;
    (r_idx * 16) + (g_idx * 4) + b_idx
}

/// Expand a 64-color palette index back into RGB bytes.
#[inline]
pub fn palette_to_rgb(idx: u8) -> [u8; 3] {
    let idx = idx % 64;
    let r = (idx / 16) * 85;
    let g = ((idx % 16) / 4) * 85;
    let b = (idx % 4) * 85;
    [r, g, b]
}

/// Discrete paint command issued by Yumon's MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaintAction {
    /// Paint a circular dab centered at (x, y) with a palette color and radius (1..8).
    Paint {
        x: u8,      // 0..63
        y: u8,      // 0..63
        color: u8,  // 0..63 (palette index)
        radius: u8, // 1..8
    },
    /// The canvas is complete.
    Done,
}

impl PaintAction {
    /// Encode paint action into discrete token IDs.
    pub fn to_tokens(&self) -> Vec<usize> {
        match *self {
            PaintAction::Paint { x, y, color, radius } => {
                let x_tok = COORD_START_ID + (x as usize).min(63);
                let y_tok = COORD_START_ID + (y as usize).min(63);
                let c_tok = COLOR_START_ID + (color as usize).min(63);
                let r_clamped = radius.clamp(1, 8) as usize;
                let r_tok = RADIUS_START_ID + (r_clamped - 1);
                vec![PAINT_ID, x_tok, y_tok, c_tok, r_tok]
            }
            PaintAction::Done => vec![DONE_ID],
        }
    }

    /// Decode discrete token IDs into a PaintAction.
    pub fn from_tokens(tokens: &[usize]) -> Option<Self> {
        if tokens.is_empty() {
            return None;
        }
        if tokens[0] == DONE_ID {
            return Some(PaintAction::Done);
        }
        if tokens[0] == PAINT_ID && tokens.len() >= 5 {
            let x_tok = tokens[1];
            let y_tok = tokens[2];
            let c_tok = tokens[3];
            let r_tok = tokens[4];

            if x_tok >= COORD_START_ID && x_tok < COORD_START_ID + 64
                && y_tok >= COORD_START_ID && y_tok < COORD_START_ID + 64
                && c_tok >= COLOR_START_ID && c_tok < COLOR_START_ID + 64
                && r_tok >= RADIUS_START_ID && r_tok < RADIUS_START_ID + 8
            {
                let x = (x_tok - COORD_START_ID) as u8;
                let y = (y_tok - COORD_START_ID) as u8;
                let color = (c_tok - COLOR_START_ID) as u8;
                let radius = (r_tok - RADIUS_START_ID + 1) as u8;
                return Some(PaintAction::Paint { x, y, color, radius });
            }
        }
        None
    }

    /// Apply this paint action to an HWC (64x64x3) RGB canvas.
    pub fn apply_to_hwc(&self, canvas: &mut [u8]) {
        match *self {
            PaintAction::Paint { x, y, color, radius } => {
                let [r, g, b] = palette_to_rgb(color);
                let cx = x as i32;
                let cy = y as i32;
                let rad = radius as i32;
                let rad_sq = rad * rad;

                let min_x = (cx - rad).max(0);
                let max_x = (cx + rad).min(CANVAS_SIZE as i32 - 1);
                let min_y = (cy - rad).max(0);
                let max_y = (cy + rad).min(CANVAS_SIZE as i32 - 1);

                for py in min_y..=max_y {
                    for px in min_x..=max_x {
                        let dx = px - cx;
                        let dy = py - cy;
                        if dx * dx + dy * dy <= rad_sq {
                            let idx = (py as usize * CANVAS_SIZE + px as usize) * 3;
                            canvas[idx] = r;
                            canvas[idx + 1] = g;
                            canvas[idx + 2] = b;
                        }
                    }
                }
            }
            PaintAction::Done => {}
        }
    }
}

/// Convert flat HWC [64*64*3] u8 bytes into planar CHW [3, 64, 64] floats normalized to [-1.0, 1.0].
pub fn hwc_to_chw_floats(hwc: &[u8]) -> Vec<f32> {
    assert_eq!(hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let plane = CANVAS_SIZE * CANVAS_SIZE;
    let mut chw = vec![0.0f32; 3 * plane];

    for y in 0..CANVAS_SIZE {
        for x in 0..CANVAS_SIZE {
            let hwc_idx = (y * CANVAS_SIZE + x) * 3;
            let pix_idx = y * CANVAS_SIZE + x;
            chw[0 * plane + pix_idx] = hwc[hwc_idx] as f32 / 255.0 * 2.0 - 1.0;
            chw[1 * plane + pix_idx] = hwc[hwc_idx + 1] as f32 / 255.0 * 2.0 - 1.0;
            chw[2 * plane + pix_idx] = hwc[hwc_idx + 2] as f32 / 255.0 * 2.0 - 1.0;
        }
    }

    chw
}

/// Generate a color-stripped painting trajectory where unpainted pixels are stripped
/// and colors are progressively revealed from background/dominant layers to fine details.
/// Returns a list of `(canvas_before_action_hwc, paint_action)`.
pub fn generate_painting_trajectory(target_hwc: &[u8], steps: usize) -> Vec<(Vec<u8>, PaintAction)> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);

    // 1. Map every pixel to its palette index
    let mut color_pixels: std::collections::HashMap<u8, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for y in 0..CANVAS_SIZE {
        for x in 0..CANVAS_SIZE {
            let idx = (y * CANVAS_SIZE + x) * 3;
            let pal = rgb_to_palette(target_hwc[idx], target_hwc[idx + 1], target_hwc[idx + 2]);
            color_pixels.entry(pal).or_default().push((x, y));
        }
    }

    // 2. Sort colors by descending frequency (background/dominant to accents/details)
    let mut sorted_colors: Vec<(u8, usize)> = color_pixels
        .iter()
        .map(|(&c, pts)| (c, pts.len()))
        .collect();
    sorted_colors.sort_by(|a, b| b.1.cmp(&a.1));

    let mut trajectory = Vec::with_capacity(steps + 1);

    // Canvas starts with neutral off-white canvas tone (stripped pixels)
    let mut current_canvas = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];

    let num_stages = steps.max(1);
    let colors_per_stage = (sorted_colors.len() + num_stages - 1) / num_stages;

    for stage in 0..num_stages {
        let canvas_snapshot = current_canvas.clone();
        let start = stage * colors_per_stage;
        let end = (start + colors_per_stage).min(sorted_colors.len());

        if start >= sorted_colors.len() {
            break;
        }

        // Primary color for this stage's action is the first/largest in this group
        let (primary_color, _) = sorted_colors[start];
        let primary_pts = &color_pixels[&primary_color];

        // Compute centroid of this color
        let sum_x: usize = primary_pts.iter().map(|&(x, _)| x).sum();
        let sum_y: usize = primary_pts.iter().map(|&(_, y)| y).sum();
        let cx = (sum_x / primary_pts.len()) as u8;
        let cy = (sum_y / primary_pts.len()) as u8;

        // Radius based on pixel count
        let count = primary_pts.len();
        let radius = if count > 600 {
            7u8
        } else if count > 250 {
            5u8
        } else if count > 80 {
            3u8
        } else {
            1u8 // Crisp pixel detail!
        };

        let action = PaintAction::Paint {
            x: cx,
            y: cy,
            color: primary_color,
            radius,
        };

        // Reveal the pixels for all colors in this stage onto current_canvas
        for &(c, _) in &sorted_colors[start..end] {
            let [r, g, b] = palette_to_rgb(c);
            for &(px, py) in &color_pixels[&c] {
                let idx = (py * CANVAS_SIZE + px) * 3;
                current_canvas[idx] = r;
                current_canvas[idx + 1] = g;
                current_canvas[idx + 2] = b;
            }
        }

        trajectory.push((canvas_snapshot, action));
    }

    // Final state: canvas is 100% the DiffusionDb image, action is Done!
    trajectory.push((current_canvas, PaintAction::Done));
    trajectory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_roundtrip() {
        for idx in 0..64 {
            let [r, g, b] = palette_to_rgb(idx as u8);
            let recovered = rgb_to_palette(r, g, b);
            assert_eq!(idx as u8, recovered);
        }
    }

    #[test]
    fn test_paint_action_token_roundtrip() {
        let action = PaintAction::Paint { x: 42, y: 17, color: 55, radius: 4 };
        let tokens = action.to_tokens();
        assert_eq!(tokens.len(), 5);
        let recovered = PaintAction::from_tokens(&tokens);
        assert_eq!(Some(action), recovered);

        let done = PaintAction::Done;
        let done_tokens = done.to_tokens();
        assert_eq!(done_tokens, vec![DONE_ID]);
        assert_eq!(Some(done), PaintAction::from_tokens(&done_tokens));
    }

    #[test]
    fn test_trajectory_generation() {
        let mut multi_color_target = vec![0u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        for y in 0..CANVAS_SIZE {
            for x in 0..CANVAS_SIZE {
                let idx = (y * CANVAS_SIZE + x) * 3;
                let col = ((x / 8) * 32) as u8;
                multi_color_target[idx] = col;
                multi_color_target[idx + 1] = 255 - col;
                multi_color_target[idx + 2] = (y * 4) as u8;
            }
        }
        let traj = generate_painting_trajectory(&multi_color_target, 6);
        assert!(traj.len() >= 2);
        assert_eq!(traj.last().unwrap().1, PaintAction::Done);
    }
}
