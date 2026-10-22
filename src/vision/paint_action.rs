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

/// Generate a progressive painting trajectory from an initial blank canvas to the target image.
/// Returns a list of `(canvas_before_action_hwc, paint_action)`.
pub fn generate_painting_trajectory(target_hwc: &[u8], steps: usize) -> Vec<(Vec<u8>, PaintAction)> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut trajectory = Vec::with_capacity(steps + 1);

    // Initial canvas: neutral canvas tone (e.g. slight warm off-white)
    let mut current_canvas = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];

    for step in 0..steps {
        let canvas_snapshot = current_canvas.clone();

        // Radii schedule: broad underpainting dabs first, then medium blocks, then fine details
        let radius = if step < steps / 4 {
            7u8 // Broad underpainting
        } else if step < steps / 2 {
            5u8 // Medium blocking
        } else if step < 3 * steps / 4 {
            3u8 // Fine shapes
        } else {
            1u8 // Fine details / highlights
        };

        // Find location with highest reconstruction error
        let mut max_err = -1.0f32;
        let mut best_x = 32u8;
        let mut best_y = 32u8;

        for y in 0..CANVAS_SIZE {
            for x in 0..CANVAS_SIZE {
                let idx = (y * CANVAS_SIZE + x) * 3;
                let dr = target_hwc[idx] as f32 - current_canvas[idx] as f32;
                let dg = target_hwc[idx + 1] as f32 - current_canvas[idx + 1] as f32;
                let db = target_hwc[idx + 2] as f32 - current_canvas[idx + 2] as f32;
                let err = dr * dr + dg * dg + db * db;
                if err > max_err {
                    max_err = err;
                    best_x = x as u8;
                    best_y = y as u8;
                }
            }
        }

        let target_idx = (best_y as usize * CANVAS_SIZE + best_x as usize) * 3;
        let target_color = rgb_to_palette(
            target_hwc[target_idx],
            target_hwc[target_idx + 1],
            target_hwc[target_idx + 2],
        );

        let action = PaintAction::Paint {
            x: best_x,
            y: best_y,
            color: target_color,
            radius,
        };

        action.apply_to_hwc(&mut current_canvas);
        trajectory.push((canvas_snapshot, action));
    }

    // Final step: canvas is complete, action is Done
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
        let dummy_target = vec![128u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        let traj = generate_painting_trajectory(&dummy_target, 6);
        assert_eq!(traj.len(), 7); // 6 steps + Done
        assert_eq!(traj.last().unwrap().1, PaintAction::Done);
    }
}
