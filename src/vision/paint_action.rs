//! Discrete paint actions, 64-color palette, coordinate quantization,
//! and canvas progression generation for Yumon Art.

use serde::{Deserialize, Serialize};

pub const CANVAS_SIZE: usize = 64;
pub const CANVAS_CHANNELS: usize = 3;
pub const PATCH_SIZE: usize = 4;
pub const GRID_SIZE: usize = CANVAS_SIZE / PATCH_SIZE; // 16
pub const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE;  // 256
pub const PALETTE_SIZE: usize = 256;

// Dedicated paint token space (mapped above the 4096 Kilo Talk vocabulary)
pub const CANVAS_START_ID: usize = 4096;
pub const CANVAS_END_ID: usize   = 4097;
pub const PAINT_ID: usize        = 4098;
pub const DONE_ID: usize         = 4099;

pub const PASS_1_ID: usize       = 4100; // Composition (8x8 blocks)
pub const PASS_2_ID: usize       = 4101; // Shapes & Silhouettes (16x16 patches)
pub const PASS_3_ID: usize       = 4102; // Fine pixel art details (64x64 pixels)

pub const ROW_START_ID: usize    = 4110; // 4110..4173 (64 row coordinates y: 0..63)
pub const COL_START_ID: usize    = 4174; // 4174..4237 (64 col coordinates x: 0..63)
pub const COLOR_START_ID: usize  = 4238; // 4238..4493 (256 8-bit palette colors: 0..255)
pub const RADIUS_START_ID: usize = 4494; // 4494..4501 (8 brush radii: 1..8)
pub const ART_VOCAB_SIZE: usize  = 4512;

/// Backward compatibility alias
pub const COORD_START_ID: usize  = 4110;

/// Map standard 24-bit RGB [0..255] into an 8-bit palette (3 bits R, 3 bits G, 2 bits B).
#[inline]
pub fn rgb_to_palette(r: u8, g: u8, b: u8) -> u8 {
    let r_bin = ((r as usize * 7 + 127) / 255).min(7) as u8;
    let g_bin = ((g as usize * 7 + 127) / 255).min(7) as u8;
    let b_bin = ((b as usize * 3 + 127) / 255).min(3) as u8;
    (r_bin * 32 + g_bin * 4 + b_bin) as u8
}

/// Expand an 8-bit palette index back into RGB bytes.
#[inline]
pub fn palette_to_rgb(idx: u8) -> [u8; 3] {
    let i = idx as usize;
    let r_bin = (i >> 5) & 0x07;
    let g_bin = (i >> 2) & 0x07;
    let b_bin = i & 0x03;
    let r = ((r_bin * 255) / 7) as u8;
    let g = ((g_bin * 255) / 7) as u8;
    let b = ((b_bin * 255) / 3) as u8;
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
                let x_tok = COL_START_ID + (x as usize).min(63);
                let y_tok = ROW_START_ID + (y as usize).min(63);
                let c_tok = COLOR_START_ID + color as usize;
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

            if x_tok >= COL_START_ID && x_tok < COL_START_ID + 64
                && y_tok >= ROW_START_ID && y_tok < ROW_START_ID + 64
                && c_tok >= COLOR_START_ID && c_tok < COLOR_START_ID + PALETTE_SIZE
                && r_tok >= RADIUS_START_ID && r_tok < RADIUS_START_ID + 8
            {
                let x = (x_tok - COL_START_ID) as u8;
                let y = (y_tok - ROW_START_ID) as u8;
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

/// Compute 2D Sobel edge gradient magnitude for an HWC [64x64x3] image.
/// Returns a flat vector of 64*64 f32 values in [0.0, 1.0].
pub fn compute_sobel_edges(hwc: &[u8]) -> Vec<f32> {
    assert_eq!(hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut lum = [0.0f32; CANVAS_SIZE * CANVAS_SIZE];
    for y in 0..CANVAS_SIZE {
        for x in 0..CANVAS_SIZE {
            let idx = (y * CANVAS_SIZE + x) * 3;
            // Standard perceptual luminance: 0.299 R + 0.587 G + 0.114 B
            lum[y * CANVAS_SIZE + x] = (0.299 * hwc[idx] as f32
                + 0.587 * hwc[idx + 1] as f32
                + 0.114 * hwc[idx + 2] as f32) / 255.0;
        }
    }

    let mut edges = vec![0.0f32; CANVAS_SIZE * CANVAS_SIZE];
    for y in 0..CANVAS_SIZE {
        let ym1 = if y == 0 { 0 } else { y - 1 };
        let yp1 = if y + 1 >= CANVAS_SIZE { CANVAS_SIZE - 1 } else { y + 1 };
        for x in 0..CANVAS_SIZE {
            let xm1 = if x == 0 { 0 } else { x - 1 };
            let xp1 = if x + 1 >= CANVAS_SIZE { CANVAS_SIZE - 1 } else { x + 1 };

            // Sobel X kernel:
            // -1  0 +1
            // -2  0 +2
            // -1  0 +1
            let gx = (lum[ym1 * CANVAS_SIZE + xp1] + 2.0 * lum[y * CANVAS_SIZE + xp1] + lum[yp1 * CANVAS_SIZE + xp1])
                   - (lum[ym1 * CANVAS_SIZE + xm1] + 2.0 * lum[y * CANVAS_SIZE + xm1] + lum[yp1 * CANVAS_SIZE + xm1]);

            // Sobel Y kernel:
            // -1 -2 -1
            //  0  0  0
            // +1 +2 +1
            let gy = (lum[yp1 * CANVAS_SIZE + xm1] + 2.0 * lum[yp1 * CANVAS_SIZE + x] + lum[yp1 * CANVAS_SIZE + xp1])
                   - (lum[ym1 * CANVAS_SIZE + xm1] + 2.0 * lum[ym1 * CANVAS_SIZE + x] + lum[ym1 * CANVAS_SIZE + xp1]);

            // Normalized gradient magnitude
            let mag = (gx * gx + gy * gy).sqrt() / 4.0;
            edges[y * CANVAS_SIZE + x] = mag.clamp(0.0, 1.0);
        }
    }
    edges
}

/// Generate Coarse-to-Fine Multiscale training samples for an image with
/// Conv2D edge detection and edge-aware coordinate mining:
/// - Pass 1: Composition (blank canvas -> 8x8 composition blocks)
/// - Pass 2: Shapes (8x8 canvas -> 16x16 shape patches)
/// - Pass 3a: Details (16x16 canvas -> native 64x64 pixels, 50% mined edges, 50% interior)
/// - Pass 3b: Refining (mixed canvas -> unpainted half, 50% mined edges, 50% interior)
/// Returns a list of `(canvas_hwc, action_tokens, is_done, target_edge)`.
pub fn generate_multiscale_samples(
    target_hwc: &[u8],
) -> Vec<(Vec<u8>, Vec<usize>, bool, f32)> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut samples = Vec::new();

    // 0. Compute 2D Sobel edge gradient map on the training target image
    let edge_map = compute_sobel_edges(target_hwc);

    // 1. Compute 8-bit quantized image and multi-level canvases:
    let canvas_blank = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];

    // Compute 8x8 block colors (composition level)
    let mut comp_colors_8x8 = [0u8; 8 * 8];
    let mut canvas_8x8 = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    for by in 0..8 {
        for bx in 0..8 {
            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;
            for py in 0..8 {
                for px in 0..8 {
                    let y = by * 8 + py;
                    let x = bx * 8 + px;
                    let idx = (y * CANVAS_SIZE + x) * 3;
                    sum_r += target_hwc[idx] as u32;
                    sum_g += target_hwc[idx + 1] as u32;
                    sum_b += target_hwc[idx + 2] as u32;
                }
            }
            let c = rgb_to_palette((sum_r / 64) as u8, (sum_g / 64) as u8, (sum_b / 64) as u8);
            comp_colors_8x8[by * 8 + bx] = c;
            let [r, g, b] = palette_to_rgb(c);
            for py in 0..8 {
                for px in 0..8 {
                    let y = by * 8 + py;
                    let x = bx * 8 + px;
                    let idx = (y * CANVAS_SIZE + x) * 3;
                    canvas_8x8[idx] = r;
                    canvas_8x8[idx + 1] = g;
                    canvas_8x8[idx + 2] = b;
                }
            }
        }
    }

    // Compute 16x16 patch colors (shape level, 4x4 patches matching visual tokens)
    let mut shape_colors_16x16 = [0u8; 16 * 16];
    let mut canvas_16x16 = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    for py in 0..16 {
        for px in 0..16 {
            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;
            for dy in 0..4 {
                for dx in 0..4 {
                    let y = py * 4 + dy;
                    let x = px * 4 + dx;
                    let idx = (y * CANVAS_SIZE + x) * 3;
                    sum_r += target_hwc[idx] as u32;
                    sum_g += target_hwc[idx + 1] as u32;
                    sum_b += target_hwc[idx + 2] as u32;
                }
            }
            let c = rgb_to_palette((sum_r / 16) as u8, (sum_g / 16) as u8, (sum_b / 16) as u8);
            shape_colors_16x16[py * 16 + px] = c;
            let [r, g, b] = palette_to_rgb(c);
            for dy in 0..4 {
                for dx in 0..4 {
                    let y = py * 4 + dy;
                    let x = px * 4 + dx;
                    let idx = (y * CANVAS_SIZE + x) * 3;
                    canvas_16x16[idx] = r;
                    canvas_16x16[idx + 1] = g;
                    canvas_16x16[idx + 2] = b;
                }
            }
        }
    }

    // Create canvas_mixed: top half is native ground-truth pixels, bottom half is 16x16 shapes
    let mut canvas_mixed = canvas_16x16.clone();
    for p in 0..2048 {
        let idx = p * 3;
        canvas_mixed[idx] = target_hwc[idx];
        canvas_mixed[idx + 1] = target_hwc[idx + 1];
        canvas_mixed[idx + 2] = target_hwc[idx + 2];
    }

    // Deterministic image seed to distribute sampled coordinates across images
    let mut seed = 2166136261usize;
    for &b in target_hwc.iter().step_by(48) {
        seed = (seed ^ (b as usize)).wrapping_mul(16777619);
    }

    // Pass 1: Full 8x8 Composition coverage (all 64 blocks per image)
    for by in 0..8 {
        for bx in 0..8 {
            let b_idx = by * 8 + bx;
            let x = bx * 8 + 4;
            let y = by * 8 + 4;
            let c = comp_colors_8x8[b_idx];

            let mut block_edge_sum = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    block_edge_sum += edge_map[(by * 8 + dy) * CANVAS_SIZE + (bx * 8 + dx)];
                }
            }
            let target_edge = block_edge_sum / 64.0;

            let tokens = vec![
                PASS_1_ID,
                ROW_START_ID + y,
                COL_START_ID + x,
                COLOR_START_ID + c as usize,
            ];
            samples.push((canvas_blank.clone(), tokens, false, target_edge));
        }
    }

    // Pass 2: Full 16x16 Shapes coverage (all 256 patches per image)
    for py in 0..16 {
        for px in 0..16 {
            let p_idx = py * 16 + px;
            let x = px * 4 + 2;
            let y = py * 4 + 2;
            let c = shape_colors_16x16[p_idx];

            let mut patch_edge_sum = 0.0f32;
            for dy in 0..4 {
                for dx in 0..4 {
                    patch_edge_sum += edge_map[(py * 4 + dy) * CANVAS_SIZE + (px * 4 + dx)];
                }
            }
            let target_edge = patch_edge_sum / 16.0;

            let tokens = vec![
                PASS_2_ID,
                ROW_START_ID + y,
                COL_START_ID + x,
                COLOR_START_ID + c as usize,
            ];
            samples.push((canvas_8x8.clone(), tokens, false, target_edge));
        }
    }

    // Pass 3a: Fine detail samples on shape canvas (Canvas 2 [16x16] -> native 64x64 pixels)
    // 4 samples per image distributed across 4 quadrants.
    // 50% are actively mined on high-gradient target edges, 50% on area interior pixels.
    const EDGE_THRESH: f32 = 0.10;
    for i in 0..32 {
        let quad_y = ((i % 4) / 2) * 32;
        let quad_x = (i % 2) * 32;
        let want_edge = i % 2 == 0;

        let hash_i = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add((i as usize).wrapping_mul(0x517CC1B727220A95));

        // Find candidate coordinates in this quadrant
        let mut chosen_x = quad_x + ((i * 23 + (hash_i >> 4)) % 32);
        let mut chosen_y = quad_y + ((i * 17 + hash_i) % 32);

        let mut best_coord = None;
        let offset = (hash_i >> 16) % 32;
        for dy in 0..32 {
            for dx in 0..32 {
                let cy = quad_y + ((dy + offset) % 32);
                let cx = quad_x + ((dx + offset) % 32);
                let e = edge_map[cy * CANVAS_SIZE + cx];
                if (want_edge && e >= EDGE_THRESH) || (!want_edge && e < EDGE_THRESH) {
                    best_coord = Some((cx, cy));
                    break;
                }
            }
            if best_coord.is_some() {
                break;
            }
        }

        if let Some((cx, cy)) = best_coord {
            chosen_x = cx;
            chosen_y = cy;
        }

        let target_idx = (chosen_y * CANVAS_SIZE + chosen_x) * 3;
        let c = rgb_to_palette(
            target_hwc[target_idx],
            target_hwc[target_idx + 1],
            target_hwc[target_idx + 2],
        );
        let target_edge = edge_map[chosen_y * CANVAS_SIZE + chosen_x];
        let tokens = vec![
            PASS_3_ID,
            ROW_START_ID + chosen_y,
            COL_START_ID + chosen_x,
            COLOR_START_ID + c as usize,
        ];
        samples.push((canvas_16x16.clone(), tokens, false, target_edge));
    }

    // Pass 3b: Fine detail samples on partially painted canvas (Canvas 3 [Mixed] -> native pixels)
    // 4 samples per image: in the unpainted bottom half (y in 32..63)
    // 50% are actively mined on high-gradient target edges, 50% on interior pixels.
    for i in 0..32 {
        let want_edge = i % 2 == 0;
        let hash_b = seed.wrapping_mul(0xD6E8FEB86659FD93).wrapping_add((i as usize).wrapping_mul(0xA24BAED4963EE407));
        let mut chosen_x = (i * 19 + (hash_b >> 8)) % CANVAS_SIZE;
        let mut chosen_y = 32 + ((i * 7 + (hash_b >> 12)) % 32);

        let mut best_coord = None;
        let offset_x = (hash_b >> 16) % CANVAS_SIZE;
        let offset_y = (hash_b >> 24) % 32;
        for dy in 0..32 {
            for dx in 0..CANVAS_SIZE {
                let cy = 32 + ((dy + offset_y) % 32);
                let cx = (dx + offset_x) % CANVAS_SIZE;
                let e = edge_map[cy * CANVAS_SIZE + cx];
                if (want_edge && e >= EDGE_THRESH) || (!want_edge && e < EDGE_THRESH) {
                    best_coord = Some((cx, cy));
                    break;
                }
            }
            if best_coord.is_some() {
                break;
            }
        }

        if let Some((cx, cy)) = best_coord {
            chosen_x = cx;
            chosen_y = cy;
        }

        let target_idx = (chosen_y * CANVAS_SIZE + chosen_x) * 3;
        let c = rgb_to_palette(
            target_hwc[target_idx],
            target_hwc[target_idx + 1],
            target_hwc[target_idx + 2],
        );
        let target_edge = edge_map[chosen_y * CANVAS_SIZE + chosen_x];
        let tokens = vec![
            PASS_3_ID,
            ROW_START_ID + chosen_y,
            COL_START_ID + chosen_x,
            COLOR_START_ID + c as usize,
        ];
        let is_last = i == 31;
        samples.push((canvas_mixed.clone(), tokens, is_last, target_edge));
    }

    samples
}

/// Generate pixel-stream training samples for an image.
/// Returns a list of `(canvas_before_chunk_hwc, pixel_tokens, is_done)`.
/// Each sample contains `chunk_len` pixels in format `[x, y, color, x, y, color, ...]`.
pub fn generate_pixel_stream_samples(
    target_hwc: &[u8],
    num_samples: usize,
    chunk_len: usize,
) -> Vec<(Vec<u8>, Vec<usize>, bool)> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let total_pixels = CANVAS_SIZE * CANVAS_SIZE; // 4096

    // Precompute target palette color index for every pixel
    let mut target_colors = Vec::with_capacity(total_pixels);
    for i in 0..total_pixels {
        let pal = rgb_to_palette(target_hwc[i * 3], target_hwc[i * 3 + 1], target_hwc[i * 3 + 2]);
        target_colors.push(pal);
    }

    let mut samples = Vec::with_capacity(num_samples);
    let chunk = chunk_len.max(1);

    // Sample progression points across the image
    for s in 0..num_samples {
        let start_pixel = if num_samples <= 1 {
            0
        } else {
            (s * (total_pixels - chunk)) / (num_samples - 1)
        };
        let end_pixel = (start_pixel + chunk).min(total_pixels);

        // Build canvas: pixels 0..start_pixel are painted with true ground-truth colors,
        // and pixels from start_pixel onwards are stripped (neutral 240).
        let mut canvas = vec![240u8; total_pixels * CANVAS_CHANNELS];
        for p in 0..start_pixel {
            let [r, g, b] = palette_to_rgb(target_colors[p]);
            let idx = p * 3;
            canvas[idx] = r;
            canvas[idx + 1] = g;
            canvas[idx + 2] = b;
        }

        // Action tokens for the next chunk of pixels: [x, y, color, ...]
        let mut tokens = Vec::with_capacity((end_pixel - start_pixel) * 3);
        for p in start_pixel..end_pixel {
            let x = p % CANVAS_SIZE;
            let y = p / CANVAS_SIZE;
            let c = target_colors[p] as usize;

            tokens.push(COORD_START_ID + x);
            tokens.push(COORD_START_ID + y);
            tokens.push(COLOR_START_ID + c);
        }

        let is_done = end_pixel >= total_pixels;
        samples.push((canvas, tokens, is_done));
    }

    samples
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
        for idx in 0..256 {
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
    fn test_multiscale_samples() {
        let dummy_target = vec![120u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        let samples = generate_multiscale_samples(&dummy_target);
        assert_eq!(samples.len(), 384); // 64 (Pass 1) + 256 (Pass 2) + 32 (Pass 3a) + 32 (Pass 3b) = 384 samples
        for s in &samples {
            assert_eq!(s.1.len(), 4); // [pass, row, col, color]
            assert!(s.3 >= 0.0 && s.3 <= 1.0); // target_edge in [0, 1]
        }
        assert!(samples.last().unwrap().2); // is_done is true for last sample
    }

    #[test]
    fn test_sobel_edge_detection() {
        // Create an image with a high-contrast vertical edge at x = 32
        let mut img = vec![0u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        for y in 0..CANVAS_SIZE {
            for x in 32..CANVAS_SIZE {
                let idx = (y * CANVAS_SIZE + x) * 3;
                img[idx] = 255;
                img[idx + 1] = 255;
                img[idx + 2] = 255;
            }
        }
        let edges = compute_sobel_edges(&img);
        assert_eq!(edges.len(), CANVAS_SIZE * CANVAS_SIZE);

        // Pixels on the vertical boundary (x = 31, 32) should have strong gradient
        let edge_val = edges[16 * CANVAS_SIZE + 32];
        assert!(edge_val > 0.4, "expected strong edge at boundary, got {}", edge_val);

        // Flat interior pixels (x = 10, y = 16) should have near-zero gradient
        let flat_val = edges[16 * CANVAS_SIZE + 10];
        assert!(flat_val < 0.01, "expected flat region to have 0 gradient, got {}", flat_val);
    }
}
