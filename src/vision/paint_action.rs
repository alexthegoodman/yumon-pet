//! Discrete paint actions, 256-color palette, coordinate quantization,
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

pub const PASS_1_ID: usize       = 4100; // Composition (8x8 blocks, 64 tokens)
pub const PASS_2_ID: usize       = 4101; // Shapes & Silhouettes (16x16 patches, 256 tokens)
pub const PASS_3_ID: usize       = 4102; // Anatomy & Features (32x32 sub-patches, 1,024 tokens)
pub const PASS_4_ID: usize       = 4103; // Fine pixel art details (64x64 pixels, 4,096 tokens)

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


/// Convert a 64x64x3 RGB image into 64 Pass 1 palette color tokens (8x8 blocks).
pub fn image_to_pass1_tokens(target_hwc: &[u8]) -> Vec<u32> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut tokens = Vec::with_capacity(64);
    for by in 0..8 {
        for bx in 0..8 {
            let mut r_sum = 0usize;
            let mut g_sum = 0usize;
            let mut b_sum = 0usize;
            for dy in 0..8 {
                for dx in 0..8 {
                    let cy = by * 8 + dy;
                    let cx = bx * 8 + dx;
                    let idx = (cy * CANVAS_SIZE + cx) * 3;
                    r_sum += target_hwc[idx] as usize;
                    g_sum += target_hwc[idx + 1] as usize;
                    b_sum += target_hwc[idx + 2] as usize;
                }
            }
            let r = (r_sum / 64) as u8;
            let g = (g_sum / 64) as u8;
            let b = (b_sum / 64) as u8;
            let c = rgb_to_palette(r, g, b);
            tokens.push((COLOR_START_ID + c as usize) as u32);
        }
    }
    tokens
}

/// Convert a 64x64x3 RGB image into 256 Pass 2 palette color tokens (16x16 patches, 4x4 pixels each).
pub fn image_to_pass2_tokens(target_hwc: &[u8]) -> Vec<u32> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut tokens = Vec::with_capacity(256);
    for py in 0..16 {
        for px in 0..16 {
            let mut r_sum = 0usize;
            let mut g_sum = 0usize;
            let mut b_sum = 0usize;
            for dy in 0..4 {
                for dx in 0..4 {
                    let cy = py * 4 + dy;
                    let cx = px * 4 + dx;
                    let idx = (cy * CANVAS_SIZE + cx) * 3;
                    r_sum += target_hwc[idx] as usize;
                    g_sum += target_hwc[idx + 1] as usize;
                    b_sum += target_hwc[idx + 2] as usize;
                }
            }
            let r = (r_sum / 16) as u8;
            let g = (g_sum / 16) as u8;
            let b = (b_sum / 16) as u8;
            let c = rgb_to_palette(r, g, b);
            tokens.push((COLOR_START_ID + c as usize) as u32);
        }
    }
    tokens
}

/// Convert a 64x64x3 RGB image into 1024 Pass 3 palette color tokens (32x32 sub-patches, 2x2 pixels each).
pub fn image_to_pass3_tokens(target_hwc: &[u8]) -> Vec<u32> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut tokens = Vec::with_capacity(32 * 32);
    for py in 0..32 {
        for px in 0..32 {
            let mut r_sum = 0usize;
            let mut g_sum = 0usize;
            let mut b_sum = 0usize;
            for dy in 0..2 {
                for dx in 0..2 {
                    let cy = py * 2 + dy;
                    let cx = px * 2 + dx;
                    let idx = (cy * CANVAS_SIZE + cx) * 3;
                    r_sum += target_hwc[idx] as usize;
                    g_sum += target_hwc[idx + 1] as usize;
                    b_sum += target_hwc[idx + 2] as usize;
                }
            }
            let r = (r_sum / 4) as u8;
            let g = (g_sum / 4) as u8;
            let b = (b_sum / 4) as u8;
            let c = rgb_to_palette(r, g, b);
            tokens.push((COLOR_START_ID + c as usize) as u32);
        }
    }
    tokens
}

/// Convert a 64x64x3 RGB image into 4096 Pass 4 palette color tokens (1 pixel per token).
pub fn image_to_pass4_tokens(target_hwc: &[u8]) -> Vec<u32> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let mut tokens = Vec::with_capacity(CANVAS_SIZE * CANVAS_SIZE);
    for p in 0..(CANVAS_SIZE * CANVAS_SIZE) {
        let idx = p * 3;
        let c = rgb_to_palette(target_hwc[idx], target_hwc[idx + 1], target_hwc[idx + 2]);
        tokens.push((COLOR_START_ID + c as usize) as u32);
    }
    tokens
}

/// Reconstruct a 64x64x3 RGB image from 64 Pass 1 palette color tokens.
pub fn pass1_tokens_to_image(tokens: &[u32]) -> Vec<u8> {
    let mut canvas = vec![0u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    for (i, &tok) in tokens.iter().take(64).enumerate() {
        let c = (tok as usize).saturating_sub(COLOR_START_ID).min(PALETTE_SIZE - 1) as u8;
        let [r, g, b] = palette_to_rgb(c);
        let by = i / 8;
        let bx = i % 8;
        for dy in 0..8 {
            for dx in 0..8 {
                let cy = by * 8 + dy;
                let cx = bx * 8 + dx;
                let idx = (cy * CANVAS_SIZE + cx) * 3;
                canvas[idx] = r;
                canvas[idx + 1] = g;
                canvas[idx + 2] = b;
            }
        }
    }
    canvas
}

/// Reconstruct a 64x64x3 RGB image from 256 Pass 2 palette color tokens.
pub fn pass2_tokens_to_image(tokens: &[u32]) -> Vec<u8> {
    let mut canvas = vec![0u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    for (i, &tok) in tokens.iter().take(256).enumerate() {
        let c = (tok as usize).saturating_sub(COLOR_START_ID).min(PALETTE_SIZE - 1) as u8;
        let [r, g, b] = palette_to_rgb(c);
        let py = i / 16;
        let px = i % 16;
        for dy in 0..4 {
            for dx in 0..4 {
                let cy = py * 4 + dy;
                let cx = px * 4 + dx;
                let idx = (cy * CANVAS_SIZE + cx) * 3;
                canvas[idx] = r;
                canvas[idx + 1] = g;
                canvas[idx + 2] = b;
            }
        }
    }
    canvas
}

/// Reconstruct a 64x64x3 RGB image from 1024 Pass 3 palette color tokens (32x32 sub-patches, 2x2 pixels each).
pub fn pass3_tokens_to_image(tokens: &[u32]) -> Vec<u8> {
    let mut canvas = vec![0u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    for (i, &tok) in tokens.iter().take(32 * 32).enumerate() {
        let c = (tok as usize).saturating_sub(COLOR_START_ID).min(PALETTE_SIZE - 1) as u8;
        let [r, g, b] = palette_to_rgb(c);
        let py = i / 32;
        let px = i % 32;
        for dy in 0..2 {
            for dx in 0..2 {
                let cy = py * 2 + dy;
                let cx = px * 2 + dx;
                let idx = (cy * CANVAS_SIZE + cx) * 3;
                canvas[idx] = r;
                canvas[idx + 1] = g;
                canvas[idx + 2] = b;
            }
        }
    }
    canvas
}

/// Reconstruct a 64x64x3 RGB image from 4096 Pass 4 palette color tokens.
pub fn pass4_tokens_to_image(tokens: &[u32]) -> Vec<u8> {
    let mut canvas = vec![0u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    for (i, &tok) in tokens.iter().take(CANVAS_SIZE * CANVAS_SIZE).enumerate() {
        let c = (tok as usize).saturating_sub(COLOR_START_ID).min(PALETTE_SIZE - 1) as u8;
        let [r, g, b] = palette_to_rgb(c);
        let idx = i * 3;
        canvas[idx] = r;
        canvas[idx + 1] = g;
        canvas[idx + 2] = b;
    }
    canvas
}

/// Generate cascaded autoregressive sequence samples for an image across 4 dyadic passes.
/// Returns a list of `(pass: u8, tokens: Vec<u32>, target_start: usize)`.
pub fn generate_cascaded_sequence_samples(
    prompt_tokens: &[u32],
    target_hwc: &[u8],
) -> Vec<(u8, Vec<u32>, usize)> {
    let pass1_colors = image_to_pass1_tokens(target_hwc);
    let pass2_colors = image_to_pass2_tokens(target_hwc);
    let pass3_colors = image_to_pass3_tokens(target_hwc);
    let pass4_colors = image_to_pass4_tokens(target_hwc);

    let mut samples = Vec::with_capacity(4);

    // Pass 1 (8x8 blocks): [BOS, prompt..., PASS_1_ID, c_0..c_63, DONE_ID]
    let mut p1_tokens = vec![1u32]; // BOS_TOKEN
    p1_tokens.extend_from_slice(prompt_tokens);
    p1_tokens.push(PASS_1_ID as u32);
    let p1_target_start = p1_tokens.len();
    p1_tokens.extend(&pass1_colors);
    p1_tokens.push(DONE_ID as u32);
    samples.push((1, p1_tokens, p1_target_start));

    // Pass 2 (16x16 patches): [BOS, prompt..., PASS_1_ID, c_0..c_63, PASS_2_ID, s_0..s_255, DONE_ID]
    let mut p2_tokens = vec![1u32]; // BOS_TOKEN
    p2_tokens.extend_from_slice(prompt_tokens);
    p2_tokens.push(PASS_1_ID as u32);
    p2_tokens.extend(&pass1_colors);
    p2_tokens.push(PASS_2_ID as u32);
    let p2_target_start = p2_tokens.len();
    p2_tokens.extend(&pass2_colors);
    p2_tokens.push(DONE_ID as u32);
    samples.push((2, p2_tokens, p2_target_start));

    // Pass 3 (32x32 sub-patches): [BOS, prompt..., PASS_2_ID, s_0..s_255, PASS_3_ID, m_0..m_1023, DONE_ID]
    let mut p3_tokens = vec![1u32]; // BOS_TOKEN
    p3_tokens.extend_from_slice(prompt_tokens);
    p3_tokens.push(PASS_2_ID as u32);
    p3_tokens.extend(&pass2_colors);
    p3_tokens.push(PASS_3_ID as u32);
    let p3_target_start = p3_tokens.len();
    p3_tokens.extend(&pass3_colors);
    p3_tokens.push(DONE_ID as u32);
    samples.push((3, p3_tokens, p3_target_start));

    // Pass 4 (64x64 pixels): [BOS, prompt..., PASS_3_ID, m_0..m_1023, PASS_4_ID, p_0..p_4095, DONE_ID]
    let mut p4_tokens = vec![1u32]; // BOS_TOKEN
    p4_tokens.extend_from_slice(prompt_tokens);
    p4_tokens.push(PASS_3_ID as u32);
    p4_tokens.extend(&pass3_colors);
    p4_tokens.push(PASS_4_ID as u32);
    let p4_target_start = p4_tokens.len();
    p4_tokens.extend(&pass4_colors);
    p4_tokens.push(DONE_ID as u32);
    samples.push((4, p4_tokens, p4_target_start));

    samples
}

/// Extract multimodal 3D coordinates (t, y, x) for a sequence of token IDs.
/// Continuous canvas space is normalized to [0.0 .. 64.0] for spatial tokens.
/// Non-spatial tokens (BOS, prompt words, pass markers, PAD) receive (t, 0.0, 0.0).
pub fn extract_sequence_coords(tokens: &[i32]) -> Vec<(f32, f32, f32)> {
    let mut coords = Vec::with_capacity(tokens.len());
    let mut current_pass: Option<u8> = None;
    let mut pass_color_idx: usize = 0;
    let mut pending_y: Option<f32> = None;
    let mut pending_x: Option<f32> = None;

    for (pos, &tok_i32) in tokens.iter().enumerate() {
        let t = pos as f32;
        let tok = tok_i32 as usize;

        if tok == PASS_1_ID {
            current_pass = Some(1);
            pass_color_idx = 0;
            coords.push((t, 0.0, 0.0));
        } else if tok == PASS_2_ID {
            current_pass = Some(2);
            pass_color_idx = 0;
            coords.push((t, 0.0, 0.0));
        } else if tok == PASS_3_ID {
            current_pass = Some(3);
            pass_color_idx = 0;
            coords.push((t, 0.0, 0.0));
        } else if tok == PASS_4_ID {
            current_pass = Some(4);
            pass_color_idx = 0;
            coords.push((t, 0.0, 0.0));
        } else if tok == DONE_ID || tok == 0 /* PAD_TOKEN */ {
            current_pass = None;
            coords.push((t, 0.0, 0.0));
        } else if let Some(pass) = current_pass {
            if tok >= COLOR_START_ID && tok < COLOR_START_ID + PALETTE_SIZE {
                let k = pass_color_idx;
                pass_color_idx += 1;
                let (y, x) = match pass {
                    1 => {
                        let row = (k / 8).min(7);
                        let col = (k % 8).min(7);
                        ((row as f32 + 0.5) * 8.0, (col as f32 + 0.5) * 8.0)
                    }
                    2 => {
                        let row = (k / 16).min(15);
                        let col = (k % 16).min(15);
                        ((row as f32 + 0.5) * 4.0, (col as f32 + 0.5) * 4.0)
                    }
                    3 => {
                        let row = (k / 32).min(31);
                        let col = (k % 32).min(31);
                        ((row as f32 + 0.5) * 2.0, (col as f32 + 0.5) * 2.0)
                    }
                    4 => {
                        let row = (k / 64).min(63);
                        let col = (k % 64).min(63);
                        ((row as f32 + 0.5) * 1.0, (col as f32 + 0.5) * 1.0)
                    }
                    _ => (0.0, 0.0),
                };
                coords.push((t, y, x));
            } else {
                coords.push((t, 0.0, 0.0));
            }
        } else if tok >= ROW_START_ID && tok < COL_START_ID {
            let y = (tok - ROW_START_ID) as f32 + 0.5;
            pending_y = Some(y);
            coords.push((t, y, pending_x.unwrap_or(0.0)));
        } else if tok >= COL_START_ID && tok < COLOR_START_ID {
            let x = (tok - COL_START_ID) as f32 + 0.5;
            pending_x = Some(x);
            coords.push((t, pending_y.unwrap_or(0.0), x));
        } else if tok >= COLOR_START_ID && tok < COLOR_START_ID + PALETTE_SIZE {
            let y = pending_y.unwrap_or(0.0);
            let x = pending_x.unwrap_or(0.0);
            coords.push((t, y, x));
        } else {
            coords.push((t, 0.0, 0.0));
        }
    }

    coords
}

/// Extract multimodal 3D coordinates (t, y, x) for a batch of sequences.
/// Result is shaped [batch][seq] with (t, y, x) triples.
pub fn extract_batch_sequence_coords(
    tokens: &[i32],
    batch: usize,
    seq: usize,
) -> Vec<Vec<(f32, f32, f32)>> {
    assert_eq!(tokens.len(), batch * seq, "tokens length must match batch * seq");
    let mut batch_coords = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &tokens[b * seq..(b + 1) * seq];
        batch_coords.push(extract_sequence_coords(row));
    }
    batch_coords
}

/// Build the complete (t, y, x) coordinate trajectory for cascaded generation with KV cache.
/// `prefix`: tokens already present in the prompt (e.g. [BOS, prompt..., PASS_1_ID, ...])
/// `pass`: target pass being generated (1, 2, 3, or 4)
/// `gen_count`: number of color tokens to be generated (e.g. 64, 256, 1024, or 4096)
pub fn build_cascaded_inference_coords(
    prefix: &[i32],
    pass: u8,
    gen_count: usize,
) -> Vec<(f32, f32, f32)> {
    let mut coords = extract_sequence_coords(prefix);
    let start_t = prefix.len();
    for k in 0..gen_count {
        let t = (start_t + k) as f32;
        let (y, x) = match pass {
            1 => {
                let row = (k / 8).min(7);
                let col = (k % 8).min(7);
                ((row as f32 + 0.5) * 8.0, (col as f32 + 0.5) * 8.0)
            }
            2 => {
                let row = (k / 16).min(15);
                let col = (k % 16).min(15);
                ((row as f32 + 0.5) * 4.0, (col as f32 + 0.5) * 4.0)
            }
            3 => {
                let row = (k / 32).min(31);
                let col = (k % 32).min(31);
                ((row as f32 + 0.5) * 2.0, (col as f32 + 0.5) * 2.0)
            }
            4 => {
                let row = (k / 64).min(63);
                let col = (k % 64).min(63);
                ((row as f32 + 0.5) * 1.0, (col as f32 + 0.5) * 1.0)
            }
            _ => (0.0, 0.0),
        };
        coords.push((t, y, x));
    }
    // Add extra padding positions for safety
    for extra in 0..16 {
        coords.push(((start_t + gen_count + extra) as f32, 0.0, 0.0));
    }
    coords
}

/// Face and Eye Region of Interest (ROI) annotation for an image.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FaceRoi {
    #[serde(default)]
    pub eyes: Vec<[usize; 2]>,
    #[serde(default)]
    pub face: Option<[usize; 4]>,
    #[serde(default)]
    pub no_face: bool,
}

/// Collection of Face & Eye ROI annotations (e.g. loaded from diffusion_face_rois.json).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FaceRoiDatabase {
    #[serde(default)]
    pub version: u32,
    #[serde(default = "default_roi_canvas_size")]
    pub canvas_size: usize,
    pub annotations: std::collections::HashMap<String, FaceRoi>,
}

fn default_roi_canvas_size() -> usize {
    CANVAS_SIZE
}

/// Compute per-token spatial weights for a pass based on optional FaceRoi.
/// - Background: `w_bg` (default 1.0)
/// - Face bounding box: `w_face` (default 3.0)
/// - Eye coordinates + 1-pixel neighborhood: `w_eye` (default 10.0)
///
/// Returns a `Vec<f32>` whose length matches the number of target color tokens for that pass:
/// - Pass 1: 64 tokens (8x8)
/// - Pass 2: 256 tokens (16x16)
/// - Pass 3: 1,024 tokens (32x32)
/// - Pass 4: 4,096 tokens (64x64)
pub fn compute_pass_spatial_weights(
    roi: Option<&FaceRoi>,
    pass: u8,
    w_eye: f32,
    w_face: f32,
    w_bg: f32,
) -> Vec<f32> {
    let (grid_dim, patch_size) = match pass {
        1 => (8, 8),
        2 => (16, 4),
        3 => (32, 2),
        4 => (64, 1),
        _ => (64, 1),
    };
    let num_tokens = grid_dim * grid_dim;

    let Some(roi) = roi else {
        return vec![w_bg; num_tokens];
    };

    if roi.no_face || (roi.eyes.is_empty() && roi.face.is_none()) {
        return vec![w_bg; num_tokens];
    }

    // 1. Build 64x64 pixel grid initialized to w_bg
    let mut grid = [w_bg; CANVAS_SIZE * CANVAS_SIZE];

    // 2. Mark face bounding box if present
    if let Some(box_coords) = roi.face {
        let min_x = box_coords[0].min(box_coords[2]).min(CANVAS_SIZE - 1);
        let max_x = box_coords[0].max(box_coords[2]).min(CANVAS_SIZE - 1);
        let min_y = box_coords[1].min(box_coords[3]).min(CANVAS_SIZE - 1);
        let max_y = box_coords[1].max(box_coords[3]).min(CANVAS_SIZE - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                grid[y * CANVAS_SIZE + x] = w_face;
            }
        }
    }

    // 3. Mark eye points (+ 1-pixel neighborhood) with highest priority
    for eye in &roi.eyes {
        let ex = eye[0].min(CANVAS_SIZE - 1) as i32;
        let ey = eye[1].min(CANVAS_SIZE - 1) as i32;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let ny = (ey + dy).clamp(0, (CANVAS_SIZE - 1) as i32) as usize;
                let nx = (ex + dx).clamp(0, (CANVAS_SIZE - 1) as i32) as usize;
                grid[ny * CANVAS_SIZE + nx] = w_eye;
            }
        }
    }

    // 4. Max-pool down to the pass token grid (ensures micro-eye signals are preserved)
    let mut token_weights = Vec::with_capacity(num_tokens);
    for gy in 0..grid_dim {
        let y_start = gy * patch_size;
        let y_end = y_start + patch_size;
        for gx in 0..grid_dim {
            let x_start = gx * patch_size;
            let x_end = x_start + patch_size;

            let mut max_w = w_bg;
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let w = grid[y * CANVAS_SIZE + x];
                    if w > max_w {
                        max_w = w;
                    }
                }
            }
            token_weights.push(max_w);
        }
    }

    token_weights
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

    #[test]
    fn test_cascaded_sequence_token_roundtrip() {
        let dummy = vec![128u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        let p1 = image_to_pass1_tokens(&dummy);
        assert_eq!(p1.len(), 64);
        let img1 = pass1_tokens_to_image(&p1);
        assert_eq!(img1.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);

        let p2 = image_to_pass2_tokens(&dummy);
        assert_eq!(p2.len(), 256);
        let img2 = pass2_tokens_to_image(&p2);
        assert_eq!(img2.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);

        let p3 = image_to_pass3_tokens(&dummy);
        assert_eq!(p3.len(), 1024);
        let img3 = pass3_tokens_to_image(&p3);
        assert_eq!(img3.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);

        let p4 = image_to_pass4_tokens(&dummy);
        assert_eq!(p4.len(), 4096);
        let img4 = pass4_tokens_to_image(&p4);
        assert_eq!(img4.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);

        let samples = generate_cascaded_sequence_samples(&[10, 20, 30], &dummy);
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].0, 1);
        assert_eq!(samples[0].1.len(), 1 + 3 + 1 + 64 + 1);
        assert_eq!(samples[1].0, 2);
        assert_eq!(samples[1].1.len(), 1 + 3 + 1 + 64 + 1 + 256 + 1);
        assert_eq!(samples[2].0, 3);
        assert_eq!(samples[2].1.len(), 1 + 3 + 1 + 256 + 1 + 1024 + 1);
        assert_eq!(samples[3].0, 4);
        assert_eq!(samples[3].1.len(), 1 + 3 + 1 + 1024 + 1 + 4096 + 1);
    }

    #[test]
    fn test_mrope_sequence_coordinates() {
        // Build a Pass 2 sequence: [BOS, prompt (2), PASS_1_ID, 64 colors, PASS_2_ID, 256 colors, DONE_ID]
        let mut seq: Vec<i32> = vec![1, 50, 51, PASS_1_ID as i32];
        for _ in 0..64 {
            seq.push((COLOR_START_ID + 5) as i32);
        }
        seq.push(PASS_2_ID as i32);
        for _ in 0..256 {
            seq.push((COLOR_START_ID + 10) as i32);
        }
        seq.push(DONE_ID as i32);

        let coords = extract_sequence_coords(&seq);
        assert_eq!(coords.len(), seq.len());

        // Text prompt & pass markers have (t, 0.0, 0.0)
        assert_eq!(coords[0], (0.0, 0.0, 0.0));
        assert_eq!(coords[1], (1.0, 0.0, 0.0));
        assert_eq!(coords[2], (2.0, 0.0, 0.0));
        assert_eq!(coords[3], (3.0, 0.0, 0.0)); // PASS_1_ID

        // First Pass 1 block (row 0, col 0): center is (4.0, 4.0)
        assert_eq!(coords[4], (4.0, 4.0, 4.0));
        // Second Pass 1 block (row 0, col 1): center is (4.0, 12.0)
        assert_eq!(coords[5], (5.0, 4.0, 12.0));
        // Last Pass 1 block (row 7, col 7): center is (60.0, 60.0)
        assert_eq!(coords[4 + 63], (67.0, 60.0, 60.0));

        // PASS_2_ID marker
        let p2_marker_idx = 4 + 64;
        assert_eq!(coords[p2_marker_idx], (p2_marker_idx as f32, 0.0, 0.0));

        // First Pass 2 patch (row 0, col 0): center is (2.0, 2.0)
        let p2_first = p2_marker_idx + 1;
        assert_eq!(coords[p2_first], (p2_first as f32, 2.0, 2.0));
        // Second Pass 2 patch (row 0, col 1): center is (2.0, 6.0)
        assert_eq!(coords[p2_first + 1], ((p2_first + 1) as f32, 2.0, 6.0));
        // Last Pass 2 patch (row 15, col 15): center is (62.0, 62.0)
        assert_eq!(coords[p2_first + 255], ((p2_first + 255) as f32, 62.0, 62.0));

        // DONE_ID marker
        let done_idx = p2_first + 256;
        assert_eq!(coords[done_idx], (done_idx as f32, 0.0, 0.0));

        // Test inference trajectory building
        let prefix = vec![1, 50, PASS_1_ID as i32];
        let inf_coords = build_cascaded_inference_coords(&prefix, 1, 64);
        assert_eq!(inf_coords.len(), prefix.len() + 64 + 16);
        assert_eq!(inf_coords[3], (3.0, 4.0, 4.0)); // first gen token
        assert_eq!(inf_coords[3 + 63], (66.0, 60.0, 60.0)); // last gen token
    }

    #[test]
    fn test_compute_pass_spatial_weights() {
        // Test an image with eyes at (11, 23) and (12, 23) and face box (10, 21, 15, 27)
        let roi = FaceRoi {
            eyes: vec![[11, 23], [12, 23]],
            face: Some([10, 21, 15, 27]),
            no_face: false,
        };

        // Pass 1: 64 tokens (8x8)
        let w_p1 = compute_pass_spatial_weights(Some(&roi), 1, 10.0, 3.0, 1.0);
        assert_eq!(w_p1.len(), 64);
        assert!(w_p1.contains(&10.0));

        // Pass 2: 256 tokens (16x16)
        let w_p2 = compute_pass_spatial_weights(Some(&roi), 2, 10.0, 3.0, 1.0);
        assert_eq!(w_p2.len(), 256);
        let eye_count = w_p2.iter().filter(|&&w| (w - 10.0).abs() < 1e-4).count();
        let bg_count = w_p2.iter().filter(|&&w| (w - 1.0).abs() < 1e-4).count();
        assert!(eye_count > 0, "Pass 2 should have eye tokens with weight 10.0");
        assert_eq!(eye_count + bg_count + w_p2.iter().filter(|&&w| (w - 3.0).abs() < 1e-4).count(), 256);

        // Pass 3: 1,024 tokens (32x32)
        let w_p3 = compute_pass_spatial_weights(Some(&roi), 3, 10.0, 3.0, 1.0);
        assert_eq!(w_p3.len(), 1024);

        // Pass 4: 4,096 tokens (64x64)
        let w_p4 = compute_pass_spatial_weights(Some(&roi), 4, 10.0, 3.0, 1.0);
        assert_eq!(w_p4.len(), 4096);
        assert_eq!(w_p4[23 * 64 + 11], 10.0);
        assert_eq!(w_p4[23 * 64 + 12], 10.0);

        // Empty / None roi should return all 1.0
        let w_none = compute_pass_spatial_weights(None, 2, 10.0, 3.0, 1.0);
        assert_eq!(w_none.len(), 256);
        assert!(w_none.iter().all(|&w| w == 1.0));
    }
}
