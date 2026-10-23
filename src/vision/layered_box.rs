//! Layered Bounding Box Art Generation for Yumon.
//!
//! Decomposes 64x64 images into <= 8 distinct color layers, each represented by
//! bounding boxes drawn back-to-front (Painter's Algorithm).
//!
//! An entire artwork is encoded as an autoregressive token sequence:
//!   [CANVAS_START, BOX, x1, y1, x2, y2, c, BOX, x1, y1, x2, y2, c, ..., DONE]
//!
//! Total sequence length is ~100-180 tokens, allowing Yumon to paint complete
//! compositions in a single coherent generation stream.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use super::paint_action::{
    CANVAS_CHANNELS, CANVAS_SIZE, CANVAS_START_ID, CANVAS_END_ID,
    PALETTE_SIZE, palette_to_rgb, rgb_to_palette,
};

pub const BOX_TOKEN_ID: usize    = 4098;
pub const DONE_TOKEN_ID: usize   = 4099;
pub const X_START_ID: usize      = 4100; // 4100..4163 (64 column coordinates x: 0..63)
pub const Y_START_ID: usize      = 4164; // 4164..4227 (64 row coordinates y: 0..63)
pub const COLOR_START_ID: usize  = 4228; // 4228..4483 (256 8-bit truecolor palette indices: 0..255)
pub const BOX_VOCAB_SIZE: usize  = 4512; // Matches existing 4512 vocab size

/// An individual colored bounding box in the layered painting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerBox {
    pub x1: u8,
    pub y1: u8,
    pub x2: u8,
    pub y2: u8,
    pub color: u8, // 0..255 palette index
}

impl LayerBox {
    /// Encode this box into 6 discrete tokens: [BOX, x1, y1, x2, y2, color].
    pub fn to_tokens(&self) -> [usize; 6] {
        [
            BOX_TOKEN_ID,
            X_START_ID + (self.x1 as usize).min(63),
            Y_START_ID + (self.y1 as usize).min(63),
            X_START_ID + (self.x2 as usize).min(63),
            Y_START_ID + (self.y2 as usize).min(63),
            COLOR_START_ID + self.color as usize,
        ]
    }
}

/// Convert a slice of boxes into a flattened token stream.
pub fn boxes_to_tokens(boxes: &[LayerBox]) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(boxes.len() * 6 + 2);
    tokens.push(CANVAS_START_ID);
    for b in boxes {
        tokens.extend_from_slice(&b.to_tokens());
    }
    tokens.push(DONE_TOKEN_ID);
    tokens
}

/// Decode a stream of tokens back into boxes.
pub fn decode_box_tokens(tokens: &[usize]) -> (Vec<LayerBox>, bool) {
    let mut boxes = Vec::new();
    let mut i = 0;
    let mut is_done = false;

    while i < tokens.len() {
        if tokens[i] == DONE_TOKEN_ID || tokens[i] == CANVAS_END_ID {
            is_done = true;
            break;
        }
        if tokens[i] == BOX_TOKEN_ID && i + 5 < tokens.len() {
            let x1_tok = tokens[i + 1];
            let y1_tok = tokens[i + 2];
            let x2_tok = tokens[i + 3];
            let y2_tok = tokens[i + 4];
            let c_tok = tokens[i + 5];

            if x1_tok >= X_START_ID && x1_tok < X_START_ID + 64
                && y1_tok >= Y_START_ID && y1_tok < Y_START_ID + 64
                && x2_tok >= X_START_ID && x2_tok < X_START_ID + 64
                && y2_tok >= Y_START_ID && y2_tok < Y_START_ID + 64
                && c_tok >= COLOR_START_ID && c_tok < COLOR_START_ID + PALETTE_SIZE
            {
                let raw_x1 = (x1_tok - X_START_ID) as u8;
                let raw_y1 = (y1_tok - Y_START_ID) as u8;
                let raw_x2 = (x2_tok - X_START_ID) as u8;
                let raw_y2 = (y2_tok - Y_START_ID) as u8;
                let color = (c_tok - COLOR_START_ID) as u8;

                let x1 = raw_x1.min(raw_x2);
                let x2 = raw_x1.max(raw_x2);
                let y1 = raw_y1.min(raw_y2);
                let y2 = raw_y1.max(raw_y2);

                boxes.push(LayerBox { x1, y1, x2, y2, color });
                i += 6;
                continue;
            }
        }
        i += 1;
    }

    (boxes, is_done)
}

/// Render a list of boxes onto an HWC (64x64x3) canvas using Painter's Algorithm.
pub fn render_boxes_to_hwc(boxes: &[LayerBox], out_hwc: &mut [u8]) {
    assert_eq!(out_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    for b in boxes {
        let [r, g, b_col] = palette_to_rgb(b.color);
        let x_end = b.x2.min((CANVAS_SIZE - 1) as u8);
        let y_end = b.y2.min((CANVAS_SIZE - 1) as u8);
        for y in b.y1..=y_end {
            for x in b.x1..=x_end {
                let idx = (y as usize * CANVAS_SIZE + x as usize) * 3;
                out_hwc[idx] = r;
                out_hwc[idx + 1] = g;
                out_hwc[idx + 2] = b_col;
            }
        }
    }
}

/// Decompose an arbitrary 64x64 image into <= max_layers color layers and bounding boxes.
pub fn extract_layered_boxes(
    target_hwc: &[u8],
    max_layers: usize,
    max_boxes_per_layer: usize,
) -> Vec<LayerBox> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let total_pixels = CANVAS_SIZE * CANVAS_SIZE;

    // 1. Map all pixels to 8-bit truecolor palette indices
    let mut pal_pixels = Vec::with_capacity(total_pixels);
    let mut color_counts: HashMap<u8, usize> = HashMap::new();
    for p in 0..total_pixels {
        let idx = p * 3;
        let c = rgb_to_palette(target_hwc[idx], target_hwc[idx + 1], target_hwc[idx + 2]);
        pal_pixels.push(c);
        *color_counts.entry(c).or_insert(0) += 1;
    }

    // 2. Select top N colors by frequency
    let mut sorted_colors: Vec<(u8, usize)> = color_counts.into_iter().collect();
    sorted_colors.sort_by(|a, b| b.1.cmp(&a.1));
    sorted_colors.truncate(max_layers.max(1));

    let top_colors: Vec<u8> = sorted_colors.iter().map(|(c, _)| *c).collect();
    let top_rgb: Vec<[u8; 3]> = top_colors.iter().map(|&c| palette_to_rgb(c)).collect();

    // 3. Re-quantize each pixel to the nearest of the selected top colors
    let mut quantized_pixels = Vec::with_capacity(total_pixels);
    let mut layer_pixel_counts = vec![0usize; top_colors.len()];
    for p in 0..total_pixels {
        let idx = p * 3;
        let r = target_hwc[idx] as i32;
        let g = target_hwc[idx + 1] as i32;
        let b = target_hwc[idx + 2] as i32;

        let mut best_layer = 0;
        let mut best_dist = i32::MAX;
        for (l_idx, rgb) in top_rgb.iter().enumerate() {
            let dr = r - rgb[0] as i32;
            let dg = g - rgb[1] as i32;
            let db = b - rgb[2] as i32;
            let dist = dr * dr + dg * dg + db * db;
            if dist < best_dist {
                best_dist = dist;
                best_layer = l_idx;
            }
        }
        quantized_pixels.push(best_layer);
        layer_pixel_counts[best_layer] += 1;
    }

    // 4. Sort layers by actual re-quantized pixel count descending
    let mut layer_indices: Vec<usize> = (0..top_colors.len()).collect();
    layer_indices.sort_by(|&a, &b| layer_pixel_counts[b].cmp(&layer_pixel_counts[a]));

    let mut boxes = Vec::new();

    // Layer 0 is the background wash: covers the entire canvas
    let bg_color = top_colors[layer_indices[0]];
    boxes.push(LayerBox {
        x1: 0,
        y1: 0,
        x2: (CANVAS_SIZE - 1) as u8,
        y2: (CANVAS_SIZE - 1) as u8,
        color: bg_color,
    });

    // Subsequent layers: find connected components and extract bounding boxes
    for &l_idx in &layer_indices[1..] {
        let color = top_colors[l_idx];
        let mut visited = vec![false; total_pixels];
        let mut components: Vec<Vec<(u8, u8)>> = Vec::new();

        for y in 0..CANVAS_SIZE {
            for x in 0..CANVAS_SIZE {
                let p = y * CANVAS_SIZE + x;
                if quantized_pixels[p] == l_idx && !visited[p] {
                    let mut comp = Vec::new();
                    let mut queue = VecDeque::new();
                    visited[p] = true;
                    queue.push_back((x as u8, y as u8));

                    while let Some((cx, cy)) = queue.pop_front() {
                        comp.push((cx, cy));
                        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                            let nx = cx as i32 + dx;
                            let ny = cy as i32 + dy;
                            if nx >= 0 && nx < CANVAS_SIZE as i32 && ny >= 0 && ny < CANVAS_SIZE as i32 {
                                let np = ny as usize * CANVAS_SIZE + nx as usize;
                                if quantized_pixels[np] == l_idx && !visited[np] {
                                    visited[np] = true;
                                    queue.push_back((nx as u8, ny as u8));
                                }
                            }
                        }
                    }

                    // Keep components with at least 3 pixels (filters isolated noise)
                    if comp.len() >= 3 {
                        components.push(comp);
                    }
                }
            }
        }

        // Sort components by size descending
        components.sort_by(|a, b| b.len().cmp(&a.len()));

        // Take top components up to max_boxes_per_layer
        for comp in components.iter().take(max_boxes_per_layer) {
            let mut min_x = u8::MAX;
            let mut max_x = 0u8;
            let mut min_y = u8::MAX;
            let mut max_y = 0u8;
            for &(px, py) in comp {
                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);
            }

            boxes.push(LayerBox {
                x1: min_x,
                y1: min_y,
                x2: max_x,
                y2: max_y,
                color,
            });
        }
    }

    boxes
}

/// Extract bounding boxes directly using the full 256 truecolor palette (no arbitrary layer count limit).
///
/// 1. Finds dominant color background wash (0, 0, 63, 63).
/// 2. Discovers connected components for colors with significant presence (>= 4 pixels).
/// 3. Filters out sparse/non-compact components (fills >= 20% of bounding box).
/// 4. Orders boxes by area descending so background/large structures are drawn first,
///    and fine details/highlights are layered on top (Painter's Algorithm).
/// 5. Caps at `max_boxes`.
pub fn extract_truecolor_boxes(
    target_hwc: &[u8],
    max_boxes: usize,
) -> Vec<LayerBox> {
    assert_eq!(target_hwc.len(), CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS);
    let total_pixels = CANVAS_SIZE * CANVAS_SIZE;

    // 1. Map all pixels to 8-bit truecolor palette indices (0..255)
    let mut pal_pixels = Vec::with_capacity(total_pixels);
    let mut color_counts = [0usize; PALETTE_SIZE];
    for p in 0..total_pixels {
        let idx = p * 3;
        let c = rgb_to_palette(target_hwc[idx], target_hwc[idx + 1], target_hwc[idx + 2]);
        pal_pixels.push(c);
        color_counts[c as usize] += 1;
    }

    // 2. Find the most dominant color for full canvas wash
    let mut best_bg_color = 0u8;
    let mut best_bg_count = 0usize;
    for (c, &cnt) in color_counts.iter().enumerate() {
        if cnt > best_bg_count {
            best_bg_count = cnt;
            best_bg_color = c as u8;
        }
    }

    let mut boxes = Vec::new();
    // Canvas wash: background
    boxes.push(LayerBox {
        x1: 0,
        y1: 0,
        x2: (CANVAS_SIZE - 1) as u8,
        y2: (CANVAS_SIZE - 1) as u8,
        color: best_bg_color,
    });

    if max_boxes <= 1 {
        return boxes;
    }

    let min_comp_len = if max_boxes > 40 { 2 } else { 4 };
    let min_color_count = if max_boxes > 40 { 2 } else { 4 };

    // 3. Collect candidate bounding boxes across all non-background colors
    let mut candidates: Vec<(LayerBox, usize, usize)> = Vec::new(); // (box, area, comp_pixels)
    let mut visited = vec![false; total_pixels];

    for y in 0..CANVAS_SIZE {
        for x in 0..CANVAS_SIZE {
            let p = y * CANVAS_SIZE + x;
            if visited[p] {
                continue;
            }
            let c = pal_pixels[p];
            if c == best_bg_color || color_counts[c as usize] < min_color_count {
                visited[p] = true;
                continue;
            }

            // BFS for connected component of color c
            let mut comp = Vec::new();
            let mut queue = VecDeque::new();
            visited[p] = true;
            queue.push_back((x as u8, y as u8));

            while let Some((cx, cy)) = queue.pop_front() {
                comp.push((cx, cy));
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx >= 0 && nx < CANVAS_SIZE as i32 && ny >= 0 && ny < CANVAS_SIZE as i32 {
                        let np = ny as usize * CANVAS_SIZE + nx as usize;
                        if !visited[np] && pal_pixels[np] == c {
                            visited[np] = true;
                            queue.push_back((nx as u8, ny as u8));
                        }
                    }
                }
            }

            if comp.len() >= min_comp_len {
                let mut min_x = u8::MAX;
                let mut max_x = 0u8;
                let mut min_y = u8::MAX;
                let mut max_y = 0u8;
                for &(px, py) in &comp {
                    min_x = min_x.min(px);
                    max_x = max_x.max(px);
                    min_y = min_y.min(py);
                    max_y = max_y.max(py);
                }

                let w = (max_x - min_x + 1) as usize;
                let h = (max_y - min_y + 1) as usize;
                let area = w * h;

                // Density filter: small accent components (area <= 9) pass directly;
                // larger components must fill at least 20% of their bounding box.
                if area <= 9 || comp.len() * 5 >= area {
                    candidates.push((
                        LayerBox {
                            x1: min_x,
                            y1: min_y,
                            x2: max_x,
                            y2: max_y,
                            color: c,
                        },
                        area,
                        comp.len(),
                    ));
                }
            }
        }
    }

    // 4. Sort candidates:
    // Large areas come FIRST (background elements, large bodies)
    // Small areas come LAST (details, eyes, highlights) -> Painter's Algorithm!
    // Break ties by component pixel density
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));

    // Take top (max_boxes - 1) candidates
    for (cand_box, _, _) in candidates.into_iter().take(max_boxes - 1) {
        boxes.push(cand_box);
    }

    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_token_roundtrip() {
        let b = LayerBox {
            x1: 10,
            y1: 20,
            x2: 30,
            y2: 40,
            color: 125,
        };
        let tokens = b.to_tokens();
        assert_eq!(tokens[0], BOX_TOKEN_ID);
        assert_eq!(tokens[1], X_START_ID + 10);
        assert_eq!(tokens[2], Y_START_ID + 20);
        assert_eq!(tokens[3], X_START_ID + 30);
        assert_eq!(tokens[4], Y_START_ID + 40);
        assert_eq!(tokens[5], COLOR_START_ID + 125);

        let stream = boxes_to_tokens(&[b]);
        let (recovered, is_done) = decode_box_tokens(&stream);
        assert!(is_done);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0], b);
    }

    #[test]
    fn test_layered_boxes_extraction() {
        let dummy_hwc = vec![120u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        let boxes = extract_layered_boxes(&dummy_hwc, 8, 3);
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].x1, 0);
        assert_eq!(boxes[0].y1, 0);
        assert_eq!(boxes[0].x2, 63);
        assert_eq!(boxes[0].y2, 63);
    }

    #[test]
    fn test_truecolor_boxes_extraction() {
        let mut dummy_hwc = vec![10u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
        // Put a red box in the middle
        for y in 10..20 {
            for x in 10..20 {
                let idx = (y * CANVAS_SIZE + x) * 3;
                dummy_hwc[idx] = 255;
                dummy_hwc[idx + 1] = 0;
                dummy_hwc[idx + 2] = 0;
            }
        }
        let boxes = extract_truecolor_boxes(&dummy_hwc, 10);
        assert!(boxes.len() >= 2);
        // Box 0 is background wash
        assert_eq!(boxes[0].x1, 0);
        assert_eq!(boxes[0].y1, 0);
        assert_eq!(boxes[0].x2, 63);
        assert_eq!(boxes[0].y2, 63);
        // Box 1 should be the red box
        assert_eq!(boxes[1].x1, 10);
        assert_eq!(boxes[1].y1, 10);
        assert_eq!(boxes[1].x2, 19);
        assert_eq!(boxes[1].y2, 19);
    }
}
