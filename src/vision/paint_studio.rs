//! Yumon Paint Studio: Multi-pass Cascaded Generative Vision Pipeline
//!
//! Cascaded inference stages:
//! - Pass 1: Global composition (64 8x8 block tokens) -> 64x64 canvas
//! - Pass 2: Shapes & Silhouettes (256 4x4 patch tokens) -> 64x64 canvas
//! - Pass 3: Fine pixel art details (1,024 2x2 sub-patch tokens) -> 64x64 canvas
//! - Pass 4: Deep Neural Super-Resolution (32x32 -> 256x256 feed-forward upscaling)

use anyhow::{Context, Result};
use burn::{
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, TensorData},
};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::brain::{
    flash_attn::backend::FlashAttention,
    kilo_vocab::KiloVocab,
    moe_model::{YumonMoeBrain, YumonMoeBrainConfig},
    samples::TrainingStage,
    tokenizer::BOS_TOKEN,
    yumon_talk::TranslationContext,
};

use super::{
    paint_action::{
        ART_VOCAB_SIZE, CANVAS_SIZE, CANVAS_CHANNELS,
        PASS_1_ID, PASS_2_ID, PASS_3_ID, PASS_4_ID,
        COLOR_START_ID, PALETTE_SIZE,
        build_cascaded_inference_coords,
        palette_to_rgb,
        pass1_tokens_to_image, pass2_tokens_to_image, pass3_tokens_to_image, pass4_tokens_to_image,
    },
    upscaler::{UpscalerModel, UpscalerModelConfig},
    colorizer::{colorize_fused_rgb, ColorizerModel, ColorizerModelConfig, DEFAULT_TEMPERATURE},
};

/// Configuration for an art generation run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaintConfig {
    pub prompt: String,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub shape_guidance: f32,
    pub passes: usize,
}

impl Default for PaintConfig {
    fn default() -> Self {
        Self {
            prompt: "a cute friendly dragon in an enchanted forest, vibrant colors, oil painting".into(),
            temperature: 1.1,
            top_k: 24,
            top_p: 0.88,
            repeat_penalty: 1.0,
            shape_guidance: 0.5,
            passes: 5,
        }
    }
}

/// Progress update emitted after each cascaded generation pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PassProgress {
    pub pass: usize,
    pub total_passes: usize,
    pub stage_name: String,
    pub description: String,
    pub width: u32,
    pub height: u32,
    pub data_url: String,
    pub elapsed_secs: f32,
}

/// Information about a completed or saved artwork.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GalleryItem {
    pub id: String,
    pub path: String,
    pub raw_path: Option<String>,
    pub prompt: String,
    pub timestamp: String,
    pub width: u32,
    pub height: u32,
    pub thumbnail_url: String,
}

/// Final result returned by `PaintStudio::generate`.
pub struct PaintResult {
    pub final_path: PathBuf,
    pub raw_path: PathBuf,
    pub metadata_path: PathBuf,
    pub prompt: String,
    pub final_data_url: String,
    pub thumbnail_url: String,
    pub width: u32,
    pub height: u32,
    pub duration: Duration,
}

/// Checkpoint and vocabulary paths for the paint studio.
#[derive(Clone, Debug)]
pub struct PaintPaths {
    pub vocab_path: PathBuf,
    pub checkpoint_pass1: Option<PathBuf>,
    pub checkpoint_pass2: Option<PathBuf>,
    pub checkpoint_pass3: Option<PathBuf>,
    pub checkpoint_pass4: Option<PathBuf>,
    pub upscaler_checkpoint: Option<PathBuf>,
    pub colorizer_checkpoint: Option<PathBuf>,
}

impl Default for PaintPaths {
    fn default() -> Self {
        Self::auto_discover()
    }
}

impl PaintPaths {
    /// Auto-discovers the highest-quality available checkpoints in workspace.
    pub fn auto_discover() -> Self {
        let vocab = PathBuf::from("training-cache/diffusion_talk_vocab.json");

        let p1 = ["checkpoints/yumon-art-pass1-db", "checkpoints/yumon-art-pass1"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists());

        let p2 = ["checkpoints/yumon-art-pass2-db", "checkpoints/yumon-art-pass2"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists());

        let p3 = ["checkpoints/yumon-art-pass3-db", "checkpoints/yumon-art-pass3"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists());

        let p4 = ["checkpoints/yumon-art-pass4-db", "checkpoints/yumon-art-pass4"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists());

        let upscaler = [
            "checkpoints/yumon-upscaler-64",
            "checkpoints/yumon-upscaler-64-gan",
            "checkpoints/yumon-upscaler-256",
        ]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists());

        let colorizer = [
            "checkpoints/yumon-colorizer-256",
        ]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists());

        Self {
            vocab_path: vocab,
            checkpoint_pass1: p1,
            checkpoint_pass2: p2,
            checkpoint_pass3: p3,
            checkpoint_pass4: p4,
            upscaler_checkpoint: upscaler,
            colorizer_checkpoint: colorizer,
        }
    }
}

/// Encodes RGB buffer (HWC) to PNG bytes.
pub fn rgb_to_png_bytes(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let img: image::RgbImage = image::ImageBuffer::from_raw(width, height, rgb.to_vec())
        .context("Failed to construct image buffer from RGB raw bytes")?;
    let mut cursor = std::io::Cursor::new(Vec::new());
    img.write_to(&mut cursor, image::ImageFormat::Png)?;
    Ok(cursor.into_inner())
}

/// Encodes RGB buffer to a data:image/png;base64 URL.
pub fn rgb_to_data_url(rgb: &[u8], width: u32, height: u32) -> Result<String> {
    use base64::Engine;
    let bytes = rgb_to_png_bytes(rgb, width, height)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/png;base64,{b64}"))
}

/// Downscales RGB buffer to a crisp square thumbnail data URL.
pub fn rgb_to_thumbnail_data_url(rgb: &[u8], width: u32, height: u32, thumb_size: u32) -> Result<String> {
    use base64::Engine;
    let img: image::RgbImage = image::ImageBuffer::from_raw(width, height, rgb.to_vec())
        .context("Failed to construct image buffer")?;
    let thumb = image::imageops::resize(&img, thumb_size, thumb_size, image::imageops::FilterType::Triangle);
    let mut cursor = std::io::Cursor::new(Vec::new());
    thumb.write_to(&mut cursor, image::ImageFormat::Png)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
    Ok(format!("data:image/png;base64,{b64}"))
}

/// Reads an existing PNG from disk and converts to thumbnail data URL.
pub fn file_to_thumbnail_data_url(path: &Path, thumb_size: u32) -> Result<String> {
    use base64::Engine;
    let img = image::open(path)?.to_rgb8();
    let thumb = image::imageops::resize(&img, thumb_size, thumb_size, image::imageops::FilterType::Triangle);
    let mut cursor = std::io::Cursor::new(Vec::new());
    thumb.write_to(&mut cursor, image::ImageFormat::Png)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
    Ok(format!("data:image/png;base64,{b64}"))
}

/// Nearest-neighbor scaling fallback.
pub fn upscale_nn(src: &[u8], width: usize, height: usize, scale: usize) -> Vec<u8> {
    let out_w = width * scale;
    let out_h = height * scale;
    let mut out = vec![0u8; out_w * out_h * 3];
    for y in 0..out_h {
        let src_y = y / scale;
        for x in 0..out_w {
            let src_x = x / scale;
            let src_idx = (src_y * width + src_x) * 3;
            let out_idx = (y * out_w + x) * 3;
            out[out_idx] = src[src_idx];
            out[out_idx + 1] = src[src_idx + 1];
            out[out_idx + 2] = src[src_idx + 2];
        }
    }
    out
}

/// Discrete token sampler with nucleus (top-p), top-k, temperature, and consecutive repeat penalty.
pub fn sample_color_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    last_color: Option<usize>,
    consecutive_repeats: usize,
    repeat_penalty: f32,
) -> usize {
    let mut working_logits = logits.to_vec();

    if let Some(c) = last_color {
        if consecutive_repeats >= 4 && c < working_logits.len() && repeat_penalty > 0.0 {
            let penalty = (consecutive_repeats as f32 - 3.0) * repeat_penalty;
            working_logits[c] -= penalty;
        }
    }

    if temperature <= 0.001 {
        return working_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    let mut candidates: Vec<(usize, f32)> = working_logits.iter().cloned().enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < candidates.len() {
        candidates.truncate(top_k);
    }

    let max_logit = candidates[0].1;
    let mut exp_probs: Vec<(usize, f32)> = candidates
        .into_iter()
        .map(|(idx, l)| (idx, ((l - max_logit) / temperature).exp()))
        .collect();
    let sum_exp: f32 = exp_probs.iter().map(|(_, p)| p).sum();
    if sum_exp <= 0.0 {
        return exp_probs.first().map(|(idx, _)| *idx).unwrap_or(0);
    }
    for (_, p) in &mut exp_probs {
        *p /= sum_exp;
    }

    if top_p > 0.0 && top_p < 1.0 {
        let mut cum_p = 0.0;
        let mut cut = exp_probs.len();
        for (i, &(_, p)) in exp_probs.iter().enumerate() {
            cum_p += p;
            if cum_p >= top_p {
                cut = (i + 1).max(1);
                break;
            }
        }
        exp_probs.truncate(cut);
        let renorm_sum: f32 = exp_probs.iter().map(|(_, p)| p).sum();
        if renorm_sum > 0.0 {
            for (_, p) in &mut exp_probs {
                *p /= renorm_sum;
            }
        }
    }

    let r = rand::random::<f32>();
    let mut acc = 0.0;
    for &(idx, p) in &exp_probs {
        acc += p;
        if acc >= r {
            return idx;
        }
    }
    exp_probs.last().map(|&(idx, _)| idx).unwrap_or(0)
}

fn load_or_init_pass_brain<B: FlashAttention>(
    pass_name: &str,
    pass_ckpt: Option<&PathBuf>,
    max_seq_len: usize,
    device: &B::Device,
) -> Result<YumonMoeBrain<B>> {
    if let Some(ckpt) = pass_ckpt {
        if ckpt.exists() {
            println!("[PaintStudio] Loading {} checkpoint from {}...", pass_name, ckpt.display());
            let (model, _, _) = YumonMoeBrain::<B>::load(ckpt.to_str().unwrap(), device)?;
            return Ok(model);
        }
    }
    println!("[PaintStudio] No {} checkpoint found — initializing compact untrained MoE...", pass_name);
    let cfg = YumonMoeBrainConfig::new(ART_VOCAB_SIZE, TrainingStage::Language)
        .with_embed_dim(64)
        .with_hidden_units(64)
        .with_n_layers(4)
        .with_attn_heads(4)
        .with_ff_dim(256)
        .with_max_seq_len(max_seq_len)
        .with_num_experts(4)
        .with_top_k(1)
        .with_is_multimodal(false);
    Ok(cfg.init(device))
}

/// The stateful generative Paint Studio managing all 5 cascaded vision models.
pub struct PaintStudio<B: FlashAttention> {
    pub vocab: KiloVocab,
    pub brain_p1: YumonMoeBrain<B>,
    pub brain_p2: Option<YumonMoeBrain<B>>,
    pub brain_p3: Option<YumonMoeBrain<B>>,
    pub brain_p4: Option<YumonMoeBrain<B>>,
    pub upscaler: Option<UpscalerModel<B>>,
    pub colorizer: Option<ColorizerModel<B>>,
    pub device: B::Device,
}

impl<B: FlashAttention> PaintStudio<B> {
    /// Loads vocabulary and models according to provided or discovered paths.
    pub fn load(paths: &PaintPaths, device: &B::Device) -> Result<Self> {
        let vocab = KiloVocab::load_from_json(&paths.vocab_path)
            .with_context(|| format!("loading vocabulary from {}", paths.vocab_path.display()))?;

        let brain_p1 = load_or_init_pass_brain(
            "Pass 1",
            paths.checkpoint_pass1.as_ref(),
            128,
            device,
        )?;

        let brain_p2 = if paths.checkpoint_pass2.is_some() {
            Some(load_or_init_pass_brain(
                "Pass 2",
                paths.checkpoint_pass2.as_ref(),
                512,
                device,
            )?)
        } else {
            None
        };

        let brain_p3 = if paths.checkpoint_pass3.is_some() {
            Some(load_or_init_pass_brain(
                "Pass 3",
                paths.checkpoint_pass3.as_ref(),
                1536,
                device,
            )?)
        } else {
            None
        };

        let brain_p4 = if paths.checkpoint_pass4.is_some() {
            Some(load_or_init_pass_brain(
                "Pass 4",
                paths.checkpoint_pass4.as_ref(),
                5500,
                device,
            )?)
        } else {
            None
        };

        let upscaler = if let Some(ckpt_dir) = &paths.upscaler_checkpoint {
            let model_path = ckpt_dir.join("model");
            if model_path.exists() || ckpt_dir.join("model.bin").exists() || ckpt_dir.join("model.mpk").exists() {
                println!("[PaintStudio] Loading Neural Upscaler from {}...", ckpt_dir.display());
                let channels = if let Ok(meta_str) = fs::read_to_string(ckpt_dir.join("metadata.json")) {
                    if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str) {
                        meta.get("channels").and_then(|c| c.as_u64()).unwrap_or(32) as usize
                    } else { 32 }
                } else { 32 };

                let config = UpscalerModelConfig::new().with_channels(channels);
                let model: UpscalerModel<B> = config.init(device);
                let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
                match model.load_file(&model_path, &recorder, device) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("[PaintStudio] Warning: Failed to load upscaler weights: {e}. Will fallback to nearest-neighbor.");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let colorizer = if let Some(ckpt_dir) = &paths.colorizer_checkpoint {
            let model_path = ckpt_dir.join("model");
            if model_path.exists() || ckpt_dir.join("model.bin").exists() || ckpt_dir.join("model.mpk").exists() {
                println!("[PaintStudio] Loading Pass 6 Colorizer (Zhang 2016) from {}...", ckpt_dir.display());
                let channels = if let Ok(meta_str) = fs::read_to_string(ckpt_dir.join("metadata.json")) {
                    if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str) {
                        meta.get("channels").and_then(|c| c.as_u64()).unwrap_or(64) as usize
                    } else { 64 }
                } else { 64 };

                let config = ColorizerModelConfig::new()
                    .with_scales_from(256)
                    .with_channels(channels)
                    .with_num_res_blocks(5);
                let model: ColorizerModel<B> = config.init(device);
                let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
                match model.load_file(&model_path, &recorder, device) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("[PaintStudio] Warning: Failed to load colorizer weights: {e}. Will remain grayscale.");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            vocab,
            brain_p1,
            brain_p2,
            brain_p3,
            brain_p4,
            upscaler,
            colorizer,
            device: device.clone(),
        })
    }

    /// Runs cascaded multi-pass generation, calling `on_progress` after each pass completes.
    pub fn generate<F>(
        &self,
        config: &PaintConfig,
        mut on_progress: F,
    ) -> Result<PaintResult>
    where
        F: FnMut(PassProgress),
    {
        let t_total = Instant::now();
        let total_passes = config.passes.clamp(1, 5);

        // 1. Tokenize prompt
        let mut ctx = TranslationContext::new();
        let prompt_tokens = self.vocab.encode(&config.prompt, &mut ctx)?;

        // ── PASS 1: Broad Composition (8x8 Blocks, 64 tokens) ──
        let t_p1 = Instant::now();
        let mut p1_prefix = vec![BOS_TOKEN as i32];
        for &tok in &prompt_tokens {
            p1_prefix.push(tok as i32);
        }
        p1_prefix.push(PASS_1_ID as i32);

        let p1_coords = build_cascaded_inference_coords(&p1_prefix, 1, 64);
        let mut cache_p1 = self.brain_p1.new_kv_cache_mrope(&[p1_coords], &self.device);
        let prefill_tensor = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(p1_prefix.clone(), [1, p1_prefix.len()]),
            &self.device,
        );
        let logits = self.brain_p1.forward_cached(prefill_tensor, &mut cache_p1);
        let mut last_logit: Vec<f32> = logits
            .slice([0..1, p1_prefix.len() - 1..p1_prefix.len(), 0..ART_VOCAB_SIZE])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut pass1_colors = Vec::with_capacity(64);
        let mut p1_last_color = None;
        let mut p1_consecutive = 0;
        for _ in 0..64 {
            let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let best_palette_idx = sample_color_token(
                color_logits,
                config.temperature,
                config.top_k,
                config.top_p,
                p1_last_color,
                p1_consecutive,
                config.repeat_penalty,
            ) as u8;
            if Some(best_palette_idx as usize) == p1_last_color {
                p1_consecutive += 1;
            } else {
                p1_last_color = Some(best_palette_idx as usize);
                p1_consecutive = 1;
            }
            let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
            pass1_colors.push(next_token);

            let step_tensor = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(vec![next_token as i32], [1, 1]),
                &self.device,
            );
            let step_logits = self.brain_p1.forward_cached(step_tensor, &mut cache_p1);
            last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
        }

        let mut current_canvas_hwc = pass1_tokens_to_image(&pass1_colors);
        let p1_url = rgb_to_data_url(&current_canvas_hwc, CANVAS_SIZE as u32, CANVAS_SIZE as u32)?;
        on_progress(PassProgress {
            pass: 1,
            total_passes,
            stage_name: "Composition".into(),
            description: "Global composition established (64 blocks)".into(),
            width: CANVAS_SIZE as u32,
            height: CANVAS_SIZE as u32,
            data_url: p1_url.clone(),
            elapsed_secs: t_p1.elapsed().as_secs_f32(),
        });

        // ── PASS 2: Shapes & Silhouettes (16x16 Patches, 256 tokens) ──
        let mut pass2_colors = Vec::with_capacity(256);
        if total_passes >= 2 {
            if let Some(brain_p2) = &self.brain_p2 {
                let t_p2 = Instant::now();
                let mut p2_prefix = vec![BOS_TOKEN as i32];
                for &tok in &prompt_tokens {
                    p2_prefix.push(tok as i32);
                }
                p2_prefix.push(PASS_1_ID as i32);
                for &tok in &pass1_colors {
                    p2_prefix.push(tok as i32);
                }
                p2_prefix.push(PASS_2_ID as i32);

                let p2_coords = build_cascaded_inference_coords(&p2_prefix, 2, 256);
                let mut cache_p2 = brain_p2.new_kv_cache_mrope(&[p2_coords], &self.device);
                let prefill_tensor = Tensor::<B, 2, Int>::from_ints(
                    TensorData::new(p2_prefix.clone(), [1, p2_prefix.len()]),
                    &self.device,
                );
                let logits = brain_p2.forward_cached(prefill_tensor, &mut cache_p2);
                let mut last_logit: Vec<f32> = logits
                    .slice([0..1, p2_prefix.len() - 1..p2_prefix.len(), 0..ART_VOCAB_SIZE])
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();

                let mut p2_last_color = None;
                let mut p2_consecutive = 0;
                for _ in 0..256 {
                    let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
                    let best_palette_idx = sample_color_token(
                        color_logits,
                        config.temperature,
                        config.top_k,
                        config.top_p,
                        p2_last_color,
                        p2_consecutive,
                        config.repeat_penalty,
                    ) as u8;
                    if Some(best_palette_idx as usize) == p2_last_color {
                        p2_consecutive += 1;
                    } else {
                        p2_last_color = Some(best_palette_idx as usize);
                        p2_consecutive = 1;
                    }
                    let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
                    pass2_colors.push(next_token);

                    let step_tensor = Tensor::<B, 2, Int>::from_ints(
                        TensorData::new(vec![next_token as i32], [1, 1]),
                        &self.device,
                    );
                    let step_logits = brain_p2.forward_cached(step_tensor, &mut cache_p2);
                    last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
                }

                current_canvas_hwc = pass2_tokens_to_image(&pass2_colors);
                let p2_url = rgb_to_data_url(&current_canvas_hwc, CANVAS_SIZE as u32, CANVAS_SIZE as u32)?;
                on_progress(PassProgress {
                    pass: 2,
                    total_passes,
                    stage_name: "Silhouette".into(),
                    description: "Subject shapes & silhouettes carved (256 patches)".into(),
                    width: CANVAS_SIZE as u32,
                    height: CANVAS_SIZE as u32,
                    data_url: p2_url,
                    elapsed_secs: t_p2.elapsed().as_secs_f32(),
                });
            }
        }

        // ── PASS 3: Anatomy & Features (32x32 Sub-Patches, 1,024 tokens) ──
        let mut pass3_colors = Vec::with_capacity(1024);
        if total_passes >= 3 {
            if let Some(brain_p3) = &self.brain_p3 {
                let t_p3 = Instant::now();
                let mut p3_prefix = vec![BOS_TOKEN as i32];
                for &tok in &prompt_tokens {
                    p3_prefix.push(tok as i32);
                }
                p3_prefix.push(PASS_2_ID as i32);
                for &tok in &pass2_colors {
                    p3_prefix.push(tok as i32);
                }
                p3_prefix.push(PASS_3_ID as i32);

                let p3_coords = build_cascaded_inference_coords(&p3_prefix, 3, 1024);
                let mut cache_p3 = brain_p3.new_kv_cache_mrope(&[p3_coords], &self.device);
                let prefill_tensor = Tensor::<B, 2, Int>::from_ints(
                    TensorData::new(p3_prefix.clone(), [1, p3_prefix.len()]),
                    &self.device,
                );
                let logits = brain_p3.forward_cached(prefill_tensor, &mut cache_p3);
                let mut last_logit: Vec<f32> = logits
                    .slice([0..1, p3_prefix.len() - 1..p3_prefix.len(), 0..ART_VOCAB_SIZE])
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();

                let mut p3_last_color = None;
                let mut p3_consecutive = 0;
                for i in 0..1024 {
                    let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
                    let py = i / 32;
                    let px = i % 32;
                    let patch_idx = (py / 2) * 16 + (px / 2);

                    let mut working_logits = color_logits.to_vec();
                    if config.shape_guidance > 0.0 && patch_idx < pass2_colors.len() {
                        let shape_token = pass2_colors[patch_idx];
                        let shape_color = (shape_token as usize).saturating_sub(COLOR_START_ID);
                        if shape_color < PALETTE_SIZE {
                            working_logits[shape_color] += config.shape_guidance;
                        }
                    }

                    let best_palette_idx = sample_color_token(
                        &working_logits,
                        config.temperature,
                        config.top_k,
                        config.top_p,
                        p3_last_color,
                        p3_consecutive,
                        config.repeat_penalty,
                    ) as u8;

                    if Some(best_palette_idx as usize) == p3_last_color {
                        p3_consecutive += 1;
                    } else {
                        p3_last_color = Some(best_palette_idx as usize);
                        p3_consecutive = 1;
                    }

                    let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
                    pass3_colors.push(next_token);

                    let step_tensor = Tensor::<B, 2, Int>::from_ints(
                        TensorData::new(vec![next_token as i32], [1, 1]),
                        &self.device,
                    );
                    let step_logits = brain_p3.forward_cached(step_tensor, &mut cache_p3);
                    last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
                }

                current_canvas_hwc = pass3_tokens_to_image(&pass3_colors);
                let p3_url = rgb_to_data_url(&current_canvas_hwc, CANVAS_SIZE as u32, CANVAS_SIZE as u32)?;
                on_progress(PassProgress {
                    pass: 3,
                    total_passes,
                    stage_name: "Features & Anatomy".into(),
                    description: "Fine pixel contours & shading resolved (1,024 pixels)".into(),
                    width: CANVAS_SIZE as u32,
                    height: CANVAS_SIZE as u32,
                    data_url: p3_url,
                    elapsed_secs: t_p3.elapsed().as_secs_f32(),
                });
            }
        }

        // ── PASS 4: Fine Grayscale Details (64x64 Pixels, 4,096 tokens) ──
        let mut pass4_colors = Vec::with_capacity(4096);
        if total_passes >= 4 {
            if let Some(brain_p4) = &self.brain_p4 {
                let t_p4 = Instant::now();
                let mut p4_prefix = vec![BOS_TOKEN as i32];
                for &tok in &prompt_tokens {
                    p4_prefix.push(tok as i32);
                }
                p4_prefix.push(PASS_3_ID as i32);
                for &tok in &pass3_colors {
                    p4_prefix.push(tok as i32);
                }
                p4_prefix.push(PASS_4_ID as i32);

                let p4_coords = build_cascaded_inference_coords(&p4_prefix, 4, 4096);
                let mut cache_p4 = brain_p4.new_kv_cache_mrope(&[p4_coords], &self.device);
                let prefill_tensor = Tensor::<B, 2, Int>::from_ints(
                    TensorData::new(p4_prefix.clone(), [1, p4_prefix.len()]),
                    &self.device,
                );
                let logits = brain_p4.forward_cached(prefill_tensor, &mut cache_p4);
                let mut last_logit: Vec<f32> = logits
                    .slice([0..1, p4_prefix.len() - 1..p4_prefix.len(), 0..ART_VOCAB_SIZE])
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();

                let mut p4_last_color = None;
                let mut p4_consecutive = 0;
                for i in 0..4096 {
                    let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
                    let py = i / 64;
                    let px = i % 64;
                    let sub_idx = (py / 2) * 32 + (px / 2);

                    let mut working_logits = color_logits.to_vec();
                    if config.shape_guidance > 0.0 && sub_idx < pass3_colors.len() {
                        let sub_token = pass3_colors[sub_idx];
                        let sub_color = (sub_token as usize).saturating_sub(COLOR_START_ID);
                        if sub_color < PALETTE_SIZE {
                            working_logits[sub_color] += config.shape_guidance;
                        }
                    }

                    let best_palette_idx = sample_color_token(
                        &working_logits,
                        config.temperature,
                        config.top_k,
                        config.top_p,
                        p4_last_color,
                        p4_consecutive,
                        config.repeat_penalty,
                    ) as u8;

                    if Some(best_palette_idx as usize) == p4_last_color {
                        p4_consecutive += 1;
                    } else {
                        p4_last_color = Some(best_palette_idx as usize);
                        p4_consecutive = 1;
                    }

                    let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
                    pass4_colors.push(next_token);

                    let step_tensor = Tensor::<B, 2, Int>::from_ints(
                        TensorData::new(vec![next_token as i32], [1, 1]),
                        &self.device,
                    );
                    let step_logits = brain_p4.forward_cached(step_tensor, &mut cache_p4);
                    last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
                }

                current_canvas_hwc = pass4_tokens_to_image(&pass4_colors);
                let p4_url = rgb_to_data_url(&current_canvas_hwc, CANVAS_SIZE as u32, CANVAS_SIZE as u32)?;
                on_progress(PassProgress {
                    pass: 4,
                    total_passes,
                    stage_name: "Fine Details".into(),
                    description: "High-resolution 64x64 grayscale pixels synthesized (4,096 tokens)".into(),
                    width: CANVAS_SIZE as u32,
                    height: CANVAS_SIZE as u32,
                    data_url: p4_url,
                    elapsed_secs: t_p4.elapsed().as_secs_f32(),
                });
            }
        }

        // ── PASS 5: Neural Super-Resolution (64x64 -> 256x256 Grayscale) ──
        let mut final_canvas_hwc = current_canvas_hwc.clone();
        let mut out_w = CANVAS_SIZE as u32;
        let mut out_h = CANVAS_SIZE as u32;
        let mut cached_final_data_url: Option<String> = None;

        if total_passes >= 5 {
            let t_p5 = Instant::now();
            if let Some(upscaler) = &self.upscaler {
                let plane = CANVAS_SIZE * CANVAS_SIZE; // 4096
                let mut flat = vec![0.0f32; plane];
                if !pass4_colors.is_empty() {
                    for (i, &tok) in pass4_colors.iter().take(plane).enumerate() {
                        let c = (tok as usize).saturating_sub(COLOR_START_ID).min(PALETTE_SIZE - 1) as u8;
                        let lum = palette_to_rgb(c)[0];
                        flat[i] = (lum as f32 / 255.0) * 2.0 - 1.0;
                    }
                } else {
                    for y in 0..CANVAS_SIZE {
                        for x in 0..CANVAS_SIZE {
                            let idx = (y * CANVAS_SIZE + x) * 3;
                            flat[y * CANVAS_SIZE + x] = (current_canvas_hwc[idx] as f32 / 255.0) * 2.0 - 1.0;
                        }
                    }
                }

                let input_tensor = Tensor::<B, 4>::from_floats(
                    TensorData::new(flat, [1, 1, CANVAS_SIZE, CANVAS_SIZE]),
                    &self.device,
                );
                let out_tensor = upscaler.forward(input_tensor);
                let out_data: Vec<f32> = out_tensor.into_data().convert::<f32>().to_vec().unwrap();
                let mut rgb_256 = vec![0u8; 256 * 256 * 3];
                for y in 0..256 {
                    for x in 0..256 {
                        let idx = y * 256 + x;
                        let val = (((out_data[idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let out_idx = (y * 256 + x) * 3;
                        rgb_256[out_idx] = val;
                        rgb_256[out_idx + 1] = val;
                        rgb_256[out_idx + 2] = val;
                    }
                }
                final_canvas_hwc = rgb_256;
                out_w = 256;
                out_h = 256;
            } else {
                final_canvas_hwc = upscale_nn(&current_canvas_hwc, CANVAS_SIZE, CANVAS_SIZE, 4);
                out_w = 256;
                out_h = 256;
            }

            let p5_url = rgb_to_data_url(&final_canvas_hwc, out_w, out_h)?;
            on_progress(PassProgress {
                pass: 5,
                total_passes,
                stage_name: "Super-Resolution".into(),
                description: "Deep neural 256x256 grayscale super-resolution synthesized".into(),
                width: out_w,
                height: out_h,
                data_url: p5_url.clone(),
                elapsed_secs: t_p5.elapsed().as_secs_f32(),
            });
            cached_final_data_url = Some(p5_url);
        }

        // ── PASS 6: Zhang 2016 Multinomial Colorization (256x256 Gray -> 256x256 RGB) ──
        if total_passes >= 6 {
            let t_p6 = Instant::now();
            if let Some(colorizer) = &self.colorizer {
                let mut raw_l = vec![0u8; 256 * 256];
                let mut flat_in = vec![0.0f32; 256 * 256];
                for i in 0..256 * 256 {
                    let val = final_canvas_hwc[i * 3]; // R = G = B from grayscale
                    raw_l[i] = val;
                    flat_in[i] = (val as f32 / 255.0) * 2.0 - 1.0;
                }

                let in_tensor = Tensor::<B, 4>::from_floats(
                    TensorData::new(flat_in, [1, 1, 256, 256]),
                    &self.device,
                );
                let (pred_a, pred_b) = colorizer.predict_ab_annealed(in_tensor, DEFAULT_TEMPERATURE);
                let color_img = colorize_fused_rgb(&raw_l, &pred_a, &pred_b, 256);
                final_canvas_hwc = color_img.into_raw();
                out_w = 256;
                out_h = 256;
            }

            let p6_url = rgb_to_data_url(&final_canvas_hwc, out_w, out_h)?;
            on_progress(PassProgress {
                pass: 6,
                total_passes,
                stage_name: "Zhang Colorization".into(),
                description: "Zhang et al. (2016) multinomial colorization synthesized vibrant RGB artwork".into(),
                width: out_w,
                height: out_h,
                data_url: p6_url.clone(),
                elapsed_secs: t_p6.elapsed().as_secs_f32(),
            });
            cached_final_data_url = Some(p6_url);
        }

        // ── Save Artwork & JSON Metadata to inference-cache ──
        let now = Local::now();
        let day_dir = now.format("%Y-%m-%d").to_string();
        let hour_dir = now.format("%H").to_string();
        let timestamp_str = now.format("%Y-%m-%d_%H-%M-%S").to_string();

        let cache_dir = PathBuf::from("inference-cache").join(&day_dir).join(&hour_dir);
        fs::create_dir_all(&cache_dir).context("creating inference-cache subfolder")?;

        // Slugify prompt for filename
        let prompt_slug: String = config
            .prompt
            .chars()
            .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
            .collect::<String>()
            .split('_')
            .filter(|s| !s.is_empty())
            .take(5)
            .collect::<Vec<_>>()
            .join("_");
        let slug = if prompt_slug.is_empty() { "artwork".to_string() } else { prompt_slug };

        let raw_filename = format!("{slug}_64_{timestamp_str}.png");
        let final_filename = format!("{slug}_{timestamp_str}.png");
        let meta_filename = format!("{slug}_{timestamp_str}.json");

        let raw_path = cache_dir.join(&raw_filename);
        let final_path = cache_dir.join(&final_filename);
        let meta_path = cache_dir.join(&meta_filename);

        // Save raw 64x64 buffer
        let raw_img: image::RgbImage = image::ImageBuffer::from_raw(
            CANVAS_SIZE as u32,
            CANVAS_SIZE as u32,
            current_canvas_hwc,
        ).context("saving raw 64x64 ImageBuffer")?;
        raw_img.save(&raw_path)?;

        // Save final upscaled buffer
        let final_img: image::RgbImage = image::ImageBuffer::from_raw(
            out_w,
            out_h,
            final_canvas_hwc.clone(),
        ).context("saving final ImageBuffer")?;
        final_img.save(&final_path)?;

        // Save metadata
        let metadata = serde_json::json!({
            "prompt": config.prompt,
            "timestamp": now.to_rfc3339(),
            "passes": total_passes,
            "width": out_w,
            "height": out_h,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "raw_file": raw_path.to_string_lossy(),
            "final_file": final_path.to_string_lossy(),
        });
        fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

        let final_data_url = match cached_final_data_url {
            Some(url) => url,
            None => rgb_to_data_url(&final_canvas_hwc, out_w, out_h)?,
        };
        let thumbnail_url = rgb_to_thumbnail_data_url(&final_canvas_hwc, out_w, out_h, 80)?;

        Ok(PaintResult {
            final_path,
            raw_path,
            metadata_path: meta_path,
            prompt: config.prompt.clone(),
            final_data_url,
            thumbnail_url,
            width: out_w,
            height: out_h,
            duration: t_total.elapsed(),
        })
    }
}

/// Recursively scans `inference-cache/` for past creations to populate the gallery.
pub fn scan_gallery(cache_dir: &Path, max_items: usize) -> Vec<GalleryItem> {
    if !cache_dir.exists() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    for entry in walkdir::WalkDir::new(cache_dir)
        .max_depth(4)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("png") {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            // Skip 64x64 intermediate raw images
            if filename.contains("_64_") || filename.starts_with('.') {
                continue;
            }
            if let Ok(metadata) = entry.metadata() {
                entries.push((path.to_path_buf(), metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)));
            }
        }
    }

    // Sort newest first
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(max_items);

    let mut items = Vec::new();
    for (path, _) in entries {
        let meta_candidate = path.with_extension("json");
        let (prompt, timestamp, w, h) = if meta_candidate.exists() {
            if let Ok(content) = fs::read_to_string(&meta_candidate) {
                if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) {
                    let p = meta.get("prompt").and_then(|s| s.as_str()).unwrap_or("Untitled Artwork").to_string();
                    let t = meta.get("timestamp").and_then(|s| s.as_str()).unwrap_or("").to_string();
                    let width = meta.get("width").and_then(|n| n.as_u64()).unwrap_or(256) as u32;
                    let height = meta.get("height").and_then(|n| n.as_u64()).unwrap_or(256) as u32;
                    (p, t, width, height)
                } else {
                    fallback_meta_from_filename(&path)
                }
            } else {
                fallback_meta_from_filename(&path)
            }
        } else {
            fallback_meta_from_filename(&path)
        };

        if let Ok(thumbnail_url) = file_to_thumbnail_data_url(&path, 80) {
            let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("art").to_string();
            items.push(GalleryItem {
                id,
                path: path.to_string_lossy().to_string(),
                raw_path: None,
                prompt,
                timestamp,
                width: w,
                height: h,
                thumbnail_url,
            });
        }
    }

    items
}

fn fallback_meta_from_filename(path: &Path) -> (String, String, u32, u32) {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("artwork");
    let parts: Vec<&str> = stem.split('_').collect();
    let prompt = if parts.len() > 1 {
        parts[..parts.len() - 1].join(" ")
    } else {
        stem.to_string()
    };
    (prompt, String::new(), 256, 256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_data_url_valid_png() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let url = rgb_to_data_url(&rgb, 2, 2).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        assert!(url.len() > 30);
    }

    #[test]
    fn test_paint_paths_discovery() {
        let paths = PaintPaths::auto_discover();
        assert!(paths.vocab_path.exists());
        if let Some(p) = &paths.upscaler_checkpoint {
            assert!(p.exists());
        }
    }

    #[test]
    fn test_scan_gallery_existing() {
        let cache_dir = Path::new("inference-cache");
        if cache_dir.exists() {
            let gallery = scan_gallery(cache_dir, 10);
            assert!(!gallery.is_empty(), "Gallery should discover existing artworks");
            assert!(gallery[0].thumbnail_url.starts_with("data:image/png;base64,"));
        }
    }
}
