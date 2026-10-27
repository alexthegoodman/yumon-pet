//! Preprocess art images and JSON manifests into Kilo Talk tokenized
//! canvas progression training pairs stored in `diffusion_kilo_samples.bin`.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    indicatif::{ProgressBar, ProgressStyle},
    rand::Rng,
    rayon::prelude::*,
    serde::{Deserialize, Serialize},
    std::{
        collections::HashMap,
        fs::File,
        io::{BufReader, BufWriter, Write},
        path::PathBuf,
    },
    yumon_pet::brain::{
        kilo_vocab::KiloVocab,
        yumon_talk::TranslationContext,
    },
    yumon_pet::vision::paint_action::{
        ART_VOCAB_SIZE, CANVAS_SIZE, FaceRoi, FaceRoiDatabase, compute_pass_spatial_weights,
        generate_cascaded_sequence_samples, ConditioningNoiseConfig, corrupt_sample_tokens,
    },
};

pub const ART_CACHE_MAGIC: &[u8; 8] = b"YUMONART";
pub const ART_CACHE_VERSION: u32 = 4;

#[derive(Serialize, Deserialize, Debug)]
pub struct ArtCacheMetadata {
    pub version: u32,
    pub total_samples: usize,
    pub images_processed: usize,
    pub canvas_size: usize,
    pub vocab_size: usize,
    pub trajectory_steps: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArtSample {
    pub prompt_text: String,
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
    #[serde(default)]
    pub spatial_weights: Option<Vec<f32>>,
}

#[derive(Deserialize, Debug)]
struct ManifestItem {
    p: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Convert art manifests and images into Kilo Talk multimodal art training cache")]
struct Args {
    /// Directory or directories containing art images and part-*.json manifests (comma-separated or repeated).
    #[arg(long = "images-dir", alias = "images-dirs", value_delimiter = ',', num_args = 1..)]
    images_dir: Vec<PathBuf>,

    /// Path to Kilo Talk vocabulary JSON file.
    #[arg(long, default_value = "training-cache/diffusion_talk_vocab.json")]
    vocab: PathBuf,

    /// Output binary cache path.
    #[arg(long, default_value = "training-cache/diffusion_kilo_samples.bin")]
    output: PathBuf,

    /// Limit number of images to convert (None = all).
    #[arg(long)]
    limit: Option<usize>,

    /// Path to face & eye ROI annotations JSON file (optional).
    #[arg(long, default_value = "training-cache/diffusion_face_rois.json")]
    rois: Option<PathBuf>,

    /// Filter to a specific pass (1, 2, 3, or 4). If omitted, caches all passes.
    #[arg(long)]
    pass: Option<usize>,

    /// Number of training samples per image across canvas progression.
    #[arg(long, default_value_t = 8)]
    samples_per_image: usize,

    /// Number of consecutive pixels to predict per sample.
    #[arg(long, default_value_t = 16)]
    chunk_len: usize,

    /// Number of sample previews to print upon completion.
    #[arg(long, default_value_t = 3)]
    previews: usize,

    /// Fraction of conditioning samples for Passes 2-4 that receive synthetic noise (0.0 = all clean, 0.5 = 50% clean / 50% noisy, 1.0 = all noisy).
    #[arg(long, default_value_t = 0.5)]
    noisy_ratio: f32,

    /// If true, generates BOTH a clean sample AND a noisy sample for each image in Passes 2-4 (exact 50/50 mix, doubling training pairs).
    #[arg(long, default_value_t = false)]
    augment_noisy: bool,

    /// Probability of perturbing a conditioning token with RGB color jitter (0.0 to 1.0).
    #[arg(long, default_value_t = 0.20)]
    noise_jitter_prob: f32,

    /// Maximum RGB jitter magnitude (0.0 to 255.0) for color perturbation.
    #[arg(long, default_value_t = 35.0)]
    noise_jitter_mag: f32,

    /// Probability of a conditioning token adopting a spatial neighbor's color (simulates stroke bleed / boundary drift).
    #[arg(long, default_value_t = 0.10)]
    noise_drift_prob: f32,

    /// Probability of a conditioning token mutating to a random palette color (salt-and-pepper / rogue dabs).
    #[arg(long, default_value_t = 0.02)]
    noise_random_prob: f32,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let images_dirs = if args.images_dir.is_empty() {
        vec![PathBuf::from("data/animals_images/images-64")]
    } else {
        args.images_dir
    };

    println!("=== Yumon Art: Art Images -> Kilo Talk Pixel Stream Preprocessor ===");
    for dir in &images_dirs {
        println!("Images directory   : {}", dir.display());
    }
    println!("Vocabulary file    : {}", args.vocab.display());
    println!("Output cache       : {}", args.output.display());
    if let Some(p) = args.pass {
        println!("Pass filter        : Pass {}", p);
    } else {
        println!("Pass filter        : All passes (1, 2, 3, 4)");
    }
    println!("\nSynthetic Conditioning Noise Configuration:");
    if args.noisy_ratio > 0.0 || args.augment_noisy {
        if args.augment_noisy {
            println!("  Mode             : Augment (1 clean + 1 noisy copy for each image in Passes 2-4)");
        } else {
            println!("  Noisy ratio      : {:.1}% probabilistic mix", args.noisy_ratio * 100.0);
        }
        println!("  RGB jitter prob  : {:.1}% (magnitude: {:.1})", args.noise_jitter_prob * 100.0, args.noise_jitter_mag);
        println!("  Neighbor drift   : {:.1}% (spatial stroke bleed / edge drift)", args.noise_drift_prob * 100.0);
        println!("  Random token     : {:.1}% (rogue / high-temp palette mutations)", args.noise_random_prob * 100.0);
    } else {
        println!("  Conditioning     : 100% Clean Ground Truth (noise disabled)");
    }

    // 1. Load Kilo Talk vocabulary
    println!("\nLoading Kilo Talk vocabulary...");
    let vocab = KiloVocab::load_from_json(&args.vocab)
        .with_context(|| format!("loading vocabulary from {}", args.vocab.display()))?;
    println!("Vocabulary loaded (size: {})", vocab.vocab_size);

    // 2. Discover manifest files across all specified directories
    let mut manifest_files: Vec<(PathBuf, PathBuf)> = Vec::new();
    for dir in &images_dirs {
        ensure!(dir.exists(), "Images directory not found: {}", dir.display());
        let mut dir_manifests = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                dir_manifests.push((dir.clone(), path));
            }
        }
        dir_manifests.sort_by(|a, b| a.1.cmp(&b.1));
        println!("Found {} manifest JSON files in {}", dir_manifests.len(), dir.display());
        manifest_files.extend(dir_manifests);
    }
    println!("Total manifest JSON files found: {}", manifest_files.len());

    if let Some(p) = args.pass {
        anyhow::ensure!(p >= 1 && p <= 4, "Invalid pass {p}; must be 1, 2, 3, or 4");
    }

    // 3. Load face & eye ROI annotations if provided
    let rois_map: HashMap<String, FaceRoi> = if let Some(ref path) = args.rois {
        if path.exists() {
            println!("\nLoading face & eye ROI annotations from {}...", path.display());
            match File::open(path) {
                Ok(f) => {
                    let rdr = BufReader::new(f);
                    let db: Result<FaceRoiDatabase, _> = serde_json::from_reader(rdr);
                    match db {
                        Ok(db) => {
                            let with_eyes = db
                                .annotations
                                .values()
                                .filter(|r| !r.no_face && (!r.eyes.is_empty() || r.face.is_some()))
                                .count();
                            let skipped = db.annotations.values().filter(|r| r.no_face).count();
                            println!(
                                "Loaded {} total ROI records: {} with annotated faces/eyes, {} marked as no-face.",
                                db.annotations.len(),
                                with_eyes,
                                skipped
                            );
                            db.annotations
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to parse ROI database: {}", e);
                            HashMap::new()
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to open ROI file: {}", e);
                    HashMap::new()
                }
            }
        } else {
            println!("ROI annotations file not found at {}; proceeding with uniform weights.", path.display());
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    // 4. Collect image items from manifests (prioritizing annotated faces first)
    println!("\nScanning manifests for available images...");
    let mut annotated_face_items: Vec<(PathBuf, String)> = Vec::new();
    let mut other_annotated_items: Vec<(PathBuf, String)> = Vec::new();
    let mut unannotated_items: Vec<(PathBuf, String)> = Vec::new();

    for (dir, manifest_path) in &manifest_files {
        let file = File::open(manifest_path)?;
        let reader = BufReader::new(file);
        let manifest: HashMap<String, ManifestItem> = serde_json::from_reader(reader)
            .with_context(|| format!("reading manifest {}", manifest_path.display()))?;

        for (filename, item) in manifest {
            let img_path = dir.join(&filename);
            if img_path.exists() {
                if let Some(roi) = rois_map.get(&filename) {
                    if !roi.no_face && (!roi.eyes.is_empty() || roi.face.is_some()) {
                        annotated_face_items.push((img_path, item.p));
                    } else {
                        other_annotated_items.push((img_path, item.p));
                    }
                } else {
                    unannotated_items.push((img_path, item.p));
                }
            }
        }
    }

    let total_available = annotated_face_items.len() + other_annotated_items.len() + unannotated_items.len();
    println!("Found {} available images across all manifests:", total_available);
    println!("  - {} with annotated face/eyes", annotated_face_items.len());
    println!("  - {} marked as no-face / skipped", other_annotated_items.len());
    println!("  - {} unannotated", unannotated_items.len());

    let mut items: Vec<(PathBuf, String)> = Vec::new();
    // Priority 1: Annotated face images
    let total_face_rois = annotated_face_items.len();
    items.extend(annotated_face_items);
    // Priority 2: Other annotated images (marked no-face)
    items.extend(other_annotated_items);
    // Priority 3: Unannotated images
    items.extend(unannotated_items);

    if let Some(limit) = args.limit {
        if items.len() > limit {
            items.truncate(limit);
        }
    }

    let final_face_count = items
        .iter()
        .filter(|(path, _)| {
            let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            rois_map.get(filename).is_some_and(|r| !r.no_face && (!r.eyes.is_empty() || r.face.is_some()))
        })
        .count();
    let final_other_count = items.len() - final_face_count;

    println!("\nSelected {} images to process (limit: {:?}):", items.len(), args.limit);
    if total_face_rois > 0 && final_face_count >= total_face_rois {
        println!("  - Priority 1: {} annotated face images (100% of available annotated faces included!)", final_face_count);
    } else {
        println!("  - Priority 1: {} annotated face images", final_face_count);
    }
    println!("  - Fill / other: {} images (unannotated or skipped)", final_other_count);

    ensure!(!items.is_empty(), "No valid image files found to process");

    // 5. Process images and generate cascaded sequence training samples
    let noise_cfg = ConditioningNoiseConfig {
        jitter_prob: args.noise_jitter_prob,
        jitter_mag: args.noise_jitter_mag,
        neighbor_drift_prob: args.noise_drift_prob,
        random_token_prob: args.noise_random_prob,
    };

    let pb = ProgressBar::new(items.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let samples: Vec<ArtSample> = items
        .par_iter()
        .flat_map(|(img_path, prompt_text)| {
            pb.inc(1);

            // Load and verify image
            let img = match image::open(img_path) {
                Ok(img) => img.to_rgb8(),
                Err(_) => return Vec::new(),
            };

            if img.width() != CANVAS_SIZE as u32 || img.height() != CANVAS_SIZE as u32 {
                return Vec::new();
            }

            let raw_hwc = img.into_raw();

            // Check if this image has annotated face/eye ROIs
            let filename = img_path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            let roi = rois_map.get(filename);
            let has_roi = roi.is_some_and(|r| !r.no_face && (!r.eyes.is_empty() || r.face.is_some()));

            // Tokenize prompt into Kilo Talk tokens
            let mut ctx = TranslationContext::new();
            let prompt_toks = match vocab.encode(prompt_text, &mut ctx) {
                Ok(toks) => toks.into_iter().map(|t| t as u32).collect::<Vec<u32>>(),
                Err(_) => return Vec::new(),
            };

            let mut rng = rand::thread_rng();

            // Generate cascaded sequence training samples
            let seq_samples = generate_cascaded_sequence_samples(&prompt_toks, &raw_hwc);

            let mut out = Vec::with_capacity(seq_samples.len() * (if args.augment_noisy { 2 } else { 1 }));
            for (pass, clean_tokens, target_start) in seq_samples {
                if let Some(target_p) = args.pass {
                    if pass != target_p as u8 {
                        continue;
                    }
                }

                let spatial_weights = if has_roi {
                    Some(compute_pass_spatial_weights(roi, pass, 10.0, 3.0, 1.0))
                } else {
                    None
                };

                if pass == 1 {
                    // Pass 1 has no conditioning image tokens (prompt -> 8x8)
                    out.push(ArtSample {
                        prompt_text: prompt_text.clone(),
                        pass,
                        tokens: clean_tokens,
                        target_start,
                        spatial_weights,
                    });
                } else if args.augment_noisy {
                    // 1. Clean sample
                    out.push(ArtSample {
                        prompt_text: prompt_text.clone(),
                        pass,
                        tokens: clean_tokens.clone(),
                        target_start,
                        spatial_weights: spatial_weights.clone(),
                    });

                    // 2. Augmented noisy sample
                    let noisy_tokens = corrupt_sample_tokens(pass, &clean_tokens, &noise_cfg, &mut rng);
                    out.push(ArtSample {
                        prompt_text: prompt_text.clone(),
                        pass,
                        tokens: noisy_tokens,
                        target_start,
                        spatial_weights,
                    });
                } else {
                    let is_noisy = args.noisy_ratio > 0.0 && rng.gen_range(0.0f32..1.0f32) < args.noisy_ratio;
                    let tokens = if is_noisy {
                        corrupt_sample_tokens(pass, &clean_tokens, &noise_cfg, &mut rng)
                    } else {
                        clean_tokens
                    };

                    out.push(ArtSample {
                        prompt_text: prompt_text.clone(),
                        pass,
                        tokens,
                        target_start,
                        spatial_weights,
                    });
                }
            }

            out
        })
        .collect();

    pb.finish_with_message("Done processing images!");

    let weighted_samples = samples.iter().filter(|s| s.spatial_weights.is_some()).count();
    let uniform_samples = samples.len() - weighted_samples;
    let p1_count = samples.iter().filter(|s| s.pass == 1).count();
    let p2_count = samples.iter().filter(|s| s.pass == 2).count();
    let p3_count = samples.iter().filter(|s| s.pass == 3).count();
    let p4_count = samples.iter().filter(|s| s.pass == 4).count();
    println!(
        "\nGenerated {} total training samples from {} images:",
        samples.len(),
        items.len()
    );
    println!("  - Pass 1 (Prompt -> 8x8)          : {}", p1_count);
    println!("  - Pass 2 (8x8 -> 16x16)           : {}", p2_count);
    println!("  - Pass 3 (16x16 -> 32x32)         : {}", p3_count);
    println!("  - Pass 4 (32x32 -> 64x64)         : {}", p4_count);
    println!("  - Samples with spatial ROI weights: {}", weighted_samples);
    println!("  - Samples with uniform weights    : {}", uniform_samples);

    // 5. Write binary cache
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(&args.output)
        .with_context(|| format!("creating output cache {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);

    // Magic + Metadata
    writer.write_all(ART_CACHE_MAGIC)?;
    let metadata = ArtCacheMetadata {
        version: ART_CACHE_VERSION,
        total_samples: samples.len(),
        images_processed: items.len(),
        canvas_size: CANVAS_SIZE,
        vocab_size: ART_VOCAB_SIZE,
        trajectory_steps: args.pass.unwrap_or(0),
    };
    let meta_bytes = bincode::serialize(&metadata)?;
    writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&meta_bytes)?;

    // Serialize each sample
    for s in &samples {
        let sample_bytes = bincode::serialize(s)?;
        writer.write_all(&(sample_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&sample_bytes)?;
    }
    writer.flush()?;

    println!("Cache successfully written to {}", args.output.display());

    // Previews
    println!("\n--- Previews ({} samples) ---", args.previews.min(samples.len()));
    for (i, s) in samples.iter().take(args.previews).enumerate() {
        println!("Sample #{}", i + 1);
        println!("  Pass         : Pass {}", s.pass);
        println!("  Prompt       : {}", s.prompt_text);
        println!("  Total tokens : {}", s.tokens.len());
        println!("  Target start : {}", s.target_start);
        println!("  Target count : {}", s.tokens.len() - s.target_start);
        if let Some(ref sw) = s.spatial_weights {
            let eye_count = sw.iter().filter(|&&w| w >= 9.0).count();
            let face_count = sw.iter().filter(|&&w| w >= 2.5 && w < 9.0).count();
            println!("  Spatial W    : Active (eye tokens: {}, face tokens: {})", eye_count, face_count);
        } else {
            println!("  Spatial W    : Uniform (1.0)");
        }
    }

    Ok(())
}
