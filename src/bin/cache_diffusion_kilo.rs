//! Preprocess art images and JSON manifests into Kilo Talk tokenized
//! canvas progression training pairs stored in `diffusion_kilo_samples.bin`.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    indicatif::{ProgressBar, ProgressStyle},
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
        ART_VOCAB_SIZE, CANVAS_SIZE, generate_multiscale_samples,
        PASS_1_ID, PASS_2_ID, PASS_3_ID,
    },
};

pub const ART_CACHE_MAGIC: &[u8; 8] = b"YUMONART";
pub const ART_CACHE_VERSION: u32 = 2;

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
    pub prompt_tokens: Vec<u32>,
    pub canvas_hwc: Vec<u8>,
    pub action_tokens: Vec<u32>,
    pub is_done: bool,
    pub target_edge: f32,
}

#[derive(Deserialize, Debug)]
struct ManifestItem {
    p: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Convert art manifests and images into Kilo Talk multimodal art training cache")]
struct Args {
    /// Directory containing art images and part-*.json manifests.
    #[arg(long, default_value = "data/animals_images/images-64")]
    images_dir: PathBuf,

    /// Path to Kilo Talk vocabulary JSON file.
    #[arg(long, default_value = "training-cache/diffusion_talk_vocab.json")]
    vocab: PathBuf,

    /// Output binary cache path.
    #[arg(long, default_value = "training-cache/diffusion_kilo_samples.bin")]
    output: PathBuf,

    /// Limit number of images to convert (None = all).
    #[arg(long)]
    limit: Option<usize>,

    /// Filter to a specific pass (1, 2, or 3). If omitted, caches all passes.
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
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Yumon Art: Art Images -> Kilo Talk Pixel Stream Preprocessor ===");
    println!("Images directory   : {}", args.images_dir.display());
    println!("Vocabulary file    : {}", args.vocab.display());
    println!("Output cache       : {}", args.output.display());
    if let Some(p) = args.pass {
        println!("Pass filter        : Pass {}", p);
    } else {
        println!("Pass filter        : All passes (1, 2, 3)");
    }

    // 1. Load Kilo Talk vocabulary
    println!("\nLoading Kilo Talk vocabulary...");
    let vocab = KiloVocab::load_from_json(&args.vocab)
        .with_context(|| format!("loading vocabulary from {}", args.vocab.display()))?;
    println!("Vocabulary loaded (size: {})", vocab.vocab_size);

    // 2. Discover manifest files
    let mut manifest_files = Vec::new();
    for entry in std::fs::read_dir(&args.images_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
            manifest_files.push(path);
        }
    }
    manifest_files.sort();
    println!("Found {} manifest JSON files", manifest_files.len());

    // 3. Collect image items from manifests
    let mut items: Vec<(PathBuf, String)> = Vec::new();
    for manifest_path in &manifest_files {
        let file = File::open(manifest_path)?;
        let reader = BufReader::new(file);
        let manifest: HashMap<String, ManifestItem> = serde_json::from_reader(reader)
            .with_context(|| format!("reading manifest {}", manifest_path.display()))?;

        for (filename, item) in manifest {
            let img_path = args.images_dir.join(&filename);
            if img_path.exists() {
                items.push((img_path, item.p));
            }
            if let Some(limit) = args.limit {
                if items.len() >= limit {
                    break;
                }
            }
        }
        if let Some(limit) = args.limit {
            if items.len() >= limit {
                break;
            }
        }
    }

    println!("Total images to process: {}", items.len());
    ensure!(!items.is_empty(), "No valid image files found in manifest");

    let target_pass_tag = match args.pass {
        Some(1) => Some(PASS_1_ID),
        Some(2) => Some(PASS_2_ID),
        Some(3) => Some(PASS_3_ID),
        Some(p) => anyhow::bail!("Invalid pass {p}; must be 1, 2, or 3"),
        None => None,
    };

    // 4. Process images and generate multiscale painting samples
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

            // Tokenize prompt into Kilo Talk tokens
            let mut ctx = TranslationContext::new();
            let prompt_toks = match vocab.encode(prompt_text, &mut ctx) {
                Ok(toks) => toks.into_iter().map(|t| t as u32).collect::<Vec<u32>>(),
                Err(_) => return Vec::new(),
            };

            // Generate coarse-to-fine multiscale training samples
            let multiscale_samples = generate_multiscale_samples(&raw_hwc);

            let mut out = Vec::with_capacity(multiscale_samples.len());
            for (canvas_hwc, tokens, is_done, target_edge) in multiscale_samples {
                if let Some(target_tag) = target_pass_tag {
                    if tokens.is_empty() || tokens[0] != target_tag {
                        continue;
                    }
                }
                let action_tokens = tokens.into_iter().map(|t| t as u32).collect();

                out.push(ArtSample {
                    prompt_text: prompt_text.clone(),
                    prompt_tokens: prompt_toks.clone(),
                    canvas_hwc,
                    action_tokens,
                    is_done,
                    target_edge,
                });
            }

            out
        })
        .collect();

    pb.finish_with_message("Done processing images!");

    println!("\nGenerated {} total training samples from {} images", samples.len(), items.len());

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
        println!("  Prompt : {}", s.prompt_text);
        println!("  Tokens : {:?}", s.prompt_tokens);
        println!("  Pixel tokens count: {}", s.action_tokens.len());
        println!("  Target edge: {:.4}", s.target_edge);
        println!("  Done   : {}", s.is_done);
    }

    Ok(())
}
