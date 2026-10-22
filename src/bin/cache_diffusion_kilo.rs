//! Preprocess DiffusionDb images and JSON manifests into Kilo Talk tokenized
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
        path::{Path, PathBuf},
    },
    yumon_pet::brain::{
        kilo_vocab::KiloVocab,
        yumon_talk::TranslationContext,
    },
    yumon_pet::vision::paint_action::{
        ART_VOCAB_SIZE, CANVAS_CHANNELS, CANVAS_SIZE, PaintAction,
        generate_painting_trajectory,
    },
};

pub const ART_CACHE_MAGIC: &[u8; 8] = b"YUMONART";
pub const ART_CACHE_VERSION: u32 = 1;

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
}

#[derive(Deserialize, Debug)]
struct ManifestItem {
    p: String,
    #[serde(default)]
    se: Option<u64>,
    #[serde(default)]
    c: Option<f32>,
    #[serde(default)]
    st: Option<u32>,
    #[serde(default)]
    sa: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Convert DiffusionDb manifests and images into Kilo Talk multimodal art training cache")]
struct Args {
    /// Directory containing DiffusionDb images and part-*.json manifests.
    #[arg(long, default_value = "data/DiffusionDb/images-64")]
    images_dir: PathBuf,

    /// Path to Kilo Talk vocabulary JSON file.
    #[arg(long, default_value = "training-cache/kilo_talk_vocab.json")]
    vocab: PathBuf,

    /// Output binary cache path.
    #[arg(long, default_value = "training-cache/diffusion_kilo_samples.bin")]
    output: PathBuf,

    /// Limit number of images to convert (None = all).
    #[arg(long)]
    limit: Option<usize>,

    /// Number of intermediate painting steps per image trajectory.
    #[arg(long, default_value_t = 6)]
    trajectory_steps: usize,

    /// Number of sample previews to print upon completion.
    #[arg(long, default_value_t = 3)]
    previews: usize,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Yumon Art: DiffusionDb -> Kilo Talk Preprocessor ===");
    println!("Images directory : {}", args.images_dir.display());
    println!("Vocabulary file  : {}", args.vocab.display());
    println!("Output cache     : {}", args.output.display());
    println!("Trajectory steps : {}", args.trajectory_steps);

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

    // 4. Process images and generate intermediate painting trajectories
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

            // Generate intermediate painting trajectory
            let trajectory = generate_painting_trajectory(&raw_hwc, args.trajectory_steps);

            let mut out = Vec::with_capacity(trajectory.len());
            for (canvas_hwc, action) in trajectory {
                let action_tokens = action.to_tokens().into_iter().map(|t| t as u32).collect();
                let is_done = matches!(action, PaintAction::Done);

                out.push(ArtSample {
                    prompt_text: prompt_text.clone(),
                    prompt_tokens: prompt_toks.clone(),
                    canvas_hwc,
                    action_tokens,
                    is_done,
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
        trajectory_steps: args.trajectory_steps,
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
        let action = PaintAction::from_tokens(&s.action_tokens.iter().map(|&t| t as usize).collect::<Vec<_>>());
        println!("  Action : {:?}", action);
        println!("  Done   : {}", s.is_done);
    }

    Ok(())
}
