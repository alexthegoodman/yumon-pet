#![recursion_limit = "256"]

/// Yumon ePet — Entry point
///
/// Subcommands:
///   train-vision  — Pre-train CNN on CIFAR-100 (classification) + FER2013 (emote)
///   train-brain   — Pre-train LSTM on SimpleWiki, conditioned on vision embeddings
///   chat          — Interactive inference: provide an image path, get a reply + emote
///   status        — Print saved model info

mod vision;
mod brain;

use clap::{Parser, Subcommand};
use anyhow::Result;
use burn::{backend::Wgpu, prelude::Module};

#[derive(Parser)]
#[command(name = "yumon", about = "Yumon ePet — tabletop AI companion")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train the Vision CNN (CIFAR-100 + FER2013)
    TrainVision {
        #[arg(long, default_value = "data/cifar-100-binary")]
        cifar_dir: String,

        #[arg(long, default_value = "data/fer2013-archive")]
        fer_dir: String,

        #[arg(long, default_value = "checkpoints/vision")]
        out_dir: String,

        #[arg(long, default_value_t = 30)]
        epochs: usize,

        #[arg(long, default_value_t = 64)]
        batch_size: usize,
    },

    /// Pre-train the LSTM language brain on SimpleWiki
    TrainBrain {
        #[arg(long, default_value = "data/simplewiki-latest-pages-articles.xml")]
        wiki_xml: String,

        #[arg(long, default_value = "checkpoints/vision")]
        vision_checkpoint: String,

        #[arg(long, default_value = "checkpoints/brain")]
        out_dir: String,

        #[arg(long, default_value_t = 30)]
        epochs: usize,

        #[arg(long, default_value_t = 32)]
        batch_size: usize,

        /// Maximum wiki articles to load (0 = all)
        #[arg(long, default_value_t = 5_000)]
        max_articles: usize,
    },

    /// Interactive chat — give an image, Yumon replies
    Chat {
        #[arg(long, default_value = "checkpoints/vision")]
        vision_checkpoint: String,

        #[arg(long, default_value = "checkpoints/brain")]
        brain_checkpoint: String,

        /// Path to an image file (JPEG/PNG). Leave empty for text-only mode.
        #[arg(long)]
        image: Option<String>,

        /// Optional text prompt to guide Yumon's reply
        #[arg(long, default_value = "")]
        prompt: String,

        /// Detected user emote override (angry/disgust/fear/happy/neutral/sad/surprise)
        #[arg(long, default_value = "neutral")]
        user_emote: String,
    },

    /// Print saved checkpoint metadata
    Status {
        #[arg(long, default_value = "checkpoints/vision")]
        vision_checkpoint: String,

        #[arg(long, default_value = "checkpoints/brain")]
        brain_checkpoint: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::TrainVision { cifar_dir, fer_dir, out_dir, epochs, batch_size } => {
            println!("🎓 Training Vision CNN...");
            vision::train::run(
                &cifar_dir, &fer_dir, &out_dir, epochs, batch_size,
            )?;
        }

        Command::TrainBrain { wiki_xml, vision_checkpoint, out_dir, epochs, batch_size, max_articles } => {
            println!("🧠 Training LSTM Brain...");
            brain::train::run(
                &wiki_xml, &vision_checkpoint, &out_dir,
                epochs, batch_size, max_articles,
            )?;
        }

        Command::Chat { vision_checkpoint, brain_checkpoint, image, prompt, user_emote } => {
            println!("💬 Yumon Chat Mode");
            run_chat(&vision_checkpoint, &brain_checkpoint, image.as_deref(), &prompt, &user_emote)?;
        }

        Command::Status { vision_checkpoint, brain_checkpoint } => {
            print_status(&vision_checkpoint, &brain_checkpoint);
        }
    }

    Ok(())
}

fn run_chat(
    vision_cp:  &str,
    brain_cp:   &str,
    image_path: Option<&str>,
    prompt:     &str,
    user_emote: &str,
) -> Result<()> {
    // use burn::backend::NdArray;
    // type B = NdArray<f32, i64, i8>;
    type B = Wgpu;
    let device = Default::default();

    // 1. Load vision model
    let vision_model = vision::model::VisionModel::<B>::load(vision_cp, &device)?;
    println!("✅ Vision model loaded from {vision_cp}");

    // 2. Load brain model + tokenizer
    let (brain_model, tokenizer) = brain::model::YumonBrain::<B>::load(brain_cp, &device)?;
    println!("✅ Brain model loaded from {brain_cp}");

    // Quantize weights for faster CPU inference
    // lots of errors, not stable to do quantization yet
    // use burn::tensor::quantization::{QuantScheme, QuantLevel, QuantMode};
    // let scheme = QuantScheme {
    //     // level: QuantLevel::Tensor,
    //     // mode: QuantMode::Symmetric,
    //     value: burn::tensor::quantization::QuantValue::Q4S,
    //     param: burn::tensor::quantization::QuantParam::BF16,
    //     store: burn::tensor::quantization::QuantStore::Native,
    //     level: QuantLevel::Tensor,
    //     mode: QuantMode::Symmetric,
    //     // ..Default::default()
    // };
    // let mut quantizer = burn::module::Quantizer { scheme, calibration: burn::tensor::quantization::Calibration::MinMax };
    // let brain_model = brain_model.quantize_weights(&mut quantizer);

    // 3. Run vision inference (or use zeros if no image)
    let (class_probs, emote_probs) = if let Some(path) = image_path {
        println!("🖼  Analysing image: {path}");
        let img = vision::loader::load_image_tensor::<B>(path, &device)?;
        vision_model.infer(img)
    } else {
        println!("ℹ️  No image provided — using neutral vision embeddings.");
        (
            vec![1.0 / vision::CIFAR_CLASSES as f32; vision::CIFAR_CLASSES],
            vec![1.0 / vision::EMOTE_CLASSES as f32; vision::EMOTE_CLASSES],
        )
    };

    // 4. Print top predicted class
    let top_class = class_probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let top_emote = emote_probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("👁  Top class  : {} ({:.1}%)", top_class, class_probs[top_class] * 100.0);
    println!("😶 User emote : {} → {}", user_emote, vision::EMOTE_NAMES[top_emote]);

    // 5. Resolve user emote index (from CLI or detected)
    let user_emote_idx = vision::emote_name_to_idx(user_emote)
        .unwrap_or(top_emote);

    // 6. Generate reply from LSTM
    let result = brain_model.generate(
        &tokenizer,
        &class_probs,
        &emote_probs,
        user_emote_idx,
        prompt,
        80,   // max tokens
        // 30,
        &device,
    );

    println!("\n┌─ Yumon says ──────────────────────────────────────────┐");
    println!("│ {}", result.reply);
    println!("└───────────────────────────────────────────────────────┘");
    println!("  Yumon emote: {} ({})", vision::EMOTE_NAMES[result.yumon_emote_idx], result.yumon_emote_idx);

    Ok(())
}

fn print_status(vision_cp: &str, brain_cp: &str) {
    println!("=== Yumon Status ===");

    let vmeta = std::path::Path::new(vision_cp).join("metadata.json");
    if vmeta.exists() {
        let s = std::fs::read_to_string(&vmeta).unwrap_or_default();
        println!("Vision: {s}");
    } else {
        println!("Vision: no checkpoint at {vision_cp}");
    }

    let bmeta = std::path::Path::new(brain_cp).join("metadata.json");
    if bmeta.exists() {
        let s = std::fs::read_to_string(&bmeta).unwrap_or_default();
        println!("Brain : {s}");
    } else {
        println!("Brain : no checkpoint at {brain_cp}");
    }
}
