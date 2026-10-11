//! CPU-only tokenizer preparation for Code (default) or the legacy Pet corpus.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use std::path::PathBuf;
    #[derive(Parser)]
    #[command(
        about = "Train Code BPE on the exact balanced curriculum and publish matching caches"
    )]
    struct Args {
        #[arg(
            long,
            default_value = "configs/yumon-code.json",
            conflicts_with = "pet"
        )]
        code_config: PathBuf,
        /// New output bundle directory (Pet mode: tokenizer directory).
        output: Option<PathBuf>,
        /// Train the legacy Pet prose tokenizer instead of Code.
        #[arg(long)]
        pet: bool,
    }
    let args = Args::parse();
    if args.pet {
        use yumon_pet::brain::{
            bpe::BpeTokenizer, samples::TrainingStage, train::stage_data_loader,
        };
        let output = args.output.unwrap_or_else(|| "yumon_bpe".into());
        anyhow::ensure!(
            !output.exists(),
            "output already exists: {}",
            output.display()
        );
        let sentences = stage_data_loader(TrainingStage::Language)
            .total_limit(1_500_000)
            .seed(4815162342)
            .load_sentences()?;

        let mut x = 0;
        for sent in &sentences {
            println!("Sentence: {:?}", sent);
            x = x + 1;
            if x > 25 {
                break;
            }
        }

        BpeTokenizer::train(sentences.iter().collect(), 16384)?.save(
            output
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("output path must be UTF-8"))?,
        )?;
        return Ok(());
    }
    let config = yumon_pet::brain::code_corpus::CodeConfig::load(&args.code_config)?;
    let output = args
        .output
        .unwrap_or_else(|| "training-cache/code_tokenizer_retrained".into());
    yumon_pet::brain::code_corpus::prepare_training_tokenizer(&config, &output)?;
    println!(
        "Prepared {}. Train with: train_code --config {}/config.json",
        output.display(),
        output.display()
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Tokenizer preparation requires a native build.");
}
