#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use yumon_pet::brain::{code_corpus::CodeConfig, train::run_code};
    #[derive(Parser)]
    #[command(about = "Train Yumon Code on RunPod from a prepared raw Rust cache")]
    struct Args {
        #[arg(long, default_value = "configs/yumon-code.json")]
        config: std::path::PathBuf,
        /// Validate config, tokenizer, cache and file split without GPU/model training.
        #[arg(long)]
        check: bool,
    }
    let args = Args::parse();
    run_code(&CodeConfig::load(&args.config)?, args.check)
}

#[cfg(target_arch = "wasm32")]
fn main() { eprintln!("Yumon Code training requires RunPod."); }
