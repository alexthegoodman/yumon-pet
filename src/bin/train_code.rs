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
        /// Run tiny CUDA/BF16 MoE training and checkpoint checks; no corpus/config required.
        #[arg(long, conflicts_with = "check")]
        smoke_test: bool,
    }
    let args = Args::parse();
    if args.smoke_test {
        return yumon_pet::brain::train::cuda_moe_smoke_test();
    }
    run_code(&CodeConfig::load(&args.config)?, args.check)
}

#[cfg(target_arch = "wasm32")]
fn main() { eprintln!("Yumon Code training requires RunPod."); }
