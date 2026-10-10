#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use yumon_pet::brain::{code_corpus::CodeConfig, train::run_code};
    #[derive(Parser)]
    #[command(about = "Train Yumon Code locally or on RunPod from a prepared Rust cache")]
    struct Args {
        #[arg(long, default_value = "configs/yumon-code.json")]
        config: std::path::PathBuf,
        /// Validate config, tokenizer, cache and file split without GPU/model training.
        #[arg(long)]
        check: bool,
        /// Require the low/mild/moderate curriculum (used by the RunPod image).
        #[arg(long, conflicts_with = "smoke_test")]
        require_curriculum: bool,
        /// Run tiny CUDA/FP32 MoE training and checkpoint checks; no corpus/config required.
        #[arg(long, conflicts_with = "check")]
        smoke_test: bool,
    }
    let args = Args::parse();
    if args.smoke_test {
        return yumon_pet::brain::train::cuda_moe_smoke_test();
    }
    let config = CodeConfig::load(&args.config)?;
    anyhow::ensure!(
        !args.require_curriculum || config.curriculum.is_some(),
        "this launch requires a curriculum bucket directory in the config"
    );
    run_code(&config, args.check)
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Yumon Code training requires RunPod.");
}
