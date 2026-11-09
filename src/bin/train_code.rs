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
        /// Override `adaptive_loss.mode` (disabled, difficulty, difficulty_progress,
        /// frequency) so experiment arms share one config file.
        #[arg(long)]
        adaptive_mode: Option<String>,
        /// Number of shared experts per block (DeepSeek-style) that run unconditionally on all tokens.
        #[arg(long)]
        num_shared_experts: Option<usize>,
        /// Tie input embeddings to the output token head (weight tying).
        #[arg(long)]
        tie_embeddings: Option<bool>,
        /// Initialize weights from a base model directory or checkpoint for fine-tuning.
        #[arg(long)]
        init_checkpoint: Option<std::path::PathBuf>,
        /// Evaluation prompt stage: "base", "sft", "dpo", or "code".
        #[arg(long, alias = "eval-stage", alias = "eval_stage")]
        stage: Option<String>,
        /// JL projection arm (docs/JL.md): `none`, `SITE:DIM` or `SITE:DIM@LAYER`,
        /// SITE = residual | attention | ffn, optional `+learned` suffix; comma-separated
        /// to stack sites. Replaces the config's `jl` maps, keeping their seed and ff_dim.
        #[arg(long)]
        jl: Option<String>,
    }
    let args = Args::parse();
    if args.smoke_test {
        return yumon_pet::brain::train::cuda_moe_smoke_test();
    }
    let mut config = CodeConfig::load(&args.config)?;
    if let Some(init) = args.init_checkpoint {
        config.init_checkpoint = Some(init);
    }
    if let Some(stage) = args.stage {
        config.eval_stage = Some(stage);
    }
    if let Some(shared) = args.num_shared_experts {
        config.num_shared_experts = shared;
    }
    if let Some(tie) = args.tie_embeddings {
        config.tie_embeddings = tie;
    }
    if let Some(mode) = &args.adaptive_mode {
        let adaptive = config.adaptive_loss.as_mut()
            .ok_or_else(|| anyhow::anyhow!("--adaptive-mode needs an adaptive_loss section in the config"))?;
        adaptive.mode = serde_json::from_value(serde_json::Value::String(mode.clone()))?;
        adaptive.enabled = true;
    }
    if let Some(spec) = &args.jl {
        let mut arm = yumon_pet::brain::jl::JlSet::parse(spec)?;
        if let Some(base) = config.jl.take() {
            // Keep the config's seed, and the narrow ff_dim of a residual map.
            for j in &mut arm.0 {
                j.seed = base.0[0].seed;
                if let Some(same) = base.0.iter().find(|b| b.site == j.site) {
                    j.seed = same.seed;
                    j.ff_dim = same.ff_dim;
                }
            }
        }
        config.jl = Some(arm);
        config.validate()?;
    }
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
