//! Explicit GPU integration check; normal curriculum tests remain CPU-only.
use super::*;
use crate::brain::{
    code_curriculum::PreparedCode,
    samples::{Action, CardinalDir, Sample},
};

#[test]
#[ignore = "requires a local WGPU GPU, or CUDA with --features cuda-training"]
fn code_curriculum_gpu_all_architectures_resume_and_causal_targets() -> Result<()> {
    let directory =
        std::env::temp_dir().join(format!("yumon-curriculum-gpu-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory)?;
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(directory.clone());
    let source = directory.join("source.rs");
    std::fs::write(&source, "fn test() { let value = 1; }")?;
    let tokenizer_path = directory.join("tokenizer");
    crate::brain::code_corpus::train_tokenizer(&[source], 300, tokenizer_path.to_str().unwrap())?;
    let tokenizer = crate::brain::code_corpus::load_tokenizer(tokenizer_path.to_str().unwrap())?;
    let device = Default::default();
    for (index, architecture) in [
        Architecture::DecoderOnly,
        Architecture::Moe {
            num_experts: 2,
            top_k: 1,
        },
        Architecture::XLstm,
        Architecture::EncoderDecoder,
    ]
    .into_iter()
    .enumerate()
    {
        let mut run = RunConfig {
            name: format!("architecture-{index}"),
            embed_dim: 16,
            hidden_units: 16,
            n_layers: 1,
            attn_heads: 2,
            ff_dim: 32,
            max_seq_len: 8,
            architecture,
            stages: vec![StageConfig {
                stage: TrainingStage::Language,
                loss_threshold: 0.0,
                epochs: 2,
                batch_size: 2,
                first_lr: 0.001,
                last_lr: 0.0001,
                weight_decay: 0.01,
                epsilon: 1e-7,
                smoothing: 0.0,
            }],
        };
        let path = directory.join(&run.name);
        std::fs::create_dir_all(&path)?;
        match architecture {
            Architecture::DecoderOnly => {
                exercise::<YumonDecBrain<TrainBackend>>(&mut run, &path, &tokenizer, &device)?
            }
            Architecture::Moe { .. } => {
                exercise::<YumonMoeBrain<TrainBackend>>(&mut run, &path, &tokenizer, &device)?
            }
            Architecture::XLstm => {
                exercise::<YumonXLstmBrain<TrainBackend>>(&mut run, &path, &tokenizer, &device)?
            }
            Architecture::EncoderDecoder => {
                exercise::<YumonBrain<TrainBackend>>(&mut run, &path, &tokenizer, &device)?
            }
        }
    }
    Ok(())
}

fn exercise<M>(
    run: &mut RunConfig,
    path: &std::path::Path,
    tokenizer: &TokenizerKind,
    device: &<TrainBackend as Backend>::Device,
) -> Result<()>
where
    M: CausalLmTrain,
    M::InnerModule: CausalLm<InnerBackend>,
{
    let sample = |id| Sample {
        input_ids: vec![BOS_TOKEN, id, 5, 0, 0, 0, 0, 0],
        target_labels: vec![id, 5, EOS_TOKEN, 0, 0, 0, 0, 0],
        action: Action::Sit,
        motion_dir: CardinalDir::None,
        world: WorldContext::default(),
        target_json: String::new(),
        pair: ("fixture.rs".into(), format!("line {id}")),
    };
    let prepared = || PreparedCode {
        training: vec![sample(4), sample(6), sample(7)],
        validation: vec![sample(8)],
        tier_ends: Some([1, 2, 3]),
        identity: serde_json::json!({}),
        seed: 42,
    };
    let train = |run: &RunConfig| {
        train_causal_lm::<M>(
            run,
            path,
            tokenizer,
            &HashMap::new(),
            &[],
            device,
            Some(prepared()),
        )
    };
    train(run)?;
    let (_, completed) = M::init_or_resume(run, tokenizer, path, device)?;
    assert_eq!(completed, 2);
    run.stages[0].epochs = 3;
    train(run)?;
    let (model, completed) = M::init_or_resume(run, tokenizer, path, device)?;
    assert_eq!(completed, 3);
    let saved = std::fs::read(path.join("model.bin"))?;
    train(run)?;
    assert_eq!(saved, std::fs::read(path.join("model.bin"))?);
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("metadata.json"))?)?;
    assert!(metadata["final_loss"].as_f64().unwrap().is_finite());
    // Changing future tokens must not change prefix predictions, including the
    // encoder-decoder adapter, whose encoder must never see the source targets.
    let inference = model.valid();
    let logits = |ids: [i32; 8]| {
        inference
            .logits(Tensor::<InnerBackend, 2, Int>::from_ints([ids], device))
            .slice([0..1, 0..3, 0..tokenizer.vocab_size()])
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
    };
    let first = logits([1, 4, 5, 6, 7, 8, 9, 2]);
    let changed = logits([1, 4, 5, 9, 8, 7, 6, 2]);
    assert!(
        first
            .iter()
            .zip(changed)
            .all(|(a, b)| (a - b).abs() <= 1e-5)
    );
    Ok(())
}
