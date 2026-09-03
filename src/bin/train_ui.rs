#![recursion_limit = "256"]

use anyhow::Result;
use burn::{
    backend::Wgpu, 
    grad_clipping::GradientClippingConfig, 
    nn::loss::CrossEntropyLossConfig, 
    optim::{AdamWConfig, GradientsParams, Optimizer}, 
    prelude::*, 
    tensor::{Int, TensorData}
};
use cubecl::wgpu::WgpuRuntime;
use serde::{Deserialize, Serialize};
use std::{
    sync::{mpsc, Arc, Mutex},
    thread,
    sync::atomic::{AtomicBool, Ordering},
};
use tao::{
    dpi::{LogicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    window::WindowBuilder,
};
use wry::{WebViewBuilder, http::Request};

use yumon_pet::brain::{
    PAD_TOKEN, BOS_TOKEN,
    bpe::{BpeTokenizer, TokenizerKind},
    chart::TrainingState,
    loader::{DataLoader, FileKind},
    samples::{TrainingStage, WorldContext},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata},
    train::{build_keyword_index, build_label_keywords, TrainBackend},
};

// ── Custom event ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum TrainerEvent {
    Progress {
        step: usize,
        loss: f32,
        avg_loss: f32,
        entropy: f32,
        epoch: usize,
        total_epochs: usize,
        batch: usize,
        total_batches: usize,
    },
    Log { msg: String },
    Done,
}

// ── IPC message from JS ──────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum IpcMsg {
    START_TRAINING {
        name: String,
        embed_dim: usize,
        n_layers: usize,
        attn_heads: usize,
        ff_dim: usize,
        max_seq_len: usize,
        epochs: usize,
        batch_size: usize,
        lr: f64,
    },
    STOP_TRAINING,
}

// ── Trainer State ─────────────────────────────────────────────────────────

struct TrainerApp {
    is_training: Arc<AtomicBool>,
    proxy: EventLoopProxy<TrainerEvent>,
}

impl TrainerApp {
    fn start_training(&self, msg: IpcMsg) {
        if let IpcMsg::START_TRAINING { 
            name, embed_dim, n_layers, attn_heads, ff_dim, max_seq_len, epochs, batch_size, lr 
        } = msg {
            let is_training = self.is_training.clone();
            let proxy = self.proxy.clone();
            is_training.store(true, Ordering::SeqCst);

            thread::spawn(move || {
                let _ = proxy.send_event(TrainerEvent::Log { msg: format!("Starting training run: {}", name) });
                
                if let Err(e) = run_training_loop(
                    &name, embed_dim, n_layers, attn_heads, ff_dim, max_seq_len, epochs, batch_size, lr,
                    is_training, proxy.clone()
                ) {
                    let _ = proxy.send_event(TrainerEvent::Log { msg: format!("Error during training: {}", e) });
                }

                let _ = proxy.send_event(TrainerEvent::Done);
            });
        }
    }

    fn stop_training(&self) {
        self.is_training.store(false, Ordering::SeqCst);
        let _ = self.proxy.send_event(TrainerEvent::Log { msg: "Stopping training...".into() });
    }
}

fn run_training_loop(
    name: &str,
    embed_dim: usize,
    n_layers: usize,
    attn_heads: usize,
    ff_dim: usize,
    max_seq_len: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    is_training: Arc<AtomicBool>,
    proxy: EventLoopProxy<TrainerEvent>,
) -> Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let label_keywords = build_label_keywords();
    let keyword_index = build_keyword_index(&label_keywords);
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

    let config = YumonBrainConfig {
        vocab_size: tokenizer.vocab_size(),
        embed_dim,
        hidden_units: embed_dim,
        n_layers,
        attn_heads,
        ff_dim,
        max_seq_len,
        training_stage: TrainingStage::Structured,
        dropout_rate: 0.05,
    };

    let mut model = config.init(&device);

    let mut loader = DataLoader::new(TrainingStage::Structured);
    loader = loader
        .add("data/ideas.txt",   FileKind::TxtLines, Some(50_000))
        .add("archive/arena_extract.txt",   FileKind::Chats, Some(25_000))
        .add("data/bible_bbe.csv", FileKind::BibleCsv, Some(25_000))
        .add("data/The-Office-Lines-V4.csv",   FileKind::DialogueCsv, Some(25_000))
        .add("data/friends_all_episodes_clean.csv",   FileKind::FriendsCsv, Some(25_000))
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None)
        .total_limit(400_000)
        .seed(42);
    
    let training_samples = loader.load(&tokenizer, &keyword_index, max_seq_len)?;
    let _ = proxy.send_event(TrainerEvent::Log { msg: format!("Loaded {} samples", training_samples.len()) });

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-7)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .with_weight_decay(0.01)
        .init();

    let ce_loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
        .with_smoothing(Some(0.1))
        .init(&device);

    let total_batches = training_samples.len() / batch_size;
    let mut global_step = 0;

    let run_dir = format!("checkpoints/brain/{}", name);
    std::fs::create_dir_all(&run_dir)?;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut idx: Vec<usize> = (0..training_samples.len()).collect();
        // Simple shuffle
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rand::thread_rng());

        for batch_num in 0..total_batches {
            if !is_training.load(Ordering::SeqCst) {
                return Ok(());
            }

            let batch_start = batch_num * batch_size;
            let batch_end = (batch_start + batch_size).min(training_samples.len());
            let batch_idx = &idx[batch_start..batch_end];
            let current_batch_size = batch_idx.len();
            if current_batch_size == 0 { continue; }

            let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * max_seq_len);
            let mut all_enc_ids: Vec<i32> = Vec::with_capacity(current_batch_size * max_seq_len);
            let mut all_dec_input_ids: Vec<i32> = Vec::with_capacity(current_batch_size * max_seq_len);

            for &i in batch_idx {
                let sample = &training_samples[i];
                all_enc_ids.extend(sample.input_ids.iter().map(|&t| t as i32));
                let target_labels = &sample.target_labels;
                let real_len = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(max_seq_len);

                let mut dec_input: Vec<i32> = vec![BOS_TOKEN as i32];
                dec_input.extend(target_labels[0..real_len.saturating_sub(1)].iter().map(|&t| t as i32));
                dec_input.resize(max_seq_len, PAD_TOKEN as i32);

                let mut lang_targets: Vec<i32> = target_labels[0..real_len].iter().map(|&t| t as i32).collect();
                lang_targets.resize(max_seq_len, PAD_TOKEN as i32);

                all_dec_input_ids.extend(dec_input);
                all_lang_targets.extend(lang_targets);
            }

            let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * max_seq_len]), &device);
            let enc_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_enc_ids, [current_batch_size, max_seq_len]), &device);
            let dec_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_dec_input_ids, [current_batch_size, max_seq_len]), &device);

            let token_logits = model.forward::<WgpuRuntime>(enc_t, dec_t.clone());

            // Entropy
            let probs = burn::tensor::activation::softmax(token_logits.clone(), 2);
            let log_probs = (probs.clone() + 1e-10).log();
            let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze::<2>();
            let non_pad_mask = dec_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
            let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).sum().div(non_pad_mask.sum().clamp_min(1.0)).into_scalar();

            // Loss
            let vocab = tokenizer.vocab_size();
            let logits_2d = token_logits.reshape([current_batch_size * max_seq_len, vocab]);
            let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

            let grads = GradientsParams::from_grads(lang_loss.backward(), &model);
            model = optimizer.step(lr, model, grads);

            let loss_val: f32 = lang_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;
            global_step += 1;

            if global_step % 10 == 0 || batch_num == total_batches - 1 {
                let _ = proxy.send_event(TrainerEvent::Progress {
                    step: global_step,
                    loss: loss_val,
                    avg_loss: epoch_loss / (batch_num + 1) as f32,
                    entropy: entropy_val,
                    epoch: epoch + 1,
                    total_epochs: epochs,
                    batch: batch_num + 1,
                    total_batches: total_batches,
                });
            }
        }

        // Save checkpoint
        let meta = BrainMetadata {
            vocab_size: tokenizer.vocab_size(),
            epochs_trained: epoch + 1,
            final_loss: epoch_loss / total_batches as f32,
            batch_size,
            training_stage: TrainingStage::Structured,
            embed_dim,
            hidden_units: embed_dim,
            n_layers,
            attn_heads,
            ff_dim,
            max_seq_len,
        };
        model.save(&run_dir, &tokenizer, &meta)?;
        let _ = proxy.send_event(TrainerEvent::Log { msg: format!("Epoch {} complete. Loss: {:.4}", epoch + 1, epoch_loss / total_batches as f32) });
    }

    Ok(())
}

fn main() -> Result<()> {
    let event_loop = EventLoopBuilder::<TrainerEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    let window = WindowBuilder::new()
        .with_title("Yumon Trainer")
        .with_inner_size(LogicalSize::new(1000, 800))
        .build(&event_loop)?;

    let is_training = Arc::new(AtomicBool::new(false));
    let app = Arc::new(TrainerApp {
        is_training: is_training.clone(),
        proxy: proxy.clone(),
    });

    let app_ipc = app.clone();
    let html = include_str!("./train_ui.html");
    
    let webview = WebViewBuilder::new()
        .with_html(html)
        .with_ipc_handler(move |msg: Request<String>| {
            if let Ok(ipc) = serde_json::from_str::<IpcMsg>(&msg.body()) {
                match ipc {
                    IpcMsg::START_TRAINING { .. } => {
                        app_ipc.start_training(ipc);
                    }
                    IpcMsg::STOP_TRAINING => {
                        app_ipc.stop_training();
                    }
                }
            }
        })
        .build(&window)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,

            Event::UserEvent(TrainerEvent::Progress { step, loss, avg_loss, entropy, epoch, total_epochs, batch, total_batches }) => {
                let js = format!(
                    "window.__update_progress({{ step: {}, loss: {}, avg_loss: {}, entropy: {}, epoch: {}, total_epochs: {}, batch: {}, total_batches: {} }})",
                    step, loss, avg_loss, entropy, epoch, total_epochs, batch, total_batches
                );
                let _ = webview.evaluate_script(&js);
            }

            Event::UserEvent(TrainerEvent::Log { msg }) => {
                let msg_escaped = msg.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
                let js = format!("window.__log(\"{}\")", msg_escaped);
                let _ = webview.evaluate_script(&js);
            }

            Event::UserEvent(TrainerEvent::Done) => {
                let _ = webview.evaluate_script("window.__training_done()");
            }

            _ => {}
        }
    });
}
