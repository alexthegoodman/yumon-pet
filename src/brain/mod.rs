#![allow(warnings)]

pub mod model;
pub mod decoder_model;
pub mod xlstm_model;
pub mod moe_model;
pub mod prediction_model;
pub mod daw_actions;
pub mod wiki;
pub mod train;
pub mod tokenizer;
pub mod bpe;
pub mod mdx;
pub mod chart;
pub mod pdf;
pub mod samples;
pub mod keywords;
pub mod fixer;
pub mod loader;
pub mod flash_attn;
pub mod classic_attn;
pub mod sentiment;
pub mod chats;

// Re-export tokenizer for convenience
pub use tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN};

// Re-export DAW prediction model types and inference
pub use prediction_model::{
    ActionPredictor, PredictedAction, PredictionMetadata, PredictionModel,
    PredictionModelConfig, predict_next_actions, resolve_prediction_checkpoint_dir,
};
pub use daw_actions::{
    ActionStep, DawAction, DawTask, ACTION_VOCAB_SIZE, BOS_ACTION, EOS_ACTION,
    MAX_ACTION_PARAMS, NUM_DAW_ACTIONS, NUM_DAW_TASKS, PAD_ACTION,
};

// ─── Context vector layout ────────────────────────────────────────────────────
//
// At each LSTM timestep, a context vector is concatenated with the token embedding:
//
//   context = [class_probs: 100, emote_probs: 7, user_emote_onehot: 7] = 114 dims
//
// This tells the LSTM *what Yumon sees* and *what the user is feeling*.

use crate::vision::{CIFAR_CLASSES, EMOTE_CLASSES};

// pub const CONTEXT_DIMS: usize = CIFAR_CLASSES + EMOTE_CLASSES + EMOTE_CLASSES;
// = 100 + 7 + 7 = 114
