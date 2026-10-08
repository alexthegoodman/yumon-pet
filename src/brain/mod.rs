#![allow(warnings)]

pub mod model;
pub mod decoder_model;
pub mod xlstm_model;
pub mod moe_model;
pub mod wiki;
pub mod train;
pub mod tokenizer;
pub mod bpe;
pub mod mdx;
pub mod chart;
pub mod pdf;
pub mod samples;
#[cfg(not(target_arch = "wasm32"))]
pub mod sample_cache;
pub mod keywords;
pub mod fixer;
pub mod loader;
mod loading;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod loading_tests;
pub mod flash_attn;
pub mod classic_attn;
pub mod sentiment;
pub mod chats;
#[cfg(not(target_arch = "wasm32"))]
pub mod am_distill;

// Re-export tokenizer for convenience
pub use tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN};


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
