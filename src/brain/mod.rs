/// Language brain subsystem — LSTM character/word model for Yumon replies.

pub mod model;
pub mod wiki;
pub mod train;
pub mod tokenizer;
pub mod bpe;
pub mod mdx;
pub mod chart;
pub mod pdf;
pub mod world_augment;

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

pub const CONTEXT_DIMS: usize = CIFAR_CLASSES + EMOTE_CLASSES + EMOTE_CLASSES;
// = 100 + 7 + 7 = 114
