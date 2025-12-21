//! Runtime for evaluating Ein programs.

mod context;
mod knowledge;

pub use context::{KBData, OptimizerType, Runtime, TextData, Tokenizer};
pub use knowledge::KnowledgeBase;