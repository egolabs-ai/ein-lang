//! Embedding-based reasoning for Tensor Logic.
//!
//! This module implements reasoning in embedding space as described in
//! Domingos' "Tensor Logic: The Language of AI" paper.
//!
//! Key insight: Relations can be embedded as tensors, and queries can be
//! answered via tensor operations. Temperature T controls the mode:
//! - T=0: Pure deductive reasoning (no hallucinations)
//! - T>0: Analogical reasoning (similar objects share inferences)

mod space;
mod trainer;

pub use space::EmbeddingSpace;
pub use trainer::TrainableEmbeddingSpace;