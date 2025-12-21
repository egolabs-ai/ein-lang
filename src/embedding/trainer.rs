//! Embedding trainer for learning relation representations.
//!
//! Learns embeddings from positive/negative examples using gradient descent.
//!
//! Loss function (binary cross-entropy style):
//! - Positive pairs: loss = -log(score)
//! - Negative pairs: loss = -log(1 - score)
//!
//! Based on kocoro-lab/tensorlogic training approach.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

use crate::tensor::SparseBool;

/// Trainable embedding space for learning relations.
pub struct TrainableEmbeddingSpace {
    /// Embedding dimension
    dim: usize,
    /// Number of objects
    num_objects: usize,
    /// Learnable object embeddings: [num_objects, dim]
    embeddings: Var,
    /// Temperature for scoring
    temperature: f64,
    /// Device
    device: Device,
}

impl TrainableEmbeddingSpace {
    /// Create a new trainable embedding space.
    ///
    /// # Arguments
    /// * `num_objects` - Number of objects to embed
    /// * `dim` - Embedding dimension
    /// * `temperature` - Temperature for sigmoid scoring (default: 0.2)
    /// * `device` - Compute device
    pub fn new(num_objects: usize, dim: usize, temperature: f64, device: &Device) -> Result<Self> {
        // Initialize with random normalized embeddings
        let emb = Tensor::randn(0.0f32, 1.0, (num_objects, dim), device)?;
        let norms = emb.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
        let emb_normalized = emb.broadcast_div(&norms)?;

        // Wrap in Var for gradient tracking
        let embeddings = Var::from_tensor(&emb_normalized)?;

        Ok(Self {
            dim,
            num_objects,
            embeddings,
            temperature,
            device: device.clone(),
        })
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get embeddings as tensor (for inference).
    pub fn embeddings(&self) -> Tensor {
        self.embeddings.as_tensor().clone()
    }

    /// Embed a relation using current embeddings.
    ///
    /// Returns the embedded relation matrix [dim, dim].
    pub fn embed_relation(&self, rel: &SparseBool) -> Result<Tensor> {
        assert_eq!(rel.arity, 2, "embed_relation requires binary relation");

        if rel.is_empty() {
            return Tensor::zeros((self.dim, self.dim), DType::F32, &self.device);
        }

        let tuples: Vec<_> = rel.tuples.iter().collect();
        let num_facts = tuples.len();

        let subjects: Vec<u32> = tuples.iter().map(|t| t[0] as u32).collect();
        let objects: Vec<u32> = tuples.iter().map(|t| t[1] as u32).collect();

        let subject_idx = Tensor::from_vec(subjects, (num_facts,), &self.device)?;
        let object_idx = Tensor::from_vec(objects, (num_facts,), &self.device)?;

        let emb_tensor = self.embeddings.as_tensor();
        let emb_subjects = emb_tensor.index_select(&subject_idx, 0)?;
        let emb_objects = emb_tensor.index_select(&object_idx, 0)?;

        emb_subjects.t()?.matmul(&emb_objects)
    }

    /// Score a batch of pairs against an embedded relation.
    ///
    /// Returns scores in [0, 1] after sigmoid with temperature.
    pub fn score_pairs(
        &self,
        emb_rel: &Tensor,
        pairs: &[(usize, usize)],
    ) -> Result<Tensor> {
        if pairs.is_empty() {
            return Tensor::zeros((0,), DType::F32, &self.device);
        }

        let subjects: Vec<u32> = pairs.iter().map(|(s, _)| *s as u32).collect();
        let objects: Vec<u32> = pairs.iter().map(|(_, o)| *o as u32).collect();
        let n = pairs.len();

        let subject_idx = Tensor::from_vec(subjects, (n,), &self.device)?;
        let object_idx = Tensor::from_vec(objects, (n,), &self.device)?;

        let emb_tensor = self.embeddings.as_tensor();
        let emb_subjects = emb_tensor.index_select(&subject_idx, 0)?; // [n, dim]
        let emb_objects = emb_tensor.index_select(&object_idx, 0)?; // [n, dim]

        // Score = subject @ emb_rel @ object (per pair)
        // [n, dim] @ [dim, dim] = [n, dim]
        let intermediate = emb_subjects.matmul(emb_rel)?;
        // Element-wise multiply and sum: (intermediate * emb_objects).sum(dim=1)
        let scores = (intermediate * emb_objects)?.sum(1)?; // [n]

        // Apply sigmoid with temperature
        let scaled = (scores / self.temperature)?;
        candle_nn::ops::sigmoid(&scaled)
    }

    /// Compute binary cross-entropy loss for positive and negative pairs.
    ///
    /// Loss = -mean(log(score_pos)) - mean(log(1 - score_neg))
    pub fn compute_loss(
        &self,
        emb_rel: &Tensor,
        positive_pairs: &[(usize, usize)],
        negative_pairs: &[(usize, usize)],
    ) -> Result<Tensor> {
        let eps = 1e-7f64;

        // Score positive pairs
        let pos_scores = self.score_pairs(emb_rel, positive_pairs)?;
        // Loss for positives: -log(score)
        let pos_loss = pos_scores.clamp(eps, 1.0 - eps)?.log()?.neg()?.mean_all()?;

        // Score negative pairs
        let neg_scores = self.score_pairs(emb_rel, negative_pairs)?;
        // Loss for negatives: -log(1 - score)
        let one_minus_neg = (neg_scores.neg()? + 1.0)?;
        let neg_loss = one_minus_neg.clamp(eps, 1.0 - eps)?.log()?.neg()?.mean_all()?;

        // Total loss
        (pos_loss + neg_loss)? / 2.0
    }

    /// Sample negative pairs (pairs not in the positive set).
    pub fn sample_negatives(
        &self,
        positive_set: &SparseBool,
        num_samples: usize,
        seed: u64,
    ) -> Vec<(usize, usize)> {
        let mut negatives = Vec::with_capacity(num_samples);
        let mut rng_state = seed;

        while negatives.len() < num_samples {
            // Simple LCG random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s = (rng_state >> 33) as usize % self.num_objects;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let o = (rng_state >> 33) as usize % self.num_objects;

            // Check not in positive set
            if !positive_set.contains(&[s, o]) {
                negatives.push((s, o));
            }
        }

        negatives
    }

    /// Train embeddings on a relation.
    ///
    /// # Arguments
    /// * `relation` - The relation to learn
    /// * `epochs` - Number of training epochs
    /// * `lr` - Learning rate
    /// * `neg_ratio` - Ratio of negative to positive samples
    ///
    /// # Returns
    /// Final loss value
    pub fn train(
        &mut self,
        relation: &SparseBool,
        epochs: usize,
        lr: f64,
        neg_ratio: usize,
    ) -> Result<f64> {
        // Collect positive pairs
        let positive_pairs: Vec<(usize, usize)> = relation
            .tuples
            .iter()
            .map(|t| (t[0], t[1]))
            .collect();

        if positive_pairs.is_empty() {
            return Ok(0.0);
        }

        let num_negatives = positive_pairs.len() * neg_ratio;

        // Create optimizer
        let params = ParamsAdamW {
            lr,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(vec![self.embeddings.clone()], params)?;

        let mut final_loss = 0.0;

        for epoch in 0..epochs {
            // Sample negatives (different each epoch)
            let negative_pairs = self.sample_negatives(relation, num_negatives, epoch as u64 * 12345);

            // Embed relation with current embeddings
            let emb_rel = self.embed_relation(relation)?;

            // Compute loss
            let loss = self.compute_loss(&emb_rel, &positive_pairs, &negative_pairs)?;

            // Backward pass
            optimizer.backward_step(&loss)?;

            final_loss = loss.to_scalar::<f32>()? as f64;

            // Optional: re-normalize embeddings to unit vectors
            if epoch % 10 == 0 {
                let emb = self.embeddings.as_tensor();
                let norms = emb.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
                let normalized = emb.broadcast_div(&norms)?;
                self.embeddings.set(&normalized)?;
            }
        }

        Ok(final_loss)
    }

    /// Query a single pair.
    pub fn query(&self, emb_rel: &Tensor, a: usize, b: usize) -> Result<f64> {
        let scores = self.score_pairs(emb_rel, &[(a, b)])?;
        let score: f32 = scores.get(0)?.to_scalar()?;
        Ok(score as f64)
    }

    /// Query all pairs, returning [num_objects, num_objects] score matrix.
    pub fn query_all_pairs(&self, emb_rel: &Tensor) -> Result<Tensor> {
        let emb = self.embeddings.as_tensor();
        let intermediate = emb.matmul(emb_rel)?;
        let scores = intermediate.matmul(&emb.t()?)?;
        let scaled = (scores / self.temperature)?;
        candle_nn::ops::sigmoid(&scaled)
    }

    /// Compose two embedded relations.
    pub fn compose(&self, rel1: &Tensor, rel2: &Tensor) -> Result<Tensor> {
        rel1.matmul(rel2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainable_space_creation() {
        let space = TrainableEmbeddingSpace::new(10, 32, 0.2, &Device::Cpu).unwrap();
        assert_eq!(space.dim(), 32);
    }

    #[test]
    fn test_score_pairs() {
        let space = TrainableEmbeddingSpace::new(10, 32, 0.2, &Device::Cpu).unwrap();

        let mut rel = SparseBool::new(2);
        rel.insert(vec![0, 1]);
        rel.insert(vec![2, 3]);

        let emb_rel = space.embed_relation(&rel).unwrap();
        let scores = space.score_pairs(&emb_rel, &[(0, 1), (2, 3), (0, 5)]).unwrap();

        assert_eq!(scores.dims(), &[3]);
    }

    #[test]
    fn test_sample_negatives() {
        let space = TrainableEmbeddingSpace::new(10, 32, 0.2, &Device::Cpu).unwrap();

        let mut rel = SparseBool::new(2);
        rel.insert(vec![0, 1]);
        rel.insert(vec![1, 2]);

        let negatives = space.sample_negatives(&rel, 5, 42);
        assert_eq!(negatives.len(), 5);

        // Check none are in the positive set
        for (s, o) in &negatives {
            assert!(!rel.contains(&[*s, *o]));
        }
    }
}