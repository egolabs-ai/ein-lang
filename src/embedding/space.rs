//! Embedding space for tensor logic reasoning.
//!
//! Objects are represented as unit vectors in a high-dimensional space.
//! Relations are represented as matrices (sum of outer products of embeddings).
//!
//! Key formulas from Domingos paper:
//! - Embed relation: EmbR[i,j] = Σ_{x,y} R(x,y) · Emb[x,i] · Emb[y,j]
//! - Query fact: score = Emb[a] @ EmbR @ Emb[b]
//! - Temperature: σ(x,T) = 1 / (1 + e^(-x/T))
//!
//! Error probability: std_dev = √(N/D) where N=facts, D=dimension

use candle_core::{DType, Device, Result, Tensor};

use crate::tensor::SparseBool;

/// Embedding space for reasoning over relations.
///
/// Objects are embedded as unit vectors, relations as matrices.
/// Temperature controls deductive (T=0) vs analogical (T>0) reasoning.
#[derive(Debug, Clone)]
pub struct EmbeddingSpace {
    /// Embedding dimension
    dim: usize,
    /// Object embeddings: [num_objects, dim], normalized to unit vectors
    embeddings: Tensor,
    /// Device (CPU/GPU)
    device: Device,
}

impl EmbeddingSpace {
    /// Create a new embedding space with random unit vector embeddings.
    ///
    /// # Arguments
    /// * `num_objects` - Number of objects to embed
    /// * `dim` - Embedding dimension (higher = lower error probability)
    /// * `device` - Compute device
    ///
    /// # Example
    /// ```
    /// use candle_core::Device;
    /// use ein::EmbeddingSpace;
    ///
    /// let space = EmbeddingSpace::new(100, 256, &Device::Cpu).unwrap();
    /// ```
    pub fn new(num_objects: usize, dim: usize, device: &Device) -> Result<Self> {
        // Generate random embeddings
        let emb = Tensor::randn(0.0f32, 1.0, (num_objects, dim), device)?;

        // Normalize to unit vectors: emb / ||emb||
        let norms = emb.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
        let embeddings = emb.broadcast_div(&norms)?;

        Ok(Self {
            dim,
            embeddings,
            device: device.clone(),
        })
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of objects.
    pub fn num_objects(&self) -> Result<usize> {
        Ok(self.embeddings.dim(0)?)
    }

    /// Get the embedding vector for an object.
    pub fn get_embedding(&self, obj: usize) -> Result<Tensor> {
        self.embeddings.get(obj)
    }

    /// Embed a binary relation as a matrix.
    ///
    /// Formula: EmbR[i,j] = Σ_{x,y} R(x,y) · Emb[x,i] · Emb[y,j]
    ///
    /// This is the sum of outer products of embeddings for each fact.
    /// Uses batched operations for GPU efficiency.
    ///
    /// # Arguments
    /// * `rel` - Sparse boolean relation (must be binary, arity=2)
    ///
    /// # Returns
    /// Tensor of shape [dim, dim] representing the embedded relation
    pub fn embed_relation(&self, rel: &SparseBool) -> Result<Tensor> {
        assert_eq!(rel.arity, 2, "embed_relation requires binary relation");

        if rel.is_empty() {
            return Tensor::zeros((self.dim, self.dim), DType::F32, &self.device);
        }

        // Collect all subject and object indices
        let tuples: Vec<_> = rel.tuples.iter().collect();
        let num_facts = tuples.len();

        let subjects: Vec<u32> = tuples.iter().map(|t| t[0] as u32).collect();
        let objects: Vec<u32> = tuples.iter().map(|t| t[1] as u32).collect();

        // Create index tensors
        let subject_idx = Tensor::from_vec(subjects, (num_facts,), &self.device)?;
        let object_idx = Tensor::from_vec(objects, (num_facts,), &self.device)?;

        // Gather embeddings: [num_facts, dim]
        let emb_subjects = self.embeddings.index_select(&subject_idx, 0)?;
        let emb_objects = self.embeddings.index_select(&object_idx, 0)?;

        // Sum of outer products: Emb_subjects.T @ Emb_objects
        // [dim, num_facts] @ [num_facts, dim] = [dim, dim]
        let emb_rel = emb_subjects.t()?.matmul(&emb_objects)?;

        Ok(emb_rel)
    }

    /// Query whether a fact (a, b) is in the embedded relation.
    ///
    /// Formula: score = Emb[a] @ EmbR @ Emb[b]
    ///
    /// # Arguments
    /// * `emb_rel` - Embedded relation matrix [dim, dim]
    /// * `a` - Subject object index
    /// * `b` - Object index
    /// * `temp` - Temperature (0 = deductive, >0 = analogical)
    ///
    /// # Returns
    /// Score in [0, 1]. At T=0, this is 0 or 1 (Boolean).
    /// At T>0, this is a soft probability.
    pub fn query(&self, emb_rel: &Tensor, a: usize, b: usize, temp: f64) -> Result<f64> {
        // Get embeddings
        let emb_a = self.embeddings.get(a)?; // [dim]
        let emb_b = self.embeddings.get(b)?; // [dim]

        // Score = emb_a @ emb_rel @ emb_b
        // First: emb_rel @ emb_b = [dim, dim] @ [dim] = [dim]
        let intermediate = emb_rel.matmul(&emb_b.unsqueeze(1)?)?.squeeze(1)?;

        // Then: emb_a @ intermediate = [dim] · [dim] = scalar
        let score = emb_a.mul(&intermediate)?.sum_all()?;
        let score_val: f32 = score.to_scalar()?;

        // Apply temperature-controlled sigmoid
        let result = if temp <= 0.0 {
            // T=0: Pure deductive mode - threshold at 0.5
            // At T=0, we want exact Boolean results
            if score_val > 0.5 {
                1.0
            } else {
                0.0
            }
        } else {
            // T>0: Soft sigmoid
            // σ(x, T) = 1 / (1 + e^(-x/T))
            let x = score_val as f64;
            1.0 / (1.0 + (-x / temp).exp())
        };

        Ok(result)
    }

    /// Query all pairs and return a score matrix.
    ///
    /// # Arguments
    /// * `emb_rel` - Embedded relation matrix [dim, dim]
    /// * `temp` - Temperature
    ///
    /// # Returns
    /// Score matrix [num_objects, num_objects]
    pub fn query_all_pairs(&self, emb_rel: &Tensor, temp: f64) -> Result<Tensor> {
        // Compute all scores at once: Emb @ EmbR @ Emb.T
        // [n, d] @ [d, d] @ [d, n] = [n, n]
        let intermediate = self.embeddings.matmul(emb_rel)?; // [n, d]
        let scores = intermediate.matmul(&self.embeddings.t()?)?; // [n, n]

        if temp <= 0.0 {
            // T=0: Threshold at 0.5
            let threshold = Tensor::full(0.5f32, scores.shape(), &self.device)?;
            let binary = scores.gt(&threshold)?.to_dtype(DType::F32)?;
            Ok(binary)
        } else {
            // T>0: Apply sigmoid with temperature
            // We need to divide by temp first, then apply sigmoid
            let scaled = (scores / temp)?;
            // Candle sigmoid
            let result = candle_nn::ops::sigmoid(&scaled)?;
            Ok(result)
        }
    }

    /// Compose two relations via matrix multiplication.
    ///
    /// If R1(x,y) and R2(y,z), then (R1 ∘ R2)(x,z) = R1 @ R2
    ///
    /// # Arguments
    /// * `rel1` - First embedded relation [dim, dim]
    /// * `rel2` - Second embedded relation [dim, dim]
    ///
    /// # Returns
    /// Composed relation [dim, dim]
    pub fn compose(&self, rel1: &Tensor, rel2: &Tensor) -> Result<Tensor> {
        rel1.matmul(rel2)
    }

    /// Get the raw score for a query (before sigmoid).
    ///
    /// Useful for debugging and understanding the embedding space.
    pub fn query_raw_score(&self, emb_rel: &Tensor, a: usize, b: usize) -> Result<f32> {
        let emb_a = self.embeddings.get(a)?;
        let emb_b = self.embeddings.get(b)?;

        let intermediate = emb_rel.matmul(&emb_b.unsqueeze(1)?)?.squeeze(1)?;
        let score = emb_a.mul(&intermediate)?.sum_all()?;

        score.to_scalar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_space_creation() {
        let space = EmbeddingSpace::new(10, 64, &Device::Cpu).unwrap();
        assert_eq!(space.dim(), 64);
        assert_eq!(space.num_objects().unwrap(), 10);
    }

    #[test]
    fn test_embeddings_are_normalized() {
        let space = EmbeddingSpace::new(10, 64, &Device::Cpu).unwrap();

        // Check that each embedding has unit norm
        for i in 0..10 {
            let emb = space.get_embedding(i).unwrap();
            let norm_sq: f32 = emb.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
            assert!(
                (norm_sq - 1.0).abs() < 0.01,
                "Embedding {} has norm^2 = {}, expected 1.0",
                i,
                norm_sq
            );
        }
    }

    #[test]
    fn test_embed_relation() {
        let space = EmbeddingSpace::new(10, 64, &Device::Cpu).unwrap();

        let mut rel = SparseBool::new(2);
        rel.insert(vec![0, 1]);
        rel.insert(vec![2, 3]);

        let emb_rel = space.embed_relation(&rel).unwrap();
        assert_eq!(emb_rel.dims(), &[64, 64]);
    }

    #[test]
    fn test_query_basic() {
        let space = EmbeddingSpace::new(10, 256, &Device::Cpu).unwrap();

        let mut rel = SparseBool::new(2);
        rel.insert(vec![0, 1]);

        let emb_rel = space.embed_relation(&rel).unwrap();

        // Query the fact that exists
        let score = space.query(&emb_rel, 0, 1, 0.0).unwrap();
        // With high dimension, should be close to 1
        assert!(score > 0.5, "Score for true fact: {}", score);
    }
}