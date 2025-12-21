//! Sparse Boolean tensors for efficient relation representation.
//!
//! A relation like Parent(x,y) is stored as a set of tuples, not a dense matrix.
//! This is efficient when the relation is sparse (few facts, large domain).

use candle_core::{Device, Result, Tensor};
use rustc_hash::FxHashSet;

/// A sparse Boolean tensor represented as a set of tuples.
///
/// For a binary relation R(x,y), we store the set {(x,y) : R(x,y) is true}.
#[derive(Debug, Clone, Default)]
pub struct SparseBool {
    /// Number of indices (arity of the relation)
    pub arity: usize,
    /// Set of tuples where the relation holds
    pub tuples: FxHashSet<Vec<usize>>,
}

impl SparseBool {
    /// Create a new empty sparse tensor with given arity.
    pub fn new(arity: usize) -> Self {
        Self {
            arity,
            tuples: FxHashSet::default(),
        }
    }

    /// Insert a tuple into the relation.
    pub fn insert(&mut self, tuple: Vec<usize>) {
        debug_assert_eq!(tuple.len(), self.arity);
        self.tuples.insert(tuple);
    }

    /// Check if a tuple is in the relation.
    pub fn contains(&self, tuple: &[usize]) -> bool {
        self.tuples.contains(tuple)
    }

    /// Number of facts in the relation.
    pub fn len(&self) -> usize {
        self.tuples.len()
    }

    /// Check if the relation is empty.
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Hash-join two relations on specified indices.
    ///
    /// Example: Parent(x,y) ⋈ Parent(y,z) on (1,0)
    /// - self_idx: which index of self to join on
    /// - other_idx: which index of other to join on
    /// - Result: tuples of length self.arity + other.arity - 1
    pub fn join(&self, other: &Self, self_idx: usize, other_idx: usize) -> Self {
        // Build hash index on `other` for the join column
        let mut index: rustc_hash::FxHashMap<usize, Vec<&Vec<usize>>> =
            rustc_hash::FxHashMap::default();

        for tuple in &other.tuples {
            index.entry(tuple[other_idx]).or_default().push(tuple);
        }

        // Join
        let new_arity = self.arity + other.arity - 1;
        let mut result = SparseBool::new(new_arity);

        for self_tuple in &self.tuples {
            let join_key = self_tuple[self_idx];

            if let Some(matches) = index.get(&join_key) {
                for other_tuple in matches {
                    // Build combined tuple
                    let mut combined = self_tuple.clone();
                    for (i, &val) in other_tuple.iter().enumerate() {
                        if i != other_idx {
                            combined.push(val);
                        }
                    }
                    result.insert(combined);
                }
            }
        }

        result
    }

    /// Project to keep only specified positions.
    ///
    /// Example: (a, b, c).project(&[0, 2]) → (a, c)
    pub fn project(&self, keep: &[usize]) -> Self {
        let mut result = SparseBool::new(keep.len());

        for tuple in &self.tuples {
            let projected: Vec<usize> = keep.iter().map(|&i| tuple[i]).collect();
            result.insert(projected);
        }

        result
    }

    /// Union of two relations (logical OR).
    pub fn union(&self, other: &Self) -> Self {
        debug_assert_eq!(self.arity, other.arity);

        let mut result = self.clone();
        for tuple in &other.tuples {
            result.insert(tuple.clone());
        }
        result
    }

    /// Convert to a dense tensor.
    ///
    /// `dims` specifies the size of each dimension.
    pub fn to_dense(&self, dims: &[usize], device: &Device) -> Result<Tensor> {
        debug_assert_eq!(dims.len(), self.arity);

        // Create zero tensor
        let total_size: usize = dims.iter().product();
        let mut data = vec![0.0f32; total_size];

        // Fill in 1s for each tuple
        for tuple in &self.tuples {
            let mut idx = 0;
            let mut stride = 1;
            for i in (0..self.arity).rev() {
                idx += tuple[i] * stride;
                stride *= dims[i];
            }
            if idx < total_size {
                data[idx] = 1.0;
            }
        }

        Tensor::from_vec(data, dims, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut rel = SparseBool::new(2);
        rel.insert(vec![0, 1]);
        rel.insert(vec![1, 2]);

        assert!(rel.contains(&[0, 1]));
        assert!(rel.contains(&[1, 2]));
        assert!(!rel.contains(&[0, 2]));
        assert_eq!(rel.len(), 2);
    }
}