//! Tensor operations for Ein.
//!
//! This module provides the core tensor operations, including einsum.

mod einsum;
mod sparse;

pub use einsum::einsum;
pub use sparse::SparseBool;