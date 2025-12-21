//! Ein: A Tensor Logic Language
//!
//! Ein implements Pedro Domingos' Tensor Logic - a unified framework where
//! logical rules and Einstein summation are fundamentally the same operation.
//!
//! # Key Insight
//!
//! A Datalog rule like `Aunt(x,z) ← Sister(x,y), Parent(y,z)` is equivalent to
//! the tensor equation `A_xz = H(S_xy · P_yz)` where H is the Heaviside step.

pub mod embedding;
pub mod error;
pub mod runtime;
pub mod syntax;
pub mod tensor;

pub use embedding::{EmbeddingSpace, TrainableEmbeddingSpace};
pub use error::{EinError, Result};
pub use runtime::{KBData, KnowledgeBase, OptimizerType, Runtime, TextData, Tokenizer};
pub use syntax::{parse, parse_equation, Constraint, Expr, Statement, TensorRef, Token};
pub use tensor::{einsum, SparseBool};