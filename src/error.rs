//! Error types for Ein.

use thiserror::Error;

/// The main error type for Ein operations.
#[derive(Debug, Error)]
pub enum EinError {
    /// Candle tensor operation failed
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Parse error
    #[error("parse error at {location}: {message}")]
    Parse { location: String, message: String },

    /// Runtime error
    #[error("runtime error: {0}")]
    Runtime(String),

    /// Unknown tensor name
    #[error("unknown tensor: {0}")]
    UnknownTensor(String),

    /// Shape mismatch
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Index error
    #[error("index error: {0}")]
    Index(String),
}

/// Result type for Ein operations.
pub type Result<T> = std::result::Result<T, EinError>;