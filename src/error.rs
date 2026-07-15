use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LoadError {
    #[error("failed to read an asset: {0}")]
    Io(#[from] io::Error),

    #[error("invalid Marian model: {0}")]
    InvalidModel(String),

    #[error("model SHA-256 mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("invalid lexical shortlist: {0}")]
    InvalidShortlist(String),

    #[error("invalid SentencePiece model: {0}")]
    InvalidSentencePiece(String),

    #[error("failed to create translation execution context: {0}")]
    ThreadPool(String),
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TranslateError {
    #[error("invalid translation input: {0}")]
    InvalidInput(String),

    #[error("tokenization failed: {0}")]
    Tokenization(String),

    #[error("inference failed: {0}")]
    Inference(String),
}
