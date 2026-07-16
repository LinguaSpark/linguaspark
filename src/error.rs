use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
/// An error encountered while loading a translation model.
pub enum LoadError {
    /// The Marian model is malformed or inconsistent.
    #[error("invalid Marian model: {0}")]
    InvalidModel(String),

    /// The model uses an architecture or feature that is not supported.
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// The lexical shortlist is malformed or inconsistent.
    #[error("invalid lexical shortlist: {0}")]
    InvalidShortlist(String),

    /// A SentencePiece vocabulary is malformed or incompatible.
    #[error("invalid SentencePiece model: {0}")]
    InvalidSentencePiece(String),
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// An error encountered while creating an inference executor.
pub enum ExecutorError {
    /// The single-threaded Rayon execution context could not be created.
    #[error("failed to create inference executor: {0}")]
    ThreadPool(String),
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// An error encountered while translating text.
pub enum TranslateError {
    /// The input or decode options are invalid.
    #[error("invalid translation input: {0}")]
    InvalidInput(String),

    /// Source encoding or target decoding failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// Model inference failed.
    #[error("inference failed: {0}")]
    Inference(String),
}
