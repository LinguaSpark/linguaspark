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

    /// A vocabulary is malformed or incompatible.
    #[error("invalid vocabulary: {0}")]
    InvalidVocabulary(String),
}

#[derive(Debug, Error)]
/// An error encountered while creating an inference executor.
#[error("failed to create inference executor: {0}")]
pub struct ExecutorError(#[source] pub(crate) rayon::ThreadPoolBuildError);

#[derive(Debug, Error)]
#[non_exhaustive]
/// An error encountered while translating text.
pub enum TranslateError {
    /// The decode options are invalid.
    #[error("invalid translation options: {0}")]
    InvalidOptions(String),

    /// Translation failed while processing the model or text.
    #[error("translation failed: {0}")]
    Runtime(String),
}
