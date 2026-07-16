#![warn(missing_docs)]

//! Pure Rust inference for Bergamot-compatible Marian translation models.

mod asset;
mod decoding;
mod error;
mod inference;
mod model;
mod runtime;
mod text;

pub use asset::{ModelAssets, VocabularyAssets};
pub use decoding::DecodeOptions;
pub use error::{LoadError, TranslateError};
pub use runtime::{StopReason, Translation, Translator};
pub use text::TokenId;
