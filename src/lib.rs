//! Pure Rust inference for Bergamot-compatible Marian translation models.
//!
//! The public library API is byte-oriented. Filesystem, network and JavaScript
//! integration belong in adapters such as the bundled CLI or a future WASM
//! wrapper.

mod asset;
mod decoding;
mod error;
mod inference;
mod model;
mod runtime;
mod text;

pub use asset::{Asset, Compression, ModelAssets, VocabularyAssets};
pub use decoding::DecodeOptions;
pub use error::{LoadError, TranslateError};
pub use runtime::{LoadOptions, StopReason, Translation, Translator};
pub use text::TokenId;
