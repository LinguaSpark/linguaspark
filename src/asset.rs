/// The assets needed by a Bergamot translation model.
///
/// All bytes must be decompressed before they cross the core library boundary.
pub struct ModelAssets {
    /// Uncompressed Marian model bytes.
    pub model: Vec<u8>,
    /// Source and target vocabulary bytes.
    pub vocabularies: VocabularyAssets,
    /// Uncompressed binary lexical shortlist bytes.
    pub shortlist: Vec<u8>,
}

/// Vocabulary assets used by a translation model.
pub enum VocabularyAssets {
    /// Source and target use the same vocabulary and token-ID space.
    Shared(Vec<u8>),
    /// Source and target use independent vocabularies and token-ID spaces.
    Separate {
        /// Serialized source vocabulary.
        source: Vec<u8>,
        /// Serialized target vocabulary.
        target: Vec<u8>,
    },
}
