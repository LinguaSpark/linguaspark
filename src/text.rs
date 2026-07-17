use sentencepiece_rs::SentencePieceProcessor;

use crate::error::{LoadError, TranslateError};

/// A vocabulary token ID.
pub type TokenId = u32;

/// `SentencePiece` vocabulary hidden behind a small translation-specific API.
///
/// Keeping the third-party processor private prevents its filesystem helpers
/// and usize-based IDs from leaking into the core inference interfaces.
pub(crate) struct Vocabulary {
    processor: SentencePieceProcessor,
}

impl Vocabulary {
    /// Load a serialized `SentencePiece` model.
    pub(crate) fn load(bytes: Vec<u8>) -> Result<Self, LoadError> {
        let processor = SentencePieceProcessor::from_serialized_model(&bytes)
            .map_err(|err| LoadError::InvalidVocabulary(err.to_string()))?;
        if processor.model().vocab_size() > TokenId::MAX as usize {
            return Err(LoadError::InvalidVocabulary(
                "vocabulary is too large for 32-bit token IDs".into(),
            ));
        }
        Ok(Self { processor })
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.processor.model().vocab_size()
    }

    #[must_use]
    pub(crate) fn eos_id(&self) -> Option<TokenId> {
        self.processor.eos_id().map(|id| id as TokenId)
    }

    #[must_use]
    pub(crate) fn unk_id(&self) -> TokenId {
        self.processor.unk_id() as TokenId
    }

    /// Encode text into vocabulary IDs.
    pub(crate) fn encode(
        &self,
        input: &str,
        add_eos: bool,
    ) -> Result<Vec<TokenId>, TranslateError> {
        let mut ids = self
            .processor
            .encode_to_ids(input)
            .map_err(|err| TranslateError::Runtime(err.to_string()))?
            .into_iter()
            .map(|id| id as TokenId)
            .collect::<Vec<_>>();
        if add_eos {
            let eos = self.eos_id().ok_or_else(|| {
                TranslateError::Runtime("source vocabulary has no EOS token".into())
            })?;
            ids.push(eos);
        }
        Ok(ids)
    }

    /// Decode vocabulary IDs into text.
    pub(crate) fn decode(&self, ids: &[TokenId]) -> Result<String, TranslateError> {
        let ids = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        self.processor
            .decode_ids(&ids)
            .map_err(|err| TranslateError::Runtime(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::Vocabulary;

    #[test]
    fn rejects_invalid_sentencepiece_model() {
        assert!(Vocabulary::load(b"not sentencepiece".to_vec()).is_err());
    }
}
