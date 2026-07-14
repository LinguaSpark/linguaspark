use sentencepiece_rs::SentencePieceProcessor;

use crate::asset::Asset;
use crate::error::{LoadError, TranslateError};

/// A vocabulary ID used at the public `LinguaSpark` boundary.
pub type TokenId = u32;

/// `SentencePiece` vocabulary hidden behind a small translation-specific API.
///
/// Keeping the third-party processor private prevents its filesystem helpers
/// and usize-based IDs from leaking into the core inference interfaces.
#[derive(Clone, Debug)]
pub struct Vocabulary {
    processor: SentencePieceProcessor,
}

impl Vocabulary {
    /// Load a serialized `SentencePiece` model.
    ///
    /// # Errors
    ///
    /// Returns an error if decompression or model parsing fails.
    pub fn load(asset: Asset) -> Result<Self, LoadError> {
        let bytes = asset.decode()?;
        let processor = SentencePieceProcessor::from_serialized_model(&bytes)
            .map_err(|err| LoadError::InvalidSentencePiece(err.to_string()))?;
        if processor.model().vocab_size() > TokenId::MAX as usize {
            return Err(LoadError::InvalidSentencePiece(
                "vocabulary is too large for 32-bit token IDs".into(),
            ));
        }
        Ok(Self { processor })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.processor.model().vocab_size()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn eos_id(&self) -> Option<TokenId> {
        self.processor.eos_id().map(|id| id as TokenId)
    }

    #[must_use]
    pub fn unk_id(&self) -> TokenId {
        self.processor.unk_id() as TokenId
    }

    /// Encode text into vocabulary IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if `SentencePiece` encoding fails or EOS was requested
    /// but the model has no EOS token.
    pub fn encode(&self, input: &str, add_eos: bool) -> Result<Vec<TokenId>, TranslateError> {
        let mut ids = self
            .processor
            .encode_to_ids(input)
            .map_err(|err| TranslateError::Tokenization(err.to_string()))?
            .into_iter()
            .map(|id| id as TokenId)
            .collect::<Vec<_>>();
        if add_eos {
            let eos = self.eos_id().ok_or_else(|| {
                TranslateError::InvalidInput("source vocabulary has no EOS token".into())
            })?;
            ids.push(eos);
        }
        Ok(ids)
    }

    /// Decode vocabulary IDs into text.
    ///
    /// # Errors
    ///
    /// Returns an error if `SentencePiece` rejects the IDs.
    pub fn decode(&self, ids: &[TokenId]) -> Result<String, TranslateError> {
        let ids = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        self.processor
            .decode_ids(&ids)
            .map_err(|err| TranslateError::Tokenization(err.to_string()))
    }

    /// Return the serialized piece associated with an ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the ID is outside the vocabulary.
    pub fn piece(&self, id: TokenId) -> Result<&str, TranslateError> {
        self.processor
            .id_to_piece(id as usize)
            .map_err(|err| TranslateError::Tokenization(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use crate::asset::Asset;

    use super::Vocabulary;

    #[test]
    fn rejects_invalid_sentencepiece_model() {
        assert!(Vocabulary::load(Asset::raw(b"not sentencepiece".to_vec())).is_err());
    }
}
