use crate::asset::{ModelAssets, VocabularyAssets};
use crate::decoding::{self, DecodeOptions};
use crate::error::{LoadError, TranslateError};
use crate::inference::Network;
use crate::model::{ModelArchive, ModelMetadata, Shortlist};
use crate::text::{TokenId, Vocabulary};

/// Options which control loading and preparation of a model.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    /// Expected SHA-256 of the uncompressed Marian model.
    pub expected_model_sha256: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfSentence,
    LengthLimit,
}

#[derive(Debug, Clone)]
pub struct Translation {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub score: f32,
    pub stop_reason: StopReason,
}

/// A loaded translation runtime.
pub struct Translator {
    metadata: ModelMetadata,
    network: Network,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    shortlist: Shortlist,
    shared_vocab: bool,
}

impl Translator {
    /// Load all model assets from owned byte buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when an asset is malformed, the vocabularies and
    /// shortlist do not match the model, or the model uses unsupported layers.
    pub fn from_assets(assets: ModelAssets, options: LoadOptions) -> Result<Self, LoadError> {
        let model = ModelArchive::load(assets.model, options.expected_model_sha256)?;
        let (source_vocab, target_vocab, shared_vocab) = match assets.vocabularies {
            VocabularyAssets::Shared(asset) => {
                let vocabulary = Vocabulary::load(asset)?;
                (vocabulary.clone(), vocabulary, true)
            }
            VocabularyAssets::Split { source, target } => {
                (Vocabulary::load(source)?, Vocabulary::load(target)?, false)
            }
        };
        let shortlist = Shortlist::load(assets.shortlist)?;
        if source_vocab.len() != model.config.dim_vocabs[0]
            || target_vocab.len() != model.config.dim_vocabs[1]
        {
            return Err(LoadError::InvalidModel(format!(
                "vocabulary sizes ({}, {}) do not match model dimensions ({}, {})",
                source_vocab.len(),
                target_vocab.len(),
                model.config.dim_vocabs[0],
                model.config.dim_vocabs[1]
            )));
        }
        if shortlist.source_vocab_size() != source_vocab.len() {
            return Err(LoadError::InvalidShortlist(format!(
                "shortlist source vocabulary size {} does not match source vocabulary {}",
                shortlist.source_vocab_size(),
                source_vocab.len()
            )));
        }
        shortlist.validate_target_vocab(target_vocab.len())?;

        let (network, metadata) = Network::compile(model)?;

        Ok(Self {
            metadata,
            network,
            source_vocab,
            target_vocab,
            shortlist,
            shared_vocab,
        })
    }

    #[must_use]
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn shortlist(&self) -> &Shortlist {
        &self.shortlist
    }

    #[must_use]
    pub fn source_vocab(&self) -> &Vocabulary {
        &self.source_vocab
    }

    #[must_use]
    pub fn target_vocab(&self) -> &Vocabulary {
        &self.target_vocab
    }

    /// Translate one sentence using deterministic greedy or beam decoding.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid decode options, tokenization failures or
    /// inference failures.
    pub fn translate(
        &self,
        input: &str,
        options: &DecodeOptions,
    ) -> Result<Translation, TranslateError> {
        validate_decode_options(options)?;
        let source = self.source_vocab.encode(input, true)?;
        if source.len() == 1 {
            return Ok(Translation {
                text: String::new(),
                token_ids: Vec::new(),
                score: 0.0,
                stop_reason: StopReason::EndOfSentence,
            });
        }

        let shortlist =
            self.shortlist
                .generate_shared(&source, self.target_vocab.len(), self.shared_vocab);
        let output = self.network.prepare_output(&shortlist)?;
        let encoded = self.network.encode(&source)?;
        let eos = self.target_vocab.eos_id().ok_or_else(|| {
            TranslateError::Inference("target vocabulary has no EOS token".into())
        })?;
        let max_len = ((source.len() as f32) * options.max_length_factor)
            .ceil()
            .max(1.0) as usize;
        let best = decoding::decode(decoding::DecodeRequest {
            network: &self.network,
            encoded: &encoded,
            output: &output,
            shortlist: &shortlist,
            forbidden: (!options.allow_unknown).then(|| self.target_vocab.unk_id()),
            eos,
            max_len,
            options,
        })?;
        let text = self.target_vocab.decode(&best.tokens)?;
        Ok(Translation {
            text,
            token_ids: best.tokens,
            score: best.score,
            stop_reason: if best.finished {
                StopReason::EndOfSentence
            } else {
                StopReason::LengthLimit
            },
        })
    }

    /// Translate several inputs sequentially.
    ///
    /// This is an API convenience rather than tensor batching. Input order is
    /// preserved and the first error stops the operation.
    ///
    /// # Errors
    ///
    /// Returns the first error produced while translating the inputs.
    pub fn translate_many<S: AsRef<str>>(
        &self,
        inputs: &[S],
        options: &DecodeOptions,
    ) -> Result<Vec<Translation>, TranslateError> {
        inputs
            .iter()
            .map(|input| self.translate(input.as_ref(), options))
            .collect()
    }
}

fn validate_decode_options(options: &DecodeOptions) -> Result<(), TranslateError> {
    if !options.max_length_factor.is_finite() || options.max_length_factor <= 0.0 {
        return Err(TranslateError::InvalidInput(
            "max_length_factor must be positive and finite".into(),
        ));
    }
    if options.beam_size == 0 {
        return Err(TranslateError::InvalidInput(
            "beam_size must be at least one".into(),
        ));
    }
    if !options.length_normalization.is_finite() || !options.word_penalty.is_finite() {
        return Err(TranslateError::InvalidInput(
            "decode score options must be finite".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DecodeOptions, validate_decode_options};

    #[test]
    fn accepts_default_decode_options() {
        validate_decode_options(&DecodeOptions::default()).unwrap();
    }

    #[test]
    fn rejects_invalid_beam_and_length_factor() {
        let options = DecodeOptions {
            beam_size: 0,
            ..DecodeOptions::default()
        };
        assert!(validate_decode_options(&options).is_err());

        for factor in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let options = DecodeOptions {
                max_length_factor: factor,
                ..DecodeOptions::default()
            };
            assert!(validate_decode_options(&options).is_err());
        }
    }

    #[test]
    fn rejects_non_finite_score_options() {
        let options = DecodeOptions {
            length_normalization: f32::NAN,
            ..DecodeOptions::default()
        };
        assert!(validate_decode_options(&options).is_err());
        let options = DecodeOptions {
            word_penalty: f32::INFINITY,
            ..DecodeOptions::default()
        };
        assert!(validate_decode_options(&options).is_err());
    }
}
