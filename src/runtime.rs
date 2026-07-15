use crate::asset::{ModelAssets, VocabularyAssets};
use crate::decoding::{self, DecodeOptions};
use crate::error::{LoadError, TranslateError};
use crate::inference::{Network, compile};
use crate::model::{ModelArchive, Shortlist};
use crate::text::{TokenId, Vocabulary};

/// Options which control loading and preparation of a model.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    /// Expected SHA-256 of the uncompressed Marian model.
    pub expected_model_sha256: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    network: Network,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    shortlist: Shortlist,
    shared_vocab: bool,
    source_eos: TokenId,
    target_eos: TokenId,
    #[cfg(not(target_family = "wasm"))]
    execution: rayon::ThreadPool,
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
        let source_eos = source_vocab.eos_id().ok_or_else(|| {
            LoadError::InvalidSentencePiece("source vocabulary has no EOS token".into())
        })?;
        let target_eos = target_vocab.eos_id().ok_or_else(|| {
            LoadError::InvalidSentencePiece("target vocabulary has no EOS token".into())
        })?;
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

        let network = compile(model)?;

        #[cfg(not(target_family = "wasm"))]
        let execution = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "linguaspark-translator".into())
            .build()
            .map_err(|err| LoadError::ThreadPool(err.to_string()))?;

        Ok(Self {
            network,
            source_vocab,
            target_vocab,
            shortlist,
            shared_vocab,
            source_eos,
            target_eos,
            #[cfg(not(target_family = "wasm"))]
            execution,
        })
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
        self.execute(|| {
            let mut translations = self.translate_batch_inner(&[input], options)?;
            Ok(translations
                .pop()
                .expect("single-input batch returned empty"))
        })
    }

    /// Translate a tensor batch using Marian-compatible padding and beam search.
    ///
    /// The caller controls request scheduling. A `Translator` is one synchronous
    /// execution unit and processes the supplied slice as a single padded batch.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid decode options, tokenization failures or
    /// inference failures.
    pub fn translate_batch<S: AsRef<str>>(
        &self,
        inputs: &[S],
        options: &DecodeOptions,
    ) -> Result<Vec<Translation>, TranslateError> {
        let inputs = inputs.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        self.execute(move || self.translate_batch_inner(&inputs, options))
    }

    fn translate_batch_inner<S: AsRef<str>>(
        &self,
        inputs: &[S],
        options: &DecodeOptions,
    ) -> Result<Vec<Translation>, TranslateError> {
        validate_decode_options(options)?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let sources = inputs
            .iter()
            .map(|input| self.source_vocab.encode(input.as_ref(), true))
            .collect::<Result<Vec<_>, _>>()?;
        let (padded, mask, width, empty_inputs) = pad_sources(&sources, self.source_eos);
        let batch_size = sources.len();
        let shortlist =
            self.shortlist
                .generate_shared(&padded, self.target_vocab.len(), self.shared_vocab);
        let output = self.network.prepare_output(&shortlist)?;
        let encoded = self
            .network
            .encode_batch(&padded, &mask, batch_size, width)?;
        let max_len = ((width as f32) * options.max_length_factor).ceil().max(1.0) as usize;
        let decoded = decoding::decode_batch(decoding::DecodeBatchRequest {
            network: &self.network,
            encoded: &encoded,
            output: &output,
            shortlist: &shortlist,
            forbidden: (!options.allow_unknown).then(|| self.target_vocab.unk_id()),
            eos: self.target_eos,
            empty_inputs: &empty_inputs,
            max_len,
            options,
        })?;
        decoded
            .into_iter()
            .map(|best| {
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
            })
            .collect()
    }

    #[cfg(not(target_family = "wasm"))]
    fn execute<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.execution.install(operation)
    }

    #[cfg(target_family = "wasm")]
    fn execute<R>(&self, operation: impl FnOnce() -> R) -> R {
        operation()
    }
}

fn pad_sources(
    sources: &[Vec<TokenId>],
    eos: TokenId,
) -> (Vec<TokenId>, Vec<bool>, usize, Vec<bool>) {
    let batch_size = sources.len();
    let width = sources.iter().map(Vec::len).max().unwrap_or(0);
    let mut padded = vec![eos; batch_size * width];
    let mut mask = vec![false; padded.len()];
    for (batch, source) in sources.iter().enumerate() {
        for (position, &token) in source.iter().enumerate() {
            let index = position * batch_size + batch;
            padded[index] = token;
            mask[index] = true;
        }
    }
    let empty = sources
        .iter()
        .map(|source| source.as_slice() == [eos])
        .collect();
    (padded, mask, width, empty)
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
    use super::{DecodeOptions, pad_sources, validate_decode_options};

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

    #[test]
    fn pads_sources_like_marian() {
        let sources = vec![vec![10, 11, 0], vec![20, 0], vec![0]];
        let (tokens, mask, width, empty) = pad_sources(&sources, 0);
        assert_eq!(width, 3);
        assert_eq!(tokens, [10, 20, 0, 11, 0, 0, 0, 0, 0]);
        assert_eq!(
            mask,
            [true, true, true, true, true, false, true, false, false]
        );
        assert_eq!(empty, [false, false, true]);
    }
}
