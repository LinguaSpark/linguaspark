use crate::asset::{ModelAssets, VocabularyAssets};
use crate::decoding::{self, DecodeOptions};
use crate::error::{ExecutorError, LoadError, TranslateError};
use crate::inference::{Network, compile};
use crate::model::{ModelArchive, Shortlist};
use crate::text::{TokenId, Vocabulary};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
/// The condition that stopped decoding.
pub enum StopReason {
    /// The decoder emitted the end-of-sentence token.
    EndOfSentence,
    /// Decoding reached the configured maximum output length.
    LengthLimit,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
/// The best translation produced for one input.
pub struct Translation {
    /// Decoded target text.
    pub text: String,

    /// Generated target token IDs, excluding the end-of-sentence token.
    pub token_ids: Vec<TokenId>,

    /// Final hypothesis score after length normalization and word penalty.
    ///
    /// This is not a calibrated probability or confidence score.
    pub score: f32,

    /// Why decoding stopped.
    pub stop_reason: StopReason,
}

/// A loaded, immutable translation model.
///
/// Cloning a model is inexpensive and shares the compiled model data.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

struct ModelInner {
    network: Network,
    vocabularies: Vocabularies,
    shortlist: Shortlist,
    source_eos: TokenId,
    target_eos: TokenId,
}

enum Vocabularies {
    Shared(Box<Vocabulary>),
    Separate {
        source: Box<Vocabulary>,
        target: Box<Vocabulary>,
    },
}

impl Vocabularies {
    fn source(&self) -> &Vocabulary {
        match self {
            Self::Shared(vocabulary) => vocabulary,
            Self::Separate { source, .. } => source,
        }
    }

    fn target(&self) -> &Vocabulary {
        match self {
            Self::Shared(vocabulary) => vocabulary,
            Self::Separate { target, .. } => target,
        }
    }

    fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }
}

/// A single-threaded inference execution slot.
///
/// Executors do not own model weights and may run any [`Model`]. Callers that
/// want parallel inference should create multiple executors. Translation
/// requires exclusive access to make the single-operation capacity explicit
/// and to permit future reuse of executor-local workspace.
pub struct Executor {
    #[cfg(not(target_family = "wasm"))]
    execution: rayon::ThreadPool,
}

impl Model {
    /// Load all model assets from owned byte buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when an asset is malformed, the vocabularies and
    /// shortlist do not match the model, or the model uses unsupported layers.
    pub fn from_assets(assets: ModelAssets) -> Result<Self, LoadError> {
        let model = ModelArchive::load(assets.model)?;
        let vocabularies = match assets.vocabularies {
            VocabularyAssets::Shared(bytes) => {
                Vocabularies::Shared(Box::new(Vocabulary::load(bytes)?))
            }
            VocabularyAssets::Separate { source, target } => Vocabularies::Separate {
                source: Box::new(Vocabulary::load(source)?),
                target: Box::new(Vocabulary::load(target)?),
            },
        };
        let source_eos = vocabularies.source().eos_id().ok_or_else(|| {
            LoadError::InvalidSentencePiece("source vocabulary has no EOS token".into())
        })?;
        let target_eos = vocabularies.target().eos_id().ok_or_else(|| {
            LoadError::InvalidSentencePiece("target vocabulary has no EOS token".into())
        })?;
        let shortlist = Shortlist::load(assets.shortlist)?;
        if vocabularies.source().len() != model.config.dim_vocabs[0]
            || vocabularies.target().len() != model.config.dim_vocabs[1]
        {
            return Err(LoadError::InvalidModel(format!(
                "vocabulary sizes ({}, {}) do not match model dimensions ({}, {})",
                vocabularies.source().len(),
                vocabularies.target().len(),
                model.config.dim_vocabs[0],
                model.config.dim_vocabs[1]
            )));
        }
        if shortlist.source_vocab_size() != vocabularies.source().len() {
            return Err(LoadError::InvalidShortlist(format!(
                "shortlist source vocabulary size {} does not match source vocabulary {}",
                shortlist.source_vocab_size(),
                vocabularies.source().len()
            )));
        }
        shortlist.validate_target_vocab(vocabularies.target().len())?;

        let network = compile(model)?;

        Ok(Self {
            inner: Arc::new(ModelInner {
                network,
                vocabularies,
                shortlist,
                source_eos,
                target_eos,
            }),
        })
    }
}

impl Executor {
    /// Create a single-threaded inference executor.
    ///
    /// # Errors
    ///
    /// Returns an error if the native execution context cannot be created.
    pub fn new() -> Result<Self, ExecutorError> {
        #[cfg(not(target_family = "wasm"))]
        let execution = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "linguaspark-executor".into())
            .build()
            .map_err(|err| ExecutorError::ThreadPool(err.to_string()))?;

        Ok(Self {
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
        &mut self,
        model: &Model,
        input: &str,
        options: &DecodeOptions,
    ) -> Result<Translation, TranslateError> {
        self.execute(|| {
            let mut translations = translate_batch_inner(&model.inner, &[input], options)?;
            Ok(translations
                .pop()
                .expect("single-input batch returned empty"))
        })
    }

    /// Translate a tensor batch using Marian-compatible padding and beam search.
    ///
    /// The caller controls request scheduling. An executor is one synchronous
    /// execution unit and processes the supplied slice as a single padded batch.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid decode options, tokenization failures or
    /// inference failures.
    pub fn translate_batch<S: AsRef<str>>(
        &mut self,
        model: &Model,
        inputs: &[S],
        options: &DecodeOptions,
    ) -> Result<Vec<Translation>, TranslateError> {
        let inputs = inputs.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        self.execute(move || translate_batch_inner(&model.inner, &inputs, options))
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

fn translate_batch_inner<S: AsRef<str>>(
    model: &ModelInner,
    inputs: &[S],
    options: &DecodeOptions,
) -> Result<Vec<Translation>, TranslateError> {
    validate_decode_options(options)?;
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let sources = inputs
        .iter()
        .map(|input| model.vocabularies.source().encode(input.as_ref(), true))
        .collect::<Result<Vec<_>, _>>()?;
    let (padded, mask, width, empty_inputs) = pad_sources(&sources, model.source_eos);
    let batch_size = sources.len();
    let shortlist = model.shortlist.generate_shared(
        &padded,
        model.vocabularies.target().len(),
        model.vocabularies.is_shared(),
    );
    let output = model.network.prepare_output(&shortlist)?;
    let encoded = model
        .network
        .encode_batch(&padded, &mask, batch_size, width)?;
    let max_len = ((width as f32) * options.max_length_factor).ceil().max(1.0) as usize;
    let decoded = decoding::decode_batch(decoding::DecodeBatchRequest {
        network: &model.network,
        encoded: &encoded,
        output: &output,
        shortlist: &shortlist,
        forbidden: (!options.allow_unknown).then(|| model.vocabularies.target().unk_id()),
        eos: model.target_eos,
        empty_inputs: &empty_inputs,
        max_len,
        options,
    })?;
    decoded
        .into_iter()
        .map(|best| {
            let text = model.vocabularies.target().decode(&best.tokens)?;
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
    use super::{DecodeOptions, Executor, Model, pad_sources, validate_decode_options};

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_runtime_types_have_expected_thread_safety() {
        assert_send_sync::<Model>();
        assert_send_sync::<Executor>();
    }

    #[test]
    fn executor_uses_one_rayon_thread() {
        let executor = Executor::new().unwrap();
        assert_eq!(executor.execute(rayon::current_num_threads), 1);
    }

    #[test]
    fn executor_survives_panicking_operation() {
        let executor = Executor::new().unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.execute(|| panic!("expected test panic"));
        }));
        assert!(panic.is_err());
        assert_eq!(executor.execute(|| 42), 42);
    }

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
