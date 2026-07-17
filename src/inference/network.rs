use crate::error::TranslateError;
use crate::inference::embedding::{Embedding, OutputProjection, PreparedOutput};
use crate::text::TokenId;

mod attention;
mod compile;
mod layers;
mod position;
mod state;

use attention::{Attention, CrossCache};
pub(crate) use compile::compile;
use layers::{FeedForward, LayerNorm, Ssru};
pub(crate) use state::{DecodeStepRequest, DecoderState, EncodedBatch};

#[derive(Clone, Copy)]
struct NetworkSpec {
    dim: usize,
    heads: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    ffn_dim: usize,
    ffn_depth: usize,
    source_vocab: usize,
    target_vocab: usize,
}

pub(crate) struct Network {
    spec: NetworkSpec,
    source_embedding: Embedding,
    target_embedding: Embedding,
    output: OutputProjection,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
}

struct EncoderLayer {
    attention: Attention,
    ffn: FeedForward,
}

struct DecoderLayer {
    ssru: Ssru,
    attention: Attention,
    ffn: FeedForward,
}

impl Network {
    pub(crate) fn new_decoder_state(&self, rows: usize) -> DecoderState {
        DecoderState::new(self.spec.decoder_layers, rows, self.spec.dim)
    }

    pub(crate) fn select_decoder_state(&self, state: DecoderState, rows: &[usize]) -> DecoderState {
        state.select(rows, self.spec.dim)
    }

    pub(crate) fn prepare_output(
        &self,
        shortlist: &[TokenId],
    ) -> Result<PreparedOutput, TranslateError> {
        self.output.prepare(shortlist)
    }

    pub(crate) fn encode_batch(
        &self,
        source: &[TokenId],
        mask: &[bool],
        batch_size: usize,
        width: usize,
    ) -> Result<EncodedBatch, TranslateError> {
        if source.len() != batch_size * width || mask.len() != source.len() {
            return Err(TranslateError::Runtime(
                "invalid source batch dimensions".into(),
            ));
        }

        // Marian stores input IDs time-major. Attention is simpler with each
        // sentence contiguous, so transpose to [batch, time] after lookup.
        let time_major = self.source_embedding.lookup(source)?;
        let mut hidden = if batch_size == 1 {
            time_major
        } else {
            let mut hidden = vec![0.0; time_major.len()];
            for time in 0..width {
                for batch in 0..batch_size {
                    let source_row = time * batch_size + batch;
                    let target_row = batch * width + time;
                    hidden[target_row * self.spec.dim..(target_row + 1) * self.spec.dim]
                        .copy_from_slice(
                            &time_major
                                [source_row * self.spec.dim..(source_row + 1) * self.spec.dim],
                        );
                }
            }
            hidden
        };
        let all_keys_valid = mask.iter().all(|&valid| valid);
        position::add_batch(&mut hidden, batch_size, width, self.spec.dim);
        for layer in &self.encoder {
            hidden = layer.apply_batch(&hidden, batch_size, width, mask, all_keys_valid)?;
        }

        let mut cross = Vec::with_capacity(self.decoder.len());
        for layer in &self.decoder {
            cross.push(CrossCache {
                keys: layer.attention.key.apply(&hidden)?,
                values: layer.attention.value.apply(&hidden)?,
            });
        }
        Ok(EncodedBatch {
            batch_size,
            width,
            mask: mask.to_vec(),
            all_keys_valid,
            cross,
        })
    }

    pub(crate) fn decode_step_batch(
        &self,
        state: &mut DecoderState,
        request: DecodeStepRequest<'_>,
    ) -> Result<Vec<f32>, TranslateError> {
        let rows = request.previous.len();
        if request.source_indices.len() != rows {
            return Err(TranslateError::Runtime(
                "invalid decoder batch dimensions".into(),
            ));
        }
        let mut hidden = self.target_embedding.lookup_optional(request.previous)?;
        position::add_same(&mut hidden, rows, request.position, self.spec.dim);

        for (index, layer) in self.decoder.iter().enumerate() {
            hidden = layer.ssru.apply(&hidden, &mut state.cells[index])?;
            hidden = layer.attention.apply_cross_batch(
                &hidden,
                request.source,
                &request.source.cross[index],
                request.source_indices,
            )?;
            hidden = layer.ffn.apply(&hidden)?;
        }

        request.output.linear.apply(&hidden)
    }
}

impl EncoderLayer {
    fn apply_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        width: usize,
        time_major_mask: &[bool],
        all_keys_valid: bool,
    ) -> Result<Vec<f32>, TranslateError> {
        let attended = self.attention.apply_self_batch(
            input,
            batch_size,
            width,
            time_major_mask,
            all_keys_valid,
        )?;
        self.ffn.apply(&attended)
    }
}
