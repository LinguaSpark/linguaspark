use crate::error::{LoadError, TranslateError};
use crate::inference::backend::{Linear, MatrixView, batched_matmul_f32};
use crate::inference::embedding::{Embedding, OutputProjection, PreparedOutput};
use crate::model::{ModelArchive, ModelConfig, ModelMetadata, Tensor, TensorData};
use crate::text::TokenId;

const LAYER_NORM_EPSILON: f32 = 1e-6;

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

impl NetworkSpec {
    fn from_config(config: &ModelConfig) -> Result<Self, LoadError> {
        let supported = config.model_type == "transformer"
            && config.transformer_ffn_activation == "relu"
            && config.transformer_decoder_autoreg == "rnn"
            && config.dec_cell == "ssru"
            && !config.layer_normalization
            && (config.tied_embeddings || config.tied_embeddings_all)
            && (!config.tied_embeddings_all || config.dim_vocabs[0] == config.dim_vocabs[1])
            && config.transformer_postprocess == "dan"
            && config.transformer_preprocess.is_empty()
            && config.transformer_postprocess_top.is_empty();
        if !supported {
            return Err(LoadError::UnsupportedArchitecture(format!(
                "type={}, autoreg={}, cell={}, activation={}, preprocess={:?}, postprocess={:?}, postprocess-top={:?}, layer-normalization={}, tied={}, tied-all={}",
                config.model_type,
                config.transformer_decoder_autoreg,
                config.dec_cell,
                config.transformer_ffn_activation,
                config.transformer_preprocess,
                config.transformer_postprocess,
                config.transformer_postprocess_top,
                config.layer_normalization,
                config.tied_embeddings,
                config.tied_embeddings_all,
            )));
        }
        Ok(Self {
            dim: config.dim_emb,
            heads: config.transformer_heads,
            encoder_layers: config.enc_depth,
            decoder_layers: config.dec_depth,
            ffn_dim: config.transformer_dim_ffn,
            ffn_depth: config.transformer_ffn_depth,
            source_vocab: config.dim_vocabs[0],
            target_vocab: config.dim_vocabs[1],
        })
    }
}

pub(crate) struct Network {
    spec: NetworkSpec,
    source_embedding: Embedding,
    target_embedding: Embedding,
    output: OutputProjection,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
}

pub(crate) struct EncodedSource {
    len: usize,
    cross: Vec<CrossCache>,
}

#[derive(Clone)]
pub(crate) struct DecoderState {
    cells: Vec<Vec<f32>>,
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

struct Attention {
    dim: usize,
    heads: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    norm: LayerNorm,
}

struct CrossCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct FeedForward {
    layers: Vec<Linear>,
    norm: LayerNorm,
}

struct Ssru {
    dim: usize,
    candidate: Linear,
    forget: Linear,
    norm: LayerNorm,
}

struct LayerNorm {
    dim: usize,
    scale: Vec<f32>,
    bias: Vec<f32>,
}

impl Network {
    pub(crate) fn compile(mut model: ModelArchive) -> Result<(Self, ModelMetadata), LoadError> {
        let spec = NetworkSpec::from_config(&model.config)?;
        if spec.heads == 0 || !spec.dim.is_multiple_of(spec.heads) {
            return Err(LoadError::InvalidModel(format!(
                "embedding dimension {} is not divisible by {} attention heads",
                spec.dim, spec.heads
            )));
        }
        if spec.ffn_depth == 0 {
            return Err(LoadError::InvalidModel(
                "transformer FFN depth must be positive".into(),
            ));
        }

        let (source_embedding, target_embedding, output_weight_name) =
            if model.config.tied_embeddings_all {
                let shared =
                    Embedding::compile(take(&mut model, "Wemb")?, spec.target_vocab, spec.dim)?;
                (shared.clone(), shared, "Wemb")
            } else {
                (
                    Embedding::compile(
                        take(&mut model, "encoder_Wemb")?,
                        spec.source_vocab,
                        spec.dim,
                    )?,
                    Embedding::compile(
                        take(&mut model, "decoder_Wemb")?,
                        spec.target_vocab,
                        spec.dim,
                    )?,
                    "decoder_Wemb",
                )
            };
        let output_bias = take_f32(&mut model, "decoder_ff_logit_out_b")?;
        let output_activation_multiplier =
            take_scalar(&mut model, &format!("{output_weight_name}_QuantMultA"))?;
        let output =
            OutputProjection::new(&target_embedding, output_activation_multiplier, output_bias)?;

        let mut encoder = Vec::with_capacity(spec.encoder_layers);
        for layer in 1..=spec.encoder_layers {
            encoder.push(EncoderLayer {
                attention: Attention::compile(&mut model, &format!("encoder_l{layer}_self"), spec)?,
                ffn: FeedForward::compile(&mut model, &format!("encoder_l{layer}_ffn"), spec)?,
            });
        }

        let mut decoder = Vec::with_capacity(spec.decoder_layers);
        for layer in 1..=spec.decoder_layers {
            decoder.push(DecoderLayer {
                ssru: Ssru::compile(&mut model, &format!("decoder_l{layer}_rnn"), spec.dim)?,
                attention: Attention::compile(
                    &mut model,
                    &format!("decoder_l{layer}_context"),
                    spec,
                )?,
                ffn: FeedForward::compile(&mut model, &format!("decoder_l{layer}_ffn"), spec)?,
            });
        }

        let network = Self {
            spec,
            source_embedding,
            target_embedding,
            output,
            encoder,
            decoder,
        };
        Ok((network, model.into_metadata()))
    }

    pub(crate) fn new_decoder_state(&self, rows: usize) -> DecoderState {
        DecoderState {
            cells: vec![vec![0.0; rows * self.spec.dim]; self.spec.decoder_layers],
        }
    }

    pub(crate) fn select_decoder_state(
        &self,
        state: &DecoderState,
        rows: &[usize],
    ) -> DecoderState {
        let cells = state
            .cells
            .iter()
            .map(|layer| {
                let mut selected = Vec::with_capacity(rows.len() * self.spec.dim);
                for &row in rows {
                    let start = row * self.spec.dim;
                    selected.extend_from_slice(&layer[start..start + self.spec.dim]);
                }
                selected
            })
            .collect();
        DecoderState { cells }
    }

    pub(crate) fn prepare_output(
        &self,
        shortlist: &[TokenId],
    ) -> Result<PreparedOutput, TranslateError> {
        self.output.prepare(shortlist)
    }

    pub(crate) fn encode(&self, source: &[TokenId]) -> Result<EncodedSource, TranslateError> {
        let len = source.len();
        let mut hidden = self.source_embedding.lookup(source)?;
        scale_and_add_positions(&mut hidden, len, 0, self.spec.dim);
        for layer in &self.encoder {
            hidden = layer.apply(&hidden, len)?;
        }

        let mut cross = Vec::with_capacity(self.decoder.len());
        for layer in &self.decoder {
            cross.push(CrossCache {
                keys: layer.attention.key.apply(&hidden)?,
                values: layer.attention.value.apply(&hidden)?,
            });
        }
        Ok(EncodedSource { len, cross })
    }

    pub(crate) fn decode_step(
        &self,
        source: &EncodedSource,
        state: &mut DecoderState,
        previous: &[Option<TokenId>],
        position: usize,
        output: &PreparedOutput,
    ) -> Result<Vec<f32>, TranslateError> {
        let rows = previous.len();
        let mut hidden = self.target_embedding.lookup_optional(previous)?;
        scale_and_add_positions(&mut hidden, rows, position, self.spec.dim);

        for (index, layer) in self.decoder.iter().enumerate() {
            hidden = layer.ssru.apply(&hidden, &mut state.cells[index])?;
            hidden = layer
                .attention
                .apply_cross(&hidden, source.len, &source.cross[index])?;
            hidden = layer.ffn.apply(&hidden)?;
        }

        output.linear.apply(&hidden)
    }
}

impl EncoderLayer {
    fn apply(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, TranslateError> {
        let attended = self.attention.apply_self(input, rows)?;
        self.ffn.apply(&attended)
    }
}

impl Attention {
    fn compile(
        model: &mut ModelArchive,
        prefix: &str,
        spec: NetworkSpec,
    ) -> Result<Self, LoadError> {
        let query = compile_linear(
            model,
            &format!("{prefix}_Wq"),
            Some(&format!("{prefix}_bq")),
        )?;
        let key = compile_linear(
            model,
            &format!("{prefix}_Wk"),
            Some(&format!("{prefix}_bk")),
        )?;
        let value = compile_linear(
            model,
            &format!("{prefix}_Wv"),
            Some(&format!("{prefix}_bv")),
        )?;
        let output = compile_linear(
            model,
            &format!("{prefix}_Wo"),
            Some(&format!("{prefix}_bo")),
        )?;
        for (name, linear) in [
            ("query", &query),
            ("key", &key),
            ("value", &value),
            ("output", &output),
        ] {
            require_linear_shape(linear, spec.dim, spec.dim, &format!("{prefix} {name}"))?;
        }
        Ok(Self {
            dim: spec.dim,
            heads: spec.heads,
            head_dim: spec.dim / spec.heads,
            query,
            key,
            value,
            output,
            norm: LayerNorm::compile(model, &format!("{prefix}_Wo_ln"), spec.dim)?,
        })
    }

    fn apply_self(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, TranslateError> {
        let queries = self.query.apply(input)?;
        let keys = self.key.apply(input)?;
        let values = self.value.apply(input)?;
        self.attend(input, &queries, &keys, &values, rows)
    }

    fn apply_cross(
        &self,
        input: &[f32],
        key_rows: usize,
        cache: &CrossCache,
    ) -> Result<Vec<f32>, TranslateError> {
        let queries = self.query.apply(input)?;
        self.attend(input, &queries, &cache.keys, &cache.values, key_rows)
    }

    fn attend(
        &self,
        residual: &[f32],
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        key_rows: usize,
    ) -> Result<Vec<f32>, TranslateError> {
        let query_rows = queries.len() / self.dim;
        let mut joined = vec![0.0; query_rows * self.dim];
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let query_views = (0..self.heads)
            .map(|head| MatrixView {
                data: &queries[head * self.head_dim..],
                rows: query_rows,
                cols: self.head_dim,
                row_stride: self.dim,
                col_stride: 1,
            })
            .collect::<Vec<_>>();
        let key_views = (0..self.heads)
            .map(|head| MatrixView {
                data: &keys[head * self.head_dim..],
                rows: self.head_dim,
                cols: key_rows,
                row_stride: 1,
                col_stride: self.dim,
            })
            .collect::<Vec<_>>();
        let mut scores = batched_matmul_f32(&query_views, &key_views)?;
        for row in scores.chunks_exact_mut(key_rows) {
            for score in row.iter_mut() {
                *score *= scale;
            }
            softmax(row);
        }

        let score_head_stride = query_rows * key_rows;
        let score_views = (0..self.heads)
            .map(|head| MatrixView {
                data: &scores[head * score_head_stride..],
                rows: query_rows,
                cols: key_rows,
                row_stride: key_rows,
                col_stride: 1,
            })
            .collect::<Vec<_>>();
        let value_views = (0..self.heads)
            .map(|head| MatrixView {
                data: &values[head * self.head_dim..],
                rows: key_rows,
                cols: self.head_dim,
                row_stride: self.dim,
                col_stride: 1,
            })
            .collect::<Vec<_>>();
        let attended = batched_matmul_f32(&score_views, &value_views)?;
        let attended_head_stride = query_rows * self.head_dim;
        for head in 0..self.heads {
            let offset = head * self.head_dim;
            let head_values =
                &attended[head * attended_head_stride..(head + 1) * attended_head_stride];
            for row in 0..query_rows {
                let source = &head_values[row * self.head_dim..(row + 1) * self.head_dim];
                let start = row * self.dim + offset;
                joined[start..start + self.head_dim].copy_from_slice(source);
            }
        }

        let projected = self.output.apply(&joined)?;
        self.norm.apply_residual(&projected, residual)
    }
}

impl FeedForward {
    fn compile(
        model: &mut ModelArchive,
        prefix: &str,
        spec: NetworkSpec,
    ) -> Result<Self, LoadError> {
        let mut layers = Vec::with_capacity(spec.ffn_depth);
        for depth in 1..=spec.ffn_depth {
            let linear = compile_linear(
                model,
                &format!("{prefix}_W{depth}"),
                Some(&format!("{prefix}_b{depth}")),
            )?;
            let expected_input = if depth == 1 { spec.dim } else { spec.ffn_dim };
            let expected_output = if depth == spec.ffn_depth {
                spec.dim
            } else {
                spec.ffn_dim
            };
            require_linear_shape(
                &linear,
                expected_input,
                expected_output,
                &format!("{prefix} layer {depth}"),
            )?;
            layers.push(linear);
        }
        Ok(Self {
            layers,
            norm: LayerNorm::compile(model, &format!("{prefix}_ffn_ln"), spec.dim)?,
        })
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, TranslateError> {
        let last = self.layers.len() - 1;
        let mut hidden = self.layers[0].apply(input)?;
        if last != 0 {
            for value in &mut hidden {
                *value = value.max(0.0);
            }
        }
        for (index, layer) in self.layers.iter().enumerate().skip(1) {
            hidden = layer.apply(&hidden)?;
            if index != last {
                for value in &mut hidden {
                    *value = value.max(0.0);
                }
            }
        }
        self.norm.apply_residual(&hidden, input)
    }
}

impl Ssru {
    fn compile(model: &mut ModelArchive, prefix: &str, dim: usize) -> Result<Self, LoadError> {
        let candidate = compile_linear(model, &format!("{prefix}_W"), None)?;
        let forget = compile_linear(
            model,
            &format!("{prefix}_Wf"),
            Some(&format!("{prefix}_bf")),
        )?;
        require_linear_shape(&candidate, dim, dim, &format!("{prefix} candidate"))?;
        require_linear_shape(&forget, dim, dim, &format!("{prefix} forget"))?;
        Ok(Self {
            dim,
            candidate,
            forget,
            norm: LayerNorm::compile(model, &format!("{prefix}_ffn_ln"), dim)?,
        })
    }

    fn apply(&self, input: &[f32], cell: &mut [f32]) -> Result<Vec<f32>, TranslateError> {
        let candidate = self.candidate.apply(input)?;
        let forget = self.forget.apply(input)?;
        if cell.len() != input.len() || !input.len().is_multiple_of(self.dim) {
            return Err(TranslateError::Inference(
                "SSRU cell has an invalid dimension".into(),
            ));
        }
        let mut output = vec![0.0; input.len()];
        for index in 0..input.len() {
            let gate = sigmoid(forget[index]);
            cell[index] = gate * cell[index] + (1.0 - gate) * candidate[index];
            output[index] = cell[index].max(0.0);
        }
        self.norm.apply_residual(&output, input)
    }
}

impl LayerNorm {
    fn compile(model: &mut ModelArchive, prefix: &str, dim: usize) -> Result<Self, LoadError> {
        let scale = take_f32(model, &format!("{prefix}_scale"))?;
        let bias = take_f32(model, &format!("{prefix}_bias"))?;
        if scale.len() != dim || bias.len() != dim {
            return Err(LoadError::InvalidModel(format!(
                "layer norm {prefix} has dimension ({}, {}), expected {dim}",
                scale.len(),
                bias.len()
            )));
        }
        Ok(Self { dim, scale, bias })
    }

    fn apply_residual(&self, output: &[f32], residual: &[f32]) -> Result<Vec<f32>, TranslateError> {
        if output.len() != residual.len() || !output.len().is_multiple_of(self.dim) {
            return Err(TranslateError::Inference(
                "invalid layer norm input shape".into(),
            ));
        }
        let mut result = output
            .iter()
            .zip(residual)
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>();
        for row in result.chunks_exact_mut(self.dim) {
            let mean = row.iter().sum::<f32>() / self.dim as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / self.dim as f32;
            let inverse_std = 1.0 / (variance + LAYER_NORM_EPSILON).sqrt();
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value - mean) * inverse_std * self.scale[index] + self.bias[index];
            }
        }
        Ok(result)
    }
}

fn compile_linear(
    model: &mut ModelArchive,
    weight_name: &str,
    bias_name: Option<&str>,
) -> Result<Linear, LoadError> {
    let activation_multiplier = take_scalar(model, &format!("{weight_name}_QuantMultA"))?;
    let weight = take(model, weight_name)?;
    let bias = bias_name.map(|name| take_f32(model, name)).transpose()?;
    Linear::compile(weight, bias, activation_multiplier)
}

fn require_linear_shape(
    linear: &Linear,
    expected_input: usize,
    expected_output: usize,
    label: &str,
) -> Result<(), LoadError> {
    let actual = linear.shape();
    if actual != (expected_input, expected_output) {
        return Err(LoadError::InvalidModel(format!(
            "{label} has shape {}x{}, expected {expected_input}x{expected_output}",
            actual.0, actual.1
        )));
    }
    Ok(())
}

fn take(model: &mut ModelArchive, name: &str) -> Result<Tensor, LoadError> {
    model
        .take_tensor(name)
        .ok_or_else(|| LoadError::InvalidModel(format!("missing required tensor {name}")))
}

fn take_f32(model: &mut ModelArchive, name: &str) -> Result<Vec<f32>, LoadError> {
    let tensor = take(model, name)?;
    match tensor.data {
        TensorData::F32(values) => Ok(values),
        _ => Err(LoadError::InvalidModel(format!(
            "tensor {name} is not float32"
        ))),
    }
}

fn take_scalar(model: &mut ModelArchive, name: &str) -> Result<f32, LoadError> {
    let tensor = take(model, name)?;
    if tensor.element_count() != 1 {
        return Err(LoadError::InvalidModel(format!(
            "tensor {name} is not a scalar"
        )));
    }
    match tensor.data {
        TensorData::F32(values) => Ok(values[0]),
        // Mozilla's precomputed-alpha models pass every parameter through the
        // intgemm serializer, including the one-element activation multiplier.
        // Recovering its scalar value is therefore required by the on-disk
        // format; it is not a fallback for quantized network weights.
        TensorData::QuantizedI8 { values, multiplier } => Ok(f32::from(values[0]) / multiplier),
        TensorData::Bytes(_) => Err(LoadError::InvalidModel(format!(
            "tensor {name} is not a numeric scalar"
        ))),
    }
}

fn scale_and_add_positions(values: &mut [f32], rows: usize, start: usize, dim: usize) {
    let embedding_scale = (dim as f32).sqrt();
    let timescales = dim / 2;
    let log_increment = 10_000.0f32.ln() / (timescales as f32 - 1.0);
    for row in 0..rows {
        let position = (start + row) as f32;
        for index in 0..dim {
            values[row * dim + index] *= embedding_scale;
        }
        for index in 0..timescales {
            let angle = position * (-(index as f32) * log_increment).exp();
            values[row * dim + index] += angle.sin();
            values[row * dim + timescales + index] += angle.cos();
        }
    }
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use crate::model::ModelConfig;

    use super::{LayerNorm, NetworkSpec, scale_and_add_positions, sigmoid, softmax};

    fn config(dim: usize, dec_depth: usize, ffn_dim: usize, tying: &str) -> ModelConfig {
        ModelConfig::parse(&format!(
            r#"
type: transformer
dim-emb: {dim}
dim-vocabs: [32000, 32000]
enc-depth: 6
dec-depth: {dec_depth}
transformer-heads: 8
transformer-dim-ffn: {ffn_dim}
transformer-ffn-depth: 2
transformer-ffn-activation: relu
transformer-decoder-autoreg: rnn
dec-cell: ssru
transformer-postprocess: dan
{tying}: true
version: test
"#
        ))
        .unwrap()
    }

    #[test]
    fn accepts_supported_mozilla_capabilities() {
        for config in [
            config(256, 2, 1536, "tied-embeddings-all"),
            config(384, 4, 1536, "tied-embeddings"),
            config(512, 2, 2048, "tied-embeddings-all"),
        ] {
            NetworkSpec::from_config(&config).unwrap();
        }
    }

    #[test]
    fn rejects_unsupported_model_features() {
        let base = config(384, 4, 1536, "tied-embeddings");
        let mut variants = Vec::new();
        let mut config = base.clone();
        config.model_type = "rnn".into();
        variants.push(config);
        let mut config = base.clone();
        config.dec_cell = "gru".into();
        variants.push(config);
        let mut config = base;
        config.transformer_ffn_activation = "gelu".into();
        variants.push(config);
        assert!(
            variants
                .iter()
                .all(|config| NetworkSpec::from_config(config).is_err())
        );
    }

    #[test]
    fn validates_embedding_tying_modes() {
        let mut neither = config(384, 4, 1536, "tied-embeddings");
        neither.tied_embeddings = false;
        assert!(NetworkSpec::from_config(&neither).is_err());

        let mut both = config(384, 4, 1536, "tied-embeddings");
        both.tied_embeddings_all = true;
        NetworkSpec::from_config(&both).unwrap();

        let mut unequal = config(384, 4, 1536, "tied-embeddings-all");
        unequal.dim_vocabs[1] += 1;
        assert!(NetworkSpec::from_config(&unequal).is_err());
    }

    #[test]
    fn position_encoding_matches_position_zero() {
        let mut values = [0.0; 4];
        scale_and_add_positions(&mut values, 1, 0, 4);
        assert_eq!(values, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn softmax_sigmoid_and_layer_norm_are_numerically_valid() {
        let mut values = [1.0, 2.0, 3.0];
        softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        assert!((sigmoid(0.0) - 0.5).abs() < f32::EPSILON);

        let norm = LayerNorm {
            dim: 2,
            scale: vec![1.0, 1.0],
            bias: vec![0.0, 0.0],
        };
        let normalized = norm.apply_residual(&[1.0, 3.0], &[0.0, 0.0]).unwrap();
        assert!((normalized[0] + 1.0).abs() < 1e-5);
        assert!((normalized[1] - 1.0).abs() < 1e-5);
    }
}
