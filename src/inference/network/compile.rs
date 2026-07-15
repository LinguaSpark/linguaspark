use crate::error::LoadError;
use crate::inference::backend::Linear;
use crate::inference::embedding::{Embedding, OutputProjection};
use crate::model::{ModelArchive, ModelConfig, Tensor, TensorData};

use super::{
    Attention, DecoderLayer, EncoderLayer, FeedForward, LayerNorm, Network, NetworkSpec, Ssru,
};

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
                "version={}, type={}, autoreg={}, cell={}, activation={}, preprocess={:?}, postprocess={:?}, postprocess-top={:?}, layer-normalization={}, tied={}, tied-all={}",
                config.version,
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

pub(crate) fn compile(mut model: ModelArchive) -> Result<Network, LoadError> {
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
            attention: Attention::compile(&mut model, &format!("decoder_l{layer}_context"), spec)?,
            ffn: FeedForward::compile(&mut model, &format!("decoder_l{layer}_ffn"), spec)?,
        });
    }

    Ok(Network {
        spec,
        source_embedding,
        target_embedding,
        output,
        encoder,
        decoder,
    })
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
        TensorData::Bytes => Err(LoadError::InvalidModel(format!(
            "tensor {name} is not a numeric scalar"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::model::ModelConfig;

    use super::NetworkSpec;

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
}
