use serde::Deserialize;

use crate::error::LoadError;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct ModelConfig {
    #[serde(rename = "type")]
    pub(crate) model_type: String,
    pub(crate) dim_emb: usize,
    pub(crate) dim_vocabs: Vec<usize>,
    pub(crate) enc_depth: usize,
    pub(crate) dec_depth: usize,
    pub(crate) transformer_heads: usize,
    pub(crate) transformer_dim_ffn: usize,
    pub(crate) transformer_ffn_depth: usize,
    pub(crate) transformer_ffn_activation: String,
    pub(crate) transformer_decoder_autoreg: String,
    pub(crate) dec_cell: String,
    #[serde(default)]
    pub(crate) tied_embeddings: bool,
    #[serde(default)]
    pub(crate) tied_embeddings_all: bool,
    #[serde(default)]
    pub(crate) layer_normalization: bool,
    #[serde(default)]
    pub(crate) transformer_postprocess: String,
    #[serde(default)]
    pub(crate) transformer_preprocess: String,
    #[serde(default)]
    pub(crate) transformer_postprocess_top: String,
    pub(crate) version: String,
}

impl ModelConfig {
    pub(crate) fn parse(yaml: &str) -> Result<Self, LoadError> {
        yaml_serde::from_str(yaml)
            .map_err(|err| LoadError::InvalidModel(format!("invalid special:model.yml: {err}")))
    }

    pub(crate) fn validate_well_formed(&self) -> Result<(), LoadError> {
        let valid = self.dim_emb >= 4
            && self.dim_emb.is_multiple_of(2)
            && self.dim_vocabs.len() == 2
            && self.dim_vocabs.iter().all(|&size| size > 0)
            && self.enc_depth > 0
            && self.dec_depth > 0
            && self.transformer_heads > 0
            && self.dim_emb.is_multiple_of(self.transformer_heads)
            && self.transformer_dim_ffn > 0
            && self.transformer_ffn_depth > 0;

        if valid {
            Ok(())
        } else {
            Err(LoadError::InvalidModel(format!(
                "invalid model dimensions: dim={}, vocabs={:?}, enc-depth={}, dec-depth={}, heads={}, ffn-dim={}, ffn-depth={}",
                self.dim_emb,
                self.dim_vocabs,
                self.enc_depth,
                self.dec_depth,
                self.transformer_heads,
                self.transformer_dim_ffn,
                self.transformer_ffn_depth,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    fn yaml(extra: &str) -> String {
        format!(
            r#"
type: transformer
dim-emb: 384
dim-vocabs: [32000, 32000]
enc-depth: 6
dec-depth: 4
transformer-heads: 8
transformer-dim-ffn: 1536
transformer-ffn-depth: 2
transformer-ffn-activation: relu
transformer-decoder-autoreg: rnn
dec-cell: ssru
version: test
{extra}
"#
        )
    }

    fn config(extra: &str) -> ModelConfig {
        ModelConfig::parse(&yaml(extra)).unwrap()
    }

    #[test]
    fn parses_minimal_config_with_defaults() {
        let config = config("");
        assert!(!config.tied_embeddings);
        assert!(!config.tied_embeddings_all);
        assert!(!config.layer_normalization);
        assert!(config.transformer_preprocess.is_empty());
        assert!(config.transformer_postprocess.is_empty());
        assert!(config.transformer_postprocess_top.is_empty());
        config.validate_well_formed().unwrap();
    }

    #[test]
    fn parses_explicit_postprocess() {
        assert_eq!(
            config("transformer-postprocess: dan").transformer_postprocess,
            "dan"
        );
    }

    #[test]
    fn rejects_malformed_yaml() {
        assert!(ModelConfig::parse("type: [").is_err());
    }

    #[test]
    fn rejects_invalid_embedding_dimensions() {
        for dim in [0, 2, 383] {
            let yaml = yaml("").replace("dim-emb: 384", &format!("dim-emb: {dim}"));
            assert!(
                ModelConfig::parse(&yaml)
                    .unwrap()
                    .validate_well_formed()
                    .is_err()
            );
        }
    }

    #[test]
    fn rejects_invalid_vocabulary_dimensions() {
        for vocabs in ["[]", "[32000]", "[32000, 0]"] {
            let yaml = yaml("").replace("[32000, 32000]", vocabs);
            assert!(
                ModelConfig::parse(&yaml)
                    .unwrap()
                    .validate_well_formed()
                    .is_err()
            );
        }
    }

    #[test]
    fn rejects_zero_depths() {
        for field in ["enc-depth: 6", "dec-depth: 4", "transformer-ffn-depth: 2"] {
            let yaml = yaml("").replace(field, &format!("{}: 0", field.split(':').next().unwrap()));
            assert!(
                ModelConfig::parse(&yaml)
                    .unwrap()
                    .validate_well_formed()
                    .is_err()
            );
        }
    }

    #[test]
    fn rejects_zero_attention_heads() {
        let yaml = yaml("").replace("transformer-heads: 8", "transformer-heads: 0");
        assert!(
            ModelConfig::parse(&yaml)
                .unwrap()
                .validate_well_formed()
                .is_err()
        );
    }

    #[test]
    fn rejects_incompatible_head_dimension() {
        let yaml = yaml("").replace("transformer-heads: 8", "transformer-heads: 7");
        assert!(
            ModelConfig::parse(&yaml)
                .unwrap()
                .validate_well_formed()
                .is_err()
        );
    }
}
