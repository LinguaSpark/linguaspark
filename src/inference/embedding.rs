use std::sync::Arc;

use crate::error::{LoadError, TranslateError};
use crate::inference::backend::Linear;
use crate::model::{Tensor, TensorData};
use crate::text::TokenId;

#[derive(Clone)]
pub(super) struct Embedding {
    rows: usize,
    cols: usize,
    values: Arc<[i8]>,
    multiplier: f32,
}

pub(super) struct OutputProjection {
    dim: usize,
    rows: usize,
    values: Arc<[i8]>,
    weight_multiplier: f32,
    activation_multiplier: f32,
    bias: Vec<f32>,
}

pub(crate) struct PreparedOutput {
    pub(super) linear: Linear,
}

impl Embedding {
    pub(super) fn compile(
        tensor: Tensor,
        expected_rows: usize,
        expected_dim: usize,
    ) -> Result<Self, LoadError> {
        if tensor.shape.as_slice() != [expected_rows, expected_dim] {
            return Err(LoadError::InvalidModel(format!(
                "embedding {} has shape {:?}, expected [{expected_rows}, {expected_dim}]",
                tensor.name, tensor.shape
            )));
        }
        let TensorData::QuantizedI8 { values, multiplier } = tensor.data else {
            return Err(LoadError::InvalidModel(format!(
                "embedding {} is not intgemm8",
                tensor.name
            )));
        };
        Ok(Self {
            rows: tensor.shape[0],
            cols: tensor.shape[1],
            values: values.into(),
            multiplier,
        })
    }

    pub(super) fn lookup(&self, ids: &[TokenId]) -> Result<Vec<f32>, TranslateError> {
        let mut output = Vec::with_capacity(ids.len() * self.cols);
        for &id in ids {
            let id = id as usize;
            if id >= self.rows {
                return Err(TranslateError::Inference(format!(
                    "embedding ID {id} is outside vocabulary {}",
                    self.rows
                )));
            }
            let start = id * self.cols;
            output.extend(
                self.values[start..start + self.cols]
                    .iter()
                    .map(|&value| f32::from(value) / self.multiplier),
            );
        }
        Ok(output)
    }

    pub(super) fn lookup_optional(
        &self,
        ids: &[Option<TokenId>],
    ) -> Result<Vec<f32>, TranslateError> {
        let mut output = vec![0.0; ids.len() * self.cols];
        for (row, id) in ids.iter().copied().enumerate() {
            let Some(id) = id else { continue };
            let id = id as usize;
            if id >= self.rows {
                return Err(TranslateError::Inference(format!(
                    "embedding ID {id} is outside vocabulary {}",
                    self.rows
                )));
            }
            let source = &self.values[id * self.cols..(id + 1) * self.cols];
            let target = &mut output[row * self.cols..(row + 1) * self.cols];
            for (target, &source) in target.iter_mut().zip(source) {
                *target = f32::from(source) / self.multiplier;
            }
        }
        Ok(output)
    }
}

impl OutputProjection {
    pub(super) fn new(
        embedding: &Embedding,
        activation_multiplier: f32,
        bias: Vec<f32>,
    ) -> Result<Self, LoadError> {
        if bias.len() != embedding.rows {
            return Err(LoadError::InvalidModel(format!(
                "output bias has {} values, expected {}",
                bias.len(),
                embedding.rows
            )));
        }
        Ok(Self {
            dim: embedding.cols,
            rows: embedding.rows,
            values: Arc::clone(&embedding.values),
            weight_multiplier: embedding.multiplier,
            activation_multiplier,
            bias,
        })
    }

    pub(super) fn prepare(&self, shortlist: &[TokenId]) -> Result<PreparedOutput, TranslateError> {
        let mut values = Vec::with_capacity(shortlist.len() * self.dim);
        let mut bias = Vec::with_capacity(shortlist.len());
        for &token in shortlist {
            let token = token as usize;
            if token >= self.rows {
                return Err(TranslateError::Inference(format!(
                    "target ID {token} exceeds output vocabulary {}",
                    self.rows
                )));
            }
            values.extend_from_slice(&self.values[token * self.dim..(token + 1) * self.dim]);
            bias.push(self.bias[token]);
        }
        let linear = Linear::from_quantized(
            "decoder_output_shortlist",
            self.dim,
            shortlist.len(),
            values,
            self.weight_multiplier,
            self.activation_multiplier,
            Some(bias),
        )
        .map_err(|err| TranslateError::Inference(err.to_string()))?;
        Ok(PreparedOutput { linear })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::model::{Tensor, TensorData};

    use super::{Embedding, OutputProjection};

    fn tensor(shape: [usize; 2], values: Vec<i8>) -> Tensor {
        Tensor {
            name: "embedding".into(),
            shape: shape.to_vec(),
            data: TensorData::QuantizedI8 {
                values,
                multiplier: 1.0,
            },
        }
    }

    #[test]
    fn compiles_and_looks_up_embedding() {
        let embedding = Embedding::compile(tensor([2, 2], vec![1, 2, 3, 4]), 2, 2).unwrap();
        assert_eq!(embedding.lookup(&[1, 0]).unwrap(), [3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn rejects_embedding_shape_and_type_mismatch() {
        assert!(Embedding::compile(tensor([2, 2], vec![1; 4]), 3, 2).is_err());
        let mut tensor = tensor([2, 2], vec![1; 4]);
        tensor.data = TensorData::F32(vec![1.0; 4]);
        assert!(Embedding::compile(tensor, 2, 2).is_err());
    }

    #[test]
    fn lookup_rejects_out_of_range_token() {
        let embedding = Embedding::compile(tensor([2, 2], vec![1; 4]), 2, 2).unwrap();
        assert!(embedding.lookup(&[2]).is_err());
    }

    #[test]
    fn optional_lookup_uses_zero_start_embedding() {
        let embedding = Embedding::compile(tensor([2, 2], vec![1, 2, 3, 4]), 2, 2).unwrap();
        assert_eq!(
            embedding.lookup_optional(&[None, Some(1)]).unwrap(),
            [0.0, 0.0, 3.0, 4.0]
        );
    }

    #[test]
    fn output_projection_shares_weights_and_selects_rows() {
        let embedding = Embedding::compile(tensor([3, 2], vec![1, 2, 3, 4, 5, 6]), 3, 2).unwrap();
        let output = OutputProjection::new(&embedding, 1.0, vec![0.1, 0.2, 0.3]).unwrap();
        assert!(Arc::ptr_eq(&embedding.values, &output.values));
        let prepared = output.prepare(&[2, 0]).unwrap();
        let logits = prepared.linear.apply(&[1.0, 1.0]).unwrap();
        assert!((logits[0] - 11.3).abs() < 1e-5);
        assert!((logits[1] - 3.1).abs() < 1e-5);
    }
}
