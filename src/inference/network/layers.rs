use crate::error::TranslateError;
use crate::inference::backend::Linear;

const LAYER_NORM_EPSILON: f32 = 1e-6;

pub(super) struct FeedForward {
    pub(super) layers: Vec<Linear>,
    pub(super) norm: LayerNorm,
}

pub(super) struct Ssru {
    pub(super) dim: usize,
    pub(super) candidate: Linear,
    pub(super) forget: Linear,
    pub(super) norm: LayerNorm,
}

pub(super) struct LayerNorm {
    pub(super) dim: usize,
    pub(super) scale: Vec<f32>,
    pub(super) bias: Vec<f32>,
}
impl FeedForward {
    pub(super) fn apply(&self, input: &[f32]) -> Result<Vec<f32>, TranslateError> {
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
    pub(super) fn apply(
        &self,
        input: &[f32],
        cell: &mut [f32],
    ) -> Result<Vec<f32>, TranslateError> {
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
    pub(super) fn apply_residual(
        &self,
        output: &[f32],
        residual: &[f32],
    ) -> Result<Vec<f32>, TranslateError> {
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

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use crate::inference::backend::Linear;

    use super::{LayerNorm, Ssru, sigmoid};

    fn linear(values: Vec<i8>, bias: Option<Vec<f32>>) -> Linear {
        Linear::from_quantized("test", 2, 2, values, 1.0, 1.0, bias).unwrap()
    }

    #[test]
    fn sigmoid_and_layer_norm_are_numerically_valid() {
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

    #[test]
    fn ssru_updates_cell_across_multiple_steps() {
        let ssru = Ssru {
            dim: 2,
            candidate: linear(vec![1, 0, 0, 1], None),
            forget: linear(vec![0; 4], Some(vec![0.0, 0.0])),
            norm: LayerNorm {
                dim: 2,
                scale: vec![1.0, 1.0],
                bias: vec![0.0, 0.0],
            },
        };
        let mut cell = vec![0.0; 2];

        ssru.apply(&[2.0, 4.0], &mut cell).unwrap();
        assert_eq!(cell, [1.0, 2.0]);

        ssru.apply(&[2.0, 4.0], &mut cell).unwrap();
        assert_eq!(cell, [1.5, 3.0]);
    }
}
