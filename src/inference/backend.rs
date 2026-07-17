use rten_gemm::{
    GemmExecutor, GemmInputA, GemmInputB, GemmOptions, GemmUninitOptions, PackedBMatrix,
    QuantParams,
};
use rten_simd::SimdOp;
use rten_tensor::NdTensorView;
use rten_vecmath::Quantize;

use crate::error::{LoadError, TranslateError};
use crate::model::{Tensor, TensorData};

pub(super) struct MatrixView<'a> {
    pub(super) data: &'a [f32],
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
}

pub(super) fn batched_matmul_f32(
    lhs: &[MatrixView<'_>],
    rhs: &[MatrixView<'_>],
) -> Result<Vec<f32>, TranslateError> {
    if lhs.len() != rhs.len() {
        return Err(TranslateError::Runtime(
            "batched matrix inputs have different lengths".into(),
        ));
    }
    if lhs.is_empty() {
        return Ok(Vec::new());
    }
    let output_stride = lhs[0]
        .rows
        .checked_mul(rhs[0].cols)
        .ok_or_else(|| TranslateError::Runtime("matrix output size overflow".into()))?;
    let expected_lhs_rows = lhs[0].rows;
    let expected_inner = lhs[0].cols;
    let expected_rhs_cols = rhs[0].cols;
    let mut lhs_matrices = Vec::with_capacity(lhs.len());
    let mut rhs_matrices = Vec::with_capacity(rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        if lhs.cols != rhs.rows
            || lhs.rows != expected_lhs_rows
            || lhs.cols != expected_inner
            || rhs.cols != expected_rhs_cols
        {
            return Err(TranslateError::Runtime(
                "batched matrix dimensions do not match".into(),
            ));
        }
        lhs_matrices.push(
            NdTensorView::from_slice_with_strides(
                [lhs.rows, lhs.cols],
                lhs.data,
                [lhs.row_stride, lhs.col_stride],
            )
            .map_err(|err| TranslateError::Runtime(format!("invalid lhs matrix view: {err}")))?,
        );
        rhs_matrices.push(
            NdTensorView::from_slice_with_strides(
                [rhs.rows, rhs.cols],
                rhs.data,
                [rhs.row_stride, rhs.col_stride],
            )
            .map_err(|err| TranslateError::Runtime(format!("invalid rhs matrix view: {err}")))?,
        );
    }
    let lhs_inputs = lhs_matrices
        .iter()
        .copied()
        .map(GemmInputA::Unpacked)
        .collect::<Vec<_>>();
    let rhs_inputs = rhs_matrices
        .iter()
        .copied()
        .map(GemmInputB::Unpacked)
        .collect::<Vec<_>>();
    let output_len = lhs
        .len()
        .checked_mul(output_stride)
        .ok_or_else(|| TranslateError::Runtime("batched matrix output size overflow".into()))?;
    let mut output = vec![0.0; output_len];
    let executor = GemmExecutor::<f32, f32, f32>::default();
    // RTen's batched API parallelizes across heads with Rayon, which regresses
    // the small head matrices used by these translation models. Calling one
    // executor sequentially avoids that scheduling overhead; each individual
    // GEMM can still use RTen's optimized kernel.
    for ((lhs, rhs), output) in lhs_inputs
        .iter()
        .zip(&rhs_inputs)
        .zip(output.chunks_exact_mut(output_stride))
    {
        // beta=0 means RTen does not read the initialized zeroes.
        executor
            .gemm(output, *lhs, *rhs, GemmOptions::default())
            .map_err(|err| TranslateError::Runtime(format!("RTen f32 GEMM failed: {err}")))?;
    }
    Ok(output)
}

/// A compiled affine transform backed exclusively by `RTen`'s int8 GEMM.
pub(super) struct Linear {
    input_dim: usize,
    output_dim: usize,
    weight: PackedBMatrix<i8>,
    activation_multiplier: f32,
    weight_multiplier: f32,
    bias: Vec<f32>,
}

impl Linear {
    pub(super) fn shape(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }

    pub(super) fn compile(
        tensor: Tensor,
        bias: Option<Vec<f32>>,
        activation_multiplier: f32,
    ) -> Result<Self, LoadError> {
        if tensor.shape.len() != 2 {
            return Err(LoadError::InvalidModel(format!(
                "linear weight {} has rank {}",
                tensor.name,
                tensor.shape.len()
            )));
        }
        let input_dim = tensor.shape[0];
        let output_dim = tensor.shape[1];
        let TensorData::QuantizedI8 {
            values,
            multiplier: weight_multiplier,
        } = tensor.data
        else {
            return Err(LoadError::InvalidModel(format!(
                "linear weight {} is not intgemm8",
                tensor.name
            )));
        };

        Self::from_quantized(
            &tensor.name,
            input_dim,
            output_dim,
            values,
            weight_multiplier,
            activation_multiplier,
            bias,
        )
    }

    pub(super) fn from_quantized(
        name: &str,
        input_dim: usize,
        output_dim: usize,
        values: Vec<i8>,
        weight_multiplier: f32,
        activation_multiplier: f32,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, LoadError> {
        if !activation_multiplier.is_finite() || activation_multiplier <= 0.0 {
            return Err(LoadError::InvalidModel(format!(
                "linear weight {name} has invalid activation multiplier {activation_multiplier}"
            )));
        }
        let expected_values = input_dim
            .checked_mul(output_dim)
            .ok_or_else(|| LoadError::InvalidModel(format!("linear weight {name} is too large")))?;
        if values.len() != expected_values {
            return Err(LoadError::InvalidModel(format!(
                "linear weight {name} has {} values, expected {expected_values}",
                values.len()
            )));
        }
        if !weight_multiplier.is_finite() || weight_multiplier <= 0.0 {
            return Err(LoadError::InvalidModel(format!(
                "linear weight {name} has invalid weight multiplier {weight_multiplier}"
            )));
        }

        // Marian stores logical [K, N] matrices physically as row-major [N, K].
        let view = transposed_view(input_dim, output_dim, &values, name)?;
        let weight = GemmExecutor::<u8, i8, i32>::default().prepack_b(view);

        let bias = bias.unwrap_or_else(|| vec![0.0; output_dim]);
        if bias.len() != output_dim {
            return Err(LoadError::InvalidModel(format!(
                "bias for {} has {} values, expected {output_dim}",
                name,
                bias.len()
            )));
        }

        Ok(Self {
            input_dim,
            output_dim,
            weight,
            activation_multiplier,
            weight_multiplier,
            bias,
        })
    }

    pub(super) fn apply(&self, input: &[f32]) -> Result<Vec<f32>, TranslateError> {
        if !input.len().is_multiple_of(self.input_dim) {
            return Err(TranslateError::Runtime(format!(
                "linear input length {} is not divisible by {}",
                input.len(),
                self.input_dim
            )));
        }
        let rows = input.len() / self.input_dim;
        let mut output = self.apply_int8(
            input,
            rows,
            &self.weight,
            self.activation_multiplier,
            self.weight_multiplier,
        )?;
        for row in output.chunks_exact_mut(self.output_dim) {
            for (value, bias) in row.iter_mut().zip(&self.bias) {
                *value += bias;
            }
        }
        Ok(output)
    }

    fn apply_int8(
        &self,
        input: &[f32],
        rows: usize,
        weight: &PackedBMatrix<i8>,
        activation_multiplier: f32,
        weight_multiplier: f32,
    ) -> Result<Vec<f32>, TranslateError> {
        // Marian Int8Shift quantizes symmetrically to [-127, 127], then adds
        // 127 before the unsigned x signed multiply. RTen's zero-point
        // correction is mathematically equivalent to intgemm PrepareBias.
        let mut quantized = vec![std::mem::MaybeUninit::uninit(); input.len()];
        let quantized = quantize_marian_shift_slice(input, &mut quantized, activation_multiplier);
        let lhs = NdTensorView::from_data([rows, self.input_dim], &*quantized);
        let mut output = vec![std::mem::MaybeUninit::uninit(); rows * self.output_dim];
        let zero_point = vec![127u8; rows];
        let executor = GemmExecutor::<u8, i8, i32>::default();
        let output = executor
            .gemm_uninit(
                &mut output,
                GemmInputA::Unpacked(lhs),
                GemmInputB::Packed(weight),
                GemmUninitOptions {
                    a_quant: Some(QuantParams {
                        zero_point: zero_point.as_slice(),
                    }),
                    ..Default::default()
                },
            )
            .map_err(|err| TranslateError::Runtime(format!("RTen int8 GEMM failed: {err}")))?;
        let scale = 1.0 / (activation_multiplier * weight_multiplier);
        Ok(output.iter().map(|&value| value as f32 * scale).collect())
    }
}

fn quantize_marian_shift_slice<'a>(
    input: &[f32],
    output: &'a mut [std::mem::MaybeUninit<u8>],
    multiplier: f32,
) -> &'a mut [u8] {
    let quantized = Quantize::new(input, output, multiplier, 127u8).dispatch();
    // RTen saturates u8 quantization to 255. Marian Int8Shift first clamps
    // the signed value to 127 and then adds 127, so its maximum is 254.
    for value in quantized.iter_mut() {
        *value = (*value).min(254);
    }
    quantized
}

#[cfg(test)]
fn quantize_marian_shift(value: f32, multiplier: f32) -> u8 {
    let signed = (value * multiplier).round_ties_even().clamp(-127.0, 127.0) as i16;
    (signed + 127) as u8
}

fn transposed_view<'a, T: Copy>(
    input_dim: usize,
    output_dim: usize,
    values: &'a [T],
    name: &str,
) -> Result<rten_tensor::Matrix<'a, T>, LoadError> {
    NdTensorView::from_slice_with_strides([input_dim, output_dim], values, [1, input_dim])
        .map_err(|err| LoadError::InvalidModel(format!("invalid layout for {name}: {err}")))
}

#[cfg(test)]
mod tests {
    use crate::model::{Tensor, TensorData};

    use super::{Linear, quantize_marian_shift, quantize_marian_shift_slice};

    fn tensor(shape: [usize; 2], values: Vec<i8>, multiplier: f32) -> Tensor {
        Tensor {
            name: "test_weight".into(),
            shape: shape.to_vec(),
            data: TensorData::QuantizedI8 { values, multiplier },
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn quantizes_marian_shift_with_ties_to_even() {
        assert_eq!(quantize_marian_shift(0.0, 1.0), 127);
        assert_eq!(quantize_marian_shift(0.5, 1.0), 127);
        assert_eq!(quantize_marian_shift(1.5, 1.0), 129);
        assert_eq!(quantize_marian_shift(-1.5, 1.0), 125);
    }

    #[test]
    fn quantization_clamps_to_marian_range() {
        assert_eq!(quantize_marian_shift(1000.0, 1.0), 254);
        assert_eq!(quantize_marian_shift(-1000.0, 1.0), 0);
    }

    #[test]
    fn simd_quantization_matches_marian_reference() {
        let input = [
            -1000.0, -127.5, -126.5, -0.5, 0.0, 0.5, 126.5, 127.5, 1000.0,
        ];
        let mut output = vec![std::mem::MaybeUninit::uninit(); input.len()];
        let actual = quantize_marian_shift_slice(&input, &mut output, 1.0);
        let expected = input
            .iter()
            .map(|&value| quantize_marian_shift(value, 1.0))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn linear_rejects_invalid_rank() {
        let mut tensor = tensor([2, 2], vec![1; 4], 1.0);
        tensor.shape = vec![4];
        assert!(Linear::compile(tensor, None, 1.0).is_err());
    }

    #[test]
    fn linear_rejects_invalid_activation_multiplier_and_bias() {
        assert!(Linear::compile(tensor([2, 2], vec![1; 4], 1.0), None, 0.0).is_err());
        assert!(Linear::compile(tensor([2, 2], vec![1; 4], 1.0), Some(vec![0.0]), 1.0).is_err());
    }

    #[test]
    fn linear_applies_single_row_with_bias() {
        // Marian stores the two output columns physically as [N, K].
        let linear = Linear::compile(
            tensor([2, 2], vec![3, 4, -1, 2], 1.0),
            Some(vec![0.5, -0.5]),
            1.0,
        )
        .unwrap();
        assert_close(&linear.apply(&[1.0, 2.0]).unwrap(), &[11.5, 2.5]);
    }

    #[test]
    fn linear_applies_multiple_rows() {
        let linear = Linear::compile(tensor([2, 2], vec![3, 4, -1, 2], 1.0), None, 1.0).unwrap();
        assert_close(
            &linear.apply(&[1.0, 2.0, -2.0, 1.0]).unwrap(),
            &[11.0, 3.0, -2.0, 4.0],
        );
    }

    #[test]
    fn linear_matches_quantized_reference() {
        let weights = [2i8, -3, 4, 1];
        let linear = Linear::compile(tensor([2, 2], weights.to_vec(), 2.0), None, 4.0).unwrap();
        let input = [0.5, -0.25];
        let quantized = input.map(|value| i32::from(quantize_marian_shift(value, 4.0)) - 127);
        let expected = [
            (quantized[0] * 2 + quantized[1] * -3) as f32 / 8.0,
            (quantized[0] * 4 + quantized[1]) as f32 / 8.0,
        ];
        assert_close(&linear.apply(&input).unwrap(), &expected);
    }
}
