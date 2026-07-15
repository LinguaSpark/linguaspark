use crate::error::TranslateError;
use crate::inference::backend::{Linear, MatrixView, batched_matmul_f32};

use super::layers::LayerNorm;
use super::state::EncodedBatch;

#[derive(Clone, Copy)]
struct AttentionShape {
    query_rows: usize,
    key_rows: usize,
    groups: usize,
}

pub(super) struct Attention {
    pub(super) dim: usize,
    pub(super) heads: usize,
    pub(super) head_dim: usize,
    pub(super) query: Linear,
    pub(super) key: Linear,
    pub(super) value: Linear,
    pub(super) output: Linear,
    pub(super) norm: LayerNorm,
}

pub(super) struct CrossCache {
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
}

impl Attention {
    pub(super) fn apply_self_batch(
        &self,
        input: &[f32],
        batch_size: usize,
        width: usize,
        time_major_mask: &[bool],
    ) -> Result<Vec<f32>, TranslateError> {
        let queries = self.query.apply(input)?;
        let keys = self.key.apply(input)?;
        let values = self.value.apply(input)?;
        let groups = batch_size * self.heads;
        let query_views = (0..groups)
            .map(|group| {
                let batch = group / self.heads;
                let head = group % self.heads;
                MatrixView {
                    data: &queries[(batch * width * self.dim) + head * self.head_dim..],
                    rows: width,
                    cols: self.head_dim,
                    row_stride: self.dim,
                    col_stride: 1,
                }
            })
            .collect::<Vec<_>>();
        let key_views = (0..groups)
            .map(|group| {
                let batch = group / self.heads;
                let head = group % self.heads;
                MatrixView {
                    data: &keys[(batch * width * self.dim) + head * self.head_dim..],
                    rows: self.head_dim,
                    cols: width,
                    row_stride: 1,
                    col_stride: self.dim,
                }
            })
            .collect::<Vec<_>>();
        let mut scores = batched_matmul_f32(&query_views, &key_views)?;
        self.finish_attention(
            input,
            &mut scores,
            AttentionShape {
                query_rows: width,
                key_rows: width,
                groups,
            },
            |group, key| {
                let batch = group / self.heads;
                time_major_mask[key * batch_size + batch]
            },
            |group| {
                let batch = group / self.heads;
                let head = group % self.heads;
                &values[(batch * width * self.dim) + head * self.head_dim..]
            },
        )
    }

    pub(super) fn apply_cross_batch(
        &self,
        input: &[f32],
        source: &EncodedBatch,
        cache: &CrossCache,
        source_indices: &[usize],
        beam_size: usize,
    ) -> Result<Vec<f32>, TranslateError> {
        let queries = self.query.apply(input)?;
        let active_batch = source_indices.len();
        let groups = beam_size * active_batch * self.heads;
        let query_views = (0..groups)
            .map(|group| {
                let row = group / self.heads;
                let head = group % self.heads;
                MatrixView {
                    data: &queries[row * self.dim + head * self.head_dim..],
                    rows: 1,
                    cols: self.head_dim,
                    row_stride: self.dim,
                    col_stride: 1,
                }
            })
            .collect::<Vec<_>>();
        let key_views = (0..groups)
            .map(|group| {
                let beam_batch = group / self.heads;
                let current_batch = beam_batch % active_batch;
                let source_batch = source_indices[current_batch];
                let head = group % self.heads;
                MatrixView {
                    data: &cache.keys
                        [(source_batch * source.width * self.dim) + head * self.head_dim..],
                    rows: self.head_dim,
                    cols: source.width,
                    row_stride: 1,
                    col_stride: self.dim,
                }
            })
            .collect::<Vec<_>>();
        let mut scores = batched_matmul_f32(&query_views, &key_views)?;
        self.finish_attention(
            input,
            &mut scores,
            AttentionShape {
                query_rows: 1,
                key_rows: source.width,
                groups,
            },
            |group, key| {
                let beam_batch = group / self.heads;
                let current_batch = beam_batch % active_batch;
                let source_batch = source_indices[current_batch];
                source.mask[key * source.batch_size + source_batch]
            },
            |group| {
                let beam_batch = group / self.heads;
                let current_batch = beam_batch % active_batch;
                let source_batch = source_indices[current_batch];
                let head = group % self.heads;
                &cache.values[(source_batch * source.width * self.dim) + head * self.head_dim..]
            },
        )
    }

    fn finish_attention<'a, M, V>(
        &self,
        residual: &[f32],
        scores: &mut [f32],
        shape: AttentionShape,
        key_is_valid: M,
        value_data: V,
    ) -> Result<Vec<f32>, TranslateError>
    where
        M: Fn(usize, usize) -> bool,
        V: Fn(usize) -> &'a [f32],
    {
        const MASK_VALUE: f32 = -99_999_999.0;
        let AttentionShape {
            query_rows,
            key_rows,
            groups,
        } = shape;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for group in 0..groups {
            let group_scores =
                &mut scores[group * query_rows * key_rows..(group + 1) * query_rows * key_rows];
            for row in group_scores.chunks_exact_mut(key_rows) {
                for (key, score) in row.iter_mut().enumerate() {
                    *score = if key_is_valid(group, key) {
                        *score * scale
                    } else {
                        MASK_VALUE
                    };
                }
                softmax(row);
            }
        }

        let score_views = (0..groups)
            .map(|group| MatrixView {
                data: &scores[group * query_rows * key_rows..],
                rows: query_rows,
                cols: key_rows,
                row_stride: key_rows,
                col_stride: 1,
            })
            .collect::<Vec<_>>();
        let value_views = (0..groups)
            .map(|group| MatrixView {
                data: value_data(group),
                rows: key_rows,
                cols: self.head_dim,
                row_stride: self.dim,
                col_stride: 1,
            })
            .collect::<Vec<_>>();
        let attended = batched_matmul_f32(&score_views, &value_views)?;
        let rows = residual.len() / self.dim;
        let mut joined = vec![0.0; residual.len()];
        let attended_head_stride = query_rows * self.head_dim;
        for group in 0..groups {
            let row_group = group / self.heads;
            let head = group % self.heads;
            let offset = head * self.head_dim;
            let head_values =
                &attended[group * attended_head_stride..(group + 1) * attended_head_stride];
            for row in 0..query_rows {
                let source = &head_values[row * self.head_dim..(row + 1) * self.head_dim];
                let output_row = row_group * query_rows + row;
                let start = output_row * self.dim + offset;
                joined[start..start + self.head_dim].copy_from_slice(source);
            }
        }
        debug_assert_eq!(rows * self.dim, joined.len());

        let projected = self.output.apply(&joined)?;
        self.norm.apply_residual(&projected, residual)
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

#[cfg(test)]
mod tests {
    use crate::inference::backend::Linear;

    use super::{Attention, CrossCache, EncodedBatch, LayerNorm, softmax};

    fn linear(values: Vec<i8>, bias: Option<Vec<f32>>) -> Linear {
        Linear::from_quantized("test", 2, 2, values, 1.0, 1.0, bias).unwrap()
    }

    fn attention() -> Attention {
        Attention {
            dim: 2,
            heads: 1,
            head_dim: 2,
            query: linear(vec![1, 0, 0, 1], None),
            key: linear(vec![1, 0, 0, 1], None),
            value: linear(vec![1, 0, 0, 1], None),
            output: linear(vec![1, 0, 0, 1], None),
            norm: LayerNorm {
                dim: 2,
                scale: vec![1.0, 1.0],
                bias: vec![0.0, 0.0],
            },
        }
    }

    #[test]
    fn softmax_is_normalized_and_ordered() {
        let mut values = [1.0, 2.0, 3.0];
        softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn cross_attention_maps_source_rows_across_beams() {
        let attention = attention();
        let source = EncodedBatch {
            batch_size: 2,
            width: 2,
            mask: vec![true; 4],
            cross: Vec::new(),
        };
        let cache = CrossCache {
            keys: vec![0.0; 8],
            values: vec![2.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 4.0],
        };
        let output = attention
            .apply_cross_batch(&[0.0; 8], &source, &cache, &[1, 0], 2)
            .unwrap();

        for (actual, expected) in
            output
                .chunks_exact(2)
                .zip([[-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
        {
            assert!((actual[0] - expected[0]).abs() < 1e-5);
            assert!((actual[1] - expected[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn cross_attention_respects_time_major_mask() {
        let attention = attention();
        let source = EncodedBatch {
            batch_size: 2,
            width: 2,
            // [time 0: batch 0, batch 1, time 1: batch 0, batch 1]
            mask: vec![true, true, false, true],
            cross: Vec::new(),
        };
        let cache = CrossCache {
            keys: vec![0.0; 8],
            values: vec![
                3.0, 0.0, 0.0, 9.0, // batch 0: only the first row is valid
                8.0, 0.0, 0.0, 4.0, // batch 1: both rows are valid
            ],
        };
        let output = attention
            .apply_cross_batch(&[0.0; 4], &source, &cache, &[0, 1], 1)
            .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] + 1.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
        assert!((output[3] + 1.0).abs() < 1e-5);
    }
}
