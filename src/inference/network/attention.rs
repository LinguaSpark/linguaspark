use crate::error::TranslateError;
use crate::inference::backend::{Linear, MatrixView, batched_matmul_f32};

use super::layers::LayerNorm;
use super::state::EncodedBatch;

#[derive(Clone, Copy)]
struct AttentionShape {
    query_rows: usize,
    key_rows: usize,
    row_groups: usize,
    all_keys_valid: bool,
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
        all_keys_valid: bool,
    ) -> Result<Vec<f32>, TranslateError> {
        let queries = self.query.apply(input)?;
        let keys = self.key.apply(input)?;
        let values = self.value.apply(input)?;
        let groups = batch_size * self.heads;
        let mut query_views = Vec::with_capacity(groups);
        let mut key_views = Vec::with_capacity(groups);
        for batch in 0..batch_size {
            for head in 0..self.heads {
                query_views.push(MatrixView {
                    data: &queries[(batch * width * self.dim) + head * self.head_dim..],
                    rows: width,
                    cols: self.head_dim,
                    row_stride: self.dim,
                    col_stride: 1,
                });
                key_views.push(MatrixView {
                    data: &keys[(batch * width * self.dim) + head * self.head_dim..],
                    rows: self.head_dim,
                    cols: width,
                    row_stride: 1,
                    col_stride: self.dim,
                });
            }
        }
        let mut scores = batched_matmul_f32(&query_views, &key_views)?;
        self.finish_attention(
            input,
            &mut scores,
            AttentionShape {
                query_rows: width,
                key_rows: width,
                row_groups: batch_size,
                all_keys_valid,
            },
            |batch| batch,
            |batch, key| time_major_mask[key * batch_size + batch],
            |batch, head| &values[(batch * width * self.dim) + head * self.head_dim..],
        )
    }

    pub(super) fn apply_cross_batch(
        &self,
        input: &[f32],
        source: &EncodedBatch,
        cache: &CrossCache,
        source_indices: &[usize],
    ) -> Result<Vec<f32>, TranslateError> {
        let queries = self.query.apply(input)?;
        let row_groups = source_indices.len();
        let groups = row_groups * self.heads;
        let mut query_views = Vec::with_capacity(groups);
        let mut key_views = Vec::with_capacity(groups);
        for row_group in 0..row_groups {
            let source_batch = source_indices[row_group];
            for head in 0..self.heads {
                query_views.push(MatrixView {
                    data: &queries[row_group * self.dim + head * self.head_dim..],
                    rows: 1,
                    cols: self.head_dim,
                    row_stride: self.dim,
                    col_stride: 1,
                });
                key_views.push(MatrixView {
                    data: &cache.keys
                        [(source_batch * source.width * self.dim) + head * self.head_dim..],
                    rows: self.head_dim,
                    cols: source.width,
                    row_stride: 1,
                    col_stride: self.dim,
                });
            }
        }
        let mut scores = batched_matmul_f32(&query_views, &key_views)?;
        self.finish_attention(
            input,
            &mut scores,
            AttentionShape {
                query_rows: 1,
                key_rows: source.width,
                row_groups,
                all_keys_valid: source.all_keys_valid,
            },
            |row_group| source_indices[row_group],
            |source_batch, key| source.mask[key * source.batch_size + source_batch],
            |source_batch, head| {
                &cache.values[(source_batch * source.width * self.dim) + head * self.head_dim..]
            },
        )
    }

    fn finish_attention<'a, G, M, V>(
        &self,
        residual: &[f32],
        scores: &mut [f32],
        shape: AttentionShape,
        group_context: G,
        key_is_valid: M,
        value_data: V,
    ) -> Result<Vec<f32>, TranslateError>
    where
        G: Fn(usize) -> usize,
        M: Fn(usize, usize) -> bool,
        V: Fn(usize, usize) -> &'a [f32],
    {
        const MASK_VALUE: f32 = -99_999_999.0;
        let AttentionShape {
            query_rows,
            key_rows,
            row_groups,
            all_keys_valid,
        } = shape;
        let groups = row_groups * self.heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        if all_keys_valid {
            for group_scores in scores.chunks_exact_mut(query_rows * key_rows) {
                for row in group_scores.chunks_exact_mut(key_rows) {
                    for score in row.iter_mut() {
                        *score *= scale;
                    }
                    softmax(row);
                }
            }
        } else {
            for row_group in 0..row_groups {
                let context = group_context(row_group);
                for head in 0..self.heads {
                    let group = row_group * self.heads + head;
                    let group_scores = &mut scores
                        [group * query_rows * key_rows..(group + 1) * query_rows * key_rows];
                    for row in group_scores.chunks_exact_mut(key_rows) {
                        for (key, score) in row.iter_mut().enumerate() {
                            *score = if key_is_valid(context, key) {
                                *score * scale
                            } else {
                                MASK_VALUE
                            };
                        }
                        softmax(row);
                    }
                }
            }
        }

        let mut score_views = Vec::with_capacity(groups);
        let mut value_views = Vec::with_capacity(groups);
        for row_group in 0..row_groups {
            let context = group_context(row_group);
            for head in 0..self.heads {
                let group = row_group * self.heads + head;
                score_views.push(MatrixView {
                    data: &scores[group * query_rows * key_rows..],
                    rows: query_rows,
                    cols: key_rows,
                    row_stride: key_rows,
                    col_stride: 1,
                });
                value_views.push(MatrixView {
                    data: value_data(context, head),
                    rows: key_rows,
                    cols: self.head_dim,
                    row_stride: self.dim,
                    col_stride: 1,
                });
            }
        }
        let attended = batched_matmul_f32(&score_views, &value_views)?;
        let rows = residual.len() / self.dim;
        let mut joined = vec![0.0; residual.len()];
        let attended_head_stride = query_rows * self.head_dim;
        for row_group in 0..row_groups {
            for head in 0..self.heads {
                let group = row_group * self.heads + head;
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
    let inverse_sum = 1.0 / sum;
    for value in values {
        *value *= inverse_sum;
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::backend::Linear;

    use super::{Attention, CrossCache, EncodedBatch, LayerNorm, softmax};

    fn linear(values: Vec<i8>, bias: Option<Vec<f32>>) -> Linear {
        Linear::from_quantized("test", 2, 2, values, 1.0, 1.0, bias).unwrap()
    }

    fn square_linear(dim: usize, values: Vec<i8>) -> Linear {
        Linear::from_quantized("test", dim, dim, values, 1.0, 1.0, None).unwrap()
    }

    fn identity(dim: usize) -> Vec<i8> {
        let mut values = vec![0; dim * dim];
        for index in 0..dim {
            values[index * dim + index] = 1;
        }
        values
    }

    fn self_attention(dim: usize, heads: usize) -> Attention {
        Attention {
            dim,
            heads,
            head_dim: dim / heads,
            query: square_linear(dim, vec![0; dim * dim]),
            key: square_linear(dim, vec![0; dim * dim]),
            value: square_linear(dim, identity(dim)),
            output: square_linear(dim, identity(dim)),
            norm: LayerNorm {
                dim,
                scale: vec![1.0; dim],
                bias: vec![0.0; dim],
            },
        }
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
    fn self_attention_isolates_batches_and_respects_time_major_mask() {
        let attention = self_attention(2, 1);
        let input = [
            2.0, 0.0, 0.0, 2.0, // batch 0
            0.0, 4.0, 4.0, 0.0, // batch 1
        ];
        let output = attention
            .apply_self_batch(
                &input,
                2,
                2,
                &[true, true, false, true], // time-major mask
                false,
            )
            .unwrap();
        let expected = [[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0], [1.0, -1.0]];
        for (actual, expected) in output.chunks_exact(2).zip(expected) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn self_attention_rejoins_multiple_heads_in_order() {
        let attention = self_attention(4, 2);
        let output = attention
            .apply_self_batch(&[1.0, 2.0, 3.0, 4.0], 1, 1, &[true], true)
            .unwrap();
        let expected = [-1.341_640_7, -0.447_213_6, 0.447_213_6, 1.341_640_7];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn cross_attention_maps_source_rows_across_beams() {
        let attention = attention();
        let source = EncodedBatch {
            batch_size: 2,
            width: 2,
            mask: vec![true; 4],
            all_keys_valid: true,
            cross: Vec::new(),
        };
        let cache = CrossCache {
            keys: vec![0.0; 8],
            values: vec![2.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 4.0],
        };
        let output = attention
            .apply_cross_batch(&[0.0; 8], &source, &cache, &[1, 0, 1, 0])
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
            all_keys_valid: false,
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
            .apply_cross_batch(&[0.0; 4], &source, &cache, &[0, 1])
            .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] + 1.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
        assert!((output[3] + 1.0).abs() < 1e-5);
    }
}
