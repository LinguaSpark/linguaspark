use crate::text::TokenId;

use crate::inference::embedding::PreparedOutput;

use super::CrossCache;

pub(crate) struct EncodedBatch {
    pub(super) batch_size: usize,
    pub(super) width: usize,
    pub(super) mask: Vec<bool>,
    pub(super) all_keys_valid: bool,
    pub(super) cross: Vec<CrossCache>,
}

pub(crate) struct DecodeStepRequest<'a> {
    pub(crate) source: &'a EncodedBatch,
    pub(crate) source_indices: &'a [usize],
    pub(crate) previous: &'a [Option<TokenId>],
    pub(crate) position: usize,
    pub(crate) output: &'a PreparedOutput,
}

#[derive(Clone)]
pub(crate) struct DecoderState {
    pub(super) cells: Vec<Vec<f32>>,
}

impl DecoderState {
    pub(super) fn new(layers: usize, rows: usize, dim: usize) -> Self {
        Self {
            cells: vec![vec![0.0; rows * dim]; layers],
        }
    }

    pub(super) fn select(self, rows: &[usize], dim: usize) -> Self {
        let selected_len = rows.len() * dim;
        if rows.iter().copied().eq(0..rows.len())
            && self.cells.iter().all(|layer| layer.len() == selected_len)
        {
            return self;
        }
        let cells = self
            .cells
            .into_iter()
            .map(|layer| {
                let mut selected = Vec::with_capacity(rows.len() * dim);
                for &row in rows {
                    let start = row * dim;
                    selected.extend_from_slice(&layer[start..start + dim]);
                }
                selected
            })
            .collect();
        Self { cells }
    }
}

#[cfg(test)]
mod tests {
    use super::DecoderState;

    #[test]
    fn selects_parent_rows_across_all_layers() {
        let state = DecoderState {
            cells: vec![
                vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                vec![100.0, 101.0, 110.0, 111.0, 120.0, 121.0],
            ],
        };
        let selected = state.select(&[2, 0, 2], 2);
        assert_eq!(selected.cells[0], [20.0, 21.0, 0.0, 1.0, 20.0, 21.0]);
        assert_eq!(
            selected.cells[1],
            [120.0, 121.0, 100.0, 101.0, 120.0, 121.0]
        );
    }
}
