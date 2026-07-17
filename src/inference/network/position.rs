fn add_position(
    values: &mut [f32],
    offset: usize,
    position: f32,
    dim: usize,
    timescales: usize,
    embedding_scale: f32,
    log_increment: f32,
) {
    for index in 0..dim {
        values[offset + index] *= embedding_scale;
    }
    for index in 0..timescales {
        let angle = position * (-(index as f32) * log_increment).exp();
        values[offset + index] += angle.sin();
        values[offset + timescales + index] += angle.cos();
    }
}

fn add_sequence(values: &mut [f32], rows: usize, start: usize, dim: usize) {
    let embedding_scale = (dim as f32).sqrt();
    let timescales = dim / 2;
    let log_increment = 10_000.0f32.ln() / (timescales as f32 - 1.0);
    for row in 0..rows {
        let position = (start + row) as f32;
        add_position(
            values,
            row * dim,
            position,
            dim,
            timescales,
            embedding_scale,
            log_increment,
        );
    }
}

pub(super) fn add_batch(values: &mut [f32], batch_size: usize, width: usize, dim: usize) {
    for batch in 0..batch_size {
        let start = batch * width * dim;
        add_sequence(&mut values[start..start + width * dim], width, 0, dim);
    }
}

pub(super) fn add_same(values: &mut [f32], rows: usize, position: usize, dim: usize) {
    let embedding_scale = (dim as f32).sqrt();
    let timescales = dim / 2;
    let log_increment = 10_000.0f32.ln() / (timescales as f32 - 1.0);
    for row in 0..rows {
        add_position(
            values,
            row * dim,
            position as f32,
            dim,
            timescales,
            embedding_scale,
            log_increment,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{add_same, add_sequence};

    #[test]
    fn position_zero_has_unit_cosines() {
        let mut values = [0.0; 4];
        add_sequence(&mut values, 1, 0, 4);
        assert_eq!(values, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn decoder_position_is_shared_across_rows() {
        let mut values = vec![1.0; 16];
        add_same(&mut values, 2, 3, 8);
        assert_eq!(&values[..8], &values[8..]);
    }
}
