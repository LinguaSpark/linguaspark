use crate::error::TranslateError;
use crate::inference::{DecodeStepRequest, EncodedBatch, Network, PreparedOutput};
use crate::text::TokenId;

/// Controls greedy and beam decoding.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    pub beam_size: usize,
    pub max_length_factor: f32,
    pub allow_unknown: bool,
    pub length_normalization: f32,
    pub word_penalty: f32,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            beam_size: 1,
            max_length_factor: 3.0,
            allow_unknown: false,
            length_normalization: 0.0,
            word_penalty: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DecodeBatchRequest<'a> {
    pub network: &'a Network,
    pub encoded: &'a EncodedBatch,
    pub output: &'a PreparedOutput,
    pub shortlist: &'a [TokenId],
    pub forbidden: Option<TokenId>,
    pub eos: TokenId,
    pub empty_inputs: &'a [bool],
    pub max_len: usize,
    pub options: &'a DecodeOptions,
}

pub(crate) struct DecodedHypothesis {
    pub tokens: Vec<TokenId>,
    pub score: f32,
    pub finished: bool,
}

pub(crate) fn decode_batch(
    request: DecodeBatchRequest<'_>,
) -> Result<Vec<DecodedHypothesis>, TranslateError> {
    let batch_size = request.empty_inputs.len();
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let start = || Hypothesis {
        history: None,
        previous: None,
        score: 0.0,
        length: 0,
        finished: false,
        state_parent: 0,
    };
    let mut beams = (0..batch_size).map(|_| vec![start()]).collect::<Vec<_>>();
    let mut completed = (0..batch_size).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut history = Vec::new();
    let mut active = (0..batch_size).collect::<Vec<_>>();
    let mut state = request.network.new_decoder_state(batch_size);

    for position in 0..request.max_len {
        if active.is_empty() {
            break;
        }
        let active_batch = active.len();
        let beam_width = active
            .iter()
            .map(|&original| beams[original].len())
            .max()
            .unwrap_or(0);
        if beam_width == 0 {
            break;
        }
        let mut previous = Vec::with_capacity(beam_width * active_batch);
        for beam_index in 0..beam_width {
            for &original in &active {
                previous.push(
                    beams[original]
                        .get(beam_index)
                        .and_then(|hypothesis| hypothesis.previous)
                        .or_else(|| (position != 0).then_some(request.eos)),
                );
            }
        }
        let mut next_state = state;
        let mut logits = request.network.decode_step_batch(
            &mut next_state,
            DecodeStepRequest {
                source: request.encoded,
                source_indices: &active,
                beam_size: beam_width,
                previous: &previous,
                position,
                output: request.output,
            },
        )?;
        let vocab = request.shortlist.len();
        let mut next_beams = (0..batch_size).map(|_| Vec::new()).collect::<Vec<_>>();

        for (current_batch, &original) in active.iter().enumerate() {
            if position == 0 && request.empty_inputs[original] {
                completed[original].push(Hypothesis {
                    history: None,
                    previous: None,
                    score: 0.0,
                    length: 1,
                    finished: true,
                    state_parent: current_batch,
                });
                continue;
            }

            let beam = &beams[original];
            let mut expansions = Vec::with_capacity(beam.len() * vocab);
            for (parent, hypothesis) in beam.iter().enumerate() {
                let row_index = parent * active_batch + current_batch;
                let row = &mut logits[row_index * vocab..(row_index + 1) * vocab];
                log_softmax_in_place(row);
                for (&log_prob, &token) in row.iter().zip(request.shortlist) {
                    if Some(token) != request.forbidden {
                        expansions.push(Expansion {
                            parent,
                            parent_row: row_index,
                            token,
                            score: hypothesis.score + log_prob,
                        });
                    }
                }
            }
            let next_beam_size = next_beam_limit(position, beam.len(), request.options.beam_size);
            retain_top_k(&mut expansions, next_beam_size);

            for expansion in expansions {
                let parent = &beam[expansion.parent];
                let finished = expansion.token == request.eos;
                let history_index = if finished {
                    parent.history
                } else {
                    history.push(HistoryNode {
                        parent: parent.history,
                        token: expansion.token,
                    });
                    Some(history.len() - 1)
                };
                let hypothesis = Hypothesis {
                    history: history_index,
                    previous: (!finished).then_some(expansion.token),
                    score: expansion.score,
                    length: parent.length + 1,
                    finished,
                    state_parent: expansion.parent_row,
                };
                if finished {
                    completed[original].push(hypothesis);
                } else {
                    next_beams[original].push(hypothesis);
                }
            }
        }

        beams = next_beams;
        if position + 1 >= request.max_len {
            break;
        }
        let next_active = active_sentences(&active, &beams);
        if next_active.is_empty() {
            break;
        }
        let parents = parent_state_rows(&beams, &next_active);
        state = request.network.select_decoder_state(&next_state, &parents);
        active = next_active;
    }

    let mut results = Vec::with_capacity(batch_size);
    for original in 0..batch_size {
        completed[original].append(&mut beams[original]);
        let best = completed[original]
            .iter()
            .max_by(|a, b| {
                normalized_score(a, request.options)
                    .total_cmp(&normalized_score(b, request.options))
            })
            .ok_or_else(|| TranslateError::Inference("beam search returned no result".into()))?;
        results.push(DecodedHypothesis {
            tokens: materialize_history(&history, best.history),
            score: normalized_score(best, request.options),
            finished: best.finished,
        });
    }
    Ok(results)
}

fn next_beam_limit(position: usize, current_size: usize, configured_size: usize) -> usize {
    if position == 0 {
        configured_size
    } else {
        current_size
    }
}

fn active_sentences(active: &[usize], beams: &[Vec<Hypothesis>]) -> Vec<usize> {
    active
        .iter()
        .copied()
        .filter(|&original| !beams[original].is_empty())
        .collect()
}

fn parent_state_rows(beams: &[Vec<Hypothesis>], active: &[usize]) -> Vec<usize> {
    let beam_width = active
        .iter()
        .map(|&original| beams[original].len())
        .max()
        .unwrap_or(0);
    let mut parents = Vec::with_capacity(beam_width * active.len());
    for beam_index in 0..beam_width {
        for &original in active {
            parents.push(
                beams[original]
                    .get(beam_index)
                    .map_or(0, |hypothesis| hypothesis.state_parent),
            );
        }
    }
    parents
}

#[derive(Clone)]
struct Hypothesis {
    history: Option<usize>,
    previous: Option<TokenId>,
    score: f32,
    length: usize,
    finished: bool,
    state_parent: usize,
}

struct HistoryNode {
    parent: Option<usize>,
    token: TokenId,
}

struct Expansion {
    parent: usize,
    parent_row: usize,
    token: TokenId,
    score: f32,
}

fn materialize_history(history: &[HistoryNode], mut index: Option<usize>) -> Vec<TokenId> {
    let mut tokens = Vec::new();
    while let Some(current) = index {
        tokens.push(history[current].token);
        index = history[current].parent;
    }
    tokens.reverse();
    tokens
}

fn normalized_score(hypothesis: &Hypothesis, options: &DecodeOptions) -> f32 {
    let length = hypothesis.length.max(1) as f32;
    (hypothesis.score - options.word_penalty * length) / length.powf(options.length_normalization)
}

fn log_softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = values
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f32>()
        .ln();
    for value in values {
        *value = *value - max - log_sum;
    }
}

fn retain_top_k(expansions: &mut Vec<Expansion>, count: usize) {
    if expansions.len() > count {
        expansions.select_nth_unstable_by(count, |a, b| b.score.total_cmp(&a.score));
        expansions.truncate(count);
    }
    expansions.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
}

#[cfg(test)]
mod tests {
    use super::{
        DecodeOptions, Expansion, HistoryNode, Hypothesis, active_sentences, log_softmax_in_place,
        materialize_history, next_beam_limit, normalized_score, parent_state_rows, retain_top_k,
    };

    fn hypothesis(score: f32, length: usize) -> Hypothesis {
        Hypothesis {
            history: None,
            previous: None,
            score,
            length,
            finished: true,
            state_parent: 0,
        }
    }

    #[test]
    fn normalizes_score_like_marian() {
        let options = DecodeOptions {
            length_normalization: 1.0,
            ..DecodeOptions::default()
        };
        assert_eq!(normalized_score(&hypothesis(-6.0, 3), &options), -2.0);
    }

    #[test]
    fn applies_word_penalty_before_length_penalty() {
        let options = DecodeOptions {
            length_normalization: 1.0,
            word_penalty: 0.5,
            ..DecodeOptions::default()
        };
        assert_eq!(normalized_score(&hypothesis(-4.0, 2), &options), -2.5);
    }

    #[test]
    fn materializes_parent_linked_history() {
        let history = vec![
            HistoryNode {
                parent: None,
                token: 10,
            },
            HistoryNode {
                parent: Some(0),
                token: 20,
            },
            HistoryNode {
                parent: Some(1),
                token: 30,
            },
        ];
        assert_eq!(materialize_history(&history, Some(2)), [10, 20, 30]);
        assert!(materialize_history(&history, None).is_empty());
    }

    #[test]
    fn log_softmax_is_normalized_and_stable() {
        let mut values = [1000.0, 1001.0, 999.0];
        log_softmax_in_place(&mut values);
        let probability_sum = values.iter().map(|value| value.exp()).sum::<f32>();
        assert!((probability_sum - 1.0).abs() < 1e-6);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn top_k_keeps_best_expansions_in_order() {
        let mut expansions = vec![
            Expansion {
                parent: 0,
                parent_row: 0,
                token: 1,
                score: -3.0,
            },
            Expansion {
                parent: 1,
                parent_row: 1,
                token: 2,
                score: -1.0,
            },
            Expansion {
                parent: 0,
                parent_row: 0,
                token: 3,
                score: -2.0,
            },
        ];
        retain_top_k(&mut expansions, 2);
        assert_eq!(
            expansions.iter().map(|item| item.token).collect::<Vec<_>>(),
            [2, 3]
        );
    }

    #[test]
    fn beam_can_expand_only_on_the_first_step() {
        assert_eq!(next_beam_limit(0, 1, 5), 5);
        assert_eq!(next_beam_limit(1, 3, 5), 3);
        assert_eq!(next_beam_limit(4, 1, 5), 1);
    }

    #[test]
    fn compacts_finished_sentences_and_builds_beam_major_parent_rows() {
        let make = |state_parent| Hypothesis {
            history: None,
            previous: Some(1),
            score: 0.0,
            length: 1,
            finished: false,
            state_parent,
        };
        let beams = vec![
            vec![make(10), make(20)],
            Vec::new(),
            vec![make(12)],
            vec![make(13), make(23), make(33)],
        ];
        let active = active_sentences(&[0, 1, 2, 3], &beams);
        assert_eq!(active, [0, 2, 3]);
        // Beam-major, then active batch. Missing beam slots use Marian's
        // dummy state row zero.
        assert_eq!(
            parent_state_rows(&beams, &active),
            [10, 12, 13, 20, 0, 23, 0, 0, 33]
        );
    }

    #[test]
    fn returned_score_uses_marian_normalization() {
        let options = DecodeOptions {
            length_normalization: 1.0,
            word_penalty: 0.5,
            ..DecodeOptions::default()
        };
        let hyp = hypothesis(-4.0, 2);
        assert_eq!(normalized_score(&hyp, &options), -2.5);
    }
}
