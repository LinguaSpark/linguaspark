use crate::error::TranslateError;
use crate::inference::{EncodedSource, Network, PreparedOutput};
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
pub(crate) struct DecodeRequest<'a> {
    pub network: &'a Network,
    pub encoded: &'a EncodedSource,
    pub output: &'a PreparedOutput,
    pub shortlist: &'a [TokenId],
    pub forbidden: Option<TokenId>,
    pub eos: TokenId,
    pub max_len: usize,
    pub options: &'a DecodeOptions,
}

pub(crate) struct DecodedHypothesis {
    pub tokens: Vec<TokenId>,
    pub score: f32,
    pub finished: bool,
}

pub(crate) fn decode(request: DecodeRequest<'_>) -> Result<DecodedHypothesis, TranslateError> {
    let mut beam = vec![Hypothesis {
        history: None,
        previous: None,
        score: 0.0,
        length: 0,
        finished: false,
    }];
    let mut state = request.network.new_decoder_state(1);
    let mut completed = Vec::new();
    let mut history = Vec::new();

    for position in 0..request.max_len {
        let previous = beam
            .iter()
            .map(|hypothesis| hypothesis.previous)
            .collect::<Vec<_>>();
        let mut next_state = state;
        let mut logits = request.network.decode_step(
            request.encoded,
            &mut next_state,
            &previous,
            position,
            request.output,
        )?;
        let vocab = request.shortlist.len();
        let mut expansions = Vec::with_capacity(beam.len() * vocab);
        for (parent, row) in logits.chunks_exact_mut(vocab).enumerate() {
            log_softmax_in_place(row);
            for (&log_prob, &token) in row.iter().zip(request.shortlist) {
                if Some(token) != request.forbidden {
                    expansions.push(Expansion {
                        parent,
                        token,
                        score: beam[parent].score + log_prob,
                    });
                }
            }
        }
        retain_top_k(&mut expansions, request.options.beam_size);

        let mut candidates = Vec::with_capacity(expansions.len());
        let mut parents = Vec::with_capacity(expansions.len());
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
            };
            if finished {
                completed.push(hypothesis);
            } else {
                parents.push(expansion.parent);
                candidates.push(hypothesis);
            }
        }
        if candidates.is_empty() {
            beam = candidates;
            break;
        }
        state = request.network.select_decoder_state(&next_state, &parents);
        beam = candidates;
    }

    // Marian also considers hypotheses which remain active when the maximum
    // length is reached, alongside hypotheses completed by EOS.
    completed.extend(beam);
    let best = completed
        .into_iter()
        .max_by(|a, b| {
            normalized_score(a, request.options).total_cmp(&normalized_score(b, request.options))
        })
        .ok_or_else(|| TranslateError::Inference("beam search returned no result".into()))?;
    Ok(DecodedHypothesis {
        tokens: materialize_history(&history, best.history),
        score: best.score,
        finished: best.finished,
    })
}

#[derive(Clone)]
struct Hypothesis {
    history: Option<usize>,
    previous: Option<TokenId>,
    score: f32,
    length: usize,
    finished: bool,
}

struct HistoryNode {
    parent: Option<usize>,
    token: TokenId,
}

struct Expansion {
    parent: usize,
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
        DecodeOptions, Expansion, HistoryNode, Hypothesis, log_softmax_in_place,
        materialize_history, normalized_score, retain_top_k,
    };

    fn hypothesis(score: f32, length: usize) -> Hypothesis {
        Hypothesis {
            history: None,
            previous: None,
            score,
            length,
            finished: true,
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
                token: 1,
                score: -3.0,
            },
            Expansion {
                parent: 1,
                token: 2,
                score: -1.0,
            },
            Expansion {
                parent: 0,
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
}
