//! Greedy CTC decoding for the PP-OCRv5 recognizer.
//!
//! Class 0 is the CTC blank, classes `1..=dict.len()` map to dictionary lines,
//! and one extra class after the dictionary (when the model has it) is a space.

/// Decoded text plus the mean probability of the emitted characters.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub text: String,
    pub confidence: f64,
}

fn softmax_in_place(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

fn looks_like_probabilities(row: &[f32]) -> bool {
    let sum: f32 = row.iter().sum();
    row.iter().all(|v| (0.0..=1.0).contains(v)) && (sum - 1.0).abs() < 0.05
}

/// Decode one sequence of `steps` x `classes` scores (row-major).
///
/// Scores that are not already probabilities are soft-maxed per step.
pub fn decode(scores: &[f32], steps: usize, classes: usize, dict: &[String]) -> Decoded {
    let mut text = String::new();
    let mut probs = Vec::new();
    let mut prev = 0usize;
    let mut row = vec![0f32; classes];
    for t in 0..steps {
        let Some(src) = scores.get(t * classes..(t + 1) * classes) else {
            break;
        };
        row.copy_from_slice(src);
        if !looks_like_probabilities(&row) {
            softmax_in_place(&mut row);
        }
        let (best, p) = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 {
                    (i, v)
                } else {
                    acc
                }
            });
        if best != 0 && best != prev {
            let ch = if best <= dict.len() {
                dict.get(best - 1).map(String::as_str)
            } else if best == dict.len() + 1 {
                Some(" ")
            } else {
                None
            };
            if let Some(ch) = ch {
                text.push_str(ch);
                probs.push(f64::from(p));
            }
        }
        prev = best;
    }
    let confidence = if probs.is_empty() {
        0.0
    } else {
        probs.iter().sum::<f64>() / probs.len() as f64
    };
    Decoded {
        text: text.trim().to_string(),
        confidence,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn dict() -> Vec<String> {
        ["a", "b", "c"].iter().map(|s| s.to_string()).collect()
    }

    fn one_hot(seq: &[usize], classes: usize, p: f32) -> Vec<f32> {
        let mut v = Vec::new();
        for &c in seq {
            let rest = (1.0 - p) / (classes as f32 - 1.0);
            v.extend((0..classes).map(|i| if i == c { p } else { rest }));
        }
        v
    }

    #[test]
    fn collapses_repeats_and_blanks_and_maps_space() {
        // a a _ a b [space] c  -> "aab c"
        let classes = 5;
        let seq = [1, 1, 0, 1, 2, 4, 3];
        let d = decode(&one_hot(&seq, classes, 0.9), seq.len(), classes, &dict());
        assert_eq!(d.text, "aab c");
        assert!((d.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn softmaxes_logits() {
        let classes = 4;
        let logits = vec![0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        let d = decode(&logits, 2, classes, &dict());
        assert_eq!(d.text, "a");
        assert!(d.confidence > 0.9 && d.confidence < 1.0);
    }

    #[test]
    fn empty_and_short_input() {
        let d = decode(&[], 3, 4, &dict());
        assert_eq!(d.text, "");
        assert_eq!(d.confidence, 0.0);
    }
}
