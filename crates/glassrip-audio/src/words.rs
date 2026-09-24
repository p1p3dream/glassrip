//! Building words from whisper tokens.
//!
//! whisper emits sub-word tokens; a token whose text starts with a space begins
//! a new word, and punctuation-only tokens attach to the word before them. Word
//! start times come from DTW token times when available (they track onsets well);
//! word ends are the token-level `t1` bounded by the next word's start and a
//! length-based cap, so a word before a long pause does not swallow the pause.

/// One non-special token with times relative to the audio passed to whisper.
#[derive(Debug, Clone, PartialEq)]
pub struct RawToken {
    /// Raw token bytes (may split UTF-8 sequences).
    pub bytes: Vec<u8>,
    /// Token probability.
    pub p: f32,
    /// Token start from timestamp heuristics, seconds.
    pub t0_s: f64,
    /// Token end from timestamp heuristics, seconds.
    pub t1_s: f64,
    /// DTW onset time, seconds, when DTW produced one.
    pub t_dtw_s: Option<f64>,
}

/// A word built from tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrWord {
    /// Word text including attached punctuation, without the leading space.
    pub w: String,
    /// Start time, seconds.
    pub start_s: f64,
    /// End time, seconds.
    pub end_s: f64,
    /// Mean probability of the word's non-punctuation tokens.
    pub p: f32,
}

struct Pending {
    bytes: Vec<u8>,
    start_s: f64,
    last_t1_s: f64,
    p_sum: f32,
    p_n: u32,
}

fn is_punct_only(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(bytes);
    let t = text.trim();
    !t.is_empty() && t.chars().all(|c| !c.is_alphanumeric())
}

/// Upper bound on a word's duration from its length.
pub fn max_word_duration_s(word: &str) -> f64 {
    let n = word.chars().filter(|c| c.is_alphanumeric()).count() as f64;
    (0.2 + 0.09 * n).clamp(0.25, 1.6)
}

/// Group tokens of one whisper segment into words.
///
/// `seg_start_s`/`seg_end_s` bound every word time.
pub fn tokens_to_words(tokens: &[RawToken], seg_start_s: f64, seg_end_s: f64) -> Vec<AsrWord> {
    let mut pending: Vec<Pending> = Vec::new();
    for tok in tokens {
        if tok.bytes.is_empty() {
            continue;
        }
        let starts_word = tok.bytes.first() == Some(&b' ');
        let punct = is_punct_only(&tok.bytes);
        let onset = tok.t_dtw_s.unwrap_or(tok.t0_s);
        let attach = match pending.last() {
            None => false,
            Some(_) if punct => true,
            Some(_) => !starts_word,
        };
        if attach {
            if let Some(cur) = pending.last_mut() {
                cur.bytes.extend_from_slice(&tok.bytes);
                cur.last_t1_s = cur.last_t1_s.max(tok.t1_s);
                if !punct {
                    cur.p_sum += tok.p;
                    cur.p_n += 1;
                }
            }
        } else if punct && pending.is_empty() {
            // leading punctuation (for example an opening quote) starts a word
            // but does not count toward its probability
            pending.push(Pending {
                bytes: tok.bytes.clone(),
                start_s: onset,
                last_t1_s: tok.t1_s,
                p_sum: 0.0,
                p_n: 0,
            });
        } else {
            pending.push(Pending {
                bytes: tok.bytes.clone(),
                start_s: onset,
                last_t1_s: tok.t1_s,
                p_sum: tok.p,
                p_n: 1,
            });
        }
    }

    let n = pending.len();
    let mut words = Vec::with_capacity(n);
    let mut prev_start = seg_start_s;
    for i in 0..n {
        let cur = &pending[i];
        let text = String::from_utf8_lossy(&cur.bytes).trim().to_string();
        if text.is_empty() {
            continue;
        }
        let start = cur.start_s.clamp(seg_start_s, seg_end_s).max(prev_start);
        let next_start = pending
            .get(i + 1)
            .map_or(seg_end_s, |nx| nx.start_s.clamp(seg_start_s, seg_end_s))
            .max(start);
        let cap = start + max_word_duration_s(&text);
        let hard_end = next_start.min(cap).max(start);
        let end = if cur.last_t1_s > start {
            cur.last_t1_s.min(hard_end)
        } else {
            hard_end
        };
        let end = end.max((start + 0.02).min(seg_end_s)).max(start);
        let p = if cur.p_n > 0 {
            cur.p_sum / cur.p_n as f32
        } else {
            1.0
        };
        prev_start = start;
        words.push(AsrWord {
            w: text,
            start_s: start,
            end_s: end,
            p,
        });
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &str, p: f32, t0: f64, t1: f64, dtw: Option<f64>) -> RawToken {
        RawToken {
            bytes: s.as_bytes().to_vec(),
            p,
            t0_s: t0,
            t1_s: t1,
            t_dtw_s: dtw,
        }
    }

    #[test]
    fn groups_subwords_and_punctuation() {
        let toks = vec![
            tok(" Hel", 0.9, 0.0, 0.2, Some(0.05)),
            tok("lo", 0.7, 0.2, 0.4, Some(0.2)),
            tok(",", 0.99, 0.4, 0.4, Some(0.4)),
            tok(" world", 0.5, 0.5, 0.9, Some(0.55)),
            tok(".", 0.99, 0.9, 0.9, Some(0.9)),
        ];
        let w = tokens_to_words(&toks, 0.0, 1.0);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].w, "Hello,");
        assert!((w[0].p - 0.8).abs() < 1e-6);
        assert!((w[0].start_s - 0.05).abs() < 1e-9);
        assert!(w[0].end_s <= w[1].start_s);
        assert_eq!(w[1].w, "world.");
        assert!((w[1].p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn end_is_capped_before_long_pause() {
        let toks = vec![
            tok(" Hi", 0.9, 0.0, 9.0, Some(0.1)),
            tok(" there", 0.9, 9.0, 9.4, Some(9.0)),
        ];
        let w = tokens_to_words(&toks, 0.0, 10.0);
        assert!(w[0].end_s < 1.0, "end {}", w[0].end_s);
        assert!((w[1].start_s - 9.0).abs() < 1e-9);
    }

    #[test]
    fn falls_back_to_t0_without_dtw_and_clamps() {
        let toks = vec![tok(" a", 0.9, -1.0, 0.3, None), tok(" b", 0.9, 0.3, 5.0, None)];
        let w = tokens_to_words(&toks, 0.0, 1.0);
        assert!((w[0].start_s - 0.0).abs() < 1e-9);
        assert!(w[1].end_s <= 1.0);
        assert!(w.iter().all(|x| x.end_s >= x.start_s));
    }

    #[test]
    fn monotonic_starts() {
        let toks = vec![
            tok(" one", 0.9, 0.0, 0.3, Some(0.5)),
            tok(" two", 0.9, 0.3, 0.6, Some(0.2)),
        ];
        let w = tokens_to_words(&toks, 0.0, 1.0);
        assert!(w[1].start_s >= w[0].start_s);
    }
}
