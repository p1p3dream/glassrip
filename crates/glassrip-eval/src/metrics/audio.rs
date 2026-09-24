//! Audio metrics: hotword WER and distinct speaker-label count (spec 9.3).
//!
//! Hotword WER is computed over hotword occurrences only. The truth lists, per
//! hotword, time windows (usually transcript segments) with the number of times
//! the word is spoken in each. A predicted occurrence is the hotword's token
//! sequence (case-insensitive, edge punctuation and a trailing possessive `'s`
//! ignored) starting at a predicted word inside a window (with tolerance).
//! Per window, `hits = min(count, occurrences)`; `misses = count - hits` (a
//! misrecognition is a miss). For hotwords marked `exhaustive`, occurrences
//! outside every window are insertions.
//! `WER = (misses + insertions) / total count`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::text::ErrorCount;

/// One time window with a known number of hotword occurrences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HotwordWindow {
    /// Start, seconds.
    pub t_start_s: f64,
    /// End, seconds.
    pub t_end_s: f64,
    /// Times the hotword is spoken in the window.
    pub count: usize,
}

/// Truth for one hotword.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hotword {
    /// Correct spelling (may be several words).
    pub word: String,
    /// Wrong forms seen in earlier transcripts (informational).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub known_misrecognitions: Vec<String>,
    /// Whether the windows list every occurrence (enables insertion counting).
    #[serde(default)]
    pub exhaustive: bool,
    /// Windows.
    pub windows: Vec<HotwordWindow>,
}

/// A predicted word with its start time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedWord {
    /// Word text.
    pub w: String,
    /// Start time, seconds.
    pub t_s: f64,
}

/// Per-hotword counts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HotwordCounts {
    /// Truth occurrences.
    pub reference: usize,
    /// Recognized occurrences inside windows.
    pub hits: usize,
    /// Truth occurrences not recognized.
    pub misses: usize,
    /// Recognized occurrences outside every window (exhaustive hotwords only).
    pub insertions: usize,
}

/// Hotword WER result.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HotwordScore {
    /// Per hotword.
    pub per_word: BTreeMap<String, HotwordCounts>,
    /// Pooled errors (misses + insertions) over pooled reference count.
    pub total: ErrorCount,
}

fn norm_token(w: &str) -> String {
    let t = w
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    t.strip_suffix("'s")
        .or_else(|| t.strip_suffix("\u{2019}s"))
        .map(str::to_string)
        .unwrap_or(t)
}

/// Start times of every occurrence of `phrase` in `words`.
fn occurrences(phrase: &str, words: &[TimedWord]) -> Vec<f64> {
    let tokens: Vec<String> = phrase.split_whitespace().map(norm_token).collect();
    if tokens.is_empty() {
        return Vec::new();
    }
    let norm: Vec<String> = words.iter().map(|w| norm_token(&w.w)).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + tokens.len() <= norm.len() {
        if norm[i..i + tokens.len()] == tokens[..] {
            out.push(words[i].t_s);
            i += tokens.len();
        } else {
            i += 1;
        }
    }
    out
}

/// Hotword WER over predicted timed words; `tolerance_s` widens every window.
pub fn hotword_wer(hotwords: &[Hotword], words: &[TimedWord], tolerance_s: f64) -> HotwordScore {
    let mut score = HotwordScore::default();
    for h in hotwords {
        let occ = occurrences(&h.word, words);
        let mut used: BTreeSet<usize> = BTreeSet::new();
        let mut c = HotwordCounts::default();
        for w in &h.windows {
            c.reference += w.count;
            let mut hits = 0;
            for (i, &t) in occ.iter().enumerate() {
                if hits == w.count {
                    break;
                }
                if !used.contains(&i)
                    && t >= w.t_start_s - tolerance_s
                    && t <= w.t_end_s + tolerance_s
                {
                    used.insert(i);
                    hits += 1;
                }
            }
            c.hits += hits;
            c.misses += w.count - hits;
        }
        if h.exhaustive {
            c.insertions = occ.len() - used.len();
        }
        score.total.add(ErrorCount {
            errors: c.misses + c.insertions,
            reference_len: c.reference,
        });
        score.per_word.insert(h.word.clone(), c);
    }
    score
}

/// Number of distinct non-empty speaker labels.
pub fn speaker_label_count<'a>(labels: impl IntoIterator<Item = &'a str>) -> usize {
    labels
        .into_iter()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<BTreeSet<_>>()
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tw(w: &str, t: f64) -> TimedWord {
        TimedWord {
            w: w.into(),
            t_s: t,
        }
    }

    #[test]
    fn hotword_wer_hand_computed() {
        let hotwords = vec![
            Hotword {
                word: "Quorra".into(),
                known_misrecognitions: vec!["Cora".into()],
                exhaustive: true,
                windows: vec![
                    HotwordWindow {
                        t_start_s: 10.0,
                        t_end_s: 20.0,
                        count: 2,
                    },
                    HotwordWindow {
                        t_start_s: 50.0,
                        t_end_s: 60.0,
                        count: 1,
                    },
                ],
            },
            Hotword {
                word: "build farm".into(),
                known_misrecognitions: vec![],
                exhaustive: false,
                windows: vec![HotwordWindow {
                    t_start_s: 0.0,
                    t_end_s: 100.0,
                    count: 1,
                }],
            },
        ];
        let words = vec![
            tw("Quorra,", 11.0),  // hit 1 (punctuation stripped)
            tw("Cora", 15.0),     // misrecognition -> miss
            tw("Quorra's", 55.0), // hit (possessive)
            tw("quorra", 80.0),   // outside windows -> insertion (exhaustive)
            tw("the", 30.0),
            tw("Build", 31.0), // two-token hotword
            tw("farm.", 31.4),
        ];
        let s = hotword_wer(&hotwords, &words, 0.0);
        let q = &s.per_word["Quorra"];
        assert_eq!((q.reference, q.hits, q.misses, q.insertions), (3, 2, 1, 1));
        let bf = &s.per_word["build farm"];
        assert_eq!((bf.reference, bf.hits, bf.misses), (1, 1, 0));
        // (1 miss + 1 insertion) / 4 references
        assert_eq!(
            s.total,
            ErrorCount {
                errors: 2,
                reference_len: 4
            }
        );
        assert!((s.total.rate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tolerance_widens_windows() {
        let h = vec![Hotword {
            word: "Quorra".into(),
            known_misrecognitions: vec![],
            exhaustive: false,
            windows: vec![HotwordWindow {
                t_start_s: 10.0,
                t_end_s: 20.0,
                count: 1,
            }],
        }];
        let words = vec![tw("Quorra", 21.5)];
        assert_eq!(hotword_wer(&h, &words, 0.0).total.errors, 1);
        assert_eq!(hotword_wer(&h, &words, 2.0).total.errors, 0);
    }

    #[test]
    fn speaker_count() {
        assert_eq!(speaker_label_count(["S0", "S1", "S0", " ", "S2"]), 3);
        assert_eq!(speaker_label_count(Vec::<&str>::new()), 0);
    }
}
