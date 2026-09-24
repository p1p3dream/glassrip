//! Assemble transcript segments from ASR words, corrections and speaker turns.

use crate::asr::AsrSegment;
use crate::assign::{assign_words, AssignConfig};
use crate::diarize::{speaker_label, Diarization};
use crate::types::{TranscriptSegment, TranscriptWord};
use crate::vocab::{correct, CorrectionConfig, Vocabulary};

struct FlatWord {
    seg: usize,
    raw: String,
    fixed: Option<String>,
    start_s: f64,
    end_s: f64,
    p: f32,
    speaker: Option<usize>,
    conf: f32,
}

fn nearest_turn_speaker(d: &Diarization, t: f64) -> Option<usize> {
    d.turns
        .iter()
        .map(|x| {
            let dist = if t < x.start_s {
                x.start_s - t
            } else if t > x.end_s {
                t - x.end_s
            } else {
                0.0
            };
            (dist, x.speaker)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, s)| s)
}

fn join_words<'a>(words: impl Iterator<Item = &'a str>) -> String {
    words.collect::<Vec<_>>().join(" ")
}

/// Build transcript segments.
///
/// ASR segments are split wherever the assigned speaker changes. Times are
/// shifted by `timeline_offset_s`.
pub fn build_segments(
    asr: &[AsrSegment],
    diar: Option<&Diarization>,
    vocab: &Vocabulary,
    correction: &CorrectionConfig,
    assign_cfg: &AssignConfig,
    timeline_offset_s: f64,
) -> Vec<TranscriptSegment> {
    let mut flat: Vec<FlatWord> = asr
        .iter()
        .enumerate()
        .flat_map(|(si, s)| {
            s.words.iter().map(move |w| FlatWord {
                seg: si,
                raw: w.w.clone(),
                fixed: None,
                start_s: w.start_s,
                end_s: w.end_s,
                p: w.p,
                speaker: None,
                conf: 0.0,
            })
        })
        .collect();

    let pairs: Vec<(&str, f32)> = flat.iter().map(|w| (w.raw.as_str(), w.p)).collect();
    let fixes = correct(&pairs, vocab, correction);
    for (w, f) in flat.iter_mut().zip(fixes) {
        w.fixed = f;
    }

    if let Some(d) = diar {
        let times: Vec<(f64, f64)> = flat.iter().map(|w| (w.start_s, w.end_s)).collect();
        let assigned = assign_words(&times, &d.turns, assign_cfg);
        for (w, a) in flat.iter_mut().zip(assigned) {
            w.speaker = a.speaker;
            w.conf = a.conf;
        }
        // unassigned words inherit from neighbours in the same ASR segment
        for i in 0..flat.len() {
            if flat[i].speaker.is_some() {
                continue;
            }
            let seg = flat[i].seg;
            let prev = flat[..i]
                .iter()
                .rev()
                .take_while(|w| w.seg == seg)
                .find_map(|w| w.speaker);
            let next = flat[i + 1..]
                .iter()
                .take_while(|w| w.seg == seg)
                .find_map(|w| w.speaker);
            let mid = (flat[i].start_s + flat[i].end_s) / 2.0;
            flat[i].speaker = prev.or(next).or_else(|| nearest_turn_speaker(d, mid));
            flat[i].conf = 0.0;
        }
    }

    let mut out: Vec<TranscriptSegment> = Vec::new();
    let mut i = 0;
    while i < flat.len() {
        let seg = flat[i].seg;
        let spk = flat[i].speaker;
        let mut j = i + 1;
        while j < flat.len() && flat[j].seg == seg && flat[j].speaker == spk {
            j += 1;
        }
        let run = &flat[i..j];
        let label = speaker_label(spk.unwrap_or(0));
        let words: Vec<TranscriptWord> = run
            .iter()
            .map(|w| TranscriptWord {
                w: w.fixed.clone().unwrap_or_else(|| w.raw.clone()),
                w_raw: w.fixed.as_ref().map(|_| w.raw.clone()),
                start_s: w.start_s + timeline_offset_s,
                end_s: w.end_s + timeline_offset_s,
                p: w.p,
                speaker_label: label.clone(),
                assign_conf: w.conf,
            })
            .collect();
        let total: f64 = run.iter().map(|w| (w.end_s - w.start_s).max(1e-3)).sum();
        let weighted: f64 = run
            .iter()
            .map(|w| (w.end_s - w.start_s).max(1e-3) * f64::from(w.conf))
            .sum();
        let start_s = run.first().map_or(0.0, |w| w.start_s) + timeline_offset_s;
        let end_s = run.iter().map(|w| w.end_s).fold(f64::MIN, f64::max) + timeline_offset_s;
        out.push(TranscriptSegment {
            segment_id: format!("seg_{:05}", out.len()),
            start_s,
            end_s: end_s.max(start_s),
            speaker_label: label,
            speaker_conf: if total > 0.0 { (weighted / total) as f32 } else { 0.0 },
            text: join_words(words.iter().map(|w| w.w.as_str())),
            text_raw: join_words(run.iter().map(|w| w.raw.as_str())),
            words,
        });
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recluster::Turn;
    use crate::words::AsrWord;

    fn w(t: &str, s: f64, e: f64, p: f32) -> AsrWord {
        AsrWord {
            w: t.into(),
            start_s: s,
            end_s: e,
            p,
        }
    }

    fn diar() -> Diarization {
        Diarization {
            turns: vec![
                Turn { start_s: 0.0, end_s: 2.0, speaker: 0, embedding_sim: None },
                Turn { start_s: 2.0, end_s: 4.0, speaker: 1, embedding_sim: None },
            ],
            labels: vec!["SPEAKER_00".into(), "SPEAKER_01".into()],
            talk_time_s: vec![2.0, 2.0],
            num_clusters_raw: 2,
            active_s: 4.0,
            centroids: vec![None, None],
        }
    }

    #[test]
    fn splits_at_speaker_change_and_offsets_times() {
        let asr = vec![AsrSegment {
            start_s: 0.0,
            end_s: 4.0,
            words: vec![
                w("Hello", 0.2, 0.6, 0.9),
                w("there.", 0.7, 1.2, 0.9),
                w("Hi", 2.3, 2.6, 0.9),
                w("Ketra.", 2.7, 3.2, 0.2),
            ],
        }];
        let segs = build_segments(
            &asr,
            Some(&diar()),
            &Vocabulary::new(&["Kethra"]),
            &CorrectionConfig::default(),
            &AssignConfig::default(),
            10.0,
        );
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker_label, "SPEAKER_00");
        assert_eq!(segs[0].text, "Hello there.");
        assert!((segs[0].start_s - 10.2).abs() < 1e-9);
        assert_eq!(segs[1].speaker_label, "SPEAKER_01");
        assert_eq!(segs[1].text, "Hi Kethra.");
        assert_eq!(segs[1].text_raw, "Hi Ketra.");
        assert_eq!(segs[1].words[1].w_raw.as_deref(), Some("Ketra."));
        assert_eq!(segs[1].segment_id, "seg_00001");
    }

    #[test]
    fn unassigned_word_inherits_from_neighbour() {
        let asr = vec![AsrSegment {
            start_s: 0.0,
            end_s: 9.0,
            words: vec![w("one", 1.0, 1.4, 0.9), w("two", 8.0, 8.4, 0.9)],
        }];
        let segs = build_segments(
            &asr,
            Some(&diar()),
            &Vocabulary::default(),
            &CorrectionConfig::default(),
            &AssignConfig::default(),
            0.0,
        );
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].words[1].assign_conf, 0.0);
    }

    #[test]
    fn without_diarization_everything_is_one_label() {
        let asr = vec![AsrSegment {
            start_s: 0.0,
            end_s: 1.0,
            words: vec![w("a", 0.0, 0.5, 0.9)],
        }];
        let segs = build_segments(
            &asr,
            None,
            &Vocabulary::default(),
            &CorrectionConfig::default(),
            &AssignConfig::default(),
            0.0,
        );
        assert_eq!(segs[0].speaker_label, "SPEAKER_00");
        assert_eq!(segs[0].speaker_conf, 0.0);
    }
}
