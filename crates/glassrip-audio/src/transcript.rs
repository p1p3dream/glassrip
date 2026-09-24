//! Assemble transcript segments from ASR words, corrections and speaker turns.

use crate::asr::AsrSegment;
use crate::assign::{assign_words, nearest_turn, AssignConfig};
use crate::diarize::{speaker_label, Diarization};
use crate::recluster::Source;
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
    source: Source,
    gap_sim: Option<f32>,
    gap_margin: Option<f32>,
}

fn join_words<'a>(words: impl Iterator<Item = &'a str>) -> String {
    words.collect::<Vec<_>>().join(" ")
}

/// Build transcript segments.
///
/// Words that no turn qualifies for keep `source: unassigned`, confidence 0,
/// and the label of the temporally nearest turn as a placeholder. ASR segments
/// are split wherever the label changes. Times are shifted by
/// `timeline_offset_s`.
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
                source: Source::Unassigned,
                gap_sim: None,
                gap_margin: None,
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
            w.source = a.source;
            w.gap_sim = a.gap_sim;
            w.gap_margin = a.gap_margin;
            if w.speaker.is_none() {
                let mid = (w.start_s + w.end_s) / 2.0;
                w.speaker = nearest_turn(&d.turns, mid).map(|t| t.speaker);
                w.conf = 0.0;
                w.source = Source::Unassigned;
            }
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
                assign_conf: w.conf.clamp(0.0, 1.0),
                source: w.source,
                gap_sim: w.gap_sim,
                gap_margin: w.gap_margin,
            })
            .collect();
        let total: f64 = run.iter().map(|w| (w.end_s - w.start_s).max(1e-3)).sum();
        let weighted: f64 = run
            .iter()
            .map(|w| (w.end_s - w.start_s).max(1e-3) * f64::from(w.conf.clamp(0.0, 1.0)))
            .sum();
        let start_s = run.first().map_or(0.0, |w| w.start_s) + timeline_offset_s;
        let end_s = run.iter().map(|w| w.end_s).fold(f64::MIN, f64::max) + timeline_offset_s;
        let speaker_conf = if total > 0.0 {
            ((weighted / total) as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(TranscriptSegment {
            segment_id: format!("seg_{:05}", out.len()),
            start_s,
            end_s: end_s.max(start_s),
            speaker_label: label,
            speaker_conf,
            text: join_words(words.iter().map(|w| w.w.as_str())),
            text_raw: join_words(run.iter().map(|w| w.raw.as_str())),
            gap_fill_words: run.iter().filter(|w| w.source == Source::GapFill).count(),
            unassigned_words: run
                .iter()
                .filter(|w| w.source == Source::Unassigned)
                .count(),
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

    fn diar(turns: Vec<Turn>) -> Diarization {
        Diarization {
            turns,
            labels: vec!["SPEAKER_00".into(), "SPEAKER_01".into()],
            talk_time_s: vec![2.0, 2.0],
            num_clusters_raw: 2,
            active_s: 4.0,
            centroids: vec![None, None],
        }
    }

    fn two_turns() -> Diarization {
        diar(vec![Turn::new(0.0, 2.0, 0), Turn::new(2.0, 4.0, 1)])
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
            Some(&two_turns()),
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
        assert!(segs
            .iter()
            .flat_map(|s| &s.words)
            .all(|w| w.source == Source::Diarizer));
    }

    #[test]
    fn unassigned_word_takes_the_nearest_turn_as_placeholder() {
        // word at 6.0 is nearer the SPEAKER_01 turn at 8.0 than the SPEAKER_00
        // turn ending at 2.0, even though SPEAKER_00 spoke earlier in the segment
        let d = diar(vec![Turn::new(0.0, 2.0, 0), Turn::new(8.0, 9.0, 1)]);
        let asr = vec![AsrSegment {
            start_s: 0.0,
            end_s: 9.0,
            words: vec![w("one", 1.0, 1.4, 0.9), w("two", 6.0, 6.4, 0.9)],
        }];
        let segs = build_segments(
            &asr,
            Some(&d),
            &Vocabulary::default(),
            &CorrectionConfig::default(),
            &AssignConfig::default(),
            0.0,
        );
        assert_eq!(segs.len(), 2);
        let two = &segs[1].words[0];
        assert_eq!(two.speaker_label, "SPEAKER_01");
        assert_eq!(two.source, Source::Unassigned);
        assert_eq!(two.assign_conf, 0.0);
        assert_eq!(segs[1].unassigned_words, 1);
    }

    #[test]
    fn segment_confidence_stays_in_unit_interval_with_zero_length_words() {
        let asr = vec![AsrSegment {
            start_s: 0.0,
            end_s: 4.0,
            words: vec![
                w("a", 1.0, 1.0, 0.9),
                w("b", 1.0, 1.0, 0.9),
                w("c", 1.5, 1.5, 0.9),
            ],
        }];
        let segs = build_segments(
            &asr,
            Some(&two_turns()),
            &Vocabulary::default(),
            &CorrectionConfig::default(),
            &AssignConfig::default(),
            0.0,
        );
        for s in &segs {
            assert!((0.0..=1.0).contains(&s.speaker_conf), "{}", s.speaker_conf);
            for x in &s.words {
                assert!((0.0..=1.0).contains(&x.assign_conf));
            }
        }
    }

    #[test]
    fn gap_fill_words_are_flagged() {
        let mut g = Turn::new(2.0, 4.0, 1);
        g.source = Source::GapFill;
        g.embedding_sim = Some(0.8);
        g.gap_margin = Some(0.2);
        let d = diar(vec![Turn::new(0.0, 2.0, 0), g]);
        let asr = vec![AsrSegment {
            start_s: 0.0,
            end_s: 4.0,
            words: vec![w("x", 0.5, 0.9, 0.9), w("y", 2.5, 3.0, 0.9)],
        }];
        let segs = build_segments(
            &asr,
            Some(&d),
            &Vocabulary::default(),
            &CorrectionConfig::default(),
            &AssignConfig::default(),
            0.0,
        );
        let y = &segs[1].words[0];
        assert_eq!(y.source, Source::GapFill);
        assert_eq!(y.gap_sim, Some(0.8));
        assert_eq!(y.gap_margin, Some(0.2));
        assert!(y.assign_conf <= 0.6);
        assert_eq!(segs[1].gap_fill_words, 1);
    }

    #[test]
    fn without_diarization_everything_is_one_placeholder_label() {
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
        assert_eq!(segs[0].words[0].source, Source::Unassigned);
    }
}
