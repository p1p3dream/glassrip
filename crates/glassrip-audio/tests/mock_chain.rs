//! Full-chain test with a mock recognizer and mock embeddings on generated tones.
//!
//! Layout (10 s at 16 kHz): tones at 1.0-3.0 s and 3.2-4.0 s (amplitude 0.1,
//! "speaker A") and 8.0-9.0 s (amplitude 0.3, "speaker B"); silence elsewhere.
//! The diarizer (synthetic turns) covers only the first region, so the second
//! region must be labeled by gap filling.

use glassrip_audio::asr::{
    transcribe_with, ChunkDecoder, ChunkingConfig, DecodedSegment, EnergyVad, SpeechDetector,
};
use glassrip_audio::assign::AssignConfig;
use glassrip_audio::diarize::Diarization;
use glassrip_audio::gapfill::{GapFillConfig, SpanEmbedder, SpanStatus};
use glassrip_audio::pipeline::{assemble, AssembleConfig};
use glassrip_audio::recluster::{Source, Turn};
use glassrip_audio::vocab::CorrectionConfig;
use glassrip_audio::words::RawToken;
use glassrip_audio::Result;

const SR: usize = 16_000;

fn tone(samples: &mut [f32], start_s: f64, end_s: f64, amp: f32) {
    let (a, b) = ((start_s * SR as f64) as usize, (end_s * SR as f64) as usize);
    for (i, x) in samples[a..b].iter_mut().enumerate() {
        *x = amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SR as f32).sin();
    }
}

fn audio() -> Vec<f32> {
    let mut s = vec![0.0f32; SR * 10];
    tone(&mut s, 1.0, 3.0, 0.1);
    tone(&mut s, 3.2, 4.0, 0.1);
    tone(&mut s, 8.0, 9.0, 0.3);
    s
}

fn tok(text: &str, p: f32, t0: f64, t1: f64, dtw: f64) -> RawToken {
    RawToken {
        bytes: text.as_bytes().to_vec(),
        p,
        t0_s: t0,
        t1_s: t1,
        t_dtw_s: Some(dtw),
    }
}

/// Emits scripted chunk-relative segments and records what it was asked.
#[derive(Default)]
struct MockDecoder {
    calls: Vec<(usize, String)>,
}

impl ChunkDecoder for MockDecoder {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }

    fn decode(&mut self, audio: &[f32], prompt: &str) -> Result<Vec<DecodedSegment>> {
        self.calls.push((audio.len(), prompt.to_string()));
        Ok(match self.calls.len() {
            1 => vec![DecodedSegment {
                start_s: 0.1,
                end_s: 3.1,
                tokens: vec![
                    tok(" Hello", 0.95, 0.1, 0.5, 0.15),
                    tok(" Ketra", 0.30, 0.6, 1.0, 0.65),
                    tok(",", 0.99, 1.0, 1.0, 1.0),
                    tok(" how", 0.9, 1.2, 1.5, 1.25),
                    tok(" are", 0.9, 1.5, 1.8, 1.55),
                    tok(" you", 0.9, 1.8, 2.2, 1.85),
                    tok("?", 0.99, 2.2, 2.2, 2.2),
                ],
            }],
            _ => vec![DecodedSegment {
                start_s: 0.1,
                end_s: 1.1,
                tokens: vec![
                    tok(" Bye", 0.9, 0.1, 0.5, 0.15),
                    tok(" now", 0.9, 0.5, 0.9, 0.55),
                    tok(".", 0.99, 0.9, 0.9, 0.9),
                ],
            }],
        })
    }
}

/// Loud audio sounds like speaker B, quiet audio like speaker A.
struct RmsEmbedder {
    ambiguous: bool,
}

impl SpanEmbedder for RmsEmbedder {
    fn embed(&mut self, audio: &[f32]) -> Result<Option<Vec<f32>>> {
        if self.ambiguous {
            return Ok(Some(vec![0.7, 0.7]));
        }
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        Ok(Some(if rms > 0.12 {
            vec![0.0, 1.0]
        } else {
            vec![1.0, 0.0]
        }))
    }
}

fn diarization() -> Diarization {
    Diarization {
        turns: vec![Turn::new(1.0, 4.0, 0), Turn::new(5.0, 6.0, 1)],
        labels: vec!["SPEAKER_00".into(), "SPEAKER_01".into()],
        talk_time_s: vec![3.0, 1.0],
        num_clusters_raw: 2,
        active_s: 4.0,
        centroids: vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])],
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn run(ambiguous: bool) -> (glassrip_audio::pipeline::Assembled, MockDecoder) {
    let samples = audio();
    let mut vad = EnergyVad::default();
    let regions = vad.detect(&samples).unwrap();
    assert_eq!(regions.len(), 2);
    assert!(
        close(regions[0].0, 0.9) && close(regions[0].1, 4.1),
        "{regions:?}"
    );
    assert!(
        close(regions[1].0, 7.9) && close(regions[1].1, 9.1),
        "{regions:?}"
    );

    let vocabulary = vec!["Kethra".to_string(), "Zorbin".to_string()];
    let chunking = ChunkingConfig {
        vocabulary: vocabulary.clone(),
        max_prompt_tokens: 200,
        carry_previous_words: 0,
        max_chunk_s: 28.0,
        max_merge_gap_s: 2.0,
    };
    let mut dec = MockDecoder::default();
    let asr = transcribe_with(&mut dec, Some(&mut vad), &samples, &chunking).unwrap();
    let mut emb = RmsEmbedder { ambiguous };
    let cfg = AssembleConfig {
        vocabulary,
        correction: CorrectionConfig::default(),
        assign: AssignConfig::default(),
        gap_fill: Some(GapFillConfig::default()),
        timeline_offset_s: 0.0,
    };
    let out = assemble(&asr, Some(diarization()), Some(&mut emb), &samples, &cfg).unwrap();
    assert_eq!(asr.chunk_spans.len(), 2);
    assert_eq!(asr.prompt_terms, vec!["Kethra", "Zorbin"]);
    (out, dec)
}

#[test]
fn chain_produces_exact_segments_with_gap_fill() {
    let (out, dec) = run(false);

    // chunk audio lengths (0.9-4.1 s and 7.9-9.1 s) and the prompt
    assert_eq!(
        dec.calls,
        vec![
            (51_200, "Kethra, Zorbin.".to_string()),
            (19_200, "Kethra, Zorbin.".to_string())
        ]
    );

    let segs = &out.segments;
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].text, "Hello Kethra, how are you?");
    assert_eq!(segs[0].text_raw, "Hello Ketra, how are you?");
    assert_eq!(segs[0].speaker_label, "SPEAKER_00");
    assert!(close(segs[0].start_s, 1.05) && close(segs[0].end_s, 3.1));
    let starts: Vec<f64> = segs[0].words.iter().map(|w| w.start_s).collect();
    let ends: Vec<f64> = segs[0].words.iter().map(|w| w.end_s).collect();
    for (got, want) in starts.iter().zip([1.05, 1.55, 2.15, 2.45, 2.75]) {
        assert!(close(*got, want), "starts {starts:?}");
    }
    for (got, want) in ends.iter().zip([1.4, 1.9, 2.4, 2.7, 3.1]) {
        assert!(close(*got, want), "ends {ends:?}");
    }
    assert_eq!(segs[0].words[1].w_raw.as_deref(), Some("Ketra,"));
    assert!(segs[0].words.iter().all(|w| w.source == Source::Diarizer));
    assert!(segs[0].words.iter().all(|w| w.assign_conf > 0.6));

    assert_eq!(segs[1].text, "Bye now.");
    assert_eq!(segs[1].speaker_label, "SPEAKER_01");
    assert!(close(segs[1].start_s, 8.05) && close(segs[1].end_s, 8.8));
    assert_eq!(segs[1].gap_fill_words, 2);
    for w in &segs[1].words {
        assert_eq!(w.source, Source::GapFill);
        assert!(w.assign_conf > 0.0 && w.assign_conf <= 0.6);
        assert_eq!(w.gap_sim, Some(1.0));
        assert_eq!(w.gap_margin, Some(1.0));
    }

    assert_eq!(out.gap_stats.spans, 1);
    assert_eq!(out.gap_stats.labeled, 1);
    assert_eq!(out.gap_spans[0].status, SpanStatus::Assigned);
    assert!(close(out.gap_spans[0].start_s, 8.05) && close(out.gap_spans[0].end_s, 8.8));
    let d = out.diarization.unwrap();
    assert_eq!(d.turns.len(), 3);
    assert_eq!(d.turns[2].source, Source::GapFill);
}

#[test]
fn chain_leaves_ambiguous_spans_unassigned() {
    let (out, _) = run(true);
    assert_eq!(out.gap_spans[0].status, SpanStatus::LowMargin);
    assert_eq!(out.gap_stats.rejected_margin, 1);
    let seg = &out.segments[1];
    assert_eq!(seg.text, "Bye now.");
    // placeholder label from the nearest turn (5.0-6.0, SPEAKER_01), flagged
    assert_eq!(seg.speaker_label, "SPEAKER_01");
    assert_eq!(seg.unassigned_words, 2);
    assert!(seg
        .words
        .iter()
        .all(|w| w.source == Source::Unassigned && w.assign_conf == 0.0));
    assert_eq!(out.diarization.unwrap().turns.len(), 2);
}
