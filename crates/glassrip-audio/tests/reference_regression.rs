//! Regression test against a private reference recording.
//!
//! Ignored by default. Run with the real models:
//!
//! ```text
//! GLASSRIP_TEST_WHISPER_MODEL=~/.glassrip/models/ggml-large-v3-turbo.bin \
//! GLASSRIP_TEST_REFERENCE_AUDIO=/private/path/meeting.flac \
//! GLASSRIP_TEST_EXPECT_FILE=/private/path/expect.json \
//! cargo test --release --features cuda --test reference_regression -- --ignored --nocapture
//! ```
//!
//! The expect file (kept outside the repository) is JSON:
//!
//! ```json
//! {
//!   "vocabulary": ["Term A", "Term B"],
//!   "speakers": 3,
//!   "expected_labels": 3,
//!   "min_counts": {"Term A": 4},
//!   "max_counts": {"Wrong variant": 0},
//!   "max_runtime_s": 98.5,
//!   "diarize_mode": "cuda",
//!   "concurrent": true
//! }
//! ```
//!
//! The run is repeated and both transcripts must match exactly.
#![cfg(feature = "diarize")]

use std::collections::BTreeSet;
use std::path::PathBuf;

use glassrip_audio::asr::{AsrConfig, VadSettings};
use glassrip_audio::assign::AssignConfig;
use glassrip_audio::diarize::{DiarizeConfig, DiarizeMode};
use glassrip_audio::extract::ExtractOptions;
use glassrip_audio::gapfill::GapFillConfig;
use glassrip_audio::metrics::count_term;
use glassrip_audio::pipeline::{run, PipelineConfig, PipelineOutput};
use glassrip_audio::vocab::CorrectionConfig;
use serde_json::Value;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is not set")))
}

fn text(out: &PipelineOutput) -> String {
    out.transcript
        .items
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::test]
#[ignore = "needs real models and a private reference recording"]
async fn reference_recording_regression() {
    let model = env_path("GLASSRIP_TEST_WHISPER_MODEL");
    let audio = env_path("GLASSRIP_TEST_REFERENCE_AUDIO");
    let expect: Value = serde_json::from_str(
        &std::fs::read_to_string(env_path("GLASSRIP_TEST_EXPECT_FILE")).unwrap(),
    )
    .unwrap();
    let models_dir = model.parent().unwrap().to_path_buf();

    let mut asr = AsrConfig::new(model);
    asr.vad = Some(VadSettings::new(models_dir.join("ggml-silero-v6.2.0.bin")));
    asr.vocabulary = expect["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mode = match expect["diarize_mode"].as_str().unwrap_or("cuda") {
        "cpu" => DiarizeMode::Cpu,
        "cuda-fast" => DiarizeMode::CudaFast,
        _ => DiarizeMode::Cuda,
    };
    let cfg = PipelineConfig {
        extract: ExtractOptions::default(),
        asr,
        diarize: Some(DiarizeConfig {
            models_dir: models_dir.join("speakrs"),
            mode,
            num_speakers: expect["speakers"].as_u64().map(|k| k as usize),
            verify_models: true,
        }),
        correction: CorrectionConfig::default(),
        assign: AssignConfig::default(),
        gap_fill: Some(GapFillConfig::default()),
        concurrent: expect["concurrent"].as_bool().unwrap_or(true),
        run_id: Some("regression".into()),
    };

    let mut runs = Vec::new();
    for i in 0..2 {
        let out = run(&audio, &cfg).await.unwrap();
        let labels: BTreeSet<&str> = out
            .transcript
            .items
            .iter()
            .map(|s| s.speaker_label.as_str())
            .collect();
        let t = text(&out);
        println!(
            "run {i}: total {:.1}s asr {:.1}s diarize {:.1}s gapfill {:.1}s labels {} segments {} backend {} words {:?} gap {:?}",
            out.timings.total_s,
            out.timings.asr_s,
            out.timings.diarize_s,
            out.timings.gapfill_s,
            labels.len(),
            out.transcript.items.len(),
            out.stats.asr_backend,
            out.stats.words_by_source,
            out.stats.gap_fill,
        );
        assert_eq!(
            labels.len() as u64,
            expect["expected_labels"].as_u64().unwrap(),
            "label count"
        );
        for (term, n) in expect["min_counts"].as_object().unwrap() {
            let got = count_term(&t, term);
            println!("  count(min) {} = {got}", term.len());
            assert!(
                got as u64 >= n.as_u64().unwrap(),
                "min count failed for a term: {got} < {n}"
            );
        }
        for (term, n) in expect["max_counts"].as_object().unwrap() {
            let got = count_term(&t, term);
            assert!(
                got as u64 <= n.as_u64().unwrap(),
                "max count failed for a term: {got} > {n}"
            );
        }
        let limit = expect["max_runtime_s"].as_f64().unwrap();
        assert!(
            out.timings.total_s <= limit,
            "runtime {:.1}s > {limit}s",
            out.timings.total_s
        );
        assert!(
            out.stats.warnings.is_empty(),
            "warnings: {:?}",
            out.stats.warnings
        );
        runs.push(out);
    }
    let key = |o: &PipelineOutput| {
        o.transcript
            .items
            .iter()
            .map(|s| {
                (
                    s.text.clone(),
                    s.speaker_label.clone(),
                    s.start_s.to_bits(),
                    s.end_s.to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&runs[0]), key(&runs[1]), "runs differ");
}
