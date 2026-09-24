//! Integration tests on synthetic audio only.
//!
//! Extraction tests run whenever `ffmpeg` and `ffprobe` are on PATH. The
//! recognition chain is covered by `mock_chain.rs`, and the real models by the
//! ignored `reference_regression.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

use glassrip_audio::extract::{extract_audio, ExtractOptions, SAMPLE_RATE};

fn have_ffmpeg() -> bool {
    ["ffmpeg", "ffprobe"].iter().all(|b| {
        Command::new(b)
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// 0.5 s of silence, 1.0 s of a 440 Hz tone (ffmpeg's default 1/8 amplitude),
/// 0.5 s of silence,
/// as 44.1 kHz stereo WAV.
fn synth_wav(dir: &Path) -> PathBuf {
    let out = dir.join("tone.wav");
    let status = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo:d=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo:d=0.5",
            "-filter_complex",
            "[1:a]aformat=channel_layouts=stereo[t];[0:a][t][2:a]concat=n=3:v=0:a=1[a]",
            "-map",
            "[a]",
        ])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    out
}

#[tokio::test]
async fn extracts_mono_16k_with_timing() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg/ffprobe not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let wav = synth_wav(dir.path());
    let audio = extract_audio(&ExtractOptions::default(), &wav)
        .await
        .unwrap();
    assert_eq!(audio.sample_rate, SAMPLE_RATE);
    assert!(
        (audio.duration_s() - 2.0).abs() < 0.05,
        "duration {}",
        audio.duration_s()
    );
    assert!(audio.video_start_s.is_none());
    assert!(audio.timeline_offset_s.abs() < 1e-6);

    let sr = SAMPLE_RATE as usize;
    let rms = |a: &[f32]| (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt();
    let silence = rms(&audio.samples[..sr / 4]);
    let tone = rms(&audio.samples[sr * 3 / 4..sr * 5 / 4]);
    assert!(silence < 1e-3, "silence rms {silence}");
    // a sine of amplitude 1/8 has rms 0.0884; allow for resampling and downmix
    assert!((tone - 0.0884).abs() < 0.01, "tone rms {tone}");
}

#[tokio::test]
async fn missing_audio_stream_is_an_error() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg/ffprobe not found");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("still.png");
    let ok = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-frames:v",
            "1",
        ])
        .arg(&png)
        .status()
        .unwrap();
    assert!(ok.success());
    let err = extract_audio(&ExtractOptions::default(), &png)
        .await
        .unwrap_err();
    assert!(
        matches!(err, glassrip_audio::AudioError::NoAudioStream(_)),
        "{err}"
    );
}
