//! Audio extraction with ffmpeg and stream start times from ffprobe.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Deserialize;
use tokio::process::Command;

use crate::error::{AudioError, Result};

/// Sample rate used by every model in this crate.
pub const SAMPLE_RATE: u32 = 16_000;

/// ffmpeg and ffprobe locations.
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    /// ffmpeg binary.
    pub ffmpeg: PathBuf,
    /// ffprobe binary.
    pub ffprobe: PathBuf,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            ffmpeg: PathBuf::from("ffmpeg"),
            ffprobe: PathBuf::from("ffprobe"),
        }
    }
}

/// Decoded audio plus timeline information.
#[derive(Debug, Clone)]
pub struct ExtractedAudio {
    /// 16 kHz mono samples.
    pub samples: Vec<f32>,
    /// Sample rate of `samples`.
    pub sample_rate: u32,
    /// `start_time` of the audio stream, seconds.
    pub audio_start_s: f64,
    /// `start_time` of the first video stream, seconds, if any.
    pub video_start_s: Option<f64>,
    /// Seconds to add to sample-derived times to land on the video timeline.
    pub timeline_offset_s: f64,
}

impl ExtractedAudio {
    /// Duration of the decoded audio, seconds.
    pub fn duration_s(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// Stream start times reported by ffprobe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamStarts {
    /// Audio stream start time, if there is an audio stream.
    pub audio_start_s: Option<f64>,
    /// First real video stream start time (cover art excluded).
    pub video_start_s: Option<f64>,
}

#[derive(Deserialize)]
struct ProbeOut {
    #[serde(default)]
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    start_time: Option<String>,
    #[serde(default)]
    disposition: Option<ProbeDisposition>,
}

#[derive(Deserialize)]
struct ProbeDisposition {
    #[serde(default)]
    attached_pic: i64,
}

/// Parse ffprobe JSON into stream start times.
pub fn parse_stream_starts(json: &str) -> Result<StreamStarts> {
    let out: ProbeOut = serde_json::from_str(json).map_err(|e| AudioError::Parse {
        what: "ffprobe json".into(),
        message: e.to_string(),
    })?;
    let parse_t = |s: &ProbeStream| -> f64 {
        s.start_time
            .as_deref()
            .and_then(|t| t.parse::<f64>().ok())
            .filter(|t| t.is_finite())
            .unwrap_or(0.0)
    };
    let audio_start_s = out
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("audio"))
        .map(parse_t);
    let video_start_s = out
        .streams
        .iter()
        .find(|s| {
            s.codec_type.as_deref() == Some("video")
                && s.disposition.as_ref().map_or(0, |d| d.attached_pic) == 0
        })
        .map(parse_t);
    Ok(StreamStarts {
        audio_start_s,
        video_start_s,
    })
}

/// Offset that moves audio sample time onto the video timeline.
///
/// Sample 0 of the decoded audio sits at the audio stream's `start_time`. When a
/// video stream exists, times are expressed relative to the video's first PTS;
/// otherwise relative to the container zero.
pub fn timeline_offset(starts: StreamStarts) -> f64 {
    let audio = starts.audio_start_s.unwrap_or(0.0);
    audio - starts.video_start_s.unwrap_or(0.0)
}

fn stderr_tail(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    let n = trimmed.chars().count();
    trimmed.chars().skip(n.saturating_sub(2000)).collect()
}

/// Run ffprobe and return stream start times.
pub async fn probe_stream_starts(opts: &ExtractOptions, input: &Path) -> Result<StreamStarts> {
    let out = Command::new(&opts.ffprobe)
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time:stream_disposition=attached_pic",
            "-of",
            "json",
        ])
        .arg(input)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| AudioError::io(&opts.ffprobe, e))?;
    if !out.status.success() {
        return Err(AudioError::Command {
            command: "ffprobe".into(),
            status: out.status.to_string(),
            stderr_tail: stderr_tail(&out.stderr),
        });
    }
    parse_stream_starts(&String::from_utf8_lossy(&out.stdout))
}

/// Convert little-endian `f32` bytes to samples. Trailing partial samples are dropped.
pub fn f32le_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode the first audio stream of `input` to 16 kHz mono `f32`.
pub async fn extract_audio(opts: &ExtractOptions, input: &Path) -> Result<ExtractedAudio> {
    let starts = probe_stream_starts(opts, input).await?;
    let Some(audio_start_s) = starts.audio_start_s else {
        return Err(AudioError::NoAudioStream(input.to_path_buf()));
    };
    let out = Command::new(&opts.ffmpeg)
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
        ])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| AudioError::io(&opts.ffmpeg, e))?;
    if !out.status.success() {
        return Err(AudioError::Command {
            command: "ffmpeg".into(),
            status: out.status.to_string(),
            stderr_tail: stderr_tail(&out.stderr),
        });
    }
    if out.stdout.len() % 4 != 0 {
        return Err(AudioError::Parse {
            what: "ffmpeg f32le output".into(),
            message: format!("{} bytes is not a whole number of samples", out.stdout.len()),
        });
    }
    let samples = f32le_to_samples(&out.stdout);
    Ok(ExtractedAudio {
        samples,
        sample_rate: SAMPLE_RATE,
        audio_start_s,
        video_start_s: starts.video_start_s,
        timeline_offset_s: timeline_offset(starts),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_starts_and_skips_cover_art() {
        let json = r#"{"streams":[
            {"codec_type":"video","start_time":"0.000000","disposition":{"attached_pic":1}},
            {"codec_type":"audio","start_time":"0.021000","disposition":{"attached_pic":0}}
        ]}"#;
        let s = parse_stream_starts(json).unwrap();
        assert_eq!(s.audio_start_s, Some(0.021));
        assert_eq!(s.video_start_s, None);
        assert!((timeline_offset(s) - 0.021).abs() < 1e-12);
    }

    #[test]
    fn offset_is_audio_minus_video() {
        let s = StreamStarts {
            audio_start_s: Some(1.25),
            video_start_s: Some(0.5),
        };
        assert!((timeline_offset(s) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn missing_start_time_is_zero() {
        let s = parse_stream_starts(r#"{"streams":[{"codec_type":"audio"}]}"#).unwrap();
        assert_eq!(s.audio_start_s, Some(0.0));
    }

    #[test]
    fn decodes_f32le() {
        let mut b = Vec::new();
        b.extend_from_slice(&0.5f32.to_le_bytes());
        b.extend_from_slice(&(-1.0f32).to_le_bytes());
        b.push(0);
        assert_eq!(f32le_to_samples(&b), vec![0.5, -1.0]);
    }
}
