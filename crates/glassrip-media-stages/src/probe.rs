//! `probe` (spec 6.1): `ffprobe` JSON into typed structs.

use std::collections::BTreeMap;
use std::path::PathBuf;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::schema::{AudioStream, MEDIA_PROBE, MediaProbe, VideoStream, v1};
use crate::util::{on_rayon, parse_rational, rational_f64, run_command};

/// Probe parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ProbeParams {
    /// Video path. Part of the key so downstream stages never read a stale path.
    pub video_path: String,
}

/// The `probe` stage.
#[derive(Debug, Clone)]
pub struct ProbeStage {
    params: ProbeParams,
    video: PathBuf,
    ffprobe: String,
    ffprobe_version: String,
}

impl ProbeStage {
    /// Stage for `video`, using the `ffprobe` binary (version string goes into the key).
    pub fn new(video: PathBuf, ffprobe: String, ffprobe_version: String) -> Self {
        Self {
            params: ProbeParams {
                video_path: video.display().to_string(),
            },
            video,
            ffprobe,
            ffprobe_version,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawOutput {
    #[serde(default)]
    streams: Vec<RawStream>,
    format: Option<RawFormat>,
}

#[derive(Debug, Deserialize)]
struct RawFormat {
    format_name: Option<String>,
    start_time: Option<String>,
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawStream {
    index: u32,
    codec_type: Option<String>,
    codec_name: Option<String>,
    profile: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    pix_fmt: Option<String>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    time_base: Option<String>,
    start_pts: Option<i64>,
    start_time: Option<String>,
    duration: Option<String>,
    nb_frames: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u32>,
    #[serde(default)]
    side_data_list: Vec<serde_json::Value>,
    #[serde(default)]
    tags: BTreeMap<String, serde_json::Value>,
}

fn num(s: &Option<String>) -> Option<f64> {
    s.as_deref()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

fn invalid(msg: impl Into<String>) -> ErrorInfo {
    ErrorInfo::new(ErrorCode::InvalidInput, msg)
}

fn rotation(s: &RawStream) -> Option<f64> {
    for sd in &s.side_data_list {
        if let Some(r) = sd.get("rotation").and_then(serde_json::Value::as_f64) {
            return Some(r);
        }
    }
    s.tags.get("rotate").and_then(|v| {
        v.as_str()
            .and_then(|t| t.parse::<f64>().ok())
            .or_else(|| v.as_f64())
    })
}

/// Parses `ffprobe -print_format json -show_streams -show_format` output. Everything except
/// the file hash and size.
pub fn parse_ffprobe(json: &[u8], video_path: &str) -> Result<MediaProbe, ErrorInfo> {
    let raw: RawOutput = serde_json::from_slice(json)
        .map_err(|e| invalid(format!("ffprobe JSON did not parse: {e}")))?;
    let format = raw
        .format
        .ok_or_else(|| invalid("ffprobe reported no format section"))?;
    let duration_s = num(&format.duration)
        .filter(|d| *d > 0.0)
        .ok_or_else(|| invalid("container duration missing or not positive; refusing to guess"))?;
    let start_time_s = num(&format.start_time).unwrap_or(0.0);
    let v = raw
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| invalid("no video stream"))?;
    let (width, height) = match (v.width, v.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => return Err(invalid("video stream has no size")),
    };
    let avg = v.avg_frame_rate.clone().unwrap_or_default();
    let r = v.r_frame_rate.clone().unwrap_or_default();
    let (avg_q, r_q) = (parse_rational(&avg), parse_rational(&r));
    // Compare rationals exactly (cross-multiplied) rather than as floats.
    let vfr = match (avg_q, r_q) {
        (Some(a), Some(b)) => {
            i128::from(a.0) * i128::from(b.1) != i128::from(b.0) * i128::from(a.1)
        }
        _ => true,
    };
    let time_base = v
        .time_base
        .clone()
        .ok_or_else(|| invalid("video stream has no time_base"))?;
    let tb =
        parse_rational(&time_base).ok_or_else(|| invalid(format!("bad time_base {time_base}")))?;
    let v_start = num(&v.start_time).unwrap_or(start_time_s);
    let start_pts = v
        .start_pts
        .unwrap_or_else(|| (v_start * tb.1 as f64 / tb.0 as f64).round() as i64);
    let video = VideoStream {
        index: v.index,
        codec: v.codec_name.clone().unwrap_or_default(),
        profile: v.profile.clone(),
        width,
        height,
        pix_fmt: v.pix_fmt.clone(),
        avg_frame_rate: avg,
        r_frame_rate: r,
        avg_fps: avg_q.map(rational_f64),
        r_fps: r_q.map(rational_f64),
        time_base,
        start_time_s: v_start,
        start_pts,
        duration_s: num(&v.duration),
        nb_frames: v.nb_frames.as_deref().and_then(|n| n.parse().ok()),
    };
    let audio = raw
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("audio"))
        .map(|a| AudioStream {
            index: a.index,
            codec: a.codec_name.clone().unwrap_or_default(),
            sample_rate: a.sample_rate.as_deref().and_then(|s| s.parse().ok()),
            channels: a.channels,
            start_time_s: num(&a.start_time),
            duration_s: num(&a.duration),
        });
    Ok(MediaProbe {
        video_path: video_path.to_string(),
        file_blake3: String::new(),
        file_size_bytes: 0,
        format_name: format.format_name.unwrap_or_default(),
        start_time_s,
        duration_s,
        end_s: start_time_s + duration_s,
        avg_fps: video.avg_fps,
        container_rotation_deg: rotation(v),
        vfr,
        audio_present: audio.is_some(),
        audio,
        video,
    })
}

impl Stage for ProbeStage {
    type Params = ProbeParams;
    type Work = ();
    type Output = MediaProbe;

    fn name(&self) -> &'static str {
        "probe"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: MEDIA_PROBE,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        Vec::new()
    }
    fn params(&self) -> &ProbeParams {
        &self.params
    }
    fn external_inputs(&self) -> Vec<PathBuf> {
        vec![self.video.clone()]
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            tool_versions: [("ffprobe".to_string(), self.ffprobe_version.clone())].into(),
            ..KeyExtras::default()
        }
    }
    fn plan(&self, _inputs: &StageInputs) -> Result<Vec<WorkItem<()>>, StageError> {
        Ok(vec![WorkItem {
            id: "video".into(),
            work: (),
        }])
    }
    async fn process(&self, ctx: &ItemContext, _work: ()) -> Result<MediaProbe, ErrorInfo> {
        let argv: Vec<String> = [
            self.ffprobe.as_str(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .chain([self.params.video_path.clone()])
        .collect();
        let out = run_command(Some(ctx), &argv).await?;
        let mut probe = parse_ffprobe(&out.stdout, &self.params.video_path)?;
        let path = self.video.clone();
        let (hash, size) = on_rayon(move || {
            let size = fs_err::metadata(&path)
                .map_err(|e| crate::util::io_error("cannot stat", &path, e))?
                .len();
            let hash = crate::blobs::hash_file(&path)
                .map_err(|e| crate::util::io_error("cannot hash", &path, e))?;
            Ok((hash, size))
        })
        .await?;
        probe.file_blake3 = hash;
        probe.file_size_bytes = size;
        Ok(probe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"streams":[
      {"index":0,"codec_type":"video","codec_name":"hevc","width":3840,"height":2160,
       "pix_fmt":"yuv420p10le","avg_frame_rate":"1115190000/37180741","r_frame_rate":"88/3",
       "time_base":"1/90000","start_pts":0,"start_time":"0.000000","duration":"2065.596722",
       "nb_frames":"61955","side_data_list":[{"side_data_type":"Display Matrix","rotation":-90}]},
      {"index":1,"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2,
       "time_base":"1/48000","start_time":"0.015208","duration":"2065.551000"}],
     "format":{"format_name":"mov,mp4","start_time":"0.000000","duration":"2065.596722"}}"#;

    #[test]
    fn parses_vfr_rotation_and_audio_offset() {
        let p = parse_ffprobe(SAMPLE.as_bytes(), "in.mp4").unwrap();
        assert!(p.vfr);
        assert_eq!(p.container_rotation_deg, Some(-90.0));
        assert_eq!(p.video.start_pts, 0);
        assert_eq!(p.audio.as_ref().unwrap().start_time_s, Some(0.015208));
        assert!((p.avg_fps.unwrap() - 29.994).abs() < 0.001);
        assert_eq!(p.video.r_fps, Some(88.0 / 3.0));
        assert!((p.end_s - 2065.596722).abs() < 1e-9);
    }

    #[test]
    fn missing_duration_is_an_error_not_zero() {
        let j = SAMPLE.replace(r#","duration":"2065.596722"}}"#, "}}");
        let e = parse_ffprobe(j.as_bytes(), "in.mp4").unwrap_err();
        assert!(e.message.contains("duration"), "{e}");
    }

    #[test]
    fn cfr_is_not_vfr() {
        let j = SAMPLE
            .replace("1115190000/37180741", "30000/1001")
            .replace("88/3", "30000/1001");
        assert!(!parse_ffprobe(j.as_bytes(), "x").unwrap().vfr);
    }
}
