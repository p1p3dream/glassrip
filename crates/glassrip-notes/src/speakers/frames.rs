//! On-demand frame decoding for visual cues.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use image::RgbImage;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

/// Decodes single frames at arbitrary times.
#[async_trait]
pub trait FrameSource: Send + Sync {
    /// The frame shown at `t_s` (seconds on the video timeline).
    async fn frame_at(&self, t_s: f64) -> Result<RgbImage, String>;
    /// The command line used for a frame at `t_s`, for the run manifest.
    fn argv(&self, t_s: f64) -> Vec<String>;
}

/// Decodes frames with `ffmpeg` input seeking, scaled to a fixed width.
#[derive(Debug, Clone)]
pub struct FfmpegFrameSource {
    ffmpeg: String,
    video: PathBuf,
    width: u32,
    height: u32,
    noautorotate: bool,
    threads: Option<u32>,
}

#[derive(serde::Deserialize)]
struct Probe {
    streams: Vec<ProbeStream>,
}

#[derive(serde::Deserialize)]
struct ProbeStream {
    width: Option<u32>,
    height: Option<u32>,
    #[serde(default)]
    side_data_list: Vec<serde_json::Value>,
}

impl FfmpegFrameSource {
    /// Probes `video` and prepares a source producing frames `width` pixels wide.
    ///
    /// With `noautorotate` the container rotation is ignored (phone recordings
    /// often carry a wrong rotation tag); otherwise a 90 degree rotation swaps
    /// the output aspect as ffmpeg's autorotate will.
    pub async fn open(video: &Path, width: u32, noautorotate: bool) -> Result<Self, String> {
        let out = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg("stream=width,height:stream_side_data=rotation")
            .args(["-of", "json"])
            .arg(video)
            .output()
            .await
            .map_err(|e| format!("ffprobe could not start: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "ffprobe failed on {}: {}",
                video.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let probe: Probe =
            serde_json::from_slice(&out.stdout).map_err(|e| format!("ffprobe output: {e}"))?;
        let s = probe.streams.first().ok_or("no video stream")?;
        let (mut w, mut h) = (s.width.ok_or("no width")?, s.height.ok_or("no height")?);
        let rotation = s
            .side_data_list
            .iter()
            .filter_map(|d| d.get("rotation").and_then(serde_json::Value::as_i64))
            .next()
            .unwrap_or(0);
        if !noautorotate && rotation.rem_euclid(180) == 90 {
            std::mem::swap(&mut w, &mut h);
        }
        if w == 0 || h == 0 || width == 0 {
            return Err("zero frame size".into());
        }
        // ffmpeg `scale=W:-2` rounds the height to an even number
        let height = ((f64::from(width) * f64::from(h) / f64::from(w) / 2.0).round() as u32) * 2;
        Ok(Self {
            ffmpeg: "ffmpeg".into(),
            video: video.to_path_buf(),
            width,
            height,
            noautorotate,
            threads: Some(4),
        })
    }

    /// Decoder threads per frame (None: ffmpeg's default, which uses every core).
    pub fn with_threads(mut self, threads: Option<u32>) -> Self {
        self.threads = threads;
        self
    }

    /// Output frame size.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[async_trait]
impl FrameSource for FfmpegFrameSource {
    fn argv(&self, t_s: f64) -> Vec<String> {
        let mut v = vec![self.ffmpeg.clone(), "-v".into(), "error".into()];
        if self.noautorotate {
            v.push("-noautorotate".into());
        }
        if let Some(n) = self.threads {
            v.extend(["-threads".into(), n.to_string()]);
        }
        v.extend([
            "-ss".into(),
            format!("{:.3}", t_s.max(0.0)),
            "-i".into(),
            self.video.display().to_string(),
            "-frames:v".into(),
            "1".into(),
            "-vf".into(),
            format!("scale={}:-2", self.width),
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "rgb24".into(),
            "pipe:1".into(),
        ]);
        v
    }

    async fn frame_at(&self, t_s: f64) -> Result<RgbImage, String> {
        let argv = self.argv(t_s);
        let (prog, args) = argv.split_first().ok_or("empty argv")?;
        let mut child = Command::new(prog)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("ffmpeg could not start: {e}"))?;
        let expected = self.width as usize * self.height as usize * 3;
        let mut buf = Vec::with_capacity(expected);
        let mut stdout = child.stdout.take().ok_or("no ffmpeg stdout")?;
        stdout
            .read_to_end(&mut buf)
            .await
            .map_err(|e| format!("reading ffmpeg output: {e}"))?;
        let out = child
            .wait_with_output()
            .await
            .map_err(|e| format!("waiting for ffmpeg: {e}"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let tail: String = err
                .chars()
                .rev()
                .take(400)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(format!(
                "ffmpeg exited with {}: {}",
                out.status,
                tail.trim()
            ));
        }
        if buf.len() != expected {
            return Err(format!(
                "ffmpeg produced {} bytes, expected {expected} for {}x{} at {t_s:.3} s",
                buf.len(),
                self.width,
                self.height
            ));
        }
        RgbImage::from_raw(self.width, self.height, buf).ok_or_else(|| "frame buffer size".into())
    }
}
