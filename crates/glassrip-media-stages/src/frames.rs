//! `frames` (spec 6.3): sampled frames as JPEG, timestamps from PTS.
//!
//! The video timeline is cut into interval buckets `[k, k+1) * interval` after the video
//! stream's start. Bucket membership is computed on integer PTS (the interval must be a
//! whole number of time-base ticks), from the demuxer's packet index, so the expected grid
//! is known before decoding. Each non-empty bucket yields one frame, `f<k:06>`:
//!
//! - `grid`: the first frame of the bucket (every frame is decoded);
//! - `sync`: the first sync (key) frame of the bucket (`-skip_frame nokey`, only sync frames
//!   are decoded, typically 20 to 30 times less work);
//! - `auto` (default): `sync` for a chunk when every non-empty bucket in it has a sync
//!   frame, else `grid`.
//!
//! Decoding runs `ffmpeg -noautorotate -copyts` per chunk of buckets, several chunks in
//! parallel, writing JPEGs to disk (bounded memory). `showinfo` output is parsed strictly:
//! a frame count, PTS range, or bucket set that differs from the expectation is an error,
//! never a positional guess. Frames go to the blob store and are linked into
//! `frames/sampled/` in the run directory.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use glassrip_core::config::HwAccel;
use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::Serialize;
use tokio::sync::OnceCell;

use crate::blobs::BlobStore;
use crate::schema::{
    FRAMES, FRAMES_DIR, FrameRecord, MEDIA_PROBE, MediaProbe, ORIENTATION, Orientation, Sampling,
    v1,
};
use crate::util::{STDERR_TAIL, on_rayon, parse_rational, run_command, tail};

/// Frame selection rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    /// Sync frames when every bucket of a chunk has one, else grid.
    #[default]
    Auto,
    /// Always the first frame of each bucket (full decode).
    Grid,
    /// Always the first sync frame; a bucket without one is an error.
    Sync,
}

/// Frame sampling parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct FramesParams {
    /// Sampling interval in seconds.
    pub interval_s: f64,
    /// Output width (height keeps the aspect, rounded to even).
    pub scale_width: u32,
    /// Decoder choice.
    pub hwaccel: HwAccel,
    /// Selection rule.
    pub sampling: SamplingMode,
    /// Chunk length in seconds (one ffmpeg process per chunk).
    pub chunk_s: f64,
    /// ffmpeg processes run in parallel.
    pub chunk_concurrency: u32,
    /// MJPEG quality (`-q:v`, 2 is best).
    pub jpeg_q: u32,
}

impl Default for FramesParams {
    fn default() -> Self {
        Self {
            interval_s: 2.0,
            scale_width: 1920,
            hwaccel: HwAccel::None,
            sampling: SamplingMode::Auto,
            chunk_s: 240.0,
            chunk_concurrency: 8,
            jpeg_q: 2,
        }
    }
}

/// One chunk of buckets decoded by a single ffmpeg run.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkPlan {
    /// Chunk index.
    pub index: u32,
    /// First bucket (inclusive).
    pub k0: u64,
    /// Last bucket (exclusive).
    pub k1: u64,
    /// Non-empty buckets expected from this chunk.
    pub buckets: BTreeSet<u64>,
    /// Selection rule.
    pub sampling: Sampling,
    /// Whether this is the final chunk (decode to the end).
    pub last: bool,
}

/// Per-item work: a bucket and its chunk.
#[derive(Debug, Clone)]
pub struct FrameWork {
    chunk: Arc<ChunkPlan>,
    bucket: u64,
}

#[derive(Debug, Clone)]
struct Grid {
    /// Stream start PTS.
    p0: i64,
    /// Interval in ticks.
    q: i64,
    /// Time base.
    tb: (i64, i64),
}

impl Grid {
    fn bucket(&self, pts: i64) -> Option<u64> {
        (pts >= self.p0).then(|| ((pts - self.p0) / self.q) as u64)
    }
    fn secs(&self, pts: i64) -> f64 {
        pts as f64 * self.tb.0 as f64 / self.tb.1 as f64
    }
    fn start_pts(&self, k: u64) -> i64 {
        self.p0 + k as i64 * self.q
    }
}

type ChunkResult = Result<Arc<BTreeMap<u64, FrameRecord>>, ErrorInfo>;

/// The `frames` stage.
#[derive(Debug)]
pub struct FramesStage {
    params: FramesParams,
    ffmpeg: String,
    ffprobe: String,
    tool_versions: BTreeMap<String, String>,
    run_root: PathBuf,
    blobs: BlobStore,
    grid: OnceLock<(Grid, MediaProbe, u32)>,
    chunks: Mutex<HashMap<u32, Arc<OnceCell<ChunkResult>>>>,
    index_command: Mutex<Option<(Vec<String>, f64)>>,
}

impl FramesStage {
    /// Stage writing into `run_root` and `blobs`. `tool_versions` (ffmpeg, ffprobe) go
    /// into the cache key.
    pub fn new(
        params: FramesParams,
        ffmpeg: String,
        ffprobe: String,
        tool_versions: BTreeMap<String, String>,
        run_root: PathBuf,
        blobs: BlobStore,
    ) -> Self {
        Self {
            params,
            ffmpeg,
            ffprobe,
            tool_versions,
            run_root,
            blobs,
            grid: OnceLock::new(),
            chunks: Mutex::new(HashMap::new()),
            index_command: Mutex::new(None),
        }
    }

    fn chunk_buckets(&self) -> u64 {
        ((self.params.chunk_s / self.params.interval_s).round() as u64).max(1)
    }
}

/// Interval in time-base ticks; errors when it is not a whole number of ticks.
pub fn interval_ticks(interval_s: f64, tb: (i64, i64)) -> Result<i64, String> {
    let exact = interval_s * tb.1 as f64 / tb.0 as f64;
    let q = exact.round();
    if q < 1.0 || (exact - q).abs() > 1e-6 * exact.max(1.0) {
        return Err(format!(
            "interval {interval_s} s is not a whole number of time-base ticks ({}/{})",
            tb.0, tb.1
        ));
    }
    Ok(q as i64)
}

/// Parses `ffprobe -show_entries packet=pts,flags -of csv=p=0` output into `(pts, is_sync)`.
pub fn parse_packets(text: &str) -> Result<Vec<(i64, bool)>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (pts, flags) = line.split_once(',').unwrap_or((line, ""));
        if pts == "N/A" {
            return Err(format!(
                "packet {i} has no pts; cannot build the frame grid"
            ));
        }
        let pts: i64 = pts
            .parse()
            .map_err(|_| format!("packet line {i} is malformed: `{line}`"))?;
        out.push((pts, flags.contains('K')));
    }
    if out.is_empty() {
        return Err("the video stream has no packets".into());
    }
    Ok(out)
}

/// Splits non-empty buckets into chunks and picks each chunk's sampling rule.
pub fn plan_chunks(
    packets: &[(i64, bool)],
    p0: i64,
    q: i64,
    chunk_buckets: u64,
    mode: SamplingMode,
) -> Result<Vec<ChunkPlan>, String> {
    let mut any = BTreeSet::new();
    let mut sync = BTreeSet::new();
    for &(pts, key) in packets {
        if pts < p0 {
            continue;
        }
        let k = ((pts - p0) / q) as u64;
        any.insert(k);
        if key {
            sync.insert(k);
        }
    }
    let Some(&last) = any.iter().next_back() else {
        return Err("no packets at or after the stream start".into());
    };
    let n_chunks = last / chunk_buckets + 1;
    let mut chunks = Vec::new();
    for c in 0..n_chunks {
        let (k0, k1) = (c * chunk_buckets, (c + 1) * chunk_buckets);
        let buckets: BTreeSet<u64> = any.range(k0..k1).copied().collect();
        if buckets.is_empty() {
            continue;
        }
        let all_sync = buckets.iter().all(|k| sync.contains(k));
        let sampling = match mode {
            SamplingMode::Grid => Sampling::Grid,
            SamplingMode::Sync if all_sync => Sampling::Sync,
            SamplingMode::Sync => {
                let missing: Vec<u64> = buckets.difference(&sync).copied().take(5).collect();
                return Err(format!(
                    "sampling = sync but buckets {missing:?} have no sync frame; use auto or grid"
                ));
            }
            SamplingMode::Auto if all_sync => Sampling::Sync,
            SamplingMode::Auto => Sampling::Grid,
        };
        chunks.push(ChunkPlan {
            index: u32::try_from(c).map_err(|_| "too many chunks".to_string())?,
            k0,
            k1,
            buckets,
            sampling,
            last: c + 1 == n_chunks,
        });
    }
    Ok(chunks)
}

/// One parsed `showinfo` frame line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShowInfo {
    /// Output frame number.
    pub n: u64,
    /// PTS in time-base ticks.
    pub pts: i64,
    /// PTS in seconds as printed.
    pub pts_time: f64,
}

fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let i = line.find(key)? + key.len();
    let rest = line[i..].trim_start();
    Some(rest.split_whitespace().next().unwrap_or(""))
}

/// Parses every `showinfo` frame line in ffmpeg's stderr. A line that looks like a frame
/// line but does not parse is an error.
pub fn parse_showinfo(stderr: &str) -> Result<Vec<ShowInfo>, String> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        if !line.contains("Parsed_showinfo") || !line.contains(" n:") || !line.contains(" pts:") {
            continue;
        }
        let bad = || format!("unparsable showinfo line: `{line}`");
        let n = field_after(line, " n:")
            .and_then(|v| v.parse().ok())
            .ok_or_else(bad)?;
        let pts = field_after(line, " pts:")
            .and_then(|v| v.parse().ok())
            .ok_or_else(bad)?;
        let pts_time = field_after(line, " pts_time:")
            .and_then(|v| v.parse().ok())
            .ok_or_else(bad)?;
        out.push(ShowInfo { n, pts, pts_time });
    }
    Ok(out)
}

fn rotation_filter(deg: u32) -> Result<&'static str, String> {
    match deg {
        0 => Ok(""),
        90 => Ok("transpose=clock,"),
        180 => Ok("hflip,vflip,"),
        270 => Ok("transpose=cclock,"),
        other => Err(format!("unsupported rotation {other}")),
    }
}

/// Builds the ffmpeg argv for one chunk.
#[allow(clippy::too_many_arguments)]
fn chunk_argv(
    ffmpeg: &str,
    video: &str,
    chunk: &ChunkPlan,
    grid: &Grid,
    rotation: u32,
    params: &FramesParams,
    hw: Option<&str>,
    out_pattern: &Path,
) -> Result<Vec<String>, String> {
    let (sp, ep) = (grid.start_pts(chunk.k0), grid.start_pts(chunk.k1));
    let select = format!(
        "select='gte(pts\\,{sp})*lt(pts\\,{ep})*(isnan(prev_pts)+lt(prev_pts\\,{sp})+gt(floor((pts-{p0})/{q})\\,floor((prev_pts-{p0})/{q})))'",
        p0 = grid.p0,
        q = grid.q
    );
    let vf = format!(
        "{select},showinfo,{}scale={}:-2:flags=bicubic",
        rotation_filter(rotation)?,
        params.scale_width
    );
    let mut a: Vec<String> = [
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-y",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    a.insert(0, ffmpeg.to_string());
    if let Some(h) = hw {
        a.extend(["-hwaccel".into(), h.into()]);
    }
    a.push("-noautorotate".into());
    if chunk.sampling == Sampling::Sync {
        a.extend(["-skip_frame".into(), "nokey".into()]);
    }
    // Seek a few seconds early (the select filter does the exact cut); read only as far
    // as the chunk needs.
    let seek = (grid.secs(sp) - 3.0).max(0.0);
    if chunk.k0 > 0 && seek > 0.0 {
        a.extend(["-ss".into(), format!("{seek:.6}")]);
    }
    if !chunk.last {
        a.extend(["-t".into(), format!("{:.6}", grid.secs(ep) - seek + 1.0)]);
    }
    a.extend(["-copyts".into(), "-i".into(), video.into()]);
    a.extend(
        ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf"]
            .iter()
            .map(|s| (*s).to_string()),
    );
    a.push(vf);
    a.extend(
        ["-fps_mode", "passthrough", "-c:v", "mjpeg", "-q:v"]
            .iter()
            .map(|s| (*s).to_string()),
    );
    a.push(params.jpeg_q.to_string());
    a.extend([
        "-f".into(),
        "image2".into(),
        "-start_number".into(),
        "0".into(),
    ]);
    a.push(out_pattern.display().to_string());
    Ok(a)
}

/// Checks showinfo lines against the chunk's expectation and returns `(bucket, info)` in
/// order.
fn validate_chunk(
    infos: &[ShowInfo],
    files: usize,
    chunk: &ChunkPlan,
    grid: &Grid,
) -> Result<Vec<(u64, ShowInfo)>, String> {
    if infos.len() != files {
        return Err(format!(
            "showinfo reported {} frames but ffmpeg wrote {files} files",
            infos.len()
        ));
    }
    let (sp, ep) = (grid.start_pts(chunk.k0), grid.start_pts(chunk.k1));
    let mut out = Vec::with_capacity(infos.len());
    for (i, s) in infos.iter().enumerate() {
        if s.n != i as u64 {
            return Err(format!(
                "showinfo frame numbers are not sequential at {i}: n={}",
                s.n
            ));
        }
        if s.pts < sp || s.pts >= ep {
            return Err(format!("frame pts {} outside chunk [{sp}, {ep})", s.pts));
        }
        let secs = grid.secs(s.pts);
        if (secs - s.pts_time).abs() > 1e-3 {
            return Err(format!(
                "pts {} with time base {}/{} is {secs} s but showinfo says {} s (time base mismatch)",
                s.pts, grid.tb.0, grid.tb.1, s.pts_time
            ));
        }
        let k = grid
            .bucket(s.pts)
            .ok_or_else(|| format!("pts {} before stream start", s.pts))?;
        if let Some((pk, _)) = out.last() {
            if k <= *pk {
                return Err(format!(
                    "two frames selected for bucket {k} (or out of order)"
                ));
            }
        }
        out.push((k, *s));
    }
    let got: BTreeSet<u64> = out.iter().map(|(k, _)| *k).collect();
    if got != chunk.buckets {
        let missing: Vec<u64> = chunk.buckets.difference(&got).copied().take(10).collect();
        let extra: Vec<u64> = got.difference(&chunk.buckets).copied().take(10).collect();
        return Err(format!(
            "decoded buckets differ from the packet index: missing {missing:?}, unexpected {extra:?} ({} expected, {} decoded)",
            chunk.buckets.len(),
            got.len()
        ));
    }
    Ok(out)
}

fn hw_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "videotoolbox"
    } else {
        "cuda"
    }
}

impl FramesStage {
    async fn decode_chunk(&self, ctx: &ItemContext, chunk: Arc<ChunkPlan>) -> ChunkResult {
        let (grid, probe, rotation) = self
            .grid
            .get()
            .ok_or_else(|| ErrorInfo::new(ErrorCode::Internal, "frames stage was not planned"))?;
        if let Some((argv, wall)) = self
            .index_command
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            ctx.record_command(argv, Some(0), Some(wall));
        }
        let tmp = self
            .run_root
            .join("frames/.tmp")
            .join(format!("chunk-{:05}", chunk.index));
        let io = |what: &str, p: &Path, e: std::io::Error| crate::util::io_error(what, p, e);
        if tmp.exists() {
            fs_err::remove_dir_all(&tmp).map_err(|e| io("cannot clear", &tmp, e))?;
        }
        fs_err::create_dir_all(&tmp).map_err(|e| io("cannot create", &tmp, e))?;
        let attempts: Vec<(Option<&str>, &str)> = match self.params.hwaccel {
            HwAccel::None => vec![(None, "software")],
            HwAccel::Auto => vec![(Some(hw_name()), hw_name()), (None, "software")],
        };
        let mut last_err = None;
        let mut decoded = None;
        for (hw, label) in attempts {
            let argv = chunk_argv(
                &self.ffmpeg,
                &probe.video_path,
                &chunk,
                grid,
                *rotation,
                &self.params,
                hw,
                &tmp.join("%06d.jpg"),
            )
            .map_err(|m| ErrorInfo::new(ErrorCode::InvalidInput, m))?;
            match run_command(Some(ctx), &argv).await {
                Ok(out) => {
                    decoded = Some((out, label));
                    break;
                }
                Err(e) => {
                    tracing::warn!(chunk = chunk.index, decoder = label, error = %e, "chunk decode failed");
                    // Start the retry from an empty directory.
                    let _ = fs_err::remove_dir_all(&tmp);
                    fs_err::create_dir_all(&tmp).map_err(|e| io("cannot create", &tmp, e))?;
                    last_err = Some(e);
                }
            }
        }
        let Some((out, decoder)) = decoded else {
            return Err(last_err
                .unwrap_or_else(|| ErrorInfo::new(ErrorCode::Internal, "no decode attempted")));
        };
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let chunk2 = Arc::clone(&chunk);
        let (grid2, blobs, run_root) = (grid.clone(), self.blobs.clone(), self.run_root.clone());
        let decoder = decoder.to_string();
        let tmp2 = tmp.clone();
        let result = on_rayon(move || {
            let fail = |m: String| {
                ErrorInfo::new(ErrorCode::ExternalCommand, m)
                    .with_raw_text(tail(stderr.as_bytes(), STDERR_TAIL))
            };
            let infos = parse_showinfo(&stderr).map_err(&fail)?;
            let files = fs_err::read_dir(&tmp2)
                .map_err(|e| crate::util::io_error("cannot list", &tmp2, e))?
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "jpg"))
                .count();
            let frames = validate_chunk(&infos, files, &chunk2, &grid2).map_err(&fail)?;
            let records: Result<Vec<(u64, FrameRecord)>, ErrorInfo> = frames
                .par_iter()
                .enumerate()
                .map(|(i, (k, s))| {
                    let src = tmp2.join(format!("{i:06}.jpg"));
                    let (width, height) = image::image_dimensions(&src).map_err(|e| {
                        ErrorInfo::new(
                            ErrorCode::ExternalCommand,
                            format!("bad JPEG {}: {e}", src.display()),
                        )
                    })?;
                    let hash = blobs
                        .put_move(&src, "jpg")
                        .map_err(|e| ErrorInfo::new(ErrorCode::Io, e.to_string()))?;
                    let frame_id = format!("f{k:06}");
                    let rel = format!("{FRAMES_DIR}/{frame_id}.jpg");
                    blobs
                        .materialize(&run_root, &rel, &hash, "frames", false)
                        .map_err(|e| ErrorInfo::new(ErrorCode::Io, e.to_string()))?;
                    Ok((
                        *k,
                        FrameRecord {
                            frame_id,
                            bucket: *k,
                            pts: s.pts,
                            pts_s: grid2.secs(s.pts),
                            path: rel,
                            blake3: hash,
                            width,
                            height,
                            sampling: chunk2.sampling,
                            decoder: decoder.clone(),
                        },
                    ))
                })
                .collect();
            Ok(Arc::new(records?.into_iter().collect::<BTreeMap<_, _>>()))
        })
        .await;
        let _ = fs_err::remove_dir_all(&tmp);
        result
    }
}

impl Stage for FramesStage {
    type Params = FramesParams;
    type Work = FrameWork;
    type Output = FrameRecord;

    fn name(&self) -> &'static str {
        "frames"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: FRAMES,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: MEDIA_PROBE,
                major: 1,
            },
            InputDecl {
                schema: ORIENTATION,
                major: 1,
            },
        ]
    }
    fn params(&self) -> &FramesParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            decoder: Some(
                match self.params.hwaccel {
                    HwAccel::None => "software",
                    HwAccel::Auto => "auto",
                }
                .into(),
            ),
            tool_versions: self.tool_versions.clone(),
            ..KeyExtras::default()
        }
    }
    fn concurrency(&self) -> usize {
        (self.params.chunk_concurrency.max(1) as usize) * self.chunk_buckets() as usize
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<FrameWork>>, StageError> {
        let probe = inputs
            .read_ok::<MediaProbe>(MEDIA_PROBE)?
            .into_iter()
            .next()
            .ok_or_else(|| StageError::Invalid("media_probe has no ok item".into()))?
            .1;
        let orientation = inputs
            .read_ok::<Orientation>(ORIENTATION)?
            .into_iter()
            .next()
            .ok_or_else(|| StageError::Invalid("orientation has no ok item".into()))?
            .1;
        let tb = parse_rational(&probe.video.time_base).ok_or_else(|| {
            StageError::Invalid(format!("bad time base {}", probe.video.time_base))
        })?;
        let q = interval_ticks(self.params.interval_s, tb).map_err(StageError::Invalid)?;
        let argv: Vec<String> = [
            self.ffprobe.as_str(),
            "-v",
            "error",
            "-select_streams",
            &format!("{}", probe.video.index),
            "-show_entries",
            "packet=pts,flags",
            "-of",
            "csv=p=0",
            &probe.video_path,
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let started = Instant::now();
        let out = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| StageError::Invalid(format!("cannot run ffprobe: {e}")))?;
        if !out.status.success() {
            return Err(StageError::Invalid(format!(
                "packet listing failed ({}): {}",
                out.status,
                tail(&out.stderr, STDERR_TAIL)
            )));
        }
        *self
            .index_command
            .lock()
            .unwrap_or_else(PoisonError::into_inner) =
            Some((argv, started.elapsed().as_secs_f64()));
        let packets =
            parse_packets(&String::from_utf8_lossy(&out.stdout)).map_err(StageError::Invalid)?;
        let grid = Grid {
            p0: probe.video.start_pts,
            q,
            tb,
        };
        let chunks = plan_chunks(
            &packets,
            grid.p0,
            q,
            self.chunk_buckets(),
            self.params.sampling,
        )
        .map_err(StageError::Invalid)?;
        let n_sync = chunks
            .iter()
            .filter(|c| c.sampling == Sampling::Sync)
            .count();
        tracing::info!(
            chunks = chunks.len(),
            sync_chunks = n_sync,
            frames = chunks.iter().map(|c| c.buckets.len()).sum::<usize>(),
            "planned frame grid"
        );
        rotation_filter(orientation.applied_rotation_deg).map_err(StageError::Invalid)?;
        // A second plan on the same stage object would reuse stale chunk results.
        self.chunks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        if self
            .grid
            .set((grid, probe, orientation.applied_rotation_deg))
            .is_err()
        {
            return Err(StageError::Invalid(
                "frames stage objects are single-use; build a new one per run".into(),
            ));
        }
        let mut items = Vec::new();
        for c in chunks {
            let c = Arc::new(c);
            for &k in &c.buckets {
                items.push(WorkItem {
                    id: format!("f{k:06}"),
                    work: FrameWork {
                        chunk: Arc::clone(&c),
                        bucket: k,
                    },
                });
            }
        }
        Ok(items)
    }
    async fn process(&self, ctx: &ItemContext, work: FrameWork) -> Result<FrameRecord, ErrorInfo> {
        let cell = {
            let mut map = self.chunks.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(map.entry(work.chunk.index).or_default())
        };
        let result = cell
            .get_or_init(|| self.decode_chunk(ctx, Arc::clone(&work.chunk)))
            .await
            .clone()?;
        result.get(&work.bucket).cloned().ok_or_else(|| {
            ErrorInfo::new(
                ErrorCode::Internal,
                format!("bucket {} missing from its decoded chunk", work.bucket),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks() {
        assert_eq!(interval_ticks(2.0, (1, 90000)).unwrap(), 180000);
        assert_eq!(interval_ticks(0.5, (1, 600)).unwrap(), 300);
        assert!(interval_ticks(0.3333, (1, 30)).is_err());
    }

    #[test]
    fn showinfo_parse_handles_wide_counters() {
        let s = "[Parsed_showinfo_1 @ 0x1] n:   0 pts:      0 pts_time:0       duration: 3\n\
                 junk line\n\
                 [Parsed_showinfo_1 @ 0x1] n:1234 pts:270997 pts_time:3.01108 duration: 3000\n";
        let v = parse_showinfo(s).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(
            v[1],
            ShowInfo {
                n: 1234,
                pts: 270997,
                pts_time: 3.01108
            }
        );
        assert!(parse_showinfo("[Parsed_showinfo_1 @ 0x1] n: x pts: 1 pts_time:0").is_err());
    }

    #[test]
    fn chunks_pick_sync_only_when_every_bucket_has_one() {
        // Buckets of 10 ticks; frames every 2 ticks; sync every 10 except bucket 3.
        let packets: Vec<(i64, bool)> = (0..60).map(|i| (i * 2, i % 5 == 0 && i != 15)).collect();
        let c = plan_chunks(&packets, 0, 10, 2, SamplingMode::Auto).unwrap();
        assert_eq!(c.len(), 6);
        assert_eq!(c[0].sampling, Sampling::Sync);
        assert_eq!(c[1].sampling, Sampling::Grid, "bucket 3 has no sync frame");
        assert!(c[5].last && !c[4].last);
        assert!(plan_chunks(&packets, 0, 10, 2, SamplingMode::Sync).is_err());
        let g = plan_chunks(&packets, 0, 10, 2, SamplingMode::Grid).unwrap();
        assert!(g.iter().all(|c| c.sampling == Sampling::Grid));
    }

    #[test]
    fn gaps_are_not_expected_buckets() {
        let packets = vec![(0, true), (5, false), (25, true)];
        let c = plan_chunks(&packets, 0, 10, 10, SamplingMode::Auto).unwrap();
        assert_eq!(c[0].buckets, [0u64, 2].into_iter().collect());
    }

    fn chunk(buckets: &[u64]) -> ChunkPlan {
        ChunkPlan {
            index: 0,
            k0: 0,
            k1: 4,
            buckets: buckets.iter().copied().collect(),
            sampling: Sampling::Grid,
            last: true,
        }
    }

    #[test]
    fn validation_is_strict() {
        let grid = Grid {
            p0: 0,
            q: 10,
            tb: (1, 10),
        };
        let si = |n, pts| ShowInfo {
            n,
            pts,
            pts_time: pts as f64 / 10.0,
        };
        let ok = [si(0, 0), si(1, 12), si(2, 20)];
        assert_eq!(
            validate_chunk(&ok, 3, &chunk(&[0, 1, 2]), &grid)
                .unwrap()
                .len(),
            3
        );
        assert!(
            validate_chunk(&ok, 2, &chunk(&[0, 1, 2]), &grid).is_err(),
            "file count"
        );
        assert!(
            validate_chunk(&ok, 3, &chunk(&[0, 1, 2, 3]), &grid).is_err(),
            "missing bucket"
        );
        let dup = [si(0, 0), si(1, 4)];
        assert!(
            validate_chunk(&dup, 2, &chunk(&[0]), &grid).is_err(),
            "two in one bucket"
        );
        let skew = [ShowInfo {
            n: 0,
            pts: 0,
            pts_time: 5.0,
        }];
        assert!(
            validate_chunk(&skew, 1, &chunk(&[0]), &grid).is_err(),
            "time base"
        );
    }

    #[test]
    fn argv_has_noautorotate_copyts_and_rotation() {
        let grid = Grid {
            p0: 0,
            q: 180000,
            tb: (1, 90000),
        };
        let mut c = chunk(&[5]);
        c.k0 = 4;
        c.k1 = 8;
        c.last = false;
        c.sampling = Sampling::Sync;
        let a = chunk_argv(
            "ffmpeg",
            "in.mp4",
            &c,
            &grid,
            90,
            &FramesParams::default(),
            None,
            Path::new("o/%06d.jpg"),
        )
        .unwrap();
        let s = a.join(" ");
        assert!(
            s.contains(
                "-noautorotate -skip_frame nokey -ss 5.000000 -t 12.000000 -copyts -i in.mp4"
            ),
            "{s}"
        );
        assert!(s.contains("transpose=clock,scale=1920:-2"), "{s}");
        assert!(s.contains("gte(pts\\,720000)*lt(pts\\,1440000)"), "{s}");
    }
}
