//! `orient` (spec 6.2): content orientation by PP-LCNet_x1_0_doc_ori vote.
//!
//! Container rotation is never trusted. About 12 frames spread over the video are decoded
//! with `-noautorotate`; each is resized (short side `sample_short_side`) and cut into a
//! grid of 224x224 tiles; tiles whose own top probability passes a floor are averaged
//! (blank tiles abstain); frames whose top probability passes the confidence floor vote. The winner must have enough votes and a clear margin
//! over the runner-up, otherwise the stage fails naming the ambiguous rotations.
//!
//! The model runs only with the `onnx`, `onnx-dynamic` or `cuda` feature. Without them the
//! stage requires `override_rotation_deg`.

use std::path::PathBuf;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use image::RgbImage;
use schemars::JsonSchema;
use serde::Serialize;

use crate::schema::{
    MEDIA_PROBE, MediaProbe, ModelRef, ORIENTATION, OrientMethod, OrientSample, Orientation,
    RotationVotes, v1,
};

/// Orientation parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct OrientParams {
    /// Frames sampled across the video.
    pub sample_frames: u32,
    /// A sample votes only when its top probability is at least this.
    pub min_confidence: f64,
    /// The winner needs at least this many votes.
    pub min_votes: u32,
    /// ... and at least this many more than the runner-up.
    pub min_margin: u32,
    /// Short side of the decoded samples (and of the tiled image).
    pub sample_short_side: u32,
    /// Tiles whose top probability is below this abstain.
    pub tile_min_confidence: f64,
    /// Skip the model and apply this clockwise rotation (0, 90, 180, 270).
    pub override_rotation_deg: Option<u32>,
}

impl Default for OrientParams {
    fn default() -> Self {
        Self {
            sample_frames: 12,
            min_confidence: 0.6,
            min_votes: 3,
            min_margin: 2,
            sample_short_side: 896,
            tile_min_confidence: 0.6,
            override_rotation_deg: None,
        }
    }
}

/// Rotations indexed like the model's classes.
pub const ROTATIONS: [u32; 4] = [0, 90, 180, 270];

/// Clockwise correction for model class `c` (`0`, `90`, `180`, `270` in the model's label
/// list). Calibrated with the `orient_calibrate` example: content turned `d` degrees
/// clockwise is labelled `d`, so the correction turns it back by `360 - d`.
pub fn correction_for_class(c: usize) -> u32 {
    (360 - ROTATIONS[c % 4]) % 360
}

/// The `orient` stage.
#[derive(Debug, Clone)]
pub struct OrientStage {
    params: OrientParams,
    ffmpeg: String,
    ffmpeg_version: String,
    models_dir: PathBuf,
    allow_download: bool,
    model_digest: Option<String>,
}

impl OrientStage {
    /// Stage using `ffmpeg` for sampling and the models directory for the classifier.
    pub fn new(
        params: OrientParams,
        ffmpeg: String,
        ffmpeg_version: String,
        models_dir: PathBuf,
        allow_download: bool,
    ) -> Self {
        let model_digest = if params.override_rotation_deg.is_some() {
            None
        } else {
            crate::models::entry(crate::models::ORIENT_MODEL)
                .ok()
                .map(|e| e.sha256)
        };
        Self {
            params,
            ffmpeg,
            ffmpeg_version,
            models_dir,
            allow_download,
            model_digest,
        }
    }
}

/// Decides the rotation from per-sample probabilities (`[0, 90, 180, 270]` corrections).
pub fn decide(
    samples: &[OrientSample],
    p: &OrientParams,
) -> Result<(u32, RotationVotes), ErrorInfo> {
    let mut counts = [0u32; 4];
    for s in samples.iter().filter(|s| s.confident) {
        if let Some(i) = ROTATIONS.iter().position(|r| *r == s.predicted_deg) {
            counts[i] += 1;
        }
    }
    let votes = RotationVotes {
        deg_0: counts[0],
        deg_90: counts[1],
        deg_180: counts[2],
        deg_270: counts[3],
    };
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|a, b| counts[*b].cmp(&counts[*a]).then(a.cmp(b)));
    let (win, second) = (order[0], order[1]);
    let describe = || {
        format!(
            "votes 0={} 90={} 180={} 270={} from {} confident of {} samples",
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts.iter().sum::<u32>(),
            samples.len()
        )
    };
    if counts[win] < p.min_votes {
        return Err(ErrorInfo::new(
            ErrorCode::Validation,
            format!(
                "orientation undecided: best rotation {} has {} votes, need {} ({}); set orient.override_rotation_deg to proceed",
                ROTATIONS[win],
                counts[win],
                p.min_votes,
                describe()
            ),
        ));
    }
    if counts[win] < counts[second] + p.min_margin {
        return Err(ErrorInfo::new(
            ErrorCode::Validation,
            format!(
                "orientation ambiguous between {} and {} degrees ({}); set orient.override_rotation_deg to proceed",
                ROTATIONS[win],
                ROTATIONS[second],
                describe()
            ),
        ));
    }
    Ok((ROTATIONS[win], votes))
}

/// ImageNet normalization used by the model.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Model input size.
pub const INPUT: u32 = 224;

/// Normalized CHW tensor of a `224 x 224` window at `(x0, y0)`.
fn tensor_at(r: &RgbImage, x0: u32, y0: u32) -> Vec<f32> {
    let n = (INPUT * INPUT) as usize;
    let mut t = vec![0f32; 3 * n];
    for y in 0..INPUT {
        for x in 0..INPUT {
            let p = r.get_pixel(x0 + x, y0 + y);
            let i = (y * INPUT + x) as usize;
            for c in 0..3 {
                t[c * n + i] = (f32::from(p[c]) / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    t
}

/// Resizes the image so its short side is `short_side` (at least 224, bilinear) and cuts a
/// centered grid of non-overlapping 224x224 tiles, as normalized CHW f32 tensors. Screen
/// text is small; tiles at a higher scale let the document model see readable glyphs.
pub fn preprocess_tiles(img: &RgbImage, short_side: u32) -> Vec<Vec<f32>> {
    let (w, h) = img.dimensions();
    let short = w.min(h).max(1);
    let target = short_side.max(INPUT);
    let scale = f64::from(target) / f64::from(short);
    let nw = ((f64::from(w) * scale).round() as u32).max(INPUT);
    let nh = ((f64::from(h) * scale).round() as u32).max(INPUT);
    let r = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle);
    let (cols, rows) = (nw / INPUT, nh / INPUT);
    let (ox, oy) = ((nw - cols * INPUT) / 2, (nh - rows * INPUT) / 2);
    let mut out = Vec::with_capacity((cols * rows) as usize);
    for ty in 0..rows {
        for tx in 0..cols {
            out.push(tensor_at(&r, ox + tx * INPUT, oy + ty * INPUT));
        }
    }
    out
}

/// Combines per-tile class probabilities: the mean over tiles whose top probability is at
/// least `tile_min_conf` (blank or ambiguous tiles abstain), else the mean over all tiles.
pub fn aggregate_tiles(tiles: &[[f64; 4]], tile_min_conf: f64) -> [f64; 4] {
    let confident: Vec<&[f64; 4]> = tiles
        .iter()
        .filter(|p| p.iter().copied().fold(0.0, f64::max) >= tile_min_conf)
        .collect();
    let pool: Vec<&[f64; 4]> = if confident.is_empty() {
        tiles.iter().collect()
    } else {
        confident
    };
    let mut m = [0.0f64; 4];
    for p in &pool {
        for (a, b) in m.iter_mut().zip(p.iter()) {
            *a += b / pool.len().max(1) as f64;
        }
    }
    m
}

/// Softmax unless the values already look like probabilities.
pub fn to_probs(v: &[f32]) -> [f64; 4] {
    let mut out = [0.0f64; 4];
    let sum: f32 = v.iter().sum();
    let is_probs = v.iter().all(|x| (0.0..=1.0).contains(x)) && (sum - 1.0).abs() < 1e-3;
    if is_probs {
        for (o, x) in out.iter_mut().zip(v) {
            *o = f64::from(*x);
        }
        return out;
    }
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = v.iter().map(|x| f64::from(x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    for (o, x) in out.iter_mut().zip(e) {
        *o = x / s;
    }
    out
}

/// Maps class probabilities to correction probabilities and builds a sample.
pub fn sample_from_class_probs(t_s: f64, class_probs: [f64; 4], min_conf: f64) -> OrientSample {
    let mut probs = [0.0f64; 4];
    for (c, p) in class_probs.iter().enumerate() {
        let corr = correction_for_class(c);
        if let Some(i) = ROTATIONS.iter().position(|r| *r == corr) {
            probs[i] += p;
        }
    }
    let (mut bi, mut bp) = (0usize, f64::NEG_INFINITY);
    for (i, p) in probs.iter().enumerate() {
        if *p > bp {
            bp = *p;
            bi = i;
        }
    }
    OrientSample {
        t_s,
        probs,
        predicted_deg: ROTATIONS[bi],
        confident: bp >= min_conf,
    }
}

/// Sample times: centers of `n` equal slices of `[start, end)`.
pub fn sample_times(start: f64, end: f64, n: u32) -> Vec<f64> {
    let n = n.max(1);
    let d = (end - start).max(0.0);
    (0..n)
        .map(|k| start + d * (f64::from(k) + 0.5) / f64::from(n))
        .collect()
}

async fn decode_sample(
    ctx: &ItemContext,
    ffmpeg: &str,
    video: &str,
    t: f64,
    short: u32,
) -> Result<RgbImage, ErrorInfo> {
    let scale = format!("scale='if(gte(iw,ih),-2,{short})':'if(gte(iw,ih),{short},-2)'");
    let argv: Vec<String> = vec![
        ffmpeg.into(),
        "-hide_banner".into(),
        "-nostdin".into(),
        "-v".into(),
        "error".into(),
        "-noautorotate".into(),
        "-ss".into(),
        format!("{t:.3}"),
        "-i".into(),
        video.into(),
        "-map".into(),
        "0:v:0".into(),
        "-frames:v".into(),
        "1".into(),
        "-vf".into(),
        scale,
        "-f".into(),
        "image2pipe".into(),
        "-c:v".into(),
        "png".into(),
        "pipe:1".into(),
    ];
    let out = crate::util::run_command(Some(ctx), &argv).await?;
    image::load_from_memory_with_format(&out.stdout, image::ImageFormat::Png)
        .map(|i| i.to_rgb8())
        .map_err(|e| {
            ErrorInfo::new(
                ErrorCode::ExternalCommand,
                format!("ffmpeg returned no decodable frame at {t:.3} s: {e}"),
            )
        })
}

impl Stage for OrientStage {
    type Params = OrientParams;
    type Work = MediaProbe;
    type Output = Orientation;

    fn name(&self) -> &'static str {
        "orient"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: ORIENTATION,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![InputDecl {
            schema: MEDIA_PROBE,
            major: 1,
        }]
    }
    fn params(&self) -> &OrientParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            model_digest: self.model_digest.clone(),
            tool_versions: [("ffmpeg".to_string(), self.ffmpeg_version.clone())].into(),
            ..KeyExtras::default()
        }
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<MediaProbe>>, StageError> {
        let probe = inputs
            .read_ok::<MediaProbe>(MEDIA_PROBE)?
            .into_iter()
            .next()
            .ok_or_else(|| StageError::Invalid("media_probe has no ok item".into()))?
            .1;
        Ok(vec![WorkItem {
            id: "orientation".into(),
            work: probe,
        }])
    }
    async fn process(
        &self,
        ctx: &ItemContext,
        probe: MediaProbe,
    ) -> Result<Orientation, ErrorInfo> {
        if let Some(r) = self.params.override_rotation_deg {
            if !ROTATIONS.contains(&r) {
                return Err(ErrorInfo::new(
                    ErrorCode::InvalidInput,
                    format!("override_rotation_deg must be 0, 90, 180 or 270, got {r}"),
                ));
            }
            return Ok(Orientation {
                applied_rotation_deg: r,
                method: OrientMethod::Override,
                votes: RotationVotes::default(),
                samples: Vec::new(),
                container_rotation_deg: probe.container_rotation_deg,
                container_rotation_trusted: false,
                model: None,
                execution_provider: None,
            });
        }
        let entry = crate::models::entry(crate::models::ORIENT_MODEL)
            .map_err(|e| ErrorInfo::new(ErrorCode::Internal, e.to_string()))?;
        let (dir, entry2, dl) = (self.models_dir.clone(), entry.clone(), self.allow_download);
        let model_path = crate::util::on_rayon(move || {
            crate::models::ensure(&dir, &entry2, dl)
                .map_err(|e| ErrorInfo::new(ErrorCode::InvalidInput, e.to_string()))
        })
        .await?;
        let times = sample_times(probe.start_time_s, probe.end_s, self.params.sample_frames);
        let mut images = Vec::with_capacity(times.len());
        for &t in &times {
            let img = decode_sample(
                ctx,
                &self.ffmpeg,
                &probe.video_path,
                t,
                self.params.sample_short_side,
            )
            .await?;
            images.push((t, img));
        }
        let opts = ClassifyOptions {
            tile_short_side: self.params.sample_short_side,
            tile_min_conf: self.params.tile_min_confidence,
            min_conf: self.params.min_confidence,
        };
        let (samples, ep) =
            crate::util::on_rayon(move || classify_all(&model_path, &images, opts)).await?;
        let (rot, votes) = decide(&samples, &self.params)?;
        Ok(Orientation {
            applied_rotation_deg: rot,
            method: OrientMethod::PpLcnetVote,
            votes,
            samples,
            container_rotation_deg: probe.container_rotation_deg,
            container_rotation_trusted: false,
            model: Some(ModelRef {
                name: entry.name,
                sha256: entry.sha256,
            }),
            execution_provider: Some(ep),
        })
    }
}

/// Classification settings.
#[derive(Debug, Clone, Copy)]
pub struct ClassifyOptions {
    /// Short side the sample is resized to before tiling.
    pub tile_short_side: u32,
    /// Tiles below this top probability abstain.
    pub tile_min_conf: f64,
    /// Frames below this top probability do not vote.
    pub min_conf: f64,
}

/// Classifies images `(t_s, image)` with the model at `model`; returns samples and the
/// execution provider name.
#[cfg(any(feature = "onnx", feature = "onnx-dynamic"))]
pub fn classify_all(
    model: &std::path::Path,
    images: &[(f64, RgbImage)],
    opts: ClassifyOptions,
) -> Result<(Vec<OrientSample>, String), ErrorInfo> {
    use ort::session::Session;
    use ort::value::Tensor;
    let err = |e: &dyn std::fmt::Display| {
        ErrorInfo::new(ErrorCode::ModelRequest, format!("onnx runtime: {e}"))
    };
    let builder = Session::builder().map_err(|e| err(&e))?;
    #[cfg(feature = "cuda")]
    let builder = builder
        .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
        .map_err(|e| err(&e))?;
    let ep = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    };
    let mut builder = builder;
    let mut session = builder.commit_from_file(model).map_err(|e| err(&e))?;
    let mut samples = Vec::with_capacity(images.len());
    for (t, img) in images {
        let crops = preprocess_tiles(img, opts.tile_short_side);
        let n = crops.len();
        let mut data = Vec::with_capacity(n * crops.first().map_or(0, Vec::len));
        for c in &crops {
            data.extend_from_slice(c);
        }
        let input = Tensor::from_array(([n, 3, INPUT as usize, INPUT as usize], data))
            .map_err(|e| err(&e))?;
        let outputs = session.run(ort::inputs![input]).map_err(|e| err(&e))?;
        let (shape, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| err(&e))?;
        if shape.len() != 2 || shape[1] != 4 || values.len() != n * 4 {
            return Err(ErrorInfo::new(
                ErrorCode::ModelRequest,
                format!("unexpected orientation model output shape {shape:?}"),
            ));
        }
        let tiles: Vec<[f64; 4]> = values.chunks(4).map(to_probs).collect();
        let probs = aggregate_tiles(&tiles, opts.tile_min_conf);
        samples.push(sample_from_class_probs(*t, probs, opts.min_conf));
    }
    Ok((samples, ep.to_string()))
}

/// Without ONNX Runtime: always an error.
#[cfg(not(any(feature = "onnx", feature = "onnx-dynamic")))]
pub fn classify_all(
    _model: &std::path::Path,
    _images: &[(f64, RgbImage)],
    _opts: ClassifyOptions,
) -> Result<(Vec<OrientSample>, String), ErrorInfo> {
    Err(ErrorInfo::new(
        ErrorCode::InvalidInput,
        "this build has no ONNX Runtime (enable the `onnx`, `onnx-dynamic` or `cuda` feature) \
         and no orient.override_rotation_deg was given",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(deg: u32, confident: bool) -> OrientSample {
        OrientSample {
            t_s: 0.0,
            probs: [0.0; 4],
            predicted_deg: deg,
            confident,
        }
    }

    #[test]
    fn clear_majority_wins() {
        let v: Vec<_> = (0..10)
            .map(|i| s(if i < 8 { 0 } else { 90 }, true))
            .collect();
        let (r, votes) = decide(&v, &OrientParams::default()).unwrap();
        assert_eq!((r, votes.deg_0, votes.deg_90), (0, 8, 2));
    }

    #[test]
    fn ambiguous_margin_names_rotations() {
        let v = vec![
            s(90, true),
            s(90, true),
            s(90, true),
            s(270, true),
            s(270, true),
        ];
        let e = decide(&v, &OrientParams::default()).unwrap_err();
        assert!(e.message.contains("between 90 and 270"), "{e}");
    }

    #[test]
    fn unconfident_samples_do_not_vote() {
        let v: Vec<_> = (0..10).map(|_| s(180, false)).collect();
        assert!(decide(&v, &OrientParams::default()).is_err());
    }

    #[test]
    fn tiles_cover_the_resized_frame() {
        let img = RgbImage::from_pixel(1920, 1080, image::Rgb([255, 255, 255]));
        let t = preprocess_tiles(&img, 896);
        // 1593 x 896 -> 7 x 4 tiles.
        assert_eq!(t.len(), 28);
        assert_eq!(t[0].len(), 3 * 224 * 224);
        assert!((t[0][0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert_eq!(preprocess_tiles(&RgbImage::new(100, 50), 896).len(), 32);
    }

    #[test]
    fn blank_tiles_abstain() {
        let tiles = [
            [0.3, 0.3, 0.2, 0.2],
            [0.9, 0.05, 0.03, 0.02],
            [0.8, 0.1, 0.05, 0.05],
        ];
        let m = aggregate_tiles(&tiles, 0.6);
        assert!((m[0] - 0.85).abs() < 1e-9);
        let m = aggregate_tiles(&tiles[..1], 0.6);
        assert_eq!(m, tiles[0]);
    }

    #[test]
    fn probs_and_classes() {
        let p = to_probs(&[0.0, 0.0, 0.0, 10.0]);
        assert!(p[3] > 0.99);
        assert_eq!(to_probs(&[0.1, 0.2, 0.3, 0.4])[2], f64::from(0.3f32));
        // Class 0 means upright; 180 is its own inverse.
        assert_eq!(correction_for_class(0), 0);
        assert_eq!(correction_for_class(2), 180);
        let smp = sample_from_class_probs(1.0, [0.9, 0.05, 0.03, 0.02], 0.6);
        assert_eq!((smp.predicted_deg, smp.confident), (0, true));
    }

    #[test]
    fn times_are_slice_centers() {
        assert_eq!(sample_times(0.0, 10.0, 2), vec![2.5, 7.5]);
    }
}
