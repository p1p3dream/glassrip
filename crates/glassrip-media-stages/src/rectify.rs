//! `rectify` (spec 6.4): per keyframe, a rectified representative image.
//!
//! - Frames of the run with a confident quad give a per-corner median quad. If fewer than
//!   `min_quad_share` of the run has one (a screen recording), the representative frame is
//!   passed through unchanged (`passthrough`).
//! - Otherwise up to `max_stack` frames (the representative first, the rest spread over the
//!   run) are warped with the median quad (`Projection::from_control_points`), aligned to
//!   the representative with the production aligner (phase-correlation-initialized ECC on a
//!   small gray copy), and combined by a per-pixel temporal median over valid samples
//!   (`warp_median`). A frame whose alignment fails is left out; with fewer than 2 frames
//!   left, the representative's warp alone is used (`warp_single`).
//!
//! Output images go to the blob store and are linked at `frames/keyframes/<id>.jpg`.

use std::collections::HashMap;
use std::path::PathBuf;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_media::plane::Plane;
use glassrip_media::production::{self, AlignParams};
use glassrip_media::warp::Affine;
use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{Interpolation, Projection, warp_into};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::Serialize;

use crate::blobs::BlobStore;
use crate::schema::{
    FRAMES, FrameRecord, KEYFRAMES, KEYFRAMES_DIR, Keyframe, Quad, RECTIFIED_KEYFRAMES,
    RectifiedKeyframe, RectifyMethod, SCREEN_QUADS, ScreenQuad, v1,
};
use crate::scoring::{Scorer, ScoringParams};

/// Rectification parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct RectifyParams {
    /// Quads below this confidence are ignored.
    pub min_quad_confidence: f64,
    /// Share of the run's frames that need a quad for warping.
    pub min_quad_share: f64,
    /// Frames in the temporal median at most.
    pub max_stack: u32,
    /// Output width cap (the quad's own size is used when smaller).
    pub max_width: u32,
    /// Width of the gray copy used for alignment.
    pub align_width: u32,
    /// JPEG quality of the representative.
    pub jpeg_quality: u8,
    /// Alignment settings (the production aligner is always used here).
    pub scoring: ScoringParams,
}

impl RectifyParams {
    /// Defaults with the given scoring settings.
    pub fn with_scoring(scoring: ScoringParams) -> Self {
        Self {
            min_quad_confidence: 0.5,
            min_quad_share: 0.5,
            max_stack: 9,
            max_width: 1920,
            align_width: 320,
            jpeg_quality: 92,
            scoring,
        }
    }
}

/// Per-keyframe work.
#[derive(Debug, Clone)]
pub struct RectifyWork {
    keyframe: Keyframe,
    frames: Vec<FrameRecord>,
    quads: Vec<Option<ScreenQuad>>,
}

/// The `rectify` stage.
#[derive(Debug, Clone)]
pub struct RectifyStage {
    params: RectifyParams,
    align: AlignParams,
    run_root: PathBuf,
    blobs: BlobStore,
}

impl RectifyStage {
    /// Stage reading frames under `run_root` and writing images into `blobs`.
    pub fn new(params: RectifyParams, run_root: PathBuf, blobs: BlobStore) -> Result<Self, String> {
        let mut prod = params.scoring.clone();
        prod.mode = glassrip_core::config::FeaturesMode::Production;
        // Validates the ECC settings.
        Scorer::new(&prod)?;
        let ecc = glassrip_media::ecc::EccParams::new(
            prod.ecc_iterations as usize,
            prod.ecc_eps,
            prod.ecc_gauss_filt_size as usize,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            align: AlignParams {
                ecc,
                min_phase_response: prod.min_phase_response,
                max_shift_frac: prod.max_shift_frac,
                max_linear_dev: prod.max_linear_dev,
            },
            params,
            run_root,
            blobs,
        })
    }
}

/// Per-corner median (numpy semantics) of several quads.
pub fn median_quad(quads: &[Quad]) -> Option<Quad> {
    if quads.is_empty() {
        return None;
    }
    let mut out = [[0.0; 2]; 4];
    for (c, corner) in out.iter_mut().enumerate() {
        for (d, v) in corner.iter_mut().enumerate() {
            let vals: Vec<f64> = quads.iter().map(|q| q[c][d]).collect();
            *v = glassrip_media::segment::median(&vals);
        }
    }
    Some(out)
}

/// Output size for a quad: mean opposite-edge lengths, capped at `max_width`, even.
pub fn target_size(q: &Quad, max_width: u32) -> (u32, u32) {
    let len = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]);
    let w = (len(q[0], q[1]) + len(q[3], q[2])) / 2.0;
    let h = (len(q[0], q[3]) + len(q[1], q[2])) / 2.0;
    let s = if w > f64::from(max_width) {
        f64::from(max_width) / w
    } else {
        1.0
    };
    let even = |v: f64| (((v * s) / 2.0).round() as u32 * 2).max(16);
    (even(w), even(h))
}

/// Up to `max` frame positions: the representative first, then evenly spread others.
pub fn stack_positions(n: usize, rep: usize, max: usize) -> Vec<usize> {
    let mut out = vec![rep];
    let others: Vec<usize> = (0..n).filter(|&i| i != rep).collect();
    let want = max.saturating_sub(1).min(others.len());
    for j in 0..want {
        let idx = (j * others.len() + others.len() / 2) / want.max(1);
        let pick = others[idx.min(others.len() - 1)];
        if !out.contains(&pick) {
            out.push(pick);
        }
    }
    out
}

fn gray_small(img: &RgbImage, width: u32) -> Plane<f32> {
    let (w, h) = img.dimensions();
    let sw = width.clamp(16, w.max(16));
    let sh = ((u64::from(h) * u64::from(sw) / u64::from(w.max(1))) as u32).max(16);
    let g = image::imageops::grayscale(img);
    let s = image::imageops::resize(&g, sw, sh, image::imageops::FilterType::Triangle);
    Plane {
        width: sw as usize,
        height: sh as usize,
        data: s.into_raw().into_iter().map(f32::from).collect(),
    }
}

/// Warped pixels and their validity mask.
type Layer = (Vec<[u8; 3]>, Vec<bool>);

/// Inverse-map bilinear warp of an RGB image; returns pixels and a validity mask.
fn warp_rgb(src: &RgbImage, m: &Affine) -> Layer {
    let (w, h) = src.dimensions();
    let (wf, hf) = ((w - 1) as f32, (h - 1) as f32);
    let n = (w * h) as usize;
    let mut px = vec![[0u8; 3]; n];
    let mut ok = vec![false; n];
    px.par_chunks_mut(w as usize)
        .zip(ok.par_chunks_mut(w as usize))
        .enumerate()
        .for_each(|(y, (prow, orow))| {
            let yf = y as f32;
            for x in 0..w as usize {
                let xf = x as f32;
                let sx = m[0] * xf + m[1] * yf + m[2];
                let sy = m[3] * xf + m[4] * yf + m[5];
                if !(0.0..=wf).contains(&sx) || !(0.0..=hf).contains(&sy) {
                    continue;
                }
                let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let (ax, ay) = (sx - x0 as f32, sy - y0 as f32);
                let (p00, p01, p10, p11) = (
                    src.get_pixel(x0, y0),
                    src.get_pixel(x1, y0),
                    src.get_pixel(x0, y1),
                    src.get_pixel(x1, y1),
                );
                let mut v = [0u8; 3];
                for (c, vc) in v.iter_mut().enumerate() {
                    let top = f32::from(p00[c]) * (1.0 - ax) + f32::from(p01[c]) * ax;
                    let bot = f32::from(p10[c]) * (1.0 - ax) + f32::from(p11[c]) * ax;
                    *vc = (top * (1.0 - ay) + bot * ay).round().clamp(0.0, 255.0) as u8;
                }
                prow[x] = v;
                orow[x] = true;
            }
        });
    (px, ok)
}

fn temporal_median(layers: &[Layer], w: u32, h: u32) -> RgbImage {
    let n = (w * h) as usize;
    let mut out = vec![0u8; 3 * n];
    out.par_chunks_mut(3).enumerate().for_each(|(i, o)| {
        let mut buf: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (px, ok) in layers {
            if ok[i] {
                for c in 0..3 {
                    buf[c].push(px[i][c]);
                }
            }
        }
        for c in 0..3 {
            let b = &mut buf[c];
            if b.is_empty() {
                continue;
            }
            b.sort_unstable();
            let m = b.len();
            o[c] = if m % 2 == 1 {
                b[m / 2]
            } else {
                (u16::from(b[m / 2 - 1]) + u16::from(b[m / 2])).div_ceil(2) as u8
            };
        }
    });
    RgbImage::from_raw(w, h, out).unwrap_or_else(|| RgbImage::new(w, h))
}

fn encode_jpeg(img: &RgbImage, q: u8) -> Result<Vec<u8>, ErrorInfo> {
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, q)
        .encode_image(img)
        .map_err(|e| ErrorInfo::new(ErrorCode::Internal, format!("JPEG encode failed: {e}")))?;
    Ok(buf)
}

impl RectifyStage {
    fn run(&self, w: RectifyWork) -> Result<RectifiedKeyframe, ErrorInfo> {
        let kf = &w.keyframe;
        let rel = format!("{KEYFRAMES_DIR}/{}.jpg", kf.keyframe_id);
        let n = w.frames.len();
        let rep = w
            .frames
            .iter()
            .position(|f| f.frame_id == kf.rep_frame_id)
            .ok_or_else(|| {
                ErrorInfo::new(ErrorCode::InvalidInput, "representative frame not in run")
            })?;
        let quads: Vec<Quad> = w
            .quads
            .iter()
            .flatten()
            .filter(|q| q.confidence >= self.params.min_quad_confidence)
            .filter_map(|q| q.quad)
            .collect();
        let need = ((self.params.min_quad_share * n as f64).ceil() as usize).max(1);
        let mq = if quads.len() >= need {
            median_quad(&quads)
        } else {
            None
        };
        let materialize = |hash: &str| {
            self.blobs
                .materialize(&self.run_root, &rel, hash, "rectify")
                .map_err(|e| ErrorInfo::new(ErrorCode::Io, e.to_string()))
        };
        let rep_frame = &w.frames[rep];
        let passthrough = |median: Option<Quad>| -> Result<RectifiedKeyframe, ErrorInfo> {
            let src = self.run_root.join(&rep_frame.path);
            let bytes =
                fs_err::read(&src).map_err(|e| crate::util::io_error("cannot read", &src, e))?;
            let hash = self
                .blobs
                .put_bytes(&bytes, "jpg")
                .map_err(|e| ErrorInfo::new(ErrorCode::Io, e.to_string()))?;
            materialize(&hash)?;
            Ok(RectifiedKeyframe {
                keyframe_id: kf.keyframe_id.clone(),
                rep_frame_id: kf.rep_frame_id.clone(),
                path: rel.clone(),
                blake3: hash,
                width: rep_frame.width,
                height: rep_frame.height,
                median_quad: median,
                n_frames_with_quad: quads.len() as u32,
                n_frames_stacked: 1,
                stacked_frame_ids: vec![rep_frame.frame_id.clone()],
                method: RectifyMethod::Passthrough,
            })
        };
        let Some(q) = mq else {
            return passthrough(None);
        };
        let (tw, th) = target_size(&q, self.params.max_width);
        let to_f = |p: [f64; 2]| (p[0] as f32, p[1] as f32);
        let dst = [
            (0.0, 0.0),
            (tw as f32, 0.0),
            (tw as f32, th as f32),
            (0.0, th as f32),
        ];
        let Some(proj) =
            Projection::from_control_points([to_f(q[0]), to_f(q[1]), to_f(q[2]), to_f(q[3])], dst)
        else {
            return passthrough(Some(q));
        };
        let positions = stack_positions(n, rep, self.params.max_stack.max(1) as usize);
        let warped: Vec<Result<RgbImage, ErrorInfo>> = positions
            .par_iter()
            .map(|&i| {
                let p = self.run_root.join(&w.frames[i].path);
                let img = image::open(&p)
                    .map_err(|e| {
                        ErrorInfo::new(
                            ErrorCode::InvalidInput,
                            format!("cannot read {}: {e}", p.display()),
                        )
                    })?
                    .to_rgb8();
                let mut out = RgbImage::new(tw, th);
                warp_into(
                    &img,
                    &proj,
                    Interpolation::Bilinear,
                    Rgb([0, 0, 0]),
                    &mut out,
                );
                Ok(out)
            })
            .collect();
        let mut imgs = Vec::with_capacity(warped.len());
        for r in warped {
            imgs.push(r?);
        }
        let reference = gray_small(&imgs[0], self.params.align_width);
        let (sx, sy) = (
            tw as f32 / reference.width as f32,
            th as f32 / reference.height as f32,
        );
        let aligned: Vec<Option<Layer>> = imgs
            .par_iter()
            .enumerate()
            .map(|(j, img)| {
                if j == 0 {
                    let px: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
                    let ok = vec![true; px.len()];
                    return Some((px, ok));
                }
                let a = production::align(
                    &reference,
                    &gray_small(img, self.params.align_width),
                    &self.align,
                );
                if !a.ok() {
                    return None;
                }
                let m = a.warp;
                let full = [
                    m[0],
                    m[1] * sx / sy,
                    m[2] * sx,
                    m[3] * sy / sx,
                    m[4],
                    m[5] * sy,
                ];
                Some(warp_rgb(img, &full))
            })
            .collect();
        let mut ids = Vec::new();
        let mut layers = Vec::new();
        for (j, a) in aligned.into_iter().enumerate() {
            if let Some(l) = a {
                ids.push(w.frames[positions[j]].frame_id.clone());
                layers.push(l);
            }
        }
        let (img, method) = if layers.len() >= 2 {
            (temporal_median(&layers, tw, th), RectifyMethod::WarpMedian)
        } else {
            ids.truncate(1);
            (imgs.swap_remove(0), RectifyMethod::WarpSingle)
        };
        let bytes = encode_jpeg(&img, self.params.jpeg_quality)?;
        let hash = self
            .blobs
            .put_bytes(&bytes, "jpg")
            .map_err(|e| ErrorInfo::new(ErrorCode::Io, e.to_string()))?;
        materialize(&hash)?;
        Ok(RectifiedKeyframe {
            keyframe_id: kf.keyframe_id.clone(),
            rep_frame_id: kf.rep_frame_id.clone(),
            path: rel,
            blake3: hash,
            width: tw,
            height: th,
            median_quad: Some(q),
            n_frames_with_quad: quads.len() as u32,
            n_frames_stacked: ids.len() as u32,
            stacked_frame_ids: ids,
            method,
        })
    }
}

impl Stage for RectifyStage {
    type Params = RectifyParams;
    type Work = RectifyWork;
    type Output = RectifiedKeyframe;

    fn name(&self) -> &'static str {
        "rectify"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: RECTIFIED_KEYFRAMES,
            version: v1(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: KEYFRAMES,
                major: 1,
            },
            InputDecl {
                schema: FRAMES,
                major: 1,
            },
            InputDecl {
                schema: SCREEN_QUADS,
                major: 1,
            },
        ]
    }
    fn params(&self) -> &RectifyParams {
        &self.params
    }
    fn concurrency(&self) -> usize {
        (crate::util::cpus() / 2).max(1)
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<RectifyWork>>, StageError> {
        let frames: HashMap<String, FrameRecord> =
            inputs.read_ok::<FrameRecord>(FRAMES)?.into_iter().collect();
        let quads: HashMap<String, ScreenQuad> = inputs
            .read_ok::<ScreenQuad>(SCREEN_QUADS)?
            .into_iter()
            .collect();
        let mut kfs: Vec<Keyframe> = inputs
            .read_ok::<Keyframe>(KEYFRAMES)?
            .into_iter()
            .map(|(_, k)| k)
            .collect();
        kfs.sort_by_key(|k| k.index);
        kfs.into_iter()
            .map(|k| {
                let fr: Result<Vec<FrameRecord>, StageError> = k
                    .frame_ids
                    .iter()
                    .map(|id| {
                        frames.get(id).cloned().ok_or_else(|| {
                            StageError::Invalid(format!(
                                "keyframe {} names unknown frame {id}",
                                k.keyframe_id
                            ))
                        })
                    })
                    .collect();
                let q = k
                    .frame_ids
                    .iter()
                    .map(|id| quads.get(id).cloned())
                    .collect();
                Ok(WorkItem {
                    id: k.keyframe_id.clone(),
                    work: RectifyWork {
                        frames: fr?,
                        quads: q,
                        keyframe: k,
                    },
                })
            })
            .collect()
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        w: RectifyWork,
    ) -> Result<RectifiedKeyframe, ErrorInfo> {
        let this = self.clone();
        crate::util::on_rayon(move || this.run(w)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_size() {
        let a = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        let b = [[2.0, 2.0], [102.0, 2.0], [102.0, 52.0], [2.0, 52.0]];
        let c = [[1.0, 1.0], [101.0, 1.0], [101.0, 51.0], [1.0, 51.0]];
        let m = median_quad(&[a, b, c]).unwrap();
        assert_eq!(m, c);
        assert_eq!(target_size(&m, 1920), (100, 50));
        assert_eq!(
            target_size(
                &[[0.0, 0.0], [4000.0, 0.0], [4000.0, 2000.0], [0.0, 2000.0]],
                1920
            ),
            (1920, 960)
        );
    }

    #[test]
    fn stack_starts_with_rep_and_spreads() {
        let s = stack_positions(20, 7, 5);
        assert_eq!(s[0], 7);
        assert_eq!(s.len(), 5);
        assert_eq!(stack_positions(1, 0, 9), vec![0]);
        assert_eq!(stack_positions(3, 1, 9).len(), 3);
    }

    #[test]
    fn median_ignores_invalid_samples() {
        let layer = |v: u8, ok: bool| (vec![[v; 3]; 4], vec![ok; 4]);
        let img = temporal_median(
            &[
                layer(10, true),
                layer(200, false),
                layer(30, true),
                layer(20, true),
            ],
            2,
            2,
        );
        assert_eq!(img.get_pixel(0, 0).0, [20, 20, 20]);
    }
}
