//! Build the upstream artifacts these stages read from a directory of
//! prototype keyframe JPEGs and their index file, so the vision branch can run
//! before the media stages exist.
//!
//! The index is a JSON object with `keyframes: [{file, t_start, t_end, t_rep}]`
//! (`file` relative to the index's directory, named `t_NNNNNN.jpg`). Optional
//! sampled frames (`t_NNNNNN.jpg`, NNNNNN = seconds) feed the variance tiebreak.
//! Images are not rectified: every screen quad is `None` and every rectified
//! keyframe is a `passthrough` of its representative image.

use std::path::{Path, PathBuf};

use glassrip_core::envelope::{content_hash, EnvelopeHeader, Outcome, Producer, Record};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::artifacts::{self, FrameView, KeyframeView, RectifiedKeyframeView, ScreenQuadView};

#[derive(Debug, Deserialize)]
struct IndexEntry {
    file: String,
    t_start: f64,
    t_end: f64,
    t_rep: f64,
    #[serde(default)]
    n_merged: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct Index {
    keyframes: Vec<IndexEntry>,
}

/// Adapter failure.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("{path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("{0}")]
    Invalid(String),
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> AdapterError + '_ {
    move |e| AdapterError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    }
}

/// What was written.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdapterSummary {
    pub keyframes: usize,
    pub frames: usize,
}

fn stem_seconds(path: &Path) -> Option<f64> {
    let stem = path.file_stem()?.to_str()?;
    stem.strip_prefix("t_")?
        .parse::<u64>()
        .ok()
        .map(|s| s as f64)
}

fn write<T: Serialize + DeserializeOwned>(
    root: &Path,
    schema: &str,
    run_id: &str,
    params: serde_json::Value,
    items: Vec<(String, T)>,
) -> Result<(), AdapterError> {
    let records: Vec<Record<T>> = items
        .into_iter()
        .map(|(id, v)| Record {
            id,
            outcome: Outcome::ok(v),
        })
        .collect();
    let version = artifacts::output_version();
    let hash = content_hash(schema, &version, &params, &records)
        .map_err(|e| AdapterError::Invalid(e.to_string()))?;
    let header = EnvelopeHeader {
        schema: schema.to_string(),
        schema_version: version,
        run_id: run_id.to_string(),
        producer: Producer::glassrip(env!("CARGO_PKG_VERSION"), None),
        inputs: Vec::new(),
        params,
        content_hash: Some(hash),
        restored_from: None,
    };
    let path = root
        .join(glassrip_core::manifest::ARTIFACTS_DIR)
        .join(format!("{schema}.jsonl"));
    glassrip_core::jsonl::write_atomic(&path, &header, &records)
        .map_err(|e| AdapterError::Invalid(e.to_string()))
}

/// Keyframe id for a representative time.
pub fn keyframe_id(t_rep_s: f64) -> String {
    format!("kf_{:06}", t_rep_s.round() as i64)
}

/// Write `glassrip.frames`, `glassrip.screen_quads`, `glassrip.keyframes`, and
/// `glassrip.rectified_keyframes` into `run_root/artifacts/`.
pub fn build_inputs(
    run_root: &Path,
    index_path: &Path,
    frames_dir: Option<&Path>,
    run_id: &str,
) -> Result<AdapterSummary, AdapterError> {
    let bytes = fs_err::read(index_path).map_err(io(index_path))?;
    let index: Index = serde_json::from_slice(&bytes)
        .map_err(|e| AdapterError::Invalid(format!("{}: {e}", index_path.display())))?;
    let base = index_path.parent().unwrap_or(Path::new("."));
    let mut keyframes = Vec::new();
    let mut rectified = Vec::new();
    let mut kf_frames = Vec::new();
    for e in &index.keyframes {
        let path = base.join(&e.file);
        let abs = if path.is_absolute() {
            path
        } else {
            std::env::current_dir().map_err(io(base))?.join(path)
        };
        let blake3 = glassrip_core::blake3_file(&abs).map_err(io(&abs))?;
        let frame_id = abs
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| AdapterError::Invalid(format!("bad file name {}", e.file)))?
            .to_string();
        let id = keyframe_id(e.t_rep);
        keyframes.push((
            id.clone(),
            KeyframeView {
                keyframe_id: id.clone(),
                rep_frame_id: frame_id.clone(),
                t_start_s: e.t_start,
                t_end_s: e.t_end,
                t_rep_s: e.t_rep,
                n_frames: e.n_merged,
            },
        ));
        rectified.push((
            id.clone(),
            RectifiedKeyframeView {
                keyframe_id: id,
                image_path: abs.display().to_string(),
                image_blake3: Some(blake3.clone()),
                median_quad: None,
                n_frames_stacked: Some(1),
                method: Some("passthrough".into()),
            },
        ));
        kf_frames.push(FrameView {
            frame_id,
            pts_s: e.t_rep,
            path: abs.display().to_string(),
            blake3: Some(blake3),
        });
    }
    if keyframes.is_empty() {
        return Err(AdapterError::Invalid("index lists no keyframes".into()));
    }
    let frames: Vec<FrameView> = match frames_dir {
        Some(dir) => {
            let mut v = Vec::new();
            for entry in fs_err::read_dir(dir).map_err(io(dir))? {
                let p = entry.map_err(io(dir))?.path();
                if p.extension().and_then(|e| e.to_str()) != Some("jpg") {
                    continue;
                }
                let Some(t) = stem_seconds(&p) else {
                    continue;
                };
                let frame_id = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                v.push(FrameView {
                    frame_id,
                    pts_s: t,
                    path: p.display().to_string(),
                    blake3: None,
                });
            }
            v.sort_by(|a, b| a.pts_s.total_cmp(&b.pts_s));
            v
        }
        None => kf_frames,
    };
    fs_err::create_dir_all(run_root.join(glassrip_core::manifest::ARTIFACTS_DIR))
        .map_err(io(run_root))?;
    let params = json!({"adapter": "prototype_keyframes", "rectified": false});
    let n_frames = frames.len();
    write(
        run_root,
        artifacts::SCREEN_QUADS,
        run_id,
        params.clone(),
        frames
            .iter()
            .map(|f| {
                (
                    f.frame_id.clone(),
                    ScreenQuadView {
                        frame_id: f.frame_id.clone(),
                        quad: None,
                        confidence: None,
                    },
                )
            })
            .collect(),
    )?;
    write(
        run_root,
        artifacts::FRAMES,
        run_id,
        params.clone(),
        frames
            .into_iter()
            .map(|f| (f.frame_id.clone(), f))
            .collect(),
    )?;
    let n = keyframes.len();
    write(
        run_root,
        artifacts::KEYFRAMES,
        run_id,
        params.clone(),
        keyframes,
    )?;
    write(
        run_root,
        artifacts::RECTIFIED_KEYFRAMES,
        run_id,
        params,
        rectified,
    )?;
    Ok(AdapterSummary {
        keyframes: n,
        frames: n_frames,
    })
}
