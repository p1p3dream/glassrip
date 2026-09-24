//! Speaker diarization.
//!
//! The backend is `speakrs` (a Rust port of the pyannote community-1 pipeline:
//! segmentation-3.0, WeSpeaker ResNet34 embeddings, PLDA, VBx) behind the
//! `diarize` feature. Everything after inference (re-clustering to a known
//! speaker count, exclusive turns, per-turn embedding similarity) is plain Rust
//! in [`finalize`] and is always compiled.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::error::Result;
#[cfg(not(feature = "diarize"))]
use crate::error::AudioError;
use crate::recluster::{
    ahc_to_k, centroids, cosine, exclusive_turns, merge_columns, normalize, talk_time,
    EmbeddingSample, Turn,
};

/// Inference device for the diarizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiarizeMode {
    /// ONNX Runtime on CPU.
    Cpu,
    /// ONNX Runtime CUDA, 1 s segmentation step.
    Cuda,
    /// ONNX Runtime CUDA, 2 s segmentation step.
    CudaFast,
}

impl DiarizeMode {
    /// Name used in artifact params.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CudaFast => "cuda-fast",
        }
    }
}

/// Diarization configuration.
#[derive(Debug, Clone)]
pub struct DiarizeConfig {
    /// Directory with the speakrs model bundle.
    pub models_dir: PathBuf,
    /// Inference device.
    pub mode: DiarizeMode,
    /// Known number of speakers; clusters are merged down to this count.
    pub num_speakers: Option<usize>,
}

/// Backend output before re-clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct RawDiarization {
    /// Frame-level binary activations, `frames x clusters`.
    pub activations: Vec<Vec<f32>>,
    /// Chunk-level embeddings with cluster ids.
    pub samples: Vec<EmbeddingSample>,
    /// Hop between activation frames, seconds.
    pub frame_step_s: f64,
    /// Length of an activation frame, seconds.
    pub frame_duration_s: f64,
    /// Hop between segmentation chunks, seconds.
    pub chunk_step_s: f64,
    /// Length of a segmentation chunk, seconds.
    pub chunk_window_s: f64,
}

/// Final diarization: exclusive turns and labels.
#[derive(Debug, Clone, PartialEq)]
pub struct Diarization {
    /// Exclusive turns in time order.
    pub turns: Vec<Turn>,
    /// Label per speaker index.
    pub labels: Vec<String>,
    /// Talk time per speaker index, seconds.
    pub talk_time_s: Vec<f64>,
    /// Clusters with speech before re-clustering.
    pub num_clusters_raw: usize,
    /// Seconds where the backend marked any speaker active.
    pub active_s: f64,
    /// Embedding centroid per speaker index (normalized), when known.
    pub centroids: Vec<Option<Vec<f32>>>,
}

/// Label for a speaker index.
pub fn speaker_label(i: usize) -> String {
    format!("SPEAKER_{i:02}")
}

/// Re-cluster to `num_speakers` (when fewer than the raw count) and build turns.
pub fn finalize(raw: &RawDiarization, num_speakers: Option<usize>) -> Diarization {
    let n_raw = raw.activations.iter().map(Vec::len).max().unwrap_or(0);
    let talk_raw = talk_time(&raw.activations, n_raw, raw.frame_step_s);
    let present = talk_raw.iter().filter(|t| **t > 0.0).count();
    let cents = centroids(&raw.samples, n_raw);

    let map: Vec<usize> = match num_speakers {
        Some(k) if k >= 1 && present > k => ahc_to_k(&cents, &talk_raw, k),
        _ => (0..n_raw).collect(),
    };
    let n_new = map.iter().copied().max().map_or(0, |m| m + 1);
    let merged = merge_columns(&raw.activations, &map, n_new);
    let (mut turns, order) = exclusive_turns(&merged, raw.frame_step_s, raw.frame_duration_s);

    // centroids of the merged clusters
    let mapped: Vec<EmbeddingSample> = raw
        .samples
        .iter()
        .filter_map(|s| {
            map.get(s.cluster).map(|&c| EmbeddingSample {
                chunk: s.chunk,
                cluster: c,
                embedding: s.embedding.clone(),
            })
        })
        .collect();
    let new_cents = centroids(&mapped, n_new);
    let mut by_chunk: BTreeMap<usize, Vec<(usize, Vec<f32>)>> = BTreeMap::new();
    for s in &mapped {
        if let Some(e) = normalize(&s.embedding) {
            by_chunk.entry(s.chunk).or_default().push((s.cluster, e));
        }
    }
    let step = raw.chunk_step_s.max(1e-6);
    for t in &mut turns {
        let Some(&col) = order.get(t.speaker) else {
            continue;
        };
        let Some(Some((centroid, _))) = new_cents.get(col) else {
            continue;
        };
        let first = ((t.start_s - raw.chunk_window_s) / step).floor().max(0.0) as usize;
        let last = (t.end_s / step).ceil().max(0.0) as usize;
        let mut acc: Option<Vec<f32>> = None;
        for (_, v) in by_chunk.range(first..=last) {
            for (c, e) in v {
                if *c != col {
                    continue;
                }
                match &mut acc {
                    Some(a) if a.len() == e.len() => {
                        for (x, y) in a.iter_mut().zip(e) {
                            *x += y;
                        }
                    }
                    Some(_) => {}
                    None => acc = Some(e.clone()),
                }
            }
        }
        t.embedding_sim = acc.and_then(|a| cosine(&a, centroid));
    }

    let mut talk = vec![0.0; order.len()];
    for t in &turns {
        if let Some(x) = talk.get_mut(t.speaker) {
            *x += t.end_s - t.start_s;
        }
    }
    let active_frames = raw
        .activations
        .iter()
        .filter(|row| row.iter().any(|v| *v > 0.5))
        .count();
    let speaker_centroids = order
        .iter()
        .map(|&col| new_cents.get(col).cloned().flatten().map(|(v, _)| v))
        .collect();
    Diarization {
        labels: (0..order.len()).map(speaker_label).collect(),
        talk_time_s: talk,
        turns,
        num_clusters_raw: present,
        active_s: active_frames as f64 * raw.frame_step_s,
        centroids: speaker_centroids,
    }
}

/// speakrs execution mode for a [`DiarizeMode`].
#[cfg(feature = "diarize")]
pub(crate) fn execution_mode(mode: DiarizeMode) -> speakrs::ExecutionMode {
    match mode {
        DiarizeMode::Cpu => speakrs::ExecutionMode::Cpu,
        DiarizeMode::Cuda => speakrs::ExecutionMode::Cuda,
        DiarizeMode::CudaFast => speakrs::ExecutionMode::CudaFast,
    }
}

/// Run the speakrs pipeline and return its raw output.
#[cfg(feature = "diarize")]
pub fn run_backend(samples: &[f32], cfg: &DiarizeConfig) -> Result<RawDiarization> {
    use crate::error::AudioError;
    use speakrs::pipeline::{FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS, SEGMENTATION_WINDOW_SECONDS};
    use speakrs::OwnedDiarizationPipeline;

    let err = |e: speakrs::PipelineError| AudioError::Diarization(e.to_string());
    let mode = execution_mode(cfg.mode);
    if !cfg.models_dir.is_dir() {
        return Err(AudioError::Model {
            name: cfg.models_dir.display().to_string(),
            message: "speakrs model directory not found".into(),
        });
    }
    let mut pipeline = OwnedDiarizationPipeline::from_dir(&cfg.models_dir, mode).map_err(err)?;
    let chunk_step_s = pipeline.segmentation_step();
    let result = pipeline.run(samples).map_err(err)?;

    let dd = &result.discrete_diarization;
    let (frames, cols) = dd.dim();
    let activations = (0..frames)
        .map(|f| (0..cols).map(|c| dd[[f, c]]).collect())
        .collect();

    let emb = &result.embeddings;
    let hc = &result.hard_clusters;
    let (n_chunks, n_spk, dim) = emb.dim();
    let (hc_chunks, hc_spk) = hc.dim();
    let mut samples_out = Vec::new();
    for chunk in 0..n_chunks.min(hc_chunks) {
        for spk in 0..n_spk.min(hc_spk) {
            let cluster = hc[[chunk, spk]];
            if cluster < 0 {
                continue;
            }
            let v: Vec<f32> = (0..dim).map(|d| emb[[chunk, spk, d]]).collect();
            if v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            samples_out.push(EmbeddingSample {
                chunk,
                cluster: usize::try_from(cluster).unwrap_or(0),
                embedding: v,
            });
        }
    }
    Ok(RawDiarization {
        activations,
        samples: samples_out,
        frame_step_s: FRAME_STEP_SECONDS,
        frame_duration_s: FRAME_DURATION_SECONDS,
        chunk_step_s,
        chunk_window_s: SEGMENTATION_WINDOW_SECONDS,
    })
}

/// Stub used when the crate is built without the `diarize` feature.
#[cfg(not(feature = "diarize"))]
pub fn run_backend(_samples: &[f32], _cfg: &DiarizeConfig) -> Result<RawDiarization> {
    Err(AudioError::FeatureDisabled("diarize"))
}

/// Diarize 16 kHz mono samples.
pub fn diarize(samples: &[f32], cfg: &DiarizeConfig) -> Result<Diarization> {
    let raw = run_backend(samples, cfg)?;
    Ok(finalize(&raw, cfg.num_speakers))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw() -> RawDiarization {
        // 3 raw clusters over 12 frames of 1 s: A(0..4), B(4..8), C(8..12) where
        // C sounds like A
        let mut acts = Vec::new();
        for f in 0..12 {
            let mut row = vec![0.0; 3];
            row[f / 4] = 1.0;
            acts.push(row);
        }
        let samples = vec![
            EmbeddingSample { chunk: 0, cluster: 0, embedding: vec![1.0, 0.0] },
            EmbeddingSample { chunk: 4, cluster: 1, embedding: vec![0.0, 1.0] },
            EmbeddingSample { chunk: 8, cluster: 2, embedding: vec![0.9, 0.1] },
        ];
        RawDiarization {
            activations: acts,
            samples,
            frame_step_s: 1.0,
            frame_duration_s: 0.0,
            chunk_step_s: 1.0,
            chunk_window_s: 2.0,
        }
    }

    #[test]
    fn without_k_keeps_all_clusters() {
        let d = finalize(&raw(), None);
        assert_eq!(d.labels.len(), 3);
        assert_eq!(d.num_clusters_raw, 3);
        assert_eq!(d.turns.len(), 3);
    }

    #[test]
    fn k_merges_similar_clusters() {
        let d = finalize(&raw(), Some(2));
        assert_eq!(d.labels, vec!["SPEAKER_00", "SPEAKER_01"]);
        let speakers: Vec<usize> = d.turns.iter().map(|t| t.speaker).collect();
        assert_eq!(speakers, vec![0, 1, 0]);
        assert!((d.talk_time_s[0] - 8.0).abs() < 1e-9);
        assert!(d.turns[0].embedding_sim.unwrap() > 0.9);
        assert_eq!(d.centroids.len(), 2);
        assert!(d.centroids.iter().all(Option::is_some));
    }

    #[test]
    fn k_larger_than_found_is_a_no_op() {
        let d = finalize(&raw(), Some(5));
        assert_eq!(d.labels.len(), 3);
    }

    #[test]
    #[cfg(not(feature = "diarize"))]
    fn backend_reports_missing_feature() {
        let cfg = DiarizeConfig {
            models_dir: PathBuf::from("/nonexistent"),
            mode: DiarizeMode::Cpu,
            num_speakers: None,
        };
        assert!(matches!(
            run_backend(&[], &cfg),
            Err(AudioError::FeatureDisabled("diarize"))
        ));
    }
}
