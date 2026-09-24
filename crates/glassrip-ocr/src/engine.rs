//! PP-OCRv5 on ONNX Runtime.

use std::path::Path;
use std::sync::Mutex;

use image::RgbImage;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;

use crate::{ctc, db, models, preprocess, OcrConfig, OcrError, RecognizedSpan, TextRecognizer};

fn rt(e: impl std::fmt::Display) -> OcrError {
    OcrError::Runtime(e.to_string())
}

/// Load ONNX Runtime from `ORT_DYLIB_PATH` once, returning an error instead of
/// letting `ort` panic on a missing library.
#[cfg(feature = "cuda")]
fn init_runtime() -> Result<(), OcrError> {
    use std::sync::OnceLock;
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let path = std::env::var_os("ORT_DYLIB_PATH")
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                "ORT_DYLIB_PATH is not set; point it at libonnxruntime.so from an ONNX Runtime \
                 GPU build for the host's CUDA major version"
                    .to_string()
            })?;
        let builder = ort::init_from(&path)
            .map_err(|e| format!("cannot load ONNX Runtime from {path:?}: {e}"))?;
        let _ = builder.commit();
        Ok(())
    })
    .clone()
    .map_err(OcrError::Runtime)
}

#[cfg(not(feature = "cuda"))]
fn init_runtime() -> Result<(), OcrError> {
    Ok(())
}

/// Execution provider this build registers.
pub fn execution_provider() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "CPU"
    }
}

fn session_builder(limit_mib: u32) -> Result<SessionBuilder, OcrError> {
    let builder = Session::builder()
        .map_err(rt)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(rt)?;
    // Without an explicit provider ONNX Runtime silently runs on CPU;
    // error_on_failure turns a broken CUDA install into an error.
    #[cfg(feature = "cuda")]
    let builder = builder
        .with_execution_providers([ort::ep::CUDA::default()
            .with_memory_limit(limit_mib as usize * 1024 * 1024)
            .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
            .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
            .build()
            .error_on_failure()])
        .map_err(|e| OcrError::Runtime(format!("CUDA execution provider: {e}")))?;
    #[cfg(not(feature = "cuda"))]
    let _ = limit_mib;
    Ok(builder)
}

struct Sessions {
    det: Session,
    rec: Session,
}

/// PP-OCRv5 detector plus recognizer. Sessions are guarded by a mutex, so one
/// engine serves one image at a time; share it with `Arc`.
pub struct PpOcrEngine {
    sessions: Mutex<Sessions>,
    dict: Vec<String>,
    cfg: OcrConfig,
    fingerprint: String,
}

impl PpOcrEngine {
    /// Load and hash-check the models in `dir`.
    pub fn new(dir: &Path, cfg: OcrConfig) -> Result<Self, OcrError> {
        let fingerprint = models::verify(dir)?;
        init_runtime()?;
        let (det_mib, rec_mib) = cfg.session_limits_mib();
        let det = session_builder(det_mib)?
            .commit_from_file(dir.join(models::DET_FILE))
            .map_err(|e| OcrError::Runtime(format!("detector: {e}")))?;
        let rec = session_builder(rec_mib)?
            .commit_from_file(dir.join(models::REC_FILE))
            .map_err(|e| OcrError::Runtime(format!("recognizer: {e}")))?;
        let dict = models::read_dict(dir)?;
        tracing::info!(
            provider = execution_provider(),
            dict = dict.len(),
            "PP-OCRv5 sessions ready"
        );
        Ok(Self {
            sessions: Mutex::new(Sessions { det, rec }),
            dict,
            cfg,
            fingerprint,
        })
    }

    fn detect(&self, s: &mut Sessions, image: &RgbImage) -> Result<Vec<db::DetBox>, OcrError> {
        let input = preprocess::det_input(image, self.cfg.det_max_side);
        let tensor = Tensor::from_array((
            [1usize, 3, input.height, input.width],
            input.data.into_boxed_slice(),
        ))
        .map_err(rt)?;
        let outputs = s.det.run(ort::inputs![tensor]).map_err(rt)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(rt)?;
        if shape.len() != 4 {
            return Err(OcrError::Output(format!(
                "detector output rank {}",
                shape.len()
            )));
        }
        let (mh, mw) = (shape[2].max(0) as usize, shape[3].max(0) as usize);
        let sx = input.scale_x * input.width as f64 / mw.max(1) as f64;
        let sy = input.scale_y * input.height as f64 / mh.max(1) as f64;
        Ok(db::boxes_from_map(
            data,
            mw,
            mh,
            sx,
            sy,
            image.width(),
            image.height(),
            &self.cfg,
        ))
    }

    fn recognize_batch(
        &self,
        s: &mut Sessions,
        crops: &[&RgbImage],
    ) -> Result<Vec<ctc::Decoded>, OcrError> {
        let (data, padded) =
            preprocess::rec_batch(crops, self.cfg.rec_height, self.cfg.rec_max_width);
        let tensor = Tensor::from_array((
            [crops.len(), 3, self.cfg.rec_height as usize, padded],
            data.into_boxed_slice(),
        ))
        .map_err(rt)?;
        let outputs = s.rec.run(ort::inputs![tensor]).map_err(rt)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(rt)?;
        if shape.len() != 3 || shape[0].max(0) as usize != crops.len() {
            return Err(OcrError::Output(format!(
                "recognizer output shape {shape:?}"
            )));
        }
        let (steps, classes) = (shape[1].max(0) as usize, shape[2].max(0) as usize);
        Ok((0..crops.len())
            .map(|n| {
                let seq = data
                    .get(n * steps * classes..(n + 1) * steps * classes)
                    .unwrap_or(&[]);
                ctc::decode(seq, steps, classes, &self.dict)
            })
            .collect())
    }
}

impl TextRecognizer for PpOcrEngine {
    fn recognize(&self, image: &RgbImage) -> Result<Vec<RecognizedSpan>, OcrError> {
        if image.width() == 0 || image.height() == 0 {
            return Err(OcrError::Input("empty image".into()));
        }
        let mut s = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let boxes = self.detect(&mut s, image)?;
        let mut crops: Vec<(usize, RgbImage)> = boxes
            .iter()
            .enumerate()
            .filter_map(|(i, b)| preprocess::crop(image, &b.bbox).map(|c| (i, c)))
            .collect();
        // Similar aspect ratios per batch keep padding small.
        crops.sort_by(|a, b| {
            let ra = f64::from(a.1.width()) / f64::from(a.1.height().max(1));
            let rb = f64::from(b.1.width()) / f64::from(b.1.height().max(1));
            ra.total_cmp(&rb)
        });
        let mut spans = Vec::new();
        for chunk in crops.chunks(self.cfg.rec_batch.max(1)) {
            let refs: Vec<&RgbImage> = chunk.iter().map(|(_, c)| c).collect();
            let decoded = self.recognize_batch(&mut s, &refs)?;
            for ((i, _), d) in chunk.iter().zip(decoded) {
                if d.text.is_empty() || d.confidence < self.cfg.drop_score {
                    continue;
                }
                spans.push(RecognizedSpan {
                    text: d.text,
                    bbox: boxes[*i].bbox,
                    confidence: d.confidence,
                    det_score: boxes[*i].score,
                });
            }
        }
        db::reading_order(&mut spans, |s| s.bbox);
        Ok(spans)
    }

    fn execution_provider(&self) -> String {
        execution_provider().to_string()
    }

    fn model_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}
