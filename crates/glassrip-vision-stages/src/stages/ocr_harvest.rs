//! `ocr_harvest`: PP-OCRv5 on every rectified keyframe (spec 6.8).
//!
//! Output spans carry a preliminary region from layout analysis: tile labels
//! and tile content are `tile`; banners, denylisted UI text, text outside the
//! shared area, and whiteboard panels are `chrome`; the rest is `unassigned`
//! until `canvas_crop` decides the canvas.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_ocr::{OcrConfig, TextRecognizer};
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::Serialize;

use crate::artifacts::{self, OcrKeyframe, OcrSpan, RectifiedKeyframeView};
use crate::layout::{self, LayoutConfig, Span};
use crate::stages::{input, internal, invalid, resolve, run_root};

/// Parameters (recorded in the envelope and the cache key).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct OcrHarvestParams {
    pub ocr: OcrConfig,
    pub layout: LayoutConfig,
}

/// The stage. `ocr` must be the configuration the recognizer was built with.
pub struct OcrHarvestStage {
    params: OcrHarvestParams,
    recognizer: Arc<dyn TextRecognizer>,
}

impl OcrHarvestStage {
    pub fn new(recognizer: Arc<dyn TextRecognizer>, ocr: OcrConfig, layout: LayoutConfig) -> Self {
        Self {
            params: OcrHarvestParams { ocr, layout },
            recognizer,
        }
    }
}

/// Turn recognized spans plus layout into the artifact item.
pub fn build_item(
    keyframe_id: &str,
    width: u32,
    height: u32,
    provider: String,
    raw: Vec<glassrip_ocr::RecognizedSpan>,
    cfg: &LayoutConfig,
) -> OcrKeyframe {
    let spans: Vec<Span> = raw
        .into_iter()
        .map(|s| Span {
            text: s.text,
            bbox: BBox::new(s.bbox.x1, s.bbox.y1, s.bbox.x2, s.bbox.y2),
            confidence: s.confidence,
        })
        .collect();
    let l = layout::analyze(&spans, f64::from(width), f64::from(height), cfg);
    let out = spans
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (region, chrome_reason) = layout::region_of(&l, i, s, None);
            OcrSpan {
                text: s.text.clone(),
                bbox: s.bbox,
                confidence: s.confidence,
                region,
                chrome_reason,
            }
        })
        .collect();
    OcrKeyframe {
        keyframe_id: keyframe_id.to_string(),
        image_width: width,
        image_height: height,
        execution_provider: provider,
        spans: out,
        tile_names: l.names,
        share_area: l.share_area,
    }
}

impl Stage for OcrHarvestStage {
    type Params = OcrHarvestParams;
    type Work = PathBuf;
    type Output = OcrKeyframe;

    fn name(&self) -> &'static str {
        "ocr_harvest"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: artifacts::OCR,
            version: artifacts::output_version(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(artifacts::RECTIFIED_KEYFRAMES)]
    }
    fn params(&self) -> &OcrHarvestParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            model_digest: Some(self.recognizer.model_fingerprint()),
            tool_versions: BTreeMap::from([(
                "ocr_execution_provider".to_string(),
                self.recognizer.execution_provider(),
            )]),
            ..KeyExtras::default()
        }
    }
    fn concurrency(&self) -> usize {
        // One engine serializes GPU work; a second item overlaps image decode.
        2
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(120))
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<PathBuf>>, StageError> {
        let root = run_root(inputs, artifacts::RECTIFIED_KEYFRAMES)?;
        Ok(inputs
            .read_ok::<RectifiedKeyframeView>(artifacts::RECTIFIED_KEYFRAMES)?
            .into_iter()
            .map(|(_, k)| WorkItem {
                id: k.keyframe_id.clone(),
                work: resolve(&root, &k.image_path),
            })
            .collect())
    }

    async fn process(&self, ctx: &ItemContext, path: PathBuf) -> Result<OcrKeyframe, ErrorInfo> {
        let recognizer = Arc::clone(&self.recognizer);
        let cfg = self.params.layout.clone();
        let id = ctx.item_id().to_string();
        tokio::task::spawn_blocking(move || {
            let img = image::open(&path)
                .map_err(|e| invalid(format!("cannot read image {}: {e}", path.display())))?
                .to_rgb8();
            let raw = recognizer
                .recognize(&img)
                .map_err(|e| internal(format!("OCR failed: {e}")))?;
            Ok(build_item(
                &id,
                img.width(),
                img.height(),
                recognizer.execution_provider(),
                raw,
                &cfg,
            ))
        })
        .await
        .map_err(|e| internal(format!("OCR task failed: {e}")))?
    }
}
