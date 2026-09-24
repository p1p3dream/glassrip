//! `classify`: screen type per keyframe (spec 6.7).
//!
//! Per keyframe: a schema-constrained model answer on a 768 px thumbnail,
//! combined in Rust with OCR keyword rules (`glassrip_vision::classify::combine`).
//! Rules see only text inside the shared area (so a chat window beside the
//! meeting does not make every frame `chat`).
//!
//! Temporal smoothing: keyframes are the segments, so any type switch already
//! falls on a keyframe boundary. On top of that, a keyframe whose answer is
//! weak (unknown without a contrary rule, or a model-only answer below
//! `smoothing_max_confidence`) and whose two neighbors agree takes the
//! neighbors' type. Smoothing never overrides a rule-backed answer and never
//! turns a keyframe with a rule for another type into a whiteboard.
//!
//! Each item needs its neighbors' answers, so per-keyframe results are memoized
//! and computed once, whichever item asks first.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, KeyExtras, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_vision::classify::{
    classify_options, classify_request_with, combine, ClassifyMethod, ClassifyRules,
    KeywordPattern, MatchMode, ScreenClass, ScreenClassOutput, ScreenType, SourceClassOutput,
    CLASSIFY_PROMPT,
};
use glassrip_vision::image_prep::THUMBNAIL_LONG_EDGE;
use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::artifacts::{
    self, ClassSource, ModelAnswer, OcrKeyframe, RectifiedKeyframeView, ScreenClassItem, TextRegion,
};
use crate::placement::{vision_error_info, PlacementMonitor};
use crate::raw_store::request_key;
use crate::stages::{input, invalid, load_rgb, resolve, run_root, text_hash};

/// Parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ClassifyParams {
    pub thumbnail_px: u32,
    pub seed: u64,
    pub rules: ClassifyRules,
    /// Model-only answers below this confidence may be smoothed.
    pub smoothing_max_confidence: f64,
}

impl Default for ClassifyParams {
    fn default() -> Self {
        Self {
            thumbnail_px: THUMBNAIL_LONG_EDGE,
            seed: 0,
            rules: meeting_rules(),
            smoothing_max_confidence: 0.8,
        }
    }
}

/// The spec's example rules with the content-studio rule widened for phone
/// footage: long words match fuzzily (OCR misreads), and the studio's own UI
/// strings count as evidence. Words that also appear on boards about the
/// studio (its product name, "content") need a second, UI-only match.
pub fn meeting_rules() -> ClassifyRules {
    let mut rules = ClassifyRules::spec_examples();
    // A bare "% " also starts whiteboard toolbar fragments ("% D"); keep the
    // prompts that do not occur in whiteboard UI.
    for r in &mut rules.rules {
        if r.screen_type == ScreenType::Code {
            r.patterns.retain(|p| p.text != "% ");
        }
    }
    for r in &mut rules.rules {
        if r.screen_type == ScreenType::Cms {
            r.patterns = r
                .patterns
                .iter()
                .map(|p| {
                    if p.text.chars().count() >= 5 && p.mode == MatchMode::Word {
                        KeywordPattern::fuzzy(&p.text)
                    } else {
                        p.clone()
                    }
                })
                .collect();
            for ui in ["Drafts", "Published", "Search list", "Translation metadata"] {
                r.patterns.push(KeywordPattern::fuzzy(ui));
            }
        }
    }
    rules
}

#[derive(Debug, Clone)]
struct Input {
    keyframe_id: String,
    image: PathBuf,
    ocr: Option<OcrKeyframe>,
}

#[derive(Debug, Clone)]
struct Raw {
    class: ScreenClass,
    model: Option<ModelAnswer>,
    model_error: Option<String>,
    rule_hits: Vec<glassrip_vision::classify::RuleHit>,
}

type Memo = HashMap<usize, Arc<tokio::sync::OnceCell<Result<Raw, ErrorInfo>>>>;

/// The stage.
pub struct ClassifyStage {
    params: ClassifyParams,
    monitor: Arc<PlacementMonitor>,
    model: String,
    digest: Option<String>,
    server_version: Option<String>,
    inputs: Mutex<Arc<Vec<Input>>>,
    memo: Mutex<Memo>,
}

impl ClassifyStage {
    /// `digest` and `server_version` go into the cache key (resolve them before running).
    pub fn new(
        params: ClassifyParams,
        monitor: Arc<PlacementMonitor>,
        model: impl Into<String>,
        digest: Option<String>,
        server_version: Option<String>,
    ) -> Self {
        Self {
            params,
            monitor,
            model: model.into(),
            digest,
            server_version,
            inputs: Mutex::new(Arc::new(Vec::new())),
            memo: Mutex::new(HashMap::new()),
        }
    }

    fn inputs(&self) -> Arc<Vec<Input>> {
        Arc::clone(&self.inputs.lock().unwrap_or_else(PoisonError::into_inner))
    }

    async fn raw(&self, idx: usize, cancel: &CancellationToken) -> Result<Raw, ErrorInfo> {
        let cell = {
            let mut memo = self.memo.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(memo.entry(idx).or_default())
        };
        cell.get_or_init(|| self.compute(idx, cancel.clone()))
            .await
            .clone()
    }

    async fn compute(&self, idx: usize, cancel: CancellationToken) -> Result<Raw, ErrorInfo> {
        let inputs = self.inputs();
        let input = inputs
            .get(idx)
            .ok_or_else(|| invalid(format!("no keyframe at index {idx}")))?;
        let rgb = load_rgb(input.image.clone()).await?;
        let frame = image::DynamicImage::ImageRgb8(rgb);
        let (request, prepared) = classify_request_with(
            &frame,
            self.params.thumbnail_px,
            classify_options(self.params.seed),
        )
        .map_err(|e| vision_error_info(&e))?;
        let key = request_key(&self.model, &request);
        let reply = self
            .monitor
            .infer_typed::<ScreenClassOutput>(request, cancel)
            .await;
        let (model, model_error): (Option<SourceClassOutput>, Option<String>) = match reply {
            Ok((out, _raw)) => (Some(out.in_source_coords(&prepared)), None),
            // Cancellation and aborts (digest change, lasting spill) stop the
            // item; other model failures fall back to the OCR rules alone.
            Err(e)
                if e.code == glassrip_core::envelope::ErrorCode::Cancelled
                    || self.monitor.abort_error().is_some() =>
            {
                return Err(e)
            }
            Err(e) => (None, Some(e.message)),
        };
        let spans = rule_spans(
            input.ocr.as_ref(),
            model.as_ref().map(|m| &m.answer.canvas_bbox),
        );
        let rules = self.params.rules.evaluate(&spans);
        let class = combine(model.as_ref(), &rules, &self.params.rules);
        Ok(Raw {
            class,
            model: model.map(|m| ModelAnswer {
                screen_type: m.answer.screen_type,
                app_hint: m.answer.app_hint.clone(),
                confidence: m.answer.confidence,
                canvas_bbox: m.answer.canvas_bbox,
                request_key: key,
            }),
            model_error,
            rule_hits: rules.hits,
        })
    }
}

fn inside(b: &BBox, s: &BBox) -> bool {
    let (x, y) = ((s.x1 + s.x2) / 2.0, (s.y1 + s.y2) / 2.0);
    x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2
}

/// OCR text the keyword rules may look at: inside the shared area when the
/// layout found one, else inside the model's content box, never chrome or tiles.
fn rule_spans(ocr: Option<&OcrKeyframe>, model_box: Option<&BBox>) -> Vec<String> {
    let Some(ocr) = ocr else {
        return Vec::new();
    };
    ocr.spans
        .iter()
        .filter(|s| matches!(s.region, TextRegion::Unassigned | TextRegion::Canvas))
        .filter(|s| ocr.share_area.is_some() || model_box.is_none_or(|b| inside(b, &s.bbox)))
        .map(|s| s.text.clone())
        .collect()
}

/// Smoothing decision for one keyframe given its neighbors.
pub fn smoothed_type(
    prev: Option<&ScreenClass>,
    cur: &ScreenClass,
    next: Option<&ScreenClass>,
    cur_rule_types: &[ScreenType],
    max_conf: f64,
) -> Option<ScreenType> {
    let (p, n) = (prev?, next?);
    let t = p.screen_type;
    if t != n.screen_type || t == ScreenType::Unknown || t == cur.screen_type {
        return None;
    }
    // A rule for another type blocks smoothing.
    if cur_rule_types.iter().any(|r| *r != t) {
        return None;
    }
    let weak_unknown = cur.screen_type == ScreenType::Unknown
        && matches!(
            cur.method,
            ClassifyMethod::LowConfidence | ClassifyMethod::Disagreement
        );
    let weak_model = cur.method == ClassifyMethod::Model && cur.confidence < max_conf;
    (weak_unknown || weak_model).then_some(t)
}

impl Stage for ClassifyStage {
    type Params = ClassifyParams;
    type Work = usize;
    type Output = ScreenClassItem;

    fn name(&self) -> &'static str {
        "classify"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: artifacts::SCREEN_CLASS,
            version: artifacts::output_version(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(artifacts::RECTIFIED_KEYFRAMES), input(artifacts::OCR)]
    }
    fn params(&self) -> &ClassifyParams {
        &self.params
    }
    fn key_extras(&self) -> KeyExtras {
        KeyExtras {
            model_digest: Some(format!(
                "{}@{}",
                self.model,
                self.digest.clone().unwrap_or_default()
            )),
            prompt_hash: Some(text_hash(CLASSIFY_PROMPT)),
            tool_versions: self
                .server_version
                .iter()
                .map(|v| ("ollama".to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
            ..KeyExtras::default()
        }
    }
    fn concurrency(&self) -> usize {
        self.monitor.client().max_in_flight()
    }
    fn item_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(600))
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<usize>>, StageError> {
        let root = run_root(inputs, artifacts::RECTIFIED_KEYFRAMES)?;
        let ocr: HashMap<String, OcrKeyframe> = inputs
            .read_ok::<OcrKeyframe>(artifacts::OCR)?
            .into_iter()
            .collect();
        let list: Vec<Input> = inputs
            .read_ok::<RectifiedKeyframeView>(artifacts::RECTIFIED_KEYFRAMES)?
            .into_iter()
            .map(|(_, k)| Input {
                ocr: ocr.get(&k.keyframe_id).cloned(),
                image: resolve(&root, &k.image_path),
                keyframe_id: k.keyframe_id,
            })
            .collect();
        let work = list
            .iter()
            .enumerate()
            .map(|(i, k)| WorkItem {
                id: k.keyframe_id.clone(),
                work: i,
            })
            .collect();
        *self.inputs.lock().unwrap_or_else(PoisonError::into_inner) = Arc::new(list);
        self.memo
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        Ok(work)
    }

    async fn process(&self, ctx: &ItemContext, idx: usize) -> Result<ScreenClassItem, ErrorInfo> {
        let cancel = ctx.cancel_token();
        let n = self.inputs().len();
        let cur = self.raw(idx, cancel).await?;
        // A neighbor that fails is treated as absent.
        let prev = match idx.checked_sub(1) {
            Some(i) => self.raw(i, cancel).await.ok(),
            None => None,
        };
        let next = if idx + 1 < n {
            self.raw(idx + 1, cancel).await.ok()
        } else {
            None
        };
        let rule_types: Vec<ScreenType> = cur.rule_hits.iter().map(|h| h.screen_type).collect();
        let smoothed = smoothed_type(
            prev.as_ref().map(|r| &r.class),
            &cur.class,
            next.as_ref().map(|r| &r.class),
            &rule_types,
            self.params.smoothing_max_confidence,
        );
        let c = &cur.class;
        let (screen_type, source, smoothed_from, confidence, canvas_bbox) = match smoothed {
            Some(t) => {
                let conf = prev
                    .iter()
                    .chain(next.iter())
                    .map(|r| r.class.confidence)
                    .fold(1.0, f64::min)
                    * 0.8;
                (t, ClassSource::Smoothed, Some(c.screen_type), conf, None)
            }
            None => (
                c.screen_type,
                ClassSource::Combined,
                None,
                c.confidence,
                c.canvas_bbox,
            ),
        };
        Ok(ScreenClassItem {
            keyframe_id: ctx.item_id().to_string(),
            screen_type,
            app_hint: c.app_hint.clone(),
            confidence,
            method: c.method,
            source,
            canvas_bbox,
            reads_board: screen_type == ScreenType::Whiteboard,
            model: cur.model.clone(),
            model_error: cur.model_error.clone(),
            rule_hits: cur.rule_hits.clone(),
            smoothed_from,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn class(t: ScreenType, method: ClassifyMethod, conf: f64) -> ScreenClass {
        ScreenClass {
            screen_type: t,
            app_hint: None,
            confidence: conf,
            method,
            canvas_bbox: None,
            canvas_bbox_issue: None,
            rule: None,
        }
    }

    #[test]
    fn meeting_rules_catch_misread_studio_ui() {
        let r = meeting_rules();
        let e = r.evaluate(&["Drafts", "Stuctue", "Article"]);
        assert_eq!(e.best().map(|h| h.screen_type), Some(ScreenType::Cms));
        // Toolbar fragments are not shell prompts.
        assert!(r.evaluate(&["% D", "size"]).best().is_none());
        assert_eq!(
            r.evaluate(&["$ cargo build"]).best().map(|h| h.screen_type),
            Some(ScreenType::Code)
        );
        // One studio word on a board is not enough.
        let e = r.evaluate(&["Relationships between content", "Content Manager"]);
        assert!(e.best().is_none());
    }

    #[test]
    fn smoothing_rules() {
        let wb = class(ScreenType::Whiteboard, ClassifyMethod::Model, 0.9);
        let cms = class(ScreenType::Cms, ClassifyMethod::Rule, 0.9);
        let unk = class(ScreenType::Unknown, ClassifyMethod::LowConfidence, 0.0);
        let weak_wb = class(ScreenType::Whiteboard, ClassifyMethod::Model, 0.6);
        // Unknown between two whiteboards becomes whiteboard.
        assert_eq!(
            smoothed_type(Some(&wb), &unk, Some(&wb), &[], 0.8),
            Some(ScreenType::Whiteboard)
        );
        // ...but not when a rule for another type fired.
        assert_eq!(
            smoothed_type(Some(&wb), &unk, Some(&wb), &[ScreenType::Cms], 0.8),
            None
        );
        // A weak model-only whiteboard between two CMS keyframes becomes CMS.
        assert_eq!(
            smoothed_type(Some(&cms), &weak_wb, Some(&cms), &[], 0.8),
            Some(ScreenType::Cms)
        );
        // A rule-decided answer is never smoothed; neighbors must agree.
        assert_eq!(
            smoothed_type(Some(&wb), &cms, Some(&wb), &[ScreenType::Cms], 0.8),
            None
        );
        assert_eq!(smoothed_type(Some(&wb), &unk, Some(&cms), &[], 0.8), None);
        assert_eq!(smoothed_type(None, &unk, Some(&wb), &[], 0.8), None);
    }
}
