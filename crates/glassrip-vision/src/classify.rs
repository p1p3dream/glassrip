//! Screen classification: a cheap model pass on a thumbnail, combined in Rust
//! with OCR keyword rules.
//!
//! Disagreement policy:
//! - A rule that fires with high confidence wins over the model.
//! - A rule that fires below that confidence (or ambiguously) and disagrees with
//!   the model yields `unknown`.
//! - With no rule, the model answer stands if its confidence reaches the
//!   configured minimum; otherwise `unknown`.
//!
//! Only `whiteboard` results proceed to board reading.

use image::DynamicImage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::backend::{GenerationOptions, VisionRequest};
use crate::error::Result;
use crate::geometry::BBox;
use crate::image_prep::{prepare_thumbnail, PreparedImage};

/// Closed set of screen types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScreenType {
    Whiteboard,
    MeetGallery,
    Slides,
    Code,
    Cms,
    Chat,
    Web,
    Desktop,
    Unknown,
}

/// Model output for the classification request (coordinates in thumbnail pixels).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenClassOutput {
    pub screen_type: ScreenType,
    /// Application name if recognizable (for example "Miro"), otherwise an empty string.
    pub app_hint: String,
    /// Main content area (the whiteboard canvas for `whiteboard`), in image pixels.
    pub canvas_bbox: BBox,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub confidence: f64,
}

impl ScreenClassOutput {
    /// Map the canvas box from thumbnail pixels to source frame pixels, clamped to the frame.
    ///
    /// The model sometimes reports coordinates past the image edge, so the result is clamped.
    pub fn in_source_coords(mut self, prepared: &PreparedImage) -> Self {
        self.canvas_bbox = self.canvas_bbox.to_source(prepared).clamped(
            f64::from(prepared.source_width),
            f64::from(prepared.source_height),
        );
        self
    }
}

/// Classification prompt; the schema text is appended by [`VisionRequest::for_output`].
pub const CLASSIFY_PROMPT: &str = "\
This image is a frame from a recording of a computer screen during a video meeting. \
Classify what the screen mainly shows.

screen_type values:
- whiteboard: a diagramming or whiteboard canvas (for example Miro) with boxes, arrows, or sticky notes
- meet_gallery: a video call view dominated by participant video tiles, or a call status screen
- slides: a presentation slide
- code: a code editor or a terminal
- cms: a content management studio (for example Sanity Studio)
- chat: a chat application (for example Slack)
- web: any other web page
- desktop: an operating system desktop, file browser, or settings
- unknown: none of the above, or unreadable

app_hint: the application name if you can recognize it, otherwise an empty string.
canvas_bbox: the main content area in pixel coordinates of this image (x1, y1 top-left; x2, y2 bottom-right). \
For a whiteboard, exclude toolbars, side panels, zoom controls, and participant video tiles.
confidence: a number from 0 to 1.";

/// Build the classification request on a 768 px thumbnail of `frame`.
pub fn classify_request(
    frame: &DynamicImage,
    options: GenerationOptions,
) -> Result<(VisionRequest, PreparedImage)> {
    let prepared = prepare_thumbnail(frame)?;
    let request = VisionRequest::for_output::<ScreenClassOutput>(
        CLASSIFY_PROMPT,
        prepared.image.clone(),
        options,
    )?;
    Ok((request, prepared))
}

/// Options suited to the tiny classification output.
pub fn classify_options(seed: u64) -> GenerationOptions {
    GenerationOptions {
        seed,
        num_predict: 256,
    }
}

/// How a keyword pattern is matched against one OCR span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MatchMode {
    /// Case-insensitive whole-word sequence (punctuation ignored).
    Word,
    /// Case-insensitive substring.
    Phrase,
    /// Span (leading whitespace trimmed) starts with the pattern, case-sensitive.
    Prefix,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeywordPattern {
    pub text: String,
    pub mode: MatchMode,
}

impl KeywordPattern {
    pub fn word(text: &str) -> Self {
        Self {
            text: text.into(),
            mode: MatchMode::Word,
        }
    }
    pub fn phrase(text: &str) -> Self {
        Self {
            text: text.into(),
            mode: MatchMode::Phrase,
        }
    }
    pub fn prefix(text: &str) -> Self {
        Self {
            text: text.into(),
            mode: MatchMode::Prefix,
        }
    }

    fn matches(&self, span: &str) -> bool {
        match self.mode {
            MatchMode::Prefix => span.trim_start().starts_with(&self.text),
            MatchMode::Phrase => span.to_lowercase().contains(&self.text.to_lowercase()),
            MatchMode::Word => {
                let pat = words(&self.text);
                let hay = words(span);
                !pat.is_empty() && hay.windows(pat.len()).any(|w| w == pat.as_slice())
            }
        }
    }
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// One OCR keyword rule. It fires when at least `min_matches` distinct patterns
/// match some OCR span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeywordRule {
    pub name: String,
    pub screen_type: ScreenType,
    #[serde(default)]
    pub app_hint: Option<String>,
    pub patterns: Vec<KeywordPattern>,
    pub min_matches: usize,
    /// Confidence assigned when the rule fires, in [0, 1].
    pub confidence: f64,
}

impl KeywordRule {
    fn evaluate(&self, spans: &[&str]) -> Option<RuleHit> {
        let matched: Vec<String> = self
            .patterns
            .iter()
            .filter(|p| spans.iter().any(|s| p.matches(s)))
            .map(|p| p.text.clone())
            .collect();
        (matched.len() >= self.min_matches.max(1)).then(|| RuleHit {
            rule: self.name.clone(),
            screen_type: self.screen_type,
            app_hint: self.app_hint.clone(),
            confidence: self.confidence,
            matched,
        })
    }
}

/// Rules plus the thresholds of the disagreement policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClassifyRules {
    pub rules: Vec<KeywordRule>,
    /// A fired rule at or above this confidence overrides the model.
    pub high_confidence: f64,
    /// Minimum model confidence to accept a model-only answer.
    pub min_model_confidence: f64,
}

impl Default for ClassifyRules {
    fn default() -> Self {
        Self::spec_examples()
    }
}

impl ClassifyRules {
    /// The example rules from the meeting-mode spec (section 6.7).
    pub fn spec_examples() -> Self {
        Self {
            rules: vec![
                KeywordRule {
                    name: "sanity_studio".into(),
                    screen_type: ScreenType::Cms,
                    app_hint: Some("Sanity Studio".into()),
                    patterns: ["Sanity", "Studio", "Structure", "Vision"]
                        .iter()
                        .map(|t| KeywordPattern::word(t))
                        .collect(),
                    min_matches: 2,
                    confidence: 0.9,
                },
                KeywordRule {
                    name: "slack_sidebar".into(),
                    screen_type: ScreenType::Chat,
                    app_hint: Some("Slack".into()),
                    patterns: [
                        "Huddles",
                        "Threads",
                        "Direct messages",
                        "Channels",
                        "Drafts & sent",
                        "Activity",
                    ]
                    .iter()
                    .map(|t| KeywordPattern::word(t))
                    .collect(),
                    min_matches: 2,
                    confidence: 0.85,
                },
                KeywordRule {
                    name: "shell_prompt".into(),
                    screen_type: ScreenType::Code,
                    app_hint: Some("Terminal".into()),
                    patterns: ["$ ", "% ", "\u{276f} ", "\u{279c} ", "PS C:\\"]
                        .iter()
                        .map(|t| KeywordPattern::prefix(t))
                        .collect(),
                    min_matches: 1,
                    confidence: 0.8,
                },
                KeywordRule {
                    name: "meet_left".into(),
                    screen_type: ScreenType::MeetGallery,
                    app_hint: Some("Google Meet".into()),
                    patterns: vec![KeywordPattern::phrase("You left the meeting")],
                    min_matches: 1,
                    confidence: 0.95,
                },
            ],
            high_confidence: 0.8,
            min_model_confidence: 0.5,
        }
    }

    /// Evaluate every rule against the OCR spans of one keyframe.
    pub fn evaluate<S: AsRef<str>>(&self, ocr_spans: &[S]) -> RuleEvaluation {
        let spans: Vec<&str> = ocr_spans.iter().map(AsRef::as_ref).collect();
        let mut hits: Vec<RuleHit> = self
            .rules
            .iter()
            .filter_map(|r| r.evaluate(&spans))
            .collect();
        hits.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then(b.matched.len().cmp(&a.matched.len()))
        });
        let conflicting = match (hits.first(), hits.get(1)) {
            (Some(a), Some(b)) => {
                a.screen_type != b.screen_type
                    && a.confidence == b.confidence
                    && a.matched.len() == b.matched.len()
            }
            _ => false,
        };
        RuleEvaluation { hits, conflicting }
    }
}

/// A rule that fired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleHit {
    pub rule: String,
    pub screen_type: ScreenType,
    pub app_hint: Option<String>,
    pub confidence: f64,
    pub matched: Vec<String>,
}

/// All fired rules, strongest first.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuleEvaluation {
    pub hits: Vec<RuleHit>,
    /// The two strongest hits tie but name different screen types.
    pub conflicting: bool,
}

impl RuleEvaluation {
    pub fn best(&self) -> Option<&RuleHit> {
        self.hits.first()
    }
}

/// How the final class was decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifyMethod {
    /// Model and rule agree.
    ModelAndRule,
    /// No rule fired; the model answer was accepted.
    Model,
    /// A high-confidence rule decided (model missing or disagreeing).
    Rule,
    /// A weak or ambiguous rule disagreed with the model.
    Disagreement,
    /// No rule and the model was missing or below the confidence floor.
    LowConfidence,
}

/// Final classification of one keyframe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenClass {
    pub screen_type: ScreenType,
    pub app_hint: Option<String>,
    pub confidence: f64,
    pub method: ClassifyMethod,
    /// Present only when the final type matches the model's answer.
    pub canvas_bbox: Option<BBox>,
    /// Name of the deciding or agreeing rule.
    pub rule: Option<String>,
}

impl ScreenClass {
    /// Gate: only whiteboards proceed to board reading.
    pub fn reads_board(&self) -> bool {
        self.screen_type == ScreenType::Whiteboard
    }
}

fn non_empty(s: &str) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// Combine the model answer (if the request succeeded) with the rule evaluation.
pub fn combine(
    model: Option<&ScreenClassOutput>,
    rules: &RuleEvaluation,
    policy: &ClassifyRules,
) -> ScreenClass {
    let best = rules.best();
    let strong = best.filter(|h| !rules.conflicting && h.confidence >= policy.high_confidence);
    let from_model =
        |m: &ScreenClassOutput, method, confidence, rule: Option<&RuleHit>| ScreenClass {
            screen_type: m.screen_type,
            app_hint: non_empty(&m.app_hint).or_else(|| rule.and_then(|r| r.app_hint.clone())),
            confidence,
            method,
            canvas_bbox: Some(m.canvas_bbox),
            rule: rule.map(|r| r.rule.clone()),
        };
    let unknown = |method, rule: Option<&RuleHit>| ScreenClass {
        screen_type: ScreenType::Unknown,
        app_hint: None,
        confidence: 0.0,
        method,
        canvas_bbox: None,
        rule: rule.map(|r| r.rule.clone()),
    };

    match (model, strong, best) {
        (Some(m), Some(hit), _) if m.screen_type == hit.screen_type => from_model(
            m,
            ClassifyMethod::ModelAndRule,
            m.confidence.max(hit.confidence),
            Some(hit),
        ),
        (_, Some(hit), _) => ScreenClass {
            screen_type: hit.screen_type,
            app_hint: hit.app_hint.clone(),
            confidence: hit.confidence,
            method: ClassifyMethod::Rule,
            canvas_bbox: None,
            rule: Some(hit.rule.clone()),
        },
        (Some(m), None, Some(weak)) => {
            let agrees = rules.hits.iter().any(|h| h.screen_type == m.screen_type);
            if agrees && m.confidence >= policy.min_model_confidence {
                from_model(m, ClassifyMethod::ModelAndRule, m.confidence, Some(weak))
            } else {
                unknown(ClassifyMethod::Disagreement, Some(weak))
            }
        }
        (Some(m), None, None) if m.confidence >= policy.min_model_confidence => {
            from_model(m, ClassifyMethod::Model, m.confidence, None)
        }
        (_, None, weak) => unknown(ClassifyMethod::LowConfidence, weak),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(t: ScreenType, conf: f64) -> ScreenClassOutput {
        ScreenClassOutput {
            screen_type: t,
            app_hint: String::new(),
            canvas_bbox: BBox::new(10.0, 20.0, 700.0, 400.0),
            confidence: conf,
        }
    }

    #[test]
    fn cms_keywords_override_a_whiteboard_answer() {
        // The reference trap: a CMS studio misread as a board.
        let rules = ClassifyRules::spec_examples();
        let eval = rules.evaluate(&["Studio", "Structure", "Vision", "Example Content Type"]);
        let out = combine(Some(&model(ScreenType::Whiteboard, 0.9)), &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Cms);
        assert_eq!(out.method, ClassifyMethod::Rule);
        assert_eq!(out.rule.as_deref(), Some("sanity_studio"));
        assert!(out.canvas_bbox.is_none());
        assert!(!out.reads_board());
    }

    #[test]
    fn single_cms_keyword_does_not_fire() {
        let rules = ClassifyRules::spec_examples();
        let eval = rules.evaluate(&["Vision board for next quarter"]);
        assert!(eval.hits.is_empty());
        let out = combine(Some(&model(ScreenType::Whiteboard, 0.9)), &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Whiteboard);
        assert_eq!(out.method, ClassifyMethod::Model);
        assert!(out.reads_board());
        assert!(out.canvas_bbox.is_some());
    }

    #[test]
    fn word_mode_ignores_substrings_inside_words() {
        let p = KeywordPattern::word("Studio");
        assert!(p.matches("Open Studio now"));
        assert!(!p.matches("Studious"));
        assert!(KeywordPattern::word("Direct messages").matches("Direct  Messages:"));
    }

    #[test]
    fn slack_sidebar_and_shell_and_meet_rules() {
        let rules = ClassifyRules::spec_examples();
        let chat = rules.evaluate(&["Threads", "Huddles", "general"]);
        assert_eq!(chat.best().map(|h| h.screen_type), Some(ScreenType::Chat));

        let code = rules.evaluate(&["$ cargo test", "running 3 tests"]);
        assert_eq!(code.best().map(|h| h.screen_type), Some(ScreenType::Code));
        assert!(rules.evaluate(&["Total: 5 $"]).hits.is_empty());

        let meet = rules.evaluate(&["You left the meeting", "Rejoin"]);
        let out = combine(None, &meet, &rules);
        assert_eq!(out.screen_type, ScreenType::MeetGallery);
        assert_eq!(out.app_hint.as_deref(), Some("Google Meet"));
    }

    #[test]
    fn agreement_takes_max_confidence_and_keeps_bbox() {
        let rules = ClassifyRules::spec_examples();
        let eval = rules.evaluate(&["Sanity", "Structure"]);
        let out = combine(Some(&model(ScreenType::Cms, 0.6)), &eval, &rules);
        assert_eq!(out.method, ClassifyMethod::ModelAndRule);
        assert!((out.confidence - 0.9).abs() < 1e-9);
        assert!(out.canvas_bbox.is_some());
        assert_eq!(out.app_hint.as_deref(), Some("Sanity Studio"));
    }

    #[test]
    fn weak_rule_disagreement_is_unknown() {
        let mut rules = ClassifyRules::spec_examples();
        rules.high_confidence = 0.99;
        let eval = rules.evaluate(&["Sanity", "Studio"]);
        let out = combine(Some(&model(ScreenType::Whiteboard, 0.95)), &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Unknown);
        assert_eq!(out.method, ClassifyMethod::Disagreement);
        // A weak rule that agrees keeps the model answer.
        let out = combine(Some(&model(ScreenType::Cms, 0.7)), &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Cms);
    }

    #[test]
    fn conflicting_rules_are_not_strong() {
        let rules = ClassifyRules {
            rules: vec![
                KeywordRule {
                    name: "a".into(),
                    screen_type: ScreenType::Slides,
                    app_hint: None,
                    patterns: vec![KeywordPattern::word("alpha")],
                    min_matches: 1,
                    confidence: 0.9,
                },
                KeywordRule {
                    name: "b".into(),
                    screen_type: ScreenType::Web,
                    app_hint: None,
                    patterns: vec![KeywordPattern::word("beta")],
                    min_matches: 1,
                    confidence: 0.9,
                },
            ],
            high_confidence: 0.8,
            min_model_confidence: 0.5,
        };
        let eval = rules.evaluate(&["alpha beta"]);
        assert!(eval.conflicting);
        let out = combine(None, &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Unknown);
    }

    #[test]
    fn low_model_confidence_without_rules_is_unknown() {
        let rules = ClassifyRules::spec_examples();
        let eval = rules.evaluate::<&str>(&[]);
        let out = combine(Some(&model(ScreenType::Slides, 0.3)), &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Unknown);
        assert_eq!(out.method, ClassifyMethod::LowConfidence);
        let out = combine(None, &eval, &rules);
        assert_eq!(out.screen_type, ScreenType::Unknown);
    }

    #[test]
    fn classify_schema_is_simple() -> Result<()> {
        let schema = crate::schema::OutputSchema::for_type::<ScreenClassOutput>()?;
        let text = schema.text();
        assert!(!text.contains("$ref") && !text.contains("anyOf"), "{text}");
        assert!(text.contains("meet_gallery"));
        Ok(())
    }
}
