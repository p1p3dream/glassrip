//! `ocr_vocabulary`: the ASR vocabulary list from on-screen text (spec 6.8, 6.12).
//!
//! Participants come from tile labels and banners; board labels and
//! identifiers come from text inside the shared area. A term must appear in at
//! least `min_keyframes` keyframes, which drops one-off OCR misreads.

use std::collections::{BTreeMap, HashMap};

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use schemars::JsonSchema;
use serde::Serialize;

use crate::artifacts::{self, OcrKeyframe, TermKind, TextRegion, Vocabulary, VocabularyTerm};
use crate::layout::{names_match, strip_ellipsis};
use crate::stages::input;

/// Parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VocabularyParams {
    pub max_terms: usize,
    pub min_keyframes: u32,
    pub min_confidence: f64,
    /// Capitalized words too common to help recognition.
    pub stopwords: Vec<String>,
}

impl Default for VocabularyParams {
    fn default() -> Self {
        Self {
            max_terms: 200,
            min_keyframes: 2,
            min_confidence: 0.85,
            stopwords: [
                "the", "this", "that", "and", "for", "with", "you", "your", "our", "set", "new",
                "all", "are", "not", "can", "from", "what", "how", "why", "when", "who", "into",
                "home", "page", "idea", "create", "share", "view", "open", "add", "edit", "more",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

/// Participant names seen in at least `min_keyframes` keyframes, longest
/// spelling kept when labels were truncated.
pub fn participants(items: &[OcrKeyframe], min_keyframes: u32) -> Vec<(String, u32, String)> {
    let mut found: Vec<(String, u32, String)> = Vec::new();
    for item in items {
        let mut seen_here: Vec<usize> = Vec::new();
        for n in &item.tile_names {
            let n = strip_ellipsis(n);
            match found.iter().position(|(f, _, _)| names_match(f, n)) {
                Some(i) => {
                    if n.len() > found[i].0.len() {
                        found[i].0 = n.to_string();
                    }
                    if !seen_here.contains(&i) {
                        found[i].1 += 1;
                        seen_here.push(i);
                    }
                }
                None => {
                    found.push((n.to_string(), 1, item.keyframe_id.clone()));
                    seen_here.push(found.len() - 1);
                }
            }
        }
    }
    found.retain(|(_, n, _)| *n >= min_keyframes);
    found
}

fn identifier_like(t: &str) -> bool {
    let inner_upper =
        t.chars().skip(1).any(char::is_uppercase) && t.chars().any(char::is_lowercase);
    let acronym = t.len() >= 2
        && t.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
    t.contains('_')
        || t.contains('/')
        || t.contains('.')
        || inner_upper
        || acronym
        || (t.chars().any(|c| c.is_ascii_digit()) && t.chars().any(char::is_alphabetic))
}

/// Build the vocabulary.
pub fn build(items: &[OcrKeyframe], p: &VocabularyParams) -> Vocabulary {
    let mut terms: Vec<VocabularyTerm> = participants(items, p.min_keyframes)
        .into_iter()
        .map(|(text, keyframes, first)| VocabularyTerm {
            text,
            kind: TermKind::Participant,
            keyframes,
            first_keyframe_id: first,
        })
        .collect();
    let stop: std::collections::HashSet<String> =
        p.stopwords.iter().map(|s| s.to_lowercase()).collect();
    // token -> (display, keyframes, first keyframe)
    let mut counts: HashMap<String, (String, u32, String)> = HashMap::new();
    for item in items {
        let mut here: std::collections::HashSet<String> = std::collections::HashSet::new();
        for s in &item.spans {
            if !matches!(s.region, TextRegion::Unassigned | TextRegion::Canvas)
                || s.confidence < p.min_confidence
            {
                continue;
            }
            for raw in s.text.split_whitespace() {
                let t = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '/');
                let letters = t.chars().filter(|c| c.is_alphabetic()).count();
                if letters < 3 || stop.contains(&t.to_lowercase()) {
                    continue;
                }
                let capital = t.chars().next().is_some_and(char::is_uppercase);
                if !(capital || identifier_like(t)) {
                    continue;
                }
                let key = t.to_lowercase();
                if here.insert(key.clone()) {
                    let e = counts
                        .entry(key)
                        .or_insert_with(|| (t.to_string(), 0, item.keyframe_id.clone()));
                    e.1 += 1;
                }
            }
        }
    }
    let mut rest: Vec<VocabularyTerm> = counts
        .into_values()
        .filter(|(text, n, _)| {
            *n >= p.min_keyframes
                && !terms.iter().any(|t| {
                    t.text
                        .to_lowercase()
                        .split_whitespace()
                        .any(|w| w == text.to_lowercase())
                })
        })
        .map(|(text, keyframes, first)| VocabularyTerm {
            kind: if identifier_like(&text) {
                TermKind::Identifier
            } else {
                TermKind::BoardLabel
            },
            text,
            keyframes,
            first_keyframe_id: first,
        })
        .collect();
    rest.sort_by(|a, b| b.keyframes.cmp(&a.keyframes).then(a.text.cmp(&b.text)));
    terms.sort_by(|a, b| b.keyframes.cmp(&a.keyframes).then(a.text.cmp(&b.text)));
    terms.extend(rest);
    terms.truncate(p.max_terms);
    Vocabulary { terms }
}

/// The stage (one item, id `vocabulary`).
pub struct VocabularyStage {
    params: VocabularyParams,
}

impl VocabularyStage {
    pub fn new(params: VocabularyParams) -> Self {
        Self { params }
    }
}

impl Stage for VocabularyStage {
    type Params = VocabularyParams;
    type Work = Vec<OcrKeyframe>;
    type Output = Vocabulary;

    fn name(&self) -> &'static str {
        "ocr_vocabulary"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: artifacts::ASR_VOCABULARY,
            version: artifacts::output_version(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(artifacts::OCR)]
    }
    fn params(&self) -> &VocabularyParams {
        &self.params
    }
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<Vec<OcrKeyframe>>>, StageError> {
        let items: BTreeMap<String, OcrKeyframe> = inputs
            .read_ok::<OcrKeyframe>(artifacts::OCR)?
            .into_iter()
            .collect();
        Ok(vec![WorkItem {
            id: "vocabulary".into(),
            work: items.into_values().collect(),
        }])
    }
    async fn process(
        &self,
        _ctx: &ItemContext,
        items: Vec<OcrKeyframe>,
    ) -> Result<Vocabulary, ErrorInfo> {
        Ok(build(&items, &self.params))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::artifacts::OcrSpan;
    use glassrip_vision::BBox;

    fn kf(id: &str, names: &[&str], texts: &[&str]) -> OcrKeyframe {
        OcrKeyframe {
            keyframe_id: id.into(),
            image_width: 100,
            image_height: 100,
            execution_provider: "CPU".into(),
            spans: texts
                .iter()
                .map(|t| OcrSpan {
                    text: t.to_string(),
                    bbox: BBox::new(0.0, 0.0, 10.0, 10.0),
                    confidence: 0.95,
                    region: TextRegion::Unassigned,
                    chrome_reason: None,
                    bg_luma: 240.0,
                })
                .collect(),
            tile_names: names.iter().map(|s| s.to_string()).collect(),
            share_area: None,
        }
    }

    #[test]
    fn participants_first_and_repeated_terms_only() {
        let items = vec![
            kf(
                "k1",
                &["Ada Quill", "Bo Tran Li..."],
                &["Ledger API", "order_svc handles", "Once"],
            ),
            kf("k2", &["Bo Tran Liu"], &["Ledger API", "order_svc", "gRPC"]),
            kf("k3", &["Ada Quill"], &["gRPC stream"]),
        ];
        let v = build(&items, &VocabularyParams::default());
        let texts: Vec<&str> = v.terms.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(&texts[..2], &["Ada Quill", "Bo Tran Liu"], "{texts:?}");
        assert!(texts.contains(&"Ledger"));
        assert!(texts.contains(&"order_svc"));
        assert!(texts.contains(&"gRPC"));
        assert!(texts.contains(&"API"));
        assert!(!texts.contains(&"Once"));
        let kind = |t: &str| v.terms.iter().find(|x| x.text == t).map(|x| x.kind);
        assert_eq!(kind("order_svc"), Some(TermKind::Identifier));
        assert_eq!(kind("Ledger"), Some(TermKind::BoardLabel));
    }
}
