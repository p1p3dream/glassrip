//! Board reading: prompt, schema-constrained output types, and the post-model
//! validation rules that need no pixels.
//!
//! The model sees only the canvas crop and reports every element with a bbox in
//! the pixels of the image it was sent. [`BoardReadOutput::to_canvas_coords`]
//! maps those boxes to canvas (crop source) pixels before [`validate_board`].

use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::backend::{GenerationOptions, VisionRequest};
use crate::error::Result;
use crate::geometry::BBox;
use crate::image_prep::PreparedImage;

/// Board-read prompt (prototype prompt minus change and previous-reading blocks, plus bboxes).
/// The schema text is appended by [`VisionRequest::for_output`].
pub const BOARD_READ_PROMPT: &str = "\
You are reading the shared whiteboard canvas from a video meeting. The image is a crop of a \
computer monitor photographed with a phone; it usually shows a Miro whiteboard with an \
architecture diagram, shared in Google Meet.

Report ONLY what is on the shared whiteboard canvas. Ignore everything outside it: browser tabs and \
bookmarks, the macOS menu bar and dock, other monitors or terminal windows, the Miro left sidebar \
(board list, \"Overview\", \"Browse\", \"Create section\"), Miro toolbars and shape menus, zoom controls, \
the Meet \"Name (Presenting...)\" banner, and video tiles of meeting participants. Names on video \
tiles or banners are NOT owner tags.

Definitions:
- nodes: rectangles/boxes in the diagram. Give each node a short unique local_id (n1, n2, ...). Use \
the full text inside the box as its text (join wrapped lines with a space).
- edges: lines or arrows connecting two nodes. \"src\" and \"dst\" must be local_id values from your \
nodes list. \"src\" is the tail, \"dst\" is the end with the arrowhead. \"label\" is the small text \
written on the line itself (for example a protocol name), or an empty string if the line has no text. \
Never use a node's text as an edge label. \"style\" is \"dashed\" for dotted/dashed lines, otherwise \
\"solid\". A dashed line may be long and curved and carry a text label; follow it to the box at each \
end and put its text in \"label\".
- stickies: colored sticky notes and cards (yellow, blue, pink, etc.) containing words, sentences, \
ideas, or questions, including notes arranged in a grid. Every such note goes here, never in \
other_visible_text. A sticky that is being edited may look white with selection handles; still list \
it once. Transcribe the full text.
- owner_tags: green sticky notes containing only a person's first name, placed on or next to a box. \
\"near\" is that box's local_id, or an empty string if it is not next to a box.
- other_visible_text: any other readable text on the canvas itself (not UI chrome) that is not \
already listed above.
- bbox: for every node, sticky, owner tag, and other text, the box around it in pixel coordinates of \
this image: x1, y1 is the top-left corner and x2, y2 the bottom-right corner.

Rules:
- Transcribe text VERBATIM, exactly as written, including punctuation. Do not correct, summarize or \
complete it.
- If text is too blurry, cut off, or too small to read, write \"[illegible]\" for the unreadable part. \
Never guess or invent text.
- List each item once. Only list items you can actually see in THIS image. Use empty lists when \
nothing applies.
- \"conf\" and \"confidence\" are numbers from 0 to 1; \"confidence\" is how legible the board is in \
this image.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EdgeStyle {
    Solid,
    Dashed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StickyColor {
    Yellow,
    Blue,
    Pink,
    Green,
    Orange,
    Purple,
    White,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardNode {
    pub local_id: String,
    pub text: String,
    pub bbox: BBox,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardEdge {
    pub src: String,
    pub dst: String,
    /// Empty string when the line carries no text.
    pub label: String,
    pub style: EdgeStyle,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sticky {
    pub text: String,
    pub color: StickyColor,
    pub bbox: BBox,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OwnerTag {
    /// Name exactly as written on the tag.
    pub name_raw: String,
    /// `local_id` of the node the tag sits on or next to, or empty.
    pub near: String,
    pub bbox: BBox,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextItem {
    pub text: String,
    pub bbox: BBox,
}

/// Model output for one board read. Array lengths are bounded to keep the schema grammar small.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardReadOutput {
    #[schemars(length(max = 60))]
    pub nodes: Vec<BoardNode>,
    #[schemars(length(max = 80))]
    pub edges: Vec<BoardEdge>,
    #[schemars(length(max = 60))]
    pub stickies: Vec<Sticky>,
    #[schemars(length(max = 20))]
    pub owner_tags: Vec<OwnerTag>,
    #[schemars(length(max = 40))]
    pub other_visible_text: Vec<TextItem>,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub confidence: f64,
}

impl BoardReadOutput {
    /// Map every bbox from sent-image pixels to canvas pixels.
    pub fn to_canvas_coords(mut self, prepared: &PreparedImage) -> Self {
        for n in &mut self.nodes {
            n.bbox = n.bbox.to_source(prepared);
        }
        for s in &mut self.stickies {
            s.bbox = s.bbox.to_source(prepared);
        }
        for o in &mut self.owner_tags {
            o.bbox = o.bbox.to_source(prepared);
        }
        for t in &mut self.other_visible_text {
            t.bbox = t.bbox.to_source(prepared);
        }
        self
    }
}

/// Build the board-read request for an already prepared canvas crop.
pub fn board_read_request(
    prepared: &PreparedImage,
    options: GenerationOptions,
) -> Result<VisionRequest> {
    VisionRequest::for_output::<BoardReadOutput>(BOARD_READ_PROMPT, prepared.image.clone(), options)
}

/// UI strings and patterns that are never board content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChromeDenylist {
    /// Whole-text matches (case-insensitive, whitespace-normalized).
    pub exact: Vec<String>,
    /// Case-insensitive substrings.
    pub contains: Vec<String>,
    /// Reject zoom percentages such as `100%` or `85 %`.
    pub zoom_percentages: bool,
}

impl ChromeDenylist {
    /// Miro and Google Meet defaults from the spec (section 6.9).
    pub fn miro_meet_defaults() -> Self {
        Self {
            exact: [
                "Overview",
                "Browse",
                "Create section",
                "Convert to",
                "Share",
                "Internal",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            contains: vec!["(Presenting".to_string()],
            zoom_percentages: true,
        }
    }

    pub fn matches(&self, text: &str) -> bool {
        let n = normalize(text);
        if n.is_empty() {
            return false;
        }
        if self.exact.iter().any(|e| normalize(e) == n) {
            return true;
        }
        if self.contains.iter().any(|c| n.contains(&normalize(c))) {
            return true;
        }
        self.zoom_percentages && is_zoom_percentage(&n)
    }
}

fn is_zoom_percentage(n: &str) -> bool {
    let Some(num) = n.strip_suffix('%') else {
        return false;
    };
    let num = num.trim_end();
    !num.is_empty() && num.len() <= 3 && num.chars().all(|c| c.is_ascii_digit())
}

/// Lowercase, trim, and collapse internal whitespace.
pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Inputs to the non-pixel validation rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardValidationConfig {
    pub denylist: ChromeDenylist,
    /// Participant names and aliases (from Meet tiles and the speaker table).
    pub participant_names: Vec<String>,
    pub max_edge_label_words: usize,
    /// Slack in pixels when checking that a bbox lies inside the canvas.
    pub bbox_tolerance_px: f64,
}

impl Default for BoardValidationConfig {
    fn default() -> Self {
        Self {
            denylist: ChromeDenylist::miro_meet_defaults(),
            participant_names: Vec::new(),
            max_edge_label_words: 6,
            bbox_tolerance_px: 2.0,
        }
    }
}

/// Canvas size in pixels (the coordinate space of the bboxes being validated).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CanvasSize {
    pub width: f64,
    pub height: f64,
}

/// Which output list an element came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ElementList {
    Nodes,
    Edges,
    Stickies,
    OwnerTags,
    OtherVisibleText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    /// Text is on the chrome denylist.
    Denylist,
    /// Bbox lies outside the canvas, so the text is chrome by definition.
    OutsideCanvas,
    /// A node whose text is a participant name.
    ParticipantName,
}

/// An element removed as UI chrome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RejectedItem {
    pub list: ElementList,
    pub text: String,
    pub bbox: Option<BBox>,
    pub reason: RejectReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    EmptyText,
    MalformedBBox,
    DuplicateNodeId,
    /// Same text in more than one list; the lower-priority copy was dropped.
    DuplicateAcrossLists,
    /// A sticky whose text is a participant name was moved to owner tags.
    StickyReclassifiedAsOwner,
    /// Edge endpoint does not reference a kept node; the edge was dropped.
    DanglingEdge,
    SelfLoop,
    /// Label exceeded the word limit and was cleared.
    LabelTooLong,
    /// Label equaled a node's text and was cleared.
    LabelIsNodeText,
    /// Label was chrome and was cleared.
    LabelIsChrome,
    /// Owner tag `near` pointed at a missing node and was cleared.
    OwnerNearMissing,
}

/// A non-fatal correction applied during validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValidationIssue {
    pub list: ElementList,
    pub kind: IssueKind,
    pub detail: String,
}

/// Board reading after validation. Bboxes are in canvas pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValidatedBoard {
    pub nodes: Vec<BoardNode>,
    pub edges: Vec<BoardEdge>,
    pub stickies: Vec<Sticky>,
    pub owner_tags: Vec<OwnerTag>,
    pub other_visible_text: Vec<TextItem>,
    pub confidence: f64,
    pub chrome_rejected: Vec<RejectedItem>,
    pub issues: Vec<ValidationIssue>,
    /// `confidence == 0` or no board content: the frame should be re-classified.
    pub needs_reclassification: bool,
}

struct Ctx<'a> {
    cfg: &'a BoardValidationConfig,
    canvas: CanvasSize,
    participants: HashSet<String>,
    rejected: Vec<RejectedItem>,
    issues: Vec<ValidationIssue>,
}

impl Ctx<'_> {
    fn issue(&mut self, list: ElementList, kind: IssueKind, detail: String) {
        self.issues.push(ValidationIssue { list, kind, detail });
    }

    fn reject(&mut self, list: ElementList, text: &str, bbox: Option<BBox>, reason: RejectReason) {
        self.rejected.push(RejectedItem {
            list,
            text: text.to_string(),
            bbox,
            reason,
        });
    }

    /// Shared per-element checks. Returns false if the element must be dropped.
    fn keep(&mut self, list: ElementList, text: &str, bbox: &BBox) -> bool {
        if text.trim().is_empty() {
            self.issue(
                list,
                IssueKind::EmptyText,
                "element with empty text dropped".into(),
            );
            return false;
        }
        if !bbox.is_well_formed() {
            self.issue(
                list,
                IssueKind::MalformedBBox,
                format!("{text:?} has bbox {bbox:?}"),
            );
            return false;
        }
        if self.cfg.denylist.matches(text) {
            self.reject(list, text, Some(*bbox), RejectReason::Denylist);
            return false;
        }
        if !bbox.is_inside(
            self.canvas.width,
            self.canvas.height,
            self.cfg.bbox_tolerance_px,
        ) {
            self.reject(list, text, Some(*bbox), RejectReason::OutsideCanvas);
            return false;
        }
        true
    }

    fn is_participant(&self, text: &str) -> bool {
        self.participants.contains(&normalize(text))
    }
}

/// Apply the non-pixel validation rules from spec section 6.11.
///
/// Rules, in order:
/// 1. Drop empty text and malformed boxes; move denylisted text and boxes outside
///    the canvas to `chrome_rejected`.
/// 2. Reject nodes whose text is a participant name; move stickies whose text is a
///    participant name to owner tags.
/// 3. Keep the first node per `local_id`.
/// 4. Exclusive membership by normalized text: a text appears in one list only.
///    When the text is a participant name the keep order is owner tags, nodes,
///    stickies, other visible text; otherwise it is nodes, stickies, owner tags,
///    other visible text. Copies in lower-ranked lists are dropped with a
///    `DuplicateAcrossLists` issue.
/// 5. Edges must reference kept nodes (never create nodes from endpoints), may not
///    be self-loops, and carry labels of at most `max_edge_label_words` words that
///    are neither a node's text nor chrome.
/// 6. Owner tag `near` must reference a kept node or be empty.
/// 7. Flag re-classification when `confidence == 0` or nothing remains.
pub fn validate_board(
    output: BoardReadOutput,
    canvas: CanvasSize,
    cfg: &BoardValidationConfig,
) -> ValidatedBoard {
    let mut ctx = Ctx {
        cfg,
        canvas,
        participants: cfg.participant_names.iter().map(|n| normalize(n)).collect(),
        rejected: Vec::new(),
        issues: Vec::new(),
    };
    let BoardReadOutput {
        nodes,
        edges,
        stickies,
        owner_tags,
        other_visible_text,
        confidence,
    } = output;

    // Rules 1-3 for nodes.
    let mut seen_ids = HashSet::new();
    let mut kept_nodes = Vec::new();
    for n in nodes {
        if !ctx.keep(ElementList::Nodes, &n.text, &n.bbox) {
            continue;
        }
        if ctx.is_participant(&n.text) {
            ctx.reject(
                ElementList::Nodes,
                &n.text,
                Some(n.bbox),
                RejectReason::ParticipantName,
            );
            continue;
        }
        if !seen_ids.insert(n.local_id.clone()) {
            ctx.issue(
                ElementList::Nodes,
                IssueKind::DuplicateNodeId,
                format!("duplicate local_id {:?} ({:?}) dropped", n.local_id, n.text),
            );
            continue;
        }
        kept_nodes.push(n);
    }
    let mut nodes = kept_nodes;

    let mut owner_tags: Vec<OwnerTag> = owner_tags
        .into_iter()
        .filter(|o| ctx.keep(ElementList::OwnerTags, &o.name_raw, &o.bbox))
        .collect();

    let mut kept_stickies = Vec::new();
    for s in stickies {
        if !ctx.keep(ElementList::Stickies, &s.text, &s.bbox) {
            continue;
        }
        if ctx.is_participant(&s.text) {
            ctx.issue(
                ElementList::Stickies,
                IssueKind::StickyReclassifiedAsOwner,
                format!(
                    "sticky {:?} is a participant name; moved to owner tags",
                    s.text
                ),
            );
            owner_tags.push(OwnerTag {
                name_raw: s.text,
                near: String::new(),
                bbox: s.bbox,
            });
        } else {
            kept_stickies.push(s);
        }
    }
    let mut stickies = kept_stickies;

    let mut other: Vec<TextItem> = other_visible_text
        .into_iter()
        .filter(|t| ctx.keep(ElementList::OtherVisibleText, &t.text, &t.bbox))
        .collect();

    // Rule 4: exclusive membership by normalized text.
    let mut owner_of: HashMap<String, ElementList> = HashMap::new();
    let priority = |key: &str, participants: &HashSet<String>| -> [ElementList; 4] {
        if participants.contains(key) {
            [
                ElementList::OwnerTags,
                ElementList::Nodes,
                ElementList::Stickies,
                ElementList::OtherVisibleText,
            ]
        } else {
            [
                ElementList::Nodes,
                ElementList::Stickies,
                ElementList::OwnerTags,
                ElementList::OtherVisibleText,
            ]
        }
    };
    let mut present: HashMap<String, HashSet<ElementList>> = HashMap::new();
    for n in &nodes {
        present
            .entry(normalize(&n.text))
            .or_default()
            .insert(ElementList::Nodes);
    }
    for s in &stickies {
        present
            .entry(normalize(&s.text))
            .or_default()
            .insert(ElementList::Stickies);
    }
    for o in &owner_tags {
        present
            .entry(normalize(&o.name_raw))
            .or_default()
            .insert(ElementList::OwnerTags);
    }
    for t in &other {
        present
            .entry(normalize(&t.text))
            .or_default()
            .insert(ElementList::OtherVisibleText);
    }
    for (key, lists) in &present {
        if lists.len() > 1 {
            if let Some(winner) = priority(key, &ctx.participants)
                .into_iter()
                .find(|l| lists.contains(l))
            {
                owner_of.insert(key.clone(), winner);
            }
        }
    }
    let dedupe = |list: ElementList, text: &str, issues: &mut Vec<ValidationIssue>| -> bool {
        match owner_of.get(&normalize(text)) {
            Some(winner) if *winner != list => {
                issues.push(ValidationIssue {
                    list,
                    kind: IssueKind::DuplicateAcrossLists,
                    detail: format!("{text:?} kept only in {winner:?}"),
                });
                false
            }
            _ => true,
        }
    };
    nodes.retain(|n| dedupe(ElementList::Nodes, &n.text, &mut ctx.issues));
    stickies.retain(|s| dedupe(ElementList::Stickies, &s.text, &mut ctx.issues));
    owner_tags.retain(|o| dedupe(ElementList::OwnerTags, &o.name_raw, &mut ctx.issues));
    other.retain(|t| dedupe(ElementList::OtherVisibleText, &t.text, &mut ctx.issues));

    // Rule 5: edges.
    let node_ids: HashSet<&str> = nodes.iter().map(|n| n.local_id.as_str()).collect();
    let node_texts: HashSet<String> = nodes.iter().map(|n| normalize(&n.text)).collect();
    let mut kept_edges = Vec::new();
    for mut e in edges {
        if !node_ids.contains(e.src.as_str()) || !node_ids.contains(e.dst.as_str()) {
            ctx.issue(
                ElementList::Edges,
                IssueKind::DanglingEdge,
                format!("edge {:?} -> {:?} references a missing node", e.src, e.dst),
            );
            continue;
        }
        if e.src == e.dst {
            ctx.issue(
                ElementList::Edges,
                IssueKind::SelfLoop,
                format!("self-loop on {:?}", e.src),
            );
            continue;
        }
        let label = e.label.trim().to_string();
        if !label.is_empty() {
            let kind = if label.split_whitespace().count() > cfg.max_edge_label_words {
                Some(IssueKind::LabelTooLong)
            } else if node_texts.contains(&normalize(&label)) {
                Some(IssueKind::LabelIsNodeText)
            } else if cfg.denylist.matches(&label) {
                Some(IssueKind::LabelIsChrome)
            } else {
                None
            };
            if let Some(kind) = kind {
                ctx.issue(
                    ElementList::Edges,
                    kind,
                    format!("label {label:?} on {:?} -> {:?} cleared", e.src, e.dst),
                );
                e.label = String::new();
            } else {
                e.label = label;
            }
        }
        kept_edges.push(e);
    }

    // Rule 6: owner tag anchors.
    for o in &mut owner_tags {
        if !o.near.is_empty() && !node_ids.contains(o.near.as_str()) {
            ctx.issues.push(ValidationIssue {
                list: ElementList::OwnerTags,
                kind: IssueKind::OwnerNearMissing,
                detail: format!("owner {:?} near {:?} cleared", o.name_raw, o.near),
            });
            o.near = String::new();
        }
    }

    // Rule 7.
    let empty =
        nodes.is_empty() && kept_edges.is_empty() && stickies.is_empty() && owner_tags.is_empty();
    ValidatedBoard {
        needs_reclassification: confidence == 0.0 || empty,
        nodes,
        edges: kept_edges,
        stickies,
        owner_tags,
        other_visible_text: other,
        confidence,
        chrome_rejected: ctx.rejected,
        issues: ctx.issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANVAS: CanvasSize = CanvasSize {
        width: 1600.0,
        height: 900.0,
    };

    fn bb(x: f64, y: f64) -> BBox {
        BBox::new(x, y, x + 120.0, y + 60.0)
    }

    fn node(id: &str, text: &str, x: f64, y: f64) -> BoardNode {
        BoardNode {
            local_id: id.into(),
            text: text.into(),
            bbox: bb(x, y),
            conf: 0.9,
        }
    }

    fn edge(src: &str, dst: &str, label: &str) -> BoardEdge {
        BoardEdge {
            src: src.into(),
            dst: dst.into(),
            label: label.into(),
            style: EdgeStyle::Solid,
            conf: 0.8,
        }
    }

    fn empty_output() -> BoardReadOutput {
        BoardReadOutput {
            nodes: vec![],
            edges: vec![],
            stickies: vec![],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.8,
        }
    }

    fn cfg() -> BoardValidationConfig {
        BoardValidationConfig {
            participant_names: vec!["Avery".into(), "Jordan".into()],
            ..BoardValidationConfig::default()
        }
    }

    #[test]
    fn clean_board_passes_unchanged() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "Queue", 400.0, 100.0),
        ];
        out.edges = vec![edge("n1", "n2", "gRPC")];
        let v = validate_board(out.clone(), CANVAS, &cfg());
        assert_eq!(v.nodes, out.nodes);
        assert_eq!(v.edges, out.edges);
        assert!(v.issues.is_empty() && v.chrome_rejected.is_empty());
        assert!(!v.needs_reclassification);
    }

    #[test]
    fn bbox_outside_canvas_is_chrome() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "Sidebar item", 1550.0, 100.0),
        ];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.nodes.len(), 1);
        assert_eq!(v.chrome_rejected[0].reason, RejectReason::OutsideCanvas);
    }

    #[test]
    fn denylist_rejects_ui_strings_banner_and_zoom() {
        let d = ChromeDenylist::miro_meet_defaults();
        assert!(d.matches("  create   SECTION "));
        assert!(d.matches("Sample Person (Presenting)"));
        assert!(d.matches("100%"));
        assert!(d.matches("85 %"));
        assert!(!d.matches("Share service"));
        assert!(!d.matches("50% of traffic"));
        let mut out = empty_output();
        out.other_visible_text = vec![TextItem {
            text: "Overview".into(),
            bbox: bb(10.0, 10.0),
        }];
        let v = validate_board(out, CANVAS, &cfg());
        assert!(v.other_visible_text.is_empty());
        assert_eq!(v.chrome_rejected[0].reason, RejectReason::Denylist);
    }

    #[test]
    fn participant_named_node_is_rejected_and_sticky_moves_to_owner() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "avery", 400.0, 100.0),
        ];
        out.stickies = vec![Sticky {
            text: "Jordan".into(),
            color: StickyColor::Green,
            bbox: bb(100.0, 200.0),
        }];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.nodes.len(), 1);
        assert_eq!(v.chrome_rejected[0].reason, RejectReason::ParticipantName);
        assert!(v.stickies.is_empty());
        assert_eq!(v.owner_tags[0].name_raw, "Jordan");
        assert!(v
            .issues
            .iter()
            .any(|i| i.kind == IssueKind::StickyReclassifiedAsOwner));
    }

    #[test]
    fn text_belongs_to_exactly_one_list() {
        let mut out = empty_output();
        out.nodes = vec![node("n1", "Cache Layer", 100.0, 100.0)];
        out.stickies = vec![Sticky {
            text: "cache layer".into(),
            color: StickyColor::Yellow,
            bbox: bb(100.0, 100.0),
        }];
        out.other_visible_text = vec![TextItem {
            text: "Cache  Layer".into(),
            bbox: bb(100.0, 100.0),
        }];
        out.owner_tags = vec![OwnerTag {
            name_raw: "Avery".into(),
            near: "n1".into(),
            bbox: bb(300.0, 300.0),
        }];
        out.edges = vec![];
        let mut o2 = out.clone();
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.nodes.len(), 1);
        assert!(v.stickies.is_empty() && v.other_visible_text.is_empty());
        assert_eq!(
            v.issues
                .iter()
                .filter(|i| i.kind == IssueKind::DuplicateAcrossLists)
                .count(),
            2
        );
        // A participant name in both owner tags and other text stays an owner tag.
        o2.other_visible_text.push(TextItem {
            text: "AVERY".into(),
            bbox: bb(500.0, 500.0),
        });
        let v = validate_board(o2, CANVAS, &cfg());
        assert_eq!(v.owner_tags.len(), 1);
        assert!(v.other_visible_text.is_empty());
    }

    #[test]
    fn dangling_edges_and_self_loops_are_dropped_never_creating_nodes() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "Queue", 400.0, 100.0),
        ];
        out.edges = vec![
            edge("n1", "n9", ""),
            edge("n2", "n2", ""),
            edge("n2", "n1", ""),
        ];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.nodes.len(), 2);
        assert_eq!(v.edges.len(), 1);
        assert!(v.issues.iter().any(|i| i.kind == IssueKind::DanglingEdge));
        assert!(v.issues.iter().any(|i| i.kind == IssueKind::SelfLoop));
    }

    #[test]
    fn edge_to_rejected_node_is_dropped() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "Share", 400.0, 100.0),
        ];
        out.edges = vec![edge("n1", "n2", "")];
        let v = validate_board(out, CANVAS, &cfg());
        assert!(v.edges.is_empty());
    }

    #[test]
    fn edge_label_rules() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "Queue", 400.0, 100.0),
            node("n3", "Store", 700.0, 100.0),
        ];
        out.edges = vec![
            edge("n1", "n2", "one two three four five six seven"),
            edge("n2", "n3", "Widget Service"),
            edge("n1", "n3", " one two three four five six "),
        ];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.edges.len(), 3);
        assert_eq!(v.edges[0].label, "");
        assert_eq!(v.edges[1].label, "");
        assert_eq!(v.edges[2].label, "one two three four five six");
        assert!(v.issues.iter().any(|i| i.kind == IssueKind::LabelTooLong));
        assert!(v
            .issues
            .iter()
            .any(|i| i.kind == IssueKind::LabelIsNodeText));
    }

    #[test]
    fn duplicate_ids_and_bad_boxes_and_owner_anchor() {
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n1", "Other Box", 400.0, 100.0),
            BoardNode {
                bbox: BBox::new(50.0, 50.0, 40.0, 60.0),
                ..node("n3", "Inverted", 0.0, 0.0)
            },
        ];
        out.owner_tags = vec![OwnerTag {
            name_raw: "Jordan".into(),
            near: "n7".into(),
            bbox: bb(300.0, 300.0),
        }];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.nodes.len(), 1);
        assert_eq!(v.owner_tags[0].near, "");
        let kinds: Vec<IssueKind> = v.issues.iter().map(|i| i.kind).collect();
        assert!(kinds.contains(&IssueKind::DuplicateNodeId));
        assert!(kinds.contains(&IssueKind::MalformedBBox));
        assert!(kinds.contains(&IssueKind::OwnerNearMissing));
    }

    #[test]
    fn zero_confidence_or_empty_triggers_reclassification() {
        let v = validate_board(empty_output(), CANVAS, &cfg());
        assert!(v.needs_reclassification);
        let mut out = empty_output();
        out.nodes = vec![node("n1", "Widget Service", 100.0, 100.0)];
        out.confidence = 0.0;
        assert!(validate_board(out, CANVAS, &cfg()).needs_reclassification);
    }

    #[test]
    fn board_schema_is_inlined_and_bounded() -> Result<()> {
        let schema = crate::schema::OutputSchema::for_type::<BoardReadOutput>()?;
        let text = schema.text();
        assert!(!text.contains("$ref"), "{text}");
        assert_eq!(schema.json()["properties"]["nodes"]["maxItems"], 60);
        assert_eq!(
            schema.json()["properties"]["edges"]["items"]["additionalProperties"],
            serde_json::json!(false)
        );
        Ok(())
    }

    #[test]
    fn prompt_has_no_change_or_previous_blocks() {
        let p = BOARD_READ_PROMPT.to_lowercase();
        assert!(!p.contains("previous"));
        assert!(!p.contains("changes_vs"));
        assert!(!p.contains("transcript"));
        assert!(p.contains("bbox"));
    }
}
