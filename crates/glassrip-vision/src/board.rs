//! Board reading: prompt, schema-constrained output types, and the post-model
//! validation rules that need no pixels.
//!
//! The model sees only the canvas crop and reports every element with a bbox in
//! the pixels of the image it was sent. [`BoardReadOutput::to_canvas_coords`] turns the wire
//! form into a [`BoardReading`] in canvas pixels, the form artifacts store
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
Never use a node's text as an edge label. \"label_bbox_2d\" is the box around the label text as \
[x1, y1, x2, y2], or [0, 0, 0, 0] if the line has no text. \"style\" is \"dashed\" for dotted/dashed lines, otherwise \
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
- bbox_2d (and label_bbox_2d): for every node, sticky, owner tag, other text, and edge label, the box \
around it as \
[x1, y1, x2, y2] in absolute pixel coordinates of this image, in exactly that order: x1, y1 is the \
top-left corner and x2, y2 the bottom-right corner, so x1 < x2 and y1 < y2.

Rules:
- Transcribe text VERBATIM, exactly as written, including punctuation. Do not correct, summarize or \
complete it.
- If text is too blurry, cut off, or too small to read, write \"[illegible]\" for the unreadable part. \
Never guess or invent text.
- List each item once. Only list items you can actually see in THIS image. Use empty lists when \
nothing applies.
- Write the JSON on a single line, without indentation or line breaks.
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

/// A node (box) on the board. Boxes are in canvas pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardNode {
    pub local_id: String,
    pub text: String,
    pub bbox: BBox,
    pub conf: f64,
}

/// A connector between two nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardEdge {
    pub src: String,
    pub dst: String,
    /// Empty string when the line carries no text.
    pub label: String,
    /// Box around the label text; `None` without a label or box.
    #[serde(default)]
    pub label_bbox: Option<BBox>,
    pub style: EdgeStyle,
    pub conf: f64,
}

/// A sticky note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sticky {
    pub text: String,
    pub color: StickyColor,
    pub bbox: BBox,
}

/// An owner tag (a green note holding a name).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OwnerTag {
    /// Name exactly as written on the tag.
    pub name_raw: String,
    /// `local_id` of the node the tag sits on or next to, or empty.
    pub near: String,
    pub bbox: BBox,
}

/// Other canvas text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextItem {
    pub text: String,
    pub bbox: BBox,
}

/// One board reading in canvas pixels (the form stored in artifacts and
/// validated). Boxes serialize as `{x1, y1, x2, y2}` objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardReading {
    pub nodes: Vec<BoardNode>,
    pub edges: Vec<BoardEdge>,
    pub stickies: Vec<Sticky>,
    pub owner_tags: Vec<OwnerTag>,
    pub other_visible_text: Vec<TextItem>,
    pub confidence: f64,
}

/// Wire form of a node: what the model writes (`bbox_2d` array, sent-image pixels).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireNode {
    pub local_id: String,
    pub text: String,
    #[serde(rename = "bbox_2d", with = "crate::geometry::bbox2d")]
    #[schemars(with = "[f64; 4]")]
    pub bbox: BBox,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f64,
}

/// Wire form of an edge; `label_bbox_2d` of `[0, 0, 0, 0]` means no box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireEdge {
    pub src: String,
    pub dst: String,
    pub label: String,
    #[serde(rename = "label_bbox_2d", with = "crate::geometry::opt_bbox2d")]
    #[schemars(with = "[f64; 4]")]
    pub label_bbox: Option<BBox>,
    pub style: EdgeStyle,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f64,
}

/// Wire form of a sticky.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireSticky {
    pub text: String,
    pub color: StickyColor,
    #[serde(rename = "bbox_2d", with = "crate::geometry::bbox2d")]
    #[schemars(with = "[f64; 4]")]
    pub bbox: BBox,
}

/// Wire form of an owner tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireOwnerTag {
    pub name_raw: String,
    pub near: String,
    #[serde(rename = "bbox_2d", with = "crate::geometry::bbox2d")]
    #[schemars(with = "[f64; 4]")]
    pub bbox: BBox,
}

/// Wire form of other canvas text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireText {
    pub text: String,
    #[serde(rename = "bbox_2d", with = "crate::geometry::bbox2d")]
    #[schemars(with = "[f64; 4]")]
    pub bbox: BBox,
}

/// Model output for one board read (the wire schema: Qwen2.5-VL `bbox_2d`
/// arrays in sent-image pixels). Array lengths are bounded to keep the
/// grammar small. Convert with [`BoardReadOutput::to_canvas_coords`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoardReadOutput {
    #[schemars(length(max = 60))]
    pub nodes: Vec<WireNode>,
    #[schemars(length(max = 80))]
    pub edges: Vec<WireEdge>,
    #[schemars(length(max = 60))]
    pub stickies: Vec<WireSticky>,
    #[schemars(length(max = 20))]
    pub owner_tags: Vec<WireOwnerTag>,
    #[schemars(length(max = 20))]
    pub other_visible_text: Vec<WireText>,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub confidence: f64,
}

impl BoardReadOutput {
    /// Map every box from sent-image pixels to canvas pixels.
    pub fn to_canvas_coords(self, prepared: &PreparedImage) -> BoardReading {
        let m = |b: BBox| b.to_source(prepared);
        BoardReading {
            nodes: self
                .nodes
                .into_iter()
                .map(|n| BoardNode {
                    local_id: n.local_id,
                    text: n.text,
                    bbox: m(n.bbox),
                    conf: n.conf,
                })
                .collect(),
            edges: self
                .edges
                .into_iter()
                .map(|e| BoardEdge {
                    src: e.src,
                    dst: e.dst,
                    label: e.label,
                    label_bbox: e.label_bbox.map(m),
                    style: e.style,
                    conf: e.conf,
                })
                .collect(),
            stickies: self
                .stickies
                .into_iter()
                .map(|s| Sticky {
                    text: s.text,
                    color: s.color,
                    bbox: m(s.bbox),
                })
                .collect(),
            owner_tags: self
                .owner_tags
                .into_iter()
                .map(|o| OwnerTag {
                    name_raw: o.name_raw,
                    near: o.near,
                    bbox: m(o.bbox),
                })
                .collect(),
            other_visible_text: self
                .other_visible_text
                .into_iter()
                .map(|t| TextItem {
                    text: t.text,
                    bbox: m(t.bbox),
                })
                .collect(),
            confidence: self.confidence,
        }
    }
}

/// Build the board-read request for an already prepared canvas crop.
pub fn board_read_request(
    prepared: &PreparedImage,
    options: GenerationOptions,
) -> Result<VisionRequest> {
    VisionRequest::for_output::<BoardReadOutput>(BOARD_READ_PROMPT, prepared.image.clone(), options)
}

/// Added to the board prompt for the one retry after a reply stopped at the
/// output limit.
pub const COMPACT_RETRY_NOTE: &str = "\
Your previous answer was cut off at the output limit. Answer again, shorter: write compact JSON \
on one single line with no spaces, indentation, or line breaks between tokens, and list at most \
as many items as the schema below allows, keeping the most legible ones.";

/// Default list budget of the compact retry, as a share of the normal `maxItems`.
pub const COMPACT_RETRY_SCALE: f64 = 0.5;

/// The retry after a truncated board read: the same image and options, the
/// board prompt plus [`COMPACT_RETRY_NOTE`], and the output schema with every
/// list budget scaled by `max_items_scale` (see
/// [`crate::schema::OutputSchema::with_scaled_max_items`]).
pub fn board_read_request_compact(
    prepared: &PreparedImage,
    options: GenerationOptions,
    max_items_scale: f64,
) -> Result<VisionRequest> {
    let schema = crate::schema::OutputSchema::for_type::<BoardReadOutput>()?
        .with_scaled_max_items(max_items_scale)?;
    let prompt = format!("{}\n\n{COMPACT_RETRY_NOTE}", BOARD_READ_PROMPT.trim_end());
    Ok(VisionRequest {
        prompt: crate::schema::prompt_with_schema(&prompt, &schema),
        image: prepared.image.clone(),
        schema,
        options,
        sampling: crate::backend::SamplingOverrides::default(),
        repetition_guard: None,
    })
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
            max_edge_label_words: 8,
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
    /// A node or other text whose text is a participant name.
    ParticipantName,
    /// Inside a detected participant video tile.
    TileRegion,
    /// Owner tag whose name matches no participant.
    OwnerNotParticipant,
    /// Owner tag not on a green tag (pixel check).
    OwnerNotOnTag,
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
    /// Non-finite box, or an inverted box with no unambiguous repair; dropped.
    MalformedBBox,
    /// Inverted box read as `[x1, x2, y1, y2]` or with swapped corners; repaired.
    BBoxRepaired,
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
    fn keep(&mut self, list: ElementList, text: &str, bbox: &mut BBox) -> bool {
        if text.trim().is_empty() {
            self.issue(
                list,
                IssueKind::EmptyText,
                "element with empty text dropped".into(),
            );
            return false;
        }
        if !bbox.is_well_formed() {
            match repair_inverted(bbox, self.canvas, self.cfg.bbox_tolerance_px) {
                Some(fixed) => {
                    self.issue(
                        list,
                        IssueKind::BBoxRepaired,
                        format!("{text:?}: bbox {bbox:?} repaired to {fixed:?}"),
                    );
                    *bbox = fixed;
                }
                None => {
                    self.issue(
                        list,
                        IssueKind::MalformedBBox,
                        format!("{text:?} has bbox {bbox:?}"),
                    );
                    return false;
                }
            }
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

/// Words in an edge label: whitespace-separated tokens with at least one
/// letter or digit ("A + B" is two words).
pub fn label_words(label: &str) -> usize {
    label
        .split_whitespace()
        .filter(|t| t.chars().any(char::is_alphanumeric))
        .count()
}

/// Repair an inverted box when exactly one reading of its four numbers gives a
/// well-formed box inside the canvas. Readings tried: `[x1, x2, y1, y2]`
/// (axis-order swap), `[x2, y2, x1, y1]` (swapped corners), and either axis
/// flipped. Returns `None` for non-finite values or zero or several candidates.
pub fn repair_inverted(b: &BBox, canvas: CanvasSize, tolerance: f64) -> Option<BBox> {
    if ![b.x1, b.y1, b.x2, b.y2].iter().all(|v| v.is_finite()) || b.is_well_formed() {
        return None;
    }
    let (p, q, r, s) = (b.x1, b.y1, b.x2, b.y2);
    let mut found: Vec<BBox> = Vec::new();
    for c in [
        BBox::new(p, r, q, s),
        BBox::new(r, s, p, q),
        BBox::new(r, q, p, s),
        BBox::new(p, s, r, q),
    ] {
        if c.is_well_formed()
            && c.is_inside(canvas.width, canvas.height, tolerance)
            && !found.contains(&c)
        {
            found.push(c);
        }
    }
    match found.as_slice() {
        [one] => Some(*one),
        _ => None,
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
    output: BoardReading,
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
    let BoardReading {
        nodes,
        edges,
        stickies,
        owner_tags: owner_tags_in,
        other_visible_text,
        confidence,
    } = output;

    // Rules 1-3 for nodes.
    let mut seen_ids = HashSet::new();
    let mut kept_nodes = Vec::new();
    for mut n in nodes {
        if !ctx.keep(ElementList::Nodes, &n.text, &mut n.bbox) {
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

    let mut owner_tags: Vec<OwnerTag> = Vec::new();
    for mut o in owner_tags_in {
        if ctx.keep(ElementList::OwnerTags, &o.name_raw, &mut o.bbox) {
            owner_tags.push(o);
        }
    }

    let mut kept_stickies = Vec::new();
    for mut s in stickies {
        if !ctx.keep(ElementList::Stickies, &s.text, &mut s.bbox) {
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

    let mut other: Vec<TextItem> = Vec::new();
    for mut t in other_visible_text {
        if !ctx.keep(ElementList::OtherVisibleText, &t.text, &mut t.bbox) {
            continue;
        }
        // Participant names outside owner tags are tile or banner text.
        if ctx.is_participant(&t.text) {
            ctx.reject(
                ElementList::OtherVisibleText,
                &t.text,
                Some(t.bbox),
                RejectReason::ParticipantName,
            );
            continue;
        }
        other.push(t);
    }

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
            let kind = if label_words(&label) > cfg.max_edge_label_words {
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
        e.label_bbox = if e.label.is_empty() {
            None
        } else {
            e.label_bbox.and_then(|b| {
                let fixed = if b.is_well_formed() {
                    Some(b)
                } else {
                    repair_inverted(&b, ctx.canvas, cfg.bbox_tolerance_px)
                };
                match fixed {
                    Some(f)
                        if f.is_inside(
                            ctx.canvas.width,
                            ctx.canvas.height,
                            cfg.bbox_tolerance_px,
                        ) =>
                    {
                        if f != b {
                            ctx.issue(
                                ElementList::Edges,
                                IssueKind::BBoxRepaired,
                                format!("label {:?}: bbox {b:?} repaired to {f:?}", e.label),
                            );
                        }
                        Some(f)
                    }
                    _ => {
                        ctx.issue(
                            ElementList::Edges,
                            IssueKind::MalformedBBox,
                            format!("label {:?}: bbox {b:?} dropped", e.label),
                        );
                        None
                    }
                }
            })
        };
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
            label_bbox: None,
            style: EdgeStyle::Solid,
            conf: 0.8,
        }
    }

    fn empty_output() -> BoardReading {
        BoardReading {
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
    fn bbox_2d_wire_format() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let n = WireNode {
            local_id: "n1".into(),
            text: "Widget Service".into(),
            bbox: BBox::new(100.0, 200.0, 220.0, 260.0),
            conf: 0.9,
        };
        let v = serde_json::to_value(&n)?;
        assert_eq!(
            v["bbox_2d"],
            serde_json::json!([100.0, 200.0, 220.0, 260.0])
        );
        assert!(v.get("bbox").is_none());
        let back: WireNode = serde_json::from_value(v)?;
        assert_eq!(back, n);
        let short =
            serde_json::json!({"local_id": "n", "text": "t", "bbox_2d": [1, 2, 3], "conf": 0.5});
        assert!(serde_json::from_value::<WireNode>(short).is_err());
        // Artifacts (the domain form) use box objects and null for no label box.
        let d = serde_json::to_value(node("n1", "Widget Service", 100.0, 200.0))?;
        assert_eq!(d["bbox"]["x2"], serde_json::json!(220.0));
        assert!(d.get("bbox_2d").is_none());
        let e = serde_json::to_value(edge("n1", "n2", ""))?;
        assert!(e["label_bbox"].is_null());
        let schema = crate::schema::OutputSchema::for_type::<BoardReadOutput>()?;
        let text = schema.text();
        assert!(
            text.contains("bbox_2d") && !text.contains("\"x1\""),
            "{text}"
        );
        assert!(BOARD_READ_PROMPT.contains("[x1, y1, x2, y2]"));
        // Properties reach the grammar in declaration order: nodes before edges
        // (edges reference node ids), and an element's text before its box.
        let keys: Vec<&String> = schema.json()["properties"]
            .as_object()
            .map(|m| m.keys().collect())
            .unwrap_or_default();
        assert_eq!(
            keys,
            [
                "nodes",
                "edges",
                "stickies",
                "owner_tags",
                "other_visible_text",
                "confidence"
            ]
        );
        let node_keys: Vec<&String> = schema.json()["properties"]["nodes"]["items"]["properties"]
            .as_object()
            .map(|m| m.keys().collect())
            .unwrap_or_default();
        assert_eq!(node_keys, ["local_id", "text", "bbox_2d", "conf"]);
        Ok(())
    }

    #[test]
    fn edge_label_words_and_label_boxes() {
        assert_eq!(
            label_words("Links between Quarry content + Frontend component"),
            6
        );
        assert_eq!(label_words("a - b"), 2);
        let mut out = empty_output();
        out.nodes = vec![
            node("n1", "Widget Service", 100.0, 100.0),
            node("n2", "Queue", 900.0, 100.0),
            node("n3", "Store", 100.0, 600.0),
        ];
        let mut long = edge(
            "n1",
            "n2",
            "Links between Quarry content + Frontend component, v2 draft notes",
        );
        long.label_bbox = Some(BBox::new(300.0, 90.0, 700.0, 110.0));
        let mut kept = edge(
            "n1",
            "n3",
            "Links between Quarry content + Frontend component",
        );
        // Inverted label box, repairable (axis-order swap).
        kept.label_bbox = Some(BBox::new(300.0, 1000.0, 400.0, 420.0));
        let mut none = edge("n2", "n3", "");
        none.label_bbox = Some(BBox::new(1.0, 1.0, 5.0, 5.0));
        out.edges = vec![long, kept, none];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.edges.len(), 3);
        assert!(v.edges[0].label.is_empty() && v.edges[0].label_bbox.is_none());
        assert_eq!(
            v.edges[1].label,
            "Links between Quarry content + Frontend component"
        );
        assert_eq!(
            v.edges[1].label_bbox,
            Some(BBox::new(300.0, 400.0, 1000.0, 420.0))
        );
        assert!(v.edges[2].label_bbox.is_none());
        // Wire format: [0, 0, 0, 0] means no label box.
        let e: WireEdge = serde_json::from_value(serde_json::json!({
            "src": "n1", "dst": "n2", "label": "", "label_bbox_2d": [0, 0, 0, 0],
            "style": "solid", "conf": 0.5
        }))
        .map_err(|e| e.to_string())
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(e.label_bbox.is_none());
    }

    #[test]
    fn inverted_boxes_repaired_only_when_unambiguous() {
        // Emitted as [x1, x2, y1, y2] for x 600..1000, y 100..300: only one reading fits.
        let axis_swap = BBox::new(600.0, 1000.0, 100.0, 300.0);
        assert_eq!(
            repair_inverted(&axis_swap, CANVAS, 2.0),
            Some(BBox::new(600.0, 100.0, 1000.0, 300.0))
        );
        // Only the x axis flipped.
        assert_eq!(
            repair_inverted(&BBox::new(50.0, 50.0, 40.0, 60.0), CANVAS, 2.0),
            Some(BBox::new(40.0, 50.0, 50.0, 60.0))
        );
        // Scrambled beyond one reading (two fit): rejected.
        let scrambled = BBox::new(337.0, 467.0, 259.0, 337.0);
        assert_eq!(repair_inverted(&scrambled, CANVAS, 2.0), None);
        // Well-formed and non-finite boxes are not candidates.
        assert_eq!(repair_inverted(&bb(1.0, 1.0), CANVAS, 2.0), None);
        assert_eq!(
            repair_inverted(&BBox::new(f64::NAN, 0.0, 1.0, 1.0), CANVAS, 2.0),
            None
        );

        let mut out = empty_output();
        out.nodes = vec![
            BoardNode {
                bbox: axis_swap,
                ..node("n1", "Widget Service", 0.0, 0.0)
            },
            BoardNode {
                bbox: scrambled,
                ..node("n2", "Queue", 0.0, 0.0)
            },
        ];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.nodes.len(), 1);
        assert!(v.nodes[0].bbox.is_well_formed());
        assert!(v.nodes.iter().all(|n| n.bbox.is_well_formed()));
        assert!(v.issues.iter().any(|i| i.kind == IssueKind::BBoxRepaired));
        assert!(v.issues.iter().any(|i| i.kind == IssueKind::MalformedBBox));
    }

    #[test]
    fn participant_names_in_other_text_are_chrome() {
        let mut out = empty_output();
        out.nodes = vec![node("n1", "Widget Service", 100.0, 100.0)];
        out.other_visible_text = vec![
            TextItem {
                text: "Jordan".into(),
                bbox: bb(500.0, 500.0),
            },
            TextItem {
                text: "v2 draft".into(),
                bbox: bb(700.0, 500.0),
            },
        ];
        let v = validate_board(out, CANVAS, &cfg());
        assert_eq!(v.other_visible_text.len(), 1);
        assert!(v
            .chrome_rejected
            .iter()
            .any(|r| r.text == "Jordan" && r.reason == RejectReason::ParticipantName));
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
            edge("n1", "n2", "one two three four five six seven eight nine"),
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
                bbox: BBox::new(f64::NAN, 50.0, 40.0, 60.0),
                ..node("n3", "Not a number", 0.0, 0.0)
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
    fn prompt_asks_for_single_line_json() {
        assert!(BOARD_READ_PROMPT.contains("single line, without indentation or line breaks"));
        assert!(COMPACT_RETRY_NOTE.contains("one single line"));
    }

    #[test]
    fn compact_retry_request_halves_list_budgets() -> Result<()> {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(64, 48));
        let prepared = crate::image_prep::prepare_board_image(&img)?;
        let options = GenerationOptions {
            seed: 3,
            num_predict: 1500,
        };
        let normal = board_read_request(&prepared, options)?;
        let compact = board_read_request_compact(&prepared, options, COMPACT_RETRY_SCALE)?;
        let props = &compact.schema.json()["properties"];
        assert_eq!(props["nodes"]["maxItems"], 30);
        assert_eq!(props["edges"]["maxItems"], 40);
        assert_eq!(props["owner_tags"]["maxItems"], 10);
        // Box tuples keep their fixed length.
        assert_eq!(
            props["nodes"]["items"]["properties"]["bbox_2d"]["maxItems"],
            4
        );
        assert!(compact.prompt.starts_with(BOARD_READ_PROMPT.trim_end()));
        assert!(compact.prompt.contains(COMPACT_RETRY_NOTE));
        assert!(compact.prompt.contains("\"maxItems\":30"));
        assert_ne!(compact.prompt, normal.prompt);
        assert_eq!(compact.options, normal.options);
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
