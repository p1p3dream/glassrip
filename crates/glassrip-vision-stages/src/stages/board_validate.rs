//! `board_validate`: schema-checked readings through semantic rules (spec 6.11).
//!
//! Readings were already validated against the JSON schema and the Rust types
//! when they came back from the model. This stage adds:
//!
//! 1. Pixel-aware list membership, measured on the masked canvas crop: an
//!    outlined light box is a node, a filled colored square is a sticky, and a
//!    green square holding only a short name is an owner tag. Elements the model
//!    put in the wrong list are moved.
//! 2. The shared non-pixel rules (`glassrip_vision::board::validate_board`) with
//!    the participant list from OCR tile labels (full names and first names).
//! 3. A node whose text is a near spelling of a participant name is rejected.
//! 4. An edge label that repeats a sticky's text is cleared.
//!
//! Deferred to `edge_direction`: checking that an edge label lies on the edge path.

use std::collections::HashMap;

use glassrip_core::envelope::ErrorInfo;
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use glassrip_vision::board::{
    normalize, validate_board, BoardNode, BoardReadOutput, BoardValidationConfig, CanvasSize,
    ElementList, OwnerTag, RejectReason, RejectedItem, Sticky, StickyColor,
};
use glassrip_vision::BBox;
use image::RgbImage;
use schemars::JsonSchema;
use serde::Serialize;

use crate::artifacts::{
    self, BoardReadingItem, BoardValidateItem, CanvasCropItem, CanvasMethod, ExtraIssue,
    MemberList, MembershipDecision, OcrKeyframe, ShapeClass,
};
use crate::layout::names_match;
use crate::pixels::{self, ShapeThresholds};
use crate::stages::board_read::canvas_image;
use crate::stages::vocabulary::participants;
use crate::stages::{input, internal, load_rgb};

/// Parameters.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BoardValidateParams {
    pub validation: BoardValidationConfig,
    pub shape: ShapeThresholds,
    /// A tile name must be seen in this many keyframes to count as a participant.
    pub participant_min_keyframes: u32,
    /// Node text at least this similar to a participant alias is rejected.
    pub participant_similarity: f64,
    /// Edge label at least this similar to a sticky text is cleared.
    pub label_sticky_similarity: f64,
}

impl Default for BoardValidateParams {
    fn default() -> Self {
        Self {
            validation: BoardValidationConfig {
                denylist: crate::layout::meeting_denylist(),
                ..BoardValidationConfig::default()
            },
            shape: ShapeThresholds::default(),
            participant_min_keyframes: 2,
            participant_similarity: 0.8,
            label_sticky_similarity: 0.85,
        }
    }
}

/// Participant names plus their first names.
pub fn participant_aliases(names: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for n in names {
        out.push(n.clone());
        if let Some(first) = n.split_whitespace().next() {
            if first.chars().filter(|c| c.is_alphabetic()).count() >= 3 {
                out.push(first.to_string());
            }
        }
    }
    out.sort();
    out.dedup_by(|a, b| normalize(a) == normalize(b));
    out
}

fn short_name(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    (1..=2).contains(&words.len())
        && text.len() <= 24
        && words.iter().all(|w| {
            w.chars()
                .all(|c| c.is_alphabetic() || c == '-' || c == '\'')
        })
}

fn decision(
    text: &str,
    bbox: glassrip_vision::BBox,
    from: MemberList,
    to: MemberList,
    shape: ShapeClass,
    m: &pixels::ElementPixels,
) -> MembershipDecision {
    MembershipDecision {
        text: text.to_string(),
        bbox,
        from,
        to,
        shape,
        fill_rgb: m.fill_rgb,
        fill_saturation: m.saturation,
        outline_fraction: m.outline_fraction,
    }
}

/// Move elements between lists by measured shape and color.
pub fn apply_membership(
    out: BoardReadOutput,
    img: &RgbImage,
    t: &ShapeThresholds,
) -> (BoardReadOutput, Vec<MembershipDecision>) {
    let BoardReadOutput {
        nodes,
        edges,
        stickies,
        owner_tags,
        other_visible_text,
        confidence,
    } = out;
    let mut decisions = Vec::new();
    let mut new_nodes = Vec::new();
    let mut new_stickies = Vec::new();
    let mut new_owners = Vec::new();
    let mut moved_id = 0usize;

    for n in nodes {
        let Some(m) = pixels::measure(img, &n.bbox, t) else {
            new_nodes.push(n);
            continue;
        };
        let shape = pixels::classify_shape(&m, t);
        let to = match shape {
            ShapeClass::FilledSticky => MemberList::Stickies,
            ShapeClass::GreenTag if short_name(&n.text) => MemberList::OwnerTags,
            ShapeClass::GreenTag => MemberList::Stickies,
            _ => MemberList::Nodes,
        };
        decisions.push(decision(&n.text, n.bbox, MemberList::Nodes, to, shape, &m));
        match to {
            MemberList::Nodes => new_nodes.push(n),
            MemberList::Stickies => new_stickies.push(Sticky {
                text: n.text,
                color: pixels::sticky_color(&m),
                bbox: n.bbox,
            }),
            MemberList::OwnerTags => new_owners.push(OwnerTag {
                name_raw: n.text,
                near: String::new(),
                bbox: n.bbox,
            }),
        }
    }
    for s in stickies {
        let Some(m) = pixels::measure(img, &s.bbox, t) else {
            new_stickies.push(s);
            continue;
        };
        let shape = pixels::classify_shape(&m, t);
        let to = match shape {
            ShapeClass::OutlinedBox => MemberList::Nodes,
            ShapeClass::GreenTag if short_name(&s.text) => MemberList::OwnerTags,
            _ => MemberList::Stickies,
        };
        decisions.push(decision(
            &s.text,
            s.bbox,
            MemberList::Stickies,
            to,
            shape,
            &m,
        ));
        match to {
            MemberList::Nodes => {
                moved_id += 1;
                new_nodes.push(BoardNode {
                    local_id: format!("moved{moved_id}"),
                    text: s.text,
                    bbox: s.bbox,
                    conf: 0.5,
                });
            }
            MemberList::OwnerTags => new_owners.push(OwnerTag {
                name_raw: s.text,
                near: String::new(),
                bbox: s.bbox,
            }),
            MemberList::Stickies => {
                let mut s = s;
                if s.color == StickyColor::White && shape == ShapeClass::FilledSticky {
                    s.color = pixels::sticky_color(&m);
                }
                new_stickies.push(s);
            }
        }
    }
    for o in owner_tags {
        let Some(m) = pixels::measure(img, &o.bbox, t) else {
            new_owners.push(o);
            continue;
        };
        let shape = pixels::classify_shape(&m, t);
        let to = match shape {
            ShapeClass::FilledSticky => MemberList::Stickies,
            ShapeClass::OutlinedBox => MemberList::Nodes,
            _ => MemberList::OwnerTags,
        };
        decisions.push(decision(
            &o.name_raw,
            o.bbox,
            MemberList::OwnerTags,
            to,
            shape,
            &m,
        ));
        match to {
            MemberList::OwnerTags => new_owners.push(o),
            MemberList::Stickies => new_stickies.push(Sticky {
                text: o.name_raw,
                color: pixels::sticky_color(&m),
                bbox: o.bbox,
            }),
            MemberList::Nodes => {
                moved_id += 1;
                new_nodes.push(BoardNode {
                    local_id: format!("moved{moved_id}"),
                    text: o.name_raw,
                    bbox: o.bbox,
                    conf: 0.5,
                });
            }
        }
    }
    (
        BoardReadOutput {
            nodes: new_nodes,
            edges,
            stickies: new_stickies,
            owner_tags: new_owners,
            other_visible_text,
            confidence,
        },
        decisions,
    )
}

fn center_in(b: &BBox, boxes: &[BBox]) -> bool {
    let (x, y) = ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0);
    boxes
        .iter()
        .any(|t| x >= t.x1 && x <= t.x2 && y >= t.y1 && y <= t.y2)
}

/// Remove elements centered inside participant tiles (canvas pixels).
fn drop_tile_text(
    mut out: BoardReadOutput,
    tiles: &[BBox],
) -> (BoardReadOutput, Vec<RejectedItem>) {
    let mut rejected = Vec::new();
    let mut reject = |list: ElementList, text: &str, bbox: BBox| {
        rejected.push(RejectedItem {
            list,
            text: text.to_string(),
            bbox: Some(bbox),
            reason: RejectReason::TileRegion,
        });
    };
    out.nodes.retain(|n| {
        let inside = center_in(&n.bbox, tiles);
        if inside {
            reject(ElementList::Nodes, &n.text, n.bbox);
        }
        !inside
    });
    out.stickies.retain(|s| {
        let inside = center_in(&s.bbox, tiles);
        if inside {
            reject(ElementList::Stickies, &s.text, s.bbox);
        }
        !inside
    });
    out.owner_tags.retain(|o| {
        let inside = center_in(&o.bbox, tiles);
        if inside {
            reject(ElementList::OwnerTags, &o.name_raw, o.bbox);
        }
        !inside
    });
    out.other_visible_text.retain(|t| {
        let inside = center_in(&t.bbox, tiles);
        if inside {
            reject(ElementList::OtherVisibleText, &t.text, t.bbox);
        }
        !inside
    });
    (out, rejected)
}

/// Validate one reading against its canvas image.
pub fn validate_reading(
    reading: &BoardReadingItem,
    canvas: &RgbImage,
    participants: &[String],
    p: &BoardValidateParams,
) -> BoardValidateItem {
    let (ox, oy) = (reading.crop_box.x1, reading.crop_box.y1);
    let tiles: Vec<BBox> = reading
        .tiles
        .iter()
        .map(|t| {
            BBox::new(
                t.bbox.x1 - ox,
                t.bbox.y1 - oy,
                t.bbox.x2 - ox,
                t.bbox.y2 - oy,
            )
        })
        .collect();
    let (result, tile_rejects) = drop_tile_text(reading.result.clone(), &tiles);
    let (mut moved, membership) = apply_membership(result, canvas, &p.shape);
    let mut extra = Vec::new();
    // A moved element can repeat one already in its new list.
    let dup = |a: (&str, &glassrip_vision::BBox), b: (&str, &glassrip_vision::BBox)| {
        normalize(a.0) == normalize(b.0) && a.1.iou(b.1) >= 0.5
    };
    let mut kept: Vec<Sticky> = Vec::new();
    for s in std::mem::take(&mut moved.stickies) {
        if kept
            .iter()
            .any(|k| dup((&k.text, &k.bbox), (&s.text, &s.bbox)))
        {
            extra.push(ExtraIssue {
                kind: "duplicate_in_list".into(),
                detail: format!("sticky {:?} listed twice", s.text),
            });
        } else {
            kept.push(s);
        }
    }
    moved.stickies = kept;
    let mut kept: Vec<OwnerTag> = Vec::new();
    for o in std::mem::take(&mut moved.owner_tags) {
        if kept
            .iter()
            .any(|k| dup((&k.name_raw, &k.bbox), (&o.name_raw, &o.bbox)))
        {
            extra.push(ExtraIssue {
                kind: "duplicate_in_list".into(),
                detail: format!("owner tag {:?} listed twice", o.name_raw),
            });
        } else {
            kept.push(o);
        }
    }
    moved.owner_tags = kept;
    let all: Vec<String> = participants
        .iter()
        .chain(&p.validation.participant_names)
        .cloned()
        .collect();
    let aliases = participant_aliases(&all);
    let mut cfg = p.validation.clone();
    cfg.participant_names.extend(aliases.iter().cloned());
    for n in &moved.nodes {
        let t = normalize(&n.text);
        let near = aliases.iter().find(|a| {
            let a = normalize(a);
            a != t && strsim::normalized_levenshtein(&a, &t) >= p.participant_similarity
        });
        if let Some(a) = near {
            extra.push(ExtraIssue {
                kind: "node_is_participant_name".into(),
                detail: format!("node {:?} is close to participant {a:?}", n.text),
            });
            cfg.participant_names.push(n.text.clone());
        }
    }
    let canvas_size = CanvasSize {
        width: reading.crop_box.width(),
        height: reading.crop_box.height(),
    };
    let mut board = validate_board(moved, canvas_size, &cfg);
    board.chrome_rejected.extend(tile_rejects);
    // Owner tags: a participant's name (when participants are known) on a green tag.
    let mut owners = Vec::new();
    for o in std::mem::take(&mut board.owner_tags) {
        let named = aliases.is_empty() || aliases.iter().any(|a| names_match(a, &o.name_raw));
        let on_tag = pixels::measure(canvas, &o.bbox, &p.shape)
            .is_some_and(|m| pixels::classify_shape(&m, &p.shape) == ShapeClass::GreenTag);
        let reason = if !named {
            Some(RejectReason::OwnerNotParticipant)
        } else if !on_tag {
            Some(RejectReason::OwnerNotOnTag)
        } else {
            None
        };
        match reason {
            Some(reason) => board.chrome_rejected.push(RejectedItem {
                list: ElementList::OwnerTags,
                text: o.name_raw.clone(),
                bbox: Some(o.bbox),
                reason,
            }),
            None => owners.push(o),
        }
    }
    board.owner_tags = owners;
    let sticky_texts: Vec<String> = board.stickies.iter().map(|s| normalize(&s.text)).collect();
    for e in &mut board.edges {
        let l = normalize(&e.label);
        if l.len() < 4 {
            continue;
        }
        let hit = sticky_texts.iter().any(|s| {
            s.contains(&l) || strsim::normalized_levenshtein(s, &l) >= p.label_sticky_similarity
        });
        if hit {
            extra.push(ExtraIssue {
                kind: "label_is_sticky_text".into(),
                detail: format!("label {:?} on {} -> {} cleared", e.label, e.src, e.dst),
            });
            e.label.clear();
        }
    }
    BoardValidateItem {
        keyframe_id: reading.keyframe_id.clone(),
        canvas_width: canvas_size.width,
        canvas_height: canvas_size.height,
        needs_reclassification: board.needs_reclassification,
        board,
        membership,
        extra_issues: extra,
    }
}

/// The stage.
pub struct BoardValidateStage {
    params: BoardValidateParams,
}

impl BoardValidateStage {
    pub fn new(params: BoardValidateParams) -> Self {
        Self { params }
    }
}

/// Work: the reading plus the run's participants.
#[derive(Debug, Clone)]
pub struct ValidateWork {
    reading: BoardReadingItem,
    participants: Vec<String>,
}

impl Stage for BoardValidateStage {
    type Params = BoardValidateParams;
    type Work = ValidateWork;
    type Output = BoardValidateItem;

    fn name(&self) -> &'static str {
        "board_validate"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: artifacts::BOARD_VALIDATE,
            version: artifacts::output_version(),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![input(artifacts::BOARD_READING), input(artifacts::OCR)]
    }
    fn params(&self) -> &BoardValidateParams {
        &self.params
    }
    fn concurrency(&self) -> usize {
        4
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<ValidateWork>>, StageError> {
        let ocr: Vec<OcrKeyframe> = inputs
            .read_ok::<OcrKeyframe>(artifacts::OCR)?
            .into_iter()
            .map(|(_, o)| o)
            .collect();
        let names: Vec<String> = participants(&ocr, self.params.participant_min_keyframes)
            .into_iter()
            .map(|(n, _, _)| n)
            .collect();
        let mut by_id: HashMap<String, BoardReadingItem> = HashMap::new();
        let mut order = Vec::new();
        for (id, r) in inputs.read_ok::<BoardReadingItem>(artifacts::BOARD_READING)? {
            order.push(id.clone());
            by_id.insert(id, r);
        }
        Ok(order
            .into_iter()
            .filter_map(|id| {
                let reading = by_id.remove(&id)?;
                let mut ps = names.clone();
                ps.extend(reading.participants.iter().cloned());
                Some(WorkItem {
                    id,
                    work: ValidateWork {
                        reading,
                        participants: ps,
                    },
                })
            })
            .collect())
    }

    async fn process(
        &self,
        _ctx: &ItemContext,
        w: ValidateWork,
    ) -> Result<BoardValidateItem, ErrorInfo> {
        let img = load_rgb(w.reading.source_image_path.clone().into()).await?;
        let crop = CanvasCropItem {
            keyframe_id: w.reading.keyframe_id.clone(),
            source_frame_id: w.reading.source_frame_id.clone(),
            source_image_path: w.reading.source_image_path.clone(),
            source_image_blake3: w.reading.source_image_blake3.clone(),
            image_width: img.width(),
            image_height: img.height(),
            canvas_bbox: w.reading.crop_box,
            raw_canvas_bbox: w.reading.crop_box,
            method: CanvasMethod::Layout,
            segment: 0,
            stabilized: false,
            share_area: None,
            tiles: Vec::new(),
            masks: w.reading.masks.clone(),
            span_regions: Vec::new(),
            canvas_text_height_px: None,
            participants: Vec::new(),
        };
        let params = self.params.clone();
        tokio::task::spawn_blocking(move || {
            let canvas = canvas_image(&img, &crop)
                .ok_or_else(|| crate::stages::invalid("canvas box is empty"))?;
            Ok(validate_reading(
                &w.reading,
                &canvas,
                &w.participants,
                &params,
            ))
        })
        .await
        .map_err(|e| internal(format!("validation task failed: {e}")))?
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use glassrip_vision::board::{BoardEdge, EdgeStyle};
    use glassrip_vision::BBox;
    use image::Rgb;

    fn fill(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, c: [u8; 3]) {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, Rgb(c));
            }
        }
    }

    /// Synthetic canvas: an outlined node, a yellow sticky, a green name tag.
    fn board_image() -> RgbImage {
        let mut img = RgbImage::from_pixel(400, 300, Rgb([248, 248, 248]));
        fill(&mut img, 20, 20, 140, 80, [40, 40, 40]);
        fill(&mut img, 23, 23, 137, 77, [255, 255, 255]);
        fill(&mut img, 200, 20, 300, 100, [252, 232, 110]);
        fill(&mut img, 200, 150, 260, 190, [100, 200, 110]);
        fill(&mut img, 20, 150, 140, 210, [40, 40, 40]);
        fill(&mut img, 23, 153, 137, 207, [255, 255, 255]);
        img
    }

    fn reading(result: BoardReadOutput) -> BoardReadingItem {
        BoardReadingItem {
            keyframe_id: "k".into(),
            source_frame_id: "f".into(),
            source_image_path: String::new(),
            source_image_blake3: None,
            crop_box: BBox::new(0.0, 0.0, 400.0, 300.0),
            masks: vec![],
            tiles: vec![],
            model: crate::artifacts::ModelRef {
                name: "m".into(),
                digest: None,
            },
            latency_s: 0.0,
            low_res: false,
            token_capped: false,
            tiled: false,
            requests: vec![],
            participants: vec![],
            result,
        }
    }

    #[test]
    fn misfiled_elements_move_by_shape_and_color() {
        let out = BoardReadOutput {
            nodes: vec![
                BoardNode {
                    local_id: "n1".into(),
                    text: "Order Service".into(),
                    bbox: BBox::new(20.0, 20.0, 140.0, 80.0),
                    conf: 0.9,
                },
                // Sticky read as a node (trap: sticky duplicated as node).
                BoardNode {
                    local_id: "n2".into(),
                    text: "Ship before launch?".into(),
                    bbox: BBox::new(200.0, 20.0, 300.0, 100.0),
                    conf: 0.9,
                },
                // Owner tag read as a node with a misspelled name.
                BoardNode {
                    local_id: "n3".into(),
                    text: "Adeline".into(),
                    bbox: BBox::new(200.0, 150.0, 260.0, 190.0),
                    conf: 0.9,
                },
                BoardNode {
                    local_id: "n4".into(),
                    text: "Ledger".into(),
                    bbox: BBox::new(20.0, 150.0, 140.0, 210.0),
                    conf: 0.9,
                },
            ],
            edges: vec![BoardEdge {
                src: "n1".into(),
                dst: "n4".into(),
                label: "Ship before launch?".into(),
                style: EdgeStyle::Solid,
                conf: 0.9,
            }],
            stickies: vec![Sticky {
                text: "Ship before launch?".into(),
                color: StickyColor::Yellow,
                bbox: BBox::new(200.0, 20.0, 300.0, 100.0),
            }],
            owner_tags: vec![],
            other_visible_text: vec![],
            confidence: 0.8,
        };
        let r = reading(out);
        let v = validate_reading(
            &r,
            &board_image(),
            &["Adaline Quill".to_string()],
            &BoardValidateParams::default(),
        );
        let nodes: Vec<&str> = v.board.nodes.iter().map(|n| n.text.as_str()).collect();
        assert_eq!(nodes, vec!["Order Service", "Ledger"], "{:?}", v.membership);
        assert_eq!(v.board.stickies.len(), 1);
        assert_eq!(v.board.owner_tags.len(), 1);
        assert_eq!(v.board.owner_tags[0].name_raw, "Adeline");
        assert_eq!(v.board.edges.len(), 1);
        assert!(v.board.edges[0].label.is_empty());
        assert!(v
            .extra_issues
            .iter()
            .any(|e| e.kind == "label_is_sticky_text"));
    }

    #[test]
    fn tile_text_and_unsupported_owners_are_chrome() {
        let out = BoardReadOutput {
            nodes: vec![BoardNode {
                local_id: "n1".into(),
                text: "Order Service".into(),
                bbox: BBox::new(20.0, 20.0, 140.0, 80.0),
                conf: 0.9,
            }],
            edges: vec![],
            stickies: vec![],
            owner_tags: vec![
                // On the green tag and a participant: kept.
                OwnerTag {
                    name_raw: "Ada".into(),
                    near: String::new(),
                    bbox: BBox::new(200.0, 150.0, 260.0, 190.0),
                },
                // On the green tag but not a participant.
                OwnerTag {
                    name_raw: "Kora".into(),
                    near: String::new(),
                    bbox: BBox::new(200.0, 150.0, 260.0, 190.0),
                },
                // A participant, but on plain canvas.
                OwnerTag {
                    name_raw: "Bo Tran".into(),
                    near: String::new(),
                    bbox: BBox::new(300.0, 220.0, 380.0, 260.0),
                },
                // Inside a video tile.
                OwnerTag {
                    name_raw: "Ada Quill".into(),
                    near: String::new(),
                    bbox: BBox::new(330.0, 30.0, 390.0, 50.0),
                },
            ],
            other_visible_text: vec![],
            confidence: 0.8,
        };
        let mut r = reading(out);
        r.tiles = vec![crate::artifacts::TileBox {
            name: "Ada Quill".into(),
            bbox: BBox::new(320.0, 10.0, 400.0, 60.0),
        }];
        let v = validate_reading(
            &r,
            &board_image(),
            &["Ada Quill".to_string(), "Bo Tran".to_string()],
            &BoardValidateParams::default(),
        );
        let owners: Vec<&str> = v
            .board
            .owner_tags
            .iter()
            .map(|o| o.name_raw.as_str())
            .collect();
        assert_eq!(owners, vec!["Ada"], "{:?}", v.board.chrome_rejected);
        let reasons: Vec<RejectReason> = v.board.chrome_rejected.iter().map(|r| r.reason).collect();
        assert!(reasons.contains(&RejectReason::OwnerNotParticipant));
        assert!(reasons.contains(&RejectReason::OwnerNotOnTag));
        assert!(reasons.contains(&RejectReason::TileRegion));
    }

    #[test]
    fn aliases_add_first_names() {
        let a = participant_aliases(&["Ada Quill".into(), "Bo Tran".into()]);
        assert!(a.contains(&"Ada".to_string()));
        assert!(!a.contains(&"Bo".to_string()));
    }
}
