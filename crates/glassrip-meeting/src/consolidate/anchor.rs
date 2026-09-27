//! Geometric anchoring of one owner tag to nodes or an edge.
//!
//! Rules, in order (sizes are relative to the tag's larger side):
//!
//! 1. **Overlap.** A tag covering at least `overlap_share` of its own area with a node
//!    box belongs to that node.
//! 2. **On a connector.** A tag the connector segment passes through belongs to that
//!    edge, unless it clearly sits beside one end's box (within `beside_share` sizes of
//!    it, at least `far_ratio` times and half a size farther from the other end); then
//!    it belongs to that node.
//! 3. **In the gap of a link.** A tag intersecting the gap between the facing sides of
//!    two connected boxes (widened by `gap_margin_share` sizes across the link) belongs
//!    to that edge (the gap whose center is nearest the
//!    tag's when several touch it), with the same beside exception.
//! 4. **Bridging two nodes.** A tag adjacent to two boxes joined by a connector
//!    (within `bridge_share` sizes of each) whose center lies between them belongs to
//!    both nodes. Without a connector between them geometry gives no answer.
//! 5. **Nearest node.** Otherwise the unique nearest box within `node_range_share`
//!    sizes.
//!
//! [`edge_end_ambiguities`] reports where the answer is ambiguous between an edge and
//! one of its end nodes: the tag touches the connector (or the link's gap) and sits
//! within `beside_share` sizes of that end.

use glassrip_vision::BBox;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A line segment.
pub type Segment = ((f64, f64), (f64, f64));

/// Anchoring thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnchorParams {
    /// Share of the tag area that must overlap a node box (rule 1).
    pub overlap_share: f64,
    /// Beside distance in tag sizes (rules 2 and 3).
    pub beside_share: f64,
    /// How much farther the other end must be (rules 2 and 3).
    pub far_ratio: f64,
    /// Margin around a link's gap, perpendicular to the link, in tag sizes (rule 3):
    /// a tag placed right beside the connector still counts as in the gap.
    pub gap_margin_share: f64,
    /// Adjacency distance in tag sizes (rule 4).
    pub bridge_share: f64,
    /// Nearest-node range in tag sizes (rule 5).
    pub node_range_share: f64,
}

impl Default for AnchorParams {
    fn default() -> Self {
        Self {
            overlap_share: 0.2,
            beside_share: 0.75,
            far_ratio: 2.0,
            gap_margin_share: 0.25,
            bridge_share: 0.5,
            node_range_share: 1.5,
        }
    }
}

/// An edge as seen in one keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeGeom<K> {
    /// Edge key.
    pub key: K,
    /// Box of end `a` and its node key.
    pub a: (K, BBox),
    /// Box of end `b` and its node key.
    pub b: (K, BBox),
    /// Connector segment (pixel termini, or facing border points).
    pub segment: Segment,
}

/// Result of anchoring one tag.
#[derive(Debug, Clone, PartialEq)]
pub enum Anchored<K> {
    /// One node (rules 1, 2/3 beside, 5).
    Node(K),
    /// Two nodes (rule 4).
    Bridge(K, K),
    /// An edge (rules 2, 3).
    Edge(K),
}

fn inter_area(a: &BBox, b: &BBox) -> f64 {
    let w = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let h = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    w * h
}

/// Distance between two boxes (0 when they touch or overlap).
pub fn box_distance(a: &BBox, b: &BBox) -> f64 {
    let dx = (a.x1 - b.x2).max(b.x1 - a.x2).max(0.0);
    let dy = (a.y1 - b.y2).max(b.y1 - a.y2).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

fn inside(p: (f64, f64), r: &BBox) -> bool {
    p.0 >= r.x1 && p.0 <= r.x2 && p.1 >= r.y1 && p.1 <= r.y2
}

fn segments_cross(p: (f64, f64), q: (f64, f64), r: (f64, f64), s: (f64, f64)) -> bool {
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let (d1, d2) = (cross(r, s, p), cross(r, s, q));
    let (d3, d4) = (cross(p, q, r), cross(p, q, s));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// True when the segment passes through the box.
pub fn segment_hits_box(seg: Segment, r: &BBox) -> bool {
    let (p, q) = seg;
    if inside(p, r) || inside(q, r) {
        return true;
    }
    let c = [(r.x1, r.y1), (r.x2, r.y1), (r.x2, r.y2), (r.x1, r.y2)];
    (0..4).any(|i| segments_cross(p, q, c[i], c[(i + 1) % 4]))
}

/// The gap between the facing sides of two boxes, when they face each other, and
/// whether the boxes are stacked (so the link between them runs vertically).
pub fn gap_region(a: &BBox, b: &BBox) -> Option<(BBox, bool)> {
    let (y1, y2) = (a.y1.max(b.y1), a.y2.min(b.y2));
    let (x1, x2) = (a.x1.max(b.x1), a.x2.min(b.x2));
    if a.x2 <= b.x1 && y2 > y1 {
        return Some((BBox::new(a.x2, y1, b.x1, y2), false));
    }
    if b.x2 <= a.x1 && y2 > y1 {
        return Some((BBox::new(b.x2, y1, a.x1, y2), false));
    }
    if a.y2 <= b.y1 && x2 > x1 {
        return Some((BBox::new(x1, a.y2, x2, b.y1), true));
    }
    if b.y2 <= a.y1 && x2 > x1 {
        return Some((BBox::new(x1, b.y2, x2, a.y1), true));
    }
    None
}

/// The gap of a link widened by `m` across the link: sideways for stacked boxes (a
/// vertical link), above and below for boxes side by side (a horizontal link).
fn widened_gap<K>(e: &EdgeGeom<K>, m: f64) -> Option<BBox> {
    gap_region(&e.a.1, &e.b.1).map(|(g, stacked)| {
        if stacked {
            BBox::new(g.x1 - m, g.y1, g.x2 + m, g.y2)
        } else {
            BBox::new(g.x1, g.y1 - m, g.x2, g.y2 + m)
        }
    })
}

/// Edges a tag touches (its connector passes through the tag, or the tag meets the
/// link's widened gap) while it sits within `beside_share` tag sizes of one end:
/// `(edge key, that end's key)`. Anchoring picks one of the two; both fit.
pub fn edge_end_ambiguities<K: Clone>(
    tag: &BBox,
    edges: &[EdgeGeom<K>],
    p: &AnchorParams,
) -> Vec<(K, K)> {
    let size = tag.width().max(tag.height()).max(1.0);
    edges
        .iter()
        .filter(|e| {
            segment_hits_box(e.segment, tag)
                || widened_gap(e, p.gap_margin_share * size)
                    .is_some_and(|g| inter_area(tag, &g) > 0.0)
        })
        .filter_map(|e| {
            let (da, db) = (box_distance(tag, &e.a.1), box_distance(tag, &e.b.1));
            let (near, dn) = if da <= db { (&e.a.0, da) } else { (&e.b.0, db) };
            (dn <= p.beside_share * size).then(|| (e.key.clone(), near.clone()))
        })
        .collect()
}

fn center(b: &BBox) -> (f64, f64) {
    ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
}

/// Anchor one tag. `nodes` are the node boxes of the keyframe.
pub fn anchor_tag<K: Clone + PartialEq>(
    tag: &BBox,
    nodes: &[(K, BBox)],
    edges: &[EdgeGeom<K>],
    p: &AnchorParams,
) -> Option<Anchored<K>> {
    let size = tag.width().max(tag.height()).max(1.0);
    let area = (tag.width() * tag.height()).max(1.0);
    // 1. Overlap.
    if let Some((k, share)) = nodes
        .iter()
        .map(|(k, b)| (k, inter_area(tag, b) / area))
        .max_by(|x, y| x.1.total_cmp(&y.1))
    {
        if share >= p.overlap_share {
            return Some(Anchored::Node(k.clone()));
        }
    }
    let beside = |e: &EdgeGeom<K>| -> Option<K> {
        let (da, db) = (box_distance(tag, &e.a.1), box_distance(tag, &e.b.1));
        let (near, dn, df) = if da <= db {
            (&e.a.0, da, db)
        } else {
            (&e.b.0, db, da)
        };
        (dn <= p.beside_share * size && df >= p.far_ratio * dn && df - dn >= 0.5 * size)
            .then(|| near.clone())
    };
    let seg_dist = |e: &EdgeGeom<K>| {
        let c = center(tag);
        let ((ax, ay), (bx, by)) = e.segment;
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 {
            (((c.0 - ax) * dx + (c.1 - ay) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        ((c.0 - ax - t * dx).powi(2) + (c.1 - ay - t * dy).powi(2)).sqrt()
    };
    // 2. On a connector.
    if let Some(e) = edges
        .iter()
        .filter(|e| segment_hits_box(e.segment, tag))
        .min_by(|x, y| seg_dist(x).total_cmp(&seg_dist(y)))
    {
        return Some(match beside(e) {
            Some(n) => Anchored::Node(n),
            None => Anchored::Edge(e.key.clone()),
        });
    }
    // 3. In the gap of a link.
    // Several gaps may touch the tag; the one whose center is nearest the tag's wins.
    let gap_dist = |e: &EdgeGeom<K>| -> Option<f64> {
        let g = widened_gap(e, p.gap_margin_share * size).filter(|g| inter_area(tag, g) > 0.0)?;
        let (c, gc) = (center(tag), center(&g));
        Some(((c.0 - gc.0).powi(2) + (c.1 - gc.1).powi(2)).sqrt())
    };
    if let Some(e) = edges
        .iter()
        .filter_map(|e| gap_dist(e).map(|d| (e, d)))
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map(|x| x.0)
    {
        return Some(match beside(e) {
            Some(n) => Anchored::Node(n),
            None => Anchored::Edge(e.key.clone()),
        });
    }
    // 4. Bridging two nodes.
    let mut near: Vec<(&K, &BBox, f64)> = nodes
        .iter()
        .map(|(k, b)| (k, b, box_distance(tag, b)))
        .filter(|x| x.2 <= p.bridge_share * size)
        .collect();
    near.sort_by(|x, y| x.2.total_cmp(&y.2));
    if near.len() >= 2 {
        let (c, a, b) = (center(tag), center(near[0].1), center(near[1].1));
        let between = |t: f64, u: f64, v: f64| (u < t && t < v) || (v < t && t < u);
        if between(c.0, a.0, b.0) || between(c.1, a.1, b.1) {
            // Only boxes joined by a connector form a bridged pair; a tag resting in
            // the seam of two unconnected boxes is ambiguous, so geometry gives up
            // and the caller falls back to the reader's `near`.
            let (x, y) = (near[0].0, near[1].0);
            let linked = edges
                .iter()
                .any(|e| (&e.a.0 == x && &e.b.0 == y) || (&e.a.0 == y && &e.b.0 == x));
            return linked.then(|| Anchored::Bridge(x.clone(), y.clone()));
        }
    }
    // 5. Nearest node.
    let mut d: Vec<(&K, f64)> = nodes
        .iter()
        .map(|(k, b)| (k, box_distance(tag, b)))
        .collect();
    d.sort_by(|x, y| x.1.total_cmp(&y.1));
    let unique = d.len() == 1 || (d.len() > 1 && d[1].1 - d[0].1 > 1.0);
    match d.first() {
        Some((k, dist)) if unique && *dist <= p.node_range_share * size => {
            Some(Anchored::Node((*k).clone()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x1: f64, y1: f64, x2: f64, y2: f64) -> BBox {
        BBox::new(x1, y1, x2, y2)
    }

    // A fictional board: two vertically stacked boxes (top, mid) linked by a vertical
    // connector, a third box (right) linked to mid by a horizontal connector, and a
    // fourth box (far) right of `right`.
    fn board() -> (Vec<(&'static str, BBox)>, Vec<EdgeGeom<&'static str>>) {
        let top = b(540.0, 270.0, 660.0, 400.0);
        let mid = b(540.0, 456.0, 660.0, 605.0);
        let right = b(977.0, 452.0, 1097.0, 600.0);
        let far = b(1182.0, 497.0, 1309.0, 573.0);
        let nodes = vec![("top", top), ("mid", mid), ("right", right), ("far", far)];
        let edges = vec![
            EdgeGeom {
                key: "top-mid",
                a: ("top", top),
                b: ("mid", mid),
                segment: ((600.0, 400.0), (600.0, 456.0)),
            },
            EdgeGeom {
                key: "mid-right",
                a: ("mid", mid),
                b: ("right", right),
                segment: ((660.0, 525.0), (977.0, 525.0)),
            },
            EdgeGeom {
                key: "right-far",
                a: ("right", right),
                b: ("far", far),
                segment: ((1097.0, 535.0), (1182.0, 535.0)),
            },
        ];
        (nodes, edges)
    }

    #[test]
    fn tag_on_a_connector_next_to_one_box_anchors_to_that_box() {
        let (n, e) = board();
        // Sits on the mid-right connector, 59 px from `right`, 153 px from `mid`.
        let got = anchor_tag(
            &b(813.0, 511.0, 918.0, 645.0),
            &n,
            &e,
            &AnchorParams::default(),
        );
        assert_eq!(got, Some(Anchored::Node("right")));
    }

    #[test]
    fn tag_on_the_middle_of_a_connector_anchors_to_the_edge() {
        let (n, e) = board();
        let got = anchor_tag(
            &b(760.0, 500.0, 870.0, 560.0),
            &n,
            &e,
            &AnchorParams::default(),
        );
        assert_eq!(got, Some(Anchored::Edge("mid-right")));
    }

    #[test]
    fn tag_in_the_gap_beside_a_link_anchors_to_the_edge() {
        let (n, e) = board();
        // Beside the top-mid connector, in the gap between the boxes, touching both.
        let got = anchor_tag(
            &b(655.0, 384.0, 760.0, 497.0),
            &n,
            &e,
            &AnchorParams::default(),
        );
        assert_eq!(got, Some(Anchored::Edge("top-mid")));
    }

    #[test]
    fn tag_bridging_two_boxes_above_their_link_anchors_to_both() {
        let (n, e) = board();
        // Above the gap between `right` and `far`, away from their connector.
        let got = anchor_tag(
            &b(1097.0, 370.0, 1201.0, 461.0),
            &n,
            &e,
            &AnchorParams::default(),
        );
        assert_eq!(got, Some(Anchored::Bridge("right", "far")));
    }

    #[test]
    fn overlap_and_nearest_node() {
        let (n, e) = board();
        let p = AnchorParams::default();
        assert_eq!(
            anchor_tag(&b(600.0, 300.0, 700.0, 340.0), &n, &e, &p),
            Some(Anchored::Node("top"))
        );
        assert_eq!(
            anchor_tag(&b(420.0, 200.0, 520.0, 240.0), &n, &e, &p),
            Some(Anchored::Node("top"))
        );
        assert_eq!(anchor_tag(&b(100.0, 900.0, 150.0, 930.0), &n, &e, &p), None);
    }

    #[test]
    fn tag_just_beside_a_links_gap_still_anchors_to_the_edge() {
        let (n, e) = board();
        // 7 px right of the gap between `top` and `mid`, spanning it vertically.
        let got = anchor_tag(
            &b(667.0, 395.0, 720.0, 470.0),
            &n,
            &e,
            &AnchorParams::default(),
        );
        assert_eq!(got, Some(Anchored::Edge("top-mid")));
    }

    #[test]
    fn a_tag_between_two_unconnected_boxes_is_not_a_bridge() {
        let (n, e) = board();
        // `right` and `far` without their connector.
        let e: Vec<_> = e.into_iter().filter(|x| x.key != "right-far").collect();
        let got = anchor_tag(
            &b(1097.0, 370.0, 1201.0, 461.0),
            &n,
            &e,
            &AnchorParams::default(),
        );
        assert_eq!(got, None);
    }
}
