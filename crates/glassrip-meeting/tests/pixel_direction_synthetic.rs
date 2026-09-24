//! Pixel check on rendered synthetic boards.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::Canvas;
use glassrip_meeting::direction::EndVerdict;
use glassrip_meeting::pixel_direction::{
    bgr_from_rgb, EdgeQuery, PixelCheckParams, PixelStatus, PreparedCanvas,
};
use glassrip_vision::board::EdgeStyle;
use glassrip_vision::BBox;

struct Case {
    canvas: Canvas,
    nodes: Vec<BBox>,
    texts: Vec<BBox>,
}

fn run(case: &Case, q: &EdgeQuery) -> glassrip_meeting::pixel_direction::PixelEvidence {
    let prepared = PreparedCanvas::new(
        &bgr_from_rgb(&case.canvas.img),
        &case.nodes,
        &case.texts,
        &PixelCheckParams::default(),
    );
    prepared.check(q)
}

fn a_box() -> BBox {
    BBox::new(80.0, 120.0, 220.0, 190.0)
}
fn b_box() -> BBox {
    BBox::new(480.0, 120.0, 620.0, 190.0)
}

fn horizontal(head_at_a: bool, head_at_b: bool, labelled: bool, dashed: bool) -> (Case, EdgeQuery) {
    let mut c = Canvas::new(720, 420);
    let (a, b) = (a_box(), b_box());
    c.node(a);
    c.node(b);
    c.connector(
        &[(a.x2, 155.0), (b.x1, 155.0)],
        head_at_a,
        head_at_b,
        dashed,
    );
    let mut texts = Vec::new();
    let label = labelled.then(|| {
        let l = c.label(350.0, 155.0, 40.0);
        texts.push(l);
        l
    });
    let q = EdgeQuery {
        src: a,
        dst: b,
        label,
        style: if dashed {
            EdgeStyle::Dashed
        } else {
            EdgeStyle::Solid
        },
    };
    (
        Case {
            canvas: c,
            nodes: vec![a, b],
            texts,
        },
        q,
    )
}

#[test]
fn labelled_solid_arrow_points_to_dst() {
    let (case, q) = horizontal(false, true, true, false);
    let ev = run(&case, &q);
    assert_eq!(ev.status, PixelStatus::Traced, "{ev:?}");
    assert_eq!(ev.verdict, EndVerdict::Forward, "{ev:?}");
}

#[test]
fn reader_reversal_is_caught() {
    // The line's head is at A, but the reader said A -> B.
    let (case, q) = horizontal(true, false, true, false);
    let ev = run(&case, &q);
    assert_eq!(ev.verdict, EndVerdict::Reverse, "{ev:?}");
}

#[test]
fn unlabelled_line_without_heads() {
    let (case, q) = horizontal(false, false, false, false);
    let ev = run(&case, &q);
    assert_eq!(ev.status, PixelStatus::Traced, "{ev:?}");
    assert_eq!(ev.verdict, EndVerdict::NoArrowhead, "{ev:?}");
}

#[test]
fn bidirectional_needs_both_heads() {
    let (case, q) = horizontal(true, true, false, false);
    assert_eq!(run(&case, &q).verdict, EndVerdict::Bidirectional);
}

#[test]
fn dashed_labelled_connector() {
    let (case, q) = horizontal(false, true, true, true);
    let ev = run(&case, &q);
    assert_eq!(ev.status, PixelStatus::Traced, "{ev:?}");
    assert_eq!(ev.verdict, EndVerdict::Forward, "{ev:?}");
}

#[test]
fn elbow_connector_is_followed() {
    let mut c = Canvas::new(720, 480);
    let a = BBox::new(60.0, 60.0, 200.0, 130.0);
    let b = BBox::new(420.0, 330.0, 560.0, 400.0);
    c.node(a);
    c.node(b);
    // Right out of A, across, then down into the top of B.
    c.connector(
        &[(a.x2, 95.0), (490.0, 95.0), (490.0, b.y1)],
        false,
        true,
        false,
    );
    let case = Case {
        canvas: c,
        nodes: vec![a, b],
        texts: vec![],
    };
    let ev = run(
        &case,
        &EdgeQuery {
            src: a,
            dst: b,
            label: None,
            style: EdgeStyle::Solid,
        },
    );
    assert_eq!(ev.status, PixelStatus::Traced, "{ev:?}");
    assert_eq!(ev.verdict, EndVerdict::Forward, "{ev:?}");
    // And read backwards it is a reversal.
    let ev = run(
        &case,
        &EdgeQuery {
            src: b,
            dst: a,
            label: None,
            style: EdgeStyle::Solid,
        },
    );
    assert_eq!(ev.verdict, EndVerdict::Reverse, "{ev:?}");
}

#[test]
fn vertical_edge_with_neighbouring_connector() {
    let mut c = Canvas::new(600, 600);
    let top = BBox::new(200.0, 40.0, 340.0, 110.0);
    let mid = BBox::new(200.0, 250.0, 340.0, 320.0);
    let side = BBox::new(440.0, 250.0, 560.0, 320.0);
    for b in [top, mid, side] {
        c.node(b);
    }
    // mid -> top (head at top), and an unrelated mid -> side connector.
    c.connector(&[(270.0, mid.y1), (270.0, top.y2)], false, true, false);
    c.connector(&[(mid.x2, 285.0), (side.x1, 285.0)], false, true, false);
    let label = c.label(270.0, 180.0, 32.0);
    let case = Case {
        canvas: c,
        nodes: vec![top, mid, side],
        texts: vec![label],
    };
    // Reader says top -> mid (reversed).
    let ev = run(
        &case,
        &EdgeQuery {
            src: top,
            dst: mid,
            label: Some(label),
            style: EdgeStyle::Solid,
        },
    );
    assert_eq!(ev.verdict, EndVerdict::Reverse, "{ev:?}");
}

#[test]
fn missing_connector_is_not_invented() {
    let mut c = Canvas::new(720, 420);
    let (a, b) = (a_box(), b_box());
    c.node(a);
    c.node(b);
    let case = Case {
        canvas: c,
        nodes: vec![a, b],
        texts: vec![],
    };
    let ev = run(
        &case,
        &EdgeQuery {
            src: a,
            dst: b,
            label: None,
            style: EdgeStyle::Solid,
        },
    );
    assert_eq!(ev.verdict, EndVerdict::Unknown, "{ev:?}");
    assert_ne!(ev.status, PixelStatus::Traced);
}
