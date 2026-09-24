//! Synthetic document-mode fixtures: fictional tickets and documentation pages
//! rendered as one tall page, then captured as overlapping viewport frames with a
//! fixed header (and optionally a fixed sidebar), the way a scrolling screen
//! recording sees them. Consecutive frames overlap by half a viewport.

use image::RgbImage;

use super::{color, line_height, text_width, wrap, Canvas, Rgb, GENERATOR_NAME, GENERATOR_VERSION};
use crate::fixture::{CaseMeta, DocsExpected, FrameRef, DOCS_EXPECTED_SCHEMA};
use crate::metrics::docs::{Block, BlockKind, Comment, DocType, Document, Link, TicketFields};
use crate::metrics::screen::ScreenType;

/// Viewport width.
pub const VIEW_W: u32 = 1280;
/// Viewport height.
pub const VIEW_H: u32 = 720;
/// Fixed header height.
pub const HEADER_H: u32 = 64;
/// Fixed sidebar width (when present).
pub const SIDEBAR_W: i64 = 240;

/// One document case.
#[derive(Debug, Clone)]
pub struct Case {
    /// Metadata.
    pub meta: CaseMeta,
    /// Gold document.
    pub document: Document,
    /// Header app name.
    pub app: &'static str,
    /// Header URL text.
    pub url: &'static str,
    /// Draw a fixed left sidebar.
    pub sidebar: bool,
}

/// Render result.
pub struct Rendered {
    /// Expected output (frames listed in order).
    pub expected: DocsExpected,
    /// Frame images, same order as `expected.frames`.
    pub frames: Vec<RgbImage>,
}

fn meta(
    case: &str,
    description: &str,
    seed: u64,
    screen_type: ScreenType,
    tags: &[&str],
) -> CaseMeta {
    CaseMeta {
        case: case.into(),
        suite: "synthetic_docs".into(),
        description: description.into(),
        generator: GENERATOR_NAME.into(),
        generator_version: GENERATOR_VERSION,
        seed,
        screen_type,
        tags: tags.iter().map(|t| t.to_string()).collect(),
    }
}

fn block(kind: BlockKind, level: Option<u8>, content: &str) -> Block {
    Block {
        kind,
        level,
        lang: None,
        content: content.into(),
        rows: vec![],
    }
}

fn table(rows: &[&[&str]]) -> Block {
    let rows: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|c| c.to_string()).collect())
        .collect();
    let content = rows
        .iter()
        .map(|r| r.join("\t"))
        .collect::<Vec<_>>()
        .join("\n");
    Block {
        kind: BlockKind::Table,
        level: None,
        lang: None,
        content,
        rows,
    }
}

fn code(lang: &str, content: &str) -> Block {
    Block {
        kind: BlockKind::Code,
        level: None,
        lang: Some(lang.into()),
        content: content.into(),
        rows: vec![],
    }
}

/// Every public document case, in a fixed order.
pub fn all_cases() -> Vec<Case> {
    let ticket = TicketFields {
        key: "ORB-412".into(),
        title: "Retry failed parcel uploads".into(),
        issue_type: "Story".into(),
        status: "In Progress".into(),
        priority: "High".into(),
        assignee: "Avery Stone".into(),
        reporter: "Jordan Vale".into(),
        labels: vec!["backend".into(), "uploads".into()],
        epic: "ORB-400".into(),
        sprint: "Sprint 14".into(),
        description: "Uploads that fail with a network error are dropped today. Retry them with backoff so a short outage does not lose parcels.".into(),
        acceptance_criteria: vec![
            "Failed uploads retry three times with backoff".into(),
            "A parcel is never uploaded twice".into(),
            "Retries show up in the upload log".into(),
        ],
        comments: vec![
            Comment {
                author: "Morgan Lee".into(),
                time: "2 days ago".into(),
                body: "Can we reuse the queue retry helper here?".into(),
            },
            Comment {
                author: "Avery Stone".into(),
                time: "2 days ago".into(),
                body: "Yes, it already handles jitter.".into(),
            },
            Comment {
                author: "Jordan Vale".into(),
                time: "1 day ago".into(),
                body: "Please add a metric for dropped parcels too.".into(),
            },
        ],
        links: vec![
            Link {
                relation: "blocks".into(),
                key: "ORB-415".into(),
            },
            Link {
                relation: "relates to".into(),
                key: "ORB-388".into(),
            },
        ],
        attachments: vec![],
    };
    let guide_blocks = vec![
        block(BlockKind::Heading, Some(1), "Queue setup guide"),
        block(BlockKind::Paragraph, None, "This page explains how to run the parcel queue on a laptop and how to read its dashboard."),
        block(BlockKind::Heading, Some(2), "Install"),
        block(BlockKind::List, None, "Install the queue tool\nCreate a local config file\nStart the worker"),
        code("shell", "queue init --local\nqueue worker start --threads 4"),
        block(BlockKind::Heading, Some(2), "Settings"),
        table(&[
            &["Name", "Default", "Meaning"],
            &["threads", "4", "Worker threads"],
            &["batch", "50", "Items per pull"],
            &["retry", "3", "Attempts per item"],
            &["backoff", "2s", "First retry delay"],
            &["ttl", "1h", "Item lifetime"],
            &["log", "info", "Log level"],
            &["port", "7400", "Dashboard port"],
            &["region", "local", "Storage region"],
        ]),
        block(BlockKind::Quote, None, "Keep threads at or below the number of cores."),
        block(BlockKind::Paragraph, None, "Open the dashboard on the configured port to see queue depth and retry counts."),
    ];
    let mut long_blocks = vec![block(BlockKind::Heading, Some(1), "Release checklist")];
    for (i, (h, p)) in [
        (
            "Freeze",
            "Announce the freeze in the release channel and stop merging features.",
        ),
        (
            "Branch",
            "Cut the release branch from main and tag the candidate build.",
        ),
        (
            "Verify",
            "Run the smoke list on staging and record every failure with its owner.",
        ),
        (
            "Notes",
            "Collect merged changes and write short release notes for support.",
        ),
        (
            "Ship",
            "Promote the candidate to production during the quiet window.",
        ),
        (
            "Watch",
            "Watch error rates for one hour and roll back if they double.",
        ),
        (
            "Close",
            "Close the release ticket and unfreeze the main branch.",
        ),
        (
            "Audit",
            "Compare the shipped build hash with the tagged candidate.",
        ),
        (
            "Support",
            "Brief the support rotation on known issues and workarounds.",
        ),
        (
            "Docs",
            "Publish the updated user guide pages that changed in this release.",
        ),
        (
            "Metrics",
            "Snapshot the release dashboard for the weekly review.",
        ),
        (
            "Cleanup",
            "Delete temporary feature flags that were fully rolled out.",
        ),
        (
            "Retro",
            "Hold a short retro and file follow-up tickets with owners.",
        ),
        (
            "Archive",
            "Archive the release channel thread and link it from the ticket.",
        ),
    ]
    .iter()
    .enumerate()
    {
        long_blocks.push(block(
            BlockKind::Heading,
            Some(2),
            &format!("Step {}: {h}", i + 1),
        ));
        long_blocks.push(block(BlockKind::Paragraph, None, p));
        long_blocks.push(block(
            BlockKind::List,
            None,
            &format!(
                "Owner confirms step {}\nChecklist item ticked\nTime recorded\nLinks attached\nNext owner paged",
                i + 1
            ),
        ));
    }
    vec![
        Case {
            meta: meta(
                "ticket_story",
                "Tracker story with fields, criteria, three comments, and links.",
                31,
                ScreenType::Ticket,
                &["ticket", "comments", "links"],
            ),
            document: Document {
                page_id: "ORB-412".into(),
                doc_type: DocType::Ticket,
                title: "Retry failed parcel uploads".into(),
                fields: Some(ticket),
                blocks: vec![],
                body_text: None,
            },
            app: "Trackly",
            url: "trackly.example/browse/ORB-412",
            sidebar: false,
        },
        Case {
            meta: meta(
                "doc_guide",
                "Documentation page with a list, a code block, an 8-row table, and a quote.",
                32,
                ScreenType::Doc,
                &["doc", "table", "code", "quote"],
            ),
            document: Document {
                page_id: "queue-setup-guide".into(),
                doc_type: DocType::Doc,
                title: "Queue setup guide".into(),
                fields: None,
                blocks: guide_blocks,
                body_text: None,
            },
            app: "Wikiwise",
            url: "wiki.example/spaces/ENG/pages/queue-setup-guide",
            sidebar: false,
        },
        Case {
            meta: meta(
                "doc_long_sticky",
                "Long page (at least 5 screens) with a fixed header and a fixed sidebar.",
                33,
                ScreenType::Doc,
                &["doc", "long_page", "sticky_header", "sidebar"],
            ),
            document: Document {
                page_id: "release-checklist".into(),
                doc_type: DocType::Doc,
                title: "Release checklist".into(),
                fields: None,
                blocks: long_blocks,
                body_text: None,
            },
            app: "Wikiwise",
            url: "wiki.example/spaces/OPS/pages/release-checklist",
            sidebar: true,
        },
    ]
}

/// A page being laid out top to bottom.
struct Page {
    canvas: Canvas,
    x: i64,
    width: i64,
    y: i64,
}

impl Page {
    fn text_lines(&mut self, text: &str, scale: i64, col: Rgb, indent: i64) {
        let max_chars = ((self.width - indent) / (8 * scale)).max(8) as usize;
        for para in text.split('\n') {
            for l in wrap(para, max_chars) {
                self.canvas.text(self.x + indent, self.y, &l, scale, col);
                self.y += line_height(scale);
            }
        }
    }

    fn gap(&mut self, px: i64) {
        self.y += px;
    }
}

const PAGE_H_MAX: u32 = 6000;

fn layout(case: &Case) -> (Canvas, i64) {
    let left = if case.sidebar { SIDEBAR_W + 40 } else { 60 };
    let width = i64::from(VIEW_W) - left - 60;
    let mut p = Page {
        canvas: Canvas::new(VIEW_W, PAGE_H_MAX, color::WHITE),
        x: left,
        width,
        y: 30,
    };
    let doc = &case.document;
    if let Some(f) = &doc.fields {
        p.text_lines(&f.key, 2, color::GRAY, 0);
        p.gap(6);
        p.text_lines(&f.title, 3, color::INK, 0);
        p.gap(20);
        let rows = [
            ("Type", f.issue_type.clone()),
            ("Status", f.status.clone()),
            ("Priority", f.priority.clone()),
            ("Assignee", f.assignee.clone()),
            ("Reporter", f.reporter.clone()),
            ("Labels", f.labels.join(", ")),
            ("Epic", f.epic.clone()),
            ("Sprint", f.sprint.clone()),
        ];
        for (k, v) in rows {
            p.canvas.text(p.x, p.y, k, 2, color::GRAY);
            p.canvas.text(p.x + 220, p.y, &v, 2, color::INK);
            p.y += line_height(2) + 8;
        }
        p.gap(24);
        p.text_lines("Description", 3, color::INK, 0);
        p.gap(8);
        p.text_lines(&f.description, 2, color::INK, 0);
        p.gap(24);
        p.text_lines("Acceptance criteria", 3, color::INK, 0);
        p.gap(8);
        for a in &f.acceptance_criteria {
            p.canvas.fill_rect(p.x + 6, p.y + 5, 6, 6, color::INK);
            p.text_lines(a, 2, color::INK, 24);
            p.gap(4);
        }
        p.gap(24);
        p.text_lines("Links", 3, color::INK, 0);
        p.gap(8);
        for l in &f.links {
            p.text_lines(&format!("{} {}", l.relation, l.key), 2, [40, 90, 170], 0);
        }
        p.gap(24);
        p.text_lines("Comments", 3, color::INK, 0);
        p.gap(8);
        for c in &f.comments {
            p.canvas.fill_rect(p.x, p.y, 32, 32, [196, 204, 220]);
            p.canvas.text(
                p.x + 44,
                p.y + 8,
                &format!("{} - {}", c.author, c.time),
                2,
                color::GRAY,
            );
            p.y += 42;
            p.text_lines(&c.body, 2, color::INK, 44);
            p.gap(20);
        }
    } else {
        for b in &doc.blocks {
            match b.kind {
                BlockKind::Heading => {
                    let scale = if b.level == Some(1) { 4 } else { 3 };
                    p.gap(10);
                    p.text_lines(&b.content, scale, color::INK, 0);
                    p.gap(10);
                }
                BlockKind::Paragraph => {
                    p.text_lines(&b.content, 2, color::INK, 0);
                    p.gap(16);
                }
                BlockKind::List => {
                    for item in b.content.split('\n') {
                        p.canvas.fill_rect(p.x + 6, p.y + 5, 6, 6, color::INK);
                        p.text_lines(item, 2, color::INK, 24);
                        p.gap(4);
                    }
                    p.gap(12);
                }
                BlockKind::Code => {
                    let lines = b.content.split('\n').count() as i64;
                    let h = lines * line_height(2) + 28;
                    p.canvas.fill_rect(p.x, p.y, p.width, h, [240, 241, 245]);
                    p.gap(14);
                    for l in b.content.split('\n') {
                        p.canvas.text(p.x + 16, p.y, l, 2, [60, 40, 120]);
                        p.y += line_height(2);
                    }
                    p.gap(30);
                }
                BlockKind::Quote => {
                    let start = p.y;
                    p.text_lines(&b.content, 2, color::GRAY, 24);
                    p.canvas
                        .fill_rect(p.x, start - 4, 6, p.y - start + 4, color::BORDER);
                    p.gap(16);
                }
                BlockKind::Table => {
                    let cols = b.rows.iter().map(Vec::len).max().unwrap_or(1).max(1) as i64;
                    let cw = p.width / cols;
                    let rh = line_height(2) + 16;
                    for (ri, row) in b.rows.iter().enumerate() {
                        if ri == 0 {
                            p.canvas.fill_rect(p.x, p.y, cw * cols, rh, color::PANEL);
                        }
                        for (ci, cell) in row.iter().enumerate() {
                            let cx = p.x + ci as i64 * cw;
                            p.canvas
                                .stroke_rect(cx, p.y, cw + 1, rh + 1, 1, color::BORDER);
                            p.canvas.text(cx + 10, p.y + 9, cell, 2, color::INK);
                        }
                        p.y += rh;
                    }
                    p.gap(20);
                }
                BlockKind::Image => {
                    p.canvas.fill_rect(p.x, p.y, 400, 200, color::PANEL);
                    p.gap(220);
                }
            }
        }
    }
    let page_h = (p.y + 40).min(i64::from(PAGE_H_MAX));
    (p.canvas, page_h)
}

fn header(c: &mut Canvas, case: &Case) {
    c.fill_rect(0, 0, i64::from(VIEW_W), i64::from(HEADER_H), color::CHROME);
    c.text(20, 16, case.app, 2, color::WHITE);
    c.text(20, 40, case.url, 1, color::CHROME_TEXT);
    let right = "Search  Help";
    c.text(
        i64::from(VIEW_W) - text_width(right, 2) - 20,
        24,
        right,
        2,
        color::CHROME_TEXT,
    );
}

fn sidebar(c: &mut Canvas) {
    c.fill_rect(
        0,
        i64::from(HEADER_H),
        SIDEBAR_W,
        i64::from(VIEW_H),
        color::PANEL,
    );
    for (i, item) in [
        "Overview",
        "Runbooks",
        "Release checklist",
        "On call",
        "Archive",
    ]
    .iter()
    .enumerate()
    {
        c.text(
            20,
            i64::from(HEADER_H) + 30 + i as i64 * 40,
            item,
            2,
            color::GRAY,
        );
    }
}

/// Renders one case into overlapping frames.
pub fn render(case: &Case) -> Rendered {
    let (page, page_h) = layout(case);
    let content_h = i64::from(VIEW_H - HEADER_H);
    let step = i64::from(VIEW_H) / 2;
    let mut offsets = Vec::new();
    let mut y = 0;
    loop {
        let last = (page_h - content_h).max(0);
        offsets.push(y.min(last));
        if y >= last {
            break;
        }
        y += step;
    }
    let mut frames = Vec::new();
    let mut refs = Vec::new();
    for (i, &off) in offsets.iter().enumerate() {
        let mut c = Canvas::new(VIEW_W, VIEW_H, color::WHITE);
        for row in 0..content_h {
            let src_y = off + row;
            if src_y >= page_h {
                break;
            }
            for x in 0..i64::from(VIEW_W) {
                let px = page.image().get_pixel(x as u32, src_y as u32).0;
                c.put(x, i64::from(HEADER_H) + row, px);
            }
        }
        if case.sidebar {
            sidebar(&mut c);
        }
        header(&mut c, case);
        frames.push(c.into_image());
        refs.push(FrameRef {
            file: format!("frames/f_{i:03}.png"),
            scroll_y: off as u32,
        });
    }
    Rendered {
        expected: DocsExpected {
            schema: DOCS_EXPECTED_SCHEMA.into(),
            schema_version: "1.0.0".into(),
            screen_type: case.meta.screen_type,
            document: case.document.clone(),
            page_height_px: page_h as u32,
            viewport_width: VIEW_W,
            viewport_height: VIEW_H,
            fixed_header_px: HEADER_H,
            frames: refs,
        },
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_overlap_and_cover_the_page() {
        for case in all_cases() {
            let r = render(&case);
            let e = &r.expected;
            let content_h = VIEW_H - HEADER_H;
            assert_eq!(e.frames[0].scroll_y, 0);
            let last = e.frames.last().unwrap();
            assert_eq!(last.scroll_y + content_h, e.page_height_px.max(content_h));
            for w in e.frames.windows(2) {
                // Consecutive frames overlap by at least half a viewport of content.
                assert!(w[1].scroll_y - w[0].scroll_y <= VIEW_H / 2);
            }
            assert_eq!(r.frames.len(), e.frames.len());
        }
    }

    #[test]
    fn long_page_needs_five_screens() {
        let case = all_cases()
            .into_iter()
            .find(|c| c.meta.case == "doc_long_sticky")
            .unwrap();
        let r = render(&case);
        assert!(r.expected.page_height_px >= 5 * (VIEW_H - HEADER_H));
    }

    #[test]
    fn guide_has_eight_row_table() {
        let case = all_cases()
            .into_iter()
            .find(|c| c.meta.case == "doc_guide")
            .unwrap();
        let tables = case.document.tables();
        assert_eq!(tables.len(), 1);
        assert!(tables[0].len() >= 9); // header + 8 rows
    }
}
