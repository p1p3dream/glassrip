//! Synthetic meeting frames: fictional whiteboards inside browser and meeting
//! chrome, plus non-board screens (content studio, chat, terminal, gallery) that
//! must never be read as boards.
//!
//! Frame layout (1920 x 1080): tab strip and URL bar on top, a board sidebar on
//! the left, a presenter banner, a zoom bar at the bottom, and a column of video
//! tiles on the right. The canvas is `(220, 110)-(1640, 1030)`. Every string drawn
//! outside the canvas (and the floating shape menu inside it) is a chrome string.

use glassrip_vision::BBox;

use super::{
    color, line_height, text_width, wrap, Canvas, Rgb, Rng, GENERATOR_NAME, GENERATOR_VERSION,
};
use crate::fixture::{BoardExpected, CaseMeta, BOARD_EXPECTED_SCHEMA};
use crate::metrics::board::{GoldBoard, GoldEdge, GoldNode, GoldOwnerTag, GoldSticky, LineStyle};
use crate::metrics::screen::ScreenType;

/// Frame width.
pub const FRAME_W: u32 = 1920;
/// Frame height.
pub const FRAME_H: u32 = 1080;
/// Canvas box in frame pixels.
pub const CANVAS: (i64, i64, i64, i64) = (220, 110, 1640, 1030);

/// A node definition: grid cell 0..5 (row-major, 3 columns x 2 rows).
#[derive(Debug, Clone)]
pub struct NodeDef {
    /// Id.
    pub id: &'static str,
    /// Label.
    pub text: &'static str,
    /// Grid cell.
    pub cell: usize,
}

/// An edge definition.
#[derive(Debug, Clone)]
pub struct EdgeDef {
    /// Tail id.
    pub src: &'static str,
    /// Head id.
    pub dst: &'static str,
    /// Label (empty for none).
    pub label: &'static str,
    /// Style.
    pub style: LineStyle,
}

/// A sticky definition.
#[derive(Debug, Clone)]
pub struct StickyDef {
    /// Text.
    pub text: &'static str,
    /// Fill.
    pub fill: Rgb,
    /// Kind.
    pub kind: &'static str,
}

/// A board scene.
#[derive(Debug, Clone)]
pub struct BoardDef {
    /// Node label scale.
    pub text_scale: i64,
    /// Nodes.
    pub nodes: Vec<NodeDef>,
    /// Edges.
    pub edges: Vec<EdgeDef>,
    /// Stickies.
    pub stickies: Vec<StickyDef>,
    /// Owner tags `(name, node id)`.
    pub owners: Vec<(&'static str, &'static str)>,
    /// Floating shape menu inside the canvas (chrome).
    pub shape_menu: bool,
}

/// What a frame shows.
#[derive(Debug, Clone)]
pub enum Scene {
    /// Whiteboard.
    Board(BoardDef),
    /// Content studio.
    Studio {
        /// Show the document editor pane.
        editor: bool,
    },
    /// Chat application.
    Chat,
    /// Terminal.
    Terminal,
    /// Video call gallery.
    Gallery,
}

/// One generated case.
#[derive(Debug, Clone)]
pub struct Case {
    /// Metadata.
    pub meta: CaseMeta,
    /// Scene.
    pub scene: Scene,
    /// Presenter (banner) name.
    pub presenter: &'static str,
    /// Video tile names.
    pub tiles: Vec<&'static str>,
}

/// Render result.
pub struct Rendered {
    /// The frame.
    pub canvas: Canvas,
    /// Expected output.
    pub expected: BoardExpected,
}

const TILES: [&str; 3] = ["Riley Park", "Casey Moreno", "Quinn Adler"];

fn meta(
    case: &str,
    description: &str,
    seed: u64,
    screen_type: ScreenType,
    tags: &[&str],
) -> CaseMeta {
    CaseMeta {
        case: case.into(),
        suite: "synthetic".into(),
        description: description.into(),
        generator: GENERATOR_NAME.into(),
        generator_version: GENERATOR_VERSION,
        seed,
        screen_type,
        tags: tags.iter().map(|t| t.to_string()).collect(),
    }
}

fn n(id: &'static str, text: &'static str, cell: usize) -> NodeDef {
    NodeDef { id, text, cell }
}

fn e(src: &'static str, dst: &'static str, label: &'static str) -> EdgeDef {
    EdgeDef {
        src,
        dst,
        label,
        style: LineStyle::Solid,
    }
}

fn dashed(src: &'static str, dst: &'static str, label: &'static str) -> EdgeDef {
    EdgeDef {
        src,
        dst,
        label,
        style: LineStyle::Dashed,
    }
}

fn s(text: &'static str, fill: Rgb, kind: &'static str) -> StickyDef {
    StickyDef { text, fill, kind }
}

/// Every public meeting-mode case, in a fixed order.
pub fn all_cases() -> Vec<Case> {
    use color::{BLUE, PINK, YELLOW};
    let wb = ScreenType::Whiteboard;
    vec![
        Case {
            meta: meta(
                "board_basic",
                "Three nodes, two labelled solid edges, one question sticky.",
                11,
                wb,
                &["basic", "labelled_edges"],
            ),
            scene: Scene::Board(BoardDef {
                text_scale: 3,
                nodes: vec![
                    n("ledger", "Ledger API", 0),
                    n("orbit", "Orbit Queue", 1),
                    n("parcel", "Parcel Store", 2),
                ],
                edges: vec![e("ledger", "orbit", "REST"), e("orbit", "parcel", "gRPC")],
                stickies: vec![s("Who owns retries?", YELLOW, "question")],
                owners: vec![],
                shape_menu: false,
            }),
            presenter: "Riley Park",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "board_dashed",
                "Four nodes with a long dashed labelled edge and an unlabelled edge.",
                12,
                wb,
                &["dashed_edge", "unlabelled_edge"],
            ),
            scene: Scene::Board(BoardDef {
                text_scale: 3,
                nodes: vec![
                    n("beacon", "Beacon Web", 0),
                    n("harbor", "Harbor Gateway", 1),
                    n("nimbus", "Nimbus Auth", 2),
                    n("kit", "Design Kit", 3),
                ],
                edges: vec![
                    e("beacon", "harbor", "GraphQL"),
                    e("harbor", "nimbus", "OAuth"),
                    e("beacon", "kit", ""),
                    dashed("kit", "nimbus", "Relates copy to layout blocks"),
                ],
                stickies: vec![
                    s("Idea: batch uploads", PINK, "idea"),
                    s("Milestone 1: reorder feed", YELLOW, "milestone"),
                ],
                owners: vec![],
                shape_menu: false,
            }),
            presenter: "Riley Park",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "board_owners",
                "Owner tags on nodes next to video tiles whose names must not become owners.",
                13,
                wb,
                &["owner_tags", "tile_names"],
            ),
            scene: Scene::Board(BoardDef {
                text_scale: 3,
                nodes: vec![
                    n("relay", "Relay Service", 0),
                    n("summit", "Summit App", 1),
                    n("ledger", "Ledger API", 3),
                    n("parcel", "Parcel Store", 4),
                ],
                edges: vec![
                    e("summit", "relay", "events"),
                    e("relay", "ledger", "SQL"),
                    e("summit", "parcel", "REST"),
                    e("ledger", "parcel", "webhook"),
                ],
                stickies: vec![s("Do we need a new queue?", YELLOW, "question")],
                owners: vec![
                    ("Avery", "summit"),
                    ("Jordan", "ledger"),
                    ("Morgan", "relay"),
                ],
                shape_menu: false,
            }),
            presenter: "Casey Moreno",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "board_small_text",
                "Six nodes with small labels and six edges.",
                14,
                wb,
                &["small_text", "dense"],
            ),
            scene: Scene::Board(BoardDef {
                text_scale: 2,
                nodes: vec![
                    n("intake", "Intake Form", 0),
                    n("rules", "Rules Engine", 1),
                    n("audit", "Audit Log", 2),
                    n("review", "Review Queue", 3),
                    n("notify", "Notifier", 4),
                    n("metrics", "Metrics Store", 5),
                ],
                edges: vec![
                    e("intake", "rules", "validate"),
                    e("rules", "audit", "append"),
                    e("rules", "notify", "trigger"),
                    e("review", "notify", "email"),
                    e("notify", "metrics", "counts"),
                    e("intake", "review", "manual"),
                ],
                stickies: vec![
                    s("Rename audit fields?", YELLOW, "question"),
                    s("Idea: weekly digest", PINK, "idea"),
                    s("Out of scope: exports", BLUE, "note"),
                ],
                owners: vec![],
                shape_menu: false,
            }),
            presenter: "Quinn Adler",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "board_sticky_grid",
                "A grid of cards next to two nodes; cards are stickies, not nodes.",
                15,
                wb,
                &["sticky_grid"],
            ),
            scene: Scene::Board(BoardDef {
                text_scale: 3,
                nodes: vec![n("feed", "Home Feed", 0), n("picker", "Card Picker", 2)],
                edges: vec![e("feed", "picker", "orders cards")],
                stickies: vec![
                    s("Step log", YELLOW, "note"),
                    s("Past entries", YELLOW, "note"),
                    s("Tips for you", YELLOW, "note"),
                    s("Group posts", YELLOW, "note"),
                    s("Unread notes", BLUE, "note"),
                    s("Reminders", YELLOW, "note"),
                    s("Two columns or one?", PINK, "question"),
                ],
                owners: vec![],
                shape_menu: false,
            }),
            presenter: "Riley Park",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "board_chrome_menu",
                "A floating shape menu on the canvas and an owner tag; menu text is chrome.",
                16,
                wb,
                &["chrome", "shape_menu", "owner_tags"],
            ),
            scene: Scene::Board(BoardDef {
                text_scale: 3,
                nodes: vec![
                    n("vault", "Vault Proxy", 0),
                    n("signal", "Signal Hub", 1),
                    n("pixel", "Pixel CDN", 4),
                ],
                edges: vec![e("vault", "signal", "mTLS"), e("signal", "pixel", "purge")],
                stickies: vec![s("Cache TTL still open", BLUE, "note")],
                owners: vec![("Casey", "signal")],
                shape_menu: true,
            }),
            presenter: "Casey Moreno",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "cms_document_list",
                "Content studio document list; a trap that must not be read as a board.",
                21,
                ScreenType::Cms,
                &["trap", "cms"],
            ),
            scene: Scene::Studio { editor: false },
            presenter: "Riley Park",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "cms_editor",
                "Content studio editing one document; a trap that must not be read as a board.",
                22,
                ScreenType::Cms,
                &["trap", "cms"],
            ),
            scene: Scene::Studio { editor: true },
            presenter: "Riley Park",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "chat_channel",
                "Chat application with channels and messages.",
                23,
                ScreenType::Chat,
                &["screen_type"],
            ),
            scene: Scene::Chat,
            presenter: "Quinn Adler",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "terminal_build",
                "Terminal running a build.",
                24,
                ScreenType::Code,
                &["screen_type"],
            ),
            scene: Scene::Terminal,
            presenter: "Casey Moreno",
            tiles: TILES.to_vec(),
        },
        Case {
            meta: meta(
                "meet_gallery",
                "Video call gallery of four participants.",
                25,
                ScreenType::MeetGallery,
                &["screen_type"],
            ),
            scene: Scene::Gallery,
            presenter: "",
            tiles: vec!["Riley Park", "Casey Moreno", "Quinn Adler", "Avery Stone"],
        },
    ]
}

struct Chrome {
    texts: Vec<String>,
}

impl Chrome {
    fn text(&mut self, c: &mut Canvas, x: i64, y: i64, t: &str, scale: i64, col: Rgb) {
        c.text(x, y, t, scale, col);
        self.texts.push(t.to_string());
    }
}

fn browser_chrome(c: &mut Canvas, ch: &mut Chrome, tabs: &[&str], url: &str) {
    c.fill_rect(0, 0, i64::from(FRAME_W), 40, color::CHROME);
    let mut x = 16;
    for t in tabs {
        let w = text_width(t, 2) + 32;
        c.fill_rect(x, 6, w, 34, [70, 72, 80]);
        ch.text(c, x + 16, 15, t, 2, color::CHROME_TEXT);
        x += w + 8;
    }
    c.fill_rect(0, 40, i64::from(FRAME_W), 40, [64, 66, 72]);
    c.fill_rect(120, 46, 900, 28, [36, 38, 42]);
    ch.text(c, 136, 52, url, 2, color::CHROME_TEXT);
}

fn tiles_column(c: &mut Canvas, ch: &mut Chrome, names: &[&str]) {
    c.fill_rect(1640, 80, 280, 1000, [24, 25, 29]);
    for (i, name) in names.iter().enumerate() {
        let y = 100 + i as i64 * 200;
        c.fill_rect(1656, y, 248, 180, color::TILE);
        // head and shoulders silhouette
        c.fill_rect(1760, y + 40, 44, 44, [90, 94, 104]);
        c.fill_rect(1736, y + 92, 92, 50, [90, 94, 104]);
        ch.text(c, 1668, y + 152, name, 2, color::CHROME_TEXT);
    }
}

fn board_sidebar(c: &mut Canvas, ch: &mut Chrome) {
    c.fill_rect(0, 80, 220, 1000, color::PANEL);
    c.fill_rect(219, 80, 1, 1000, color::BORDER);
    for (i, item) in [
        "Boards",
        "Templates",
        "Recent",
        "Starred",
        "Create frame",
        "Share",
    ]
    .iter()
    .enumerate()
    {
        ch.text(c, 20, 120 + i as i64 * 44, item, 2, color::GRAY);
    }
}

fn banner(c: &mut Canvas, ch: &mut Chrome, presenter: &str) {
    c.fill_rect(220, 80, 1420, 30, [232, 240, 254]);
    if !presenter.is_empty() {
        ch.text(
            c,
            240,
            87,
            &format!("{presenter} (Presenting)"),
            2,
            [40, 70, 140],
        );
    }
}

fn zoom_bar(c: &mut Canvas, ch: &mut Chrome) {
    c.fill_rect(220, 1030, 1420, 50, color::PANEL);
    ch.text(c, 1380, 1047, "100%", 2, color::GRAY);
    ch.text(c, 1470, 1047, "Fit", 2, color::GRAY);
    ch.text(c, 1540, 1047, "Undo", 2, color::GRAY);
}

fn bb(x: i64, y: i64, w: i64, h: i64) -> BBox {
    BBox::new(x as f64, y as f64, (x + w) as f64, (y + h) as f64)
}

#[derive(Clone, Copy)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

impl Rect {
    fn center(&self) -> (f64, f64) {
        (
            self.x as f64 + self.w as f64 / 2.0,
            self.y as f64 + self.h as f64 / 2.0,
        )
    }

    /// Point where the ray from the center toward `(dx, dy)` leaves the box, pushed out by `gap`.
    fn exit(&self, dx: f64, dy: f64, gap: f64) -> (f64, f64) {
        let (cx, cy) = self.center();
        let (hw, hh) = (self.w as f64 / 2.0 + gap, self.h as f64 / 2.0 + gap);
        let tx = if dx.abs() > 1e-9 {
            hw / dx.abs()
        } else {
            f64::INFINITY
        };
        let ty = if dy.abs() > 1e-9 {
            hh / dy.abs()
        } else {
            f64::INFINITY
        };
        let t = tx.min(ty);
        (cx + dx * t, cy + dy * t)
    }
}

fn draw_board(c: &mut Canvas, ch: &mut Chrome, def: &BoardDef, rng: &mut Rng) -> GoldBoard {
    let (cx0, cy0, cx1, cy1) = CANVAS;
    c.fill_rect(cx0, cy0, cx1 - cx0, cy1 - cy0, color::WHITE);
    let mut y = cy0 + 20;
    while y < cy1 {
        let mut x = cx0 + 20;
        while x < cx1 {
            c.put(x, y, [214, 218, 226]);
            x += 40;
        }
        y += 40;
    }

    // Nodes.
    let scale = def.text_scale;
    let (cell_w, cell_h) = (446, 250);
    let mut rects: Vec<(&str, Rect)> = Vec::new();
    let mut gold = GoldBoard::default();
    for node in &def.nodes {
        let col = (node.cell % 3) as i64;
        let row = (node.cell / 3) as i64;
        let max_chars = ((cell_w - 200) / (8 * scale)).max(6) as usize;
        let lines = wrap(node.text, max_chars);
        let text_w = lines
            .iter()
            .map(|l| text_width(l, scale))
            .max()
            .unwrap_or(0);
        let w = (text_w + 56).max(200);
        let h = lines.len() as i64 * line_height(scale) + 40;
        let x = 260 + col * cell_w + (cell_w - w) / 2 + rng.range(-12, 12);
        let y = 150 + row * cell_h + (cell_h - h) / 2 + rng.range(-20, 20);
        c.fill_rect(x, y, w, h, color::WHITE);
        c.stroke_rect(x, y, w, h, 3, color::NODE);
        for (i, l) in lines.iter().enumerate() {
            let lw = text_width(l, scale);
            c.text(
                x + (w - lw) / 2,
                y + 20 + i as i64 * line_height(scale),
                l,
                scale,
                color::INK,
            );
        }
        rects.push((node.id, Rect { x, y, w, h }));
        gold.nodes.push(GoldNode {
            id: node.id.into(),
            text: node.text.into(),
            aliases: vec![],
            bbox: Some(bb(x, y, w, h)),
            core: true,
        });
    }
    let rect_of = |id: &str| rects.iter().find(|(i, _)| *i == id).map(|(_, r)| *r);

    // Edges.
    for edge in &def.edges {
        let (Some(a), Some(b)) = (rect_of(edge.src), rect_of(edge.dst)) else {
            continue;
        };
        let (ax, ay) = a.center();
        let (bx, by) = b.center();
        let len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt().max(1.0);
        let (dx, dy) = ((bx - ax) / len, (by - ay) / len);
        let start = a.exit(dx, dy, 4.0);
        let tip = b.exit(-dx, -dy, 4.0);
        let base = (tip.0 - dx * 24.0, tip.1 - dy * 24.0);
        let dash = (edge.style == LineStyle::Dashed).then_some((16, 10));
        c.line(
            (start.0.round() as i64, start.1.round() as i64),
            (base.0.round() as i64, base.1.round() as i64),
            3,
            dash,
            color::EDGE,
        );
        let (px, py) = (-dy * 10.0, dx * 10.0);
        c.fill_triangle(
            [
                (tip.0.round() as i64, tip.1.round() as i64),
                ((base.0 + px).round() as i64, (base.1 + py).round() as i64),
                ((base.0 - px).round() as i64, (base.1 - py).round() as i64),
            ],
            color::EDGE,
        );
        if !edge.label.is_empty() {
            let lw = text_width(edge.label, 2);
            let (mx, my) = ((start.0 + tip.0) / 2.0, (start.1 + tip.1) / 2.0);
            // Labels of mostly horizontal edges sit above the line; others sit on it.
            let lift = if dx.abs() > dy.abs() { 30 } else { 8 };
            let (lx, ly) = (mx.round() as i64 - lw / 2, my.round() as i64 - lift);
            c.fill_rect(lx - 6, ly - 5, lw + 12, 26, color::WHITE);
            c.text(lx, ly, edge.label, 2, [70, 74, 84]);
        }
        gold.edges.push(GoldEdge {
            src: edge.src.into(),
            dst: edge.dst.into(),
            label: edge.label.into(),
            label_aliases: vec![],
            style: edge.style,
            directed: true,
        });
    }

    // Owner tags.
    for (name, near) in &def.owners {
        let Some(r) = rect_of(near) else { continue };
        let w = text_width(name, 2) + 20;
        let (x, y) = (r.x + r.w - w / 2, r.y - 18);
        c.fill_rect(x, y, w, 34, color::GREEN);
        c.text(x + 10, y + 9, name, 2, color::WHITE);
        gold.owners.push(GoldOwnerTag {
            name: (*name).into(),
            near: (*near).into(),
        });
    }

    // Stickies: rows of up to five in the bottom band.
    for (i, st) in def.stickies.iter().enumerate() {
        let (col, row) = ((i % 5) as i64, (i / 5) as i64);
        let (w, h) = (230, 140);
        let x = 270 + col * (w + 34) + rng.range(-6, 6);
        let y = 690 + row * (h + 24) + rng.range(-6, 6);
        c.fill_rect(x + 4, y + 4, w, h, [220, 222, 228]);
        c.fill_rect(x, y, w, h, st.fill);
        for (li, l) in wrap(st.text, 12).iter().enumerate() {
            c.text(
                x + 14,
                y + 16 + li as i64 * line_height(2),
                l,
                2,
                color::INK,
            );
        }
        gold.stickies.push(GoldSticky {
            text: st.text.into(),
            aliases: vec![],
            bbox: Some(bb(x, y, w, h)),
            kind: Some(st.kind.into()),
        });
    }

    // Floating shape menu (chrome inside the canvas).
    if def.shape_menu {
        if let Some((_, r)) = rects.first() {
            let (x, y) = (r.x + 10, r.y + r.h + 26);
            c.fill_rect(x + 3, y + 3, 380, 44, [200, 204, 212]);
            c.fill_rect(x, y, 380, 44, color::WHITE);
            c.stroke_rect(x, y, 380, 44, 1, color::BORDER);
            ch.text(c, x + 14, y + 14, "Convert to", 2, color::GRAY);
            ch.text(c, x + 200, y + 14, "Lock", 2, color::GRAY);
            ch.text(c, x + 280, y + 14, "Copy", 2, color::GRAY);
        }
    }
    gold
}

fn draw_studio(c: &mut Canvas, editor: bool) {
    let (x0, y0) = (0, 80);
    c.fill_rect(x0, y0, 1640, 1000, color::WHITE);
    c.fill_rect(x0, y0, 1640, 44, [248, 248, 250]);
    c.fill_rect(x0, y0 + 44, 1640, 1, color::BORDER);
    c.text(20, y0 + 14, "Plinth Studio", 2, color::INK);
    c.text(300, y0 + 14, "Structure", 2, color::GRAY);
    c.text(480, y0 + 14, "Vision", 2, color::GRAY);
    c.text(620, y0 + 14, "Releases", 2, color::GRAY);
    // Type pane.
    c.fill_rect(0, y0 + 45, 300, 955, color::PANEL);
    c.text(20, y0 + 70, "Content", 2, color::INK);
    for (i, t) in ["Article", "Series", "Journey", "Author", "Card"]
        .iter()
        .enumerate()
    {
        let y = y0 + 120 + i as i64 * 48;
        if i == 0 {
            c.fill_rect(0, y - 12, 300, 40, [226, 232, 244]);
        }
        c.text(36, y, t, 2, color::INK);
    }
    // Document list.
    c.fill_rect(300, y0 + 45, 1, 955, color::BORDER);
    c.text(330, y0 + 70, "Article", 2, color::INK);
    let titles = [
        "Morning stretch basics",
        "Hydration myths",
        "Weekly check-in guide",
        "Sleep and focus",
        "Snack swaps that stick",
        "Walking after meals",
        "Reading food labels",
        "Setting small goals",
    ];
    for (i, t) in titles.iter().enumerate() {
        let y = y0 + 120 + i as i64 * 64;
        c.fill_rect(330, y - 8, 36, 36, [210, 216, 228]);
        c.text(380, y, t, 2, color::INK);
        c.text(380, y + 22, "Published", 1, color::GRAY);
    }
    if editor {
        c.fill_rect(840, y0 + 45, 1, 955, color::BORDER);
        c.text(870, y0 + 70, "Hydration myths", 3, color::INK);
        let fields = [
            ("Title", "Hydration myths"),
            ("Slug", "hydration-myths"),
            ("Summary", "Five things to unlearn"),
            ("Reading time", "4 min"),
        ];
        for (i, (k, v)) in fields.iter().enumerate() {
            let y = y0 + 140 + i as i64 * 110;
            c.text(870, y, k, 2, color::GRAY);
            c.stroke_rect(870, y + 26, 700, 44, 2, color::BORDER);
            c.text(886, y + 40, v, 2, color::INK);
        }
        c.fill_rect(1430, y0 + 900, 140, 44, [40, 120, 80]);
        c.text(1452, y0 + 914, "Publish", 2, color::WHITE);
    }
}

fn draw_chat(c: &mut Canvas) {
    c.fill_rect(0, 80, 1640, 1000, color::WHITE);
    c.fill_rect(0, 80, 280, 1000, [58, 36, 70]);
    c.text(20, 100, "Northwind Team", 2, color::WHITE);
    for (i, ch) in ["# general", "# releases", "# design", "# random"]
        .iter()
        .enumerate()
    {
        c.text(30, 160 + i as i64 * 40, ch, 2, [220, 206, 230]);
    }
    c.text(310, 100, "# releases", 3, color::INK);
    let msgs = [
        ("Avery Stone", "10:42", "Build 4.2 is on the staging lane."),
        (
            "Jordan Vale",
            "10:44",
            "Thanks. Running the smoke list now.",
        ),
        (
            "Morgan Lee",
            "10:51",
            "Found one flaky test in the export job.",
        ),
        ("Avery Stone", "10:53", "Can you file it and tag me?"),
    ];
    for (i, (who, t, body)) in msgs.iter().enumerate() {
        let y = 180 + i as i64 * 110;
        c.fill_rect(310, y, 48, 48, [196, 204, 220]);
        c.text(376, y, who, 2, color::INK);
        c.text(376 + text_width(who, 2) + 20, y, t, 2, color::GRAY);
        c.text(376, y + 30, body, 2, color::INK);
    }
    c.stroke_rect(310, 960, 1300, 56, 2, color::BORDER);
    c.text(330, 980, "Message #releases", 2, color::GRAY);
}

fn draw_terminal(c: &mut Canvas) {
    c.fill_rect(0, 80, 1640, 1000, [18, 18, 20]);
    let lines = [
        "$ cargo build --release -p parcel-store",
        "   Compiling parcel-store v0.4.1",
        "    Finished release profile in 41.20s",
        "$ cargo test -p parcel-store",
        "running 12 tests",
        "test store::roundtrip ... ok",
        "test store::evicts_oldest ... ok",
        "test result: ok. 12 passed; 0 failed",
        "$ git status --short",
        " M src/store.rs",
        "$ _",
    ];
    for (i, l) in lines.iter().enumerate() {
        let col = if l.starts_with('$') {
            [120, 220, 140]
        } else {
            [220, 222, 226]
        };
        c.text(30, 110 + i as i64 * 40, l, 2, col);
    }
}

fn draw_gallery(c: &mut Canvas, ch: &mut Chrome, names: &[&str]) {
    c.fill_rect(0, 80, 1920, 1000, [32, 33, 38]);
    for (i, name) in names.iter().enumerate() {
        let (col, row) = ((i % 2) as i64, (i / 2) as i64);
        let (x, y) = (120 + col * 860, 110 + row * 460);
        c.fill_rect(x, y, 820, 430, [60, 63, 72]);
        c.fill_rect(x + 360, y + 110, 100, 100, [110, 114, 126]);
        c.fill_rect(x + 300, y + 230, 220, 120, [110, 114, 126]);
        ch.text(c, x + 20, y + 396, name, 2, color::WHITE);
    }
    ch.text(c, 820, 1044, "4 people in call", 2, color::CHROME_TEXT);
}

/// Renders one case.
pub fn render(case: &Case) -> Rendered {
    let mut rng = Rng::new(case.meta.seed);
    let mut c = Canvas::new(FRAME_W, FRAME_H, color::WHITE);
    let mut ch = Chrome { texts: Vec::new() };
    let mut board = GoldBoard::default();
    let mut canvas_bbox = None;
    match &case.scene {
        Scene::Board(def) => {
            browser_chrome(
                &mut c,
                &mut ch,
                &["Q3 roadmap - Boardly", "Inbox (3)"],
                "boardly.example/app/board/q3-roadmap",
            );
            board_sidebar(&mut c, &mut ch);
            banner(&mut c, &mut ch, case.presenter);
            zoom_bar(&mut c, &mut ch);
            board = draw_board(&mut c, &mut ch, def, &mut rng);
            let (x0, y0, x1, y1) = CANVAS;
            canvas_bbox = Some(BBox::new(x0 as f64, y0 as f64, x1 as f64, y1 as f64));
        }
        Scene::Studio { editor } => {
            browser_chrome(
                &mut c,
                &mut ch,
                &["Plinth Studio", "Inbox (3)"],
                "plinth.example/studio/desk/article",
            );
            draw_studio(&mut c, *editor);
        }
        Scene::Chat => {
            browser_chrome(
                &mut c,
                &mut ch,
                &["releases - Northwind", "Inbox (3)"],
                "chat.example/northwind/releases",
            );
            draw_chat(&mut c);
        }
        Scene::Terminal => {
            c.fill_rect(0, 0, i64::from(FRAME_W), 80, color::CHROME);
            ch.text(
                &mut c,
                20,
                30,
                "parcel-store - zsh - 160x48",
                2,
                color::CHROME_TEXT,
            );
            draw_terminal(&mut c);
        }
        Scene::Gallery => {
            browser_chrome(
                &mut c,
                &mut ch,
                &["Weekly sync - Call", "Inbox (3)"],
                "call.example/abc-defg-hij",
            );
            draw_gallery(&mut c, &mut ch, &case.tiles);
        }
    }
    if !matches!(case.scene, Scene::Gallery) {
        tiles_column(&mut c, &mut ch, &case.tiles);
    }
    let mut participants: Vec<String> = case.tiles.iter().map(|t| t.to_string()).collect();
    if let Scene::Board(def) = &case.scene {
        participants.extend(def.owners.iter().map(|(name, _)| name.to_string()));
    }
    Rendered {
        canvas: c,
        expected: BoardExpected {
            schema: BOARD_EXPECTED_SCHEMA.into(),
            schema_version: "1.0.0".into(),
            screen_type: case.meta.screen_type,
            frame_width: FRAME_W,
            frame_height: FRAME_H,
            canvas_bbox,
            board,
            chrome_texts: ch.texts,
            participants,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_are_valid_and_unique() {
        let cases = all_cases();
        let mut names: Vec<&str> = cases.iter().map(|c| c.meta.case.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), cases.len());
        for case in &cases {
            let r = render(case);
            crate::fixture::validate_board(&case.meta.case, &r.expected.board).unwrap();
            // Every gold box lies inside the canvas.
            for n in &r.expected.board.nodes {
                let b = n.bbox.unwrap();
                assert!(
                    b.x1 >= 220.0 && b.x2 <= 1640.0 && b.y1 >= 110.0 && b.y2 <= 1030.0,
                    "{}",
                    n.id
                );
            }
            // Chrome strings never collide with board content.
            for n in &r.expected.board.nodes {
                assert!(!crate::metrics::board::is_chrome(
                    &n.text,
                    &r.expected.chrome_texts
                ));
            }
        }
    }

    #[test]
    fn render_is_deterministic() {
        let case = &all_cases()[0];
        let a = render(case);
        let b = render(case);
        assert_eq!(a.canvas.image().as_raw(), b.canvas.image().as_raw());
        assert_eq!(a.expected, b.expected);
    }
}
