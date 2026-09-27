//! Orthogonal edge routing around cards.
//!
//! Each edge is routed as an axis-aligned polyline on a sparse grid (an
//! orthogonal visibility graph) whose lines are the cards' sides pushed out by
//! a clearance, the middles of the gaps between them, a ring around the whole
//! architecture, the port stubs, and tracks beside the edges routed before.
//! A* over (grid point, heading) minimizes length plus a penalty per bend, per
//! crossing of an earlier edge and per pixel run along one, so a route keeps off
//! every card, bends little, crosses little and keeps parallel edges apart.
//!
//! An edge leaves its source and enters its target perpendicular to one of the
//! card's sides, at a port spaced from the other ports on that side; the side
//! is the route's choice. Soft obstacles (zone titles and badges) are avoided
//! when a route exists without them. When no route exists at all, the edge is
//! drawn straight between the cards' borders and flagged as a fallback.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::scene::R;

/// A point.
pub type Pt = (f64, f64);

/// Clearance between a route and any card; the port stubs are this long.
pub const CLEAR: f64 = 16.0;
/// Spacing of the parallel tracks beside routed edges.
pub const TRACK: f64 = 12.0;
/// Cost of one bend, in pixels of length.
const BEND: f64 = 40.0;
/// Cost of crossing an earlier edge.
const CROSS: f64 = 90.0;
/// Cost of passing through a corner or end of an earlier edge (two corners
/// meeting read as one line crossing another).
const TOUCH: f64 = 240.0;
/// Cost per pixel of running closer than [`TRACK`] along an earlier edge.
const SHARE: f64 = 8.0;
/// Cost per pixel of running along a card's clearance line (routes prefer
/// the middle of a gap and leave the lines by the cards to their ports).
const HUG: f64 = 0.3;
/// Minimum distance between two ports on one side of a card (an edge that
/// carries a label asks for more, see [`Router::route`]).
pub const PORT_GAP: f64 = 24.0;
/// Ports keep this far from a card's corners.
const PORT_INSET: f64 = 18.0;
/// Most lines per grid axis: past it the tracks beside earlier edges are
/// dropped (then their lines), so a very dense board bounds the search
/// instead of exhausting memory; routes still avoid every card.
const MAX_AXIS: usize = 480;
/// Hard limit of lines per grid axis: a board whose cards alone need more is
/// not searched (bounded memory and time); its edges are drawn as declared
/// fallbacks, which validation reports as warnings.
const MAX_GRID_AXIS: usize = 800;
/// Distance of the ring lines around the architecture.
const RING: [f64; 2] = [40.0, 80.0];
const EPS: f64 = 0.5;

/// A side of a card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Top.
    Top,
    /// Right.
    Right,
    /// Bottom.
    Bottom,
    /// Left.
    Left,
}

impl Side {
    const ALL: [Side; 4] = [Side::Top, Side::Right, Side::Bottom, Side::Left];

    /// Heading leaving the card through this side (0 +x, 1 -x, 2 +y, 3 -y).
    fn out(self) -> usize {
        match self {
            Side::Right => 0,
            Side::Left => 1,
            Side::Bottom => 2,
            Side::Top => 3,
        }
    }

    fn normal(self) -> Pt {
        match self {
            Side::Right => (1.0, 0.0),
            Side::Left => (-1.0, 0.0),
            Side::Bottom => (0.0, 1.0),
            Side::Top => (0.0, -1.0),
        }
    }

    /// The point on side `self` of `r` at `along` (x on top and bottom, y on
    /// left and right).
    fn port(self, r: &R, along: f64) -> Pt {
        match self {
            Side::Top => (along, r.y),
            Side::Bottom => (along, r.bottom()),
            Side::Left => (r.x, along),
            Side::Right => (r.right(), along),
        }
    }

    fn stub(self, r: &R, along: f64) -> Pt {
        let p = self.port(r, along);
        let n = self.normal();
        (p.0 + n.0 * CLEAR, p.1 + n.1 * CLEAR)
    }

    /// True when the side faces `other` (its center lies beyond the side).
    fn faces(self, r: &R, other: &R) -> bool {
        match self {
            Side::Top => other.cy() < r.y,
            Side::Bottom => other.cy() > r.bottom(),
            Side::Left => other.cx() < r.x,
            Side::Right => other.cx() > r.right(),
        }
    }
}

/// A routed edge.
#[derive(Debug, Clone, PartialEq)]
pub struct Routed {
    /// Polyline from the source's border to the target's border.
    pub points: Vec<Pt>,
    /// No orthogonal route existed: a straight line between the borders.
    pub fallback: bool,
}

/// Routes edges one after another; later edges avoid the earlier ones.
pub struct Router {
    cards: Vec<R>,
    soft: Vec<R>,
    /// Lines a route may cross but should not run along (zone borders).
    guides: Vec<(Pt, Pt)>,
    bounds: R,
    segs: Vec<(Pt, Pt)>,
    /// (card, side, position, spacing asked for).
    ports: Vec<(usize, Side, f64, f64)>,
}

/// An end of a route search: a port, its stub, and the heading at the stub.
#[derive(Debug, Clone, Copy)]
struct End {
    side: Side,
    along: f64,
    stub: Pt,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Open {
    f: f64,
    g: f64,
    state: usize,
}

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, o: &Self) -> Ordering {
        // min-heap on f, then on g (deeper first), then on state (determinism)
        o.f.total_cmp(&self.f)
            .then(self.g.total_cmp(&o.g))
            .then(o.state.cmp(&self.state))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

fn strictly_inside(p: Pt, o: &R) -> bool {
    p.0 > o.x + EPS && p.0 < o.right() - EPS && p.1 > o.y + EPS && p.1 < o.bottom() - EPS
}

/// Drops repeated points and the middle of three collinear points.
pub fn simplify(pts: &[Pt]) -> Vec<Pt> {
    let mut out: Vec<Pt> = Vec::new();
    for &p in pts {
        if out
            .last()
            .is_some_and(|q| (q.0 - p.0).abs() < EPS && (q.1 - p.1).abs() < EPS)
        {
            continue;
        }
        if out.len() >= 2 {
            let (a, b) = (out[out.len() - 2], out[out.len() - 1]);
            let same_x = (a.0 - b.0).abs() < EPS && (b.0 - p.0).abs() < EPS;
            let same_y = (a.1 - b.1).abs() < EPS && (b.1 - p.1).abs() < EPS;
            if same_x || same_y {
                out.pop();
            }
        }
        out.push(p);
    }
    out
}

/// Distance from `p` to the segment `a`-`b`.
fn seg_dist(p: Pt, a: Pt, b: Pt) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (x, y) = (a.0 + dx * t, a.1 + dy * t);
    ((p.0 - x).powi(2) + (p.1 - y).powi(2)).sqrt()
}

/// Where the line from the center of `r` toward `to` leaves `r`.
fn border_toward(r: &R, to: Pt) -> Pt {
    let c = (r.cx(), r.cy());
    let (dx, dy) = (to.0 - c.0, to.1 - c.1);
    let tx = if dx.abs() > 1e-9 {
        r.w / 2.0 / dx.abs()
    } else {
        f64::INFINITY
    };
    let ty = if dy.abs() > 1e-9 {
        r.h / 2.0 / dy.abs()
    } else {
        f64::INFINITY
    };
    let t = tx.min(ty).min(1.0);
    (c.0 + dx * t, c.1 + dy * t)
}

/// Sorted, deduplicated coordinates (rounded to whole pixels) within `lo..=hi`.
fn axis(mut v: Vec<f64>, lo: f64, hi: f64) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    let mut v: Vec<f64> = v.into_iter().map(|x| x.round().clamp(lo, hi)).collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    v
}

fn index_of(v: &[f64], x: f64) -> Option<usize> {
    let x = x.round();
    v.binary_search_by(|p| p.total_cmp(&x)).ok()
}

impl Router {
    /// A router for `cards` (hard obstacles), `soft` boxes (avoided when
    /// possible) and `guides` (lines not to run along, such as zone borders)
    /// inside `bounds`.
    pub fn new(cards: Vec<R>, soft: Vec<R>, guides: Vec<(Pt, Pt)>, bounds: R) -> Self {
        Self {
            cards,
            soft,
            guides,
            bounds,
            segs: Vec::new(),
            ports: Vec::new(),
        }
    }

    /// Every segment routed so far.
    pub fn segments(&self) -> &[(Pt, Pt)] {
        &self.segs
    }

    /// The free port on side `side` of card `card` nearest to where it would
    /// line up with `other` (the middle of the span both cards can take a port
    /// in, so both ends pick the same line; else the middle of the side), at
    /// least `gap` from the side's other ports.
    fn slot(&self, card: usize, side: Side, other: &R, gap: f64) -> Option<f64> {
        let r = self.cards.get(card)?;
        let (lo, hi, mid, olo, ohi) = match side {
            Side::Top | Side::Bottom => (
                r.x + PORT_INSET,
                r.right() - PORT_INSET,
                r.cx(),
                other.x + PORT_INSET,
                other.right() - PORT_INSET,
            ),
            Side::Left | Side::Right => (
                r.y + PORT_INSET,
                r.bottom() - PORT_INSET,
                r.cy(),
                other.y + PORT_INSET,
                other.bottom() - PORT_INSET,
            ),
        };
        let (a, b) = (lo.max(olo), hi.min(ohi));
        let want = if b >= a { (a + b) / 2.0 } else { mid }.round();
        let used: Vec<(f64, f64)> = self
            .ports
            .iter()
            .filter(|(c, s, _, _)| *c == card && *s == side)
            .map(|p| (p.2, p.3.max(gap)))
            .collect();
        (0..64)
            .map(|k| {
                let step = f64::from((k + 1) / 2) * 12.0;
                if k % 2 == 1 {
                    want + step
                } else {
                    want - step
                }
            })
            .filter(|x| *x >= lo - EPS && *x <= hi + EPS)
            .filter(|x| used.iter().all(|(u, g)| (u - x).abs() >= g - EPS))
            // a stub on (or right beside) an earlier route would join it
            .find(|x| {
                let stub = side.stub(r, *x);
                self.segs
                    .iter()
                    .all(|(p, q)| seg_dist(stub, *p, *q) >= TRACK - EPS)
            })
    }

    fn ends(&self, card: usize, other: usize, sides: &[Side], gap: f64) -> Vec<End> {
        let (Some(r), Some(o)) = (self.cards.get(card), self.cards.get(other)) else {
            return Vec::new();
        };
        sides
            .iter()
            .filter_map(|s| {
                let along = self.slot(card, *s, o, gap)?;
                Some(End {
                    side: *s,
                    along,
                    stub: s.stub(r, along),
                })
            })
            .collect()
    }

    /// Routes an edge from card `a` to card `b` and records it, so later
    /// edges avoid it. Its ports keep `gap` (at least [`PORT_GAP`]) from the
    /// other ports on their sides, so an edge with a label leaves room for it
    /// beside a parallel neighbor.
    pub fn route(&mut self, a: usize, b: usize, gap: f64) -> Routed {
        let gap = gap.max(PORT_GAP);
        let (Some(ra), Some(rb)) = (self.cards.get(a).copied(), self.cards.get(b).copied()) else {
            return Routed {
                points: Vec::new(),
                fallback: true,
            };
        };
        // a loop leaves on the right and comes back on the top
        let (src_sides, dst_sides): (&[Side], &[Side]) = if a == b {
            (&[Side::Right], &[Side::Top])
        } else {
            (&Side::ALL, &Side::ALL)
        };
        let srcs = self.ends(a, b, src_sides, gap);
        let dsts = self.ends(b, a, dst_sides, gap);
        let hard: Vec<R> = self.cards.iter().map(|r| r.inflate(CLEAR)).collect();
        let mut with_soft = hard.clone();
        with_soft.extend(self.soft.iter().copied());
        let found = self
            .solve(&ra, &rb, &srcs, &dsts, &with_soft)
            .or_else(|| self.solve(&ra, &rb, &srcs, &dsts, &hard));
        let routed = match found {
            Some((points, s, t)) => {
                self.ports.push((a, s.side, s.along, gap));
                self.ports.push((b, t.side, t.along, gap));
                Routed {
                    points,
                    fallback: false,
                }
            }
            None => {
                let points = if a == b {
                    // a loop around its own top right corner, outside the card
                    let (x, y) = (ra.right() + CLEAR, ra.y - CLEAR);
                    vec![
                        (ra.right(), ra.cy()),
                        (x, ra.cy()),
                        (x, y),
                        (ra.cx(), y),
                        (ra.cx(), ra.y),
                    ]
                } else {
                    vec![
                        border_toward(&ra, (rb.cx(), rb.cy())),
                        border_toward(&rb, (ra.cx(), ra.cy())),
                    ]
                };
                Routed {
                    points,
                    fallback: true,
                }
            }
        };
        for w in routed.points.windows(2) {
            self.segs.push((w[0], w[1]));
        }
        routed
    }

    /// Adds the cost of running closer than [`TRACK`] along the axis-aligned
    /// line `p`-`q` to the grid's moves.
    fn add_share(p: Pt, q: Pt, xs: &[f64], ys: &[f64], h_share: &mut [f64], v_share: &mut [f64]) {
        let ny = ys.len();
        if (p.1 - q.1).abs() < EPS {
            let (x0, x1) = (p.0.min(q.0), p.0.max(q.0));
            for (j, y) in ys.iter().enumerate() {
                if (y - p.1).abs() < TRACK - EPS {
                    for i in 0..xs.len().saturating_sub(1) {
                        let o = xs[i + 1].min(x1) - xs[i].max(x0);
                        if o > 0.0 {
                            h_share[i * ny + j] += o * SHARE;
                        }
                    }
                }
            }
        } else if (p.0 - q.0).abs() < EPS {
            let (y0, y1) = (p.1.min(q.1), p.1.max(q.1));
            for (i, x) in xs.iter().enumerate() {
                if (x - p.0).abs() < TRACK - EPS {
                    for j in 0..ny.saturating_sub(1) {
                        let o = ys[j + 1].min(y1) - ys[j].max(y0);
                        if o > 0.0 {
                            v_share[i * ny + j] += o * SHARE;
                        }
                    }
                }
            }
        }
    }

    /// A* over the grid from any source stub to any target stub.
    fn solve(
        &self,
        ra: &R,
        rb: &R,
        srcs: &[End],
        dsts: &[End],
        obst: &[R],
    ) -> Option<(Vec<Pt>, End, End)> {
        if srcs.is_empty() || dsts.is_empty() {
            return None;
        }
        let bd = self.bounds;
        let mut xs = vec![bd.x, bd.right()];
        let mut ys = vec![bd.y, bd.bottom()];
        for o in obst {
            xs.extend([o.x, o.right()]);
            ys.extend([o.y, o.bottom()]);
        }
        // the middles of the gaps between obstacles
        for v in [&mut xs, &mut ys] {
            let mut s = v.clone();
            s.sort_by(f64::total_cmp);
            s.dedup();
            v.extend(s.windows(2).map(|w| (w[0] + w[1]) / 2.0));
        }
        // rings around the whole architecture
        if let Some(first) = self.cards.first() {
            let all = self.cards.iter().fold(*first, |u, r| {
                let x = u.x.min(r.x);
                let y = u.y.min(r.y);
                R {
                    x,
                    y,
                    w: u.right().max(r.right()) - x,
                    h: u.bottom().max(r.bottom()) - y,
                }
            });
            for d in RING {
                let g = all.inflate(CLEAR + d);
                xs.extend([g.x, g.right()]);
                ys.extend([g.y, g.bottom()]);
            }
        }
        for e in srcs.iter().chain(dsts) {
            xs.push(e.stub.0);
            ys.push(e.stub.1);
        }
        // earlier edges and guides: their lines and a track on each side,
        // within the grid bound (tracks go first, then the lines)
        let (mut lx, mut ly, mut tx, mut ty) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (p, q) in self.segs.iter().chain(&self.guides) {
            lx.extend([p.0, q.0]);
            ly.extend([p.1, q.1]);
            if (p.0 - q.0).abs() < EPS {
                tx.extend([p.0 - TRACK, p.0 + TRACK]);
            }
            if (p.1 - q.1).abs() < EPS {
                ty.extend([p.1 - TRACK, p.1 + TRACK]);
            }
        }
        let bounded = |base: Vec<f64>, lines: Vec<f64>, tracks: Vec<f64>, lo: f64, hi: f64| {
            let full: Vec<f64> = base.iter().chain(&lines).chain(&tracks).copied().collect();
            let full = axis(full, lo, hi);
            if full.len() <= MAX_AXIS {
                return full;
            }
            let some: Vec<f64> = base.iter().chain(&lines).copied().collect();
            let some = axis(some, lo, hi);
            if some.len() <= MAX_AXIS {
                return some;
            }
            axis(base, lo, hi)
        };
        let xs = bounded(xs, lx, tx, bd.x, bd.right());
        let ys = bounded(ys, ly, ty, bd.y, bd.bottom());
        let (nx, ny) = (xs.len(), ys.len());
        if nx < 2 || ny < 2 || nx > MAX_GRID_AXIS || ny > MAX_GRID_AXIS {
            return None;
        }
        let at = |i: usize, j: usize| i * ny + j;
        let blocked = |p: Pt| obst.iter().any(|o| strictly_inside(p, o));
        let mut node_ok = vec![false; nx * ny];
        let mut h_ok = vec![false; nx * ny];
        let mut v_ok = vec![false; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                node_ok[at(i, j)] = !blocked((xs[i], ys[j]));
                if i + 1 < nx {
                    h_ok[at(i, j)] = !blocked(((xs[i] + xs[i + 1]) / 2.0, ys[j]));
                }
                if j + 1 < ny {
                    v_ok[at(i, j)] = !blocked((xs[i], (ys[j] + ys[j + 1]) / 2.0));
                }
            }
        }
        // penalties from earlier edges, and for hugging a card's clearance line
        let mut h_share = vec![0.0f64; nx * ny];
        let mut v_share = vec![0.0f64; nx * ny];
        // (by index: every obstacle side is a grid line)
        let span = |v: &[f64], lo: f64, hi: f64| {
            let a = v.partition_point(|x| *x < lo - EPS);
            let b = v.partition_point(|x| *x <= hi + EPS);
            a..b
        };
        for o in obst.iter().take(self.cards.len()) {
            let cols = span(&xs, o.x, o.right());
            for y in [o.y, o.bottom()] {
                if let Some(j) = index_of(&ys, y) {
                    for i in cols.clone() {
                        if i + 1 < cols.end {
                            h_share[at(i, j)] += (xs[i + 1] - xs[i]) * HUG;
                        }
                    }
                }
            }
            let rows = span(&ys, o.y, o.bottom());
            for x in [o.x, o.right()] {
                if let Some(i) = index_of(&xs, x) {
                    for j in rows.clone() {
                        if j + 1 < rows.end {
                            v_share[at(i, j)] += (ys[j + 1] - ys[j]) * HUG;
                        }
                    }
                }
            }
        }
        // entering a node lying on an earlier vertical (horizontal) segment by
        // a horizontal (vertical) move crosses it
        let mut cross_v = vec![0u32; nx * ny];
        let mut cross_h = vec![0u32; nx * ny];
        let mut touch = vec![0u32; nx * ny];
        for (p, q) in &self.guides {
            Self::add_share(*p, *q, &xs, &ys, &mut h_share, &mut v_share);
        }
        for (p, q) in &self.segs {
            Self::add_share(*p, *q, &xs, &ys, &mut h_share, &mut v_share);
            for v in [p, q] {
                if let (Some(i), Some(j)) = (index_of(&xs, v.0), index_of(&ys, v.1)) {
                    touch[at(i, j)] += 1;
                }
            }
            if (p.1 - q.1).abs() < EPS {
                let (x0, x1) = (p.0.min(q.0), p.0.max(q.0));
                if let Some(j) = index_of(&ys, p.1) {
                    for i in 0..nx {
                        if xs[i] >= x0 - EPS && xs[i] <= x1 + EPS {
                            cross_h[at(i, j)] += 1;
                        }
                    }
                }
            } else if (p.0 - q.0).abs() < EPS {
                let (y0, y1) = (p.1.min(q.1), p.1.max(q.1));
                if let Some(i) = index_of(&xs, p.0) {
                    for j in 0..ny {
                        if ys[j] >= y0 - EPS && ys[j] <= y1 + EPS {
                            cross_v[at(i, j)] += 1;
                        }
                    }
                }
            }
        }

        let states = nx * ny * 4;
        let mut g = vec![f64::INFINITY; states];
        let mut parent = vec![usize::MAX; states];
        let mut heap = BinaryHeap::new();
        let targets: Vec<(usize, usize, &End)> = dsts
            .iter()
            .filter_map(|e| {
                let i = index_of(&xs, e.stub.0)?;
                let j = index_of(&ys, e.stub.1)?;
                node_ok[at(i, j)].then_some((at(i, j), e.side.out() ^ 1, e))
            })
            .collect();
        if targets.is_empty() {
            return None;
        }
        let h = |i: usize, j: usize| {
            targets
                .iter()
                .map(|(_, _, e)| (xs[i] - e.stub.0).abs() + (ys[j] - e.stub.1).abs())
                .fold(f64::INFINITY, f64::min)
        };
        let mut starts: Vec<(usize, &End)> = Vec::new();
        for e in srcs {
            let (Some(i), Some(j)) = (index_of(&xs, e.stub.0), index_of(&ys, e.stub.1)) else {
                continue;
            };
            if !node_ok[at(i, j)] {
                continue;
            }
            let s = at(i, j) * 4 + e.side.out();
            let on = cross_h[at(i, j)] + cross_v[at(i, j)];
            let g0 = CLEAR
                + CROSS * f64::from(on)
                + TOUCH * f64::from(touch[at(i, j)])
                + if e.side.faces(ra, rb) { 0.0 } else { BEND };
            if g0 < g[s] {
                g[s] = g0;
                heap.push(Open {
                    f: g0 + h(i, j),
                    g: g0,
                    state: s,
                });
                starts.push((s, e));
            }
        }
        let mut best: Option<(f64, usize, &End)> = None;
        while let Some(Open { f, g: gs, state }) = heap.pop() {
            if gs > g[state] {
                continue;
            }
            if best.is_some_and(|(b, _, _)| f >= b) {
                break;
            }
            let (node, head) = (state / 4, state % 4);
            let (i, j) = (node / ny, node % ny);
            for (tn, inward, e) in &targets {
                if *tn == node {
                    let on = cross_h[node] + cross_v[node];
                    let total = gs
                        + CLEAR
                        + CROSS * f64::from(on)
                        + TOUCH * f64::from(touch[node])
                        + if head == *inward { 0.0 } else { BEND };
                    if best.is_none_or(|(b, _, _)| total < b) {
                        best = Some((total, state, e));
                    }
                }
            }
            for nh in 0..4 {
                if nh == head ^ 1 {
                    continue;
                }
                let (ni, nj, len, share, ok) = match nh {
                    0 if i + 1 < nx => (
                        i + 1,
                        j,
                        xs[i + 1] - xs[i],
                        h_share[at(i, j)],
                        h_ok[at(i, j)],
                    ),
                    1 if i > 0 => (
                        i - 1,
                        j,
                        xs[i] - xs[i - 1],
                        h_share[at(i - 1, j)],
                        h_ok[at(i - 1, j)],
                    ),
                    2 if j + 1 < ny => (
                        i,
                        j + 1,
                        ys[j + 1] - ys[j],
                        v_share[at(i, j)],
                        v_ok[at(i, j)],
                    ),
                    3 if j > 0 => (
                        i,
                        j - 1,
                        ys[j] - ys[j - 1],
                        v_share[at(i, j - 1)],
                        v_ok[at(i, j - 1)],
                    ),
                    _ => continue,
                };
                let n2 = at(ni, nj);
                if !ok || !node_ok[n2] {
                    continue;
                }
                let crossings = if nh < 2 { cross_v[n2] } else { cross_h[n2] };
                let cost = len
                    + share
                    + CROSS * f64::from(crossings)
                    + TOUCH * f64::from(touch[n2])
                    + if nh == head { 0.0 } else { BEND };
                let s2 = n2 * 4 + nh;
                let g2 = gs + cost;
                if g2 < g[s2] {
                    g[s2] = g2;
                    parent[s2] = state;
                    heap.push(Open {
                        f: g2 + h(ni, nj),
                        g: g2,
                        state: s2,
                    });
                }
            }
        }
        let (_, last, dst) = best?;
        let mut chain = vec![last];
        let mut s = last;
        while parent[s] != usize::MAX {
            s = parent[s];
            chain.push(s);
        }
        chain.reverse();
        let first = chain[0];
        let src = starts.iter().find(|(s, _)| *s == first).map(|(_, e)| **e)?;
        let mut pts = vec![src.side.port(ra, src.along)];
        pts.extend(chain.iter().map(|s| {
            let n = s / 4;
            (xs[n / ny], ys[n % ny])
        }));
        pts.push(dst.side.port(rb, dst.along));
        Some((simplify(&pts), src, *dst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(x: f64, y: f64) -> R {
        R {
            x,
            y,
            w: 220.0,
            h: 132.0,
        }
    }

    fn bounds() -> R {
        R {
            x: 20.0,
            y: 140.0,
            w: 1600.0,
            h: 1000.0,
        }
    }

    fn hits(p: Pt, q: Pt, r: &R) -> bool {
        crate::scene::line_hits(&(p.0, p.1, q.0, q.1), r)
    }

    fn orthogonal(pts: &[Pt]) -> bool {
        pts.windows(2)
            .all(|w| (w[0].0 - w[1].0).abs() < EPS || (w[0].1 - w[1].1).abs() < EPS)
    }

    #[test]
    fn simplify_drops_collinear_and_repeated_points() {
        let p = simplify(&[(0.0, 0.0), (0.0, 0.0), (0.0, 5.0), (0.0, 9.0), (4.0, 9.0)]);
        assert_eq!(p, vec![(0.0, 0.0), (0.0, 9.0), (4.0, 9.0)]);
    }

    #[test]
    fn aligned_cards_get_a_straight_line() {
        let cards = vec![card(100.0, 200.0), card(100.0, 500.0)];
        let mut r = Router::new(cards, Vec::new(), Vec::new(), bounds());
        let out = r.route(0, 1, PORT_GAP);
        assert!(!out.fallback);
        assert_eq!(out.points, vec![(210.0, 332.0), (210.0, 500.0)]);
    }

    #[test]
    fn a_card_in_the_way_is_routed_around() {
        let cards = vec![card(100.0, 200.0), card(100.0, 450.0), card(100.0, 700.0)];
        let mut r = Router::new(cards.clone(), Vec::new(), Vec::new(), bounds());
        let out = r.route(0, 2, PORT_GAP);
        assert!(!out.fallback);
        assert!(orthogonal(&out.points), "{:?}", out.points);
        for w in out.points.windows(2) {
            for c in &cards {
                assert!(!hits(w[0], w[1], c), "{:?} through {c:?}", out.points);
            }
        }
        assert!(out.points.len() >= 4, "{:?}", out.points);
    }

    #[test]
    fn parallel_edges_use_separate_ports_and_tracks() {
        let cards = vec![card(100.0, 200.0), card(600.0, 200.0)];
        let mut r = Router::new(cards, Vec::new(), Vec::new(), bounds());
        let a = r.route(0, 1, PORT_GAP);
        let b = r.route(0, 1, PORT_GAP);
        let c = r.route(1, 0, PORT_GAP);
        for (p, q) in [(&a, &b), (&a, &c), (&b, &c)] {
            assert!(!p.fallback && !q.fallback);
            for s in p.points.windows(2) {
                for t in q.points.windows(2) {
                    let both_h = (s[0].1 - s[1].1).abs() < EPS && (t[0].1 - t[1].1).abs() < EPS;
                    if both_h && (s[0].1 - t[0].1).abs() < TRACK {
                        let o = s[0].0.max(s[1].0).min(t[0].0.max(t[1].0))
                            - s[0].0.min(s[1].0).max(t[0].0.min(t[1].0));
                        assert!(o <= 0.0, "{:?} runs along {:?}", p.points, q.points);
                    }
                }
            }
        }
    }

    #[test]
    fn no_room_falls_back_to_a_straight_line() {
        // the target is walled in by cards on every side
        let cards = vec![
            card(100.0, 150.0),
            card(700.0, 500.0),
            card(700.0, 380.0),
            card(700.0, 620.0),
            card(480.0, 500.0),
            card(920.0, 500.0),
        ];
        let tight = R {
            x: 60.0,
            y: 140.0,
            w: 1200.0,
            h: 640.0,
        };
        let mut r = Router::new(cards, Vec::new(), Vec::new(), tight);
        let out = r.route(0, 1, PORT_GAP);
        assert!(out.fallback);
        assert_eq!(out.points.len(), 2);
    }
}
