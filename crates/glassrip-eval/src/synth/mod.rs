//! Deterministic generator for the public synthetic fixtures (spec 9.1).
//!
//! Everything here is fictional: boards, names, products, and pages are made up
//! for the fixtures. Rendering uses only integer pixel operations and the
//! public-domain `font8x8` glyphs, so the same generator version produces
//! byte-identical PNGs on every platform. Run it with
//! `cargo run -p glassrip-eval --example gen_synthetic`.

pub mod boards;
pub mod docs;

use std::path::{Path, PathBuf};

use font8x8::{UnicodeFonts, BASIC_FONTS};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageEncoder, RgbImage};

use crate::error::{write_json, write_text, EvalError, Result};
use crate::fixture::CaseMeta;

/// Generator version recorded in every `meta.toml`.
pub const GENERATOR_VERSION: u32 = 1;
/// Generator name recorded in every `meta.toml`.
pub const GENERATOR_NAME: &str = "gen_synthetic";

/// RGB color.
pub type Rgb = [u8; 3];

/// Common colors.
pub mod color {
    use super::Rgb;
    /// White.
    pub const WHITE: Rgb = [255, 255, 255];
    /// Near-black text.
    pub const INK: Rgb = [28, 30, 36];
    /// Mid gray text.
    pub const GRAY: Rgb = [110, 114, 124];
    /// Light panel gray.
    pub const PANEL: Rgb = [242, 243, 246];
    /// Border gray.
    pub const BORDER: Rgb = [206, 209, 216];
    /// Browser chrome dark.
    pub const CHROME: Rgb = [48, 50, 56];
    /// Chrome text.
    pub const CHROME_TEXT: Rgb = [225, 227, 232];
    /// Node outline blue.
    pub const NODE: Rgb = [37, 78, 160];
    /// Edge line.
    pub const EDGE: Rgb = [60, 64, 72];
    /// Sticky yellow.
    pub const YELLOW: Rgb = [255, 236, 140];
    /// Sticky pink.
    pub const PINK: Rgb = [255, 196, 214];
    /// Sticky blue.
    pub const BLUE: Rgb = [186, 218, 255];
    /// Owner tag green.
    pub const GREEN: Rgb = [46, 160, 88];
    /// Video tile background.
    pub const TILE: Rgb = [38, 40, 46];
}

/// SplitMix64: tiny, fast, and fully specified, so seeds reproduce everywhere.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Seeded generator.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `lo..=hi` (returns `lo` when `hi <= lo`).
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }
}

/// Glyph cell size in font pixels.
pub const GLYPH: i64 = 8;

/// An RGB drawing surface with clipped integer primitives.
#[derive(Debug, Clone)]
pub struct Canvas {
    img: RgbImage,
}

impl Canvas {
    /// A `w` x `h` canvas filled with `bg`.
    pub fn new(w: u32, h: u32, bg: Rgb) -> Self {
        Self {
            img: RgbImage::from_pixel(w, h, image::Rgb(bg)),
        }
    }

    /// Width.
    pub fn width(&self) -> i64 {
        i64::from(self.img.width())
    }

    /// Height.
    pub fn height(&self) -> i64 {
        i64::from(self.img.height())
    }

    /// The image.
    pub fn image(&self) -> &RgbImage {
        &self.img
    }

    /// Consumes the canvas.
    pub fn into_image(self) -> RgbImage {
        self.img
    }

    /// Sets one pixel (clipped).
    pub fn put(&mut self, x: i64, y: i64, c: Rgb) {
        if x >= 0 && y >= 0 && x < self.width() && y < self.height() {
            self.img.put_pixel(x as u32, y as u32, image::Rgb(c));
        }
    }

    /// Filled rectangle.
    pub fn fill_rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: Rgb) {
        let (x0, y0) = (x.max(0), y.max(0));
        let (x1, y1) = ((x + w).min(self.width()), (y + h).min(self.height()));
        for yy in y0..y1 {
            for xx in x0..x1 {
                self.img.put_pixel(xx as u32, yy as u32, image::Rgb(c));
            }
        }
    }

    /// Rectangle outline of thickness `t`.
    pub fn stroke_rect(&mut self, x: i64, y: i64, w: i64, h: i64, t: i64, c: Rgb) {
        self.fill_rect(x, y, w, t, c);
        self.fill_rect(x, y + h - t, w, t, c);
        self.fill_rect(x, y, t, h, c);
        self.fill_rect(x + w - t, y, t, h, c);
    }

    fn stamp(&mut self, x: i64, y: i64, t: i64, c: Rgb) {
        let r = t / 2;
        self.fill_rect(x - r, y - r, t, t, c);
    }

    /// Line of thickness `t`; `dash` = `Some((on, off))` draws a dashed line.
    pub fn line(
        &mut self,
        from: (i64, i64),
        to: (i64, i64),
        t: i64,
        dash: Option<(i64, i64)>,
        c: Rgb,
    ) {
        let (mut x, mut y) = from;
        let (dx, dy) = ((to.0 - x).abs(), -(to.1 - y).abs());
        let (sx, sy) = (if x < to.0 { 1 } else { -1 }, if y < to.1 { 1 } else { -1 });
        let mut err = dx + dy;
        let mut step: i64 = 0;
        loop {
            let on = dash.is_none_or(|(a, b)| step % (a + b) < a);
            if on {
                self.stamp(x, y, t, c);
            }
            if x == to.0 && y == to.1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
            step += 1;
        }
    }

    /// Filled triangle (pixel centers inside by edge functions).
    pub fn fill_triangle(&mut self, p: [(i64, i64); 3], c: Rgb) {
        let min_x = p.iter().map(|q| q.0).min().unwrap_or(0);
        let max_x = p.iter().map(|q| q.0).max().unwrap_or(0);
        let min_y = p.iter().map(|q| q.1).min().unwrap_or(0);
        let max_y = p.iter().map(|q| q.1).max().unwrap_or(0);
        let edge = |a: (i64, i64), b: (i64, i64), x: i64, y: i64| {
            (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0)
        };
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let e0 = edge(p[0], p[1], x, y);
                let e1 = edge(p[1], p[2], x, y);
                let e2 = edge(p[2], p[0], x, y);
                if (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0) {
                    self.put(x, y, c);
                }
            }
        }
    }

    /// Draws `text` with its top-left at `(x, y)`; returns the drawn width.
    pub fn text(&mut self, x: i64, y: i64, text: &str, scale: i64, c: Rgb) -> i64 {
        let mut cx = x;
        for ch in text.chars() {
            let glyph = BASIC_FONTS.get(ch).or_else(|| BASIC_FONTS.get('?'));
            if let Some(rows) = glyph {
                for (ry, bits) in rows.iter().enumerate() {
                    for rx in 0..8 {
                        if bits & (1 << rx) != 0 {
                            self.fill_rect(cx + rx * scale, y + ry as i64 * scale, scale, scale, c);
                        }
                    }
                }
            }
            cx += GLYPH * scale;
        }
        cx - x
    }
}

/// Width of `text` at `scale`.
pub fn text_width(text: &str, scale: i64) -> i64 {
    text.chars().count() as i64 * GLYPH * scale
}

/// Line height at `scale` (glyph plus leading).
pub fn line_height(scale: i64) -> i64 {
    GLYPH * scale + 3 * scale
}

/// Greedy word wrap to at most `max_chars` per line (long words are kept whole).
pub fn wrap(text: &str, max_chars: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for w in text.split_whitespace() {
        let extra = if cur.is_empty() {
            w.len()
        } else {
            cur.len() + 1 + w.len()
        };
        if extra > max_chars && !cur.is_empty() {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Encodes a PNG deterministically (best compression, adaptive filter).
pub fn png_bytes(img: &RgbImage) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    PngEncoder::new_with_quality(&mut buf, CompressionType::Best, FilterType::Adaptive)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| EvalError::Image {
            path: PathBuf::from("<memory>"),
            message: e.to_string(),
        })?;
    Ok(buf)
}

fn write_png(path: &Path, img: &RgbImage) -> Result<()> {
    let bytes = png_bytes(img)?;
    if let Some(parent) = path.parent() {
        fs_err::create_dir_all(parent).map_err(|e| EvalError::io(parent, e))?;
    }
    fs_err::write(path, bytes).map_err(|e| EvalError::io(path, e))
}

fn write_meta(dir: &Path, meta: &CaseMeta) -> Result<()> {
    let text = toml::to_string(meta).map_err(|e| EvalError::Other(format!("meta.toml: {e}")))?;
    write_text(&dir.join("meta.toml"), &text)
}

/// Files written by [`generate_all`].
#[derive(Debug, Default)]
pub struct Generated {
    /// Every file written, relative to the output root.
    pub files: Vec<PathBuf>,
    /// Total bytes written.
    pub bytes: u64,
}

impl Generated {
    fn add(&mut self, root: &Path, path: &Path) {
        if let Ok(m) = fs_err::metadata(path) {
            self.bytes += m.len();
        }
        self.files
            .push(path.strip_prefix(root).unwrap_or(path).to_path_buf());
    }
}

/// Writes every public fixture under `root` (`root/synthetic/...` and
/// `root/synthetic_docs/...`). Existing files are overwritten; the caller decides
/// whether to clear stale cases first.
pub fn generate_all(root: &Path) -> Result<Generated> {
    let mut out = Generated::default();
    for case in boards::all_cases() {
        let dir = root.join("synthetic").join(&case.meta.case);
        let rendered = boards::render(&case);
        let frame = dir.join("frame.png");
        write_png(&frame, rendered.canvas.image())?;
        write_json(&dir.join("expected.json"), &rendered.expected)?;
        write_meta(&dir, &case.meta)?;
        for f in [frame, dir.join("expected.json"), dir.join("meta.toml")] {
            out.add(root, &f);
        }
    }
    for case in docs::all_cases() {
        let dir = root.join("synthetic_docs").join(&case.meta.case);
        let rendered = docs::render(&case);
        for (frame, img) in rendered.expected.frames.iter().zip(&rendered.frames) {
            let p = dir.join(&frame.file);
            write_png(&p, img)?;
            out.add(root, &p);
        }
        write_json(&dir.join("expected.json"), &rendered.expected)?;
        write_meta(&dir, &case.meta)?;
        for f in [dir.join("expected.json"), dir.join("meta.toml")] {
            out.add(root, &f);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_reproducible() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let xs: Vec<u64> = (0..4).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..4).map(|_| b.next_u64()).collect();
        assert_eq!(xs, ys);
        // SplitMix64 reference value for seed 0 (first output).
        assert_eq!(Rng::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
        let mut r = Rng::new(1);
        for _ in 0..100 {
            let v = r.range(-3, 3);
            assert!((-3..=3).contains(&v));
        }
        assert_eq!(r.range(5, 5), 5);
    }

    #[test]
    fn wrap_and_measure() {
        assert_eq!(wrap("alpha beta gamma", 10), vec!["alpha beta", "gamma"]);
        assert_eq!(wrap("", 10), Vec::<String>::new());
        assert_eq!(
            wrap("supercalifragilistic x", 5),
            vec!["supercalifragilistic", "x"]
        );
        assert_eq!(text_width("abc", 2), 48);
    }

    #[test]
    fn primitives_draw_expected_pixels() {
        let mut c = Canvas::new(20, 20, color::WHITE);
        c.fill_rect(2, 2, 3, 3, color::INK);
        assert_eq!(c.image().get_pixel(3, 3).0, color::INK);
        assert_eq!(c.image().get_pixel(5, 5).0, color::WHITE);
        c.fill_rect(-5, -5, 3, 3, color::INK); // fully clipped
        c.line((0, 10), (19, 10), 1, Some((2, 2)), color::GREEN);
        assert_eq!(c.image().get_pixel(0, 10).0, color::GREEN);
        assert_eq!(c.image().get_pixel(2, 10).0, color::WHITE);
        let w = c.text(0, 0, "I", 1, color::INK);
        assert_eq!(w, 8);
    }
}
