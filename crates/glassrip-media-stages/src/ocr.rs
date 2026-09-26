//! OCR checks for `orient` (spec 6.2).
//!
//! - **180 degree confirmation** (ONNX features): text lines are found with a PP-OCR
//!   detector on the frame turned by the voted correction; each line is recognized with the
//!   PP-OCRv5 English recognizer as is and turned 180 degrees, and the mean CTC confidence of
//!   the two readings is compared. Upright text reads with clearly higher confidence.
//! - **Pure-Rust fallback** (`ocrs` feature, no ONNX): every sample is read with `ocrs` at
//!   the four rotations and each rotation is scored by dictionary hits.
//!
//! The decision rules and post-processing are plain functions, tested without models.

use std::collections::HashSet;
use std::sync::OnceLock;

use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use image::RgbImage;

use crate::schema::ConfirmStatus;

/// An axis-aligned text line box in source-image pixels (`x1`, `y1` exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineBox {
    /// Left.
    pub x0: u32,
    /// Top.
    pub y0: u32,
    /// Right (exclusive).
    pub x1: u32,
    /// Bottom (exclusive).
    pub y1: u32,
}

impl LineBox {
    /// Width.
    pub fn w(&self) -> u32 {
        self.x1 - self.x0
    }
    /// Height.
    pub fn h(&self) -> u32 {
        self.y1 - self.y0
    }
}

/// Text line boxes from a detector probability map (`w x h`, row-major): threshold, 4-connected
/// components, DB-style unclip (grow by `area * 1.5 / perimeter`), keep horizontal lines at
/// least 4 map pixels tall, scale to the source image, largest first, at most `max`.
#[allow(clippy::too_many_arguments)]
pub fn boxes_from_prob(
    prob: &[f32],
    w: usize,
    h: usize,
    thresh: f32,
    src_w: u32,
    src_h: u32,
    max: usize,
) -> Vec<LineBox> {
    let mut seen = vec![false; w * h];
    let mut out: Vec<(usize, LineBox)> = Vec::new();
    let (sx, sy) = (f64::from(src_w) / w as f64, f64::from(src_h) / h as f64);
    let mut stack = Vec::new();
    for start in 0..w * h {
        if seen[start] || prob[start] <= thresh {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1, mut n) = (w, h, 0usize, 0usize, 0usize);
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            n += 1;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            let mut push = |j: usize| {
                if !seen[j] && prob[j] > thresh {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                push(i - 1);
            }
            if x + 1 < w {
                push(i + 1);
            }
            if y > 0 {
                push(i - w);
            }
            if y + 1 < h {
                push(i + w);
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f64, (y1 - y0 + 1) as f64);
        if bh < 4.0 || bw < 1.5 * bh || n < 20 {
            continue;
        }
        let d = n as f64 * 1.5 / (2.0 * (bw + bh));
        let fx0 = ((x0 as f64 - d) * sx).max(0.0);
        let fy0 = ((y0 as f64 - d) * sy).max(0.0);
        let fx1 = (((x1 + 1) as f64 + d) * sx).min(f64::from(src_w));
        let fy1 = (((y1 + 1) as f64 + d) * sy).min(f64::from(src_h));
        let b = LineBox {
            x0: fx0 as u32,
            y0: fy0 as u32,
            x1: fx1 as u32,
            y1: fy1 as u32,
        };
        if b.w() >= 8 && b.h() >= 4 {
            out.push((b.w() as usize * b.h() as usize, b));
        }
    }
    out.sort_by_key(|x| std::cmp::Reverse(x.0));
    out.into_iter().take(max).map(|(_, b)| b).collect()
}

/// Mean max-probability over non-blank CTC steps (blank = class 0) and the number of such
/// steps, for a `t x c` output. Rows that are not probability distributions are softmaxed.
pub fn ctc_confidence(out: &[f32], t: usize, c: usize) -> (f64, usize) {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for row in out.chunks(c).take(t) {
        let total: f32 = row.iter().sum();
        let is_prob = row.iter().all(|v| (0.0..=1.0).contains(v)) && (total - 1.0).abs() < 1e-2;
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, v) in row.iter().enumerate() {
            if *v > bv {
                bv = *v;
                bi = i;
            }
        }
        if bi == 0 {
            continue;
        }
        let p = if is_prob {
            f64::from(bv)
        } else {
            let e: f64 = row.iter().map(|v| f64::from(v - bv).exp()).sum();
            1.0 / e
        };
        sum += p;
        n += 1;
    }
    (if n == 0 { 0.0 } else { sum / n as f64 }, n)
}

/// Confirmation scores for a chosen rotation against its 180 degree flip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlipScores {
    /// Mean line confidence as chosen.
    pub chosen: f64,
    /// Mean line confidence turned 180 degrees.
    pub flipped: f64,
    /// Lines compared.
    pub lines: usize,
}

/// Accepts the chosen rotation when it reads better than its flip by `margin`; with fewer
/// than `min_lines` lines there is nothing to confirm with (the vote stands). Otherwise an
/// error naming both candidates.
pub fn confirm_flip(
    s: FlipScores,
    chosen_deg: u32,
    min_lines: usize,
    margin: f64,
) -> Result<ConfirmStatus, ErrorInfo> {
    if s.lines < min_lines {
        return Ok(ConfirmStatus::InsufficientText);
    }
    if s.chosen - s.flipped >= margin {
        return Ok(ConfirmStatus::Confirmed);
    }
    Err(ErrorInfo::new(
        ErrorCode::Validation,
        format!(
            "orientation ambiguous between {chosen_deg} and {} degrees: text recognition confidence {:.3} vs {:.3} over {} lines (margin {margin}); set orient.override_rotation_deg to proceed",
            (chosen_deg + 180) % 360,
            s.chosen,
            s.flipped,
            s.lines
        ),
    ))
}

/// Common English and UI words used to score the `ocrs` fallback (lowercase, 3+ letters).
const WORDS: &str = "the and for are but not you all any can had her was one our out day get has him his how man new now old see two way who did its let put say she too use \
that with have this will your from they know want been good much some time very when come here just like long make many more only over such take than them well were \
what about after again also back because before being below between both could down each even every first give going great into last little look most must never next \
other people right same should since still their there these thing think those through under until where which while would write year years work world \
add edit file view help open close save share search find home page settings window tools insert format create delete update select copy paste undo redo new list \
table text image link user name email account profile message messages chat meeting call video audio join leave start stop next previous back forward done cancel \
content design system server service app application mobile frontend backend data model models api test tests notes note task tasks board team project projects \
status review draft publish article articles section sections item items field fields value values type types code build run version release issue issues comment \
comments document documents folder folders date time today calendar week month report reports option options filter sort group view views member members overview browse";

fn dictionary() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| WORDS.split_whitespace().collect())
}

/// Number of dictionary words (3+ letters, case-insensitive) in `text`.
pub fn dictionary_hits(text: &str) -> u32 {
    let dict = dictionary();
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| w.len() >= 3)
        .filter(|w| dict.contains(w.to_ascii_lowercase().as_str()))
        .count() as u32
}

/// Picks the rotation with the most dictionary hits (`hits[i]` for corrections
/// `[0, 90, 180, 270]`). Needs `min_hits` and at least `min_ratio` times the runner-up;
/// otherwise an error naming the candidates.
pub fn decide_by_hits(hits: [u32; 4], min_hits: u32, min_ratio: f64) -> Result<u32, ErrorInfo> {
    let rot = crate::orient::ROTATIONS;
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|a, b| hits[*b].cmp(&hits[*a]).then(a.cmp(b)));
    let (w, s) = (order[0], order[1]);
    let describe = format!(
        "dictionary hits 0={} 90={} 180={} 270={}",
        hits[0], hits[1], hits[2], hits[3]
    );
    if hits[w] < min_hits {
        return Err(ErrorInfo::new(
            ErrorCode::Validation,
            format!(
                "orientation undecided: best rotation {} has {} dictionary hits, need {min_hits} ({describe}); set orient.override_rotation_deg to proceed",
                rot[w], hits[w]
            ),
        ));
    }
    if f64::from(hits[w]) < min_ratio * f64::from(hits[s]) {
        return Err(ErrorInfo::new(
            ErrorCode::Validation,
            format!(
                "orientation ambiguous between {} and {} degrees ({describe}); set orient.override_rotation_deg to proceed",
                rot[w], rot[s]
            ),
        ));
    }
    Ok(rot[w])
}

/// Turns an image clockwise by `deg` (0, 90, 180, 270).
pub fn rotate_cw(img: &RgbImage, deg: u32) -> RgbImage {
    match deg {
        90 => image::imageops::rotate90(img),
        180 => image::imageops::rotate180(img),
        270 => image::imageops::rotate270(img),
        _ => img.clone(),
    }
}

#[cfg(any(feature = "onnx", feature = "onnx-dynamic"))]
pub use onnx_impl::*;

#[cfg(any(feature = "onnx", feature = "onnx-dynamic"))]
mod onnx_impl {
    use super::*;
    use ort::session::Session;
    use ort::value::Tensor;

    fn err(e: &dyn std::fmt::Display) -> ErrorInfo {
        ErrorInfo::new(ErrorCode::ModelRequest, format!("onnx runtime: {e}"))
    }

    /// Opens an ONNX session (CUDA provider with the `cuda` feature); returns the provider.
    pub fn onnx_session(path: &std::path::Path) -> Result<(Session, &'static str), ErrorInfo> {
        let builder = Session::builder().map_err(|e| err(&e))?;
        #[cfg(feature = "cuda")]
        let builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| err(&e))?;
        let mut builder = builder;
        let s = builder.commit_from_file(path).map_err(|e| err(&e))?;
        Ok((
            s,
            if cfg!(feature = "cuda") {
                "cuda"
            } else {
                "cpu"
            },
        ))
    }

    /// PP-OCR detector plus PP-OCRv5 recognizer.
    pub struct PpOcr {
        det: Session,
        rec: Session,
        rec_h: u32,
    }

    const DET_MAX_SIDE: u32 = 960;
    const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    impl PpOcr {
        /// Loads both models.
        pub fn new(det: &std::path::Path, rec: &std::path::Path) -> Result<Self, ErrorInfo> {
            let (det, _) = onnx_session(det)?;
            let (rec, _) = onnx_session(rec)?;
            let rec_h = rec
                .inputs()
                .first()
                .and_then(|i| i.dtype().tensor_shape().and_then(|s| s.get(2).copied()))
                .filter(|h| *h > 0)
                .map_or(48, |h| h as u32);
            Ok(Self { det, rec, rec_h })
        }

        /// Text line boxes in `img` (BGR input, ImageNet normalization, as PaddleOCR).
        pub fn detect(&mut self, img: &RgbImage, max: usize) -> Result<Vec<LineBox>, ErrorInfo> {
            let (w, h) = img.dimensions();
            let s = (f64::from(DET_MAX_SIDE) / f64::from(w.max(h))).min(1.0);
            let r32 = |v: f64| (v.round() as u32).max(32).div_ceil(32) * 32;
            let (dw, dh) = (r32(f64::from(w) * s), r32(f64::from(h) * s));
            let r = image::imageops::resize(img, dw, dh, image::imageops::FilterType::Triangle);
            let n = (dw * dh) as usize;
            let mut t = vec![0f32; 3 * n];
            for (i, p) in r.pixels().enumerate() {
                for c in 0..3 {
                    // Channel order B, G, R.
                    let v = f32::from(p[2 - c]) / 255.0;
                    t[c * n + i] = (v - DET_MEAN[c]) / DET_STD[c];
                }
            }
            let input = Tensor::from_array(([1usize, 3, dh as usize, dw as usize], t))
                .map_err(|e| err(&e))?;
            let out = self.det.run(ort::inputs![input]).map_err(|e| err(&e))?;
            let (shape, data) = out[0].try_extract_tensor::<f32>().map_err(|e| err(&e))?;
            let (mh, mw) = match shape.len() {
                4 => (shape[2] as usize, shape[3] as usize),
                3 => (shape[1] as usize, shape[2] as usize),
                _ => return Err(err(&format!("unexpected detector output {shape:?}"))),
            };
            if data.len() < mh * mw {
                return Err(err(&"detector output too short"));
            }
            Ok(boxes_from_prob(&data[..mh * mw], mw, mh, 0.3, w, h, max))
        }

        /// CTC confidence and non-blank step count for one line image.
        pub fn recognize(&mut self, line: &RgbImage) -> Result<(f64, usize), ErrorInfo> {
            let (w, h) = line.dimensions();
            let rw = ((f64::from(w) * f64::from(self.rec_h) / f64::from(h.max(1))).round() as u32)
                .clamp(16, 1600);
            let r = image::imageops::resize(
                line,
                rw,
                self.rec_h,
                image::imageops::FilterType::Triangle,
            );
            let n = (rw * self.rec_h) as usize;
            let mut t = vec![0f32; 3 * n];
            for (i, p) in r.pixels().enumerate() {
                for c in 0..3 {
                    t[c * n + i] = (f32::from(p[2 - c]) / 255.0 - 0.5) / 0.5;
                }
            }
            let input = Tensor::from_array(([1usize, 3, self.rec_h as usize, rw as usize], t))
                .map_err(|e| err(&e))?;
            let out = self.rec.run(ort::inputs![input]).map_err(|e| err(&e))?;
            let (shape, data) = out[0].try_extract_tensor::<f32>().map_err(|e| err(&e))?;
            if shape.len() != 3 {
                return Err(err(&format!("unexpected recognizer output {shape:?}")));
            }
            Ok(ctc_confidence(data, shape[1] as usize, shape[2] as usize))
        }

        /// Compares every detected line as is and turned 180 degrees. `images` are already
        /// turned by the chosen correction.
        pub fn flip_scores(
            &mut self,
            images: &[RgbImage],
            max_lines: usize,
        ) -> Result<FlipScores, ErrorInfo> {
            let (mut a, mut b, mut lines) = (0.0f64, 0.0f64, 0usize);
            for img in images {
                for bx in self.detect(img, max_lines)? {
                    let crop =
                        image::imageops::crop_imm(img, bx.x0, bx.y0, bx.w(), bx.h()).to_image();
                    let (ca, na) = self.recognize(&crop)?;
                    let (cb, nb) = self.recognize(&image::imageops::rotate180(&crop))?;
                    if na.max(nb) < 2 {
                        continue;
                    }
                    a += ca;
                    b += cb;
                    lines += 1;
                }
            }
            let m = |v: f64| if lines == 0 { 0.0 } else { v / lines as f64 };
            Ok(FlipScores {
                chosen: m(a),
                flipped: m(b),
                lines,
            })
        }
    }
}

/// Dictionary hits per rotation `[0, 90, 180, 270]` with `ocrs` over the given samples.
#[cfg(feature = "ocrs")]
pub fn ocrs_hits(
    det: &std::path::Path,
    rec: &std::path::Path,
    images: &[RgbImage],
) -> Result<[u32; 4], ErrorInfo> {
    use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
    let e =
        |x: &dyn std::fmt::Display| ErrorInfo::new(ErrorCode::ModelRequest, format!("ocrs: {x}"));
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(rten::Model::load_file(det).map_err(|x| e(&x))?),
        recognition_model: Some(rten::Model::load_file(rec).map_err(|x| e(&x))?),
        ..Default::default()
    })
    .map_err(|x| e(&x))?;
    let mut hits = [0u32; 4];
    for img in images {
        for (k, deg) in crate::orient::ROTATIONS.iter().enumerate() {
            let turned = rotate_cw(img, *deg);
            let src =
                ImageSource::from_bytes(turned.as_raw(), turned.dimensions()).map_err(|x| e(&x))?;
            let input = engine.prepare_input(src).map_err(|x| e(&x))?;
            let text = engine.get_text(&input).map_err(|x| e(&x))?;
            hits[k] += dictionary_hits(&text);
        }
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_keep_horizontal_lines_and_scale() {
        // 40x20 map with one 20x4 line and one 3x3 blob.
        let (w, h) = (40usize, 20usize);
        let mut p = vec![0f32; w * h];
        for y in 5..9 {
            for x in 10..30 {
                p[y * w + x] = 0.9;
            }
        }
        for y in 14..17 {
            for x in 2..5 {
                p[y * w + x] = 0.9;
            }
        }
        let b = boxes_from_prob(&p, w, h, 0.3, 80, 40, 10);
        assert_eq!(b.len(), 1);
        // Unclip grows by 80 * 1.5 / 48 = 2.5 map pixels; source is 2x the map.
        assert_eq!(
            b[0],
            LineBox {
                x0: 15,
                y0: 5,
                x1: 65,
                y1: 23
            }
        );
    }

    #[test]
    fn ctc_confidence_skips_blanks_and_softmaxes_logits() {
        let probs = [0.9f32, 0.1, 0.0, 0.2, 0.8, 0.0, 0.1, 0.0, 0.9];
        let (c, n) = ctc_confidence(&probs, 3, 3);
        assert_eq!(n, 2);
        assert!((c - 0.85).abs() < 1e-6);
        let logits = [0.0f32, 10.0, 0.0];
        assert!(ctc_confidence(&logits, 1, 3).0 > 0.99);
    }

    #[test]
    fn flip_confirmation_rules() {
        let s = |chosen, flipped, lines| FlipScores {
            chosen,
            flipped,
            lines,
        };
        assert_eq!(
            confirm_flip(s(0.9, 0.6, 10), 0, 3, 0.05).unwrap(),
            ConfirmStatus::Confirmed
        );
        assert_eq!(
            confirm_flip(s(0.2, 0.9, 1), 0, 3, 0.05).unwrap(),
            ConfirmStatus::InsufficientText
        );
        let e = confirm_flip(s(0.7, 0.9, 10), 90, 3, 0.05).unwrap_err();
        assert!(e.message.contains("between 90 and 270"), "{e}");
        assert!(
            confirm_flip(s(0.82, 0.8, 10), 0, 3, 0.05).is_err(),
            "within margin"
        );
    }

    #[test]
    fn dictionary_scoring_and_decision() {
        assert_eq!(dictionary_hits("The Mobile App, and a new PAGE: hello!"), 6);
        assert_eq!(dictionary_hits("ǝɥʇ dnoɹƃ sı"), 0);
        assert_eq!(decide_by_hits([40, 2, 5, 1], 5, 1.5).unwrap(), 0);
        let e = decide_by_hits([30, 0, 25, 0], 5, 1.5).unwrap_err();
        assert!(e.message.contains("between 0 and 180"), "{e}");
        assert!(decide_by_hits([2, 0, 1, 0], 5, 1.5).is_err());
    }

    #[test]
    fn rotation_turns_clockwise() {
        let mut img = RgbImage::new(3, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let r = rotate_cw(&img, 90);
        assert_eq!(r.dimensions(), (2, 3));
        assert_eq!(
            r.get_pixel(1, 0).0,
            [255, 0, 0],
            "top-left moves to top-right"
        );
    }
}
