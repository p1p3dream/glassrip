//! PP-OCRv5 tile-name reader on ONNX Runtime (feature `ocr`).
//!
//! Ported from the repository's `src/extract/gpu_ocr.rs` engine (same model
//! files, input normalization and CTC decoding). Tile names need 2D boxes rather
//! than full-width line bands, so detection here extracts connected components
//! from the DB probability map and expands them by the usual unclip distance.
//! ONNX Runtime is loaded at run time from `ORT_DYLIB_PATH` (see the
//! glassrip-audio README for the CUDA 12 setup); with `ocr-cuda` the CUDA
//! execution provider is registered and a missing CUDA install is an error.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use image::imageops::FilterType;
use image::RgbImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

use super::cues::{ScreenText, TileReader};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const REC_HEIGHT: u32 = 48;

/// Detection and recognition settings.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrParams {
    /// Longest side fed to the detector.
    pub det_max_side: u32,
    /// Probability threshold of the text map.
    pub det_threshold: f32,
    /// Minimum mean probability of a component.
    pub box_threshold: f32,
    /// Unclip ratio (DB post-processing).
    pub unclip_ratio: f32,
    /// Boxes shorter than this (frame pixels) are skipped.
    pub min_text_height: u32,
    /// Boxes taller than this (frame pixels) are skipped.
    pub max_text_height: u32,
    /// Crops per recognition batch.
    pub rec_batch: usize,
    /// CUDA arena limit per session, MiB (`ocr-cuda` only).
    pub cuda_mem_limit_mib: usize,
}

impl Default for OcrParams {
    fn default() -> Self {
        Self {
            det_max_side: 1920,
            det_threshold: 0.3,
            box_threshold: 0.5,
            unclip_ratio: 1.5,
            min_text_height: 10,
            max_text_height: 80,
            rec_batch: 16,
            // the detector needs about 2 GiB at 1920-wide frames; 1.5 GiB failed
            // every frame on the reference meeting
            cuda_mem_limit_mib: 3072,
        }
    }
}

/// PP-OCRv5 detector plus recognizer.
pub struct PpOcrReader {
    det: Mutex<Session>,
    rec: Mutex<Session>,
    dict: Vec<String>,
    params: OcrParams,
    provider: &'static str,
    model_dir: PathBuf,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn session(path: &Path, _cuda_mem_limit_mib: usize) -> Result<Session, String> {
    let builder = Session::builder()
        .map_err(err)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(err)?;
    #[cfg(feature = "ocr-cuda")]
    let builder = builder
        .with_execution_providers([ort::ep::CUDA::default()
            // bounded arena: the GPU is shared with the model servers
            .with_memory_limit(_cuda_mem_limit_mib << 20)
            .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
            .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
            .build()
            .error_on_failure()])
        .map_err(|e| format!("CUDA execution provider: {e}"))?;
    let mut builder = builder;
    builder
        .commit_from_file(path)
        .map_err(|e| format!("loading {}: {e}", path.display()))
}

impl PpOcrReader {
    /// Loads `det.onnx`, `rec.onnx` and `dict.txt` from `model_dir`.
    pub fn new(model_dir: &Path, params: OcrParams) -> Result<Self, String> {
        let dict_text = std::fs::read_to_string(model_dir.join("dict.txt"))
            .map_err(|e| format!("reading dict.txt: {e}"))?;
        Ok(Self {
            det: Mutex::new(session(
                &model_dir.join("det.onnx"),
                params.cuda_mem_limit_mib,
            )?),
            rec: Mutex::new(session(
                &model_dir.join("rec.onnx"),
                params.cuda_mem_limit_mib,
            )?),
            dict: dict_text.lines().map(String::from).collect(),
            params,
            provider: if cfg!(feature = "ocr-cuda") {
                "CUDA"
            } else {
                "CPU"
            },
            model_dir: model_dir.to_path_buf(),
        })
    }

    fn detect(&self, img: &RgbImage) -> Result<Vec<[u32; 4]>, String> {
        let (ow, oh) = img.dimensions();
        let scale = (self.params.det_max_side as f32 / ow.max(oh) as f32).min(1.0);
        let nw = (((ow as f32 * scale) as u32).max(32)).div_ceil(32) * 32;
        let nh = (((oh as f32 * scale) as u32).max(32)).div_ceil(32) * 32;
        let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);
        let mut t = Array4::<f32>::zeros((1, 3, nh as usize, nw as usize));
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                t[[0, c, y as usize, x as usize]] = (f32::from(p[c]) / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let input = Tensor::from_array(t).map_err(err)?;
        let mut det = self
            .det
            .lock()
            .map_err(|_| "detector lock poisoned".to_string())?;
        let outputs = det.run(ort::inputs![input]).map_err(err)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(err)?;
        if shape.len() != 4 {
            return Err(format!("detector output rank {}", shape.len()));
        }
        let (mh, mw) = (shape[2] as usize, shape[3] as usize);
        let (sx, sy) = (ow as f32 / mw as f32, oh as f32 / mh as f32);
        let on = |i: usize| data.get(i).is_some_and(|v| *v > self.params.det_threshold);
        let mut seen = vec![false; mh * mw];
        let mut boxes = Vec::new();
        let mut stack = Vec::new();
        for start in 0..mh * mw {
            if seen[start] || !on(start) {
                continue;
            }
            seen[start] = true;
            stack.push(start);
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
            let (mut n, mut psum) = (0usize, 0f32);
            while let Some(i) = stack.pop() {
                let (x, y) = (i % mw, i / mw);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                n += 1;
                psum += data.get(i).copied().unwrap_or(0.0);
                for (dx, dy) in [
                    (-1i64, 0i64),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                ] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= mw as i64 || ny >= mh as i64 {
                        continue;
                    }
                    let j = ny as usize * mw + nx as usize;
                    if !seen[j] && on(j) {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if n < 6 || psum / (n as f32) < self.params.box_threshold {
                continue;
            }
            let (w, h) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
            let d = w * h * self.params.unclip_ratio / (2.0 * (w + h));
            let fx0 = ((x0 as f32 - d) * sx).max(0.0);
            let fy0 = ((y0 as f32 - d) * sy).max(0.0);
            let fx1 = (((x1 + 1) as f32 + d) * sx).min(ow as f32);
            let fy1 = (((y1 + 1) as f32 + d) * sy).min(oh as f32);
            let bh = (fy1 - fy0) as u32;
            if bh < self.params.min_text_height
                || bh > self.params.max_text_height
                || fx1 - fx0 < fy1 - fy0
            {
                continue;
            }
            boxes.push([fx0 as u32, fy0 as u32, fx1 as u32, fy1 as u32]);
        }
        Ok(boxes)
    }

    fn recognize(&self, img: &RgbImage, boxes: &[[u32; 4]]) -> Result<Vec<(String, f32)>, String> {
        let mut crops: Vec<(usize, RgbImage)> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let crop =
                    image::imageops::crop_imm(img, b[0], b[1], b[2] - b[0], b[3] - b[1]).to_image();
                let nw = ((crop.width() as f32 / crop.height().max(1) as f32) * REC_HEIGHT as f32)
                    .clamp(8.0, 1600.0) as u32;
                (
                    i,
                    image::imageops::resize(&crop, nw, REC_HEIGHT, FilterType::Triangle),
                )
            })
            .collect();
        crops.sort_by_key(|(_, c)| c.width());
        let mut out = vec![(String::new(), 0.0f32); boxes.len()];
        let mut rec = self
            .rec
            .lock()
            .map_err(|_| "recognizer lock poisoned".to_string())?;
        for batch in crops.chunks(self.params.rec_batch.max(1)) {
            let pw = batch
                .iter()
                .map(|(_, c)| c.width())
                .max()
                .unwrap_or(8)
                .max(320)
                // few distinct widths keep the CUDA arena from fragmenting
                .div_ceil(160)
                * 160;
            let mut t = Array4::<f32>::zeros((batch.len(), 3, REC_HEIGHT as usize, pw as usize));
            for (bi, (_, c)) in batch.iter().enumerate() {
                for (x, y, p) in c.enumerate_pixels() {
                    for ch in 0..3 {
                        t[[bi, ch, y as usize, x as usize]] = f32::from(p[ch]) / 127.5 - 1.0;
                    }
                }
            }
            let input = Tensor::from_array(t).map_err(err)?;
            let outputs = rec.run(ort::inputs![input]).map_err(err)?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(err)?;
            if shape.len() != 3 {
                return Err(format!("recognizer output rank {}", shape.len()));
            }
            let (steps, classes) = (shape[1] as usize, shape[2] as usize);
            for (bi, (orig, _)) in batch.iter().enumerate() {
                let base = bi * steps * classes;
                let (mut text, mut probs, mut prev) = (String::new(), Vec::new(), 0usize);
                for s in 0..steps {
                    let row = data
                        .get(base + s * classes..base + (s + 1) * classes)
                        .unwrap_or(&[]);
                    let (best, val) =
                        row.iter()
                            .enumerate()
                            .fold((0usize, f32::NEG_INFINITY), |acc, (i, v)| {
                                if *v > acc.1 {
                                    (i, *v)
                                } else {
                                    acc
                                }
                            });
                    if best != 0 && best != prev {
                        match self.dict.get(best - 1) {
                            Some(ch) => text.push_str(ch),
                            // the class after the dictionary is the space
                            None => text.push(' '),
                        }
                        probs.push(val);
                    }
                    prev = best;
                }
                let conf = if probs.is_empty() {
                    0.0
                } else {
                    probs.iter().sum::<f32>() / probs.len() as f32
                };
                if let Some(slot) = out.get_mut(*orig) {
                    *slot = (text.trim().to_string(), conf);
                }
            }
        }
        Ok(out)
    }
}

impl TileReader for PpOcrReader {
    fn read(&self, img: &RgbImage) -> Result<Vec<ScreenText>, String> {
        let boxes = self.detect(img)?;
        let texts = self.recognize(img, &boxes)?;
        Ok(boxes
            .into_iter()
            .zip(texts)
            .filter(|(_, (t, _))| !t.is_empty())
            .map(|(bbox, (text, confidence))| ScreenText {
                text,
                bbox,
                confidence,
            })
            .collect())
    }

    fn describe(&self) -> String {
        format!(
            "pp-ocrv5 ({}, det_max_side {}, models {})",
            self.provider,
            self.params.det_max_side,
            self.model_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
        )
    }
}
