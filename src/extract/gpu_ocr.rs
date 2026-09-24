use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

const MODEL_DIR_ENV: &str = "GLASSRIP_MODEL_DIR";
const DEFAULT_MODEL_DIR: &str = ".glassrip/models";

const DET_MODEL: &str = "det.onnx";
const REC_MODEL: &str = "rec.onnx";
const DICT_FILE: &str = "dict.txt";

pub fn default_model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(MODEL_DIR_ENV) {
        return PathBuf::from(dir);
    }
    dirs_or_home().join(DEFAULT_MODEL_DIR)
}

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn is_available(model_dir: &Path) -> bool {
    model_dir.join(DET_MODEL).exists()
        && model_dir.join(REC_MODEL).exists()
        && model_dir.join(DICT_FILE).exists()
}

pub fn check_models(model_dir: &Path) -> Result<()> {
    if !is_available(model_dir) {
        bail!(
            "GPU OCR models not found in {}\n\
             Download PP-OCRv5 English ONNX models:\n\
             \n\
             mkdir -p {dir}\n\
             # From https://huggingface.co/monkt/paddleocr-onnx\n\
             # Download: detection/v5/det.onnx -> {dir}/det.onnx\n\
             # Download: languages/english/rec.onnx -> {dir}/rec.onnx\n\
             # Download: languages/english/dict.txt -> {dir}/dict.txt\n\
             \n\
             Or set GLASSRIP_MODEL_DIR to your model directory.",
            model_dir.display(),
            dir = model_dir.display(),
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
mod engine {
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use ndarray::Array4;
    use ort::session::Session;
    use ort::value::Tensor;

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    const DET_MAX_SIDE: u32 = 960;
    const REC_HEIGHT: u32 = 48;
    const DET_THRESHOLD: f32 = 0.3;

    fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
        anyhow::anyhow!("{e}")
    }

    /// Name of the ONNX Runtime execution provider this build uses.
    pub fn execution_provider() -> &'static str {
        if cfg!(feature = "gpu-cuda") {
            "CUDA"
        } else {
            "CPU"
        }
    }

    fn session_builder() -> Result<ort::session::builder::SessionBuilder> {
        let builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;

        // Without an explicit provider ONNX Runtime silently runs on CPU.
        // error_on_failure turns a missing or broken CUDA install into an
        // error instead of a quiet CPU fallback.
        #[cfg(feature = "gpu-cuda")]
        let builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(ort_err)
            .context("failed to register the CUDA execution provider")?;

        Ok(builder)
    }

    pub struct GpuOcrEngine {
        det_session: Session,
        rec_session: Session,
        dictionary: Vec<String>,
    }

    impl GpuOcrEngine {
        pub fn new(model_dir: &Path) -> Result<Self> {
            super::check_models(model_dir)?;

            let det_session = session_builder()?
                .commit_from_file(model_dir.join(super::DET_MODEL))
                .map_err(ort_err)
                .context("failed to load detection model")?;

            let rec_session = session_builder()?
                .commit_from_file(model_dir.join(super::REC_MODEL))
                .map_err(ort_err)
                .context("failed to load recognition model")?;

            let dict_text = std::fs::read_to_string(model_dir.join(super::DICT_FILE))
                .context("failed to read dictionary")?;
            let dictionary: Vec<String> = dict_text.lines().map(String::from).collect();

            eprintln!(
                "  Loaded detection model and recognition model ({} chars in dictionary) on {} execution provider",
                dictionary.len(),
                execution_provider()
            );

            Ok(Self {
                det_session,
                rec_session,
                dictionary,
            })
        }

        pub fn extract_code_from_frame(&mut self, frame_path: &Path) -> Result<String> {
            let img = image::open(frame_path)
                .with_context(|| format!("failed to open: {}", frame_path.display()))?;
            let rgb = img.to_rgb8();

            let line_boxes = self.detect_text_lines(&rgb)?;

            let mut lines = Vec::new();
            for (y1, y2) in &line_boxes {
                let y1 = (*y1).min(rgb.height().saturating_sub(1));
                let y2 = (*y2).min(rgb.height());
                if y2 <= y1 || y2 - y1 < 3 {
                    continue;
                }
                let crop =
                    image::imageops::crop_imm(&rgb, 0, y1, rgb.width(), y2 - y1).to_image();
                if let Ok(text) = self.recognize_line(&crop) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        lines.push(trimmed.to_string());
                    }
                }
            }

            Ok(lines.join("\n"))
        }

        /// One result per frame, in order; a failed frame does not stop the batch.
        pub fn extract_batch(&mut self, frame_paths: &[PathBuf]) -> Vec<Result<String>> {
            frame_paths
                .iter()
                .map(|p| self.extract_code_from_frame(p))
                .collect()
        }

        fn detect_text_lines(
            &mut self,
            img: &image::RgbImage,
        ) -> Result<Vec<(u32, u32)>> {
            let (orig_w, orig_h) = (img.width(), img.height());

            let scale = DET_MAX_SIDE as f32 / orig_w.max(orig_h) as f32;
            let scale = scale.min(1.0);
            let new_w = ((orig_w as f32 * scale) as u32).max(32);
            let new_h = ((orig_h as f32 * scale) as u32).max(32);
            let new_w = new_w.div_ceil(32) * 32;
            let new_h = new_h.div_ceil(32) * 32;

            let resized = image::imageops::resize(
                img, new_w, new_h, image::imageops::FilterType::Triangle,
            );
            let tensor = image_to_tensor(&resized, new_w, new_h);
            let input = Tensor::from_array(tensor).map_err(ort_err)?;
            let outputs = self.det_session
                .run(ort::inputs![input])
                .map_err(ort_err)?;

            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
            if shape.len() != 4 {
                anyhow::bail!("unexpected detection model output rank: {}, expected 4", shape.len());
            }
            let out_h = shape[2] as usize;
            let out_w = shape[3] as usize;
            let scale_y = orig_h as f32 / new_h as f32;

            // Compute per-row text density: fraction of columns above threshold
            let mut row_density: Vec<f32> = Vec::with_capacity(out_h);
            for y in 0..out_h {
                let mut count = 0u32;
                for x in 0..out_w {
                    if data[y * out_w + x] > DET_THRESHOLD {
                        count += 1;
                    }
                }
                row_density.push(count as f32 / out_w as f32);
            }

            // Find text lines: contiguous runs where density > min threshold
            // Split runs at local density minima to separate merged lines
            let min_density = 0.02;
            let mut lines: Vec<(u32, u32)> = Vec::new();
            let mut in_text = false;
            let mut line_start = 0usize;

            for (y, &density) in row_density.iter().enumerate().take(out_h) {
                if density > min_density {
                    if !in_text {
                        line_start = y;
                        in_text = true;
                    }
                } else if in_text {
                    let y1 = (line_start as f32 * scale_y) as u32;
                    let y2 = (y as f32 * scale_y) as u32;
                    split_merged_lines(y1, y2, orig_h, &mut lines);
                    in_text = false;
                }
            }
            if in_text {
                let y1 = (line_start as f32 * scale_y) as u32;
                let y2 = (out_h as f32 * scale_y) as u32;
                split_merged_lines(y1, y2, orig_h, &mut lines);
            }

            // If detection found nothing, estimate line height and slice uniformly
            if lines.is_empty() {
                let est_line_h = estimate_line_height(orig_h);
                let mut y = 0u32;
                while y + est_line_h <= orig_h {
                    lines.push((y, y + est_line_h));
                    y += est_line_h;
                }
                if y < orig_h && orig_h - y > est_line_h / 2 {
                    lines.push((y, orig_h));
                }
            }

            Ok(lines)
        }

    }

    fn split_merged_lines(y1: u32, y2: u32, img_h: u32, out: &mut Vec<(u32, u32)>) {
        let height = y2 - y1;
        let est_line = estimate_line_height(img_h);
        let max_single = (est_line as f32 * 1.8) as u32;

        if height <= max_single {
            out.push((y1, y2));
        } else {
            let n_lines = ((height as f32 / est_line as f32).round() as u32).max(2);
            let step = height / n_lines;
            for i in 0..n_lines {
                let ly1 = y1 + i * step;
                let ly2 = if i == n_lines - 1 { y2 } else { ly1 + step };
                out.push((ly1, ly2));
            }
        }
    }

    impl GpuOcrEngine {

        fn recognize_line(&mut self, crop: &image::RgbImage) -> Result<String> {
            let (w, h) = (crop.width(), crop.height());
            if w == 0 || h == 0 {
                return Ok(String::new());
            }

            let new_h = REC_HEIGHT;
            let new_w = ((w as f32 / h as f32) * new_h as f32).max(1.0) as u32;
            let padded_w = new_w.max(320);

            let resized = image::imageops::resize(
                crop, new_w, new_h, image::imageops::FilterType::Triangle,
            );

            let tensor = rec_image_to_tensor(&resized, new_w, new_h, padded_w);
            let input = Tensor::from_array(tensor).map_err(ort_err)?;
            let outputs = self
                .rec_session
                .run(ort::inputs![input])
                .map_err(ort_err)?;

            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
            if shape.len() != 3 {
                anyhow::bail!("unexpected recognition model output rank: {}, expected 3", shape.len());
            }

            // CTC decode: [1, seq_len, num_classes]
            let seq_len = shape[1] as usize;
            let num_classes = shape[2] as usize;

            // First pass: find the best class at each timestep
            let mut best_indices: Vec<usize> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for c in 0..num_classes {
                    let val = data[t * num_classes + c];
                    if val > best_val {
                        best_val = val;
                        best_idx = c;
                    }
                }
                best_indices.push(best_idx);
            }

            // Compute blank run lengths between non-blank chars to find adaptive threshold
            let mut blank_runs: Vec<u32> = Vec::new();
            let mut current_run = 0u32;
            let mut seen_char = false;
            for &idx in &best_indices {
                if idx == 0 {
                    current_run += 1;
                } else {
                    if seen_char && current_run > 0 {
                        blank_runs.push(current_run);
                    }
                    current_run = 0;
                    seen_char = true;
                }
            }

            // Space threshold: use median blank run * 2.5
            // Inter-character gaps are small; word gaps are much larger
            let space_threshold = if blank_runs.len() > 2 {
                let mut sorted = blank_runs.clone();
                sorted.sort();
                let median = sorted[sorted.len() / 2];
                ((median as f32 * 2.5) as u32).max(4)
            } else {
                8
            };

            // Second pass: decode with space insertion
            let mut text = String::new();
            let mut prev_idx = 0usize;
            let mut blank_run = 0u32;
            let mut has_content = false;

            for &best_idx in &best_indices {
                if best_idx == 0 {
                    blank_run += 1;
                } else {
                    if blank_run >= space_threshold && has_content {
                        text.push(' ');
                    }
                    blank_run = 0;

                    if best_idx != prev_idx {
                        if let Some(ch) = self.dictionary.get(best_idx - 1) {
                            text.push_str(ch);
                            has_content = true;
                        }
                    }
                }
                prev_idx = best_idx;
            }

            Ok(text)
        }
    }

    fn estimate_line_height(img_height: u32) -> u32 {
        // For typical code editors at 1080p: ~25px per line
        // Scale proportionally for other resolutions
        ((img_height as f32 / 1080.0) * 25.0).max(12.0) as u32
    }

    // Recognition: (x/255 - 0.5) / 0.5 = x/127.5 - 1.0, zero-padded to padded_w
    fn rec_image_to_tensor(img: &image::RgbImage, w: u32, h: u32, padded_w: u32) -> Array4<f32> {
        let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, padded_w as usize));
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pixel = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = pixel[c] as f32 / 127.5 - 1.0;
                }
            }
        }
        tensor
    }

    // Detection: ImageNet normalization
    fn image_to_tensor(img: &image::RgbImage, w: u32, h: u32) -> Array4<f32> {
        let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        for y in 0..h as usize {
            for x in 0..w as usize {
                let pixel = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                }
            }
        }
        tensor
    }
}

#[cfg(feature = "gpu")]
pub use engine::{execution_provider, GpuOcrEngine};

#[cfg(all(test, feature = "gpu"))]
mod tests {
    #[test]
    fn execution_provider_matches_features() {
        let expected = if cfg!(feature = "gpu-cuda") { "CUDA" } else { "CPU" };
        assert_eq!(super::execution_provider(), expected);
    }

    #[test]
    fn missing_models_error_before_session_build() {
        let dir = tempfile::tempdir().unwrap();
        let err = super::GpuOcrEngine::new(dir.path()).err().unwrap();
        assert!(err.to_string().contains("GPU OCR models not found"), "{err}");
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuOcrEngine;

#[cfg(not(feature = "gpu"))]
impl GpuOcrEngine {
    pub fn new(_model_dir: &Path) -> Result<Self> {
        bail!("GPU OCR requires the 'gpu' feature. Rebuild with: cargo build --features gpu");
    }

    pub fn extract_code_from_frame(&mut self, _frame_path: &Path) -> Result<String> {
        bail!("GPU OCR not available");
    }

    pub fn extract_batch(&mut self, frame_paths: &[PathBuf]) -> Vec<Result<String>> {
        frame_paths
            .iter()
            .map(|_| Err(anyhow::anyhow!("GPU OCR not available")))
            .collect()
    }
}
