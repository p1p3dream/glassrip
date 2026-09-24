use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;

use crate::types::SampledFrame;

type F32Image = ImageBuffer<Luma<f32>, Vec<f32>>;

pub fn sample_frames(
    video_path: &Path,
    work_dir: &Path,
    ssim_threshold: f64,
) -> Result<Vec<Vec<SampledFrame>>> {
    let threshold = ssim_threshold.clamp(0.0, 1.0);
    let keyframes = extract_keyframes(video_path, work_dir)?;
    let unique = ssim_dedup(keyframes, threshold);
    Ok(detect_scroll_sequences(unique))
}

pub fn extract_keyframes(video_path: &Path, work_dir: &Path) -> Result<Vec<SampledFrame>> {
    let frames_dir = work_dir.join("keyframes");
    if frames_dir.exists() {
        std::fs::remove_dir_all(&frames_dir)?;
    }
    std::fs::create_dir_all(&frames_dir)?;

    let mut results = extract_via_select_filter(video_path, &frames_dir)?;

    if results.len() < 5 {
        let scene_results = extract_via_scene_change(video_path, &frames_dir)?;
        let seen: HashSet<i64> = results
            .iter()
            .map(|r| (r.timestamp * 10.0).round() as i64)
            .collect();
        for r in scene_results {
            if !seen.contains(&((r.timestamp * 10.0).round() as i64)) {
                results.push(r);
            }
        }
        results.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    Ok(results)
}

pub fn ssim_dedup(frames: Vec<SampledFrame>, threshold: f64) -> Vec<SampledFrame> {
    if frames.is_empty() {
        return vec![];
    }

    let first_gray = match load_gray(&frames[0].path, 320) {
        Some(g) => g,
        None => return frames,
    };

    let mut prev_gray = first_gray;
    let mut iter = frames.into_iter();
    let mut kept = vec![iter.next().unwrap()];

    for frame in iter {
        let curr_gray = match load_gray(&frame.path, 320) {
            Some(g) => g,
            None => continue,
        };

        if prev_gray.dimensions() != curr_gray.dimensions() {
            kept.push(frame);
            prev_gray = curr_gray;
            continue;
        }

        let score = compute_ssim(&prev_gray, &curr_gray);
        if score < threshold {
            kept.push(frame);
            prev_gray = curr_gray;
        }
    }

    kept
}

pub fn detect_scroll_sequences(frames: Vec<SampledFrame>) -> Vec<Vec<SampledFrame>> {
    if frames.is_empty() {
        return vec![];
    }

    let mut iter = frames.into_iter();
    let mut groups: Vec<Vec<SampledFrame>> = vec![vec![iter.next().unwrap()]];

    for frame in iter {
        let is_scroll = {
            let prev = groups.last().unwrap().last().unwrap();
            is_vertical_scroll(&prev.path, &frame.path)
        };
        if is_scroll {
            groups.last_mut().unwrap().push(frame);
        } else {
            groups.push(vec![frame]);
        }
    }

    groups
}

fn extract_via_select_filter(
    video_path: &Path,
    frames_dir: &Path,
) -> Result<Vec<SampledFrame>> {
    let pattern = frames_dir.join("kf_%06d.png");

    let ffmpeg_out = Command::new("ffmpeg")
        .args([
            "-v",
            "info",
            "-y",
            "-i",
            video_path.to_str().context("non-utf8 video path")?,
            "-vf",
            r"select=eq(pict_type\,I),showinfo",
            "-vsync",
            "vfr",
            "-frame_pts",
            "1",
            "-q:v",
            "2",
            pattern.to_str().context("non-utf8 output pattern")?,
        ])
        .output()
        .context("ffmpeg keyframe extraction failed")?;
    if !ffmpeg_out.status.success() {
        anyhow::bail!(
            "ffmpeg keyframe extraction exited with {}",
            ffmpeg_out.status
        );
    }

    let stderr = String::from_utf8_lossy(&ffmpeg_out.stderr);
    let mut keyframe_times: Vec<f64> = Vec::new();
    for line in stderr.lines() {
        if let Some(ts) = parse_pts_time(line) {
            keyframe_times.push(ts);
        }
    }

    let frame_files = collect_frame_files(frames_dir, "kf_");

    if frame_files.len() != keyframe_times.len() && !keyframe_times.is_empty() {
        eprintln!(
            "Warning: {} keyframe files but {} timestamps; timestamps may be inaccurate",
            frame_files.len(),
            keyframe_times.len()
        );
    }

    Ok(frame_files
        .into_iter()
        .enumerate()
        .map(|(i, path)| SampledFrame {
            timestamp: keyframe_times.get(i).copied().unwrap_or_else(|| {
                eprintln!("Warning: no timestamp for keyframe {i}, estimating");
                i as f64 * 2.0
            }),
            path,
            is_keyframe: true,
        })
        .collect())
}

fn extract_via_scene_change(
    video_path: &Path,
    frames_dir: &Path,
) -> Result<Vec<SampledFrame>> {
    let pattern = frames_dir.join("sc_%06d.png");

    let sc_out = Command::new("ffmpeg")
        .args([
            "-v",
            "info",
            "-y",
            "-i",
            video_path.to_str().context("non-utf8 video path")?,
            "-vf",
            "select='gt(scene,0.3)',showinfo",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            pattern.to_str().context("non-utf8 output pattern")?,
        ])
        .output()
        .context("ffmpeg scene change extraction failed")?;
    if !sc_out.status.success() {
        eprintln!(
            "Warning: ffmpeg scene change extraction exited with {}",
            sc_out.status
        );
    }

    let stderr = String::from_utf8_lossy(&sc_out.stderr);
    let mut timestamps: Vec<f64> = Vec::new();
    for line in stderr.lines() {
        if let Some(ts) = parse_pts_time(line) {
            timestamps.push(ts);
        }
    }

    let frame_files = collect_frame_files(frames_dir, "sc_");
    Ok(frame_files
        .into_iter()
        .enumerate()
        .map(|(i, path)| SampledFrame {
            timestamp: timestamps.get(i).copied().unwrap_or_else(|| {
                eprintln!("Warning: no timestamp for scene frame {i}, estimating");
                i as f64 * 5.0
            }),
            path,
            is_keyframe: false,
        })
        .collect())
}

fn parse_pts_time(line: &str) -> Option<f64> {
    let idx = line.find("pts_time:")?;
    let rest = &line[idx + "pts_time:".len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

fn collect_frame_files(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "png")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
        })
        .collect();
    files.sort();
    files
}

fn load_gray(path: &Path, target_width: u32) -> Option<GrayImage> {
    let img = image::open(path).ok()?.to_luma8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    let scale = target_width as f64 / w as f64;
    let new_h = (h as f64 * scale).round() as u32;
    Some(image::imageops::resize(
        &img,
        target_width,
        new_h.max(1),
        image::imageops::FilterType::Triangle,
    ))
}

fn to_f32(img: &GrayImage) -> F32Image {
    let (w, h) = img.dimensions();
    ImageBuffer::from_fn(w, h, |x, y| Luma([img.get_pixel(x, y)[0] as f32]))
}

fn mul_images(a: &F32Image, b: &F32Image) -> F32Image {
    let (w, h) = a.dimensions();
    ImageBuffer::from_fn(w, h, |x, y| {
        Luma([a.get_pixel(x, y)[0] * b.get_pixel(x, y)[0]])
    })
}

fn compute_ssim(a: &GrayImage, b: &GrayImage) -> f64 {
    let (w, h) = a.dimensions();
    if w == 0 || h == 0 {
        return 1.0;
    }

    let a_f = to_f32(a);
    let b_f = to_f32(b);
    let sigma = 1.5_f32;

    let mu_a = gaussian_blur_f32(&a_f, sigma);
    let mu_b = gaussian_blur_f32(&b_f, sigma);
    let blur_a_sq = gaussian_blur_f32(&mul_images(&a_f, &a_f), sigma);
    let blur_b_sq = gaussian_blur_f32(&mul_images(&b_f, &b_f), sigma);
    let blur_ab = gaussian_blur_f32(&mul_images(&a_f, &b_f), sigma);

    let c1 = 6.5025_f32;
    let c2 = 58.5225_f32;
    let pixel_count = (w * h) as f64;
    let mut sum = 0.0_f64;

    for y in 0..h {
        for x in 0..w {
            let ma = mu_a.get_pixel(x, y)[0];
            let mb = mu_b.get_pixel(x, y)[0];
            let sa_sq = blur_a_sq.get_pixel(x, y)[0] - ma * ma;
            let sb_sq = blur_b_sq.get_pixel(x, y)[0] - mb * mb;
            let sab = blur_ab.get_pixel(x, y)[0] - ma * mb;

            let num = (2.0 * ma * mb + c1) * (2.0 * sab + c2);
            let den = (ma * ma + mb * mb + c1) * (sa_sq + sb_sq + c2);
            sum += (num / den) as f64;
        }
    }

    sum / pixel_count
}

fn is_vertical_scroll(prev_path: &Path, curr_path: &Path) -> bool {
    let prev = match image::open(prev_path) {
        Ok(img) => img.to_luma8(),
        Err(_) => return false,
    };
    let curr = match image::open(curr_path) {
        Ok(img) => img.to_luma8(),
        Err(_) => return false,
    };

    let h = prev.height().min(curr.height());
    let w = prev.width().min(curr.width());
    if h < 40 || w < 40 {
        return false;
    }

    let scale = 320.0 / w as f64;
    let sw = 320u32;
    let sh = (h as f64 * scale).round().max(1.0) as u32;

    let prev_small =
        image::imageops::resize(&prev, sw, sh, image::imageops::FilterType::Triangle);
    let curr_small =
        image::imageops::resize(&curr, sw, sh, image::imageops::FilterType::Triangle);

    for pct in &[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let scroll_px = (sh as f64 * pct) as u32;
        let overlap = sh - scroll_px;
        if overlap < 10 {
            continue;
        }

        let prev_bottom =
            image::imageops::crop_imm(&prev_small, 0, scroll_px, sw, overlap).to_image();
        let curr_top = image::imageops::crop_imm(&curr_small, 0, 0, sw, overlap).to_image();

        if prev_bottom.dimensions() != curr_top.dimensions() {
            continue;
        }

        if compute_ssim(&prev_bottom, &curr_top) > 0.85 {
            return true;
        }
    }

    false
}
