use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::GrayImage;
use imageproc::edges::canny;

use crate::types::{CodeRegion, RegionType};

pub fn detect_code_region(frame_path: &Path) -> Result<Option<CodeRegion>> {
    let img = match image::open(frame_path) {
        Ok(img) => img,
        Err(_) => return Ok(None),
    };

    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return Ok(None);
    }

    let total = (w * h) as f64;
    let dark_count = gray.pixels().filter(|p| p[0] < 80).count() as f64;
    let light_count = gray.pixels().filter(|p| p[0] > 200).count() as f64;
    let dark_ratio = dark_count / total;
    let light_ratio = light_count / total;

    let region = if dark_ratio > 0.3 && dark_ratio > light_ratio {
        find_dark_code_region(&gray)
    } else if light_ratio > 0.3 {
        find_light_code_region(&gray)
    } else {
        CodeRegion {
            x: 0,
            y: 0,
            w,
            h,
            region_type: RegionType::Editor,
        }
    };

    Ok(Some(clamp_region(&region, w, h)))
}

pub fn crop_to_region(frame_path: &Path, region: &CodeRegion, out_path: &Path) -> Result<PathBuf> {
    let img = match image::open(frame_path) {
        Ok(img) => img,
        Err(_) => return Ok(frame_path.to_path_buf()),
    };

    let (w, h) = (img.width(), img.height());
    let clamped = clamp_region(region, w, h);

    if clamped.w == 0 || clamped.h == 0 {
        return Ok(frame_path.to_path_buf());
    }

    let cropped = img.crop_imm(clamped.x, clamped.y, clamped.w, clamped.h);
    image::DynamicImage::ImageRgb8(cropped.to_rgb8())
        .save(out_path)
        .context("failed to save cropped image")?;
    Ok(out_path.to_path_buf())
}

fn find_dark_code_region(gray: &GrayImage) -> CodeRegion {
    let (w, h) = gray.dimensions();

    let col_dark: Vec<f64> = (0..w)
        .map(|x| {
            let count = (0..h).filter(|&y| gray.get_pixel(x, y)[0] < 80).count();
            count as f64 / h as f64
        })
        .collect();

    let row_dark: Vec<f64> = (0..h)
        .map(|y| {
            let count = (0..w).filter(|&x| gray.get_pixel(x, y)[0] < 80).count();
            count as f64 / w as f64
        })
        .collect();

    let x_start = first_above(&col_dark, 0.5, 0);
    let x_end = last_above(&col_dark, 0.5, w.saturating_sub(1)) + 1;
    let y_start = first_above(&row_dark, 0.5, 0);
    let y_end = last_above(&row_dark, 0.5, h.saturating_sub(1)) + 1;

    let (x_start, y_start) = trim_ui_chrome(gray, x_start, y_start, x_end, y_end);

    let rw = x_end.saturating_sub(x_start).max(100);
    let rh = y_end.saturating_sub(y_start).max(100);

    let region_type = classify_region(
        gray,
        x_start,
        y_start,
        rw.min(w.saturating_sub(x_start)),
        rh.min(h.saturating_sub(y_start)),
    );

    CodeRegion {
        x: x_start,
        y: y_start,
        w: rw,
        h: rh,
        region_type,
    }
}

fn find_light_code_region(gray: &GrayImage) -> CodeRegion {
    let (w, h) = gray.dimensions();
    let edges = canny(gray, 50.0, 150.0);

    let row_edge_density: Vec<f64> = (0..h)
        .map(|y| {
            let sum: f64 = (0..w).map(|x| edges.get_pixel(x, y)[0] as f64).sum();
            sum / w as f64
        })
        .collect();

    let col_edge_density: Vec<f64> = (0..w)
        .map(|x| {
            let sum: f64 = (0..h).map(|y| edges.get_pixel(x, y)[0] as f64).sum();
            sum / h as f64
        })
        .collect();

    let y_start = first_above(&row_edge_density, 5.0, 0);
    let y_end = last_above(&row_edge_density, 5.0, h.saturating_sub(1)) + 1;
    let x_start = first_above(&col_edge_density, 5.0, 0);
    let x_end = last_above(&col_edge_density, 5.0, w.saturating_sub(1)) + 1;

    let pad = 10_u32;
    let x_start = x_start.saturating_sub(pad);
    let y_start = y_start.saturating_sub(pad);
    let x_end = (x_end + pad).min(w);
    let y_end = (y_end + pad).min(h);

    CodeRegion {
        x: x_start,
        y: y_start,
        w: x_end.saturating_sub(x_start),
        h: y_end.saturating_sub(y_start),
        region_type: RegionType::Editor,
    }
}

fn trim_ui_chrome(gray: &GrayImage, mut x0: u32, mut y0: u32, x1: u32, y1: u32) -> (u32, u32) {
    let (w, h) = gray.dimensions();
    let sidebar_px = 30_u32.max((w as f64 * 0.04) as u32);
    let tabbar_px = 20_u32.max((h as f64 * 0.04) as u32);

    if x1.saturating_sub(x0) > (w as f64 * 0.6) as u32 {
        let sx1 = (x0 + sidebar_px).min(w);
        let sy1 = y1.min(h);
        if sx1 > x0 && sy1 > y0 {
            if region_std_dev(gray, x0, y0, sx1, sy1) < 15.0 {
                x0 += sidebar_px;
            }
        }
    }

    if y1.saturating_sub(y0) > (h as f64 * 0.5) as u32 {
        let ty1 = (y0 + tabbar_px).min(h);
        let tx1 = x1.min(w);
        if ty1 > y0 && tx1 > x0 {
            if region_std_dev(gray, x0, y0, tx1, ty1) < 20.0 {
                y0 += tabbar_px;
            }
        }
    }

    (x0, y0)
}

fn classify_region(gray: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> RegionType {
    let (img_w, img_h) = gray.dimensions();
    let x_end = (x + w).min(img_w);
    let y_end = (y + h).min(img_h);

    let mut sum = 0.0_f64;
    let mut bright_count = 0_u64;
    let mut count = 0_u64;

    for py in y..y_end {
        for px in x..x_end {
            let v = gray.get_pixel(px, py)[0];
            sum += v as f64;
            if v > 180 {
                bright_count += 1;
            }
            count += 1;
        }
    }

    if count == 0 {
        return RegionType::Editor;
    }

    let mean_val = sum / count as f64;
    let bright_ratio = bright_count as f64 / count as f64;

    if mean_val < 40.0 && bright_ratio < 0.15 {
        RegionType::Terminal
    } else {
        RegionType::Editor
    }
}

fn region_std_dev(gray: &GrayImage, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let mut sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    let mut count = 0_u64;

    for y in y0..y1 {
        for x in x0..x1 {
            let v = gray.get_pixel(x, y)[0] as f64;
            sum += v;
            sq_sum += v * v;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f64;
    (sq_sum / count as f64 - mean * mean).max(0.0).sqrt()
}

fn clamp_region(region: &CodeRegion, frame_w: u32, frame_h: u32) -> CodeRegion {
    let x = region.x.min(frame_w.saturating_sub(1));
    let y = region.y.min(frame_h.saturating_sub(1));
    let w = region.w.min(frame_w - x).max(1);
    let h = region.h.min(frame_h - y).max(1);
    CodeRegion {
        x,
        y,
        w,
        h,
        region_type: region.region_type.clone(),
    }
}

fn first_above(arr: &[f64], threshold: f64, default: u32) -> u32 {
    arr.iter()
        .position(|&v| v > threshold)
        .map(|i| i as u32)
        .unwrap_or(default)
}

fn last_above(arr: &[f64], threshold: f64, default: u32) -> u32 {
    arr.iter()
        .rposition(|&v| v > threshold)
        .map(|i| i as u32)
        .unwrap_or(default)
}
