//! Development aid: writes every intermediate buffer for a frame pair as raw little-endian
//! files plus a JSON summary, so each primitive can be compared against a reference
//! implementation. Usage:
//! `cargo run -p glassrip-media --release --example dump_stages -- A.jpg B.jpg OUT_DIR`

use std::path::{Path, PathBuf};

use glassrip_media::decode::{decode_bgr, decode_gray};
use glassrip_media::features::{frame_features, pair_score, small_gray};
use glassrip_media::gaussian::blur_u8;
use glassrip_media::ink::{
    adaptive_threshold_mean_inv, bgr_to_gray, color_mask, ink_change, BLOCK, OFFSET,
};
use glassrip_media::plane::Plane;
use glassrip_media::resize::{area_bgr, area_gray, half_linear_gray};
use glassrip_media::sharpness::laplacian_variance;
use glassrip_media::warp::affine_linear_f32;

fn write_f32(path: &Path, p: &Plane<f32>) -> std::io::Result<()> {
    let bytes: Vec<u8> = p.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)
}

fn dump_frame(frame: &Path, out: &Path, tag: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let gray = decode_gray(frame)?;
    let bgr = decode_bgr(frame)?;
    std::fs::write(out.join(format!("{tag}_gray.u8")), &gray.data)?;
    std::fs::write(out.join(format!("{tag}_bgr.u8")), &bgr.data)?;
    let area = area_gray(&gray, 6);
    std::fs::write(out.join(format!("{tag}_small_area.u8")), &area.data)?;
    std::fs::write(
        out.join(format!("{tag}_small_blur.u8")),
        &blur_u8(&area, 5, 1.2).data,
    )?;
    write_f32(&out.join(format!("{tag}_small.f32")), &small_gray(&gray))?;
    let im = area_bgr(&bgr, 3);
    std::fs::write(out.join(format!("{tag}_bgr640.u8")), &im.data)?;
    let g640 = bgr_to_gray(&im);
    std::fs::write(out.join(format!("{tag}_gray640.u8")), &g640.data)?;
    let g = blur_u8(&g640, 3, 0.0);
    std::fs::write(out.join(format!("{tag}_ink_g.u8")), &g.data)?;
    std::fs::write(
        out.join(format!("{tag}_dark.u8")),
        &adaptive_threshold_mean_inv(&g, BLOCK, OFFSET).data,
    )?;
    std::fs::write(out.join(format!("{tag}_col.u8")), &color_mask(&im).data)?;
    std::fs::write(
        out.join(format!("{tag}_half.u8")),
        &half_linear_gray(&g).data,
    )?;
    let f = frame_features(frame)?;
    std::fs::write(out.join(format!("{tag}_ink_m.u8")), &f.ink.mask.data)?;
    write_f32(&out.join(format!("{tag}_align.f32")), &f.ink.align)?;
    Ok(laplacian_variance(&gray))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: dump_stages A.jpg B.jpg OUT_DIR".into());
    }
    let (fa, fb, out) = (
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
    );
    std::fs::create_dir_all(&out)?;
    let sharp_a = dump_frame(&fa, &out, "a")?;
    let sharp_b = dump_frame(&fb, &out, "b")?;
    let a = frame_features(&fa)?;
    let b = frame_features(&fb)?;
    let ps = pair_score(&a.small, &b.small);
    let bw = affine_linear_f32(&b.small, &ps.ecc.warp, 320, 180);
    write_f32(&out.join("bw.f32"), &bw)?;
    write_f32(
        &out.join("mu1.f32"),
        &glassrip_media::gaussian::blur_f32(&a.small, 7, 1.5),
    )?;
    write_f32(
        &out.join("ssim_map.f32"),
        &glassrip_media::ssim::ssim_map(&a.small, &bw),
    )?;
    let ink = ink_change(&a.ink, &b.ink);
    let summary = serde_json::json!({
        "sharp_a": sharp_a,
        "sharp_b": sharp_b,
        "ssim": ps.ssim,
        "frac": ps.changed_frac,
        "shift": ps.shift,
        "warp": ps.ecc.warp,
        "ecc_ok": ps.ecc.ok(),
        "ecc_iters": ps.ecc.iterations,
        "ink": ink.value,
        "ink_warp": ink.ecc.warp,
        "ink_ok": ink.ecc.ok(),
        "ink_iters": ink.ecc.iterations,
    });
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(())
}
