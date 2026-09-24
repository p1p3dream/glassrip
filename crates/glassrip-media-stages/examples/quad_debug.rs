//! Runs screen-quad detection on JPEG/PNG files and writes copies with the quad drawn.
//!
//! ```text
//! cargo run --release -p glassrip-media-stages --example quad_debug -- OUT_DIR IMAGE...
//! ```

use anyhow::{Context, Result};
use glassrip_media_stages::quad::{QuadParams, detect_quad};
use imageproc::drawing::draw_line_segment_mut;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().context("usage: quad_debug OUT_DIR IMAGE...")?);
    std::fs::create_dir_all(&out)?;
    for path in args {
        let img = image::open(&path)?;
        let d = detect_quad(&img.to_luma8(), &QuadParams::default());
        println!(
            "{path}: quad {:?} confidence {:.2} area {:?}",
            d.quad, d.confidence, d.area_frac
        );
        let mut rgb = img.to_rgb8();
        if let Some(q) = d.quad {
            for i in 0..4 {
                let (a, b) = (q[i], q[(i + 1) % 4]);
                for o in -2..=2 {
                    let o = o as f32;
                    draw_line_segment_mut(
                        &mut rgb,
                        (a[0] as f32 + o, a[1] as f32 + o),
                        (b[0] as f32 + o, b[1] as f32 + o),
                        image::Rgb([255, 0, 0]),
                    );
                }
            }
        }
        let name = std::path::Path::new(&path)
            .file_name()
            .context("image path has no file name")?;
        rgb.save(out.join(name))?;
    }
    Ok(())
}
