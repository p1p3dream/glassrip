//! Orientation end to end on a synthetic rotated text page.
//!
//! Skipped unless `GLASSRIP_TEST_MODELS_DIR` points at a models directory holding the pinned
//! orientation models (ONNX builds: `orient/*`; `ocrs` builds: `ocrs/*`). Needs ffmpeg.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;

use glassrip_core::envelope::{Record, SchemaReq};
use glassrip_core::jsonl;
use glassrip_media_stages::pipeline::{MediaRunOptions, run_media_stages};
use glassrip_media_stages::schema::{ORIENTATION, Orientation};
use image::{Rgb, RgbImage};
use tokio_util::sync::CancellationToken;

/// 5x7 bitmap glyphs for `a` to `z`, one row per string, `#` = ink.
const GLYPHS: [[&str; 7]; 26] = [
    [
        ".....", ".....", ".###.", "....#", ".####", "#...#", ".####",
    ], // a
    [
        "#....", "#....", "####.", "#...#", "#...#", "#...#", "####.",
    ], // b
    [
        ".....", ".....", ".###.", "#....", "#....", "#...#", ".###.",
    ], // c
    [
        "....#", "....#", ".####", "#...#", "#...#", "#...#", ".####",
    ], // d
    [
        ".....", ".....", ".###.", "#...#", "#####", "#....", ".###.",
    ], // e
    [
        "..##.", ".#..#", ".#...", "###..", ".#...", ".#...", ".#...",
    ], // f
    [
        ".....", ".####", "#...#", "#...#", ".####", "....#", ".###.",
    ], // g
    [
        "#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#",
    ], // h
    [
        "..#..", ".....", ".##..", "..#..", "..#..", "..#..", ".###.",
    ], // i
    [
        "...#.", ".....", "..##.", "...#.", "...#.", "#..#.", ".##..",
    ], // j
    [
        "#....", "#....", "#..#.", "#.#..", "##...", "#.#..", "#..#.",
    ], // k
    [
        ".##..", "..#..", "..#..", "..#..", "..#..", "..#..", ".###.",
    ], // l
    [
        ".....", ".....", "##.#.", "#.#.#", "#.#.#", "#.#.#", "#.#.#",
    ], // m
    [
        ".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#",
    ], // n
    [
        ".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.",
    ], // o
    [
        ".....", "####.", "#...#", "#...#", "####.", "#....", "#....",
    ], // p
    [
        ".....", ".####", "#...#", "#...#", ".####", "....#", "....#",
    ], // q
    [
        ".....", ".....", "#.##.", "##..#", "#....", "#....", "#....",
    ], // r
    [
        ".....", ".....", ".####", "#....", ".###.", "....#", "####.",
    ], // s
    [
        ".#...", ".#...", "###..", ".#...", ".#...", ".#..#", "..##.",
    ], // t
    [
        ".....", ".....", "#...#", "#...#", "#...#", "#..##", ".##.#",
    ], // u
    [
        ".....", ".....", "#...#", "#...#", "#...#", ".#.#.", "..#..",
    ], // v
    [
        ".....", ".....", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#.",
    ], // w
    [
        ".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#",
    ], // x
    [
        ".....", "#...#", "#...#", "#...#", ".####", "....#", ".###.",
    ], // y
    [
        ".....", ".....", "#####", "...#.", "..#..", ".#...", "#####",
    ], // z
];

const TEXT: &str = "the design review for the new home page will start after the team \
    has read the project notes and the list of open items for this week we want to \
    share the draft with every member of the board and update the status of each task \
    before the next meeting so that people know what work is done and what is still \
    in progress please add your comments to the document and let the group know when \
    the first version of the mobile app is ready for a test";

/// A page of text lines drawn with the bitmap font at `scale`.
fn page(w: u32, h: u32, scale: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([250, 250, 250]));
    let (gw, gh) = (6 * scale, 10 * scale);
    let margin = 3 * gw;
    let (mut x, mut y) = (margin, 2 * gh);
    let words: Vec<&str> = TEXT.split_whitespace().cycle().take(400).collect();
    for word in words {
        let ww = word.len() as u32 * gw;
        if x + ww > w - margin {
            x = margin;
            y += gh;
        }
        if y + gh > h - gh {
            break;
        }
        for ch in word.bytes() {
            let g = &GLYPHS[usize::from(ch - b'a')];
            for (ry, row) in g.iter().enumerate() {
                for (rx, c) in row.bytes().enumerate() {
                    if c == b'#' {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                img.put_pixel(
                                    x + rx as u32 * scale + dx,
                                    y + ry as u32 * scale + dy,
                                    Rgb([20, 20, 20]),
                                );
                            }
                        }
                    }
                }
            }
            x += gw;
        }
        x += gw;
    }
    img
}

fn video_from(img: &RgbImage, dir: &Path) -> std::path::PathBuf {
    let png = dir.join("page.png");
    img.save(&png).unwrap();
    let out = dir.join("page.mp4");
    let st = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "10",
            "-t",
            "6",
            "-i",
        ])
        .arg(&png)
        .args([
            "-c:v", "mpeg4", "-q:v", "2", "-g", "5", "-pix_fmt", "yuv420p",
        ])
        .arg(&out)
        .status()
        .unwrap();
    assert!(st.success());
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rotated_text_page_is_detected() {
    let Some(models) = std::env::var_os("GLASSRIP_TEST_MODELS_DIR") else {
        eprintln!("GLASSRIP_TEST_MODELS_DIR not set; skipping orientation end-to-end test");
        return;
    };
    let ws = tempfile::tempdir().unwrap();
    // Content turned 90 degrees clockwise needs a 270 degree clockwise correction.
    for (turn, expect) in [(0u32, 0u32), (90, 270), (180, 180)] {
        let upright = page(1920, 1080, 4);
        let turned = match turn {
            90 => image::imageops::rotate90(&upright),
            180 => image::imageops::rotate180(&upright),
            _ => upright,
        };
        let dir = ws.path().join(format!("turn{turn}"));
        std::fs::create_dir_all(&dir).unwrap();
        let video = video_from(&turned, &dir);
        let mut o = MediaRunOptions::new(video, dir.join("run"), ws.path());
        o.models_dir = models.clone().into();
        o.allow_model_download = false;
        o.orient.sample_frames = 4;
        o.orient.min_votes = 3;
        o.orient.fallback_frames = 1;
        o.config.orient.sample_frames = 4;
        o.selection.until = Some("orient".into());
        let mut reports = Vec::new();
        run_media_stages(&o, CancellationToken::new(), &mut reports)
            .await
            .unwrap();
        let path = dir
            .join("run/artifacts")
            .join(format!("{ORIENTATION}.jsonl"));
        let r = jsonl::read::<Record<Orientation>>(&path, &SchemaReq::new(ORIENTATION, 1))
            .unwrap()
            .items
            .remove(0)
            .outcome
            .result
            .unwrap();
        eprintln!(
            "turn {turn}: correction {} via {:?}, votes {:?}, confirmation {:?}",
            r.applied_rotation_deg, r.method, r.votes, r.confirmation
        );
        assert_eq!(r.applied_rotation_deg, expect, "{r:?}");
    }
}
