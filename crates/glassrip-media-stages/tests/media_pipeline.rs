//! End-to-end media stages on generated videos (synthetic content only).
//!
//! Needs `ffmpeg` and `ffprobe` on PATH: the tests fail without them unless
//! `GLASSRIP_SKIP_FFMPEG_TESTS=1` is set.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use glassrip_core::envelope::{Record, SchemaReq};
use glassrip_core::jsonl;
use glassrip_core::manifest::StageStatus;
use glassrip_core::runner::StageReport;
use glassrip_media_stages::frames::SamplingMode;
use glassrip_media_stages::pipeline::{MEDIA_STAGES, MediaRunOptions, run_media_stages};
use glassrip_media_stages::schema::{
    BoundaryReason, FRAMES, FrameRecord, KEYFRAMES, Keyframe, MEDIA_PROBE, MediaProbe,
    RECTIFIED_KEYFRAMES, RectifiedKeyframe, RectifyMethod, SCREEN_QUADS, ScreenQuad,
};
use image::{Rgb, RgbImage};
use tokio_util::sync::CancellationToken;

/// True when ffmpeg and ffprobe are available. Missing tools fail the test unless
/// `GLASSRIP_SKIP_FFMPEG_TESTS=1` is set, so CI without ffmpeg cannot pass silently.
fn have_ffmpeg() -> bool {
    let ok = |t: &str| {
        Command::new(t)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    };
    if ok("ffmpeg") && ok("ffprobe") {
        return true;
    }
    if std::env::var("GLASSRIP_SKIP_FFMPEG_TESTS").as_deref() == Ok("1") {
        eprintln!("ffmpeg/ffprobe not found; skipped because GLASSRIP_SKIP_FFMPEG_TESTS=1");
        return false;
    }
    panic!("ffmpeg and ffprobe are required (set GLASSRIP_SKIP_FFMPEG_TESTS=1 to skip)");
}

/// A "phone photo of a monitor": dark surround, bright tilted screen with a slide pattern.
fn slide(seed: u32) -> RgbImage {
    let q = [
        [300.0, 170.0],
        [1620.0, 230.0],
        [1560.0, 910.0],
        [250.0, 870.0],
    ];
    let inside = |x: f64, y: f64| {
        (0..4).all(|i| {
            let (a, b): ([f64; 2], [f64; 2]) = (q[i], q[(i + 1) % 4]);
            (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0.0
        })
    };
    RgbImage::from_fn(1920, 1080, |x, y| {
        if !inside(f64::from(x), f64::from(y)) {
            return Rgb([25, 25, 30]);
        }
        // Distinct "diagram" per slide: boxes on a grid whose layout depends on the seed.
        let (cx, cy) = (x / 160, y / 120);
        let on = (cx * 7 + cy * 3 + seed * 5) % 4 == 0;
        let border = x % 160 < 6 || y % 120 < 6;
        if on && border {
            Rgb([20, 40, 160])
        } else if on {
            Rgb([200, 220, 250])
        } else {
            Rgb([245, 245, 245])
        }
    })
}

/// Three 6 s slides, 30 fps, sync frame every 0.5 s, with a tone track.
fn make_video(dir: &Path) -> PathBuf {
    let mut args: Vec<String> = vec![
        "-hide_banner".into(),
        "-v".into(),
        "error".into(),
        "-y".into(),
    ];
    for s in 0..3u32 {
        let p = dir.join(format!("slide{s}.png"));
        slide(s).save(&p).unwrap();
        args.extend(["-loop", "1", "-framerate", "30", "-t", "6", "-i"].map(String::from));
        args.push(p.display().to_string());
    }
    args.extend(
        [
            "-f",
            "lavfi",
            "-t",
            "18",
            "-i",
            "sine=frequency=440:sample_rate=16000",
        ]
        .map(String::from),
    );
    args.extend(
        [
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-g",
            "15",
            "-bf",
            "0",
            "-c:a",
            "aac",
        ]
        .map(String::from),
    );
    let out = dir.join("synthetic.mp4");
    args.push(out.display().to_string());
    let st = Command::new("ffmpeg").args(&args).status().unwrap();
    assert!(st.success(), "ffmpeg failed to build the synthetic video");
    out
}

fn items<T: serde::de::DeserializeOwned>(run: &Path, schema: &str) -> Vec<T> {
    jsonl::read::<Record<T>>(
        &run.join(format!("artifacts/{schema}.jsonl")),
        &SchemaReq::new(schema, 1),
    )
    .unwrap()
    .items
    .into_iter()
    .filter_map(|r| r.outcome.result)
    .collect()
}

fn options(video: &Path, ws: &Path, run: &str) -> MediaRunOptions {
    let mut o = MediaRunOptions::new(video.to_path_buf(), ws.join(run), ws);
    o.orient.override_rotation_deg = Some(0);
    o.allow_model_download = false;
    o.chunk_s = 4.0;
    o.chunk_concurrency = 2;
    o
}

async fn run(o: &MediaRunOptions) -> Vec<StageReport> {
    let mut reports = Vec::new();
    run_media_stages(o, CancellationToken::new(), &mut reports)
        .await
        .unwrap();
    reports
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slides_to_rectified_keyframes_then_all_cached() {
    if !have_ffmpeg() {
        return;
    }
    let ws = tempfile::tempdir().unwrap();
    let video = make_video(ws.path());

    let first = run(&options(&video, ws.path(), "run-a")).await;
    let names: Vec<&str> = first.iter().map(|r| r.stage.as_str()).collect();
    assert_eq!(names, MEDIA_STAGES);
    assert!(
        first
            .iter()
            .all(|r| r.status == StageStatus::Ok && r.items_error == 0),
        "{first:?}"
    );

    let ra = ws.path().join("run-a");
    let probe: Vec<MediaProbe> = items(&ra, MEDIA_PROBE);
    assert!(probe[0].audio_present && !probe[0].vfr);
    assert!(
        (probe[0].duration_s - 18.0).abs() < 0.2,
        "{}",
        probe[0].duration_s
    );

    let frames: Vec<FrameRecord> = items(&ra, FRAMES);
    assert_eq!(frames.len(), 9, "one frame per 2 s bucket");
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(f.frame_id, format!("f{k:06}"));
        assert!(
            f.pts_s >= 2.0 * k as f64 && f.pts_s < 2.0 * (k + 1) as f64,
            "{f:?}"
        );
        assert_eq!((f.width, f.height), (1920, 1080));
        assert!(ra.join(&f.path).is_file());
    }

    let quads: Vec<ScreenQuad> = items(&ra, SCREEN_QUADS);
    assert!(quads.iter().all(|q| q.quad.is_some()), "{quads:?}");

    let kfs: Vec<Keyframe> = items(&ra, KEYFRAMES);
    assert_eq!(kfs.len(), 3, "{kfs:#?}");
    assert_eq!(kfs[0].boundary.reason, BoundaryReason::Start);
    for k in &kfs[1..] {
        assert_ne!(k.boundary.reason, BoundaryReason::Start);
    }
    // Slides change at 6 s and 12 s; frames sit near bucket centers (odd seconds).
    assert!(
        (6.0..8.0).contains(&kfs[1].t_start_s) && (12.0..14.0).contains(&kfs[2].t_start_s),
        "{kfs:#?}"
    );
    assert!((kfs[2].t_end_s - probe[0].end_s).abs() < 1e-9);

    let rect: Vec<RectifiedKeyframe> = items(&ra, RECTIFIED_KEYFRAMES);
    assert_eq!(rect.len(), 3);
    for r in &rect {
        assert_eq!(r.method, RectifyMethod::WarpMedian, "{r:?}");
        assert!(r.n_frames_stacked >= 2);
        // The screen quad is about 1320 x 690 pixels.
        assert!(
            (r.width as i64 - 1320).abs() < 40 && (r.height as i64 - 690).abs() < 40,
            "{r:?}"
        );
        let img = image::open(ra.join(&r.path)).unwrap();
        assert_eq!((img.width(), img.height()), (r.width, r.height));
    }

    // A new run directory with the same cache: every stage restores, files reappear.
    let second = run(&options(&video, ws.path(), "run-b")).await;
    assert!(
        second.iter().all(|r| r.status == StageStatus::Cached),
        "{second:?}"
    );
    let rb = ws.path().join("run-b");
    for r in items::<RectifiedKeyframe>(&rb, RECTIFIED_KEYFRAMES) {
        assert_eq!(
            glassrip_media_stages::blobs::hash_file(&rb.join(&r.path)).unwrap(),
            r.blake3
        );
    }
    for f in items::<FrameRecord>(&rb, FRAMES) {
        assert!(rb.join(&f.path).is_file());
    }

    // Grid sampling decodes every frame and picks the same buckets.
    let mut g = options(&video, ws.path(), "run-grid");
    g.sampling = SamplingMode::Grid;
    g.selection.until = Some("frames".into());
    let rep = run(&g).await;
    let frames_rep = rep.iter().find(|r| r.stage == "frames").unwrap();
    assert_eq!(frames_rep.status, StageStatus::Ok);
    assert_eq!(rep.last().map(|r| r.status), Some(StageStatus::Skipped));
    let gf: Vec<FrameRecord> = items(&ws.path().join("run-grid"), FRAMES);
    let ids = |v: &[FrameRecord]| v.iter().map(|f| f.frame_id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&gf), ids(&frames));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rotated_video_is_turned_upright_by_override() {
    if !have_ffmpeg() {
        return;
    }
    let ws = tempfile::tempdir().unwrap();
    let dir = ws.path();
    // Portrait-coded 1080x1920 test pattern; a 90 degree clockwise correction makes it
    // 1920x1080.
    let video = dir.join("portrait.mp4");
    let st = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1080x1920:rate=10",
            "-t",
            "4",
            "-c:v",
            "mpeg4",
            "-g",
            "5",
        ])
        .arg(&video)
        .status()
        .unwrap();
    assert!(st.success());
    let mut o = options(&video, dir, "run-rot");
    o.orient.override_rotation_deg = Some(90);
    o.selection.until = Some("frames".into());
    run(&o).await;
    let frames: Vec<FrameRecord> = items(&dir.join("run-rot"), FRAMES);
    assert_eq!(frames.len(), 2);
    assert!(
        frames.iter().all(|f| (f.width, f.height) == (1920, 1080)),
        "{frames:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn production_accepts_non_16_9_frames() {
    if !have_ffmpeg() {
        return;
    }
    let ws = tempfile::tempdir().unwrap();
    let dir = ws.path();
    let video = dir.join("four_three.mp4");
    let st = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1440x1080:rate=10",
            "-t",
            "6",
            "-c:v",
            "mpeg4",
            "-g",
            "5",
        ])
        .arg(&video)
        .status()
        .unwrap();
    assert!(st.success());
    let mut o = options(&video, dir, "run-43");
    o.selection.until = Some("keyframes".into());
    let rep = run(&o).await;
    for r in &rep {
        assert_eq!(r.items_error, 0, "{r:?}");
    }
    let frames: Vec<FrameRecord> = items(&dir.join("run-43"), FRAMES);
    assert!(frames.iter().all(|f| (f.width, f.height) == (1920, 1440)));
    let feats: Vec<glassrip_media_stages::schema::FrameFeatures> =
        items(&dir.join("run-43"), glassrip_media_stages::schema::FEATURES);
    assert_eq!(feats.len(), frames.len());
}
