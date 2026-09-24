use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Semaphore;

use crate::extract::{gpu_ocr, ocr, vlm};
use crate::frames::{regions, sampler};
use crate::output::json_out::write_output;
use crate::stitch::{self, scroll};
use crate::types::*;

#[derive(Debug, Clone)]
pub struct ScrapeArgs {
    pub video: PathBuf,
    pub output: PathBuf,
    pub model: String,
    pub ollama_host: String,
    pub ssim_threshold: f64,
    pub ocr: bool,
    pub gpu_ocr: bool,
    pub model_dir: PathBuf,
    pub work_dir: Option<PathBuf>,
    pub parallel: usize,
    pub refine: bool,
    pub refine_model: String,
    pub refine_agents: usize,
    /// Per-request VLM timeout in seconds.
    pub vlm_timeout_secs: u64,
    /// Fraction of frames (0.0 to 1.0) allowed to fail before the run fails.
    pub max_frame_failure_rate: f64,
}

/// Default for `ScrapeArgs::max_frame_failure_rate`.
pub const DEFAULT_MAX_FRAME_FAILURE_RATE: f64 = 0.10;

impl ScrapeArgs {
    pub fn vlm_options(&self) -> vlm::VlmOptions {
        vlm::VlmOptions {
            timeout: std::time::Duration::from_secs(self.vlm_timeout_secs),
            ..vlm::VlmOptions::default()
        }
    }
}

pub async fn run_pipeline(args: &ScrapeArgs) -> Result<()> {
    let temp_dir;
    let work_dir = match &args.work_dir {
        Some(dir) => dir.as_path(),
        None => {
            temp_dir = tempfile::Builder::new()
                .prefix("glassrip_")
                .tempdir()?;
            temp_dir.path()
        }
    };

    println!("Work directory: {}", work_dir.display());
    let video_path = args.video.canonicalize()?;
    let duration = get_duration(&video_path);

    println!("Sampling frames...");
    let frame_groups = sampler::sample_frames(&video_path, work_dir, args.ssim_threshold)?;
    let total_frames: usize = frame_groups.iter().map(|g| g.len()).sum();
    println!(
        "Found {total_frames} unique frames in {} groups",
        frame_groups.len()
    );

    if frame_groups.is_empty() {
        anyhow::bail!("No code frames found in video");
    }

    println!("Detecting code regions...");
    let layout = if let Some(first) = frame_groups[0].first() {
        regions::detect_code_region(&first.path)?
    } else {
        None
    };

    let cropped_dir = work_dir.join("cropped");
    if cropped_dir.exists() {
        std::fs::remove_dir_all(&cropped_dir)?;
    }
    std::fs::create_dir_all(&cropped_dir)?;

    let mut all_paths: Vec<PathBuf> = Vec::new();
    let mut group_boundaries: Vec<(usize, usize)> = Vec::new();

    for group in &frame_groups {
        let start = all_paths.len();
        for frame in group {
            let path = if let Some(ref region) = layout {
                let out_path = cropped_dir.join(format!(
                    "crop_{}.png",
                    frame
                        .path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                ));
                regions::crop_to_region(&frame.path, region, &out_path)?;
                out_path
            } else {
                frame.path.clone()
            };
            all_paths.push(path);
        }
        group_boundaries.push((start, group.len()));
    }

    let method = if args.gpu_ocr {
        "GPU OCR (PaddleOCR/ONNX)"
    } else if args.ocr {
        "OCR (tesseract)"
    } else {
        &format!("VLM ({})", args.model)
    };
    println!(
        "Extracting {} frames via {}{}...",
        all_paths.len(),
        method,
        if args.gpu_ocr {
            String::new()
        } else {
            format!(" with {} parallel workers", args.parallel)
        },
    );

    let frame_results: Vec<Result<String>> = if args.gpu_ocr {
        let mut engine = gpu_ocr::GpuOcrEngine::new(&args.model_dir)?;
        println!("  GPU OCR engine initialized");
        let start = std::time::Instant::now();
        let codes = engine.extract_batch(&all_paths);
        let elapsed = start.elapsed().as_secs_f64();
        let fps = if elapsed > 0.001 {
            all_paths.len() as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "  Extracted {} frames in {:.1}s ({:.1} fps)",
            all_paths.len(),
            elapsed,
            fps
        );
        codes
    } else if args.ocr {
        let sem = Arc::new(Semaphore::new(args.parallel));
        let mut handles = Vec::with_capacity(all_paths.len());
        for path in &all_paths {
            let sem = sem.clone();
            let path = path.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.context("semaphore closed")?;
                tokio::task::spawn_blocking(move || ocr::extract_code_from_frame(&path))
                    .await
                    .context("OCR task panicked")?
            }));
        }
        let mut codes = Vec::with_capacity(handles.len());
        for (i, handle) in handles.into_iter().enumerate() {
            codes.push(match handle.await {
                Ok(r) => r,
                Err(e) => Err(anyhow::anyhow!("OCR worker for frame {i} failed to complete: {e}")),
            });
        }
        codes
    } else {
        let path_refs: Vec<&Path> = all_paths.iter().map(|p| p.as_path()).collect();
        vlm::extract_batch(
            &path_refs,
            &args.ollama_host,
            &args.model,
            args.parallel,
            &args.vlm_options(),
        )
        .await?
    };

    let all_codes = apply_frame_failure_policy(frame_results, &all_paths, args.max_frame_failure_rate)?;

    println!("Building revisions...");
    let revisions = build_revisions(&frame_groups, &group_boundaries, &all_codes);

    println!("Stitching revisions...");
    let all_contents: Vec<String> = revisions.iter().map(|r| r.content.clone()).collect();
    let stitched = scroll::stitch_all_revisions(&all_contents);
    let cleaned = stitch::clean::clean_hallucinations(&stitched);
    let deduped = stitch::dedup::dedup_blocks(&cleaned);
    let section_deduped = stitch::dedup::dedup_sections(&deduped);
    let passage_deduped = stitch::dedup::dedup_passages(&section_deduped);

    let final_content = if args.refine {
        stitch::refine::refine_text(&passage_deduped, &args.refine_model, args.refine_agents).await?
    } else {
        passage_deduped
    };
    let stitch_lines = final_content.lines().count();
    println!(
        "  Stitched {stitch_lines} lines from {} revisions",
        revisions.len()
    );

    let language = detect_language(&final_content);
    let filename = infer_filename(&language);

    let extracted_file = ExtractedFile {
        filename: filename.clone(),
        language: language.clone(),
        final_content: final_content.clone(),
        revisions,
    };

    let output = PipelineOutput {
        files: vec![extracted_file],
        terminal_commands: Vec::new(),
        annotations: Vec::new(),
        duration,
        source_video: video_path.to_string_lossy().into_owned(),
    };

    let out_path = write_output(&output, &args.output)?;
    let line_count = final_content.lines().count();
    println!("\nOutput written to {}", out_path.display());
    println!(
        "  {} code revisions extracted",
        output.files[0].revisions.len()
    );
    println!("  Final file: {filename} ({line_count} lines)");

    Ok(())
}

/// Log and drop failed frames. The run fails only when the failure rate
/// exceeds `max_rate`, or when no frame succeeded at all.
fn apply_frame_failure_policy(
    results: Vec<Result<String>>,
    paths: &[PathBuf],
    max_rate: f64,
) -> Result<Vec<Option<String>>> {
    let total = results.len();
    let mut failed = 0usize;
    let codes: Vec<Option<String>> = results
        .into_iter()
        .enumerate()
        .map(|(i, r)| match r {
            Ok(code) => Some(code),
            Err(e) => {
                failed += 1;
                let name = paths.get(i).map(|p| p.display().to_string()).unwrap_or_default();
                eprintln!("  Warning: frame {i} ({name}) failed, skipping: {e:#}");
                None
            }
        })
        .collect();
    check_failure_rate(failed, total, max_rate)?;
    if failed > 0 {
        eprintln!("  {failed}/{total} frame(s) failed and were skipped");
    }
    Ok(codes)
}

fn check_failure_rate(failed: usize, total: usize, max_rate: f64) -> Result<()> {
    if total == 0 || failed == 0 {
        return Ok(());
    }
    if failed == total {
        anyhow::bail!("all {total} frame(s) failed extraction");
    }
    let rate = failed as f64 / total as f64;
    if rate > max_rate {
        anyhow::bail!(
            "{failed} of {total} frames failed extraction ({:.1}%), above the allowed {:.1}% \
             (--max-frame-failure-rate)",
            rate * 100.0,
            max_rate * 100.0
        );
    }
    Ok(())
}

/// Build one revision per frame group from the successful frames. A group
/// whose frames all failed is skipped; its timestamp comes from its first
/// successful frame.
fn build_revisions(
    frame_groups: &[Vec<SampledFrame>],
    group_boundaries: &[(usize, usize)],
    all_codes: &[Option<String>],
) -> Vec<CodeRevision> {
    let mut revisions: Vec<CodeRevision> = Vec::new();
    let mut previous_code: Option<String> = None;

    for (group_idx, (&(start, count), group)) in
        group_boundaries.iter().zip(frame_groups).enumerate()
    {
        let succeeded: Vec<(f64, String)> = group
            .iter()
            .zip(all_codes.iter().skip(start).take(count))
            .filter_map(|(frame, code)| code.as_ref().map(|c| (frame.timestamp, c.clone())))
            .collect();
        let Some(&(timestamp, _)) = succeeded.first() else {
            println!(
                "  Group {}/{}: all {count} frame(s) failed, skipped",
                group_idx + 1,
                frame_groups.len(),
            );
            continue;
        };
        let codes: Vec<String> = succeeded.into_iter().map(|(_, c)| c).collect();
        let code = if codes.len() > 1 {
            scroll::stitch_scroll_sequence(&codes)
        } else {
            codes.into_iter().next().unwrap_or_default()
        };

        let diff = scroll::compute_diff(previous_code.as_deref(), &code);
        let diff = if diff.is_empty() { None } else { Some(diff) };
        let line_count = code.lines().count();

        revisions.push(CodeRevision {
            timestamp,
            content: code.clone(),
            narration: None,
            frame_index: group_idx,
            diff,
        });

        println!(
            "  Group {}/{}: {} frame(s), {line_count} lines",
            group_idx + 1,
            frame_groups.len(),
            count,
        );

        previous_code = Some(code);
    }

    revisions
}

fn get_duration(video_path: &Path) -> f64 {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(video_path)
        .output();

    match output {
        Ok(out) => {
            if !out.status.success() {
                eprintln!("Warning: ffprobe failed to get video duration");
                return 0.0;
            }
            match String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse::<f64>()
            {
                Ok(d) => d,
                Err(_) => {
                    eprintln!("Warning: could not parse video duration");
                    0.0
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: ffprobe not available: {e}");
            0.0
        }
    }
}

fn detect_language(code: &str) -> String {
    let indicators: &[(&str, &[&str])] = &[
        (
            "rust",
            &["fn ", "let mut ", "impl ", "pub fn", "use std::", "-> "],
        ),
        (
            "python",
            &[
                "import ", "class ", "self.", "print(", "elif ", "except ", "async def ",
            ],
        ),
        (
            "typescript",
            &["interface ", ": string", ": number", "export ", "import {"],
        ),
        (
            "javascript",
            &[
                "const ",
                "let ",
                "function ",
                "=>",
                "console.log",
                "require(",
            ],
        ),
        (
            "go",
            &["func ", "package ", "import (", "fmt.", ":= ", "go func"],
        ),
        (
            "java",
            &[
                "public class",
                "public static",
                "System.out",
                "import java",
            ],
        ),
        ("ruby", &["end\n", "puts ", "require '", "attr_"]),
        (
            "c",
            &["#include", "int main", "printf(", "malloc(", "void "],
        ),
    ];

    let mut best_lang = "unknown";
    let mut best_score = 0usize;

    for (lang, patterns) in indicators {
        let score = patterns.iter().filter(|p| code.contains(**p)).count();
        if score > best_score {
            best_score = score;
            best_lang = lang;
        }
    }

    best_lang.to_string()
}

fn infer_filename(language: &str) -> String {
    match language {
        "python" => "extracted.py",
        "javascript" => "extracted.js",
        "typescript" => "extracted.ts",
        "rust" => "extracted.rs",
        "go" => "extracted.go",
        "java" => "Extracted.java",
        "c" => "extracted.c",
        "ruby" => "extracted.rb",
        _ => "extracted.txt",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_rust() {
        let code = "pub fn main() {\n    let mut x = 1;\n    println!(\"{}\", x);\n}\n";
        assert_eq!(detect_language(code), "rust");
    }

    #[test]
    fn detect_python() {
        let code = "import os\nclass Foo:\n    def bar(self):\n        print('hello')\n";
        assert_eq!(detect_language(code), "python");
    }

    #[test]
    fn detect_typescript() {
        let code = "import { Component } from 'react';\ninterface Props {\n  name: string;\n  count: number;\n}\nexport default Component;\n";
        assert_eq!(detect_language(code), "typescript");
    }

    #[test]
    fn detect_javascript_not_typescript() {
        let code = "const x = 1;\nlet y = 2;\nconsole.log(x + y);\nfunction add(a, b) { return a + b; }\n";
        assert_eq!(detect_language(code), "javascript");
    }

    #[test]
    fn detect_go() {
        let code = "package main\nimport (\n    \"fmt\"\n)\nfunc main() {\n    x := 1\n    fmt.Println(x)\n}\n";
        assert_eq!(detect_language(code), "go");
    }

    #[test]
    fn detect_unknown() {
        let code = "hello world\nfoo bar baz\n";
        assert_eq!(detect_language(code), "unknown");
    }

    fn frame(t: f64) -> SampledFrame {
        SampledFrame {
            timestamp: t,
            path: PathBuf::from(format!("f{t}.png")),
            is_keyframe: true,
        }
    }

    #[test]
    fn failure_rate_under_threshold_passes() {
        assert!(check_failure_rate(0, 10, 0.1).is_ok());
        assert!(check_failure_rate(1, 10, 0.1).is_ok());
        assert!(check_failure_rate(0, 0, 0.1).is_ok());
    }

    #[test]
    fn failure_rate_over_threshold_fails() {
        let err = check_failure_rate(2, 10, 0.1).unwrap_err();
        assert!(err.to_string().contains("2 of 10"), "{err}");
    }

    #[test]
    fn all_frames_failed_fails_even_at_rate_one() {
        assert!(check_failure_rate(3, 3, 1.0).is_err());
    }

    #[test]
    fn failure_policy_skips_failed_frames() {
        let results = vec![
            Ok("a".to_string()),
            Err(anyhow::anyhow!("boom")),
            Ok("c".to_string()),
        ];
        let paths: Vec<PathBuf> = ["1.png", "2.png", "3.png"].iter().map(PathBuf::from).collect();
        let codes = apply_frame_failure_policy(results, &paths, 0.5).unwrap();
        assert_eq!(codes, vec![Some("a".to_string()), None, Some("c".to_string())]);
    }

    #[test]
    fn failure_policy_errors_above_threshold() {
        let results = vec![Ok("a".to_string()), Err(anyhow::anyhow!("boom"))];
        let paths: Vec<PathBuf> = ["1.png", "2.png"].iter().map(PathBuf::from).collect();
        assert!(apply_frame_failure_policy(results, &paths, 0.1).is_err());
    }

    #[test]
    fn revisions_skip_failed_frames_and_groups() {
        let groups = vec![vec![frame(1.0), frame(2.0)], vec![frame(5.0)], vec![frame(9.0)]];
        let bounds = vec![(0, 2), (2, 1), (3, 1)];
        let codes = vec![None, Some("x = 1".to_string()), None, Some("y = 2".to_string())];
        let revs = build_revisions(&groups, &bounds, &codes);
        assert_eq!(revs.len(), 2);
        assert_eq!(revs[0].timestamp, 2.0);
        assert_eq!(revs[0].content, "x = 1");
        assert_eq!(revs[0].frame_index, 0);
        assert_eq!(revs[1].timestamp, 9.0);
        assert_eq!(revs[1].frame_index, 2);
        assert!(revs[1].diff.is_some());
    }

    #[test]
    fn revisions_all_success_match_previous_behavior() {
        let groups = vec![vec![frame(1.0)], vec![frame(3.0)]];
        let bounds = vec![(0, 1), (1, 1)];
        let codes = vec![Some("a".to_string()), Some("b".to_string())];
        let revs = build_revisions(&groups, &bounds, &codes);
        assert_eq!(revs.len(), 2);
        assert_eq!(revs[0].timestamp, 1.0);
        assert!(revs[0].diff.is_none());
        assert_eq!(revs[1].timestamp, 3.0);
    }

    #[test]
    fn vlm_options_use_timeout_flag() {
        let args = ScrapeArgs {
            video: PathBuf::from("v.mp4"),
            output: PathBuf::from("out"),
            model: "m".into(),
            ollama_host: "http://localhost:11434".into(),
            ssim_threshold: 0.95,
            ocr: false,
            gpu_ocr: false,
            model_dir: PathBuf::from("models"),
            work_dir: None,
            parallel: 1,
            refine: false,
            refine_model: "r".into(),
            refine_agents: 1,
            vlm_timeout_secs: 45,
            max_frame_failure_rate: DEFAULT_MAX_FRAME_FAILURE_RATE,
        };
        assert_eq!(args.vlm_options().timeout, std::time::Duration::from_secs(45));
    }

    #[test]
    fn infer_filename_known() {
        assert_eq!(infer_filename("rust"), "extracted.rs");
        assert_eq!(infer_filename("python"), "extracted.py");
    }

    #[test]
    fn infer_filename_unknown() {
        assert_eq!(infer_filename("unknown"), "extracted.txt");
        assert_eq!(infer_filename("brainfuck"), "extracted.txt");
    }
}
