//! Preflight (spec 5.3, 8.1): before anything runs, every tool, feature, and
//! model the selected stages need must be present. All problems are reported
//! together; nothing runs when one is found.
//!
//! The text model is the exception: `notes` needs it only when the recording
//! has speech, which is known after transcription. A missing text model is a
//! warning here; the run checks it again before `notes` (see
//! [`super::run`]), which fails then unless the notes restore from cache or
//! come from the board alone.
//!
//! GPU placement itself is checked lazily by the vision stages' placement
//! monitor (Ollama `/api/ps` before the first request and periodically) and by
//! the notes stage for the text model. With the `cuda` feature, preflight also
//! reports the GPU's other tenants by name (`nvidia-smi`); glassrip never stops them.

use glassrip_core::graph::{Plan, StageDecision};

use super::backends::Backends;

/// Media stages (ffmpeg and ffprobe).
pub const MEDIA_STAGES: [&str; 7] = [
    "probe",
    "orient",
    "frames",
    "screen_quad",
    "features",
    "keyframes",
    "rectify",
];

/// What the selected stages need.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Needs {
    /// ffmpeg and ffprobe (media stages, `audio_extract`).
    pub ffmpeg: bool,
    /// PP-OCRv5 (`ocr_harvest`).
    pub ocr: bool,
    /// The vision model (`classify`, `board_read`).
    pub vision: bool,
    /// Optional vision model use (`edge_direction` fallback).
    pub vision_optional: bool,
    /// The text model (`notes`).
    pub text: bool,
    /// The recognizer (`asr`).
    pub asr: bool,
    /// The diarizer (`diarize`).
    pub diarize: bool,
}

impl Needs {
    /// Needs of the stages the plan runs.
    pub fn from_plan(plan: &Plan) -> Self {
        let runs = |s: &str| matches!(plan.decision(s), Some(StageDecision::Run { .. }));
        Self {
            ffmpeg: MEDIA_STAGES.iter().any(|s| runs(s)) || runs("audio_extract"),
            ocr: runs("ocr_harvest"),
            vision: runs("classify") || runs("board_read"),
            vision_optional: runs("edge_direction"),
            text: runs("notes"),
            asr: runs("asr"),
            diarize: runs("diarize"),
        }
    }
}

/// ffmpeg and ffprobe version strings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolVersions {
    /// `ffmpeg -version` first line.
    pub ffmpeg: Option<String>,
    /// `ffprobe -version` first line.
    pub ffprobe: Option<String>,
}

/// Checks tools and backends; returns the tool versions or every problem found.
pub fn check(
    needs: &Needs,
    backends: &Backends,
    ffmpeg: &str,
    ffprobe: &str,
) -> Result<ToolVersions, Vec<String>> {
    let mut problems = Vec::new();
    let mut tools = ToolVersions::default();
    if needs.ffmpeg {
        match glassrip_media_stages::util::tool_version(ffmpeg) {
            Ok(v) => tools.ffmpeg = Some(v),
            Err(e) => problems.push(format!("ffmpeg is required ({ffmpeg}): {e}")),
        }
        match glassrip_media_stages::util::tool_version(ffprobe) {
            Ok(v) => tools.ffprobe = Some(v),
            Err(e) => problems.push(format!("ffprobe is required ({ffprobe}): {e}")),
        }
    }
    let mut need = |wanted: bool, stage: &str, avail: Result<(), &String>| {
        if let (true, Err(reason)) = (wanted, avail) {
            problems.push(format!("{stage}: {reason}"));
        }
    };
    need(needs.ocr, "ocr_harvest", backends.ocr.as_ref().map(|_| ()));
    need(
        needs.vision,
        "classify/board_read",
        backends.vision.as_ref().map(|_| ()),
    );
    if let (true, Err(reason)) = (needs.text, &backends.text) {
        tracing::warn!(
            %reason,
            "text model unavailable: notes come from the board alone if no speech is \
             transcribed; with speech the run stops at notes"
        );
    }
    need(needs.asr, "asr", backends.asr.as_ref().map(|_| ()));
    need(
        needs.diarize,
        "diarize",
        backends.diarize.as_ref().map(|_| ()),
    );
    if problems.is_empty() {
        if cfg!(feature = "cuda") {
            if let Some(report) = gpu_tenants() {
                tracing::info!(tenants = %report, "GPU processes before the run");
            }
        }
        Ok(tools)
    } else {
        Err(problems)
    }
}

/// Other GPU processes with their memory, from `nvidia-smi` (None when it
/// is not available).
pub fn gpu_tenants() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    out.status.success().then(|| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("; ")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glassrip_core::graph::{meeting_mode_stage_decls, Selection, StageGraph};

    fn plan(until: Option<&str>, from: Option<&str>) -> Plan {
        StageGraph::new(meeting_mode_stage_decls())
            .unwrap()
            .plan(&Selection {
                until: until.map(str::to_string),
                from: from.map(str::to_string),
                ..Selection::default()
            })
            .unwrap()
    }

    #[test]
    fn needs_follow_the_selection() {
        let media = Needs::from_plan(&plan(Some("rectify"), None));
        assert!(media.ffmpeg && !media.ocr && !media.vision && !media.text && !media.asr);
        let all = Needs::from_plan(&plan(None, None));
        assert!(all.ocr && all.vision && all.text && all.asr && all.diarize);
        let tail = Needs::from_plan(&plan(None, Some("notes")));
        assert!(tail.text && !tail.ffmpeg && !tail.vision);
    }

    #[test]
    fn every_missing_backend_is_reported_at_once() {
        let needs = Needs::from_plan(&plan(None, None));
        let problems = check(
            &needs,
            &Backends::none("unavailable here"),
            "ffmpeg",
            "ffprobe",
        )
        .unwrap_err();
        for stage in ["ocr_harvest", "classify/board_read", "asr", "diarize"] {
            assert!(
                problems.iter().any(|p| p.starts_with(stage)),
                "{stage}: {problems:?}"
            );
        }
        // A silent recording needs no text model: that is known only after
        // transcription, so a missing one does not fail preflight.
        assert!(
            !problems.iter().any(|p| p.starts_with("notes")),
            "{problems:?}"
        );
        // Media-only runs need nothing but the tools.
        let media = Needs::from_plan(&plan(Some("rectify"), None));
        let r = check(&media, &Backends::none("x"), "ffmpeg", "ffprobe");
        match r {
            Ok(t) => assert!(t.ffmpeg.is_some()),
            Err(p) => assert!(p.iter().all(|m| m.contains("is required")), "{p:?}"),
        }
    }
}
