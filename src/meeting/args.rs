//! `glassrip meeting` command-line arguments (spec section 4) and their
//! resolution into [`MeetingOptions`].

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use glassrip_core::config::{Config, HwAccel};
use glassrip_core::graph::Selection;

use super::run::MeetingOptions;

/// `--hwaccel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum HwAccelArg {
    /// Software decode (deterministic pixels).
    None,
    /// VideoToolbox or CUDA, with one software retry on failure.
    Auto,
}

/// `--speakers auto|N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerCount {
    /// Detect automatically.
    Auto,
    /// Known count.
    Known(u32),
}

fn parse_speakers(s: &str) -> Result<SpeakerCount, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(SpeakerCount::Auto);
    }
    match s.parse::<u32>() {
        Ok(n) if (1..=100).contains(&n) => Ok(SpeakerCount::Known(n)),
        _ => Err(format!(
            "expected `auto` or a count from 1 to 100, got `{s}`"
        )),
    }
}

fn parse_interval(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && (0.01..=3600.0).contains(&v) => Ok(v),
        _ => Err(format!("expected seconds between 0.01 and 3600, got `{s}`")),
    }
}

/// Arguments of `glassrip meeting`.
#[derive(Debug, Clone, Args)]
pub struct MeetingArgs {
    /// Meeting recording (screen or phone video).
    pub video: PathBuf,
    /// Output directory (default: ./<video-stem>.glassrip/).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Configuration file (default: ./glassrip.toml when present).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Ollama URL (overrides GLASSRIP_OLLAMA_URL, OLLAMA_HOST, and the config).
    #[arg(long)]
    pub host: Option<String>,
    /// Vision model (default: models.vision, qwen2.5vl:7b).
    #[arg(long)]
    pub vision_model: Option<String>,
    /// Text model for notes synthesis (default: models.text).
    #[arg(long)]
    pub text_model: Option<String>,
    /// whisper.cpp ggml model name or path (default: models.asr, large-v3-turbo).
    #[arg(long)]
    pub asr_model: Option<String>,
    /// Participant display names, comma separated.
    #[arg(long)]
    pub participants: Option<String>,
    /// Speaker count: `auto` or a number.
    #[arg(long, value_parser = parse_speakers)]
    pub speakers: Option<SpeakerCount>,
    /// Frame sampling interval, seconds (default 2.0).
    #[arg(long, value_parser = parse_interval)]
    pub interval: Option<f64>,
    /// Decoder: software (`none`, default) or hardware (`auto`).
    #[arg(long, value_enum)]
    pub hwaccel: Option<HwAccelArg>,
    /// Start at this stage (earlier artifacts must exist in the output directory).
    #[arg(long)]
    pub from_stage: Option<String>,
    /// Stop after this stage.
    #[arg(long)]
    pub until_stage: Option<String>,
    /// Bypass the cache for this stage (repeatable).
    #[arg(long)]
    pub force_stage: Vec<String>,
    /// Permit partial CPU placement of the models (warn instead of failing preflight).
    #[arg(long)]
    pub allow_spill: bool,
    /// Run on a GPU host over the LAN (`user@host`); not implemented yet (M5).
    #[arg(long)]
    pub remote: Option<String>,
}

/// Loads `--config`, else `./glassrip.toml` when present, else defaults.
pub fn load_config(path: Option<&Path>) -> anyhow::Result<Config> {
    let default = PathBuf::from("glassrip.toml");
    let p = match path {
        Some(p) => Some(p.to_path_buf()),
        None => default.is_file().then_some(default),
    };
    match p {
        Some(p) => Config::load(&p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display())),
        None => Ok(Config::default()),
    }
}

/// Splits `"A B, C D"` into trimmed, non-empty names.
pub fn split_names(s: &str) -> Vec<String> {
    s.split(',')
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .collect()
}

/// Filesystem-safe stem of the video file name.
pub fn video_stem(video: &Path) -> String {
    let stem = video
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let safe = glassrip_render::file_safe(&stem);
    if safe.is_empty() {
        "meeting".into()
    } else {
        safe
    }
}

impl MeetingArgs {
    /// Applies the flags over the configuration and builds the run options.
    pub fn resolve(&self) -> anyhow::Result<MeetingOptions> {
        if let Some(target) = &self.remote {
            anyhow::bail!(
                "--remote {target}: remote runs are not implemented yet (milestone M5); \
                 copy the video to the GPU host and run `glassrip meeting` there"
            );
        }
        if !self.video.is_file() {
            anyhow::bail!("{} not found", self.video.display());
        }
        let mut config = load_config(self.config.as_deref())?;
        config.ollama.host = glassrip_eval::cli::resolve_host(self.host.as_deref(), &config);
        if let Some(m) = &self.vision_model {
            config.models.vision = m.clone();
        }
        if let Some(m) = &self.text_model {
            config.models.text = m.clone();
        }
        if let Some(m) = &self.asr_model {
            config.models.asr = m.clone();
        }
        match self.speakers {
            Some(SpeakerCount::Auto) => config.audio.speakers = None,
            Some(SpeakerCount::Known(n)) => config.audio.speakers = Some(n),
            None => {}
        }
        if let Some(i) = self.interval {
            config.frames.interval_s = i;
        }
        match self.hwaccel {
            Some(HwAccelArg::None) => config.frames.hwaccel = HwAccel::None,
            Some(HwAccelArg::Auto) => config.frames.hwaccel = HwAccel::Auto,
            None => {}
        }
        if self.allow_spill {
            config.gpu.allow_spill = true;
        }
        // Round-trip through TOML so every override is range-checked like a file.
        let config = Config::from_toml_str(&config.to_toml_string()?)
            .map_err(|e| anyhow::anyhow!("invalid settings: {e}"))?;

        let stem = video_stem(&self.video);
        let out_dir = self
            .out
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{stem}.glassrip")));
        let workspace = std::env::var_os("GLASSRIP_WORKSPACE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let selection = Selection {
            from: self.from_stage.clone(),
            until: self.until_stage.clone(),
            force: self.force_stage.iter().cloned().collect::<BTreeSet<_>>(),
        };
        let mut opts = MeetingOptions::new(self.video.clone(), out_dir, &workspace, config);
        opts.stem = stem;
        opts.participants = self
            .participants
            .as_deref()
            .map(split_names)
            .unwrap_or_default();
        opts.selection = selection;
        Ok(opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: MeetingArgs,
    }

    fn parse(extra: &[&str]) -> Result<MeetingArgs, clap::Error> {
        let mut argv = vec!["x", "call.mp4"];
        argv.extend_from_slice(extra);
        Wrap::try_parse_from(argv).map(|w| w.args)
    }

    #[test]
    fn spec_flags_parse() {
        let a = parse(&[
            "--out",
            "o",
            "--host",
            "http://h:1",
            "--vision-model",
            "v:1",
            "--text-model",
            "t:2",
            "--asr-model",
            "small",
            "--participants",
            "Ada Quill, Bo Tran",
            "--speakers",
            "3",
            "--interval",
            "1.5",
            "--hwaccel",
            "auto",
            "--from-stage",
            "classify",
            "--until-stage",
            "board_state",
            "--force-stage",
            "board_read",
            "--force-stage",
            "classify",
            "--allow-spill",
        ])
        .unwrap();
        assert_eq!(a.speakers, Some(SpeakerCount::Known(3)));
        assert_eq!(a.interval, Some(1.5));
        assert_eq!(a.hwaccel, Some(HwAccelArg::Auto));
        assert_eq!(a.force_stage, vec!["board_read", "classify"]);
        assert!(a.allow_spill);
        assert_eq!(
            split_names(a.participants.as_deref().unwrap()),
            vec!["Ada Quill", "Bo Tran"]
        );
        assert_eq!(
            parse(&["--speakers", "auto"]).unwrap().speakers,
            Some(SpeakerCount::Auto)
        );
        assert!(parse(&["--speakers", "0"]).is_err());
        assert!(parse(&["--interval", "0"]).is_err());
        assert!(parse(&["--hwaccel", "gpu"]).is_err());
    }

    #[test]
    fn remote_is_not_yet_supported() {
        let a = parse(&["--remote", "user@gpu-host"]).unwrap();
        let err = a.resolve().unwrap_err().to_string();
        assert!(
            err.contains("not implemented yet") && err.contains("M5"),
            "{err}"
        );
    }

    #[test]
    fn stems_are_file_safe() {
        assert_eq!(video_stem(Path::new("/x/My Call (1).mp4")), "My-Call--1");
        assert_eq!(video_stem(Path::new("/x/.mp4")), "mp4");
    }

    #[test]
    fn resolve_applies_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("sync call.mp4");
        std::fs::write(&video, b"not really a video").unwrap();
        let cfg = dir.path().join("g.toml");
        std::fs::write(&cfg, "[frames]\ninterval_s = 4.0\n").unwrap();
        let a = Wrap::try_parse_from([
            "x",
            video.to_str().unwrap(),
            "--config",
            cfg.to_str().unwrap(),
            "--speakers",
            "2",
            "--vision-model",
            "v:9",
            "--until-stage",
            "keyframes",
        ])
        .unwrap()
        .args;
        let o = a.resolve().unwrap();
        assert_eq!(o.config.frames.interval_s, 4.0);
        assert_eq!(o.config.audio.speakers, Some(2));
        assert_eq!(o.config.models.vision, "v:9");
        assert_eq!(o.stem, "sync-call");
        assert_eq!(o.out_dir, PathBuf::from("sync-call.glassrip"));
        assert_eq!(o.selection.until.as_deref(), Some("keyframes"));
    }
}
