//! `glassrip.toml` configuration.
//!
//! Every section and field has a default, so an empty file is valid. Unknown keys are
//! rejected at every level (`deny_unknown_fields`), and values are range-checked with
//! `garde` after parsing. [`Config::json_schema`] returns the JSON Schema for editors
//! and for the generated `glassrip.config.schema.json`.

use std::path::{Path, PathBuf};

use garde::Validate;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Error loading configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("cannot read config: {0}")]
    Io(#[from] std::io::Error),
    /// TOML syntax error, wrong type, or unknown key.
    #[error("invalid config TOML: {0}")]
    Parse(#[from] toml::de::Error),
    /// A value is out of range.
    #[error("invalid config values: {0}")]
    Invalid(#[from] garde::Report),
    /// Serialization failed.
    #[error("cannot serialize config: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// Top-level configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Ollama server and request options.
    #[garde(dive)]
    pub ollama: OllamaConfig,
    /// Model names.
    #[garde(dive)]
    pub models: ModelsConfig,
    /// Frame sampling (6.3).
    #[garde(dive)]
    pub frames: FramesConfig,
    /// Orientation detection (6.2).
    #[garde(dive)]
    pub orient: OrientConfig,
    /// Features, alignment, and ink (6.5).
    #[garde(dive)]
    pub features: FeaturesConfig,
    /// Keyframe segmentation (6.6).
    #[garde(dive)]
    pub keyframes: KeyframesConfig,
    /// Screen classification (6.7).
    #[garde(dive)]
    pub classify: ClassifyConfig,
    /// Canvas crop and chrome masking (6.9).
    #[garde(dive)]
    pub canvas: CanvasConfig,
    /// Board reading (6.10).
    #[garde(dive)]
    pub board_read: BoardReadConfig,
    /// Board validation and edge direction (6.11).
    #[garde(dive)]
    pub edge_direction: EdgeDirectionConfig,
    /// Board state consolidation (6.11).
    #[garde(dive)]
    pub board_state: BoardStateConfig,
    /// Audio extraction, ASR, diarization, assignment (6.12).
    #[garde(dive)]
    pub audio: AudioConfig,
    /// Notes synthesis (6.14).
    #[garde(dive)]
    pub notes: NotesConfig,
    /// GPU scheduling (5.3).
    #[garde(dive)]
    pub gpu: GpuConfig,
    /// Stage runner.
    #[garde(dive)]
    pub runner: RunnerConfig,
    /// Evaluation.
    #[garde(dive)]
    pub eval: EvalConfig,
}

impl Config {
    /// Parses and validates TOML text.
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Reads, parses, and validates a file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        Self::from_toml_str(&fs_err::read_to_string(path)?)
    }

    /// Serializes to TOML (every field, with current values).
    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// JSON Schema for the configuration file.
    pub fn json_schema() -> serde_json::Value {
        schemars::schema_for!(Config).to_value()
    }
}

/// Ollama server and request options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct OllamaConfig {
    /// Server URL.
    #[garde(url)]
    pub host: String,
    /// Per-request timeout for 7b-class models, seconds.
    #[garde(range(min = 1))]
    pub request_timeout_s: u64,
    /// Per-request timeout for 30B-class models, seconds.
    #[garde(range(min = 1))]
    pub large_model_request_timeout_s: u64,
    /// Context length, identical for every request in a run.
    #[garde(range(min = 512))]
    pub num_ctx: u32,
    /// Maximum tokens generated per request.
    #[garde(range(min = 1))]
    pub num_predict: u32,
    /// Keep-alive for the loaded model during a phase.
    #[garde(length(min = 1))]
    pub keep_alive: String,
    /// Sampling temperature.
    #[garde(range(min = 0.0, max = 2.0))]
    pub temperature: f64,
    /// Sampling seed.
    #[garde(skip)]
    pub seed: u64,
    /// Server slot count (`OLLAMA_NUM_PARALLEL`); client concurrency matches it.
    #[garde(range(min = 1, max = 64))]
    pub num_parallel: u32,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".into(),
            request_timeout_s: 120,
            large_model_request_timeout_s: 400,
            num_ctx: 8192,
            num_predict: 2048,
            keep_alive: "30m".into(),
            temperature: 0.0,
            seed: 0,
            num_parallel: 4,
        }
    }
}

/// Model names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    /// Vision model for classification, board reading, and edge checks.
    #[garde(length(min = 1))]
    pub vision: String,
    /// Text model for notes synthesis.
    #[garde(length(min = 1))]
    pub text: String,
    /// whisper.cpp ggml model name or path.
    #[garde(length(min = 1))]
    pub asr: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            vision: "qwen2.5vl:7b".into(),
            text: "qwen3.6:27b".into(),
            asr: "large-v3-turbo".into(),
        }
    }
}

/// Hardware decode choice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HwAccel {
    /// Software decode (deterministic pixels).
    #[default]
    None,
    /// VideoToolbox or CUDA, with one software retry on failure.
    Auto,
}

/// Frame sampling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct FramesConfig {
    /// Sampling interval in seconds.
    #[garde(range(min = 0.01, max = 3600.0))]
    pub interval_s: f64,
    /// Output frame width in pixels (height keeps aspect).
    #[garde(range(min = 16, max = 8192))]
    pub scale_width: u32,
    /// Decoder selection.
    #[garde(skip)]
    pub hwaccel: HwAccel,
}

impl Default for FramesConfig {
    fn default() -> Self {
        Self {
            interval_s: 2.0,
            scale_width: 1920,
            hwaccel: HwAccel::None,
        }
    }
}

/// Orientation detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct OrientConfig {
    /// Frames sampled across the video for the rotation vote.
    #[garde(range(min = 1, max = 1000))]
    pub sample_frames: u32,
    /// Skip orientation detection and apply this clockwise rotation (0, 90, 180,
    /// or 270), for example for screen recordings; absent means detect.
    #[garde(skip)]
    pub override_rotation_deg: Option<u32>,
}

impl Default for OrientConfig {
    fn default() -> Self {
        Self {
            sample_frames: 12,
            override_rotation_deg: None,
        }
    }
}

/// Features mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FeaturesMode {
    /// Improved alignment and masking (default).
    #[default]
    Production,
    /// Reproduces the reference implementation exactly (parity gate only).
    PrototypeCompat,
}

/// Features, alignment, and ink metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct FeaturesConfig {
    /// Mode (part of the cache key).
    #[garde(skip)]
    pub mode: FeaturesMode,
    /// ECC iteration limit.
    #[garde(range(min = 1, max = 10000))]
    pub ecc_iterations: u32,
    /// ECC convergence epsilon.
    #[garde(range(min = 1e-12, max = 1.0))]
    pub ecc_eps: f64,
    /// ECC Gaussian filter size.
    #[garde(range(min = 1, max = 31))]
    pub ecc_gauss_filt_size: u32,
    /// Absolute gray difference counted as a changed pixel.
    #[garde(range(min = 0, max = 255))]
    pub changed_pixel_delta: u32,
    /// Fixed evaluation inset in `prototype_compat` mode, pixels.
    #[garde(range(max = 1000))]
    pub compat_inset_px: u32,
    /// SSIM Gaussian window size.
    #[garde(range(min = 1, max = 63))]
    pub ssim_window: u32,
    /// SSIM Gaussian sigma.
    #[garde(range(min = 0.01, max = 100.0))]
    pub ssim_sigma: f64,
    /// Ink adaptive threshold block size.
    #[garde(range(min = 3, max = 255))]
    pub ink_block_size: u32,
    /// Ink adaptive threshold constant.
    #[garde(range(min = 0.0, max = 255.0))]
    pub ink_threshold_c: f64,
    /// Minimum HSV saturation for colored ink.
    #[garde(range(min = 0, max = 255))]
    pub ink_min_saturation: u32,
    /// Minimum HSV value for colored ink.
    #[garde(range(min = 0, max = 255))]
    pub ink_min_value: u32,
    /// Border masked out of the ink comparison, pixels.
    #[garde(range(max = 1000))]
    pub ink_border_px: u32,
    /// Dilation kernel size for the ink comparison.
    #[garde(range(min = 1, max = 63))]
    pub ink_dilate_px: u32,
    /// `production`: largest accepted deviation of an alignment's linear part from the
    /// identity (scale, shear); larger warps count as `align_failed`.
    #[garde(range(min = 0.0, max = 1.0))]
    pub production_max_linear_dev: f64,
    /// `production`: pairs whose warp covers less than this share of the frame count as
    /// `align_failed`.
    #[garde(range(min = 0.0, max = 1.0))]
    pub production_min_valid_frac: f64,
    /// `production`: minimum phase-correlation response trusted as a translation.
    #[garde(range(min = 0.0, max = 1.0))]
    pub production_min_phase_response: f64,
    /// `production`: largest accepted translation as a share of width or height.
    #[garde(range(min = 0.0, max = 1.0))]
    pub production_max_shift_frac: f64,
}

impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            mode: FeaturesMode::Production,
            ecc_iterations: 60,
            ecc_eps: 1e-4,
            ecc_gauss_filt_size: 3,
            changed_pixel_delta: 25,
            compat_inset_px: 15,
            ssim_window: 7,
            ssim_sigma: 1.5,
            ink_block_size: 31,
            ink_threshold_c: 12.0,
            ink_min_saturation: 70,
            ink_min_value: 80,
            ink_border_px: 20,
            ink_dilate_px: 5,
            production_max_linear_dev: 0.15,
            production_min_valid_frac: 0.5,
            production_min_phase_response: 0.05,
            production_max_shift_frac: 0.5,
        }
    }
}

/// Keyframe segmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct KeyframesConfig {
    /// A pair differs when SSIM is below this.
    #[garde(range(min = 0.0, max = 1.0))]
    pub ssim_threshold: f64,
    /// A pair differs when the changed-pixel fraction is above this.
    #[garde(range(min = 0.0, max = 1.0))]
    pub changed_frac_threshold: f64,
    /// A pair differs when the ink change is above this.
    #[garde(range(min = 0.0, max = 1.0))]
    pub ink_threshold: f64,
    /// Samples a change must persist for before a new run starts.
    #[garde(range(min = 1, max = 100))]
    pub persistence: u32,
    /// Singleton merge: minimum SSIM against the neighbor representative.
    #[garde(range(min = 0.0, max = 1.0))]
    pub merge_min_ssim: f64,
    /// Singleton merge: maximum changed fraction against the neighbor representative.
    #[garde(range(min = 0.0, max = 1.0))]
    pub merge_max_changed_frac: f64,
    /// Singleton merge: a singleton whose sharpness is below this times the median
    /// sharpness merges regardless of content.
    #[garde(range(min = 0.0, max = 1.0))]
    pub merge_blur_ratio: f64,
    /// `production`: the representative is the run frame nearest the run's temporal
    /// center among frames within this share of the best sharpness (merged singletons
    /// excluded). `prototype_compat` always takes the sharpest frame.
    #[garde(range(min = 0.0, max = 1.0))]
    pub production_rep_sharpness_tolerance: f64,
}

impl Default for KeyframesConfig {
    fn default() -> Self {
        Self {
            ssim_threshold: 0.80,
            changed_frac_threshold: 0.10,
            ink_threshold: 0.05,
            persistence: 2,
            merge_min_ssim: 0.70,
            merge_max_changed_frac: 0.20,
            merge_blur_ratio: 0.35,
            production_rep_sharpness_tolerance: 0.05,
        }
    }
}

/// Screen classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct ClassifyConfig {
    /// Long edge of the classification thumbnail, pixels.
    #[garde(range(min = 64, max = 4096))]
    pub thumbnail_px: u32,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self { thumbnail_px: 768 }
    }
}

/// Canvas crop and chrome masking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct CanvasConfig {
    /// UI strings never accepted as board content.
    #[garde(skip)]
    pub chrome_denylist: Vec<String>,
    /// Banner substrings marking conferencing UI.
    #[garde(skip)]
    pub banner_patterns: Vec<String>,
}

impl Default for CanvasConfig {
    fn default() -> Self {
        Self {
            chrome_denylist: [
                "Overview",
                "Browse",
                "Create section",
                "Convert to",
                "Share",
                "Internal",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            banner_patterns: vec!["(Presenting".into()],
        }
    }
}

/// Board reading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct BoardReadConfig {
    /// Crops with a shorter long edge are upscaled and marked low resolution.
    #[garde(range(min = 64, max = 8192))]
    pub min_long_edge_px: u32,
    /// Long edge sent to the model.
    #[garde(range(min = 64, max = 8192))]
    pub target_long_edge_px: u32,
    /// Upscale factor for low-resolution crops.
    #[garde(range(min = 1.0, max = 4.0))]
    pub low_res_upscale: f64,
    /// Median canvas text height (pixels) below which tiling is used.
    #[garde(range(min = 0.0, max = 1000.0))]
    pub tiling_text_height_px: f64,
    /// Tiles per side.
    #[garde(range(min = 1, max = 8))]
    pub tile_grid: u32,
    /// Tile overlap fraction.
    #[garde(range(min = 0.0, max = 0.5))]
    pub tile_overlap: f64,
}

impl Default for BoardReadConfig {
    fn default() -> Self {
        Self {
            min_long_edge_px: 1280,
            target_long_edge_px: 1920,
            low_res_upscale: 1.5,
            tiling_text_height_px: 14.0,
            tile_grid: 2,
            tile_overlap: 0.12,
        }
    }
}

/// Board validation and edge direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct EdgeDirectionConfig {
    /// Distance from an endpoint searched for an arrowhead, pixels.
    #[garde(range(min = 1, max = 500))]
    pub arrowhead_search_px: u32,
    /// Maximum words in an edge label.
    #[garde(range(min = 1, max = 100))]
    pub max_label_words: u32,
}

impl Default for EdgeDirectionConfig {
    fn default() -> Self {
        Self {
            arrowhead_search_px: 15,
            max_label_words: 8,
        }
    }
}

/// Board state consolidation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct BoardStateConfig {
    /// Fuzzy text match ratio.
    #[garde(range(min = 0.0, max = 1.0))]
    pub fuzzy_ratio: f64,
    /// Keyframes needed to support an element.
    #[garde(range(min = 1, max = 1000))]
    pub min_support_keyframes: u32,
    /// Alternatively, fraction of board keyframes in the interval.
    #[garde(range(min = 0.0, max = 1.0))]
    pub min_support_fraction: f64,
    /// Consistent keyframes needed to open or move an owner tag.
    #[garde(range(min = 1, max = 1000))]
    pub owner_min_keyframes: u32,
}

impl Default for BoardStateConfig {
    fn default() -> Self {
        Self {
            fuzzy_ratio: 0.85,
            min_support_keyframes: 2,
            min_support_fraction: 0.10,
            owner_min_keyframes: 2,
        }
    }
}

/// Audio pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    /// Extraction sample rate, Hz.
    #[garde(range(min = 8000, max = 48000))]
    pub sample_rate_hz: u32,
    /// ASR language.
    #[garde(length(min = 2))]
    pub language: String,
    /// Beam size.
    #[garde(range(min = 1, max = 32))]
    pub beam_size: u32,
    /// VAD model file name.
    #[garde(length(min = 1))]
    pub vad_model: String,
    /// Maximum tokens in the vocabulary prompt.
    #[garde(range(min = 0, max = 1000))]
    pub vocabulary_max_tokens: u32,
    /// Words without an overlapping turn go to the nearest turn within this, seconds.
    #[garde(range(min = 0.0, max = 60.0))]
    pub nearest_turn_s: f64,
    /// Known speaker count; absent means detect automatically.
    #[garde(inner(range(min = 1, max = 100)))]
    pub speakers: Option<u32>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            language: "en".into(),
            beam_size: 5,
            vad_model: "ggml-silero-v6.2.0.bin".into(),
            vocabulary_max_tokens: 200,
            nearest_turn_s: 0.5,
            speakers: None,
        }
    }
}

/// Notes synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct NotesConfig {
    /// Dropped-item fraction above which the notes are marked degraded.
    #[garde(range(min = 0.0, max = 1.0))]
    pub max_drop_fraction: f64,
    /// Transcripts longer than this must yield decisions, action items, and a summary.
    #[garde(range(min = 0.0))]
    pub min_transcript_s: f64,
    /// Repair retries for items failing citation validation.
    #[garde(range(max = 10))]
    pub repair_retries: u32,
}

impl Default for NotesConfig {
    fn default() -> Self {
        Self {
            max_drop_fraction: 0.30,
            min_transcript_s: 300.0,
            repair_retries: 1,
        }
    }
}

/// GPU scheduling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    /// Loaded models above this size run with one slot and without concurrent ASR, GB.
    #[garde(range(min = 0.0))]
    pub sequential_model_threshold_gb: f64,
    /// Permit partial CPU placement of the vision model.
    #[garde(skip)]
    pub allow_spill: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            sequential_model_threshold_gb: 14.0,
            allow_spill: false,
        }
    }
}

/// Stage runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct RunnerConfig {
    /// A stage fails when its item error rate exceeds this.
    #[garde(range(min = 0.0, max = 1.0))]
    pub max_item_error_rate: f64,
    /// Appended items between JSONL `sync_data` checkpoints.
    #[garde(range(min = 1))]
    pub checkpoint_every: u32,
    /// Default per-item timeout, seconds (stages may override); absent means none.
    #[garde(inner(range(min = 0.001)))]
    pub item_timeout_s: Option<f64>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_item_error_rate: 0.10,
            checkpoint_every: 16,
            item_timeout_s: None,
        }
    }
}

/// Evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct EvalConfig {
    /// Root of the private fixtures (never inside the public repo). Absent means the
    /// private suite is skipped.
    #[garde(skip)]
    pub private_fixtures: Option<PathBuf>,
    /// Gate fails when any F1 drops more than this many points below baseline.
    #[garde(range(min = 0.0, max = 100.0))]
    pub max_f1_drop_points: f64,
    /// Tolerance when joining golden times to keyframes, seconds.
    #[garde(range(min = 0.0))]
    pub time_join_tolerance_s: f64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            private_fixtures: None,
            max_f1_drop_points: 2.0,
            time_join_tolerance_s: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_gives_defaults() {
        let c = Config::from_toml_str("").unwrap();
        assert_eq!(c, Config::default());
        assert_eq!(c.frames.interval_s, 2.0);
        assert_eq!(c.keyframes.ssim_threshold, 0.80);
        assert_eq!(c.keyframes.changed_frac_threshold, 0.10);
        assert_eq!(c.keyframes.ink_threshold, 0.05);
        assert_eq!(c.keyframes.persistence, 2);
        assert_eq!(
            (
                c.keyframes.merge_min_ssim,
                c.keyframes.merge_max_changed_frac,
                c.keyframes.merge_blur_ratio
            ),
            (0.70, 0.20, 0.35)
        );
        assert_eq!(c.models.vision, "qwen2.5vl:7b");
        assert_eq!(c.ollama.num_ctx, 8192);
        assert_eq!(c.ollama.num_predict, 2048);
        assert_eq!(c.ollama.host, "http://localhost:11434");
        assert_eq!(
            (
                c.ollama.request_timeout_s,
                c.ollama.large_model_request_timeout_s
            ),
            (120, 400)
        );
        assert_eq!(c.runner.max_item_error_rate, 0.10);
        assert_eq!(c.eval.private_fixtures, None);
        assert_eq!(c.frames.hwaccel, HwAccel::None);
        assert_eq!(c.features.mode, FeaturesMode::Production);
    }

    #[test]
    fn partial_override_keeps_other_defaults() {
        let c = Config::from_toml_str(
            r#"
            [frames]
            interval_s = 1.0
            [eval]
            private_fixtures = "/tmp/private-fixtures"
            [audio]
            speakers = 3
            [features]
            mode = "prototype_compat"
            "#,
        )
        .unwrap();
        assert_eq!(c.frames.interval_s, 1.0);
        assert_eq!(c.frames.scale_width, 1920);
        assert_eq!(c.audio.speakers, Some(3));
        assert_eq!(c.features.mode, FeaturesMode::PrototypeCompat);
        assert_eq!(
            c.eval.private_fixtures,
            Some(PathBuf::from("/tmp/private-fixtures"))
        );
    }

    #[test]
    fn unknown_fields_rejected() {
        for text in [
            "unknown_top = 1",
            "[frames]\ninterval = 2.0",
            "[nonexistent]\nx = 1",
            "[keyframes]\nssim_threshhold = 0.8",
        ] {
            assert!(
                matches!(Config::from_toml_str(text), Err(ConfigError::Parse(_))),
                "{text}"
            );
        }
        assert!(matches!(
            Config::from_toml_str("[features]\nmode = \"fast\""),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn out_of_range_rejected() {
        for text in [
            "[keyframes]\nssim_threshold = 1.5",
            "[frames]\ninterval_s = 0.0",
            "[runner]\nmax_item_error_rate = -0.1",
            "[ollama]\nhost = \"not a url\"",
            "[audio]\nspeakers = 0",
            "[models]\nvision = \"\"",
        ] {
            assert!(
                matches!(Config::from_toml_str(text), Err(ConfigError::Invalid(_))),
                "{text}"
            );
        }
    }

    #[test]
    fn round_trips_through_toml() {
        let text = Config::default().to_toml_string().unwrap();
        assert_eq!(Config::from_toml_str(&text).unwrap(), Config::default());
    }

    #[test]
    fn schema_generates() {
        let schema = Config::json_schema();
        let text = schema.to_string();
        assert!(text.contains("ssim_threshold"));
        assert!(text.contains("prototype_compat"));
        assert!(text.contains("additionalProperties"));
    }
}
