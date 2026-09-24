//! Run manifest (`run.lock.json`) and the run directory lock.
//!
//! [`RunDir::open`] takes an exclusive `fs4` lock on `<run>/.glassrip.lock` and fails
//! with [`ManifestError::AlreadyLocked`] if another process (or another handle in this
//! process) holds it. The lock is released when the [`RunDir`] is dropped. Every
//! manifest update is written atomically.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};

use crate::atomic::{self, AtomicWriteError};
use crate::envelope::{InputRef, Producer};

/// Manifest file name inside a run directory.
pub const MANIFEST_FILE: &str = "run.lock.json";
/// Lock file name inside a run directory.
pub const LOCK_FILE: &str = ".glassrip.lock";
/// Directory holding artifacts inside a run directory.
pub const ARTIFACTS_DIR: &str = "artifacts";
/// Directory (inside `artifacts/`) holding resumable partial outputs.
pub const PARTIAL_DIR: &str = ".partial";
/// Schema name of the manifest.
pub const MANIFEST_SCHEMA: &str = "glassrip.run_manifest";
/// Current manifest schema major version.
pub const MANIFEST_MAJOR: u64 = 1;

/// Status of a stage in a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    /// Not started.
    Pending,
    /// In progress (a crash leaves this behind).
    Running,
    /// Completed by running.
    Ok,
    /// Output restored from cache.
    Cached,
    /// Not selected for this run.
    Skipped,
    /// Error rate over threshold or a stage-level error.
    Failed,
    /// Stopped by cancellation.
    Cancelled,
    /// Was `running` when the previous process stopped (crash or kill); set when a
    /// run directory is reopened.
    Interrupted,
}

/// Per-stage manifest entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StageRecord {
    /// Status.
    pub status: StageStatus,
    /// Stage implementation version.
    pub stage_version: u32,
    /// Cache key of the stage's output.
    pub cache_key: Option<String>,
    /// Start time (Unix seconds).
    pub started_unix_s: Option<f64>,
    /// End time (Unix seconds).
    pub finished_unix_s: Option<f64>,
    /// Wall time in seconds.
    pub wall_s: Option<f64>,
    /// Items planned.
    pub items_total: u64,
    /// Items with status ok.
    pub items_ok: u64,
    /// Items with status error.
    pub items_error: u64,
    /// Items with status skipped.
    pub items_skipped: u64,
    /// Stage-level error message.
    pub error: Option<String>,
    /// Content hash of the stage's output artifact.
    #[serde(default)]
    pub content_hash: Option<String>,
}

impl StageRecord {
    /// A fresh record with the given status.
    pub fn new(status: StageStatus, stage_version: u32) -> Self {
        Self {
            status,
            stage_version,
            cache_key: None,
            started_unix_s: None,
            finished_unix_s: None,
            wall_s: None,
            items_total: 0,
            items_ok: 0,
            items_error: 0,
            items_skipped: 0,
            error: None,
            content_hash: None,
        }
    }
}

/// One external command a stage ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CommandRecord {
    /// Stage that ran it.
    pub stage: String,
    /// Item it ran for, if any.
    pub item_id: Option<String>,
    /// Full argv.
    pub argv: Vec<String>,
    /// Exit code, if it exited normally.
    pub exit_code: Option<i32>,
    /// Wall time in seconds.
    pub wall_s: Option<f64>,
}

/// The `run.lock.json` manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    /// Always [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// Manifest schema version.
    pub schema_version: Version,
    /// Run identifier.
    pub run_id: String,
    /// Producing tool (includes the git sha).
    pub producer: Producer,
    /// Run inputs (for example the source video).
    pub inputs: Vec<InputRef>,
    /// Tool versions (ffmpeg, ffprobe, Ollama server, glassrip, Cargo.lock hash).
    pub tool_versions: BTreeMap<String, String>,
    /// Model name to digest.
    pub model_digests: BTreeMap<String, String>,
    /// External commands run, in order.
    pub commands: Vec<CommandRecord>,
    /// Per-stage status and timings.
    pub stages: BTreeMap<String, StageRecord>,
    /// Creation time (Unix seconds).
    pub created_unix_s: f64,
    /// Last update time (Unix seconds).
    pub updated_unix_s: f64,
}

impl RunManifest {
    /// A new, empty manifest.
    pub fn new(run_id: impl Into<String>, producer: Producer) -> Self {
        let now = unix_now();
        Self {
            schema: MANIFEST_SCHEMA.to_string(),
            schema_version: Version::new(MANIFEST_MAJOR, 0, 0),
            run_id: run_id.into(),
            producer,
            inputs: Vec::new(),
            tool_versions: BTreeMap::new(),
            model_digests: BTreeMap::new(),
            commands: Vec::new(),
            stages: BTreeMap::new(),
            created_unix_s: now,
            updated_unix_s: now,
        }
    }
}

/// Current time as Unix seconds.
pub fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Deserialize)]
struct ManifestProbe {
    schema: String,
    schema_version: Version,
}

/// Error from the manifest or lock.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// Another process holds the run directory lock.
    #[error("run directory {0} is locked by another glassrip process")]
    AlreadyLocked(PathBuf),
    /// Filesystem failure.
    #[error("run directory I/O on {path}: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// Atomic write failed.
    #[error(transparent)]
    Atomic(#[from] AtomicWriteError),
    /// The existing manifest could not be parsed.
    #[error("cannot parse {path}: {source}")]
    Parse {
        /// Manifest path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: serde_json::Error,
    },
    /// The existing manifest has a schema this build does not understand.
    #[error("{path}: expected {MANIFEST_SCHEMA} major {MANIFEST_MAJOR}, found {schema} {version}")]
    Schema {
        /// Manifest path.
        path: PathBuf,
        /// Schema found.
        schema: String,
        /// Version found.
        version: Version,
    },
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> ManifestError + '_ {
    move |source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// A locked run directory and its manifest.
#[derive(Debug)]
pub struct RunDir {
    root: PathBuf,
    manifest: RunManifest,
    _lock: fs_err::File,
}

impl RunDir {
    /// Creates (if needed) and locks a run directory.
    ///
    /// If `run.lock.json` exists it is loaded (resume): its stage records and commands
    /// are kept, and `run_id` and `producer` are replaced by the given values.
    pub fn open(root: &Path, run_id: &str, producer: Producer) -> Result<Self, ManifestError> {
        fs_err::create_dir_all(root.join(ARTIFACTS_DIR)).map_err(io_err(root))?;
        let lock_path = root.join(LOCK_FILE);
        let lock = fs_err::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(io_err(&lock_path))?;
        match fs4::fs_err3::FileExt::try_lock(&lock) {
            Ok(()) => {}
            Err(fs4::TryLockError::WouldBlock) => {
                return Err(ManifestError::AlreadyLocked(root.to_path_buf()));
            }
            Err(fs4::TryLockError::Error(e)) => return Err(io_err(&lock_path)(e)),
        }

        let manifest_path = root.join(MANIFEST_FILE);
        let existing = atomic::is_file(&manifest_path).map_err(io_err(&manifest_path))?;
        let manifest = if existing {
            let bytes = fs_err::read(&manifest_path).map_err(io_err(&manifest_path))?;
            // Phase 1: identify the schema loosely, so a foreign version is reported
            // as a schema mismatch rather than a field-level parse error.
            let probe: ManifestProbe =
                serde_json::from_slice(&bytes).map_err(|source| ManifestError::Parse {
                    path: manifest_path.clone(),
                    source,
                })?;
            if probe.schema != MANIFEST_SCHEMA || probe.schema_version.major != MANIFEST_MAJOR {
                return Err(ManifestError::Schema {
                    path: manifest_path,
                    schema: probe.schema,
                    version: probe.schema_version,
                });
            }
            // Phase 2: strict parse of the current version.
            let mut m: RunManifest =
                serde_json::from_slice(&bytes).map_err(|source| ManifestError::Parse {
                    path: manifest_path.clone(),
                    source,
                })?;
            m.run_id = run_id.to_string();
            m.producer = producer;
            for (name, record) in &mut m.stages {
                if record.status == StageStatus::Running {
                    tracing::warn!(stage = %name, "stage was running when the previous process stopped");
                    record.status = StageStatus::Interrupted;
                    record.error =
                        Some("interrupted: the previous process stopped mid-stage".into());
                }
            }
            m
        } else {
            RunManifest::new(run_id, producer)
        };
        let mut dir = Self {
            root: root.to_path_buf(),
            manifest,
            _lock: lock,
        };
        dir.save()?;
        Ok(dir)
    }

    /// Run directory root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Current manifest.
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }

    /// Path of an artifact by schema name: `<run>/artifacts/<schema>.jsonl`.
    pub fn artifact_path(&self, schema: &str) -> PathBuf {
        self.root
            .join(ARTIFACTS_DIR)
            .join(format!("{schema}.jsonl"))
    }

    /// Removes resumable partial outputs (`artifacts/.partial/*`) last modified more
    /// than `older_than` before `now`, plus orphaned atomic-write temp files older
    /// than the cache grace window. Safe because the caller holds the run lock.
    pub fn gc_partials(
        &self,
        older_than: std::time::Duration,
        now: SystemTime,
    ) -> Result<Vec<PathBuf>, ManifestError> {
        let before = |age| now.checked_sub(age).unwrap_or(UNIX_EPOCH);
        let artifacts = self.root.join(ARTIFACTS_DIR);
        let partials = artifacts.join(PARTIAL_DIR);
        let mut candidates = Vec::new();
        for (dir, only_temp) in [(&self.root, true), (&artifacts, true), (&partials, false)] {
            if !atomic::is_dir(dir).map_err(io_err(dir))? {
                continue;
            }
            for entry in fs_err::read_dir(dir).map_err(io_err(dir))? {
                let path = entry.map_err(io_err(dir))?.path();
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let is_temp = name.starts_with(".glassrip-tmp-");
                if only_temp && !is_temp {
                    continue;
                }
                let cutoff = if is_temp {
                    before(crate::cache::DEFAULT_GC_GRACE)
                } else {
                    before(older_than)
                };
                candidates.push((path, cutoff));
            }
        }
        let mut removed = Vec::new();
        for (path, cutoff) in candidates {
            let meta = fs_err::symlink_metadata(&path).map_err(io_err(&path))?;
            if meta.is_file() && meta.modified().map_err(io_err(&path))? < cutoff {
                fs_err::remove_file(&path).map_err(io_err(&path))?;
                removed.push(path);
            }
        }
        removed.sort();
        Ok(removed)
    }

    /// Directory holding resumable partial outputs.
    pub fn partials_dir(&self) -> PathBuf {
        self.root.join(ARTIFACTS_DIR).join(PARTIAL_DIR)
    }

    /// Path of an artifact relative to the run root (as recorded in `inputs`).
    pub fn artifact_rel_path(schema: &str) -> String {
        format!("{ARTIFACTS_DIR}/{schema}.jsonl")
    }

    /// Mutates the manifest and saves it atomically.
    pub fn update<F: FnOnce(&mut RunManifest)>(&mut self, f: F) -> Result<(), ManifestError> {
        f(&mut self.manifest);
        self.save()
    }

    /// Saves the manifest atomically.
    pub fn save(&mut self) -> Result<(), ManifestError> {
        self.manifest.updated_unix_s = unix_now();
        let mut bytes =
            serde_json::to_vec_pretty(&self.manifest).map_err(|source| ManifestError::Parse {
                path: self.root.join(MANIFEST_FILE),
                source,
            })?;
        bytes.push(b'\n');
        atomic::write_atomic(&self.root.join(MANIFEST_FILE), &bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn producer() -> Producer {
        Producer::glassrip("0.1.0", Some("deadbeef".into()))
    }

    #[test]
    fn lock_contention() {
        let dir = tempfile::tempdir().unwrap();
        let first = RunDir::open(dir.path(), "r1", producer()).unwrap();
        match RunDir::open(dir.path(), "r2", producer()) {
            Err(ManifestError::AlreadyLocked(p)) => assert_eq!(p, dir.path()),
            other => panic!("expected AlreadyLocked, got {other:?}"),
        }
        drop(first);
        RunDir::open(dir.path(), "r3", producer()).unwrap();
    }

    #[test]
    fn manifest_updates_persist_and_resume() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut run = RunDir::open(dir.path(), "r1", producer()).unwrap();
            run.update(|m| {
                m.tool_versions.insert("ffmpeg".into(), "7.1".into());
                m.model_digests
                    .insert("vision".into(), "sha256:0001".into());
                m.commands.push(CommandRecord {
                    stage: "probe".into(),
                    item_id: None,
                    argv: vec!["ffprobe".into(), "-print_format".into(), "json".into()],
                    exit_code: Some(0),
                    wall_s: Some(0.1),
                });
                m.stages
                    .insert("probe".into(), StageRecord::new(StageStatus::Ok, 1));
            })
            .unwrap();
        }
        let text = fs_err::read_to_string(dir.path().join(MANIFEST_FILE)).unwrap();
        let on_disk: RunManifest = serde_json::from_str(&text).unwrap();
        assert_eq!(on_disk.stages["probe"].status, StageStatus::Ok);
        assert_eq!(on_disk.producer.git_sha.as_deref(), Some("deadbeef"));

        let run = RunDir::open(dir.path(), "r2", producer()).unwrap();
        assert_eq!(run.manifest().run_id, "r2");
        assert_eq!(run.manifest().commands.len(), 1);
        assert_eq!(run.manifest().stages["probe"].status, StageStatus::Ok);
        assert_eq!(
            run.artifact_path("glassrip.frames"),
            dir.path().join("artifacts/glassrip.frames.jsonl")
        );
    }

    #[test]
    fn foreign_manifest_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = RunManifest::new("r0", producer());
        m.schema_version = Version::new(2, 0, 0);
        fs_err::write(
            dir.path().join(MANIFEST_FILE),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            RunDir::open(dir.path(), "r1", producer()),
            Err(ManifestError::Schema { .. })
        ));
    }

    #[test]
    fn foreign_version_with_new_fields_reports_schema() {
        let dir = tempfile::tempdir().unwrap();
        let mut v = serde_json::to_value(RunManifest::new("r0", producer())).unwrap();
        v["schema_version"] = serde_json::json!("2.0.0");
        v["field_from_the_future"] = serde_json::json!(true);
        v.as_object_mut().unwrap().remove("commands");
        fs_err::write(
            dir.path().join(MANIFEST_FILE),
            serde_json::to_vec(&v).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            RunDir::open(dir.path(), "r1", producer()),
            Err(ManifestError::Schema { .. })
        ));
    }

    #[test]
    fn running_stage_becomes_interrupted_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut run = RunDir::open(dir.path(), "r1", producer()).unwrap();
            run.update(|m| {
                m.stages
                    .insert("frames".into(), StageRecord::new(StageStatus::Running, 1));
                m.stages
                    .insert("probe".into(), StageRecord::new(StageStatus::Ok, 1));
            })
            .unwrap();
            // Dropped without finishing, as after a crash.
        }
        let run = RunDir::open(dir.path(), "r2", producer()).unwrap();
        assert_eq!(
            run.manifest().stages["frames"].status,
            StageStatus::Interrupted
        );
        assert_eq!(run.manifest().stages["probe"].status, StageStatus::Ok);
        let on_disk: RunManifest =
            serde_json::from_slice(&fs_err::read(dir.path().join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(on_disk.stages["frames"].status, StageStatus::Interrupted);
    }

    #[test]
    fn gc_partials_by_age() {
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::open(dir.path(), "r1", producer()).unwrap();
        fs_err::create_dir_all(run.partials_dir()).unwrap();
        let old = run.partials_dir().join("frames-0123.jsonl");
        let new = run.partials_dir().join("frames-4567.jsonl");
        let old_tmp = dir.path().join("artifacts/.glassrip-tmp-abc");
        let keep = dir.path().join("artifacts/glassrip.frames.jsonl");
        for p in [&old, &new, &old_tmp, &keep] {
            fs_err::write(p, b"x").unwrap();
        }
        let past = SystemTime::now() - std::time::Duration::from_secs(40 * 86_400);
        for p in [&old, &old_tmp, &keep] {
            fs_err::File::options()
                .write(true)
                .open(p)
                .unwrap()
                .file()
                .set_modified(past)
                .unwrap();
        }
        let removed = run
            .gc_partials(
                std::time::Duration::from_secs(30 * 86_400),
                SystemTime::now(),
            )
            .unwrap();
        assert_eq!(removed, {
            let mut v = vec![old.clone(), old_tmp.clone()];
            v.sort();
            v
        });
        assert!(new.exists() && keep.exists());
    }

    #[test]
    fn schema_generates() {
        let _ = schemars::schema_for!(RunManifest);
    }
}
