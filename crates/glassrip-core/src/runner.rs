//! The [`Stage`] trait and the stage runner.
//!
//! A stage is a pure function from input artifacts and params to one output
//! artifact. It is split into [`Stage::plan`] (read inputs, list work items) and
//! [`Stage::process`] (one item). The [`Runner`]:
//!
//! 1. applies the [`Plan`] (`--from-stage`, `--until-stage`, `--force-stage`),
//! 2. loads and schema-checks the declared input artifacts,
//! 3. computes the cache key and restores the output from cache when possible,
//! 4. otherwise processes items with per-item failure isolation, a per-item timeout,
//!    and cancellation, appending each result to a crash-safe partial JSONL file,
//! 5. fails the stage when the item error rate exceeds the threshold, and
//! 6. writes the final artifact atomically, caches it when no item failed, and
//!    records status, timings, and external commands in `run.lock.json`.
//!
//! A rerun after a crash or cancellation resumes from the partial file: items that
//! already succeeded are not processed again, and failed items are retried.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use schemars::JsonSchema;
use semver::Version;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, info, info_span, warn};

use crate::atomic::{self, AtomicWriteError};
use crate::cache::{Cache, CacheError, CacheKey, CacheKeyParts, CacheLease};
use crate::canonical::{self, CanonicalJsonError};
use crate::config::RunnerConfig;
use crate::envelope::{
    EnvelopeHeader, ErrorCode, ErrorInfo, InputRef, Keyed, Outcome, Record, RestoredFrom,
    SchemaReq, Status, content_hash,
};
use crate::graph::{GraphError, Plan, Selection, StageDecision, StageDecl, StageGraph};
use crate::jsonl::{self, JsonlError, JsonlWriter};
use crate::manifest::{CommandRecord, ManifestError, RunDir, StageRecord, StageStatus, unix_now};

/// Cache file extension for stage outputs.
pub const OUTPUT_EXT: &str = "jsonl";

/// The artifact a stage produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactSpec {
    /// Schema name, for example `glassrip.frames`.
    pub schema: &'static str,
    /// Schema version written into the envelope.
    pub version: Version,
}

/// An artifact a stage consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputDecl {
    /// Schema name.
    pub schema: &'static str,
    /// Required major version.
    pub major: u64,
}

/// Extra cache key components a stage may contribute.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyExtras {
    /// Model digest (model stages).
    pub model_digest: Option<String>,
    /// Hash of the prompt text (model stages).
    pub prompt_hash: Option<String>,
    /// Decoder (frame stages).
    pub decoder: Option<String>,
    /// Features mode (feature stages).
    pub features_mode: Option<String>,
    /// Stage-specific tool versions (ffmpeg, Ollama server, ...), merged over the
    /// runner's.
    pub tool_versions: BTreeMap<String, String>,
}

/// One unit of work planned by a stage.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkItem<W> {
    /// Stable item id; becomes the output record id.
    pub id: String,
    /// Stage-specific payload.
    pub work: W,
}

/// Error from [`Stage::plan`].
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    /// Reading an input artifact failed.
    #[error(transparent)]
    Input(#[from] JsonlError),
    /// A declared input was not loaded (a bug in the stage's declarations).
    #[error("input artifact `{0}` was not declared by this stage")]
    Undeclared(String),
    /// The inputs are unusable.
    #[error("{0}")]
    Invalid(String),
}

/// A pipeline stage.
pub trait Stage: Send + Sync {
    /// Parameters; serialized into the envelope and the cache key.
    type Params: Serialize + JsonSchema + Send + Sync;
    /// Per-item work payload.
    type Work: Send;
    /// Per-item result type.
    type Output: Serialize + DeserializeOwned + JsonSchema + Send + Sync;

    /// Stage name (also the cache subdirectory).
    fn name(&self) -> &'static str;
    /// Implementation version; bump when output would change for the same inputs.
    fn version(&self) -> u32;
    /// Output artifact.
    fn output(&self) -> ArtifactSpec;
    /// Input artifacts.
    fn inputs(&self) -> Vec<InputDecl>;
    /// Parameters.
    fn params(&self) -> &Self::Params;
    /// Files outside the run directory this stage reads (for example the source
    /// video); hashed into the key and recorded in the envelope.
    fn external_inputs(&self) -> Vec<PathBuf> {
        Vec::new()
    }
    /// Extra cache key components.
    fn key_extras(&self) -> KeyExtras {
        KeyExtras::default()
    }
    /// Items processed concurrently.
    fn concurrency(&self) -> usize {
        1
    }
    /// Per-item timeout; overrides the runner default when set.
    fn item_timeout(&self) -> Option<Duration> {
        None
    }
    /// Reads inputs and lists the work items.
    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<Self::Work>>, StageError>;
    /// Processes one item. An `Err` is recorded as an error item; it does not stop
    /// the stage unless the error rate passes the threshold.
    fn process(
        &self,
        ctx: &ItemContext,
        work: Self::Work,
    ) -> impl Future<Output = Result<Self::Output, ErrorInfo>> + Send;
}

/// The graph declaration of a stage.
pub fn stage_decl<S: Stage>(stage: &S) -> StageDecl {
    StageDecl {
        name: stage.name().to_string(),
        output: stage.output().schema.to_string(),
        inputs: stage
            .inputs()
            .iter()
            .map(|i| i.schema.to_string())
            .collect(),
    }
}

#[derive(Debug, Clone)]
struct LoadedInput {
    path: PathBuf,
    header: EnvelopeHeader,
    req: SchemaReq,
}

/// Input artifacts available to [`Stage::plan`].
#[derive(Debug, Clone, Default)]
pub struct StageInputs {
    artifacts: BTreeMap<String, LoadedInput>,
}

impl StageInputs {
    fn get(&self, schema: &str) -> Result<&LoadedInput, StageError> {
        self.artifacts
            .get(schema)
            .ok_or_else(|| StageError::Undeclared(schema.to_string()))
    }

    /// Header of an input artifact.
    pub fn header(&self, schema: &str) -> Result<&EnvelopeHeader, StageError> {
        Ok(&self.get(schema)?.header)
    }

    /// Path of an input artifact.
    pub fn path(&self, schema: &str) -> Result<&Path, StageError> {
        Ok(&self.get(schema)?.path)
    }

    /// Reads an input artifact's items (deduplicated by id, last write wins).
    pub fn read<T: DeserializeOwned + Keyed>(&self, schema: &str) -> Result<Vec<T>, StageError> {
        let input = self.get(schema)?;
        Ok(jsonl::read::<T>(&input.path, &input.req)?.items)
    }

    /// Reads a stage-output artifact and returns `(id, result)` for items with
    /// status `ok`.
    pub fn read_ok<T: DeserializeOwned>(
        &self,
        schema: &str,
    ) -> Result<Vec<(String, T)>, StageError> {
        Ok(self
            .read::<Record<T>>(schema)?
            .into_iter()
            .filter_map(|r| r.outcome.result.map(|v| (r.id, v)))
            .collect())
    }
}

/// Per-item context handed to [`Stage::process`].
#[derive(Debug, Clone)]
pub struct ItemContext {
    stage: &'static str,
    item_id: String,
    cancel: CancellationToken,
    commands: Arc<Mutex<Vec<CommandRecord>>>,
}

impl ItemContext {
    /// Stage name.
    pub fn stage(&self) -> &'static str {
        self.stage
    }

    /// Item id.
    pub fn item_id(&self) -> &str {
        &self.item_id
    }

    /// The run's cancellation token (for long operations that can stop early).
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel
    }

    /// Records an external command in the run manifest.
    pub fn record_command(&self, argv: Vec<String>, exit_code: Option<i32>, wall_s: Option<f64>) {
        self.commands
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(CommandRecord {
                stage: self.stage.to_string(),
                item_id: Some(self.item_id.clone()),
                argv,
                exit_code,
                wall_s,
            });
    }
}

/// Runner settings.
#[derive(Debug, Clone, PartialEq)]
pub struct RunnerOptions {
    /// A stage fails when `errors / items` exceeds this.
    pub max_item_error_rate: f64,
    /// Default per-item timeout.
    pub default_item_timeout: Option<Duration>,
    /// Appends between JSONL `sync_data` checkpoints.
    pub checkpoint_every: usize,
    /// Tool versions included in every cache key (glassrip version, Cargo.lock hash).
    pub tool_versions: BTreeMap<String, String>,
}

impl Default for RunnerOptions {
    fn default() -> Self {
        Self::from_config(&RunnerConfig::default())
    }
}

impl RunnerOptions {
    /// Options from the `[runner]` config section.
    pub fn from_config(cfg: &RunnerConfig) -> Self {
        Self {
            max_item_error_rate: cfg.max_item_error_rate,
            default_item_timeout: cfg
                .item_timeout_s
                .and_then(|s| Duration::try_from_secs_f64(s).ok()),
            checkpoint_every: usize::try_from(cfg.checkpoint_every).unwrap_or(usize::MAX),
            tool_versions: BTreeMap::new(),
        }
    }
}

/// What happened to one stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageReport {
    /// Stage name.
    pub stage: String,
    /// Final status (`ok`, `cached`, or `skipped`; failures are errors).
    pub status: StageStatus,
    /// Cache key, when computed.
    pub cache_key: Option<CacheKey>,
    /// Content hash of the output artifact.
    pub content_hash: Option<String>,
    /// Items planned.
    pub items_total: u64,
    /// Items ok.
    pub items_ok: u64,
    /// Items with errors.
    pub items_error: u64,
    /// Items skipped.
    pub items_skipped: u64,
    /// Items processed in this call (excludes resumed and cached items).
    pub items_processed: u64,
    /// Wall time.
    pub wall_s: f64,
    /// Output artifact path, when produced.
    pub output: Option<PathBuf>,
}

/// Runner error.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// Graph or selection problem.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// The stage is not part of the runner's graph.
    #[error("stage `{0}` is not in the stage graph")]
    StageNotInGraph(String),
    /// The stage's declarations differ from its graph entry.
    #[error(
        "stage `{stage}` declarations differ from the graph: graph has {expected:?}, stage has {found:?}"
    )]
    DeclMismatch {
        /// Stage.
        stage: String,
        /// Graph entry.
        expected: Box<StageDecl>,
        /// Stage's own declaration.
        found: Box<StageDecl>,
    },
    /// A declared input artifact does not exist in the run directory.
    #[error(
        "stage `{stage}` needs {schema} at {path}, which does not exist (run the producing stage first)"
    )]
    MissingInputArtifact {
        /// Stage.
        stage: String,
        /// Missing schema.
        schema: String,
        /// Expected path.
        path: PathBuf,
    },
    /// An input artifact failed to load or failed its schema check.
    #[error("stage `{stage}`: input check failed: {source}")]
    Input {
        /// Stage.
        stage: String,
        /// Cause.
        #[source]
        source: JsonlError,
    },
    /// An input could not be hashed.
    #[error("cannot hash input {path}: {source}")]
    HashInput {
        /// File.
        path: PathBuf,
        /// Cause.
        #[source]
        source: io::Error,
    },
    /// Params are not a JSON object.
    #[error("stage `{stage}`: params must serialize to a JSON object")]
    ParamsNotObject {
        /// Stage.
        stage: String,
    },
    /// Params or schema could not be serialized.
    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
    /// Canonical JSON failure.
    #[error(transparent)]
    Canonical(#[from] CanonicalJsonError),
    /// Cache failure.
    #[error(transparent)]
    Cache(#[from] CacheError),
    /// JSONL failure.
    #[error(transparent)]
    Jsonl(#[from] JsonlError),
    /// Manifest failure.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// Atomic write failure.
    #[error(transparent)]
    Atomic(#[from] AtomicWriteError),
    /// Filesystem failure.
    #[error("runner I/O on {path}: {source}")]
    Io {
        /// Path.
        path: PathBuf,
        /// Cause.
        #[source]
        source: io::Error,
    },
    /// Planning failed.
    #[error("stage `{stage}` could not plan its work: {source}")]
    Plan {
        /// Stage.
        stage: String,
        /// Cause.
        #[source]
        source: StageError,
    },
    /// Two work items share an id.
    #[error("stage `{stage}` planned duplicate item id `{id}`")]
    DuplicateItemId {
        /// Stage.
        stage: String,
        /// Id.
        id: String,
    },
    /// Too many items failed.
    #[error("stage `{stage}` failed: {errors} of {total} items failed (threshold {threshold})")]
    ErrorRateExceeded {
        /// Stage.
        stage: String,
        /// Failed items.
        errors: u64,
        /// Planned items.
        total: u64,
        /// Allowed error rate.
        threshold: f64,
    },
    /// The run was cancelled.
    #[error("stage `{stage}` cancelled; rerun to resume")]
    Cancelled {
        /// Stage.
        stage: String,
    },
}

/// Runs stages against a locked run directory.
///
/// Holds a shared [`CacheLease`] for its lifetime, so `cache gc` cannot delete
/// entries while a run is using them.
///
/// [`Runner::run_stage_shared`] takes `&self`, so independent stages can run
/// concurrently on one runner (for example the audio branch next to board
/// reading); the manifest is updated under a lock that is never held across an
/// await.
#[derive(Debug)]
pub struct Runner {
    run: Mutex<RunDir>,
    graph: StageGraph,
    plan: Plan,
    cache: Cache,
    _cache_lease: CacheLease,
    opts: RunnerOptions,
    cancel: CancellationToken,
    commands: Arc<Mutex<Vec<CommandRecord>>>,
}

#[derive(Debug, Default, Clone, Copy)]
struct Counts {
    ok: u64,
    error: u64,
    skipped: u64,
}

impl Counts {
    fn add(&mut self, status: Status) {
        match status {
            Status::Ok => self.ok += 1,
            Status::Error => self.error += 1,
            Status::Skipped => self.skipped += 1,
        }
    }

    fn of<T>(records: &[Record<T>]) -> Self {
        let mut c = Self::default();
        for r in records {
            c.add(r.outcome.status);
        }
        c
    }
}

/// A verified cache entry ready to be restored.
struct CachedArtifact {
    header: EnvelopeHeader,
    counts: Counts,
    total: u64,
    content_hash: String,
    /// Bytes after the header line (the item lines), copied verbatim.
    body: Vec<u8>,
}

async fn run_with_timeout<T, F>(fut: F, timeout: Option<Duration>) -> Outcome<T>
where
    F: Future<Output = Result<T, ErrorInfo>>,
{
    let result = match timeout {
        Some(limit) => match tokio::time::timeout(limit, fut).await {
            Ok(r) => r,
            Err(_) => Err(ErrorInfo::new(
                ErrorCode::Timeout,
                format!("item exceeded its {:.3} s timeout", limit.as_secs_f64()),
            )),
        },
        None => fut.await,
    };
    match result {
        Ok(v) => Outcome::ok(v),
        Err(e) => Outcome::error(e),
    }
}

impl Runner {
    /// Builds a runner for a validated graph and selection, taking a shared lease on
    /// the cache.
    pub fn new(
        run: RunDir,
        graph: StageGraph,
        selection: &Selection,
        cache: Cache,
        opts: RunnerOptions,
        cancel: CancellationToken,
    ) -> Result<Self, RunnerError> {
        let plan = graph.plan(selection)?;
        let lease = cache.lease()?;
        Ok(Self {
            run: Mutex::new(run),
            graph,
            plan,
            cache,
            _cache_lease: lease,
            opts,
            cancel,
            commands: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// The run directory. Do not hold the guard across a call that runs a stage
    /// (the stage updates the manifest through the same lock).
    pub fn run_dir(&self) -> MutexGuard<'_, RunDir> {
        self.run.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Mutable access to the run directory (for recording run-level manifest data).
    pub fn run_dir_mut(&mut self) -> &mut RunDir {
        self.run.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    /// Releases the runner and its cache lease, returning the (still locked) run
    /// directory.
    pub fn into_run_dir(self) -> RunDir {
        self.run
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `f` on the run directory under the lock (never across an await).
    fn with_run<R>(&self, f: impl FnOnce(&mut RunDir) -> R) -> R {
        let mut guard = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut guard)
    }

    /// The resolved stage plan.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The cancellation token.
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel
    }

    /// Runs (or restores, or skips) one stage.
    pub async fn run_stage<S: Stage>(&mut self, stage: &S) -> Result<StageReport, RunnerError> {
        self.run_stage_shared(stage).await
    }

    /// [`Runner::run_stage`] through a shared reference, so stages that do not
    /// depend on each other can run concurrently (join their futures). The
    /// caller is responsible for running a stage only after its inputs exist.
    pub async fn run_stage_shared<S: Stage>(&self, stage: &S) -> Result<StageReport, RunnerError> {
        let name = stage.name();
        let found = stage_decl(stage);
        match self.graph.decl(name) {
            None => return Err(RunnerError::StageNotInGraph(name.to_string())),
            Some(expected) if *expected != found => {
                return Err(RunnerError::DeclMismatch {
                    stage: name.to_string(),
                    expected: Box::new(expected.clone()),
                    found: Box::new(found),
                });
            }
            Some(_) => {}
        }
        let span = info_span!("stage", stage = name, version = stage.version());
        let started = Instant::now();
        let result = self.run_stage_inner(stage, started).instrument(span).await;

        // Commands recorded by items are kept whatever the stage outcome.
        let new_commands =
            std::mem::take(&mut *self.commands.lock().unwrap_or_else(PoisonError::into_inner));
        if !new_commands.is_empty() {
            if let Err(e) = self.with_run(|r| r.update(|m| m.commands.extend(new_commands))) {
                warn!(stage = name, error = %e, "could not record commands in manifest");
            }
        }

        if let Err(err) = &result {
            let status = match err {
                RunnerError::Cancelled { .. } => StageStatus::Cancelled,
                _ => StageStatus::Failed,
            };
            let message = err.to_string();
            let wall = started.elapsed().as_secs_f64();
            // Best effort: the original error is more useful than a manifest error.
            if let Err(manifest_err) = self.set_stage(name, stage.version(), |r| {
                r.status = status;
                r.error = Some(message);
                r.finished_unix_s = Some(unix_now());
                r.wall_s = Some(wall);
            }) {
                warn!(stage = name, error = %manifest_err, "could not record stage failure in manifest");
            }
        }
        result
    }

    fn set_stage<F: FnOnce(&mut StageRecord)>(
        &self,
        name: &str,
        version: u32,
        f: F,
    ) -> Result<(), ManifestError> {
        self.with_run(|run| {
            run.update(|m| {
                let record = m
                    .stages
                    .entry(name.to_string())
                    .or_insert_with(|| StageRecord::new(StageStatus::Pending, version));
                record.stage_version = version;
                f(record);
            })
        })
    }

    fn io_err(path: &Path) -> impl FnOnce(io::Error) -> RunnerError + '_ {
        move |source| RunnerError::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    /// Reads and verifies a cache entry. Any problem is returned as a message so the
    /// caller can treat it as a miss.
    fn load_cached<S: Stage>(
        &self,
        path: &Path,
        req: &SchemaReq,
    ) -> Result<CachedArtifact, String> {
        let bytes = fs_err::read(path).map_err(|e| e.to_string())?;
        if !bytes.ends_with(b"\n") {
            return Err("entry ends in a partial line".into());
        }
        let (header, items) = jsonl::parse_bytes::<Record<S::Output>>(path, &bytes, req)
            .map_err(|e| e.to_string())?;
        let recorded = header
            .content_hash
            .clone()
            .ok_or_else(|| "entry header has no content_hash".to_string())?;
        let actual = content_hash(
            &header.schema,
            &header.schema_version,
            &header.params,
            &items,
        )
        .map_err(|e| e.to_string())?;
        if actual != recorded {
            return Err(format!(
                "content hash mismatch: header {recorded}, items {actual}"
            ));
        }
        let header_len = bytes
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        Ok(CachedArtifact {
            counts: Counts::of(&items),
            total: items.len() as u64,
            header,
            content_hash: recorded,
            body: bytes[header_len..].to_vec(),
        })
    }

    /// Looks up the cache. A missing entry is a miss; an unreadable or corrupt entry
    /// is quarantined and also treated as a miss.
    fn cache_lookup<S: Stage>(
        &self,
        name: &str,
        key: &CacheKey,
        req: &SchemaReq,
    ) -> Option<CachedArtifact> {
        let path = match self.cache.get_path(name, key, OUTPUT_EXT) {
            Ok(Some(p)) => p,
            Ok(None) => return None,
            Err(e) => {
                warn!(stage = name, key = %key, error = %e, "cache lookup failed; recomputing");
                return None;
            }
        };
        match self.load_cached::<S>(&path, req) {
            Ok(cached) => Some(cached),
            Err(reason) => {
                warn!(stage = name, key = %key, path = %path.display(), %reason, "corrupt cache entry; recomputing");
                match self.cache.quarantine(name, key, OUTPUT_EXT) {
                    Ok(Some(dest)) => {
                        warn!(stage = name, to = %dest.display(), "quarantined cache entry")
                    }
                    Ok(None) => {}
                    Err(e) => warn!(stage = name, error = %e, "could not quarantine cache entry"),
                }
                None
            }
        }
    }

    async fn run_stage_inner<S: Stage>(
        &self,
        stage: &S,
        started: Instant,
    ) -> Result<StageReport, RunnerError> {
        let name = stage.name();
        let version = stage.version();
        let mut report = StageReport {
            stage: name.to_string(),
            status: StageStatus::Skipped,
            cache_key: None,
            content_hash: None,
            items_total: 0,
            items_ok: 0,
            items_error: 0,
            items_skipped: 0,
            items_processed: 0,
            wall_s: 0.0,
            output: None,
        };

        let force = match self.plan.decision(name) {
            None => return Err(RunnerError::StageNotInGraph(name.to_string())),
            Some(StageDecision::Skip(reason)) => {
                info!(stage = name, ?reason, "stage skipped by selection");
                if !self.with_run(|r| r.manifest().stages.contains_key(name)) {
                    self.set_stage(name, version, |r| r.status = StageStatus::Skipped)?;
                }
                return Ok(report);
            }
            Some(StageDecision::Run { force }) => force,
        };
        if self.cancel.is_cancelled() {
            return Err(RunnerError::Cancelled {
                stage: name.to_string(),
            });
        }

        // Inputs: existence, schema check, content hashes.
        let mut artifacts = BTreeMap::new();
        let mut input_refs = Vec::new();
        let mut input_hashes = Vec::new();
        for decl in stage.inputs() {
            let path = self.with_run(|r| r.artifact_path(decl.schema));
            if !atomic::is_file(&path).map_err(Self::io_err(&path))? {
                return Err(RunnerError::MissingInputArtifact {
                    stage: name.to_string(),
                    schema: decl.schema.to_string(),
                    path,
                });
            }
            let req = SchemaReq::new(decl.schema, decl.major);
            let input_err = |source| RunnerError::Input {
                stage: name.to_string(),
                source,
            };
            let header = jsonl::read_header(&path, &req).map_err(input_err)?;
            // The content hash excludes run metadata, so an upstream rerun that
            // produces identical items leaves this stage's key unchanged.
            let hash = match &header.content_hash {
                Some(h) => h.clone(),
                None => jsonl::compute_content_hash(&path, &req).map_err(|source| {
                    RunnerError::Input {
                        stage: name.to_string(),
                        source,
                    }
                })?,
            };
            input_refs.push(InputRef {
                path: RunDir::artifact_rel_path(decl.schema),
                blake3: hash.clone(),
                schema: Some(decl.schema.to_string()),
                schema_version: Some(header.schema_version.clone()),
            });
            input_hashes.push(hash);
            artifacts.insert(decl.schema.to_string(), LoadedInput { path, header, req });
        }
        for path in stage.external_inputs() {
            let hash = crate::blake3_file(&path).map_err(|source| RunnerError::HashInput {
                path: path.clone(),
                source,
            })?;
            input_refs.push(InputRef {
                path: path.display().to_string(),
                blake3: hash.clone(),
                schema: None,
                schema_version: None,
            });
            input_hashes.push(hash);
        }

        // Cache key.
        let params = canonical::to_checked_value(stage.params())?;
        if !params.is_object() {
            return Err(RunnerError::ParamsNotObject {
                stage: name.to_string(),
            });
        }
        let output = stage.output();
        let schema_hash = canonical::canonical_hash(&json!({
            "schema": output.schema,
            "schema_version": output.version.to_string(),
            "json_schema": schemars::schema_for!(Record<S::Output>),
        }))?;
        let extras = stage.key_extras();
        let mut tool_versions = self.opts.tool_versions.clone();
        tool_versions.extend(extras.tool_versions);
        let key = CacheKeyParts {
            stage: name.to_string(),
            stage_version: version,
            params: params.clone(),
            input_hashes,
            model_digest: extras.model_digest,
            prompt_hash: extras.prompt_hash,
            schema_hash: Some(schema_hash),
            decoder: extras.decoder,
            features_mode: extras.features_mode,
            tool_versions,
        }
        .key()?;
        report.cache_key = Some(key.clone());
        let out_path = self.with_run(|r| r.artifact_path(output.schema));
        let (run_id, producer) =
            self.with_run(|r| (r.manifest().run_id.clone(), r.manifest().producer.clone()));
        let started_unix = unix_now();
        let key_text = key.to_string();
        self.set_stage(name, version, |r| {
            r.status = StageStatus::Running;
            r.cache_key = Some(key_text);
            r.content_hash = None;
            r.started_unix_s = Some(started_unix);
            r.finished_unix_s = None;
            r.wall_s = None;
            r.error = None;
        })?;

        let out_req = SchemaReq::new(output.schema, output.version.major);

        // Cache hit: restore with a header rewritten for this run; item lines are
        // copied byte for byte, so the content hash is unchanged.
        if !force {
            if let Some(cached) = self.cache_lookup::<S>(name, &key, &out_req) {
                let mut header = cached.header.clone();
                header.restored_from = Some(RestoredFrom {
                    run_id: std::mem::take(&mut header.run_id),
                    cache_key: key.to_string(),
                });
                header.run_id = run_id.clone();
                header.producer = producer.clone();
                let line = jsonl::header_line(&header)?;
                atomic::write_atomic_with(&out_path, |w| {
                    w.write_all(&line)?;
                    w.write_all(&cached.body)
                })?;
                let wall = started.elapsed().as_secs_f64();
                let (counts, total, hash) = (cached.counts, cached.total, cached.content_hash);
                let hash_text = hash.clone();
                self.set_stage(name, version, |r| {
                    r.status = StageStatus::Cached;
                    r.content_hash = Some(hash_text);
                    r.items_total = total;
                    r.items_ok = counts.ok;
                    r.items_error = counts.error;
                    r.items_skipped = counts.skipped;
                    r.finished_unix_s = Some(unix_now());
                    r.wall_s = Some(wall);
                })?;
                info!(stage = name, key = %key, "restored from cache");
                report.status = StageStatus::Cached;
                report.content_hash = Some(hash);
                report.items_total = total;
                report.items_ok = counts.ok;
                report.items_error = counts.error;
                report.items_skipped = counts.skipped;
                report.wall_s = wall;
                report.output = Some(out_path);
                return Ok(report);
            }
        }

        // Partial (resumable) store.
        let mut header = EnvelopeHeader {
            schema: output.schema.to_string(),
            schema_version: output.version.clone(),
            run_id,
            producer,
            inputs: input_refs,
            params,
            content_hash: None,
            restored_from: None,
        };
        let partial = self
            .with_run(|r| r.partials_dir())
            .join(format!("{name}-{}.jsonl", &key.as_str()[..16]));
        let partial_exists = atomic::metadata_opt(&partial)
            .map_err(Self::io_err(&partial))?
            .is_some();
        if force && partial_exists {
            fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
        }
        let (writer, resumed) = match JsonlWriter::<Record<S::Output>>::open_resume(
            &partial, &header,
        ) {
            Ok(pair) => pair,
            Err(
                JsonlError::HeaderMismatch(_)
                | JsonlError::BadLine { .. }
                | JsonlError::MissingHeader(_)
                | JsonlError::Header { .. },
            ) => {
                warn!(stage = name, path = %partial.display(), "discarding unusable partial output");
                fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
                JsonlWriter::open_resume(&partial, &header)?
            }
            Err(e) => return Err(e.into()),
        };
        let mut writer = writer.with_checkpoint_every(self.opts.checkpoint_every);
        if resumed.truncated_bytes > 0 {
            warn!(
                stage = name,
                bytes = resumed.truncated_bytes,
                "truncated a partial trailing line"
            );
        }

        // Plan.
        let inputs = StageInputs { artifacts };
        let work = stage.plan(&inputs).map_err(|source| RunnerError::Plan {
            stage: name.to_string(),
            source,
        })?;
        let mut order = Vec::with_capacity(work.len());
        let mut ids = HashSet::with_capacity(work.len());
        for w in &work {
            if !ids.insert(w.id.clone()) {
                return Err(RunnerError::DuplicateItemId {
                    stage: name.to_string(),
                    id: w.id.clone(),
                });
            }
            order.push(w.id.clone());
        }
        let mut done: HashMap<String, Record<S::Output>> = resumed
            .items
            .into_iter()
            .filter(|r| ids.contains(&r.id))
            .map(|r| (r.id.clone(), r))
            .collect();
        let todo: Vec<WorkItem<S::Work>> = work
            .into_iter()
            .filter(|w| done.get(&w.id).is_none_or(|r| r.outcome.is_error()))
            .collect();
        info!(
            stage = name,
            planned = order.len(),
            resumed = done.values().filter(|r| !r.outcome.is_error()).count(),
            to_process = todo.len(),
            "processing items"
        );

        // Process.
        let timeout = stage.item_timeout().or(self.opts.default_item_timeout);
        let cancel = self.cancel.clone();
        let commands = Arc::clone(&self.commands);
        let mut cancelled = false;
        let mut processed = 0u64;
        {
            let mut results = futures_util::stream::iter(todo.into_iter().map(|w| {
                let ctx = ItemContext {
                    stage: name,
                    item_id: w.id.clone(),
                    cancel: cancel.clone(),
                    commands: Arc::clone(&commands),
                };
                let cancel = cancel.clone();
                let id = w.id;
                async move {
                    if cancel.is_cancelled() {
                        return (id, None);
                    }
                    let span = info_span!("item", id = %id);
                    let fut = stage.process(&ctx, w.work).instrument(span);
                    let outcome = tokio::select! {
                        biased;
                        () = cancel.cancelled() => None,
                        outcome = run_with_timeout(fut, timeout) => Some(outcome),
                    };
                    (id, outcome)
                }
            }))
            .buffer_unordered(stage.concurrency().max(1));

            while let Some((id, outcome)) = results.next().await {
                let Some(outcome) = outcome else {
                    cancelled = true;
                    continue;
                };
                if let Some(e) = &outcome.error {
                    warn!(stage = name, item = %id, code = ?e.code, message = %e.message, "item failed");
                }
                let record = Record { id, outcome };
                writer.append(&record)?;
                processed += 1;
                done.insert(record.id.clone(), record);
            }
        }
        writer.checkpoint()?;

        let mut counts = Counts::default();
        for id in &order {
            if let Some(r) = done.get(id) {
                counts.add(r.outcome.status);
            }
        }
        let total = order.len() as u64;
        let wall = started.elapsed().as_secs_f64();
        self.set_stage(name, version, |r| {
            r.items_total = total;
            r.items_ok = counts.ok;
            r.items_error = counts.error;
            r.items_skipped = counts.skipped;
        })?;

        if cancelled || self.cancel.is_cancelled() {
            return Err(RunnerError::Cancelled {
                stage: name.to_string(),
            });
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = if total == 0 {
            0.0
        } else {
            counts.error as f64 / total as f64
        };
        if rate > self.opts.max_item_error_rate {
            return Err(RunnerError::ErrorRateExceeded {
                stage: name.to_string(),
                errors: counts.error,
                total,
                threshold: self.opts.max_item_error_rate,
            });
        }

        // Finalize.
        let items: Vec<Record<S::Output>> = order.iter().filter_map(|id| done.remove(id)).collect();
        let hash = content_hash(
            &header.schema,
            &header.schema_version,
            &header.params,
            &items,
        )?;
        header.content_hash = Some(hash.clone());
        jsonl::write_atomic(&out_path, &header, &items)?;
        drop(writer);
        if counts.error == 0 {
            // A cache that cannot be written (read-only, full) costs a future
            // recompute, not this run.
            if let Err(e) = self.cache.put_file(name, &key, OUTPUT_EXT, &out_path) {
                warn!(stage = name, error = %e, "could not store output in cache");
            }
            fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
        } else {
            info!(
                stage = name,
                errors = counts.error,
                "not caching output with failed items; rerun retries them"
            );
        }
        let hash_text = hash.clone();
        self.set_stage(name, version, |r| {
            r.status = StageStatus::Ok;
            r.content_hash = Some(hash_text);
            r.finished_unix_s = Some(unix_now());
            r.wall_s = Some(wall);
        })?;
        info!(
            stage = name,
            ok = counts.ok,
            errors = counts.error,
            wall_s = wall,
            "stage finished"
        );

        report.status = StageStatus::Ok;
        report.content_hash = Some(hash);
        report.items_total = total;
        report.items_ok = counts.ok;
        report.items_error = counts.error;
        report.items_skipped = counts.skipped;
        report.items_processed = processed;
        report.wall_s = wall;
        report.output = Some(out_path);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::Producer;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const SOURCE: &str = "glassrip.test_source";
    const DOUBLED: &str = "glassrip.test_doubled";

    #[derive(Debug, Clone, Serialize, JsonSchema)]
    struct SourceParams {
        n: usize,
        scale: f64,
    }

    #[derive(Default)]
    struct Behavior {
        fail: HashSet<usize>,
        slow: HashSet<usize>,
        cancel_at: Option<(usize, CancellationToken)>,
        offset: u64,
    }

    struct SourceStage {
        params: SourceParams,
        calls: Arc<AtomicUsize>,
        behavior: Arc<Mutex<Behavior>>,
        external: Vec<PathBuf>,
        timeout: Option<Duration>,
    }

    impl SourceStage {
        fn new(n: usize) -> Self {
            Self {
                params: SourceParams { n, scale: 1.0 },
                calls: Arc::new(AtomicUsize::new(0)),
                behavior: Arc::new(Mutex::new(Behavior::default())),
                external: Vec::new(),
                timeout: None,
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl Stage for SourceStage {
        type Params = SourceParams;
        type Work = usize;
        type Output = u64;

        fn name(&self) -> &'static str {
            "source"
        }
        fn version(&self) -> u32 {
            1
        }
        fn output(&self) -> ArtifactSpec {
            ArtifactSpec {
                schema: SOURCE,
                version: Version::new(1, 0, 0),
            }
        }
        fn inputs(&self) -> Vec<InputDecl> {
            Vec::new()
        }
        fn params(&self) -> &SourceParams {
            &self.params
        }
        fn external_inputs(&self) -> Vec<PathBuf> {
            self.external.clone()
        }
        fn item_timeout(&self) -> Option<Duration> {
            self.timeout
        }
        fn plan(&self, _inputs: &StageInputs) -> Result<Vec<WorkItem<usize>>, StageError> {
            Ok((0..self.params.n)
                .map(|i| WorkItem {
                    id: format!("item-{i:03}"),
                    work: i,
                })
                .collect())
        }
        async fn process(&self, ctx: &ItemContext, i: usize) -> Result<u64, ErrorInfo> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let (fail, slow, cancel, offset) = {
                let b = self.behavior.lock().unwrap();
                let cancel = b
                    .cancel_at
                    .as_ref()
                    .filter(|(at, _)| *at == i)
                    .map(|(_, t)| t.clone());
                (b.fail.contains(&i), b.slow.contains(&i), cancel, b.offset)
            };
            if i == 0 {
                ctx.record_command(
                    vec![
                        "synthetic-tool".into(),
                        "--item".into(),
                        ctx.item_id().into(),
                    ],
                    Some(0),
                    Some(0.0),
                );
            }
            if let Some(token) = cancel {
                token.cancel();
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            if slow {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            if fail {
                return Err(ErrorInfo::new(
                    ErrorCode::Validation,
                    format!("synthetic failure {i}"),
                ));
            }
            Ok(i as u64 * 10 + offset)
        }
    }

    #[derive(Debug, Clone, Serialize, JsonSchema)]
    struct NoParams {}

    struct DoubleStage {
        params: NoParams,
        calls: Arc<AtomicUsize>,
    }

    impl DoubleStage {
        fn new() -> Self {
            Self {
                params: NoParams {},
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Stage for DoubleStage {
        type Params = NoParams;
        type Work = u64;
        type Output = u64;

        fn name(&self) -> &'static str {
            "double"
        }
        fn version(&self) -> u32 {
            1
        }
        fn output(&self) -> ArtifactSpec {
            ArtifactSpec {
                schema: DOUBLED,
                version: Version::new(1, 0, 0),
            }
        }
        fn inputs(&self) -> Vec<InputDecl> {
            vec![InputDecl {
                schema: SOURCE,
                major: 1,
            }]
        }
        fn params(&self) -> &NoParams {
            &self.params
        }
        fn concurrency(&self) -> usize {
            4
        }
        fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<u64>>, StageError> {
            Ok(inputs
                .read_ok::<u64>(SOURCE)?
                .into_iter()
                .map(|(id, v)| WorkItem { id, work: v })
                .collect())
        }
        async fn process(&self, _ctx: &ItemContext, v: u64) -> Result<u64, ErrorInfo> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            Ok(v * 2)
        }
    }

    /// Waits on a shared barrier in its only item: two of these finish only when
    /// they run at the same time.
    struct BarrierStage {
        name: &'static str,
        schema: &'static str,
        barrier: Arc<tokio::sync::Barrier>,
        params: NoParams,
    }

    impl Stage for BarrierStage {
        type Params = NoParams;
        type Work = ();
        type Output = u64;

        fn name(&self) -> &'static str {
            self.name
        }
        fn version(&self) -> u32 {
            1
        }
        fn output(&self) -> ArtifactSpec {
            ArtifactSpec {
                schema: self.schema,
                version: Version::new(1, 0, 0),
            }
        }
        fn inputs(&self) -> Vec<InputDecl> {
            Vec::new()
        }
        fn params(&self) -> &NoParams {
            &self.params
        }
        fn plan(&self, _inputs: &StageInputs) -> Result<Vec<WorkItem<()>>, StageError> {
            Ok(vec![WorkItem {
                id: "only".into(),
                work: (),
            }])
        }
        async fn process(&self, _ctx: &ItemContext, _w: ()) -> Result<u64, ErrorInfo> {
            self.barrier.wait().await;
            Ok(1)
        }
    }

    #[tokio::test]
    async fn independent_stages_run_concurrently_on_one_runner() {
        let env = env();
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let stage = |name, schema| BarrierStage {
            name,
            schema,
            barrier: Arc::clone(&barrier),
            params: NoParams {},
        };
        let (left, right) = (stage("left", "x.left"), stage("right", "x.right"));
        let graph = StageGraph::new(vec![stage_decl(&left), stage_decl(&right)]).unwrap();
        let run = RunDir::open(
            &env.root.join("run"),
            "run",
            Producer::glassrip("0.1.0", None),
        )
        .unwrap();
        let runner = Runner::new(
            run,
            graph,
            &Selection::default(),
            env.cache.clone(),
            RunnerOptions::default(),
            CancellationToken::new(),
        )
        .unwrap();
        let both = async {
            tokio::join!(
                runner.run_stage_shared(&left),
                runner.run_stage_shared(&right)
            )
        };
        let (a, b) = tokio::time::timeout(Duration::from_secs(10), both)
            .await
            .expect("stages did not run concurrently");
        assert_eq!(a.unwrap().status, StageStatus::Ok);
        assert_eq!(b.unwrap().status, StageStatus::Ok);
        let m = runner.run_dir().manifest().clone();
        assert_eq!(m.stages["left"].status, StageStatus::Ok);
        assert_eq!(m.stages["right"].status, StageStatus::Ok);
        let run = runner.into_run_dir();
        assert!(run.artifact_path("x.left").is_file());
        assert!(run.artifact_path("x.right").is_file());
    }

    fn graph() -> StageGraph {
        StageGraph::new(vec![
            StageDecl::new("source", SOURCE, &[]),
            StageDecl::new("double", DOUBLED, &[SOURCE]),
        ])
        .unwrap()
    }

    struct Env {
        _dir: tempfile::TempDir,
        root: PathBuf,
        cache: Cache,
    }

    fn env() -> Env {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        Env {
            cache: Cache::in_workspace(&root),
            root,
            _dir: dir,
        }
    }

    fn runner(env: &Env, run: &str, sel: &Selection) -> Runner {
        runner_with(
            env,
            run,
            sel,
            RunnerOptions::default(),
            CancellationToken::new(),
        )
    }

    fn runner_with(
        env: &Env,
        run: &str,
        sel: &Selection,
        opts: RunnerOptions,
        cancel: CancellationToken,
    ) -> Runner {
        let dir =
            RunDir::open(&env.root.join(run), run, Producer::glassrip("0.1.0", None)).unwrap();
        Runner::new(dir, graph(), sel, env.cache.clone(), opts, cancel).unwrap()
    }

    fn read_doubled(r: &Runner) -> Vec<Record<u64>> {
        jsonl::read::<Record<u64>>(
            &r.run_dir().artifact_path(DOUBLED),
            &SchemaReq::new(DOUBLED, 1),
        )
        .unwrap()
        .items
    }

    #[tokio::test]
    async fn runs_then_restores_from_cache() {
        let env = env();
        let source = SourceStage::new(5);
        let double = DoubleStage::new();

        let mut r1 = runner(&env, "run-a", &Selection::default());
        let s = r1.run_stage(&source).await.unwrap();
        let d = r1.run_stage(&double).await.unwrap();
        assert_eq!(
            (s.status, s.items_ok, s.items_processed),
            (StageStatus::Ok, 5, 5)
        );
        assert_eq!((d.status, d.items_ok), (StageStatus::Ok, 5));
        let items = read_doubled(&r1);
        assert_eq!(items.len(), 5);
        assert_eq!(
            items[2],
            Record {
                id: "item-002".into(),
                outcome: Outcome::ok(40)
            }
        );
        let m = r1.run_dir().manifest().clone();
        assert_eq!(m.stages["source"].status, StageStatus::Ok);
        assert_eq!(m.commands.len(), 1);
        assert_eq!(m.commands[0].argv[0], "synthetic-tool");
        let bytes_a = fs_err::read(r1.run_dir().artifact_path(DOUBLED)).unwrap();
        assert!(
            !r1.run_dir()
                .root()
                .join("artifacts/.partial")
                .read_dir()
                .unwrap()
                .any(|_| true)
        );

        let mut r2 = runner(&env, "run-b", &Selection::default());
        let s2 = r2.run_stage(&source).await.unwrap();
        let d2 = r2.run_stage(&double).await.unwrap();
        assert_eq!(s2.status, StageStatus::Cached);
        assert_eq!(d2.status, StageStatus::Cached);
        assert_eq!(s2.cache_key, s.cache_key);
        assert_eq!(source.calls(), 5);
        assert_eq!(double.calls.load(Ordering::SeqCst), 5);
        let bytes_b = fs_err::read(r2.run_dir().artifact_path(DOUBLED)).unwrap();
        let body = |b: &[u8]| {
            let i = b.iter().position(|c| *c == b'\n').unwrap();
            b[i + 1..].to_vec()
        };
        assert_eq!(
            body(&bytes_b),
            body(&bytes_a),
            "item lines restored byte for byte"
        );
        assert_eq!(d2.content_hash, d.content_hash);
        assert_eq!(
            r2.run_dir().manifest().stages["double"].status,
            StageStatus::Cached
        );
    }

    #[tokio::test]
    async fn params_and_external_inputs_change_key() {
        let env = env();
        let ext = env.root.join("synthetic-input.bin");
        fs_err::write(&ext, b"one").unwrap();
        let mut source = SourceStage::new(2);
        source.external = vec![ext.clone()];

        let mut r = runner(&env, "run-a", &Selection::default());
        let k1 = r.run_stage(&source).await.unwrap().cache_key;
        drop(r);

        fs_err::write(&ext, b"two").unwrap();
        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.status, StageStatus::Ok);
        assert_ne!(rep.cache_key, k1);
        drop(r);

        source.params.n = 3;
        let mut r = runner(&env, "run-c", &Selection::default());
        assert_eq!(r.run_stage(&source).await.unwrap().status, StageStatus::Ok);
        assert_eq!(source.calls(), 2 + 2 + 3);
    }

    #[tokio::test]
    async fn force_bypasses_cache() {
        let env = env();
        let source = SourceStage::new(3);
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);
        let sel = Selection {
            force: ["source".to_string()].into(),
            ..Default::default()
        };
        let mut r = runner(&env, "run-b", &sel);
        assert_eq!(r.run_stage(&source).await.unwrap().status, StageStatus::Ok);
        assert_eq!(source.calls(), 6);
    }

    #[tokio::test]
    async fn error_rate_threshold() {
        let env = env();
        // 1 of 10 = 10%: at the threshold, the stage passes.
        let source = SourceStage::new(10);
        source.behavior.lock().unwrap().fail = [3].into();
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.status, rep.items_ok, rep.items_error),
            (StageStatus::Ok, 9, 1)
        );
        // Downstream sees only ok items.
        let d = r.run_stage(&DoubleStage::new()).await.unwrap();
        assert_eq!(d.items_total, 9);
        drop(r);

        // 2 of 10 = 20%: the stage fails and writes no artifact.
        let source = SourceStage::new(10);
        source.behavior.lock().unwrap().fail = [3, 7].into();
        let mut r = runner(&env, "run-b", &Selection::default());
        match r.run_stage(&source).await {
            Err(RunnerError::ErrorRateExceeded { errors, total, .. }) => {
                assert_eq!((errors, total), (2, 10))
            }
            other => panic!("expected ErrorRateExceeded, got {other:?}"),
        }
        assert!(!r.run_dir().artifact_path(SOURCE).exists());
        let rec = r.run_dir().manifest().stages["source"].clone();
        assert_eq!((rec.status, rec.items_error), (StageStatus::Failed, 2));
        assert_eq!(
            r.run_dir().manifest().commands.len(),
            1,
            "commands from a failed stage are still recorded"
        );
        assert!(rec.error.as_deref().unwrap_or("").contains("2 of 10"));
    }

    #[tokio::test]
    async fn resume_retries_only_failed_items() {
        let env = env();
        let source = SourceStage::new(10);
        source.behavior.lock().unwrap().fail = [4].into();
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.items_error, 1);
        drop(r);
        assert!(
            env.cache.ls().unwrap().is_empty(),
            "outputs with failed items are not cached"
        );

        source.behavior.lock().unwrap().fail.clear();
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (10, 0, 1)
        );
        assert_eq!(source.calls(), 11);
        assert_eq!(env.cache.ls().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn timeout_is_an_item_error() {
        let env = env();
        let mut source = SourceStage::new(10);
        source.timeout = Some(Duration::from_millis(50));
        source.behavior.lock().unwrap().slow = [5].into();
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.items_error, 1);
        let env_items = jsonl::read::<Record<u64>>(
            &r.run_dir().artifact_path(SOURCE),
            &SchemaReq::new(SOURCE, 1),
        )
        .unwrap();
        let failed = env_items.items.iter().find(|x| x.id == "item-005").unwrap();
        assert_eq!(
            failed.outcome.error.as_ref().map(|e| e.code),
            Some(ErrorCode::Timeout)
        );
    }

    #[tokio::test]
    async fn cancellation_stops_and_resume_finishes() {
        let env = env();
        let token = CancellationToken::new();
        let source = SourceStage::new(8);
        source.behavior.lock().unwrap().cancel_at = Some((3, token.clone()));
        let mut r = runner_with(
            &env,
            "run-a",
            &Selection::default(),
            RunnerOptions::default(),
            token,
        );
        assert!(matches!(
            r.run_stage(&source).await,
            Err(RunnerError::Cancelled { .. })
        ));
        assert_eq!(
            source.calls(),
            4,
            "items after the cancel point never start"
        );
        assert_eq!(
            r.run_dir().manifest().stages["source"].status,
            StageStatus::Cancelled
        );
        assert_eq!(r.run_dir().manifest().stages["source"].items_ok, 3);
        assert!(!r.run_dir().artifact_path(SOURCE).exists());
        // A cancelled runner refuses further stages.
        assert!(matches!(
            r.run_stage(&DoubleStage::new()).await,
            Err(RunnerError::Cancelled { .. })
        ));
        drop(r);

        source.behavior.lock().unwrap().cancel_at = None;
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.items_ok, rep.items_processed), (8, 5));
        assert_eq!(source.calls(), 9);
    }

    #[tokio::test]
    async fn selection_and_missing_inputs() {
        let env = env();
        let source = SourceStage::new(2);
        let double = DoubleStage::new();

        let until = Selection {
            until: Some("source".into()),
            ..Default::default()
        };
        let mut r = runner(&env, "run-a", &until);
        assert_eq!(r.run_stage(&source).await.unwrap().status, StageStatus::Ok);
        assert_eq!(
            r.run_stage(&double).await.unwrap().status,
            StageStatus::Skipped
        );
        assert_eq!(
            r.run_dir().manifest().stages["double"].status,
            StageStatus::Skipped
        );
        drop(r);

        let from = Selection {
            from: Some("double".into()),
            ..Default::default()
        };
        let mut r = runner(&env, "run-b", &from);
        assert_eq!(
            r.run_stage(&source).await.unwrap().status,
            StageStatus::Skipped
        );
        assert!(matches!(
            r.run_stage(&double).await,
            Err(RunnerError::MissingInputArtifact { .. })
        ));
        drop(r);

        // From-stage reuses the upstream artifact already in the run directory.
        let mut r = runner(&env, "run-a", &from);
        assert_eq!(
            r.run_stage(&source).await.unwrap().status,
            StageStatus::Skipped
        );
        assert_eq!(r.run_stage(&double).await.unwrap().status, StageStatus::Ok);
        assert_eq!(source.calls(), 2);
    }

    #[tokio::test]
    async fn rejects_stage_outside_graph_or_mismatched() {
        struct Stray(SourceStage);
        impl Stage for Stray {
            type Params = SourceParams;
            type Work = usize;
            type Output = u64;
            fn name(&self) -> &'static str {
                "double"
            }
            fn version(&self) -> u32 {
                1
            }
            fn output(&self) -> ArtifactSpec {
                self.0.output()
            }
            fn inputs(&self) -> Vec<InputDecl> {
                Vec::new()
            }
            fn params(&self) -> &SourceParams {
                self.0.params()
            }
            fn plan(&self, i: &StageInputs) -> Result<Vec<WorkItem<usize>>, StageError> {
                self.0.plan(i)
            }
            async fn process(&self, ctx: &ItemContext, w: usize) -> Result<u64, ErrorInfo> {
                self.0.process(ctx, w).await
            }
        }
        let env = env();
        let mut r = runner(&env, "run-a", &Selection::default());
        assert!(matches!(
            r.run_stage(&Stray(SourceStage::new(1))).await,
            Err(RunnerError::DeclMismatch { .. })
        ));
        let g = StageGraph::new(vec![StageDecl::new("other", "x.o", &[])]).unwrap();
        let dir = RunDir::open(
            &env.root.join("run-b"),
            "run-b",
            Producer::glassrip("0.1.0", None),
        )
        .unwrap();
        let mut r = Runner::new(
            dir,
            g,
            &Selection::default(),
            env.cache.clone(),
            RunnerOptions::default(),
            CancellationToken::new(),
        )
        .unwrap();
        assert!(matches!(
            r.run_stage(&SourceStage::new(1)).await,
            Err(RunnerError::StageNotInGraph(_))
        ));
    }

    #[tokio::test]
    async fn schema_mismatch_on_input_fails_loudly() {
        let env = env();
        let mut r = runner(&env, "run-a", &Selection::default());
        let header = EnvelopeHeader {
            schema: SOURCE.into(),
            schema_version: Version::new(2, 0, 0),
            run_id: "run-a".into(),
            producer: Producer::glassrip("0.1.0", None),
            inputs: vec![],
            params: json!({}),
            content_hash: None,
            restored_from: None,
        };
        jsonl::write_atomic::<Record<u64>>(&r.run_dir().artifact_path(SOURCE), &header, &[])
            .unwrap();
        match r.run_stage(&DoubleStage::new()).await {
            Err(RunnerError::Input {
                source: JsonlError::Header { .. },
                ..
            }) => {}
            other => panic!("expected input header error, got {other:?}"),
        }
    }

    fn read_header_of(r: &Runner, schema: &str) -> EnvelopeHeader {
        jsonl::read_header(
            &r.run_dir().artifact_path(schema),
            &SchemaReq::new(schema, 1),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn forced_upstream_with_identical_items_keeps_downstream_cached() {
        let env = env();
        let source = SourceStage::new(5);
        let double = DoubleStage::new();
        let mut r = runner(&env, "run-a", &Selection::default());
        let s1 = r.run_stage(&source).await.unwrap();
        r.run_stage(&double).await.unwrap();
        drop(r);

        let sel = Selection {
            force: ["source".to_string()].into(),
            ..Default::default()
        };
        let mut r = runner(&env, "run-b", &sel);
        let s2 = r.run_stage(&source).await.unwrap();
        assert_eq!(s2.status, StageStatus::Ok, "forced stage recomputes");
        assert_eq!(
            s2.content_hash, s1.content_hash,
            "identical items, identical content hash"
        );
        assert_eq!(read_header_of(&r, SOURCE).run_id, "run-b");
        let d2 = r.run_stage(&double).await.unwrap();
        assert_eq!(d2.status, StageStatus::Cached, "downstream stays cached");
        assert_eq!(double.calls.load(Ordering::SeqCst), 5);
        assert_eq!(
            r.run_dir().manifest().stages["source"].content_hash,
            s1.content_hash
        );
    }

    #[tokio::test]
    async fn changed_upstream_items_recompute_downstream() {
        let env = env();
        let source = SourceStage::new(5);
        let double = DoubleStage::new();
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&source).await.unwrap();
        r.run_stage(&double).await.unwrap();
        drop(r);

        source.behavior.lock().unwrap().offset = 1;
        let sel = Selection {
            force: ["source".to_string()].into(),
            ..Default::default()
        };
        let mut r = runner(&env, "run-b", &sel);
        r.run_stage(&source).await.unwrap();
        let d = r.run_stage(&double).await.unwrap();
        assert_eq!((d.status, d.items_processed), (StageStatus::Ok, 5));
        assert_eq!(read_doubled(&r)[1].outcome.result, Some(22));
    }

    #[tokio::test]
    async fn restored_artifact_header_names_current_run() {
        let env = env();
        let source = SourceStage::new(3);
        let mut r = runner(&env, "run-a", &Selection::default());
        let first = r.run_stage(&source).await.unwrap();
        let original = fs_err::read_to_string(r.run_dir().artifact_path(SOURCE)).unwrap();
        drop(r);

        let mut r = runner(&env, "run-b", &Selection::default());
        let restored = r.run_stage(&source).await.unwrap();
        assert_eq!(restored.status, StageStatus::Cached);
        let header = read_header_of(&r, SOURCE);
        assert_eq!(header.run_id, "run-b");
        assert_eq!(
            header.restored_from,
            Some(RestoredFrom {
                run_id: "run-a".into(),
                cache_key: first.cache_key.as_ref().unwrap().to_string(),
            })
        );
        assert_eq!(header.content_hash, first.content_hash);
        let now = fs_err::read_to_string(r.run_dir().artifact_path(SOURCE)).unwrap();
        let body = |t: &str| t.split_once('\n').map(|(_, b)| b.to_string()).unwrap();
        assert_eq!(body(&now), body(&original), "item lines are byte-identical");
        assert_eq!(
            jsonl::compute_content_hash(
                &r.run_dir().artifact_path(SOURCE),
                &SchemaReq::new(SOURCE, 1)
            )
            .unwrap(),
            first.content_hash.unwrap()
        );
    }

    #[tokio::test]
    async fn corrupt_cache_entry_is_quarantined_and_recomputed() {
        let env = env();
        let source = SourceStage::new(3);
        let mut r = runner(&env, "run-a", &Selection::default());
        let first = r.run_stage(&source).await.unwrap();
        drop(r);
        let key = first.cache_key.unwrap();
        let entry = env.cache.entry_path("source", &key, OUTPUT_EXT).unwrap();

        // Still valid JSON, but an item changed: the content hash no longer matches.
        let text = fs_err::read_to_string(&entry).unwrap();
        fs_err::write(&entry, text.replace("\"result\":10", "\"result\":11")).unwrap();
        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.status, StageStatus::Ok);
        assert_eq!(source.calls(), 6);
        let quarantined: Vec<_> =
            fs_err::read_dir(env.cache.root().join(crate::cache::QUARANTINE_DIR))
                .unwrap()
                .collect();
        assert_eq!(quarantined.len(), 1);
        drop(r);

        // Garbage bytes: also a miss, never a run failure.
        fs_err::write(&entry, b"not json at all\n").unwrap();
        let mut r = runner(&env, "run-c", &Selection::default());
        assert_eq!(r.run_stage(&source).await.unwrap().status, StageStatus::Ok);
        assert_eq!(source.calls(), 9);
        drop(r);
        let mut r = runner(&env, "run-d", &Selection::default());
        assert_eq!(
            r.run_stage(&source).await.unwrap().status,
            StageStatus::Cached
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn read_only_cache_serves_hits() {
        use std::os::unix::fs::PermissionsExt;
        let env = env();
        let source = SourceStage::new(3);
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);

        let set_mode = |mode: u32| {
            for e in env.cache.ls().unwrap() {
                fs_err::set_permissions(&e.path, std::fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        set_mode(0o444);
        let mut r = runner(&env, "run-b", &Selection::default());
        let result = r.run_stage(&source).await;
        set_mode(0o644);
        assert_eq!(result.unwrap().status, StageStatus::Cached);
        assert_eq!(source.calls(), 3);
    }

    #[tokio::test]
    async fn gc_refused_while_runner_holds_cache() {
        let env = env();
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&SourceStage::new(2)).await.unwrap();
        assert!(matches!(
            env.cache.gc(Duration::ZERO, std::time::SystemTime::now()),
            Err(CacheError::Busy(_))
        ));
        drop(r);
        assert!(
            env.cache
                .gc(Duration::ZERO, std::time::SystemTime::now())
                .is_ok()
        );
    }

    #[tokio::test]
    async fn non_finite_params_rejected() {
        let env = env();
        let mut source = SourceStage::new(2);
        source.params.scale = f64::NAN;
        let mut r = runner(&env, "run-a", &Selection::default());
        match r.run_stage(&source).await {
            Err(RunnerError::Canonical(CanonicalJsonError::NonFinite { path })) => {
                assert_eq!(path, "$.scale");
            }
            other => panic!("expected NonFinite, got {other:?}"),
        }
        assert_eq!(source.calls(), 0);
    }
}
