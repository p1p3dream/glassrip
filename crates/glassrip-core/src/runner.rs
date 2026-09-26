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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
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
    /// False for a stage whose result is more than its artifact (say files it
    /// writes outside the run directory), or whose key cannot identify it: it is
    /// never restored from the cache, resumed from a partial, or stored in the
    /// cache, but runs whenever selected, as if forced.
    fn cacheable(&self) -> bool {
        true
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
    /// A [`recurrent`](ErrorInfo::recurrent) item failure becomes terminal once
    /// it ends the item identically on this many consecutive runs (at least 1).
    pub terminal_after_repeats: u32,
    /// Items to recompute even when their stage's output is cached or their
    /// failure is terminal, by stage name and item id. The rest of the stage is
    /// restored, not recomputed (unlike a forced stage).
    pub force_items: BTreeMap<String, BTreeSet<String>>,
}

pub use crate::config::DEFAULT_TERMINAL_AFTER_REPEATS;

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
            terminal_after_repeats: cfg.terminal_after_repeats,
            force_items: BTreeMap::new(),
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

/// A stage's loaded inputs and cache key.
struct KeyedInputs {
    artifacts: BTreeMap<String, LoadedInput>,
    input_refs: Vec<InputRef>,
    params: serde_json::Value,
    key: CacheKey,
}

/// A verified cache entry ready to be restored.
struct CachedArtifact {
    header: EnvelopeHeader,
    counts: Counts,
    total: u64,
    content_hash: String,
    /// Bytes after the header line (the item lines), copied verbatim.
    body: Vec<u8>,
    /// Failed items whose failure is not settled yet (recurrent, not terminal):
    /// a rerun retries them and restores the rest.
    unsettled: u64,
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
        let unsettled = items
            .iter()
            .filter(|r| {
                r.outcome
                    .error
                    .as_ref()
                    .is_some_and(|e| !e.settled(self.repeats()))
            })
            .count() as u64;
        Ok(CachedArtifact {
            unsettled,
            counts: Counts::of(&items),
            total: items.len() as u64,
            header,
            content_hash: recorded,
            body: bytes[header_len..].to_vec(),
        })
    }

    /// A cached output whose error rate exceeds the limit: restoring it fails
    /// the stage.
    fn over_limit(&self, c: &CachedArtifact) -> bool {
        #[allow(clippy::cast_precision_loss)]
        let rate = if c.total == 0 {
            0.0
        } else {
            c.counts.error as f64 / c.total as f64
        };
        rate > self.opts.max_item_error_rate
    }

    /// Consecutive identical runs that settle a recurrent failure.
    fn repeats(&self) -> u32 {
        self.opts.terminal_after_repeats.max(1)
    }

    /// The item records of a validated cache entry, parsed from the same bytes
    /// whose content hash was checked (the entry is not read a second time,
    /// so a concurrent replacement cannot pair one entry's hash with another's
    /// records).
    fn cached_records<S: Stage>(
        c: &CachedArtifact,
        req: &SchemaReq,
    ) -> Option<Vec<Record<S::Output>>> {
        let mut bytes = jsonl::header_line(&c.header).ok()?;
        bytes.extend_from_slice(&c.body);
        jsonl::parse_bytes::<Record<S::Output>>(Path::new("cache entry"), &bytes, req)
            .ok()
            .map(|(_, items)| items)
    }

    /// A stage's partial output file and the marker naming the cache entry it
    /// was seeded from.
    fn partial_paths(&self, name: &str, key: &CacheKey) -> (PathBuf, PathBuf) {
        let partial = self
            .with_run(|r| r.partials_dir())
            .join(format!("{name}-{}.jsonl", &key.as_str()[..16]));
        let marker = partial.with_extension("seed");
        (partial, marker)
    }

    /// How this run directory's partial relates to the cache entry with content
    /// hash `hash`, from the marker next to it. A partial with a lineage is
    /// newer than that entry (work finished after it, before a crash), so it
    /// is resumed instead of the entry being restored.
    fn lineage(&self, name: &str, key: &CacheKey, hash: &str) -> Option<Lineage> {
        let (partial, marker) = self.partial_paths(name, key);
        if !atomic::is_file(&partial).unwrap_or(false) {
            return None;
        }
        let text = fs_err::read_to_string(&marker).ok()?;
        match text.trim().split_once(' ')? {
            ("seed", h) if h == hash => Some(Lineage::Seeded),
            ("force", h) if h == hash => Some(Lineage::Forced),
            _ => None,
        }
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

    /// Loads and checks a stage's declared inputs and derives its cache key.
    fn keyed_inputs<S: Stage>(&self, stage: &S) -> Result<KeyedInputs, RunnerError> {
        let name = stage.name();
        let version = stage.version();
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
        Ok(KeyedInputs {
            artifacts,
            input_refs,
            params,
            key,
        })
    }

    /// True when running `stage` now would restore its output from the cache:
    /// the stage is selected without `--force-stage`, its inputs exist, and a
    /// valid cache entry exists for its key. Model stages use this to skip model
    /// preflight on a fully cached rerun. Errors (missing inputs) mean "no".
    pub fn cache_hit<S: Stage>(&self, stage: &S) -> bool {
        if !stage.cacheable() {
            return false;
        }
        match self.plan.decision(stage.name()) {
            Some(StageDecision::Run { force: false }) => {}
            _ => return false,
        }
        let Ok(k) = self.keyed_inputs(stage) else {
            return false;
        };
        let output = stage.output();
        let req = SchemaReq::new(output.schema, output.version.major);
        match self.cache.get_path(stage.name(), &k.key, OUTPUT_EXT) {
            Ok(Some(path)) => {
                self.opts
                    .force_items
                    .get(stage.name())
                    .is_none_or(BTreeSet::is_empty)
                    && self.load_cached::<S>(&path, &req).is_ok_and(|c| {
                        c.unsettled == 0
                            && !self.over_limit(&c)
                            && self
                                .lineage(stage.name(), &k.key, &c.content_hash)
                                .is_none()
                    })
            }
            _ => false,
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
            // A stage that is not cacheable always starts over.
            Some(StageDecision::Run { force }) => force || !stage.cacheable(),
        };
        if self.cancel.is_cancelled() {
            return Err(RunnerError::Cancelled {
                stage: name.to_string(),
            });
        }

        let KeyedInputs {
            artifacts,
            input_refs,
            params,
            key,
        } = self.keyed_inputs(stage)?;
        let output = stage.output();
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

        // Items forced one by one: the rest of a cached output seeds the partial
        // below instead of being restored as is.
        let forced_items: HashSet<String> = if force {
            HashSet::new()
        } else {
            self.opts
                .force_items
                .get(name)
                .map(|ids| ids.iter().cloned().collect())
                .unwrap_or_default()
        };
        let mut seed: Option<Vec<Record<S::Output>>> = None;
        // Content hash of the cache entry the seed comes from: the provenance a
        // seeded partial records, so a later run can tell whether the partial
        // descends from the current cache entry (and is newer) or predates it.
        let mut seed_hash: Option<String> = None;

        // Cache hit: restore with a header rewritten for this run; item lines are
        // copied byte for byte, so the content hash is unchanged.
        // A cached output with unsettled or forced items seeds the partial instead.
        // Looked up for a forced stage too: its partial records the entry it
        // supersedes.
        let entry = self.cache_lookup::<S>(name, &key, &out_req);
        let lineage = entry
            .as_ref()
            .filter(|_| !force)
            .and_then(|c| self.lineage(name, &key, &c.content_hash));
        // The entry a forced run's partial supersedes (this run's, or the one a
        // resumed forced run superseded).
        let superseded = entry
            .as_ref()
            .filter(|_| force || lineage == Some(Lineage::Forced))
            .map(|c| c.content_hash.clone());
        let cached = if force { None } else { entry };
        let cached = match cached {
            // A forced run stopped before finalizing: its partial supersedes the
            // entry and is resumed alone, not merged with the entry's records.
            Some(_) if lineage == Some(Lineage::Forced) => {
                info!(stage = name, key = %key, "resuming a forced run of this stage");
                None
            }
            // A partial seeded from this entry is newer than it (say a forced
            // item finished before a crash): resume it, even when the entry
            // would restore as is.
            Some(c)
                if !forced_items.is_empty()
                    || c.unsettled > 0
                    || lineage == Some(Lineage::Seeded) =>
            {
                info!(
                    stage = name,
                    key = %key,
                    unsettled = c.unsettled,
                    forced = forced_items.len(),
                    "restoring cached items and retrying the unsettled or forced ones"
                );
                seed = Self::cached_records::<S>(&c, &out_req);
                seed_hash = seed.as_ref().map(|_| c.content_hash.clone());
                None
            }
            other => other,
        };
        if let Some(cached) = cached {
            if self.over_limit(&cached) {
                // Only settled failures are restored; they fail the stage as they
                // did when they were recorded.
                return Err(RunnerError::ErrorRateExceeded {
                    stage: name.to_string(),
                    errors: cached.counts.error,
                    total: cached.total,
                    threshold: self.opts.max_item_error_rate,
                });
            }
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
        let (partial, seed_marker) = self.partial_paths(name, &key);
        let partial_exists = atomic::metadata_opt(&partial)
            .map_err(Self::io_err(&partial))?
            .is_some();
        // A partial with the lineage of the current cache entry holds work
        // finished after it: it is resumed (a crash before finalizing must not
        // throw away a recovered item).
        let descends = lineage.is_some();
        // A forced stage starts over. Any other partial next to a cache seed
        // predates that cache entry (another run finalized after it was
        // written), so it must not override the newer cached state.
        let discard = partial_exists && (force || (seed.is_some() && !descends));
        if discard {
            fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
        }
        // The marker lives and dies with its partial.
        if discard || !partial_exists {
            remove_if_exists(&seed_marker)?;
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
                remove_if_exists(&seed_marker)?;
                JsonlWriter::open_resume(&partial, &header)?
            }
            Err(e) => return Err(e.into()),
        };
        let mut writer = writer.with_checkpoint_every(self.opts.checkpoint_every);
        if let Some(h) = &superseded {
            // Before any item is processed: from here on the partial is newer
            // than the entry it supersedes.
            atomic::write_atomic(&seed_marker, format!("force {h}").as_bytes())?;
        }
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
        let mut resumed_items = resumed.items;
        if resumed_items.is_empty() {
            if let (Some(seed), Some(hash)) = (seed.take(), seed_hash.as_ref()) {
                // A fresh partial starts from the cached output, so the items not
                // forced are kept, not recomputed. The marker is written once the
                // seed is on stable storage: from then on the partial is a
                // descendant of this cache entry.
                for r in seed {
                    writer.append(&r)?;
                    resumed_items.push(r);
                }
                writer.checkpoint()?;
                atomic::write_atomic(&seed_marker, format!("seed {hash}").as_bytes())?;
            }
        } else if let Some(seed) = seed.take().filter(|_| descends) {
            // Merged per item: the seed first, then the partial's records, which
            // were written after it and so win for any item they cover.
            resumed_items = seed.into_iter().chain(resumed_items).collect();
        }
        for id in &forced_items {
            if !ids.contains(id) {
                warn!(stage = name, item = %id, "forced item is not planned by this stage");
            }
        }
        let mut done: HashMap<String, Record<S::Output>> = resumed_items
            .into_iter()
            .filter(|r| ids.contains(&r.id))
            .map(|r| (r.id.clone(), r))
            .collect();
        // A terminal failure is the item's settled answer (see
        // `ErrorInfo::terminal`): resume keeps it like a result. A forced stage
        // removed its partial output above, so it retries everything; a forced
        // item is retried whatever its record says.
        let repeats = self.repeats();
        let settled = |r: &Record<S::Output>| {
            !forced_items.contains(&r.id)
                && (!r.outcome.is_error()
                    || r.outcome.error.as_ref().is_some_and(|e| e.settled(repeats)))
        };
        let todo: Vec<WorkItem<S::Work>> = work
            .into_iter()
            .filter(|w| done.get(&w.id).is_none_or(|r| !settled(r)))
            .collect();
        info!(
            stage = name,
            planned = order.len(),
            resumed = done.values().filter(|r| settled(r)).count(),
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
                let Some(mut outcome) = outcome else {
                    cancelled = true;
                    continue;
                };
                // A recurrent failure counts the consecutive runs that ended the
                // item the same way (the previous one is in the resumed partial)
                // and settles only once it has recurred often enough.
                if let Some(e) = outcome
                    .error
                    .as_mut()
                    .filter(|e| e.recurrent && !e.terminal)
                {
                    let before = done
                        .get(&id)
                        .and_then(|r| r.outcome.error.as_ref())
                        .filter(|p| p.recurrent && p.same_failure(e))
                        .map_or(0, |p| p.occurrences.max(1));
                    e.occurrences = before.saturating_add(1);
                    e.terminal = e.occurrences >= repeats;
                }
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
        // Finalize.
        let items: Vec<Record<S::Output>> = order.iter().filter_map(|id| done.remove(id)).collect();
        let hash = content_hash(
            &header.schema,
            &header.schema_version,
            &header.params,
            &items,
        )?;
        header.content_hash = Some(hash.clone());
        // Plain retryable failures (transport, placement) keep the output out of
        // the cache. Recurrent ones do not: the cached output carries their count
        // to the next run, which retries only them (see the cache lookup above).
        let retryable = items
            .iter()
            .filter(|r| {
                r.outcome
                    .error
                    .as_ref()
                    .is_some_and(|e| !e.settled(repeats) && !e.recurrent)
            })
            .count();
        let unsettled = items
            .iter()
            .filter(|r| {
                r.outcome
                    .error
                    .as_ref()
                    .is_some_and(|e| !e.settled(repeats) && e.recurrent)
            })
            .count();
        if rate > self.opts.max_item_error_rate {
            // Cached all the same when nothing plainly retryable failed, so the
            // next run (in any run directory) retries only the unsettled items
            // and keeps counting their recurrences; a restored output over the
            // limit fails the stage again (see the cache hit above). A stage
            // that is not cacheable stores nothing and never resumes.
            if !stage.cacheable() {
                drop(writer);
                fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
                remove_if_exists(&seed_marker)?;
            } else if retryable == 0 {
                let over = self
                    .with_run(|r| r.partials_dir())
                    .join(format!("{name}-{}.over-limit.jsonl", &key.as_str()[..16]));
                jsonl::write_atomic(&over, &header, &items)?;
                if let Err(e) = self.cache.put_file(name, &key, OUTPUT_EXT, &over) {
                    warn!(stage = name, error = %e, "could not store output in cache; dropping the partial output");
                }
                // Stored: the cache holds the stage's state now, and the partial
                // would only go stale against it. Not stored: a kept partial (and
                // its seed marker) would be resumed as is by every later run,
                // failing the stage again without retrying anything until the
                // cache is writable; without it the next run starts from the
                // cache entry (or from scratch) and retries what is unsettled.
                drop(writer);
                fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
                remove_if_exists(&seed_marker)?;
                fs_err::remove_file(&over).map_err(Self::io_err(&over))?;
            }
            return Err(RunnerError::ErrorRateExceeded {
                stage: name.to_string(),
                errors: counts.error,
                total,
                threshold: self.opts.max_item_error_rate,
            });
        }
        jsonl::write_atomic(&out_path, &header, &items)?;
        drop(writer);
        if !stage.cacheable() {
            // Never restored, so nothing is stored; the output is written.
            fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
            remove_if_exists(&seed_marker)?;
        } else if retryable == 0 {
            // Terminal failures are cached with the results: the cache key holds
            // everything that determines them (inputs, params, model digest), so a
            // rerun restores them instead of asking again. `--force-stage` or a
            // forced item retries them. Unsettled recurrent failures are cached
            // with their count; the next run retries only those items.
            if counts.error > 0 {
                info!(
                    stage = name,
                    errors = counts.error,
                    unsettled,
                    "caching output with item failures; unsettled ones are retried on the next run, \
                     terminal ones by --force-stage or a forced item"
                );
            }
            match self.cache.put_file(name, &key, OUTPUT_EXT, &out_path) {
                // The cache holds the stage's state now.
                Ok(_) => {
                    fs_err::remove_file(&partial).map_err(Self::io_err(&partial))?;
                    remove_if_exists(&seed_marker)?;
                }
                // A cache that cannot be written (read-only, full) costs this run
                // nothing; the partial is kept, so the next run resumes it
                // instead of restoring an older entry.
                Err(e) => {
                    warn!(stage = name, error = %e, "could not store output in cache; keeping the partial output")
                }
            }
        } else {
            info!(
                stage = name,
                errors = counts.error,
                retryable,
                "not caching output with retryable failed items; rerun retries them"
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

/// How a run directory's partial output relates to a cache entry (recorded in
/// the marker next to the partial).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lineage {
    /// Seeded from the entry's records, then extended.
    Seeded,
    /// A forced run of the stage that supersedes the entry.
    Forced,
}

/// Removes a file that may not exist.
fn remove_if_exists(path: &Path) -> Result<(), RunnerError> {
    match fs_err::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(RunnerError::Io {
            path: path.to_path_buf(),
            source: e,
        }),
        _ => Ok(()),
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
        /// Failing items whose failure is terminal (deterministic).
        terminal: HashSet<usize>,
        /// Failing items whose failure is recurrent (a model reply).
        recurrent: HashSet<usize>,
        /// Varies the failure text (a different reply each run).
        variant: u64,
        /// Failures carry a fixed signature (one loop stopped at another byte).
        signed: bool,
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
            let (fail, terminal, recurrent, variant, signed, slow, cancel, offset) = {
                let b = self.behavior.lock().unwrap();
                let cancel = b
                    .cancel_at
                    .as_ref()
                    .filter(|(at, _)| *at == i)
                    .map(|(_, t)| t.clone());
                (
                    b.fail.contains(&i),
                    b.terminal.contains(&i),
                    b.recurrent.contains(&i),
                    b.variant,
                    b.signed,
                    b.slow.contains(&i),
                    cancel,
                    b.offset,
                )
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
                let mut e = ErrorInfo::new(ErrorCode::Validation, format!("synthetic failure {i}"))
                    .with_raw_text(format!("synthetic reply {variant}"));
                if signed {
                    e = e.with_signature("synthetic loop");
                }
                return Err(if terminal {
                    e.terminal()
                } else if recurrent {
                    e.terminal_if_repeated()
                } else {
                    e
                });
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
    async fn cache_hit_predicts_a_restore() {
        let env = env();
        let src = SourceStage::new(3);
        let mut r1 = runner(&env, "a", &Selection::default());
        assert!(!r1.cache_hit(&src), "nothing cached yet");
        r1.run_stage(&src).await.unwrap();
        drop(r1);
        let r2 = runner(&env, "b", &Selection::default());
        assert!(r2.cache_hit(&src));
        let forced = runner(
            &env,
            "c",
            &Selection {
                force: std::collections::BTreeSet::from(["source".to_string()]),
                ..Selection::default()
            },
        );
        assert!(!forced.cache_hit(&src), "a forced stage is never a hit");
        // Different params mean a different key.
        let mut other = SourceStage::new(3);
        other.params.scale = 2.0;
        assert!(!r2.cache_hit(&other));
        // Missing inputs are not a hit (no error).
        assert!(!r2.cache_hit(&DoubleStage::new()));
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
    async fn terminal_failures_are_cached_and_restored_unless_forced() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.terminal = [4].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_error), (StageStatus::Ok, 1));
        drop(r);
        assert_eq!(
            env.cache.ls().unwrap().len(),
            1,
            "an output whose only failures are terminal is cached"
        );

        // A rerun (same or another run directory) restores the failure record
        // instead of asking again, even though the item would now succeed.
        source.behavior.lock().unwrap().fail.clear();
        for run in ["run-a", "run-b"] {
            let mut r = runner(&env, run, &Selection::default());
            let rep = r.run_stage(&source).await.unwrap();
            assert_eq!(
                (rep.status, rep.items_ok, rep.items_error),
                (StageStatus::Cached, 9, 1),
                "{run}"
            );
            let items = jsonl::read::<Record<u64>>(
                &r.run_dir().artifact_path(SOURCE),
                &SchemaReq::new(SOURCE, 1),
            )
            .unwrap()
            .items;
            let failed = items.iter().find(|x| x.id == "item-004").unwrap();
            assert!(failed.outcome.error.as_ref().is_some_and(|e| e.terminal));
        }
        assert_eq!(source.calls(), 10);

        // --force-stage retries it.
        let sel = Selection {
            force: ["source".to_string()].into(),
            ..Default::default()
        };
        let mut r = runner(&env, "run-c", &sel);
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (
                rep.status,
                rep.items_ok,
                rep.items_error,
                rep.items_processed
            ),
            (StageStatus::Ok, 10, 0, 10)
        );
    }

    #[tokio::test]
    async fn resume_keeps_terminal_failures_and_retries_the_rest() {
        let env = env();
        let source = SourceStage::new(20);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [3, 4].into();
            b.terminal = [4].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.items_error, 2);
        drop(r);
        assert!(
            env.cache.ls().unwrap().is_empty(),
            "a retryable failure keeps the output out of the cache"
        );

        source.behavior.lock().unwrap().fail.clear();
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (19, 1, 1),
            "only the retryable failure is processed again"
        );
        assert_eq!(source.calls(), 21);
        assert_eq!(env.cache.ls().unwrap().len(), 1);
    }

    fn failed_record(r: &Runner, id: &str) -> ErrorInfo {
        jsonl::read::<Record<u64>>(
            &r.run_dir().artifact_path(SOURCE),
            &SchemaReq::new(SOURCE, 1),
        )
        .unwrap()
        .items
        .into_iter()
        .find(|x| x.id == id)
        .and_then(|x| x.outcome.error)
        .unwrap()
    }

    /// Codex 6: one malformed model reply is not permanent. The next run, in a
    /// fresh run directory, retries only that item; an identical second failure
    /// settles it, and later runs restore it without asking again.
    #[tokio::test]
    async fn a_recurrent_failure_is_retried_next_run_and_settles_when_it_repeats() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.recurrent = [4].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_error), (StageStatus::Ok, 1));
        let e = failed_record(&r, "item-004");
        assert!(e.recurrent && !e.terminal && e.occurrences == 1, "{e:?}");
        assert!(
            !r.cache_hit(&source),
            "an unsettled failure needs the model"
        );
        drop(r);
        assert_eq!(env.cache.ls().unwrap().len(), 1, "the rest is cached");

        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (
                rep.status,
                rep.items_ok,
                rep.items_error,
                rep.items_processed
            ),
            (StageStatus::Ok, 9, 1, 1),
            "only the unsettled item is asked again"
        );
        let e = failed_record(&r, "item-004");
        assert!(e.terminal && e.occurrences == 2, "{e:?}");
        assert!(r.cache_hit(&source));
        drop(r);
        assert_eq!(source.calls(), 11);

        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_error), (StageStatus::Cached, 1));
        assert_eq!(source.calls(), 11, "a settled failure is not asked again");
    }

    #[tokio::test]
    async fn a_recurrent_failure_that_recovers_is_replaced() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.recurrent = [4].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);
        source.behavior.lock().unwrap().fail.clear();
        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (10, 0, 1)
        );
        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_ok), (StageStatus::Cached, 10));
        assert_eq!(source.calls(), 11);
    }

    #[tokio::test]
    async fn different_recurrent_failures_do_not_settle() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.recurrent = [4].into();
        }
        for (k, run) in ["run-a", "run-b", "run-c"].into_iter().enumerate() {
            source.behavior.lock().unwrap().variant = k as u64;
            let mut r = runner(&env, run, &Selection::default());
            r.run_stage(&source).await.unwrap();
            let e = failed_record(&r, "item-004");
            assert!(!e.terminal && e.occurrences == 1, "{run}: {e:?}");
        }
        assert_eq!(source.calls(), 12);
    }

    /// Codex review 5: one loop stopped at another byte (another chunking) is the
    /// same failure by its signature, so it still settles.
    #[tokio::test]
    async fn one_loop_stopped_at_other_bytes_still_settles() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.recurrent = [4].into();
            b.signed = true;
        }
        for (k, run) in ["run-a", "run-b"].into_iter().enumerate() {
            source.behavior.lock().unwrap().variant = k as u64;
            let mut r = runner(&env, run, &Selection::default());
            r.run_stage(&source).await.unwrap();
        }
        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.status, StageStatus::Cached);
        let e = failed_record(&r, "item-004");
        assert!(e.terminal && e.occurrences == 2, "{e:?}");
        assert_eq!(source.calls(), 11);
    }

    /// Codex review 2: the recurrence threshold is not in the cache key, so a
    /// raised threshold reopens failures settled under a lower one.
    #[tokio::test]
    async fn a_raised_threshold_reopens_settled_failures() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.recurrent = [4].into();
        }
        for run in ["run-a", "run-b"] {
            let mut r = runner(&env, run, &Selection::default());
            r.run_stage(&source).await.unwrap();
        }
        assert_eq!(source.calls(), 11);
        let opts = RunnerOptions {
            terminal_after_repeats: 3,
            ..RunnerOptions::default()
        };
        let mut r = runner_with(
            &env,
            "run-c",
            &Selection::default(),
            opts.clone(),
            CancellationToken::new(),
        );
        assert!(!r.cache_hit(&source));
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_processed), (StageStatus::Ok, 1));
        let e = failed_record(&r, "item-004");
        assert!(e.terminal && e.occurrences == 3, "{e:?}");
        drop(r);
        let mut r = runner_with(
            &env,
            "run-d",
            &Selection::default(),
            opts,
            CancellationToken::new(),
        );
        assert!(r.cache_hit(&source));
        assert_eq!(
            r.run_stage(&source).await.unwrap().status,
            StageStatus::Cached
        );
        assert_eq!(source.calls(), 12);
    }

    /// GLM review M3: over the error-rate limit, in a fresh run directory each
    /// time (the CLI default), the failed item is retried alone and settles; the
    /// rest is never recomputed.
    #[tokio::test]
    async fn over_the_error_rate_fresh_runs_retry_only_the_failure() {
        let env = env();
        let source = SourceStage::new(5);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2].into();
            b.recurrent = [2].into();
        }
        for (run, calls) in [("run-a", 5), ("run-b", 6), ("run-c", 6)] {
            let mut r = runner(&env, run, &Selection::default());
            assert!(
                matches!(
                    r.run_stage(&source).await,
                    Err(RunnerError::ErrorRateExceeded {
                        errors: 1,
                        total: 5,
                        ..
                    })
                ),
                "{run}"
            );
            assert_eq!(source.calls(), calls, "{run}");
        }
        // Once the item recovers (forced, since its failure settled), the stage
        // passes and the rest is restored.
        source.behavior.lock().unwrap().fail.clear();
        let mut opts = RunnerOptions::default();
        opts.force_items
            .insert("source".into(), ["item-002".to_string()].into());
        let mut r = runner_with(
            &env,
            "run-d",
            &Selection::default(),
            opts,
            CancellationToken::new(),
        );
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.items_ok, rep.items_processed), (5, 1));
    }

    /// Codex review round 2: a run directory's old partial does not override a
    /// newer cached output that another run produced.
    #[tokio::test]
    async fn an_old_partial_never_overrides_a_newer_cache() {
        let env = env();
        let source = SourceStage::new(5);
        {
            // A settled failure plus a plain one: run-a keeps a partial, uncached.
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2, 3].into();
            b.terminal = [2].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        assert!(r.run_stage(&source).await.is_err());
        drop(r);
        assert!(env.cache.ls().unwrap().is_empty());
        // Another run, with the items recovered, caches a clean output.
        source.behavior.lock().unwrap().fail.clear();
        let mut r = runner(&env, "run-b", &Selection::default());
        assert_eq!(r.run_stage(&source).await.unwrap().items_ok, 5);
        drop(r);
        // Back in run-a, forcing one item seeds from that cache, not from the old
        // partial and its settled failure.
        let mut opts = RunnerOptions::default();
        opts.force_items
            .insert("source".into(), ["item-004".to_string()].into());
        let mut r = runner_with(
            &env,
            "run-a",
            &Selection::default(),
            opts,
            CancellationToken::new(),
        );
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (5, 0, 1)
        );
    }

    /// Final review (Codex 5): a partial seeded from the current cache entry is
    /// newer than it. A resumed run that recovers an item and then stops before
    /// finalizing keeps that success; the next run neither drops it for the
    /// older cached failure nor settles the recovered item as a failure.
    #[tokio::test]
    async fn a_partial_seeded_from_the_cache_keeps_work_done_after_it() {
        let env = env();
        let source = SourceStage::new(20);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4, 6].into();
            b.recurrent = [4, 6].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(rep.items_error, 2);
        drop(r);
        assert_eq!(env.cache.ls().unwrap().len(), 1, "cached, both unsettled");

        // Item 4 recovers; the run stops at item 6 before finalizing.
        let token = CancellationToken::new();
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [6].into();
            b.cancel_at = Some((6, token.clone()));
        }
        let mut r = runner_with(
            &env,
            "run-b",
            &Selection::default(),
            RunnerOptions::default(),
            token,
        );
        assert!(matches!(
            r.run_stage(&source).await,
            Err(RunnerError::Cancelled { .. })
        ));
        drop(r);
        assert_eq!(source.calls(), 22);

        // Item 4 would fail again if asked: it must not be asked.
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4, 6].into();
            b.cancel_at = None;
        }
        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (19, 1, 1),
            "only item 6 is retried"
        );
        assert_eq!(source.calls(), 23);
        let items = jsonl::read::<Record<u64>>(
            &r.run_dir().artifact_path(SOURCE),
            &SchemaReq::new(SOURCE, 1),
        )
        .unwrap()
        .items;
        let four = items.iter().find(|x| x.id == "item-004").unwrap();
        assert_eq!(four.outcome.result, Some(40), "the recovery survives");
        let e = failed_record(&r, "item-006");
        assert!(e.terminal && e.occurrences == 2, "{e:?}");
        drop(r);

        // The finalized output is the cache now; a fresh run restores it.
        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.status, rep.items_ok, rep.items_error),
            (StageStatus::Cached, 19, 1)
        );
        assert_eq!(source.calls(), 23);
    }

    /// GLM runner review round 1: a partial seeded from an older cache entry
    /// is discarded once another run has finalized a newer entry, even though
    /// it carries a seed marker.
    #[tokio::test]
    async fn a_partial_seeded_from_an_older_cache_entry_is_discarded() {
        let env = env();
        let source = SourceStage::new(20);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4, 6].into();
            b.recurrent = [4, 6].into();
        }
        let mut r = runner(&env, "run-x", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);

        // run-a seeds from that entry, recovers item 4, and stops at item 6.
        let token = CancellationToken::new();
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [6].into();
            b.cancel_at = Some((6, token.clone()));
        }
        let mut r = runner_with(
            &env,
            "run-a",
            &Selection::default(),
            RunnerOptions::default(),
            token,
        );
        assert!(r.run_stage(&source).await.is_err());
        drop(r);

        // run-b finalizes a newer entry: both failures recur and settle.
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4, 6].into();
            b.cancel_at = None;
        }
        let mut r = runner(&env, "run-b", &Selection::default());
        assert_eq!(r.run_stage(&source).await.unwrap().items_error, 2);
        drop(r);
        let calls = source.calls();

        // Back in run-a with one item forced: the seed is the newer entry, and
        // the old partial (marker of the older entry) does not override it.
        let mut opts = RunnerOptions::default();
        opts.force_items
            .insert("source".into(), ["item-010".to_string()].into());
        let mut r = runner_with(
            &env,
            "run-a",
            &Selection::default(),
            opts,
            CancellationToken::new(),
        );
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (18, 2, 1)
        );
        assert_eq!(source.calls(), calls + 1);
        assert!(
            failed_record(&r, "item-004").terminal,
            "the newer entry wins"
        );
    }

    /// Codex runner review round 2: a forced stage stopped before finalizing is
    /// resumed by the next run, not replaced by the entry it superseded.
    #[tokio::test]
    async fn a_forced_stage_stopped_before_finalizing_is_resumed() {
        let env = env();
        let source = SourceStage::new(20);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2].into();
            b.terminal = [2].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        assert_eq!(r.run_stage(&source).await.unwrap().items_error, 1);
        drop(r);

        let token = CancellationToken::new();
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail.clear();
            b.cancel_at = Some((5, token.clone()));
        }
        let sel = Selection {
            force: ["source".to_string()].into(),
            ..Default::default()
        };
        let mut r = runner_with(&env, "run-b", &sel, RunnerOptions::default(), token);
        assert!(r.run_stage(&source).await.is_err());
        drop(r);
        assert_eq!(source.calls(), 26);

        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2].into();
            b.cancel_at = None;
        }
        let mut r = runner(&env, "run-b", &Selection::default());
        assert!(!r.cache_hit(&source));
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (
                rep.status,
                rep.items_ok,
                rep.items_error,
                rep.items_processed
            ),
            (StageStatus::Ok, 20, 0, 15),
            "the forced run continues where it stopped"
        );
        drop(r);
        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_ok), (StageStatus::Cached, 20));
        assert_eq!(source.calls(), 41);
    }

    /// Codex runner review round 2: an output the cache cannot store keeps its
    /// partial, so a recovery is not lost to the older entry.
    #[tokio::test]
    async fn a_failed_cache_write_keeps_the_partial() {
        use std::os::unix::fs::PermissionsExt;
        let env = env();
        let source = SourceStage::new(20);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.recurrent = [4].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);

        // Item 4 recovers, but the entry's directory is read-only.
        source.behavior.lock().unwrap().fail.clear();
        let dirs: Vec<PathBuf> = env
            .cache
            .ls()
            .unwrap()
            .into_iter()
            .map(|e| e.path.parent().unwrap().to_path_buf())
            .collect();
        let set_mode = |mode: u32| {
            for d in &dirs {
                fs_err::set_permissions(d, std::fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        set_mode(0o555);
        let mut r = runner(&env, "run-b", &Selection::default());
        let result = r.run_stage(&source).await;
        set_mode(0o755);
        assert_eq!(result.unwrap().items_ok, 20);
        drop(r);
        assert_eq!(source.calls(), 21);

        // Item 4 would fail again if asked.
        source.behavior.lock().unwrap().fail = [4].into();
        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (20, 0, 0)
        );
        assert_eq!(source.calls(), 21);
    }

    /// Kimi final N5: an over-limit output the cache cannot store does not pin
    /// the run directory to its partial. The next run starts from the cache entry
    /// and retries the unsettled item, which now recovers.
    #[tokio::test]
    async fn an_over_limit_output_the_cache_cannot_store_is_not_resumed() {
        use std::os::unix::fs::PermissionsExt;
        let env = env();
        let source = SourceStage::new(1);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [0].into();
            b.recurrent = [0].into();
        }
        // Over the limit, cached with the item unsettled.
        let mut r = runner(&env, "run-a", &Selection::default());
        assert!(r.run_stage(&source).await.is_err());
        drop(r);
        assert_eq!(source.calls(), 1);

        // The item fails again while the entry's directory is read-only.
        let dirs: Vec<PathBuf> = env
            .cache
            .ls()
            .unwrap()
            .into_iter()
            .map(|e| e.path.parent().unwrap().to_path_buf())
            .collect();
        let set_mode = |mode: u32| {
            for d in &dirs {
                fs_err::set_permissions(d, std::fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        set_mode(0o555);
        let mut r = runner(&env, "run-b", &Selection::default());
        let result = r.run_stage(&source).await;
        set_mode(0o755);
        assert!(matches!(result, Err(RunnerError::ErrorRateExceeded { .. })));
        drop(r);
        assert_eq!(source.calls(), 2);

        // The item recovers: it is asked again, not read back from the partial.
        source.behavior.lock().unwrap().fail.clear();
        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.status, rep.items_ok, rep.items_processed),
            (StageStatus::Ok, 1, 1)
        );
        assert_eq!(source.calls(), 3);
    }

    /// Codex runner review round 2: a resume on a cache miss keeps the marker
    /// with its partial, so the partial is still recognized once the same
    /// entry is back.
    #[tokio::test]
    async fn a_resume_on_a_cache_miss_keeps_the_lineage() {
        let env = env();
        let source = SourceStage::new(20);
        let set = |fail: &[usize], cancel: Option<(usize, CancellationToken)>| {
            let mut b = source.behavior.lock().unwrap();
            b.fail = fail.iter().copied().collect();
            b.recurrent = [4, 6].into();
            b.cancel_at = cancel;
        };
        set(&[4, 6], None);
        let mut r = runner(&env, "run-x", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);
        let resume_stopping_at_6 = |run: &'static str| {
            let token = CancellationToken::new();
            set(&[6], Some((6, token.clone())));
            runner_with(
                &env,
                run,
                &Selection::default(),
                RunnerOptions::default(),
                token,
            )
        };
        // Seeded, item 4 recovered, stopped.
        let mut r = resume_stopping_at_6("run-b");
        assert!(r.run_stage(&source).await.is_err());
        drop(r);
        // The entry is evicted; a resume on the miss stops again.
        for e in env.cache.ls().unwrap() {
            fs_err::remove_file(&e.path).unwrap();
        }
        let mut r = resume_stopping_at_6("run-b");
        assert!(r.run_stage(&source).await.is_err());
        drop(r);
        // Another run puts the same entry back.
        set(&[4, 6], None);
        let mut r = runner(&env, "run-y", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);
        let calls = source.calls();

        let mut r = runner(&env, "run-b", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (rep.items_ok, rep.items_error, rep.items_processed),
            (19, 1, 1),
            "item 4's recovery is kept; only item 6 is asked"
        );
        assert_eq!(source.calls(), calls + 1);
    }

    /// Codex runner review round 1: a forced item that recovers into a seeded
    /// partial survives a crash before finalizing, even though the next run
    /// forces nothing and the cache entry would restore as is.
    #[tokio::test]
    async fn a_forced_recovery_survives_a_crash_before_finalizing() {
        let env = env();
        let source = SourceStage::new(20);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2].into();
            b.terminal = [2].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        assert_eq!(r.run_stage(&source).await.unwrap().items_error, 1);
        drop(r);

        // Items 2 and 5 forced: 2 recovers, the run stops at 5.
        let token = CancellationToken::new();
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail.clear();
            b.cancel_at = Some((5, token.clone()));
        }
        let mut opts = RunnerOptions::default();
        opts.force_items.insert(
            "source".into(),
            ["item-002".to_string(), "item-005".to_string()].into(),
        );
        let mut r = runner_with(&env, "run-b", &Selection::default(), opts, token);
        assert!(matches!(
            r.run_stage(&source).await,
            Err(RunnerError::Cancelled { .. })
        ));
        drop(r);
        assert_eq!(source.calls(), 22);

        // Nothing forced now, and item 2 would fail again if asked.
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2].into();
            b.cancel_at = None;
        }
        let mut r = runner(&env, "run-b", &Selection::default());
        assert!(
            !r.cache_hit(&source),
            "the newer partial is resumed, not the cache entry restored"
        );
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (
                rep.status,
                rep.items_ok,
                rep.items_error,
                rep.items_processed
            ),
            (StageStatus::Ok, 20, 0, 0)
        );
        assert_eq!(source.calls(), 22, "the recovery is kept, not asked again");
        drop(r);

        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_ok), (StageStatus::Cached, 20));
        assert_eq!(source.calls(), 22);
    }

    /// Codex review round 2: a settled output over the limit is not a hit that
    /// `cache_hit` promises and `run_stage` then refuses.
    #[tokio::test]
    async fn cache_hit_agrees_with_an_over_limit_restore() {
        let env = env();
        let source = SourceStage::new(5);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [2].into();
            b.terminal = [2].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        assert!(r.run_stage(&source).await.is_err());
        drop(r);
        let mut r = runner(&env, "run-b", &Selection::default());
        assert!(!r.cache_hit(&source));
        assert!(matches!(
            r.run_stage(&source).await,
            Err(RunnerError::ErrorRateExceeded { .. })
        ));
        assert_eq!(source.calls(), 5, "the settled output is not recomputed");
    }

    /// Codex 6: a one-item stage over the error-rate limit used to keep its failed
    /// partial on every resume. The item is retried, and settles only when the
    /// same failure comes back.
    #[tokio::test]
    async fn a_one_item_stage_over_the_error_rate_retries_on_resume() {
        let env = env();
        let source = SourceStage::new(1);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [0].into();
            b.recurrent = [0].into();
        }
        for _ in 0..2 {
            let mut r = runner(&env, "run-a", &Selection::default());
            assert!(matches!(
                r.run_stage(&source).await,
                Err(RunnerError::ErrorRateExceeded { .. })
            ));
        }
        assert_eq!(source.calls(), 2, "resume retried the failed item");
        let mut r = runner(&env, "run-a", &Selection::default());
        assert!(r.run_stage(&source).await.is_err());
        assert_eq!(source.calls(), 2, "two identical failures settle it");
        drop(r);

        // Forcing the one item retries it without --force-stage.
        source.behavior.lock().unwrap().fail.clear();
        let mut opts = RunnerOptions::default();
        opts.force_items
            .insert("source".into(), ["item-000".to_string()].into());
        let mut r = runner_with(
            &env,
            "run-a",
            &Selection::default(),
            opts,
            CancellationToken::new(),
        );
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.items_ok, rep.items_processed), (1, 1));
        assert_eq!(source.calls(), 3);
    }

    #[tokio::test]
    async fn forcing_one_item_recomputes_only_that_item_of_a_cached_stage() {
        let env = env();
        let source = SourceStage::new(10);
        {
            let mut b = source.behavior.lock().unwrap();
            b.fail = [4].into();
            b.terminal = [4].into();
        }
        let mut r = runner(&env, "run-a", &Selection::default());
        r.run_stage(&source).await.unwrap();
        drop(r);
        source.behavior.lock().unwrap().fail.clear();
        let mut opts = RunnerOptions::default();
        opts.force_items
            .insert("source".into(), ["item-004".to_string()].into());
        let mut r = runner_with(
            &env,
            "run-b",
            &Selection::default(),
            opts,
            CancellationToken::new(),
        );
        assert!(
            !r.cache_hit(&source),
            "a forced item needs the stage to run"
        );
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!(
            (
                rep.status,
                rep.items_ok,
                rep.items_error,
                rep.items_processed
            ),
            (StageStatus::Ok, 10, 0, 1)
        );
        assert_eq!(source.calls(), 11);
        let mut r = runner(&env, "run-c", &Selection::default());
        let rep = r.run_stage(&source).await.unwrap();
        assert_eq!((rep.status, rep.items_ok), (StageStatus::Cached, 10));
    }

    #[test]
    fn cancellation_and_timeouts_are_never_terminal() {
        for code in [ErrorCode::Timeout, ErrorCode::Cancelled] {
            let e = ErrorInfo::new(code, "x").terminal_if_repeated();
            assert!(!e.recurrent && !e.terminal, "{e:?}");
        }
        let json = serde_json::to_value(ErrorInfo::new(ErrorCode::Io, "x").terminal_if_repeated())
            .unwrap();
        assert_eq!(json.get("recurrent"), Some(&serde_json::Value::Bool(true)));
        assert!(
            json.get("occurrences").is_none(),
            "absent when zero: {json}"
        );
        assert!(
            ErrorInfo::new(ErrorCode::ModelRequest, "x")
                .terminal()
                .terminal
        );
        assert!(!ErrorInfo::new(ErrorCode::Timeout, "x").terminal().terminal);
        assert!(
            !ErrorInfo::new(ErrorCode::Cancelled, "x")
                .terminal()
                .terminal
        );
        let json = serde_json::to_value(ErrorInfo::new(ErrorCode::Io, "x")).unwrap();
        assert!(json.get("terminal").is_none(), "absent when false: {json}");
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
