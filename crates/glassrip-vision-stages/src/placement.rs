//! Model-server supervision around vision requests (spec 5.3, 8.1).
//!
//! One [`PlacementMonitor`] is shared by every model stage of a run:
//!
//! - Preflight runs once before the first request and records the model digest.
//! - After every `check_every` completed requests it checks GPU placement and
//!   re-resolves the digest. A spill pauses new submissions (requests in flight
//!   finish) and polls until the model is fully on the GPU again; a spill that
//!   outlasts `max_pause` aborts, as does a changed digest.
//! - [`PlacementMonitor::begin_stage`] re-checks the digest before each model stage.
//! - [`PlacementMonitor::infer_typed`] retries a request once after a transient
//!   failure, first waiting for the server (`/api/version`, up to
//!   `server_wait`) and re-running preflight with the digest check, as for a
//!   server restart. Every reply is checked against the context budget.
//!
//! An abort is sticky: every later request fails with the same error, and the
//! pipeline reports the stage as aborted rather than mixing model versions.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_vision::{
    OllamaBackend, Placement, RawResponse, VisionBackend, VisionClient, VisionError, VisionRequest,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Source of placement and identity information.
#[async_trait]
pub trait PlacementProbe: Send + Sync {
    /// Load the model if needed and check placement; errors on spill unless allowed.
    async fn preflight(&self) -> Result<Placement, VisionError>;
    /// Current placement; `None` when the model is not loaded.
    async fn placement(&self) -> Result<Option<Placement>, VisionError>;
    /// Current model digest as the server reports it.
    async fn digest(&self) -> Result<String, VisionError>;
    /// True when the server answers (for example `/api/version`).
    async fn server_up(&self) -> bool;
}

#[async_trait]
impl PlacementProbe for OllamaBackend {
    async fn preflight(&self) -> Result<Placement, VisionError> {
        VisionBackend::preflight(self).await
    }
    async fn placement(&self) -> Result<Option<Placement>, VisionError> {
        OllamaBackend::placement(self).await
    }
    async fn digest(&self) -> Result<String, VisionError> {
        self.resolve_digest().await
    }
    async fn server_up(&self) -> bool {
        self.server_version().await.is_ok()
    }
}

/// Placement that is always fully on the GPU with a fixed digest (replay and tests).
pub struct StaticProbe;

#[async_trait]
impl PlacementProbe for StaticProbe {
    async fn preflight(&self) -> Result<Placement, VisionError> {
        Ok(on_gpu())
    }
    async fn placement(&self) -> Result<Option<Placement>, VisionError> {
        Ok(Some(on_gpu()))
    }
    async fn digest(&self) -> Result<String, VisionError> {
        Ok("static".into())
    }
    async fn server_up(&self) -> bool {
        true
    }
}

fn on_gpu() -> Placement {
    Placement {
        model: "static".into(),
        size_bytes: 1,
        size_vram_bytes: 1,
        fully_on_gpu: true,
        context_length: None,
        concurrency_hint: 1,
    }
}

/// Kind of check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    Preflight,
    MidRun,
    /// Poll while paused for a spill.
    Paused,
    /// Digest check before a model stage.
    StageStart,
    /// Wait-for-server and re-preflight after a transient failure.
    Recovery,
}

/// One recorded check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementCheck {
    pub kind: CheckKind,
    /// Completed requests when the check ran.
    pub after_requests: usize,
    pub fully_on_gpu: Option<bool>,
    pub size_bytes: Option<u64>,
    pub size_vram_bytes: Option<u64>,
    pub digest: Option<String>,
    pub error: Option<String>,
    /// True when submissions were paused because of this check.
    pub paused: bool,
}

/// Monitor settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonitorConfig {
    pub check_every: usize,
    pub poll_interval: Duration,
    /// A spill lasting longer than this aborts the stage (spec 5.3).
    pub max_pause: Duration,
    /// How long to wait for a restarted server (spec 8.1: 120 s).
    pub server_wait: Duration,
    /// Fixed context size of every request; replies are checked against it.
    pub num_ctx: u32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_every: 10,
            poll_interval: Duration::from_secs(2),
            max_pause: Duration::from_secs(600),
            server_wait: Duration::from_secs(120),
            num_ctx: 8192,
        }
    }
}

/// Check a reply against the context budget: generation must not have stopped
/// on the length limit, and `prompt_eval_count + eval_count` must fit
/// `num_ctx`. Ollama reports only newly evaluated prompt tokens (a reused,
/// cached prefix is excluded), so the sum is a lower bound on usage and a
/// small `prompt_eval_count` is not an error.
pub fn check_usage(raw: &RawResponse, num_ctx: u32) -> Result<(), ErrorInfo> {
    if raw.done_reason.as_deref() == Some("length") {
        return Err(truncated_info(&raw.raw_text));
    }
    let prompt = raw.prompt_eval_count.unwrap_or(0);
    let output = raw.eval_count.unwrap_or(0);
    if prompt.saturating_add(output) > num_ctx {
        return Err(ErrorInfo::new(
            ErrorCode::ModelRequest,
            format!("prompt {prompt} + output {output} tokens exceed num_ctx {num_ctx}"),
        ));
    }
    Ok(())
}

fn truncated_info(raw_text: &str) -> ErrorInfo {
    ErrorInfo::new(
        ErrorCode::ModelRequest,
        "generation stopped at the output limit (done_reason length)",
    )
    .with_raw_text(raw_text.to_string())
}

/// Why [`PlacementMonitor::infer_typed_detailed`] failed.
#[derive(Debug, Clone, PartialEq)]
pub enum InferFailure {
    /// The reply stopped at the output limit (`done_reason: length`). The caller
    /// may retry with a smaller output (see `board_read`'s compact retry).
    Truncated(ErrorInfo),
    /// Any other failure.
    Other(ErrorInfo),
}

impl InferFailure {
    /// The item error.
    pub fn into_info(self) -> ErrorInfo {
        match self {
            Self::Truncated(e) | Self::Other(e) => e,
        }
    }
}

impl From<ErrorInfo> for InferFailure {
    fn from(e: ErrorInfo) -> Self {
        Self::Other(e)
    }
}

/// Preflight, periodic checks, recovery, and abort state shared by every model
/// stage of a run.
pub struct PlacementMonitor {
    probe: Arc<dyn PlacementProbe>,
    client: VisionClient,
    cfg: MonitorConfig,
    completed: AtomicUsize,
    preflight: tokio::sync::OnceCell<Result<Placement, String>>,
    checking: tokio::sync::Mutex<()>,
    recovering: tokio::sync::Mutex<()>,
    digest: Mutex<Option<String>>,
    abort: Mutex<Option<ErrorInfo>>,
    checks: Mutex<Vec<PlacementCheck>>,
}

impl PlacementMonitor {
    pub fn new(probe: Arc<dyn PlacementProbe>, client: VisionClient, cfg: MonitorConfig) -> Self {
        Self {
            probe,
            client,
            cfg,
            completed: AtomicUsize::new(0),
            preflight: tokio::sync::OnceCell::new(),
            checking: tokio::sync::Mutex::new(()),
            recovering: tokio::sync::Mutex::new(()),
            digest: Mutex::new(None),
            abort: Mutex::new(None),
            checks: Mutex::new(Vec::new()),
        }
    }

    pub fn client(&self) -> &VisionClient {
        &self.client
    }

    pub fn config(&self) -> &MonitorConfig {
        &self.cfg
    }

    fn record(&self, c: PlacementCheck) {
        self.checks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(c);
    }

    /// Every check so far.
    pub fn checks(&self) -> Vec<PlacementCheck> {
        self.checks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Completed requests so far.
    pub fn completed(&self) -> usize {
        self.completed.load(Ordering::SeqCst)
    }

    /// The abort error, once the run must stop using the model.
    pub fn abort_error(&self) -> Option<ErrorInfo> {
        self.abort
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn set_abort(&self, e: ErrorInfo) -> ErrorInfo {
        let mut a = self.abort.lock().unwrap_or_else(PoisonError::into_inner);
        a.get_or_insert(e).clone()
    }

    fn check_record(
        kind: CheckKind,
        after: usize,
        r: &Result<Option<Placement>, String>,
    ) -> PlacementCheck {
        let p = r.as_ref().ok().and_then(Option::as_ref);
        PlacementCheck {
            kind,
            after_requests: after,
            fully_on_gpu: p.map(|p| p.fully_on_gpu),
            size_bytes: p.map(|p| p.size_bytes),
            size_vram_bytes: p.map(|p| p.size_vram_bytes),
            digest: None,
            error: r.as_ref().err().cloned(),
            paused: false,
        }
    }

    /// Re-resolve the digest and compare it with the one seen at preflight.
    /// A mismatch (or [`VisionError::DigestChanged`]) aborts.
    pub async fn check_digest(&self, kind: CheckKind) -> Result<(), ErrorInfo> {
        let r = self.probe.digest().await;
        let known = self
            .digest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let outcome = match (&r, known) {
            (Ok(d), Some(k)) if *d != k => Err(VisionError::DigestChanged {
                model: self.client.backend().id().model,
                previous: k,
                current: d.clone(),
            }),
            (Ok(d), None) => {
                *self.digest.lock().unwrap_or_else(PoisonError::into_inner) = Some(d.clone());
                Ok(())
            }
            (Ok(_), Some(_)) => Ok(()),
            (Err(e @ VisionError::DigestChanged { .. }), _) => Err(e.clone_digest()),
            // A failed lookup is not proof of a change; the next check retries.
            (Err(_), _) => Ok(()),
        };
        self.record(PlacementCheck {
            kind,
            after_requests: self.completed(),
            fully_on_gpu: None,
            size_bytes: None,
            size_vram_bytes: None,
            digest: r.as_ref().ok().cloned(),
            error: r.as_ref().err().map(ToString::to_string),
            paused: false,
        });
        outcome.map_err(|e| {
            tracing::error!(error = %e, "model digest changed; aborting");
            self.set_abort(ErrorInfo::new(ErrorCode::ModelRequest, e.to_string()))
        })
    }

    /// Run the preflight once (recording the digest); every caller gets its result.
    pub async fn ensure_preflight(&self) -> Result<(), ErrorInfo> {
        if let Some(e) = self.abort_error() {
            return Err(e);
        }
        let r = self
            .preflight
            .get_or_init(|| async {
                let r = self.probe.preflight().await.map_err(|e| e.to_string());
                let as_opt = r.clone().map(Some);
                self.record(Self::check_record(CheckKind::Preflight, 0, &as_opt));
                r
            })
            .await;
        r.as_ref().map(|_| ()).map_err(|m| {
            ErrorInfo::new(ErrorCode::ModelRequest, format!("preflight failed: {m}"))
        })?;
        if self
            .digest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_none()
        {
            self.check_digest(CheckKind::Preflight).await?;
        }
        Ok(())
    }

    /// Check before a model stage starts: preflight (once) and the digest.
    pub async fn begin_stage(&self) -> Result<(), ErrorInfo> {
        self.ensure_preflight().await?;
        self.check_digest(CheckKind::StageStart).await
    }

    /// Wait for the server, re-run preflight, and check the digest (spec 8.1).
    /// Concurrent callers share one recovery.
    async fn recover(&self) -> Result<(), ErrorInfo> {
        let _guard = self.recovering.lock().await;
        let started = Instant::now();
        let mut up = self.probe.server_up().await;
        while !up && started.elapsed() < self.cfg.server_wait {
            tokio::time::sleep(self.cfg.poll_interval).await;
            up = self.probe.server_up().await;
        }
        let r = if up {
            self.probe
                .preflight()
                .await
                .map(Some)
                .map_err(|e| e.to_string())
        } else {
            Err(format!(
                "model server did not answer within {:.0} s",
                self.cfg.server_wait.as_secs_f64()
            ))
        };
        self.record(Self::check_record(
            CheckKind::Recovery,
            self.completed(),
            &r,
        ));
        if let Err(m) = r {
            return Err(ErrorInfo::new(
                ErrorCode::ModelRequest,
                format!("recovery failed: {m}"),
            ));
        }
        self.check_digest(CheckKind::Recovery).await
    }

    /// Send one request: preflight, the request, one recovery-and-retry after a
    /// transient failure, the periodic checks, and the context-budget check.
    pub async fn infer_typed<T: DeserializeOwned>(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<(T, RawResponse), ErrorInfo> {
        self.infer_typed_detailed(request, cancel)
            .await
            .map_err(InferFailure::into_info)
    }

    /// [`Self::infer_typed`], reporting a reply cut off at the output limit as
    /// [`InferFailure::Truncated`] (the request is not retried here).
    pub async fn infer_typed_detailed<T: DeserializeOwned>(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<(T, RawResponse), InferFailure> {
        self.ensure_preflight().await?;
        let mut retried = false;
        let result = loop {
            let r = self
                .client
                .infer_typed::<T>(request.clone(), cancel.clone())
                .await;
            match r {
                Err(e) if e.is_transient() && !retried => {
                    tracing::warn!(error = %e, "transient model failure; waiting for the server");
                    retried = true;
                    self.recover().await?;
                }
                r => break r,
            }
        };
        self.after_request().await;
        if let Some(e) = self.abort_error() {
            return Err(e.into());
        }
        let (value, raw) = result.map_err(|e| {
            let info = vision_error_info(&e);
            if e.is_truncated() {
                InferFailure::Truncated(info)
            } else {
                InferFailure::Other(info)
            }
        })?;
        if raw.done_reason.as_deref() == Some("length") {
            return Err(InferFailure::Truncated(truncated_info(&raw.raw_text)));
        }
        check_usage(&raw, self.cfg.num_ctx)?;
        Ok((value, raw))
    }

    /// Count a completed request and run the periodic checks.
    pub async fn after_request(&self) {
        let n = self.completed.fetch_add(1, Ordering::SeqCst) + 1;
        if self.cfg.check_every == 0 || !n.is_multiple_of(self.cfg.check_every) {
            return;
        }
        // One checker at a time; a concurrent trigger is covered by the running one.
        let Ok(_guard) = self.checking.try_lock() else {
            return;
        };
        if self.check_digest(CheckKind::MidRun).await.is_err() {
            return;
        }
        let r = self.probe.placement().await.map_err(|e| e.to_string());
        let spilling = matches!(&r, Ok(Some(p)) if !p.fully_on_gpu);
        let mut rec = Self::check_record(CheckKind::MidRun, n, &r);
        rec.paused = spilling;
        self.record(rec);
        if !spilling {
            return;
        }
        tracing::warn!(
            after = n,
            "model spilling out of VRAM; pausing new requests"
        );
        self.client.pause();
        let started = Instant::now();
        loop {
            tokio::time::sleep(self.cfg.poll_interval).await;
            let r = self.probe.placement().await.map_err(|e| e.to_string());
            let ok = matches!(&r, Ok(Some(p)) if p.fully_on_gpu);
            let mut rec = Self::check_record(CheckKind::Paused, self.completed(), &r);
            rec.paused = !ok;
            self.record(rec);
            if ok {
                break;
            }
            if started.elapsed() >= self.cfg.max_pause {
                let e = self.set_abort(ErrorInfo::new(
                    ErrorCode::ModelRequest,
                    format!(
                        "model still spilling out of VRAM after {:.0} s paused",
                        self.cfg.max_pause.as_secs_f64()
                    ),
                ));
                tracing::error!(error = %e, "aborting");
                break;
            }
        }
        // Resume so paused requests reach the abort check and fail fast.
        self.client.resume();
    }
}

trait CloneDigest {
    fn clone_digest(&self) -> VisionError;
}

impl CloneDigest for VisionError {
    fn clone_digest(&self) -> VisionError {
        match self {
            VisionError::DigestChanged {
                model,
                previous,
                current,
            } => VisionError::DigestChanged {
                model: model.clone(),
                previous: previous.clone(),
                current: current.clone(),
            },
            other => VisionError::Protocol(other.to_string()),
        }
    }
}

/// Map a vision error to an item error.
pub fn vision_error_info(e: &VisionError) -> ErrorInfo {
    match e {
        VisionError::Cancelled => ErrorInfo::new(ErrorCode::Cancelled, e.to_string()),
        VisionError::Timeout { .. } => ErrorInfo::new(ErrorCode::Timeout, e.to_string()),
        VisionError::SchemaInvalid { raw_text, .. } => {
            ErrorInfo::new(ErrorCode::SchemaParse, e.to_string()).with_raw_text(raw_text.clone())
        }
        VisionError::Decode { .. } => ErrorInfo::new(ErrorCode::SchemaParse, e.to_string()),
        VisionError::Truncated { raw_text, .. } => {
            ErrorInfo::new(ErrorCode::ModelRequest, e.to_string()).with_raw_text(raw_text.clone())
        }
        VisionError::Image(_) | VisionError::ImageTooManyTokens { .. } => {
            ErrorInfo::new(ErrorCode::InvalidInput, e.to_string())
        }
        _ => ErrorInfo::new(ErrorCode::ModelRequest, e.to_string()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use glassrip_vision::{BackendId, Durations};
    use std::sync::atomic::AtomicBool;

    struct Flaky {
        spilling: AtomicBool,
        polls: AtomicUsize,
        recover_after: usize,
    }

    #[async_trait]
    impl PlacementProbe for Flaky {
        async fn preflight(&self) -> Result<Placement, VisionError> {
            Ok(on_gpu())
        }
        async fn placement(&self) -> Result<Option<Placement>, VisionError> {
            let n = self.polls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                self.spilling.store(true, Ordering::SeqCst);
            } else if n >= self.recover_after {
                self.spilling.store(false, Ordering::SeqCst);
            }
            let mut p = on_gpu();
            p.fully_on_gpu = !self.spilling.load(Ordering::SeqCst);
            Ok(Some(p))
        }
        async fn digest(&self) -> Result<String, VisionError> {
            Ok("d1".into())
        }
        async fn server_up(&self) -> bool {
            true
        }
    }

    struct Nop;

    #[async_trait]
    impl VisionBackend for Nop {
        fn id(&self) -> BackendId {
            BackendId {
                backend: "nop".into(),
                model: "nop".into(),
                digest: None,
                server_version: None,
            }
        }
        async fn preflight(&self) -> glassrip_vision::Result<Placement> {
            Ok(on_gpu())
        }
        async fn infer(
            &self,
            _r: VisionRequest,
            _c: CancellationToken,
        ) -> glassrip_vision::Result<RawResponse> {
            Err(VisionError::Config("unused".into()))
        }
    }

    fn cfg(max_pause: Duration) -> MonitorConfig {
        MonitorConfig {
            check_every: 3,
            poll_interval: Duration::from_millis(5),
            max_pause,
            ..MonitorConfig::default()
        }
    }

    #[tokio::test]
    async fn pauses_on_spill_and_resumes() {
        let client = VisionClient::new(Arc::new(Nop), 2).unwrap();
        let probe = Arc::new(Flaky {
            spilling: AtomicBool::new(false),
            polls: AtomicUsize::new(0),
            recover_after: 2,
        });
        let m = PlacementMonitor::new(probe, client.clone(), cfg(Duration::from_secs(5)));
        m.ensure_preflight().await.unwrap();
        for _ in 0..3 {
            m.after_request().await;
        }
        let checks = m.checks();
        assert_eq!(checks[0].kind, CheckKind::Preflight);
        let mid: Vec<&PlacementCheck> = checks
            .iter()
            .filter(|c| c.kind == CheckKind::MidRun && c.fully_on_gpu.is_some())
            .collect();
        assert!(mid[0].paused);
        assert!(checks.last().unwrap().fully_on_gpu.unwrap());
        assert!(!client.is_paused());
        assert!(m.abort_error().is_none());
    }

    #[tokio::test]
    async fn spill_past_max_pause_aborts() {
        let client = VisionClient::new(Arc::new(Nop), 2).unwrap();
        let probe = Arc::new(Flaky {
            spilling: AtomicBool::new(false),
            polls: AtomicUsize::new(0),
            recover_after: usize::MAX,
        });
        let m = PlacementMonitor::new(probe, client, cfg(Duration::from_millis(30)));
        m.ensure_preflight().await.unwrap();
        for _ in 0..3 {
            m.after_request().await;
        }
        let e = m.abort_error().unwrap();
        assert!(e.message.contains("still spilling"), "{e}");
        assert!(m.ensure_preflight().await.is_err());
    }

    #[tokio::test]
    async fn preflight_failure_is_shared() {
        struct Bad;
        #[async_trait]
        impl PlacementProbe for Bad {
            async fn preflight(&self) -> Result<Placement, VisionError> {
                Err(VisionError::ModelNotFound { model: "x".into() })
            }
            async fn placement(&self) -> Result<Option<Placement>, VisionError> {
                Ok(None)
            }
            async fn digest(&self) -> Result<String, VisionError> {
                Ok("d".into())
            }
            async fn server_up(&self) -> bool {
                true
            }
        }
        let client = VisionClient::new(Arc::new(Nop), 1).unwrap();
        let m = PlacementMonitor::new(Arc::new(Bad), client, MonitorConfig::default());
        assert!(m.ensure_preflight().await.is_err());
        assert!(m.ensure_preflight().await.is_err());
        assert_eq!(m.checks().len(), 1);
    }

    fn raw(prompt: Option<u32>, eval: Option<u32>, done: &str) -> RawResponse {
        RawResponse {
            raw_text: "{}".into(),
            json: serde_json::json!({}),
            prompt_eval_count: prompt,
            eval_count: eval,
            durations: Durations::default(),
            attempts: 1,
            repaired: false,
            done_reason: Some(done.into()),
        }
    }

    #[test]
    fn usage_checks_catch_truncation_but_not_cache_hits() {
        assert!(check_usage(&raw(Some(400), Some(200), "stop"), 8192).is_ok());
        assert!(check_usage(&raw(Some(400), Some(200), "length"), 8192).is_err());
        assert!(check_usage(&raw(Some(8000), Some(300), "stop"), 8192).is_err());
        // A cache hit reports few prompt tokens; that is not an error.
        assert!(check_usage(&raw(Some(50), Some(300), "stop"), 8192).is_ok());
        assert!(check_usage(&raw(None, None, "stop"), 8192).is_ok());
    }
}
