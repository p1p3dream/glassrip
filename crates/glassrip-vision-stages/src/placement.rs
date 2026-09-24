//! GPU placement checks around model requests (spec 5.3).
//!
//! A preflight runs once before the first request of a run. After every
//! `check_every` completed requests the monitor asks the server where the model
//! sits; if it has started spilling out of VRAM, the shared [`VisionClient`]
//! pauses new submissions (requests in flight finish) and polls until the model
//! is fully on the GPU again, then resumes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_vision::{OllamaBackend, Placement, VisionBackend, VisionClient, VisionError};
use serde::{Deserialize, Serialize};

/// Source of placement information.
#[async_trait]
pub trait PlacementProbe: Send + Sync {
    /// Load the model if needed and check placement; errors on spill unless allowed.
    async fn preflight(&self) -> Result<Placement, VisionError>;
    /// Current placement; `None` when the model is not loaded.
    async fn placement(&self) -> Result<Option<Placement>, VisionError>;
}

#[async_trait]
impl PlacementProbe for OllamaBackend {
    async fn preflight(&self) -> Result<Placement, VisionError> {
        VisionBackend::preflight(self).await
    }
    async fn placement(&self) -> Result<Option<Placement>, VisionError> {
        OllamaBackend::placement(self).await
    }
}

/// Placement that is always fully on the GPU (replay and tests).
pub struct StaticProbe;

#[async_trait]
impl PlacementProbe for StaticProbe {
    async fn preflight(&self) -> Result<Placement, VisionError> {
        Ok(on_gpu())
    }
    async fn placement(&self) -> Result<Option<Placement>, VisionError> {
        Ok(Some(on_gpu()))
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

/// Kind of placement check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    Preflight,
    MidRun,
    /// Poll while paused for a spill.
    Paused,
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
    pub error: Option<String>,
    /// True when submissions were paused because of this check.
    pub paused: bool,
}

/// Monitor settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonitorConfig {
    pub check_every: usize,
    pub poll_interval: Duration,
    pub max_pause: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_every: 10,
            poll_interval: Duration::from_secs(2),
            max_pause: Duration::from_secs(600),
        }
    }
}

/// Preflight plus mid-run placement checks, shared by every model stage of a run.
pub struct PlacementMonitor {
    probe: Arc<dyn PlacementProbe>,
    client: VisionClient,
    cfg: MonitorConfig,
    completed: AtomicUsize,
    preflight: tokio::sync::OnceCell<Result<Placement, String>>,
    checking: tokio::sync::Mutex<()>,
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
            checks: Mutex::new(Vec::new()),
        }
    }

    pub fn client(&self) -> &VisionClient {
        &self.client
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
            error: r.as_ref().err().cloned(),
            paused: false,
        }
    }

    /// Run the preflight once; every caller gets its result.
    pub async fn ensure_preflight(&self) -> Result<(), ErrorInfo> {
        let r = self
            .preflight
            .get_or_init(|| async {
                let r = self.probe.preflight().await.map_err(|e| e.to_string());
                let as_opt = r.clone().map(Some);
                self.record(Self::check_record(CheckKind::Preflight, 0, &as_opt));
                r
            })
            .await;
        r.as_ref()
            .map(|_| ())
            .map_err(|m| ErrorInfo::new(ErrorCode::ModelRequest, format!("preflight failed: {m}")))
    }

    /// Call after each completed request (success or failure).
    pub async fn after_request(&self) {
        let n = self.completed.fetch_add(1, Ordering::SeqCst) + 1;
        if self.cfg.check_every == 0 || !n.is_multiple_of(self.cfg.check_every) {
            return;
        }
        // One checker at a time; a concurrent trigger is covered by the running one.
        let Ok(_guard) = self.checking.try_lock() else {
            return;
        };
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
            if ok || started.elapsed() >= self.cfg.max_pause {
                if !ok {
                    tracing::error!("model still spilling after the maximum pause; resuming");
                }
                break;
            }
        }
        self.client.resume();
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
    use glassrip_vision::{BackendId, RawResponse, VisionRequest};
    use std::sync::atomic::AtomicBool;
    use tokio_util::sync::CancellationToken;

    struct Flaky {
        spilling: AtomicBool,
        polls: AtomicUsize,
    }

    #[async_trait]
    impl PlacementProbe for Flaky {
        async fn preflight(&self) -> Result<Placement, VisionError> {
            Ok(on_gpu())
        }
        async fn placement(&self) -> Result<Option<Placement>, VisionError> {
            let n = self.polls.fetch_add(1, Ordering::SeqCst);
            // Spill on the first mid-run check, recover on the second poll.
            if n == 0 {
                self.spilling.store(true, Ordering::SeqCst);
            } else if n >= 2 {
                self.spilling.store(false, Ordering::SeqCst);
            }
            let mut p = on_gpu();
            p.fully_on_gpu = !self.spilling.load(Ordering::SeqCst);
            Ok(Some(p))
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

    #[tokio::test]
    async fn pauses_on_spill_and_resumes() {
        let client = VisionClient::new(Arc::new(Nop), 2).unwrap();
        let probe = Arc::new(Flaky {
            spilling: AtomicBool::new(false),
            polls: AtomicUsize::new(0),
        });
        let m = PlacementMonitor::new(
            probe,
            client.clone(),
            MonitorConfig {
                check_every: 3,
                poll_interval: Duration::from_millis(5),
                max_pause: Duration::from_secs(5),
            },
        );
        m.ensure_preflight().await.unwrap();
        for _ in 0..3 {
            m.after_request().await;
        }
        let checks = m.checks();
        assert_eq!(checks[0].kind, CheckKind::Preflight);
        assert_eq!(checks[1].kind, CheckKind::MidRun);
        assert!(checks[1].paused);
        assert!(checks.last().unwrap().fully_on_gpu.unwrap());
        assert!(!client.is_paused());
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
        }
        let client = VisionClient::new(Arc::new(Nop), 1).unwrap();
        let m = PlacementMonitor::new(Arc::new(Bad), client, MonitorConfig::default());
        assert!(m.ensure_preflight().await.is_err());
        assert!(m.ensure_preflight().await.is_err());
        assert_eq!(m.checks().len(), 1);
    }
}
