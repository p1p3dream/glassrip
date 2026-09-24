//! Bounded, pausable request submission.
//!
//! [`VisionClient`] keeps at most `max_in_flight` requests at the server (set it
//! to the server slot count) and has a pause gate: while paused, no new request
//! starts, and requests already in flight finish normally. The gate is meant for
//! mid-run VRAM spill: pause when `/api/ps` shows spilling, resume once the
//! competing GPU work is done.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use tokio::sync::{watch, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::backend::{RawResponse, VisionBackend, VisionRequest};
use crate::error::{Result, VisionError};

/// Concurrency-bounded front end to a [`VisionBackend`]. Cheap to clone.
#[derive(Clone)]
pub struct VisionClient {
    backend: Arc<dyn VisionBackend>,
    permits: Arc<Semaphore>,
    max_in_flight: usize,
    gate: Arc<watch::Sender<bool>>,
}

impl std::fmt::Debug for VisionClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionClient")
            .field("backend", &self.backend.id())
            .field("max_in_flight", &self.max_in_flight)
            .field("paused", &self.is_paused())
            .finish()
    }
}

impl VisionClient {
    /// `max_in_flight` must be at least 1; use the server slot count (4 for qwen2.5vl:7b).
    pub fn new(backend: Arc<dyn VisionBackend>, max_in_flight: usize) -> Result<Self> {
        if max_in_flight == 0 {
            return Err(VisionError::Config("max_in_flight must be at least 1".into()));
        }
        let (gate, _) = watch::channel(true);
        Ok(Self {
            backend,
            permits: Arc::new(Semaphore::new(max_in_flight)),
            max_in_flight,
            gate: Arc::new(gate),
        })
    }

    pub fn backend(&self) -> &Arc<dyn VisionBackend> {
        &self.backend
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Requests currently holding a slot.
    pub fn in_flight(&self) -> usize {
        self.max_in_flight
            .saturating_sub(self.permits.available_permits())
    }

    /// Stop starting new requests.
    pub fn pause(&self) {
        self.gate.send_replace(false);
    }

    /// Allow new requests again.
    pub fn resume(&self) {
        self.gate.send_replace(true);
    }

    pub fn is_paused(&self) -> bool {
        !*self.gate.borrow()
    }

    async fn wait_open(&self, cancel: &CancellationToken) -> Result<()> {
        let mut rx = self.gate.subscribe();
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(VisionError::Cancelled),
            r = rx.wait_for(|open| *open) => r
                .map(|_| ())
                .map_err(|_| VisionError::Protocol("pause gate closed".into())),
        }
    }

    /// Wait for the gate and a slot, then run the request.
    pub async fn infer(&self, request: VisionRequest, cancel: CancellationToken) -> Result<RawResponse> {
        let permit = loop {
            self.wait_open(&cancel).await?;
            let permit = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(VisionError::Cancelled),
                p = self.permits.clone().acquire_owned() => p
                    .map_err(|_| VisionError::Protocol("request semaphore closed".into()))?,
            };
            // The gate may have closed while this task waited for a slot.
            if !self.is_paused() {
                break permit;
            }
            drop(permit);
        };
        let result = self.backend.infer(request, cancel).await;
        drop(permit);
        result
    }

    /// Run the request and decode the validated output as `T`.
    pub async fn infer_typed<T: DeserializeOwned>(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<(T, RawResponse)> {
        let raw = self.infer(request, cancel).await?;
        let value = raw.decode::<T>()?;
        Ok((value, raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendId, Durations, GenerationOptions, Placement};
    use crate::image_prep::EncodedImage;
    use async_trait::async_trait;
    use image::{DynamicImage, RgbImage};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct Empty {}

    #[derive(Default)]
    struct CountingBackend {
        current: AtomicUsize,
        peak: AtomicUsize,
        done: AtomicUsize,
    }

    #[async_trait]
    impl VisionBackend for CountingBackend {
        fn id(&self) -> BackendId {
            BackendId {
                backend: "fake".into(),
                model: "fake".into(),
                digest: None,
                server_version: None,
            }
        }
        async fn preflight(&self) -> Result<Placement> {
            Err(VisionError::Config("unused".into()))
        }
        async fn infer(&self, _r: VisionRequest, _c: CancellationToken) -> Result<RawResponse> {
            let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(30)).await;
            self.current.fetch_sub(1, Ordering::SeqCst);
            self.done.fetch_add(1, Ordering::SeqCst);
            Ok(RawResponse {
                raw_text: "{}".into(),
                json: serde_json::json!({}),
                prompt_eval_count: None,
                eval_count: None,
                durations: Durations::default(),
                attempts: 1,
                repaired: false,
                done_reason: None,
            })
        }
    }

    fn request() -> VisionRequest {
        let img = DynamicImage::ImageRgb8(RgbImage::new(28, 28));
        let image = match EncodedImage::encode(&img) {
            Ok(i) => i,
            Err(e) => panic!("{e}"),
        };
        match VisionRequest::for_output::<Empty>("p", image, GenerationOptions::default()) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        }
    }

    #[tokio::test]
    async fn bounds_in_flight_requests() -> Result<()> {
        let backend = Arc::new(CountingBackend::default());
        let client = VisionClient::new(backend.clone(), 3)?;
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..12 {
            let c = client.clone();
            set.spawn(async move { c.infer(request(), CancellationToken::new()).await });
        }
        while let Some(r) = set.join_next().await {
            assert!(matches!(r, Ok(Ok(_))));
        }
        assert_eq!(backend.peak.load(Ordering::SeqCst), 3);
        assert_eq!(backend.done.load(Ordering::SeqCst), 12);
        Ok(())
    }

    #[tokio::test]
    async fn pause_blocks_new_requests_until_resume() -> Result<()> {
        let backend = Arc::new(CountingBackend::default());
        let client = VisionClient::new(backend.clone(), 4)?;
        client.pause();
        assert!(client.is_paused());
        let c = client.clone();
        let task = tokio::spawn(async move { c.infer(request(), CancellationToken::new()).await });
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(backend.done.load(Ordering::SeqCst), 0);
        client.resume();
        assert!(matches!(task.await, Ok(Ok(_))));
        assert_eq!(backend.done.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn cancel_while_paused() -> Result<()> {
        let client = VisionClient::new(Arc::new(CountingBackend::default()), 1)?;
        client.pause();
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(matches!(
            client.infer(request(), cancel).await,
            Err(VisionError::Cancelled)
        ));
        Ok(())
    }
}
