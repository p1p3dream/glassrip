//! Shared HTTP retry policy: exponential backoff with jitter, and
//! `Retry-After` support for 429 and 503 responses.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::StatusCode;

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Total attempts, including the first request.
    pub max_attempts: u32,
    /// Backoff before the first retry (before jitter).
    pub base_delay: Duration,
    /// Upper bound on a computed backoff delay.
    pub max_delay: Duration,
    /// Upper bound on a server-provided `Retry-After` delay.
    pub max_retry_after: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            max_retry_after: Duration::from_secs(120),
        }
    }
}

impl RetryPolicy {
    /// Backoff before retry number `retry` (0 for the first retry), using
    /// "equal jitter": half of the exponential delay is fixed and the other
    /// half is random, so concurrent workers do not retry in lockstep.
    pub fn backoff(&self, retry: u32) -> Duration {
        let factor = 2u32.saturating_pow(retry);
        let exp = self.base_delay.saturating_mul(factor).min(self.max_delay);
        let half = exp / 2;
        half + jitter_up_to(exp - half)
    }

    /// Delay before retry number `retry`, preferring the server's
    /// `Retry-After` when one was given.
    pub fn delay_for(&self, retry: u32, retry_after: Option<Duration>) -> Duration {
        match retry_after {
            Some(d) => d.min(self.max_retry_after),
            None => self.backoff(retry),
        }
    }
}

/// Statuses whose `Retry-After` header is honored (RFC 9110 allows it on
/// 429 and 503).
pub fn honors_retry_after(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status == StatusCode::SERVICE_UNAVAILABLE
}

/// 429 and 5xx are worth retrying; other statuses are not.
pub fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

/// `Retry-After` in delta-seconds form (integer or decimal). The HTTP-date
/// form is not supported and falls back to computed backoff.
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    let secs: f64 = value.parse().ok()?;
    if secs.is_finite() && secs >= 0.0 {
        // try_from avoids the panic from_secs_f64 raises when the value
        // overflows Duration; such headers fall back to computed backoff.
        Duration::try_from_secs_f64(secs).ok()
    } else {
        None
    }
}

fn jitter_up_to(max: Duration) -> Duration {
    let max_nanos = u64::try_from(max.as_nanos()).unwrap_or(u64::MAX);
    if max_nanos == 0 {
        return Duration::ZERO;
    }
    // Each RandomState gets fresh keys, which is enough entropy for jitter
    // without pulling in a random number crate.
    let r = RandomState::new().build_hasher().finish();
    Duration::from_nanos(r % (max_nanos.saturating_add(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    fn policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(1000),
            max_retry_after: Duration::from_secs(10),
        }
    }

    #[test]
    fn backoff_grows_exponentially_within_jitter_bounds() {
        let p = policy();
        for retry in 0..4u32 {
            let exp = Duration::from_millis(100 * 2u64.pow(retry));
            for _ in 0..50 {
                let d = p.backoff(retry);
                assert!(
                    d >= exp / 2 && d <= exp,
                    "retry {retry}: {d:?} not in [{:?}, {exp:?}]",
                    exp / 2
                );
            }
        }
    }

    #[test]
    fn backoff_is_capped() {
        let p = policy();
        for _ in 0..50 {
            assert!(p.backoff(20) <= Duration::from_millis(1000));
        }
    }

    #[test]
    fn backoff_has_jitter() {
        let p = policy();
        let samples: std::collections::HashSet<Duration> = (0..20).map(|_| p.backoff(3)).collect();
        assert!(
            samples.len() > 1,
            "backoff produced identical delays: {samples:?}"
        );
    }

    #[test]
    fn retryable_statuses() {
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_status(StatusCode::OK));
    }

    #[test]
    fn retry_after_parsing() {
        let mut h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
        h.insert(RETRY_AFTER, HeaderValue::from_static("7"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(7)));
        h.insert(RETRY_AFTER, HeaderValue::from_static("0.5"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_millis(500)));
        h.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&h), None);
        h.insert(RETRY_AFTER, HeaderValue::from_static("-3"));
        assert_eq!(parse_retry_after(&h), None);
        h.insert(
            RETRY_AFTER,
            HeaderValue::from_static("99999999999999999999"),
        );
        assert_eq!(parse_retry_after(&h), None);
        h.insert(RETRY_AFTER, HeaderValue::from_static("1e300"));
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn retry_after_status_covers_429_and_503() {
        assert!(honors_retry_after(StatusCode::TOO_MANY_REQUESTS));
        assert!(honors_retry_after(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!honors_retry_after(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(!honors_retry_after(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn delay_prefers_retry_after_and_caps_it() {
        let p = policy();
        assert_eq!(
            p.delay_for(0, Some(Duration::from_secs(3))),
            Duration::from_secs(3)
        );
        assert_eq!(
            p.delay_for(0, Some(Duration::from_secs(999))),
            Duration::from_secs(10)
        );
        assert!(p.delay_for(0, None) <= Duration::from_millis(100));
    }
}
