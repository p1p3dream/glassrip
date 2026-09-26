//! Artifact envelopes, per-item outcomes, and schema checks.
//!
//! Every artifact is an [`Envelope`]: identifying metadata plus `items` (always a
//! list). Consumers call [`Envelope::check`] (or [`EnvelopeHeader::check`]) with a
//! [`SchemaReq`] and fail with a typed [`SchemaError`] when the schema name or major
//! version does not match.

use schemars::JsonSchema;
use semver::Version;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Name of the tool recorded in [`Producer::tool`] by default.
pub const TOOL_NAME: &str = "glassrip";

/// Who produced an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    /// Tool name, for example `glassrip`.
    pub tool: String,
    /// Tool version (semver string).
    pub version: String,
    /// Git commit of the producing build, if known.
    pub git_sha: Option<String>,
}

impl Producer {
    /// A producer record for this crate's tool name with the given version and sha.
    pub fn glassrip(version: impl Into<String>, git_sha: Option<String>) -> Self {
        Self {
            tool: TOOL_NAME.to_string(),
            version: version.into(),
            git_sha,
        }
    }
}

/// One input an artifact was derived from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputRef {
    /// Path of the input, relative to the run directory when inside it.
    pub path: String,
    /// blake3 of the input file (lowercase hex).
    pub blake3: String,
    /// Schema name when the input is an artifact; `None` for external files such as
    /// the source video.
    pub schema: Option<String>,
    /// Schema version when the input is an artifact.
    pub schema_version: Option<Version>,
}

/// Envelope metadata without the items: the JSONL header record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeHeader {
    /// Artifact schema name, for example `glassrip.frames`.
    pub schema: String,
    /// Semver of the schema.
    pub schema_version: Version,
    /// Run identifier.
    pub run_id: String,
    /// Producing tool.
    pub producer: Producer,
    /// Inputs this artifact was derived from.
    pub inputs: Vec<InputRef>,
    /// Stage parameters (a JSON object; structured, never prose).
    pub params: Value,
    /// Content hash of the finished artifact (see [`content_hash()`]); absent while
    /// an artifact is still being written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Set when the artifact was restored from the cache rather than computed in
    /// this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restored_from: Option<RestoredFrom>,
}

/// Provenance of an artifact restored from the cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RestoredFrom {
    /// Run that originally computed the artifact.
    pub run_id: String,
    /// Cache key it was restored under.
    pub cache_key: String,
}

/// A complete artifact: header fields plus items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Envelope<T> {
    /// Artifact schema name.
    pub schema: String,
    /// Semver of the schema.
    pub schema_version: Version,
    /// Run identifier.
    pub run_id: String,
    /// Producing tool.
    pub producer: Producer,
    /// Inputs this artifact was derived from.
    pub inputs: Vec<InputRef>,
    /// Stage parameters (a JSON object).
    pub params: Value,
    /// Content hash of the artifact (see [`content_hash()`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Cache provenance, when restored from the cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restored_from: Option<RestoredFrom>,
    /// Items (always a list).
    pub items: Vec<T>,
}

/// What a consumer requires of an artifact: its schema name and major version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaReq {
    /// Required schema name.
    pub schema: String,
    /// Required major version.
    pub major: u64,
}

impl SchemaReq {
    /// Builds a requirement.
    pub fn new(schema: impl Into<String>, major: u64) -> Self {
        Self {
            schema: schema.into(),
            major,
        }
    }
}

/// A schema mismatch or malformed envelope.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SchemaError {
    /// The artifact carries a different schema name.
    #[error("expected schema `{expected}`, found `{found}`")]
    WrongSchema {
        /// Required name.
        expected: String,
        /// Name found in the artifact.
        found: String,
    },
    /// The artifact's major version is not the one this consumer understands.
    #[error("schema `{schema}`: expected major version {expected_major}, found {found}")]
    UnsupportedMajor {
        /// Schema name.
        schema: String,
        /// Required major version.
        expected_major: u64,
        /// Version found in the artifact.
        found: Version,
    },
    /// `params` is not a JSON object.
    #[error("schema `{schema}`: params must be a JSON object, found {kind}")]
    ParamsNotObject {
        /// Schema name.
        schema: String,
        /// JSON kind that was found.
        kind: &'static str,
    },
    /// The envelope's identifying fields could not be read.
    #[error("malformed envelope: {0}")]
    Malformed(String),
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn check_parts(
    schema: &str,
    version: &Version,
    params: &Value,
    req: &SchemaReq,
) -> Result<(), SchemaError> {
    if schema != req.schema {
        return Err(SchemaError::WrongSchema {
            expected: req.schema.clone(),
            found: schema.to_string(),
        });
    }
    if version.major != req.major {
        return Err(SchemaError::UnsupportedMajor {
            schema: schema.to_string(),
            expected_major: req.major,
            found: version.clone(),
        });
    }
    if !params.is_object() {
        return Err(SchemaError::ParamsNotObject {
            schema: schema.to_string(),
            kind: json_kind(params),
        });
    }
    Ok(())
}

impl EnvelopeHeader {
    /// Checks schema name, major version, and that `params` is an object.
    pub fn check(&self, req: &SchemaReq) -> Result<(), SchemaError> {
        check_parts(&self.schema, &self.schema_version, &self.params, req)
    }

    /// Attaches items to form a full envelope.
    pub fn with_items<T>(self, items: Vec<T>) -> Envelope<T> {
        Envelope {
            schema: self.schema,
            schema_version: self.schema_version,
            run_id: self.run_id,
            producer: self.producer,
            inputs: self.inputs,
            params: self.params,
            content_hash: self.content_hash,
            restored_from: self.restored_from,
            items,
        }
    }
}

impl<T> Envelope<T> {
    /// Checks schema name, major version, and that `params` is an object.
    pub fn check(&self, req: &SchemaReq) -> Result<(), SchemaError> {
        check_parts(&self.schema, &self.schema_version, &self.params, req)
    }

    /// Splits into header and items.
    pub fn into_parts(self) -> (EnvelopeHeader, Vec<T>) {
        (
            EnvelopeHeader {
                schema: self.schema,
                schema_version: self.schema_version,
                run_id: self.run_id,
                producer: self.producer,
                inputs: self.inputs,
                params: self.params,
                content_hash: self.content_hash,
                restored_from: self.restored_from,
            },
            self.items,
        )
    }
}

/// Error reading an envelope from JSON.
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeReadError {
    /// Schema name or version did not match.
    #[error(transparent)]
    Schema(#[from] SchemaError),
    /// The JSON did not match the current type (unknown fields, wrong types).
    #[error("envelope does not match the expected type: {0}")]
    Parse(#[from] serde_json::Error),
}

#[derive(Deserialize)]
struct EnvelopeProbe {
    schema: String,
    schema_version: Version,
    #[serde(default)]
    params: Value,
}

impl<T: DeserializeOwned> Envelope<T> {
    /// Parses an envelope, checking schema name and major version **before** strict
    /// parsing so that a version mismatch reports as [`SchemaError`], not as a
    /// field-level parse error.
    pub fn from_json_checked(bytes: &[u8], req: &SchemaReq) -> Result<Self, EnvelopeReadError> {
        let probe: EnvelopeProbe =
            serde_json::from_slice(bytes).map_err(|e| SchemaError::Malformed(e.to_string()))?;
        check_parts(&probe.schema, &probe.schema_version, &probe.params, req)?;
        Ok(serde_json::from_slice(bytes)?)
    }
}

impl EnvelopeHeader {
    /// Parses a header from a JSON value, checking schema and major version first.
    pub fn from_value_checked(value: Value, req: &SchemaReq) -> Result<Self, EnvelopeReadError> {
        let probe: EnvelopeProbe = serde_json::from_value(value.clone())
            .map_err(|e| SchemaError::Malformed(e.to_string()))?;
        check_parts(&probe.schema, &probe.schema_version, &probe.params, req)?;
        Ok(serde_json::from_value(value)?)
    }
}

#[derive(Serialize)]
struct ContentView<'a, T> {
    schema: &'a str,
    schema_version: String,
    params: &'a Value,
    items: Vec<&'a T>,
}

/// Content hash of an artifact: blake3 over the canonical JSON of its schema name,
/// schema version, params, and items sorted by id.
///
/// It deliberately excludes `run_id`, `producer`, and `inputs`, so rerunning a
/// stage that produces identical items yields the same hash and downstream cache
/// keys do not change. `items` must already be deduplicated by id.
pub fn content_hash<T: Serialize + Keyed>(
    schema: &str,
    schema_version: &Version,
    params: &Value,
    items: &[T],
) -> Result<String, crate::canonical::CanonicalJsonError> {
    let mut sorted: Vec<&T> = items.iter().collect();
    sorted.sort_by(|a, b| a.key().cmp(b.key()));
    crate::canonical::canonical_hash(&ContentView {
        schema,
        schema_version: schema_version.to_string(),
        params,
        items: sorted,
    })
}

/// Status of one unit of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Produced a result.
    Ok,
    /// Failed; see the error.
    Error,
    /// Intentionally not processed.
    Skipped,
}

/// Closed set of error codes recorded in [`ErrorInfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The item exceeded its time budget.
    Timeout,
    /// The run was cancelled while the item was in flight.
    Cancelled,
    /// A filesystem operation failed.
    Io,
    /// The item's input was invalid or missing.
    InvalidInput,
    /// An external command exited non-zero or produced unusable output.
    ExternalCommand,
    /// A model request failed (connection, HTTP status, preflight).
    ModelRequest,
    /// Model output could not be parsed against its schema.
    SchemaParse,
    /// Output parsed but failed semantic validation.
    Validation,
    /// Frame alignment failed.
    AlignFailed,
    /// A bug or unexpected condition inside glassrip.
    Internal,
}

/// Error details for a failed item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ErrorInfo {
    /// Machine-readable code.
    pub code: ErrorCode,
    /// Human-readable message.
    pub message: String,
    /// Raw text involved in the failure (model reply, stderr tail), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    /// The failure is a settled answer to the item's inputs: running the item
    /// again with the same cache key would fail the same way. The runner caches
    /// terminal failures like results and does not retry them on resume;
    /// `--force-stage` (or forcing the item) retries them. Set directly only for
    /// provably invalid inputs (an image or request that cannot be sent); a model
    /// reply is never known to recur after one sighting, so a
    /// [`recurrent`](Self::recurrent) failure becomes terminal only when the runner
    /// sees it again, identically, on consecutive runs. Timeouts, cancellations,
    /// and connection errors are never terminal.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub terminal: bool,
    /// The failure is a model's answer to the item's content (a reply that failed
    /// validation, was cut off, or looped): it may or may not recur, so it is
    /// retried on the next run and becomes [`terminal`](Self::terminal) once the
    /// same failure (code, message, and raw text) comes back on enough
    /// consecutive runs (`RunnerOptions::terminal_after_repeats`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub recurrent: bool,
    /// Consecutive runs that ended the item with this same failure, counted by
    /// the runner for [`recurrent`](Self::recurrent) failures.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub occurrences: u32,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl ErrorInfo {
    /// Builds an error without raw text.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            raw_text: None,
            terminal: false,
            recurrent: false,
            occurrences: 0,
        }
    }

    /// Adds raw text.
    pub fn with_raw_text(mut self, raw: impl Into<String>) -> Self {
        self.raw_text = Some(raw.into());
        self
    }

    /// Marks the failure terminal (see [`ErrorInfo::terminal`]). Cancellation and
    /// timeouts stay retryable whatever the caller says.
    pub fn terminal(mut self) -> Self {
        self.terminal = !matches!(self.code, ErrorCode::Cancelled | ErrorCode::Timeout);
        self
    }

    /// Marks the failure [`recurrent`](Self::recurrent): terminal only once it
    /// recurs identically across runs. Cancellation and timeouts stay plainly
    /// retryable.
    pub fn terminal_if_repeated(mut self) -> Self {
        self.recurrent = !matches!(self.code, ErrorCode::Cancelled | ErrorCode::Timeout);
        self
    }

    /// Same failure as `other` for counting recurrences: code, message, and raw
    /// text all equal.
    pub fn same_failure(&self, other: &ErrorInfo) -> bool {
        self.code == other.code && self.message == other.message && self.raw_text == other.raw_text
    }
}

impl std::fmt::Display for ErrorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

/// Outcome of one unit of work: `status`, `result`, `error`.
///
/// Invariants (enforced on deserialization): `ok` has a result and no error; `error`
/// has an error and no result; `skipped` has neither result nor error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, try_from = "OutcomeRepr<T>")]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Outcome<T> {
    /// Status.
    pub status: Status,
    /// Result when `status == ok`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,
    /// Error when `status == error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}

/// Wire form of [`Outcome`] before invariant checks.
#[doc(hidden)]
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, bound(deserialize = "T: DeserializeOwned"))]
pub struct OutcomeRepr<T> {
    status: Status,
    #[serde(default)]
    result: Option<T>,
    #[serde(default)]
    error: Option<ErrorInfo>,
}

impl<T> TryFrom<OutcomeRepr<T>> for Outcome<T> {
    type Error = String;

    fn try_from(r: OutcomeRepr<T>) -> Result<Self, Self::Error> {
        match (r.status, r.result.is_some(), r.error.is_some()) {
            (Status::Ok, true, false)
            | (Status::Error, false, true)
            | (Status::Skipped, false, false) => Ok(Self {
                status: r.status,
                result: r.result,
                error: r.error,
            }),
            (status, has_result, has_error) => Err(format!(
                "inconsistent outcome: status {status:?} with result present = {has_result}, error present = {has_error}"
            )),
        }
    }
}

impl<T> Outcome<T> {
    /// A successful outcome.
    pub fn ok(result: T) -> Self {
        Self {
            status: Status::Ok,
            result: Some(result),
            error: None,
        }
    }

    /// A failed outcome.
    pub fn error(error: ErrorInfo) -> Self {
        Self {
            status: Status::Error,
            result: None,
            error: Some(error),
        }
    }

    /// A skipped outcome.
    pub fn skipped() -> Self {
        Self {
            status: Status::Skipped,
            result: None,
            error: None,
        }
    }

    /// True for `status == ok`.
    pub fn is_ok(&self) -> bool {
        self.status == Status::Ok
    }

    /// True for `status == error`.
    pub fn is_error(&self) -> bool {
        self.status == Status::Error
    }
}

/// Something with a stable id used for joins and last-write-wins dedup.
pub trait Keyed {
    /// Stable id.
    fn key(&self) -> &str;
}

/// A stage output item: a stable id plus its outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Record<T> {
    /// Stable item id (for example a frame or keyframe id).
    pub id: String,
    /// Outcome of processing this item.
    pub outcome: Outcome<T>,
}

impl<T> Keyed for Record<T> {
    fn key(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct Frame {
        frame_id: String,
        pts_s: f64,
    }

    fn header(version: &str) -> EnvelopeHeader {
        EnvelopeHeader {
            schema: "glassrip.frames".into(),
            schema_version: Version::parse(version).unwrap(),
            run_id: "run-0001".into(),
            producer: Producer::glassrip("0.1.0", Some("abc123".into())),
            inputs: vec![InputRef {
                path: "input/synthetic.mp4".into(),
                blake3: crate::blake3_hex(b"synthetic"),
                schema: None,
                schema_version: None,
            }],
            params: json!({"interval_s": 2.0}),
            content_hash: None,
            restored_from: None,
        }
    }

    fn envelope(version: &str) -> Envelope<Record<Frame>> {
        header(version).with_items(vec![
            Record {
                id: "f0".into(),
                outcome: Outcome::ok(Frame {
                    frame_id: "f0".into(),
                    pts_s: 0.0,
                }),
            },
            Record {
                id: "f1".into(),
                outcome: Outcome::error(ErrorInfo::new(ErrorCode::Timeout, "decode took too long")),
            },
        ])
    }

    #[test]
    fn round_trip() {
        let env = envelope("1.2.0");
        let bytes = serde_json::to_vec(&env).unwrap();
        let back = Envelope::<Record<Frame>>::from_json_checked(
            &bytes,
            &SchemaReq::new("glassrip.frames", 1),
        )
        .unwrap();
        assert_eq!(back, env);
    }

    #[test]
    fn major_mismatch_is_typed_error() {
        let bytes = serde_json::to_vec(&envelope("2.0.0")).unwrap();
        let err = Envelope::<Record<Frame>>::from_json_checked(
            &bytes,
            &SchemaReq::new("glassrip.frames", 1),
        )
        .unwrap_err();
        match err {
            EnvelopeReadError::Schema(SchemaError::UnsupportedMajor {
                expected_major,
                found,
                ..
            }) => {
                assert_eq!(expected_major, 1);
                assert_eq!(found, Version::new(2, 0, 0));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn minor_bump_is_accepted() {
        let env = envelope("1.9.3");
        assert!(env.check(&SchemaReq::new("glassrip.frames", 1)).is_ok());
    }

    #[test]
    fn wrong_schema_name_is_typed_error() {
        let env = envelope("1.0.0");
        assert!(matches!(
            env.check(&SchemaReq::new("glassrip.keyframes", 1)),
            Err(SchemaError::WrongSchema { .. })
        ));
    }

    #[test]
    fn params_must_be_object() {
        let mut env = envelope("1.0.0");
        env.params = json!("sample every two seconds");
        assert!(matches!(
            env.check(&SchemaReq::new("glassrip.frames", 1)),
            Err(SchemaError::ParamsNotObject { kind: "string", .. })
        ));
    }

    #[test]
    fn unknown_fields_rejected() {
        let mut v = serde_json::to_value(envelope("1.0.0")).unwrap();
        v["surprise"] = json!(1);
        let bytes = serde_json::to_vec(&v).unwrap();
        let err = Envelope::<Record<Frame>>::from_json_checked(
            &bytes,
            &SchemaReq::new("glassrip.frames", 1),
        )
        .unwrap_err();
        assert!(matches!(err, EnvelopeReadError::Parse(_)));

        let mut v = serde_json::to_value(envelope("1.0.0")).unwrap();
        v["items"][0]["outcome"]["result"]["extra"] = json!(true);
        let bytes = serde_json::to_vec(&v).unwrap();
        assert!(
            Envelope::<Record<Frame>>::from_json_checked(
                &bytes,
                &SchemaReq::new("glassrip.frames", 1)
            )
            .is_err()
        );
    }

    #[test]
    fn outcome_invariants_enforced() {
        let bad = [
            json!({"status": "ok"}),
            json!({"status": "error", "result": 1}),
            json!({"status": "ok", "result": 1, "error": {"code": "io", "message": "x"}}),
            json!({"status": "skipped", "result": 1}),
            json!({"status": "maybe"}),
            json!({"status": "error", "error": {"code": "not_a_code", "message": "x"}}),
        ];
        for b in bad {
            assert!(
                serde_json::from_value::<Outcome<u32>>(b.clone()).is_err(),
                "{b}"
            );
        }
        let good: Outcome<u32> =
            serde_json::from_value(json!({"status": "ok", "result": 3})).unwrap();
        assert_eq!(good, Outcome::ok(3));
        let skipped: Outcome<u32> = serde_json::from_value(json!({"status": "skipped"})).unwrap();
        assert_eq!(skipped, Outcome::skipped());
        assert_eq!(
            serde_json::to_value(Outcome::<u32>::skipped()).unwrap(),
            json!({"status": "skipped"})
        );
    }

    #[test]
    fn content_hash_ignores_run_metadata_and_order() {
        let env = envelope("1.0.0");
        let h = |e: &Envelope<Record<Frame>>| {
            content_hash(&e.schema, &e.schema_version, &e.params, &e.items).unwrap()
        };
        let base = h(&env);

        let mut other = env.clone();
        other.run_id = "run-9999".into();
        other.producer = Producer::glassrip("9.9.9", Some("fff".into()));
        other.inputs.clear();
        other.items.reverse();
        assert_eq!(h(&other), base);

        let mut changed = env.clone();
        changed.items[0].outcome = Outcome::ok(Frame {
            frame_id: "f0".into(),
            pts_s: 0.5,
        });
        assert_ne!(h(&changed), base);
        let mut params = env.clone();
        params.params = json!({"interval_s": 1.0});
        assert_ne!(h(&params), base);
        let mut version = env;
        version.schema_version = Version::new(1, 1, 0);
        assert_ne!(h(&version), base);
    }

    #[test]
    fn json_schemas_generate() {
        let s = schemars::schema_for!(Envelope<Record<Frame>>);
        let text = serde_json::to_string(&s).unwrap();
        assert!(text.contains("schema_version"));
        assert!(text.contains("timeout"));
        let _ = schemars::schema_for!(EnvelopeHeader);
        let _ = schemars::schema_for!(Outcome<Frame>);
    }
}
