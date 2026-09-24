//! Crash-safe JSONL artifacts.
//!
//! Layout: line 1 is `{"record":"header", ...envelope fields except items}`; every
//! following line is `{"record":"item", ...item fields}`. Items are keyed by id
//! ([`Keyed`]) and the **last** line for an id wins on read, so appending a newer
//! record for an id replaces the older one without rewriting the file.
//!
//! Writers append whole lines with a single `write_all` and call `sync_data` at
//! checkpoints. On open for resume, a trailing partial line (a crash mid-write) is
//! truncated away before anything else is appended.

use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::atomic::{self, AtomicWriteError};
use crate::envelope::{Envelope, EnvelopeHeader, EnvelopeReadError, Keyed, SchemaReq};

/// Default number of appended lines between `sync_data` checkpoints.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 16;

/// Error from the JSONL store.
#[derive(Debug, thiserror::Error)]
pub enum JsonlError {
    /// Filesystem failure.
    #[error("jsonl I/O on {path}: {source}")]
    Io {
        /// File involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// Atomic rewrite failed.
    #[error(transparent)]
    Atomic(#[from] AtomicWriteError),
    /// The file has no header line.
    #[error("{0}: missing header record")]
    MissingHeader(PathBuf),
    /// A complete line could not be parsed.
    #[error("{path}:{line}: {message}")]
    BadLine {
        /// File.
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        /// Parse error.
        message: String,
    },
    /// The header failed its schema check.
    #[error("{path}: {source}")]
    Header {
        /// File.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: EnvelopeReadError,
    },
    /// On resume, the existing header differs from the expected one.
    #[error("{0}: existing header does not match this run's header")]
    HeaderMismatch(PathBuf),
    /// An item could not be serialized.
    #[error("cannot serialize item: {0}")]
    Serialize(#[from] serde_json::Error),
    /// The filesystem is full.
    #[error("disk full writing {path} ({}): {source}", atomic::describe_space(*available_bytes))]
    DiskFull {
        /// File involved.
        path: PathBuf,
        /// Bytes available on that filesystem, if measurable.
        available_bytes: Option<u64>,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// The content hash could not be computed.
    #[error(transparent)]
    Canonical(#[from] crate::canonical::CanonicalJsonError),
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> JsonlError + '_ {
    move |source| match atomic::available_space_if_full(path, &source) {
        Some(available_bytes) => JsonlError::DiskFull {
            path: path.to_path_buf(),
            available_bytes,
            source,
        },
        None => JsonlError::Io {
            path: path.to_path_buf(),
            source,
        },
    }
}

#[derive(Serialize)]
#[serde(tag = "record", rename_all = "lowercase")]
enum LineOut<'a, T> {
    Header(&'a EnvelopeHeader),
    Item(&'a T),
}

#[derive(Deserialize)]
#[serde(tag = "record", rename_all = "lowercase")]
enum LineIn<T> {
    Item(T),
}

fn encode_line<T: Serialize>(line: &LineOut<'_, T>) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec(line)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Returns the length of the longest prefix of `bytes` ending in `\n`.
fn complete_prefix_len(bytes: &[u8]) -> usize {
    bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1)
}

/// Items read from a JSONL artifact, deduplicated by id (last write wins), in order of
/// first appearance.
fn dedupe<T: Keyed>(items: Vec<T>) -> Vec<T> {
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut out: Vec<T> = Vec::with_capacity(items.len());
    for item in items {
        match index.get(item.key()) {
            Some(&i) => out[i] = item,
            None => {
                index.insert(item.key().to_string(), out.len());
                out.push(item);
            }
        }
    }
    out
}

fn parse_lines<T: DeserializeOwned + Keyed>(
    path: &Path,
    complete: &[u8],
    req: Option<&SchemaReq>,
) -> Result<(EnvelopeHeader, Vec<T>), JsonlError> {
    // `complete` ends in a newline (or is empty), so the final split element is the
    // empty remainder after it; every other element is a physical line.
    let mut physical: Vec<&[u8]> = complete.split(|b| *b == b'\n').collect();
    physical.pop();
    let mut lines = physical.into_iter().enumerate();
    let (_, first) = lines
        .next()
        .ok_or_else(|| JsonlError::MissingHeader(path.to_path_buf()))?;
    let mut header_value: Value =
        serde_json::from_slice(first).map_err(|e| JsonlError::BadLine {
            path: path.to_path_buf(),
            line: 1,
            message: e.to_string(),
        })?;
    match header_value
        .as_object_mut()
        .and_then(|m| m.remove("record"))
    {
        Some(Value::String(s)) if s == "header" => {}
        _ => return Err(JsonlError::MissingHeader(path.to_path_buf())),
    }
    let header = match req {
        Some(req) => EnvelopeHeader::from_value_checked(header_value, req),
        None => serde_json::from_value(header_value).map_err(EnvelopeReadError::from),
    }
    .map_err(|source| JsonlError::Header {
        path: path.to_path_buf(),
        source,
    })?;

    let mut items = Vec::new();
    for (i, line) in lines {
        if line.iter().all(u8::is_ascii_whitespace) {
            return Err(JsonlError::BadLine {
                path: path.to_path_buf(),
                line: i + 1,
                message: "blank line".to_string(),
            });
        }
        let LineIn::Item(item) =
            serde_json::from_slice::<LineIn<T>>(line).map_err(|e| JsonlError::BadLine {
                path: path.to_path_buf(),
                line: i + 1,
                message: e.to_string(),
            })?;
        items.push(item);
    }
    Ok((header, dedupe(items)))
}

/// Encodes a header record line (including the trailing newline).
pub fn header_line(header: &EnvelopeHeader) -> Result<Vec<u8>, serde_json::Error> {
    encode_line::<()>(&LineOut::Header(header))
}

/// Parses the complete lines of an in-memory JSONL artifact (a trailing partial line
/// is ignored), checking the header against `req`.
pub fn parse_bytes<T: DeserializeOwned + Keyed>(
    path: &Path,
    bytes: &[u8],
    req: &SchemaReq,
) -> Result<(EnvelopeHeader, Vec<T>), JsonlError> {
    parse_lines(path, &bytes[..complete_prefix_len(bytes)], Some(req))
}

/// An item read as raw JSON, keyed by its `id` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RawItem(pub Value);

impl Keyed for RawItem {
    fn key(&self) -> &str {
        self.0.get("id").and_then(Value::as_str).unwrap_or("")
    }
}

/// Computes the content hash of a JSONL artifact from its items (see
/// [`crate::envelope::content_hash`]), without trusting the header's value.
pub fn compute_content_hash(path: &Path, req: &SchemaReq) -> Result<String, JsonlError> {
    let env = read::<RawItem>(path, req)?;
    Ok(crate::envelope::content_hash(
        &env.schema,
        &env.schema_version,
        &env.params,
        &env.items,
    )?)
}

/// Reads only the header line of a JSONL artifact and checks it against `req`.
pub fn read_header(path: &Path, req: &SchemaReq) -> Result<EnvelopeHeader, JsonlError> {
    let file = fs_err::File::open(path).map_err(io_err(path))?;
    let mut first = Vec::new();
    io::BufReader::new(file)
        .read_until(b'\n', &mut first)
        .map_err(io_err(path))?;
    let complete = &first[..complete_prefix_len(&first)];
    parse_lines::<NoItem>(path, complete, Some(req)).map(|(h, _)| h)
}

#[derive(Deserialize)]
struct NoItem {}

impl Keyed for NoItem {
    fn key(&self) -> &str {
        ""
    }
}

/// Reads a JSONL artifact, checks its header, and returns it as an [`Envelope`] with
/// deduplicated items. A trailing partial line is ignored (the file is not modified).
pub fn read<T: DeserializeOwned + Keyed>(
    path: &Path,
    req: &SchemaReq,
) -> Result<Envelope<T>, JsonlError> {
    let bytes = fs_err::read(path).map_err(io_err(path))?;
    let complete = &bytes[..complete_prefix_len(&bytes)];
    let (header, items) = parse_lines(path, complete, Some(req))?;
    Ok(header.with_items(items))
}

/// Atomically writes a complete JSONL artifact (header plus one line per item).
pub fn write_atomic<T: Serialize>(
    path: &Path,
    envelope_header: &EnvelopeHeader,
    items: &[T],
) -> Result<(), JsonlError> {
    let mut buf = encode_line::<T>(&LineOut::Header(envelope_header))?;
    for item in items {
        buf.extend(encode_line(&LineOut::Item(item))?);
    }
    atomic::write_atomic(path, &buf)?;
    Ok(())
}

/// How [`JsonlWriter::open_resume`] found the file.
#[derive(Debug)]
pub struct Resumed<T> {
    /// Items already present (deduplicated, last write wins).
    pub items: Vec<T>,
    /// Bytes of a trailing partial line that were truncated away.
    pub truncated_bytes: u64,
    /// True when the file did not exist (or was empty) and a new one was started.
    pub fresh: bool,
}

fn open_append(path: &Path) -> Result<fs_err::File, JsonlError> {
    fs_err::OpenOptions::new()
        .read(true)
        .append(true)
        .open(path)
        .map_err(io_err(path))
}

/// Append-only writer for a JSONL artifact (the file is opened in append mode).
#[derive(Debug)]
pub struct JsonlWriter<T> {
    path: PathBuf,
    file: fs_err::File,
    since_checkpoint: usize,
    checkpoint_every: usize,
    _marker: PhantomData<fn(&T)>,
}

impl<T: Serialize + DeserializeOwned + Keyed> JsonlWriter<T> {
    /// Creates (or truncates) the file and writes the header, synced.
    pub fn create(path: &Path, header: &EnvelopeHeader) -> Result<Self, JsonlError> {
        // The header is written atomically, so the file never exists without it.
        atomic::write_atomic(path, &encode_line::<T>(&LineOut::Header(header))?)?;
        Ok(Self::from_file(path, open_append(path)?))
    }

    fn from_file(path: &Path, file: fs_err::File) -> Self {
        Self {
            path: path.to_path_buf(),
            file,
            since_checkpoint: 0,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
            _marker: PhantomData,
        }
    }

    /// Opens an existing file for resume, or creates it.
    ///
    /// A trailing partial line is truncated. The existing header must equal `header`
    /// except for `run_id` (a resumed run gets a new id); otherwise
    /// [`JsonlError::HeaderMismatch`] is returned and the file is left untouched.
    pub fn open_resume(
        path: &Path,
        header: &EnvelopeHeader,
    ) -> Result<(Self, Resumed<T>), JsonlError> {
        if atomic::metadata_opt(path).map_err(io_err(path))?.is_none() {
            return Ok((
                Self::create(path, header)?,
                Resumed {
                    items: Vec::new(),
                    truncated_bytes: 0,
                    fresh: true,
                },
            ));
        }
        let mut file = open_append(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(io_err(path))?;
        let keep = complete_prefix_len(&bytes);
        if keep == 0 {
            drop(file);
            return Ok((
                Self::create(path, header)?,
                Resumed {
                    items: Vec::new(),
                    truncated_bytes: bytes.len() as u64,
                    fresh: true,
                },
            ));
        }
        let (existing, items) = parse_lines::<T>(path, &bytes[..keep], None)?;
        let mut comparable = existing.clone();
        comparable.run_id.clone_from(&header.run_id);
        if &comparable != header {
            return Err(JsonlError::HeaderMismatch(path.to_path_buf()));
        }
        let truncated = (bytes.len() - keep) as u64;
        if truncated > 0 {
            file.set_len(keep as u64).map_err(io_err(path))?;
            file.sync_all().map_err(io_err(path))?;
        }
        Ok((
            Self::from_file(path, file),
            Resumed {
                items,
                truncated_bytes: truncated,
                fresh: false,
            },
        ))
    }

    /// Sets how many appends happen between automatic `sync_data` calls (minimum 1).
    pub fn with_checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = n.max(1);
        self
    }

    /// Appends one item as a single whole-line write; syncs at checkpoints.
    pub fn append(&mut self, item: &T) -> Result<(), JsonlError> {
        let line = encode_line(&LineOut::Item(item))?;
        self.file.write_all(&line).map_err(io_err(&self.path))?;
        self.since_checkpoint += 1;
        if self.since_checkpoint >= self.checkpoint_every {
            self.checkpoint()?;
        }
        Ok(())
    }

    /// Flushes appended lines to stable storage (`sync_data`).
    pub fn checkpoint(&mut self) -> Result<(), JsonlError> {
        self.file.sync_data().map_err(io_err(&self.path))?;
        self.since_checkpoint = 0;
        Ok(())
    }

    /// Path of the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{ErrorCode, ErrorInfo, Outcome, Producer, Record};
    use semver::Version;
    use serde_json::json;

    type Rec = Record<u32>;

    fn header(run_id: &str) -> EnvelopeHeader {
        EnvelopeHeader {
            schema: "glassrip.test_numbers".into(),
            schema_version: Version::new(1, 0, 0),
            run_id: run_id.into(),
            producer: Producer::glassrip("0.1.0", None),
            inputs: vec![],
            params: json!({"n": 3}),
            content_hash: None,
            restored_from: None,
        }
    }

    fn req() -> SchemaReq {
        SchemaReq::new("glassrip.test_numbers", 1)
    }

    fn rec(id: &str, v: u32) -> Rec {
        Record {
            id: id.into(),
            outcome: Outcome::ok(v),
        }
    }

    #[test]
    fn header_and_items_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        let mut w = JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
        w.append(&rec("a", 1)).unwrap();
        w.checkpoint().unwrap();
        let text = fs_err::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with(r#"{"record":"header","schema":"glassrip.test_numbers""#));
        assert!(lines[1].starts_with(r#"{"record":"item","id":"a""#));
        let h = read_header(&path, &req()).unwrap();
        assert_eq!(h, header("r1"));
    }

    #[test]
    fn last_write_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        let mut w = JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
        w.append(&rec("a", 1)).unwrap();
        w.append(&rec("b", 2)).unwrap();
        w.append(&Record {
            id: "a".into(),
            outcome: Outcome::error(ErrorInfo::new(ErrorCode::Io, "x")),
        })
        .unwrap();
        w.append(&rec("a", 3)).unwrap();
        w.checkpoint().unwrap();
        let env = read::<Rec>(&path, &req()).unwrap();
        assert_eq!(env.items, vec![rec("a", 3), rec("b", 2)]);
    }

    #[test]
    fn partial_line_truncated_on_resume() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        {
            let mut w = JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
            w.append(&rec("a", 1)).unwrap();
            w.append(&rec("b", 2)).unwrap();
            w.checkpoint().unwrap();
        }
        // Simulate a crash mid-write of a third record.
        let mut f = fs_err::OpenOptions::new().append(true).open(&path).unwrap();
        let fragment = br#"{"record":"item","id":"c","outc"#;
        f.write_all(fragment).unwrap();
        drop(f);

        // A plain read ignores the partial line without modifying the file.
        let before = fs_err::metadata(&path).unwrap().len();
        assert_eq!(read::<Rec>(&path, &req()).unwrap().items.len(), 2);
        assert_eq!(fs_err::metadata(&path).unwrap().len(), before);

        let (mut w, resumed) = JsonlWriter::<Rec>::open_resume(&path, &header("r2")).unwrap();
        assert!(!resumed.fresh);
        assert_eq!(resumed.truncated_bytes, fragment.len() as u64);
        assert_eq!(resumed.items, vec![rec("a", 1), rec("b", 2)]);
        w.append(&rec("c", 3)).unwrap();
        w.append(&rec("a", 10)).unwrap();
        w.checkpoint().unwrap();

        let env = read::<Rec>(&path, &req()).unwrap();
        assert_eq!(env.items, vec![rec("a", 10), rec("b", 2), rec("c", 3)]);
        // Every line is complete JSON.
        for line in fs_err::read_to_string(&path).unwrap().lines() {
            serde_json::from_str::<Value>(line).unwrap();
        }
    }

    #[test]
    fn resume_rejects_different_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
        let mut other = header("r1");
        other.params = json!({"n": 4});
        assert!(matches!(
            JsonlWriter::<Rec>::open_resume(&path, &other),
            Err(JsonlError::HeaderMismatch(_))
        ));
    }

    #[test]
    fn resume_on_partial_header_starts_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        fs_err::write(&path, br#"{"record":"head"#).unwrap();
        let (_, resumed) = JsonlWriter::<Rec>::open_resume(&path, &header("r1")).unwrap();
        assert!(resumed.fresh);
        assert_eq!(read_header(&path, &req()).unwrap(), header("r1"));
    }

    #[test]
    fn corrupt_complete_line_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
        let mut f = fs_err::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"{\"record\":\"item\",\"id\":\"a\"}\n")
            .unwrap();
        drop(f);
        assert!(matches!(
            read::<Rec>(&path, &req()),
            Err(JsonlError::BadLine { line: 2, .. })
        ));
    }

    #[test]
    fn line_numbers_are_physical_and_blank_lines_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        let mut w = JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
        w.append(&rec("a", 1)).unwrap();
        w.checkpoint().unwrap();
        let mut f = fs_err::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"\n").unwrap();
        drop(f);
        assert!(matches!(
            read::<Rec>(&path, &req()),
            Err(JsonlError::BadLine { line: 3, ref message, .. }) if message == "blank line"
        ));

        let path2 = dir.path().join("b.jsonl");
        let mut w = JsonlWriter::<Rec>::create(&path2, &header("r1")).unwrap();
        w.append(&rec("a", 1)).unwrap();
        w.append(&rec("b", 2)).unwrap();
        w.checkpoint().unwrap();
        let mut f = fs_err::OpenOptions::new()
            .append(true)
            .open(&path2)
            .unwrap();
        f.write_all(b"{\"record\":\"item\",\"id\":\"c\"}\n")
            .unwrap();
        drop(f);
        assert!(matches!(
            read::<Rec>(&path2, &req()),
            Err(JsonlError::BadLine { line: 4, .. })
        ));
    }

    #[test]
    fn writer_appends_after_external_growth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        let mut w = JsonlWriter::<Rec>::create(&path, &header("r1")).unwrap();
        // Another handle appends; an append-mode writer must not overwrite it.
        let mut other = fs_err::OpenOptions::new().append(true).open(&path).unwrap();
        other
            .write_all(&encode_line(&LineOut::Item(&rec("x", 9))).unwrap())
            .unwrap();
        drop(other);
        w.append(&rec("a", 1)).unwrap();
        w.checkpoint().unwrap();
        assert_eq!(
            read::<Rec>(&path, &req()).unwrap().items,
            vec![rec("x", 9), rec("a", 1)]
        );
    }

    #[test]
    fn content_hash_from_raw_items_matches_typed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        let items = vec![rec("b", 2), rec("a", 1)];
        write_atomic::<Rec>(&path, &header("r1"), &items).unwrap();
        let h = header("r1");
        let typed =
            crate::envelope::content_hash(&h.schema, &h.schema_version, &h.params, &items).unwrap();
        assert_eq!(compute_content_hash(&path, &req()).unwrap(), typed);
    }

    #[test]
    fn disk_full_reports_path_and_space() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        let err = io_err(&path)(io::Error::new(io::ErrorKind::StorageFull, "no space"));
        match &err {
            JsonlError::DiskFull {
                path: p,
                available_bytes,
                ..
            } => {
                assert_eq!(p, &path);
                assert!(available_bytes.is_some());
            }
            other => panic!("expected DiskFull, got {other:?}"),
        }
        assert!(err.to_string().contains("bytes available"));
        assert!(matches!(
            io_err(&path)(io::Error::other("x")),
            JsonlError::Io { .. }
        ));
    }

    #[test]
    fn header_schema_checked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jsonl");
        write_atomic::<Rec>(&path, &header("r1"), &[rec("a", 1)]).unwrap();
        assert!(matches!(
            read::<Rec>(&path, &SchemaReq::new("glassrip.test_numbers", 2)),
            Err(JsonlError::Header { .. })
        ));
        assert_eq!(read::<Rec>(&path, &req()).unwrap().items, vec![rec("a", 1)]);
    }
}
