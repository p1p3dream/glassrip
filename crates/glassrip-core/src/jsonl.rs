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
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
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
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> JsonlError + '_ {
    move |source| JsonlError::Io {
        path: path.to_path_buf(),
        source,
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
    let mut lines = complete
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .enumerate();
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

/// Append-only writer for a JSONL artifact.
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
        if let Some(parent) = path.parent() {
            fs_err::create_dir_all(parent).map_err(io_err(path))?;
        }
        let mut file = fs_err::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .read(true)
            .open(path)
            .map_err(io_err(path))?;
        file.write_all(&encode_line::<T>(&LineOut::Header(header))?)
            .map_err(io_err(path))?;
        file.sync_all().map_err(io_err(path))?;
        if let Some(parent) = path.parent() {
            atomic::sync_dir(parent).map_err(io_err(path))?;
        }
        Ok(Self::from_file(path, file))
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
        if !path.exists() {
            return Ok((
                Self::create(path, header)?,
                Resumed {
                    items: Vec::new(),
                    truncated_bytes: 0,
                    fresh: true,
                },
            ));
        }
        let mut file = fs_err::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(io_err(path))?;
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
        file.seek(SeekFrom::End(0)).map_err(io_err(path))?;
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
