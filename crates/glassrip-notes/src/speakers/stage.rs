//! The `name_speakers` stage.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glassrip_audio::types::TranscriptSegment;
use glassrip_core::envelope::{ErrorCode, ErrorInfo};
use glassrip_core::runner::{
    ArtifactSpec, InputDecl, ItemContext, Stage, StageError, StageInputs, WorkItem,
};
use schemars::JsonSchema;
use semver::Version;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{OnceCell, Semaphore};
use tokio::task::JoinSet;

use super::address::find_addresses;
use super::cues::{observe, TileParams, TileReader};
use super::frames::FrameSource;
use super::vote::{plan_samples, time_key, vote, VoteParams};
use super::{FrameObservation, SpeakersRecord, SpeakersSummary};
use crate::people::AliasTable;
use crate::schemas;

/// Parameters of `name_speakers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NameSpeakersParams {
    /// Participant display names (`--participants`).
    pub participants: Vec<String>,
    /// Source video for on-demand frames (None: no visual cues).
    pub video: Option<PathBuf>,
    /// Decoded frame width, pixels.
    pub frame_width: u32,
    /// Ignore the container rotation tag when decoding.
    pub noautorotate: bool,
    /// Frames decoded and read concurrently.
    pub decode_concurrency: usize,
    /// Tile-name OCR description (engine and provider), for the cache key.
    pub ocr_engine: Option<String>,
    /// OCR tile names must appear in at least this many keyframes to add a participant.
    pub ocr_min_keyframes: usize,
    /// Tile matching and ring detection.
    pub tiles: TileParams,
    /// Vote weights.
    pub vote: VoteParams,
}

impl Default for NameSpeakersParams {
    fn default() -> Self {
        Self {
            participants: Vec::new(),
            video: None,
            frame_width: 1920,
            noautorotate: true,
            decode_concurrency: 8,
            ocr_engine: None,
            ocr_min_keyframes: 2,
            tiles: TileParams::default(),
            vote: VoteParams::default(),
        }
    }
}

/// Inputs gathered by `plan`, shared by every item.
#[derive(Debug)]
pub struct PlanData {
    segments: Vec<TranscriptSegment>,
    table: Arc<AliasTable>,
    keyframes: usize,
}

struct Analysis {
    records: BTreeMap<String, SpeakersRecord>,
}

/// `name_speakers`: `glassrip.transcript` (+ keyframes, OCR) to `glassrip.speakers`.
///
/// With visual cues the stage holds the OCR sessions (and their GPU memory)
/// until it is dropped; drop it before GPU phase C so the text model fits.
pub struct NameSpeakersStage {
    params: NameSpeakersParams,
    frames: Option<Arc<dyn FrameSource>>,
    reader: Option<Arc<dyn TileReader>>,
    analysis: OnceCell<Result<Arc<Analysis>, ErrorInfo>>,
}

impl NameSpeakersStage {
    /// A stage without visual cues (address and role cues only).
    pub fn new(params: NameSpeakersParams) -> Self {
        Self {
            params,
            frames: None,
            reader: None,
            analysis: OnceCell::new(),
        }
    }

    /// Enables visual cues with a frame source and a tile reader.
    pub fn with_visual(
        mut self,
        frames: Arc<dyn FrameSource>,
        reader: Arc<dyn TileReader>,
    ) -> Self {
        self.params.ocr_engine = Some(reader.describe());
        self.frames = Some(frames);
        self.reader = Some(reader);
        self
    }

    async fn analyze(&self, data: &PlanData, ctx: &ItemContext) -> Result<Analysis, ErrorInfo> {
        let p = &self.params;
        let no_speech = data.segments.is_empty();
        let no_people = data.table.people().is_empty();
        let addresses = find_addresses(
            &data.segments,
            &data.table,
            p.vote.response_window_s,
            p.vote.min_name_score,
        );
        let plan = plan_samples(&data.segments, &p.vote);
        let mut frames: BTreeMap<i64, FrameObservation> = BTreeMap::new();
        let started = Instant::now();
        // with no participant names no tile can match, so frames are not decoded
        let visual = if no_people {
            None
        } else {
            self.frames.as_ref().zip(self.reader.as_ref())
        };
        if let Some((src, reader)) = visual {
            let sem = Arc::new(Semaphore::new(p.decode_concurrency.max(1)));
            let mut set = JoinSet::new();
            for t in plan.all_times() {
                if ctx.cancel_token().is_cancelled() {
                    return Err(ErrorInfo::new(
                        ErrorCode::Cancelled,
                        "cancelled while decoding frames",
                    ));
                }
                let (src, reader, table, sem) =
                    (src.clone(), reader.clone(), data.table.clone(), sem.clone());
                let tiles = p.tiles.clone();
                let argv = src.argv(t);
                set.spawn(async move {
                    let _permit = sem.acquire_owned().await;
                    let t0 = Instant::now();
                    let frame = src.frame_at(t).await;
                    let wall = t0.elapsed().as_secs_f64();
                    let ok = frame.is_ok();
                    let obs = match frame {
                        Err(e) => FrameObservation {
                            t_s: t,
                            tiles: vec![],
                            presenter: None,
                            error: Some(e),
                        },
                        Ok(img) => tokio::task::spawn_blocking(move || match reader.read(&img) {
                            Ok(texts) => observe(t, &img, &texts, &table, &tiles),
                            Err(e) => FrameObservation {
                                t_s: t,
                                tiles: vec![],
                                presenter: None,
                                error: Some(format!("ocr: {e}")),
                            },
                        })
                        .await
                        .unwrap_or_else(|e| FrameObservation {
                            t_s: t,
                            tiles: vec![],
                            presenter: None,
                            error: Some(format!("ocr task: {e}")),
                        }),
                    };
                    (argv, ok, wall, obs)
                });
            }
            while let Some(res) = set.join_next().await {
                let (argv, ok, wall, obs) = res
                    .map_err(|e| ErrorInfo::new(ErrorCode::Internal, format!("frame task: {e}")))?;
                ctx.record_command(argv, Some(if ok { 0 } else { 1 }), Some(wall));
                frames.insert(time_key(obs.t_s), obs);
            }
        }
        let decode_s = started.elapsed().as_secs_f64();
        let out = vote(
            &data.segments,
            &data.table,
            &addresses,
            &plan,
            &frames,
            &p.vote,
        );

        let mut records = BTreeMap::new();
        for person in data.table.people() {
            records.insert(
                format!("person:{}", person.person_id),
                SpeakersRecord::Person(person.clone()),
            );
        }
        for l in &out.labels {
            records.insert(
                format!("label:{}", l.label),
                SpeakersRecord::Label(l.clone()),
            );
        }
        let relabeled = out
            .segments
            .iter()
            .filter(|s| {
                matches!(
                    s.source,
                    super::SpeakerSource::VisualRelabel | super::SpeakerSource::EvidenceRelabel
                )
            })
            .count();
        for s in out.segments {
            records.insert(
                format!("segment:{}", s.segment_id),
                SpeakersRecord::Segment(s),
            );
        }
        let method = if no_speech {
            "none: no transcript segments (no speech)"
        } else if no_people {
            "none: no participant names (labels left unnamed)"
        } else if self.frames.is_some() {
            "vote: visual tiles (on-demand frames) + direct address + role"
        } else {
            "vote: direct address + role (no video)"
        };
        let mut notes = Vec::new();
        if no_speech {
            notes.push(
                "no transcript segments (no speech detected), so there are no speakers to name"
                    .to_string(),
            );
        }
        if no_people {
            notes.push(format!(
                "no participant names (no --participants and no on-screen tile name seen in {} or more keyframes); {} diarization labels left unnamed",
                p.ocr_min_keyframes,
                out.labels.len()
            ));
        }
        notes.push(format!(
            "{} direct addresses; {} keyframes available; frame decode and read {:.1} s",
            addresses.len(),
            data.keyframes,
            decode_s
        ));
        records.insert(
            "summary".into(),
            SpeakersRecord::Summary(SpeakersSummary {
                method: method.into(),
                frames_decoded: frames.len(),
                frames_failed: frames.values().filter(|o| o.error.is_some()).count(),
                frames_with_tiles: frames.values().filter(|o| !o.tiles.is_empty()).count(),
                frames_with_highlight: frames
                    .values()
                    .filter(|o| !o.highlighted().is_empty())
                    .count(),
                relabeled_segments: relabeled,
                presenter: out.presenter,
                gap_fill: out.gap_fill,
                notes: Some(notes.join("; ")),
            }),
        );
        Ok(Analysis { records })
    }
}

/// Tile names from a `glassrip.ocr` item, read leniently: spans with
/// `region: tile`, plus a `tile_names` list (strings or objects with `text`).
fn tile_texts(item: &Value) -> Vec<String> {
    let spans = item
        .get("spans")
        .or_else(|| item.get("text_spans"))
        .and_then(Value::as_array);
    let from_spans = spans
        .into_iter()
        .flatten()
        .filter(|s| s.get("region").and_then(Value::as_str) == Some("tile"))
        .filter_map(|s| s.get("text").and_then(Value::as_str));
    let names = item.get("tile_names").and_then(Value::as_array);
    let from_names = names
        .into_iter()
        .flatten()
        .filter_map(|n| n.as_str().or_else(|| n.get("text").and_then(Value::as_str)));
    from_spans
        .chain(from_names)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Builds the alias table from participants and OCR tile names.
///
/// With participants given, they are authoritative and tile names only add
/// spellings of them (tile OCR also picks up bookmarks and app names). Without
/// participants, tile names seen in at least `min_keyframes` keyframes become
/// participants.
pub fn build_table(
    participants: &[String],
    ocr_items: &[Value],
    min_keyframes: usize,
) -> AliasTable {
    let mut table = AliasTable::from_names(participants);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for item in ocr_items {
        let mut seen: Vec<String> = tile_texts(item);
        seen.sort();
        seen.dedup();
        for t in seen {
            *counts.entry(t).or_default() += 1;
        }
    }
    for (text, n) in counts {
        if n < min_keyframes || text.to_lowercase().contains("presenting") {
            continue;
        }
        match table.match_screen_text(&text) {
            Some(m) if m.score >= 0.85 => {
                table.add_alias(m.person, text.trim_end_matches(['.', '\u{2026}']).trim())
            }
            Some(_) => {}
            None if participants.is_empty() => {
                let clean = text.trim_end_matches(['.', '\u{2026}']).trim();
                let words = clean.split_whitespace().count();
                let alpha = clean
                    .chars()
                    .all(|c| c.is_alphabetic() || c.is_whitespace() || c == '-' || c == '\'');
                if (2..=4).contains(&words) && alpha {
                    table.add_person(clean);
                }
            }
            None => {}
        }
    }
    table
}

impl Stage for NameSpeakersStage {
    type Params = NameSpeakersParams;
    type Work = (Arc<PlanData>, String);
    type Output = SpeakersRecord;

    fn name(&self) -> &'static str {
        "name_speakers"
    }
    fn version(&self) -> u32 {
        1
    }
    fn output(&self) -> ArtifactSpec {
        ArtifactSpec {
            schema: schemas::SPEAKERS,
            version: Version::new(1, 1, 0),
        }
    }
    fn inputs(&self) -> Vec<InputDecl> {
        vec![
            InputDecl {
                schema: schemas::TRANSCRIPT,
                major: 1,
            },
            InputDecl {
                schema: schemas::KEYFRAMES,
                major: 1,
            },
            InputDecl {
                schema: schemas::OCR,
                major: 1,
            },
        ]
    }
    fn params(&self) -> &NameSpeakersParams {
        &self.params
    }
    fn external_inputs(&self) -> Vec<PathBuf> {
        self.params.video.iter().cloned().collect()
    }
    fn item_timeout(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs(3600))
    }

    fn plan(&self, inputs: &StageInputs) -> Result<Vec<WorkItem<Self::Work>>, StageError> {
        let mut segments: Vec<TranscriptSegment> = inputs
            .read_ok::<TranscriptSegment>(schemas::TRANSCRIPT)?
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        segments.sort_by(|a, b| {
            a.start_s
                .total_cmp(&b.start_s)
                .then(a.segment_id.cmp(&b.segment_id))
        });
        let ocr: Vec<Value> = inputs
            .read_ok::<Value>(schemas::OCR)?
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        let keyframes = inputs.read_ok::<Value>(schemas::KEYFRAMES)?.len();
        let table = build_table(
            &self.params.participants,
            &ocr,
            self.params.ocr_min_keyframes,
        );
        // No participant names is not an error: with no speech there is nothing
        // to name, and with speech every label is left unnamed (see `analyze`).
        if table.people().is_empty() {
            tracing::warn!(
                segments = segments.len(),
                "no participant names (no --participants, no repeated OCR tile names); speakers stay unnamed"
            );
        }
        let mut ids: Vec<String> = table
            .people()
            .iter()
            .map(|p| format!("person:{}", p.person_id))
            .collect();
        let mut labels: Vec<&str> = segments.iter().map(|s| s.speaker_label.as_str()).collect();
        labels.sort_unstable();
        labels.dedup();
        ids.extend(labels.iter().map(|l| format!("label:{l}")));
        ids.extend(segments.iter().map(|s| format!("segment:{}", s.segment_id)));
        ids.push("summary".into());
        let data = Arc::new(PlanData {
            segments,
            table: Arc::new(table),
            keyframes,
        });
        Ok(ids
            .into_iter()
            .map(|id| WorkItem {
                id: id.clone(),
                work: (data.clone(), id),
            })
            .collect())
    }

    async fn process(
        &self,
        ctx: &ItemContext,
        work: Self::Work,
    ) -> Result<SpeakersRecord, ErrorInfo> {
        let (data, id) = work;
        let analysis = self
            .analysis
            .get_or_init(|| async { self.analyze(&data, ctx).await.map(Arc::new) })
            .await
            .clone()?;
        analysis
            .records
            .get(&id)
            .cloned()
            .ok_or_else(|| ErrorInfo::new(ErrorCode::Internal, format!("no record for item {id}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ocr_tile_names_extend_the_table() {
        let item = |names: &[&str]| serde_json::json!({"keyframe_id": "k", "spans": names.iter().map(|n| serde_json::json!({"text": n, "region": "tile"})).collect::<Vec<_>>()});
        let ocr = vec![
            item(&["Avery Quinn", "Tomas Brennan", "Rohan Dasgu..."]),
            item(&[
                "Tomas Brennan",
                "Rohan Dasgu...",
                "Avery Quinn (Presenting)",
            ]),
            item(&["Noise Once"]),
        ];
        let t = build_table(&["Avery Quinn".into(), "Rohan Dasgupta".into()], &ocr, 2);
        let ids: Vec<&str> = t.people().iter().map(|p| p.person_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["avery-quinn", "rohan-dasgupta"],
            "participants are authoritative"
        );
        assert!(t.people()[1].aliases.contains(&"Rohan Dasgu".to_string()));
        let t = build_table(&[], &ocr, 2);
        let ids: Vec<&str> = t.people().iter().map(|p| p.person_id.as_str()).collect();
        // "Avery Quinn" is a plain tile name in only one keyframe
        assert_eq!(ids, vec!["rohan-dasgu", "tomas-brennan"]);
        let named = serde_json::json!({"keyframe_id": "k", "tile_names": ["Avery Qui", {"text": "Avery Quinn"}]});
        assert_eq!(tile_texts(&named), vec!["Avery Qui", "Avery Quinn"]);
    }
}
