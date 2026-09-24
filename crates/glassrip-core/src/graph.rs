//! Stage graph validation and stage selection.
//!
//! Stages declare the artifact they produce and the artifacts they consume. A
//! [`StageGraph`] rejects duplicate stages, two producers for one artifact, inputs no
//! stage produces, and cycles; it yields a deterministic topological order (ties
//! broken by declaration order).
//!
//! [`Selection`] implements `--from-stage`, `--until-stage`, and `--force-stage`:
//!
//! - `from S`: run `S` and its descendants; `S` is forced (cache bypassed). Other
//!   stages are skipped and their existing artifacts are reused.
//! - `until S`: run only `S` and its ancestors.
//! - `force S`: bypass the cache for `S`. Downstream stages re-key through their
//!   input hashes and rerun only if `S`'s output bytes changed.
//!
//! When both `from` and `until` are given, the selected set is the intersection.

use std::collections::{BTreeSet, HashMap};

/// A stage's position in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDecl {
    /// Stage name.
    pub name: String,
    /// Artifact schema name the stage produces.
    pub output: String,
    /// Artifact schema names the stage consumes.
    pub inputs: Vec<String>,
}

impl StageDecl {
    /// Builds a declaration.
    pub fn new(name: &str, output: &str, inputs: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            output: output.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Graph validation error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// Two stages share a name.
    #[error("duplicate stage `{0}`")]
    DuplicateStage(String),
    /// Two stages produce the same artifact.
    #[error("artifact `{artifact}` is produced by both `{first}` and `{second}`")]
    DuplicateProducer {
        /// Artifact.
        artifact: String,
        /// First producer.
        first: String,
        /// Second producer.
        second: String,
    },
    /// An input artifact is produced by no stage.
    #[error("stage `{stage}` consumes `{artifact}`, which no stage produces")]
    MissingInput {
        /// Consuming stage.
        stage: String,
        /// Missing artifact.
        artifact: String,
    },
    /// The graph has a cycle.
    #[error("stage graph has a cycle: {}", .0.join(" -> "))]
    Cycle(Vec<String>),
    /// A selection names a stage that is not in the graph.
    #[error("unknown stage `{0}`")]
    UnknownStage(String),
    /// The selection is empty (for example `from` is downstream of `until`).
    #[error("stage selection is empty: `{from}` is not an ancestor of `{until}`")]
    EmptySelection {
        /// From stage.
        from: String,
        /// Until stage.
        until: String,
    },
}

/// A validated DAG of stages.
#[derive(Debug, Clone)]
pub struct StageGraph {
    decls: Vec<StageDecl>,
    by_name: HashMap<String, usize>,
    /// Upstream stage indices per stage (deduplicated, sorted).
    parents: Vec<Vec<usize>>,
    /// Downstream stage indices per stage.
    children: Vec<Vec<usize>>,
    order: Vec<usize>,
}

impl StageGraph {
    /// Validates the declarations and builds the graph.
    pub fn new(decls: Vec<StageDecl>) -> Result<Self, GraphError> {
        let mut by_name = HashMap::new();
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (i, d) in decls.iter().enumerate() {
            if by_name.insert(d.name.clone(), i).is_some() {
                return Err(GraphError::DuplicateStage(d.name.clone()));
            }
            if let Some(&j) = producer.get(d.output.as_str()) {
                return Err(GraphError::DuplicateProducer {
                    artifact: d.output.clone(),
                    first: decls[j].name.clone(),
                    second: d.name.clone(),
                });
            }
            producer.insert(&d.output, i);
        }
        let n = decls.len();
        let mut parents = vec![Vec::new(); n];
        let mut children = vec![Vec::new(); n];
        for (i, d) in decls.iter().enumerate() {
            for input in &d.inputs {
                let &p = producer
                    .get(input.as_str())
                    .ok_or_else(|| GraphError::MissingInput {
                        stage: d.name.clone(),
                        artifact: input.clone(),
                    })?;
                parents[i].push(p);
                children[p].push(i);
            }
        }
        for list in parents.iter_mut().chain(children.iter_mut()) {
            list.sort_unstable();
            list.dedup();
        }

        // Kahn's algorithm, always taking the lowest declaration index available.
        let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
        let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(i) = ready.pop_first() {
            order.push(i);
            for &c in &children[i] {
                indegree[c] -= 1;
                if indegree[c] == 0 {
                    ready.insert(c);
                }
            }
        }
        if order.len() != n {
            let remaining: BTreeSet<usize> = (0..n).filter(|&i| indegree[i] > 0).collect();
            let cycle = find_cycle(&parents, &remaining)
                .into_iter()
                .map(|i| decls[i].name.clone())
                .collect();
            return Err(GraphError::Cycle(cycle));
        }
        Ok(Self {
            decls,
            by_name,
            parents,
            children,
            order,
        })
    }

    /// Stage names in topological order.
    pub fn order(&self) -> Vec<&str> {
        self.order
            .iter()
            .map(|&i| self.decls[i].name.as_str())
            .collect()
    }

    /// Declaration of a stage.
    pub fn decl(&self, name: &str) -> Option<&StageDecl> {
        self.by_name.get(name).map(|&i| &self.decls[i])
    }

    fn index(&self, name: &str) -> Result<usize, GraphError> {
        self.by_name
            .get(name)
            .copied()
            .ok_or_else(|| GraphError::UnknownStage(name.to_string()))
    }

    fn closure(&self, start: usize, edges: &[Vec<usize>]) -> BTreeSet<usize> {
        let mut seen = BTreeSet::from([start]);
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            for &j in &edges[i] {
                if seen.insert(j) {
                    stack.push(j);
                }
            }
        }
        seen
    }

    /// `name` plus every stage downstream of it.
    pub fn descendants(&self, name: &str) -> Result<BTreeSet<String>, GraphError> {
        let i = self.index(name)?;
        Ok(self.names(&self.closure(i, &self.children)))
    }

    /// `name` plus every stage upstream of it.
    pub fn ancestors(&self, name: &str) -> Result<BTreeSet<String>, GraphError> {
        let i = self.index(name)?;
        Ok(self.names(&self.closure(i, &self.parents)))
    }

    fn names(&self, set: &BTreeSet<usize>) -> BTreeSet<String> {
        set.iter().map(|&i| self.decls[i].name.clone()).collect()
    }

    /// Resolves a selection into a plan.
    pub fn plan(&self, sel: &Selection) -> Result<Plan, GraphError> {
        let all: BTreeSet<usize> = (0..self.decls.len()).collect();
        let from_set = match &sel.from {
            Some(s) => self.closure(self.index(s)?, &self.children),
            None => all.clone(),
        };
        let until_set = match &sel.until {
            Some(s) => self.closure(self.index(s)?, &self.parents),
            None => all,
        };
        let selected: BTreeSet<usize> = from_set.intersection(&until_set).copied().collect();
        if selected.is_empty() {
            if let (Some(from), Some(until)) = (&sel.from, &sel.until) {
                return Err(GraphError::EmptySelection {
                    from: from.clone(),
                    until: until.clone(),
                });
            }
        }
        let mut force = BTreeSet::new();
        for s in &sel.force {
            force.insert(self.decls[self.index(s)?].name.clone());
        }
        if let Some(from) = &sel.from {
            force.insert(from.clone());
        }
        let mut decisions = HashMap::new();
        for (i, d) in self.decls.iter().enumerate() {
            let decision = if selected.contains(&i) {
                StageDecision::Run {
                    force: force.contains(&d.name),
                }
            } else if !from_set.contains(&i) {
                StageDecision::Skip(SkipReason::BeforeFrom)
            } else {
                StageDecision::Skip(SkipReason::AfterUntil)
            };
            decisions.insert(d.name.clone(), decision);
        }
        Ok(Plan {
            order: self.order().into_iter().map(str::to_string).collect(),
            decisions,
        })
    }
}

fn find_cycle(parents: &[Vec<usize>], remaining: &BTreeSet<usize>) -> Vec<usize> {
    // Every node left after Kahn still has an unprocessed parent, which is itself
    // left. Walking parents within the remaining set therefore never dead-ends and
    // must revisit a node; the revisited stretch is a cycle (reversed at the end so it
    // reads in data-flow direction).
    let Some(&start) = remaining.first() else {
        return Vec::new();
    };
    let mut path = vec![start];
    let mut pos: HashMap<usize, usize> = HashMap::from([(start, 0)]);
    let mut cur = start;
    loop {
        let Some(&next) = parents[cur].iter().find(|p| remaining.contains(p)) else {
            return path;
        };
        if let Some(&p) = pos.get(&next) {
            let mut cycle = path[p..].to_vec();
            cycle.push(next);
            cycle.reverse();
            return cycle;
        }
        pos.insert(next, path.len());
        path.push(next);
        cur = next;
    }
}

/// `--from-stage`, `--until-stage`, `--force-stage`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Selection {
    /// Start at this stage.
    pub from: Option<String>,
    /// Stop after this stage.
    pub until: Option<String>,
    /// Bypass the cache for these stages.
    pub force: BTreeSet<String>,
}

/// Why a stage is skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Upstream of (or unrelated to) `--from-stage`.
    BeforeFrom,
    /// Not needed for `--until-stage`.
    AfterUntil,
}

/// What to do with a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageDecision {
    /// Run it (from cache unless `force`).
    Run {
        /// Bypass the cache.
        force: bool,
    },
    /// Do not run it.
    Skip(SkipReason),
}

/// A resolved selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Topological order of all stages.
    pub order: Vec<String>,
    decisions: HashMap<String, StageDecision>,
}

impl Plan {
    /// Decision for a stage, or `None` if it is not in the graph.
    pub fn decision(&self, stage: &str) -> Option<StageDecision> {
        self.decisions.get(stage).copied()
    }

    /// Names of stages that will run, in topological order.
    pub fn selected(&self) -> Vec<&str> {
        self.order
            .iter()
            .filter(|s| {
                matches!(
                    self.decisions.get(s.as_str()),
                    Some(StageDecision::Run { .. })
                )
            })
            .map(String::as_str)
            .collect()
    }
}

/// The meeting-mode stage graph: stage names and the artifacts they exchange.
///
/// Artifact names follow the data contracts; stages whose artifact is not named there
/// use `glassrip.<stage>`. `diarize` reads the audio only, so ASR and diarization can
/// run concurrently.
pub fn meeting_mode_stage_decls() -> Vec<StageDecl> {
    let d = StageDecl::new;
    vec![
        d("probe", "glassrip.media_probe", &[]),
        d("orient", "glassrip.orientation", &["glassrip.media_probe"]),
        d(
            "frames",
            "glassrip.frames",
            &["glassrip.media_probe", "glassrip.orientation"],
        ),
        d("screen_quad", "glassrip.screen_quads", &["glassrip.frames"]),
        // Features run on unrectified frames (spec 6.4), so they do not read the quads.
        d("features", "glassrip.features", &["glassrip.frames"]),
        d(
            "keyframes",
            "glassrip.keyframes",
            &["glassrip.features", "glassrip.media_probe"],
        ),
        d(
            "rectify",
            "glassrip.rectified_keyframes",
            &[
                "glassrip.keyframes",
                "glassrip.frames",
                "glassrip.screen_quads",
            ],
        ),
        d(
            "ocr_harvest",
            "glassrip.ocr",
            &["glassrip.rectified_keyframes"],
        ),
        d(
            "ocr_vocabulary",
            "glassrip.asr_vocabulary",
            &["glassrip.ocr"],
        ),
        d(
            "classify",
            "glassrip.screen_class",
            &["glassrip.rectified_keyframes", "glassrip.ocr"],
        ),
        d(
            "canvas_crop",
            "glassrip.canvas_crop",
            &[
                "glassrip.screen_class",
                "glassrip.frames",
                "glassrip.screen_quads",
                "glassrip.rectified_keyframes",
                "glassrip.keyframes",
                "glassrip.ocr",
            ],
        ),
        d(
            "board_read",
            "glassrip.board_reading",
            &["glassrip.canvas_crop", "glassrip.ocr"],
        ),
        d(
            "board_validate",
            "glassrip.board_validate",
            &["glassrip.board_reading", "glassrip.ocr"],
        ),
        d(
            "edge_direction",
            "glassrip.edge_direction",
            &["glassrip.board_validate", "glassrip.canvas_crop"],
        ),
        d(
            "board_state",
            "glassrip.board_state",
            &[
                "glassrip.board_validate",
                "glassrip.edge_direction",
                "glassrip.keyframes",
                "glassrip.ocr",
            ],
        ),
        d("audio_extract", "glassrip.audio", &["glassrip.media_probe"]),
        d(
            "asr",
            "glassrip.asr",
            &["glassrip.audio", "glassrip.asr_vocabulary"],
        ),
        d("diarize", "glassrip.diarization", &["glassrip.audio"]),
        d(
            "assign_words",
            "glassrip.transcript",
            &["glassrip.asr", "glassrip.diarization"],
        ),
        d(
            "name_speakers",
            "glassrip.speakers",
            &["glassrip.transcript", "glassrip.keyframes", "glassrip.ocr"],
        ),
        d(
            "notes",
            "glassrip.meeting_notes",
            &[
                "glassrip.board_state",
                "glassrip.speakers",
                "glassrip.transcript",
            ],
        ),
        d(
            "render",
            "glassrip.render",
            &["glassrip.meeting_notes", "glassrip.board_state"],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> StageGraph {
        StageGraph::new(vec![
            StageDecl::new("a", "x.a", &[]),
            StageDecl::new("b", "x.b", &["x.a"]),
            StageDecl::new("c", "x.c", &["x.b"]),
            StageDecl::new("d", "x.d", &["x.a"]),
        ])
        .unwrap()
    }

    #[test]
    fn meeting_graph_is_a_dag() {
        let g = StageGraph::new(meeting_mode_stage_decls()).unwrap();
        let order = g.order();
        let pos = |s: &str| order.iter().position(|x| *x == s).unwrap();
        assert!(pos("frames") < pos("features") && pos("frames") < pos("screen_quad"));
        assert!(pos("screen_quad") < pos("rectify") && pos("features") < pos("keyframes"));
        assert!(!g.ancestors("features").unwrap().contains("screen_quad"));
        assert!(pos("keyframes") < pos("rectify"));
        assert!(pos("ocr_harvest") < pos("classify"));
        assert!(pos("ocr_harvest") < pos("asr"));
        // The ASR vocabulary comes from on-screen text and is consumed by asr.
        assert!(pos("ocr_vocabulary") < pos("asr"));
        assert!(
            g.decl("asr")
                .is_some_and(|d| d.inputs.iter().any(|i| i == "glassrip.asr_vocabulary"))
        );
        assert!(g.descendants("ocr_vocabulary").unwrap().contains("asr"));
        assert!(pos("board_state") < pos("notes"));
        assert_eq!(order.last(), Some(&"render"));
        // board_read takes no transcript input in v1.
        let upstream = g.ancestors("board_read").unwrap();
        assert!(!upstream.contains("asr") && !upstream.contains("assign_words"));
    }

    #[test]
    fn cycle_rejected() {
        let err = StageGraph::new(vec![
            StageDecl::new("root", "x.root", &[]),
            StageDecl::new("a", "x.a", &["x.root", "x.c"]),
            StageDecl::new("b", "x.b", &["x.a"]),
            StageDecl::new("c", "x.c", &["x.b"]),
        ])
        .unwrap_err();
        match err {
            GraphError::Cycle(stages) => {
                assert_eq!(stages.first(), stages.last());
                let inner: BTreeSet<_> = stages.iter().cloned().collect();
                assert_eq!(inner, BTreeSet::from(["a".into(), "b".into(), "c".into()]));
            }
            other => panic!("expected cycle, got {other:?}"),
        }
        assert!(matches!(
            StageGraph::new(vec![StageDecl::new("self", "x.s", &["x.s"])]),
            Err(GraphError::Cycle(_))
        ));
    }

    #[test]
    fn missing_input_and_duplicates_rejected() {
        assert!(matches!(
            StageGraph::new(vec![StageDecl::new("a", "x.a", &["x.nope"])]),
            Err(GraphError::MissingInput { .. })
        ));
        assert!(matches!(
            StageGraph::new(vec![
                StageDecl::new("a", "x.a", &[]),
                StageDecl::new("a", "x.b", &[])
            ]),
            Err(GraphError::DuplicateStage(_))
        ));
        assert!(matches!(
            StageGraph::new(vec![
                StageDecl::new("a", "x.a", &[]),
                StageDecl::new("b", "x.a", &[])
            ]),
            Err(GraphError::DuplicateProducer { .. })
        ));
    }

    #[test]
    fn deterministic_order() {
        assert_eq!(chain().order(), vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn selection_semantics() {
        let g = chain();
        let all = g.plan(&Selection::default()).unwrap();
        assert_eq!(all.selected(), vec!["a", "b", "c", "d"]);
        assert_eq!(all.decision("a"), Some(StageDecision::Run { force: false }));

        let from_b = g
            .plan(&Selection {
                from: Some("b".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(from_b.selected(), vec!["b", "c"]);
        assert_eq!(
            from_b.decision("b"),
            Some(StageDecision::Run { force: true })
        );
        assert_eq!(
            from_b.decision("c"),
            Some(StageDecision::Run { force: false })
        );
        assert_eq!(
            from_b.decision("a"),
            Some(StageDecision::Skip(SkipReason::BeforeFrom))
        );
        assert_eq!(
            from_b.decision("d"),
            Some(StageDecision::Skip(SkipReason::BeforeFrom))
        );

        let until_b = g
            .plan(&Selection {
                until: Some("b".into()),
                force: BTreeSet::from(["a".to_string()]),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(until_b.selected(), vec!["a", "b"]);
        assert_eq!(
            until_b.decision("a"),
            Some(StageDecision::Run { force: true })
        );
        assert_eq!(
            until_b.decision("c"),
            Some(StageDecision::Skip(SkipReason::AfterUntil))
        );

        assert!(matches!(
            g.plan(&Selection {
                from: Some("c".into()),
                until: Some("d".into()),
                ..Default::default()
            }),
            Err(GraphError::EmptySelection { .. })
        ));
        assert!(matches!(
            g.plan(&Selection {
                until: Some("zzz".into()),
                ..Default::default()
            }),
            Err(GraphError::UnknownStage(_))
        ));
    }
}
