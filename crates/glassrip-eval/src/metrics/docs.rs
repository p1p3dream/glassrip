//! Document-mode metrics (spec 9.4).
//!
//! | Metric | Definition here |
//! |---|---|
//! | Body CER | CER of the rendered body text (explicit `body_text`, else block contents joined by newlines) vs the gold body, after whitespace normalization |
//! | Structured ticket fields | exact match, case-insensitive and trimmed, on `key`, `title`, `type`, `status`, `priority`, `assignee`, `reporter`, `epic`, `sprint`; `labels` as a sorted normalized list; `links` as a sorted list of `relation key`; pooled over issues |
//! | Free-text ticket fields | pooled CER per issue over `description`, `acceptance_criteria` (joined by newlines), and `comments[i].body` paired by index (missing or extra comments count fully); reported as the per-issue median |
//! | Block-structure F1 | longest common subsequence of block kind signatures (`heading2`, `paragraph`, `list`, `table`, `code`, `quote`, `image`); pooled |
//! | Table cell accuracy | tables paired by order; every gold cell is correct when the predicted cell at the same row and column matches after whitespace normalization; gold columns beyond the predicted table width are also counted as truncated |
//! | Coverage | the pipeline's `completeness.coverage` per page (minimum and per page) |
//! | Hallucinated spans | output text fields with no provenance entry from OCR and no model-only flag |

use serde::{Deserialize, Serialize};

use super::{Counts, Tally};
use crate::text::{cer_count, collapse_ws, normalize_label, ErrorCount};

/// Page type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocType {
    /// Issue tracker ticket.
    Ticket,
    /// Documentation page.
    Doc,
}

/// Block kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    /// Heading (with `level`).
    Heading,
    /// Paragraph.
    Paragraph,
    /// Bulleted or numbered list.
    List,
    /// Table (with `rows`).
    Table,
    /// Code block (with `lang`).
    Code,
    /// Quote.
    Quote,
    /// Image.
    Image,
}

/// One block of a documentation page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    /// Kind.
    pub kind: BlockKind,
    /// Heading level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<u8>,
    /// Code language.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Text content (list items joined by newlines; table cells by tabs and newlines).
    #[serde(default)]
    pub content: String,
    /// Table rows (tables only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<Vec<String>>,
}

impl Block {
    /// Kind signature used for block-structure matching.
    pub fn signature(&self) -> String {
        match self.kind {
            BlockKind::Heading => format!("heading{}", self.level.unwrap_or(1)),
            other => serde_json::to_value(other)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
        }
    }
}

/// Ticket comment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Comment {
    /// Author.
    #[serde(default)]
    pub author: String,
    /// Time as shown.
    #[serde(default)]
    pub time: String,
    /// Body.
    #[serde(default)]
    pub body: String,
}

/// Issue link.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Link {
    /// Relation (for example `blocks`).
    #[serde(default)]
    pub relation: String,
    /// Linked key.
    #[serde(default)]
    pub key: String,
}

/// Ticket fields (spec 6.16).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TicketFields {
    /// Issue key.
    pub key: String,
    /// Title.
    pub title: String,
    /// Issue type.
    #[serde(rename = "type")]
    pub issue_type: String,
    /// Status.
    pub status: String,
    /// Priority.
    pub priority: String,
    /// Assignee.
    pub assignee: String,
    /// Reporter.
    pub reporter: String,
    /// Labels.
    pub labels: Vec<String>,
    /// Epic.
    pub epic: String,
    /// Sprint.
    pub sprint: String,
    /// Description (markdown).
    pub description: String,
    /// Acceptance criteria.
    pub acceptance_criteria: Vec<String>,
    /// Comments.
    pub comments: Vec<Comment>,
    /// Links.
    pub links: Vec<Link>,
    /// Attachment names.
    pub attachments: Vec<String>,
}

/// A document page (gold or predicted).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Page id.
    pub page_id: String,
    /// Type.
    #[serde(rename = "type")]
    pub doc_type: DocType,
    /// Title.
    #[serde(default)]
    pub title: String,
    /// Ticket fields (tickets only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<TicketFields>,
    /// Blocks (documentation pages).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<Block>,
    /// Full body text in reading order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_text: Option<String>,
}

impl Document {
    /// Body text: explicit `body_text`, else block contents (or ticket free text) joined by newlines.
    pub fn body(&self) -> String {
        if let Some(b) = &self.body_text {
            return b.clone();
        }
        if let Some(f) = &self.fields {
            let mut parts = vec![f.description.clone()];
            parts.extend(f.acceptance_criteria.iter().cloned());
            parts.extend(f.comments.iter().map(|c| c.body.clone()));
            return parts.join("\n");
        }
        self.blocks
            .iter()
            .map(|b| b.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Tables in order.
    pub fn tables(&self) -> Vec<&Vec<Vec<String>>> {
        self.blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Table)
            .map(|b| &b.rows)
            .collect()
    }
}

/// Where a predicted field's text came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Field path, for example `title`, `description`, `comments[1].body`, `blocks[3]`.
    pub field: String,
    /// `ocr` or `model`.
    #[serde(default)]
    pub source: String,
    /// Model text explicitly flagged as model-only.
    #[serde(default)]
    pub model_only: bool,
}

/// Predicted page with provenance and completeness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredDocument {
    /// The page content.
    #[serde(flatten)]
    pub doc: Document,
    /// Per-field provenance.
    #[serde(default)]
    pub provenance: Vec<Provenance>,
    /// `completeness.coverage` from the pipeline.
    #[serde(default)]
    pub coverage: Option<f64>,
}

/// Structured ticket fields: pooled exact matches over the 11 fields.
pub fn structured_fields(gold: &TicketFields, pred: &TicketFields) -> Tally {
    let eq = |a: &str, b: &str| a.trim().to_lowercase() == b.trim().to_lowercase();
    let list = |v: &[String]| {
        let mut out: Vec<String> = v.iter().map(|s| normalize_label(s)).collect();
        out.sort();
        out
    };
    let links = |v: &[Link]| {
        let mut out: Vec<String> = v
            .iter()
            .map(|l| {
                format!(
                    "{} {}",
                    normalize_label(&l.relation),
                    normalize_label(&l.key)
                )
            })
            .collect();
        out.sort();
        out
    };
    let mut t = Tally::default();
    for (a, b) in [
        (&gold.key, &pred.key),
        (&gold.title, &pred.title),
        (&gold.issue_type, &pred.issue_type),
        (&gold.status, &pred.status),
        (&gold.priority, &pred.priority),
        (&gold.assignee, &pred.assignee),
        (&gold.reporter, &pred.reporter),
        (&gold.epic, &pred.epic),
        (&gold.sprint, &pred.sprint),
    ] {
        t.record(eq(a, b));
    }
    t.record(list(&gold.labels) == list(&pred.labels));
    t.record(links(&gold.links) == links(&pred.links));
    t
}

/// Free-text CER counts for one issue.
pub fn free_text_errors(gold: &TicketFields, pred: &TicketFields) -> ErrorCount {
    let mut e = cer_count(&pred.description, &gold.description);
    e.add(cer_count(
        &pred.acceptance_criteria.join("\n"),
        &gold.acceptance_criteria.join("\n"),
    ));
    let n = gold.comments.len().max(pred.comments.len());
    for i in 0..n {
        let g = gold.comments.get(i).map_or("", |c| c.body.as_str());
        let p = pred.comments.get(i).map_or("", |c| c.body.as_str());
        // An extra predicted comment has an empty reference: every character is an insertion.
        e.add(cer_count(p, g));
    }
    e
}

/// Block-structure counts: LCS of kind signatures.
pub fn block_structure(gold: &[Block], pred: &[Block]) -> Counts {
    let g: Vec<String> = gold.iter().map(Block::signature).collect();
    let p: Vec<String> = pred.iter().map(Block::signature).collect();
    let lcs = lcs_len(&g, &p);
    Counts::from_matches(lcs, g.len(), p.len())
}

fn lcs_len<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            dp[i + 1][j + 1] = if a[i] == b[j] {
                dp[i][j] + 1
            } else {
                dp[i][j + 1].max(dp[i + 1][j])
            };
        }
    }
    dp[a.len()][b.len()]
}

/// Table cell results.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableCells {
    /// Correct over all gold cells.
    pub cells: Tally,
    /// Gold cells in columns the prediction never captured.
    pub truncated_cells: usize,
}

impl TableCells {
    /// Pools another result.
    pub fn add(&mut self, o: TableCells) {
        self.cells.add(o.cells);
        self.truncated_cells += o.truncated_cells;
    }
}

/// Table cell accuracy with tables paired by order.
pub fn table_cells(gold: &[&Vec<Vec<String>>], pred: &[&Vec<Vec<String>>]) -> TableCells {
    let mut out = TableCells::default();
    for (i, g) in gold.iter().enumerate() {
        let p = pred.get(i);
        let width = p.map_or(0, |p| p.iter().map(Vec::len).max().unwrap_or(0));
        for (r, row) in g.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                let got = p.and_then(|p| p.get(r)).and_then(|row| row.get(c));
                out.cells
                    .record(got.is_some_and(|x| collapse_ws(x) == collapse_ws(cell)));
                if c >= width {
                    out.truncated_cells += 1;
                }
            }
        }
    }
    out
}

/// Text fields of a document as `(field path, text)`, non-empty only.
pub fn text_fields(doc: &Document) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut push = |k: String, v: &str| {
        if !v.trim().is_empty() {
            out.push((k, v.to_string()));
        }
    };
    push("title".into(), &doc.title);
    if let Some(f) = &doc.fields {
        for (k, v) in [
            ("key", &f.key),
            ("type", &f.issue_type),
            ("status", &f.status),
            ("priority", &f.priority),
            ("assignee", &f.assignee),
            ("reporter", &f.reporter),
            ("epic", &f.epic),
            ("sprint", &f.sprint),
            ("description", &f.description),
        ] {
            push(k.into(), v);
        }
        for (i, l) in f.labels.iter().enumerate() {
            push(format!("labels[{i}]"), l);
        }
        for (i, a) in f.acceptance_criteria.iter().enumerate() {
            push(format!("acceptance_criteria[{i}]"), a);
        }
        for (i, c) in f.comments.iter().enumerate() {
            push(format!("comments[{i}].body"), &c.body);
        }
    }
    for (i, b) in doc.blocks.iter().enumerate() {
        push(format!("blocks[{i}]"), &b.content);
    }
    out
}

/// Fields with text but neither OCR provenance nor a model-only flag.
pub fn hallucinated_spans(pred: &PredDocument) -> Vec<String> {
    text_fields(&pred.doc)
        .into_iter()
        .filter(|(k, _)| {
            !pred
                .provenance
                .iter()
                .any(|p| p.field == *k && (p.source.eq_ignore_ascii_case("ocr") || p.model_only))
        })
        .map(|(k, _)| k)
        .collect()
}

/// Per-page document scores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DocScore {
    /// Body character errors.
    pub body: ErrorCount,
    /// Structured field matches (tickets).
    pub fields: Tally,
    /// Free-text character errors (tickets).
    pub free_text: Option<ErrorCount>,
    /// Block structure counts.
    pub blocks: Counts,
    /// Table cells.
    pub tables: TableCells,
    /// Reported coverage.
    pub coverage: Option<f64>,
    /// Hallucinated field paths.
    pub hallucinated: Vec<String>,
}

/// Scores one predicted page against its gold page.
pub fn score_document(gold: &Document, pred: &PredDocument) -> DocScore {
    let mut s = DocScore {
        body: cer_count(&pred.doc.body(), &gold.body()),
        blocks: block_structure(&gold.blocks, &pred.doc.blocks),
        tables: table_cells(&gold.tables(), &pred.doc.tables()),
        coverage: pred.coverage,
        hallucinated: hallucinated_spans(pred),
        ..Default::default()
    };
    if let Some(gf) = &gold.fields {
        let empty = TicketFields::default();
        let pf = pred.doc.fields.as_ref().unwrap_or(&empty);
        s.fields = structured_fields(gf, pf);
        s.free_text = Some(free_text_errors(gf, pf));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(kind: BlockKind, level: Option<u8>, content: &str) -> Block {
        Block {
            kind,
            level,
            lang: None,
            content: content.into(),
            rows: vec![],
        }
    }

    fn ticket() -> TicketFields {
        TicketFields {
            key: "ORB-12".into(),
            title: "Retry uploads".into(),
            issue_type: "Story".into(),
            status: "In Progress".into(),
            priority: "High".into(),
            assignee: "Avery Stone".into(),
            reporter: "Jordan Vale".into(),
            labels: vec!["backend".into(), "queue".into()],
            epic: "ORB-1".into(),
            sprint: "Sprint 4".into(),
            description: "Retry failed uploads".into(),
            acceptance_criteria: vec!["Retries 3 times".into()],
            comments: vec![Comment {
                author: "Riley".into(),
                time: "2d".into(),
                body: "Looks good".into(),
            }],
            links: vec![Link {
                relation: "blocks".into(),
                key: "ORB-13".into(),
            }],
            attachments: vec![],
        }
    }

    #[test]
    fn structured_fields_hand_computed() {
        let g = ticket();
        let mut p = ticket();
        p.status = " in progress ".into(); // case/trim: still correct
        p.priority = "Medium".into(); // wrong
        p.labels = vec!["Queue".into(), "backend".into()]; // order and case ignored
        p.links = vec![]; // wrong
        let t = structured_fields(&g, &p);
        assert_eq!(
            t,
            Tally {
                correct: 9,
                total: 11
            }
        );
    }

    #[test]
    fn free_text_hand_computed() {
        let g = ticket();
        let mut p = ticket();
        p.description = "Retry faild uploads".into(); // 1 deletion of 20
        p.comments.push(Comment {
            body: "extra".into(),
            ..Default::default()
        }); // 5 insertions
        let e = free_text_errors(&g, &p);
        // reference: 20 (description) + 15 (criteria) + 10 (comment) = 45; errors 1 + 5
        assert_eq!(
            e,
            ErrorCount {
                errors: 6,
                reference_len: 45
            }
        );
    }

    #[test]
    fn block_structure_lcs() {
        let gold = vec![
            block(BlockKind::Heading, Some(1), "Intro"),
            block(BlockKind::Paragraph, None, "a"),
            block(BlockKind::List, None, "b"),
            block(BlockKind::Code, None, "c"),
        ];
        let pred = vec![
            block(BlockKind::Heading, Some(2), "Intro"), // wrong level
            block(BlockKind::Paragraph, None, "a"),
            block(BlockKind::Code, None, "c"),
        ];
        // LCS = paragraph, code = 2: P = 2/3, R = 2/4
        assert_eq!(
            block_structure(&gold, &pred),
            Counts {
                tp: 2,
                fp: 1,
                fn_: 2
            }
        );
    }

    #[test]
    fn table_cells_and_truncation() {
        let g = vec![
            vec!["a".to_string(), "b".into(), "c".into()],
            vec!["1".to_string(), "2".into(), "3".into()],
        ];
        let p = vec![
            vec!["a".to_string(), "b".into()],
            vec!["1".to_string(), "x".into()],
        ];
        let t = table_cells(&[&g], &[&p]);
        // correct: a, b, 1 = 3 of 6; column 3 never captured: 2 truncated
        assert_eq!(
            t.cells,
            Tally {
                correct: 3,
                total: 6
            }
        );
        assert_eq!(t.truncated_cells, 2);
        let missing = table_cells(&[&g], &[]);
        assert_eq!(
            missing.cells,
            Tally {
                correct: 0,
                total: 6
            }
        );
        assert_eq!(missing.truncated_cells, 6);
    }

    #[test]
    fn hallucination_and_body() {
        let doc = Document {
            page_id: "p".into(),
            doc_type: DocType::Doc,
            title: "Guide".into(),
            fields: None,
            blocks: vec![
                block(BlockKind::Paragraph, None, "from ocr"),
                block(BlockKind::Paragraph, None, "model flagged"),
                block(BlockKind::Paragraph, None, "model unflagged"),
            ],
            body_text: None,
        };
        let pred = PredDocument {
            doc: doc.clone(),
            provenance: vec![
                Provenance {
                    field: "title".into(),
                    source: "ocr".into(),
                    model_only: false,
                },
                Provenance {
                    field: "blocks[0]".into(),
                    source: "ocr".into(),
                    model_only: false,
                },
                Provenance {
                    field: "blocks[1]".into(),
                    source: "model".into(),
                    model_only: true,
                },
                Provenance {
                    field: "blocks[2]".into(),
                    source: "model".into(),
                    model_only: false,
                },
            ],
            coverage: Some(0.97),
        };
        assert_eq!(hallucinated_spans(&pred), vec!["blocks[2]".to_string()]);
        assert_eq!(doc.body(), "from ocr\nmodel flagged\nmodel unflagged");
        let s = score_document(&doc, &pred);
        assert_eq!(s.body.errors, 0);
        assert_eq!(
            s.blocks,
            Counts {
                tp: 3,
                fp: 0,
                fn_: 0
            }
        );
        assert_eq!(s.coverage, Some(0.97));
        assert!(s.free_text.is_none());
    }
}
