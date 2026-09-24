//! Markdown notes and their round-trip check.

use std::collections::BTreeSet;

use glassrip_notes::board::{first_seen, last_seen, removed_at, BoardExt, BoardStateItem};
use glassrip_notes::notes::{MeetingNotes, NotesStatus, QuestionSource, Quote, QuoteMatch};
use glassrip_notes::text::{mmss, sanitize_dashes};
use minijinja::Environment;
use pulldown_cmark::{Event, Options, Parser, Tag};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::facts::{deferred_nodes, derive_grids, final_nodes, node_owners, owner_summary};
use crate::style::role_of;
use crate::RenderError;

/// Escapes text for a table cell.
pub fn cell(s: String) -> String {
    s.replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// Escapes model, board and transcript text for inline markdown, so it cannot
/// start headings, emphasis, links, images, code spans or raw HTML. Pipes are
/// left to [`cell`].
pub fn inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' | '\r' => out.push(' '),
            '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '#' | '!' | '~' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn when(a: f64, b: f64) -> String {
    if b - a >= 5.0 {
        format!("{} to {}", mmss(a), mmss(b))
    } else {
        mmss(a)
    }
}

fn quote(q: &Option<Quote>) -> String {
    match q {
        Some(q) if q.matched == QuoteMatch::Corrected => format!(
            " \"{}\" (quote matches the vocabulary-corrected transcript)",
            inline(&q.text)
        ),
        Some(q) => format!(" \"{}\"", inline(&q.text)),
        None => String::new(),
    }
}

/// Where the rendered files are, for links.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Links {
    /// SVG file name per board id.
    pub svg: Vec<(String, String)>,
    /// PNG preview file name per board id.
    pub png: Vec<(String, String)>,
}

/// Header facts that do not come from the notes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MarkdownMeta {
    /// Meeting date and time text.
    pub date: Option<String>,
    /// Source description.
    pub source: Option<String>,
}

#[derive(Serialize)]
struct Arch {
    title: String,
    svg: Option<String>,
    png: Option<String>,
    final_time: String,
    rows: Vec<[String; 4]>,
    groups: Vec<String>,
}

#[derive(Serialize)]
struct Owner {
    name: String,
    items: Vec<String>,
}

#[derive(Serialize)]
struct Ctx {
    title: String,
    meta: Vec<(String, String)>,
    summary: Vec<String>,
    boards: Vec<Arch>,
    decisions: Vec<String>,
    owners: Vec<Owner>,
    questions: Vec<String>,
    timeline: Vec<(String, String)>,
    caveats: Vec<String>,
    speakers: Vec<[String; 4]>,
    transcript: Vec<String>,
}

/// Renders the notes as markdown.
pub fn render_markdown(
    env: &Environment<'_>,
    notes: &MeetingNotes,
    boards: &[BoardStateItem],
    links: &Links,
    meta: &MarkdownMeta,
) -> Result<String, RenderError> {
    let s = |t: &str| inline(&sanitize_dashes(t));
    let title = notes
        .title
        .clone()
        .unwrap_or_else(|| "Meeting notes".into());
    let mut m = Vec::new();
    if let Some(d) = &meta.date {
        m.push(("Date".to_string(), d.clone()));
    }
    m.push((
        "Length".into(),
        format!("{} min", (notes.duration_s / 60.0).round()),
    ));
    let people: Vec<String> = notes
        .people
        .iter()
        .map(|p| {
            if notes.presenter.as_deref() == Some(p.display_name.as_str()) {
                format!("{} (presenting)", p.display_name)
            } else {
                p.display_name.clone()
            }
        })
        .collect();
    m.push(("Participants".into(), people.join(", ")));
    m.push((
        "Source".into(),
        meta.source.clone().unwrap_or_else(|| {
            "Meeting recording. Speaker-attributed transcript and board read by glassrip; every item cites transcript segments or board events and was checked by the validator.".into()
        }),
    ));
    if notes.report.status == NotesStatus::Degraded {
        m.push(("Status".into(), "**Degraded**: see Caveats".into()));
    }

    let summary = notes
        .summary
        .iter()
        .map(|p| format!("{} ({})", s(&p.text), when(p.t_start_s, p.t_end_s)))
        .collect();

    let mut archs = Vec::new();
    for b in boards {
        let deferred = deferred_nodes(b, &notes.decisions);
        let owners = node_owners(b, notes);
        let grids = derive_grids(b);
        let mut rows: Vec<[String; 4]> = final_nodes(b)
            .into_iter()
            .map(|n| {
                let status = match deferred.get(&n.id) {
                    Some(t) => format!("**Deferred** ({})", mmss(*t)),
                    None => format!("On the board from {}", mmss(first_seen(&n.lifetimes))),
                };
                [
                    cell(s(&n.text)),
                    role_of(&n.text).describe().to_string(),
                    cell(s(&owner_summary(
                        owners.get(&n.id).map_or(&[][..], Vec::as_slice),
                    ))),
                    status,
                ]
            })
            .collect();
        for n in b.nodes.iter().filter(|n| !n.in_final) {
            let gone = removed_at(&n.lifetimes)
                .or_else(|| last_seen(&n.lifetimes))
                .unwrap_or(0.0);
            rows.push([
                cell(s(&n.text)),
                role_of(&n.text).describe().to_string(),
                "none shown".into(),
                format!(
                    "Removed (seen {} to {})",
                    mmss(first_seen(&n.lifetimes)),
                    mmss(gone)
                ),
            ]);
        }
        let groups = grids
            .iter()
            .map(|g| {
                let members: Vec<String> = g.members.iter().map(|m| s(&m.text)).collect();
                format!(
                    "**Grouped boxes** ({} rows by {} columns, no arrows, from {}): {}",
                    g.rows,
                    g.columns,
                    mmss(g.first_seen_s),
                    members.join(", ")
                )
            })
            .collect();
        let find = |v: &Vec<(String, String)>| {
            v.iter()
                .find(|(id, _)| id == &b.board_id)
                .map(|x| x.1.clone())
        };
        archs.push(Arch {
            title: s(&b.board_id),
            svg: find(&links.svg),
            png: find(&links.png),
            final_time: mmss(b.end_s()),
            rows,
            groups,
        });
    }

    let decisions = notes
        .decisions
        .iter()
        .map(|d| {
            format!(
                "**{}** ({}).{}",
                s(&d.text),
                when(d.t_start_s, d.t_end_s),
                quote(&d.quote)
            )
        })
        .collect();

    let mut order: Vec<String> = notes
        .people
        .iter()
        .map(|p| p.display_name.clone())
        .collect();
    order.push("Everyone".into());
    let mut owners: Vec<Owner> = Vec::new();
    for name in order {
        let items: Vec<String> = notes
            .action_items
            .iter()
            .filter(|a| a.owner == name)
            .map(|a| {
                format!(
                    "{} ({}).{}",
                    s(&a.task),
                    when(a.t_s, a.t_end_s),
                    quote(&a.quote)
                )
            })
            .collect();
        if !items.is_empty() {
            owners.push(Owner { name, items });
        }
    }

    let questions = notes
        .open_questions
        .iter()
        .map(|q| {
            let t = q.t_s.map(mmss).unwrap_or_default();
            let src = match q.source {
                QuestionSource::Transcript => format!("asked at {t}"),
                QuestionSource::Board => format!("board sticky, first seen {t}"),
                QuestionSource::BoardAndTranscript => format!("board sticky; discussed at {t}"),
            };
            format!("**{}** ({src}).{}", s(&q.text), quote(&q.quote))
        })
        .collect();

    let timeline = notes
        .timeline
        .iter()
        .map(|t| (when(t.t_start_s, t.t_end_s), cell(s(&t.text))))
        .collect();
    let caveats = notes.caveats.iter().map(|c| s(&c.text)).collect();
    let speakers = notes
        .speakers
        .iter()
        .map(|l| {
            [
                l.label.clone(),
                cell(l.name.clone()),
                format!("{:.2}", l.confidence),
                format!("{:.0} s", l.talk_time_s),
            ]
        })
        .collect();
    let transcript = notes
        .transcript
        .iter()
        .map(|p| {
            let who = if p.flagged {
                format!("{} (speaker uncertain)", inline(&p.speaker))
            } else {
                inline(&p.speaker)
            };
            format!(
                "[{}] {who}: {}",
                mmss(p.start_s),
                inline(&sanitize_dashes(p.text.trim()))
            )
        })
        .collect();

    let ctx = Ctx {
        title: s(&title),
        meta: m.into_iter().map(|(k, v)| (k, cell(v))).collect(),
        summary,
        boards: archs,
        decisions,
        owners,
        questions,
        timeline,
        caveats,
        speakers,
        transcript,
    };
    let t = env
        .get_template("notes.md")
        .map_err(|e| RenderError::Template(format!("{e:#}")))?;
    let out = t
        .render(&ctx)
        .map_err(|e| RenderError::Template(format!("{e:#}")))?;
    Ok(sanitize_dashes_keep_lines(&out))
}

/// Applies the dash rule line by line (keeps line structure and table rules).
fn sanitize_dashes_keep_lines(s: &str) -> String {
    s.lines()
        .map(|l| {
            if l.contains('\u{2014}') || l.contains('\u{2013}') {
                sanitize_dashes(l)
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// Markdown round-trip results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MarkdownChecks {
    /// Table blocks in the source.
    pub tables_expected: usize,
    /// Tables the parser recognized.
    pub tables_parsed: usize,
    /// Table rows whose cell count differs from the header.
    pub bad_rows: Vec<String>,
    /// Links and images.
    pub links: usize,
    /// Links with an empty target.
    pub empty_links: usize,
    /// Relative targets that are not rendered files.
    pub missing_targets: Vec<String>,
    /// Headings.
    pub headings: usize,
    /// Em or en dashes present.
    pub dashes: bool,
    /// All checks passed.
    pub ok: bool,
}

fn cells(line: &str) -> usize {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    let mut n = 1;
    let mut prev = ' ';
    for c in t.chars() {
        if c == '|' && prev != '\\' {
            n += 1;
        }
        prev = c;
    }
    n
}

fn is_delimiter(line: &str) -> bool {
    let t = line.trim();
    t.contains('-') && t.contains('|') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// Parses the markdown back and checks tables, links and dashes.
pub fn check_markdown(md: &str, files: &BTreeSet<String>) -> MarkdownChecks {
    let lines: Vec<&str> = md.lines().collect();
    let mut tables_expected = 0;
    let mut bad_rows = Vec::new();
    let mut i = 1;
    while i < lines.len() {
        if is_delimiter(lines[i]) && lines[i - 1].contains('|') {
            tables_expected += 1;
            let width = cells(lines[i - 1]);
            if cells(lines[i]) != width {
                bad_rows.push(format!("delimiter row at line {}", i + 1));
            }
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start().starts_with('|') {
                if cells(lines[j]) != width {
                    bad_rows.push(format!(
                        "line {}: {} cells, header has {width}",
                        j + 1,
                        cells(lines[j])
                    ));
                }
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    let (mut tables_parsed, mut links, mut empty_links, mut headings) = (0, 0, 0, 0);
    let mut missing = Vec::new();
    for ev in Parser::new_ext(md, opts) {
        match ev {
            Event::Start(Tag::Table(_)) => tables_parsed += 1,
            Event::Start(Tag::Heading { .. }) => headings += 1,
            Event::Start(Tag::Link { dest_url, .. })
            | Event::Start(Tag::Image { dest_url, .. }) => {
                links += 1;
                let d = dest_url.to_string();
                if d.trim().is_empty() {
                    empty_links += 1;
                } else if !d.contains("://")
                    && !d.starts_with('#')
                    && !d.starts_with("mailto:")
                    && !files.contains(&d)
                {
                    missing.push(d);
                }
            }
            _ => {}
        }
    }
    let dashes = md.contains('\u{2014}') || md.contains('\u{2013}');
    let ok = tables_expected == tables_parsed
        && bad_rows.is_empty()
        && empty_links == 0
        && missing.is_empty()
        && !dashes
        && headings > 0;
    MarkdownChecks {
        tables_expected,
        tables_parsed,
        bad_rows,
        links,
        empty_links,
        missing_targets: missing,
        headings,
        dashes,
        ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broken_tables_and_links_are_caught() {
        let files: BTreeSet<String> = ["a.svg".to_string()].into_iter().collect();
        let good = "# T\n\n| a | b |\n|---|---|\n| 1 | 2 \\| 3 |\n\n![x](a.svg)\n";
        assert!(check_markdown(good, &files).ok);
        let bad = "# T\n\n| a | b |\n|---|---|\n| 1 | 2 | 3 |\n\n![x](b.svg)\n[y]()\n";
        let c = check_markdown(bad, &files);
        assert!(!c.ok);
        assert_eq!(c.bad_rows.len(), 1);
        assert_eq!(c.missing_targets, vec!["b.svg"]);
        assert_eq!(c.empty_links, 1);
    }

    #[test]
    fn model_text_cannot_inject_markdown() {
        let evil = "# Heading [click](http://x) **bold** <b>x</b> `code` ![img](a.png)";
        let md = format!("- {}\n", inline(evil));
        let mut opts = Options::empty();
        opts.insert(Options::ENABLE_TABLES);
        let bad = Parser::new_ext(&md, opts).any(|e| {
            matches!(
                e,
                Event::Start(Tag::Heading { .. })
                    | Event::Start(Tag::Link { .. })
                    | Event::Start(Tag::Image { .. })
                    | Event::Start(Tag::Strong)
                    | Event::Start(Tag::Emphasis)
                    | Event::Code(_)
                    | Event::Html(_)
                    | Event::InlineHtml(_)
            )
        });
        assert!(!bad, "{md}");
    }

    #[test]
    fn cells_are_escaped() {
        assert_eq!(cell("a|b\nc".into()), "a\\|b c");
    }
}
