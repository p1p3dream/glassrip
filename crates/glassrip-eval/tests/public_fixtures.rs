//! The committed public fixtures are exactly what the generator writes, they
//! load and validate, and the repository passes the built-in privacy checks.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use glassrip_eval::fixture::{load_board_suite, load_docs_suite};

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn generator_is_deterministic_and_matches_committed_fixtures() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let ga = glassrip_eval::synth::generate_all(a.path()).unwrap();
    let gb = glassrip_eval::synth::generate_all(b.path()).unwrap();
    assert_eq!(ga.files, gb.files);
    let committed = fixtures_root();
    for rel in &ga.files {
        let fa = fs_err::read(a.path().join(rel)).unwrap();
        let fb = fs_err::read(b.path().join(rel)).unwrap();
        assert_eq!(fa, fb, "{} differs between two runs", rel.display());
        let fc = committed.join(rel);
        assert!(
            fc.is_file(),
            "{} is not committed; run gen_synthetic",
            rel.display()
        );
        if rel.extension().and_then(|e| e.to_str()) == Some("png") {
            // Compare pixels, so an encoder upgrade alone does not fail the test.
            let pa = image::open(a.path().join(rel)).unwrap().to_rgb8();
            let pc = image::open(&fc).unwrap().to_rgb8();
            assert!(
                pa == pc,
                "{} pixels differ from the committed fixture",
                rel.display()
            );
        } else {
            let text_c = fs_err::read_to_string(&fc).unwrap();
            assert_eq!(
                String::from_utf8(fa).unwrap(),
                text_c,
                "{} differs from the committed fixture",
                rel.display()
            );
        }
    }
    assert!(
        ga.bytes < 4 * 1024 * 1024,
        "public fixtures should stay small"
    );
}

#[test]
fn committed_suites_load() {
    let boards = load_board_suite(&fixtures_root().join("synthetic")).unwrap();
    assert!(boards.len() >= 10);
    assert!(boards
        .iter()
        .any(|c| c.meta.tags.iter().any(|t| t == "trap")));
    let docs = load_docs_suite(&fixtures_root().join("synthetic_docs")).unwrap();
    assert!(docs.len() >= 3);
}

#[test]
fn tracked_files_pass_builtin_privacy_checks() {
    let root = workspace_root();
    let Some(files) = glassrip_eval::privacy::tracked_files(&root) else {
        eprintln!("git unavailable; skipping");
        return;
    };
    let findings = glassrip_eval::privacy::scan_builtin(&root, &files);
    assert!(findings.is_empty(), "{findings:#?}");
}

#[test]
fn docs_suite_scores_predictions_end_to_end() {
    use glassrip_eval::metrics::docs::{text_fields, PredDocument, Provenance};
    let cases = load_docs_suite(&fixtures_root().join("synthetic_docs")).unwrap();
    let preds = tempfile::tempdir().unwrap();
    for (i, case) in cases.iter().enumerate() {
        let doc = case.expected.document.clone();
        // Every field has OCR provenance, except the first case, whose title is unflagged model text.
        let provenance = text_fields(&doc)
            .into_iter()
            .map(|(field, _)| Provenance {
                source: if i == 0 && field == "title" {
                    "model".into()
                } else {
                    "ocr".into()
                },
                field,
                model_only: false,
            })
            .collect();
        let pred = PredDocument {
            doc,
            provenance,
            coverage: Some(1.0),
        };
        let dir = preds.path().join(&case.name);
        fs_err::create_dir_all(&dir).unwrap();
        fs_err::write(
            dir.join("documents.json"),
            serde_json::to_string(&pred).unwrap(),
        )
        .unwrap();
    }
    let run = glassrip_eval::suite::run_docs_suite(&cases, Some(preds.path()));
    let m = &run.metrics;
    assert!(run.errors.is_empty(), "{:?}", run.errors);
    assert_eq!(m["docs.body_cer"], 0.0);
    assert_eq!(m["docs.fields.accuracy"], 1.0);
    assert_eq!(m["docs.blocks.f1"], 1.0);
    assert_eq!(m["docs.tables.cell_accuracy"], 1.0);
    assert_eq!(m["docs.coverage_min"], 1.0);
    assert_eq!(m["docs.hallucinated_spans"], 1.0);
    assert_eq!(m["docs.pages_missing"], 0.0);
}

#[test]
fn committed_hashed_denylist_and_allowlist_work_together() {
    use glassrip_eval::privacy::{
        scan_text, Allowlist, HashedDenylist, Matcher, ALLOWLIST, HASHED_DENYLIST,
    };
    let root = workspace_root();
    let list = HashedDenylist::parse(&fs_err::read_to_string(root.join(HASHED_DENYLIST)).unwrap())
        .unwrap();
    assert!(!list.hashes.is_empty());
    let allow = Allowlist::parse(&fs_err::read_to_string(root.join(ALLOWLIST)).unwrap());
    let m = Matcher::Hashed(&list);
    let words = "vitamin\nviv\nvivid\n";
    let audio = Path::new("crates/glassrip-audio/common_words.txt");
    assert!(scan_text(audio, words, &m, &allow, "h").is_empty());
    assert_eq!(
        scan_text(Path::new("src/other.rs"), words, &m, &allow, "h").len(),
        1
    );
}
