//! `glassrip meeting` through its command entry point: a run that fails
//! preflight writes nothing (no output directory, no log). Fictional input.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use clap::Parser;
use glassrip::meeting::MeetingArgs;

#[derive(Parser)]
struct Wrap {
    #[command(flatten)]
    args: MeetingArgs,
}

#[tokio::test]
async fn failed_preflight_leaves_no_output_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let video = tmp.path().join("call.mp4");
    std::fs::write(&video, b"not decoded before preflight").unwrap();
    let out = tmp.path().join("call.glassrip");
    // `asr` needs a whisper model; this one does not exist, so preflight fails
    // whatever the build features, without contacting a model server.
    let args = Wrap::try_parse_from([
        "meeting",
        video.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--asr-model",
        tmp.path().join("missing-model.bin").to_str().unwrap(),
        "--until-stage",
        "asr",
    ])
    .unwrap()
    .args;
    let err = glassrip::meeting::cli::run(args).await.unwrap_err();
    let text = format!("{err:#}");
    assert!(text.contains("preflight failed"), "{text}");
    assert!(text.contains("asr: whisper model"), "{text}");
    assert!(text.contains("nothing was written"), "{text}");
    assert!(
        !out.exists(),
        "a failed preflight must not create {}",
        out.display()
    );
}
