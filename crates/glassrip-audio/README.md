# glassrip-audio

The audio branch of glassrip meeting mode, in Rust: audio extraction, speech
recognition, speaker diarization, and word-level speaker assignment. It produces
`glassrip.transcript` and `glassrip.speakers` artifacts.

## Pipeline

1. **Extract** (`extract`): `ffmpeg -i IN -map 0:a:0 -vn -ac 1 -ar 16000 -f f32le pipe:1`
   into `Vec<f32>`. ffprobe supplies the audio and video stream `start_time`,
   and every output time is shifted onto the video timeline.
2. **ASR** (`asr`): whisper.cpp through `whisper-rs` 0.16 with
   `ggml-large-v3-turbo` (or `ggml-large-v3`), beam 5, English, DTW token
   timestamps using the model's alignment-head preset, flash attention off (it
   disables DTW), `split_on_word`, and per-token probabilities. Words are built
   from tokens (`words`).
   - Silero VAD (`ggml-silero-v6.2.0.bin`) runs through whisper.cpp's VAD API
     inside this crate, and speech is decoded in chunks of at most 28 s. With
     whisper.cpp's internal VAD, only segment times are mapped back to the
     original timeline; token and DTW times are not. Chunking here keeps every
     word time on the original timeline.
   - Each chunk is decoded with the vocabulary prompt (`set_initial_prompt`,
     at most 200 tokens, names first), so the vocabulary conditions every
     window (whisper-rs 0.16 does not expose `carry_initial_prompt`).
3. **Vocabulary correction** (`vocab`): a word is replaced by a vocabulary term
   only when its probability is low, its phonetic key matches, and the
   normalized edit distance is small. Capitalized words matched to capitalized
   terms use a higher probability limit, since whisper often emits a known
   proper noun with fair confidence for an unfamiliar name. The key is a small
   Soundex-style code (digraph folding, voicing merge, vowels dropped).
   `text_raw` and `w_raw` keep the verbatim ASR output.
4. **Diarization** (`diarize`, feature `diarize`): `speakrs` 0.5 (pyannote
   community-1 port: segmentation-3.0, WeSpeaker ResNet34, PLDA, VBx). With a
   known speaker count K, cluster centroids are merged by agglomerative
   clustering (cosine, size-weighted centroid linkage) down to K (`recluster`),
   and every frame keeps one speaker so turns never overlap.
5. **Gap filling** (`gapfill`): on far-field recordings the segmentation model
   can mark long stretches of real speech as silence. ASR words outside every
   turn are grouped into spans, each span is embedded with the same WeSpeaker
   model (the span audio is tiled to fill the 10 s window), and the span joins
   the nearest speaker centroid with a margin-weighted score.
6. **Assignment** (`assign`, `transcript`): each word goes to the exclusive
   turn it overlaps most, else the nearest turn within 0.5 s. Word confidence
   combines overlap fraction, distance to the turn boundary, and the turn's
   embedding similarity to its speaker centroid. Segments split wherever the
   speaker changes.

`pipeline::run` wires the stages together; ASR and diarization can run
concurrently (`PipelineConfig::concurrent`). The `transcribe` example runs the
pipeline on a file and writes `transcript.json`, `speakers.json`, `turns.json`
and `report.json`, optionally comparing against a baseline transcript.

## Features

| Feature | Effect |
|---|---|
| (default) | ASR only. Metal on macOS, CPU elsewhere. Diarization returns `FeatureDisabled`. |
| `diarize` | speakrs diarization and gap filling. |
| `cuda` | `diarize` plus whisper.cpp CUDA and the ONNX Runtime CUDA provider, with ONNX Runtime loaded at run time (`ort/load-dynamic`). |

## Build notes

- **cmake** is required by `whisper-rs-sys` (Homebrew `cmake` on macOS).
- **BLAS for speakrs**: speakrs selects a BLAS backend per target. On Linux
  x86_64 its default links Intel MKL statically. On macOS arm64 its default
  builds OpenBLAS from source and needs the gfortran runtime at link time:

  ```sh
  LIBRARY_PATH=/opt/homebrew/opt/gcc/lib/gcc/current cargo build -p glassrip-audio --features diarize
  ```

  Declaring different speakrs BLAS features per target is rejected by Cargo
  (speakrs renames `ndarray-linalg` per backend), so the crate declares one.
- **ONNX Runtime** (`cuda` feature): speakrs requires `ort ^2.0.0-rc.12`, which
  Cargo unifies with the workspace's `2.0.0-rc.13`; no patch is needed. The
  `cuda` feature enables `load-dynamic`, so point `ORT_DYLIB_PATH` at
  `libonnxruntime.so` from the official ONNX Runtime 1.28 GPU build for CUDA 12
  (see `models.toml`). The prebuilt binaries that `ort` downloads for rc.13
  link the CUDA 13 runtime. Cargo unifies features, so enabling `cuda` here
  also turns on `load-dynamic` for every `ort` user in the same build.
- **CUDA runtime**: the ONNX Runtime 1.28 CUDA 12 provider needs a CUDA 12.8 or
  newer runtime (`cudaLibraryGetKernel`). With an older toolkit, extract the
  NVIDIA `cuda_cudart` 12.9 redistributable listed in `models.toml` and put its
  `lib/` first on `LD_LIBRARY_PATH`. whisper.cpp kernels built with an earlier
  12.x toolkit run on the newer 12.x runtime.
- speakrs registers the CUDA execution provider with `error_on_failure`, so a
  missing CUDA library is an error rather than a silent CPU fallback.

## Models

`models.toml` pins every model file to an upstream revision and SHA-256.
Binaries are never committed. Files live under `$GLASSRIP_MODELS_DIR` or
`~/.glassrip/models` at the listed relative paths (speakrs files under
`speakrs/`). `models::verify` checks a file against its hash; the example's
`--verify-models` flag checks every file a run uses.

## Running

```sh
export ORT_DYLIB_PATH=~/.glassrip/models/ort/onnxruntime-linux-x64-gpu_cuda12-1.28.2/lib/libonnxruntime.so
export LD_LIBRARY_PATH=~/.glassrip/models/ort/cuda_cudart-linux-x86_64-12.9.79-archive/lib:/usr/local/cuda/lib64
cargo run --release --features cuda --example transcribe -- meeting.flac --out out/ \
    --vocab "Kethra, Zorbin, blue server" --speakers 3 --concurrent
```

## Tests

`cargo test -p glassrip-audio` runs unit tests and synthetic-audio integration
tests (generated tones and silence; extraction tests need `ffmpeg` and
`ffprobe`). Set `GLASSRIP_TEST_WHISPER_MODEL` to a ggml model path to also run
the whisper smoke test. No test uses recorded speech.

## Licenses and attribution

This crate is MIT licensed. Model files are downloaded separately and keep
their own licenses:

| Model | License | Attribution |
|---|---|---|
| Whisper large-v3 and large-v3-turbo (ggml conversions from whisper.cpp) | MIT | OpenAI; ggml conversion by the whisper.cpp project |
| Silero VAD v6.2.0 (ggml conversion) | MIT | Silero Team; ggml conversion by ggml-org |
| pyannote segmentation-3.0 | MIT | pyannote.audio, Hervé Bredin and contributors |
| WeSpeaker ResNet34 (VoxCeleb) embedding and its split and batched variants | CC-BY-4.0 | WeSpeaker project; ONNX packaging by the speakrs project |
| PLDA parameters from pyannote speaker-diarization-community-1 | CC-BY-4.0 | pyannote.audio and pyannoteAI |
| ONNX Runtime 1.28.2 GPU (CUDA 12) | MIT | Microsoft |
| CUDA runtime 12.9 redistributable | NVIDIA CUDA EULA | NVIDIA |

CC-BY-4.0 requires attribution when redistributing outputs or models; keep
this table with any distribution that bundles those files.
