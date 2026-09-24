# glassrip-media

Frame features, ECC alignment, the ink metric and keyframe segmentation for glassrip meeting
mode. `prototype_compat` mode reproduces the Python/OpenCV prototype (opencv-python-headless
5.0.0.93 on arm64) bit for bit; it exists for the parity gate.

## What is reproduced exactly

| Stage | Reference behavior ported |
|---|---|
| JPEG decode | libjpeg-turbo ISLOW IDCT, fancy chroma upsampling, table-driven YCbCr to BGR (baseline files; others fall back to `zune-jpeg`) |
| Sharpness | 3x3 Laplacian, REFLECT_101, numpy pairwise variance |
| Pair-score image | 6x6 area mean (f32 scale, round half to even), OpenCV fixed-point 5x5 Gaussian |
| ECC | `findTransformECC` MOTION_AFFINE step by step, including OpenCV's f32/f64 accumulation order |
| Warps | OpenCV 5 float-coordinate bilinear and nearest kernels, zero border |
| SSIM | f32 7x7 Gaussian in OpenCV's separable order, numpy's buffered mean order |
| Ink | 3x3 area mean, BGR to gray, binomial blur, adaptive mean threshold, HSV S/V, 5x5 dilation |
| Segmentation | `segment_runs.py` persistence rule and `build_keyframes.py` singleton merge |

Two behaviors differ from a literal reading of the OpenCV sources or the spec, because the
reference run did something else:

- **ECC failure keeps the partial warp.** The Python bindings update the caller's warp array
  in place, so when OpenCV raises (NaN correlation or `lambda_d <= 0`) the prototype used the
  matrix as it stood, not the identity it passed in.
- **BGR to gray uses Carotene's coefficients.** The arm64 wheels dispatch `cvtColor` to the
  Carotene HAL: `(9798 R + 19235 G + 3735 B + 2^14) >> 15`, not OpenCV's 14-bit formula.

## Fused multiply-add

Bit-exactness requires fused multiply-add semantics wherever OpenCV's NEON code uses
`v_fma`/`v_muladd` (warps, float filters, dot products, the LU solve). The code expresses these
with `f32::mul_add`, which Rust defines as a single rounding on every target, so results do not
depend on the CPU. On x86_64 builds without the `fma` target feature, `mul_add` runs through a
software routine: results are unchanged but the parity run is roughly 2x slower. Set
`RUSTFLAGS="-C target-cpu=x86-64-v3"` (or `native`) in the build environment; the parity binary
prints a warning when built without it. Target flags are deliberately not set in this
repository.

## Parity gate

```text
cargo run -p glassrip-media --release --bin parity -- \
    --frames DIR --reference DIR --keyframes FILE [--checksums FILE] [--report FILE]
```

Reference data is private and never stored here. `--checksums` (default
`REFERENCE/parity_checksums.sha256`) is a `sha256sum`-style file pinning `pair_scores.json`,
`ink_pairs.json`, `keyframes.json` and `frames` (all frames concatenated in filename order).
The gated criteria are listed in the module documentation of `src/bin/parity.rs`.

## Debugging aid

`cargo run -p glassrip-media --release --example dump_stages -- A.jpg B.jpg OUT_DIR` writes
every intermediate buffer for a frame pair as raw little-endian files so each primitive can be
diffed against a reference implementation.
