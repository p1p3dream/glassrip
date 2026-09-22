#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${GLASSRIP_MODEL_DIR:-$HOME/.glassrip/models}"
REPO="monkt/paddleocr-onnx"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$MODEL_DIR"

echo "Downloading PP-OCRv5 English ONNX models to $MODEL_DIR..."

for file in "detection/v5/det.onnx:det.onnx" \
            "languages/english/rec.onnx:rec.onnx" \
            "languages/english/dict.txt:dict.txt"; do
    src="${file%%:*}"
    dst="${file##*:}"
    if [ -f "$MODEL_DIR/$dst" ]; then
        echo "  $dst already exists, skipping"
    else
        echo "  Downloading $dst..."
        curl -L -o "$MODEL_DIR/$dst" "$BASE/$src"
    fi
done

echo "Done. Models in $MODEL_DIR"
ls -lh "$MODEL_DIR"
