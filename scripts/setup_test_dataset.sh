#!/usr/bin/env bash
# Builds a tiny synthetic "dataset" for testing this pipeline against a real
# Annotator server, with no real robot footage and no Gemini calls required.
#
# Usage: scripts/setup_test_dataset.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-/tmp/gemini-annotator-smoketest}"
VIDEO="$OUT_DIR/_source.mp4"
DATASET_DIR="$OUT_DIR/dataset"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Generating a 6s synthetic test video with ffmpeg ..."
ffmpeg -y -v error -f lavfi -i "testsrc=duration=6:size=320x240:rate=30" \
  -pix_fmt yuv420p "$VIDEO"

echo "Scaffolding it into a minimal dataset Annotator can open ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"
"$PYTHON" -m gemini_annotator.cli scaffold --video "$VIDEO" --out "$DATASET_DIR"

echo
echo "Test dataset ready at: $DATASET_DIR"
echo "Next: point a running Annotator server's --data-root at $OUT_DIR (or its parent),"
echo "then see the README's Testing section."
