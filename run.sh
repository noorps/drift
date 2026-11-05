#!/usr/bin/env bash
# one command: sweep every knob, dump activations, analyse, draw, summarise.
# usage: bash run.sh <label>      results land in results/<label>/
set -euo pipefail
cd "$(dirname "$0")"
LABEL="${1:-$(date +%Y%m%d-%H%M)}"
PY="${PYTHON:-python3}"
MODELS="${MODELS:-resnet,text}"
echo "== results/$LABEL"
$PY -m drift.run --label "$LABEL" --models "$MODELS" --depth-n "${DEPTH_N:-64}"
$PY -m drift.plots "$LABEL"
$PY -m drift.flips "$LABEL"
$PY -m drift.report "$LABEL"
echo "== done: results/$LABEL"
ls "results/$LABEL"
