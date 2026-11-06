#!/usr/bin/env bash
# bring a packed results folder home: unpack the base64 chunks, then regenerate everything that was left out.
# usage: bash tools_unpack.sh <chunkdir> <label>
set -euo pipefail
cd "$(dirname "$0")"
python -m drift.pack unpack "$1" results/
python -m drift.plots "$2"
python -m drift.flips "$2"
python -m drift.report "$2" --readme
python -m drift.export_web "$2"
ls results/"$2"
