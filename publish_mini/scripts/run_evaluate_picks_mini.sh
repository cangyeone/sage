#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/Users/anaconda3/bin/python3}"
PICK_JSONL="${1:-$ROOT/data/picks/phasenet.mini.phase.jsonl}"

mkdir -p "$ROOT/data/picks" "$ROOT/eval_picks"

"$PYTHON" "$ROOT/scripts/evaluate_picks.py" \
  --auto-jsonl "$PICK_JSONL" \
  --label-json "$ROOT/data/label/annotations_mini_two_hours.json" \
  --index-db "${LOCAL_PICK_DB:-${TMPDIR:-/tmp}/seismicx_cont_mini_picks.sqlite}" \
  --outdir "$ROOT/eval_picks/example" \
  --waveform-db "$ROOT/data/index/waveform_index.sqlite" \
  --tp-tol 1.5 \
  --err-window 5.0 \
  --build-index \
  --drop-existing \
  --plot
