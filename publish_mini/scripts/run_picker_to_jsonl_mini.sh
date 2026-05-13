#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/Users/anaconda3/bin/python3}"
MODEL="${1:-$ROOT/pickers/phasenet.jit}"
OUTPUT="${2:-$ROOT/data/picks/phasenet.mini.phase.jsonl}"

mkdir -p "$ROOT/data/picks"

"$PYTHON" "$ROOT/scripts/run_picker_to_jsonl.py" \
  --h5_input "$ROOT/data/hdf5/continuous_waveform_usa_*.h5" \
  --picker_model "$MODEL" \
  --output_jsonl "$OUTPUT" \
  --canonical_input_length 360000 \
  --max_picks_per_sample 0 \
  --reload_model_interval 2000
