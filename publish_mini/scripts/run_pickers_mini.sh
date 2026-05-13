#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/Users/anaconda3/bin/python3}"

mkdir -p "$ROOT/data/picks"

"$PYTHON" "$ROOT/scripts/run_picker_to_jsonl.py" \
  --h5_input "$ROOT/data/hdf5/continuous_waveform_usa_*.h5" \
  --picker_model "$ROOT/pickers/phasenet.jit" \
  --output_jsonl "$ROOT/data/picks/phasenet.mini.phase.jsonl" \
  --device "${DEVICE:-cpu}" \
  --canonical_input_length 360000 \
  --max_picks_per_sample 0 \
  --no_auto_restart

"$PYTHON" "$ROOT/scripts/run_picker_to_jsonl.py" \
  --h5_input "$ROOT/data/hdf5/continuous_waveform_usa_*.h5" \
  --picker_model "$ROOT/pickers/pnsn.v3.diff.jit" \
  --output_jsonl "$ROOT/data/picks/pnsn_v3_diff.mini.phase.jsonl" \
  --device "${DEVICE:-cpu}" \
  --canonical_input_length 360000 \
  --max_picks_per_sample 0 \
  --no_auto_restart

"$PYTHON" "$ROOT/scripts/run_picker_to_jsonl.py" \
  --h5_input "$ROOT/data/hdf5/continuous_waveform_usa_*.h5" \
  --picker_model "$ROOT/pickers/skynet.jit" \
  --output_jsonl "$ROOT/data/picks/skynet.mini.phase.jsonl" \
  --device "${DEVICE:-cpu}" \
  --canonical_input_length 360000 \
  --max_picks_per_sample 0 \
  --no_auto_restart
