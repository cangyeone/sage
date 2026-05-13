#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/Users/anaconda3/bin/python3}"
RAW_ROOT="${SEISMICX_RAW_ROOT:-/Volumes/Data/continous_dataset_tool/data/continous_usa/data}"

mkdir -p "$ROOT/data/hdf5"

"$PYTHON" "$ROOT/scripts/makeh5_flex_seg.py" \
  --input_dir "$RAW_ROOT/07/06" \
  --loc_file "$ROOT/data/label/stations.csv" \
  --output "$ROOT/data/hdf5/continuous_waveform_usa.h5" \
  --split_interval hour \
  --include_split_file_id 20190706_04 \
  --num_workers "${NUM_WORKERS:-8}" \
  --max_pending "${MAX_PENDING:-16}" \
  --compression gzip \
  --compression_opts 4

"$PYTHON" "$ROOT/scripts/makeh5_flex_seg.py" \
  --input_dir "$RAW_ROOT/11/14" \
  --loc_file "$ROOT/data/label/stations.csv" \
  --output "$ROOT/data/hdf5/continuous_waveform_usa.h5" \
  --split_interval hour \
  --include_split_file_id 20211114_16 \
  --num_workers "${NUM_WORKERS:-8}" \
  --max_pending "${MAX_PENDING:-16}" \
  --compression gzip \
  --compression_opts 4
