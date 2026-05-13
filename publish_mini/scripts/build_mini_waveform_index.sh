#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/Users/anaconda3/bin/python3}"
LOCAL_DB="${LOCAL_DB:-${TMPDIR:-/tmp}/seismicx_cont_mini_waveform_index.sqlite}"
TARGET_DB="$ROOT/data/index/waveform_index.sqlite"

mkdir -p "$ROOT/data/index"
rm -f "$LOCAL_DB" "$LOCAL_DB-shm" "$LOCAL_DB-wal"

(
  cd "$ROOT"
  "$PYTHON" utils/hdf5_waveform_index.py build \
    --h5 "data/hdf5/continuous_waveform_usa_*.h5" \
    --db "$LOCAL_DB"
)

cp "$LOCAL_DB" "$TARGET_DB"
rm -f "$LOCAL_DB" "$LOCAL_DB-shm" "$LOCAL_DB-wal"

echo "[OK] copied local SQLite index to $TARGET_DB"
