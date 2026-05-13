#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/Users/anaconda3/bin/python3}"
LOCAL_DB="${LOCAL_DB:-${TMPDIR:-/tmp}/seismicx_cont_mini_consensus.sqlite}"

"$PYTHON" "$ROOT/scripts/build_consensus_picks_json.py" \
  --auto-jsonl \
    "$ROOT/data/picks/phasenet.mini.phase.jsonl" \
    "$ROOT/data/picks/pnsn_v3_diff.mini.phase.jsonl" \
    "$ROOT/data/picks/skynet.mini.phase.jsonl" \
  --label-json "$ROOT/data/label/annotations_mini_two_hours.json" \
  --index-db "$LOCAL_DB" \
  --out-json "$ROOT/data/label/consensus_mini_picks.json" \
  --build-index \
  --drop-existing \
  --is-phase 1.5 \
  --min-models 3 \
  --human-match 1.5
