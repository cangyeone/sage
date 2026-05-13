#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

(
  cd "$ROOT"
  find . -type f \
    ! -name 'manifest_sha256.txt' \
    ! -name '._*' \
    ! -path './*/._*' \
    ! -path './*/__pycache__/*' \
    ! -path './.DS_Store' \
    ! -path './*/.DS_Store' \
    | LC_ALL=C sort \
    | while read -r path; do
        shasum -a 256 "$path"
      done
) > "$ROOT/manifest_sha256.txt"

echo "[OK] wrote $ROOT/manifest_sha256.txt"
