#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <drive_project_dir>"
  exit 1
fi

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVE_DIR="$1"
ARTIFACTS_ROOT="$DRIVE_DIR"

echo "Push artifacts to Drive"
echo "  from: $LOCAL_DIR"
echo "  to:   $ARTIFACTS_ROOT"

mkdir -p "$ARTIFACTS_ROOT/outputs" "$ARTIFACTS_ROOT/data"

for rel in outputs/checkpoints outputs/metrics outputs/reports data/datasets data/smoke_datasets; do
  if [[ -d "$LOCAL_DIR/$rel" ]]; then
    mkdir -p "$ARTIFACTS_ROOT/$rel"
    rsync -av "$LOCAL_DIR/$rel/" "$ARTIFACTS_ROOT/$rel/"
  else
    echo "Skip missing: $LOCAL_DIR/$rel"
  fi
done

echo "Artifact push complete."
