#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <drive_project_dir>"
  exit 1
fi

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVE_DIR="$1"
ARTIFACTS_ROOT="$DRIVE_DIR"

echo "Pull artifacts from Drive"
echo "  from: $ARTIFACTS_ROOT"
echo "  to:   $LOCAL_DIR"

mkdir -p "$LOCAL_DIR/outputs" "$LOCAL_DIR/data"

for rel in outputs/checkpoints outputs/metrics outputs/reports data/datasets data/smoke_datasets; do
  if [[ -d "$ARTIFACTS_ROOT/$rel" ]]; then
    mkdir -p "$LOCAL_DIR/$rel"
    rsync -av "$ARTIFACTS_ROOT/$rel/" "$LOCAL_DIR/$rel/"
  else
    echo "Skip missing: $ARTIFACTS_ROOT/$rel"
  fi
done

echo "Artifact pull complete."
