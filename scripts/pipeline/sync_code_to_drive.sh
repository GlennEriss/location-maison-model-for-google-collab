#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <drive_project_dir>"
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_DIR="$1"
CODE_DIR="$DEST_DIR/code"

mkdir -p "$CODE_DIR"
mkdir -p "$CODE_DIR/data"

echo "Sync code to Drive"
echo "  from: $SRC_DIR"
echo "  to:   $CODE_DIR"

rsync -av \
  --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pycache_local/' \
  --exclude '__pycache__/' \
  --exclude 'logs/' \
  --exclude 'outputs/' \
  --exclude 'data/datasets/' \
  --exclude 'data/smoke_datasets/' \
  --exclude 'data/processed/' \
  --exclude '.cache/' \
  "$SRC_DIR/README.md" \
  "$SRC_DIR/requirements.txt" \
  "$SRC_DIR/.gitignore" \
  "$SRC_DIR/config" \
  "$SRC_DIR/notebooks" \
  "$SRC_DIR/src" \
  "$SRC_DIR/scripts" \
  "$CODE_DIR/"

rsync -av \
  --delete \
  --exclude '.DS_Store' \
  "$SRC_DIR/data/source_annonces" \
  "$SRC_DIR/data/source_properties" \
  "$SRC_DIR/data/post-for-facebook" \
  "$CODE_DIR/data/"

echo "Code sync complete."
