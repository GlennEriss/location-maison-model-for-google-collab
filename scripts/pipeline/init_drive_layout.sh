#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <drive_project_dir>"
  exit 1
fi

DRIVE_DIR="$1"

echo "Initialize Drive layout"
echo "  root: $DRIVE_DIR"

mkdir -p \
  "$DRIVE_DIR/code" \
  "$DRIVE_DIR/data/source_annonces" \
  "$DRIVE_DIR/data/source_properties" \
  "$DRIVE_DIR/data/post-for-facebook" \
  "$DRIVE_DIR/data/datasets" \
  "$DRIVE_DIR/data/smoke_datasets" \
  "$DRIVE_DIR/data/processed/relevance" \
  "$DRIVE_DIR/outputs/checkpoints" \
  "$DRIVE_DIR/outputs/metrics" \
  "$DRIVE_DIR/outputs/reports" \
  "$DRIVE_DIR/logs/app" \
  "$DRIVE_DIR/logs/train" \
  "$DRIVE_DIR/logs/error"

cat > "$DRIVE_DIR/README_LAYOUT.txt" <<'EOF'
location-maison-model-for-google-collab - Drive layout

Recommended usage:
- code/                  -> synced project code
- data/source_annonces/  -> archived normalized ads
- data/source_properties/-> Firestore properties exports
- data/post-for-facebook/-> raw Facebook posts
- data/datasets/         -> generated training datasets
- data/smoke_datasets/   -> tiny local/validation datasets
- data/processed/        -> filtered/intermediate files
- outputs/checkpoints/   -> LoRA checkpoints from training
- outputs/metrics/       -> metrics JSON files
- outputs/reports/       -> prediction/evaluation reports
- logs/                  -> run logs
EOF

echo "Drive layout initialized."
