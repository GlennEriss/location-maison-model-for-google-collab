#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path> [interval_seconds]"
  exit 1
fi

CONFIG_PATH="$1"
INTERVAL="${2:-10}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while true; do
  clear || true
  echo "=== RUN STATUS ($(date '+%Y-%m-%d %H:%M:%S')) ==="
  (
    cd "$PROJECT_DIR"
    PYTHONPATH=src "$PYTHON_BIN" -m location_maison_model_annonce.cli.status --config "$CONFIG_PATH"
  )
  echo
  echo "Refresh in ${INTERVAL}s - Ctrl+C to stop"
  sleep "$INTERVAL"
done
