from __future__ import annotations

import argparse
import json
from pathlib import Path

from location_maison_model_annonce.core.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the latest run/evaluation state")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--json", action="store_true", help="Print raw RUN_STATE.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_state_path = Path(config.paths.checkpoint_dir).parent / "RUN_STATE.json"

    if not run_state_path.exists():
        print(f"No run state found at {run_state_path}")
        return

    payload = json.loads(run_state_path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Status          : {payload.get('status')}")
    print(f"Updated at      : {payload.get('updated_at')}")

    if payload.get("checkpoint_mode") is not None:
        print(f"Checkpoint mode : {payload.get('checkpoint_mode')}")
    if payload.get("latest_checkpoint"):
        print(f"Latest checkpoint: {payload.get('latest_checkpoint')}")
    if payload.get("resume_checkpoint"):
        print(f"Resume checkpoint: {payload.get('resume_checkpoint')}")
    if payload.get("adapter_checkpoint"):
        print(f"Adapter checkpoint: {payload.get('adapter_checkpoint')}")

    if payload.get("dataset_dir"):
        print(f"Dataset dir     : {payload.get('dataset_dir')}")
    if payload.get("dataset_file"):
        print(f"Dataset file    : {payload.get('dataset_file')}")
    if payload.get("split"):
        print(f"Split           : {payload.get('split')}")

    for key, label in (
        ("train_examples", "Train examples"),
        ("validation_examples", "Validation examples"),
        ("test_examples", "Test examples"),
        ("example_count", "Example count"),
        ("epochs", "Epochs"),
        ("batch_size", "Batch size"),
        ("gradient_accumulation_steps", "Grad accum"),
        ("max_seq_length", "Max seq length"),
        ("evaluated_examples", "Evaluated examples"),
    ):
        if payload.get(key) is not None:
            print(f"{label:<16}: {payload.get(key)}")

    for key, label in (
        ("train_loss", "Train loss"),
        ("eval_loss", "Eval loss"),
        ("perplexity", "Perplexity"),
        ("json_valid_rate", "JSON valid rate"),
        ("type_accuracy", "Type accuracy"),
        ("status_accuracy", "Status accuracy"),
        ("tags_f1", "Tags F1"),
        ("numeric_exact_match", "Numeric exact"),
    ):
        if payload.get(key) is not None:
            print(f"{label:<16}: {payload.get(key)}")

    if payload.get("metrics_file"):
        print(f"Metrics file    : {payload.get('metrics_file')}")
    if payload.get("report_file"):
        print(f"Report file     : {payload.get('report_file')}")

    if payload.get("error_type") or payload.get("error_message"):
        print(f"Error type      : {payload.get('error_type')}")
        print(f"Error message   : {payload.get('error_message')}")


if __name__ == "__main__":
    main()
