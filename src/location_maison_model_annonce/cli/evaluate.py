from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone

from location_maison_model_annonce.core.config import load_config
from location_maison_model_annonce.observability.logging import configure_logging
from location_maison_model_annonce.training.data import load_jsonl
from location_maison_model_annonce.training.evaluation import (
    compute_metrics,
    export_prediction_report,
    generate_predictions,
    load_model_for_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test"],
        help="Dataset split used for business evaluation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_state_path = Path(config.paths.checkpoint_dir).parent / "RUN_STATE.json"

    log_dir = Path(config.paths.log_dir)
    configure_logging(
        log_dir,
        config.logging["app_file"],
        config.logging["error_file"],
        config.logging["train_file"],
    )

    logger = logging.getLogger("evaluate")
    dataset_file = Path(config.paths.dataset_dir) / f"{args.split}.jsonl"
    examples: List[Dict[str, Any]] = load_jsonl(dataset_file)
    logger.info(
        "Loaded %s examples from %s",
        len(examples),
        dataset_file,
        extra={"split": args.split, "dataset_file": str(dataset_file), "example_count": len(examples)},
    )
    write_run_state(
        run_state_path,
        status="evaluate_starting",
        split=args.split,
        dataset_file=str(dataset_file),
        example_count=len(examples),
    )

    tokenizer, model = load_model_for_evaluation(config.model_dump())
    write_run_state(
        run_state_path,
        status="evaluate_running",
        split=args.split,
        dataset_file=str(dataset_file),
        example_count=len(examples),
    )
    logger.info("Starting business evaluation for split=%s", args.split, extra={"split": args.split})
    report_path = Path(config.paths.report_dir) / f"{args.split}_predictions.json"
    partial_report_path = report_path.with_suffix(".partial.jsonl")
    progress_state_path = report_path.with_suffix(".progress.json")
    try:
        predictions = generate_predictions(
            examples,
            tokenizer,
            model,
            config.model_dump()["runtime"],
            partial_output_path=partial_report_path,
            progress_state_path=progress_state_path,
            on_progress=lambda progress: write_run_state(
                run_state_path,
                status="evaluate_running",
                split=args.split,
                dataset_file=str(dataset_file),
                example_count=len(examples),
                report_file=str(report_path),
                partial_report_file=str(partial_report_path),
                progress_file=str(progress_state_path),
                **progress,
            ),
        )
        metrics = compute_metrics(predictions)
    except Exception as exc:
        write_run_state(
            run_state_path,
            status="evaluate_failed",
            split=args.split,
            dataset_file=str(dataset_file),
            example_count=len(examples),
            report_file=str(report_path),
            partial_report_file=str(partial_report_path),
            progress_file=str(progress_state_path),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    output_path = Path(config.logging["metrics_file"])
    metrics.dump(output_path)
    export_prediction_report(predictions, report_path)
    cleanup_resume_artifacts(partial_report_path, progress_state_path)
    logger.info("Metrics written to %s", output_path)
    logger.info("Prediction report written to %s", report_path)
    logger.info(
        "json_valid_rate=%.4f type_accuracy=%.4f status_accuracy=%.4f tags_f1=%.4f numeric_exact_match=%.4f",
        metrics.json_valid_rate,
        metrics.type_accuracy,
        metrics.status_accuracy,
        metrics.tags_f1,
        metrics.numeric_exact_match,
    )
    write_run_state(
        run_state_path,
        status="evaluate_completed",
        split=args.split,
        dataset_file=str(dataset_file),
        example_count=len(examples),
        metrics_file=str(output_path),
        report_file=str(report_path),
        partial_report_file=str(partial_report_path),
        progress_file=str(progress_state_path),
        json_valid_rate=metrics.json_valid_rate,
        type_accuracy=metrics.type_accuracy,
        status_accuracy=metrics.status_accuracy,
        tags_f1=metrics.tags_f1,
        numeric_exact_match=metrics.numeric_exact_match,
        evaluated_examples=metrics.evaluated_examples,
    )

def write_run_state(path: Path, status: str, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **payload,
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def cleanup_resume_artifacts(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
