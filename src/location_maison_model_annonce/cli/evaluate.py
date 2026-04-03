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
    load_partial_predictions,
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
    parser.add_argument("--start-index", type=int, default=0, help="0-based start index within the split")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of examples to process in this run; 0 means process all remaining examples",
    )
    parser.add_argument(
        "--auto-chunk",
        action="store_true",
        help="Automatically keep processing remaining examples chunk by chunk until completion",
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
    run_evaluation(args, config.model_dump(), run_state_path, logging.getLogger("evaluate"))


def run_evaluation(args: argparse.Namespace, config: Dict[str, Any], run_state_path: Path, logger: logging.Logger) -> None:
    dataset_file = Path(config["paths"]["dataset_dir"]) / f"{args.split}.jsonl"
    all_examples: List[Dict[str, Any]] = load_jsonl(dataset_file)
    report_path = Path(config["paths"]["report_dir"]) / f"{args.split}_predictions.json"
    partial_report_path = report_path.with_suffix(".partial.jsonl")
    progress_state_path = report_path.with_suffix(".progress.json")

    tokenizer = None
    model = None
    chunk_index = 0

    while True:
        cached_predictions = load_partial_predictions(partial_report_path)
        cached_ids = {item["example_id"] for item in cached_predictions}
        examples = select_examples_for_run(all_examples, cached_ids, start_index=args.start_index, limit=args.limit)

        logger.info(
            "Loaded %s examples from %s (selected %s for this run, cached=%s, chunk=%s)",
            len(all_examples),
            dataset_file,
            len(examples),
            len(cached_predictions),
            chunk_index + 1,
            extra={
                "split": args.split,
                "dataset_file": str(dataset_file),
                "example_count": len(all_examples),
                "selected_example_count": len(examples),
                "cached_prediction_count": len(cached_predictions),
                "start_index": args.start_index,
                "limit": args.limit,
                "auto_chunk": args.auto_chunk,
                "chunk_index": chunk_index + 1,
            },
        )
        write_run_state(
            run_state_path,
            status="evaluate_starting",
            split=args.split,
            dataset_file=str(dataset_file),
            example_count=len(all_examples),
            selected_example_count=len(examples),
            cached_prediction_count=len(cached_predictions),
            start_index=args.start_index,
            limit=args.limit,
            auto_chunk=args.auto_chunk,
            chunk_index=chunk_index + 1,
        )

        if not examples:
            finalize_cached_only_run(
                run_state_path=run_state_path,
                config=config,
                args=args,
                all_examples=all_examples,
                cached_predictions=cached_predictions,
                dataset_file=dataset_file,
                report_path=report_path,
                partial_report_path=partial_report_path,
                progress_state_path=progress_state_path,
                logger=logger,
                chunk_index=chunk_index + 1,
            )
            return

        if tokenizer is None or model is None:
            tokenizer, model = load_model_for_evaluation(config)

        write_run_state(
            run_state_path,
            status="evaluate_running",
            split=args.split,
            dataset_file=str(dataset_file),
            example_count=len(all_examples),
            selected_example_count=len(examples),
            cached_prediction_count=len(cached_predictions),
            start_index=args.start_index,
            limit=args.limit,
            auto_chunk=args.auto_chunk,
            chunk_index=chunk_index + 1,
        )
        logger.info(
            "Starting business evaluation for split=%s chunk=%s auto_chunk=%s",
            args.split,
            chunk_index + 1,
            args.auto_chunk,
            extra={"split": args.split, "chunk_index": chunk_index + 1, "auto_chunk": args.auto_chunk},
        )
        try:
            generate_predictions(
                examples,
                tokenizer,
                model,
                config["runtime"],
                partial_output_path=partial_report_path,
                progress_state_path=progress_state_path,
                on_progress=lambda progress: write_run_state(
                    run_state_path,
                    status="evaluate_running",
                    split=args.split,
                    dataset_file=str(dataset_file),
                    example_count=len(all_examples),
                    selected_example_count=len(examples),
                    cached_prediction_count=len(cached_predictions),
                    report_file=str(report_path),
                    partial_report_file=str(partial_report_path),
                    progress_file=str(progress_state_path),
                    start_index=args.start_index,
                    limit=args.limit,
                    auto_chunk=args.auto_chunk,
                    chunk_index=chunk_index + 1,
                    **progress,
                ),
            )
            merged_predictions = merge_predictions_for_examples(all_examples, load_partial_predictions(partial_report_path))
            metrics = compute_metrics(merged_predictions)
        except Exception as exc:
            write_run_state(
                run_state_path,
                status="evaluate_failed",
                split=args.split,
                dataset_file=str(dataset_file),
                example_count=len(all_examples),
                selected_example_count=len(examples),
                cached_prediction_count=len(cached_predictions),
                report_file=str(report_path),
                partial_report_file=str(partial_report_path),
                progress_file=str(progress_state_path),
                start_index=args.start_index,
                limit=args.limit,
                auto_chunk=args.auto_chunk,
                chunk_index=chunk_index + 1,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        output_path = Path(config["logging"]["metrics_file"])
        metrics.dump(output_path)
        export_prediction_report(merged_predictions, report_path)
        completed = len(merged_predictions) >= len(all_examples)
        if completed:
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
            status="evaluate_completed" if completed else "evaluate_partial_completed",
            split=args.split,
            dataset_file=str(dataset_file),
            example_count=len(all_examples),
            selected_example_count=len(examples),
            cached_prediction_count=len(cached_predictions),
            metrics_file=str(output_path),
            report_file=str(report_path),
            partial_report_file=str(partial_report_path),
            progress_file=str(progress_state_path),
            start_index=args.start_index,
            limit=args.limit,
            auto_chunk=args.auto_chunk,
            chunk_index=chunk_index + 1,
            json_valid_rate=metrics.json_valid_rate,
            type_accuracy=metrics.type_accuracy,
            status_accuracy=metrics.status_accuracy,
            tags_f1=metrics.tags_f1,
            numeric_exact_match=metrics.numeric_exact_match,
            evaluated_examples=metrics.evaluated_examples,
        )

        if completed or not args.auto_chunk:
            return

        chunk_index += 1
        logger.info(
            "Auto-chunk enabled: continuing to next chunk after chunk=%s",
            chunk_index,
            extra={"chunk_index": chunk_index, "auto_chunk": True},
        )


def select_examples_for_run(
    examples: List[Dict[str, Any]],
    cached_ids: set[str],
    *,
    start_index: int,
    limit: int,
) -> List[Dict[str, Any]]:
    start = max(start_index, 0)
    remaining = [example for example in examples[start:] if example["example_id"] not in cached_ids]
    if limit > 0:
        return remaining[:limit]
    return remaining


def merge_predictions_for_examples(
    examples: List[Dict[str, Any]], cached_predictions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    cached_by_id = {entry["example_id"]: entry for entry in cached_predictions}
    return [cached_by_id[item["example_id"]] for item in examples if item["example_id"] in cached_by_id]


def finalize_cached_only_run(
    *,
    run_state_path: Path,
    config: Dict[str, Any],
    args: argparse.Namespace,
    all_examples: List[Dict[str, Any]],
    cached_predictions: List[Dict[str, Any]],
    dataset_file: Path,
    report_path: Path,
    partial_report_path: Path,
    progress_state_path: Path,
    logger: logging.Logger,
    chunk_index: int,
) -> None:
    logger.info("No examples selected for this run; exporting current cached report only.")
    merged_predictions = merge_predictions_for_examples(all_examples, cached_predictions)
    export_prediction_report(merged_predictions, report_path)
    metrics = compute_metrics(merged_predictions)
    output_path = Path(config["logging"]["metrics_file"])
    metrics.dump(output_path)
    completed = len(merged_predictions) >= len(all_examples)
    if completed:
        cleanup_resume_artifacts(partial_report_path, progress_state_path)
    write_run_state(
        run_state_path,
        status="evaluate_completed" if completed else "evaluate_partial_completed",
        split=args.split,
        dataset_file=str(dataset_file),
        example_count=len(all_examples),
        selected_example_count=0,
        cached_prediction_count=len(cached_predictions),
        metrics_file=str(output_path),
        report_file=str(report_path),
        partial_report_file=str(partial_report_path),
        progress_file=str(progress_state_path),
        start_index=args.start_index,
        limit=args.limit,
        auto_chunk=args.auto_chunk,
        chunk_index=chunk_index,
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
