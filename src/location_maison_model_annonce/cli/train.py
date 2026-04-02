from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from location_maison_model_annonce.core.config import load_config
from location_maison_model_annonce.data.bootstrap_generator import generate_bootstrap_dataset
from location_maison_model_annonce.observability.logging import configure_logging
from location_maison_model_annonce.training.evaluation import compute_metrics, export_prediction_report, generate_predictions
from location_maison_model_annonce.training.evaluation import find_latest_checkpoint
from location_maison_model_annonce.training.data import load_training_dataset
from location_maison_model_annonce.training.runtime import resolve_runtime
from location_maison_model_annonce.training.trainer import build_trainer, export_training_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train location-maison-model-annonce")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_state_path = Path(config.paths.checkpoint_dir).parent / "RUN_STATE.json"

    log_dir = Path(config.paths.log_dir)
    app_file = config.logging["app_file"]
    error_file = config.logging["error_file"]
    train_file = config.logging["train_file"]
    configure_logging(log_dir, app_file, error_file, train_file)

    logger = logging.getLogger("train")
    logger.info("Training bootstrap initialized")
    logger.info("Base model: %s", config.model["base_model"])
    logger.info("Strategy: %s", config.training["strategy"])
    logger.info("Dataset target size: %s", config.dataset["initial_target_size"])
    logger.info(
        "Training config epochs=%s batch_size=%s grad_accum=%s max_seq_length=%s checkpoint_dir=%s",
        config.training["epochs"],
        config.training["batch_size"],
        config.training["gradient_accumulation_steps"],
        config.training["max_seq_length"],
        config.paths.checkpoint_dir,
    )

    dataset_dir = Path(config.paths.dataset_dir)
    train_file_path = dataset_dir / "train.jsonl"
    if not train_file_path.exists():
        logger.info("No dataset found in %s. Generating bootstrap dataset first.", dataset_dir)
        generate_bootstrap_dataset(config.model_dump(), dataset_dir)

    dataset = load_training_dataset(dataset_dir)
    logger.info(
        "Loaded dataset sizes train=%s validation=%s test=%s",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )

    tokenizer, model = resolve_runtime(config.model_dump()["model"], config.model_dump()["runtime"])
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    adapter_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    checkpoint_mode = "fresh"
    checkpoint_meta = read_checkpoint_state(latest_checkpoint)

    if latest_checkpoint and checkpoint_meta.get("is_resumable"):
        if checkpoint_meta.get("is_completed"):
            checkpoint_mode = "continue_from_completed_adapter"
            adapter_checkpoint = str(latest_checkpoint)
            logger.info(
                "Found completed checkpoint %s (step=%s/%s); starting a new cycle from this adapter.",
                latest_checkpoint,
                checkpoint_meta.get("global_step"),
                checkpoint_meta.get("max_steps"),
            )
        else:
            checkpoint_mode = "resume_interrupted_run"
            resume_checkpoint = str(latest_checkpoint)
            adapter_checkpoint = str(latest_checkpoint)
            logger.info(
                "Found interrupted checkpoint %s (step=%s/%s); resuming trainer state from this checkpoint.",
                latest_checkpoint,
                checkpoint_meta.get("global_step"),
                checkpoint_meta.get("max_steps"),
            )
    elif latest_checkpoint:
        checkpoint_mode = "continue_from_adapter_only"
        adapter_checkpoint = str(latest_checkpoint)
        logger.info("Found adapter checkpoint without trainer state: %s", latest_checkpoint)
    else:
        logger.info("No adapter checkpoint found, starting from base model + fresh LoRA adapter")

    logger.info(
        "Checkpoint decision mode=%s adapter_checkpoint=%s resume_checkpoint=%s",
        checkpoint_mode,
        adapter_checkpoint,
        resume_checkpoint,
    )

    write_run_state(
        run_state_path,
        status="starting",
        checkpoint_mode=checkpoint_mode,
        latest_checkpoint=str(latest_checkpoint) if latest_checkpoint else None,
        adapter_checkpoint=adapter_checkpoint,
        resume_checkpoint=resume_checkpoint,
        dataset_dir=str(dataset_dir),
        train_examples=len(dataset["train"]),
        validation_examples=len(dataset["validation"]),
        test_examples=len(dataset["test"]),
        epochs=config.training["epochs"],
        batch_size=config.training["batch_size"],
        gradient_accumulation_steps=config.training["gradient_accumulation_steps"],
        max_seq_length=config.training["max_seq_length"],
    )

    trainer = build_trainer(
        config.model_dump(),
        dataset,
        tokenizer,
        model,
        adapter_checkpoint=adapter_checkpoint,
    )

    logger.info("Starting training loop")
    write_run_state(
        run_state_path,
        status="training",
        checkpoint_mode=checkpoint_mode,
        latest_checkpoint=str(latest_checkpoint) if latest_checkpoint else None,
        adapter_checkpoint=adapter_checkpoint,
        resume_checkpoint=resume_checkpoint,
    )

    try:
        train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as exc:
        last_checkpoint_after_failure = find_latest_checkpoint(checkpoint_dir)
        write_run_state(
            run_state_path,
            status="failed",
            checkpoint_mode=checkpoint_mode,
            latest_checkpoint=str(last_checkpoint_after_failure) if last_checkpoint_after_failure else None,
            adapter_checkpoint=adapter_checkpoint,
            resume_checkpoint=resume_checkpoint,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    logger.info("Training completed")

    logger.info("Running validation evaluation")
    write_run_state(
        run_state_path,
        status="evaluating_validation",
        checkpoint_mode=checkpoint_mode,
        latest_checkpoint=str(find_latest_checkpoint(checkpoint_dir)) if find_latest_checkpoint(checkpoint_dir) else None,
        adapter_checkpoint=adapter_checkpoint,
        resume_checkpoint=resume_checkpoint,
        train_loss=train_output.metrics.get("train_loss"),
    )
    eval_output = trainer.evaluate()
    metrics = export_training_metrics(Path(config.logging["metrics_file"]), train_output.metrics, eval_output)

    logger.info("Running business evaluation on test split")
    write_run_state(
        run_state_path,
        status="evaluating_test",
        checkpoint_mode=checkpoint_mode,
        latest_checkpoint=str(find_latest_checkpoint(checkpoint_dir)) if find_latest_checkpoint(checkpoint_dir) else None,
        adapter_checkpoint=adapter_checkpoint,
        resume_checkpoint=resume_checkpoint,
        train_loss=metrics.train_loss,
        eval_loss=metrics.eval_loss,
        perplexity=metrics.perplexity,
    )
    predictions = generate_predictions(
        list(dataset["test"]),
        tokenizer,
        trainer.model,
        config.model_dump()["runtime"],
    )
    business_metrics = compute_metrics(predictions)
    business_metrics.train_loss = metrics.train_loss
    business_metrics.eval_loss = metrics.eval_loss
    business_metrics.perplexity = metrics.perplexity
    business_metrics.dump(Path(config.logging["metrics_file"]))

    report_path = Path(config.paths.report_dir) / "test_predictions.json"
    export_prediction_report(predictions, report_path)

    logger.info("Metrics exported to %s", config.logging["metrics_file"])
    logger.info("Prediction report exported to %s", report_path)
    logger.info(
        "Train loss=%s Eval loss=%s Perplexity=%s TypeAcc=%s StatusAcc=%s TagsF1=%s NumericExact=%s",
        business_metrics.train_loss,
        business_metrics.eval_loss,
        business_metrics.perplexity,
        business_metrics.type_accuracy,
        business_metrics.status_accuracy,
        business_metrics.tags_f1,
        business_metrics.numeric_exact_match,
    )
    write_run_state(
        run_state_path,
        status="completed",
        checkpoint_mode=checkpoint_mode,
        latest_checkpoint=str(find_latest_checkpoint(checkpoint_dir)) if find_latest_checkpoint(checkpoint_dir) else None,
        adapter_checkpoint=adapter_checkpoint,
        resume_checkpoint=resume_checkpoint,
        metrics_file=config.logging["metrics_file"],
        report_file=str(report_path),
        train_loss=business_metrics.train_loss,
        eval_loss=business_metrics.eval_loss,
        perplexity=business_metrics.perplexity,
        json_valid_rate=business_metrics.json_valid_rate,
        type_accuracy=business_metrics.type_accuracy,
        status_accuracy=business_metrics.status_accuracy,
        tags_f1=business_metrics.tags_f1,
        numeric_exact_match=business_metrics.numeric_exact_match,
        evaluated_examples=business_metrics.evaluated_examples,
    )

def read_checkpoint_state(checkpoint_path: Path | None) -> dict[str, object]:
    if checkpoint_path is None:
        return {"is_resumable": False, "is_completed": False, "global_step": None, "max_steps": None}

    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return {"is_resumable": False, "is_completed": False, "global_step": None, "max_steps": None}

    try:
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"is_resumable": False, "is_completed": False, "global_step": None, "max_steps": None}

    global_step = payload.get("global_step")
    max_steps = payload.get("max_steps")
    is_completed = (
        isinstance(global_step, int)
        and isinstance(max_steps, int)
        and max_steps > 0
        and global_step >= max_steps
    )
    return {
        "is_resumable": True,
        "is_completed": is_completed,
        "global_step": global_step,
        "max_steps": max_steps,
    }


def write_run_state(path: Path, status: str, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **payload,
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
