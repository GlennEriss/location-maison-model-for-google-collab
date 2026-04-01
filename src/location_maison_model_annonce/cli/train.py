from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
    latest_checkpoint = find_latest_checkpoint(Path(config.paths.checkpoint_dir))
    if latest_checkpoint:
        logger.info("Continuing training from adapter checkpoint %s", latest_checkpoint)
    else:
        logger.info("No adapter checkpoint found, starting from base model + fresh LoRA adapter")

    trainer = build_trainer(
        config.model_dump(),
        dataset,
        tokenizer,
        model,
        adapter_checkpoint=str(latest_checkpoint) if latest_checkpoint else None,
    )

    logger.info("Starting training loop")
    train_output = trainer.train()
    logger.info("Training completed")

    logger.info("Running validation evaluation")
    eval_output = trainer.evaluate()
    metrics = export_training_metrics(Path(config.logging["metrics_file"]), train_output.metrics, eval_output)

    logger.info("Running business evaluation on test split")
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


if __name__ == "__main__":
    main()
