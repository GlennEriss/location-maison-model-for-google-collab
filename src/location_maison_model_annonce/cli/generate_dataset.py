from __future__ import annotations

import argparse
import logging
from pathlib import Path

from location_maison_model_annonce.core.config import load_config
from location_maison_model_annonce.data.bootstrap_generator import generate_bootstrap_dataset
from location_maison_model_annonce.observability.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate bootstrap training dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    log_dir = Path(config.paths.log_dir)
    configure_logging(
        log_dir,
        config.logging["app_file"],
        config.logging["error_file"],
        config.logging["train_file"],
    )

    logger = logging.getLogger("dataset")
    dataset_dir = Path(config.paths.dataset_dir)
    manifest = generate_bootstrap_dataset(config.model_dump(), dataset_dir)
    logger.info("Bootstrap dataset generated with %s examples", manifest.generated_examples)
    logger.info("Split counts: %s", manifest.split_counts)
    logger.info("Difficulty counts: %s", manifest.difficulty_counts)
    logger.info("Output files: %s", manifest.output_files)


if __name__ == "__main__":
    main()
