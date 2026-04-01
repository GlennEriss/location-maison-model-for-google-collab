from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from location_maison_model_annonce.core.config import load_config
from location_maison_model_annonce.observability.logging import configure_logging
from location_maison_model_annonce.training.evaluation import generate_predictions, load_model_for_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a prediction with the latest trained adapter")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--text", help="Annonce text to analyze")
    parser.add_argument("--input-file", help="Path to a text file containing the annonce")
    parser.add_argument("--batch-file", help="Path to a text file with one annonce per line")
    parser.add_argument("--interactive", action="store_true", help="Keep the model loaded and test multiple annonces")
    parser.add_argument("--max-new-tokens", type=int, default=180, help="Generation length limit")
    return parser


def resolve_description(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8").strip()

    payload = sys.stdin.read().strip()
    if payload:
        return payload
    raise ValueError("Provide --text, --input-file, or pipe an annonce through stdin.")


def build_prediction_payload(description: str, generated_text: str, predicted_json: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "description": description,
        "generated_text": generated_text,
        "predicted_json": predicted_json,
    }


def predict_one(
    description: str,
    tokenizer: Any,
    model: Any,
    runtime_cfg: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    prediction = generate_predictions(
        [
            {
                "example_id": "manual-predict",
                "instruction": "Transforme cette annonce immobiliere en JSON structure.",
                "description": description,
                "target_json": {},
            }
        ],
        tokenizer,
        model,
        runtime_cfg,
        max_new_tokens=max_new_tokens,
    )[0]
    return build_prediction_payload(description, prediction["generated_text"], prediction["predicted_json"])


def run_interactive_session(
    tokenizer: Any,
    model: Any,
    runtime_cfg: dict[str, Any],
    max_new_tokens: int,
) -> None:
    print("Mode interactif actif. Colle une annonce puis appuie sur Entree.")
    print("Tape 'exit' ou 'quit' pour quitter.")
    while True:
        try:
            description = input("\nannonce> ").strip()
        except EOFError:
            print()
            break

        if not description:
            continue
        if description.lower() in {"exit", "quit"}:
            break

        payload = predict_one(description, tokenizer, model, runtime_cfg, max_new_tokens)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_batch_file(
    batch_file: str,
    tokenizer: Any,
    model: Any,
    runtime_cfg: dict[str, Any],
    max_new_tokens: int,
) -> None:
    lines = [line.strip() for line in Path(batch_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, description in enumerate(lines, start=1):
        payload = predict_one(description, tokenizer, model, runtime_cfg, max_new_tokens)
        payload["index"] = index
        print(json.dumps(payload, ensure_ascii=False))


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

    logger = logging.getLogger("predict")
    logger.info("Loading latest model for prediction")

    tokenizer, model = load_model_for_evaluation(config.model_dump())
    runtime_cfg = config.model_dump()["runtime"]
    max_new_tokens = int(args.max_new_tokens)

    if args.interactive:
        run_interactive_session(tokenizer, model, runtime_cfg, max_new_tokens)
        return

    if args.batch_file:
        run_batch_file(args.batch_file, tokenizer, model, runtime_cfg, max_new_tokens)
        return

    description = resolve_description(args)
    payload = predict_one(description, tokenizer, model, runtime_cfg, max_new_tokens)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
