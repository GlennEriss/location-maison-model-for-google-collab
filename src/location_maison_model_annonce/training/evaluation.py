from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from peft import PeftModel

from location_maison_model_annonce.training.data import build_prompt
from location_maison_model_annonce.training.metrics import TrainingMetrics
from location_maison_model_annonce.training.runtime import resolve_runtime

ALLOWED_TAGS = {
    "Travail",
    "Famille",
    "Couple",
    "Villa",
    "Sous barrière",
    "Meublé",
    "Centre-ville",
    "Vacances",
    "Nature",
    "Montagne",
    "Piscine",
    "Animaux admis",
    "Commerces proches",
    "Transport proche",
    "Parking",
    "Wi-Fi",
    "Sécurisé",
    "Vélo",
    "Activités sportives",
    "Adapté aux enfants",
    "Accessible handicapés",
    "Étudiant",
    "Calme et tranquillité",
    "Proche de la plage",
    "Duplex",
    "Boutique",
    "Balcon",
    "Terrasse",
    "Collocation",
    "Garage",
    "Court séjour",
    "Propriétaire",
    "Agence",
}
TYPE_ALIASES = {
    "Appartement": "Apartment",
    "appartement": "Apartment",
    "Maison": "Home",
    "maison": "Home",
    "villa": "Villa",
    "VILLA": "Villa",
    "studio": "Studio",
    "STUDIO": "Studio",
    "room": "Room",
    "ROOM": "Room",
    "boutique": "Shop",
    "BOUTIQUE": "Shop",
    "bureau": "Desk",
    "BUREAU": "Desk",
    "immeuble": "Building",
    "IMMEUBLE": "Building",
    "terrain": "Land",
    "Terrain": "Land",
    "kiosque": "Kiosk",
    "Kiosque": "Kiosk",
}

NUMERIC_FIELDS = [
    "price",
    "area",
    "nbrRooms",
    "nbrKitchens",
    "nbrBathrooms",
    "nbrToilets",
    "nbrFloors",
    "nbrGarages",
    "nbrPiscine",
    "nbrApartments",
    "nbrFloorApartment",
    "nbrFloorStudio",
    "nbrToilet",
]


def load_model_for_evaluation(config: Dict[str, Any]) -> Tuple[Any, Any]:
    tokenizer, model = resolve_runtime(config["model"], config["runtime"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    logger = logging.getLogger("evaluate.runtime")

    if latest_checkpoint:
        logger.info("Loading LoRA adapter from %s", latest_checkpoint)
        model = PeftModel.from_pretrained(model, str(latest_checkpoint))
    else:
        logger.warning("No checkpoint found in %s. Falling back to base model evaluation.", checkpoint_dir)

    model.eval()
    return tokenizer, model


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    candidates = [path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    if not candidates:
        return None
    return sorted(candidates, key=checkpoint_sort_key)[-1]


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.split("-")[-1]), path.name
    except (TypeError, ValueError):
        return -1, path.name


def generate_predictions(
    examples: List[Dict[str, Any]],
    tokenizer: Any,
    model: Any,
    runtime_cfg: Dict[str, Any],
    max_new_tokens: int = 256,
    partial_output_path: Optional[Path] = None,
    progress_state_path: Optional[Path] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    device = resolve_inference_device(model, runtime_cfg)
    effective_max_new_tokens = resolve_generation_max_new_tokens(device, runtime_cfg, max_new_tokens)
    predictions: List[Dict[str, Any]] = []
    logger = logging.getLogger("evaluate.progress")
    total_examples = len(examples)
    started_at = time.monotonic()
    log_every = 25 if total_examples >= 200 else 10
    cached_predictions = load_partial_predictions(partial_output_path)
    cached_predictions_by_id = {entry["example_id"]: entry for entry in cached_predictions}
    completed_ids = {item["example_id"] for item in cached_predictions}
    if cached_predictions:
        resumed_count = len(cached_predictions)
        remaining_count = max(total_examples - resumed_count, 0)
        predictions = [cached_predictions_by_id[item["example_id"]] for item in examples if item["example_id"] in cached_predictions_by_id]
        logger.info(
            "Resume detected: %s/%s predictions already cached, %s remaining",
            resumed_count,
            total_examples,
            remaining_count,
            extra={
                "completed_examples": resumed_count,
                "total_examples": total_examples,
                "remaining_examples": remaining_count,
                "progress_pct": round((resumed_count / total_examples) * 100, 2) if total_examples else 100.0,
                "last_example_id": predictions[-1]["example_id"] if predictions else None,
                "resumed_from_cache": True,
            },
        )
        if on_progress:
            on_progress(
                {
                    "completed_examples": resumed_count,
                    "total_examples": total_examples,
                    "remaining_examples": remaining_count,
                    "progress_pct": round((resumed_count / total_examples) * 100, 2) if total_examples else 100.0,
                    "last_example_id": predictions[-1]["example_id"] if predictions else None,
                    "resumed_from_cache": True,
                }
            )

    logger.info(
        "Starting prediction generation on %s examples",
        total_examples,
        extra={
            "total_examples": total_examples,
            "device": str(device),
            "max_new_tokens": effective_max_new_tokens,
            "requested_max_new_tokens": max_new_tokens,
        },
    )

    writer = None
    if partial_output_path is not None:
        partial_output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = partial_output_path.open("a", encoding="utf-8")

    for index, example in enumerate(examples, start=1):
        if example["example_id"] in completed_ids:
            continue
        prompt = build_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=effective_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed_json = normalize_prediction(try_parse_json(generated_text))
        clean_generated_text = (
            json.dumps(parsed_json, ensure_ascii=False, sort_keys=True) if parsed_json is not None else generated_text
        )
        prediction_record = {
            "example_id": example["example_id"],
            "generated_text": clean_generated_text,
            "predicted_json": parsed_json,
            "target_json": example["target_json"],
        }
        predictions.append(prediction_record)
        if writer is not None:
            writer.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            writer.flush()

        completed_count = len(predictions)

        if progress_state_path is not None:
            write_progress_state(
                progress_state_path,
                completed_examples=completed_count,
                total_examples=total_examples,
                last_example_id=example["example_id"],
                partial_output_path=str(partial_output_path) if partial_output_path else None,
            )

        if completed_count % log_every == 0 or completed_count == total_examples:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = completed_count / elapsed
            remaining = max(total_examples - completed_count, 0)
            eta_seconds = remaining / rate if rate else 0.0
            progress_payload = {
                "completed_examples": completed_count,
                "total_examples": total_examples,
                "progress_pct": round((completed_count / total_examples) * 100, 2) if total_examples else 100.0,
                "elapsed_seconds": round(elapsed, 2),
                "eta_seconds": round(eta_seconds, 2),
                "examples_per_second": round(rate, 4),
                "last_example_id": example["example_id"],
            }
            logger.info(
                "Prediction progress %s/%s (%.1f%%) elapsed=%s eta=%s rate=%.2f ex/s",
                completed_count,
                total_examples,
                (completed_count / total_examples) * 100 if total_examples else 100.0,
                format_duration(elapsed),
                format_duration(eta_seconds),
                rate,
                extra=progress_payload,
            )
            if on_progress:
                on_progress(progress_payload)

    if writer is not None:
        writer.close()
    return predictions


def load_partial_predictions(partial_output_path: Optional[Path]) -> List[Dict[str, Any]]:
    if partial_output_path is None or not partial_output_path.exists():
        return []

    predictions: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line in partial_output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        example_id = payload.get("example_id")
        if not isinstance(example_id, str) or example_id in seen_ids:
            continue
        predictions.append(payload)
        seen_ids.add(example_id)
    return predictions


def write_progress_state(
    progress_state_path: Path,
    *,
    completed_examples: int,
    total_examples: int,
    last_example_id: Optional[str],
    partial_output_path: Optional[str],
) -> None:
    progress_state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "completed_examples": completed_examples,
        "total_examples": total_examples,
        "progress_pct": round((completed_examples / total_examples) * 100, 2) if total_examples else 100.0,
        "last_example_id": last_example_id,
        "partial_output_path": partial_output_path,
    }
    progress_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_inference_device(model: Any, runtime_cfg: Dict[str, Any]) -> torch.device:
    preferred = str(runtime_cfg.get("device_preference", "cpu")).lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_generation_max_new_tokens(device: torch.device, runtime_cfg: Dict[str, Any], requested: int) -> int:
    if device.type != "cpu":
        return requested
    cpu_limit = int(runtime_cfg.get("cpu_eval_max_new_tokens") or 0)
    if cpu_limit <= 0:
        cpu_limit = 96
    return min(requested, cpu_limit)


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_prediction(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None

    normalized = dict(payload)
    type_property = normalized.get("typeProperty")
    if isinstance(type_property, str):
        normalized["typeProperty"] = TYPE_ALIASES.get(type_property, type_property)

    tags = normalized.get("tags")
    if isinstance(tags, list):
        filtered_tags: List[str] = []
        for tag in tags:
            if isinstance(tag, str) and tag in ALLOWED_TAGS and tag not in filtered_tags:
                filtered_tags.append(tag)
        normalized["tags"] = filtered_tags

    if normalized.get("typeProperty") == "Studio":
        normalized["nbrRooms"] = 1
        normalized["roomType"] = None

    if normalized.get("typeProperty") == "Room":
        normalized["nbrRooms"] = 1
        normalized["nbrKitchens"] = None
        normalized["nbrBathrooms"] = None
        normalized["nbrToilets"] = None
        normalized["nbrFloorStudio"] = None
        normalized["numeroStudio"] = None

    return normalized


def compute_metrics(predictions: List[Dict[str, Any]]) -> TrainingMetrics:
    total = max(len(predictions), 1)
    valid_json = 0
    type_hits = 0
    status_hits = 0
    tag_f1_total = 0.0
    numeric_hits = 0
    numeric_total = 0

    for item in predictions:
        predicted = item["predicted_json"]
        target = item["target_json"]

        if predicted is not None:
            valid_json += 1

        if predicted and predicted.get("typeProperty") == target.get("typeProperty"):
            type_hits += 1

        if predicted and predicted.get("status") == target.get("status"):
            status_hits += 1

        tag_f1_total += tags_f1(predicted.get("tags") if predicted else [], target.get("tags") or [])

        if predicted:
            for field in NUMERIC_FIELDS:
                if field in target:
                    numeric_total += 1
                    if predicted.get(field) == target.get(field):
                        numeric_hits += 1
        else:
            for field in NUMERIC_FIELDS:
                if field in target:
                    numeric_total += 1

    return TrainingMetrics(
        json_valid_rate=valid_json / total,
        type_accuracy=type_hits / total,
        status_accuracy=status_hits / total,
        tags_f1=tag_f1_total / total,
        numeric_exact_match=(numeric_hits / numeric_total) if numeric_total else 0.0,
        evaluated_examples=len(predictions),
    )


def tags_f1(predicted_tags: List[str], target_tags: List[str]) -> float:
    predicted_set = set(predicted_tags or [])
    target_set = set(target_tags or [])
    if not predicted_set and not target_set:
        return 1.0
    if not predicted_set or not target_set:
        return 0.0

    tp = len(predicted_set & target_set)
    precision = tp / len(predicted_set)
    recall = tp / len(target_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def export_prediction_report(predictions: List[Dict[str, Any]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=True), encoding="utf-8")
