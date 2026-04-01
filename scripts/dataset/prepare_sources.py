from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
POSTS_DIR = ROOT / "data" / "post-for-facebook"
PROPERTIES_DIR = ROOT / "data" / "source_properties"
OUTPUT_DIR = ROOT / "data" / "processed" / "relevance"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    offer_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    irrelevant_rows: list[dict[str, Any]] = []
    uncertain_rows: list[dict[str, Any]] = []

    for path in sorted(POSTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            text = clean_text(row.get("text") or row.get("message") or row.get("description"))
            normalized = {
                "source_file": path.name,
                "source_index": index,
                "facebookUrl": row.get("facebookUrl"),
                "user": row.get("user"),
                "likesCount": row.get("likesCount"),
                "commentsCount": row.get("commentsCount"),
                "text": text,
                "classification": classify_post(text),
            }
            bucket = normalized["classification"]
            if bucket == "offer":
                offer_rows.append(normalized)
            elif bucket == "request":
                request_rows.append(normalized)
            elif bucket == "uncertain":
                uncertain_rows.append(normalized)
            else:
                irrelevant_rows.append(normalized)

    write_jsonl(OUTPUT_DIR / "offers.jsonl", offer_rows)
    write_jsonl(OUTPUT_DIR / "requests.jsonl", request_rows)
    write_jsonl(OUTPUT_DIR / "uncertain.jsonl", uncertain_rows)
    write_jsonl(OUTPUT_DIR / "irrelevant.jsonl", irrelevant_rows)

    inventory = {
        "posts": {
            "offer": len(offer_rows),
            "request": len(request_rows),
            "uncertain": len(uncertain_rows),
            "irrelevant": len(irrelevant_rows),
        },
        "property_exports": [path.name for path in sorted(PROPERTIES_DIR.glob("*")) if path.is_file()],
    }
    (OUTPUT_DIR / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


def classify_post(text: str) -> str:
    normalized = text.lower()
    if not normalized:
        return "irrelevant"
    if any(keyword in normalized for keyword in ("bonjour besoin", "je cherche", "besoin d'une", "recherche", "cherche")):
        return "request"
    offer_keywords = (
        "à louer",
        "a louer",
        "à vendre",
        "a vendre",
        "studio",
        "appartement",
        "villa",
        "maison",
        "terrain",
        "immeuble",
        "bureau",
        "boutique",
        "kiosque",
        "chambre",
    )
    if any(keyword in normalized for keyword in offer_keywords):
        return "offer"
    return "uncertain"


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\r", " ").split())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
