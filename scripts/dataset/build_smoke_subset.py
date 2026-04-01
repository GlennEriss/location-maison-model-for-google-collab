from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a tiny smoke-test dataset from the existing JSONL splits.")
    parser.add_argument("--source-dir", default="data/datasets", help="Directory containing train/validation/test JSONL files")
    parser.add_argument("--output-dir", default="data/smoke_datasets", help="Directory where the smoke dataset will be written")
    parser.add_argument("--train-count", type=int, default=6, help="Number of train examples")
    parser.add_argument("--validation-count", type=int, default=2, help="Number of validation examples")
    parser.add_argument("--test-count", type=int, default=2, help="Number of test examples")
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows: list[dict[str, Any]], count: int, split: str) -> list[dict[str, Any]]:
    selected = []
    for index, row in enumerate(rows[:count], start=1):
        updated = dict(row)
        updated["split"] = split
        updated["example_id"] = f"smoke-{split}-{index:03d}"
        metadata = dict(updated.get("metadata") or {})
        metadata["smoke_test"] = True
        updated["metadata"] = metadata
        selected.append(updated)
    return selected


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    train_rows = select_rows(load_jsonl(source_dir / "train.jsonl"), args.train_count, "train")
    validation_rows = select_rows(load_jsonl(source_dir / "validation.jsonl"), args.validation_count, "validation")
    test_rows = select_rows(load_jsonl(source_dir / "test.jsonl"), args.test_count, "test")

    all_rows = train_rows + validation_rows + test_rows
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "smoke_dataset.jsonl", all_rows)

    manifest = {
        "generated_examples": len(all_rows),
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
