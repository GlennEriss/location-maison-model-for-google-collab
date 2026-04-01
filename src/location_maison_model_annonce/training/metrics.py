from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    json_valid_rate: float
    type_accuracy: float
    status_accuracy: float
    tags_f1: float
    numeric_exact_match: float
    evaluated_examples: int = 0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    perplexity: float = 0.0

    def dump(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
