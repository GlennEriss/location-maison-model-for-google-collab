from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MetricsFileCallback(TrainerCallback):
    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        payload = {"global_step": state.global_step, "epoch": state.epoch, "logs": logs}
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
