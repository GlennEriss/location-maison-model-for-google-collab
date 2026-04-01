from __future__ import annotations

from typing import Any, Dict

from peft import LoraConfig, TaskType


def build_lora_config(training_cfg: Dict[str, Any], lora_cfg: Dict[str, Any]) -> LoraConfig:
    task_type_name = str(training_cfg.get("task_type", "causal_lm")).upper()
    task_type = TaskType.CAUSAL_LM if task_type_name == "CAUSAL_LM" else TaskType.CAUSAL_LM
    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type=task_type,
    )
