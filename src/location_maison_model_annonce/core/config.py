from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Union

import yaml
from pydantic import BaseModel


class ProjectPaths(BaseModel):
    raw_data_dir: str
    interim_data_dir: str
    processed_data_dir: str
    dataset_dir: str
    log_dir: str
    app_log_dir: str
    train_log_dir: str
    error_log_dir: str
    checkpoint_dir: str
    metrics_dir: str
    report_dir: str


class AppConfig(BaseModel):
    project: dict[str, Any]
    paths: ProjectPaths
    dataset: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    lora: dict[str, Any]
    logging: dict[str, Any]
    runtime: dict[str, Any]


def load_config(config_path: Union[str, Path]) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return AppConfig.model_validate(payload)
