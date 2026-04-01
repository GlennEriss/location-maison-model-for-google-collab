from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


STANDARD_LOG_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = {key: value for key, value in record.__dict__.items() if key not in STANDARD_LOG_KEYS}
        if extra:
            payload["extra"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(
    log_dir: Path,
    app_log_file: str,
    error_log_file: str,
    train_log_file: Optional[str] = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    app_path = log_dir / app_log_file
    app_path.parent.mkdir(parents=True, exist_ok=True)
    app_handler = logging.FileHandler(app_path)
    app_handler.setFormatter(JsonFormatter())

    error_path = log_dir / error_log_file
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(error_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JsonFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(app_handler)
    logger.addHandler(error_handler)

    if train_log_file:
        train_path = log_dir / train_log_file
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_handler = logging.FileHandler(train_path)
        train_handler.setFormatter(JsonFormatter())
        logger.addHandler(train_handler)
