from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict

from datasets import DatasetDict
from peft import PeftModel, get_peft_model
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from location_maison_model_annonce.training.callbacks import MetricsFileCallback
from location_maison_model_annonce.training.data import build_training_prompt
from location_maison_model_annonce.training.lora import build_lora_config
from location_maison_model_annonce.training.metrics import TrainingMetrics


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_training_prompt(example)
        encoded = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    keep_columns = dataset["train"].column_names
    return dataset.map(tokenize, remove_columns=keep_columns)


def build_trainer(
    config: Dict[str, Any],
    dataset: DatasetDict,
    tokenizer: Any,
    model: Any,
    adapter_checkpoint: str | None = None,
) -> Trainer:
    logger = logging.getLogger("train.trainer")
    paths = config["paths"]
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    logging_cfg = config["logging"]

    peft_config = build_lora_config(training_cfg, lora_cfg)
    if adapter_checkpoint:
        logger.info("Loading trainable LoRA adapter from %s", adapter_checkpoint)
        model = PeftModel.from_pretrained(model, adapter_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    tokenized = tokenize_dataset(dataset, tokenizer, int(training_cfg["max_seq_length"]))
    use_cuda_fp16 = bool(str(training_cfg.get("mixed_precision", "")).lower() == "fp16" and torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()
    use_mps = bool(config["runtime"].get("device_preference") == "mps" and torch.backends.mps.is_available())
    use_cpu_fallback = not use_cuda and not use_mps
    train_dataset_size = max(len(tokenized["train"]), 1)
    effective_batch_size = max(
        int(training_cfg["batch_size"]) * int(training_cfg["gradient_accumulation_steps"]),
        1,
    )
    steps_per_epoch = max(math.ceil(train_dataset_size / effective_batch_size), 1)
    warmup_steps = max(
        int(round(steps_per_epoch * float(training_cfg["epochs"]) * float(training_cfg["warmup_ratio"]))),
        0,
    )
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(Path(paths["train_log_dir"]))
    dataloader_num_workers = int(config["runtime"]["num_workers"])
    if use_cpu_fallback:
        dataloader_num_workers = min(dataloader_num_workers, 1)
        logger.warning(
            "Trainer is running in CPU/RAM fallback mode. Expect much slower training/evaluation than on GPU."
        )
    training_args = TrainingArguments(
        output_dir=str(Path(paths["checkpoint_dir"])),
        num_train_epochs=float(training_cfg["epochs"]),
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        per_device_eval_batch_size=int(training_cfg["batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(training_cfg["weight_decay"]),
        logging_steps=int(training_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(training_cfg["eval_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=bool(use_cuda),
        fp16=use_cuda_fp16,
        bf16=False,
        gradient_checkpointing=True,
    )

    logger.info("Training arguments prepared")
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[MetricsFileCallback(Path(logging_cfg["metrics_file"]))],
    )


def export_training_metrics(metrics_path: Path, train_result: Dict[str, Any], eval_result: Dict[str, Any]) -> TrainingMetrics:
    eval_loss = float(eval_result.get("eval_loss", 0.0))
    perplexity = math.exp(eval_loss) if eval_loss and eval_loss < 20 else 0.0
    metrics = TrainingMetrics(
        json_valid_rate=0.0,
        type_accuracy=0.0,
        status_accuracy=0.0,
        tags_f1=0.0,
        numeric_exact_match=0.0,
        evaluated_examples=0,
        train_loss=float(train_result.get("train_loss", 0.0)),
        eval_loss=eval_loss,
        perplexity=perplexity,
    )
    metrics.dump(metrics_path)
    return metrics
