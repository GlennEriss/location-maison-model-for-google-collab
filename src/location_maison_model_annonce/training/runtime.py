from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_runtime(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    logger = logging.getLogger("train.runtime")
    base_model = model_cfg["base_model"]
    model_source = resolve_model_source(base_model)
    local_files_only = model_source != base_model

    if local_files_only:
        logger.info("Using cached local model snapshot at %s", model_source)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    preferred_device = str(runtime_cfg.get("device_preference", "auto")).lower()
    target_device = choose_runtime_device(preferred_device, logger)
    use_cuda = target_device == "cuda"
    use_mps = target_device == "mps"
    device_map = None
    torch_dtype = torch.float16 if (use_mps or use_cuda) else torch.float32

    if use_mps:
        configure_mps_memory_budget(runtime_cfg, logger)
    elif target_device == "cpu":
        configure_cpu_runtime(runtime_cfg, logger)

    quantization = str(model_cfg.get("quantization", "none")).lower()
    if quantization != "none":
        logger.warning("Quantization '%s' is ignored on this local runtime; loading standard weights instead.", quantization)

    if use_cuda:
        configure_cuda_runtime(logger)
    logger.info("Loading model %s with dtype=%s on %s", model_source, torch_dtype, target_device)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        dtype=torch_dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )

    if use_cuda:
        model.to("cuda")
    elif use_mps:
        model.to("mps")

    return tokenizer, model


def choose_runtime_device(preferred_device: str, logger: logging.Logger) -> str:
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()

    if preferred_device in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return "cuda"
        if preferred_device == "cuda":
            logger.warning("CUDA was requested but is unavailable; falling back to CPU.")

    if preferred_device in {"auto", "mps"}:
        if mps_available:
            return "mps"
        if preferred_device == "mps" and mps_built and not mps_available:
            logger.warning(
                "MPS is built into torch but unavailable at runtime. On this Mac, recreating the venv with "
                "/opt/homebrew/bin/python3.13 is the recommended next step."
            )
        elif preferred_device == "mps" and not mps_built:
            logger.warning("Torch was installed without MPS support; falling back to CPU.")

    return "cpu"


def configure_cuda_runtime(logger: logging.Logger) -> None:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    logger.info("CUDA runtime optimizations enabled (tf32/high matmul precision where supported).")


def configure_cpu_runtime(runtime_cfg: Dict[str, Any], logger: logging.Logger) -> None:
    cpu_threads = int(runtime_cfg.get("cpu_num_threads") or 0)
    if cpu_threads <= 0:
        cpu_count = os.cpu_count() or 1
        cpu_threads = max(min(cpu_count, 4), 1)

    torch.set_num_threads(cpu_threads)
    interop_threads = max(min(cpu_threads // 2, 2), 1)
    torch.set_num_interop_threads(interop_threads)
    logger.info(
        "CPU/RAM fallback enabled with num_threads=%s interop_threads=%s. This mode is much slower than GPU but keeps the pipeline runnable.",
        cpu_threads,
        interop_threads,
    )


def configure_mps_memory_budget(runtime_cfg: Dict[str, Any], logger: logging.Logger) -> None:
    budget_gb = runtime_cfg.get("mps_memory_budget_gb")
    if not budget_gb:
        return

    recommended_memory_fn = getattr(torch.mps, "recommended_max_memory", None)
    if recommended_memory_fn is None:
        logger.warning("Unable to read recommended MPS max memory; keeping default allocator watermark.")
        return

    recommended_bytes = recommended_memory_fn()
    if not recommended_bytes:
        logger.warning("MPS recommended max memory returned 0; keeping default allocator watermark.")
        return

    budget_bytes = float(budget_gb) * 1024**3
    ratio = max(min(budget_bytes / float(recommended_bytes), 1.0), 0.05)
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{ratio:.4f}"
    logger.info(
        "Configured MPS memory budget to %.2f GiB (watermark ratio=%s, recommended max=%.2f GiB)",
        float(budget_gb),
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
        float(recommended_bytes) / 1024**3,
    )


def resolve_model_source(base_model: str) -> str:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_dir = cache_root / ("models--" + base_model.replace("/", "--"))
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return base_model

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return base_model
    return str(snapshots[-1])
