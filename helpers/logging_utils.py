"""
Project: dementia_progression
File: helpers/logging_utils.py

Synopsis:
    Logging utilities for writing auditable run-specific logs to each timestamped
    output directory.

Author:
    puru panta (purupanta@uky.edu)

Date Created:
    2026-05-19

Last Updated:
    2026-05-22

Version:
    1.51
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _reset_logger(logger: logging.Logger) -> None:
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)


def create_file_logger(name: str, log_path: str | Path, *, console: bool = False) -> logging.Logger:
    """Create an isolated named logger with one UTF-8 file handler.

    The project intentionally uses separate log files so LLM errors, resolved
    configuration, runtime environment, and pipeline progress can be reviewed
    independently after long runs.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    _reset_logger(logger)
    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def setup_logger(log_path: str | Path) -> logging.Logger:
    """Backward-compatible pipeline logger constructor."""
    return create_file_logger("dementia_progression.pipeline", log_path, console=True)


def setup_run_loggers(logs_dir: str | Path) -> dict[str, logging.Logger]:
    """Create all run-specific loggers requested for the pipeline."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    loggers = {
        "pipeline": create_file_logger("dementia_progression.pipeline", logs_dir / "pipeline.log", console=True),
        "llm": create_file_logger("dementia_progression.llm", logs_dir / "llm.log", console=False),
        "config": create_file_logger("dementia_progression.config", logs_dir / "config.log", console=False),
        "runtime": create_file_logger("dementia_progression.runtime", logs_dir / "runtime.log", console=False),
        "gpu": create_file_logger("dementia_progression.gpu", logs_dir / "gpu.log", console=False),
        "warnings": create_file_logger("dementia_progression.warnings", logs_dir / "warnings.log", console=False),
    }

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        loggers["warnings"].warning("%s:%s | %s | %s", filename, lineno, category.__name__, message)

    warnings.showwarning = _showwarning
    logging.captureWarnings(True)
    return loggers


def log_resolved_config(config_logger: logging.Logger, cfg: dict[str, Any]) -> None:
    """Write resolved YAML plus environment overrides to config.log."""
    config_logger.info("Resolved configuration after YAML inheritance and environment overrides follows.")
    config_logger.info("%s", json.dumps(cfg, indent=2, sort_keys=True, default=str))


def log_runtime_environment(runtime_logger: logging.Logger, *, project_root: str | Path, input_csv: str | Path, output_dir: str | Path) -> None:
    """Write runtime, Python, platform, and selected environment variables."""
    runtime_logger.info("Project root: %s", Path(project_root).resolve())
    runtime_logger.info("Input CSV: %s", Path(input_csv).resolve())
    runtime_logger.info("Output directory: %s", Path(output_dir).resolve())
    runtime_logger.info("Python executable: %s", sys.executable)
    runtime_logger.info("Python version: %s", sys.version.replace("\n", " "))
    runtime_logger.info("Platform: %s", platform.platform())
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "GPU_ID",
        "CUDA_GPU_ID",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LLM_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "LLM_MAX_UNIQUE_TEXTS",
        "LLM_NUM_PREDICT",
        "LLM_REQUEST_TIMEOUT_S",
        "LLM_PERSISTENT_CACHE_PATH",
        "DEMENTIA_OLLAMA_MODE",
        "DEMENTIA_RUN_TIMESTAMP",
    ]
    for key in keys:
        runtime_logger.info("ENV %s=%s", key, os.environ.get(key, "<unset>"))


def log_gpu_environment(gpu_logger: logging.Logger) -> None:
    """Write GPU/CUDA visibility and a compact nvidia-smi snapshot when available."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "GPU_ID",
        "CUDA_GPU_ID",
        "OLLAMA_HOST",
        "OLLAMA_BASE_URL",
        "DEMENTIA_OLLAMA_MODE",
        "DEMENTIA_OLLAMA_PORT",
        "OLLAMA_NUM_PARALLEL",
        "OLLAMA_MAX_LOADED_MODELS",
        "OLLAMA_CONTEXT_LENGTH",
    ]
    gpu_logger.info("GPU and Ollama visibility environment follows.")
    for key in keys:
        gpu_logger.info("ENV %s=%s", key, os.environ.get(key, "<unset>"))
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(query_cmd, text=True, capture_output=True, timeout=10, check=False)
        gpu_logger.info("nvidia-smi query returncode=%s", result.returncode)
        if result.stdout.strip():
            gpu_logger.info("nvidia-smi gpu summary:\n%s", result.stdout.strip())
        if result.stderr.strip():
            gpu_logger.warning("nvidia-smi query stderr:\n%s", result.stderr.strip())
    except FileNotFoundError:
        gpu_logger.warning("nvidia-smi command not found; GPU inventory was not logged.")
    except Exception as exc:  # pragma: no cover - environment-specific diagnostic only
        gpu_logger.warning("nvidia-smi query failed: %s", exc)
