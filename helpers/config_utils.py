"""
Project: dementia_progression
File: helpers/config_utils.py

Synopsis:
    YAML configuration-loading utilities, including base-config inheritance and project-
    level setting resolution.

Author:
    puru panta (purupanta@uky.edu)

Date Created:
    2026-05-19

Last Updated:
    2026-05-19

Version:
    1.0

Purpose:
    Supports the dementia_progression pipeline for medication-state-aware and LLM-
    enhanced machine learning prediction of next-visit dementia progression among mild
    cognitive impairment visits.

Notes:
    This project uses participant-level train-validation-test splitting to prevent
    participant-level leakage. Neuropathology variables are excluded from model training
    and used only for secondary biological plausibility anchoring. Medication-state and
    LLM-derived features are interpreted as predictive patient-state representations,
    not as causal medication effects.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

BASE_KEYS = ("_base", "base_config", "inherits")

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out

def _as_base_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError("Base config reference must be a string or list of strings.")

def load_config(path: str | Path, visited: set[Path] | None = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    visited = visited or set()
    if path in visited:
        raise ValueError(f"Circular config inheritance detected at {path}")
    visited.add(path)
    raw = load_yaml(path)
    base_refs: list[str] = []
    for key in BASE_KEYS:
        if key in raw:
            base_refs.extend(_as_base_list(raw.pop(key)))
    merged: Dict[str, Any] = {}
    for ref in base_refs:
        base_path = Path(ref)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        merged = _deep_merge(merged, load_config(base_path, visited=visited.copy()))
    return _deep_merge(merged, raw)
