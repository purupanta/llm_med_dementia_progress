"""
Project: dementia_progression
File: helpers/io_utils.py

Synopsis:
    Input/output utilities for reading selected columns, creating output directories,
    and writing reproducible artifacts.

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
from typing import Any, Iterable, List
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out

def build_drug_columns(prefix: str, max_drug_columns: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, int(max_drug_columns) + 1)]

def _coerce_optional_int(value: Any, name: str) -> int | None:
    """Coerce YAML row-window values such as null, "all", or an integer."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "none", "null"}:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, null, or 'all'; got {value!r}") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, null, or 'all'; got {value!r}") from exc


def normalize_input_record_selection(selection_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize YAML-controlled raw input row selection without a mode field.

    Row indexes are zero-based over data rows, excluding the header. The minimum
    index is inclusive. The maximum index is exclusive. Therefore,
    min_row_index=0 and max_row_index=200 reads the first 200 raw input records.
    Use max_row_index: "all" to read through the end of the file.
    """
    cfg = dict(selection_cfg or {})
    row_min = _coerce_optional_int(cfg.get("min_row_index", 0), "min_row_index")
    row_max = _coerce_optional_int(cfg.get("max_row_index", None), "max_row_index")
    if row_min is None:
        row_min = 0
    if row_min < 0:
        raise ValueError("runtime.input_record_selection.min_row_index must be >= 0.")
    if row_max is not None and row_max <= row_min:
        raise ValueError("runtime.input_record_selection.max_row_index must be greater than min_row_index.")

    return {
        "min_row_index": row_min,
        "max_row_index": row_max,
        "max_row_index_label": "all" if row_max is None else row_max,
        "row_window_label": f"{row_min}:" + ("all" if row_max is None else str(row_max)),
    }

def load_selected_columns(
    csv_path: str | Path,
    usecols: list[str],
    row_min_index: int = 0,
    row_max_index: int | None = None,
) -> pd.DataFrame:
    """Read selected CSV columns with an optional zero-based raw-row window.

    Parameters
    ----------
    csv_path:
        Input CSV path.
    usecols:
        Required columns to load.
    row_min_index:
        Zero-based inclusive first raw data row to read, excluding the header.
    row_max_index:
        Zero-based exclusive raw data row boundary. None reads to the end.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if int(row_min_index) < 0:
        raise ValueError("row_min_index must be >= 0.")
    if row_max_index is not None and int(row_max_index) <= int(row_min_index):
        raise ValueError("row_max_index must be greater than row_min_index.")

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing[:25]}{' ...' if len(missing)>25 else ''}")
    dtype_map = {c: "string" for c in usecols if str(c).upper().startswith("DRUG")}

    read_kwargs: dict[str, Any] = {
        "usecols": usecols,
        "dtype": dtype_map,
        "low_memory": True,
    }
    if int(row_min_index) > 0:
        read_kwargs["skiprows"] = range(1, int(row_min_index) + 1)
    if row_max_index is not None:
        read_kwargs["nrows"] = int(row_max_index) - int(row_min_index)

    return pd.read_csv(csv_path, **read_kwargs)
