"""
Project: dementia_progression
File: helpers/cohort.py

Synopsis:
    Cohort-construction utilities for visit ordering, next-visit outcome labeling, MCI
    visit selection, and cohort summaries.

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
from typing import Dict
import pandas as pd

def add_visit_order_columns(df: pd.DataFrame, id_col: str, visit_number_col: str, visit_year_col: str, visit_month_col: str, visit_day_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in [visit_number_col, visit_year_col, visit_month_col, visit_day_col]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values([id_col, visit_number_col, visit_year_col, visit_month_col, visit_day_col], kind="mergesort").reset_index(drop=True)
    return out

def build_next_visit_labels(df: pd.DataFrame, id_col: str, diagnosis_col: str, dementia_code: int) -> pd.DataFrame:
    out = df.copy()
    out[diagnosis_col] = pd.to_numeric(out[diagnosis_col], errors="coerce")
    out["next_diagnosis"] = out.groupby(id_col, sort=False)[diagnosis_col].shift(-1)
    out["has_next_visit"] = out["next_diagnosis"].notna().astype(int)
    out["next_visit_dementia"] = (out["next_diagnosis"] == int(dementia_code)).astype("Int64")
    out.loc[out["has_next_visit"] == 0, "next_visit_dementia"] = pd.NA
    return out

def select_mci_with_next_visit(df: pd.DataFrame, diagnosis_col: str, mci_code: int) -> pd.DataFrame:
    cur = pd.to_numeric(df[diagnosis_col], errors="coerce")
    mask = cur.eq(int(mci_code)) & df["has_next_visit"].eq(1)
    return df.loc[mask].copy()

def summarize_cohort(df: pd.DataFrame, id_col: str) -> Dict[str, int | float]:
    y = df["next_visit_dementia"].dropna().astype(int)
    return {
        "rows": int(len(df)),
        "unique_participants": int(df[id_col].nunique()),
        "events": int(y.sum()),
        "non_events": int((y == 0).sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
    }
