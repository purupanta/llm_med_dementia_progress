"""
Project: dementia_progression
File: helpers/evaluation.py

Synopsis:
    Prediction-evaluation utilities for discrimination, precision, Brier score, balanced
    accuracy, and threshold-based model summaries.

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
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

def evaluate_binary_predictions(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true_arr)),
        "prevalence": float(np.mean(y_true_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "average_precision": float(average_precision_score(y_true_arr, y_prob_arr)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "accuracy_at_0_5": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "sensitivity_at_0_5": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity_at_0_5": float(tn / max(tn + fp, 1)),
        "precision_at_0_5": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

def build_metrics_table(model_probabilities: Dict[str, np.ndarray], y_true: pd.Series | np.ndarray, threshold: float = 0.50) -> pd.DataFrame:
    rows = []
    for model_name, y_prob in model_probabilities.items():
        row = evaluate_binary_predictions(y_true, y_prob, threshold=threshold)
        row["model"] = model_name
        rows.append(row)
    cols = [
        "model", "n", "prevalence", "roc_auc", "average_precision", "brier_score",
        "accuracy_at_0_5", "balanced_accuracy_at_0_5", "sensitivity_at_0_5",
        "specificity_at_0_5", "precision_at_0_5", "f1_at_0_5", "tp", "fp", "tn", "fn",
    ]
    return pd.DataFrame(rows)[cols].sort_values("roc_auc", ascending=False).reset_index(drop=True)
