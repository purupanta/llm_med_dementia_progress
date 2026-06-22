"""
Project: dementia_progression
File: helpers/modeling.py

Synopsis:
    Model-development utilities for participant-level splitting, preprocessing
    pipelines, model fitting, prediction, tuning, and persistence.

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
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


def _has_both_classes(y: pd.Series | np.ndarray) -> bool:
    y_arr = np.asarray(y).astype(int)
    return len(np.unique(y_arr)) >= 2


def group_train_test_split_indices(
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
    return train_idx, test_idx


def group_train_validation_test_split_indices(
    y: pd.Series,
    groups: pd.Series,
    *,
    validation_size: float,
    test_size: float,
    random_state: int,
    max_attempts: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create participant-level train/validation/test indices without group leakage.

    Sizes are interpreted as proportions of the full analytic cohort. The first split
    reserves the final held-out test set. The remaining development set is then split
    into training and validation sets. The function retries nearby random seeds so that
    each split contains both outcome classes whenever the data support it.
    """
    validation_size = float(validation_size)
    test_size = float(test_size)
    if len(y) == 0:
        raise ValueError("No analytic cohort rows are available for participant-level train-validation-test splitting.")
    if pd.Series(groups).nunique() < 3:
        raise ValueError("At least 3 unique participant groups are required for participant-level train-validation-test splitting.")
    if pd.Series(y).dropna().nunique() < 2:
        raise ValueError("Both outcome classes are required for participant-level train-validation-test splitting.")
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must both be positive.")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.0.")
    development_size = 1.0 - test_size
    validation_fraction_within_development = validation_size / development_size
    last_split: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    y_reset = pd.Series(y).reset_index(drop=True)
    groups_reset = pd.Series(groups).reset_index(drop=True)

    for attempt in range(int(max_attempts)):
        seed = int(random_state) + attempt
        test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        development_idx, test_idx = next(test_splitter.split(np.zeros(len(y_reset)), y_reset, groups=groups_reset))
        development_y = y_reset.iloc[development_idx].reset_index(drop=True)
        development_groups = groups_reset.iloc[development_idx].reset_index(drop=True)
        validation_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction_within_development,
            random_state=seed + 100_003,
        )
        train_rel, validation_rel = next(
            validation_splitter.split(np.zeros(len(development_idx)), development_y, groups=development_groups)
        )
        train_idx = development_idx[train_rel]
        validation_idx = development_idx[validation_rel]
        last_split = (train_idx, validation_idx, test_idx)
        if (
            _has_both_classes(y_reset.iloc[train_idx])
            and _has_both_classes(y_reset.iloc[validation_idx])
            and _has_both_classes(y_reset.iloc[test_idx])
        ):
            return train_idx, validation_idx, test_idx

    if last_split is None:
        raise RuntimeError("Could not create a participant-level train/validation/test split.")
    train_idx, validation_idx, test_idx = last_split
    if not _has_both_classes(y_reset.iloc[train_idx]):
        raise ValueError("Training split does not contain both outcome classes.")
    if not _has_both_classes(y_reset.iloc[validation_idx]):
        raise ValueError("Validation split does not contain both outcome classes.")
    if not _has_both_classes(y_reset.iloc[test_idx]):
        raise ValueError("Test split does not contain both outcome classes.")
    return train_idx, validation_idx, test_idx


def build_logistic_pipeline(preprocessor, cfg: dict) -> Pipeline:
    logistic_cfg = cfg.get("modeling", {}).get("logistic", {})
    solver = str(logistic_cfg.get("solver", "lbfgs"))
    model = LogisticRegression(
        max_iter=int(logistic_cfg.get("max_iter", 3000)),
        class_weight=logistic_cfg.get("class_weight", "balanced"),
        C=float(logistic_cfg.get("C", 0.5)),
        solver=solver,
        random_state=int(cfg.get("random_state", 42)),
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_hist_gradient_pipeline(preprocessor, cfg: dict) -> Pipeline:
    hgb_cfg = cfg.get("modeling", {}).get("hist_gradient_boosting", {})
    model = HistGradientBoostingClassifier(
        max_iter=int(hgb_cfg.get("max_iter", 300)),
        learning_rate=float(hgb_cfg.get("learning_rate", 0.04)),
        max_leaf_nodes=int(hgb_cfg.get("max_leaf_nodes", 15)),
        l2_regularization=float(hgb_cfg.get("l2_regularization", 0.05)),
        min_samples_leaf=int(hgb_cfg.get("min_samples_leaf", 30)),
        early_stopping=bool(hgb_cfg.get("early_stopping", True)),
        validation_fraction=float(hgb_cfg.get("validation_fraction", 0.15)),
        n_iter_no_change=int(hgb_cfg.get("n_iter_no_change", 20)),
        random_state=int(cfg.get("random_state", 42)),
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def fit_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    use_balanced_sample_weight: bool,
) -> Pipeline:
    """Fit a model pipeline with safeguards for very small development subsets.

    Range-limited development runs may contain only a small number of MCI visits.
    HistGradientBoostingClassifier's internal early-stopping validation split can
    fail when a class has too few examples. For those small diagnostic runs, we
    disable internal early stopping while preserving the full-data configuration.
    """
    y_arr = np.asarray(y_train).astype(int)
    if name.endswith("hist_gradient_boosting"):
        class_counts = np.bincount(y_arr, minlength=2) if y_arr.size else np.array([0, 0])
        if y_arr.size < 100 or class_counts.min() < 5:
            pipeline.set_params(model__early_stopping=False)
        if use_balanced_sample_weight:
            sw = compute_sample_weight(class_weight="balanced", y=y_arr)
            return pipeline.fit(X_train, y_train, model__sample_weight=sw)
    return pipeline.fit(X_train, y_train)


def predict_proba_positive(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    scores = pipeline.decision_function(X)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-12)


def save_models_bundle(path: str, models: Dict[str, Pipeline], metadata: dict) -> None:
    dump({"models": models, "metadata": metadata}, path, compress=3)
