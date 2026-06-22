"""
Project: dementia_progression
File: src/pipeline_step3_split_fit.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Creates participant-level train/validation/test partitions, tunes model hyperparameters on validation data, refits final models, and writes validation audit outputs.

Design:
    This module contains the executable logic for Step 3/7 participant split and model fitting. It is intentionally
    separated from the main orchestrator so that GitHub users can inspect,
    test, and maintain one numbered pipeline stage at a time.
"""

from __future__ import annotations

from src.pipeline_context import PipelineContext
from src.pipeline_common import *  # Reuse the validated v1.49 helper surface.


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 3/7 participant split and model fitting and store downstream state in ``ctx``.

    Returns
    -------
    bool
        ``True`` when downstream stages should continue; ``False`` when the
        stage wrote controlled-stop outputs for a range-limited/non-modelable run.
    """
    cfg = ctx.cfg
    output_dir = ctx.output_dir
    logger = ctx.logger
    decimal_places = ctx.decimal_places
    progress_enabled = ctx.progress_enabled
    threshold = ctx.threshold
    modeling_df = ctx.modeling_df
    y = ctx.y
    groups = ctx.groups
    clinical_features = ctx.clinical_features
    medication_features = ctx.medication_features
    llm_medication_features = ctx.llm_medication_features
    neural_medication_features = ctx.neural_medication_features
    trajectory_clinical_features = getattr(ctx, "trajectory_clinical_features", clinical_features)
    trajectory_medication_features = getattr(ctx, "trajectory_medication_features", medication_features)
    trajectory_llm_medication_features = getattr(ctx, "trajectory_llm_medication_features", llm_medication_features)
    trajectory_neural_medication_features = getattr(ctx, "trajectory_neural_medication_features", neural_medication_features)
    clinical_X = ctx.clinical_X
    med_X = ctx.med_X
    llm_X = ctx.llm_X
    neural_X = ctx.neural_X
    trajectory_clinical_X = getattr(ctx, "trajectory_clinical_X", clinical_X)
    trajectory_med_X = getattr(ctx, "trajectory_med_X", med_X)
    trajectory_llm_X = getattr(ctx, "trajectory_llm_X", llm_X)
    trajectory_neural_X = getattr(ctx, "trajectory_neural_X", neural_X)

    progress_step("Step 3/7: Creating participant-level train-validation-test split", enabled=progress_enabled, logger=logger)
    step3_bar = progress_bar(total=5, desc="Step 3/7 split and model fitting", enabled=progress_enabled, unit="task", logger=logger)
    split_cfg = cfg.get("split", {})
    validation_size = float(split_cfg.get("validation_size", cfg.get("validation_size", 0.10)))
    test_size = float(split_cfg.get("test_size", cfg.get("test_size", 0.20)))
    train_idx, validation_idx, test_idx = group_train_validation_test_split_indices(
        y,
        groups,
        validation_size=validation_size,
        test_size=test_size,
        random_state=int(cfg["random_state"]),
    )
    train_final_idx = np.concatenate([train_idx, validation_idx])
    y_train = y.iloc[train_idx]
    y_validation = y.iloc[validation_idx]
    y_test = y.iloc[test_idx]
    y_train_final = y.iloc[train_final_idx]
    logger.info(
        "Participant-level train-validation-test split: train_rows=%s validation_rows=%s test_rows=%s train_participants=%s validation_participants=%s test_participants=%s",
        len(train_idx),
        len(validation_idx),
        len(test_idx),
        groups.iloc[train_idx].nunique(),
        groups.iloc[validation_idx].nunique(),
        groups.iloc[test_idx].nunique(),
    )
    split_summary = _build_split_summary(
        modeling_df,
        y,
        groups,
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
        id_col=cfg["id_col"],
    )
    _round_csv(split_summary, output_file(output_dir, "step1c_participant_train_validation_test_split_summary.csv"), decimal_places)
    step3_bar.update("Create participant-level train-validation-test split and write summary")

    X_clin_train = clinical_X.iloc[train_idx].copy()
    X_clin_validation = clinical_X.iloc[validation_idx].copy()
    X_clin_test = clinical_X.iloc[test_idx].copy()
    X_clin_train_final = clinical_X.iloc[train_final_idx].copy()
    X_med_train = med_X.iloc[train_idx].copy()
    X_med_validation = med_X.iloc[validation_idx].copy()
    X_med_test = med_X.iloc[test_idx].copy()
    X_med_train_final = med_X.iloc[train_final_idx].copy()
    X_llm_train = llm_X.iloc[train_idx].copy()
    X_llm_validation = llm_X.iloc[validation_idx].copy()
    X_llm_test = llm_X.iloc[test_idx].copy()
    X_llm_train_final = llm_X.iloc[train_final_idx].copy()
    X_neural_train = neural_X.iloc[train_idx].copy()
    X_neural_validation = neural_X.iloc[validation_idx].copy()
    X_neural_test = neural_X.iloc[test_idx].copy()
    X_neural_train_final = neural_X.iloc[train_final_idx].copy()
    X_traj_clin_train = trajectory_clinical_X.iloc[train_idx].copy()
    X_traj_clin_validation = trajectory_clinical_X.iloc[validation_idx].copy()
    X_traj_clin_test = trajectory_clinical_X.iloc[test_idx].copy()
    X_traj_clin_train_final = trajectory_clinical_X.iloc[train_final_idx].copy()
    X_traj_med_train = trajectory_med_X.iloc[train_idx].copy()
    X_traj_med_validation = trajectory_med_X.iloc[validation_idx].copy()
    X_traj_med_test = trajectory_med_X.iloc[test_idx].copy()
    X_traj_med_train_final = trajectory_med_X.iloc[train_final_idx].copy()
    X_traj_llm_train = trajectory_llm_X.iloc[train_idx].copy()
    X_traj_llm_validation = trajectory_llm_X.iloc[validation_idx].copy()
    X_traj_llm_test = trajectory_llm_X.iloc[test_idx].copy()
    X_traj_llm_train_final = trajectory_llm_X.iloc[train_final_idx].copy()
    X_traj_neural_train = trajectory_neural_X.iloc[train_idx].copy()
    X_traj_neural_validation = trajectory_neural_X.iloc[validation_idx].copy()
    X_traj_neural_test = trajectory_neural_X.iloc[test_idx].copy()
    X_traj_neural_train_final = trajectory_neural_X.iloc[train_final_idx].copy()
    step3_bar.update("Construct train, validation, test, and train-plus-validation matrices")

    use_hgb_weights = bool(cfg.get("modeling", {}).get("hist_gradient_boosting", {}).get("use_balanced_sample_weight", True))
    model_specs = {
        "clinical_only_logistic": {
            "feature_lists": clinical_features,
            "X_train": X_clin_train,
            "X_validation": X_clin_validation,
            "X_test": X_clin_test,
            "X_train_final": X_clin_train_final,
        },
        "structured_medication_aware_logistic": {
            "feature_lists": medication_features,
            "X_train": X_med_train,
            "X_validation": X_med_validation,
            "X_test": X_med_test,
            "X_train_final": X_med_train_final,
        },
        "llm_enhanced_medication_aware_logistic": {
            "feature_lists": llm_medication_features,
            "X_train": X_llm_train,
            "X_validation": X_llm_validation,
            "X_test": X_llm_test,
            "X_train_final": X_llm_train_final,
        },
        "clinical_only_hist_gradient_boosting": {
            "feature_lists": clinical_features,
            "X_train": X_clin_train,
            "X_validation": X_clin_validation,
            "X_test": X_clin_test,
            "X_train_final": X_clin_train_final,
        },
        "structured_medication_aware_hist_gradient_boosting": {
            "feature_lists": medication_features,
            "X_train": X_med_train,
            "X_validation": X_med_validation,
            "X_test": X_med_test,
            "X_train_final": X_med_train_final,
        },
        "llm_enhanced_medication_aware_hist_gradient_boosting": {
            "feature_lists": llm_medication_features,
            "X_train": X_llm_train,
            "X_validation": X_llm_validation,
            "X_test": X_llm_test,
            "X_train_final": X_llm_train_final,
        },
    }
    if longitudinal_trajectory_feature_columns(modeling_df):
        model_specs.update(
            {
                "longitudinal_clinical_trajectory_logistic": {
                    "feature_lists": trajectory_clinical_features,
                    "X_train": X_traj_clin_train,
                    "X_validation": X_traj_clin_validation,
                    "X_test": X_traj_clin_test,
                    "X_train_final": X_traj_clin_train_final,
                },
                "longitudinal_medication_trajectory_logistic": {
                    "feature_lists": trajectory_medication_features,
                    "X_train": X_traj_med_train,
                    "X_validation": X_traj_med_validation,
                    "X_test": X_traj_med_test,
                    "X_train_final": X_traj_med_train_final,
                },
                "longitudinal_llm_medication_trajectory_logistic": {
                    "feature_lists": trajectory_llm_medication_features,
                    "X_train": X_traj_llm_train,
                    "X_validation": X_traj_llm_validation,
                    "X_test": X_traj_llm_test,
                    "X_train_final": X_traj_llm_train_final,
                },
                "longitudinal_clinical_trajectory_hist_gradient_boosting": {
                    "feature_lists": trajectory_clinical_features,
                    "X_train": X_traj_clin_train,
                    "X_validation": X_traj_clin_validation,
                    "X_test": X_traj_clin_test,
                    "X_train_final": X_traj_clin_train_final,
                },
                "longitudinal_medication_trajectory_hist_gradient_boosting": {
                    "feature_lists": trajectory_medication_features,
                    "X_train": X_traj_med_train,
                    "X_validation": X_traj_med_validation,
                    "X_test": X_traj_med_test,
                    "X_train_final": X_traj_med_train_final,
                },
                "longitudinal_llm_medication_trajectory_hist_gradient_boosting": {
                    "feature_lists": trajectory_llm_medication_features,
                    "X_train": X_traj_llm_train,
                    "X_validation": X_traj_llm_validation,
                    "X_test": X_traj_llm_test,
                    "X_train_final": X_traj_llm_train_final,
                },
            }
        )
    if neural_text_feature_columns(modeling_df):
        model_specs.update(
            {
                "neural_medication_aware_logistic": {
                    "feature_lists": neural_medication_features,
                    "X_train": X_neural_train,
                    "X_validation": X_neural_validation,
                    "X_test": X_neural_test,
                    "X_train_final": X_neural_train_final,
                },
                "neural_medication_aware_hist_gradient_boosting": {
                    "feature_lists": neural_medication_features,
                    "X_train": X_neural_train,
                    "X_validation": X_neural_validation,
                    "X_test": X_neural_test,
                    "X_train_final": X_neural_train_final,
                },
                "longitudinal_neural_medication_trajectory_logistic": {
                    "feature_lists": trajectory_neural_medication_features,
                    "X_train": X_traj_neural_train,
                    "X_validation": X_traj_neural_validation,
                    "X_test": X_traj_neural_test,
                    "X_train_final": X_traj_neural_train_final,
                },
                "longitudinal_neural_medication_trajectory_hist_gradient_boosting": {
                    "feature_lists": trajectory_neural_medication_features,
                    "X_train": X_traj_neural_train,
                    "X_validation": X_traj_neural_validation,
                    "X_test": X_traj_neural_test,
                    "X_train_final": X_traj_neural_train_final,
                },
            }
        )

    modeling_runtime_cfg = cfg.get("modeling", {}) or {}
    enabled_families = [str(x).strip() for x in modeling_runtime_cfg.get("enabled_model_families", []) if str(x).strip()]
    if enabled_families:
        allowed_families = set(enabled_families)
        model_specs = {name: spec for name, spec in model_specs.items() if _model_family_from_name(name) in allowed_families}
    enabled_arms = [str(x).strip() for x in modeling_runtime_cfg.get("enabled_model_arms", []) if str(x).strip()]
    if enabled_arms:
        allowed_arms = set(enabled_arms)
        model_specs = {name: spec for name, spec in model_specs.items() if name in allowed_arms}
    if not model_specs:
        raise ValueError("No model arms remain after applying modeling.enabled_model_families/enabled_model_arms filters.")
    logger.info("Enabled Step 3 model arms: %s", sorted(model_specs))
    step3_bar.update("Define clinical, structured-medication, LLM-enhanced, and optional neural model arms")

    progress_step("Step 3/7: Tuning models on validation set and refitting final models", enabled=progress_enabled, logger=logger)
    fitted = {}
    validation_models = {}
    probabilities: dict[str, np.ndarray] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    tuning_frames: list[pd.DataFrame] = []
    selected_model_cfgs: dict[str, dict[str, Any]] = {}
    refit_train_validation = bool(
        cfg.get("modeling", {}).get("hyperparameter_tuning", {}).get("refit_train_validation_for_final_test", True)
    )
    for name, spec in progress_iter(model_specs.items(), enabled=progress_enabled, desc="Model arms", total=len(model_specs), unit="model"):
        logger.info("Selecting model configuration on validation set: %s", name)
        selected_cfg, tuning_df, validation_prob, train_only_model = _tune_model_on_validation(
            name,
            spec["feature_lists"],
            spec["X_train"],
            y_train,
            spec["X_validation"],
            y_validation,
            cfg,
            threshold=threshold,
            progress_enabled=progress_enabled,
        )
        selected_model_cfgs[name] = selected_cfg
        tuning_frames.append(tuning_df)
        validation_probabilities[name] = validation_prob
        validation_models[name] = train_only_model

        family = _model_family_from_name(name)
        final_X_train = spec["X_train_final"] if refit_train_validation else spec["X_train"]
        final_y_train = y_train_final if refit_train_validation else y_train
        logger.info(
            "Fitting final model for held-out test: %s | refit_train_validation=%s | rows=%s",
            name,
            refit_train_validation,
            len(final_X_train),
        )
        final_model = _build_pipeline_for_family(family, spec["feature_lists"], selected_cfg)
        fitted[name] = fit_pipeline(name, final_model, final_X_train.copy(), final_y_train.copy(), use_balanced_sample_weight=use_hgb_weights)
        probabilities[name] = predict_proba_positive(fitted[name], spec["X_test"].copy())
        logger.info("Finished final model: %s", name)
    step3_bar.update("Tune validation models, refit final models, and generate test predictions")

    models = {name: fitted[name] for name in model_specs}
    validation_metrics_df = build_metrics_table(validation_probabilities, y_validation, threshold=threshold)
    _round_csv(validation_metrics_df, output_file(output_dir, "step3e_validation_model_performance.csv"), decimal_places)
    tuning_results_df = pd.concat(tuning_frames, ignore_index=True) if tuning_frames else pd.DataFrame()
    _round_csv(tuning_results_df, output_file(output_dir, "step3f_validation_hyperparameter_tuning_results.csv"), decimal_places)
    step3_bar.update("Write validation performance and tuning audit outputs")
    step3_bar.close()



    ctx.validation_size = validation_size
    ctx.test_size = test_size
    ctx.train_idx = train_idx
    ctx.validation_idx = validation_idx
    ctx.test_idx = test_idx
    ctx.train_final_idx = train_final_idx
    ctx.y_train = y_train
    ctx.y_validation = y_validation
    ctx.y_test = y_test
    ctx.y_train_final = y_train_final
    ctx.model_specs = model_specs
    ctx.fitted = fitted
    ctx.validation_models = validation_models
    ctx.probabilities = probabilities
    ctx.validation_probabilities = validation_probabilities
    ctx.tuning_frames = tuning_frames
    ctx.selected_model_cfgs = selected_model_cfgs
    ctx.models = models
    ctx.validation_metrics_df = validation_metrics_df
    ctx.tuning_results_df = tuning_results_df
    return True
