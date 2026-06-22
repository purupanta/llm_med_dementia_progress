"""
Project: dementia_progression
File: src/pipeline_step4_evaluation.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Evaluates held-out model performance, bootstrap intervals, incremental value, calibration, decision-curve, permutation-importance, grouped-importance, and temporal-validation outputs.

Design:
    This module contains the executable logic for Step 4/7 held-out evaluation and validation analyses. It is intentionally
    separated from the main orchestrator so that GitHub users can inspect,
    test, and maintain one numbered pipeline stage at a time.
"""

from __future__ import annotations

from src.pipeline_context import PipelineContext
from src.pipeline_common import *  # Reuse the validated v1.49 helper surface.


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 4/7 held-out evaluation and validation analyses and store downstream state in ``ctx``.

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
    y_test = ctx.y_test
    test_idx = ctx.test_idx
    train_idx = ctx.train_idx
    validation_idx = ctx.validation_idx
    train_final_idx = ctx.train_final_idx
    validation_size = ctx.validation_size
    test_size = ctx.test_size
    model_specs = ctx.model_specs
    fitted = ctx.fitted
    models = ctx.models
    probabilities = ctx.probabilities
    validation_models = ctx.validation_models
    validation_probabilities = ctx.validation_probabilities
    validation_metrics_df = ctx.validation_metrics_df
    y_validation = ctx.y_validation
    clinical_X = ctx.clinical_X
    med_X = ctx.med_X
    llm_X = ctx.llm_X
    clinical_features = ctx.clinical_features
    medication_features = ctx.medication_features
    llm_medication_features = ctx.llm_medication_features
    neural_medication_features = ctx.neural_medication_features
    trajectory_llm_X = getattr(ctx, "trajectory_llm_X", llm_X)
    trajectory_llm_medication_features = getattr(ctx, "trajectory_llm_medication_features", llm_medication_features)

    progress_step("Step 4/7: Evaluating held-out test model performance", enabled=progress_enabled, logger=logger)
    step4_bar = progress_bar(total=6, desc="Step 4/7 evaluation and validation outputs", enabled=progress_enabled, unit="task", logger=logger)
    metrics_df = build_metrics_table(probabilities, y_test, threshold=threshold)
    _round_csv(metrics_df, output_file(output_dir, "step3a_heldout_test_model_performance.csv"), decimal_places)
    step4_bar.update("Evaluate held-out test performance and write main metrics")

    validation_cfg = cfg.get("validation", {})
    preferred_final_model = str(cfg.get("modeling", {}).get("final_predictive_model_preference", "auto_best_validation_auc"))
    selection_rule = preferred_final_model
    if preferred_final_model.lower() in {"auto", "auto_best_validation_auc", "best_validation_auc"}:
        validation_ranked = validation_metrics_df.sort_values(
            ["roc_auc", "average_precision", "brier_score"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        final_model_name = str(validation_ranked.iloc[0]["model"])
        selection_rule = "auto_best_validation_auc"
    elif preferred_final_model in model_specs:
        final_model_name = preferred_final_model
    else:
        fallback_ranked = validation_metrics_df.sort_values(
            ["roc_auc", "average_precision", "brier_score"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        final_model_name = str(fallback_ranked.iloc[0]["model"])
        selection_rule = f"configured_model_unavailable_fallback_to_auto_best_validation_auc:{preferred_final_model}"
    final_model_spec = model_specs[final_model_name]
    final_risk_col = f"risk_{final_model_name}"
    heldout_row = metrics_df.loc[metrics_df["model"].eq(final_model_name)].iloc[0].to_dict()
    validation_row = validation_metrics_df.loc[validation_metrics_df["model"].eq(final_model_name)].iloc[0].to_dict()
    final_model_selection_df = pd.DataFrame([{
        "selection_rule": selection_rule,
        "configured_preference": preferred_final_model,
        "selected_final_model": final_model_name,
        "validation_roc_auc": validation_row.get("roc_auc", np.nan),
        "validation_average_precision": validation_row.get("average_precision", np.nan),
        "validation_brier_score": validation_row.get("brier_score", np.nan),
        "heldout_roc_auc": heldout_row.get("roc_auc", np.nan),
        "heldout_average_precision": heldout_row.get("average_precision", np.nan),
        "heldout_brier_score": heldout_row.get("brier_score", np.nan),
        "interpretation": "Final predictive model selected using validation-set performance before held-out test evaluation." if selection_rule == "auto_best_validation_auc" else "Final predictive model selected from configured model preference."
    }])
    _round_csv(final_model_selection_df, output_file(output_dir, "step4m_validation_selected_final_model.csv"), decimal_places)
    n_bootstrap = int(validation_cfg.get("n_bootstrap", 200))
    if n_bootstrap > 0:
        logger.info("Running bootstrap confidence intervals: n_bootstrap=%s", n_bootstrap)
        ci_df = _bootstrap_metric_ci(
            probabilities,
            y_test,
            n_bootstrap=n_bootstrap,
            random_state=int(cfg.get("random_state", 42)),
            threshold=threshold,
            progress_enabled=progress_enabled,
        )
        _round_csv(ci_df, output_file(output_dir, "step3b_heldout_test_model_performance_bootstrap_ci.csv"), decimal_places)
        bootstrap_delta_pairs = [
            ("logistic_regression: structured medication-aware minus clinical-only", "structured_medication_aware_logistic", "clinical_only_logistic"),
            ("logistic_regression: LLM-enhanced medication-aware minus clinical-only", "llm_enhanced_medication_aware_logistic", "clinical_only_logistic"),
            ("logistic_regression: LLM-enhanced minus structured medication-aware", "llm_enhanced_medication_aware_logistic", "structured_medication_aware_logistic"),
            ("hist_gradient_boosting: structured medication-aware minus clinical-only", "structured_medication_aware_hist_gradient_boosting", "clinical_only_hist_gradient_boosting"),
            ("hist_gradient_boosting: LLM-enhanced medication-aware minus clinical-only", "llm_enhanced_medication_aware_hist_gradient_boosting", "clinical_only_hist_gradient_boosting"),
            ("hist_gradient_boosting: LLM-enhanced minus structured medication-aware", "llm_enhanced_medication_aware_hist_gradient_boosting", "structured_medication_aware_hist_gradient_boosting"),
        ]
        bootstrap_delta_pairs = [(label, enhanced, reference) for label, enhanced, reference in bootstrap_delta_pairs if enhanced in probabilities and reference in probabilities]
        delta_ci_df = _bootstrap_delta_ci(
            probabilities,
            y_test,
            pairs=bootstrap_delta_pairs,
            n_bootstrap=n_bootstrap,
            random_state=int(cfg.get("random_state", 42)),
            threshold=threshold,
            progress_enabled=progress_enabled,
        )
        _round_csv(delta_ci_df, output_file(output_dir, "step3d_medication_incremental_value_bootstrap_ci.csv"), decimal_places)
        logger.info("Finished bootstrap confidence intervals")
        step4_bar.update("Compute bootstrap confidence intervals and delta confidence intervals")
    else:
        step4_bar.update("Skip bootstrap confidence intervals because n_bootstrap <= 0")

    def _metric(name: str, metric: str) -> float:
        return float(metrics_df.loc[metrics_df["model"].eq(name), metric].iloc[0])

    delta_rows = []
    delta_specs = [
        ("logistic_regression", "structured medication-aware minus clinical-only", "structured_medication_aware_logistic", "clinical_only_logistic"),
        ("logistic_regression", "LLM-enhanced medication-aware minus clinical-only", "llm_enhanced_medication_aware_logistic", "clinical_only_logistic"),
        ("logistic_regression", "LLM-enhanced minus structured medication-aware", "llm_enhanced_medication_aware_logistic", "structured_medication_aware_logistic"),
        ("hist_gradient_boosting", "structured medication-aware minus clinical-only", "structured_medication_aware_hist_gradient_boosting", "clinical_only_hist_gradient_boosting"),
        ("hist_gradient_boosting", "LLM-enhanced medication-aware minus clinical-only", "llm_enhanced_medication_aware_hist_gradient_boosting", "clinical_only_hist_gradient_boosting"),
        ("hist_gradient_boosting", "LLM-enhanced minus structured medication-aware", "llm_enhanced_medication_aware_hist_gradient_boosting", "structured_medication_aware_hist_gradient_boosting"),
    ]
    if "longitudinal_llm_medication_trajectory_hist_gradient_boosting" in model_specs:
        delta_specs.extend(
            [
                ("logistic_regression", "longitudinal clinical trajectory minus clinical-only", "longitudinal_clinical_trajectory_logistic", "clinical_only_logistic"),
                ("logistic_regression", "longitudinal medication trajectory minus structured medication-aware", "longitudinal_medication_trajectory_logistic", "structured_medication_aware_logistic"),
                ("logistic_regression", "longitudinal LLM medication trajectory minus LLM-enhanced snapshot", "longitudinal_llm_medication_trajectory_logistic", "llm_enhanced_medication_aware_logistic"),
                ("hist_gradient_boosting", "longitudinal clinical trajectory minus clinical-only", "longitudinal_clinical_trajectory_hist_gradient_boosting", "clinical_only_hist_gradient_boosting"),
                ("hist_gradient_boosting", "longitudinal medication trajectory minus structured medication-aware", "longitudinal_medication_trajectory_hist_gradient_boosting", "structured_medication_aware_hist_gradient_boosting"),
                ("hist_gradient_boosting", "longitudinal LLM medication trajectory minus LLM-enhanced snapshot", "longitudinal_llm_medication_trajectory_hist_gradient_boosting", "llm_enhanced_medication_aware_hist_gradient_boosting"),
            ]
        )
    if "neural_medication_aware_logistic" in model_specs:
        delta_specs.extend(
            [
                ("logistic_regression", "ClinicalBERT/SapBERT neural medication-aware minus clinical-only", "neural_medication_aware_logistic", "clinical_only_logistic"),
                ("logistic_regression", "ClinicalBERT/SapBERT neural medication-aware minus LLM-enhanced", "neural_medication_aware_logistic", "llm_enhanced_medication_aware_logistic"),
                ("hist_gradient_boosting", "ClinicalBERT/SapBERT neural medication-aware minus clinical-only", "neural_medication_aware_hist_gradient_boosting", "clinical_only_hist_gradient_boosting"),
                ("hist_gradient_boosting", "ClinicalBERT/SapBERT neural medication-aware minus LLM-enhanced", "neural_medication_aware_hist_gradient_boosting", "llm_enhanced_medication_aware_hist_gradient_boosting"),
            ]
        )
    for family, label, med_name, reference_name in delta_specs:
        if med_name not in set(metrics_df["model"]) or reference_name not in set(metrics_df["model"]):
            continue
        delta_rows.append(
            {
                "comparison": f"{family}: {label}",
                "model_family": family,
                "enhanced_model": med_name,
                "reference_model": reference_name,
                "delta_roc_auc": _metric(med_name, "roc_auc") - _metric(reference_name, "roc_auc"),
                "delta_average_precision": _metric(med_name, "average_precision") - _metric(reference_name, "average_precision"),
                "delta_brier_score": _metric(med_name, "brier_score") - _metric(reference_name, "brier_score"),
                "delta_balanced_accuracy_at_0_5": _metric(med_name, "balanced_accuracy_at_0_5") - _metric(reference_name, "balanced_accuracy_at_0_5"),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    _round_csv(delta_df, output_file(output_dir, "step3c_medication_incremental_value_vs_clinical_only.csv"), decimal_places)
    llm_delta_df = delta_df[delta_df["comparison"].str.contains("LLM-enhanced minus structured", regex=False)].copy()
    _round_csv(llm_delta_df, output_file(output_dir, "step3g_llm_incremental_value_vs_structured_medication.csv"), decimal_places)
    neural_delta_df = delta_df[delta_df["comparison"].str.contains("ClinicalBERT/SapBERT", regex=False)].copy()
    if not neural_delta_df.empty:
        _round_csv(neural_delta_df, output_file(output_dir, "step3h_neural_representation_incremental_value.csv"), decimal_places)
    step4_bar.update("Write medication incremental-value, LLM incremental-value, and neural incremental-value tables")

    progress_step("Step 4/7: Generating calibration and decision-curve outputs", enabled=progress_enabled, logger=logger)
    calibration_df = _calibration_by_decile(probabilities, y_test, n_bins=int(validation_cfg.get("calibration_bins", 10)))
    _round_csv(calibration_df, output_file(output_dir, "step4d_heldout_test_calibration_by_decile.csv"), decimal_places)

    thresholds = [float(x) for x in validation_cfg.get("decision_curve_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5])]
    dca = _decision_curve(probabilities, y_test, thresholds)
    _round_csv(dca, output_file(output_dir, "step4e_heldout_test_decision_curve_analysis.csv"), decimal_places)
    step4_bar.update("Generate held-out calibration and decision-curve outputs")

    cal_cfg = validation_cfg.get("calibration", {})
    if bool(cal_cfg.get("enabled", True)):
        logger.info("Running validation-set isotonic/sigmoid calibration for final model: %s", final_model_name)
        cal_outputs = _fit_validation_calibrators(
            validation_models[final_model_name],
            final_model_spec["X_validation"],
            y_validation,
            final_model_spec["X_test"],
            y_test,
            thresholds,
            threshold=threshold,
        )
        _round_csv(cal_outputs["performance"], output_file(output_dir, "step4h_validation_calibrated_model_performance.csv"), decimal_places)
        cal_predictions = modeling_df.iloc[test_idx][[cfg["id_col"], cfg["visit_number_col"], "next_visit_dementia"]].copy().reset_index(drop=True)
        cal_predictions = pd.concat([cal_predictions, cal_outputs["predictions"].drop(columns=["next_visit_dementia"], errors="ignore")], axis=1)
        _round_csv(cal_predictions, output_file(output_dir, "step4i_validation_calibrated_heldout_test_risk_predictions.csv"), decimal_places)
        _round_csv(cal_outputs["decision_curve"], output_file(output_dir, "step4j_validation_calibrated_decision_curve_analysis.csv"), decimal_places)
        step4_bar.update("Run validation-set calibration sensitivity analysis")
    else:
        step4_bar.update("Skip calibration sensitivity analysis because it is disabled")

    temporal_cfg = validation_cfg.get("temporal_validation", {})
    if bool(temporal_cfg.get("enabled", True)):
        logger.info("Running temporal validation analysis")
        temporal_df = _temporal_validation(
            cfg, clinical_X, med_X, llm_X, y, modeling_df, clinical_features, medication_features, llm_medication_features,
            trajectory_X=trajectory_llm_X, trajectory_features=trajectory_llm_medication_features, threshold=threshold
        )
        _round_csv(temporal_df, output_file(output_dir, "step4l_temporal_validation_model_performance.csv"), decimal_places)
        step4_bar.update("Run temporal-validation sensitivity analysis")
    else:
        step4_bar.update("Skip temporal-validation sensitivity analysis because it is disabled")

    publication_table = metrics_df[
        [
            "model",
            "n",
            "prevalence",
            "roc_auc",
            "average_precision",
            "brier_score",
            "accuracy_at_0_5",
            "balanced_accuracy_at_0_5",
            "sensitivity_at_0_5",
            "specificity_at_0_5",
            "precision_at_0_5",
            "f1_at_0_5",
        ]
    ].copy()
    publication_table["feature_set"] = publication_table["model"].map(
        {
            "clinical_only_logistic": "clinical_only_structured_patient_state",
            "structured_medication_aware_logistic": "clinical_plus_structured_medication_state",
            "llm_enhanced_medication_aware_logistic": "clinical_plus_structured_and_llm_medication_state",
            "clinical_only_hist_gradient_boosting": "clinical_only_structured_patient_state",
            "structured_medication_aware_hist_gradient_boosting": "clinical_plus_structured_medication_state",
            "llm_enhanced_medication_aware_hist_gradient_boosting": "clinical_plus_structured_and_llm_medication_state",
            "neural_medication_aware_logistic": "clinical_plus_structured_llm_and_bioclinicalbert_sapbert_medication_state",
            "neural_medication_aware_hist_gradient_boosting": "clinical_plus_structured_llm_and_bioclinicalbert_sapbert_medication_state",
            "longitudinal_clinical_trajectory_logistic": "clinical_plus_longitudinal_trajectory_state",
            "longitudinal_medication_trajectory_logistic": "clinical_plus_structured_medication_longitudinal_trajectory_state",
            "longitudinal_llm_medication_trajectory_logistic": "clinical_plus_llm_medication_longitudinal_trajectory_state",
            "longitudinal_clinical_trajectory_hist_gradient_boosting": "clinical_plus_longitudinal_trajectory_state",
            "longitudinal_medication_trajectory_hist_gradient_boosting": "clinical_plus_structured_medication_longitudinal_trajectory_state",
            "longitudinal_llm_medication_trajectory_hist_gradient_boosting": "clinical_plus_llm_medication_longitudinal_trajectory_state",
            "longitudinal_neural_medication_trajectory_logistic": "clinical_plus_neural_medication_longitudinal_trajectory_state",
            "longitudinal_neural_medication_trajectory_hist_gradient_boosting": "clinical_plus_neural_medication_longitudinal_trajectory_state",
        }
    )
    publication_table["model_family"] = publication_table["model"].map(
        {
            "clinical_only_logistic": "balanced_logistic_regression",
            "structured_medication_aware_logistic": "balanced_logistic_regression",
            "llm_enhanced_medication_aware_logistic": "balanced_logistic_regression",
            "clinical_only_hist_gradient_boosting": "hist_gradient_boosting",
            "structured_medication_aware_hist_gradient_boosting": "hist_gradient_boosting",
            "llm_enhanced_medication_aware_hist_gradient_boosting": "hist_gradient_boosting",
            "neural_medication_aware_logistic": "balanced_logistic_regression",
            "neural_medication_aware_hist_gradient_boosting": "hist_gradient_boosting",
            "longitudinal_clinical_trajectory_logistic": "balanced_logistic_regression",
            "longitudinal_medication_trajectory_logistic": "balanced_logistic_regression",
            "longitudinal_llm_medication_trajectory_logistic": "balanced_logistic_regression",
            "longitudinal_neural_medication_trajectory_logistic": "balanced_logistic_regression",
            "longitudinal_clinical_trajectory_hist_gradient_boosting": "hist_gradient_boosting",
            "longitudinal_medication_trajectory_hist_gradient_boosting": "hist_gradient_boosting",
            "longitudinal_llm_medication_trajectory_hist_gradient_boosting": "hist_gradient_boosting",
            "longitudinal_neural_medication_trajectory_hist_gradient_boosting": "hist_gradient_boosting",
        }
    )
    publication_table = publication_table[
        ["model", "model_family", "feature_set"] + [c for c in publication_table.columns if c not in {"model", "model_family", "feature_set"}]
    ]
    _round_csv(publication_table, output_file(output_dir, "step5a_publication_model_performance_table.csv"), decimal_places)
    step4_bar.close()



    ctx.metrics_df = metrics_df
    ctx.validation_cfg = validation_cfg
    ctx.preferred_final_model = preferred_final_model
    ctx.final_model_name = final_model_name
    ctx.final_model_spec = final_model_spec
    ctx.final_risk_col = final_risk_col
    ctx.ci_df = ci_df
    ctx.delta_df = delta_df
    ctx.delta_ci_df = delta_ci_df
    ctx.calibration_df = calibration_df
    ctx.publication_table = publication_table
    return True
