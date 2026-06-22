"""
Project: dementia_progression
File: src/pipeline_step5_predictions.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Writes held-out predicted-risk tables, clinical-versus-medication risk deltas, and top reclassification examples used for manuscript interpretation.

Design:
    This module contains the executable logic for Step 5/7 predicted-risk and reclassification outputs. It is intentionally
    separated from the main orchestrator so that GitHub users can inspect,
    test, and maintain one numbered pipeline stage at a time.
"""

from __future__ import annotations

from src.pipeline_context import PipelineContext
from src.pipeline_common import *  # Reuse the validated v1.49 helper surface.


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 5/7 predicted-risk and reclassification outputs and store downstream state in ``ctx``.

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
    test_idx = ctx.test_idx
    probabilities = ctx.probabilities
    metrics_df = ctx.metrics_df
    final_risk_col = ctx.final_risk_col
    validation_cfg = ctx.validation_cfg
    y_test = ctx.y_test
    fitted = ctx.fitted
    final_model_name = ctx.final_model_name
    final_model_spec = ctx.final_model_spec

    progress_step("Step 5/7: Writing held-out test predicted risk and reclassification outputs", enabled=progress_enabled, logger=logger)
    step5_bar = progress_bar(total=5, desc="Step 5/7 prediction and interpretation outputs", enabled=progress_enabled, unit="task", logger=logger)
    pred_cols = [
        cfg["id_col"],
        cfg["visit_number_col"],
        cfg["visit_year_col"],
        cfg["visit_month_col"],
        cfg["visit_day_col"],
        "next_diagnosis",
        "next_visit_dementia",
        "SEX",
        "HISPANIC",
        "RACE",
        "NACCAGE",
        "NACCDAYS",
        "prior_mci_visit_count",
        "days_since_prior_mci_visit",
        "days_since_first_mci_visit",
        "medication_count",
        "medication_psychotropic_count",
        "medication_neuro_count",
        "medication_cardiometabolic_count",
    ] + llm_medication_feature_columns(modeling_df) + neural_text_feature_columns(modeling_df)
    pred_cols = [c for c in pred_cols if c in modeling_df.columns]
    predictions = modeling_df.iloc[test_idx][pred_cols].copy().reset_index(drop=True)
    for name, prob in probabilities.items():
        predictions[f"risk_{name}"] = prob
        predictions[f"high_risk_{name}"] = _classification_labels(prob, threshold)
    def _add_delta(out_col: str, enhanced_col: str, reference_col: str) -> None:
        if enhanced_col in predictions.columns and reference_col in predictions.columns:
            predictions[out_col] = predictions[enhanced_col] - predictions[reference_col]

    def _add_reclassification(out_col: str, enhanced_high_col: str, reference_high_col: str) -> None:
        if enhanced_high_col in predictions.columns and reference_high_col in predictions.columns:
            predictions[out_col] = ((predictions[reference_high_col] == 0) & (predictions[enhanced_high_col] == 1)).astype(int)

    _add_delta("risk_delta_structured_medication_logistic_minus_clinical_logistic", "risk_structured_medication_aware_logistic", "risk_clinical_only_logistic")
    _add_delta("risk_delta_structured_medication_hgb_minus_clinical_hgb", "risk_structured_medication_aware_hist_gradient_boosting", "risk_clinical_only_hist_gradient_boosting")
    _add_delta("risk_delta_llm_hgb_minus_structured_medication_hgb", "risk_llm_enhanced_medication_aware_hist_gradient_boosting", "risk_structured_medication_aware_hist_gradient_boosting")
    _add_delta("risk_delta_llm_logistic_minus_structured_medication_logistic", "risk_llm_enhanced_medication_aware_logistic", "risk_structured_medication_aware_logistic")
    _add_delta("risk_delta_neural_hgb_minus_llm_hgb", "risk_neural_medication_aware_hist_gradient_boosting", "risk_llm_enhanced_medication_aware_hist_gradient_boosting")
    _add_delta("risk_delta_neural_hgb_minus_clinical_hgb", "risk_neural_medication_aware_hist_gradient_boosting", "risk_clinical_only_hist_gradient_boosting")
    _add_reclassification("reclassified_high_risk_by_structured_medication_logistic", "high_risk_structured_medication_aware_logistic", "high_risk_clinical_only_logistic")
    _add_reclassification("reclassified_high_risk_by_structured_medication_hgb", "high_risk_structured_medication_aware_hist_gradient_boosting", "high_risk_clinical_only_hist_gradient_boosting")
    _add_reclassification("reclassified_high_risk_by_llm_hgb_vs_structured_medication_hgb", "high_risk_llm_enhanced_medication_aware_hist_gradient_boosting", "high_risk_structured_medication_aware_hist_gradient_boosting")
    _round_csv(predictions, output_file(output_dir, "step4a_heldout_test_predicted_risk_scores.csv"), decimal_places)
    step5_bar.update("Write held-out predicted risk scores and high-risk indicators")

    subgroup_df = _subgroup_metrics(
        predictions,
        outcome_col="next_visit_dementia",
        risk_col=final_risk_col,
        group_cols=["SEX", "HISPANIC", "RACE"],
        min_n=int(validation_cfg.get("subgroup_min_n", 100)),
        threshold=threshold,
    )
    if not subgroup_df.empty:
        _round_csv(subgroup_df, output_file(output_dir, "step4f_heldout_test_subgroup_performance_final_model.csv"), decimal_places)
        step5_bar.update("Write subgroup performance outputs")
    else:
        step5_bar.update("Skip subgroup output because no subgroup met the minimum size")

    perm_cfg = validation_cfg.get("permutation_importance", {})
    if bool(perm_cfg.get("enabled", True)):
        logger.info("Running permutation importance for final model: %s", final_model_name)
        perm_df = _run_permutation_importance(
            fitted[final_model_name],
            final_model_spec["X_test"],
            y_test,
            max_rows=int(perm_cfg.get("max_test_rows", 2500)),
            n_repeats=int(perm_cfg.get("n_repeats", 2)),
            random_state=int(cfg.get("random_state", 42)),
        )
        _round_csv(perm_df, output_file(output_dir, "step4g_heldout_test_permutation_importance_final_model.csv"), decimal_places)
        step5_bar.update("Compute and write individual permutation importance")
    else:
        step5_bar.update("Skip individual permutation importance because it is disabled")

    grouped_perm_cfg = validation_cfg.get("grouped_permutation_importance", {})
    if bool(grouped_perm_cfg.get("enabled", True)):
        logger.info("Running grouped permutation importance for final model: %s", final_model_name)
        grouped_perm_df = _grouped_permutation_importance(
            fitted[final_model_name],
            final_model_spec["X_test"],
            y_test,
            _default_feature_groups(modeling_df),
            max_rows=int(grouped_perm_cfg.get("max_test_rows", 1200)),
            n_repeats=int(grouped_perm_cfg.get("n_repeats", 3)),
            random_state=int(cfg.get("random_state", 42)),
            progress_enabled=progress_enabled,
        )
        _round_csv(grouped_perm_df, output_file(output_dir, "step4k_heldout_test_grouped_permutation_importance.csv"), decimal_places)
        step5_bar.update("Compute and write grouped permutation importance")
    else:
        step5_bar.update("Skip grouped permutation importance because it is disabled")

    reclass_cols = [
        cfg["id_col"],
        cfg["visit_number_col"],
        "next_visit_dementia",
        "NACCAGE",
        "NACCDAYS",
        "prior_mci_visit_count",
        "days_since_prior_mci_visit",
        "days_since_first_mci_visit",
        "medication_count",
        "medication_psychotropic_count",
        "medication_neuro_count",
        "medication_cardiometabolic_count",
        "risk_clinical_only_logistic",
        "risk_structured_medication_aware_logistic",
        "risk_delta_structured_medication_logistic_minus_clinical_logistic",
        "high_risk_clinical_only_logistic",
        "high_risk_structured_medication_aware_logistic",
        "reclassified_high_risk_by_structured_medication_logistic",
        "risk_clinical_only_hist_gradient_boosting",
        "risk_structured_medication_aware_hist_gradient_boosting",
        "risk_delta_structured_medication_hgb_minus_clinical_hgb",
        "risk_llm_enhanced_medication_aware_hist_gradient_boosting",
        "risk_delta_llm_hgb_minus_structured_medication_hgb",
        "high_risk_llm_enhanced_medication_aware_hist_gradient_boosting",
        "reclassified_high_risk_by_llm_hgb_vs_structured_medication_hgb",
        "risk_neural_medication_aware_hist_gradient_boosting",
        "risk_delta_neural_hgb_minus_llm_hgb",
        "risk_delta_neural_hgb_minus_clinical_hgb",
        "high_risk_neural_medication_aware_hist_gradient_boosting",
        "high_risk_clinical_only_hist_gradient_boosting",
        "high_risk_structured_medication_aware_hist_gradient_boosting",
        "reclassified_high_risk_by_structured_medication_hgb",
        "reclassified_high_risk_by_llm_hgb_vs_structured_medication_hgb",
    ]
    reclass_cols = [c for c in reclass_cols if c in predictions.columns]
    reclass = predictions[reclass_cols].copy()
    _round_csv(reclass, output_file(output_dir, "step4b_heldout_test_clinical_vs_medication_reclassification.csv"), decimal_places)
    sort_candidates = [
        "reclassified_high_risk_by_structured_medication_hgb",
        "risk_delta_structured_medication_hgb_minus_clinical_hgb",
        "reclassified_high_risk_by_structured_medication_logistic",
        "risk_delta_structured_medication_logistic_minus_clinical_logistic",
    ]
    sort_cols = [c for c in sort_candidates if c in reclass.columns]
    if sort_cols:
        reclass_examples = reclass.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(100)
    else:
        reclass_examples = reclass.head(100)
    _round_csv(reclass_examples, output_file(output_dir, "step4c_heldout_test_top_reclassified_examples.csv"), decimal_places)
    step5_bar.update("Write clinical-versus-medication reclassification tables")
    step5_bar.close()

    # Neuropathology anchoring: merge held-out predictions with NP variables; never use these as predictors.


    ctx.predictions = predictions
    ctx.reclass = reclass
    ctx.reclass_examples = reclass_examples
    return True
