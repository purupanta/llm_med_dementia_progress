"""
Project: dementia_progression
File: src/pipeline_step7_reporting.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Writes run metadata, publication-ready narrative summaries, table indices,
    file synopses, final reports, and the optional trained-model bundle.

Design:
    This module contains the executable logic for Step 7/7 metadata, documentation,
    and GitHub-ready output packaging. It is separated from the orchestrator so
    final reporting can be maintained independently from model fitting.
"""

from __future__ import annotations

from src.pipeline_context import PipelineContext
from src.pipeline_common import *


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 7/7 final reporting and store no additional downstream state."""
    cfg = ctx.cfg
    input_csv = ctx.input_csv
    output_dir = ctx.output_dir
    logger = ctx.logger
    decimal_places = ctx.decimal_places
    progress_enabled = ctx.progress_enabled
    threshold = ctx.threshold
    record_selection = ctx.record_selection
    step1_raw_rows_loaded = ctx.step1_raw_rows_loaded
    df = ctx.df
    cohort_summary = ctx.cohort_summary
    validation_size = ctx.validation_size
    test_size = ctx.test_size
    train_idx = ctx.train_idx
    validation_idx = ctx.validation_idx
    test_idx = ctx.test_idx
    train_final_idx = ctx.train_final_idx
    groups = ctx.groups
    clinical_features = ctx.clinical_features
    medication_features = ctx.medication_features
    llm_medication_features = ctx.llm_medication_features
    neural_medication_features = ctx.neural_medication_features
    modeling_df = ctx.modeling_df
    models = ctx.models
    fitted = ctx.fitted
    metrics_df = ctx.metrics_df
    delta_df = ctx.delta_df
    llm_audit = ctx.llm_audit
    neural_audit = ctx.neural_audit
    best_model = ctx.best_model
    best_validation_model = ctx.best_validation_model
    selected_tuning_rows = ctx.selected_tuning_rows
    final_model_name = ctx.final_model_name
    np_cols_present = ctx.np_cols_present

    progress_step("Step 7/7: Writing metadata, manuscript tables, documentation indices, and model bundle", enabled=progress_enabled, logger=logger)
    step7_bar = progress_bar(total=5, desc="Step 7/7 final metadata and packaging", enabled=progress_enabled, unit="task", logger=logger)
    step7_bar.update("Select best validation and held-out test models")
    metadata = {
        "project_name": cfg.get("project_name"),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "input_record_selection": record_selection,
        "runtime_environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        },
        "raw_rows_loaded_before_step1_required_row_filter": int(step1_raw_rows_loaded),
        "rows_after_step1_final_source_file": int(len(df)),
        "raw_unique_participants_after_step1_final_source_file": int(df[cfg["id_col"]].nunique()),
        "cohort": cohort_summary,
        "split_design": "participant_level_train_validation_test",
        "split_proportions_requested": {
            "training": float(1.0 - validation_size - test_size),
            "validation": float(validation_size),
            "held_out_test": float(test_size),
        },
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "test_rows": int(len(test_idx)),
        "final_model_fit_rows_train_plus_validation": int(len(train_final_idx)),
        "train_participants": int(groups.iloc[train_idx].nunique()),
        "validation_participants": int(groups.iloc[validation_idx].nunique()),
        "test_participants": int(groups.iloc[test_idx].nunique()),
        "final_model_fit_participants_train_plus_validation": int(groups.iloc[train_final_idx].nunique()),
        "participant_leakage_check": {
            "train_validation_overlap": int(len(set(groups.iloc[train_idx]) & set(groups.iloc[validation_idx]))),
            "train_test_overlap": int(len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))),
            "validation_test_overlap": int(len(set(groups.iloc[validation_idx]) & set(groups.iloc[test_idx]))),
        },
        "threshold": threshold,
        "n_bootstrap_ci": int(cfg.get("validation", {}).get("n_bootstrap", 200)),
        "calibration_bins": int(cfg.get("validation", {}).get("calibration_bins", 10)),
        "feature_counts": {
            "clinical_only": {k: len(v) for k, v in clinical_features.items()},
            "structured_medication_aware": {k: len(v) for k, v in medication_features.items()},
            "llm_enhanced_medication_aware": {k: len(v) for k, v in llm_medication_features.items()},
            "bioclinicalbert_sapbert_neural_medication_aware": {k: len(v) for k, v in neural_medication_features.items()},
            "structured_medication_state_columns": len(structured_medication_feature_columns(modeling_df)),
            "llm_medication_state_columns": len(llm_medication_feature_columns(modeling_df)),
            "neural_text_representation_columns": len(neural_text_feature_columns(modeling_df)),
            "medication_state_columns": len(medication_feature_columns(modeling_df, include_llm=True, include_neural_text=True)),
            "neuropathology_columns_loaded_but_excluded_from_prediction": len(np_cols_present),
        },
        "model_set": list(models.keys()),
        "selected_best_model_by_validation_roc_auc": best_validation_model,
        "best_model_by_final_held_out_test_roc_auc": best_model,
        "validation_selected_hyperparameters": selected_tuning_rows,
        "primary_interpretable_model": cfg.get("modeling", {}).get("primary_interpretable_model", "neural_medication_aware_logistic"),
        "final_predictive_model_preference": cfg.get("modeling", {}).get(
            "final_predictive_model_preference", "neural_medication_aware_hist_gradient_boosting"
        ),
        "final_model_used_for_subgroup_importance_and_neuropathology": final_model_name,
        "manuscript_claim": "The project evaluates structured medication-state, optional LLM-enhanced medication-state, and BioClinicalBERT/SapBERT medication-text neural representations as incremental patient-state signals beyond the clinical-only representation. LLM features are valid as LLM-derived manuscript features only when the abstraction audit reports effective_provider=ollama; fallback outputs are included for testing and reproducibility checks. Neural text representations are valid only when the neural audit reports successful BioClinicalBERT/SapBERT feature creation. Neuropathology variables are used only for secondary biological anchoring and are excluded from prediction.",
        "llm_medication_state": {
            "enabled": bool(cfg.get("llm_medication_state", {}).get("enabled", False)),
            "requested_provider": str(cfg.get("llm_medication_state", {}).get("provider", "")),
            "ollama_model": str(cfg.get("llm_medication_state", {}).get("ollama_model", "")),
            "max_unique_texts": cfg.get("llm_medication_state", {}).get("max_unique_texts", None),
            "num_predict": cfg.get("llm_medication_state", {}).get("num_predict", None),
            "max_prompt_chars": cfg.get("llm_medication_state", {}).get("max_prompt_chars", None),
            "request_timeout_s": cfg.get("llm_medication_state", {}).get("request_timeout_s", None),
            "reachability_timeout_s": cfg.get("llm_medication_state", {}).get("reachability_timeout_s", None),
            "keep_alive": cfg.get("llm_medication_state", {}).get("keep_alive", None),
            "ollama_format_mode": cfg.get("llm_medication_state", {}).get("ollama_format_mode", None),
            "persistent_cache_enabled": cfg.get("llm_medication_state", {}).get("persistent_cache_enabled", None),
            "persistent_cache_path": cfg.get("llm_medication_state", {}).get("persistent_cache_path", None),
            "model_profile": cfg.get("llm_medication_state", {}).get("model_profile", None),
            "model_comparison_enabled": cfg.get("llm_medication_state", {}).get("model_comparison_enabled", None),
            "model_comparison_candidates": cfg.get("llm_medication_state", {}).get("model_comparison_candidates", None),
            "llm_input_canonicalization_enabled": cfg.get("llm_medication_state", {}).get("llm_input_canonicalization_enabled", None),
            "llm_input_pretrained_normalization_enabled": cfg.get("llm_medication_state", {}).get("llm_input_pretrained_normalization_enabled", None),
            "llm_input_pretrained_normalization_backend": cfg.get("llm_medication_state", {}).get("llm_input_pretrained_normalization_backend", None),
            "llm_input_sapbert_model": cfg.get("llm_medication_state", {}).get("llm_input_sapbert_model", None),
            "llm_input_bioclinicalbert_model": cfg.get("llm_medication_state", {}).get("llm_input_bioclinicalbert_model", None),
            "record_coverage_target": cfg.get("llm_medication_state", {}).get("record_coverage_target", None),
            "effective_provider_counts": llm_audit.get("effective_provider", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if isinstance(llm_audit, pd.DataFrame) and not llm_audit.empty else {},
        },
        "neural_text_representations": {
            "enabled": bool(cfg.get("neural_text_representations", {}).get("enabled", False)),
            "configured_providers": list((cfg.get("neural_text_representations", {}).get("providers", {}) or {}).keys()),
            "audit": neural_audit.to_dict(orient="records") if isinstance(neural_audit, pd.DataFrame) and not neural_audit.empty else [],
        },
        "v2_3_train_validation_test_additions": [
            "participant-level 70:10:20 train-validation-test split",
            "validation-set hyperparameter selection for logistic and hist-gradient boosting models",
            "final model refit on training plus validation before held-out test evaluation",
            "validation-set isotonic and sigmoid calibration sensitivity outputs",
            "grouped permutation importance for feature-block interpretation",
            "calendar-time temporal validation sensitivity analysis",
            "optional LLM-enhanced medication-state abstraction and model-arm comparison"
        ],
    }
    with open(output_file(output_dir, "step5d_run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    step7_bar.update("Write run metadata JSON")

    (output_file(output_dir, "step5b_publication_main_findings_summary.md")).write_text(
        _make_publication_findings(metrics_df, delta_df, metadata), encoding="utf-8"
    )
    step7_bar.update("Write publication findings summary")

    table_index = pd.DataFrame(
        [
            {"output_file": "step3a_heldout_test_model_performance.csv", "manuscript_use": "Main model performance table"},
            {"output_file": "step3b_heldout_test_model_performance_bootstrap_ci.csv", "manuscript_use": "95% CIs for predictive metrics"},
            {"output_file": "step3e_validation_model_performance.csv", "manuscript_use": "Validation-set model comparison used before final held-out testing"},
            {"output_file": "step3g_llm_incremental_value_vs_structured_medication.csv", "manuscript_use": "LLM-enhanced medication-state incremental value compared with structured medication-state features"},
            {"output_file": "step3h_neural_representation_incremental_value.csv", "manuscript_use": "BioClinicalBERT/SapBERT neural medication-text representation incremental value"},
            {"output_file": "step3f_validation_hyperparameter_tuning_results.csv", "manuscript_use": "Validation-set hyperparameter selection audit table"},
            {"output_file": "step3c_medication_incremental_value_vs_clinical_only.csv", "manuscript_use": "Medication-feature incremental value table"},
            {"output_file": "step3d_medication_incremental_value_bootstrap_ci.csv", "manuscript_use": "95% CIs for medication-feature deltas"},
            {"output_file": "step4d_heldout_test_calibration_by_decile.csv", "manuscript_use": "Calibration table/figure source"},
            {"output_file": "step4e_heldout_test_decision_curve_analysis.csv", "manuscript_use": "Decision-curve analysis source"},
            {"output_file": "step4g_heldout_test_permutation_importance_final_model.csv", "manuscript_use": "Individual feature-importance table/figure source"},
            {"output_file": "step4h_validation_calibrated_model_performance.csv", "manuscript_use": "Calibrated risk-model performance sensitivity analysis"},
            {"output_file": "step4j_validation_calibrated_decision_curve_analysis.csv", "manuscript_use": "Calibrated decision-curve sensitivity analysis"},
            {"output_file": "step4k_heldout_test_grouped_permutation_importance.csv", "manuscript_use": "Grouped feature-block importance for medication-state contribution"},
            {"output_file": "step4l_temporal_validation_model_performance.csv", "manuscript_use": "Calendar-time temporal validation sensitivity analysis"},
            {"output_file": "step6c_neuropathology_anchor_risk_associations.csv", "manuscript_use": "Secondary neuropathology anchoring table"},
        ]
    )
    table_index["output_file"] = table_index["output_file"].map(public_output_name)
    _round_csv(table_index, output_file(output_dir, "step7a_manuscript_table_index.csv"), decimal_places)
    _round_csv(output_file_synopsis_frame(), output_file(output_dir, "step7c_output_file_synopsis.csv"), decimal_places)
    _round_csv(project_file_synopsis_frame(), output_file(output_dir, "step7d_project_file_synopsis.csv"), decimal_places)

    final_report = [
        "# Dementia progression final outputs report",
        "",
        f"Best model by final held-out test AUROC: `{best_model}`.",
        f"Best model by validation-set AUROC before final testing: `{best_validation_model}`.",
        "",
        "Primary manuscript model set: clinical-only, structured medication-aware, LLM-enhanced medication-aware, and optional BioClinicalBERT/SapBERT neural medication-aware feature representations under matched logistic and hist-gradient boosting model families.",
        "",
        "Do not use neuropathology variables as prediction features. V2.3 uses neuropathology only as a secondary biological anchor in the subset where these variables are available.",
        "",
        "Recommended manuscript framing: Neuropathology-Anchored PharmacoCognitive Machine Learning for medication-state-aware dementia progression risk stratification in MCI. This version adds strict LLM medication-state abstraction, required pretrained SapBERT/BioClinicalBERT medication normalization, optional BioClinicalBERT/SapBERT neural medication-text representation model arms, participant-level train-validation-test splitting, validation-set model selection, validation-set calibration sensitivity, grouped feature-block importance, and calendar-time temporal validation outputs.",
    ]
    (output_file(output_dir, "step7b_final_outputs_report.md")).write_text("\n".join(final_report) + "\n", encoding="utf-8")
    step7_bar.update("Write manuscript table index, file synopses, and final report")

    if bool(cfg.get("output_options", {}).get("save_model_bundle", True)):
        try:
            save_models_bundle(str(output_file(output_dir, "step5c_reproducible_trained_model_bundle.joblib")), fitted, metadata)
            step7_bar.update("Save reproducible trained model bundle")
        except Exception as exc:
            logger.warning("Could not save model bundle: %s", exc)
            step7_bar.update("Attempt model-bundle save and record warning")
    else:
        step7_bar.update("Skip model-bundle save because output option is disabled")
    step7_bar.close()

    progress_step(f"Pipeline completed successfully. Outputs saved to {output_dir}", enabled=progress_enabled, logger=logger)
    logger.info("Saved outputs to %s", output_dir)
    logger.info("metrics:\n%s", metrics_df.to_string(index=False))




    return True
