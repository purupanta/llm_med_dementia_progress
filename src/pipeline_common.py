"""
Project: dementia_progression
File: src/pipeline_common.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-25

Synopsis:
    Shared imports, constants, helper functions, output-file synopsis tables,
    validation utilities, and model-development helpers used by the refactored
    stepwise dementia progression pipeline.

Notes:
    This module was extracted from the validated v1.49 monolithic controller so
    that every analytic stage can live in a separate step file while preserving
    the tested computational behavior.
"""


from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# Avoid OpenMP oversubscription/deadlock on shared CPU/GPU servers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.cohort import add_visit_order_columns, build_next_visit_labels, select_mci_with_next_visit, summarize_cohort
from helpers.config_utils import load_config
from helpers.evaluation import build_metrics_table, evaluate_binary_predictions
from helpers.features import (
    build_feature_lists,
    build_feature_summary,
    infer_column_type,
    add_longitudinal_trajectory_features,
    longitudinal_trajectory_feature_columns,
    build_preprocessor,
    format_binary_columns_as_nullable_int,
    format_integral_numeric_columns_as_nullable_int,
    load_official_nacc_missing_codes,
    merge_missing_code_maps,
    llm_medication_feature_columns,
    neural_text_feature_columns,
    make_feature_frame,
    medication_feature_columns,
    recode_binary_columns,
    replace_column_specific_missing_codes,
    replace_global_missing_codes,
    replace_systemwide_structural_missing_codes,
    structured_medication_feature_columns,
)
from helpers.io_utils import build_drug_columns, ensure_dir, load_selected_columns, normalize_input_record_selection, unique_preserve_order
from helpers.logging_utils import log_gpu_environment, log_resolved_config, log_runtime_environment, setup_run_loggers
from helpers.llm_medication_state import apply_llm_medication_state, build_llm_medication_state_quality_audit, build_llm_medication_model_comparison_audit, build_llm_medication_state_certification_audit
from helpers.neural_text_representations import add_neural_text_representations
from helpers.medications import add_medication_category_features, combine_drug_columns, extract_unique_drug_names_from_columns
from helpers.progress import progress_bar, progress_iter, progress_step
from helpers.modeling import (
    build_hist_gradient_pipeline,
    build_logistic_pipeline,
    fit_pipeline,
    group_train_test_split_indices,
    group_train_validation_test_split_indices,
    predict_proba_positive,
    save_models_bundle,
)


OUTPUT_FILE_SYNOPSIS: dict[str, str] = {
    "logs/launcher.log": "Shell-level launcher audit, including selected GPU, selected Ollama mode, isolated/shared Ollama URL, and server startup diagnostics.",
    "logs/config.log": "Resolved configuration after YAML inheritance and environment overrides for reproducibility.",
    "logs/pipeline.log": "Main pipeline progress, step summaries, cohort dimensions, modeling status, and run completion messages.",
    "logs/llm.log": "Detailed LLM medication-state abstraction diagnostics, including Ollama reachability, warmup, timing, failures, fallback status, and audit counts.",
    "logs/runtime.log": "Runtime environment audit, including Python executable, platform, paths, GPU environment variables, and LLM environment overrides.",
    "logs/warnings.log": "Python warnings and pipeline warnings captured during execution.",
    "s1a_nacc_w_last_visit_filtered.csv": "Step 1 source file after YAML raw-row selection and row filtering for required downstream columns.",
    "s1b_incl_excl_cols.csv": "Step 1 source-column audit with included/excluded status, data type, maximum observed size, and concise description.",
    "s1c_nacc_w_last_visit_filtered_final.csv": "Final Step 1 downstream source after removing columns that are entirely null, blank, whitespace, or blank-space across retained rows.",
    "s1d_input_data_profile.csv": "Step 1 input profile summarizing row selection, required-column row filtering, all-blank column removal, and final source dimensions.",
    "s1e_official_nacc_dictionary_download_audit.csv": "Step 1 audit showing which official NACC dictionary CSV files were found or auto-downloaded under resources/nacc_official and whether they validated as CSV files.",
    "step1a_input_data_profile.csv": "Input-data profile summarizing YAML-controlled raw row-selection settings, rows loaded for processing, participants, loaded columns, and loaded neuropathology columns.",
    "step1b_mci_next_visit_cohort_summary.csv": "Analytic cohort summary for MCI visits with follow-up and next-visit dementia outcome.",
    "step1c_participant_train_validation_test_split_summary.csv": "Participant-level 70:10:20 train-validation-test split summary with participant-overlap leakage checks.",
    "step2a_clinical_only_feature_summary.csv": "Clinical-only feature representation summary, including feature type and missingness.",
    "step2b_structured_medication_aware_feature_summary.csv": "Structured medication-aware feature representation summary after adding medication count and medication-category variables.",
    "step2d_llm_medication_state_abstraction.csv": "Deduplicated audit table for optional LLM medication-state abstraction, including effective provider, parsed domains, schema-repair flags, and quality-control fields.",
    "step2h_llm_medication_state_quality_audit.csv": "Compact Step 2 LLM medication-state quality audit with unique-text coverage, visit-level coverage, blank/noninformative medication-text counts, fallback counts, schema-repair counts, and manuscript-safe interpretation flags.",
    "step2i_llm_model_comparison_audit.csv": "Optional Step 2 audit comparing the primary medication-state LLM with candidate local Ollama models on a sampled medication-text set.",
    "step2j_llm_model_agreement_summary.csv": "Optional summary of Step 2 candidate-model agreement for medication-state domain indicators.",
    "s2_medication_state_features/s2e_llm_medication_state_abstraction_partial.csv": "Live partial checkpoint of the Step 2 LLM medication-state audit file, updated atomically during the unique-medication-text loop.",
    "s2_medication_state_features/s2h_llm_medication_state_quality_audit_partial.csv": "Live partial quality audit for Step 2 LLM abstraction, updated atomically during the unique-medication-text loop.",
    "s2_medication_state_features/s2k_llm_certification_audit_partial.csv": "Live partial certification audit for Step 2 LLM abstraction, updated atomically during the unique-medication-text loop.",
    "s2_medication_state_features/s2_llm_medication_state_checkpoint_progress.json": "Live JSON checkpoint summarizing Step 2 LLM audit progress, write rule, processed unique texts, visit records covered by completed unique texts, and provider counts.",
    "step2e_llm_enhanced_medication_feature_summary.csv": "Feature summary for the clinical plus structured medication plus LLM-derived medication-state representation.",
    "step2c_analysis_dataset_structured_medication_state_features.csv": "Analysis-ready visit-level dataset containing outcome, clinical variables, structured medication-state features, and neuropathology columns loaded only for anchoring.",
    "step2f_analysis_dataset_llm_enhanced_medication_state.csv": "Analysis-ready visit-level dataset with structured medication-state and LLM-enhanced medication-state features.",
    "s2_medication_state_features/s2h_llm_medication_state_quality_audit.csv": "Compact quality-control audit for Step 2 LLM medication-state abstraction and fallback coverage.",
    "s2_medication_state_features/s2k_llm_certification_audit.csv": "Conservative certification audit distinguishing structurally valid LLM output from fallback/review rows; not a clinical ground-truth accuracy claim.",
    "step2l_neural_text_representation_audit.csv": "Audit file documenting BioClinicalBERT and SapBERT medication-text representation creation, model paths, component counts, cache files, and provider status.",
    "step2m_neural_medication_feature_summary.csv": "Feature summary for the ClinicalBERT/SapBERT neural medication-text representation feature set.",
    "step2n_analysis_dataset_neural_medication_state.csv": "Analysis-ready visit-level dataset including structured medication-state, LLM-enhanced medication-state, and BioClinicalBERT/SapBERT neural medication-text representation features.",
    "step2o_longitudinal_trajectory_feature_summary.csv": "Feature summary for leakage-safe prior-visit, change-score, and annualized-slope patient-state trajectory predictors.",
    "step2p_llm_drug_dictionary_mapping_audit.csv": "Audit verifying that unique raw drug names and LLM drug-dictionary rows align by stable llm_dictionary_key rather than positional drug_dictionary_index.",
    "step3a_heldout_test_model_performance.csv": "Final held-out test performance for the clinical-only, structured medication-aware, and LLM-enhanced medication-aware model set.",
    "step3b_heldout_test_model_performance_bootstrap_ci.csv": "Bootstrap confidence intervals for final held-out test discrimination, precision, Brier score, and threshold metrics.",
    "step3c_medication_incremental_value_vs_clinical_only.csv": "Structured medication-aware and LLM-enhanced medication-aware performance deltas relative to matched clinical-only models.",
    "step3g_llm_incremental_value_vs_structured_medication.csv": "LLM-enhanced medication-aware performance deltas relative to structured medication-aware models within each model family.",
    "step3h_neural_representation_incremental_value.csv": "ClinicalBERT/SapBERT neural medication-text representation performance deltas relative to clinical-only and LLM-enhanced medication-aware models.",
    "step3d_medication_incremental_value_bootstrap_ci.csv": "Bootstrap confidence intervals for structured medication-aware and LLM-enhanced medication-aware incremental performance deltas.",
    "step3e_validation_model_performance.csv": "Validation-set performance used during model development before final held-out testing.",
    "step3f_validation_hyperparameter_tuning_results.csv": "Validation-set hyperparameter-selection audit table for logistic regression and hist-gradient boosting.",
    "step4a_heldout_test_predicted_risk_scores.csv": "Held-out test visit-level predicted risk scores from each final model.",
    "step4b_heldout_test_clinical_vs_medication_reclassification.csv": "Held-out test comparison of clinical-only, structured medication-aware, and LLM-enhanced medication-aware predicted risks and risk-category reclassification.",
    "step4c_heldout_test_top_reclassified_examples.csv": "Highest reclassification examples for audit and interpretability review.",
    "step4d_heldout_test_calibration_by_decile.csv": "Calibration summary comparing mean predicted and observed risk by held-out test risk decile.",
    "step4e_heldout_test_decision_curve_analysis.csv": "Decision-curve analysis for evaluating net benefit across risk thresholds on the held-out test set.",
    "step4f_heldout_test_subgroup_performance_final_model.csv": "Subgroup performance for the final LLM-enhanced medication-aware hist-gradient boosting model.",
    "step4g_heldout_test_permutation_importance_final_model.csv": "Individual feature permutation importance for the final model on the held-out test subset.",
    "step4h_validation_calibrated_model_performance.csv": "Held-out performance after validation-set calibration sensitivity analysis.",
    "step4i_validation_calibrated_heldout_test_risk_predictions.csv": "Validation-calibrated risk predictions for held-out test records.",
    "step4j_validation_calibrated_decision_curve_analysis.csv": "Decision-curve sensitivity analysis using validation-calibrated predicted risks.",
    "step4k_heldout_test_grouped_permutation_importance.csv": "Grouped feature-block permutation importance for interpreting structured and LLM-enhanced medication-state contribution.",
    "step4l_temporal_validation_model_performance.csv": "Calendar-time temporal validation sensitivity analysis.",
    "step4m_validation_selected_final_model.csv": "Audit of the validation-selected final predictive model used for calibrated predictions and publication risk outputs. Selection uses validation performance, not held-out test performance.",
    "step5a_publication_model_performance_table.csv": "Manuscript-ready model performance table for reporting primary predictive results.",
    "step5b_publication_main_findings_summary.md": "Manuscript-oriented narrative summary of the main results and structured/LLM-enhanced medication-state incremental value.",
    "step5c_reproducible_trained_model_bundle.joblib": "Serialized fitted models, preprocessing objects, and metadata for reproducibility.",
    "step5d_run_metadata.json": "Machine-readable metadata documenting configuration, cohort, split design, feature counts, selected hyperparameters, and manuscript-safe claims.",
    "step6a_neuropathology_anchor_cohort.csv": "Autopsy-subset dataset for secondary neuropathology anchoring; not used for model training.",
    "step6b_neuropathology_anchor_outcome_summary.csv": "Neuropathology outcome counts and availability summary in the anchor cohort.",
    "step6c_neuropathology_anchor_risk_associations.csv": "Associations between predicted clinical risk and neuropathology outcomes in the autopsy subset.",
    "step6d_neuropathology_anchor_risk_quartiles.csv": "Neuropathology burden summarized across quartiles of predicted clinical risk.",
    "step6e_neuropathology_anchor_interpretation.md": "Manuscript-oriented interpretation of the secondary neuropathology anchoring analysis.",
    "step7a_manuscript_table_index.csv": "Index of manuscript-ready result tables and their recommended use.",
    "step7b_final_outputs_report.md": "Final run report summarizing the validation design, LLM-enhanced model set, main performance result, and interpretation boundaries.",
    "step7c_output_file_synopsis.csv": "Complete synopsis of output files generated by the pipeline.",
    "step7d_project_file_synopsis.csv": "Synopsis of major source-code, configuration, script, documentation, and test files included in the project.",
}

PROJECT_FILE_SYNOPSIS: dict[str, str] = {
    "configs/project.yaml": "Primary full-data configuration for the dementia_progression analysis.",
    "configs/_base.yaml": "Shared project defaults for input row selection, cohort construction, feature groups, modeling, validation, progress bars, and outputs.",
    "configs/_base_llm_medication_state.yaml": "Separate optional LLM medication-state abstraction configuration with provider, Ollama model, speed controls, persistent cache behavior, fallback behavior, and LLM output names.",
    "configs/_base_neural_text_representations.yaml": "BioClinicalBERT and SapBERT medication-text representation configuration for neural no-finetuning feature extraction.",
    "configs/smoke_synthetic.yaml": "Synthetic smoke-test configuration for end-to-end pipeline validation.",
    "src/run_pipeline.py": "Main pipeline controller for cohort construction, feature representation, model development, evaluation, reporting, and neuropathology anchoring.",
    "helpers/cohort.py": "Cohort-construction helper functions for visit ordering, next-visit labels, MCI selection, and cohort summaries.",
    "helpers/features.py": "Feature-engineering helper functions for clinical-only and medication-aware model matrices.",
    "helpers/medications.py": "Structured medication-state helper functions for medication text, medication counts, and medication category features.",
    "helpers/llm_medication_state.py": "Optional local LLM medication-state abstraction helper for clinically interpretable medication-state domains with fast-mode settings and persistent cache reuse.",
    "helpers/neural_text_representations.py": "Optional BioClinicalBERT and SapBERT medication-text representation helper that creates cached neural patient-state features.",
    "helpers/modeling.py": "Model-development helper functions for participant-level splitting, pipelines, predictions, and model persistence.",
    "helpers/evaluation.py": "Prediction-performance helper functions for binary outcome metrics and model-performance tables.",
    "helpers/io_utils.py": "Input/output helper functions for reading selected columns and managing output directories.",
    "helpers/config_utils.py": "YAML configuration-loading helper functions, including base-config inheritance.",
    "helpers/logging_utils.py": "Pipeline logging helper function for writing timestamped run logs.",
    "helpers/progress.py": "Progress-bar helper functions used for high-level stage messages and tqdm-wrapped loops.",
    "run_pipelines.sh": "Primary one-command shell launcher that centralizes runtime settings, CUDA GPU selection, paths, and then calls src/run_pipeline.sh.",
    "run_pipeline.sh": "Backward-compatible wrapper that delegates to run_pipelines.sh.",
    "src/run_pipeline.sh": "Executable src-level launcher that validates paths and runs the Python pipeline controller.",
    "tests/make_synthetic_nacc.py": "Synthetic NACC-like data generator for smoke testing.",
    "tests/test_core_pipeline_helpers.py": "Unit tests for core helpers and project output conventions.",
    "docs/README.md": "Primary project documentation and rerun instructions.",
    "docs/V2_11_NACC_AUTODOWNLOAD_AND_RAW_KEYED_LLM_DICTIONARY_FIX.md": "Version-specific documentation for YAML-controlled official NACC dictionary auto-download and raw-keyed stable LLM drug-dictionary mapping.",
    "docs/V2_11_VALIDATION_SUMMARY.md": "Version-specific validation summary for raw-keyed Step 2 dictionary mapping, smoke testing, and real-input validation checks.",
    "docs/VALIDATION_SUMMARY.md": "Validation summary for compile checks, tests, smoke run, full run, and main results.",
    "docs/OUTPUT_FILE_SYNOPSIS.md": "Human-readable synopsis of all manuscript and audit output files.",
    "docs/PROJECT_FILE_SYNOPSIS.md": "Human-readable synopsis of the source-code and configuration files.",
    "docs/dementia_progression_stepwise_output_file_guide.docx": "Word guide describing the s1_source_data_preparation through s7_reporting_documentation runtime output folders, ordered output naming convention, file creation logic, and recommended use of each file.",
}



# Runtime output paths use direct per-step folders. The folders are created only
# when a file is actually written to them through output_file().
OUTPUT_FILE_RENAME: dict[str, str] = {
    "s1a_nacc_w_last_visit_filtered.csv": "s1_source_data_preparation/s1a_nacc_w_last_visit_filtered.csv",
    "s1b_incl_excl_cols.csv": "s1_source_data_preparation/s1b_incl_excl_cols.csv",
    "s1c_nacc_w_last_visit_filtered_final.csv": "s1_source_data_preparation/s1c_nacc_w_last_visit_filtered_final.csv",
    "s1d_input_data_profile.csv": "s1_source_data_preparation/s1d_input_data_profile.csv",
    "s1e_official_nacc_dictionary_download_audit.csv": "s1_source_data_preparation/s1e_official_nacc_dictionary_download_audit.csv",
    "step1a_input_data_profile.csv": "s1_source_data_preparation/s1d_input_data_profile.csv",
    "step1b_mci_next_visit_cohort_summary.csv": "s2_medication_state_features/s2a_mci_next_visit_cohort_summary.csv",
    "step2a_clinical_only_feature_summary.csv": "s2_medication_state_features/s2b_clinical_only_feature_summary.csv",
    "step2b_structured_medication_aware_feature_summary.csv": "s2_medication_state_features/s2c_structured_medication_aware_feature_summary.csv",
    "step2c_analysis_dataset_structured_medication_state_features.csv": "s2_medication_state_features/s2d_analysis_dataset_structured_medication_state_features.csv",
    "step2d_unique_raw_drug_name_source.csv": "s2_medication_state_features/s2e_unique_raw_drug_name_source.csv",
    "step2d_llm_medication_state_abstraction.csv": "s2_medication_state_features/s2e_llm_medication_state_abstraction.csv",
    "step2e_llm_enhanced_medication_feature_summary.csv": "s2_medication_state_features/s2f_llm_enhanced_medication_feature_summary.csv",
    "step2f_analysis_dataset_llm_enhanced_medication_state.csv": "s2_medication_state_features/s2g_analysis_dataset_llm_enhanced_medication_state.csv",
    "step2h_llm_medication_state_quality_audit.csv": "s2_medication_state_features/s2h_llm_medication_state_quality_audit.csv",
    "step2i_llm_model_comparison_audit.csv": "s2_medication_state_features/s2i_llm_model_comparison_audit.csv",
    "step2j_llm_model_agreement_summary.csv": "s2_medication_state_features/s2j_llm_model_agreement_summary.csv",
    "step2k_llm_certification_audit.csv": "s2_medication_state_features/s2k_llm_certification_audit.csv",
    "step2l_neural_text_representation_audit.csv": "s2_medication_state_features/s2l_neural_text_representation_audit.csv",
    "step2m_neural_medication_feature_summary.csv": "s2_medication_state_features/s2m_neural_medication_feature_summary.csv",
    "step2n_analysis_dataset_neural_medication_state.csv": "s2_medication_state_features/s2n_analysis_dataset_neural_medication_state.csv",
    "step2o_longitudinal_trajectory_feature_summary.csv": "s2_medication_state_features/s2o_longitudinal_trajectory_feature_summary.csv",
    "step2p_llm_drug_dictionary_mapping_audit.csv": "s2_medication_state_features/s2p_llm_drug_dictionary_mapping_audit.csv",
    "step1c_participant_train_validation_test_split_summary.csv": "s3_model_training_validation/s3a_participant_train_validation_test_split_summary.csv",
    "step3a_heldout_test_model_performance.csv": "s3_model_training_validation/s3b_heldout_test_model_performance.csv",
    "step3b_heldout_test_model_performance_bootstrap_ci.csv": "s3_model_training_validation/s3c_heldout_test_model_performance_bootstrap_ci.csv",
    "step3c_medication_incremental_value_vs_clinical_only.csv": "s3_model_training_validation/s3d_medication_incremental_value_vs_clinical_only.csv",
    "step3d_medication_incremental_value_bootstrap_ci.csv": "s3_model_training_validation/s3e_medication_incremental_value_bootstrap_ci.csv",
    "step3e_validation_model_performance.csv": "s3_model_training_validation/s3f_validation_model_performance.csv",
    "step3f_validation_hyperparameter_tuning_results.csv": "s3_model_training_validation/s3g_validation_hyperparameter_tuning_results.csv",
    "step3g_llm_incremental_value_vs_structured_medication.csv": "s3_model_training_validation/s3h_llm_incremental_value_vs_structured_medication.csv",
    "step3h_neural_representation_incremental_value.csv": "s3_model_training_validation/s3i_neural_representation_incremental_value.csv",
    "step4a_heldout_test_predicted_risk_scores.csv": "s4_prediction_evaluation/s4a_heldout_test_predicted_risk_scores.csv",
    "step4b_heldout_test_clinical_vs_medication_reclassification.csv": "s4_prediction_evaluation/s4b_heldout_test_clinical_vs_medication_reclassification.csv",
    "step4c_heldout_test_top_reclassified_examples.csv": "s4_prediction_evaluation/s4c_heldout_test_top_reclassified_examples.csv",
    "step4d_heldout_test_calibration_by_decile.csv": "s4_prediction_evaluation/s4d_heldout_test_calibration_by_decile.csv",
    "step4e_heldout_test_decision_curve_analysis.csv": "s4_prediction_evaluation/s4e_heldout_test_decision_curve_analysis.csv",
    "step4f_heldout_test_subgroup_performance_final_model.csv": "s4_prediction_evaluation/s4f_heldout_test_subgroup_performance_final_model.csv",
    "step4g_heldout_test_permutation_importance_final_model.csv": "s4_prediction_evaluation/s4g_heldout_test_permutation_importance_final_model.csv",
    "step4h_validation_calibrated_model_performance.csv": "s4_prediction_evaluation/s4h_validation_calibrated_model_performance.csv",
    "step4i_validation_calibrated_heldout_test_risk_predictions.csv": "s4_prediction_evaluation/s4i_validation_calibrated_heldout_test_risk_predictions.csv",
    "step4j_validation_calibrated_decision_curve_analysis.csv": "s4_prediction_evaluation/s4j_validation_calibrated_decision_curve_analysis.csv",
    "step4k_heldout_test_grouped_permutation_importance.csv": "s4_prediction_evaluation/s4k_heldout_test_grouped_permutation_importance.csv",
    "step4l_temporal_validation_model_performance.csv": "s4_prediction_evaluation/s4l_temporal_validation_model_performance.csv",
    "step4m_validation_selected_final_model.csv": "s4_prediction_evaluation/s4m_validation_selected_final_model.csv",
    "step5a_publication_model_performance_table.csv": "s5_publication_outputs/s5a_publication_model_performance_table.csv",
    "step5b_publication_main_findings_summary.md": "s5_publication_outputs/s5b_publication_main_findings_summary.md",
    "step5c_reproducible_trained_model_bundle.joblib": "s5_publication_outputs/s5c_reproducible_trained_model_bundle.joblib",
    "step5d_run_metadata.json": "s5_publication_outputs/s5d_run_metadata.json",
    "step6a_neuropathology_anchor_cohort.csv": "s6_neuropathology_anchor_analysis/s6a_neuropathology_anchor_cohort.csv",
    "step6b_neuropathology_anchor_outcome_summary.csv": "s6_neuropathology_anchor_analysis/s6b_neuropathology_anchor_outcome_summary.csv",
    "step6c_neuropathology_anchor_risk_associations.csv": "s6_neuropathology_anchor_analysis/s6c_neuropathology_anchor_risk_associations.csv",
    "step6d_neuropathology_anchor_risk_quartiles.csv": "s6_neuropathology_anchor_analysis/s6d_neuropathology_anchor_risk_quartiles.csv",
    "step6e_neuropathology_anchor_interpretation.md": "s6_neuropathology_anchor_analysis/s6e_neuropathology_anchor_interpretation.md",
    "step7a_manuscript_table_index.csv": "s7_reporting_documentation/s7a_manuscript_table_index.csv",
    "step7b_final_outputs_report.md": "s7_reporting_documentation/s7b_final_outputs_report.md",
    "step7c_output_file_synopsis.csv": "s7_reporting_documentation/s7c_output_file_synopsis.csv",
    "step7d_project_file_synopsis.csv": "s7_reporting_documentation/s7d_project_file_synopsis.csv",
}


def public_output_name(logical_name: str) -> str:
    """Return the current public relative output path for a logical legacy name."""
    return OUTPUT_FILE_RENAME.get(str(logical_name), str(logical_name))


def output_file(output_dir: Path, logical_name: str) -> Path:
    """Resolve a logical output name and create its step folder only at write time."""
    rel = public_output_name(str(logical_name))
    path = output_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path



def _env_bool(value: str | None, default: bool | None = None) -> bool | None:
    """Parse a shell-provided boolean string for runtime config overrides."""
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {value!r}")


def apply_environment_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply runtime settings exported by run_pipelines.sh.

    The YAML file remains the reproducible default configuration. The shell
    launcher can override project identity, input path, output base, and datetime
    behavior without editing YAML. This keeps the requested one-command workflow
    auditable while preserving a stable config file.
    """
    out = copy.deepcopy(cfg)
    scalar_overrides = {
        "DEMENTIA_PROJECT_NAME": "project_name",
        "PROJECT_NAME": "project_name",
        "DEMENTIA_INPUT_CSV": "input_csv",
        "INPUT_CSV": "input_csv",
        "DEMENTIA_OUTPUT_DIR": "output_dir",
        "OUTPUT_DIR": "output_dir",
    }
    for env_name, cfg_key in scalar_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            out[cfg_key] = value

    append_datetime = _env_bool(os.environ.get("DEMENTIA_APPEND_DATETIME"), default=None)
    if append_datetime is not None:
        output_options = out.setdefault("output_options", {})
        output_options["append_datetime_to_output_dir"] = append_datetime

    llm_cfg = out.setdefault("llm_medication_state", {})
    llm_string_env_map = {
        "LLM_PROVIDER": "provider",
        "OLLAMA_BASE_URL": "ollama_base_url",
        "OLLAMA_MODEL": "ollama_model",
        "LLM_FALLBACK_PROVIDER": "fallback_provider",
        "LLM_SELECTION_STRATEGY": "selection_strategy",
        "LLM_KEEP_ALIVE": "keep_alive",
        "LLM_REQUEST_BACKEND": "request_backend",
        "LLM_OLLAMA_FORMAT_MODE": "ollama_format_mode",
        "LLM_PERSISTENT_CACHE_PATH": "persistent_cache_path",
        "LLM_DRUG_DICTIONARY_PERSISTENT_CACHE_PATH": "drug_dictionary_persistent_cache_path",
        "LLM_DRUG_DICTIONARY_FILENAME": "drug_dictionary_filename",
        "LLM_ABSTRACTION_UNIT": "abstraction_unit",
        "LLM_MODEL_PROFILE": "model_profile",
        "LLM_INPUT_ALIAS_RESOURCE_PATH": "llm_input_alias_resource_path",
        "LLM_INPUT_PRETRAINED_NORMALIZATION_BACKEND": "llm_input_pretrained_normalization_backend",
        "LLM_INPUT_SAPBERT_MODEL": "llm_input_sapbert_model",
        "LLM_INPUT_BIOCLINICALBERT_MODEL": "llm_input_bioclinicalbert_model",
        "LLM_INPUT_PRETRAINED_VOCABULARY_PATH": "llm_input_pretrained_vocabulary_path",
        "LLM_INPUT_PRETRAINED_DEVICE": "llm_input_pretrained_device",
    }
    for env_name, cfg_key in llm_string_env_map.items():
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            llm_cfg[cfg_key] = value
    if os.environ.get("OLLAMA_MODEL") and str(os.environ.get("OLLAMA_MODEL") or "").strip():
        if os.environ.get("OLLAMA_MODEL_FROM_LAUNCHER") == "1":
            # Preserve the model that run_pipelines.sh actually probed successfully.
            # The downstream helper reapplies this after named model-profile expansion
            # so profile defaults cannot silently revert to an uninstalled model and
            # trigger HTTP 404 during row-level /api/generate calls.
            llm_cfg["_launcher_selected_ollama_model"] = str(os.environ.get("OLLAMA_MODEL")).strip()
        elif not os.environ.get("LLM_MODEL_PROFILE"):
            llm_cfg["model_profile"] = "custom"

    llm_int_env_map = {
        "LLM_MAX_UNIQUE_TEXTS": "max_unique_texts",
        "LLM_MAX_ATTEMPTS": "max_attempts",
        "LLM_NUM_PREDICT": "num_predict",
        "LLM_NUM_CTX": "num_ctx",
        "LLM_MAX_PROMPT_CHARS": "max_prompt_chars",
        "LLM_PROGRESS_LOG_INTERVAL": "progress_log_interval",
        "LLM_INITIAL_FAILURE_ABORT_COUNT": "initial_failure_abort_count",
        "LLM_MAX_CONCURRENT_REQUESTS": "max_concurrent_requests",
        "LLM_CACHE_WRITE_INTERVAL": "cache_write_interval",
        "LLM_MODEL_COMPARISON_SAMPLE_SIZE": "model_comparison_sample_size",
        "LLM_MODEL_COMPARISON_NUM_PREDICT": "model_comparison_num_predict",
        "LLM_MODEL_COMPARISON_MAX_ATTEMPTS": "model_comparison_max_attempts",
        "LLM_INPUT_PRETRAINED_BATCH_SIZE": "llm_input_pretrained_batch_size",
    }
    for env_name, cfg_key in llm_int_env_map.items():
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            if cfg_key == "max_unique_texts" and str(value).strip().lower() in {"none", "null", "all", "full", "unlimited", "no_limit"}:
                llm_cfg[cfg_key] = None
            else:
                llm_cfg[cfg_key] = int(value)

    llm_float_env_map = {
        "LLM_REQUEST_TIMEOUT_S": "request_timeout_s",
        "LLM_TEMPERATURE": "temperature",
        "LLM_REACHABILITY_TIMEOUT_S": "reachability_timeout_s",
        "LLM_CONNECT_TIMEOUT_S": "connect_timeout_s",
        "LLM_PARTIAL_S2_WRITE_PERCENT_INTERVAL": "partial_s2_write_percent_interval",
        "LLM_PARTIAL_S2_WRITE_LATE_START_PERCENT": "partial_s2_write_late_start_percent",
        "LLM_PARTIAL_S2_WRITE_LATE_PERCENT_INTERVAL": "partial_s2_write_late_percent_interval",
        "LLM_PARTIAL_S2_WRITE_MIN_SECONDS": "partial_s2_write_min_seconds",
        "LLM_RECORD_COVERAGE_TARGET": "record_coverage_target",
        "LLM_MODEL_COMPARISON_REQUEST_TIMEOUT_S": "model_comparison_request_timeout_s",
        "LLM_CERTIFICATION_MIN_CONFIDENCE": "llm_certification_min_confidence",
        "LLM_INPUT_PRETRAINED_SIMILARITY_THRESHOLD": "llm_input_pretrained_similarity_threshold",
    }
    for env_name, cfg_key in llm_float_env_map.items():
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            llm_cfg[cfg_key] = float(value)
    enabled = _env_bool(os.environ.get("LLM_MEDICATION_STATE_ENABLED"), default=None)
    if enabled is not None:
        llm_cfg["enabled"] = enabled
    fail_on_error = _env_bool(os.environ.get("LLM_FAIL_ON_ERROR"), default=None)
    if fail_on_error is not None:
        llm_cfg["fail_on_error"] = fail_on_error
    require_ollama = _env_bool(os.environ.get("LLM_REQUIRE_OLLAMA"), default=None)
    if require_ollama is not None:
        llm_cfg["require_ollama_when_enabled"] = require_ollama
    min_successes = os.environ.get("LLM_REQUIRE_MIN_SUCCESSFUL_PARSES")
    if min_successes is not None and str(min_successes).strip() != "":
        llm_cfg["require_min_ollama_successes"] = int(min_successes)
    persistent_cache_enabled = _env_bool(os.environ.get("LLM_PERSISTENT_CACHE_ENABLED"), default=None)
    if persistent_cache_enabled is not None:
        llm_cfg["persistent_cache_enabled"] = persistent_cache_enabled
    warmup_enabled = _env_bool(os.environ.get("LLM_WARMUP_ENABLED"), default=None)
    if warmup_enabled is not None:
        llm_cfg["warmup_enabled"] = warmup_enabled
    structured_json_schema_enabled = _env_bool(os.environ.get("LLM_STRUCTURED_JSON_SCHEMA_ENABLED"), default=None)
    if structured_json_schema_enabled is not None:
        llm_cfg["structured_json_schema_enabled"] = structured_json_schema_enabled
    row_level_cpu_fallback_enabled = _env_bool(os.environ.get("LLM_ROW_LEVEL_CPU_FALLBACK_ENABLED"), default=None)
    if row_level_cpu_fallback_enabled is not None:
        llm_cfg["row_level_cpu_fallback_enabled"] = row_level_cpu_fallback_enabled
    llm_certification_enabled = _env_bool(os.environ.get("LLM_CERTIFICATION_ENABLED"), default=None)
    if llm_certification_enabled is not None:
        llm_cfg["llm_certification_enabled"] = llm_certification_enabled
    llm_certification_require_no_manual_review = _env_bool(os.environ.get("LLM_CERTIFICATION_REQUIRE_NO_MANUAL_REVIEW"), default=None)
    if llm_certification_require_no_manual_review is not None:
        llm_cfg["llm_certification_require_no_manual_review"] = llm_certification_require_no_manual_review
    partial_s2_write_enabled = _env_bool(os.environ.get("LLM_PARTIAL_S2_WRITE_ENABLED"), default=None)
    if partial_s2_write_enabled is not None:
        llm_cfg["partial_s2_write_enabled"] = partial_s2_write_enabled
    model_comparison_enabled = _env_bool(os.environ.get("LLM_MODEL_COMPARISON_ENABLED"), default=None)
    if model_comparison_enabled is not None:
        llm_cfg["model_comparison_enabled"] = model_comparison_enabled
    canon_enabled = _env_bool(os.environ.get("LLM_INPUT_CANONICALIZATION_ENABLED"), default=None)
    if canon_enabled is not None:
        llm_cfg["llm_input_canonicalization_enabled"] = canon_enabled
    sort_tokens = _env_bool(os.environ.get("LLM_INPUT_SORT_UNIQUE_TOKENS"), default=None)
    if sort_tokens is not None:
        llm_cfg["llm_input_sort_unique_tokens"] = sort_tokens
    alias_norm = _env_bool(os.environ.get("LLM_INPUT_ALIAS_NORMALIZATION_ENABLED"), default=None)
    if alias_norm is not None:
        llm_cfg["llm_input_alias_normalization_enabled"] = alias_norm
    pretrained_norm_enabled = _env_bool(os.environ.get("LLM_INPUT_PRETRAINED_NORMALIZATION_ENABLED"), default=None)
    if pretrained_norm_enabled is not None:
        llm_cfg["llm_input_pretrained_normalization_enabled"] = pretrained_norm_enabled
    pretrained_norm_required = _env_bool(os.environ.get("LLM_INPUT_PRETRAINED_NORMALIZATION_REQUIRED"), default=None)
    if pretrained_norm_required is not None:
        llm_cfg["llm_input_pretrained_normalization_required"] = pretrained_norm_required
    pretrained_local_files_only = _env_bool(os.environ.get("LLM_INPUT_PRETRAINED_LOCAL_FILES_ONLY"), default=None)
    if pretrained_local_files_only is not None:
        llm_cfg["llm_input_pretrained_local_files_only"] = pretrained_local_files_only
    if (pretrained_norm_enabled is not None or pretrained_norm_required is not None or pretrained_local_files_only is not None) and not os.environ.get("LLM_MODEL_PROFILE"):
        llm_cfg["model_profile"] = "custom"
    strict_response_validation = _env_bool(os.environ.get("LLM_STRICT_RESPONSE_VALIDATION_ENABLED"), default=None)
    if strict_response_validation is not None:
        llm_cfg["strict_response_validation_enabled"] = strict_response_validation
    strict_cache_validation = _env_bool(os.environ.get("LLM_STRICT_CACHE_VALIDATION_ENABLED"), default=None)
    if strict_cache_validation is not None:
        llm_cfg["strict_cache_validation_enabled"] = strict_cache_validation
    candidates = os.environ.get("LLM_MODEL_COMPARISON_CANDIDATES")
    if candidates is not None and str(candidates).strip() != "":
        llm_cfg["model_comparison_candidates"] = [x.strip() for x in str(candidates).split(",") if x.strip()]

    neural_cfg = out.setdefault("neural_text_representations", {})
    neural_enabled = _env_bool(os.environ.get("NEURAL_TEXT_REPRESENTATIONS_ENABLED"), default=None)
    if neural_enabled is not None:
        neural_cfg["enabled"] = neural_enabled
    neural_strict = _env_bool(os.environ.get("NEURAL_TEXT_STRICT_MODEL_LOADING"), default=None)
    if neural_strict is not None:
        neural_cfg["strict_model_loading"] = neural_strict
    neural_local_only = _env_bool(os.environ.get("NEURAL_TEXT_LOCAL_FILES_ONLY"), default=None)
    if neural_local_only is not None:
        neural_cfg["local_files_only"] = neural_local_only
    neural_string_env_map = {
        "NEURAL_TEXT_COLUMN": "text_column",
        "NEURAL_TEXT_DEVICE": "device",
        "NEURAL_TEXT_CACHE_DIR": "cache_dir",
    }
    for env_name, cfg_key in neural_string_env_map.items():
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            neural_cfg[cfg_key] = value
    neural_int_env_map = {
        "NEURAL_TEXT_BATCH_SIZE": "batch_size",
        "NEURAL_TEXT_MAX_LENGTH": "max_length",
        "NEURAL_TEXT_N_COMPONENTS": "n_components",
    }
    for env_name, cfg_key in neural_int_env_map.items():
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            neural_cfg[cfg_key] = int(value)
    providers_cfg = neural_cfg.setdefault("providers", {})
    if os.environ.get("NEURAL_TEXT_BIOCLINICALBERT_MODEL"):
        providers_cfg.setdefault("bioclinicalbert", {})["model_name_or_path"] = os.environ["NEURAL_TEXT_BIOCLINICALBERT_MODEL"]
    if os.environ.get("NEURAL_TEXT_SAPBERT_MODEL"):
        providers_cfg.setdefault("sapbert", {})["model_name_or_path"] = os.environ["NEURAL_TEXT_SAPBERT_MODEL"]

    progress_enabled = _env_bool(os.environ.get("PROGRESS_BARS"), default=None)
    if progress_enabled is not None:
        progress_cfg = out.setdefault("progress", {})
        progress_cfg["enabled"] = progress_enabled

    runtime_cfg = out.setdefault("runtime", {})
    input_selection = runtime_cfg.setdefault("input_record_selection", {})
    min_index = os.environ.get("DEMENTIA_MIN_ROW_INDEX") or os.environ.get("INPUT_MIN_ROW_INDEX")
    if min_index is not None and str(min_index).strip() != "":
        input_selection["min_row_index"] = int(min_index)
    max_index = os.environ.get("DEMENTIA_MAX_ROW_INDEX") or os.environ.get("INPUT_MAX_ROW_INDEX")
    if max_index is not None and str(max_index).strip() != "":
        if str(max_index).strip().lower() in {"all", "none", "null"}:
            input_selection["max_row_index"] = "all"
        else:
            input_selection["max_row_index"] = int(max_index)

    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the dementia_progression Neuropathology-Anchored PharmacoCognitive Machine Learning pipeline."
    )
    ap.add_argument("config_positional", nargs="?", help="Optional positional YAML config path.")
    ap.add_argument("--config", dest="config", default=None, help="Path to YAML config file.")
    args = ap.parse_args()
    args.config = (
        args.config
        or args.config_positional
        or os.environ.get("CONFIG_PATH")
        or os.environ.get("PIPELINE_CONFIG")
        or "configs/project.yaml"
    )
    return args


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (project_root / path).resolve()


def _latest_matching_output_dir(output_base: Path, separator: str = "_") -> Path | None:
    """Return the latest existing timestamped output directory for resume mode."""
    parent = output_base.parent
    prefix = f"{output_base.name}{separator}"
    candidates = [p for p in parent.glob(prefix + "*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)[0]


def build_output_dir(project_root: Path, cfg: dict[str, Any]) -> Path:
    """Resolve the run output directory, including YAML-controlled resume mode.

    Normal mode writes to a new timestamped folder. Resume mode writes to a prior
    output folder so cancelled long runs can continue using the same logs/partials
    and the persistent Step 2 LLM cache. Resume intentionally does not skip all
    deterministic stages; it reruns cheap deterministic steps and resumes the
    expensive LLM medication abstraction from completed cached rows.
    """
    output_base = resolve_path(project_root, cfg["output_dir"])
    output_options = cfg.get("output_options", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}
    resume_cfg = runtime_cfg.get("resume", {}) or {}
    separator = str(
        cfg.get("output_datetime_separator")
        or output_options.get("output_datetime_separator")
        or "_"
    )

    env_resume_dir = os.environ.get("DEMENTIA_RESUME_OUTPUT_DIR") or os.environ.get("RESUME_OUTPUT_DIR")
    env_resume_enabled = str(os.environ.get("DEMENTIA_RESUME", os.environ.get("RESUME_PIPELINE", ""))).strip().lower() in {"1", "true", "yes", "y", "on"}
    resume_enabled = bool(resume_cfg.get("enabled", False)) or bool(env_resume_enabled) or bool(env_resume_dir)
    if resume_enabled:
        resume_dir_raw = env_resume_dir or str(resume_cfg.get("output_dir", "") or "").strip()
        if resume_dir_raw:
            resume_dir = resolve_path(project_root, resume_dir_raw)
        elif bool(resume_cfg.get("use_latest_matching_output_dir", False)):
            latest = _latest_matching_output_dir(output_base, separator=separator)
            if latest is None:
                raise FileNotFoundError(f"Resume requested but no prior output directory matched {output_base.name}{separator}* under {output_base.parent}")
            resume_dir = latest
        else:
            raise ValueError("Resume requested but runtime.resume.output_dir is blank and use_latest_matching_output_dir is false.")
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume output directory does not exist: {resume_dir}")
        cfg.setdefault("runtime", {}).setdefault("resume", {})["active_output_dir"] = str(resume_dir)
        return resume_dir

    append_datetime = bool(
        cfg.get("append_datetime_to_output_dir", False)
        or output_options.get("append_datetime_to_output_dir", False)
    )
    if not append_datetime:
        return output_base

    datetime_format = str(
        cfg.get("output_datetime_format")
        or output_options.get("output_datetime_format")
        or "%Y%m%d_%H%M%S"
    )
    timestamp = os.environ.get("DEMENTIA_RUN_TIMESTAMP") or datetime.now().strftime(datetime_format)
    return output_base.parent / f"{output_base.name}{separator}{timestamp}"


def build_usecols(cfg: dict[str, Any]) -> list[str]:
    visit_cols = [
        cfg["id_col"],
        cfg["visit_number_col"],
        cfg["visit_year_col"],
        cfg["visit_month_col"],
        cfg["visit_day_col"],
        cfg["last_visit_col"],
        cfg["current_diagnosis_col"],
    ]
    drug_cols = build_drug_columns(cfg["drug_prefix"], int(cfg["max_drug_columns"]))
    return unique_preserve_order(
        visit_cols + list(cfg.get("selected_base_columns", [])) + list(cfg.get("neuropathology_columns", [])) + drug_cols
    )



def progress_enabled_from_config(cfg: dict[str, Any]) -> bool:
    """Return whether console progress bars should be displayed for this run."""
    progress_cfg = cfg.get("progress", {}) or {}
    return bool(progress_cfg.get("enabled", True))


def _round_csv(df: pd.DataFrame, path: Path, decimal_places: int = 6) -> None:
    """Write a CSV with stable precision and clean integer/binary formatting.

    True binary columns are exported as nullable integer 0/1 rather than
    0.0/1.0. Other whole-number numeric columns are also exported as nullable
    integers to avoid cosmetic decimals in clinical codes and counts. Continuous
    numeric variables and neural embedding components remain floating-point
    values rounded only for CSV readability.
    """
    out = df.copy()
    out = format_binary_columns_as_nullable_int(out)
    out = format_integral_numeric_columns_as_nullable_int(out)
    float_cols = out.select_dtypes(include=["float", "float64", "float32"]).columns
    if len(float_cols):
        out[float_cols] = out[float_cols].round(decimal_places)
    out.to_csv(path, index=False)


def _classification_labels(prob: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(prob, dtype=float) >= float(threshold)).astype(int)


def add_visit_time_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Create current-visit timing and prior-MCI-history features without future leakage.

    Earlier versions used raw ``NACCDAYS`` for elapsed-time features. In this
    project extract, ``NACCDAYS`` can be constant within participant after the
    last-visit construction, which makes derived intervals non-informative. The
    corrected implementation computes elapsed time from current-visit date
    components (VISITYR/VISITMO/VISITDAY) and excludes raw ``NACCDAYS`` from
    Step 2 modeling feature lists.
    """
    out = df.copy()
    id_col = cfg["id_col"]
    visit_number_col = cfg["visit_number_col"]
    visit_year_col = cfg["visit_year_col"]
    visit_month_col = cfg["visit_month_col"]
    visit_day_col = cfg["visit_day_col"]
    visit_cols = [visit_number_col, visit_year_col, visit_month_col, visit_day_col]
    for col in ["NACCAGE"] + visit_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if all(c in out.columns for c in [visit_year_col, visit_month_col, visit_day_col]):
        visit_dates = pd.to_datetime(
            {
                "year": pd.to_numeric(out[visit_year_col], errors="coerce"),
                "month": pd.to_numeric(out[visit_month_col], errors="coerce"),
                "day": pd.to_numeric(out[visit_day_col], errors="coerce"),
            },
            errors="coerce",
        )
    else:
        visit_dates = pd.Series(pd.NaT, index=out.index)
    out["_visit_date_for_timing"] = visit_dates

    sort_cols = [id_col] + [c for c in ["_visit_date_for_timing", visit_number_col, visit_year_col, visit_month_col, visit_day_col] if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    out["prior_mci_visit_count"] = out.groupby(id_col).cumcount().astype("Int64")

    date_series = out["_visit_date_for_timing"]
    out["days_since_prior_mci_visit"] = out.groupby(id_col)["_visit_date_for_timing"].diff().dt.days.astype("Float64")
    first_dates = out.groupby(id_col)["_visit_date_for_timing"].transform("first")
    out["days_since_first_mci_visit"] = (date_series - first_dates).dt.days.astype("Float64")
    out = out.drop(columns=["_visit_date_for_timing"])
    return out


def visit_time_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "NACCAGE",
        "NACCVNUM",
        "VISITYR",
        "prior_mci_visit_count",
        "days_since_prior_mci_visit",
        "days_since_first_mci_visit",
    ]
    return [c for c in candidates if c in df.columns]



def _weighted_auc_from_counts(y_arr: np.ndarray, prob_arr: np.ndarray, counts: np.ndarray, order_asc: np.ndarray, group_starts: np.ndarray) -> float:
    """Compute weighted ROC AUC for a bootstrap sample represented by counts.

    This avoids repeatedly materializing bootstrap arrays and is much faster
    than calling sklearn metrics thousands of times during manuscript CI runs.
    Tied probabilities receive the standard 0.5 credit for positive-negative
    pairs within the same score group.
    """
    w_sorted = counts[order_asc].astype(float)
    y_sorted = y_arr[order_asc].astype(int)
    pos_total = float(np.sum(w_sorted * y_sorted))
    neg_total = float(np.sum(w_sorted * (1 - y_sorted)))
    if pos_total <= 0 or neg_total <= 0:
        return np.nan
    pos_group = np.add.reduceat(w_sorted * y_sorted, group_starts)
    neg_group = np.add.reduceat(w_sorted * (1 - y_sorted), group_starts)
    cum_neg_before = np.cumsum(neg_group) - neg_group
    numerator = np.sum(pos_group * cum_neg_before + 0.5 * pos_group * neg_group)
    return float(numerator / (pos_total * neg_total))


def _weighted_average_precision_from_counts(y_arr: np.ndarray, counts: np.ndarray, order_desc: np.ndarray) -> float:
    """Compute weighted average precision for a bootstrap sample represented by counts."""
    w_sorted = counts[order_desc].astype(float)
    y_sorted = y_arr[order_desc].astype(int)
    pos_total = float(np.sum(w_sorted * y_sorted))
    if pos_total <= 0:
        return np.nan
    cum_tp = np.cumsum(w_sorted * y_sorted)
    cum_all = np.cumsum(w_sorted)
    precision = np.divide(cum_tp, cum_all, out=np.zeros_like(cum_tp, dtype=float), where=cum_all > 0)
    return float(np.sum(precision * w_sorted * y_sorted) / pos_total)


def _bootstrap_fast_metrics(
    y_arr: np.ndarray,
    prob_arr: np.ndarray,
    counts: np.ndarray,
    *,
    threshold: float,
    order_asc: np.ndarray,
    group_starts: np.ndarray,
    order_desc: np.ndarray,
) -> dict[str, float]:
    """Return bootstrap metrics using count weights instead of copied arrays."""
    y_float = y_arr.astype(float)
    total = float(np.sum(counts))
    auc = _weighted_auc_from_counts(y_arr, prob_arr, counts, order_asc, group_starts)
    ap = _weighted_average_precision_from_counts(y_arr, counts, order_desc)
    brier = float(np.sum(counts * (prob_arr - y_float) ** 2) / total)
    pred = (prob_arr >= float(threshold)).astype(int)
    tp = float(np.sum(counts * ((pred == 1) & (y_arr == 1))))
    fp = float(np.sum(counts * ((pred == 1) & (y_arr == 0))))
    tn = float(np.sum(counts * ((pred == 0) & (y_arr == 0))))
    fn = float(np.sum(counts * ((pred == 0) & (y_arr == 1))))
    sensitivity = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    return {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "brier_score": float(brier),
        "balanced_accuracy_at_0_5": float((sensitivity + specificity) / 2.0),
    }


def _bootstrap_orders(prob_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute sort orders and probability-tie groups for fast bootstrap CIs."""
    order_asc = np.argsort(prob_arr, kind="mergesort")
    sorted_prob = prob_arr[order_asc]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_prob) != 0) + 1]
    order_desc = order_asc[::-1]
    return order_asc, group_starts, order_desc


def _bootstrap_metric_ci(
    model_probabilities: dict[str, np.ndarray],
    y_true: pd.Series,
    *,
    n_bootstrap: int,
    random_state: int,
    threshold: float,
    progress_enabled: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    y_arr = np.asarray(y_true).astype(int)
    n = len(y_arr)
    rows = []
    for model_name, prob in progress_iter(model_probabilities.items(), enabled=progress_enabled, desc="Bootstrap model CIs", total=len(model_probabilities), unit="model"):
        prob_arr = np.asarray(prob, dtype=float)
        order_asc, group_starts, order_desc = _bootstrap_orders(prob_arr)
        boot = {"roc_auc": [], "average_precision": [], "brier_score": [], "balanced_accuracy_at_0_5": []}
        for _ in progress_iter(range(int(n_bootstrap)), enabled=progress_enabled, desc=f"Bootstrap {model_name}", total=int(n_bootstrap), unit="rep", leave=False):
            counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
            if np.sum(counts * y_arr) <= 0 or np.sum(counts * (1 - y_arr)) <= 0:
                continue
            m = _bootstrap_fast_metrics(
                y_arr,
                prob_arr,
                counts,
                threshold=threshold,
                order_asc=order_asc,
                group_starts=group_starts,
                order_desc=order_desc,
            )
            for metric in boot:
                boot[metric].append(float(m[metric]))
        row = {"model": model_name, "n_bootstrap_valid": min(len(v) for v in boot.values()) if boot else 0}
        for metric, values in boot.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_delta_ci(
    model_probabilities: dict[str, np.ndarray],
    y_true: pd.Series,
    *,
    pairs: list[tuple[str, str, str]],
    n_bootstrap: int,
    random_state: int,
    threshold: float,
    progress_enabled: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state) + 1009)
    y_arr = np.asarray(y_true).astype(int)
    n = len(y_arr)
    precomputed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, prob in model_probabilities.items():
        prob_arr = np.asarray(prob, dtype=float)
        order_asc, group_starts, order_desc = _bootstrap_orders(prob_arr)
        precomputed[name] = (prob_arr, order_asc, group_starts, order_desc)
    rows = []
    for comparison, med_model, clin_model in progress_iter(pairs, enabled=progress_enabled, desc="Bootstrap delta CIs", total=len(pairs), unit="comparison"):
        delta = {"roc_auc": [], "average_precision": [], "brier_score": [], "balanced_accuracy_at_0_5": []}
        med_prob, med_asc, med_starts, med_desc = precomputed[med_model]
        clin_prob, clin_asc, clin_starts, clin_desc = precomputed[clin_model]
        for _ in progress_iter(range(int(n_bootstrap)), enabled=progress_enabled, desc=f"Delta {comparison[:36]}", total=int(n_bootstrap), unit="rep", leave=False):
            counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
            if np.sum(counts * y_arr) <= 0 or np.sum(counts * (1 - y_arr)) <= 0:
                continue
            med_m = _bootstrap_fast_metrics(
                y_arr, med_prob, counts, threshold=threshold, order_asc=med_asc, group_starts=med_starts, order_desc=med_desc
            )
            clin_m = _bootstrap_fast_metrics(
                y_arr, clin_prob, counts, threshold=threshold, order_asc=clin_asc, group_starts=clin_starts, order_desc=clin_desc
            )
            for metric in delta:
                delta[metric].append(float(med_m[metric] - clin_m[metric]))
        row = {"comparison": comparison, "medication_aware_model": med_model, "clinical_only_model": clin_model}
        for metric, values in delta.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            row[f"delta_{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"delta_{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _calibration_by_decile(model_probabilities: dict[str, np.ndarray], y_true: pd.Series, *, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    y_arr = np.asarray(y_true).astype(int)
    for model_name, prob in model_probabilities.items():
        tmp = pd.DataFrame({"y_true": y_arr, "risk": np.asarray(prob, dtype=float)})
        try:
            tmp["risk_decile"] = pd.qcut(tmp["risk"], q=int(n_bins), labels=False, duplicates="drop") + 1
        except ValueError:
            tmp["risk_decile"] = 1
        for decile, chunk in tmp.groupby("risk_decile", dropna=False):
            rows.append(
                {
                    "model": model_name,
                    "risk_decile": int(decile) if pd.notna(decile) else -1,
                    "n": int(len(chunk)),
                    "events": int(chunk["y_true"].sum()),
                    "observed_event_rate": float(chunk["y_true"].mean()),
                    "mean_predicted_risk": float(chunk["risk"].mean()),
                    "min_predicted_risk": float(chunk["risk"].min()),
                    "max_predicted_risk": float(chunk["risk"].max()),
                    "calibration_error": float(chunk["risk"].mean() - chunk["y_true"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "risk_decile"]).reset_index(drop=True)


def _decision_curve(model_probabilities: dict[str, np.ndarray], y_true: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    y_arr = np.asarray(y_true).astype(int)
    n = len(y_arr)
    prevalence = float(y_arr.mean())
    rows = []
    for threshold in thresholds:
        pt = float(threshold)
        if not 0 < pt < 1:
            continue
        treat_all_nb = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        treat_none_nb = 0.0
        for model_name, prob in model_probabilities.items():
            pred = np.asarray(prob, dtype=float) >= pt
            tp = int(((pred == 1) & (y_arr == 1)).sum())
            fp = int(((pred == 1) & (y_arr == 0)).sum())
            nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
            rows.append(
                {
                    "model": model_name,
                    "threshold_probability": pt,
                    "net_benefit": float(nb),
                    "treat_all_net_benefit": float(treat_all_nb),
                    "treat_none_net_benefit": float(treat_none_nb),
                    "true_positives": tp,
                    "false_positives": fp,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)



def _model_family_from_name(model_name: str) -> str:
    if model_name.endswith("logistic"):
        return "logistic"
    if model_name.endswith("hist_gradient_boosting"):
        return "hist_gradient_boosting"
    raise ValueError(f"Unknown model family for {model_name}")


def _cfg_with_model_params(cfg: dict[str, Any], family: str, params: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("modeling", {})
    out["modeling"].setdefault(family, {})
    out["modeling"][family].update(params)
    return out


def _deduplicate_candidate_params(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for params in candidates:
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(params)
    return out


def _candidate_params_for_family(cfg: dict[str, Any], family: str) -> list[dict[str, Any]]:
    modeling_cfg = cfg.get("modeling", {})
    tuning_cfg = modeling_cfg.get("hyperparameter_tuning", {})
    if not bool(tuning_cfg.get("enabled", True)):
        return [{}]
    if family == "logistic":
        base_c = float(modeling_cfg.get("logistic", {}).get("C", 0.5))
        c_grid = tuning_cfg.get("logistic_C_grid", [0.25, 0.5, 1.0])
        candidates = [{"C": base_c}] + [{"C": float(c)} for c in c_grid]
        return _deduplicate_candidate_params(candidates)
    if family == "hist_gradient_boosting":
        base = dict(modeling_cfg.get("hist_gradient_boosting", {}))
        candidate_grid = tuning_cfg.get("hist_gradient_boosting_grid")
        if not candidate_grid:
            candidate_grid = [
                {},
                {"learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 0.05, "min_samples_leaf": 40, "max_iter": 500},
                {"learning_rate": 0.025, "max_leaf_nodes": 15, "l2_regularization": 0.10, "min_samples_leaf": 40, "max_iter": 700},
                {"learning_rate": 0.04, "max_leaf_nodes": 15, "l2_regularization": 0.05, "min_samples_leaf": 30, "max_iter": 500},
                {"learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 0.10, "min_samples_leaf": 50, "max_iter": 500},
            ]
        candidates = [base]
        for item in candidate_grid:
            params = dict(base)
            params.update(dict(item or {}))
            candidates.append(params)
        return _deduplicate_candidate_params(candidates)
    return [{}]


def _build_pipeline_for_family(family: str, feature_lists: dict[str, list[str]], cfg: dict[str, Any]):
    if family == "logistic":
        return build_logistic_pipeline(build_preprocessor(feature_lists, scale_numeric=True), cfg)
    if family == "hist_gradient_boosting":
        return build_hist_gradient_pipeline(build_preprocessor(feature_lists, scale_numeric=False), cfg)
    raise ValueError(f"Unknown model family: {family}")


def _tune_model_on_validation(
    model_name: str,
    feature_lists: dict[str, list[str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    cfg: dict[str, Any],
    *,
    threshold: float,
    progress_enabled: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, Any]:
    """Select model hyperparameters using only the participant-level validation set."""
    family = _model_family_from_name(model_name)
    rows: list[dict[str, Any]] = []
    candidate_outputs: list[tuple[dict[str, Any], np.ndarray, Any, dict[str, float]]] = []
    use_hgb_weights = bool(cfg.get("modeling", {}).get("hist_gradient_boosting", {}).get("use_balanced_sample_weight", True))
    candidates = _candidate_params_for_family(cfg, family)
    for i, params in enumerate(progress_iter(candidates, enabled=progress_enabled, desc=f"Tune {model_name}", total=len(candidates), unit="candidate", leave=False), start=1):
        candidate_id = f"{family}_{i:03d}"
        candidate_cfg = _cfg_with_model_params(cfg, family, params)
        model = _build_pipeline_for_family(family, feature_lists, candidate_cfg)
        fitted_model = fit_pipeline(model_name, model, X_train.copy(), y_train.copy(), use_balanced_sample_weight=use_hgb_weights)
        validation_prob = predict_proba_positive(fitted_model, X_validation.copy())
        metrics = evaluate_binary_predictions(y_validation, validation_prob, threshold=threshold)
        row = {
            "model": model_name,
            "model_family": family,
            "candidate_id": candidate_id,
            "candidate_params_json": json.dumps(params, sort_keys=True),
            "validation_n": int(metrics["n"]),
            "validation_prevalence": float(metrics["prevalence"]),
            "validation_roc_auc": float(metrics["roc_auc"]),
            "validation_average_precision": float(metrics["average_precision"]),
            "validation_brier_score": float(metrics["brier_score"]),
            "validation_balanced_accuracy_at_0_5": float(metrics["balanced_accuracy_at_0_5"]),
        }
        rows.append(row)
        candidate_outputs.append((candidate_cfg, validation_prob, fitted_model, row))
    tuning_df = pd.DataFrame(rows)
    tuning_df = tuning_df.sort_values(
        ["validation_roc_auc", "validation_average_precision", "validation_brier_score"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    tuning_df["candidate_rank_within_model"] = np.arange(1, len(tuning_df) + 1)
    best_candidate_id = str(tuning_df.iloc[0]["candidate_id"])
    tuning_df["selected_by_validation"] = tuning_df["candidate_id"].eq(best_candidate_id).astype(int)
    best_index = next(i for i, (_, _, _, row) in enumerate(candidate_outputs) if row["candidate_id"] == best_candidate_id)
    best_cfg, best_validation_prob, best_train_only_model, _ = candidate_outputs[best_index]
    return best_cfg, tuning_df, best_validation_prob, best_train_only_model


def _build_split_summary(
    modeling_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    id_col: str,
) -> pd.DataFrame:
    rows = []
    participant_sets = {
        "training": set(modeling_df.iloc[train_idx][id_col]),
        "validation": set(modeling_df.iloc[validation_idx][id_col]),
        "held_out_test": set(modeling_df.iloc[test_idx][id_col]),
    }
    train_validation_overlap = int(len(participant_sets["training"] & participant_sets["validation"]))
    train_test_overlap = int(len(participant_sets["training"] & participant_sets["held_out_test"]))
    validation_test_overlap = int(len(participant_sets["validation"] & participant_sets["held_out_test"]))
    for split_name, idx in [
        ("training", train_idx),
        ("validation", validation_idx),
        ("held_out_test", test_idx),
        ("final_model_training_plus_validation", np.concatenate([train_idx, validation_idx])),
    ]:
        yy = y.iloc[idx]
        rows.append(
            {
                "split": split_name,
                "rows": int(len(idx)),
                "unique_participants": int(groups.iloc[idx].nunique()),
                "events": int(yy.sum()),
                "non_events": int(len(yy) - yy.sum()),
                "event_prevalence": float(yy.mean()),
                "train_validation_overlap_participants": train_validation_overlap,
                "train_test_overlap_participants": train_test_overlap,
                "validation_test_overlap_participants": validation_test_overlap,
                "participant_leakage_detected": int(
                    train_validation_overlap > 0 or train_test_overlap > 0 or validation_test_overlap > 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _fit_validation_calibrators(
    train_only_model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float],
    *,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Fit calibration maps on validation predictions and apply them to the held-out test set."""
    p_validation = predict_proba_positive(train_only_model, X_validation.copy())
    p_test_raw = predict_proba_positive(train_only_model, X_test.copy())
    if len(np.unique(y_validation.astype(int))) < 2:
        note = pd.DataFrame([{"status": "skipped", "reason": "Validation set did not contain both outcome classes."}])
        return {"performance": note, "predictions": note, "decision_curve": note}

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_validation, y_validation.astype(int).to_numpy())
    p_test_iso = np.clip(iso.transform(p_test_raw), 0.0, 1.0)

    platt = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
    platt.fit(p_validation.reshape(-1, 1), y_validation.astype(int).to_numpy())
    p_test_sigmoid = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    cal_probs = {
        "validation_uncalibrated_train_only_hgb": p_test_raw,
        "validation_isotonic_calibrated_hgb": p_test_iso,
        "validation_sigmoid_calibrated_hgb": p_test_sigmoid,
    }
    perf = build_metrics_table(cal_probs, y_test, threshold=threshold)
    perf["calibration_method"] = perf["model"].map(
        {
            "validation_uncalibrated_train_only_hgb": "none_train_only_model",
            "validation_isotonic_calibrated_hgb": "isotonic_validation_set_calibration",
            "validation_sigmoid_calibrated_hgb": "sigmoid_platt_validation_set_calibration",
        }
    )
    perf["calibration_validation_rows"] = int(len(y_validation))
    pred = pd.DataFrame(
        {
            "next_visit_dementia": y_test.astype(int).to_numpy(),
            "risk_validation_uncalibrated_train_only_hgb": p_test_raw,
            "risk_validation_isotonic_calibrated_hgb": p_test_iso,
            "risk_validation_sigmoid_calibrated_hgb": p_test_sigmoid,
        }
    )
    dca = _decision_curve(cal_probs, y_test, thresholds)
    return {"performance": perf, "predictions": pred, "decision_curve": dca}



def _fit_internal_calibrators(
    cfg: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_lists: dict[str, list[str]],
    thresholds: list[float],
    *,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Train internal calibration models for the final medication-aware HGB risk score.

    The final manuscript model remains the primary uncalibrated holdout model. This
    function uses a training-only subtrain/calibration split to estimate isotonic and
    sigmoid recalibration maps, then applies them once to the untouched holdout set.
    """
    cal_cfg = cfg.get("validation", {}).get("calibration", {})
    fraction = float(cal_cfg.get("calibration_fraction", 0.20))
    if not 0.05 <= fraction <= 0.50:
        fraction = 0.20

    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=int(random_state) + 211)
    rel_subtrain, rel_cal = next(splitter.split(X_train, y_train, groups=groups_train))
    if len(np.unique(y_train.iloc[rel_subtrain])) < 2 or len(np.unique(y_train.iloc[rel_cal])) < 2:
        note = pd.DataFrame([{"status": "skipped", "reason": "Calibration split did not contain both outcome classes."}])
        return {"performance": note, "predictions": note, "decision_curve": note}

    base = build_hist_gradient_pipeline(build_preprocessor(feature_lists, scale_numeric=False), cfg)
    use_hgb_weights = bool(cfg.get("modeling", {}).get("hist_gradient_boosting", {}).get("use_balanced_sample_weight", True))
    base = fit_pipeline(
        "medication_aware_hist_gradient_boosting",
        base,
        X_train.iloc[rel_subtrain].copy(),
        y_train.iloc[rel_subtrain].copy(),
        use_balanced_sample_weight=use_hgb_weights,
    )
    p_cal = predict_proba_positive(base, X_train.iloc[rel_cal].copy())
    p_test_raw = predict_proba_positive(base, X_test)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_cal, y_train.iloc[rel_cal].astype(int).to_numpy())
    p_test_iso = np.clip(iso.transform(p_test_raw), 0.0, 1.0)

    platt = LogisticRegression(C=1.0, solver="liblinear", random_state=int(random_state))
    platt.fit(p_cal.reshape(-1, 1), y_train.iloc[rel_cal].astype(int).to_numpy())
    p_test_sigmoid = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    cal_probs = {
        "internal_uncalibrated_subtrain_hgb": p_test_raw,
        "isotonic_calibrated_hgb": p_test_iso,
        "sigmoid_calibrated_hgb": p_test_sigmoid,
    }
    perf = build_metrics_table(cal_probs, y_test, threshold=float(cfg.get("modeling", {}).get("threshold", 0.50)))
    perf["calibration_method"] = perf["model"].map(
        {
            "internal_uncalibrated_subtrain_hgb": "none_internal_subtrain_model",
            "isotonic_calibrated_hgb": "isotonic_training_only_calibration",
            "sigmoid_calibrated_hgb": "sigmoid_platt_training_only_calibration",
        }
    )
    perf["calibration_train_rows"] = int(len(rel_cal))
    perf["model_train_rows"] = int(len(rel_subtrain))

    pred = pd.DataFrame(
        {
            "next_visit_dementia": y_test.astype(int).to_numpy(),
            "risk_internal_uncalibrated_subtrain_hgb": p_test_raw,
            "risk_isotonic_calibrated_hgb": p_test_iso,
            "risk_sigmoid_calibrated_hgb": p_test_sigmoid,
        }
    )
    dca = _decision_curve(cal_probs, y_test, thresholds)
    return {"performance": perf, "predictions": pred, "decision_curve": dca}


def _grouped_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: dict[str, list[str]],
    *,
    max_rows: int,
    n_repeats: int,
    random_state: int,
    progress_enabled: bool = False,
) -> pd.DataFrame:
    if len(X_test) > int(max_rows):
        sample = X_test.sample(n=int(max_rows), random_state=int(random_state))
        y_sample = y_test.loc[sample.index]
    else:
        sample = X_test.copy()
        y_sample = y_test.copy()
    y_arr = y_sample.astype(int).to_numpy()
    baseline_prob = predict_proba_positive(model, sample)
    baseline_auc = float(roc_auc_score(y_arr, baseline_prob)) if len(np.unique(y_arr)) == 2 else np.nan
    rng = np.random.default_rng(int(random_state) + 911)
    rows = []
    for group_name, cols in progress_iter(groups.items(), enabled=progress_enabled, desc="Grouped permutation", total=len(groups), unit="group"):
        cols = [c for c in cols if c in sample.columns]
        if not cols:
            continue
        drops = []
        for _ in progress_iter(range(int(n_repeats)), enabled=progress_enabled, desc=f"Permute {group_name}", total=int(n_repeats), unit="rep", leave=False):
            permuted = sample.copy()
            perm_index = rng.permutation(len(permuted))
            permuted.loc[:, cols] = permuted.loc[permuted.index[perm_index], cols].to_numpy()
            prob = predict_proba_positive(model, permuted)
            auc = float(roc_auc_score(y_arr, prob)) if len(np.unique(y_arr)) == 2 else np.nan
            drops.append(baseline_auc - auc)
        rows.append(
            {
                "feature_group": group_name,
                "n_features_in_group": int(len(cols)),
                "features": "; ".join(cols),
                "baseline_auc": baseline_auc,
                "importance_mean_auc_drop": float(np.nanmean(drops)),
                "importance_sd": float(np.nanstd(drops, ddof=1)) if len(drops) > 1 else 0.0,
                "n_repeats": int(n_repeats),
                "n_rows_used": int(len(sample)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean_auc_drop", ascending=False).reset_index(drop=True)


def _default_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    med = structured_medication_feature_columns(df)
    llm_med = llm_medication_feature_columns(df)
    neural_med = neural_text_feature_columns(df)
    return {
        "cognitive_clinical_severity": ["NACCMMSE", "NACCMOCA", "CDRSUM", "CDRGLOB", "NACCGDS"],
        "age_visit_time": [
            "NACCAGE", "NACCDAYS", "NACCVNUM", "VISITYR", "prior_mci_visit_count",
            "days_since_prior_mci_visit", "days_since_first_mci_visit",
        ],
        "demographics_genetic_context": ["SEX", "HISPANIC", "RACE", "EDUC", "NACCAPOE"],
        "neuropsychiatric_symptoms": ["DEL", "HALL", "AGIT", "DEPD", "ANX", "ELAT", "APA", "DISN", "IRR", "MOT", "NITE", "APP", "DEP2YRS", "NPSYDEV", "PSYCDIS", "BIPOLDX", "SCHIZOP", "PTSDDX"],
        "comorbidity_functional_context": ["PARK", "GAITNPH", "HYCEPH", "BRNINJ", "SMOKYRS", "PACKSPER", "QUITSMOK", "ALCOCCAS", "ALCFREQ", "CVHATT", "CBSTROKE", "DIABETES", "HYPERTEN", "HYPERCHO", "B12DEF", "THYROID", "ARTH", "INCONTU", "INCONTF"],
        "structured_medication_state": med,
        "llm_enhanced_medication_state": llm_med,
        "bioclinicalbert_sapbert_neural_medication_state": neural_med,
        "longitudinal_patient_state_trajectory": longitudinal_trajectory_feature_columns(df),
    }


def _temporal_validation(
    cfg: dict[str, Any],
    clinical_X: pd.DataFrame,
    med_X: pd.DataFrame,
    llm_X: pd.DataFrame,
    y: pd.Series,
    modeling_df: pd.DataFrame,
    clinical_features: dict[str, list[str]],
    medication_features: dict[str, list[str]],
    llm_medication_features: dict[str, list[str]],
    *,
    trajectory_X: pd.DataFrame | None = None,
    trajectory_features: dict[str, list[str]] | None = None,
    threshold: float,
) -> pd.DataFrame:
    tv_cfg = cfg.get("validation", {}).get("temporal_validation", {})
    split_year = int(tv_cfg.get("test_min_year", 2019))
    year_col = str(tv_cfg.get("year_col", "VISITYR"))
    min_test_events = int(tv_cfg.get("min_test_events", 50))
    min_train_events = int(tv_cfg.get("min_train_events", 100))
    if year_col not in modeling_df.columns:
        return pd.DataFrame([{"status": "skipped", "reason": f"{year_col} not available"}])
    years = pd.to_numeric(modeling_df[year_col], errors="coerce")
    train_mask = years < split_year
    test_mask = years >= split_year
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    if len(y_train) == 0 or len(y_test) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
        return pd.DataFrame([{"status": "skipped", "reason": "Temporal split did not contain both classes in train and test.", "test_min_year": split_year}])
    if int(y_train.sum()) < min_train_events or int(y_test.sum()) < min_test_events:
        return pd.DataFrame([{"status": "skipped", "reason": "Temporal split had too few outcome events.", "test_min_year": split_year, "train_events": int(y_train.sum()), "test_events": int(y_test.sum())}])

    use_hgb_weights = bool(cfg.get("modeling", {}).get("hist_gradient_boosting", {}).get("use_balanced_sample_weight", True))
    temporal_models = {
        "temporal_clinical_only_logistic": (build_logistic_pipeline(build_preprocessor(clinical_features, scale_numeric=True), cfg), clinical_X.loc[train_mask], clinical_X.loc[test_mask]),
        "temporal_structured_medication_aware_logistic": (build_logistic_pipeline(build_preprocessor(medication_features, scale_numeric=True), cfg), med_X.loc[train_mask], med_X.loc[test_mask]),
        "temporal_llm_enhanced_medication_aware_logistic": (build_logistic_pipeline(build_preprocessor(llm_medication_features, scale_numeric=True), cfg), llm_X.loc[train_mask], llm_X.loc[test_mask]),
        "temporal_clinical_only_hist_gradient_boosting": (build_hist_gradient_pipeline(build_preprocessor(clinical_features, scale_numeric=False), cfg), clinical_X.loc[train_mask], clinical_X.loc[test_mask]),
        "temporal_structured_medication_aware_hist_gradient_boosting": (build_hist_gradient_pipeline(build_preprocessor(medication_features, scale_numeric=False), cfg), med_X.loc[train_mask], med_X.loc[test_mask]),
        "temporal_llm_enhanced_medication_aware_hist_gradient_boosting": (build_hist_gradient_pipeline(build_preprocessor(llm_medication_features, scale_numeric=False), cfg), llm_X.loc[train_mask], llm_X.loc[test_mask]),
    }
    if trajectory_X is not None and trajectory_features is not None and longitudinal_trajectory_feature_columns(modeling_df):
        temporal_models["temporal_longitudinal_llm_medication_trajectory_logistic"] = (
            build_logistic_pipeline(build_preprocessor(trajectory_features, scale_numeric=True), cfg),
            trajectory_X.loc[train_mask],
            trajectory_X.loc[test_mask],
        )
        temporal_models["temporal_longitudinal_llm_medication_trajectory_hist_gradient_boosting"] = (
            build_hist_gradient_pipeline(build_preprocessor(trajectory_features, scale_numeric=False), cfg),
            trajectory_X.loc[train_mask],
            trajectory_X.loc[test_mask],
        )
    probs: dict[str, np.ndarray] = {}
    for name, (model, Xtr, Xte) in temporal_models.items():
        fit_name = name.replace("temporal_", "")
        fitted_model = fit_pipeline(fit_name, model, Xtr.copy(), y_train.copy(), use_balanced_sample_weight=use_hgb_weights)
        probs[name] = predict_proba_positive(fitted_model, Xte.copy())
    out = build_metrics_table(probs, y_test, threshold=threshold)
    out["temporal_split_year_rule"] = f"train_{year_col}_lt_{split_year}_test_{year_col}_ge_{split_year}"
    out["train_rows"] = int(train_mask.sum())
    out["test_rows"] = int(test_mask.sum())
    out["train_events"] = int(y_train.sum())
    out["test_events"] = int(y_test.sum())
    out["train_unique_participants"] = int(modeling_df.loc[train_mask, cfg["id_col"]].nunique())
    out["test_unique_participants"] = int(modeling_df.loc[test_mask, cfg["id_col"]].nunique())
    return out

def _subgroup_metrics(predictions: pd.DataFrame, *, outcome_col: str, risk_col: str, group_cols: list[str], min_n: int, threshold: float) -> pd.DataFrame:
    rows = []
    for group_col in group_cols:
        if group_col not in predictions.columns:
            continue
        for group_value, chunk in predictions.groupby(group_col, dropna=False):
            if len(chunk) < int(min_n) or chunk[outcome_col].nunique(dropna=True) < 2:
                continue
            metrics = evaluate_binary_predictions(chunk[outcome_col], chunk[risk_col].to_numpy(), threshold=threshold)
            metrics["group_column"] = group_col
            metrics["group_value"] = group_value
            rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["group_column", "group_value"]).reset_index(drop=True)


def _run_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    max_rows: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    if len(X_test) > int(max_rows):
        sample = X_test.sample(n=int(max_rows), random_state=int(random_state))
        y_sample = y_test.loc[sample.index]
    else:
        sample = X_test
        y_sample = y_test
    result = permutation_importance(
        model,
        sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(sample.columns),
                "importance_mean_auc_drop": result.importances_mean,
                "importance_sd": result.importances_std,
                "n_repeats": int(n_repeats),
                "n_rows_used": int(len(sample)),
            }
        )
        .sort_values("importance_mean_auc_drop", ascending=False)
        .reset_index(drop=True)
    )


def _neuropathology_outcomes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    definitions: dict[str, str] = {}

    def num(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)

    if "NPADNC" in out.columns:
        s = num("NPADNC")
        out["np_high_adnc"] = np.where(s.notna(), (s >= 3).astype(float), np.nan)
        definitions["np_high_adnc"] = "High Alzheimer disease neuropathologic change, defined from NPADNC >= 3 when available."
    if "NACCBRAA" in out.columns:
        s = num("NACCBRAA")
        out["np_high_braak"] = np.where(s.notna(), (s >= 5).astype(float), np.nan)
        definitions["np_high_braak"] = "High Braak neurofibrillary stage, defined from NACCBRAA >= 5 when available."
    if "NACCNEUR" in out.columns:
        s = num("NACCNEUR")
        out["np_moderate_frequent_neuritic_plaques"] = np.where(s.notna(), (s >= 2).astype(float), np.nan)
        definitions["np_moderate_frequent_neuritic_plaques"] = "Moderate/frequent neuritic plaques, defined from NACCNEUR >= 2 when available."
    if "NACCLEWY" in out.columns:
        s = num("NACCLEWY")
        out["np_lewy_body_pathology_present"] = np.where(s.notna(), (s > 0).astype(float), np.nan)
        definitions["np_lewy_body_pathology_present"] = "Lewy body pathology present, defined from NACCLEWY > 0 when available."
    if "NPINF" in out.columns:
        s = num("NPINF")
        out["np_infarct_present"] = np.where(s.notna(), (s > 0).astype(float), np.nan)
        definitions["np_infarct_present"] = "Neuropathologic infarct present, defined from NPINF > 0 when available."
    return out, definitions


def _neuropathology_anchor_outputs(predictions_with_np: pd.DataFrame, *, final_risk_col: str, clinical_risk_col: str) -> dict[str, pd.DataFrame | str]:
    np_df, definitions = _neuropathology_outcomes(predictions_with_np)
    outcome_cols = list(definitions.keys())
    available = np_df[outcome_cols].notna().any(axis=1) if outcome_cols else pd.Series(False, index=np_df.index)
    anchor = np_df.loc[available].copy()
    summary_rows = []
    assoc_rows = []
    quartile_rows = []
    if not anchor.empty:
        try:
            anchor["final_model_risk_quartile"] = pd.qcut(anchor[final_risk_col], q=4, labels=False, duplicates="drop") + 1
        except ValueError:
            anchor["final_model_risk_quartile"] = 1
        for outcome in outcome_cols:
            valid = anchor.loc[anchor[outcome].notna()].copy()
            if valid.empty:
                continue
            y = valid[outcome].astype(int).to_numpy()
            summary_rows.append(
                {
                    "outcome": outcome,
                    "definition": definitions[outcome],
                    "n_with_outcome_observed": int(len(valid)),
                    "n_positive": int(y.sum()),
                    "outcome_prevalence": float(y.mean()),
                }
            )
            for risk_col, label in [(final_risk_col, "medication_aware_hgb_risk"), (clinical_risk_col, "clinical_only_hgb_risk")]:
                risk = valid[risk_col].astype(float).to_numpy()
                row = {
                    "outcome": outcome,
                    "risk_score": label,
                    "n": int(len(valid)),
                    "outcome_prevalence": float(y.mean()),
                    "mean_risk_if_outcome_negative": float(np.nanmean(risk[y == 0])) if np.any(y == 0) else np.nan,
                    "mean_risk_if_outcome_positive": float(np.nanmean(risk[y == 1])) if np.any(y == 1) else np.nan,
                    "mean_risk_difference_positive_minus_negative": float(np.nanmean(risk[y == 1]) - np.nanmean(risk[y == 0])) if np.any(y == 1) and np.any(y == 0) else np.nan,
                }
                if len(np.unique(y)) == 2:
                    row["roc_auc_for_neuropathology_outcome"] = float(roc_auc_score(y, risk))
                    row["average_precision_for_neuropathology_outcome"] = float(average_precision_score(y, risk))
                else:
                    row["roc_auc_for_neuropathology_outcome"] = np.nan
                    row["average_precision_for_neuropathology_outcome"] = np.nan
                assoc_rows.append(row)
            for q, chunk in valid.groupby("final_model_risk_quartile", dropna=False):
                quartile_rows.append(
                    {
                        "outcome": outcome,
                        "final_model_risk_quartile": int(q) if pd.notna(q) else -1,
                        "n": int(len(chunk)),
                        "n_positive": int(chunk[outcome].astype(int).sum()),
                        "outcome_rate": float(chunk[outcome].astype(float).mean()),
                        "mean_final_model_risk": float(chunk[final_risk_col].astype(float).mean()),
                    }
                )
    interpretation = [
        "# Neuropathology anchoring interpretation",
        "",
        "Neuropathology variables are used only as secondary biological anchoring outcomes in the autopsy/neuropathology subset. They are not included in any predictive feature set for next-visit dementia progression, which prevents postmortem-label leakage into the prospective clinical prediction task.",
        "",
    ]
    if anchor.empty:
        interpretation.append("No usable neuropathology anchor subset was available after applying the analytic/test cohort restrictions.")
    else:
        interpretation.append(
            f"The neuropathology anchor subset included {len(anchor):,} held-out MCI visits with at least one available neuropathology-derived outcome. These analyses should be interpreted as biological plausibility checks, not as causal medication-effect estimates."
        )
        interpretation.append("")
        interpretation.append("Key outputs: `s6_neuropathology_anchor_analysis/s6b_neuropathology_anchor_outcome_summary.csv`, `s6_neuropathology_anchor_analysis/s6c_neuropathology_anchor_risk_associations.csv`, and `s6_neuropathology_anchor_analysis/s6d_neuropathology_anchor_risk_quartiles.csv`.")
    summary_columns = [
        "outcome", "definition", "n_with_outcome_observed", "n_positive", "outcome_prevalence"
    ]
    association_columns = [
        "outcome", "risk_score", "n", "outcome_prevalence",
        "mean_risk_if_outcome_negative", "mean_risk_if_outcome_positive",
        "mean_risk_difference_positive_minus_negative",
        "roc_auc_for_neuropathology_outcome", "average_precision_for_neuropathology_outcome",
    ]
    quartile_columns = [
        "outcome", "final_model_risk_quartile", "n", "n_positive",
        "outcome_rate", "mean_final_model_risk",
    ]
    return {
        "anchor_cohort": anchor,
        "summary": pd.DataFrame(summary_rows, columns=summary_columns),
        "associations": pd.DataFrame(assoc_rows, columns=association_columns),
        "quartiles": pd.DataFrame(quartile_rows, columns=quartile_columns),
        "interpretation": "\n".join(interpretation) + "\n",
    }


def _make_publication_findings(metrics_df: pd.DataFrame, delta_df: pd.DataFrame, metadata: dict[str, Any]) -> str:
    best = metrics_df.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]
    lines = [
        "# Publication main findings",
        "",
        f"The analytic cohort included {metadata['cohort']['rows']:,} MCI visits from {metadata['cohort']['unique_participants']:,} participants, with {metadata['cohort']['events']:,} next-visit dementia progression events (prevalence {metadata['cohort']['prevalence']:.3f}).",
        "",
        f"The best-performing model by AUROC was `{best['model']}`, with AUROC={best['roc_auc']:.3f}, average precision={best['average_precision']:.3f}, Brier score={best['brier_score']:.3f}, and balanced accuracy={best['balanced_accuracy_at_0_5']:.3f}.",
        "",
        "The final manuscript model set includes clinical-only, structured medication-aware, and LLM-enhanced medication-aware representations under matched logistic regression and hist-gradient boosting model families.",
        "",
        "Neuropathology variables are not used as predictors. They are used only for secondary biological anchoring in the subset with available postmortem variables.",
        "",
        "Structured and LLM-enhanced medication-state value should be interpreted as incremental predictive information and clinically reviewable patient-state representation, not as evidence that medication exposure causes dementia progression.",
        "",
        "## Medication-feature deltas",
        "",
    ]
    for _, row in delta_df.iterrows():
        lines.append(
            f"- {row['comparison']}: AUROC delta={row['delta_roc_auc']:.3f}, average-precision delta={row['delta_average_precision']:.3f}, Brier-score delta={row['delta_brier_score']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Recommended claim",
            "",
            "Structured and LLM-enhanced medication-state features are evaluated as incremental patient-state representations beyond the clinical-only representation, which included demographic, cognitive, clinical severity, neuropsychiatric, comorbidity, and visit-time features. Logistic regression provides an interpretable estimate of feature-representation contribution, while tuned hist-gradient boosting provides the final nonlinear machine-learning model for optimized predictive performance. Neuropathology anchoring provides a secondary biological plausibility analysis without contaminating the prospective prediction task.",
        ]
    )
    return "\n".join(lines) + "\n"



def output_file_synopsis_frame() -> pd.DataFrame:
    """Return a manuscript-friendly synopsis table for all pipeline outputs."""
    df = pd.DataFrame(
        [{"output_file": public_output_name(name), "synopsis": synopsis} for name, synopsis in OUTPUT_FILE_SYNOPSIS.items()]
    )
    return df.drop_duplicates(subset=["output_file"], keep="first").reset_index(drop=True)


def project_file_synopsis_frame() -> pd.DataFrame:
    """Return a compact synopsis table for major project files."""
    return pd.DataFrame(
        [{"project_file": name, "synopsis": synopsis} for name, synopsis in PROJECT_FILE_SYNOPSIS.items()]
    )




def _blankish_series(series: pd.Series) -> pd.Series:
    """Return True for missing, empty, whitespace-only, or blank-space values."""
    text = series.astype("string").str.strip().fillna("")
    return series.isna() | text.eq("")


def _step1_required_columns(cfg: dict[str, Any]) -> list[str]:
    """Columns that must be nonblank for a row to enter downstream analysis."""
    configured = (cfg.get("step1", {}) or {}).get("required_columns_for_row_filter")
    if configured:
        return unique_preserve_order([str(c) for c in configured])
    return unique_preserve_order([cfg["id_col"], cfg["visit_number_col"], cfg["visit_year_col"], cfg["current_diagnosis_col"]])


def _filter_rows_missing_required_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required Step 1 row-filter columns are missing from loaded input: {missing_cols}")
    keep = pd.Series(True, index=df.index)
    missing_by_col: dict[str, int] = {}
    for col in required_columns:
        blank = _blankish_series(df[col])
        missing_by_col[col] = int(blank.sum())
        keep &= ~blank
    filtered = df.loc[keep].copy().reset_index(drop=True)
    audit_rows = []
    for col in required_columns:
        audit_rows.append({"required_column": col, "rows_missing_or_blank": missing_by_col[col]})
    audit_rows.append({"required_column": "ANY_REQUIRED_COLUMN", "rows_missing_or_blank": int((~keep).sum())})
    return filtered, pd.DataFrame(audit_rows)


def _drop_all_blank_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are entirely missing, blank, whitespace, or blank-space."""
    dropped: list[str] = []
    keep_cols: list[str] = []
    for col in df.columns:
        if bool(_blankish_series(df[col]).all()):
            dropped.append(col)
        else:
            keep_cols.append(col)
    return df.loc[:, keep_cols].copy(), dropped


def _column_max_size(df: pd.DataFrame, col: str) -> int | str:
    if col not in df.columns:
        return ""
    values = df[col].dropna().astype(str)
    if values.empty:
        return 0
    return int(values.map(len).max())


def _source_column_description(col: str, status: str) -> str:
    """Return a conservative fallback description when no official NACC text exists."""
    descriptions = {
        "NACCID": "NACC participant identifier used for grouped splitting and leakage prevention.",
        "NACCVNUM": "NACC visit number used for longitudinal visit ordering.",
        "VISITYR": "Visit year used for longitudinal ordering and temporal validation.",
        "VISITMO": "Visit month used for longitudinal visit ordering.",
        "VISITDAY": "Visit day used for longitudinal visit ordering.",
        "Last visit": "Source indicator for last recorded visit, retained for audit and source compatibility.",
        "NACCUDSD": "Current visit cognitive diagnosis code used for MCI cohort construction and next-visit dementia labeling.",
    }
    if col in descriptions:
        return descriptions[col]
    if str(col).upper().startswith("DRUG"):
        return "Source medication text field used to construct structured and optional LLM-enhanced medication-state features."
    if status == "included_final":
        return "Loaded from the source CSV and retained in the final Step 1 downstream input file; official NACC description was not found in resources/nacc_official/*.csv."
    if status == "excluded_all_blank_after_row_filter":
        return "Loaded from the source CSV but removed from the final Step 1 file because all retained rows were null, blank, whitespace, or blank-space only; official NACC description was not found in resources/nacc_official/*.csv."
    return "Available in the source CSV but not selected by the YAML-controlled pipeline input column list; official NACC description was not found in resources/nacc_official/*.csv."


def _canonical_dictionary_header(value: Any) -> str:
    """Normalize dictionary-column headers so heterogeneous NACC CSVs can be parsed."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _choose_dictionary_column(columns: list[str], candidates: list[str]) -> str | None:
    """Select the first matching column from an official NACC dictionary file."""
    normalized = {_canonical_dictionary_header(c): c for c in columns}
    for candidate in candidates:
        key = _canonical_dictionary_header(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = _canonical_dictionary_header(candidate)
        for norm, original in normalized.items():
            if key and key in norm:
                return original
    return None



DEFAULT_NACC_OFFICIAL_DICTIONARY_DOWNLOADS = [
    {
        "name": "uds_rdd_csv",
        "filename": "uds_rdd.csv",
        "url": "https://files.alz.washington.edu/documentation/uds4-rdd.csv",
        "required": True,
    },
    {
        "name": "np_rdd_csv",
        "filename": "np_rdd.csv",
        "url": "https://files.alz.washington.edu/documentation/rdd-np.csv",
        "required": True,
    },
    {
        "name": "genetics_rdd_csv",
        "filename": "genetics_rdd.csv",
        "url": "https://files.alz.washington.edu/documentation/rdd-gen.csv",
        "required": False,
    },
]


def _resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    out = Path(path).expanduser()
    if out.is_absolute():
        return out
    base = Path(project_root).resolve() if project_root else PROJECT_ROOT
    return base / out


def _csv_file_has_rows(path: Path) -> bool:
    """Return True when a path appears to be a nonempty CSV text file."""
    try:
        if not path.exists() or path.stat().st_size <= 20:
            return False
        head = path.read_bytes()[:4096].lower()
        if b"<html" in head or b"<!doctype html" in head:
            return False
        if b"," not in head and b"variable" not in head:
            return False
        return True
    except Exception:
        return False


def ensure_official_nacc_dictionary_files(
    dictionary_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    autodownload_cfg: dict[str, Any] | None = None,
    logger=None,
) -> pd.DataFrame:
    """Optionally download official NACC dictionary CSV files when missing.

    Download behavior is controlled entirely by YAML. When disabled, this function
    only reports existing files. When enabled, it downloads only missing/nonempty
    CSVs into ``resources/nacc_official`` and never silently fabricates dictionary
    content. If a required file cannot be downloaded and ``fail_on_error`` is true,
    a RuntimeError is raised so the analysis cannot proceed under the false
    assumption that official dictionary rules were active.
    """
    cfg = autodownload_cfg or {}
    path = _resolve_project_path(dictionary_dir, project_root=project_root)
    path.mkdir(parents=True, exist_ok=True)
    entries = cfg.get("files") or DEFAULT_NACC_OFFICIAL_DICTIONARY_DOWNLOADS
    enabled = bool(cfg.get("enabled", False))
    overwrite = bool(cfg.get("overwrite_existing", False))
    timeout = float(cfg.get("timeout_seconds", 30) or 30)
    fail_on_error = bool(cfg.get("fail_on_error", False))
    user_agent = str(cfg.get("user_agent", "dementia_progression_pipeline/2.12"))
    rows: list[dict[str, Any]] = []

    for item in entries:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("filename") or "nacc_dictionary")
        filename = str(item.get("filename") or f"{name}.csv")
        url = str(item.get("url") or "")
        required = bool(item.get("required", False))
        dest = path / filename
        existed_before = dest.exists()
        valid_before = _csv_file_has_rows(dest)
        status = "present" if valid_before else "missing"
        error = ""
        downloaded = False

        if enabled and (overwrite or not valid_before):
            if not url:
                status = "no_url_configured"
                error = "download URL not configured"
            else:
                tmp = dest.with_name(f".{dest.name}.download.{os.getpid()}")
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        payload = response.read()
                    if len(payload) <= 20:
                        raise RuntimeError("downloaded file is unexpectedly small")
                    head = payload[:4096].lower()
                    if b"<html" in head or b"<!doctype html" in head:
                        raise RuntimeError("download returned HTML instead of CSV")
                    tmp.write_bytes(payload)
                    if not _csv_file_has_rows(tmp):
                        raise RuntimeError("downloaded file did not validate as a CSV dictionary")
                    os.replace(tmp, dest)
                    downloaded = True
                    status = "downloaded"
                except Exception as exc:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    status = "download_failed"
                    error = f"{type(exc).__name__}: {exc}"
                    if logger:
                        logger.warning("Official NACC dictionary auto-download failed | name=%s | url=%s | error=%s", name, url, error)
                    if required and fail_on_error:
                        raise RuntimeError(
                            f"Required official NACC dictionary file could not be downloaded: {name} from {url}. {error}"
                        ) from exc

        rows.append({
            "dictionary_name": name,
            "filename": filename,
            "url": url,
            "required": int(required),
            "autodownload_enabled": int(enabled),
            "existed_before": int(existed_before),
            "valid_before": int(valid_before),
            "downloaded": int(downloaded),
            "exists_after": int(dest.exists()),
            "valid_after": int(_csv_file_has_rows(dest)),
            "size_bytes": int(dest.stat().st_size) if dest.exists() else 0,
            "status": status,
            "error": error,
        })
    audit = pd.DataFrame(rows)
    if logger:
        logger.info(
            "Official NACC dictionary file check completed | dir=%s | autodownload=%s | files_present=%s/%s",
            path,
            enabled,
            int(audit["valid_after"].sum()) if not audit.empty else 0,
            len(audit),
        )
    return audit


def load_official_nacc_column_descriptions(
    dictionary_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    logger: Any | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load official NACC variable descriptions from every CSV in resources/nacc_official.

    The NACC dictionary files are not guaranteed to use identical headers across
    releases/forms. This parser therefore accepts common variants for the
    variable-name column and the description/label column. Descriptions are
    keyed by upper-case variable name so that `NACCID` and `naccid` resolve to
    the same official text.
    """
    base = Path(project_root) if project_root is not None else PROJECT_ROOT
    path = Path(dictionary_dir)
    if not path.is_absolute():
        path = (base / path).resolve()

    descriptions: dict[str, str] = {}
    sources: dict[str, str] = {}
    if not path.exists():
        if logger is not None:
            logger.warning("Official NACC dictionary directory not found: %s", path)
        return descriptions, sources

    name_candidates = [
        "Col_Name", "Column", "Column Name", "Variable", "Variable Name",
        "VARNAME", "VAR_NAME", "Name", "Field", "Field Name",
        "Data Element", "Data Element Name", "NACC Variable", "UDS Variable",
    ]
    description_candidates = [
        "Col_Descr", "Column Description", "Description", "Variable Description",
        "Variable Label", "Label", "Definition", "Question", "Question Text",
        "Data Element Description", "Item Description", "Short Description",
        "Long Description", "Text", "Title",
    ]

    for csv_path in sorted(path.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path, dtype=str, low_memory=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(csv_path, dtype=str, low_memory=False, encoding="latin1")
        except Exception as exc:  # pragma: no cover - defensive logging only
            if logger is not None:
                logger.warning("Could not read official NACC dictionary CSV %s: %s", csv_path, exc)
            continue

        if frame.empty:
            continue
        name_col = _choose_dictionary_column(list(frame.columns), name_candidates)
        descr_col = _choose_dictionary_column(list(frame.columns), description_candidates)
        if name_col is None or descr_col is None:
            if logger is not None:
                logger.warning(
                    "Skipped official NACC dictionary CSV without identifiable variable/description columns: %s",
                    csv_path,
                )
            continue

        for _, row in frame.iterrows():
            name = str(row.get(name_col, "")).strip()
            description = str(row.get(descr_col, "")).strip()
            if not name or name.lower() in {"nan", "none"}:
                continue
            if not description or description.lower() in {"nan", "none"}:
                continue
            key = name.upper()
            # Preserve the first official description encountered to avoid
            # nondeterministic changes when a variable appears in multiple form files.
            if key not in descriptions:
                descriptions[key] = description
                sources[key] = str(csv_path.relative_to(base)) if csv_path.is_relative_to(base) else str(csv_path)

    if logger is not None:
        logger.info(
            "Loaded official NACC column descriptions: dictionary_dir=%s descriptions=%s",
            path,
            len(descriptions),
        )
    return descriptions, sources


def _official_or_fallback_column_description(
    col: str,
    status: str,
    official_descriptions: dict[str, str] | None,
    official_sources: dict[str, str] | None,
) -> tuple[str, str, str, str]:
    """Return description text, source type, source file, and match status.

    `Col_Descr` should contain the actual official NACC variable description
    whenever a matching dictionary entry is available. Fallback text is used
    only when the variable truly cannot be found in the official dictionary
    files supplied under resources/nacc_official/*.csv.
    """
    col_text = str(col)
    upper_key = col_text.upper()
    if official_descriptions:
        if col_text in official_descriptions:
            return (
                official_descriptions[col_text],
                "official_nacc_dictionary",
                (official_sources or {}).get(col_text, "resources/nacc_official/*.csv"),
                "matched_exact",
            )
        if upper_key in official_descriptions:
            match_status = "matched_exact" if col_text == upper_key else "matched_case_insensitive"
            return (
                official_descriptions[upper_key],
                "official_nacc_dictionary",
                (official_sources or {}).get(upper_key, "resources/nacc_official/*.csv"),
                match_status,
            )
    return _source_column_description(col, status), "pipeline_fallback", "", "not_found"


def _build_step1_column_audit(
    *,
    source_header: list[str],
    selected_usecols: list[str],
    row_filtered_df: pd.DataFrame,
    final_df: pd.DataFrame,
    dropped_all_blank_columns: list[str],
    official_descriptions: dict[str, str] | None = None,
    official_description_sources: dict[str, str] | None = None,
) -> pd.DataFrame:
    selected_set = set(selected_usecols)
    final_set = set(final_df.columns)
    dropped_set = set(dropped_all_blank_columns)
    rows: list[dict[str, Any]] = []
    for i, col in enumerate(source_header, start=1):
        if col in final_set:
            status = "included_final"
            dtype = str(final_df[col].dtype)
            max_size = _column_max_size(final_df, col)
        elif col in selected_set and col in dropped_set:
            status = "excluded_all_blank_after_row_filter"
            dtype = str(row_filtered_df[col].dtype) if col in row_filtered_df.columns else ""
            max_size = _column_max_size(row_filtered_df, col)
        elif col in selected_set:
            status = "excluded_after_step1_row_filter"
            dtype = str(row_filtered_df[col].dtype) if col in row_filtered_df.columns else ""
            max_size = _column_max_size(row_filtered_df, col)
        else:
            status = "excluded_not_selected_by_yaml"
            dtype = "not_loaded"
            max_size = "not_loaded"
        col_descr, col_descr_source, col_descr_source_file, col_descr_match_status = _official_or_fallback_column_description(
            col, status, official_descriptions, official_description_sources
        )
        rows.append(
            {
                "#": i,
                "Col_Name": col,
                "Col_Included_Excluded": status,
                "Col_DataType": dtype,
                "Col_DataMaxSize": max_size,
                "Col_Descr": col_descr,
                "Col_Descr_Source": col_descr_source,
                "Col_Descr_Source_File": col_descr_source_file,
                "Col_Descr_Match_Status": col_descr_match_status,
            }
        )
    return pd.DataFrame(rows)

def _modeling_eligibility_reason(y: pd.Series, groups: pd.Series) -> str | None:
    """Return None when the cohort can support participant-level train-validation-test modeling.

    Small YAML-selected raw-row windows are useful for testing I/O and cohort creation,
    but they may produce zero eligible MCI visits or too few outcome events for a valid
    three-way participant-level split. In those cases the run should stop cleanly with
    audit outputs rather than crashing inside scikit-learn.
    """
    n_rows = int(len(y))
    n_participants = int(pd.Series(groups).nunique()) if n_rows else 0
    if n_rows == 0:
        return "No eligible MCI visits with a subsequent visit were available after applying the YAML raw-row selection."
    if n_participants < 3:
        return f"Only {n_participants} participant(s) were available; at least 3 participants are required for train-validation-test splitting."
    y_nonmissing = pd.Series(y).dropna().astype(int)
    if y_nonmissing.nunique() < 2:
        return "The selected analytic cohort contains only one outcome class; supervised binary model development requires both events and non-events."
    class_counts = y_nonmissing.value_counts().to_dict()
    min_class_count = int(min(class_counts.values())) if class_counts else 0
    if min_class_count < 3:
        return f"The smaller outcome class contains only {min_class_count} row(s); at least 3 are required to support train, validation, and held-out test partitions."
    return None


def _write_graceful_stop_outputs(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    decimal_places: int,
    reason: str,
    record_selection: dict[str, Any],
    raw_rows_loaded: int,
    cohort_summary: dict[str, Any] | None = None,
    modeling_rows: int = 0,
    modeling_participants: int = 0,
    logger=None,
) -> None:
    """Write auditable outputs when a range-limited run is too small for modeling.

    This is a successful controlled stop, not a model-development result. It protects
    development runs such as min_row_index=0/max_row_index=200 from failing when that
    raw slice does not contain enough eligible MCI follow-up visits.
    """
    cohort_summary = cohort_summary or {}
    status = pd.DataFrame(
        [
            {
                "pipeline_status": "completed_without_modeling",
                "reason": reason,
                "input_row_window": record_selection.get("row_window_label", "unknown"),
                "input_row_min_index_inclusive": record_selection.get("min_row_index", ""),
                "input_row_max_index_exclusive": record_selection.get("max_row_index_label", ""),
                "raw_rows_loaded_for_processing": int(raw_rows_loaded),
                "analytic_cohort_rows": int(cohort_summary.get("rows", modeling_rows) or 0),
                "analytic_cohort_participants": int(cohort_summary.get("unique_participants", modeling_participants) or 0),
                "analytic_cohort_events": int(cohort_summary.get("events", 0) or 0),
                "analytic_cohort_non_events": int(cohort_summary.get("non_events", 0) or 0),
            }
        ]
    )
    _round_csv(status, output_file(output_dir, "step1c_participant_train_validation_test_split_summary.csv"), decimal_places)
    _round_csv(status, output_file(output_dir, "step3a_heldout_test_model_performance.csv"), decimal_places)
    _round_csv(status, output_file(output_dir, "step5a_publication_model_performance_table.csv"), decimal_places)

    metadata = {
        "project_name": cfg.get("project_name"),
        "pipeline_status": "completed_without_modeling",
        "reason": reason,
        "input_csv": cfg.get("input_csv"),
        "output_dir": str(output_dir),
        "input_record_selection": record_selection,
        "raw_rows_loaded_for_processing": int(raw_rows_loaded),
        "cohort_summary": cohort_summary,
        "modeling_rows": int(modeling_rows),
        "modeling_participants": int(modeling_participants),
        "manuscript_note": "This run was range-limited or otherwise too small for valid model development. Use a larger row window or max_row_index: 'all' for manuscript analysis.",
    }
    with open(output_file(output_dir, "step5d_run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    report = [
        "# Controlled stop report",
        "",
        "The pipeline completed without Python error, but model development was not performed because the selected analytic cohort was too small or did not contain both outcome classes.",
        "",
        f"**Reason:** {reason}",
        "",
        f"**Raw row window:** {record_selection.get('row_window_label', 'unknown')}",
        f"**Raw rows loaded:** {int(raw_rows_loaded)}",
        f"**Analytic cohort rows:** {int(cohort_summary.get('rows', modeling_rows) or 0)}",
        f"**Analytic cohort participants:** {int(cohort_summary.get('unique_participants', modeling_participants) or 0)}",
        f"**Events:** {int(cohort_summary.get('events', 0) or 0)}",
        f"**Non-events:** {int(cohort_summary.get('non_events', 0) or 0)}",
        "",
        "For manuscript-grade model outputs, set `runtime.input_record_selection.max_row_index: \"all\"` or use a sufficiently large row range before rerunning.",
    ]
    (output_file(output_dir, "step5b_publication_main_findings_summary.md")).write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_file(output_dir, "step7b_final_outputs_report.md")).write_text("\n".join(report) + "\n", encoding="utf-8")
    _round_csv(
        pd.DataFrame(
            [
                {"output_file": public_output_name("step5d_run_metadata.json"), "manuscript_use": "Run metadata and controlled-stop reason"},
                {"output_file": public_output_name("step7b_final_outputs_report.md"), "manuscript_use": "Controlled-stop report for range-limited development run"},
            ]
        ),
        output_file(output_dir, "step7a_manuscript_table_index.csv"),
        decimal_places,
    )
    _round_csv(output_file_synopsis_frame(), output_file(output_dir, "step7c_output_file_synopsis.csv"), decimal_places)
    _round_csv(project_file_synopsis_frame(), output_file(output_dir, "step7d_project_file_synopsis.csv"), decimal_places)
    if logger is not None:
        logger.warning("Controlled stop without modeling: %s", reason)


# Re-export private helper names intentionally. The refactored step modules and
# legacy tests import validated helper functions whose names start with an
# underscore because they originated inside the former monolithic controller.
__all__ = [name for name in globals() if not name.startswith("__")]
