"""
Project: dementia_progression
File: src/pipeline_step6_neuropathology.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Runs secondary neuropathology anchoring analyses using held-out predictions while keeping neuropathology variables excluded from model training.

Design:
    This module contains the executable logic for Step 6/7 neuropathology anchoring. It is intentionally
    separated from the main orchestrator so that GitHub users can inspect,
    test, and maintain one numbered pipeline stage at a time.
"""

from __future__ import annotations

from src.pipeline_context import PipelineContext
from src.pipeline_common import *  # Reuse the validated v1.49 helper surface.


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 6/7 neuropathology anchoring and store downstream state in ``ctx``.

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
    modeling_df = ctx.modeling_df
    test_idx = ctx.test_idx
    predictions = ctx.predictions
    final_risk_col = ctx.final_risk_col
    metrics_df = ctx.metrics_df
    validation_metrics_df = ctx.validation_metrics_df
    tuning_results_df = ctx.tuning_results_df

    progress_step("Step 6/7: Running neuropathology anchoring analysis", enabled=progress_enabled, logger=logger)
    step6_bar = progress_bar(total=2, desc="Step 6/7 neuropathology anchoring", enabled=progress_enabled, unit="task", logger=logger)
    np_cols_present = [c for c in cfg.get("neuropathology_columns", []) if c in modeling_df.columns]
    step6_bar.update("Resolve available neuropathology anchor columns")
    if np_cols_present:
        pred_with_np = predictions.copy()
        np_values = modeling_df.iloc[test_idx][np_cols_present].reset_index(drop=True)
        pred_with_np = pd.concat([pred_with_np, np_values], axis=1)
        anchor_outputs = _neuropathology_anchor_outputs(
            pred_with_np,
            final_risk_col=final_risk_col,
            clinical_risk_col="risk_clinical_only_hist_gradient_boosting",
        )
        _round_csv(anchor_outputs["anchor_cohort"], output_file(output_dir, "step6a_neuropathology_anchor_cohort.csv"), decimal_places)
        _round_csv(anchor_outputs["summary"], output_file(output_dir, "step6b_neuropathology_anchor_outcome_summary.csv"), decimal_places)
        _round_csv(anchor_outputs["associations"], output_file(output_dir, "step6c_neuropathology_anchor_risk_associations.csv"), decimal_places)
        _round_csv(anchor_outputs["quartiles"], output_file(output_dir, "step6d_neuropathology_anchor_risk_quartiles.csv"), decimal_places)
        (output_file(output_dir, "step6e_neuropathology_anchor_interpretation.md")).write_text(str(anchor_outputs["interpretation"]), encoding="utf-8")
        step6_bar.update("Write neuropathology anchor cohort, summaries, associations, quartiles, and interpretation")
    else:
        step6_bar.update("Skip neuropathology anchoring because no configured neuropathology columns are present")
    step6_bar.close()

    best_model = str(metrics_df.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]["model"])
    best_validation_model = str(validation_metrics_df.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]["model"])
    selected_tuning_rows = (
        tuning_results_df.loc[tuning_results_df.get("selected_by_validation", pd.Series(dtype=int)).eq(1)]
        .sort_values(["model"])
        .to_dict(orient="records")
        if not tuning_results_df.empty
        else []
    )


    ctx.np_cols_present = np_cols_present
    ctx.best_model = best_model
    ctx.best_validation_model = best_validation_model
    ctx.selected_tuning_rows = selected_tuning_rows
    return True
