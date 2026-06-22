"""
Project: dementia_progression
File: src/pipeline_step1_source_cohort.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Loads configured NACC source columns, applies required-row filtering, writes the authoritative Step 1 downstream source file, and constructs the MCI next-visit analytic cohort.

Design:
    This module contains the executable logic for Step 1/7 source and cohort finalization. It is intentionally
    separated from the main orchestrator so that GitHub users can inspect,
    test, and maintain one numbered pipeline stage at a time.
"""

from __future__ import annotations

from src.pipeline_context import PipelineContext
from src.pipeline_common import *  # Reuse the validated v1.49 helper surface.


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 1/7 source and cohort finalization and store downstream state in ``ctx``.

    Returns
    -------
    bool
        ``True`` when downstream stages should continue; ``False`` when the
        stage wrote controlled-stop outputs for a range-limited/non-modelable run.
    """
    cfg = ctx.cfg
    input_csv = ctx.input_csv
    output_dir = ctx.output_dir
    logger = ctx.logger
    decimal_places = ctx.decimal_places
    progress_enabled = ctx.progress_enabled


    progress_step("Step 1/7: Loading, row-filtering, and finalizing NACC source columns", enabled=progress_enabled, logger=logger)
    step1_bar = progress_bar(total=11, desc="Step 1/7 source and cohort", enabled=progress_enabled, unit="task", logger=logger)
    step1_bar.update("Resolve configured source columns and row window")
    usecols = build_usecols(cfg)
    runtime_cfg = cfg.get("runtime", {}) or {}
    record_selection = normalize_input_record_selection(runtime_cfg.get("input_record_selection"))
    source_header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    step1_bar.update("Read input header")
    df_loaded = load_selected_columns(
        input_csv,
        usecols=usecols,
        row_min_index=int(record_selection["min_row_index"]),
        row_max_index=record_selection["max_row_index"],
    )
    step1_bar.update("Load selected rows and columns")
    step1_raw_rows_loaded = int(len(df_loaded))
    step1_required_cols = _step1_required_columns(cfg)
    df_row_filtered, step1_required_audit = _filter_rows_missing_required_columns(df_loaded, step1_required_cols)
    step1_bar.update("Filter rows missing required downstream columns")
    s1a_path = output_file(output_dir, "s1a_nacc_w_last_visit_filtered.csv")
    _round_csv(df_row_filtered, s1a_path, decimal_places)
    step1_bar.update("Write s1a row-filtered source file")
    df_final, step1_all_blank_dropped_cols = _drop_all_blank_columns(df_row_filtered)
    step1_bar.update("Remove all-blank columns")
    s1c_path = output_file(output_dir, "s1c_nacc_w_last_visit_filtered_final.csv")
    _round_csv(df_final, s1c_path, decimal_places)
    step1_bar.update("Write s1c final downstream source file")
    # Load official NACC data-dictionary descriptions before writing the
    # source-column audit. When official files are available under
    # resources/nacc_official/*.csv, Col_Descr is populated from those files.
    step1_cfg = cfg.get("step1", {}) or {}
    official_dictionary_dir = step1_cfg.get("official_nacc_dictionary_dir", "resources/nacc_official")
    official_dictionary_download_audit = ensure_official_nacc_dictionary_files(
        official_dictionary_dir,
        project_root=ctx.project_root,
        autodownload_cfg=step1_cfg.get("official_nacc_dictionary_autodownload", {}),
        logger=logger,
    )
    _round_csv(official_dictionary_download_audit, output_file(output_dir, "s1e_official_nacc_dictionary_download_audit.csv"), decimal_places)
    official_descriptions, official_description_sources = load_official_nacc_column_descriptions(
        official_dictionary_dir,
        project_root=ctx.project_root,
        logger=logger,
    )
    if step1_cfg.get("require_official_column_descriptions") and not official_descriptions:
        raise RuntimeError(
            "Step 1 requires official NACC column descriptions, but no descriptions were loaded from "
            f"{step1_cfg.get('official_nacc_dictionary_dir', 'resources/nacc_official')}"
        )
    step1_col_audit = _build_step1_column_audit(
        source_header=source_header,
        selected_usecols=usecols,
        row_filtered_df=df_row_filtered,
        final_df=df_final,
        dropped_all_blank_columns=step1_all_blank_dropped_cols,
        official_descriptions=official_descriptions,
        official_description_sources=official_description_sources,
    )
    _round_csv(step1_col_audit, output_file(output_dir, "s1b_incl_excl_cols.csv"), decimal_places)
    step1_bar.update("Write Step 1 column audit")
    step1_profile = pd.DataFrame(
        [
            {
                "input_row_window": record_selection["row_window_label"],
                "input_row_min_index_inclusive": record_selection["min_row_index"],
                "input_row_max_index_exclusive": record_selection["max_row_index_label"],
                "raw_rows_loaded_for_processing_before_step1_required_row_filter": step1_raw_rows_loaded,
                "rows_removed_missing_required_step1_columns": int(step1_required_audit.loc[step1_required_audit["required_column"].eq("ANY_REQUIRED_COLUMN"), "rows_missing_or_blank"].iloc[0]),
                "rows_retained_after_required_step1_row_filter": int(len(df_row_filtered)),
                "raw_unique_participants_retained_after_required_step1_row_filter": int(df_row_filtered[cfg["id_col"]].nunique()) if cfg["id_col"] in df_row_filtered.columns else 0,
                "columns_loaded_before_all_blank_filter": int(len(df_row_filtered.columns)),
                "columns_removed_all_blank_null_whitespace_or_blankspace": int(len(step1_all_blank_dropped_cols)),
                "columns_retained_in_final_step1_downstream_file": int(len(df_final.columns)),
                "neuropathology_columns_retained_in_final_step1_file": int(sum(c in df_final.columns for c in cfg.get("neuropathology_columns", []))),
                "required_row_filter_columns": "; ".join(step1_required_cols),
                "all_blank_columns_removed": "; ".join(step1_all_blank_dropped_cols),
            }
        ]
    )
    _round_csv(step1_profile, output_file(output_dir, "s1d_input_data_profile.csv"), decimal_places)
    step1_bar.update("Write Step 1 input profile")
    # Enforce the requested handoff: all downstream analysis reads the final Step 1 file.
    df = pd.read_csv(s1c_path, low_memory=True)
    step1_bar.update("Reload s1c as authoritative downstream source")
    logger.info(
        "Step 1 source finalized: loaded_rows=%s retained_rows=%s final_cols=%s dropped_all_blank_cols=%s row_window=%s",
        step1_raw_rows_loaded,
        len(df),
        len(df.columns),
        len(step1_all_blank_dropped_cols),
        record_selection["row_window_label"],
    )

    progress_step("Step 1/7: Constructing MCI next-visit analytic cohort", enabled=progress_enabled, logger=logger)
    ordered = add_visit_order_columns(
        df=df,
        id_col=cfg["id_col"],
        visit_number_col=cfg["visit_number_col"],
        visit_year_col=cfg["visit_year_col"],
        visit_month_col=cfg["visit_month_col"],
        visit_day_col=cfg["visit_day_col"],
    )
    labeled = build_next_visit_labels(
        df=ordered,
        id_col=cfg["id_col"],
        diagnosis_col=cfg["current_diagnosis_col"],
        dementia_code=int(cfg["dementia_code"]),
    )
    cohort = select_mci_with_next_visit(labeled, diagnosis_col=cfg["current_diagnosis_col"], mci_code=int(cfg["mci_code"]))
    cohort_summary = summarize_cohort(cohort, cfg["id_col"])
    _round_csv(pd.DataFrame([cohort_summary]), output_file(output_dir, "step1b_mci_next_visit_cohort_summary.csv"), decimal_places)
    step1_bar.update("Construct MCI next-visit cohort and write summary")
    step1_bar.close()
    logger.info("Analytic cohort: %s", cohort_summary)

    early_reason = _modeling_eligibility_reason(
        pd.Series(cohort["next_visit_dementia"], dtype="Int64") if "next_visit_dementia" in cohort.columns else pd.Series(dtype="Int64"),
        cohort[cfg["id_col"]] if cfg["id_col"] in cohort.columns else pd.Series(dtype="object"),
    )
    if early_reason is not None and len(cohort) == 0:
        progress_step(f"Controlled stop before modeling: {early_reason}", enabled=progress_enabled, logger=logger)
        _write_graceful_stop_outputs(
            output_dir=output_dir,
            cfg=cfg,
            decimal_places=decimal_places,
            reason=early_reason,
            record_selection=record_selection,
            raw_rows_loaded=step1_raw_rows_loaded,
            cohort_summary=cohort_summary,
            modeling_rows=0,
            modeling_participants=0,
            logger=logger,
        )
        return False



    ctx.record_selection = record_selection
    ctx.step1_raw_rows_loaded = step1_raw_rows_loaded
    ctx.df = df
    ctx.cohort = cohort
    ctx.cohort_summary = cohort_summary
    return True
