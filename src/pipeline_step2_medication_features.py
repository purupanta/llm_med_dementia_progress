"""
Project: dementia_progression
File: src/pipeline_step2_medication_features.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Builds structured medication features, unique-drug LLM dictionary features, optional BioClinicalBERT/SapBERT neural text representations, analysis datasets, and model-ready feature matrices.

Design:
    This module contains the executable logic for Step 2/7 medication-state feature engineering. It is intentionally
    separated from the main orchestrator so that GitHub users can inspect,
    test, and maintain one numbered pipeline stage at a time.
"""

from __future__ import annotations

import csv
import re
import shutil
from src.pipeline_context import PipelineContext
from src.pipeline_common import *  # Reuse the validated v1.49 helper surface.
from helpers.features import _expand_missing_code_variants, _is_pipeline_derived_column


def _systemwide_structural_missing_codes(cfg: dict) -> list[object]:
    """Return negative NACC system-wide structural missing codes configured for Step 2."""
    codes = cfg.get("systemwide_structural_missing_codes", None)
    if codes is None:
        codes = cfg.get("structural_missing_codes_systemwide", None)
    if codes is None:
        codes = [-4, -4.4, -5]
    return list(codes or [])


def _source_like_columns_with_values(df, values: list[object]) -> dict[str, int]:
    """Count configured structural codes in source-like columns, skipping derived features."""
    expanded = _expand_missing_code_variants(values)
    numeric_values: list[float] = []
    for value in expanded:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    counts: dict[str, int] = {}
    for col in df.columns:
        if _is_pipeline_derived_column(str(col)):
            continue
        series = df[col]
        mask = series.isin(expanded)
        if numeric_values:
            mask = mask | pd.to_numeric(series, errors="coerce").isin(numeric_values)
        n = int(mask.fillna(False).sum())
        if n:
            counts[str(col)] = n
    return counts


def _assert_step2_analysis_clean(
    source_cohort,
    analysis_df,
    cfg: dict,
    dataset_label: str,
) -> None:
    """Fail fast if Step 2 analysis export still contains known value-cleaning errors.

    This guard is intentionally stricter than ordinary unit tests because the
    user reviews S2 CSVs directly. It prevents a successful pipeline run from
    writing an analysis dataset that still has the historical global-missing-code
    defect, NPPMIH structural-missing defect, raw NACCDAYS leakage, or medication
    count inconsistency.
    """
    problems: list[str] = []

    if analysis_df is None or len(analysis_df) == 0:
        problems.append(f"{dataset_label}: analysis dataset is empty")

    temporal_cfg = cfg.get("temporal_handling", {}) or {}
    if bool(temporal_cfg.get("exclude_raw_naccdays_from_step2_outputs", True)) and "NACCDAYS" in analysis_df.columns:
        problems.append(f"{dataset_label}: raw NACCDAYS is present although it must be excluded from Step 2 analysis outputs")

    if "NPPMIH" in analysis_df.columns:
        nppmih = pd.to_numeric(analysis_df["NPPMIH"], errors="coerce")
        bad = int(nppmih.eq(-4.4).sum())
        if bad:
            problems.append(f"{dataset_label}: NPPMIH still contains -4.4 in {bad} rows; this structural missing code must be blank/NaN")

    # v2.5 guard: negative NACC structural missing sentinels (-4, -4.4, -5)
    # must not remain in source-like analysis columns. Positive special codes
    # remain column-specific and are not globally removed.
    systemwide_counts = _source_like_columns_with_values(analysis_df, _systemwide_structural_missing_codes(cfg))
    if systemwide_counts:
        preview = dict(list(systemwide_counts.items())[:12])
        problems.append(f"{dataset_label}: system-wide negative NACC structural missing code(s) remain in source-like columns: {preview}")

    # v2.4 guard: all configured column-specific structural missing codes must
    # already be blank before analysis/export. This covers NACC -4
    # not-available/not-collected sentinels in neuropathology, cognitive, smoking,
    # and alcohol fields without applying global missing rules to valid 8/9 values.
    for col, codes in (cfg.get("missing_codes_by_column", {}) or {}).items():
        if col not in analysis_df.columns:
            continue
        expanded_codes = _expand_missing_code_variants(codes)
        if not expanded_codes:
            continue
        series = analysis_df[col]
        mask = series.isin(expanded_codes)
        numeric_codes = []
        for code in expanded_codes:
            try:
                numeric_codes.append(float(code))
            except (TypeError, ValueError):
                pass
        if numeric_codes:
            numeric_series = pd.to_numeric(series, errors="coerce")
            mask = mask | numeric_series.isin(numeric_codes)
        bad = int(mask.fillna(False).sum())
        if bad:
            problems.append(f"{dataset_label}: {col} still contains configured structural missing code(s) {list(codes)} in {bad} rows")

    for count_col in ["medication_psychotropic_count", "medication_neuro_count", "medication_cardiometabolic_count"]:
        if count_col in analysis_df.columns and "medication_count" in analysis_df.columns:
            domain_count = pd.to_numeric(analysis_df[count_col], errors="coerce")
            med_count = pd.to_numeric(analysis_df["medication_count"], errors="coerce")
            bad = int((domain_count > med_count).fillna(False).sum())
            if bad:
                problems.append(f"{dataset_label}: {count_col} exceeds medication_count in {bad} rows; domain counts must be unique-medication counts")

    preserve_map = cfg.get("preserve_values_by_column", {}) or {}
    for col, values in preserve_map.items():
        if col not in analysis_df.columns or col not in source_cohort.columns:
            continue
        raw = pd.to_numeric(source_cohort[col], errors="coerce").reset_index(drop=True)
        out = pd.to_numeric(analysis_df[col], errors="coerce").reset_index(drop=True)
        if len(raw) != len(out):
            problems.append(f"{dataset_label}: cannot validate preserved values for {col} because source/output row counts differ")
            continue
        preserve_vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(float).tolist()
        if not preserve_vals:
            continue
        mask = raw.astype(float).isin(preserve_vals)
        lost = int((mask & out.isna()).sum())
        if lost:
            problems.append(f"{dataset_label}: valid preserved values for {col} were converted to missing in {lost} rows")

    if problems:
        detail = "\n - ".join(problems)
        raise ValueError(
            "Step 2 analysis-clean validation failed. The pipeline stopped before downstream modeling because the exported "
            f"analysis dataset is not safe for interpretation.\n - {detail}"
        )


def _recompute_visit_time_features_for_export(df, cfg: dict):
    """Recompute prior-MCI timing features in-place while preserving row order.

    This final export guard is deliberately independent of prior feature-engineering
    state. It prevents stale all-zero interval columns from being written if a real
    run accidentally carries an older intermediate frame forward.
    """
    out = df.copy()
    id_col = cfg.get("id_col", "NACCID")
    visit_number_col = cfg.get("visit_number_col", "NACCVNUM")
    y_col = cfg.get("visit_year_col", "VISITYR")
    m_col = cfg.get("visit_month_col", "VISITMO")
    d_col = cfg.get("visit_day_col", "VISITDAY")
    required = [id_col, y_col, m_col, d_col]
    if not all(c in out.columns for c in required):
        return out

    visit_dates = pd.to_datetime(
        {
            "year": pd.to_numeric(out[y_col], errors="coerce"),
            "month": pd.to_numeric(out[m_col], errors="coerce"),
            "day": pd.to_numeric(out[d_col], errors="coerce"),
        },
        errors="coerce",
    )
    tmp = pd.DataFrame({
        "_orig_order": np.arange(len(out)),
        "_id": out[id_col].astype(str).to_numpy(),
        "_date": visit_dates.to_numpy(),
    })
    if visit_number_col in out.columns:
        tmp["_visit_number"] = pd.to_numeric(out[visit_number_col], errors="coerce").to_numpy()
    else:
        tmp["_visit_number"] = np.arange(len(out))
    tmp = tmp.sort_values(["_id", "_date", "_visit_number", "_orig_order"], kind="mergesort")
    tmp["_prior_count"] = tmp.groupby("_id").cumcount()
    tmp["_prior_days"] = tmp.groupby("_id")["_date"].diff().dt.days
    first_dates = tmp.groupby("_id")["_date"].transform("first")
    tmp["_first_days"] = (tmp["_date"] - first_dates).dt.days
    tmp = tmp.sort_values("_orig_order")
    out["prior_mci_visit_count"] = tmp["_prior_count"].astype("Int64").to_numpy()
    out["days_since_prior_mci_visit"] = tmp["_prior_days"].astype("Float64").to_numpy()
    out["days_since_first_mci_visit"] = tmp["_first_days"].astype("Float64").to_numpy()
    return out


def _restore_preserved_values_from_source(source_cohort, analysis_df, cfg: dict):
    """Restore explicitly valid values if an older missing-code path blanked them.

    Values such as VISITMO 8/9 and NACCAGE 88/95/99 are valid in this project
    extract. This function uses the source cohort as the authoritative value
    source and restores only values named in preserve_values_by_column.
    """
    out = analysis_df.copy()
    if source_cohort is None or len(source_cohort) != len(out):
        return out
    preserve_map = cfg.get("preserve_values_by_column", {}) or {}
    for col, values in preserve_map.items():
        if col not in out.columns or col not in source_cohort.columns:
            continue
        raw = pd.to_numeric(source_cohort[col], errors="coerce").reset_index(drop=True)
        cur = pd.to_numeric(out[col], errors="coerce").reset_index(drop=True)
        vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(float).tolist()
        if not vals:
            continue
        restore_mask = raw.astype(float).isin(vals) & cur.isna()
        if int(restore_mask.sum()):
            out.loc[restore_mask.to_numpy(), col] = source_cohort[col].reset_index(drop=True).loc[restore_mask].to_numpy()
    return out


def _finalize_step2_analysis_frame(source_cohort, analysis_df, cfg: dict, dataset_label: str):
    """Apply final real-run cleaning immediately before modeling/export.

    This is a defensive finalization layer. It makes the generated S2 datasets
    clean even if an older intermediate frame still contains raw NACCDAYS,
    NPPMIH=-4.4, stale all-zero temporal intervals, over-broad missing-code
    blanking, or medication-domain burden counts.
    """
    out = analysis_df.copy()

    # Restore valid values first, before structural missing-code replacement.
    out = _restore_preserved_values_from_source(source_cohort, out, cfg)

    # Recompute medication indicators/counts from medication_text so domain counts
    # remain unique-medication counts and cannot exceed medication_count.
    if "medication_text" in out.columns:
        out = add_medication_category_features(out)

    # Recompute timing fields from visit date components and drop raw NACCDAYS from
    # analysis/modeling outputs when configured.
    if bool((cfg.get("feature_engineering", {}) or {}).get("add_visit_time_features", True)):
        out = _recompute_visit_time_features_for_export(out, cfg)
    if bool((cfg.get("temporal_handling", {}) or {}).get("exclude_raw_naccdays_from_step2_outputs", True)):
        out = out.drop(columns=["NACCDAYS"], errors="ignore")

    # Reapply system-wide negative structural missing codes and column-specific
    # missing codes. Positive special codes remain dictionary/column-specific.
    out = replace_systemwide_structural_missing_codes(out, _systemwide_structural_missing_codes(cfg))
    missing_code_map = cfg.get("missing_codes_by_column", {}) or {}
    out = replace_column_specific_missing_codes(
        out,
        missing_code_map,
        preserve_values_by_column=cfg.get("preserve_values_by_column", {}),
    )
    if "NPPMIH" in out.columns:
        nppmih = pd.to_numeric(out["NPPMIH"], errors="coerce")
        out.loc[nppmih.eq(-4.4), "NPPMIH"] = np.nan

    out = recode_binary_columns(out, cfg.get("binary_columns", []))
    return out


def _post_write_step2_csv_guard(path, cfg: dict, dataset_label: str) -> None:
    """Read the just-written CSV back and fail if old defects remain."""
    if not path.exists():
        raise ValueError(f"{dataset_label}: expected CSV was not written: {path}")
    check = pd.read_csv(path, low_memory=False, index_col=False)
    problems: list[str] = []
    if bool((cfg.get("temporal_handling", {}) or {}).get("exclude_raw_naccdays_from_step2_outputs", True)) and "NACCDAYS" in check.columns:
        problems.append("raw NACCDAYS is still present in the CSV header")
    if "NPPMIH" in check.columns:
        bad = int(pd.to_numeric(check["NPPMIH"], errors="coerce").eq(-4.4).sum())
        if bad:
            problems.append(f"NPPMIH=-4.4 remains in {bad} rows")
    systemwide_counts = _source_like_columns_with_values(check, _systemwide_structural_missing_codes(cfg))
    if systemwide_counts:
        problems.append(f"system-wide negative NACC structural missing code(s) remain in source-like columns: {dict(list(systemwide_counts.items())[:12])}")
    for col, codes in (cfg.get("missing_codes_by_column", {}) or {}).items():
        if col not in check.columns:
            continue
        expanded_codes = _expand_missing_code_variants(codes)
        mask = check[col].isin(expanded_codes)
        numeric_codes = []
        for code in expanded_codes:
            try:
                numeric_codes.append(float(code))
            except (TypeError, ValueError):
                pass
        if numeric_codes:
            mask = mask | pd.to_numeric(check[col], errors="coerce").isin(numeric_codes)
        bad = int(mask.fillna(False).sum())
        if bad:
            problems.append(f"{col} still contains configured structural missing code(s) {list(codes)} in {bad} rows")
    for count_col in ["medication_psychotropic_count", "medication_neuro_count", "medication_cardiometabolic_count"]:
        if count_col in check.columns and "medication_count" in check.columns:
            bad = int((pd.to_numeric(check[count_col], errors="coerce") > pd.to_numeric(check["medication_count"], errors="coerce")).fillna(False).sum())
            if bad:
                problems.append(f"{count_col} exceeds medication_count in {bad} rows")
    # Raw text check for cosmetic .0 in core integer-coded columns.
    integer_like = [
        cfg.get("visit_number_col", "NACCVNUM"), cfg.get("visit_month_col", "VISITMO"), cfg.get("visit_day_col", "VISITDAY"),
        "NACCAGE", "prior_mci_visit_count", "medication_count", "medication_psychotropic_count",
        "medication_neuro_count", "medication_cardiometabolic_count", "next_visit_dementia",
    ]
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        sampled_rows = []
        for _, row in zip(range(200), reader):
            sampled_rows.append(row)
    for col in integer_like:
        if col in header:
            idx = header.index(col)
            for vals in sampled_rows:
                if idx < len(vals) and re.fullmatch(r"-?\d+\.0", str(vals[idx]).strip()):
                    problems.append(f"{col} still has .0 integer formatting in the written CSV")
                    break
    if problems:
        raise ValueError(f"{dataset_label}: post-write CSV guard failed for {path}: " + "; ".join(problems))


def _write_step2_analysis_csv(
    source_cohort,
    analysis_df,
    cfg: dict,
    output_dir,
    logical_name: str,
    decimal_places: int,
    dataset_label: str,
):
    """Finalize, validate, and write a Step 2 analysis dataset with clean CSV formatting."""
    final_df = _finalize_step2_analysis_frame(source_cohort, analysis_df, cfg, dataset_label)
    _assert_step2_analysis_clean(source_cohort, final_df, cfg, dataset_label)
    path = output_file(output_dir, logical_name)
    _round_csv(final_df, path, decimal_places)
    _post_write_step2_csv_guard(path, cfg, dataset_label)
    return path


def run_step(ctx: PipelineContext) -> bool:
    """Execute Step 2/7 medication-state feature engineering and store downstream state in ``ctx``.

    Returns
    -------
    bool
        ``True`` when downstream stages should continue; ``False`` when the
        stage wrote controlled-stop outputs for a range-limited/non-modelable run.
    """
    cfg = ctx.cfg
    output_dir = ctx.output_dir
    logger = ctx.logger
    llm_logger = ctx.llm_logger
    decimal_places = ctx.decimal_places
    progress_enabled = ctx.progress_enabled
    cohort = ctx.cohort
    record_selection = ctx.record_selection
    step1_raw_rows_loaded = ctx.step1_raw_rows_loaded
    cohort_summary = ctx.cohort_summary

    progress_step("Step 2/7: Creating structured, LLM-enhanced, and BioClinicalBERT/SapBERT medication-state features", enabled=progress_enabled, logger=logger)
    step2_bar = progress_bar(total=12, desc="Step 2/7 medication features", enabled=progress_enabled, unit="task", logger=logger)
    drug_cols = [c for c in build_drug_columns(cfg["drug_prefix"], int(cfg["max_drug_columns"])) if c in cohort.columns]
    step2_bar.update("Resolve DRUG columns")
    llm_step2_cfg = cfg.get("llm_medication_state", {}) or {}
    unique_raw_drug_source = pd.DataFrame()
    drug_token_abstraction_enabled = bool(llm_step2_cfg.get("enabled", False)) and str(llm_step2_cfg.get("abstraction_unit", "drug_token")).strip().lower() in {"drug", "drug_token", "unique_drug", "unique_drug_name", "drug_name"}
    if drug_token_abstraction_enabled:
        unique_raw_drug_source = extract_unique_drug_names_from_columns(cohort, drug_cols)
        _round_csv(unique_raw_drug_source, output_file(output_dir, "step2d_unique_raw_drug_name_source.csv"), decimal_places)
        logger.info("Unique raw drug-name source dictionary created before combining DRUG columns: unique_drugs=%s", len(unique_raw_drug_source))
        step2_bar.update("Build unique raw drug-name source dictionary before combining DRUG columns")
    else:
        step2_bar.update("Skip unique raw drug-name source dictionary because drug-token abstraction is disabled")
    cohort = combine_drug_columns(cohort, drug_cols)
    step2_bar.update("Combine DRUG columns into medication_text and counts")
    cohort = add_medication_category_features(cohort)
    step2_bar.update("Create structured medication category and burden features")
    logger.info("Structured medication features created: %s medication columns", len(structured_medication_feature_columns(cohort)))

    llm_audit = pd.DataFrame()
    if bool(cfg.get("llm_medication_state", {}).get("enabled", False)):
        llm_cache_name = str(cfg.get("llm_medication_state", {}).get("cache_filename", "step2d_llm_medication_state_abstraction.csv"))
        llm_cache_path = output_file(output_dir, llm_cache_name)
        partial_llm_audit_path = llm_cache_path.with_name(f"{llm_cache_path.stem}_partial.csv")
        partial_llm_progress_path = llm_cache_path.with_name("s2_llm_medication_state_checkpoint_progress.json")
        cohort, llm_audit = apply_llm_medication_state(
            cohort,
            cfg,
            logger=logger,
            llm_logger=llm_logger,
            partial_audit_path=partial_llm_audit_path,
            partial_progress_path=partial_llm_progress_path,
        )
        _round_csv(llm_audit, llm_cache_path, decimal_places)
        llm_quality_audit = build_llm_medication_state_quality_audit(llm_audit, cohort)
        _round_csv(llm_quality_audit, output_file(output_dir, "step2h_llm_medication_state_quality_audit.csv"), decimal_places)
        llm_model_comparison_audit, llm_model_agreement_summary = build_llm_medication_model_comparison_audit(llm_audit, cfg, logger=logger)
        _round_csv(llm_model_comparison_audit, output_file(output_dir, "step2i_llm_model_comparison_audit.csv"), decimal_places)
        _round_csv(llm_model_agreement_summary, output_file(output_dir, "step2j_llm_model_agreement_summary.csv"), decimal_places)
        llm_certification_audit = build_llm_medication_state_certification_audit(llm_audit, cohort)
        _round_csv(llm_certification_audit, output_file(output_dir, "step2k_llm_certification_audit.csv"), decimal_places)
        if drug_token_abstraction_enabled:
            dict_filename = str(cfg.get("llm_medication_state", {}).get("drug_dictionary_filename", "s2e_unique_drug_name_llm_dictionary.csv"))
            dict_path = partial_llm_audit_path.with_name(dict_filename)
            mapping_audit_rows = []
            try:
                llm_dictionary = pd.read_csv(dict_path) if dict_path.exists() else pd.DataFrame()
                raw_keys = set(unique_raw_drug_source.get("llm_dictionary_key", pd.Series(dtype=str)).fillna("").astype(str)) if not unique_raw_drug_source.empty else set()
                llm_keys = set(llm_dictionary.get("llm_dictionary_key", pd.Series(dtype=str)).fillna("").astype(str)) if not llm_dictionary.empty else set()
                raw_keys.discard("")
                llm_keys.discard("")
                index_join_mismatch_rows = 0
                if not unique_raw_drug_source.empty and not llm_dictionary.empty and "drug_dictionary_index" in unique_raw_drug_source.columns and "drug_dictionary_index" in llm_dictionary.columns:
                    merged_by_index = unique_raw_drug_source[["drug_dictionary_index", "llm_dictionary_key"]].merge(
                        llm_dictionary[["drug_dictionary_index", "llm_dictionary_key"]],
                        on="drug_dictionary_index",
                        suffixes=("_raw_source", "_llm_dictionary"),
                        how="inner",
                    )
                    index_join_mismatch_rows = int((merged_by_index["llm_dictionary_key_raw_source"].astype(str) != merged_by_index["llm_dictionary_key_llm_dictionary"].astype(str)).sum())
                raw_missing = int(len(raw_keys - llm_keys))
                llm_missing = int(len(llm_keys - raw_keys))
                stable_key_mapping_pass = int(raw_missing == 0 and llm_missing == 0 and len(raw_keys) == len(llm_keys) and len(raw_keys) > 0)
                mapping_audit_rows.append({
                    "raw_source_unique_drugs": int(len(unique_raw_drug_source)),
                    "llm_dictionary_unique_drugs": int(len(llm_dictionary)),
                    "stable_key_overlap_count": int(len(raw_keys & llm_keys)),
                    "raw_keys_missing_from_llm_dictionary": raw_missing,
                    "llm_keys_missing_from_raw_source": llm_missing,
                    "index_join_mismatch_rows": index_join_mismatch_rows,
                    "stable_key_mapping_pass": stable_key_mapping_pass,
                    "join_key_used_for_downstream_mapping": "llm_dictionary_key",
                    "positional_index_used_for_downstream_mapping": 0,
                    "llm_dictionary_file": str(dict_path),
                })
            except RuntimeError:
                raise
            except Exception as exc:
                mapping_audit_rows.append({
                    "raw_source_unique_drugs": int(len(unique_raw_drug_source)),
                    "llm_dictionary_unique_drugs": 0,
                    "stable_key_overlap_count": 0,
                    "raw_keys_missing_from_llm_dictionary": 0,
                    "llm_keys_missing_from_raw_source": 0,
                    "index_join_mismatch_rows": 0,
                    "stable_key_mapping_pass": 0,
                    "join_key_used_for_downstream_mapping": "llm_dictionary_key",
                    "positional_index_used_for_downstream_mapping": 0,
                    "llm_dictionary_file": str(dict_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            mapping_audit_df = pd.DataFrame(mapping_audit_rows)
            _round_csv(mapping_audit_df, output_file(output_dir, "step2p_llm_drug_dictionary_mapping_audit.csv"), decimal_places)
            if bool(llm_step2_cfg.get("drug_dictionary_enforce_stable_key_mapping", True)):
                try:
                    stable_key_mapping_pass = int(mapping_audit_df.get("stable_key_mapping_pass", pd.Series([0])).iloc[0])
                except Exception:
                    stable_key_mapping_pass = 0
                if not stable_key_mapping_pass:
                    row0 = mapping_audit_df.iloc[0].to_dict() if not mapping_audit_df.empty else {}
                    raise RuntimeError(
                        "Step 2 LLM drug dictionary stable-key audit failed after writing s2p audit: "
                        f"raw_unique_drugs={row0.get('raw_source_unique_drugs')}, "
                        f"llm_unique_drugs={row0.get('llm_dictionary_unique_drugs')}, "
                        f"raw_missing_from_llm={row0.get('raw_keys_missing_from_llm_dictionary')}, "
                        f"llm_missing_from_raw={row0.get('llm_keys_missing_from_raw_source')}. "
                        "Do not use LLM-enhanced medication-state outputs until the raw-drug dictionary and LLM dictionary align by llm_dictionary_key."
                    )
        step2_bar.update("Run LLM medication-state abstraction and write audit")
        logger.info(
            "LLM medication-state features created: %s columns | quality_audit=%s | model_comparison=%s | certification=%s",
            len(llm_medication_feature_columns(cohort)),
            llm_quality_audit.to_dict(orient="records")[0] if not llm_quality_audit.empty else {},
            llm_model_agreement_summary.to_dict(orient="records")[0] if not llm_model_agreement_summary.empty else {},
            llm_certification_audit.to_dict(orient="records")[0] if not llm_certification_audit.empty else {},
        )
    else:
        step2_bar.update("Skip LLM medication-state abstraction because it is disabled")

    neural_audit = pd.DataFrame()
    if bool(cfg.get("neural_text_representations", {}).get("enabled", False)):
        cohort, neural_audit = add_neural_text_representations(cohort, cfg, output_dir=output_dir, logger=logger)
        _round_csv(neural_audit, output_file(output_dir, "step2l_neural_text_representation_audit.csv"), decimal_places)
        step2_bar.update("Create BioClinicalBERT/SapBERT neural medication-text representations")
        logger.info("BioClinicalBERT/SapBERT medication-text representation features created: %s columns", len(neural_text_feature_columns(cohort)))
    else:
        step2_bar.update("Skip BioClinicalBERT/SapBERT neural medication-text representations because they are disabled")

    if bool(cfg.get("feature_engineering", {}).get("add_visit_time_features", True)):
        cohort = add_visit_time_features(cohort, cfg)
        step2_bar.update("Create visit-time and prior-MCI-history features")
        logger.info("Visit-time/MCI-history features created: %s", visit_time_feature_columns(cohort))
    else:
        step2_bar.update("Skip visit-time feature engineering because it is disabled")

    # Corrected missing-value handling: do not apply broad NACC sentinel values
    # globally. Codes such as 8, 9, 88, 95, 96, 97, 98, and 99 are valid in
    # several variables (for example visit month/day, age, education years,
    # cognitive scores, and NPPMIH categories). Missingness is therefore applied
    # only through column-specific rules configured in YAML and, when supplied,
    # inferred conservatively from official NACC dictionary CSVs.
    official_missing_cfg = cfg.get("official_nacc_missing_codes", {}) or {}
    official_missing_map = {}
    if bool(official_missing_cfg.get("enabled", True)):
        official_missing_dictionary_dir = official_missing_cfg.get("dictionary_dir", "resources/nacc_official")
        ensure_official_nacc_dictionary_files(
            official_missing_dictionary_dir,
            project_root=PROJECT_ROOT,
            autodownload_cfg=official_missing_cfg.get("autodownload", {}),
            logger=logger,
        )
        official_missing_map = load_official_nacc_missing_codes(
            official_missing_dictionary_dir,
            project_root=PROJECT_ROOT,
            target_columns=list(cohort.columns),
            preserve_values_by_column=cfg.get("preserve_values_by_column", {}),
            logger=logger,
        )
    missing_code_map = merge_missing_code_maps(official_missing_map, cfg.get("missing_codes_by_column", {}))
    modeling_df = replace_systemwide_structural_missing_codes(cohort, _systemwide_structural_missing_codes(cfg))
    modeling_df = replace_column_specific_missing_codes(
        modeling_df,
        missing_code_map,
        preserve_values_by_column=cfg.get("preserve_values_by_column", {}),
    )
    if bool(cfg.get("allow_global_missing_codes", False)) and cfg.get("missing_codes_global"):
        logger.warning(
            "Global missing-code replacement is enabled. This is not recommended for NACC analysis because sentinel values can be valid in other columns."
        )
        modeling_df = replace_global_missing_codes(modeling_df, cfg.get("missing_codes_global", []))
    modeling_df = recode_binary_columns(modeling_df, cfg.get("binary_columns", []))
    # LLM medication-state features are pipeline-derived values. They should never be
    # recoded as NACC missing values or left missing in the analysis dataset. If any
    # mapping edge case leaves a numeric LLM feature blank, use the conservative
    # non-certified/no-exposure value 0 rather than propagating NaN into modeling.
    for _llm_col in [c for c in modeling_df.columns if str(c).startswith("llm_")]:
        if pd.api.types.is_numeric_dtype(modeling_df[_llm_col]):
            modeling_df[_llm_col] = pd.to_numeric(modeling_df[_llm_col], errors="coerce").fillna(0)
    # Final defensive cleanup before feature-list construction and modeling. This
    # makes the in-memory modeling frame match the analysis-clean S2 exports.
    modeling_df = _finalize_step2_analysis_frame(cohort, modeling_df, cfg, "Step 2 modeling frame")
    _assert_step2_analysis_clean(cohort, modeling_df, cfg, "Step 2 modeling frame")
    step2_bar.update("Apply configured missing-value, binary recoding, and final analysis-clean safeguards")

    traj_cfg = (cfg.get("feature_engineering", {}) or {}).get("trajectory_features", {}) or {}
    if bool(traj_cfg.get("enabled", cfg.get("feature_engineering", {}).get("add_trajectory_features", False))):
        modeling_df = add_longitudinal_trajectory_features(modeling_df, cfg)
        _assert_step2_analysis_clean(cohort, modeling_df, cfg, "Step 2 trajectory-enhanced modeling frame")
        logger.info("Longitudinal trajectory features created: %s", len(longitudinal_trajectory_feature_columns(modeling_df)))
        step2_bar.update("Create leakage-safe longitudinal trajectory features")
    else:
        step2_bar.update("Skip longitudinal trajectory feature engineering because it is disabled")

    y = modeling_df["next_visit_dementia"].astype(int)
    groups = modeling_df[cfg["id_col"]]

    modeling_reason = _modeling_eligibility_reason(y, groups)
    if modeling_reason is not None:
        step2_bar.close()
        progress_step(f"Controlled stop before model fitting: {modeling_reason}", enabled=progress_enabled, logger=logger)
        _write_graceful_stop_outputs(
            output_dir=output_dir,
            cfg=cfg,
            decimal_places=decimal_places,
            reason=modeling_reason,
            record_selection=record_selection,
            raw_rows_loaded=step1_raw_rows_loaded,
            cohort_summary=cohort_summary,
            modeling_rows=len(modeling_df),
            modeling_participants=int(groups.nunique()) if len(groups) else 0,
            logger=logger,
        )
        return False

    progress_step("Step 2/7: Building clinical-only, structured medication-aware, LLM-enhanced, and neural medication-aware feature matrices", enabled=progress_enabled, logger=logger)
    clinical_features = build_feature_lists(
        modeling_df,
        categorical_columns=cfg.get("categorical_columns", []),
        numeric_columns=cfg.get("numeric_columns", []),
        binary_columns=cfg.get("binary_columns", []),
        include_medication=False,
    )
    medication_features = build_feature_lists(
        modeling_df,
        categorical_columns=cfg.get("categorical_columns", []),
        numeric_columns=cfg.get("numeric_columns", []),
        binary_columns=cfg.get("binary_columns", []),
        include_medication=True,
        include_llm_medication=False,
    )
    llm_medication_features = build_feature_lists(
        modeling_df,
        categorical_columns=cfg.get("categorical_columns", []),
        numeric_columns=cfg.get("numeric_columns", []),
        binary_columns=cfg.get("binary_columns", []),
        include_medication=True,
        include_llm_medication=True,
    )
    neural_medication_features = build_feature_lists(
        modeling_df,
        categorical_columns=cfg.get("categorical_columns", []),
        numeric_columns=cfg.get("numeric_columns", []),
        binary_columns=cfg.get("binary_columns", []),
        include_medication=True,
        include_llm_medication=True,
        include_neural_text=True,
    )

    def _with_trajectory_features(base_features: dict[str, list[str]], *, include_medication_trajectory: bool = True) -> dict[str, list[str]]:
        traj_cols = longitudinal_trajectory_feature_columns(modeling_df)
        if not include_medication_trajectory:
            traj_cols = [c for c in traj_cols if "med" not in str(c).lower() and "llm_" not in str(c).lower()]
        out = {k: list(v) for k, v in base_features.items()}
        out.setdefault("numeric", [])
        for col in traj_cols:
            if col not in out.get("categorical", []) and col not in out.get("numeric", []) and col not in out.get("binary", []):
                out["numeric"].append(col)
        return out

    trajectory_clinical_features = _with_trajectory_features(clinical_features, include_medication_trajectory=False)
    trajectory_medication_features = _with_trajectory_features(medication_features, include_medication_trajectory=True)
    trajectory_llm_medication_features = _with_trajectory_features(llm_medication_features, include_medication_trajectory=True)
    trajectory_neural_medication_features = _with_trajectory_features(neural_medication_features, include_medication_trajectory=True)
    step2_bar.update("Build feature-list definitions for all model arms")

    clinical_X = make_feature_frame(modeling_df, clinical_features)
    med_X = make_feature_frame(modeling_df, medication_features)
    llm_X = make_feature_frame(modeling_df, llm_medication_features)
    neural_X = make_feature_frame(modeling_df, neural_medication_features)
    trajectory_clinical_X = make_feature_frame(modeling_df, trajectory_clinical_features)
    trajectory_med_X = make_feature_frame(modeling_df, trajectory_medication_features)
    trajectory_llm_X = make_feature_frame(modeling_df, trajectory_llm_medication_features)
    trajectory_neural_X = make_feature_frame(modeling_df, trajectory_neural_medication_features)
    _round_csv(build_feature_summary(clinical_X, clinical_features), output_file(output_dir, "step2a_clinical_only_feature_summary.csv"), decimal_places)
    _round_csv(build_feature_summary(med_X, medication_features), output_file(output_dir, "step2b_structured_medication_aware_feature_summary.csv"), decimal_places)
    _round_csv(build_feature_summary(llm_X, llm_medication_features), output_file(output_dir, "step2e_llm_enhanced_medication_feature_summary.csv"), decimal_places)
    if neural_text_feature_columns(modeling_df):
        _round_csv(build_feature_summary(neural_X, neural_medication_features), output_file(output_dir, "step2m_neural_medication_feature_summary.csv"), decimal_places)
    traj_cols = longitudinal_trajectory_feature_columns(modeling_df)
    if traj_cols:
        traj_only = {"categorical": [], "numeric": traj_cols, "binary": []}
        _round_csv(build_feature_summary(modeling_df[traj_cols].copy(), traj_only), output_file(output_dir, "step2o_longitudinal_trajectory_feature_summary.csv"), decimal_places)
    step2_bar.update("Create feature matrices and write feature summaries")

    if bool(cfg.get("output_options", {}).get("write_analysis_dataset", True)):
        np_cols_present = [c for c in cfg.get("neuropathology_columns", []) if c in modeling_df.columns]
        analysis_cols = unique_preserve_order(
            [
                cfg["id_col"],
                cfg["visit_number_col"],
                cfg["visit_year_col"],
                cfg["visit_month_col"],
                cfg["visit_day_col"],
                cfg["current_diagnosis_col"],
                "next_diagnosis",
                "next_visit_dementia",
                "medication_count",
                "medication_psychotropic_count",
                "medication_neuro_count",
                "medication_cardiometabolic_count",
                "medication_text",
            ]
            + visit_time_feature_columns(modeling_df)
            + np_cols_present
            + medication_feature_columns(modeling_df, include_llm=True, include_neural_text=True)
            + longitudinal_trajectory_feature_columns(modeling_df)
            + clinical_features["categorical"]
            + clinical_features["numeric"]
            + clinical_features["binary"]
        )
        analysis_cols = [c for c in analysis_cols if c in modeling_df.columns]
        # NACCDAYS is retained in Step 1 for source traceability but is excluded
        # from Step 2 analysis datasets/modeling because this extract's last-visit
        # construction can make it non-informative within participant. Corrected
        # elapsed-time features are recomputed from VISITYR/VISITMO/VISITDAY.
        if bool(cfg.get("temporal_handling", {}).get("exclude_raw_naccdays_from_step2_outputs", True)):
            analysis_cols = [c for c in analysis_cols if c != "NACCDAYS"]
        structured_analysis_cols = [c for c in analysis_cols if not c.startswith("llm_") and c not in neural_text_feature_columns(modeling_df)]
        llm_analysis_cols = [c for c in analysis_cols if c not in neural_text_feature_columns(modeling_df)]
        logger.info("Writing S2D structured analysis dataset: rows=%s cols=%s", len(modeling_df), len(structured_analysis_cols))
        s2d_path = _write_step2_analysis_csv(
            source_cohort=cohort,
            analysis_df=modeling_df[structured_analysis_cols],
            cfg=cfg,
            output_dir=output_dir,
            logical_name="step2c_analysis_dataset_structured_medication_state_features.csv",
            decimal_places=decimal_places,
            dataset_label="s2d structured medication-state analysis dataset",
        )
        logger.info("Writing S2G LLM-enhanced analysis dataset: rows=%s cols=%s", len(modeling_df), len(llm_analysis_cols))
        if list(llm_analysis_cols) == list(structured_analysis_cols):
            # When LLM abstraction is disabled, S2G is intentionally identical to
            # S2D. Copying avoids redundant finalization and keeps the offline
            # validation profile fast while preserving the expected output file.
            s2g_path = output_file(output_dir, "step2f_analysis_dataset_llm_enhanced_medication_state.csv")
            shutil.copyfile(s2d_path, s2g_path)
            _post_write_step2_csv_guard(s2g_path, cfg, "s2g LLM-enhanced medication-state analysis dataset")
        else:
            _write_step2_analysis_csv(
                source_cohort=cohort,
                analysis_df=modeling_df[llm_analysis_cols],
                cfg=cfg,
                output_dir=output_dir,
                logical_name="step2f_analysis_dataset_llm_enhanced_medication_state.csv",
                decimal_places=decimal_places,
                dataset_label="s2g LLM-enhanced medication-state analysis dataset",
            )
        if neural_text_feature_columns(modeling_df):
            logger.info("Writing S2N neural analysis dataset: rows=%s cols=%s", len(modeling_df), len(analysis_cols))
            _write_step2_analysis_csv(
                source_cohort=cohort,
                analysis_df=modeling_df[analysis_cols],
                cfg=cfg,
                output_dir=output_dir,
                logical_name="step2n_analysis_dataset_neural_medication_state.csv",
                decimal_places=decimal_places,
                dataset_label="s2n neural medication-state analysis dataset",
            )
        step2_bar.update("Write structured, LLM-enhanced, and neural analysis datasets")
    else:
        step2_bar.update("Skip analysis dataset exports because output option is disabled")
    step2_bar.close()


    ctx.cohort = cohort
    ctx.llm_audit = llm_audit
    ctx.neural_audit = neural_audit
    ctx.modeling_df = modeling_df
    ctx.y = y
    ctx.groups = groups
    ctx.clinical_features = clinical_features
    ctx.medication_features = medication_features
    ctx.llm_medication_features = llm_medication_features
    ctx.neural_medication_features = neural_medication_features
    ctx.trajectory_clinical_features = trajectory_clinical_features
    ctx.trajectory_medication_features = trajectory_medication_features
    ctx.trajectory_llm_medication_features = trajectory_llm_medication_features
    ctx.trajectory_neural_medication_features = trajectory_neural_medication_features
    ctx.clinical_X = clinical_X
    ctx.med_X = med_X
    ctx.llm_X = llm_X
    ctx.neural_X = neural_X
    ctx.trajectory_clinical_X = trajectory_clinical_X
    ctx.trajectory_med_X = trajectory_med_X
    ctx.trajectory_llm_X = trajectory_llm_X
    ctx.trajectory_neural_X = trajectory_neural_X
    return True
