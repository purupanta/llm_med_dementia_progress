"""
Project: dementia_progression
File: helpers/features.py

Synopsis:
    Feature-engineering utilities for clinical-only, structured medication-aware, and
    LLM-enhanced medication-state model matrices.

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
from typing import Any, Dict, Iterable, List
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _is_pipeline_derived_column(col: str) -> bool:
    """Return True for variables created by this pipeline, not raw NACC-coded fields.

    NACC missing-value sentinels such as 8, 9, 88, and 99 should be applied to
    original clinical/categorical fields only. Derived medication burden counts,
    structured medication indicators, LLM medication-state indicators, model
    predictions, and visit-history variables may legitimately take values such
    as 8 or 9. Treating those values as NACC missing codes would corrupt the
    generated features, especially llm_medication_state_domain_count.
    """
    derived_exact = {
        "medication_count",
        "medication_psychotropic_count",
        "medication_neuro_count",
        "medication_cardiometabolic_count",
        "prior_mci_visit_count",
        "days_since_prior_mci_visit",
        "days_since_first_mci_visit",
        "next_diagnosis",
        "next_visit_dementia",
    }
    derived_prefixes = (
        "med_cat_",
        "llm_",
        "bioclinicalbert_medtxt_",
        "sapbert_medtxt_",
        "clinicalbert_sapbert_medtxt_",
        "risk_",
        "high_risk_",
        "delta_risk_",
        "reclassification_",
        "traj_",
    )
    return col in derived_exact or col.startswith(derived_prefixes)


def replace_global_missing_codes(df: pd.DataFrame, missing_codes: Iterable[object]) -> pd.DataFrame:
    """Replace broad NACC missing-value sentinel codes only in source-like columns.

    This intentionally skips pipeline-derived features. For example, an LLM
    medication-state domain count of 8 is a valid count, not a NACC missing code.

    Column-specific structural missing codes that are not general NACC sentinels
    should be configured separately through ``missing_codes_by_column`` and
    applied with :func:`replace_column_specific_missing_codes`.
    """
    out = df.copy()
    missing_codes = list(set(missing_codes))
    for col in out.columns:
        if _is_pipeline_derived_column(str(col)):
            continue
        out[col] = out[col].replace(missing_codes, np.nan)
    return out


def _expand_missing_code_variants(codes: Iterable[object]) -> list[object]:
    """Return numeric and string variants for configured missing codes.

    CSV readers can preserve a code as either a numeric value or a string.
    Including both variants prevents a structural missing code such as ``-4.4``
    from surviving because of dtype-specific parsing differences.
    """
    expanded: list[object] = []
    seen: set[str] = set()
    for code in list(codes or []):
        variants = [code]
        if isinstance(code, str):
            stripped = code.strip()
            variants.append(stripped)
            try:
                variants.append(float(stripped))
            except ValueError:
                pass
        else:
            variants.append(str(code))
            try:
                variants.append(float(code))
            except (TypeError, ValueError):
                pass
        for variant in variants:
            key = f"{type(variant).__name__}:{variant}"
            if key not in seen:
                seen.add(key)
                expanded.append(variant)
    return expanded


def _numeric_equivalent(value: object) -> float | None:
    """Return a numeric equivalent when possible, otherwise None."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _remove_preserved_codes(codes: Iterable[object], preserved: Iterable[object]) -> list[object]:
    """Remove explicitly preserved valid values from a candidate missing-code list."""
    preserved_list = _expand_missing_code_variants(list(preserved or []))
    preserved_text = {str(v).strip().lower() for v in preserved_list}
    preserved_numeric = {x for x in (_numeric_equivalent(v) for v in preserved_list) if x is not None}
    out: list[object] = []
    seen: set[str] = set()
    for code in list(codes or []):
        text = str(code).strip().lower()
        numeric = _numeric_equivalent(code)
        if text in preserved_text or (numeric is not None and numeric in preserved_numeric):
            continue
        key = f"{type(code).__name__}:{code}"
        if key not in seen:
            seen.add(key)
            out.append(code)
    return out


def merge_missing_code_maps(*maps: dict[str, Iterable[object]] | None) -> dict[str, list[object]]:
    """Merge multiple column-specific missing-code maps without creating global rules."""
    merged: dict[str, list[object]] = {}
    for mapping in maps:
        for col, codes in dict(mapping or {}).items():
            if not col:
                continue
            key = str(col)
            bucket = merged.setdefault(key, [])
            for code in list(codes or []):
                if code not in bucket:
                    bucket.append(code)
    return merged


def _canonical_dictionary_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _choose_dictionary_column(columns: list[str], candidates: list[str]) -> str | None:
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


def _split_code_values(value: object) -> list[object]:
    """Parse one or more coded values from heterogeneous NACC dictionary cells."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    # Keep decimals such as -4.4. Split common separators and ranges conservatively.
    text = text.replace(";", ",").replace("|", ",")
    parts: list[str] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Extract numeric-looking tokens from expressions such as "8=Unknown" or "8 Unknown".
        matches = re.findall(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?(?![A-Za-z0-9])", chunk)
        if matches:
            parts.extend(matches)
        else:
            parts.append(chunk)
    parsed: list[object] = []
    for part in parts:
        try:
            num = float(part)
            if num.is_integer():
                parsed.append(int(num))
            else:
                parsed.append(num)
        except ValueError:
            parsed.append(part)
    return parsed


def _looks_like_missing_label(value: object) -> bool:
    """Return True when a dictionary label describes missing or non-observed data."""
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return False
    missing_terms = (
        "missing", "unknown", "not available", "not assessed", "not asked", "not applicable",
        "not app", "not collected", "did not collect", "not done", "not performed",
        "refused", "don't know", "dont know", "unable to determine", "no information",
        "blank", "not reported", "not recorded", "not obtained", "not administered",
        "form version", "skipped",
    )
    return any(term in text for term in missing_terms)


def load_official_nacc_missing_codes(
    dictionary_dir: str | Path | None,
    *,
    project_root: str | Path | None = None,
    target_columns: Iterable[str] | None = None,
    preserve_values_by_column: dict[str, Iterable[object]] | None = None,
    logger: Any | None = None,
) -> dict[str, list[object]]:
    """Infer column-specific missing codes from official NACC dictionary CSVs when present.

    The parser is intentionally conservative. It only returns codes tied to a
    named variable in a dictionary row whose label/meaning explicitly indicates
    missing, unknown, not collected, not available, refused, or similar non-
    observed status. It never creates a global missing-code rule.
    """
    if dictionary_dir is None or str(dictionary_dir).strip() == "":
        return {}
    root = Path(project_root) if project_root is not None else Path.cwd()
    path = Path(str(dictionary_dir)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        if logger is not None:
            logger.warning("Official NACC missing-code dictionary directory not found: %s", path)
        return {}
    target_set = {str(c).upper() for c in (target_columns or [])}
    name_candidates = [
        "Col_Name", "Column", "Column Name", "Variable", "Variable Name", "VARNAME", "VAR_NAME",
        "Name", "Field", "Field Name", "Data Element", "Data Element Name", "NACC Variable", "UDS Variable",
    ]
    code_candidates = [
        "Code", "Value", "Value Code", "Coded Value", "Allowable Value", "Allowable Values",
        "Response Code", "Response", "Missing Code", "Missing Codes", "Data Value", "Numeric Code",
    ]
    label_candidates = [
        "Label", "Meaning", "Value Label", "Code Label", "Description", "Value Description",
        "Response Label", "Response Meaning", "Definition", "Text", "Notes", "Comment",
    ]
    explicit_missing_candidates = ["Missing Code", "Missing Codes", "Missing Values", "NA Codes", "Unknown Codes"]
    missing_map: dict[str, list[object]] = {}
    for csv_path in sorted(path.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path, dtype=str, low_memory=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(csv_path, dtype=str, low_memory=False, encoding="latin1")
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not read official NACC missing-code CSV %s: %s", csv_path, exc)
            continue
        if frame.empty:
            continue
        columns = list(frame.columns)
        name_col = _choose_dictionary_column(columns, name_candidates)
        if name_col is None:
            continue
        explicit_missing_col = _choose_dictionary_column(columns, explicit_missing_candidates)
        code_col = _choose_dictionary_column(columns, code_candidates)
        label_col = _choose_dictionary_column(columns, label_candidates)
        for _, row in frame.iterrows():
            var = str(row.get(name_col, "")).strip()
            if not var or var.lower() in {"nan", "none"}:
                continue
            var_key = var.upper()
            if target_set and var_key not in target_set:
                continue
            candidate_codes: list[object] = []
            if explicit_missing_col is not None:
                candidate_codes.extend(_split_code_values(row.get(explicit_missing_col)))
            if code_col is not None and label_col is not None and _looks_like_missing_label(row.get(label_col)):
                candidate_codes.extend(_split_code_values(row.get(code_col)))
            if not candidate_codes:
                continue
            # Use the actual column spelling from target_columns when available.
            out_col = next((str(c) for c in (target_columns or []) if str(c).upper() == var_key), var)
            bucket = missing_map.setdefault(out_col, [])
            for code in candidate_codes:
                if code not in bucket:
                    bucket.append(code)
    # Explicitly preserve values known to be valid in the user's NACC extract.
    if preserve_values_by_column:
        for col, preserve in preserve_values_by_column.items():
            key = next((k for k in missing_map if k.upper() == str(col).upper()), str(col))
            if key in missing_map:
                missing_map[key] = _remove_preserved_codes(missing_map[key], preserve)
    missing_map = {k: v for k, v in missing_map.items() if v}
    if logger is not None:
        logger.info("Loaded official NACC column-specific missing-code rules: dictionary_dir=%s columns=%s", path, len(missing_map))
    return missing_map




def replace_systemwide_structural_missing_codes(
    df: pd.DataFrame,
    structural_codes: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Replace system-wide negative NACC structural missing codes in source-like columns.

    NACC negative special codes such as -4, -4.4, and -5 represent structural
    missingness/not-available/form-not-submitted status rather than measured
    clinical values. They are safe to remove broadly from raw/source-like NACC
    variables, while still skipping pipeline-derived variables and neural
    embeddings where negative values can be legitimate numeric features.

    Positive special codes such as 8, 9, 88, 95, 96, 97, 98, 99, and 999 are
    intentionally *not* handled here; those must remain column-specific because
    they can be valid values in variables such as visit month, age, education,
    cognitive scores, and selected neuropathology fields.
    """
    out = df.copy()
    codes = list(structural_codes or [-4, -4.4, -5])
    expanded_codes = _expand_missing_code_variants(codes)
    numeric_codes: list[float] = []
    for code in expanded_codes:
        try:
            numeric_codes.append(float(code))
        except (TypeError, ValueError):
            continue
    for col in out.columns:
        if _is_pipeline_derived_column(str(col)):
            continue
        series = out[col]
        mask = series.isin(expanded_codes)
        if numeric_codes:
            mask = mask | pd.to_numeric(series, errors="coerce").isin(numeric_codes)
        if mask.any():
            out.loc[mask, col] = np.nan
    return out

def replace_column_specific_missing_codes(
    df: pd.DataFrame,
    missing_codes_by_column: dict[str, Iterable[object]] | None,
    *,
    preserve_values_by_column: dict[str, Iterable[object]] | None = None,
) -> pd.DataFrame:
    """Replace variable-specific structural missing codes with missing values.

    Some NACC variables have sentinel values that are valid only for that
    variable. For example, ``NPPMIH = -4.4`` means the NP Form version did not
    collect the item in that way and must be analyzed as missing, not as a
    negative neuropathology value. This function applies such codes only to
    explicitly configured columns so that genuine values in unrelated variables
    are not altered.
    """
    out = df.copy()
    preserve_values_by_column = dict(preserve_values_by_column or {})
    for col, codes in dict(missing_codes_by_column or {}).items():
        if col not in out.columns:
            continue
        if _is_pipeline_derived_column(str(col)):
            continue
        preserve = preserve_values_by_column.get(col, preserve_values_by_column.get(str(col).upper(), []))
        codes = _remove_preserved_codes(codes, preserve)
        expanded_codes = _expand_missing_code_variants(codes)
        if expanded_codes:
            series = out[col]
            mask = series.isin(expanded_codes)
            numeric_codes: list[float] = []
            for code in expanded_codes:
                try:
                    numeric_codes.append(float(code))
                except (TypeError, ValueError):
                    continue
            if numeric_codes:
                numeric_series = pd.to_numeric(series, errors="coerce")
                mask = mask | numeric_series.isin(numeric_codes)
            out.loc[mask, col] = np.nan
    return out


def recode_binary_columns(df: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
    """Recode configured binary features to nullable integer 0/1 values.

    The nullable Int64 dtype preserves missing values while exporting clean
    manuscript/audit CSV values as 0/1 rather than 0.0/1.0. Code 2 is mapped
    to 0 only for binary NACC fields where 1 denotes yes/present and 2 denotes
    no/absent. Other non-0/1 values are left as missing after coercion.
    """
    out = df.copy()
    for col in binary_columns:
        if col in out.columns:
            numeric = pd.to_numeric(out[col], errors="coerce")
            recoded = numeric.map({0: 0, 1: 1, 2: 0}).where(numeric.notna(), pd.NA)
            out[col] = recoded.astype("Int64")
    return out



def _object_series_looks_numeric(series: pd.Series, sample_size: int = 200) -> bool:
    """Fast precheck to avoid expensive numeric coercion of free-text columns."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return True
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample.ne("")].head(int(sample_size))
    if sample.empty:
        return False
    numeric_pattern = re.compile(r"^-?\d+(?:\.\d+)?$")
    return bool(sample.map(lambda x: bool(numeric_pattern.match(x))).all())

def _is_binary_like_series(series: pd.Series) -> bool:
    """Return True when all nonmissing values are exactly 0/1 or 0.0/1.0."""
    # Duplicate column names can make df[col] return a DataFrame; never infer
    # binary status from a multi-column object.
    if not isinstance(series, pd.Series):
        return False
    if series.dropna().empty:
        return False
    if not _object_series_looks_numeric(series):
        return False
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().sum() != series.isna().sum():
        return False
    values = set(numeric.dropna().astype(float).unique().tolist())
    return values.issubset({0.0, 1.0})


def format_binary_columns_as_nullable_int(df: pd.DataFrame, binary_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Format true binary-like columns as pandas nullable Int64 for CSV export."""
    out = df.copy()
    candidate_cols = list(binary_columns or [])
    if not candidate_cols:
        candidate_cols = [c for c in out.columns if _is_binary_like_series(out[c])]
    for col in candidate_cols:
        if col in out.columns and _is_binary_like_series(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _is_integral_numeric_series(series: pd.Series, tolerance: float = 1e-9) -> bool:
    """Return True when nonmissing numeric values are all whole numbers.

    This is used only for CSV presentation. It converts integer-coded clinical
    variables, categorical codes, counts, and visit fields to nullable integer
    output so files show 0/1/2 rather than 0.0/1.0/2.0. It does not alter
    continuous variables that contain genuine fractional values such as CDRSUM,
    CDRGLOB, NPPMIH, predicted probabilities, or neural embeddings.
    """
    if not isinstance(series, pd.Series) or series.dropna().empty:
        return False
    if not _object_series_looks_numeric(series):
        return False
    numeric = pd.to_numeric(series, errors="coerce")
    # Do not coerce object columns with nonnumeric labels.
    if numeric.isna().sum() != series.isna().sum():
        return False
    valid = numeric.dropna().astype(float)
    if valid.empty:
        return False
    return bool((valid.sub(valid.round()).abs() <= tolerance).all())


def format_integral_numeric_columns_as_nullable_int(
    df: pd.DataFrame,
    exclude_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Format whole-number numeric columns as nullable Int64 for clean CSV export.

    The conversion is presentation-safe: only columns whose nonmissing values are
    exactly integral are converted. Fractional variables remain floats. Missing
    values remain blank/NA in the exported CSV.
    """
    out = df.copy()
    excluded = set(exclude_columns or [])
    for col in out.columns:
        if col in excluded:
            continue
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype("Int64")
            continue
        if _is_integral_numeric_series(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _coerce_numeric_series_for_trajectory(series: pd.Series) -> pd.Series:
    """Return a numeric series suitable for longitudinal feature calculations."""
    return pd.to_numeric(series, errors="coerce")


def _safe_annualized_change(delta: pd.Series, days: pd.Series) -> pd.Series:
    """Annualize change using prior-visit interval; invalid/nonpositive intervals become missing."""
    denom = pd.to_numeric(days, errors="coerce") / 365.25
    out = pd.to_numeric(delta, errors="coerce") / denom.replace({0.0: np.nan})
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def longitudinal_trajectory_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return features derived only from visits prior to or at the current MCI visit."""
    return [c for c in df.columns if str(c).startswith("traj_")]


def add_longitudinal_trajectory_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add leakage-safe longitudinal patient-state trajectory features.

    The outcome is next-visit dementia progression, so trajectory predictors must
    use only information available at the current MCI visit or earlier. This
    function sorts records within participant by visit date/visit number, creates
    prior values by within-participant shift(1), and computes prior-to-current
    changes and annualized slopes. It never uses next-visit diagnosis or any
    future measurements.
    """
    traj_cfg = (cfg.get("feature_engineering", {}) or {}).get("trajectory_features", {}) or {}
    if not bool(traj_cfg.get("enabled", cfg.get("feature_engineering", {}).get("add_trajectory_features", False))):
        return df
    out = df.copy()
    id_col = str(cfg.get("id_col", "NACCID"))
    visit_num_col = str(cfg.get("visit_number_col", "NACCVNUM"))
    y_col = str(cfg.get("visit_year_col", "VISITYR"))
    m_col = str(cfg.get("visit_month_col", "VISITMO"))
    d_col = str(cfg.get("visit_day_col", "VISITDAY"))
    if id_col not in out.columns:
        return out

    # Build a stable within-participant temporal order. Date is preferred, then
    # visit number, then original row order. This preserves deterministic behavior
    # for partially missing dates.
    if all(c in out.columns for c in [y_col, m_col, d_col]):
        visit_dates = pd.to_datetime(
            {
                "year": pd.to_numeric(out[y_col], errors="coerce"),
                "month": pd.to_numeric(out[m_col], errors="coerce"),
                "day": pd.to_numeric(out[d_col], errors="coerce"),
            },
            errors="coerce",
        )
    else:
        visit_dates = pd.Series(pd.NaT, index=out.index)
    work = pd.DataFrame({
        "_orig_order": np.arange(len(out)),
        "_id": out[id_col].astype(str).to_numpy(),
        "_date": visit_dates.to_numpy(),
    })
    work["_visit_number"] = pd.to_numeric(out[visit_num_col], errors="coerce") if visit_num_col in out.columns else np.arange(len(out))
    work = work.sort_values(["_id", "_date", "_visit_number", "_orig_order"], kind="mergesort")
    ordered_index = work["_orig_order"].to_numpy()
    prior_count = work.groupby("_id").cumcount()
    prior_days = work.groupby("_id")["_date"].diff().dt.days
    first_dates = work.groupby("_id")["_date"].transform("first")
    first_days = (work["_date"] - first_dates).dt.days

    # Core trajectory availability/time features.
    tmp = pd.DataFrame(index=ordered_index)
    tmp["traj_has_prior_mci_visit"] = (prior_count > 0).astype(int).to_numpy()
    tmp["traj_prior_mci_visit_count"] = prior_count.astype("Int64").to_numpy()
    tmp["traj_days_since_prior_mci_visit"] = prior_days.astype("Float64").to_numpy()
    tmp["traj_years_since_prior_mci_visit"] = (pd.to_numeric(prior_days, errors="coerce") / 365.25).astype("Float64").to_numpy()
    tmp["traj_days_since_first_mci_visit"] = first_days.astype("Float64").to_numpy()
    tmp["traj_years_since_first_mci_visit"] = (pd.to_numeric(first_days, errors="coerce") / 365.25).astype("Float64").to_numpy()

    # Numeric current/prior/change/slope trajectories. Keep a conservative list;
    # all columns are existing current-visit predictors, not future outcomes.
    default_numeric = [
        "NACCMMSE", "NACCMOCA", "CDRSUM", "CDRGLOB", "NACCGDS", "NACCAGE", "EDUC",
        "medication_count", "medication_psychotropic_count", "medication_neuro_count", "medication_cardiometabolic_count",
        "llm_medication_state_domain_count", "llm_polypharmacy_complexity_score",
    ]
    numeric_cols = list(traj_cfg.get("numeric_columns", default_numeric) or default_numeric)
    max_numeric = int(traj_cfg.get("max_numeric_columns", 40))
    numeric_cols = [c for c in numeric_cols if c in out.columns][:max_numeric]
    ordered = out.iloc[ordered_index].copy()
    ordered_groups = work["_id"].to_numpy()
    for col in numeric_cols:
        current = _coerce_numeric_series_for_trajectory(ordered[col]).reset_index(drop=True)
        prior = current.groupby(ordered_groups, sort=False).shift(1)
        delta = current - prior
        slope = _safe_annualized_change(delta, prior_days.reset_index(drop=True))
        tmp[f"traj_prior_{col}"] = prior.to_numpy()
        tmp[f"traj_delta_{col}"] = delta.to_numpy()
        tmp[f"traj_annualized_delta_{col}"] = slope.to_numpy()

    # Binary transition indicators: current-on, prior-off => new exposure/symptom.
    default_binary = [
        "med_cat_antidepressant", "med_cat_antipsychotic", "med_cat_benzodiazepine", "med_cat_sleep", "med_cat_dementia",
        "med_cat_anticholinergic", "DEL", "HALL", "AGIT", "DEPD", "ANX", "PARK", "DIABETES", "HYPERTEN",
        "llm_psychotropic_exposure", "llm_sedative_hypnotic_exposure", "llm_cognitive_symptomatic_treatment",
        "llm_polypharmacy_complexity_high", "llm_anticholinergic_exposure",
    ]
    binary_cols = list(traj_cfg.get("binary_transition_columns", default_binary) or default_binary)
    max_binary = int(traj_cfg.get("max_binary_transition_columns", 50))
    binary_cols = [c for c in binary_cols if c in out.columns][:max_binary]
    for col in binary_cols:
        current = pd.to_numeric(ordered[col], errors="coerce").reset_index(drop=True)
        prior = current.groupby(ordered_groups, sort=False).shift(1)
        tmp[f"traj_prior_{col}"] = prior.to_numpy()
        tmp[f"traj_new_{col}"] = ((current.eq(1)) & (prior.eq(0))).astype("Int64").to_numpy()
        tmp[f"traj_stopped_{col}"] = ((current.eq(0)) & (prior.eq(1))).astype("Int64").to_numpy()

    # Restore original row order and append to output.
    tmp = tmp.sort_index()
    for col in tmp.columns:
        out[col] = tmp[col].to_numpy()
    return out

def structured_medication_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return medication features derived from structured medication columns only."""
    named = {
        "medication_count", "medication_psychotropic_count", "medication_neuro_count",
        "medication_cardiometabolic_count",
    }
    return [c for c in df.columns if c.startswith("med_cat_") or c in named]


def llm_medication_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric clinical LLM medication-state features used for modeling.

    Process and audit indicators are intentionally excluded. For example,
    llm_parse_ok indicates whether a real Ollama JSON object was parsed, and
    llm_confidence/manual-review fields describe abstraction certainty rather than
    medication-state burden. They remain in output CSVs for auditability but are not
    model predictors.
    """
    excluded_exact = {"llm_parse_ok", "llm_confidence", "llm_manual_review"}
    excluded_suffixes = ("_summary", "_provider", "_error", "_rationale", "_text", "_status", "_source", "_reason")
    out = []
    for col in df.columns:
        if not col.startswith("llm_"):
            continue
        if col in excluded_exact or col.endswith(excluded_suffixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def neural_text_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric BioClinicalBERT/SapBERT medication-text representation features."""
    prefixes = ("bioclinicalbert_medtxt_", "sapbert_medtxt_", "clinicalbert_sapbert_medtxt_")
    out = []
    for col in df.columns:
        if str(col).startswith(prefixes) and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def medication_feature_columns(
    df: pd.DataFrame,
    *,
    include_llm: bool = True,
    include_neural_text: bool = False,
) -> list[str]:
    """Return medication features used by medication-aware models."""
    cols = structured_medication_feature_columns(df)
    if include_llm:
        cols += llm_medication_feature_columns(df)
    if include_neural_text:
        cols += neural_text_feature_columns(df)
    return list(dict.fromkeys([c for c in cols if c in df.columns]))


def build_feature_lists(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    binary_columns: List[str],
    include_medication: bool,
    *,
    include_llm_medication: bool = True,
    include_neural_text: bool = False,
) -> Dict[str, List[str]]:
    present_categorical = [c for c in categorical_columns if c in df.columns]
    present_numeric = [c for c in numeric_columns if c in df.columns]
    present_binary = [c for c in binary_columns if c in df.columns]
    if include_medication:
        for col in medication_feature_columns(df, include_llm=include_llm_medication, include_neural_text=include_neural_text):
            if col not in present_categorical and col not in present_numeric and col not in present_binary:
                present_numeric.append(col)
    return {"categorical": present_categorical, "numeric": present_numeric, "binary": present_binary}

def _onehot_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(feature_lists: Dict[str, List[str]], *, scale_numeric: bool) -> ColumnTransformer:
    transformers = []
    if feature_lists["categorical"]:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _onehot_dense()),
        ]), feature_lists["categorical"]))
    if feature_lists["numeric"]:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), feature_lists["numeric"]))
    if feature_lists["binary"]:
        transformers.append(("binary", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), feature_lists["binary"]))
    if not transformers:
        raise ValueError("No usable features were found for modeling.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)

def make_feature_frame(df: pd.DataFrame, feature_lists: Dict[str, List[str]]) -> pd.DataFrame:
    cols = feature_lists["categorical"] + feature_lists["numeric"] + feature_lists["binary"]
    return df[cols].copy()

def infer_column_type(column: str) -> str:
    """Classify a model feature into a clinically interpretable source domain.

    This label is written to Step 2 feature-summary files so that reviewers can
    distinguish clinical, structured-medication, LLM-medication, and neural
    medication-text representation variables without manually reading the code.
    The function is intentionally deterministic and audit-oriented; it does not
    change model inputs.
    """
    col = str(column).strip()
    upper = col.upper()
    lower = col.lower()

    if lower.startswith("traj_"):
        if "med" in lower or "llm_" in lower:
            return "longitudinal_medication_trajectory"
        return "longitudinal_clinical_trajectory"
    if lower.startswith("llm_"):
        return "medication_llm"
    if lower.startswith(("bioclinicalbert_medtxt_", "sapbert_medtxt_", "clinicalbert_sapbert_medtxt_")):
        return "medication_neural"
    if lower in {"medication_count", "medication_text"} or lower.startswith("medication_"):
        return "medication_structured"
    if upper.startswith("DRUG"):
        return "medication_source_text"

    demographic = {"SEX", "HISPANIC", "RACE", "NACCAGE", "EDUC"}
    genetic = {"NACCAPOE"}
    visit_time = {
        "NACCVNUM", "VISITYR", "VISITMO", "VISITDAY", "NACCDAYS",
        "PRIOR_MCI_VISIT_COUNT", "DAYS_SINCE_PRIOR_MCI_VISIT", "DAYS_SINCE_FIRST_MCI_VISIT",
        "prior_mci_visit_count".upper(), "days_since_prior_mci_visit".upper(), "days_since_first_mci_visit".upper(),
    }
    cognitive_severity = {"NACCMMSE", "NACCMOCA", "CDRSUM", "CDRGLOB", "NACCGDS", "NACCUDSD"}
    neuropsychiatric = {
        "DEL", "HALL", "AGIT", "DEPD", "ANX", "ELAT", "APA", "DISN",
        "IRR", "MOT", "NITE", "APP", "DEP2YRS", "NPSYDEV", "PSYCDIS",
        "BIPOLDX", "SCHIZOP", "PTSDDX",
    }
    vascular_metabolic = {"CVHATT", "CBSTROKE", "DIABETES", "HYPERTEN", "HYPERCHO", "B12DEF", "THYROID"}
    neurologic = {"PARK", "GAITNPH", "HYCEPH", "BRNINJ"}
    functional_geriatric = {"ARTH", "INCONTU", "INCONTF"}
    lifestyle_substance = {"SMOKYRS", "PACKSPER", "QUITSMOK", "ALCOCCAS", "ALCFREQ"}
    neuropathology = {
        "NPFORMVER", "NPADNC", "NACCBRAA", "NACCNEUR", "NACCLEWY",
        "NPLBOD", "NPINF", "NPPMIH", "NPTAN", "NPLAC", "NPTHAL",
        "NPART", "NPOANG", "NPFTDTAU",
    }

    if upper in demographic:
        return "demographic"
    if upper in genetic:
        return "genetic"
    if upper in visit_time:
        return "visit_time"
    if upper in cognitive_severity:
        return "cognitive_clinical_severity"
    if upper in neuropsychiatric:
        return "neuropsychiatric"
    if upper in vascular_metabolic:
        return "vascular_metabolic_comorbidity"
    if upper in neurologic:
        return "neurologic_comorbidity"
    if upper in functional_geriatric:
        return "functional_geriatric_comorbidity"
    if upper in lifestyle_substance:
        return "lifestyle_substance"
    if upper in neuropathology:
        return "neuropathology_anchor_only"
    return "clinical_other"


def build_feature_summary(df: pd.DataFrame, feature_lists: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for feature_type, cols in feature_lists.items():
        for col in cols:
            rows.append({
                "column": col,
                "column_type": infer_column_type(col),
                "feature_type": feature_type,
                "missing_n": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
            })
    return pd.DataFrame(rows).sort_values(["feature_type", "column"]).reset_index(drop=True)
