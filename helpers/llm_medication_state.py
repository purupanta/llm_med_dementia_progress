"""
Project: dementia_progression
File: helpers/llm_medication_state.py

Synopsis:
    Optional local LLM medication-state abstraction utilities with fast YAML-controlled
    settings, provider-strict execution, persistent caching, and transparent fallback behavior.

Author:
    puru panta (purupanta@uky.edu)

Date Created:
    2026-05-19

Last Updated:
    2026-05-27

Version:
    2.13

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

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helpers.progress import progress_bar
from helpers.medications import normalize_medication_text, canonicalize_medication_text_for_llm, load_medication_alias_map, build_frozen_encoder_medication_normalizer

LLM_MEDICATION_STATE_POLICY_VERSION = "v2_13_raw_keyed_dictionary_fast_cache"

# Numeric medication-state features written back to the visit-level modeling dataset.
# Important: llm_parse_ok is retained for audit compatibility but is excluded from
# model feature selection in helpers/features.py because it is a process indicator.
LLM_FEATURE_COLUMNS = [
    "llm_parse_ok",
    "llm_anticholinergic_exposure",
    "llm_psychotropic_exposure",
    "llm_sedative_hypnotic_exposure",
    "llm_cognitive_symptomatic_treatment",
    "llm_polypharmacy_complexity_high",
    "llm_neuropsychiatric_treatment_signal",
    "llm_cardiometabolic_treatment_signal",
    "llm_pain_sedation_signal",
    "llm_neurologic_treatment_signal",
    "llm_medication_state_domain_count",
    "llm_polypharmacy_complexity_score",
    "llm_confidence",
    "llm_manual_review",
]

TEXT_AUDIT_COLUMNS = [
    "medication_text",
    "canonical_medication_text",
    "effective_provider",
    "requested_provider",
    "llm_model",
    "llm_input_normalization_requested_backend",
    "llm_input_normalization_backend",
    "llm_input_encoder_model",
    "llm_input_encoder_status",
    "llm_input_encoder_similarity_threshold",
    "llm_input_encoder_vocab_size",
    "feature_source",
    "llm_summary",
    "llm_rationale",
    "llm_error",
    "llm_parse_error",
    "llm_raw_response",
    "llm_response_json",
]

LLM_AUDIT_STATUS_COLUMNS = [
    "llm_processing_policy_version",
    "llm_status",
    "llm_attempted",
    "fallback_parse_ok",
    "llm_attempt_count",
    "llm_fallback_reason",
    "llm_elapsed_ms",
    "llm_schema_violation_count",
    "llm_schema_violation_fields",
    "llm_binary_repair_applied",
    "llm_domain_count_raw",
    "llm_domain_count_recomputed",
    "llm_domain_count_repair_applied",
    "llm_polypharmacy_repair_applied",
    "llm_confidence_raw",
    "llm_confidence_repair_applied",
    "llm_quality_repair_applied",
    "llm_clinical_anchor_repair_applied",
    "llm_clinical_anchor_repair_fields",
    "llm_formula_fields_computed_in_python",
    "llm_abstraction_unit",
    "llm_drug_dictionary_aggregation_applied",
    "llm_drug_dictionary_token_count",
    "llm_drug_dictionary_ollama_token_count",
    "llm_drug_dictionary_fallback_token_count",
    "llm_drug_dictionary_tokens",
]

LLM_CERTIFICATION_COLUMNS = [
    "llm_structural_quality_pass",
    "llm_certified_for_primary_analysis",
    "llm_certification_status",
    "llm_certification_reasons",
]

DRUG_DICTIONARY_IDENTITY_COLUMNS = [
    "raw_drug_name",
    "normalized_drug_name",
    "canonical_drug_name",
    "llm_dictionary_key",
    "mapping_provider",
    "mapping_similarity_score",
    "llm_abstraction_status",
]

KEYWORDS = {
    "anticholinergic": [
        "diphenhydramine", "benadryl", "oxybutynin", "tolterodine", "detrol", "solifenacin", "darifenacin",
        "benztropine", "trihexyphenidyl", "hydroxyzine", "amitriptyline", "elavil", "nortriptyline", "paroxetine",
        "cyclobenzaprine", "promethazine", "meclizine",
    ],
    "antidepressant": [
        "sertraline", "citalopram", "escitalopram", "fluoxetine", "paroxetine", "venlafaxine", "duloxetine", "cymbalta",
        "bupropion", "mirtazapine", "remeron", "trazodone", "amitriptyline", "nortriptyline", "desvenlafaxine",
    ],
    "antipsychotic": [
        "quetiapine", "olanzapine", "risperidone", "aripiprazole", "haloperidol", "clozapine", "ziprasidone",
        "geodon", "paliperidone", "lurasidone",
    ],
    "benzodiazepine": [
        "alprazolam", "lorazepam", "ativan", "clonazepam", "diazepam", "temazepam", "oxazepam", "chlordiazepoxide",
        "clorazepate", "triazolam",
    ],
    "sleep": ["zolpidem", "ambien", "eszopiclone", "lunesta", "zaleplon", "sonata", "temazepam", "ramelteon", "suvorexant", "melatonin"],
    "dementia": ["donepezil", "aricept", "rivastigmine", "exelon", "galantamine", "razadyne", "memantine", "namenda"],
    "cardiometabolic": [
        "lisinopril", "losartan", "valsartan", "amlodipine", "metoprolol", "carvedilol", "hydrochlorothiazide",
        "chlorthalidone", "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "pitavastatin", "livalo",
        "metformin", "insulin", "glipizide", "glyburide", "semaglutide", "sitagliptin", "warfarin", "apixaban", "rivaroxaban", "clopidogrel",
        "aspirin", "levothyroxine", "synthroid", "liothyronine", "cytomel", "methimazole", "propylthiouracil", "ptu",
    ],
    "pain_sedation": ["morphine", "oxycodone", "hydrocodone", "vicodin", "fentanyl", "tramadol", "codeine", "hydromorphone", "gabapentin", "neurontin", "pregabalin", "lyrica"],
    "neurologic": ["levodopa", "carbidopa", "sinemet", "ropinirole", "pramipexole", "selegiline", "rasagiline", "levetiracetam", "lamotrigine", "topiramate", "valproate", "divalproex", "gabapentin", "neurontin", "pregabalin", "lyrica"],
}

CLINICAL_LLM_REQUIRED_KEYS = [
    "parse_ok",
    "anticholinergic_exposure",
    "psychotropic_exposure",
    "sedative_hypnotic_exposure",
    "cognitive_symptomatic_treatment",
    "neuropsychiatric_treatment_signal",
    "cardiometabolic_treatment_signal",
    "pain_sedation_signal",
    "neurologic_treatment_signal",
    "confidence",
    "manual_review",
    "summary",
    "rationale",
]

# v1.48 speed design: the LLM returns only clinically judgment-dependent fields.
# Formula-derived fields are computed in Python after parsing to reduce tokens,
# improve internal consistency, and avoid unnecessary row-level repairs.
OPTIONAL_LEGACY_FORMULA_KEYS = [
    "polypharmacy_complexity_high",
    "medication_state_domain_count",
    "polypharmacy_complexity_score",
]

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "required": CLINICAL_LLM_REQUIRED_KEYS,
    "properties": {
        "parse_ok": {"type": "integer", "enum": [0, 1]},
        "anticholinergic_exposure": {"type": "integer", "enum": [0, 1]},
        "psychotropic_exposure": {"type": "integer", "enum": [0, 1]},
        "sedative_hypnotic_exposure": {"type": "integer", "enum": [0, 1]},
        "cognitive_symptomatic_treatment": {"type": "integer", "enum": [0, 1]},
        "neuropsychiatric_treatment_signal": {"type": "integer", "enum": [0, 1]},
        "cardiometabolic_treatment_signal": {"type": "integer", "enum": [0, 1]},
        "pain_sedation_signal": {"type": "integer", "enum": [0, 1]},
        "neurologic_treatment_signal": {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "manual_review": {"type": "integer", "enum": [0, 1]},
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
    },
}


class OllamaMedicationStateError(RuntimeError):
    """Exception carrying the last raw Ollama response for row-level audit."""

    def __init__(self, message: str, *, raw_response: str = "", parse_error: str = "", attempt_count: int = 0):
        super().__init__(message)
        self.raw_response = str(raw_response or "")
        self.parse_error = str(parse_error or "")
        self.attempt_count = int(attempt_count or 0)


def _truncate(value: Any, limit: int = 4000) -> str:
    return _audit_string(value, "")[: int(limit)]



def _bool01(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _audit_string(value: Any, default: str = "") -> str:
    """Return a clean audit string without propagating pandas NaN as literal 'nan'."""
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        if value is None:
            return default
    return str(value)


def _clean_text(text: Any) -> str:
    """Canonicalize medication text and remove blank/non-codable placeholders."""
    return normalize_medication_text(text)


def _drug_dictionary_identity(text: Any, *, mapping_provider: str = "", mapping_similarity_score: float | str | None = None) -> dict[str, Any]:
    """Return stable identity fields for unique-drug dictionary rows.

    The stable key is the SHA-256 hash of normalized_drug_name. It is independent
    of row order and therefore safe for cache lookup, dictionary joins, and
    aggregation back to visit-level medication lists.
    """
    raw = _audit_string(text, "")
    normalized = normalize_medication_text(raw)
    canonical = normalized
    stable_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""
    try:
        similarity = float(mapping_similarity_score) if mapping_similarity_score is not None and str(mapping_similarity_score) != "" else 1.0
    except Exception:
        similarity = 1.0
    return {
        "raw_drug_name": raw,
        "normalized_drug_name": normalized,
        "canonical_drug_name": canonical,
        "llm_dictionary_key": stable_key,
        "mapping_provider": str(mapping_provider or "normalized_exact"),
        "mapping_similarity_score": max(0.0, min(1.0, similarity)),
    }


def _drug_dictionary_key(text: Any) -> str:
    return str(_drug_dictionary_identity(text).get("llm_dictionary_key", ""))


def _token_count(medication_text: str) -> int:
    normalized = normalize_medication_text(medication_text)
    if not normalized:
        return 0
    return len([t for t in re.split(r"\s*\|\s*", normalized) if t.strip()])


def _split_medication_tokens(medication_text: Any) -> list[str]:
    """Return unique normalized drug-name tokens from a medication text string."""
    normalized = normalize_medication_text(medication_text)
    if not normalized:
        return []
    return list(dict.fromkeys([t.strip() for t in re.split(r"\s*\|\s*", normalized) if t.strip()]))


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _merge_model_profile_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Apply an optional model profile without mutating the original config.

    Launcher-selected Ollama models must remain authoritative after profile
    expansion. Earlier versions applied the named profile after environment
    overrides, which could silently replace a probed/installed fallback model
    such as qwen2.5:3b-instruct with the profile default
    qwen2.5:7b-instruct. When that default was not installed, row-level
    /api/generate calls failed with HTTP 404 although the launcher probe had
    already selected a valid installed model.
    """
    merged = dict(settings or {})
    launcher_selected_model = str(merged.get("_launcher_selected_ollama_model", "") or "").strip()
    profile_name = str(merged.get("model_profile", "fast_qwen3b") or "").strip()
    profiles = merged.get("model_profiles", {}) or {}
    if profile_name and profile_name != "custom" and profile_name in profiles and isinstance(profiles.get(profile_name), dict):
        profile_values = dict(profiles[profile_name])
        for key, value in profile_values.items():
            # Profiles are explicit runtime recipes. Runtime model failover is
            # applied immediately below so the selected installed model is not
            # overwritten by the named profile default.
            merged[key] = value
    if launcher_selected_model:
        merged["ollama_model"] = launcher_selected_model
    return merged


def _comparison_feature_vector(row: dict[str, Any] | pd.Series) -> dict[str, int]:
    return {key: int(float(row.get("llm_" + key, row.get(key, 0)) or 0)) for key in DOMAIN_BINARY_KEYS}


BINARY_SCHEMA_KEYS = [
    "anticholinergic_exposure",
    "psychotropic_exposure",
    "sedative_hypnotic_exposure",
    "cognitive_symptomatic_treatment",
    "polypharmacy_complexity_high",
    "neuropsychiatric_treatment_signal",
    "cardiometabolic_treatment_signal",
    "pain_sedation_signal",
    "neurologic_treatment_signal",
]

DOMAIN_BINARY_KEYS = [
    "anticholinergic_exposure",
    "psychotropic_exposure",
    "sedative_hypnotic_exposure",
    "cognitive_symptomatic_treatment",
    "polypharmacy_complexity_high",
    "neuropsychiatric_treatment_signal",
    "cardiometabolic_treatment_signal",
    "pain_sedation_signal",
    "neurologic_treatment_signal",
]


def _binary01_with_violation(value: Any, field_name: str) -> tuple[int, bool, str]:
    """Coerce an LLM binary field to 0/1 and report schema violations.

    Positive numeric counts such as 2 or 3 are repaired to 1 because they indicate
    the exposure/domain is present, but they are still flagged as schema violations
    because the requested output is binary.
    """
    if isinstance(value, bool):
        return int(value), False, ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if np.isnan(value):
                return 0, True, field_name
        except Exception:
            pass
        num = float(value)
        if num in {0.0, 1.0}:
            return int(num), False, ""
        return int(num > 0.0), True, field_name
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "present", "positive"}:
            return 1, False, ""
        if text in {"0", "false", "no", "n", "absent", "negative", "none", ""}:
            return 0, False, ""
        try:
            num = float(text)
            if num in {0.0, 1.0}:
                return int(num), False, ""
            return int(num > 0.0), True, field_name
        except Exception:
            return 0, True, field_name
    return int(bool(value)), True, field_name


def _clinical_anchor_binaries(medication_text: str) -> dict[str, int]:
    """High-specificity medication-class anchors used to audit and repair missed LLM positives.

    The anchors are intentionally limited to medication classes already represented
    in the project feature schema. They do not estimate dementia risk and they do
    not replace the LLM; they prevent clearly named medications such as Cymbalta,
    Ambien, Ativan, Razadyne, tolterodine, or simvastatin from being coded as if
    no relevant medication-state domain were present.
    """
    text = _clean_text(medication_text)
    n_meds = _token_count(text)
    anticholinergic = _contains_any(text, KEYWORDS["anticholinergic"])
    antidepressant = _contains_any(text, KEYWORDS["antidepressant"])
    antipsychotic = _contains_any(text, KEYWORDS["antipsychotic"])
    benzodiazepine = _contains_any(text, KEYWORDS["benzodiazepine"])
    sleep = _contains_any(text, KEYWORDS["sleep"])
    dementia = _contains_any(text, KEYWORDS["dementia"])
    cardiometabolic = _contains_any(text, KEYWORDS["cardiometabolic"])
    pain_sedation = _contains_any(text, KEYWORDS["pain_sedation"])
    neurologic = _contains_any(text, KEYWORDS["neurologic"])
    psychotropic = antidepressant or antipsychotic or benzodiazepine or sleep
    sedative = benzodiazepine or sleep
    neuropsychiatric = psychotropic or dementia
    return {
        "anticholinergic_exposure": int(anticholinergic),
        "psychotropic_exposure": int(psychotropic),
        "sedative_hypnotic_exposure": int(sedative),
        "cognitive_symptomatic_treatment": int(dementia),
        "polypharmacy_complexity_high": int(n_meds >= 5),
        "neuropsychiatric_treatment_signal": int(neuropsychiatric),
        "cardiometabolic_treatment_signal": int(cardiometabolic),
        "pain_sedation_signal": int(pain_sedation),
        "neurologic_treatment_signal": int(neurologic),
    }


def _validated_medication_state_features(src: dict[str, Any], medication_text: str = "") -> dict[str, Any]:
    """Return validated/recomputed medication-state features plus QC audit fields.

    v1.48 asks the LLM for clinical judgment fields only. Formula-derived fields
    (polypharmacy flag, domain count, and complexity score) are computed in Python
    after parsing. Legacy cached rows that contain these fields are still accepted
    and audited if inconsistent.
    """
    violations: list[str] = []
    binaries: dict[str, int] = {}
    formula_fields_present = any(k in src for k in OPTIONAL_LEGACY_FORMULA_KEYS)
    for key in BINARY_SCHEMA_KEYS:
        if key == "polypharmacy_complexity_high" and key not in src:
            # This field is now formula-derived, not LLM-generated. It is filled
            # after medication count is known and should not be counted as a
            # binary repair when absent from a v1.48 LLM response.
            val, bad, bad_field = 0, False, ""
        else:
            val, bad, bad_field = _binary01_with_violation(src.get(key, 0), key)
        binaries[key] = int(val)
        if bad and bad_field:
            violations.append(bad_field)

    # Preserve the model's original polypharmacy flag only for legacy responses.
    # In v1.48 the high-polypharmacy flag is intentionally computed in Python.
    poly_raw_before_anchor = int(binaries.get("polypharmacy_complexity_high", 0))
    clinical_anchor = _clinical_anchor_binaries(medication_text)
    clinical_anchor_fields: list[str] = []
    for key in BINARY_SCHEMA_KEYS:
        if key == "polypharmacy_complexity_high":
            continue
        if int(clinical_anchor.get(key, 0)) == 1 and int(binaries.get(key, 0)) == 0:
            binaries[key] = 1
            clinical_anchor_fields.append(key)

    med_count = _token_count(medication_text)
    poly_raw = poly_raw_before_anchor
    poly_recomputed = int(med_count >= 5) if med_count >= 0 else poly_raw
    poly_repair = int(formula_fields_present and poly_raw != poly_recomputed)
    binaries["polypharmacy_complexity_high"] = poly_recomputed

    raw_domain_value = src.get("medication_state_domain_count", "")
    raw_domain = _safe_float(raw_domain_value, 0.0) if "medication_state_domain_count" in src else ""
    domain_recomputed = int(sum(int(binaries[k]) for k in DOMAIN_BINARY_KEYS))
    domain_repair = int("medication_state_domain_count" in src and int(round(float(raw_domain))) != domain_recomputed)

    raw_score_value = src.get("polypharmacy_complexity_score", "")
    raw_score = _safe_float(raw_score_value, 0.0) if "polypharmacy_complexity_score" in src else ""
    recomputed_score = 0
    if med_count >= 5:
        recomputed_score += 2
    if med_count >= 10:
        recomputed_score += 1
    for key in [
        "anticholinergic_exposure",
        "psychotropic_exposure",
        "sedative_hypnotic_exposure",
        "cardiometabolic_treatment_signal",
        "neurologic_treatment_signal",
    ]:
        recomputed_score += int(binaries.get(key, 0))
    recomputed_score = min(5, int(recomputed_score))
    score_repair = int("polypharmacy_complexity_score" in src and int(round(float(raw_score))) != recomputed_score)

    raw_conf = _safe_float(src.get("confidence", np.nan), np.nan)
    confidence_repair = 0
    if raw_conf != raw_conf or raw_conf <= 0.0 or raw_conf > 1.0:
        # Confidence is certainty in the extraction, not clinical risk intensity.
        # Empty/noninformative rows receive low confidence; repaired LLM rows receive
        # moderate confidence and manual review.
        raw_conf_display = "" if raw_conf != raw_conf else raw_conf
        conf = 0.20 if med_count == 0 else (0.60 if violations or domain_repair or poly_repair or score_repair else 0.75)
        confidence_repair = 1
    else:
        raw_conf_display = raw_conf
        conf = max(0.0, min(1.0, float(raw_conf)))

    manual_raw, manual_bad, manual_field = _binary01_with_violation(src.get("manual_review", 0), "manual_review")
    if manual_bad:
        violations.append(manual_field)
    clinical_anchor_repair = int(bool(clinical_anchor_fields))
    repair_any = int(bool(violations) or domain_repair or poly_repair or score_repair or confidence_repair or clinical_anchor_repair)
    # Structural repairs are formula/schema normalization steps applied after a
    # parseable Ollama JSON response. They should be audited, but they should not
    # automatically convert a true LLM row into manual review. Manual review is
    # reserved for explicit model uncertainty or unreliability, especially an
    # invalid/missing confidence value or a bad manual_review field.
    manual_review = int(manual_raw or confidence_repair or manual_bad)

    return {
        **binaries,
        "medication_state_domain_count": domain_recomputed,
        "polypharmacy_complexity_score": recomputed_score,
        "confidence": conf,
        "manual_review": manual_review,
        "__schema_violation_count": int(len(set(violations))),
        "__schema_violation_fields": ";".join(sorted(set(violations))),
        "__binary_repair_applied": int(bool(violations)),
        "__domain_count_raw": raw_domain,
        "__domain_count_recomputed": domain_recomputed,
        "__domain_count_repair_applied": domain_repair,
        "__polypharmacy_repair_applied": poly_repair,
        "__confidence_raw": raw_conf_display,
        "__confidence_repair_applied": confidence_repair,
        "__quality_repair_applied": repair_any,
        "__clinical_anchor_repair_applied": clinical_anchor_repair,
        "__clinical_anchor_repair_fields": ";".join(sorted(set(clinical_anchor_fields))),
        "__formula_fields_computed_in_python": int(not formula_fields_present),
    }


def local_clinical_abstraction(medication_text: str) -> dict[str, Any]:
    """Transparent fallback abstraction used when a true LLM is unavailable.

    This function intentionally returns the same schema as the LLM response. It is
    used for unit tests, smoke tests, and fail-safe runs. Outputs generated by this
    fallback should not be described in a manuscript as LLM-derived.
    """
    text = _clean_text(medication_text)
    n_meds = _token_count(text)
    anticholinergic = _contains_any(text, KEYWORDS["anticholinergic"])
    antidepressant = _contains_any(text, KEYWORDS["antidepressant"])
    antipsychotic = _contains_any(text, KEYWORDS["antipsychotic"])
    benzodiazepine = _contains_any(text, KEYWORDS["benzodiazepine"])
    sleep = _contains_any(text, KEYWORDS["sleep"])
    dementia = _contains_any(text, KEYWORDS["dementia"])
    cardiometabolic = _contains_any(text, KEYWORDS["cardiometabolic"])
    pain_sedation = _contains_any(text, KEYWORDS["pain_sedation"])
    neurologic = _contains_any(text, KEYWORDS["neurologic"])
    psychotropic = antidepressant or antipsychotic or benzodiazepine or sleep
    sedative = benzodiazepine or sleep
    neuropsychiatric = psychotropic or dementia
    domains = [
        anticholinergic, psychotropic, sedative, dementia, n_meds >= 5,
        neuropsychiatric, cardiometabolic, pain_sedation, neurologic,
    ]
    domain_count = int(sum(bool(x) for x in domains))
    score = 0
    if n_meds >= 5:
        score += 2
    if n_meds >= 10:
        score += 1
    score += int(anticholinergic) + int(psychotropic) + int(sedative) + int(cardiometabolic) + int(neurologic)
    score = min(score, 5)
    if not text:
        summary = "No medication text available."
        confidence = 0.20
        manual_review = 1
    else:
        summary = "Medication list suggests " + (", ".join([
            label for label, flag in [
                ("anticholinergic exposure", anticholinergic),
                ("psychotropic treatment", psychotropic),
                ("sedative or pain-related treatment", sedative),
                ("cardiometabolic treatment", cardiometabolic),
                ("neurologic or cognitive treatment", neurologic or dementia),
                ("polypharmacy complexity", n_meds >= 5),
            ] if flag
        ]) or "limited class-specific medication burden") + "."
        confidence = 0.80 if domain_count else 0.65
        manual_review = 0
    return {
        "parse_ok": 1,
        "anticholinergic_exposure": int(anticholinergic),
        "psychotropic_exposure": int(psychotropic),
        "sedative_hypnotic_exposure": int(sedative),
        "cognitive_symptomatic_treatment": int(dementia),
        "polypharmacy_complexity_high": int(n_meds >= 5),
        "neuropsychiatric_treatment_signal": int(neuropsychiatric),
        "cardiometabolic_treatment_signal": int(cardiometabolic),
        "pain_sedation_signal": int(pain_sedation),
        "neurologic_treatment_signal": int(neurologic),
        "medication_state_domain_count": domain_count,
        "polypharmacy_complexity_score": score,
        "confidence": confidence,
        "manual_review": manual_review,
        "summary": summary,
        "rationale": "Local fallback abstraction based only on current-visit medication text.",
    }


def _balanced_json_candidates(raw: str) -> list[str]:
    """Return balanced top-level JSON-object substrings from a possibly noisy response."""
    candidates: list[str] = []
    in_string = False
    escape = False
    depth = 0
    start: int | None = None
    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start:i + 1])
                    start = None
    return candidates


def _extract_json_object(text: Any) -> dict[str, Any]:
    """Parse one medication-state JSON object from an Ollama response.

    A row is counted as true LLM-derived only after this function returns a
    dictionary and the required schema keys are present. The parser is tolerant of
    common Ollama/model wrappers, Markdown fences, and brief leading/trailing text,
    but it does not invent missing medication-state fields.
    """
    if isinstance(text, dict):
        # If a full Ollama /api/generate envelope was accidentally passed through,
        # parse the response field rather than treating the envelope as the answer.
        if "response" in text and not any(k in text for k in JSON_SCHEMA["required"]):
            return _extract_json_object(text.get("response", ""))
        return text
    raw = str(text or "").strip()
    raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json|JSON)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    for candidate in [raw, *_balanced_json_candidates(raw)]:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            if "response" in parsed and not any(k in parsed for k in JSON_SCHEMA["required"]):
                return _extract_json_object(parsed.get("response", ""))
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    preview = raw[:500].replace("\n", " ")
    raise ValueError(f"Could not parse a JSON object from LLM response. raw_preview={preview!r}")


def _feature_source(provider: str, status: str) -> str:
    provider_text = str(provider or "").lower()
    status_text = str(status or "").lower()
    if provider_text == "ollama" and "success" in status_text:
        return "ollama_json"
    if provider_text == "mock":
        return "mock_local_abstraction"
    if provider_text in {"local_clinical_abstraction", "local", "fallback"}:
        return "local_fallback_abstraction"
    if provider_text == "disabled":
        return "disabled"
    return provider_text or "unknown"


def _normalize_response(
    obj: dict[str, Any],
    *,
    provider: str,
    requested_provider: str,
    model: str,
    error: str = "",
    parse_error: str = "",
    raw_response: str = "",
    response_json: str = "",
    status: str = "success",
    attempted: int = 0,
    attempt_count: int = 0,
    fallback_reason: str = "",
    fallback_parse_ok: int = 0,
    elapsed_ms: float | int | None = None,
    llm_parse_ok_override: int | None = None,
    medication_text: str = "",
    canonical_medication_text: str = "",
) -> dict[str, Any]:
    if str(provider or "").lower() == "ollama":
        # Do not merge local fallback formula fields into true LLM JSON; v1.48
        # intentionally omits formula-derived fields from the prompt and computes
        # them below. Summary/rationale still default to blank for safety.
        src = {**(obj or {})}
        src.setdefault("summary", "")
        src.setdefault("rationale", "")
    else:
        defaults = local_clinical_abstraction("")
        src = {**defaults, **(obj or {})}
    validated = _validated_medication_state_features(src, medication_text=medication_text)
    provider_text = str(provider or "")
    status_text = str(status or "success")
    is_true_ollama_success = provider_text.lower() == "ollama" and "success" in status_text.lower()
    llm_parse_ok = int(is_true_ollama_success) if llm_parse_ok_override is None else int(llm_parse_ok_override)
    if not response_json and is_true_ollama_success:
        try:
            response_json = json.dumps({k: v for k, v in src.items() if not str(k).startswith("__")}, sort_keys=True)
        except Exception:
            response_json = ""
    row = {
        "effective_provider": provider_text,
        "canonical_medication_text": canonical_medication_text or medication_text,
        "requested_provider": requested_provider,
        "llm_model": model,
        "feature_source": _feature_source(provider_text, status_text),
        "llm_summary": str(src.get("summary", ""))[:500],
        "llm_rationale": str(src.get("rationale", ""))[:500],
        "llm_error": str(error or "")[:500],
        "llm_parse_error": str(parse_error or "")[:500],
        "llm_raw_response": _truncate(raw_response, 4000),
        "llm_response_json": _truncate(response_json, 4000),
        "llm_processing_policy_version": LLM_MEDICATION_STATE_POLICY_VERSION,
        "llm_status": status_text[:120],
        "llm_attempted": int(attempted or 0),
        "fallback_parse_ok": int(fallback_parse_ok or 0),
        "llm_attempt_count": int(attempt_count or attempted or 0),
        "llm_fallback_reason": str(fallback_reason or "")[:500],
        "llm_elapsed_ms": round(float(elapsed_ms or 0.0), 3),
        "llm_schema_violation_count": int(validated.get("__schema_violation_count", 0)),
        "llm_schema_violation_fields": str(validated.get("__schema_violation_fields", ""))[:500],
        "llm_binary_repair_applied": int(validated.get("__binary_repair_applied", 0)),
        "llm_domain_count_raw": validated.get("__domain_count_raw", ""),
        "llm_domain_count_recomputed": int(validated.get("__domain_count_recomputed", 0)),
        "llm_domain_count_repair_applied": int(validated.get("__domain_count_repair_applied", 0)),
        "llm_polypharmacy_repair_applied": int(validated.get("__polypharmacy_repair_applied", 0)),
        "llm_confidence_raw": validated.get("__confidence_raw", ""),
        "llm_confidence_repair_applied": int(validated.get("__confidence_repair_applied", 0)),
        "llm_quality_repair_applied": int(validated.get("__quality_repair_applied", 0)),
        "llm_clinical_anchor_repair_applied": int(validated.get("__clinical_anchor_repair_applied", 0)),
        "llm_clinical_anchor_repair_fields": str(validated.get("__clinical_anchor_repair_fields", ""))[:500],
        "llm_formula_fields_computed_in_python": int(validated.get("__formula_fields_computed_in_python", 0)),
        "llm_abstraction_unit": "drug_token" if _token_count(medication_text) <= 1 else "medication_text",
        "llm_drug_dictionary_aggregation_applied": 0,
        "llm_drug_dictionary_token_count": _token_count(medication_text),
        "llm_drug_dictionary_ollama_token_count": 0,
        "llm_drug_dictionary_fallback_token_count": 0,
        "llm_drug_dictionary_tokens": _truncate(normalize_medication_text(medication_text), 1000),
        "llm_parse_ok": llm_parse_ok,
        "llm_anticholinergic_exposure": int(validated.get("anticholinergic_exposure", 0)),
        "llm_psychotropic_exposure": int(validated.get("psychotropic_exposure", 0)),
        "llm_sedative_hypnotic_exposure": int(validated.get("sedative_hypnotic_exposure", 0)),
        "llm_cognitive_symptomatic_treatment": int(validated.get("cognitive_symptomatic_treatment", 0)),
        "llm_polypharmacy_complexity_high": int(validated.get("polypharmacy_complexity_high", 0)),
        "llm_neuropsychiatric_treatment_signal": int(validated.get("neuropsychiatric_treatment_signal", 0)),
        "llm_cardiometabolic_treatment_signal": int(validated.get("cardiometabolic_treatment_signal", 0)),
        "llm_pain_sedation_signal": int(validated.get("pain_sedation_signal", 0)),
        "llm_neurologic_treatment_signal": int(validated.get("neurologic_treatment_signal", 0)),
        "llm_medication_state_domain_count": int(validated.get("medication_state_domain_count", 0)),
        "llm_polypharmacy_complexity_score": int(validated.get("polypharmacy_complexity_score", 0)),
        "llm_confidence": max(0.0, min(1.0, _safe_float(validated.get("confidence", 0.0)))),
        "llm_manual_review": int(validated.get("manual_review", 0)),
    }
    return row


def _certification_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    return {
        "min_confidence": float(settings.get("llm_certification_min_confidence", 0.50) or 0.50),
        "require_no_manual_review": bool(settings.get("llm_certification_require_no_manual_review", True)),
        "require_ollama_provider": bool(settings.get("llm_certification_require_ollama_provider", True)),
        "enabled": bool(settings.get("llm_certification_enabled", True)),
    }


def _certify_normalized_row(row: dict[str, Any], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add structural certification fields to one normalized medication-state row.

    Certification is intentionally conservative. It does not prove clinical ground-truth
    correctness; it only certifies that a row is a real LLM-generated row with internally
    valid JSON, no parser/repair flags, adequate confidence, and no unresolved review flag.
    Rows that fail certification remain auditable and may still be used as fallback or
    review rows, but they should not be described as certified LLM medication abstraction.
    """
    cfg = _certification_settings(settings)
    reasons: list[str] = []
    provider = _audit_string(row.get("effective_provider", "")).lower()
    if cfg["require_ollama_provider"] and provider != "ollama":
        reasons.append("not_ollama_provider")
    if int(float(row.get("llm_parse_ok", 0) or 0)) != 1:
        reasons.append("llm_parse_not_ok")
    if int(float(row.get("fallback_parse_ok", 0) or 0)) != 0 and provider == "ollama":
        reasons.append("ollama_row_has_fallback_parse_ok")
    if _audit_string(row.get("llm_error", "")).strip():
        reasons.append("llm_error_present")
    if _audit_string(row.get("llm_parse_error", "")).strip():
        reasons.append("llm_parse_error_present")
    # Certification is based on the final post-repair row. Formula-derived repairs
    # such as domain-count and polypharmacy-score recomputation are expected with
    # small local LLMs and are fully audited in separate columns. They do not, by
    # themselves, invalidate primary-analysis use after the final fields are
    # internally coherent. Confidence repair remains disqualifying because it means
    # the model did not provide a usable certainty estimate.
    repair_cols_blocking = [
        "llm_confidence_repair_applied",
    ]
    repair_cols_audited_nonblocking = [
        "llm_schema_violation_count",
        "llm_binary_repair_applied",
        "llm_domain_count_repair_applied",
        "llm_polypharmacy_repair_applied",
        "llm_quality_repair_applied",
        "llm_clinical_anchor_repair_applied",
    ]
    repaired_nonblocking = False
    for col in repair_cols_blocking:
        try:
            if float(row.get(col, 0) or 0) > 0:
                reasons.append(f"{col}_gt_0")
        except Exception:
            reasons.append(f"{col}_invalid")
    for col in repair_cols_audited_nonblocking:
        try:
            if float(row.get(col, 0) or 0) > 0:
                repaired_nonblocking = True
        except Exception:
            reasons.append(f"{col}_invalid")
    try:
        conf = float(row.get("llm_confidence", 0) or 0)
    except Exception:
        conf = 0.0
    if conf < cfg["min_confidence"]:
        reasons.append(f"confidence_below_{cfg['min_confidence']}")
    if cfg["require_no_manual_review"] and int(float(row.get("llm_manual_review", 0) or 0)) != 0:
        reasons.append("manual_review_required")
    summary = _audit_string(row.get("llm_summary", "")).strip()
    rationale = _audit_string(row.get("llm_rationale", "")).strip()
    if provider == "ollama" and not summary:
        reasons.append("summary_empty")
    if provider == "ollama" and not rationale:
        reasons.append("rationale_empty")
    structural_pass = int(len(reasons) == 0)
    certified = int(cfg["enabled"] and structural_pass == 1)
    row["llm_structural_quality_pass"] = structural_pass
    row["llm_certified_for_primary_analysis"] = certified
    clinical_anchor_repaired = False
    try:
        clinical_anchor_repaired = float(row.get("llm_clinical_anchor_repair_applied", 0) or 0) > 0
    except Exception:
        clinical_anchor_repaired = False
    if certified and clinical_anchor_repaired:
        status = "certified_llm_clinically_anchored_after_audited_repair"
    elif certified and repaired_nonblocking:
        status = "certified_llm_structural_quality_after_audited_repair"
    elif certified:
        status = "certified_llm_structural_quality"
    else:
        status = "not_certified_requires_review_or_fallback"
    row["llm_certification_status"] = status
    row["llm_certification_reasons"] = ";".join(reasons)
    return row


def add_llm_certification_columns(audit: pd.DataFrame, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a copy of an audit table with conservative LLM certification columns."""
    audit = audit.copy() if audit is not None else pd.DataFrame()
    for col in LLM_CERTIFICATION_COLUMNS:
        if col not in audit.columns:
            audit[col] = "" if col.endswith("status") or col.endswith("reasons") else 0
    if audit.empty:
        return audit
    original_index = audit.index
    rows = []
    for rec in audit.to_dict(orient="records"):
        rows.append(_certify_normalized_row(rec, settings=settings))
    out = pd.DataFrame(rows)
    try:
        out.index = original_index
    except Exception:
        pass
    return out


def build_llm_medication_state_certification_audit(audit: pd.DataFrame, record_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a certification summary for conservative LLM-output claims.

    This file is intentionally phrased as certification of structural/audit quality, not
    clinical ground truth. Clinical accuracy still requires manual expert validation.
    """
    audit = add_llm_certification_columns(audit)
    total_unique = int(len(audit)) if audit is not None else 0
    if audit is None or audit.empty:
        return pd.DataFrame([{
            "certification_scope": "structural_llm_output_quality_not_clinical_ground_truth",
            "total_unique_medication_texts": 0,
            "unique_ollama_rows": 0,
            "unique_certified_llm_rows": 0,
            "unique_uncertified_llm_rows": 0,
            "unique_certified_llm_percent": 0.0,
            "visit_level_records": 0,
            "visit_level_certified_llm_records": 0,
            "visit_level_certified_llm_percent": 0.0,
            "certification_flag": "no_audit_rows",
            "interpretation": "No LLM output is available to certify.",
        }])
    provider = audit.get("effective_provider", pd.Series([""] * total_unique)).fillna("").astype(str).str.lower()
    certified = pd.to_numeric(audit.get("llm_certified_for_primary_analysis", pd.Series([0] * total_unique)), errors="coerce").fillna(0).astype(int)
    ollama_rows = int(provider.eq("ollama").sum())
    certified_rows = int(certified.eq(1).sum())
    uncertified_ollama = int(((provider.eq("ollama")) & certified.ne(1)).sum())
    reason_counts = {}
    if "llm_certification_reasons" in audit.columns:
        for text in audit["llm_certification_reasons"].fillna("").astype(str):
            for part in [x for x in text.split(";") if x.strip()]:
                reason_counts[part] = reason_counts.get(part, 0) + 1
    visit_records = 0
    visit_certified = 0
    if record_df is not None and not record_df.empty:
        visit_records = int(len(record_df))
        # map canonical medication text to certification; record_df uses the same mapping-derived provider columns
        if "llm_certified_for_primary_analysis" in record_df.columns:
            visit_certified = int(pd.to_numeric(record_df["llm_certified_for_primary_analysis"], errors="coerce").fillna(0).eq(1).sum())
        elif "llm_medication_state_provider" in record_df.columns:
            visit_certified = 0
    if certified_rows == 0:
        flag = "no_certified_llm_rows_do_not_claim_llm_accuracy"
    elif uncertified_ollama > 0:
        flag = "mixed_certified_and_uncertified_llm_rows_report_cautiously"
    else:
        flag = "all_observed_ollama_rows_pass_structural_certification_not_ground_truth"
    return pd.DataFrame([{
        "certification_scope": "structural_llm_output_quality_not_clinical_ground_truth",
        "total_unique_medication_texts": total_unique,
        "unique_ollama_rows": ollama_rows,
        "unique_certified_llm_rows": certified_rows,
        "unique_uncertified_llm_rows": uncertified_ollama,
        "unique_certified_llm_percent": round(100.0 * certified_rows / total_unique, 3) if total_unique else 0.0,
        "unique_certified_among_ollama_percent": round(100.0 * certified_rows / ollama_rows, 3) if ollama_rows else 0.0,
        "visit_level_records": visit_records,
        "visit_level_certified_llm_records": visit_certified,
        "visit_level_certified_llm_percent": round(100.0 * visit_certified / visit_records, 3) if visit_records else 0.0,
        "certification_flag": flag,
        "dominant_uncertified_reasons": json.dumps(dict(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]), sort_keys=True),
        "interpretation": "Certified rows passed strict structural LLM-output gates; this is not a claim of 100 percent clinical ground-truth accuracy without manual validation.",
    }])


@dataclass
class OllamaSettings:
    base_url: str
    model: str
    timeout: float
    max_attempts: int
    temperature: float
    num_predict: int
    max_prompt_chars: int
    keep_alive: str
    structured_json_schema_enabled: bool = False
    ollama_format_mode: str = "none"  # none | json | schema | auto
    request_backend: str = "auto"  # auto | curl | urllib
    num_ctx: int = 1024
    connect_timeout_s: float = 10.0
    strict_response_validation_enabled: bool = True


def _ollama_reachable(base_url: str, timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return int(response.status) < 500
    except Exception:
        return False


def _build_prompt(medication_text: str, max_prompt_chars: int) -> str:
    """Build a compact JSON-only medication-state prompt with clinical fields only.

    Formula-derived fields are intentionally excluded from the LLM response and
    recomputed in Python. This lowers token generation time without removing any
    clinical medication-domain judgment from the LLM task.
    """
    med = str(medication_text or "")[: int(max_prompt_chars)]
    template = {
        "parse_ok": 1,
        "anticholinergic_exposure": 0,
        "psychotropic_exposure": 0,
        "sedative_hypnotic_exposure": 0,
        "cognitive_symptomatic_treatment": 0,
        "neuropsychiatric_treatment_signal": 0,
        "cardiometabolic_treatment_signal": 0,
        "pain_sedation_signal": 0,
        "neurologic_treatment_signal": 0,
        "confidence": 0.80,
        "manual_review": 0,
        "summary": "brief medication-state summary",
        "rationale": "brief rationale naming the medication classes detected",
    }
    return (
        "You are extracting medication-state features for dementia-progression research. "
        "Return ONE complete valid JSON object only. Return exactly one valid JSON object and no Markdown. Use only 0 or 1 for binary fields. "
        "Code clinically evident medication domains from the medication list. "
        "Antidepressants, antipsychotics, benzodiazepines, and sleep agents imply psychotropic exposure. "
        "Benzodiazepines and Z-drugs imply sedative_hypnotic_exposure. "
        "Donepezil, galantamine, rivastigmine, or memantine imply cognitive_symptomatic_treatment. "
        "Statins, antihypertensives, antithrombotics, thyroid drugs, and diabetes drugs imply cardiometabolic_treatment_signal. "
        "Opioids and gabapentinoids imply pain_sedation_signal; Parkinson or antiseizure drugs imply neurologic_treatment_signal. "
        "Do not output medication counts, domain counts, polypharmacy flags, or complexity scores; those are computed after parsing. "
        f"Required JSON keys and example: {json.dumps(template, separators=(',', ':'))}. "
        f"Medication list: {med}"
    )


def _build_repair_prompt(medication_text: str, max_prompt_chars: int, previous_response: str, parse_error: str) -> str:
    med = str(medication_text or "")[: int(max_prompt_chars)]
    template = {
        "parse_ok": 1,
        "anticholinergic_exposure": 0,
        "psychotropic_exposure": 0,
        "sedative_hypnotic_exposure": 0,
        "cognitive_symptomatic_treatment": 0,
        "neuropsychiatric_treatment_signal": 0,
        "cardiometabolic_treatment_signal": 0,
        "pain_sedation_signal": 0,
        "neurologic_treatment_signal": 0,
        "confidence": 0.80,
        "manual_review": 0,
        "summary": "brief medication-state summary",
        "rationale": "brief rationale naming the medication classes detected",
    }
    return (
        "Repair the previous response into exactly one valid JSON object with the required keys. "
        "Use only 0 or 1 for binary fields. Do not include domain_count, medication_count, "
        "polypharmacy flag, or polypharmacy score. Preserve clinically evident positives: "
        "Cymbalta/duloxetine and other antidepressants are psychotropic; benzodiazepines/Z-drugs are sedative-hypnotic; "
        "donepezil/galantamine/rivastigmine/memantine are cognitive therapy; statin/thyroid/diabetes/BP drugs are cardiometabolic/endocrine; "
        "opioids/gabapentinoids are pain-sedation; Parkinson and antiseizure drugs are neurologic. "
        f"Required JSON example: {json.dumps(template, separators=(',', ':'))}. "
        f"Medication list: {med}. Parse error: {str(parse_error)[:500]}. Previous response: {str(previous_response)[:1000]}"
    )


def _ollama_format_candidates(settings: OllamaSettings) -> list[Any]:
    """Return Ollama output-format candidates.

    The default is deliberately no explicit Ollama `format` argument. Several
    local Qwen/Ollama/GPU configurations can answer ordinary generation requests
    but stall or become very slow when `format: "json"` or full JSON-schema
    grammar mode is requested. The prompt still requires a JSON-only response,
    and the downstream parser/schema validation still require a valid JSON
    object before `llm_parse_ok` can become 1.
    """
    mode = str(getattr(settings, "ollama_format_mode", "none") or "none").strip().lower()
    if mode in {"none", "plain", "no_format", "off", "false"}:
        return [None]
    if mode == "json":
        return ["json"]
    if mode in {"schema", "json_schema"}:
        return [JSON_SCHEMA, "json", None]
    if mode == "auto":
        if bool(settings.structured_json_schema_enabled):
            return [JSON_SCHEMA, "json", None]
        return [None, "json"]
    # Backward-compatible fallback for unexpected values. Prefer no explicit
    # format because it is least likely to stall local Ollama generation.
    return [None]


def _ollama_payload(prompt: str, settings: OllamaSettings, format_value: Any) -> dict[str, Any]:
    """Build a compact Ollama /api/generate payload with bounded context."""
    payload = {
        "model": settings.model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": str(settings.keep_alive),
        "options": {
            "temperature": float(settings.temperature),
            "num_predict": int(settings.num_predict),
            "num_ctx": int(settings.num_ctx),
        },
    }
    if format_value is not None:
        payload["format"] = format_value
    return payload


def _parse_ollama_generate_body(body_text: str) -> str:
    body = json.loads(body_text)
    if isinstance(body, dict) and "error" in body and body.get("error"):
        raise RuntimeError(str(body.get("error")))
    return str((body or {}).get("response", ""))


def _ollama_generate_once_urllib(prompt: str, settings: OllamaSettings, format_value: Any) -> str:
    data = json.dumps(_ollama_payload(prompt, settings, format_value)).encode("utf-8")
    url = settings.base_url.rstrip("/") + "/api/generate"
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=float(settings.timeout)) as response:
        return _parse_ollama_generate_body(response.read().decode("utf-8", errors="replace"))


def _ollama_generate_once_curl(prompt: str, settings: OllamaSettings, format_value: Any) -> str:
    """Call Ollama through curl with a hard wall-clock timeout.

    On some local Ollama/GPU configurations, Python urlopen can appear to wait
    indefinitely when the server accepts the socket but the generation request
    stalls. curl --max-time gives a process-level timeout, so the pipeline can
    log the failure, advance the LLM progress bar, and fail clearly under strict
    guardrails instead of remaining at 0%.
    """
    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError("curl executable was not found for Ollama request_backend=curl")
    url = settings.base_url.rstrip("/") + "/api/generate"
    payload = json.dumps(_ollama_payload(prompt, settings, format_value))
    cmd = [
        curl_path,
        "--silent",
        "--show-error",
        "--write-out", "\n__HTTP_STATUS__:%{http_code}",
        "--connect-timeout", str(max(1.0, float(settings.connect_timeout_s))),
        "--max-time", str(max(1.0, float(settings.timeout))),
        "--header", "Content-Type: application/json",
        "--data-binary", "@-",
        url,
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=payload,
            text=True,
            capture_output=True,
            timeout=float(settings.timeout) + 5.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"curl Ollama request exceeded hard timeout_s={settings.timeout}") from exc
    stdout = completed.stdout or ""
    http_status = ""
    body_text = stdout
    marker = "\n__HTTP_STATUS__:"
    if marker in stdout:
        body_text, http_status = stdout.rsplit(marker, 1)
        http_status = http_status.strip()
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        detail_parts = [f"curl exit code {completed.returncode}"]
        if http_status:
            detail_parts.append(f"http_status={http_status}")
        if stderr:
            detail_parts.append(f"stderr={stderr}")
        if body_text.strip():
            detail_parts.append(f"body={body_text.strip()[:500]}")
        raise RuntimeError("curl Ollama request failed: " + " | ".join(detail_parts))
    if http_status and http_status.isdigit() and int(http_status) >= 400:
        detail = body_text.strip() or f"HTTP {http_status}"
        raise RuntimeError(f"curl Ollama request failed: http_status={http_status} | body={detail[:500]}")
    return _parse_ollama_generate_body(body_text)


def _ollama_generate_once(prompt: str, settings: OllamaSettings, format_value: Any) -> str:
    backend = str(settings.request_backend or "auto").strip().lower()
    if backend == "auto":
        backend = "curl" if shutil.which("curl") else "urllib"
    if backend == "curl":
        return _ollama_generate_once_curl(prompt, settings, format_value)
    if backend == "urllib":
        return _ollama_generate_once_urllib(prompt, settings, format_value)
    raise ValueError(f"Unsupported Ollama request backend: {settings.request_backend!r}. Expected auto, curl, or urllib.")


def _ollama_base_url_host_port(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(str(base_url or ""))
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    return host, port


def _ollama_pids_on_port(port: int) -> list[int]:
    try:
        completed = subprocess.run(["ss", "-ltnp"], text=True, capture_output=True, check=False, timeout=5)
    except Exception:
        return []
    pids: list[int] = []
    for line in (completed.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[3].endswith(f":{port}"):
            continue
        for match in re.findall(r"pid=([0-9]+)", line):
            pid = int(match)
            if pid not in pids:
                pids.append(pid)
    return pids


def _pid_is_ollama(pid: int) -> bool:
    try:
        completed = subprocess.run(["ps", "-p", str(pid), "-o", "comm=", "-o", "args="], text=True, capture_output=True, check=False, timeout=5)
    except Exception:
        return False
    text = (completed.stdout or "").lower()
    return "ollama" in text


def _ollama_tags_reachable(base_url: str, timeout_s: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(str(base_url).rstrip("/") + "/api/tags", timeout=float(timeout_s)) as response:
            return int(getattr(response, "status", 500)) < 500
    except Exception:
        return False


def _restart_isolated_ollama_cpu_for_row_recovery(base_url: str, llm_log=None) -> bool:
    """Restart only the isolated project Ollama port with CUDA hidden.

    This is a row-level recovery path used when the launcher generation probe
    succeeds but the first medication rows fail repeatedly under strict LLM
    guardrails. It never manages port 11434, which may belong to another
    pipeline, and it refuses to kill any non-Ollama process.
    """
    try:
        host, port = _ollama_base_url_host_port(base_url)
    except Exception as exc:
        if llm_log:
            llm_log.exception("CPU fallback restart skipped because Ollama base URL could not be parsed | base_url=%s | error=%s", base_url, exc)
        return False
    if port == 11434:
        if llm_log:
            llm_log.error("CPU fallback restart refused for port 11434; this port may be used by another pipeline | base_url=%s", base_url)
        return False
    if port != 11435:
        if llm_log:
            llm_log.warning("CPU fallback restart allowed only for isolated project ports; requested port=%s | base_url=%s", port, base_url)
        return False

    pids = _ollama_pids_on_port(port)
    for pid in pids:
        if not _pid_is_ollama(pid):
            if llm_log:
                llm_log.error("CPU fallback restart refused: pid=%s on port=%s is not an Ollama process", pid, port)
            return False
    for pid in pids:
        if llm_log:
            llm_log.warning("CPU fallback restart: stopping only isolated Ollama pid=%s bound to %s:%s; preserving 11434", pid, host, port)
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except Exception as exc:
            if llm_log:
                llm_log.warning("CPU fallback restart: SIGTERM failed for pid=%s | error=%s", pid, exc)
    deadline = time.time() + 10.0
    while time.time() < deadline and _ollama_pids_on_port(port):
        time.sleep(0.5)
    for pid in _ollama_pids_on_port(port):
        if _pid_is_ollama(pid):
            if llm_log:
                llm_log.warning("CPU fallback restart: SIGKILL only isolated Ollama pid=%s still bound to port=%s", pid, port)
            try:
                os.kill(pid, 9)
            except Exception:
                pass
    if _ollama_pids_on_port(port):
        if llm_log:
            llm_log.error("CPU fallback restart failed: port %s remained occupied after stopping isolated Ollama", port)
        return False

    ollama_bin = os.environ.get("OLLAMA_BIN") or shutil.which("ollama") or "/home/ppanta/opt/ollama/bin/ollama"
    if not Path(ollama_bin).exists():
        if llm_log:
            llm_log.error("CPU fallback restart failed: Ollama binary not found | OLLAMA_BIN=%s", ollama_bin)
        return False
    env = os.environ.copy()
    env.update({
        "OLLAMA_HOST": f"{host}:{port}",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "GGML_VK_VISIBLE_DEVICES": "",
        "OLLAMA_NUM_PARALLEL": env.get("OLLAMA_NUM_PARALLEL", "1"),
        "OLLAMA_MAX_LOADED_MODELS": env.get("OLLAMA_MAX_LOADED_MODELS", "1"),
        "OLLAMA_CONTEXT_LENGTH": env.get("OLLAMA_CONTEXT_LENGTH", "1024"),
        "OLLAMA_KEEP_ALIVE": env.get("OLLAMA_KEEP_ALIVE", "30m"),
        "OLLAMA_DEBUG": env.get("OLLAMA_DEBUG", "INFO"),
    })
    log_path = Path(env.get("DEMENTIA_ROW_CPU_FALLBACK_LOG", "op/ollama_row_cpu_fallback.log"))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab")
    except Exception:
        log_fh = subprocess.DEVNULL
    try:
        proc = subprocess.Popen([ollama_bin, "serve"], env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as exc:
        if llm_log:
            llm_log.exception("CPU fallback restart failed to start Ollama | binary=%s | error=%s", ollama_bin, exc)
        try:
            if log_fh is not subprocess.DEVNULL:
                log_fh.close()
        except Exception:
            pass
        return False
    if llm_log:
        llm_log.warning("CPU fallback restart: started isolated CPU Ollama pid=%s on %s:%s; preserving 11434", proc.pid, host, port)
    for waited in range(0, 61):
        if _ollama_tags_reachable(base_url, timeout_s=2.0):
            if llm_log:
                llm_log.warning("CPU fallback restart succeeded after %ss | base_url=%s", waited, base_url)
            return True
        time.sleep(1.0)
    if llm_log:
        llm_log.error("CPU fallback restart did not become reachable within 60s | base_url=%s", base_url)
    return False


def _validate_llm_object(obj: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in JSON_SCHEMA["required"] if k not in obj]
    if missing:
        raise ValueError("LLM JSON missing required keys: " + ", ".join(missing))
    return obj


def _strict_validate_llm_object(obj: dict[str, Any], medication_text: str) -> dict[str, Any]:
    """Validate required LLM JSON shape without rejecting repairable content.

    Earlier v1.43 behavior rejected parseable Ollama JSON when formula-derived
    fields such as domain count, polypharmacy flag, score, confidence, or binary
    0/1 coding were internally inconsistent. In real Qwen2.5 3B output, those
    inconsistencies caused valid row-level LLM responses to be counted as
    failures and sent to local fallback.

    The corrected behavior is: parseable JSON with the required keys remains a
    true Ollama row; structural inconsistencies are repaired later by
    _validated_medication_state_features and fully audited in the repair columns.
    Only unparseable JSON or missing required keys should fail this stage.
    """
    obj = _validate_llm_object(obj)

    # Keep the raw LLM clinical-domain fields. _normalize_response subsequently
    # coerces binary fields, recomputes deterministic formula fields, clamps
    # confidence, and marks manual_review/audit repair flags as needed. This
    # prevents a scientifically misleading local-fallback substitution when the
    # model gave parseable but not perfectly formula-consistent JSON.
    for key in ["summary", "rationale"]:
        if obj.get(key) is None:
            obj[key] = ""
        else:
            obj[key] = str(obj.get(key, ""))
    return obj


def call_ollama_medication_state(medication_text: str, settings: OllamaSettings) -> dict[str, Any]:
    """Call Ollama and return a parsed medication-state JSON object.

    The returned dictionary includes internal audit keys prefixed with __ so the
    caller can save the raw response and parse metadata without exposing them as
    model features.
    """
    last_error: Exception | None = None
    last_raw = ""
    last_parse_error = ""
    attempt_no = 0
    prompt = _build_prompt(medication_text, settings.max_prompt_chars)
    max_attempts = max(1, int(settings.max_attempts))
    for attempt_no in range(1, max_attempts + 1):
        for fmt in _ollama_format_candidates(settings):
            try:
                raw = _ollama_generate_once(prompt, settings, fmt)
                last_raw = raw
                parsed = _extract_json_object(raw)
                if bool(getattr(settings, "strict_response_validation_enabled", True)):
                    parsed = _strict_validate_llm_object(parsed, medication_text)
                else:
                    parsed = _validate_llm_object(parsed)
                parsed["__raw_response"] = raw
                parsed["__response_json"] = json.dumps({k: parsed.get(k, "") for k in JSON_SCHEMA["required"]}, sort_keys=True)
                parsed["__attempt_count"] = attempt_no
                return parsed
            except Exception as exc:
                last_error = exc
                last_parse_error = str(exc)
                # Try the next configured Ollama output-format candidate for the
                # same attempt, then use a repair prompt if all candidates fail.
                continue
        prompt = _build_repair_prompt(medication_text, settings.max_prompt_chars, last_raw, last_parse_error)
        time.sleep(0.25)
    raise OllamaMedicationStateError(
        f"Ollama medication-state abstraction failed after {attempt_no} attempt(s): {last_error}",
        raw_response=last_raw,
        parse_error=last_parse_error,
        attempt_count=attempt_no,
    )


def warmup_ollama_medication_state(settings: OllamaSettings) -> bool:
    """Run a schema-compatible Ollama preflight request before patient-level abstraction.

    The warmup exercises the same parser and required medication-state schema used
    for real rows. Formula-derived fields are structurally repaired downstream, so
    preflight success means the model can return the required JSON shape, not that
    every clinical field has external ground-truth accuracy.
    """
    warmup_settings = OllamaSettings(
        base_url=settings.base_url,
        model=settings.model,
        timeout=max(float(settings.timeout), 120.0),
        max_attempts=max(1, int(settings.max_attempts)),
        temperature=0.0,
        num_predict=max(int(settings.num_predict), 160),
        max_prompt_chars=max(int(settings.max_prompt_chars), 200),
        keep_alive=settings.keep_alive,
        structured_json_schema_enabled=settings.structured_json_schema_enabled,
        ollama_format_mode=settings.ollama_format_mode,
        request_backend=settings.request_backend,
        num_ctx=settings.num_ctx,
        connect_timeout_s=settings.connect_timeout_s,
        strict_response_validation_enabled=settings.strict_response_validation_enabled,
    )
    obj = call_ollama_medication_state("donepezil | sertraline", warmup_settings)
    _validate_llm_object(obj)
    return True


def _selected_unique_texts(texts: pd.Series, max_unique: int | None, strategy: str) -> list[str]:
    clean = texts.fillna("").apply(normalize_medication_text).astype(str)
    # Blank, whitespace-only, and non-codable placeholders are never sent to the LLM.
    clean_nonempty = clean[clean.apply(_token_count).gt(0)]
    if max_unique is None or int(max_unique) <= 0:
        return list(dict.fromkeys(clean_nonempty.tolist()))
    max_unique = int(max_unique)
    if strategy == "most_frequent":
        counts = Counter(clean_nonempty.tolist())
        return [text for text, _ in counts.most_common(max_unique)]
    return list(dict.fromkeys(clean_nonempty.tolist()))[:max_unique]



def _resolve_cache_path(cache_path: str | None, project_root: str | Path | None = None) -> Path | None:
    """Resolve the optional persistent LLM medication-state cache path."""
    if not cache_path:
        return None
    path = Path(str(cache_path)).expanduser()
    if not path.is_absolute():
        base = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        path = base / path
    return path


def _load_persistent_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load previously computed LLM/fallback medication-state abstractions."""
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        cached = pd.read_csv(cache_path)
    except Exception:
        return {}
    if "medication_text" not in cached.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in cached.to_dict(orient="records"):
        text = _audit_string(row.get("medication_text", ""))
        if not text:
            continue
        cached_row = {k: row.get(k) for k in [*TEXT_AUDIT_COLUMNS[1:], *DRUG_DICTIONARY_IDENTITY_COLUMNS, *LLM_AUDIT_STATUS_COLUMNS, *LLM_FEATURE_COLUMNS] if k in row}
        provider = str(cached_row.get("effective_provider", "")).lower()
        old_error = _audit_string(cached_row.get("llm_error", ""))
        if provider != "ollama" and ("Ollama medication-state abstraction failed" in old_error or old_error == "ollama_not_used_or_text_not_selected"):
            cached_row["llm_error"] = ""
            cached_row.setdefault("llm_status", "cached_local_abstraction_cleaned")
            cached_row.setdefault("llm_fallback_reason", old_error[:500])
        out[text] = cached_row
        canonical_text = _audit_string(row.get("canonical_medication_text", ""))
        if canonical_text and canonical_text != text:
            out[canonical_text] = cached_row
    return out


def _atomic_write_dataframe_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Atomically write a CSV so interrupted runs do not leave a corrupt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _atomic_write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Atomically write a small JSON progress/checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _write_persistent_cache(cache_path: Path | None, audit: pd.DataFrame) -> None:
    """Write/update the persistent cache without changing pipeline CSV outputs."""
    if cache_path is None or audit.empty or "medication_text" not in audit.columns:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if cache_path.exists():
        try:
            existing = pd.read_csv(cache_path)
        except Exception:
            existing = pd.DataFrame()
    combined = pd.concat([existing, audit], ignore_index=True) if not existing.empty else audit.copy()
    combined = combined.drop_duplicates(subset=["medication_text"], keep="last")
    _atomic_write_dataframe_csv(combined, cache_path)


def build_llm_medication_state_quality_audit(audit: pd.DataFrame, record_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a compact QA table for Step 2 medication-state abstraction.

    The table separates unique-medication-text coverage from visit-level coverage and
    reports every repair applied after parsing. This makes fast budgeted LLM runs
    manuscript-safe because the output explicitly shows how much of the medication
    representation was true Ollama JSON vs transparent local fallback.
    """
    audit = audit.copy() if audit is not None else pd.DataFrame()
    if not audit.empty and "medication_text" in audit.columns:
        med_text = audit["medication_text"].apply(normalize_medication_text).fillna("")
    else:
        med_text = pd.Series(dtype=str)
    total_unique = int(len(audit))
    nonempty_unique = int(med_text.apply(_token_count).gt(0).sum()) if len(med_text) else 0
    empty_unique = int(total_unique - nonempty_unique)
    provider = audit.get("effective_provider", pd.Series([""] * total_unique)).fillna("").astype(str).str.lower() if total_unique else pd.Series(dtype=str)
    status = audit.get("llm_status", pd.Series([""] * total_unique)).fillna("").astype(str) if total_unique else pd.Series(dtype=str)

    def _num_col(name: str) -> pd.Series:
        if name not in audit.columns:
            return pd.Series([0] * total_unique, dtype=float)
        return pd.to_numeric(audit[name], errors="coerce").fillna(0)

    row: dict[str, Any] = {
        "total_unique_medication_texts": total_unique,
        "nonempty_unique_medication_texts": nonempty_unique,
        "empty_or_noninformative_unique_medication_texts": empty_unique,
        "unique_ollama_provider_rows": int(provider.eq("ollama").sum()) if total_unique else 0,
        "unique_local_or_fallback_rows": int((~provider.eq("ollama")).sum()) if total_unique else 0,
        "unique_llm_parse_ok_rows": int(_num_col("llm_parse_ok").eq(1).sum()) if total_unique else 0,
        "unique_fallback_parse_ok_rows": int(_num_col("fallback_parse_ok").eq(1).sum()) if total_unique else 0,
        "unique_schema_violation_rows": int(_num_col("llm_schema_violation_count").gt(0).sum()) if total_unique else 0,
        "unique_binary_repair_rows": int(_num_col("llm_binary_repair_applied").gt(0).sum()) if total_unique else 0,
        "unique_domain_count_repair_rows": int(_num_col("llm_domain_count_repair_applied").gt(0).sum()) if total_unique else 0,
        "unique_polypharmacy_repair_rows": int(_num_col("llm_polypharmacy_repair_applied").gt(0).sum()) if total_unique else 0,
        "unique_confidence_repair_rows": int(_num_col("llm_confidence_repair_applied").gt(0).sum()) if total_unique else 0,
        "unique_any_quality_repair_rows": int(_num_col("llm_quality_repair_applied").gt(0).sum()) if total_unique else 0,
        "unique_ollama_coverage_percent": round(100.0 * int(provider.eq("ollama").sum()) / total_unique, 3) if total_unique else 0.0,
        "unique_nonempty_ollama_coverage_percent": round(100.0 * int(provider.eq("ollama").sum()) / nonempty_unique, 3) if nonempty_unique else 0.0,
        "dominant_statuses": json.dumps(status.value_counts(dropna=False).head(20).to_dict(), sort_keys=True),
        "llm_input_normalization_backend_counts": json.dumps(audit.get("llm_input_normalization_backend", pd.Series([""] * total_unique)).fillna("").astype(str).value_counts(dropna=False).to_dict(), sort_keys=True) if total_unique else "{}",
        "llm_input_normalization_requested_backend_counts": json.dumps(audit.get("llm_input_normalization_requested_backend", pd.Series([""] * total_unique)).fillna("").astype(str).value_counts(dropna=False).to_dict(), sort_keys=True) if total_unique else "{}",
        "llm_input_encoder_models": json.dumps(sorted(set(audit.get("llm_input_encoder_model", pd.Series([""] * total_unique)).fillna("").astype(str).tolist())), sort_keys=True) if total_unique else "[]",
    }

    if record_df is not None and not record_df.empty:
        r = record_df.copy()
        r_provider = r.get("llm_medication_state_provider", pd.Series([""] * len(r))).fillna("").astype(str).str.lower()
        r_med_text = r.get("medication_text", pd.Series([""] * len(r))).apply(normalize_medication_text).fillna("")
        total_records = int(len(r))
        ollama_records = int(r_provider.eq("ollama").sum())
        row.update({
            "visit_level_records": total_records,
            "visit_level_ollama_provider_records": ollama_records,
            "visit_level_local_or_fallback_records": int(total_records - ollama_records),
            "visit_level_empty_or_noninformative_medication_records": int(r_med_text.apply(_token_count).eq(0).sum()),
            "visit_level_ollama_coverage_percent": round(100.0 * ollama_records / total_records, 3) if total_records else 0.0,
        })
    else:
        row.update({
            "visit_level_records": 0,
            "visit_level_ollama_provider_records": 0,
            "visit_level_local_or_fallback_records": 0,
            "visit_level_empty_or_noninformative_medication_records": 0,
            "visit_level_ollama_coverage_percent": 0.0,
        })

    row["manuscript_interpretation_flag"] = (
        "full_or_near_full_llm_coverage" if row["visit_level_ollama_coverage_percent"] >= 90 else
        "partial_llm_coverage_report_as_budgeted_llm_plus_structured_fallback" if row["unique_ollama_provider_rows"] > 0 else
        "fallback_only_do_not_describe_as_llm_derived"
    )
    return pd.DataFrame([row])



def build_llm_medication_model_comparison_audit(
    audit: pd.DataFrame,
    cfg: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally compare the primary Step 2 LLM model with candidate LLMs.

    This function is disabled by default because it can add substantial runtime and
    requires candidate Ollama models to be installed locally. When enabled, it samples
    medication texts from the primary audit table, obtains candidate-model JSON
    abstractions through the same schema/parser/repair path, and reports feature-level
    agreement. When disabled or when Ollama is unavailable, it still writes a compact
    not-run audit row so downstream documentation remains explicit.
    """
    settings = _merge_model_profile_settings((cfg.get("llm_medication_state", {}) or {}))
    enabled = bool(settings.get("model_comparison_enabled", False))
    primary_model = str(settings.get("ollama_model", ""))
    candidates = settings.get("model_comparison_candidates", []) or []
    if isinstance(candidates, str):
        candidates = [x.strip() for x in candidates.split(",") if x.strip()]
    sample_size = max(0, int(settings.get("model_comparison_sample_size", 0) or 0))
    base_url = str(settings.get("ollama_base_url", "http://127.0.0.1:11435"))
    provider = str(settings.get("provider", "auto")).lower().strip()
    audit = audit.copy() if audit is not None else pd.DataFrame()
    base_columns = [
        "comparison_status", "medication_text", "primary_model", "candidate_model",
        "primary_effective_provider", "candidate_effective_provider", "candidate_error",
        "domain_binary_agreement", "domain_binary_disagreement_count",
        "primary_domain_count", "candidate_domain_count", "candidate_confidence",
        "candidate_manual_review", "primary_features_json", "candidate_features_json",
    ]
    if not enabled:
        rows = [{
            "comparison_status": "not_run_disabled",
            "medication_text": "", "primary_model": primary_model, "candidate_model": "",
            "primary_effective_provider": "", "candidate_effective_provider": "", "candidate_error": "",
            "domain_binary_agreement": "", "domain_binary_disagreement_count": "",
            "primary_domain_count": "", "candidate_domain_count": "", "candidate_confidence": "",
            "candidate_manual_review": "", "primary_features_json": "{}", "candidate_features_json": "{}",
        }]
        comp = pd.DataFrame(rows, columns=base_columns)
        summ = pd.DataFrame([{
            "model_comparison_enabled": False,
            "comparison_status": "not_run_disabled",
            "primary_model": primary_model,
            "candidate_models": json.dumps(candidates),
            "sample_size_requested": sample_size,
            "comparison_rows": 0,
            "mean_domain_binary_agreement": "",
            "mean_domain_binary_disagreement_count": "",
        }])
        return comp, summ
    if audit.empty or sample_size <= 0 or not candidates:
        status = "not_run_no_audit_or_candidates"
        comp = pd.DataFrame([{col: "" for col in base_columns}])
        comp.loc[0, "comparison_status"] = status
        comp.loc[0, "primary_model"] = primary_model
        summ = pd.DataFrame([{"model_comparison_enabled": True, "comparison_status": status, "primary_model": primary_model, "candidate_models": json.dumps(candidates), "sample_size_requested": sample_size, "comparison_rows": 0}])
        return comp, summ
    reachable = _ollama_reachable(base_url, timeout=float(settings.get("reachability_timeout_s", 2.0)))
    if provider not in {"auto", "ollama"} or not reachable:
        status = "not_run_ollama_unavailable_or_provider_not_ollama"
        comp = pd.DataFrame([{col: "" for col in base_columns}])
        comp.loc[0, "comparison_status"] = status
        comp.loc[0, "primary_model"] = primary_model
        comp.loc[0, "candidate_model"] = ",".join(candidates)
        summ = pd.DataFrame([{"model_comparison_enabled": True, "comparison_status": status, "primary_model": primary_model, "candidate_models": json.dumps(candidates), "sample_size_requested": sample_size, "comparison_rows": 0}])
        return comp, summ

    provider_col = audit.get("effective_provider", pd.Series([""] * len(audit))).fillna("").astype(str).str.lower()
    sample = audit.loc[provider_col.eq("ollama")].copy()
    if sample.empty:
        sample = audit.copy()
    sample = sample.head(sample_size)
    rows: list[dict[str, Any]] = []
    for candidate_model in candidates:
        candidate_settings = OllamaSettings(
            base_url=base_url,
            model=str(candidate_model),
            timeout=float(settings.get("model_comparison_request_timeout_s", settings.get("request_timeout_s", 180.0))),
            max_attempts=int(settings.get("model_comparison_max_attempts", 1) or 1),
            temperature=float(settings.get("temperature", 0.0)),
            num_predict=int(settings.get("model_comparison_num_predict", settings.get("num_predict", 192))),
            max_prompt_chars=int(settings.get("max_prompt_chars", 500)),
            keep_alive=str(settings.get("keep_alive", "10m")),
            structured_json_schema_enabled=bool(settings.get("structured_json_schema_enabled", False)),
            ollama_format_mode=str(settings.get("ollama_format_mode", "json")),
            request_backend=str(settings.get("request_backend", "auto")),
            num_ctx=int(settings.get("num_ctx", 1024)),
            connect_timeout_s=float(settings.get("connect_timeout_s", 10.0)),
        )
        for _, primary in sample.iterrows():
            med_text = _audit_string(primary.get("canonical_medication_text", primary.get("medication_text", "")))
            primary_vec = _comparison_feature_vector(primary)
            try:
                obj = call_ollama_medication_state(med_text, candidate_settings)
                cand_norm = _normalize_response(
                    obj, provider="ollama", requested_provider="model_comparison", model=str(candidate_model),
                    status="ollama_success", attempted=1, attempt_count=int(obj.get("__attempt_count", 1) or 1),
                    raw_response=str(obj.get("__raw_response", "")), response_json=str(obj.get("__response_json", "")),
                    medication_text=med_text,
                )
                cand_vec = _comparison_feature_vector(cand_norm)
                disagreements = sum(int(primary_vec[k] != cand_vec[k]) for k in DOMAIN_BINARY_KEYS)
                agreement = round(1.0 - disagreements / max(1, len(DOMAIN_BINARY_KEYS)), 4)
                status = "compared"
                error = ""
            except Exception as exc:
                cand_norm = _normalize_response(
                    local_clinical_abstraction(med_text), provider="comparison_failed", requested_provider="model_comparison",
                    model=str(candidate_model), status="candidate_model_failed", attempted=1,
                    fallback_parse_ok=0, llm_parse_ok_override=0, error=str(exc)[:500], medication_text=med_text,
                )
                cand_vec = _comparison_feature_vector(cand_norm)
                disagreements = ""
                agreement = ""
                status = "candidate_model_failed"
                error = str(exc)[:500]
            rows.append({
                "comparison_status": status,
                "medication_text": med_text,
                "primary_model": primary_model,
                "candidate_model": str(candidate_model),
                "primary_effective_provider": _audit_string(primary.get("effective_provider", "")),
                "candidate_effective_provider": _audit_string(cand_norm.get("effective_provider", "")),
                "candidate_error": error,
                "domain_binary_agreement": agreement,
                "domain_binary_disagreement_count": disagreements,
                "primary_domain_count": int(float(primary.get("llm_medication_state_domain_count", 0) or 0)),
                "candidate_domain_count": int(float(cand_norm.get("llm_medication_state_domain_count", 0) or 0)),
                "candidate_confidence": cand_norm.get("llm_confidence", ""),
                "candidate_manual_review": cand_norm.get("llm_manual_review", ""),
                "primary_features_json": json.dumps(primary_vec, sort_keys=True),
                "candidate_features_json": json.dumps(cand_vec, sort_keys=True),
            })
    comp = pd.DataFrame(rows, columns=base_columns)
    valid = pd.to_numeric(comp.get("domain_binary_agreement", pd.Series(dtype=float)), errors="coerce")
    dis = pd.to_numeric(comp.get("domain_binary_disagreement_count", pd.Series(dtype=float)), errors="coerce")
    summ = pd.DataFrame([{
        "model_comparison_enabled": True,
        "comparison_status": "completed" if valid.notna().any() else "completed_with_no_successful_candidate_rows",
        "primary_model": primary_model,
        "candidate_models": json.dumps(candidates),
        "sample_size_requested": sample_size,
        "comparison_rows": int(len(comp)),
        "successful_comparison_rows": int(valid.notna().sum()),
        "failed_comparison_rows": int(comp["comparison_status"].ne("compared").sum()),
        "mean_domain_binary_agreement": round(float(valid.mean()), 4) if valid.notna().any() else "",
        "mean_domain_binary_disagreement_count": round(float(dis.mean()), 4) if dis.notna().any() else "",
    }])
    return comp, summ


def apply_llm_medication_state(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    logger=None,
    llm_logger=None,
    partial_audit_path: str | Path | None = None,
    partial_progress_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create LLM-enhanced medication-state features and an abstraction audit table."""
    out = df.copy()
    settings = _merge_model_profile_settings(cfg.get("llm_medication_state", {}) or {})
    llm_log = llm_logger if llm_logger is not None else logger
    enabled = bool(settings.get("enabled", False))
    if not enabled or str(settings.get("provider", "off")).lower() in {"off", "false", "none", "disabled"}:
        for col in LLM_FEATURE_COLUMNS:
            out[col] = 0.0
        # v2.7: Do not emit a blank medication_text sentinel row into S2E.
        # Empty/null/whitespace medication text is not an abstraction unit and
        # therefore should not be present in the downstream LLM audit table.
        audit = pd.DataFrame(columns=[*TEXT_AUDIT_COLUMNS, *LLM_AUDIT_STATUS_COLUMNS, *LLM_FEATURE_COLUMNS])
        return out, audit

    project_root = cfg.get("project_root", None)
    raw_med_texts = out.get("medication_text", pd.Series([""] * len(out), index=out.index)).apply(normalize_medication_text).fillna("").astype(str)
    canonicalization_enabled = bool(settings.get("llm_input_canonicalization_enabled", True))
    sort_tokens = bool(settings.get("llm_input_sort_unique_tokens", True))
    alias_normalization_enabled = bool(settings.get("llm_input_alias_normalization_enabled", True))
    alias_resource_path = settings.get("llm_input_alias_resource_path", "resources/medication/medication_aliases.csv")
    if project_root and alias_resource_path and not Path(str(alias_resource_path)).is_absolute():
        alias_resource_path = Path(str(project_root)) / str(alias_resource_path)
    alias_map = load_medication_alias_map(alias_resource_path) if alias_normalization_enabled else {}

    pretrained_requested_backend = str(settings.get("llm_input_pretrained_normalization_backend", "alias_only") or "alias_only").strip().lower()
    pretrained_enabled = bool(settings.get("llm_input_pretrained_normalization_enabled", pretrained_requested_backend not in {"", "none", "off", "false", "alias_only"}))
    pretrained_token_normalizer = None
    pretrained_meta = {
        "requested_backend": pretrained_requested_backend,
        "actual_backend": "alias_only",
        "status": "disabled",
        "model": "",
        "vocab_size": 0,
        "similarity_threshold": float(settings.get("llm_input_pretrained_similarity_threshold", 0.86) or 0.86),
    }
    if pretrained_enabled:
        vocab_path = settings.get("llm_input_pretrained_vocabulary_path", "resources/medication/medication_reference_vocabulary.csv")
        if project_root and vocab_path and not Path(str(vocab_path)).is_absolute():
            vocab_path = Path(str(project_root)) / str(vocab_path)
        pretrained_token_normalizer, pretrained_meta = build_frozen_encoder_medication_normalizer(
            requested_backend=pretrained_requested_backend,
            sapbert_model=settings.get("llm_input_sapbert_model", None),
            bioclinicalbert_model=settings.get("llm_input_bioclinicalbert_model", None),
            vocabulary_path=vocab_path,
            alias_map=alias_map,
            similarity_threshold=float(settings.get("llm_input_pretrained_similarity_threshold", 0.86) or 0.86),
            batch_size=int(settings.get("llm_input_pretrained_batch_size", 64) or 64),
            device=str(settings.get("llm_input_pretrained_device", "auto") or "auto"),
            local_files_only=bool(settings.get("llm_input_pretrained_local_files_only", False)),
            required=bool(settings.get("llm_input_pretrained_normalization_required", False)),
            logger=llm_log,
        )
    if canonicalization_enabled:
        med_texts = raw_med_texts.apply(
            lambda x: canonicalize_medication_text_for_llm(
                x,
                sort_tokens=sort_tokens,
                normalize_aliases=alias_normalization_enabled,
                alias_map=alias_map,
                pretrained_token_normalizer=pretrained_token_normalizer,
            )
        ).fillna("").astype(str)
    else:
        med_texts = raw_med_texts

    # v2.11: In drug-token abstraction mode, dictionary identity must be based on the
    # original cleaned raw medication tokens, not on any alias/pretrained canonicalization.
    # This prevents unsafe encoder/canonicalization substitutions from changing the join key
    # used to merge LLM abstractions back to visit-level rows. Canonicalized text is retained
    # only as an audit field, while raw-key text drives dictionary keys and aggregation.
    raw_key_med_texts = raw_med_texts.apply(normalize_medication_text).fillna("").astype(str)

    out["medication_text"] = raw_med_texts
    out["llm_canonical_medication_text"] = med_texts
    out["llm_drug_dictionary_source_medication_text"] = raw_key_med_texts
    out["llm_input_normalization_backend"] = str(pretrained_meta.get("actual_backend", "alias_only"))
    out["llm_input_normalization_requested_backend"] = str(pretrained_meta.get("requested_backend", pretrained_requested_backend))
    out["llm_input_encoder_model"] = str(pretrained_meta.get("model", ""))
    out["llm_input_encoder_status"] = str(pretrained_meta.get("status", ""))
    out["llm_input_encoder_similarity_threshold"] = float(pretrained_meta.get("similarity_threshold", 0.0) or 0.0)
    out["llm_input_encoder_vocab_size"] = int(pretrained_meta.get("vocab_size", 0) or 0)
    requested_provider = str(settings.get("provider", "auto")).lower().strip()
    fallback_provider = str(settings.get("fallback_provider", "local_clinical_abstraction")).lower().strip()
    model = str(settings.get("ollama_model", ""))
    base_url = str(settings.get("ollama_base_url", "http://127.0.0.1:11434"))
    timeout = float(settings.get("request_timeout_s", 15.0))
    fail_on_error = bool(settings.get("fail_on_error", False))
    progress_enabled = bool((cfg.get("progress", {}) or {}).get("enabled", True))
    abstraction_unit = str(settings.get("abstraction_unit", "medication_text") or "medication_text").strip().lower()
    drug_token_mode = abstraction_unit in {"drug", "drug_token", "unique_drug", "unique_drug_name", "drug_name"}
    dictionary_source_series = raw_key_med_texts if drug_token_mode else med_texts
    token_lists_by_record = dictionary_source_series.apply(_split_medication_tokens)
    token_counts = Counter(t for tokens in token_lists_by_record.tolist() for t in tokens if t)

    persistent_cache_enabled = bool(settings.get("persistent_cache_enabled", True))
    if drug_token_mode:
        cache_setting = settings.get("drug_dictionary_persistent_cache_path", "op/llm_medication_drug_dictionary_cache.csv")
    else:
        cache_setting = settings.get("persistent_cache_path", "op/llm_medication_state_cache.csv")
    persistent_cache_path = _resolve_cache_path(cache_setting, project_root) if persistent_cache_enabled else None
    cached_mapping = _load_persistent_cache(persistent_cache_path)
    if drug_token_mode and bool(settings.get("drug_dictionary_cache_require_stable_key", True)):
        stable_cached_mapping: dict[str, dict[str, Any]] = {}
        for _cache_text, _cache_row in cached_mapping.items():
            _expected_key = _drug_dictionary_key(_cache_text)
            _row_key = _audit_string(_cache_row.get("llm_dictionary_key", ""))
            _row_norm = normalize_medication_text(_cache_row.get("normalized_drug_name", _cache_text))
            _expected_norm = normalize_medication_text(_cache_text)
            if _expected_key and _row_key == _expected_key and _row_norm == _expected_norm:
                stable_cached_mapping[_cache_text] = _cache_row
        if llm_log and len(stable_cached_mapping) != len(cached_mapping):
            llm_log.warning(
                "Ignored unstable legacy LLM drug-dictionary cache rows without matching llm_dictionary_key: kept=%s ignored=%s path=%s",
                len(stable_cached_mapping), len(cached_mapping) - len(stable_cached_mapping), persistent_cache_path,
            )
        cached_mapping = stable_cached_mapping
    cache_lookup_aliases: dict[str, list[str]] = {}
    if not drug_token_mode:
        try:
            for raw_value, canonical_value in zip(raw_med_texts.tolist(), med_texts.tolist()):
                raw_key = _audit_string(raw_value)
                canonical_key = _audit_string(canonical_value)
                if canonical_key and raw_key and raw_key != canonical_key:
                    cache_lookup_aliases.setdefault(canonical_key, []).append(raw_key)
        except Exception:
            cache_lookup_aliases = {}

    def _cached_row_for_processing_text(text: str) -> dict[str, Any] | None:
        cached = cached_mapping.get(text)
        if cached is not None:
            return cached
        for alt_key in cache_lookup_aliases.get(text, []):
            cached = cached_mapping.get(alt_key)
            if cached is not None:
                return cached
        return None

    max_unique_raw = settings.get("max_unique_texts", None)
    if max_unique_raw is None:
        max_unique = None
    else:
        max_unique_text = str(max_unique_raw).strip().lower()
        if max_unique_text in {"", "none", "null", "all", "full", "unlimited", "no_limit"}:
            max_unique = None
        else:
            try:
                max_unique = int(float(max_unique_text))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "llm_medication_state.max_unique_texts must be an integer or one of "
                    "['all', 'none', 'null', 'full', 'unlimited']; "
                    f"received {max_unique_raw!r}"
                ) from exc
            if max_unique < 0:
                raise ValueError(
                    "llm_medication_state.max_unique_texts must be non-negative, null, or 'all'; "
                    f"received {max_unique_raw!r}"
                )
    selection_strategy = str(settings.get("selection_strategy", "most_frequent"))
    if drug_token_mode:
        processing_series = pd.Series([t for t, _c in token_counts.most_common()], dtype=str)
    else:
        processing_series = med_texts
    if selection_strategy.lower().strip() in {"record_coverage", "coverage", "visit_coverage"}:
        target = max(0.0, min(1.0, float(settings.get("record_coverage_target", 0.80) or 0.80)))
        if drug_token_mode:
            counts = Counter(token_counts)
        else:
            counts = Counter([x for x in med_texts.tolist() if _token_count(x) > 0])
        selected_list: list[str] = []
        running = 0
        denominator = max(1, sum(counts.values()))
        for text, count in counts.most_common():
            selected_list.append(text)
            running += int(count)
            if (running / denominator) >= target:
                break
            if max_unique is not None and len(selected_list) >= max_unique:
                break
        selected = set(selected_list)
    else:
        selected = set(_selected_unique_texts(processing_series, max_unique, selection_strategy))
    ollama_settings = OllamaSettings(
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_attempts=int(settings.get("max_attempts", 1)),
        temperature=float(settings.get("temperature", 0.0)),
        num_predict=int(settings.get("num_predict", 96)),
        max_prompt_chars=int(settings.get("max_prompt_chars", 500)),
        keep_alive=str(settings.get("keep_alive", "10m")),
        structured_json_schema_enabled=bool(settings.get("structured_json_schema_enabled", False)),
        ollama_format_mode=str(settings.get("ollama_format_mode", "schema" if bool(settings.get("structured_json_schema_enabled", False)) else "none")),
        request_backend=str(settings.get("request_backend", "auto")),
        num_ctx=int(settings.get("num_ctx", 1024)),
        connect_timeout_s=float(settings.get("connect_timeout_s", 10.0)),
        strict_response_validation_enabled=bool(settings.get("strict_response_validation_enabled", True)),
    )
    require_ollama_when_enabled = bool(settings.get("require_ollama_when_enabled", False))
    require_min_ollama_successes = int(settings.get("require_min_ollama_successes", 0) or 0)
    reachable = _ollama_reachable(base_url, timeout=float(settings.get("reachability_timeout_s", 0.5)))
    use_ollama = requested_provider == "ollama" or (requested_provider == "auto" and reachable)
    warmup_enabled = bool(settings.get("warmup_enabled", True))
    if llm_log:
        llm_log.info(
            "LLM medication-state abstraction initialized | requested_provider=%s | fallback_provider=%s | ollama_reachable=%s | use_ollama=%s | warmup_enabled=%s | structured_json_schema_enabled=%s | require_ollama_when_enabled=%s | require_min_ollama_successes=%s | model=%s | selected_unique_texts=%s | total_unique_texts=%s | raw_unique_medication_texts=%s | canonicalization_enabled=%s | alias_normalization_enabled=%s | alias_count=%s | cache_rows=%s | request_timeout_s=%s | connect_timeout_s=%s | max_attempts=%s | num_predict=%s | num_ctx=%s | max_prompt_chars=%s | request_backend=%s | ollama_format_mode=%s | persistent_cache_path=%s | input_normalization_requested_backend=%s | input_normalization_backend=%s | input_encoder_model=%s | input_encoder_status=%s | input_encoder_vocab_size=%s",
            requested_provider, fallback_provider, reachable, use_ollama, warmup_enabled, ollama_settings.structured_json_schema_enabled, require_ollama_when_enabled, require_min_ollama_successes, model, len(selected), dictionary_source_series.nunique(), raw_med_texts.nunique(), canonicalization_enabled, alias_normalization_enabled, len(alias_map), len(cached_mapping), timeout, ollama_settings.connect_timeout_s, ollama_settings.max_attempts, ollama_settings.num_predict, ollama_settings.num_ctx, ollama_settings.max_prompt_chars, ollama_settings.request_backend, ollama_settings.ollama_format_mode, persistent_cache_path, pretrained_meta.get("requested_backend", ""), pretrained_meta.get("actual_backend", ""), pretrained_meta.get("model", ""), pretrained_meta.get("status", ""), pretrained_meta.get("vocab_size", 0)
        )
    if logger:
        logger.info(
            "LLM medication-state abstraction: enabled provider=%s ollama_reachable=%s use_ollama=%s require_ollama_when_enabled=%s selected_unique_texts=%s cache_rows=%s",
            requested_provider, reachable, use_ollama, require_ollama_when_enabled, len(selected), len(cached_mapping)
        )
    warmup_error = ""
    if use_ollama and warmup_enabled:
        start = perf_counter()
        if llm_log:
            llm_log.info("Ollama medication-state JSON preflight starting | base_url=%s | model=%s | timeout_s=%s | structured_json_schema_enabled=%s | ollama_format_mode=%s", base_url, model, timeout, ollama_settings.structured_json_schema_enabled, ollama_settings.ollama_format_mode)
        try:
            warmup_ollama_medication_state(ollama_settings)
            if llm_log:
                llm_log.info("Ollama medication-state JSON preflight succeeded | elapsed_ms=%.3f", (perf_counter() - start) * 1000.0)
        except Exception as exc:
            warmup_error = str(exc)
            if llm_log:
                llm_log.exception(
                    "Ollama medication-state JSON preflight failed | base_url=%s | model=%s | timeout_s=%s | error=%s",
                    base_url, model, timeout, exc,
                )
            if fail_on_error:
                raise
            if require_ollama_when_enabled and requested_provider in {"auto", "ollama"}:
                msg = (
                    f"LLM medication-state abstraction requires a usable Ollama JSON response, but the JSON preflight failed at {base_url} "
                    f"for model {model}. Error: {warmup_error}. Review logs/llm.log. "
                    "Common fixes: confirm the model name with `ollama list`, run `ollama run <model> \"hello\"`, "
                    "increase LLM_REQUEST_TIMEOUT_S, or set LLM_WARMUP_ENABLED=0 to bypass only this preflight and let row-level calls be audited."
                )
                if llm_log:
                    llm_log.error(msg)
                raise RuntimeError(msg)
            use_ollama = False
            if llm_log:
                llm_log.warning("Ollama preflight failed; using fallback/local abstraction because strict Ollama requirement is disabled | error=%s", exc)
    elif use_ollama and not warmup_enabled:
        if llm_log:
            llm_log.info("Ollama JSON preflight skipped by configuration; row-level Ollama calls will start immediately and strict success guardrails remain active.")

    if require_ollama_when_enabled and requested_provider in {"auto", "ollama"} and not use_ollama:
        msg = (
            f"LLM medication-state abstraction requires Ollama, but Ollama was not usable at {base_url}. "
            f"Preflight_error={warmup_error or '<none>'}. "
            "Check OLLAMA_BASE_URL/DEMENTIA_OLLAMA_MODE, start Ollama, confirm `ollama list`, "
            "or set LLM_REQUIRE_OLLAMA=0 only for a transparent fallback-only test run."
        )
        if llm_log:
            llm_log.error(msg)
        raise RuntimeError(msg)

    mapping: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    # v2.7: Only true nonempty medication text/tokens are downstream LLM
    # abstraction units. Blank, whitespace-only, null, and noninformative values
    # are kept in the visit-level analysis dataset as no-medication-text records,
    # but they are not processed into S2E and are not written as blank audit rows.
    if drug_token_mode:
        unique_order = [t for t, _c in token_counts.most_common() if _token_count(t) > 0]
    else:
        unique_order = [t for t in dict.fromkeys(dictionary_source_series.tolist()) if _token_count(t) > 0]
    audit_columns = [*TEXT_AUDIT_COLUMNS, *LLM_AUDIT_STATUS_COLUMNS, *LLM_CERTIFICATION_COLUMNS, *LLM_FEATURE_COLUMNS]
    progress_log_interval = max(1, int(settings.get("progress_log_interval", 50) or 50))
    selected_nonempty_count = sum(1 for text in selected if _token_count(text) > 0)
    initial_failure_abort_count = int(settings.get("initial_failure_abort_count", 3) or 0)
    row_level_cpu_fallback_enabled = bool(settings.get("row_level_cpu_fallback_enabled", True))
    row_level_cpu_fallback_attempted = False
    consecutive_initial_failures = 0
    max_concurrent_requests = max(1, int(settings.get("max_concurrent_requests", 1) or 1))
    cache_write_interval = max(0, int(settings.get("cache_write_interval", 0) or 0))
    completed_since_cache_write = 0
    # v2.13: avoid repeatedly rewriting already flushed rows during long Step 2
    # LLM loops. Every row is still processed and cached; checkpoint writes only
    # newly completed rows plus one final forced flush.
    cache_flushed_texts: set[str] = set()

    partial_s2_write_enabled = bool(settings.get("partial_s2_write_enabled", True))
    partial_s2_write_percent_interval = max(0.1, float(settings.get("partial_s2_write_percent_interval", 1) or 1))
    partial_s2_write_late_start_percent = max(0.0, min(100.0, float(settings.get("partial_s2_write_late_start_percent", 95) or 95)))
    partial_s2_write_late_percent_interval = max(0.1, float(settings.get("partial_s2_write_late_percent_interval", 0.5) or 0.5))
    partial_s2_write_min_seconds = max(0.0, float(settings.get("partial_s2_write_min_seconds", 120) or 0))
    partial_audit_path_resolved = Path(partial_audit_path) if partial_audit_path else None
    partial_progress_path_resolved = Path(partial_progress_path) if partial_progress_path else None
    partial_quality_audit_path_resolved = None
    partial_certification_audit_path_resolved = None
    if partial_audit_path_resolved is not None:
        partial_quality_audit_path_resolved = partial_audit_path_resolved.with_name("s2h_llm_medication_state_quality_audit_partial.csv")
        partial_certification_audit_path_resolved = partial_audit_path_resolved.with_name("s2k_llm_certification_audit_partial.csv")
    last_partial_write_percent = -1.0
    last_partial_write_time = 0.0
    last_partial_write_rows = 0

    normalization_audit_fields = {
        "llm_input_normalization_requested_backend": str(pretrained_meta.get("requested_backend", pretrained_requested_backend)),
        "llm_input_normalization_backend": str(pretrained_meta.get("actual_backend", "alias_only")),
        "llm_input_encoder_model": str(pretrained_meta.get("model", "")),
        "llm_input_encoder_status": str(pretrained_meta.get("status", "")),
        "llm_input_encoder_similarity_threshold": float(pretrained_meta.get("similarity_threshold", 0.0) or 0.0),
        "llm_input_encoder_vocab_size": int(pretrained_meta.get("vocab_size", 0) or 0),
    }

    def _attach_normalization_audit(norm: dict[str, Any]) -> dict[str, Any]:
        norm.update(normalization_audit_fields)
        return norm

    def _attach_drug_dictionary_identity(text: str, norm: dict[str, Any]) -> dict[str, Any]:
        identity = _drug_dictionary_identity(
            text,
            mapping_provider=str(pretrained_meta.get("actual_backend", "alias_only")),
            mapping_similarity_score=1.0,
        )
        norm.update(identity)
        norm["llm_abstraction_status"] = _audit_string(norm.get("llm_status", ""))
        return norm

    def _norm_from_cache(cached: dict[str, Any], cached_provider: str) -> dict[str, Any]:
        cached_error = _audit_string(cached.get("llm_error", ""))
        if cached_provider != "ollama" and (
            "Ollama medication-state abstraction failed" in cached_error
            or cached_error == "ollama_not_used_or_text_not_selected"
        ):
            cached_error = ""
        norm = {
            "effective_provider": _audit_string(cached.get("effective_provider", "cached"), "cached"),
            "canonical_medication_text": _audit_string(cached.get("canonical_medication_text", cached.get("medication_text", ""))),
            "requested_provider": _audit_string(cached.get("requested_provider", requested_provider), requested_provider),
            "llm_model": _audit_string(cached.get("llm_model", model), model),
            "feature_source": _audit_string(cached.get("feature_source", _feature_source(cached_provider, cached.get("llm_status", "cached_previous_abstraction")))),
            "llm_summary": _audit_string(cached.get("llm_summary", ""))[:500],
            "llm_rationale": _audit_string(cached.get("llm_rationale", ""))[:500],
            "llm_error": cached_error[:500],
            "llm_parse_error": _audit_string(cached.get("llm_parse_error", ""))[:500],
            "llm_raw_response": _truncate(cached.get("llm_raw_response", ""), 4000),
            "llm_response_json": _truncate(cached.get("llm_response_json", ""), 4000),
            "llm_processing_policy_version": LLM_MEDICATION_STATE_POLICY_VERSION,
            "llm_status": _audit_string(cached.get("llm_status", "cached_previous_abstraction"), "cached_previous_abstraction")[:120],
            "llm_attempted": int(float(cached.get("llm_attempted", 0) or 0)),
            "fallback_parse_ok": int(float(cached.get("fallback_parse_ok", 1 if cached_provider != "ollama" else 0) or 0)),
            "llm_attempt_count": int(float(cached.get("llm_attempt_count", cached.get("llm_attempted", 0)) or 0)),
            "llm_fallback_reason": _audit_string(cached.get("llm_fallback_reason", ""))[:500],
            "llm_elapsed_ms": _safe_float(cached.get("llm_elapsed_ms", 0.0)),
        }
        for col in LLM_AUDIT_STATUS_COLUMNS:
            if col not in norm:
                default = "" if col.endswith("fields") or col.endswith("raw") or col.endswith("version") or col.endswith("status") or col.endswith("reason") else 0
                if col == "llm_processing_policy_version":
                    norm[col] = LLM_MEDICATION_STATE_POLICY_VERSION
                else:
                    norm[col] = cached.get(col, default)
        for col in LLM_FEATURE_COLUMNS:
            norm[col] = cached.get(col, 0.0)
        # Upgrade legacy cache rows: only true Ollama JSON success may have llm_parse_ok=1.
        if cached_provider != "ollama":
            norm["llm_parse_ok"] = 0
            if norm["fallback_parse_ok"] == 0:
                norm["fallback_parse_ok"] = 1
        norm = _attach_normalization_audit(norm)
        norm = add_llm_certification_columns(pd.DataFrame([norm]), settings).iloc[0].to_dict()
        return norm

    def _recover_legacy_parseable_ollama_cache(cached: dict[str, Any] | None, text: str) -> dict[str, Any] | None:
        """Recover parseable LLM JSON saved by older strict-validation fallback cache rows.

        v1.43 could save rows as local fallback even though the raw Ollama response
        contained valid JSON with all required medication-state keys. The failure was
        caused by formula inconsistencies such as domain counts or polypharmacy scores,
        which are now repaired and audited. Without this recovery step, a resumed run
        can continue displaying stale ``ollama_failed_fallback_succeeded`` rows from an
        old cache/partial file even after the parser has been fixed.
        """
        if not cached:
            return None
        raw = _audit_string(cached.get("llm_raw_response", ""))
        if not raw.strip():
            return None
        cached_provider = _audit_string(cached.get("effective_provider", "")).lower()
        cached_status = _audit_string(cached.get("llm_status", "")).lower()
        cached_parse_error = _audit_string(cached.get("llm_parse_error", "")).lower()
        looks_like_legacy_failed_ollama = (
            cached_provider != "ollama"
            and (
                "ollama_failed_fallback" in cached_status
                or "llm json failed strict quality validation" in cached_parse_error
                or "ollama medication-state abstraction failed" in _audit_string(cached.get("llm_fallback_reason", "")).lower()
            )
        )
        if not looks_like_legacy_failed_ollama:
            return None
        try:
            parsed = _extract_json_object(raw)
            parsed = _strict_validate_llm_object(parsed, text)
            response_json = json.dumps({k: parsed.get(k, "") for k in JSON_SCHEMA["required"]}, sort_keys=True)
            recovered_model = _audit_string(cached.get("llm_model", model), model)
            norm = _normalize_response(
                parsed,
                provider="ollama",
                requested_provider=_audit_string(cached.get("requested_provider", requested_provider), requested_provider),
                model=recovered_model,
                status="ollama_success_recovered_from_legacy_cache",
                attempted=int(float(cached.get("llm_attempted", 1) or 1)),
                attempt_count=int(float(cached.get("llm_attempt_count", cached.get("llm_attempted", 1)) or 1)),
                raw_response=raw,
                response_json=response_json,
                fallback_parse_ok=0,
                elapsed_ms=_safe_float(cached.get("llm_elapsed_ms", 0.0), 0.0),
                medication_text=text,
                canonical_medication_text=_audit_string(cached.get("canonical_medication_text", text), text),
            )
            norm["llm_fallback_reason"] = "recovered_parseable_ollama_json_from_legacy_strict_validation_cache"
            norm = _attach_normalization_audit(norm)
            norm = add_llm_certification_columns(pd.DataFrame([norm]), settings).iloc[0].to_dict()
            if llm_log:
                llm_log.info("Recovered parseable Ollama JSON from legacy fallback cache | medication_text_preview=%s", str(text)[:180])
            return norm
        except Exception as exc:
            if llm_log:
                llm_log.warning("Could not recover legacy fallback cache row; row will be reprocessed | medication_text_preview=%s | error=%s", str(text)[:180], exc)
            return None

    strict_cache_validation = bool(settings.get("strict_cache_validation_enabled", True))

    def _cached_ollama_row_passes_quality(cached: dict[str, Any]) -> bool:
        if not strict_cache_validation:
            return True
        cached_model = _audit_string(cached.get("llm_model", ""))
        if cached_model and cached_model != model:
            return False
        expected_norm_backend = _audit_string(normalization_audit_fields.get("llm_input_normalization_backend", ""))
        cached_norm_backend = _audit_string(cached.get("llm_input_normalization_backend", ""))
        if expected_norm_backend and cached_norm_backend and cached_norm_backend != expected_norm_backend:
            return False
        if expected_norm_backend not in {"", "alias_only", "auto_fallback_alias_only"} and not cached_norm_backend:
            return False
        if int(float(cached.get("llm_parse_ok", 0) or 0)) != 1:
            return False
        if _audit_string(cached.get("llm_error", "")) or _audit_string(cached.get("llm_parse_error", "")):
            return False
        # Repaired true-Ollama rows remain cache-compatible. Repair flags are
        # quality/audit indicators, not reasons to discard a parseable LLM row.
        # Certification columns still distinguish fully clean rows from rows that
        # required structural repair or manual review.
        return True

    def _cache_compatible_for_current_run(cached_provider: str, should_try_llm: bool, cached: dict[str, Any] | None = None) -> bool:
        cached_provider = str(cached_provider or "").lower()
        cached = cached or {}
        if should_try_llm:
            return bool(cached_provider == "ollama" and _cached_ollama_row_passes_quality(cached))
        if requested_provider == "mock":
            return bool(cached_provider == "mock")
        if requested_provider == "local_clinical_abstraction":
            return bool(cached_provider in {"local_clinical_abstraction", "local", "fallback"})
        # For auto/ollama when Ollama is unavailable or a row is not selected, rebuild
        # the transparent fallback status for this run instead of reusing possibly stale
        # mock/local rows from a previous development run.
        return False

    def _local_norm_for_text(text: str) -> tuple[dict[str, Any], str, str]:
        obj = local_clinical_abstraction(text)
        provider = "mock" if requested_provider == "mock" else fallback_provider
        if _token_count(text) <= 0:
            status = "empty_medication_text_local_fallback"
            reason = "empty_or_noninformative_medication_text"
        elif requested_provider == "mock":
            status = "mock_abstraction"
            reason = ""
        elif use_ollama and text not in selected:
            status = "local_abstraction_not_selected_by_max_unique_texts"
            reason = f"not_selected_by_max_unique_texts_{max_unique}"
        elif not use_ollama:
            status = "ollama_unavailable_local_fallback"
            reason = f"ollama_unavailable_at_{base_url}"
        else:
            status = "local_abstraction_not_selected_for_ollama"
            reason = "not_selected_for_ollama_unknown_reason"
        norm = _normalize_response(
            obj,
            provider=provider,
            requested_provider=requested_provider,
            model=model,
            error="",
            status=status,
            attempted=0,
            fallback_reason=reason,
            fallback_parse_ok=1,
            elapsed_ms=0.0,
            llm_parse_ok_override=0,
            medication_text=text,
        )
        return norm, status, status

    def _ollama_norm_or_fallback(text: str) -> tuple[dict[str, Any], str, bool, str | None]:
        start = perf_counter()
        try:
            obj = call_ollama_medication_state(text, ollama_settings)
            elapsed = (perf_counter() - start) * 1000.0
            norm = _normalize_response(
                obj,
                provider="ollama",
                requested_provider=requested_provider,
                model=model,
                status="ollama_success",
                attempted=1,
                attempt_count=int(obj.get("__attempt_count", 1) or 1),
                raw_response=str(obj.get("__raw_response", "")),
                response_json=str(obj.get("__response_json", "")),
                fallback_parse_ok=0,
                elapsed_ms=elapsed,
                medication_text=text,
            )
            if llm_log:
                llm_log.info("Ollama success | elapsed_ms=%.3f | medication_text_preview=%s", elapsed, str(text)[:180])
            return norm, "ollama_success", True, None
        except Exception as exc:
            elapsed = (perf_counter() - start) * 1000.0
            raw_preview = _truncate(getattr(exc, "raw_response", ""), 800)
            parse_preview = _truncate(getattr(exc, "parse_error", str(exc)), 800)
            if llm_log:
                llm_log.exception(
                    "Ollama failed; local fallback pending | elapsed_ms=%.3f | medication_text_preview=%s | parse_error=%s | raw_response_preview=%s",
                    elapsed, str(text)[:180], parse_preview, raw_preview,
                )
            if fail_on_error:
                raise
            obj = local_clinical_abstraction(text)
            norm = _normalize_response(
                obj,
                provider=fallback_provider,
                requested_provider=requested_provider,
                model=model,
                error="",
                parse_error=getattr(exc, "parse_error", ""),
                raw_response=getattr(exc, "raw_response", ""),
                status="ollama_failed_fallback_succeeded",
                attempted=1,
                attempt_count=getattr(exc, "attempt_count", 1),
                fallback_reason=str(exc),
                fallback_parse_ok=1,
                elapsed_ms=elapsed,
                llm_parse_ok_override=0,
                medication_text=text,
            )
            return norm, "ollama_failed_fallback", False, str(exc)

    def _drug_mapping_by_stable_key() -> dict[str, dict[str, Any]]:
        out_by_key: dict[str, dict[str, Any]] = {}
        for _token_text, _row in mapping.items():
            _key = _audit_string(_row.get("llm_dictionary_key", "")) or _drug_dictionary_key(_token_text)
            if _key:
                out_by_key[_key] = _row
        return out_by_key

    def _aggregate_drug_dictionary_rows(med_text: str, *, default_norm: dict[str, Any] | None = None) -> dict[str, Any]:
        """Aggregate drug-token dictionary abstractions back to one visit/list-level row using stable drug-name keys."""
        tokens = _split_medication_tokens(med_text)
        if not tokens:
            base = dict(default_norm or _normalize_response(
                local_clinical_abstraction(""),
                provider=fallback_provider,
                requested_provider=requested_provider,
                model=model,
                error="",
                status="empty_or_unmapped_medication_text",
                attempted=0,
                fallback_reason="empty_or_unmapped_medication_text",
                fallback_parse_ok=1,
                llm_parse_ok_override=0,
                medication_text="",
            ))
            base.update({
                "canonical_medication_text": "",
                "llm_abstraction_unit": "drug_dictionary_visit_aggregation",
                "llm_drug_dictionary_aggregation_applied": 1,
                "llm_drug_dictionary_token_count": 0,
                "llm_drug_dictionary_ollama_token_count": 0,
                "llm_drug_dictionary_fallback_token_count": 0,
                "llm_drug_dictionary_tokens": "",
            })
            return base
        mapping_by_key = _drug_mapping_by_stable_key()
        token_rows = []
        missing: list[str] = []
        for t in tokens:
            row = mapping_by_key.get(_drug_dictionary_key(t)) or mapping.get(t)
            if row is None:
                missing.append(t)
            else:
                token_rows.append(row)
        if missing:
            for t in missing:
                norm, _status, _label = _local_norm_for_text(t)
                mapping[t] = _attach_drug_dictionary_identity(t, _attach_normalization_audit(norm))
            mapping_by_key = _drug_mapping_by_stable_key()
            token_rows = [mapping_by_key.get(_drug_dictionary_key(t)) or mapping.get(t) for t in tokens]
            token_rows = [r for r in token_rows if r is not None]
        provider_values = [_audit_string(r.get("effective_provider", "")).lower() for r in token_rows]
        ollama_token_count = sum(1 for p in provider_values if p == "ollama")
        fallback_token_count = len(token_rows) - ollama_token_count
        if len(token_rows) > 0 and ollama_token_count == len(token_rows):
            provider = "ollama"
            status = "ollama_drug_dictionary_aggregated_success"
            fallback_parse_ok = 0
            llm_parse_ok_override = 1
            attempted = 1
        elif ollama_token_count > 0:
            provider = "mixed_drug_dictionary"
            status = "mixed_drug_dictionary_aggregated_with_fallback_tokens"
            fallback_parse_ok = 1
            llm_parse_ok_override = 0
            attempted = 1
        else:
            provider = fallback_provider if requested_provider != "mock" else "mock"
            status = "local_drug_dictionary_aggregated"
            fallback_parse_ok = 1
            llm_parse_ok_override = 0
            attempted = 0
        agg: dict[str, Any] = {
            "parse_ok": int(ollama_token_count == len(token_rows) and len(token_rows) > 0),
            "anticholinergic_exposure": max(int(float(r.get("llm_anticholinergic_exposure", 0) or 0)) for r in token_rows),
            "psychotropic_exposure": max(int(float(r.get("llm_psychotropic_exposure", 0) or 0)) for r in token_rows),
            "sedative_hypnotic_exposure": max(int(float(r.get("llm_sedative_hypnotic_exposure", 0) or 0)) for r in token_rows),
            "cognitive_symptomatic_treatment": max(int(float(r.get("llm_cognitive_symptomatic_treatment", 0) or 0)) for r in token_rows),
            "neuropsychiatric_treatment_signal": max(int(float(r.get("llm_neuropsychiatric_treatment_signal", 0) or 0)) for r in token_rows),
            "cardiometabolic_treatment_signal": max(int(float(r.get("llm_cardiometabolic_treatment_signal", 0) or 0)) for r in token_rows),
            "pain_sedation_signal": max(int(float(r.get("llm_pain_sedation_signal", 0) or 0)) for r in token_rows),
            "neurologic_treatment_signal": max(int(float(r.get("llm_neurologic_treatment_signal", 0) or 0)) for r in token_rows),
            "confidence": min(max(0.0, min(1.0, _safe_float(r.get("llm_confidence", 0.0), 0.0))) for r in token_rows),
            "manual_review": max(int(float(r.get("llm_manual_review", 0) or 0)) for r in token_rows),
            "summary": "Visit medication-state features aggregated from stable-keyed unique-drug dictionary entries.",
            "rationale": "Each unique drug name was abstracted once using llm_dictionary_key=sha256(normalized_drug_name), then visit-level domain indicators were formed by union across matched medication-token keys.",
        }
        raw_json_payload = {
            "abstraction_unit": "drug_token_dictionary",
            "tokens": tokens,
            "token_count": len(tokens),
            "ollama_token_count": ollama_token_count,
            "fallback_token_count": fallback_token_count,
            "aggregated_features": agg,
        }
        norm = _normalize_response(
            agg,
            provider=provider,
            requested_provider=requested_provider,
            model=model,
            status=status,
            attempted=attempted,
            attempt_count=sum(int(float(r.get("llm_attempt_count", 0) or 0)) for r in token_rows),
            raw_response=json.dumps(raw_json_payload, sort_keys=True),
            response_json=json.dumps(raw_json_payload, sort_keys=True),
            fallback_parse_ok=fallback_parse_ok,
            elapsed_ms=sum(float(r.get("llm_elapsed_ms", 0.0) or 0.0) for r in token_rows),
            llm_parse_ok_override=llm_parse_ok_override,
            medication_text=med_text,
            canonical_medication_text=med_text,
        )
        norm.update({
            "llm_abstraction_unit": "drug_dictionary_visit_aggregation",
            "llm_drug_dictionary_aggregation_applied": 1,
            "llm_drug_dictionary_token_count": int(len(tokens)),
            "llm_drug_dictionary_ollama_token_count": int(ollama_token_count),
            "llm_drug_dictionary_fallback_token_count": int(fallback_token_count),
            "llm_drug_dictionary_tokens": _truncate(" | ".join(tokens), 1000),
        })
        return norm

    def _write_cache_checkpoint(force: bool = False) -> None:
        nonlocal completed_since_cache_write, cache_flushed_texts
        if persistent_cache_path is None:
            return
        if cache_write_interval <= 0 and not force:
            return
        if not force and completed_since_cache_write < cache_write_interval:
            return
        pending_texts = [text for text in unique_order if text in mapping and text not in cache_flushed_texts]
        if not pending_texts:
            completed_since_cache_write = 0
            return
        rows = [{"medication_text": text, **mapping[text]} for text in pending_texts]
        checkpoint_columns = [*DRUG_DICTIONARY_IDENTITY_COLUMNS, *audit_columns] if drug_token_mode else audit_columns
        checkpoint_columns = [c for c in checkpoint_columns if c != "medication_text"]
        checkpoint = pd.DataFrame(rows, columns=["medication_text", *checkpoint_columns])
        checkpoint = add_llm_certification_columns(checkpoint, settings)
        _write_persistent_cache(persistent_cache_path, checkpoint)
        cache_flushed_texts.update(pending_texts)
        if llm_log:
            llm_log.info(
                "LLM medication-state persistent cache checkpoint written | new_rows=%s | total_flushed_rows=%s | path=%s",
                len(rows), len(cache_flushed_texts), persistent_cache_path,
            )
        completed_since_cache_write = 0

    def _partial_write_due(force: bool = False) -> bool:
        nonlocal last_partial_write_percent, last_partial_write_time, last_partial_write_rows
        if force:
            return True
        if not partial_s2_write_enabled or partial_audit_path_resolved is None:
            return False
        total = max(1, len(unique_order))
        processed = len(mapping)
        if processed <= 0:
            return False
        percent = 100.0 * processed / total
        interval = partial_s2_write_late_percent_interval if percent >= partial_s2_write_late_start_percent else partial_s2_write_percent_interval
        next_threshold = 0.0 if last_partial_write_percent < 0 else last_partial_write_percent + interval
        percent_due = percent >= next_threshold or (processed == total and last_partial_write_rows < total)
        time_due = False
        if partial_s2_write_min_seconds > 0 and processed > last_partial_write_rows:
            time_due = (time.time() - last_partial_write_time) >= partial_s2_write_min_seconds
        return bool(percent_due or time_due)

    def _write_partial_s2_checkpoint(force: bool = False, latest_status: str = "") -> None:
        nonlocal last_partial_write_percent, last_partial_write_time, last_partial_write_rows
        if not _partial_write_due(force=force):
            return
        rows = [{"medication_text": text, **mapping[text]} for text in unique_order if text in mapping]
        if not rows:
            return
        partial_audit = pd.DataFrame(rows, columns=audit_columns)
        partial_audit = add_llm_certification_columns(partial_audit, settings)
        processed = len(partial_audit)
        total = max(1, len(unique_order))
        percent = round(100.0 * processed / total, 4)

        processed_texts = set(partial_audit["medication_text"].fillna("").astype(str)) if "medication_text" in partial_audit.columns else set()
        processed_record_df = pd.DataFrame()
        if processed_texts:
            if drug_token_mode:
                record_mask = token_lists_by_record.apply(lambda toks: any(t in processed_texts for t in toks))
            else:
                record_mask = dictionary_source_series.fillna("").astype(str).isin(processed_texts)
            if record_mask.any():
                processed_record_df = out.loc[record_mask].copy()
                processed_record_texts = dictionary_source_series.loc[record_mask].fillna("").astype(str).tolist()
                if drug_token_mode:
                    processed_mapped_records = [_aggregate_drug_dictionary_rows(text) for text in processed_record_texts]
                else:
                    processed_mapped_records = [mapping.get(text, {}) for text in processed_record_texts]
                processed_feature_df = pd.DataFrame(processed_mapped_records, index=processed_record_df.index)
                processed_feature_df = add_llm_certification_columns(processed_feature_df, settings)
                processed_record_df["llm_medication_state_provider"] = processed_feature_df.get("effective_provider", pd.Series([""] * len(processed_record_df), index=processed_record_df.index)).fillna("").astype(str)
                processed_record_df["llm_certified_for_primary_analysis"] = pd.to_numeric(
                    processed_feature_df.get("llm_certified_for_primary_analysis", pd.Series([0] * len(processed_record_df), index=processed_record_df.index)),
                    errors="coerce",
                ).fillna(0).astype(int)

        partial_quality_audit = build_llm_medication_state_quality_audit(partial_audit, processed_record_df)
        partial_certification_audit = build_llm_medication_state_certification_audit(partial_audit, processed_record_df)

        if partial_audit_path_resolved is not None:
            _atomic_write_dataframe_csv(partial_audit, partial_audit_path_resolved)
        if partial_quality_audit_path_resolved is not None:
            _atomic_write_dataframe_csv(partial_quality_audit, partial_quality_audit_path_resolved)
        if partial_certification_audit_path_resolved is not None:
            _atomic_write_dataframe_csv(partial_certification_audit, partial_certification_audit_path_resolved)
        progress_payload = {
            "status": "completed" if processed >= len(unique_order) else "running",
            "processed_unique_texts": int(processed),
            "total_unique_texts": int(len(unique_order)),
            "processed_visit_records_with_completed_unique_text": int(len(processed_record_df)),
            "total_visit_records": int(len(out)),
            "progress_percent": percent,
            "latest_status": str(latest_status or ""),
            "ollama_success_rows": int(stats.get("ollama_success", 0)),
            "ollama_failure_fallback_rows": int(stats.get("ollama_failure_fallback_success", 0)),
            "cache_hit_rows": int(stats.get("cache_hit", 0)),
            "legacy_cache_recovered_rows": int(stats.get("legacy_cache_recovered", 0)),
            "llm_processing_policy_version": LLM_MEDICATION_STATE_POLICY_VERSION,
            "partial_audit_file": str(partial_audit_path_resolved) if partial_audit_path_resolved is not None else "",
            "partial_quality_audit_file": str(partial_quality_audit_path_resolved) if partial_quality_audit_path_resolved is not None else "",
            "partial_certification_audit_file": str(partial_certification_audit_path_resolved) if partial_certification_audit_path_resolved is not None else "",
            "persistent_cache_file": str(persistent_cache_path) if persistent_cache_path is not None else "",
            "write_rule": {
                "percent_interval_before_late_start": partial_s2_write_percent_interval,
                "late_start_percent": partial_s2_write_late_start_percent,
                "percent_interval_after_late_start": partial_s2_write_late_percent_interval,
                "min_seconds_between_writes": partial_s2_write_min_seconds,
            },
            "updated_unix_time": round(time.time(), 3),
        }
        if partial_progress_path_resolved is not None:
            _atomic_write_json(progress_payload, partial_progress_path_resolved)
        last_partial_write_percent = percent
        last_partial_write_time = time.time()
        last_partial_write_rows = processed
        if llm_log:
            llm_log.info(
                "Partial Step 2 LLM checkpoints written | processed=%s/%s | progress_percent=%.3f | audit_path=%s | quality_path=%s | certification_path=%s | progress_path=%s",
                processed, len(unique_order), percent, partial_audit_path_resolved, partial_quality_audit_path_resolved, partial_certification_audit_path_resolved, partial_progress_path_resolved,
            )

    if llm_log:
        llm_log.info(
            "LLM medication-state progress initialized | unique_texts=%s | selected_nonempty_for_ollama=%s | progress_bar_enabled=%s | progress_log_interval=%s | initial_failure_abort_count=%s | row_level_cpu_fallback_enabled=%s | max_concurrent_requests=%s | cache_write_interval=%s | partial_s2_write_enabled=%s | partial_s2_write_percent_interval=%s | partial_s2_write_late_start_percent=%s | partial_s2_write_late_percent_interval=%s | partial_s2_write_min_seconds=%s | partial_audit_path=%s | partial_quality_path=%s | partial_certification_path=%s",
            len(unique_order), selected_nonempty_count, progress_enabled, progress_log_interval, initial_failure_abort_count, row_level_cpu_fallback_enabled, max_concurrent_requests, cache_write_interval, partial_s2_write_enabled, partial_s2_write_percent_interval, partial_s2_write_late_start_percent, partial_s2_write_late_percent_interval, partial_s2_write_min_seconds, partial_audit_path_resolved, partial_quality_audit_path_resolved, partial_certification_audit_path_resolved,
        )

    llm_pending_texts: list[str] = []
    local_pending_texts: list[str] = []
    with progress_bar(
        total=len(unique_order),
        desc="Step 2 LLM medication-state abstraction",
        enabled=progress_enabled,
        unit="unique_text",
        logger=None,
    ) as llm_bar:
        # First pass: queue true Ollama rows before materializing local fallback rows.
        # This makes live partial audit files scientifically interpretable during a run:
        # completed LLM-selected rows appear early instead of a long fallback-only prefix.
        for text in unique_order:
            cached = _cached_row_for_processing_text(text)
            cached_provider = str((cached or {}).get("effective_provider", "")).lower()
            should_try_llm = (text in selected) and use_ollama and (_token_count(text) > 0)
            cache_compatible = bool(cached) and _cache_compatible_for_current_run(cached_provider, should_try_llm, cached=cached)
            recovered_legacy_norm = _recover_legacy_parseable_ollama_cache(cached, text) if should_try_llm and bool(cached) else None
            if should_try_llm and recovered_legacy_norm is not None:
                progress_label = "cache_recovered:ollama_json"
                mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(recovered_legacy_norm))
                stats["cache_hit"] += 1
                stats["legacy_cache_recovered"] += 1
                completed_since_cache_write += 1
                llm_bar.update(progress_label)
                _write_cache_checkpoint(force=False)
                _write_partial_s2_checkpoint(force=False, latest_status=progress_label)
            elif should_try_llm and cache_compatible:
                progress_label = f"cache_hit:{cached_provider or 'unknown'}"
                norm = _norm_from_cache(cached, cached_provider)
                mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(norm))
                stats["cache_hit"] += 1
                completed_since_cache_write += 1
                llm_bar.update(progress_label)
                _write_cache_checkpoint(force=False)
                _write_partial_s2_checkpoint(force=False, latest_status=progress_label)
            elif should_try_llm:
                llm_pending_texts.append(text)
            else:
                local_pending_texts.append(text)

        # Strict early validation remains sequential. This catches a broken Ollama server
        # before launching many parallel requests and preserves the row-level CPU recovery.
        early_validation_count = 0
        while llm_pending_texts and use_ollama and require_ollama_when_enabled and stats.get("ollama_success", 0) == 0:
            text = llm_pending_texts.pop(0)
            early_validation_count += 1
            if llm_log:
                llm_log.info(
                    "Ollama early validation row request starting | index=%s | model=%s | timeout_s=%s | connect_timeout_s=%s | request_backend=%s | num_ctx=%s | max_concurrent_requests=%s | medication_text_preview=%s",
                    early_validation_count, model, timeout, ollama_settings.connect_timeout_s, ollama_settings.request_backend, ollama_settings.num_ctx, max_concurrent_requests, str(text)[:180]
                )
            norm, progress_label, success, error_message = _ollama_norm_or_fallback(text)
            if success:
                stats["ollama_success"] += 1
                consecutive_initial_failures = 0
            else:
                stats["ollama_failure_fallback_success"] += 1
                consecutive_initial_failures += 1
                if (
                    row_level_cpu_fallback_enabled
                    and not row_level_cpu_fallback_attempted
                    and initial_failure_abort_count > 0
                    and consecutive_initial_failures >= initial_failure_abort_count
                ):
                    row_level_cpu_fallback_attempted = True
                    if llm_log:
                        llm_log.warning(
                            "Initial Ollama row failures reached %s with zero successful JSON parses; attempting CPU fallback restart on isolated port before aborting | base_url=%s",
                            consecutive_initial_failures, base_url,
                        )
                    if _restart_isolated_ollama_cpu_for_row_recovery(base_url, llm_log=llm_log):
                        consecutive_initial_failures = 0
                        stats["ollama_row_cpu_fallback_restart"] += 1
                        retry_norm, retry_label, retry_success, retry_error = _ollama_norm_or_fallback(text)
                        if retry_success:
                            norm = retry_norm
                            progress_label = "ollama_success_after_cpu_fallback"
                            stats["ollama_success"] += 1
                            success = True
                        else:
                            norm = retry_norm
                            progress_label = retry_label
                            error_message = retry_error
                    elif llm_log:
                        llm_log.error("Row-level CPU fallback restart failed or was refused; strict guardrail will abort if configured.")
                if not success and initial_failure_abort_count > 0 and consecutive_initial_failures >= initial_failure_abort_count:
                    msg = (
                        f"Ollama generation failed for the first {consecutive_initial_failures} attempted medication-text requests with zero successful JSON parses. "
                        f"Row-level CPU fallback attempted={row_level_cpu_fallback_attempted}. "
                        f"Aborting early instead of processing all {len(unique_order)} rows as fallback. Last error: {error_message}. "
                        "Check logs/llm.log and logs/ollama_server.log; test the server with curl; or use LLM_REQUIRE_OLLAMA=0 only for transparent fallback testing."
                    )
                    if llm_log:
                        llm_log.error(msg)
                    raise RuntimeError(msg)
            mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(norm))
            completed_since_cache_write += 1
            llm_bar.update(progress_label)
            if llm_log and (len(mapping) == 1 or len(mapping) % progress_log_interval == 0 or len(mapping) == len(unique_order)):
                llm_log.info(
                    "LLM medication-state progress | processed=%s/%s | latest_status=%s | ollama_success=%s | ollama_failure_fallback=%s | cache_hit=%s",
                    len(mapping), len(unique_order), progress_label, stats.get("ollama_success", 0), stats.get("ollama_failure_fallback_success", 0), stats.get("cache_hit", 0),
                )
            _write_cache_checkpoint(force=False)
            _write_partial_s2_checkpoint(force=False, latest_status=progress_label if 'progress_label' in locals() else '')

        if llm_pending_texts and max_concurrent_requests > 1:
            if llm_log:
                llm_log.info(
                    "Starting parallel Ollama medication-state abstraction | pending_rows=%s | max_concurrent_requests=%s",
                    len(llm_pending_texts), max_concurrent_requests,
                )
            with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
                future_to_text = {executor.submit(_ollama_norm_or_fallback, text): text for text in llm_pending_texts}
                for future in as_completed(future_to_text):
                    text = future_to_text[future]
                    norm, progress_label, success, _error_message = future.result()
                    mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(norm))
                    if success:
                        stats["ollama_success"] += 1
                    else:
                        stats["ollama_failure_fallback_success"] += 1
                    completed_since_cache_write += 1
                    llm_bar.update(progress_label)
                    if llm_log and (len(mapping) == 1 or len(mapping) % progress_log_interval == 0 or len(mapping) == len(unique_order)):
                        llm_log.info(
                            "LLM medication-state progress | processed=%s/%s | latest_status=%s | ollama_success=%s | ollama_failure_fallback=%s | cache_hit=%s",
                            len(mapping), len(unique_order), progress_label, stats.get("ollama_success", 0), stats.get("ollama_failure_fallback_success", 0), stats.get("cache_hit", 0),
                        )
                    _write_cache_checkpoint(force=False)
                    _write_partial_s2_checkpoint(force=False, latest_status=progress_label)
        else:
            for text in llm_pending_texts:
                if llm_log and (len(mapping) == 0 or (len(mapping) + 1) % progress_log_interval == 0):
                    llm_log.info(
                        "Ollama row request starting | processed=%s/%s | model=%s | timeout_s=%s | connect_timeout_s=%s | request_backend=%s | num_ctx=%s | medication_text_preview=%s",
                        len(mapping) + 1, len(unique_order), model, timeout, ollama_settings.connect_timeout_s, ollama_settings.request_backend, ollama_settings.num_ctx, str(text)[:180]
                    )
                norm, progress_label, success, _error_message = _ollama_norm_or_fallback(text)
                mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(norm))
                if success:
                    stats["ollama_success"] += 1
                else:
                    stats["ollama_failure_fallback_success"] += 1
                completed_since_cache_write += 1
                llm_bar.update(progress_label)
                if llm_log and (len(mapping) == 1 or len(mapping) % progress_log_interval == 0 or len(mapping) == len(unique_order)):
                    llm_log.info(
                        "LLM medication-state progress | processed=%s/%s | latest_status=%s | ollama_success=%s | ollama_failure_fallback=%s | cache_hit=%s",
                        len(mapping), len(unique_order), progress_label, stats.get("ollama_success", 0), stats.get("ollama_failure_fallback_success", 0), stats.get("cache_hit", 0),
                    )
                _write_cache_checkpoint(force=False)
                _write_partial_s2_checkpoint(force=False, latest_status=progress_label)

        # Final pass: materialize nonselected/empty/local-only rows after selected LLM rows.
        for text in local_pending_texts:
            cached = _cached_row_for_processing_text(text)
            cached_provider = str((cached or {}).get("effective_provider", "")).lower()
            cache_compatible = bool(cached) and _cache_compatible_for_current_run(cached_provider, should_try_llm=False, cached=cached)
            if cache_compatible:
                progress_label = f"cache_hit:{cached_provider or 'unknown'}"
                norm = _norm_from_cache(cached, cached_provider)
                mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(norm))
                stats["cache_hit"] += 1
            elif requested_provider in {"mock", "local_clinical_abstraction", "auto", "ollama"}:
                norm, status, progress_label = _local_norm_for_text(text)
                mapping[text] = _attach_drug_dictionary_identity(text, _attach_normalization_audit(norm))
                stats[status] += 1
            else:
                if llm_log:
                    llm_log.error("Unsupported llm_medication_state provider: %s", requested_provider)
                raise ValueError(f"Unsupported llm_medication_state provider: {requested_provider}")
            completed_since_cache_write += 1
            llm_bar.update(progress_label)
            _write_cache_checkpoint(force=False)
            _write_partial_s2_checkpoint(force=False, latest_status=progress_label)

    _write_cache_checkpoint(force=True)
    _write_partial_s2_checkpoint(force=True, latest_status="completed")

    audit_columns = [*TEXT_AUDIT_COLUMNS, *LLM_AUDIT_STATUS_COLUMNS, *LLM_CERTIFICATION_COLUMNS, *LLM_FEATURE_COLUMNS]

    if drug_token_mode:
        drug_rows = []
        for idx, text in enumerate(unique_order, start=1):
            if text not in mapping:
                continue
            rec = {"medication_text": text, **mapping[text]}
            identity = _drug_dictionary_identity(
                text,
                mapping_provider=str(pretrained_meta.get("actual_backend", "alias_only")),
                mapping_similarity_score=1.0,
            )
            rec.update(identity)
            rec["drug_dictionary_index"] = idx  # audit row number only; never used as a join key
            rec["drug_name"] = identity["normalized_drug_name"]
            rec["drug_name_record_frequency"] = int(token_counts.get(text, 0))
            rec["llm_abstraction_unit"] = "drug_token"
            rec["llm_abstraction_status"] = _audit_string(rec.get("llm_status", ""))
            rec["llm_drug_dictionary_aggregation_applied"] = 0
            rec["llm_drug_dictionary_token_count"] = 1 if _token_count(text) > 0 else 0
            rec["llm_drug_dictionary_tokens"] = text
            drug_rows.append(rec)
        drug_audit_columns = [
            "drug_dictionary_index",
            "raw_drug_name",
            "drug_name",
            "normalized_drug_name",
            "canonical_drug_name",
            "llm_dictionary_key",
            "mapping_provider",
            "mapping_similarity_score",
            "llm_abstraction_status",
            "drug_name_record_frequency",
            *audit_columns,
        ]
        drug_audit = pd.DataFrame(drug_rows, columns=drug_audit_columns)
        drug_audit = add_llm_certification_columns(drug_audit, settings)
        dictionary_output_path = None
        if partial_audit_path_resolved is not None:
            dictionary_output_path = partial_audit_path_resolved.with_name(str(settings.get("drug_dictionary_filename", "s2e_unique_drug_name_llm_dictionary.csv")))
        elif persistent_cache_path is not None:
            dictionary_output_path = persistent_cache_path.with_name(str(settings.get("drug_dictionary_filename", "s2e_unique_drug_name_llm_dictionary.csv")))
        if dictionary_output_path is not None:
            _atomic_write_dataframe_csv(drug_audit, dictionary_output_path)
            if llm_log:
                llm_log.info("Unique drug-name LLM dictionary written | rows=%s | path=%s", len(drug_audit), dictionary_output_path)
        unique_visit_text_order = [t for t in dict.fromkeys(dictionary_source_series.tolist()) if _token_count(t) > 0]
        default_norm = _normalize_response(
            local_clinical_abstraction(""),
            provider=fallback_provider,
            requested_provider=requested_provider,
            model=model,
            error="",
            status="empty_or_unmapped_medication_text",
            attempted=0,
            fallback_reason="empty_or_unmapped_medication_text",
            fallback_parse_ok=1,
            llm_parse_ok_override=0,
            medication_text="",
        )
        audit_rows = [{"medication_text": text, **_aggregate_drug_dictionary_rows(text, default_norm=default_norm)} for text in unique_visit_text_order]
        audit = pd.DataFrame(audit_rows, columns=audit_columns)
    else:
        audit_rows = [{"medication_text": text, **mapping[text]} for text in unique_order if text in mapping]
        audit = pd.DataFrame(audit_rows, columns=audit_columns)
        default_norm = _normalize_response(
            local_clinical_abstraction(""),
            provider=fallback_provider,
            requested_provider=requested_provider,
            model=model,
            error="",
            status="empty_or_unmapped_medication_text",
            attempted=0,
            fallback_reason="empty_or_unmapped_medication_text",
            fallback_parse_ok=1,
            llm_parse_ok_override=0,
            medication_text="",
        )

    audit = add_llm_certification_columns(audit, settings)
    ollama_success_count = int((audit.get("effective_provider", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "ollama").sum()) if not audit.empty else 0
    ollama_attempt_count = int(pd.to_numeric(audit.get("llm_attempted", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not audit.empty else 0
    if require_min_ollama_successes > 0 and requested_provider in {"auto", "ollama"} and ollama_success_count < require_min_ollama_successes:
        msg = (
            f"LLM medication-state abstraction produced {ollama_success_count} successful Ollama JSON parses "
            f"from {ollama_attempt_count} attempted rows, below required minimum {require_min_ollama_successes}. "
            "Review logs/llm.log and s2_medication_state_features/s2e_llm_medication_state_abstraction.csv; do not describe fallback-only outputs as LLM-derived."
        )
        if llm_log:
            llm_log.error(msg)
        if fail_on_error or require_ollama_when_enabled:
            raise RuntimeError(msg)
    if not drug_token_mode:
        _write_persistent_cache(persistent_cache_path, audit)
    if llm_log:
        llm_log.info("LLM medication-state abstraction completed | abstraction_unit=%s | stats=%s | audit_rows=%s | ollama_attempt_rows=%s | ollama_success_rows=%s | output_error_rows=%s", "drug_token" if drug_token_mode else "medication_text", dict(stats), len(audit), ollama_attempt_count, ollama_success_count, int(audit.get("llm_error", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum()) if not audit.empty else 0)

    if drug_token_mode:
        mapped_records = [_aggregate_drug_dictionary_rows(text, default_norm=default_norm) for text in dictionary_source_series.tolist()]
    else:
        mapped_records = [mapping.get(text, default_norm) if _token_count(text) > 0 else default_norm for text in dictionary_source_series.tolist()]
    feature_df = pd.DataFrame(mapped_records, index=out.index)
    feature_df = add_llm_certification_columns(feature_df, settings)

    for col in LLM_FEATURE_COLUMNS:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
        out[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)
    for col in ["llm_summary", "effective_provider", "feature_source", "llm_status", "llm_error", "llm_fallback_reason"]:
        if col not in feature_df.columns:
            feature_df[col] = ""
    out["llm_medication_state_summary"] = feature_df["llm_summary"].fillna("")
    out["llm_medication_state_provider"] = feature_df["effective_provider"].fillna("")
    out["llm_medication_state_feature_source"] = feature_df["feature_source"].fillna("")
    out["llm_medication_state_status"] = feature_df["llm_status"].fillna("")
    out["llm_medication_state_error"] = feature_df["llm_error"].fillna("")
    out["llm_medication_state_fallback_reason"] = feature_df["llm_fallback_reason"].fillna("")
    for col in LLM_CERTIFICATION_COLUMNS:
        if col in feature_df.columns:
            if col in {"llm_structural_quality_pass", "llm_certified_for_primary_analysis"}:
                out[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0).astype(int)
            else:
                out[col] = feature_df[col].fillna("").astype(str)
    return out, audit
