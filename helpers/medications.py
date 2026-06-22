"""
Project: dementia_progression
File: helpers/medications.py

Synopsis:
    Structured medication-state utilities for medication text construction, medication
    counts, and medication-category features.

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
import csv
import hashlib
import re
import string
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
import pandas as pd

CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "med_cat_benzodiazepine": [r"alprazolam", r"lorazepam", r"clonazepam", r"diazepam", r"temazepam", r"oxazepam", r"chlordiazepoxide"],
    "med_cat_antidepressant": [r"sertraline", r"citalopram", r"escitalopram", r"fluoxetine", r"paroxetine", r"venlafaxine", r"duloxetine", r"bupropion", r"mirtazapine", r"trazodone", r"amitriptyline", r"nortriptyline"],
    "med_cat_antipsychotic": [r"quetiapine", r"olanzapine", r"risperidone", r"aripiprazole", r"haloperidol", r"clozapine", r"ziprasidone"],
    "med_cat_sleep": [r"zolpidem", r"eszopiclone", r"zaleplon", r"melatonin", r"ramelteon", r"suvorexant", r"temazepam"],
    "med_cat_dementia": [r"donepezil", r"rivastigmine", r"galantamine", r"memantine"],
    "med_cat_anticholinergic_proxy": [r"diphenhydramine", r"oxybutynin", r"tolterodine", r"benztropine", r"amitriptyline", r"paroxetine", r"hydroxyzine"],
    "med_cat_statin": [r"atorvastatin", r"simvastatin", r"rosuvastatin", r"pravastatin", r"lovastatin", r"pitavastatin"],
    "med_cat_antihypertensive": [r"lisinopril", r"losartan", r"valsartan", r"amlodipine", r"metoprolol", r"carvedilol", r"hydrochlorothiazide", r"chlorthalidone"],
    "med_cat_anticoagulant_antiplatelet": [r"warfarin", r"apixaban", r"rivaroxaban", r"dabigatran", r"clopidogrel", r"aspirin"],
    "med_cat_diabetes": [r"metformin", r"glipizide", r"glyburide", r"insulin", r"empagliflozin", r"semaglutide", r"sitagliptin"],
    "med_cat_thyroid": [r"levothyroxine", r"liothyronine", r"methimazole", r"propylthiouracil"],
    "med_cat_opioid": [r"morphine", r"oxycodone", r"hydrocodone", r"fentanyl", r"tramadol", r"codeine", r"hydromorphone", r"buprenorphine"],
    "med_cat_antiparkinson": [r"levodopa", r"carbidopa", r"ropinirole", r"pramipexole", r"selegiline", r"rasagiline"],
    "med_cat_antiepileptic": [r"gabapentin", r"pregabalin", r"levetiracetam", r"lamotrigine", r"topiramate", r"valproate", r"divalproex"],
}
# Values in raw DRUG columns that are not medication names.  These are treated as
# structurally missing medication text, not as a drug exposure.  The list includes
# common NACC/codebook-style placeholders and the explicit user-observed artifact
# "*not codable*".
NON_MED_TOKENS = {
    "", "nan", "none", "unknown", "unk", "not reported", "not recorded", "not available",
    "missing", "n/a", "na", "null", ".", "-", "--", "no", "no meds", "no med",
    "no medication", "no medications", "not applicable", "not app",
    "not codable", "not codeable", "non codable", "non codeable", "uncodable",
    "not coded", "not entered", "not specified", "not otherwise specified",
}

# No-training medication alias normalization used before LLM abstraction.
# This is not a trained classifier; it only reduces superficial brand/generic spelling
# differences before the LLM sees a unique medication text. Users can extend or
# replace this via resources/medication/medication_aliases.csv.
DEFAULT_MEDICATION_ALIASES: Dict[str, str] = {
    "aricept": "donepezil",
    "namenda": "memantine",
    "namenda xr": "memantine",
    "exelon": "rivastigmine",
    "razadyne": "galantamine",
    "seroquel": "quetiapine",
    "zyprexa": "olanzapine",
    "risperdal": "risperidone",
    "abilify": "aripiprazole",
    "zoloft": "sertraline",
    "celexa": "citalopram",
    "lexapro": "escitalopram",
    "prozac": "fluoxetine",
    "paxil": "paroxetine",
    "effexor": "venlafaxine",
    "cymbalta": "duloxetine",
    "wellbutrin": "bupropion",
    "remeron": "mirtazapine",
    "desyrel": "trazodone",
    "xanax": "alprazolam",
    "ativan": "lorazepam",
    "klonopin": "clonazepam",
    "valium": "diazepam",
    "restoril": "temazepam",
    "ambien": "zolpidem",
    "lunesta": "eszopiclone",
    "neurontin": "gabapentin",
    "lyrica": "pregabalin",
    "keppra": "levetiracetam",
    "lamictal": "lamotrigine",
    "sinemet": "carbidopa levodopa",
    "requip": "ropinirole",
    "mirapex": "pramipexole",
    "benadryl": "diphenhydramine",
    "ditropan": "oxybutynin",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "crestor": "rosuvastatin",
    "pravachol": "pravastatin",
    "glucophage": "metformin",
    "coumadin": "warfarin",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "plavix": "clopidogrel",
    "synthroid": "levothyroxine",
}

def _non_med_key(text: str) -> str:
    """Normalize a token for sentinel/non-medication matching."""
    text = str(text or "").strip().lower()
    text = text.strip(string.whitespace + "*[](){}<>;:,\"\'`~")
    text = re.sub(r"[_/\\]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_non_medication_token(value: object) -> bool:
    """Return True when a raw DRUG value is a placeholder rather than medication text."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    key = _non_med_key(str(value))
    return key in NON_MED_TOKENS

def _clean_token(value: object) -> str:
    if is_non_medication_token(value):
        return ""
    text = str(value).strip().lower()
    text = text.strip(string.whitespace)
    text = re.sub(r"\s+", " ", text)
    # Remove only wrapper asterisks/quotes that surround a token; do not alter interior medication names.
    text = text.strip("*\"\'` ")
    return "" if is_non_medication_token(text) else text

def normalize_medication_text(value: object) -> str:
    """Return canonical pipe-delimited medication text with blank/non-codable tokens removed.

    This function is used by both the structured medication features and the LLM
    abstraction layer so that blank, whitespace-only, and *not codable* values are
    represented consistently as an empty medication list.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    raw = str(value)
    parts = re.split(r"\s*\|\s*", raw)
    tokens = [_clean_token(part) for part in parts]
    tokens = [token for token in tokens if token]
    return " | ".join(dict.fromkeys(tokens))



def load_medication_alias_map(path: object | None = None) -> Dict[str, str]:
    """Load a no-training medication alias map from CSV and merge it with defaults.

    Expected CSV columns are alias and canonical. Missing files are allowed and
    simply return the built-in defaults. This keeps the pipeline training-free and
    reproducible while allowing project-specific medication dictionaries when
    available.
    """
    aliases = dict(DEFAULT_MEDICATION_ALIASES)
    if path is None or str(path).strip() == "":
        return aliases
    p = Path(str(path)).expanduser()
    if not p.exists():
        return aliases
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = _clean_token(row.get("alias", ""))
            canonical = _clean_token(row.get("canonical", ""))
            if alias and canonical:
                aliases[alias] = canonical
    return aliases


def normalize_medication_alias_token(token: object, alias_map: Dict[str, str] | None = None) -> str:
    """Map brand/synonym medication tokens to canonical generic names without training."""
    cleaned = _clean_token(token)
    if not cleaned:
        return ""
    aliases = alias_map or DEFAULT_MEDICATION_ALIASES
    return aliases.get(cleaned, cleaned)

def canonicalize_medication_text_for_llm(
    value: object,
    *,
    sort_tokens: bool = True,
    normalize_aliases: bool = True,
    alias_map: Dict[str, str] | None = None,
    pretrained_token_normalizer: Callable[[object], tuple[str, float, str]] | None = None,
) -> str:
    """Return an LLM-facing canonical medication string.

    The raw visit-level medication_text may differ only by ordering, repeated tokens,
    spacing, noninformative placeholders, or common brand/generic aliases. This
    no-training normalization reduces duplicate LLM requests and gives the LLM a
    cleaner medication concept list.
    """
    normalized = normalize_medication_text(value)
    if not normalized:
        return ""
    tokens = [t.strip() for t in normalized.split(" | ") if t.strip()]
    if normalize_aliases:
        aliases = alias_map or DEFAULT_MEDICATION_ALIASES
        tokens = [normalize_medication_alias_token(t, aliases) for t in tokens]
        tokens = [t for t in tokens if t]
    if pretrained_token_normalizer is not None:
        normalized_tokens: list[str] = []
        for token in tokens:
            mapped, _score, _status = pretrained_token_normalizer(token)
            mapped = _clean_token(mapped)
            if mapped:
                normalized_tokens.append(mapped)
        tokens = normalized_tokens
    tokens = list(dict.fromkeys(tokens))
    if sort_tokens:
        tokens = sorted(tokens)
    return " | ".join(tokens)



# Default pretrained encoder model names for no-training biomedical normalization.
# These models are used only as frozen embedding models; no fine-tuning/training is
# performed by this project. When the model or optional dependencies are unavailable,
# the caller can either fail explicitly or fall back to alias-only normalization.
DEFAULT_PRETRAINED_NORMALIZER_MODELS: Dict[str, str] = {
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


def _strip_regex_token(pattern: str) -> str:
    """Convert simple regex-style medication patterns to readable vocabulary terms."""
    text = str(pattern or "")
    text = text.replace(r"\\b", " ")
    text = re.sub(r"[\\^$.*+?{}\[\]()`|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return _clean_token(text)


def load_medication_reference_vocabulary(
    path: object | None = None,
    *,
    alias_map: Dict[str, str] | None = None,
) -> List[str]:
    """Load canonical medication vocabulary for optional frozen-encoder normalization.

    This vocabulary is not a supervised training set. It is a set of candidate
    medication names used for nearest-neighbor normalization with a pretrained
    biomedical encoder such as SapBERT or BioClinicalBERT.
    """
    terms: list[str] = []
    aliases = alias_map or DEFAULT_MEDICATION_ALIASES
    terms.extend(list(aliases.keys()))
    terms.extend(list(aliases.values()))
    for patterns in CATEGORY_PATTERNS.values():
        for pat in patterns:
            term = _strip_regex_token(pat)
            if term:
                terms.append(term)
    if path is not None and str(path).strip():
        p = Path(str(path)).expanduser()
        if p.exists():
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ("canonical", "medication", "term", "alias", "name"):
                        val = _clean_token(row.get(key, ""))
                        if val:
                            terms.append(val)
    return sorted(dict.fromkeys([t for t in terms if t and not is_non_medication_token(t)]))


class FrozenEncoderMedicationNormalizer:
    """No-training medication normalizer backed by a frozen biomedical encoder.

    The class embeds a candidate medication vocabulary once and maps each medication
    token to its nearest vocabulary term when cosine similarity exceeds a threshold.
    It supports SapBERT and BioClinicalBERT through Hugging Face transformers. No
    gradient updates, fitting, or project-specific training occur.
    """

    def __init__(
        self,
        *,
        backend: str,
        model_name_or_path: str,
        vocabulary: List[str],
        similarity_threshold: float = 0.86,
        batch_size: int = 64,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.backend = str(backend).strip().lower()
        self.model_name_or_path = str(model_name_or_path).strip()
        self.vocabulary = [str(v).strip().lower() for v in vocabulary if str(v).strip()]
        self.similarity_threshold = float(similarity_threshold)
        self.batch_size = max(1, int(batch_size or 64))
        if not self.vocabulary:
            raise ValueError("Frozen encoder medication normalization requires a nonempty vocabulary.")
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional packages
            raise ImportError(
                "Frozen encoder medication normalization requires torch and transformers. "
                "Run 'python -m pip install -r requirements-pretrained.txt' inside the active virtual environment, "
                "or use llm_input_pretrained_normalization_backend=alias_only only for non-production debugging."
            ) from exc
        self._torch = torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=bool(local_files_only))
        self.model = AutoModel.from_pretrained(self.model_name_or_path, local_files_only=bool(local_files_only))
        self.model.to(self.device)
        self.model.eval()
        self.vocab_embeddings = self._embed_texts(self.vocabulary)
        self._cache: dict[str, tuple[str, float, str]] = {}

    def _embed_texts(self, texts: List[str]):  # pragma: no cover - optional heavy path
        torch = self._torch
        arrays = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                out = self.model(**encoded)
                token_emb = out.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                summed = (token_emb * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / denom
                emb = emb / emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
                arrays.append(emb.detach().cpu().numpy())
        import numpy as _np
        return _np.vstack(arrays)

    def normalize_token(self, token: object) -> tuple[str, float, str]:
        cleaned = _clean_token(token)
        if not cleaned:
            return "", 0.0, "empty"
        if cleaned in self._cache:
            return self._cache[cleaned]
        if cleaned in self.vocabulary:
            result = (cleaned, 1.0, "exact_vocab")
            self._cache[cleaned] = result
            return result
        import numpy as _np
        emb = self._embed_texts([cleaned])[0]
        sims = self.vocab_embeddings @ emb
        idx = int(_np.argmax(sims))
        score = float(sims[idx])
        if score >= self.similarity_threshold:
            result = (self.vocabulary[idx], round(score, 6), "encoder_match")
        else:
            result = (cleaned, round(score, 6), "below_threshold_kept_original")
        self._cache[cleaned] = result
        return result


def build_frozen_encoder_medication_normalizer(
    *,
    requested_backend: str = "alias_only",
    sapbert_model: str | None = None,
    bioclinicalbert_model: str | None = None,
    vocabulary_path: object | None = None,
    alias_map: Dict[str, str] | None = None,
    similarity_threshold: float = 0.86,
    batch_size: int = 64,
    device: str = "auto",
    local_files_only: bool = False,
    required: bool = False,
    logger: Any = None,
) -> tuple[Callable[[object], tuple[str, float, str]] | None, dict[str, Any]]:
    """Construct an optional no-training pretrained medication normalizer.

    Returns a token-normalization function plus metadata. If the requested model is
    unavailable and required=False, returns (None, metadata) so the caller can use
    alias-only canonicalization transparently while recording the actual backend.
    """
    backend = str(requested_backend or "alias_only").strip().lower()
    if backend in {"", "none", "off", "false", "alias", "alias_only"}:
        return None, {
            "requested_backend": backend or "alias_only",
            "actual_backend": "alias_only",
            "status": "alias_only_requested",
            "model": "",
            "vocab_size": 0,
            "similarity_threshold": float(similarity_threshold),
        }
    alias_map = alias_map or DEFAULT_MEDICATION_ALIASES
    vocabulary = load_medication_reference_vocabulary(vocabulary_path, alias_map=alias_map)
    sapbert_model = sapbert_model or DEFAULT_PRETRAINED_NORMALIZER_MODELS["sapbert"]
    bioclinicalbert_model = bioclinicalbert_model or DEFAULT_PRETRAINED_NORMALIZER_MODELS["bioclinicalbert"]
    candidates: list[tuple[str, str]]
    if backend in {"ensemble", "sapbert_bioclinicalbert", "sapbert_bioclinicalbert_ensemble", "bioclinicalbert_sapbert_ensemble"}:
        # Accuracy path: load both frozen encoders and use them as a no-training
        # medication normalization ensemble. This intentionally fails if either
        # encoder cannot load when required=True.
        errors: list[str] = []
        try:
            sapbert_norm = FrozenEncoderMedicationNormalizer(
                backend="sapbert",
                model_name_or_path=sapbert_model,
                vocabulary=vocabulary,
                similarity_threshold=float(similarity_threshold),
                batch_size=int(batch_size or 64),
                device=str(device or "auto"),
                local_files_only=bool(local_files_only),
            )
        except Exception as exc:  # pragma: no cover - optional package/model availability
            errors.append(f"sapbert:{sapbert_model}: {type(exc).__name__}: {exc}")
            sapbert_norm = None
        try:
            bioclinical_norm = FrozenEncoderMedicationNormalizer(
                backend="bioclinicalbert",
                model_name_or_path=bioclinicalbert_model,
                vocabulary=vocabulary,
                similarity_threshold=float(similarity_threshold),
                batch_size=int(batch_size or 64),
                device=str(device or "auto"),
                local_files_only=bool(local_files_only),
            )
        except Exception as exc:  # pragma: no cover - optional package/model availability
            errors.append(f"bioclinicalbert:{bioclinicalbert_model}: {type(exc).__name__}: {exc}")
            bioclinical_norm = None
        if sapbert_norm is not None and bioclinical_norm is not None:
            def _ensemble(token: object) -> tuple[str, float, str]:
                cleaned = _clean_token(token)
                if not cleaned:
                    return "", 0.0, "empty"
                s_text, s_score, s_status = sapbert_norm.normalize_token(cleaned)
                b_text, b_score, b_status = bioclinical_norm.normalize_token(cleaned)
                if s_text == b_text:
                    return s_text, round(max(float(s_score), float(b_score)), 6), f"ensemble_agree:{s_status}+{b_status}"
                # Prefer exact vocabulary matches, then the stronger above-threshold score.
                if s_status == "exact_vocab" and b_status != "exact_vocab":
                    return s_text, float(s_score), f"ensemble_sapbert_exact_bioclinicalbert_{b_status}"
                if b_status == "exact_vocab" and s_status != "exact_vocab":
                    return b_text, float(b_score), f"ensemble_bioclinicalbert_exact_sapbert_{s_status}"
                s_good = "below_threshold" not in str(s_status) and float(s_score) >= float(similarity_threshold)
                b_good = "below_threshold" not in str(b_status) and float(b_score) >= float(similarity_threshold)
                if s_good and (not b_good or float(s_score) >= float(b_score)):
                    return s_text, round(float(s_score), 6), f"ensemble_sapbert_selected_bioclinicalbert_{b_status}"
                if b_good:
                    return b_text, round(float(b_score), 6), f"ensemble_bioclinicalbert_selected_sapbert_{s_status}"
                # Neither encoder met threshold; keep the original cleaned token.
                return cleaned, round(max(float(s_score), float(b_score)), 6), f"ensemble_below_threshold_kept_original:{s_status}+{b_status}"
            if logger:
                logger.info(
                    "Loaded no-training pretrained medication normalizer ensemble | backends=sapbert,bioclinicalbert | models=%s,%s | vocab_size=%s | threshold=%s",
                    sapbert_model, bioclinicalbert_model, len(vocabulary), similarity_threshold,
                )
            return _ensemble, {
                "requested_backend": backend,
                "actual_backend": "ensemble_sapbert_bioclinicalbert",
                "status": "loaded",
                "model": f"sapbert={sapbert_model};bioclinicalbert={bioclinicalbert_model}",
                "vocab_size": len(vocabulary),
                "similarity_threshold": float(similarity_threshold),
            }
        if required:
            raise RuntimeError(
                "Requested SapBERT+BioClinicalBERT medication normalization ensemble could not be loaded and required=True. The launcher preflight should catch missing torch/transformers/model-cache problems before Step 2. Run 'python -m pip install -r requirements-pretrained.txt' and verify model access/cache, then rerun. Details: "
                + " | ".join(errors)
            )
        return None, {
            "requested_backend": backend,
            "actual_backend": "ensemble_fallback_alias_only",
            "status": "unavailable_fallback_alias_only",
            "model": "",
            "vocab_size": len(vocabulary),
            "similarity_threshold": float(similarity_threshold),
            "load_errors": " | ".join(errors)[:1000],
        }
    if backend in {"auto", "pretrained", "encoder", "sapbert_then_bioclinicalbert"}:
        candidates = [("sapbert", sapbert_model), ("bioclinicalbert", bioclinicalbert_model)]
    elif backend == "sapbert":
        candidates = [("sapbert", sapbert_model)]
    elif backend in {"clinicalbert", "bioclinicalbert", "bio_clinicalbert"}:
        candidates = [("bioclinicalbert", bioclinicalbert_model)]
    elif backend == "mock_encoder":
        # Test-only deterministic lightweight encoder analogue. It does not claim to
        # be ClinicalBERT/SapBERT; it verifies the pipeline wiring without downloads.
        from difflib import SequenceMatcher
        vocab = vocabulary
        def _mock(token: object) -> tuple[str, float, str]:
            cleaned = _clean_token(token)
            if not cleaned:
                return "", 0.0, "empty"
            if cleaned in vocab:
                return cleaned, 1.0, "exact_vocab"
            best = max(vocab, key=lambda v: SequenceMatcher(None, cleaned, v).ratio()) if vocab else cleaned
            score = SequenceMatcher(None, cleaned, best).ratio() if vocab else 0.0
            if score >= float(similarity_threshold):
                return best, round(float(score), 6), "mock_encoder_match"
            return cleaned, round(float(score), 6), "below_threshold_kept_original"
        return _mock, {
            "requested_backend": backend,
            "actual_backend": "mock_encoder",
            "status": "mock_encoder_loaded_for_tests",
            "model": "mock_encoder",
            "vocab_size": len(vocab),
            "similarity_threshold": float(similarity_threshold),
        }
    else:
        raise ValueError(f"Unsupported llm_input_pretrained_normalization_backend: {requested_backend!r}")

    errors: list[str] = []
    for candidate_backend, candidate_model in candidates:
        try:
            normalizer = FrozenEncoderMedicationNormalizer(
                backend=candidate_backend,
                model_name_or_path=candidate_model,
                vocabulary=vocabulary,
                similarity_threshold=float(similarity_threshold),
                batch_size=int(batch_size or 64),
                device=str(device or "auto"),
                local_files_only=bool(local_files_only),
            )
            if logger:
                logger.info(
                    "Loaded no-training pretrained medication normalizer | backend=%s | model=%s | vocab_size=%s | threshold=%s | device=%s",
                    candidate_backend, candidate_model, len(vocabulary), similarity_threshold, getattr(normalizer, "device", ""),
                )
            return normalizer.normalize_token, {
                "requested_backend": backend,
                "actual_backend": candidate_backend,
                "status": "loaded",
                "model": candidate_model,
                "vocab_size": len(vocabulary),
                "similarity_threshold": float(similarity_threshold),
            }
        except Exception as exc:  # pragma: no cover - optional package/model availability
            msg = f"{candidate_backend}:{candidate_model}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            if logger:
                logger.warning("Pretrained medication normalizer unavailable | %s", msg)
    if required:
        raise RuntimeError(
            "Requested pretrained medication normalization could not be loaded and required=True. "
            + " | ".join(errors)
        )
    return None, {
        "requested_backend": backend,
        "actual_backend": f"{backend}_fallback_alias_only" if backend != "auto" else "auto_fallback_alias_only",
        "status": "unavailable_fallback_alias_only",
        "model": "",
        "vocab_size": len(vocabulary),
        "similarity_threshold": float(similarity_threshold),
        "load_errors": " | ".join(errors)[:1000],
    }


def combine_drug_columns(df: pd.DataFrame, drug_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    present_cols = [c for c in drug_columns if c in out.columns]
    med_lists, med_texts, med_counts = [], [], []
    if not present_cols:
        out["medication_list"] = [[] for _ in range(len(out))]
        out["medication_text"] = ""
        out["medication_count"] = 0
        return out
    for row in out[present_cols].itertuples(index=False, name=None):
        tokens: list[str] = []
        for value in row:
            normalized = normalize_medication_text(value)
            if normalized:
                tokens.extend([t for t in normalized.split(" | ") if t.strip()])
        unique_tokens = list(dict.fromkeys(tokens))
        med_lists.append(unique_tokens)
        med_texts.append(" | ".join(unique_tokens))
        med_counts.append(len(unique_tokens))
    out["medication_list"] = med_lists
    out["medication_text"] = med_texts
    out["medication_count"] = med_counts
    return out


def extract_unique_drug_names_from_columns(df: pd.DataFrame, drug_columns: Iterable[str]) -> pd.DataFrame:
    """Return a row-frequency dictionary of unique raw drug-name tokens before visit-level combining.

    This supports the v1.49 fast Step 2 design: raw DRUG columns are first
    tokenized into unique drug names, then downstream LLM/BioClinicalBERT/SapBERT
    abstraction can operate on the drug-name dictionary instead of every unique
    full medication-list combination.
    """
    present_cols = [c for c in drug_columns if c in df.columns]
    counts: dict[str, dict[str, Any]] = {}
    if not present_cols:
        return pd.DataFrame(columns=[
            "drug_dictionary_index", "drug_name", "normalized_drug_name",
            "record_frequency", "source_column_count", "source_columns",
        ])
    for row in df[present_cols].itertuples(index=False, name=None):
        row_tokens: dict[str, set[str]] = {}
        for col, value in zip(present_cols, row):
            normalized = normalize_medication_text(value)
            if not normalized:
                continue
            for token in [t.strip() for t in normalized.split(" | ") if t.strip()]:
                norm = normalize_medication_text(token)
                if not norm:
                    continue
                row_tokens.setdefault(norm, set()).add(col)
        for token, cols in row_tokens.items():
            rec = counts.setdefault(token, {"record_frequency": 0, "source_columns": set()})
            rec["record_frequency"] += 1
            rec["source_columns"].update(cols)
    rows = []
    for idx, (token, rec) in enumerate(sorted(counts.items(), key=lambda kv: (-int(kv[1]["record_frequency"]), kv[0])), start=1):
        cols = sorted(rec["source_columns"])
        normalized_token = normalize_medication_text(token)
        stable_key = hashlib.sha256(normalized_token.encode("utf-8")).hexdigest() if normalized_token else ""
        rows.append({
            "drug_dictionary_index": idx,
            "raw_drug_name": token,
            "drug_name": token,
            "normalized_drug_name": normalized_token,
            "canonical_drug_name": normalized_token,
            "llm_dictionary_key": stable_key,
            "record_frequency": int(rec["record_frequency"]),
            "source_column_count": int(len(cols)),
            "source_columns": ";".join(cols),
        })
    return pd.DataFrame(rows)

def _token_list_from_medication_text(value: object) -> list[str]:
    normalized = normalize_medication_text(value)
    if not normalized:
        return []
    return [t.strip() for t in str(normalized).split(" | ") if t.strip()]


def _unique_token_count_matching_patterns(value: object, category_names: Iterable[str]) -> int:
    """Count unique medication tokens matching any pattern in the supplied domains."""
    tokens = _token_list_from_medication_text(value)
    if not tokens:
        return 0
    patterns: list[str] = []
    for name in category_names:
        patterns.extend(CATEGORY_PATTERNS.get(name, []))
    if not patterns:
        return 0
    regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
    return len({token for token in tokens if regex.search(token)})


def add_medication_category_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    med_text = out.get("medication_text", pd.Series([""] * len(out), index=out.index)).apply(normalize_medication_text)
    out["medication_text"] = med_text
    out["medication_count"] = med_text.apply(lambda x: len(_token_list_from_medication_text(x))).astype(int)
    for category, patterns in CATEGORY_PATTERNS.items():
        regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
        out[category] = med_text.apply(lambda x: int(any(regex.search(token) for token in _token_list_from_medication_text(x))))

    # These are true unique-medication counts, not overlapping domain-burden sums.
    # Therefore medication_psychotropic_count cannot exceed medication_count.
    psychotropic_categories = [
        "med_cat_benzodiazepine", "med_cat_antidepressant", "med_cat_antipsychotic",
        "med_cat_sleep", "med_cat_anticholinergic_proxy",
    ]
    neuro_categories = ["med_cat_dementia", "med_cat_antiparkinson", "med_cat_antiepileptic"]
    cardiometabolic_categories = [
        "med_cat_statin", "med_cat_antihypertensive", "med_cat_anticoagulant_antiplatelet",
        "med_cat_diabetes", "med_cat_thyroid",
    ]
    out["medication_psychotropic_count"] = med_text.apply(lambda x: _unique_token_count_matching_patterns(x, psychotropic_categories)).astype(int)
    out["medication_neuro_count"] = med_text.apply(lambda x: _unique_token_count_matching_patterns(x, neuro_categories)).astype(int)
    out["medication_cardiometabolic_count"] = med_text.apply(lambda x: _unique_token_count_matching_patterns(x, cardiometabolic_categories)).astype(int)
    return out
