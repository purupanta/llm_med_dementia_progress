"""
Project: dementia_progression
File: helpers/neural_text_representations.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Optional BioClinicalBERT and SapBERT medication-text representation helpers.

Purpose:
    Creates visit-level medication-text representation features from frozen
    pretrained clinical/biomedical encoders when the local model dependencies and
    model cache are available.

Notes:
    These features are predictive patient-state representations and should not be
    interpreted as causal medication effects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

from helpers.progress import progress_iter


def neural_text_feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("bioclinicalbert_medtxt_", "sapbert_medtxt_", "clinicalbert_sapbert_medtxt_")
    return [c for c in df.columns if c.startswith(prefixes) and pd.api.types.is_numeric_dtype(df[c])]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _provider_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rep_cfg = cfg.get("neural_text_representations", {}) or {}
    providers = rep_cfg.get("providers", {}) or {}
    out = []
    for provider_name, provider_cfg in providers.items():
        provider_cfg = provider_cfg or {}
        if not _as_bool(provider_cfg.get("enabled", True), True):
            continue
        out.append({"name": str(provider_name).strip().lower(), **provider_cfg})
    return out


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _load_hf_model(model_name_or_path: str, device: str, *, local_files_only: bool = False):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "BioClinicalBERT/SapBERT representations require torch and transformers. "
            "Install the project requirements or disable neural_text_representations."
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def _embed_texts(texts: list[str], *, model_name_or_path: str, batch_size: int, max_length: int, device: str, local_files_only: bool, progress_enabled: bool, desc: str) -> np.ndarray:
    import torch

    tokenizer, model, device = _load_hf_model(model_name_or_path, device, local_files_only=local_files_only)
    vectors: list[np.ndarray] = []
    for start in progress_iter(range(0, len(texts), int(batch_size)), enabled=progress_enabled, desc=desc, total=(len(texts) + int(batch_size) - 1) // int(batch_size), unit="batch"):
        batch = texts[start : start + int(batch_size)]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors)


def _project_embeddings(embeddings: np.ndarray, *, n_components: int, random_state: int) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    n_components = int(max(1, min(n_components, embeddings.shape[1])))
    if n_components >= embeddings.shape[1]:
        return embeddings.astype(np.float32)
    projector = GaussianRandomProjection(n_components=n_components, random_state=int(random_state))
    return projector.fit_transform(embeddings).astype(np.float32)


def _cache_key(provider_name: str, model_name_or_path: str, texts: list[str], n_components: int, max_length: int) -> str:
    payload = {
        "provider": provider_name,
        "model": model_name_or_path,
        "text_hashes": [_sha256(x) for x in texts],
        "n_components": int(n_components),
        "max_length": int(max_length),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _cache_dir(cfg: dict[str, Any], output_dir: Path) -> Path:
    rep_cfg = cfg.get("neural_text_representations", {}) or {}
    base = rep_cfg.get("cache_dir", "op/neural_text_representation_cache")
    path = Path(str(base)).expanduser()
    if not path.is_absolute():
        project_root = Path(str(cfg.get("project_root", output_dir.parent))).resolve()
        path = project_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_neural_text_representations(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    output_dir: str | Path,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append BioClinicalBERT/SapBERT medication-text representation features.

    When enabled, this function encodes unique medication_text values with each
    configured Hugging Face encoder, applies a fixed random projection to control
    dimensionality, and maps the projected representation back to visits.
    """
    out = df.copy()
    rep_cfg = cfg.get("neural_text_representations", {}) or {}
    enabled = _as_bool(rep_cfg.get("enabled", False), False)
    audit_rows: list[dict[str, Any]] = []
    if not enabled:
        return out, pd.DataFrame([{"provider": "disabled", "status": "disabled", "n_unique_texts": 0, "n_components": 0}])

    text_col = str(rep_cfg.get("text_column", "medication_text"))
    if text_col not in out.columns:
        if _as_bool(rep_cfg.get("strict_model_loading", True), True):
            raise ValueError(f"neural_text_representations.text_column not found: {text_col}")
        return out, pd.DataFrame([{"provider": "all", "status": "skipped_text_column_missing", "text_column": text_col}])

    texts_all = out[text_col].map(_clean_text).fillna("").astype(str)
    unique_texts = list(dict.fromkeys(texts_all.tolist()))
    nonempty_texts = [t for t in unique_texts if t]
    if not nonempty_texts:
        return out, pd.DataFrame([{"provider": "all", "status": "skipped_no_nonempty_text", "n_unique_texts": 0}])

    random_state = int(rep_cfg.get("random_state", cfg.get("random_state", 42)))
    batch_size = int(rep_cfg.get("batch_size", 16))
    max_length = int(rep_cfg.get("max_length", 96))
    n_components = int(rep_cfg.get("n_components", 32))
    device = str(rep_cfg.get("device", "auto"))
    local_files_only = _as_bool(rep_cfg.get("local_files_only", False), False)
    strict = _as_bool(rep_cfg.get("strict_model_loading", True), True)
    progress_enabled = bool((cfg.get("progress", {}) or {}).get("enabled", True))
    cdir = _cache_dir(cfg, Path(output_dir))

    provider_feature_prefixes: list[str] = []
    for provider in _provider_configs(cfg):
        name = provider["name"]
        model_name_or_path = str(provider.get("model_name_or_path", "")).strip()
        if not model_name_or_path:
            msg = f"No model_name_or_path configured for provider={name}"
            if strict:
                raise ValueError(msg)
            audit_rows.append({"provider": name, "status": "skipped_missing_model_path", "error": msg})
            continue
        provider_components = int(provider.get("n_components", n_components))
        provider_max_length = int(provider.get("max_length", max_length))
        prefix = str(provider.get("feature_prefix", f"{name}_medtxt_")).strip()
        provider_feature_prefixes.append(prefix)
        key = _cache_key(name, model_name_or_path, nonempty_texts, provider_components, provider_max_length)
        cache_npz = cdir / f"{name}_{key}.npz"
        status = "computed"
        try:
            if cache_npz.exists():
                loaded = np.load(cache_npz, allow_pickle=False)
                projected = loaded["features"].astype(np.float32)
                status = "cache_hit"
            else:
                embeddings = _embed_texts(
                    nonempty_texts,
                    model_name_or_path=model_name_or_path,
                    batch_size=batch_size,
                    max_length=provider_max_length,
                    device=device,
                    local_files_only=_as_bool(provider.get("local_files_only", local_files_only), local_files_only),
                    progress_enabled=progress_enabled,
                    desc=f"{name} medication-text embeddings",
                )
                projected = _project_embeddings(embeddings, n_components=provider_components, random_state=random_state)
                np.savez_compressed(cache_npz, features=projected)
            feature_cols = [f"{prefix}{i + 1:03d}" for i in range(projected.shape[1])]
            mapping = {text: projected[i, :] for i, text in enumerate(nonempty_texts)}
            matrix = np.zeros((len(out), len(feature_cols)), dtype=np.float32)
            for i, text in enumerate(texts_all.tolist()):
                vec = mapping.get(text)
                if vec is not None:
                    matrix[i, :] = vec
            for j, col in enumerate(feature_cols):
                out[col] = matrix[:, j]
            audit_rows.append(
                {
                    "provider": name,
                    "status": status,
                    "model_name_or_path": model_name_or_path,
                    "text_column": text_col,
                    "n_unique_texts": len(unique_texts),
                    "n_nonempty_unique_texts": len(nonempty_texts),
                    "n_components": len(feature_cols),
                    "feature_prefix": prefix,
                    "cache_file": str(cache_npz),
                    "device_requested": device,
                    "local_files_only": _as_bool(provider.get("local_files_only", local_files_only), local_files_only),
                }
            )
            if logger:
                logger.info("Neural text representation created: provider=%s features=%s status=%s", name, len(feature_cols), status)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to create neural text representation for provider={name}: {exc}") from exc
            audit_rows.append(
                {
                    "provider": name,
                    "status": "failed_non_strict",
                    "model_name_or_path": model_name_or_path,
                    "error": str(exc),
                }
            )
            if logger:
                logger.warning("Neural text representation skipped for provider=%s because: %s", name, exc)

    # Optional interaction/ensemble summary features created from both encoders when available.
    bioclinical_cols = [c for c in out.columns if c.startswith("bioclinicalbert_medtxt_")]
    sapbert_cols = [c for c in out.columns if c.startswith("sapbert_medtxt_")]
    if bioclinical_cols and sapbert_cols:
        n = min(len(bioclinical_cols), len(sapbert_cols))
        for i in range(n):
            out[f"clinicalbert_sapbert_medtxt_mean_{i + 1:03d}"] = (
                pd.to_numeric(out[bioclinical_cols[i]], errors="coerce").fillna(0.0)
                + pd.to_numeric(out[sapbert_cols[i]], errors="coerce").fillna(0.0)
            ) / 2.0
        audit_rows.append({"provider": "clinicalbert_sapbert_ensemble", "status": "created_mean_fusion", "n_components": n})

    audit = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame([{"provider": "all", "status": "no_enabled_provider"}])
    return out, audit
