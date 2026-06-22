"""
Project: dementia_progression
File: helpers/progress.py

Synopsis:
    Progress-bar and progress-message helper utilities for visible execution tracking.

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

import sys
from typing import Iterable, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - defensive fallback for minimal environments
    _tqdm = None


class ProgressBar:
    """Small manual progress-bar wrapper used for numbered pipeline steps."""

    def __init__(
        self,
        *,
        total: int,
        desc: str,
        enabled: bool = True,
        unit: str = "task",
        leave: bool = True,
        logger=None,
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.enabled = enabled
        self.unit = unit
        self.leave = leave
        self.logger = logger
        self.current = 0
        self._bar = None
        if self.enabled and _tqdm is not None:
            self._bar = _tqdm(total=self.total, desc=self.desc, unit=self.unit, dynamic_ncols=True, leave=self.leave)
        elif self.enabled:
            print(f"[progress] {self.desc}: 0/{self.total} {self.unit}", file=sys.stderr, flush=True)

    def update(self, label: str = "", n: int = 1) -> None:
        """Advance the progress bar and log the current subtask."""
        inc = max(int(n), 0)
        self.current = min(self.total, self.current + inc) if self.total else self.current + inc
        if self.logger is not None and label:
            self.logger.info("%s progress %s/%s: %s", self.desc, self.current, self.total, label)
        if self._bar is not None:
            if label:
                self._bar.set_postfix_str(str(label)[:80])
            self._bar.update(inc)
        elif self.enabled:
            suffix = f" - {label}" if label else ""
            print(f"[progress] {self.desc}: {self.current}/{self.total} {self.unit}{suffix}", file=sys.stderr, flush=True)

    def close(self) -> None:
        """Close the progress bar cleanly."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def progress_bar(
    *,
    total: int,
    desc: str,
    enabled: bool = True,
    unit: str = "task",
    leave: bool = True,
    logger=None,
) -> ProgressBar:
    """Create a manual progress bar for a named pipeline step."""
    return ProgressBar(total=total, desc=desc, enabled=enabled, unit=unit, leave=leave, logger=logger)


def progress_iter(
    iterable: Iterable[T],
    *,
    enabled: bool = True,
    desc: str = "",
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
) -> Iterable[T]:
    """Return an iterable wrapped in a progress bar when progress reporting is enabled."""
    if not enabled or _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)


def progress_step(message: str, *, enabled: bool = True, logger=None) -> None:
    """Print and log a high-level pipeline step message."""
    if logger is not None:
        logger.info(message)
    if enabled:
        print(f"[progress] {message}", file=sys.stderr, flush=True)
