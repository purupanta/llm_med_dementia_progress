"""
Project: dementia_progression
File: src/pipeline_context.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-22
Last Updated: 2026-05-22

Synopsis:
    Lightweight mutable context object passed between numbered pipeline-stage
    modules. The object makes data dependencies explicit at the orchestrator
    boundary while avoiding a large monolithic main function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineContext:
    """Mutable state container for the stepwise pipeline.

    Attributes are attached by the orchestrator and by each numbered step. This
    keeps step functions testable and enables future replacement of individual
    stages without changing the public command-line interface.
    """

    pass
