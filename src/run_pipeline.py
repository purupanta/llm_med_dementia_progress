"""
Project: dementia_progression
File: src/run_pipeline.py

Author: puru panta (purupanta@uky.edu)
Date Created: 2026-05-19
Last Updated: 2026-05-22

Synopsis:
    Refactored command-line orchestrator for the dementia progression pipeline.
    The executable logic for each numbered analytic stage lives in a separate
    `src/pipeline_step*.py` module. This file parses configuration, prepares the
    run context, and calls Step 1 through Step 7 in order.

GitHub readiness:
    The previous validated v1.49 monolithic pipeline was decomposed into
    step-specific files without intentionally changing the analytic outputs.
    Helper functions are re-exported from `src.pipeline_common` for backward
    compatibility with the existing tests and scripts.
"""

from __future__ import annotations

# Support both direct script execution and module execution:
#   python src/run_pipeline.py --config configs/project.yaml
#   python -m src.run_pipeline --config configs/project.yaml
# Direct script execution initially places src/ on sys.path rather than the
# project root, so the project root is inserted before importing src.* modules.
import sys
from pathlib import Path

_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

from src.pipeline_common import *  # Backward-compatible helper re-export.
from src.pipeline_common import _drop_all_blank_columns, _filter_rows_missing_required_columns, _neuropathology_anchor_outputs
from src.pipeline_context import PipelineContext
from src.pipeline_step1_source_cohort import run_step as run_step1
from src.pipeline_step2_medication_features import run_step as run_step2
from src.pipeline_step3_split_fit import run_step as run_step3
from src.pipeline_step4_evaluation import run_step as run_step4
from src.pipeline_step5_predictions import run_step as run_step5
from src.pipeline_step6_neuropathology import run_step as run_step6
from src.pipeline_step7_reporting import run_step as run_step7


def initialize_context() -> PipelineContext:
    """Parse runtime inputs, resolve configuration, initialize logs, and return context."""
    args = parse_args()
    cfg = apply_environment_overrides(load_config(args.config))
    project_root = PROJECT_ROOT
    cfg["project_root"] = str(project_root)
    input_csv = resolve_path(project_root, cfg["input_csv"])
    output_dir = ensure_dir(build_output_dir(project_root, cfg))
    logs_dir = ensure_dir(output_dir / "logs")
    run_loggers = setup_run_loggers(logs_dir)

    ctx = PipelineContext()
    ctx.args = args
    ctx.cfg = cfg
    ctx.project_root = project_root
    ctx.input_csv = input_csv
    ctx.output_dir = output_dir
    ctx.logs_dir = logs_dir
    ctx.run_loggers = run_loggers
    ctx.logger = run_loggers["pipeline"]
    ctx.llm_logger = run_loggers["llm"]
    ctx.config_logger = run_loggers["config"]
    ctx.runtime_logger = run_loggers["runtime"]
    ctx.gpu_logger = run_loggers["gpu"]
    ctx.warning_logger = run_loggers["warnings"]
    ctx.decimal_places = int(cfg.get("output_options", {}).get("decimal_places", 6))
    ctx.threshold = float(cfg.get("modeling", {}).get("threshold", 0.50))
    ctx.progress_enabled = progress_enabled_from_config(cfg)
    return ctx


def log_run_start(ctx: PipelineContext) -> None:
    """Write reproducibility logs before numbered analytic steps begin."""
    log_resolved_config(ctx.config_logger, ctx.cfg)
    log_runtime_environment(ctx.runtime_logger, project_root=ctx.project_root, input_csv=ctx.input_csv, output_dir=ctx.output_dir)
    log_gpu_environment(ctx.gpu_logger)
    ctx.warning_logger.info("warnings.log initialized. Python warnings and pipeline warnings will be written here.")
    ctx.llm_logger.info("llm.log initialized for medication-state abstraction calls, fallback events, timing, and provider diagnostics.")

    progress_step(f"Starting project: {ctx.cfg.get('project_name')} | output_dir={ctx.output_dir}", enabled=ctx.progress_enabled, logger=ctx.logger)
    ctx.logger.info("Starting project: %s", ctx.cfg.get("project_name"))
    ctx.logger.info("Input CSV: %s", ctx.input_csv)
    ctx.logger.info("Output directory: %s", ctx.output_dir)
    ctx.logger.info("Detailed logs directory: %s", ctx.logs_dir)
    if (ctx.cfg.get("runtime", {}) or {}).get("resume", {}).get("active_output_dir"):
        ctx.logger.info(
            "Resume mode active: writing to existing output_dir=%s and reusing persistent Step 2 LLM cache when compatible",
            ctx.output_dir,
        )


def run_manuscript_readiness_gate(ctx: PipelineContext) -> None:
    """Run the strict manuscript-readiness output gate after Step 7.

    The gate writes `manuscript_readiness_gate_report.csv` and
    `manuscript_readiness_overall_status.csv` in the run output directory. When
    configured with fail_pipeline_on_error=true, any hard FAIL raises a
    RuntimeError so invalid LLM-enhanced outputs cannot be mistaken for
    manuscript-ready artifacts.
    """
    gate_cfg = (ctx.cfg.get("manuscript_readiness_gate", {}) or {})
    if not bool(gate_cfg.get("enabled", True)):
        ctx.logger.info("Manuscript-readiness gate disabled by configuration.")
        return
    try:
        from scripts.manuscript_readiness_gate import run_gate
        report = run_gate(ctx.output_dir, write_report=bool(gate_cfg.get("write_report", True)))
        fail_count = int(report["status"].eq("FAIL").sum()) if not report.empty else 1
        warn_count = int(report["status"].eq("WARN").sum()) if not report.empty else 0
        ctx.logger.info("Manuscript-readiness gate completed: fail_count=%s warn_count=%s", fail_count, warn_count)
        if fail_count and bool(gate_cfg.get("fail_pipeline_on_error", True)):
            raise RuntimeError(
                f"Manuscript-readiness gate failed with {fail_count} hard failure(s). "
                f"See {ctx.output_dir / 'manuscript_readiness_gate_report.csv'}."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        if bool(gate_cfg.get("fail_pipeline_on_error", True)):
            raise RuntimeError(f"Manuscript-readiness gate could not complete: {type(exc).__name__}: {exc}") from exc
        ctx.logger.warning("Manuscript-readiness gate could not complete: %s: %s", type(exc).__name__, exc)


def main() -> None:
    """Run Step 1 through Step 7 in the validated order."""
    ctx = initialize_context()
    log_run_start(ctx)

    if not run_step1(ctx):
        return
    if not run_step2(ctx):
        return
    run_step3(ctx)
    run_step4(ctx)
    run_step5(ctx)
    run_step6(ctx)
    run_step7(ctx)
    run_manuscript_readiness_gate(ctx)


if __name__ == "__main__":
    main()
