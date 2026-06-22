#!/usr/bin/env bash
# Project: dementia_progression
# File: src/run_pipeline.sh
#
# Synopsis:
#   Executable src-level launcher that validates paths, preserves runtime settings, and
#   calls the Python pipeline controller.
#
# Author:
#   puru panta (purupanta@uky.edu)
#
# Date Created:
#   2026-05-19
#
# Last Updated:
#   2026-05-19
#
# Version:
#   1.0
#
# Purpose:
#   Supports the dementia_progression pipeline for medication-state-aware and LLM-
#   enhanced machine learning prediction of next-visit dementia progression among mild
#   cognitive impairment visits.
#
# Notes:
#   This project uses participant-level train-validation-test splitting to prevent
#   participant-level leakage. Neuropathology variables are excluded from model training
#   and used only for secondary biological plausibility anchoring. Medication-state and
#   LLM-derived features are interpreted as predictive patient-state representations,
#   not as causal medication effects.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLI_CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|-c)
      if [[ $# -lt 2 ]]; then
        echo "[dementia_progression][ERROR] --config requires a YAML path" >&2
        exit 2
      fi
      CLI_CONFIG="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash src/run_pipeline.sh [config.yaml] | bash src/run_pipeline.sh --config config.yaml"
      exit 0
      ;;
    --*)
      echo "[dementia_progression][ERROR] Unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ -z "$CLI_CONFIG" ]]; then
        CLI_CONFIG="$1"
        shift
      else
        echo "[dementia_progression][ERROR] Unexpected extra argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

CONFIG_PATH="${CLI_CONFIG:-${CONFIG_PATH:-${PIPELINE_CONFIG:-$PROJECT_ROOT/configs/project.yaml}}}"
DEMENTIA_PROJECT_NAME="${DEMENTIA_PROJECT_NAME:-dementia_progression}"

export CONFIG_PATH DEMENTIA_PROJECT_NAME
# Do not set default DEMENTIA_OUTPUT_DIR or DEMENTIA_APPEND_DATETIME here: doing so
# would override output_dir and append_datetime_to_output_dir from the selected YAML.
# Only preserve these variables when the user explicitly set them before launch.
if [[ -n "${DEMENTIA_OUTPUT_DIR+x}" ]]; then
  export DEMENTIA_OUTPUT_DIR
fi
if [[ -n "${DEMENTIA_APPEND_DATETIME+x}" ]]; then
  export DEMENTIA_APPEND_DATETIME
fi
if [[ -n "${DEMENTIA_INPUT_CSV+x}" ]]; then
  export DEMENTIA_INPUT_CSV
fi
export PROGRESS_BARS="${PROGRESS_BARS:-true}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-2}}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[dementia_progression][ERROR] Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -n "${DEMENTIA_INPUT_CSV+x}" && ! -f "$DEMENTIA_INPUT_CSV" ]]; then
  echo "[dementia_progression][ERROR] Input CSV not found: $DEMENTIA_INPUT_CSV" >&2
  echo "[dementia_progression][ERROR] Place the NACC file at: ip/data_nacc65/investigator_nacc65 w last visit.csv or set input_csv in the YAML config." >&2
  exit 3
fi

echo "[dementia_progression] Executing Python pipeline: python -m src.run_pipeline --config $CONFIG_PATH"
python -m src.run_pipeline --config "$CONFIG_PATH"
