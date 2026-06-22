#!/usr/bin/env bash
# Project: dementia_progression
# File: run_pipeline.sh
#
# Synopsis:
#   Backward-compatible root launcher that sets default GPU visibility and delegates to
#   run_pipelines.sh.
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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Explicit CUDA GPU selection for users who still call this wrapper directly.
# Physical GPU 2 is the project default. After CUDA_VISIBLE_DEVICES is set,
# CUDA libraries usually expose that physical GPU inside the process as cuda:0.
export GPU_ID="${GPU_ID:-2}"
export CUDA_GPU_ID="${CUDA_GPU_ID:-$GPU_ID}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="$CUDA_GPU_ID"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$CUDA_GPU_ID}"

echo "[dementia_progression] run_pipeline.sh wrapper using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec bash "$PROJECT_ROOT/run_pipelines.sh" "$@"
