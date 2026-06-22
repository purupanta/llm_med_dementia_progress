#!/usr/bin/env bash
# Project: dementia_progression
# File: run_pipelines.sh
#
# Synopsis:
#   Primary one-command launcher for the complete dementia progression prediction
#   pipeline. It writes launcher/config/pipeline/LLM/runtime/warnings logs under each
#   timestamped output folder and can run this project on an isolated Ollama port.
#
# Author:
#   puru panta (purupanta@uky.edu)
#
# Date Created:
#   2026-05-19
#
# Last Updated:
#   2026-05-22

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Command-line config parsing
# -----------------------------------------------------------------------------
# Supported forms:
#   bash run_pipelines.sh
#   bash run_pipelines.sh configs/smoke_synthetic.yaml
#   bash run_pipelines.sh --config configs/smoke_synthetic.yaml
#   bash run_pipeline.sh configs/smoke_synthetic.yaml
#   bash run_pipeline.sh --config configs/smoke_synthetic.yaml
#
# Priority is: explicit CLI config > PIPELINE_CONFIG environment variable >
# configs/project.yaml default. This makes the wrapper arguments work while
# preserving environment-driven runs when no CLI config is supplied.
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
      cat <<'EOF'
Usage:
  bash run_pipelines.sh [config.yaml]
  bash run_pipelines.sh --config config.yaml
  bash run_pipeline.sh [config.yaml]
  bash run_pipeline.sh --config config.yaml

Examples:
  bash run_pipeline.sh configs/project.yaml
  bash run_pipeline.sh configs/smoke_synthetic.yaml
  PIPELINE_CONFIG=configs/smoke_synthetic.yaml bash run_pipelines.sh
EOF
      exit 0
      ;;
    --*)
      echo "[dementia_progression][ERROR] Unknown launcher option: $1" >&2
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

# -----------------------------------------------------------------------------
# User-editable runtime settings
# -----------------------------------------------------------------------------
export DEMENTIA_PROJECT_NAME="${DEMENTIA_PROJECT_NAME:-dementia_progression}"

resolve_config_path() {
  local raw_config="$1"
  local candidate=""
  if [[ -z "$raw_config" ]]; then
    raw_config="configs/project.yaml"
  fi
  if [[ "$raw_config" = /* ]]; then
    candidate="$raw_config"
  elif [[ -f "$PROJECT_ROOT/$raw_config" ]]; then
    candidate="$PROJECT_ROOT/$raw_config"
  elif [[ -f "$PWD/$raw_config" ]]; then
    candidate="$PWD/$raw_config"
  else
    # Keep the expected project-root path for a clear error message below.
    candidate="$PROJECT_ROOT/$raw_config"
  fi
  python - "$candidate" <<'PYRESOLVE'
import pathlib, sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PYRESOLVE
}

RAW_PIPELINE_CONFIG="${CLI_CONFIG:-${PIPELINE_CONFIG:-configs/project.yaml}}"
export CONFIG_PATH="$(resolve_config_path "$RAW_PIPELINE_CONFIG")"
export PIPELINE_CONFIG="$CONFIG_PATH"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[dementia_progression][ERROR] Config file not found: $CONFIG_PATH" >&2
  echo "[dementia_progression][ERROR] Project root resolved to: $PROJECT_ROOT" >&2
  echo "[dementia_progression][ERROR] Current working directory after launcher setup: $PWD" >&2
  echo "[dementia_progression][ERROR] Run from a clean extracted project folder that contains configs/project.yaml, for example:" >&2
  echo "  cd dementia_progression_v2.14" >&2
  echo "  bash run_pipeline.sh configs/project.yaml" >&2
  if [[ -d "$PROJECT_ROOT/configs" ]]; then
    echo "[dementia_progression][ERROR] Available config files under $PROJECT_ROOT/configs:" >&2
    find "$PROJECT_ROOT/configs" -maxdepth 1 -type f -name '*.yaml' -printf '  %p\n' | sort >&2 || true
  else
    echo "[dementia_progression][ERROR] No configs/ directory exists under $PROJECT_ROOT." >&2
  fi
  exit 2
fi

# Do not force input_csv from the launcher when a non-project config is supplied;
# otherwise synthetic/smoke configs cannot use their own input path.
if [[ -n "${DEMENTIA_INPUT_CSV+x}" ]]; then
  export DEMENTIA_INPUT_CSV
elif [[ "$(basename "$PIPELINE_CONFIG")" == "project.yaml" ]]; then
  export DEMENTIA_INPUT_CSV="$PROJECT_ROOT/ip/data_nacc65/investigator_nacc65 w last visit.csv"
fi

# Output can still be overridden from the shell. If it is not overridden, read
# output_dir from the fully merged YAML config so smoke/profile configs keep their
# own output folder instead of being forced to the production output base.
CONFIG_OUTPUT_DIR="$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import sys, pathlib
config_path = pathlib.Path(sys.argv[1]).resolve()
project_root = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project_root))
try:
    from helpers.config_utils import load_config
    cfg = load_config(config_path)
    print(str(cfg.get("output_dir", "./op/dementia_progression")))
except Exception:
    print("./op/dementia_progression")
PY
)"
export DEMENTIA_OUTPUT_DIR="${DEMENTIA_OUTPUT_DIR:-$CONFIG_OUTPUT_DIR}"
CONFIG_APPEND_DATETIME="$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PYAPPEND'
import sys, pathlib
config_path = pathlib.Path(sys.argv[1]).resolve()
project_root = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project_root))
try:
    from helpers.config_utils import load_config
    cfg = load_config(config_path)
    val = (cfg.get("output_options") or {}).get("append_datetime_to_output_dir", True)
    print("true" if bool(val) else "false")
except Exception:
    print("true")
PYAPPEND
)"
export DEMENTIA_APPEND_DATETIME="${DEMENTIA_APPEND_DATETIME:-$CONFIG_APPEND_DATETIME}"

# YAML/env-controlled resume. This is intentionally resolved before launcher.log is
# opened so launcher, Ollama, and Python logs all land in the resumed run folder.
# Environment override: DEMENTIA_RESUME_OUTPUT_DIR=/path/to/op/dementia_progression_YYYYmmdd_HHMMSS
RESOLVED_RESUME_OUTPUT_DIR="${DEMENTIA_RESUME_OUTPUT_DIR:-${RESUME_OUTPUT_DIR:-}}"
if [[ -z "$RESOLVED_RESUME_OUTPUT_DIR" ]]; then
  RESOLVED_RESUME_OUTPUT_DIR="$(python - "$CONFIG_PATH" "$PROJECT_ROOT" "$DEMENTIA_OUTPUT_DIR" <<'PY' || true
import sys, pathlib
config_path = pathlib.Path(sys.argv[1]).resolve()
project_root = pathlib.Path(sys.argv[2]).resolve()
output_base = pathlib.Path(sys.argv[3])
if not output_base.is_absolute():
    output_base = (project_root / output_base).resolve()
sys.path.insert(0, str(project_root))
try:
    from helpers.config_utils import load_config
    cfg = load_config(config_path)
except Exception:
    print("")
    raise SystemExit(0)
runtime = cfg.get("runtime") or {}
resume = runtime.get("resume") or {}
enabled = bool(resume.get("enabled", False))
if not enabled:
    print("")
    raise SystemExit(0)
resume_dir = str(resume.get("output_dir", "") or "").strip()
if resume_dir:
    p = pathlib.Path(resume_dir)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    print(str(p))
    raise SystemExit(0)
if bool(resume.get("use_latest_matching_output_dir", False)):
    sep = str((cfg.get("output_options") or {}).get("output_datetime_separator", "_"))
    matches = [p for p in output_base.parent.glob(output_base.name + sep + "*") if p.is_dir()]
    if matches:
        matches.sort(key=lambda x: (x.stat().st_mtime, x.name), reverse=True)
        print(str(matches[0]))
        raise SystemExit(0)
print("")
PY
)"
fi

# Use one shared timestamp for the shell launcher and Python pipeline so all logs
# and outputs are placed in the same run directory. In resume mode, reuse the
# prior output directory instead of creating a new timestamped folder.
if [[ -n "$RESOLVED_RESUME_OUTPUT_DIR" ]]; then
  export DEMENTIA_RESUME_OUTPUT_DIR="$RESOLVED_RESUME_OUTPUT_DIR"
  RUN_OUTPUT_DIR="$RESOLVED_RESUME_OUTPUT_DIR"
  mkdir -p "$RUN_OUTPUT_DIR/logs"
  export DEMENTIA_APPEND_DATETIME="false"
else
  export DEMENTIA_RUN_TIMESTAMP="${DEMENTIA_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  if [[ "${DEMENTIA_APPEND_DATETIME,,}" =~ ^(true|1|yes|y|on)$ ]]; then
    RUN_OUTPUT_DIR="${DEMENTIA_OUTPUT_DIR}_${DEMENTIA_RUN_TIMESTAMP}"
  else
    RUN_OUTPUT_DIR="$DEMENTIA_OUTPUT_DIR"
  fi
  mkdir -p "$RUN_OUTPUT_DIR/logs"
fi
exec > >(tee -a "$RUN_OUTPUT_DIR/logs/launcher.log") 2>&1
if [[ -n "${DEMENTIA_RESUME_OUTPUT_DIR:-}" ]]; then
  echo "[resume] resume mode active; output_dir=$DEMENTIA_RESUME_OUTPUT_DIR"
fi

# Explicit CUDA GPU selection. Physical GPU 2 is the default for this project.
# GPU 3 is intentionally excluded by default because it may be used by another
# pipeline. The launcher checks only GPU 2 by default, then falls back to CPU. Inside a CUDA process, the selected physical GPU is presented as
# cuda:0 after CUDA_VISIBLE_DEVICES.
export DEMENTIA_EXCLUDED_GPUS="${DEMENTIA_EXCLUDED_GPUS:-3}"
export GPU_ID="${GPU_ID:-2}"
export CUDA_GPU_ID="${CUDA_GPU_ID:-$GPU_ID}"

gpu_is_excluded() {
  local gpu="$1"
  local excluded_csv=",${DEMENTIA_EXCLUDED_GPUS// /},"
  [[ "$excluded_csv" == *",${gpu},"* ]]
}

if gpu_is_excluded "$CUDA_GPU_ID"; then
  echo "[dementia_progression] requested/default CUDA GPU $CUDA_GPU_ID is excluded by DEMENTIA_EXCLUDED_GPUS=$DEMENTIA_EXCLUDED_GPUS; switching to GPU 2"
  export GPU_ID="2"
  export CUDA_GPU_ID="2"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="$CUDA_GPU_ID"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$CUDA_GPU_ID}"

# Reproducible, server-safe CPU threading.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# -----------------------------------------------------------------------------
# Required pretrained-encoder runtime preflight
# -----------------------------------------------------------------------------
# The full accuracy profile uses frozen SapBERT and BioClinicalBERT. Check these
# dependencies before starting Ollama or entering analytic Step 1/2 so failures do
# not occur midway through medication-feature construction.
export PRETRAINED_AUTO_INSTALL_DEPS="${PRETRAINED_AUTO_INSTALL_DEPS:-1}"
export PRETRAINED_MODEL_LOAD_PREFLIGHT="${PRETRAINED_MODEL_LOAD_PREFLIGHT:-1}"
export PRETRAINED_PREFLIGHT_ENABLED="${PRETRAINED_PREFLIGHT_ENABLED:-1}"
if [[ "$PRETRAINED_PREFLIGHT_ENABLED" == "1" ]]; then
  echo "[preflight] PRETRAINED_AUTO_INSTALL_DEPS=$PRETRAINED_AUTO_INSTALL_DEPS"
  echo "[preflight] PRETRAINED_MODEL_LOAD_PREFLIGHT=$PRETRAINED_MODEL_LOAD_PREFLIGHT"
  python "$PROJECT_ROOT/scripts/preflight_pretrained_runtime.py" --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
else
  echo "[preflight] Frozen pretrained encoder preflight disabled by PRETRAINED_PREFLIGHT_ENABLED=$PRETRAINED_PREFLIGHT_ENABLED"
fi

# -----------------------------------------------------------------------------
# Config-aware Ollama skip for no-Ollama profiles
# -----------------------------------------------------------------------------
# Smoke/local configs that explicitly use provider=mock, provider=local_clinical_abstraction,
# provider=off, or llm_medication_state.enabled=false should not start/probe Ollama
# unless the user explicitly sets DEMENTIA_OLLAMA_MODE. Full project.yaml remains
# provider=auto and keeps the isolated Ollama behavior by default.
CONFIG_LLM_PROVIDER="$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import sys, pathlib
config_path = pathlib.Path(sys.argv[1]).resolve()
project_root = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project_root))
try:
    from helpers.config_utils import load_config
    cfg = load_config(config_path)
    llm = cfg.get("llm_medication_state") or {}
    enabled = bool(llm.get("enabled", True))
    provider = str(llm.get("provider", "auto") or "auto").strip().lower()
    print(f"enabled={str(enabled).lower()};provider={provider}")
except Exception:
    print("enabled=true;provider=auto")
PY
)"
if [[ -z "${DEMENTIA_OLLAMA_MODE+x}" ]]; then
  case "$CONFIG_LLM_PROVIDER" in
    enabled=false*|*provider=mock|*provider=local_clinical_abstraction|*provider=off)
      export DEMENTIA_OLLAMA_MODE="off"
      echo "[ollama] config uses no-Ollama LLM provider; defaulting DEMENTIA_OLLAMA_MODE=off | $CONFIG_LLM_PROVIDER"
      ;;
  esac
fi

# -----------------------------------------------------------------------------
# Ollama isolation controls
# -----------------------------------------------------------------------------
# Default is isolated so this project does not compete with a different pipeline
# already using the default Ollama host http://127.0.0.1:11434.
# Modes:
#   isolated : use/start http://127.0.0.1:11435 for this project
#   shared   : intentionally use DEMENTIA_SHARED_OLLAMA_BASE_URL, default 11434
#   off      : do not start/check Ollama; set LLM_PROVIDER=off/local/mock as needed
export DEMENTIA_OLLAMA_MODE="${DEMENTIA_OLLAMA_MODE:-isolated}"
export DEMENTIA_OLLAMA_HOST="${DEMENTIA_OLLAMA_HOST:-127.0.0.1}"
export DEMENTIA_OLLAMA_PORT="${DEMENTIA_OLLAMA_PORT:-11435}"
export DEMENTIA_SHARED_OLLAMA_BASE_URL="${DEMENTIA_SHARED_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export OLLAMA_AUTO_START="${OLLAMA_AUTO_START:-1}"
export OLLAMA_STARTUP_WAIT_SECONDS="${OLLAMA_STARTUP_WAIT_SECONDS:-45}"
export OLLAMA_PROBE_TIMEOUT_SECONDS="${OLLAMA_PROBE_TIMEOUT_SECONDS:-180}"
export OLLAMA_PROBE_ENABLED="${OLLAMA_PROBE_ENABLED:-1}"
# Probe output mode. Default is JSON because the medication-state pipeline
# requires parseable JSON. This catches JSON-mode generation problems before
# Step 2 starts. Options: json | none.
export OLLAMA_PROBE_FORMAT_MODE="${OLLAMA_PROBE_FORMAT_MODE:-json}"
# If the isolated 11435 server answers /api/tags but cannot complete /api/generate,
# restart only the process bound to the isolated project port. Never touch 11434.
export OLLAMA_RESTART_ON_PROBE_FAIL="${OLLAMA_RESTART_ON_PROBE_FAIL:-1}"
export OLLAMA_STOP_TIMEOUT_SECONDS="${OLLAMA_STOP_TIMEOUT_SECONDS:-10}"
# If generation fails on the current isolated server, try a controlled GPU failover.
# This only restarts the Ollama process bound to DEMENTIA_OLLAMA_PORT and never touches 11434.
export OLLAMA_FAILOVER_GPU_ENABLED="${OLLAMA_FAILOVER_GPU_ENABLED:-1}"
export DEMENTIA_GPU_CANDIDATES="${DEMENTIA_GPU_CANDIDATES:-2}"
# Model probing remains supported, but defaults to the configured model only.
# This keeps a plain bash run simple: GPU 2 first, then CPU fallback.
# Let YAML choose the LLM model profile unless the user explicitly overrides it.
: "${LLM_MODEL_PROFILE:=}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
export OLLAMA_MODEL_FROM_LAUNCHER="${OLLAMA_MODEL_FROM_LAUNCHER:-1}"
# Prefer the accuracy model, but automatically use an already installed smaller local model
# if qwen2.5:7b-instruct is absent. The selected model is logged and exported to Python.
# This prevents a hard stop when the isolated 11435 server is healthy but the preferred
# 7B model has not been pulled yet.
export OLLAMA_MODEL_FAILOVER_ENABLED="${OLLAMA_MODEL_FAILOVER_ENABLED:-1}"
export OLLAMA_MODEL_CANDIDATES="${OLLAMA_MODEL_CANDIDATES:-qwen2.5:7b-instruct,qwen2.5:3b-instruct,qwen2.5:1.5b-instruct,llama3.2:3b}"
# Do not force pretrained-normalization settings from the launcher; let YAML decide.
# Full project.yaml requires SapBERT+BioClinicalBERT, while smoke configs can disable it.
: "${LLM_INPUT_PRETRAINED_NORMALIZATION_BACKEND:=}"
: "${LLM_INPUT_PRETRAINED_NORMALIZATION_REQUIRED:=}"
# Last resort: if all GPU candidates fail generation, start the isolated 11435
# server with CUDA hidden so Ollama can use CPU. This is slower but avoids a hard
# stop when the GPU backend is hanging. Port 11434 is still preserved.
export OLLAMA_CPU_FALLBACK_ENABLED="${OLLAMA_CPU_FALLBACK_ENABLED:-1}"
export OLLAMA_CPU_PROBE_TIMEOUT_SECONDS="${OLLAMA_CPU_PROBE_TIMEOUT_SECONDS:-240}"
# Conservative Ollama server settings for GTX 1080 class GPUs.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-1024}"
# v2.13: align default Ollama server parallelism with the accuracy_fast_qwen7b YAML profile.
# This changes throughput only; prompts/model/schema and audit gates are unchanged.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-3}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-10m}"
export OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama 2>/dev/null || true)}"

ollama_reachable() {
  local base_url="$1"
  python - "$base_url" <<'PY'
import sys, urllib.request
base = sys.argv[1].rstrip('/')
try:
    with urllib.request.urlopen(base + '/api/tags', timeout=1.5) as r:
        raise SystemExit(0 if int(getattr(r, 'status', 500)) < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}


pids_listening_on_port() {
  local port="$1"
  ss -ltnp 2>/dev/null | awk -v pat=":${port}" '$4 ~ pat {print $0}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

normalize_gpu_candidates() {
  local raw="$1"
  local excluded_csv=",${DEMENTIA_EXCLUDED_GPUS// /},"
  echo "$raw" | tr ',' ' ' | awk -v excluded="$excluded_csv" '{
    for (i=1; i<=NF; i++) {
      if ($i != "" && index(excluded, "," $i ",") == 0 && !seen[$i]++) {
        printf "%s%s", (out++ ? " " : ""), $i
      }
    }
  }'
}

normalize_model_candidates() {
  local raw="$1"
  echo "$raw" | tr ',' ' ' | awk '{
    for (i=1; i<=NF; i++) {
      if ($i != "" && !seen[$i]++) {
        printf "%s%s", (out++ ? " " : ""), $i
      }
    }
  }'
}

set_project_cuda_gpu() {
  local gpu="$1"
  export GPU_ID="$gpu"
  export CUDA_GPU_ID="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export NVIDIA_VISIBLE_DEVICES="$gpu"
  unset OLLAMA_LLM_LIBRARY || true
  echo "[ollama] selected project CUDA GPU=$gpu for isolated port ${DEMENTIA_OLLAMA_PORT}; preserving any server on 127.0.0.1:11434"
}

set_project_cpu_mode() {
  export GPU_ID="cpu"
  export CUDA_GPU_ID="cpu"
  export CUDA_VISIBLE_DEVICES=""
  export NVIDIA_VISIBLE_DEVICES="none"
  export HIP_VISIBLE_DEVICES=""
  export ROCR_VISIBLE_DEVICES=""
  export GGML_VK_VISIBLE_DEVICES=""
  export LLM_REQUEST_TIMEOUT_S="${LLM_REQUEST_TIMEOUT_S:-300}"
  echo "[ollama] selected CPU fallback for isolated port ${DEMENTIA_OLLAMA_PORT}; preserving any server on 127.0.0.1:11434"
}

ollama_model_installed() {
  local base_url="$1"
  local model="$2"
  python - "$base_url" "$model" <<'PYMODEL'
import json
import sys
import urllib.request
base = sys.argv[1].rstrip('/')
target = sys.argv[2]
try:
    with urllib.request.urlopen(base + '/api/tags', timeout=3.0) as r:
        data = json.load(r)
except Exception:
    # If tags cannot be parsed, do not block a direct generation attempt.
    raise SystemExit(0)
models = data.get('models', []) if isinstance(data, dict) else []
names = set()
for item in models:
    if isinstance(item, dict):
        for key in ('name', 'model'):
            val = item.get(key)
            if val:
                names.add(str(val))
raise SystemExit(0 if target in names else 1)
PYMODEL
}

try_ollama_model_probes() {
  local base_url="$1"
  local raw_models="${OLLAMA_MODEL:-qwen2.5:7b-instruct},${OLLAMA_MODEL_CANDIDATES}"
  local models
  models="$(normalize_model_candidates "$raw_models")"
  local model
  for model in $models; do
    if [[ "$OLLAMA_MODEL_FAILOVER_ENABLED" != "1" && "$model" != "${OLLAMA_MODEL:-qwen2.5:7b-instruct}" ]]; then
      continue
    fi
    if ! ollama_model_installed "$base_url" "$model"; then
      echo "[ollama] model candidate not installed on isolated server; skipping | model=$model"
      continue
    fi
    export OLLAMA_MODEL="$model"
    echo "[ollama] probing installed model candidate on isolated server | model=$OLLAMA_MODEL | host=$base_url"
    if ollama_generate_probe "$base_url"; then
      echo "[ollama] selected Ollama model for this run | OLLAMA_MODEL=$OLLAMA_MODEL"
      return 0
    fi
    echo "[ollama] generation probe failed for model candidate | model=$model"
  done
  return 3
}

stop_isolated_ollama_port() {
  local port="${DEMENTIA_OLLAMA_PORT}"
  local host="${DEMENTIA_OLLAMA_HOST}"

  if [[ "$port" == "11434" ]]; then
    echo "[ollama][ERROR] Refusing to stop port 11434 because another pipeline may be using the default Ollama server." >&2
    return 4
  fi

  local pids
  pids="$(pids_listening_on_port "$port" || true)"
  if [[ -z "$pids" ]]; then
    echo "[ollama] no process is listening on isolated port ${host}:${port}; nothing to stop"
    return 0
  fi

  local pid comm args
  for pid in $pids; do
    comm="$(ps -p "$pid" -o comm= 2>/dev/null | tr -d ' ' || true)"
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$comm" != "ollama" && "$args" != *"ollama"* ]]; then
      echo "[ollama][ERROR] Refusing to stop non-Ollama process pid=$pid on port $port: $args" >&2
      return 5
    fi
    echo "[ollama] stopping isolated Ollama pid=$pid bound to ${host}:${port}; preserving any server on 127.0.0.1:11434"
    kill "$pid" 2>/dev/null || true
  done

  local waited=0
  while [[ "$waited" -lt "$OLLAMA_STOP_TIMEOUT_SECONDS" ]]; do
    if [[ -z "$(pids_listening_on_port "$port" || true)" ]]; then
      echo "[ollama] isolated port ${host}:${port} stopped after ${waited}s"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  pids="$(pids_listening_on_port "$port" || true)"
  if [[ -n "$pids" ]]; then
    for pid in $pids; do
      comm="$(ps -p "$pid" -o comm= 2>/dev/null | tr -d ' ' || true)"
      args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if [[ "$comm" == "ollama" || "$args" == *"ollama"* ]]; then
        echo "[ollama] isolated Ollama pid=$pid still bound to port $port after ${OLLAMA_STOP_TIMEOUT_SECONDS}s; sending SIGKILL to this port-specific process only"
        kill -KILL "$pid" 2>/dev/null || true
      else
        echo "[ollama][ERROR] Non-Ollama process remains on isolated port $port; not killing pid=$pid" >&2
        return 6
      fi
    done
  fi
}

ollama_generate_probe() {
  local base_url="$1"
  local model="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
  if [[ "$OLLAMA_PROBE_ENABLED" != "1" ]]; then
    echo "[ollama] generate probe skipped | OLLAMA_PROBE_ENABLED=$OLLAMA_PROBE_ENABLED"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "[ollama] curl not found; generate probe skipped"
    return 0
  fi
  echo "[ollama] generate probe starting | host=$base_url | model=$model | timeout=${OLLAMA_PROBE_TIMEOUT_SECONDS}s | format_mode=$OLLAMA_PROBE_FORMAT_MODE"
  local payload
  if [[ "${OLLAMA_PROBE_FORMAT_MODE,,}" == "json" ]]; then
    payload='{"model":"'"$model"'","prompt":"Return only this JSON object: {\"ready\": true}","stream":false,"format":"json","options":{"temperature":0,"num_predict":32,"num_ctx":1024}}'
  else
    payload='{"model":"'"$model"'","prompt":"Say OK only.","stream":false,"options":{"temperature":0,"num_predict":8,"num_ctx":1024}}'
  fi
  local response_file="$RUN_OUTPUT_DIR/logs/ollama_generate_probe_response.json"
  if printf '%s' "$payload" | curl --silent --show-error --fail       --connect-timeout 10 --max-time "$OLLAMA_PROBE_TIMEOUT_SECONDS"       --header 'Content-Type: application/json' --data-binary @-       "$base_url/api/generate" > "$response_file" 2>> "$RUN_OUTPUT_DIR/logs/ollama_server.log"; then
    echo "[ollama] generate probe succeeded | response=$response_file"
    return 0
  fi
  echo "[ollama] generate probe failed or timed out | response=$response_file | see logs/ollama_server.log" >&2
  echo "[ollama] This prevents the pipeline from appearing stuck at 0% during the first LLM row." >&2
  return 3
}

maybe_start_isolated_ollama() {
  local requested_gpu="${1:-$CUDA_VISIBLE_DEVICES}"
  local base_url="http://${DEMENTIA_OLLAMA_HOST}:${DEMENTIA_OLLAMA_PORT}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-$base_url}"
  export OLLAMA_HOST="${DEMENTIA_OLLAMA_HOST}:${DEMENTIA_OLLAMA_PORT}"

  if ollama_reachable "$OLLAMA_BASE_URL"; then
    echo "[ollama] isolated server already responsive | host=$OLLAMA_BASE_URL | requested_gpu=$requested_gpu"
    return 0
  fi

  if [[ "$OLLAMA_AUTO_START" != "1" ]]; then
    echo "[ollama] isolated server not responsive and OLLAMA_AUTO_START=$OLLAMA_AUTO_START | host=$OLLAMA_BASE_URL"
    return 0
  fi

  if [[ -z "$OLLAMA_BIN" || ! -x "$OLLAMA_BIN" ]]; then
    echo "[ollama] command not found; Python pipeline will use fallback if provider=auto and fail_on_error=false"
    return 0
  fi

  if [[ "$requested_gpu" == "cpu" ]]; then
    set_project_cpu_mode
  else
    set_project_cuda_gpu "$requested_gpu"
  fi
  echo "[ollama] starting isolated project server | OLLAMA_HOST=$OLLAMA_HOST | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<hidden>} | NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>} | log=$RUN_OUTPUT_DIR/logs/ollama_server.log"
  nohup env     OLLAMA_HOST="$OLLAMA_HOST"     CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"     NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"     HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"     ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"     GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-}"     CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER"     OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH"     OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"     OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS"     OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE"     OLLAMA_LOAD_TIMEOUT="$OLLAMA_LOAD_TIMEOUT"     OLLAMA_DEBUG="${OLLAMA_DEBUG:-INFO}"     "$OLLAMA_BIN" serve >> "$RUN_OUTPUT_DIR/logs/ollama_server.log" 2>&1 &
  echo "$!" > "$RUN_OUTPUT_DIR/logs/ollama_server.pid"

  local waited=0
  while [[ "$waited" -lt "$OLLAMA_STARTUP_WAIT_SECONDS" ]]; do
    if ollama_reachable "$OLLAMA_BASE_URL"; then
      echo "[ollama] isolated server responsive after ${waited}s | host=$OLLAMA_BASE_URL | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[ollama] isolated server did not become responsive within ${OLLAMA_STARTUP_WAIT_SECONDS}s | host=$OLLAMA_BASE_URL | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  return 1
}

try_isolated_ollama_gpu_failover() {
  local base_url="$1"
  local candidates
  candidates="$(normalize_gpu_candidates "$DEMENTIA_GPU_CANDIDATES")"
  if [[ "$DEMENTIA_OLLAMA_PORT" == "11434" ]]; then
    echo "[ollama][ERROR] Refusing GPU failover on port 11434 because another pipeline may be using it." >&2
    return 4
  fi
  if [[ "$OLLAMA_FAILOVER_GPU_ENABLED" != "1" ]]; then
    echo "[ollama] GPU failover disabled | OLLAMA_FAILOVER_GPU_ENABLED=$OLLAMA_FAILOVER_GPU_ENABLED" >&2
    return 3
  fi
  echo "[ollama] generation probe failed; trying isolated-port GPU failover candidates: $candidates"
  local gpu
  for gpu in $candidates; do
    echo "[ollama] trying isolated Ollama on CUDA GPU $gpu at ${DEMENTIA_OLLAMA_HOST}:${DEMENTIA_OLLAMA_PORT}; preserving 11434"
    stop_isolated_ollama_port || return $?
    set_project_cuda_gpu "$gpu"
    maybe_start_isolated_ollama "$gpu" || true
    if ! ollama_reachable "$base_url"; then
      echo "[ollama] isolated server not reachable after start on GPU $gpu"
      continue
    fi
    if try_ollama_model_probes "$base_url"; then
      echo "[ollama] generation probe succeeded after GPU/model failover | selected_gpu=$gpu | selected_model=${OLLAMA_MODEL:-<yaml>} | host=$base_url"
      return 0
    fi
    echo "[ollama] generation probe still failed on GPU $gpu for all installed model candidates"
  done

  if [[ "$OLLAMA_CPU_FALLBACK_ENABLED" == "1" ]]; then
    echo "[ollama] isolated GPU 2 candidate failed or was unavailable; trying CPU fallback on port ${DEMENTIA_OLLAMA_PORT}; preserving 11434"
    stop_isolated_ollama_port || return $?
    local old_probe_timeout="$OLLAMA_PROBE_TIMEOUT_SECONDS"
    export OLLAMA_PROBE_TIMEOUT_SECONDS="$OLLAMA_CPU_PROBE_TIMEOUT_SECONDS"
    maybe_start_isolated_ollama "cpu" || true
    if ollama_reachable "$base_url" && try_ollama_model_probes "$base_url"; then
      echo "[ollama] generation probe succeeded with CPU fallback | selected_model=${OLLAMA_MODEL:-<yaml>} | host=$base_url"
      return 0
    fi
    export OLLAMA_PROBE_TIMEOUT_SECONDS="$old_probe_timeout"
    echo "[ollama] CPU fallback generation probe failed"
  fi

  echo "[ollama][ERROR] generation probe failed for isolated GPU candidate(s): $candidates" >&2
  echo "[ollama][ERROR] CPU fallback enabled=$OLLAMA_CPU_FALLBACK_ENABLED. Port 11434 was preserved." >&2
  echo "[ollama][ERROR] Checked model candidates: ${OLLAMA_MODEL:-<unset>},${OLLAMA_MODEL_CANDIDATES:-<unset>}" >&2
  echo "[ollama][ERROR] Pull at least one candidate on this machine, for example: OLLAMA_HOST=http://127.0.0.1:${DEMENTIA_OLLAMA_PORT} ollama pull qwen2.5:3b-instruct" >&2
  echo "[ollama][ERROR] Check $RUN_OUTPUT_DIR/logs/ollama_server.log and ollama_generate_probe_response.json." >&2
  return 3
}

case "${DEMENTIA_OLLAMA_MODE,,}" in
  isolated)
    maybe_start_isolated_ollama
    ;;
  shared)
    export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-$DEMENTIA_SHARED_OLLAMA_BASE_URL}"
    if ollama_reachable "$OLLAMA_BASE_URL"; then
      echo "[ollama] using intentionally shared Ollama server | host=$OLLAMA_BASE_URL"
    else
      echo "[ollama] shared server not responsive | host=$OLLAMA_BASE_URL"
    fi
    ;;
  off)
    echo "[ollama] Ollama preflight disabled by DEMENTIA_OLLAMA_MODE=off"
    ;;
  *)
    echo "[ollama] invalid DEMENTIA_OLLAMA_MODE=$DEMENTIA_OLLAMA_MODE; expected isolated, shared, or off" >&2
    exit 2
    ;;
esac

if [[ "${DEMENTIA_OLLAMA_MODE,,}" != "off" ]] && [[ -n "${OLLAMA_BASE_URL:-}" ]]; then
  if ollama_reachable "$OLLAMA_BASE_URL"; then
    if ! try_ollama_model_probes "$OLLAMA_BASE_URL"; then
      if [[ "${DEMENTIA_OLLAMA_MODE,,}" == "isolated" && "$OLLAMA_RESTART_ON_PROBE_FAIL" == "1" ]]; then
        echo "[ollama] generate probe failed on isolated port ${DEMENTIA_OLLAMA_PORT}; using port-safe model/GPU/CPU failover and preserving 11434"
        try_isolated_ollama_gpu_failover "$OLLAMA_BASE_URL" || exit 3
      else
        exit 3
      fi
    fi
  fi
fi

# Console progress bars. Set PROGRESS_BARS=false to disable tqdm-style progress.
export PROGRESS_BARS="${PROGRESS_BARS:-true}"

# Python runtime settings.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Runtime audit printout
# -----------------------------------------------------------------------------
echo "[dementia_progression] project_root=$PROJECT_ROOT"
echo "[dementia_progression] project_name=$DEMENTIA_PROJECT_NAME"
echo "[dementia_progression] config=$CONFIG_PATH"
echo "[dementia_progression] input_csv=${DEMENTIA_INPUT_CSV:-<config>}"
echo "[dementia_progression] output_dir=$RUN_OUTPUT_DIR"
echo "[dementia_progression] append_datetime=$DEMENTIA_APPEND_DATETIME"
echo "[dementia_progression] run_timestamp=${DEMENTIA_RUN_TIMESTAMP:-<resume/no_timestamp>}"
echo "[dementia_progression] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[dementia_progression] DEMENTIA_GPU_CANDIDATES=$DEMENTIA_GPU_CANDIDATES"
echo "[dementia_progression] DEMENTIA_EXCLUDED_GPUS=$DEMENTIA_EXCLUDED_GPUS"
echo "[dementia_progression] OLLAMA_FAILOVER_GPU_ENABLED=$OLLAMA_FAILOVER_GPU_ENABLED"
echo "[dementia_progression] LLM_MODEL_PROFILE=${LLM_MODEL_PROFILE:-<yaml>}"
echo "[dementia_progression] LLM_INPUT_PRETRAINED_NORMALIZATION_BACKEND=${LLM_INPUT_PRETRAINED_NORMALIZATION_BACKEND:-<yaml>}"
echo "[dementia_progression] LLM_INPUT_PRETRAINED_NORMALIZATION_REQUIRED=${LLM_INPUT_PRETRAINED_NORMALIZATION_REQUIRED:-<yaml>}"
echo "[dementia_progression] OLLAMA_MODEL_FAILOVER_ENABLED=$OLLAMA_MODEL_FAILOVER_ENABLED"
echo "[dementia_progression] OLLAMA_MODEL_CANDIDATES=$OLLAMA_MODEL_CANDIDATES"
echo "[dementia_progression] OLLAMA_CPU_FALLBACK_ENABLED=$OLLAMA_CPU_FALLBACK_ENABLED"
echo "[dementia_progression] OLLAMA_CPU_PROBE_TIMEOUT_SECONDS=$OLLAMA_CPU_PROBE_TIMEOUT_SECONDS"
echo "[dementia_progression] OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH"
echo "[dementia_progression] OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
echo "[dementia_progression] OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"
echo "[dementia_progression] llm_config=$PROJECT_ROOT/configs/_base_llm_medication_state.yaml"
echo "[dementia_progression] DEMENTIA_OLLAMA_MODE=$DEMENTIA_OLLAMA_MODE"
echo "[dementia_progression] OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-<yaml>}"
echo "[dementia_progression] OLLAMA_RESTART_ON_PROBE_FAIL=$OLLAMA_RESTART_ON_PROBE_FAIL"
echo "[dementia_progression] OLLAMA_PROBE_FORMAT_MODE=$OLLAMA_PROBE_FORMAT_MODE"
echo "[dementia_progression] OLLAMA_STOP_TIMEOUT_SECONDS=$OLLAMA_STOP_TIMEOUT_SECONDS"
echo "[dementia_progression] LLM_PROVIDER_OVERRIDE=${LLM_PROVIDER:-<yaml>}"
echo "[dementia_progression] OLLAMA_MODEL_OVERRIDE=${OLLAMA_MODEL:-<yaml>}"
echo "[dementia_progression] LLM_MAX_UNIQUE_TEXTS_OVERRIDE=${LLM_MAX_UNIQUE_TEXTS:-<yaml>}"
echo "[dementia_progression] LLM_REQUIRE_OLLAMA_OVERRIDE=${LLM_REQUIRE_OLLAMA:-<yaml>}"
echo "[dementia_progression] LLM_REQUIRE_MIN_SUCCESSFUL_PARSES_OVERRIDE=${LLM_REQUIRE_MIN_SUCCESSFUL_PARSES:-<yaml>}"
echo "[dementia_progression] LLM_NUM_PREDICT_OVERRIDE=${LLM_NUM_PREDICT:-<yaml>}"
echo "[dementia_progression] LLM_NUM_CTX_OVERRIDE=${LLM_NUM_CTX:-<yaml>}"
echo "[dementia_progression] LLM_REQUEST_TIMEOUT_S_OVERRIDE=${LLM_REQUEST_TIMEOUT_S:-<yaml>}"
echo "[dementia_progression] LLM_CONNECT_TIMEOUT_S_OVERRIDE=${LLM_CONNECT_TIMEOUT_S:-<yaml>}"
echo "[dementia_progression] LLM_REQUEST_BACKEND_OVERRIDE=${LLM_REQUEST_BACKEND:-<yaml>}"
echo "[dementia_progression] LLM_INITIAL_FAILURE_ABORT_COUNT_OVERRIDE=${LLM_INITIAL_FAILURE_ABORT_COUNT:-<yaml>}"
echo "[dementia_progression] LLM_STRUCTURED_JSON_SCHEMA_ENABLED_OVERRIDE=${LLM_STRUCTURED_JSON_SCHEMA_ENABLED:-<yaml>}"
echo "[dementia_progression] LLM_OLLAMA_FORMAT_MODE_OVERRIDE=${LLM_OLLAMA_FORMAT_MODE:-<yaml>}"
echo "[dementia_progression] LLM_WARMUP_ENABLED_OVERRIDE=${LLM_WARMUP_ENABLED:-<yaml>}"
echo "[dementia_progression] LLM_PERSISTENT_CACHE_PATH_OVERRIDE=${LLM_PERSISTENT_CACHE_PATH:-<yaml>}"
echo "[dementia_progression] PRETRAINED_PREFLIGHT_ENABLED=$PRETRAINED_PREFLIGHT_ENABLED"
echo "[dementia_progression] PRETRAINED_AUTO_INSTALL_DEPS=$PRETRAINED_AUTO_INSTALL_DEPS"
echo "[dementia_progression] PRETRAINED_MODEL_LOAD_PREFLIGHT=$PRETRAINED_MODEL_LOAD_PREFLIGHT"
echo "[dementia_progression] PROGRESS_BARS=$PROGRESS_BARS"

exec bash "$PROJECT_ROOT/src/run_pipeline.sh"
