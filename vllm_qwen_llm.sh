#!/usr/bin/env bash
#SBATCH --job-name=vllm-qwen-llm
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"

source "${REPO_DIR}/common.sh"
CONFIG_FILE="${CONFIG_FILE:-${REPO_DIR}/vllm.env}"
load_vllm_config

WORKDIR="${WORKDIR:-${REPO_DIR}}"
VENV_DIR="${VENV_DIR:-${WORKDIR}/.venv}"
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:3g.20gb:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "${SLURM_JOB_ID:-unknown}" >"${JOB_ID_FILE}"

if [[ ! -d "${WORKDIR}" ]]; then
  log "Missing WORKDIR: ${WORKDIR}"
  exit 1
fi
set +u
cd "${WORKDIR}"
# shellcheck disable=SC1090
source "${CONDA_SH_PATH:?Set CONDA_SH_PATH to your conda.sh path}"
conda init
conda activate vllm_env
require_command curl
require_command python
require_command vllm
set +u
NODE="$(hostname -f 2>/dev/null || hostname)"
export VLLM_NODE="${NODE}"

log "Node: ${NODE}"
log "Python: $(command -v python)"
log "vLLM: $(command -v vllm)"
vllm --version
python -c 'import vllm; print("vllm:", vllm.__version__)'
python -c 'import torch; print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available(), "cuda:", torch.version.cuda)'
nvidia-smi || true

JOB_NUM="${SLURM_JOB_ID:-99999}"
PORT_BASE="${PORT_BASE:-40000}"
PORT_SPAN="${PORT_SPAN:-20000}"
PORT_CHAT_OFFSET="${PORT_CHAT_OFFSET:-4}"
PORT_CHAT_EXPLICIT_SOURCE=""
BASE=$((PORT_BASE + (JOB_NUM % PORT_SPAN)))
REQUESTED_PORT_CHAT="${PORT_CHAT:-$((BASE + PORT_CHAT_OFFSET))}"
if [[ -n "${PORT_CHAT:-}" ]]; then
  PORT_CHAT_EXPLICIT_SOURCE="PORT_CHAT"
fi
PORT_CHAT="$(resolve_service_port "chat" "${REQUESTED_PORT_CHAT}" "${PORT_CHAT_EXPLICIT_SOURCE}")"
log "Port: chat=${PORT_CHAT}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"
if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
  log "Ignoring TRANSFORMERS_CACHE in favor of HF_HOME=${HF_HOME}."
  unset TRANSFORMERS_CACHE
fi
log "HF_HOME: ${HF_HOME}"

# LLM-only defaults for one 20GB Ampere GPU.
export MODEL_CHAT="${MODEL_CHAT:-Qwen/Qwen3.5-9B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export GPU_MEM_UTIL_CHAT="${GPU_MEM_UTIL_CHAT:-0.85}"
export CHAT_DTYPE="${CHAT_DTYPE:-auto}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_HEALTH_POLL_INTERVAL_S="${VLLM_HEALTH_POLL_INTERVAL_S:-5}"
export VLLM_RUN_SMOKE_TESTS="${VLLM_RUN_SMOKE_TESTS:-1}"
export VLLM_DEBUG="${VLLM_DEBUG:-0}"
export PREFETCH_MODELS="${PREFETCH_MODELS:-1}"
export VLLM_PORT_SEARCH_LIMIT="${VLLM_PORT_SEARCH_LIMIT:-25}"
export CHAT_MAX_NUM_SEQS="${CHAT_MAX_NUM_SEQS:-}"
export CHAT_MAX_NUM_BATCHED_TOKENS="${CHAT_MAX_NUM_BATCHED_TOKENS:-}"
export CHAT_EXTRA_ARGS="${CHAT_EXTRA_ARGS:---quantization bitsandbytes --load-format bitsandbytes}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

log "Chat model : ${MODEL_CHAT}"
log "Max context: ${MAX_MODEL_LEN}"
log "GPU memory utilization: ${GPU_MEM_UTIL_CHAT}"
log "Engine ready timeout: ${VLLM_ENGINE_READY_TIMEOUT_S}s"
log "Health poll interval: ${VLLM_HEALTH_POLL_INTERVAL_S}s"
log "Smoke tests: ${VLLM_RUN_SMOKE_TESTS}"
log "Chat extra args: ${CHAT_EXTRA_ARGS}"

if [[ "${VLLM_DEBUG}" == "1" ]]; then
  print_debug_environment
  log "vllm serve --help (first 120 lines):"
  load_vllm_serve_help | sed -n '1,120p' || true
fi

prefetch_model() {
  local label="$1"
  local model_name="$2"

  if [[ "${PREFETCH_MODELS}" != "1" ]]; then
    log "Skipping ${label} model prefetch (PREFETCH_MODELS=${PREFETCH_MODELS})."
    return 0
  fi

  log "Prefetching ${label} model ${model_name}..."
  MODEL_TO_PREFETCH="${model_name}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(repo_id=os.environ["MODEL_TO_PREFETCH"])
print(f"Prefetch done for {os.environ['MODEL_TO_PREFETCH']}.")
PY
}

debug_set_xtrace() {
  if [[ "${VLLM_DEBUG}" == "1" ]]; then
    set -x
  fi
}

debug_clear_xtrace() {
  if [[ "${VLLM_DEBUG}" == "1" ]]; then
    { set +x; } 2>/dev/null
  fi
}

CHAT_LOG="${RUN_DIR}/vllm_chat_${SLURM_JOB_ID:-manual}.log"

CHAT_CMD=(
  env
  -u VLLM_NODE
  -u VLLM_CHAT_PORT
  CUDA_VISIBLE_DEVICES=0
  vllm
  serve
  "${MODEL_CHAT}"
  --host
  0.0.0.0
  --port
  "${PORT_CHAT}"
  --dtype
  "${CHAT_DTYPE}"
  --max-model-len
  "${MAX_MODEL_LEN}"
  --gpu-memory-utilization
  "${GPU_MEM_UTIL_CHAT}"
)
append_optional_arg CHAT_CMD --max-num-seqs "${CHAT_MAX_NUM_SEQS}"
append_optional_arg CHAT_CMD --max-num-batched-tokens "${CHAT_MAX_NUM_BATCHED_TOKENS}"
append_extra_args_from_string CHAT_CMD "${CHAT_EXTRA_ARGS}"

PID_CHAT=""
PGID_CHAT=""
CHAT_MODEL_ID=""

cleanup() {
  log "Shutting down vLLM..."
  stop_process_group_or_pid "${PID_CHAT:-}" "${PGID_CHAT:-}" "chat vLLM"
  if [[ -n "${PID_CHAT:-}" ]]; then
    wait "${PID_CHAT}" 2>/dev/null || true
  fi
  log "Done."
}
trap cleanup EXIT SIGTERM SIGINT

start_chat_server() {
  log "Starting vLLM chat on GPU0 log=${CHAT_LOG}"
  log "Chat command: $(render_command "${CHAT_CMD[@]}")"
  debug_set_xtrace
  launch_logged_command "${CHAT_LOG}" "${CHAT_CMD[@]}"
  debug_clear_xtrace
  PID_CHAT="${LAUNCH_PID}"
  PGID_CHAT="${LAUNCH_PGID}"
  log "Chat PID=${PID_CHAT}${PGID_CHAT:+ PGID=${PGID_CHAT}}"
}

debug_set_xtrace
prefetch_model "chat" "${MODEL_CHAT}"
debug_clear_xtrace

start_chat_server

wait_for_vllm_ready "chat" "${PORT_CHAT}" "${PID_CHAT}" "${CHAT_LOG}" "${VLLM_ENGINE_READY_TIMEOUT_S}" "${CHAT_CMD[@]}"
CHAT_MODEL_ID="${VLLM_READY_MODEL_ID}"
run_chat_smoke_test "${PORT_CHAT}" "${CHAT_MODEL_ID}"

CHAT_API_BASE_LOCAL="http://127.0.0.1:${PORT_CHAT}/v1"
CHAT_API_BASE_NODE="http://${NODE}:${PORT_CHAT}/v1"

write_export_file "${RUNTIME_ENV_FILE}" \
  SLURM_JOB_ID "${SLURM_JOB_ID:-unknown}" \
  VLLM_NODE "${NODE}" \
  VLLM_CHAT_PORT "${PORT_CHAT}" \
  VLLM_CHAT_MODEL_ID "${CHAT_MODEL_ID}" \
  VLLM_CHAT_API_BASE_LOCAL "${CHAT_API_BASE_LOCAL}" \
  VLLM_CHAT_API_BASE_NODE "${CHAT_API_BASE_NODE}" \
  VLLM_CHAT_LOG "${CHAT_LOG}" \
  LITELLM_MODEL "openai/${CHAT_MODEL_ID}" \
  LITELLM_API_BASE "${CHAT_API_BASE_LOCAL}" \
  LITELLM_API_KEY "dummy"

SLURM_RUNTIME_JOB_ID="${SLURM_JOB_ID:-unknown}" \
  VLLM_RUNTIME_NODE="${NODE}" \
  VLLM_RUNTIME_CHAT_MODEL_ID="${CHAT_MODEL_ID}" \
  VLLM_RUNTIME_CHAT_API_BASE_LOCAL="${CHAT_API_BASE_LOCAL}" \
  VLLM_RUNTIME_CHAT_API_BASE_NODE="${CHAT_API_BASE_NODE}" \
  VLLM_RUNTIME_CHAT_PORT="${PORT_CHAT}" \
  VLLM_RUNTIME_CHAT_LOG="${CHAT_LOG}" \
  python - <<'PY' >"${RUNTIME_JSON_FILE}"
import json
import os
import sys

payload = {
    "slurm_job_id": os.environ["SLURM_RUNTIME_JOB_ID"],
    "node": os.environ["VLLM_RUNTIME_NODE"],
    "chat_model": f"openai/{os.environ['VLLM_RUNTIME_CHAT_MODEL_ID']}",
    "chat_api_base_local": os.environ["VLLM_RUNTIME_CHAT_API_BASE_LOCAL"],
    "chat_api_base_node": os.environ["VLLM_RUNTIME_CHAT_API_BASE_NODE"],
    "chat_port": os.environ["VLLM_RUNTIME_CHAT_PORT"],
    "chat_log": os.environ["VLLM_RUNTIME_CHAT_LOG"],
    "api_key": "dummy",
}
json.dump(payload, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY

log "============================================"
log "Chat API local : ${CHAT_API_BASE_LOCAL}"
log "Chat API node  : ${CHAT_API_BASE_NODE}"
log ""
log "LiteLLM env for this repo:"
log "  export LITELLM_MODEL=openai/${CHAT_MODEL_ID}"
log "  export LITELLM_API_BASE=${CHAT_API_BASE_LOCAL}"
log "  export LITELLM_API_KEY=dummy"
log ""
log "Runtime env saved to: ${RUNTIME_ENV_FILE}"
log "Runtime json saved to: ${RUNTIME_JSON_FILE}"
log "vLLM log:"
log "  ${CHAT_LOG}"
log ""
log "Tunnel from laptop:"
log "  ssh -N \\"
log "    -L ${PORT_CHAT}:${NODE}:${PORT_CHAT} \\"
log "    gupt_ad@LOGIN_NODE"
log "============================================"

wait "${PID_CHAT}"
