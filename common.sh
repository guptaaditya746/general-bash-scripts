#!/usr/bin/env bash

set -euo pipefail

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${COMMON_SCRIPT_DIR}}"
REPO_DIR="${REPO_DIR:-$(cd "${COMMON_SCRIPT_DIR}/.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_DIR}/config/vllm.env}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/run}"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-${RUN_DIR}/vllm_runtime.env}"
RUNTIME_JSON_FILE="${RUNTIME_JSON_FILE:-${RUN_DIR}/vllm_runtime.json}"
JOB_ID_FILE="${JOB_ID_FILE:-${RUN_DIR}/vllm_job_id}"

mkdir -p "${RUN_DIR}"

log() { echo "[$(date '+%Y-%m-%d %T')] $*"; }

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  local command_name="$1"
  if ! command_exists "${command_name}"; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
}

load_vllm_config() {
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Missing config file: ${CONFIG_FILE}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_FILE}"
  set +a
}

resolve_python_bin() {
  if [[ -n "${VLLM_PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${VLLM_PYTHON_BIN}"
    return 0
  fi
  if command_exists python3; then
    printf '%s\n' "python3"
    return 0
  fi
  if command_exists python; then
    printf '%s\n' "python"
    return 0
  fi
  return 1
}

render_command() {
  local -a rendered=()
  local arg
  for arg in "$@"; do
    rendered+=("$(printf '%q' "${arg}")")
  done
  (
    IFS=' '
    printf '%s' "${rendered[*]}"
  )
}

append_optional_arg() {
  local -n target_array="$1"
  local flag="$2"
  local value="${3:-}"
  if [[ -n "${value}" ]]; then
    target_array+=("${flag}" "${value}")
  fi
}

append_extra_args_from_string() {
  local -n target_array="$1"
  local extra_args="${2:-}"
  local python_bin

  if [[ -z "${extra_args}" ]]; then
    return 0
  fi

  python_bin="$(resolve_python_bin)" || {
    echo "Cannot parse extra args without python3 or python." >&2
    exit 1
  }

  while IFS= read -r -d '' arg; do
    target_array+=("${arg}")
  done < <(
    EXTRA_ARGS="${extra_args}" "${python_bin}" -c 'import os, shlex, sys
for value in shlex.split(os.environ.get("EXTRA_ARGS", "")):
    sys.stdout.buffer.write(value.encode("utf-8"))
    sys.stdout.buffer.write(b"\0")'
  )
}

declare -a RESERVED_PORTS=()
LAUNCH_PID=""
LAUNCH_PGID=""
VLLM_READY_MODEL_ID=""
VLLM_SERVE_HELP_CACHE=""

port_is_reserved() {
  local port="$1"
  local reserved_port
  for reserved_port in "${RESERVED_PORTS[@]}"; do
    if [[ "${reserved_port}" == "${port}" ]]; then
      return 0
    fi
  done
  return 1
}

reserve_port() {
  local port="$1"
  if ! port_is_reserved "${port}"; then
    RESERVED_PORTS+=("${port}")
  fi
}

port_is_listening() {
  local port="$1"
  if command_exists ss; then
    ss -ltnH "sport = :${port}" 2>/dev/null | grep -q '.'
    return
  fi
  if command_exists lsof; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return
  fi
  if command_exists netstat; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
    return
  fi
  return 1
}

port_is_available() {
  local port="$1"
  ! port_is_reserved "${port}" && ! port_is_listening "${port}"
}

print_listener_diagnostics() {
  if command_exists ss; then
    log "TCP listeners (ss -ltnp):"
    ss -ltnp 2>/dev/null || ss -ltn 2>/dev/null || true
    return
  fi
  if command_exists netstat; then
    log "TCP listeners (netstat -ltnp):"
    netstat -ltnp 2>/dev/null || netstat -ltn 2>/dev/null || true
    return
  fi
  log "Port diagnostics unavailable: neither ss nor netstat is installed."
}

resolve_service_port() {
  local service_name="$1"
  local requested_port="$2"
  local explicit_port_var="${3:-}"
  local search_limit="${VLLM_PORT_SEARCH_LIMIT:-25}"
  local candidate_port
  local search_count=0

  if [[ -n "${explicit_port_var}" ]]; then
    if port_is_available "${requested_port}"; then
      reserve_port "${requested_port}"
      printf '%s\n' "${requested_port}"
      return 0
    fi
    log "Configured ${service_name} port ${requested_port} from ${explicit_port_var} is already occupied."
    print_listener_diagnostics
    return 1
  fi

  candidate_port="${requested_port}"
  while ((search_count <= search_limit)); do
    if port_is_available "${candidate_port}"; then
      if [[ "${candidate_port}" != "${requested_port}" ]]; then
        log "Computed ${service_name} port ${requested_port} is busy; using ${candidate_port} instead."
      fi
      reserve_port "${candidate_port}"
      printf '%s\n' "${candidate_port}"
      return 0
    fi
    candidate_port=$((candidate_port + 1))
    search_count=$((search_count + 1))
  done

  log "Could not find a free ${service_name} port starting from ${requested_port} within ${search_limit} retries."
  print_listener_diagnostics
  return 1
}

launch_logged_command() {
  local log_file="$1"
  shift
  local -a command=("$@")

  if command_exists setsid; then
    setsid "${command[@]}" >>"${log_file}" 2>&1 &
    LAUNCH_PID=$!
    LAUNCH_PGID="${LAUNCH_PID}"
    return 0
  fi

  "${command[@]}" >>"${log_file}" 2>&1 &
  LAUNCH_PID=$!
  LAUNCH_PGID=""
}

stop_process_group_or_pid() {
  local pid="${1:-}"
  local pgid="${2:-}"
  local label="${3:-process}"
  local signal_name
  local _attempt
  local pid_alive=0
  local group_alive=0

  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    pid_alive=1
  fi
  if [[ -n "${pgid}" ]] && kill -0 -- "-${pgid}" 2>/dev/null; then
    group_alive=1
  fi
  if ((pid_alive == 0 && group_alive == 0)); then
    return 0
  fi

  for signal_name in TERM KILL; do
    if ((group_alive == 1)); then
      kill "-${signal_name}" -- "-${pgid}" 2>/dev/null || true
    fi
    if ((pid_alive == 1)); then
      kill "-${signal_name}" "${pid}" 2>/dev/null || true
    fi
    for _attempt in 1 2 3; do
      pid_alive=0
      group_alive=0
      if kill -0 "${pid}" 2>/dev/null; then
        pid_alive=1
      fi
      if [[ -n "${pgid}" ]] && kill -0 -- "-${pgid}" 2>/dev/null; then
        group_alive=1
      fi
      if ((pid_alive == 0 && group_alive == 0)); then
        return 0
      fi
      sleep 1
    done
  done

  log "Failed to stop ${label} cleanly (pid=${pid}, pgid=${pgid:-n/a})."
  return 0
}

http_get() {
  local url="$1"
  curl -fsS --connect-timeout 2 --max-time 15 "${url}"
}

extract_first_model_id_from_json() {
  local python_bin
  python_bin="$(resolve_python_bin)" || {
    echo "Cannot inspect /v1/models without python3 or python." >&2
    return 1
  }
  "${python_bin}" -c 'import json, sys
payload = json.load(sys.stdin)
for entry in payload.get("data") or []:
    model_id = entry.get("id")
    if model_id:
        sys.stdout.write(str(model_id))
        raise SystemExit(0)
raise SystemExit(1)'
}

fetch_first_model_id() {
  local port="$1"
  http_get "http://127.0.0.1:${port}/v1/models" | extract_first_model_id_from_json
}

print_nvidia_smi() {
  if command_exists nvidia-smi; then
    log "nvidia-smi:"
    nvidia-smi || true
    return
  fi
  log "nvidia-smi not available."
}

print_vllm_failure_diagnostics() {
  local service_name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local reason="$5"
  shift 5
  local -a command=("$@")
  local node_name

  node_name="${VLLM_NODE:-$(hostname -f 2>/dev/null || hostname)}"
  log "Readiness failure for ${service_name}: ${reason}"
  log "Node: ${node_name}"
  log "Port: ${port}"
  if [[ -n "${pid}" ]]; then
    log "PID: ${pid}"
  fi
  if ((${#command[@]} > 0)); then
    log "Command: $(render_command "${command[@]}")"
  fi
  if [[ -f "${log_file}" ]]; then
    log "Last 200 lines of ${log_file}:"
    tail -n 200 "${log_file}" || true
  else
    log "Log file not found: ${log_file}"
  fi
  print_nvidia_smi
  print_listener_diagnostics
}

wait_for_vllm_ready() {
  local service_name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local timeout_s="$5"
  shift 5
  local -a command=("$@")
  local poll_interval_s="${VLLM_HEALTH_POLL_INTERVAL_S:-5}"
  local start_time="${SECONDS}"
  local deadline=$((start_time + timeout_s))
  local next_progress_time=$((start_time + 60))
  local model_id

  VLLM_READY_MODEL_ID=""
  log "Waiting for ${service_name} readiness on 127.0.0.1:${port} (timeout=${timeout_s}s, poll=${poll_interval_s}s)..."

  while ((SECONDS < deadline)); do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      print_vllm_failure_diagnostics "${service_name}" "${port}" "${pid}" "${log_file}" "process exited before readiness" "${command[@]}"
      return 1
    fi

    if http_get "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      if model_id="$(fetch_first_model_id "${port}" 2>/dev/null)"; then
        VLLM_READY_MODEL_ID="${model_id}"
        log "${service_name} ready on 127.0.0.1:${port} (model_id=${model_id})."
        return 0
      fi
    fi

    if ((SECONDS >= next_progress_time)); then
      log "Still waiting for ${service_name} on 127.0.0.1:${port} ($((SECONDS - start_time))s/${timeout_s}s)..."
      next_progress_time=$((next_progress_time + 60))
    fi

    sleep "${poll_interval_s}"
  done

  print_vllm_failure_diagnostics "${service_name}" "${port}" "${pid}" "${log_file}" "timeout after ${timeout_s}s" "${command[@]}"
  return 1
}

build_json_payload() {
  local python_expression="$1"
  local python_bin
  python_bin="$(resolve_python_bin)" || {
    echo "Cannot build JSON payload without python3 or python." >&2
    return 1
  }
  "${python_bin}" -c "${python_expression}"
}

run_chat_smoke_test() {
  local port="$1"
  local model_id="$2"
  local response
  local python_bin

  if [[ "${VLLM_RUN_SMOKE_TESTS:-1}" != "1" ]]; then
    log "Skipping chat smoke test (VLLM_RUN_SMOKE_TESTS=${VLLM_RUN_SMOKE_TESTS:-0})."
    return 0
  fi

  python_bin="$(resolve_python_bin)" || {
    echo "Cannot run chat smoke test without python3 or python." >&2
    return 1
  }

  response="$(
    SMOKE_MODEL_ID="${model_id}" "${python_bin}" -c 'import json, os, sys
payload = {
    "model": os.environ["SMOKE_MODEL_ID"],
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 8,
    "temperature": 0,
}
json.dump(payload, sys.stdout)' |
      curl -fsS --connect-timeout 2 --max-time 30 \
        -H 'Content-Type: application/json' \
        --data-binary @- \
        "http://127.0.0.1:${port}/v1/chat/completions"
  )"

  printf '%s' "${response}" | "${python_bin}" -c 'import json, sys
payload = json.load(sys.stdin)
choices = payload.get("choices") or []
if not choices:
    raise SystemExit("chat smoke test returned no choices")
message = choices[0].get("message") or {}
content = message.get("content")
if content in (None, ""):
    raise SystemExit("chat smoke test returned an empty message")
print(content if isinstance(content, str) else "ok")' >/dev/null

  log "Chat smoke test passed on 127.0.0.1:${port}."
}

run_embedding_smoke_test() {
  local port="$1"
  local model_id="$2"
  local response
  local python_bin

  if [[ "${VLLM_RUN_SMOKE_TESTS:-1}" != "1" ]]; then
    log "Skipping embeddings smoke test (VLLM_RUN_SMOKE_TESTS=${VLLM_RUN_SMOKE_TESTS:-0})."
    return 0
  fi

  python_bin="$(resolve_python_bin)" || {
    echo "Cannot run embeddings smoke test without python3 or python." >&2
    return 1
  }

  response="$(
    SMOKE_MODEL_ID="${model_id}" "${python_bin}" -c 'import json, os, sys
payload = {
    "model": os.environ["SMOKE_MODEL_ID"],
    "input": ["ping"],
    "encoding_format": "float",
}
json.dump(payload, sys.stdout)' |
      curl -fsS --connect-timeout 2 --max-time 30 \
        -H 'Content-Type: application/json' \
        --data-binary @- \
        "http://127.0.0.1:${port}/v1/embeddings"
  )"

  printf '%s' "${response}" | "${python_bin}" -c 'import json, sys
payload = json.load(sys.stdin)
rows = payload.get("data") or []
if not rows:
    raise SystemExit("embeddings smoke test returned no data rows")
embedding = rows[0].get("embedding")
if not isinstance(embedding, list) or not embedding:
    raise SystemExit("embeddings smoke test returned an empty embedding vector")
print(len(embedding))' >/dev/null

  log "Embeddings smoke test passed on 127.0.0.1:${port}."
}

write_export_file() {
  local file_path="$1"
  shift

  : >"${file_path}"
  while (($# >= 2)); do
    printf 'export %s=%q\n' "$1" "$2" >>"${file_path}"
    shift 2
  done
}

print_debug_environment() {
  log "Relevant environment (sanitized):"
  env |
    grep -E '^(SLURM_|CUDA_|GLOO_|NCCL_|TORCH_|HF_|HUGGINGFACE_|VLLM_|MODEL_|PORT_|MAX_MODEL_LEN|EMBED_MAX_MODEL_LEN|GPU_MEM_UTIL_|CHAT_|EMBED_|START_SERVERS_SEQUENTIALLY|PREFETCH_MODELS)=' |
    grep -Ev '(API_KEY|TOKEN|PASSWORD|SECRET)' |
    sort ||
    true
}

load_vllm_serve_help() {
  if [[ -z "${VLLM_SERVE_HELP_CACHE}" ]]; then
    VLLM_SERVE_HELP_CACHE="$(vllm serve --help 2>&1 || true)"
  fi
  printf '%s' "${VLLM_SERVE_HELP_CACHE}"
}

vllm_serve_supports() {
  local flag="$1"
  load_vllm_serve_help | grep -F -- "${flag}" >/dev/null 2>&1
}
