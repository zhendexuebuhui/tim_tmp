#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
START_TIMEOUT="${START_TIMEOUT:-1800}"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"
LOG_RETENTION="${LOG_RETENTION:-10}"
NPU_DEVICE_IDS="${NPU_DEVICE_IDS:-2 3 5 7}"

RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/runtime/qwen3.6-27b}"
PID_FILE="${RUNTIME_DIR}/server.pid"
PROFILE_FILE="${RUNTIME_DIR}/profile"
STARTED_AT_FILE="${RUNTIME_DIR}/started_at"
CURRENT_LOG_FILE="${RUNTIME_DIR}/current_log"
STATE_FILE="${RUNTIME_DIR}/service_state"

BASE_URL="http://${HOST}:${PORT}"
STARTED_BY_THIS_COMMAND=0

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

error() {
  printf '[ERROR] %s\n' "$*" >&2
}

die() {
  error "$*"
  exit 1
}

usage() {
  cat <<'EOF'
Manage the Qwen3.6-27B vLLM-Ascend service inside vllm-ascend-env.

Usage:
  manage_qwen3_6_27b.sh check
  manage_qwen3_6_27b.sh start [optimized|safe]
  manage_qwen3_6_27b.sh stop
  manage_qwen3_6_27b.sh restart [optimized|safe]
  manage_qwen3_6_27b.sh status
  manage_qwen3_6_27b.sh logs [-f]
  manage_qwen3_6_27b.sh test
  manage_qwen3_6_27b.sh help | --help | -h

Profiles:
  optimized  Default. Enables MTP, FULL_DECODE_ONLY graph mode, CPU binding,
             async scheduling, and prefix caching.
  safe       Disables the optional performance features above for diagnosis.

Common environment overrides:
  MODEL_PATH, SERVED_MODEL_NAME, HOST, PORT, TP_SIZE, DP_SIZE, DTYPE,
  MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
  GPU_MEMORY_UTILIZATION, OFFLINE_MODE, START_TIMEOUT, STOP_TIMEOUT,
  LOG_RETENTION, NPU_DEVICE_IDS, RUNTIME_DIR

Examples:
  ./scripts/manage_qwen3_6_27b.sh check
  ./scripts/manage_qwen3_6_27b.sh start
  ./scripts/manage_qwen3_6_27b.sh start safe
  ./scripts/manage_qwen3_6_27b.sh logs -f
  MAX_MODEL_LEN=131072 ./scripts/manage_qwen3_6_27b.sh restart optimized
  OFFLINE_MODE=0 ./scripts/manage_qwen3_6_27b.sh start
EOF
}

ensure_runtime_dir() {
  mkdir -p -- "${RUNTIME_DIR}"
  [[ -w "${RUNTIME_DIR}" ]] || die "Runtime directory is not writable: ${RUNTIME_DIR}"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_configuration() {
  is_positive_integer "${PORT}" || die "PORT must be a positive integer: ${PORT}"
  (( PORT <= 65535 )) || die "PORT must be <= 65535: ${PORT}"
  is_positive_integer "${TP_SIZE}" || die "TP_SIZE must be a positive integer: ${TP_SIZE}"
  is_positive_integer "${DP_SIZE}" || die "DP_SIZE must be a positive integer: ${DP_SIZE}"
  is_positive_integer "${MAX_MODEL_LEN}" || die "MAX_MODEL_LEN must be a positive integer"
  is_positive_integer "${MAX_NUM_SEQS}" || die "MAX_NUM_SEQS must be a positive integer"
  is_positive_integer "${MAX_NUM_BATCHED_TOKENS}" || die "MAX_NUM_BATCHED_TOKENS must be a positive integer"
  is_positive_integer "${START_TIMEOUT}" || die "START_TIMEOUT must be a positive integer"
  is_positive_integer "${STOP_TIMEOUT}" || die "STOP_TIMEOUT must be a positive integer"
  is_positive_integer "${LOG_RETENTION}" || die "LOG_RETENTION must be a positive integer"
  [[ "${OFFLINE_MODE}" == "0" || "${OFFLINE_MODE}" == "1" ]] || die "OFFLINE_MODE must be 0 or 1"
  [[ "${HOST}" == "127.0.0.1" || "${HOST}" == "localhost" ]] || warn "HOST=${HOST} exposes behavior beyond the recommended loopback-only default"
}

validate_profile() {
  case "$1" in
    optimized|safe) ;;
    *) die "Unknown profile '$1'. Expected optimized or safe." ;;
  esac
}

proc_start_time() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat"
}

read_managed_process() {
  local pid recorded_start pgid actual_start cmdline

  [[ -r "${PID_FILE}" ]] || return 1
  read -r pid recorded_start pgid < "${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${recorded_start}" =~ ^[0-9]+$ && "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  actual_start="$(proc_start_time "${pid}" 2>/dev/null || true)"
  [[ -n "${actual_start}" && "${actual_start}" == "${recorded_start}" ]] || return 1

  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *vllm* && "${cmdline}" == *serve* ]] || return 1
  printf '%s %s\n' "${pid}" "${pgid}"
}

remove_process_metadata() {
  rm -f -- "${PID_FILE}" "${STARTED_AT_FILE}"
}

cleanup_stale_pid_file() {
  if [[ -e "${PID_FILE}" ]] && ! read_managed_process >/dev/null 2>&1; then
    warn "Removing stale PID metadata: ${PID_FILE}"
    remove_process_metadata
  fi
}

api_ready() {
  curl -fsS --max-time 5 "${BASE_URL}/v1/models" >/dev/null 2>&1
}

load_active_endpoint() {
  local -a state=()
  [[ -r "${STATE_FILE}" ]] || return 0
  mapfile -t state < "${STATE_FILE}"
  (( ${#state[@]} >= 3 )) || return 0
  [[ -n "${state[0]}" && "${state[1]}" =~ ^[1-9][0-9]*$ && ${state[1]} -le 65535 && -n "${state[2]}" ]] || return 0
  HOST="${state[0]}"
  PORT="${state[1]}"
  SERVED_MODEL_NAME="${state[2]}"
  BASE_URL="http://${HOST}:${PORT}"
}

port_is_open() {
  (exec 3<>"/dev/tcp/${HOST}/${PORT}") >/dev/null 2>&1
}

show_port_owner() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port "$" {print}' >&2 || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >&2 || true
  fi
}

current_log_path() {
  local path
  if [[ -r "${CURRENT_LOG_FILE}" ]]; then
    path="$(<"${CURRENT_LOG_FILE}")"
    [[ -f "${path}" ]] && printf '%s\n' "${path}" && return 0
  fi

  path="$(find "${RUNTIME_DIR}" -maxdepth 1 -type f -name 'serve-*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  [[ -n "${path}" ]] && printf '%s\n' "${path}"
}

cleanup_old_logs() {
  local index=0 entry old_log
  while IFS= read -r entry; do
    [[ -n "${entry}" ]] || continue
    index=$((index + 1))
    if (( index > LOG_RETENTION )); then
      old_log="${entry#* }"
      [[ "${old_log}" == "${RUNTIME_DIR}"/serve-*.log ]] || continue
      rm -f -- "${old_log}"
    fi
  done < <(find "${RUNTIME_DIR}" -maxdepth 1 -type f -name 'serve-*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr)
}

check_model_files() {
  [[ -d "${MODEL_PATH}" ]] || die "Model directory not found: ${MODEL_PATH}"
  [[ -r "${MODEL_PATH}/config.json" ]] || die "Missing model config: ${MODEL_PATH}/config.json"
  [[ -r "${MODEL_PATH}/tokenizer_config.json" ]] || die "Missing tokenizer config: ${MODEL_PATH}/tokenizer_config.json"
  find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q . \
    || die "No model weight files (*.safetensors or *.bin) found in ${MODEL_PATH}"
}

check_npu_devices() {
  local id count=0
  for id in ${NPU_DEVICE_IDS}; do
    [[ "${id}" =~ ^[0-9]+$ ]] || die "Invalid NPU device id in NPU_DEVICE_IDS: ${id}"
    [[ -e "/dev/davinci${id}" ]] || die "NPU device is not visible in the container: /dev/davinci${id}"
    count=$((count + 1))
  done
  (( count == TP_SIZE )) || die "Visible NPU list has ${count} devices, but TP_SIZE=${TP_SIZE}"
}

run_preflight() {
  local check_port="${1:-1}"

  validate_configuration
  ensure_runtime_dir
  require_command vllm
  require_command curl
  require_command setsid
  require_command awk
  require_command ps
  require_command find
  require_command sort
  require_command tail
  check_model_files
  check_npu_devices
  cleanup_stale_pid_file

  if [[ "${check_port}" == "1" ]] && port_is_open; then
    if read_managed_process >/dev/null 2>&1; then
      info "Port ${HOST}:${PORT} is used by the managed Qwen service."
    else
      show_port_owner
      die "Port ${HOST}:${PORT} is already in use by an unmanaged process"
    fi
  fi
}

configure_environment() {
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
  export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"

  if [[ "${OFFLINE_MODE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  else
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
  fi
}

build_vllm_command() {
  local profile="$1"
  VLLM_COMMAND=(
    vllm serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --data-parallel-size "${DP_SIZE}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --trust-remote-code
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
  )

  if [[ "${profile}" == "optimized" ]]; then
    VLLM_COMMAND+=(
      --enable-prefix-caching
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
      --additional-config '{"enable_cpu_binding":true}'
      --async-scheduling
    )
  else
    VLLM_COMMAND+=(--no-enable-prefix-caching)
  fi
}

write_process_metadata() {
  local pid="$1" pgid="$2" start_time="$3" profile="$4" log_path="$5"
  local tmp_file="${PID_FILE}.tmp"

  printf '%s %s %s\n' "${pid}" "${start_time}" "${pgid}" > "${tmp_file}"
  mv -f -- "${tmp_file}" "${PID_FILE}"
  printf '%s\n' "${profile}" > "${PROFILE_FILE}"
  date +%s > "${STARTED_AT_FILE}"
  printf '%s\n' "${log_path}" > "${CURRENT_LOG_FILE}"
  printf '%s\n%s\n%s\n' "${HOST}" "${PORT}" "${SERVED_MODEL_NAME}" > "${STATE_FILE}"
}

show_log_tail() {
  local lines="${1:-80}" log_path
  log_path="$(current_log_path 2>/dev/null || true)"
  if [[ -n "${log_path}" ]]; then
    printf '\nLast %s log lines (%s):\n' "${lines}" "${log_path}" >&2
    tail -n "${lines}" -- "${log_path}" >&2 || true
  fi
}

stop_service() {
  local managed pid pgid waited=0 self_pgid

  ensure_runtime_dir
  managed="$(read_managed_process 2>/dev/null || true)"
  if [[ -z "${managed}" ]]; then
    cleanup_stale_pid_file
    info "Qwen3.6-27B is not running."
    return 0
  fi

  read -r pid pgid <<< "${managed}"
  self_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  [[ "${pgid}" != "${self_pgid}" ]] || die "Refusing to signal the management script's own process group"

  info "Stopping Qwen3.6-27B (PID ${pid}, PGID ${pgid})..."
  kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

  while kill -0 -- "-${pgid}" 2>/dev/null; do
    if (( waited >= STOP_TIMEOUT )); then
      warn "Graceful shutdown exceeded ${STOP_TIMEOUT}s; forcing process-group cleanup."
      kill -KILL -- "-${pgid}" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  remove_process_metadata
  info "Qwen3.6-27B stopped."
}

cancel_start() {
  trap - INT TERM
  warn "Startup cancelled; stopping the model process."
  if (( STARTED_BY_THIS_COMMAND == 1 )); then
    stop_service || true
  fi
  exit 130
}

start_service() {
  local profile="${1:-optimized}" managed pid pgid start_time timestamp log_path
  local elapsed=0

  validate_profile "${profile}"
  ensure_runtime_dir
  cleanup_stale_pid_file

  managed="$(read_managed_process 2>/dev/null || true)"
  if [[ -n "${managed}" ]]; then
    local active_profile="unknown"
    [[ -r "${PROFILE_FILE}" ]] && active_profile="$(<"${PROFILE_FILE}")"
    load_active_endpoint
    if api_ready; then
      info "Qwen3.6-27B is already running and ready (profile: ${active_profile})."
      if [[ "${active_profile}" != "${profile}" ]]; then
        info "Use 'restart ${profile}' to change profiles."
      fi
      return 0
    fi
    die "A managed Qwen process is already running but its API is not ready; use status/logs or stop it first"
  fi

  run_preflight 1
  configure_environment
  build_vllm_command "${profile}"

  timestamp="$(date '+%Y%m%d-%H%M%S')"
  log_path="${RUNTIME_DIR}/serve-${timestamp}-${profile}.log"
  : > "${log_path}"
  printf 'Profile: %s\nStarted: %s\nCommand:' "${profile}" "$(date --iso-8601=seconds)" >> "${log_path}"
  printf ' %q' "${VLLM_COMMAND[@]}" >> "${log_path}"
  printf '\n\n' >> "${log_path}"

  info "Starting Qwen3.6-27B with profile '${profile}'..."
  setsid "${VLLM_COMMAND[@]}" >> "${log_path}" 2>&1 < /dev/null &
  pid=$!
  STARTED_BY_THIS_COMMAND=1

  sleep 1
  kill -0 "${pid}" 2>/dev/null || {
    printf '%s\n' "${log_path}" > "${CURRENT_LOG_FILE}"
    show_log_tail
    die "vLLM exited immediately; optimized mode is not automatically changed to safe"
  }

  pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
  start_time="$(proc_start_time "${pid}" 2>/dev/null || true)"
  [[ "${pgid}" =~ ^[1-9][0-9]*$ && "${start_time}" =~ ^[0-9]+$ ]] || {
    kill -TERM "${pid}" 2>/dev/null || true
    die "Failed to record the vLLM process metadata"
  }
  write_process_metadata "${pid}" "${pgid}" "${start_time}" "${profile}" "${log_path}"
  cleanup_old_logs

  trap cancel_start INT TERM
  info "Waiting up to ${START_TIMEOUT}s for ${BASE_URL}/v1/models ..."
  while (( elapsed < START_TIMEOUT )); do
    if api_ready; then
      trap - INT TERM
      STARTED_BY_THIS_COMMAND=0
      info "Qwen3.6-27B is ready."
      info "Profile: ${profile}"
      info "API: ${BASE_URL}/v1"
      info "Log: ${log_path}"
      return 0
    fi

    if ! read_managed_process >/dev/null 2>&1; then
      trap - INT TERM
      show_log_tail
      remove_process_metadata
      die "vLLM exited before the API became ready; try 'start safe' after reviewing the log"
    fi

    sleep 2
    elapsed=$((elapsed + 2))
    if (( elapsed % 30 == 0 )); then
      info "Still loading (${elapsed}s elapsed)..."
    fi
  done

  trap - INT TERM
  error "API did not become ready within ${START_TIMEOUT}s."
  show_log_tail
  stop_service || true
  die "Startup timed out; optimized mode was not automatically changed to safe"
}

format_uptime() {
  local total="$1" days hours minutes seconds
  days=$((total / 86400))
  hours=$(((total % 86400) / 3600))
  minutes=$(((total % 3600) / 60))
  seconds=$((total % 60))
  if (( days > 0 )); then
    printf '%dd %02dh %02dm %02ds' "${days}" "${hours}" "${minutes}" "${seconds}"
  else
    printf '%02dh %02dm %02ds' "${hours}" "${minutes}" "${seconds}"
  fi
}

status_service() {
  local managed pid pgid profile="unknown" log_path="none" started_at now uptime state

  ensure_runtime_dir
  cleanup_stale_pid_file
  managed="$(read_managed_process 2>/dev/null || true)"
  if [[ -z "${managed}" ]]; then
    printf 'Status: stopped\n'
    [[ -r "${PROFILE_FILE}" ]] && printf 'Last profile: %s\n' "$(<"${PROFILE_FILE}")"
    log_path="$(current_log_path 2>/dev/null || true)"
    [[ -n "${log_path}" ]] && printf 'Last log: %s\n' "${log_path}"
    return 1
  fi

  read -r pid pgid <<< "${managed}"
  load_active_endpoint
  [[ -r "${PROFILE_FILE}" ]] && profile="$(<"${PROFILE_FILE}")"
  log_path="$(current_log_path 2>/dev/null || true)"
  state="loading"
  api_ready && state="ready"

  uptime="unknown"
  if [[ -r "${STARTED_AT_FILE}" ]]; then
    started_at="$(<"${STARTED_AT_FILE}")"
    now="$(date +%s)"
    [[ "${started_at}" =~ ^[0-9]+$ && "${now}" =~ ^[0-9]+$ && ${now} -ge ${started_at} ]] \
      && uptime="$(format_uptime "$((now - started_at))")"
  fi

  printf 'Status: %s\n' "${state}"
  printf 'Mode: %s\n' "${profile}"
  printf 'PID: %s\n' "${pid}"
  printf 'PGID: %s\n' "${pgid}"
  printf 'Uptime: %s\n' "${uptime}"
  printf 'Model: %s\n' "${SERVED_MODEL_NAME}"
  printf 'API: %s/v1\n' "${BASE_URL}"
  [[ -n "${log_path}" ]] && printf 'Log: %s\n' "${log_path}"
}

logs_service() {
  local mode="${1:-}" log_path
  [[ -z "${mode}" || "${mode}" == "-f" ]] || die "Usage: logs [-f]"
  ensure_runtime_dir
  log_path="$(current_log_path 2>/dev/null || true)"
  [[ -n "${log_path}" ]] || die "No Qwen3.6-27B log file found"

  if [[ "${mode}" == "-f" ]]; then
    tail -n 100 -f -- "${log_path}"
  else
    tail -n 100 -- "${log_path}"
  fi
}

test_service() {
  local response managed
  require_command curl
  managed="$(read_managed_process 2>/dev/null || true)"
  [[ -n "${managed}" ]] || die "Qwen3.6-27B is not running under this manager"
  load_active_endpoint
  api_ready || die "API is not ready at ${BASE_URL}/v1"

  info "Model registry is reachable. Sending a short non-thinking request..."
  response="$(curl -fsS --max-time 180 "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @- <<JSON
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
  "temperature": 0,
  "max_tokens": 16,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
)" || die "Chat completion request failed"

  if command -v python3 >/dev/null 2>&1; then
    RESPONSE_JSON="${response}" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["RESPONSE_JSON"])
choices = payload.get("choices") or []
if not choices:
    raise SystemExit("No choices returned by the model")
message = choices[0].get("message") or {}
content = message.get("content")
if not content:
    raise SystemExit("The model returned an empty content field")
print(f"Response: {content}")
PY
  else
    [[ "${response}" == *'"choices"'* ]] || die "The response does not contain choices: ${response}"
    printf 'Response JSON: %s\n' "${response}"
  fi
  info "Inference smoke test passed."
}

check_service() {
  run_preflight 1
  info "Environment check passed."
  info "Model path: ${MODEL_PATH}"
  info "NPU devices: ${NPU_DEVICE_IDS}"
  info "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}"
  info "API: ${BASE_URL}/v1"
  info "Runtime directory: ${RUNTIME_DIR}"
}

restart_service() {
  local profile="${1:-}"
  if [[ -z "${profile}" && -r "${PROFILE_FILE}" ]]; then
    profile="$(<"${PROFILE_FILE}")"
  fi
  profile="${profile:-optimized}"
  validate_profile "${profile}"
  stop_service
  start_service "${profile}"
}

main() {
  local command="${1:-help}"
  shift || true

  case "${command}" in
    help|--help|-h)
      (( $# == 0 )) || die "The help command does not accept arguments"
      usage
      ;;
    check)
      (( $# == 0 )) || die "Usage: check"
      check_service
      ;;
    start)
      (( $# <= 1 )) || die "Usage: start [optimized|safe]"
      start_service "${1:-optimized}"
      ;;
    stop)
      (( $# == 0 )) || die "Usage: stop"
      stop_service
      ;;
    restart)
      (( $# <= 1 )) || die "Usage: restart [optimized|safe]"
      restart_service "${1:-}"
      ;;
    status)
      (( $# == 0 )) || die "Usage: status"
      status_service
      ;;
    logs)
      (( $# <= 1 )) || die "Usage: logs [-f]"
      logs_service "${1:-}"
      ;;
    test)
      (( $# == 0 )) || die "Usage: test"
      test_service
      ;;
    *)
      usage >&2
      die "Unknown command: ${command}"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
