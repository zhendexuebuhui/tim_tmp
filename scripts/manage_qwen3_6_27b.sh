#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-2}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
API_SERVER_COUNT="${API_SERVER_COUNT:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
START_TIMEOUT="${START_TIMEOUT:-1800}"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"
LOG_RETENTION="${LOG_RETENTION:-10}"
NPU_DEVICE_IDS="${NPU_DEVICE_IDS:-2 3 5 7}"
ENABLE_REDUCE_SAMPLE="${ENABLE_REDUCE_SAMPLE:-1}"
RUN_LABEL="${RUN_LABEL:-}"
METRICS_INTERVAL="${METRICS_INTERVAL:-2}"

BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-32768}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-100}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-1}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-8}"
BENCH_THINKING="${BENCH_THINKING:-0}"
BENCH_NPU_MONITOR="${BENCH_NPU_MONITOR:-1}"
BENCH_NPU_SAMPLE_INTERVAL="${BENCH_NPU_SAMPLE_INTERVAL:-1}"
BENCH_TEMPERATURE="${BENCH_TEMPERATURE:-}"

RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/runtime/qwen3.6-27b}"
STATE_DIR="${STATE_DIR:-/run/qwen3.6-27b}"
PID_FILE="${STATE_DIR}/server.pid"
PROFILE_FILE="${STATE_DIR}/profile"
STARTED_AT_FILE="${STATE_DIR}/started_at"
CURRENT_LOG_FILE="${STATE_DIR}/current_log"
STATE_FILE="${STATE_DIR}/service_state"
RUN_ID_FILE="${STATE_DIR}/run_id"
PROC_ROOT="${PROC_ROOT:-/proc}"

BASE_URL="http://${HOST}:${PORT}"
STARTED_BY_THIS_COMMAND=0
NPU_SAMPLER_PID=""

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

require_vllm_ascend_container() {
  if [[ "${IN_VLLM_ASCEND_CONTAINER:-}" != "1" ]]; then
    error "This script can only run inside the designated vLLM-Ascend container."
    error "Enter it with: docker exec -it vllm-ascend-env bash"
    exit 1
  fi
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
  manage_qwen3_6_27b.sh metrics [-f]
  manage_qwen3_6_27b.sh test
  manage_qwen3_6_27b.sh bench
  manage_qwen3_6_27b.sh help | --help | -h

Profiles:
  optimized  Default. Enables MTP, FULL_DECODE_ONLY graph mode, CPU binding,
             reduce-sample, async scheduling, and prefix caching.
  safe       Disables the optional performance features above for diagnosis.

Common environment overrides:
  MODEL_PATH, SERVED_MODEL_NAME, HOST, PORT, TP_SIZE, DP_SIZE, DTYPE,
  MAX_MODEL_LEN, MAX_NUM_SEQS,
  MAX_NUM_BATCHED_TOKENS, API_SERVER_COUNT, GPU_MEMORY_UTILIZATION,
  OFFLINE_MODE, START_TIMEOUT, STOP_TIMEOUT, LOG_RETENTION,
  NPU_DEVICE_IDS, ENABLE_REDUCE_SAMPLE, RUN_LABEL, METRICS_INTERVAL,
  RUNTIME_DIR, STATE_DIR

Benchmark environment overrides:
  BENCH_INPUT_LEN, BENCH_OUTPUT_LEN, BENCH_NUM_PROMPTS,
  BENCH_REQUEST_RATE, BENCH_MAX_CONCURRENCY, BENCH_THINKING,
  BENCH_NPU_MONITOR, BENCH_NPU_SAMPLE_INTERVAL, BENCH_TEMPERATURE

Examples:
  ./scripts/manage_qwen3_6_27b.sh check
  ./scripts/manage_qwen3_6_27b.sh start
  ./scripts/manage_qwen3_6_27b.sh start safe
  ./scripts/manage_qwen3_6_27b.sh logs -f
  ./scripts/manage_qwen3_6_27b.sh metrics -f
  ./scripts/manage_qwen3_6_27b.sh bench
  BENCH_NUM_PROMPTS=5 BENCH_INPUT_LEN=16384 \
    BENCH_MAX_CONCURRENCY=2 ./scripts/manage_qwen3_6_27b.sh bench
  BENCH_NPU_SAMPLE_INTERVAL=2 ./scripts/manage_qwen3_6_27b.sh bench
  MAX_MODEL_LEN=131072 ./scripts/manage_qwen3_6_27b.sh restart optimized
  OFFLINE_MODE=0 ./scripts/manage_qwen3_6_27b.sh start
EOF
}

ensure_runtime_dir() {
  mkdir -p -- "${RUNTIME_DIR}" "${STATE_DIR}"
  [[ -w "${RUNTIME_DIR}" ]] || die "Runtime directory is not writable: ${RUNTIME_DIR}"
  [[ -w "${STATE_DIR}" ]] || die "State directory is not writable: ${STATE_DIR}"
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
  is_positive_integer "${API_SERVER_COUNT}" \
    || die "API_SERVER_COUNT must be a positive integer: ${API_SERVER_COUNT}"
  is_positive_integer "${MAX_MODEL_LEN}" || die "MAX_MODEL_LEN must be a positive integer"
  is_positive_integer "${MAX_NUM_SEQS}" || die "MAX_NUM_SEQS must be a positive integer"
  is_positive_integer "${MAX_NUM_BATCHED_TOKENS}" || die "MAX_NUM_BATCHED_TOKENS must be a positive integer"
  is_positive_integer "${START_TIMEOUT}" || die "START_TIMEOUT must be a positive integer"
  is_positive_integer "${STOP_TIMEOUT}" || die "STOP_TIMEOUT must be a positive integer"
  is_positive_integer "${LOG_RETENTION}" || die "LOG_RETENTION must be a positive integer"
  [[ "${OFFLINE_MODE}" == "0" || "${OFFLINE_MODE}" == "1" ]] || die "OFFLINE_MODE must be 0 or 1"
  [[ "${ENABLE_REDUCE_SAMPLE}" == "0" || "${ENABLE_REDUCE_SAMPLE}" == "1" ]] \
    || die "ENABLE_REDUCE_SAMPLE must be 0 or 1"
  [[ -z "${RUN_LABEL}" || "${RUN_LABEL}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "RUN_LABEL may contain only letters, numbers, dots, underscores, and hyphens"
  [[ "${HOST}" == "127.0.0.1" || "${HOST}" == "localhost" ]] || warn "HOST=${HOST} exposes behavior beyond the recommended loopback-only default"
}

validate_profile() {
  case "$1" in
    optimized|safe) ;;
    *) die "Unknown profile '$1'. Expected optimized or safe." ;;
  esac
}

proc_start_time() {
  local pid="$1" stat rest
  [[ -r "${PROC_ROOT}/${pid}/stat" ]] || return 1
  IFS= read -r stat < "${PROC_ROOT}/${pid}/stat" || return 1
  rest="${stat##*) }"
  [[ "${rest}" != "${stat}" ]] || return 1
  awk '{print $20}' <<< "${rest}"
}

proc_group_id() {
  local pid="$1" stat rest remainder
  [[ -r "${PROC_ROOT}/${pid}/stat" ]] || return 1
  IFS= read -r stat < "${PROC_ROOT}/${pid}/stat" || return 1
  rest="${stat##*) }"
  [[ "${rest}" != "${stat}" ]] || return 1
  remainder="${rest#* }"
  remainder="${remainder#* }"
  printf '%s\n' "${remainder%% *}"
}

proc_state() {
  local pid="$1" stat rest
  [[ -r "${PROC_ROOT}/${pid}/stat" ]] || return 1
  IFS= read -r stat < "${PROC_ROOT}/${pid}/stat" || return 1
  rest="${stat##*) }"
  [[ "${rest}" != "${stat}" ]] || return 1
  printf '%s\n' "${rest%% *}"
}

process_has_run_id() {
  local pid="$1" expected="$2" entry
  [[ -n "${expected}" && -r "${PROC_ROOT}/${pid}/environ" ]] || return 1
  while IFS= read -r -d '' entry; do
    [[ "${entry}" == "VLLM_MANAGER_RUN_ID=${expected}" ]] && return 0
  done < "${PROC_ROOT}/${pid}/environ"
  return 1
}

find_managed_group_member() {
  local pgid="$1" run_id="${2:-}" candidate member_pgid state command comm=""

  while read -r candidate member_pgid state; do
    [[ "${member_pgid}" == "${pgid}" && "${state}" != Z* ]] || continue
    if [[ -n "${run_id}" ]]; then
      process_has_run_id "${candidate}" "${run_id}" && {
        printf '%s\n' "${candidate}"
        return 0
      }
      continue
    fi

    command="$(tr '\0' ' ' < "${PROC_ROOT}/${candidate}/cmdline" 2>/dev/null || true)"
    [[ -r "${PROC_ROOT}/${candidate}/comm" ]] && IFS= read -r comm < "${PROC_ROOT}/${candidate}/comm" || true
    command+=" ${comm}"
    [[ "${command,,}" == *vllm* ]] && {
      printf '%s\n' "${candidate}"
      return 0
    }
  done < <(ps -eo pid=,pgid=,stat= 2>/dev/null)
  return 1
}

read_managed_process() {
  local pid recorded_start pgid actual_start actual_pgid actual_state run_id="" member

  [[ -r "${PID_FILE}" ]] || return 1
  read -r pid recorded_start pgid < "${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${recorded_start}" =~ ^[0-9]+$ && "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "${RUN_ID_FILE}" ]] && run_id="$(<"${RUN_ID_FILE}")"

  if kill -0 "${pid}" 2>/dev/null; then
    actual_start="$(proc_start_time "${pid}" 2>/dev/null || true)"
    actual_pgid="$(proc_group_id "${pid}" 2>/dev/null || true)"
    actual_state="$(proc_state "${pid}" 2>/dev/null || true)"
    if [[ -n "${actual_start}" && "${actual_start}" == "${recorded_start}" \
      && "${actual_pgid}" == "${pgid}" && "${actual_state}" != Z ]]; then
      printf '%s %s\n' "${pid}" "${pgid}"
      return 0
    fi
  fi

  member="$(find_managed_group_member "${pgid}" "${run_id}" 2>/dev/null || true)"
  [[ -n "${member}" ]] || return 1
  printf '%s %s\n' "${member}" "${pgid}"
}

remove_process_metadata() {
  rm -f -- "${PID_FILE}" "${STARTED_AT_FILE}" "${RUN_ID_FILE}"
}

endpoint_serves_model() {
  local base_url="$1" expected_model="$2" response compact
  response="$(curl -fsS --max-time 5 "${base_url}/v1/models")" || return 1
  compact="$(printf '%s' "${response}" | tr -d '[:space:]')"
  awk -v needle="\"id\":\"${expected_model}\"" 'index($0, needle) { found=1 } END { exit !found }' <<< "${compact}"
}

listener_pids() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null \
      | awk -v port=":${port}" '$4 ~ port "$" {print}' \
      | grep -oE 'pid=[0-9]+' \
      | cut -d= -f2 \
      | sort -nu
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -nu
  fi
}

recover_process_metadata() {
  local pid pgid start_time member recovery_host recovery_port recovery_model recovery_base
  local -a state=()

  [[ -r "${STATE_FILE}" && -r "${PROFILE_FILE}" && -r "${CURRENT_LOG_FILE}" ]] || return 1
  mapfile -t state < "${STATE_FILE}"
  (( ${#state[@]} >= 3 )) || return 1
  recovery_host="${state[0]}"
  recovery_port="${state[1]}"
  recovery_model="${state[2]}"
  [[ -n "${recovery_host}" && "${recovery_port}" =~ ^[1-9][0-9]*$ \
    && ${recovery_port} -le 65535 && -n "${recovery_model}" ]] || return 1
  recovery_base="http://${recovery_host}:${recovery_port}"
  endpoint_serves_model "${recovery_base}" "${recovery_model}" || return 1

  while IFS= read -r pid; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || continue
    pgid="$(proc_group_id "${pid}" 2>/dev/null || true)"
    [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || continue
    member="$(find_managed_group_member "${pgid}" 2>/dev/null || true)"
    [[ -n "${member}" ]] || continue
    start_time="$(proc_start_time "${member}" 2>/dev/null || true)"
    [[ "${pgid}" =~ ^[1-9][0-9]*$ && "${start_time}" =~ ^[0-9]+$ ]] || continue
    printf '%s %s %s\n' "${member}" "${start_time}" "${pgid}" > "${PID_FILE}.tmp"
    mv -f -- "${PID_FILE}.tmp" "${PID_FILE}"
    warn "Recovered managed vLLM process metadata from ${recovery_host}:${recovery_port} (PID ${member}, PGID ${pgid})."
    return 0
  done < <(listener_pids "${recovery_port}")
  return 1
}

cleanup_stale_pid_file() {
  read_managed_process >/dev/null 2>&1 && return 0
  recover_process_metadata && return 0
  if [[ -e "${PID_FILE}" ]]; then
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
  local id count=0 expected_count
  for id in ${NPU_DEVICE_IDS}; do
    [[ "${id}" =~ ^[0-9]+$ ]] || die "Invalid NPU device id in NPU_DEVICE_IDS: ${id}"
    [[ -e "/dev/davinci${id}" ]] || die "NPU device is not visible in the container: /dev/davinci${id}"
    count=$((count + 1))
  done
  expected_count=$((TP_SIZE * DP_SIZE))
  (( count == expected_count )) \
    || die "Visible NPU list has ${count} devices, but TP_SIZE*DP_SIZE=${expected_count}"
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
  local profile="$1" additional_config
  VLLM_COMMAND=(
    vllm serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --api-server-count "${API_SERVER_COUNT}"
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
    additional_config='{"enable_cpu_binding":true}'
    if [[ "${ENABLE_REDUCE_SAMPLE}" == "1" ]]; then
      additional_config='{"enable_cpu_binding":true,"enable_reduce_sample":true}'
    fi
    VLLM_COMMAND+=(
      --enable-prefix-caching
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
      --additional-config "${additional_config}"
      --async-scheduling
    )
  else
    VLLM_COMMAND+=(--no-enable-prefix-caching)
  fi
}

write_process_metadata() {
  local pid="$1" pgid="$2" start_time="$3" profile="$4" log_path="$5" run_id="$6"
  local tmp_file="${PID_FILE}.tmp"

  printf '%s %s %s\n' "${pid}" "${start_time}" "${pgid}" > "${tmp_file}"
  mv -f -- "${tmp_file}" "${PID_FILE}"
  printf '%s\n' "${profile}" > "${PROFILE_FILE}"
  date +%s > "${STARTED_AT_FILE}"
  printf '%s\n' "${log_path}" > "${CURRENT_LOG_FILE}"
  printf '%s\n%s\n%s\n' "${HOST}" "${PORT}" "${SERVED_MODEL_NAME}" > "${STATE_FILE}"
  printf '%s\n' "${run_id}" > "${RUN_ID_FILE}"
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
  cleanup_stale_pid_file
  managed="$(read_managed_process 2>/dev/null || true)"
  if [[ -z "${managed}" ]]; then
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
  local profile="${1:-optimized}" managed pid pgid start_time timestamp log_path label_suffix="" run_id
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
  [[ -n "${RUN_LABEL}" ]] && label_suffix="-${RUN_LABEL}"
  log_path="${RUNTIME_DIR}/serve-${timestamp}${label_suffix}-${profile}.log"
  : > "${log_path}"
  printf 'Profile: %s\nStarted: %s\nCommand:' "${profile}" "$(date --iso-8601=seconds)" >> "${log_path}"
  printf ' %q' "${VLLM_COMMAND[@]}" >> "${log_path}"
  printf '\n\n' >> "${log_path}"

  info "Starting Qwen3.6-27B with profile '${profile}'..."
  run_id="$(date +%s%N)-$$-${RANDOM}"
  VLLM_MANAGER_RUN_ID="${run_id}" setsid "${VLLM_COMMAND[@]}" >> "${log_path}" 2>&1 < /dev/null &
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
  write_process_metadata "${pid}" "${pgid}" "${start_time}" "${profile}" "${log_path}" "${run_id}"
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

fetch_metric_snapshot() {
  local raw
  raw="$(curl -fsS --max-time 5 "${BASE_URL}/metrics")" || return 1

  awk '
    BEGIN {
      prompt = generation = running = waiting = 0
      kv_sum = kv_max = kv_count = 0
      prefix_hits = prefix_queries = accepted = draft = 0
      have_prompt = have_generation = 0
    }
    $1 !~ /^#/ && NF >= 2 {
      name = $1
      sub(/\{.*/, "", name)
      value = $NF + 0

      if (name == "vllm:prompt_tokens_total") {
        prompt += value
        have_prompt++
      } else if (name == "vllm:generation_tokens_total") {
        generation += value
        have_generation++
      } else if (name == "vllm:num_requests_running") {
        running += value
      } else if (name == "vllm:num_requests_waiting") {
        waiting += value
      } else if (name == "vllm:kv_cache_usage_perc") {
        kv_sum += value
        if (kv_count == 0 || value > kv_max) kv_max = value
        kv_count++
      } else if (name == "vllm:prefix_cache_hits_total") {
        prefix_hits += value
      } else if (name == "vllm:prefix_cache_queries_total") {
        prefix_queries += value
      } else if (name == "vllm:spec_decode_num_accepted_tokens_total") {
        accepted += value
      } else if (name == "vllm:spec_decode_num_draft_tokens_total") {
        draft += value
      }
    }
    END {
      kv_avg = kv_count > 0 ? kv_sum / kv_count : -1
      if (kv_count == 0) kv_max = -1
      printf "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %d\n", \
        prompt, generation, running, waiting, kv_avg, kv_max, prefix_hits, \
        prefix_queries, accepted, draft, have_prompt, have_generation
    }
  ' <<< "${raw}"
}

validate_metric_snapshot() {
  local snapshot="$1"
  local -a values=()

  read -r -a values <<< "${snapshot}"
  (( ${#values[@]} == 12 )) || die "Unexpected response from the metrics parser"
  [[ "${values[10]}" =~ ^[1-9][0-9]*$ ]] \
    || die "The metrics endpoint does not expose vllm:prompt_tokens_total"
  [[ "${values[11]}" =~ ^[1-9][0-9]*$ ]] \
    || die "The metrics endpoint does not expose vllm:generation_tokens_total"
}

print_metrics_header() {
  printf '%-19s %13s %12s %7s %7s %15s %27s %27s\n' \
    "Timestamp" "Prefill tok/s" "Decode tok/s" "Running" "Waiting" \
    "KV avg/max" "Prefix hit recent/lifetime" "MTP accept recent/lifetime"
}

print_metrics_row() {
  local current="$1" previous="$2" elapsed="$3" timestamp
  local -a values=() previous_values=()

  read -r -a values <<< "${current}"
  read -r -a previous_values <<< "${previous}"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  awk \
    -v timestamp="${timestamp}" -v elapsed="${elapsed}" \
    -v prompt="${values[0]}" -v generation="${values[1]}" \
    -v running="${values[2]}" -v waiting="${values[3]}" \
    -v kv_avg="${values[4]}" -v kv_max="${values[5]}" \
    -v prefix_hits="${values[6]}" -v prefix_queries="${values[7]}" \
    -v accepted="${values[8]}" -v draft="${values[9]}" \
    -v prev_prompt="${previous_values[0]}" -v prev_generation="${previous_values[1]}" \
    -v prev_prefix_hits="${previous_values[6]}" -v prev_prefix_queries="${previous_values[7]}" \
    -v prev_accepted="${previous_values[8]}" -v prev_draft="${previous_values[9]}" '
    function percentage(numerator, denominator) {
      return denominator > 0 ? sprintf("%.1f%%", 100 * numerator / denominator) : "n/a"
    }
    BEGIN {
      prefill_rate = prompt >= prev_prompt ? sprintf("%.1f", (prompt - prev_prompt) / elapsed) : "n/a"
      decode_rate = generation >= prev_generation ? sprintf("%.1f", (generation - prev_generation) / elapsed) : "n/a"
      kv = kv_avg >= 0 ? sprintf("%.1f%%/%.1f%%", 100 * kv_avg, 100 * kv_max) : "n/a"

      prefix_queries_delta = prefix_queries - prev_prefix_queries
      prefix_hits_delta = prefix_hits - prev_prefix_hits
      prefix_recent = prefix_queries_delta > 0 && prefix_hits_delta >= 0 \
        ? percentage(prefix_hits_delta, prefix_queries_delta) : "n/a"
      prefix = prefix_recent "/" percentage(prefix_hits, prefix_queries)

      draft_delta = draft - prev_draft
      accepted_delta = accepted - prev_accepted
      mtp_recent = draft_delta > 0 && accepted_delta >= 0 \
        ? percentage(accepted_delta, draft_delta) : "n/a"
      mtp = mtp_recent "/" percentage(accepted, draft)

      printf "%-19s %13s %12s %7.0f %7.0f %15s %27s %27s\n", \
        timestamp, prefill_rate, decode_rate, running, waiting, kv, prefix, mtp
    }
  '
}

metrics_service() {
  local mode="${1:-}" previous current previous_time current_time elapsed
  local managed

  [[ -z "${mode}" || "${mode}" == "-f" ]] || die "Usage: metrics [-f]"
  [[ "${METRICS_INTERVAL}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "METRICS_INTERVAL must be a positive number"
  awk -v value="${METRICS_INTERVAL}" 'BEGIN { exit !(value > 0) }' \
    || die "METRICS_INTERVAL must be greater than zero"

  require_command curl
  require_command awk
  require_command date
  require_command sleep
  managed="$(read_managed_process 2>/dev/null || true)"
  [[ -n "${managed}" ]] || die "Qwen3.6-27B is not running under this manager"
  load_active_endpoint
  api_ready || die "API is not ready at ${BASE_URL}/v1"

  previous="$(fetch_metric_snapshot)" \
    || die "Failed to read Prometheus metrics from ${BASE_URL}/metrics"
  validate_metric_snapshot "${previous}"
  previous_time="$(date +%s.%N)"

  info "Metrics: ${BASE_URL}/metrics; interval: ${METRICS_INTERVAL}s"
  print_metrics_header
  while true; do
    sleep "${METRICS_INTERVAL}"
    if ! current="$(fetch_metric_snapshot)"; then
      if [[ "${mode}" == "-f" ]]; then
        warn "Failed to read ${BASE_URL}/metrics; retrying"
        continue
      fi
      die "Failed to read Prometheus metrics from ${BASE_URL}/metrics"
    fi
    validate_metric_snapshot "${current}"
    current_time="$(date +%s.%N)"
    elapsed="$(awk -v current="${current_time}" -v previous="${previous_time}" \
      'BEGIN { printf "%.9f", current - previous }')"
    print_metrics_row "${current}" "${previous}" "${elapsed}"
    previous="${current}"
    previous_time="${current_time}"
    [[ "${mode}" == "-f" ]] || break
  done
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

validate_bench_configuration() {
  is_positive_integer "${BENCH_INPUT_LEN}" || die "BENCH_INPUT_LEN must be a positive integer"
  is_positive_integer "${BENCH_OUTPUT_LEN}" || die "BENCH_OUTPUT_LEN must be a positive integer"
  is_positive_integer "${BENCH_NUM_PROMPTS}" || die "BENCH_NUM_PROMPTS must be a positive integer"
  is_positive_integer "${BENCH_MAX_CONCURRENCY}" || die "BENCH_MAX_CONCURRENCY must be a positive integer"
  [[ "${BENCH_THINKING}" == "0" || "${BENCH_THINKING}" == "1" ]] \
    || die "BENCH_THINKING must be 0 or 1"
  [[ "${BENCH_NPU_MONITOR}" == "0" || "${BENCH_NPU_MONITOR}" == "1" ]] \
    || die "BENCH_NPU_MONITOR must be 0 or 1"

  [[ "${BENCH_NPU_SAMPLE_INTERVAL}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "BENCH_NPU_SAMPLE_INTERVAL must be a positive number"
  awk -v value="${BENCH_NPU_SAMPLE_INTERVAL}" 'BEGIN { exit !(value > 0) }' \
    || die "BENCH_NPU_SAMPLE_INTERVAL must be greater than zero"

  if [[ "${BENCH_REQUEST_RATE}" != "inf" ]]; then
    [[ "${BENCH_REQUEST_RATE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
      || die "BENCH_REQUEST_RATE must be a positive number or inf"
    awk -v value="${BENCH_REQUEST_RATE}" 'BEGIN { exit !(value > 0) }' \
      || die "BENCH_REQUEST_RATE must be greater than zero"
  fi

  if [[ -n "${BENCH_TEMPERATURE}" ]]; then
    [[ "${BENCH_TEMPERATURE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
      || die "BENCH_TEMPERATURE must be a non-negative number"
  fi

  if (( BENCH_INPUT_LEN + BENCH_OUTPUT_LEN > MAX_MODEL_LEN )); then
    warn "Requested benchmark tokens (${BENCH_INPUT_LEN}+${BENCH_OUTPUT_LEN}) exceed MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  fi
}

cleanup_old_bench_files() {
  local pattern index entry old_file
  for pattern in 'bench-*.log' 'bench-*.json' 'bench-*-npu.csv' 'bench-*-summary.txt' 'bench-*-npu-monitor.err'; do
    index=0
    while IFS= read -r entry; do
      [[ -n "${entry}" ]] || continue
      index=$((index + 1))
      if (( index > LOG_RETENTION )); then
        old_file="${entry#* }"
        [[ "${old_file}" == "${RUNTIME_DIR}"/bench-* ]] || continue
        rm -f -- "${old_file}"
      fi
    done < <(find "${RUNTIME_DIR}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null | sort -nr)
  done
}

collect_npu_snapshot() {
  local timestamp="${1:-$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')}" output

  output="$(npu-smi info 2>/dev/null)" || return 1
  printf '%s\n' "${output}" | awk \
    -v timestamp="${timestamp}" \
    -v configured_ids="${NPU_DEVICE_IDS}" \
    -v expected_count="$((TP_SIZE * DP_SIZE))" '
    function trim(value) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      return value
    }
    function save_row(id, payload, values, value_count) {
      value_count = split(payload, values, /[[:space:]\/]+/)
      if (value_count < 5) return
      rows[id] = timestamp "," id "," values[1] "," power "," temperature "," \
        values[2] "," values[3] "," values[4] "," values[5]
      if (!(id in seen)) {
        seen[id] = 1
        order[++row_count] = id
      }
    }
    BEGIN {
      wanted_count = split(configured_ids, wanted_order, /[[:space:]]+/)
      for (i = 1; i <= wanted_count; i++) wanted[wanted_order[i]] = 1
    }
    {
      column_count = split($0, columns, "|")
      if (column_count < 5) next
      left = trim(columns[2])
      middle = trim(columns[3])
      payload = trim(columns[4])

      if (middle ~ /^(OK|Warning|Alarm|Critical|UNKNOWN)$/) {
        part_count = split(left, parts, /[[:space:]]+/)
        metric_count = split(payload, metrics, /[[:space:]\/]+/)
        if (part_count >= 2 && parts[1] ~ /^[0-9]+$/ && metric_count >= 2) {
          pending_id = parts[1]
          power = metrics[1]
          temperature = metrics[2]
        }
        next
      }

      if (pending_id != "" && left ~ /^[0-9]+$/ && middle ~ /:/) {
        save_row(pending_id, payload)
        pending_id = ""
      }
    }
    END {
      matched = 0
      for (i = 1; i <= wanted_count; i++) {
        if (wanted_order[i] in rows) matched++
      }

      if (matched == wanted_count) {
        for (i = 1; i <= wanted_count; i++) print rows[wanted_order[i]]
      } else if (row_count == expected_count) {
        for (i = 1; i <= row_count; i++) print rows[order[i]]
      } else {
        for (i = 1; i <= wanted_count; i++) {
          if (wanted_order[i] in rows) print rows[wanted_order[i]]
        }
      }
    }
  '
}

npu_sampler_loop() {
  local csv_path="$1" error_path="$2" timestamp snapshot

  trap 'exit 0' INT TERM
  while true; do
    timestamp="$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    if snapshot="$(collect_npu_snapshot "${timestamp}")"; then
      if [[ -n "${snapshot}" ]]; then
        printf '%s\n' "${snapshot}" >> "${csv_path}"
      else
        printf '[%s] npu-smi output did not match the expected table format\n' "${timestamp}" >> "${error_path}"
      fi
    else
      printf '[%s] npu-smi info failed\n' "${timestamp}" >> "${error_path}"
    fi
    sleep "${BENCH_NPU_SAMPLE_INTERVAL}"
  done
}

start_npu_sampler() {
  local csv_path="$1" error_path="$2"

  printf 'timestamp,npu_id,aicore_pct,power_w,temp_c,memory_used_mb,memory_total_mb,hbm_used_mb,hbm_total_mb\n' \
    > "${csv_path}"
  : > "${error_path}"

  if [[ "${BENCH_NPU_MONITOR}" == "0" ]]; then
    info "NPU telemetry monitoring is disabled (BENCH_NPU_MONITOR=0)."
    return 1
  fi
  if ! command -v npu-smi >/dev/null 2>&1; then
    warn "npu-smi is unavailable; continuing benchmark without NPU telemetry."
    return 1
  fi

  npu_sampler_loop "${csv_path}" "${error_path}" &
  NPU_SAMPLER_PID=$!
  info "Sampling NPU telemetry every ${BENCH_NPU_SAMPLE_INTERVAL}s (PID ${NPU_SAMPLER_PID})."
}

stop_npu_sampler() {
  if [[ "${NPU_SAMPLER_PID}" =~ ^[1-9][0-9]*$ ]]; then
    kill -TERM "${NPU_SAMPLER_PID}" 2>/dev/null || true
    wait "${NPU_SAMPLER_PID}" 2>/dev/null || true
  fi
  NPU_SAMPLER_PID=""
}

cancel_bench() {
  trap - INT TERM EXIT
  warn "Benchmark cancelled; stopping NPU telemetry sampler."
  stop_npu_sampler
  exit 130
}

show_npu_summary() {
  local csv_path="$1"

  awk -F, -v configured_ids="${NPU_DEVICE_IDS}" -v interval="${BENCH_NPU_SAMPLE_INTERVAL}" '
    function max(a, b) { return a > b ? a : b }
    function remember_device(id) {
      if (!(id in device_seen)) {
        device_seen[id] = 1
        discovered_order[++device_count] = id
      }
    }
    function print_device(id, avg_ai, avg_power, avg_temp, avg_mem, avg_hbm, mem_text, hbm_text) {
      if (!(id in count)) return
      avg_ai = ai_sum[id] / count[id]
      avg_power = power_sum[id] / count[id]
      avg_temp = temp_sum[id] / count[id]
      avg_mem = mem_sum[id] / count[id]
      avg_hbm = hbm_sum[id] / count[id]
      mem_text = mem_total[id] > 0 ? sprintf("%.0f/%.0f/%.0f", avg_mem, mem_max[id], mem_total[id]) : "n/a"
      hbm_text = hbm_total[id] > 0 \
        ? sprintf("%.0f/%.0f/%.0f (%.1f%%)", avg_hbm, hbm_max[id], hbm_total[id], 100 * hbm_max[id] / hbm_total[id]) \
        : "n/a"
      printf "  %-5s %-7d %6.1f/%-6.1f %8.1f%% %7.1f/%-7.1f %6.1f/%-6.1f %-14s %s\n", \
        id, count[id], avg_ai, ai_max[id], 100 * ai_below_50[id] / count[id], \
        avg_power, power_max[id], avg_temp, temp_max[id], mem_text, hbm_text
      printed[id] = 1
    }
    NR == 1 { next }
    NF >= 9 {
      id = $2
      remember_device(id)
      count[id]++
      total_rows++
      ai_sum[id] += $3
      ai_total += $3
      ai_max[id] = max(ai_max[id], $3)
      if ($3 < 50) ai_below_50[id]++
      power_sum[id] += $4
      power_max[id] = max(power_max[id], $4)
      temp_sum[id] += $5
      temp_max[id] = max(temp_max[id], $5)
      overall_temp_max = max(overall_temp_max, $5)
      mem_sum[id] += $6
      mem_max[id] = max(mem_max[id], $6)
      mem_total[id] = max(mem_total[id], $7)
      hbm_sum[id] += $8
      hbm_max[id] = max(hbm_max[id], $8)
      hbm_total[id] = max(hbm_total[id], $9)
      timestamp_count[$1]++
      timestamp_power[$1] += $4
      timestamp_hbm[$1] += $8
      timestamp_hbm_total[$1] += $9
    }
    END {
      printf "\nNPU telemetry summary:\n"
      if (total_rows == 0) {
        print "  No valid NPU telemetry samples were collected."
        exit
      }

      printf "  Sampling interval: %ss; devices: %d; rows: %d\n", interval, device_count, total_rows
      print "  NPU   Samples AICore avg/max  Samples<50  Power avg/max(W) Temp avg/max(C) Memory avg/max/total(MB) HBM avg/max/total(MB)"

      configured_count = split(configured_ids, configured_order, /[[:space:]]+/)
      configured_present = 0
      for (i = 1; i <= configured_count; i++) {
        if (configured_order[i] in count) configured_present++
      }
      if (configured_present == device_count && configured_count == device_count) {
        for (i = 1; i <= configured_count; i++) print_device(configured_order[i])
      }
      for (i = 1; i <= device_count; i++) {
        id = discovered_order[i]
        if (!(id in printed)) print_device(id)
      }

      min_device_ai = -1
      for (id in count) {
        device_ai = ai_sum[id] / count[id]
        if (min_device_ai < 0 || device_ai < min_device_ai) min_device_ai = device_ai
        if (device_ai > max_device_ai) max_device_ai = device_ai
        total_power_avg += power_sum[id] / count[id]
        total_hbm_avg += hbm_sum[id] / count[id]
        total_hbm_capacity += hbm_total[id]
      }
      for (timestamp in timestamp_count) {
        if (timestamp_count[timestamp] != device_count) continue
        complete_timestamp_count++
        total_power_peak = max(total_power_peak, timestamp_power[timestamp])
        total_hbm_peak = max(total_hbm_peak, timestamp_hbm[timestamp])
      }

      printf "  Overall AICore avg: %.1f%%; per-card average range: %.1f percentage points\n", \
        ai_total / total_rows, max_device_ai - min_device_ai
      if (complete_timestamp_count > 0) {
        printf "  Total power avg/peak: %.1f/%.1f W; maximum temperature: %.1f C\n", \
          total_power_avg, total_power_peak, overall_temp_max
      } else {
        printf "  Total power avg/peak: %.1f/n/a W; maximum temperature: %.1f C\n", \
          total_power_avg, overall_temp_max
      }
      if (total_hbm_capacity > 0) {
        if (complete_timestamp_count > 0) {
          printf "  Total HBM avg/peak/capacity: %.1f/%.1f/%.1f GiB (peak %.1f%%)\n", \
            total_hbm_avg / 1024, total_hbm_peak / 1024, total_hbm_capacity / 1024, \
            100 * total_hbm_peak / total_hbm_capacity
        } else {
          printf "  Total HBM avg/peak/capacity: %.1f/n/a/%.1f GiB\n", \
            total_hbm_avg / 1024, total_hbm_capacity / 1024
        }
      }
    }
  ' "${csv_path}"
}

show_bench_summary() {
  local log_path="$1"
  printf '\nBenchmark summary:\n'
  awk '
    BEGIN {
      labels["Successful requests"] = 1
      labels["Failed requests"] = 1
      labels["Benchmark duration (s)"] = 1
      labels["Total input tokens"] = 1
      labels["Total generated tokens"] = 1
      labels["Request throughput (req/s)"] = 1
      labels["Output token throughput (tok/s)"] = 1
      labels["Peak output token throughput (tok/s)"] = 1
      labels["Peak concurrent requests"] = 1
      labels["Total Token throughput (tok/s)"] = 1
      labels["Mean TTFT (ms)"] = 1
      labels["Median TTFT (ms)"] = 1
      labels["P99 TTFT (ms)"] = 1
      labels["Mean TPOT (ms)"] = 1
      labels["Median TPOT (ms)"] = 1
      labels["P99 TPOT (ms)"] = 1
      labels["Mean ITL (ms)"] = 1
      labels["Median ITL (ms)"] = 1
      labels["P99 ITL (ms)"] = 1
      labels["Mean E2EL (ms)"] = 1
      labels["Median E2EL (ms)"] = 1
      labels["P99 E2EL (ms)"] = 1
    }
    {
      line = $0
      gsub(/\r/, "", line)
      separator = index(line, ":")
      if (separator == 0) next
      label = substr(line, 1, separator - 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", label)
      if (label in labels) print "  " line
    }
  ' "${log_path}"
}

bench_service() {
  local managed timestamp log_path result_filename result_path bench_status thinking_json
  local label_suffix="" extra_body_json
  local npu_csv_path npu_error_path summary_path sampler_started=0

  require_command vllm
  require_command curl
  require_command tee
  require_command awk
  validate_bench_configuration
  ensure_runtime_dir

  managed="$(read_managed_process 2>/dev/null || true)"
  [[ -n "${managed}" ]] || die "Qwen3.6-27B is not running under this manager"
  load_active_endpoint
  api_ready || die "API is not ready at ${BASE_URL}/v1"

  timestamp="$(date '+%Y%m%d-%H%M%S')"
  [[ -n "${RUN_LABEL}" ]] && label_suffix="-${RUN_LABEL}"
  log_path="${RUNTIME_DIR}/bench-${timestamp}${label_suffix}.log"
  result_filename="bench-${timestamp}${label_suffix}.json"
  result_path="${RUNTIME_DIR}/${result_filename}"
  npu_csv_path="${RUNTIME_DIR}/bench-${timestamp}${label_suffix}-npu.csv"
  npu_error_path="${RUNTIME_DIR}/bench-${timestamp}${label_suffix}-npu-monitor.err"
  summary_path="${RUNTIME_DIR}/bench-${timestamp}${label_suffix}-summary.txt"
  thinking_json=false
  [[ "${BENCH_THINKING}" == "1" ]] && thinking_json=true
  extra_body_json="{\"chat_template_kwargs\":{\"enable_thinking\":${thinking_json}}}"
  if [[ -n "${BENCH_TEMPERATURE}" ]]; then
    extra_body_json="{\"temperature\":${BENCH_TEMPERATURE},\"chat_template_kwargs\":{\"enable_thinking\":${thinking_json}}}"
  fi

  BENCH_COMMAND=(
    vllm bench serve
    --backend openai-chat
    --host "${HOST}"
    --port "${PORT}"
    --endpoint /v1/chat/completions
    --model "${SERVED_MODEL_NAME}"
    --tokenizer "${MODEL_PATH}"
    --dataset-name random
    --random-input-len "${BENCH_INPUT_LEN}"
    --random-output-len "${BENCH_OUTPUT_LEN}"
    --num-prompts "${BENCH_NUM_PROMPTS}"
    --request-rate "${BENCH_REQUEST_RATE}"
    --max-concurrency "${BENCH_MAX_CONCURRENCY}"
    --seed "${SEED}"
    --trust-remote-code
    --extra-body "${extra_body_json}"
    --save-result
    --result-dir "${RUNTIME_DIR}"
    --result-filename "${result_filename}"
  )

  : > "${log_path}"
  printf 'Started: %s\nCommand:' "$(date --iso-8601=seconds)" >> "${log_path}"
  printf ' %q' "${BENCH_COMMAND[@]}" >> "${log_path}"
  printf '\n\n' >> "${log_path}"

  info "Starting serving benchmark against ${BASE_URL}/v1/chat/completions"
  info "Input/output: ${BENCH_INPUT_LEN}/${BENCH_OUTPUT_LEN}; prompts: ${BENCH_NUM_PROMPTS}; request rate: ${BENCH_REQUEST_RATE}; max concurrency: ${BENCH_MAX_CONCURRENCY}; thinking: ${BENCH_THINKING}; temperature: ${BENCH_TEMPERATURE:-model default}"

  if start_npu_sampler "${npu_csv_path}" "${npu_error_path}"; then
    sampler_started=1
  fi
  trap cancel_bench INT TERM
  trap stop_npu_sampler EXIT

  set +e
  "${BENCH_COMMAND[@]}" 2>&1 | tee -a "${log_path}"
  bench_status=${PIPESTATUS[0]}
  set -e

  stop_npu_sampler
  trap - INT TERM EXIT

  {
    show_bench_summary "${log_path}"
    if (( sampler_started == 1 )); then
      show_npu_summary "${npu_csv_path}"
    fi
  } | tee "${summary_path}"

  if [[ -s "${npu_error_path}" ]]; then
    warn "Some NPU samples failed; details: ${npu_error_path}"
  else
    rm -f -- "${npu_error_path}"
  fi
  if (( sampler_started == 0 )); then
    rm -f -- "${npu_csv_path}"
  fi

  if (( bench_status != 0 )); then
    error "vllm bench failed with exit code ${bench_status}."
    error "Raw output: ${log_path}"
    error "Partial summary: ${summary_path}"
    return "${bench_status}"
  fi

  info "Raw output: ${log_path}"
  [[ -f "${result_path}" ]] && info "JSON result: ${result_path}"
  [[ -f "${npu_csv_path}" && ${sampler_started} -eq 1 ]] && info "NPU telemetry CSV: ${npu_csv_path}"
  info "Combined summary: ${summary_path}"
  cleanup_old_bench_files
}

check_service() {
  run_preflight 1
  info "Environment check passed."
  info "Model path: ${MODEL_PATH}"
  info "NPU devices: ${NPU_DEVICE_IDS}"
  info "Parallelism: TP=${TP_SIZE}, DP=${DP_SIZE}"
  info "API servers: ${API_SERVER_COUNT}"
  info "Sequence capacity: ${MAX_NUM_SEQS} per DP replica; approximately $((MAX_NUM_SEQS * DP_SIZE)) total running requests"
  info "API: ${BASE_URL}/v1"
  info "Persistent runtime directory: ${RUNTIME_DIR}"
  info "Container state directory: ${STATE_DIR}"
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
  require_vllm_ascend_container
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
    metrics)
      (( $# <= 1 )) || die "Usage: metrics [-f]"
      metrics_service "${1:-}"
      ;;
    test)
      (( $# == 0 )) || die "Usage: test"
      test_service
      ;;
    bench)
      (( $# == 0 )) || die "Usage: bench"
      bench_service
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
