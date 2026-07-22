#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_VARIANT="${MODEL_VARIANT:-quanttrio}"
MODEL_PATH="${MODEL_PATH:-/root/models/MiniMax-M2.7-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minimax-m2.7}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"
DTYPE="${DTYPE:-half}"
MODEL_IMPL="${MODEL_IMPL:-sglang}"
QUANTIZATION_OVERRIDE="${QUANTIZATION_OVERRIDE:-}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
START_TIMEOUT="${START_TIMEOUT:-1800}"
STOP_TIMEOUT="${STOP_TIMEOUT:-90}"
NPU_DEVICE_IDS="${NPU_DEVICE_IDS:-2 3 5 7}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/runtime/minimax-m2.7}"
VERIFY_REQUESTS="${VERIFY_REQUESTS:-20}"
VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-64}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-2048}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-256}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-8}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-1}"
BENCH_NPU_MONITOR="${BENCH_NPU_MONITOR:-1}"
BENCH_NPU_SAMPLE_INTERVAL="${BENCH_NPU_SAMPLE_INTERVAL:-1}"
BENCH_MATRIX_INPUTS="${BENCH_MATRIX_INPUTS:-4096 16384 32768}"
BENCH_MATRIX_CONCURRENCIES="${BENCH_MATRIX_CONCURRENCIES:-1 2 4}"

MAX_MODEL_LEN_OVERRIDE="${MAX_MODEL_LEN:-}"
MAX_RUNNING_REQUESTS_OVERRIDE="${MAX_RUNNING_REQUESTS:-}"
MEM_FRACTION_STATIC_OVERRIDE="${MEM_FRACTION_STATIC:-}"
CHUNKED_PREFILL_SIZE_OVERRIDE="${CHUNKED_PREFILL_SIZE:-}"

PID_FILE="${RUNTIME_DIR}/server.pid"
PROFILE_FILE="${RUNTIME_DIR}/profile"
LOG_POINTER="${RUNTIME_DIR}/current_log"
STATE_FILE="${RUNTIME_DIR}/service_state"
BASE_URL="http://${HOST}:${PORT}"
MAX_MODEL_LEN=""
MAX_RUNNING_REQUESTS=""
MEM_FRACTION_STATIC=""
CHUNKED_PREFILL_SIZE=""
NPU_SAMPLER_PID=""

info(){ printf '[INFO] %s\n' "$*"; }
warn(){ printf '[WARN] %s\n' "$*" >&2; }
die(){ printf '[ERROR] %s\n' "$*" >&2; exit 1; }
positive_int(){ [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
require(){ command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

usage(){ cat <<'USAGE'
Manage MiniMax-M2.7 on SGLang-Ascend. Run inside sglang-ascend-env.

Usage:
  manage_minimax_m2_7.sh check
  manage_minimax_m2_7.sh start [safe|optimized]
  manage_minimax_m2_7.sh stop
  manage_minimax_m2_7.sh restart [safe|optimized]
  manage_minimax_m2_7.sh status
  manage_minimax_m2_7.sh logs [-f]
  manage_minimax_m2_7.sh test
  manage_minimax_m2_7.sh verify
  manage_minimax_m2_7.sh bench
  manage_minimax_m2_7.sh bench-matrix

Default checkpoint:
  MODEL_VARIANT=quanttrio
  MODEL_PATH=/root/models/MiniMax-M2.7-AWQ

Fallback:
  MODEL_VARIANT=cyankiwi MODEL_PATH=/root/models/MiniMax-M2.7-AWQ-4bit \
    bash ./scripts/manage_minimax_m2_7.sh check

Pre-quantized checkpoints are auto-detected from config.json. Set
QUANTIZATION_OVERRIDE only for loader diagnosis.
USAGE
}

apply_profile(){
  case "$1" in
    safe)
      MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-4096}"
      MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS_OVERRIDE:-1}"
      MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC_OVERRIDE:-0.75}"
      CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE_OVERRIDE:-4096}" ;;
    optimized)
      MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-16384}"
      MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS_OVERRIDE:-4}"
      MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC_OVERRIDE:-0.82}"
      CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE_OVERRIDE:-4096}" ;;
    *) die "Unknown profile: $1" ;;
  esac
}

validate(){
  positive_int "${PORT}" && ((PORT <= 65535)) || die "Invalid PORT=${PORT}"
  positive_int "${TP_SIZE}" || die "Invalid TP_SIZE=${TP_SIZE}"
  positive_int "${MAX_MODEL_LEN}" || die "Invalid MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  positive_int "${MAX_RUNNING_REQUESTS}" || die "Invalid MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
  positive_int "${CHUNKED_PREFILL_SIZE}" || die "Invalid CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}"
  [[ "${MODEL_VARIANT}" == quanttrio || "${MODEL_VARIANT}" == cyankiwi ]] || die "Invalid MODEL_VARIANT"
  [[ "${OFFLINE_MODE}" == 0 || "${OFFLINE_MODE}" == 1 ]] || die "OFFLINE_MODE must be 0 or 1"
  [[ "${MEM_FRACTION_STATIC}" =~ ^0([.][0-9]+)?$|^1([.]0+)?$ ]] || die "Invalid MEM_FRACTION_STATIC"
}

load_state(){
  local -a s=()
  [[ -r "${STATE_FILE}" ]] || return 0
  mapfile -t s < "${STATE_FILE}"
  (( ${#s[@]} >= 6 )) || return 0
  HOST="${s[0]}"; PORT="${s[1]}"; SERVED_MODEL_NAME="${s[2]}"
  MAX_MODEL_LEN="${s[3]}"; MODEL_PATH="${s[4]}"; MODEL_VARIANT="${s[5]}"
  BASE_URL="http://${HOST}:${PORT}"
}

pid_alive(){
  [[ -r "${PID_FILE}" ]] || return 1
  local pid; pid="$(<"${PID_FILE}")"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q 'sglang.launch_server'
}

api_ready(){ curl -fsS --max-time 5 "${BASE_URL}/v1/models" >/dev/null 2>&1; }
current_log(){ [[ -r "${LOG_POINTER}" ]] && cat "${LOG_POINTER}"; }
show_log(){ local p; p="$(current_log 2>/dev/null || true)"; [[ -f "${p}" ]] && tail -n "${1:-100}" "${p}" >&2 || true; }
help_has(){ python3 -m sglang.launch_server --help 2>/dev/null | grep -q -- "$1"; }

check_model(){
  [[ -d "${MODEL_PATH}" ]] || die "Model directory missing: ${MODEL_PATH}"
  [[ -r "${MODEL_PATH}/config.json" ]] || die "Missing config.json"
  [[ -r "${MODEL_PATH}/tokenizer_config.json" ]] || die "Missing tokenizer_config.json"
  find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q . || die "No weight files found"
  MODEL_PATH_ENV="${MODEL_PATH}" MODEL_VARIANT_ENV="${MODEL_VARIANT}" python3 - <<'PY'
import json, os
from pathlib import Path
c=json.loads((Path(os.environ['MODEL_PATH_ENV'])/'config.json').read_text())
a=c.get('architectures') or []
if 'MiniMaxM2ForCausalLM' not in a: raise SystemExit(f'Unexpected architectures: {a}')
q=c.get('quantization_config') or {}
m=q.get('quant_method')
e='awq' if os.environ['MODEL_VARIANT_ENV']=='quanttrio' else 'compressed-tensors'
if m!=e: raise SystemExit(f'Expected quant_method={e!r}, found {m!r}')
print('architectures:', a)
print('quantization_config:', json.dumps(q, ensure_ascii=False, sort_keys=True))
PY
}

check_runtime(){
  local id count=0
  for id in ${NPU_DEVICE_IDS}; do [[ -e "/dev/davinci${id}" ]] || die "Missing /dev/davinci${id}"; count=$((count+1)); done
  (( count == TP_SIZE )) || die "NPU count ${count} != TP_SIZE ${TP_SIZE}"
  EXPECTED_NPU_COUNT="${TP_SIZE}" python3 - <<'PY'
import inspect, os, torch, torch_npu, sglang
import sglang.srt.models.minimax_m2 as minimax
print('sglang:', getattr(sglang,'__version__','unknown'))
print('torch:', torch.__version__)
print('torch_npu:', getattr(torch_npu,'__version__','unknown'))
print('npu_available:', torch.npu.is_available())
print('npu_count:', torch.npu.device_count())
print('minimax_module:', inspect.getfile(minimax))
if not torch.npu.is_available(): raise SystemExit('NPU unavailable')
if torch.npu.device_count()!=int(os.environ['EXPECTED_NPU_COUNT']): raise SystemExit('Unexpected logical NPU count')
if not hasattr(minimax,'MiniMaxM2ForCausalLM'): raise SystemExit('Native MiniMaxM2ForCausalLM missing')
PY
}

preflight(){
  mkdir -p "${RUNTIME_DIR}"
  require python3; require curl; require setsid; require awk; require find; require tail
  check_model; check_runtime
}

configure_env(){
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
  if [[ "${OFFLINE_MODE}" == 1 ]]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; else unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true; fi
}

build_command(){
  local profile="$1"
  CMD=(python3 -m sglang.launch_server
    --model-path "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}" --port "${PORT}" --tp "${TP_SIZE}" --dtype "${DTYPE}"
    --context-length "${MAX_MODEL_LEN}" --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}" --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --trust-remote-code)
  [[ -n "${MODEL_IMPL}" ]] && help_has --model-impl && CMD+=(--model-impl "${MODEL_IMPL}")
  [[ -n "${QUANTIZATION_OVERRIDE}" ]] && CMD+=(--quantization "${QUANTIZATION_OVERRIDE}")
  if [[ "${profile}" == safe ]]; then
    help_has --disable-radix-cache && CMD+=(--disable-radix-cache) || warn "--disable-radix-cache unavailable"
    help_has --disable-cuda-graph && CMD+=(--disable-cuda-graph) || warn "--disable-cuda-graph unavailable"
  fi
}

start_service(){
  local profile="${1:-safe}" ts log pid elapsed=0
  apply_profile "${profile}"; validate; preflight
  pid_alive && die "Managed service already running (PID $(<"${PID_FILE}"))"
  api_ready && die "Port ${PORT} already serves an unmanaged API"
  configure_env; build_command "${profile}"
  ts="$(date '+%Y%m%d-%H%M%S')"; log="${RUNTIME_DIR}/serve-${ts}-${profile}.log"; :>"${log}"
  printf 'Profile: %s\nStarted: %s\nCommand:' "${profile}" "$(date --iso-8601=seconds)" >>"${log}"; printf ' %q' "${CMD[@]}" >>"${log}"; printf '\n\n' >>"${log}"
  info "Starting MiniMax-M2.7 (${profile}, ${MODEL_VARIANT})"
  setsid "${CMD[@]}" >>"${log}" 2>&1 </dev/null & pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"; printf '%s\n' "${profile}" >"${PROFILE_FILE}"; printf '%s\n' "${log}" >"${LOG_POINTER}"
  printf '%s\n%s\n%s\n%s\n%s\n%s\n' "${HOST}" "${PORT}" "${SERVED_MODEL_NAME}" "${MAX_MODEL_LEN}" "${MODEL_PATH}" "${MODEL_VARIANT}" >"${STATE_FILE}"
  info "Waiting up to ${START_TIMEOUT}s for ${BASE_URL}/v1/models"
  while ((elapsed<START_TIMEOUT)); do
    api_ready && { info "Ready: ${BASE_URL}/v1"; info "Log: ${log}"; return 0; }
    pid_alive || { show_log; rm -f "${PID_FILE}"; die "SGLang exited before readiness"; }
    sleep 2; elapsed=$((elapsed+2)); ((elapsed%30==0)) && info "Still loading (${elapsed}s)"
  done
  show_log; stop_service || true; die "Startup timed out"
}

stop_service(){
  load_state
  if ! pid_alive; then rm -f "${PID_FILE}"; info "MiniMax-M2.7 is not running"; return 0; fi
  local pid waited=0; pid="$(<"${PID_FILE}")"; info "Stopping PID ${pid}"
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  while kill -0 "${pid}" 2>/dev/null; do
    ((waited>=STOP_TIMEOUT)) && { warn "Forcing shutdown"; kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true; break; }
    sleep 1; waited=$((waited+1))
  done
  rm -f "${PID_FILE}"; info "Stopped"
}

status_service(){
  load_state
  if pid_alive; then local state=loading; api_ready && state=ready; printf 'Status: %s\nPID: %s\nProfile: %s\nVariant: %s\nModel path: %s\nContext: %s\nAPI: %s/v1\n' "$state" "$(<"${PID_FILE}")" "$(cat "${PROFILE_FILE}" 2>/dev/null || echo unknown)" "${MODEL_VARIANT}" "${MODEL_PATH}" "${MAX_MODEL_LEN}" "${BASE_URL}"; else printf 'Status: stopped\n'; return 1; fi
}

logs_service(){ local p; p="$(current_log 2>/dev/null || true)"; [[ -f "$p" ]] || die "No service log"; [[ "${1:-}" == -f ]] && tail -n100 -f "$p" || tail -n100 "$p"; }
require_ready(){ pid_alive || die "Managed service is not running"; load_state; api_ready || die "API not ready"; }

request_once(){
  curl -fsS --max-time 300 "${BASE_URL}/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @- <<JSON
{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"$1"}],"temperature":0,"max_tokens":$2}
JSON
}

validate_response(){ RESPONSE_JSON="$1" python3 - <<'PY'
import json, os
p=json.loads(os.environ['RESPONSE_JSON']); c=p.get('choices') or []
if not c: raise SystemExit('No choices returned')
m=c[0].get('message') or {}; t=(m.get('reasoning_content') or '')+(m.get('content') or '')
if not t.strip(): raise SystemExit('Empty output')
if 'nan' in t.lower(): raise SystemExit('NaN-like output')
print(t.replace('\n',' ')[:500])
PY
}

test_service(){ require_ready; local r; r="$(request_once 'Reply with exactly: OK' 32)"; validate_response "$r"; info "Smoke test passed"; }
verify_service(){
  require_ready; positive_int "${VERIFY_REQUESTS}" || die "Invalid VERIFY_REQUESTS"; local ts log i r
  ts="$(date '+%Y%m%d-%H%M%S')"; log="${RUNTIME_DIR}/verify-${ts}.log"; :>"$log"
  for ((i=1;i<=VERIFY_REQUESTS;i++)); do r="$(request_once 'Calculate 1234 + 5678. Return only the number.' "${VERIFY_MAX_TOKENS}")"; validate_response "$r" >>"$log"; api_ready || die "API failed after request ${i}"; info "Verify ${i}/${VERIFY_REQUESTS}"; done
  info "Verification passed: ${log}"
}

npu_loop(){ local f="$1"; trap 'exit 0' INT TERM; while true; do printf '===== %s =====\n' "$(date --iso-8601=ns)" >>"$f"; npu-smi info >>"$f" 2>&1 || true; sleep "${BENCH_NPU_SAMPLE_INTERVAL}"; done; }
stop_sampler(){ [[ "${NPU_SAMPLER_PID}" =~ ^[1-9][0-9]*$ ]] && { kill "${NPU_SAMPLER_PID}" 2>/dev/null || true; wait "${NPU_SAMPLER_PID}" 2>/dev/null || true; }; NPU_SAMPLER_PID=""; }

bench_service(){
  require_ready; positive_int "${BENCH_INPUT_LEN}" && positive_int "${BENCH_OUTPUT_LEN}" && positive_int "${BENCH_MAX_CONCURRENCY}" || die "Invalid benchmark lengths/concurrency"
  ((BENCH_INPUT_LEN+BENCH_OUTPUT_LEN<=MAX_MODEL_LEN)) || die "Benchmark exceeds active context ${MAX_MODEL_LEN}"
  local ts p rc=0; ts="$(date '+%Y%m%d-%H%M%S')"; p="${RUNTIME_DIR}/bench-${ts}-in${BENCH_INPUT_LEN}-out${BENCH_OUTPUT_LEN}-c${BENCH_MAX_CONCURRENCY}"
  if [[ "${BENCH_NPU_MONITOR}" == 1 ]] && command -v npu-smi >/dev/null; then npu_loop "${p}-npu.log" & NPU_SAMPLER_PID=$!; fi
  trap stop_sampler EXIT INT TERM
  python3 -m sglang.bench_serving --backend sglang-oai-chat --host "${HOST}" --port "${PORT}" \
    --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --dataset-name random --random-input-len "${BENCH_INPUT_LEN}" --random-output-len "${BENCH_OUTPUT_LEN}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" --request-rate "${BENCH_REQUEST_RATE}" --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --output-file "${p}.jsonl" --output-details 2>&1 | tee "${p}.log" || rc=${PIPESTATUS[0]}
  stop_sampler; trap - EXIT INT TERM; ((rc==0)) || die "Benchmark failed: ${p}.log"; info "Benchmark: ${p}.log"
}

bench_matrix(){ require_ready; local i c; for i in ${BENCH_MATRIX_INPUTS}; do for c in ${BENCH_MATRIX_CONCURRENCIES}; do if ((i+BENCH_OUTPUT_LEN>MAX_MODEL_LEN)); then warn "Skip input=${i}, concurrency=${c}: context=${MAX_MODEL_LEN}"; else BENCH_INPUT_LEN="$i" BENCH_MAX_CONCURRENCY="$c" bench_service; fi; done; done; }

case "${1:-}" in
  check) apply_profile safe; validate; preflight; info "Preflight passed: ${MODEL_VARIANT} ${MODEL_PATH}" ;;
  start) start_service "${2:-safe}" ;;
  stop) stop_service ;;
  restart) profile="${2:-$(cat "${PROFILE_FILE}" 2>/dev/null || echo safe)}"; load_state; stop_service; start_service "$profile" ;;
  status) status_service ;;
  logs) logs_service "${2:-}" ;;
  test) test_service ;;
  verify) verify_service ;;
  bench) bench_service ;;
  bench-matrix) bench_matrix ;;
  help|--help|-h|'') usage ;;
  *) usage; die "Unknown command: ${1}" ;;
esac
