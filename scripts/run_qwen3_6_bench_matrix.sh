#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MANAGER="${MANAGER:-${SCRIPT_DIR}/manage_qwen3_6_27b.sh}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/runtime/qwen3.6-27b}"
MATRIX_RUN_ID="${MATRIX_RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"

usage() {
  cat <<'EOF'
Run the Qwen3.6-27B benchmark matrix.

Usage:
  run_qwen3_6_bench_matrix.sh A|B|C|D|E|F|G|H|I|J|all

Historical rounds:
  A  TP=4, DP=1, max-seqs=8,  benchmark concurrency=8
  B  TP=4, DP=1, max-seqs=16, benchmark concurrency=16
  C  Same as B, with enable_reduce_sample=true
  D  TP=2, DP=2, max-seqs=8 per DP replica, benchmark concurrency=8
  E  Same as D, with benchmark concurrency=16

New production-baseline rounds:
  F  TP=2, DP=2, global service concurrency=8, API servers=2, benchmark concurrency=8
  G  Same as F, benchmark concurrency=4
  H  Same as F, API servers=1
  I  Same as F, API servers=4
  J  Same as F, API servers=8

Every round uses 32768 input tokens, 1024 output tokens, 64 prompts,
request-rate=inf, Thinking disabled, and temperature=0. Each round starts
from a stopped service and stops it after success or failure. "all" runs
only the new rounds F -> G -> H -> I -> J and stops on the first failure.
A-E remain available for explicit historical reruns.
EOF
}

info() {
  printf '[MATRIX] %s\n' "$*"
}

die() {
  printf '[MATRIX ERROR] %s\n' "$*" >&2
  exit 1
}

configure_round() {
  local round="$1"

  SERVICE_MAX_CONCURRENCY=8
  API_SERVER_COUNT=2

  case "${round}" in
    A)
      TP_SIZE=4
      DP_SIZE=1
      MAX_NUM_SEQS=8
      BENCH_MAX_CONCURRENCY=8
      ENABLE_REDUCE_SAMPLE=0
      ;;
    B)
      TP_SIZE=4
      DP_SIZE=1
      MAX_NUM_SEQS=16
      BENCH_MAX_CONCURRENCY=16
      ENABLE_REDUCE_SAMPLE=0
      ;;
    C)
      TP_SIZE=4
      DP_SIZE=1
      MAX_NUM_SEQS=16
      BENCH_MAX_CONCURRENCY=16
      ENABLE_REDUCE_SAMPLE=1
      ;;
    D)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=8
      BENCH_MAX_CONCURRENCY=8
      ENABLE_REDUCE_SAMPLE=0
      ;;
    E)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=8
      BENCH_MAX_CONCURRENCY=16
      ENABLE_REDUCE_SAMPLE=0
      ;;
    F)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=
      BENCH_MAX_CONCURRENCY=8
      ENABLE_REDUCE_SAMPLE=1
      ;;
    G)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=
      BENCH_MAX_CONCURRENCY=4
      ENABLE_REDUCE_SAMPLE=1
      ;;
    H)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=
      BENCH_MAX_CONCURRENCY=8
      ENABLE_REDUCE_SAMPLE=1
      API_SERVER_COUNT=1
      ;;
    I)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=
      BENCH_MAX_CONCURRENCY=8
      ENABLE_REDUCE_SAMPLE=1
      API_SERVER_COUNT=4
      ;;
    J)
      TP_SIZE=2
      DP_SIZE=2
      MAX_NUM_SEQS=
      BENCH_MAX_CONCURRENCY=8
      ENABLE_REDUCE_SAMPLE=1
      API_SERVER_COUNT=8
      ;;
    *)
      die "Unknown round '${round}'. Expected A, B, C, D, E, F, G, H, I, J, or all."
      ;;
  esac

  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    EFFECTIVE_MAX_NUM_SEQS="${MAX_NUM_SEQS}"
  else
    (( SERVICE_MAX_CONCURRENCY % DP_SIZE == 0 )) \
      || die "SERVICE_MAX_CONCURRENCY=${SERVICE_MAX_CONCURRENCY} must be divisible by DP_SIZE=${DP_SIZE}"
    EFFECTIVE_MAX_NUM_SEQS=$((SERVICE_MAX_CONCURRENCY / DP_SIZE))
  fi

  RUN_LABEL="matrix-${MATRIX_RUN_ID}-${round}"
}

run_manager() {
  env \
    RUNTIME_DIR="${RUNTIME_DIR}" \
    RUN_LABEL="${RUN_LABEL}" \
    TP_SIZE="${TP_SIZE}" \
    DP_SIZE="${DP_SIZE}" \
    MAX_MODEL_LEN=262144 \
    SERVICE_MAX_CONCURRENCY="${SERVICE_MAX_CONCURRENCY}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    MAX_NUM_BATCHED_TOKENS=16384 \
    API_SERVER_COUNT="${API_SERVER_COUNT}" \
    ENABLE_REDUCE_SAMPLE="${ENABLE_REDUCE_SAMPLE}" \
    BENCH_INPUT_LEN=32768 \
    BENCH_OUTPUT_LEN=1024 \
    BENCH_NUM_PROMPTS=64 \
    BENCH_REQUEST_RATE=inf \
    BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY}" \
    BENCH_THINKING=0 \
    BENCH_TEMPERATURE=0 \
    bash "${MANAGER}" "$@"
}

write_manifest_header() {
  local round="$1" manifest_path="$2"

  {
    printf 'Matrix run: %s\n' "${MATRIX_RUN_ID}"
    printf 'Round: %s\n' "${round}"
    printf 'Started: %s\n' "$(date --iso-8601=seconds)"
    printf 'Profile: optimized\n'
    printf 'TP_SIZE: %s\n' "${TP_SIZE}"
    printf 'DP_SIZE: %s\n' "${DP_SIZE}"
    printf 'MAX_MODEL_LEN: 262144\n'
    printf 'SERVICE_MAX_CONCURRENCY: %s\n' "${SERVICE_MAX_CONCURRENCY}"
    printf 'MAX_NUM_SEQS: %s\n' "${EFFECTIVE_MAX_NUM_SEQS}"
    printf 'MAX_NUM_SEQS_SOURCE: %s\n' "$([[ -n "${MAX_NUM_SEQS}" ]] && printf explicit || printf derived)"
    printf 'MAX_NUM_BATCHED_TOKENS: 16384\n'
    printf 'API_SERVER_COUNT: %s\n' "${API_SERVER_COUNT}"
    printf 'ENABLE_REDUCE_SAMPLE: %s\n' "${ENABLE_REDUCE_SAMPLE}"
    printf 'BENCH_INPUT_LEN: 32768\n'
    printf 'BENCH_OUTPUT_LEN: 1024\n'
    printf 'BENCH_NUM_PROMPTS: 64\n'
    printf 'BENCH_REQUEST_RATE: inf\n'
    printf 'BENCH_MAX_CONCURRENCY: %s\n' "${BENCH_MAX_CONCURRENCY}"
    printf 'BENCH_THINKING: 0\n'
    printf 'BENCH_TEMPERATURE: 0\n'
  } > "${manifest_path}"
}

run_round_inner() (
  local round="$1" manifest_path="$2" exit_status stop_status

  set -Eeuo pipefail

  cleanup() {
    exit_status=$?
    trap - EXIT INT TERM
    set +e
    run_manager stop
    stop_status=$?
    if (( exit_status == 0 && stop_status != 0 )); then
      exit_status=${stop_status}
    fi
    {
      printf 'Finished: %s\n' "$(date --iso-8601=seconds)"
      printf 'Exit status: %s\n' "${exit_status}"
    } >> "${manifest_path}"
    exit "${exit_status}"
  }

  interrupted() {
    exit 130
  }

  trap cleanup EXIT
  trap interrupted INT TERM

  info "Round ${round}: checking the current environment and port ownership."
  run_manager check || exit $?
  info "Round ${round}: stopping any managed Qwen3.6 service."
  run_manager stop || exit $?
  info "Round ${round}: rechecking the clean experiment configuration."
  run_manager check || exit $?
  info "Round ${round}: starting optimized TP=${TP_SIZE}/DP=${DP_SIZE}, max-seqs=${EFFECTIVE_MAX_NUM_SEQS}/DP, API servers=${API_SERVER_COUNT} service."
  run_manager start optimized || exit $?
  info "Round ${round}: running the inference smoke test."
  run_manager test || exit $?
  info "Round ${round}: running the single formal benchmark."
  run_manager bench || exit $?
  info "Round ${round}: benchmark completed."
)

run_round() {
  local round="$1" orchestration_log manifest_path status

  configure_round "${round}"
  mkdir -p -- "${RUNTIME_DIR}"
  [[ -w "${RUNTIME_DIR}" ]] || die "Runtime directory is not writable: ${RUNTIME_DIR}"
  orchestration_log="${RUNTIME_DIR}/matrix-${MATRIX_RUN_ID}-${round}.log"
  manifest_path="${RUNTIME_DIR}/matrix-${MATRIX_RUN_ID}-${round}-manifest.txt"
  write_manifest_header "${round}" "${manifest_path}"

  set +e
  run_round_inner "${round}" "${manifest_path}" 2>&1 | tee "${orchestration_log}"
  status=${PIPESTATUS[0]}
  set -e

  if (( status != 0 )); then
    printf '[MATRIX ERROR] Round %s failed with exit status %s. See %s and %s.\n' \
      "${round}" "${status}" "${orchestration_log}" "${manifest_path}" >&2
    return "${status}"
  fi

  info "Round ${round} succeeded. Orchestration log: ${orchestration_log}"
}

main() {
  local selection="${1:-}"
  local round

  (( $# == 1 )) || {
    usage >&2
    exit 2
  }
  [[ -r "${MANAGER}" ]] || die "Management script is not readable: ${MANAGER}"
  command -v tee >/dev/null 2>&1 || die "Required command not found: tee"

  case "${selection}" in
    A|B|C|D|E|F|G|H|I|J)
      run_round "${selection}"
      ;;
    a|b|c|d|e|f|g|h|i|j)
      run_round "${selection^^}"
      ;;
    all)
      for round in F G H I J; do
        run_round "${round}" || exit $?
      done
      ;;
    help|--help|-h)
      usage
      ;;
    *)
      usage >&2
      die "Unknown selection '${selection}'."
      ;;
  esac
}

main "$@"
