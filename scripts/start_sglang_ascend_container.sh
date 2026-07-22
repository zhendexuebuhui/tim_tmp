#!/usr/bin/env bash

set -Eeuo pipefail

IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.13.post1-cann9.0.0-910b}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-ascend-env}"
SHM_SIZE="${SHM_SIZE:-64g}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
CACHE_ROOT="${CACHE_ROOT:-/data/.cache}"
NPU_DEVICE_IDS="${NPU_DEVICE_IDS:-2 3 5 7}"
VERIFY_AFTER_CREATE="${VERIFY_AFTER_CREATE:-1}"

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not installed"
[[ -d "${MODEL_ROOT}" ]] || die "Model root does not exist: ${MODEL_ROOT}"
mkdir -p -- "${CACHE_ROOT}"
[[ -d "${CACHE_ROOT}" ]] || die "Cache root does not exist: ${CACHE_ROOT}"
[[ "${VERIFY_AFTER_CREATE}" == "0" || "${VERIFY_AFTER_CREATE}" == "1" ]] \
  || die "VERIFY_AFTER_CREATE must be 0 or 1"

docker image inspect "${IMAGE}" >/dev/null 2>&1 \
  || die "Docker image is not available locally: ${IMAGE}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  state="$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}")"
  die "Container ${CONTAINER_NAME} already exists (state: ${state}). Use 'docker start ${CONTAINER_NAME}' or remove it explicitly before recreating it."
fi

DOCKER_ARGS=(
  docker run
  --name "${CONTAINER_NAME}"
  --shm-size="${SHM_SIZE}"
  --net=host
  --ulimit memlock=-1:-1
)

count=0
for id in ${NPU_DEVICE_IDS}; do
  [[ "${id}" =~ ^[0-9]+$ ]] || die "Invalid NPU device id: ${id}"
  [[ -e "/dev/davinci${id}" ]] || die "Missing NPU device: /dev/davinci${id}"
  DOCKER_ARGS+=(--device "/dev/davinci${id}")
  count=$((count + 1))
done
(( count > 0 )) || die "NPU_DEVICE_IDS is empty"

for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "${device}" ]] || die "Missing required Ascend device: ${device}"
  DOCKER_ARGS+=(--device "${device}")
done

REQUIRED_MOUNTS=(
  /usr/local/dcmi
  /usr/local/Ascend/driver/tools/hccn_tool
  /usr/local/bin/npu-smi
  /usr/local/Ascend/driver/lib64
  /usr/local/Ascend/driver/version.info
  /etc/ascend_install.info
)
for path in "${REQUIRED_MOUNTS[@]}"; do
  [[ -e "${path}" ]] || die "Missing required host path: ${path}"
done

DOCKER_ARGS+=(
  -v /usr/local/dcmi:/usr/local/dcmi
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v "${CACHE_ROOT}":/root/.cache
  -v "${MODEL_ROOT}":/root/models
  -itd
  "${IMAGE}"
  bash
)

info "Creating ${CONTAINER_NAME} from ${IMAGE}"
info "NPU devices: ${NPU_DEVICE_IDS}; shared memory: ${SHM_SIZE}"
"${DOCKER_ARGS[@]}"

if [[ "${VERIFY_AFTER_CREATE}" == "1" ]]; then
  info "Checking SGLang, torch-npu and visible NPU devices..."
  if ! docker exec "${CONTAINER_NAME}" bash -lc "python3 - <<'PY'
import torch
import torch_npu
import sglang

print('sglang:', getattr(sglang, '__version__', 'unknown'))
print('torch:', torch.__version__)
print('torch_npu:', getattr(torch_npu, '__version__', 'unknown'))
print('npu_available:', torch.npu.is_available())
print('npu_count:', torch.npu.device_count())
if not torch.npu.is_available():
    raise SystemExit('Ascend NPU is not available in the container')
if torch.npu.device_count() != ${count}:
    raise SystemExit(f'Expected ${count} NPUs, found {torch.npu.device_count()}')
PY"; then
    warn "Environment verification failed. The container was retained for diagnosis."
    exit 1
  fi
fi

info "Container created. It is persistent: docker stop will not delete it."
printf 'Enter: docker exec -it %q bash\n' "${CONTAINER_NAME}"
printf 'Repository: cd /root/models/tim_tmp\n'
printf 'Next: bash ./scripts/manage_minimax_m2_7.sh check\n'
