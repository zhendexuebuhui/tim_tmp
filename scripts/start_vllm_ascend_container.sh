export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0rc1-openeuler

docker run \
  --name vllm-ascend-env \
  --shm-size=32g \
  --net=host \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci5 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data/.cache:/root/.cache \
  -v /data/models:/root/models \
  -itd "$IMAGE" bash