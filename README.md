# tim_tmp

## 当前目录映射

宿主机的 `/data/models` 挂载到容器的 `/root/models`，因此：

- 宿主机仓库：`/data/models/tim_tmp`
- 容器内仓库：`/root/models/tim_tmp`
- 容器内模型：`/root/models/Qwen3.6-27B`

容器仍由 [`scripts/start_vllm_ascend_container.sh`](scripts/start_vllm_ascend_container.sh) 创建。该脚本使用 `docker run -itd`，停止容器不会自动删除它。执行 `docker start vllm-ascend-env` 重新启动容器后，需要再次显式启动模型服务。

## 快速开始

进入已经启动的 `vllm-ascend-env` 容器：

```bash
docker exec -it vllm-ascend-env bash
cd /root/models/tim_tmp
```

先执行预检，再启动默认的 `optimized` 配置：

```bash
./scripts/manage_qwen3_6_27b.sh check
./scripts/manage_qwen3_6_27b.sh start
```

启动完成后，API 地址为：

```text
http://127.0.0.1:8000/v1
```

服务只监听本机回环地址，不设置 API Key。宿主机和使用 host 网络的当前容器可以访问，其他服务器不能直接访问。

## 管理命令

```bash
# 环境、模型、NPU 和端口预检
./scripts/manage_qwen3_6_27b.sh check

# 启动；不指定模式时默认 optimized
./scripts/manage_qwen3_6_27b.sh start
./scripts/manage_qwen3_6_27b.sh start optimized
./scripts/manage_qwen3_6_27b.sh start safe

# 停止与重启
./scripts/manage_qwen3_6_27b.sh stop
./scripts/manage_qwen3_6_27b.sh restart
./scripts/manage_qwen3_6_27b.sh restart safe

# 状态、日志与推理测试
./scripts/manage_qwen3_6_27b.sh status
./scripts/manage_qwen3_6_27b.sh logs
./scripts/manage_qwen3_6_27b.sh logs -f
./scripts/manage_qwen3_6_27b.sh test

# 内置帮助
./scripts/manage_qwen3_6_27b.sh --help
```

`restart` 不指定模式时会沿用上一次的模式。`status` 会明确显示当前是 `optimized` 还是 `safe`。

## 两种运行模式

| 功能 | optimized（默认） | safe |
|---|---:|---:|
| MTP，`num_speculative_tokens=3` | 开启 | 关闭 |
| `FULL_DECODE_ONLY` 图模式 | 开启 | 关闭 |
| CPU binding | 开启 | 关闭 |
| Async scheduling | 开启 | 关闭 |
| Prefix caching | 开启 | 关闭 |

`optimized` 启动失败时不会自动回退到 `safe`。请先通过 `logs` 查看原因，再显式执行：

```bash
./scripts/manage_qwen3_6_27b.sh start safe
```

Qwen3.6-27B 使用混合 KV Cache。vLLM-Ascend 官方文档指出，启用 Prefix Caching 时有效块大小可能较大，短重复前缀不一定能够命中缓存。本配置仍在 `optimized` 中开启它，适合重复公共前缀超过约 2048 Token 的 Agent 工作负载；测试真实流量时应关注命中率和显存占用。

## 默认 vLLM 参数

| 参数 | 默认值 |
|---|---|
| 模型路径 | `/root/models/Qwen3.6-27B` |
| Served model name | `qwen3.6-27b` |
| Host / Port | `127.0.0.1:8000` |
| Tensor / Data parallel | `TP=4` / `DP=1` |
| Dtype | `bfloat16` |
| Max model length | `262144` |
| Max sequences | `8` |
| Max batched tokens | `16384` |
| NPU memory utilization | `0.90` |
| Tool-call parser | `qwen3_coder` |
| Reasoning parser | `qwen3` |

模型默认开启 Thinking。客户端可针对单次请求关闭：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

工具调用已通过 `--enable-auto-tool-choice --tool-call-parser qwen3_coder` 启用，推理内容通过 `--reasoning-parser qwen3` 拆分为结构化字段。脚本不限制图片输入，保留模型的多模态能力。

## API 测试

脚本自带快速测试：

```bash
./scripts/manage_qwen3_6_27b.sh test
```

也可以直接调用 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## 环境变量覆盖

脚本内置的默认值可以直接使用，也可以在单次启动时覆盖。例如：

```bash
MAX_MODEL_LEN=131072 \
MAX_NUM_SEQS=4 \
MAX_NUM_BATCHED_TOKENS=8192 \
./scripts/manage_qwen3_6_27b.sh restart optimized
```

主要可覆盖变量：

```text
MODEL_PATH
SERVED_MODEL_NAME
HOST
PORT
TP_SIZE
DP_SIZE
DTYPE
MAX_MODEL_LEN
MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION
OFFLINE_MODE
START_TIMEOUT
STOP_TIMEOUT
LOG_RETENTION
NPU_DEVICE_IDS
RUNTIME_DIR
```

默认 `OFFLINE_MODE=1`，强制从本地模型目录加载。需要允许框架联网补齐文件时：

```bash
OFFLINE_MODE=0 ./scripts/manage_qwen3_6_27b.sh start
```

## 进程与日志

- `start` 会在后台创建独立进程组，并等待 `/v1/models` 真正就绪。
- 默认启动超时为 1800 秒；超时或等待期间按 `Ctrl+C` 会停止本次模型进程。
- `stop` 先发送 `SIGTERM`，默认等待 60 秒，超时后才发送 `SIGKILL`。
- 已有健康实例时再次执行 `start` 不会重复启动。
- 8000 端口被非本脚本进程占用时会安全失败，不会自动杀进程或切换端口。
- 运行状态、PID 和日志保存在 `runtime/qwen3.6-27b/`，该目录不会提交到 Git。
- 每次启动生成独立日志，只保留最近 10 份；`logs` 显示最后 100 行，`logs -f` 持续跟踪。

## 参考

- [vLLM-Ascend v0.23.0：Qwen3.5-27B / Qwen3.6-27B](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)
- [vLLM：Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [vLLM：Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
