# Qwen3.6-27B on vLLM-Ascend

本仓库用于在 4 张 Ascend 910B2 上管理 Qwen3.6-27B 服务，并保存可复盘的在线 Benchmark、服务日志和 NPU 监控数据。

当前推荐直接使用默认 `optimized` 配置：

```bash
./scripts/manage_qwen3_6_27b.sh check
./scripts/manage_qwen3_6_27b.sh start optimized
```

默认配置已经固化为：`TP=2`、`DP=2`、`API_SERVER_COUNT=2`、`MAX_NUM_SEQS=8/每个 DP 副本`。实际业务并发建议由客户端、网关或任务队列控制：交互式 Agent 优先控制在 4，吞吐优先可使用 8。

> `MAX_NUM_SEQS=8` 是每个 DP 副本的调度容量，不是整个服务的全局并发限制。两个 DP 副本理论上各可运行 8 条序列；超出的请求由服务排队，而不是立即拒绝。

## 现场环境

| 项目 | 当前值 |
|---|---|
| NPU | Ascend 910B2 × 4，设备 `2 3 5 7` |
| 单卡 / 总 HBM | 64 GiB / 256 GiB |
| 容器镜像 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-openeuler` |
| 容器名 | `vllm-ascend-env` |
| 模型 | Qwen3.6-27B，`bfloat16` |
| 宿主机模型目录 | `/data/models` |
| 容器内模型目录 | `/root/models` |
| 宿主机仓库 | `/data/models/tim_tmp` |
| 容器内仓库 | `/root/models/tim_tmp` |
| API | `127.0.0.1:8000`，未配置 API Key |

容器由 [`scripts/start_vllm_ascend_container.sh`](scripts/start_vllm_ascend_container.sh) 创建，使用 host 网络、32 GiB shared memory，并挂载 4 张 NPU、Ascend 驱动、`npu-smi`、模型和缓存目录。

该脚本没有使用 `--rm`，因此 `docker stop` 不会删除容器；也没有配置自动重启策略。首次创建和后续恢复分别使用：

```bash
# 首次创建
bash ./scripts/start_vllm_ascend_container.sh

# 已存在但已停止
docker start vllm-ascend-env
```

容器重启后，模型服务不会自动启动，需要再次运行管理脚本。

## 快速开始

进入容器并启动服务：

```bash
docker exec -it vllm-ascend-env bash
cd /root/models/tim_tmp

./scripts/manage_qwen3_6_27b.sh check
./scripts/manage_qwen3_6_27b.sh start optimized
./scripts/manage_qwen3_6_27b.sh test
```

查看状态和日志：

```bash
./scripts/manage_qwen3_6_27b.sh status
./scripts/manage_qwen3_6_27b.sh logs -f
```

停止服务：

```bash
./scripts/manage_qwen3_6_27b.sh stop
```

## 当前默认配置

以下值来自当前 [`scripts/manage_qwen3_6_27b.sh`](scripts/manage_qwen3_6_27b.sh)：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `TP_SIZE` | `2` | 每个模型副本使用 2 张 NPU |
| `DP_SIZE` | `2` | 两个独立 DP 副本，共使用 4 张 NPU |
| `API_SERVER_COUNT` | `2` | 前端 API Server 进程数 |
| `MAX_NUM_SEQS` | `8` | 每个 DP 副本的序列上限，不是全局限流 |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | 32K Prompt 至少分成两个 Prefill Chunk |
| `MAX_MODEL_LEN` | `262144` | 最大模型上下文 |
| `GPU_MEMORY_UTILIZATION` | `0.90` | NPU 显存利用率目标 |
| `DTYPE` | `bfloat16` | 模型计算精度 |
| `ENABLE_REDUCE_SAMPLE` | `1` | optimized 中启用 reduce-sample |
| Served model name | `qwen3.6-27b` | API 请求中的模型名 |
| Model path | `/root/models/Qwen3.6-27B` | 容器内模型路径 |

### 运行模式

| 功能 | optimized（默认） | safe |
|---|---:|---:|
| MTP：`qwen3_5_mtp`，draft tokens = 3 | 开启 | 关闭 |
| `FULL_DECODE_ONLY` 图模式 | 开启 | 关闭 |
| CPU binding | 开启 | 关闭 |
| `enable_reduce_sample` | 默认开启，可用环境变量关闭 | 关闭 |
| Async scheduling | 开启 | 关闭 |
| Prefix caching | 开启 | 明确关闭 |

`optimized` 启动失败时不会自动切换到 `safe`。先查看日志，再显式使用诊断模式：

```bash
./scripts/manage_qwen3_6_27b.sh logs
./scripts/manage_qwen3_6_27b.sh start safe
```

`safe` 与 `optimized` 使用相同的 TP/DP、API Server、上下文和序列容量，只关闭可选性能功能，便于定位兼容性或启动问题。

Qwen3.6-27B 使用混合 KV Cache。Prefix Cache 更适合存在较长公共前缀的 Agent 请求；短重复前缀不一定命中。现场 Benchmark 中每卡 HBM 峰值约为 60.5～61.1 GiB，即约 92.4%～93.2%，余量较小，修改上下文、batch 或缓存参数后应重新关注 HBM 峰值。

## 管理命令

```bash
# 环境、模型文件、NPU 数量和端口所有权预检
./scripts/manage_qwen3_6_27b.sh check

# 启动；省略模式时默认 optimized
./scripts/manage_qwen3_6_27b.sh start
./scripts/manage_qwen3_6_27b.sh start optimized
./scripts/manage_qwen3_6_27b.sh start safe

# 重启；省略模式时沿用上一次模式
./scripts/manage_qwen3_6_27b.sh restart
./scripts/manage_qwen3_6_27b.sh restart safe

# 停止、状态、日志、冒烟测试和 Benchmark
./scripts/manage_qwen3_6_27b.sh stop
./scripts/manage_qwen3_6_27b.sh status
./scripts/manage_qwen3_6_27b.sh logs
./scripts/manage_qwen3_6_27b.sh logs -f
./scripts/manage_qwen3_6_27b.sh test
./scripts/manage_qwen3_6_27b.sh bench

# 帮助
./scripts/manage_qwen3_6_27b.sh --help
```

## API 使用

服务默认只监听本机回环地址。宿主机和使用 host 网络的当前容器可以访问，其他服务器不能直接访问。

主要兼容端点：

| 协议 | URL |
|---|---|
| OpenAI Chat Completions | `http://127.0.0.1:8000/v1/chat/completions` |
| OpenAI Models | `http://127.0.0.1:8000/v1/models` |
| Anthropic Messages | `http://127.0.0.1:8000/v1/messages` |

服务端没有配置 API Key；某些客户端若强制要求该字段，可填任意非空占位值。

### OpenAI 兼容调用

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

模型默认开启 Thinking。单次请求可通过 `chat_template_kwargs.enable_thinking=false` 关闭。

工具调用已通过 `--enable-auto-tool-choice --tool-call-parser qwen3_coder` 启用；推理内容通过 `--reasoning-parser qwen3` 拆分为结构化字段。脚本没有禁用图片输入，保留模型的多模态能力。

## Benchmark

服务就绪后执行：

```bash
./scripts/manage_qwen3_6_27b.sh bench
```

默认 Benchmark 负载：

| 参数 | 默认值 |
|---|---:|
| 输入 Token | `32768` |
| 输出 Token | `1024` |
| 请求数 | `100` |
| 请求速率 | `1 req/s` |
| 最大并发 | `8` |
| Thinking | 关闭 |

可通过环境变量临时覆盖。例如：

```bash
BENCH_INPUT_LEN=16384 \
BENCH_OUTPUT_LEN=1024 \
BENCH_NUM_PROMPTS=5 \
BENCH_REQUEST_RATE=1 \
BENCH_MAX_CONCURRENCY=2 \
./scripts/manage_qwen3_6_27b.sh bench
```

如需测试 Thinking，设置 `BENCH_THINKING=1`。`BENCH_REQUEST_RATE=inf` 会立即提交请求，再由 `BENCH_MAX_CONCURRENCY` 控制客户端并发。

### 输出产物

Benchmark 期间默认每秒调用一次 `npu-smi info`，记录四张卡的 AICore、功率、温度和 HBM。产物保存在：

```text
runtime/qwen3.6-27b/bench-时间戳.log
runtime/qwen3.6-27b/bench-时间戳.json
runtime/qwen3.6-27b/bench-时间戳-npu.csv
runtime/qwen3.6-27b/bench-时间戳-summary.txt
```

其中 CSV 是逐秒原始数据，`summary.txt` 合并服务指标与四卡指标。单次 `npu-smi` 调用失败只记录到 `bench-时间戳-npu-monitor.err`，不会中止 Benchmark。

```bash
# 调整采样周期
BENCH_NPU_SAMPLE_INTERVAL=2 ./scripts/manage_qwen3_6_27b.sh bench

# 关闭 NPU 监控
BENCH_NPU_MONITOR=0 ./scripts/manage_qwen3_6_27b.sh bench
```

## A–J 现场结果

以下结果均来自 2026-07-22 的同一台 4×910B2 服务器。固定负载为 32768 输入 Token、1024 输出 Token、64 请求、`request-rate=inf`、Thinking 关闭、`temperature=0`。A–J 共 640/640 请求成功。

这些数字只用于当前软件、模型、硬件和负载下的配置比较，不代表通用性能承诺。

| 轮次 | 核心差异 | Bench 并发 | 输出吞吐 tok/s | 平均 TTFT | 平均 TPOT | ITL P99 | AICore 均值 | 四卡均值极差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | TP4/DP1，max-seqs=8，reduce off | 8 | 103.99 | 13.29s | 63.42ms | 3.457s | 39.6% | 0.7pp |
| B | TP4/DP1，max-seqs=16，reduce off | 16 | 114.27 | 28.15s | 110.97ms | 3.590s | 40.3% | 1.7pp |
| C | TP4/DP1，max-seqs=16，reduce on | 16 | 118.36 | 30.58s | 104.26ms | 3.580s | 41.4% | 1.1pp |
| D | TP2/DP2，max-seqs=8/副本，API=2，reduce off | 8 | **147.13** | 12.52s | 40.93ms | 3.713s | **59.4%** | **2.1pp** |
| E | TP2/DP2，max-seqs=8/副本，API=2，reduce off | 16 | 154.87 | 36.07s | 60.33ms | 3.785s | 54.4% | 17.4pp |
| F | TP2/DP2，max-seqs=4/副本，API=2，reduce on | 8 | 117.25 | 30.90s | 34.70ms | 3.663s | 53.9% | 16.1pp |
| G | 同 F | 4 | 109.24 | **8.78s** | **27.76ms** | **0.731s** | 55.8% | 9.9pp |
| H | 同 F，API=1 | 8 | 90.11 | 47.91s | 38.64ms | 3.666s | 37.8% | 51.1pp |
| I | 同 F，API=4 | 8 | 117.96 | 30.21s | 35.29ms | 3.660s | 52.8% | 10.7pp |
| J | 同 F，API=8 | 8 | 116.46 | 32.93s | 34.09ms | 3.545s | 54.6% | 16.5pp |

### 已确认的结论

- `TP=2/DP=2` 明显优于 `TP=4/DP=1`，因此已经成为默认拓扑。
- D 是最接近当前默认服务的吞吐基线。与 A 相比，D 的输出吞吐提高约 41.5%，平均 TTFT 略低，AICore 均值更高。
- E 将客户端并发从 8 提高到 16，只比 D 增加约 5.3% 输出吞吐，但平均 TTFT 增加约 188%，不适合作为交互式默认值。
- G 表明客户端并发 4 是更好的交互式档位：与 F 相比只损失约 6.8% 输出吞吐，但平均 TTFT 降低约 71.6%，ITL P99 降低约 80%。
- H 的单 API Server 出现严重 DP 失衡，一组 TP2 的 AICore 约 63%，另一组约 12.5%；不应将 `API_SERVER_COUNT=1` 用于生产。
- F/I/J 的输出吞吐只在 116.46～117.96 tok/s 之间，API Server 从 2 增加到 4 或 8 没有实质收益，默认保持 2。
- F–J 的 `max-seqs=4/副本` 在 DP 调度不均时过早形成排队，因此已回退为当前默认的 `max-seqs=8/副本`。
- C 相比 B 的输出吞吐提高约 3.6%，但这是 TP4/DP1 下的单次结果；当前 TP2/DP2、max-seqs=8、reduce-sample 开启的完整默认组合没有做严格重复对照，因此不能直接承诺其吞吐等于或高于 D。

### 当前生产取舍

- 服务端：使用默认 `optimized`，即 TP2/DP2、API Server 2、max-seqs 8/副本、reduce-sample 开启。
- 交互式代码审计、Claude Code/OpenCode、Agent：客户端并发优先设为 4。
- 批量吞吐任务：客户端并发可设为 8。
- 不建议使用 API Server 1、4 或 8；也不建议用 `MAX_NUM_SEQS=4` 模拟全局并发限制。

## 历史实验矩阵

[`scripts/run_qwen3_6_bench_matrix.sh`](scripts/run_qwen3_6_bench_matrix.sh) 保留 A–J 单轮入口：

```bash
./scripts/run_qwen3_6_bench_matrix.sh A
./scripts/run_qwen3_6_bench_matrix.sh D
./scripts/run_qwen3_6_bench_matrix.sh F
```

`all` 当前只按 F → G → H → I → J 执行；A–E 只能单独指定。每轮都会执行 `check → stop → check → start optimized → test → bench → stop`，失败时停止后续轮次。

```bash
./scripts/run_qwen3_6_bench_matrix.sh all
```

> 注意：该矩阵脚本保留了 F–J 当时的旧 `SERVICE_MAX_CONCURRENCY=8` 推导与 manifest 展示逻辑，而当前管理脚本已经删除该变量并将 `MAX_NUM_SEQS` 默认改为 8。使用当前 `master` 重新执行 F–J 时，实际服务会采用管理脚本默认的 max-seqs=8，但矩阵 manifest 仍可能显示旧的派生值 4。因此，仓库内已提交的 A–J 产物是有效历史现场；当前矩阵脚本不应被视为 F–J 旧配置的严格复现工具。

矩阵产物使用同一个运行 ID 分组：

```text
runtime/qwen3.6-27b/matrix-运行ID-轮次.log
runtime/qwen3.6-27b/matrix-运行ID-轮次-manifest.txt
runtime/qwen3.6-27b/serve-时间戳-matrix-运行ID-轮次-optimized.log
runtime/qwen3.6-27b/bench-时间戳-matrix-运行ID-轮次-*.{log,json,csv,txt}
```

## 环境变量覆盖

脚本默认值可以在单次命令中覆盖：

```bash
MAX_MODEL_LEN=131072 \
MAX_NUM_SEQS=8 \
MAX_NUM_BATCHED_TOKENS=8192 \
./scripts/manage_qwen3_6_27b.sh restart optimized
```

主要服务变量：

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
API_SERVER_COUNT
GPU_MEMORY_UTILIZATION
OFFLINE_MODE
START_TIMEOUT
STOP_TIMEOUT
LOG_RETENTION
NPU_DEVICE_IDS
ENABLE_REDUCE_SAMPLE
RUN_LABEL
RUNTIME_DIR
```

主要 Benchmark 变量：

```text
BENCH_INPUT_LEN
BENCH_OUTPUT_LEN
BENCH_NUM_PROMPTS
BENCH_REQUEST_RATE
BENCH_MAX_CONCURRENCY
BENCH_THINKING
BENCH_NPU_MONITOR
BENCH_NPU_SAMPLE_INTERVAL
BENCH_TEMPERATURE
```

默认 `OFFLINE_MODE=1`，强制从本地模型目录加载。需要允许框架联网补齐文件时：

```bash
OFFLINE_MODE=0 ./scripts/manage_qwen3_6_27b.sh start optimized
```

## 进程与日志

- `start` 在后台创建独立进程组，并等待 `/v1/models` 就绪。
- 默认启动超时为 1800 秒；等待期间按 `Ctrl+C` 会停止本次模型进程。
- `stop` 先发送 `SIGTERM`，默认等待 60 秒，超时后再发送 `SIGKILL`。
- 已有健康实例时再次执行 `start` 不会重复启动。
- 8000 端口被未知进程占用时会安全失败，不会自动杀进程或切换端口。
- 运行状态、PID 和日志保存在 `runtime/qwen3.6-27b/`。
- 每次启动和 Benchmark 生成独立文件，默认各保留最近 10 份。
- Git 跟踪 `serve-*.log` 及 Benchmark 的 log、JSON、NPU CSV、summary 和矩阵 manifest；PID、profile、状态文件和监控错误等临时文件仍被忽略。

I/J 的历史日志中，`EngineDeadError`、强制清理进程以及 shared-memory/semaphore 泄漏警告都发生在 Benchmark 已完成后的关闭阶段；对应轮次 64/64 请求成功。这些警告不代表推理失败，但说明当前 vLLM-Ascend 多进程优雅停机仍不完善。

提交现场产物后，远端才能用于后续分析：

```bash
git add runtime/qwen3.6-27b/
git commit -m "Add runtime benchmark logs"
git push
```

## 参考

- [vLLM-Ascend v0.23.0：Qwen3.5-27B / Qwen3.6-27B](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)
- [vLLM：Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- [vLLM：Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [vLLM：Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
