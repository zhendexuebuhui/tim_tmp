# MiniMax-M2.7 AWQ on SGLang-Ascend

本页用于在 4 张 Ascend 910B2（设备 `2 3 5 7`，总 HBM 256 GiB）上验证 MiniMax-M2.7 社区 INT4 checkpoint。

当前目标不是直接追求生产性能，而是依次确认：

1. SGLang 原生 `MiniMaxM2ForCausalLM` 能识别模型；
2. QuantTrio 标准 AWQ 权重能够加载；
3. MiniMax MoE 专家计算能在 Ascend kernel 上执行；
4. 连续请求没有崩溃、空输出或明显数值异常；
5. 基础路径成立后再扩大上下文和并发。

## 模型优先级

| 优先级 | checkpoint | 实际格式 | 用途 |
|---|---|---|---|
| 1 | `QuantTrio/MiniMax-M2.7-AWQ` | 标准 AWQ，W4A16 | 默认验证对象 |
| 2 | `cyankiwi/MiniMax-M2.7-AWQ-4bit` | compressed-tensors INT4 | 兼容性对照 |

默认配置：

```text
MODEL_VARIANT=quanttrio
MODEL_PATH=/root/models/MiniMax-M2.7-AWQ
```

管理脚本默认不传 `--quantization`，由 checkpoint 的 `config.json` 自动识别。只有日志明确显示 loader 自动识别错误时，才设置 `QUANTIZATION_OVERRIDE`。

## 模型目录

宿主机：

```text
/data/models/
├── MiniMax-M2.7-AWQ
└── MiniMax-M2.7-AWQ-4bit
```

容器内对应：

```text
/root/models/
├── MiniMax-M2.7-AWQ
└── MiniMax-M2.7-AWQ-4bit
```

模型目录至少需要：

```text
config.json
tokenizer.json
tokenizer_config.json
*.safetensors
```

## 创建容器

默认镜像：

```text
quay.io/ascend/sglang:v0.5.13.post1-cann9.0.0-910b
```

首次创建：

```bash
cd /data/models/tim_tmp
bash ./scripts/start_sglang_ascend_container.sh
```

脚本使用 host 网络、64 GiB shared memory、4 张 NPU、Ascend driver 映射，并挂载：

```text
/data/models -> /root/models
/data/.cache -> /root/.cache
```

脚本不使用 `--rm`。容器已存在但停止时：

```bash
docker start sglang-ascend-env
```

进入容器：

```bash
docker exec -it sglang-ascend-env bash
cd /root/models/tim_tmp
```

仓库通过 contents API 创建的脚本可能没有 executable bit。以下两种方式均可：

```bash
bash ./scripts/manage_minimax_m2_7.sh check
```

或一次性设置：

```bash
chmod +x scripts/start_sglang_ascend_container.sh scripts/manage_minimax_m2_7.sh
```

## 预检

```bash
bash ./scripts/manage_minimax_m2_7.sh check
```

`check` 会验证：

- 模型、tokenizer 和权重文件存在；
- `architectures` 包含 `MiniMaxM2ForCausalLM`；
- QuantTrio 的 `quant_method` 为 `awq`；
- SGLang 包含原生 `MiniMaxM2ForCausalLM`；
- torch-npu 可用且逻辑 NPU 数量为 4；
- `/dev/davinci2/3/5/7` 在容器内可见。

预检失败时不要继续加载模型。

## 第一阶段：safe

`safe` 是默认 profile：

| 参数 | 默认值 |
|---|---:|
| TP | 4 |
| dtype | half/FP16 |
| context length | 4096 |
| max running requests | 1 |
| mem fraction static | 0.75 |
| chunked prefill | 4096 |
| Radix cache | 当前版本支持开关时关闭 |
| CUDA graph | 当前版本支持开关时关闭 |
| EP/MTP/推测解码 | 不启用 |

启动：

```bash
bash ./scripts/manage_minimax_m2_7.sh start safe
```

状态和日志：

```bash
bash ./scripts/manage_minimax_m2_7.sh status
bash ./scripts/manage_minimax_m2_7.sh logs -f
```

重点检查：

```bash
grep -Ei \
  'MiniMax|AWQ|quant|compressed|fallback|unsupported|not implemented|aclnn|CANN|OOM|NaN|error' \
  runtime/minimax-m2.7/serve-*.log
```

## 冒烟和稳定性验证

单次请求：

```bash
bash ./scripts/manage_minimax_m2_7.sh test
```

默认连续 20 次请求：

```bash
bash ./scripts/manage_minimax_m2_7.sh verify
```

调整轮数：

```bash
VERIFY_REQUESTS=50 bash ./scripts/manage_minimax_m2_7.sh verify
```

验证通过只说明基础生成链路稳定，不等于已经具备生产可用性。还要测试真实漏洞样本、长上下文和并发。

## 第二阶段：optimized

基础路径通过后：

```bash
bash ./scripts/manage_minimax_m2_7.sh restart optimized
```

`optimized` 仍不启用 EP、MTP 或推测解码，仅放宽：

| 参数 | 默认值 |
|---|---:|
| context length | 16384 |
| max running requests | 4 |
| mem fraction static | 0.82 |
| chunked prefill | 4096 |
| Radix cache | 使用 SGLang 默认行为 |

需要 32K benchmark 时，先扩大服务上下文：

```bash
MAX_MODEL_LEN=65536 \
  bash ./scripts/manage_minimax_m2_7.sh restart optimized
```

每次调整上下文、并发或静态显存比例后，都要重新检查四卡 HBM 峰值。

## Benchmark

默认轻量测试：

```bash
bash ./scripts/manage_minimax_m2_7.sh bench
```

默认负载：

```text
input=2048
output=256
num_prompts=8
request_rate=inf
max_concurrency=1
```

16K 输入、并发 2：

```bash
BENCH_INPUT_LEN=16384 \
BENCH_OUTPUT_LEN=512 \
BENCH_NUM_PROMPTS=8 \
BENCH_MAX_CONCURRENCY=2 \
  bash ./scripts/manage_minimax_m2_7.sh bench
```

矩阵：

```bash
bash ./scripts/manage_minimax_m2_7.sh bench-matrix
```

默认矩阵：

```text
input:       4096 16384 32768
concurrency: 1 2 4
```

超过当前服务上下文的组合会自动跳过。

产物：

```text
runtime/minimax-m2.7/serve-*.log
runtime/minimax-m2.7/verify-*.log
runtime/minimax-m2.7/bench-*.log
runtime/minimax-m2.7/bench-*.jsonl
runtime/minimax-m2.7/bench-*-npu.log
```

NPU 文件按时间保存原始 `npu-smi info`，用于复盘 AICore、功耗、温度和 HBM。

## cyankiwi fallback

```bash
bash ./scripts/manage_minimax_m2_7.sh stop

MODEL_VARIANT=cyankiwi \
MODEL_PATH=/root/models/MiniMax-M2.7-AWQ-4bit \
  bash ./scripts/manage_minimax_m2_7.sh check

MODEL_VARIANT=cyankiwi \
MODEL_PATH=/root/models/MiniMax-M2.7-AWQ-4bit \
  bash ./scripts/manage_minimax_m2_7.sh start safe
```

状态文件会保存实际模型路径和变体，因此后续 `test`、`verify`、`bench` 和无参数 `restart` 会继续使用当前 checkpoint。

只在自动识别明确失败时尝试：

```bash
QUANTIZATION_OVERRIDE=awq \
  bash ./scripts/manage_minimax_m2_7.sh start safe
```

或：

```bash
MODEL_VARIANT=cyankiwi \
MODEL_PATH=/root/models/MiniMax-M2.7-AWQ-4bit \
QUANTIZATION_OVERRIDE=compressed-tensors \
  bash ./scripts/manage_minimax_m2_7.sh start safe
```

## 错误解释

| 日志或现象 | 结论 |
|---|---|
| `Unsupported architecture MiniMaxM2ForCausalLM` | 模型类未注册或镜像不匹配 |
| 原生类预检通过，但加载时权重缺失 | checkpoint 命名与模型实现不匹配 |
| `Unsupported quantization method` | loader 不支持该量化格式 |
| `npu_fused_experts` / `not implemented` | 模型已识别，失败在 MoE/NPU kernel |
| 加载后 OOM | 容量、分片或临时工作区问题 |
| 能启动但输出为空、乱码、NaN 或持续重复 | 数值或 kernel 路径不可用于生产 |
| TP=4 分片或整除错误 | 当前四卡并行路径不兼容 |

## 推荐现场顺序

```text
创建容器
  -> check QuantTrio
  -> start safe
  -> test
  -> verify 20 次
  -> 真实漏洞样本小批量验证
  -> restart optimized
  -> 4K/16K/32K benchmark
  -> 与现有 Qwen3.6-27B 基线比较
```

MiniMax 路径完成上述验证前，保留现有 Qwen3.6 vLLM 服务作为回退基线。
