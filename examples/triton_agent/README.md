# Triton Agent: Claude Code Benchmark and RL Training

本目录用于基于 `verl + uni-agent + Claude Code` 进行 Triton/Ascend NPUKernelBench 轨迹生成、benchmark 和 RL 训练。

当前推荐的执行链路是：

```text
verl / vLLM model side
  -> Anthropic-to-OpenAI shim
  -> Claude Code in sandbox
  -> local CANNBot CLAUDE.md / .claude/skills / .claude/refs
  -> verifier / perf scripts on Ascend NPU
  -> metrics.json / reward
```

Claude Code 路径主要使用：

- `start_claude_code_vllm_server.sh`: 单独拉起 benchmark 用 vLLM server。
- `run_claude_code_vllm_benchmark.sh`: 跑纯推理 benchmark。
- `start_remote_claude_sandboxes.sh`: 在 agent/验证机器上预启动 Claude Code sandbox。
- `train_claude_code_megatron.sh`: 跑 verl + uni-agent + Claude Code 的 RL 训练。
- `config/triton_claude_code_blackbox.yaml`: verl/uni-agent 配置。
- `workspace/agent_workdir`: Claude Code 每条轨迹的基础工作区模板。

## 0. 镜像和环境要求

Claude Code sandbox 镜像需要提前包含：

- `claude` CLI 和 Node.js 运行环境。
- `python3 -m swerex.server` 所需 Python 包。
- CANN / driver runtime 可见的验证环境。
- `/opt/conda/envs/evaluator-py311/bin/python`，其中包含 `torch`、`torch_npu`、Triton Ascend 相关包。
- `sudo`。Claude Code 自身必须以非 root 用户运行，验证命令通过 `tools/run_npu_command.sh` 使用 root 权限执行。

默认镜像名：

```bash
export TRITON_CLAUDE_IMAGE=triton-claude-code-env:latest
```

如果镜像名不同，在 benchmark 和训练前都显式设置 `TRITON_CLAUDE_IMAGE`。

## 1. 部署形态

### 1.1 合并部署

训推和 agent/验证环境在同一台机器上。推荐仍然先用 `start_remote_claude_sandboxes.sh` 在本机预启动 sandbox，然后让 benchmark/RL 通过 `local_attach` 连接这些本机端口。这样 benchmark 和 RL 的 rollout 路径最接近。

```bash
export TRITON_REMOTE_SANDBOX_NUM=4
export TRITON_REMOTE_SANDBOX_BASE_PORT=18000
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export TRITON_EVAL_DEVICE_IDS=0,1,2,3
export TRITON_CLAUDE_IMAGE=triton-claude-code-env:latest

bash examples/triton_agent/start_remote_claude_sandboxes.sh
```

之后在同一台机器上使用：

```bash
export TRITON_REMOTE_SANDBOX_HOST=http://127.0.0.1
export TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
```

也可以设置 `TRITON_SANDBOX_DEPLOYMENT=local` 让 uni-agent 动态创建本机 Docker sandbox，但这条路径不如预启动 sandbox 方便排查 NPU/root/sudo 问题。

### 1.2 分离部署

A 节点放训练和模型推理，B 节点放 Claude Code 和算子验证 sandbox。B 节点不需要加入 Ray 集群，只需要 Docker、NPU driver/runtime 和 sandbox 镜像。

B 节点启动 sandbox：

```bash
cd /path/to/uni-agent-claudecode

export TRITON_CLAUDE_IMAGE=triton-claude-code-env:latest
export TRITON_REMOTE_SANDBOX_NUM=4
export TRITON_REMOTE_SANDBOX_BASE_PORT=18000
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export TRITON_EVAL_DEVICE_IDS=0,1,2,3

bash examples/triton_agent/start_remote_claude_sandboxes.sh
```

A 节点连接 B 节点：

```bash
export TRITON_REMOTE_SANDBOX_HOST=http://<B_NODE_IP>
export TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
```

同时需要保证 B 节点容器能访问 A 节点上的 shim 地址。A 节点通常设置：

```bash
export TRITON_SHIM_BIND_HOST=0.0.0.0
export TRITON_SHIM_PUBLIC_HOST=<A_NODE_IP_REACHABLE_FROM_B>
```

如果网络有防火墙，需要放通 A 节点上的临时 shim 端口，以及 A/B 间的 sandbox 端口。

## 2. Benchmark

Benchmark 用来测试“固定模型 + Claude Code + sandbox 验证”的纯推理效果，不启动 RL 训练。

### 2.1 A 节点启动 vLLM server

```bash
cd /path/to/uni-agent-claudecode

export MODEL_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct
export SERVED_MODEL_NAME=triton-claude-code-model
export PROXY_PORT=5000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_TP_SIZE=8

bash examples/triton_agent/start_claude_code_vllm_server.sh
```

检查：

```bash
curl http://127.0.0.1:5000/v1/models
```

### 2.2 启动 sandbox

合并部署时，在同一台机器执行第 1.1 节的 sandbox 启动命令。

分离部署时，在 B 节点执行第 1.2 节的 sandbox 启动命令。

### 2.3 A 节点运行 benchmark

合并部署示例：

```bash
cd /path/to/uni-agent-claudecode

export OPENAI_BASE_URL=http://127.0.0.1:5000/v1
export OPENAI_API_KEY=EMPTY
export SERVED_MODEL_NAME=triton-claude-code-model

export BENCHMARK_SANDBOX_MODE=remote
export TRITON_REMOTE_SANDBOX_HOST=http://127.0.0.1
export TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export SHIM_BIND_HOST=0.0.0.0
export SHIM_PUBLIC_HOST=127.0.0.1

export BENCHMARK_LEVEL=level1
export BENCHMARK_START=0
export BENCHMARK_NUM=10

bash examples/triton_agent/run_claude_code_vllm_benchmark.sh
```

分离部署示例：

```bash
cd /path/to/uni-agent-claudecode

export OPENAI_BASE_URL=http://127.0.0.1:5000/v1
export OPENAI_API_KEY=EMPTY
export SERVED_MODEL_NAME=triton-claude-code-model

export BENCHMARK_SANDBOX_MODE=remote
export TRITON_REMOTE_SANDBOX_HOST=http://<B_NODE_IP>
export TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export SHIM_BIND_HOST=0.0.0.0
export SHIM_PUBLIC_HOST=<A_NODE_IP_REACHABLE_FROM_B>

export BENCHMARK_LEVEL=level1
export BENCHMARK_START=0
export BENCHMARK_NUM=10

bash examples/triton_agent/run_claude_code_vllm_benchmark.sh
```

输出位置：

- `examples/triton_agent/vllm_benchmark_runs/<timestamp>/summary.jsonl`
- `examples/triton_agent/vllm_benchmark_runs/<timestamp>/artifacts/<op>/conversation.log`
- `examples/triton_agent/vllm_benchmark_runs/<timestamp>/artifacts/<op>/claude_code_stdout.log`
- `examples/triton_agent/vllm_benchmark_runs/<timestamp>/artifacts/<op>/metrics.json`

常用参数：

```bash
export BENCHMARK_ROOT=examples/triton_agent/benchmarks/NPUKernelBench
export BENCHMARK_LEVEL=level1
export BENCHMARK_START=0
export BENCHMARK_NUM=50
export TRITON_CLAUDE_MAX_TURNS=35
export BENCHMARK_CLAUDE_EXTRA_ARGS="--max-turns ${TRITON_CLAUDE_MAX_TURNS}"
export BENCHMARK_RUN_ROOT=/path/to/output_dir
```

`BENCHMARK_SANDBOX_MODE=local` 会直接在当前机器运行 `claude` CLI，不经过远端 swerex sandbox。该模式适合快速调试 CLI 和 prompt，但不建议作为 RL rollout 的对齐 benchmark。

## 3. RL 训练

RL 训练不需要提前启动 `start_claude_code_vllm_server.sh`。训练脚本会通过 verl 创建训练/推理所需的 vLLM rollout engine，并通过 uni-agent blackbox gateway 调用 Claude Code runner。

### 3.1 合并部署训练

推荐先在同一台机器预启动 sandbox：

```bash
export TRITON_REMOTE_SANDBOX_NUM=4
export TRITON_REMOTE_SANDBOX_BASE_PORT=18000
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export TRITON_EVAL_DEVICE_IDS=0,1,2,3
export TRITON_CLAUDE_IMAGE=triton-claude-code-env:latest

bash examples/triton_agent/start_remote_claude_sandboxes.sh
```

然后启动训练：

```bash
cd /path/to/uni-agent-claudecode

export MODEL_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export N_GPUS=16

export TRITON_SANDBOX_DEPLOYMENT=local_attach
export TRITON_REMOTE_SANDBOX_HOST=http://127.0.0.1
export TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export TRITON_SHIM_BIND_HOST=0.0.0.0
export TRITON_SHIM_PUBLIC_HOST=127.0.0.1

bash examples/triton_agent/train_claude_code_megatron.sh
```

如果希望 uni-agent 动态创建本机 sandbox，可以改成：

```bash
export TRITON_SANDBOX_DEPLOYMENT=local
export TRITON_EVAL_DEVICE_IDS=0,1,2,3
export TRITON_EVAL_DEVICE_COUNT=1
```

这种方式不需要先运行 `start_remote_claude_sandboxes.sh`。

### 3.2 分离部署训练

B 节点先启动 sandbox：

```bash
cd /path/to/uni-agent-claudecode

export TRITON_CLAUDE_IMAGE=triton-claude-code-env:latest
export TRITON_REMOTE_SANDBOX_NUM=4
export TRITON_REMOTE_SANDBOX_BASE_PORT=18000
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export TRITON_EVAL_DEVICE_IDS=0,1,2,3

bash examples/triton_agent/start_remote_claude_sandboxes.sh
```

A 节点启动训练：

```bash
cd /path/to/uni-agent-claudecode

export MODEL_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export N_GPUS=16
export MASTER_ADDR=<A_NODE_IP>

export TRITON_SANDBOX_DEPLOYMENT=local_attach
export TRITON_REMOTE_SANDBOX_HOST=http://<B_NODE_IP>
export TRITON_REMOTE_SANDBOX_PORTS=18000,18001,18002,18003
export TRITON_REMOTE_SANDBOX_AUTH_TOKEN=tok
export TRITON_SHIM_BIND_HOST=0.0.0.0
export TRITON_SHIM_PUBLIC_HOST=<A_NODE_IP_REACHABLE_FROM_B>

export TRITON_CLAUDE_ARTIFACT_DIR=/path/to/claude_code_results
export PROJECT_NAME=triton-claude-code
export EXPERIMENT_NAME=debug_run

bash examples/triton_agent/train_claude_code_megatron.sh
```

B 节点不需要加入 Ray 集群。只有 A 节点上的训练/推理进程需要 Ray；如果需要扩展成多训练节点，则这些训练节点按 verl/Ray 的方式加入集群，B 节点仍然只是 sandbox 服务。

## 4. 数据集

训练脚本会自动调用：

```bash
python3 examples/triton_agent/prepare_kernelbench_claude_code_data.py
```

常用变量：

```bash
export TRITON_KERNELBENCH_DATASET=examples/triton_agent/benchmarks/NPUKernelBench
export TRITON_KERNELBENCH_LEVELS=level_1
export TRITON_KERNELBENCH_PARQUET=examples/triton_agent/kernelbench_claude_code.parquet
export TRITON_KERNELBENCH_MAX_ROWS=128
export TRITON_KERNELBENCH_VAL_MAX_ROWS=16
export TRITON_KERNELBENCH_VAL_START=0
export TRITON_KERNELBENCH_ARCH=ascend910b1
```

`TRITON_KERNELBENCH_DATASET` 可以是本地目录、本地 parquet 文件所在目录，也可以是 HuggingFace dataset 名称。当前仓内 benchmark 默认使用本地 `examples/triton_agent/benchmarks/NPUKernelBench`。

## 5. 训练常用参数

```bash
export BATCH_SIZE=8
export ROLLOUT_N=8
export VAL_BATCH_SIZE=16
export VAL_ROLLOUT_N=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=10
export SAVE_FREQ=10
export ACTOR_LR=2e-6

export MAX_PROMPT_LENGTH=36864
export MAX_RESPONSE_LENGTH=4096
export MAX_MODEL_LEN=40960
export MAX_NUM_BATCHED_TOKENS=16384

export AGENT_NUM_WORKERS=8
export MAX_CONCURRENT_SESSIONS=4
export TRITON_CLAUDE_MAX_TURNS=50
export TRITON_CLAUDE_TIME_BUDGET_SEC=1800
```

`MAX_CONCURRENT_SESSIONS` 建议不超过可用 sandbox endpoint 数量。默认会根据 `TRITON_REMOTE_SANDBOX_PORTS` 自动推断。

Reward 相关：

```bash
export TRITON_REWARD_AST_OK=0.10
export TRITON_REWARD_COMPILE_OK=0.10
export TRITON_REWARD_CORRECTNESS_OK=0.55
export TRITON_REWARD_ALL_CORRECT_BONUS=0.10
export TRITON_REWARD_SPEEDUP_MAX=0.40
export TRITON_REWARD_TARGET_SPEEDUP=2.0
```

## 6. 日志和产物

Benchmark：

- `vllm_benchmark_runs/<timestamp>/summary.jsonl`
- `vllm_benchmark_runs/<timestamp>/remote_endpoints.tsv`
- `vllm_benchmark_runs/<timestamp>/manifest.tsv`
- `vllm_benchmark_runs/<timestamp>/artifacts/<op>/conversation.log`
- `vllm_benchmark_runs/<timestamp>/artifacts/<op>/claude_code_stdout.log`
- `vllm_benchmark_runs/<timestamp>/artifacts/<op>/metrics.json`

RL 训练：

- 训练主日志：`logs` 环境变量，默认 `/home/p00938733/verl-claude-code.log`
- rollout 进度：`TRITON_PROGRESS_COUNTER_FILE`
- Claude Code 轨迹归档：`TRITON_CLAUDE_ARTIFACT_DIR`
- checkpoint：`CKPTS_DIR`，默认 `./ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}`

`metrics.json` 是 reward 的主要输入。成功标准是 verifier 产物显示 `passed_cases == total_cases`，AST check 只作为预检查，不代表正确性通过。

## 7. 常见检查

检查 vLLM：

```bash
curl http://127.0.0.1:5000/v1/models
```

检查 B 节点 sandbox 容器：

```bash
docker ps | grep triton-cc-sandbox
docker logs --tail 100 triton-cc-sandbox-0
```

检查容器内 NPU 验证 Python：

```bash
docker exec triton-cc-sandbox-0 bash -lc '
  echo OPERATOR_PYTHON=${OPERATOR_PYTHON}
  echo ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}
  ${OPERATOR_PYTHON:-/opt/conda/envs/evaluator-py311/bin/python} - <<PY
import torch, torch_npu
x = torch.ones(1).npu()
print(torch.npu.device_count(), torch.npu.is_available(), x)
PY
'
```

如果 B 节点能启动 sandbox，但 Claude Code 报模型 API 连接失败，优先检查：

- `TRITON_SHIM_PUBLIC_HOST` 或 `SHIM_PUBLIC_HOST` 是否是 B 节点可访问的 A 节点 IP。
- A 节点是否监听在 `0.0.0.0`。
- A/B 防火墙是否放通 shim 临时端口。
- B 容器内是否能 `curl http://<A_NODE_IP>:<shim_port>/healthz`。

