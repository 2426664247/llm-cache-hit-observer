# Cache Hit Proxy

<p align="center">
  <strong>OpenAI-compatible LLM Prompt Cache 观测与估算代理</strong>
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-0.2.0-2f6feb?style=flat-square">
  <img alt="python" src="https://img.shields.io/badge/python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-proxy-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="status" src="https://img.shields.io/badge/status-research%20preview-555?style=flat-square">
</p>

`cache_hit_proxy` 是一个本地 HTTP 代理，用于观测 `POST /v1/chat/completions` 请求的 prompt token、Prompt Cache 命中估计，以及上游模型服务返回的真实 `usage`。它会尽量保持 OpenAI-compatible API 语义不变，同时把每次请求写入结构化 JSONL trace，方便复盘、对照和调试。

> 这个仓库只包含代理工具本体，不包含个人实验结果、本地服务信息、API key、raw response 或私有运行日志。

## 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [运行模式](#运行模式)
- [Tokenizer 预设](#tokenizer-预设)
- [Trace 输出](#trace-输出)
- [Agent 使用指南](#agent-使用指南)
- [测试](#测试)
- [安全与提交边界](#安全与提交边界)
- [English Summary](#english-summary)

## 功能特性

| 能力 | 说明 |
| --- | --- |
| OpenAI-compatible 代理 | 接收本地 `/v1/chat/completions` 请求，并转发到上游模型服务。 |
| 本地 token 估算 | 根据 tokenizer preset 或 vLLM `/tokenize` 估算 prompt token。 |
| Prompt Cache 命中估算 | 基于请求历史、block size 和前缀匹配估算可复用缓存 token。 |
| 真实 usage 读取 | 支持从 JSON 响应、SSE stream event 或 vLLM Prometheus metrics 读取真实 usage。 |
| 多运行模式 | 覆盖普通 chat、OpenClaw agent、pi-ai payload 和 vLLM probe 场景。 |
| JSONL trace | 每次请求落盘为结构化记录，便于离线分析和准确性对照。 |

## 安装

建议使用 Python 3.12 或更新版本。

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY>/cache_hit_proxy
python -m pip install -r requirements.txt
```

检查工具版本：

```bash
python main.py --version
```

下载或检查 tokenizer-only 资源：

```bash
python download_tokenizers.py
python download_tokenizers.py --check-only
```

下载脚本只拉取 tokenizer、chat template 等轻量资源，不下载模型权重。

## 快速开始

启动一个通用 OpenAI-compatible 代理：

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

然后把你的客户端 `base_url` 指向本地代理：

```text
http://127.0.0.1:8787/v1
```

API key 或 `Authorization` header 仍然按上游供应商的要求传入客户端。代理会转发常规请求头，但不会要求你把密钥写进仓库文件。

如果上游不是标准的 `{base}/v1/chat/completions` 路径，可以显式传完整地址：

```bash
python main.py \
  --port 8787 \
  --target-chat-url https://YOUR_PROVIDER_FULL_CHAT_COMPLETIONS_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

## 运行模式

通过 `--conversation-mode` 选择请求处理方式：

| 模式 | 适用场景 | 说明 |
| --- | --- | --- |
| `simple_streaming` | 普通 OpenAI-compatible chat/completions 请求 | 默认通用模式，适合直接观测上游 usage。 |
| `openclaw_agent` | OpenClaw agent 请求 | 使用 raw HTTP body 做输入 token 来源，并加入多轮缓存估计修正。 |
| `piai_probe` | pi-ai 最终 provider payload 分析 | 读取顶层 `tools`、`thinking`、`reasoning_effort`，并从 content block 提取文本。 |
| `vllm_probe` | vLLM Prefix Cache 观测 | 通过 `/tokenize` 估算 token，并从 `/metrics` 读取真实 cache hit delta。 |

### OpenClaw Agent 模式

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --conversation-mode openclaw_agent \
  --raw-request-capture none
```

默认情况下，内存中的历史请求只按 idle TTL 清理，不限制请求数量。只有在需要手动限制内存时，才设置 `--max-history-requests`。

### vLLM Probe 模式

如果 vLLM 使用标准 OpenAI-compatible 地址：

```bash
python main.py \
  --port 8787 \
  --target-base-url http://YOUR_VLLM_HOST:8000/v1 \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe \
  --session-id vllm_probe
```

如果 chat、tokenize、metrics 地址需要分别指定：

```bash
python main.py \
  --port 8787 \
  --target-chat-url http://YOUR_VLLM_HOST:8000/v1/chat/completions \
  --vllm-tokenize-url http://YOUR_VLLM_HOST:8000/tokenize \
  --vllm-metrics-url http://YOUR_VLLM_HOST:8000/metrics \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe
```

使用 `vllm_probe` 时，尽量避免同一个 vLLM 进程上混入无关请求，否则 metrics delta 可能包含其他流量。

## Tokenizer 预设

prompt token 估算通过 `--tokenizer-preset` 选择 tokenizer：

| Preset | 行为 |
| --- | --- |
| `deepseek-v4-pro` | 使用内置 DeepSeek tokenizer 和 prompt renderer。 |
| `glm-5.1` | 使用本地 GLM tokenizer 和官方 chat template。 |
| `qwen3-coder-plus` | 使用本地 Qwen3-Coder tokenizer 和官方 chat template。 |
| `kimi-k2.6` | 使用本地 Kimi tokenizer 和官方 chat template。 |
| `doubao-seed-2-0-code-preview-260215` / `volcanoengine` | fallback 到 DeepSeek tokenizer，可本地估算，但预期会有误差。 |
| `vllm` | 调用正在运行的 vLLM server `/tokenize`，适合泛用 vLLM OpenAI-compatible 服务。 |

默认 tokenizer 目录规则：

- DeepSeek 与 Doubao fallback 使用 `./deepseek_tokenizer`。
- GLM、Qwen、Kimi 使用 `./tokenizers/<preset>`。
- `vllm` 默认由 `--target-base-url` 推导 `/tokenize`，也可以通过 `--vllm-tokenize-url` 显式指定。

如需指定其他 tokenizer 目录，使用：

```bash
--tokenizer-dir /path/to/tokenizer
```

## Trace 输出

每次代理请求会写入：

```text
traces/{session_id}.jsonl
```

常用字段：

| 字段 | 含义 |
| --- | --- |
| `conversation_mode` | 当前请求使用的代理模式。 |
| `tokenizer_preset` | 当前请求使用的 tokenizer preset。 |
| `predicted_input_tokens` | 本地估算输入 token。 |
| `actual_input_tokens` | 上游或 vLLM 观测到的真实输入 token。 |
| `estimated_cached_tokens` | 本地估算缓存命中 token。 |
| `actual_cached_tokens` | 上游 usage 或 vLLM metrics 中的真实缓存命中 token。 |
| `difference_tokens` | `actual_cached_tokens - estimated_cached_tokens`。 |
| `status` | 本次估算状态，例如正常、未知、上游错误或明显高估。 |

如果需要字节级回放请求，可以使用 `--raw-request-capture base64`；默认建议保持 `--raw-request-capture none`，避免 trace 中包含敏感请求内容。

## Agent 使用指南

如果后续有 Agent 读取这个仓库并需要根据自己的环境调用工具，请让它先读：

```text
AGENTS.md
```

这份文档说明了如何根据上游 provider、vLLM 地址、tokenizer 资源和安全边界选择参数，并明确禁止依赖个人本地路径、实验目录、密钥或私有 trace。

## 测试

```bash
python -m pytest
```

当前测试覆盖 tokenizer adapter 行为和 vLLM metrics 解析逻辑。

如果只是修改 README 或示例文档，不需要调用真实 provider；如果修改代理逻辑，建议至少运行单元测试。

## 安全与提交边界

请不要提交：

- API key、`.env`、本地配置、认证文件。
- `traces/*.jsonl`、raw request、raw response、日志和 provider 实验结果。
- `node_modules/`、`__pycache__/`、`.pytest_cache/`、下载的 tokenizer 大目录。
- 本地机器路径、SSH alias、隧道端口、私有模型服务信息。

公开仓库应只包含代理源码、测试、文档、必要的轻量 tokenizer 资源和空 trace 目录占位文件。

## English Summary

`cache_hit_proxy` is a local OpenAI-compatible proxy for observing prompt tokens, prompt-cache estimates, and provider-reported usage on `POST /v1/chat/completions`.

Quick start:

```bash
python -m pip install -r requirements.txt
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

Point your client to:

```text
http://127.0.0.1:8787/v1
```

For automation agents, see [`AGENTS.md`](AGENTS.md).
