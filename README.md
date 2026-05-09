# cache_hit_proxy

## 中文文档

`cache_hit_proxy` 是一个本地代理，用于观察 `POST /v1/chat/completions` 请求的 prompt token、缓存命中估计和供应商返回的实际 usage。代理会尽量保持上游 OpenAI-compatible API 语义不变，同时在本地写入每次请求的 trace。

### 功能

- 转发请求到上游 OpenAI-compatible API。
- 在本地估算输入 token 和缓存命中 token。
- 从上游响应中读取实际 usage，包括 JSON 响应和 SSE stream event。
- `vllm_probe` 模式下，从 vLLM Prometheus metrics 读取真实 prompt-cache 命中增量。
- 将每次请求记录为 JSONL trace，便于后续分析。

### 安装

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

### Tokenizer 预设

prompt token 估算全部在本地完成，通过 `--tokenizer-preset` 选择 tokenizer：

- `deepseek-v4-pro`：使用内置 DeepSeek tokenizer 和 DeepSeek prompt renderer。
- `glm-5.1`：使用本地 GLM tokenizer 和官方 chat template。
- `qwen3-coder-plus`：使用本地 Qwen3-Coder tokenizer 和官方 chat template。
- `kimi-k2.6`：使用本地 Kimi tokenizer 和官方 chat template。
- `doubao-seed-2-0-code-preview-260215` / `volcanoengine`：fallback 到 DeepSeek tokenizer，可本地估算，但会有误差。
- `vllm`：调用正在运行的 vLLM server `/tokenize`，适合泛用 vLLM OpenAI-compatible 服务。

DeepSeek 和 Doubao fallback 默认使用 `./deepseek_tokenizer`。GLM、Qwen、Kimi 默认使用 `./tokenizers/<preset>`。如需指定其他目录，使用 `--tokenizer-dir`。`vllm` 默认由 `--target-base-url` 推导 `/tokenize`，也可通过 `--vllm-tokenize-url` 显式指定。

只下载或检查 tokenizer 文件：

```bash
python download_tokenizers.py
python download_tokenizers.py --check-only
```

下载脚本只拉 tokenizer、chat template 等文件，不下载模型权重。

### Conversation Mode

通过 `--conversation-mode` 选择不同的请求处理方式：

- `simple_streaming`：默认模式，适合普通 chat/completions 请求，输入 token 来源为 `deepseek_prompt_encoding`。
- `openclaw_agent`：适合 OpenClaw agent 请求，输入 token 来源为 `openclaw_raw_body`，并加入 OpenClaw 专用的多轮缓存估计修正。
- `piai_probe`：适合 pi-ai 最终 OpenAI-compatible payload，会读取顶层 `tools`、`thinking`、`reasoning_effort`，并从 content block 中提取用户文本。
- `vllm_probe`：适合观察 vLLM prefix cache，使用 block size `784`，实际缓存命中来自 vLLM `/metrics`。

### OpenClaw Agent 模式

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-preset deepseek-v4-pro \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --conversation-mode openclaw_agent \
  --raw-request-capture none
```

默认情况下，内存中的历史请求只按 idle TTL 清理，不限制请求数量。只有在需要手动限制内存时，才设置 `--max-history-requests`。

### 普通 Streaming 模式

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

GLM 本地 tokenizer 示例：

```bash
python main.py \
  --port 8787 \
  --target-base-url https://open.bigmodel.cn/api/paas/v4 \
  --tokenizer-preset glm-5.1 \
  --conversation-mode simple_streaming
```

### pi-ai Probe 模式

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode piai_probe
```

### vLLM Probe 模式

```bash
python main.py \
  --port 8787 \
  --target-base-url http://127.0.0.1:8000/v1 \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe \
  --session-id vllm_probe_session
```

`vllm_probe` 会优先从 vLLM `/metrics` 的 `vllm:cache_config_info` 读取真实 `block_size`；读不到时回退到 `--block-size`，默认 `784`。如果使用 `--target-chat-url` 而不是 `--target-base-url`，还需要传入 `--vllm-metrics-url` 和 `--vllm-tokenize-url`：

```bash
python main.py \
  --port 8787 \
  --target-chat-url http://127.0.0.1:8000/v1/chat/completions \
  --vllm-metrics-url http://127.0.0.1:8000/metrics \
  --vllm-tokenize-url http://127.0.0.1:8000/tokenize \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe
```

使用 vLLM probe 时，尽量避免同一个 vLLM 进程上同时有无关流量，否则 metrics delta 可能混入其他请求。

### OpenClaw 接入

- 将 OpenClaw 模型的 `baseUrl` 指向代理，例如 `http://127.0.0.1:8787/v1`。
- API 类型保持为 `openai-completions`。

`openclaw_agent` 模式会读取代理收到的原始 HTTP body，并用相同 bytes 转发给上游，所以 trace 里的 raw body 更接近 OpenClaw 最终实际请求。

### Tokenizer 验证

如果要对比本地估算和供应商 API 返回的 usage，可以设置环境变量或使用已有本地配置，然后运行：

```bash
python validate_tokenizers.py --sample short
python validate_tokenizers.py --sample long-1000
```

验证脚本会对每个已配置供应商发送一次小 completion 请求，`max_tokens=1`，并生成本地 Markdown 报告。报告文件默认被 git 忽略，且不应包含 API key。

### Trace 字段

常用字段：

- `conversation_mode`
- `input_token_source`
- `tokenizer_preset`
- `tokenizer_effective_preset`
- `tokenizer_runtime`
- `tokenizer_dir`
- `tokenizer_warning`
- `predicted_input_tokens`
- `actual_input_tokens`
- `input_tokens_difference`
- `raw_request_body_sha256`
- `raw_request_body_size_bytes`
- `raw_request_body_tokenizer_tokens`
- `raw_request_capture_mode`
- `estimated_cached_tokens`
- `actual_cached_tokens`
- `actual_uncached_tokens`
- `cache_unit_source`
- `cache_unit_fallback_reason`
- `cache_block_size`
- `openclaw_session_cache_floor_tokens`
- `openclaw_global_cache_floor_tokens`
- `vllm_metrics_url`
- `vllm_prompt_tokens_cached_delta`
- `vllm_metrics_error`
- `vllm_block_size_warning`

Trace 路径：

```text
traces/{session_id}.jsonl
```

### 说明

- `difference_tokens` 表示 `actual_cached_tokens - estimated_cached_tokens`。
- 当 `difference_tokens < -1280` 时，`status` 会标记为 `overestimated`，用于发现本地估计缓存命中过高的情况。
- `actual_uncached_tokens` 只作为诊断上下文，不直接决定异常状态。
- `piai_probe` 是针对当前分析过的 pi-ai payload 形态做的窄适配，不是通用 tools 模式。
- 如果需要字节级回放请求，可以使用 `--raw-request-capture base64`。

## English Documentation

`cache_hit_proxy` is a local proxy for observing prompt tokens, cache-hit estimates, and provider-reported usage on `POST /v1/chat/completions`. It forwards requests to an upstream OpenAI-compatible API while writing local per-request traces.

### Features

- Proxies requests to upstream OpenAI-compatible APIs.
- Estimates input tokens and cached tokens locally.
- Reads actual usage from upstream JSON responses or SSE stream events.
- In `vllm_probe` mode, reads real prompt-cache hit deltas from vLLM Prometheus metrics.
- Writes per-request trace logs as JSONL.

### Install

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

### Tokenizer Presets

Prompt-token estimation runs locally. Pick a tokenizer with `--tokenizer-preset`:

- `deepseek-v4-pro`: bundled DeepSeek tokenizer and DeepSeek prompt renderer.
- `glm-5.1`: local GLM tokenizer and official chat template.
- `qwen3-coder-plus`: local Qwen3-Coder tokenizer and official chat template.
- `kimi-k2.6`: local Kimi tokenizer and official chat template.
- `doubao-seed-2-0-code-preview-260215` / `volcanoengine`: DeepSeek tokenizer fallback, usable locally with expected drift.
- `vllm`: calls a running vLLM server `/tokenize`, suitable for generic vLLM OpenAI-compatible services.

DeepSeek and Doubao fallback use `./deepseek_tokenizer` by default. GLM, Qwen, and Kimi use `./tokenizers/<preset>` by default. Use `--tokenizer-dir` to override the directory. `vllm` derives `/tokenize` from `--target-base-url` by default, or accepts `--vllm-tokenize-url`.

Download or check tokenizer-only files:

```bash
python download_tokenizers.py
python download_tokenizers.py --check-only
```

The downloader only requests tokenizer and chat-template assets. It does not download model weights.

### Conversation Mode

Use `--conversation-mode` to switch behavior:

- `simple_streaming`: default mode for direct chat/completions requests; input token source is `deepseek_prompt_encoding`.
- `openclaw_agent`: for OpenClaw agent requests; input token source is `openclaw_raw_body`, with OpenClaw-specific multi-turn cache estimation adjustments.
- `piai_probe`: for pi-ai final OpenAI-compatible payloads; reads top-level `tools`, `thinking`, and `reasoning_effort`, and extracts user text from content blocks.
- `vllm_probe`: for vLLM prefix-cache observation; uses block size `784`, and reads actual cache hits from vLLM `/metrics`.

### OpenClaw Agent Mode

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-preset deepseek-v4-pro \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --conversation-mode openclaw_agent \
  --raw-request-capture none
```

By default, in-memory cache history is pruned only by idle TTL and is not capped by request count. Set `--max-history-requests` only when you need a manual memory cap.

### Simple Streaming Mode

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

Example with the GLM local tokenizer:

```bash
python main.py \
  --port 8787 \
  --target-base-url https://open.bigmodel.cn/api/paas/v4 \
  --tokenizer-preset glm-5.1 \
  --conversation-mode simple_streaming
```

### pi-ai Probe Mode

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode piai_probe
```

### vLLM Probe Mode

```bash
python main.py \
  --port 8787 \
  --target-base-url http://127.0.0.1:8000/v1 \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe \
  --session-id vllm_probe_session
```

`vllm_probe` first reads the real `block_size` from `vllm:cache_config_info` on `/metrics`; if unavailable, it falls back to `--block-size`, defaulting to `784`. If you use `--target-chat-url` instead of `--target-base-url`, also pass `--vllm-metrics-url` and `--vllm-tokenize-url`:

```bash
python main.py \
  --port 8787 \
  --target-chat-url http://127.0.0.1:8000/v1/chat/completions \
  --vllm-metrics-url http://127.0.0.1:8000/metrics \
  --vllm-tokenize-url http://127.0.0.1:8000/tokenize \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe
```

When using vLLM probe mode, avoid unrelated concurrent traffic to the same vLLM process if you need precise per-request metric deltas.

### OpenClaw Wiring

- Point the OpenClaw model `baseUrl` to the proxy, for example `http://127.0.0.1:8787/v1`.
- Keep the model API type as `openai-completions`.

`openclaw_agent` reads the exact inbound HTTP body and forwards the same bytes upstream, so the raw body in the trace is close to the final request emitted by OpenClaw.

### Tokenizer Validation

To compare local estimates with provider-reported usage, set provider keys in environment variables or the existing local config, then run:

```bash
python validate_tokenizers.py --sample short
python validate_tokenizers.py --sample long-1000
```

The validation script sends one small completion request per configured provider with `max_tokens=1`, then writes local Markdown reports. Generated reports are ignored by git and should not contain API keys.

### Trace Fields

Common fields:

- `conversation_mode`
- `input_token_source`
- `tokenizer_preset`
- `tokenizer_effective_preset`
- `tokenizer_runtime`
- `tokenizer_dir`
- `tokenizer_warning`
- `predicted_input_tokens`
- `actual_input_tokens`
- `input_tokens_difference`
- `raw_request_body_sha256`
- `raw_request_body_size_bytes`
- `raw_request_body_tokenizer_tokens`
- `raw_request_capture_mode`
- `estimated_cached_tokens`
- `actual_cached_tokens`
- `actual_uncached_tokens`
- `cache_unit_source`
- `cache_unit_fallback_reason`
- `cache_block_size`
- `openclaw_session_cache_floor_tokens`
- `openclaw_global_cache_floor_tokens`
- `vllm_metrics_url`
- `vllm_prompt_tokens_cached_delta`
- `vllm_metrics_error`
- `vllm_block_size_warning`

Trace path:

```text
traces/{session_id}.jsonl
```

### Notes

- `difference_tokens` means `actual_cached_tokens - estimated_cached_tokens`.
- `status` is marked as `overestimated` when `difference_tokens < -1280`, which helps catch cases where the local estimate expected too much cache reuse.
- `actual_uncached_tokens` is recorded as diagnostic context only and does not directly drive anomaly status.
- `piai_probe` is intentionally narrow and targets the currently analyzed pi-ai payload shape, not a generic tools mode.
- Use `--raw-request-capture base64` if you need byte-level request replay.
