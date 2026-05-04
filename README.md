# cache_hit_proxy

一个用于估算 Prompt Cache 命中情况的本地 API 代理工具，同时保持上游 API 语义不变。

## 中文说明

### 项目定位

`cache_hit_proxy` 的目标是：

- 在不改动上游 API 的前提下，透明代理 `chat/completions` 请求。
- 本地估算缓存命中（估计值）。
- 读取上游响应中的缓存字段（真实值）。
- 输出估计值与真实值对比，帮助调试缓存策略。

### 当前支持与后续规划

- 当前已实现：DeepSeek 路径（tokenizer、编码规则、缓存估算流程）。
- 后续计划：扩展到更多接口/Provider（保持相同的估算与对比框架）。
- 因此，`deepseek_tokenizer` 目录是“当前实现依赖”，不是项目最终唯一方向。

### 功能概览

- 接收 OpenAI 兼容 `POST /v1/chat/completions`。
- 请求体原样转发到上游 API。
- 响应体原样返回给客户端。
- 可从普通 JSON 响应和 SSE 流式响应中读取 `usage` 字段。
- 每个进程自动生成 `session_id`。
- 每个请求自动生成递增 `request_id`（`req_0001`、`req_0002`...）。
- 使用 DeepSeek tokenizer 统计本地输入 token 数。
- token 化前使用 DeepSeek V4 编码规则构造 prompt。
- 使用保守的精确前缀匹配估算缓存命中。
- 从响应 `usage` 中读取 `actual_input_tokens`、`actual_cached_tokens`。
- 将请求记录写入 JSONL 日志，并打印终端摘要。

### 快速开始

1. 安装依赖：

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

2. 准备 tokenizer（当前 DeepSeek 实现需要）：

```powershell
Invoke-WebRequest -Uri "https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip" -OutFile "deepseek_tokenizer.zip"
Expand-Archive -LiteralPath "deepseek_tokenizer.zip" -DestinationPath ".\\_tmp_deepseek_tokenizer"
$src = (Get-ChildItem ".\\_tmp_deepseek_tokenizer" -Directory | Select-Object -First 1).FullName
Move-Item -LiteralPath $src -Destination ".\\deepseek_tokenizer"
Remove-Item -LiteralPath ".\\_tmp_deepseek_tokenizer" -Recurse -Force
Remove-Item -LiteralPath ".\\deepseek_tokenizer.zip" -Force
```

3. 启动代理：

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-dir ./deepseek_tokenizer \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --max-history-requests 2000
```

4. 客户端接入：

- 将客户端 `base_url` 从 `https://api.deepseek.com` 改为 `http://127.0.0.1:8787`。
- 请求路径保持 `/v1/chat/completions` 不变。

### 估算策略（当前实现）

- 只做“从 token 0 开始”的精确前缀匹配。
- 仅比较同模型历史请求。
- 每个请求持久化两个边界前缀单元：
- 请求输入边界。
- 模型输出边界。
- 估计命中 token = 可匹配到的最长边界前缀单元长度。
- 按块大小对齐（默认 `64`）。
- 当前不实现共享前缀自动沉淀与固定间隔持久化。

### 遗忘与淘汰

为避免历史无限增长，工具实现了本地淘汰机制：

- 空闲 TTL 淘汰：超过 `--cache-idle-ttl-hours` 未命中的历史会被移除。
- 容量上限淘汰：最多保留 `--max-history-requests` 条历史。
- 命中续期：被匹配到的历史项会刷新空闲计时。

### 参数说明

- `--port`：本地代理端口。
- `--target-base-url`：上游 API 基地址。
- `--tokenizer-dir`：本地 tokenizer 目录（当前 DeepSeek 实现使用）。
- `--block-size`：估算器使用的缓存块大小。
- `--cache-idle-ttl-hours`：空闲淘汰小时数。
- `--max-history-requests`：内存历史最大条数。

### 日志与字段

- 日志文件：`traces/{session_id}.jsonl`
- 字段包括：
- `session_id`
- `request_id`
- `timestamp`
- `model`
- `messages`
- `canonical_text`
- `local_input_tokens`
- `estimated_cached_tokens`
- `estimated_cache_hit_rate`
- `matched_request_id`
- `match_strategy`
- `actual_input_tokens`
- `actual_cached_tokens`
- `actual_cache_hit_rate`
- `usage_source`
- `difference_tokens`
- `status`

说明：

- `usage_source` 为 `json_usage` 时，表示命中信息来自普通 JSON 响应。
- `usage_source` 为 `sse_usage` 时，表示命中信息来自 SSE 流式响应中的 `data:` 事件。
- 不会记录敏感请求头（如 `Authorization`）。

### 目录结构

```text
cache_hit_proxy/
|-- README.md
|-- requirements.txt
|-- main.py
|-- request_recorder.py
|-- cache_estimator.py
|-- usage_reader.py
|-- deepseek_encoding.py
|-- deepseek_tokenizer/
|   |-- tokenizer.json
|   |-- tokenizer_config.json
|   `-- deepseek_tokenizer.py
`-- traces/
    `-- .gitkeep
```

### 已知限制（V1）

- 当前仅支持 `POST /v1/chat/completions`。
- 单进程单会话（session）。
- 当前实现绑定 DeepSeek tokenizer 与编码规则。
- 仅做前缀估算，不覆盖 provider 全部缓存内部策略。
- 当前支持从 SSE 流式响应中提取 `usage` 用于观测，但代理本身仍会先缓冲完整上游响应，再返回给客户端。
- 当 `usage` 缺少缓存字段时，状态为 `actual_cache_unknown`。

---

## English

### Project Scope

`cache_hit_proxy` is a local proxy used to estimate prompt-cache hits while keeping upstream API semantics unchanged.

It is designed to:

- transparently proxy `chat/completions` requests,
- estimate cache hits locally,
- read actual cache-related usage fields from upstream responses,
- compare estimated vs actual metrics for debugging and analysis.

### Current Support and Roadmap

- Implemented now: DeepSeek path (tokenizer, encoding rules, cache estimation flow).
- Planned next: extend the same framework to additional providers/interfaces.
- So `deepseek_tokenizer` is a current implementation dependency, not the final single-provider direction.

### Features

- Accepts OpenAI-compatible `POST /v1/chat/completions`.
- Forwards request body to upstream without modification.
- Returns upstream response body to client without modification.
- Reads `usage` from both standard JSON responses and SSE streaming responses.
- Auto-generates a `session_id` per process.
- Auto-generates incremental `request_id` per request.
- Uses DeepSeek tokenizer for local token counting.
- Applies DeepSeek V4 prompt encoding rules before tokenization.
- Estimates cache hit with conservative exact-prefix matching.
- Reads `actual_input_tokens` and `actual_cached_tokens` from response usage.
- Writes JSONL traces and prints per-request summaries.

### Quick Start

1. Install dependencies:

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

2. Prepare tokenizer files (required for current DeepSeek implementation):

```powershell
Invoke-WebRequest -Uri "https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip" -OutFile "deepseek_tokenizer.zip"
Expand-Archive -LiteralPath "deepseek_tokenizer.zip" -DestinationPath ".\\_tmp_deepseek_tokenizer"
$src = (Get-ChildItem ".\\_tmp_deepseek_tokenizer" -Directory | Select-Object -First 1).FullName
Move-Item -LiteralPath $src -Destination ".\\deepseek_tokenizer"
Remove-Item -LiteralPath ".\\_tmp_deepseek_tokenizer" -Recurse -Force
Remove-Item -LiteralPath ".\\deepseek_tokenizer.zip" -Force
```

3. Run proxy:

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-dir ./deepseek_tokenizer \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --max-history-requests 2000
```

4. Client integration:

- Change client `base_url` from `https://api.deepseek.com` to `http://127.0.0.1:8787`.
- Keep request path `/v1/chat/completions`.

### Estimation Strategy (Current)

- Exact prefix match only (must start at token 0).
- Compare only against history with the same model.
- Persist two boundary prefix units per request:
- request-input boundary,
- model-output boundary.
- Estimated cached tokens = longest matched boundary prefix unit length.
- Prefix units are snapped by block size (default `64`).
- No shared-prefix persistence or fixed-interval persistence in current version.

### Forgetting / Eviction

To avoid unbounded growth:

- Idle TTL eviction: remove history entries not used for `--cache-idle-ttl-hours`.
- Size cap eviction: keep up to `--max-history-requests` entries.
- Hit refresh: matched entries refresh their idle timer.

### Arguments

- `--port`: local proxy port.
- `--target-base-url`: upstream API base URL.
- `--tokenizer-dir`: local tokenizer directory (used by current DeepSeek path).
- `--block-size`: cache block size used by estimator.
- `--cache-idle-ttl-hours`: idle eviction threshold in hours.
- `--max-history-requests`: max in-memory history entries.

### Logging

- Trace file: `traces/{session_id}.jsonl`
- Fields:
- `session_id`
- `request_id`
- `timestamp`
- `model`
- `messages`
- `canonical_text`
- `local_input_tokens`
- `estimated_cached_tokens`
- `estimated_cache_hit_rate`
- `matched_request_id`
- `match_strategy`
- `actual_input_tokens`
- `actual_cached_tokens`
- `actual_cache_hit_rate`
- `usage_source`
- `difference_tokens`
- `status`

Notes:

- `usage_source = json_usage` means the cache metrics came from a standard JSON response.
- `usage_source = sse_usage` means the cache metrics were extracted from SSE `data:` events.
- Sensitive headers (e.g. `Authorization`) are not logged.

### V1 Limits

- Supports only `POST /v1/chat/completions`.
- Single process / single auto-generated session.
- Current implementation is tied to DeepSeek tokenizer and encoding rules.
- Prefix-based approximation only, not a full provider-side cache simulation.
- SSE usage extraction is supported for observability, but the proxy still buffers the full upstream response before returning it to the client.
- If cache fields are absent in usage, status is `actual_cache_unknown`.
