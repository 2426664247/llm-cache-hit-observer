# cache_hit_proxy

一个用于估算 Prompt Cache 命中情况的本地 API 代理工具，同时保持上游 API 语义不变。  
A local API proxy for estimating prompt-cache hit behavior while preserving upstream API semantics.

## 目录 / Table of Contents

- [快速开始](#快速开始--quick-start)
- [功能概览](#功能概览--what-it-does)
- [估算规则](#估算规则对齐文档思路--estimation-rules-doc-aligned)
- [遗忘与淘汰](#遗忘与淘汰--forgetting--eviction)
- [安装](#安装--install)
- [准备 Tokenizer](#准备-deepseek-tokenizer-文件--prepare-deepseek-tokenizer-files)
- [启动代理](#启动代理--run-proxy)
- [接入客户端](#接入现有客户端--connect-your-existing-client)
- [日志与字段](#日志--logging)
- [限制](#v1-限制--v1-limits)

## 快速开始 / Quick Start

- 1) 安装依赖  
  Install dependencies

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

- 2) 准备 tokenizer 文件到 `./deepseek_tokenizer`  
  Prepare tokenizer files under `./deepseek_tokenizer`

- 3) 启动代理（默认端口 8787）  
  Run proxy (default port 8787)

```bash
python main.py --port 8787 --target-base-url https://api.deepseek.com --tokenizer-dir ./deepseek_tokenizer
```

- 4) 客户端把 `base_url` 指向本地代理（路径仍是 `/v1/chat/completions`）  
  Point client `base_url` to the local proxy (path stays `/v1/chat/completions`)

**期望输出（示例） / Expected output (example)**

```
[Session session_YYYYMMDD_HHMMSS] req_0002
Model: deepseek-v4-flash
Local input tokens: 2319
Actual input tokens: 2319
Estimated cached tokens: 2240
Actual cached tokens: 128
```

## 功能概览 / What It Does

- 接收 OpenAI 兼容的 `POST /v1/chat/completions` 请求。  
  Accepts OpenAI-compatible `POST /v1/chat/completions`.
- 请求体原样转发到上游 API。  
  Forwards the request body to upstream API without modification.
- 响应体原样返回给客户端。  
  Returns upstream response body to client without modification.
- 每个代理进程自动生成一个 `session_id`。  
  Auto-generates one `session_id` per proxy process.
- 每次请求自动生成递增 `request_id`（`req_0001`, `req_0002`, ...）。  
  Auto-generates incremental `request_id` per request.
- 使用 DeepSeek tokenizer 统计本地输入 token 数。  
  Uses DeepSeek tokenizer for local input token counting.
- token 化前使用 DeepSeek V4 编码规则构造 prompt。  
  Uses DeepSeek V4 prompt encoding rules before tokenization.
- 采用保守的“精确前缀匹配”估算缓存命中。  
  Estimates cache hit with conservative exact prefix matching.
- 从响应 `usage` 读取真实值（`actual_input_tokens`, `actual_cached_tokens`）。  
  Reads actual usage from API response.
- 将结果写入 JSONL 日志，并在终端打印摘要。  
  Writes JSONL traces and prints per-request summary to terminal.

## 项目结构 / Project Structure

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

## 估算规则（对齐文档思路） / Estimation Rules (Doc-Aligned)

- 只做“从 token 0 开始”的精确前缀匹配。  
  Exact prefix match only (must start from token 0).
- 只与同模型历史请求进行比较。  
  Compare against previous requests with the same model.
- 每个请求仅持久化两类边界单元：请求输入边界、模型输出边界。  
  Persist only two boundary units per request: request-input boundary and model-output boundary.
- 估计命中 token = 可匹配到的最长边界前缀单元长度。  
  Estimated cached tokens = longest matched boundary prefix unit.
- 所有单元按块大小对齐（默认 `64`）。  
  Units are snapped to block size (default `64`).
- 不实现跨旧历史的共享前缀沉淀。  
  No shared/common-prefix persistence across older history.
- 不实现固定间隔持久化。  
  No fixed-interval persistence.

## 遗忘与淘汰 / Forgetting & Eviction

DeepSeek 文档对缓存清理描述为 best-effort：不再使用后会自动清理，通常在数小时到数天。  
DeepSeek docs describe cleanup as best-effort: once no longer used, cache is automatically cleared, usually within hours to days.

本工具在本地额外增加淘汰策略，避免历史无限增长：  
This proxy adds local eviction to avoid unbounded growth:

- 空闲 TTL 淘汰：超过 `--cache-idle-ttl-hours` 未命中的历史会被移除。  
  Idle TTL eviction: remove history entries not used for `--cache-idle-ttl-hours`.
- 容量上限淘汰：内存历史最多保留 `--max-history-requests` 条。  
  Size cap eviction: keep at most `--max-history-requests` entries in memory.
- 命中续期：被匹配到的历史项会刷新空闲计时。  
  Hit refresh: matched entries refresh their idle timer.

## 安装 / Install

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

## 准备 DeepSeek Tokenizer 文件 / Prepare DeepSeek Tokenizer Files

请将 tokenizer 文件放到本地目录（默认：`cache_hit_proxy/deepseek_tokenizer`）。  
Place tokenizer files under a local directory (default: `cache_hit_proxy/deepseek_tokenizer`).

PowerShell 示例 / PowerShell example:

```powershell
Invoke-WebRequest -Uri "https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip" -OutFile "deepseek_tokenizer.zip"
Expand-Archive -LiteralPath "deepseek_tokenizer.zip" -DestinationPath ".\\_tmp_deepseek_tokenizer"
$src = (Get-ChildItem ".\\_tmp_deepseek_tokenizer" -Directory | Select-Object -First 1).FullName
Move-Item -LiteralPath $src -Destination ".\\deepseek_tokenizer"
Remove-Item -LiteralPath ".\\_tmp_deepseek_tokenizer" -Recurse -Force
Remove-Item -LiteralPath ".\\deepseek_tokenizer.zip" -Force
```

## 启动代理 / Run Proxy

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-dir ./deepseek_tokenizer \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --max-history-requests 2000
```

参数说明 / Arguments:

- `--port`：本地代理端口。  
  Local proxy port.
- `--target-base-url`：上游 API 基地址。  
  Upstream API base URL.
- `--tokenizer-dir`：本地 DeepSeek tokenizer 目录。  
  Local DeepSeek tokenizer directory.
- `--block-size`：估算器使用的缓存块大小。  
  Cache storage unit size used by estimator.
- `--cache-idle-ttl-hours`：历史空闲超过该小时数后淘汰。  
  Evict inactive history after this many hours.
- `--max-history-requests`：内存历史最大条数。  
  Max in-memory history entries.

## 接入现有客户端 / Connect Your Existing Client

将客户端 `base_url` 从上游地址改为本地代理：  
Change client `base_url` from upstream to local proxy:

- `https://api.deepseek.com` -> `http://127.0.0.1:8787`

接口路径保持 `/v1/chat/completions` 不变。  
Keep API path as `/v1/chat/completions`.

## 日志 / Logging

每次请求会追加写入：`traces/{session_id}.jsonl`。  
Each request is appended to `traces/{session_id}.jsonl`.

主要字段 / Main fields:

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
- `difference_tokens`
- `status`

不会记录敏感请求头（例如 `Authorization`）。  
Sensitive headers (like `Authorization`) are not logged.

## V1 限制 / V1 Limits

- 仅支持 `POST /v1/chat/completions`。  
  Supports only `POST /v1/chat/completions`.
- 单进程、单自动 session。  
  Single process / single auto-generated session.
- 仅支持 DeepSeek tokenizer 与对应编码规则。  
  DeepSeek-only tokenizer and encoding rules.
- 仅做前缀估算，不覆盖 provider 侧完整缓存策略细节。  
  Prefix-based estimate only; does not implement full provider-side cache policies.
- 若 `usage` 缺少缓存字段，则状态为 `actual_cache_unknown`。  
  If cache fields are absent in usage, status is `actual_cache_unknown`.
