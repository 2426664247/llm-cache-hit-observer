# cache_hit_proxy

Minimal local API proxy for estimating prompt-cache hit rate while preserving upstream API semantics.

## What It Does

- Accepts OpenAI-compatible `POST /v1/chat/completions`
- Forwards request body to upstream API without modification
- Returns upstream response body to client without modification
- Auto-generates one `session_id` per proxy process
- Auto-generates incremental `request_id` per request (`req_0001`, `req_0002`, ...)
- Uses DeepSeek tokenizer for local input token counting
- Uses DeepSeek V4 prompt encoding rules before tokenization
- Estimates cache hit with conservative exact prefix matching
- Reads actual usage from API response (`actual_input_tokens`, `actual_cached_tokens`)
- Writes JSONL traces and prints per-request summary to terminal

## Project Structure

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

## Estimation Rules (Doc-Aligned)

- Exact prefix match only (must start from token 0).
- Compare against all previous requests with the same model.
- Persist only two boundary units per request:
  - request-input boundary
  - model-output boundary
- Estimated cached tokens = longest matched boundary prefix unit.
- Units are snapped to block size (default `64`).
- No shared/common-prefix persistence across older history.
- No fixed-interval persistence.

## Forgetting / Eviction

DeepSeek docs describe cache cleanup as best-effort: once no longer used, cache is automatically cleared, usually within hours to days.

This proxy adds local eviction to avoid unbounded growth:

- Idle TTL eviction: history entries not used for `--cache-idle-ttl-hours` are removed.
- Size cap eviction: keep at most `--max-history-requests` entries in memory.
- Hit refresh: if an entry is matched for cache estimation, its idle timer is refreshed.

## Install

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

## Prepare DeepSeek Tokenizer Files

Place tokenizer files under a local directory (default: `cache_hit_proxy/deepseek_tokenizer`).

PowerShell example:

```powershell
Invoke-WebRequest -Uri "https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip" -OutFile "deepseek_tokenizer.zip"
Expand-Archive -LiteralPath "deepseek_tokenizer.zip" -DestinationPath ".\\_tmp_deepseek_tokenizer"
$src = (Get-ChildItem ".\\_tmp_deepseek_tokenizer" -Directory | Select-Object -First 1).FullName
Move-Item -LiteralPath $src -Destination ".\\deepseek_tokenizer"
Remove-Item -LiteralPath ".\\_tmp_deepseek_tokenizer" -Recurse -Force
Remove-Item -LiteralPath ".\\deepseek_tokenizer.zip" -Force
```

## Run Proxy

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-dir ./deepseek_tokenizer \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --max-history-requests 2000
```

Arguments:

- `--port`: local proxy port
- `--target-base-url`: upstream API base URL
- `--tokenizer-dir`: local DeepSeek tokenizer directory
- `--block-size`: cache storage unit size used by estimator
- `--cache-idle-ttl-hours`: evict inactive history after this many hours
- `--max-history-requests`: max in-memory history entries

## How To Connect Your Existing Client

Change client `base_url` from upstream to local proxy:

- from: `https://api.deepseek.com`
- to: `http://127.0.0.1:8787`

Keep API path as `/v1/chat/completions`.

## Logging

Each request is appended to:

- `traces/{session_id}.jsonl`

Fields include:

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

Sensitive headers (like `Authorization`) are not logged.

## V1 Limits

- Supports only `POST /v1/chat/completions`
- Single process / single auto-generated session
- DeepSeek-only tokenizer and encoding rules
- Prefix-based estimate only (no ttl/session-eviction/provider-specific cache policies)
- If cache fields are absent in usage, status is `actual_cache_unknown`
