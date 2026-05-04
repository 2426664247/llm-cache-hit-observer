# cache_hit_proxy

Local proxy for cache-hit estimation and usage comparison on `POST /v1/chat/completions`.

## What it does

- Proxies requests to upstream OpenAI-compatible API without changing endpoint semantics.
- Estimates cache hits locally.
- Reads actual usage from upstream response (`usage` in JSON or SSE stream events).
- Writes per-request trace logs as JSONL.

## Conversation mode selector

Use `--conversation-mode` to switch behavior presets:

- `simple_streaming` (default): for direct/simple streaming chat requests.
  - input token source: `deepseek_prompt_encoding`
- `openclaw_agent`: for OpenClaw agent-style requests.
  - input token source: `openclaw_raw_body`
  - cache estimation includes an OpenClaw-only multi-turn session floor
    derived from prior actual cached tokens in the same proxy session.
    This improves cache-hit estimation when raw request bodies drift
    between turns.
  - first-turn estimation can also use a global floor seeded from historical
    OpenClaw traces (same model) to avoid defaulting to 0 on a cold proxy run.

## Install

```bash
cd cache_hit_proxy
pip install -r requirements.txt
```

## Tokenizer files

Current implementation requires local DeepSeek tokenizer files in `./deepseek_tokenizer`.

## Run (OpenClaw agent mode)

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-dir ./deepseek_tokenizer \
  --block-size 64 \
  --cache-idle-ttl-hours 24 \
  --conversation-mode openclaw_agent \
  --raw-request-capture none
```

By default, in-memory cache history is unlimited by request count and is
pruned only by idle TTL (`--cache-idle-ttl-hours`, default 24 hours). Set
`--max-history-requests` to a positive number only when you want a manual
memory safety cap.

## Run (simple streaming mode)

```bash
python main.py \
  --port 8787 \
  --target-base-url https://api.deepseek.com \
  --tokenizer-dir ./deepseek_tokenizer \
  --conversation-mode simple_streaming
```

## OpenClaw wiring

- Keep OpenClaw model `baseUrl` pointing to proxy, for example `http://127.0.0.1:8787/v1`.
- Keep model API type as `openai-completions`.

## Why openclaw mode is closer to real context

- Proxy reads exact inbound bytes (`request.body()`).
- Proxy forwards upstream with `content=body_bytes`.
- So raw body in proxy trace is byte-level payload from OpenClaw side.

## Trace fields (key)

- `conversation_mode`
- `input_token_source`
- `predicted_input_tokens`
- `actual_input_tokens`
- `input_tokens_difference`
- `raw_request_body_sha256`
- `raw_request_body_size_bytes`
- `raw_request_body_tokenizer_tokens`
- `raw_request_capture_mode`
- `raw_request_body_utf8` (optional)
- `raw_request_body_base64` (optional)
- `raw_request_body_truncated`
- `estimated_cached_tokens`
- `actual_cached_tokens`
- `actual_uncached_tokens`
- `cache_estimation_diff_threshold_tokens`
- `estimation_denominator_tokens`
- `cache_unit_source`
- `cache_unit_fallback_reason`
- `openclaw_session_cache_floor_tokens`
- `openclaw_global_cache_floor_tokens`

Trace file path:

- `traces/{session_id}.jsonl`

## Notes

- `difference_tokens` is `actual_cached_tokens - estimated_cached_tokens`.
- `status` flags `overestimated` when `difference_tokens < -1280`; otherwise it is
  `normal` when actual usage is available. This catches cases where the proxy expected
  cache reuse but the upstream service returned much less cached context.
- `actual_uncached_tokens` is recorded as diagnostic context only; it does not drive
  anomaly status because prompt growth or topic shifts can legitimately add uncached input.
- `raw-stream` in OpenClaw is output-stream oriented; it is not a request-payload logger.
- `cacheTrace stream:context` is useful context snapshot, but not guaranteed to be final HTTP request body.
- If you need byte-level replay of raw requests, run with `--raw-request-capture base64`.
