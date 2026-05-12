# Cache Hit Proxy

Local OpenAI-compatible proxy for observing prompt-token usage and prompt-cache behavior.

`cache_hit_proxy` forwards `POST /v1/chat/completions` requests to an upstream model API, estimates prompt/cache tokens locally, reads provider-reported usage when available, and writes per-request JSONL traces for later analysis.

## What It Does

| Feature | Description |
| --- | --- |
| Transparent proxy | Accepts OpenAI-compatible chat completions requests and forwards them upstream. |
| Local token estimation | Estimates prompt tokens with local tokenizer presets or vLLM `/tokenize`. |
| Cache-hit estimation | Compares each request with in-memory history to estimate reusable prefix/cache tokens. |
| Usage extraction | Reads actual input/cached tokens from JSON responses, SSE events, or vLLM metrics. |
| Trace logging | Writes structured traces to `traces/{session_id}.jsonl`. |

## Requirements

- Python 3.12 or newer is recommended.
- An upstream OpenAI-compatible chat completions endpoint.
- Provider credentials, if required, supplied by the client request headers.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional tokenizer assets:

```bash
python download_tokenizers.py
python download_tokenizers.py --check-only
```

The downloader fetches tokenizer/chat-template files only. It does not download model weights.

## Quick Start

Start the proxy for a generic OpenAI-compatible provider:

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

Point your client to the local proxy:

```text
http://127.0.0.1:8787/v1
```

Keep the same `Authorization` header or API key setting you would normally send to the upstream provider. The proxy forwards non-hop-by-hop headers upstream.

## Common Modes

| Mode | Use When |
| --- | --- |
| `simple_streaming` | You want a generic proxy for OpenAI-compatible chat completions. |
| `openclaw_agent` | You are measuring OpenClaw agent requests and want raw-body based estimation. |
| `piai_probe` | You are analyzing pi-ai style final provider payloads. |
| `vllm_probe` | You are measuring a local/remote vLLM service with `/tokenize` and `/metrics`. |

### vLLM Probe

```bash
python main.py \
  --port 8787 \
  --target-base-url http://127.0.0.1:8000/v1 \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe \
  --session-id vllm_probe
```

If your chat endpoint, tokenizer endpoint, and metrics endpoint do not share the same base URL, pass them explicitly:

```bash
python main.py \
  --port 8787 \
  --target-chat-url http://127.0.0.1:8000/v1/chat/completions \
  --vllm-tokenize-url http://127.0.0.1:8000/tokenize \
  --vllm-metrics-url http://127.0.0.1:8000/metrics \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe
```

For clean per-request metric deltas, avoid unrelated traffic to the same vLLM process while probing.

## Tokenizer Presets

| Preset | Behavior |
| --- | --- |
| `deepseek-v4-pro` | Uses the bundled DeepSeek tokenizer and prompt renderer. |
| `glm-5.1` | Uses local GLM tokenizer assets and chat template. |
| `qwen3-coder-plus` | Uses local Qwen3-Coder tokenizer assets and chat template. |
| `kimi-k2.6` | Uses local Kimi tokenizer assets and chat template. |
| `doubao-seed-2-0-code-preview-260215` / `volcanoengine` | Uses the DeepSeek fallback tokenizer; expect estimation drift. |
| `vllm` | Calls a running vLLM server's `/tokenize` endpoint. |

Use `--tokenizer-dir` when your tokenizer assets are stored outside the default locations.

## Trace Output

Each proxied request appends one JSON object to:

```text
traces/{session_id}.jsonl
```

Useful fields include:

| Field | Meaning |
| --- | --- |
| `predicted_input_tokens` | Local prompt-token estimate. |
| `actual_input_tokens` | Provider/vLLM-reported prompt tokens, if available. |
| `estimated_cached_tokens` | Local cached-token estimate. |
| `actual_cached_tokens` | Provider/vLLM-reported cached tokens, if available. |
| `difference_tokens` | `actual_cached_tokens - estimated_cached_tokens`. |
| `status` | Diagnostic status for the request. |
| `conversation_mode` | Proxy processing mode. |
| `tokenizer_preset` | Tokenizer preset used for the request. |

Traces can contain request metadata and optional raw request bodies. Do not commit traces from private workloads.

## Agent Guide

If another coding agent needs to run this tool in a new environment, point it to [`AGENTS.md`](AGENTS.md). That file explains how to choose the correct mode, tokenizer, target URL, and safety settings without relying on any local machine details.

## Tests

```bash
python -m pytest
```

The current tests cover tokenizer adapter behavior and vLLM metrics parsing.
