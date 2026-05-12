# Agent Usage Guide

This document is for coding agents or automation agents that need to use `cache_hit_proxy` from a freshly cloned repository. It intentionally avoids machine-specific paths, private endpoints, API keys, experiment outputs, and local run history.

## Scope

This repository is the cache-hit proxy tool only.

Do not assume any surrounding workspace exists. Do not require local experiment folders, private model services, personal SSH aliases, raw traces, or provider-specific reports. If those files exist in a developer's working directory, treat them as private context and do not include them in commits or examples.

## What The Tool Does

`cache_hit_proxy` runs a local FastAPI service that accepts OpenAI-compatible `POST /v1/chat/completions` requests, forwards them to an upstream provider, estimates prompt/cache tokens, reads actual usage when possible, and writes JSONL traces.

The local proxy URL exposed to clients is usually:

```text
http://127.0.0.1:8787/v1
```

The upstream URL is supplied by the user or environment and must not be hard-coded.

## Setup Checklist

1. Verify Python:

```bash
python --version
```

Python 3.12+ is recommended.

2. Install dependencies from the repository root or from this directory:

```bash
python -m pip install -r requirements.txt
```

3. If using a local tokenizer preset, ensure tokenizer assets exist:

```bash
python download_tokenizers.py --check-only
```

If assets are missing and the user approves network access, run:

```bash
python download_tokenizers.py
```

For `--tokenizer-preset vllm`, tokenizer assets are not required locally because the proxy calls the vLLM `/tokenize` endpoint.

## Choose The Right Mode

Use this decision table before starting the proxy:

| Environment | `--conversation-mode` | `--tokenizer-preset` | Required URLs |
| --- | --- | --- | --- |
| Generic OpenAI-compatible provider | `simple_streaming` | Best matching local preset | `--target-base-url` or `--target-chat-url` |
| DeepSeek-compatible payloads | `simple_streaming` | `deepseek-v4-pro` | Provider base/chat URL |
| OpenClaw agent traffic | `openclaw_agent` | Usually `deepseek-v4-pro`, unless user specifies otherwise | Provider base/chat URL |
| pi-ai final payload inspection | `piai_probe` | Usually `deepseek-v4-pro`, unless user specifies otherwise | Provider base/chat URL |
| vLLM service with metrics | `vllm_probe` | `vllm` | Chat URL/base URL, `/tokenize`, and `/metrics` |
| Unknown tokenizer | Start with the user's provider-specific preset; if unavailable, disclose expected drift | Matching URL from user | Provider base/chat URL |

Never silently treat tokenizer drift as exact. If the tokenizer is a fallback, say so in the run notes.

## Choose The Target URL

Prefer `--target-base-url` when the upstream follows the standard OpenAI path:

```text
{base}/v1/chat/completions
```

Use:

```bash
--target-base-url https://YOUR_PROVIDER_BASE_URL
```

If the provider uses a custom full endpoint, use:

```bash
--target-chat-url https://YOUR_PROVIDER_FULL_CHAT_COMPLETIONS_URL
```

Do not bake API keys into either URL. Credentials should be sent by the client exactly as they would be sent to the upstream provider, usually through the `Authorization` header.

## Common Start Commands

Generic provider:

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

Provider with a nonstandard chat endpoint:

```bash
python main.py \
  --port 8787 \
  --target-chat-url https://YOUR_PROVIDER_FULL_CHAT_COMPLETIONS_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

OpenClaw agent traffic:

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode openclaw_agent \
  --raw-request-capture none
```

vLLM probe using standard local endpoints:

```bash
python main.py \
  --port 8787 \
  --target-base-url http://YOUR_VLLM_HOST:8000/v1 \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe \
  --session-id vllm_probe
```

vLLM probe using explicit endpoints:

```bash
python main.py \
  --port 8787 \
  --target-chat-url http://YOUR_VLLM_HOST:8000/v1/chat/completions \
  --vllm-tokenize-url http://YOUR_VLLM_HOST:8000/tokenize \
  --vllm-metrics-url http://YOUR_VLLM_HOST:8000/metrics \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe
```

## Client Wiring

After the proxy is running, configure the client application as follows:

```text
base_url = http://127.0.0.1:8787/v1
api_key  = the same value normally used for the upstream provider
model    = the upstream model name
```

The proxy forwards normal request headers upstream. Do not store the API key in repository files, examples, traces, or generated reports.

## Trace Handling

Default trace path:

```text
traces/{session_id}.jsonl
```

Treat traces as private by default. They may contain model names, prompts, request metadata, provider usage, and optionally raw request bodies.

Recommended defaults:

```bash
--raw-request-capture none
```

Use `--raw-request-capture utf8` or `--raw-request-capture base64` only when the user explicitly needs request replay or byte-level debugging.

## Validation And Tests

Run unit tests after code changes:

```bash
python -m pytest
```

Tokenizer validation calls real providers and may consume API quota:

```bash
python validate_tokenizers.py --sample short
python validate_tokenizers.py --sample long-1000
```

Only run provider validation when the user has supplied credentials and asked for provider/tokenizer accuracy checks.

## Safety Rules For Agents

- Do not commit `traces/`, raw requests, raw responses, logs, local configs, API keys, or tokenizer/model artifact directories.
- Do not mention private local paths, SSH aliases, tunnel ports, machine names, or personal provider accounts in public docs.
- Do not assume a specific provider. Ask for or read the target URL/model from the user's environment.
- Do not run long provider experiments unless explicitly requested.
- For vLLM metric probes, warn that unrelated traffic to the same vLLM process can pollute cache-hit deltas.
- If exact tokenizer support is unavailable, report the fallback and expected drift instead of presenting estimates as exact.

## Minimal Agent Run Summary Template

When reporting a run back to the user, include:

```text
Mode: <conversation-mode>
Tokenizer: <tokenizer-preset or tokenizer-dir>
Target: <base URL domain or local host only, no secrets>
Proxy URL: http://127.0.0.1:<port>/v1
Trace: traces/<session-id>.jsonl
Notes: <fallbacks, metric caveats, or validation status>
```
