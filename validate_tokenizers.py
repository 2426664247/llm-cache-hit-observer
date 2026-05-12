from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from request_recorder import RequestRecorder
from usage_reader import read_actual_usage
from tokenizer_adapters import default_tokenizer_dir_for_preset, normalize_tokenizer_preset


VALIDATION_SHORT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise tokenizer validation assistant.",
    },
    {
        "role": "user",
        "content": "请只回答 OK。Token test: cache prefix 12345.",
    },
]


def _build_long_1000_messages() -> List[Dict[str, str]]:
    segment = (
        "段落{idx:03d}: 请比较缓存命中率、前缀复用和 tokenizer 差异。"
        "Use stable English words, numbers 12345, JSON keys like cache_prefix "
        "and request_id, then keep the wording identical.\n"
    )
    return [
        {
            "role": "system",
            "content": "You are a concise tokenizer validation assistant. Return only OK.",
        },
        {
            "role": "user",
            "content": "".join(segment.format(idx=i) for i in range(1, 24))
            + "最后请只回答 OK。",
        },
    ]


def _validation_messages(sample: str) -> List[Dict[str, str]]:
    if sample == "short":
        return VALIDATION_SHORT_MESSAGES
    if sample == "long-1000":
        return _build_long_1000_messages()
    raise ValueError(f"unsupported validation sample: {sample}")


@dataclass(frozen=True)
class ProviderCase:
    provider: str
    model: str
    tokenizer_preset: str
    base_url: str
    api_key: Optional[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_config_text() -> str:
    path = _repo_root() / "config" / "config.txt"
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_config_pair(config_text: str, label: str) -> tuple[Optional[str], Optional[str]]:
    pattern = re.compile(rf"^{re.escape(label)}:(?P<key>\S+)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(config_text)
    if not match:
        return None, None
    key = match.group("key").strip()
    rest = config_text[match.end() :].splitlines()
    model = None
    for line in rest:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            break
        model = stripped
        break
    return key, model


def _env_or_config_key(env_name: str, config_text: str, label: str) -> Optional[str]:
    value = os.getenv(env_name)
    if value:
        return value
    key, _ = _extract_config_pair(config_text, label)
    return key


def _env_or_config_model(env_name: str, config_text: str, label: str, default: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    _, model = _extract_config_pair(config_text, label)
    return model or default


def _canonical_model(provider: str, value: str, default: str) -> str:
    env_override = os.getenv(f"{provider.upper()}_MODEL")
    if env_override:
        return env_override

    normalized = value.strip()
    folded = normalized.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4pro": "deepseek-v4-pro",
        "glm-5.1": "glm-5.1",
        "glm5.1": "glm-5.1",
        "qwen3.6-plus": "qwen3-coder-plus",
        "qwen3-6-plus": "qwen3-coder-plus",
        "qwen3-coder-plus": "qwen3-coder-plus",
        "kimi-k2.6": "kimi-k2.6",
        "kimi-k2-6": "kimi-k2.6",
        "kimi-k2.6": "kimi-k2.6",
        "doubao-seed-2.0-code": "doubao-seed-2-0-code-preview-260215",
        "doubao-seed-2-0-code": "doubao-seed-2-0-code-preview-260215",
        "doubao-seed-2-0-code-preview-260215": "doubao-seed-2-0-code-preview-260215",
    }
    return aliases.get(folded, default if normalized != default and provider in {"qwen", "kimi"} else normalized)


def _provider_cases(config_text: str) -> List[ProviderCase]:
    deepseek_model = _env_or_config_model(
        "DEEPSEEK_MODEL", config_text, "deepseek", "deepseek-v4-pro"
    )
    glm_model = _env_or_config_model("GLM_MODEL", config_text, "GLM", "glm-5.1")
    qwen_model = _env_or_config_model(
        "QWEN_MODEL", config_text, "Qwen", "qwen3-coder-plus"
    )
    kimi_model = _env_or_config_model("KIMI_MODEL", config_text, "Kimi", "kimi-k2.6")
    volcano_model = _env_or_config_model(
        "VOLCANO_MODEL",
        config_text,
        "VolcanoEngine",
        "doubao-seed-2-0-code-preview-260215",
    )
    return [
        ProviderCase(
            provider="DeepSeek",
            model=_canonical_model("deepseek", deepseek_model, "deepseek-v4-pro"),
            tokenizer_preset="deepseek-v4-pro",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=_env_or_config_key("DEEPSEEK_API_KEY", config_text, "deepseek"),
        ),
        ProviderCase(
            provider="GLM",
            model=_canonical_model("glm", glm_model, "glm-5.1"),
            tokenizer_preset="glm-5.1",
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            api_key=_env_or_config_key("GLM_API_KEY", config_text, "GLM"),
        ),
        ProviderCase(
            provider="Qwen",
            model=_canonical_model("qwen", qwen_model, "qwen3-coder-plus"),
            tokenizer_preset="qwen3-coder-plus",
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=_env_or_config_key("QWEN_API_KEY", config_text, "Qwen"),
        ),
        ProviderCase(
            provider="Kimi",
            model=_canonical_model("kimi", kimi_model, "kimi-k2.6"),
            tokenizer_preset="kimi-k2.6",
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            api_key=_env_or_config_key("KIMI_API_KEY", config_text, "Kimi"),
        ),
        ProviderCase(
            provider="VolcanoEngine",
            model=_canonical_model(
                "volcano",
                volcano_model,
                "doubao-seed-2-0-code-preview-260215",
            ),
            tokenizer_preset="doubao-seed-2-0-code-preview-260215",
            base_url=os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            api_key=_env_or_config_key("VOLCANO_API_KEY", config_text, "VolcanoEngine"),
        ),
    ]


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _extract_prompt_tokens(response_json: Any) -> tuple[Optional[int], Optional[str]]:
    if not isinstance(response_json, dict):
        return None, None
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None, None
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value, f"usage.{key}"
        if isinstance(value, float):
            return int(value), f"usage.{key}"
        if isinstance(value, str):
            try:
                return int(value), f"usage.{key}"
            except ValueError:
                pass
    return None, "usage"


def _make_request_body(
    model: str,
    provider: str = "",
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    temperature = 1 if provider.lower() == "kimi" else 0
    return {
        "model": model,
        "messages": messages if messages is not None else VALIDATION_SHORT_MESSAGES,
        "max_tokens": 1,
        "temperature": temperature,
        "stream": False,
    }


def _local_estimate(
    case: ProviderCase,
    tokenizer_dir: str,
    hf_local_files_only: bool,
    messages: List[Dict[str, str]],
) -> tuple[Optional[int], Optional[Dict[str, Any]], Optional[str]]:
    traces_dir = Path(__file__).resolve().parent / "traces"
    try:
        recorder = RequestRecorder(
            str(traces_dir),
            tokenizer_dir=tokenizer_dir,
            tokenizer_preset=case.tokenizer_preset,
            hf_local_files_only=hf_local_files_only,
            block_size=64,
            cache_idle_ttl_hours=0,
        )
        body = _make_request_body(case.model, case.provider, messages)
        record = recorder.create_request_record(
            session_id="tokenizer_validation",
            request_id=f"validation_{case.provider.lower()}",
            timestamp=datetime.now().isoformat(timespec="seconds"),
            request_body=body,
            request_body_bytes=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            conversation_mode="simple_streaming",
        )
        return int(record.get("local_input_tokens", 0)), record, None
    except Exception as exc:
        return None, None, str(exc)


def _call_provider(
    case: ProviderCase,
    timeout_seconds: float,
    messages: List[Dict[str, str]],
) -> tuple[Optional[int], Optional[str], str, Optional[str]]:
    if not case.api_key:
        return None, None, "skipped", "missing_api_key"

    try:
        response = httpx.post(
            _chat_url(case.base_url),
            headers={
                "Authorization": f"Bearer {case.api_key}",
                "Content-Type": "application/json",
            },
            json=_make_request_body(case.model, case.provider, messages),
            timeout=timeout_seconds,
        )
        response_body = response.content
        try:
            response_json = response.json()
        except Exception:
            response_json = {}

        if response.status_code >= 400:
            message = None
            if isinstance(response_json, dict):
                error = response_json.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or error.get("code") or response.status_code)
            return None, None, "api_error", message or f"http_{response.status_code}"

        prompt_tokens, usage_source = _extract_prompt_tokens(response_json)
        usage_metrics = read_actual_usage(
            response_json=response_json,
            estimated_cached_tokens=0,
            local_input_tokens=prompt_tokens or 0,
            response_body=response_body,
            response_content_type=response.headers.get("content-type"),
        )
        usage_source = usage_source or usage_metrics.get("usage_source")
        return prompt_tokens, usage_source, "ok", None
    except Exception as exc:
        return None, None, "api_error", str(exc)


def _diff_fields(local_tokens: Optional[int], api_tokens: Optional[int]) -> tuple[Optional[int], Optional[float]]:
    if local_tokens is None or api_tokens is None:
        return None, None
    diff = local_tokens - api_tokens
    pct = (diff / api_tokens * 100.0) if api_tokens > 0 else None
    return diff, pct


def _safe_note(note: Optional[str]) -> str:
    if not note:
        return ""
    # Keep docs useful without leaking bearer-like secrets from exception text.
    return re.sub(r"(sk-[A-Za-z0-9_\-]{8})[A-Za-z0-9_\-]+", r"\1...", note)


def _format_markdown(
    rows: List[Dict[str, Any]],
    sample: str,
    messages: List[Dict[str, str]],
) -> str:
    lines = [
        "# Tokenizer Validation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Sample: {sample}",
        "",
        "Prompt:",
        "",
        "```json",
        json.dumps(messages, ensure_ascii=False, indent=2),
        "```",
        "",
        "| Provider | Model | Preset | Effective preset | Local prompt tokens | API prompt tokens | Diff | Diff % | Usage source | Status | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        diff_pct = row.get("diff_pct")
        lines.append(
            "| {provider} | {model} | {preset} | {effective} | {local} | {api} | {diff} | {pct} | {usage} | {status} | {notes} |".format(
                provider=row["provider"],
                model=row["model"],
                preset=row["preset"],
                effective=row.get("effective_preset") or "",
                local=row["local_prompt_tokens"] if row["local_prompt_tokens"] is not None else "",
                api=row["api_prompt_tokens"] if row["api_prompt_tokens"] is not None else "",
                diff=row["diff_tokens"] if row["diff_tokens"] is not None else "",
                pct=f"{diff_pct:.1f}%" if isinstance(diff_pct, float) else "",
                usage=row.get("usage_source") or "",
                status=row["status"],
                notes=_safe_note(row.get("notes")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Local prompt tokens are computed without calling a tokenization API.",
            "- Kimi validation uses temperature=1 because the tested API rejected temperature=0; prompt tokens are unaffected by this sampling parameter.",
            "- VolcanoEngine/Doubao uses the DeepSeek tokenizer fallback by design, so some drift is expected.",
            "- API keys and bearer tokens are intentionally omitted from this document.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local tokenizer estimates against provider usage")
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--sample",
        choices=("short", "long-1000"),
        default="short",
        help="Validation prompt sample to send.",
    )
    parser.add_argument(
        "--hf-local-files-only",
        type=lambda value: str(value).strip().lower() not in {"0", "false", "no", "off"},
        default=True,
        help="Use local HF tokenizer files only (default: true).",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Provider request timeout seconds.")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Only compute local estimates; do not call providers.",
    )
    args = parser.parse_args()

    messages = _validation_messages(args.sample)
    if args.output is None:
        output_name = (
            "TOKENIZER_VALIDATION.md"
            if args.sample == "short"
            else "TOKENIZER_VALIDATION_1000TOKENS.md"
        )
        args.output = str(Path(__file__).resolve().parent / output_name)

    config_text = _read_config_text()
    base_dir = Path(__file__).resolve().parent
    rows: List[Dict[str, Any]] = []
    for case in _provider_cases(config_text):
        preset = normalize_tokenizer_preset(case.tokenizer_preset)
        tokenizer_dir = str(default_tokenizer_dir_for_preset(preset, base_dir))
        local_tokens, record, local_error = _local_estimate(
            case=case,
            tokenizer_dir=tokenizer_dir,
            hf_local_files_only=args.hf_local_files_only,
            messages=messages,
        )
        effective = record.get("tokenizer_effective_preset") if record else ""
        warning = record.get("tokenizer_warning") if record else None
        api_tokens: Optional[int] = None
        usage_source: Optional[str] = None
        status = "local_error" if local_error else "ok"
        note = local_error or warning
        if not local_error and not args.skip_api:
            api_tokens, usage_source, status, api_note = _call_provider(
                case,
                args.timeout,
                messages,
            )
            if api_note:
                note = "; ".join([x for x in [note, api_note] if x])
            if case.provider.lower() == "kimi":
                note = "; ".join(
                    [x for x in [note, "api_temperature=1_required_by_model"] if x]
                )
        elif args.skip_api and not local_error:
            status = "local_only"
            note = "; ".join([x for x in [note, "api_skipped"] if x])

        diff, diff_pct = _diff_fields(local_tokens, api_tokens)
        rows.append(
            {
                "provider": case.provider,
                "model": case.model,
                "preset": preset,
                "effective_preset": effective,
                "local_prompt_tokens": local_tokens,
                "api_prompt_tokens": api_tokens,
                "diff_tokens": diff,
                "diff_pct": diff_pct,
                "usage_source": usage_source,
                "status": status,
                "notes": note,
            }
        )

    output_path = Path(args.output)
    output_path.write_text(_format_markdown(rows, args.sample, messages), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
