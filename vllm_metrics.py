import math
import re
from typing import Any, Dict, Optional

try:
    from usage_reader import CACHE_ESTIMATION_DIFF_THRESHOLD_TOKENS
except Exception:  # pragma: no cover - package execution path
    from .usage_reader import CACHE_ESTIMATION_DIFF_THRESHOLD_TOKENS  # type: ignore


VLLM_PROBE_BLOCK_SIZE = 784

_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+[-+]?\d+(?:\.\d+)?)?$"
)


def _parse_labels(raw_labels: Optional[str]) -> Dict[str, str]:
    if not raw_labels:
        return {}

    labels: Dict[str, str] = {}
    pos = 0
    length = len(raw_labels)
    while pos < length:
        while pos < length and raw_labels[pos] in " ,":
            pos += 1
        key_start = pos
        while pos < length and raw_labels[pos] != "=":
            pos += 1
        if pos >= length:
            break
        key = raw_labels[key_start:pos].strip()
        pos += 1
        if pos >= length or raw_labels[pos] != '"':
            break
        pos += 1
        value_chars: list[str] = []
        escaped = False
        while pos < length:
            char = raw_labels[pos]
            pos += 1
            if escaped:
                value_chars.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                break
            value_chars.append(char)
        if key:
            labels[key] = "".join(value_chars)
        while pos < length and raw_labels[pos] != ",":
            pos += 1
        if pos < length and raw_labels[pos] == ",":
            pos += 1
    return labels


def parse_vllm_metrics_text(metrics_text: str) -> Dict[str, float]:
    counters = {
        "vllm_prompt_tokens_cached": 0.0,
        "vllm_prefix_cache_hits": 0.0,
        "vllm_prefix_cache_queries": 0.0,
        "vllm_prompt_tokens_local_cache_hit": 0.0,
        "vllm_prompt_tokens_local_compute": 0.0,
    }
    seen = {key: False for key in counters}

    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue

        if name == "vllm:prompt_tokens_cached_total":
            counters["vllm_prompt_tokens_cached"] += value
            seen["vllm_prompt_tokens_cached"] = True
        elif name == "vllm:prefix_cache_hits_total":
            counters["vllm_prefix_cache_hits"] += value
            seen["vllm_prefix_cache_hits"] = True
        elif name == "vllm:prefix_cache_queries_total":
            counters["vllm_prefix_cache_queries"] += value
            seen["vllm_prefix_cache_queries"] = True
        elif name == "vllm:prompt_tokens_by_source_total":
            source = _parse_labels(raw_labels).get("source")
            if source == "local_cache_hit":
                counters["vllm_prompt_tokens_local_cache_hit"] += value
                seen["vllm_prompt_tokens_local_cache_hit"] = True
            elif source == "local_compute":
                counters["vllm_prompt_tokens_local_compute"] += value
                seen["vllm_prompt_tokens_local_compute"] = True

    return {
        key: value
        for key, value in counters.items()
        if seen[key]
    }


def compute_vllm_metrics_delta(
    before: Optional[Dict[str, float]],
    after: Optional[Dict[str, float]],
    existing_error: Optional[str] = None,
) -> Dict[str, Any]:
    fields = {
        "vllm_prompt_tokens_cached_delta": None,
        "vllm_prefix_cache_hits_delta": None,
        "vllm_prefix_cache_queries_delta": None,
        "vllm_prompt_tokens_local_cache_hit_delta": None,
        "vllm_prompt_tokens_local_compute_delta": None,
        "vllm_metrics_error": existing_error,
    }
    if existing_error:
        return fields
    if before is None or after is None:
        fields["vllm_metrics_error"] = "metrics_snapshot_missing"
        return fields

    mapping = {
        "vllm_prompt_tokens_cached_delta": "vllm_prompt_tokens_cached",
        "vllm_prefix_cache_hits_delta": "vllm_prefix_cache_hits",
        "vllm_prefix_cache_queries_delta": "vllm_prefix_cache_queries",
        "vllm_prompt_tokens_local_cache_hit_delta": "vllm_prompt_tokens_local_cache_hit",
        "vllm_prompt_tokens_local_compute_delta": "vllm_prompt_tokens_local_compute",
    }
    missing: list[str] = []
    negative: list[str] = []

    for field, key in mapping.items():
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            missing.append(key)
            continue
        delta = after_value - before_value
        if delta < 0:
            negative.append(key)
            continue
        fields[field] = int(round(delta))

    if missing:
        fields["vllm_metrics_error"] = "missing_metrics:" + ",".join(sorted(missing))
    elif negative:
        fields["vllm_metrics_error"] = "counter_reset_or_negative_delta:" + ",".join(
            sorted(negative)
        )
    return fields


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def extract_input_tokens_from_response(response_json: Any) -> Optional[int]:
    if not isinstance(response_json, dict):
        return None
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    if prompt_tokens is not None:
        return prompt_tokens
    return _as_int(usage.get("input_tokens"))


def read_actual_usage_from_vllm_metrics(
    response_json: Any,
    estimated_cached_tokens: int,
    local_input_tokens: int,
    metrics_delta: Dict[str, Any],
) -> Dict[str, Any]:
    actual_input_tokens = extract_input_tokens_from_response(response_json)
    denominator = (
        actual_input_tokens
        if actual_input_tokens is not None and actual_input_tokens > 0
        else local_input_tokens
    )

    actual_cached_tokens = metrics_delta.get("vllm_prompt_tokens_cached_delta")
    if not isinstance(actual_cached_tokens, int):
        return {
            "actual_input_tokens": actual_input_tokens,
            "actual_cached_tokens": None,
            "actual_cache_hit_rate": None,
            "actual_uncached_tokens": None,
            "cache_estimation_diff_threshold_tokens": CACHE_ESTIMATION_DIFF_THRESHOLD_TOKENS,
            "difference_tokens": None,
            "status": "actual_cache_unknown",
            "usage_source": "vllm_metrics_delta",
        }

    actual_cache_hit_rate = actual_cached_tokens / denominator if denominator > 0 else None
    actual_uncached_tokens = max(denominator - actual_cached_tokens, 0) if denominator > 0 else None
    difference_tokens = actual_cached_tokens - estimated_cached_tokens
    status = (
        "overestimated"
        if difference_tokens < -CACHE_ESTIMATION_DIFF_THRESHOLD_TOKENS
        else "normal"
    )

    return {
        "actual_input_tokens": actual_input_tokens,
        "actual_cached_tokens": actual_cached_tokens,
        "actual_cache_hit_rate": actual_cache_hit_rate,
        "actual_uncached_tokens": actual_uncached_tokens,
        "cache_estimation_diff_threshold_tokens": CACHE_ESTIMATION_DIFF_THRESHOLD_TOKENS,
        "difference_tokens": difference_tokens,
        "status": status,
        "usage_source": "vllm_metrics_delta",
    }
