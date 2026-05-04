import json
from typing import Any, Dict, Optional, Tuple


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


def _pick_first_int(candidates: list[Any]) -> Optional[int]:
    for item in candidates:
        parsed = _as_int(item)
        if parsed is not None:
            return parsed
    return None


def _extract_cached_tokens(usage: Dict[str, Any]) -> Optional[int]:
    direct_candidates = [
        usage.get("cached_tokens"),
        usage.get("prompt_cache_hit_tokens"),
        usage.get("cache_hit_tokens"),
        usage.get("input_cached_tokens"),
    ]
    cached_tokens = _pick_first_int(direct_candidates)
    if cached_tokens is not None:
        return cached_tokens

    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return _as_int(details.get("cached_tokens"))
    return None


def _extract_input_tokens(usage: Dict[str, Any]) -> Optional[int]:
    return _pick_first_int([usage.get("prompt_tokens"), usage.get("input_tokens")])


def _looks_like_sse_stream(
    response_content_type: Optional[str],
    response_body: Any,
) -> bool:
    ct = (response_content_type or "").lower()
    if "text/event-stream" in ct:
        return True

    if isinstance(response_body, (bytes, bytearray)):
        return bytes(response_body[:64]).lstrip().startswith(b"data:")
    if isinstance(response_body, str):
        return response_body.lstrip().startswith("data:")
    return False


def _iter_sse_data_payloads(text: str) -> list[str]:
    payloads: list[str] = []
    current_data_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            if current_data_lines:
                payloads.append("\n".join(current_data_lines))
                current_data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            current_data_lines.append(line[5:].lstrip())

    if current_data_lines:
        payloads.append("\n".join(current_data_lines))
    return payloads


def _extract_usage_from_sse_body(response_body: Any) -> Optional[Dict[str, Any]]:
    if isinstance(response_body, (bytes, bytearray)):
        text = bytes(response_body).decode("utf-8", errors="ignore")
    elif isinstance(response_body, str):
        text = response_body
    else:
        return None

    latest_usage: Optional[Dict[str, Any]] = None
    for payload in _iter_sse_data_payloads(text):
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        usage = event.get("usage")
        if isinstance(usage, dict):
            latest_usage = usage
    return latest_usage


def _extract_usage_object(
    response_json: Any,
    response_body: Any = None,
    response_content_type: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if isinstance(response_json, dict):
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            return usage, "json_usage"

    if not _looks_like_sse_stream(response_content_type, response_body):
        return None, None

    usage_from_sse = _extract_usage_from_sse_body(response_body)
    if isinstance(usage_from_sse, dict):
        return usage_from_sse, "sse_usage"
    return None, None


def _judge_estimation_status(
    difference_tokens: int,
    denominator_tokens: int,
) -> str:
    # Lower-bound policy:
    # - actual >= estimated is considered normal (conservative estimate).
    # - only clearly optimistic estimates are flagged as overestimated.
    tolerance = max(64, int(denominator_tokens * 0.1))
    if difference_tokens >= 0:
        return "normal"
    if abs(difference_tokens) <= tolerance:
        return "normal"
    if difference_tokens < 0:
        return "overestimated"
    return "normal"


def read_actual_usage(
    response_json: Any,
    estimated_cached_tokens: int,
    local_input_tokens: int,
    response_body: Any = None,
    response_content_type: Optional[str] = None,
) -> Dict[str, Any]:
    usage, usage_source = _extract_usage_object(
        response_json=response_json,
        response_body=response_body,
        response_content_type=response_content_type,
    )
    if not isinstance(usage, dict):
        return {
            "actual_input_tokens": None,
            "actual_cached_tokens": None,
            "actual_cache_hit_rate": None,
            "difference_tokens": None,
            "status": "actual_cache_unknown",
            "usage_source": usage_source,
        }

    actual_input_tokens = _extract_input_tokens(usage)
    actual_cached_tokens = _extract_cached_tokens(usage)
    if actual_cached_tokens is None:
        return {
            "actual_input_tokens": actual_input_tokens,
            "actual_cached_tokens": None,
            "actual_cache_hit_rate": None,
            "difference_tokens": None,
            "status": "actual_cache_unknown",
            "usage_source": usage_source,
        }

    denominator = (
        actual_input_tokens
        if actual_input_tokens is not None and actual_input_tokens > 0
        else local_input_tokens
    )

    if denominator > 0:
        actual_cache_hit_rate = actual_cached_tokens / denominator
    else:
        actual_cache_hit_rate = None

    difference_tokens = actual_cached_tokens - estimated_cached_tokens
    if denominator > 0:
        status = _judge_estimation_status(difference_tokens, denominator)
    else:
        status = "normal"

    return {
        "actual_input_tokens": actual_input_tokens,
        "actual_cached_tokens": actual_cached_tokens,
        "actual_cache_hit_rate": actual_cache_hit_rate,
        "difference_tokens": difference_tokens,
        "status": status,
        "usage_source": usage_source,
    }
