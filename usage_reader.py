from typing import Any, Dict, Optional


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
) -> Dict[str, Any]:
    if not isinstance(response_json, dict):
        return {
            "actual_input_tokens": None,
            "actual_cached_tokens": None,
            "actual_cache_hit_rate": None,
            "difference_tokens": None,
            "status": "actual_cache_unknown",
        }

    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return {
            "actual_input_tokens": None,
            "actual_cached_tokens": None,
            "actual_cache_hit_rate": None,
            "difference_tokens": None,
            "status": "actual_cache_unknown",
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
    }
