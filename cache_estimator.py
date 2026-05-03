from typing import Any, Dict, List


def _starts_with(full_tokens: List[int], prefix_tokens: List[int]) -> bool:
    plen = len(prefix_tokens)
    if plen == 0 or plen > len(full_tokens):
        return False
    return full_tokens[:plen] == prefix_tokens


def estimate_cache_hit(
    current_request: Dict[str, Any],
    history_requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_tokens = current_request.get("token_ids", [])
    current_model = current_request.get("model")
    current_input_tokens = len(current_tokens) if isinstance(current_tokens, list) else 0

    if not isinstance(current_tokens, list):
        current_tokens = []

    best_cached_tokens = 0
    matched_request_id = None

    # Conservative policy: only reuse cache candidates from the same model.
    for history_item in history_requests:
        if history_item.get("model") != current_model:
            continue

        units = history_item.get("persisted_prefix_units_tokens", [])
        if not isinstance(units, list):
            continue

        # Each history item can expose multiple persisted boundary units.
        for unit in units:
            if not isinstance(unit, list):
                continue
            if not _starts_with(current_tokens, unit):
                continue
            unit_len = len(unit)
            if unit_len > best_cached_tokens:
                # Use the longest matched prefix unit as the estimated cached span.
                best_cached_tokens = unit_len
                matched_request_id = history_item.get("request_id")

    hit_rate = (best_cached_tokens / current_input_tokens) if current_input_tokens > 0 else 0.0
    if best_cached_tokens == 0:
        matched_request_id = None

    return {
        "estimated_cached_tokens": best_cached_tokens,
        "estimated_cache_hit_rate": hit_rate,
        "matched_request_id": matched_request_id,
        "match_strategy": "deepseek_boundary_prefix_match",
    }
