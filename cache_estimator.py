from statistics import median
from typing import Any, Dict, List


def _starts_with(full_tokens: List[int], prefix_tokens: List[int]) -> bool:
    plen = len(prefix_tokens)
    if plen == 0 or plen > len(full_tokens):
        return False
    return full_tokens[:plen] == prefix_tokens


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _snap_down(token_count: int, block_size: int) -> int:
    if block_size <= 1:
        return max(token_count, 0)
    if token_count < block_size:
        return 0
    return token_count - (token_count % block_size)


def estimate_cache_hit(
    current_request: Dict[str, Any],
    history_requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_tokens = current_request.get("token_ids", [])
    current_model = current_request.get("model")
    current_source = str(current_request.get("_cache_unit_source", "deepseek_prompt_encoding"))
    block_size = current_request.get("cache_block_size", 64)
    try:
        block_size_i = max(int(block_size), 1)
    except Exception:
        block_size_i = 64
    current_input_tokens = len(current_tokens) if isinstance(current_tokens, list) else 0
    current_estimation_input_tokens = current_request.get("cache_estimation_input_tokens")
    if isinstance(current_estimation_input_tokens, int) and current_estimation_input_tokens > 0:
        denominator_tokens = current_estimation_input_tokens
    else:
        denominator_tokens = current_input_tokens

    if not isinstance(current_tokens, list):
        current_tokens = []

    best_cached_tokens = 0
    matched_request_id = None
    openclaw_actual_cached_history: List[int] = []
    openclaw_global_floor_tokens = current_request.get("_openclaw_global_floor_tokens")
    if not isinstance(openclaw_global_floor_tokens, int) or openclaw_global_floor_tokens < 0:
        openclaw_global_floor_tokens = 0

    # Conservative policy: only reuse cache candidates from the same model.
    for history_item in history_requests:
        if history_item.get("model") != current_model:
            continue

        history_actual_cached = history_item.get("_actual_cached_tokens")
        if isinstance(history_actual_cached, int) and history_actual_cached > 0:
            openclaw_actual_cached_history.append(history_actual_cached)

        history_source = str(history_item.get("_cache_unit_source", "deepseek_prompt_encoding"))

        if current_source == "raw_request_body" and history_source == "raw_request_body":
            history_tokens = history_item.get("token_ids", [])
            if not isinstance(history_tokens, list):
                continue
            shared_len = _common_prefix_len(current_tokens, history_tokens)
            candidate = _snap_down(shared_len, block_size_i)
            if candidate > best_cached_tokens:
                best_cached_tokens = candidate
                matched_request_id = history_item.get("request_id")
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

    hit_rate = (best_cached_tokens / denominator_tokens) if denominator_tokens > 0 else 0.0
    if best_cached_tokens == 0:
        matched_request_id = None

    if current_source == "raw_request_body":
        # OpenClaw mode patch: reuse a multi-turn session cache floor derived from
        # prior actual cached tokens. This avoids severe underestimation when
        # raw-body exact prefix drifts but upstream cache remains warm.
        session_floor = 0
        if openclaw_actual_cached_history and denominator_tokens > 0:
            floor_candidate = int(median(openclaw_actual_cached_history))
            floor_candidate = _snap_down(floor_candidate, block_size_i)
            if floor_candidate > denominator_tokens:
                floor_candidate = denominator_tokens
            session_floor = max(floor_candidate, 0)

        if session_floor > best_cached_tokens:
            best_cached_tokens = session_floor
            matched_request_id = "__openclaw_session_floor__"

        global_floor = _snap_down(openclaw_global_floor_tokens, block_size_i)
        if global_floor > denominator_tokens:
            global_floor = denominator_tokens
        if global_floor > best_cached_tokens:
            best_cached_tokens = global_floor
            matched_request_id = "__openclaw_global_floor__"

        hit_rate = (best_cached_tokens / denominator_tokens) if denominator_tokens > 0 else 0.0
        match_strategy = "raw_body_lcp_block_aligned"
    else:
        match_strategy = "deepseek_boundary_prefix_match"

    return {
        "estimated_cached_tokens": best_cached_tokens,
        "estimated_cache_hit_rate": hit_rate,
        "matched_request_id": matched_request_id,
        "match_strategy": match_strategy,
        "estimation_denominator_tokens": denominator_tokens,
        "openclaw_session_cache_floor_tokens": (
            best_cached_tokens if matched_request_id == "__openclaw_session_floor__" else 0
        ),
        "openclaw_global_cache_floor_tokens": (
            best_cached_tokens if matched_request_id == "__openclaw_global_floor__" else 0
        ),
    }
