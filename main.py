import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from cache_estimator import estimate_cache_hit
from request_recorder import RequestRecorder
from usage_reader import read_actual_usage


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local prompt-cache hit proxy")
    parser.add_argument("--port", type=int, default=8787, help="Local proxy port")
    parser.add_argument(
        "--target-base-url",
        type=str,
        required=True,
        help="Target model API base URL, e.g. https://api.deepseek.com",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "deepseek_tokenizer"),
        help="Local DeepSeek tokenizer directory",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Cache storage unit size for estimation (default: 64)",
    )
    parser.add_argument(
        "--cache-idle-ttl-hours",
        type=float,
        default=24.0,
        help="Evict history entries if not used for this many hours (default: 24)",
    )
    parser.add_argument(
        "--max-history-requests",
        type=int,
        default=2000,
        help="Hard cap for in-memory history size (default: 2000)",
    )
    parser.add_argument(
        "--conversation-mode",
        type=str,
        default="simple_streaming",
        choices=["simple_streaming", "openclaw_agent"],
        help=(
            "Conversation mode preset. "
            "simple_streaming matches legacy direct chat behavior (v1 defaults); "
            "openclaw_agent matches OpenClaw agent behavior (v2 defaults)."
        ),
    )
    parser.add_argument(
        "--raw-request-capture",
        type=str,
        default="none",
        choices=["none", "utf8", "base64"],
        help="Optional raw request capture mode in traces (default: none)",
    )
    parser.add_argument(
        "--raw-request-max-chars",
        type=int,
        default=12000,
        help="Max characters for utf8 raw request capture (default: 12000)",
    )
    args = parser.parse_args()

    if args.conversation_mode == "simple_streaming":
        args.input_token_source = "deepseek_prompt_encoding"
    else:
        args.input_token_source = "openclaw_raw_body"

    return args


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_session_id() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def build_target_chat_url(target_base_url: str) -> str:
    base = target_base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def filter_request_headers(headers: Dict[str, str]) -> Dict[str, str]:
    forwarded: Dict[str, str] = {}
    for key, value in headers.items():
        k = key.lower()
        if k in HOP_BY_HOP_HEADERS or k in {"host", "content-length"}:
            continue
        forwarded[key] = value
    return forwarded


def filter_response_headers(headers: Dict[str, str]) -> Dict[str, str]:
    filtered: Dict[str, str] = {}
    for key, value in headers.items():
        k = key.lower()
        if k in HOP_BY_HOP_HEADERS or k == "content-length":
            continue
        filtered[key] = value
    return filtered


def print_summary(
    session_id: str,
    request_id: str,
    request_record: Dict[str, Any],
    estimate: Dict[str, Any],
    usage_metrics: Dict[str, Any],
    conversation_mode: str,
    input_token_source: str,
    predicted_input_tokens: int,
) -> None:
    print(f"[Session {session_id}] {request_id}")
    print(f"Model: {request_record.get('model')}")
    print(f"Conversation mode: {conversation_mode}")
    print(f"Local input tokens: {request_record.get('local_input_tokens')}")
    print(f"Input token source: {input_token_source}")
    print(f"Cache unit source: {request_record.get('_cache_unit_source')}")
    fallback_reason = request_record.get("_cache_unit_fallback_reason")
    if fallback_reason:
        print(f"Cache unit fallback: {fallback_reason}")
    print(f"Predicted input tokens: {predicted_input_tokens}")
    actual_input_tokens = usage_metrics.get("actual_input_tokens")
    if actual_input_tokens is None:
        print("Actual input tokens: unknown")
    else:
        print(f"Actual input tokens: {actual_input_tokens}")
    print()
    print(f"Estimated cached tokens: {estimate.get('estimated_cached_tokens')}")
    print(f"Estimation denominator tokens: {estimate.get('estimation_denominator_tokens')}")
    openclaw_floor = estimate.get("openclaw_session_cache_floor_tokens")
    if isinstance(openclaw_floor, int) and openclaw_floor > 0:
        print(f"OpenClaw session cache floor: {openclaw_floor}")
    openclaw_global_floor = estimate.get("openclaw_global_cache_floor_tokens")
    if isinstance(openclaw_global_floor, int) and openclaw_global_floor > 0:
        print(f"OpenClaw global cache floor: {openclaw_global_floor}")
    est_rate = estimate.get("estimated_cache_hit_rate", 0.0) * 100
    print(f"Estimated cache hit rate: {est_rate:.1f}%")
    print(f"Matched with: {estimate.get('matched_request_id')}")
    print(f"Match strategy: {estimate.get('match_strategy')}")
    print()

    actual_cached_tokens = usage_metrics.get("actual_cached_tokens")
    if actual_cached_tokens is None:
        print("Actual cached tokens: unknown")
        usage_source = usage_metrics.get("usage_source")
        if usage_source:
            print(f"Usage source: {usage_source}")
        print(f"Status: {usage_metrics.get('status')}")
        print()
        return

    print(f"Actual cached tokens: {actual_cached_tokens}")
    actual_rate = usage_metrics.get("actual_cache_hit_rate")
    if actual_rate is None:
        print("Actual cache hit rate: unknown")
    else:
        print(f"Actual cache hit rate: {actual_rate * 100:.1f}%")
    usage_source = usage_metrics.get("usage_source")
    if usage_source:
        print(f"Usage source: {usage_source}")
    print(f"Difference: {usage_metrics.get('difference_tokens')}")
    print(f"Status: {usage_metrics.get('status')}")
    print()


def build_log_payload(
    request_record: Dict[str, Any],
    estimate: Dict[str, Any],
    usage_metrics: Dict[str, Any],
    conversation_mode: str,
    input_token_source: str,
    predicted_input_tokens: int,
    raw_request_capture_mode: str,
    raw_request_utf8: str | None,
    raw_request_base64: str | None,
    raw_request_truncated: bool,
    status_override: str | None = None,
) -> Dict[str, Any]:
    actual_input_tokens = usage_metrics.get("actual_input_tokens")
    input_tokens_difference = None
    if isinstance(actual_input_tokens, int):
        input_tokens_difference = actual_input_tokens - predicted_input_tokens

    payload = {
        "conversation_mode": conversation_mode,
        "session_id": request_record.get("session_id"),
        "request_id": request_record.get("request_id"),
        "timestamp": request_record.get("timestamp"),
        "model": request_record.get("model"),
        "messages": request_record.get("messages"),
        "canonical_text": request_record.get("canonical_text"),
        "local_input_tokens": request_record.get("local_input_tokens"),
        "input_token_source": input_token_source,
        "predicted_input_tokens": predicted_input_tokens,
        "input_tokens_difference": input_tokens_difference,
        "raw_request_body_sha256": request_record.get("raw_request_body_sha256"),
        "raw_request_body_size_bytes": request_record.get("raw_request_body_size_bytes"),
        "raw_request_body_tokenizer_tokens": request_record.get("raw_request_body_tokenizer_tokens"),
        "cache_unit_source": request_record.get("_cache_unit_source"),
        "cache_unit_fallback_reason": request_record.get("_cache_unit_fallback_reason"),
        "raw_request_capture_mode": raw_request_capture_mode,
        "raw_request_body_utf8": raw_request_utf8,
        "raw_request_body_base64": raw_request_base64,
        "raw_request_body_truncated": raw_request_truncated,
        "estimated_cached_tokens": estimate.get("estimated_cached_tokens"),
        "estimated_cache_hit_rate": estimate.get("estimated_cache_hit_rate"),
        "estimation_denominator_tokens": estimate.get("estimation_denominator_tokens"),
        "openclaw_session_cache_floor_tokens": estimate.get("openclaw_session_cache_floor_tokens"),
        "openclaw_global_cache_floor_tokens": estimate.get("openclaw_global_cache_floor_tokens"),
        "matched_request_id": estimate.get("matched_request_id"),
        "match_strategy": estimate.get("match_strategy"),
        "actual_input_tokens": usage_metrics.get("actual_input_tokens"),
        "actual_cached_tokens": usage_metrics.get("actual_cached_tokens"),
        "actual_cache_hit_rate": usage_metrics.get("actual_cache_hit_rate"),
        "usage_source": usage_metrics.get("usage_source"),
        "difference_tokens": usage_metrics.get("difference_tokens"),
        "status": status_override or usage_metrics.get("status"),
    }
    return payload


def create_app(
    target_base_url: str,
    session_id: str,
    tokenizer_dir: str,
    block_size: int,
    cache_idle_ttl_hours: float,
    max_history_requests: int,
    conversation_mode: str,
    input_token_source: str,
    raw_request_capture: str,
    raw_request_max_chars: int,
) -> FastAPI:
    app = FastAPI(title="Cache Hit Proxy", version="0.2.0")
    traces_dir = Path(__file__).resolve().parent / "traces"
    recorder = RequestRecorder(
        str(traces_dir),
        tokenizer_dir=tokenizer_dir,
        block_size=block_size,
        cache_idle_ttl_hours=cache_idle_ttl_hours,
        max_history_requests=max_history_requests,
    )
    chat_url = build_target_chat_url(target_base_url)

    app.state.request_counter = 0
    app.state.session_id = session_id

    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(request: Request) -> Response:
        body_bytes = await request.body()
        raw_request_utf8: str | None = None
        raw_request_base64: str | None = None
        raw_request_truncated = False
        if raw_request_capture == "utf8":
            decoded = body_bytes.decode("utf-8", errors="replace")
            if raw_request_max_chars > 0 and len(decoded) > raw_request_max_chars:
                raw_request_utf8 = decoded[:raw_request_max_chars]
                raw_request_truncated = True
            else:
                raw_request_utf8 = decoded
        elif raw_request_capture == "base64":
            import base64

            raw_request_base64 = base64.b64encode(body_bytes).decode("ascii")

        request_json: Any
        try:
            request_json = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            request_json = {}

        app.state.request_counter += 1
        request_id = f"req_{app.state.request_counter:04d}"
        timestamp = now_iso()

        try:
            request_record = recorder.create_request_record(
                session_id=session_id,
                request_id=request_id,
                timestamp=timestamp,
                request_body=request_json,
                request_body_bytes=body_bytes,
            )
            request_record = recorder.apply_input_token_source(
                request_record=request_record,
                input_token_source=input_token_source,
            )
        except Exception:
            request_record = {
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": timestamp,
                "model": None,
                "messages": [],
                "canonical_text": "",
                "token_ids": [],
                "local_input_tokens": 0,
                "raw_request_body_sha256": None,
                "raw_request_body_size_bytes": 0,
                "raw_request_body_tokenizer_tokens": None,
                "raw_request_body_token_ids": [],
                "persisted_prefix_units_tokens": [],
                "cache_block_size": block_size,
                "cache_estimation_input_tokens": 0,
                "_cache_unit_source": "deepseek_prompt_encoding",
                "_cache_unit_fallback_reason": "create_request_record_failed",
            }

        predicted_input_tokens = request_record.get("local_input_tokens", 0)
        if input_token_source == "openclaw_raw_body":
            raw_count = request_record.get("raw_request_body_tokenizer_tokens")
            if isinstance(raw_count, int) and raw_count > 0:
                predicted_input_tokens = raw_count

        try:
            # Estimate against current in-memory history before forwarding upstream.
            history = recorder.history_snapshot()
            estimate = estimate_cache_hit(request_record, history)
            recorder.touch_request(estimate.get("matched_request_id"))
        except Exception:
            estimate = {
                "estimated_cached_tokens": 0,
                "estimated_cache_hit_rate": 0.0,
                "matched_request_id": None,
                "match_strategy": "deepseek_boundary_prefix_match",
                "estimation_denominator_tokens": predicted_input_tokens,
                "openclaw_session_cache_floor_tokens": 0,
                "openclaw_global_cache_floor_tokens": 0,
            }

        forwarded_headers = filter_request_headers(dict(request.headers))
        query = request.url.query
        target_url = f"{chat_url}?{query}" if query else chat_url

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                upstream = await client.post(
                    target_url,
                    content=body_bytes,
                    headers=forwarded_headers,
                )
        except Exception as exc:
            # Upstream failures are surfaced to client and also logged as a trace entry.
            usage_metrics = {
                "actual_input_tokens": None,
                "actual_cached_tokens": None,
                "actual_cache_hit_rate": None,
                "difference_tokens": None,
                "status": "upstream_error",
            }
            log_payload = build_log_payload(
                request_record=request_record,
                estimate=estimate,
                usage_metrics=usage_metrics,
                conversation_mode=conversation_mode,
                input_token_source=input_token_source,
                predicted_input_tokens=predicted_input_tokens,
                raw_request_capture_mode=raw_request_capture,
                raw_request_utf8=raw_request_utf8,
                raw_request_base64=raw_request_base64,
                raw_request_truncated=raw_request_truncated,
                status_override="upstream_error",
            )
            try:
                recorder.append_jsonl(session_id, log_payload)
                recorder.append_history(request_record)
                print_summary(
                    session_id,
                    request_id,
                    request_record,
                    estimate,
                    usage_metrics,
                    conversation_mode=conversation_mode,
                    input_token_source=input_token_source,
                    predicted_input_tokens=predicted_input_tokens,
                )
            except Exception:
                pass
            return JSONResponse(
                status_code=502,
                content={"error": f"upstream request failed: {exc}"},
            )

        response_body = upstream.content
        response_headers = filter_response_headers(dict(upstream.headers))

        try:
            response_json = json.loads(response_body) if response_body else {}
        except Exception:
            response_json = {}

        try:
            usage_metrics = read_actual_usage(
                response_json=response_json,
                estimated_cached_tokens=estimate["estimated_cached_tokens"],
                local_input_tokens=predicted_input_tokens,
                response_body=response_body,
                response_content_type=upstream.headers.get("content-type"),
            )
        except Exception:
            usage_metrics = {
                "actual_input_tokens": None,
                "actual_cached_tokens": None,
                "actual_cache_hit_rate": None,
                "difference_tokens": None,
                "status": "actual_cache_unknown",
            }

        actual_cached_for_history = usage_metrics.get("actual_cached_tokens")
        if isinstance(actual_cached_for_history, int) and actual_cached_for_history >= 0:
            request_record["_actual_cached_tokens"] = actual_cached_for_history
        else:
            request_record["_actual_cached_tokens"] = None

        log_payload = build_log_payload(
            request_record=request_record,
            estimate=estimate,
            usage_metrics=usage_metrics,
            conversation_mode=conversation_mode,
            input_token_source=input_token_source,
            predicted_input_tokens=predicted_input_tokens,
            raw_request_capture_mode=raw_request_capture,
            raw_request_utf8=raw_request_utf8,
            raw_request_base64=raw_request_base64,
            raw_request_truncated=raw_request_truncated,
        )

        try:
            # Attach output-boundary units so the next request can match a longer prefix.
            request_record = recorder.attach_response_cache_units(request_record, response_json)
            recorder.append_jsonl(session_id, log_payload)
            recorder.append_history(request_record)
            print_summary(
                session_id,
                request_id,
                request_record,
                estimate,
                usage_metrics,
                conversation_mode=conversation_mode,
                input_token_source=input_token_source,
                predicted_input_tokens=predicted_input_tokens,
            )
        except Exception:
            pass

        return Response(
            content=response_body,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return app


def main() -> None:
    args = parse_args()
    tokenizer_path = Path(args.tokenizer_dir).resolve()
    if not tokenizer_path.exists():
        raise SystemExit(
            f"Tokenizer directory not found: {tokenizer_path}. "
            "Please download DeepSeek tokenizer files first."
        )

    session_id = make_session_id()
    app = create_app(
        args.target_base_url,
        session_id,
        tokenizer_dir=str(tokenizer_path),
        block_size=args.block_size,
        cache_idle_ttl_hours=args.cache_idle_ttl_hours,
        max_history_requests=args.max_history_requests,
        conversation_mode=args.conversation_mode,
        input_token_source=args.input_token_source,
        raw_request_capture=args.raw_request_capture,
        raw_request_max_chars=max(args.raw_request_max_chars, 0),
    )
    print(f"Session ID: {session_id}")
    print(f"Target base URL: {args.target_base_url}")
    print(f"Tokenizer dir: {tokenizer_path}")
    print(f"Block size: {args.block_size}")
    print(f"Cache idle TTL (hours): {args.cache_idle_ttl_hours}")
    print(f"Max history requests: {args.max_history_requests}")
    print(f"Conversation mode: {args.conversation_mode}")
    print(f"Input token source: {args.input_token_source}")
    print(f"Raw request capture: {args.raw_request_capture}")
    print(f"Raw request max chars: {max(args.raw_request_max_chars, 0)}")
    print(f"Listening on: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
