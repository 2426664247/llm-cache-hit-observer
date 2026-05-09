import copy
import hashlib
import json
import time
from pathlib import Path
from threading import Lock
from statistics import median
from typing import Any, Dict, List, Optional

try:
    from tokenizer_adapters import TokenizerAdapter, create_tokenizer_adapter
except Exception:  # pragma: no cover - package execution path
    from .tokenizer_adapters import TokenizerAdapter, create_tokenizer_adapter  # type: ignore


def _stable_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snap_down(token_count: int, block_size: int) -> int:
    # Align to provider-like cache granularity (default 64-token blocks).
    if block_size <= 1:
        return max(token_count, 0)
    if token_count < block_size:
        return 0
    return token_count - (token_count % block_size)


def canonicalize_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return _stable_to_text(messages)

    chunks: List[str] = []
    for item in messages:
        if not isinstance(item, dict):
            role = "unknown"
            content = _stable_to_text(item)
        else:
            role = str(item.get("role", "unknown"))
            content = _stable_to_text(item.get("content", ""))
        chunks.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(chunks)


def _extract_thinking_mode(request_body: Dict[str, Any]) -> str:
    thinking = request_body.get("thinking")
    if isinstance(thinking, dict):
        mode = str(thinking.get("type", "")).strip().lower()
        if mode == "enabled":
            return "thinking"

    model = str(request_body.get("model", "")).strip().lower()
    if "reasoner" in model:
        return "thinking"
    return "chat"


def _extract_reasoning_effort(request_body: Dict[str, Any]) -> Optional[str]:
    value = request_body.get("reasoning_effort")
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"high", "max"}:
        return normalized
    return None


def _extract_text_from_content_blocks(content: Any) -> Optional[str]:
    if not isinstance(content, list):
        return None

    text_parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).strip().lower() != "text":
            continue
        text_parts.append(str(block.get("text", "")))

    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def _normalize_messages_for_v4(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return [{"role": "user", "content": _stable_to_text(messages)}]

    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            normalized.append({"role": "user", "content": _stable_to_text(item)})
            continue

        msg = copy.deepcopy(item)
        msg["role"] = str(msg.get("role", "user"))

        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        elif not isinstance(content, str):
            msg["content"] = _stable_to_text(content)

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            fixed_tool_calls: List[Dict[str, Any]] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_copy = copy.deepcopy(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    arguments = fn.get("arguments")
                    if arguments is None:
                        fn["arguments"] = "{}"
                    elif not isinstance(arguments, str):
                        fn["arguments"] = _stable_to_text(arguments)
                    tc_copy["function"] = fn
                fixed_tool_calls.append(tc_copy)
            msg["tool_calls"] = fixed_tool_calls

        normalized.append(msg)

    return normalized


def _normalize_messages_for_piai_probe(
    messages: Any,
    top_level_tools: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        normalized = _normalize_messages_for_v4(messages)
    else:
        rewritten_messages: List[Any] = []
        for item in messages:
            if not isinstance(item, dict):
                rewritten_messages.append(item)
                continue

            item_copy = copy.deepcopy(item)
            extracted_text = _extract_text_from_content_blocks(item_copy.get("content"))
            if extracted_text is not None:
                item_copy["content"] = extracted_text
            rewritten_messages.append(item_copy)
        normalized = _normalize_messages_for_v4(rewritten_messages)

    if not isinstance(top_level_tools, list) or not top_level_tools:
        return normalized

    tools_copy = copy.deepcopy(top_level_tools)
    for msg in normalized:
        role = str(msg.get("role", "")).strip().lower()
        if role in {"system", "developer"}:
            msg["tools"] = tools_copy
            return normalized

    normalized.insert(
        0,
        {
            "role": "system",
            "content": "",
            "tools": tools_copy,
        },
    )
    return normalized


def _extract_assistant_message_from_response(response_json: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response_json, dict):
        return None

    choices = response_json.get("choices")
    if not isinstance(choices, list):
        return None

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        assistant_message: Dict[str, Any] = {"role": "assistant"}
        content = message.get("content")
        if content is None:
            assistant_message["content"] = ""
        elif isinstance(content, str):
            assistant_message["content"] = content
        else:
            assistant_message["content"] = _stable_to_text(content)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            assistant_message["tool_calls"] = tool_calls

        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            assistant_message["reasoning_content"] = reasoning_content

        return assistant_message
    return None


class RequestRecorder:
    def __init__(
        self,
        traces_dir: str,
        tokenizer_dir: str,
        tokenizer_preset: str = "deepseek-v4-pro",
        hf_local_files_only: bool = True,
        block_size: int = 64,
        cache_idle_ttl_hours: float = 24.0,
        max_history_requests: int = 0,
        tokenizer_adapter: Optional[TokenizerAdapter] = None,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = tokenizer_adapter or create_tokenizer_adapter(
            preset=tokenizer_preset,
            tokenizer_dir=tokenizer_dir,
            hf_local_files_only=hf_local_files_only,
        )
        self._history: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._block_size = max(int(block_size), 1)
        self._cache_idle_ttl_seconds = max(float(cache_idle_ttl_hours), 0.0) * 3600.0
        try:
            self._max_history_requests = int(max_history_requests)
        except Exception:
            self._max_history_requests = 0
        self._openclaw_global_floor_by_model = self._load_openclaw_global_floor_by_model()

    def _now_epoch(self) -> float:
        return time.time()

    def _prune_history_locked(self, now_epoch: Optional[float] = None) -> None:
        now = self._now_epoch() if now_epoch is None else float(now_epoch)

        # TTL-based forgetting: entries not reused within idle window are dropped.
        if self._cache_idle_ttl_seconds > 0:
            kept: List[Dict[str, Any]] = []
            for item in self._history:
                last_used = item.get("_last_used_epoch", item.get("_created_epoch", now))
                try:
                    last_used_f = float(last_used)
                except Exception:
                    last_used_f = now
                if now - last_used_f <= self._cache_idle_ttl_seconds:
                    kept.append(item)
            self._history = kept

        if self._max_history_requests > 0 and len(self._history) > self._max_history_requests:
            # Keep newest items only when hitting memory cap.
            self._history = self._history[-self._max_history_requests :]

    def _encode_prompt(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        try:
            from deepseek_encoding import encode_messages
        except Exception:  # pragma: no cover - package execution path
            from .deepseek_encoding import encode_messages  # type: ignore

        try:
            return encode_messages(
                messages,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
        except Exception:
            return canonicalize_messages(messages)

    def _load_openclaw_global_floor_by_model(self) -> Dict[str, int]:
        # OpenClaw-only seed floor from historical traces, so first turn can
        # approximate warm-cache behavior across sessions.
        model_samples: Dict[str, List[int]] = {}
        for path in self.traces_dir.glob("*.jsonl"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if row.get("conversation_mode") != "openclaw_agent":
                            continue
                        if row.get("input_token_source") != "openclaw_raw_body":
                            continue
                        model = row.get("model")
                        if not isinstance(model, str) or not model:
                            continue
                        cached = row.get("actual_cached_tokens")
                        if not isinstance(cached, int) or cached <= 0:
                            continue
                        model_samples.setdefault(model, []).append(cached)
            except Exception:
                continue

        floors: Dict[str, int] = {}
        for model, samples in model_samples.items():
            if not samples:
                continue
            m = int(median(samples))
            m = _snap_down(m, self._block_size)
            if m > 0:
                floors[model] = m
        return floors

    def create_request_record(
        self,
        session_id: str,
        request_id: str,
        timestamp: str,
        request_body: Any,
        request_body_bytes: Optional[bytes] = None,
        conversation_mode: str = "simple_streaming",
    ) -> Dict[str, Any]:
        body = request_body if isinstance(request_body, dict) else {}
        messages = body.get("messages", [])
        canonical_text = canonicalize_messages(messages)

        thinking_mode = _extract_thinking_mode(body)
        reasoning_effort: Optional[str] = None
        if conversation_mode == "piai_probe":
            messages_for_encoding = _normalize_messages_for_piai_probe(
                messages=messages,
                top_level_tools=body.get("tools"),
            )
            reasoning_effort = _extract_reasoning_effort(body)
        else:
            messages_for_encoding = _normalize_messages_for_v4(messages)

        if self._tokenizer.info.effective_preset == "deepseek-v4-pro":
            prompt_text = self._encode_prompt(
                messages_for_encoding,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
            token_ids = self._tokenizer.tokenize_text(prompt_text)
        else:
            token_ids = self._tokenizer.tokenize_messages(
                messages_for_encoding,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
        raw_body_sha256: Optional[str] = None
        raw_body_size_bytes = 0
        raw_body_token_count: Optional[int] = None
        raw_body_token_ids: List[int] = []
        if isinstance(request_body_bytes, (bytes, bytearray)):
            raw_bytes = bytes(request_body_bytes)
            raw_body_size_bytes = len(raw_bytes)
            raw_body_sha256 = _sha256_hex(raw_bytes)
            try:
                raw_body_text = raw_bytes.decode("utf-8", errors="replace")
                raw_body_token_ids = self._tokenizer.tokenize_text(raw_body_text)
                raw_body_token_count = len(raw_body_token_ids)
            except Exception:
                raw_body_token_count = None
                raw_body_token_ids = []

        snapped_prefix_len = _snap_down(len(token_ids), self._block_size)
        # Request boundary unit: the aligned input prefix is persisted for future matching.
        request_prefix_tokens = token_ids[:snapped_prefix_len] if snapped_prefix_len > 0 else []
        units: List[List[int]] = [request_prefix_tokens] if request_prefix_tokens else []

        return {
            "session_id": session_id,
            "request_id": request_id,
            "timestamp": timestamp,
            "model": body.get("model"),
            "messages": messages,
            "canonical_text": canonical_text,
            "token_ids": token_ids,
            "local_input_tokens": len(token_ids),
            "raw_request_body_sha256": raw_body_sha256,
            "raw_request_body_size_bytes": raw_body_size_bytes,
            "raw_request_body_tokenizer_tokens": raw_body_token_count,
            "raw_request_body_token_ids": raw_body_token_ids,
            "persisted_prefix_units_tokens": units,
            "cache_block_size": self._block_size,
            "cache_estimation_input_tokens": len(token_ids),
            "_openclaw_global_floor_tokens": self._openclaw_global_floor_by_model.get(
                str(body.get("model", "")),
                0,
            ),
            "_cache_unit_source": "deepseek_prompt_encoding",
            "_cache_unit_fallback_reason": None,
            "tokenizer_preset": self._tokenizer.info.preset,
            "tokenizer_effective_preset": self._tokenizer.info.effective_preset,
            "tokenizer_runtime": self._tokenizer.info.runtime,
            "tokenizer_dir": self._tokenizer.info.tokenizer_dir,
            "tokenizer_warning": self._tokenizer.info.warning,
            "_thinking_mode": thinking_mode,
            "_reasoning_effort": reasoning_effort,
            "_messages_for_encoding": messages_for_encoding,
        }

    def apply_input_token_source(
        self,
        request_record: Dict[str, Any],
        input_token_source: str,
    ) -> Dict[str, Any]:
        if input_token_source != "openclaw_raw_body":
            request_record["_cache_unit_source"] = "deepseek_prompt_encoding"
            request_record["_cache_unit_fallback_reason"] = None
            return request_record

        raw_ids = request_record.get("raw_request_body_token_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            request_record["_cache_unit_source"] = "deepseek_prompt_encoding"
            request_record["_cache_unit_fallback_reason"] = "raw_request_body_unavailable"
            return request_record

        block_size = request_record.get("cache_block_size", self._block_size)
        try:
            block_size_i = max(int(block_size), 1)
        except Exception:
            block_size_i = self._block_size

        snapped_prefix_len = _snap_down(len(raw_ids), block_size_i)
        request_prefix_tokens = raw_ids[:snapped_prefix_len] if snapped_prefix_len > 0 else []

        # Switch cache-estimation basis to raw request body token stream.
        request_record["token_ids"] = raw_ids
        request_record["persisted_prefix_units_tokens"] = (
            [request_prefix_tokens] if request_prefix_tokens else []
        )
        request_record["cache_estimation_input_tokens"] = len(raw_ids)
        request_record["_cache_unit_source"] = "raw_request_body"
        request_record["_cache_unit_fallback_reason"] = None
        return request_record

    def attach_response_cache_units(
        self,
        request_record: Dict[str, Any],
        response_json: Any,
    ) -> Dict[str, Any]:
        if request_record.get("_cache_unit_source") == "raw_request_body":
            # Keep unit space consistent: raw-body mode only persists request-boundary units.
            return request_record

        # DeepSeek conservative baseline:
        # keep request boundary + output boundary, but do not add shared-prefix / interval units.
        units = request_record.get("persisted_prefix_units_tokens")
        if not isinstance(units, list):
            units = []

        existing = {tuple(u) for u in units if isinstance(u, list)}

        assistant_message = _extract_assistant_message_from_response(response_json)
        if assistant_message:
            thinking_mode = str(request_record.get("_thinking_mode", "chat"))
            reasoning_effort = request_record.get("_reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            base_messages = request_record.get("_messages_for_encoding")
            if not isinstance(base_messages, list):
                base_messages = _normalize_messages_for_v4(request_record.get("messages", []))

            output_messages = copy.deepcopy(base_messages)
            output_messages.append(assistant_message)
            if self._tokenizer.info.effective_preset == "deepseek-v4-pro":
                output_prompt = self._encode_prompt(
                    output_messages,
                    thinking_mode=thinking_mode,
                    reasoning_effort=reasoning_effort,
                )
                output_tokens = self._tokenizer.tokenize_text(output_prompt)
            else:
                output_tokens = self._tokenizer.tokenize_messages(
                    output_messages,
                    thinking_mode=thinking_mode,
                    reasoning_effort=reasoning_effort,
                    add_generation_prompt=False,
                )
            snapped_len = _snap_down(len(output_tokens), self._block_size)
            if snapped_len > 0:
                # Output boundary unit: include assistant turn, then align and persist.
                output_unit = output_tokens[:snapped_len]
                key = tuple(output_unit)
                if key not in existing:
                    units.append(output_unit)

        request_record["persisted_prefix_units_tokens"] = units
        return request_record

    def history_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._prune_history_locked()
            return copy.deepcopy(self._history)

    def touch_request(self, request_id: Optional[str]) -> None:
        if not request_id:
            return
        with self._lock:
            now = self._now_epoch()
            for item in reversed(self._history):
                if item.get("request_id") == request_id:
                    item["_last_used_epoch"] = now
                    break
            self._prune_history_locked(now)

    def append_history(self, request_record: Dict[str, Any]) -> None:
        with self._lock:
            now = self._now_epoch()
            record_copy = copy.deepcopy(request_record)
            record_copy["_created_epoch"] = now
            record_copy["_last_used_epoch"] = now
            self._history.append(record_copy)
            self._prune_history_locked(now)

    def append_jsonl(
        self,
        session_id: str,
        log_payload: Dict[str, Any],
    ) -> None:
        trace_path = self.traces_dir / f"{session_id}.jsonl"
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_payload, ensure_ascii=False) + "\n")
