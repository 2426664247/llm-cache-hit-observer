import copy
import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

try:
    from deepseek_encoding import encode_messages
except Exception:  # pragma: no cover - package execution path
    from .deepseek_encoding import encode_messages  # type: ignore


def _stable_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


class DeepSeekTokenizer:
    def __init__(self, tokenizer_dir: str) -> None:
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "DeepSeek tokenizer requires 'tokenizers'. "
                "Please install dependencies from requirements.txt."
            ) from exc

        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise RuntimeError(f"tokenizer.json not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def tokenize_text(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(text)
        return [int(t) for t in encoded.ids]


class RequestRecorder:
    def __init__(
        self,
        traces_dir: str,
        tokenizer_dir: str,
        block_size: int = 64,
        cache_idle_ttl_hours: float = 24.0,
        max_history_requests: int = 2000,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = DeepSeekTokenizer(tokenizer_dir)
        self._history: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._block_size = max(int(block_size), 1)
        self._cache_idle_ttl_seconds = max(float(cache_idle_ttl_hours), 0.0) * 3600.0
        self._max_history_requests = max(int(max_history_requests), 1)

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

        if len(self._history) > self._max_history_requests:
            # Keep newest items only when hitting memory cap.
            self._history = self._history[-self._max_history_requests :]

    def _encode_prompt(self, messages: List[Dict[str, Any]], thinking_mode: str) -> str:
        try:
            return encode_messages(messages, thinking_mode=thinking_mode)
        except Exception:
            return canonicalize_messages(messages)

    def create_request_record(
        self,
        session_id: str,
        request_id: str,
        timestamp: str,
        request_body: Any,
    ) -> Dict[str, Any]:
        body = request_body if isinstance(request_body, dict) else {}
        messages = body.get("messages", [])
        canonical_text = canonicalize_messages(messages)

        thinking_mode = _extract_thinking_mode(body)
        messages_for_encoding = _normalize_messages_for_v4(messages)
        prompt_text = self._encode_prompt(messages_for_encoding, thinking_mode=thinking_mode)
        token_ids = self._tokenizer.tokenize_text(prompt_text)
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
            "persisted_prefix_units_tokens": units,
            "cache_block_size": self._block_size,
            "_thinking_mode": thinking_mode,
            "_messages_for_encoding": messages_for_encoding,
        }

    def attach_response_cache_units(
        self,
        request_record: Dict[str, Any],
        response_json: Any,
    ) -> Dict[str, Any]:
        # DeepSeek conservative baseline:
        # keep request boundary + output boundary, but do not add shared-prefix / interval units.
        units = request_record.get("persisted_prefix_units_tokens")
        if not isinstance(units, list):
            units = []

        existing = {tuple(u) for u in units if isinstance(u, list)}

        assistant_message = _extract_assistant_message_from_response(response_json)
        if assistant_message:
            thinking_mode = str(request_record.get("_thinking_mode", "chat"))
            base_messages = request_record.get("_messages_for_encoding")
            if not isinstance(base_messages, list):
                base_messages = _normalize_messages_for_v4(request_record.get("messages", []))

            output_messages = copy.deepcopy(base_messages)
            output_messages.append(assistant_message)
            output_prompt = self._encode_prompt(output_messages, thinking_mode=thinking_mode)
            output_tokens = self._tokenizer.tokenize_text(output_prompt)
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
