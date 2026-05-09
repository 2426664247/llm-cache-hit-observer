import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from request_recorder import RequestRecorder  # noqa: E402
from tokenizer_adapters import (  # noqa: E402
    TokenizerInfo,
    default_tokenizer_dir_for_preset,
    effective_tokenizer_preset,
    normalize_tokenizer_preset,
)


class FakeTokenizerAdapter:
    def __init__(self) -> None:
        self.info = TokenizerInfo(
            preset="glm-5.1",
            effective_preset="glm-5.1",
            tokenizer_dir="fake",
        )

    def tokenize_text(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def tokenize_messages(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str,
        reasoning_effort: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        ids: List[int] = []
        for msg in messages:
            ids.extend(ord(ch) for ch in str(msg.get("role", "")))
            ids.extend(ord(ch) for ch in str(msg.get("content", "")))
        if add_generation_prompt:
            ids.append(999)
        return ids


class TokenizerAdapterTests(unittest.TestCase):
    def test_preset_aliases_and_doubao_fallback(self) -> None:
        self.assertEqual(normalize_tokenizer_preset("qwen"), "qwen3-coder-plus")
        self.assertEqual(normalize_tokenizer_preset("volcano"), "volcanoengine")
        self.assertEqual(
            effective_tokenizer_preset("doubao-seed-2-0-code-preview-260215"),
            "deepseek-v4-pro",
        )

    def test_default_dirs(self) -> None:
        base = Path("cache_hit_proxy")
        self.assertEqual(
            default_tokenizer_dir_for_preset("deepseek-v4-pro", base),
            base / "deepseek_tokenizer",
        )
        self.assertEqual(
            default_tokenizer_dir_for_preset("glm-5.1", base),
            base / "tokenizers" / "glm-5.1",
        )
        self.assertEqual(
            default_tokenizer_dir_for_preset("volcanoengine", base),
            base / "deepseek_tokenizer",
        )

    def test_request_recorder_uses_adapter_metadata(self) -> None:
        recorder = RequestRecorder(
            traces_dir=str(Path(__file__).resolve().parents[1] / "traces"),
            tokenizer_dir="fake",
            tokenizer_adapter=FakeTokenizerAdapter(),
            block_size=4,
            cache_idle_ttl_hours=0,
        )
        body = {
            "model": "glm-5.1",
            "messages": [{"role": "user", "content": "hi"}],
        }
        record = recorder.create_request_record(
            session_id="test",
            request_id="req_1",
            timestamp="2026-05-09T00:00:00",
            request_body=body,
            request_body_bytes=b"{}",
            conversation_mode="simple_streaming",
        )
        self.assertEqual(record["tokenizer_preset"], "glm-5.1")
        self.assertEqual(record["tokenizer_effective_preset"], "glm-5.1")
        self.assertGreater(record["local_input_tokens"], 0)
        self.assertEqual(record["tokenizer_runtime"], "local")


if __name__ == "__main__":
    unittest.main()
