from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

try:
    from deepseek_encoding import encode_messages
except Exception:  # pragma: no cover - package execution path
    from .deepseek_encoding import encode_messages  # type: ignore


SUPPORTED_TOKENIZER_PRESETS = {
    "deepseek-v4-pro",
    "glm-5.1",
    "qwen3-coder-plus",
    "kimi-k2.6",
    "doubao-seed-2-0-code-preview-260215",
    "volcanoengine",
}

HF_REPO_BY_PRESET = {
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "glm-5.1": "zai-org/GLM-5.1",
    "qwen3-coder-plus": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
}

HF_PRESETS = {"glm-5.1", "qwen3-coder-plus", "kimi-k2.6"}
DEEPSEEK_FALLBACK_PRESETS = {
    "deepseek-v4-pro",
    "doubao-seed-2-0-code-preview-260215",
    "volcanoengine",
}


@dataclass(frozen=True)
class TokenizerInfo:
    preset: str
    effective_preset: str
    tokenizer_dir: str
    runtime: str = "local"
    warning: Optional[str] = None


class TokenizerAdapter(Protocol):
    info: TokenizerInfo

    def tokenize_text(self, text: str) -> List[int]:
        ...

    def tokenize_messages(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str,
        reasoning_effort: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        ...


def normalize_tokenizer_preset(preset: Optional[str]) -> str:
    normalized = (preset or "deepseek-v4-pro").strip().lower()
    aliases = {
        "deepseek": "deepseek-v4-pro",
        "deepseek_v4_pro": "deepseek-v4-pro",
        "deepseek-v4": "deepseek-v4-pro",
        "ds": "deepseek-v4-pro",
        "glm": "glm-5.1",
        "glm_5_1": "glm-5.1",
        "qwen": "qwen3-coder-plus",
        "qwen3_coder_plus": "qwen3-coder-plus",
        "kimi": "kimi-k2.6",
        "kimi_k2_6": "kimi-k2.6",
        "doubao": "doubao-seed-2-0-code-preview-260215",
        "volcano": "volcanoengine",
        "volcano_engine": "volcanoengine",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_TOKENIZER_PRESETS:
        supported = ", ".join(sorted(SUPPORTED_TOKENIZER_PRESETS))
        raise ValueError(f"unsupported tokenizer preset '{preset}'. Supported: {supported}")
    return normalized


def effective_tokenizer_preset(preset: str) -> str:
    if preset in {"doubao-seed-2-0-code-preview-260215", "volcanoengine"}:
        return "deepseek-v4-pro"
    return preset


def default_tokenizer_dir_for_preset(preset: str, base_dir: Path) -> Path:
    effective = effective_tokenizer_preset(normalize_tokenizer_preset(preset))
    if effective == "deepseek-v4-pro":
        return base_dir / "deepseek_tokenizer"
    return base_dir / "tokenizers" / effective


def _tokenizer_warning_for_preset(preset: str, effective: str) -> Optional[str]:
    if preset != effective and effective == "deepseek-v4-pro":
        return "official_local_tokenizer_unavailable_using_deepseek_fallback"
    return None


class DeepSeekV4TokenizerAdapter:
    def __init__(self, tokenizer_dir: str, preset: str = "deepseek-v4-pro") -> None:
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "DeepSeek tokenizer requires 'tokenizers'. "
                "Please install dependencies from requirements.txt."
            ) from exc

        normalized = normalize_tokenizer_preset(preset)
        effective = effective_tokenizer_preset(normalized)
        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise RuntimeError(f"tokenizer.json not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.info = TokenizerInfo(
            preset=normalized,
            effective_preset=effective,
            tokenizer_dir=str(Path(tokenizer_dir).resolve()),
            warning=_tokenizer_warning_for_preset(normalized, effective),
        )

    def tokenize_text(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(text)
        return [int(t) for t in encoded.ids]

    def tokenize_messages(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str,
        reasoning_effort: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        # DeepSeek rendering decides whether to add the assistant transition from
        # the message roles. The add_generation_prompt flag is honored by the
        # caller for HF templates, but is intentionally not needed here.
        prompt_text = encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        return self.tokenize_text(prompt_text)


class HuggingFaceChatTemplateTokenizerAdapter:
    def __init__(
        self,
        tokenizer_dir: str,
        preset: str,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face chat-template tokenizers require 'transformers'. "
                "Please install dependencies from requirements.txt."
            ) from exc

        normalized = normalize_tokenizer_preset(preset)
        if normalized not in HF_PRESETS:
            raise ValueError(f"preset '{preset}' is not an HF chat-template preset")
        tokenizer_path = Path(tokenizer_dir)
        if not tokenizer_path.exists():
            raise RuntimeError(f"tokenizer directory not found: {tokenizer_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.info = TokenizerInfo(
            preset=normalized,
            effective_preset=normalized,
            tokenizer_dir=str(tokenizer_path.resolve()),
        )

    def tokenize_text(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(t) for t in encoded]

    def tokenize_messages(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str,
        reasoning_effort: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        template_messages = _prepare_hf_messages(messages)
        token_ids = self.tokenizer.apply_chat_template(
            template_messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        if hasattr(token_ids, "keys") and "input_ids" in token_ids:
            token_ids = token_ids.get("input_ids", [])
        return [int(t) for t in token_ids]


def _prepare_hf_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        if role == "developer":
            role = "system"

        content = msg.get("content", "")
        if not isinstance(content, str):
            import json

            content = json.dumps(content, ensure_ascii=False, sort_keys=True)

        item: Dict[str, Any] = {"role": role, "content": content}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in msg:
                item[key] = msg[key]
        prepared.append(item)
    return prepared


def create_tokenizer_adapter(
    preset: str,
    tokenizer_dir: str,
    hf_local_files_only: bool = True,
) -> TokenizerAdapter:
    normalized = normalize_tokenizer_preset(preset)
    effective = effective_tokenizer_preset(normalized)
    if effective == "deepseek-v4-pro":
        return DeepSeekV4TokenizerAdapter(tokenizer_dir=tokenizer_dir, preset=normalized)
    return HuggingFaceChatTemplateTokenizerAdapter(
        tokenizer_dir=tokenizer_dir,
        preset=effective,
        local_files_only=hf_local_files_only,
        trust_remote_code=True,
    )
