from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from tokenizer_adapters import (
    HF_PRESETS,
    HF_REPO_BY_PRESET,
    default_tokenizer_dir_for_preset,
    normalize_tokenizer_preset,
)


ALLOW_PATTERNS = [
    "tokenizer*",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tiktoken.model",
    "tokenization_*.py",
    "special_tokens_map.json",
    "added_tokens.json",
]

RAW_FILES_BY_PRESET = {
    "glm-5.1": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ],
    "qwen3-coder-plus": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    ],
    "kimi-k2.6": [
        "config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "tokenization_kimi.py",
        "tool_declaration_ts.py",
        "tiktoken.model",
    ],
}


def download_tokenizer(preset: str, output_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "download_tokenizers.py requires 'huggingface_hub'. "
            "Please install dependencies from requirements.txt."
        ) from exc

    normalized = normalize_tokenizer_preset(preset)
    if normalized not in HF_PRESETS:
        print(f"[skip] {normalized}: uses bundled/default tokenizer or fallback")
        return

    repo_id = HF_REPO_BY_PRESET[normalized]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {normalized}: {repo_id} -> {output_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=["*.safetensors", "*.bin", "*.gguf", "*.pt", "*.pth"],
    )


def download_tokenizer_raw(preset: str, output_dir: Path) -> None:
    normalized = normalize_tokenizer_preset(preset)
    if normalized not in HF_PRESETS:
        print(f"[skip] {normalized}: uses bundled/default tokenizer or fallback")
        return
    repo_id = HF_REPO_BY_PRESET[normalized]
    files = RAW_FILES_BY_PRESET[normalized]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{name}"
        target = output_dir / name
        print(f"[raw] {normalized}: {name}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"[raw] {normalized}: optional missing {name}")
                continue
            raise
        target.write_bytes(data)


def check_tokenizer(preset: str, output_dir: Path) -> bool:
    normalized = normalize_tokenizer_preset(preset)
    if normalized not in HF_PRESETS:
        ok = (output_dir / "tokenizer.json").exists()
        status = "ok" if ok else "missing"
        print(f"[check] {normalized}: {status} ({output_dir})")
        return ok

    required_any = [
        output_dir / "tokenizer.json",
        output_dir / "tiktoken.model",
    ]
    ok = output_dir.exists() and any(path.exists() for path in required_any)
    if normalized in {"glm-5.1", "qwen3-coder-plus"}:
        ok = ok and (output_dir / "chat_template.jinja").exists()
    if normalized == "kimi-k2.6":
        ok = (
            ok
            and (output_dir / "tokenization_kimi.py").exists()
            and (output_dir / "tool_declaration_ts.py").exists()
        )
    status = "ok" if ok else "missing"
    print(f"[check] {normalized}: {status} ({output_dir})")
    return ok


def iter_presets(raw_presets: Iterable[str]) -> list[str]:
    presets = [normalize_tokenizer_preset(item) for item in raw_presets]
    if not presets:
        presets = sorted(HF_PRESETS)
    return presets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local tokenizer-only assets")
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Tokenizer preset to download. Repeatable. Defaults to all HF presets.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="Base cache_hit_proxy directory (default: this script's directory).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether tokenizer files are present.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    failures = 0
    for preset in iter_presets(args.preset):
        output_dir = default_tokenizer_dir_for_preset(preset, base_dir)
        if args.check_only:
            if not check_tokenizer(preset, output_dir):
                failures += 1
            continue
        try:
            download_tokenizer(preset, output_dir)
            if not check_tokenizer(preset, output_dir):
                print(f"[warn] {preset}: snapshot incomplete, trying raw download")
                download_tokenizer_raw(preset, output_dir)
                if not check_tokenizer(preset, output_dir):
                    failures += 1
        except Exception as exc:
            print(f"[warn] {preset}: snapshot download failed: {exc}")
            try:
                download_tokenizer_raw(preset, output_dir)
                if not check_tokenizer(preset, output_dir):
                    failures += 1
            except Exception as raw_exc:
                failures += 1
                print(f"[error] {preset}: raw download failed: {raw_exc}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
