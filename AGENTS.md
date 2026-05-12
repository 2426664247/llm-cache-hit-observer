# Agent 快速使用文档

这份文档写给后续阅读、维护或调用 `cache_hit_proxy` 的 Agent。目标是让 Agent 在几分钟内理解项目边界、核心文件、运行方式和安全规则。

## 先读结论

`cache_hit_proxy` 是一个本地 OpenAI-compatible HTTP 代理。它接收客户端发来的 `POST /v1/chat/completions`，转发给上游模型服务，在本地估算 prompt token 与 Prompt Cache 命中，并把上游真实 `usage` 与本地估算一起写入 JSONL trace。

这个仓库只应包含代理工具本体。不要把实验结果、本地服务信息、API key、raw response、trace 数据、日志或个人路径写入文档或提交。

## 项目边界

你可以关注：

- 代理源码。
- 单元测试。
- README 和本 Agent 文档。
- 必要的轻量 tokenizer 资源。
- `traces/.gitkeep` 空目录占位。

你不应该提交：

- `traces/*.jsonl`
- `*.log`
- `TOKENIZER_VALIDATION*.md`
- `tokenizers/`
- `__pycache__/`
- `.pytest_cache/`
- 本地配置、密钥、机器路径、SSH alias、私有 provider 信息。

如果你是在一个更大的开发工作区里看到实验目录或私有文档，把它们视为用户本地上下文，不要当成这个仓库的一部分。

## 5 分钟读懂代码

建议按这个顺序看：

| 文件 | 你应该了解什么 |
| --- | --- |
| `README.md` | 项目用途、启动方式、模式和安全边界。 |
| `main.py` | CLI 参数、FastAPI app、请求转发、trace 写入主流程。 |
| `request_recorder.py` | 请求规范化、tokenizer 调用、历史请求管理。 |
| `cache_estimator.py` | 前缀匹配、block 对齐、缓存命中估算。 |
| `usage_reader.py` | 从 JSON/SSE 响应读取真实 usage。 |
| `vllm_metrics.py` | vLLM `/metrics` 解析、delta 计算、真实 cache hit 读取。 |
| `tokenizer_adapters.py` | tokenizer preset、HF tokenizer、本地 fallback 和 vLLM `/tokenize` adapter。 |
| `tests/` | 当前行为契约和回归测试。 |

主流程心智模型：

```text
client request
  -> main.py receives body
  -> request_recorder builds local token record
  -> cache_estimator estimates reusable prefix/cache tokens
  -> main.py forwards request upstream
  -> usage_reader or vllm_metrics reads actual usage
  -> main.py appends traces/{session_id}.jsonl
  -> original upstream response returns to client
```

## 环境准备

从仓库目录进入工具目录：

```bash
cd cache_hit_proxy
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

检查版本：

```bash
python main.py --version
```

本地 tokenizer 检查：

```bash
python download_tokenizers.py --check-only
```

如果用户允许下载缺失资源：

```bash
python download_tokenizers.py
```

## 如何选择运行模式

| 场景 | `--conversation-mode` | `--tokenizer-preset` | 备注 |
| --- | --- | --- | --- |
| 普通 OpenAI-compatible provider | `simple_streaming` | 按 provider 选择最接近 preset | 最通用。 |
| DeepSeek 风格请求 | `simple_streaming` | `deepseek-v4-pro` | 使用内置 tokenizer。 |
| OpenClaw agent 流量 | `openclaw_agent` | 通常 `deepseek-v4-pro` | raw body token 来源更重要。 |
| pi-ai 最终 payload | `piai_probe` | 通常 `deepseek-v4-pro` | 窄适配，不要当通用 tools 模式。 |
| vLLM 服务 | `vllm_probe` | `vllm` | 需要 `/tokenize` 和 `/metrics`。 |
| tokenizer 不确定 | 先按用户提供的 provider/model 选择 | 如果 fallback，要明确说明误差 | 不要假装精确。 |

## 如何选择上游 URL

优先用 `--target-base-url`：

```bash
--target-base-url https://YOUR_PROVIDER_BASE_URL
```

代码会推导：

```text
{base}/v1/chat/completions
```

如果 provider 的 endpoint 不符合这个规则，使用完整 URL：

```bash
--target-chat-url https://YOUR_PROVIDER_FULL_CHAT_COMPLETIONS_URL
```

不要把 API key 放进 URL、README、AGENTS.md、trace 或提交记录。API key 应由客户端按上游 provider 原本要求传入，例如通过 `Authorization` header。

## 常用启动模板

通用 provider：

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

非标准 chat endpoint：

```bash
python main.py \
  --port 8787 \
  --target-chat-url https://YOUR_PROVIDER_FULL_CHAT_COMPLETIONS_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode simple_streaming
```

OpenClaw agent：

```bash
python main.py \
  --port 8787 \
  --target-base-url https://YOUR_PROVIDER_BASE_URL \
  --tokenizer-preset deepseek-v4-pro \
  --conversation-mode openclaw_agent \
  --raw-request-capture none
```

vLLM 标准 endpoint：

```bash
python main.py \
  --port 8787 \
  --target-base-url http://YOUR_VLLM_HOST:8000/v1 \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe \
  --session-id vllm_probe
```

vLLM 显式 endpoint：

```bash
python main.py \
  --port 8787 \
  --target-chat-url http://YOUR_VLLM_HOST:8000/v1/chat/completions \
  --vllm-tokenize-url http://YOUR_VLLM_HOST:8000/tokenize \
  --vllm-metrics-url http://YOUR_VLLM_HOST:8000/metrics \
  --tokenizer-preset vllm \
  --conversation-mode vllm_probe
```

启动后客户端指向：

```text
http://127.0.0.1:8787/v1
```

## Trace 处理规则

默认 trace：

```text
traces/{session_id}.jsonl
```

Trace 可能包含 prompt、模型名、usage、请求形态和诊断字段。默认视为私有数据。

推荐默认参数：

```bash
--raw-request-capture none
```

只有用户明确需要请求回放或字节级调试时，才使用：

```bash
--raw-request-capture utf8
--raw-request-capture base64
```

## 测试和验证

代码变更后运行：

```bash
python -m pytest
```

当前测试重点：

- tokenizer adapter 行为。
- vLLM metrics 解析和 delta 计算。

真实 provider tokenizer 验证会消耗 API quota，只在用户明确要求并已提供凭据时运行：

```bash
python validate_tokenizers.py --sample short
python validate_tokenizers.py --sample long-1000
```

## 常见任务指引

新增 tokenizer preset：

1. 看 `tokenizer_adapters.py`。
2. 添加 preset 规范化、默认目录和加载逻辑。
3. 如需下载资源，更新 `download_tokenizers.py`。
4. 增加或更新 `tests/test_tokenizer_adapters.py`。
5. 更新 README 的 tokenizer 表格。

调整 usage 解析：

1. 看 `usage_reader.py`。
2. 为 JSON 或 SSE 中的新字段增加解析分支。
3. 保持字段缺失时的容错行为。
4. 增加测试，避免破坏已有 provider。

调整 vLLM metrics：

1. 看 `vllm_metrics.py`。
2. 保持 metrics 缺失或格式变化时不崩溃。
3. 跑 `tests/test_vllm_metrics.py`。
4. README 只写通用 endpoint，不写个人 vLLM 地址。

调整代理入口或 CLI：

1. 看 `main.py` 的 `parse_args()` 和 `create_app()`。
2. 新参数应有默认值和 help 文案。
3. 如果参数影响用户使用，更新 README 和本文件。

## 提交前检查

提交前至少运行：

```bash
python main.py --version
python -m pytest
git status --short
```

确认没有这些内容进入 staged files：

```text
traces/*.jsonl
*.log
TOKENIZER_VALIDATION*.md
tokenizers/
__pycache__/
.pytest_cache/
.env
```

如果文档里出现了真实 provider 地址、API key、个人路径、SSH alias、隧道端口或实验目录名，提交前必须移除。

## Agent 回复用户时的摘要模板

```text
模式: <conversation-mode>
Tokenizer: <tokenizer-preset 或 tokenizer-dir>
上游: <只写域名或本地主机, 不写密钥>
代理: http://127.0.0.1:<port>/v1
Trace: traces/<session-id>.jsonl
验证: <pytest / version / provider validation>
注意: <fallback、vLLM metrics 污染风险或 raw capture 风险>
```
