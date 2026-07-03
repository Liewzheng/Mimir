# Mimir

> **A plastic memory well for coding agents.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-mimir--core-blue.svg)](https://pypi.org/project/mimir-core/)
[![GitHub](https://img.shields.io/badge/github-Liewzheng%2FMimir-black.svg)](https://github.com/Liewzheng/Mimir)

Unlike vector notebooks, **Mimir** reshapes its own embedding space as you work.
One matrix. Zero cloud. Always local.

---

## Why Mimir?

Most agent memory tools are **filing cabinets**: they store text, then search it.
They don't learn. They don't adapt. They just retrieve.

Mimir is different.

It maintains a small, fixed-size **prototype matrix** that is updated with every
interaction via Hebbian-style local learning. The same embedding model keeps its
base weights, but Mimir overlays a fast, plastic layer that bends toward *your*
domain, *your* project, and *your* habits.

The result: a memory system that feels less like a database and more like a
second brain that gets sharper the more you use it.

### What makes Mimir unique

| | Other memory tools | Mimir |
|---|---|---|
| **Core model** | Store and retrieve discrete facts | Learn a plastic prototype matrix |
| **Learning** | Heuristic scoring, TTL decay, or no learning at all | Hebbian / Oja local updates + Markov prediction |
| **State size** | Often grows with history (SQLite/vector DB) | Fixed `[k × dim]` matrix, typically < 100 MB |
| **Offline** | Needs cloud API or hosted vector DB | Runs locally with llama-server, sentence-transformers, or fake backend |
| **Inference** | Calls DB/index on every recall | Pure matrix operation, no locks, predictable latency |
| **Hook noise** | Stores every short reply, including "ok" and "continue" | Language-aware small-talk filtering + importance gating |
| **Security** | Stores secrets verbatim | Automatic API key / token / password redaction |
| **Project context** | Manual copy-paste of instruction files | Auto-discovers `AGENTS.md`, `CLAUDE.md`, `.cursorrules` |
| **Quality** | Duplicate memories accumulate | Duplicate blocking + contradiction hints at store time |

Secret redaction patterns can be customized or disabled entirely via
`MimirConfig(redaction_patterns=...)` (`None` = defaults, `[]` = disabled).

---

## Quick Start

```bash
pip install mimir-core
```

### Use it from Python

```python
from mimir import Mimir, MimirConfig

config = MimirConfig(
    base_model="all-MiniLM-L6-v2",
    num_prototypes=64,
    top_k=4,
)
mimir = Mimir(config)

# Encode
emb = mimir.encode("hello world")
print(emb.shape)  # (1, 384)

# Learn
report = mimir.learn("hello world", importance=1.0)
print(report)

# Save / load
mimir.save("checkpoint.pt")
mimir.load("checkpoint.pt")
```

### Plug it into your coding agent

Mimir exposes an MCP server and Agent CLI hooks for Kimi Code, Claude Code,
Codex, and OpenCode. Each workspace is isolated under `~/.mimir/workspaces/`.

#### Start the MCP server

```bash
# Local embedding backend (recommended)
llama-server \
  --model Qwen3-Embedding-8B-Q4_K_M.gguf \
  --embeddings \
  --port 11435

# Or use sentence-transformer if you don't have llama-server
mimir mcp --backend sentence-transformer
```

#### Configure Kimi Code / Claude Code / Codex (`.mcp.json`)

```json
{
  "mcpServers": {
    "mimir": {
      "command": "mimir",
      "args": ["mcp", "--backend", "sentence-transformer"]
    }
  }
}
```

#### Configure OpenCode (`.opencode/opencode.jsonc`)

```jsonc
{
  "mcp": {
    "mimir": {
      "type": "local",
      "command": ["mimir", "mcp", "--backend", "sentence-transformer"]
    }
  }
}
```

#### One-shot setup with `mimir setup`

Instead of editing config files by hand, you can install hooks automatically:

```bash
mimir setup kimi-code
mimir setup claude-code
mimir setup codex
mimir setup opencode
```

Use `--base-dir` to write the configuration somewhere other than the agent's
default config directory (useful for custom dotfiles layouts or CI):

```bash
mimir setup kimi-code --base-dir ~/my-configs/kimi-code
```

This writes the correct hook definitions for each agent CLI and is safe to run
multiple times.

## For AI Agents

If you are an AI agent installing Mimir on behalf of a user, use this checklist.

### 1. Install Mimir

```bash
pip install mimir-core
```

Or install from the repo root:

```bash
pip install -e .
```

Verify:

```bash
mimir --help
python -m mimir.hooks.mimir_turn --help
```

### 2. Start an embedding backend

Choose one backend. The default is `llama-server` at `http://127.0.0.1:11435`.

- **sentence-transformer** — easiest, no extra server, first load ~20s:
  ```bash
  mimir mcp --backend sentence-transformer
  ```
- **llama-server** — higher quality, requires a running server:
  ```bash
  llama-server \
    --model Qwen3-Embedding-8B-Q4_K_M.gguf \
    --embeddings \
    --port 11435
  ```
- **ollama** — if the user already runs Ollama:
  ```bash
  mimir mcp --backend ollama --base-url http://127.0.0.1:11434 --model ...
  ```

### 3. Add the MCP server to the user's agent config

- **OpenCode** (`~/.config/opencode/opencode.json`):
  ```json
  {
    "mcp": {
      "mimir": {
        "type": "local",
        "command": ["mimir", "mcp", "--backend", "sentence-transformer"]
      }
    }
  }
  ```
- **Claude Code / Kimi Code / Codex** (`.mcp.json`):
  ```json
  {
    "mcpServers": {
      "mimir": {
        "command": "mimir",
        "args": ["mcp", "--backend", "sentence-transformer"]
      }
    }
  }
  ```

### 4. Install automatic agent hooks (optional)

```bash
mimir setup kimi-code
mimir setup claude-code
mimir setup codex
mimir setup opencode
```

This adds `UserPromptSubmit`/`Stop` hooks so Mimir recalls context on each turn
and stores the user/assistant exchange automatically.

### 5. Tell the user to restart their agent CLI

MCP servers and hooks are loaded on startup. After restarting, the agent can use:

- `store(text)` / `recall(query)` via MCP.
- Automatic recall/store via hooks if installed.

---

#### Manual hook configuration

```toml
# ~/.kimi-code/config.toml
[[hooks]]
event = "UserPromptSubmit"
command = "python3 -m mimir.hooks.mimir_turn"
timeout = 10

[[hooks]]
event = "Stop"
command = "python3 -m mimir.hooks.mimir_turn"
timeout = 10
```

Both the MCP `store()` and the hook `Stop` path apply the same redaction,
filtering, and duplicate-blocking pipeline, so secrets are never persisted
regardless of how a memory is captured.

See [`docs/mcp-user-guide.md`](docs/mcp-user-guide.md) for the full hook guide.

---

## How It Works

```text
Text
  │
  ▼
[Base embedding model] ───────┐
  │                            │
  ▼                            │
Base embedding                │
  │                            │
  ▼                            │
[Prototype matrix lookup]     │
  │                            │
  ▼                            │
Sparse prototype activation   │
  │                            │
  ▼                            │
Residual modulation ◄─────────┘
  │
  ▼
Mimir embedding
```

- **Slow weights**: the frozen base embedding model gives stable semantic priors.
- **Fast weights**: a fixed-capacity prototype matrix encodes your local domain.
- **Learning**: each input activates the nearest prototypes and nudges them
  toward the new observation.
- **Prediction**: a first-order Markov transition matrix predicts the next
  prototype and emits a `surprise_score`.
- **Forgetting**: prototype strength decays exponentially; weak prototypes are
  overwritten when capacity is full.

This design is inspired by **Prototype Theory** in cognitive psychology and
**Predictive Coding** in neuroscience: memory is not a pile of events, but a
compressed set of typical examples that continuously updates itself.

---

## MCP Tools

| Tool | Purpose |
|---|---|
| `store(text, importance=1.0)` | Store and learn from text. Secrets are redacted before storage; the response `text` field is the redacted form. With async store enabled, returns `"pending"` and processes in the background. |
| `recall(query, top_k=5, min_score=0.0)` | Hybrid vector + BM25 recall, reranked by lifecycle metadata |
| `consolidate()` | Consolidate the working memory buffer |
| `forget()` | Clear the current session's working memory |
| `checkpoint(name)` | Save a named checkpoint |
| `restore(name)` | Restore to a named checkpoint |
| `status()` | Show session stats. When async store is enabled, includes `async_store` with `enabled` and `pending_count`. |

`store()` may also include `reason`, `similar_memory`, or `contradictions` in its
response when a memory is rejected or appears to contradict an existing memory.

---

## Quality Gate

`store()` runs a lightweight quality gate before learning:

- **Duplicate blocking**: near-duplicate memories (cosine similarity ≥ 0.95) are
  rejected instead of accumulating.
- **Contradiction hints**: simple negation/polarity checks flag pairs like
  "I use Python" vs "I don't use Python". The memory is still stored, but the
  result includes the hint so the agent can ask before acting on stale context.

Both checks can be disabled via `MimirConfig`:

```python
MimirConfig(quality_gate_enabled=False)
```

Or tune the thresholds:

```python
MimirConfig(
    quality_gate_enabled=True,
    quality_gate_duplicate_threshold=0.95,
    quality_gate_contradiction_threshold=0.85,
)
```

### Async store

For MCP / agent integrations where the embedding backend is slow, you can defer
`store()` to a background worker so the tool returns immediately:

```python
MimirConfig(
    async_store_enabled=True,
    async_store_queue_size=1000,
    async_store_flush_timeout=5.0,
)
```

When async storage is enabled, `store()` returns one of:

- `{"stored": "pending", "text": ..., "memory_count": ..., "pending_count": ...}`
  when the item is enqueued successfully.
- `{"stored": False, "text": ..., "memory_count": ..., "reason": "queue_full"}`
  when the queue is at capacity.

The background worker performs duplicate checks, learning, and persistence. On
MCP server shutdown the queue is flushed so pending memories are not lost.

`status()` includes an `async_store` dictionary with `enabled`, `pending_count`,
and `worker_alive` so you can monitor the queue health.

### Other useful configuration fields

| Field | Default | Purpose |
|---|---|---|
| `redaction_enabled` | `True` | Enable secret redaction |
| `redaction_patterns` | `None` | Custom regex list (`None` = defaults, `[]` = disabled) |
| `project_context_enabled` | `True` | Auto-ingest `AGENTS.md` / `CLAUDE.md` / `.cursorrules` |
| `project_context_importance` | `1.5` | Importance assigned to project context memories |
| `async_store_enabled` | `False` | Defer embedding/learning to background worker |
| `async_store_queue_size` | `1000` | Max pending items for async store |
| `async_store_flush_timeout` | `5.0` | Seconds to wait for flush on shutdown |

## Programming Interface

If you want to use Mimir inside your own Python code:

```python
from mimir.adapters.agents import InMemoryAgentAdapter, Message
from mimir.core.config import MimirConfig

adapter = InMemoryAgentAdapter(
    config=MimirConfig(base_model="all-MiniLM-L6-v2", top_k=4),
)

adapter.observe([
    Message(role="user", content="请用 Python 写快排"),
    Message(role="assistant", content="..."),
])
adapter.consolidate()
memories = adapter.recall("Python 排序", top_k=3)

print(adapter.memory_count)
adapter.clear_memories()
```

`AgentMemoryInterface` also exposes `encode(texts)` to retrieve embeddings for a
list of texts, which is useful for custom duplicate checks or integrations:

```python
embeddings = adapter.encode(["hello world", "goodbye world"])
```

See [`docs/agent-integration.md`](docs/agent-integration.md) for the adapter API.

---

## CLI

```bash
# Encode
mimir encode --backend sentence-transformer "hello world"

# Learn
mimir learn --backend llama-server "重要上下文"

# Evaluate
python -m mimir.eval --backend llama-server --top-k 4
```

---

## Status & Roadmap

Mimir is currently **v0.3.0**.

- [x] MVP: encode / learn / save / load
- [x] Top-k sparse prototype activation
- [x] EventBus + PredictionPolicy + surprise score
- [x] MCP server + Agent CLI hooks
- [x] BM25 + lifecycle hybrid recall
- [x] Multi-language small-talk filtering for automatic hook capture
- [x] Secret redaction for API keys, tokens, and passwords
- [x] Project context discovery (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`)
- [x] `mimir setup <agent>` one-shot configuration
- [x] Duplicate blocking and contradiction hints
- [x] Async embedding queue
- [ ] SQLite-backed memory metadata
- [ ] HITL preview before storing high-impact memories

See [`docs/roadmap.md`](docs/roadmap.md) for the full roadmap.

---

## Development

Mimir uses [GitHub Actions](.github/workflows/ci.yml) to run checks on every
push and PR. The CI matrix tests Python 3.10, 3.11, and 3.12, plus the OpenCode
TypeScript plugin.

Set up a local development environment:

```bash
pip install -e ".[dev,server,api]"
```

The extras are:

- `dev` — pytest, ruff, mypy
- `server` — sentence-transformers embedding backend
- `api` — OpenAI-compatible client dependencies

Run the same checks as CI:

```bash
ruff check mimir
mypy mimir
python -m pytest --tb=short
```

For the OpenCode plugin you need **Node.js 20+**:

```bash
cd plugins/opencode
npm ci
npm run typecheck
```

### Secret scanning

Some test fixtures and evaluation data contain intentionally fake secrets or
public benchmark dialogue. These files are listed in `.gitleaksignore` so
Gitleaks does not flag them. Do **not** add real credentials to those files or
to the ignore list.

Run Gitleaks locally before committing:

```bash
gitleaks detect --source . --verbose
```

If it reports a false positive in test/eval data, add the file to
`.gitleaksignore` only after confirming it contains no real credentials, and
keep the ignore list in sync with CI.

---

## Embedding Backend Performance

Mimir supports multiple embedding backends. Choose based on your hardware and latency requirements.

Measured on Apple Silicon (M-series), 128-sample batch:

| Backend | Model | Dim | Cold Start | Short Text | Long Text | Batch-128 | Throughput | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sentence-transformer | all-MiniLM-L6-v2 | 384 | ~20.6s* | 4.9ms | 6.1ms | 23ms | ~4,500/s | ~350MB |
| llama-server | Qwen3-Embedding-8B-Q4_K_M | 4096 | ~43ms | 32ms | 641ms | 10,509ms | ~12/s | ~288MB |

\* Cold start includes one-time model download/load. Subsequent runs are fast.

**Recommendation**

- Use `sentence-transformer` for local development and everyday use.
- Use `llama-server` when you need higher-quality embeddings and can accept higher latency.

Run the benchmark yourself:

```bash
python scripts/benchmark_embedding_backends.py
```

---

## License

Apache-2.0

---

*Mimir is named after the Norse guardian of the well of wisdom — a source of
knowledge that deepens with every visit.*
