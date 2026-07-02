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

#### Auto-recall and auto-learn with hooks

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
| `store(text, importance=1.0)` | Store and learn from text |
| `recall(query, top_k=5, min_score=0.0)` | Retrieve relevant memories |
| `consolidate()` | Consolidate the working memory buffer |
| `forget()` | Clear the current session's working memory |
| `checkpoint(name)` | Save a named checkpoint |
| `restore(name)` | Restore to a named checkpoint |
| `status()` | Show session stats |

---

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

Mimir is currently **v0.2.0**.

- [x] MVP: encode / learn / save / load
- [x] Top-k sparse prototype activation
- [x] EventBus + PredictionPolicy + surprise score
- [x] MCP server + Agent CLI hooks
- [ ] BM25/keyword fallback for recall
- [ ] Async embedding queue
- [ ] SQLite-backed memory metadata
- [ ] Project context discovery

See [`docs/roadmap.md`](docs/roadmap.md) for the full roadmap.

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
