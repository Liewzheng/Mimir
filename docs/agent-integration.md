# Agent CLI 集成指南

Mimir 可以作为 agent CLI（如 opencode、kimi code、claude code、codex 等）的外部记忆层。本模块提供与具体 Mimir 实现解耦的通用抽象接口。

---

## 核心抽象

```python
from mimir.adapters.agents import (
    AgentMemoryInterface,
    InMemoryAgentAdapter,
    Message,
)
from mimir.core.config import MimirConfig

adapter = InMemoryAgentAdapter(
    config=MimirConfig(base_model="all-MiniLM-L6-v2"),
)
```

### Message

```python
Message(role="user", content="你好", metadata={"turn": 1})
```

- `role`: `"user"`, `"assistant"`, `"system"`, `"tool"` 等。
- `content`: 消息文本。
- `metadata`: 任意附加字段。
- `timestamp`: UTC 时间戳，默认当前时间。

### Memory

```python
Memory(
    text="检索到的记忆文本",
    embedding=[0.1, 0.2, ...],  # list[float]，后端无关
    score=0.85,
    created_at=datetime.now(timezone.utc),
    source=Message(...),
    metadata={"role": "assistant"},
)
```

- `text`: 记忆文本内容。
- `embedding`: 记忆的嵌入向量，以 `list[float]` 形式返回。
- `score`: 混合相关度分数（0~1）。默认融合向量 cosine similarity 与 BM25 关键词匹配，并用生命周期元数据重排；可通过 `use_bm25` 和 `use_lifecycle` 关闭。
- `created_at`: 记忆创建时间。
- `source`: 来源 `Message`（可选）。
- `metadata`: 附加字段。

所有 agent 后端都实现以下方法：

| 方法 | 说明 |
|---|---|
| `observe(messages: list[Message])` | 观察一组消息，编码并写入工作记忆。 |
| `recall(query: str, top_k=5, min_score=0.0, use_bm25=True, use_lifecycle=True, lifecycle_weight=0.3)` | 混合向量 + BM25 检索，并可用生命周期元数据（recency、importance、access）重排。 |
| `consolidate()` | 主动巩固当前工作记忆中的所有内容。 |
| `checkpoint(path)` | 持久化 Mimir 状态。 |
| `restore(path)` | 从检查点恢复 Mimir 状态；工作记忆会被清空。 |
| `reset()` | 清空工作记忆并重置 step。 |

---

## 与 opencode / kimi code / claude code / codex 集成

这些 agent CLI 通常在每次用户输入后维护一个对话历史。集成方式如下：

```python
# 在 agent 初始化时创建 adapter
adapter = InMemoryAgentAdapter(
    config=MimirConfig(
        base_model="all-MiniLM-L6-v2",
        num_prototypes=64,
        top_k=4,
    ),
    learn_on_observe=False,  # 默认：显式 consolidate() 才学习
)

# 每次对话轮次结束后
adapter.observe([
    Message(role="user", content=user_input),
    Message(role="assistant", content=assistant_output),
])

# 周期性地巩固记忆（例如每 N 轮或会话结束时）
adapter.consolidate()

# 构造 prompt 前，检索相关记忆
memories = adapter.recall(user_input, top_k=3)
context = "\n".join(f"- {m.text}" for m in memories)
```

### 学习模式

| `learn_on_observe` | 行为 |
|---|---|
| `False`（默认） | `observe()` 只编码和存储；调用 `consolidate()` 时才学习。避免重复强化。 |
| `True` | `observe()` 时立即学习；`consolidate()` 会再次强化全部记忆。适合实时学习场景。 |

### 推荐接入点

| Agent CLI | 接入点 |
|---|---|
| opencode | 在 `system prompt` 组装前调用 `recall()`，在每次用户/助手交互后调用 `observe()`。 |
| kimi code | 通过工具/函数调用封装 `observe` 和 `recall`。 |
| claude code | 在 `MessageManager` 或类似组件中注入 adapter。 |
| codex | 在 `Codex` 的会话循环中使用 adapter 管理长期记忆。 |

---

## 持久化

```python
# checkpoint_dir 默认为 ~/.mimir/checkpoints，会自动创建
# path 必须是相对于 checkpoint_dir 的相对路径
adapter.checkpoint("my-agent-session.pt")
adapter.restore("my-agent-session.pt")  # 恢复后工作记忆会被清空
```

检查点只保存 Mimir 的 prototype 矩阵和预测策略状态。工作记忆在 `restore()` 后会被清空，这是为了避免恢复旧会话的临时上下文。

---

## 设计原则

1. **后端无关**：`AgentMemoryInterface` 不暴露 `Mimir` 类型，agent 代码可以不依赖具体实现。
2. **显式控制**：默认 `learn_on_observe=False`，学习由 `consolidate()` 显式触发；也可开启实时学习。
3. **容量可控**：通过 `max_memories` 限制工作记忆大小，最旧的记忆会被淘汰。
4. **可测试**：`InMemoryAgentAdapter` 可注入 `FakeEngine` 进行单元测试。

## 已知限制

- **P1 阶段 `recall()` 为线性扫描**：`InMemoryAgentAdapter` 每次 recall 会对全部工作记忆做 cosine similarity，时间复杂度为 O(n × d)。默认 `max_memories=10_000` 在普通对话场景足够；高频大容量场景请在 P2 接入 FAISS 等向量索引。
- `checkpoint()` 只保存 Mimir 状态，不保存工作记忆。`restore()` 后需重新观察当前会话上下文。

---

*创建于 2026-06-27 | 对应 P1 agent 集成*
