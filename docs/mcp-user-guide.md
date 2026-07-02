# Mimir MCP 使用手册

> 本手册面向最终用户，介绍如何在常见 Agent CLI 中安装、配置和使用 Mimir 记忆系统。

---

## 目录

1. [Mimir 是什么](#mimir-是什么)
2. [安装](#安装)
3. [启动本地嵌入后端](#启动本地嵌入后端)
4. [与 Agent CLI 集成](#与-agent-cli-集成)
   - [OpenCode](#opencode)
   - [Kimi Code](#kimi-code)
   - [Claude Code](#claude-code)
   - [Codex](#codex)
5. [自动记忆的两种方式](#自动记忆的两种方式)
6. [可用 MCP 工具](#可用-mcp-工具)
7. [工作空间与持久化](#工作空间与持久化)
8. [常见问题](#常见问题)

---

## Mimir 是什么

Mimir 是一个**可学习的本地记忆系统**，通过 MCP（Model Context Protocol）接入 agent CLI。它会：

- **存储**对话中的重要事实、偏好、决策。
- **检索**与当前话题相关的历史记忆。
- **学习**用户的领域表达，让检索越来越准。
- **持久化**到本地磁盘，重启后记忆不丢失。

所有数据默认保存在 `~/.mimir/workspaces/<workspace-hash>/`。

---

## 安装

```bash
pip install mimir-core
```

如果你使用 Qwen3-Embedding 等本地模型，只需要核心包即可。如果没有本地 embedding 服务，安装可选依赖：

```bash
pip install "mimir-core[server]"   # 包含 sentence-transformers
```

---

## 启动本地嵌入后端

Mimir 支持三种 embedding 后端：

| 后端 | 命令行参数 | 说明 |
|---|---|---|
| `llama-server` | `--backend llama-server` | 推荐，本地 GPU/CPU 推理，延迟最低 |
| `sentence-transformer` | `--backend sentence-transformer` | 零额外进程，首次自动下载模型 |
| `fake` | `--backend fake` | 仅用于测试，输出固定维度随机向量 |

推荐使用 llama-server 运行本地 embedding 模型：

```bash
llama-server \
  --model /path/to/Qwen3-Embedding-8B-Q4_K_M.gguf \
  --embeddings \
  --port 11435
```

MCP 默认连接 `http://127.0.0.1:11435`。启动后验证：

```bash
curl -s http://127.0.0.1:11435/embedding \
  -X POST -H "Content-Type: application/json" \
  -d '{"content":"hello"}' | head -c 100
```

如果没有 llama-server，可以直接用 sentence-transformer：

```bash
mimir mcp --backend sentence-transformer
```

---

## 与 Agent CLI 集成

### OpenCode

配置文件：`~/.config/opencode/opencode.json` 或项目 `.opencode/opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mimir": {
      "type": "local",
      "command": ["mimir", "mcp", "--backend", "sentence-transformer"]
    }
  },
  "instructions": [
    "You are connected to the Mimir memory system (MCP server 'mimir').",
    "Before each reply, call the recall tool to retrieve relevant memories.",
    "After each turn, if the conversation contains durable facts (preferences, decisions, key paths, recurring issues), call the store tool with a concise text."
  ]
}
```

注意：不同客户端对 MCP 工具名的前缀规则不同。OpenCode 通常会把 `store` 暴露为 `mimir_store` 或 `mcp_mimir_store`，请以 `opencode mcp list` 的输出为准。

验证：

```bash
opencode mcp list
opencode run "记住我喜欢用中文交流"
```

### Kimi Code

Kimi Code 同时支持 hook 和 Skill。推荐用 hook 做自动 recall 与自动 consolidate，Skill 负责判断"什么值得记"并调用 store 工具。

> **Kimi Code 限制**：早期版本的 `Stop` hook 事件不携带本轮对话内容；最新版本已支持 `messages` 数组，Mimir 可以自动保存完整 user/assistant 轮次。系统提示（`<system-reminder>`）和 hook 输出会被过滤，不会存入记忆。
>
> 可以通过 `--recall-score-threshold` 调整召回门槛（默认 0.7）：
>
> ```toml
> command = "python3 -m mimir.hooks.mimir_turn --recall-score-threshold 0.8"
> ```

#### 1. 配置 MCP Server

配置文件：`~/.kimi-code/mcp.json`

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

#### 2. 配置 Hook（推荐）

在 `~/.kimi-code/config.toml` 中加入：

```toml
# 每轮开始前：从 Mimir 召回相关记忆，自动注入上下文
[[hooks]]
event = "UserPromptSubmit"
command = "python3 -m mimir.hooks.mimir_turn"
timeout = 10

# 每轮结束时：把本轮对话学进 Mimir，强化预测
[[hooks]]
event = "Stop"
command = "python3 -m mimir.hooks.mimir_turn"
timeout = 10

# 会话启动时：简要提示已加载多少记忆
[[hooks]]
event = "SessionStart"
command = "python3 -m mimir.hooks.mimir_turn"
timeout = 10
```

如果你的 embedding 后端不是默认的 sentence-transformer，在 `command` 里补上参数：

```toml
command = "python3 -m mimir.hooks.mimir_turn --backend llama-server --base-url http://127.0.0.1:11435"
```

#### 3. Skill Fallback（可选）

如果 kimi-code 的版本不支持 hook，或你希望模型额外决定"什么值得记"，可保留 Skill：

创建目录和文件：`~/.kimi-code/skills/mimir-memory/SKILL.md`

```markdown
---
name: mimir-memory
description: Before finishing your turn, decide if anything is worth remembering (preferences, decisions, recurring issues, explicit "remember" commands) and call the Mimir store tool.
type: prompt
whenToUse: When the user provides information that should be persisted across sessions.
disableModelInvocation: false
---

You are connected to the Mimir memory system (MCP server `mimir`).

1. If the user explicitly asks to remember something, or if the turn contains durable facts (preferences, decisions, key paths), call the store tool with a concise Chinese `text`.
2. Do NOT call the recall tool; the hook already injects relevant memories at the start of each turn.

Tool names may appear as `mcp__mimir__store`, `mcp__mimir__consolidate`, `mcp__mimir__status`, depending on the client.
```

> 注意：Skill 中不再要求调用 recall，否则会和 hook 注入的记忆重复。

验证：

```bash
kimi doctor
kimi -p "记住我喜欢用中文交流"
kimi -p "告诉我你记住了什么"
```

### Claude Code

Claude Code 支持 `Stop` hook。结合系统提示使用：

配置文件：`.claude/settings.json` 或 `~/.claude/settings.json`

```json
{
  "mcpServers": {
    "mimir": {
      "command": "mimir",
      "args": ["mcp", "--backend", "sentence-transformer"]
    }
  },
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          { "type": "command", "command": "python3 -m mimir.hooks.mimir_turn" }
        ]
      }
    ]
  }
}
```

同时可在 `CLAUDE.md` / `AGENTS.md` 中写入：

> 每轮对话结束时，Mimir 已通过 hook 自动保存了本轮内容。你可以直接回复，无需额外调用 store。

Claude Code 的 `UserPromptSubmit` hook 不在公开文档中；如果需要每轮自动 recall，建议在系统提示中要求模型主动调用 recall 工具，或依赖 agent 自身的上下文。

### Codex

Codex 的 hook 系统与 Claude Code 类似。配置文件：`~/.codex/config.toml` 或项目 `.codex/config.toml`

```toml
[mcp_servers.mimir]
command = "mimir"
args = ["mcp", "--backend", "sentence-transformer"]

[hooks.Stop]
[[hooks.Stop.hooks]]
type = "command"
command = "python3 -m mimir.hooks.mimir_turn"
timeout = 10
```

Codex 目前没有 `UserPromptSubmit` hook，因此自动 recall 需要依赖系统提示或模型主动调用 MCP 工具。

---

## 自动记忆的两种方式

| 方式 | 说明 | 适用工具 |
|------|------|----------|
| **Hook（推荐）** | 在 agent 的 `UserPromptSubmit` / `Stop` / `SessionStart` 事件中调用 `mimir.hooks.mimir_turn`，确定性自动 recall 和学习。 | Kimi Code、Claude Code、Codex |
| **Skill / instructions** | 通过系统提示让模型自主调用 store/recall。当 agent 不支持 hook 时使用。 | OpenCode、Kimi Code、Aider、Continue.dev |

推荐组合：

- **Kimi Code**：Hook 为主（自动 recall + 自动学习），Skill 只负责"判断什么值得记"。
- **OpenCode**：用 `instructions` 驱动模型主动调用工具。
- **Claude Code / Codex**：用 `Stop` hook 自动学习，系统提示补充 recall。

---

## 可用 MCP 工具

Mimir MCP server 注册的工具名如下。不同 Agent CLI 可能会自动加上前缀（例如 Kimi Code 可能显示为 `mcp__mimir__store`，OpenCode 可能显示为 `mimir_store`），请以 `mcp list` 的实际输出为准。

| 工具 | 参数 | 说明 |
|------|------|------|
| `store` | `text: string`, `importance: float = 1.0` | 存储并学习一条文本。 |
| `recall` | `query: string`, `top_k: int = 5`, `min_score: float = 0.0` | 检索相关记忆。内部使用向量 + BM25 混合打分，并按生命周期元数据（recency、importance、access）重排。 |
| `consolidate` | 无 | 巩固工作记忆中的所有内容。 |
| `forget` | 无 | 清空当前会话记忆。 |
| `checkpoint` | `name: string` | 保存命名检查点。 |
| `restore` | `name: string` | 恢复到命名检查点。 |
| `status` | 无 | 查看会话统计。 |
| `summarize_memories` | 无 | 让 LLM 整理当前工作记忆并返回摘要。 |
| `replace_memories` | `memories: list[str]` | 用整理后的记忆全量替换当前工作记忆。 |

---

## 工作空间与持久化

Mimir 按**工作空间**隔离记忆。工作空间由当前目录的 git root 或绝对路径决定。

```text
~/.mimir/workspaces/<hash>/
├── checkpoints/
│   └── session          # Mimir 学习状态
├── memories.json        # 工作记忆
└── checkpoints/         # 命名检查点（与 session 同目录）
```

- 同一项目在不同目录启动，会复用同一个工作空间。
- 不同项目互不影响。
- MCP server 退出时会自动保存。

---

## 常见问题

### Q1: MCP server 启动失败，提示连接不上 embedding 后端

如果使用 llama-server，确保已启动：

```bash
curl http://127.0.0.1:11435/embedding -X POST \
  -H "Content-Type: application/json" \
  -d '{"content":"test"}'
```

如果使用 sentence-transformer，启动时加上 `--backend sentence-transformer`。

### Q2: 切换 embedding 后端后报维度错误

不同后端的输出维度不同。清空工作空间后重新启动：

```bash
rm -rf ~/.mimir/workspaces/<hash>
```

### Q3: 模型不主动调用 Mimir 工具

- 检查 instructions / Skill 是否正确加载。
- 确保 agent 处于 `auto` 或 `yolo` 权限模式，或者手动批准工具调用。
- 在 Skill 的 `whenToUse` 中写明 `ALWAYS` 可提高触发率。
- 不同客户端的工具名前缀不同，请以 `mcp list` 输出为准。

### Q4: 如何备份或迁移记忆

复制整个工作空间目录即可：

```bash
cp -r ~/.mimir/workspaces/<hash> ~/.mimir/workspaces/<hash>.backup
```

---

*对应 Mimir 版本：v0.2.0 及以上*
