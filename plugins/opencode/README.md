# Mimir OpenCode Plugin

OpenCode 插件化集成 Mimir 长期记忆。

## 功能

- `chat.message`：用户发送新消息时，以该消息为 query 调用 Mimir 召回，并把相关记忆以 `<system-reminder>` 形式注入当前用户消息上下文。
- `event`：监听 `session.next.step.ended`，获取本轮 assistant 回复，把 user/assistant exchange 写回 Mimir。

## 安装

1. 确保 Mimir Python 包已安装在当前环境，并且 `python3 -m mimir.hooks.mimir_turn` 可用。
2. 在 OpenCode 配置中加入插件（路径根据实际仓库位置调整）：

```json
{
  "plugins": [
    {
      "package": "file:///Users/isletspace/Workspace/gitlab.islet.space/engram/plugins/opencode",
      "options": {}
    }
  ]
}
```

3. 重启 OpenCode。

## 配置选项

`options` 支持覆盖 Mimir 默认参数：

| 选项 | 说明 | 默认值 |
|---|---|---|
| `python` | Python 可执行文件路径 | `python3` |
| `backend` | 嵌入后端 | `llama-server` |
| `baseUrl` | 嵌入后端地址 | `http://127.0.0.1:11435` |
| `model` | sentence-transformer 模型名 | `all-MiniLM-L6-v2` |
| `baseDir` | Mimir 工作区根目录 | `~/.mimir/workspaces` |
| `numPrototypes` | 原型数量 | `64` |
| `topK` | 推理时激活原型数 | `4` |
| `recallTopK` | 召回记忆条数 | `5` |
| `recallScoreThreshold` | 召回最低相似度 | `0.7` |

## 开发

```bash
cd plugins/opencode
npm install
npm run typecheck
```

## 注意事项

- 该插件依赖 `@opencode-ai/plugin` 的 experimental hook，后续 OpenCode 版本升级时可能需要同步调整。
- 召回内容被注入到当前用户消息中，而不是作为独立 system message（OpenCode 当前 `chat.message` hook 不支持直接注入独立 system message）。
