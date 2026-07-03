import type { Plugin } from "@opencode-ai/plugin"

import { observe } from "./observe.js"
import { recall } from "./recall.js"

const MimirPlugin: Plugin = async (input, options = {}) => {
  const userCache = new Map<string, { messageID: string; text: string }>()

  return {
    "chat.message": async (hookInput, hookOutput) => {
      try {
        const result = await recall(
          hookInput,
          hookOutput,
          input.directory,
          options,
        )
        // `result` is undefined only when the user message has no text.
        // When present, cache the original query so the turn-end observer can
        // pair it with the assistant response and store the exchange.
        if (result) {
          userCache.set(hookInput.sessionID, {
            messageID: hookInput.messageID ?? hookOutput.message.id,
            text: result.query,
          })
        }
      } catch (error) {
        console.error("[Mimir] recall failed:", error)
      }
    },
    event: async ({ event }) => {
      try {
        await observe(
          input.client,
          event as { type: string; properties: Record<string, unknown> },
          userCache,
          input.directory,
          options,
        )
      } catch (error) {
        console.error("[Mimir] observe failed:", error)
      }
    },
  }
}

export default { server: MimirPlugin, id: "mimir-opencode-plugin" }
