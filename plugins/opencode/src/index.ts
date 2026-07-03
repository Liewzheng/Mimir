import type { Plugin } from "@opencode-ai/plugin"

import { observe } from "./observe.js"
import { recall } from "./recall.js"

const MimirPlugin: Plugin = async (input, options = {}) => {
  const userCache = new Map<string, { messageID: string; text: string }>()

  return {
    "chat.message": async (hookInput, hookOutput) => {
      const result = await recall(
        hookInput,
        hookOutput,
        input.directory,
        options,
      )
      if (result) {
        userCache.set(hookInput.sessionID, {
          messageID: hookInput.messageID ?? hookOutput.message.id,
          text: result.query,
        })
      }
    },
    event: async ({ event }) => {
      await observe(
        input.client,
        event as { type: string; properties: Record<string, unknown> },
        userCache,
        input.directory,
        options,
      )
    },
  }
}

export default { server: MimirPlugin, id: "mimir-opencode-plugin" }
