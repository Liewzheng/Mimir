import type { PluginInput } from "@opencode-ai/plugin"
import type { Part } from "@opencode-ai/sdk"

import { observe as mimirObserve } from "./mimir.js"

/**
 * Cache of user message texts keyed by session ID.
 *
 * The plugin observes turn-end events and needs the original user text paired
 * with the assistant response to store the exchange as a single memory.
 */
export type UserCache = Map<
  string,
  { messageID: string; text: string }
>

/** Concatenate all text parts in a message into a single trimmed string. */
function extractText(parts: Part[]): string {
  return parts
    .filter(
      (p): p is Part & { type: "text"; text: string } =>
        p.type === "text" && "text" in p && typeof p.text === "string",
    )
    .map((p) => p.text)
    .join("\n")
    .trim()
}

/**
 * Observe a completed assistant turn and store the user/assistant exchange
 * as a memory in Mimir.
 *
 * @param client OpenCode plugin client used to fetch the assistant message.
 * @param event The turn-end event from OpenCode.
 * @param userCache Cache holding the user text for the current session.
 * @param workspacePath Path to the current workspace, used for session isolation.
 * @param options Plugin configuration forwarded to the Mimir CLI.
 */
export async function observe(
  client: PluginInput["client"],
  event: { type: string; properties: Record<string, unknown> },
  userCache: UserCache,
  workspacePath: string,
  options: Record<string, unknown>,
): Promise<void> {
  if (event.type !== "session.next.step.ended") return

  const properties = event.properties as {
    sessionID: string
    assistantMessageID: string
  }
  const { sessionID, assistantMessageID } = properties
  if (typeof sessionID !== "string" || typeof assistantMessageID !== "string") {
    console.error("[Mimir] session.next.step.ended event missing expected properties")
    return
  }

  const cached = userCache.get(sessionID)
  if (!cached) return

  let assistantText = ""
  try {
    const response = await client.session.message({
      path: { id: sessionID, messageID: assistantMessageID },
    })
    assistantText = extractText(response.data?.parts ?? [])
  } catch (error) {
    console.error("[Mimir] failed to fetch assistant message:", error)
    return
  }

  if (!assistantText) return

  await mimirObserve(
    [
      { role: "user", content: cached.text },
      { role: "assistant", content: assistantText },
    ],
    workspacePath,
    {
      python: options.python as string | undefined,
      backend: options.backend as string | undefined,
      baseUrl: options.baseUrl as string | undefined,
      model: options.model as string | undefined,
      baseDir: options.baseDir as string | undefined,
      numPrototypes: options.numPrototypes as number | undefined,
      topK: options.topK as number | undefined,
      recallTopK: options.recallTopK as number | undefined,
      recallScoreThreshold: options.recallScoreThreshold as number | undefined,
    },
  )
}
