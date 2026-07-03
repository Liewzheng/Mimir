import type { PluginInput } from "@opencode-ai/plugin"
import type { Part } from "@opencode-ai/sdk"

import { observe as mimirObserve } from "./mimir.js"

export type UserCache = Map<
  string,
  { messageID: string; text: string }
>

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

  mimirObserve(
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
