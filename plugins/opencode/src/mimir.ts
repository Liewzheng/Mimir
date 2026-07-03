import { spawn } from "node:child_process"

export type MimirOptions = {
  python?: string
  backend?: string
  baseUrl?: string
  model?: string
  baseDir?: string
  numPrototypes?: number
  topK?: number
  recallTopK?: number
  recallScoreThreshold?: number
}

const DEFAULT_OPTIONS: Required<Omit<MimirOptions, "baseDir">> = {
  python: "python3",
  backend: "llama-server",
  baseUrl: "http://127.0.0.1:11435",
  model: "all-MiniLM-L6-v2",
  numPrototypes: 64,
  topK: 4,
  recallTopK: 5,
  recallScoreThreshold: 0.7,
}

function buildArgs(options: MimirOptions, workspacePath: string): string[] {
  const python = options.python ?? DEFAULT_OPTIONS.python
  const args = [python, "-m", "mimir.hooks.mimir_turn", "--format", "json"]
  args.push("--backend", options.backend ?? DEFAULT_OPTIONS.backend)
  args.push("--base-url", options.baseUrl ?? DEFAULT_OPTIONS.baseUrl)
  args.push("--model", options.model ?? DEFAULT_OPTIONS.model)
  args.push("--num-prototypes", String(options.numPrototypes ?? DEFAULT_OPTIONS.numPrototypes))
  args.push("--top-k", String(options.topK ?? DEFAULT_OPTIONS.topK))
  args.push("--recall-top-k", String(options.recallTopK ?? DEFAULT_OPTIONS.recallTopK))
  args.push("--recall-score-threshold", String(options.recallScoreThreshold ?? DEFAULT_OPTIONS.recallScoreThreshold))
  if (options.baseDir) {
    args.push("--base-dir", options.baseDir)
  }
  args.push("--workspace-path", workspacePath)
  return args
}

function runMimir(args: string[], payload: Record<string, unknown>): Promise<unknown> {
  return new Promise((resolve) => {
    const child = spawn(args[0], args.slice(1), {
      stdio: ["pipe", "pipe", "pipe"],
    })
    let stdout = ""
    let stderr = ""
    let timeoutId: ReturnType<typeof setTimeout> | undefined

    let finished = false
    const finish = (result: unknown) => {
      if (finished) return
      finished = true
      if (timeoutId) clearTimeout(timeoutId)
      resolve(result)
    }

    timeoutId = setTimeout(() => {
      child.kill("SIGTERM")
      finish(undefined)
    }, 15000)

    child.stdin.write(JSON.stringify(payload), "utf-8", (err) => {
      if (err) {
        console.error("[Mimir] failed to write payload:", err)
        finish(undefined)
        return
      }
      child.stdin.end()
    })

    child.stdout.on("data", (data) => {
      stdout += data
    })
    child.stderr.on("data", (data) => {
      stderr += data
    })

    child.on("error", (error) => {
      console.error("[Mimir] failed to invoke:", error)
      finish(undefined)
    })

    child.on("close", (code) => {
      if (code !== 0) {
        console.error("[Mimir] hook exited with code", code, stderr)
        finish(undefined)
        return
      }
      const out = stdout.trim()
      if (!out) {
        finish(undefined)
        return
      }
      try {
        finish(JSON.parse(out))
      } catch (cause) {
        console.error("[Mimir] failed to parse hook output:", out, cause)
        finish(undefined)
      }
    })
  })
}

export async function recall(
  query: string,
  workspacePath: string,
  options: MimirOptions = {},
): Promise<string | undefined> {
  const args = buildArgs(options, workspacePath)
  const payload = {
    hook_event_name: "UserPromptSubmit",
    prompt: query,
  }
  const result = await runMimir(args, payload)
  if (!result || typeof result !== "object") return undefined
  const recallText = (result as { recall?: string | null }).recall
  return recallText && recallText.trim() ? recallText : undefined
}

export async function observe(
  exchange: { role: "user" | "assistant"; content: string }[],
  workspacePath: string,
  options: MimirOptions = {},
): Promise<void> {
  const args = buildArgs(options, workspacePath)
  const payload = {
    hook_event_name: "Stop",
    messages: exchange,
  }
  const result = await runMimir(args, payload)
  if (!result || typeof result !== "object" || (result as { status?: string }).status !== "ok") {
    console.error("[Mimir] observe did not return ok:", result)
  }
}
