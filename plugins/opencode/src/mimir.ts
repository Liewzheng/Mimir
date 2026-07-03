import { spawnSync } from "node:child_process"

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

function runMimir(args: string[], payload: Record<string, unknown>): unknown {
  const result = spawnSync(args[0], args.slice(1), {
    input: JSON.stringify(payload),
    encoding: "utf-8",
    timeout: 15000,
  })

  if (result.error) {
    console.error("[Mimir] failed to invoke:", result.error)
    return undefined
  }
  if (result.status !== 0) {
    console.error("[Mimir] hook exited with code", result.status, result.stderr)
    return undefined
  }

  const stdout = result.stdout.trim()
  if (!stdout) return undefined
  try {
    return JSON.parse(stdout)
  } catch (cause) {
    console.error("[Mimir] failed to parse hook output:", stdout, cause)
    return undefined
  }
}

export function recall(
  query: string,
  workspacePath: string,
  options: MimirOptions = {},
): string | undefined {
  const args = buildArgs(options, workspacePath)
  const payload = {
    hook_event_name: "UserPromptSubmit",
    prompt: query,
  }
  const result = runMimir(args, payload)
  if (!result || typeof result !== "object") return undefined
  const recallText = (result as { recall?: string | null }).recall
  return recallText && recallText.trim() ? recallText : undefined
}

export function observe(
  exchange: { role: "user" | "assistant"; content: string }[],
  workspacePath: string,
  options: MimirOptions = {},
): void {
  const args = buildArgs(options, workspacePath)
  const payload = {
    hook_event_name: "Stop",
    messages: exchange,
  }
  const result = runMimir(args, payload)
  if (!result || typeof result !== "object" || (result as { status?: string }).status !== "ok") {
    console.error("[Mimir] observe did not return ok:", result)
  }
}
