# Skill Distillation from Agent Tool Calls

> Architecture design for detecting repetitive tool/command patterns and
> automatically extracting reusable skills (aliases and workflows) in Mimir.

## Status

Prototype implemented. Phase 1 (observation + extraction) is functional; Phase 2
(validation / revision) and Phase 3 (interception) are future work.

## Context

AI agents often execute the same long command sequences repeatedly, e.g.

```bash
adb -s xxxxx shell 'yyyyyy'
cube programmer cli --connect --write --verify
gh pr create --title "fix: ..." --body "..." --base main --reviewer alice,bob
```

Humans naturally react to repetition with annoyance and create aliases, shell
functions, or scripts. Current agents lack this internal pressure, so they keep
paying the full cost every time. Mimir, as a plastic memory system for coding
agents, is the right place to add this capability.

## Goals

- Detect repeated tool-call patterns from agent execution traces.
- Extract reusable skills: short aliases and multi-step workflows.
- Store skills separately from episodic memory with their own lifecycle.
- Inject top skills into agent context so the agent can reuse them without
  explicit recall.
- Validate and revise skills automatically when they fail or become outdated.

## Non-Goals

- This design does not aim to block or intercept tool calls in the first phase.
  Observation and suggestion are sufficient.
- We do not attempt to derive skills from natural-language prompts alone.
  Tool-call traces are the primary signal.
- We do not replace project-level configuration files (CLAUDE.md, AGENTS.md).
  Mimir skills are agent-personal, cross-project shortcuts.

## Key Concepts

| Term | Meaning |
|---|---|
| **Working Buffer** | A short-term ring buffer of recent tool-call events used to compute repetition and frustration. |
| **Frustration Score** | A numeric pressure that accumulates when a pattern repeats. It is the internal trigger for skill extraction. |
| **Skeleton** | The longest common subsequence (LCS) of a cluster of commands. Fixed parts stay literal; variable positions become `{slot}` placeholders. |
| **Fixed Ratio** | `len(fixed_part) / len(full_command)`. A high ratio means the command is genuinely repetitive, not just sharing a common prefix. |
| **Skill Store** | A separate repository for extracted skills (aliases and workflows). It is persisted independently from the prototype memory matrix. |
| **Alias** | A literal shortcut → expansion mapping, e.g. `cuprg` → `cube programmer cli`. |
| **Workflow** | A parameterized tool-call sequence that may involve multiple steps or tools. |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent CLI (Kimi Code / etc.)                                    │
│  • PreToolUse   ─────┐                                          │
│  • PostToolUse  ─────┼──► Mimir Skill Hook                      │
│  • Stop         ─────┘                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Mimir                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐    │
│  │ Working Buffer   │  │ Skill Tracker    │  │ Skill Store  │    │
│  │ (short-term)     │──►│ (frustration +   │──►│ (aliases +   │    │
│  │                  │  │  LCS extraction) │  │  workflows)  │    │
│  └──────────────────┘  └──────────────────┘  └──────────────┘    │
│           │                                                 │   │
│           ▼                                                 ▼   │
│  ┌──────────────────┐                              ┌──────────┐ │
│  │ Prototype Matrix │                              │ Context  │ │
│  │ (existing memory)│                              │ injection│ │
│  └──────────────────┘                              └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Hook Integration

Kimi Code exposes `PreToolUse`, `PostToolUse`, and `PostToolUseFailure` events.

### Phase 1: Observation only (recommended)

```toml
# ~/.kimi-code/config.toml
[[hooks]]
event = "PostToolUse"
command = "python3 -m mimir.hooks.skill_observer"
timeout = 5
```

`skill_observer` receives a JSON payload:

```json
{
  "hook_event_name": "PostToolUse",
  "session_id": "abc123",
  "cwd": "/path/to/project",
  "tool_name": "Shell",
  "tool_input": {"command": "adb -s xxxxx shell 'yyyyyy'"},
  "tool_output": "..."
}
```

It records the event into the Working Buffer, updates the Skill Tracker, and
optionally schedules an asynchronous extraction.

### Phase 2: Optional pre-check

`PreToolUse` can later be used to check whether a high-confidence skill exists
before the tool runs. However, blocking a tool call and replacing its output
is limited by the agent CLI's hook protocol. We recommend starting with
post-hoc observation and context injection, then evaluating whether interception
is worth the complexity.

## Frustration Score

The score measures how much a command pattern "deserves" to be compressed into
a skill.

```text
frustration = max(0, repeat_count - 5) × len(skeleton_fixed_part) × fixed_ratio
```

- `repeat_count`: number of occurrences in the sliding window.
- `5`: cold-start guard. Patterns must repeat at least 6 times before scoring.
- `len(skeleton_fixed_part)`: length of the fixed literal part of the skeleton.
- `fixed_ratio`: `fixed_length / average_command_length`, capped at 1.0.

Why this formula:

- Long, stable commands produce higher frustration than short or noisy ones.
- A high fixed ratio means the pattern is genuinely predictable, not just
  sharing a common prefix.
- The 5-repetition guard prevents premature extraction of transient patterns.

Pipeline commands (`cat a | grep b | sort`) are treated as single commands. The
`|` characters are part of the skeleton and contribute to its length.

## Skeleton Extraction

Given a cluster of similar command strings, we compute the LCS across all pairs
and reconstruct a skeleton with placeholder slots.

Example:

```text
adb -s ABC123 shell 'reboot bootloader'
adb -s DEF456 shell 'reboot bootloader'
adb -s GHI789 shell 'reboot bootloader'
```

Skeleton:

```text
adb -s {device_id} shell 'reboot bootloader'
```

Variable detection heuristics (in order of reliability):

1. Hex / UUID / hash-like tokens.
2. Absolute or relative paths.
3. Numbers.
4. Strings that differ across executions while surrounding text is identical.

For the first prototype, literal LCS plus simple regex heuristics is enough.
Semantic classification can be added later if needed.

## Skill Store

Skills are stored separately from the prototype memory matrix because they have
very different lifecycles:

| Aspect | Memory | Skill |
|---|---|---|
| Content | Past interactions | Executable templates |
| Update | Continuous, Hebbian | Revision-based, versioned |
| Recall | Passive, similarity | Active, pattern match |
| Size | Dense matrix | Sparse, structured list |

Proposed skill record:

```python
@dataclass
class Skill:
    id: str
    type: Literal["alias", "workflow"]
    name: str
    trigger_pattern: str  # literal template for both alias and workflow
    expansion: str | None  # alias only
    template: str | None  # workflow only: prompt or tool-call chain
    required_context: list[str]
    confidence: float
    usage_count: int
    failure_count: int
    version: int
    deprecated: bool
    created_at: str  # ISO 8601
    last_used: str  # ISO 8601
```

Persistence:

- `skills.jsonl` or `skills.pt` next to the existing checkpoint.
- Loaded on Mimir startup and saved after every extraction or revision.

## Context Injection

Only the top-N skills are injected into the agent context automatically. The
rest remain in the Skill Store and can be queried explicitly.

Selection criteria:

```text
score = frustration × confidence × recency_weight
```

Default N = 10. This can be made adaptive based on context window size.

Injected format (example):

```markdown
## Your active shortcuts
- `cuprg` = `cube programmer cli`
- `adb-reboot` = `adb -s {device_id} shell 'reboot bootloader'`
- `py-lint` = `ruff check . && mypy mimir`
```

## Skill Lifecycle

```
observe → extract (draft) → validate → promote → use → revise / deprecate
```

### Extract

Triggered when `frustration > threshold`. Produces a draft skill with
confidence computed from the cluster size and fixed ratio.

### Validate

For a configurable fraction of executions, run both the skill shortcut and the
full reasoning path and compare outputs. If they diverge, confidence drops.

### Promote

A draft becomes active when confidence exceeds `skill_min_confidence` (default
0.85) or after a minimum number of successful uses.

### Revise

When a skill repeatedly fails, diagnose the failure type:

- Over-generalized: too many parts became variables. Tighten or split.
- Under-generalized: needed variables were fixed. Add slots.
- Context mismatch: missing preconditions. Add guards.

Deprecated skills are kept for history, not deleted.

### Deprecate

A skill is deprecated when confidence stays below a threshold for too long, or
when the user explicitly rejects it.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Over-extraction | 5-repetition guard, fixed-ratio threshold, confidence-based promotion. |
| Wrong variable generalization | Validate with dual-track execution; keep versioning. |
| Context pollution | Inject only top-N skills; allow explicit recall for others. |
| Confusing skill with user preference | Separate user-defined aliases from auto-extracted workflows. |
| Agent CLI hook limitations | Start with PostToolUse observation; intercept only after validation. |
| Sensitive data in commands | Reuse existing Mimir redactor before storing command strings. Implemented in `skill_observer`. |
| Prompt injection via skill output | Escape backticks in injected skill names/expansions. Implemented in `SkillInjector`. |

## Implementation Phases

### Phase 1: Minimum viable prototype (this PR)

- `mimir/hooks/skill_observer.py`: receive `PostToolUse`, append to Working Buffer.
- `mimir/skills/tracker.py`: compute frustration and trigger extraction.
- `mimir/skills/extractor.py`: LCS-based skeleton extraction.
- `mimir/skills/store.py`: JSONL persistence.
- `mimir/skills/injector.py`: inject top-N skills into `UserPromptSubmit` context.
- Tests for the tracker and extractor on synthetic command sequences.

### Phase 2: Validation and revision

- Dual-track validation for workflows.
- Automatic revision heuristics.
- Confidence decay and deprecation.

### Phase 3: Interception

- `PreToolUse` pre-check and optional shortcut execution.
- Requires deeper integration with the agent CLI protocol.

## Configuration

The skill subsystem has its own configuration classes. These fields are **not**
part of `MimirConfig`; they live in `SkillTrackerConfig` and `InjectorConfig`.

```python
@dataclass
class SkillTrackerConfig:
    window_size: int = 50
    min_repetitions: int = 5
    frustration_threshold: float = 50.0
    min_fixed_ratio: float = 0.6

@dataclass
class InjectorConfig:
    max_active: int = 10
    min_confidence: float = 0.85
```

Future phases may wire these into `MimirConfig` once the lifecycle and
validation mechanisms are stable.

## Open Questions

1. Should user-defined aliases be treated as immutable skills with higher priority
   than auto-extracted workflows?
2. How do we handle multi-tool workflows where the repetition spans several
   `PostToolUse` events rather than a single command?
3. Should skills be scoped per project, or always global/cross-project?

*Last updated: 2026-07-03*
