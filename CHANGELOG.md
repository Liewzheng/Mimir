# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-07-03

### Added

- GitHub Actions CI now caches pip dependencies and enforces `timeout-minutes` on
  all jobs.
- Gitleaks secret-scanning job added to CI.
- README now includes a Development section with local setup, check commands,
  extras explanations, and secret-scanning notes.

### Changed

- CI Node.js version is now defined by a workflow-level `NODE_VERSION`
  environment variable.

### Fixed

- Replaced `math.exp2` with `2 ** x` in lifecycle scorer to restore Python 3.10
  compatibility (`math.exp2` was added in Python 3.11).

## [0.3.0] - 2026-07-03

### Added

- **Async embedding queue**: `store()` can be configured to return immediately
  (`{"stored": "pending"}`) while a background worker handles embedding,
  duplicate checks, learning, and persistence. Configurable via
  `MimirConfig(async_store_enabled=True)`.
- **Secret redaction**: automatic masking of API keys, GitHub tokens, JWTs,
  passwords, AWS credentials, and `Authorization` headers before memories are
  stored.
- **Project Context Discovery**: `AGENTS.md`, `CLAUDE.md`, and `.cursorrules`
  files found in the workspace root are automatically ingested as high-importance
  memories when a session starts. This can be disabled with
  `MimirConfig(project_context_enabled=False)`.
- **Quality gate**: `store()` now blocks near-duplicate memories
  (embedding cosine similarity ≥ 0.95) and returns simple contradiction hints
  when a new memory appears to contradict an existing one. Thresholds and
  enablement are configurable via `MimirConfig`.
- **`mimir setup <agent>` CLI**: one-shot configuration of Mimir hooks for
  Kimi Code, Claude Code, and Codex. Use `--base-dir` to override the default
  agent configuration directory.
- **Agent adapter `encode()`**: public method to retrieve embeddings for a list
  of texts, usable for custom duplicate checks or integrations.
- **New `MimirConfig` fields**: `quality_gate_enabled`,
  `quality_gate_duplicate_threshold`, `quality_gate_contradiction_threshold`,
  `project_context_enabled`, `project_context_importance`,
  `redaction_patterns` (custom regex list; `None` = defaults, `[]` = disabled), and
  `async_store_enabled` / `async_store_queue_size` /
  `async_store_flush_timeout` for non-blocking `store()`.

### Changed

- `store()` returns the redacted form of the text in the `text` field instead of
  the original raw input, so callers never receive secrets back. Rejection
  responses (filter/duplicate) also return the redacted form.
- `SessionManager` now reuses the filtering / redaction / learning pipeline for
  both project context ingestion and explicit `store()` calls.
- Project context importance default is now `1.5` (previously hard-coded `2.0`).
- Minimum dependency versions raised: `httpx >= 0.27.2` (was `>= 0.25.0`) and
  `sentence-transformers >= 2.5.0` (was `>= 2.3.0`).

### Fixed

- Hook `Stop` handler now applies redaction, small-talk filtering, and a
  lightweight duplicate check before observing the last exchange, matching the
  MCP `store()` behavior.
- Custom `redaction_patterns` are now correctly passed through `MimirConfig` to
  the `Redactor` used by `SessionManager`.
- Project context files are no longer re-ingested every session if an identical
  memory already exists in the loaded state.
- `Redactor` now accepts plain regex strings in addition to compiled pattern
  objects.

## [0.2.0] - 2026-07-01

- Initial release of Mimir core: plastic memory well with PPN, hybrid retrieval,
  and agent adapter.

[Unreleased]: https://github.com/Liewzheng/Mimir/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Liewzheng/Mimir/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Liewzheng/Mimir/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Liewzheng/Mimir/releases/tag/v0.2.0
