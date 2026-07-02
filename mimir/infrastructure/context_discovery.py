"""Project context discovery for agent-specific instruction files.

Coding agents often rely on project-level instruction files such as
``CLAUDE.md``, ``AGENTS.md``, or ``.cursorrules`` to learn conventions,
architecture decisions, and workflows. Mimir can ingest these files as
high-importance memories so that they are automatically recalled during
coding sessions.

Discovery is intentionally conservative: only well-known agent instruction
files are loaded by default, and files larger than ``max_file_size`` are
skipped to avoid polluting the memory buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Common instruction files used by agent CLIs. Case-insensitive matching is
# applied when scanning the workspace root.
DEFAULT_CONTEXT_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    ".claude.md",
)

DEFAULT_MAX_FILE_SIZE = 100_000  # bytes


@dataclass(frozen=True)
class ContextFile:
    """A discovered project context file."""

    path: Path
    content: str


class ContextDiscovery:
    """Scan a workspace for agent instruction files and load their contents."""

    def __init__(
        self,
        context_files: tuple[str, ...] = DEFAULT_CONTEXT_FILES,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        self.context_files = {name.lower(): name for name in context_files}
        self.max_file_size = max_file_size

    def discover(self, workspace_path: Path) -> list[ContextFile]:
        """Return all readable context files in *workspace_path*.

        Only regular files directly inside *workspace_path* are considered.
        Symbolic links and files that resolve outside the workspace are ignored
        to prevent accidental reads of unrelated or sensitive files.
        """
        root = Path(workspace_path).resolve()
        if not root.is_dir():
            return []

        found: list[ContextFile] = []
        for path in root.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if not path.resolve().is_relative_to(root):
                    continue
            except (OSError, ValueError):
                continue
            canonical = self.context_files.get(path.name.lower())
            if canonical is None:
                continue
            content = self._read(path)
            if content is None:
                continue
            found.append(ContextFile(path=path, content=content))

        # Stable order: by canonical file name.
        order = {name: i for i, name in enumerate(self.context_files.values())}
        found.sort(key=lambda cf: order.get(cf.path.name.lower(), len(order)))
        return found

    def _read(self, path: Path) -> str | None:
        try:
            size = path.stat().st_size
            if size > self.max_file_size:
                return None
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def format_memory(self, context_file: ContextFile) -> str:
        """Format a context file as a memory string.

        The header makes the source explicit so recalled context is not mistaken
        for user-generated instructions.
        """
        return (
            f"[Project Context from {context_file.path.name}]\n"
            f"{context_file.content.strip()}"
        )
