"""Atomic file-write utilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically via a temp file + rename.

    The temporary file is created in the same directory as the target file so
    that ``os.replace`` can move it into place without crossing filesystem
    boundaries. If the write or rename fails, the temp file is removed before
    the exception propagates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(data)
            tmp_file.flush()
            os.replace(tmp_path, path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise


def atomic_write_json(path: Path, data: Any) -> None:
    """Serialize ``data`` to JSON and write it atomically to ``path``."""
    atomic_write(path, json.dumps(data, indent=2))
