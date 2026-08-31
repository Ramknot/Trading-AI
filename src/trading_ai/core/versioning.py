"""Repository version metadata for non-trading analytical components."""

from __future__ import annotations

import subprocess
from pathlib import Path


def detect_git_commit(project_root: Path) -> str | None:
    """Read HEAD without requiring Git, with a bounded CLI fallback."""

    git_directory = project_root / ".git"
    try:
        if git_directory.is_file():
            pointer = git_directory.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                git_directory = (
                    project_root / pointer.split(":", 1)[1].strip()
                ).resolve()
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if len(head) == 40 and all(
            character in "0123456789abcdef" for character in head.lower()
        ):
            return head
        if head.startswith("ref: "):
            reference = head[5:]
            reference_path = git_directory / reference
            if reference_path.is_file():
                commit = reference_path.read_text(encoding="utf-8").strip()
                if len(commit) == 40:
                    return commit
            packed_refs = git_directory / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if not line.startswith(("#", "^")) and line.endswith(
                        f" {reference}"
                    ):
                        commit = line.split(" ", 1)[0]
                        if len(commit) == 40:
                            return commit
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


__all__ = ["detect_git_commit"]
