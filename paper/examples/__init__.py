"""Common utilities for paper experiments."""

from pathlib import Path

__all__ = ["get_project_root"]


def get_project_root(current_path: Path | None = None) -> Path:
    """Return the repository root given a path inside an example folder."""

    if current_path is None:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    # Assume structure .../<repo>/paper/examples/<experiment>
    for parent in current_path.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback: two levels up from paper directory
    return current_path.parents[2]
