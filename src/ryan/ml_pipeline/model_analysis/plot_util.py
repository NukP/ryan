""" "
This module contains functions for facilitating loadding frequently-used files such as the trained dataset.
This makes importing files seamless across notebook located at different directory.
"""

from pathlib import Path


def find_project_root(start: Path | None = None, marker: str = ".project-root") -> Path:
    """
    Find the nearest ancestor directory (including start) that contains a marker.

    Args:
        start: Directory to start searching from. Defaults to the current working directory.
        marker: File or directory name that identifies the project root.

    Returns:
        The Path to the directory that contains the marker.

    Raises:
        FileNotFoundError: If no marker is found from start up to the filesystem root.
    """
    if start is None:
        start = Path.cwd()

    start = start.resolve()

    for p in [start, *start.parents]:
        if (p / marker).exists():
            return p

    raise FileNotFoundError(f"Could not find project root marker '{marker}' starting from: {start}")
