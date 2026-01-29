""" "
This module contains functions for facilitating loading frequently-used files such as the trained dataset.
This makes importing files seamless across notebook located at different directory.
"""

from pathlib import Path

import pandas as pd


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


def get_path_from_root_dir(target_filename: str, dir_project_root: Path | None = None) -> Path:
    """
    Locate a unique file or directory name under the project root.

    Args:
        target_filename: File or directory name to search for under the project root.
        dir_project_root: Project root to search within. Defaults to find_project_root().

    Returns:
        The Path to the single matching file or directory.

    Raises:
        ValueError: If zero or multiple matches are found.
    """
    if dir_project_root is None:
        dir_project_root = find_project_root()
    ls_target_path = list(dir_project_root.rglob(target_filename))
    if len(ls_target_path) != 1:
        paths_str = "\n".join(str(p) for p in ls_target_path)
        raise ValueError(
            f"Expected exactly one path for '{target_filename}', but found {len(ls_target_path)}:\n{paths_str}"
        )

    return ls_target_path[0]


def get_df_dataset(
    path_root_project: Path | None = None, dataset_path_from_root: Path | None = None, dataset_filename: str = ""
) -> pd.DataFrame:
    if path_root_project is None:
        path_root_project = find_project_root()
    if dataset_path_from_root is not None:
        path_dataset = path_root_project / dataset_path_from_root
    else:
        pass  # continue from this
