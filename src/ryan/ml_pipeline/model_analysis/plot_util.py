""" "
This module contains functions for facilitating loading frequently-used files such as the trained dataset.
This makes importing files seamless across notebook located at different directory.
"""

import json
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
        target_filename: File name to search for under the project root.
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


def get_df_dataset(path_dataset: Path | None = None) -> pd.DataFrame:
    """
    Load the dataset into a DataFrame, dropping rows with missing values.

    Args:
        path_dataset: Path to the dataset file. If None, searches the project root
            for the default dataset filename.

    Returns:
        A pandas DataFrame containing the dataset with NaNs removed.
    """
    if path_dataset is None:
        path_dataset = get_path_from_root_dir("dataset_22Jan20226_itteration6_CO2only.xlsx")
    df_dataset = pd.read_excel(path_dataset)
    df_dataset = df_dataset.dropna()
    return df_dataset


def get_dict_best_params(path_json_best_params: Path | None = None) -> dict:
    """
    Load the best-parameters (from hyperparameter tuning) JSON file into a dictionary.

    Args:
        path_json_best_params: Path to the JSON file. If None, searches the project
            root for the default filename.

    Returns:
        A dict containing the best hyperparameters.
    """
    if path_json_best_params is None:
        path_json_best_params = get_path_from_root_dir("summarized_model_hyperparameter.json")
    with open(path_json_best_params, "r") as f:
        dict_best_params = json.load(f)
    return dict_best_params
