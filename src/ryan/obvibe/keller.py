"""
This module contain an axiliary functions
"""

import json
import os
import shutil
from functools import wraps
from typing import Any, Union

import h5py
import numpy as np
import pandas as pd
import pybis


def make_new_property(
    openbis_object: pybis.Openbis,
    new_property_code: str,
    new_property_label: str,
    new_property_description: str,
    new_property_data_type: str,
    collection_type_code: str = "battery_premise2",
) -> None:
    """
    Create a new property type in openBIS and assign it to a specified collection type.

    Parameters:
    openbis_object (Openbis): An authenticated instance of the openBIS API.
    new_property_code (str): The unique code for the new property type.
    new_property_label (str): The label for the new property type.
    new_property_description (str): A brief description of the new property type.
    new_property_data_type (str): The data type of the new property (e.g., 'VARCHAR', 'INTEGER').
    collection_type_code (str, optional): The code of the collection type to which the new property will be assigned.
                                          Defaults to 'battery_premise2'.

    Returns:
    None

    Raises:
    ValueError: If the specified collection type does not exist in openBIS.
    """
    # Create a new property type and save it
    new_property = openbis_object.new_property_type(
        code=new_property_code,
        label=new_property_label,
        description=new_property_description,
        dataType=new_property_data_type,
    )
    new_property.save()

    # Retrieve the specified collection type
    collection_type = openbis_object.get_collection_type(collection_type_code)
    if collection_type is None:
        raise ValueError(f"Collection type '{collection_type_code}' does not exist in openBIS.")

    # Assign the newly created property to the collection type
    collection_type.assign_property(new_property_code)


def get_openbis_obj(dir_pat: str, url: str = r"https://openbis-empa-lab501.ethz.ch/") -> pybis.Openbis:
    """
    Get the openbis object from PAT.

    Parameters:
    dir_pat (str): The directory of the PAT file.
    url (str, optional): The URL of the openBIS server. Defaults to r'https://openbis-empa-lab501.ethz.ch/'.

    ReturnS:
    pybis.Openbis: The openbis object.
    """
    with open(dir_pat, "r") as f:
        token = f.read().strip()
    ob = pybis.Openbis(url)
    ob.set_token(token)
    return ob


def get_metadata_from_json(dir_json: str, path: str) -> Union[str, float]:
    """
    Get the metadata from a JSON file.

    Parameters:
    dir_json (str): The directory of the JSON file.
    path (str): The path to the metadata.

    Returns:
    info: The metadata.
    """
    with open(dir_json, "r") as f:
        json_file = json.load(f)

    info = json_file
    for key in path.split("||"):
        info = info[key]
    return info


def get_permid_specific_type(
    experiment_name: str, dataset_type: str, openbis_obj: str, default_space: str = "/TEST_SPACE_PYBIS/TEST_UPLOAD"
) -> str:
    """Retrieve the permId of a dataset of a specific type in a specific experiment.

    The function will take care of makeing sure the name of all required stuff is in an upper case as requried by OpenBis

    Args:
        experiment_name (str): The name of the experiment. For example: 240906_kigr_gen4_01,
        dataset_type (str): The type of the dataset. For example: premise_cucumber_raw_json.
        openbis_obj (str): The openBIS object.
        default_space (str): The default space to search in.

    Returns:
        str: The permId of the dataset.
    """
    ob = openbis_obj
    # Construct the experiment identifier
    experiment_identifier = f"{default_space}/{experiment_name.upper()}"

    # Get the experiment object
    experiment = ob.get_experiment(experiment_identifier)

    # Retrieve datasets associated with the experiment
    datasets = experiment.get_datasets()

    # Filter datasets by type, comparing in uppercase
    filtered_datasets = [ds for ds in datasets if ds.type.code.upper() == dataset_type.upper()]

    # Extract permIds from the filtered datasets
    perm_ids = [ds.permId for ds in filtered_datasets]

    if len(perm_ids) == 0:
        raise ValueError(f"No datasets of type '{dataset_type}' found in experiment '{experiment_name}'")
    if len(perm_ids) > 1:
        raise ValueError(f"Multiple datasets of type '{dataset_type}' found in experiment '{experiment_name}'")

    return perm_ids[0]


# This trick will download the file, pass the path to the decorated function, and clean up the file afterward.
def with_downloaded_file(openbis_obj, destination="temp_files"):
    """
    Decorator to handle downloading a file, passing its path and permId to the decorated function,
    and cleaning up the file afterward.

    Args:
        openbis_obj: The OpenBIS object.
        destination: The base folder where files will be downloaded.

    Returns:
        Decorated function that receives the path to the downloaded file and the permId as arguments.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(perm_id, *args, **kwargs):
            # Download the dataset
            dataset = openbis_obj.get_dataset(perm_id)
            dataset.download(destination=destination, create_default_folders=True)
            dir_downloaded = os.path.join(destination, perm_id, "original")
            downloaded_filename = os.listdir(dir_downloaded)[0]
            path_downloaded_file = os.path.join(dir_downloaded, downloaded_filename)

            try:
                # Call the decorated function with the downloaded file path and permId
                result = func(path_downloaded_file, perm_id, *args, **kwargs)
            finally:
                # Clean up: Delete the downloaded files
                shutil.rmtree(os.path.join(destination, perm_id))
            return result

        return wrapper

    return decorator


def format_h5_metadata(file_path: Union[str, bytes], key: str = "metadata", max_depth: int = 10) -> dict:
    """
    Loads and formats HDF5 scalar metadata stored as a JSON string.

    Args:
        file_path (str | bytes): Path to the HDF5 file.
        key (str): Key inside the HDF5 file pointing to the metadata object.
        max_depth (int): Maximum depth for recursive cleaning.

    Returns:
        dict: A cleaned and truncated nested dictionary.
    """

    def truncate_large_lists(obj: Any) -> Any:
        if isinstance(obj, list) and len(obj) > 10:
            return obj[:3] + ["..."] + obj[-2:]
        return obj

    def clean_for_use(obj: Any, depth: int = 0) -> Any:
        if depth >= max_depth:
            return f"<truncated at depth {max_depth}>"
        if isinstance(obj, dict):
            return {k: clean_for_use(truncate_large_lists(v), depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_use(truncate_large_lists(v), depth + 1) for v in obj]
        else:
            return obj

    with h5py.File(file_path, "r") as f:
        raw = f[key][()]
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded)
        dict_cleaned = clean_for_use(parsed)

    return dict_cleaned


def upload_dataset(ob: pybis.Openbis, dir_data: str, target_permID: str, sample_type: str) -> None:
    """
    A helper function use to upload a dataset to OpenBis.

    This function will upload the dataset to OpenBis regardless if the target object on OpenBis is an experiment or an object.

    Args:
        ob (pybis.Openbis): The OpenBIS object.
        dir_data (str): The directory of the dataset to upload.
        target_permID (str): The permId of the target object on OpenBis.
        sample_type (str): The type of the sample to upload (this is the type in OpenBis)

    Returns:
        None
    """
    identifier = None
    target_ob_is_experiment = None

    # Get the identifier of the target object regardless if it is an experiment or an object in OpenBis.
    try:
        identifier = ob.get_object(target_permID).identifier
        target_ob_is_experiment = False
    except Exception:
        try:
            identifier = ob.get_experiment(target_permID).identifier
            target_ob_is_experiment = True
        except Exception:
            raise ValueError(f"Cannot find the target object/experiment with permId {target_permID}")

    # Create a new dataset object and upload them.
    if target_ob_is_experiment == None:
        raise ValueError(
            f"The target object with permId {target_permID} is not an experiment or an object. Please check the object type."
        )
    elif target_ob_is_experiment:
        new_dataset = ob.new_dataset(type=sample_type, file=dir_data, experiment=identifier)
    else:
        new_dataset = ob.new_dataset(type=sample_type, file=dir_data, sample=identifier)
    new_dataset.save()


def gen_protocol_data(dir_h5_raw: str) -> None:
    """
    Generate the protocol and data to support the state object in OpenBis.

    This function processes the raw.h5 file from the Cucumber cycler manager and generates two DataFrames:
    one for protocol parameters and another for measurement data. It handles formation cycles (cycles 0, 1, and 2)
    differently from long-term cycling based on the Biologic technique codes.
    After extracting the relevant data, it saves the DataFrames as HDF5 files in the same directory as the raw.h5 file.

    Supported technique codes:
        100: Open Circuit Voltage (OCV)
        101: Chronoamperometry (CA)
        102: Chronopotentiometry (CP)
        103: Cyclic Voltammetry (CV)
        155: Chronopotentiometry with potential limit (CPLIMIT)
        157: Cyclic Voltammetry with potential limit (CVLIMIT)

    Args:
        dir_h5_raw (str): The directory of the raw.h5 file.

    Returns:
        None
    """
    # Load the raw data
    df_raw = pd.read_hdf(dir_h5_raw, key="data")

    # Validate technique numbers using set operations
    supported_techniques = {100, 101, 102, 103, 155, 157}
    unsupported = set(df_raw["technique"].unique()) - supported_techniques
    if unsupported:
        raise ValueError(f"Unsupported protocol numbers found: {', '.join(map(str, unsupported))}")

    # Compute the time offset from the first timestamp
    time_offset = df_raw["uts"] - df_raw["uts"].iloc[0]

    # Create a boolean mask for techniques that require specific handling:
    # For techniques [100, 102, 155], Applied Voltage is not used and Measured Current is not available.
    mask = df_raw["technique"].isin([100, 102, 155])

    # Create the protocol DataFrame using vectorized operations
    df_protocol = pd.DataFrame({
        "Time (s)": time_offset,
        "Applied Voltage (V)": np.where(mask, np.nan, df_raw["V (V)"]),
        "Applied Current (A)": np.where(mask, df_raw["I (A)"], np.nan),
    })

    # Create the data DataFrame using vectorized operations
    df_data = pd.DataFrame({
        "Time (s)": time_offset,
        "Timestamp (uts)": df_raw["uts"],
        "Measured Potential (V)": np.where(mask, df_raw["V (V)"], np.nan),
        "Measured Current (A)": np.where(mask, np.nan, df_raw["I (A)"]),
    })

    # Find the separation between formation and long-term cycling
    idx_cut = int(df_raw[df_raw["Cycle"] == 2].index[-1])

    # Split and reset the index to remove the 'index' column from the saved files
    df_formation_cycle_protocol = df_protocol.iloc[:idx_cut].reset_index(drop=True)
    df_formation_cycle_data = df_data.iloc[:idx_cut].reset_index(drop=True)
    df_long_term_cycle_protocol = df_protocol.iloc[idx_cut + 1 :].reset_index(drop=True)
    df_long_term_cycle_data = df_data.iloc[idx_cut + 1 :].reset_index(drop=True)

    dir_save = dir_h5_raw.parent

    # Save using the fixed format to eliminate the _i_table sub-layer and preserve original column names
    df_formation_cycle_protocol.to_hdf(
        dir_save / "formation_cycle_protocol.h5", key="protocol", mode="w", format="fixed"
    )
    df_formation_cycle_data.to_hdf(dir_save / "formation_cycle_data.h5", key="data", mode="w", format="fixed")
    df_long_term_cycle_protocol.to_hdf(
        dir_save / "long_term_cycle_protocol.h5", key="protocol", mode="w", format="fixed"
    )
    df_long_term_cycle_data.to_hdf(dir_save / "long_term_cycle_data.h5", key="data", mode="w", format="fixed")
