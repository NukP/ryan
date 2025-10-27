"""This auxliary module contain functions used in feature engineering of many custom functions in custom_columns
module."""
from typing import Tuple, Dict, List, Union
import pandas as pd
import os
import time
from typing import TYPE_CHECKING

# I put this to avoid cyclical import issues.
if TYPE_CHECKING:
    from main import DataPackage


def convert_seconds(seconds: int) -> tuple:
    """
    Converts a given number of seconds into hours, minutes, and seconds.

    This function takes an integer representing a number of seconds and
    converts it into a tuple representing the equivalent amount of time
    in hours, minutes, and seconds. This function is used for convert the time requires for performing task.

    Args:
        seconds (int): The total number of seconds to convert.

    Returns:
        tuple: A tuple in the format (hours, minutes, seconds).
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds


def get_time_slot(time_slot_option: str, data_package: "DataPackage") -> Tuple[float, float]:
    """
    Get start time and end time for the range of interest for various function in custom_columns module.

    For time_slot_option = global, the time start at the beginning of the electrochemical measurement (time at idx==0 in electro.xlsx file).
    For the time_slot_option = delta the time start at 1 second (so that it does not interfere with the cycle before) after the injection of the previous GC run.
    Except for idx==0, I let the time start 5 mins before as this is generally the time between the injection. All the time slot ends at the time of the GC injection.

    Args:
        time_slot_option (str): Type of time slot, global or delta.
        data_package (Any): data_package from main.gen_dataset.DataPackage.

    Raises:
        ValueError: If the given time_slot_option is not global or delta.

    Returns:
        Tuple[float, float]: [t_start, t_end] start time and end time for the range using in various function in custom_columns module.
    """
    idx = data_package.idx
    df_gc = data_package.df_gc
    df_electro = data_package.df_electro
    t_end = df_gc["time"][idx]
    if time_slot_option == "delta":
        if idx == 0:
            t_start = t_end - 360
        else:
            t_start = df_gc["time"][idx - 1] + 1
    elif time_slot_option == "global":
        t_start = df_electro["Unnamed: 0"][0]
    else:
        raise ValueError('time slot option can only be "delta" or "global"')
    return t_start, t_end


def process_row(
    args_tuple: Tuple[bool, int, Dict[str, pd.DataFrame], List[str], List[str], List[str], bool, str, str]
) -> Dict[str, Union[int, str, float]]:
    """
    A function used for extracting and calculating values for engineered features in main.gen_dataset.

    This function is used exclusively by main.gen_dataset. It is separated here as it is required by multiprocessing library.

    Args:
        args_tuple (Tuple): Tuple contains parameters from main.gen_dataset.

    Returns:
        Dict[str, Union[int, str, float]]: Dictionary of values for each column for the given index.
    """
    from .main import DataPackage
    from .custom_columns import process_feature

    (
        pass_fl,
        idx,
        dict_df,
        ls_input_gen,
        ls_input_extract,
        ls_output_extract,
        ls_output_gen,
        pre_gen_electro_df,
        gc_excel_dir,
    ) = args_tuple
    row_data = {col: dict_df[pass_fl.split(".xlsx")[0]].loc[idx, col] for col in ls_input_extract + ls_output_extract}
    if pre_gen_electro_df:
        fln_electro = pass_fl.split(".")[0] + ".electro"
        data_pack = DataPackage(
            df_gc=dict_df[pass_fl.split(".xlsx")[0]],
            df_electro=dict_df[fln_electro],
            idx=idx,
            fln=pass_fl,
            dir=gc_excel_dir,
        )
    else:
        data_pack = DataPackage(df_gc=dict_df[pass_fl.split(".xlsx")[0]], idx=idx, fln=pass_fl, dir=gc_excel_dir)
    ls_t_genfunc = []
    for gen_func in ls_output_gen + ls_input_gen:
        ts_genfunc = time.time()
        custom_col_name, custom_col_data = process_feature(gen_func, data_pack)
        row_data[custom_col_name] = custom_col_data
        te_genfunc = time.time()
        t_genfunc = te_genfunc - ts_genfunc
        ls_t_genfunc.append((custom_col_name, t_genfunc))
    return row_data


def read_excel(args: Tuple[str, str]) -> Tuple[str, pd.DataFrame]:
    """
    Function used in generating dict of df using multi processing library.

    Args:
        args: Tuple containing (fln, dir_path): Excel filename and directory path.

    Returns:
        Tuple[str, pd.Dataframe]: [os.path.basename(fln).split(".xlsx")[0], df] Name of the file and pd.Dataframe read from the file
        (with some modification based on type of files.)
    """
    from .main import mod_df

    fln, dir_path = args
    path_fl = os.path.join(dir_path, fln)
    if fln.split(".")[1] == "electro":
        df = pd.read_excel(path_fl, skiprows=[1])
    elif fln.split(".")[1] == "GCdata":
        try:
            df = mod_df(path_fl)
        except Exception as e:
            print(f"Failed to generate df from file: {fln}: {e}")
    else:
        print(f"Warning: invalid file type: {fln}")
    return os.path.basename(fln).split(".xlsx")[0], df


def t_to_idx(df: pd.DataFrame, t: float) -> float:
    """
    Convert the given time value to the index of the dataframe with the closet value.

    This function will be used to convert the time domain into an index.

    Args:
        df (pd.DataFrame): Target dataframe to be working on.
        t (float): Time value

    Returns:
        float: Index of the cell that has the value closet to the given t.
    """
    try:
        target_column = df["Unnamed: 0"]  # in case of df_electro
    except:
        pass
    try:
        target_column = df["time"]  # in case of df_GC_excel
    except:
        target_column = df[df.columns[0]]  # in general case, the time column should be the first column.

    diff = abs(target_column - t)
    idx_closet = diff.idxmin()
    return idx_closet


def gen_selected_val_list(df: pd.DataFrame, target_column: str, idx_start: int, idx_end: int, mode: str, treshold_value: float) -> list:
    """
    Generates a list of tuples containing the index and the value from a specified column of a DataFrame, filtered based
    on a mode ('positive' or 'negative').

    This function filters values in a specified column of the DataFrame, either for positive or negative values, within a given index range.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        target_column (str): The column name in 'df' to filter values from.
        idx_start (int): The starting index (inclusive) for filtering.
        idx_end (int): The ending index (inclusive) for filtering.
        mode (str): Mode of filtering. Accepts 'positive' for values > treshold_value or 'negative' for values < treshold_value.
        treshold_value (float): A value for for filtering positive and negative mode. This value is set in the custom_columns module.

    Returns:
        list of tuple: A list of tuples, each containing (index, value), where 'index' is the DataFrame index and 'value' is the corresponding value in 'target_column'.
                        Values are included based on the 'mode' specified (either positive or negative).

    Raises:
        ValueError: If 'mode' is not 'positive' or 'negative'.
    """
    ls_sel_val = []
    if mode == "positive":
        ls_sel_val = [(idx, df[target_column][idx]) for idx in range(idx_start, idx_end + 1) if df[target_column][idx] > treshold_value]
    elif mode == "negative":
        ls_sel_val = [(idx, df[target_column][idx]) for idx in range(idx_start, idx_end + 1) if df[target_column][idx] < treshold_value]
    else:
        raise ValueError('The mode can only either be "positive" or "negative".')
    return ls_sel_val


def check_input_ls(ls_idx: list) -> list:
    """
    Processes the input list to ensure it contains valid indices and converts it into a sorted list of indices.

    This function checks if the input list is non-empty and contains either integers or tuples of (index, value).
    It then extracts the indices from the list (if necessary) and returns a sorted list of these indices.

    Args:
        ls_idx (list): A list of integers or tuples. If tuples, they are expected to be in the format (index, value).

    Returns:
        list: A sorted list of indices extracted from the input list.

    Raises:
        ValueError: If the input list is empty or contains elements of invalid type.

    Example:
        >>> check_input_ls([(1, 'a'), (2, 'b'), (3, 'c')])
        [1, 2, 3]
        >>> check_input_ls([1, 2, 3])
        [1, 2, 3]
        >>> check_input_ls([])
        ValueError: The list is empty. Please check the input ls_idx.
    """
    # Check if the list is empty
    if not ls_idx:
        raise ValueError("The list is empty. Please check the input ls_idx.")
    # Determine the type of list elements and extract indices
    if isinstance(ls_idx[0], tuple):
        ls_idx_to_process = [idx for idx, _ in ls_idx]
    elif isinstance(ls_idx[0], int):
        ls_idx_to_process = ls_idx
    else:
        raise ValueError("Invalid data type in ls_idx. Please check the input ls_idx.")

    return sorted(ls_idx_to_process)


def gen_window_section_idx_list(ls_idx: list) -> list:
    """
    Generates a list of continuous index ranges (windows) from a list of indices.

    This function processes a list of indices, identifying continuous sequences (windows) and returning these as ranges.
    Each range is represented as a tuple (start_index, end_index). The function relies on 'check_input_ls' to preprocess the input list.

    Args:
        ls_idx (list): A list of indices or tuples. If tuples, they are expected to be in the format (index, value).
                       The list should be processed by 'check_input_ls' to ensure correct format.

    Returns:
        list of tuple: A list of tuples, each representing a continuous range of indices.
                       Each tuple is in the format (start_index, end_index). For a single index range, both elements are the same.

    Example:
        >>> gen_window_section_idx_list([(1, 'a'), (2, 'b'), (4, 'c'), (5, 'd')])
        [(1, 2), (4, 5)]
        >>> gen_window_section_idx_list([1, 2, 4, 5])
        [(1, 2), (4, 5)]
    """
    sorted_ls_number = check_input_ls(ls_idx)
    windows = []
    start = sorted_ls_number[0]

    for i in range(1, len(sorted_ls_number)):
        if sorted_ls_number[i] != sorted_ls_number[i - 1] + 1:
            windows.append((start, sorted_ls_number[i - 1]))
            start = sorted_ls_number[i]

    windows.append((start, sorted_ls_number[-1]))
    return windows


def sum_val_selected_windows(df: pd.DataFrame, column_name: str, ls_sel_windows: list) -> float:
    """
    Calculates the sum of differences in a specified column's values at the start and end of each window in a list.

    This function iterates over a list of index 'windows' and computes the difference between the value at the end and the start of each window
    in a specified column of a DataFrame. It then sums these differences and returns the total.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column in the DataFrame from which values will be used.
        ls_sel_windows (list of tuple): A list of tuples, where each tuple contains two indices (start, end) representing a window.

    Returns:
        float: The sum of the differences in the specified column's values at the start and end of each window.

    Example:
        >>> df = pd.DataFrame({'A': [10, 20, 30, 40, 50]})
        >>> sum_val_selected_windows(df, 'A', [(0, 1), (2, 4)])
        40 # (20-10) + (50-30) = 10 + 30 = 40
    """
    sum_value = 0
    for window in ls_sel_windows:
        idx_start, idx_end = window
        sum_value += df[column_name][idx_end] - df[column_name][idx_start]

    return sum_value


def get_avg_selected_val(df: pd.DataFrame, column_name: str, ls_windows: list) -> float:
    """
    Calculates the average value of a specified column at selected indices in a DataFrame.

    This function first processes the input list of windows or indices using the 'check_input_ls' function to ensure
    it contains valid indices. It then calculates the sum of the values at these indices in the specified column of the DataFrame.
    Finally, it computes and returns the average of these values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column in the DataFrame from which values will be used.
        ls_windows (list of int or tuple): A list of indices or tuples. If tuples, they are expected to be in the format (index, value).
                                           This list is processed by 'check_input_ls' to ensure correct format.

    Returns:
        float: The average value of the selected indices in the specified column.
    """
    sorted_ls = check_input_ls(ls_windows)
    sum_value = 0
    for interest_idx in sorted_ls:
        sum_value += df[column_name][interest_idx]
    avg_value = sum_value / len(ls_windows)
    return avg_value


def unpack_data_cc(data_package: "DataPackage", time_slot_option: str):
    """
    Get df_electro, idx_start, idx_end, t_start, t_end parameters that can be used for functions in custom_columns.

    This function is made to reduece the redundancy in generating f_electro, idx_start, idx_end, t_start, t_end parameters.
    This can be used in many functions in custom_columns module.

    Args:
        data_package (DataPackage class): DataPackage class, define in main.get_dataset function.
        time_slot_option (str): Type of time scale, global or delta, this will be fed into get_time_slot function.

    Returns:
        Tuple: (df_electro, idx_start, idx_end, t_start, t_end)
    """
    from .main import DataPackage

    df_electro = data_package.df_electro
    t_start, t_end = get_time_slot(time_slot_option=time_slot_option, data_package=data_package)
    idx_start = t_to_idx(df_electro, t_start)
    idx_end = t_to_idx(df_electro, t_end)
    packed_args = df_electro, idx_start, idx_end, t_start, t_end
    return packed_args


def get_metadata_cell_value(column_name: str, data_package: "DataPackage") -> Union[str, float]:
    """
    Retrieves the metadata value for a given column from an associated Excel file.

    Args:
        column_name (str): The metadata column to retrieve the value from.
        data_package (DataPackage): DataPackage class, define in main.get_dataset function.

    Returns:
        Union[str, float]: The value found in the 'Value' column for the given column name.
    """
    fln = data_package.fln
    fl_dir = data_package.dir
    metadata_fln = fln.split(".")[0].split("datagram_")[1] + "-metadata.xlsx"
    dir_fl_metadata = os.path.join(fl_dir, metadata_fln)
    df_metadata = pd.read_excel(dir_fl_metadata)
    # Remove leading and trailing whitespace from column names
    df_metadata.columns = df_metadata.columns.str.strip()
    # Remove leading and trailing whitespace from the 'Metadata' column values
    df_metadata["Metadata"] = df_metadata["Metadata"].str.strip()
    # Ensure the column_name is also stripped of any whitespace
    column_name = column_name.strip()
    value = df_metadata.loc[df_metadata["Metadata"] == column_name, "Value"].values[0]
    return value


def pseudolist_conv(pseudo_list_str: str) -> dict:
    """
    Converts a string representation of a pseudo-list structure into a dictionary.

    The input string should represent a series of key-value pairs enclosed in brackets,
    similar to list entries, but intended to be converted into dictionary items. This function
    is robust enough to handle single or multiple key-value pairs, converting numeric values
    to floats whenever possible.

    Args:
        pseudo_list_str (str): The string representation of the pseudo-list structure,
        e.g., "[KCl,1],[H2SO4,0.1]".

    Returns:
        dict: A dictionary with keys and values extracted from the input string,
        where values are converted to floats if appropriate.

    Note:
        - This function assumes the input string is well-formed, with key-value pairs
        correctly separated by commas and enclosed in brackets.
        - Values are converted to floats when possible; otherwise, they remain as strings.
    """
    components = pseudo_list_str.replace(" ", "").split("],[")
    composition_dict = {}
    for component in components:
        cleaned_component = component.replace("[", "").replace("]", "")
        chemical, concentration = cleaned_component.split(",")
        composition_dict[chemical] = float(concentration)

    return composition_dict


def cal_cathode_ions(data_package: "DataPackage") -> dict:
    """
    Calculates the composition of cathode ions based on the cathode electrolyte compartment's solute content.

    This function parses the cathode electrolyte compartment solute content to identify the presence and concentration
    of specific ions (K+, Na+, sulphate, Cl- and HCO3-). It is part of a series of functions used for processing electrolyte
    metadata within the `treat_metadata` module. The input `data_package` should be an instance of `main.DataPackage`,
    although it's not explicitly imported to avoid cyclical import issues.

    Parameters:
    - data_package: An instance of `main.DataPackage` containing electrolyte compartment data.

    Returns:
    - dict: A dictionary with keys representing ion types (K+, Na+, sulphate, Cl- and HCO3-) and values their total concentrations.

    Note:
    Due to potential cyclical import issues, `main.DataPackage` is not explicitly imported but is required as input.
    """
    dict_cathode_ions = {"K+": 0, "Na+": 0, "sulphate": 0, "Cl-": 0, "HCO3-": 0}
    str_cathode_composition = get_metadata_cell_value(
        column_name="Cathode electrolyte compartment solute content [name, concentration in M]", data_package=data_package
    )
    dict_cathode_composition = pseudolist_conv(str_cathode_composition)
    if "HCl" in dict_cathode_composition.keys():
        dict_cathode_ions["Cl-"] += dict_cathode_composition["HCl"]
    if "KCl" in dict_cathode_composition.keys():
        dict_cathode_ions["Cl-"] += dict_cathode_composition["KCl"]
        dict_cathode_ions["K+"] += dict_cathode_composition["KCl"]
    if "KOH" in dict_cathode_composition.keys():
        dict_cathode_ions["K+"] += dict_cathode_composition["KOH"]
    if "NaCl" in dict_cathode_composition.keys():
        dict_cathode_ions["Na+"] += dict_cathode_composition["NaCl"]
        dict_cathode_ions["Cl-"] += dict_cathode_composition["NaCl"]
    if "H2SO4" in dict_cathode_composition.keys():
        dict_cathode_ions["sulphate"] += dict_cathode_composition["H2SO4"]
    if "KHCO3" in dict_cathode_composition.keys():
        dict_cathode_ions["HCO3-"] += dict_cathode_composition["KHCO3"]
        dict_cathode_ions["K+"] += dict_cathode_composition["KHCO3"]
    return dict_cathode_ions


def cal_anode_ions(data_package: "DataPackage") -> dict:
    """
    Calculates the concentration of specific ions in the anode electrolyte compartment.

    This function parses the anode electrolyte compartment solute content to identify the presence and concentration
    of specific ions (K+, Cl-, sulphate, HCO3-, and phosphate). It is part of a series of functions used for processing
    electrolyte metadata within the `treat_metadata` module. The input `data_package` should be an instance of
    `main.DataPackage`, although it's not explicitly imported to avoid cyclical import issues.

    Parameters:
    - data_package: An instance of `main.DataPackage` containing electrolyte compartment data.

    Returns:
    - dict: A dictionary with keys representing ion types (K+, Cl-, sulphate, HCO3-, and phosphate) and values their total concentrations.

    Note:
    Due to potential cyclical import issues, `main.DataPackage` is not explicitly imported but is required as input.
    """
    dict_anode_ions = {"K+": 0, "Cl-": 0, "sulphate": 0, "HCO3-": 0, "phosphate": 0}
    str_anode_composition = get_metadata_cell_value(
        column_name="Anode electrolyte compartment solute content [name, concentration in M]", data_package=data_package
    )
    dict_anode_composition = pseudolist_conv(str_anode_composition)
    if "KCl" in dict_anode_composition.keys():
        dict_anode_ions["Cl-"] += dict_anode_composition["KCl"]
        dict_anode_ions["K+"] += dict_anode_composition["KCl"]
    if "KHCO3" in dict_anode_composition.keys():
        dict_anode_ions["HCO3-"] += dict_anode_composition["KHCO3"]
        dict_anode_ions["K+"] += dict_anode_composition["KHCO3"]
    if "H3PO4" in dict_anode_composition.keys():
        dict_anode_ions["phosphate"] += dict_anode_composition["H3PO4"]
    if "H2SO4" in dict_anode_composition.keys():
        dict_anode_ions["sulphate"] += dict_anode_composition["H2SO4"]
    return dict_anode_ions


def cal_cathode_gas_misture(data_package: "DataPackage") -> dict:
    """
    Calculates the concentration of specific gases in the cathode gas mixture.

    This function parses the cathode gas mixture to identify the presence and concentration
    of specific gases (CO2 and CO). It is part of a series of functions used for processing
    gas mixture metadata within the `treat_metadata` module. The input `data_package` should be an instance of
    `main.DataPackage`, although it's not explicitly imported to avoid cyclical import issues.

    Parameters:
    - data_package: An instance of `main.DataPackage` containing gas mixture data.

    Returns:
    - dict: A dictionary with keys representing gas types (CO2 and CO) and values their total concentrations.

    Note:
    Due to potential cyclical import issues, `main.DataPackage` is not explicitly imported but is required as input.
    """
    dict_cathode_gas_misture = {"CO2": 0, "CO": 0}
    str_cathode_gas_misture = get_metadata_cell_value(
        column_name="Cathode gas mixture and concentration [chemical formula, mol fraction]", data_package=data_package
    )
    dict_mixtures = pseudolist_conv(str_cathode_gas_misture)
    if "CO2" in dict_mixtures.keys():
        dict_cathode_gas_misture["CO2"] += dict_mixtures["CO2"]
    if "CO" in dict_mixtures.keys():
        dict_cathode_gas_misture["CO"] += dict_mixtures["CO"]
    return dict_cathode_gas_misture
