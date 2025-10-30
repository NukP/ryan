"""
This module contains a function that will be usued by main.gen_dataset to generate a custom features. The following
description applies to all of the functions except process_feature.

Args: All of the functions take data_package as an argument. The data_package is an instance of main.DataPackage.

Returns: All of the function returns:
    new_col_data: Any, - Data for the specific cell.
"""

from typing import Callable, Tuple, Union

import numpy as np
import xarray as xr

from . import auxiliary as aux
from . import main

TRESHOLD_CURRENT = -2.5  # in mA. This is the threshold current for which the current is considered negative.


def process_feature(
    func: Callable[[main.DataPackage], Union[None, Tuple[str, Union[int, float]]]], data_package: main.DataPackage
) -> Tuple[str, Union[None, Tuple[str, Union[int, float]]]]:
    """
    Process a feature using a custom function.

    This function is the function that will be called by main.gen_dataset

    Args:
        func (Callable[[DataPackage], Union[None, Tuple[str, Union[int, float]]]]): A custom function for feature engineering.
        data_package (DataPackage): An instance of DataPackage containing data required for feature processing.

    Returns:
        Tuple[str, Union[None, Tuple[str, Union[int, float]]]]: A tuple containing the function name and the processed feature data.
            The processed feature data is a tuple with the first element being the custom column name and the second element
            being the value (int or float), or None if the DataFrame is empty.
    """
    # For the first part of main.gen_dataset where a dummy data is used, this is solely to return the name of the function to be used as a column header.
    if data_package.df_gc.empty:
        return func.__name__, None
    # For when process_row function is callled for feature engineering.
    new_col_data = func(data_package)
    return func.__name__, new_col_data


# Feature - dummy
def dummy(data_package: main.DataPackage):
    """
    This is a dummy code meant for just testing the aux.process_row function.

    It doesn't requre any computation, so this should be a good measure of how long does it take to just prepare the
    data class for the calculation.
    """
    new_col_data = "dummy"
    return new_col_data


# Feature no.1
def global_time(data_package: main.DataPackage):
    """Total time from the begining of the elctrochemical measurement until the time of the GC injection."""
    t_start, t_end = aux.get_time_slot(time_slot_option="global", data_package=data_package)
    new_col_data = t_end - t_start
    return new_col_data


# Feature no.2
def global_Q(data_package: main.DataPackage):
    """Total charges that has been passed from the begining of the elctrochemical measurement until the time of the GC
    injection."""
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "global")
    q_global = df_electro["Q [A·s]"][idx_end] - df_electro["Q [A·s]"][idx_start]
    new_col_data = q_global
    return new_col_data


# Feature no.3
def delta_Q(data_package: main.DataPackage):
    """Charges that has been passed since the previous GC injection to the current injection point."""
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    q_global = df_electro["Q [A·s]"][idx_end] - df_electro["Q [A·s]"][idx_start]
    new_col_data = q_global
    return new_col_data


# Feature no.4
def global_time_positive_i(data_package: main.DataPackage):
    """
    The global time which the current in positive.

    The treshold of -1 is chosen instead of 0 is becasue Alesandro mentioned that some slighly negative current is still
    considered positive. The actual applied cathodic current is much larger than this. Also, a small amount of negative
    current is enough to oxidize the metal catalyst surface.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "global")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="positive",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Unnamed: 0", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.5
def delta_time_positive_i(data_package: main.DataPackage):
    """The total time (from the previous GC injection to the current GC injection) when current in a postive region."""
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="positive",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Unnamed: 0", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.6
def global_time_negative_i(data_package: main.DataPackage):
    """The global time which the current in negative."""
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "global")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="negative",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Unnamed: 0", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.7
def delta_time_negative_i(data_package: main.DataPackage):
    """The total time (from the previous GC injection to the current GC injection) when current in a negative region."""
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="negative",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Unnamed: 0", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.8
def delta_time_posneg_ratio(data_package: main.DataPackage):
    """
    The ratio between the time where the current is positive to the time where the current is negative.

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    if delta_time_negative_i(data_package) == 0:
        new_col_data = np.NaN
    else:
        new_col_data = delta_time_positive_i(data_package) / delta_time_negative_i(data_package)
    return new_col_data


# Feature no.9
def global_time_posneg_ratio(data_package: main.DataPackage):
    """
    The ratio between the time where the current is positive to the time where the current is negative.

    Considering the global time.
    """
    if delta_time_negative_i(data_package) == 0:
        new_col_data = np.NaN
    else:
        new_col_data = global_time_positive_i(data_package) / global_time_negative_i(data_package)
    return new_col_data


# Feature no.10
def delta_q_pos_i(data_package: main.DataPackage):
    """
    The total charge passed when the current is positive (the threshold for which current is positive is defined in
    aux.gen_selected_val_list.)

    Considering the time form the previous gc injection to the current gc injection.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="positive",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Q [A·s]", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.11
def global_q_pos_i(data_package: main.DataPackage):
    """
    The total charge passed when the current is positive (the threshold for which current is positive is defined in
    aux.gen_selected_val_list.)

    Considering the time since the start of electrochemical measurement start until the current GC injection.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "global")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="positive",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Q [A·s]", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.12
def delta_q_neg_i(data_package: main.DataPackage):
    """
    The total charge passed when the current is negative (the threshold for which current is positive is defined in
    aux.gen_selected_val_list.)

    Considering the time form the previous gc injection to the current gc injection.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="negative",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Q [A·s]", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.13
def global_q_neg_i(data_package: main.DataPackage):
    """
    The total charge passed when the current is negative (the threshold for which current is positive is defined in
    aux.gen_selected_val_list.)

    Considering the time since the start of electrochemical measurement start until the current GC injection.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "global")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="negative",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        ls_window_idx_ipos = aux.gen_window_section_idx_list(ls_idx_ipos)
        total_time_ipos = aux.sum_val_selected_windows(
            df=df_electro, column_name="Q [A·s]", ls_sel_windows=ls_window_idx_ipos
        )
        new_col_data = total_time_ipos
    else:
        new_col_data = 0
    return new_col_data


# Feature no.14
def delta_q_posneg_ratio(data_package: main.DataPackage):
    """
    The ratio between the passed charge where the current is positive to the passed charge where the current is
    negative.

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    if delta_q_neg_i(data_package) == 0:
        new_col_data = np.NaN
    else:
        new_col_data = delta_q_pos_i(data_package) / delta_q_neg_i(data_package)
    return new_col_data


# Feature no.15
def global_q_posneg_ratio(data_package: main.DataPackage):
    """
    The ratio between the passed charge where the current is positive to the passed charge where the current is
    negative.

    Considering the time since the start of electrochemical measurement start until the current GC injection.
    """
    if global_q_neg_i(data_package) == 0:
        new_col_data = np.NaN
    else:
        new_col_data = global_q_pos_i(data_package) / global_q_neg_i(data_package)
    return new_col_data


# Feature no.16
def delta_i_neg_avg(data_package: main.DataPackage):
    """
    The average negative current. (the threshold for which current is positive is defined in aux.gen_selected_val_list.)

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="negative",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        new_col_data = aux.get_avg_selected_val(df=df_electro, column_name="I [mA]", ls_windows=ls_idx_ipos)
    else:
        new_col_data = 0
    return new_col_data


# Feature no.17
def delta_i_pos_avg(data_package: main.DataPackage):
    """
    The average positive current. (the threshold for which current is positive is defined in aux.gen_selected_val_list.)

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="I [mA]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="positive",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        new_col_data = aux.get_avg_selected_val(df=df_electro, column_name="I [mA]", ls_windows=ls_idx_ipos)
    else:
        new_col_data = 0
    return new_col_data


# Feature no.18
def delta_i_posneg_avg_ratio(data_package: main.DataPackage):
    """
    The ratio between the average positive and negative current.

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    if delta_i_neg_avg(data_package) == 0:
        new_col_data = np.NaN
    else:
        new_col_data = delta_i_pos_avg(data_package) / delta_i_neg_avg(data_package)
    return new_col_data


# Feature no.19 # Check with Ale
def delta_e_pos_avg(data_package: main.DataPackage):
    """
    The average positive potential. (the threshold for which current is positive is defined in
    aux.gen_selected_val_list.)

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="Ewe [V]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="positive",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        new_col_data = aux.get_avg_selected_val(df=df_electro, column_name="Ewe [V]", ls_windows=ls_idx_ipos)
    else:
        new_col_data = 0
    return new_col_data


# Feature no.20 # Check with Ale
def delta_e_neg_avg(data_package: main.DataPackage):
    """
    The average negative potential. (the threshold for which current is positive is defined in
    aux.gen_selected_val_list.)

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_idx_ipos = aux.gen_selected_val_list(
        df=df_electro,
        target_column="Ewe [V]",
        idx_start=int(idx_start),
        idx_end=int(idx_end),
        mode="negative",
        treshold_value=TRESHOLD_CURRENT,
    )
    if ls_idx_ipos:
        new_col_data = aux.get_avg_selected_val(df=df_electro, column_name="Ewe [V]", ls_windows=ls_idx_ipos)
    else:
        new_col_data = 0
    return new_col_data


# Feature no.21
def delta_e_posneg_avg_ratio(data_package: main.DataPackage):
    """
    The ratio between the average positive and negative potential.

    Considering the time scale from the previous GC injection to the current GC injection point.
    """
    if delta_e_neg_avg(data_package) == 0:
        new_col_data = np.NaN
    else:
        new_col_data = delta_e_pos_avg(data_package) / delta_e_neg_avg(data_package)
    return new_col_data


def delta_r(data_package: main.DataPackage):
    """The calculated resistance from the previous GC injection to the current GC injection."""
    idx = data_package.idx
    df_gc = data_package.df_gc
    if idx == 0:
        new_col_data = np.NaN
    else:
        delta_r = df_gc["R [Ω]"][idx] - df_gc["R [Ω]"][idx - 1]
        new_col_data = delta_r
    return new_col_data


def delta_flow_out(data_package: main.DataPackage):
    """The difference in the flow_out between the previous GC injection to the current GC injection."""
    idx = data_package.idx
    df_gc = data_package.df_gc
    if idx == 0:
        new_col_data = np.NaN
    else:
        delta_flow_out = df_gc["fout [smL/min]"][idx] - df_gc["fout [smL/min]"][idx - 1]
        new_col_data = delta_flow_out
    return new_col_data


def delta_Eapp(data_package: main.DataPackage):
    """The difference in the applied potential between the previous GC injection to the current GC injection."""
    idx = data_package.idx
    df_gc = data_package.df_gc
    if idx == 0:
        new_col_data = np.NaN
    else:
        delta_flow_out = df_gc["Eapp [V]"][idx] - df_gc["Eapp [V]"][idx - 1]
        new_col_data = delta_flow_out
    return new_col_data


def dummy_print_dir_nc(data_package: main.DataPackage):
    """This is a dummy function that prints the directory of the datagram."""
    dir_datagram = data_package.dir_datagram
    df = xr.open_dataset(dir_datagram, group="electro", engine="h5netcdf")
    new_col_data = df["uts"][0].item()
    return new_col_data


def delta_Eapp_fluctuation(data_package: main.DataPackage):
    """The fluctuation (standard deviation) of the applied potential between the previous GC injection to the current GC injection.

    ** Regardless of Ewe or Eapp being used, the value would be the same since the correction for the potential would be the same
    in all of the point in the same experiment. The standard deviation would be the same.
    """
    df_electro, idx_start, idx_end, _, _ = aux.unpack_data_cc(data_package, "delta")
    ls_sel_val = []
    ls_sel_val = [
        df_electro["Ewe [V]"][idx]
        for idx in range(idx_start, idx_end + 1)
        if not (-1 < df_electro["Ewe [V]"][idx] < -0.5)
    ]
    if len(ls_sel_val) < 0:
        new_col_data = np.NaN
    else:
        new_col_data = np.std(ls_sel_val)
    return new_col_data
