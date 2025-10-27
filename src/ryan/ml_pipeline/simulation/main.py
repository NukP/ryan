import pandas as pd
from typing import List, Tuple, Any, Union, Dict
from dataclasses import dataclass
from . import gen_functions as gen
import os
import logging
import time


# Defaul column name list, used by generate_dataframe function.
default_ls_column_names = [
    "global_time",
    "I (mA)",
    "global_Q",
    "delta_Q",
    "global_time_positive_i",
    "delta_time_positive_i",
    "global_time_negative_i",
    "delta_time_negative_i",
    "delta_q_pos_i",
    "global_q_pos_i",
    "delta_q_neg_i",
    "global_q_neg_i",
    "global_q_posneg_ratio",
    "delta_i_neg_avg",
    "delta_i_pos_avg",
    "delta_i_posneg_avg_ratio",
]


@dataclass
class DataPackage:
    """
    A data class for packaging the parameters needed for main.simulation and data generation functions in gen_functions
    module.

    Attributes:
        global_time (float): The current global time for the simulation. This should be greater
                             than the last time value in the DataFrame to avoid ValueError.
        current (float): The current in mA used in the simulation.
        cutoff_current (float): The cutoff current value used to determine the positive and negative currents.
        df (pd.DataFrame): A DataFrame containing historical simulation data.
        delta_t (float): The difference between the current global time and the last time value in the DataFrame.
                          This is a read-only property.
    """

    global_time: float
    current: float
    cutoff_current: float
    df: pd.DataFrame

    @property
    def delta_t(self):
        return self.global_time - self.df["global_time"].iloc[-1]


def simulate(
    global_time: float, current: float, path_df: Union[str, pd.DataFrame] = None, export_excel: bool = False, cutoff_current: float = -1
) -> pd.DataFrame:
    """
    Calculate the input features for the model from the input time (global_time) and current.

    Args:
        global_time (float): The time value for which the input features are calculated.
        current (float): The current 'I(mA)' value used to calculate the input features.
        path_df (Union[str, pd.DataFrame]): The file path to the DataFrame containing historical data if export_excel is True,
                                             otherwise the DataFrame itself.
        export_excel (bool): Flag indicating whether to export the DataFrame to an Excel file.
        cutoff_current (float): The cutoff current value used to determine the positive and negative currents. The default value is -1.

    Returns:
        pd.DataFrame: The DataFrame containing the calculated input features.

    Raises:
        FileNotFoundError: If the file path to the DataFrame for simulation log does not exist.
        ValueError: If export_excel is not a boolean value or if the global time value is less than or equal to the last global time value in the DataFrame.
    """
    if export_excel and not os.path.exists(path_df):
        raise FileNotFoundError("The file path to the DataFrame for simulation log does not exist.")
    elif export_excel is True:
        df = pd.read_excel(path_df)
    elif not export_excel:
        df = path_df
    else:
        raise ValueError("export_excel must be a boolean value.")
    if export_excel is True:
        if not df.empty and df["global_time"].iloc[-1] >= global_time:
            raise ValueError("Invalid time value. The global time must be greater than the last global time in the DataFrame.")

    dict_for_concat: Dict[str, Any] = {}
    dict_for_concat.update({"global_time": global_time})
    dict_for_concat.update({"I (mA)": current})
    data_package = DataPackage(global_time, current, cutoff_current, df)
    ls_column_to_gen = [column for column in df.columns if column not in ["I (mA)", "global_time"]]
    for column in ls_column_to_gen:
        dict_for_concat.update({column: getattr(gen, f"gen_{column}")(data_package)})
    df_to_concat = pd.DataFrame(dict_for_concat, index=[0])
    if export_excel is True:
        df = pd.concat([df, df_to_concat], ignore_index=True)
        df.to_excel(path_df, index=False)
    return df_to_concat


def generate_dataframe(
    fl_excel_name: str = None, export_excel: bool = False, ls_column_names: List[str] = default_ls_column_names
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame with the given column names and export it to an Excel file.

    This function is used to generate the initial DataFrame for the simulate function.

    Args:
        fl_excel_name (str, optional): The name of the Excel file to save the DataFrame. Required if export_excel is True.
        export_excel (bool): Flag indicating whether to export the DataFrame to an Excel file.
        ls_column_names (List[str], optional): A list of column names for the DataFrame. Defaults to default_ls_column_names.

    Returns:
        pd.DataFrame: The generated DataFrame.
    """
    if export_excel and fl_excel_name is None:
        raise ValueError("fl_excel_name is required when export_excel is True.")

    data = [[0] * len(ls_column_names)]
    df = pd.DataFrame(data, columns=ls_column_names)

    if export_excel:
        df.to_excel(fl_excel_name, index=False)

    return df


def run_sim_exp(
    ls_tuple_t_i: List[Tuple[int, int]],
    n_e_per_mol_product: float,
    obj_model: Any,
    obj_scaler: Any,
    path_df: str,
    export_excel: bool = False,
) -> pd.DataFrame:
    """
    Run simulation experiment for given parameters.

    Args:
        ls_tuple_t_i (List[Tuple[int, int]]): List of tuples containing time and current values.
        n_e_per_mol_product (float): Number of electrons per mole of product. e.g. for ethylene, the value is 12.
        obj_model (Any): Model object used for prediction.
        obj_scaler (Any): Scaler object used for transforming data, make the input suitable for the model.
        path_df (str): Path to the dataframe. This dataframe is used to store the historical data from the prediction.
        export_excel (bool): Flag indicating whether to export the DataFrame to an Excel file. This parameter is not used in the function, but it is used in simulate function.

    Returns:
        pd.DataFrame: DataFrame containing the simulation experiment results.
    """
    results = []  # List to accumulate results

    for t, i in ls_tuple_t_i:
        df_from_sim = simulate(t, i, path_df, export_excel)
        if not export_excel:
            path_df = pd.concat([path_df, df_from_sim], ignore_index=True)

        fe = obj_model.predict(obj_scaler.transform(df_from_sim))[0]

        if df_from_sim["delta_Q"].iloc[-1] < 0:
            delta_product_produced = abs(df_from_sim["delta_Q"].iloc[-1]) * fe * n_e_per_mol_product / 96485
        else:
            delta_product_produced = 0

        cumulative_product = delta_product_produced if not results else results[-1]["cumulative_product"] + delta_product_produced

        # Append results as a dictionary
        results.append(
            {
                "global_t": t,
                "current": i,
                "fe": fe,
                "delta_product_produced": delta_product_produced,
                "cumulative_product": cumulative_product,
            }
        )

    # Convert list of dictionaries to DataFrame
    df_sim_exp = pd.DataFrame(results)

    return df_sim_exp


def training_manager(
    dir_saving: str, electrochemical_protocol: List[Tuple[float, float]], export_excel: bool = False, **kwargs_for_sun_sim_exp: Any
) -> float:
    """
    This function will take the tuple representing the electrochemical protocol (time, current) and return the total
    product produced.

    This function will be used to train the ML model agent for finding the optimal electrochemical protocol to get the highest product produced in the given time.
    The ML model agent will interact with this function. This function will also save the historical data in the excel file for both the simulated values and the predicted values
    from the model in the Excel format. The Excel file will be saved in the folder in the given path.

    Args:
        dir_saving (str): The directory path to save the Excel files.
        electrochemical_protocol (List[Tuple[float, float]]): List of tuples containing time and current values. This list will be passed to the run_sim_exp function.
        export_excel (bool): Flag indicating whether to export the DataFrame to an Excel file.
        **kwargs_for_sun_sim_exp (Any): The keyword arguments for the run_sim_exp function.

    Returns:
        float: The total product produced.
    """
    try:
        if export_excel is True:
            # Create the directory for saving the Excel files.
            dir_simulation_history = os.path.join(dir_saving, "Simulation_history")
            os.makedirs(dir_simulation_history, exist_ok=True)
            dir_model_logs = os.path.join(dir_saving, "Model_logs")
            os.makedirs(dir_model_logs, exist_ok=True)

            # Assign num_iteration from the last file index in Simulation history folder
            num_iteration = len(os.listdir(dir_simulation_history)) + 1
            current_time = int(time.time())
            logging.info(f"The current iteration is: {num_iteration} at Unix Time: {current_time}")
            print(f"The current iteration is: {num_iteration} at Unix Time: {current_time}")

            # Generate file names with Unix timestamp
            simulation_history_filename = f"simulation_history_{num_iteration}_atUTX_{current_time}.xlsx"
            dir_simulation_history_path = os.path.join(dir_simulation_history, simulation_history_filename)
            generate_dataframe(fl_excel_name=dir_simulation_history_path, export_excel=export_excel)
            df_sim = run_sim_exp(
                ls_tuple_t_i=electrochemical_protocol,
                path_df=dir_simulation_history_path,
                export_excel=export_excel,
                **kwargs_for_sun_sim_exp,
            )
            model_log_filename = f"model_log_{num_iteration}_atUTX_{current_time}.xlsx"
            df_sim.to_excel(os.path.join(dir_model_logs, model_log_filename), index=False)
        elif not export_excel:
            df_sim = run_sim_exp(
                ls_tuple_t_i=electrochemical_protocol,
                path_df=generate_dataframe(export_excel=export_excel),
                **kwargs_for_sun_sim_exp,
            )
            pass
        else:
            raise ValueError("export_excel must be a boolean value.")

        return df_sim["cumulative_product"].iloc[-1]

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
