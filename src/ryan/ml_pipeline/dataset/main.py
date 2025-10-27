import os
import pandas as pd
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from . import auxiliary as aux
import time
from tqdm.notebook import tqdm
import warnings

# add here to suppress the warning from openpyxl
warnings.filterwarnings("ignore", message="Data Validation extension is not supported and will be removed", module="openpyxl.*")


@dataclass
class DataPackage:
    """A data class for packing the data from gen_dataset function to be used by many custom function in
    custom_functions module."""

    df_gc: pd.DataFrame
    idx: int
    fln: str
    dir: str
    df_electro: pd.DataFrame = None  # type: ignore
    fl_dir: str = None  # type: ignore
    fl_electro_dir: str = None  # type: ignore
    #dir_datagram: str = None  # type: ignore

    def __post_init__(self):
        self.fl_dir = os.path.join(self.dir, self.fln)
        fln_electro = self.fln.split(".")[0] + ".electro.xlsx"
        self.fl_electro_dir = os.path.join(self.dir, fln_electro)
        #self.dir_datagram = os.path.join(self.dir, self.fln.split(".")[0] + ".nc")


def mod_df(fl_excel: str) -> pd.DataFrame:
    """
    This creates a respective dataframe from the given GC Excel file. The function modify the header that originally
    comes in two row format and containing multiple merge-cell header to be easier to work with so that further data
    prasing can be done more easily.

    Args:
        fl_excel (str): Path of the GC Excel file from yadg/dgpost. These Excel files contains raw data from the experiment.

    Returns:
        pd.DataFrame: A modified dataframe from the given Excel file.
    """
    dataframe = pd.read_excel(fl_excel, header=[0, 1])
    modified_headers = []
    for col in dataframe.columns:
        if col == dataframe.columns[0]:
            new_header = "time"
        elif col[1] and not pd.isna(col[1]) and "Unnamed" not in col[1]:
            new_header = f"{col[0]}-{col[1]}"
        else:
            new_header = col[0]
        modified_headers.append(new_header)
    dataframe.columns = modified_headers
    dataframe = dataframe.iloc[2:].reset_index(drop=True)
    return dataframe


def gen_all_excel_ls(excel_fol_dir: str) -> list:
    """
    This function returns a list of all of the Excel files that ends with GCdata.xlsx.

    Args:
        excel_fol_dir (str): Path to the folder containing the excel files

    Returns:
        list:  List containing the file name of the excel files that ends with GCdata.xlsx in the given excel_fol_dir
    """
    ls_gcexcel_all = [fl_excel for fl_excel in os.listdir(excel_fol_dir) if fl_excel.endswith("GCdata.xlsx")]
    return ls_gcexcel_all


def condition_fln_check_word(item: str, word: str) -> bool:
    """
    This function is use as part of the condition for filtering the filename. Check whether there is the given word is
    contained in any of part of the original string (item) that is separtaed by -. The funnction returns bool.

    Args:
        item (str): Item to be chceked.
        word (str): Targeted word.

    Returns:
        bool: True if the target word contains in the item.
    """
    return word in item.split("-")


def gen_fil_excel_ls(conditions_to_check: list, ls_pre_fil: list) -> list:
    """
    This function take the list of all of the GC Excel files and filter out according to the given conditions. The
    function returns list of file name of the GC Excel file whcih pass the given conditions.

    Args:
        conditions_to_check (list): List contains filtering condition.
        ls_pre_fil (list): List contains file name of GC Excel file to be filtered.

    Returns:
        list: List containing filtered GC Excel file.
    """

    def check_all_conditions(item):
        return all(condition(item) for condition in conditions_to_check)

    ls_fln_filtered = [fln for fln in ls_pre_fil if check_all_conditions(fln)]
    return ls_fln_filtered


def condition_idx_check_fe_exceed(df: pd.DataFrame, idx: int, treshold: int = 1) -> bool:
    """
    This function is used as part of the condition for checking each gc injection point (idx check). This function check
    if overall Faradaic efficiency (gas) from the GC is exceeding the given trehold or not.

    Args:
        df (pd.DataFrame): adjusted data frame generated from mod_df function.
        idx (int): index inside the dataframe. Each index represent one GC injection point.

    Returns:
        bool: True if the sum of overall fe is not exceeding the given treshold. The default for the treshold is set to 1.
    """
    ls_col_fe = [col for col in df.columns if col.split("-")[0] == "fe"]
    return sum([df[name][idx] for name in ls_col_fe if not pd.isna(df[name][idx])]) < treshold


def gen_fil_idx_ls(conditions_idx: list, df: pd.DataFrame) -> list:
    """
    This function generate a list of index in which that given parameters in df[idx] passes the given conditions.

    Args:
        conditions_idx (list): List contain index filtering conditions.
        df (pd.DataFrame): adjusted data frame generated from mod_df function.

    Returns:
        list: a list of index in which that given parameters in df[idx] passes the given conditions.
    """

    def check_all_conditions_idx(df, idx):
        return all(con(df, idx) for con in conditions_idx)

    ls_filtered_idx = [idx for idx in range(0, len(df)) if check_all_conditions_idx(df, idx)]
    return ls_filtered_idx


def gen_dataset(
    gc_excel_dir: str,
    conditions_fln: list = [],
    conditions_idx: list = [],
    ls_input_extract: list = [],
    ls_input_gen: list = [],
    ls_output_extract: list = [],
    ls_output_gen: list = [],
    pre_gen_electro_df: bool = True,
    export_df: bool = False,
    num_cpu: int = 1,
) -> pd.DataFrame:
    """
    This function generate dataset from filtered file name and indexes. The function can both extarct the data from the
    raw data frame and supoort generating new data from the raw data. The function return the dataset in the form of
    pd.Dataframe with a possbility to export the dataset as Excel.

    For pre_gen_electro_df: The electro.xlsx is requried in many custom functions for feature engineering, but the file is quite large.
    Therefore, it takes quite sometime to load. This electro file is not requried if no feature engineering will be executed and one only
    intend to extract the feature from the already-existing parameteres in the GC.xlsx file.
    During the loading of dataframe, all of the cpu cores will be used. But this usually does not take long. For the steps where the value of each
    value of the row are generated (where aux.process_row are called), This tend to be longer and in some cases, RAM will be full and slow normal OS operation down.
    It also found that when this happens, it takes even longer than using 1 cpu cores due to the memory bottle neck. This only occurs when many files are loaded.
    For small number of files, using all cpu cores delivers faster calculation. The number of cpu cores used in this later process can be specified by specified in
    num_cpu args. Note that this only control the number of cpu used in aux.process_row and not during the dict_df loading.

    Args:
        gc_excel_dir (str): Path to a folder containing a GC Excel files.
        conditions_fln (list, optional): List containing functions for filtering file name. Defaults to [].
        conditions_idx (list, optional): List containing functions for filtering row (index). Defaults to [].
        ls_input_extract (list, optional): List of column to be extracted from the respective GC Excel file.
            These colums will be placed first as they are regard as the input for the dataset (for machine learning purpose). Defaults to [].
        ls_input_gen (list, optional): List of generated input column. These colums will be placed second. Defaults to [].
        ls_output_extract (list, optional): List of column to be extracted from the respective GC Excel file.
             The columns here are regards as the output for machine learning. These columns will be placed third. Defaults to [].
        ls_output_gen (list, optional): List of generated output column. These columns will be placed last. Defaults to [].
        pre_gen_electro_df (bool, optional): If True, read the electro.xlsx files into the dict_df
        export_df (bool, optional): If specified, the dataset will be exported as excel file at the given path. Defaults to False.
        num_cpu (float, optional): Number of cpu cores used for calling aux.process_row. If not specified, all of the cpu cores will be used.

    Returns:
        pd.DataFrame: Data dframe containg the dataset for Machine learning.
    """
    from .custom_columns import process_feature

    # Create dummy df to call the custom function in order to extract gen column name.
    dummy_package = DataPackage(df_gc=pd.DataFrame(), idx="dummy", fln="dummy", dir="dummy")  # type: ignore

    ls_input_gen_name = []
    for gen_func in ls_input_gen:
        col_name, _ = process_feature(gen_func, dummy_package)
        ls_input_gen_name.append(col_name)

    ls_output_gen_name = []
    for gen_func in ls_output_gen:
        col_name, _ = process_feature(gen_func, dummy_package)
        ls_output_gen_name.append(col_name)

    ls_header = ls_input_extract + ls_input_gen_name + ls_output_extract + ls_output_gen_name
    df_dataset = pd.DataFrame(columns=ls_header)

    # Filtering gc_excel files using keyword in the file name.
    ls_gcexcell_all = gen_all_excel_ls(gc_excel_dir)
    ls_gcexcel_fil = gen_fil_excel_ls(conditions_fln, ls_gcexcell_all)
    ls_electroexcel = [fl.split(".")[0] + ".electro.xlsx" for fl in ls_gcexcel_fil]

    if pre_gen_electro_df:
        list_fl = ls_gcexcel_fil + ls_electroexcel
    else:
        list_fl = ls_gcexcel_fil
    dict_df = {}
    start_time = time.time()
    print("Start loading associated Excel files process.")
    with Pool(processes=cpu_count()) as pool:
        args_list = [(fln, gc_excel_dir) for fln in list_fl]
        for (fln, df) in tqdm(pool.imap_unordered(aux.read_excel, args_list), total=len(args_list)):
            dict_df[fln] = df
    print("Excel loading completed.\n")
    end_time = time.time()
    run_h, run_m, run_s = aux.convert_seconds(end_time - start_time)  # type: ignore
    print(f"Time required for generating dic_df: {run_h:.0f} hours, {run_m:.0f} minutes and {run_s:.2f} seconds.\n")

    start_time = time.time()

    # Filtering idx in each file.
    print("Start extracting and calculating values for features for each Index. \n")
    ls_flidx_to_process = []
    for pass_fl in ls_gcexcel_fil:
        ls_fil_idx = gen_fil_idx_ls(conditions_idx, dict_df[pass_fl.split(".xlsx")[0]])

        for idx in ls_fil_idx:
            ls_flidx_to_process.append((pass_fl, idx))
    total_rows = len(ls_flidx_to_process)
    process_args = [
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
        )
        for pass_fl, idx in ls_flidx_to_process
    ]

    if not num_cpu:
        num_cpu = cpu_count()

    with Pool(processes=num_cpu) as pool:
        all_row_data = []
        for row_data in tqdm(pool.imap_unordered(aux.process_row, process_args), total=total_rows):  # type: ignore
            all_row_data.append(row_data)

    df_dataset = pd.DataFrame(all_row_data, columns=ls_header)
    end_time = time.time()
    run_h, run_m, run_s = aux.convert_seconds(end_time - start_time)  # type: ignore
    print(f"Time required for populating features: {run_h:.0f} hours, {run_m:.0f} minutes and {run_s:.2f} seconds.")

    if export_df:
        df_dataset.to_excel(export_df, index=False)
    return df_dataset
