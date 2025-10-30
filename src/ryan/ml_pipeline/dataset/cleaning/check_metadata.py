"""
This modulke contains functions that will be used to check the validiity of the metadata files. I will make this module
now compatible for metadata v2.

Check list are as follows:
1. When substrate is PTFE, the value of pore size = 0.3.
2. When substrate is PVDF, the value of pore size should be 0.3 or 0.8.
3. When substrate is PVDF-HFP, the value of pore size can be 0.2, 0.7, 1.1 (IS THAT CORRECT).
4. When [Cu, 1] is the catalyst, loading is 0.5.
5. When [Ag, 1] is the catalyst, loading is 0.5.
6. When cathode electrolyte is [KHCO3, 1] the cathode pH must be 7.8.
7. When cathode electrolyte is [KCl, 1] or [KCl, 2], the cathode pH must be 4.5 (IS THAT CORRECT).
8. When cathode electrolyte is [H2SO4,1], the cathode pH must be 0.
9. When anode electrolyte is [H2SO4,1], the anode pH must be 0.
10. When cathode electrolyte is [H2SO4,0.1], the cathode pH must be 1.
11. When anode electrolyte is [H2SO4,0.1], the anode pH must be 1.
12. When anode electrolyte is [KHCO3, 1] the anode pH must be 7.8 IF THERE IS CO2 FLOW.
13. When anode electrolyte is [KHCO3, 1] the anode pH must be 8.5 WITH NO CO2 FLOW.
14. When anode is Commercial IrOx on Ti, anode catalyst component should be [IrOx, 0.2], [TaOx, 0.8], anode catalyst loading should be 1 mg/cm², anode substrate components should be [Ti, 2000, 1000].
15. When anode is Electrodeposited IrOx on Ti, anode catalyst component should be [IrOx, 1], anode catalyst loading should be empty, anode substrate components should be [Ti, 100, 500].
16. For GC, the %FE for C2H4 must be in [0%, 60%]. Flag which injection and experiment the concentration is outside this range.
17. For GC, the %FE for C3H8 must be in [0%, 5%]. Flag which injection and experiment the concentration is outside this range.
18. For GC, the %FE for CH4 must be in [0%, 70%]. Flag which injection and experiment the concentration is outside this range.
19. CO experiments are treated differently: CO2RR requires C: 4 in dgpost GC and LC recipe, CORR requires C: 2.
20. For typos in metadata, export an Excel file highlighting the values in all metadata and their frequency to quickly identify typos.
21. Check the date in the name of the file and the date in the Experiment date entry.
22. Cathode geometric surface area should be 1 or 0.2 cm².
"""

import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from .utility import convert_to_dict, get_value_from_df


def check_condition_1(df_metadata: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 1 is met:

    - When the substrate is PTFE, the value of pore size should be 0.3 µm.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    values = get_value_from_df(
        df_metadata, "Cathode microporous layer components [chemical formula, pore size in µm, thickness in µm]"
    )
    dict_cathode = convert_to_dict(values, "chemical_composiiton", "pore_size", "thickness")
    if dict_cathode["chemical_composiiton"] == "PTFE":
        if float(dict_cathode["pore_size"]) == 0.3:
            package = (True, None)
        else:
            package = (False, f"Pore size should be 0.3, instead it is {dict_cathode['pore_size']}")
        return package


def check_condition_2(df_metadata: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 2 is met:

    - When the substrate is PVDF, the value of pore size should be 0.3 µm or 0.8 µm.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    values = get_value_from_df(
        df_metadata, "Cathode microporous layer components [chemical formula, pore size in µm, thickness in µm]"
    )
    dict_cathode = convert_to_dict(values, "chemical_composiiton", "pore_size", "thickness")
    if dict_cathode["chemical_composiiton"] == "PVDF":
        if float(dict_cathode["pore_size"]) == 0.3 or float(dict_cathode["pore_size"]) == 0.8:
            package = (True, None)
        else:
            package = (False, f"Pore size should be 0.3 or 0.8, instead it is {dict_cathode['pore_size']}")
        return package


def check_condition_3(df_metadata: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 3 is met:

    - When the substrate is PVDF-HFP, the value of pore size can be 0.2 µm, 0.7 µm, or 1.1 µm.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    values = get_value_from_df(
        df_metadata, "Cathode microporous layer components [chemical formula, pore size in µm, thickness in µm]"
    )
    dict_cathode = convert_to_dict(values, "chemical_composiiton", "pore_size", "thickness")
    if dict_cathode["chemical_composiiton"] == "PVDF-HFP":
        if (
            float(dict_cathode["pore_size"]) == 0.2
            or float(dict_cathode["pore_size"]) == 0.7
            or float(dict_cathode["pore_size"]) == 1.1
        ):
            package = (True, None)
        else:
            package = (False, f"Pore size should be 0.2, 0.7 or 1.1 instead it is {dict_cathode['pore_size']}")
        return package


# Need clarification.
def check_condition_4(df_metadata: DataFrame):
    """
    Check if condition 4 is met:
    - When [Cu, 1] is the catalyst, loading should be 0.5.
    If the condition is met, return True. Otherwise, return False.
    """
    if get_value_from_df(df_metadata, "Cathode catalyst components [chemical formula, mol fraction]") == "[Cu, 1]":
        if float(get_value_from_df(df_metadata, "Cathode catalyst loading ")) == 0.5:
            package = (True, None)
        else:
            incorrect_value = get_value_from_df(df_metadata, "Cathode catalyst loading ")
            package = (
                False,
                f"Loading should be 0.5, instead it is {incorrect_value}",
            )
        return package


def check_condition_5(df_metadata: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 5 is met:

    - When [Ag, 1] is the catalyst, the loading should be 0.5.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    if get_value_from_df(df_metadata, "Cathode catalyst components [chemical formula, mol fraction]") == "[Ag, 1]":
        if float(get_value_from_df(df_metadata, "Cathode catalyst loading ")) == 0.5:
            package = (True, None)
        else:
            package = (
                False,
                f"Loading should be 0.5, instead it is {get_value_from_df(df_metadata, 'Cathode catalyst loading ')}",
            )
        return package


def check_condition_6(df_metadata: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 6 is met:

    - When the cathode electrolyte is [KHCO3, 1], the cathode pH should be 7.8.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    if (
        get_value_from_df(df_metadata, "Cathode electrolyte compartment solute content [name, concentration in M]")
        == "[KHCO3, 1]"
    ):
        cathode_pH = float(get_value_from_df(df_metadata, "Cathode compartment electrolyte pH - SET"))
        if cathode_pH == 7.8:
            package = (True, None)
        else:
            package = (
                False,
                f"The pH should be 7.8, instead it is {cathode_pH}",
            )
        return package


def check_condition_7(df_metadata: pd.DataFrame):
    """
    Check if condition 7 is met:

    - When the cathode electrolyte is [KCl, 1], the cathode pH should be 3.8.
    - When the cathode electrolyte is [KCl, 2], the cathode pH should be 3.7.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    cathode_electrolyte = get_value_from_df(
        df_metadata, "Cathode electrolyte compartment solute content [name, concentration in M]"
    )
    cathode_pH = float(get_value_from_df(df_metadata, "Cathode compartment electrolyte pH - SET"))

    # Initialize the package as True by default
    package = (True, None)

    # Check for [KCl, 1]
    if cathode_electrolyte == "[KCl, 1]":
        if cathode_pH != 3.8:  # pH should be 3.8 for [KCl, 1]
            package = (False, f"The pH should be 3.8 for [KCl, 1], instead it is {cathode_pH}")
        return package

    # Check for [KCl, 2]
    if cathode_electrolyte == "[KCl, 2]":
        if cathode_pH != 3.7:  # pH should be 3.7 for [KCl, 2]
            package = (False, f"The pH should be 3.7 for [KCl, 2], instead it is {cathode_pH}")
        return package

    return package


def check_condition_8(df_metadata: DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 8 is met:

    - When the cathode electrolyte is [H2SO4, 1], the cathode pH should be 0.

    Args:
        df_metadata (DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    cathode_electrolyte = get_value_from_df(
        df_metadata, "Cathode electrolyte compartment solute content [name, concentration in M]"
    )
    if "[H2SO4, 1]" in cathode_electrolyte:
        cathode_pH = float(get_value_from_df(df_metadata, "Cathode compartment electrolyte pH - SET"))
        if cathode_pH == 0:
            package = (True, None)
        else:
            package = (
                False,
                f"The pH should be 0, instead it is {cathode_pH}",
            )
        return package


def check_condition_9(df_metadata: DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 9 is met:

    - When the anode electrolyte is [H2SO4, 1], the anode pH should be 0.

    Args:
        df_metadata (DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    anode_electrolyte = get_value_from_df(
        df_metadata, "Anode electrolyte compartment solute content [name, concentration in M]"
    )
    if "[H2SO4, 1]" in anode_electrolyte:
        anode_pH = float(get_value_from_df(df_metadata, "Anode compartment electrolyte pH - SET"))
        if anode_pH == 0:
            package = (True, None)
        else:
            package = (
                False,
                f"The pH should be 0, instead it is {anode_pH}",
            )
        return package


def check_condition_10(df_metadata: DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 10 is met:

    - When the cathode electrolyte is [H2SO4, 0.1], the cathode pH should be 1.

    Args:
        df_metadata (DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    cathode_electrolyte = get_value_from_df(
        df_metadata, "Cathode electrolyte compartment solute content [name, concentration in M]"
    )
    if "[H2SO4, 0.1]" in cathode_electrolyte:
        cathode_pH = float(get_value_from_df(df_metadata, "Cathode compartment electrolyte pH - SET"))
        if cathode_pH == 1:
            package = (True, None)
        else:
            package = (
                False,
                f"The pH should be 1, instead it is {cathode_pH}",
            )
        return package


def check_condition_11(df_metadata: DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check if condition 11 is met:

    - When the anode electrolyte is [H2SO4, 0.1], the anode pH should be 1.

    Args:
        df_metadata (DataFrame): DataFrame containing the metadata information.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the condition is met
        and an optional string with an error message if the condition is not met.
    """
    anode_electrolyte = get_value_from_df(
        df_metadata, "Anode electrolyte compartment solute content [name, concentration in M]"
    )
    if "[H2SO4, 0.1]" in anode_electrolyte:
        anode_pH = float(get_value_from_df(df_metadata, "Anode compartment electrolyte pH - SET"))
        if anode_pH == 1:
            package = (True, None)
        else:
            package = (
                False,
                f"The pH should be 1, instead it is {anode_pH}",
            )
        return package


def check_condition_12(df_metadata: DataFrame):
    """
    Check if condition 12 is met:
    - When anode electrolyte is [KHCO3, 1], the anode pH should be 7.8 IF THERE IS CO2 FLOW.
    If the condition is met, return True. Otherwise, return False.
    """
    anode_gas_presence = get_value_from_df(df_metadata, "Anode gas present?")
    if anode_gas_presence == "Yes" and "[KHCO3, 1]" in get_value_from_df(
        df_metadata, "Anode electrolyte compartment solute content [name, concentration in M]"
    ):
        anode_electrolyte_ph = get_value_from_df(df_metadata, "Anode compartment electrolyte pH - SET")
        if anode_electrolyte_ph == 7.8:
            package = (True, None)
        else:
            package = (False, f"Anode pH should be 7.8, instead it is {anode_electrolyte_ph}")
        return package
    pass


# There is no files where there is no Flow.
def check_condition_13(df_metadata: DataFrame):
    """
    Check if condition 13 is met:
    - When anode electrolyte is [KHCO3, 1], the anode pH should be 8.5 WITH NO CO2 FLOW.
    If the condition is met, return True. Otherwise, return False.
    """
    pass


def check_condition_14(df_metadata: DataFrame):
    """
    Check if condition 14 is met:
    - When anode is Commercial IrOx on Ti, the anode catalyst component should be [IrOx, 0.2], [TaOx, 0.8],
      anode catalyst loading should be 1 mg/cm², anode substrate components should be [Ti, 2000, 1000].
    If the condition is met, return True. Otherwise, return False.
    """
    anode_cat_name = get_value_from_df(df_metadata, "Anode catalyst name")
    if anode_cat_name == "Commercial IrOx on Ti":
        anode_cat_component = get_value_from_df(
            df_metadata, "Anode catalyst components [chemical formula, mol fraction]"
        )
        anode_cat_loading = str(get_value_from_df(df_metadata, "Anode catalyst loading"))
        anode_sub_component = get_value_from_df(
            df_metadata, "Anode microporous layer components [chemical formula, pore size in µm, thickness in µm]"
        )
        dict_issue = {}
        if anode_cat_component == "[IrOx, 0.2], [TaOx, 0.8]":
            dict_issue["anode catalyst component"] = True
        else:
            dict_issue["anode catalyst component"] = False
        if anode_cat_loading == "1":
            dict_issue["anode catalyst loading"] = True
        else:
            dict_issue["anode catalyst loading"] = False
        if anode_sub_component == "[Ti, 2000, 1000]":
            dict_issue["anode substrate component"] = True
        else:
            dict_issue["anode substrate component"] = False

        if all(dict_issue.values()):
            package = (True, None)
        else:
            issues = []
            for key, value in dict_issue.items():
                if not value:
                    issues.append(key)
            package = (False, f"Following issues were found: {', '.join(issues)}")
        return package


def check_condition_15(df_metadata: DataFrame):
    """
    Check if condition 15 is met:
    - When anode is Electrodeposited IrOx on Ti, the anode catalyst component should be [IrOx, 1],
      anode catalyst loading should be empty, anode substrate components should be [Ti, 100, 500].
    If the condition is met, return True. Otherwise, return False.
    """
    anode_cat_name = get_value_from_df(df_metadata, "Anode catalyst name")
    if anode_cat_name == "Electrodeposited IrOx on Ti":
        anode_cat_component = get_value_from_df(
            df_metadata, "Anode catalyst components [chemical formula, mol fraction]"
        )
        anode_cat_loading = str(get_value_from_df(df_metadata, "Anode catalyst loading"))
        anode_sub_component = get_value_from_df(
            df_metadata, "Anode microporous layer components [chemical formula, pore size in µm, thickness in µm]"
        )
        dict_issue = {}
        if anode_cat_component == "[IrOx, 1]":
            dict_issue["anode catalyst component"] = True
        else:
            dict_issue["anode catalyst component"] = False
        if anode_cat_loading == "0.2":
            dict_issue["anode catalyst loading"] = True
        else:
            dict_issue["anode catalyst loading"] = False
        if anode_sub_component == "[Ti, 100, 500]":
            dict_issue["anode substrate component"] = True
        else:
            dict_issue["anode substrate component"] = False

        if all(dict_issue.values()):
            package = (True, None)
        else:
            issues = []
            for key, value in dict_issue.items():
                if not value:
                    issues.append(key)
            package = (False, f"Following issues were found: {', '.join(issues)}")
        return package


# Condition 16-18 are for GC data and not metadata.

# Condition 19, Fixed already in dynamic recipe generation in AutoplotDB.

# Condition 20 is about typo in metadata, I will write a separate function for this.


def check_condition_21(df_metadata: pd.DataFrame):
    """
    Check if condition 21 is met:
    - Check the date in the name of the file and the date in the Experiment date entry.
    If the condition is met, return (True, None). Otherwise, return (False, <error message>).
    """
    try:
        filename = get_value_from_df(df_metadata, "Experiment name")
        experiment_date = get_value_from_df(df_metadata, "Experiment date")

        # Updated regex pattern to match date in yyyymmdd format
        date_pattern = r"(?<!\d)(\d{8})(?!\d)"
        match = re.search(date_pattern, filename)

        if match:
            file_date_str = match.group(1)
            file_date = datetime.strptime(file_date_str, "%Y%m%d")

            if isinstance(experiment_date, str):
                try:
                    exp_date = datetime.strptime(experiment_date, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return (False, "There are likely a typo in the experiment date")
            elif isinstance(experiment_date, datetime):
                exp_date = experiment_date
            else:
                return (False, "Invalid experiment date format")

            if file_date.date() == exp_date.date():
                package = (True, None)
            else:
                package = (False, "The date in the filename and the specified experiment date are not matching")
        else:
            package = (False, "No valid date found in the filename")

    except Exception as e:
        package = (False, f"Error encountered: {str(e)}")

    return package


def check_condition_22(df_metadata: DataFrame):
    """
    Check if condition 22 is met:
    - Cathode geometric surface area should be 1 or 0.2 cm².
    If the condition is met, return True. Otherwise, return False.
    """
    cathode_geo_area = get_value_from_df(df_metadata, "Cathode geometric surface area")
    if cathode_geo_area == 1 or cathode_geo_area == 0.2:
        package = (True, None)
    else:
        package = (False, f"Cathode geometric surface area should be 1 or 0.2 cm², instead it is {cathode_geo_area}")
    return package
