"""
This moduel contain function that will be used for generating an Exel table listing differences between the metadata
files.

This Excel file will be used to generate a Sankey flow diagram. For now, this module will be mainly used with ML31
workbook.
"""

import os

import pandas as pd
from ml_co2.dataset.cleaning import utility as util


def extract_metadata(
    list_metadata: list,
    column_name_in_metadata_file: str,
    prefered_column_name: str,
    idecies_from_pseudolist: list = None,  # type: ignore
    enforce_type_float: bool = False,
) -> tuple:
    """
    This function is a basic metadata extraction function.

    This function is meant to be used if the metadata is simple and no additional adjustment is requried.
    If there has to be any additional adjustment, then a new function should be created.
    The function will return a tuple of (prefered_column_name, list_of_values)

    Args:
    list_metadata: list of directory of the metadata files to be extracted
    column_name_in_metadata_file: The name of the column in the metadata file (to be extracted)
    prefered_column_name: The name of the column that will be used to store the metadata
    enforce_type_float: If the extracted value should be converted to float, if this is True, then the extracted value will be converted to float

    return:
    tuple: (prefered_column_name, list_of_values)
    """
    ls = []
    for metadata in list_metadata:
        try:
            if idecies_from_pseudolist is None:
                metadata_value = util.get_value_from_excel(metadata, column_name_in_metadata_file)
            else:
                psuedo_list = util.process_pseudo_list(
                    util.get_value_from_excel(metadata, column_name_in_metadata_file)
                )
                metadata_value = util.get_element_from_pseudolist(psuedo_list, idecies_from_pseudolist)
            if enforce_type_float:
                ls.append(float(metadata_value))
            else:
                ls.append(metadata_value)
        except:
            raise ValueError(f"There is an error with the file {metadata}")
    return (prefered_column_name, ls)


def get_cathode_catalyst_comp(list_metadata: list):
    ls = []
    for metadata in list_metadata:
        try:
            returned_string = util.get_value_from_excel(
                metadata, "Cathode catalyst components [chemical formula, mol fraction]"
            )
        except:
            print(metadata)
            raise (f"There is an error with the file {metadata}")  # type: ignore
        match returned_string:
            case "[Ag, 1]":
                ls.append("Ag")
            case "[Cu, 1]":
                ls.append("Cu")
            case "[Ti, 0.6], [Cu, 0.4]":
                ls.append("Ti0.6Cu0.4")
            case _:
                raise ValueError(f"Unkown cathode catalyst composition: {returned_string}, with file {metadata}")
    return ("Cathode catalyst composition", ls)


def get_cathode_overlayer_component(list_metadata: list):
    ls = []
    for metadata in list_metadata:
        try:
            returned_string = util.get_value_from_excel(
                metadata, "Cathode overlayer components [chemical formula, particle size in µm, concentration in mg/L]"
            )
        except:
            print(metadata)
            raise (f"There is an error with the file {metadata}")  # type: ignore
        match returned_string:
            case "Not Available":
                ls.append("Not Available")
            case "Yes":
                ls.append("Not Available")
            case "[Ti, 0.1]":
                ls.append("0.1 µm Ti")
            case " [Cu, 0.025]":
                ls.append("0.025 µm Cu")
            case "[Nafion, 0.5]":
                ls.append("0.5 µm Nafion")
            case "[PVDF-HFP fibers, 0.2]":
                ls.append("0.2 µm PVDF-HFP fibers")
            case "[Ti, 0.01]":
                ls.append("0.01 µm Ti")
            case "[PVDF-HFP fibers, 1.1]":
                ls.append("1.1 µm PVDF-HFP fibers")
            case _:
                raise ValueError(f"Unkown cathode overlayer component: {returned_string}, with file {metadata}")
    return ("Cathode overlayer component", ls)
