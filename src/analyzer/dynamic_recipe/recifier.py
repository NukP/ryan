""""
This module contains functions used to dynamically generate yadg and dgpost recipes based on the availability of data files in the data folder.
This module will call template for the recipe from templates module.
"""
import pandas as pd
from typing import Dict, Union
from . import templates  
from pathlib import Path


def generate_recipe(yadg_recipe_dir: Path, dgpost_recipe_dir: Path, data_folder_dir: Path) -> None: 
    """
    Generate a recipe for yadg and dgpost.

    This function generates the necessary recipes for both the yadg and dgpost script (if the manual_recipe is set to be False). 
    It first generates the YADG recipe by calling `generate_recipe_yadg` with the given directory, 
    and then generates the DGPost recipe by calling `generate_recipe_dgpost` with the specified 
    directories.

    Args:
        yadg_recipe_dir (Path): The directory where the yadg recipe will be generated.
        dgpost_recipe_dir (Path): The directory where the dgpost recipe will be generated.
        data_folder_dir (Path): The directory containing the raw experiment data to be analyzed.
    
    Returns:
        None
    """
    if not isinstance(yadg_recipe_dir, Path):
        yadg_recipe_dir = Path(yadg_recipe_dir)
    if not isinstance(dgpost_recipe_dir, Path):
        dgpost_recipe_dir = Path(dgpost_recipe_dir)
    if not isinstance(data_folder_dir, Path):
        data_folder_dir = Path(data_folder_dir)

    generate_recipe_yadg(yadg_recipe_dir)
    generate_recipe_dgpost(dgpost_recipe_dir, data_folder_dir)


def generate_recipe_dgpost(dgpost_recipe_dir: Path, data_folder_dir: Path) -> None:
    """
    Generate a recipe for dgpost.

    This function generates various YAML configuration files required by the dgpost script. 
    It first checks the data in the specified directory and then creates YAML files for
    electrochemistry, impedance, gas-products, and, optionally, liquid-products
    configurations based on the presence of LC data.
    The templates are stored in templates.py module.

    Args:
        dgpost_recipe_dir (Path): The directory where the dgpost recipe files will be generated.
        data_folder_dir (Path): The directory containing the raw experiment data to be analyzed.

    Returns:
        None
    """
    if not isinstance(dgpost_recipe_dir, Path):
        dgpost_recipe_dir = Path(dgpost_recipe_dir)
    if not isinstance(data_folder_dir, Path):
        data_folder_dir = Path(data_folder_dir)

    dict_check_data = check_data(data_folder_dir)

    # electro yaml
    (dgpost_recipe_dir / "01_dynamically_generated_dgpost_electro.yaml").write_text(
        templates.template_dgpost_electro
    )

    # peis yaml
    (dgpost_recipe_dir / "02_dynamically_generated_dgpost_peis.yaml").write_text(
        templates.template_dgpost_peis
    )

    # GC yaml
    (dgpost_recipe_dir / "03_dynamically_generated_dgpost_GC.yaml").write_text(
        generate_recipe_dgpost_gc(dict_check_data)
    )

    # LC yaml
    if dict_check_data['has_lc']:
        (dgpost_recipe_dir / "04_dynamically_generated_dgpost_LC.yaml").write_text(
            generate_recipe_dgpost_lc(dict_check_data)
        )


def generate_recipe_yadg(yadg_recipe_dir: Path) -> None:
    """
    Generate a recipe for yadg.

    This function generates a YAML recipe for yadg by writing a predefined template
    to a file in the specified directory. This template is stores in templates.py module.

    Args:
        yadg_recipe_dir (Path): The directory where the yadg recipe file will be generated.

    Returns:
        None
    """
    if not isinstance(yadg_recipe_dir, Path):
        yadg_recipe_dir = Path(yadg_recipe_dir)

    (yadg_recipe_dir / "dynamically_generate_yadg_recipe.yaml").write_text(templates.template_yadg)


def check_data(data_folder_dir: Path) -> Dict[str, Union[float, bool]]:
    """
    Extract pH from the metadata files and determine the presence of temperature, pressure, and LC data.

    This function extracts the pH value from a metadata file and checks for the existence of temperature, 
    pressure, and LC data files within the specified data folder directory. The results are returned as a dictionary.

    If the metadata file is not found in the data folder, the function will look for it in the corresponding
    Output/<exp_name> folder.

    Args:
        data_folder_dir (Path): The directory containing the raw experiment data to be analyzed.

    Returns:
        Dict[str, Union[float, bool]]: A dictionary with the following keys:
            - 'pH': (float) The pH value extracted from the metadata.
            - 'cathode_reaction': (str) The cathode reaction extracted from the metadata.
            - 'has_temp': (bool) True if the temperature data file is found, False otherwise.
            - 'has_pressure': (bool) True if the pressure data file is found, False otherwise.
            - 'has_lc': (bool) True if the LC data file is found, False otherwise.
    """
    if not isinstance(data_folder_dir, Path):
        data_folder_dir = Path(data_folder_dir)

    try:
        metadata_files_dir = next(data_folder_dir.glob("*-metadata.xlsx"))
    except StopIteration:
        exp_name = data_folder_dir.parent.name  # parent of 'data'
        output_metadata = list(Path("Output", exp_name).glob("*-metadata.xlsx"))
        if not output_metadata:
            raise FileNotFoundError(
                f"No metadata file found in {data_folder_dir} or Output/{exp_name}"
            )
        metadata_files_dir = output_metadata[0]

    df_metadata = pd.read_excel(metadata_files_dir, sheet_name="Metadata")
    ph = float(df_metadata[df_metadata["Metadata"] == "Cathode compartment electrolyte pH - SET"]["Value"].values[0])
    cathode_reaction = df_metadata[df_metadata["Metadata"] == "Cathode reaction"]["Value"].values[0]

    has_temp = any(data_folder_dir.glob("*temperature_for_yadg.csv"))
    has_pressure = any(data_folder_dir.glob("*pressure_for_yadg.csv"))
    has_lc = any(data_folder_dir.glob("*LC-data.xlsx"))

    return {
        "pH": ph,
        "cathode_reaction": cathode_reaction,
        "has_temp": has_temp,
        "has_pressure": has_pressure,
        "has_lc": has_lc,
    }
  

def generate_recipe_dgpost_gc(dict_check_data: dict) -> str:
    """
    Generate a dgpost recipe for GC data analysis.

    This function generates a dgpost recipe specifically for analyzing GC data, 
    incorporating temperature and pressure data if available. The generated recipe 
    is based on templates that are dynamically modified depending on the presence of 
    temperature and pressure data. The pH value is also included in the recipe.

    Args:
        dict_check_data (dict): A dictionary containing the pH value and boolean flags 
            indicating the availability of temperature and pressure data. Expected keys are:
            - 'pH': (float) The pH value extracted from the metadata.
            - 'cathode_reaction': (str) The cathode reaction extracted from the metadata.
            - 'has_temp': (bool) True if the temperature data file is found, False otherwise.
            - 'has_pressure': (bool) True if the pressure data file is found, False otherwise.

    Returns:
        str: A string containing the filled-in dgpost recipe for GC data analysis.
    """
    template_dgpost_gc_main = templates.template_dgpost_gc_main
    template_constant_t = templates.template_constant_t
    template_has_temp = templates.template_has_temp
    template_has_pressure = templates.template_has_pressure
    
    # If has_temp is False, modify the global variable template_has_temp, otherwise use it as is.
    if not dict_check_data['has_temp']:
        template_has_temp = ""  # Clear the template if temperature data is not available
    else:
        template_constant_t = ""  # Clear the constant temperature if temperature data is available

    # If has_pressure is False, modify the global variable template_has_pressure, otherwise use it as is.
    if not dict_check_data['has_pressure']:
        template_has_pressure = ""  # Clear the template if pressure data is not available

    #Specify the charge per carbon atom based on the cathode reaction
    if dict_check_data['cathode_reaction'] == 'Carbon dioxide reduction':
        charge_per_carbon_atom = 4
    elif dict_check_data['cathode_reaction'] == 'Carbon monoxide reduction':
        charge_per_carbon_atom = 2
    else:
        raise ValueError("The cathode reaction filled in the metadata is not recognized. Charge per carbon atom cannot be determined.")

    # Now use the templates in the function
    filled_gc_template = template_dgpost_gc_main.format(
        ph=dict_check_data['pH'],
        charge_C=charge_per_carbon_atom,
        template_constant_t=template_constant_t,
        template_has_temp=template_has_temp,
        template_has_pressure=template_has_pressure
    )
    
    # Removing excess empty lines
    formatted_gc_template = "\n".join([line for line in filled_gc_template.split("\n") if line.strip()])
    return formatted_gc_template

def generate_recipe_dgpost_lc(dict_check_data: dict) -> str:
    """
    Generate a dgpost recipe for LC data analysis.

    This function generates a dgpost recipe specifically for analyzing LC data, 
    incorporating temperature data if available. The generated recipe is based on templates 
    that are dynamically modified depending on the presence of temperature data. The pH value 
    is also included in the recipe.

    Args:
        dict_check_data (dict): A dictionary containing the pH value and a boolean flag 
            indicating the availability of temperature data. Expected keys are:
            - 'pH': (float) The pH value extracted from the metadata.
            - 'cathode_reaction': (str) The cathode reaction extracted from the metadata.
            - 'has_temp': (bool) True if the temperature data file is found, False otherwise.

    Returns:
        str: A string containing the filled-in dgpost recipe for LC data analysis.
    """
    template_dgpost_lc_main = templates.template_dgpost_lc_main
    template_constant_t = templates.template_constant_t
    template_has_temp = templates.template_has_temp

    # If has_temp is False, modify the global variable template_has_temp, otherwise use it as is.
    if not dict_check_data['has_temp']:
        template_has_temp = ""  # Clear the template if temperature data is not available
    else:
        template_constant_t = ""  # Clear the constant temperature if temperature data is available

    #Specify the charge per carbon atom based on the cathode reaction
    if dict_check_data['cathode_reaction'] == 'Carbon dioxide reduction':
        charge_per_carbon_atom = 4
    elif dict_check_data['cathode_reaction'] == 'Carbon monoxide reduction':
        charge_per_carbon_atom = 2
    else:
        raise ValueError("The cathode reaction filled in the metadata is not recognized. Charge per carbon atom cannot be determined.")

    # Now use the templates in the function
    filled_lc_template = template_dgpost_lc_main.format(
        ph=dict_check_data['pH'],
        charge_C=charge_per_carbon_atom,
        template_constant_t=template_constant_t,
        template_has_temp=template_has_temp,
    )
   
    # Removing excess empty lines
    formatted_lc_template = "\n".join([line for line in filled_lc_template.split("\n") if line.strip()])
    return formatted_lc_template
