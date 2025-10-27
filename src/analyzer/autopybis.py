import pybis
import pathlib as pl
import os
import pandas as pd
import random
import string
import zipfile
import shutil
import glob
from contextlib import contextmanager
import xml.etree.ElementTree as ET
from string import Formatter
from dataclasses import dataclass
from typing import Generator
from io import BufferedReader
import requests
from typing import Union, Optional, Dict
import pytz
from datetime import datetime
import traceback

# This class aids filling for space, project and experimental code for PyBis login.
class Identifiers:
    def __init__(self, space_code: str, project_code: str, experiment_code: str) -> None:
        self.space_code = space_code
        self.project_code = project_code
        self.experiment_code = experiment_code

    @property
    def space_identifier(self) -> str:
        return self.space_code

    @property
    def project_identifier(self) -> str:
        return f"/{self.space_identifier}/{self.project_code}"

    @property
    def experiment_identifier(self) -> str:
        return f"{self.project_identifier}/{self.experiment_code}"


# This class is used to facilitate dataset upload to OpenBis.
class Dataset:
    ident = None

    def __init__(self, openbis_instance, dataset_type=None, upload_data=None, data_zipname=None) -> None:
        self.ob = openbis_instance
        self.type = dataset_type
        self.data = upload_data
        self.data_zipname = data_zipname
        self.experiment = Dataset.ident.experiment_identifier

    def upload(self):
        excl_print(f"Start uploading: {self.type} data type.")
        # Use self.data_zipname if it is not None, otherwise use 'file.zip'
        zip_filename = self.data_zipname if self.data_zipname else "file.zip"
        # Check if the filename ends with '.zip', add it if not
        if not zip_filename.endswith(".zip"):
            zip_filename += ".zip"

        if isinstance(self.data, list):  # If self.data is a list, zip the files
            with zipfile.ZipFile(zip_filename, "w") as zipf:
                for file in self.data:
                    # Add files to the zip file with their relative paths
                    zipf.write(file, arcname=os.path.basename(file))
            # Upload the path to the zip file
            self.ob.new_dataset(type=self.type, experiment=self.experiment, file=os.path.abspath(zip_filename)).save()
            # Remove the zip file
            os.remove(zip_filename)
        else:  # If self.data is not a list (a single file), proceed as before
            self.ob.new_dataset(type=self.type, experiment=self.experiment, file=self.data).save()
        excl_print(f"Finish uploading: {self.type} data type.")


# This class is used to upload png to OpenBis
@dataclass
class OpenbisElnUpload:
    type: str
    files: dict[str, pl.Path]
    sessionID: str

    @contextmanager
    def to_form(self) -> Generator[dict[str, Union[int, str, tuple[str, BufferedReader]]], "OpenbisElnUpload", None]:
        try:
            with ExitStack() as stack:
                fs = {name: (str(file), open(file, "rb")) for name, file in self.files.items()}
                rest = {"sessionKeysNumber": (None, 1), "sessionID": (None, self.sessionID)}
                sk_mapping = {"sessionKey_0": (None, f) for f in self.files.keys()}
                yield rest | fs | sk_mapping
        finally:
            print("bye")


# This class is used to upload png to OpenBis
@dataclass
class OpenbisFileserviceResponse:
    url: str


def upload_file_to_lims(ob: Openbis, file: pl.Path) -> OpenbisFileserviceResponse:
    """
    Upload a file to the ELN-LIMS file service.

    This is used to add images or other attachments to the ELN. For more information, see
    https://unlimited.ethz.ch/display/openBISDoc2010/openBIS+Application+Server+Upload+Service
    """
    path = f"{ob.url}/openbis/openbis/file-service/eln-lims?"
    params = {"type": "Files", "id": 1, "sessionID": ob.token, "startByte": 0, "endByte": 0}
    resp = requests.post(path, params=params, files={"upload": (file.name, open(file, "rb"))}, verify=False)
    return OpenbisFileserviceResponse(**resp.json())


def extract_document(prop: str) -> ET:
    """Extracts an HTML Document from an Openbis ELN entry."""
    ET.parse(prop)
    breakpoint()


def new_eln_entry(ob: Openbis, collection: str, template: str, images: dict = None) -> Sample:
    """Creates a new ELN entry with the given template and adds it to the selected collection."""
    filled_template = fill_eln_template(ob, template, images)
    tree = ET.fromstring(filled_template)
    doc = ET.tostring(tree, encoding="unicode", xml_declaration=True)
    entry = ob.new_object(type="ENTRY", experiment=collection, props={"$document": doc})
    return entry


def fill_eln_template(ob: Openbis, template: str, files: dict) -> str:
    """Given a template string and a list of files, replace all the template variables with the ELN upload files."""
    f = Formatter()
    files_to_upload = {name: files[name] for lit, name, fmt, conv in f.parse(template) if name and files}
    paths = {name: upload_file_to_lims(ob, file).url for name, file in files_to_upload.items()}
    merged_dict = {**paths, **locals()}  # Merge two dictionaries
    return template.format(**merged_dict)


def get_file_from_lims(ob: Openbis, name: str):
    path = f"{ob.url}/openbis/openbis/upload"


# This function is used to generate a unique experimental code as this is required by OpenBis.
def generate_random_string():
    characters = string.ascii_uppercase + string.digits
    random_string = "".join(random.choices(characters, k=8))
    return random_string


# Help making the printing pop out in terminal.
def excl_print(note):
    print("****************************************************************")
    print("                                                                ")
    print(note)
    print("                                                                ")
    print("****************************************************************")


# These two test functions are used for uploading png image to OpenBis to be used as a thumbnail the result section.
def test_eln_template(temp_openbis: Openbis):
    with tf.NamedTemporaryFile("w") as temp_file:
        files = {"f1": pl.Path(temp_file.name)}
        template = """<img src="{f1}">"""
        filled = fill_eln_template(temp_openbis, template, files)


def test_eln_filling(temp_jpg: pl.Path, temp_openbis: Openbis, exp: str):
    files = {"f1": temp_jpg}
    template = """<html><head></head><body><h1>Test</h1><img src="{f1}"></img></body></html>"""
    filled = new_eln_entry(temp_openbis, exp, template, files)
    filled.save()
    print(filled)


# Uploading new experiment along with the associated dataset onto a degsinated space on OpenBis
def dataset_upload(folder_name, space_code, project_code, ob):
    experiment_code = generate_random_string()
    ident = Identifiers(space_code, project_code, experiment_code)
    Dataset.ident = Identifiers(space_code, project_code, experiment_code)
    # Create the new experiment.
    exp = ob.new_experiment(code=ident.experiment_code, type="ECHEM6", project=ident.project_identifier)

    exp.p["$name"] = folder_name
    exp.save()

    #Upload png image file to OpenBis server.
    png_path = os.path.join("Graph_Export", f"datagram_{folder_name}.GCdata.png")
    thumbnail = pl.Path(png_path)
    test_eln_filling(thumbnail, ob, exp=ident.experiment_identifier)
    res = upload_file_to_lims(ob, thumbnail)

    # Add the URL of the image as an HTML string to the property
    html_img = f'<img src="https://openbis-empa-lab501.ethz.ch/{res.url}">'
    exp.p["default_experiment.experimental_results"] = html_img
    exp.save()

    # # Extract files in output.zip
    output_path = "Output"
    # zip_path = os.path.join(output_path, folder_name, f"datagram_{folder_name}.zip")
    # destination_directory = os.path.join(os.path.dirname(zip_path), "Extracted")
    # # Check if the directory already exists. If it does, remove it (and all its contents). Then create a new one.
    # if os.path.exists(destination_directory):
    #     shutil.rmtree(destination_directory)
    # os.makedirs(destination_directory)
    # path_extract_folder = os.path.join(output_path, "Output", folder_name, "Extracted")

    # # Extract the zip file contents to this directory
    # with zipfile.ZipFile(zip_path) as zipObj:
    #     zipObj.extractall(destination_directory)

    # Filling metadata info from the metadata file
    try:
        excel_metadata = glob.glob(os.path.join(output_path, folder_name, "Extracted", "*Metadata.xlsx"))[0]
        df_metadata = pd.read_excel(excel_metadata, sheet_name="Metadata")
    except:
        excel_metadata = glob.glob(os.path.join(output_path, folder_name, "*Metadata.xlsx"))[0]
        df_metadata = pd.read_excel(excel_metadata, sheet_name="Metadata")
    mask_true = df_metadata["Value"] == "Yes"
    mask_false = df_metadata["Value"] == "No"
    df_metadata.loc[mask_true, "Value"] = True
    df_metadata.loc[mask_false, "Value"] = False
    for idx in range(0, len(df_metadata["Value"])):
        if not pd.isna(df_metadata["OpenBis code"][idx]) and not pd.isna(df_metadata["Value"][idx]):
            try:
                if df_metadata["Type"][idx] == 'Float':
                    value = float(df_metadata["Value"][idx])
                else:
                    value = df_metadata["Value"][idx]
                exp.p[df_metadata["OpenBis code"][idx]] = value
                exp.save()
            except Exception as e:
                print("                                ")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Error at idx: ", str(idx), "OpenBis code: ", df_metadata["OpenBis code"][idx], "Value: ", df_metadata["Value"][idx])
                print(f'Exception: {e}')
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("                                ")
    # Extract and upload the experiment start/end date.
    exp_start, exp_end = get_experiment_times(os.path.join("Output", folder_name))
    exp.p["exp_start_timestamp"] = exp_start
    exp.p["exp_end_timestamp"] = exp_end
    exp.save()

    # Zip all the data in the output folder and upload into OpenBis
    dir_data_folder = os.path.join(output_path, folder_name)

    # Define the zip file name
    zip_file_name = f"all_data_{folder_name}.zip"
    zip_file_path = os.path.join(dir_data_folder, zip_file_name)

    try:
        # Create the zip archive
        shutil.make_archive(zip_file_path[:-4], 'zip', dir_data_folder)
        
        ds_all_data = Dataset(ob)
        ds_all_data.type = "ECHEM6_DATASET" 
        ds_all_data.data = zip_file_path
        ds_all_data.upload()

    except Exception as e:
        print(f"Error occurred while creating the zip file: {e}")

    finally:
        # Remove the zip file
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"Deleted zip file: {zip_file_path}")


    # # Uploading datagram
    # try:
    #     ds_datagram = Dataset(ob)
    #     ds_datagram.type = "LA273_DATAGRAM"
    #     ds_datagram.data = glob.glob(os.path.join(output_path, folder_name, "*.json"))[0]
    #     ds_datagram.upload()
    # except:
    #     pass

    # # Uploading GC_excel_data
    # try:
    #     ds_gc_data = Dataset(ob)
    #     ds_gc_data.type = "LA273_ANALYZED_DATA_GC"
    #     ds_gc_data.data = os.path.join(output_path, folder_name, f"datagram_{folder_name}.GCdata.xlsx")
    #     ds_gc_data.upload()
    # except:
    #     pass

    # # Uploading LC_excel_data
    # try:
    #     ds_lc_data = Dataset(ob)
    #     ds_lc_data.type = "LA273_ANALYZED_DATA_LC"
    #     ds_lc_data.data = glob.glob(os.path.join(output_path, folder_name, "*LCdata.xlsx"))[0]
    #     ds_lc_data.upload()
    # except:
    #     pass

    # # Uploading temp data
    # try:
    #     ds_temp = Dataset(ob)
    #     ds_temp.type = "LA273_RAWDATA_TEMP"
    #     ds_temp.data = glob.glob(os.path.join(output_path, folder_name, "Extracted", "*temperature_for_yadg.csv"))[0]
    #     ds_temp.upload()
    # except:
    #     pass

    # # Uploading flow data
    # try:
    #     ds_flow = Dataset(ob)
    #     ds_flow.type = "LA273_RAWDATA_FLOW"
    #     ds_flow.data = glob.glob(os.path.join(output_path, folder_name, "Extracted", "*flow_for_yadg.csv"))[0]
    #     ds_flow.upload()
    # except:
    #     pass

    # # Uploading pressure data
    # try:
    #     ds_pressure = Dataset(ob)
    #     ds_pressure.type = "LA273_RAWDATA_PRESSURE"
    #     ds_pressure.data = glob.glob(os.path.join(output_path, folder_name, "Extracted", "*pressure_for_yadg.csv"))[0]
    #     ds_pressure.upload()
    # except:
    #     pass

    # # Uploading echem data
    # try:

    #     file_echem = []
    #     for file_extension in ["*.mpr", "*.mgr", "*.mps"]:
    #         file_echem.extend(glob.glob(os.path.join(path_extract_folder, file_extension)))
    #     ds_echem = Dataset(ob)
    #     ds_echem.type = "LA273_RAWDATA_ECHEM"
    #     ds_echem.data = file_echem
    #     ds_echem.data_zipname = "Echem_files.zip"
    #     ds_echem.upload()
    # except:
    #     pass

    # # Uploading GC zip file
    # try:
    #     ds_gc_raw = Dataset(ob)
    #     ds_gc_raw.type = "LA273_RAWDATA_GC"
    #     ds_gc_raw.data = glob.glob(os.path.join(output_path, folder_name, "Extracted", "*-GC.zip"))[0]
    #     ds_gc_raw.upload()
    # except:
    #     pass

    # # Uploading other raw data

    # try:
    #     # Define excluded files and extension. These files have already been uploaded.
    #     excluded_extensions = [".mpr", ".mgr", ".mps"]
    #     excluded_files = [ds_gc_raw.data, ds_pressure.data, ds_temp.data, ds_flow.data, ds_lc_data.data]

    #     # Create a list to store the file paths
    #     file_list = []

    #     # Loop through all files in the directory
    #     for filepath in glob.glob(path_extract_folder + "/*"):
    #         # Check if file path is not in excluded files and does not have an excluded extension
    #         if filepath not in excluded_files and os.path.splitext(filepath)[1] not in excluded_extensions:
    #             # Add the file path to the list
    #             file_list.append(filepath)

    #     # Upload the files in file_list to OpenBis.
    #     ds_other = Dataset(ob)
    #     ds_other.type = "LA273_RAWDATA_OTHER"
    #     ds_other.data = file_list
    #     ds_other.data_zipname = "Other_data"
    #     ds_other.upload()
    # except:
    #     pass

    # # Delte the Extracted folder.
    # destination_directory = os.path.join(os.path.dirname(zip_path), "Extracted")
    # shutil.rmtree(destination_directory)


def run(space_code, project_code):
    url = r"https://openbis-empa-lab501.ethz.ch/"
    token_file = "Recipe/workflow/OpenBis_PAT.txt"

    with open(token_file, "r") as file:
        token = file.read().strip()
    ob = pybis.Openbis(url)
    valid = ob.set_token(token)

    output_path = "Output"
    folders = [folder for folder in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, folder))]
    sorted_folders = sorted(folders)
    for fld in sorted_folders:
        try:
            dataset_upload(fld, space_code=space_code, project_code=project_code, ob=ob)
        except Exception as e:
            traceback.print_exc()


# This function helps with extracting start and end time form the GC.xlsx file
def get_experiment_times(input_path):
    # Look for the file that ends with GCdata.xlsx
    for file in os.listdir(input_path):
        if file.endswith("GCdata.xlsx"):
            full_path = os.path.join(input_path, file)
            # Read the Excel file into a DataFrame
            df = pd.read_excel(full_path)
            # Drop NA values from the first column and reset the index
            non_empty_rows = df[df.columns[0]].dropna().reset_index(drop=True)
            # Get the start time and adjust for GMT+2
            start_time_unix = non_empty_rows[0]
            start_time_naive = datetime.utcfromtimestamp(start_time_unix)
            start_time_with_tz = start_time_naive.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Etc/GMT-2"))
            start_time = start_time_with_tz.strftime("%Y-%m-%d %H:%M:%S")
            # Get the end time and adjust for GMT+2
            end_time_unix = non_empty_rows.iloc[-1]
            end_time_naive = datetime.utcfromtimestamp(end_time_unix)
            end_time_with_tz = end_time_naive.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Etc/GMT-2"))
            end_time = end_time_with_tz.strftime("%Y-%m-%d %H:%M:%S")

            return start_time, end_time

    raise FileNotFoundError("No file named GCdata.xlsx found in the specified path!")
