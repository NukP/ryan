"""
This module serve as the main module for uploading catalysis experiment

This module will be primarily used to facilitate electrocatalysis CO2 reduction (Itteration 5 standard) to OpenBis.
It will be first used to assit the upload itteration 5 dataset for the ML paper. Later, it may be used as part of AutoplotDB to assit with OpenBis communication.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pybis
import requests

from .vibing import Identifiers


def _upload_to_eln_lims(ob: pybis.Openbis, file: Path) -> str:
    """
    Upload a file to the ELN-LIMS file service and return its relative URL.

    Args:
        ob (pybis.Openbis): Authenticated Openbis connection.
        file (Path): Path to the file to upload.

    Returns:
        str: Relative URL returned by the file service (e.g. 'openbis/openbis/upload?...').
    """
    endpoint = f"{ob.url.rstrip('/')}/openbis/openbis/file-service/eln-lims"
    params = {
        "type": "Files",
        "id": 1,
        "sessionID": ob.token,
        "startByte": 0,
        "endByte": 0,
    }
    with open(file, "rb") as fh:
        resp = requests.post(endpoint, params=params, files={"upload": (file.name, fh)}, verify=False)
    resp.raise_for_status()
    return resp.json()["url"]


def _cast_metadata_value(raw: Any, typ: Optional[str]) -> Any:
    """Coerce Excel values into the right Python types for openBIS."""
    if pd.isna(raw):
        return None

    # Normalize simple True/False text
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() in {"yes", "true"}:
            raw = True
        elif s.lower() in {"no", "false"}:
            raw = False

    t = (typ or "").strip().lower()

    if t in {"float", "double", "number"}:
        return float(raw)
    if t in {"int", "integer"}:
        return int(float(raw))
    if t in {"bool", "boolean"}:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            return raw.strip().lower() in {"true", "yes", "1"}
        return bool(raw)
    if t in {"date", "datetime", "timestamp"}:
        # ISO 8601 string is usually safe for openBIS text/date props
        return pd.to_datetime(raw).isoformat()

    # Default: keep as-is (string/object)
    return raw


def _read_metadata_table(path: Path) -> pd.DataFrame:
    """Read the metadata table; fall back to first sheet if 'Metadata' is absent."""
    try:
        return pd.read_excel(path, sheet_name="Metadata")
    except Exception:
        return pd.read_excel(path)


@dataclass
class Files:
    dir_experiment_folder: Path

    def __post_init__(self):
        if isinstance(self.dir_experiment_folder, str):
            self.dir_experiment_folder = Path(self.dir_experiment_folder)

    @property
    def exp_name(self) -> str:
        return self.dir_experiment_folder.name

    @property
    def dir_datagram(self) -> Path:
        nc_files = list(self.dir_experiment_folder.glob("*.nc"))

        if not nc_files:
            raise FileNotFoundError("There is no datagram in the folder. Please check.")
        elif len(nc_files) > 1:
            raise FileExistsError("There should be only 1 datagram in the folder.")

        return nc_files[0]

    @property
    def dir_metadata(self) -> Path:
        metadata_files = list(self.dir_experiment_folder.glob("*-metadata.xlsx"))

        if not metadata_files:
            raise FileNotFoundError("There is no metadata file in the folder. Please check.")
        elif len(metadata_files) > 1:
            raise FileExistsError("There should be only 1 metadata file in the folder.")

        return metadata_files[0]

    @property
    def dir_gcdata(self) -> Path:
        gcdata_files = list(self.dir_experiment_folder.glob("*.GCdata.xlsx"))

        if not gcdata_files:
            raise FileNotFoundError("There is no GC data file in the folder. Please check.")
        elif len(gcdata_files) > 1:
            raise FileExistsError("There should be only 1 GC data file in the folder.")

        return gcdata_files[0]

    @property
    def dir_exported_graph(self) -> Path:
        path = self.dir_experiment_folder.parent.parent / "Graph_Export" / f"datagram_{self.exp_name}.GCdata.png"
        if not path.exists():
            raise FileNotFoundError(f"Exported graphs directory not found: {path}")
        return path

    @property
    def dir_input_data_zip(self) -> Path:
        """Path to the input_data_<exp_name>.zip file inside the temporary folder."""
        path = self.dir_experiment_folder / "temporarily" / f"input_data_{self.exp_name}.zip"
        if not path.exists():
            raise FileNotFoundError(f"Input data zip not found: {path}")
        return path

    @property
    def dir_output_data_zip(self) -> Path:
        """Path to the output_data_<exp_name>.zip file inside the temporary folder."""
        path = self.dir_experiment_folder / "temporarily" / f"output_data_{self.exp_name}.zip"
        if not path.exists():
            raise FileNotFoundError(f"Output data zip not found: {path}")
        return path


def manage_files(dir_experiment_folder: Path):
    """Prepare experiment files into structured zip archives for OpenBis upload.

    The function organizes selected files from the experiment folder into
    `input_data` and `output_data` directories, then zips them into
    `input_data.zip` and `output_data.zip` inside a temporary folder.

    Args:
        dir_experiment_folder (Path): Path to the folder containing the experiment data.

    Returns:
        None: The function modifies the filesystem by creating directories and zip archives.
    """
    # Create the temporarily folder for organizing files folders.
    if not isinstance(dir_experiment_folder, Path):
        Path(dir_experiment_folder)
    dir_temp = dir_experiment_folder / "temporarily"
    Path(dir_temp).mkdir(parents=True, exist_ok=True)
    dir_input_data = dir_temp / "input_data"
    Path(dir_input_data).mkdir(parents=True, exist_ok=True)
    dir_output_data = dir_temp / "output_data"
    Path(dir_output_data).mkdir(parents=True, exist_ok=True)

    # Copy files into input folder
    zip_files = list(dir_experiment_folder.glob("*.zip"))
    if len(zip_files) == 1:
        from zipfile import ZipFile

        with ZipFile(zip_files[0], "r") as zf:
            for file_name in zf.namelist():
                if file_name.endswith(".mps") or file_name.endswith(".mpr") or file_name.endswith("-GC.zip"):
                    source_path = zf.extract(file_name, dir_temp)
                    shutil.copy2(source_path, dir_input_data / Path(file_name).name)

    # lc_files = list(dir_experiment_folder.glob("*-LC.xlsx"))
    # if len(lc_files) == 1:
    #     shutil.copy2(lc_files[0], dir_input_data)

    flow_file = dir_experiment_folder / "flow_for_yadg.csv"
    if flow_file.exists():
        shutil.copy2(flow_file, dir_input_data / "flow_data.csv")

    pressure_file = dir_experiment_folder / "pressure_for_yadg.csv"
    if pressure_file.exists():
        shutil.copy2(pressure_file, dir_input_data / "pressure_data.csv")

    temperature_file = dir_experiment_folder / "temperature_for_yadg.csv"
    if temperature_file.exists():
        shutil.copy2(temperature_file, dir_input_data / "temperature_data.csv")

    # Copy files into output folder
    src_recipe = dir_experiment_folder / "recipe_for_dgbowl"
    dst_recipe = dir_output_data / "recipe_for_dgbowl"
    if src_recipe.exists():
        shutil.copytree(src_recipe, dst_recipe, dirs_exist_ok=True)

    electro_files = list(dir_experiment_folder.glob("*.electro.xlsx"))
    if len(electro_files) == 1:
        shutil.copy2(str(electro_files[0]), dir_output_data)

    gcdata_files = list(dir_experiment_folder.glob("*.GCdata.xlsx"))
    if len(gcdata_files) == 1:
        shutil.copy2(gcdata_files[0], dir_output_data)

    datagram_files = list(dir_experiment_folder.glob("*.nc"))
    if len(datagram_files) == 1:
        shutil.copy2(datagram_files[0], dir_output_data)

    # zip the input and output data folders
    folder = Files(dir_experiment_folder)
    shutil.make_archive(str(dir_temp / f"input_data_{folder.exp_name}"), "zip", dir_input_data)
    shutil.make_archive(str(dir_temp / f"output_data_{folder.exp_name}"), "zip", dir_output_data)


def experiment_upload(
    dir_experiment_folder: Path,
    ob: pybis.Openbis,
    space_code: str = "LA273",
    project_code: str = "Dataset_For_Machine_Learning_Paper".upper(),
):
    """
    This will be a main function to upload an experiment

    Args:
        dir_experiment_folder (Path): Path to the experiment folder containing all necessary files.
        ob (pybis.Openbis): OpenBis connection object.
        space_code (str): OpenBis space code where the experiment will be uploaded.
        project_code (str): OpenBis project code where the experiment will be uploaded.
    """

    # Setting up initial parameters for OpenBis upload
    folder = Files(dir_experiment_folder)
    exp_code = folder.exp_name
    ident = Identifiers(space_code, project_code, exp_code)

    # Helper function for dataset upload
    def _upload_dataset(ds_type: str, file_path: Path, raise_on_error: bool = False):
        if not file_path.exists():
            print(f"[SKIP] {ds_type}: file not found -> {file_path}")
            return

        try:
            # IMPORTANT: pass the experiment OBJECT (or exp.permId), not the identifier string
            ds = ob.new_dataset(
                type=ds_type, experiment=exp, file=str(file_path)
            )  # <-- this avoids ob.get_experiment(...) lookup
            ds.save()
            print(f"[OK] Uploaded {ds_type}: {file_path.name}")

        except Exception as e:
            import traceback

            print(f"[ERROR] {ds_type}: {file_path} -> upload failed")
            print(f"  Exception: {type(e).__name__}")
            print(f"  Args     : {getattr(e, 'args', ())}")
            print("  Traceback:")
            print(traceback.format_exc())
            if raise_on_error:
                raise

    # Create the new experiment in OpenBis
    exp = ob.new_experiment(code=ident.experiment_code, type="ECHEM6", project=ident.project_identifier)
    exp.p["$name"] = folder.exp_name
    exp.save()

    # --- Upload PNG and show it in the ELN “Experimental results” field ---
    thumbnail = folder.dir_exported_graph
    rel_url = _upload_to_eln_lims(ob, thumbnail)  # hacky but required: use file-service directly
    full_url = f"{ob.url.rstrip('/')}/{rel_url.lstrip('/')}"
    html_img = f'<img src="{full_url}" alt="GC Result" style="max-width: 100%;">'
    exp.p["default_experiment.experimental_results"] = html_img
    exp.save()

    # --- Apply metadata to experiment ---
    md_path = folder.dir_metadata
    df_metadata = _read_metadata_table(md_path)

    # Normalize Yes/No to booleans up-front (keeps integers/NaNs intact)
    if "Value" in df_metadata.columns:
        df_metadata["Value"] = df_metadata["Value"].replace({"Yes": True, "No": False})

    required_cols = {"OpenBis code", "Value"}
    missing = required_cols - set(df_metadata.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {', '.join(sorted(missing))}")

    errors = []
    any_change = False

    for idx, row in df_metadata.iterrows():
        code = row.get("OpenBis code")
        val = row.get("Value")
        typ = row.get("Type")

        if pd.isna(code) or pd.isna(val):
            continue

        try:
            coerced = _cast_metadata_value(val, typ)
            if coerced is not None:
                # IMPORTANT: pybis doesn’t support dict.update; assign item-by-item
                exp.p[str(code)] = coerced
                any_change = True
        except Exception as e:
            errors.append((idx, code, val, str(e)))

    # Save once after all assignments
    if any_change:
        exp.save()

    if errors:
        print("\n--- Metadata assignment issues ---")
        for idx, code, val, err in errors:
            print(f"[row {idx}] code='{code}' value='{val}' -> {err}")
        print("--- End of issues ---\n")

    # Uploaing data
    manage_files(dir_experiment_folder)
    _upload_dataset("ELECTROCATALYSIS_INPUT_DATA", folder.dir_input_data_zip)
    _upload_dataset("ELECTROCATALYSIS_METADATA", folder.dir_metadata)
    _upload_dataset("ELECTROCATALYSIS_OUTPUT_DATA", folder.dir_output_data_zip)
