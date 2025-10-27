import os
import re
import pandas as pd
import zipfile
import glob
import json
import pytz
from datetime import datetime
import time
import shutil
import traceback


def stage_manager(folder_path, folder_name, show_error=False):

    if show_error == True:
        try:
            move_gc_files(folder_path)
        except Exception as e:
            traceback.print_exc()
        pass_status = []
        failed_status = []
        exp_name = folder_name.split("_")[1]
        try:
            praser_pressure(folder_path)
        except Exception as e:
            traceback.print_exc()
        try:
            praser_temp(folder_path)
        except Exception as e:
            traceback.print_exc()
        try:
            praser_flow(folder_path)
        except Exception as e:
            traceback.print_exc()

    else:
        try:
            move_gc_files(folder_path)
        except:
            pass
        pass_status = []
        failed_status = []
        exp_name = folder_name.split("_")[1]
        try:
            praser_pressure(folder_path)
        except:
            pass
        try:
            praser_temp(folder_path)
        except:
            pass
        try:
            praser_flow(folder_path)
        except:
            pass

    # Append the name of the unit folder into a list.
    unit_folder_list = []
    for name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, name)):
            # Using regex to check if directory name is UX where X is an integer
            if re.match(r"^U[0-9]+$", name):
                unit_folder_list.append(name)
    status_report = {}
    for unit in unit_folder_list:
        # Check the success of slicing flow, temp and pressure data.
        unit_folder_path = os.path.join(folder_path, unit)
        pass_status = []
        failed_status = []
        if any(filename.endswith("flow_for_yadg.csv") for filename in os.listdir(unit_folder_path)):
            pass_status.append("Flow")
            proceed = True
        else:
            failed_status.append("Flow")
            proceed = False
        if any(filename.endswith("temperature_for_yadg.csv") for filename in os.listdir(unit_folder_path)):
            pass_status.append("Temperature")
        else:
            failed_status.append("Temperature")

        if any(filename.endswith("pressure_for_yadg.csv") for filename in os.listdir(unit_folder_path)):
            pass_status.append("Pressure")
        else:
            failed_status.append("Pressure")
        if proceed == True:
            move_from_folder = unit_folder_path
            move_to_folder = os.path.join(r"Recipe/data_for_dgbowl/", f"{exp_name}_Multiplex_{unit}", "data")
            shutil.copytree(move_from_folder, move_to_folder)
        status_report[unit] = {"pass": pass_status, "failed": failed_status, "proceed": proceed}

    return status_report

    # This function is used to move the GC files to the appropriate folder.


def move_gc_files(folder_path):
    gc_collection = []
    for filename in os.listdir(folder_path):
        if filename.startswith("collection") and filename.endswith("GC.zip"):
            gc_collection.append(os.path.join(folder_path, filename))

    for gc_collection_file in gc_collection:
        name_split = os.path.basename(gc_collection_file).split("-")
        list_unit = [unit for idx, unit in enumerate(name_split) if idx != 0 and idx != len(name_split) - 1]

        directory_to_extract_to = os.path.dirname(gc_collection_file)
        new_folder_name = os.path.splitext(os.path.basename(gc_collection_file))[0]  # Remove '.zip' from the filename
        new_folder_path = os.path.join(directory_to_extract_to, new_folder_name)  # New folder path

        if not os.path.exists(new_folder_path):  # If the directory does not exist
            os.makedirs(new_folder_path)  # Create directory

        with zipfile.ZipFile(gc_collection_file, "r") as zip_ref:
            zip_ref.extractall(new_folder_path)  # Extract all to the new directory

        # Get the list of files in the extracted directory, sort it
        files = sorted([f for f in os.listdir(new_folder_path) if os.path.isfile(os.path.join(new_folder_path, f))])

        # Loop over the files and move them to the respective UX-GC directories
        for i, file_name in enumerate(files):
            unit_index = i % len(list_unit)
            old_file_path = os.path.join(new_folder_path, file_name)
            new_dir_path = os.path.join(folder_path, f"{list_unit[unit_index]}-GC")
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            new_file_path = os.path.join(new_dir_path, file_name)
            shutil.move(old_file_path, new_file_path)  # Move the file to UX-GC

        # For each UX-GC directory, zip the content and then remove the directory
        for unit in list_unit:
            dir_path = os.path.join(folder_path, f"{unit}-GC")
            with zipfile.ZipFile(f"{dir_path}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        # Here we just add the file name to the archive and not the whole path
                        zipf.write(os.path.join(root, file), arcname=file)
            shutil.rmtree(dir_path)  # remove UX-GC directory

        # Move the zip file to the UX directory and delete the original collection-GC directory
        for unit in list_unit:
            old_path = os.path.join(folder_path, f"{unit}-GC.zip")
            new_path = os.path.join(folder_path, unit, f"{unit}-GC.zip")
            shutil.move(old_path, new_path)  # Move UX-GC.zip to UX

        shutil.rmtree(new_folder_path)  # Remove the collection-GC directory


def praser_pressure(folder_path):
    # Read the combined pressure file
    try:
        raw_pressure = pd.read_csv(glob.glob(os.path.join(folder_path, "*-pressure.csv"))[0])
    except:
        raw_pressure = pd.read_csv(glob.glob(os.path.join(folder_path, "*-pressure.txt"))[0], delimiter="\t")

    # Read the pressure log file and retrieve the strat time in unix timestamp.
    pressure_log = glob.glob(os.path.join(folder_path, "*-pressurelog.txt"))[0]
    with open(pressure_log, "r") as f:
        last_line = f.readlines()[-1]
        start_time_str = last_line.split(" Start Graph ")[0]
        start_time = datetime.strptime(start_time_str, "%m/%d/%Y %H:%M:%S.%f")  # Updated format string
        start_time_unix = int(time.mktime(start_time.timetuple()))

    # Slice and put the pressure file into individual folder
    for unit_num in range(1, 13):
        try:
            df_p = pd.DataFrame()
            df_p.insert(0, "", raw_pressure["Time [s]"] + start_time_unix)
            df_p.insert(1, "Gas(Read)[mbar]", raw_pressure[f"U{unit_num}Gas(Read)[mbar]"])
            df_p.insert(2, "Liquid(Read)[mbar]", raw_pressure[f"U{unit_num}Liquid(Read)[mbar]"])
            df_p.to_csv(os.path.join(folder_path, f"U{unit_num}", f"U{unit_num}_pressure_for_yadg.csv"), index=False)
        except:
            pass


def praser_temp(folder_path):
    raw_temp = pd.read_csv(glob.glob(os.path.join(folder_path, "*-temperature.csv"))[0])

    for unit_num in range(1, 13):
        try:
            df_temp = pd.DataFrame()
            df_temp.insert(0, "", raw_temp["Unnamed: 0"])
            df_temp.insert(1, "Cell temperature (C)", raw_temp[f"Unit {unit_num} Last (C)"])
            df_temp.insert(2, "Room temperature (C)", raw_temp["RT-external Last (C)"])
            df_temp.to_csv(os.path.join(folder_path, f"U{unit_num}", f"U{unit_num}_temperature_for_yadg.csv"), index=False)
        except:
            pass


# Function used to retrieve time stamp from the GC data.
def retrieve_timestamps(zip_filepath):
    timestamps = []

    # Open the zip file
    with zipfile.ZipFile(zip_filepath, "r") as z:
        # Iterate over files in the zip file
        for filename in z.namelist():
            # Only process files with .fusion-data extension
            if filename.endswith(".fusion-data"):
                with z.open(filename) as json_file:
                    data = json.load(json_file)
                    if "runTimeStamp" in data:
                        timestamp = data["runTimeStamp"]
                        # The format should be 'yyyy-mm-ddThh:nn:ss.msZ'
                        if isinstance(timestamp, str) and len(timestamp) >= 24:
                            # Parse string into datetime object
                            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                            # Make it aware of its timezone
                            dt = dt.replace(tzinfo=pytz.UTC)
                            # Convert datetime object to Unix timestamp
                            unix_timestamp = dt.timestamp()
                            timestamps.append(unix_timestamp)

    return timestamps


# Flow data slicing function
def filter_data(df, gc_sampling_times, removal_window=120, collecting_window=340):
    # Initialize a mask to mark the rows to keep
    mask = pd.Series([False] * len(df), index=df.index)
    # For each gc sampling time, update the mask to include the desired data.
    for gc_time in gc_sampling_times:
        mask = mask | (
            (df["Unix timestamp"] > gc_time + removal_window) & (df["Unix timestamp"] < gc_time + removal_window + collecting_window)
        )
    # Apply the mask to the dataframe to filter rows
    new_df = df[mask]
    # Reset the index of the new DataFrame and drop the old index
    new_df = new_df.reset_index(drop=True)

    return new_df


# Convert non-standard flow into nml/min ones
def get_Vstd(Vm, Pm, Tm):
    return (Pm * 10 / 1013.25) * (273.15 / (273.15 + Tm)) * Vm


def flow_adjust(df_flow):
    df_flow_adj = pd.DataFrame({"Unix timestamp": [], "Flow": []})
    for idx in range(0, len(df_flow)):
        if df_flow["Unit of pressure"][idx] == "mBar":
            df_append = pd.DataFrame({"Unix timestamp": [df_flow["Unix timestamp"][idx]], "Flow": [df_flow["Flow"][idx]]})
        elif df_flow["Unit of pressure"][idx] == "kPa":
            df_append = pd.DataFrame(
                {
                    "Unix timestamp": [df_flow["Unix timestamp"][idx]],
                    "Flow": [
                        get_Vstd(
                            Vm=df_flow["Flow"][idx], Pm=df_flow["Measured flow pressure"][idx], Tm=df_flow["Measured flow temperature"][idx]
                        )
                    ],
                }
            )
        df_flow_adj = pd.concat([df_flow_adj, df_append], ignore_index=True)
    return df_flow_adj


def praser_flow(folder_path):
    # Retrive the path to the flow file.
    flow_collection = []
    for filename in os.listdir(folder_path):
        if filename.startswith("collection") and filename.endswith("flow.csv"):
            flow_collection.append(os.path.join(folder_path, filename))

    for flow_path in flow_collection:
        flow_filename = os.path.basename(flow_path)
        name_split = os.path.basename(flow_filename).split("-")
        list_unit = [unit for idx, unit in enumerate(name_split) if idx != 0 and idx != len(name_split) - 1]

        for unit in list_unit:
            # Assgin the GC slot
            flow_raw = flow_adjust(pd.read_csv(flow_path))
            gc_timestamp = retrieve_timestamps(glob.glob(os.path.join(folder_path, unit, "*GC.zip"))[0])
            df_flow_sliced = filter_data(df=flow_raw, gc_sampling_times=gc_timestamp, collecting_window=150)

            # save the sliced flow file into respective folder.
            df_flow_sliced = df_flow_sliced.rename(columns={"Unix timestamp": "Time"})
            df_flow_sliced = df_flow_sliced.rename(columns={"Flow": "Flow (nml per min)"})
            df_flow_sliced.to_csv(os.path.join(folder_path, unit, f"{unit}_flow_for_yadg.csv"), index=False)
