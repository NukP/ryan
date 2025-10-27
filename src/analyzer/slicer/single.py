# Slice and prepare files (data from a single run experiment) for yadg and dgpost
import os
import glob
import pandas as pd
import datetime
from datetime import datetime, timedelta
import time
import pytz
import zipfile
import json
import shutil
import traceback


def stage_manager(folder_path, folder_name, show_error=False):
    pass_status = []
    failed_status = []
    if show_error == True:
        try:
            praser_pressure(folder_path)
        except Exception as e:
            traceback.print_exc()
        try:
            praser_temp(folder_path)
        except Exception as e:
            traceback.print_exc()
        try:
            praser_flow_custom(folder_path)
        except:
            try:
                praser_flow_drycal(folder_path)
            except Exception as e:
                traceback.print_exc()
    else:
        try:
            praser_pressure(folder_path)
        except:
            pass
        try:
            praser_temp(folder_path)
        except:
            pass
        try:
            praser_flow_custom(folder_path)
        except:
            try:
                praser_flow_drycal(folder_path)
            except:
                pass

    # Check the success of slicing flow, temp and pressure data.
    status_report = {}
    pass_status = []
    failed_status = []
    if any(filename.endswith("flow_for_yadg.csv") for filename in os.listdir(folder_path)):
        pass_status.append("Flow")
        proceed = True
    else:
        failed_status.append("Flow")
        proceed = False
    if any(filename.endswith("temperature_for_yadg.csv") for filename in os.listdir(folder_path)):
        pass_status.append("Temperature")
    else:
        failed_status.append("Temperature")

    if any(filename.endswith("pressure_for_yadg.csv") for filename in os.listdir(folder_path)):
        pass_status.append("Pressure")
    else:
        failed_status.append("Pressure")
    if proceed == True:
        move_from_folder = folder_path
        move_to_folder = os.path.join(r"Recipe/data_for_dgbowl/", folder_name, "data")
        shutil.copytree(move_from_folder, move_to_folder)
    status_report[folder_name] = {"pass": pass_status, "failed": failed_status, "proceed": proceed}

    return status_report


def praser_temp(folder_path):
    # find the original temp file and read it as a dataframe.
    for fln in os.listdir(folder_path):
        if "-temperature.csv" in fln:
            temp_file_path = os.path.join(folder_path, fln)
    df_original = pd.read_csv(temp_file_path)
    # Create a new dataframe to transfer the desired data
    df_transfer = pd.DataFrame()
    df_transfer.insert(0, "", df_original["Unnamed: 0"])
    for unit_num in range(1, 13):
        try:
            df_transfer.insert(1, "Cell temperature (C)", df_original[f"Unit {unit_num} Last (C)"])
        except:
            pass
    df_transfer.insert(2, "Room temperature (C)", df_original["RT-external Last (C)"])
    # Export df_transfer to csv
    df_transfer.to_csv(os.path.join(folder_path, "temperature_for_yadg.csv"), index=False)


def praser_pressure(folder_path):
    try:
        pressure_file = glob.glob(os.path.join(folder_path, "*-pressure.txt"))[0]
        pressure_df = pd.read_csv(pressure_file, sep="\t")
    except:
        pressure_file = glob.glob(os.path.join(folder_path, "*-pressure.csv"))[0]
        pressure_df = pd.read_csv(pressure_file)
    pressure_log_files = glob.glob(os.path.join(folder_path, "*-pressurelog.txt"))
    if not pressure_log_files:
        raise FileNotFoundError("No pressure log file found.")

    with open(pressure_log_files[0], "r") as f:
        last_line = f.readlines()[-1]
        start_time_str = last_line.split(" Start Graph ")[0]
        start_time = datetime.strptime(start_time_str, "%m/%d/%Y %H:%M:%S.%f")  # Updated format string
        start_time_unix = int(time.mktime(start_time.timetuple()))

    reactor_columns = [col for col in pressure_df.columns if "Gas(Read)[mbar]" in col or "Liquid(Read)[mbar]" in col]

    for column in reactor_columns:
        reactor_name = column[:2]
        new_file_name = f"{reactor_name}-pressure.csv"
        new_file_path = os.path.join(folder_path, new_file_name)
        timestamps = pressure_df["Time [s]"] + start_time_unix

        new_df = pd.DataFrame(
            {
                "": timestamps,
                "Gas(Read)[mbar]": pressure_df[f"{reactor_name}Gas(Read)[mbar]"],
                "Liquid(Read)[mbar]": pressure_df[f"{reactor_name}Liquid(Read)[mbar]"],
            }
        )

    new_df.to_csv(os.path.join(folder_path, "pressure_for_yadg.csv"), index=False)


def drycal_convert(filepath):
    # Read the CSV file, skipping the first 3 lines
    df = pd.read_csv(filepath, skiprows=3)

    # Create new DataFrame
    new_df = pd.DataFrame()

    # Extract date string from filename and convert date string to datetime object
    filename = os.path.basename(filepath)
    try:
        date_str = filename.split("_")[0]
        date = datetime.strptime(date_str, "%Y%m%d")
    except:
        date_str = filename.split("-")[0]
        date = datetime.strptime(date_str, "%Y%m%d")

    # Convert time data to seconds
    time_in_sec = df["Time"].apply(lambda t: datetime.strptime(t, "%I:%M:%S %p").time())
    time_in_sec = time_in_sec.apply(lambda t: timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())

    # Add UTC timestamp to new DataFrame
    new_df["Time"] = date.timestamp() + time_in_sec

    # Add flow data to new DataFrame
    new_df["Flow (nml per min)"] = df["DryCal smL/min "]

    return new_df


# Function used to retrieve time stamp from the GC data. From modified from Empa-17_3
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

    # This function will remove the flow data within the range of GC sampling time.


# filtering function for the flow data recorded from Drycal flow meter program.
def filter_data_drycal(df, gc_sampling_times, removal_window=75):
    # Initialize a mask to mark the rows to keep
    mask = pd.Series([True] * len(df), index=df.index)

    # For each gc sampling time, update the mask to exclude rows within removal window
    for gc_time in gc_sampling_times:
        mask = mask & ((df["Time"] < (gc_time - removal_window)) | (df["Time"] > (gc_time + removal_window)))

    # Apply the mask to the dataframe to filter rows
    new_df = df[mask]

    # Reset the index of the new DataFrame and drop the old index
    new_df = new_df.reset_index(drop=True)

    return new_df


def praser_flow_drycal(folder_path, removal_window=75):
    df_formatted_flow = drycal_convert(glob.glob(os.path.join(folder_path, "*flow.csv"))[0])
    gc_time = retrieve_timestamps(glob.glob(os.path.join(folder_path, "*GC.zip"))[0])
    df_sliced = filter_data_drycal(df=df_formatted_flow, gc_sampling_times=gc_time, removal_window=removal_window)
    df_sliced.to_csv(os.path.join(folder_path, "flow_for_yadg.csv"), index=False)


# filtering function for the flow data recorded from custom-made flow program.
def filter_data_custom(df, gc_sampling_times, removal_window=120, collecting_window=150):
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


# Convert non-standard flow into nml/min ones - Adapted from Empa-17
def get_Vstd(Vm, Pm, Tm):
    return (Pm * 10 / 1013.25) * (273.15 / (273.15 + Tm)) * Vm


# Convert non-standard flow into nml/min ones - Adapted from Empa-17
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


def praser_flow_custom(folder_path, collecting_window=150):
    df_flow = flow_adjust(pd.read_csv(glob.glob(os.path.join(folder_path, "*flow.csv"))[0]))
    gc_time = retrieve_timestamps(glob.glob(os.path.join(folder_path, "*GC.zip"))[0])
    df_flow_sliced = filter_data_custom(df_flow, gc_sampling_times=gc_time, collecting_window=collecting_window)
    df_flow_sliced = df_flow_sliced.rename(columns={"Unix timestamp": "Time"})
    df_flow_sliced = df_flow_sliced.rename(columns={"Flow": "Flow (nml per min)"})
    df_flow_sliced.to_csv(os.path.join(folder_path, "flow_for_yadg.csv"), index=False)
