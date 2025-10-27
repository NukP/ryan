"""
This module contains auxiliary functions for that helps managing files such as clearing directories and copying files.
"""
import os
import shutil
from typing import Dict, List
from .slicer import single, multiplex
import zipfile
import json
from collections import Counter

def clear_dir(output_dir: str) -> None:
    """
    Removes all directories in the specified output directory, 
    except those named '.gitignore'.

    Args:
        output_dir (str): The path to the output directory to clean.

    Returns:
        None
    """
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item != '.gitignore':
            shutil.rmtree(item_path)


def data_slicer(base_dir: str = 'Data/', show_error: bool = False) -> None:
    """
    Processes the data folders for yadg/dgpost status, printing the results for each unit.
    
    Args:
        base_dir (str, optional): The base directory containing data folders. Defaults to 'Data/'.
        show_error (bool, optional): Flag to indicate whether errors from single.stage_manager and multiplex.stage_manager
        should be shown or not. Defaults to False.
    
    Returns:
        None
    """
    print('Data slicing or preparation for yadg,dgpost status: \n')
    
    dirs = [d for d in os.listdir(base_dir) if d != '.DS_Store' and d != 'holder.gitignore']
    
    for file in sorted(dirs):
        print('------------------------------------------------------------------')
        print('Folder: ', file)
        folder_path = os.path.join(base_dir, file)
        
        if file.startswith('Multiplex'):
            status_report = multiplex.stage_manager(folder_path, folder_name=file, show_error=show_error)
        else:
            status_report = single.stage_manager(folder_path, folder_name=file, show_error=show_error)
        
        for unit, status in status_report.items():
            print(f'Successfully sliced: {", ".join(status["pass"])}')
            
            if status["failed"]:
                print(f'Failed to slice: {", ".join(status["failed"])}')

            if status['proceed']:
                print(f'{unit} will be subjected to yadg/dgpost')
            else:
                print(f'!! WARNING: {unit} will NOT be subjected to yadg/dgpost')
            print()
        
        print('------------------------------------------------------------------')


def update_gc_zip_annotation(
    data_folder_dir: str, annotation_name: str, sequence_location: str
) -> None:
    """
    Update annotations in a zip file ending with "-GC.zip" within a given directory.

    Args:
        data_folder_dir (str): Directory containing the data folder.
        annotation_name (str): New value for the "name" field in "annotations".
        sequence_location (str): New value for the "location" field in "sequence".
    """
    # Step 1: Locate the zip file ending with "-GC.zip"
    zip_file_path = None
    for root, _, files in os.walk(data_folder_dir):
        for file in files:
            if file.endswith("-GC.zip"):
                zip_file_path = os.path.join(root, file)
                break
        if zip_file_path:
            break

    if not zip_file_path:
        print("No zip file ending with '-GC.zip' found in the specified directory.")
        return

    print(f"Located zip file: {zip_file_path}")

    # Step 2: Define a temporary directory for unzipping
    temp_dir = os.path.join(data_folder_dir, "temp_unzip")

    try:
        # Ensure the temporary directory is empty before processing
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Step 3: Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Step 4: Update the annotations in all .fusion-data files
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".fusion-data"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Update the "name" field in "annotations"
                    if "annotations" in data and isinstance(data["annotations"], dict):
                        data["annotations"]["name"] = annotation_name

                    # Update the "location" field in "sequence"
                    if "sequence" in data and isinstance(data["sequence"], dict):
                        data["sequence"]["location"] = sequence_location

                    # Save the updated content back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)

        # Step 5: Rezip the content back and overwrite the original zip file
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_ref.write(file_path, os.path.relpath(file_path, temp_dir))

        print(f"Updated zip file saved at: {zip_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Step 6: Clean up temporary files and directories
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")


def analyze_annotations(data_folder_dir: str) -> None:
    """
    Analyze and print the occurrences of unique values in "annotations:name" and "sequence:location"
    within the first zip file ending with "-GC.zip" found in the given directory.

    Args:
        data_folder_dir (str): Directory containing the data folder.
    """
    # Step 1: Locate the zip file ending with "-GC.zip"
    zip_file_path = None
    for root, _, files in os.walk(data_folder_dir):
        for file in files:
            if file.endswith("-GC.zip"):
                zip_file_path = os.path.join(root, file)
                break
        if zip_file_path:
            break

    if not zip_file_path:
        print("No zip file ending with '-GC.zip' found in the specified directory.")
        return

    print(f"Located zip file: {zip_file_path}")

    # Step 2: Initialize counters for annotations:name and sequence:location
    name_counter = Counter()
    location_counter = Counter()

    # Step 3: Analyze the contents of the zip file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                # Only process .fusion-data files
                if file_name.endswith(".fusion-data"):
                    with zip_ref.open(file_name) as file:
                        try:
                            # Load JSON and extract fields
                            data = json.load(file)
                            
                            # Count occurrences of "annotations:name"
                            if "annotations" in data and isinstance(data["annotations"], dict):
                                name = data["annotations"].get("name", "No name field found")
                                name_counter[name] += 1

                            # Count occurrences of "sequence:location"
                            if "sequence" in data and isinstance(data["sequence"], dict):
                                location = data["sequence"].get("location", "No location field found")
                                location_counter[location] += 1

                        except json.JSONDecodeError:
                            # Skip files that are not valid JSON
                            print(f"File {file_name} is not a valid JSON format.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Step 4: Print the variety and occurrences of annotations:name and sequence:location
    print("\nOccurrences of annotations:name:")
    for name, count in name_counter.items():
        print(f"  Name: {name} : Occurrence: {count}")

    print("\nOccurrences of sequence:location:")
    for location, count in location_counter.items():
        print(f"  Location: {location} : Occurrence: {count}")
