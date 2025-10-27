import shutil
import time
import pandas as pd
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from .dynamic_recipe import recifier
from typing import List
from pathlib import Path


def run_and_stream(cmd: List[str]) -> None:
    """
    Run a subprocess command and stream combined stdout+stderr live to Jupyter.
    Raises CalledProcessError on non-zero exit.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    buf = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")     # live output in notebook
        buf.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output="".join(buf)
        )
    

def exclaim_print(word: str, color: str = "yellow") -> None:
    """
    Print a highlighted message surrounded by stars in a chosen color.

    The number of stars above and below the message is equal to
    the length of the message plus 10 characters. The output is
    wrapped with blank lines for visual separation.

    Args:
        word: The message to display.
        color: The color of the text and stars. Supported values are:
               'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
               Defaults to 'yellow'.

    Returns:
        None
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    reset = "\033[0m"
    c = colors.get(color.lower(), colors["yellow"])
    stars = "*" * (len(word) + 10)

    print("\n")
    print(f"{c}{stars} \n")
    print(f"{word} \n")
    print(f"{stars}{reset}")
    print("\n")


def lc_check(folder_path: Path) -> None:
    """
    Check the time offset in the liquid chromatography file.

    Args:
        folder_path: Path to the data folder.

    Returns:
        None
    """
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    try:
        matches = list(folder_path.glob("*LC.xlsx"))
        if not matches:
            return
        df_lc = pd.read_excel(matches[0], sheet_name=1)
        if df_lc["Time offset"].isnull().any():
            if matches[0].exists():
                matches[0].unlink()
            exclaim_print("WARNING!! The time offset of the LC file is incorrect, the LC file will not be processed.")
    except Exception:
        pass


def run_yadg_dgpost(d: Path, show_log: bool = False, manual_recipe: bool = True, clean_temp_dir: bool = True) -> None:
    """
    Run yadg and dgpost on the given data folder.

    Args:
        d: Directory name or Path under Recipe/data_for_dgbowl to process.
        show_log: If True, stream stdout+stderr live and print log (from terminal) codes.
        manual_recipe: If True, copy recipes from Recipe/yadg and Recipe/dgpost; else generate dynamically.
        clean_temp_dir: If True, remove Recipe/data_for_dgbowl/<d> after processing.

    Returns:
        None
    """
    if not isinstance(d, Path):
        d = Path(d)

    base = Path("Recipe") / "data_for_dgbowl" / d
    input_path = base / "data"
    yadg_recipe_dir = base / "recipe" / "yadg"
    dgpost_recipe_dir = base / "recipe" / "dgpost"
    output_dir = Path("Output") / d
    recipe_output_dir = output_dir / "recipe_for_dgbowl"

    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(manual_recipe, bool):
        raise ValueError("manual_recipe should be a boolean value. (True/False)")

    yadg_recipe_dir.mkdir(parents=True, exist_ok=True)
    dgpost_recipe_dir.mkdir(parents=True, exist_ok=True)

    if manual_recipe:
        yadg_src_list = sorted([p for p in Path("Recipe/yadg").iterdir() if p.is_file()])
        if not yadg_src_list:
            raise FileNotFoundError("No yadg recipe found in Recipe/yadg.")
        shutil.copy2(yadg_src_list[0], yadg_recipe_dir)
        for recipe in sorted([p for p in Path("Recipe/dgpost").iterdir() if p.is_file()]):
            shutil.copy2(recipe, dgpost_recipe_dir)
    else:
        recifier.generate_recipe(str(yadg_recipe_dir), str(dgpost_recipe_dir), str(input_path))

    if not recipe_output_dir.exists():
        shutil.copytree(base / "recipe", recipe_output_dir)

    yadg_files = sorted([p for p in yadg_recipe_dir.iterdir() if p.is_file()])
    if not yadg_files:
        raise FileNotFoundError(f"No yadg recipe found in {yadg_recipe_dir}")
    yadg_recipe = yadg_files[0]

    dgpost_recipes = sorted([p for p in dgpost_recipe_dir.iterdir() if p.is_file()])

    try:
        metadata_files = list(input_path.glob("*-metadata.xlsx"))
        if not metadata_files:
            raise FileNotFoundError("Metadata file not found in the data folder.")
        shutil.move(str(metadata_files[0]), output_dir / metadata_files[0].name)
    except:
        pass

    files_to_transfer = []
    for pattern in ["*flow_for_yadg.csv", "*pressure_for_yadg.csv", "*temperature_for_yadg.csv"]:
        match = list(input_path.glob(pattern))
        if match:
            files_to_transfer.append(match[0])
    for f in files_to_transfer:
        shutil.copy2(f, output_dir / f.name)

    lc_check(input_path)
    exclaim_print(f"Start yadg on: {d}")

    yadg_cmd = [
        "yadg", "preset", "-p", "-a",
        str(yadg_recipe),
        str(input_path),
        str(output_dir / f"datagram_{d}.nc"),
        "--ignore-merge-errors",
    ]

    yadg_success = True

    if show_log:
        try:
            run_and_stream(yadg_cmd)
        except subprocess.CalledProcessError as e:
            yadg_success = False
            exclaim_print(f"[FAILED]!! yadg failed for {d}. dgpost will not be run.", color="red")
    else:
        try:
            subprocess.run(yadg_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            yadg_success = False
            exclaim_print(f"[FAILED]!! yadg failed for {d}. dgpost will not be run.", color="red")

    if yadg_success:
        exclaim_print(f"Start dgpost on: {d}")
        for recipe in dgpost_recipes:
            dgpost_cmd = [
                "dgpost",
                str(recipe),
                "--patch",
                str(output_dir / f"datagram_{d}"),
                "-v",
            ]
            if show_log:
                try:
                    run_and_stream(dgpost_cmd)
                except subprocess.CalledProcessError as e:
                    exclaim_print(f"\n[FAILED] dgpost for {d} with recipe {recipe.name}\n", color="red")
            else:
                try:
                    subprocess.run(dgpost_cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError:
                    pass

    if clean_temp_dir:
        try:
            shutil.rmtree(base)
        except Exception:
            time.sleep(2)
            shutil.rmtree(base)


def auto_dgbowl(multi_processing: bool = True, **kwargs) -> None:
    """
    Run yadg and dgpost on all the data folders in the data_for_dgbowl directory.

    Args:
        multi_processing: If True, use multiprocessing to process folders.
        **kwargs: Forwarded to run_yadg_dgpost (e.g., show_log, manual_recipe, clean_temp_dir).

    Returns:
        None
    """
    data_root = Path("Recipe") / "data_for_dgbowl"
    dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    if not dirs:
        print("No data folders found.")
        return None

    if multi_processing:
        num_processes = min(cpu_count(), len(dirs))
        exclaim_print("Start processing yadg/dgpost (Multiprocessing)", color="green")
        print("!!Logs will not be printed in the notebook when using multiprocessing!!")
        print("Please use single processing to see the logs in the notebook.")
        func = partial(run_yadg_dgpost, **kwargs)
        with Pool(processes=num_processes) as pool:
            pool.map(func, [Path(p.name) for p in dirs])
    else:
        exclaim_print("Start processing yadg/dgpost (Using single CPU core)", color="green")
        for p in dirs:
            run_yadg_dgpost(Path(p.name), **kwargs)

    return None
