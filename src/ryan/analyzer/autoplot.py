from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import gridspec
import os
import glob
from typing import Optional
import traceback

# Auxilary functions
def hue(name: str) -> str:
    """
    Returns the hexadecimal color code corresponding to the given color name. This is used for customize the color for
    graph plotting.

    Args:
        name (str): The name of the color.

    Returns:
        str: The hexadecimal color code.
    """
    dictionary = {
        "royalblue": "#4169e1",
        "C0": "#4169e1",
        "red": "#FD3216",
        "C1": "#FD3216",
        "purple": "#AB63FA",
        "C2": "#AB63FA",
        "orange": "#FFA15A",
        "C3": "#FFA15A",
        "lightgreen": "#B6E880",
        "C4": "#B6E880",
        "lamen": "#3fa3f4",
        "C5": "#3fa3f4",
        "pink": "#FF97FF",
        "C6": "#FF97FF",
        "brown": "#B16833",
        "C7": "#B16833",
        "rose": "#FF6692",
        "C8": "#FF6692",
        "C9": "mediumslateblue",
        "H2": "#4169e1",
        "C2H4": "#FD3216",
        "CO": "#AB63FA",
        "CH4": "#FFA15A",
        "EtOH": "#B6E880",
        "Overall": "#3fa3f4",
    }
    return dictionary[name]


def get_chem_name(name: str) -> str:
    """
    Returns the correct name for the chemical used in the analysis to be displayed in the legend of the graph.

    Args:
        name (str): The name of the chemical.

    Returns:
        str: The correct name for the chemical.
    """
    dictionary = {
        "C3H6": "C$_3$H$_6$",
        "N2O": "N$_2$O",
        "H2": "H$_2$",
        "C3H8": "C$_3$H$_8$",
        "CO2": "CO$_2$",
        "C2H6": "C$_2$H$_6$",
        "C2H4": "C$_2$H$_4$",
        "CH4": "CH$_4$",
        "N2": "N$_2$",
        "O2": "O$_2$",
        "Overall": "FE$_{tot}$ (Gas)",
    }
    try:
        chem_name = dictionary[name]
    except:
        chem_name = name
    return chem_name


def gen_df_feGC(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the dataframe associated with Gas chromatography (GC) results from the dataframe containing all
    experimental results.

    Args:
        input_df (pd.DataFrame): The input dataframe containing all experimental results.

    Returns:
        pd.DataFrame: The dataframe containing the GC results.
    """
    for idx_col, col_name in enumerate(input_df.columns):
        if col_name == "fe":
            idx_fe = idx_col  # locate what is the iloc idx for the column fe
    for idx_col_findFeEnding in range(idx_fe + 1, len(input_df.columns)):
        col_name = input_df.columns[idx_col_findFeEnding]
        if col_name.split(":")[0] != "Unnamed":
            fe_endRange = idx_col_findFeEnding  # locate the end bound of the set of column containing FE information.
            break
    df_feGC = pd.DataFrame()
    for idx_col_findChemicalName in range(idx_fe, fe_endRange):
        col_name = input_df.iloc[0, idx_col_findChemicalName]
        col_data = input_df.iloc[2:, idx_col_findChemicalName]
        df_feGC.insert(len(df_feGC.columns), col_name, col_data)
    df_feGC.insert(len(df_feGC.columns), "Overall", df_feGC.sum(axis=1))
    df_feGC.insert(0, "utx", input_df["Unnamed: 0"][2:])
    df_feGC.insert(1, "time", input_df["Unnamed: 0"][2:] - input_df["Unnamed: 0"][2])
    df_feGC.reset_index(drop=True, inplace=True)
    df_feGC.index += 1
    return df_feGC


def gen_df_feLC(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the dataframe associated with Liquid Chromatography (LC) results from the dataframe containing all
    experimental results.

    Args:
        input_df (pd.DataFrame): The input dataframe containing all experimental results.

    Returns:
        pd.DataFrame: The dataframe containing the LC results.
    """
    for idx_col, col_name in enumerate(input_df.columns):
        if col_name == "fe(LC)":
            idx_fe = idx_col  # locate what is the iloc idx for the column fe
        elif col_name == "fe":
            idx_fe = idx_col
    for idx_col_findFeEnding in range(idx_fe + 1, len(input_df.columns)):
        col_name = input_df.columns[idx_col_findFeEnding]
        if col_name.split(":")[0] != "Unnamed":
            fe_endRange = idx_col_findFeEnding  # locate the end bound of the set of column containing FE information.
            break
    df_feLC = pd.DataFrame()
    for idx_col_findChemicalName in range(idx_fe, fe_endRange):
        col_name = input_df.iloc[0, idx_col_findChemicalName]
        col_data = input_df.iloc[2:, idx_col_findChemicalName]
        df_feLC.insert(len(df_feLC.columns), col_name, col_data)
    df_feLC.insert(len(df_feLC.columns), "Overall", df_feLC.sum(axis=1))
    df_feLC.insert(0, "time", input_df["Unnamed: 0"][2:] - input_df["Unnamed: 0"][2])
    return df_feLC


def get_chem_nameLC(name: str) -> str:
    """
    Retrieve the correct chemical name for the legend in the LC graph. Please include the correct chemical name in
    dictionary here.

    Parameters:
        name (str): The name of the chemical.

    Returns:
        str: The correct chemical name for the legend.
    """
    dictionary = {
        "Overall": "FE$_{tot}$ (Liquid)",
    }
    try:
        chem_name = dictionary[name]
    except:
        chem_name = name
    return chem_name


# Visualization functions
def make_graphLC(path_LC: str, fig: Optional[str] = None, dpi: Optional[int] = None) -> None:
    """
    Plot the result of the experiment with Liquid Chromatography (LC) data.

    Args:
        path_LC (str): The path of the output file (in Excel format).
        fig (Optional[str], optional): If set to None, the graph will only be displayed and not saved. If a file name is provided, the graph will be saved with that name. Defaults to None.
        dpi (Optional[int], optional): The dpi of the output graph should it be exported. Defaults to None.
    """
    df_rawLC = pd.read_excel(path_LC)  # Make a raw df from the LC file.
    df_feLC = gen_df_feLC(df_rawLC)
    ls_chemicalLC = [x for x in df_feLC.columns if x not in ["time"]]

    # Plotting
    # Controlling the overall size of the plot
    fig_LC = plt.figure()
    filename = os.path.basename(path_LC)
    title = filename.split("\\")[-1].split(".xlsx")[0]
    fig_LC.suptitle(title, fontsize=12)
    fig_LC.subplots_adjust(top=0.92)
    spec = gridspec.GridSpec(ncols=1, nrows=1, width_ratios=[1], hspace=0.5, height_ratios=[1])

    # FE from LC
    ax0 = fig_LC.add_subplot(spec[0])
    ax0.minorticks_on()
    ax0.tick_params(which="both", direction="in")
    for idx, chem_name in enumerate(ls_chemicalLC):
        ax0.plot(
            df_feLC["time"] / 60,
            df_feLC[chem_name] * 100,
            marker="o",
            alpha=0.8,
            label=get_chem_nameLC(chem_name),
            color=plt.cm.rainbow(idx / len(ls_chemicalLC)),
        )

    ax0.legend(bbox_to_anchor=(1, 1))
    ax0.set_ylabel("Faradaic efficiency (%)")
    ax0.set_xlabel("Reaction time (min)")
    if fig is not None:
        try:
            fig_LC.savefig(fig, dpi=dpi, bbox_inches="tight")
        except:
            fig_LC.savefig(fig, bbox_inches="tight")
    plt.show(fig_LC)
    plt.close(fig_LC)


def make_graphGC(path: str,
                fig: Optional[str] = None,
                dpi: Optional[int] = None,
                y_label_shift:float = -0.085,
                ytick_label_size:float = 11,
                xtick_label_size:float = 11,
                yaxis_label_size:float = 11
) -> None:
    """
    Plot the result of the experiment with Gas Chromatography (GC) data.

    Args:
        path (str): The path of the output file (in Excel format).
        fig (Optional[str], optional): If set to None, the graph will only be displayed and not saved. 
                                       If a file name is provided, the graph will be saved with that name. 
                                       Defaults to None.
        dpi (Optional[int], optional): The dpi of the output graph should it be exported. Defaults to None.
        y_label_shift (float, optional): Controlling the shift of the ylabel. Defaults to -0.085.
        ytick_label_size (float, optional): Controlling font size of numbers on y-axis. Defaults to 11.
        xtick_label_size (float, optional): Controlling font size of numbers on x-axis. Defaults to 11.
        yaxis_label_size (float, optional): Controlling font size of y-axis label. Defaults to 11.

    Returns:
        None
    """
    directory = os.path.dirname(path)
    try:
        electrochem_file = os.path.join(directory, [file for file in os.listdir(directory) if file.endswith('.electro.xlsx')][0])
    except:
        raise ValueError("No electrochem file found in the directory")
    df_electrochem = pd.read_excel(electrochem_file, skiprows=[1], engine='openpyxl')
    df_electrochem.columns = ['utx_timestamp'] + df_electrochem.columns[1:].tolist()
    try:
        metadata_file = os.path.join(directory, [file for file in os.listdir(directory) if file.endswith('-metadata.xlsx')][0])
    except:
        raise ValueError("No metadata file found in the directory")
    df_metadata = pd.read_excel(metadata_file, sheet_name='Metadata')
    
    dir_flow_file = glob.glob(os.path.join(directory, '*flow_for_yadg.csv'))[0]
    df_flow_raw = pd.read_csv(dir_flow_file)
    

    df_raw = pd.read_excel(path)  # Make a raw df from the GC file.
    # Generating df for plotting
    df_raw["Unnamed: 0"] = pd.to_numeric(df_raw["Unnamed: 0"], errors="coerce")
    df_param = pd.DataFrame()
    df_param.insert(0, "time", df_raw["Unnamed: 0"][2:].reset_index(drop=True) - df_raw["Unnamed: 0"].iloc[2])
    df_param.insert(len(df_param.columns), "Eapp", df_raw["Eapp [V]"][2:].reset_index(drop=True))
    df_param.insert(len(df_param.columns), "Eref", df_raw["Eref [V]"][2:].reset_index(drop=True))
    df_param.insert(len(df_param.columns), "Ewe", df_raw["Ewe [V]"][2:].reset_index(drop=True))
    df_param.insert(len(df_param.columns), "I", df_raw["I [mA]"][2:].reset_index(drop=True))
    df_param.insert(len(df_param.columns), "R", df_raw["R [Ω]"][2:].reset_index(drop=True))
    df_param.insert(len(df_param.columns), "pH", df_raw["pH"][2:].reset_index(drop=True))
    df_param.insert(len(df_param.columns), "T", df_raw["T [°C]"][2:].reset_index(drop=True))
    try:
        df_param.insert(len(df_param.columns), "fout", df_raw["fout [smL/min]"][2:].reset_index(drop=True))
    except:
        df_param.insert(len(df_param.columns), "fout_ml", df_raw["fout [ml/min]"][2:].reset_index(drop=True))
        print("Warning! the flow is not standardized. Please make sure the flow is smL/min")
    try:
        df_param.insert(len(df_param.columns), "Pgas", df_raw["P_gas [mbar]"][2:].reset_index(drop=True))
        df_param.insert(len(df_param.columns), "Pliquid", df_raw["P_liquid [mbar]"][2:].reset_index(drop=True))
        dir_pressure_file = glob.glob(os.path.join(directory, '*pressure_for_yadg.csv'))[0]
        df_pressure_raw = pd.read_csv(dir_pressure_file)
        P_data = True
    except:
        P_data = False
    try:
        df_param.insert(len(df_param.columns), "T_am", df_raw["T_ambient [°C]"][2:].reset_index(drop=True))
        df_temp_file = glob.glob(os.path.join(directory, '*temperature_for_yadg.csv'))[0]
        df_temp_raw = pd.read_csv(df_temp_file)
    except:
        pass

    # Setiing xlim for the graph
    tmin = df_param["time"].min()/60
    tmax = df_param["time"].max()/60
    xmin = round(tmin - 0.05 * (tmax - tmin))
    xmax = round(tmax + 0.05 * (tmax - tmin))

    start_time = df_raw["Unnamed: 0"].iloc[2]

    plt.rcParams['ytick.labelsize'] = ytick_label_size  # Y-axis tick labels
    plt.rcParams['xtick.labelsize'] = xtick_label_size  # X-axis tick labels
    plt.rcParams['axes.labelsize'] = yaxis_label_size  # Y-axis labels

    # Extracting metadata
    electrolyte_ph = float(df_metadata[df_metadata['Metadata'] == 'Cathode compartment electrolyte pH - SET']['Value'].values[0])
    catholyte_volume = df_metadata[df_metadata['Metadata'] == 'Cathode compartment electrolyte volume']['Value'].values[0]
    catodic_electrolyte = df_metadata[df_metadata['Metadata'] == 'Cathode electrolyte compartment solute content [name, concentration in M]']['Value'].values[0]
    anolyte_volume = df_metadata[df_metadata['Metadata'] == 'Anode compartment electrolyte volume']['Value'].values[0]
    anodic_electrolyte = df_metadata[df_metadata['Metadata'] == 'Anode electrolyte compartment solute content [name, concentration in M]']['Value'].values[0]

    # Generate df containf Faradaic efficiency from the GC.
    df_feGC = gen_df_feGC(df_raw)
    # Generate the list of major and minor product
    ls_chemical = df_feGC.columns
    ls_major_product = ["H2", "C2H4", "CO", "CH4", "EtOH"]  # Input the chemical name of the major product here.
    ls_remove = ls_major_product + ["time", "Overall", "CO2", "utx"]
    ls_minor_product = [x for x in ls_chemical if x not in ls_remove]
    
    # Plotting
    # Controlling the overall size of the plot,
    fig_GC = plt.figure()
    
    # Draw vertical line for legend adjustment
    # fig_GC.canvas.draw()
    # #line_position = 0.065  # For yaxis label calibration
    # line_position = 0.918  # For label calibration
    # line = plt.Line2D([line_position, line_position], [0, 1], transform=fig_GC.transFigure, color="red", linewidth=1, linestyle="-")
    # fig_GC.add_artist(line)
    
    
    try:
        filename = os.path.basename(path)
        title = filename.split(".xlsx")[0].split("datagram_")[-1].split(".GCdata")[0] 
    except:
        filename = os.path.basename(path)
        title = filename.split(".xlsx")[0] 
    fig_GC.suptitle(title, fontsize=12)
    fig_GC.subplots_adjust(top=0.95)
    r_exp = 0.75 # Ratio for the experimental data
    r0 = 1.3 # Ratio for the other graphs that are not FE plot
    r1 = 3 # ratio for the FE plot
    if P_data is False:
        fig_GC.set_figheight(17)
        fig_GC.set_figwidth(8)
        spec = gridspec.GridSpec(ncols=1, nrows=8, width_ratios=[1], hspace=0.3, height_ratios=[r_exp, r0, r0, r0, r0, r0, r1, r1])
    elif P_data is True:
        fig_GC.set_figheight(19)
        fig_GC.set_figwidth(8)
        spec = gridspec.GridSpec(ncols=1, nrows=9, width_ratios=[1], hspace=0.3, height_ratios=[r_exp, r0, r0, r0, r0, r0, r0, r1, r1])

    graph_position = 0
    # Text box for metadata
    ax_exp = fig_GC.add_subplot(spec[graph_position])
    ax_exp.axis('off')
    textstr = '\n'.join((
    f'Cathode pH = {electrolyte_ph:.2f}',
    f'Cathode volume = {catholyte_volume} mL',
    f'Cathode solute content [salt, molar concentration] = {catodic_electrolyte}',
    f'Anode volume = {anolyte_volume} mL',
    f'Anode solute content [salt, molar concentration] = {anodic_electrolyte}'
))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_exp.text(0.15, 1.3, textstr, transform=ax_exp.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, linespacing=1.5)
    graph_position += 1


    # Potential
    ax_eapp = fig_GC.add_subplot(spec[graph_position])
    ax_eapp.minorticks_on()
    ax_eapp.tick_params(which="both", direction="in")
    ax_eapp.scatter(df_param["time"] / 60, df_param["Eapp"], color="orangered", marker="o", alpha=0.8, label = "E$_{app}$ - Interpolated data")
    ax_eapp.scatter(df_param["time"] / 60, df_param["Eref"], color="royalblue", marker="o", alpha=0.8, label = "E$_{ref}$ - Interpolated data")
    ax_eapp.scatter(df_param["time"] / 60, df_param["Ewe"], color="darkorange", marker="o", alpha=0.8, label = "E$_{WE}$ - Interpolated data")
    ax_eapp.plot((df_electrochem["utx_timestamp"]-start_time) / 60, df_electrochem["Ewe [V]"], color="orange", alpha=0.5, label = "E$_{WE}$ - Raw data")
    ax_eapp.set_ylabel("Potential (V)")
    ax_eapp.yaxis.set_label_coords(y_label_shift, 0.5)
    ax_eapp.set_xlim(xmin, xmax)
    ax_eapp.legend(loc="upper right", bbox_to_anchor=(1.380, 1.06))
    graph_position += 1

    # Current
    ax_i = fig_GC.add_subplot(spec[graph_position])
    ax_i.minorticks_on()
    ax_i.tick_params(which="both", direction="in")
    ax_i.scatter(df_param["time"] / 60, df_param["I"], color="darkorange", marker="o", alpha=0.8, label = "Interpolated data")
    ax_i.plot((df_electrochem["utx_timestamp"]-start_time) / 60, df_electrochem["I [mA]"], color="orange", alpha=0.8, label = "Raw data")
    ax_i.set_ylabel("I (mA)")
    ax_i.yaxis.set_label_coords(y_label_shift, 0.5)
    ax_i.set_xlim(xmin, xmax)
    ax_i.legend(loc="upper right", bbox_to_anchor=(1.31, 1.06))
    graph_position += 1

    # Temperatur_temp
    ax_temp = fig_GC.add_subplot(spec[graph_position])
    ax_temp.minorticks_on()
    ax_temp.tick_params(which="both", direction="in")
    ax_temp.set_ylabel("T (°C)")
    ax_temp.yaxis.set_label_coords(y_label_shift, 0.5)
    if df_param["T"].std() == 0:
        ax_temp.plot(df_param["time"] / 60, df_param["T"], color="red", marker="o", alpha=0.8, label="Default temperature")
        ax_temp.legend(loc="upper right")
        ax_temp.legend(loc="upper right", bbox_to_anchor=(1.345, 1.06))
    elif df_param["T"].std() != 0:
        ax_temp.scatter(df_param["time"] / 60, df_param["T"], color="darkorange", marker="o", alpha=0.8, label="Cell temparature - Interpolated data")
        ax_temp.plot((df_temp_raw.iloc[:,0]-start_time)/60, df_temp_raw['Cell temperature (C)'], color="darkorange", alpha=0.8, label="Cell temperature - Raw data")
        ax_temp.legend(loc="upper right", bbox_to_anchor=(1.525, 1.06))
    try:
        ax_temp.scatter(df_param["time"] / 60, df_param["T_am"], color="royalblue", marker="o", alpha=0.8, label="Room temparature - Interpolated data")
        ax_temp.plot((df_temp_raw.iloc[:,0]-start_time)/60, df_temp_raw['Room temperature (C)'], color="royalblue", alpha=0.8, label="Room temperature - Raw data")
    except:
        pass
    ax_temp.set_xlim(xmin, xmax)
    
    graph_position += 1

    # Flow
    ax_flow = fig_GC.add_subplot(spec[graph_position])
    ax_flow.minorticks_on()
    ax_flow.tick_params(which="both", direction="in")
    ax_flow.set_xlim(xmin, xmax)
    try:
        ax_flow.scatter(df_param["time"] / 60, df_param["fout"], color="darkorange", marker="o", alpha=0.8, label="f$_{out}$ - Interpolated data")
        
    except:
        ax_flow.scatter(df_param["time"] / 60, df_param["fout_ml"], color="darkorange", marker="o", alpha=0.8, label="f$_{out}$ - Interpolated data")
    ax_flow.plot((df_flow_raw.iloc[:,0]-start_time)/60, df_flow_raw['Flow (nml per min)'], color="darkorange", alpha=0.8, label="f$_{out}$ - Raw data")    
    ax_flow.set_ylabel("f$_{out}$ (mL/min)")
    ax_flow.yaxis.set_label_coords(y_label_shift, 0.5)
    ax_flow.legend(loc="upper right", bbox_to_anchor=(1.367, 1.06))
    graph_position += 1

    # Resistance
    ax_r = fig_GC.add_subplot(spec[graph_position])
    ax_r.minorticks_on()
    ax_r.tick_params(which="both", direction="in")
    ax_r.plot(df_param["time"] / 60, df_param["R"], color="darkorange", marker="o", alpha=0.8)
    ax_r.set_ylabel("R (Ω)")
    ax_r.yaxis.set_label_coords(y_label_shift, 0.5)
    ax_r.set_xlim(xmin, xmax)
    graph_position += 1

    # Pressure
    if P_data is True:
        ax_p = fig_GC.add_subplot(spec[graph_position])
        ax_p.minorticks_on()
        ax_p.tick_params(which="both", direction="in")
        ax_p.scatter(df_param["time"] / 60, df_param["Pgas"], color="orangered", marker="o", alpha=0.8, label="P$_{gas}$ - Interpolated data")
        ax_p.plot((df_pressure_raw.iloc[:,0]-start_time)/60, df_pressure_raw['Gas(Read)[mbar]'], color="orangered", alpha=0.8, label="P$_{gas}$ - Raw data")
        ax_p.scatter(df_param["time"] / 60, df_param["Pliquid"], color="royalblue", marker="o", alpha=0.8, label="P$_{liquid}$ - Interpolated data")
        ax_p.plot((df_pressure_raw.iloc[:,0]-start_time)/60, df_pressure_raw['Liquid(Read)[mbar]'], color="royalblue", alpha=0.5, label="P$_{liquid}$ - Raw data")
        ax_p.set_ylabel("Pressure (mbar)")
        ax_p.yaxis.set_label_coords(y_label_shift, 0.5)
        ax_p.legend(loc="upper right", bbox_to_anchor=(1.39, 1.06))
        ax_p.set_xlim(xmin, xmax)
        graph_position += 1

    # Faradaic efficiency (From GC) - Major prodcuts
    # The Overall fe include all of the Faradaic efficiency from both major and minor products.
    ax_fe_major = fig_GC.add_subplot(spec[graph_position])
    ax_fe_major.minorticks_on() 
    ax_fe_major.tick_params(which='both', direction='in')
    

    for idx_chemical, chemical_name in enumerate(ls_major_product + ["Overall"]):
        ax_fe_major.plot(
            df_feGC["time"] / 60,
            df_feGC[chemical_name] * 100,
            label=get_chem_name(chemical_name),
            marker="o",
            alpha=0.8,
            color=hue(f"C{idx_chemical}"),
        )

    ax_fe_major.legend(loc="upper right", bbox_to_anchor=(1.235, 1.027))
    ax_fe_major.set_ylim(-5, 200)
    ax_fe_major.set_ylabel("Faradaic efficiency (%)")
    ax_fe_major.yaxis.set_label_coords(y_label_shift, 0.5)
    ax_fe_major.set_title("FE, Major products (GC)")
    ax_fe_major.set_xlim(xmin, xmax)
    graph_position += 1

    # Faradaic efficiency (From GC) - Minor prodcuts
    ax_fe_minor = fig_GC.add_subplot(spec[graph_position])
    ax_fe_minor.minorticks_on()
    ax_fe_minor.tick_params(which="minor", direction="in")
    ax_fe_minor.tick_params(direction="in")

    for idx_chemical, chemical_name in enumerate(ls_minor_product):
        ax_fe_minor.plot(
            df_feGC["time"] / 60,
            df_feGC[chemical_name] * 100,
            label=get_chem_name(chemical_name),
            marker="o",
            alpha=0.8,
            color=plt.cm.rainbow(idx_chemical / len(ls_minor_product)),
        )

    ax_fe_minor.legend(loc="upper right", bbox_to_anchor=(1.305, 1.025))
    ax_fe_minor.set_ylabel("Faradaic efficiency (%)")
    ax_fe_minor.yaxis.set_label_coords(y_label_shift, 0.5)
    ax_fe_minor.set_xlabel("Reaction time (min)")
    ax_fe_minor.set_title("FE, Minor products (GC)")
    ax_fe_minor.set_ylim(-0.3, 5)
    ax_fe_minor.set_xlim(xmin, xmax)
    graph_position += 1

    # Exporting figure
    if fig is not None:
        try:
            fig_GC.savefig(fig, dpi=dpi, bbox_inches="tight")
        except:
            fig_GC.savefig(fig, bbox_inches="tight")
    plt.show(fig_GC)
    plt.close(fig_GC)



def dir_make_graph(
    dir: str = 'Output', type: str = "GC", save_fig: bool = False, format: str = "svg", dpi: int = 300, path_output: str = None, **kwargs
) -> None:
    """
    Plot graphs for files in the given directory.

    Args:
        dir (str): The directory where the result files from the analysis are stored.
        type (str, optional): The type of graph to plot. Valid options are 'GC' and 'LC'. Defaults to 'GC'.
        save_fig (bool, optional): If True, save the graph with the file format indicated in the format parameter. Defaults to False.
        format (str, optional): The file format to save the graph in. Defaults to 'svg'.
        dpi (int, optional): The DPI (dots per inch) of the saved graph. Defaults to 300.
        path_output (str, optional): The path to save the graph. If None, the graph will be saved in the same directory as the Jupyter notebook that runs this function.
        **kwargs: Additional keyword arguments passed to the `make_graphGC` function. These can include:
            - y_label_shift (float): Controlling the shift of the ylabel.
            - ytick_label_size (float): Controlling font size of numbers on y-axis.
            - xtick_label_size (float): Controlling font size of numbers on x-axis.
            - yaxis_label_size (float): Controlling font size of y-axis label.

    Returns:
        None
    """
    if type == "GC":
        substr = "GCdata.xlsx"
    elif type == "LC":
        substr = "LCdata.xlsx"
    else:
        raise ValueError("Invalid graph type. The valid graph types are: GC or LC")

    matching_files = []
    if not os.path.isdir(dir):
        print(f"The path '{dir}' is not a valid directory.")
    else:
        for root, _, files in os.walk(dir):
            for file in files:
                if substr in file:
                    matching_files.append(os.path.join(root, file))

        if not matching_files:
            print(f"No files containing '{substr}' were found in the given path.")

    for file_path in sorted(matching_files):
        if type == "GC":
            if save_fig is False:
                try:
                    make_graphGC(path=file_path, **kwargs)
                except Exception as e:
                    print("There is an error while trying to plot: " + file_path + ". Please check the input file")
                    traceback.print_exc() 
            elif save_fig is True:
                if path_output is None:
                    if os.path.isdir(os.path.join(dir, "Graph Export")):
                        path_output = os.path.join(dir, "Graph Export")
                    else:
                        os.mkdir(os.path.join(dir, "Graph Export"))
                        path_output = os.path.join(dir, "Graph Export")
                file_name = os.path.basename(file_path).split(".xlsx")[0]
                file_name_full = file_name + "." + format
                fig = os.path.join(path_output, file_name_full)
                try:
                    make_graphGC(file_path, fig=fig, dpi=dpi, **kwargs)
                except Exception as e:
                    print("There is an error while trying to plot: " + file_path + ". Please check the input file")
                    traceback.print_exc() 
            else:
                print("save_fig can only be True or False")
                return ()
        elif type == "LC":
            if save_fig is False:
                try:
                    make_graphLC(file_path)
                except:
                    print("There is an error while trying to plot: " + file_path + " Please check the input file")
            elif save_fig is True:
                if path_output is None:
                    if os.path.isdir(os.path.join(dir, "Graph Export")):
                        path_output = os.path.join(dir, "Graph Export")
                    else:
                        os.mkdir(os.path.join(dir, "Graph Export"))
                        path_output = os.path.join(dir, "Graph Export")
                file_name = os.path.basename(file_path).split(".xlsx")[0]
                file_name_full = file_name + "." + format
                fig = os.path.join(path_output, file_name_full)
                make_graphLC(file_path, fig=fig, dpi=dpi)
            else:
                print("save_fig can only be True or False")
                return ()
        else:
            print("Invalid file type. The supported file types are 'GC' and 'LC'")
            return ()
