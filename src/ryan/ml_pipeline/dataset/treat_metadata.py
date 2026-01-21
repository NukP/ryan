"""This moduel contains functions that will be used with custom_columns.process_feature() These fuctions will be used
specifically for treating metadata."""

from . import auxiliary as aux
from . import main


def cathode_catlyst_name(data_package: main.DataPackage) -> str:
    """Get the cathode catalyst name."""
    return aux.get_metadata_cell_value(column_name="Cathode catalyst name", data_package=data_package)


def file_name(data_package: main.DataPackage) -> str:
    """Retrieve filename."""
    return data_package.fln.split(".")[0].split("datagram_")[1]


def cathode_geometric_surface_area(data_package: main.DataPackage) -> float:
    """Retrieve cathode geometric surface area."""
    return aux.get_metadata_cell_value(column_name="Cathode geometric surface area", data_package=data_package)


def ion_exchange_membrane(data_package: main.DataPackage) -> str:
    """Type of ion exchange membrane used to seprate anode and cathode compartment."""
    return aux.get_metadata_cell_value(column_name="Ion exchange membrane type", data_package=data_package)


def cathode_gas_flow(data_package: main.DataPackage) -> float:
    """Cathode gas flow."""
    return aux.get_metadata_cell_value(column_name="Cathode gas flow - SET", data_package=data_package)


def cathode_gas_mix(data_package: main.DataPackage) -> str:
    """Return gas mixture in cathode compartment."""
    string_value = aux.get_metadata_cell_value(
        column_name="Cathode gas mixture and concentration [chemical formula, mol fraction]", data_package=data_package
    )
    converted_dict = aux.pseudolist_conv(string_value)
    gas = list(converted_dict.keys())[0]
    return gas


def cathode_set_ph(data_package: main.DataPackage) -> float:
    """Return Set pH inside cathode compartment."""
    return aux.get_metadata_cell_value(
        column_name="Cathode compartment electrolyte pH - SET", data_package=data_package
    )


def cathodic_chloride_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of Cl- (M) in cathode compartment."""
    return aux.cal_cathode_ions(data_package)["Cl-"]


def cathodic_sulphate_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of sulphate (M) in cathode compartment."""
    return aux.cal_cathode_ions(data_package)["sulphate"]


def cathodic_potassium_ion_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of K+ (M) in cathode compartment."""
    return aux.cal_cathode_ions(data_package)["K+"]


def cathodic_sodium_ion_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of Na+ (M) in cathode compartment."""
    return aux.cal_cathode_ions(data_package)["Na+"]


def cathodic_hydrogencarbonate_ion_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of HCO3- (M) in cathode compartment."""
    return aux.cal_cathode_ions(data_package)["HCO3-"]


def cathode_volume(data_package: main.DataPackage) -> float:
    """Return volume of cathode electrolyte in mL."""
    return aux.get_metadata_cell_value(column_name="Cathode compartment electrolyte volume", data_package=data_package)


def anode_catalyst_name(data_package: main.DataPackage) -> str:
    """Retrun the anode catalyst name."""
    return aux.get_metadata_cell_value(column_name="Anode catalyst name", data_package=data_package)


def anode_electrolyte_flow(data_package: main.DataPackage) -> float:
    """Return the flow rate of anode electrolyte."""
    return aux.get_metadata_cell_value(column_name="Anode compartment electrolyte flow", data_package=data_package)


def anode_electrolyte_ph(data_package: main.DataPackage) -> float:
    """Return the pH of anode electrolyte."""
    return aux.get_metadata_cell_value(column_name="Anode compartment electrolyte pH - SET", data_package=data_package)


def anode_electrolyte_volume(data_package: main.DataPackage) -> float:
    """Return the volume of anode electrolyte."""
    return aux.get_metadata_cell_value(column_name="Anode compartment electrolyte volume", data_package=data_package)


def anode_gas_flow(data_package: main.DataPackage) -> float:
    """Return the flow rate of anode gas."""
    return aux.get_metadata_cell_value(column_name="Anode gas flow - SET", data_package=data_package)


def cathode_catalyst_loading(data_package: main.DataPackage) -> float:
    """Return the loading of cathode catalyst min mg/cm2."""
    return aux.get_metadata_cell_value(column_name="Cathode catalyst loading", data_package=data_package)


def cathode_electrolyte_flow(data_package: main.DataPackage) -> float:
    """Return the flow rate of cathode electrolyte in mL/min."""
    return aux.get_metadata_cell_value(column_name="Cathode compartment electrolyte flow", data_package=data_package)


def cathode_reaction(data_package: main.DataPackage) -> str:
    """Return cathode reaction name."""
    return aux.get_metadata_cell_value(column_name="Cathode reaction", data_package=data_package)


def electrical_connection_type(data_package: main.DataPackage) -> str:
    """Return the type of electrical connection."""
    return aux.get_metadata_cell_value(column_name="Electrical connection type", data_package=data_package)


def cathode_catalyst_thickness(data_package: main.DataPackage) -> float:
    """Return the thickness of cathode catalyst in um."""
    return aux.get_metadata_cell_value(column_name="Cathode catalyst thickness", data_package=data_package)


def cathode_substrate_name(data_package: main.DataPackage) -> str:
    """Return the name of cathode substrate."""
    return aux.get_metadata_cell_value(column_name="Cathode substrate name", data_package=data_package)


def anodic_chloride_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of Cl- (M) in anode compartment."""
    return aux.cal_anode_ions(data_package)["Cl-"]


def anodic_sulphate_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of sulphate (M) in anode compartment."""
    return aux.cal_anode_ions(data_package)["sulphate"]


def anodic_potassium_ion_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of K+ (M) in anode compartment."""
    return aux.cal_anode_ions(data_package)["K+"]


def anodic_hydrogencarbonate_ion_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of HCO3- (M) in anode compartment."""
    return aux.cal_anode_ions(data_package)["HCO3-"]


def anodic_phosphate_ion_conc(data_package: main.DataPackage) -> float:
    """Return the concentration of phosphate (M) in anode compartment."""
    return aux.cal_anode_ions(data_package)["phosphate"]


def anodic_gas_mixture(data_package: main.DataPackage) -> str:
    """Return the gas mixture in anode compartment."""
    string_value = aux.get_metadata_cell_value(
        column_name="Anode gas mixture and concentration [chemical formula, mol fraction]", data_package=data_package
    )
    converted_dict = aux.pseudolist_conv(string_value)
    gas = list(converted_dict.keys())[0]
    return gas


def cathodic_gas_co2_molfrac(data_package: main.DataPackage) -> float:
    """Return the molfraction of CO2 in the cathode compartment."""
    return aux.cal_cathode_gas_misture(data_package)["CO2"]


def cathodic_gas_co_molfrac(data_package: main.DataPackage) -> float:
    """Return the molfraction of CO in the cathode compartment."""
    return aux.cal_cathode_gas_misture(data_package)["CO"]


def cathode_over_layer_bool(data_package: main.DataPackage) -> bool:
    """Return whether or not an overlayer is present on the cathode."""
    cat_overlayer = aux.get_metadata_cell_value(
        column_name="Cathode overlayer components [chemical formula, particle size in µm, concentration in mg/L]",
        data_package=data_package,
    )
    if cat_overlayer == "None":
        return False
    else:
        return True


def cathode_substrate_chemical_formula(data_package: main.DataPackage) -> str:
    """Return the chemical formula of the cathode substrate."""
    cathode_psuedo_list = aux.get_metadata_cell_value(
        column_name="Cathode microporous layer components [chemical formula, pore size in µm, thickness in µm]",
        data_package=data_package,
    )
    return cathode_psuedo_list.split(",")[0].split("[")[1]


def cathode_substrate_pore_size(data_package: main.DataPackage) -> float:
    "Return pore size of the cathode substrate in μm"
    cathode_psuedo_list = aux.get_metadata_cell_value(
        column_name="Cathode microporous layer components [chemical formula, pore size in µm, thickness in µm]",
        data_package=data_package,
    )
    return float(cathode_psuedo_list.split(",")[1])
