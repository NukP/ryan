"""
This module contains dictionary used to link between system.
"""

# This list of dictionary is used to map the metadata from the analyzed json file to the openbis code
metadata_mapping = [
    {"metadata": "Sample ID", "json_path": "metadata||sample_data||Sample ID", "openbis_code": "sample_id"},
    {"metadata": "Run ID", "json_path": "metadata||sample_data||Run ID", "openbis_code": "run_id"},
    {
        "metadata": "Actual N:P Ratio",
        "json_path": "metadata||sample_data||Actual N:P ratio",
        "openbis_code": "actual_np_ratio",
    },
    {"metadata": "Rack Position", "json_path": "metadata||sample_data||Rack position", "openbis_code": "rack_position"},
    {"metadata": "Separator", "json_path": "metadata||sample_data||Separator", "openbis_code": "separator"},
    {
        "metadata": "Electrolyte Name",
        "json_path": "metadata||sample_data||Electrolyte name",
        "openbis_code": "electrolyte_name",
    },
    {
        "metadata": "Electrolyte Description",
        "json_path": "metadata||sample_data||Electrolyte description",
        "openbis_code": "electrolyte_description",
    },
    {
        "metadata": "Electrolyte Position",
        "json_path": "metadata||sample_data||Electrolyte position",
        "openbis_code": "electrolyte_position",
    },
    {
        "metadata": "Electrolyte Amount (uL)",
        "json_path": "metadata||sample_data||Electrolyte amount (uL)",
        "openbis_code": "electrolyte_amount_ul",
    },
    {
        "metadata": "Electrolyte Dispense Order",
        "json_path": "metadata||sample_data||Electrolyte dispense order",
        "openbis_code": "electrolyte_dispense_order",
    },
    {
        "metadata": "Electrolyte Before Separator (uL)",
        "json_path": "metadata||sample_data||Electrolyte amount before separator (uL)",
        "openbis_code": "electrolyte_amount_before_separator",
    },
    {
        "metadata": "Electrolyte After Separator (uL)",
        "json_path": "metadata||sample_data||Electrolyte amount after separator (uL)",
        "openbis_code": "electrolyte_amount_after_separator",
    },
    {"metadata": "Anode Type", "json_path": "metadata||sample_data||Anode type", "openbis_code": "anode_type"},
    {
        "metadata": "Anode Description",
        "json_path": "metadata||sample_data||Anode description",
        "openbis_code": "anode_description",
    },
    {
        "metadata": "Anode Diameter (mm)",
        "json_path": "metadata||sample_data||Anode diameter (mm)",
        "openbis_code": "anode_diameter_mm",
    },
    {
        "metadata": "Anode Mass (mg)",
        "json_path": "metadata||sample_data||Anode mass (mg)",
        "openbis_code": "anode_mass_mg",
    },
    {
        "metadata": "Anode Current Collector Mass (mg)",
        "json_path": "metadata||sample_data||Anode current collector mass (mg)",
        "openbis_code": "anode_current_collector_mass_mg",
    },
    {
        "metadata": "Anode Active Material Fraction",
        "json_path": "metadata||sample_data||Anode active material mass fraction",
        "openbis_code": "anode_active_material_mass_fraction",
    },
    {
        "metadata": "Anode Active Material Mass (mg)",
        "json_path": "metadata||sample_data||Anode active material mass (mg)",
        "openbis_code": "anode_active_material_mass_mg",
    },
    {
        "metadata": "Anode C-Rate Areal Capacity (mAh/cm²)",
        "json_path": "metadata||sample_data||Anode C-rate definition areal capacity (mAh/cm²)",
        "openbis_code": "anode_crate_definition_areal_capacity",
    },
    {
        "metadata": "Anode Balancing Specific Capacity (mAh/g)",
        "json_path": "metadata||sample_data||Anode balancing specific capacity (mAh/g)",
        "openbis_code": "anode_balancing_specific_capacity",
    },
    {"metadata": "Cathode Type", "json_path": "metadata||sample_data||Cathode type", "openbis_code": "cathode_type"},
    {
        "metadata": "Cathode Description",
        "json_path": "metadata||sample_data||Cathode description",
        "openbis_code": "cathode_description",
    },
    {
        "metadata": "Cathode Diameter (mm)",
        "json_path": "metadata||sample_data||Cathode diameter (mm)",
        "openbis_code": "cathode_diameter_mm",
    },
    {
        "metadata": "Cathode Mass (mg)",
        "json_path": "metadata||sample_data||Cathode mass (mg)",
        "openbis_code": "cathode_mass_mg",
    },
    {
        "metadata": "Cathode Active Material Mass (mg)",
        "json_path": "metadata||sample_data||Cathode active material mass (mg)",
        "openbis_code": "cathode_active_material_mass_mg",
    },
    {
        "metadata": "Cathode C-Rate Areal Capacity (mAh/cm²)",
        "json_path": "metadata||sample_data||Cathode C-rate definition areal capacity (mAh/cm²)",
        "openbis_code": "cathode_crate_definition_areal_capacity",
    },
    {
        "metadata": "Cathode Balancing Specific Capacity (mAh/g)",
        "json_path": "metadata||sample_data||Cathode balancing specific capacity (mAh/g)",
        "openbis_code": "cathode_balancing_specific_capacity",
    },
    {"metadata": "Casing Type", "json_path": "metadata||sample_data||Casing type", "openbis_code": "casing_type"},
    {
        "metadata": "Separator Diameter (mm)",
        "json_path": "metadata||sample_data||Separator diameter (mm)",
        "openbis_code": "separator_diameter_mm",
    },
    {"metadata": "Spacer (mm)", "json_path": "metadata||sample_data||Spacer (mm)", "openbis_code": "spacer_mm"},
]

# This dictionary is used to map between the metadata column in the excelf or ontology and the path in analyzed json file.
dict_excel_to_json = data_dict = {
    "Cell ID": "metadata||sample_data||Sample ID",
    "Separator manufacturer": "metadata||sample_data||Separator",
    "Electrolyte manufacturer": "metadata||sample_data||Electrolyte name",
    "Negative electrode coating active material": "metadata||sample_data||Anode type",
    "Negative electrode diameter": "metadata||sample_data||Anode diameter (mm)",
    "Negative electrode coating active material mass fraction": "metadata||sample_data||Anode active material mass fraction",
    "Positive electrode coating active material": "metadata||sample_data||Cathode type",
    "Positive electrode diameter": "metadata||sample_data||Cathode diameter (mm)",
    "Positive electrode coating active material mass fraction": "metadata||sample_data||Cathode active material mass fraction",
    "Cell case": "metadata||sample_data||Casing type",
    "Separator diameter": "metadata||sample_data||Separator diameter (mm)",
}

# This map the short name of the operator to the full name
user_mapping = {
    "kigr": "Graham Kimbell",
    "svfe": "Enea Svaluto-Ferro",
    "resa": "Sanja Renka",
    "lisc": "Lina Sofie Scholz",
    "maal": "Alan Matiatos",
}
