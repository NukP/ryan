""""
This module contains the templates for the dynamically generated recipes for yadg and dgpost.
"""
  
template_yadg = """metadata:
    provenance:
        type: manual
    version: "4.2"
    timezone: Europe/Berlin
steps:  
  - parser: electrochem
    input:
        folders: ["."]
        suffix: "mpr"
        contains: "PEIS"
    parameters:
        filetype: "eclab.mpr"
    tag: impedance
  - parser: electrochem
    input:
        folders: ["."]
        suffix: "mpr"
    parameters:
        filetype: "eclab.mpr"
        transpose: false
    tag: electrochemistry
  - parser: basiccsv
    input:
        folders: ["."]
        suffix: "csv"
        contains: "flow_for_yadg"
    parameters:
        timestamp:
            uts:
                index: 0
        strip: \\"
        units:
            Flow (nml per min): smL/min
    tag: outlet_flow
  - parser: chromdata
    input:
        folders: ["."]
        suffix: "zip"
        contains: "GC"
    parameters: 
        filetype: "fusion.zip"
    tag: gas_products
  - parser: chromdata
    input:
        folders: ["."]
        suffix: "xlsx"
        contains: "LC"
    externaldate:
        using:
            filename:
                format: "%Y-%m-%d-%H-%M-%S%z"
                len: 24 
    parameters:
        filetype: "empalc.xlsx"
    tag: liquid_products
  - parser: basiccsv
    input:
        folders: ["."]
        suffix: "csv"
        contains: "temperature_for_yadg"
    parameters:
        timestamp:
            uts:
                index: 0
        strip: \\"
        units:
            Cell temperature (C): degC
            Room temperature (C): degC
    tag: temperature
  - parser: basiccsv
    input:
        folders: ["."]
        suffix: "csv"
        contains: "pressure_for_yadg"
    parameters:
        timestamp:
            uts:
                index: 0
        strip: \\"
        units:
            Gas(Read)[mbar]: mbar
            Liquid(Read)[mbar]: mbar
    tag: pressure
    """


template_dgpost_electro = """version: '2.1'
load:
  - as: dg
    path: $patch.nc
    type: netcdf
extract:
  - into: df
    from: dg
    at:
      step: electrochemistry
    columns:
      - key: I
        as: I1
      - key: <I>
        as: I2
      - key: Ewe
        as: Ewe1
      - key: <Ewe>
        as: Ewe2
transform:
  - table: df
    with: table.combine_columns
    using:
      - a: I1
        b: I2
        output: I
        fillnan: true
      - a: Ewe1
        b: Ewe2
        output: Ewe
  - table: df
    with: electrochemistry.charge
    using:
      - I: I
        output: Q
save:
  - table: df
    as: $PATCH.electro.pkl
  - table: df
    as: $PATCH.electro.xlsx
    sigma: False
"""


template_dgpost_peis = """version: '2.1'
load:
  - as: dg
    path: $patch.nc
    type: netcdf
extract:
  - into: peis
    from: dg
    at:
      step: impedance
    columns:
      - key: freq
        as: freq
      - key: Re(Z)
        as: real
      - key: -Im(Z)
        as: imag
      - key: cycle number
        as: cycle
pivot:
  - table: peis
    as: traces
    using: cycle
    columns: [freq, real, imag]
    timestamp: first
transform:
  - table: traces
    with: impedance.lowest_real_impedance
    using:
      - real: real
        imag: imag
save:
  - table: traces
    as: $PATCH.peis.pkl
"""


template_dgpost_gc_main = """version: '2.1'
load:
  - as: dg
    path: $PATCH.nc
    type: netcdf
  - as: electrodata
    path: $PATCH.electro.pkl
    type: table
  - as: peis
    path: $PATCH.peis.pkl
    type: table
extract:
  - into: df
    from: dg
    at:
      step: gas_products
    columns:
      - key: concentration->*
        as: xout
  - into: df
    from: electrodata
    columns:
      - key: I
        as: I
      - key: Ewe
        as: Ewe
  - into: df
    from: dg
    at:
      step: outlet_flow
    columns:
      - key: Flow (nml per min)
        as: fout
  - into: df
    from: peis
    columns:
      - key: min Re(Z)
        as: R
  - into: df
    from: null
    constants:
      - value: {ph}
        as: pH
      - value: 0
        units: degC
        as: Tref
      {template_constant_t}
      - value: 0.197
        units: V
        as: Eref
      - value: 22.5
        units: ml/min
        as: fin
      - value: 1
        as: xin->CO2
  {template_has_temp}
  {template_has_pressure}
transform:
  - table: df
    with: rates.flow_to_molar
    using:
      - flow: fout
        x: xout
        Tref: Tref
        output: nout
      - flow: fin
        x: xin
        Tref: Tref
        output: nin
  - table: df
    with: electrochemistry.nernst
    using:
      - Ewe: Ewe
        Eref: Eref
        R: R
        I: I
        pH: pH
        T: T
  - table: df
    with: electrochemistry.fe
    using:
      - rate: nout
        I: I
        charges:
          C: {charge_C}
          H: 1
          O: -2
save:
  - table: df
    as: $PATCH.GCdata.pkl
  - table: df
    as: $PATCH.GCdata.xlsx
    sigma: False
"""


template_constant_t = """- value: 20
        units: degC
        as: T"""


template_has_temp = """
  - into: df
    from: dg
    at: 
      step: temperature
    columns:
      - key: Cell temperature (C)
        as: T
      - key: Room temperature (C)
        as: T_ambient"""


template_has_pressure = """- into: df
    from: dg
    at:
      step: "pressure"
    columns:
      - key: Gas(Read)[mbar]
        as: P_gas
      - key: Liquid(Read)[mbar]
        as: P_liquid"""

template_dgpost_lc_main = """version: '2.1'
load:
  - as: dg
    path: $PATCH.nc
    type: netcdf
  - as: electrodata
    path: $PATCH.electro.pkl
    type: table   
  - as: peis
    path: $PATCH.peis.pkl
    type: table
extract:
  - into: df
    from: dg
    at:
      step: liquid_products
    columns:
      - key: concentration
        as: concentration
  - into: df
    from: electrodata
    columns:
      - key: Q
        as: Q
      - key: Ewe
        as: Ewe
      - key: I
        as: I  
  - into: df
    from: dg
    at:
      step: outlet_flow
    columns:
      - key: Flow (nml per min)
        as: fout
  - into: df
    from: peis
    columns:
      - key: min Re(Z)
        as: R
  - into: df
    constants:
      - value: {ph}
        as: pH
      - value: 0
        units: degC
        as: Tref
      {template_constant_t}
      - value: 0.197
        units: V
        as: Eref
      - value: 22.5
        units: ml/min
        as: fin
      - value: 1
        as: xin->CO2
      - value: 100
        units: ml
        as: V
  {template_has_temp}
transform:
  - table: df
    with: electrochemistry.average_current
    using:
      - Q: Q
        output: <I>  
  - table: df
    with: rates.flow_to_molar
    using:
      - flow: fin
        x: xin
        Tref: Tref
        output: nin
  - table: df
    with: rates.batch_to_molar
    using:
      - c: concentration
        V: V
        output: nout
  - table: df
    with: electrochemistry.nernst
    using:
      - Ewe: Ewe
        Eref: Eref
        R: R
        I: I
        pH: pH
        T: T
  - table: df
    with: electrochemistry.fe
    using:
      - rate: nout
        I: I
        charges:
          C: {charge_C}
          H: 1
          O: -2
  - table: df
    with: catalysis.conversion
    using:
      - feedstock: CO2
        type: mixed
        rin: nin
        rout: nout
        output: Xpm        
save:
  - table: df
    as: $PATCH.LCdata.pkl
  - table: df
    as: $PATCH.LCdata.xlsx
    sigma: False"""