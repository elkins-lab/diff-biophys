# Default NMR parameters (Vuister & Bax 1993, J. Am. Chem. Soc. 115, 7772-7777)
KARPLUS_A = 6.51
KARPLUS_B = -1.76
KARPLUS_C = 1.60

# Default Ring Current Intensities (Consistent with synth-nmr/SHIFTX2)
RING_INTENSITIES = {
    "PHE": 1.2,
    "TYR": 1.2,
    "TRP": 1.3,
    "HIS": 0.5,
    "HID": 0.5,
    "HIE": 0.5,
    "HIP": 0.5,
}

# Try to pull from synth-nmr if installed
try:
    import synth_nmr.chemical_shifts as sc
    import synth_nmr.j_coupling as sj

    KARPLUS_A = sj.KARPLUS_PARAMS.get("A", KARPLUS_A)
    KARPLUS_B = sj.KARPLUS_PARAMS.get("B", KARPLUS_B)
    KARPLUS_C = sj.KARPLUS_PARAMS.get("C", KARPLUS_C)

    RING_INTENSITIES.update(sc.RING_INTENSITIES)
except (ImportError, AttributeError, KeyError):
    # ImportError  : synth-nmr is not installed (expected in many environments)
    # AttributeError: synth-nmr exists but lacks the expected module attributes
    # KeyError      : KARPLUS_PARAMS dict is missing expected keys
    pass
