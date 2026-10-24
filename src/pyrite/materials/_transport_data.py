"""Leaf transport element data shared by catalog validation and Monte Carlo.

``Z`` and ``A`` use CIAAW's 2024 standard atomic weights; ``J_keV`` and the
Sternheimer density-effect coefficients use the PDG Atomic and Nuclear
Properties tables (themselves based on the ICRU stopping-power compilation):

* https://ciaaw.org/atomic-weights.htm
* https://pdg.lbl.gov/2025/AtomicNuclearProperties/
"""

TRANSPORT_ELEMENTS = {
    "B": {"Z": 5, "A": 10.81, "J_keV": 0.076},
    "C": {"Z": 6, "A": 12.011, "J_keV": 0.078},
    "N": {"Z": 7, "A": 14.007, "J_keV": 0.082},
    "P": {"Z": 15, "A": 30.974, "J_keV": 0.173},
    "Si": {"Z": 14, "A": 28.085, "J_keV": 0.173},
    "Ge": {"Z": 32, "A": 72.630, "J_keV": 0.350},
    "Se": {"Z": 34, "A": 78.971, "J_keV": 0.348},
    "Te": {"Z": 52, "A": 127.60, "J_keV": 0.485},
    "S": {"Z": 16, "A": 32.06, "J_keV": 0.180},
    "Ti": {"Z": 22, "A": 47.867, "J_keV": 0.233},
    "V": {"Z": 23, "A": 50.9415, "J_keV": 0.245},
    "Fe": {"Z": 26, "A": 55.845, "J_keV": 0.286},
    "Mo": {"Z": 42, "A": 95.95, "J_keV": 0.424},
    "Nb": {"Z": 41, "A": 92.906, "J_keV": 0.417},
    "Pd": {"Z": 46, "A": 106.42, "J_keV": 0.477},
    "W": {"Z": 74, "A": 183.84, "J_keV": 0.727},
    "Zr": {"Z": 40, "A": 91.224, "J_keV": 0.393},
    "Hf": {"Z": 72, "A": 178.49, "J_keV": 0.705},
    "Ta": {"Z": 73, "A": 180.94788, "J_keV": 0.718},
    "Re": {"Z": 75, "A": 186.207, "J_keV": 0.736},
    "Pt": {"Z": 78, "A": 195.08, "J_keV": 0.790},
    "Bi": {"Z": 83, "A": 208.98040, "J_keV": 0.823},
    "O": {"Z": 8, "A": 15.999, "J_keV": 0.095},
    "Al": {"Z": 13, "A": 26.982, "J_keV": 0.166},
    "Li": {"Z": 3, "A": 6.94, "J_keV": 0.040},
    "F": {"Z": 9, "A": 18.998403163, "J_keV": 0.115},
    "H": {"Z": 1, "A": 1.008, "J_keV": 0.0218},
    "Ar": {"Z": 18, "A": 39.948, "J_keV": 0.188},
    "Cr": {"Z": 24, "A": 51.9961, "J_keV": 0.257},
    "Mn": {"Z": 25, "A": 54.938044, "J_keV": 0.272},
    "Ni": {"Z": 28, "A": 58.6934, "J_keV": 0.311},
    "Cu": {"Z": 29, "A": 63.546, "J_keV": 0.322},
    "Au": {"Z": 79, "A": 196.966569, "J_keV": 0.79},
    "Pb": {"Z": 82, "A": 207.2, "J_keV": 0.823},
    "Na": {"Z": 11, "A": 22.98976928, "J_keV": 0.149},
    "Ca": {"Z": 20, "A": 40.078, "J_keV": 0.191},
    "Ga": {"Z": 31, "A": 69.723, "J_keV": 0.334},
    "As": {"Z": 33, "A": 74.921595, "J_keV": 0.347},
    "Cd": {"Z": 48, "A": 112.414, "J_keV": 0.469},
    "I": {"Z": 53, "A": 126.90447, "J_keV": 0.491},
    "Cs": {"Z": 55, "A": 132.90545196, "J_keV": 0.488},
    "Ba": {"Z": 56, "A": 137.327, "J_keV": 0.491},
}

# Sternheimer density-effect coefficients, read from the header of each
# element's PDG muon energy-loss table (``MUE/muE_<slug>.txt``); the trailing
# comment names the slug. The PDG ``I`` in those same headers agrees with
# ``J_keV`` above for every element except Pd, where PDG reads 470.0 eV against
# the 477.0 eV carried here -- see the transport ledger.
#
# NOTHING IN THE TRANSPORT PATH READS THIS TABLE. delta is omitted from
# eq-stopping-bs (every call site passes 0.0); these coefficients exist to
# *bound* the size of that omission, which is what
# ``transport.sternheimer_delta`` computes and
# ``test_stopping_density_effect.py`` pins.
#
# They are per ELEMENT in its own elemental solid. delta is a bulk property of
# the medium, so it does not Bragg-add and these values are not a compound's
# delta; N, O, H, F and Ar, which appear in the catalog only inside solids or
# fluids, are listed at their liquid-phase parameters because no elemental-solid phase exists.
STERNHEIMER_DENSITY_EFFECT = {
    "B": {
        "a": 0.5622,
        "k": 2.4512,
        "x0": 0.0305,
        "x1": 1.9688,
        "C_bar": 2.8477,
        "delta0": 0.14,
    },  # boron_B
    "C": {
        "a": 0.2076,
        "k": 2.9532,
        "x0": -0.009,
        "x1": 2.4817,
        "C_bar": 2.8926,
        "delta0": 0.14,
    },  # carbon_graphite_C
    "N": {
        "a": 0.5329,
        "k": 3.0,
        "x0": 0.3039,
        "x1": 2.0,
        "C_bar": 3.9996,
        "delta0": 0.0,
    },  # nitrogen_liquid
    "P": {
        "a": 0.2361,
        "k": 2.9158,
        "x0": 0.1696,
        "x1": 2.7815,
        "C_bar": 4.5214,
        "delta0": 0.14,
    },  # phosphorus_P
    "Si": {
        "a": 0.1492,
        "k": 3.2546,
        "x0": 0.2015,
        "x1": 2.8716,
        "C_bar": 4.4355,
        "delta0": 0.14,
    },  # silicon_Si
    "Ge": {
        "a": 0.0719,
        "k": 3.3306,
        "x0": 0.3376,
        "x1": 3.6096,
        "C_bar": 5.1411,
        "delta0": 0.14,
    },  # germanium_Ge
    "Se": {
        "a": 0.0657,
        "k": 3.4317,
        "x0": 0.2258,
        "x1": 3.6264,
        "C_bar": 5.321,
        "delta0": 0.1,
    },  # selenium_Se
    "Te": {
        "a": 0.1382,
        "k": 3.0354,
        "x0": 0.3296,
        "x1": 3.4418,
        "C_bar": 5.7131,
        "delta0": 0.14,
    },  # tellurium_Te
    "S": {
        "a": 0.3399,
        "k": 2.6456,
        "x0": 0.158,
        "x1": 2.7159,
        "C_bar": 4.6659,
        "delta0": 0.14,
    },  # sulfur_S
    "Ti": {
        "a": 0.1566,
        "k": 3.0302,
        "x0": 0.0957,
        "x1": 3.0386,
        "C_bar": 4.445,
        "delta0": 0.12,
    },  # titanium_Ti
    "V": {
        "a": 0.1544,
        "k": 3.0163,
        "x0": 0.0691,
        "x1": 3.0322,
        "C_bar": 4.2659,
        "delta0": 0.14,
    },  # vanadium_V
    "Fe": {
        "a": 0.1468,
        "k": 2.9632,
        "x0": -0.0012,
        "x1": 3.1531,
        "C_bar": 4.2911,
        "delta0": 0.12,
    },  # iron_Fe
    "Mo": {
        "a": 0.1053,
        "k": 3.2549,
        "x0": 0.2267,
        "x1": 3.2784,
        "C_bar": 4.8793,
        "delta0": 0.14,
    },  # molybdenum_Mo
    "Nb": {
        "a": 0.1388,
        "k": 3.093,
        "x0": 0.1785,
        "x1": 3.2201,
        "C_bar": 5.0141,
        "delta0": 0.14,
    },  # niobium_Nb
    "Pd": {
        "a": 0.2418,
        "k": 2.7239,
        "x0": 0.0563,
        "x1": 3.0555,
        "C_bar": 4.9358,
        "delta0": 0.14,
    },  # palladium_Pd
    "W": {
        "a": 0.1551,
        "k": 2.8447,
        "x0": 0.2167,
        "x1": 3.496,
        "C_bar": 5.4059,
        "delta0": 0.14,
    },  # tungsten_W
    "Zr": {
        "a": 0.0718,
        "k": 3.4533,
        "x0": 0.2957,
        "x1": 3.489,
        "C_bar": 5.1774,
        "delta0": 0.14,
    },  # zirconium_Zr
    "Hf": {
        "a": 0.2292,
        "k": 2.6155,
        "x0": 0.1965,
        "x1": 3.4337,
        "C_bar": 5.7139,
        "delta0": 0.14,
    },  # hafnium_Hf
    "Ta": {
        "a": 0.178,
        "k": 2.7623,
        "x0": 0.2117,
        "x1": 3.4805,
        "C_bar": 5.5262,
        "delta0": 0.14,
    },  # tantalum_Ta
    "Re": {
        "a": 0.1518,
        "k": 2.8627,
        "x0": 0.0559,
        "x1": 3.4845,
        "C_bar": 5.3445,
        "delta0": 0.08,
    },  # rhenium_Re
    "Pt": {
        "a": 0.1113,
        "k": 3.0417,
        "x0": 0.1484,
        "x1": 3.6212,
        "C_bar": 5.4732,
        "delta0": 0.12,
    },  # platinum_Pt
    "Bi": {
        "a": 0.0941,
        "k": 3.1671,
        "x0": 0.4152,
        "x1": 3.8248,
        "C_bar": 6.3505,
        "delta0": 0.14,
    },  # bismuth_Bi
    "O": {
        "a": 0.5223,
        "k": 3.0,
        "x0": 0.2868,
        "x1": 2.0,
        "C_bar": 3.9471,
        "delta0": 0.0,
    },  # oxygen_liquid
    "Al": {
        "a": 0.0802,
        "k": 3.6345,
        "x0": 0.1708,
        "x1": 3.0127,
        "C_bar": 4.2395,
        "delta0": 0.12,
    },  # aluminum_Al
    "H": {
        "a": 0.1348,
        "k": 5.6249,
        "x0": 0.44,
        "x1": 1.8856,
        "C_bar": 3.0977,
        "delta0": 0.0,
    },  # hydrogen_liquid
    "Li": {
        "a": 0.9514,
        "k": 2.4993,
        "x0": 0.1304,
        "x1": 1.6397,
        "C_bar": 3.1221,
        "delta0": 0.14,
    },  # lithium_Li
    "F": {
        "a": 0.145,
        "k": 3.0,
        "x0": 0.2,
        "x1": 3.0,
        "C_bar": 4.105,
        "delta0": 0.0,
    },  # fluorine_liquid
    "Ar": {
        "a": 0.1956,
        "k": 3.0,
        "x0": 0.2,
        "x1": 3.0,
        "C_bar": 5.2146,
        "delta0": 0.0,
    },  # argon_liquid
    "Cr": {
        "a": 0.1542,
        "k": 2.9896,
        "x0": 0.034,
        "x1": 3.0451,
        "C_bar": 4.1781,
        "delta0": 0.14,
    },  # chromium_Cr
    "Mn": {
        "a": 0.1497,
        "k": 2.9796,
        "x0": 0.0447,
        "x1": 3.1074,
        "C_bar": 4.2702,
        "delta0": 0.14,
    },  # manganese_Mn
    "Ni": {
        "a": 0.165,
        "k": 2.843,
        "x0": -0.0566,
        "x1": 3.1851,
        "C_bar": 4.3115,
        "delta0": 0.1,
    },  # nickel_Ni
    "Cu": {
        "a": 0.1434,
        "k": 2.9044,
        "x0": -0.0254,
        "x1": 3.2792,
        "C_bar": 4.419,
        "delta0": 0.08,
    },  # copper_Cu
    "Au": {
        "a": 0.0976,
        "k": 3.1101,
        "x0": 0.2021,
        "x1": 3.6979,
        "C_bar": 5.5747,
        "delta0": 0.14,
    },  # gold_Au
    "Pb": {
        "a": 0.0936,
        "k": 3.1608,
        "x0": 0.3776,
        "x1": 3.8073,
        "C_bar": 6.2018,
        "delta0": 0.14,
    },  # lead_Pb
    "Na": {
        "a": 0.0777,
        "k": 3.6452,
        "x0": 0.288,
        "x1": 3.1962,
        "C_bar": 5.0526,
        "delta0": 0.08,
    },  # sodium_Na
    "Ca": {
        "a": 0.1564,
        "k": 3.0745,
        "x0": 0.3228,
        "x1": 3.1191,
        "C_bar": 5.0396,
        "delta0": 0.14,
    },  # calcium_Ca
    "Ga": {
        "a": 0.0944,
        "k": 3.1314,
        "x0": 0.2267,
        "x1": 3.5434,
        "C_bar": 4.9353,
        "delta0": 0.14,
    },  # gallium_Ga
    "As": {
        "a": 0.0663,
        "k": 3.4176,
        "x0": 0.1767,
        "x1": 3.5702,
        "C_bar": 5.051,
        "delta0": 0.08,
    },  # arsenic_As
    "Cd": {
        "a": 0.2461,
        "k": 2.6772,
        "x0": 0.1281,
        "x1": 3.1667,
        "C_bar": 5.2727,
        "delta0": 0.14,
    },  # cadmium_Cd
    "I": {
        "a": 0.2377,
        "k": 2.7276,
        "x0": 0.0549,
        "x1": 3.2596,
        "C_bar": 5.9488,
        "delta0": 0.0,
    },  # iodine_I
    "Cs": {
        "a": 0.1823,
        "k": 2.8866,
        "x0": 0.5473,
        "x1": 3.5914,
        "C_bar": 6.9135,
        "delta0": 0.14,
    },  # caesium_Cs
    "Ba": {
        "a": 0.1827,
        "k": 2.8906,
        "x0": 0.419,
        "x1": 3.4547,
        "C_bar": 6.3153,
        "delta0": 0.14,
    },  # barium_Ba
}

__all__ = ["STERNHEIMER_DENSITY_EFFECT", "TRANSPORT_ELEMENTS"]
