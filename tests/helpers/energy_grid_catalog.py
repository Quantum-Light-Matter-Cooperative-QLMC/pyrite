"""Minimal single-material catalog stub shared by energy-grid apply/add tests."""

BASE_TOML = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0, 100.0] }

[energy_grids.hopg]
line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2600.0, num = 864, endpoint = true } }, source = "derived" },
  { energy_keV = 100.0, grid = { linspace = { start = 50.0, stop = 4600.0, num = 1518, endpoint = true } }, source = "derived" },
]

[profiles.standard.overrides.hopg]
E_grid_brem = { arange = { start = 0.0, stop = 136500.0, step = 25.0 } }

[materials.hopg]
display_name = "HOPG"
"""

COMBINED = {
    "hopg": {
        "line_rows": [
            {"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 2700.0, "num": 897},
            {"energy_keV": 100.0, "start_eV": 50.0, "stop_eV": 4600.0, "num": 1518},
        ],
        "brem": {"stop_eV": 140000.0, "step_eV": 25.0, "raw_eV": 133000.0},
    }
}
