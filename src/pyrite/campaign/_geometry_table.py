"""Numeric case geometry summaries for configuration inspection."""

import numpy as np

from .._energy_grid_encoding import decode_energy_grid


def geometry_table(cases):
    """A one-row-per-config DataFrame summarizing the geometry of a case list,
    for a quick sanity check before running."""
    import pandas as pd

    def _grid(encoded):
        """Compact label for a legacy uniform triple or an exact grid array."""
        if encoded is None:
            return "-"
        values = decode_energy_grid(encoded)
        if isinstance(encoded, tuple):
            return f"{values[0] / 1e3:g}-{values[-1] / 1e3:g} keV @ {encoded[2]:g} eV"
        if values.size == 1:
            return f"{values[0] / 1e3:g} keV (1 value)"
        return f"{values[0] / 1e3:g}-{values[-1] / 1e3:g} keV ({values.size} values)"

    rows, seen = [], set()
    for c in cases:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        same = [k for k in cases if k["name"] == c["name"]]
        rows.append(
            {
                "config": c["name"],
                "beam_uvw": c["beam_uvw"],
                "surface_hkl": c.get("surface_hkl"),
                "refl": len(c["hkl_list"]),
                "t [um]": c["thickness_ang"] / 1e4,
                "width [mm]": c.get("crystal_width_mm"),
                "height [mm]": c.get("crystal_height_mm"),
                "polar [deg]": round(c["tilt_deg"], 2),
                "azim [deg]": round(c["tilt_azim_deg"], 2),
                "energies [keV]": [k["E0_keV"] for k in same],
                "line grid": _grid(c.get("E_grid_line", c["E_grid"])),
                "brem grid": _grid(c.get("E_grid_brem")),
                "theta_obs [deg]": round(np.degrees(c["theta_obs_rad"]), 1),
                "dOmega [sr]": c["domega_sr"],
            }
        )
    return pd.DataFrame(rows)
