"""Inventory real channeling prerequisites without substituting missing data (#525).

Run with ``PYRITE_MC_BACKEND=cpu uv run python checks/channeling_material_survey.py``.
Print JSON for every bundled crystal and material. This is an availability
inventory, not a physics-validation verdict or a simulation support guarantee.
No transport, frozen-phonon ensemble, or synthetic thermal rows are used.
"""

from __future__ import annotations

import json

import numpy as np

from pyrite.materials.catalog import load_material_catalog
from pyrite.materials.channeling import (
    ChannelingDataError,
    load_thermal_sites,
    longitudinal_kinematics,
    planar_profile,
    transverse_states,
)
from pyrite.materials.channeling.entry import plane_incidence
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators


def inventory() -> dict:
    """Probe profiles and shell construction independently for each crystal."""
    catalog = load_material_catalog()
    thermal = load_thermal_sites()
    gamma, _ = longitudinal_kinematics(3e6)
    rows = []
    for key, crystal in sorted(catalog.crystals.items()):
        elements = sorted({element for element, _ in crystal.basis})
        planes = sorted(
            {(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), crystal.surface_hkl}
        )
        profiles = []
        for plane in planes:
            # Entry geometry is independent of missing thermal data. Check it
            # even when the profile itself will be refused.
            incidence = plane_incidence(key, plane, 3e6)
            result = {
                "plane_hkl": list(plane),
                "face_normal_incidence_deg": float(np.rad2deg(incidence.geometric_rad)),
            }
            try:
                profile = planar_profile(key, plane, 293.15)
            except ChannelingDataError as exc:
                result["error"] = str(exc)
            else:
                lo, hi = profile.extrema()
                result.update(
                    period_ang=profile.period_ang,
                    well_depth_eV=hi - lo,
                    bound_states_3MeV=transverse_states(profile, gamma).n_bound,
                )
            profiles.append(result)
        try:
            shells = catalog_shell_oscillators(key)
        except ValueError as exc:
            electronic = {"error": str(exc)}
        else:
            electronic = {
                "source": shells.conduction_source,
                "oscillators": len(shells.oscillators),
                "electrons_per_formula": shells.electrons_per_formula,
            }
        rows.append(
            {
                "crystal": key,
                "formula": crystal.formula,
                "structure_validation_id": crystal.validation_id,
                "cod_id": crystal.cod_id,
                "surface_hkl": list(crystal.surface_hkl),
                "mosaic_fwhm_deg": crystal.mosaic_fwhm_deg,
                "missing_thermal_elements": [el for el in elements if (key, el) not in thermal],
                "electronic": electronic,
                "profiles": profiles,
            }
        )
    return {
        "temperature_K": 293.15,
        "kinetic_energy_eV": 3e6,
        "crystals": rows,
        "materials": {
            key: {
                "crystal_key": material.crystal_key,
                "stack": bool(material.stack) or material.substrate is not None,
            }
            for key, material in sorted(catalog.materials.items())
        },
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, allow_nan=False))
