"""Shared case and aperture definitions for the #341 photon full-track check.

Electron pencil beams at normal incidence on a 5 cm x 5 cm slab at least as
thick as the electron CSDA range, so bremsstrahlung leaves through attenuating
material. Both engines report photons on the same two ideal square
apertures, in PyRITE's sample frame (Angstrom; slab from z = 0 to z = T,
beam along +z):

- ``T``: 10 cm x 10 cm, centred 10 cm behind the exit face, facing +z;
- ``R``: 10 cm x 10 cm, centred 10 cm in front of the entrance face, facing -z.

Geant4 TestEm5 places the absorber at ``-T/2 <= x <= T/2`` with the beam
along +x, so ``z = x + T/2`` and its (y, z) map to PyRITE's (x, y).
"""

import numpy as np

MM = 1.0e7  # Angstrom per mm
LATERAL_MM = 50.0
APERTURE_MM = 100.0
DISTANCE_MM = 100.0

#: case -> (element, TestEm5 material, thickness mm, energy MeV)
CASES = {
    "si_0p3mev": ("Si", "Silicon", 0.5, 0.3),
    "si_1mev": ("Si", "Silicon", 2.5, 1.0),
    "si_3mev": ("Si", "Silicon", 8.0, 3.0),
    "si_10mev": ("Si", "Silicon", 25.0, 10.0),
    "w_0p3mev": ("W", "Tungsten", 0.1, 0.3),
    "w_1mev": ("W", "Tungsten", 0.5, 1.0),
    "w_3mev": ("W", "Tungsten", 1.5, 3.0),
    "w_10mev": ("W", "Tungsten", 4.0, 10.0),
}
#: element -> (density g/cm^3, atomic weight); Si uses the catalog density in PyRITE
DENSITY = {"Si": (2.33, 28.09), "W": (19.3, 183.84)}


def edges_eV(energy_MeV: float) -> np.ndarray:
    """0.5 keV bins from 1 to 100 keV (resolve W K/L lines), then 60 log bins."""
    fine = np.arange(1.0e3, 100.0e3 + 1.0, 500.0)
    top = energy_MeV * 1.0e6
    if top <= fine[-1]:
        return fine[fine <= top]
    return np.concatenate([fine, np.geomspace(fine[-1], top, 61)[1:]])


def apertures(thickness_mm: float) -> dict:
    """Aperture centre, in-plane axes and half size in PyRITE's frame (Angstrom)."""
    half = 0.5 * APERTURE_MM * MM
    z_t = (thickness_mm + DISTANCE_MM) * MM
    return {
        "T": dict(center=(0.0, 0.0, z_t), x=(1.0, 0.0, 0.0), y=(0.0, 1.0, 0.0), half=half),
        "R": dict(
            center=(0.0, 0.0, -DISTANCE_MM * MM), x=(1.0, 0.0, 0.0), y=(0.0, -1.0, 0.0), half=half
        ),
    }


def hits(aperture: dict, position, direction) -> np.ndarray:
    """Ordinary straight-line hits on one aperture (vacuum outside the slab)."""
    c = np.asarray(aperture["center"])
    ex, ey = np.asarray(aperture["x"]), np.asarray(aperture["y"])
    normal = np.cross(ex, ey)
    position, direction = np.asarray(position, float), np.asarray(direction, float)
    cos = direction @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((c - position) @ normal) / cos
    point = position + t[:, None] * direction
    u, v = (point - c) @ ex, (point - c) @ ey
    half = aperture["half"]
    return (cos > 0) & (t > 0) & (np.abs(u) <= half) & (np.abs(v) <= half)
