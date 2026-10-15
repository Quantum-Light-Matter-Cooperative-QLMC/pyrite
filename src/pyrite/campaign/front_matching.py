"""Scene geometry for the linear bunch-front phase-matching inverse.

Validation: linear-bunch-front-tilt, surface-hkl-orientation
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..detectors import Detector
from ..materials import CATALOG
from ..materials.crystal import HBARC_EV_ANG, beta_from_Ee, reciprocal_g_vector
from ..montecarlo.bunch_front import BunchFrontTilt
from ..montecarlo.geometry import beam_entry_jacobian, sample_to_lab_R
from .geometry import Slab, resolve_crystal_cut

if TYPE_CHECKING:
    from .model import Scene


def phase_matched_front_for_scene(
    scene: "Scene",
    hkl: tuple[int, int, int],
    photon_energy_eV: float | None,
    observation_direction_lab: Sequence[float] | None,
) -> BunchFrontTilt:
    """Solve K_eff=0 in the same sample frame used by the coherent reducer.

    Source: joint phase law in linear-bunch-front-tilt; vacuum line dispersion
    omega=beta*b.g/(1-beta*b.n). Assumes parallel monoenergetic rays and one
    flat slab's mean crystallite orientation, including any ``Slab.cut``
    resolved exactly as case lowering resolves it. Matching preserves residual
    timing damping; a zero required gradient returns BunchFrontTilt(0).
    Validation: linear-bunch-front-tilt, line-energy-dispersion, surface-hkl-orientation
    """
    target = scene.target_spec
    if not isinstance(target, Slab) or target.entrance_face is not None:
        raise ValueError("phase matching requires a flat single Slab")
    # Reuse the public source-composition checks even if this scene is untilted.
    replace(scene.beam, bunch_front_tilt=BunchFrontTilt(1.0))
    indices = np.asarray(hkl)
    if indices.shape != (3,) or indices.dtype.kind not in "iu" or not np.any(indices):
        raise ValueError("hkl must contain three integer Miller indices, not all zero")
    crystal_key = (
        CATALOG.material(target.material).crystal_key
        if target.material in CATALOG.materials
        else target.material
    )
    if crystal_key is None:
        raise ValueError("phase matching requires a crystalline target")
    crystal = CATALOG.crystal(crystal_key)
    g, _ = reciprocal_g_vector(indices, crystal.lattice)
    orientation = resolve_crystal_cut(target.cut, crystal.beam_uvw, crystal.surface_hkl).rotation(
        crystal.lattice
    )
    if orientation is not None:
        g = orientation @ g
    # Use the same resolved half-degree target angles as case lowering.
    tilts, azimuths = target._angles()
    tilt, azimuth = np.deg2rad([tilts.item(), azimuths.item()])
    rotation = sample_to_lab_R(tilt, azimuth)
    b = rotation.T @ np.array([0.0, 0.0, 1.0])
    if observation_direction_lab is None:
        if not isinstance(scene.detector, Detector):
            raise ValueError("a planar detector requires observation_direction_lab")
        theta = np.deg2rad(scene.detector.observation_angle_deg)
        direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    else:
        direction = np.asarray(observation_direction_lab, dtype=float)
    if (
        direction.shape != (3,)
        or not np.all(np.isfinite(direction))
        or not np.isfinite(np.linalg.norm(direction))
        or np.linalg.norm(direction) == 0.0
    ):
        raise ValueError("observation_direction_lab must be a finite nonzero 3-vector")
    n = rotation.T @ (direction / np.linalg.norm(direction))
    energy_keV = float(scene.beam.energy_keV)
    if not np.isfinite(energy_keV) or energy_keV <= 0.0:
        raise ValueError("phase matching requires finite positive beam energy")
    beta = float(beta_from_Ee(energy_keV * 1e3))
    if not np.isfinite(beta) or not 0.0 < beta < 1.0:
        raise ValueError("phase matching requires finite positive beam energy")
    omega = (
        beta * float(b @ g) / (1.0 - beta * float(b @ n))
        if photon_energy_eV is None
        else float(photon_energy_eV) / HBARC_EV_ANG
    )
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("photon energy must be finite and positive; select a resonant hkl")
    a = beam_entry_jacobian(tilt, azimuth).T @ (n[:2] - b[:2] / beta + g[:2] / omega)
    magnitude = float(np.linalg.norm(a))
    if not np.isfinite(magnitude):
        raise ValueError("phase-matched gradient is not finite")
    return BunchFrontTilt(
        tilt_deg=float(np.rad2deg(np.arctan(beta * magnitude))),
        azimuth_deg=0.0 if magnitude == 0.0 else float(np.rad2deg(np.arctan2(a[1], a[0]))),
    )
