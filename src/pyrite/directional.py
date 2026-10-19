"""Detector-free escaped directional PXR/CBS source maps."""

from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from ._energy_bins import EnergyBins
from ._numerics import Numerics, Physics
from ._scene_geometry import case_rotation
from .api import run_provenance
from .campaign.beam import Beam
from .campaign.geometry import Slab, Stack
from .campaign.lowering import build_case
from .campaign.model import Scene
from .campaign.profiles import case_content_key
from .campaign.target import Target
from .montecarlo.runner import case_table_markers
from .montecarlo.runner.directional_source import source_outputs
from .results.directional import DirectionalSpectrum


def observation_directions(angle_deg: ArrayLike, plane_azimuth_deg: float = 0.0) -> np.ndarray:
    """Lab directions sin(theta)*(cos(phi),sin(phi),0)+cos(theta)*(0,0,1).

    Zero follows the nominal incoming +z beam; +90 points toward the plane's
    transverse basis (+x by default), ±180 is backward. This is a signed plane
    slice, with no azimuth or solid-angle integration.
    """
    angles = np.asarray(angle_deg, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.all(np.isfinite(angles)):
        raise ValueError("angle_deg must be a finite nonempty one-dimensional array")
    if np.any(np.abs(angles) > 180) or not np.isfinite(plane_azimuth_deg):
        raise ValueError("angles must lie in [-180, 180] and plane azimuth must be finite")
    theta, phi = np.deg2rad(angles), np.deg2rad(plane_azimuth_deg)
    return np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )


def directional_source(
    beam: Beam,
    target: Slab | Stack | Target,
    *,
    energy_eV: ArrayLike,
    angle_deg: ArrayLike | None = None,
    plane_azimuth_deg: float = 0.0,
    numerics: Numerics | None = None,
    seed: int = 1,
    direction_batch: int = 16,
) -> DirectionalSpectrum:
    """Calculate a detector-free incoherent PXR/CBS angular slice.

    Parameters
    ----------
    beam, target
        Scalar public Beam and flat Slab, Stack or Target. Tilts use the shared
        nearest-half-degree lowering convention. Grooves are unsupported.
    energy_eV
        Explicit strictly increasing point grid starting at or above 10 eV, shared at every angle;
        at least two points and no photon energy above the incident electron.
        No narrow automatic directional line window is substituted. Caller owns
        bounds/spacing and must refine both energy and angular grids.
    angle_deg
        Ordered point samples in [-180,180]; default 360 samples [-180,180).
        A supplied pair of periodic endpoints is rejected to prevent duplication.
    plane_azimuth_deg
        Lab transverse plane basis azimuth, degrees; zero gives the x–z plane.
    numerics
        Fixed electron count, reflection/mosaic and transport controls. Adaptive
        precision and automatic energy grids are unsupported. brem_trials is
        replaced by cxr_trials; no extra population or excluded channel is scored.
    seed, direction_batch
        Nonnegative integer RNG seed and positive bounded scoring batch size.

    Returns
    -------
    DirectionalSpectrum
        Escaped density per incident electron/eV/sr and current-scaled rate.
        Sample-face grazing directions (|n_sample,z| <= 1e-10) are explicitly
        unsupported (NaN plus mask), never interpreted as zero emission.
    """
    grid = np.asarray(energy_eV, dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.all(np.isfinite(grid))
        or grid[0] < 10
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(
            "energy_eV must contain at least two positive increasing finite points, starting at 10 eV or higher"
        )
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(direction_batch) is not int or direction_batch <= 0:
        raise ValueError("direction_batch must be a positive integer")
    angles = np.asarray(
        np.linspace(-180, 180, 360, endpoint=False) if angle_deg is None else angle_deg, dtype=float
    )
    lab = observation_directions(angles, plane_azimuth_deg)
    if np.any(np.diff(angles) <= 0) or (len(angles) > 1 and angles[-1] - angles[0] >= 360):
        raise ValueError("angle_deg must increase without duplicated periodic endpoints")
    controls = Numerics() if numerics is None else numerics
    if not isinstance(controls, Numerics):
        raise TypeError("numerics must be a Numerics")
    if controls.precision is not None:
        raise ValueError("directional_source requires fixed counts; precision is unsupported")
    controls = replace(controls, brem_trials=controls.cxr_trials, energy_bins=EnergyBins(cxr=grid))
    scene = Scene(beam, target, detector=None, physics=Physics(), numerics=controls)
    slab = scene.target_spec
    if getattr(slab, "entrance_face", None) is not None:
        raise ValueError("directional_source supports flat slabs/stacks; grooves are unsupported")
    if getattr(beam.longitudinal, "kind", "gaussian") not in {"point", "gaussian"}:
        raise ValueError(
            "directional_source does not support direction-targeted longitudinal modes"
        )
    case = replace(build_case(scene), seed=seed)
    if grid[-1] > case["E0_keV"] * 1000:
        raise ValueError("energy_eV must not exceed the incident electron energy")
    sample = lab @ case_rotation(case)
    density, supported = source_outputs(
        case, sample, direction_batch=direction_batch, transport_core=controls.transport_core
    )
    tables = case_table_markers(case)
    return DirectionalSpectrum(
        angles,
        grid.copy(),
        lab,
        sample,
        density,
        supported,
        case,
        {
            "plane_azimuth_deg": float(plane_azimuth_deg),
            "numerics": controls,
            "identity_digest": case_content_key(case, xsgen_tables=tables),
            "energy_grid": "explicit point samples; no automatic window",
            **run_provenance(case, tables),
        },
    )
