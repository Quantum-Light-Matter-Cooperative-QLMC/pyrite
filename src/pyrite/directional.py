"""Detector-free escaped directional source maps (PXR/CBS, characteristic, continuum)."""

from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from ._cxr_grid_policy import kinematic_cxr_stop_eV
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
from .materials import CATALOG
from .materials.crystal import reciprocal_g_vector
from .montecarlo.runner import case_table_markers
from .montecarlo.runner.directional_source import SOURCE_COMPONENTS, source_outputs
from .montecarlo.runner.transport_options import kernel_schedule
from .results.directional import COMPONENT_IDS, DirectionalSpectrum

#: Public component names accepted by :func:`directional_sources`.
DIRECTIONAL_COMPONENTS = tuple(SOURCE_COMPONENTS)


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


def _lowered_source(beam, target, grid, numerics, seed, *, transition=False):
    """Validated detector-free case on ``grid``, plus its resolved numerics."""
    controls = Numerics() if numerics is None else numerics
    if not isinstance(controls, Numerics):
        raise TypeError("numerics must be a Numerics")
    if controls.precision is not None:
        raise ValueError("directional_source requires fixed counts; precision is unsupported")
    controls = replace(controls, brem_trials=controls.cxr_trials, energy_bins=EnergyBins(cxr=grid))
    physics = Physics(transition_radiation="history" if transition else "off")
    scene = Scene(beam, target, detector=None)
    slab = scene.target_spec
    if getattr(slab, "entrance_face", None) is not None:
        raise ValueError("directional_source supports flat slabs/stacks; grooves are unsupported")
    if transition and slab.footprint is not None:
        raise ValueError("directional TR requires an infinite planar slab (footprint=None)")
    if getattr(beam.longitudinal, "kind", "gaussian") not in {"point", "gaussian"}:
        raise ValueError(
            "directional_source does not support direction-targeted longitudinal modes"
        )
    return replace(build_case(scene, physics=physics, numerics=controls), seed=seed), controls


def source_line_ceiling_eV(
    beam: Beam, target: Slab | Stack | Target, *, numerics: Numerics | None = None
) -> float:
    """:func:`line_window_ceiling_eV` for a scene, without transport or scoring.

    Lets a caller choose an explicit grid that covers every scored line before
    calling :func:`directional_source` with the same beam, target and numerics.
    """
    case, _ = _lowered_source(beam, target, np.array([10.0, 20.0]), numerics, 1)
    return line_window_ceiling_eV(case)


def line_window_ceiling_eV(case) -> float:
    """Highest photon energy any scored PXR/CBS line can reach, in eV.

    The case's closed-form kinematic bound
    ``hbar c beta |g|_max / (1 - beta)`` over every radiating layer's
    reflections, capped by the incident kinetic energy. The atomic-data cap
    is omitted, so the value can only be high. It is direction independent, because
    scattered electrons may point anywhere. Relativistic beams Doppler-shift
    near-forward lines far above their side-on energies, so an explicit grid
    stopping below this value can leave empty bands that are window
    truncation, not emission nulls.

    Validation: line-grid-kinematic-bandwidth
    Validation: line-grid-emission-ceiling
    """
    radiators = case.get("layer_radiators") or [case]
    magnitudes = [
        reciprocal_g_vector(hkl, CATALOG.crystal(radiator["crystal"]).lattice)[1]
        for radiator in radiators
        if radiator is not None
        for hkl in radiator["hkl_list"]
    ]
    energy_keV = float(case["E0_keV"])
    return min(kinematic_cxr_stop_eV(magnitudes, energy_keV), energy_keV * 1e3)


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

    Experimental; canonical import: ``pyrite.radiation.directional_source``.

    Equivalent to ``directional_sources(..., components=("pxr-cbs",))["pxr-cbs"]``;
    see :func:`directional_sources` for the parameters.
    """
    return directional_sources(
        beam,
        target,
        energy_eV=energy_eV,
        angle_deg=angle_deg,
        plane_azimuth_deg=plane_azimuth_deg,
        numerics=numerics,
        seed=seed,
        direction_batch=direction_batch,
    )["pxr-cbs"]


def directional_sources(
    beam: Beam,
    target: Slab | Stack | Target,
    *,
    energy_eV: ArrayLike,
    angle_deg: ArrayLike | None = None,
    plane_azimuth_deg: float = 0.0,
    numerics: Numerics | None = None,
    seed: int = 1,
    direction_batch: int = 16,
    components: tuple[str, ...] = ("pxr-cbs",),
) -> dict[str, DirectionalSpectrum]:
    """Calculate detector-free escaped angular slices for selected components.

    Experimental; canonical import: ``pyrite.radiation.directional_sources``.

    One transport serves every component and direction. Components are kept
    separate (never summed): ``"pxr-cbs"`` (incoherent PXR/CBS with their
    interference), ``"characteristic"`` (isotropic atomic relaxation lines)
    and ``"bremsstrahlung"`` (continuum; BremsLib angular model when the
    tables resolve under the default ``Physics``, otherwise isotropic EEDL).
    Characteristic and continuum use the live runner's lower-cutoff
    population, which equals the line population here because
    ``brem_trials`` is replaced by ``cxr_trials``.
    ``"transition"`` adds history-coherent planar-stack TR, using that same
    population. It requires an infinite planar target and the existing X-ray
    dielectric band; finite lateral interfaces and optical response are not
    inferred. Components remain separate; no TR–line cross term is added.

    Parameters
    ----------
    beam, target
        Scalar public Beam and flat Slab, Stack or Target. Tilts retain full precision during lowering. Grooves are unsupported.
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
    components
        Subset of :data:`DIRECTIONAL_COMPONENTS`, scored in this order.

    Returns
    -------
    dict[str, DirectionalSpectrum]
        One result per requested component, sharing angles, grid, case and
        provenance. Each holds escaped density per incident electron/eV/sr and current-scaled rate.
        A finite footprint evaluates every direction, tracing escape to the
        first prism face (side exits included). Only the laterally infinite
        slab marks face-parallel directions (``|n_sample,z| <= 1e-10``)
        unsupported (NaN plus mask), never interpreted as zero emission.
        ``provenance["line_window_ceiling_eV"]`` reports the kinematic line
        ceiling; a grid stopping below it truncates relativistic lines.
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
    components = tuple(dict.fromkeys(components))
    if not components or not set(components) <= set(DIRECTIONAL_COMPONENTS):
        raise ValueError(f"components must be a nonempty subset of {DIRECTIONAL_COMPONENTS}")
    case, controls = _lowered_source(
        beam, target, grid, numerics, seed, transition="transition" in components
    )
    if grid[-1] > case["E0_keV"] * 1000:
        raise ValueError("energy_eV must not exceed the incident electron energy")
    sample = lab @ case_rotation(case)
    with kernel_schedule(controls):
        densities, supported = source_outputs(
            case,
            sample,
            components=components,
            direction_batch=direction_batch,
            transport_core=controls.transport_core,
        )
    tables = case_table_markers(case)
    provenance = {
        "plane_azimuth_deg": float(plane_azimuth_deg),
        "numerics": controls,
        "identity_digest": case_content_key(case, xsgen_tables=tables),
        "energy_grid": "explicit point samples; no automatic window",
        "line_window_ceiling_eV": line_window_ceiling_eV(case),
        **run_provenance(case, tables),
    }
    return {
        name: DirectionalSpectrum(
            angles,
            grid.copy(),
            lab,
            sample,
            densities[name],
            supported,
            case,
            provenance,
            component=COMPONENT_IDS[name],
        )
        for name in components
    }
