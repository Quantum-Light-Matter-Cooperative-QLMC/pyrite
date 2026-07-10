"""
config.py
=============

Shared run configuration for the CXR pipeline, imported by BOTH notebooks so the
scan-runner (``scan.ipynb``) and the visualization driver
(``analysis.ipynb``) can never drift apart: they build the SAME
:class:`results.Settings` and the SAME per-material :class:`sweep.Sweep`,
so the viz notebook is guaranteed to be looking at the checkpoint the runner
wrote. Edit a material's grid in :mod:`cxr_mc.materials` and both notebooks pick
it up; detector/analysis knobs still live here.

  * :func:`default_settings` -- beam current, electron counts, detector flags.
  * :func:`material_sweep`   -- the full parametric scan for a material (thickness,
    beam energies, polar/azimuthal tilt sweeps, the line/brem energy grids).
  * :func:`trajectory_sweep` -- a small, dedicated geometry sweep for the electron
    -penetration figures (a handful of polar tilts at normal azimuth, 2 energies).
  * :data:`COLLAPSE_AZIMUTH` -- keep only the best azimuth per (tilt, energy).
"""

from dataclasses import replace

import numpy as np

from . import materials as _materials
from .materials import MaterialGrid, material_crystal_key
from .results import Settings
from .sweep import Sweep

_MATERIAL_GRIDS = _materials.MATERIAL_GRIDS
MATERIALS = _materials.MATERIALS


# When the azimuth is swept, collapse it: for each (polar tilt, energy) keep only
# the azimuth with the highest spectral peak. False -> show every azimuth.
COLLAPSE_AZIMUTH = True

# Penetration-plot transport angles. Keep this intentionally sparse because each
# value triggers direct CPU trajectory MC in the analysis app.
PENETRATION_TILT_DEG = (0.0, -15.0, -45.0, -75.0)


def default_settings():
    """The analysis / detector / unit knobs shared by the runner and the plots.
    The runner uses n_electrons*; every notebook that plots must use the SAME
    detector flags (apply_detector_qe / brem_source) so the displayed spectra
    match what was simulated."""
    return Settings(
        beam_current_na=5.0,
        n_electrons=300,  # transport electrons per line spectrum
        n_electrons_brem=150,  # transport electrons per background
        # OFF: the intrinsic spectra stay intrinsic (no legacy SDD polymer-window
        # QE). The Timepix3 / Eagle XO views apply their own QE downstream.
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",  # "mc" | "external" | "none"
    )


def material_grid(material) -> MaterialGrid:
    """The raw per-material grid dict (thickness, energies, tilt sweeps, grids)."""
    if material not in _MATERIAL_GRIDS:
        raise ValueError(f"unknown material {material!r} (have {list(_MATERIAL_GRIDS)})")
    return _MATERIAL_GRIDS[material]


def material_sweep(material: str, *, theta_obs_deg=90.0, **overrides):
    """The full parametric :class:`sweep.Sweep` for ``material`` (the geometry
    the runner scans and the viz notebook reduces). ``overrides`` replace any grid
    field, e.g. ``material_sweep("ptse2", thickness_ang=2e4)``. For a named-stack
    key the Sweep's material is the film crystal; the registry key stays the
    CLI/checkpoint name."""
    film = material_crystal_key(material)
    sweep = Sweep(material=film, theta_obs_deg=theta_obs_deg, **material_grid(material))
    return replace(sweep, **overrides) if overrides else sweep


def trajectory_sweep(
    material: str,
    *,
    tilts=PENETRATION_TILT_DEG,
    energies=(30, 60),
    thickness_ang: float | None = None,
    n_tilts: int | None = None,
    tilt_span: float | None = None,
):
    """A small dedicated geometry sweep for the electron-penetration figures: a
    handful of polar tilts at normal azimuth, two beam energies (transport only,
    so the energy grids are irrelevant -- kept for build_cases). ``n_tilts`` panels
    span from ``-tilt_span`` to normal incidence when supplied; otherwise the
    sparse default :data:`PENETRATION_TILT_DEG` set is used.

    Always uses ONE thickness: ``thickness_ang`` when explicitly supplied,
    otherwise the geometric midpoint of the material's thickness array if it is a
    sweep (e.g. HOPG), or the scalar itself. This avoids inheriting a 40-element
    thickness loop that would (a) make penetration_survival_chart silently pick
    the thinnest slab and (b) make trajectory_chart plot nearly-invisible
    grazing tracks.

    Carries the material's ``substrate``/``stack`` through (when present) so a
    film-on-substrate material (e.g. mos2 on sapphire) gets its ``abs_layers``
    stack here too -- without this the penetration figures silently transported
    electrons through the free-standing film only, never reaching the
    substrate, even though the spectrum runner always sees the full stack."""
    p = material_grid(material)
    if thickness_ang is None:
        thick_arr = np.atleast_1d(np.asarray(p["thickness_ang"], dtype=float))
        thick = float(thick_arr[len(thick_arr) // 2])
    else:
        thick = float(thickness_ang)
    stack_kwargs = {}
    if "stack" in p:
        stack_kwargs["stack"] = p["stack"]
    elif "substrate" in p:
        stack_kwargs["substrate"] = p["substrate"]
    if n_tilts is not None or tilt_span is not None:
        count = 9 if n_tilts is None else int(n_tilts)
        span = 80.0 if tilt_span is None else float(tilt_span)
        tilt_values = np.linspace(-span, 0.0, count, endpoint=True)
    else:
        tilt_values = tuple(float(t) for t in tilts)
    return Sweep(
        material=material_crystal_key(material),  # named stacks: the film
        thickness_ang=thick,
        energy_keV=list(energies),
        tilt_deg=tilt_values,
        tilt_azim_deg=0.0,
        theta_obs_deg=90.0,
        E_grid_line=p["E_grid_line"],
        E_grid_brem=p["E_grid_brem"],
        **stack_kwargs,
    )
