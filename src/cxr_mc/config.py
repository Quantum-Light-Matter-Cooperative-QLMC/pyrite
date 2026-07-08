"""
config.py
=============

Shared run configuration for the CXR pipeline, imported by BOTH notebooks so the
scan-runner (``scan.ipynb``) and the visualization driver
(``analysis.ipynb``) can never drift apart: they build the SAME
:class:`results.Settings` and the SAME per-material :class:`sweep.Sweep`,
so the viz notebook is guaranteed to be looking at the checkpoint the runner
wrote. Edit a material's grid (or the detector/analysis knobs) here ONCE and both
notebooks pick it up.

  * :func:`default_settings` -- beam current, electron counts, detector flags.
  * :func:`material_sweep`   -- the full parametric scan for a material (thickness,
    beam energies, polar/azimuthal tilt sweeps, the line/brem energy grids).
  * :func:`trajectory_sweep` -- a small, dedicated geometry sweep for the electron
    -penetration figures (a handful of polar tilts at normal azimuth, 2 energies).
  * :data:`COLLAPSE_AZIMUTH` -- keep only the best azimuth per (tilt, energy).
"""

from dataclasses import replace
from typing import NotRequired, TypedDict

import numpy as np

from .results import Settings
from .sweep import Layer, ScalarOrSeq, Sweep


class MaterialGrid(TypedDict):
    """The per-material scan grid: the geometry + energy fields that vary per
    material and are spread into :class:`sweep.Sweep`. Typing the grids with this
    (rather than ``dict[str, Any]``) lets pyright validate the literals in
    :data:`_MATERIAL_GRIDS` and the ``**grid`` spread in :func:`material_sweep`.

    A registry key can also name a full STACK (film + substrate-side layers):
    ``stack`` holds the :class:`sweep.Layer` list under the film, and
    :data:`_STACK_FILMS` maps the registry key (the CLI/checkpoint name, e.g.
    ``cxr scan mos2-on-sio2-si``) to the film crystal key."""

    thickness_ang: ScalarOrSeq
    energy_keV: ScalarOrSeq
    tilt_deg: ScalarOrSeq
    tilt_azim_deg: ScalarOrSeq
    E_grid_line: np.ndarray
    E_grid_brem: np.ndarray
    substrate: NotRequired[str]
    stack: NotRequired[tuple[Layer, ...]]


# named-stack registry keys -> the FILM crystal key (a CRYSTALS material). Keys
# absent here are their own film (the single-material default).
_STACK_FILMS = {
    "mos2-on-sio2-si": "mos2",
}


# When the azimuth is swept, collapse it: for each (polar tilt, energy) keep only
# the azimuth with the highest spectral peak. False -> show every azimuth.
COLLAPSE_AZIMUTH = True

# Product target: few-layer 2H-MoTe2 with c = 13.41 A (two layers per cell).
_MOTE2_PRODUCT_LAYER_PITCH_ANG = 13.41 / 2.0

# Few-layer 2H-MoS2: c = 12.294 A (crystal_structures.toml), two layers per cell.
_MOS2_LAYER_PITCH_ANG = 12.294 / 2.0


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


# ---- per-material parametric scan grids --------------------------------------
# Each entry is the geometry + energy grids for one material's full sweep. Keep
# the line grid fine + narrow (the expensive coherent lines top out at a few keV)
# and the brem grid coarse + WIDE (out to the beam energy) -- see sweep.
_MATERIAL_GRIDS: dict[str, MaterialGrid] = {
    # Thickness study: total flux + CXR/brem ratio vs thickness at a few key
    # tilts (negative = entrance-toward-detector = high flux). Single azimuth
    # (pitch plane) and single energy so plot_metric_vs(x="thickness_ang",
    # hue="tilt_deg") has nothing to silently collapse -- one clean curve per tilt.
    # For an energy comparison instead, add 40 to energy_keV and use hue="E0_keV".
    "hopg": {
        "thickness_ang": np.logspace(
            np.log10(0.1e4), np.log10(30e4), 40, endpoint=True
        ),  # 0.1-30 um
        "energy_keV": [30],
        "tilt_deg": np.linspace(-89.9, -5, 10, endpoint=True),
        "tilt_azim_deg": 0.0,
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 30.0),
    },
    "diamond": {
        "thickness_ang": 10e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89.9, 89.9, 60, endpoint=True),
        "tilt_azim_deg": np.linspace(-89.9, -0.1, 30, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 30.0),
    },
    "silicon": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-89, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),  # (was a stray 1-tuple in the nb)
        "E_grid_brem": np.arange(0.0, 60000.0, 30.0),
    },
    "mose2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-89, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 30.0),
    },
    "wse2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 30, endpoint=True),
        "tilt_azim_deg": np.linspace(-85, -0.1, 15),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
    "mote2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-85, 85, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-85, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
    "mote2_product": {
        "thickness_ang": _MOTE2_PRODUCT_LAYER_PITCH_ANG * np.arange(3, 7),
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-85, 85, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-85, -0.1, 15, endpoint=True),
        "substrate": "sapphire",
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
    "ptse2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-89, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
    "hfse2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-89, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 30.0),
    },
    "zrse2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 50, endpoint=True),
        "tilt_azim_deg": np.linspace(-89, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
    "ws2": {
        "thickness_ang": 1e4,
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-89, 89, 30, endpoint=True),
        "tilt_azim_deg": np.linspace(-85, -0.1, 15, endpoint=True),
        "E_grid_line": np.arange(50.0, 3500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
    "mos2": {
        "thickness_ang": 1e4,
        "energy_keV": [20, 30],
        "tilt_deg": np.linspace(-85, -0.1, 25, endpoint=True),
        "tilt_azim_deg": np.linspace(-85, -0.1, 10, endpoint=True),
        "substrate": "sapphire",
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
    },
    # Named device stack: few-layer 2H-MoS2 on a thin thermal a-SiO2 (285 nm,
    # the common device oxide -- adjust to the actual wafer) over thick
    # crystalline Si. Run as `cxr scan mos2-on-sio2-si`.
    "mos2-on-sio2-si": {
        "thickness_ang": _MOS2_LAYER_PITCH_ANG * np.arange(3, 7),
        "energy_keV": [30, 45, 60],
        "tilt_deg": np.linspace(-85, 85, 40, endpoint=True),
        "tilt_azim_deg": np.linspace(-85, -0.1, 15, endpoint=True),
        "stack": (Layer("sio2", 2850.0), Layer("silicon", 5e6)),
        "E_grid_line": np.arange(50.0, 4500.0, 1.0),
        "E_grid_brem": np.arange(0.0, 60000.0, 25.0),
    },
}

MATERIALS = tuple(_MATERIAL_GRIDS)


def material_grid(material) -> MaterialGrid:
    """The raw per-material grid dict (thickness, energies, tilt sweeps, grids)."""
    if material not in _MATERIAL_GRIDS:
        raise ValueError(f"unknown material {material!r} (have {list(_MATERIAL_GRIDS)})")
    return _MATERIAL_GRIDS[material]


def material_sweep(material: str, *, theta_obs_deg=90.0, **overrides):
    """The full parametric :class:`sweep.Sweep` for ``material`` (the geometry
    the runner scans and the viz notebook reduces). ``overrides`` replace any grid
    field, e.g. ``material_sweep("ptse2", thickness_ang=2e4)``. For a named-stack
    key (:data:`_STACK_FILMS`) the Sweep's material is the film crystal; the
    registry key stays the CLI/checkpoint name."""
    film = _STACK_FILMS.get(material, material)
    sweep = Sweep(material=film, theta_obs_deg=theta_obs_deg, **material_grid(material))
    return replace(sweep, **overrides) if overrides else sweep


def trajectory_sweep(material: str, *, n_tilts=9, energies=(30, 60), tilt_span=80.0):
    """A small dedicated geometry sweep for the electron-penetration figures: a
    handful of polar tilts at normal azimuth, two beam energies (transport only,
    so the energy grids are irrelevant -- kept for build_cases). ``n_tilts`` panels
    span +-``tilt_span`` degrees.

    Always uses ONE representative thickness: the geometric midpoint of the
    material's thickness array if it is a sweep (e.g. HOPG), or the scalar
    itself. This avoids inheriting a 40-element thickness loop that would (a)
    make penetration_survival_chart silently pick the thinnest slab and (b)
    make trajectory_chart plot nearly-invisible grazing tracks.

    Carries the material's ``substrate``/``stack`` through (when present) so a
    film-on-substrate material (e.g. mos2 on sapphire) gets its ``abs_layers``
    stack here too -- without this the penetration figures silently transported
    electrons through the free-standing film only, never reaching the
    substrate, even though the spectrum runner always sees the full stack."""
    p = material_grid(material)
    thick_arr = np.atleast_1d(np.asarray(p["thickness_ang"], dtype=float))
    thick = float(thick_arr[len(thick_arr) // 2])
    stack_kwargs = {}
    if "stack" in p:
        stack_kwargs["stack"] = p["stack"]
    elif "substrate" in p:
        stack_kwargs["substrate"] = p["substrate"]
    return Sweep(
        material=_STACK_FILMS.get(material, material),  # named stacks: the film
        thickness_ang=thick,
        energy_keV=list(energies),
        tilt_deg=np.linspace(-tilt_span, tilt_span, n_tilts, endpoint=True),
        tilt_azim_deg=0.0,
        theta_obs_deg=90.0,
        E_grid_line=p["E_grid_line"],
        E_grid_brem=p["E_grid_brem"],
        **stack_kwargs,
    )
