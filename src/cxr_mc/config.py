"""
config.py
=============

Shared run configuration for the CXR pipeline, imported by BOTH marimo apps so the
scan runner (``notebooks/scan_app.py``) and visualization driver
(``notebooks/analysis_app.py``) can never drift apart: they build the SAME
:class:`results.Settings` and the SAME per-material :class:`sweep.Sweep`,
so the visualization app is guaranteed to be looking at the checkpoint the runner
wrote. Material identities and grids come from :data:`cxr_mc.materials.CATALOG`;
detector/analysis knobs still live here.

  * :func:`default_settings` -- beam current, electron counts, detector flags.
  * :func:`material_sweep`   -- the full parametric scan for a material (thickness,
    beam energies, polar/azimuthal tilt sweeps, the line/brem energy grids).
  * :func:`trajectory_sweep` -- a small, dedicated geometry sweep for the electron
    -penetration figures (a handful of polar tilts at normal azimuth, 2 energies).
  * :data:`COLLAPSE_AZIMUTH` -- keep only the best azimuth per (tilt, energy).
"""

from dataclasses import replace

import numpy as np

from .materials import CATALOG, MaterialSpec
from .montecarlo import simulate_trajectories
from .results import Settings
from .sweep import Sweep

MATERIALS = CATALOG.material_keys


# When the azimuth is swept, collapse it: for each (polar tilt, energy) keep only
# the azimuth with the highest spectral peak. False -> show every azimuth.
COLLAPSE_AZIMUTH = True

# Penetration-plot transport angles. Keep this intentionally sparse because each
# value triggers direct CPU trajectory MC in the analysis app.
PENETRATION_TILT_DEG = (0.0, 15.0, 45.0, 75.0)


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


def _material_spec(material: str) -> MaterialSpec:
    try:
        return CATALOG.material(material)
    except KeyError:
        raise ValueError(
            f"unknown material {material!r} (have {list(CATALOG.material_keys)})"
        ) from None


def material_grid(material: str) -> dict[str, object]:
    """Return a mapping-style projection of a catalog material's scan grid."""
    spec = _material_spec(material)
    scan = spec.scan
    grid: dict[str, object] = {
        "thickness_ang": scan.thickness_ang,
        "energy_keV": scan.energy_keV,
        "tilt_deg": scan.tilt_deg,
        "tilt_azim_deg": scan.tilt_azim_deg,
        "E_grid_line": scan.E_grid_line,
        "E_grid_line_by_energy": scan.E_grid_line_by_energy,
        "E_grid_brem": scan.E_grid_brem,
    }
    if spec.substrate is not None:
        grid["substrate"] = spec.substrate
    if spec.stack:
        grid["stack"] = spec.stack
    return grid


def material_sweep(material: str, *, theta_obs_deg=90.0, **overrides):
    """The full parametric :class:`sweep.Sweep` for ``material`` (the geometry
    the runner scans and the viz notebook reduces). ``overrides`` replace any grid
    field, e.g. ``material_sweep("ptse2", thickness_ang=2e4)``. For a named-stack
    key the Sweep's material is the film crystal; the catalog material key stays the
    CLI/checkpoint name."""
    spec = _material_spec(material)
    scan = spec.scan
    sweep = Sweep(
        material=spec.crystal_key,
        theta_obs_deg=theta_obs_deg,
        thickness_ang=scan.thickness_ang,
        energy_keV=scan.energy_keV,
        tilt_deg=scan.tilt_deg,
        tilt_azim_deg=scan.tilt_azim_deg,
        E_grid_line=scan.E_grid_line,
        E_grid_line_by_energy=scan.E_grid_line_by_energy,
        E_grid_brem=scan.E_grid_brem,
        substrate=spec.substrate,
        stack=spec.stack or None,
    )
    return replace(sweep, **overrides) if overrides else sweep


def trajectory_sweep(
    material: str,
    *,
    tilts=PENETRATION_TILT_DEG,
    energies=(30, 50),
    thickness_ang: float | None = None,
    n_tilts: int | None = None,
    tilt_span: float | None = None,
):
    """A small dedicated geometry sweep for the electron-penetration figures: a
    handful of polar tilts at normal azimuth, two beam energies (transport only,
    so the energy grids are irrelevant -- kept for build_cases). ``n_tilts`` panels
    span from normal incidence to ``tilt_span`` when supplied; otherwise the
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
    spec = _material_spec(material)
    scan = spec.scan
    if thickness_ang is None:
        thick_arr = np.atleast_1d(np.asarray(scan.thickness_ang, dtype=float))
        thick = float(thick_arr[len(thick_arr) // 2])
    else:
        thick = float(thickness_ang)
    stack_kwargs = {}
    if spec.stack:
        stack_kwargs["stack"] = spec.stack
    elif spec.substrate is not None:
        stack_kwargs["substrate"] = spec.substrate
    if n_tilts is not None or tilt_span is not None:
        count = 9 if n_tilts is None else int(n_tilts)
        span = 80.0 if tilt_span is None else float(tilt_span)
        tilt_values = np.linspace(0.0, span, count, endpoint=True)
    else:
        tilt_values = tuple(float(t) for t in tilts)
    return Sweep(
        material=spec.crystal_key,  # named stacks: the film
        thickness_ang=thick,
        energy_keV=list(energies),
        tilt_deg=tilt_values,
        tilt_azim_deg=0.0,
        theta_obs_deg=90.0,
        E_grid_line=scan.E_grid_line,
        E_grid_line_by_energy=scan.E_grid_line_by_energy,
        E_grid_brem=scan.E_grid_brem,
        **stack_kwargs,
    )


# Below this fraction of the incident electron population is still exiting
# the far face, a thicker slab in the same (material, beam energy) sweep is
# statistically indistinguishable from "the beam is dead" -- see
# gate_cases_by_penetration.
PENETRATION_SURVIVAL_FLOOR = 0.05
PENETRATION_WATCHDOG_NE = 500  # electrons per pre-run transmission check


def gate_cases_by_penetration(
    cases,
    *,
    floor: float = PENETRATION_SURVIVAL_FLOOR,
    Ne: int = PENETRATION_WATCHDOG_NE,
    seed: int = 0,
):
    """Drop thickness values a beam energy has already died in, before the
    (expensive) spectral Monte Carlo ever runs them.

    For each beam energy present in ``cases``, walks that energy's distinct
    ``thickness_ang`` values ascending and runs one direct-CPU transmission
    check per thickness with :func:`cxr_mc.montecarlo.simulate_trajectories`
    -- the SAME prebuilt penetration-depth transport behind
    :func:`cxr_mc.plots.plot_penetration_survival` -- at normal incidence
    (``beam_dir`` left at its ``simulate_trajectories`` default, +z), using
    the (energy, thickness) group's normal-incidence case for
    composition/stack, and ``Ne`` electrons. The transmitted fraction
    (``n_transmitted / Ne``) is a direct Monte-Carlo estimate of the
    fraction of the original beam still exiting the far face of that
    thickness of crystal -- the average remaining electron population at
    the end of that slab.

    Normal incidence is the reference geometry because it MAXIMIZES
    transmission at fixed nominal thickness: any nonzero tilt lengthens the
    in-material path length needed to reach a given depth below the entry
    surface by ~1/cos(tilt), so the beam sees more scattering and stopping
    power per unit depth at any tilt > 0 than at normal incidence. If the
    beam is already dead at normal incidence for a given thickness, it is
    at least as dead at every larger tilt in the sweep -- so checking only
    tilt=0 is a safe, cheap proxy for the whole tilt grid (one trajectory
    MC per (energy, thickness) instead of one per (energy, thickness, tilt,
    azimuth)).

    The first thickness whose transmitted fraction drops below ``floor`` is
    KEPT (so the "beam is basically dead" case is still represented in the
    checkpoint) but every LARGER thickness for that same energy is dropped
    WITHOUT running the check -- transmission is monotonically
    non-increasing with thickness at fixed energy, so a thicker slab can
    only be equally or more dead.

    Returns ``(kept_cases, dropped_cases)``, both preserving the input's
    relative order. ``dropped_cases`` is empty when no beam energy's
    transmitted fraction crosses ``floor`` anywhere in the sweep's
    thickness grid.
    """
    reference_case: dict[tuple[float, float], dict] = {}
    thicknesses_by_energy: dict[float, set] = {}
    for case in cases:
        energy = case["E0_keV"]
        thickness = case["thickness_ang"]
        thicknesses_by_energy.setdefault(energy, set()).add(thickness)
        key = (energy, thickness)
        current = reference_case.get(key)
        if current is None or abs(case["tilt_deg"]) < abs(current["tilt_deg"]):
            reference_case[key] = case

    cutoff_by_energy: dict[float, float | None] = {}
    for energy, thicknesses in thicknesses_by_energy.items():
        cutoff = None
        for thickness in sorted(thicknesses):
            ref = reference_case[(energy, thickness)]
            abs_layers = ref.get("abs_layers")
            total_thickness = float(abs_layers[-1][1]) if abs_layers is not None else thickness
            segs = simulate_trajectories(
                energy,
                Ne,
                total_thickness,
                composition=ref["composition"],
                layers=abs_layers,
                seed=seed,
            )
            fraction = float(segs["n_transmitted"]) / Ne
            if fraction < floor:
                cutoff = thickness
                break
        cutoff_by_energy[energy] = cutoff

    kept, dropped = [], []
    for case in cases:
        cutoff = cutoff_by_energy.get(case["E0_keV"])
        if cutoff is not None and case["thickness_ang"] > cutoff:
            dropped.append(case)
        else:
            kept.append(case)
    return kept, dropped


def format_penetration_watchdog_summary(
    dropped,
    *,
    floor: float = PENETRATION_SURVIVAL_FLOOR,
    material: str | None = None,
) -> str | None:
    """Format the ``gate_cases_by_penetration`` drop summary printed by both
    ``cxr_mc.scan`` (CLI) and ``notebooks/scan_app.py`` (interactive) --
    shared here so the two call sites can't drift on how they compute
    ``dead_energies`` from ``dropped``.

    Returns ``None`` when ``dropped`` is empty (nothing to report). When
    ``material`` is given, prefixes the message with ``"{material}: "`` and
    appends the floor-percentage explanation -- the CLI shape, where the
    material isn't otherwise obvious from context. When ``material`` is
    ``None``, omits both -- the notebook shape, where the material is
    already visible in the UI.
    """
    if not dropped:
        return None
    dead_energies = sorted({c["E0_keV"] for c in dropped})
    energies_str = ", ".join(f"{e:g} keV" for e in dead_energies)
    if material is not None:
        return (
            f"{material}: penetration watchdog dropped {len(dropped)} case(s) "
            f"at {len(dead_energies)} beam energy(ies) "
            f"({energies_str}) -- "
            f"electron population already below {100 * floor:g}% "
            f"before those thicknesses"
        )
    return (
        f"penetration watchdog dropped {len(dropped)} case(s) at "
        f"{len(dead_energies)} beam energy(ies) "
        f"({energies_str})"
    )
