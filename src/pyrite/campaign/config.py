"""
config.py
=============

Shared run configuration for the CXR pipeline, imported by BOTH marimo apps so the
scan runner (``src/pyrite/apps/scan_app.py``) and visualization driver
(``src/pyrite/apps/analysis_app.py``) can never drift apart: they build the SAME
:class:`results.Settings` and the SAME per-material :class:`sweep.Sweep`,
so the visualization app is guaranteed to be looking at the checkpoint the runner
wrote. Material identities and grids come from :data:`pyrite.materials.CATALOG`;
detector/analysis knobs still live here.

  * :func:`default_settings` -- beam current, electron counts, detector flags.
  * :func:`material_sweep`   -- the full parametric scan for a material (thickness,
    beam energies, polar/azimuthal tilt sweeps, the line/brem energy grids).
  * :func:`trajectory_sweep` -- a small, dedicated geometry sweep for the electron
    -penetration figures (a handful of polar tilts at a configurable azimuth,
    normal incidence by default, 2 energies).
  * :data:`COLLAPSE_AZIMUTH` -- keep only the best azimuth per (tilt, energy).
"""

import dataclasses
import warnings
from dataclasses import replace
from typing import Any

import numpy as np

from .._numerics import CONVERGENCE_KEYS
from ..detectors import Detector, EnergyBins, Timepix3
from ..materials import CATALOG, MaterialSpec, load_material_catalog
from ..montecarlo import simulate_trajectories
from ..montecarlo.transverse import TransverseDistribution
from ..results import Settings
from .longitudinal import LongitudinalDistribution
from .profiles import get_fidelity_preset, resolve_numerics
from .sweep import BeamSpec, Sweep, beam_replace, target_from_flat, target_replace

# Override keys that address the beam (BeamSpec) rather than the Sweep itself,
# so ``material_sweep(..., energy_keV=[30, 60])`` and the legacy scalar-spot
# ``beam_fwhm_mm=`` keep working after the beam moved onto ``Sweep.beam``.
_BEAM_OVERRIDE_KEYS = frozenset(
    {f.name for f in dataclasses.fields(BeamSpec)} | {"beam_fwhm_mm", "transverse_fwhm_mm"}
)

# Override keys that address the target geometry rather than the Sweep itself.
# They are InitVars on ``Sweep`` -- ``dataclasses.replace`` would drop them
# silently -- so they rebuild the target the same way beam keys rebuild the beam.
_TARGET_OVERRIDE_KEYS = frozenset(
    {
        "thickness_ang",
        "tilt_deg",
        "tilt_azim_deg",
        "crystal_width_mm",
        "crystal_height_mm",
        "groove_spacing_ang",
        "substrate",
        "substrate_thickness_ang",
        "stack",
        "allow_normal_incidence",
        "mosaic",
    }
)

MATERIALS = CATALOG.material_keys


# When the azimuth is swept, collapse it: for each (polar tilt, energy) keep only
# the azimuth with the highest spectral peak. False -> show every azimuth.
COLLAPSE_AZIMUTH = True

# Penetration-plot transport angles. Keep this intentionally sparse because each
# value triggers direct CPU trajectory MC in the analysis app.
PENETRATION_TILT_DEG = (0.0, 15.0, 45.0, 75.0)


def default_settings(fidelity: str = "full"):
    """The analysis / detector / unit knobs shared by the runner and the plots.
    The runner uses n_electrons*; every notebook that plots must use the SAME
    detector flags (apply_detector_qe / brem_source) so the displayed spectra
    match what was simulated."""
    settings = Settings(
        beam_current_na=5.0,
        n_electrons=300,  # transport electrons per line spectrum
        n_electrons_brem=150,  # transport electrons per background
        # OFF: source spectra stay response-free (no legacy SDD polymer-window
        # QE). The Timepix3 / Eagle XO views apply their own QE downstream.
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",  # "mc" | "external" | "none"
    )
    return get_fidelity_preset(fidelity).apply_settings(settings)


#: The detector every profile starts from: a Timepix3 response at 90 deg, per
#: issue #52. ``materials.catalog`` validates the acceptance fields of a
#: ``[profiles.NAME.detector]`` block but does not build the detector -- it sits
#: below ``detectors`` in the package graph -- so the default response and the
#: construction both live here, on the driver side.
DEFAULT_CATALOG_DETECTOR = Detector(response=Timepix3())


def catalog_detector(catalog_profile: str = "standard") -> Detector:
    """Build the detector a catalog profile resolves to.

    ``Catalog.profile_detector`` returns validated acceptance fields; an empty
    mapping means every default stands.
    """
    spec = _catalog(catalog_profile).profile_detector(catalog_profile)
    return replace(DEFAULT_CATALOG_DETECTOR, **dict(spec))


def _catalog(catalog_profile: str = "standard"):
    """The bundled singleton for ``standard`` (avoids a re-parse), else the
    profile-resolved catalog. Shared by :func:`_material_spec` and the beam-block
    resolution in :func:`material_sweep`; both hit the same cached instance."""
    return (
        CATALOG if catalog_profile == "standard" else load_material_catalog(profile=catalog_profile)
    )


def _material_spec(material: str, catalog_profile: str = "standard") -> MaterialSpec:
    catalog = _catalog(catalog_profile)
    try:
        return catalog.material(material)
    except KeyError:
        raise ValueError(
            f"unknown material {material!r} (have {list(catalog.material_keys)})"
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


def material_sweep(
    material: str,
    *,
    fidelity="full",
    theta_obs_deg: Any = None,
    detector: Any = None,
    catalog_profile="standard",
    profile=None,
    **overrides: Any,
):
    """The full parametric :class:`sweep.Sweep` for ``material`` (the geometry
    the runner scans and the viz notebook reduces). ``overrides`` replace any grid
    field, e.g. ``material_sweep("ptse2", thickness_ang=2e4)``. For a named-stack
    key the Sweep's material is the film crystal; the catalog material key stays the
    CLI/checkpoint name.

    ``profile`` is a campaign alias for ``catalog_profile`` (fidelity now has its
    own ``fidelity=`` keyword; it no longer squats on ``profile=``)."""
    if profile is not None:
        catalog_profile = profile
    spec = _material_spec(material, catalog_profile=catalog_profile)
    scan = spec.scan
    # Energy is the per-material scan grid; the profile's optional
    # ``[profiles.*.beam]`` block supplies only the distribution fields
    # (transverse size, bunch, rep-rate/charge). ``standard`` has none -> the
    # BeamSpec defaults, bit-for-bit legacy.
    beam = BeamSpec(energy_keV=scan.energy_keV)
    beam_fields = _catalog(catalog_profile).profile_beam(catalog_profile)
    if beam_fields:
        beam_changes = dict(beam_fields)
        longitudinal = beam_changes.get("longitudinal")
        if longitudinal is not None:
            beam_changes["longitudinal"] = LongitudinalDistribution(**dict(longitudinal))
        transverse = beam_changes.get("transverse")
        if transverse is not None:
            beam_changes["transverse"] = TransverseDistribution(**dict(transverse))
            # The spot FWHMs default to 1 mm and are mutually exclusive with a
            # Courant-Snyder policy, so a profile that supplies one must clear
            # the other -- otherwise every such profile would fail build_cases.
            beam_changes.setdefault("transverse_fwhm_x_mm", None)
            beam_changes.setdefault("transverse_fwhm_y_mm", None)
        beam = beam_replace(beam, **beam_changes)
    resolved_catalog_detector = catalog_detector(catalog_profile)
    legacy_detector = {
        "observation_angle_deg": theta_obs_deg,
        "polar_acceptance_deg": overrides.pop("dtheta_obs_deg", None),
        "solid_angle_sr": overrides.pop("domega_sr", None),
    }
    supplied_legacy = {key: value for key, value in legacy_detector.items() if value is not None}
    if detector is not None and supplied_legacy:
        conflicts = [
            key for key, value in supplied_legacy.items() if getattr(detector, key) != value
        ]
        if conflicts:
            raise ValueError(
                "conflicting detector= and legacy detector override(s): " + ", ".join(conflicts)
            )
    resolved_detector = (
        detector if detector is not None else replace(resolved_catalog_detector, **supplied_legacy)
    )
    current_bins = resolved_detector.energy_bins
    missing = object()
    line_override = overrides.pop("E_grid_line", missing)
    legacy_line_override = overrides.pop("e_grid_eV", missing)
    if legacy_line_override is not missing:
        warnings.warn(
            "e_grid_eV is deprecated; use detector=Detector(energy_bins=EnergyBins(line=...)) "
            "or E_grid_line= during the D7 compatibility window",
            DeprecationWarning,
            stacklevel=2,
        )
    line_by_energy_override = overrides.pop("E_grid_line_by_energy", missing)
    brem_override = overrides.pop("E_grid_brem", missing)
    resolved_detector = replace(
        resolved_detector,
        energy_bins=EnergyBins(
            line=(
                line_override
                if line_override is not missing
                else legacy_line_override
                if legacy_line_override is not missing
                else current_bins.line
                if current_bins.line is not None
                else scan.E_grid_line
            ),
            line_by_energy=(
                line_by_energy_override
                if line_by_energy_override is not missing
                else current_bins.line_by_energy
                if current_bins.line_by_energy is not None
                else scan.E_grid_line_by_energy
            ),
            brem=(
                brem_override
                if brem_override is not missing
                else current_bins.brem
                if current_bins.brem is not None
                else scan.E_grid_brem
            ),
        ),
    )
    sweep = Sweep(
        material=spec.crystal_key,
        detector=resolved_detector,
        beam=beam,
        # The catalog speaks the flat geometry vocabulary, so the scan grid is
        # projected onto the target here rather than handed to Sweep's retired
        # flat aliases.
        target=target_from_flat(
            spec.crystal_key,
            thickness_ang=scan.thickness_ang,
            tilt_deg=scan.tilt_deg,
            tilt_azim_deg=scan.tilt_azim_deg,
            substrate=spec.substrate,
            stack=spec.stack or None,
        ),
        n_electrons=scan.n_electrons,
        n_electrons_brem=scan.n_electrons_brem,
    )
    sweep = get_fidelity_preset(fidelity).apply_sweep(sweep)
    numerics = resolve_numerics(
        _catalog(catalog_profile).profile_numerics(catalog_profile), fidelity=fidelity
    )
    sweep = replace(
        sweep,
        **{key: numerics.effective[key] for key in CONVERGENCE_KEYS},
    )
    if not overrides:
        return sweep
    # Split beam-addressed overrides (energy_keV, spot/bunch fields) from
    # Sweep-level ones so both keep working through the single **overrides API.
    beam_over = {k: overrides.pop(k) for k in list(overrides) if k in _BEAM_OVERRIDE_KEYS}
    target_over = {k: overrides.pop(k) for k in list(overrides) if k in _TARGET_OVERRIDE_KEYS}
    if overrides:
        sweep = replace(sweep, **overrides)
    if beam_over:
        sweep = replace(sweep, beam=beam_replace(sweep.beam, **beam_over))
    if target_over:
        assert sweep.target is not None
        sweep = replace(sweep, target=target_replace(sweep.target, **target_over))
    return sweep


def trajectory_sweep(
    material: str,
    *,
    tilts=PENETRATION_TILT_DEG,
    energies=(30, 50),
    thickness_ang: float | None = None,
    n_tilts: int | None = None,
    tilt_span: float | None = None,
    azim_deg: float = 0.0,
    groove_spacing_ang: float | None = None,
):
    """A small dedicated geometry sweep for the electron-penetration figures: a
    handful of polar tilts at a configurable azimuth (``azim_deg``, normal
    incidence by default), two beam energies (transport only, so the energy
    grids are irrelevant -- kept for build_cases). ``n_tilts`` panels span from
    normal incidence to ``tilt_span`` when supplied; otherwise the sparse
    default :data:`PENETRATION_TILT_DEG` set is used.

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
    stack = spec.stack or None
    substrate = spec.substrate if not stack else None
    # Blazed sawtooth entrance-face grooves (docs/superpowers/plans/
    # 2026-07-23-blazed-groove-geometry.md): forwarded straight to Sweep, which
    # validates the restricted geometry (tilt_azim_deg == 180, 0 < tilt < 90,
    # theta_obs = 90, no substrate/stack). None (the default) leaves the sweep
    # ungrooved, bit-for-bit as before.
    if n_tilts is not None or tilt_span is not None:
        count = 9 if n_tilts is None else int(n_tilts)
        span = 80.0 if tilt_span is None else float(tilt_span)
        tilt_values = np.linspace(0.0, span, count, endpoint=True)
    else:
        tilt_values = tuple(float(t) for t in tilts)
    return Sweep(
        material=spec.crystal_key,  # named stacks: the film
        beam=BeamSpec(energy_keV=list(energies)),
        detector=Detector(
            energy_bins=EnergyBins(
                line=scan.E_grid_line,
                line_by_energy=scan.E_grid_line_by_energy,
                brem=scan.E_grid_brem,
            )
        ),
        target=target_from_flat(
            spec.crystal_key,
            thickness_ang=thick,
            tilt_deg=tilt_values,
            tilt_azim_deg=float(azim_deg),
            # transport-only study; normal incidence (tilt=0) is its baseline, so
            # it opts out of the emission-sweep tilt=0 ban (issue_notes.md #1).
            allow_normal_incidence=True,
            stack=stack,
            substrate=substrate,
            groove_spacing_ang=groove_spacing_ang,
            # Finite 5x5 mm footprint for grooved and ungrooved alike (the groove
            # escape treats the slab as laterally periodic and only the launch
            # stage sees the footprint -- see montecarlo.spectrum.mc_spectrum), so
            # the penetration figures record the same electron hit/miss as the
            # sweep.
            crystal_width_mm=5.0,
            crystal_height_mm=5.0,
        ),
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
    check per thickness with :func:`pyrite.montecarlo.simulate_trajectories`
    -- the SAME prebuilt penetration-depth transport behind
    :func:`pyrite.plots.plot_penetration_survival` -- at normal incidence
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
    ``pyrite.runs.scan`` (CLI) and ``src/pyrite/apps/scan_app.py`` (interactive) --
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
