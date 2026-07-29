"""
sweep.py
============

Define a parameter sweep and expand it into the per-case dicts that
``montecarlo.run_case`` consumes.

The driver notebook sets ONE :class:`Sweep`. Every physical parameter accepts
**either a single value (fixed) or a sequence/array (swept)**; :func:`build_cases`
takes the Cartesian product over whatever is swept. So

    Sweep(tilt_deg=30.0,               tilt_azim_deg=0.0)          # one case geometry
    Sweep(tilt_deg=np.linspace(1,36,14), tilt_azim_deg=[0,9,18])   # 14*3 geometries

The beam (central energy, transverse spot, longitudinal bunch, rep-rate) lives
on ``Sweep.beam`` as a :class:`BeamSpec`; ``beam.energy_keV`` is the primary
swept axis and accepts a scalar or sequence like any other grid.

are both valid and need no other code changes.

Crystallography (composition, dominant reflections, zone axis, B-factor, default
energy grid) is looked up per material; the detector geometry defaults to the
2x2 Timepix3 quad. Only ``materials.crystal`` and the immutable material catalog
are imported here (no GPU), so this module is cheap to import and test.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

import numpy as np

from ._energy_grid import decode_energy_grid, encode_energy_grid
from .materials import CATALOG, LayerSpec
from .materials._transport_data import TRANSPORT_ELEMENTS
from .materials.crystal import dominant_reflections

ScalarOrSeq = float | Sequence[float] | np.ndarray
MATERIAL_LABELS = {key: material.label for key, material in CATALOG.materials.items()}


# ---- beam phase space -------------------------------------------------------
# Speed of light in transport-clock units: the electron clock is sum(L/beta) in
# Angstrom (c=1), so a bunch length in fs converts via c = 2997.924580 Ang/fs.
# Mirrors plots.trajectories.C_ANG_PER_FS (the same physical constant; kept in
# sync so a bunch sampled in fs and an age plotted in fs use one conversion).
C_ANG_PER_FS = 2997.924580

# Relativistic-regime ceiling for the beam energy (PLACEHOLDER, decision 9
# sub-decision open). ``beta_from_keV`` is already relativistic, but the Zhai
# PXR/CBS radiation kernels are a NONRELATIVISTIC derivation, so a MeV-scale
# beam (REGAE, 3-5 MeV) would leave the model's validity *silently*. BeamSpec
# refuses (raises) above this ceiling rather than return wrong numbers. 300 keV
# is the top of the current catalog working range (hopg) and is allowed
# (strict ``>`` refusal); it sits well below the 3-5 MeV REGAE regime. The exact
# value needs a physics call and is flagged in the validation ledger.
# Validation: relativistic-ceiling (placeholder)
BEAM_ENERGY_CEILING_KEV = 300.0

_UNSET = object()


@dataclass(frozen=True)
class BeamSpec:
    """The full electron-beam phase-space description, in one object.

    A beam is a distribution over the 6-D phase space ``(x, x', y, y', z/t, d)``
    plus its central energy ``E0``. This dataclass owns every *beam* property --
    what describes the electrons *before* they reach the crystal -- so the whole
    beam lives in one place instead of being scattered across ``Sweep`` and
    ``results.Settings``. Fields, staged by what is wired today vs. reserved as a
    documented extension point (inert default -> bit-for-bit legacy):

    * ``energy_keV`` -- central beam kinetic energy, the primary swept axis
      (``build_cases`` products over it). Scalar or sequence.
    * ``transverse_fwhm_x_mm`` / ``_y_mm`` -- per-plane Gaussian spot FWHM [mm]
      (the old isotropic ``Sweep.beam_fwhm_mm``, now elliptical -- decision 8).
      ``None`` on both recovers the legacy point-source beam. Use
      :meth:`isotropic` for the common equal-plane spot.
    * ``bunch_length_fs`` -- RMS longitudinal bunch length [fs] (THIS task).
      ``None`` -> legacy point bunch (all electrons at t=0). ``long_shape`` picks
      the sampling law; ``long_offsets_fs`` supplies explicit per-particle
      offsets (overriding the shape). Incoherent today, so a no-op on the
      spectrum -- the sampled offsets are the input the future coherent form
      factor consumes.
    * ``rep_rate_hz`` / ``bunch_charge_pc`` -- pulsed-source rep rate and
      single-bunch charge; a detected-flux multiplier only, and the source of
      truth for average current ``I = bunch_charge_pc * rep_rate_hz``.
    * ``divergence_mrad`` / ``energy_spread_frac`` -- FUTURE (x',y' and d RMS ->
      full 6-D emittance). Inert defaults keep every current run bit-for-bit.

    Mean-vs-spread: each phase-space axis has a *mean* set elsewhere (energy mean
    = ``energy_keV``; direction mean = tilt geometry; position mean = origin) and
    a *spread* owned here (``energy_spread_frac``, ``divergence_mrad``, the
    transverse FWHMs). Macro-particle *counts* (``n_electrons``) are numerical
    sampling, NOT a beam property, and deliberately stay out of ``BeamSpec``.
    """

    energy_keV: ScalarOrSeq = (30.0, 45.0, 60.0)
    transverse_fwhm_x_mm: float | None = 1.0
    transverse_fwhm_y_mm: float | None = 1.0
    bunch_length_fs: float | None = None
    long_shape: str = "gaussian"
    long_offsets_fs: tuple[float, ...] | None = None
    rep_rate_hz: float = 5000.0
    bunch_charge_pc: float = 1.0
    divergence_mrad: float | None = None
    energy_spread_frac: float | None = None

    @classmethod
    def isotropic(cls, transverse_fwhm_mm: float | None = 1.0, **kw: Any) -> "BeamSpec":
        """Convenience ctor for an azimuthally-symmetric spot (x == y)."""
        return cls(
            transverse_fwhm_x_mm=transverse_fwhm_mm,
            transverse_fwhm_y_mm=transverse_fwhm_mm,
            **kw,
        )


def beam_replace(beam: BeamSpec, **changes: Any) -> BeamSpec:
    """:func:`dataclasses.replace` for a :class:`BeamSpec` that also accepts the
    legacy isotropic aliases ``beam_fwhm_mm`` / ``transverse_fwhm_mm`` (each sets
    BOTH transverse planes). Used to route the historical scalar-spot overrides
    onto the per-plane fields without every call site knowing the split."""
    iso = changes.pop("transverse_fwhm_mm", _UNSET)
    if iso is _UNSET:
        iso = changes.pop("beam_fwhm_mm", _UNSET)
    else:
        changes.pop("beam_fwhm_mm", None)
    if iso is not _UNSET:
        changes.setdefault("transverse_fwhm_x_mm", iso)
        changes.setdefault("transverse_fwhm_y_mm", iso)
    return replace(beam, **changes)


def pm(*hkls: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return reflections together with their negatives."""
    return [item for hkl in hkls for item in (tuple(hkl), tuple(-x for x in hkl))]


# ---- Timepix3 quad geometry (fixed hardware) --------------------------------
TIMEPIX3_PIXEL_PITCH_M = 55e-6
TIMEPIX3_CHIP_WIDTH_M = 256 * TIMEPIX3_PIXEL_PITCH_M
TIMEPIX3_DISTANCE_M = 0.4
TIMEPIX3_DTHETA_OBS_DEG = float(
    np.degrees(2 * np.arctan((TIMEPIX3_CHIP_WIDTH_M / 2) / TIMEPIX3_DISTANCE_M))
)
TIMEPIX3_DOMEGA_SR = float(TIMEPIX3_CHIP_WIDTH_M**2 / TIMEPIX3_DISTANCE_M**2)


def fmt_thickness(t_ang):
    """Compact human thickness label from Angstroms: 316A / 31.6nm / 17um / 1mm."""
    if t_ang < 1e2:
        return f"{t_ang:g}A"
    if t_ang < 1e4:
        return f"{t_ang / 10:g}nm"
    if t_ang < 1e7:
        return f"{t_ang / 1e4:g}um"
    return f"{t_ang / 1e7:g}mm"


def substrate_composition(substrate):
    """Number-density composition [(element, n_per_Ang3), ...] for a substrate.
    Amorphous presets ('sio2') come from bulk density; a crystalline
    substrate already in CRYSTALS (e.g. 'silicon') uses its unit-cell density."""
    key = substrate.lower()
    if key in CATALOG.media:
        return list(CATALOG.media[key].composition)
    if substrate in CATALOG.crystals:
        return list(CATALOG.crystal(substrate).composition)
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(CATALOG.media)} "
        f"or a crystal key in {list(CATALOG.crystals)}"
    )


def stack_layers(film_composition, film_thickness_ang, stack):
    """Absorber stack [(z_top, z_bot, composition), ...] for a film at the
    entrance face (z=0..t_film) followed by each :class:`LayerSpec` in ``stack``,
    boundaries accumulating downward. Attach as a case's ``abs_layers`` so
    emitted lines/brem are attenuated by the WHOLE stack (each crystalline
    layer still RADIATES via its own layer_radiator). Beam enters the film
    side; with positive tilt (toward-detector, Zhai convention; front exit)
    the lower layers sit BEHIND the emission and do not attenuate -- they
    bite the back-exit / transmission geometry. See
    docs/multilayer-materials.md."""
    t_f = float(film_thickness_ang)
    layers = [(0.0, t_f, [(el, float(n)) for el, n in film_composition])]
    z = t_f
    for lay in stack:
        t = float(lay.thickness_ang)
        layers.append((z, z + t, substrate_composition(lay.material)))
        z += t
    return layers


def film_on_substrate_layers(
    film_composition, film_thickness_ang, substrate, substrate_thickness_ang
):
    """The 2-layer special case of :func:`stack_layers`: one film on one
    substrate (kept as the stable public name for that common stack)."""
    return stack_layers(
        film_composition,
        film_thickness_ang,
        (LayerSpec(substrate, substrate_thickness_ang),),
    )


def _radiator(cp, *, beam_uvw=None, azimuth_rad=None):
    """Coherent-radiator dict from a :func:`crystal_params` result ``cp``: crystal,
    hkl_list, B_ang2, and its mutually exclusive beam_uvw/surface_hkl orientation.
    An explicit beam_uvw overrides and clears ``cp``'s reciprocal surface. The
    ``azimuth_rad`` key is included only when given -- a bare substrate radiator
    carries no azimuth of its own (:func:`substrate_radiator`); only a stack/film
    use of a radiator (:func:`layer_radiator`, :func:`build_cases`) does. The one
    constructor behind all three radiator-dict call sites."""
    rad: dict[str, Any] = dict(
        crystal=cp["crystal"],
        hkl_list=cp["hkl_list"],
        B_ang2=cp["B_ang2"],
        beam_uvw=cp["beam_uvw"] if beam_uvw is None else tuple(beam_uvw),
        surface_hkl=cp["surface_hkl"] if beam_uvw is None else None,
    )
    if azimuth_rad is not None:
        rad["azimuth_rad"] = float(azimuth_rad)
    return rad


def substrate_radiator(substrate, n_families=4):
    """Coherent-radiation crystal params for a substrate, or None if it radiates
    no lines. A CRYSTALLINE substrate (a CRYSTALS key, e.g. 'silicon') returns
    {crystal, hkl_list, B_ang2, beam_uvw, surface_hkl} (from crystal_params) so it emits its
    own PXR/CBS; an AMORPHOUS preset ('sio2') returns None (it only
    absorbs + brems). This is the per-layer-radiation half of the multilayer
    feature -- the absorber stack (film_on_substrate_layers) is the other half.
    See docs/multilayer-materials.md."""
    if substrate.lower() in CATALOG.media:
        return None  # amorphous: no coherent lines
    if substrate in CATALOG.crystals:
        return _radiator(crystal_params(substrate, n_families))
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(CATALOG.media)} "
        f"or a crystal key in {list(CATALOG.crystals)}"
    )


def layer_radiator(layer: LayerSpec, n_families: int = 4):
    """Coherent radiator params for one stack :class:`LayerSpec`, or None if the
    layer is amorphous. Same dict as :func:`substrate_radiator` plus the
    per-layer orientation: ``beam_uvw`` (overridden if the spec sets one, clearing
    any catalog surface_hkl) and ``azimuth_rad`` (the spec's in-plane rotation,
    radians)."""
    rad = substrate_radiator(layer.material, n_families)
    if rad is None:
        return None
    if layer.beam_uvw is not None:
        rad["beam_uvw"] = tuple(layer.beam_uvw)
        rad["surface_hkl"] = None
    rad["azimuth_rad"] = float(np.deg2rad(layer.azimuth_deg))
    return rad


def crystal_params(material: str, n_families: int = 4) -> dict[str, Any]:
    """Fixed crystallography for a material: composition, the dominant
    reflections, exactly one direct beam axis [uvw] or reciprocal surface (hkl),
    the isotropic B-factor, and a sensible default photon-energy grid. Override
    the grid via Sweep.e_grid_eV."""
    crystal_key = material
    if material in CATALOG.materials:
        crystal_key = CATALOG.material(material).crystal_key
    if crystal_key not in CATALOG.crystals:
        raise ValueError(f"unknown material {material!r} (have {list(CATALOG.crystals)})")
    spec = CATALOG.crystal(crystal_key)
    if spec.E_grid is None:
        raise ValueError(f"crystal {crystal_key!r} has no configured photon-energy grid")
    hkl_list: list[tuple[int, ...]]
    if not spec.hkl_list:
        hkl_list = dominant_reflections(crystal_key, n_families=n_families, B_ang2=spec.B_ang2)
    else:
        hkl_list = list(spec.hkl_list)
    return dict(
        crystal=crystal_key,
        composition=list(spec.composition),
        hkl_list=hkl_list,
        beam_uvw=spec.beam_uvw,
        surface_hkl=spec.surface_hkl,
        B_ang2=spec.B_ang2,
        E_grid=spec.E_grid,
    )


@dataclass
class Sweep:
    """One parameter sweep.

    Each of ``thickness_ang``, ``beam.energy_keV``, ``tilt_deg``,
    ``tilt_azim_deg``, ``crystal_width_mm``, and ``crystal_height_mm`` is either
    a single number (fixed) or a sequence/array (swept); build_cases() takes the
    product. The transverse crystal dimensions must be both ``None`` (the legacy
    infinite slab) or both strictly positive full dimensions in mm; they
    default to a finite 5x5 mm footprint.

    ``beam`` is the :class:`BeamSpec` owning EVERY beam property: the swept
    central ``energy_keV`` (moved off ``Sweep`` -- decision 2), the transverse
    spot (the old ``beam_fwhm_mm``, now per-plane -- decision 8), and the
    longitudinal bunch / rep-rate / (future) emittance fields. Defaults to
    ``BeamSpec()`` -- a 1 mm isotropic spot, point bunch, 5 kHz / 1 pC.
    The remaining fields are fixed setup that rarely changes per run.
    """

    material: str  # required: no default, so a Sweep can't silently load MoSe2
    thickness_ang: ScalarOrSeq = 2e4
    beam: BeamSpec = field(default_factory=BeamSpec)
    tilt_deg: ScalarOrSeq = 30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    # Optional electron-count grids (catalog ``[profiles.*]`` settings,
    # ``n_electrons`` / ``n_electrons_brem`` keys): None -> build_cases falls
    # back to the caller's settings-level counts. Single values are typical;
    # multiple values sweep transport statistics like any other grid, and the
    # case name gains a ``ne=<line>/<brem>`` suffix so checkpoint resume
    # (keyed on (name, E0_keV)) never conflates statistics variants.
    n_electrons: ScalarOrSeq | None = None
    n_electrons_brem: ScalarOrSeq | None = None
    groove_spacing_ang: float | None = None
    # Blazed sawtooth grooves on the beam-entrance face (docs/superpowers/
    # plans/2026-07-23-blazed-groove-geometry.md). Scalar, not sweepable in
    # v1. Requires tilt_azim_deg == 180, 0 < tilt_deg < 90, theta_obs = 90,
    # no substrate/stack, no finite footprint.
    crystal_width_mm: ScalarOrSeq | None = 5.0
    crystal_height_mm: ScalarOrSeq | None = 5.0
    # fixed setup (single values) ------------------------------------------
    theta_obs_deg: float = 90.0
    n_families: int = 4
    # Optional resolved-profile cap applied after catalog-pinned or dynamically
    # selected reflections. None preserves the complete production set.
    max_reflections: int | None = None
    # two independent photon-energy grids (None -> per-material defaults):
    #   E_grid_line : fine + NARROW; where the coherent lines are evaluated (the
    #       expensive sinc^2). Lines are kinematically capped at a few keV, so it
    #       need not extend past ~4 keV.
    #   E_grid_brem : coarse + WIDE; where the smooth bremsstrahlung is evaluated
    #       (cheap). Extend to 20-40 keV / the beam energy to model the full
    #       measured spectrum without inflating the line cost. Default spans the
    #       line start up to the highest beam energy at a 50 eV step.
    E_grid_line: np.ndarray | None = None
    E_grid_line_by_energy: Mapping[float, np.ndarray] | None = None
    E_grid_brem: np.ndarray | None = None
    e_grid_eV: np.ndarray | None = None  # deprecated: alias for E_grid_line
    dtheta_obs_deg: float | None = None  # None -> Timepix3 default
    domega_sr: float | None = None  # None -> Timepix3 default
    beam_uvw: tuple | None = None  # None -> per-material default
    # GPU memory knobs: segments per matmul in the spectrum / brem kernels (None
    # -> 40000 / 20000 defaults). Lower them (e.g. 4000) to cap peak GPU memory on
    # a busy or shared device; the cost is only a little extra loop overhead.
    spec_chunk: int | None = None
    brem_chunk: int | None = None
    # crystal mosaicity (the INITIAL ANALYTIC broadening; switchable per run).
    #   mosaic=False (default) -> OFF: perfect crystal, an exact no-op.
    #   mosaic=True            -> apply the per-crystal mosaic_fwhm_deg from
    #       the catalog. Crystals without a value (diamond, silicon, the
    #       TMDs) stay perfect even with mosaic=True, so this is the "optional,
    #       excluding crystals which lack such data" switch -- HOPG has a value (0.8
    #       deg) and so DOES broaden when mosaic=True.
    #   mosaic_fwhm_deg        -> override the per-crystal value for ALL crystals in
    #       the sweep (e.g. HOPG ZYA 0.4 / ZYB 0.8 / ZYH 3.5), or supply one for a
    #       crystal that has none. Ignored unless mosaic=True.
    mosaic: bool = False
    mosaic_fwhm_deg: float | None = None
    # how the mosaic broadening (when mosaic=True) is applied:
    #   "analytic" (default) -> the cheap energy-shift Gaussian added in quadrature
    #       at detector convolution (results.store_result); the intrinsic spectrum is
    #       untouched, so one record re-broadens to any grade (plot_mosaic_comparison).
    #   "mc"                  -> the EXACT per-orientation average INSIDE mc_spectrum
    #       (broadens PXR+CBS, captures the amplitude variation + asymmetric lineshape
    #       and the mosaic yield change); the analytic term is then suppressed so the
    #       broadening is not double-counted. Costs mosaic_nodes**2 x the line hot loop
    #       (serial under CuPy). See docs/crystal-mosaicity.md.
    mosaic_route: str = "analytic"
    mosaic_nodes: int = 5  # Gauss-Hermite nodes/tilt-axis for mosaic_route="mc" (K=nodes^2)
    # film-on-substrate stack (optional; multilayer feature, docs/multilayer-materials.md).
    # substrate=None -> free-standing film (unchanged). Otherwise each case gets an
    # abs_layers stack that drives BOTH multilayer electron transport (substrate
    # backscatter into the film + substrate bremsstrahlung) AND cross-stack
    # self-absorption of the film's lines/brem. Every CRYSTALLINE layer radiates its
    # own coherent PXR/CBS lines (the film, and a crystalline substrate e.g.
    # "silicon" or "sapphire"), summed incoherently; an amorphous substrate ("sio2") adds
    # only brem + absorption.
    substrate: str | None = None  # "sio2" | a crystal key e.g. "silicon"/"sapphire"
    substrate_thickness_ang: float = 5e6  # 0.5 mm default
    # general N-layer stack under the film (mutually exclusive with substrate=;
    # substrate="x" is sugar for stack=(LayerSpec("x", substrate_thickness_ang),)).
    # Each LayerSpec carries its own thickness + orientation (beam_uvw, azimuth_deg),
    # so e.g. a few-layer film / thin a-SiO2 / thick crystalline Si device stack
    # is stack=(LayerSpec("sio2", 2850), LayerSpec("silicon", 5e6)).
    stack: Sequence[LayerSpec] | None = None
    # Transport-only escape hatch for the polar tilt_deg==0 ban (issue_notes.md
    # #1). tilt=0 is a degenerate ZERO coherent-line geometry, so emission /
    # line-grid sweeps must never sample it; the penetration-depth study, though,
    # is transport only and uses normal incidence (tilt=0, normal azimuth) as its
    # physical baseline. Only that study sets this True. azimuth==90 stays banned
    # regardless -- no legitimate use, and it carried the azim-90 ranking bug.
    allow_normal_incidence: bool = False


def _seq(x):
    """Normalize a scalar-or-sequence into a 1-D float array, order preserved."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def _quantized_angles(values: ScalarOrSeq) -> np.ndarray:
    """Round degrees to nearest half away from zero at ties, then stable-unique."""
    source = _seq(values)
    scaled = source * 2.0
    quantized = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled) / 2.0
    return np.asarray(list(dict.fromkeys(float(value) for value in quantized)), dtype=float)


def _reject_banned_angles(
    tilts: np.ndarray, azimuths: np.ndarray, *, allow_normal_incidence: bool = False
) -> None:
    """Refuse the degenerate geometries no emission sweep may use (issue_notes.md #1).

    polar tilt == 0 deg radiates zero coherent-line intensity (the tilt=0
    degeneracy the line-grid coverage search flags), and azimuth == 90 deg sits
    on a symmetry axis that mis-ranked the coverage search (the azim-90 bug).
    Guarding at case-build time makes every downstream path -- production sweeps,
    catalog grids, the analyze_line_grid_bounds diagnostic -- structurally
    incapable of silently sampling them; callers must pick a small nonzero tilt
    (e.g. 5 deg) and an azimuth off 90 (e.g. 100-180 deg) instead.

    ``allow_normal_incidence`` is the transport-only escape hatch: the
    penetration-depth study is not an emission sweep and uses tilt=0 as its
    physical baseline, so it opts out of the tilt check only. The azimuth==90 ban
    is unconditional -- it has no legitimate use.
    """
    if not allow_normal_incidence and np.any(np.isclose(tilts, 0.0)):
        raise ValueError(
            "polar tilt_deg == 0 is disallowed for emission sweeps (degenerate "
            "zero coherent-line geometry); use a small nonzero tilt such as 5 deg, "
            "or set Sweep(allow_normal_incidence=True) for a transport-only study"
        )
    if np.any(np.isclose(azimuths, 90.0)):
        raise ValueError(
            "tilt_azim_deg == 90 is disallowed (degenerate symmetry axis, the "
            "azim-90 ranking bug); use an azimuth off 90 such as 100-180 deg"
        )


def _reject_relativistic_energies(energies: np.ndarray) -> None:
    """Refuse beam energies above the nonrelativistic PXR/CBS validity ceiling.

    ``beta_from_keV`` is relativistic, but the Zhai PXR/CBS radiation kernels are
    a NONRELATIVISTIC derivation, so a MeV-scale beam (REGAE, 3-5 MeV) would
    leave the model's validity *silently*. ``build_cases`` raises here rather
    than emit wrong numbers -- no valid result exists above the ceiling today, so
    a warning would be a correctness trap (decision 9). See
    :data:`BEAM_ENERGY_CEILING_KEV` (placeholder) and the future relativistic /
    channeling work (TODO On-Hold #1/#2).

    Validation: relativistic-ceiling
    """
    hottest = float(np.max(np.atleast_1d(energies)))
    if hottest > BEAM_ENERGY_CEILING_KEV:
        raise ValueError(
            f"beam energy {hottest:g} keV exceeds the nonrelativistic PXR/CBS "
            f"model ceiling of {BEAM_ENERGY_CEILING_KEV:g} keV: the Zhai PXR/CBS "
            "kernels are a nonrelativistic derivation and yield no valid result "
            "above it. Relativistic / channeling support (REGAE 3-5 MeV) is "
            "future work (TODO On-Hold #1/#2)."
        )


def _reject_invalid_groove_geometry(
    groove_spacing_ang: float, sweep: "Sweep", tilts: np.ndarray, azimuths: np.ndarray, stack
) -> None:
    """Refuse case geometries the v1 blazed-groove entrance face cannot model
    (docs/superpowers/plans/2026-07-23-blazed-groove-geometry.md). Grooves are
    a single-slab feature machined into one flat face at a fixed in-plane
    orientation (montecarlo.groove.blazed_groove_spec), so every case must sit
    at tilt_azim_deg == 180, 0 < tilt_deg < 90, and theta_obs_deg == 90, and no
    substrate/stack may be present. Mirrors the theta_obs check in
    montecarlo.groove.blazed_groove_spec so the restriction is enforced both at
    case-build time and at spec construction.

    A finite crystal footprint IS allowed and is the default (crystal_width_mm/
    crystal_height_mm, same 5x5 mm as flat sweeps): the sub-micron groove phase
    and the mm-scale footprint are independent in transport -- the groove offsets
    the entry point within one period, the footprint kills beam electrons whose
    (tilt-elongated) spot lands off the crystal -- so the launch-time hit/miss
    classification (stored as ``hit_frac``, surfaced in the analysis_app heatmap)
    applies unchanged to grooved crystals. The alive electrons still see the
    laterally periodic groove escape profile."""
    if groove_spacing_ang <= 0:
        raise ValueError("groove_spacing_ang must be positive")
    if not np.allclose(azimuths, 180.0):
        raise ValueError("grooves require tilt_azim_deg == 180 for every case")
    if not np.all((tilts > 0.0) & (tilts < 90.0)):
        raise ValueError("grooves require 0 < tilt_deg < 90 for every case")
    if not np.isclose(sweep.theta_obs_deg, 90.0):
        raise ValueError("grooves require theta_obs_deg == 90 for every case")
    if stack is not None:
        raise ValueError("grooves are v1 single-slab only (no substrate/stack)")


def _line_grid_for_energy(sweep: Sweep, default_grid: np.ndarray, energy_keV: float) -> np.ndarray:
    """Select the fixed, legacy, mapped, or material-default line grid."""
    fixed = sweep.E_grid_line if sweep.E_grid_line is not None else sweep.e_grid_eV
    if fixed is not None:
        return np.asarray(fixed, dtype=float)
    if sweep.E_grid_line_by_energy is None:
        return np.asarray(default_grid, dtype=float)
    try:
        return np.asarray(sweep.E_grid_line_by_energy[float(energy_keV)], dtype=float)
    except KeyError:
        raise ValueError(f"no E_grid_line configured for beam energy {energy_keV:g} keV") from None


def build_cases(sweep: Sweep, n_electrons=450, n_electrons_brem=100, coherent_emission=False):
    """Expand a :class:`Sweep` into a list of run_case dicts (the Cartesian
    product over the swept thickness / tilt / azimuth / footprint, each
    crossed with every beam energy). ``crystal_width_mm`` and
    ``crystal_height_mm`` are full dimensions: both ``None`` recovers the
    legacy infinite slab, otherwise both must be positive (default: a finite
    5x5 mm footprint). Each case also carries the beam phase-space fields from
    ``sweep.beam`` (:class:`BeamSpec`): ``beam_fwhm_mm`` (x-plane FWHM; default a
    1 mm Gaussian spot, ``None`` recovers the legacy point source), the
    elliptical ``beam_fwhm_y_mm`` and the longitudinal ``bunch_length_fs`` /
    ``long_shape`` / ``long_offsets_fs`` keys joining only when they diverge from
    the isotropic point-bunch default. Optional ``n_electrons`` / ``n_electrons_brem`` sweep
    grids (catalog profile settings) cross electron-count statistics into the
    product and suffix the case name with ``ne=<line>/<brem>``; ``None`` keeps
    the scalar counts passed by the caller. Returns the ``cases`` list;
    preview it with :func:`geometry_table`."""
    cp = crystal_params(sweep.material, sweep.n_families)
    if sweep.max_reflections is not None:
        cp["hkl_list"] = cp["hkl_list"][: sweep.max_reflections]
    # line grid: fine + narrow (per-material default, per-energy mapping,
    # E_grid_line, or the deprecated e_grid_eV alias). brem grid: coarse + wide
    # -- each case spans up to that case's beam energy because brem cuts off at
    # the particle energy.
    # A uniform E_grid_brem keeps the legacy start/spacing behavior and extends
    # to each beam energy. Scalar/nonuniform grids are explicit and stay exact.
    energies = _seq(sweep.beam.energy_keV)
    _reject_relativistic_energies(energies)
    line_grids = tuple(
        _line_grid_for_energy(sweep, cp["E_grid"], float(energy)) for energy in energies
    )
    fixed_line_grid = sweep.E_grid_line if sweep.E_grid_line is not None else sweep.e_grid_eV
    if sweep.E_grid_brem is not None:
        brem_grid = np.asarray(sweep.E_grid_brem, float)
    else:
        if fixed_line_grid is not None:
            brem_start = float(np.atleast_1d(fixed_line_grid)[0])
        elif sweep.E_grid_line_by_energy is not None:
            brem_start = min(
                float(np.atleast_1d(grid)[0]) for grid in sweep.E_grid_line_by_energy.values()
            )
        else:
            brem_start = float(cp["E_grid"][0])
        brem_grid = np.arange(brem_start, float(energies.max()) * 1e3 + 50.0, 50.0)  # type: ignore[reportArgumentType]

    dtheta = TIMEPIX3_DTHETA_OBS_DEG if sweep.dtheta_obs_deg is None else sweep.dtheta_obs_deg
    domega = TIMEPIX3_DOMEGA_SR if sweep.domega_sr is None else sweep.domega_sr
    beam_uvw = cp["beam_uvw"] if sweep.beam_uvw is None else sweep.beam_uvw
    surface_hkl = cp["surface_hkl"] if sweep.beam_uvw is None else None
    # Beam phase-space fields threaded onto every case (constant across the
    # product). The legacy ``beam_fwhm_mm`` key stays == the x-plane FWHM so a
    # default isotropic spot is bit-for-bit; the elliptical ``beam_fwhm_y_mm``
    # and the longitudinal bunch / pulse-rate keys join ONLY when they actually
    # diverge from the point-bunch/default-source configuration, so existing
    # case payloads and checkpoint matching remain stable.
    b = sweep.beam
    fwhm_x = None if b.transverse_fwhm_x_mm is None else float(b.transverse_fwhm_x_mm)
    fwhm_y = None if b.transverse_fwhm_y_mm is None else float(b.transverse_fwhm_y_mm)
    beam_case: dict[str, Any] = {"beam_fwhm_mm": fwhm_x}
    if b.bunch_charge_pc != 1.0 or b.rep_rate_hz != 5000.0:
        beam_case["bunch_charge_pc"] = float(b.bunch_charge_pc)
        beam_case["rep_rate_hz"] = float(b.rep_rate_hz)
    if fwhm_y != fwhm_x:
        beam_case["beam_fwhm_y_mm"] = fwhm_y
    if b.bunch_length_fs is not None or b.long_offsets_fs is not None:
        beam_case["long_shape"] = b.long_shape
        if b.bunch_length_fs is not None:
            beam_case["bunch_length_fs"] = float(b.bunch_length_fs)
        if b.long_offsets_fs is not None:
            beam_case["long_offsets_fs"] = tuple(float(x) for x in b.long_offsets_fs)
    material_spec = CATALOG.materials.get(sweep.material)
    label = material_spec.label if material_spec is not None else sweep.material
    width_src, height_src = sweep.crystal_width_mm, sweep.crystal_height_mm
    if width_src is None and height_src is None:
        footprints = [(None, None)]
    elif width_src is None or height_src is None:
        raise ValueError("crystal_width_mm and crystal_height_mm must be supplied together")
    else:
        widths, heights = _seq(width_src), _seq(height_src)
        if not (
            np.all(np.isfinite(widths) & (widths > 0.0))
            and np.all(np.isfinite(heights) & (heights > 0.0))
        ):
            raise ValueError("crystal_width_mm and crystal_height_mm must be finite and positive")
        footprints = list(product(widths, heights))

    # crystal mosaicity (analytic, optional): None unless the run enables it AND the
    # crystal has a mosaic_fwhm_deg (or the Sweep overrides it). None -> perfect
    # crystal, so store_result adds no mosaic term (exact no-op).
    mosaic_deg = None
    if sweep.mosaic:
        mosaic_deg = (
            sweep.mosaic_fwhm_deg
            if sweep.mosaic_fwhm_deg is not None
            else CATALOG.crystal(cp["crystal"]).mosaic_fwhm_deg
        )
    mosaic_fwhm_rad = float(np.deg2rad(mosaic_deg)) if mosaic_deg else None
    if sweep.mosaic_route not in ("analytic", "mc"):
        raise ValueError(f"mosaic_route must be 'analytic' or 'mc', got {sweep.mosaic_route!r}")
    # exact MC route: do the broadening INSIDE mc_spectrum and turn the analytic
    # store_result term OFF (mutually exclusive -- applying both double-counts).
    # Both are None when there is no mosaic to apply.
    mosaic_mc = mosaic_fwhm_rad is not None and sweep.mosaic_route == "mc"
    mosaic_analytic_rad = None if mosaic_mc else mosaic_fwhm_rad
    mosaic_mc_rad = mosaic_fwhm_rad if mosaic_mc else None

    brem_case_grid = encode_energy_grid(brem_grid)
    tilts = _quantized_angles(sweep.tilt_deg)
    azimuths = _quantized_angles(sweep.tilt_azim_deg)
    _reject_banned_angles(tilts, azimuths, allow_normal_incidence=sweep.allow_normal_incidence)

    def _electron_counts(grid, fallback, label):
        if grid is None:
            return [int(fallback)]
        raw = _seq(grid)
        counts = [int(value) for value in raw]
        if any(count <= 0 or count != value for count, value in zip(counts, raw, strict=True)):
            raise ValueError(f"{label} electron counts must be positive integers")
        return counts

    ne_pairs = list(
        product(
            _electron_counts(sweep.n_electrons, n_electrons, "n_electrons"),
            _electron_counts(sweep.n_electrons_brem, n_electrons_brem, "n_electrons_brem"),
        )
    )
    explicit_ne = sweep.n_electrons is not None or sweep.n_electrons_brem is not None

    # normalize the substrate sugar onto the general stack (mutually exclusive)
    stack = sweep.stack
    if sweep.substrate is not None:
        if stack is not None:
            raise ValueError("give either substrate= or stack=, not both")
        stack = (LayerSpec(sweep.substrate, sweep.substrate_thickness_ang),)

    if sweep.groove_spacing_ang is not None:
        _reject_invalid_groove_geometry(sweep.groove_spacing_ang, sweep, tilts, azimuths, stack)

    cases = []
    for i_c, (thickness, tilt, azim, (width, height)) in enumerate(
        product(
            _seq(sweep.thickness_ang),
            tilts,
            azimuths,
            footprints,
        )
    ):
        name = f"{label} {fmt_thickness(thickness)} pol={tilt:g} az={azim:g}"
        abs_layers = None
        layer_radiators = None
        if stack is not None:
            name = f"{name} on {'+'.join(lay.material for lay in stack)}"
            abs_layers = stack_layers(cp["composition"], thickness, stack)
            # per-layer coherent radiators, aligned with abs_layers: the film (its
            # own crystal params) then one per stack Layer (crystal params + the
            # Layer's own orientation if crystalline, None if amorphous). This is
            # what lets a crystalline substrate emit its own PXR/CBS lines
            # (per-layer radiation, slice 3).
            layer_radiators = [
                # stack azimuths are relative to the film -> azimuth_rad=0.0
                _radiator(cp, beam_uvw=beam_uvw, azimuth_rad=0.0),
                *(layer_radiator(lay, sweep.n_families) for lay in stack),
            ]
        if width is not None:
            name = f"{name} footprint={width:g}x{height:g}mm"
        if sweep.groove_spacing_ang is not None:
            name = f"{name} groove={sweep.groove_spacing_ang / 1e4:g}um"
        for i_e, E0 in enumerate(energies):
            line_case_grid = encode_energy_grid(line_grids[i_e])
            for i_n, (ne_line, ne_brem) in enumerate(ne_pairs):
                case_name = f"{name} ne={ne_line}/{ne_brem}" if explicit_ne else name
                cases.append(
                    dict(
                        name=case_name,
                        crystal=cp["crystal"],
                        composition=cp["composition"],
                        hkl_list=cp["hkl_list"],
                        B_ang2=cp["B_ang2"],
                        E0_keV=float(E0),
                        thickness_ang=float(thickness),
                        crystal_width_mm=None if width is None else float(width),
                        crystal_height_mm=None if height is None else float(height),
                        **beam_case,
                        E_grid=line_case_grid,  # legacy key (== line grid)
                        E_grid_line=line_case_grid,
                        E_grid_brem=(
                            (
                                brem_case_grid[0],
                                float(E0) * 1e3 + brem_case_grid[2],
                                brem_case_grid[2],
                            )
                            if isinstance(brem_case_grid, tuple)
                            else brem_case_grid
                        ),
                        theta_obs_rad=np.deg2rad(sweep.theta_obs_deg),
                        tilt_deg=float(tilt),
                        tilt_azim_deg=float(azim),
                        **(
                            {"groove_spacing_ang": float(sweep.groove_spacing_ang)}
                            if sweep.groove_spacing_ang is not None
                            else {}
                        ),
                        # coherent segment sum: divergence-only key (absent -> the
                        # incoherent default, bit-for-bit case payload).
                        **({"coherent_emission": True} if coherent_emission else {}),
                        beam_uvw=beam_uvw,
                        surface_hkl=surface_hkl,
                        mosaic_fwhm_rad=mosaic_analytic_rad,  # analytic term (None if route="mc")
                        mosaic_mc_fwhm_rad=mosaic_mc_rad,  # exact MC route (None if route="analytic")
                        mosaic_mc_nodes=sweep.mosaic_nodes,
                        abs_layers=abs_layers,  # None -> single slab; else film-on-substrate stack
                        layer_radiators=layer_radiators,  # per-layer coherent radiators (None -> slab)
                        brem_file=None,
                        Ne=ne_line,
                        Ne_brem=ne_brem,
                        seed=1000 * i_c + 10 * i_e + 1 + 100_000_000 * i_n,
                        spec_chunk=sweep.spec_chunk,  # GPU rows/matmul (None -> run_case default)
                        brem_chunk=sweep.brem_chunk,
                        # used downstream (detector model / unit scaling); ignored by run_case:
                        dtheta_obs_rad=np.deg2rad(dtheta),
                        domega_sr=domega,
                    )
                )
    return cases


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


# ---- compute-cost proxy (progress weighting; instrumentation, not physics) ----
# The flat "N of M cases" progress readout misrepresents reality because per-case
# cost varies by multiples across a sweep: a 60 keV case runs several times the
# matmul work of a 20 keV one (deeper penetration -> more transport segments,
# wider per-E0 line grid). These helpers estimate the RELATIVE matmul-element
# count per case so bars/meters can denominate by compute instead of case count.
#
# This is instrumentation, NOT a physics claim -- no validation-ledger entry.
# Only the ORDERING of the weights matters (callers normalize by the sweep
# total), so every global constant (the electron mean free path, unit choices)
# cancels and the proxy is deliberately kept coarse: do not over-fit it.

_COST_E_CUT_KEV = 5.0  # matches transport.simulate_trajectories' default E_cut_keV


def _joy_luo_dEds_keV_per_ang(composition, E_keV):
    """Joy-Luo modified-Bethe stopping-power magnitude [keV/Angstrom].

    Mirrors ``montecarlo.transport._dEds_compound`` (same constants) so the cost
    proxy's CSDA range tracks the transport model whose runtime it predicts,
    while importing only the leaf ``TRANSPORT_ELEMENTS`` table -- ``sweep.py``
    stays GPU-free and cheap to import. Additive over elements with number
    densities ``n_i`` [1/Angstrom^3]; ``E_keV`` may be a scalar or an array."""
    E_keV = np.asarray(E_keV, dtype=float)
    total = np.zeros_like(E_keV)
    for el, n_i in composition:
        p = TRANSPORT_ELEMENTS[el]
        Z = p["Z"]
        k = 0.731 + 0.0688 * np.log10(Z)
        total = total + (n_i / 0.602214076) * Z * np.log(
            1.166 * (E_keV + k * p["J_keV"]) / p["J_keV"]
        )
    return 7.85e-4 / E_keV * total  # magnitude only (transport uses the negative)


def _csda_profile(composition, E0_keV, e_cut_keV=_COST_E_CUT_KEV, n_quad=64):
    """Cumulative continuous-slowing-down range [Angstrom] vs energy.

    Returns ``(E, cum)`` on ``n_quad`` points from ``e_cut_keV`` to ``E0_keV``,
    where ``cum[i]`` is the CSDA path length from ``e_cut`` up to ``E[i]``
    (trapezoid integral of ``1/|dE/ds|``). ``cum[-1]`` is the full range at
    ``E0``. Monotonic in ``E0``; used both for the range and to invert it (how
    far a slab lets an electron travel before it enters the next layer)."""
    E = np.linspace(e_cut_keV, E0_keV, n_quad)
    inv_dEds = 1.0 / _joy_luo_dEds_keV_per_ang(composition, E)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv_dEds[1:] + inv_dEds[:-1]) * np.diff(E))])
    return E, cum


def _mean_segments_per_electron(case):
    """Relative transport-segment count per incident electron for a case.

    N_seg per electron ~ (path length) / (mean free path); taking the free path
    as a case-independent constant (it cancels in the normalized weights), the
    ordering is set by the CSDA path length through the stack. Electrons slow
    from ``E0`` and deposit over ``min(range-in-layer, layer thickness)`` before
    entering the next layer at the reduced energy, so a multilayer/stack case
    walks its ``abs_layers`` rather than assuming a single slab; a bare slab is
    the one-layer case over ``thickness_ang``."""
    abs_layers = case.get("abs_layers")
    if abs_layers:
        layers = [(float(z_bot) - float(z_top), comp) for (z_top, z_bot, comp) in abs_layers]
    else:
        layers = [(float(case["thickness_ang"]), case["composition"])]
    path = 0.0
    E = float(case["E0_keV"])
    for thickness_ang, comp in layers:
        if E <= _COST_E_CUT_KEV:
            break
        grid, cum = _csda_profile(comp, E)
        total_range = float(cum[-1])
        if total_range <= thickness_ang:
            path += total_range  # electron stops inside this layer
            break
        path += thickness_ang  # electron crosses this layer, enters the next slower
        E = float(np.interp(total_range - thickness_ang, cum, grid))
    return path


def case_cost(case):
    """Estimated RELATIVE matmul-element count for one ``run_case`` dict.

    The dominant runtime is the chunked ``mc_spectrum`` / ``mc_brem_spectrum``
    matmuls of shape ``(n_segments, nbins)``, so per case::

        cost ~ N_seg_line * nbins_line  +  N_seg_brem * nbins_brem

    with ``N_seg = electrons * segments-per-electron`` (see
    :func:`_mean_segments_per_electron`) and ``nbins_line`` / ``nbins_brem`` the
    evaluated grid widths (both vary by ``E0``). Pure function of the case dict;
    runs no transport. Callers normalize by the sweep total, so the absolute
    scale is meaningless -- only the ordering across cases is used."""
    nbins_line = decode_energy_grid(case.get("E_grid_line", case["E_grid"])).size
    brem = case.get("E_grid_brem")
    nbins_brem = decode_energy_grid(brem).size if brem is not None else 0
    steps = _mean_segments_per_electron(case)
    n_line = float(case.get("Ne", 0) or 0)
    n_brem = float(case.get("Ne_brem", 0) or 0)
    return steps * (n_line * nbins_line + n_brem * nbins_brem)


def sweep_cost_weights(cases):
    """Relative compute weight of every case: ``({(name, E0_keV): cost}, total)``.

    Deterministic from the case list alone, so a viewer can rebuild the same
    weights from a reconstructed sweep without the weights being shipped in a
    checkpoint. ``total`` is ``sum(weights.values())`` (``0.0`` for no cases)."""
    weights = {(c["name"], float(c["E0_keV"])): case_cost(c) for c in cases}
    return weights, float(sum(weights.values()))


def _grid_key(case):
    """(name, E0_keV) progress key for a case -- what checkpoints/weights index on."""
    return (case["name"], float(case["E0_keV"]))


def scan_grid_rows(cases, *, cached=(), excluded=(), running=(), done=(), weights=None):
    """Aggregate a case list into rows for the notebook ``scan_grid`` renderer.

    One row per distinct polar tilt, one cell per distinct beam energy; each cell
    reports the FRACTION of its ``(tilt, E0_keV)`` cases in each state and the
    SUMMED relative compute cost of that group (so the renderer can size the
    heavy corner). Pass every requested case (kept + penetration-excluded) as
    ``cases`` so fully-excluded or unrequested ``(tilt, E0)`` slots render empty.

    ``cached`` / ``running`` / ``done`` are iterables of ``(name, E0_keV)`` keys
    (see :func:`sweep_cost_weights`); ``excluded`` may be either such keys or the
    dropped case dicts (as returned by ``config.gate_cases_by_penetration``).
    ``weights`` is the ``{(name, E0): cost}`` map (default: recomputed via
    :func:`case_cost`). Returns ``(energies, rows)`` with ``energies`` sorted
    ascending. Instrumentation only -- no physics claim."""
    from collections import defaultdict

    cached, running, done = set(cached), set(running), set(done)
    excluded_keys = {item if isinstance(item, tuple) else _grid_key(item) for item in excluded}
    if weights is None:
        weights = {_grid_key(c): case_cost(c) for c in cases}

    energies = sorted({float(c["E0_keV"]) for c in cases})
    tilts = sorted({float(c["tilt_deg"]) for c in cases})
    agg: dict[tuple[float, float], dict[str, float]] = defaultdict(
        lambda: {"total": 0, "cached": 0, "done": 0, "running": 0, "excluded": 0, "weight": 0.0}
    )
    for c in cases:
        tilt, E0 = float(c["tilt_deg"]), float(c["E0_keV"])
        key = _grid_key(c)
        cell = agg[(tilt, E0)]
        cell["total"] += 1
        cell["weight"] += weights.get(key, 0.0)
        if key in excluded_keys:
            cell["excluded"] += 1
        elif key in done:
            cell["done"] += 1
        elif key in running:
            cell["running"] += 1
        elif key in cached:
            cell["cached"] += 1

    rows = []
    for tilt in tilts:
        cells = []
        for E0 in energies:
            cell = agg.get((tilt, E0))
            if not cell or cell["total"] == 0:
                cells.append(None)
                continue
            n = cell["total"]
            cells.append(
                {
                    "cached": cell["cached"] / n,
                    "done": cell["done"] / n,
                    "running": cell["running"] / n,
                    "excluded": cell["excluded"] / n,
                    "weight": cell["weight"],
                }
            )
        rows.append({"label": f"{tilt:g}°", "cells": cells})
    return energies, rows
