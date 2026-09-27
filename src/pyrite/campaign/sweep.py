"""
sweep.py
============

Define a parameter sweep and expand it into the typed per-case records that
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

import warnings
from collections.abc import Sequence
from dataclasses import InitVar, asdict, dataclass, field, replace
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._energy_grid_encoding import decode_energy_grid, encode_energy_grid
from .._grid_semantics import resolution_num
from .._line_grid_policy import (
    AUTOMATIC_BANDWIDTH_POLICY,
    LineGridPolicy,
    environment_overrides_present,
    kinematic_line_stop_eV,
    line_quadrature_from_payload,
    line_start_eV,
    resolve_line_grid_policy,
)
from .._numerics import (
    validate_bremsstrahlung_model,
    validate_elastic_model,
    validate_inelastic_numerics,
)
from .._photon_continuum_floor import floored_lattice_start_eV
from ..detectors import Detector
from ..materials import CATALOG, LayerSpec
from ..materials.crystal import reciprocal_g_vector
from ..montecarlo.case import Case
from ..montecarlo.transport import spliced_stopping_keV_per_ang
from ..montecarlo.transverse import TransverseDistribution, resolve_transverse_distribution
from .geometry import (  # noqa: F401  (re-exported: pyrite.campaign.sweep is the stable import path)
    BlazedGrooves,
    Footprint,
    Layer,
    LoweredTarget,
    ScalarOrSeq,
    Slab,
    Stack,
    Target,
    _quantized_angles,
    _radiator,
    _reject_banned_angles,
    _RetiredFlatInput,
    _seq,
    crystal_params,
    film_on_substrate_layers,
    layer_radiator,
    retired_flat_input,
    stack_layers,
    substrate_composition,
    substrate_radiator,
    target_flat_fields,
    target_from_flat,
    target_replace,
)
from .longitudinal import LongitudinalDistribution, resolve_longitudinal_distribution

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

if TYPE_CHECKING:
    from ..montecarlo.gdf import GDFBeam

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
    * ``longitudinal`` -- frozen declarative Gaussian, wavelength-matched
      microtrain, or compressed-bunch policy. It is mutually exclusive with
      the legacy flat bunch fields and resolves per material/energy/geometry
      case without depending on macro-particle count.
    * ``rep_rate_hz`` / ``bunch_charge_pc`` -- pulsed-source rep rate and
      single-bunch charge; a detected-flux multiplier only, and the source of
      truth for average current ``I = bunch_charge_pc * rep_rate_hz``.
    * ``transverse`` -- frozen declarative Courant-Snyder policy, the canonical
      way to give the beam a finite emittance. Normalized emittance is stored
      and the geometric value is derived per case, because geometric emittance
      is not invariant across the swept ``energy_keV`` axis. Mutually exclusive
      with the legacy spot FWHMs, which describe a zero-emittance waist.
    * ``energy_spread_frac`` -- RMS relative energy deviation, sampled as an
      independent uncorrelated draw (no chirp model: ``<t delta> = 0``).
    * ``divergence_mrad`` -- DERIVED, read-only. The RMS slope at the case
      energy, available once a ``transverse`` policy is set. It is not a stored
      input: a fixed number here would be energy-independent, and the physical
      RMS slope of one beam falls as ``1/sqrt(beta*gamma)``.

    Reference plane: every field describes the beam **at the crystal entrance
    face**. There is no space-charge model and no source-to-crystal beamline
    transport, so gun-exit numbers must not be entered here and read as
    physical. See ``docs/physics/beam-transport/beam-phase-space.md``.

    Mean-vs-spread: each phase-space axis has a *mean* set elsewhere (energy mean
    = ``energy_keV``; direction mean = tilt geometry; position mean = origin) and
    a *spread* owned here (``energy_spread_frac``, the ``transverse`` policy, the
    transverse FWHMs). Macro-particle *counts* (``n_electrons``) are numerical
    sampling, NOT a beam property, and deliberately stay out of ``BeamSpec``.
    Charge and rep rate are normalization, not phase space, and stay inert on
    the sampler.
    """

    energy_keV: ScalarOrSeq = (30.0, 45.0, 60.0)
    transverse_fwhm_x_mm: float | None = 1.0
    transverse_fwhm_y_mm: float | None = 1.0
    bunch_length_fs: float | None = None
    long_shape: str = "gaussian"
    long_offsets_fs: tuple[float, ...] | None = None
    longitudinal: LongitudinalDistribution | None = None
    transverse: TransverseDistribution | None = None
    rep_rate_hz: float = 5000.0
    bunch_charge_pc: float = 1.0
    divergence_mrad: float | None = None
    energy_spread_frac: float | None = None
    source: str = "analytic"
    gdf_shape_only: bool = False
    gdf_path: str | None = None
    gdf_time_s: float | None = None
    gdf_time_tolerance_s: float = 1e-15
    gdf_screen_position_m: float | None = None
    gdf_screen_tolerance_m: float = 1e-9
    gdf_normalization: str = "pyrite_current"
    gdf_repetition_rate_hz: float | None = None
    gdf_z_origin_m: float | None = None

    def gdf_beam(self) -> GDFBeam | None:
        """Validate source settings and load a GPT snapshot when selected."""
        if not isinstance(self.gdf_shape_only, bool):
            raise ValueError("gdf_shape_only must be a boolean")
        if self.source not in {"analytic", "gpt_gdf"}:
            raise ValueError("beam source must be analytic or gpt_gdf")
        if self.source == "analytic":
            if (
                self.gdf_shape_only
                or self.gdf_path is not None
                or self.gdf_time_s is not None
                or self.gdf_screen_position_m is not None
                or self.gdf_screen_tolerance_m != 1e-9
                or self.gdf_z_origin_m is not None
                or self.gdf_repetition_rate_hz is not None
                or self.gdf_normalization != "pyrite_current"
                or self.gdf_time_tolerance_s != 1e-15
            ):
                raise ValueError("GDF settings require source='gpt_gdf'")
            return None
        if not self.gdf_path:
            raise ValueError("gpt_gdf requires gdf_path")
        if self.gdf_z_origin_m is None or not np.isfinite(self.gdf_z_origin_m):
            raise ValueError(
                "gpt_gdf requires explicit finite gdf_z_origin_m (target origin in GPT lab coordinates)"
            )
        if (
            self.transverse is not None
            or self.longitudinal is not None
            or self.bunch_length_fs is not None
            or self.long_offsets_fs is not None
            or self.energy_spread_frac is not None
            or self.divergence_mrad is not None
            or self.long_shape != "gaussian"
            or self.transverse_fwhm_x_mm is not None
            or self.transverse_fwhm_y_mm is not None
        ):
            raise ValueError(
                "gpt_gdf is incompatible with analytic phase-space fields; clear both spot FWHMs"
            )
        rate = self.gdf_repetition_rate_hz
        if self.gdf_normalization == "gdf_charge":
            if rate is None or not np.isfinite(rate) or rate <= 0:
                raise ValueError("gdf_charge requires finite positive gdf_repetition_rate_hz")
        elif rate is not None:
            raise ValueError("gdf_repetition_rate_hz requires gdf_normalization='gdf_charge'")
        from ..montecarlo.gdf import load_gdf_beam

        return load_gdf_beam(
            self.gdf_path,
            self.gdf_time_s,
            self.gdf_time_tolerance_s,
            self.gdf_normalization,
            screen_position_m=self.gdf_screen_position_m,
            screen_tolerance_m=self.gdf_screen_tolerance_m,
        )

    def derived_divergence_mrad(self, energy_keV: float) -> tuple[float, float] | None:
        """Per-plane RMS slope [mrad] at ``energy_keV``, or ``None`` if collimated.

        The canonical replacement for the legacy ``divergence_mrad`` input,
        which cannot be right across a multi-energy sweep: at fixed normalized
        emittance the physical RMS slope falls as ``1/sqrt(beta*gamma)``.
        """
        if self.transverse is None:
            return None
        resolved = resolve_transverse_distribution(self.transverse, energy_keV=energy_keV)
        return (resolved.x.sigma_slope_rad * 1e3, resolved.y.sigma_slope_rad * 1e3)

    @classmethod
    def isotropic(cls, transverse_fwhm_mm: float | None = 1.0, **kw: Any) -> BeamSpec:
        """Convenience ctor for an azimuthally-symmetric spot (x == y)."""
        return cls(
            transverse_fwhm_x_mm=transverse_fwhm_mm,
            transverse_fwhm_y_mm=transverse_fwhm_mm,
            **kw,
        )

    @classmethod
    def with_transverse(cls, transverse: TransverseDistribution, **kw: Any) -> BeamSpec:
        """Convenience ctor for a Courant-Snyder beam, clearing the legacy spot.

        The spot FWHMs default to 1 mm, and they are mutually exclusive with a
        ``transverse`` policy, so they have to be cleared explicitly. This does
        that in one step rather than making every caller remember it.
        """
        return cls(
            transverse=transverse,
            transverse_fwhm_x_mm=None,
            transverse_fwhm_y_mm=None,
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


@dataclass
class Sweep:
    """One parameter sweep.

    ``target`` owns the whole geometry: the film, its swept thickness/tilt, the
    footprint, any entrance face, and any layers beneath it. The flat geometry
    spellings (``thickness_ang``, ``tilt_deg``, ``tilt_azim_deg``,
    ``crystal_width_mm``, ``crystal_height_mm``, ``groove_spacing_ang``,
    ``substrate``, ``substrate_thickness_ang``, ``stack``,
    ``allow_normal_incidence``, ``mosaic``) remain accepted as construction-time aliases
    that build that target; they are ``InitVar``s, so they do NOT survive as
    attributes -- read geometry through ``sweep.target`` (or
    :func:`~pyrite.campaign.geometry.target_flat_fields` for the flat view).
    Passing both ``target=`` and any flat geometry input is an error.

    Each of ``target.thickness``, ``beam.energy_keV``, ``target.tilt_deg``, and
    ``target.tilt_azim_deg`` is either a single number (fixed) or a
    sequence/array (swept); build_cases() takes the product. The transverse
    crystal dimensions must be both ``None`` (the legacy infinite slab) or both
    strictly positive full dimensions in mm; they default to a finite 5x5 mm
    footprint.

    ``beam`` is the :class:`BeamSpec` owning EVERY beam property: the swept
    central ``energy_keV`` (moved off ``Sweep`` -- decision 2), the transverse
    spot (the old ``beam_fwhm_mm``, now per-plane -- decision 8), and the
    longitudinal bunch / rep-rate / (future) emittance fields. Defaults to
    ``BeamSpec()`` -- a 1 mm isotropic spot, point bunch, 5 kHz / 1 pC.
    The remaining fields are fixed setup that rarely changes per run.
    """

    # ``material`` stays a real field: it is the sweep's identity, read far
    # beyond geometry. ``__post_init__`` keeps it == ``target.material``, so it
    # is a mirror of the target rather than a second source of truth.
    material: str | None = None
    thickness_ang: InitVar[ScalarOrSeq] = retired_flat_input("thickness_ang")
    beam: BeamSpec = field(default_factory=BeamSpec)
    tilt_deg: InitVar[ScalarOrSeq] = retired_flat_input("tilt_deg")
    tilt_azim_deg: InitVar[ScalarOrSeq] = retired_flat_input("tilt_azim_deg")
    # Optional electron-count grids (catalog ``[profiles.*]`` settings,
    # ``n_electrons`` / ``n_electrons_brem`` keys): None -> build_cases falls
    # back to the caller's settings-level counts. Single values are typical;
    # multiple values sweep transport statistics like any other grid, and the
    # case name gains a ``ne=<line>/<brem>`` suffix so checkpoint resume
    # (keyed on (name, E0_keV)) never conflates statistics variants.
    n_electrons: ScalarOrSeq | None = None
    n_electrons_brem: ScalarOrSeq | None = None
    groove_spacing_ang: InitVar[float | None] = retired_flat_input("groove_spacing_ang")
    # Blazed sawtooth grooves on the beam-entrance face (docs/superpowers/
    # plans/2026-07-23-blazed-groove-geometry.md). Scalar, not sweepable in
    # v1. Requires tilt_azim_deg == 180, 0 < tilt_deg < 90, theta_obs = 90,
    # and no substrate/stack. A finite footprint IS compatible and is the
    # default -- see geometry.BlazedGrooves, which owns those rules now.
    crystal_width_mm: InitVar[ScalarOrSeq | None] = retired_flat_input("crystal_width_mm")
    crystal_height_mm: InitVar[ScalarOrSeq | None] = retired_flat_input("crystal_height_mm")
    # fixed setup (single values) ------------------------------------------
    theta_obs_deg: InitVar[float | None] = None
    n_families: int = 4
    # Optional resolved-profile cap applied after catalog-pinned or dynamically
    # selected reflections. None preserves the complete production set.
    max_reflections: int | None = None
    dtheta_obs_deg: InitVar[float | None] = None
    domega_sr: InitVar[float | None] = None
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
    # ``mosaic`` is a property of the target crystal, so it lives on
    # ``Sweep.target`` and this is an alias that normalizes onto it. The
    # ``mosaic_fwhm_deg`` override stays here: it is a run-level substitution for
    # the catalog value, not target state.
    mosaic: InitVar[bool] = retired_flat_input("mosaic")
    mosaic_fwhm_deg: float | None = None
    # how the mosaic broadening (when mosaic=True) is applied:
    #   "analytic" (default) -> the cheap energy-shift Gaussian added in quadrature
    #       at detector convolution (results.store_result); the response-free source spectrum is
    #       untouched, so one record re-broadens to any grade (plot_mosaic_comparison).
    #   "mc"                  -> the EXACT per-orientation average INSIDE mc_spectrum
    #       (broadens PXR+CBS, captures the amplitude variation + asymmetric lineshape
    #       and the mosaic yield change); the analytic term is then suppressed so the
    #       broadening is not double-counted. Costs mosaic_nodes**2 x the line hot loop
    #       (serial under CuPy). See docs/physics/materials/crystal-mosaicity.md.
    mosaic_route: str = "analytic"
    mosaic_nodes: int = 5  # Gauss-Hermite nodes/tilt-axis for mosaic_route="mc" (K=nodes^2)
    # Per-call automatic line-grid policy (issue #101). The TOP of the precedence
    # chain together with an explicit ``EnergyBins.line``: supplying one here
    # outranks ``PYRITE_ENERGY_GRID_*`` and any stored catalog row. Accepts
    # ``max_spacing_eV``, ``backend_safety_ulps``, ``max_points``, and a
    # per-observable ``rtol`` mapping. ``None`` leaves the chain to env > stored
    # artifact > built-in automatic resolution.
    line_grid_policy: dict[str, Any] | None = None
    # film-on-substrate stack (optional; multilayer feature, docs/physics/materials/multilayer-materials.md).
    # substrate=None -> free-standing film (unchanged). Otherwise each case gets an
    # abs_layers stack that drives BOTH multilayer electron transport (substrate
    # backscatter into the film + substrate bremsstrahlung) AND cross-stack
    # self-absorption of the film's lines/brem. Every CRYSTALLINE layer radiates its
    # own coherent PXR/CBS lines (the film, and a crystalline substrate e.g.
    # "silicon" or "sapphire"), summed incoherently; an amorphous substrate ("sio2") adds
    # only brem + absorption.
    # RETIRED in favour of ``target=Stack.on_substrate(...)``: still accepted,
    # but it warns. The catalog TOML keeps its own ``substrate`` key -- that is a
    # data vocabulary, not the object model.
    substrate: InitVar[str | None] = retired_flat_input("substrate")
    substrate_thickness_ang: InitVar[float] = retired_flat_input("substrate_thickness_ang")
    # general N-layer stack under the film (mutually exclusive with substrate=;
    # substrate="x" is sugar for stack=(LayerSpec("x", substrate_thickness_ang),)).
    # Each LayerSpec carries its own thickness + orientation (beam_uvw, azimuth_deg),
    # so e.g. a few-layer film / thin a-SiO2 / thick crystalline Si device stack
    # is stack=(LayerSpec("sio2", 2850), LayerSpec("silicon", 5e6)).
    stack: InitVar[Sequence[LayerSpec] | None] = retired_flat_input("stack")
    # Transport-only escape hatch for the polar tilt_deg==0 ban (issue_notes.md
    # #1). tilt=0 is a degenerate ZERO coherent-line geometry, so emission /
    # line-grid sweeps must never sample it; the penetration-depth study, though,
    # is transport only and uses normal incidence (tilt=0, normal azimuth) as its
    # physical baseline. Only that study sets this True. azimuth==90 stays banned
    # regardless -- no legitimate use, and it carried the azim-90 ranking bug.
    allow_normal_incidence: InitVar[bool] = retired_flat_input("allow_normal_incidence")
    # Canonical detector owner. Kept after existing fields so legacy positional
    # Sweep construction retains its historical argument order.
    detector: Detector | None = None
    # Canonical geometry owner. Every flat geometry input above is an InitVar
    # that normalizes onto this field, so the target is the only geometry state
    # a Sweep carries and ``dataclasses.replace`` preserves it.
    target: Target | None = None

    def __post_init__(
        self,
        thickness_ang: ScalarOrSeq,
        tilt_deg: ScalarOrSeq,
        tilt_azim_deg: ScalarOrSeq,
        groove_spacing_ang: float | None,
        crystal_width_mm: ScalarOrSeq | None,
        crystal_height_mm: ScalarOrSeq | None,
        theta_obs_deg: float | None,
        dtheta_obs_deg: float | None,
        domega_sr: float | None,
        mosaic: bool,
        substrate: str | None,
        substrate_thickness_ang: float,
        stack: Sequence[LayerSpec] | None,
        allow_normal_incidence: bool,
    ) -> None:
        """Normalize the legacy flat detector and geometry inputs onto their owners.

        Both follow one shape: a nested object (``detector``, ``target``) is the
        canonical spelling, the flat aliases still construct one, and supplying
        both is an error rather than a silent precedence rule. Geometry lands on
        ``target`` here so an invalid target fails in the user's script instead of
        after profile resolution.
        """
        self._normalize_detector(theta_obs_deg, dtheta_obs_deg, domega_sr)
        # Only the sentinel means "not supplied": an explicit None is a statement
        # (``crystal_width_mm=None`` IS the infinite slab, ``substrate=None`` IS a
        # free-standing film), so it reaches the target like any other value.
        flat: dict[str, Any] = {
            name: value
            for name, value in (
                ("thickness_ang", thickness_ang),
                ("tilt_deg", tilt_deg),
                ("tilt_azim_deg", tilt_azim_deg),
                ("groove_spacing_ang", groove_spacing_ang),
                ("crystal_width_mm", crystal_width_mm),
                ("crystal_height_mm", crystal_height_mm),
                ("substrate", substrate),
                ("substrate_thickness_ang", substrate_thickness_ang),
                ("stack", stack),
                ("allow_normal_incidence", allow_normal_incidence),
                ("mosaic", mosaic),
            )
            if not isinstance(value, _RetiredFlatInput)
        }
        retired = [name for name in ("substrate", "substrate_thickness_ang") if name in flat]
        if retired:
            warnings.warn(
                f"Sweep({'=, '.join(retired)}=) is deprecated; build the target instead: "
                "target=Stack.on_substrate(material, thickness_ang, substrate, "
                "substrate_thickness_ang)",
                DeprecationWarning,
                stacklevel=3,
            )

        if self.target is None:
            if self.material is None:
                raise TypeError("Sweep needs a material= (or a target= carrying one)")
            self.target = target_from_flat(self.material, **flat)
        else:
            if not isinstance(self.target, (Slab, Stack)):
                raise TypeError("target must be a Slab or a Stack")
            if flat:
                joined = ", ".join(sorted(flat))
                raise ValueError(
                    f"conflicting nested target and legacy flat geometry input(s): {joined}; "
                    "supply a Slab/Stack or the flat geometry fields, not both"
                )
            if self.material is not None and self.material != self.target.material:
                # replace(sweep, material=...) is the one flat geometry override
                # that still works, because material is a real field the target
                # mirrors -- it re-selects the film rather than adding a source.
                self.target = target_replace(self.target, material=self.material)
        self.material = self.target.material

    def _normalize_detector(
        self,
        theta_obs_deg: float | None,
        dtheta_obs_deg: float | None,
        domega_sr: float | None,
    ) -> None:
        """Normalize legacy flat detector inputs onto ``detector``.

        A supplied nested detector and flat aliases may coexist only when their
        active values agree. This keeps old construction working while making
        mixed-source mistakes fail at the public boundary.
        """
        supplied = {
            "theta_obs_deg": theta_obs_deg,
            "dtheta_obs_deg": dtheta_obs_deg,
            "domega_sr": domega_sr,
        }
        nested = self.detector
        if nested is not None and not isinstance(nested, Detector):
            raise TypeError("detector must be a Detector")
        base = Detector() if nested is None else nested
        detector_values = {
            "theta_obs_deg": base.observation_angle_deg,
            "dtheta_obs_deg": base.polar_acceptance_deg,
            "domega_sr": base.solid_angle_sr,
        }
        conflicts = [
            name
            for name, value in supplied.items()
            if nested is not None and value is not None and value != detector_values[name]
        ]
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(
                f"conflicting nested detector and legacy flat input(s): {joined}; "
                "supply Detector values or flat theta_obs_deg/dtheta_obs_deg/domega_sr, "
                "not both"
            )
        if nested is None:
            base = replace(
                base,
                observation_angle_deg=(
                    base.observation_angle_deg if theta_obs_deg is None else theta_obs_deg
                ),
                polar_acceptance_deg=(
                    base.polar_acceptance_deg if dtheta_obs_deg is None else dtheta_obs_deg
                ),
                solid_angle_sr=base.solid_angle_sr if domega_sr is None else domega_sr,
            )
        self.detector = base


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


def sweep_crystal_params(sweep: Sweep) -> dict[str, Any]:
    """The crystallography mapping a sweep's cases are built from.

    :func:`build_cases` and the reline path (:mod:`pyrite.checkpoints.recompute`)
    must agree on this, since it is the ``cp`` argument
    :func:`_line_grid_for_energy` resolves a line grid against.
    """
    assert sweep.target is not None  # Sweep.__post_init__ always resolves one
    cp = crystal_params(sweep.target.material, sweep.n_families)
    if sweep.max_reflections is not None:
        cp["hkl_list"] = cp["hkl_list"][: sweep.max_reflections]
    return cp


def _automatic_line_grid_policy(sweep: Sweep, cp: dict, energy_keV: float) -> LineGridPolicy:
    """Resolve the automatic case-local line-grid policy for one beam energy.

    Bandwidth is closed form (:func:`kinematic_line_stop_eV`) and costs a dot
    product per reflection. Resolution is only *declared* here -- the runner
    measures it from the case's own trajectories -- so building a case starts no
    simulation, reads no checkpoint, and writes nothing to the profile or the
    catalog.
    """
    spec = CATALOG.crystal(cp["crystal"])
    magnitudes = [reciprocal_g_vector(hkl, spec.lattice)[1] for hkl in cp["hkl_list"]]
    start = line_start_eV(energy_keV)
    stop = kinematic_line_stop_eV(magnitudes, energy_keV)
    return resolve_line_grid_policy(
        start_eV=start,
        stop_eV=stop,
        bandwidth_policy=AUTOMATIC_BANDWIDTH_POLICY,
        per_call=sweep.line_grid_policy,
    )


def _line_grid_for_energy(
    sweep: Sweep, cp: dict, energy_keV: float
) -> tuple[np.ndarray, dict | None]:
    """Select this beam energy's line grid, and any automatic policy behind it.

    Precedence, highest first (issue #101):

    1. an explicit user grid -- ``EnergyBins.line``, or a per-call
       ``Sweep.line_grid_policy``;
    2. ``PYRITE_ENERGY_GRID_*`` environment policy, which turns automatic
       resolution on even where a stored row exists;
    3. the stored catalog/profile artifact -- this energy's
       ``E_grid_line_by_energy`` row, or, when no mapping is configured at all,
       the crystal's own ``E_grid``. Legacy v1 artifacts still decode and are
       still used here;
    4. built-in automatic resolution.

    A stored mapping that does not cover this beam energy is no longer an error:
    it falls through to (4). ``pyrite material energy-grid derive`` is an
    inspection and prewarming command, not a prerequisite for running a valid
    material at a supported beam energy.

    Returns ``(grid, policy_payload)``. ``policy_payload`` is ``None`` when the
    coordinates are final; otherwise ``grid`` is the coarsest admissible grid
    and the runner refines it under the returned policy, from the case's own
    trajectories.
    """
    assert sweep.detector is not None
    bins = sweep.detector.energy_bins
    override = bool(sweep.line_grid_policy) or environment_overrides_present()
    if bins.line is not None and not sweep.line_grid_policy:
        return np.asarray(bins.line, dtype=float), None
    if not override:
        if bins.line_by_energy is None:
            return np.asarray(cp["E_grid"], dtype=float), None
        stored = bins.line_by_energy.get(float(energy_keV))
        if stored is not None:
            return np.asarray(stored, dtype=float), None
    policy = _automatic_line_grid_policy(sweep, cp, float(energy_keV))
    num = resolution_num(policy.start_eV, policy.stop_eV, policy.max_spacing_eV)
    grid = np.linspace(policy.start_eV, policy.stop_eV, num)
    return grid, policy.payload()


def _case_elements(case: Case) -> tuple[str, ...]:
    """Every element the case's crystal and absorber layers radiate from."""
    compositions = [case["composition"]]
    if case.get("abs_layers"):
        compositions.extend(comp for _, _, comp in case["abs_layers"])
    return tuple(dict.fromkeys(str(row[0]) for comp in compositions for row in comp))


def _resolve_auto_bremsstrahlung(cases: list[Case]) -> list[Case]:
    """Replace ``bremsstrahlung_model="auto"`` by the source each case will use.

    A case records only the resolved choice, so identity and content keys
    never depend on what happened to be installed unless BremsLib really ran.
    """
    from ..xsgen.bremslib.tables import resolve_bremsstrahlung_model

    resolved: dict[tuple[str, ...], str] = {}
    out = []
    for case in cases:
        elements = _case_elements(case)
        if elements not in resolved:
            resolved[elements] = resolve_bremsstrahlung_model("auto", elements)
        out.append(
            replace(case, bremsstrahlung_model="bremslib")
            if resolved[elements] == "bremslib"
            else case
        )
    return out


def build_cases(
    sweep: Sweep,
    n_electrons=450,
    n_electrons_brem=100,
    coherent_emission=False,
    straggling=False,
    energy_model="frozen",
    max_dE_frac=0.0,
    inelastic_model="continuous",
    inelastic_cutoff_eV=None,
    elastic_model="elsepa",
    bremsstrahlung_model="auto",
    secondary_threshold_eV=None,
):
    """Expand a :class:`Sweep` into a list of :class:`montecarlo.Case` records (the Cartesian
    product over the swept thickness / tilt / azimuth / footprint, each
    crossed with every beam energy). ``crystal_width_mm`` and
    ``crystal_height_mm`` are full dimensions: both ``None`` recovers the
    legacy infinite slab, otherwise both must be positive (default: a finite
    5x5 mm footprint). Each case also carries the beam phase-space fields from
    ``sweep.beam`` (:class:`BeamSpec`): ``beam_fwhm_mm`` (x-plane FWHM; default a
    1 mm Gaussian spot, ``None`` recovers the legacy point source), the
    elliptical ``beam_fwhm_y_mm`` and either the legacy longitudinal
    ``bunch_length_fs`` / ``long_shape`` / ``long_offsets_fs`` keys or one
    resolved ``longitudinal_distribution`` record. Declarative policies resolve
    only after material, energy, geometry, and catalog reflection provenance are
    known; sampled offsets never enter case identity. Optional ``n_electrons`` /
    ``n_electrons_brem`` sweep grids (catalog profile settings) cross
    electron-count statistics into the product and suffix the case name with
    ``ne=<line>/<brem>``; ``None`` keeps the scalar counts passed by the caller.
    Returns the ``cases`` list; preview it with :func:`geometry_table`."""
    assert sweep.target is not None  # Sweep.__post_init__ always resolves one
    validate_inelastic_numerics(
        inelastic_model, inelastic_cutoff_eV, energy_model, secondary_threshold_eV
    )
    validate_elastic_model(elastic_model)
    validate_bremsstrahlung_model(bremsstrahlung_model)
    target = sweep.target
    cp = sweep_crystal_params(sweep)
    # line grid: fine + narrow (per-material default or detector mapping/fixed
    # binning). brem grid: coarse + wide
    # -- each case spans up to that case's beam energy because brem cuts off at
    # the particle energy.
    # A uniform E_grid_brem keeps the legacy start/spacing behavior and extends
    # to each beam energy. Scalar/nonuniform grids are explicit and stay exact.
    gdf = sweep.beam.gdf_beam()
    if gdf is not None and coherent_emission:
        raise ValueError("gpt_gdf currently supports incoherent emission only")
    energies = (
        _seq(sweep.beam.energy_keV)
        if gdf is None or sweep.beam.gdf_shape_only
        else np.array([gdf.energy_keV.max()])
    )
    _reject_relativistic_energies(energies)
    resolved_line = tuple(_line_grid_for_energy(sweep, cp, float(energy)) for energy in energies)
    line_grids = tuple(grid for grid, _ in resolved_line)
    line_policies = tuple(policy for _, policy in resolved_line)
    assert sweep.detector is not None
    bins = sweep.detector.energy_bins
    fixed_line_grid = bins.line
    if bins.brem is not None:
        brem_grid = np.asarray(bins.brem, float)
    else:
        if fixed_line_grid is not None:
            brem_start = float(np.atleast_1d(fixed_line_grid)[0])
        elif bins.line_by_energy:
            brem_start = min(float(np.atleast_1d(grid)[0]) for grid in bins.line_by_energy.values())
        elif bins.line_by_energy is None:
            brem_start = float(cp["E_grid"][0])
        else:
            # A configured-but-empty mapping: every energy resolved
            # automatically, so the continuum starts where those grids do.
            brem_start = min(float(np.atleast_1d(grid)[0]) for grid in line_grids)
        brem_grid = np.arange(brem_start, float(energies.max()) * 1e3 + 50.0, 50.0)  # type: ignore[reportArgumentType]

    assert sweep.detector is not None
    dtheta = (
        TIMEPIX3_DTHETA_OBS_DEG
        if sweep.detector.polar_acceptance_deg is None
        else sweep.detector.polar_acceptance_deg
    )
    domega = (
        TIMEPIX3_DOMEGA_SR
        if sweep.detector.solid_angle_sr is None
        else sweep.detector.solid_angle_sr
    )
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
    if gdf is not None:
        beam_case["gdf_source"] = {
            "path": str(b.gdf_path),
            "time_s": gdf.time_s,
            "tolerance_s": b.gdf_time_tolerance_s,
            "normalization": b.gdf_normalization,
            "z_origin_m": b.gdf_z_origin_m,
            "sha256": gdf.sha256,
            **({"shape_only": True} if b.gdf_shape_only else {}),
            **(
                {
                    "screen_position_m": gdf.screen_position_m,
                    "screen_tolerance_m": b.gdf_screen_tolerance_m,
                }
                if gdf.screen_position_m is not None
                else {}
            ),
        }
    if b.bunch_charge_pc != 1.0 or b.rep_rate_hz != 5000.0:
        beam_case["bunch_charge_pc"] = float(b.bunch_charge_pc)
        beam_case["rep_rate_hz"] = float(b.rep_rate_hz)
    if gdf is not None and b.gdf_normalization == "gdf_charge":
        assert gdf.absolute_charge_c is not None
        beam_case["bunch_charge_pc"] = gdf.absolute_charge_c * 1e12
        beam_case["rep_rate_hz"] = b.gdf_repetition_rate_hz
    if fwhm_y != fwhm_x:
        beam_case["beam_fwhm_y_mm"] = fwhm_y
    if b.energy_spread_frac is not None:
        beam_case["energy_spread_frac"] = float(b.energy_spread_frac)
    if b.longitudinal is not None and (
        b.bunch_length_fs is not None or b.long_offsets_fs is not None or b.long_shape != "gaussian"
    ):
        raise ValueError(
            "longitudinal policy is incompatible with legacy bunch_length_fs, "
            "long_shape, and long_offsets_fs fields"
        )
    if b.transverse is not None and (
        b.transverse_fwhm_x_mm is not None
        or b.transverse_fwhm_y_mm is not None
        or b.divergence_mrad is not None
    ):
        # Spot FWHM, divergence and emittance are three numbers for two
        # independent moments plus a correlation. Resolving that by precedence
        # would hand back a plausible beam that is not the requested one, so it
        # is an error. Note the FWHM fields DEFAULT to a 1 mm spot: a transverse
        # policy has to clear them explicitly (BeamSpec.with_transverse does).
        raise ValueError(
            "transverse policy is incompatible with the legacy transverse_fwhm_x_mm, "
            "transverse_fwhm_y_mm, and divergence_mrad fields; clear them "
            "(they default to a 1 mm spot) or use BeamSpec.with_transverse"
        )
    if b.bunch_length_fs is not None or b.long_offsets_fs is not None:
        beam_case["long_shape"] = b.long_shape
        if b.bunch_length_fs is not None:
            beam_case["bunch_length_fs"] = float(b.bunch_length_fs)
        if b.long_offsets_fs is not None:
            beam_case["long_offsets_fs"] = tuple(float(x) for x in b.long_offsets_fs)
    # Case names are persisted: they key the results dict and feed `_case_key`,
    # which `pyrite checkpoint gc` compares against. So they are built from the
    # stable catalog key, never from the display label -- a label is display
    # metadata and editing one must not invalidate stored checkpoints.
    material_spec = CATALOG.materials.get(target.material)
    name_stem = material_spec.key if material_spec is not None else target.material
    # crystal mosaicity (analytic, optional): None unless the run enables it AND the
    # crystal has a mosaic_fwhm_deg (or the Sweep overrides it). None -> perfect
    # crystal, so store_result adds no mosaic term (exact no-op).
    mosaic_deg = None
    if target.mosaic:
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
    if isinstance(brem_case_grid, tuple) and sweep.material is not None:
        # A uniform band declares a `start`, but the energy below which the
        # continuum and photon-escape models carry no validity is a property of
        # the *medium*, not of the profile that declared the band. Resolving it
        # here -- the first place the band and the material meet -- is what lets
        # a profile-level default stay at 0.0 and still never produce a node
        # outside the modelled band, and what keeps two profiles sharing the
        # same material agreeing on the grid, which a per-profile stored start
        # could not do. A declared start ABOVE the floor is an ordinary
        # bandwidth choice and is kept, matching
        # ``energy_grid.floor.geometric_continuum_grid``. Nonuniform grids are
        # passed through: their builders resolve their own floor, and a sweep
        # with no catalog material names no medium to take a floor from.
        start, stop, step = brem_case_grid
        brem_case_grid = (max(start, floored_lattice_start_eV(sweep.material, step)), stop, step)

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

    # Geometry is owned end to end by the target: it validated itself at Sweep
    # construction, validate_against carries the one target x detector rule, and
    # lower() is the whole geometry product -- so nothing below branches on
    # geometry, it only spreads what the target produced.
    target.validate_against(sweep.detector)
    geometries = target.lower(
        cp, name_stem=name_stem, beam_uvw=beam_uvw, n_families=sweep.n_families
    )

    if gdf is not None and any(
        g.case_keys().get("groove_spacing_ang") is not None for g in geometries
    ):
        raise ValueError("gpt_gdf currently requires a flat entrance face")
    cases = []
    for i_c, geometry in enumerate(geometries):
        name = geometry.name
        for i_e, E0 in enumerate(energies):
            line_case_grid = encode_energy_grid(line_grids[i_e])
            resolved_transverse = None
            if b.transverse is not None:
                resolved_transverse = asdict(
                    resolve_transverse_distribution(b.transverse, energy_keV=float(E0))
                )
            resolved_longitudinal = None
            if b.longitudinal is not None:
                crystal_spec = CATALOG.crystal(cp["crystal"])
                resolved_longitudinal = asdict(
                    resolve_longitudinal_distribution(
                        b.longitudinal,
                        material=target.material,
                        crystal=cp["crystal"],
                        lattice=crystal_spec.lattice,
                        hkl_list=cp["hkl_list"],
                        surface_hkl=surface_hkl,
                        beam_uvw=beam_uvw,
                        energy_keV=float(E0),
                        theta_obs_deg=float(sweep.detector.observation_angle_deg),
                        tilt_deg=geometry.tilt_deg,
                    )
                )
            for i_n, (ne_line, ne_brem) in enumerate(ne_pairs):
                case_name = f"{name} ne={ne_line}/{ne_brem}" if explicit_ne else name
                cases.append(
                    Case(
                        name=case_name,
                        crystal=cp["crystal"],
                        composition=cp["composition"],
                        hkl_list=cp["hkl_list"],
                        B_ang2=cp["B_ang2"],
                        E0_keV=float(E0),
                        # thickness, footprint, tilts, stack layers, and the
                        # groove spacing when there is one (absent otherwise)
                        **geometry.case_keys(),
                        **beam_case,
                        **(
                            {"longitudinal_distribution": resolved_longitudinal}
                            if resolved_longitudinal is not None
                            else {}
                        ),
                        **(
                            {"transverse_distribution": resolved_transverse}
                            if resolved_transverse is not None
                            else {}
                        ),
                        E_grid=line_case_grid,  # legacy key (== line grid)
                        E_grid_line=line_case_grid,
                        # Automatic resolution only. Absent keeps every
                        # explicit/stored-grid case payload bit-for-bit, so
                        # existing checkpoints stay addressable.
                        **(
                            {"line_grid_policy": line_policies[i_e]}
                            if line_policies[i_e] is not None
                            else {}
                        ),
                        # Divergence-only (#116): node sampling keeps its payload.
                        **(
                            {"line_quadrature": "bin-mean"}
                            if line_quadrature_from_payload(line_policies[i_e]) == "bin-mean"
                            else {}
                        ),
                        E_grid_brem=(
                            (
                                brem_case_grid[0],
                                float(E0) * 1e3 + brem_case_grid[2],
                                brem_case_grid[2],
                            )
                            if isinstance(brem_case_grid, tuple)
                            else brem_case_grid
                        ),
                        theta_obs_rad=np.deg2rad(sweep.detector.observation_angle_deg),
                        # coherent segment sum: divergence-only key (absent -> the
                        # incoherent default, bit-for-bit case payload).
                        **({"coherent_emission": True} if coherent_emission else {}),
                        **({"straggling": True} if straggling else {}),
                        **({"energy_model": "midpoint"} if energy_model == "midpoint" else {}),
                        **({"max_dE_frac": float(max_dE_frac)} if max_dE_frac > 0.0 else {}),
                        # Opt-in shell soft/hard inelastic mode: divergence-only
                        # keys, so continuous-stopping case payloads (and their
                        # content keys) stay bit-for-bit.
                        **(
                            {
                                "inelastic_model": inelastic_model,
                                "inelastic_cutoff_eV": float(cast(float, inelastic_cutoff_eV)),
                            }
                            if inelastic_model != "continuous"
                            else {}
                        ),
                        # Opt-in secondary transport (#94): divergence-only.
                        **(
                            {"secondary_threshold_eV": float(secondary_threshold_eV)}
                            if secondary_threshold_eV is not None
                            else {}
                        ),
                        # ELSEPA elastic model (the default): divergence-only, so
                        # an explicit "mott" case keeps its historical payload.
                        **({"elastic_model": "elsepa"} if elastic_model == "elsepa" else {}),
                        # Opt-in BremsLib continuum: divergence-only, like the above.
                        **(
                            {"bremsstrahlung_model": "bremslib"}
                            if bremsstrahlung_model == "bremslib"
                            else {}
                        ),
                        beam_uvw=beam_uvw,
                        surface_hkl=surface_hkl,
                        mosaic_fwhm_rad=mosaic_analytic_rad,  # analytic term (None if route="mc")
                        mosaic_mc_fwhm_rad=mosaic_mc_rad,  # exact MC route (None if route="analytic")
                        mosaic_mc_nodes=sweep.mosaic_nodes,
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
    if bremsstrahlung_model == "auto":
        cases = _resolve_auto_bremsstrahlung(cases)
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


def _dEds_magnitude_keV_per_ang(composition, E_keV):
    """Transport stopping-power magnitude [keV/Angstrom].

    Delegates to ``montecarlo.transport.spliced_stopping_keV_per_ang`` rather
    than re-stating the constants, so the cost proxy's CSDA range cannot drift
    from the transport model whose runtime it predicts -- the copy that used to
    live here went stale the moment the stopping model changed. ``sweep.py``
    already imports ``montecarlo.case``, so this adds no import cost.
    Additive over elements with number densities ``n_i`` [1/Angstrom^3];
    ``E_keV`` may be a scalar or an array."""
    # magnitude only (transport uses the negative)
    return -spliced_stopping_keV_per_ang(composition, E_keV)


def _csda_profile(composition, E0_keV, e_cut_keV=_COST_E_CUT_KEV, n_quad=64):
    """Cumulative continuous-slowing-down range [Angstrom] vs energy.

    Returns ``(E, cum)`` on ``n_quad`` points from ``e_cut_keV`` to ``E0_keV``,
    where ``cum[i]`` is the CSDA path length from ``e_cut`` up to ``E[i]``
    (trapezoid integral of ``1/|dE/ds|``). ``cum[-1]`` is the full range at
    ``E0``. Monotonic in ``E0``; used both for the range and to invert it (how
    far a slab lets an electron travel before it enters the next layer)."""
    E = np.linspace(e_cut_keV, E0_keV, n_quad)
    inv_dEds = 1.0 / _dEds_magnitude_keV_per_ang(composition, E)
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
