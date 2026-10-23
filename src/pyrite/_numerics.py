"""Low-level typed calculation-numerics policy and profile validation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal, cast

import numpy as np

from ._energy_bins import EnergyBins
from ._precision import Precision

MosaicRoute = Literal["analytic", "mc"]
EmissionMode = Literal["incoherent", "coherent"]
EmissionSelection = (
    EmissionMode
    | tuple[EmissionMode, ...]
    | list[EmissionMode]
    | set[EmissionMode]
    | frozenset[EmissionMode]
)
LegacyEmissionPolicy = Literal["incoherent", "coherent", "both"]


def _emission_selection(value: object) -> tuple[EmissionMode, ...]:
    """Normalize API selections or stable legacy policies to ordered modes."""
    if isinstance(value, str):
        if value == "both":
            return ("incoherent", "coherent")
        modes = (value,)
    elif isinstance(value, (tuple, list, set, frozenset)):
        modes = tuple(value)
    else:
        raise TypeError(
            "emission must be a mode string or a tuple, list, set, or frozenset of modes"
        )
    if not modes or any(
        not isinstance(mode, str) or mode not in {"incoherent", "coherent"} for mode in modes
    ):
        raise ValueError("emission must contain one or both of 'incoherent' and 'coherent'")
    return tuple(mode for mode in ("incoherent", "coherent") if mode in modes)


SAMPLING_KEYS = ("n_electrons", "n_electrons_brem")
CONVERGENCE_KEYS = ("n_families", "max_reflections", "mosaic_nodes", "mosaic_route")
CHANNELING_KEYS = (
    "channeling_model",
    "channeling_plane",
    "channeling_axial",
    "channeling_basis_size",
    "channeling_bloch_cells",
    "channeling_dz_ang",
    "channeling_zone_max_ang",
)
TRANSPORT_KEYS = (
    *CHANNELING_KEYS,
    "straggling",
    "energy_model",
    "max_dE_frac",
    "inelastic_model",
    "inelastic_cutoff_eV",
    "secondary_threshold_eV",
    "elastic_model",
    "bremsstrahlung_model",
    "radiative_model",
    "radiative_cutoff_eV",
    "pair_production_model",
    "positron_transport",
    "atomic_electron_deflection",
    "transition_radiation",
    "transition_lattice_origin_ang",
    "shell_ionization_model",
)
#: Collision-loss schemes a run may select: ``"auto"`` (the default) plus the
#: ``simulate_trajectories`` schemes of
#: ``montecarlo.transport.hard_inelastic.INELASTIC_MODELS`` (a test keeps the
#: two in step) without importing the transport package here. ``"auto"``
#: resolves per case to ``"shell-soft-hard"`` where the shell model covers
#: every layer and to ``"continuous"`` otherwise; a case records only the result.
INELASTIC_MODELS = ("auto", "continuous", "shell-soft-hard")
DEFAULT_INELASTIC_CUTOFF_EV = 50.0
#: Elastic models a case may select. ``"elsepa"`` (the default, issue #89)
#: samples resolved ELSEPA tables; ``"mott"`` is the historical model.
ELASTIC_MODELS = ("mott", "elsepa")
#: Continuum bremsstrahlung sources a run may select (issue #86): the packaged
#: EEDL evaluation, the released BremsLib tables, or ``"auto"`` (the default),
#: which is BremsLib when every layer element's table is installed and EEDL,
#: with a warning, otherwise. A case records only the resolved choice.
BREMSSTRAHLUNG_MODELS = ("auto", "eedl", "bremslib")
#: Opt-in pair conversion of coupled hard photons (issue #275); mirrors
#: ``montecarlo.transport.pair_production.PAIR_PRODUCTION_MODELS`` (a test
#: keeps the two in step) without importing the transport package here.
PAIR_PRODUCTION_MODELS = ("penelope-2024",)
#: Atomic-electron angular deflection (issue #317): ``"kawrakow"`` (the
#: default) folds it into the elastic rate as ``Z(Z + xi)``; ``"none"`` keeps
#: elastic-only deflection. Mirrors
#: ``montecarlo.transport.atomic_electrons.ATOMIC_ELECTRON_DEFLECTION_MODELS``
#: plus the ``"none"`` opt-out (a test keeps them in step).
ATOMIC_ELECTRON_DEFLECTION_MODELS = ("kawrakow", "none")
#: Characteristic primary-vacancy cross sections: EEDL (default) or the opt-in
#: Bote--Salvat K/L/M fit (#92). Kept in step with
#: ``montecarlo.spectrum.characteristic._IONIZATION_MODELS`` by a test.
SHELL_IONIZATION_MODELS = ("eedl", "bote-salvat")
#: The default resolves coupling when BremsLib tables are available.
RADIATIVE_MODELS = ("auto", "uncoupled", "bremslib-soft-hard")
DEFAULT_RADIATIVE_CUTOFF_EV = 1000.0
#: Quantum channeling-zone selectors (ADR-0018, issue #506). ``"off"`` (the
#: default) leaves every case, identity and RNG stream unchanged; ``"planar"``
#: runs the planar density-matrix entrance zone with a one-way handoff.
CHANNELING_MODELS = ("off", "planar", "axial")
#: Transition-radiation component selectors (#103). ``"off"`` (the default)
#: leaves every case and identity unchanged; ``"history"`` sums each primary
#: history's boundary crossings coherently; ``"bunch"`` also keeps the
#: cross-history (CTR) terms with the physical bunch population.
TRANSITION_RADIATION_MODES = ("off", "history", "bunch", "two-beam")
PROFILE_NUMERICS_KEYS = (*SAMPLING_KEYS, *CONVERGENCE_KEYS, *TRANSPORT_KEYS)


@dataclass(frozen=True)
class CxrReflections:
    """Control CXR reflection-family truncation and mosaic quadrature.

    Parameters
    ----------
    n_families
        Number of dominant reflection families selected automatically.
    max_reflections
        Optional cap on individual reflections after family expansion.
    mosaic_nodes
        Number of quadrature nodes used for mosaic averaging.
    mosaic_route
        Mosaic integration strategy, ``"analytic"`` or ``"mc"``.
    """

    n_families: int = 4
    max_reflections: int | None = None
    mosaic_nodes: int = 5
    mosaic_route: MosaicRoute = "analytic"

    def __post_init__(self) -> None:
        if type(self.n_families) is not int or self.n_families <= 0:
            raise ValueError("n_families must be a positive integer")
        if self.max_reflections is not None and (
            type(self.max_reflections) is not int or self.max_reflections <= 0
        ):
            raise ValueError("max_reflections must be a positive integer or None")
        if type(self.mosaic_nodes) is not int or self.mosaic_nodes <= 0:
            raise ValueError("mosaic_nodes must be a positive integer")
        if self.mosaic_route not in {"analytic", "mc"}:
            raise ValueError("mosaic_route must be 'analytic' or 'mc'")


@dataclass(frozen=True)
class Physics:
    """Choose emission and particle-interaction models, independently of sampling.

    ``emission`` selects one or both of the incoherent and coherent CXR modes; ``brem_source``
    selects MC, external, or no continuum (the in-memory API has no external
    spectrum and returns zeros for the last two). Transport model defaults and
    cutoff units are identical to the profile schema: cutoffs are in eV.
    Integration compatibility is checked when combined with :class:`Numerics`.

    Parameters
    ----------
    emission
        A mode string (``"incoherent"`` or ``"coherent"``), or a nonempty
        tuple, list, set, or frozenset of modes. Stored as a deduplicated tuple
        in incoherent-then-coherent order. Coherent scoring also exposes its
        incoherent companion from the same transport. The input alias
        Use ``("incoherent", "coherent")`` to select both modes.
    brem_source
        ``"mc"``, ``"external"``, or ``None`` continuum policy. ``None``
        disables continuum scoring and is stored as ``"none"`` for case and
        checkpoint compatibility; the string ``"none"`` is also accepted.
    straggling
        Enable stochastic per-flight energy-loss straggling.
    inelastic_model, inelastic_cutoff_eV
        ``"auto"`` (default) uses the ``"shell-soft-hard"`` mixed scheme with
        explicit hard inelastic collisions for each case whose layers all have
        shell data, when ungrooved and ``energy_model="midpoint"``, and
        otherwise warns and keeps ``"continuous"`` stopping. ``W_c`` is the
        energy-loss cutoff in eV (default 50 under ``"auto"``; required by
        explicit ``"shell-soft-hard"``, which also requires
        ``energy_model="midpoint"``; invalid with ``"continuous"``).
    secondary_threshold_eV
        Opt-in transport of hard-collision secondaries (shell-soft-hard only):
        one threshold in eV is both production cut and secondary tracking
        cutoff. ``None`` (default) transports no secondaries.
    elastic_model
        ``"elsepa"`` (default) samples full ELSEPA differential cross
        sections from the released tables (``pyrite tables fetch elsepa``);
        ``"mott"`` keeps the historical screened-Rutherford angles calibrated
        to NIST Mott transport cross sections.
    bremsstrahlung_model
        ``"auto"`` (default) uses the released BremsLib tables with their
        angular model (``pyrite tables fetch bremslib``) when every layer
        element's table is installed, and otherwise warns and falls back to
        EEDL. ``"bremslib"`` requires the tables; ``"eedl"`` selects the
        fetched EEDL continuum with an isotropic photon angle.
    radiative_model, radiative_cutoff_eV
        ``"auto"`` (default) couples when BremsLib resolves for every layer
        and otherwise uses uncoupled EEDL scoring. ``"uncoupled"`` always
        scores the continuum from electron track length. ``"bremslib-soft-hard"`` removes
        radiative loss during transport: soft BremsLib loss below the photon
        cutoff ``k_c`` in eV and sampled hard photons above it. The default
        cutoff is 1000 eV. Coupling requires ``energy_model="midpoint"``;
        ``k_c`` must not exceed the electron cutoffs.
    pair_production_model
        Opt-in pair conversion of coupled hard photons above ``2 m_e c^2``
        (``"penelope-2024"``); requires ``secondary_threshold_eV`` and coupled
        radiative transport, and applies only to cases that resolve to it.
        Positrons are recorded, not transported, unless
        ``positron_transport``. ``None`` (default) converts nothing.
    positron_transport
        Opt-in transport of pair positrons above the secondary threshold
        (issue #276; Bhabha, positron SBETHE/ELSEPA, ``F_p``-scaled BremsLib);
        requires ``pair_production_model``. Positrons annihilate in flight and
        at rest (#295). ``False`` (default) records them only.
    atomic_electron_deflection
        ``"kawrakow"`` (default) adds angular deflection by atomic electrons
        to the elastic rate, ``Z^2 -> Z(Z + xi)``: ``xi = 1`` under continuous
        stopping, and the Moller share above ``W_c`` removed under shell
        soft/hard transport. A deprecated-``"mott"`` case that resolves to
        shell soft/hard keeps elastic-only deflection, with a warning.
        ``"none"`` reproduces elastic-only transport bit for bit.
    channeling_model
        ``"off"`` (default) treats the crystal as amorphous for transport.
        ``"planar"`` (ADR-0018) owns each electron from crystal entry in a
        planar quantum channeling zone until it dechannels, then hands it to
        native transport. Requires ``channeling_plane``, midpoint integration,
        ``radiative_model="uncoupled"``, incoherent emission and no secondary
        transport; see :class:`Numerics` for its zone numerics.
    channeling_plane
        Miller indices ``(h, k, l)`` of the channeling plane family; required
        by, and only accepted with, ``channeling_model="planar"``.
    transition_radiation
        ``"off"`` (default), ``"history"`` or ``"bunch"``: score planar-stack
        transition radiation from the transported boundary crossings as the
        separate ``spec_transition`` component (#103), per history or with
        coherent bunch (CTR) terms. Requires ``energy_model="midpoint"``;
        excludes channeling.
        ``"two-beam"`` opts into experimental transported-history Maxwell
        PXR/DTR/CBS scoring (CPU, one perfect crystal layer and reflection).
        It replaces the line scorer and retains signed TR/line interference.
    transition_lattice_origin_ang
        Lattice origin relative to the entrance face, in Å, for ``"two-beam"``;
        omitted means (0, 0, 0). It changes the surface-termination phase.
    shell_ionization_model
        ``"eedl"`` (default) characteristic primary-vacancy cross sections, or
        the opt-in ``"bote-salvat"`` K/L/M fit (#92; ``pyrite tables fetch
        bote-salvat``), which keeps EEDL for higher shells and the EADL cascade.
    """

    emission: EmissionSelection = "incoherent"
    brem_source: Literal["mc", "external", "none"] | None = "mc"
    straggling: bool = False
    inelastic_model: Literal["auto", "continuous", "shell-soft-hard"] = "auto"
    inelastic_cutoff_eV: float | None = None
    secondary_threshold_eV: float | None = None
    elastic_model: Literal["mott", "elsepa"] = "elsepa"
    bremsstrahlung_model: Literal["auto", "eedl", "bremslib"] = "auto"
    radiative_model: Literal["auto", "uncoupled", "bremslib-soft-hard"] = "auto"
    radiative_cutoff_eV: float | None = None
    pair_production_model: Literal["penelope-2024"] | None = None
    positron_transport: bool = False
    atomic_electron_deflection: Literal["kawrakow", "none"] = "kawrakow"
    channeling_model: Literal["off", "planar", "axial"] = "off"
    channeling_plane: tuple[int, int, int] | None = None
    channeling_axial: Mapping[str, object] | None = None
    transition_radiation: Literal["off", "history", "bunch", "two-beam"] = "off"
    transition_lattice_origin_ang: tuple[float, float, float] | None = None
    shell_ionization_model: Literal["eedl", "bote-salvat"] = "eedl"

    def __post_init__(self) -> None:
        if self.emission == "both":
            raise ValueError(
                'emission="both" was removed in 0.15.0; use emission=("incoherent", "coherent")'
            )
        object.__setattr__(self, "emission", _emission_selection(self.emission))
        if self.transition_lattice_origin_ang is not None:
            object.__setattr__(
                self, "transition_lattice_origin_ang", tuple(self.transition_lattice_origin_ang)
            )
        if self.channeling_plane is not None:
            object.__setattr__(self, "channeling_plane", _miller(self.channeling_plane))
        if self.channeling_axial is not None:
            from .campaign.axial_spec import resolve_axial_spec

            object.__setattr__(self, "channeling_axial", resolve_axial_spec(self.channeling_axial))
        if self.brem_source is None:
            object.__setattr__(self, "brem_source", "none")
        if self.brem_source not in {"mc", "external", "none"}:
            raise ValueError("brem_source must be 'mc', 'external', None, or 'none'")
        self.validate_transport("midpoint")

    @property
    def emission_policy(self) -> LegacyEmissionPolicy:
        """Stable tri-state policy for CLI, catalog, checkpoint, and observation boundaries."""
        modes = cast(tuple[EmissionMode, ...], self.emission)
        return "both" if len(modes) == 2 else modes[0]

    @property
    def coherent_emission(self) -> bool:
        """Whether the selection enables coherent scoring alongside its companion."""
        return "coherent" in self.emission

    def validate_transport(self, energy_model: str) -> None:
        """Check model values and their compatibility with the integration scheme."""
        if not isinstance(self.straggling, bool):
            raise TypeError("straggling must be a bool")
        validate_inelastic_numerics(
            self.inelastic_model,
            self.inelastic_cutoff_eV,
            energy_model,
            self.secondary_threshold_eV,
        )
        validate_elastic_model(self.elastic_model)
        validate_bremsstrahlung_model(self.bremsstrahlung_model)
        validate_radiative_numerics(
            self.radiative_model,
            self.radiative_cutoff_eV,
            energy_model,
            self.straggling,
            self.bremsstrahlung_model,
        )
        validate_pair_production_numerics(
            self.pair_production_model,
            self.secondary_threshold_eV,
            self.radiative_model,
            self.bremsstrahlung_model,
            self.positron_transport,
        )
        validate_atomic_electron_deflection(self.atomic_electron_deflection)
        validate_shell_ionization_model(self.shell_ionization_model)
        validate_channeling_physics(self, energy_model)
        validate_transition_physics(self, energy_model)


PHYSICS_FIELDS = tuple(f.name for f in fields(Physics))


@dataclass(frozen=True)
class Numerics:
    """Control sampling, integration, CXR truncation, and execution.

    ``cxr_trials`` and ``brem_trials`` count macro-electron trials. Optional
    ``cxr_chunk`` and ``brem_chunk`` bound spectrum-kernel batches.
    ``energy_model`` is frozen left-endpoint or midpoint integration;
    ``max_dE_frac`` limits predicted fractional mean loss per row (zero disables
    substeps). ``transport_core`` chooses auto, lockstep, per-electron, or CUDA.
    ``energy_bins`` holds sampled true-photon energies; ``None`` uses catalog
    defaults.
    ``cxr_reflections`` controls reflection truncation and mosaic quadrature.
    ``precision`` requests adaptive sampling instead of the two fixed counts.
    The ``channeling_*`` fields are the planar channeling zone's numerics, used
    only with ``Physics(channeling_model="planar")``: ``channeling_basis_size``
    is the plane-wave band ``|h| <= H`` of the transverse basis,
    ``channeling_bloch_cells`` the Bloch-grid supercell size,
    ``channeling_dz_ang`` the density-matrix depth step [angstrom] and
    ``channeling_zone_max_ang`` an optional zone-length cap [angstrom] beyond
    which still-channeled electrons are handed off (``None``: the first
    layer's thickness).
    Select ``PYRITE_MC_BACKEND`` before import; models belong to :class:`Physics`.
    """

    cxr_trials: int = 450
    brem_trials: int = 100
    cxr_chunk: int | None = None
    brem_chunk: int | None = None
    transport_core: str = "auto"
    energy_model: Literal["frozen", "midpoint"] = field(default="midpoint", kw_only=True)
    max_dE_frac: float = field(default=0.0, kw_only=True)
    cxr_reflections: CxrReflections = field(default_factory=CxrReflections, kw_only=True)
    precision: Precision | None = field(default=None, kw_only=True)
    energy_bins: EnergyBins | None = field(default=None, kw_only=True)
    channeling_basis_size: int = field(default=16, kw_only=True)
    channeling_bloch_cells: int = field(default=16, kw_only=True)
    channeling_dz_ang: float = field(default=250.0, kw_only=True)
    channeling_zone_max_ang: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        validate_channeling_numerics(
            self.channeling_basis_size,
            self.channeling_bloch_cells,
            self.channeling_dz_ang,
            self.channeling_zone_max_ang,
        )
        if self.precision is not None:
            if not isinstance(self.precision, Precision):
                raise TypeError("precision must be a Precision or None")
            if self.transport_core == "lockstep":
                raise ValueError("adaptive precision requires the per-electron or CUDA core")
        for name in ("cxr_trials", "brem_trials"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cxr_chunk", "brem_chunk"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(self.cxr_reflections, CxrReflections):
            raise TypeError("cxr_reflections must be a CxrReflections")
        if self.energy_bins is not None and not isinstance(self.energy_bins, EnergyBins):
            raise TypeError("energy_bins must be an EnergyBins or None")
        if self.energy_model not in {"frozen", "midpoint"}:
            raise ValueError("energy_model must be 'frozen' or 'midpoint'")
        if (
            isinstance(self.max_dE_frac, bool)
            or not isinstance(self.max_dE_frac, (int, float))
            or not np.isfinite(self.max_dE_frac)
            or self.max_dE_frac < 0.0
        ):
            raise ValueError("max_dE_frac must be finite and non-negative")
        if self.max_dE_frac > 0.0 and self.energy_model != "midpoint":
            raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")


def resolve_physics(physics: Physics | None, numerics: Numerics) -> Physics:
    """Resolve model defaults and validate transport integration."""
    if physics is not None and not isinstance(physics, Physics):
        raise TypeError("physics must be a Physics or None")
    resolved = Physics() if physics is None else physics
    resolved.validate_transport(numerics.energy_model)
    return resolved


def _miller(value: object) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("channeling_plane must be three integer Miller indices")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError("channeling_plane must be three integer Miller indices")
    hkl = (int(value[0]), int(value[1]), int(value[2]))
    if hkl == (0, 0, 0):
        raise ValueError("channeling_plane must be nonzero")
    return hkl


def validate_channeling_physics(physics: "Physics", energy_model: str) -> None:
    """Validate the channeling selector and its run-level compatibility (#506).

    The zone is a one-way handoff into native transport; combinations whose
    support would be non-trivial are refused before any work (ADR-0017
    tier-3 rule). Per-case gates (material data, bound states, energy, axis
    proximity, geometry) run in ``campaign.channeling_cases``.
    """
    model = physics.channeling_model
    if model not in CHANNELING_MODELS:
        raise ValueError(f"channeling_model must be one of {', '.join(CHANNELING_MODELS)}")
    if model == "off":
        if physics.channeling_plane is not None:
            raise ValueError("channeling_plane requires channeling_model='planar'")
        if physics.channeling_axial is not None:
            raise ValueError("channeling_axial requires a matching channeling_model='axial'")
        return
    if model == "axial":
        if physics.channeling_plane is not None:
            raise ValueError("channeling_plane is planar-only; use channeling_axial.axis_uvw")
        if physics.channeling_axial is None:
            raise ValueError("channeling_model='axial' requires channeling_axial")
        if energy_model != "midpoint":
            raise ValueError("channeling_model='axial' requires energy_model='midpoint'")
        if physics.radiative_model != "uncoupled":
            raise ValueError("axial channeling requires radiative_model='uncoupled'")
        if physics.secondary_threshold_eV is not None:
            raise ValueError("axial channeling cannot transport secondaries")
        if "coherent" in cast(tuple[str, ...], physics.emission):
            raise ValueError("axial channeling currently requires incoherent emission")
        return
    if physics.channeling_axial is not None:
        raise ValueError("channeling_axial requires channeling_model='axial'")
    if physics.channeling_plane is None:
        raise ValueError("channeling_model='planar' requires channeling_plane")
    if energy_model != "midpoint":
        raise ValueError("channeling_model='planar' requires energy_model='midpoint'")
    if physics.radiative_model != "uncoupled":
        raise ValueError(
            "channeling_model='planar' requires radiative_model='uncoupled': coupled "
            "radiative transport inside the zone is unsupported"
        )
    if physics.secondary_threshold_eV is not None:
        raise ValueError("channeling_model='planar' does not support secondary transport")
    if "coherent" in cast(tuple[str, ...], physics.emission):
        raise ValueError("channeling_model='planar' supports incoherent emission only")


def validate_transition_physics(physics: "Physics", energy_model: str) -> None:
    """Validate the transition-radiation selector against the run (#103).

    Boundary crossings come from midpoint row events; channeling owns the
    entrance zone without those events. Per-case gates (grooves, adaptive
    precision, temporal profiles) are ``Case`` validation.
    """
    mode = physics.transition_radiation
    origin = physics.transition_lattice_origin_ang
    if origin is not None:
        if mode != "two-beam":
            raise ValueError(
                "transition_lattice_origin_ang requires transition_radiation='two-beam'"
            )
        if len(origin) != 3 or not all(np.isfinite(x) for x in origin):
            raise ValueError("transition_lattice_origin_ang must contain three finite coordinates")
    if mode not in TRANSITION_RADIATION_MODES:
        raise ValueError(
            f"transition_radiation must be one of {', '.join(TRANSITION_RADIATION_MODES)}"
        )
    if mode == "off":
        return
    if energy_model != "midpoint":
        raise ValueError("transition_radiation requires energy_model='midpoint'")
    if physics.channeling_model != "off":
        raise ValueError(
            f"transition_radiation does not support channeling_model={physics.channeling_model!r}"
        )


def validate_channeling_numerics(
    basis_size: object, bloch_cells: object, dz_ang: object, zone_max_ang: object
) -> None:
    """Validate the planar channeling zone's numerical controls."""
    for name, value in (
        ("channeling_basis_size", basis_size),
        ("channeling_bloch_cells", bloch_cells),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("channeling_dz_ang", dz_ang), ("channeling_zone_max_ang", zone_max_ang)):
        if value is None and name == "channeling_zone_max_ang":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")


def validate_atomic_electron_deflection(model: object) -> None:
    """Validate a run-level atomic-electron deflection model name."""
    if model not in ATOMIC_ELECTRON_DEFLECTION_MODELS:
        raise ValueError(
            "atomic_electron_deflection must be one of "
            + ", ".join(ATOMIC_ELECTRON_DEFLECTION_MODELS)
        )


def validate_shell_ionization_model(model: object) -> None:
    """Validate a run-level characteristic shell-ionization model name."""
    if model not in SHELL_IONIZATION_MODELS:
        raise ValueError(
            "shell_ionization_model must be one of " + ", ".join(SHELL_IONIZATION_MODELS)
        )


def validate_pair_production_numerics(
    model: object,
    secondary_threshold_eV: object,
    radiative_model: object,
    bremsstrahlung_model: object,
    positron_transport: object = False,
) -> None:
    """Validate opt-in pair conversion: it needs secondaries and coupled photons.

    ``positron_transport`` (#276) needs pair conversion.
    """
    if not isinstance(positron_transport, bool):
        raise TypeError("positron_transport must be a bool")
    if positron_transport and model is None:
        raise ValueError("positron_transport requires pair_production_model")
    if model is None:
        return
    if model not in PAIR_PRODUCTION_MODELS:
        raise ValueError(
            f"pair_production_model must be one of {', '.join(PAIR_PRODUCTION_MODELS)} or None"
        )
    if secondary_threshold_eV is None:
        raise ValueError("pair_production_model requires secondary_threshold_eV")
    if radiative_model == "uncoupled" or bremsstrahlung_model == "eedl":
        raise ValueError("pair_production_model requires coupled BremsLib radiative transport")


def validate_elastic_model(model: object) -> None:
    """Validate a case-level elastic scattering model name."""
    if model not in ELASTIC_MODELS:
        raise ValueError(f"elastic_model must be one of {', '.join(ELASTIC_MODELS)}")


def validate_bremsstrahlung_model(model: object) -> None:
    """Validate a case-level continuum bremsstrahlung source name."""
    if model not in BREMSSTRAHLUNG_MODELS:
        raise ValueError(f"bremsstrahlung_model must be one of {', '.join(BREMSSTRAHLUNG_MODELS)}")


def validate_radiative_numerics(
    model: object,
    cutoff_eV: object,
    energy_model: object,
    straggling: object,
    bremsstrahlung_model: object,
) -> None:
    """Validate the coupled radiative mode's run-level settings.

    Geometry (no grooves) and the cutoff's bound by the electron cutoffs are
    checked per case, where both are known.
    """
    if model not in RADIATIVE_MODELS:
        raise ValueError(f"radiative_model must be one of {', '.join(RADIATIVE_MODELS)}")
    if model == "uncoupled":
        if cutoff_eV is not None:
            raise ValueError("radiative_cutoff_eV requires radiative_model='bremslib-soft-hard'")
        return
    if cutoff_eV is None:
        cutoff_eV = DEFAULT_RADIATIVE_CUTOFF_EV
    if (
        isinstance(cutoff_eV, bool)
        or not isinstance(cutoff_eV, (int, float))
        or not np.isfinite(cutoff_eV)
        or cutoff_eV <= 0.0
    ):
        raise ValueError(
            "radiative_model='bremslib-soft-hard' requires a finite positive radiative_cutoff_eV"
        )
    if energy_model != "midpoint":
        raise ValueError("radiative_model='bremslib-soft-hard' requires energy_model='midpoint'")
    if model == "bremslib-soft-hard" and bremsstrahlung_model == "eedl":
        raise ValueError("radiative_model='bremslib-soft-hard' requires BremsLib, not 'eedl'")


def validate_inelastic_numerics(
    model: object, cutoff_eV: object, energy_model: object, secondary_threshold_eV: object = None
) -> None:
    """Validate the inelastic mode's settings (not its per-material W_cb).

    The per-material ``W_c > W_cb`` requirement and the secondary threshold's
    table coverage are checked where the layer materials are known, by the
    transport entry point.
    """
    if model not in INELASTIC_MODELS:
        raise ValueError(f"inelastic_model must be one of {', '.join(INELASTIC_MODELS)}")
    if secondary_threshold_eV is not None:
        if model != "shell-soft-hard":
            raise ValueError("secondary_threshold_eV requires inelastic_model='shell-soft-hard'")
        if (
            isinstance(secondary_threshold_eV, bool)
            or not isinstance(secondary_threshold_eV, (int, float))
            or not np.isfinite(secondary_threshold_eV)
            or secondary_threshold_eV <= 0.0
        ):
            raise ValueError("secondary_threshold_eV must be finite and positive")
    if model == "continuous":
        if cutoff_eV is not None:
            raise ValueError("inelastic_cutoff_eV requires inelastic_model='shell-soft-hard'")
        return
    if model == "auto" and cutoff_eV is None:
        return
    if (
        isinstance(cutoff_eV, bool)
        or not isinstance(cutoff_eV, (int, float))
        or not np.isfinite(cutoff_eV)
        or cutoff_eV <= 0.0
    ):
        raise ValueError(
            f"inelastic_model={model!r} requires a finite positive inelastic_cutoff_eV"
        )
    if model == "shell-soft-hard" and energy_model != "midpoint":
        raise ValueError("inelastic_model='shell-soft-hard' requires energy_model='midpoint'")


def electron_counts(value: object) -> tuple[int, ...]:
    """Decode one profile electron-count grid to validated integer values."""
    if isinstance(value, Mapping):
        value = cast(Mapping[str, object], value).get("values")
    if isinstance(value, np.ndarray):
        value = list(value.flat)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        value = (value,)
    raw_counts = tuple(value)
    if not raw_counts:
        raise ValueError("electron counts must be nonempty positive integers")
    counts = []
    for item in raw_counts:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item <= 0:
            raise ValueError("electron counts must be nonempty positive integers")
        counts.append(int(item))
    return tuple(counts)


def validate_profile_numerics(values: Mapping[str, object]) -> None:
    """Validate explicit profile numerics through the public typed objects."""
    unknown = set(values) - set(PROFILE_NUMERICS_KEYS)
    if unknown:
        raise ValueError(f"unknown numerics field: {sorted(unknown)[0]}")
    for key in SAMPLING_KEYS:
        if key in values:
            try:
                electron_counts(values[key])
            except ValueError as exc:
                raise ValueError(f"{key} {exc}") from None
    convergence_values: dict[str, Any] = {
        key: values[key] for key in CONVERGENCE_KEYS if key in values
    }
    transport_values: dict[str, Any] = {key: values[key] for key in TRANSPORT_KEYS if key in values}
    reflections = CxrReflections(**convergence_values)
    physics = Physics(
        **{key: value for key, value in transport_values.items() if key in PHYSICS_FIELDS}
    )
    numerics = Numerics(
        **{key: value for key, value in transport_values.items() if key not in PHYSICS_FIELDS},
        cxr_reflections=reflections,
    )
    resolve_physics(physics, numerics)
