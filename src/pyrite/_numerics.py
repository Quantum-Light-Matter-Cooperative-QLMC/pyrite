"""Low-level typed calculation-numerics policy and profile validation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

MosaicRoute = Literal["analytic", "mc"]

SAMPLING_KEYS = ("n_electrons", "n_electrons_brem")
CONVERGENCE_KEYS = ("n_families", "max_reflections", "mosaic_nodes", "mosaic_route")
TRANSPORT_KEYS = (
    "straggling",
    "energy_model",
    "max_dE_frac",
    "inelastic_model",
    "inelastic_cutoff_eV",
    "elastic_model",
    "bremsstrahlung_model",
)
#: ``simulate_trajectories`` collision-loss schemes; mirrors
#: ``montecarlo.transport.hard_inelastic.INELASTIC_MODELS`` (a test keeps the
#: two in step) without importing the transport package here.
INELASTIC_MODELS = ("continuous", "shell-soft-hard")
#: Elastic models a case may select. ``"elsepa"`` (the default, issue #89)
#: samples resolved ELSEPA tables; ``"mott"`` is the historical model.
ELASTIC_MODELS = ("mott", "elsepa")
#: Continuum bremsstrahlung sources a run may select (issue #86): the packaged
#: EEDL evaluation, the released BremsLib tables, or ``"auto"`` (the default),
#: which is BremsLib when every layer element's table is installed and EEDL,
#: with a warning, otherwise. A case records only the resolved choice.
BREMSSTRAHLUNG_MODELS = ("auto", "eedl", "bremslib")
PROFILE_NUMERICS_KEYS = (*SAMPLING_KEYS, *CONVERGENCE_KEYS, *TRANSPORT_KEYS)


@dataclass(frozen=True)
class Convergence:
    """Control numerical convergence and truncation of line calculations.

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
class Numerics:
    """Configure simulation sampling, transport, and execution.

    Parameters
    ----------
    n_electrons
        Macro-electron count for line transport.
    n_electrons_brem
        Macro-electron count for bremsstrahlung transport.
    spec_chunk, brem_chunk
        Optional spectrum-kernel chunk sizes; ``None`` lets runtime choose.
    transport_core
        ``"auto"``, ``"lockstep"``, ``"per-electron"``, or ``"cuda"``.
    backend
        Requested array backend. An explicit value must match the backend
        selected when PyRITE was imported.
    straggling
        Enable stochastic per-flight energy-loss straggling.
    energy_model
        ``"frozen"`` left-endpoint energy or ``"midpoint"`` integration.
    max_dE_frac
        Maximum predicted fractional mean loss per row. Zero disables
        substepping; a positive value requires ``energy_model="midpoint"``.
    inelastic_model, inelastic_cutoff_eV
        ``"continuous"`` stopping, or the opt-in ``"shell-soft-hard"`` mixed
        scheme with its energy-loss cutoff ``W_c`` in eV (required by, and
        only valid with, that mode, which also requires
        ``energy_model="midpoint"``).
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
        packaged EEDL continuum with an isotropic photon angle.
    convergence
        Reflection and mosaic convergence controls.
    """

    n_electrons: int = 450
    n_electrons_brem: int = 100
    spec_chunk: int | None = None
    brem_chunk: int | None = None
    transport_core: str = "auto"
    backend: str = "auto"
    straggling: bool = False
    energy_model: Literal["frozen", "midpoint"] = "frozen"
    max_dE_frac: float = 0.0
    inelastic_model: Literal["continuous", "shell-soft-hard"] = "continuous"
    inelastic_cutoff_eV: float | None = None
    elastic_model: Literal["mott", "elsepa"] = "elsepa"
    bremsstrahlung_model: Literal["auto", "eedl", "bremslib"] = "auto"
    convergence: Convergence = field(default_factory=Convergence)

    def __post_init__(self) -> None:
        for name in ("n_electrons", "n_electrons_brem"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("spec_chunk", "brem_chunk"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(self.straggling, bool):
            raise TypeError("straggling must be a bool")
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
        validate_inelastic_numerics(
            self.inelastic_model, self.inelastic_cutoff_eV, self.energy_model
        )
        validate_elastic_model(self.elastic_model)
        validate_bremsstrahlung_model(self.bremsstrahlung_model)


def validate_elastic_model(model: object) -> None:
    """Validate a case-level elastic scattering model name."""
    if model not in ELASTIC_MODELS:
        raise ValueError(f"elastic_model must be one of {', '.join(ELASTIC_MODELS)}")


def validate_bremsstrahlung_model(model: object) -> None:
    """Validate a case-level continuum bremsstrahlung source name."""
    if model not in BREMSSTRAHLUNG_MODELS:
        raise ValueError(f"bremsstrahlung_model must be one of {', '.join(BREMSSTRAHLUNG_MODELS)}")


def validate_inelastic_numerics(model: object, cutoff_eV: object, energy_model: object) -> None:
    """Validate the opt-in inelastic mode's settings (not its per-material W_cb).

    The per-material ``W_c > W_cb`` requirement is checked where the layer
    materials are known, by the transport entry point.
    """
    if model not in INELASTIC_MODELS:
        raise ValueError(f"inelastic_model must be one of {', '.join(INELASTIC_MODELS)}")
    if model == "continuous":
        if cutoff_eV is not None:
            raise ValueError("inelastic_cutoff_eV requires inelastic_model='shell-soft-hard'")
        return
    if (
        isinstance(cutoff_eV, bool)
        or not isinstance(cutoff_eV, (int, float))
        or not np.isfinite(cutoff_eV)
        or cutoff_eV <= 0.0
    ):
        raise ValueError(
            "inelastic_model='shell-soft-hard' requires a finite positive inelastic_cutoff_eV"
        )
    if energy_model != "midpoint":
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
    convergence = Convergence(**convergence_values)
    Numerics(
        **transport_values,
        convergence=convergence,
    )
