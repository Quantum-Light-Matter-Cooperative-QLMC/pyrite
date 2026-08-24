"""Low-level typed calculation-numerics policy and profile validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

MosaicRoute = Literal["analytic", "mc"]

SAMPLING_KEYS = ("n_electrons", "n_electrons_brem")
CONVERGENCE_KEYS = ("n_families", "max_reflections", "mosaic_nodes", "mosaic_route")
TRANSPORT_KEYS = ("straggling", "energy_model", "max_dE_frac")
PROFILE_NUMERICS_KEYS = (*SAMPLING_KEYS, *CONVERGENCE_KEYS, *TRANSPORT_KEYS)


@dataclass(frozen=True)
class Convergence:
    """Result-affecting convergence and truncation controls."""

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
    """Sampling, transport, convergence, and execution controls."""

    n_electrons: int = 450
    n_electrons_brem: int = 100
    spec_chunk: int | None = None
    brem_chunk: int | None = None
    transport_core: str = "auto"
    backend: str = "auto"
    straggling: bool = False
    energy_model: Literal["frozen", "midpoint"] = "frozen"
    max_dE_frac: float = 0.0
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
            raise ValueError("straggling must be a bool")
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
