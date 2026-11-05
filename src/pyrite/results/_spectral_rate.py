"""Shared boundary rules for explicit spectral current normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from scipy.constants import elementary_charge

from .._units import _scalar_quantity

if TYPE_CHECKING:
    from pint import Quantity

NormalizationBasis = Literal["per_incident_electron", "per_second", "per_current"]

_COUNT_BASES = {
    "per_incident_electron": "photons_per_incident_electron",
    "per_second": "photons_per_second",
    "per_current": "photons_per_second_per_ampere",
}
_RATE_UNITS = {"per_incident_electron": "", "per_second": " / s", "per_current": " / s / A"}


def count_basis(basis: str) -> str:
    """Validate the semantic state independently of Pint's dimensions."""
    if basis not in _COUNT_BASES:
        raise ValueError("unsupported spectral normalization_basis")
    return _COUNT_BASES[basis]


def rate_units(basis: str) -> str:
    count_basis(basis)
    return _RATE_UNITS[basis]


def current_amperes(current: Quantity) -> float:
    """Finite nonnegative scalar mean current, accepting external registries."""
    return _scalar_quantity(current, unit="A", field="current", expected="current", allow_zero=True)


def per_current_factor(basis: str) -> float:
    """I/|e| or 1/|e| from per-primary yields; never divide by beam current.

    Assumes a fixed stored yield and independently repeated realizations or
    bunches, with nonnegative mean current magnitude. Coherent yields retain
    their stored bunch population/shape: changing current does not recompute
    charge-dependent radiation. Per-current output uses ampere as its denominator. No energy,
    solid-angle, transmission, response or exposure factor is included.
    Source: BIPM SI ampere, https://www.bipm.org/en/si-base-units/ampere
    Validation: spectral-explicit-rate-normalization
    """
    count_basis(basis)
    if basis != "per_incident_electron":
        raise ValueError("current normalization has already been applied")
    return 1.0 / elementary_charge


def rate_factor(basis: str, current: Quantity) -> float:
    """I/|e| for the fixed stored yield; zero current gives zero finite rates.

    Source and bunch assumptions follow :func:`per_current_factor`.
    None is not a default-current request.
    Validation: spectral-explicit-rate-normalization
    """
    return per_current_factor(basis) * current_amperes(current)


def table_metadata(view: object) -> dict[str, str | bool]:
    """Semantic columns shared by native spectral tables."""
    return {
        name: getattr(view, name)
        for name in (
            "sampling",
            "normalization",
            "normalization_basis",
            "count_basis",
            "measure",
            "acceptance_applied",
        )
    }


def quantity_columns(columns: dict[str, Quantity], basis: str) -> dict[str, dict[str, str | None]]:
    """Physical column units; semantic labels live in the table itself."""
    return {
        name: {
            "unit": str(value.units),
            "basis": None if name.startswith("photon_energy") or name == "angle" else basis,
        }
        for name, value in columns.items()
    }
