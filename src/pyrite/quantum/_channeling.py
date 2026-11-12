"""Public channeling run section and its private case-lowering boundary (#1017).

:class:`Channeling` is the only public spelling of the channeling zone. Case
lowering, analysis :class:`~pyrite.results.store.Settings`, dataset identities
and stored checkpoints keep the historical flat ``channeling_*`` record, so the
helpers here translate between the two without changing any case key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np

from .._channeling_table import (
    CHANNELING_FIELD_TABLE_KEYS as CHANNELING_FIELD_TABLE_KEYS,
)
from .._channeling_table import CHANNELING_OFF_FIELDS as CHANNELING_OFF_FIELDS
from .._channeling_table import CHANNELING_TABLE_KEYS as CHANNELING_TABLE_KEYS
from .._channeling_table import channeling_table_values as channeling_table_values
from .._spec_defaults import UNSET, _Unset
from .._units import _positive_length_quantity, normalize_unit

if TYPE_CHECKING:
    from pint import Quantity

    from .._numerics import Physics

ChannelingModel = Literal["planar", "axial"]
ChannelingAdmission = Literal["admitted", "accept-inaccurate"]


def _angstrom(value: float) -> Quantity:
    """Fresh registry-owned scalar angstrom read-back."""
    from ..units import Q_

    return Q_(float(value), "angstrom")


def _length_spelling(owner: str, numeric_name: str, numeric: Any, name: str, semantic: object):
    """Resolve a numeric-angstrom field against its keyword-only semantic length."""
    if semantic is UNSET:
        return numeric
    if numeric is not UNSET:
        raise TypeError(f"{owner}.{numeric_name} and {name} are mutually exclusive")
    return _positive_length_quantity(semantic, unit="angstrom", field=f"{owner}.{name}", shape=())


def miller_indices(value: object, name: str = "Channeling.plane") -> tuple[int, int, int]:
    """Validate three nonzero integer Miller indices."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must be three integer Miller indices")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError(f"{name} must be three integer Miller indices")
    hkl = (int(value[0]), int(value[1]), int(value[2]))
    if hkl == (0, 0, 0):
        raise ValueError(f"{name} must be nonzero")
    return hkl


def validate_channeling_numerics(
    basis_size: object, bloch_cells: object, step_ang: object, zone_max_ang: object
) -> None:
    """Validate the planar channeling zone's numerical controls."""
    for name, value in (("basis_size", basis_size), ("bloch_cells", bloch_cells)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"ChannelingNumerics.{name} must be a positive integer")
    for name, value in (("step_ang", step_ang), ("zone_max_ang", zone_max_ang)):
        if value is None and name == "zone_max_ang":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"ChannelingNumerics.{name} must be finite and positive")


@dataclass(frozen=True, init=False)
class ChannelingNumerics:
    """Planar channeling-zone discretization (ADR-0018).

    ``basis_size`` is the plane-wave band ``|h| <= H`` of the transverse
    basis and ``bloch_cells`` the Bloch-grid supercell size; both are counts.
    ``step_ang`` is the density-matrix depth step [Å]; ``zone_max_ang`` is an
    optional zone-length cap [Å] beyond which still-channeled electrons are
    handed off (``None``: the first layer's thickness). Keyword-only
    ``step_length`` and ``zone_max_length`` take explicit positive scalar Pint
    lengths instead; the properties read them back as fresh Å quantities.
    Axial fibre cases take their numerics from the repository lookup and
    ignore these controls.
    """

    basis_size: int = 16
    bloch_cells: int = 16
    step_ang: float = 250.0
    zone_max_ang: float | None = None

    def __init__(
        self,
        basis_size: int = 16,
        bloch_cells: int = 16,
        step_ang: float | _Unset = UNSET,
        zone_max_ang: float | None | _Unset = UNSET,
        *,
        step_length: object = UNSET,
        zone_max_length: object = UNSET,
    ) -> None:
        owner = "ChannelingNumerics"
        step = _length_spelling(owner, "step_ang", step_ang, "step_length", step_length)
        zone = _length_spelling(
            owner, "zone_max_ang", zone_max_ang, "zone_max_length", zone_max_length
        )
        object.__setattr__(self, "basis_size", basis_size)
        object.__setattr__(self, "bloch_cells", bloch_cells)
        object.__setattr__(self, "step_ang", 250.0 if step is UNSET else step)
        object.__setattr__(self, "zone_max_ang", None if zone is UNSET else zone)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_ang",
            normalize_unit(
                self.step_ang, "angstrom", field="ChannelingNumerics.step_ang", expected="length"
            ),
        )
        if self.zone_max_ang is not None:
            object.__setattr__(
                self,
                "zone_max_ang",
                normalize_unit(
                    self.zone_max_ang,
                    "angstrom",
                    field="ChannelingNumerics.zone_max_ang",
                    expected="length",
                ),
            )
        validate_channeling_numerics(
            self.basis_size, self.bloch_cells, self.step_ang, self.zone_max_ang
        )
        object.__setattr__(self, "basis_size", int(self.basis_size))
        object.__setattr__(self, "bloch_cells", int(self.bloch_cells))

    @property
    def step_length(self) -> Quantity:
        """Fresh Å quantity for the density-matrix depth step."""
        return _angstrom(self.step_ang)

    @property
    def zone_max_length(self) -> Quantity | None:
        """Fresh Å zone-length cap, or ``None`` when uncapped."""
        return None if self.zone_max_ang is None else _angstrom(self.zone_max_ang)

    def with_step_length(self, step_length: object) -> Self:
        """Copy with an explicit positive step length; preserve other controls."""
        step = _length_spelling("ChannelingNumerics", "step_ang", UNSET, "step_length", step_length)
        return replace(self, step_ang=step)

    def with_zone_max_length(self, zone_max_length: object) -> Self:
        """Copy with an explicit positive length cap, or ``None`` to remove it."""
        if zone_max_length is None:
            return replace(self, zone_max_ang=None)
        zone = _length_spelling(
            "ChannelingNumerics", "zone_max_ang", UNSET, "zone_max_length", zone_max_length
        )
        return replace(self, zone_max_ang=zone)


@dataclass(frozen=True)
class Channeling:
    """Opt-in quantum channeling zone of a run (ADR-0018); ``None`` is off.

    Pass as ``simulate(..., channeling=Channeling(...))``. ``model="planar"``
    owns each electron from crystal entry in the planar density-matrix zone of
    the ``plane`` Miller family until it dechannels, then hands it to native
    transport. ``model="axial"`` takes ``axial``, a ``channeling_axial``
    version-2 fibre selector (``{"version": 2, "axis_uvw": [u, v, w]}``);
    only repository-calibrated fibre cases are admitted (#623, #994).
    ``numerics`` holds the planar zone discretization.

    ``admission="admitted"`` (default) runs only cases in the measured
    admission tables; ``"accept-inaccurate"`` runs any other case,
    acknowledging that its results may be very inaccurate, and marks them
    unadmitted with no error budget (#1105).

    A run with channeling requires midpoint integration,
    ``radiative_model="uncoupled"``, incoherent emission, no secondary
    transport and no transition radiation.
    """

    model: ChannelingModel
    plane: tuple[int, int, int] | None = None
    axial: Mapping[str, object] | None = None
    numerics: ChannelingNumerics = field(default_factory=ChannelingNumerics)
    admission: ChannelingAdmission = "admitted"

    def __post_init__(self) -> None:
        from ._admission import CHANNELING_ADMISSIONS

        if self.model not in ("planar", "axial"):
            raise ValueError(
                "Channeling.model must be 'planar' or 'axial'; pass channeling=None for off"
            )
        if not isinstance(self.numerics, ChannelingNumerics):
            raise TypeError("Channeling.numerics must be a ChannelingNumerics")
        if self.admission not in CHANNELING_ADMISSIONS:
            raise ValueError(
                f"Channeling.admission must be one of {', '.join(CHANNELING_ADMISSIONS)}"
            )
        if self.model == "planar":
            if self.axial is not None:
                raise ValueError("Channeling.axial requires model='axial'")
            if self.plane is None:
                raise ValueError("Channeling(model='planar') requires plane")
            object.__setattr__(self, "plane", miller_indices(self.plane))
            return
        if self.plane is not None:
            raise ValueError("Channeling.plane is planar-only; use Channeling.axial axis_uvw")
        if self.axial is None:
            raise ValueError("Channeling(model='axial') requires axial")
        from ._axial_spec import resolve_axial_spec

        object.__setattr__(self, "axial", resolve_axial_spec(self.axial))


def channeling_fields(channeling: Channeling | None) -> dict[str, object]:
    """Private lowering output: the stored flat ``channeling_*`` record."""
    if channeling is None:
        return dict(CHANNELING_OFF_FIELDS)
    if not isinstance(channeling, Channeling):
        raise TypeError("channeling must be a Channeling or None")
    numerics = channeling.numerics
    return {
        "channeling_model": channeling.model,
        "channeling_plane": channeling.plane,
        "channeling_axial": None if channeling.axial is None else dict(channeling.axial),
        "channeling_admission": channeling.admission,
        "channeling_basis_size": numerics.basis_size,
        "channeling_bloch_cells": numerics.bloch_cells,
        "channeling_dz_ang": numerics.step_ang,
        "channeling_zone_max_ang": numerics.zone_max_ang,
    }


def channeling_from_fields(values: Mapping[str, object]) -> Channeling | None:
    """Rebuild the public section from a flat record (profiles, Settings).

    Absent keys take their off defaults. An ``"off"`` model with any selector
    or admission opt-in is refused; numerics alone with ``"off"`` are inert.
    """
    flat = {
        **CHANNELING_OFF_FIELDS,
        **{k: v for k, v in values.items() if k in CHANNELING_OFF_FIELDS},
    }
    model = flat["channeling_model"]
    if model in (None, "off"):
        for name in ("plane", "axial"):
            if flat[f"channeling_{name}"] is not None:
                raise ValueError(f"[channeling] {name} requires model='planar' or 'axial'")
        if flat["channeling_admission"] not in (None, "admitted"):
            raise ValueError("[channeling] admission requires a channeling model")
        ChannelingNumerics(
            cast(int, flat["channeling_basis_size"]),
            cast(int, flat["channeling_bloch_cells"]),
            cast(float, flat["channeling_dz_ang"]),
            cast(float | None, flat["channeling_zone_max_ang"]),
        )
        return None
    return Channeling(
        model=cast(ChannelingModel, model),
        plane=cast(tuple[int, int, int] | None, flat["channeling_plane"]),
        axial=cast(Mapping[str, object] | None, flat["channeling_axial"]),
        numerics=ChannelingNumerics(
            cast(int, flat["channeling_basis_size"]),
            cast(int, flat["channeling_bloch_cells"]),
            cast(float, flat["channeling_dz_ang"]),
            cast(float | None, flat["channeling_zone_max_ang"]),
        ),
        admission=cast(ChannelingAdmission, flat["channeling_admission"] or "admitted"),
    )


def validate_channeling_run(
    channeling: Channeling | None, physics: Physics, energy_model: str
) -> None:
    """Refuse run-level model combinations the channeling zone cannot support.

    Per-case gates (crystal data, energy, bound states, admission tables,
    geometry) run in ``campaign.channeling_cases`` and ``Case`` validation.
    """
    if channeling is None:
        return
    label = f"Channeling(model={channeling.model!r})"
    if energy_model != "midpoint":
        raise ValueError(f"{label} requires energy_model='midpoint'")
    if physics.radiative_model != "uncoupled":
        raise ValueError(
            f"{label} requires radiative_model='uncoupled': coupled radiative loss "
            "inside the channeling zone is not modelled"
        )
    if physics.secondary_threshold_eV is not None:
        raise ValueError(f"{label} does not support secondary transport")
    if "coherent" in cast(tuple[str, ...], physics.emission):
        raise ValueError(f"{label} supports incoherent emission only")
    if physics.transition_radiation != "off":
        raise ValueError(f"transition_radiation does not support {label}")


__all__ = ["Channeling", "ChannelingNumerics"]
