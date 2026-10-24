"""Typed, import-light quantum propagation configuration (issue #680).

Only the established planar density-matrix zone can be lowered to production
cases. Unsupported recipes fail before case construction; FFT-grid escape is
never interpreted as physical dechanneling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from ._numerics import Numerics, Physics, _miller, validate_channeling_numerics


@dataclass(frozen=True)
class QuantumRegion:
    """Physical region and handoff cap, not an FFT grid or numerical boundary."""

    kind: Literal["free", "planar", "axial"] = "free"
    plane_hkl: tuple[int, int, int] | None = None
    axis_uvw: tuple[int, int, int] | None = None
    zone_max_ang: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("free", "planar", "axial"):
            raise ValueError("unknown quantum region kind")
        if self.kind == "planar":
            if self.plane_hkl is None:
                raise ValueError("planar quantum region requires plane_hkl")
            object.__setattr__(self, "plane_hkl", _miller(self.plane_hkl))
            if self.axis_uvw is not None:
                raise ValueError("planar quantum region cannot set axis_uvw")
        elif self.kind == "axial":
            if self.axis_uvw is None:
                raise ValueError("axial quantum region requires axis_uvw")
            object.__setattr__(self, "axis_uvw", _miller(self.axis_uvw))
            if self.plane_hkl is not None:
                raise ValueError("axial quantum region cannot set plane_hkl")
        elif any(v is not None for v in (self.plane_hkl, self.axis_uvw, self.zone_max_ang)):
            raise ValueError("free quantum region cannot set alignment or zone cap")
        if self.zone_max_ang is not None:
            validate_channeling_numerics(16, 16, 250.0, self.zone_max_ang)


@dataclass(frozen=True)
class QuantumDynamics:
    """Physical dynamics recipe, separate from discretization and source."""

    kind: Literal["free", "channeling"] = "free"

    def __post_init__(self) -> None:
        if self.kind not in ("free", "channeling"):
            raise ValueError("unknown quantum dynamics kind")


@dataclass(frozen=True)
class QuantumNumerics:
    """Propagation representation and numerical integration controls."""

    representation: Literal["bloch", "real-space"] = "bloch"
    method: Literal["coherent", "density-matrix", "mcwf"] = "density-matrix"
    basis_size: int = 16
    bloch_cells: int = 16
    step_ang: float = 250.0

    def __post_init__(self) -> None:
        if self.representation not in ("bloch", "real-space"):
            raise ValueError("unknown quantum representation")
        if self.method not in ("coherent", "density-matrix", "mcwf"):
            raise ValueError("unknown quantum propagation method")
        validate_channeling_numerics(self.basis_size, self.bloch_cells, self.step_ang, None)


@dataclass(frozen=True)
class QuantumSetup:
    """Optional reusable quantum stage owned by a Scene.

    The initial state and nonchanneling solvers are intentionally not wired to
    production transport yet. Unsupported configurations explicitly refuse.
    """

    region: QuantumRegion = field(default_factory=QuantumRegion)
    dynamics: QuantumDynamics = field(default_factory=QuantumDynamics)
    numerics: QuantumNumerics = field(default_factory=QuantumNumerics)

    def __post_init__(self) -> None:
        for name, cls in (
            ("region", QuantumRegion),
            ("dynamics", QuantumDynamics),
            ("numerics", QuantumNumerics),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"QuantumSetup.{name} must be {cls.__name__}")

    def lower(self, physics: Physics, numerics: Numerics) -> tuple[Physics, Numerics]:
        """Adapt supported planar configuration to unchanged versioned case fields."""
        if self.region.kind == "axial":
            from .campaign.axial_spec import refuse_unverified_axial_run

            refuse_unverified_axial_run()
        if (
            self.region.kind != "planar"
            or self.dynamics.kind != "channeling"
            or self.numerics.representation != "bloch"
            or self.numerics.method != "density-matrix"
        ):
            raise ValueError(
                "quantum production supports only planar Bloch density-matrix "
                "channeling; other configurations are diagnostic-only"
            )
        if physics.channeling_model != "off" or physics.channeling_axial is not None:
            raise ValueError("Scene.quantum conflicts with legacy Physics.channeling_model")
        if numerics.precision is not None:
            raise ValueError("adaptive precision is unsupported by planar quantum channeling")
        values = dict(
            channeling_basis_size=self.numerics.basis_size,
            channeling_bloch_cells=self.numerics.bloch_cells,
            channeling_dz_ang=self.numerics.step_ang,
            channeling_zone_max_ang=self.region.zone_max_ang,
        )
        defaults = Numerics()
        for name, value in values.items():
            current = getattr(numerics, name)
            if current != getattr(defaults, name) and current != value:
                raise ValueError(f"Scene.quantum conflicts with legacy Numerics.{name}")
        return (
            replace(physics, channeling_model="planar", channeling_plane=self.region.plane_hkl),
            replace(numerics, **values),
        )
