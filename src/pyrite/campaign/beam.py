"""The public electron beam: one choice per kind of beam property.

:class:`Beam` holds exactly one transverse description, one longitudinal
description, and an optional GPT/GDF source, each a small frozen object, so an
invalid mix cannot be constructed and the type hints list what is allowed:

* transverse -- :class:`Spot` (Gaussian entrance spot), :class:`PointSpot`, or a
  :class:`~pyrite.montecarlo.transverse.TransverseDistribution`
  (Courant--Snyder);
* longitudinal -- :class:`PointBunch`, :class:`BunchLength`, :class:`Offsets`,
  or a :class:`~pyrite.campaign.longitudinal.LongitudinalDistribution` policy;
* source -- analytic by default, or :class:`GdfSource` through
  :meth:`Beam.from_gdf`.

The transport layers consume the flat :class:`~pyrite.campaign.sweep.BeamSpec`
record, an internal lowering target that also backs profile files and case
payloads. :meth:`Beam.to_spec` lowers to it field-for-field.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .._api_deprecations import warn_api
from ..montecarlo.transverse import TransverseDistribution
from .longitudinal import LongitudinalDistribution
from .sweep import BeamSpec


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class Spot:
    """Gaussian entrance spot, a zero-emittance waist.

    Parameters
    ----------
    fwhm_x_mm
        Full width at half maximum along x in mm.
    fwhm_y_mm
        FWHM along y in mm; ``None`` (default) makes the spot round.
    """

    fwhm_x_mm: float = 1.0
    fwhm_y_mm: float | None = None

    def __post_init__(self) -> None:
        _positive("fwhm_x_mm", self.fwhm_x_mm)
        if self.fwhm_y_mm is not None:
            _positive("fwhm_y_mm", self.fwhm_y_mm)

    @property
    def plane_y_mm(self) -> float:
        """The y FWHM, mirroring x when unset."""
        return self.fwhm_x_mm if self.fwhm_y_mm is None else self.fwhm_y_mm


@dataclass(frozen=True)
class PointSpot:
    """Point-source transverse distribution (no spot size, no divergence)."""


@dataclass(frozen=True)
class PointBunch:
    """Every electron arrives at t = 0."""


@dataclass(frozen=True)
class BunchLength:
    """Sampled arrival times with an RMS duration.

    Parameters
    ----------
    rms_fs
        RMS bunch duration in fs.
    shape
        Sampling law name, ``"gaussian"`` by default.
    """

    rms_fs: float
    shape: str = "gaussian"

    def __post_init__(self) -> None:
        _positive("rms_fs", self.rms_fs)
        if not isinstance(self.shape, str) or not self.shape:
            raise ValueError("shape must be a non-empty string")


@dataclass(frozen=True)
class Offsets:
    """Explicit per-electron arrival offsets in fs, replacing random sampling."""

    offsets_fs: tuple[float, ...]
    shape: str = "gaussian"

    def __post_init__(self) -> None:
        values = tuple(float(x) for x in self.offsets_fs)
        if not values:
            raise ValueError("offsets_fs must not be empty")
        if not all(np.isfinite(values)):
            raise ValueError("offsets_fs must be finite")
        object.__setattr__(self, "offsets_fs", values)


@dataclass(frozen=True)
class GdfSource:
    """A GPT time-output snapshot or selected screen as the electron source.

    Its correlated energies replace ``Beam.energy_keV`` during case
    construction, and it carries its own phase space, so it excludes the
    analytic transverse/longitudinal choices and ``energy_spread_frac``.

    Parameters
    ----------
    path
        GDF file.
    z_origin_m
        Target-origin lab z coordinate in meters (GPT axes are PyRITE lab axes).
    time_s, time_tolerance_s
        Snapshot time and absolute tolerance.
    screen_position_m, screen_tolerance_m
        Select a screen instead of a time snapshot.
    normalization
        ``"pyrite_current"`` keeps the configured source normalization;
        ``"gdf_charge"`` derives current from the file's absolute charge and
        ``Beam.rep_rate_hz``.
    shape_only
        Keep positions/directions/weights, assign each sweep energy, discard
        imported crossing times.
    """

    path: str
    z_origin_m: float
    time_s: float | None = None
    time_tolerance_s: float = 1e-15
    screen_position_m: float | None = None
    screen_tolerance_m: float = 1e-9
    normalization: str = "pyrite_current"
    shape_only: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("GdfSource requires a path")
        if not np.isfinite(self.z_origin_m):
            raise ValueError(
                "GdfSource requires a finite z_origin_m (target origin, GPT lab frame)"
            )
        if self.normalization not in {"pyrite_current", "gdf_charge"}:
            raise ValueError("normalization must be 'pyrite_current' or 'gdf_charge'")
        if not isinstance(self.shape_only, bool):
            raise ValueError("shape_only must be a boolean")


Transverse = Spot | PointSpot | TransverseDistribution
Longitudinal = PointBunch | BunchLength | Offsets | LongitudinalDistribution


#: Placeholder for a beam property the caller left out. Through the #451
#: deprecation window it resolves, with a warning, to the historical default.
_UNSET: Any = cast(Any, object())


@dataclass(frozen=True)
class Beam:
    """Describe the electron beam at the crystal entrance plane.

    Each kind of beam property takes exactly one description, so the allowed
    combinations are the type signature rather than prose:

    >>> Beam(energy_keV=30.0, transverse=Spot(fwhm_x_mm=0.5), longitudinal=PointBunch())  # doctest: +SKIP
    >>> Beam(  # doctest: +SKIP
    ...     energy_keV=30.0, transverse=PointSpot(), longitudinal=BunchLength(rms_fs=100.0)
    ... )
    >>> Beam.from_gdf("beam.gdf", z_origin_m=0.0, time_s=1e-9)  # doctest: +SKIP

    Parameters
    ----------
    energy_keV
        Central kinetic energy in keV. A scalar creates one scene; campaign
        builders may expand a sequence.
    transverse
        Required: :class:`Spot`, :class:`PointSpot`, or a Courant--Snyder
        :class:`~pyrite.montecarlo.transverse.TransverseDistribution`. Omitting it
        is deprecated and falls back to a 1 mm round :class:`Spot`.
    longitudinal
        Required: :class:`PointBunch`, :class:`BunchLength`, :class:`Offsets`, or a
        :class:`~pyrite.campaign.longitudinal.LongitudinalDistribution` policy.
        Omitting it is deprecated and falls back to :class:`PointBunch`.
    energy_spread_frac
        RMS fractional kinetic-energy spread; ``None`` is monoenergetic.
    rep_rate_hz
        Pulse repetition rate in Hz used for flux normalization.
    bunch_charge_pc
        Charge per bunch in pC used for flux normalization.
    gdf
        A :class:`GdfSource`; build through :meth:`from_gdf`.

    Notes
    -----
    Sampling counts and execution controls belong to :class:`Numerics`.
    Divergence is derived from ``transverse`` and the energy; read it with
    :meth:`derived_divergence_mrad`.
    """

    energy_keV: Any = (30.0, 45.0, 60.0)
    transverse: Transverse = _UNSET
    longitudinal: Longitudinal = _UNSET
    energy_spread_frac: float | None = None
    rep_rate_hz: float = 5000.0
    bunch_charge_pc: float = 1.0
    gdf: GdfSource | None = None

    def __post_init__(self) -> None:
        # stacklevel 4: warn_api -> __post_init__ -> __init__ -> the caller.
        if self.transverse is _UNSET:
            warn_api("Beam() without transverse=", stacklevel=4)
            object.__setattr__(self, "transverse", Spot())
        if self.longitudinal is _UNSET:
            warn_api("Beam() without longitudinal=", stacklevel=4)
            object.__setattr__(self, "longitudinal", PointBunch())
        if not isinstance(self.transverse, (Spot, PointSpot, TransverseDistribution)):
            raise TypeError("Beam.transverse must be a Spot, PointSpot, or TransverseDistribution")
        if not isinstance(
            self.longitudinal, (PointBunch, BunchLength, Offsets, LongitudinalDistribution)
        ):
            raise TypeError(
                "Beam.longitudinal must be a PointBunch, BunchLength, Offsets, or "
                "LongitudinalDistribution"
            )
        if self.gdf is not None:
            if not isinstance(self.gdf, GdfSource):
                raise TypeError("Beam.gdf must be a GdfSource")
            if (
                not isinstance(self.transverse, PointSpot)
                or not isinstance(self.longitudinal, PointBunch)
                or self.energy_spread_frac is not None
            ):
                raise ValueError(
                    "a GDF source carries its own phase space: use Beam.from_gdf and omit "
                    "transverse, longitudinal and energy_spread_frac"
                )
            if self.gdf.normalization == "gdf_charge" and not (
                np.isfinite(self.rep_rate_hz) and self.rep_rate_hz > 0
            ):
                raise ValueError("gdf_charge requires finite positive rep_rate_hz")

    @classmethod
    def from_gdf(
        cls,
        path: str,
        *,
        z_origin_m: float,
        time_s: float | None = None,
        time_tolerance_s: float = 1e-15,
        screen_position_m: float | None = None,
        screen_tolerance_m: float = 1e-9,
        normalization: str = "pyrite_current",
        shape_only: bool = False,
        energy_keV: Any = (30.0, 45.0, 60.0),
        rep_rate_hz: float = 5000.0,
        bunch_charge_pc: float = 1.0,
    ) -> "Beam":
        """Build a beam whose phase space comes from a GPT GDF file."""
        return cls(
            energy_keV=energy_keV,
            transverse=PointSpot(),
            longitudinal=PointBunch(),
            rep_rate_hz=rep_rate_hz,
            bunch_charge_pc=bunch_charge_pc,
            gdf=GdfSource(
                path=path,
                z_origin_m=z_origin_m,
                time_s=time_s,
                time_tolerance_s=time_tolerance_s,
                screen_position_m=screen_position_m,
                screen_tolerance_m=screen_tolerance_m,
                normalization=normalization,
                shape_only=shape_only,
            ),
        )

    def derived_divergence_mrad(self, energy_keV: float) -> tuple[float, float] | None:
        """Per-plane RMS slope [mrad] at ``energy_keV``; ``None`` unless Courant--Snyder."""
        return self.to_spec().derived_divergence_mrad(energy_keV)

    def to_spec(self) -> BeamSpec:
        """Lower to the flat :class:`BeamSpec` the transport layers consume."""
        fields: dict[str, Any] = {
            "energy_keV": self.energy_keV,
            "energy_spread_frac": self.energy_spread_frac,
            "rep_rate_hz": self.rep_rate_hz,
            "bunch_charge_pc": self.bunch_charge_pc,
            "transverse_fwhm_x_mm": None,
            "transverse_fwhm_y_mm": None,
        }
        transverse = self.transverse
        if isinstance(transverse, Spot):
            fields["transverse_fwhm_x_mm"] = float(transverse.fwhm_x_mm)
            fields["transverse_fwhm_y_mm"] = float(transverse.plane_y_mm)
        elif isinstance(transverse, TransverseDistribution):
            fields["transverse"] = transverse
        longitudinal = self.longitudinal
        if isinstance(longitudinal, BunchLength):
            fields["bunch_length_fs"] = float(longitudinal.rms_fs)
            fields["long_shape"] = longitudinal.shape
        elif isinstance(longitudinal, Offsets):
            fields["long_offsets_fs"] = longitudinal.offsets_fs
            fields["long_shape"] = longitudinal.shape
        elif isinstance(longitudinal, LongitudinalDistribution):
            fields["longitudinal"] = longitudinal
        if self.gdf is not None:
            gdf = self.gdf
            fields.update(
                source="gpt_gdf",
                gdf_path=gdf.path,
                gdf_time_s=gdf.time_s,
                gdf_time_tolerance_s=gdf.time_tolerance_s,
                gdf_screen_position_m=gdf.screen_position_m,
                gdf_screen_tolerance_m=gdf.screen_tolerance_m,
                gdf_normalization=gdf.normalization,
                gdf_z_origin_m=gdf.z_origin_m,
                gdf_shape_only=gdf.shape_only,
            )
        return BeamSpec(**fields)

    @classmethod
    def from_spec(cls, spec: BeamSpec) -> "Beam":
        """Build the equivalent beam from a flat :class:`BeamSpec`."""
        if isinstance(spec, cls):
            return spec
        if spec.source == "gpt_gdf":
            if (
                spec.transverse is not None
                or spec.longitudinal is not None
                or spec.bunch_length_fs is not None
                or spec.long_offsets_fs is not None
                or spec.energy_spread_frac is not None
                or spec.long_shape != "gaussian"
                or spec.transverse_fwhm_x_mm is not None
                or spec.transverse_fwhm_y_mm is not None
            ):
                raise ValueError(
                    "gpt_gdf is incompatible with analytic phase-space fields; clear both spot FWHMs"
                )
            if not spec.gdf_path:
                raise ValueError("gpt_gdf requires gdf_path")
            if spec.gdf_z_origin_m is None:
                raise ValueError("gpt_gdf requires explicit finite gdf_z_origin_m")
            gdf = GdfSource(
                path=str(spec.gdf_path),
                z_origin_m=float(spec.gdf_z_origin_m),  # type: ignore[arg-type]
                time_s=spec.gdf_time_s,
                time_tolerance_s=spec.gdf_time_tolerance_s,
                screen_position_m=spec.gdf_screen_position_m,
                screen_tolerance_m=spec.gdf_screen_tolerance_m,
                normalization=spec.gdf_normalization,
                shape_only=spec.gdf_shape_only,
            )
            return cls(
                energy_keV=spec.energy_keV,
                transverse=PointSpot(),
                longitudinal=PointBunch(),
                rep_rate_hz=spec.rep_rate_hz,
                bunch_charge_pc=spec.bunch_charge_pc,
                gdf=gdf,
            )
        transverse: Transverse
        if spec.transverse is not None:
            if spec.transverse_fwhm_x_mm is not None or spec.transverse_fwhm_y_mm is not None:
                raise ValueError(
                    "transverse policy is incompatible with the legacy transverse_fwhm_x_mm, "
                    "transverse_fwhm_y_mm fields; clear them (they default to a 1 mm spot)"
                )
            transverse = spec.transverse
        elif spec.transverse_fwhm_x_mm is None and spec.transverse_fwhm_y_mm is None:
            transverse = PointSpot()
        elif spec.transverse_fwhm_x_mm is None or spec.transverse_fwhm_y_mm is None:
            raise ValueError("set both transverse spot FWHMs, or neither for a point source")
        else:
            x, y = float(spec.transverse_fwhm_x_mm), float(spec.transverse_fwhm_y_mm)
            transverse = Spot(x, None if y == x else y)
        flat_bunch = (
            spec.bunch_length_fs is not None
            or spec.long_offsets_fs is not None
            or spec.long_shape != "gaussian"
        )
        longitudinal: Longitudinal
        if spec.longitudinal is not None:
            if flat_bunch:
                raise ValueError(
                    "longitudinal policy is incompatible with legacy bunch_length_fs, "
                    "long_shape, and long_offsets_fs fields"
                )
            longitudinal = spec.longitudinal
        elif spec.long_offsets_fs is not None:
            longitudinal = Offsets(tuple(spec.long_offsets_fs), spec.long_shape)
        elif spec.bunch_length_fs is not None:
            longitudinal = BunchLength(spec.bunch_length_fs, spec.long_shape)
        elif flat_bunch:
            raise ValueError("long_shape needs bunch_length_fs or long_offsets_fs")
        else:
            longitudinal = PointBunch()
        return cls(
            energy_keV=spec.energy_keV,
            transverse=transverse,
            longitudinal=longitudinal,
            energy_spread_frac=spec.energy_spread_frac,
            rep_rate_hz=spec.rep_rate_hz,
            bunch_charge_pc=spec.bunch_charge_pc,
        )


__all__ = [
    "Beam",
    "BunchLength",
    "GdfSource",
    "Offsets",
    "PointBunch",
    "PointSpot",
    "Spot",
]


def __getattr__(name: str) -> Any:
    """Serve the deprecated ``Point`` alias of :class:`PointSpot` (issue #451)."""
    if name == "Point":
        warn_api("pyrite.campaign.beam.Point")
        return PointSpot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
