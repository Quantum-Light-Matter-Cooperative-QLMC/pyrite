"""The public electron beam: one choice per kind of beam property.

:class:`Beam` holds exactly one transverse description, one longitudinal
description, and an optional GPT/GDF source, each a small frozen object, so an
invalid mix cannot be constructed and the type hints list what is allowed:

* transverse -- :class:`TransverseSpot` (Gaussian entrance spot), :class:`PointSpot`, or a
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

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .._spec_defaults import UNSET as _UNSET
from .._spec_defaults import _Unset
from .._units import _is_pint_quantity, normalize_unit
from ..montecarlo.bunch_front import BunchFrontTilt, validate_front_source
from ..montecarlo.transverse import TransverseDistribution
from ..quantum_state import TransverseStateSpec
from .longitudinal import LongitudinalDistribution
from .sweep import BeamSpec

if TYPE_CHECKING:
    from pint import Quantity


def _kinetic_energy_keV(value: object) -> float | np.ndarray:
    """Validate an explicit energy quantity and own its numeric keV magnitude."""
    if not _is_pint_quantity(value):
        raise TypeError("Beam.kinetic_energy requires a Pint quantity with energy units")
    magnitude = normalize_unit(value, "keV", field="Beam.kinetic_energy", expected="energy")
    energy = np.asarray(magnitude)
    if energy.dtype.kind not in "iuf" or energy.ndim > 1:
        raise TypeError(
            "Beam.kinetic_energy must be a real scalar or one-dimensional numeric array"
        )
    if energy.size == 0 or not np.all(np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Beam.kinetic_energy must contain finite positive energies")
    return _positive("Beam.kinetic_energy", float(energy)) if energy.ndim == 0 else energy.copy()


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _timing_quantity_fs(value: object, *, field: str, ndim: int) -> np.ndarray:
    """Own a real explicit-time quantity representable in numeric fs storage."""
    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires a Pint quantity with time units")
    magnitude = normalize_unit(value, "fs", field=field, expected="time")
    array = np.asarray(magnitude)
    if array.dtype.kind not in "iuf" or array.ndim != ndim:
        shape = "scalar" if ndim == 0 else "one-dimensional array"
        raise TypeError(f"{field} must be a real {shape}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain nonempty finite times")
    with np.errstate(over="ignore", under="ignore"):
        stored = array.astype(float)
    if not np.all(np.isfinite(stored)) or np.any((array != 0) & (stored == 0)):
        raise ValueError(
            f"{field} must be representable in numeric fs storage without overflow or underflow"
        )
    return stored


def _spot_width_mm(value: object, *, field: str) -> float:
    """Validate an explicit scalar length and own a numeric mm magnitude."""
    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires a Pint quantity with length units")
    magnitude = normalize_unit(value, "mm", field=field, expected="length")
    array = np.asarray(magnitude)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar length quantity")
    with np.errstate(over="ignore", under="ignore"):
        stored = float(array)
    if not np.isfinite(stored) or (array != 0 and stored == 0.0):
        raise ValueError(f"{field} is not representable in numeric mm storage")
    return _positive(field, stored)


@dataclass(frozen=True, init=False)
class TransverseSpot:
    """Gaussian entrance spot, a zero-emittance waist.

    Parameters
    ----------
    fwhm_x_mm
        Full width at half maximum along x in mm.
    fwhm_y_mm
        FWHM along y in mm; ``None`` (default) makes the spot round.
    fwhm_x, fwhm_y
        Keyword-only, strictly quantity-valued alternative lengths. Each is
        mutually exclusive with the matching ``*_mm`` argument. ``fwhm_y``
        sets an explicit y width; omit it for a round spot.
    """

    fwhm_x_mm: float = 1.0
    fwhm_y_mm: float | None = None

    def __init__(
        self,
        fwhm_x_mm: float | _Unset = _UNSET,
        fwhm_y_mm: float | None | _Unset = _UNSET,
        *,
        fwhm_x: object = _UNSET,
        fwhm_y: object = _UNSET,
    ) -> None:
        if fwhm_x is not _UNSET:
            if fwhm_x_mm is not _UNSET:
                raise TypeError("TransverseSpot.fwhm_x_mm and fwhm_x are mutually exclusive")
            fwhm_x_mm = _spot_width_mm(fwhm_x, field="TransverseSpot.fwhm_x")
        elif fwhm_x_mm is _UNSET:
            fwhm_x_mm = 1.0
        if fwhm_y is not _UNSET:
            if fwhm_y_mm is not _UNSET:
                raise TypeError("TransverseSpot.fwhm_y_mm and fwhm_y are mutually exclusive")
            fwhm_y_mm = _spot_width_mm(fwhm_y, field="TransverseSpot.fwhm_y")
        elif fwhm_y_mm is _UNSET:
            fwhm_y_mm = None
        object.__setattr__(self, "fwhm_x_mm", fwhm_x_mm)
        object.__setattr__(self, "fwhm_y_mm", fwhm_y_mm)
        self.__post_init__()

    @property
    def fwhm_x(self) -> "Quantity":
        """Fresh x FWHM length quantity in mm, independent of stored fields."""
        from ..units import Q_

        return Q_(float(self.fwhm_x_mm), "mm")

    @property
    def fwhm_y(self) -> "Quantity":
        """Fresh effective y FWHM in mm; a round spot mirrors x."""
        from ..units import Q_

        return Q_(float(self.plane_y_mm), "mm")

    def with_fwhm_x(self, fwhm_x: object) -> Self:
        """Replace x FWHM via a quantity; preserve the implicit-round y."""
        return replace(
            self, fwhm_x_mm=_spot_width_mm(fwhm_x, field="TransverseSpot.fwhm_x")
        )

    def with_fwhm_y(self, fwhm_y: object) -> Self:
        """Replace y FWHM via a quantity; ``None`` restores roundness."""
        y_mm = (
            None
            if fwhm_y is None
            else _spot_width_mm(fwhm_y, field="TransverseSpot.fwhm_y")
        )
        return replace(self, fwhm_y_mm=y_mm)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fwhm_x_mm",
            normalize_unit(
                self.fwhm_x_mm, "mm", field="TransverseSpot.fwhm_x_mm", expected="length"
            ),
        )
        if self.fwhm_y_mm is not None:
            object.__setattr__(
                self,
                "fwhm_y_mm",
                normalize_unit(
                    self.fwhm_y_mm,
                    "mm",
                    field="TransverseSpot.fwhm_y_mm",
                    expected="length",
                ),
            )
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


# Compatibility spelling; TransverseSpot is the canonical public descriptor.
Spot = TransverseSpot


@dataclass(frozen=True)
class PointBunch:
    """Every electron arrives at t = 0."""


@dataclass(frozen=True, init=False)
class BunchLength:
    """Sampled arrival times with an RMS duration.

    Parameters
    ----------
    rms_fs
        RMS bunch duration in fs. Mutually exclusive with ``rms_duration``.
    rms_duration
        Keyword-only real scalar Pint time quantity, finite and positive.
        Read-back is a fresh quantity in fs; numeric storage remains ``rms_fs``.
    shape
        Sampling law name, ``"gaussian"`` by default.
    """

    rms_fs: float
    shape: str = "gaussian"

    def __init__(
        self,
        rms_fs: float | _Unset = _UNSET,
        shape: str = "gaussian",
        *,
        rms_duration: object = _UNSET,
    ) -> None:
        if rms_duration is not _UNSET:
            if rms_fs is not _UNSET:
                raise TypeError("BunchLength.rms_fs and rms_duration are mutually exclusive")
            rms_fs = _positive(
                "BunchLength.rms_duration",
                _timing_quantity_fs(rms_duration, field="BunchLength.rms_duration", ndim=0).item(),
            )
        elif rms_fs is _UNSET:
            raise TypeError("BunchLength requires rms_fs or rms_duration")
        object.__setattr__(self, "rms_fs", rms_fs)
        object.__setattr__(self, "shape", shape)
        self.__post_init__()

    @property
    def rms_duration(self) -> "Quantity":
        """RMS duration as a fresh scalar quantity in fs; no sampling occurs."""
        from ..units import Q_

        return Q_(float(self.rms_fs), "fs")

    def with_rms_duration(self, rms_duration: object) -> Self:
        """Replace duration using an explicit time quantity, preserving shape.

        ``dataclasses.replace`` continues to use the numeric ``rms_fs`` field;
        supplying ``rms_duration`` there conflicts with its forwarded field.
        """
        rms_fs = _positive(
            "BunchLength.rms_duration",
            _timing_quantity_fs(rms_duration, field="BunchLength.rms_duration", ndim=0).item(),
        )
        return replace(self, rms_fs=rms_fs)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rms_fs",
            normalize_unit(self.rms_fs, "fs", field="BunchLength.rms_fs", expected="time"),
        )
        _positive("rms_fs", self.rms_fs)
        if not isinstance(self.shape, str) or not self.shape:
            raise ValueError("shape must be a non-empty string")


@dataclass(frozen=True, init=False)
class Offsets:
    """Explicit per-electron arrival offsets, replacing random sampling.

    Parameters
    ----------
    offsets_fs
        Nonempty finite numeric offsets in fs; signed values and zero are valid.
        Mutually exclusive with ``arrival_offsets``.
    shape
        Retained longitudinal sampling-law label, default ``"gaussian"``.
    arrival_offsets
        Keyword-only Pint time quantity with a nonempty one-dimensional real
        array magnitude. Values must remain finite and representable in numeric
        fs storage. Read-back returns a fresh quantity with a copied array in fs.
        Numeric dataclass fields and per-electron ordering are unchanged.
    """

    offsets_fs: tuple[float, ...]
    shape: str = "gaussian"

    def __init__(
        self,
        offsets_fs: tuple[float, ...] | _Unset = _UNSET,
        shape: str = "gaussian",
        *,
        arrival_offsets: object = _UNSET,
    ) -> None:
        if arrival_offsets is not _UNSET:
            if offsets_fs is not _UNSET:
                raise TypeError("Offsets.offsets_fs and arrival_offsets are mutually exclusive")
            offsets_fs = tuple(
                _timing_quantity_fs(
                    arrival_offsets, field="Offsets.arrival_offsets", ndim=1
                ).tolist()
            )
        elif offsets_fs is _UNSET:
            raise TypeError("Offsets requires offsets_fs or arrival_offsets")
        object.__setattr__(self, "offsets_fs", offsets_fs)
        object.__setattr__(self, "shape", shape)
        self.__post_init__()

    @property
    def arrival_offsets(self) -> "Quantity":
        """Fresh quantity array of signed per-electron arrival offsets in fs."""
        from ..units import Q_

        return Q_(np.array(self.offsets_fs, dtype=float), "fs")

    def with_arrival_offsets(self, arrival_offsets: object) -> Self:
        """Replace explicit time offsets, preserving order and the shape label.

        Use numeric ``offsets_fs`` with ``dataclasses.replace``. Passing
        ``arrival_offsets`` there conflicts with the numeric field it forwards.
        """
        offsets = _timing_quantity_fs(arrival_offsets, field="Offsets.arrival_offsets", ndim=1)
        return replace(self, offsets_fs=tuple(offsets.tolist()))

    def __post_init__(self) -> None:
        offsets = normalize_unit(self.offsets_fs, "fs", field="Offsets.offsets_fs", expected="time")
        values = tuple(float(x) for x in offsets)
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
        for name, unit, expected in (
            ("z_origin_m", "m", "length"),
            ("time_s", "s", "time"),
            ("time_tolerance_s", "s", "time"),
            ("screen_position_m", "m", "length"),
            ("screen_tolerance_m", "m", "length"),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    normalize_unit(value, unit, field=f"GdfSource.{name}", expected=expected),
                )
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


Transverse = TransverseSpot | PointSpot | TransverseDistribution
Longitudinal = PointBunch | BunchLength | Offsets | LongitudinalDistribution


@dataclass(frozen=True, init=False)
class Beam:
    """Describe the electron beam at the crystal entrance plane.

    Each kind of beam property takes exactly one description, so the allowed
    combinations are the type signature rather than prose:

    >>> Beam(energy_keV=30.0, transverse=TransverseSpot(fwhm_x_mm=0.5), longitudinal=PointBunch())  # doctest: +SKIP
    >>> Beam(  # doctest: +SKIP
    ...     energy_keV=30.0, transverse=PointSpot(), longitudinal=BunchLength(rms_fs=100.0)
    ... )
    >>> Beam.from_gdf("beam.gdf", z_origin_m=0.0, time_s=1e-9)  # doctest: +SKIP

    Parameters
    ----------
    energy_keV
        Central kinetic energy in keV. A scalar creates one scene; campaign
        builders may expand a sequence. Compatible Pint quantities are converted
        to numeric keV. Mutually exclusive with ``kinetic_energy``; omitting
        both preserves the default grid ``(30, 45, 60)`` keV.
    kinetic_energy
        Keyword-only Pint energy quantity: a real scalar or one-dimensional
        array of finite positive values. Bare numbers and strings are rejected.
        Read-back returns a fresh quantity in keV; storage remains numeric.
    transverse
        Required: :class:`TransverseSpot`, :class:`PointSpot`, or a Courant--Snyder
        :class:`~pyrite.montecarlo.transverse.TransverseDistribution`.
    longitudinal
        Required: :class:`PointBunch`, :class:`BunchLength`, :class:`Offsets`, or a
        :class:`~pyrite.campaign.longitudinal.LongitudinalDistribution` policy.
    energy_spread_frac
        RMS fractional kinetic-energy spread; ``None`` is monoenergetic.
    rep_rate_hz
        Pulse repetition rate in Hz used for flux normalization.
    bunch_charge_pc
        Charge per bunch in pC used for flux normalization.
    gdf
        A :class:`GdfSource`; build through :meth:`from_gdf`.
    bunch_front_tilt
        Optional :class:`BunchFrontTilt` in degrees. Correlates incoming
        arrival time with lab transverse position without rotating the mean
        beam. Only monoenergetic Gaussian/point analytic sources are supported.

    Notes
    -----
    Sampling counts and execution controls belong to :class:`Numerics`.
    Divergence is derived from ``transverse`` and the energy; read it with
    :meth:`derived_divergence_mrad`.
    """

    energy_keV: Any = (30.0, 45.0, 60.0)
    transverse: Transverse = field(kw_only=True)
    longitudinal: Longitudinal = field(kw_only=True)
    energy_spread_frac: float | None = None
    rep_rate_hz: float = 5000.0
    bunch_charge_pc: float = 1.0
    gdf: GdfSource | None = None
    bunch_front_tilt: BunchFrontTilt | None = None
    state: TransverseStateSpec | None = field(default=None, kw_only=True)

    def __init__(
        self,
        energy_keV: Any = _UNSET,
        energy_spread_frac: float | None = None,
        rep_rate_hz: float = 5000.0,
        bunch_charge_pc: float = 1.0,
        gdf: GdfSource | None = None,
        bunch_front_tilt: BunchFrontTilt | None = None,
        *,
        transverse: Transverse,
        longitudinal: Longitudinal,
        state: TransverseStateSpec | None = None,
        kinetic_energy: object = _UNSET,
    ) -> None:
        if kinetic_energy is not _UNSET:
            if energy_keV is not _UNSET:
                raise TypeError("Beam.energy_keV and kinetic_energy are mutually exclusive")
            energy_keV = _kinetic_energy_keV(kinetic_energy)
        elif energy_keV is _UNSET:
            energy_keV = (30.0, 45.0, 60.0)
        values = {
            "energy_keV": energy_keV,
            "transverse": transverse,
            "longitudinal": longitudinal,
            "energy_spread_frac": energy_spread_frac,
            "rep_rate_hz": rep_rate_hz,
            "bunch_charge_pc": bunch_charge_pc,
            "gdf": gdf,
            "bunch_front_tilt": bunch_front_tilt,
            "state": state,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def kinetic_energy(self) -> "Quantity":
        """Configured kinetic energy as a fresh scalar/array quantity in keV.

        Array magnitudes are copied. This property does not override correlated
        GDF particle energies. Use ``energy_keV`` for numeric storage/lowering.
        """
        from ..units import Q_

        energy = np.asarray(self.energy_keV)
        magnitude = energy.copy() if energy.ndim else energy.item()
        return Q_(magnitude, "keV")

    def with_kinetic_energy(self, kinetic_energy: object) -> Self:
        """Return a beam with a validated scalar/array energy quantity.

        All other fields are preserved. ``dataclasses.replace`` supports the
        numeric ``energy_keV`` field; passing ``kinetic_energy`` to it conflicts
        with the numeric field it forwards to the constructor.
        """
        return replace(self, energy_keV=_kinetic_energy_keV(kinetic_energy))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_keV",
            normalize_unit(
                self.energy_keV,
                "keV",
                field="Beam.energy_keV",
                expected="energy",
            ),
        )
        object.__setattr__(
            self,
            "rep_rate_hz",
            normalize_unit(
                self.rep_rate_hz,
                "Hz",
                field="Beam.rep_rate_hz",
                expected="frequency",
            ),
        )
        object.__setattr__(
            self,
            "bunch_charge_pc",
            normalize_unit(
                self.bunch_charge_pc,
                "pC",
                field="Beam.bunch_charge_pc",
                expected="charge",
            ),
        )
        if self.state is not None and not isinstance(self.state, TransverseStateSpec):
            raise TypeError("Beam.state must be a TransverseStateSpec or None")
        if self.state is not None and self.gdf is not None:
            raise ValueError("Beam.state conflicts with a GDF source carrying its own phase space")
        if not isinstance(self.transverse, (TransverseSpot, PointSpot, TransverseDistribution)):
            raise TypeError(
                "Beam.transverse must be a TransverseSpot, PointSpot, or TransverseDistribution"
            )
        if not isinstance(
            self.longitudinal, (PointBunch, BunchLength, Offsets, LongitudinalDistribution)
        ):
            raise TypeError(
                "Beam.longitudinal must be a PointBunch, BunchLength, Offsets, or "
                "LongitudinalDistribution"
            )
        if self.bunch_front_tilt is not None and not isinstance(
            self.bunch_front_tilt, BunchFrontTilt
        ):
            raise TypeError("Beam.bunch_front_tilt must be a BunchFrontTilt")
        front = validate_front_source(
            self.bunch_front_tilt,
            transverse=self.transverse
            if isinstance(self.transverse, TransverseDistribution)
            else None,
            energy_spread=self.energy_spread_frac,
            source="gpt_gdf" if self.gdf is not None else "analytic",
            longitudinal=self.longitudinal
            if isinstance(self.longitudinal, LongitudinalDistribution)
            else None,
            offsets=self.longitudinal if isinstance(self.longitudinal, Offsets) else None,
            shape=self.longitudinal.shape
            if isinstance(self.longitudinal, BunchLength)
            else "gaussian",
        )
        object.__setattr__(self, "bunch_front_tilt", front)
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
        if self.state is not None:
            raise ValueError(
                "Beam.state cannot yet be lowered to transport: custom quantum "
                "entry-state projection is not yet integrated in production (#680)"
            )
        fields: dict[str, Any] = {
            "energy_keV": self.energy_keV,
            "energy_spread_frac": self.energy_spread_frac,
            "rep_rate_hz": self.rep_rate_hz,
            "bunch_charge_pc": self.bunch_charge_pc,
            "transverse_fwhm_x_mm": None,
            "transverse_fwhm_y_mm": None,
            "bunch_front_tilt": self.bunch_front_tilt,
        }
        transverse = self.transverse
        if isinstance(transverse, TransverseSpot):
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
                or spec.bunch_front_tilt is not None
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
            transverse = TransverseSpot(x, None if y == x else y)
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
            bunch_front_tilt=spec.bunch_front_tilt,
        )


__all__ = [
    "Beam",
    "BunchFrontTilt",
    "BunchLength",
    "GdfSource",
    "Offsets",
    "PointBunch",
    "PointSpot",
    "Spot",
    "TransverseSpot",
]
