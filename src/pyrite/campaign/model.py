"""Public scene, numerics, analysis, and path-addressed sweep objects."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

import numpy as np

from .._numerics import CxrReflections, Numerics, Physics
from .._numerics import EmissionMode as EmissionMode
from .._numerics import MosaicRoute as MosaicRoute
from .._precision import Precision
from ..detectors import Detector, EnergyBins
from ..instrument import Acquisition, Filter, FilterPlate, NativeAcquisition, PlanarDetector
from ..instrument.model import validate_downstream_scene
from .beam import Beam, BunchFrontTilt
from .geometry import Slab, Stack
from .geometry import Target as LegacyTarget
from .sweep import BeamSpec
from .target import Target

BremSource = Literal["mc", "external", "none"]

_SEGMENT = re.compile(r"(?P<name>[A-Za-z_]\w*)(?P<indexes>(?:\[\d+\])*)\Z")
_INDEX = re.compile(r"\[(\d+)\]")


def _one_float(name: str, value: Any) -> float:
    values = np.atleast_1d(np.asarray(value, dtype=float))
    if values.size != 1:
        raise ValueError(f"Scene requires scalar {name}; put multiple values in Sweep.axes")
    return float(values[0])


def _scalar_target(target: LegacyTarget | Target) -> LegacyTarget | Target:
    if isinstance(target, Target):
        return target
    if isinstance(target, Slab):
        return replace(
            target,
            thickness_ang=_one_float("target.thickness_ang", target.thickness_ang),
            tilt_deg=_one_float("target.tilt_deg", target.tilt_deg),
            tilt_azim_deg=_one_float("target.tilt_azim_deg", target.tilt_azim_deg),
        )
    if isinstance(target, Stack):
        layers = tuple(
            replace(
                layer,
                thickness_ang=_one_float(
                    f"target.layers[{index}].thickness_ang", layer.thickness_ang
                ),
            )
            for index, layer in enumerate(target.layers)
        )
        return replace(
            target,
            layers=layers,
            tilt_deg=_one_float("target.tilt_deg", target.tilt_deg),
            tilt_azim_deg=_one_float("target.tilt_azim_deg", target.tilt_azim_deg),
        )
    raise TypeError("Scene.target must be a Slab or Stack or Target")


@dataclass(frozen=True)
class Analysis:
    """Control presentation-time scaling of stored spectra.

    Parameters
    ----------
    beam_current_na
        Average beam current in nA used to convert per-electron yields to rates.
    apply_detector_qe
        Apply detector quantum efficiency during analysis.
    convolve_with_det
        Convolve spectra with detector energy resolution.
    """

    beam_current_na: float = 5.0
    apply_detector_qe: bool = False
    convolve_with_det: bool = False


@dataclass(frozen=True)
class Scene:
    """Collect one fully resolved physical simulation configuration.

    Parameters
    ----------
    beam
        Scalar electron-beam description. Express varying energies with
        :class:`Sweep`.
    target
        Scalar :class:`~pyrite.Target`, :class:`~pyrite.Slab` or :class:`~pyrite.Stack`.
    detector
        Scalar observation model or physical planar detector.
    filters
        Ordered :class:`~pyrite.Filter` or :class:`~pyrite.FilterPlate` instances
        between the source and a planar detector.
    physics
        Emission and particle-interaction models. Defaults to :class:`Physics`.
    numerics
        Sampling, true-energy grids, and execution controls. Used by sweeps
        unless their ``cases(numerics=...)`` call supplies an override.

    Raises
    ------
    TypeError
        If components have incompatible public API types.
    ValueError
        If geometry, scoring, or policy values are inconsistent.
    """

    beam: Beam
    target: LegacyTarget | Target
    detector: Detector | PlanarDetector = field(default_factory=Detector)
    filters: tuple[FilterPlate | Filter, ...] = ()
    acquisition: Acquisition | NativeAcquisition | None = None
    physics: Physics = field(default_factory=Physics, kw_only=True)
    numerics: Numerics = field(default_factory=Numerics, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.numerics, Numerics):
            raise TypeError("Scene.numerics must be a Numerics")
        if self.physics is not None and not isinstance(self.physics, Physics):
            raise TypeError("Scene.physics must be a Physics")
        physics = Physics() if self.physics is None else self.physics
        object.__setattr__(self, "physics", physics)
        if isinstance(self.beam, BeamSpec):
            object.__setattr__(self, "beam", Beam.from_spec(self.beam))
        if not isinstance(self.beam, Beam):
            raise TypeError("Scene.beam must be a Beam")
        if not isinstance(self.detector, (Detector, PlanarDetector)):
            raise TypeError("Scene.detector must be a Detector or PlanarDetector")
        try:
            filters = tuple(self.filters)
        except TypeError as exc:
            raise TypeError("Scene.filters must be an iterable of FilterPlate") from exc
        if any(not isinstance(plate, (FilterPlate, Filter)) for plate in filters):
            raise TypeError("Scene.filters must contain only FilterPlate or Filter objects")
        object.__setattr__(self, "filters", filters)
        if self.acquisition is not None and not isinstance(
            self.acquisition, (Acquisition, NativeAcquisition)
        ):
            raise TypeError("Scene.acquisition must be an Acquisition, native request, or None")
        if filters and not isinstance(self.detector, PlanarDetector):
            raise TypeError("Scene.filters require a physical PlanarDetector")
        if self.acquisition is not None:
            if not isinstance(self.detector, PlanarDetector) or self.detector.pixels is None:
                raise TypeError("Scene.acquisition requires a physical pixel detector")
            if self.detector.response is None:
                raise ValueError("Scene.acquisition requires an explicit detector response")
        if isinstance(self.detector, PlanarDetector):
            validate_downstream_scene(self.filter_plates, self.detector)
        object.__setattr__(
            self,
            "beam",
            replace(self.beam, energy_keV=_one_float("beam.energy_keV", self.beam.energy_keV)),
        )
        object.__setattr__(self, "target", _scalar_target(self.target))

    @property
    def target_spec(self) -> LegacyTarget:
        """Source geometry lowered after scalar body/sweep resolution."""
        return self.target.to_slab() if isinstance(self.target, Target) else self.target

    @property
    def filter_plates(self) -> tuple[FilterPlate, ...]:
        """Ordered legacy plates for observation and trajectory consumers."""
        return tuple(
            plate.to_plate() if isinstance(plate, Filter) else plate for plate in self.filters
        )

    def phase_matched_bunch_front_tilt(
        self,
        hkl: tuple[int, int, int],
        *,
        photon_energy_eV: float | None = None,
        observation_direction_lab: Sequence[float] | None = None,
    ) -> BunchFrontTilt:
        """Return the front cancelling transverse phase for one slab reflection.

        Uses the beam energy, resolved target angles and catalog orientation.
        Target angles follow the existing nearest-half-degree case resolution;
        the returned bunch-front angles retain full precision.
        ``hkl`` selects the reflection explicitly. By default the scalar
        detector's central direction is used; a planar detector requires an
        explicit lab-frame direction. A supplied direction is normalized.
        ``photon_energy_eV=None`` uses the vacuum, zero-scattering line estimate;
        pass the desired photon energy to match a refractively shifted peak.

        Matches the mean crystal orientation at one energy and direction, not
        every mosaic node or detector ray. Residual timing spread is unchanged.
        Requires the supported monoenergetic point/Gaussian source and a flat
        single slab; other source compositions and stacks raise ValueError.
        The scene is unchanged; assign the returned descriptor to a new Beam.

        Source: joint Gaussian phase mismatch K-omega J^-T a=0, hence
        a=J.T@(n_xy-b_xy/beta0+g_xy/omega). Zero mismatch gives zero tilt.
        Validation: linear-bunch-front-tilt, line-energy-dispersion
        """
        from .front_matching import phase_matched_front_for_scene

        return phase_matched_front_for_scene(self, hkl, photon_energy_eV, observation_direction_lab)


def _parts(path: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not path:
        raise ValueError("axis path must not be empty")
    parts = []
    for raw in path.split("."):
        match = _SEGMENT.fullmatch(raw)
        if match is None:
            raise ValueError(f"invalid axis path {path!r}: malformed segment {raw!r}")
        parts.append(
            (
                match.group("name"),
                tuple(int(value) for value in _INDEX.findall(match.group("indexes"))),
            )
        )
    return tuple(parts)


def _descend(value: Any, name: str, indexes: tuple[int, ...], path: str) -> Any:
    if not hasattr(value, name):
        raise ValueError(
            f"invalid axis path {path!r}: {type(value).__name__} has no field {name!r}"
        )
    child = getattr(value, name)
    for index in indexes:
        if not isinstance(child, Sequence) or isinstance(child, (str, bytes)):
            raise ValueError(f"invalid axis path {path!r}: {name!r} is not indexable")
        if index >= len(child):
            raise ValueError(f"invalid axis path {path!r}: index {index} is out of range")
        child = child[index]
    return child


def _replace_indexes(container: Any, indexes: tuple[int, ...], leaf: Any, path: str) -> Any:
    if not indexes:
        return leaf
    if not isinstance(container, Sequence) or isinstance(container, (str, bytes)):
        raise ValueError(f"invalid axis path {path!r}: value is not indexable")
    index = indexes[0]
    if index >= len(container):
        raise ValueError(f"invalid axis path {path!r}: index {index} is out of range")
    values = list(container)
    values[index] = _replace_indexes(values[index], indexes[1:], leaf, path)
    return tuple(values) if isinstance(container, tuple) else values


def _replace_path(
    value: Any, parts: tuple[tuple[str, tuple[int, ...]], ...], leaf: Any, path: str
) -> Any:
    name, indexes = parts[0]
    child = _descend(value, name, indexes, path)
    replacement = leaf if len(parts) == 1 else _replace_path(child, parts[1:], leaf, path)
    replacement = _replace_indexes(getattr(value, name), indexes, replacement, path)
    try:
        return replace(value, **{name: replacement})
    except TypeError as exc:
        raise ValueError(f"invalid axis path {path!r}: {name!r} is not replaceable") from exc


def _axis_label(path: str, value: Any) -> str:
    if isinstance(value, float):
        rendered = f"{value:g}"
    else:
        rendered = str(value)
    return f"{path}={rendered}"


@dataclass(frozen=True)
class Sweep:
    """Define a Cartesian product over fields of a scalar base scene.

    Parameters
    ----------
    base
        Scalar scene copied for each product point.
    axes
        Ordered mapping from dotted dataclass paths to non-empty replacement
        sequences. Indexed segments address sequence elements, for example
        ``"target.layers[1].thickness_ang"``.

    Raises
    ------
    TypeError
        If ``base`` or the axis collections have invalid types.
    ValueError
        If an axis is empty or a path cannot be resolved and replaced.
    """

    base: Scene
    axes: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    _legacy_source: tuple[Any, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.base, Scene):
            raise TypeError("Sweep.base must be a Scene")
        normalized: dict[str, tuple[Any, ...]] = {}
        for path, raw_values in self.axes.items():
            if isinstance(raw_values, (str, bytes)):
                raise TypeError(f"axis {path!r} values must be a non-string iterable")
            try:
                values = tuple(raw_values)
            except TypeError as exc:
                raise TypeError(f"axis {path!r} values must be a non-string iterable") from exc
            if not values:
                raise ValueError(f"axis {path!r} must contain at least one value")
            if path.startswith("numerics.energy_bins.") and self.base.numerics.energy_bins is None:
                object.__setattr__(
                    self,
                    "base",
                    replace(
                        self.base, numerics=replace(self.base.numerics, energy_bins=EnergyBins())
                    ),
                )
            parsed = _parts(path)
            _replace_path(self.base, tuple(parsed), values[0], path)
            if path in normalized and (
                len(normalized[path]) != len(values)
                or any(
                    not np.array_equal(old, new)
                    for old, new in zip(normalized[path], values, strict=True)
                )
            ):
                raise ValueError(f"conflicting old and new axis paths for {path!r}")
            normalized[path] = values
        object.__setattr__(self, "axes", normalized)

    def expand(self) -> tuple[tuple[str, Scene], ...]:
        """Return labeled scenes in Cartesian-product order.

        Returns
        -------
        tuple of tuple
            ``(label, scene)`` pairs in axis insertion order. With no axes, the
            base scene is returned with an empty label.
        """
        if not self.axes:
            return (("", self.base),)
        paths = tuple(self.axes)
        expanded = []
        for values in product(*(self.axes[path] for path in paths)):
            scene = self.base
            for path, value in zip(paths, values, strict=True):
                scene = _replace_path(scene, _parts(path), value, path)
            expanded.append(
                (
                    " ".join(
                        _axis_label(path, value) for path, value in zip(paths, values, strict=True)
                    ),
                    scene,
                )
            )
        return tuple(expanded)

    def cases(self, numerics: Numerics | None = None) -> list[Any]:
        """Lower every expanded scene without executing or persisting it.

        Parameters
        ----------
        numerics
            Sampling and execution controls; overrides the base scene before axes apply.

        Returns
        -------
        list of pyrite.montecarlo.Case
            Cases in the same order as :meth:`expand`.
        """
        from .lowering import build_sweep_cases

        return build_sweep_cases(self, numerics)


__all__ = [
    "Analysis",
    "Beam",
    "CxrReflections",
    "Physics",
    "Numerics",
    "Precision",
    "Scene",
    "Sweep",
]
