"""Public scene, numerics, analysis, and path-addressed sweep objects."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import wraps
from itertools import product
from typing import Any, Literal

import numpy as np

from .._api_deprecations import _hide_compat_parameters, warn_api
from .._numerics import Convergence, CxrReflections, Numerics, Physics, _emission_selection
from .._numerics import EmissionMode as EmissionMode
from .._numerics import MosaicRoute as MosaicRoute
from .._precision import Precision
from ..detectors import Detector, EnergyBins
from ..instrument import Acquisition, FilterPlate, PixelScorer, PlanarDetector
from ..instrument.model import validate_downstream_scene
from .beam import Beam
from .geometry import Slab, Stack, Target
from .sweep import BeamSpec

BremSource = Literal["mc", "external", "none"]

_SEGMENT = re.compile(r"(?P<name>[A-Za-z_]\w*)(?P<indexes>(?:\[\d+\])*)\Z")
_INDEX = re.compile(r"\[(\d+)\]")


def _one_float(name: str, value: Any) -> float:
    values = np.atleast_1d(np.asarray(value, dtype=float))
    if values.size != 1:
        raise ValueError(f"Scene requires scalar {name}; put multiple values in Sweep.axes")
    return float(values[0])


def _scalar_target(target: Target) -> Target:
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
    raise TypeError("Scene.target must be a Slab or Stack")


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


def _scene_policy(cls):
    """Forward deprecated scene policies without warning again on copies."""
    original = cls.__init__

    @wraps(original)
    def initialize(self, *args, **kwargs):
        if len(args) > 6:
            if len(args) > 8:
                raise TypeError("Scene accepts at most eight positional arguments")
            for name, value in zip(("emission", "brem_source"), args[6:], strict=False):
                if name in kwargs:
                    raise TypeError(f"multiple values for {name}")
                kwargs[name] = value
            args = args[:6]
        physics = kwargs.get("physics")
        policies = dict(kwargs.get("_policy_overrides", ()))
        explicit = kwargs.get("_physics_explicit")
        if explicit is None:
            explicit = physics is not None
        elif not explicit and physics is not None and physics != replace(Physics(), **policies):
            explicit = True
        for name in ("emission", "brem_source"):
            if name in kwargs:
                value = kwargs.pop(name)
                if name == "emission":
                    value = _emission_selection(value)
                policies[name] = value
                warn_api(f"Scene({name}=)", stacklevel=3)
                if physics is not None and explicit and getattr(physics, name) != value:
                    raise ValueError(f"conflicting Scene.{name} and Physics.{name}")
                physics = replace(Physics() if physics is None else physics, **{name: value})
        kwargs["physics"] = Physics() if physics is None else physics
        kwargs["_physics_explicit"] = explicit
        kwargs["_policy_overrides"] = tuple(policies.items())
        original(self, *args, **kwargs)

    _hide_compat_parameters(initialize, original)
    cls.__init__ = initialize
    cls.emission = property(lambda self: self.physics.emission_policy)
    cls.brem_source = property(lambda self: self.physics.brem_source)
    return cls


@_scene_policy
@dataclass(frozen=True)
class Scene:
    """Collect one fully resolved physical simulation configuration.

    Parameters
    ----------
    beam
        Scalar electron-beam description. Express varying energies with
        :class:`Sweep`.
    target
        Scalar :class:`~pyrite.Slab` or :class:`~pyrite.Stack` target.
    detector
        Scalar observation model or physical planar detector.
    filters
        Ordered filter plates between the source and a planar detector.
    pixel_scorer
        Deprecated: set :attr:`PlanarDetector.scorer` instead. A value given
        here is moved onto the detector and this field reads ``None``.
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
    target: Target
    detector: Detector | PlanarDetector = field(default_factory=Detector)
    filters: tuple[FilterPlate, ...] = ()
    pixel_scorer: PixelScorer | None = None
    acquisition: Acquisition | None = None
    physics: Physics = field(default_factory=Physics, kw_only=True)
    numerics: Numerics = field(default_factory=Numerics, kw_only=True)
    _physics_explicit: bool | None = field(default=None, repr=False, compare=False, kw_only=True)
    _policy_overrides: tuple[tuple[str, Any], ...] = field(
        default=(), repr=False, compare=False, kw_only=True
    )

    def __post_init__(self) -> None:
        if not isinstance(self.numerics, Numerics):
            raise TypeError("Scene.numerics must be a Numerics")
        if self._physics_explicit is None:
            object.__setattr__(self, "_physics_explicit", self.physics is not None)
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
        if any(not isinstance(plate, FilterPlate) for plate in filters):
            raise TypeError("Scene.filters must contain only FilterPlate objects")
        object.__setattr__(self, "filters", filters)
        if self.pixel_scorer is not None:
            # stacklevel 4: warn_api -> __post_init__ -> __init__ -> the caller.
            warn_api("Scene(pixel_scorer=)", stacklevel=4)
            object.__setattr__(
                self, "detector", detector_with_scorer(self.detector, self.pixel_scorer)
            )
            object.__setattr__(self, "pixel_scorer", None)
        if self.acquisition is not None and not isinstance(self.acquisition, Acquisition):
            raise TypeError("Scene.acquisition must be an Acquisition or None")
        if filters and not isinstance(self.detector, PlanarDetector):
            raise TypeError("Scene.filters require a physical PlanarDetector")
        if self.acquisition is not None:
            if not isinstance(self.detector, PlanarDetector) or self.detector.pixels is None:
                raise TypeError("Scene.acquisition requires a physical pixel detector")
            if self.detector.response is None:
                raise ValueError("Scene.acquisition requires an explicit detector response")
        if isinstance(self.detector, PlanarDetector):
            validate_downstream_scene(filters, self.detector)
        object.__setattr__(
            self,
            "beam",
            replace(self.beam, energy_keV=_one_float("beam.energy_keV", self.beam.energy_keV)),
        )
        object.__setattr__(self, "target", _scalar_target(self.target))


def detector_with_scorer(detector: Any, scorer: Any) -> PlanarDetector:
    """Move a deprecated loose ``pixel_scorer`` onto its pixel detector (#451)."""
    if not isinstance(scorer, PixelScorer):
        raise TypeError("pixel_scorer must be a PixelScorer or None")
    if not isinstance(detector, PlanarDetector):
        raise TypeError("pixel_scorer requires a physical PlanarDetector")
    if detector.pixels is None:
        raise ValueError("pixel_scorer requires PlanarDetector.pixels")
    if detector.scorer not in (None, PixelScorer(), scorer):
        raise ValueError(
            "pixel_scorer conflicts with PlanarDetector.scorer; set only PlanarDetector.scorer"
        )
    return replace(detector, scorer=scorer)


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
        if isinstance(value, Scene) and name == "physics":
            return replace(value, physics=replacement, _physics_explicit=True)
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
            if path.startswith("numerics.") and path.split(".")[1] in Physics.__dataclass_fields__:
                name = path.split(".")[1]
                warn_api(f"Numerics({name}=)", stacklevel=4)
                path = "physics." + path.removeprefix("numerics.")
            if path.startswith("numerics.convergence"):
                warn_api("Numerics(convergence=)", stacklevel=4)
                path = path.replace("numerics.convergence", "numerics.cxr_reflections", 1)
            if path.startswith("detector.energy_bins"):
                warn_api(f"{type(self.base.detector).__name__}(energy_bins=)", stacklevel=4)
                object.__setattr__(
                    self,
                    "base",
                    replace(
                        self.base,
                        numerics=replace(
                            self.base.numerics,
                            energy_bins=(
                                self.base.detector.energy_bins
                                if self.base.numerics.energy_bins is None
                                else self.base.numerics.energy_bins
                            ),
                        ),
                        detector=replace(self.base.detector, _energy_bins_explicit=False),
                    ),
                )
                path = path.replace("detector.energy_bins", "numerics.energy_bins", 1)
            if path in {"emission", "brem_source"}:
                warn_api(f"Scene({path}=)", stacklevel=4)
                path = "physics." + path
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
    "Convergence",
    "CxrReflections",
    "Physics",
    "Numerics",
    "Precision",
    "Scene",
    "Sweep",
]
