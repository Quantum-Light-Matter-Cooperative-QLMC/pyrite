"""Public scene, numerics, analysis, and path-addressed sweep objects."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

import numpy as np

from ..detectors import Detector
from .geometry import Slab, Stack, Target
from .sweep import BeamSpec

EmissionMode = Literal["incoherent", "coherent", "both"]
XrayDispersion = Literal["vacuum", "refractive"]
BremSource = Literal["mc", "external", "none"]
MosaicRoute = Literal["analytic", "mc"]

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
class Convergence:
    """Result-affecting convergence and truncation controls."""

    n_families: int = 4
    max_reflections: int | None = None
    mosaic_nodes: int = 5
    mosaic_route: MosaicRoute = "analytic"

    def __post_init__(self) -> None:
        if self.n_families <= 0:
            raise ValueError("n_families must be positive")
        if self.max_reflections is not None and self.max_reflections <= 0:
            raise ValueError("max_reflections must be positive or None")
        if self.mosaic_nodes <= 0:
            raise ValueError("mosaic_nodes must be positive")
        if self.mosaic_route not in {"analytic", "mc"}:
            raise ValueError("mosaic_route must be 'analytic' or 'mc'")


@dataclass(frozen=True)
class Numerics:
    """Sampling and result-invariant execution controls."""

    n_electrons: int = 450
    n_electrons_brem: int = 100
    spec_chunk: int | None = None
    brem_chunk: int | None = None
    transport_core: str = "auto"
    backend: str = "auto"
    convergence: Convergence = field(default_factory=Convergence)

    def __post_init__(self) -> None:
        for name in ("n_electrons", "n_electrons_brem"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("spec_chunk", "brem_chunk"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")


@dataclass(frozen=True)
class Analysis:
    """Presentation and compatibility scaling controls."""

    beam_current_na: float = 5.0
    apply_detector_qe: bool = False
    convolve_with_det: bool = False


@dataclass(frozen=True)
class Scene:
    """One fully resolved physical configuration."""

    beam: BeamSpec
    target: Target
    detector: Detector = field(default_factory=Detector)
    emission: EmissionMode = "incoherent"
    xray_dispersion: XrayDispersion = "vacuum"
    brem_source: BremSource = "mc"

    def __post_init__(self) -> None:
        if not isinstance(self.beam, BeamSpec):
            raise TypeError("Scene.beam must be a BeamSpec")
        if not isinstance(self.detector, Detector):
            raise TypeError("Scene.detector must be a Detector")
        object.__setattr__(
            self,
            "beam",
            replace(self.beam, energy_keV=_one_float("beam.energy_keV", self.beam.energy_keV)),
        )
        object.__setattr__(self, "target", _scalar_target(self.target))
        if self.emission not in {"incoherent", "coherent", "both"}:
            raise ValueError("emission must be 'incoherent', 'coherent', or 'both'")
        if self.xray_dispersion not in {"vacuum", "refractive"}:
            raise ValueError("xray_dispersion must be 'vacuum' or 'refractive'")
        if self.brem_source not in {"mc", "external", "none"}:
            raise ValueError("brem_source must be 'mc', 'external', or 'none'")


def _parts(path: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not path:
        raise ValueError("axis path must not be empty")
    parts = []
    for raw in path.split("."):
        match = _SEGMENT.fullmatch(raw)
        if match is None:
            raise ValueError(f"invalid axis path {path!r}: malformed segment {raw!r}")
        parts.append(
            (match.group("name"), tuple(int(value) for value in _INDEX.findall(match.group("indexes"))))
        )
    return tuple(parts)


def _descend(value: Any, name: str, indexes: tuple[int, ...], path: str) -> Any:
    if not hasattr(value, name):
        raise ValueError(f"invalid axis path {path!r}: {type(value).__name__} has no field {name!r}")
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


def _replace_path(value: Any, parts: tuple[tuple[str, tuple[int, ...]], ...], leaf: Any, path: str) -> Any:
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
    """A base scene plus ordered axes addressed by dotted field paths."""

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
            parsed = _parts(path)
            _replace_path(self.base, parsed, values[0], path)
            normalized[path] = values
        object.__setattr__(self, "axes", normalized)

    def expand(self) -> tuple[tuple[str, Scene], ...]:
        """Return mechanically labeled scenes in Cartesian-product order."""
        if not self.axes:
            return (("", self.base),)
        paths = tuple(self.axes)
        expanded = []
        for values in product(*(self.axes[path] for path in paths)):
            scene = self.base
            for path, value in zip(paths, values, strict=True):
                scene = _replace_path(scene, _parts(path), value, path)
            expanded.append((" ".join(_axis_label(path, value) for path, value in zip(paths, values, strict=True)), scene))
        return tuple(expanded)

    @classmethod
    def from_legacy(cls, old_sweep: Any, settings: Any) -> Sweep:
        """Convert the D7 `campaign.sweep.Sweep`/`Settings` pair."""
        from .legacy import adapt_legacy_sweep

        warnings.warn(
            "Sweep.from_legacy() is a compatibility bridge for the D7 support window",
            DeprecationWarning,
            stacklevel=2,
        )
        return adapt_legacy_sweep(old_sweep, settings)


__all__ = ["Analysis", "Convergence", "Numerics", "Scene", "Sweep"]
