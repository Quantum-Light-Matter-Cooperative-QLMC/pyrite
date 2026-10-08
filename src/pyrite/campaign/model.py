"""Public scene, numerics, analysis, and path-addressed sweep objects."""

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

import numpy as np

from .._numerics import Convergence, Numerics
from .._numerics import MosaicRoute as MosaicRoute
from ..detectors import Detector
from ..instrument import Acquisition, FilterPlate, PixelScorer, PlanarDetector
from ..instrument.model import validate_downstream_scene
from .geometry import Slab, Stack, Target
from .sweep import BeamSpec

EmissionMode = Literal["incoherent", "coherent", "both"]
BremSource = Literal["mc", "external", "none"]

#: Removal target for the public `Sweep.from_legacy()` bridge.
#:
#: The bridge previously cited "the D7 support window" with no end date, which
#: is not a schedule. 0.3.0 is the first release that names a target, so the
#: two-minor window runs to 0.5.0 -- the same clock the module re-export cohort
#: restarted on for the same reason. `tests/test_deprecation_schedule.py` holds
#: this constant to the shipping `__version__` alongside the three registries.
FROM_LEGACY_REMOVE_IN = "0.5.0"

_SEGMENT = re.compile(r"(?P<name>[A-Za-z_]\w*)(?P<indexes>(?:\[\d+\])*)\Z")
_INDEX = re.compile(r"\[(\d+)\]")


class Beam(BeamSpec):
    """Describe the electron beam at the crystal entrance plane.

    Parameters
    ----------
    energy_keV
        Central kinetic energy in keV. A scalar creates one scene; campaign
        builders may expand a sequence.
    transverse_fwhm_x_mm, transverse_fwhm_y_mm
        Gaussian entrance-spot FWHM values in mm. Set both to ``None`` for a
        point source. Mutually exclusive with ``transverse``.
    bunch_length_fs
        RMS bunch duration in fs. ``None`` selects a point bunch.
    long_shape
        Sampling law for ``bunch_length_fs``.
    long_offsets_fs
        Explicit per-electron arrival offsets in fs, replacing random sampling.
    longitudinal
        Declarative longitudinal-distribution policy. Mutually exclusive with
        the legacy flat longitudinal fields.
    transverse
        Declarative Courant--Snyder transverse-distribution policy.
    rep_rate_hz
        Pulse repetition rate in Hz used for flux normalization.
    bunch_charge_pc
        Charge per bunch in pC used for flux normalization.
    divergence_mrad
        Deprecated input placeholder. Divergence is derived from
        ``transverse`` and the case energy; supplying a value is rejected.
    energy_spread_frac
        RMS fractional kinetic-energy spread; ``None`` is monoenergetic.

    source, gdf_path, gdf_time_s, gdf_time_tolerance_s
        ``source="gpt_gdf"`` loads a native GPT time-output snapshot or selected screen. Its
        correlated energies replace ``energy_keV`` during case construction.
        Default absolute time tolerance is 1e-15 seconds.
    gdf_shape_only
        Preserve imported positions/directions/weights, assign each sweep energy,
        and discard imported crossing times. Default False imports energies.
    gdf_normalization
        ``pyrite_current`` retains configured source normalization;
        ``gdf_charge`` derives current from absolute bunch charge and the
        resolved shared ``rep_rate_hz`` (finite and positive). File-derived
        charge replaces configured ``bunch_charge_pc`` in this mode.
    gdf_z_origin_m
        Required explicit target-origin lab z coordinate in meters. GPT axes
        are PyRITE lab axes; individual rays project onto the tilted entrance.
        Clear both transverse FWHMs and omit analytic phase-space policies.

    Notes
    -----
    Sampling counts and execution controls belong to :class:`Numerics`.
    """


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
        Optional request for factorized spatial scoring on a detector grid.
    emission
        ``"incoherent"``, ``"coherent"``, or ``"both"``.
    brem_source
        ``"mc"``, ``"external"``, or ``"none"``. The high-level API returns
        zeros for the latter two because it accepts no external spectrum.

    Raises
    ------
    TypeError
        If components have incompatible public API types.
    ValueError
        If geometry, scoring, or policy values are inconsistent.
    """

    beam: BeamSpec
    target: Target
    detector: Detector | PlanarDetector = field(default_factory=Detector)
    filters: tuple[FilterPlate, ...] = ()
    pixel_scorer: PixelScorer | None = None
    acquisition: Acquisition | None = None
    emission: EmissionMode = "incoherent"
    brem_source: BremSource = "mc"

    def __post_init__(self) -> None:
        if not isinstance(self.beam, BeamSpec):
            raise TypeError("Scene.beam must be a BeamSpec")
        if not isinstance(self.detector, (Detector, PlanarDetector)):
            raise TypeError("Scene.detector must be a Detector or PlanarDetector")
        try:
            filters = tuple(self.filters)
        except TypeError as exc:
            raise TypeError("Scene.filters must be an iterable of FilterPlate") from exc
        if any(not isinstance(plate, FilterPlate) for plate in filters):
            raise TypeError("Scene.filters must contain only FilterPlate objects")
        object.__setattr__(self, "filters", filters)
        if self.pixel_scorer is not None and not isinstance(self.pixel_scorer, PixelScorer):
            raise TypeError("Scene.pixel_scorer must be a PixelScorer or None")
        if self.acquisition is not None and not isinstance(self.acquisition, Acquisition):
            raise TypeError("Scene.acquisition must be an Acquisition or None")
        if filters and not isinstance(self.detector, PlanarDetector):
            raise TypeError("Scene.filters require a physical PlanarDetector")
        if self.pixel_scorer is not None:
            if not isinstance(self.detector, PlanarDetector):
                raise TypeError("Scene.pixel_scorer requires a physical PlanarDetector")
            if self.detector.pixels is None:
                raise ValueError("Scene.pixel_scorer requires PlanarDetector.pixels")
            if any(
                requested > available
                for requested, available in zip(
                    self.pixel_scorer.angular_shape,
                    self.detector.pixels.shape,
                    strict=True,
                )
            ):
                raise ValueError("pixel scorer angular shape cannot exceed detector pixel shape")
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
        if self.emission not in {"incoherent", "coherent", "both"}:
            raise ValueError("emission must be 'incoherent', 'coherent', or 'both'")
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
            parsed = _parts(path)
            _replace_path(self.base, parsed, values[0], path)
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
            Sampling and convergence controls. Defaults to :class:`Numerics`.

        Returns
        -------
        list of pyrite.montecarlo.Case
            Cases in the same order as :meth:`expand`.
        """
        from .lowering import build_sweep_cases

        return build_sweep_cases(self, numerics)

    @classmethod
    def from_legacy(cls, old_sweep: Any, settings: Any) -> Sweep:
        """Convert the D7 `campaign.sweep.Sweep`/`Settings` pair."""
        from .legacy import adapt_configured_sweep

        warnings.warn(
            f"Sweep.from_legacy() is deprecated and will be removed in "
            f"{FROM_LEGACY_REMOVE_IN}; use pyrite.api.build_configured_cases("
            f"old_sweep, settings) to lower the pair to cases, or build a Sweep "
            f"directly from a Scene",
            DeprecationWarning,
            stacklevel=2,
        )
        return adapt_configured_sweep(old_sweep, settings)


__all__ = ["Analysis", "Beam", "Convergence", "Numerics", "Scene", "Sweep"]
