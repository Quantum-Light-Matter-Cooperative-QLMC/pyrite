"""Target geometry: the closed set of target variants and their lowering to
case keys.

A target is exactly one of :class:`Slab` or :class:`Stack`, optionally bounded
by a :class:`Footprint` and -- slab only -- machined with a
:class:`BlazedGrooves` entrance face. That closed set is deliberate: arbitrary
geometry is a recorded non-goal, so there is no dispatch protocol, no region
algebra, and no user-supplied target class.

Each variant owns its own validity at construction, so an invalid target raises
in the user's script rather than after profile resolution, and :meth:`Slab.lower`
/ :meth:`Stack.lower` produce exactly the case keys that exist today --
``abs_layers``, ``layer_radiators``, ``crystal_width_mm``, ``crystal_height_mm``
and ``groove_spacing_ang``. Transport reads the same payload it always did.

Fields are ``ScalarOrSeq`` because :func:`pyrite.campaign.sweep.build_cases`
still takes the Cartesian product: a target is a sweep TEMPLATE here, not a
single geometry. Narrowing to scalars belongs with the dotted-axis machinery of
``refactor/scene-object-model``.

The material and layer resolution helpers that lowering needs live here too, so
this module stays below :mod:`pyrite.campaign.sweep` and imports without it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ..detectors import Detector
from ..materials import CATALOG, LayerSpec
from ..materials.crystal import dominant_reflections

ScalarOrSeq = float | Sequence[float] | np.ndarray


def _seq(x):
    """Normalize a scalar-or-sequence into a 1-D float array, order preserved."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def _quantized_angles(values: ScalarOrSeq) -> np.ndarray:
    """Round degrees to nearest half away from zero at ties, then stable-unique."""
    source = _seq(values)
    scaled = source * 2.0
    quantized = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled) / 2.0
    return np.asarray(list(dict.fromkeys(float(value) for value in quantized)), dtype=float)


def fmt_thickness(t_ang):
    """Compact human thickness label from Angstroms: 316A / 31.6nm / 17um / 1mm."""
    if t_ang < 1e2:
        return f"{t_ang:g}A"
    if t_ang < 1e4:
        return f"{t_ang / 10:g}nm"
    if t_ang < 1e7:
        return f"{t_ang / 1e4:g}um"
    return f"{t_ang / 1e7:g}mm"


def substrate_composition(substrate):
    """Number-density composition [(element, n_per_Ang3), ...] for a substrate.
    Amorphous presets ('sio2') come from bulk density; a crystalline
    substrate already in CRYSTALS (e.g. 'silicon') uses its unit-cell density."""
    key = substrate.lower()
    if key in CATALOG.media:
        return list(CATALOG.media[key].composition)
    if substrate in CATALOG.crystals:
        return list(CATALOG.crystal(substrate).composition)
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(CATALOG.media)} "
        f"or a crystal key in {list(CATALOG.crystals)}"
    )


def stack_layers(film_composition, film_thickness_ang, stack):
    """Absorber stack [(z_top, z_bot, composition), ...] for a film at the
    entrance face (z=0..t_film) followed by each :class:`LayerSpec` in ``stack``,
    boundaries accumulating downward. Attach as a case's ``abs_layers`` so
    emitted lines/brem are attenuated by the WHOLE stack (each crystalline
    layer still RADIATES via its own layer_radiator). Beam enters the film
    side; with positive tilt (toward-detector, Zhai convention; front exit)
    the lower layers sit BEHIND the emission and do not attenuate -- they
    bite the back-exit / transmission geometry. See
    docs/physics/materials/multilayer-materials.md."""
    t_f = float(film_thickness_ang)
    layers = [(0.0, t_f, [(el, float(n)) for el, n in film_composition])]
    z = t_f
    for lay in stack:
        t = float(lay.thickness_ang)
        layers.append((z, z + t, substrate_composition(lay.material)))
        z += t
    return layers


def film_on_substrate_layers(
    film_composition, film_thickness_ang, substrate, substrate_thickness_ang
):
    """The 2-layer special case of :func:`stack_layers`: one film on one
    substrate (kept as the stable public name for that common stack)."""
    return stack_layers(
        film_composition,
        film_thickness_ang,
        (LayerSpec(substrate, substrate_thickness_ang),),
    )


def _radiator(cp, *, beam_uvw=None, azimuth_rad=None):
    """Coherent-radiator dict from a :func:`crystal_params` result ``cp``: crystal,
    hkl_list, B_ang2, and its mutually exclusive beam_uvw/surface_hkl orientation.
    An explicit beam_uvw overrides and clears ``cp``'s reciprocal surface. The
    ``azimuth_rad`` key is included only when given -- a bare substrate radiator
    carries no azimuth of its own (:func:`substrate_radiator`); only a stack/film
    use of a radiator (:func:`layer_radiator`, :meth:`Stack.lower`) does. The one
    constructor behind all three radiator-dict call sites."""
    rad: dict[str, Any] = dict(
        crystal=cp["crystal"],
        hkl_list=cp["hkl_list"],
        B_ang2=cp["B_ang2"],
        beam_uvw=cp["beam_uvw"] if beam_uvw is None else tuple(beam_uvw),
        surface_hkl=cp["surface_hkl"] if beam_uvw is None else None,
    )
    if azimuth_rad is not None:
        rad["azimuth_rad"] = float(azimuth_rad)
    return rad


def substrate_radiator(substrate, n_families=4):
    """Coherent-radiation crystal params for a substrate, or None if it radiates
    no lines. A CRYSTALLINE substrate (a CRYSTALS key, e.g. 'silicon') returns
    {crystal, hkl_list, B_ang2, beam_uvw, surface_hkl} (from crystal_params) so it emits its
    own PXR/CBS; an AMORPHOUS preset ('sio2') returns None (it only
    absorbs + brems). This is the per-layer-radiation half of the multilayer
    feature -- the absorber stack (film_on_substrate_layers) is the other half.
    See docs/physics/materials/multilayer-materials.md."""
    if substrate.lower() in CATALOG.media:
        return None  # amorphous: no coherent lines
    if substrate in CATALOG.crystals:
        return _radiator(crystal_params(substrate, n_families))
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(CATALOG.media)} "
        f"or a crystal key in {list(CATALOG.crystals)}"
    )


def layer_radiator(layer: LayerSpec, n_families: int = 4):
    """Coherent radiator params for one stack :class:`LayerSpec`, or None if the
    layer is amorphous. Same dict as :func:`substrate_radiator` plus the
    per-layer orientation: ``beam_uvw`` (overridden if the spec sets one, clearing
    any catalog surface_hkl) and ``azimuth_rad`` (the spec's in-plane rotation,
    radians)."""
    rad = substrate_radiator(layer.material, n_families)
    if rad is None:
        return None
    if layer.beam_uvw is not None:
        rad["beam_uvw"] = tuple(layer.beam_uvw)
        rad["surface_hkl"] = None
    rad["azimuth_rad"] = float(np.deg2rad(layer.azimuth_deg))
    return rad


def crystal_params(material: str, n_families: int = 4) -> dict[str, Any]:
    """Fixed crystallography for a material: composition, the dominant
    reflections, exactly one direct beam axis [uvw] or reciprocal surface (hkl),
    the isotropic B-factor, and a sensible default photon-energy grid. Override
    the grid via ``Sweep.detector.energy_bins``."""
    crystal_key = material
    if material in CATALOG.materials:
        crystal_key = CATALOG.material(material).crystal_key
    if crystal_key not in CATALOG.crystals:
        raise ValueError(f"unknown material {material!r} (have {list(CATALOG.crystals)})")
    spec = CATALOG.crystal(crystal_key)
    if spec.E_grid is None:
        raise ValueError(f"crystal {crystal_key!r} has no configured photon-energy grid")
    hkl_list: list[tuple[int, ...]]
    if not spec.hkl_list:
        hkl_list = dominant_reflections(crystal_key, n_families=n_families, B_ang2=spec.B_ang2)
    else:
        hkl_list = list(spec.hkl_list)
    return dict(
        crystal=crystal_key,
        composition=list(spec.composition),
        hkl_list=hkl_list,
        beam_uvw=spec.beam_uvw,
        surface_hkl=spec.surface_hkl,
        B_ang2=spec.B_ang2,
        E_grid=spec.E_grid,
    )


def _reject_banned_angles(
    tilts: np.ndarray, azimuths: np.ndarray, *, allow_normal_incidence: bool = False
) -> None:
    """Refuse the degenerate geometries no emission sweep may use (issue_notes.md #1).

    polar tilt == 0 deg radiates zero coherent-line intensity (the tilt=0
    degeneracy the line-grid coverage search flags), and azimuth == 90 deg sits
    on a symmetry axis that mis-ranked the coverage search (the azim-90 bug).
    Guarding at construction makes every downstream path -- production sweeps,
    catalog grids, the analyze_line_grid_bounds diagnostic -- structurally
    incapable of silently sampling them; callers must pick a small nonzero tilt
    (e.g. 5 deg) and an azimuth off 90 (e.g. 100-180 deg) instead.

    ``allow_normal_incidence`` is the transport-only escape hatch: the
    penetration-depth study is not an emission sweep and uses tilt=0 as its
    physical baseline, so it opts out of the tilt check only. The azimuth==90 ban
    is unconditional -- it has no legitimate use.
    """
    if not allow_normal_incidence and np.any(np.isclose(tilts, 0.0)):
        raise ValueError(
            "polar tilt_deg == 0 is disallowed for emission sweeps (degenerate "
            "zero coherent-line geometry); use a small nonzero tilt such as 5 deg, "
            "or set allow_normal_incidence=True for a transport-only study"
        )
    if np.any(np.isclose(azimuths, 90.0)):
        raise ValueError(
            "tilt_azim_deg == 90 is disallowed (degenerate symmetry axis, the "
            "azim-90 ranking bug); use an azimuth off 90 such as 100-180 deg"
        )


def _positive_values(name: str, values: ScalarOrSeq) -> np.ndarray:
    """One vocabulary for every swept positive-length geometry field."""
    try:
        numbers = _seq(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number or a sequence of numbers") from exc
    if numbers.size == 0:
        raise ValueError(f"{name} must supply at least one value")
    if not np.all(np.isfinite(numbers) & (numbers > 0.0)):
        raise ValueError(f"{name} must be finite and positive")
    return numbers


@dataclass(frozen=True)
class Footprint:
    """Bounded rectangular crystal footprint: FULL width and height in mm.

    Supplying a ``Footprint`` at all is what makes the crystal transversely
    finite; ``footprint=None`` is the legacy infinite slab. Holding both
    dimensions on one object makes the historical "both or neither" pairing rule
    unrepresentable rather than checked.

    Parameters
    ----------
    width_mm, height_mm
        Full transverse dimensions in mm. Each may be a positive scalar or a
        sequence expanded by campaign case construction.
    """

    width_mm: ScalarOrSeq
    height_mm: ScalarOrSeq

    def __post_init__(self) -> None:
        _positive_values("Footprint.width_mm", self.width_mm)
        _positive_values("Footprint.height_mm", self.height_mm)

    def _pairs(self) -> list[tuple[float, float]]:
        """The (width, height) product, in build_cases order."""
        return list(product(_seq(self.width_mm), _seq(self.height_mm)))


DEFAULT_FOOTPRINT = Footprint(5.0, 5.0)
"""The historical default bounded footprint. Frozen, so one instance is shared."""

DEFAULT_SUBSTRATE_THICKNESS_ANG = 5e6
"""The historical ``substrate_thickness_ang`` default: a 0.5 mm wafer."""


@dataclass(frozen=True)
class BlazedGrooves:
    """Blazed sawtooth grooves machined into the beam-entrance face.

    A v1 single-slab feature at a fixed in-plane orientation, so it only exists
    on :class:`Slab` -- which turns "grooves forbid a substrate/stack" into a
    type constraint. A finite footprint IS compatible and is the default: the
    sub-micron groove phase and the mm-scale footprint are independent in
    transport. See docs/validation/geometry/blazed-groove-geometry.md.

    Parameters
    ----------
    spacing_ang
        Positive groove period in angstroms. The supported geometry fixes the
        groove orientation and derives facet angles from the target tilt.
    """

    spacing_ang: float

    def __post_init__(self) -> None:
        if isinstance(self.spacing_ang, bool) or not isinstance(self.spacing_ang, (int, float)):
            raise TypeError("BlazedGrooves.spacing_ang must be a single number")
        spacing = float(self.spacing_ang)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("BlazedGrooves.spacing_ang must be finite and positive")


@dataclass(frozen=True)
class Layer:
    """One layer of a :class:`Stack`, in beam-entrance order.

    ``thickness_ang`` is ``ScalarOrSeq`` only for the film (``Stack.layers[0]``);
    every layer beneath it is fixed, which :class:`Stack` enforces.

    Parameters
    ----------
    material
        Catalog material or medium key.
    thickness_ang
        Layer thickness in angstroms. Only the entrance film may use a sequence.
    beam_uvw
        Optional three-index crystal direction aligned with the incident beam.
    azimuth_deg
        In-plane crystal rotation in degrees relative to the entrance film.
    """

    material: str
    thickness_ang: ScalarOrSeq
    beam_uvw: tuple[int, int, int] | None = None
    azimuth_deg: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Layer.material must be a non-empty material key")
        _positive_values(f"Layer.thickness_ang ({self.material!r})", self.thickness_ang)
        if self.beam_uvw is not None and len(tuple(self.beam_uvw)) != 3:
            raise ValueError(
                f"Layer.beam_uvw ({self.material!r}) must be a 3-component [uvw] direction"
            )
        if not math.isfinite(float(self.azimuth_deg)):
            raise ValueError(f"Layer.azimuth_deg ({self.material!r}) must be finite")

    def spec(self, thickness_ang: float) -> LayerSpec:
        """This layer as the fixed-thickness :class:`LayerSpec` lowering consumes."""
        return LayerSpec(
            self.material,
            float(thickness_ang),
            None if self.beam_uvw is None else tuple(self.beam_uvw),  # type: ignore[arg-type]
            float(self.azimuth_deg),
        )


@dataclass(frozen=True)
class LoweredTarget:
    """One point of a target's geometry product, as the case keys it becomes.

    ``name`` is the full geometry half of the case name; :meth:`case_keys` is the
    payload fragment, and it omits ``groove_spacing_ang`` for an ungrooved target
    so existing case payloads stay bit-for-bit.
    """

    name: str
    thickness_ang: float
    tilt_deg: float
    tilt_azim_deg: float
    crystal_width_mm: float | None
    crystal_height_mm: float | None
    groove_spacing_ang: float | None
    abs_layers: list | None
    layer_radiators: list | None

    def case_keys(self) -> dict[str, Any]:
        keys: dict[str, Any] = dict(
            thickness_ang=self.thickness_ang,
            crystal_width_mm=self.crystal_width_mm,
            crystal_height_mm=self.crystal_height_mm,
            tilt_deg=self.tilt_deg,
            tilt_azim_deg=self.tilt_azim_deg,
            abs_layers=self.abs_layers,
            layer_radiators=self.layer_radiators,
        )
        if self.groove_spacing_ang is not None:
            keys["groove_spacing_ang"] = self.groove_spacing_ang
        return keys


class _TargetGeometry:
    """Angle validity and product lowering shared by the target variants.

    Not a dataclass base: :class:`Slab` and :class:`Stack` need different leading
    positional fields, so each declares its own and inherits only behaviour.
    """

    tilt_deg: ScalarOrSeq
    tilt_azim_deg: ScalarOrSeq
    footprint: Footprint | None
    allow_normal_incidence: bool
    mosaic: bool

    # -- variant hooks --------------------------------------------------------

    def _entrance_face(self) -> BlazedGrooves | None:
        return None

    def _film_thickness(self) -> ScalarOrSeq:
        raise NotImplementedError

    def _below_film(self) -> tuple[LayerSpec, ...] | None:
        """Layers beneath the film, or None for a bare slab (no ``abs_layers``)."""
        raise NotImplementedError

    # -- validity -------------------------------------------------------------

    def _angles(self) -> tuple[np.ndarray, np.ndarray]:
        """Quantize before validating: the checks must see the angles cases use."""
        return _quantized_angles(self.tilt_deg), _quantized_angles(self.tilt_azim_deg)

    def _validate_geometry(self) -> None:
        if self.footprint is not None and not isinstance(self.footprint, Footprint):
            raise TypeError("footprint must be a Footprint or None (the infinite slab)")
        tilts, azimuths = self._angles()
        _reject_banned_angles(
            tilts, azimuths, allow_normal_incidence=bool(self.allow_normal_incidence)
        )
        if self._entrance_face() is None:
            return
        if not np.allclose(azimuths, 180.0):
            raise ValueError("blazed grooves require tilt_azim_deg == 180 for every case")
        if not np.all((tilts > 0.0) & (tilts < 90.0)):
            raise ValueError("blazed grooves require 0 < tilt_deg < 90 for every case")

    def validate_against(self, detector: Detector) -> None:
        """Validate the target x detector leg the target cannot own alone.

        Grooves are machined for one escape direction, so they need
        ``theta_obs_deg == 90``; that couples the target to a detector it does
        not and should not hold, which is why this is a separate call rather than
        a ``__post_init__`` check. Mirrors the theta_obs check in
        :func:`pyrite.montecarlo.groove.blazed_groove_spec`.
        """
        if self._entrance_face() is None:
            return
        if not np.isclose(detector.observation_angle_deg, 90.0):
            raise ValueError("blazed grooves require theta_obs_deg == 90 for every case")

    # -- lowering -------------------------------------------------------------

    def _footprints(self) -> list[tuple[float | None, float | None]]:
        if self.footprint is None:
            return [(None, None)]
        return list(self.footprint._pairs())

    def lower(
        self,
        cp: dict[str, Any],
        *,
        name_stem: str,
        beam_uvw,
        n_families: int = 4,
    ) -> tuple[LoweredTarget, ...]:
        """Expand the target into its geometry product, in build_cases order.

        ``name_stem`` leads every generated case name. It is the stable catalog
        key, not the display label: case names are persisted as checkpoint record
        keys, so editing a label must not rename stored cases.

        ``cp`` is the FILM's resolved :func:`crystal_params` and ``beam_uvw`` its
        resolved orientation. Both are passed in rather than looked up: they need
        ``n_families`` and a sweep-level orientation override, which are numerics,
        not geometry, and a target that reached into the catalog would pull the
        campaign layer into materials at the wrong point.
        """
        tilts, azimuths = self._angles()
        below = self._below_film()
        face = self._entrance_face()
        groove_spacing_ang = None if face is None else float(face.spacing_ang)
        layer_radiators = None
        if below is not None:
            # per-layer coherent radiators, aligned with abs_layers: the film (its
            # own crystal params, azimuth 0 because stack azimuths are relative to
            # it) then one per layer beneath, None where that layer is amorphous.
            layer_radiators = [
                _radiator(cp, beam_uvw=beam_uvw, azimuth_rad=0.0),
                *(layer_radiator(lay, n_families) for lay in below),
            ]

        lowered = []
        for thickness, tilt, azim, (width, height) in product(
            _seq(self._film_thickness()), tilts, azimuths, self._footprints()
        ):
            name = f"{name_stem} {fmt_thickness(thickness)} pol={tilt:g} az={azim:g}"
            abs_layers = None
            if below is not None:
                name = f"{name} on {'+'.join(lay.material for lay in below)}"
                abs_layers = stack_layers(cp["composition"], thickness, below)
            if width is not None:
                name = f"{name} footprint={width:g}x{height:g}mm"
            if groove_spacing_ang is not None:
                name = f"{name} groove={groove_spacing_ang / 1e4:g}um"
            lowered.append(
                LoweredTarget(
                    name=name,
                    thickness_ang=float(thickness),
                    tilt_deg=float(tilt),
                    tilt_azim_deg=float(azim),
                    crystal_width_mm=None if width is None else float(width),
                    crystal_height_mm=None if height is None else float(height),
                    groove_spacing_ang=groove_spacing_ang,
                    abs_layers=abs_layers,
                    layer_radiators=layer_radiators,
                )
            )
        return tuple(lowered)


@dataclass(frozen=True)
class Slab(_TargetGeometry):
    """Describe a single crystal slab, optionally bounded or grooved.

    Parameters
    ----------
    material
        Catalog material key for the radiator.
    thickness_ang
        Positive thickness in angstroms; a sequence is expanded by campaign
        builders.
    tilt_deg, tilt_azim_deg
        Polar and azimuthal crystal tilt angles in degrees. Scalars or sequences.
    footprint
        Finite full width and height. ``None`` selects a laterally infinite slab.
    entrance_face
        Optional supported blazed-groove geometry.
    allow_normal_incidence
        Permit normally incident geometry that is otherwise rejected as a
        likely configuration error.
    mosaic
        Apply the catalog crystal's mosaic spread when one is defined.
    """

    material: str
    thickness_ang: ScalarOrSeq = 2e4
    tilt_deg: ScalarOrSeq = 30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    footprint: Footprint | None = DEFAULT_FOOTPRINT
    entrance_face: BlazedGrooves | None = None
    allow_normal_incidence: bool = False
    # Crystal mosaicity is a property of the target crystal, not of the run: True
    # applies the catalog's per-crystal ``mosaic_fwhm_deg`` (crystals without one
    # stay perfect). The QUADRATURE choices that consume it -- ``mosaic_route``
    # and ``mosaic_nodes`` -- stay on ``Sweep``; they are numerics.
    mosaic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Slab.material must be a non-empty material key")
        _positive_values("Slab.thickness_ang", self.thickness_ang)
        if self.entrance_face is not None and not isinstance(self.entrance_face, BlazedGrooves):
            raise TypeError("Slab.entrance_face must be a BlazedGrooves or None")
        self._validate_geometry()

    def _entrance_face(self) -> BlazedGrooves | None:
        return self.entrance_face

    def _film_thickness(self) -> ScalarOrSeq:
        return self.thickness_ang

    def _below_film(self) -> tuple[LayerSpec, ...] | None:
        return None


@dataclass(frozen=True)
class Stack(_TargetGeometry):
    """A crystal film on one or more layers beneath it, in beam-entrance order.

    ``layers[0]`` is the film: it carries the swept thickness and is the material
    whose crystal params drive the case. Every layer beneath it is fixed, so a
    stack has no grooved entrance face -- that is a slab-only feature.

    Parameters
    ----------
    layers
        Film-first sequence containing at least two :class:`Layer` objects.
        Only the first layer may have multiple thickness values.
    tilt_deg, tilt_azim_deg
        Polar and azimuthal stack tilt angles in degrees. Scalars or sequences.
    footprint
        Finite full width and height. ``None`` selects infinite lateral extent.
    allow_normal_incidence
        Permit normally incident geometry.
    mosaic
        Apply the entrance crystal's catalog mosaic spread when available.
    """

    layers: tuple[Layer, ...]
    tilt_deg: ScalarOrSeq = 30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    footprint: Footprint | None = DEFAULT_FOOTPRINT
    allow_normal_incidence: bool = False
    mosaic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.layers, Layer) or not isinstance(self.layers, Sequence):
            raise TypeError("Stack.layers must be a sequence of Layer, film first")
        layers = tuple(self.layers)
        if any(not isinstance(layer, Layer) for layer in layers):
            raise TypeError("Stack.layers must contain only Layer values")
        if len(layers) < 2:
            raise ValueError(
                "Stack.layers needs the film plus at least one layer beneath it; "
                "use Slab for a free-standing film"
            )
        for layer in layers[1:]:
            if _seq(layer.thickness_ang).size != 1:
                raise ValueError(
                    f"Stack.layers: only the film may sweep thickness_ang; "
                    f"{layer.material!r} sits beneath it and must be a single value"
                )
        object.__setattr__(self, "layers", layers)
        self._validate_geometry()

    @classmethod
    def on_substrate(
        cls,
        material: str,
        thickness_ang: ScalarOrSeq,
        substrate: str,
        substrate_thickness_ang: float = DEFAULT_SUBSTRATE_THICKNESS_ANG,
        *,
        tilt_deg: ScalarOrSeq = 30.0,
        tilt_azim_deg: ScalarOrSeq = 0.0,
        footprint: Footprint | None = DEFAULT_FOOTPRINT,
        allow_normal_incidence: bool = False,
        mosaic: bool = False,
    ) -> Stack:
        """A film on one substrate wafer -- the sanctioned one-layer stack.

        This is what the retired ``Sweep(substrate=...)`` pair meant, spelled as a
        constructor: :class:`Stack` itself requires the film plus at least one
        layer beneath it, so a caller who wants only a substrate builds it here.
        """
        return cls(
            layers=(
                Layer(material, thickness_ang),
                Layer(substrate, substrate_thickness_ang),
            ),
            tilt_deg=tilt_deg,
            tilt_azim_deg=tilt_azim_deg,
            footprint=footprint,
            allow_normal_incidence=allow_normal_incidence,
            mosaic=mosaic,
        )

    @property
    def material(self) -> str:
        """The film material -- the one whose crystal params drive the case."""
        return self.layers[0].material

    def _film_thickness(self) -> ScalarOrSeq:
        return self.layers[0].thickness_ang

    def _below_film(self) -> tuple[LayerSpec, ...]:
        return tuple(layer.spec(float(_seq(layer.thickness_ang)[0])) for layer in self.layers[1:])


Target = Slab | Stack
"""The closed target variant set. Not an extension point -- see the module docstring."""


class _RetiredFlatInput:
    """The class attribute a retired flat geometry ``InitVar`` leaves behind.

    It does two jobs. It is the "was not supplied at all" sentinel that ``None``
    cannot serve -- ``crystal_width_mm=None`` IS the explicit infinite slab, a
    different statement from not mentioning the footprint. And it is what a READ
    of the retired name returns: ``dataclasses`` keeps an ``InitVar``'s default
    as a class attribute and ``dataclasses.replace`` reads it back, so the name
    cannot simply be deleted. Every use of the value other than an identity or
    ``isinstance`` check raises, so a stale ``sweep.tilt_deg`` reader fails
    loudly instead of silently seeing ``None``.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<retired flat geometry input {self._name!r}>"

    def _retired(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError(
            f"{self._name!r} is not state on Sweep: geometry lives on sweep.target "
            f"(target_flat_fields(sweep.target) gives the flat projection)"
        )

    __bool__ = _retired
    __eq__ = _retired
    __ne__ = _retired
    __lt__ = _retired
    __le__ = _retired
    __gt__ = _retired
    __ge__ = _retired
    __float__ = _retired
    __int__ = _retired
    __iter__ = _retired
    __len__ = _retired
    __getitem__ = _retired
    __hash__ = None  # type: ignore[assignment]


def retired_flat_input(name: str) -> Any:
    """The default for a retired flat geometry ``InitVar`` named *name*."""
    return _RetiredFlatInput(name)


def target_from_flat(
    material: str,
    *,
    thickness_ang: ScalarOrSeq = 2e4,
    tilt_deg: ScalarOrSeq = 30.0,
    tilt_azim_deg: ScalarOrSeq = 0.0,
    crystal_width_mm: ScalarOrSeq | None = 5.0,
    crystal_height_mm: ScalarOrSeq | None = 5.0,
    groove_spacing_ang: float | None = None,
    substrate: str | None = None,
    substrate_thickness_ang: float = DEFAULT_SUBSTRATE_THICKNESS_ANG,
    stack: Sequence[LayerSpec] | None = None,
    allow_normal_incidence: bool = False,
    mosaic: bool = False,
) -> Target:
    """Build the target described by the legacy flat geometry inputs.

    The defaults are the historical :class:`pyrite.campaign.sweep.Sweep` field
    defaults, so a flat construction that named nothing lowers exactly as it did.
    Every geometry RULE is owned by the target objects; the three checks here
    exist solely because the flat inputs are separate keys -- footprint pairing
    (unrepresentable once both dimensions sit on one :class:`Footprint`), the
    substrate/stack exclusion, and grooves-forbid-a-stack (a type constraint once
    ``entrance_face`` is reachable only through :class:`Slab`).
    """
    footprint = None
    if crystal_width_mm is not None or crystal_height_mm is not None:
        if crystal_width_mm is None or crystal_height_mm is None:
            raise ValueError("crystal_width_mm and crystal_height_mm must be supplied together")
        footprint = Footprint(crystal_width_mm, crystal_height_mm)

    if substrate is not None and stack is not None:
        raise ValueError("give either substrate= or stack=, not both")
    if groove_spacing_ang is not None and (substrate is not None or stack is not None):
        raise ValueError("grooves are v1 single-slab only (no substrate/stack)")

    # the substrate sugar IS the one-layer stack, so it goes through its helper
    if substrate is not None:
        return Stack.on_substrate(
            material,
            thickness_ang,
            substrate,
            substrate_thickness_ang,
            tilt_deg=tilt_deg,
            tilt_azim_deg=tilt_azim_deg,
            footprint=footprint,
            allow_normal_incidence=allow_normal_incidence,
            mosaic=mosaic,
        )

    if stack is not None:
        return Stack(
            layers=(
                Layer(material, thickness_ang),
                *(
                    Layer(lay.material, lay.thickness_ang, lay.beam_uvw, lay.azimuth_deg)
                    for lay in stack
                ),
            ),
            tilt_deg=tilt_deg,
            tilt_azim_deg=tilt_azim_deg,
            footprint=footprint,
            allow_normal_incidence=allow_normal_incidence,
            mosaic=mosaic,
        )
    return Slab(
        material,
        thickness_ang=thickness_ang,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        footprint=footprint,
        entrance_face=(None if groove_spacing_ang is None else BlazedGrooves(groove_spacing_ang)),
        allow_normal_incidence=allow_normal_incidence,
        mosaic=mosaic,
    )


def target_flat_fields(target: Target) -> dict[str, Any]:
    """The flat geometry inputs that rebuild ``target`` through
    :func:`target_from_flat`.

    Exactly one default-oriented layer beneath the film inverts to the
    ``substrate=`` sugar rather than the general ``stack=`` spelling, because
    that IS its definition -- ``substrate="x"`` means
    ``stack=(LayerSpec("x", substrate_thickness_ang),)``. Keeping the sugar makes
    the projection lossless in both directions and lets the legacy identity
    payload keep the spelling it has always hashed.
    """
    footprint = target.footprint
    face = target._entrance_face()
    below = target._below_film()
    substrate = None
    substrate_thickness_ang = DEFAULT_SUBSTRATE_THICKNESS_ANG
    stack: tuple[LayerSpec, ...] | None = None if below is None else tuple(below)
    if stack is not None and len(stack) == 1 and stack[0].beam_uvw is None:
        only = stack[0]
        if float(only.azimuth_deg) == 0.0:
            substrate = only.material
            substrate_thickness_ang = float(only.thickness_ang)
            stack = None
    return {
        "material": target.material,
        "thickness_ang": target._film_thickness(),
        "tilt_deg": target.tilt_deg,
        "tilt_azim_deg": target.tilt_azim_deg,
        "crystal_width_mm": None if footprint is None else footprint.width_mm,
        "crystal_height_mm": None if footprint is None else footprint.height_mm,
        "groove_spacing_ang": None if face is None else face.spacing_ang,
        "substrate": substrate,
        "substrate_thickness_ang": substrate_thickness_ang,
        "stack": stack,
        "allow_normal_incidence": target.allow_normal_incidence,
        "mosaic": target.mosaic,
    }


def target_replace(target: Target, **changes: Any) -> Target:
    """``target`` with flat geometry overrides applied, rebuilt and revalidated.

    Overrides are flat because that is the vocabulary the CLI, catalog profiles
    and the sweep override path already speak. A rebuild rather than a field
    replace is required: ``substrate=`` / ``stack=`` change which variant the
    target IS, and a footprint dimension has to merge with the one it keeps.
    """
    flat = target_flat_fields(target)
    unknown = sorted(set(changes) - set(flat))
    if unknown:
        raise TypeError(f"target_replace: unknown geometry override(s): {', '.join(unknown)}")
    if "substrate" in changes or "stack" in changes:
        # The pair states the layers beneath the film as a whole; keeping the
        # inherited stack would make substrate=None a no-op instead of a clear.
        flat["substrate"] = None
        flat["stack"] = None
        flat["substrate_thickness_ang"] = DEFAULT_SUBSTRATE_THICKNESS_ANG
    elif "substrate_thickness_ang" in changes and flat["substrate"] is None:
        raise ValueError("substrate_thickness_ang override requires substrate=")
    flat.update(changes)
    return target_from_flat(**flat)
