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
from dataclasses import dataclass, field, replace
from itertools import product
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import numpy as np

from .._formatting import fmt_thickness
from .._spec_defaults import UNSET as _UNSET
from .._spec_defaults import _Unset
from .._units import _positive_length_quantity as _geometry_length
from .._units import normalize_unit
from ..materials import CATALOG, LayerSpec
from ..materials._composition import resolve_composition
from ..materials.crystal import dominant_reflections
from ._geometry_table import geometry_table as geometry_table
from ._geometry_values import (
    _angle_quantity,
    _angle_spelling,
    _as_miller,
    _miller,
    _positive_values,
    _quantity_geometry,
    _reject_banned_angles,
    _scalar_angle_quantity,
    _seq,
    _unique_angles,
)

ScalarOrSeq = float | Sequence[float] | np.ndarray

if TYPE_CHECKING:
    from pint import Quantity


class _ObservationDirection(Protocol):
    """Only the reference direction needed for target/observation validation."""

    @property
    def observation_angle_deg(self) -> float: ...


def substrate_composition(substrate):
    """Number-density composition [(element, n_per_Ang3), ...] for a substrate.
    Amorphous presets ('sio2') come from bulk density; a crystalline
    substrate already in CRYSTALS (e.g. 'silicon') uses its unit-cell density."""
    try:
        return list(resolve_composition(substrate.lower(), catalog=CATALOG, namespaces=("medium",)))
    except KeyError:
        pass
    try:
        return list(resolve_composition(substrate, catalog=CATALOG, namespaces=("crystal",)))
    except KeyError:
        pass
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
    and the isotropic B-factor."""
    crystal_key = material
    if material in CATALOG.materials:
        crystal_key = CATALOG.material(material).crystal_key
    if crystal_key not in CATALOG.crystals:
        raise ValueError(f"unknown material {material!r} (have {list(CATALOG.crystals)})")
    spec = CATALOG.crystal(crystal_key)
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
    )


@dataclass(frozen=True, init=False)
class Footprint:
    """Bounded rectangular crystal footprint: FULL width and height in mm.

    Supplying a ``Footprint`` at all makes the crystal transversely finite;
    ``footprint=None`` is the legacy infinite slab. Holding both dimensions
    together makes the historical "both or neither" pairing rule unrepresentable.

    Parameters
    ----------
    width_mm, height_mm
        Legacy full transverse dimensions in mm, scalar or sweep sequence;
        positional arguments remain supported and may contain compatible Pint
        quantities. The numeric fields remain the serialized/lowered contract.
    width, height
        Keyword-only explicit Pint lengths, scalar or one-dimensional sweep
        arrays. Each conflicts with its matching ``*_mm`` spelling. Quantity
        read-back returns detached mm values; ``with_width`` / ``with_height``
        replace one dimension without changing the other.
    """

    width_mm: ScalarOrSeq
    height_mm: ScalarOrSeq

    def __init__(
        self,
        width_mm: ScalarOrSeq | _Unset = _UNSET,
        height_mm: ScalarOrSeq | _Unset = _UNSET,
        *,
        width: object = _UNSET,
        height: object = _UNSET,
    ) -> None:
        if width is not _UNSET:
            if width_mm is not _UNSET:
                raise TypeError("Footprint.width_mm and width are mutually exclusive")
            width_mm = _geometry_length(width, unit="mm", field="Footprint.width")
        elif width_mm is _UNSET:
            raise TypeError("Footprint requires width_mm or width")
        if height is not _UNSET:
            if height_mm is not _UNSET:
                raise TypeError("Footprint.height_mm and height are mutually exclusive")
            height_mm = _geometry_length(height, unit="mm", field="Footprint.height")
        elif height_mm is _UNSET:
            raise TypeError("Footprint requires height_mm or height")
        object.__setattr__(self, "width_mm", width_mm)
        object.__setattr__(self, "height_mm", height_mm)
        self.__post_init__()

    @property
    def width(self) -> Quantity:
        """Fresh full x width in mm, retaining the original scalar/sweep shape."""
        return _quantity_geometry(self.width_mm, "mm")

    @property
    def height(self) -> Quantity:
        """Fresh full y height in mm, retaining the original scalar/sweep shape."""
        return _quantity_geometry(self.height_mm, "mm")

    def with_width(self, width: object) -> Self:
        """Return a copy with an explicitly unit-bearing full x width."""
        return replace(self, width_mm=_geometry_length(width, unit="mm", field="Footprint.width"))

    def with_height(self, height: object) -> Self:
        """Return a copy with an explicitly unit-bearing full y height."""
        return replace(
            self, height_mm=_geometry_length(height, unit="mm", field="Footprint.height")
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "width_mm",
            normalize_unit(self.width_mm, "mm", field="Footprint.width_mm", expected="length"),
        )
        object.__setattr__(
            self,
            "height_mm",
            normalize_unit(self.height_mm, "mm", field="Footprint.height_mm", expected="length"),
        )
        _positive_values("Footprint.width_mm", self.width_mm)
        _positive_values("Footprint.height_mm", self.height_mm)

    def _pairs(self) -> list[tuple[float, float]]:
        """The (width, height) product, in build_cases order."""
        return list(product(_seq(self.width_mm), _seq(self.height_mm)))


DEFAULT_FOOTPRINT = Footprint(5.0, 5.0)
"""The historical default bounded footprint. Frozen, so one instance is shared."""

#: Placeholder for an omitted ``footprint``. ``None`` already means the infinite
#: slab, so "not supplied" needs its own value; through the #451 deprecation
#: window it resolves, with a warning, to :data:`DEFAULT_FOOTPRINT`.

DEFAULT_SUBSTRATE_THICKNESS_ANG = 5e6
"""The historical ``substrate_thickness_ang`` default: a 0.5 mm wafer."""


@dataclass(frozen=True, init=False)
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
    spacing
        Keyword-only scalar Pint length, mutually exclusive with ``spacing_ang``.
        Read-back is a fresh angstrom quantity; ``with_spacing`` replaces it.
        Numeric fields, positional construction and serialization stay unchanged.
        ``dataclasses.replace`` uses ``spacing_ang``; passing ``spacing`` conflicts.
    """

    spacing_ang: float

    def __init__(self, spacing_ang: float | _Unset = _UNSET, *, spacing: object = _UNSET) -> None:
        if spacing is not _UNSET:
            if spacing_ang is not _UNSET:
                raise TypeError("BlazedGrooves received both spacing_ang and spacing")
            spacing_ang = float(
                _geometry_length(spacing, unit="angstrom", field="BlazedGrooves.spacing", shape=())
            )
        elif spacing_ang is _UNSET:
            raise TypeError("BlazedGrooves requires spacing_ang or spacing")
        object.__setattr__(self, "spacing_ang", spacing_ang)
        self.__post_init__()

    @property
    def spacing(self) -> Quantity:
        """Groove period as a fresh scalar quantity in angstroms."""
        return _quantity_geometry(self.spacing_ang, "angstrom")

    def with_spacing(self, spacing: object) -> Self:
        """Replace the period using an explicit scalar length quantity."""
        return replace(
            self,
            spacing_ang=_geometry_length(
                spacing, unit="angstrom", field="BlazedGrooves.spacing", shape=()
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spacing_ang",
            normalize_unit(
                self.spacing_ang,
                "angstrom",
                field="BlazedGrooves.spacing_ang",
                expected="length",
            ),
        )
        if isinstance(self.spacing_ang, bool) or not isinstance(self.spacing_ang, (int, float)):
            raise TypeError("BlazedGrooves.spacing_ang must be a single number")
        spacing = float(self.spacing_ang)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("BlazedGrooves.spacing_ang must be finite and positive")


Miller = tuple[int, int, int]


@dataclass(frozen=True, init=False)
class CrystalCut:
    """Orient the crystal lattice relative to a slab's entrance face.

    The cut face normal is the sample ``+z`` axis and defines the zero of the
    target's ``tilt_deg``/``tilt_azim_deg``: ``tilt_deg=0`` is normal
    incidence on the cut face, and tilting moves face and lattice together.
    The cut fixes the lattice under that face by the rotation
    ``R_miscut @ R_roll @ R_axis``: ``R_axis`` takes the selected axis onto the
    face normal, ``R_roll`` is a right-handed roll about it, and ``R_miscut``
    tilts the lattice off it, i.e. the face is cut ``miscut_deg`` off the
    selected plane.

    Parameters
    ----------
    surface_hkl, beam_uvw
        Mutually exclusive integer axis placed along the face normal: a
        reciprocal ``(hkl)`` plane normal or a direct ``[uvw]`` direction. Both
        ``None`` keep the catalog crystal's axis. Reflection indices are chosen
        separately (``Physics``/catalog ``hkl_list``), never from this axis.
    roll_deg
        Right-handed in-plane roll about the face normal, degrees.
    miscut_deg, miscut_azim_deg
        Continuous lattice tilt off the face normal (``0 <= miscut_deg < 90``)
        and the sample-frame azimuth it tilts toward, degrees.
    roll, miscut, miscut_azimuth
        Keyword-only explicit scalar Pint plane angles, each conflicting with
        its ``*_deg`` spelling. Read-back is a fresh degree quantity;
        ``with_roll``/``with_miscut``/``with_miscut_azimuth`` replace them.

    An all-default cut is the catalog orientation; omitting the cut and passing
    ``CrystalCut()`` lower to identical cases.

    Validation: surface-hkl-orientation
    """

    surface_hkl: Miller | None = None
    beam_uvw: Miller | None = None
    roll_deg: float = 0.0
    miscut_deg: float = 0.0
    miscut_azim_deg: float = 0.0

    def __init__(
        self,
        surface_hkl: Miller | None = None,
        beam_uvw: Miller | None = None,
        roll_deg: float | _Unset = _UNSET,
        miscut_deg: float | _Unset = _UNSET,
        miscut_azim_deg: float | _Unset = _UNSET,
        *,
        roll: object = _UNSET,
        miscut: object = _UNSET,
        miscut_azimuth: object = _UNSET,
    ) -> None:
        object.__setattr__(self, "surface_hkl", surface_hkl)
        object.__setattr__(self, "beam_uvw", beam_uvw)
        for numeric_name, numeric, semantic_name, semantic in (
            ("roll_deg", roll_deg, "roll", roll),
            ("miscut_deg", miscut_deg, "miscut", miscut),
            ("miscut_azim_deg", miscut_azim_deg, "miscut_azimuth", miscut_azimuth),
        ):
            value = _angle_spelling(
                "CrystalCut",
                numeric_name,
                numeric,
                semantic_name,
                semantic,
                unset=_UNSET,
                default=0.0,
                scalar=True,
            )
            object.__setattr__(self, numeric_name, value)
        self.__post_init__()

    @property
    def roll(self) -> Quantity:
        """Fresh scalar degree quantity for the roll about the face normal."""
        return _quantity_geometry(self.roll_deg, "degree")

    def with_roll(self, roll: object) -> Self:
        """Copy with an explicit plane-angle roll; preserve axis and miscut."""
        return replace(self, roll_deg=_scalar_angle_quantity(roll, field="CrystalCut.roll"))

    @property
    def miscut(self) -> Quantity:
        """Fresh scalar degree quantity for the lattice tilt off the face normal."""
        return _quantity_geometry(self.miscut_deg, "degree")

    def with_miscut(self, miscut: object) -> Self:
        """Copy with an explicit plane-angle miscut; ``0 <= miscut < 90 deg``."""
        degrees = _scalar_angle_quantity(miscut, field="CrystalCut.miscut")
        return replace(self, miscut_deg=degrees)

    @property
    def miscut_azimuth(self) -> Quantity:
        """Fresh scalar degree quantity for the sample-frame miscut azimuth."""
        return _quantity_geometry(self.miscut_azim_deg, "degree")

    def with_miscut_azimuth(self, miscut_azimuth: object) -> Self:
        """Copy with an explicit plane-angle miscut azimuth; preserve other settings."""
        degrees = _scalar_angle_quantity(miscut_azimuth, field="CrystalCut.miscut_azimuth")
        return replace(self, miscut_azim_deg=degrees)

    def __post_init__(self) -> None:
        surface = _miller("surface_hkl", self.surface_hkl)
        direct = _miller("beam_uvw", self.beam_uvw)
        if surface is not None and direct is not None:
            raise ValueError("CrystalCut.surface_hkl and beam_uvw are mutually exclusive")
        object.__setattr__(self, "surface_hkl", surface)
        object.__setattr__(self, "beam_uvw", direct)
        for name in ("roll_deg", "miscut_deg", "miscut_azim_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
                raise TypeError(f"CrystalCut.{name} must be a single real angle in degrees")
            if not math.isfinite(float(value)):
                raise ValueError(f"CrystalCut.{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.miscut_deg < 90.0:
            raise ValueError("CrystalCut.miscut_deg must satisfy 0 <= miscut_deg < 90")


@dataclass(frozen=True)
class ResolvedCut:
    """A target's effective lattice orientation, as the case keys carry it.

    ``beam_uvw``/``surface_hkl`` are the axis after catalog fallback;
    ``azimuth_rad`` and ``recip_miscut_rad`` join a case only when nonzero, so a
    default orientation keeps its historical payload. ``label`` is the case-name
    suffix and is empty exactly when the orientation equals the catalog's.
    """

    beam_uvw: Miller | None
    surface_hkl: Miller | None
    azimuth_rad: float = 0.0
    recip_miscut_rad: tuple[float, float] | None = None
    label: str = ""

    def case_keys(self) -> dict[str, Any]:
        keys: dict[str, Any] = {}
        if self.azimuth_rad:
            keys["azimuth_rad"] = self.azimuth_rad
        if self.recip_miscut_rad is not None:
            keys["recip_miscut_rad"] = self.recip_miscut_rad
        return keys

    def rotation(self, lattice: Any) -> np.ndarray | None:
        """The lattice rotation every consumer applies to reciprocal vectors."""
        from ..materials.crystal import _orientation_R

        return _orientation_R(
            lattice,
            self.beam_uvw,
            self.azimuth_rad,
            self.recip_miscut_rad,
            surface_hkl=self.surface_hkl,
        )


def resolve_crystal_cut(
    cut: CrystalCut | None,
    beam_uvw: Any,
    surface_hkl: Any,
) -> ResolvedCut:
    """Apply ``cut`` to an already resolved catalog/sweep axis.

    The single resolver shared by case lowering and bunch-front matching, so
    both see the same lattice orientation.

    Validation: surface-hkl-orientation
    """
    base_uvw, base_hkl = _as_miller(beam_uvw), _as_miller(surface_hkl)
    if cut is None:
        return ResolvedCut(base_uvw, base_hkl)
    parts = []
    if cut.surface_hkl is not None and cut.surface_hkl != base_hkl:
        parts.append("cut=({})".format(",".join(map(str, cut.surface_hkl))))
    elif cut.beam_uvw is not None and (cut.beam_uvw != base_uvw or base_hkl is not None):
        parts.append("cut=[{}]".format(",".join(map(str, cut.beam_uvw))))
    if cut.surface_hkl is not None or cut.beam_uvw is not None:
        base_uvw, base_hkl = cut.beam_uvw, cut.surface_hkl
    if cut.roll_deg:
        parts.append(f"roll={cut.roll_deg:.12g}")
    miscut = None
    if cut.miscut_deg:
        miscut = (math.radians(cut.miscut_deg), math.radians(cut.miscut_azim_deg))
        parts.append(f"miscut={cut.miscut_deg:.12g}@{cut.miscut_azim_deg:.12g}")
    return ResolvedCut(
        base_uvw,
        base_hkl,
        azimuth_rad=math.radians(cut.roll_deg),
        recip_miscut_rad=miscut,
        label=" ".join(parts),
    )


@dataclass(frozen=True, init=False)
class Layer:
    """One layer of a :class:`Stack`, in beam-entrance order.

    ``thickness_ang`` is ``ScalarOrSeq`` only for the film (``Stack.layers[0]``);
    every layer beneath it is fixed, which :class:`Stack` enforces.

    Parameters
    ----------
    material
        Catalog material or medium key.
    thickness_ang
        Legacy numeric angstroms; only the entrance film may use a sequence.
    thickness
        Keyword-only Pint length, scalar or one-dimensional sweep quantity.
        Conflicts with ``thickness_ang``. Read-back is a detached angstrom
        quantity; ``with_thickness`` replaces it without changing stored fields.
    beam_uvw
        Optional three-index crystal direction aligned with the incident beam.
    azimuth_deg
        In-plane crystal rotation in degrees relative to the entrance film.
    azimuth
        Keyword-only explicit scalar Pint angle, conflicting with ``azimuth_deg``.
        Read-back is a fresh degree quantity; ``with_azimuth`` replaces it.
        Signed and multi-turn angles retain their configured values.
    """

    material: str
    thickness_ang: ScalarOrSeq
    beam_uvw: tuple[int, int, int] | None = None
    azimuth_deg: float = 0.0

    def __init__(
        self,
        material: str,
        thickness_ang: ScalarOrSeq | _Unset = _UNSET,
        beam_uvw: tuple[int, int, int] | None = None,
        azimuth_deg: float | _Unset = _UNSET,
        *,
        thickness: object = _UNSET,
        azimuth: object = _UNSET,
    ) -> None:
        if thickness is not _UNSET:
            if thickness_ang is not _UNSET:
                raise TypeError("Layer.thickness_ang and thickness are mutually exclusive")
            thickness_ang = _geometry_length(thickness, unit="angstrom", field="Layer.thickness")
        elif thickness_ang is _UNSET:
            raise TypeError("Layer requires thickness_ang or thickness")
        azimuth_deg = _angle_spelling(
            "Layer",
            "azimuth_deg",
            azimuth_deg,
            "azimuth",
            azimuth,
            unset=_UNSET,
            default=0.0,
            scalar=True,
        )
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "thickness_ang", thickness_ang)
        object.__setattr__(self, "beam_uvw", beam_uvw)
        object.__setattr__(self, "azimuth_deg", azimuth_deg)
        self.__post_init__()

    @property
    def thickness(self) -> Quantity:
        """Fresh angstrom-valued layer thickness, scalar or film sweep."""
        return _quantity_geometry(self.thickness_ang, "angstrom")

    def with_thickness(self, thickness: object) -> Self:
        """Copy with an explicitly unit-bearing thickness; retain other fields."""
        return replace(
            self,
            thickness_ang=_geometry_length(thickness, unit="angstrom", field="Layer.thickness"),
        )

    @property
    def azimuth(self) -> Quantity:
        """Fresh scalar degree quantity for rotation relative to the entrance film."""
        return _quantity_geometry(self.azimuth_deg, "degree")

    def with_azimuth(self, azimuth: object) -> Self:
        """Copy with an explicit plane angle; preserve material, thickness and direction."""
        return replace(self, azimuth_deg=_scalar_angle_quantity(azimuth, field="Layer.azimuth"))

    def __post_init__(self) -> None:
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Layer.material must be a non-empty material key")
        object.__setattr__(
            self,
            "thickness_ang",
            normalize_unit(
                self.thickness_ang,
                "angstrom",
                field=f"Layer.thickness_ang ({self.material!r})",
                expected="length",
            ),
        )
        object.__setattr__(
            self,
            "azimuth_deg",
            normalize_unit(
                self.azimuth_deg,
                "degree",
                field=f"Layer.azimuth_deg ({self.material!r})",
                expected="angle",
                kind="angle",
            ),
        )
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
    crystal_orientation: dict[str, Any] = field(default_factory=dict)

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
        # Divergence-only: a catalog-oriented target adds nothing.
        keys.update(self.crystal_orientation)
        return keys


def _tilt_inputs(
    owner: str,
    tilt_deg: ScalarOrSeq | _Unset,
    tilt_azim_deg: ScalarOrSeq | _Unset,
    *,
    tilt: object,
    tilt_azimuth: object,
) -> tuple[ScalarOrSeq, ScalarOrSeq]:
    """Resolve numeric and semantic tilt spellings, preserving the numeric defaults."""
    return (
        _angle_spelling(owner, "tilt_deg", tilt_deg, "tilt", tilt, unset=_UNSET, default=30.0),
        _angle_spelling(
            owner,
            "tilt_azim_deg",
            tilt_azim_deg,
            "tilt_azimuth",
            tilt_azimuth,
            unset=_UNSET,
            default=0.0,
        ),
    )


class _TargetGeometry:
    """Angle validity and product lowering shared by the target variants.

    Not a dataclass base: :class:`Slab` and :class:`Stack` need different leading
    positional fields, so each declares its own and inherits only behaviour.
    """

    # The concrete targets are frozen dataclasses. Expose their fields as
    # read-only to the type checker without adding descriptors to this runtime
    # mixin; writable annotations here look like invalid overrides to `ty`.
    if TYPE_CHECKING:

        @property
        def tilt_deg(self) -> ScalarOrSeq: ...

        @property
        def tilt_azim_deg(self) -> ScalarOrSeq: ...

        @property
        def footprint(self) -> Footprint | None: ...

        @property
        def allow_normal_incidence(self) -> bool: ...

        @property
        def mosaic(self) -> bool: ...

    # -- semantic orientation quantities ---------------------------------------

    @property
    def tilt(self) -> Quantity:
        """Fresh degree-valued polar tilt, keeping scalar/sweep shape."""
        return _quantity_geometry(self.tilt_deg, "degree")

    def with_tilt(self, tilt: object) -> Self:
        """Copy with an explicit Pint polar tilt; preserve other settings."""
        label = f"{type(self).__name__}.tilt"
        return cast(Self, replace(cast(Any, self), tilt_deg=_angle_quantity(tilt, field=label)))

    @property
    def tilt_azimuth(self) -> Quantity:
        """Fresh degree-valued tilt azimuth, keeping scalar/sweep shape."""
        return _quantity_geometry(self.tilt_azim_deg, "degree")

    def with_tilt_azimuth(self, tilt_azimuth: object) -> Self:
        """Copy with an explicit Pint tilt azimuth; preserve other settings."""
        label = f"{type(self).__name__}.tilt_azimuth"
        degrees = _angle_quantity(tilt_azimuth, field=label)
        return cast(Self, replace(cast(Any, self), tilt_azim_deg=degrees))

    # -- variant hooks --------------------------------------------------------

    def _entrance_face(self) -> BlazedGrooves | None:
        return None

    def _crystal_cut(self) -> CrystalCut | None:
        return None

    def _film_thickness(self) -> ScalarOrSeq:
        raise NotImplementedError

    def _below_film(self) -> tuple[LayerSpec, ...] | None:
        """Layers beneath the film, or None for a bare slab (no ``abs_layers``)."""
        raise NotImplementedError

    # -- validity -------------------------------------------------------------

    def _angles(self) -> tuple[np.ndarray, np.ndarray]:
        """Validate the exact angles used by cases."""
        return _unique_angles(self.tilt_deg), _unique_angles(self.tilt_azim_deg)

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

    def validate_against(self, detector: _ObservationDirection) -> None:
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
        orientation: ResolvedCut | None = None,
    ) -> tuple[LoweredTarget, ...]:
        """Expand the target into its geometry product, in build_cases order.

        ``name_stem`` leads every generated case name. It is the stable catalog
        key, not the display label: case names are persisted as checkpoint record
        keys, so editing a label must not rename stored cases.

        ``cp`` is the FILM's resolved :func:`crystal_params` and ``beam_uvw`` its
        resolved orientation. Both are passed in rather than looked up: they need
        ``n_families`` and a sweep-level orientation override, which are numerics,
        not geometry, and a target that reached into the catalog would pull the
        campaign layer into materials at the wrong point. ``orientation`` is the
        film's :func:`resolve_crystal_cut` result; its nonzero roll/miscut keys
        and name label join every point.
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
            if orientation is not None and orientation.label:
                name = f"{name} {orientation.label}"
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
                    crystal_orientation=({} if orientation is None else orientation.case_keys()),
                )
            )
        return tuple(lowered)


@dataclass(frozen=True, init=False)
class Slab(_TargetGeometry):
    """Describe a single crystal slab, optionally bounded or grooved.

    Parameters
    ----------
    material
        Catalog material key for the radiator.
    thickness_ang
        Legacy numeric angstrom thickness, scalar or campaign sweep sequence.
        Omission preserves the historical 20000 Å default.
    thickness
        Keyword-only explicit Pint length, scalar or one-dimensional sweep.
        Conflicts with ``thickness_ang``; the ``thickness`` property returns
        a fresh angstrom quantity and ``with_thickness`` provides replacement.
    tilt_deg, tilt_azim_deg
        Polar and azimuthal crystal tilt angles in degrees. Scalars or sequences.
    tilt, tilt_azimuth
        Keyword-only explicit Pint plane angles, scalar or one-dimensional
        sweep, conflicting with ``tilt_deg``/``tilt_azim_deg``. Read-back is a
        fresh degree quantity; ``with_tilt``/``with_tilt_azimuth`` replace them.
    footprint
        Required: a finite :class:`Footprint`, or ``None`` for a laterally
        infinite slab.
    entrance_face
        Optional supported blazed-groove geometry.
    allow_normal_incidence
        Permit normally incident geometry that is otherwise rejected as a
        likely configuration error.
    mosaic
        Apply the catalog crystal's mosaic spread when one is defined.
    cut
        Optional :class:`CrystalCut` orienting the lattice relative to the
        flat entrance face; ``None`` keeps the catalog orientation.
    """

    material: str
    thickness_ang: ScalarOrSeq = 2e4
    tilt_deg: ScalarOrSeq = 30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    footprint: Footprint | None = field(kw_only=True)
    entrance_face: BlazedGrooves | None = None
    allow_normal_incidence: bool = False
    # Crystal mosaicity is a property of the target crystal, not of the run: True
    # applies the catalog's per-crystal ``mosaic_fwhm_deg`` (crystals without one
    # stay perfect). The QUADRATURE choices that consume it -- ``mosaic_route``
    # and ``mosaic_nodes`` -- stay on ``Sweep``; they are numerics.
    mosaic: bool = False
    cut: CrystalCut | None = None

    def __init__(
        self,
        material: str,
        thickness_ang: ScalarOrSeq | _Unset = _UNSET,
        tilt_deg: ScalarOrSeq | _Unset = _UNSET,
        tilt_azim_deg: ScalarOrSeq | _Unset = _UNSET,
        entrance_face: BlazedGrooves | None = None,
        allow_normal_incidence: bool = False,
        mosaic: bool = False,
        cut: CrystalCut | None = None,
        *,
        footprint: Footprint | None,
        thickness: object = _UNSET,
        tilt: object = _UNSET,
        tilt_azimuth: object = _UNSET,
    ) -> None:
        if thickness is not _UNSET:
            if thickness_ang is not _UNSET:
                raise TypeError("Slab.thickness_ang and thickness are mutually exclusive")
            thickness_ang = _geometry_length(thickness, unit="angstrom", field="Slab.thickness")
        elif thickness_ang is _UNSET:
            thickness_ang = 2e4
        tilt_deg, tilt_azim_deg = _tilt_inputs(
            "Slab", tilt_deg, tilt_azim_deg, tilt=tilt, tilt_azimuth=tilt_azimuth
        )
        for name, value in (
            ("material", material),
            ("thickness_ang", thickness_ang),
            ("tilt_deg", tilt_deg),
            ("tilt_azim_deg", tilt_azim_deg),
            ("footprint", footprint),
            ("entrance_face", entrance_face),
            ("allow_normal_incidence", allow_normal_incidence),
            ("mosaic", mosaic),
            ("cut", cut),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def thickness(self) -> Quantity:
        """Fresh angstrom-valued thickness, keeping scalar/sweep shape."""
        return _quantity_geometry(self.thickness_ang, "angstrom")

    def with_thickness(self, thickness: object) -> Self:
        """Copy with a Pint thickness; preserve orientation and other settings."""
        return replace(
            self,
            thickness_ang=_geometry_length(thickness, unit="angstrom", field="Slab.thickness"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Slab.material must be a non-empty material key")
        for name, unit, expected, kind in (
            ("thickness_ang", "angstrom", "length", None),
            ("tilt_deg", "degree", "angle", "angle"),
            ("tilt_azim_deg", "degree", "angle", "angle"),
        ):
            object.__setattr__(
                self,
                name,
                normalize_unit(
                    getattr(self, name),
                    unit,
                    field=f"Slab.{name}",
                    expected=expected,
                    kind=kind,
                ),
            )
        if self.cut is not None:
            if not isinstance(self.cut, CrystalCut):
                raise TypeError("Slab.cut must be a CrystalCut or None")
            if self.entrance_face is not None:
                raise ValueError("Slab.cut requires a flat entrance face (no grooves)")
        _positive_values("Slab.thickness_ang", self.thickness_ang)
        if self.entrance_face is not None and not isinstance(self.entrance_face, BlazedGrooves):
            raise TypeError("Slab.entrance_face must be a BlazedGrooves or None")
        self._validate_geometry()

    def _entrance_face(self) -> BlazedGrooves | None:
        return self.entrance_face

    def _crystal_cut(self) -> CrystalCut | None:
        return self.cut

    def _film_thickness(self) -> ScalarOrSeq:
        return self.thickness_ang

    def _below_film(self) -> tuple[LayerSpec, ...] | None:
        return None


@dataclass(frozen=True, init=False)
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
    tilt, tilt_azimuth
        Keyword-only explicit Pint plane angles, scalar or one-dimensional
        sweep, conflicting with ``tilt_deg``/``tilt_azim_deg``. Read-back is a
        fresh degree quantity; ``with_tilt``/``with_tilt_azimuth`` replace them.
    footprint
        Required: a finite :class:`Footprint`, or ``None`` for infinite lateral
        extent.
    allow_normal_incidence
        Permit normally incident geometry.
    mosaic
        Apply the entrance crystal's catalog mosaic spread when available.
    """

    layers: tuple[Layer, ...]
    tilt_deg: ScalarOrSeq = 30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    footprint: Footprint | None = field(kw_only=True)
    allow_normal_incidence: bool = False
    mosaic: bool = False

    def __init__(
        self,
        layers: Sequence[Layer],
        tilt_deg: ScalarOrSeq | _Unset = _UNSET,
        tilt_azim_deg: ScalarOrSeq | _Unset = _UNSET,
        allow_normal_incidence: bool = False,
        mosaic: bool = False,
        *,
        footprint: Footprint | None,
        tilt: object = _UNSET,
        tilt_azimuth: object = _UNSET,
    ) -> None:
        tilt_deg, tilt_azim_deg = _tilt_inputs(
            "Stack", tilt_deg, tilt_azim_deg, tilt=tilt, tilt_azimuth=tilt_azimuth
        )
        for name, value in (
            ("layers", layers),
            ("tilt_deg", tilt_deg),
            ("tilt_azim_deg", tilt_azim_deg),
            ("footprint", footprint),
            ("allow_normal_incidence", allow_normal_incidence),
            ("mosaic", mosaic),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

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
        for name in ("tilt_deg", "tilt_azim_deg"):
            object.__setattr__(
                self,
                name,
                normalize_unit(
                    getattr(self, name),
                    "degree",
                    field=f"Stack.{name}",
                    expected="angle",
                    kind="angle",
                ),
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
        tilt_deg: ScalarOrSeq | _Unset = _UNSET,
        tilt_azim_deg: ScalarOrSeq | _Unset = _UNSET,
        footprint: Footprint | None,
        allow_normal_incidence: bool = False,
        mosaic: bool = False,
        tilt: object = _UNSET,
        tilt_azimuth: object = _UNSET,
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
            tilt=tilt,
            tilt_azimuth=tilt_azimuth,
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


_FLAT_EXPORTS = frozenset(
    {
        "_RetiredFlatInput",
        "retired_flat_input",
        "target_flat_fields",
        "target_from_flat",
        "target_replace",
    }
)


def __getattr__(name: str) -> Any:
    # The flat-vocabulary adapters live in ``flat_geometry``, which imports the
    # target variants from here; resolve the historical names lazily.
    if name in _FLAT_EXPORTS:
        from . import flat_geometry

        return getattr(flat_geometry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
