"""Inverse adapters between target variants and the legacy flat geometry vocabulary.

Catalog data, profile overrides and identity digests still speak the flat keys
(``thickness_ang``, ``tilt_deg``, ``crystal_width_mm``, ``substrate``, ...).
:func:`target_from_flat`, :func:`target_flat_fields` and :func:`target_replace`
translate between that vocabulary and :mod:`pyrite.campaign.geometry` targets;
every geometry rule stays owned by the target objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..materials import LayerSpec
from .geometry import (
    DEFAULT_SUBSTRATE_THICKNESS_ANG,
    BlazedGrooves,
    CrystalCut,
    Footprint,
    Layer,
    ScalarOrSeq,
    Slab,
    Stack,
    Target,
)


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
    cut: CrystalCut | None = None,
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
    if cut is not None and (substrate is not None or stack is not None):
        raise ValueError("a crystal cut is single-slab only (no substrate/stack)")

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
        cut=cut,
    )


def target_flat_fields(target: Target) -> dict[str, Any]:
    """The flat geometry inputs that rebuild ``target`` through
    :func:`target_from_flat`.

    Exactly one default-oriented layer beneath the film inverts to the
    ``substrate=`` sugar rather than the general ``stack=`` spelling, because
    that IS its definition -- ``substrate="x"`` means
    ``stack=(LayerSpec("x", substrate_thickness_ang),)``. Keeping the sugar makes
    the projection lossless in both directions and lets the legacy identity
    payload keep the spelling it has always hashed. ``cut`` joins only when set,
    for the same reason.
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
    cut = target._crystal_cut()
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
        **({} if cut is None else {"cut": cut}),
    }


def target_replace(target: Target, **changes: Any) -> Target:
    """``target`` with flat geometry overrides applied, rebuilt and revalidated.

    Overrides are flat because that is the vocabulary the CLI, catalog profiles
    and the sweep override path already speak. A rebuild rather than a field
    replace is required: ``substrate=`` / ``stack=`` change which variant the
    target IS, and a footprint dimension has to merge with the one it keeps.
    """
    flat = target_flat_fields(target)
    unknown = sorted(set(changes) - set(flat) - {"cut"})
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
