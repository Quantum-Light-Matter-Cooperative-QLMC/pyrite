"""Thick-target screen: warn when photon transport would matter (#439, #341).

PyRITE removes interacting photons by straight-line Beer--Lambert escape and
does not transport them, so Compton build-up, pair daughters and their
radiation are missing. That is negligible in thin targets; above the pair
threshold it grows with the traversed thickness in radiation lengths. This
screen warns per case; it changes no case key, identity, or result.

Validation: radiation-length-tsai
"""

import math
import warnings
from collections.abc import Mapping
from typing import Any

from ..materials.radiation_length import radiation_length_ang

#: Warn above this traversed thickness in radiation lengths. At 0.05 X0 a
#: high-energy photon converts with probability 1 - exp(-7/9 * 0.05) ~ 4%, the
#: first percent-level loss of photon secondaries.
THICK_TARGET_X0_FRACTION = 0.05
#: Below 2 m_e c^2 no photon converts and the beam's own range keeps the
#: radiating depth far below X0 for every catalog material.
THICK_TARGET_MIN_E0_KEV = 1021.99790


class ThickTargetWarning(UserWarning):
    """A case traverses enough radiation lengths to need photon transport."""


def _case_layers(case: Mapping[str, Any]):
    layers = case.get("abs_layers")
    if layers is not None:
        return list(layers)
    return [(0.0, float(case["thickness_ang"]), case["composition"])]


def traversed_radiation_lengths(case: Mapping[str, Any]) -> float:
    """Normal-incidence-corrected stack thickness ``sum_i t_i / X0_i / cos(tilt)``.

    Straight-line primary path through every transport layer at the case
    tilt; scattering only lengthens it, so this is a lower bound.
    """
    total = sum(
        (float(bottom) - float(top)) / radiation_length_ang(composition)
        for top, bottom, composition in _case_layers(case)
    )
    cos_tilt = math.cos(math.radians(float(case.get("tilt_deg") or 0.0)))
    return total / cos_tilt if cos_tilt > 0.0 else math.inf


def warn_thick_target(case: Mapping[str, Any]) -> None:
    """Warn once per case above the pair threshold and ``THICK_TARGET_X0_FRACTION``."""
    if float(case["E0_keV"]) <= THICK_TARGET_MIN_E0_KEV:
        return
    depth = traversed_radiation_lengths(case)
    if depth <= THICK_TARGET_X0_FRACTION:
        return
    warnings.warn(
        f"thick-target screen: {case['name']} at {case['E0_keV']:g} keV traverses "
        f"{depth:.3g} X0 (> {THICK_TARGET_X0_FRACTION:g}); photons are attenuated, not "
        "transported, so Compton build-up and pair secondaries are missing. Photon "
        "transport is tracked in issue #341.",
        ThickTargetWarning,
        stacklevel=2,
    )
