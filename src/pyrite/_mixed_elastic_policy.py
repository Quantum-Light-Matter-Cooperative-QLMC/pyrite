"""Selector and parameters of opt-in mixed elastic transport (#894).

A leaf module so the configuration owners and the transport core share one
definition without importing each other. The model is documented in
``docs/validation/beam-transport/mixed-elastic-transport.md``.
"""

import math
from typing import NamedTuple

TRANSPORT_MODES = ("single-scattering", "mixed")
#: Case identity of the scheme: bump when its numerical output changes.
MIXED_ELASTIC_MODEL = "penelope-random-hinge.v2"
#: PENELOPE's accepted range of C1 and C2 (PENELOPE-2024, Sec. 4.4).
MIXED_PARAMETER_MAX = 0.2
MIXED_DEFAULT_C1 = 0.05
MIXED_DEFAULT_C2 = 0.05
#: Urban losses have no upper bound, so they leave a step's hazard window; the
#: re-verification at e25dede8b measured the resulting bias and did not isolate it.
MIXED_URBAN_REFUSAL = (
    "transport_mode='mixed' does not support Urban straggling yet; "
    "use inelastic_model='shell-soft-hard' or turn straggling off"
)
#: CXR scores each transport row with its own finite-time line shape, so the
#: L/100 step cap sets the line width: #990 measured 96 eV against 1.9 eV
#: detailed (HOPG 400 nm) and a 3x narrower peak (Si 1 mm).
MIXED_CXR_REFUSAL = (
    "transport_mode='mixed' does not support CXR line spectra: the mixed step cap "
    "sets the finite-time line width (#990); use transport_mode='single-scattering', "
    "or simulate_trajectories for transport-only mixed runs"
)


def refuse_mixed_cxr(case) -> None:
    """Refuse CXR line scoring of a case lowered with ``transport_mode="mixed"``."""
    if case.get("transport_mode") == "mixed":
        raise ValueError(MIXED_CXR_REFUSAL)


class MixedElasticParameters(NamedTuple):
    """Resolved parameters of ``transport_mode="mixed"``.

    ``c1`` bounds the mean soft deflection ``1 - <cos chi>`` per hard step,
    ``c2`` the mean fractional soft energy loss per step, and ``max_step_um``
    is an optional absolute step limit [um].
    """

    c1: float = MIXED_DEFAULT_C1
    c2: float = MIXED_DEFAULT_C2
    max_step_um: float | None = None


def validate_mixed_parameters(c1, c2, max_step_um=None):
    """Refuse parameters outside the accepted scheme; ``c1 = 0`` is detailed.

    Validation: mixed-elastic-split
    """
    for name, value in (("mixed_c1", c1), ("mixed_c2", c2)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= MIXED_PARAMETER_MAX
        ):
            raise ValueError(f"{name} must be a number in [0, {MIXED_PARAMETER_MAX}]")
    if c1 > 0.0 and c2 <= 0.0:
        raise ValueError("mixed_c2 must be positive when mixed_c1 is positive")
    if max_step_um is not None and (
        isinstance(max_step_um, bool)
        or not isinstance(max_step_um, (int, float))
        or not math.isfinite(max_step_um)
        or max_step_um <= 0.0
    ):
        raise ValueError("mixed_max_step_um must be finite and positive")


def validate_transport_mode(mode, c1=None, c2=None, max_step_um=None):
    """Validate a selector and its parameters, which apply only under ``"mixed"``."""
    if mode not in TRANSPORT_MODES:
        raise ValueError(f"transport_mode must be one of {', '.join(TRANSPORT_MODES)}")
    if mode != "mixed":
        if c1 is not None or c2 is not None or max_step_um is not None:
            raise ValueError("mixed_c1, mixed_c2 and mixed_max_step_um need transport_mode='mixed'")
        return
    validate_mixed_parameters(
        MIXED_DEFAULT_C1 if c1 is None else c1,
        MIXED_DEFAULT_C2 if c2 is None else c2,
        max_step_um,
    )


def mixed_case_keys(mode, c1=None, c2=None, max_step_um=None):
    """Divergence-only case and identity keys; empty for single scattering."""
    if mode != "mixed":
        return {}
    keys = {
        "transport_mode": "mixed",
        "mixed_c1": float(MIXED_DEFAULT_C1 if c1 is None else c1),
        "mixed_c2": float(MIXED_DEFAULT_C2 if c2 is None else c2),
    }
    if max_step_um is not None:
        keys["mixed_max_step_um"] = float(max_step_um)
    return keys


def mixed_provenance_markers(resolved):
    """Model marker recording mixed transport in a resolved physics view."""
    if resolved.get("transport_mode") != "mixed":
        return {}
    return {"mixed_elastic_model": MIXED_ELASTIC_MODEL}
