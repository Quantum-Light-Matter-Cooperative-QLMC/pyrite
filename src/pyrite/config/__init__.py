"""Public physics, numerical, and sampling configuration.

This is a lazy canonical import facade. Existing root and implementation imports
remain valid and refer to the same objects; no configuration is converted here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite._config_presets import (
        load_numerics as load_numerics,
    )
    from pyrite._config_presets import (
        load_physics as load_physics,
    )
    from pyrite._numerics import (
        CxrReflections as CxrReflections,
    )
    from pyrite._numerics import (
        Physics as Physics,
    )
    from pyrite._numerics_groups import (
        ExecutionConfig as ExecutionConfig,
    )
    from pyrite._numerics_groups import (
        RadiationNumerics as RadiationNumerics,
    )
    from pyrite._numerics_groups import (
        TransportNumerics as TransportNumerics,
    )
    from pyrite._photon_config import PhotonTransport as PhotonTransport
    from pyrite._precision import Precision as Precision
    from pyrite._sampling import (
        AdaptiveSampling as AdaptiveSampling,
    )
    from pyrite._sampling import (
        FixedSampling as FixedSampling,
    )
    from pyrite.campaign.model import Numerics as Numerics

_EXPORTS = {
    "AdaptiveSampling": ("pyrite._sampling", "AdaptiveSampling"),
    "CxrReflections": ("pyrite._numerics", "CxrReflections"),
    "ExecutionConfig": ("pyrite._numerics_groups", "ExecutionConfig"),
    "FixedSampling": ("pyrite._sampling", "FixedSampling"),
    "Numerics": ("pyrite.campaign.model", "Numerics"),
    "PhotonTransport": ("pyrite._photon_config", "PhotonTransport"),
    "Physics": ("pyrite._numerics", "Physics"),
    "Precision": ("pyrite._precision", "Precision"),
    "RadiationNumerics": ("pyrite._numerics_groups", "RadiationNumerics"),
    "TransportNumerics": ("pyrite._numerics_groups", "TransportNumerics"),
    "load_numerics": ("pyrite._config_presets", "load_numerics"),
    "load_physics": ("pyrite._config_presets", "load_physics"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a canonical API name without eagerly loading numerical engines."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
