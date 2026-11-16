"""The ``none`` spelling that turns an optional setting off (#1242).

Convention ("off follows type", ``docs/repo-design/profile-settings.md``):
boolean features turn off with ``--no-x``/``False``; non-boolean optional
settings (a model choice or an enabling threshold) turn off with the literal
``"none"``. ``"off"`` is a deprecated alias of ``"none"``.

TOML has no null, so an explicit ``"none"`` is stored: it overrides a preset
that enables the setting, whereas an absent key inherits the preset. Readers
map the stored spelling to the typed objects' off value with
:func:`normalize_off_values`; internal case, ``Settings`` and identity records
keep their historical spelling, so ``none``, ``off`` and absent resolve to the
same digests.

Import-light: the catalog parser, presets, profile editing and the typed
``Physics`` share it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

#: Public spelling of an optional non-boolean setting that is off.
NONE = "none"
#: Deprecated alias of :data:`NONE`.
DEPRECATED_OFF = "off"
#: Off-able field -> the value the typed objects and internal records use.
#: ``channeling_model`` is the flat ``[channeling] model`` value (the public
#: API passes ``channeling=None``); stored ``Settings`` keep ``"off"``.
OFF_VALUES: dict[str, object] = {
    "secondary_threshold_eV": None,
    "pair_production_model": None,
    "transition_radiation": NONE,
    "channeling_model": NONE,
}


def warn_off_spelling(field: str, *, stacklevel: int = 3) -> None:
    """Warn that ``"off"`` for ``field`` is deprecated in favour of ``"none"``."""
    from ._util.deprecations import API_DEPRECATIONS

    entry = API_DEPRECATIONS['"off"']
    warnings.warn(
        f'{field}="off" is deprecated and will be removed in {entry.remove_in}; '
        f'use {field}="none" instead',
        FutureWarning,
        stacklevel=stacklevel,
    )


def off_value(field: str, value: Any, *, warn: bool = True) -> Any:
    """Map ``"none"`` (or deprecated ``"off"``) for ``field`` to its off value.

    Other values, and fields without an off spelling, pass through. ``warn``
    is false only where an internal record already holds ``"off"``.
    """
    if field not in OFF_VALUES or value not in (NONE, DEPRECATED_OFF):
        return value
    if value == DEPRECATED_OFF and warn:
        warn_off_spelling(field, stacklevel=4)
    return OFF_VALUES[field]


def normalize_off_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``values`` with every off spelling mapped to its field's off value."""
    return {key: off_value(key, value) for key, value in values.items()}


def is_on(field: str, value: Any) -> bool:
    """Whether a stored or typed ``value`` enables off-able ``field``."""
    return value is not None and off_value(field, value, warn=False) != OFF_VALUES[field]
