"""Grouped SETTING / VALUE / SOURCE views of Numerics and Physics fields.

``pyrite profile numerics show`` and ``pyrite profile physics show`` are
disjoint: Numerics owns sampling, reflections/mosaic, integration and the
channeling grid; Physics owns emission, models, cutoffs and the channeling
model (the API partition ``pyrite._numerics.PHYSICS_FIELDS``). Named presets
render through the same groups.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from pyrite._numerics import CHANNELING_KEYS, CONVERGENCE_KEYS, PHYSICS_FIELDS, SAMPLING_KEYS
from pyrite._precision import Precision
from pyrite.campaign.profile_precision import policy_view
from pyrite.console.tables import emit_settings, value_source

Row = Mapping[str, Any]

_NUMERICS_GROUPS = (
    ("Sampling", (*SAMPLING_KEYS, "precision")),
    ("Reflections and mosaic", CONVERGENCE_KEYS),
    ("CXR grid", ("cxr_grid_policy",)),
    ("Channeling grid", tuple(key for key in CHANNELING_KEYS if key not in PHYSICS_FIELDS)),
)
_PHYSICS_GROUPS = (
    ("Emission", ("emission",)),
    ("Channeling", tuple(key for key in CHANNELING_KEYS if key in PHYSICS_FIELDS)),
)
_POLICY_LABELS = (
    ("target_rse", "target rse"),
    ("min_histories", "min histories"),
    ("max_histories", "max histories"),
    ("brem_target_rse", "brem target rse"),
    ("observables", "observables"),
)


def _grouped(rows: Iterable[Row], groups, rest: str) -> list[tuple[str, list[Row]]]:
    rows = list(rows)
    owner = {key: title for title, keys in groups for key in keys}
    titles = [title for title, _keys in groups]
    # The residual group sits just before the channeling group.
    titles.insert(len(titles) - 1, rest)
    return [
        (title, selected)
        for title in titles
        if (selected := [row for row in rows if owner.get(row["key"], rest) == title])
    ]


def _plain(rows: Iterable[Row]) -> list[tuple[str, object, object]]:
    return [(row["label"], row["effective"], row["source"]) for row in rows]


def sampling_rows(rows: list[Row], adaptive: Mapping[str, Any] | None):
    """Sampling rows led by the history-count mode; fixed counts unused when adaptive."""
    if adaptive is None:
        policy = next((row for row in rows if row["key"] == "precision"), None)
        if policy is None:
            source = next(
                (row["source"] for row in rows if row["explicit"] is not None), "built-in"
            )
            return [("mode", "fixed", source), *_plain(rows)]
        source = policy["source"]
        effective = policy_view(Precision.from_dict(policy["effective"]).to_dict())
        return [
            ("mode", "adaptive", source),
            *((key, value, source) for key, value in effective.items() if value is not None),
        ]
    effective, source = adaptive["effective"], adaptive["source"]
    if effective is None:
        # Fixed mode follows from the profile's own settings (counts, emission, ...).
        origin = next(
            (row["source"] for row in rows if value_source(row["source"]) != "default"), "profile"
        )
        return [("mode", f"fixed ({adaptive['fixed_reason']})", origin), *_plain(rows)]
    return [
        ("mode", f"adaptive ({value_source(source)} policy)", source),
        *(
            (label, effective[key], source)
            for key, label in _POLICY_LABELS
            if effective.get(key) is not None
        ),
        *((row["label"], "unused (adaptive)", row["source"]) for row in rows),
    ]


def emit_numerics(rows: Iterable[Row], *, adaptive: Mapping[str, Any] | None = None) -> None:
    """Render Numerics-owned rows in groups; physics rows are ignored."""
    selected = [row for row in rows if row["key"] not in PHYSICS_FIELDS]
    for title, group in _grouped(selected, _NUMERICS_GROUPS, "Integration"):
        emit_settings(
            title, sampling_rows(group, adaptive) if title == "Sampling" else _plain(group)
        )


def emit_physics(rows: Iterable[Row]) -> None:
    """Render Physics-owned rows (plus emission) in groups."""
    selected = [row for row in rows if row["key"] in PHYSICS_FIELDS or row["key"] == "emission"]
    for title, group in _grouped(selected, _PHYSICS_GROUPS, "Models and cutoffs"):
        emit_settings(title, _plain(group))
