"""Profile adaptive precision policies and fixed-count transitions."""

import tomlkit

from pyrite._numerics import SAMPLING_KEYS, validate_profile_numerics
from pyrite._precision import (
    DEFAULT_PRECISION,
    REQUIRED_FIELDS,
    Precision,
    default_precision_fallback,
)
from pyrite.detectors.editing import _toml_value

from ._profile_document import existing_profile, profile_overrides, values_item
from .profile_numerics import profile_numerics_values

SAMPLING_FLAGS = {"n_electrons": "cxr-trials", "n_electrons_brem": "brem-trials"}


_INITIAL_FIELDS = (*REQUIRED_FIELDS, "observables")


def profile_precision_values(profile):
    """Return the explicit ``precision`` table as plain values, or ``None``."""
    table = profile.get("precision")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("precision must be a table")
    return dict(table.unwrap() if hasattr(table, "unwrap") else table)


def precision_resolution(profile):
    """Resolve a profile row's electron-count mode for display.

    Returns ``(explicit, effective, source, fixed_reason)``: ``source`` is
    ``"profile"``, ``"default"`` or ``"fixed"``. Per-material count overrides,
    GDF beams and grooved faces also keep fixed counts; they are resolved per
    run, not here.
    """
    explicit = profile_precision_values(profile)
    if explicit is not None:
        return explicit, Precision.from_dict(explicit).to_dict(), "profile", None
    reason = default_precision_fallback(
        fixed_counts=any(key in profile for key in SAMPLING_KEYS),
        emission=profile.get("emission"),
        secondary_threshold_eV=profile.get("secondary_threshold_eV"),
        pair_production_model=profile.get("pair_production_model"),
        positron_transport=bool(profile.get("positron_transport", False)),
    )
    if reason is not None:
        return None, None, "fixed", reason
    return None, DEFAULT_PRECISION.to_dict(), "default", None


def _precision_table(values):
    table = tomlkit.table()
    for key in Precision.__dataclass_fields__:
        if key in values and values[key] is not None:
            value = values[key]
            if key == "observables":
                value = ["cxr" if name == "line" else name for name in value]
            table[key] = _toml_value(value)
    return table


def fixed_counts(profile):
    """Return the profile's explicit fixed counts as ``{key: [counts]}``."""
    return {key: profile_numerics_values(profile)[key] for key in SAMPLING_KEYS if key in profile}


def describe_counts(counts):
    """Render fixed counts with their CLI spellings: ``line-trials 500, brem-trials 100``."""
    return ", ".join(
        f"{SAMPLING_FLAGS[key]} {','.join(str(value) for value in counts[key])}"
        for key in SAMPLING_KEYS
        if key in counts
    )


def precision_blockers(profile):
    """Why adaptive precision cannot apply to this profile row, before any write.

    Covers the profile-level routes a run refuses: coherent emission, particle
    cascades, positron transport, physical-detector observation directions and
    per-material count overrides. GDF beams and grooved targets are resolved per
    run.
    """
    blockers = []
    emission = profile.get("emission")
    if emission in ("coherent", "both"):
        blockers.append(f"emission is {emission!r} (adaptive is incoherent-only)")
    if "secondary_threshold_eV" in profile or "pair_production_model" in profile:
        blockers.append("particle cascades (secondaries or pair production) are enabled")
    if profile.get("positron_transport"):
        blockers.append("positron transport is enabled")
    if "physical_detector" in profile:
        blockers.append("the physical detector scores observation directions")
    for material, row in sorted(profile_overrides(profile).items()):
        keys = [key for key in SAMPLING_KEYS if isinstance(row, dict) and key in row]
        if keys:
            blockers.append(
                f"[overrides.{material}] sets {', '.join(SAMPLING_FLAGS[key] for key in keys)}; "
                "remove those keys from the catalog first"
            )
    return blockers


def _check_adaptive(name, profile):
    blockers = precision_blockers(profile)
    if blockers:
        raise ValueError(f"profile {name!r} cannot use adaptive precision: {'; '.join(blockers)}")


def _initial_policy(current, updates):
    """Merge ``updates`` over ``current``, or over the default budget for a first policy."""
    if current is None:
        current = {key: DEFAULT_PRECISION.to_dict()[key] for key in _INITIAL_FIELDS}
        current["observables"] = list(current["observables"])
    merged = {**current, **updates}
    Precision.from_dict(merged)
    return merged


def set_precision(document, name, updates):
    """Validate and stage an adaptive precision policy; return overwritten fields.

    A first policy starts from :data:`DEFAULT_PRECISION`'s budget and
    observables; an existing policy keeps every field ``updates`` leaves out.
    Fixed counts are refused: ``enable_precision`` replaces them atomically.
    """
    target = existing_profile(document, name)
    current = profile_precision_values(target)
    merged = _initial_policy(current, updates)
    counts = fixed_counts(target)
    if counts:
        raise ValueError(
            f"profile {name!r} sets fixed electron counts ({describe_counts(counts)}); "
            f"replace them with: pyrite numerics precision enable {name} [OPTIONS]"
        )
    _check_adaptive(name, target)
    target["precision"] = _precision_table(merged)
    current = current or {}
    return tuple(key for key in updates if key in current and current[key] != updates[key])


def enable_precision(document, name, updates):
    """Switch a profile to adaptive precision atomically.

    Removes fixed counts, then writes a policy only when ``updates`` are given
    or one already exists (unspecified fields as in :func:`set_precision`);
    otherwise :data:`DEFAULT_PRECISION` applies. Returns the removed counts;
    nothing is staged on refusal.
    """
    target = existing_profile(document, name)
    _check_adaptive(name, target)
    current = profile_precision_values(target)
    policy = None
    if updates or current is not None:
        policy = _initial_policy(current, updates)
    counts = fixed_counts(target)
    for key in counts:
        target.pop(key)
    if policy is not None:
        target["precision"] = _precision_table(policy)
    return counts


def disable_precision(document, name, counts):
    """Remove any adaptive policy and set fixed ``counts``; return the removed policy.

    ``counts`` maps sampling keys to one count each; a key it omits keeps the
    profile's existing count grid and must already exist.
    """
    target = existing_profile(document, name)
    missing = [
        SAMPLING_FLAGS[key] for key in SAMPLING_KEYS if key not in counts and key not in target
    ]
    if missing:
        raise ValueError(f"fixed counts need {', '.join(missing)}")
    validate_profile_numerics({key: [value] for key, value in counts.items()})
    removed = profile_precision_values(target)
    if removed is not None:
        target.pop("precision")
    for key, value in counts.items():
        if fixed_counts(target).get(key) != [value]:
            target[key] = values_item([value])
    return removed


def reset_precision(document, name, fields=()):
    """Remove selected optional precision fields, or the whole policy when empty."""
    target = existing_profile(document, name)
    current = profile_precision_values(target)
    if current is None:
        return ()
    if not fields:
        target.pop("precision")
        return tuple(current)
    required = [field for field in fields if field in REQUIRED_FIELDS]
    if required:
        raise ValueError(
            f"{', '.join(required)} is required by the policy; reset without fields to "
            f"remove the policy, or 'pyrite numerics precision disable {name}' to select "
            "fixed electron counts"
        )
    removed = tuple(field for field in fields if field in current)
    remaining = {key: value for key, value in current.items() if key not in fields}
    Precision.from_dict(remaining)
    target["precision"] = _precision_table(remaining)
    return removed
