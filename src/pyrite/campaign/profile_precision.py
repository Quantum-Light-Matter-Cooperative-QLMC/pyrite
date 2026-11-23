"""Profile adaptive-sampling policies and fixed-count transitions.

Tables are staged in the internal stored vocabulary (``precision``,
``min_electrons``, ...); catalog rendering writes ``adaptive_sampling`` and
``*_histories`` (``pyrite._cxr_vocabulary``).
"""

import tomlkit

from pyrite._cxr_vocabulary import catalog_key
from pyrite._numerics import SAMPLING_KEYS, validate_profile_numerics
from pyrite._precision import (
    DEFAULT_PRECISION,
    LEGACY_STORED_KEYS,
    REQUIRED_FIELDS,
    STORED_KEYS,
    Precision,
    default_precision_fallback,
    requests_brem_stopping,
)
from pyrite.detectors.editing import _toml_value

from ._profile_document import existing_profile, profile_overrides, values_item
from .profile_numerics import profile_numerics_values

BREM_REQUIRED_STORED = ("brem_target_rse", "brem_max_electrons")

SAMPLING_FLAGS = {"n_electrons": "cxr-histories", "n_electrons_brem": "brem-histories"}


#: Fields a first partial edit copies from the default policy.
_INITIAL_FIELDS = (*REQUIRED_FIELDS, "min_electrons", "observables")

#: Brem stopping keys and the CXR keys they mirror (stored vocabulary, #1398).
_MIRRORED = {
    "brem_target_rse": "target_rse",
    "brem_max_electrons": "max_electrons",
    "brem_min_electrons": "min_electrons",  # last: capped at the brem maximum
}


def profile_precision_values(profile):
    """Return the explicit adaptive-sampling table as plain values, or ``None``."""
    table = profile.get("precision")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("adaptive_sampling must be a table")
    values = dict(table.unwrap() if hasattr(table, "unwrap") else table)
    # Keys renamed in #1253 (``stability_blocks``) are read under their new name.
    for legacy, key in LEGACY_STORED_KEYS.items():
        if legacy in values:
            values.setdefault(key, values.pop(legacy))
    return values


def policy_view(values):
    """User-facing view of a stored-vocabulary policy: catalog field names, ``cxr``."""
    if values is None:
        return None
    return {
        catalog_key(key): (
            ["cxr" if name == "line" else name for name in value] if key == "observables" else value
        )
        for key, value in values.items()
    }


def precision_resolution(profile):
    """Resolve a profile row's history-count mode for display.

    Returns ``(explicit, effective, source, fixed_reason)``: ``source`` is
    ``"profile"``, ``"default"`` or ``"fixed"``. Policies use catalog field names
    (:func:`policy_view`). Per-material count overrides,
    GDF beams and grooved faces also keep fixed counts; they are resolved per
    run, not here.
    """
    explicit = profile_precision_values(profile)
    if explicit is not None:
        effective = Precision.from_dict(explicit).to_dict()
        return policy_view(explicit), policy_view(effective), "profile", None
    reason = default_precision_fallback(
        fixed_counts=any(key in profile for key in SAMPLING_KEYS),
        emission=profile.get("emission"),
    )
    if reason is not None:
        return None, None, "fixed", reason
    return None, policy_view(DEFAULT_PRECISION.to_dict()), "default", None


def precision_table(values):
    """Build a TOML table from stored-vocabulary policy values."""
    table = tomlkit.table()
    for key in (STORED_KEYS.get(name, name) for name in Precision.__dataclass_fields__):
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
    """Render fixed counts with their CLI spellings: ``cxr-histories 500, brem-histories 100``."""
    return ", ".join(
        f"{SAMPLING_FLAGS[key]} {','.join(str(value) for value in counts[key])}"
        for key in SAMPLING_KEYS
        if key in counts
    )


def precision_blockers(profile):
    """Why adaptive sampling cannot apply to this profile row, before any write.

    Covers the profile-level routes a run refuses: coherent emission,
    physical-detector observation directions and per-material count overrides.
    Particle cascades run adaptive (#1240). GDF beams and grooved targets are
    resolved per run.
    """
    blockers = []
    emission = profile.get("emission")
    if emission in ("coherent", "both"):
        blockers.append(f"emission is {emission!r} (adaptive is incoherent-only)")
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
        raise ValueError(f"profile {name!r} cannot use adaptive sampling: {'; '.join(blockers)}")


def merge_policy(current, updates):
    """Merge ``updates`` over ``current``, or over the default budget for a first policy.

    Brem stopping mirrors the CXR setup (#1398): a request whose brem keys are
    not supplied writes them as the merged CXR values (the minimum capped at the
    brem maximum), so a stored policy never relies on the deprecated
    inheritance; with the target and maximum given, an omitted minimum keeps the
    mirrored default. Setting ``observables`` without brem and no brem key withdraws
    the request and drops its keys.
    """
    if current is None:
        current = {key: DEFAULT_PRECISION.to_dict()[key] for key in _INITIAL_FIELDS}
        current["observables"] = list(current["observables"])
    merged = {**current, **updates}
    observables = {
        "cxr" if name == "line" else name for name in merged.get("observables", ("cxr",))
    }
    if "observables" in updates and observables != {"cxr", "brem"}:
        if not any(key in updates for key in _MIRRORED):
            for key in _MIRRORED:
                merged.pop(key, None)
    if requests_brem_stopping(merged) and "cxr" in observables:
        # Same rule as the Python policy: omitted required keys take the line values,
        # and then the minimum does too; with both required keys given, an omitted
        # minimum is left to the mirrored default (100, at most the brem maximum).
        inherits = any(merged.get(key) is None for key in BREM_REQUIRED_STORED)
        fill_minimum = inherits and merged.get("brem_min_electrons") is None
        for brem_key, key in _MIRRORED.items():
            if merged.get(brem_key) is None and merged.get(key) is not None:
                if brem_key != "brem_min_electrons" or inherits:
                    merged[brem_key] = merged[key]
        if fill_minimum:
            merged["brem_min_electrons"] = min(
                merged["brem_min_electrons"], merged["brem_max_electrons"]
            )
    Precision.from_dict(merged)
    return merged


def set_precision(document, name, updates):
    """Validate and stage an adaptive-sampling policy; return overwritten fields.

    A first policy starts from :data:`DEFAULT_PRECISION`'s budget and
    observables; an existing policy keeps every field ``updates`` leaves out.
    Fixed counts are refused: ``enable_precision`` replaces them atomically.
    """
    target = existing_profile(document, name)
    current = profile_precision_values(target)
    merged = merge_policy(current, updates)
    counts = fixed_counts(target)
    if counts:
        raise ValueError(
            f"profile {name!r} sets fixed history counts ({describe_counts(counts)}); "
            f"replace them with: pyrite profile numerics adaptive-sampling enable {name} [OPTIONS]"
        )
    _check_adaptive(name, target)
    target["precision"] = precision_table(merged)
    current = current or {}
    return tuple(key for key in updates if key in current and current[key] != updates[key])


def enable_precision(document, name, updates):
    """Switch a profile to adaptive sampling atomically.

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
        policy = merge_policy(current, updates)
    counts = fixed_counts(target)
    for key in counts:
        target.pop(key)
    if policy is not None:
        target["precision"] = precision_table(policy)
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
    """Remove selected optional policy fields, or the whole policy when empty."""
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
            f"{', '.join(catalog_key(key) for key in required)} is required by the policy; "
            "reset without fields to remove the policy, or "
            f"'pyrite profile numerics adaptive-sampling disable {name}' to select fixed history counts"
        )
    removed = tuple(field for field in fields if field in current)
    remaining = {key: value for key, value in current.items() if key not in fields}
    missing = [key for key in BREM_REQUIRED_STORED if key not in remaining]
    if missing and requests_brem_stopping(remaining):
        raise ValueError(
            f"{', '.join(catalog_key(key) for key in missing)} is required while brem stopping "
            "is requested; reset the brem fields and observable together, or set them"
        )
    Precision.from_dict(remaining)
    target["precision"] = precision_table(remaining)
    return removed
