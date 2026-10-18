"""Profile line-grid policy validation, edits and source inspection."""

import tomlkit

from pyrite._cxr_grid_policy import ENVIRONMENT_NAMES as CXR_GRID_ENVIRONMENT_NAMES
from pyrite._cxr_grid_policy import PROFILE_CXR_GRID_SELECTORS, check_selector_combination
from pyrite._env import env_value

from ._profile_document import energy_grid_refs, existing_profile, profile_overrides


class CxrGridPolicyConflict(ValueError):
    """A line-grid policy combination no case could resolve."""


def profile_cxr_grid_values(profile):
    """Return the explicit ``line_grid_policy`` table as plain values, or ``None``."""
    table = profile.get("line_grid_policy")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("line_grid_policy must be a table")
    return dict(table.unwrap() if hasattr(table, "unwrap") else table)


def cxr_grid_fields(explicit):
    """Explicit, effective and source rows for each profile line-grid selector."""
    explicit = explicit or {}
    return [
        {
            "key": key,
            "explicit": explicit.get(key),
            "effective": explicit.get(key, choices[0]),
            "source": "profile" if key in explicit else "built-in default",
            "choices": list(choices),
        }
        for key, choices in PROFILE_CXR_GRID_SELECTORS.items()
    ]


def _check_cxr_grid_policy(profile, policy):
    """Refuse a non-empty POLICY that no case of PROFILE could resolve."""
    if not policy:
        return
    effective = {row["key"]: row["effective"] for row in cxr_grid_fields(policy)}
    try:
        check_selector_combination(**effective)
    except ValueError as exc:
        raise CxrGridPolicyConflict(str(exc)) from None
    # Automatic resolution refuses the coherent route without feature windows
    # (#117); `windows = true` resolves it inside per-line windows (#350).
    if profile.get("emission") in ("coherent", "both") and policy.get("windows") is not True:
        raise CxrGridPolicyConflict(
            "a line-grid policy without feature windows resolves every line grid "
            f"automatically, which the {profile['emission']} emission route does not "
            "support (#117); add 'windows = true' to [line_grid_policy], or keep an "
            "explicit or stored line grid for coherent emission"
        )
    max_de_frac = profile.get("max_dE_frac", 0.0)
    if effective["quadrature"] == "bin-mean" and max_de_frac > 0.0:
        raise CxrGridPolicyConflict(
            "bin-mean quadrature is incompatible with a positive max_dE_frac; reset it "
            "with 'pyrite profile numerics reset NAME maximum-fractional-energy-loss'"
        )


def set_cxr_grid_policy(document, name, updates):
    """Validate and stage line-grid selectors; return ``{key: previous}`` for changed keys."""
    target = existing_profile(document, name)
    current = profile_cxr_grid_values(target) or {}
    merged = {**current, **updates}
    _check_cxr_grid_policy(target, merged)
    # Edit in place: TOML-only keys (`windows`, `max_points`) stay as written.
    table = target.get("line_grid_policy")
    if table is None:
        table = target["line_grid_policy"] = tomlkit.table()
    for key in PROFILE_CXR_GRID_SELECTORS:
        if key in updates:
            table[key] = updates[key]
    return {key: current.get(key) for key, value in updates.items() if current.get(key) != value}


def reset_cxr_grid_policy(document, name, fields=()):
    """Remove selected selectors, or the whole policy when FIELDS is empty."""
    target = existing_profile(document, name)
    current = profile_cxr_grid_values(target)
    if current is None:
        return ()
    selected = tuple(fields) or tuple(current)
    removed = tuple(key for key in selected if key in current)
    remaining = {key: value for key, value in current.items() if key not in selected}
    _check_cxr_grid_policy(target, remaining)
    if remaining:
        for key in removed:
            target["line_grid_policy"].pop(key)
    else:
        target.pop("line_grid_policy")
    return removed


def cxr_grid_sources(document, profile):
    """Describe which line-grid source this profile's cases use, before resolution.

    Mirrors ``campaign.sweep._cxr_grid_for_energy``: any profile policy turns
    automatic resolution on for every case and outranks explicit and stored
    grids; otherwise explicit ``E_grid_line`` wins, then a stored per-energy row
    (``energy_grid_refs``/``E_grid_line_by_energy``), then automatic resolution
    for uncovered energies. ``PYRITE_ENERGY_GRID_*`` outranks stored rows only.
    """
    policy = profile_cxr_grid_values(profile)
    overrides = profile_overrides(profile)
    shared = document.get("energy_grids", {})
    environment = [name for name in CXR_GRID_ENVIRONMENT_NAMES if env_value(name) is not None]
    return {
        "automatic_for_every_case": bool(policy),
        "explicit_line_grid": "E_grid_line" in profile,
        "explicit_line_grid_overrides": sorted(
            material
            for material, row in overrides.items()
            if isinstance(row, dict) and "E_grid_line" in row
        ),
        "energy_grid_refs": sorted(energy_grid_refs(profile)),
        "shared_energy_grids": sorted(shared if isinstance(shared, dict) else ()),
        "environment": environment,
    }
