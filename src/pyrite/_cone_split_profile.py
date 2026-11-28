"""The catalog profile ``cone_split`` key (#1384) for detector-cone splitting (#379).

``cone_split = true`` opts a profile into :class:`~pyrite._cone_splitting.ConeSplitting`
with the measured defaults; a ``[profiles.NAME.cone_split]`` table overrides
individual fields. TOML has no null, so ``angle_scale = "none"`` disables the
angle-scaled roulette. The estimator ``model`` is versioned by PyRITE and is not
a profile field. Absent, transport stays analog and nothing joins case or
dataset identity. Import-light: the material catalog parser uses it.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ._cone_splitting import _REFUSED_KEYS, LOW_GAIN_ENERGY_KEV, ConeSplitting

#: Profile fields, in display order, with their CLI spellings.
FIELDS = {
    "cone_factor": "cone-factor",
    "min_energy_fraction": "min-energy-fraction",
    "angle_scale": "angle-scale",
    "target_weight": "target-weight",
    "max_copies_per_electron": "max-copies-per-electron",
}
NONE = "none"


def cone_split_from_profile(value: object) -> ConeSplitting:
    """Return the policy a profile ``cone_split`` value describes.

    Raises ``ValueError`` naming the bad field or the accepted forms.
    """
    if value is True:
        return ConeSplitting()
    if not isinstance(value, Mapping):
        raise ValueError(
            "must be true (the measured defaults) or a table of cone split fields; "
            "remove the key to turn splitting off"
        )
    payload = dict(value)
    unknown = sorted(set(payload) - set(FIELDS))
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; allowed: {sorted(FIELDS)}")
    if payload.get("angle_scale") == NONE:
        payload["angle_scale"] = None
    try:
        return ConeSplitting(**payload)
    except TypeError as exc:
        raise ValueError(str(exc)) from None


def stored_fields(policy: ConeSplitting) -> dict[str, Any]:
    """Profile spellings of ``policy``'s fields (``angle_scale`` ``None`` -> ``"none"``)."""
    values = {key: getattr(policy, key) for key in FIELDS}
    if values["angle_scale"] is None:
        values["angle_scale"] = NONE
    return values


def _on(value: object) -> bool:
    return value not in (None, False, "off", NONE)


def profile_refusals(row: Mapping[str, object]) -> list[str]:
    """Reasons ``row`` (a preset-folded profile) cannot run detector-cone splitting.

    Mirrors :meth:`ConeSplitting.validate_case` on the profile keys that decide
    it, so a refused combination fails at catalog load rather than mid-run.
    Per-material routes (GDF beams, grooved faces) still refuse at case build.
    """
    reasons = []
    emission = row.get("emission")
    if emission in ("coherent", "both"):
        reasons.append(f"emission {emission!r}: {_REFUSED_KEYS['coherent_emission']}")
    for key in (
        "temporal_profile",
        "secondary_threshold_eV",
        "pair_production_model",
        "positron_transport",
        "transition_radiation",
    ):
        if _on(row.get(key)):
            reasons.append(f"{key}: {_REFUSED_KEYS[key]}")
    if _on(row.get("channeling_model")):
        reasons.append(f"channeling: {_REFUSED_KEYS['channeling']}")
    if row.get("transport_mode") == "mixed":
        reasons.append("transport_mode 'mixed' has no split port")
    if row.get("elastic_model") == "mott":
        reasons.append("elastic_model 'mott': splitting needs the default elastic_model 'elsepa'")
    if "physical_detector" in row:
        reasons.append("a physical pixel detector has no single detector direction")
    return reasons


def energy_refusal(policy: ConeSplitting, energies_keV: Iterable[float]) -> str | None:
    """The cone half-angle refusal for the first beam energy that has one, else ``None``."""
    for energy in energies_keV:
        try:
            policy.half_angle_rad(float(energy))
        except ValueError as exc:
            return str(exc)
    return None


def low_gain_energies(energies_keV: Iterable[float]) -> list[float]:
    """Beam energies below the #203 near-unit-gain threshold."""
    return [float(energy) for energy in energies_keV if float(energy) < LOW_GAIN_ENERGY_KEV]


def check_profile(
    row: Mapping[str, object],
    path: str,
    energies_keV: Iterable[float] | None,
    add: Callable[[str, str], None],
) -> ConeSplitting | None:
    """Validate ``row["cone_split"]``, if any; report each problem with ``add(path, message)``."""
    if "cone_split" not in row:
        return None
    key_path = f"{path}.cone_split"
    try:
        policy = cone_split_from_profile(row["cone_split"])
    except ValueError as exc:
        add(key_path, str(exc))
        return None
    for reason in profile_refusals(row):
        add(key_path, f"detector-cone splitting does not support {reason}")
    refusal = None if energies_keV is None else energy_refusal(policy, energies_keV)
    if refusal is not None:
        add(key_path, refusal)
    return policy


def profile_cone_splits(profiles: Mapping[str, Mapping[str, object]]) -> dict[str, ConeSplitting]:
    """Policies of the validated profile rows that opt in, keyed by profile."""
    return {
        name: cone_split_from_profile(row["cone_split"])
        for name, row in profiles.items()
        if "cone_split" in row
    }


__all__ = [
    "FIELDS",
    "check_profile",
    "cone_split_from_profile",
    "energy_refusal",
    "low_gain_energies",
    "profile_cone_splits",
    "profile_refusals",
    "stored_fields",
]
