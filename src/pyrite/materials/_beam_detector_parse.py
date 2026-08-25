"""Beam and detector row parsing for material-catalog profiles."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from ._catalog_decode import _Errors, _number, _table

# Beam *distribution* fields settable in a ``[profiles.NAME.beam]`` block. These
# mirror the non-energy fields of ``sweep.BeamSpec`` (energy stays the per-material
# ``ScanSpec.energy_keV`` scan grid -- decision 2); the isotropic aliases
# ``transverse_fwhm_mm`` / ``beam_fwhm_mm`` route onto BOTH transverse planes via
# ``sweep.beam_replace`` when the block is applied in ``config.material_sweep``.
# Kept as local literals here to avoid a ``materials -> sweep`` import cycle.
_BEAM_POSITIVE_KEYS = frozenset(
    {
        "transverse_fwhm_x_mm",
        "transverse_fwhm_y_mm",
        "transverse_fwhm_mm",
        "beam_fwhm_mm",
        "bunch_length_fs",
        "rep_rate_hz",
        "bunch_charge_pc",
        "divergence_mrad",
        "energy_spread_frac",
    }
)
_BEAM_LONG_SHAPES = frozenset({"gaussian", "uniform"})
_BEAM_KEYS = _BEAM_POSITIVE_KEYS | {
    "long_shape",
    "long_offsets_fs",
    "longitudinal",
    "transverse",
}
_DETECTOR_KEYS = frozenset({"observation_angle_deg", "polar_acceptance_deg", "solid_angle_sr"})
#: Accepted range per acceptance field, as ``(minimum, maximum, strictly_positive)``.
#: These mirror ``detectors.spec.Detector.__post_init__`` exactly, including its
#: message wording, so a catalog block and a hand-built ``Detector`` reject the
#: same values. ``campaign.config`` applies a parsed block onto the default
#: detector and would raise there too; the duplication buys a catalog-path error
#: at load time instead of a bare exception mid-sweep.
_DETECTOR_FIELD_BOUNDS: tuple[tuple[str, float, float, bool], ...] = (
    ("observation_angle_deg", 0.0, 180.0, False),
    ("polar_acceptance_deg", 0.0, 180.0, True),
    ("solid_angle_sr", 0.0, 4.0 * math.pi, True),
)
#: Upper-bound wording for the two strictly positive fields, quoted from
#: ``Detector.__post_init__`` so the two paths report a ceiling identically.
_DETECTOR_UPPER_BOUND_MESSAGES = {
    "polar_acceptance_deg": (
        "polar_acceptance_deg is a full polar span and must be <= 180 degrees"
    ),
    "solid_angle_sr": "solid_angle_sr must be <= 4*pi sr",
}
#: A profile that omits a detector, or overrides only acceptance fields, resolves
#: to the driver's default response (Timepix3 at 90 deg, per issue #52).
_DEPRECATED_DETECTOR_KEYS = frozenset(
    {
        "response_model",
        "qe_curve",
        "pixel_pitch_um",
        "sensor_thickness_um",
        "distance_mm",
        "threshold_eV",
    }
)
_FILTER_KEYS = frozenset(
    {
        "name",
        "material",
        "thickness_mm",
        "size_mm",
        "distance_mm",
        "polar_deg",
        "azimuth_deg",
        "roll_deg",
        "offset_mm",
    }
)
_PHYSICAL_DETECTOR_KEYS = frozenset(
    {"distance_mm", "polar_deg", "azimuth_deg", "roll_deg", "offset_mm", "shape", "pitch_mm"}
)
_LONGITUDINAL_KINDS = frozenset({"gaussian", "microtrain", "compressed"})
_LONGITUDINAL_KEYS = frozenset(
    {
        "kind",
        "envelope_rms_fs",
        "retained_coherence",
        "target_reflection",
        "spacing_periods",
        "modulation_depth",
        "timing_jitter_fs",
    }
)

# alpha_twiss is legitimately negative -- a diverging beam past its waist -- so
# the Twiss keys split into positive-magnitude and signed sets. Putting them all
# in a positive-only set would reject valid profiles.
_TRANSVERSE_POSITIVE_KEYS = frozenset(
    {
        "normalized_emittance_x_mm_mrad",
        "beta_twiss_x_m",
        "normalized_emittance_y_mm_mrad",
        "beta_twiss_y_m",
    }
)
_TRANSVERSE_SIGNED_KEYS = frozenset({"alpha_twiss_x", "alpha_twiss_y"})
_TRANSVERSE_KEYS = _TRANSVERSE_POSITIVE_KEYS | _TRANSVERSE_SIGNED_KEYS
_TRANSVERSE_REQUIRED_KEYS = ("normalized_emittance_x_mm_mrad", "beta_twiss_x_m")


def _parse_longitudinal_policy(raw: object, path: str, errors: _Errors) -> dict[str, object] | None:
    """Validate one declarative longitudinal distribution policy."""
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_LONGITUDINAL_KEYS))
    kind = table.get("kind")
    if not isinstance(kind, str) or kind not in _LONGITUDINAL_KINDS:
        errors.add(f"{path}.kind", f"must be one of {sorted(_LONGITUDINAL_KINDS)}")
        return None

    out: dict[str, object] = {"kind": kind}
    envelope = table.get("envelope_rms_fs")
    if kind in {"gaussian", "microtrain"}:
        number = _number(envelope)
        if number is None or number <= 0:
            errors.add(f"{path}.envelope_rms_fs", "must be a finite positive number")
        else:
            out["envelope_rms_fs"] = number
    elif envelope is not None:
        errors.add(f"{path}.envelope_rms_fs", "must be omitted for compressed")

    eta = table.get("retained_coherence", 0.9)
    eta_number = _number(eta)
    if eta_number is None or not 0 < eta_number <= 1:
        errors.add(f"{path}.retained_coherence", "must satisfy 0 < eta <= 1")
    elif "retained_coherence" in table:
        out["retained_coherence"] = eta_number

    spacing = table.get("spacing_periods", 1)
    if type(spacing) is not int or spacing < 1:
        errors.add(f"{path}.spacing_periods", "must be a positive integer")
    elif "spacing_periods" in table:
        out["spacing_periods"] = spacing

    modulation = table.get("modulation_depth", 1.0)
    modulation_number = _number(modulation)
    if modulation_number is None or not 0 <= modulation_number <= 1:
        errors.add(f"{path}.modulation_depth", "must satisfy 0 <= depth <= 1")
    elif "modulation_depth" in table:
        out["modulation_depth"] = modulation_number

    jitter = table.get("timing_jitter_fs", 0.0)
    jitter_number = _number(jitter)
    if jitter_number is None or jitter_number < 0:
        errors.add(f"{path}.timing_jitter_fs", "must be finite and non-negative")
    elif "timing_jitter_fs" in table:
        out["timing_jitter_fs"] = jitter_number

    reflection = table.get("target_reflection")
    if reflection is not None:
        valid = (
            isinstance(reflection, list)
            and len(reflection) == 3
            and all(type(item) is int for item in reflection)
            and any(reflection)
        )
        if not valid:
            errors.add(f"{path}.target_reflection", "must be a nonzero integer triple")
        else:
            out["target_reflection"] = tuple(reflection)

    targeted = (
        reflection is not None
        or eta_number != 0.9
        or spacing != 1
        or modulation_number != 1.0
        or jitter_number != 0.0
    )
    if kind == "gaussian" and targeted:
        errors.add(path, "gaussian does not accept target-line or modulation controls")
    return out


def _parse_transverse_policy(raw: object, path: str, errors: _Errors) -> dict[str, object] | None:
    """Validate one declarative transverse Courant-Snyder policy.

    The x-plane emittance and beta are required; the y-plane keys are optional
    and mirror x when omitted. ``alpha_twiss_*`` is signed and only has to be
    finite -- see :data:`_TRANSVERSE_SIGNED_KEYS`.
    """
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_TRANSVERSE_KEYS))
    out: dict[str, object] = {}
    for key, value in table.items():
        number = _number(value)
        if key in _TRANSVERSE_SIGNED_KEYS:
            if number is None:
                errors.add(f"{path}.{key}", "must be a finite number")
            else:
                out[key] = number
        elif key in _TRANSVERSE_POSITIVE_KEYS:
            if number is None or number <= 0:
                errors.add(f"{path}.{key}", "must be a finite positive number")
            else:
                out[key] = number
    missing = [key for key in _TRANSVERSE_REQUIRED_KEYS if key not in out]
    if missing:
        errors.add(path, f"missing required {', '.join(missing)}")
        return None
    return out


def _parse_profile_beam(raw: object, path: str, errors: _Errors) -> dict[str, object] | None:
    """Structurally validate a ``[profiles.NAME.beam]`` distribution block.

    Distribution fields only (transverse size, bunch length/shape/offsets,
    rep-rate, charge, reserved divergence/spread) -- ``energy_keV`` is NOT
    accepted (it stays the per-material scan grid, decision 2). Positive
    magnitudes must be finite and ``> 0``; ``long_shape`` is one of
    :data:`_BEAM_LONG_SHAPES`; ``long_offsets_fs`` is an array of finite numbers
    (any sign) coerced to a tuple. Returns the cleaned mapping, or ``None`` when
    the block is empty or fully rejected.
    """
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_BEAM_KEYS))
    out: dict[str, object] = {}
    for key, value in table.items():
        if key == "long_shape":
            if not isinstance(value, str) or value not in _BEAM_LONG_SHAPES:
                errors.add(f"{path}.long_shape", f"must be one of {sorted(_BEAM_LONG_SHAPES)}")
            else:
                out[key] = value
        elif key == "long_offsets_fs":
            offsets = [_number(item) for item in value] if isinstance(value, list) else None
            if not offsets or any(item is None for item in offsets):
                errors.add(f"{path}.long_offsets_fs", "must be a non-empty array of finite numbers")
            else:
                out[key] = tuple(offsets)
        elif key == "longitudinal":
            policy = _parse_longitudinal_policy(value, f"{path}.longitudinal", errors)
            if policy is not None:
                out[key] = MappingProxyType(policy)
        elif key == "transverse":
            transverse = _parse_transverse_policy(value, f"{path}.transverse", errors)
            if transverse is not None:
                out[key] = MappingProxyType(transverse)
        elif key in _BEAM_POSITIVE_KEYS:
            number = _number(value)
            if number is None or number <= 0:
                errors.add(f"{path}.{key}", "must be a finite positive number")
            else:
                out[key] = number
    return out or None


def _parse_beams(raw: object, errors: _Errors) -> dict[str, Mapping[str, object]]:
    """Parse ``[beams.NAME]`` named beam objects.

    Same distribution fields as an inline ``[profiles.NAME.beam]`` block, plus
    an optional ``label``, validated with the identical field parser
    (:func:`_parse_profile_beam`) so a named beam and its equivalent inline
    block decode to bit-identical payloads. ``label`` is display-only metadata
    and never joins the resolved payload a profile reference produces (decision
    3: only field *values* may affect ``parameter_sha256``, never the beam's
    name or label).
    """
    table = _table(raw, "beams", errors)
    if table is None:
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for key, value in table.items():
        path = f"beams.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        label = row.get("label")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            errors.add(f"{path}.label", "must be a nonempty string")
            label = None
        fields = {k: v for k, v in row.items() if k != "label"}
        beam = _parse_profile_beam(fields, path, errors)
        if beam is None:
            errors.add(path, "must define at least one beam field")
            continue
        if label is not None:
            beam = {**beam, "label": label}
        out[key] = MappingProxyType(beam)
    return out


def _parse_profile_detector(raw: object, path: str, errors: _Errors) -> Mapping[str, object] | None:
    """Validate one portable ``[profiles.NAME.detector]`` block.

    Returns the validated acceptance fields, not a built ``Detector``: this
    module owns configuration parsing and stays below the detector forward
    models, so a driver -- ``campaign.config`` -- applies the result onto the
    default detector. This is the same shape ``_parse_physical_detector``
    already hands to ``instrument``.

    Fields are checked in ``Detector.__post_init__`` order and the first failure
    wins, so a block that was rejected before is rejected here with the same
    message.
    """
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_DETECTOR_KEYS | _DEPRECATED_DETECTOR_KEYS))
    for key in sorted(table.keys() & _DEPRECATED_DETECTOR_KEYS):
        warnings.warn(
            f"{path}.{key} is deprecated and ignored; configure a Detector response object",
            DeprecationWarning,
            stacklevel=3,
        )
    spec: dict[str, object] = {}
    for key, minimum, maximum, strictly_positive in _DETECTOR_FIELD_BOUNDS:
        if key not in table:
            continue
        raw_value = table[key]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            errors.add(path, f"{key} must be a real number")
            return None
        value = float(raw_value)
        if not math.isfinite(value):
            errors.add(path, f"{key} must be finite")
            return None
        # A strictly positive field is range-checked against its own upper bound
        # only after the shared ">= 0" test, matching _positive_optional().
        upper = maximum if not strictly_positive else None
        if value < minimum or (upper is not None and value > upper):
            bound = "" if upper is None else f" and <= {upper:g}"
            errors.add(path, f"{key} must be >= {minimum:g}{bound}")
            return None
        if strictly_positive:
            if value == 0.0:
                errors.add(path, f"{key} must be positive")
                return None
            if value > maximum:
                errors.add(path, _DETECTOR_UPPER_BOUND_MESSAGES[key])
                return None
        spec[key] = value
    return MappingProxyType(spec)


def _pair(
    raw: object, path: str, errors: _Errors, *, integer: bool = False
) -> tuple[int | float, int | float] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        errors.add(path, "must be a two-item array")
        return None
    if integer:
        if any(type(item) is not int or item <= 0 for item in raw):
            errors.add(path, "must contain two positive integers")
            return None
        return cast(tuple[int, int], (raw[0], raw[1]))
    first = _number(raw[0])
    second = _number(raw[1])
    if first is None or second is None:
        errors.add(path, "must contain two finite numbers")
        return None
    return first, second


def _parse_filter_rows(
    raw: object, path: str, errors: _Errors
) -> tuple[Mapping[str, object], ...] | None:
    if not isinstance(raw, list):
        errors.add(path, "must be an array of tables")
        return None
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(raw):
        row_path = f"{path}[{index}]"
        row = _table(item, row_path, errors)
        if row is None:
            continue
        errors.keys(row, row_path, set(_FILTER_KEYS))
        material = row.get("material")
        if not isinstance(material, str) or not material:
            errors.add(f"{row_path}.material", "must be a nonempty catalog key")
        name = row.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            errors.add(f"{row_path}.name", "must be a nonempty string")
        cleaned: dict[str, object] = {key: row[key] for key in ("name", "material") if key in row}
        for key in ("thickness_mm", "distance_mm"):
            value = _number(row.get(key))
            if value is None or value <= 0:
                errors.add(f"{row_path}.{key}", "must be a finite positive number")
            else:
                cleaned[key] = value
        size = _pair(row.get("size_mm"), f"{row_path}.size_mm", errors)
        if size is not None:
            if any(float(item) <= 0 for item in size):
                errors.add(f"{row_path}.size_mm", "must contain positive numbers")
            else:
                cleaned["size_mm"] = size
        for key, default in (("polar_deg", 90.0), ("azimuth_deg", 0.0), ("roll_deg", 0.0)):
            value = _number(row.get(key, default))
            if value is None or key == "polar_deg" and not 0 <= value <= 180:
                errors.add(
                    f"{row_path}.{key}",
                    "must be finite" if key != "polar_deg" else "must be between 0 and 180",
                )
            else:
                cleaned[key] = value
        offset = _pair(row.get("offset_mm", [0.0, 0.0]), f"{row_path}.offset_mm", errors)
        if offset is not None:
            cleaned["offset_mm"] = offset
        rows.append(MappingProxyType(cleaned))
    return tuple(rows)


def _parse_physical_detector(
    raw: object, path: str, errors: _Errors
) -> Mapping[str, object] | None:
    row = _table(raw, path, errors)
    if row is None:
        return None
    errors.keys(row, path, set(_PHYSICAL_DETECTOR_KEYS))
    cleaned: dict[str, object] = {}
    distance = _number(row.get("distance_mm"))
    if distance is None or distance <= 0:
        errors.add(f"{path}.distance_mm", "must be a finite positive number")
    else:
        cleaned["distance_mm"] = distance
    for key, default in (("polar_deg", 90.0), ("azimuth_deg", 0.0), ("roll_deg", 0.0)):
        value = _number(row.get(key, default))
        if value is None or key == "polar_deg" and not 0 <= value <= 180:
            errors.add(
                f"{path}.{key}",
                "must be finite" if key != "polar_deg" else "must be between 0 and 180",
            )
        else:
            cleaned[key] = value
    offset = _pair(row.get("offset_mm", [0.0, 0.0]), f"{path}.offset_mm", errors)
    if offset is not None:
        cleaned["offset_mm"] = offset
    shape = _pair(row.get("shape", [256, 256]), f"{path}.shape", errors, integer=True)
    if shape is not None:
        cleaned["shape"] = shape
    pitch = _pair(row.get("pitch_mm", [0.055, 0.055]), f"{path}.pitch_mm", errors)
    if pitch is not None:
        if any(float(item) <= 0 for item in pitch):
            errors.add(f"{path}.pitch_mm", "must contain positive numbers")
        else:
            cleaned["pitch_mm"] = pitch
    return MappingProxyType(cleaned)


def _parse_detectors(
    raw: object, errors: _Errors
) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
    """Parse named ``[detectors.NAME]`` geometry objects.

    Named objects use the same decoder as legacy inline profile detector
    blocks. ``label`` is display-only and is never part of the resolved
    acceptance spec.
    """
    table = _table(raw, "detectors", errors)
    if table is None:
        return {}, {}
    detectors: dict[str, Mapping[str, object]] = {}
    labels: dict[str, str] = {}
    for key, value in table.items():
        path = f"detectors.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        label = row.get("label")
        if label is not None:
            if not isinstance(label, str) or not label.strip():
                errors.add(f"{path}.label", "must be a nonempty string")
            else:
                labels[key] = label
        fields = {name: item for name, item in row.items() if name != "label"}
        if not fields:
            errors.add(path, "must define at least one detector geometry field")
            continue
        detector = _parse_profile_detector(fields, path, errors)
        if detector is not None:
            detectors[key] = detector
    return detectors, labels
