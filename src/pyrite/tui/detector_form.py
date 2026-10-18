"""UI-independent detector form model for the terminal editor.

Fields mirror the ``pyrite detector create scalar|planar`` options one to one:
the same option names, units and input domains. Typed values go through
:func:`pyrite.detectors.editing.pixel_changes`, :func:`create_detector` and
:func:`patch_detector` into a :class:`~pyrite.catalog_edit.CatalogDraft`, so an
editor edit serializes exactly like the equivalent CLI command (also returned,
see :meth:`StagedEdit.cli_args`). Nothing here writes until
:meth:`DetectorSession.save`.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite._catalog_layout import object_sources, read_text, require_writable_object
from pyrite._cxr_vocabulary import translate_catalog_text
from pyrite.catalog_edit import CatalogDraft
from pyrite.console.output import (
    FINITE_FLOAT,
    MEASURED_EDGES_CSV,
    NONNEGATIVE_FLOAT,
    NONNEGATIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
)
from pyrite.detectors.editing import create_detector, patch_detector, pixel_changes

KINDS = ("scalar", "planar")
_POSITIVE_PAIR = click.Tuple((POSITIVE_FLOAT, POSITIVE_FLOAT))


@dataclass(frozen=True)
class Field:
    """One editor input; ``key`` is the CLI option's parameter name."""

    key: str
    flag: str
    label: str
    unit: str
    type: click.ParamType
    kind: str
    section: str
    help: str

    @property
    def choices(self):
        return tuple(self.type.choices) if isinstance(self.type, click.Choice) else None

    @property
    def arity(self):
        return len(self.type.types) if isinstance(self.type, click.Tuple) else 1


def _f(key, flag, label, unit, kind, section, type_, help_):
    return Field(key, flag, label, unit, type_, kind, section, help_)


# fmt: off
FIELDS = (
    _f("observation_angle_deg", "--observation-angle", "Observation angle", "deg", "scalar",
       "Acceptance", click.FloatRange(min=0.0, max=180.0), "Angle from the beam axis, [0, 180]."),
    _f("polar_acceptance_deg", "--polar-acceptance", "Polar acceptance", "deg", "scalar",
       "Acceptance", click.FloatRange(min=0.0, max=180.0, min_open=True),
       "Full polar acceptance span, (0, 180]."),
    _f("solid_angle_sr", "--solid-angle", "Solid angle", "sr", "scalar", "Acceptance",
       click.FloatRange(min=0.0, max=4.0 * math.pi, min_open=True), "Solid angle, (0, 4 pi]."),
    _f("distance_mm", "--distance-mm", "Distance", "mm", "planar", "Geometry", POSITIVE_FLOAT,
       "Source-to-detector distance; required to create."),
    _f("polar_deg", "--polar-deg", "Centre polar angle", "deg", "planar", "Geometry",
       click.FloatRange(0.0, 180.0), "Detector-centre polar angle from the beam axis, [0, 180]."),
    _f("azimuth_deg", "--azimuth-deg", "Centre azimuth", "deg", "planar", "Geometry",
       FINITE_FLOAT, "Detector-centre azimuth."),
    _f("roll_deg", "--roll-deg", "Roll", "deg", "planar", "Geometry", FINITE_FLOAT,
       "Roll about the detector normal."),
    _f("offset_mm", "--offset-mm", "Offset X Y", "mm", "planar", "Geometry",
       click.Tuple((FINITE_FLOAT, FINITE_FLOAT)), "Detector-local centre offset."),
    _f("shape", "--shape", "Pixels ROWS COLS", "", "planar", "Geometry",
       click.Tuple((POSITIVE_INT, POSITIVE_INT)), "Pixel counts; new tables default to 256 256."),
    _f("pitch_mm", "--pitch-mm", "Pitch Y X", "mm", "planar", "Geometry", _POSITIVE_PAIR,
       "Pixel pitch; new tables default to 0.055 0.055."),
    _f("angular_shape", "--angular-shape", "Angular shape ROWS COLS", "", "planar", "Scoring",
       click.Tuple((POSITIVE_INT, POSITIVE_INT)), "Directions evaluated per transport; <= pixels."),
    _f("reconstruction", "--reconstruction", "Reconstruction", "", "planar", "Scoring",
       click.Choice(("nearest_tile", "bilinear_tile")), "Own tile, or blended neighbour tiles."),
    _f("response", "--response", "Response", "", "planar", "Response",
       click.Choice(("ideal", "timepix3")), "Unit-efficiency ideal, or uncalibrated Timepix3."),
    _f("timepix_thickness_um", "--timepix-thickness-um", "Timepix3 thickness", "um", "planar",
       "Response", POSITIVE_FLOAT, "Sensor thickness; timepix3 only."),
    _f("timepix_bias_v", "--timepix-bias-v", "Timepix3 bias", "V", "planar", "Response",
       POSITIVE_FLOAT, "Sensor bias; timepix3 only."),
    _f("timepix_input_bin_ev", "--timepix-input-bin-ev", "Timepix3 input bin", "eV", "planar",
       "Response", POSITIVE_FLOAT, "Response-model incident-energy bin width."),
    _f("timepix_output_bin_ev", "--timepix-output-bin-ev", "Timepix3 output bin", "eV",
       "planar", "Response", POSITIVE_FLOAT, "Native measured-energy bin width."),
    _f("timepix_samples", "--timepix-samples", "Timepix3 samples", "", "planar", "Response",
       POSITIVE_INT, "Response Monte Carlo photons per input bin."),
    _f("timepix_seed", "--timepix-seed", "Timepix3 seed", "", "planar", "Response",
       NONNEGATIVE_INT, "Response-matrix seed."),
    _f("exposure_s", "--exposure-s", "Exposure", "s", "planar", "Acquisition", POSITIVE_FLOAT,
       "Integration time."),
    _f("reporting_bins", "--reporting-bins", "Reporting bins", "", "planar", "Acquisition",
       click.Choice(("native",)), "Union of native response bins; excludes explicit bins."),
    _f("measured_edges_ev", "--measured-edges-ev", "Measured edges", "eV", "planar",
       "Acquisition", MEASURED_EDGES_CSV, "Comma-separated, strictly increasing bin edges."),
    _f("measured_range_ev", "--measured-range-ev", "Measured range MIN MAX", "eV", "planar",
       "Acquisition", click.Tuple((NONNEGATIVE_FLOAT, NONNEGATIVE_FLOAT)),
       "Uniform range; set the bin width too."),
    _f("measured_bin_width_ev", "--measured-bin-width-ev", "Measured bin width", "eV", "planar",
       "Acquisition", POSITIVE_FLOAT, "Uniform reporting-bin width (not resolution)."),
    _f("hit_threshold_ev", "--hit-threshold-ev", "Hit threshold", "eV", "planar", "Acquisition",
       NONNEGATIVE_FLOAT, "Post-response measured-energy cut."),
    _f("realization", "--realization", "Realization", "", "planar", "Acquisition",
       click.Choice(("expected", "poisson")), "Expected counts or a seeded Poisson draw."),
    _f("realization_seed", "--realization-seed", "Realization seed", "", "planar", "Acquisition",
       NONNEGATIVE_INT, "Poisson realization seed; poisson only."),
)
# fmt: on
BY_KEY = {item.key: item for item in FIELDS}
LABEL = "label"
# Fields the shared builder checks together: a change to one re-sends the rest.
_GROUPS = (
    {"reporting_bins", "measured_edges_ev", "measured_range_ev", "measured_bin_width_ev"},
    {"realization", "realization_seed"},
    {"response", *(key for key in BY_KEY if key.startswith("timepix_"))},
)
# TOML location of each option: (section or None, key[, key]).
_TOML = {
    **{key: (None, key) for key in ("observation_angle_deg", "polar_acceptance_deg")},
    "solid_angle_sr": (None, "solid_angle_sr"),
    **{key: (None, key) for key in ("distance_mm", "polar_deg", "azimuth_deg", "roll_deg")},
    **{key: (None, key) for key in ("offset_mm", "shape", "pitch_mm")},
    "angular_shape": ("scorer", "angular_shape"),
    "reconstruction": ("scorer", "reconstruction"),
    "response": ("response", "kind"),
    "timepix_thickness_um": ("response", "thickness_um"),
    "timepix_bias_v": ("response", "bias_v"),
    "timepix_input_bin_ev": ("response", "dE_mc"),
    "timepix_output_bin_ev": ("response", "dE_out"),
    "timepix_samples": ("response", "n_mc"),
    "timepix_seed": ("response", "seed"),
    "exposure_s": ("acquisition", "exposure_s"),
    "reporting_bins": ("acquisition", "reporting_bins"),
    "measured_edges_ev": ("acquisition", "measured_edges_eV"),
    "measured_range_ev": ("acquisition", "measured_min_eV", "measured_max_eV"),
    "measured_bin_width_ev": ("acquisition", "measured_bin_width_eV"),
    "hit_threshold_ev": ("acquisition", "hit_threshold_eV"),
    "realization": ("acquisition", "mode"),
    "realization_seed": ("acquisition", "seed"),
}


class FormError(ValueError):
    """Invalid editor input; ``fields`` maps field keys to messages."""

    def __init__(self, message, fields=None):
        self.fields = dict(fields or {})
        super().__init__(message)


def fields_for(kind):
    return tuple(item for item in FIELDS if item.kind == kind)


def detector_kind(row):
    return "planar" if "distance_mm" in row else "scalar"


def _text(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list | tuple):
        return " ".join(_text(item) for item in value)
    return str(value)


def form_values(row):
    """Stored detector ``row`` as editor text, keyed like :data:`FIELDS`."""
    values = {LABEL: str(row.get("label", ""))}
    for key, (section, *names) in _TOML.items():
        table = row if section is None else row.get(section, {})
        if not isinstance(table, dict) or not all(name in table for name in names):
            continue
        items = [table[name] for name in names]
        if key == "measured_edges_ev":
            values[key] = ",".join(_text(item) for item in items[0])
        else:
            values[key] = " ".join(_text(item) for item in items)
    return values


def parse_field(key, text):
    """Typed value of one field's ``text``; ``None`` when blank."""
    text = text.strip()
    if not text:
        return None
    spec = BY_KEY[key]
    raw = text.replace(",", " ").split() if spec.arity > 1 else text
    if spec.arity > 1 and len(raw) != spec.arity:
        raise FormError(f"{spec.label}: enter {spec.arity} numbers", {key: "wrong count"})
    try:
        value = spec.type.convert(raw, None, None)
    except click.BadParameter as exc:
        raise FormError(f"{spec.label}: {exc.message}", {key: exc.message}) from None
    return tuple(value) if spec.arity > 1 else value


def parse_fields(values, keys):
    """Parse every ``keys`` entry of ``values``, collecting all field errors."""
    typed, errors = {}, {}
    for key in keys:
        try:
            typed[key] = parse_field(key, values.get(key, ""))
        except FormError as exc:
            errors.update(exc.fields)
            errors[key] = str(exc)
    if errors:
        raise FormError("; ".join(errors.values()), errors)
    return typed


def _changes(kind, typed):
    scalar = {key: value for key, value in typed.items() if BY_KEY[key].kind == "scalar"}
    scalar = {key: value for key, value in scalar.items() if value is not None}
    options = {item.key: typed.get(item.key) for item in fields_for("planar")}
    try:
        pixel = pixel_changes(options) if kind == "planar" else ({}, {}, {}, {})
    except ValueError as exc:
        raise FormError(str(exc)) from None
    return scalar, pixel


def _cli_value(spec, value):
    if spec.key == "measured_edges_ev":
        return [",".join(_text(item) for item in value)]
    if spec.arity > 1:
        return [_text(item) for item in value]
    return [_text(value)]


@dataclass(frozen=True)
class DetectorEntry:
    name: str
    kind: str
    source: str
    label: str
    writable: bool


@dataclass
class StagedEdit:
    """A validated, unsaved draft plus the equivalent ``pyrite detector`` call."""

    draft: CatalogDraft
    mode: str
    name: str
    kind: str
    source: str | None = None
    label: str | None = None
    options: dict = field(default_factory=dict)

    def diff(self):
        return self.draft.diff()

    def cli_commands(self):
        """``pyrite`` argument lists that stage the same catalog change, in order.

        ``detector create clone`` takes no field options, so a modified clone is
        a clone followed by ``detector set``.
        """
        fields = [
            item
            for key, value in self.options.items()
            for item in (BY_KEY[key].flag, *_cli_value(BY_KEY[key], value))
        ]
        label = ["--label", self.label] if self.label is not None else []
        if self.mode == "create":
            return [["detector", "create", self.kind, self.name, *label, *fields]]
        commands = []
        if self.mode == "clone":
            commands.append(
                ["detector", "create", "clone", self.name, "--from", self.source, *label]
            )
            label = []
        if label or fields:
            commands.append(["detector", "set", self.name, "--yes", *label, *fields])
        return commands


class DetectorSession:
    """Detector browser/editor state over one explicitly selected catalog path.

    The caller resolves catalog precedence once (``--catalog`` >
    ``PYRITE_CATALOG`` > ``catalog.path`` > bundled with the user layer).
    """

    def __init__(self, path):
        self.path = Path(path)
        self.reload()

    def reload(self):
        self.original = read_text(self.path)
        self.document = tomlkit.parse(translate_catalog_text(self.original))
        self.sources = object_sources(self.path, "detectors")

    def rows(self):
        detectors = self.document.get("detectors", {})
        return {name: row for name, row in detectors.items() if isinstance(row, dict)}

    def entries(self):
        return [
            DetectorEntry(
                name,
                detector_kind(row),
                self.sources.get(name, "catalog"),
                str(row.get("label", "")),
                self.sources.get(name) != "bundled",
            )
            for name, row in self.rows().items()
        ]

    def form(self, name):
        return form_values(self.rows()[name])

    def _draft(self):
        document = tomlkit.parse(translate_catalog_text(self.original))
        return CatalogDraft(self.path, self.original, document)

    def stage(self, mode, name, values, *, kind="scalar", source=None, initial=None):
        """Validate an editor submission into a :class:`StagedEdit`; write nothing.

        ``mode`` is ``create`` (all non-blank ``kind`` fields), ``clone`` (copy
        ``source``, then apply fields that differ from ``initial``) or ``edit``
        (patch ``name`` with fields that differ from ``initial``).
        """
        initial = initial or {}
        name = name.strip()
        if mode == "create":
            keys = [item.key for item in fields_for(kind)]
        else:
            kind = detector_kind(self.rows()[source if mode == "clone" else name])
            keys = self._changed(kind, values, initial)
        label = values.get(LABEL, "").strip()
        label_changed = label != initial.get(LABEL, "").strip()
        if mode == "create":
            label_changed = bool(label)
        if mode == "edit" and not keys and not label_changed:
            raise FormError("no changes to save")
        typed = parse_fields(values, keys)
        scalar, pixel = _changes(kind, typed)
        draft = self._draft()
        try:
            if mode == "edit":
                require_writable_object(self.path, "detectors", name)
                target = draft.document["detectors"][name]
                if label_changed:
                    target["label"] = label
                patch_detector(target, scalar, pixel)
            else:
                create_detector(
                    draft.document,
                    name,
                    source=source if mode == "clone" else None,
                    label=label if label_changed else None,
                    scalar=scalar,
                    pixel=pixel,
                    kind=kind if mode == "create" else None,
                )
            draft.validate()
        except FormError:
            raise
        except (OSError, ValueError, ParseError) as exc:
            raise FormError(str(exc)) from None
        return StagedEdit(
            draft,
            mode,
            name,
            kind,
            source=source if mode == "clone" else None,
            label=label if label_changed else None,
            options={key: value for key, value in typed.items() if value is not None},
        )

    @staticmethod
    def _changed(kind, values, initial):
        """Changed ``kind`` fields, widened to the cross-checked groups they belong to."""
        keys = {
            item.key
            for item in fields_for(kind)
            if values.get(item.key, "").strip() != initial.get(item.key, "").strip()
        }
        cleared = sorted(key for key in keys if not values.get(key, "").strip())
        if cleared:
            labels = ", ".join(BY_KEY[key].label for key in cleared)
            raise FormError(
                f"cannot clear {labels} here; use: pyrite detector reset NAME SECTION",
                dict.fromkeys(cleared, "clearing is not supported"),
            )
        for group in _GROUPS:
            if keys & group:
                keys |= {key for key in group if values.get(key, "").strip()}
        return [item.key for item in fields_for(kind) if item.key in keys]

    def save(self, staged):
        """Atomically save ``staged``; reject stale drafts and read-only objects."""
        try:
            staged.draft.save()
        except (OSError, ValueError, ParseError) as exc:
            raise FormError(str(exc)) from None
        self.reload()
