"""Notebook authoring transactions over the existing catalog document.

No simulation, alternate schema, global catalog selection, or implicit save.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tomlkit

from .._catalog_layout import require_writable_object, selected_catalog
from .._cxr_vocabulary import translate_catalog_text
from ..catalog_edit import CatalogDraft
from ..detectors.editing import create_detector
from ._profile_document import existing_profile
from .config_presets import effective_profile
from .profile_cxr_grid import _check_cxr_grid_policy, profile_cxr_grid_values
from .profile_detectors import (
    migrate_profile_detectors,
    profile_detector_entries,
    profile_detector_slot,
)
from .profile_lifecycle import create_profile
from .profile_numerics import profile_resolution
from .profile_precision import precision_blockers, precision_resolution


def plain(value: Any) -> Any:
    """JSON-safe inspection values; never used as an identity serializer."""
    if hasattr(value, "unwrap"):
        return plain(value.unwrap())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def new_name(rows: Mapping, name: str) -> None:
    """Use directory-catalog naming rules even for a single-file catalog."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("name: start with a letter/digit; use letters, digits, '.', '_', '-'")
    if name.casefold() in {key.casefold() for key in rows}:
        raise ValueError(f"name: {name!r} already exists; choose a new name")


@dataclass
class ConfigurationEditor:
    """One selected profile and its unsaved CatalogDraft, suitable for UI state."""

    draft: CatalogDraft
    profile: str

    @classmethod
    def open(cls, path: Path | str | None = None, profile: str = "standard") -> ConfigurationEditor:
        draft = CatalogDraft.open(selected_catalog() if path is None else path)
        existing_profile(draft.document, profile)
        return cls(draft, profile)

    @property
    def row(self) -> Any:
        return existing_profile(self.draft.document, self.profile)

    def select(self, name: str) -> ConfigurationEditor:
        """Change selection without discarding other unsaved objects."""
        result = deepcopy(self)
        existing_profile(result.draft.document, name)
        result.profile = name
        return result

    def create(self, name: str, source: str | None = None) -> ConfigurationEditor:
        """Reuse canonical new/clone semantics, including material override policy."""
        result = deepcopy(self)
        new_name(result.draft.document["profiles"], name)
        create_profile(
            result.draft.document,
            name,
            source,
            updates={},
            beam_name=None,
            detector_name=None,
            transport_updates={},
            materials=None,
        )
        result.profile = name
        return result

    def patch(self, values: Mapping[str, Any]) -> ConfigurationEditor:
        """Stage explicit profile changes; None removes an override to restore inheritance."""
        result = deepcopy(self)
        require_writable_object(result.draft.path, "profiles", result.profile)
        for key, value in values.items():
            if value is None:
                result.row.pop(key, None)
            else:
                result.row[key] = deepcopy(value)
        return result

    def beam(self, name: str, values: Mapping[str, Any]) -> ConfigurationEditor:
        """Create a private named beam and attach it; never patch a shared beam."""
        result = deepcopy(self)
        require_writable_object(result.draft.path, "profiles", result.profile)
        document = result.draft.document
        new_name(document.get("beams", {}), name)
        if "beams" not in document:
            document["beams"] = tomlkit.table()
        document["beams"][name] = deepcopy(dict(values))
        result.row["beam"] = name
        return result

    def detector(
        self,
        identifier: str,
        *,
        values: Mapping[str, Any] | str | None = None,
        action: str = "set",
        source: str | None = None,
        migrate: bool = False,
    ) -> ConfigurationEditor:
        """Edit profile-local slots; named edits become explicit inline copies.

        Adding/duplicating legacy slots requires explicit migration. A final
        slot cannot be removed: that would silently restore a default detector.
        """
        result = deepcopy(self)
        require_writable_object(result.draft.path, "profiles", result.profile)
        row = result.row
        if action in {"set", "add"} and not values:
            raise ValueError("detectors: select a named detector or specify explicit geometry")
        if migrate and "detectors" not in row:
            migrate_profile_detectors(result.draft.document, result.profile)
        if action == "migrate":
            if "detectors" not in row:
                migrate_profile_detectors(result.draft.document, result.profile)
            return result
        entries = profile_detector_entries(row)
        if action in {"add", "duplicate"}:
            new_name(entries, identifier)
            if "detectors" not in row:
                raise ValueError("detectors: explicitly convert legacy slots to a collection first")
            if action == "duplicate":
                if source not in entries:
                    raise ValueError(f"detectors: unknown source ID {source!r}")
                values = deepcopy(entries[source])
            row["detectors"][identifier] = deepcopy(values)
        elif action == "remove":
            if len(entries) <= 1:
                raise ValueError("detectors: keep at least one explicit detector")
            table, key = profile_detector_slot(row, identifier)
            del table[key]
        elif action == "set":
            table, key = profile_detector_slot(row, identifier)
            if key == "detector" and isinstance(values, Mapping) and "distance_mm" in values:
                raise ValueError(
                    "detectors: convert legacy slots to a collection before inline pixel edits"
                )
            table[key] = deepcopy(values)
        else:
            raise ValueError(f"unknown detector action: {action}")
        return result

    def named_detector(self, name: str, values: Mapping[str, Any]) -> ConfigurationEditor:
        """Create a reusable detector through the shared domain edit service."""
        result = deepcopy(self)
        scalar = {key: value for key, value in values.items() if key not in {"label"}}
        if "distance_mm" in scalar:
            sections = [scalar.pop(key, {}) for key in ("scorer", "response", "acquisition")]
            create_detector(result.draft.document, name, pixel=(scalar, *sections))
        else:
            create_detector(result.draft.document, name, scalar=scalar)
        return result

    def validate(self) -> Any:
        """Validate selected profile and every changed profile before any persistence."""
        catalog = self.draft.validate(profile=self.profile)
        self._validate_combinations(catalog, self.profile)
        original = CatalogDraft(
            self.draft.path,
            self.draft.original,
            tomlkit.parse(translate_catalog_text(self.draft.original)),
        )
        for table in ("profiles", "beams", "detectors"):
            previous = original.document.get(table, {})
            for name, row in self.draft.document.get(table, {}).items():
                if plain(previous.get(name)) != plain(row):
                    require_writable_object(self.draft.path, table, name)
                    if table == "profiles" and name != self.profile:
                        changed_catalog = self.draft.validate(profile=name)
                        self._validate_combinations(changed_catalog, name)
        return catalog

    def _validate_combinations(self, catalog: Any, name: str) -> None:
        """Apply existing cross-setting admission checks to effective preset values."""
        from .._config_presets import physics_from_values
        from .._numerics import Numerics

        row = effective_profile(self.draft.document, existing_profile(self.draft.document, name))
        physics_from_values(row).validate_transport(
            row.get("energy_model", Numerics().energy_model)
        )
        _check_cxr_grid_policy(row, profile_cxr_grid_values(row))
        if catalog.profile_precision(name) is None:
            return
        blockers = precision_blockers(row)
        if any("distance_mm" in entry for entry in catalog.profile_detector_set(name).values()):
            blockers.append("pixel observation directions require fixed trial counts")
        beam = catalog.profile_beam(name)
        if beam is not None and beam.get("source") == "gpt_gdf":
            blockers.append("GDF sources require fixed trial counts")
        if blockers:
            raise ValueError(f"profiles.{name}.precision: {'; '.join(blockers)}")

    def errors(self) -> tuple[str, ...]:
        try:
            self.validate()
        except (ValueError, TypeError, OSError) as exc:
            return tuple(getattr(exc, "errors", (str(exc),)))
        return ()

    def save(self) -> ConfigurationEditor:
        result = deepcopy(self)
        result.validate()
        result.draft.save()
        result.draft = CatalogDraft.open(result.draft.path)
        return result

    def resolved(self) -> dict[str, Any]:
        """Inspect effective catalog values, with each member's resolved scan geometry."""
        catalog = self.validate()
        members = catalog.profile_materials(self.profile) or catalog.material_keys
        effective = effective_profile(self.draft.document, self.row)
        return plain(
            {
                "profile": self.profile,
                "beam": catalog.profile_beam(self.profile),
                "detectors": catalog.profile_detector_set(self.profile),
                "emission": catalog.profile_emission(self.profile) or "incoherent",
                "numerics": profile_resolution(self.draft.document, self.row),
                "sampling_profile_policy": dict(
                    zip(
                        ("explicit", "effective", "source", "fixed_reason"),
                        precision_resolution(effective),
                        strict=True,
                    )
                ),
                "resolution_note": "Profile defaults precede per-case material/grid/model admission and sampling fallbacks.",
                "cxr_grid_policy": catalog.profile_cxr_grid_policy(self.profile),
                "materials": {name: catalog.material(name) for name in members},
            }
        )

    def resolved_json(self) -> str:
        return json.dumps(self.resolved(), indent=2, allow_nan=False)


def run_command(
    editor: ConfigurationEditor,
    *,
    checkpoint_dir: str = "",
    trajectories: bool = False,
    trajectory_dir: str = "",
    trajectory_storage: str = "compact",
    backend: str = "inherit",
    workers: int | None = None,
) -> str:
    """Executable existing CLI options; execution-only choices never enter a profile."""
    if trajectory_storage not in {"compact", "throughput"}:
        raise ValueError("trajectory_storage: choose compact or throughput")
    if backend not in {"inherit", "auto", "cpu", "cuda", "rocm", "sycl"}:
        raise ValueError("backend: unsupported array backend")
    if workers is not None and (type(workers) is not int or workers < 0):
        raise ValueError("workers: must be a non-negative integer")
    command = ["pyrite", "--catalog", str(editor.draft.path.resolve()), "run", editor.profile]
    if checkpoint_dir:
        command += ["--checkpoint-dir", checkpoint_dir]
    if trajectories:
        if not trajectory_dir.strip():
            raise ValueError("trajectory_dir: choose a capture directory")
        command += [f"--trajectories={trajectory_dir}", "--trajectory-storage", trajectory_storage]
    if workers is not None:
        command += ["--workers", str(workers)]
    if backend != "inherit":
        command = ["env", f"PYRITE_MC_BACKEND={backend}", *command]
    return shlex.join(command)
