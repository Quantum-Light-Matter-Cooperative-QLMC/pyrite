"""Read-only, invocation-local table selection for convergence experiments.

These helpers admit pinned inputs; they never generate, resample or publish
cross sections and never certify a numerical-convergence gate.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .._util.hashing import canonical_json_digest, file_sha256
from ..xsgen.elsepa.catalog import ElementElastic, joined_arrays
from ..xsgen.store import (
    KEY_SCHEMA,
    MANIFEST_SCHEMA,
    StoredTable,
    manifest_digest,
    resolve,
)


def nested_energy_grid(energies, level: int) -> tuple[float, ...]:
    """Insert log-energy midpoints at L0--L3, preserving every original node."""
    if isinstance(level, bool) or level not in range(4):
        raise ValueError("ELSEPA energy refinement level must be 0, 1, 2 or 3")
    grid = np.asarray(energies, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or grid[0] <= 0
        or not np.all(np.diff(grid) > 0)
    ):
        raise ValueError("energy grid must be finite, positive and strictly increasing")
    for _ in range(level):
        refined = np.empty(2 * grid.size - 1)
        refined[::2] = grid
        refined[1::2] = np.sqrt(grid[:-1]) * np.sqrt(grid[1:])
        if not np.all(np.diff(refined) > 0):
            raise ValueError("energy refinement exhausted floating-point resolution")
        grid = refined
    return tuple(float(e) for e in grid)


def _request(manifest: Mapping[str, Any], model: dict | None = None) -> dict:
    return dict(
        schema=KEY_SCHEMA,
        **{k: manifest[k] for k in ("code", "code_version", "quantity", "target")},
        model=manifest["model"] if model is None else model,
    )


def checked_table(table: StoredTable) -> dict[str, np.ndarray]:
    """Verify the manifest, request key and array hash before admitting a table."""
    manifest = table.manifest
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("manifest_sha256") != manifest_digest(manifest)
        or manifest.get("key") != table.key
        or canonical_json_digest(_request(manifest)) != table.key
    ):
        raise ValueError(f"invalid table manifest/request identity: {table.key}")
    return table.arrays()


def _explicit_table(path: Path) -> StoredTable:
    path = Path(path).resolve()
    if path.suffix != ".npz":
        raise ValueError("table override must name an NPZ with its JSON sidecar")
    import json

    manifest = json.loads(path.with_suffix(".json").read_text())
    table = StoredTable(manifest["key"], path, manifest, "user")
    checked_table(table)
    return table


def select_elastic_tables(entries, level: int = 0, paths=()) -> tuple[ElementElastic, ...]:
    """Select exact regenerated energy grids, preserving source and physics.

    Explicit NPZ/JSON pairs override lookup for this invocation only. Missing
    refined keys fail; native tables are never substituted for refined ones.
    The native angular mesh is held fixed and remains unqualified.
    """
    # Validate even when the input layer is empty.
    nested_energy_grid([1.0, 2.0], level)
    overrides = {}
    for path in paths:
        table = _explicit_table(path)
        if table.key in overrides:
            raise ValueError(f"duplicate table override: {table.key}")
        overrides[table.key] = table
    selected, used = [], set()
    for entry in entries:
        tables, arrays = [], []
        for baseline in entry.tables:
            original = checked_table(baseline)
            model = dict(baseline.manifest["model"])
            if level:
                model["energies_eV"] = list(nested_energy_grid(model["energies_eV"], level))
            key = canonical_json_digest(_request(baseline.manifest, model))
            table = overrides.get(key)
            if table is not None:
                used.add(key)
            elif level == 0:
                table = baseline
            else:
                table = resolve(key)
            if table is None:
                raise ValueError(f"missing regenerated ELSEPA L{level} table request {key}")
            payload = checked_table(table)
            if table.key != key:
                raise ValueError("ELSEPA table differs from the requested scientific identity")
            if not np.array_equal(payload["energy_eV"], model["energies_eV"]):
                raise ValueError("ELSEPA payload energy grid differs from its generator request")
            if not np.array_equal(payload["mu"], original["mu"]):
                raise ValueError("energy refinement must preserve the native angular grid")
            tables.append(table)
            arrays.append(payload)
        selected.append(
            ElementElastic(entry.element, entry.z, tuple(tables), joined_arrays(*arrays))
        )
    if set(overrides) != used:
        raise ValueError(
            f"unused table override; differs from requested identity: {sorted(set(overrides) - used)}"
        )
    return tuple(selected)


def select_stopping_table(baseline: StoredTable, path: Path | None = None) -> StoredTable:
    """Admit an invocation-local SBETHE regeneration on the same native grid.

    The current generator exposes no independent density control. Densifying
    the output by interpolation is unsupported and is explicitly rejected.
    """
    original = checked_table(baseline)
    if path is None:
        return baseline
    table = _explicit_table(path)
    arrays = checked_table(table)
    if table.key != baseline.key:
        raise ValueError("SBETHE table differs from the requested scientific identity")
    for name in (k for k in original if k.endswith("energy_eV")):
        if name not in arrays or not np.array_equal(arrays[name], original[name]):
            raise ValueError("SBETHE native grid refinement is unsupported by this generator")
    return table


def elastic_refinement_plan(entries, levels=(1, 2, 3)) -> dict:
    """Pin fresh-generator requests without generating or resampling any data.

    Request keys are exactly those admitted by ``select_elastic_tables``.
    Native angular grids stay fixed and unqualified; output does not attest
    that any regenerated tables or independent generator evidence exist.
    """
    levels = tuple(levels)
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(type(level) is not int or level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("need unique ELSEPA refinement levels from 1, 2, 3")
    baselines, requests = {}, {}
    for entry in entries:
        for table in entry.tables:
            original = checked_table(table)
            model = dict(table.manifest["model"])
            if not np.array_equal(original["energy_eV"], model["energies_eV"]):
                raise ValueError("native ELSEPA payload differs from its generator energy request")
            baseline = table_record(table)
            if table.key in baselines and baselines[table.key] != baseline:
                raise ValueError("conflicting payloads for one nominal ELSEPA request")
            baselines[table.key] = baseline
            for level in sorted(levels):
                refined = model | dict(
                    energies_eV=list(nested_energy_grid(model["energies_eV"], level))
                )
                request = _request(table.manifest, refined)
                key = canonical_json_digest(request)
                requests[key] = dict(
                    key=key,
                    request=request,
                    level=level,
                    baseline_key=table.key,
                    energy_nodes=len(refined["energies_eV"]),
                    angular_nodes=len(original["mu"]),
                    generated_payload_sha256=None,
                    generator_evidence_sha256=None,
                    status="requires-independent-generation",
                )
    if not baselines:
        raise ValueError("no native ELSEPA tables to refine")
    return dict(
        baselines=[baselines[k] for k in sorted(baselines)],
        requests=[requests[k] for k in sorted(requests)],
        unsupported=["native angular/inverse-CDF mesh bound"],
        numerical_gate="unmeasured",
        generation_performed=False,
    )


def table_coverage(energies_eV, cutoff_keV: float, energy_keV: float) -> dict:
    """Record actual nodes and refuse interpolation beyond their support."""
    grid = nested_energy_grid(energies_eV, 0)
    if not (
        np.isfinite(cutoff_keV)
        and np.isfinite(energy_keV)
        and 0 < cutoff_keV < energy_keV
        and grid[0] <= cutoff_keV * 1e3
        and grid[-1] >= energy_keV * 1e3
    ):
        raise ValueError("requested electron range exceeds table coverage")
    return dict(nodes=len(grid), energy_eV=list(grid))


def table_record(table: StoredTable) -> dict:
    """Hash the admitted payload and retain its source/request/grid provenance."""
    arrays = checked_table(table)
    grids = {
        name: np.asarray(values).tolist()
        for name, values in arrays.items()
        if name
        in {"energy_eV", "stopping_energy_eV", "integrated_energy_eV", "mu", "t1_MeV", "k_over_t1"}
    }
    return dict(
        key=table.key,
        manifest_sha256=table.digest,
        payload_sha256=file_sha256(table.path),
        arrays_sha256=table.manifest["arrays_sha256"],
        request=_request(table.manifest),
        grids=grids,
        shapes={name: list(values.shape) for name, values in arrays.items()},
    )


def benchmark_table_audit(entries, stopping, brem, cutoff_keV: float, energy_keV: float) -> dict:
    """Verify coverage of the effective joined tables, recording unresolved meshes."""
    if not entries or {entry.element for entry in entries} - brem.keys():
        raise ValueError("missing elemental source tables for benchmark coverage audit")
    elastic = []
    for entry in entries:
        elastic.append(
            dict(
                element=entry.element,
                coverage=table_coverage(entry.arrays["energy_eV"], cutoff_keV, energy_keV),
                tables=[table_record(table) for table in entry.tables],
            )
        )
    stopping_arrays = checked_table(stopping)
    stopping_record = table_record(stopping)
    stopping_record["coverage"] = table_coverage(
        stopping_arrays["stopping_energy_eV"], cutoff_keV, energy_keV
    )
    radiative = []
    for element, prepared in brem.items():
        source = resolve(prepared.key)
        if source is None or source.digest != prepared.digest:
            raise ValueError(f"BremsLib source identity unavailable or changed for {element}")
        record = table_record(source)
        record.update(
            element=element,
            coverage=table_coverage(prepared.incident_energy_keV * 1e3, cutoff_keV, energy_keV),
        )
        radiative.append(record)
    return dict(
        elsepa=elastic,
        sbethe=stopping_record,
        bremslib=radiative,
        unresolved={
            "sbethe_energy_grid": "unsupported: current generator has no independent density control",
            "elsepa_angular_grid": "unresolved: native angular/inverse-CDF grid held fixed",
            "bremslib_source_mesh": "unresolved: staged refinement adds no independent source information",
            "moment_tolerance_ladder": "unmeasured: --moment-level 0..3 available; no numerical gate",
        },
    )
