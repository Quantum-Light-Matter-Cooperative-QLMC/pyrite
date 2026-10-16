"""Audited production-FP64 validation wrapper; no numerical monkey patches."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np


def grid_coverage(grid, requested_keV, unit_eV=False):
    grid = np.asarray(grid, dtype=float) / (1000 if unit_eV else 1)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("invalid table energy grid")
    low, high = requested_keV
    return dict(
        min_keV=float(grid[0]),
        max_keV=float(grid[-1]),
        nodes=len(grid),
        requested_keV=[low, high],
        covered=bool(grid[0] <= low <= high <= grid[-1]),
    )


def source_hashes():
    import pyrite.montecarlo.transport as transport

    root = Path(__file__).parent
    paths = list(Path(transport.__file__).parent.glob("*.py")) + [
        root / n
        for n in [
            "benchmark_validation.py",
            "pyrite_adapter.py",
            "precision_checks.py",
            "paired_equivalence.py",
        ]
    ]
    return {
        p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
        for p in sorted(paths)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args, adapter_args = parser.parse_known_args()
    root = Path(__file__).parent
    protocol = json.loads(args.protocol.read_bytes())
    control = protocol["arms"][args.arm]
    manifest = json.loads((root / "manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["id"] == adapter_args[0])
    from paired_equivalence import coupling_record
    from precision_checks import trajectory_checks

    import pyrite.montecarlo.transport as transport

    namespace = runpy.run_path(str(root / "pyrite_adapter.py"))
    original_score = namespace["score_segments"]
    blocks, coverage, resolved = [], {}, {}
    original_simulate = transport.simulate_trajectories

    def simulate(**kwargs):
        if not coverage:
            request = [kwargs["E_cut_keV"], kwargs["E0_keV"]]
            for i, layer in enumerate(kwargs["elastic_tables"]):
                for j, table in enumerate(layer):
                    coverage[f"elsepa-{i}-{j}"] = grid_coverage(table["energy_eV"], request, True)
            for i, table in enumerate(kwargs["stopping_tables"]):
                coverage[f"sbethe-{i}"] = grid_coverage(table["stopping_energy_eV"], request, True)
            for element, table in kwargs["bremslib_tables"].items():
                coverage[f"bremslib-{element}"] = grid_coverage(table.incident_energy_keV, request)
            for key in [
                "E_cut_keV",
                "energy_model",
                "max_dE_frac",
                "elastic_model",
                "atomic_electron_deflection",
                "straggling",
                "radiative_model",
                "radiative_cutoff_eV",
                "transport_core",
                "crystal_width_mm",
                "crystal_height_mm",
            ]:
                resolved[key] = kwargs[key]
        return original_simulate(**kwargs)

    def score(result, count):
        extra, checks = trajectory_checks(result, count, scene["energy_keV"], control["cutoff_eV"])
        blocks.append(dict(primaries=count, **checks))
        return {**original_score(result, count), **extra}

    namespace["main"].__globals__["score_segments"] = score
    transport.simulate_trajectories = simulate
    sys.argv = [
        sys.argv[0],
        *adapter_args,
        "--max-de-frac",
        str(control["max_de_frac"]),
        "--cutoff-ev",
        str(control["cutoff_eV"]),
        "--audit-cxr",
    ]
    try:
        namespace["main"]()
    finally:
        transport.simulate_trajectories = original_simulate
    output = Path(adapter_args[adapter_args.index("--output") + 1])
    record = json.loads(output.read_text())
    if sum(b["primaries"] for b in blocks) != 2 * record["histories"]:
        raise ValueError("incomplete warmup/measurement trajectory audit")
    record.update(
        schema="pyrite.equal-precision.cheap-validation-sample.v1",
        arm=args.arm,
        validation_protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        primary_coupling=coupling_record(record["seed"], record["histories"]),
        validation_sources=source_hashes(),
        table_coverage=coverage,
        resolved_controls=resolved,
        trajectory_blocks=blocks,
        timing_scope="Audit-instrumented diagnostic; no speed ratio qualification",
    )
    output.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
