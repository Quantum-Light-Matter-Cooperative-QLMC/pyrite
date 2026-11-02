"""Bound candidate mixture moment changes over every pinned ELSEPA table.

Run: uv run python checks/elsepa_interpolation_moments.py --output <path.json>
Reads installed tables only; never generates/fetches them or samples tracks.
Validation: elsepa-energy-mixture-moments
"""

import argparse
import json
import time
from pathlib import Path

from pyrite.validation._elsepa_interpolation import _pair_moment_bounds
from pyrite.xsgen.elsepa.catalog import elemental_solid, joined_arrays
from pyrite.xsgen.elsepa.release import load_release_index
from pyrite.xsgen.store import require


def analyze(subdivisions):
    started = time.perf_counter()
    result = {"subdivisions": subdivisions, "projectiles": {}}
    for species in ("electron", "positron"):
        release = load_release_index(projectile=species)
        if release is None:
            raise ValueError(f"missing {species} release index")
        records = []
        muffins = []
        for entry in release.tables:
            table = require(entry.key)
            if table.digest != entry.manifest_sha256:
                raise ValueError(f"{species} {entry.label}: release manifest digest differs")
            arrays = table.arrays()
            if table.manifest["model"]["mode"] == "muffin_tin":
                muffins.append((entry, table))
            energy = arrays["energy_eV"]
            pairs = []
            for row in range(len(energy) - 1):
                bound = _pair_moment_bounds(
                    arrays["mu"],
                    arrays["dcs_cm2_sr"][row],
                    arrays["dcs_cm2_sr"][row + 1],
                    subdivisions=subdivisions,
                )
                bound["energy_interval_eV"] = energy[row : row + 2].tolist()
                pairs.append(bound)
            absolute = max(pairs, key=lambda b: b["max_second_moment_difference_upper"])
            relative = max(pairs, key=lambda b: b["max_relative_second_moment_difference_upper"])
            records.append(
                {
                    "label": entry.label,
                    "key": entry.key,
                    "manifest_sha256": table.digest,
                    "intervals": len(pairs),
                    "max_absolute": absolute,
                    "max_relative": relative,
                }
            )
        joins = []
        free_entries = {entry.label: entry for entry in release.tables}
        for entry, muffin in muffins:
            solid = elemental_solid(entry.label)
            if solid is None:
                raise ValueError(
                    f"released muffin-tin label is not an elementary solid: {entry.label}"
                )
            free_entry = free_entries[f"Z={solid.z}"]
            free = require(free_entry.key)
            arrays = joined_arrays(free.arrays(), muffin.arrays())
            energy = arrays["energy_eV"]
            row = int((energy <= 1e6).sum()) - 1
            bound = _pair_moment_bounds(
                arrays["mu"],
                arrays["dcs_cm2_sr"][row],
                arrays["dcs_cm2_sr"][row + 1],
                subdivisions=subdivisions,
            )
            bound["energy_interval_eV"] = energy[row : row + 2].tolist()
            joins.append(
                {
                    "label": entry.label + " (joined crossover)",
                    "source_keys": [entry.key, free_entry.key],
                    "max_absolute": bound,
                    "max_relative": bound,
                }
            )
        candidates = records + joins
        result["projectiles"][species] = {
            "release_archive_sha256": release.archive_sha256,
            "tables": len(records),
            "intervals": sum(r["intervals"] for r in records),
            "crossover_intervals": len(joins),
            "first_moment_difference": 0.0,
            "max_absolute": max(
                candidates, key=lambda r: r["max_absolute"]["max_second_moment_difference_upper"]
            ),
            "max_relative": max(
                candidates,
                key=lambda r: r["max_relative"]["max_relative_second_moment_difference_upper"],
            ),
            "records": records,
            "crossovers": joins,
        }
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.subdivisions < 1:
        parser.error("--subdivisions must be positive")
    result = analyze(args.subdivisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for species, data in result["projectiles"].items():
        print(
            species,
            data["tables"],
            "tables;",
            data["intervals"],
            "intervals; max gap",
            data["max_absolute"]["max_absolute"]["max_second_moment_difference_upper"],
            "; max relative gap",
            data["max_relative"]["max_relative"]["max_relative_second_moment_difference_upper"],
        )
    print("elapsed", result["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
