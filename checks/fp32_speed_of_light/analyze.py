"""Summarize a #831 ``run.sbatch`` evidence directory.

Speedup per (variant, scene) against ``exact`` at the same seed: warm kernel
time (CUDA events, ``*.stats.json``), kernel time per retained row, and the
#821 harness warm transport time. Median and [min, max] over seeds.

Deviation preview per (variant, scene): paired per-primary observables of the
#821 harness (``*.npz``) against ``exact`` at the same seed. ``identical`` is
the fraction of primaries whose value is bitwise equal (trajectory
divergence); ``rel`` the relative difference of the means; ``z`` the paired
mean difference over its standard error, pooled over seeds.

    uv run python checks/fp32_speed_of_light/analyze.py evidence/fp32sol831/<job> [...]
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

OBSERVABLES = (
    "transmitted",
    "backscattered",
    "stopped",
    "transmitted_energy_keV",
    "path_um",
    "rows",
    "photon_keV_ge10keV",
    "cone_source_photons_ge10kev",
)


def load(dirs):
    runs = {}
    for d in map(Path, dirs):
        for p in d.glob("*.stats.json"):
            stats = json.loads(p.read_text())
            bench = p.with_name(p.name.replace(".stats.json", ".json"))
            if not bench.exists():
                continue
            b = json.loads(bench.read_text())
            seed = b["seed"]
            runs[(stats["variant"], b["scene"]["id"], seed)] = dict(
                kernel_s=stats["per_seed"][str(seed)]["kernel_s"],
                launches=stats["per_seed"][str(seed)]["launches"],
                transport_s=b["timings_s"]["warm_transport"],
                rows=b["work"]["warm"]["retained_rows"],
                kernels=stats["kernels"],
                npz=bench.with_suffix(".npz"),
            )
    return runs


def fmt(values):
    return f"{statistics.median(values):.2f}× [{min(values):.2f}, {max(values):.2f}]"


def main():
    runs = load(sys.argv[1:])
    variants = sorted({v for v, _, _ in runs})
    scenes = sorted({s for _, s, _ in runs})
    print("| variant | scene | seeds | kernel | kernel/row | warm transport | regs | local B |")
    print("|---|---|---|---|---|---|---|---|")
    for v in variants:
        for s in scenes:
            seeds = sorted(
                k for vv, ss, k in runs if vv == v and ss == s and (("exact", s, k) in runs)
            )
            if not seeds:
                continue
            k, kr, t = [], [], []
            for seed in seeds:
                a, b = runs[("exact", s, seed)], runs[(v, s, seed)]
                k.append(a["kernel_s"] / b["kernel_s"])
                kr.append((a["kernel_s"] / a["rows"]) / (b["kernel_s"] / b["rows"]))
                t.append(a["transport_s"] / b["transport_s"])
            attrs = next(iter(runs[(v, s, seeds[0])]["kernels"].values()))
            base = runs[("exact", s, seeds[0])]
            print(
                f"| {v} | {s} | {len(seeds)} | {fmt(k)} | {fmt(kr)} | {fmt(t)} | "
                f"{attrs['num_regs']} | {attrs['local_size_bytes']} |"
                + (
                    f" exact kernel {statistics.median(runs[('exact', s, q)]['kernel_s'] for q in seeds):.2f} s,"
                    f" {base['rows'] / 1e6:.1f} M rows"
                    if v == "exact"
                    else ""
                )
            )
    print()
    print("| variant | scene | observable | identical | exact mean | rel | z |")
    print("|---|---|---|---|---|---|---|")
    for v in variants:
        if v == "exact":
            continue
        for s in scenes:
            seeds = sorted(
                k for vv, ss, k in runs if vv == v and ss == s and (("exact", s, k) in runs)
            )
            if not seeds:
                continue
            pairs = [
                (np.load(runs[("exact", s, q)]["npz"]), np.load(runs[(v, s, q)]["npz"]))
                for q in seeds
            ]
            for name in OBSERVABLES:
                if name not in pairs[0][0]:
                    continue
                a = np.concatenate([x[name] for x, _ in pairs])
                b = np.concatenate([y[name] for _, y in pairs])
                d = b - a
                se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else np.nan
                z = d.mean() / se if se > 0 else 0.0
                rel = d.mean() / a.mean() if a.mean() else np.nan
                print(
                    f"| {v} | {s} | {name} | {np.mean(a == b):.3f} | {a.mean():.4g} | "
                    f"{rel:+.2e} | {z:+.2f} |"
                )


if __name__ == "__main__":
    main()
