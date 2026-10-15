"""Summarize immutable adapter-pilot evidence without publishing speed ratios."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyrite.perf.precision import primary_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    failures = []
    for path in sorted(args.input.rglob("*.json")):
        if path.name.endswith(".failed.json"):
            failures.append(
                {
                    "path": str(path.relative_to(args.input)),
                    **json.loads(path.read_text(encoding="utf-8")),
                }
            )
            continue
        # A killed launcher can leave a late child-written result. Such a
        # sample is failed evidence even when its score checksum is complete.
        if path.with_suffix(".failed.json").exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "pyrite.equal-precision.sample.v1":
            continue
        if record.get("core") == "cuda":
            warmup = record.get("warmup", {})
            if (
                record.get("cache_state") != "warmed-in-process"
                or warmup.get("repeats", 0) < 1
                or warmup.get("histories") != record["histories"]
                or warmup.get("same_scene_tables_numerics") is not True
                or warmup.get("device_synchronized") is not True
            ):
                raise ValueError("CUDA sample lacks a same-workload synchronized warmup")
        raw = path.with_suffix(".npz")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record["raw_scores_sha256"]:
            raise ValueError(f"corrupt raw scores: {raw}")
        if record.get("cxr_execution_audit", {}).get("calls"):
            raise ValueError("PXR/CBS computation in benchmark")
        arm = record.get("core", record["engine"])
        if record["engine"] == "geant4" and record.get("physics", "empenelope") != "empenelope":
            arm = f"geant4-{record['physics']}"
        key = (
            record["manifest_sha256"],
            record["scene"]["id"],
            arm,
            record["histories"],
        )
        groups[key].append((path, record))
    results = []
    for (digest, scene, arm, n), records in sorted(groups.items()):
        seeds = [r["seed"] for _, r in records]
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seed in one scene/arm/population")
        values = defaultdict(list)
        observable_fields = None
        walls = []
        compute_times = []
        warmup_times = []
        native_times = []
        for path, _record in records:
            if "fresh_process_with_logging" in _record["timings_s"]:
                native_times.append(_record["timings_s"]["fresh_process_with_logging"])
            if "transport_and_emission" in _record["timings_s"]:
                compute_times.append(_record["timings_s"]["transport_and_emission"])
                warmup_times.append(_record["warmup"]["wall_s"])
            with np.load(path.with_suffix(".npz")) as scores:
                fields = set(scores.files)
                if "photons_ge10kev" not in fields:
                    raise ValueError(f"missing required observable photons_ge10kev: {path}")
                if observable_fields is None:
                    observable_fields = fields
                elif fields != observable_fields:
                    raise ValueError(
                        f"observable fields differ within scene/arm/population: {path}; "
                        f"missing={sorted(observable_fields - fields)}, "
                        f"extra={sorted(fields - observable_fields)}"
                    )
                for key in scores.files:
                    if scores[key].shape != (n,):
                        raise ValueError("score count differs from incident primary count")
                    values[key].append(scores[key])
            walls.append(
                json.loads(path.with_suffix(".wall.json").read_text(encoding="utf-8"))["wall_s"]
            )
        observables = {key: primary_moments(np.concatenate(v)) for key, v in values.items()}
        for key, scores in values.items():
            means = np.array([v.mean() for v in scores])
            observables[key]["replica_se"] = (
                float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None
            )
        results.append(
            dict(
                manifest_sha256=digest,
                scene=scene,
                arm=arm,
                histories_per_replica=n,
                replicas=len(records),
                seeds=seeds,
                wall_median_s=float(np.median(walls)),
                wall_min_s=min(walls),
                wall_max_s=max(walls),
                wall_sum_s=sum(walls),
                warmed_compute_median_s=float(np.median(compute_times)) if compute_times else None,
                warmed_compute_min_s=min(compute_times) if compute_times else None,
                warmed_compute_max_s=max(compute_times) if compute_times else None,
                warmup_median_s=float(np.median(warmup_times)) if warmup_times else None,
                native_process_median_s=float(np.median(native_times)) if native_times else None,
                native_process_min_s=min(native_times) if native_times else None,
                native_process_max_s=max(native_times) if native_times else None,
                observables=observables,
                numerical_gate="unmeasured",
                equivalence_gate="unmeasured",
                T_epsilon="unqualified: numerical and equivalence gates not established",
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "pyrite.equal-precision.summary.v1",
        "results": results,
        "failures": failures,
        "interpretation": "Adapter pilot only. No supported speed ratio or time-to-qualified-precision claim.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "| Scene | Arm | Histories × replicas | Wall median [min,max], s | Pooled source yield ≥10 keV | Pooled RSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        obs = r["observables"]["photons_ge10kev"]
        rse = f"{obs['rse'] * 100:.1f}%" if obs["rse"] is not None else "unresolved"
        lines.append(
            f"| {r['scene']} | {r['arm']} | {r['histories_per_replica']} × {r['replicas']} | {r['wall_median_s']:.3f} [{r['wall_min_s']:.3f}, {r['wall_max_s']:.3f}] | {obs['mean']:.6g} | {rse} |"
        )
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if results:
        plot(results, args.output / "precision.png")
    print(
        json.dumps(
            {
                "groups": len(results),
                "records": sum(r["replicas"] for r in results),
                "failures": len(failures),
            }
        )
    )


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = sorted({r["scene"] for r in results})
    columns = min(2, len(scenes))
    row_count = (len(scenes) + columns - 1) // columns
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(12, 3.6 * row_count),
        squeeze=False,
        layout="constrained",
    )
    for scene, ax in zip(scenes, axes.flat, strict=False):
        for arm in sorted({r["arm"] for r in results}):
            rows = sorted(
                [r for r in results if r["scene"] == scene and r["arm"] == arm],
                key=lambda r: r["histories_per_replica"],
            )
            rows = [r for r in rows if r["observables"]["photons_ge10kev"]["rse"] is not None]
            if rows:
                xs = [r["wall_sum_s"] for r in rows]
                ys = [100 * r["observables"]["photons_ge10kev"]["rse"] for r in rows]
                ax.plot(xs, ys, "o--", label=arm, alpha=0.8)
                for r, x, y in zip(rows, xs, ys, strict=True):
                    ax.annotate(
                        f"{r['histories_per_replica']}×{r['replicas']}",
                        (x, y),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=7,
                    )
        ax.set(title=scene)
        if not ax.lines:
            ax.text(
                0.5, 0.5, "All source scores zero\nPrecision unresolved", ha="center", va="center"
            )
            ax.set_axis_off()
            continue
        ax.set(
            xlabel="Total process wall time, including warmup (s)",
            ylabel="Pooled source-yield RSE (%)",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.margins(x=0.12, y=0.18)
    for ax in list(axes.flat)[len(scenes) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Observed source-yield precision; accuracy gates open\nLabels: histories per replica × replica count"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
