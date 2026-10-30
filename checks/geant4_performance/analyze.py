"""Summarize #821 benchmark evidence: agreement, work, timing and figure of merit.

Reads every ``g4-*`` / ``py*-*`` sample JSON (with its ``.npz`` scores) in one
or more evidence directories. Configurations are ``geant4-<threads>t``,
``pyrite-cuda`` (discrete photons) and
``pyrite-cuda-continuous-<mode>-c<chunk>`` (weighted per-segment estimator,
cone observable only). Per scene it reports

* agreement of pooled per-primary means with Geant4 (difference in combined SE)
  for the discrete 4π observables and, for every PyRITE arm, the cone yield;
* work per primary (Geant4 steps by limiting process, PyRITE rows by kind);
* warm request time, throughput and cold end-to-end time (Geant4:
  initialization, warmup run and measured run; PyRITE: imports/tables and the
  first, cold request);
* figures of merit FOM = 1 / (RSE^2 T_warm) for the >= 10 keV 4π yield and the
  >= 10 keV cone yield (diagnostic: equal-history samples).

With ``--plan`` (a ``precision_plan.py`` plan, evidence from
``precision.sbatch``) it also reports the #510 headline: per (scene, set,
target) and configuration, the held-out runs at the planned history count,
whether every headline observable of the set met the target RSE in each run,
``T(eps)`` (warm request seconds, median and range over seeds), the PyRITE
transport/scoring split, and T(eps) speed ratios against each Geant4 thread
count (median and range over all seed pairs). Over-budget or unmet targets are
listed as such and give no ratio.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

AGREEMENT = (
    "transmitted",
    "backscattered",
    "stopped",
    "transmitted_energy_keV",
    "path_um",
    "photons_1_10keV",
    "photons_10_50keV",
    "photons_50_100keV",
    "photons_ge100keV",
    "photon_keV_ge10keV",
)
GE10 = ("photons_10_50keV", "photons_50_100keV", "photons_ge100keV")
CONE = "cone_source_photons_ge10kev"
G4_WORK = ("steps", "steps_coulomb", "steps_ioni", "steps_brem", "steps_other")
PY_WORK = ("rows", "rows_elastic", "rows_substep", "rows_hard_radiative")


def config_name(rec):
    if rec["engine"] == "geant4":
        return f"geant4-{rec['threads']}t"
    if rec.get("estimator") == "continuous":
        s = rec["cone_scoring"]
        return f"pyrite-cuda-continuous-{s['mode']}-c{s['chunk']}"
    return "pyrite-cuda"


def load(dirs):
    samples = []
    for d in dirs:
        for path in sorted(Path(d).glob("*.json")):
            try:
                rec = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # provenance files such as sync.json
            if rec.get("schema") != "pyrite.geant4-performance.sample.v1":
                continue
            with np.load(path.with_suffix(".npz")) as z:
                rec["scores"] = {k: z[k] for k in z.files}
            rec["path"] = str(path)
            samples.append(rec)
    return samples


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def fom(values, seconds):
    m, se = mean_se(values)
    return 1.0 / ((se / m) ** 2 * seconds) if se > 0 and m > 0 else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def ratio(a, b):
    return a / b if a is not None and b else None


def compare(g4_values, py_values):
    gm, gs = mean_se(np.concatenate(g4_values))
    pm, ps = mean_se(np.concatenate(py_values))
    combined = math.hypot(gs, ps)
    return dict(
        geant4=gm,
        pyrite=pm,
        geant4_se=gs,
        pyrite_se=ps,
        ratio=pm / gm if gm else None,
        z=(pm - gm) / combined if combined else None,
    )


def timing(recs):
    warm = [r["timings_s"]["warm_request"] for r in recs]
    if recs[0]["engine"] == "geant4":
        cold = [
            sum(r["timings_s"][k] for k in ("init", "warmup_run", "warm_request")) for r in recs
        ]
    else:
        cold = [r["timings_s"]["imports_and_tables"] + r["timings_s"]["cold_request"] for r in recs]
    out = dict(
        samples=len(recs),
        histories=sorted({r["histories"] for r in recs}),
        seeds=sorted(r["seed"] for r in recs),
        warm_s_median=statistics.median(warm),
        warm_s_range=[min(warm), max(warm)],
        cold_s_median=statistics.median(cold),
        primaries_per_s_median=statistics.median(
            r["histories"] / w for r, w in zip(recs, warm, strict=True)
        ),
        fom_4pi_median=median(
            fom(sum(r["scores"][k] for k in GE10), w) if GE10[0] in r["scores"] else None
            for r, w in zip(recs, warm, strict=True)
        ),
        fom_cone_median=median(
            fom(r["scores"][CONE], w) if CONE in r["scores"] else None
            for r, w in zip(recs, warm, strict=True)
        ),
    )
    if recs[0]["engine"] != "geant4":
        out["scoring_s_median"] = median(r["work"]["warm"].get("scoring_s") for r in recs)
        out["peak_rss_gib_max"] = max(r["peak_rss_kib"] for r in recs) / 2**20
    return out


def summarize(samples):
    by_scene = defaultdict(lambda: defaultdict(list))
    for rec in samples:
        by_scene[rec["scene"]["id"]][config_name(rec)].append(rec)
    out = {}
    for scene, configs in sorted(by_scene.items()):
        row = {"timing": {n: timing(rs) for n, rs in sorted(configs.items())}}
        g4 = [r for n, rs in configs.items() if n.startswith("geant4") for r in rs]
        py = configs.get("pyrite-cuda", [])
        row["agreement"] = {}
        if g4 and py:
            for key in AGREEMENT:
                row["agreement"][key] = compare(
                    [r["scores"][key] for r in g4], [r["scores"][key] for r in py]
                )
            row["work"] = {
                "geant4_steps_per_primary": {
                    k: float(np.mean(np.concatenate([r["scores"][k] for r in g4]))) for k in G4_WORK
                },
                "pyrite_rows_per_primary": {
                    k: float(np.mean(np.concatenate([r["scores"][k] for r in py]))) for k in PY_WORK
                },
            }
        g4_cone = [r["scores"][CONE] for r in g4 if CONE in r["scores"]]
        for name, rs in configs.items():
            if name.startswith("pyrite") and g4_cone and CONE in rs[0]["scores"]:
                row["agreement"][f"cone[{name}]"] = compare(
                    g4_cone, [r["scores"][CONE] for r in rs]
                )
        for pname, ref in row["timing"].items():
            if not pname.startswith("pyrite"):
                continue
            ref["vs_geant4"] = {
                gname: dict(
                    warm_throughput=ratio(
                        ref["primaries_per_s_median"], t["primaries_per_s_median"]
                    ),
                    fom_4pi=ratio(ref["fom_4pi_median"], t["fom_4pi_median"]),
                    fom_cone=ratio(ref["fom_cone_median"], t["fom_cone_median"]),
                )
                for gname, t in row["timing"].items()
                if gname.startswith("geant4")
            }
        out[scene] = row
    return out


def rse(values):
    m, se = mean_se(values)
    return se / abs(m) if m and se > 0 else math.inf


def time_to_precision(samples, plan):
    runs = defaultdict(list)
    for rec in samples:
        runs[(rec["scene"]["id"], config_name(rec), rec["histories"])].append(rec)
    rows = []
    for item in plan["items"]:
        row = {k: item[k] for k in ("scene", "config", "set", "target_rse", "histories", "status")}
        recs = runs.get((item["scene"], item["config"], item["histories"]), [])
        if item["status"] != "planned" or not recs:
            row["status"] = item["status"] if item["status"] != "planned" else "missing"
            rows.append(row)
            continue
        achieved = []
        for r in recs:
            scores = dict(r["scores"])
            if GE10[0] in scores:
                scores["photons_ge10keV"] = sum(scores[k] for k in GE10)
            achieved.append({name: rse(scores[name]) for name in item["per_observable_histories"]})
        met = [all(v <= item["target_rse"] for v in a.values()) for a in achieved]
        warm = [r["timings_s"]["warm_request"] for r in recs]
        row.update(
            status="met" if all(met) else "precision-unmet",
            seeds=sorted(r["seed"] for r in recs),
            met_per_seed=met,
            worst_rse=max(max(a.values()) for a in achieved),
            T_s=warm,
            T_s_median=statistics.median(warm),
        )
        if recs[0]["engine"] != "geant4":
            row["transport_s_median"] = statistics.median(
                r["timings_s"]["warm_transport"] for r in recs
            )
            row["scoring_s_median"] = median(r["work"]["warm"].get("scoring_s") for r in recs)
        rows.append(row)
    met = {
        (r["scene"], r["set"], r["target_rse"], r["config"]): r
        for r in rows
        if r["status"] == "met"
    }
    ratios = []
    for (scene, name_set, eps, config), py in sorted(met.items()):
        if not config.startswith("pyrite"):
            continue
        for (s2, set2, eps2, g4name), g4 in met.items():
            if (s2, set2, eps2) != (scene, name_set, eps) or not g4name.startswith("geant4"):
                continue
            pairs = [g / p for g in g4["T_s"] for p in py["T_s"]]
            ratios.append(
                dict(
                    scene=scene,
                    set=name_set,
                    target_rse=eps,
                    pyrite=config,
                    geant4=g4name,
                    speedup_median=statistics.median(pairs),
                    speedup_range=[min(pairs), max(pairs)],
                )
            )
    return dict(runs=rows, speedup=ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--plan", type=Path, help="precision_plan.py plan: report T(eps)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = load(args.evidence)
    summary = summarize(samples)
    if args.plan:
        summary["time_to_precision"] = time_to_precision(samples, json.loads(args.plan.read_text()))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    for scene, row in summary.items():
        if scene == "time_to_precision":
            continue
        print(f"\n== {scene}")
        for name, t in row["timing"].items():
            print(
                f"  {name:38s} n={t['samples']} warm {t['warm_s_median']:8.3f}s "
                f"cold {t['cold_s_median']:8.3f}s {t['primaries_per_s_median']:10.1f}/s "
                f"FOM4pi {t['fom_4pi_median']} FOMcone {t['fom_cone_median']}"
            )
            for g, v in t.get("vs_geant4", {}).items():
                print(f"      vs {g}: {json.dumps(v)}")
        for key, a in row["agreement"].items():
            print(
                f"  {key:52s} G4 {a['geant4']:.5g} PyRITE {a['pyrite']:.5g} "
                f"ratio {a['ratio']} z {a['z']}"
            )
        if row.get("work"):
            print("  work", json.dumps(row["work"]))
    if args.plan:
        ttp = summary["time_to_precision"]
        print("\n== T(eps)")
        for r in ttp["runs"]:
            extra = (
                f" T {r['T_s_median']:.3f}s worst RSE {r['worst_rse']:.4f}" if "T_s" in r else ""
            )
            print(
                f"  {r['scene']} {r['set']} eps={r['target_rse']:g} {r['config']} "
                f"N={r['histories']}: {r['status']}{extra}"
            )
        for r in ttp["speedup"]:
            print(
                f"  speedup {r['scene']} {r['set']} eps={r['target_rse']:g} "
                f"{r['pyrite']} vs {r['geant4']}: {r['speedup_median']:.3g} {r['speedup_range']}"
            )


if __name__ == "__main__":
    main()
