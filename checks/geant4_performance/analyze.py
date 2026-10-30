"""Summarize #821 benchmark evidence: agreement, work, timing and figure of merit.

Reads every ``g4-*.json`` / ``py-*.json`` sample (with its ``.npz`` scores) in
one or more evidence directories. Per scene it reports

* physics agreement of pooled per-primary means (difference in combined SE);
* work per primary (Geant4 steps by limiting process, PyRITE rows by kind);
* warm request time (median [min, max] over seeds) and throughput per engine
  configuration, plus cold end-to-end time (Geant4: initialization, warmup run
  and measured run; PyRITE: imports/tables and the first, cold request);
* figure of merit FOM = 1 / (RSE^2 T) of the >= 10 keV photon yield per sample,
  whose ratio is the equal-precision speed ratio independent of history count.
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


def config_name(rec):
    return f"geant4-{rec['threads']}t" if rec["engine"] == "geant4" else "pyrite-cuda"


def load(dirs):
    samples = []
    for d in dirs:
        for path in sorted(Path(d).glob("*.json")):
            if not path.name.startswith(("g4-", "py-")):
                continue
            rec = json.loads(path.read_text())
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


def summarize(samples):
    by_scene = defaultdict(lambda: defaultdict(list))
    for rec in samples:
        by_scene[rec["scene"]["id"]][config_name(rec)].append(rec)
    out = {}
    for scene, configs in by_scene.items():
        row = {"timing": {}, "agreement": {}, "work": {}}
        for name, recs in sorted(configs.items()):
            warm = [r["timings_s"]["warm_request"] for r in recs]
            if recs[0]["engine"] == "geant4":
                cold = [
                    r["timings_s"]["init"]
                    + r["timings_s"]["warmup_run"]
                    + r["timings_s"]["warm_request"]
                    for r in recs
                ]
            else:
                cold = [
                    r["timings_s"]["imports_and_tables"] + r["timings_s"]["cold_request"]
                    for r in recs
                ]
            foms, rates = [], []
            for r in recs:
                y = sum(r["scores"][k] for k in GE10)
                m, se = mean_se(y)
                foms.append(1.0 / ((se / m) ** 2 * r["timings_s"]["warm_request"]))
                rates.append(r["histories"] / r["timings_s"]["warm_request"])
            row["timing"][name] = dict(
                samples=len(recs),
                histories=sorted({r["histories"] for r in recs}),
                seeds=sorted(r["seed"] for r in recs),
                warm_s_median=statistics.median(warm),
                warm_s_range=[min(warm), max(warm)],
                cold_s_median=statistics.median(cold),
                primaries_per_s_median=statistics.median(rates),
                fom_ge10keV_median=statistics.median(foms),
            )
        # Pool all seeds of the 1-thread Geant4 reference and PyRITE for agreement.
        g4 = [r for n, rs in configs.items() if n.startswith("geant4") for r in rs]
        py = configs.get("pyrite-cuda", [])
        if g4 and py:
            for key in AGREEMENT:
                gm, gs = mean_se(np.concatenate([r["scores"][key] for r in g4]))
                pm, ps = mean_se(np.concatenate([r["scores"][key] for r in py]))
                combined = math.hypot(gs, ps)
                row["agreement"][key] = dict(
                    geant4=gm,
                    pyrite=pm,
                    geant4_se=gs,
                    pyrite_se=ps,
                    ratio=pm / gm if gm else None,
                    z=(pm - gm) / combined if combined else None,
                )
            gwork = {
                k: float(np.mean(np.concatenate([r["scores"][k] for r in g4])))
                for k in ("steps", "steps_coulomb", "steps_ioni", "steps_brem", "steps_other")
            }
            pwork = {
                k: float(np.mean(np.concatenate([r["scores"][k] for r in py])))
                for k in ("rows", "rows_elastic", "rows_substep", "rows_hard_radiative")
            }
            row["work"] = {"geant4_steps_per_primary": gwork, "pyrite_rows_per_primary": pwork}
        if "pyrite-cuda" in row["timing"]:
            ref = row["timing"]["pyrite-cuda"]
            for name, t in row["timing"].items():
                if name.startswith("geant4"):
                    t["pyrite_speedup_warm"] = (
                        ref["primaries_per_s_median"] / t["primaries_per_s_median"]
                    )
                    t["pyrite_speedup_fom"] = ref["fom_ge10keV_median"] / t["fom_ge10keV_median"]
        out[scene] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(load(args.evidence))
    text = json.dumps(summary, indent=1, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    for scene, row in summary.items():
        print(f"\n== {scene}")
        for name, t in row["timing"].items():
            extra = ""
            if "pyrite_speedup_warm" in t:
                extra = f"  PyRITE x{t['pyrite_speedup_warm']:.2f} warm, x{t['pyrite_speedup_fom']:.2f} FOM"
            print(
                f"  {name:14s} n={t['samples']} warm {t['warm_s_median']:8.3f}s "
                f"cold {t['cold_s_median']:8.3f}s  {t['primaries_per_s_median']:10.1f}/s{extra}"
            )
        for key, a in row["agreement"].items():
            print(
                f"  {key:24s} G4 {a['geant4']:.5g}  PyRITE {a['pyrite']:.5g}  "
                f"ratio {a['ratio'] if a['ratio'] is None else round(a['ratio'], 4)}  z {a['z'] if a['z'] is None else round(a['z'], 2)}"
            )
        if row["work"]:
            print("  work", json.dumps(row["work"]))


if __name__ == "__main__":
    main()
