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
  >= 10 keV cone yield; ratios are equal-precision speed ratios.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(load(args.evidence))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    for scene, row in summary.items():
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


if __name__ == "__main__":
    main()
