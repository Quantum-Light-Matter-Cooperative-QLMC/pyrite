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
listed as such and give no ratio. The frozen ``scenes.json``
``precision.equivalence`` gate (#1041) marks a ratio ``headline`` only for an
equivalent (scene, set, PyRITE arm) at its evaluation target; every other
ratio is a qualified outcome.

Beside VOV, every ``T(eps)`` row carries the #1339 diagnostics of each headline
observable, pooled over its seeds: SHIFT/SE (where skewness misplaces the
symmetric interval) and the Hill tail slope (below 3 the variance may be
infinite; reported from 50 000 scores). FOM = 1 / (RSE^2 T) is reported per
row, with its stability between an arm's two planned history counts and a
FOM-ratio speed estimate that never becomes a headline. A plan with
``protocol_version >= 3`` also applies the frozen ``amendment_v3`` rules of
``scenes.json``: a row at the history floor gives no headline, a cone
speedup needs a pooled continuous-arm tail slope of at least 3, and a
skew-flagged TOST cell must also pass with its SHIFT-corrected interval.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyrite.montecarlo.runner.adaptive import TAIL_MIN_SAMPLES, RunningMoments

HERE = Path(__file__).resolve().parent
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


def vov(values):
    """MCNP/Geant4 variance of the variance, ``M4/M2**2 - 1/n`` (central sums).

    The same quantity as ``G4ConvergenceTester``'s VOV and PyRITE's adaptive
    ``variance_of_variance`` (whose gate is ``VOV + 1/n``).
    """
    x = np.asarray(values, dtype=float)
    d = x - x.mean()
    m2 = float(np.dot(d, d))
    return float(np.sum(d**4) / m2**2 - 1.0 / x.size) if m2 > 0 else math.inf


def vov_slope(values, points=16):
    """Least-squares slope of ln VOV against ln n over second-half prefixes.

    About -1 when the fourth moment is sampled (MCNP and Geant4 test VOV
    falling as 1/N); shallower or positive when a heavy score arrives late.
    ``None`` with fewer than three prefixes of positive VOV.
    """
    x = np.asarray(values, dtype=float)
    pts = [
        (n, vov(x[:n]))
        for n in sorted({x.size * k // points for k in range(points // 2, points + 1)})
        if n > 1
    ]
    pts = [(n, v) for n, v in pts if 0 < v < math.inf]
    if len(pts) < 3:
        return None
    return float(np.polyfit(np.log([n for n, _ in pts]), np.log([v for _, v in pts]), 1)[0])


def _values(rec, name):
    scores = rec["scores"]
    return sum(scores[k] for k in GE10) if name == "photons_ge10keV" else scores[name]


def shape(arrays):
    """#1339 SHIFT/SE and tail slope of the concatenated per-primary scores."""
    moments = RunningMoments()
    for values in arrays:
        moments.add_block(np.asarray(values, dtype=float))
    rel_se, shift = moments.relative_se(), moments.shift()
    return dict(
        n=moments.n,
        shift=shift,
        shift_over_se=shift / rel_se if shift is not None and rel_se else None,
        tail_slope=moments.tail_slope(),
    )


def fom_stability(rows, tolerance):
    """FOM at an arm's largest planned count over its smallest, per (scene, config, set).

    FOM is constant when RSE falls as 1/sqrt(N) and fixed costs are negligible;
    a ratio above ``tolerance`` marks fixed-cost-dominated small runs, below
    ``1/tolerance`` a variance that grows faster than 1/N (a late heavy score).
    """
    by_arm = defaultdict(list)
    for r in rows:
        if r.get("fom_median"):
            by_arm[(r["scene"], r["config"], r["set"])].append(r)
    out = {}
    for key, rs in by_arm.items():
        lo = min(rs, key=lambda r: r["histories"])
        hi = max(rs, key=lambda r: r["histories"])
        if lo["histories"] == hi["histories"]:
            out[key] = dict(status="single-N", histories=[lo["histories"]])
            continue
        q = hi["fom_median"] / lo["fom_median"]
        out[key] = dict(
            status="stable" if 1 / tolerance <= q <= tolerance else "unstable",
            histories=[lo["histories"], hi["histories"]],
            ratio=q,
        )
    return out


def tail_check(samples, tail_min):
    """Pooled cone tail slope per (scene, config) over runs of >= TAIL_MIN_SAMPLES."""
    pooled = defaultdict(list)
    for rec in samples:
        if CONE in rec["scores"] and rec["histories"] >= TAIL_MIN_SAMPLES:
            pooled[(rec["scene"]["id"], config_name(rec))].append(rec)
    out = {}
    for key, recs in sorted(pooled.items()):
        d = shape([r["scores"][CONE] for r in recs])
        slope = d["tail_slope"]
        d.update(
            runs=len(recs),
            status="insufficient" if slope is None else "ok" if slope >= tail_min else "heavy",
        )
        out[key] = d
    return out


def time_to_precision(samples, plan, min_histories=None, fom_tolerance=1.15):
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
        achieved, vovs = [], []
        names = item["per_observable_histories"]
        for r in recs:
            scores = dict(r["scores"])
            if GE10[0] in scores:
                scores["photons_ge10keV"] = sum(scores[k] for k in GE10)
            achieved.append({name: rse(scores[name]) for name in names})
            vovs.append(
                {name: dict(vov=vov(scores[name]), slope=vov_slope(scores[name])) for name in names}
            )
        met = [all(v <= item["target_rse"] for v in a.values()) for a in achieved]
        warm = [r["timings_s"]["warm_request"] for r in recs]
        foms = [1 / (max(a.values()) ** 2 * w) for a, w in zip(achieved, warm, strict=True)]
        row.update(
            fom_per_seed=foms,
            fom_median=statistics.median(foms),
            floor_bound=item["histories"] == min_histories,
            shape={name: shape([_values(r, name) for r in recs]) for name in names},
            status="met" if all(met) else "precision-unmet",
            seeds=sorted(r["seed"] for r in recs),
            met_per_seed=met,
            worst_rse=max(max(a.values()) for a in achieved),
            rse_per_seed=achieved,
            vov_per_seed=vovs,
            worst_vov=max(v["vov"] for a in vovs for v in a.values()),
            T_s=warm,
            T_s_median=statistics.median(warm),
        )
        if recs[0]["engine"] != "geant4":
            row["transport_s_median"] = statistics.median(
                r["timings_s"]["warm_transport"] for r in recs
            )
            row["scoring_s_median"] = median(r["work"]["warm"].get("scoring_s") for r in recs)
        rows.append(row)
    stability = fom_stability(rows, fom_tolerance)
    largest = {}
    for r in rows:
        key = (r["scene"], r["set"], r["config"])
        if r.get("fom_median") and r["histories"] > largest.get(key, {"histories": 0})["histories"]:
            largest[key] = r
    fom_ratios = []
    for (scene, name_set, config), py in sorted(largest.items()):
        if not config.startswith("pyrite"):
            continue
        for (s2, set2, g4name), g4 in sorted(largest.items()):
            if (s2, set2) != (scene, name_set) or not g4name.startswith("geant4"):
                continue
            arms = [stability[(scene, c, name_set)]["status"] for c in (config, g4name)]
            fom_ratios.append(
                dict(
                    scene=scene,
                    set=name_set,
                    pyrite=config,
                    geant4=g4name,
                    histories=[py["histories"], g4["histories"]],
                    fom_ratio=py["fom_median"] / g4["fom_median"],
                    stability=arms,
                    status="stable" if arms == ["stable", "stable"] else "unverified",
                )
            )
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
                    floor_bound=py["floor_bound"] or g4["floor_bound"],
                )
            )
    return dict(
        runs=rows,
        speedup=ratios,
        fom_stability=[
            dict(scene=k[0], config=k[1], set=k[2], **v) for k, v in sorted(stability.items())
        ],
        speedup_fom=fom_ratios,
    )


def _pooled(recs, name):
    values = []
    for r in recs:
        scores = r["scores"]
        values.append(sum(scores[k] for k in GE10) if name == "photons_ge10keV" else scores[name])
    return mean_se(np.concatenate(values))


def holm(pvalues, alpha):
    """Holm step-down rejections (same order as ``pvalues``)."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    reject = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - rank):
            break
        reject[i] = True
    return reject


def equivalence(samples, plan, spec, ttp, amendment=None):
    """Frozen #1041 TOST gate per (scene, set, PyRITE arm) at the evaluation target.

    Each arm pools its held-out runs at the planned count. An observable passes
    when the ``ci_level`` delta-method CI of ln(PyRITE/Geant4) lies inside
    [ln(1-margin), ln(1+margin)]; a set is equivalent only if every headline
    observable passes and both arms met the target RSE on every seed
    (intersection-union). Discrepancy flags are Holm-adjusted two-sided tests of
    ratio = 1 across all evaluated cells.

    Every cell reports each arm's pooled SHIFT/SE. Under ``amendment`` (v3) a
    cell whose larger ``|SHIFT/SE|`` exceeds ``skew_shift_over_se`` passes only
    if the interval recentred by each arm's SHIFT also lies inside the margin,
    a speedup at the history floor is never headline, and a cone speedup is
    headline only with an ``ok`` pooled tail slope.
    """
    from statistics import NormalDist

    eps, margins = spec["evaluate_at_target_rse"], spec["margins_rel"]
    zq = NormalDist().inv_cdf(0.5 + spec["ci_level"] / 2)
    runs = defaultdict(list)
    for rec in samples:
        runs[(rec["scene"]["id"], config_name(rec), rec["histories"])].append(rec)
    status = {
        (r["scene"], r["config"], r["set"], r["target_rse"]): r["status"] for r in ttp["runs"]
    }
    items = [i for i in plan["items"] if i["target_rse"] == eps]
    cells, groups = [], []
    for py in items:
        if not py["config"].startswith("pyrite"):
            continue
        refs = [
            i
            for i in items
            if i["config"].startswith(spec["reference"])
            and (i["scene"], i["set"]) == (py["scene"], py["set"])
        ]
        py_recs = runs.get((py["scene"], py["config"], py["histories"]), [])
        ref_recs = [
            r for i in refs for r in runs.get((i["scene"], i["config"], i["histories"]), [])
        ]
        met = status.get((py["scene"], py["config"], py["set"], eps)) == "met" and any(
            status.get((i["scene"], i["config"], i["set"], eps)) == "met" for i in refs
        )
        group = dict(scene=py["scene"], set=py["set"], pyrite=py["config"], precision_met=met)
        groups.append(group)
        if not refs or any(i["status"] == "unplannable" for i in refs):
            group["status"] = "reference-unplannable"
            continue
        if not py_recs or not ref_recs:
            group["status"] = "missing"
            continue
        group["cells"] = []
        for name in py["per_observable_histories"]:
            pm, ps = _pooled(py_recs, name)
            gm, gs = _pooled(ref_recs, name)
            if pm <= 0 or gm <= 0:
                cell = dict(observable=name, status="undefined")
            else:
                lr, se = math.log(pm / gm), math.hypot(ps / pm, gs / gm)
                lo, hi = math.exp(lr - zq * se), math.exp(lr + zq * se)
                m = margins[name]
                inside = 1 - m <= lo and hi <= 1 + m
                py_shape = shape([_values(r, name) for r in py_recs])
                g4_shape = shape([_values(r, name) for r in ref_recs])
                sose = [abs(d["shift_over_se"] or 0.0) for d in (py_shape, g4_shape)]
                cell = dict(
                    observable=name,
                    ratio=pm / gm,
                    ci=[lo, hi],
                    margin=m,
                    shift_over_se=dict(
                        pyrite=py_shape["shift_over_se"], geant4=g4_shape["shift_over_se"]
                    ),
                    p_equal=2 * (1 - NormalDist().cdf(abs(lr) / se)) if se > 0 else 0.0,
                )
                if amendment and max(sose) > amendment["skew_shift_over_se"]:
                    lr2 = (
                        lr
                        + math.log1p(py_shape["shift"] or 0.0)
                        - math.log1p(g4_shape["shift"] or 0.0)
                    )
                    lo2, hi2 = math.exp(lr2 - zq * se), math.exp(lr2 + zq * se)
                    cell.update(skew_flag=True, ci_shifted=[lo2, hi2])
                    inside = inside and 1 - m <= lo2 and hi2 <= 1 + m
                cell["status"] = "pass" if inside else "fail"
            group["cells"].append(cell)
            cells.append(cell)
        group["status"] = (
            "equivalent"
            if met and all(c["status"] == "pass" for c in group["cells"])
            else "not-equivalent"
        )
    tested = [c for c in cells if "p_equal" in c]
    for c, flag in zip(tested, holm([c["p_equal"] for c in tested], 0.05), strict=True):
        c["discrepancy_holm"] = flag
    for r in ttp["speedup"]:
        g = next(
            (
                g
                for g in groups
                if (g["scene"], g["set"], g["pyrite"]) == (r["scene"], r["set"], r["pyrite"])
            ),
            None,
        )
        headline = r["target_rse"] == eps and g and g["status"] == "equivalent"
        if amendment:
            tail = ttp.get("tail", {}).get(f"{r['scene']}|{r['pyrite']}", {}).get("status")
            r["tail"] = tail if r["set"] == "cone" else None
            headline = headline and not r["floor_bound"] and (r["set"] != "cone" or tail == "ok")
        r["headline"] = bool(headline)
    return dict(target_rse=eps, ci_level=spec["ci_level"], groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--plan", type=Path, help="precision_plan.py plan: report T(eps)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = load(args.evidence)
    summary = summarize(samples)
    if args.plan:
        plan = json.loads(args.plan.read_text())
        precision = json.loads((HERE / "scenes.json").read_text())["precision"]
        spec = precision.get("equivalence")
        amendment = (
            spec.get("amendment_v3") if spec and plan.get("protocol_version", 2) >= 3 else None
        )
        ttp = time_to_precision(
            samples,
            plan,
            min_histories=precision["min_histories"],
            fom_tolerance=(amendment or {}).get("fom_stability_factor", 1.15),
        )
        ttp["tail"] = {
            f"{s}|{c}": d
            for (s, c), d in tail_check(
                samples, (amendment or {}).get("tail_slope_min", 3.0)
            ).items()
        }
        ttp["protocol_version"] = plan.get("protocol_version", 2)
        if spec:
            ttp["equivalence"] = equivalence(samples, plan, spec, ttp, amendment)
        summary["time_to_precision"] = ttp
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
            if "fom_median" in r:
                worst = {k: (d["shift_over_se"], d["tail_slope"]) for k, d in r["shape"].items()}
                extra += f" FOM {r['fom_median']:.4g}" + (" floor" if r["floor_bound"] else "")
                extra += " SHIFT/SE,slope " + json.dumps(worst)
            print(
                f"  {r['scene']} {r['set']} eps={r['target_rse']:g} {r['config']} "
                f"N={r['histories']}: {r['status']}{extra}"
            )
        print("\n== FOM stability (largest / smallest planned N)")
        for f in ttp["fom_stability"]:
            print(
                f"  {f['scene']} {f['set']} {f['config']} N={f['histories']}: {f['status']}"
                + (f" {f['ratio']:.3f}" if "ratio" in f else "")
            )
        print("\n== cone tail slope (pooled runs of >= 50 000 histories)")
        for key, d in ttp["tail"].items():
            print(f"  {key} runs={d['runs']} n={d['n']}: slope {d['tail_slope']} {d['status']}")
        for g in ttp.get("equivalence", {}).get("groups", []):
            print(f"  equivalence {g['scene']} {g['set']} {g['pyrite']}: {g['status']}")
            for c in g.get("cells", []):
                if "ci" in c:
                    print(
                        f"    {c['observable']:28s} ratio {c['ratio']:.4f} CI [{c['ci'][0]:.4f}, "
                        f"{c['ci'][1]:.4f}] margin {c['margin']:g} {c['status']}"
                        + (" DISCREPANT (Holm)" if c.get("discrepancy_holm") else "")
                    )
        for r in ttp["speedup"]:
            label = "headline" if r.get("headline") else "qualified"
            notes = (" floor" if r["floor_bound"] else "") + (
                f" tail={r['tail']}" if r.get("tail") else ""
            )
            print(
                f"  speedup[{label}] {r['scene']} {r['set']} eps={r['target_rse']:g} "
                f"{r['pyrite']} vs {r['geant4']}: {r['speedup_median']:.3g} {r['speedup_range']}"
                + notes
            )
        for r in ttp["speedup_fom"]:
            print(
                f"  fom-ratio[{r['status']}] {r['scene']} {r['set']} {r['pyrite']} vs "
                f"{r['geant4']} N={r['histories']}: {r['fom_ratio']:.3g} {r['stability']}"
            )


if __name__ == "__main__":
    main()
