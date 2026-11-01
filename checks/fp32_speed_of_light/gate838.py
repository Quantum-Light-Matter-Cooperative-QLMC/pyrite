"""#838 FP32-FP64 refinement gate for ``transport_precision="mixed"``.

#510 requires the confidence bound on a numerical change to stay below one
third of the statistical-error budget. Each ``gate838.sbatch`` run pair shares
scene, estimator and seed, so the per-primary observables of the #821 harness
(``*.npz``) pair primary by primary. For each headline observable

    d_i = mixed_i - fp64_i,  bound = (|mean d| + 1.96 se(d)) / |mean fp64|

pooled over seeds, and the gate at target RSE ``eps`` is ``bound < eps / 3``.
Each test is an equivalence test with margin ``eps / 3``: ``pass`` when the
whole 95% interval of the relative change lies inside the margin, ``FAIL``
when it lies wholly outside (``|rel| - resolution > eps / 3``), and
``unresolved`` otherwise, which needs more histories, not a verdict.
The verdict passes only with every test ``pass``.
Headline membership follows ``scenes.json`` ``precision`` (sets, floors,
bands below the beam energy). Distributions: the transmitted-energy spectrum
(20 bins over [0, E0]) and the path-length distribution (20 equal-count bins
of the FP64 sample), as per-primary bin indicators. Each bin with
probability >= the fraction floor gets the same bound, and a paired Hotelling
T^2 over all populated bins tests for any shift (p reported as a diagnostic).

    uv run python checks/fp32_speed_of_light/gate838.py evidence/gate838/<job> \\
        [--output summary.json]
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE.parent / "geant4_performance" / "scenes.json").read_text())
TARGETS = (0.02, 0.01)
Z95 = 1.959963984540054
BANDS_LOWER_KEV = {"photons_10_50keV": 10, "photons_50_100keV": 50, "photons_ge100keV": 100}
FRACTIONS = ("transmitted", "backscattered", "stopped")
NAME = re.compile(r"^(fp64|mixed)-(discrete|continuous)-(.+)-(\d+)\.npz$")


def load(dirs):
    runs = defaultdict(dict)
    for d in map(Path, dirs):
        for p in d.glob("*.npz"):
            m = NAME.match(p.name)
            if m and not p.name.endswith(".grid.npz"):
                prec, est, scene, seed = m.groups()
                runs[(est, scene, int(seed))][prec] = dict(np.load(p))
    return {k: v for k, v in runs.items() if {"fp64", "mixed"} <= v.keys()}


def observables(scores):
    out = dict(scores)
    ge10 = [k for k in BANDS_LOWER_KEV if k in scores]
    if ge10:
        out["photons_ge10keV"] = sum(scores[k] for k in ge10)
    return out


def headline(name, mean, scene):
    floor = SPEC["precision"]["headline_floor"]
    if name in BANDS_LOWER_KEV and BANDS_LOWER_KEV[name] >= scene["energy_keV"]:
        return False
    if name in FRACTIONS or name == "transmitted_energy_keV":
        return mean >= floor["fraction_mean"]
    if name.startswith("photons_"):
        return mean >= floor["photons_per_primary"]
    return name.startswith("cone_")


def bound(a, b):
    d = b - a
    se = d.std(ddof=1) / math.sqrt(d.size)
    mean_a = a.mean()
    rel = d.mean() / mean_a if mean_a else math.nan
    return dict(
        resolution=float(Z95 * se / abs(mean_a)) if mean_a else math.nan,
        fp64_mean=float(mean_a),
        rel=float(rel),
        z=float(d.mean() / se) if se > 0 else 0.0,
        bound=float((abs(d.mean()) + Z95 * se) / abs(mean_a)) if mean_a else math.nan,
        identical=float(np.mean(a == b)),
    )


def outcome(r, eps):
    """Equivalence-test outcome of one relative change at margin ``eps / 3``."""
    if r["bound"] < eps / 3:
        return "pass"
    if abs(r["rel"]) - r["resolution"] > eps / 3:
        return "FAIL"
    return "unresolved"


def hotelling(ia, ib):
    """Paired T^2 over bin indicators; returns (T^2, dof, p) via the F law."""
    from scipy import stats

    d = ib - ia
    n = d.shape[0]
    m = d.mean(axis=0)
    s = np.cov(d, rowvar=False)
    if not np.any(s):
        return 0.0, 0, 1.0
    w, v = np.linalg.eigh(np.atleast_2d(s))
    keep = w > w.max() * 1e-10
    proj = v[:, keep].T @ m
    t2 = float(n * np.sum(proj**2 / w[keep]))
    k = int(keep.sum())
    f = t2 * (n - k) / (k * (n - 1))
    return t2, k, float(stats.f.sf(f, k, n - k))


def distribution(name, a, b, edges, scene_floor):
    ia = np.stack([(a >= lo) & (a < hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)], 1)
    ib = np.stack([(b >= lo) & (b < hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)], 1)
    ia, ib = ia.astype(float), ib.astype(float)
    populated = ia.mean(0) >= scene_floor
    bins = [bound(ia[:, j], ib[:, j]) for j in np.flatnonzero(populated)]
    t2, k, p = hotelling(ia[:, populated], ib[:, populated])
    rank = {"pass": 0, "unresolved": 1, "FAIL": 2}
    verdicts = {}
    for eps in TARGETS:
        worst = max((outcome(x, eps) for x in bins), key=rank.get, default="pass")
        verdicts[str(eps)] = worst
    return dict(
        pass_=verdicts,
        name=name,
        bins=len(bins),
        worst_bound=max(x["bound"] for x in bins) if bins else 0.0,
        worst_abs_z=max(abs(x["z"]) for x in bins) if bins else 0.0,
        hotelling_t2=t2,
        dof=k,
        p=p,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = load(args.dirs)
    scenes = {s["id"]: s for s in SPEC["scenes"]}
    floor = SPEC["precision"]["headline_floor"]["fraction_mean"]
    groups = defaultdict(list)
    for (est, scene, seed), pair in sorted(runs.items()):
        groups[(est, scene)].append((seed, pair))
    rows, dists = [], []
    for (est, sid), pairs in sorted(groups.items()):
        scene = scenes[sid]
        fa = [observables(p["fp64"]) for _, p in pairs]
        fb = [observables(p["mixed"]) for _, p in pairs]
        names = sorted(set(fa[0]) & set(fb[0]))
        for name in names:
            a = np.concatenate([x[name] for x in fa])
            b = np.concatenate([y[name] for y in fb])
            if not headline(name, a.mean(), scene):
                continue
            r = bound(a, b) | dict(estimator=est, scene=sid, observable=name, n=int(a.size))
            r["pass"] = {str(eps): outcome(r, eps) for eps in TARGETS}
            rows.append(r)
        if est != "discrete":
            continue
        e = np.concatenate([x["transmitted_energy_keV"] for x in fa])
        eb = np.concatenate([y["transmitted_energy_keV"] for y in fb])
        t = np.concatenate([x["transmitted"] for x in fa]).astype(bool)
        tb = np.concatenate([y["transmitted"] for y in fb]).astype(bool)
        e = np.where(t, e, -1.0)
        eb = np.where(tb, eb, -1.0)
        edges = np.linspace(0.0, scene["energy_keV"] * (1 + 1e-9), 21)
        path = np.concatenate([x["path_um"] for x in fa])
        pathb = np.concatenate([y["path_um"] for y in fb])
        pedges = np.quantile(path, np.linspace(0, 1, 21))
        pedges[-1] = np.inf
        for dist in (
            distribution("transmitted_energy", e, eb, edges, floor),
            distribution("path_um", path, pathb, pedges, floor),
        ):
            dist |= dict(scene=sid, n=int(path.size))
            dist["pass"] = dist.pop("pass_")
            dists.append(dist)
    print(
        "| est | scene | observable | n | identical | fp64 mean | rel | z | bound | eps=2% | eps=1% |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['estimator']} | {r['scene']} | {r['observable']} | {r['n']} | "
            f"{r['identical']:.3f} | {r['fp64_mean']:.4g} | {r['rel']:+.2e} | {r['z']:+.2f} | "
            f"{r['bound']:.2e} | {r['pass']['0.02']} | {r['pass']['0.01']} |"
        )
    print()
    print("| scene | distribution | bins | worst bin bound | worst |z| | T^2 p | eps=2% | eps=1% |")
    print("|---|---|---|---|---|---|---|---|")
    for d in dists:
        print(
            f"| {d['scene']} | {d['name']} | {d['bins']} | {d['worst_bound']:.2e} | "
            f"{d['worst_abs_z']:.2f} | {d['p']:.3f} | {d['pass']['0.02']} | {d['pass']['0.01']} |"
        )
    verdict = {
        str(eps): max(
            (x["pass"][str(eps)] for x in (*rows, *dists)),
            key={"pass": 0, "unresolved": 1, "FAIL": 2}.get,
            default="pass",
        )
        for eps in TARGETS
    }
    print(f"\nverdict: {verdict}")
    if args.output:
        args.output.write_text(
            json.dumps(dict(observables=rows, distributions=dists, verdict=verdict), indent=1)
        )


if __name__ == "__main__":
    main()
