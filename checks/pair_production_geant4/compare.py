"""Pair-conversion e-/e+ spectra: PyRITE against Geant4 TestEm5 (issue #275).

``reduce`` turns the raw TestEm5 conversion records (one line per conversion
of the primary photon, see ``patch_stepping.py``) into binned reference
histograms under ``reference/``. Only conversions at the full beam energy are
kept, so the photon has lost no energy before converting; angles are taken
about the photon's own pre-step direction. ``emstandard_opt4``'s 5D model
also emits the recoil nucleus (an ion, eV-scale kinetic energy), which is
dropped; an event with a second electron is a triplet, counted and excluded
from the pair spectra.

``compare`` (the default) samples ``pair_production.sample_pair`` at the same
photon energy and Z and compares, per case and Geant4 physics list:

* the electron's share of the pair kinetic energy, ``x = E_- / (k - 2 m c^2)``;
* the polar cosine of the electron and of the positron about the photon.

The statistic is the two-sample chi-square on the reference bins (bins with
fewer than 10 counts merged), with the Kolmogorov distance and mean
differences reported. ``empenelope`` uses PENELOPE's own pair model
(G4PenelopeGammaConversionModel), so it must agree within statistics (exit 1
otherwise). ``emstandard_opt0`` (G4BetheHeitlerModel with modified-Tsai
angles) and ``emstandard_opt4`` (G4BetheHeitler5D) are independent models.
Their differences are reported, not gated.

Validation: pair-production-sampling

Run:
  uv run python checks/pair_production_geant4/compare.py reduce RAW_DIR
  uv run python checks/pair_production_geant4/compare.py [--samples N] [--output REPORT.json]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
MC2_EV = 510998.95
#: case -> (Z, photon energy [eV]); materials as in make_macros.py.
CASES = {"c_2mev": (6, 2.0e6), "c_5mev": (6, 5.0e6), "pb_2mev": (82, 2.0e6), "pb_5mev": (82, 5.0e6)}
PHYSICS = ("empenelope", "emstandard_opt0", "emstandard_opt4")
X_EDGES = np.linspace(0.0, 1.0, 41)
COS_EDGES = np.linspace(-1.0, 1.0, 81)


def _observables(x, cos_minus, cos_plus):
    return {
        "x": np.histogram(x, X_EDGES)[0],
        "cos_minus": np.histogram(cos_minus, COS_EDGES)[0],
        "cos_plus": np.histogram(cos_plus, COS_EDGES)[0],
    }


def _moments(x, cos_minus, cos_plus):
    return {
        "mean_x": float(np.mean(x)),
        "mean_cos_minus": float(np.mean(cos_minus)),
        "mean_cos_plus": float(np.mean(cos_plus)),
    }


def reduce(raw_dir: Path) -> None:
    """Bin every ``<case>_<physics>.txt.gz`` in ``raw_dir`` into ``reference/``."""
    REFERENCE.mkdir(exist_ok=True)
    for case, (Z, energy) in CASES.items():
        for physics in PHYSICS:
            path = raw_dir / f"{case}_{physics}.txt.gz"
            xs, cm, cp = [], [], []
            n_lines = n_full = n_triplet = 0
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    n_lines += 1
                    fields = line.split()
                    k = float(fields[1])
                    if abs(k - energy) > 1e-6 * energy:
                        continue
                    n_full += 1
                    photon = np.array([float(v) for v in fields[2:5]])
                    secondaries = [
                        fields[i : i + 5]
                        for i in range(5, len(fields), 5)
                        if abs(int(fields[i])) < 1_000_000_000  # drop recoil ions
                    ]
                    if len(secondaries) != 2:
                        n_triplet += 1
                        continue
                    by_pdg = {int(s[0]): s for s in secondaries}
                    electron, positron = by_pdg[11], by_pdg[-11]
                    e_minus = float(electron[1])
                    xs.append(e_minus / (k - 2.0 * MC2_EV))
                    cm.append(float(np.dot(photon, [float(v) for v in electron[2:5]])))
                    cp.append(float(np.dot(photon, [float(v) for v in positron[2:5]])))
            x, cos_minus, cos_plus = map(np.asarray, (xs, cm, cp))
            record = {
                "case": case,
                "physics": physics,
                "Z": Z,
                "photon_energy_eV": energy,
                "conversions": n_lines,
                "full_energy_conversions": n_full,
                "triplet_events": n_triplet,
                "pairs": int(x.size),
                **_moments(x, cos_minus, cos_plus),
                "x_edges": X_EDGES.tolist(),
                "cos_edges": COS_EDGES.tolist(),
                **{k: v.tolist() for k, v in _observables(x, cos_minus, cos_plus).items()},
            }
            out = REFERENCE / f"{case}_{physics}.json"
            out.write_text(json.dumps(record, indent=1) + "\n")
            print(f"{out.name}: {x.size} pairs, {n_triplet} triplet of {n_full} full-energy")


def _two_sample_chi2(a, b):
    """Two-sample chi-square for unequal totals, low-count bins merged."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    merged_a, merged_b, acc_a, acc_b = [], [], 0.0, 0.0
    for ai, bi in zip(a, b, strict=True):
        acc_a, acc_b = acc_a + ai, acc_b + bi
        if acc_a >= 10 and acc_b >= 10:
            merged_a.append(acc_a)
            merged_b.append(acc_b)
            acc_a = acc_b = 0.0
    if merged_a:
        merged_a[-1] += acc_a
        merged_b[-1] += acc_b
    A, B = np.asarray(merged_a), np.asarray(merged_b)
    ka, kb = np.sqrt(B.sum() / A.sum()), np.sqrt(A.sum() / B.sum())
    chi2 = float(np.sum((ka * A - kb * B) ** 2 / (A + B)))
    dof = A.size - 1
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


def _ks_distance(a, b):
    ca, cb = np.cumsum(a) / np.sum(a), np.cumsum(b) / np.sum(b)
    return float(np.max(np.abs(ca - cb)))


def _pyrite_samples(Z, energy, n, seed):
    from pyrite.montecarlo.transport.pair_production import sample_pair

    rng = np.random.default_rng(seed)
    photon = np.array([1.0, 0.0, 0.0])
    x, cm, cp = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        pair = sample_pair(energy, Z, photon, rng)
        x[i] = pair.electron_eV / (energy - 2.0 * MC2_EV)
        cm[i] = pair.electron_direction[0]
        cp[i] = pair.positron_direction[0]
    return x, cm, cp


def compare(samples: int, output: str | None) -> int:
    report, failed = [], False
    for index, (case, (Z, energy)) in enumerate(CASES.items()):
        x, cm, cp = _pyrite_samples(Z, energy, samples, seed=27_500 + index)
        ours = _observables(x, cm, cp)
        moments = _moments(x, cm, cp)
        for physics in PHYSICS:
            ref = json.loads((REFERENCE / f"{case}_{physics}.json").read_text())
            row = {
                "case": case,
                "physics": physics,
                "pairs_geant4": ref["pairs"],
                "triplet_fraction_geant4": ref["triplet_events"]
                / max(ref["full_energy_conversions"], 1),
            }
            for name in ("x", "cos_minus", "cos_plus"):
                chi2, dof, p = _two_sample_chi2(ours[name], ref[name])
                row[name] = {
                    "chi2_per_dof": chi2 / dof,
                    "p": p,
                    "ks_distance": _ks_distance(ours[name], ref[name]),
                    "mean_pyrite": moments[f"mean_{name}"],
                    "mean_geant4": ref[f"mean_{name}"],
                }
            gated = physics == "empenelope"
            row["gated"] = gated
            row["pass"] = all(row[n]["p"] > 1e-3 for n in ("x", "cos_minus", "cos_plus"))
            failed |= gated and not row["pass"]
            report.append(row)
            print(
                f"{case:8s} {physics:16s} N={ref['pairs']:7d} "
                + "  ".join(
                    f"{n}: chi2/dof {row[n]['chi2_per_dof']:.2f} p {row[n]['p']:.3g} "
                    f"D {row[n]['ks_distance']:.3f} <> {row[n]['mean_pyrite']:.4f}/"
                    f"{row[n]['mean_geant4']:.4f}"
                    for n in ("x", "cos_minus", "cos_plus")
                )
                + (
                    f"  triplet {row['triplet_fraction_geant4']:.4f}"
                    if physics.endswith("4")
                    else ""
                )
                + ("" if not gated else ("  PASS" if row["pass"] else "  FAIL")),
                flush=True,
            )
    if output:
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", nargs="?", default="compare", choices=("compare", "reduce"))
    parser.add_argument("raw_dir", nargs="?", type=Path)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.mode == "reduce":
        if args.raw_dir is None:
            parser.error("reduce needs RAW_DIR")
        reduce(args.raw_dir)
        return 0
    return compare(args.samples, args.output)


if __name__ == "__main__":
    sys.exit(main())
