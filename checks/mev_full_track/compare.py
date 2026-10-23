"""3-100 MeV full-track transport: PyRITE against Geant4 TestEm5 (issue #439).

``reduce-geant4 RAW_DIR`` reads each ``<case>.log.gz`` (``EXIT`` and ``EBREM``
lines from ``patch_tracking.py`` and the #182 emission patch) and its
``<case>_h1_h63.csv`` track-length histogram into ``reference/<case>.json``.
``reduce-pyrite RAW_DIR`` merges each case's ``<case>_pyrite_<shard>.npz``
shards from ``pyrite_bench.py`` into ``results/<case>.json``; with
``--variant point`` it merges the point-nucleus control shards
(``run_control.sbatch``) into ``results/<case>_point.json``. Both use the
same per-primary observables, so standard errors come from per-primary sums.

``compare`` (the default) applies the gates fixed before the PyRITE results
were inspected. An observable passes when ``|PyRITE - Geant4| <=
max(0.05 |Geant4|, 3 sigma)``, sigma combining both standard errors (the #439
acceptance: reference uncertainty or 5 %, whichever is larger). Gated:

* fraction of primaries not transmitted;
* mean continuous deposit per primary, ``E0 - E_exit - sum k``;
* mean energy loss of transmitted primaries;
* mean and rms polar exit angle of transmitted primaries;
* primary bremsstrahlung energy per primary for ``k >= 10 keV`` and
  ``k >= 100 keV``, and the photon count for ``k >= 100 keV``.

Reported, not gated: photon counts above 1 and 10 keV (PenBrem has no
Ter-Mikaelian suppression, which reaches ``gamma hbar omega_p`` ~ 16 keV for
W at 100 MeV); the isolated model check (BremsLib vacuum moments on Geant4's
own primary track-length spectrum against Geant4's photons); the
implementation check (PyRITE's realized hard photons against BremsLib moments
with the coupled plasma energy on PyRITE's own track-length spectrum); and
Kolmogorov distances of the binned loss, angle and photon spectra.

Validation: mev-full-track-transport

Run:
  uv run python checks/mev_full_track/compare.py reduce-geant4 RAW_DIR
  uv run python checks/mev_full_track/compare.py reduce-pyrite RAW_DIR
  uv run python checks/mev_full_track/compare.py [--output REPORT.json]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
RESULTS = HERE / "results"
TARGETS = {"w": ("W", 19.3, 183.84, 100e-4), "si": ("Si", 2.329, 28.085, 1000e-4)}
ENERGIES_MEV = (3, 10, 30, 50, 100)
CASES = {f"{t}_{e}mev": (t, float(e)) for t in TARGETS for e in ENERGIES_MEV}
PHOTON_THRESHOLDS_EV = (1e3, 1e4, 1e5, 1e6)
GATED = (
    "not_transmitted",
    "deposit_eV",
    "transmitted_loss_eV",
    "theta_mean_rad",
    "theta_rms_rad",
    "photon_energy_ge_10keV_eV",
    "photon_energy_ge_100keV_eV",
    "photon_count_ge_100keV",
)
REL_TOLERANCE = 0.05


def _mean_se(values):
    values = np.asarray(values, float)
    if values.size == 0:
        return 0.0, 0.0
    se = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
    return float(values.mean()), float(se)


def _edges(E0_eV):
    return {
        "loss": np.linspace(3.0, np.log10(E0_eV), 81),
        "theta": np.linspace(-5.0, np.log10(np.pi), 81),
        "photon": np.linspace(3.0, np.log10(E0_eV), 81),
    }


def observables(E0_eV, fate, E_exit_eV, cos_exit, photon_primary, photon_k_eV):
    """Per-primary observables, their means and standard errors, and spectra.

    ``fate``: 0 transmitted, 1 reflected, 2 absorbed, 3 side exit.
    """
    n = fate.size
    k_sum = np.bincount(photon_primary, weights=photon_k_eV, minlength=n)
    out = {"primaries": int(n)}
    for i, name in enumerate(("transmitted", "reflected", "absorbed", "side")):
        out[f"n_{name}"] = int(np.count_nonzero(fate == i))
    out["not_transmitted"] = _mean_se(fate != 0)
    out["deposit_eV"] = _mean_se(E0_eV - E_exit_eV - k_sum)
    t = fate == 0
    loss = E0_eV - E_exit_eV[t]
    theta = np.arccos(np.clip(cos_exit[t], -1.0, 1.0))
    out["transmitted_loss_eV"] = _mean_se(loss)
    out["theta_mean_rad"] = _mean_se(theta)
    mean_sq, se_sq = _mean_se(theta**2)
    rms = np.sqrt(mean_sq)
    out["theta_rms_rad"] = (float(rms), float(se_sq / (2.0 * rms)) if rms > 0 else 0.0)
    for threshold in PHOTON_THRESHOLDS_EV:
        keep = photon_k_eV >= threshold
        label = f"{threshold / 1e3:g}keV".replace("1000keV", "1MeV")
        count = np.bincount(photon_primary[keep], minlength=n)
        energy = np.bincount(photon_primary[keep], weights=photon_k_eV[keep], minlength=n)
        out[f"photon_count_ge_{label}"] = _mean_se(count)
        out[f"photon_energy_ge_{label}_eV"] = _mean_se(energy)
    edges = _edges(E0_eV)
    out["hist"] = {
        "loss": np.histogram(np.log10(np.maximum(loss, 1.0)), edges["loss"])[0].tolist(),
        "theta": np.histogram(np.log10(np.maximum(theta, 1e-12)), edges["theta"])[0].tolist(),
        "photon": np.histogram(np.log10(photon_k_eV), edges["photon"])[0].tolist(),
    }
    out["edges"] = {k: v.tolist() for k, v in edges.items()}
    return out


def _parse_h63(path):
    """TestEm5 ASCII CSV: per-bin ``Sw`` (summed step length, um), no under/overflow."""
    rows = [line for line in Path(path).read_text().splitlines() if line and line[0] != "#"]
    header = rows[0].split(",")
    sw = np.array([float(r.split(",")[header.index("Sw")]) for r in rows[1:]])
    return sw[1:-1]


def _geant4_arrays(log_path):
    exits, photons = {}, []
    with gzip.open(log_path, "rt") as handle:
        for line in handle:
            body = line.split("> ", 1)[1] if line.startswith("G4WT") else line
            if body.startswith("EXIT "):
                f = body.split()
                exits[int(f[1])] = ({"T": 0, "R": 1, "A": 2}[f[2]], float(f[3]), float(f[4]))
            elif body.startswith("EBREM "):
                f = body.split()
                photons.append((int(f[1]), float(f[3])))
    events = np.array(sorted(exits))
    if not np.array_equal(events, np.arange(events.size)):
        raise SystemExit(f"{log_path}: EXIT events are not 0..N-1")
    fate, E_exit, cos_exit = (np.array(c) for c in zip(*(exits[e] for e in events), strict=True))
    fate = fate.astype(np.int8)
    # A primary that left through the 1 cm side faces is neither transmitted
    # nor reflected in TestEm5 but keeps its energy; stopped primaries end at 0.
    fate[(fate == 2) & (E_exit > 0.0)] = 3
    ph = np.array(sorted(set(photons))) if photons else np.zeros((0, 2))
    return fate, E_exit, cos_exit, ph[:, 0].astype(np.int64), ph[:, 1]


def reduce_geant4(raw_dir):
    REFERENCE.mkdir(exist_ok=True)
    for case, (_, E0) in CASES.items():
        fate, E_exit, cos_exit, ph_primary, ph_k = _geant4_arrays(raw_dir / f"{case}.log.gz")
        record = {"case": case, "code": "geant4-11.4.2-testem5-empenelope", "E0_eV": E0 * 1e6}
        record.update(observables(E0 * 1e6, fate, E_exit, cos_exit, ph_primary, ph_k))
        record["track_length_um"] = _parse_h63(raw_dir / f"{case}_h1_h63.csv").tolist()
        (REFERENCE / f"{case}.json").write_text(json.dumps(record) + "\n")
        print(f"{case}: {fate.size} primaries, {ph_k.size} photons")


def _result_path(case, variant):
    return RESULTS / (f"{case}.json" if variant == "pyrite" else f"{case}_{variant}.json")


def reduce_pyrite(raw_dir, variant="pyrite"):
    RESULTS.mkdir(exist_ok=True)
    for case, (_, E0) in CASES.items():
        shards = sorted(
            raw_dir.glob(f"{case}_{variant}_*.npz"), key=lambda p: int(p.stem.split("_")[-1])
        )
        if not shards:
            print(f"{case}: no shards")
            continue
        parts, offset, track, metas = [], 0, None, []
        for path in shards:
            with np.load(path) as d:
                parts.append(
                    (
                        d["fate"],
                        d["E_exit_eV"],
                        d["cos_exit"],
                        d["photon_primary"] + offset,
                        d["photon_k_eV"],
                    )
                )
                offset += d["fate"].size
                track = d["track_length_um"] if track is None else track + d["track_length_um"]
                metas.append(json.loads(str(d["meta"])))
        fate, E_exit, cos_exit, ph_primary, ph_k = (
            np.concatenate(c) for c in zip(*parts, strict=True)
        )
        record = {"case": case, "code": "pyrite", "E0_eV": E0 * 1e6, "shards": len(shards)}
        record.update(
            observables(E0 * 1e6, fate, E_exit, cos_exit, ph_primary.astype(np.int64), ph_k)
        )
        record["track_length_um"] = track.tolist()
        keep = ("sbethe", "elsepa", "bremslib", "radiative", "density_per_ang3", "thickness_ang")
        record["meta"] = {k: metas[0][k] for k in keep}
        record["meta"]["elastic_override"] = metas[0].get("elastic_override")
        record["runtime_s"] = sum(m["runtime_s"] for m in metas)
        _result_path(case, variant).write_text(json.dumps(record) + "\n")
        print(f"{case}: {fate.size} primaries from {len(shards)} shards")


def _model_moments(case, track_um, plasma_eV, thresholds_eV):
    """BremsLib hard energy and count per primary above each threshold on a track spectrum."""
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    target, E0 = CASES[case]
    element, rho, A, _ = TARGETS[target]
    table = load_bremsstrahlung_tables([element])[element]
    n_cm3 = rho * 6.02214076e23 / TRANSPORT_ELEMENTS[element]["A"]
    # 100 merged bins: each partition restages the table, and these checks
    # are reported, not gated (bin-centre error ~E0/200 on a smooth moment).
    track = np.asarray(track_um).reshape(100, -1).sum(axis=1)
    centres = (np.arange(track.size) + 0.5) * E0 * 1e6 / track.size
    energy = np.zeros(len(thresholds_eV))
    count = np.zeros(len(thresholds_eV))
    for T, length_um in zip(centres, track, strict=True):
        if length_um <= 0.0 or T <= 1.0e4:
            continue
        for j, threshold in enumerate(thresholds_eV):
            if threshold >= T:
                continue
            part = build_radiative_partition(table, T, threshold, plasma_energy_eV=plasma_eV)
            energy[j] += part.hard_stopping_cs_eV_cm2 * n_cm3 * length_um * 1e-4
            count[j] += part.hard_rate_cs_cm2 * n_cm3 * length_um * 1e-4
    return energy, count


def compare(output, variant="pyrite"):
    report, failed = [], False
    for case in CASES:
        ref_path, ours_path = REFERENCE / f"{case}.json", _result_path(case, variant)
        if not (ref_path.exists() and ours_path.exists()):
            print(f"{case}: missing reference or result, skipped")
            continue
        ref, ours = json.loads(ref_path.read_text()), json.loads(ours_path.read_text())
        row = {"case": case, "gated": {}, "reported": {}}
        for name in GATED:
            (a, sa), (b, sb) = ours[name], ref[name]
            sigma = float(np.hypot(sa, sb))
            allowed = max(REL_TOLERANCE * abs(b), 3.0 * sigma)
            ok = abs(a - b) <= allowed
            failed |= not ok
            row["gated"][name] = dict(
                pyrite=a,
                geant4=b,
                ratio=a / b if b else None,
                z=(a - b) / sigma if sigma else None,
                pass_=ok,
            )
        for name in ("photon_count_ge_1keV", "photon_count_ge_10keV", "photon_energy_ge_1MeV_eV"):
            (a, sa), (b, sb) = ours[name], ref[name]
            sigma = float(np.hypot(sa, sb))
            row["reported"][name] = dict(
                pyrite=a, geant4=b, ratio=a / b if b else None, z=(a - b) / sigma if sigma else None
            )
        for name in ("loss", "theta", "photon"):
            ca = np.cumsum(ours["hist"][name]) / max(np.sum(ours["hist"][name]), 1)
            cb = np.cumsum(ref["hist"][name]) / max(np.sum(ref["hist"][name]), 1)
            row["reported"][f"ks_{name}"] = float(np.max(np.abs(ca - cb)))
        thresholds = (1e4, 1e5)
        N_ref, N_ours = ref["primaries"], ours["primaries"]
        iso_E, iso_N = _model_moments(case, ref["track_length_um"], 0.0, thresholds)
        plasma = ours["meta"]["radiative"].get("plasma_energy_eV_by_layer", [0.0])[0]
        imp_E, imp_N = _model_moments(case, ours["track_length_um"], plasma, thresholds)
        for j, label in enumerate(("10keV", "100keV")):
            g4_E, g4_sE = ref[f"photon_energy_ge_{label}_eV"]
            g4_N, g4_sN = ref[f"photon_count_ge_{label}"]
            py_E, py_sE = ours[f"photon_energy_ge_{label}_eV"]
            py_N, py_sN = ours[f"photon_count_ge_{label}"]
            row["reported"][f"isolated_energy_ge_{label}"] = dict(
                bremslib=iso_E[j] / N_ref,
                geant4=g4_E,
                ratio=iso_E[j] / N_ref / g4_E,
                z=(iso_E[j] / N_ref - g4_E) / g4_sE,
            )
            row["reported"][f"isolated_count_ge_{label}"] = dict(
                bremslib=iso_N[j] / N_ref,
                geant4=g4_N,
                ratio=iso_N[j] / N_ref / g4_N,
                z=(iso_N[j] / N_ref - g4_N) / g4_sN,
            )
            row["reported"][f"implementation_energy_ge_{label}"] = dict(
                partition=imp_E[j] / N_ours,
                realized=py_E,
                ratio=py_E / (imp_E[j] / N_ours),
                z=(py_E - imp_E[j] / N_ours) / py_sE,
            )
            row["reported"][f"implementation_count_ge_{label}"] = dict(
                partition=imp_N[j] / N_ours,
                realized=py_N,
                ratio=py_N / (imp_N[j] / N_ours),
                z=(py_N - imp_N[j] / N_ours) / py_sN,
            )
        row["fractions"] = {
            code: {
                k: rec[f"n_{k}"] / rec["primaries"]
                for k in ("transmitted", "reflected", "absorbed", "side")
            }
            for code, rec in (("pyrite", ours), ("geant4", ref))
        }
        row["pass"] = all(v["pass_"] for v in row["gated"].values())
        report.append(row)
        print(f"{case:10s} {'PASS' if row['pass'] else 'FAIL'}")
        for name, v in row["gated"].items():
            print(
                f"   {name:28s} {v['pyrite']:.6g} / {v['geant4']:.6g}  ratio {v['ratio'] if v['ratio'] is None else round(v['ratio'], 4)}  z {v['z'] if v['z'] is None else round(v['z'], 2)}  {'ok' if v['pass_'] else 'FAIL'}"
            )
        for name, v in row["reported"].items():
            if isinstance(v, dict):
                print(
                    f"   ({name}) ratio {v['ratio']:.4f} z {v['z']:.2f}"
                    if v.get("ratio") is not None
                    else f"   ({name}) n/a"
                )
            else:
                print(f"   ({name}) {v:.4f}")
    if output:
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "mode", nargs="?", default="compare", choices=("compare", "reduce-geant4", "reduce-pyrite")
    )
    parser.add_argument("raw_dir", nargs="?", type=Path)
    parser.add_argument("--output")
    parser.add_argument(
        "--variant",
        default="pyrite",
        help="shard tag: pyrite (production) or point (point-nucleus control)",
    )
    args = parser.parse_args(argv)
    if args.mode == "reduce-geant4":
        reduce_geant4(args.raw_dir)
        return 0
    if args.mode == "reduce-pyrite":
        reduce_pyrite(args.raw_dir, args.variant)
        return 0
    return compare(args.output, args.variant)


if __name__ == "__main__":
    sys.exit(main())
