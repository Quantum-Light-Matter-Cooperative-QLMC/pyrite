"""Summarize raw TestEm5 output against the issue #182 PyRITE cutoff sweep.

Run explicitly after copying the six PyRITE JSON records into ``results/``.
The printed sigma values describe sampling differences, not model agreement.
"""

import csv
import gzip
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("w_300kev", "si_300kev", "w_800kev", "si_800kev")
CUTOFFS_EV = (1000, 5000, 10000)


def _reference(case):
    with gzip.open(HERE / "reference" / f"{case}.log.gz", "rt") as handle:
        log = handle.read()
    sample = re.search(r"The run was (\d+) e- of ([\d.]+) keV", log)
    transmit = re.search(r"primary particle transmitted = ([\d.]+) %", log)
    reflect = re.search(r"primary particle reflected = ([\d.]+) %", log)
    gammas = re.search(r"Number of secondaries per event : Gammas = ([\d.]+)", log)
    assert sample and transmit and reflect and gammas
    count = int(sample.group(1))
    energy_keV = int(case.split("_")[1].removesuffix("kev"))
    assert float(sample.group(2)) == energy_keV
    path = HERE / "reference" / f"{case}_h1_h62.csv"
    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert len(rows) == energy_keV // 10 + 2  # underflow, ten-keV bins, overflow
    bins = [int(float(row["Sw"])) for row in rows[1:-1]]
    total = sum(bins) + int(float(rows[0]["Sw"])) + int(float(rows[-1]["Sw"]))
    # Run.cc prints the mean with four significant digits.
    assert abs(total / count - float(gammas.group(1))) < 0.0001
    with (HERE / "reference" / f"{case}_h1_h3.csv").open(newline="") as handle:
        all_gamma = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    assert [row["Sw"] for row in rows] == [row["Sw"] for row in all_gamma]
    return {
        "Ne": count,
        "n_transmitted": round(float(transmit.group(1)) * count / 100),
        "n_backscattered": round(float(reflect.group(1)) * count / 100),
        "n_photons_ge_10kev": sum(bins[1:]),
        "photon_bin_counts": bins,
    }


def _poisson_z(pyrite, geant4, n_pyrite, n_geant4):
    variance = pyrite / n_pyrite**2 + geant4 / n_geant4**2
    if variance == 0.0:
        return 0.0
    return (pyrite / n_pyrite - geant4 / n_geant4) / math.sqrt(variance)


def _binomial_z(pyrite, geant4, n_pyrite, n_geant4):
    p = pyrite / n_pyrite
    g = geant4 / n_geant4
    variance = p * (1.0 - p) / n_pyrite + g * (1.0 - g) / n_geant4
    return (p - g) / math.sqrt(variance)


def test_compare_full_track_reference():
    print("case cutoff_eV observable PyRITE Geant4 delta z")
    for case in CASES:
        reference = _reference(case)
        energy_keV = int(case.split("_")[1].removesuffix("kev"))
        baseline = None
        for cutoff in CUTOFFS_EV if energy_keV == 300 else (1000,):
            path = HERE / "results" / f"{case}_{cutoff}ev.json"
            sample = json.loads(path.read_text())
            assert sample["case"] == case
            assert sample["radiative_cutoff_eV"] == cutoff
            assert sample["elastic_model"] == "mott"
            assert sample["max_hard_recoil_residual_eV_c"] < 1e-5
            n_pyrite, n_geant4 = sample["Ne"], reference["Ne"]
            for label, key in (
                ("transmitted", "n_transmitted"),
                ("backscattered", "n_backscattered"),
            ):
                pyrite = sample[key]
                geant4 = reference[key]
                z = _binomial_z(pyrite, geant4, n_pyrite, n_geant4)
                print(f"{case} {cutoff} {label} {pyrite}/{n_pyrite} {geant4}/{n_geant4} {z:+.2f}")
            pyrite = sample["n_hard_photons_ge_10kev"]
            geant4 = reference["n_photons_ge_10kev"]
            z = _poisson_z(pyrite, geant4, n_pyrite, n_geant4)
            print(
                f"{case} {cutoff} photons>=10keV {pyrite}/{n_pyrite} {geant4}/{n_geant4} {z:+.2f}"
            )
            # Three-standard-error screening tolerance for the
            # comparable source-yield observable. This is not a physics sign-off.
            assert abs(z) <= 3.0
            if cutoff == 1000:
                baseline = sample
            else:
                assert baseline is not None
                cutoff_z = _poisson_z(
                    pyrite,
                    baseline["n_hard_photons_ge_10kev"],
                    n_pyrite,
                    baseline["Ne"],
                )
                print(f"{case} {cutoff} cutoff-vs-1keV photons>=10keV {cutoff_z:+.2f}")
                assert abs(cutoff_z) <= 3.0
            windows = [
                ("10-50keV", 1, 5),
                ("50-100keV", 5, 10),
                ("100-300keV", 10, 30),
            ]
            if energy_keV == 800:
                windows.append(("300-800keV", 30, 80))
            for label, start, stop in windows:
                pyrite = sum(sample["photon_bin_counts"][start:stop])
                geant4 = sum(reference["photon_bin_counts"][start:stop])
                z = _poisson_z(pyrite, geant4, n_pyrite, n_geant4)
                print(f"{case} {cutoff} {label} {pyrite}/{n_pyrite} {geant4}/{n_geant4} {z:+.2f}")

    # Elastic-model controls: only the elastic option differs from the Mott runs.
    # Terminal fractions are reported, not asserted; see README disposition.
    for case in ("w_300kev", "w_800kev", "si_800kev"):
        sr = json.loads((HERE / "results" / f"{case}_sr_1000ev.json").read_text())
        assert sr["case"] == case
        assert sr["elastic_model"] == "sr"
        reference = _reference(case)
        for label, key in (
            ("transmitted", "n_transmitted"),
            ("backscattered", "n_backscattered"),
        ):
            z = _binomial_z(sr[key], reference[key], sr["Ne"], reference["Ne"])
            print(
                f"{case} sr-control {label} {sr[key]}/{sr['Ne']} {reference[key]}/{reference['Ne']} {z:+.2f}"
            )
        z = _poisson_z(
            sr["n_hard_photons_ge_10kev"],
            reference["n_photons_ge_10kev"],
            sr["Ne"],
            reference["Ne"],
        )
        print(f"{case} sr-control photons>=10keV {z:+.2f}")
        assert abs(z) <= 3.0
