"""Isolated radiative energy, photon angle and recoil against Geant4 (issue #182).

Run explicitly after the twelve ``results/*_emission.json.gz`` PyRITE records
and the four ``reference/*_emission.log.gz`` Geant4 logs are present. Geant4
logs one ``EBREM`` line per primary ``eBrem`` photon (see README). The
acceptance tolerances below were fixed before the 100k-primary runs were
inspected.
"""

import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

HERE = Path(__file__).resolve().parent
CASES = ("w_300kev", "si_300kev", "w_800kev", "si_800kev")
CUTOFFS_EV = (1000, 5000, 10000)
REST_EV = 510_998.95
SPLIT_EV = 10_000.0

# Relative allowance for the differential cross-section source: PyRITE samples
# BremsLib partial-wave DCS, Geant4 PenBrem the scaled Seltzer-Berger DCS.
MODEL_REL_TOL = 0.05
# Photon polar-angle models also differ (BremsLib DDCS vs PENELOPE's fitted
# shape functions); absolute allowance on the mean cosine.
MEAN_COS_TOL = 0.03
# Soft-estimator allowance on total radiative energy across cutoffs: row-mean
# rather than predicted-midpoint energy, and a 2001-node soft-moment table.
CUTOFF_REL_TOL = 0.005
# Track-length expectation allowance: 1 keV energy bins evaluated at their
# centre, while transport samples the hazard at each flight's start energy.
EXPECTATION_REL_TOL = 0.01
SIGMA = 3.0
_AVOGADRO = 6.02214076e23
# TestEm5 DetectorConstruction materials: (element, g/mol, g/cm3).
GEANT4_MATERIAL = {"w": ("W", 183.85, 19.30), "si": ("Si", 28.09, 2.330)}


def _momentum(kinetic_eV):
    return np.sqrt(kinetic_eV * (kinetic_eV + 2.0 * REST_EV))


def _geant4(case):
    """Per-photon arrays reconstructed from the Geant4 emission log."""
    with gzip.open(HERE / "reference" / f"{case}_emission.log.gz", "rt") as handle:
        log = handle.read()
    sample = re.search(r"The run was (\d+) e- of ([\d.]+) (keV|MeV)", log)
    assert sample
    count = int(sample.group(1))
    assert re.search(r"Energy balance :  edep \+ eleak = ", log)
    rows = np.array(
        [line.split()[1:] for line in re.findall(r"EBREM [^\n]*", log)], dtype=float
    ).reshape(-1, 9)
    event = rows[:, 0].astype(np.int64)
    after, photon = rows[:, 1], rows[:, 2]
    u_after, n = rows[:, 3:6], rows[:, 6:9]
    before = after + photon
    p_in, p_out = _momentum(before), _momentum(after)
    # PenBrem sets u_after = unit(p_in u - k n); invert for the pre-emission u.
    c_after = np.einsum("ij,ij->i", u_after, n)
    a = -photon * c_after + np.sqrt(p_in**2 - photon**2 * (1.0 - c_after**2))
    u = (a[:, None] * u_after + photon[:, None] * n) / p_in[:, None]
    live = after > 0.0
    assert np.allclose(np.linalg.norm(u[live], axis=1), 1.0, atol=1e-9)
    native_recoil = (a - p_out)[:, None] * u_after
    path = HERE / "reference" / f"{case}_emission_h1_h63.csv"
    with path.open() as handle:
        lines = [line for line in handle if not line.startswith("#")]
    track_um = np.array([float(line.split(",")[1]) for line in lines[1:]])
    assert track_um[0] == 0.0 and track_um[-1] == 0.0  # no under/overflow
    return {
        "Ne": count,
        "track_ang": track_um[1:-1] * 1e4,
        "event": event,
        "live": live,
        "before_eV": before,
        "photon_eV": photon,
        "cos": np.einsum("ij,ij->i", u, n),
        "deflection_cos": np.einsum("ij,ij->i", u, u_after),
        "native_recoil_eV_c": np.linalg.norm(native_recoil, axis=1),
    }


def _pyrite(case, cutoff):
    path = HERE / "results" / f"{case}_{cutoff}ev_emission.json.gz"
    with gzip.open(path, "rt") as handle:
        sample = json.load(handle)
    assert sample["case"] == case
    assert sample["radiative_cutoff_eV"] == cutoff
    assert sample["elastic_model"] == "mott"
    assert sample["max_hard_recoil_residual_eV_c"] < 1e-5
    photons = np.asarray(sample["hard_photons"], dtype=float).reshape(-1, 4)
    return sample, {
        "Ne": sample["Ne"],
        "event": photons[:, 0].astype(np.int64),
        "before_eV": photons[:, 1],
        "photon_eV": photons[:, 2],
        "cos": photons[:, 3],
    }


def _expected(track_ang, element, number_density_ang3, threshold_eV):
    """BremsLib photon energy and count per unit primary for a track-length spectrum.

    ``track_ang`` is the summed track length in 1 keV bins of electron energy.
    Returns (energy [eV], count) for photons above ``threshold_eV``.
    """
    table = load_bremsstrahlung_tables([element])[element]
    centres = (np.arange(track_ang.size) + 0.5) * 1000.0
    energy = number = 0.0
    for length, kinetic in zip(track_ang, centres, strict=True):
        if length <= 0.0 or kinetic <= threshold_eV:
            continue
        part = build_radiative_partition(table, kinetic, threshold_eV)
        scale = number_density_ang3 * 1e16 * length  # Angstrom^-3 cm^2 Angstrom -> 1
        energy += scale * part.hard_stopping_cs_eV_cm2
        number += scale * part.hard_rate_cs_cm2
    return energy, number


def _per_primary_mean(event, weight, count):
    """Mean and standard error of a per-primary sum."""
    totals = np.bincount(event, weights=weight, minlength=count)
    return float(totals.mean()), float(totals.std(ddof=1) / math.sqrt(count))


def _pyrite_convention_recoil(before, photon, cosine):
    """|q| and q along the electron when the electron keeps its direction."""
    along = _momentum(before) - _momentum(before - photon)
    magnitude = np.sqrt(along**2 + photon**2 - 2.0 * along * photon * cosine)
    return magnitude, along - photon * cosine


def _mean(values):
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


_FAILURES = []


def _check(label, pyrite, geant4, rel_tol, abs_tol=0.0):
    (p, sp), (g, sg) = pyrite, geant4
    sigma = math.hypot(sp, sg)
    delta = p - g
    allowed = rel_tol * abs(g) + abs_tol + SIGMA * sigma
    z = delta / sigma if sigma else 0.0
    rel = delta / g if g else 0.0
    ok = abs(delta) <= allowed
    print(
        f"{label} {p:.6g}±{sp:.2g} {g:.6g}±{sg:.2g} rel={rel:+.4f} z={z:+.2f}"
        f" {'ok' if ok else 'FAIL'}"
    )
    if not ok:
        _FAILURES.append(f"{label}: |{delta:.4g}| > {allowed:.4g}")


def test_compare_emission_reference():
    _FAILURES.clear()
    for case in CASES:
        geant4 = _geant4(case)
        # Photon energy integrals in matched windows: k > 1 keV (PyRITE's
        # hard photons at the 1 keV cutoff) and k >= 10 keV.
        g_ge1 = _per_primary_mean(
            geant4["event"], geant4["photon_eV"] * (geant4["photon_eV"] > 1000.0), geant4["Ne"]
        )
        g_ge10 = _per_primary_mean(
            geant4["event"], geant4["photon_eV"] * (geant4["photon_eV"] >= SPLIT_EV), geant4["Ne"]
        )
        print(f"{case} geant4 minimum eBrem photon {geant4['photon_eV'].min():.1f} eV")
        # Radiative physics isolated from elastic transport: Geant4's realized
        # photons against BremsLib moments on Geant4's own primary track-length
        # spectrum. The difference measures PenBrem against BremsLib.
        element, molar_mass, density = GEANT4_MATERIAL[case.split("_")[0]]
        g_density = density * _AVOGADRO / molar_mass / 1e24
        print(
            f"{case} geant4 primary track {geant4['track_ang'].sum() / geant4['Ne'] / 1e4:.4g} um/e"
        )
        for threshold, label in ((1000.0, "k>1keV"), (SPLIT_EV, "k>=10keV")):
            energy, number = _expected(geant4["track_ang"], element, g_density, threshold)
            keep = (
                geant4["photon_eV"] > threshold
                if threshold == 1000.0
                else (geant4["photon_eV"] >= threshold)
            )
            realized = _per_primary_mean(geant4["event"], geant4["photon_eV"] * keep, geant4["Ne"])
            _check(
                f"{case} geant4-path BremsLib/PenBrem E_rad({label}) eV/e",
                (energy / geant4["Ne"], 0.0),
                realized,
                MODEL_REL_TOL,
            )
            n_real = int(np.count_nonzero(keep))
            _check(
                f"{case} geant4-path BremsLib/PenBrem N({label}) /e",
                (number / geant4["Ne"], 0.0),
                (n_real / geant4["Ne"], math.sqrt(n_real) / geant4["Ne"]),
                MODEL_REL_TOL,
            )
        totals = {}
        for cutoff in CUTOFFS_EV:
            sample, pyrite = _pyrite(case, cutoff)
            count = pyrite["Ne"]
            ge10 = pyrite["photon_eV"] >= SPLIT_EV
            p_ge10 = _per_primary_mean(pyrite["event"], pyrite["photon_eV"] * ge10, count)
            _check(f"{case} {cutoff} E_rad(k>=10keV) eV/e", p_ge10, g_ge10, MODEL_REL_TOL)
            if cutoff == 1000:
                p_ge1 = _per_primary_mean(pyrite["event"], pyrite["photon_eV"], count)
                _check(f"{case} {cutoff} E_rad(k>1keV) eV/e", p_ge1, g_ge1, MODEL_REL_TOL)
            # Implementation check isolated from elastic transport: PyRITE's
            # realized hard photons against BremsLib moments on its own
            # track-length spectrum.
            track = np.asarray(sample["track_length_ang_per_keV_bin"], dtype=float)
            energy, number = _expected(track, element, sample["number_density_ang3"], cutoff)
            _check(
                f"{case} {cutoff} pyrite-path E_hard eV/e",
                _per_primary_mean(pyrite["event"], pyrite["photon_eV"], count),
                (energy / count, 0.0),
                EXPECTATION_REL_TOL,
            )
            n_hard = pyrite["photon_eV"].size
            _check(
                f"{case} {cutoff} pyrite-path N_hard /e",
                (n_hard / count, math.sqrt(n_hard) / count),
                (number / count, 0.0),
                EXPECTATION_REL_TOL,
            )
            print(f"{case} {cutoff} pyrite primary track {track.sum() / count / 1e4:.4g} um/e")
            mean = sample["radiative_estimate_sum_eV"] / count
            variance = sample["radiative_estimate_sumsq_eV2"] / count - mean**2
            totals[cutoff] = (mean, math.sqrt(variance * count / (count - 1) / count))
            print(
                f"{case} {cutoff} soft {sample['soft_radiative_estimate_sum_eV'] / count:.4g}"
                f" hard+soft {mean:.6g} eV/e"
            )
        # Threshold convergence of the isolated radiative loss. Runs share a
        # seed, so treating them as independent is approximate.
        for cutoff in CUTOFFS_EV[1:]:
            _check(
                f"{case} {cutoff}-vs-1000 E_rad(hard+soft) eV/e",
                totals[cutoff],
                totals[1000],
                CUTOFF_REL_TOL,
            )

        # Photon angle and recoil for k >= 10 keV at the 1 keV cutoff. Recoil is
        # evaluated in PyRITE's convention (electron keeps its direction) on
        # both samples, so it compares the joint (T, k, theta) distributions.
        _, pyrite = _pyrite(case, 1000)
        p_mask = pyrite["photon_eV"] >= SPLIT_EV
        g_mask = (geant4["photon_eV"] >= SPLIT_EV) & geant4["live"]
        p_cos, g_cos = pyrite["cos"][p_mask], geant4["cos"][g_mask]
        _check(f"{case} <cos theta_gamma>", _mean(p_cos), _mean(g_cos), 0.0, MEAN_COS_TOL)
        edges = np.linspace(-1.0, 1.0, 11)
        p_hist = np.histogram(p_cos, edges)[0] / p_cos.size
        g_hist = np.histogram(g_cos, edges)[0] / g_cos.size
        print(f"{case} cos-theta fraction PyRITE {np.round(p_hist, 4).tolist()}")
        print(f"{case} cos-theta fraction Geant4 {np.round(g_hist, 4).tolist()}")
        p_q, p_along = _pyrite_convention_recoil(
            pyrite["before_eV"][p_mask], pyrite["photon_eV"][p_mask], p_cos
        )
        g_q, g_along = _pyrite_convention_recoil(
            geant4["before_eV"][g_mask], geant4["photon_eV"][g_mask], g_cos
        )
        _check(f"{case} <|q|> eV/c", _mean(p_q), _mean(g_q), MODEL_REL_TOL)
        _check(f"{case} <q.u> eV/c", _mean(p_along), _mean(g_along), MODEL_REL_TOL)
        # Convention difference, reported: Geant4 deflects the electron to
        # conserve momentum with the photon, leaving a collinear target recoil.
        native = geant4["native_recoil_eV_c"][g_mask]
        deflection = np.degrees(np.arccos(np.clip(geant4["deflection_cos"][g_mask], -1.0, 1.0)))
        print(
            f"{case} geant4-convention <|q|> {native.mean():.6g} eV/c;"
            f" electron deflection mean {deflection.mean():.3g} deg,"
            f" p99 {np.percentile(deflection, 99):.3g} deg, max {deflection.max():.3g} deg"
        )
    assert not _FAILURES, "\n".join(_FAILURES)
