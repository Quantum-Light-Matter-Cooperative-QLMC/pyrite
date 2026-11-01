"""MeV full-track transport against Geant4 11.4.2 TestEm5 (issue #439).

Validation: mev-full-track-transport

The lab-box benchmark in ``checks/mev_full_track`` transports 100,000
primaries per case; this module pins its committed reductions and reruns two
cases at small statistics against the same Geant4 references and gates.
"""

import json
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[3] / "checks" / "mev_full_track"
CASES = [f"{t}_{e}mev" for t in ("w", "si") for e in (3, 10, 30, 50, 100)]
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
#: Gate failures traced to the reference: Geant4's GS Mott correction assumes
#: a point nucleus, PyRITE's ELSEPA tables a Fermi charge distribution, which
#: suppresses the large-angle exit tail above ~0.3 rad in W at 50-100 MeV.
#: The point-nucleus control (``<case>_point.json``) passes them.
REFERENCE_MODEL_FAILURES = {("w_50mev", "theta_rms_rad"), ("w_100mev", "theta_rms_rad")}


def _gate(pyrite, geant4):
    (a, sa), (b, sb) = pyrite, geant4
    return abs(a - b) <= max(0.05 * abs(b), 3.0 * np.hypot(sa, sb))


def _load(case, variant=""):
    reference = json.loads((BENCH / "reference" / f"{case}.json").read_text())
    result = json.loads((BENCH / "results" / f"{case}{variant}.json").read_text())
    return reference, result


@pytest.mark.parametrize("case", CASES)
def test_committed_reduction_meets_gates(case):
    """Every gated observable of the 100k-primary run passes, except the
    reference-model failures recorded above, which must still fail (a change
    that closes them needs the ledger row updated)."""
    reference, result = _load(case)
    assert result["primaries"] == reference["primaries"] == 100_000
    for name in GATED:
        passed = _gate(result[name], reference[name])
        assert passed != ((case, name) in REFERENCE_MODEL_FAILURES), (case, name)


@pytest.mark.parametrize("case", sorted({case for case, _ in REFERENCE_MODEL_FAILURES}))
def test_point_nucleus_control_meets_gates(case):
    """With a point-nucleus ELSEPA W table, as Geant4's GS Mott correction
    assumes, the reference-model failures close and every gate passes."""
    reference, result = _load(case, "_point")
    assert result["meta"]["elastic_override"] == "point-nucleus"
    for name in GATED:
        assert _gate(result[name], reference[name]), (case, name)


def _observables(E0_eV, fate, E_exit_eV, cos_exit, photon_primary, photon_k_eV):
    """Means and standard errors as in ``checks/mev_full_track/compare.py``."""

    def mean_se(values):
        values = np.asarray(values, float)
        return values.mean(), values.std(ddof=1) / np.sqrt(values.size)

    n = fate.size
    t = fate == 0
    theta = np.arccos(np.clip(cos_exit[t], -1.0, 1.0))
    mean_sq, se_sq = mean_se(theta**2)
    out = {
        "not_transmitted": mean_se(fate != 0),
        "deposit_eV": mean_se(
            E0_eV - E_exit_eV - np.bincount(photon_primary, photon_k_eV, minlength=n)
        ),
        "transmitted_loss_eV": mean_se(E0_eV - E_exit_eV[t]),
        "theta_mean_rad": mean_se(theta),
        "theta_rms_rad": (np.sqrt(mean_sq), se_sq / (2.0 * np.sqrt(mean_sq))),
    }
    for label, threshold in (("10keV", 1e4), ("100keV", 1e5)):
        keep = photon_k_eV >= threshold
        energy = np.bincount(photon_primary[keep], photon_k_eV[keep], minlength=n)
        out[f"photon_energy_ge_{label}_eV"] = mean_se(energy)
        out[f"photon_count_ge_{label}"] = mean_se(np.bincount(photon_primary[keep], minlength=n))
    return out


def _run(case, n, batch=50):
    """The ``pyrite_bench.py`` production stack for ``n`` primaries."""
    from pyrite.materials import CATALOG
    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.events import (
        EVENT_CUTOFF,
        EVENT_EXIT_BOTTOM,
        EVENT_EXIT_SIDE,
        EVENT_EXIT_TOP,
        EVENT_HARD_RADIATIVE,
    )
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    target, E0_MeV = case.split("_")[0], float(case.split("_")[1][:-3])
    if target == "si":
        key, element, thickness = "silicon", "Si", 1.0e7
        density = CATALOG.crystal("silicon").composition[0][1]
    else:
        key, element, thickness = "W", "W", 1.0e6
        density = 19.3 * 6.02214076e23 / 183.84 / 1e24
    composition = [(element, density)]
    kwargs = dict(
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in resolve_layer_tables(composition)]],
        stopping_tables=[resolve_composition_table(key, composition).arrays()],
        atomic_electron_deflection="kawrakow",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        bremslib_tables=load_bremsstrahlung_tables([element]),
        energy_model="midpoint",
        straggling=False,
        E_cut_keV=10.0,
        crystal_width_mm=10.0,
        crystal_height_mm=10.0,
        transport_core="per-electron",
        max_steps=300_000,
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    codes = {EVENT_EXIT_BOTTOM: 0, EVENT_EXIT_TOP: 1, EVENT_CUTOFF: 2, EVENT_EXIT_SIDE: 3}
    fate, E_exit, cos_exit, ph_primary, ph_k = [], [], [], [], []
    for b, start in enumerate(range(0, n, batch)):
        r = simulate_trajectories(
            E0_keV=E0_MeV * 1e3,
            Ne=batch,
            composition=composition,
            thickness_ang=thickness,
            seed=97_439 + b,
            **kwargs,
        )
        eid, kind = np.asarray(r["electron_id"]), np.asarray(r["event_kind"])
        last = np.r_[np.flatnonzero(np.diff(eid)), eid.size - 1]
        assert last.size == batch
        code = np.array([codes[k] for k in kind[last]])
        fate.append(code)
        E_exit.append(np.where(code == 2, 0.0, np.asarray(r["E_end_keV"])[last] * 1e3))
        cos_exit.append(np.asarray(r["v_hat"])[last] @ np.asarray(r["initial_v_hat"])[0])
        hard = np.flatnonzero(kind == EVENT_HARD_RADIATIVE)
        ph_primary.append(start + eid[hard] - eid[0])
        ph_k.append(np.asarray(r["hard_radiative_k_eV"])[hard])
    return _observables(
        E0_MeV * 1e6, *(np.concatenate(x) for x in (fate, E_exit, cos_exit, ph_primary, ph_k))
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", ["si_30mev", "w_10mev"])
def test_production_transport_meets_geant4_gates(case):
    """Regression anchor: 300 primaries of the production stack pass every
    gate against the 100k-primary Geant4 reference (~4,800 steps/primary).
    PyRITE's standard error is floored at the reference spread scaled to 300
    primaries, so a rare outcome that happens not to occur still has width."""
    from pyrite.xsgen._errors import TableNotFoundError

    n = 300
    reference, _ = _load(case)
    try:
        result = _run(case, n)
    except TableNotFoundError as exc:
        pytest.skip(f"pinned transport tables are not installed: {exc}")
    scale = np.sqrt(reference["primaries"] / n)
    for name in GATED:
        mean, se = result[name]
        floored = (mean, max(se, scale * reference[name][1]))
        assert _gate(floored, reference[name]), (name, floored, reference[name])
