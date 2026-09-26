"""Full-track BremsLib benchmark driver for issue #182.

Run explicitly through ``pyrite-dev test``; this check is outside the fast
test suite because each case transports thousands of primary electrons.
"""

import gzip
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from pyrite.montecarlo.transport.events import (
    EVENT_CUTOFF,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    check_segment_event_contract,
)
from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

_AVOGADRO = 6.02214076e23
_CASES = {
    "w_300kev": ("W", 19.3, 183.84, 100_000.0, 300.0),
    "si_300kev": ("Si", 2.329, 28.085, 1_000_000.0, 300.0),
    "w_800kev": ("W", 19.3, 183.84, 100_000.0, 800.0),
    "si_800kev": ("Si", 2.329, 28.085, 1_000_000.0, 800.0),
}


def _batch(seed, count, cutoff_eV, element, number_density_ang3, energy_keV, tables, **kwargs):
    """Transport one seeded batch and reduce it to per-primary and per-photon data."""
    result = simulate_trajectories(
        E0_keV=energy_keV,
        Ne=count,
        composition=[(element, number_density_ang3)],
        E_cut_keV=10.0,
        seed=seed,
        energy_model="midpoint",
        transport_lut_config=TransportLUTConfig(enabled=False),
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=cutoff_eV,
        bremslib_tables=tables,
        **kwargs,
    )
    check_segment_event_contract(result)
    photons = np.asarray(result["hard_radiative_k_eV"], dtype=float)
    payload = photons > 0.0
    before_eV = np.asarray(result["E_end_keV"], dtype=float)[payload] * 1000.0
    after_eV = before_eV - photons[payload]
    assert np.all(after_eV >= 0.0)
    rest_eV = 510_998.95
    p_before = np.sqrt(before_eV * (before_eV + 2.0 * rest_eV))
    p_after = np.sqrt(after_eV * (after_eV + 2.0 * rest_eV))
    recoil_residual = (
        (p_before - p_after)[:, None] * np.asarray(result["v_hat"])[payload]
        - photons[payload, None] * np.asarray(result["hard_radiative_direction"])[payload]
        - np.asarray(result["hard_radiative_target_momentum_eV_c"])[payload]
    )
    event_kind = np.asarray(result["event_kind"])
    terminal = np.isin(
        event_kind, [EVENT_EXIT_TOP, EVENT_EXIT_BOTTOM, EVENT_EXIT_SIDE, EVENT_CUTOFF]
    )
    assert int(np.count_nonzero(terminal)) == count
    terminal_energy_keV = np.asarray(result["E_end_keV"])[terminal] - photons[terminal] * 1e-3
    # Soft radiation joins the continuous stopping, so rows do not record it.
    # Re-evaluate the same BremsLib soft first moment at each row's mean energy;
    # the log-energy table interpolates it to better than 1e-5.
    grid_eV = np.geomspace(10_000.0 * (1.0 - 1e-9), energy_keV * 1000.0, 2001)
    soft_cm2 = [
        build_radiative_partition(tables[element], e, min(cutoff_eV, e)).soft_stopping_cs_eV_cm2
        for e in grid_eV
    ]
    row_mean_eV = 500.0 * (
        np.asarray(result["E_start_keV"], dtype=float)
        + np.asarray(result["E_end_keV"], dtype=float)
    )
    soft_per_ang = (
        number_density_ang3
        * 1e24
        * 1e-8
        * np.interp(np.log(row_mean_eV), np.log(grid_eV), soft_cm2)
    )
    soft_row_eV = soft_per_ang * np.asarray(result["L_ang"], dtype=float)
    electron_id = np.asarray(result["electron_id"], dtype=np.int64)
    cos_photon = np.einsum(
        "ij,ij->i",
        np.asarray(result["v_hat"])[payload],
        np.asarray(result["hard_radiative_direction"])[payload],
    )
    return {
        "n_transmitted": int(result["n_transmitted"]),
        "n_backscattered": int(result["n_backscattered"]),
        "n_cutoff_stopped": int(result["n_cutoff_stopped"]),
        "max_recoil_residual": float(np.max(np.abs(recoil_residual), initial=0.0)),
        "debit_keV": float(count * energy_keV - terminal_energy_keV.sum()),
        "soft_primary_eV": np.bincount(electron_id, weights=soft_row_eV, minlength=count),
        "hard_primary_eV": np.bincount(electron_id, weights=photons, minlength=count),
        "photons": np.column_stack([electron_id[payload], before_eV, photons[payload], cos_photon]),
        # Track length [Angstrom] by row-mean energy in 1 keV bins, as the
        # Geant4 reference scores its primary.
        "track_length_ang": np.histogram(
            row_mean_eV,
            bins=np.arange(0.0, energy_keV * 1000.0 + 1000.0, 1000.0),
            weights=np.asarray(result["L_ang"], dtype=float),
        )[0],
    }


def test_full_track_bremslib_benchmark():
    case_name = os.environ.get("PYRITE_BENCH_CASE", "w_300kev")
    element, density_g_cm3, atomic_weight_g_mol, thickness_ang, energy_keV = _CASES[case_name]
    count = int(os.environ.get("PYRITE_BENCH_NE", "1000"))
    # Batches bound the per-call segment buffer; batch b uses seed 12345 + b,
    # so a single batch reproduces the unbatched runs.
    batch_size = int(os.environ.get("PYRITE_BENCH_BATCH", "10000"))
    cutoff_eV = float(os.environ.get("PYRITE_BENCH_CUTOFF_EV", "1000"))
    core = os.environ.get("PYRITE_BENCH_CORE", "per-electron")
    elastic_model = os.environ.get("PYRITE_BENCH_ELASTIC", "mott")
    assert elastic_model in ("mott", "sr")
    number_density_ang3 = density_g_cm3 * _AVOGADRO / atomic_weight_g_mol / 1e24
    tables = load_bremsstrahlung_tables([element])
    assert element in tables

    t0 = perf_counter()
    batches = []
    for index, start in enumerate(range(0, count, batch_size)):
        size = min(batch_size, count - start)
        batch = _batch(
            12345 + index,
            size,
            cutoff_eV,
            element,
            number_density_ang3,
            energy_keV,
            tables,
            thickness_ang=thickness_ang,
            transport_core=core,
            elastic_model=elastic_model,
        )
        batch["photons"][:, 0] += start
        batches.append(batch)
    elapsed_s = perf_counter() - t0

    max_recoil_residual = max(b["max_recoil_residual"] for b in batches)
    assert max_recoil_residual < 1e-5
    photons = np.concatenate([b["photons"] for b in batches])
    soft_primary_eV = np.concatenate([b["soft_primary_eV"] for b in batches])
    radiative_primary_eV = soft_primary_eV + np.concatenate([b["hard_primary_eV"] for b in batches])
    emitted = photons[:, 2]
    bins_eV = np.arange(0.0, energy_keV * 1000.0 + 10_000.0, 10_000.0)
    hist = np.histogram(emitted, bins=bins_eV)[0]
    summary = {
        "case": case_name,
        "pyrite_seed": 12345,
        "batch_size": batch_size,
        "transport_core": core,
        "elastic_model": elastic_model,
        "Ne": count,
        "E0_keV": energy_keV,
        "E_cut_keV": 10.0,
        "radiative_cutoff_eV": cutoff_eV,
        "element": element,
        "density_g_cm3": density_g_cm3,
        "number_density_ang3": number_density_ang3,
        "thickness_ang": thickness_ang,
        "bremslib_table_key": tables[element].key,
        "bremslib_table_digest": tables[element].digest,
        "n_transmitted": sum(b["n_transmitted"] for b in batches),
        "n_backscattered": sum(b["n_backscattered"] for b in batches),
        "n_cutoff_stopped": sum(b["n_cutoff_stopped"] for b in batches),
        "n_hard_photons": int(emitted.size),
        "n_hard_photons_ge_10kev": int(np.count_nonzero(emitted >= 10_000.0)),
        "hard_photon_energy_sum_eV": float(emitted.sum()),
        "max_hard_recoil_residual_eV_c": max_recoil_residual,
        "primary_energy_debit_sum_keV": sum(b["debit_keV"] for b in batches),
        "soft_radiative_estimate_sum_eV": float(soft_primary_eV.sum()),
        "radiative_estimate_sum_eV": float(radiative_primary_eV.sum()),
        "radiative_estimate_sumsq_eV2": float(np.sum(radiative_primary_eV**2)),
        "track_length_ang_per_keV_bin": np.sum(
            [b["track_length_ang"] for b in batches], axis=0
        ).tolist(),
        "photon_bin_edges_eV": bins_eV.tolist(),
        "photon_bin_counts": hist.tolist(),
        "runtime_s": elapsed_s,
    }
    print("PYRITE_BENCHMARK=" + json.dumps(summary, sort_keys=True))
    # Per hard photon: primary id, pre-emission energy, photon energy and
    # cosine to the pre-emission electron direction.
    summary["hard_photons"] = photons.tolist()
    encoded = json.dumps(summary, sort_keys=True)
    output = os.environ.get("PYRITE_BENCH_OUTPUT")
    if output:
        if output.endswith(".gz"):
            with gzip.open(output, "wt") as handle:
                handle.write(encoded + "\n")
        else:
            Path(output).write_text(encoded + "\n")
