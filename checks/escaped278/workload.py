"""Frozen W transport and cone quadrature shared by independent #278 runs."""

import math

import numpy as np

from checks.escaped278.run import COMP, rules


def load_tables():
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    return (
        resolve_layer_tables(COMP),
        resolve_composition_table("W", COMP),
        load_bremsstrahlung_tables(["W"]),
    )


def transport(tables, energy, n, *, seed, offset=0):
    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
    from pyrite.montecarlo.transport.continuation import TransportContinuationConfig

    elastic, stopping, brem = tables
    return simulate_trajectories(
        E0_keV=energy,
        composition=COMP,
        thickness_ang=100 * 1e4,
        crystal_width_mm=10.0,
        crystal_height_mm=10.0,
        beam_dir=np.array([0.0, 0.0, 1.0]),
        E_cut_keV=10.0,
        energy_model="midpoint",
        max_dE_frac=0.05,
        max_steps=2000000,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in elastic]],
        stopping_tables=[stopping.arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        bremslib_tables=brem,
        radiative_moments="table",
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=False),
        per_electron_config=PerElectronTransportConfig(
            continuation=TransportContinuationConfig(rows_per_round=1024, device_staging=True)
        ),
        Ne=n,
        seed=seed,
        _electron_start=offset,
    )


def angular(center, order):
    from pyrite.perf.comparison import cone_rule
    from pyrite.perf.detector_scoring import detector_axis

    axis = detector_axis(center)
    if order == 0:
        return np.array([axis]), np.array([2 * math.pi * (1 - math.cos(math.radians(1)))])
    return cone_rule(axis, math.radians(1), order, 2 * order)


def recipe(tables, energy, order):
    from pyrite.montecarlo.spectrum.brem_prepared import _PreparedBremsLibBands

    edges = np.geomspace(1e4, energy * 1000.0, 7)
    bands = rules(edges, order)
    return _PreparedBremsLibBands(
        [g for g, _ in bands],
        [q for _, q in bands],
        composition=COMP,
        bremslib_tables=tables[2],
        cutoff_eV=1000.0,
        layers=[(0.0, 1e6, COMP)],
    )


def moments(scores):
    """Whole-primary moments; unresolved means never receive finite RSE."""
    total, squares, n = scores.sum(0), (scores * scores).sum(0), len(scores)
    return statistics(total, squares, n)


def statistics(total, squares, n):
    total, squares = np.asarray(total), np.asarray(squares)
    mean = total / max(n, 1)
    variance = np.maximum(0, (squares - total * total / max(n, 1)) / max(n - 1, 1))
    rse = np.divide(
        np.sqrt(variance / max(n, 1)),
        mean,
        out=np.full_like(mean, np.inf),
        where=(mean > 0) & (n > 1),
    )
    return dict(
        mean_per_primary=mean.tolist(),
        rse=[float(v) if np.isfinite(v) else None for v in rse],
        converged=n >= 10000 and bool(np.all(rse <= 0.02)),
    )
