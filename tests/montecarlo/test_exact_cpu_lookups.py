"""Output-identical CPU lookup shortcuts: lazy midpoint cutoff, indexed moments (#841)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import TransportLUTConfig, cores, host_pass, simulate_trajectories
from pyrite.montecarlo.transport._jit_cutoff_bound import (
    add_soft_moment_bound,
    lut_cutoff_bound,
    sbethe_cutoff_bound,
    widen_cutoff_bound,
)
from pyrite.montecarlo.transport._jit_moment_lookup import (
    build_moment_guide,
    table_moments,
    transport_layer_moments,
)
from pyrite.montecarlo.transport.hard_radiative import radiative_core_args
from pyrite.montecarlo.transport.stopping import (
    _dEds_sbethe_packed_scalar,
    pack_sbethe_stopping_tables,
    prepare_sbethe_stopping_tables,
)
from tests.helpers.bremslib import synthetic_bremslib_arrays

SI = CATALOG.crystal("silicon").composition
W = CATALOG.crystal("tungsten").composition


def _bremslib(atomic_number):
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=atomic_number)
    return replace(
        table,
        incident_energy_keV=np.r_[10.0, table.incident_energy_keV[1:-1], 200.0],
        scaled_sdcs_mb=table.scaled_sdcs_mb * 1e5,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 1e5,
    )


BREMSLIB = {"Si": _bremslib(14), "W": _bremslib(74)}


def _sbethe(key):
    from pyrite.montecarlo import shell_configuration
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    return resolve_catalog_table(key).arrays()


def _packed_two_layer():
    args = ([SI, W], BREMSLIB, 1000.0, np.array([10.0]), np.array([150.0]), 7, 1)
    return radiative_core_args(*args, moment_mode="table")


def _query_energies(energy):
    """Dense log sweep plus every node, both 1-ulp neighbours and endpoints."""
    sweep = np.geomspace(energy[0], energy[-1], 20_001)
    nodes = np.r_[energy, np.nextafter(energy, -np.inf), np.nextafter(energy, np.inf)]
    nodes = nodes[(nodes >= energy[0]) & (nodes <= energy[-1])]
    midpoints = np.sqrt(energy[:-1] * energy[1:])
    return np.r_[sweep, nodes, midpoints]


def _layer_nodes(packed, layer):
    n_layers = packed[2].size
    plasma = packed[-1]
    start = int(plasma[n_layers + layer])
    count = int(plasma[2 * n_layers + layer])
    return plasma[3 * n_layers + start : 3 * n_layers + start + count]


def test_indexed_moment_search_is_bitwise_the_bisection_on_dense_sweeps():
    packed = _packed_two_layer()
    plasma = packed[-1]
    guide = build_moment_guide(packed[2], plasma)
    assert guide[5]
    scratch = np.empty(packed[4].shape[1])
    for layer in range(2):
        for energy_eV in _query_energies(_layer_nodes(packed, layer)):
            expected = transport_layer_moments(
                *packed[2:-1], layer, energy_eV, 1000.0, scratch, plasma
            )
            actual = table_moments(plasma, guide, layer, energy_eV)
            assert np.array(actual).tobytes() == np.array(expected).tobytes(), energy_eV
    for bad in [np.nextafter(_layer_nodes(packed, 1)[0], -np.inf), np.inf, np.nan]:
        with pytest.raises(ValueError, match="outside prepared"):
            table_moments(plasma, guide, 1, bad)


def _synthetic_buffer(rows):
    """Packed buffer with arbitrary nondecreasing log rows (ties and clusters)."""
    n = len(rows)
    counts = np.array([r.size for r in rows], dtype=float)
    starts = np.r_[0.0, np.cumsum(counts[:-1])]
    energies = np.concatenate([np.exp(r) for r in rows])
    logs = np.concatenate(rows)
    moments = np.random.default_rng(3).uniform(0.0, 1.0, 3 * logs.size)
    return np.r_[np.zeros(n), starts, counts, energies, logs, moments]


def test_guided_walk_reproduces_bisection_on_clustered_and_tied_rows():
    # Two-layer quadrature arguments only fix the call types; table mode ignores them.
    quadrature = _packed_two_layer()[2:-1]
    rng = np.random.default_rng(11)
    clustered = np.sort(np.r_[np.log(1e4), rng.normal(10.0, 1e-3, 400), np.log(1e6)])
    tied = np.r_[np.log(1e4), np.repeat(np.log(5e4), 5), np.log(2e5), np.log(1e6)]
    two = np.log(np.array([1e4, 1e6]))
    for rows in [(clustered, tied), (two, clustered)]:
        plasma = _synthetic_buffer(rows)
        guide = build_moment_guide(quadrature[0], plasma)
        assert guide[5]
        for layer, row in enumerate(rows):
            x = np.r_[np.linspace(row[0], row[-1], 5001), row, rng.uniform(row[0], row[-1], 2000)]
            for energy_eV in np.clip(np.exp(x), np.exp(row[0]), np.exp(row[-1])):
                expected = transport_layer_moments(
                    *quadrature, layer, energy_eV, 1.0, np.empty(1), plasma
                )
                actual = table_moments(plasma, guide, layer, energy_eV)
                assert np.array(actual).tobytes() == np.array(expected).tobytes()
    decreasing = _synthetic_buffer([np.r_[two[0], 11.0, 10.0, two[1]], two])
    refused = build_moment_guide(quadrature[0], decreasing)
    assert not refused[5]
    for energy_eV in np.exp(np.linspace(two[0], two[1], 101)):
        expected = transport_layer_moments(*quadrature, 0, energy_eV, 1.0, np.empty(1), decreasing)
        actual = table_moments(decreasing, refused, 0, energy_eV)
        assert np.array(actual).tobytes() == np.array(expected).tobytes()
    assert not build_moment_guide(quadrature[0], np.zeros(2))[5]  # exact quadrature buffer


def test_cutoff_bounds_dominate_every_evaluated_rate_and_refuse_unproven_tables():
    tables = prepare_sbethe_stopping_tables([_sbethe("silicon"), _sbethe("tungsten")], 2)
    _, counts, log_energy, log_stopping = pack_sbethe_stopping_tables(tables, 2)
    bound = np.zeros(2)
    assert sbethe_cutoff_bound(bound, True, counts, log_stopping)
    packed = _packed_two_layer()
    cutoffs = np.array([12.0, 10.0])
    assert add_soft_moment_bound(True, bound, packed[2], packed[-1], cutoffs)
    assert widen_cutoff_bound(True, bound)
    guide = build_moment_guide(packed[2], packed[-1])
    for layer in range(2):
        for energy in np.geomspace(10.0, 150.0, 4001):
            soft, _ = table_moments(packed[-1], guide, layer, energy * 1e3)
            rate = _dEds_sbethe_packed_scalar(
                log_energy, log_stopping, layer, counts[layer], energy
            )
            rate -= soft * 1e-3
            assert rate < 0.0 and -rate <= bound[layer]
    # Coverage below the smallest cutoff and exact quadrature refuse the skip.
    nel, plasma = packed[2], packed[-1]
    assert not add_soft_moment_bound(True, bound.copy(), nel, plasma, np.array([9.0]))
    assert not add_soft_moment_bound(True, bound.copy(), nel, np.zeros(2), cutoffs)
    assert not add_soft_moment_bound(False, bound.copy(), nel, plasma, cutoffs)
    assert not add_soft_moment_bound(True, bound.copy(), nel, plasma, np.zeros(0))
    assert not sbethe_cutoff_bound(np.zeros(1), False, np.array([1]), np.array([[-1.0]]))
    assert not sbethe_cutoff_bound(np.zeros(1), True, np.array([1]), np.array([[-800.0]]))
    assert not widen_cutoff_bound(True, np.array([0.0]))
    lut = np.zeros(1)
    assert not lut_cutoff_bound(lut, np.array([[-1.0, 0.0]]))
    assert not lut_cutoff_bound(lut, np.array([[-1.0, -1e-9]]))
    assert lut_cutoff_bound(lut, np.array([[-1.0, -2.0]])) and lut[0] == 2.0


# ---- trajectory equality against cores compiled without the shortcuts ----------

_EXACT = {
    (False, False, False): "_transport_core_ungrooved",
    (False, True, False): "_transport_core_ungrooved_inelastic",
    (False, False, True): "_transport_core_ungrooved_radiative",
    (False, True, True): "_transport_core_ungrooved_inelastic_radiative",
    (True, False, False): "_transport_core_ungrooved_perelectron",
    (True, True, False): "_transport_core_ungrooved_perelectron_inelastic",
    (True, False, True): "_transport_core_ungrooved_perelectron_radiative",
    (True, True, True): "_transport_core_ungrooved_perelectron_inelastic_radiative",
}
_LUT = {
    (False, False): "_transport_core_ungrooved_lut",
    (False, True): "_transport_core_ungrooved_lut_inelastic",
    (True, False): "_transport_core_ungrooved_perelectron_lut",
    (True, True): "_transport_core_ungrooved_perelectron_lut_inelastic",
}


def _reference_core(monkeypatch, per_electron, inelastic, radiative, lut):
    flags = dict(per_electron=per_electron, inelastic=inelastic, radiative=radiative)
    reference = cores.make_cpu_transport_core(
        **flags, lut=lut, lazy_cutoff=False, indexed_moments=False
    )
    if lut:
        monkeypatch.setattr(host_pass, _LUT[(per_electron, inelastic)], reference)
    else:
        monkeypatch.setattr(cores, _EXACT[(per_electron, inelastic, radiative)], reference)


def _assert_bitwise(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_bitwise(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _assert_bitwise(a, b, f"{path}[{i}]")
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert actual.tobytes() == expected.tobytes(), path
    elif isinstance(expected, float):
        assert np.array(actual).tobytes() == np.array(expected).tobytes(), path
    else:
        assert actual == expected, path


def _case(core, inelastic, radiative, layered, energy_model, lut):
    kwargs = dict(
        E0_keV=60.0,
        Ne=24,
        E_cut_keV=10.0,
        seed=19,
        thickness_ang=300_000.0,
        energy_model=energy_model,
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=lut),
        # The LUT has no ELSEPA counterpart; screened Rutherford selects it.
        elastic_model="sr" if lut else "elsepa",
        radiative_model="bremslib-soft-hard" if radiative else "uncoupled",
    )
    if layered:
        kwargs["layers"] = [(0.0, 3_000.0, W), (3_000.0, 300_000.0, SI)]
        kwargs["stopping_tables"] = [_sbethe("tungsten"), _sbethe("silicon")]
        materials = ["tungsten", "silicon"]
    else:
        kwargs["composition"] = SI
        kwargs["stopping_tables"] = [_sbethe("silicon")]
        materials = ["silicon"]
    if inelastic:
        kwargs |= dict(
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=50.0,
            inelastic_materials=materials,
        )
    if radiative:
        kwargs |= dict(radiative_cutoff_eV=1_000.0, bremslib_tables=BREMSLIB)
    return kwargs


CASES = [
    # (core, inelastic, radiative, layered, energy_model, lut)
    ("per-electron", False, True, False, "midpoint", False),
    ("per-electron", True, True, True, "midpoint", False),
    ("per-electron", False, False, True, "midpoint", False),
    ("per-electron", True, False, False, "midpoint", False),
    ("per-electron", False, False, False, "frozen", False),
    ("lockstep", False, True, True, "midpoint", False),
    ("lockstep", True, False, False, "midpoint", False),
    ("per-electron", False, False, True, "midpoint", True),
    ("lockstep", False, False, False, "midpoint", True),
]


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: "-".join(map(str, case)))
def test_trajectories_are_bitwise_identical_without_the_shortcuts(case, monkeypatch):
    kwargs = _case(*case)
    candidate = simulate_trajectories(**kwargs)
    core, inelastic, radiative, _, _, lut = case
    _reference_core(monkeypatch, core == "per-electron", inelastic, radiative, lut)
    reference = simulate_trajectories(**kwargs)
    # Thick targets: the truncation branch the skip must not mask is exercised.
    assert reference["n_cutoff_stopped"] > 0
    _assert_bitwise(candidate, reference)
