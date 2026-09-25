"""Opt-in shell soft/hard inelastic transport (``inelastic_model``).

Validation: shell-soft-hard-transport
"""

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    resolve_transport_core,
    simulate_trajectories,
)
from pyrite.montecarlo.transport.events import SegmentEvent, check_segment_event_contract
from pyrite.montecarlo.transport.hard_inelastic import (
    _hard_primary_cosine,
    _sample_hard_transfer_eV,
    _soft_loss_sample_keV,
    hard_stream_keys,
)
from pyrite.montecarlo.transport.kinematics import stream_keys
from pyrite.montecarlo.transport.shell_partition import catalog_shell_partition
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators
from pyrite.montecarlo.transport.shell_sampling import (
    BRANCHES,
    sample_shell_hard_collision,
)
from pyrite.montecarlo.transport.shell_transport import (
    build_shell_inelastic_tables,
    hard_event_energy_accounting,
)
from pyrite.montecarlo.transport.stopping import prepare_sbethe_stopping_table
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

HARD = int(SegmentEvent.HARD_INELASTIC)


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _table(key):
    return resolve_catalog_table(key).arrays()


def _composition(key):
    if key in CATALOG.crystals:
        return CATALOG.crystal(key).composition
    return CATALOG.media[key].composition


def _run(key="silicon", *, cutoff=50.0, Ne=40, E0=20.0, E_cut=10.0, **kw):
    kw.setdefault("energy_model", "midpoint")
    return simulate_trajectories(
        E0,
        Ne,
        kw.pop("thickness_ang", 2.0e4),
        composition=_composition(key),
        E_cut_keV=E_cut,
        seed=kw.pop("seed", 11),
        stopping_tables=[_table(key)],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=cutoff,
        inelastic_materials=[key],
        **kw,
    )


@pytest.mark.parametrize("key", ("silicon", "mos2"))
@pytest.mark.parametrize("energy_eV", (6.0e3, 3.0e4))
def test_numba_transfer_and_recoil_match_host_sampler(key, energy_eV):
    material = catalog_shell_oscillators(key)
    part = catalog_shell_partition(key, energy_eV, 50.0)
    probabilities = part.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    checked = 0
    for flat in np.flatnonzero(probabilities > 0.0):
        index, branch = divmod(int(flat), 3)
        osc = material.oscillators[index]
        # A channel uniform at the middle of this channel's probability bin.
        u_channel = (cumulative[flat] - 0.5 * probabilities[flat]) / cumulative[-1]
        for u_loss, u_recoil in ((0.0, 0.0), (0.37, 0.61), (0.93, 0.08)):
            host = sample_shell_hard_collision(material, part, u_channel, u_loss, u_recoil, 0.25)
            assert host.loss.oscillator_index == index and host.loss.branch == BRANCHES[branch]
            w = _sample_hard_transfer_eV(
                energy_eV,
                osc.ionization_energy_eV,
                osc.resonance_energy_eV,
                branch,
                50.0,
                u_loss,
            )
            assert w == pytest.approx(host.loss.transfer_eV, rel=1e-10, abs=1e-9)
            cosine = _hard_primary_cosine(
                energy_eV, osc.ionization_energy_eV, osc.resonance_energy_eV, branch, w, u_recoil
            )
            assert cosine == pytest.approx(host.cos_primary, rel=1e-9, abs=1e-12)
            checked += 1
    assert checked >= 6


@pytest.mark.parametrize(
    ("mean", "variance"),
    [(10.0, 0.5), (10.0, 20.0), (10.0, 60.0)],
    ids=["truncated-gaussian", "uniform", "delta-plus-uniform"],
)
def test_soft_loss_sampler_reproduces_both_moments(mean, variance):
    keys = stream_keys(5, 40000)
    draws = np.array([_soft_loss_sample_keV(mean, variance, k, np.uint64(0))[0] for k in keys])
    assert np.all(draws >= 0.0)
    n = draws.size
    assert draws.mean() == pytest.approx(mean, abs=5.0 * np.sqrt(variance / n))
    assert draws.var() == pytest.approx(variance, rel=0.03)


def test_soft_loss_sampler_limits():
    key = np.uint64(123)
    assert _soft_loss_sample_keV(0.0, 1.0, key, np.uint64(0)) == (0.0, np.uint64(0))
    assert _soft_loss_sample_keV(2.5, 0.0, key, np.uint64(0)) == (2.5, np.uint64(0))


def test_tables_close_to_the_transport_stopping_at_every_node():
    arrays = _table("silicon")
    prepared = prepare_sbethe_stopping_table(arrays)
    tables = build_shell_inelastic_tables(["silicon"], 50.0, [prepared], 5.0, 30.0)
    log_e, soft_log_s = tables.soft_stopping_tables[0]
    energies_eV = np.exp(log_e) * 1e3
    lo = int(np.flatnonzero(np.isclose(prepared[0], log_e[0], rtol=0.0, atol=1e-15))[0])
    full = np.exp(prepared[1][lo : lo + log_e.size]) * 1e3  # eV/Angstrom
    soft = np.exp(soft_log_s) * 1e3
    n = tables.n_channels[0]
    for j, energy in enumerate(energies_eV):
        part = catalog_shell_partition("silicon", float(energy), 50.0)
        total = part.soft.total[1] + part.hard.total[1]
        hard_mean = part.hard.total[1] / total * full[j]
        assert soft[j] + hard_mean == pytest.approx(full[j], rel=1e-12)
        assert tables.hard_rate_per_ang[0, j] == pytest.approx(
            part.hard.total[0] / total * full[j], rel=1e-12
        )
        assert tables.channel_rate_per_ang[0, :n, j].sum() == pytest.approx(
            tables.hard_rate_per_ang[0, j], rel=1e-12
        )
    assert np.exp(log_e[0]) <= 5.0 and np.exp(log_e[-1]) >= 30.0


@pytest.mark.parametrize(
    ("core", "lut"), [("lockstep", True), ("lockstep", False), ("per-electron", True)]
)
def test_cutoff_above_every_channel_is_bitwise_the_continuous_transport(core, lut):
    common = dict(
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=lut),
        energy_model="midpoint",
    )
    legacy = simulate_trajectories(
        20.0,
        30,
        2.0e4,
        composition=CATALOG.crystal("silicon").composition,
        E_cut_keV=10.0,
        seed=3,
        stopping_tables=[_table("silicon")],
        **common,
    )
    shell = _run(cutoff=1.0e9, Ne=30, seed=3, **common)
    assert not np.any(shell["event_kind"] == HARD)
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "E_end_keV", "t_end_ang", "event_kind"):
        np.testing.assert_array_equal(shell[key], legacy[key])
    assert np.all(shell["hard_channel"] == -1)


@pytest.mark.parametrize(
    ("core", "lut", "straggling"),
    [
        ("lockstep", True, False),
        ("lockstep", False, True),
        ("per-electron", True, True),
        ("per-electron", False, False),
    ],
)
def test_hard_events_honour_contract_and_energy_bookkeeping(core, lut, straggling):
    result = _run(
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=lut),
        straggling=straggling,
        max_dE_frac=0.05,
    )
    check_segment_event_contract(result)
    kind = result["event_kind"]
    hard = kind == HARD
    assert hard.sum() > 50
    order = np.lexsort((result["substep_id"], result["flight_id"], result["electron_id"]))
    rows = np.flatnonzero(hard[order])
    a, b = order[rows], order[rows + 1]
    assert np.all(result["electron_id"][a] == result["electron_id"][b])
    drop = result["E_end_keV"][a] - result["E_start_keV"][b]
    np.testing.assert_allclose(drop, result["hard_W_keV"][a], rtol=1e-12, atol=1e-12)
    assert np.all(result["hard_W_keV"][a] > 0.050)
    # A collision that would leave the primary below cutoff ends a CUTOFF row.
    absorbed = (kind == int(SegmentEvent.CUTOFF)) & (result["hard_channel"] >= 0)
    assert np.all(result["E_end_keV"][absorbed] - result["hard_W_keV"][absorbed] <= 10.0)
    assert np.all(result["hard_channel"][~hard & ~absorbed] == -1)
    accounting = hard_event_energy_accounting(result, production_threshold_eV=100.0)
    np.testing.assert_allclose(
        accounting["deposit_keV"] + accounting["secondary_keV"] + accounting["binding_keV"],
        accounting["transfer_keV"],
        rtol=1e-12,
    )
    assert accounting["vacancy"].sum() >= 0
    assert result["inelastic"]["cutoff_eV"] == 50.0


def test_primary_energy_is_conserved_per_electron():
    result = _run(Ne=20, thickness_ang=2.0e5)
    eid = result["electron_id"]
    soft = result["E_start_keV"] - result["E_end_keV"]
    hard = result["hard_W_keV"]
    order = np.lexsort((result["substep_id"], result["flight_id"], eid))
    last = order[np.r_[eid[order][1:] != eid[order][:-1], True]]
    lost = np.bincount(eid, weights=soft + hard, minlength=20)
    # Every electron stops (thick slab); a collision-absorbed primary keeps
    # E_end - W >= 0 as its local deposit, the rest end at the cutoff.
    residual = result["E_end_keV"][last] - hard[last]
    np.testing.assert_allclose(lost[eid[last]] + residual, 20.0, rtol=1e-12)


def test_same_seed_reproduces_and_hard_stream_is_disjoint():
    first, second = _run(seed=5), _run(seed=5)
    for key in ("r_mid", "E_end_keV", "hard_W_keV", "hard_channel"):
        np.testing.assert_array_equal(first[key], second[key])
    keys, hard = stream_keys(5, 64), hard_stream_keys(5, 64)
    assert not np.intersect1d(keys, hard).size


def test_mode_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="conduction-band resonance"):
        _run(cutoff=16.7)
    with pytest.raises(ValueError, match="conduction-band resonance"):
        _run("sio2", cutoff=20.0)
    with pytest.raises(ValueError, match="energy_model='midpoint'"):
        _run(energy_model="frozen")
    with pytest.raises(NotImplementedError, match="CUDA"):
        _run(transport_core="cuda")
    with pytest.raises(ValueError, match="stopping_tables"):
        simulate_trajectories(
            20.0,
            4,
            2.0e4,
            composition=CATALOG.crystal("silicon").composition,
            energy_model="midpoint",
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=50.0,
            inelastic_materials=["silicon"],
        )
    with pytest.raises(ValueError, match="require inelastic_model"):
        simulate_trajectories(
            20.0, 4, 2.0e4, element="Si", n_atoms_per_ang3=0.05, inelastic_cutoff_eV=50.0
        )
    with pytest.raises(ValueError, match="inelastic_model must be"):
        simulate_trajectories(
            20.0, 4, 2.0e4, element="Si", n_atoms_per_ang3=0.05, inelastic_model="x"
        )


def test_auto_core_stays_on_the_cpu(monkeypatch):
    from pyrite.montecarlo.transport import batching

    monkeypatch.setattr(batching, "_cuda_transport_available", lambda: True)
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE", raising=False)
    assert resolve_transport_core("auto", 10**6) == "cuda"
    assert resolve_transport_core("auto", 10**6, cpu_only=True) == "lockstep"
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "cuda")
    with pytest.raises(NotImplementedError, match="CUDA"):
        resolve_transport_core("auto", 10**6, cpu_only=True)
