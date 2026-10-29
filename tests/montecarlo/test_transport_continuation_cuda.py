"""Bounded transport continuation on the exact CUDA kernel (#789, #797).

Continuation changes only the schedule: each history is resumed from the
kernel's own loop-carried state, so every returned array must equal the
full-history run of the same kernel *exactly*, across row quanta, active
populations and refill orders. No tolerance is relaxed. Scheduling without a
device is covered by ``test_transport_continuation.py``. These tests need a
CUDA device and run through ``pyrite remote``.

Validation: gpu-transport-core
"""

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    batching,
    check_segment_event_contract,
    simulate_trajectories,
)
from pyrite.montecarlo.transport import continuation as continuation_module
from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
from pyrite.montecarlo.transport.continuation import (
    TransportContinuationConfig,
    _drive_continuation_rounds,
)
from pyrite.montecarlo.transport.events import (
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    TransportStepLimitError,
)
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table
from tests.helpers.bremslib import synthetic_bremslib_arrays

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

NO_LUT = TransportLUTConfig(enabled=False)
REFERENCE = PerElectronTransportConfig(seg_capacity=16, probe_electrons=0)


def _continued(rows_per_round, active_histories=None, **kwargs):
    cont = TransportContinuationConfig(
        rows_per_round=rows_per_round, active_histories=active_histories
    )
    return replace(REFERENCE, continuation=cont, **kwargs)


def _table(scale, atomic_number=6):
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=atomic_number)
    return replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * scale,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * scale,
    )


def _radiative(config, *, Ne=300, seed=42, **kw):
    return simulate_trajectories(
        E0_keV=60.0,
        Ne=Ne,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=seed,
        energy_model="midpoint",
        max_dE_frac=0.05,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": _table(1e5)},
        transport_core="cuda",
        transport_lut_config=NO_LUT,
        per_electron_config=config,
        **kw,
    )


def _layered(config, *, energy_model="midpoint", **kw):
    """Uncoupled two-layer prism: layer boundaries, side, top and bottom exits."""
    return simulate_trajectories(
        E0_keV=40.0,
        Ne=400,
        thickness_ang=5.0e4,
        layers=[(0.0, 1.0e4, [("C", 0.176)]), (1.0e4, 5.0e4, [("W", 0.0632)])],
        elastic_model="sr",
        crystal_width_mm=2.0e-3,
        crystal_height_mm=2.0e-3,
        E_cut_keV=5.0,
        seed=7,
        energy_model=energy_model,
        radiative_model="uncoupled",
        transport_core="cuda",
        transport_lut_config=NO_LUT,
        per_electron_config=config,
        **kw,
    )


def _host(value):
    return cupy.asnumpy(value) if isinstance(value, cupy.ndarray) else value


def _assert_identical(reference, candidate):
    assert reference.keys() == candidate.keys()
    for key, expected in reference.items():
        expected, actual = _host(expected), _host(candidate[key])
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype, key
            assert actual.shape == expected.shape, key
            # Bitwise, NaN-aware: the same kernel arithmetic must be replayed.
            assert np.array_equal(
                expected.view(np.uint8), np.ascontiguousarray(actual).view(np.uint8)
            ), key
        elif isinstance(expected, dict):
            _assert_identical(expected, actual)
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected), key
            _assert_identical(dict(enumerate(expected)), dict(enumerate(actual)))
        else:
            assert actual == expected, key


@pytest.fixture(scope="module")
def radiative_reference():
    return _radiative(REFERENCE)


@pytest.mark.parametrize(
    ("rows_per_round", "active", "staging"),
    [(1, None, False), (7, None, False), (7, 37, True), (64, 5, False), (4096, None, True)],
)
def test_coupled_radiative_rows_match_full_history(
    radiative_reference, rows_per_round, active, staging
):
    config = _continued(rows_per_round, active)
    config = replace(config, continuation=replace(config.continuation, device_staging=staging))
    result = _radiative(config)
    _assert_identical(radiative_reference, result)
    assert np.count_nonzero(result["event_kind"] == EVENT_HARD_RADIATIVE) > 0
    check_segment_event_contract(result)


def test_refill_order_does_not_change_rows(radiative_reference, monkeypatch):
    order = np.random.default_rng(11).permutation(300).astype(np.int32)
    monkeypatch.setattr(
        batching,
        "_drive_continuation_rounds",
        partial(_drive_continuation_rounds, history_order=order),
    )
    _assert_identical(radiative_reference, _radiative(_continued(5, 23)))


def test_budget_bounds_population_and_matches(radiative_reference):
    records = []
    cont = TransportContinuationConfig(rows_per_round=8, on_round=records.append)
    budget = 50 * (8 * (2 * 125 + 24) + 13)
    config = replace(REFERENCE, scratch_budget_bytes=budget, continuation=cont)
    _assert_identical(radiative_reference, _radiative(config))
    records = [r for r in records if r["phase"] == "round"]
    assert max(r["launched"] for r in records) == 50
    assert all(r["scratch_bytes"] <= budget for r in records)
    assert sum(r["rows"] for r in records) == radiative_reference["electron_id"].size


@pytest.mark.parametrize("energy_model", ["midpoint", "frozen"])
def test_layered_prism_exits_match_full_history(energy_model):
    reference = _layered(REFERENCE, energy_model=energy_model)
    assert reference["n_side_exited"] > 0
    assert reference["n_backscattered"] > 0
    assert np.unique(reference["layer"]).size == 2
    _assert_identical(reference, _layered(_continued(3, 41), energy_model=energy_model))


def test_device_resident_rows_match_full_history():
    reference = _layered(REFERENCE, keep_segments_on_device=True)
    candidate = _layered(_continued(9, 64), keep_segments_on_device=True)
    _assert_identical(reference, candidate)


def test_step_limit_is_reported_identically():
    with pytest.raises(TransportStepLimitError) as expected:
        _layered(REFERENCE, max_steps=6)
    with pytest.raises(TransportStepLimitError) as actual:
        _layered(_continued(4, 50), max_steps=6)
    assert str(actual.value) == str(expected.value)


def _shell(config, *, secondaries=False, straggling=False, radiative=False, **kw):
    """Silicon under shell soft/hard inelastic transport (covered material)."""
    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    if secondaries:
        kw["secondary_threshold_eV"] = 10_000.0
    if radiative:
        kw |= {
            "radiative_model": "bremslib-soft-hard",
            "radiative_cutoff_eV": 1_000.0,
            "bremslib_tables": {"Si": _table(1e5, atomic_number=14)},
        }
    return simulate_trajectories(
        30.0,
        kw.pop("Ne", 200),
        kw.pop("thickness_ang", 2.0e4),
        composition=CATALOG.crystal("silicon").composition,
        E_cut_keV=10.0,
        seed=kw.pop("seed", 11),
        energy_model="midpoint",
        max_dE_frac=0.05,
        elastic_model="sr",
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=["silicon"],
        straggling=straggling,
        transport_core="cuda",
        transport_lut_config=NO_LUT,
        per_electron_config=config,
        **kw,
    )


SHELL_MODES = [
    pytest.param({}, id="shell"),
    pytest.param({"secondaries": True}, id="shell-secondaries"),
    pytest.param({"straggling": True}, id="shell-straggling"),
    pytest.param({"straggling": True, "radiative": True}, id="shell-straggling-radiative"),
    pytest.param({"secondaries": True, "straggling": True, "radiative": True}, id="shell-all"),
]


@pytest.mark.parametrize("mode", SHELL_MODES)
@pytest.mark.parametrize(
    ("rows_per_round", "active", "staging"),
    [(1, None, False), (7, 37, True), (4096, None, False)],
)
def test_shell_rows_match_full_history(mode, rows_per_round, active, staging):
    reference = _shell(REFERENCE, **mode)
    config = _continued(rows_per_round, active)
    config = replace(config, continuation=replace(config.continuation, device_staging=staging))
    result = _shell(config, **mode)
    _assert_identical(reference, result)
    assert np.count_nonzero(reference["event_kind"] == EVENT_HARD_INELASTIC) > 0
    if mode.get("radiative"):
        assert np.count_nonzero(reference["event_kind"] == EVENT_HARD_RADIATIVE) > 0
    if not mode.get("secondaries"):
        # Cascade output appends secondary histories; the row contract is per pass.
        check_segment_event_contract(result)


def test_shell_refill_order_and_small_population(monkeypatch):
    mode = {"secondaries": True, "straggling": True}
    reference = _shell(REFERENCE, **mode)

    def permuted(*args, **kwargs):
        # Every transport pass, the secondary cascade's included, gets its own order.
        order = np.random.default_rng(5).permutation(args[3]).astype(np.int32)
        return _drive_continuation_rounds(*args, **kwargs, history_order=order)

    monkeypatch.setattr(batching, "_drive_continuation_rounds", permuted)
    _assert_identical(reference, _shell(_continued(3, 5), **mode))


def test_shell_budget_bounds_population_and_matches():
    mode = {"secondaries": True, "radiative": True}
    reference = _shell(REFERENCE, **mode)
    records = []
    cont = TransportContinuationConfig(rows_per_round=8, on_round=records.append)
    # Midpoint 115 + hard pair 10 + secondary directions 24 + photon pair 10.
    budget = 40 * (8 * (2 * 159 + 24) + 13)
    config = replace(REFERENCE, scratch_budget_bytes=budget, continuation=cont)
    _assert_identical(reference, _shell(config, **mode))
    rounds = [r for r in records if r["phase"] == "round"]
    assert rounds[0]["row_bytes"] == 159
    assert max(r["launched"] for r in rounds) == 40
    assert all(r["scratch_bytes"] <= budget for r in rounds)


@pytest.mark.parametrize("energy_model", ["midpoint", "frozen"])
@pytest.mark.parametrize("staging", [False, True])
def test_urban_straggling_matches_full_history(energy_model, staging):
    reference = _layered(REFERENCE, energy_model=energy_model, straggling=True)
    assert reference["n_side_exited"] > 0
    config = _continued(3, 41)
    config = replace(config, continuation=replace(config.continuation, device_staging=staging))
    _assert_identical(reference, _layered(config, energy_model=energy_model, straggling=True))


def test_urban_straggling_with_radiation_matches_full_history():
    reference = _radiative(REFERENCE, straggling=True)
    for quantum, active in ((1, None), (64, 5), (4096, None)):
        _assert_identical(reference, _radiative(_continued(quantum, active), straggling=True))


def test_shell_step_limit_is_reported_identically():
    with pytest.raises(TransportStepLimitError) as expected:
        _shell(REFERENCE, straggling=True, max_steps=6)
    with pytest.raises(TransportStepLimitError) as actual:
        _shell(_continued(4, 50), straggling=True, max_steps=6)
    assert str(actual.value) == str(expected.value)


def test_lut_is_rejected():
    with pytest.raises(NotImplementedError, match="transport LUT"):
        simulate_trajectories(
            E0_keV=40.0,
            Ne=16,
            thickness_ang=1.0e4,
            element="C",
            n_atoms_per_ang3=0.176,
            elastic_model="sr",
            radiative_model="uncoupled",
            transport_core="cuda",
            transport_lut_config=TransportLUTConfig(),
            per_electron_config=_continued(8),
        )


# Production elastic model (#799): ELSEPA always runs the exact kernel.
_ELSEPA = dict(
    E0_keV=30.0,
    thickness_ang=5.0e4,
    element="Si",
    n_atoms_per_ang3=0.04996,
    seed=11,
    elastic_model="elsepa",
    energy_model="midpoint",
    transport_core="cuda",
)


def _elsepa(config, Ne=200, **kw):
    return simulate_trajectories(Ne=Ne, per_electron_config=config, **(_ELSEPA | kw))


@pytest.mark.parametrize("energy_model", ["midpoint", "frozen"])
def test_elsepa_rows_match_full_history(energy_model):
    reference = _elsepa(REFERENCE, energy_model=energy_model)
    for config in (_continued(5, 31), _continued(256)):
        _assert_identical(reference, _elsepa(config, energy_model=energy_model))


@pytest.mark.slow
def test_mott_rows_match_full_history():
    reference = _elsepa(REFERENCE, elastic_model="mott")
    _assert_identical(reference, _elsepa(_continued(7, 31), elastic_model="mott"))


def test_elsepa_coupled_radiative_rows_match_full_history():
    kw = dict(
        element="C",
        n_atoms_per_ang3=0.1,
        E0_keV=60.0,
        thickness_ang=4_000.0,
        E_cut_keV=10.0,
        max_dE_frac=0.05,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": _table(1e5)},
    )
    reference = _elsepa(REFERENCE, **kw)
    assert np.count_nonzero(reference["event_kind"] == EVENT_HARD_RADIATIVE) > 0
    _assert_identical(reference, _elsepa(_continued(6, 40), **kw))


def test_electron_block_with_beam_spot_matches_full_history():
    """An adaptive block draws its slice of the streams; spot and energy spread.

    Blocks take no in-block bunch sampling (the block driver offsets times).
    """
    kw = dict(
        _electron_start=70,
        _energy_range_keV=(5.0, 30.1),
        crystal_width_mm=2e-3,
        crystal_height_mm=2e-3,
        beam_fwhm_mm=1e-3,
        energy_spread_frac=1e-3,
    )
    reference = _elsepa(REFERENCE, Ne=90, **kw)
    assert reference["n_missed"] > 0
    _assert_identical(reference, _elsepa(_continued(7, 19), Ne=90, **kw))


def test_auto_schedule_continues_and_matches(monkeypatch, caplog):
    caplog.set_level("INFO", logger=continuation_module.__name__)
    monkeypatch.setattr(continuation_module, "CONTINUATION_AUTO_MIN_HISTORIES", 100)
    reference = _elsepa(REFERENCE)
    auto = replace(REFERENCE, schedule="auto")
    _assert_identical(reference, _elsepa(auto))
    assert "auto: continuation for 200 histories" in caplog.text
    caplog.clear()
    lut = simulate_trajectories(
        Ne=200,
        per_electron_config=auto,
        **(_ELSEPA | {"elastic_model": "sr", "transport_lut_config": TransportLUTConfig()}),
    )
    assert lut["electron_id"].size
    assert "full-history because the CUDA transport LUT" in caplog.text


_RUNNER_SCRIPT = textwrap.dedent(
    """
    import hashlib, json, sys
    import numpy as np
    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
    from pyrite.montecarlo import runner

    case = build_cases(Sweep(material=sys.argv[1], thickness_ang=2.0e4,
        beam=BeamSpec(energy_keV=60.0, energy_spread_frac=1e-3, bunch_length_fs=5.0),
        tilt_deg=30.0, n_electrons=96, n_electrons_brem=64))[0]
    out = runner.run_case(case, transport_core="cuda")
    digest = {}
    for key in sorted(out):
        value = out[key]
        if isinstance(value, np.ndarray) and value.dtype != object:
            digest[key] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    print(json.dumps(digest))
    """
)


def _runner_digest(material, schedule):
    env = os.environ | {"PYRITE_MC_TRANSPORT_SCHEDULE": schedule}
    done = subprocess.run(
        [sys.executable, "-c", _RUNNER_SCRIPT, material], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.slow
@pytest.mark.parametrize("material", ["pbwo4", "silicon"])
def test_runner_production_defaults_identical_under_env_continuation(material):
    """Production defaults (ELSEPA, coupled BremsLib, ``auto`` inelastic:
    continuous for pbwo4, shell soft/hard for silicon) reach continuation
    through the process environment and return identical spectra."""
    reference = _runner_digest(material, "full-history")
    assert reference
    assert _runner_digest(material, "continuation") == reference


@pytest.mark.slow
def test_runner_retry_and_block_monitor_match_full_history(monkeypatch):
    """Real step-limit retries and an early block stop preserve the output."""
    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
    from pyrite.montecarlo import runner

    case = build_cases(
        Sweep(
            material="hopg",
            thickness_ang=1e4,
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=30.0,
            n_electrons=60,
            n_electrons_brem=60,
        )
    )[0]
    real_simulate = runner.simulate_trajectories

    class Monitor:
        def __init__(self):
            self.blocks = []

        def on_block(self, start, stop, block):
            self.blocks.append((start, stop, block["electron_id"].size))

        def should_stop(self, stop):
            return stop >= 40

    def run(config):
        budgets = []

        def simulate(*args, max_steps, **kwargs):
            budgets.append(max_steps)
            kwargs["transport_lut_config"] = NO_LUT
            return real_simulate(*args, max_steps=max_steps, per_electron_config=config, **kwargs)

        monkeypatch.setattr(runner, "simulate_trajectories", simulate)
        monitor = Monitor()
        result = runner._transport_case(
            case, transport_core="cuda", block_electrons=20, block_monitor=monitor
        )
        return result, monitor.blocks, budgets

    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", "1")
    reference, ref_blocks, ref_budgets = run(REFERENCE)
    candidate, blocks, budgets = run(_continued(7, 13))
    assert ref_budgets == budgets
    assert max(budgets) > 1
    assert blocks == ref_blocks and [b[:2] for b in blocks] == [(0, 20), (20, 40)]
    _assert_identical(reference["segs"], candidate["segs"])
