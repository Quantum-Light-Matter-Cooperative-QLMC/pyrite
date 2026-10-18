"""Saved reductions: analytic depth attribution, tallies, and block invariance."""

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from pyrite.campaign.beam_metrics import capture_beam_metrics, initial_state_metrics
from pyrite.montecarlo.trajectories import TransportView, write_trajectory_artifact
from pyrite.montecarlo.trajectory_balance import capture_secondary_energy_balance
from pyrite.montecarlo.trajectory_summary import (
    depth_loss_profile,
    iter_primary_summaries,
    summarize_trajectories,
)
from pyrite.montecarlo.transport.secondaries import secondary_energy_balance

CASE = dict(
    name="saved Si",
    crystal="silicon",
    E0_keV=10.0,
    thickness_ang=10.0,
    theta_obs_rad=np.pi / 2,
    tilt_deg=0.0,
    tilt_azim_deg=0.0,
)


def transport(n=12):
    history = np.repeat(np.arange(n), 3)
    kinds = np.tile([8, 5, 7], n)
    kinds[1::3] = np.resize([4, 5, 6, 7], n)
    primary = np.tile([True, True, False], n)
    track = np.repeat(np.arange(n) * 2, 3) + ~primary
    result = dict(
        electron_id=history,
        track_id=track,
        parent_id=np.where(primary, -1, track - 1),
        generation=(~primary).astype(int),
        r_mid=np.tile([[0.0, 0.0, 2.0], [0.0, 0.0, 6.0], [0.0, 0.0, 3.0]], (n, 1)),
        v_hat=np.tile([0.0, 0.0, 1.0], (3 * n, 1)),
        L_ang=np.full(3 * n, 4.0),
        E_start_keV=np.tile([10.0, 8.5, 1.0], n),
        E_end_keV=np.tile([9.5, 8.0, 0.8], n),
        t_start_ang=np.tile([0.0, 4.0, 4.0], n),
        t_end_ang=np.tile([4.0, 8.0, 8.0], n),
        t0_ang=np.zeros(3 * n),
        flight_id=np.tile([0, 1, 0], n),
        substep_id=np.zeros(3 * n, dtype=int),
        event_kind=kinds,
        layer=np.zeros(3 * n, dtype=int),
        hard_channel=np.tile([0, -1, -1], n),
        hard_W_keV=np.tile([1.0, 0.0, 0.0], n),
        Ne=n,
        n_backscattered=n // 4,
        n_transmitted=n // 4,
        n_side_exited=n // 4,
        n_cutoff_stopped=n // 4,
        initial_r_ang=np.column_stack([np.arange(n), np.zeros((n, 2))]),
        initial_v_hat=np.tile([0.0, 0.0, 1.0], (n, 1)),
        initial_E_keV=np.full(n, 10.0),
        initial_t0_ang=np.arange(n, dtype=float),
        secondaries=dict(threshold_eV=500.0),
        inelastic=dict(model="shell-soft-hard", channels=[[dict(code=0, inner=False)]]),
        secondary_tracks=dict(
            track_id=np.arange(2 * n),
            generation=np.tile([0, 1], n),
            electron_id=np.repeat(np.arange(n), 2),
            launch_E_keV=np.tile([10.0, 1.0], n),
        ),
    )
    return result


@pytest.fixture
def capture(tmp_path):
    data = transport()
    # Interleave material rows in transported order; persistence sorts histories.
    order = np.arange(36).reshape(12, 3).T.ravel()
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (36,):
            data[key] = value[order]
    return write_trajectory_artifact(tmp_path / "capture.h5", data, case=CASE), data


@pytest.mark.parametrize("vz", [1.0, -1.0])
def test_depth_loss_analytic_and_transverse(vz):
    block = dict(
        r_mid=np.array([[0.0, 0.0, 1.0]]),
        L_ang=np.array([4.0]),
        v_hat=np.array([[0.0, 0.0, vz]]),
        E_start_keV=np.array([20.0]),
        E_end_keV=np.array([8.0]),
        layer=np.array([2]),
    )
    table = depth_loss_profile(block, [-1, 0, 2, 3])
    np.testing.assert_allclose(table.continuous_loss_keV, [3, 6, 3])
    assert table.layer.tolist() == [2, 2, 2]
    assert depth_loss_profile(block, [0, 2]).continuous_loss_keV.sum() == 6
    block["v_hat"][0, 2] = 0
    for z in [1.0, 2.0]:
        block["r_mid"][0, 2] = z
        np.testing.assert_allclose(
            depth_loss_profile(block, [0, 1, 2]).continuous_loss_keV, [0, 12]
        )


def summary(path, max_rows):
    return summarize_trajectories(
        path,
        depth_edges_ang=[-1, 0, 2, 4, 8, 10],
        energy_edges_keV=[0, 0.5, 2, 10],
        max_rows=max_rows,
    )


@pytest.mark.parametrize("schema", [2, 3])
def test_block_invariance_and_primary_fates(capture, schema):
    from tests.helpers.trajectory_schema import downgrade

    path, data = capture
    if schema == 2:
        downgrade(path, 2)
    small, large = summary(path, 4), summary(path, 1000)
    for key in small:
        pd.testing.assert_frame_equal(small[key], large[key])
    fates = small["fates"].set_index("fate")
    for fate in ("backscattered", "transmitted", "side_exited", "cutoff_stopped"):
        assert fates.loc[fate, "count"] == data[f"n_{fate}"]
        assert fates.loc[fate, "fraction"] == 0.25
    assert small["depth_loss"].continuous_loss_keV.sum() == pytest.approx(14.4)
    assert small["hard_events"]["count"].sum() == 12
    assert small["generations"].tracks.tolist() == [12, 12]
    assert small["exit_energy"]["count"].sum() == 9
    per_history = pd.concat(iter_primary_summaries(path, max_rows=4), ignore_index=True)
    assert len(per_history) == 12
    assert per_history.penetration_ang.tolist() == [8.0] * 12
    assert per_history.range_ang.tolist() == [8.0] * 12


def test_row_only_mode_skips_shared_arrays_and_eager_index(capture, monkeypatch):
    path, _ = capture

    def fail(*args, **kwargs):
        raise AssertionError("streaming reductions must not materialize the history index")

    monkeypatch.setattr(TransportView, "electron_index", fail)
    getitem = h5py.Dataset.__getitem__
    reads = []

    def guarded(dataset, key):
        if dataset.name.startswith("/transport/") and dataset.ndim:
            assert key != (), dataset.name
            if isinstance(key, slice):
                reads.append((key.stop or len(dataset)) - (key.start or 0))
        return getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    summary(path, 4)
    assert reads and max(reads) <= 4


def test_streamed_secondary_energy_ledger_matches_canonical(capture):
    path, data = capture
    expected = secondary_energy_balance(data)
    for size in [4, 1000]:
        got = capture_secondary_energy_balance(path, max_rows=size).set_index("term").energy_keV
        for term, value in expected.items():
            assert got[term] == pytest.approx(value, abs=1e-12)
        assert got["residual_keV"] == pytest.approx(0.0, abs=1e-12)
    assert summary(path, 4)["depth_loss"].continuous_loss_keV.sum() == pytest.approx(
        expected["continuous_keV"]
    )


def test_beam_metrics_reuse_sampled_definition(capture):
    path, data = capture
    beam = SimpleNamespace(bunch_charge_pc=1.0, rep_rate_hz=5000.0)
    metrics, sampled, total = capture_beam_metrics(str(path), beam, max_particles=5)
    reference = initial_state_metrics(
        *(
            data[key][:5]
            for key in ("initial_r_ang", "initial_v_hat", "initial_t0_ang", "initial_E_keV")
        ),
        beam,
    )
    assert sampled == 5 and total == 12
    assert metrics.sigma_t_fs == reference.sigma_t_fs
    assert metrics.x.position_sigma_mm == reference.x.position_sigma_mm


@pytest.mark.parametrize("edges", [[1], [1, 0], [0, np.nan]])
def test_invalid_edges(capture, edges):
    with pytest.raises(ValueError):
        summarize_trajectories(capture[0], depth_edges_ang=edges, energy_edges_keV=[0, 10])


@pytest.mark.parametrize("positrons", [False, True])
def test_pair_and_annihilation_ledger_remaps_original_rows(tmp_path, positrons):
    from pyrite.montecarlo.transport.pair_production import PAIR_THRESHOLD_EV

    data = transport(12)
    first = np.arange(0, 36, 3)
    data["E_start_keV"][first] = 2000.0
    data["E_end_keV"][first] = 1999.5
    data["E_start_keV"][first + 1] = 898.5
    data["E_end_keV"][first + 1] = 898.0
    data["event_kind"][first] = 9
    data["hard_radiative_k_eV"] = np.zeros(36)
    data["hard_radiative_k_eV"][first] = 1.1e6
    data["secondary_tracks"]["launch_E_keV"][::2] = 2000.0
    rest = PAIR_THRESHOLD_EV * 1e-3
    events = dict(
        row=first,
        electron_id=np.arange(12),
        electron_launched=np.zeros(12, dtype=bool),
        electron_keV=np.full(12, 0.2),
        positron_keV=np.full(12, 1100.0 - rest - 0.2),
    )
    pair = dict(events=events, positrons_transported=positrons)
    if positrons:
        events.update(positron_launched=np.zeros(12, dtype=bool), positron_fate=np.full(12, 1))
        pair["annihilation_photons"] = dict(
            event=np.repeat(np.arange(12), 2),
            absorbed=np.tile([False, True], 12),
            k_keV=np.full(24, rest / 2),
        )
    data["pair_production"] = pair
    # Force original row IDs to differ from stored row IDs.
    order = np.arange(36).reshape(12, 3).T.ravel()
    inverse = np.argsort(order)
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (36,):
            data[key] = value[order]
    events["row"] = inverse[first]
    expected = secondary_energy_balance(data)
    path = write_trajectory_artifact(tmp_path / "pairs.h5", data, case=CASE)
    for size in [4, 1000]:
        actual = capture_secondary_energy_balance(path, max_rows=size).set_index("term").energy_keV
        for term, value in expected.items():
            assert actual[term] == pytest.approx(value, abs=1e-9)
        assert actual["residual_keV"] == pytest.approx(0.0, abs=1e-9)


def test_missed_histories_keep_transport_fraction_denominator(tmp_path):
    data = transport()
    data["Ne"] = 15
    data["n_missed"] = 2
    path = write_trajectory_artifact(tmp_path / "missed.h5", data, case=CASE)
    fates = summary(path, 4)["fates"].set_index("fate")
    assert fates.loc["transmitted", "fraction"] == 3 / 15
    assert fates.loc["missed", "count"] == 2
    assert fates.loc["without_rows", "count"] == 1
    assert fates.fraction.sum() == pytest.approx(1.0)
