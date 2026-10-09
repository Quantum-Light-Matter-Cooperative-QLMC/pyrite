"""Opt-in transport of shell hard-collision secondaries (``secondary_threshold_eV``).

Validation: shell-secondary-transport
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.characteristic import mc_characteristic_spectrum
from pyrite.montecarlo.transport import PerElectronTransportConfig, simulate_trajectories
from pyrite.montecarlo.transport.events import (
    EVENT_CUTOFF,
    check_segment_event_contract,
)
from pyrite.montecarlo.transport.hard_inelastic import (
    _hard_secondary_cosine,
    _hard_secondary_direction,
    _sample_hard_transfer_eV,
    hard_stream_keys,
)
from pyrite.montecarlo.transport.kinematics import stream_keys
from pyrite.montecarlo.transport.secondaries import (
    secondary_energy_balance,
    secondary_stream_keys,
)
from pyrite.montecarlo.transport.shell_partition import catalog_shell_partition
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators
from pyrite.montecarlo.transport.shell_sampling import (
    BRANCHES,
    sample_shell_hard_collision,
    shell_collision_world_directions,
)
from pyrite.montecarlo.transport.shell_transport import hard_event_energy_accounting
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table
from tests.helpers.bremslib import synthetic_bremslib_arrays

ROW_KEYS = ("r_mid", "v_hat", "L_ang", "E_start_keV", "E_end_keV", "t_end_ang", "hard_W_keV")


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _composition(key):
    if key in CATALOG.crystals:
        return CATALOG.crystal(key).composition
    return CATALOG.media[key].composition


def _run(key="silicon", *, threshold=1000.0, Ne=40, E0=20.0, E_cut=2.0, **kw):
    kw.setdefault("transport_core", "per-electron")
    return simulate_trajectories(
        E0,
        Ne,
        kw.pop("thickness_ang", 2.0e4),
        composition=_composition(key),
        E_cut_keV=E_cut,
        seed=kw.pop("seed", 11),
        energy_model="midpoint",
        stopping_tables=[resolve_catalog_table(key).arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=[key],
        secondary_threshold_eV=threshold,
        **kw,
    )


def _as_tracks(result):
    """Rows keyed by track, for the per-trajectory event contract."""
    return dict(result, electron_id=result["track_id"])


@pytest.mark.slow
def test_batch_progress_covers_every_secondary_generation_without_changing_rows():
    reports = []
    kwargs = dict(Ne=8, per_electron_config=PerElectronTransportConfig(max_batch=2, seg_capacity=8))
    off = _run(**kwargs)
    on = _run(**kwargs, transport_progress=lambda done, total: reports.append((done, total)))
    totals = on["secondaries"]["tracks_per_generation"]
    assert len(totals) >= 2
    assert [total for done, total in reports if done == total] == list(totals)
    for field in (*ROW_KEYS, "track_id", "parent_id", "generation", "electron_id"):
        np.testing.assert_array_equal(off[field], on[field])


@pytest.mark.parametrize("key", ("silicon", "mos2"))
@pytest.mark.parametrize("energy_eV", (6.0e3, 3.0e4))
def test_in_kernel_secondary_direction_matches_host_sampler(key, energy_eV):
    material = catalog_shell_oscillators(key)
    part = catalog_shell_partition(key, energy_eV, 50.0)
    probabilities = part.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    incoming = np.array([0.3, -0.4, 0.5])
    incoming /= np.linalg.norm(incoming)
    checked = 0
    for flat in np.flatnonzero(probabilities > 0.0):
        index, branch = divmod(int(flat), 3)
        osc = material.oscillators[index]
        u_channel = (cumulative[flat] - 0.5 * probabilities[flat]) / cumulative[-1]
        for u_loss, u_recoil, u_phi in ((0.37, 0.61, 0.25), (0.93, 0.08, 0.8), (0.5, 0.0, 0.0)):
            host = sample_shell_hard_collision(material, part, u_channel, u_loss, u_recoil, u_phi)
            if host.cos_secondary is None:
                continue
            world = shell_collision_world_directions(host, tuple(incoming))
            w = _sample_hard_transfer_eV(
                energy_eV,
                osc.ionization_energy_eV,
                osc.resonance_energy_eV,
                branch,
                50.0,
                u_loss,
            )
            assert host.loss.branch == BRANCHES[branch]
            cosine = _hard_secondary_cosine(
                energy_eV,
                osc.ionization_energy_eV,
                osc.resonance_energy_eV,
                branch,
                w,
                u_recoil,
            )
            np.testing.assert_allclose(cosine, host.cos_secondary, rtol=0.0, atol=1e-13)
            direction = _hard_secondary_direction(
                *incoming,
                energy_eV,
                osc.ionization_energy_eV,
                osc.resonance_energy_eV,
                branch,
                w,
                u_recoil,
                u_phi,
            )
            # sin = sqrt(1 - cos^2) maps a one-ulp cosine difference near cos = 1
            # (u_recoil = 0 puts Q at Q_min, a forward secondary) to ~sqrt(2 eps).
            near_forward = 1.0 - host.cos_secondary < 1e-12
            atol = np.sqrt(8.0 * np.finfo(float).eps) if near_forward else 1e-9
            np.testing.assert_allclose(direction, world.secondary, rtol=0.0, atol=atol)
            checked += 1
    assert checked >= 6


def test_off_allocates_nothing_and_on_keeps_primary_rows_bitwise():
    """``None`` adds no field; turning it on changes no primary row."""
    for core, lut in (("lockstep", True), ("lockstep", False), ("per-electron", False)):
        kw = {"transport_core": core}
        if not lut:
            from pyrite.montecarlo.transport import TransportLUTConfig

            kw["transport_lut_config"] = TransportLUTConfig(enabled=False)
        off = _run(threshold=None, **kw)
        on = _run(**kw)
        for field in ("track_id", "parent_id", "generation", "hard_secondary_v_hat"):
            assert field not in off
        primary = on["generation"] == 0
        assert np.count_nonzero(~primary) > 0
        for field in (*ROW_KEYS, "electron_id", "flight_id", "event_kind", "hard_channel"):
            np.testing.assert_array_equal(np.asarray(on[field])[primary], off[field])
        for field in ("n_backscattered", "n_transmitted", "n_cutoff_stopped", "Ne"):
            assert on[field] == off[field]


@pytest.mark.parametrize("key", ("silicon", "mos2"))
@pytest.mark.parametrize("straggling", (False, True))
def test_energy_and_particle_balance_close_over_all_generations(key, straggling):
    result = _run(key, Ne=30, E0=30.0, thickness_ang=3.0e4, straggling=straggling)
    check_segment_event_contract(_as_tracks(result))
    terms = secondary_energy_balance(result)
    assert terms["binding_reserved_keV"] > 0.0 and terms["subthreshold_keV"] > 0.0
    # Sums of ~1e4 float64 row differences of O(10 keV) values.
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    # The identity closes for each primary history, not only in aggregate.
    by_history = secondary_energy_balance(result, per_history=True)
    for name, value in terms.items():
        assert by_history[name].shape == (30,)
        np.testing.assert_allclose(by_history[name].sum(), value, rtol=1e-12, atol=1e-9)
    assert np.all(np.abs(by_history["residual_keV"]) <= 1e-9 * 30.0)
    assert np.count_nonzero(by_history["subthreshold_keV"]) > 1

    info = result["secondaries"]
    tracks = result["secondary_tracks"]
    assert info["n_generations"] >= 2
    generation = result["generation"]
    for g, counts in enumerate(info["counts_per_generation"]):
        n = info["tracks_per_generation"][g]
        assert sum(counts.values()) + (result["n_missed"] if g == 0 else 0) == n
        # Generation g+1 has one track per launched hard row of generation g.
        rows = generation == g
        acc = hard_event_energy_accounting(
            {k: np.asarray(result[k])[rows] for k in ("hard_channel", "layer", "hard_W_keV")}
            | {"inelastic": result["inelastic"]}
        )
        launched = int(np.count_nonzero(acc["secondary_keV"] * 1e3 > info["threshold_eV"]))
        expected = info["tracks_per_generation"][g + 1] if g + 1 < info["n_generations"] else 0
        assert launched == expected
    assert tracks["track_id"].size == sum(info["tracks_per_generation"])


def test_launch_state_identity_and_direction():
    result = _run(Ne=40)
    tracks = result["secondary_tracks"]
    track = result["track_id"]
    secondary = tracks["generation"] > 0
    assert np.all(tracks["parent_id"][secondary] < tracks["track_id"][secondary])
    assert np.all(tracks["launch_E_keV"][secondary] > 1.0)
    # electron_id is the primary history of every row.
    np.testing.assert_array_equal(result["electron_id"], tracks["electron_id"][track])
    assert np.all(result["electron_id"] < result["Ne"])
    np.testing.assert_array_equal(result["parent_id"], tracks["parent_id"][track])
    np.testing.assert_array_equal(result["generation"], tracks["generation"][track])
    # Each secondary's first row is its launch state, and the launch sits on
    # its parent's hard row: end point, clock and recorded direction.
    for t in np.flatnonzero(secondary)[:25]:
        rows = np.flatnonzero(track == t)
        first = rows[np.argmin(result["flight_id"][rows] * 10**6 + result["substep_id"][rows])]
        start = result["r_mid"][first] - 0.5 * result["L_ang"][first] * result["v_hat"][first]
        np.testing.assert_allclose(start, tracks["launch_r_ang"][t], atol=1e-8)
        np.testing.assert_allclose(result["v_hat"][first], tracks["launch_v_hat"][t], atol=1e-15)
        assert result["E_start_keV"][first] == tracks["launch_E_keV"][t]
        assert result["t_start_ang"][first] == tracks["launch_t_ang"][t]
        parent_rows = np.flatnonzero(
            (track == tracks["parent_id"][t]) & (result["hard_channel"] >= 0)
        )
        order = parent_rows[np.argsort(result["flight_id"][parent_rows])]
        hard = order[tracks["parent_hard_ordinal"][t]]
        end = result["r_mid"][hard] + 0.5 * result["L_ang"][hard] * result["v_hat"][hard]
        np.testing.assert_allclose(end, tracks["launch_r_ang"][t], rtol=0.0, atol=1e-9)
        np.testing.assert_array_equal(
            result["hard_secondary_v_hat"][hard], tracks["launch_v_hat"][t]
        )
        assert result["t_end_ang"][hard] == tracks["launch_t_ang"][t]
        assert result["t0_ang"][first] == result["t0_ang"][hard]


def test_secondary_rows_do_not_depend_on_batch_size():
    reference = _run(Ne=24)
    small = _run(
        Ne=24,
        per_electron_config=PerElectronTransportConfig(max_batch=5, seg_capacity=8),
    )
    assert reference["secondaries"] == small["secondaries"]
    for field in (*ROW_KEYS, "track_id", "parent_id", "generation", "electron_id"):
        np.testing.assert_array_equal(small[field], reference[field])


def test_secondary_stream_keys_are_keyed_on_parent_and_ordinal():
    keys = secondary_stream_keys(7, np.array([0, 0, 1, 5]), np.array([0, 1, 0, 0]))
    assert keys.dtype == np.uint64 and np.unique(keys).size == 4
    # The same (parent, ordinal) gives the same key whatever else is launched.
    assert secondary_stream_keys(7, np.array([1]), np.array([0]))[0] == keys[2]
    assert secondary_stream_keys(8, np.array([1]), np.array([0]))[0] != keys[2]
    primary = stream_keys(7, 1000)
    parents, ordinals = np.meshgrid(np.arange(100), np.arange(10))
    many = secondary_stream_keys(7, parents.ravel(), ordinals.ravel())
    assert np.unique(many).size == many.size
    assert not np.intersect1d(many, primary).size
    assert not np.intersect1d(many, hard_stream_keys(7, 1000)).size


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.0}, "finite and positive"),
        ({"threshold": 500.0}, "SBETHE table floor"),
        ({"collect_diagnostics": True}, "collect_diagnostics"),
        ({"max_secondary_tracks": 1}, "max_secondary_tracks"),
        ({"max_secondary_generations": 0}, "positive integer"),
    ],
)
def test_invalid_secondary_configuration_raises(kwargs, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _run(Ne=20, **kwargs)


def test_secondaries_require_the_shell_mode():
    with pytest.raises(ValueError, match="shell-soft-hard"):
        simulate_trajectories(
            20.0,
            5,
            2.0e4,
            composition=_composition("silicon"),
            E_cut_keV=2.0,
            energy_model="midpoint",
            secondary_threshold_eV=1000.0,
        )


def test_generation_cap_fails_loudly():
    with pytest.raises(RuntimeError, match="max_secondary_generations=1"):
        _run(Ne=20, E0=100.0, thickness_ang=1.0e5, max_secondary_generations=1)


def test_per_shell_characteristic_yield_ignores_explicit_vacancies():
    """Vacancies stay bookkeeping: the track-length yield of the primaries is
    unchanged, and only secondary tracks add characteristic path."""
    kw = {"Ne": 30, "E0": 20.0, "thickness_ang": 2.0e4}
    off = _run("mos2", threshold=None, **kw)
    on = _run("mos2", **kw)
    composition = _composition("mos2")
    grid = np.arange(2000.0, 2600.0, 2.0)  # Mo L and S K lines

    def spectrum(segments):
        return mc_characteristic_spectrum(segments, grid, composition=composition, E_cut_keV=2.0)

    primary = on["generation"] == 0
    primaries_only = {
        k: (
            np.asarray(v)[primary]
            if k
            in (
                *ROW_KEYS,
                "E_keV",
                "E_repr_keV",
                "elec_id",
                "electron_id",
                "layer",
                "t_ang",
                "t0_ang",
                "t_start_ang",
                "event_kind",
                "flight_id",
                "substep_id",
                "hard_channel",
            )
            else v
        )
        for k, v in on.items()
    }
    no_vacancies = dict(off, hard_channel=np.full_like(off["hard_channel"], -1))
    no_vacancies["hard_W_keV"] = np.zeros_like(off["hard_W_keV"])
    reference = spectrum(off)
    np.testing.assert_array_equal(spectrum(primaries_only), reference)
    np.testing.assert_array_equal(spectrum(no_vacancies), reference)
    # Secondary tracks above the 2 keV scoring floor add yield, never remove it.
    assert np.all(spectrum(on) >= reference) and spectrum(on).sum() > reference.sum()


def test_coupled_radiative_secondaries_balance_energy():
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=14)
    table = replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 1e5,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 1e5,
    )
    result = _run(
        Ne=20,
        E0=60.0,
        E_cut=10.0,
        threshold=10_000.0,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"Si": table},
    )
    check_segment_event_contract(_as_tracks(result))
    terms = secondary_energy_balance(result)
    assert terms["radiated_keV"] > 0.0
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    with pytest.raises(ValueError, match="radiative_cutoff_eV must not exceed"):
        _run(
            Ne=5,
            E0=60.0,
            E_cut=10.0,
            threshold=10_000.0,
            radiative_model="bremslib-soft-hard",
            radiative_cutoff_eV=20_000.0,
            bremslib_tables={"Si": table},
        )


def test_absorbing_collisions_still_launch():
    result = _run(Ne=60, E0=6.0, E_cut=3.0, threshold=1000.0, thickness_ang=4.0e4)
    tracks = result["secondary_tracks"]
    kinds = result["event_kind"]
    track = result["track_id"]
    absorbing = np.flatnonzero((kinds == EVENT_CUTOFF) & (result["hard_channel"] >= 0))
    parents = set(tracks["parent_id"][tracks["generation"] > 0].tolist())
    assert absorbing.size and any(int(track[i]) in parents for i in absorbing)


def test_incoherent_lines_add_tracks_and_coherent_rejects_showers():
    """Flights group by track: a shower's spectrum is the sum over its tracks."""
    from pyrite.montecarlo.spectrum import mc_spectrum

    result = _run(Ne=12, max_dE_frac=0.05, elastic_model="sr")
    kwargs = {
        "crystal": "silicon",
        "hkl_list": [(2, 2, 0)],
        "B_ang2": 0.47,
        "theta_obs_rad": np.deg2rad(119.0),
        "composition": _composition("silicon"),
    }
    grid = np.linspace(500.0, 6000.0, 400)

    def part(mask):
        return {
            k: (
                np.asarray(v)[mask]
                if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape
                else v
            )
            for k, v in result.items()
        }

    primary = result["generation"] == 0
    whole = mc_spectrum(result, grid, **kwargs)
    assert whole.sum() > 0.0
    split = mc_spectrum(part(primary), grid, **kwargs) + mc_spectrum(part(~primary), grid, **kwargs)
    np.testing.assert_allclose(whole, split, rtol=1e-9, atol=1e-12 * whole.max())
    with pytest.raises(NotImplementedError, match="secondary transport"):
        mc_spectrum(result, grid, coherent=True, **kwargs)


def test_case_threshold_reaches_runner_transport():
    """A case's secondary_threshold_eV drives the production transport call."""
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins
    from pyrite.montecarlo.runner import _transport_case

    detector = pr.Detector(
        energy_bins=EnergyBins(line=np.linspace(1500, 2000, 20), brem=np.linspace(1500, 9000, 20))
    )
    numerics = pr.Numerics(
        n_electrons=8,
        n_electrons_brem=8,
        energy_model="midpoint",
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        secondary_threshold_eV=1000.0,
        elastic_model="mott",
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=20.0), pr.Slab("silicon", thickness_ang=5.0e4, tilt_deg=30.0), detector
    )
    case = api.build_case(scene, numerics)
    assert case["secondary_threshold_eV"] == 1000.0
    segments = _transport_case(case, transport_core="per-electron")["segs"]
    assert segments["secondaries"]["threshold_eV"] == 1000.0
    assert segments["generation"].size == segments["L_ang"].size


def test_threshold_above_every_secondary_launches_nothing():
    """Limit: T_s above E0 launches no track and leaves the primary rows unchanged."""
    off = _run(threshold=None, Ne=20)
    high = _run(threshold=30_000.0, Ne=20)
    assert high["secondaries"]["n_generations"] == 1
    assert np.all(high["generation"] == 0) and np.all(high["parent_id"] == -1)
    np.testing.assert_array_equal(high["track_id"], high["electron_id"])
    for field in ROW_KEYS:
        np.testing.assert_array_equal(high[field], off[field])
    terms = secondary_energy_balance(high)
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
