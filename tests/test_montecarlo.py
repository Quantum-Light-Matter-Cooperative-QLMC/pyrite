"""montecarlo.py: the "no silent default material" guards + the transport table.



Importing montecarlo prints a GPU/CPU banner and is otherwise CPU-only here; no

full sweep is run (that lives in checks/)."""

import numpy as np
import pytest

import cxr_mc.montecarlo.transport as transport
from cxr_mc.montecarlo import (
    TRANSPORT_ELEMENTS,
    _normalize_composition,
    mc_spectrum,
    simulate_trajectories,
)
from cxr_mc.sweep import crystal_params


def test_normalize_requires_material():
    with pytest.raises(ValueError):
        _normalize_composition(None, None, None)


def test_normalize_composition_passthrough():
    assert _normalize_composition(None, None, [("Mo", 0.01)]) == [("Mo", 0.01)]


def test_simulate_trajectories_refuses_silent_default():
    with pytest.raises(ValueError):
        simulate_trajectories(30.0, 4, 1e4)  # no element / composition given


def test_mc_spectrum_requires_crystal_and_hkl():
    with pytest.raises(TypeError):
        mc_spectrum({"E_keV": np.array([30.0])}, np.arange(50.0, 100.0, 1.0))


def test_mc_spectrum_requires_B_ang2():
    # crystal + hkl supplied, but B_ang2 left at its sentinel -> explicit ValueError

    with pytest.raises(ValueError):
        mc_spectrum({}, np.arange(50.0, 100.0, 1.0), crystal="silicon", hkl_list=[(1, 1, 1)])


def test_tellurium_in_transport_table():
    assert TRANSPORT_ELEMENTS["Te"]["Z"] == 52

    assert TRANSPORT_ELEMENTS["Te"]["A"] == pytest.approx(127.6, abs=0.1)


def test_vanadium_transport_parameters():
    assert TRANSPORT_ELEMENTS["V"] == {
        "Z": 23,
        "A": pytest.approx(50.9415, abs=0.0001),
        "J_keV": 0.245,
    }


def test_titanium_transport_parameters():
    assert TRANSPORT_ELEMENTS["Ti"] == {
        "Z": 22,
        "A": pytest.approx(47.867, abs=0.001),
        "J_keV": 0.233,
    }


def test_niobium_transport_parameters_and_fallback(monkeypatch):
    params = TRANSPORT_ELEMENTS["Nb"]
    assert params == {"Z": 41, "A": pytest.approx(92.906, abs=0.001), "J_keV": 0.417}

    mott_calls = []
    alpha_calls = []
    original_alpha_sr_joy = transport._alpha_sr_joy

    def missing_mott_table(element, Z):
        mott_calls.append((element, Z))
        raise FileNotFoundError

    def spy_alpha_sr_joy(Z, E_keV):
        alpha_calls.append((Z, E_keV.copy()))
        return original_alpha_sr_joy(Z, E_keV)

    monkeypatch.setattr(transport, "_mott_alpha_table", missing_mott_table)
    monkeypatch.setattr(transport, "_alpha_sr_joy", spy_alpha_sr_joy)

    previous_no_mott = set(transport._NO_MOTT)
    transport._NO_MOTT.discard("Nb")
    energies = np.array([30.0, 25.0])
    try:
        transport._sample_cos_theta(41, energies, np.random.default_rng(1), "mott", "Nb")
        transport._sample_cos_theta(41, energies, np.random.default_rng(2), "mott", "Nb")

        assert mott_calls == [("Nb", 41)]
        assert len(alpha_calls) == 2
        for Z, called_energies in alpha_calls:
            assert Z == 41
            np.testing.assert_array_equal(called_energies, energies)
        assert "Nb" in transport._NO_MOTT
    finally:
        transport._NO_MOTT.clear()
        transport._NO_MOTT.update(previous_no_mott)


@pytest.mark.parametrize(
    ("element", "expected"),
    [
        ("Fe", {"Z": 26, "A": 55.845, "J_keV": 0.286}),
        ("Bi", {"Z": 83, "A": 208.98040, "J_keV": 0.823}),
        ("Re", {"Z": 75, "A": 186.207, "J_keV": 0.736}),
        ("Ta", {"Z": 73, "A": 180.94788, "J_keV": 0.718}),
    ],
)
def test_new_element_transport_parameters(element, expected):
    assert TRANSPORT_ELEMENTS[element] == expected


@pytest.mark.parametrize(("element", "Z"), [("Fe", 26), ("Bi", 83), ("Re", 75), ("Ta", 73)])
def test_new_elements_use_analytic_fallback_without_mott_table(monkeypatch, element, Z):
    mott_calls = []
    alpha_calls = []
    original_alpha_sr_joy = transport._alpha_sr_joy

    def missing_mott_table(called_element, called_Z):
        mott_calls.append((called_element, called_Z))
        raise FileNotFoundError

    def spy_alpha_sr_joy(called_Z, E_keV):
        alpha_calls.append((called_Z, E_keV.copy()))
        return original_alpha_sr_joy(called_Z, E_keV)

    monkeypatch.setattr(transport, "_mott_alpha_table", missing_mott_table)
    monkeypatch.setattr(transport, "_alpha_sr_joy", spy_alpha_sr_joy)

    previous_no_mott = set(transport._NO_MOTT)
    transport._NO_MOTT.discard(element)
    energies = np.array([30.0, 25.0])
    try:
        cos_theta = transport._sample_cos_theta(
            Z, energies, np.random.default_rng(1), "mott", element
        )
        transport._sample_cos_theta(Z, energies, np.random.default_rng(2), "mott", element)

        assert mott_calls == [(element, Z)]
        assert len(alpha_calls) == 2
        for called_Z, called_energies in alpha_calls:
            assert called_Z == Z
            np.testing.assert_array_equal(called_energies, energies)
        assert np.all((-1.0 <= cos_theta) & (cos_theta <= 1.0))
        assert element in transport._NO_MOTT
    finally:
        transport._NO_MOTT.clear()
        transport._NO_MOTT.update(previous_no_mott)


def test_hbn_composition_runs_transport():
    cp = crystal_params("hbn")

    segs = simulate_trajectories(
        30.0,
        4,
        100.0,
        composition=cp["composition"],
        seed=123,
        max_steps=5,
    )

    assert segs["Ne"] == 4
    assert len(segs["E_keV"]) > 0


def test_beam_fwhm_mm_zero_and_none_are_equivalent():
    # 0 is falsy, same branch as None -> point source, no RNG draw for the offset
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=123, max_steps=5)
    none_segs = simulate_trajectories(30.0, 20, 100.0, beam_fwhm_mm=None, **kw)
    zero_segs = simulate_trajectories(30.0, 20, 100.0, beam_fwhm_mm=0.0, **kw)
    assert np.array_equal(none_segs["r_mid"], zero_segs["r_mid"])


def test_beam_fwhm_mm_offsets_transverse_position_only():
    # A finite beam spot must be a pure (x, y) translation of each electron's
    # WHOLE trajectory -- every other physics output (energies, directions,
    # segment lengths, ages, backscatter/transmit counts) is untouched, because
    # the offset is drawn from an RNG stream independent of the transport rng.
    cp = crystal_params("hbn")
    args = (30.0, 200, 100.0)
    common = dict(composition=cp["composition"], seed=7, max_steps=20)
    point = simulate_trajectories(*args, beam_fwhm_mm=None, **common)
    finite = simulate_trajectories(*args, beam_fwhm_mm=1.0, **common)

    for key in ("E_keV", "v_hat", "L_ang", "t_ang", "elec_id", "layer"):
        assert np.array_equal(point[key], finite[key]), key
    for key in (
        "n_backscattered",
        "n_transmitted",
        "n_stopped",
        "n_side_exited",
        "n_missed",
        "Ne",
    ):
        assert point[key] == finite[key]

    # depth (z) is identical; transverse (x, y) differs by a per-electron
    # constant offset shared across every segment that electron emits
    assert np.array_equal(point["r_mid"][:, 2], finite["r_mid"][:, 2])
    d_xy = finite["r_mid"][:, :2] - point["r_mid"][:, :2]
    per_electron_offset = {}
    for eid, dxy in zip(finite["elec_id"], d_xy, strict=False):
        if eid in per_electron_offset:
            assert np.allclose(per_electron_offset[eid], dxy)
        else:
            per_electron_offset[eid] = dxy
    assert any(np.linalg.norm(v) > 0 for v in per_electron_offset.values())


def test_beam_fwhm_mm_matches_gaussian_sigma():
    # statistical check: the sampled transverse spread matches sigma = FWHM /
    # (2 sqrt(2 ln 2)), converted mm -> Angstrom (1 mm = 1e7 Ang)
    cp = crystal_params("hbn")
    fwhm_mm = 1.0
    segs = simulate_trajectories(
        30.0,
        20000,
        100.0,
        composition=cp["composition"],
        seed=7,
        max_steps=1,
        beam_fwhm_mm=fwhm_mm,
    )
    # first segment per electron carries the raw entry offset (before any
    # elastic-scattering-driven lateral drift accumulates)
    _, first_idx = np.unique(segs["elec_id"], return_index=True)
    x0y0 = (
        segs["r_mid"][first_idx, :2]
        - 0.5 * segs["L_ang"][first_idx, None] * segs["v_hat"][first_idx, :2]
    )
    expected_sigma_ang = fwhm_mm * 1.0e7 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.std(x0y0) == pytest.approx(expected_sigma_ang, rel=0.05)


def test_beam_fwhm_mm_seed_reproducible():
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=42, max_steps=5, beam_fwhm_mm=1.0)
    a = simulate_trajectories(30.0, 50, 100.0, **kw)
    b = simulate_trajectories(30.0, 50, 100.0, **kw)
    assert np.array_equal(a["r_mid"], b["r_mid"])


def test_finite_footprint_truncates_lateral_transport_and_counts_side_exit():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30.0,
        64,
        1e8,
        composition=cp["composition"],
        seed=4,
        elastic_model="sr",
        beam_dir=[0.8, 0.0, 0.6],
        max_steps=1,
        crystal_width_mm=1e-6,
        crystal_height_mm=1e-6,
    )
    assert segs["n_side_exited"] == 64
    assert segs["n_transmitted"] == segs["n_backscattered"] == 0
    assert np.all(np.abs(segs["r_mid"][:, 0]) < 5.0)


def test_finite_footprint_counts_missed_gaussian_entries_without_changing_ne():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30.0,
        2_000,
        100.0,
        composition=cp["composition"],
        seed=7,
        elastic_model="sr",
        max_steps=1,
        beam_fwhm_mm=1.0,
        crystal_width_mm=1e-6,
        crystal_height_mm=1e-6,
    )
    assert segs["Ne"] == 2_000
    assert segs["n_missed"] > 1_900
    assert np.all(np.abs(segs["r_mid"][:, :2]) <= 5.0)


def test_all_missed_entries_return_typed_empty_segment_arrays():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30.0,
        4,
        100.0,
        composition=cp["composition"],
        seed=2,
        elastic_model="sr",
        beam_fwhm_mm=1e8,
        crystal_width_mm=1e-6,
        crystal_height_mm=1e-6,
    )
    assert segs["n_missed"] == segs["Ne"] == 4
    assert segs["r_mid"].shape == segs["v_hat"].shape == (0, 3)
    assert segs["L_ang"].shape == segs["E_keV"].shape == segs["t_ang"].shape == (0,)
    assert segs["elec_id"].dtype == np.dtype("int64")
    assert segs["layer"].dtype == np.dtype("int16")


def test_omitted_footprint_is_bitwise_legacy_transport():
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=19, elastic_model="sr", max_steps=5)
    old = simulate_trajectories(30.0, 40, 100.0, **kw)
    new = simulate_trajectories(
        30.0,
        40,
        100.0,
        crystal_width_mm=None,
        crystal_height_mm=None,
        **kw,
    )
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer"):
        assert np.array_equal(old[key], new[key]), key


@pytest.mark.parametrize(
    ("width_mm", "height_mm"), [(None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0)]
)
def test_finite_footprint_rejects_invalid_dimension_pairs(width_mm, height_mm):
    cp = crystal_params("hbn")
    with pytest.raises(ValueError):
        simulate_trajectories(
            30.0,
            4,
            100.0,
            composition=cp["composition"],
            crystal_width_mm=width_mm,
            crystal_height_mm=height_mm,
        )
