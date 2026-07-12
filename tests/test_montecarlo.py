"""montecarlo.py: the "no silent default material" guards + the transport table.



Importing montecarlo prints a GPU/CPU banner and is otherwise CPU-only here; no

full sweep is run (that lives in checks/)."""

import numpy as np
import pytest

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
    for key in ("n_backscattered", "n_transmitted", "n_stopped", "Ne"):
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
