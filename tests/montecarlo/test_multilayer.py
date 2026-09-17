"""Film-on-substrate (multilayer) escape geometry and sweep plumbing.

The ordinary tests stay CPU/NumPy so their fp64 assertions remain reproducible.
The one end-to-end CUDA regression runs in an explicitly spawned device session:
it protects the float32 multilayer accumulation contract without relaxing the
CPU anchor in ``checks/multilayer_slice3_check.py``.
"""

import os
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from pyrite.campaign.sweep import (
    BeamSpec,
    LayerSpec,
    Stack,
    Sweep,
    build_cases,
    film_on_substrate_layers,
    layer_radiator,
    stack_layers,
    substrate_composition,
    substrate_radiator,
)
from pyrite.detectors import Detector, EnergyBins
from pyrite.montecarlo import (
    _layer_dz,
    _layer_path_length,
    _mu_total_inv_ang,
    _segments_in_layer,
    _spectrum_case,
    _stack_tau,
    _transport_case,
    mc_characteristic_spectrum,
    mc_spectrum,
    simulate_trajectories,
)

try:
    import cupy as cp

    _HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _HAS_CUDA = False

_DEVICE_SESSION = os.environ.get("PYRITE_TEST_BACKEND") == "cuda"


def test_layer_dz_back_exit():
    # photon exits the back face (n_z>0): escape ray spans depths [z_mid, z_total]
    z = np.array([100.0, 500.0, 900.0])
    assert np.allclose(_layer_dz(z, +1.0, 0.0, 500.0), [400.0, 0.0, 0.0])  # film
    assert np.allclose(_layer_dz(z, +1.0, 500.0, 1000.0), [500.0, 500.0, 100.0])  # sub


def test_layer_dz_front_exit():
    # photon exits the entrance face (n_z<0): escape ray spans depths [0, z_mid]
    z = np.array([100.0, 500.0, 900.0])
    assert np.allclose(_layer_dz(z, -1.0, 0.0, 500.0), [100.0, 500.0, 500.0])  # film
    # the substrate is crossed only by a segment that sits INSIDE it (z=900)
    assert np.allclose(_layer_dz(z, -1.0, 500.0, 1000.0), [0.0, 0.0, 400.0])


def test_layer_path_length_keeps_lateral_ray_in_current_layer():
    path = _layer_path_length(np.array([250.0, 750.0]), 0.0, np.array([7.0, 7.0]), 0.0, 500.0)
    np.testing.assert_allclose(path, [7.0, 0.0])


def test_stack_tau_with_side_exit_stops_before_substrate():
    film, sub = [("C", 0.176)], [("W", 0.0632)]
    layers = [(0.0, 500.0, film), (500.0, 1000.0, sub)]
    z, energy = np.array([250.0]), np.array([1500.0])
    tau = _stack_tau(layers, z, 0.0, energy, exit_distance_ang=np.array([10.0]))
    np.testing.assert_allclose(tau, 10.0 * _mu_total_inv_ang(film, energy))


def test_substrate_transparent_to_film_segments_on_front_exit():
    # first-slice geometry point: film-region segments (z < t_film) leaving the
    # front face do NOT cross the substrate behind them (so a negative-tilt,
    # high-flux geometry sees no substrate attenuation of the film lines).
    z_film = np.array([10.0, 100.0, 480.0])
    assert np.allclose(_layer_dz(z_film, -1.0, 500.0, 1000.0), 0.0)


def test_stack_tau_single_layer_matches_slab():
    # one layer over [0, T] must equal the single-slab L_esc * mu, both faces
    z = np.array([200.0, 1500.0, 4000.0])
    E = np.full(3, 1500.0)
    comp = [("Si", 0.04994)]
    T = 5000.0
    mu = _mu_total_inv_ang(comp, E)
    for n_z in (-0.7, +0.7):
        tau = _stack_tau([(0.0, T, comp)], z, n_z, E)
        L_esc = z / (-n_z) if n_z < 0 else (T - z) / n_z
        assert np.allclose(tau, L_esc * mu)


def test_stack_tau_substrate_adds_depth_on_back_exit():
    # adding a substrate behind the film increases tau for a back exit
    z = np.array([100.0, 300.0])
    E = np.full(2, 1500.0)
    film = [("Mo", 0.019), ("Se", 0.038)]
    sub = substrate_composition("sio2")
    tau_film = _stack_tau([(0.0, 500.0, film)], z, +0.7, E)
    tau_stack = _stack_tau([(0.0, 500.0, film), (500.0, 1e6, sub)], z, +0.7, E)
    assert np.all(tau_stack > tau_film)


def test_stack_tau_back_exit_substrate_closed_form():
    # closed form of the slice-1 path integral: on a BACK exit a substrate behind
    # the film adds EXACTLY mu_sub * t_sub / |n_z| to the optical depth of every
    # film segment, independent of emission depth (the film escape path is shared).
    # This is what checks/multilayer_validation_check.py confirms end-to-end as the
    # integrated film-line flux ratio through mc_spectrum.
    z = np.array([50.0, 200.0, 480.0])  # all inside the film [0, 500]
    E = np.full(3, 1438.0)
    film = [("Mo", 0.019), ("Se", 0.038)]
    sub = substrate_composition("sio2")
    t_film, t_sub, n_z = 500.0, 1.5e4, 0.7
    tau_film = _stack_tau([(0.0, t_film, film)], z, n_z, E)
    tau_stack = _stack_tau([(0.0, t_film, film), (t_film, t_film + t_sub, sub)], z, n_z, E)
    mu_sub = _mu_total_inv_ang(sub, E)
    assert np.allclose(tau_stack - tau_film, mu_sub * t_sub / abs(n_z))


def test_substrate_composition_presets_and_crystal():
    sio2 = dict(substrate_composition("sio2"))
    assert sio2["Si"] == pytest.approx(0.02205)
    assert sio2["O"] == pytest.approx(2 * sio2["Si"], rel=1e-3)  # SiO2 stoichiometry
    si = dict(substrate_composition("silicon"))  # crystalline, from CRYSTALS
    assert 0.04 < si["Si"] < 0.06
    sapphire = dict(substrate_composition("sapphire"))  # crystalline, from CRYSTALS
    assert sapphire["O"] == pytest.approx(1.5 * sapphire["Al"])
    with pytest.raises(ValueError):
        substrate_composition("unobtainium")
    with pytest.raises(ValueError):
        substrate_composition("al2o3")


def test_film_on_substrate_layers_structure():
    comp = [("Mo", 0.01), ("Se", 0.02)]
    (z0, z1, c_film), (z2, z3, c_sub) = film_on_substrate_layers(comp, 500.0, "sio2", 1e6)
    assert (z0, z1, z2, z3) == (0.0, 500.0, 500.0, 500.0 + 1e6)
    assert c_film == comp
    assert dict(c_sub)["Si"] == pytest.approx(0.02205)


def test_transport_single_layer_explicit_equals_none():
    # an explicit one-layer stack reproduces the None (single-material) transport
    # bit-for-bit (same RNG stream -> identical segments)
    comp = [("Mo", 0.019), ("Se", 0.038)]
    kw: dict[str, Any] = dict(E_cut_keV=5.0, seed=7)
    a = simulate_trajectories(30.0, 150, 1e4, composition=comp, **kw)
    b = simulate_trajectories(30.0, 150, 1e4, layers=[(0.0, 1e4, comp)], **kw)
    assert a["n_backscattered"] == b["n_backscattered"]
    assert a["L_ang"].size == b["L_ang"].size
    assert np.array_equal(a["L_ang"], b["L_ang"])
    assert np.array_equal(a["r_mid"], b["r_mid"])
    assert b["n_layers"] == 1 and set(b["layer"].tolist()) == {0}


def test_transport_backscatter_increases_with_substrate_Z():
    # a high-Z substrate backscatters more electrons into the thin film, raising
    # the film-region (layer 0) electron path vs a low-Z substrate
    film = [("C", 0.176)]
    t_f = 100.0  # 10 nm film
    stack_lo = [(0.0, t_f, film), (t_f, 1e5, [("C", 0.176)])]
    stack_hi = [(0.0, t_f, film), (t_f, 1e5, [("W", 0.0632)])]
    kw: dict[str, Any] = dict(E_cut_keV=5.0, seed=3, elastic_model="sr")  # SR: no Mott table needed
    lo = simulate_trajectories(30.0, 500, 1e5, layers=stack_lo, **kw)
    hi = simulate_trajectories(30.0, 500, 1e5, layers=stack_hi, **kw)
    film_path_lo = lo["L_ang"][lo["layer"] == 0].sum()
    film_path_hi = hi["L_ang"][hi["layer"] == 0].sum()
    assert film_path_hi > film_path_lo
    assert hi["n_backscattered"] > lo["n_backscattered"]
    assert hi["n_layers"] == 2


def test_build_cases_attaches_abs_layers_only_with_substrate():
    plain = build_cases(Sweep(material="mose2", tilt_deg=30.0, beam=BeamSpec(energy_keV=30.0)))
    assert all(c["abs_layers"] is None for c in plain)

    stacked = build_cases(
        Sweep(
            material="mose2",
            beam=BeamSpec(energy_keV=30.0),
            target=Stack.on_substrate("mose2", 2e4, "sio2", 1e6, tilt_deg=30.0),
        )
    )
    for c in stacked:
        assert c["abs_layers"] is not None and len(c["abs_layers"]) == 2
        (z0, z1, _), (z2, _z3, _) = c["abs_layers"]
        assert z0 == 0.0 and z1 == c["thickness_ang"] and z2 == c["thickness_ang"]
        assert "on sio2" in c["name"]


# ---- slice 3: per-layer coherent radiation -----------------------------------
def test_substrate_radiator_crystalline_vs_amorphous():
    # a crystalline substrate carries its own radiator (crystal params); an
    # amorphous preset radiates no coherent lines (None)
    assert substrate_radiator("sio2") is None
    with pytest.raises(ValueError):
        substrate_radiator("al2o3")
    si = substrate_radiator("silicon")
    assert si is not None
    assert si["crystal"] == "silicon"
    assert set(si) == {"crystal", "hkl_list", "B_ang2", "beam_uvw", "surface_hkl"}
    assert len(si["hkl_list"]) > 0
    sapphire = substrate_radiator("sapphire")
    assert sapphire is not None
    assert sapphire["crystal"] == "sapphire"
    assert set(sapphire) == {"crystal", "hkl_list", "B_ang2", "beam_uvw", "surface_hkl"}
    assert len(sapphire["hkl_list"]) > 0
    with pytest.raises(ValueError):
        substrate_radiator("unobtainium")


def test_build_cases_layer_radiators_match_stack():
    # no substrate -> single slab, no per-layer radiators
    plain = build_cases(Sweep(material="mose2", tilt_deg=30.0, beam=BeamSpec(energy_keV=30.0)))
    assert all(c["layer_radiators"] is None for c in plain)

    # amorphous substrate -> [film radiator, None] (substrate adds no lines)
    amorph = build_cases(
        Sweep(
            material="mose2",
            beam=BeamSpec(energy_keV=30.0),
            target=Stack.on_substrate("mose2", 2e4, "sio2", tilt_deg=30.0),
        )
    )[0]
    film, sub = amorph["layer_radiators"]
    assert sub is None
    # the film radiator must match the case's scalar crystal keys exactly
    assert film["crystal"] == amorph["crystal"]
    assert film["hkl_list"] == amorph["hkl_list"]
    assert film["B_ang2"] == amorph["B_ang2"]
    assert film["beam_uvw"] == amorph["beam_uvw"]

    # crystalline substrate -> [film radiator, substrate radiator]
    cryst = build_cases(
        Sweep(
            material="mose2",
            beam=BeamSpec(energy_keV=30.0),
            target=Stack.on_substrate("mose2", 2e4, "sapphire", tilt_deg=30.0),
        )
    )[0]
    assert cryst["layer_radiators"][1]["crystal"] == "sapphire"
    assert len(cryst["layer_radiators"]) == len(cryst["abs_layers"]) == 2


# ---------------------------------------------------------------------------
# N-layer stacks (Sweep.stack / LayerSpec): the film plus an arbitrary list of
# substrate-side layers, each with its own thickness + crystal orientation.


def test_stack_layers_three_layer_boundaries():
    # film (0..t_f) then each stack layer stacked below with cumulative z bounds
    film = [("Mo", 0.011), ("S", 0.023)]
    layers = stack_layers(film, 100.0, [LayerSpec("sio2", 900.0), LayerSpec("silicon", 5e6)])
    assert [(z0, z1) for z0, z1, _ in layers] == [
        (0.0, 100.0),
        (100.0, 1000.0),
        (1000.0, 1000.0 + 5e6),
    ]
    assert layers[0][2] == [("Mo", 0.011), ("S", 0.023)]
    assert layers[1][2] == substrate_composition("sio2")
    assert layers[2][2] == substrate_composition("silicon")


def test_layer_radiator_orientation_overrides():
    # amorphous layer -> no coherent radiator
    assert layer_radiator(LayerSpec("sio2", 900.0)) is None
    # crystalline layer: beam_uvw + in-plane azimuth are per-layer specifiable
    rad = layer_radiator(LayerSpec("silicon", 5e6, beam_uvw=(1, 1, 1), azimuth_deg=30.0))
    assert rad is not None
    assert rad["crystal"] == "silicon"
    assert rad["beam_uvw"] == (1, 1, 1)
    assert rad["surface_hkl"] is None
    assert rad["azimuth_rad"] == pytest.approx(np.pi / 6)
    # sapphire inherits its catalog c-cut when not overridden, in the catalog's
    # own reciprocal spelling -- the override above clears it, it does not merge
    sap = layer_radiator(LayerSpec("sapphire", 5e6))
    assert sap is not None
    assert sap["surface_hkl"] == (0, 0, 1)
    assert sap["beam_uvw"] is None
    assert sap["azimuth_rad"] == 0.0


def test_build_cases_stack_three_layers():
    sw = Sweep(
        material="mos2",
        thickness_ang=100.0,
        tilt_deg=30.0,
        beam=BeamSpec(energy_keV=30.0),
        stack=(
            LayerSpec("sio2", 900.0),
            LayerSpec("silicon", 5e6, azimuth_deg=15.0),
        ),
    )
    case = build_cases(sw)[0]
    assert len(case["abs_layers"]) == len(case["layer_radiators"]) == 3
    film_rad, oxide, si = case["layer_radiators"]
    assert film_rad["crystal"] == "mos2"
    assert oxide is None
    assert si["crystal"] == "silicon"
    assert si["azimuth_rad"] == pytest.approx(np.deg2rad(15.0))
    assert case["abs_layers"][2][0] == pytest.approx(1000.0)
    assert "on sio2+silicon" in case["name"]


def test_build_cases_substrate_sugar_matches_single_layer_stack():
    kw: dict[str, Any] = dict(material="mose2", tilt_deg=30.0, beam=BeamSpec(energy_keV=30.0))
    with pytest.deprecated_call():
        sugar = build_cases(Sweep(**kw, substrate="sapphire", substrate_thickness_ang=1e6))[0]
    general = build_cases(Sweep(**kw, stack=(LayerSpec("sapphire", 1e6),)))[0]
    helper = build_cases(
        Sweep(
            material="mose2",
            beam=BeamSpec(energy_keV=30.0),
            target=Stack.on_substrate("mose2", 2e4, "sapphire", 1e6, tilt_deg=30.0),
        )
    )[0]
    for key in ("abs_layers", "layer_radiators", "name"):
        assert sugar[key] == general[key] == helper[key]


@pytest.mark.filterwarnings("ignore:Sweep.substrate:DeprecationWarning")
def test_build_cases_rejects_substrate_plus_stack():
    with pytest.raises(ValueError):
        build_cases(
            Sweep(
                material="mose2",
                tilt_deg=30.0,
                beam=BeamSpec(energy_keV=30.0),
                substrate="sio2",
                stack=(LayerSpec("silicon", 5e6),),
            )
        )


def test_spectrum_case_passes_per_layer_azimuth(monkeypatch):
    # each layer's radiator must reach mc_spectrum with ITS OWN azimuth_rad
    from pyrite.montecarlo import runner

    calls = []

    def fake_spec(segs, E_grid, **kw):
        calls.append(kw)
        return np.zeros_like(E_grid)

    monkeypatch.setattr(runner, "mc_spectrum", fake_spec)
    monkeypatch.setattr(runner, "mc_brem_spectrum", lambda *a, **k: np.zeros(3))
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *a, **k: np.zeros(3),
    )

    E = np.linspace(1000.0, 3000.0, 3)
    segs = dict(
        layer=np.array([0, 1, 1]),
        L_ang=np.array([1.0, 2.0, 3.0]),
        n_backscattered=0,
        n_missed=0,
        Ne=1,
        n_layers=2,
    )
    tp = dict(
        E_grid=E,
        E_brem=E,
        n_hat=np.array([0.0, 0.0, 1.0]),
        segs=segs,
        segs_b=segs,
        Ne_lines=1,
        Ne_brem=1,
    )
    comp = [("Mo", 0.011)]
    case = dict(
        crystal="mose2",
        E0_keV=30.0,
        composition=comp,
        hkl_list=[(0, 0, 2)],
        B_ang2=0.6,
        abs_layers=[(0.0, 10.0, comp), (10.0, 20.0, [("Si", 0.05)])],
        layer_radiators=[
            dict(
                crystal="mose2",
                hkl_list=[(0, 0, 2)],
                B_ang2=0.6,
                beam_uvw=(0, 0, 2),
                azimuth_rad=0.0,
            ),
            dict(
                crystal="silicon",
                hkl_list=[(2, 2, 0)],
                B_ang2=0.46,
                beam_uvw=(4, 4, 0),
                azimuth_rad=0.5,
            ),
        ],
    )
    runner._spectrum_case(case, tp)
    assert [c["azimuth_rad"] for c in calls] == [0.0, 0.5]


# ---- CUDA multilayer accumulation -------------------------------------------

_CUDA_ACCUMULATION_RTOL = 2e-5
_CUDA_ACCUMULATION_ATOL_FRACTION = 2e-6


def _cuda_accumulation_case():
    """Small crystalline film/substrate case with both layer radiators active."""
    line = np.arange(100.0, 3500.0, 20.0)
    brem = np.arange(100.0, 30000.0, 200.0)
    sweep = Sweep(
        material="mose2",
        tilt_deg=-30.0,
        beam=BeamSpec(energy_keV=30.0),
        target=Stack.on_substrate("mose2", 300.0, "silicon", 3000.0, tilt_deg=-30.0),
        detector=Detector(energy_bins=EnergyBins(line=line, brem=brem)),
    )
    case = build_cases(sweep, n_electrons=48, n_electrons_brem=24)[0]
    # The manual reference below uses the same bounded launches as the runner;
    # it only changes the association of the four per-layer contributions.
    case["spec_chunk"] = 128
    case["brem_chunk"] = 128
    return case


def _manual_cuda_layer_sum(case, transport):
    """Independently associate coherent + characteristic terms per layer."""
    manual = np.zeros_like(transport["E_grid"], dtype=float)
    for layer, radiator in enumerate(case["layer_radiators"]):
        segments = _segments_in_layer(transport["segs"], layer)
        if segments["L_ang"].size == 0:
            continue
        characteristic = mc_characteristic_spectrum(
            segments,
            transport["E_grid"],
            composition=case["abs_layers"][layer][2],
            n_hat=transport["n_hat"],
            chunk=case["brem_chunk"],
            layers=case["abs_layers"],
            electron_limit=transport["Ne_brem"],
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        )
        coherent = np.zeros_like(manual)
        if radiator is not None:
            coherent = mc_spectrum(
                segments,
                transport["E_grid"],
                crystal=radiator["crystal"],
                hkl_list=radiator["hkl_list"],
                n_hat=transport["n_hat"],
                B_ang2=radiator["B_ang2"],
                composition=case["abs_layers"][layer][2],
                beam_uvw=radiator.get("beam_uvw"),
                surface_hkl=radiator.get("surface_hkl"),
                azimuth_rad=radiator.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
                recip_miscut_rad=radiator.get("recip_miscut_rad", case.get("recip_miscut_rad")),
                sinc_cutoff=case.get("sinc_cutoff"),
                chunk=case["spec_chunk"],
                layers=case["abs_layers"],
                coherent=False,
                electron_limit=transport["Ne_lines"],
                E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            )
        manual += coherent + characteristic
    return manual


@pytest.mark.skipif(
    not _DEVICE_SESSION,
    reason="end-to-end CUDA backend body; driven by the CUDA child-session test",
)
def test_cuda_multilayer_spectrum_case_matches_per_layer_sum():
    """CUDA transport and spectrum retain both crystalline-layer contributions.

    The runner sums coherent layers first and characteristic layers second;
    the reference instead sums each layer's coherent and characteristic terms
    together.  CUDA uses float32 kernels, so their non-associative reductions
    differ by bounded rounding.  ``2e-5`` is roughly 168 float32 epsilons for
    the reduction/association chain; ``2e-6 * peak`` is the absolute floor for
    bins near a cancellation, where a relative comparison alone is unstable.
    """
    from pyrite._backend import REAL, xp

    assert getattr(xp, "__name__", "") == "cupy"
    assert np.dtype(REAL) == np.dtype(np.float32)

    case = _cuda_accumulation_case()
    transport = _transport_case(case, transport_core="cuda", keep_segments_on_device=True)
    assert int(transport["segs"]["n_layers"]) == 2
    assert _segments_in_layer(transport["segs"], 1)["L_ang"].size > 0

    actual = _spectrum_case(case, transport)["spec"]
    expected = _manual_cuda_layer_sum(case, transport)
    peak = float(np.max(np.abs(expected)))
    assert peak > 0.0
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_CUDA_ACCUMULATION_RTOL,
        atol=peak * _CUDA_ACCUMULATION_ATOL_FRACTION,
    )


@pytest.mark.skipif(_DEVICE_SESSION, reason="this is the CUDA child session")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA device required")
def test_cuda_multilayer_spectrum_case_device_suite():
    """Run the end-to-end multilayer body with the CUDA backend selected.

    ``tests/conftest.py`` deliberately pins normal test sessions to CPU.  A
    child process lifts only that pin, then asserts that its selected test
    passed so a missing backend cannot turn this regression into a silent skip.
    """
    env = dict(os.environ)
    env["PYRITE_TEST_BACKEND"] = "cuda"
    env.pop("PYRITE_MC_BACKEND", None)
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            __file__,
            "-k",
            "cuda_multilayer_spectrum_case_matches_per_layer_sum",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(pathlib.Path(__file__).resolve().parents[2]),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout, completed.stdout
