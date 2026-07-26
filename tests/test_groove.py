import importlib

import numpy as np
import pytest

from cxr_mc.materials.attenuation import _mu_total_inv_ang
from cxr_mc.montecarlo import _to_cpu
from cxr_mc.montecarlo._backend import REAL
from cxr_mc.montecarlo.geometry import X_MAX, Z_MAX
from cxr_mc.montecarlo.groove import (
    blazed_groove_spec,
    entry_points,
    escape_distance_ang,
    first_surface_event,
    in_material,
    surface_depth_ang,
)
from cxr_mc.montecarlo.spectrum import _brem_dsigma_dk, mc_brem_spectrum
from cxr_mc.montecarlo.transport import (
    TRANSPORT_ELEMENTS,
    beta_from_keV,
    simulate_trajectories,
)

TP = np.deg2rad(45.0)
SPEC = blazed_groove_spec(
    spacing_ang=2.0e4,
    theta_obs_rad=np.pi / 2,
    tilt_polar_rad=TP,
    tilt_azim_rad=np.pi,
)
SHALLOW_SPEC = blazed_groove_spec(
    spacing_ang=2.0,
    theta_obs_rad=np.pi / 2,
    tilt_polar_rad=TP,
    tilt_azim_rad=np.pi,
)
MARCH_SPEC = blazed_groove_spec(
    spacing_ang=2.0,
    theta_obs_rad=np.pi / 2,
    tilt_polar_rad=TP,
    tilt_azim_rad=np.pi,
)
_transport_module = importlib.import_module("cxr_mc.montecarlo.transport")
_SPECTRUM_RTOL = max(2e-12, 3.0 * float(np.finfo(REAL).eps))


def _z_surf(x, spec):
    """Brute-force sawtooth profile height (depth of the surface) at x."""
    tp = spec.tilt_polar_rad
    lam, h = spec.spacing_ang, spec.depth_ang
    u = np.mod(x, lam)
    x_valley = h * np.tan(tp)
    return np.where(u <= x_valley, u / np.tan(tp), (lam - u) * np.tan(tp))


def _march_escape(x, z, spec, ds=None):
    """Independent state bracket plus bisection along n_hat."""
    tp = spec.tilt_polar_rad
    n = np.array([np.cos(tp), -np.sin(tp)])
    p = np.array([x, z], dtype=float)
    ds = 0.02 * spec.spacing_ang if ds is None else ds

    def inside(s):
        q = p + s * n
        return bool(q[1] > 0.0 and q[1] >= _z_surf(q[0], spec))

    lo, hi = 0.0, ds
    for _ in range(10_000):
        if not inside(hi):
            break
        lo, hi = hi, hi + ds
    else:
        raise AssertionError("independent escape bracket not found")
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return hi


def _march_transition(position, direction, spec, transition, ds=2.5e-5):
    """Independent fine-step reference with bisection at a state transition."""

    def inside(s):
        q = position + s * direction
        return bool(q[2] >= _z_surf(q[0], spec) and q[2] <= 1e6)

    lo = 1e-6
    before = inside(lo)
    for _ in range(400_000):
        hi = lo + ds
        after = inside(hi)
        matched = (before and not after) if transition == "exit" else (not before and after)
        if matched:
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if inside(mid) == before:
                    lo = mid
                else:
                    hi = mid
            return hi
        lo = hi
        before = after
    return np.inf


def test_surface_depth_matches_both_analytic_facets():
    x = np.array([0.0, SPEC.depth_ang * np.tan(TP), SPEC.spacing_ang])
    np.testing.assert_allclose(surface_depth_ang(x, SPEC), [0.0, SPEC.depth_ang, 0.0])


def test_material_predicate_includes_surface_and_excludes_groove_void():
    x = 0.25 * SPEC.spacing_ang
    z = surface_depth_ang(x, SPEC)
    assert in_material(np.array([x, 0.0, z]), 1e6, SPEC)
    assert not in_material(np.array([x, 0.0, z - 1e-5]), 1e6, SPEC)


def test_surface_event_exit_then_later_entry_matches_reference_march():
    p = np.array(
        [
            0.25 * MARCH_SPEC.spacing_ang,
            0.0,
            0.75 * MARCH_SPEC.depth_ang,
        ]
    )
    d = np.array([1.0, 0.0, -0.2])
    d /= np.linalg.norm(d)
    s_exit = first_surface_event(p, d, MARCH_SPEC, transition="exit")
    p_vac = p + s_exit * d
    s_entry = first_surface_event(p_vac, d, MARCH_SPEC, transition="entry")
    np.testing.assert_allclose(
        s_exit,
        _march_transition(p, d, MARCH_SPEC, "exit"),
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        s_entry,
        _march_transition(p_vac, d, MARCH_SPEC, "entry"),
        rtol=2e-4,
    )


def test_tangent_surface_event_is_skipped():
    p = np.array([0.0, 0.0, 0.0])
    d = np.array([np.sin(TP), 0.0, np.cos(TP)])
    working_normal = np.array([np.cos(TP), 0.0, -np.sin(TP)])
    assert abs(np.dot(working_normal, d)) <= 8 * np.finfo(float).eps
    assert np.isinf(first_surface_event(p, d, SPEC, transition="exit"))


def test_surface_event_accepts_one_ulp_above_valley_band_endpoint():
    spec = blazed_groove_spec(
        spacing_ang=2.0,
        theta_obs_rad=np.pi / 2,
        tilt_polar_rad=0.61,
        tilt_azim_rad=np.pi,
    )
    valley_x = spec.depth_ang * np.tan(spec.tilt_polar_rad)
    p = np.array([valley_x, 0.0, spec.depth_ang + 1.0])
    d = np.array([0.0, 0.0, -1.0])

    event = first_surface_event(p, d, spec, transition="exit")

    assert event == pytest.approx(1.0, abs=8 * np.finfo(float).eps)


def test_surface_event_preserves_tiny_nonzero_vertical_direction():
    tiny_dz = -0.5 * 32 * np.finfo(float).eps
    z0 = np.nextafter(SPEC.depth_ang, np.inf)
    s_to_depth = (z0 - SPEC.depth_ang) / -tiny_dz
    x_valley = SPEC.depth_ang * np.tan(TP)
    p = np.array([x_valley - 1.0 - s_to_depth, 0.0, z0])
    d = np.array([1.0, 0.0, tiny_dz])
    expected = (SPEC.spacing_ang + p[2] - p[0]) / (d[0] - d[2])

    event = first_surface_event(p, d, SPEC, transition="exit")

    assert event == pytest.approx(expected, rel=1e-12)


def test_surface_event_preserves_tiny_nonzero_facet_rate():
    st, ct = np.sin(TP), np.cos(TP)
    tangent = np.array([st, 0.0, ct])
    working_normal = np.array([ct, 0.0, -st])
    d = tangent + 0.75 * 32 * np.finfo(float).eps * working_normal
    d /= np.linalg.norm(d)
    expected = 5.0e3
    q = np.array([0.25 * SPEC.spacing_ang, 0.0, 0.25 * SPEC.spacing_ang])
    p = q - expected * d
    rate = np.dot(working_normal, d)
    assert 0.0 < rate <= 32 * np.finfo(float).eps

    event = first_surface_event(p, d, SPEC, transition="exit")

    assert event == pytest.approx(expected, rel=2e-2)


def test_depth_closes_unit_cell():
    assert SPEC.depth_ang == pytest.approx(SPEC.spacing_ang * np.sin(TP) * np.cos(TP))


def test_escape_matches_ray_march():
    rng = np.random.default_rng(7)
    lam, h = MARCH_SPEC.spacing_ang, MARCH_SPEC.depth_ang
    x = rng.uniform(0.0, 3 * lam, 200)
    z = rng.uniform(0.0, 4 * h, 200)
    inside = z >= _z_surf(x, MARCH_SPEC) + 1e-6
    x, z = x[inside], z[inside]
    L = escape_distance_ang(x, z, MARCH_SPEC)
    ref = np.array([_march_escape(xi, zi, MARCH_SPEC) for xi, zi in zip(x, z, strict=False)])
    np.testing.assert_allclose(
        L,
        ref,
        rtol=5e-11,
        atol=5e-11 * MARCH_SPEC.spacing_ang,
    )


def test_escape_flat_limit_small_depth():
    tiny = blazed_groove_spec(
        spacing_ang=1.0,
        theta_obs_rad=np.pi / 2,
        tilt_polar_rad=TP,
        tilt_azim_rad=np.pi,
    )
    z = np.array([5.0e4])
    L = escape_distance_ang(np.array([1234.5]), z, tiny)
    assert L[0] == pytest.approx(z[0] / np.sin(TP), rel=1e-3)


def test_escape_never_exceeds_flat_path_and_nonnegative():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 5 * SPEC.spacing_ang, 500)
    z = rng.uniform(SPEC.depth_ang, 10 * SPEC.depth_ang, 500)
    L = escape_distance_ang(x, z, SPEC)
    assert np.all(L >= 0.0)
    assert np.all(L <= z / np.sin(TP) + 1e-9)


def test_entry_points_land_on_relief_facet():
    rng = np.random.default_rng(11)
    x0 = rng.uniform(-3 * SPEC.spacing_ang, 3 * SPEC.spacing_ang, 300)
    xe, ze = entry_points(x0, SPEC)
    assert np.all((ze >= 0.0) & (ze < SPEC.depth_ang))
    # every entry point lies ON the surface profile
    assert np.allclose(ze, _z_surf(xe, SPEC), atol=1e-6)
    # displacement is along the beam direction
    assert np.allclose((xe - x0) / np.sin(TP) * np.cos(TP), ze, atol=1e-6)


def test_blazed_groove_spec_validates_geometry():
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.deg2rad(119.0), TP, np.pi)  # theta_obs != 90
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.pi / 2, TP, 0.0)  # azim != 180
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.pi / 2, 0.0, np.pi)  # tp == 0
    with pytest.raises(ValueError):
        blazed_groove_spec(-1.0, np.pi / 2, TP, np.pi)  # spacing <= 0


_SIM_KW = dict(
    E0_keV=60.0,
    Ne=200,
    thickness_ang=2.0e5,
    element="C",
    n_atoms_per_ang3=0.1136,
    seed=42,
    elastic_model="sr",
)


def _tilt_kw():
    from cxr_mc.montecarlo.geometry import tilted_geometry

    beam, _ = tilted_geometry(np.pi / 2, TP, np.pi)
    return dict(beam_dir=beam, tilt_polar_rad=TP, tilt_azim_rad=np.pi)


def test_groove_none_is_bitwise_identical():
    a = simulate_trajectories(**_SIM_KW, **_tilt_kw())
    b = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=None)
    for k in ("E_keV", "L_ang", "t_ang", "elec_id"):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(a["r_mid"], b["r_mid"])


def test_groove_entries_start_on_relief_facet():
    segs = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)
    # first segment of each electron starts at its entry point; entry depths
    # span [0, h) and multiple phases are sampled (uniform-phase fallback).
    # elec_id is NOT globally sorted (each event-driven iteration only appends
    # a locally-increasing subset of still-alive ids), so recover each
    # electron's first-occurrence index directly rather than via searchsorted.
    _, first = np.unique(segs["elec_id"], return_index=True)
    r0 = segs["r_mid"][first]  # midpoints of first segments sit at z >= z_entry
    assert np.all(r0[:, 2] >= 0.0)
    # phases genuinely vary across electrons
    assert np.unique(np.round(r0[:, 0], 3)).size > 10


def test_groove_entry_depth_reproducible_with_seed():
    a = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)
    b = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)
    np.testing.assert_array_equal(a["r_mid"], b["r_mid"])


def test_groove_transport_records_only_material_segments_and_vacuum_invariants():
    out = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)

    assert {
        "vacuum_start_ang",
        "vacuum_end_ang",
        "vacuum_E_keV",
        "vacuum_t_ang",
        "vacuum_elec_id",
    } <= out.keys()
    assert np.all(in_material(out["r_mid"], out["thickness_ang"], SPEC))
    assert out["vacuum_start_ang"].shape == out["vacuum_end_ang"].shape
    assert out["vacuum_start_ang"].shape[1:] == (3,)
    assert out["vacuum_start_ang"].shape[0] > 0
    assert np.all(out["vacuum_E_keV"] > 0.0)
    assert np.all(out["vacuum_elec_id"] >= 0)
    assert np.all(out["vacuum_elec_id"] < _SIM_KW["Ne"])


def test_vacuum_leg_preserves_energy_direction_and_advances_clock():
    out = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=SPEC)

    for vac_i, electron in enumerate(out["vacuum_elec_id"]):
        later = np.flatnonzero(
            (out["elec_id"] == electron) & (out["t_ang"] > out["vacuum_t_ang"][vac_i])
        )
        assert later.size
        next_i = later[np.argmin(out["t_ang"][later])]
        length = np.linalg.norm(out["vacuum_end_ang"][vac_i] - out["vacuum_start_ang"][vac_i])
        np.testing.assert_allclose(
            out["t_ang"][next_i],
            out["vacuum_t_ang"][vac_i] + length / beta_from_keV(out["vacuum_E_keV"][vac_i]),
            rtol=2e-13,
        )
        assert out["E_keV"][next_i] == pytest.approx(out["vacuum_E_keV"][vac_i])
        np.testing.assert_allclose(
            out["v_hat"][next_i],
            (out["vacuum_end_ang"][vac_i] - out["vacuum_start_ang"][vac_i]) / length,
            rtol=2e-13,
            atol=2e-13,
        )


def test_groove_none_preserves_legacy_arrays_bit_for_bit():
    old = simulate_trajectories(**_SIM_KW, **_tilt_kw())
    explicit = simulate_trajectories(**_SIM_KW, **_tilt_kw(), groove=None)

    assert old.keys() == explicit.keys()
    for key in old:
        np.testing.assert_equal(old[key], explicit[key])
    assert explicit["vacuum_start_ang"].shape == (0, 3)
    assert explicit["vacuum_elec_id"].dtype == np.int64


def _one_electron_transport(**overrides):
    kwargs = dict(
        E0_keV=60.0,
        Ne=1,
        thickness_ang=100.0,
        element="C",
        n_atoms_per_ang3=1e-12,
        E_cut_keV=5.0,
        seed=9,
        max_steps=4,
        elastic_model="sr",
        beam_dir=np.array([0.0, 0.0, 1.0]),
        groove=SHALLOW_SPEC,
    )
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


def test_surface_cutoff_stops_before_vacuum_reentry(monkeypatch):
    transitions = []

    def surface_event(_position, _direction, _spec, transition=None):
        transitions.append(transition)
        if transition == "entry":
            raise AssertionError("cutoff electron searched for vacuum re-entry")
        return 1.0

    monkeypatch.setattr(_transport_module, "first_surface_event", surface_event)
    out = _one_electron_transport(
        n_atoms_per_ang3=0.1136,
        E_cut_keV=60.0,
    )

    assert transitions == ["exit"]
    assert out["vacuum_start_ang"].shape == (0, 3)
    assert out["n_backscattered"] == 0
    assert out["n_stopped"] == 1


def test_permanent_surface_exit_counts_backscatter(monkeypatch):
    monkeypatch.setattr(
        _transport_module,
        "first_surface_event",
        lambda _p, _d, _spec, transition=None: 0.5 if transition == "exit" else np.inf,
    )

    out = _one_electron_transport()

    assert out["n_backscattered"] == 1
    assert out["n_transmitted"] == out["n_side_exited"] == 0
    assert out["n_stopped"] == 0
    assert out["vacuum_start_ang"].shape == (0, 3)


def test_surface_reentry_does_not_consume_material_step_budget(monkeypatch):
    exit_calls = 0

    def surface_event(_position, _direction, _spec, transition=None):
        nonlocal exit_calls
        if transition == "entry":
            return 1.0
        exit_calls += 1
        return 0.5 if exit_calls == 1 else np.inf

    monkeypatch.setattr(_transport_module, "first_surface_event", surface_event)

    out = _one_electron_transport(max_steps=1)

    assert out["vacuum_start_ang"].shape == (1, 3)
    assert out["n_transmitted"] == 1
    assert out["n_backscattered"] == out["n_side_exited"] == out["n_stopped"] == 0


def test_surface_event_exhaustion_raises_instead_of_classifying_survivor_stopped(
    monkeypatch,
):
    monkeypatch.setattr(
        _transport_module,
        "first_surface_event",
        lambda _p, _d, _spec, transition=None: 0.5 if transition == "exit" else 1.0,
    )

    with pytest.raises(RuntimeError, match="grooved surface event limit exhausted"):
        _one_electron_transport(max_steps=1)


def test_reentry_resumes_material_stopping_and_elastic_scattering(monkeypatch):
    exit_calls = 0

    def surface_event(_position, _direction, _spec, transition=None):
        nonlocal exit_calls
        if transition == "entry":
            return 1.0
        exit_calls += 1
        return 0.5 if exit_calls == 1 else np.inf

    monkeypatch.setattr(_transport_module, "first_surface_event", surface_event)

    out = _one_electron_transport(
        thickness_ang=1e6,
        n_atoms_per_ang3=0.1136,
        max_steps=3,
    )

    assert out["vacuum_start_ang"].shape == (1, 3)
    vacuum_direction = out["vacuum_end_ang"][0] - out["vacuum_start_ang"][0]
    vacuum_direction /= np.linalg.norm(vacuum_direction)
    assert out["E_keV"][1] == pytest.approx(out["vacuum_E_keV"][0])
    np.testing.assert_allclose(out["v_hat"][1], vacuum_direction)
    assert out["E_keV"][2] < out["E_keV"][1]
    assert not np.allclose(out["v_hat"][2], out["v_hat"][1])


def test_repeated_zero_length_surface_events_raise(monkeypatch):
    monkeypatch.setattr(
        _transport_module,
        "first_surface_event",
        lambda _p, _d, _spec, transition=None: 0.5e-6 if transition == "exit" else 1.0,
    )

    with pytest.raises(RuntimeError, match="repeated zero-length grooved surface events"):
        _one_electron_transport(max_steps=2)


def test_layer_and_back_face_events_precede_far_surface(monkeypatch):
    monkeypatch.setattr(
        _transport_module,
        "first_surface_event",
        lambda _p, _d, _spec, transition=None: 100.0,
    )
    layers = [
        (0.0, 2.0, [("C", 1e-12)]),
        (2.0, 4.0, [("C", 1e-12)]),
    ]

    out = _one_electron_transport(layers=layers)

    assert set(out["layer"]) == {0, 1}
    assert out["n_transmitted"] == 1
    assert out["n_backscattered"] == out["n_side_exited"] == 0
    assert out["vacuum_start_ang"].shape == (0, 3)


def test_finite_side_exit_before_reentry_records_no_vacuum_leg(monkeypatch):
    prism_calls = 0

    def prism_exit(*_args, **_kwargs):
        nonlocal prism_calls
        prism_calls += 1
        if prism_calls == 1:
            return np.array([50.0]), np.array([Z_MAX])
        return np.array([0.25]), np.array([X_MAX])

    monkeypatch.setattr(_transport_module, "first_prism_exit", prism_exit)
    monkeypatch.setattr(
        _transport_module,
        "first_surface_event",
        lambda _p, _d, _spec, transition=None: 0.5 if transition == "exit" else 1.0,
    )

    out = _one_electron_transport(
        crystal_width_mm=1e-5,
        crystal_height_mm=1e-5,
    )

    assert out["n_side_exited"] == 1
    assert out["n_backscattered"] == out["n_transmitted"] == 0
    assert out["vacuum_start_ang"].shape == (0, 3)


from cxr_mc.montecarlo.spectrum import mc_spectrum


def _hopg_spectrum(groove=None, thickness_ang=2.0e5):
    from cxr_mc.montecarlo.geometry import tilted_geometry

    beam, n_hat = tilted_geometry(np.pi / 2, TP, np.pi)
    segs = simulate_trajectories(
        E0_keV=60.0,
        Ne=400,
        thickness_ang=thickness_ang,
        element="C",
        n_atoms_per_ang3=0.1136,
        seed=42,
        elastic_model="sr",
        beam_dir=beam,
        tilt_polar_rad=TP,
        tilt_azim_rad=np.pi,
        groove=groove,
    )
    E_grid = np.linspace(500.0, 3000.0, 400)
    return mc_spectrum(
        segs,
        E_grid,
        crystal="hopg",
        hkl_list=[(0, 0, 2)],
        n_hat=n_hat,
        B_ang2=0.0,
        composition=[("C", 0.1136)],
        groove=groove,
    )


def test_spectrum_groove_none_bitwise():
    np.testing.assert_array_equal(_hopg_spectrum(), _hopg_spectrum(groove=None))


def test_spectrum_groove_transport_is_deterministic():
    flat = _hopg_spectrum()
    grooved = _hopg_spectrum(groove=SPEC)
    repeated = _hopg_spectrum(groove=SPEC)

    assert np.all(np.isfinite(grooved))
    assert grooved.sum() > 0.0
    assert not np.array_equal(grooved, flat)
    np.testing.assert_array_equal(grooved, repeated)


def test_spectrum_groove_rejects_layers():
    from cxr_mc.montecarlo.geometry import tilted_geometry

    _, n_hat = tilted_geometry(np.pi / 2, TP, np.pi)
    segs = simulate_trajectories(
        E0_keV=60.0,
        Ne=10,
        thickness_ang=1.0e4,
        element="C",
        n_atoms_per_ang3=0.1136,
        seed=1,
        elastic_model="sr",
    )
    with pytest.raises(ValueError):
        mc_spectrum(
            segs,
            np.linspace(500.0, 3000.0, 50),
            crystal="hopg",
            hkl_list=[(0, 0, 2)],
            n_hat=n_hat,
            B_ang2=0.0,
            composition=[("C", 0.1136)],
            groove=SPEC,
            layers=[(0.0, 1.0e4, [("C", 0.1136)])],
        )


def test_escape_gain_matches_analytic_mean():
    """Uniform-phase emitters at fixed deep z: mean exp(-L/L_abs) boost equals
    the closed-form sawtooth average (path savings uniform in [0, h)/sin tp)."""
    rng = np.random.default_rng(5)
    L_abs = 6.0e4  # [Ang] ~ HOPG (002) scale
    z = np.full(20000, 8.0e4)
    x = rng.uniform(0.0, 50 * SPEC.spacing_ang, z.size)
    L = escape_distance_ang(x, z, SPEC)
    t_grooved = np.exp(-L / L_abs).mean()
    t_flat = np.exp(-(z[0] / np.sin(TP)) / L_abs)
    h, sg = SPEC.depth_ang, np.sin(TP)
    analytic = (sg * L_abs / h) * np.expm1(h / (sg * L_abs))
    assert t_grooved / t_flat == pytest.approx(analytic, rel=1e-2)


def _brem_segments():
    return {
        "r_mid": np.array(
            [
                [1.0e3, 0.0, 1.5e4],
                [6.0e3, 0.0, 2.0e4],
                [1.1e4, 0.0, 2.5e4],
            ]
        ),
        "L_ang": np.array([800.0, 1200.0, 600.0]),
        "E_keV": np.array([30.0, 24.0, 18.0]),
        "Ne": 3,
        "thickness_ang": 1.0e5,
    }


def _independent_brem_with_escape(segments, grid, escape_ang, *, composition):
    """Direct segment sum with caller-supplied Beer--Lambert path lengths."""
    mu = _mu_total_inv_ang(composition, grid)
    transmission = np.exp(-np.asarray(escape_ang)[:, None] * mu[None, :])
    spectrum = np.zeros(grid.size)
    path_cm = segments["L_ang"] * 1e-8
    for element, number_density in composition:
        dsigma = np.asarray(
            _to_cpu(
                _brem_dsigma_dk(
                    TRANSPORT_ELEMENTS[element]["Z"],
                    segments["E_keV"],
                    grid,
                )
            )
        )
        spectrum += (number_density * 1e24 * path_cm) @ (dsigma * transmission)
    return spectrum / (4.0 * np.pi) / segments["Ne"]


def test_brem_groove_gain_matches_beer_lambert_escape():
    segments = _brem_segments()
    grid = np.linspace(700.0, 5000.0, 32)
    composition = [("C", 0.1136)]
    n_hat = np.array([np.cos(TP), 0.0, -np.sin(TP)])

    flat = mc_brem_spectrum(segments, grid, composition=composition, n_hat=n_hat)
    grooved = mc_brem_spectrum(
        segments,
        grid,
        composition=composition,
        n_hat=n_hat,
        groove=SPEC,
    )
    expected = _independent_brem_with_escape(
        segments,
        grid,
        escape_distance_ang(
            segments["r_mid"][:, 0],
            segments["r_mid"][:, 2],
            SPEC,
        ),
        composition=composition,
    )

    np.testing.assert_allclose(grooved, expected, rtol=_SPECTRUM_RTOL)
    assert np.all(grooved >= flat)


def test_brem_groove_rejects_layers_and_wrong_direction():
    segments = _brem_segments()
    grid = np.linspace(700.0, 5000.0, 8)
    composition = [("C", 0.1136)]
    n_hat = np.array([np.cos(TP), 0.0, -np.sin(TP)])

    with pytest.raises(ValueError):
        mc_brem_spectrum(
            segments,
            grid,
            composition=composition,
            n_hat=n_hat,
            groove=SPEC,
            layers=[(0.0, 1.0e5, composition)],
        )
    with pytest.raises(ValueError):
        mc_brem_spectrum(
            segments,
            grid,
            composition=composition,
            n_hat=[0.0, 0.0, -1.0],
            groove=SPEC,
        )
