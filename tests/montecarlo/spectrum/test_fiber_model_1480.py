"""Phase II full-fiber model: domain, tensor, intensity and subtraction parity."""

import numpy as np
import pytest

from pyrite.config import CxrReflections, FiberAverage, Numerics
from pyrite.materials.channeling.thermal import thermal_site
from pyrite.materials.crystal import (
    CRYSTALS,
    dominant_reflections,
    emission_coupling_tables,
)
from pyrite.montecarlo.spectrum.cxr import fiber
from pyrite.montecarlo.spectrum.cxr.fiber_case import score_fiber_case

_NUMERICS = ("cutoff_invang", "azimuth_count", "reflection_batch_size")


def _model(**kwargs):
    numerics = {"cutoff_invang": 4.0, "azimuth_count": 4} | {
        k: kwargs.pop(k) for k in _NUMERICS if k in kwargs
    }
    model = dict(temperature_K=thermal_site("hopg", "C").temperature_K) | kwargs
    return fiber.FiberCXRModel(FiberAverage(**numerics), **model)


def test_full_domain_is_opt_in_and_thermal_projection():
    domain = _model().prepare()
    site = thermal_site("hopg", "C")
    temperature = domain.model.temperature_K
    values = domain._thermal_lookup()
    assert CRYSTALS["hopg"]["texture"] == "fiber"
    assert all(h[:2] == (0, 0) for h in dominant_reflections("hopg", g_max_invang=4.0))
    assert (0, 0, 2) in values
    assert (1, 0, 0) in values
    assert values[(0, 0, 2)] == pytest.approx(site.b_factor_along(1.0, temperature))
    assert values[(1, 0, 0)] == pytest.approx(site.b_factor_along(0.0, temperature))
    assert all(values[tuple(-i for i in h)] == b for h, b in values.items())


def test_tensor_batched_couplings_equal_individual_scalar_rows():
    domain = _model().prepare()
    lookup = domain._thermal_lookup()
    hkls = [(0, 0, 2), (1, 0, 0), (-1, 0, 0)]
    energy = np.array([500.0, 1000.0, 2000.0])
    batched = emission_coupling_tables("hopg", hkls, energy, lookup.__getitem__)
    for row, hkl in enumerate(hkls):
        scalar = emission_coupling_tables("hopg", [hkl], energy, lookup[hkl])
        for actual, expected in zip(batched, scalar, strict=True):
            np.testing.assert_allclose(actual[row], expected[0], rtol=1e-13, atol=0)


def test_intensity_average_batching_and_components(monkeypatch):
    domain = _model(reflection_batch_size=2).prepare()
    calls = []

    def scorer(segments, grid, crystal, hkls, *, B_ang2, azimuth_rad, **options):
        assert crystal == "hopg"
        assert len(hkls) <= 2
        calls.extend((h, azimuth_rad) for h in hkls)
        value = sum(B_ang2(h) * (2 + np.cos(6 * azimuth_rad)) for h in hkls)
        return tuple(np.full(len(grid), x * value) for x in (1, 2, 3))

    monkeypatch.setattr(fiber, "mc_cxr_spectrum", scorer)
    result = domain.spectrum({}, np.array([100.0, 200.0]), components=True)
    expected = sum(b * (3 if h[:2] == (0, 0) else 2) for h, b in domain._thermal_lookup().items())
    np.testing.assert_allclose(result, expected * np.array([[1, 1], [2, 2], [3, 3]]))
    for h in domain.hkls:
        assert sum(row == h for row, _ in calls) == (1 if h[:2] == (0, 0) else 4)


def test_shell_weights_match_scalar_reflection_sum():
    from pyrite.montecarlo.spectrum.brem_lattice_share import reflection_shell_weights

    domain = _model().prepare()
    g, weights = domain.shell_weights([("C", 1.0)])
    expected = {}
    for h, b in domain._thermal_lookup().items():
        shell, value = reflection_shell_weights("hopg", [h], b, [("C", 1.0)])
        expected[shell[0]] = expected.get(shell[0], 0.0) + value["C"][0]
    np.testing.assert_array_equal(g, sorted(expected))
    np.testing.assert_allclose(weights["C"], [expected[x] for x in g], rtol=1e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"azimuth_count": 0},
        {"azimuth_count": 2.5},
        {"crystal": "hbn"},
        {"cutoff_invang": float("nan")},
        {"temperature_K": 10000},
    ],
)
def test_invalid_model_refuses(kwargs):
    with pytest.raises(ValueError):
        _model(**kwargs)


def test_unsupported_spectrum_options_refuse():
    domain = _model().prepare()
    for options in (
        {"coherent": True},
        {"surface_hkl": (1, 0, 0)},
        {"truncation_audit": {}},
        {"mosaic_sampling": "history"},
    ):
        with pytest.raises(ValueError):
            domain.spectrum({}, [100.0, 200.0], **options)


def test_paired_case_refuses_different_electron_populations():
    case = dict(crystal="hopg", bremsstrahlung_model="bremslib")
    tp = dict(segs={"Ne": 5}, Ne_lines=5, Ne_brem=4)
    with pytest.raises(ValueError, match="same complete electron population"):
        score_fiber_case(_model(), case, tp, [100.0, 200.0])


def test_correction_uses_one_combined_shell_sum(monkeypatch):
    from types import SimpleNamespace

    from pyrite.montecarlo.runner import lattice_share

    domain = _model().prepare()
    g, weights = domain.shell_weights([("C", 1.0)])
    kernel = SimpleNamespace(g_invang=np.array([0.1, 10.0]), T_keV=np.array([1.0, 1000.0]))
    monkeypatch.setattr(lattice_share, "_kernel", lambda: kernel)
    calls = []
    base, corrected = object(), object()

    def subtract(entries, used_kernel, magnitudes, *, T_floor_keV):
        assert used_kernel is kernel and T_floor_keV == 5.0
        assert len(entries) == 1 and entries[0][:3] == (base, "C", 1.0)
        np.testing.assert_array_equal(magnitudes, g)
        np.testing.assert_array_equal(entries[0][3], weights["C"])
        calls.append(1)
        return [(corrected, SimpleNamespace(removed_fraction=0.1, clamped_nodes=2))]

    monkeypatch.setattr(fiber, "_lattice_share_tables", subtract)
    tables, record = domain.corrected_bremslib_tables(
        {"C": base},
        [("C", 1.0)],
        E_cut_keV=5.0,
        max_energy_keV=300.0,
    )
    assert tables["C"] is corrected and calls == [1]
    assert record["elements"][0]["clamped_nodes"] == 2
    with pytest.raises(ValueError, match="energy interval"):
        domain.corrected_bremslib_tables(
            {"C": base},
            [("C", 1.0)],
            E_cut_keV=5.0,
            max_energy_keV=2000.0,
        )


def test_paired_case_reuses_scene_and_domain(monkeypatch):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_tables, emission

    model = _model(reflection_batch_size=3)
    domain = model.prepare()
    seen = []
    segments = {"Ne": 4, "E_keV": np.array([30.0])}
    n_hat = np.array([0.0, 0.0, -1.0])
    grid = np.array([100.0, 200.0])
    case = dict(
        crystal="hopg",
        E0_keV=30.0,
        composition=[("C", 1.0)],
        bremsstrahlung_model="bremslib",
        hkl_list=[(0, 0, 2)],
        B_ang2=1.3,
    )
    tp = dict(segs=segments, Ne_lines=4, Ne_brem=4, n_hat=n_hat)
    base, corrected = {"C": object()}, {"C": object()}
    monkeypatch.setattr(case_tables, "_case_bremslib_tables", lambda _: base)
    monkeypatch.setattr(runner, "_segments_on_device", lambda x: x)

    def correction(self, tables, composition, **options):
        assert self.hkls == domain.hkls and self.b_ang2 == domain.b_ang2
        assert tables is base and options == dict(E_cut_keV=5.0, max_energy_keV=30.0)
        return corrected, {"corrected": True}

    monkeypatch.setattr(fiber.FiberReflectionSet, "corrected_bremslib_tables", correction)

    def line(segs, axis, pilot, direction, abs_layers, groove, **options):
        assert segs is segments and direction is n_hat and options["Ne"] == 4
        assert pilot["reflection_selection"] is None and options["coherent"] is False
        np.testing.assert_array_equal(axis, grid)
        for h in pilot["hkl_list"]:
            assert pilot["B_ang2"](h) == domain._thermal_lookup()[h]
            seen.append(h)
        return np.full(grid.shape, len(pilot["hkl_list"]), dtype=float)

    def continuum(segs, axis, pilot, direction, abs_layers, **options):
        assert segs is segments and direction is n_hat and options["Ne"] == 4
        assert options["tables"] is corrected
        np.testing.assert_array_equal(axis, grid)
        return np.ones(grid.shape)

    monkeypatch.setattr(runner, "_cxr_for_segments", line)
    monkeypatch.setattr(emission, "_layer_brem_from_segments", continuum)
    result = score_fiber_case(model, case, tp, grid)
    np.testing.assert_allclose(result["cxr"], len(domain.hkls))
    np.testing.assert_allclose(result["total"], len(domain.hkls) + 1)
    assert set(seen) == set(domain.hkls)
    assert case["hkl_list"] == [(0, 0, 2)] and case["B_ang2"] == 1.3


def test_roll_nodes_are_distinct_modulo_the_sixfold_period(monkeypatch):
    """Full-circle nodes 2 pi i/6 would collapse to one orientation."""
    domain = _model(azimuth_count=6).prepare()
    rolls = set()

    def scorer(segments, grid, crystal, hkls, *, B_ang2, azimuth_rad, **options):
        if any(h[:2] != (0, 0) for h in hkls):
            rolls.add(round(azimuth_rad % (np.pi / 3), 12))
        return np.ones(len(grid))

    monkeypatch.setattr(fiber, "mc_cxr_spectrum", scorer)
    domain.spectrum({}, np.array([100.0, 200.0]))
    assert len(rolls) == 6


def test_reflection_set_must_be_roll_closed():
    domain = _model().prepare()
    keep = [i for i, h in enumerate(domain.hkls) if h != (1, 0, 2) and h != (-1, 0, -2)]
    with pytest.raises(ValueError, match="60-degree rolls"):
        fiber.FiberReflectionSet(
            domain.model,
            tuple(domain.hkls[i] for i in keep),
            tuple(domain.b_ang2[i] for i in keep),
        )


@pytest.mark.parametrize("cutoff", [6.0, 8.0, 10.0])
def test_wider_domains_prepare_roll_closed(cutoff):
    """Projected B of roll mates differs only by rounding; closure must hold."""
    domain = _model(cutoff_invang=cutoff).prepare()
    assert len(domain.hkls) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cutoff_invang": 0.0},
        {"cutoff_invang": True},
        {"azimuth_count": 0},
        {"reflection_batch_size": 1.5},
    ],
)
def test_fiber_average_validates(kwargs):
    with pytest.raises(ValueError):
        FiberAverage(**({"cutoff_invang": 8.0} | kwargs))


def test_fiber_numerics_sit_beside_mosaic_controls():
    average = FiberAverage(cutoff_invang=8.0, azimuth_count=16)
    controls = CxrReflections(mosaic_nodes=7, mosaic_route="mc", fiber=average)
    assert CxrReflections().fiber is None
    with pytest.raises(TypeError):
        CxrReflections(fiber={"cutoff_invang": 8.0})
    site = thermal_site("hopg", "C")
    for source in (average, controls, Numerics(cxr_reflections=controls)):
        model = fiber.FiberCXRModel.from_numerics(source)
        assert model.numerics is average
        assert model.temperature_K == site.temperature_K
        assert (model.cutoff_invang, model.azimuth_count) == (8.0, 16)
        assert model.reflection_batch_size == 256
    with pytest.raises(ValueError, match="basal-only"):
        fiber.FiberCXRModel.from_numerics(CxrReflections())
    with pytest.raises(ValueError):
        fiber.FiberCXRModel.from_numerics(FiberAverage(cutoff_invang=80.0))


def _real_kernel_scene():
    """Small frozen segment set; deterministic, CPU, seconds to score."""
    rng = np.random.default_rng(7)
    n = 24
    v = rng.normal(size=(n, 3)) * [0.3, 0.3, 1.0] + [0, 0, 2.0]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r_mid = np.column_stack(
        [rng.uniform(-2e4, 2e4, n), rng.uniform(-2e4, 2e4, n), rng.uniform(50, 3950, n)]
    )
    segments = dict(
        r_mid=r_mid,
        v_hat=v,
        L_ang=rng.uniform(20, 80, n),
        E_keV=rng.uniform(25, 30, n),
        t_ang=np.zeros(n),
        t0_ang=np.zeros(n),
        elec_id=np.repeat(np.arange(4), n // 4),
        layer=np.zeros(n, dtype=np.int16),
        Ne=4,
        thickness_ang=4000.0,
        n_backscattered=0,
        n_missed=0,
    )
    options = dict(surface_hkl=(0, 0, 1), theta_obs_rad=np.deg2rad(60.0))
    return segments, np.linspace(1000.0, 12000.0, 600), options


def _rolled(domain, segments, grid, options, phi, hkls):
    return np.asarray(
        fiber.mc_cxr_spectrum(
            segments,
            grid,
            "hopg",
            list(hkls),
            B_ang2=domain._thermal_lookup().__getitem__,
            azimuth_rad=phi,
            line_quadrature="bin-mean",
            **options,
        )
    )


def test_real_kernel_is_sixty_degree_periodic():
    """Validation: cxr-fiber-intensity-average (sector-closure premise, real kernel)."""
    domain = _model().prepare()
    segments, grid, options = _real_kernel_scene()
    other = [h for h in domain.hkls if h[:2] != (0, 0)]
    phi = 0.37
    base = _rolled(domain, segments, grid, options, phi, other)
    scale = np.abs(base).sum()
    assert scale > 0
    for shift in (np.pi / 3, 2 * np.pi / 3):
        rolled = _rolled(domain, segments, grid, options, phi + shift, other)
        assert np.abs(rolled - base).sum() / scale < 1e-12
    half = _rolled(domain, segments, grid, options, phi + np.pi / 6, other)
    assert np.abs(half - base).sum() / scale > 1e-2


def test_sector_rule_matches_independent_full_circle_mean():
    """Validation: cxr-fiber-intensity-average (sector rule vs full-circle average).

    97 full-circle nodes (coprime with 6, so no aliasing onto the sixfold
    period) give an orientation mean that never assumes the 60-degree sector.
    """
    segments, grid, options = _real_kernel_scene()
    domain = _model(azimuth_count=32).prepare()
    basal = [h for h in domain.hkls if h[:2] == (0, 0)]
    other = [h for h in domain.hkls if h[:2] != (0, 0)]
    nodes = np.linspace(0.0, 2 * np.pi, 97, endpoint=False) + 0.011
    reference = _rolled(domain, segments, grid, options, 0.0, basal) + np.mean(
        [_rolled(domain, segments, grid, options, x, other) for x in nodes], axis=0
    )
    got = np.asarray(domain.spectrum(segments, grid, **options))
    total = reference.sum()
    assert abs(got.sum() - total) / total < 1e-4
    assert np.abs(got - reference).sum() / total < 1e-4
