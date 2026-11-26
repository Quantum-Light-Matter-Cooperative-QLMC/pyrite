"""Measured reflection selection over the full fiber domain (#1484)."""

import numpy as np
import pytest

from pyrite.config import FiberAverage
from pyrite.materials.channeling.thermal import thermal_site
from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.montecarlo.runner.reflection_cutoff import geometric_tail_fraction
from pyrite.montecarlo.spectrum.cxr import fiber_selection
from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel
from pyrite.montecarlo.spectrum.cxr.fiber_case import score_fiber_case

_LATTICE = CRYSTALS["hopg"]["lattice"]
_GRID = np.array([100.0, 300.0, 500.0])
_WIDTHS = np.array([200.0, 200.0, 200.0])
_BACKGROUND = 50.0


def _model(cutoff=16.0, rolls=3):
    average = FiberAverage(cutoff_invang=cutoff, azimuth_count=rolls)
    return FiberCXRModel(average, thermal_site("hopg", "C").temperature_K)


def _norm(hkl):
    return float(reciprocal_g_vector(tuple(hkl), _LATTICE)[1])


def _line(hkl):
    """Per-reflection density: decays with |g|; roll-averaged it is exact (see scorer)."""
    return 1e-3 * np.exp(-0.9 * _norm(hkl))


def _share(hkl):
    return 0.2 * _line(hkl)


@pytest.mark.parametrize("cutoff", [4.0, 9.0])
def test_shells_partition_the_prepared_domain(cutoff):
    model = _model(cutoff)
    union = []
    for upper, members in fiber_selection.iter_fiber_shells(model):
        assert all(upper - 2.0 < _norm(h) <= upper + 1e-9 for h in members)
        model.domain(members)  # each shell alone is Friedel- and roll-closed
        union.extend(members)
        prepared = _model(upper).prepare().hkls
        assert tuple(sorted(union)) == prepared
    assert upper == cutoff


class _Continuum:
    calls: list = []

    def __init__(self, model, case, segments, n_hat, band_eV, *, groove, Ne, floor_keV, max_keV):
        assert band_eV == (0.0, 600.0)
        assert (floor_keV, max_keV) == (5.0, 30.0)
        _Continuum.calls.append(Ne)

    def background(self):
        return _BACKGROUND

    def share(self, hkls):
        return float(sum(_share(h) for h in hkls) * _WIDTHS.sum())


@pytest.fixture
def scene(monkeypatch):
    from pyrite.montecarlo import runner

    seen = []

    def line(segs, axis, pilot, n_hat, abs_layers, groove, *, coherent, Ne, table_cache):
        assert coherent is False and table_cache is None
        np.testing.assert_array_equal(axis, _GRID)
        seen.append((Ne, len(pilot["hkl_list"]), pilot["azimuth_rad"]))
        # Roll-dependent per row, roll-average exactly _line over the sector.
        roll = sum(_line(h) for h in pilot["hkl_list"])
        wobble = 0.0 if all(h[:2] == (0, 0) for h in pilot["hkl_list"]) else 0.5
        return np.full(axis.shape, roll * (1 + wobble * np.cos(6 * pilot["azimuth_rad"])))

    monkeypatch.setattr(runner, "_cxr_for_segments", line)
    monkeypatch.setattr(runner, "_segments_on_device", lambda x: x)
    monkeypatch.setattr(fiber_selection, "_FiberShellContinuum", _Continuum)
    _Continuum.calls = []
    ids = np.repeat(np.arange(8), 3)
    segs = {"Ne": 8, "elec_id": ids, "L_ang": np.ones(ids.size)}
    tp = dict(segs=segs, Ne_lines=8, Ne_brem=8, n_hat=np.array([0.0, 0.0, 1.0]))
    pilot = dict(crystal="hopg", composition=[("C", 1.0)], azimuth_rad=0.0)
    return pilot, tp, seen


def _expected_stop(model, tol):
    """Independent stop: exact roll-averaged net shell yields and the geometric tail."""
    yields, kept = [], []
    decaying = 0
    for upper, members in fiber_selection.iter_fiber_shells(model):
        net = sum(_line(h) - _share(h) for h in members) * _WIDTHS.sum()
        decaying = decaying + 1 if yields and abs(net) < abs(yields[-1]) else 0
        yields.append(net)
        kept.extend(members)
        tail = geometric_tail_fraction(yields, _BACKGROUND)
        if decaying >= 2 and tail <= tol:
            return upper, sorted(kept), tail
    raise AssertionError("synthetic sequence must converge below the cap")


def test_selection_stops_on_roll_averaged_net_rows(scene):
    pilot, tp, seen = scene
    model = _model()
    domain, cxr, record = fiber_selection.select_fiber_reflections(
        model, pilot, tp, _GRID, 1e-3, floor_keV=5.0, max_keV=30.0
    )
    upper, kept, tail = _expected_stop(model, 1e-3)
    assert record["converged"] and record["cutoff_invang"] == upper
    assert record["tail_fraction"] == pytest.approx(tail, rel=1e-9)
    assert list(domain.hkls) == kept == list(_model(upper).prepare().hkls)
    assert domain.record()["cutoff_invang"] == upper
    assert record["metric"] == "net" and record["background_yield"] == _BACKGROUND
    assert record["domain"] == "full-fiber" and record["azimuth_count"] == 3
    assert record["g_max_invang"] == 16.0 and "selection_histories" not in record
    # Whole population selected: the shell sum is the production spectrum.
    np.testing.assert_allclose(cxr, sum(_line(h) for h in kept), rtol=1e-12)
    assert {ne for ne, *_ in seen} == {8} and _Continuum.calls == [8]
    rolls = {round(phi, 12) for *_, phi in seen}
    assert rolls == {round(np.pi / 3 * (i + 0.5) / 3, 12) for i in range(3)} | {0.0}


def test_set_identity_carries_domain_and_roll_count(scene):
    pilot, tp, _ = scene
    records = [
        fiber_selection.select_fiber_reflections(
            _model(rolls=rolls), pilot, tp, _GRID, 1e-3, floor_keV=5.0, max_keV=30.0
        )[2]
        for rolls in (3, 4)
    ]
    assert records[0]["reflections"] == records[1]["reflections"]
    assert records[0]["selected_set_sha256"] != records[1]["selected_set_sha256"]


def test_unconverged_cap_keeps_every_shell(scene):
    pilot, tp, _ = scene
    domain, _, record = fiber_selection.select_fiber_reflections(
        _model(cutoff=6.0), pilot, tp, _GRID, 1e-6, floor_keV=5.0, max_keV=30.0
    )
    assert not record["converged"] and record["cutoff_invang"] == 6.0
    assert domain.hkls == _model(6.0).prepare().hkls


def test_prefix_selects_then_case_rescores_every_history(scene, monkeypatch):
    from pyrite.montecarlo.runner import case_tables, emission
    from pyrite.montecarlo.spectrum.cxr import fiber

    pilot, tp, seen = scene
    domain, cxr, record = fiber_selection.select_fiber_reflections(
        _model(), pilot, tp, _GRID, 1e-3, floor_keV=5.0, max_keV=30.0, histories=3
    )
    assert cxr is None and _Continuum.calls == [3]
    assert record["selection_histories"]["lines"] == 3
    assert record["selection_histories"]["rows"] == 9
    assert {ne for ne, *_ in seen} == {3}

    monkeypatch.setattr(case_tables, "_case_bremslib_tables", lambda _: {"C": object()})
    monkeypatch.setattr(
        fiber.FiberReflectionSet,
        "corrected_bremslib_tables",
        lambda self, tables, composition, **kw: (tables, {"shells": len(self.hkls)}),
    )
    monkeypatch.setattr(
        emission, "_layer_brem_from_segments", lambda *a, **kw: np.ones(_GRID.shape)
    )
    seen.clear()
    case = dict(
        pilot, E0_keV=30.0, bremsstrahlung_model="bremslib", hkl_list=[(0, 0, 2)], B_ang2=1.3
    )
    result = score_fiber_case(
        _model(), case, tp, _GRID, line_yield_tolerance=1e-3, selection_histories=3
    )
    selection = result["reflection_selection"]
    assert selection["cutoff_invang"] == record["cutoff_invang"]
    assert result["model"]["cutoff_invang"] == record["cutoff_invang"]
    assert result["model"]["hkls"] == domain.hkls
    # Prefix rows (Ne=3) select; the selected set is then scored on all 8.
    assert {ne for ne, *_ in seen} == {3, 8}
    np.testing.assert_allclose(result["cxr"], sum(_line(h) for h in domain.hkls), rtol=1e-12)
